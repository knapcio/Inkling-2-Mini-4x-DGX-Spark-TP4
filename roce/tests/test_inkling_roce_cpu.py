"""CPU tests for the Inkling-2-Mini RoCEnante adapter (no GPU, no RDMA, no SGLang needed).

Ported from our MiniMax-M3.1 recipe's ``test_mm31_roce_cpu.py``; sizes use Inkling-2-Mini
(hidden 2304, padded vocab 201,024, FP32 logits gathered as bf16 pairs).

The b12x runtime is replaced by a fake whose collectives run over gloo, and SGLang's
``GroupCoordinator`` by a stand-in with the image's method shapes, patched by the real hook
module (``tp4/inkling_sgl_roce.py``), so these tests pin:
- limit parsing and the capability vote across 4 real (gloo) ranks, including the
  fail-together paths (a rank that cannot take part, limits or start-up policy that differ);
- routing: eligible all-reduces (bf16 attention, fp32 MoE combine) and all-gathers (logits)
  go to RoCE, everything else (too big, wrong dtype, non-TP groups) to the stock path,
  identically on every rank; in-place results land in the caller's tensor;
- the prepared-plan API (one plan + session per runtime) and the legacy API;
- the post-import patcher (patch once, loud failure).
The vLLM hooks of the MiniMax package are not part of this recipe.

Run: ``python3 roce/tests/test_inkling_roce_cpu.py`` (no pytest needed) or with pytest.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
TP4 = ROOT.parent / "tp4"
for p in (str(ROOT), str(TP4)):
    if p not in sys.path:
        sys.path.insert(0, p)

from inkling_roce import adapter as adapter_mod  # noqa: E402
import inkling_sgl_roce as hook  # noqa: E402

WORLD = 4
HIDDEN = 2304  # Inkling-2-Mini: 4.5 KiB per bf16 row, 9 KiB per fp32 row
SHARD = 201024 // WORLD  # padded vocab per rank


# -- fakes ---------------------------------------------------------------------------


class FakeRuntime:
    """Same surface as ``b12x.comm.roce.AllReduce``; collectives over the gloo group."""

    instances: list = []

    def __init__(self, exchange_group, device, max_size, max_gather_bytes):
        self.group = exchange_group
        self.max_size = max_size
        self.max_gather_bytes = max_gather_bytes
        self.hca_names = ("rocep1s0f0", "roceP2p1s0f0")
        self.prepared: list = []
        self.captures = 0
        self.closed = False
        self.poisoned = False
        self.calls: list = []
        FakeRuntime.instances.append(self)

    @classmethod
    def from_exchange_group(cls, *, exchange_group, device, max_size, max_gather_bytes):
        return cls(exchange_group, device, max_size, max_gather_bytes)

    def prepare(self, dtypes=(torch.bfloat16,), *, padded_gather=False):
        self.prepared.append(tuple(dtypes))

    def should_allreduce(self, inp):
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_size and nbytes % 16 == 0

    def all_reduce(self, inp, *, out=None, stream=None):
        self.calls.append(("all_reduce", inp.numel() * inp.element_size()))
        acc = inp.float().clone()
        dist.all_reduce(acc, group=self.group)
        if os.environ.get("FAKE_ROCE_CORRUPT") == "1":
            acc += 1  # a broken transport: wrong sums on this rank only
        if out is None:
            return acc.to(inp.dtype)
        out.copy_(acc.to(inp.dtype))
        return out

    def should_all_gather(self, inp, dim=-1):
        if dim < 0:
            dim += inp.dim()
        if dim not in (0, inp.dim() - 1) or not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_gather_bytes

    def all_gather(self, inp, *, dim=-1, out=None, stream=None):
        self.calls.append(("all_gather", tuple(inp.shape), dim))
        parts = [torch.empty_like(inp) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(parts, inp.contiguous(), group=self.group)
        res = torch.cat(parts, dim=dim)
        if out is None:
            return res
        out.copy_(res)
        return out

    @contextmanager
    def capture(self, stream=None):
        self.captures += 1
        yield self

    def check_health(self):
        if self.poisoned:
            raise RuntimeError("RoCE collective timed out (fake)")

    def stats(self):
        return {"spin_limit": 300000000, "max_size": self.max_size}

    def close(self):
        self.closed = True


class FakePlannedRuntime(FakeRuntime):
    """The prepared-plan surface of the image's b12x (FlashInfer 0.7.2 / 0.7.4): no ``prepare()``,
    and every collective needs a prepared ``comm.roce`` plan."""

    prepare = None  # the image runtime has no prepare(); calling it would be a bug

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self._threads, self.rail_count = 512, 2

    def _require(self, plan):
        assert plan is not None and plan.runtime is self and plan.prepared, "unprepared plan"

    def all_reduce(self, inp, *, out=None, stream=None, plan):
        self._require(plan)
        return FakeRuntime.all_reduce(self, inp, out=out)

    def all_gather(self, inp, *, dim=-1, out=None, stream=None, plan):
        self._require(plan)
        return FakeRuntime.all_gather(self, inp, dim=dim, out=out)


class FakePlan:
    instances: list = []

    def __init__(self, query, runtime):
        self.query, self.runtime, self.prepared = query, runtime, False
        FakePlan.instances.append(self)

    def request(self, *, name, prepare_call):
        return types.SimpleNamespace(name=name, plan=self, prepare_call=prepare_call)


class FakeSession:
    instances: list = []

    def __init__(self, *, device=None, autotune=True, compile_workers=None):
        self.device, self.autotune, self.compile_workers = device, autotune, compile_workers
        self.prepared_requests: list = []
        self.closed = False
        FakeSession.instances.append(self)

    def prepare(self, requests, **kw):
        for req in requests:
            # materialize: the dtypes in the query are what the runtime compiles
            req.plan.runtime.prepared.append(tuple(getattr(torch, d) for d in req.plan.query["call"]["dtypes"]))
            req.plan.prepared = True
            call = req.prepare_call(object())
            call.run()
            self.prepared_requests.append(req.name)

    def close(self):
        self.closed = True


def fake_roce_module(*, api=1, supported=True, plan_api=False):
    mod = types.SimpleNamespace()
    mod.API_VERSION = api
    mod.is_supported = lambda device=None: supported
    if plan_api:
        mod.AllReduce = FakePlannedRuntime
        mod.query_from_runtime = lambda rt, **kw: dict(kw, world_size=rt.world_size, rank=rt.rank)
        mod.plan = lambda query, *, runtime: FakePlan(query, runtime)
    else:
        mod.AllReduce = FakeRuntime
    return mod


def install_fake_roce(monkey_modules: dict, **kwargs):
    """Make ``from b12x.comm import roce`` (and ``b12x.preparation``) return the fakes."""
    roce = fake_roce_module(**kwargs)
    comm = types.ModuleType("b12x.comm")
    comm.roce = roce
    prep = types.ModuleType("b12x.preparation")
    prep.PreparationSession = FakeSession
    prep.PreparedCall = lambda run, output=None: types.SimpleNamespace(run=run, output=output)
    pkg = types.ModuleType("b12x")
    pkg.comm = comm
    pkg.preparation = prep
    for name, mod in (("b12x", pkg), ("b12x.comm", comm), ("b12x.comm.roce", roce),
                      ("b12x.preparation", prep)):
        monkey_modules[name] = sys.modules.get(name)
        sys.modules[name] = mod
    return roce


def restore_modules(saved: dict):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def make_fake_sglang_parallel_state():
    """A module with a ``GroupCoordinator`` shaped like the Inkling image's SGLang (03fde9cc):
    ``all_reduce`` -> ``_all_reduce_in_place`` (custom op, NCCL with custom AR disabled),
    ``all_gather(dim)`` -> dim-0 ``_all_gather_into_tensor`` into a fresh buffer + reshape."""
    mod = types.ModuleType("fake_parallel_state")

    class Ctx:
        def __init__(self, stream):
            self.stream = stream

    class GroupCoordinator:
        def __init__(self, cpu_group, unique_name):
            self.cpu_group = cpu_group
            self.device_group = object()
            self.device = types.SimpleNamespace(type="cuda")  # the adapter itself gets a CPU device
            self.unique_name = unique_name
            self.world_size = dist.get_world_size(cpu_group)
            self.stock_calls: list = []
            self.destroyed = False

        def all_reduce(self, input_):
            self._all_reduce_in_place(input_)
            return input_

        def _all_reduce_in_place(self, input_):
            self.stock_calls.append(("all_reduce", str(input_.dtype), input_.numel()))
            acc = input_.float() if input_.is_floating_point() else input_.clone()
            dist.all_reduce(acc, group=self.cpu_group)
            input_.copy_(acc.to(input_.dtype))

        def _all_reduce_out_place(self, input_, outplace_all_reduce_method):
            out = input_.clone()
            self._all_reduce_in_place(out)
            return out

        def all_gather(self, input_, dim=-1):
            if dim < 0:
                dim += input_.dim()
            size = input_.size()
            output = torch.empty((size[0] * self.world_size,) + size[1:], dtype=input_.dtype)
            self._all_gather_into_tensor(output, input_)
            output = output.reshape((self.world_size,) + size).movedim(0, dim)
            return output.reshape(size[:dim] + (self.world_size * size[dim],) + size[dim + 1:])

        def _all_gather_into_tensor(self, output, input):
            self.stock_calls.append(("all_gather", tuple(input.shape)))
            parts = [torch.empty_like(input) for _ in range(self.world_size)]
            dist.all_gather(parts, input.contiguous(), group=self.cpu_group)
            output.copy_(torch.cat(parts, 0))

        @contextmanager
        def graph_capture(self, graph_capture_context=None, stream=None):
            yield graph_capture_context or Ctx(stream)

        def destroy(self):
            self.destroyed = True

    mod.GroupCoordinator = GroupCoordinator
    return mod


def _cpu_adapter(**kw):
    """The real adapter, on a CPU device (the coordinator above poses as CUDA for the hook)."""
    return adapter_mod.InklingRoceAllReduce(group=kw["group"], device_group=kw["device_group"],
                                            device=torch.device("cpu"))


@contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -- 1. limits -------------------------------------------------------------------------


def test_parse_byte_size():
    p = adapter_mod._parse_byte_size
    assert p("4096") == 4096
    assert p("2MB") == p("2MiB") == p("2m") == 2 << 20
    assert p("84KB") == 84 << 10
    assert p(" 16 mib ") == 16 << 20
    for bad in ("", "2XB", "abc", "1.5MB"):
        try:
            p(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_read_limits_defaults_and_validation():
    assert adapter_mod.read_limits({}) == (2 << 20, 16 << 20)
    assert adapter_mod.read_limits({"INKLING_TP4_ROCE_MAX_SIZE": "1MiB", "INKLING_TP4_ROCE_GATHER_MAX_SIZE": "0"}) == (1 << 20, 0)
    for bad in ({"INKLING_TP4_ROCE_MAX_SIZE": "0"}, {"INKLING_TP4_ROCE_MAX_SIZE": "100"}, {"INKLING_TP4_ROCE_GATHER_MAX_SIZE": "-16"},
                {"INKLING_TP4_ROCE_GATHER_MAX_SIZE": "10"}):
        try:
            adapter_mod.read_limits(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert adapter_mod.require_enabled({}) is True
    assert adapter_mod.require_enabled({"INKLING_TP4_ROCE_REQUIRE": "0"}) is False


# -- 2. four gloo ranks: vote + routing ---------------------------------------------------


def _rank_main(rank, scenario, tmpdir):
    """One rank of a 4-process gloo job; writes its verdict to ``<tmpdir>/rank<r>.json``."""
    result: dict = {"rank": rank}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{tmpdir}/store", rank=rank, world_size=WORLD,
            timeout=timedelta(seconds=60),
        )
        group = dist.group.WORLD
        saved: dict = {}
        supported = not (
            (scenario.startswith("rank2_unsupported") or scenario.startswith("unsupported_and")) and rank == 2
        )
        install_fake_roce(saved, supported=supported, plan_api=scenario.startswith("plan"))
        if scenario == "selftest_bad" and rank == 3:
            os.environ["FAKE_ROCE_CORRUPT"] = "1"
        if scenario == "limits_differ" and rank == 1:
            os.environ["INKLING_TP4_ROCE_MAX_SIZE"] = "1MiB"
        if scenario == "selftest_differs" and rank == 2:
            os.environ["INKLING_TP4_ROCE_SELFTEST"] = " 0 "  # normalized: same as "0"
        if scenario == "require_differs" and rank == 3:
            os.environ["INKLING_TP4_ROCE_REQUIRE"] = "0"
        if scenario == "unsupported_and_require_differs":
            if rank == 1:
                os.environ["INKLING_TP4_ROCE_REQUIRE"] = "0"
        if scenario.endswith("_lenient"):
            os.environ["INKLING_TP4_ROCE_REQUIRE"] = "0"
        single_node = scenario == "single_node"
        adapter_mod._in_the_same_node_as = lambda g, source_rank=0: [single_node or r == 0 for r in range(WORLD)]

        mod = make_fake_sglang_parallel_state()
        hook._LIVE.clear()
        hook.patch_parallel_state(mod, adapter_factory=_cpu_adapter)
        try:
            tp = mod.GroupCoordinator(group, "tp:0")
        except RuntimeError as exc:
            result["raised"] = str(exc)
            return
        comm = tp.inkling_roce_comm
        result["enabled"] = comm is not None and not comm.disabled
        result["verdict"] = getattr(comm, "verdict", None)
        result["live"] = len(hook._LIVE)
        if scenario.endswith("_differs"):
            result["survived"] = True  # must not happen: a policy mismatch raises on every rank
            return
        world = mod.GroupCoordinator(group, "world:0")
        result["world_has_comm"] = world.inkling_roce_comm is not None

        torch.manual_seed(1234 + rank)
        # (rows, dtype): attention bf16 c1 x 8 .. c16 x 8, the 2 MiB edge (455 bf16 rows fit, 456 do
        # not), the fp32 MoE combine (128 rows fit, 228 rows = 2 MiB + 4 KiB do not)
        routed = []
        for rows, dtype in ((1, torch.bfloat16), (8, torch.bfloat16), (128, torch.bfloat16),
                            (455, torch.bfloat16), (456, torch.bfloat16),
                            (8, torch.float32), (128, torch.float32), (227, torch.float32),
                            (228, torch.float32)):
            x = torch.randn(rows, HIDDEN).to(dtype)
            ref = x.float().clone()
            dist.all_reduce(ref, group=group)
            before = len(tp.stock_calls)
            y = tp.all_reduce(x)
            assert y is x, "in-place contract broken"
            assert torch.allclose(y.float(), ref.to(dtype).float(), atol=0.25), (rows, dtype)
            routed.append((rows, str(dtype).replace("torch.", ""), len(tp.stock_calls) == before))
        result["routed"] = routed
        # int32 never goes to RoCE
        before = len(tp.stock_calls)
        tp.all_reduce(torch.ones(8, dtype=torch.int32))
        result["int32_stock"] = len(tp.stock_calls) == before + 1
        # logits: FP32 shard [8, SHARD] gathered along the last dim as bf16 pairs
        shard = torch.full((8, SHARD), float(rank) + 0.5, dtype=torch.float32)
        before = len(tp.stock_calls)
        g = tp.all_gather(shard.view(torch.bfloat16), dim=-1).view(torch.float32)
        result["gather_routed"] = len(tp.stock_calls) == before
        result["gather_shape"] = list(g.shape)
        result["gather_ok"] = all(
            bool((g[:, r * SHARD:(r + 1) * SHARD] == r + 0.5).all()) for r in range(WORLD)
        )
        g0 = tp.all_gather(torch.full((2, 8), float(rank)), dim=0)
        result["gather0_ok"] = list(g0[:, 0].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]
        big = torch.zeros(128, 2 * SHARD, dtype=torch.bfloat16)  # c16 x 8 rows: 25.7 MB > 16 MiB
        before = len(tp.stock_calls)
        tp.all_gather(big, dim=-1)
        result["big_gather_stock"] = len(tp.stock_calls) == before + 1
        runtime = FakeRuntime.instances[-1] if FakeRuntime.instances else None
        result["prepared"] = [list(map(str, d)) for d in runtime.prepared] if runtime else []
        result["api"] = getattr(comm._runtime, "api", None) if comm else None
        result["selftest_calls"] = sum(1 for c in runtime.calls if c[0] == "all_reduce" and c[1] == 64 * 2) \
            if runtime else 0
        # graph_capture re-entry must not prepare (or plan) a second time
        with tp.graph_capture(stream="cap"):
            pass
        result["prepared_after_capture"] = len(runtime.prepared) if runtime else 0
        result["plans"] = len(FakePlan.instances)
        session = FakeSession.instances[-1] if FakeSession.instances else None
        result["session"] = None if session is None else {
            "autotune": session.autotune, "compile_workers": session.compile_workers,
            "requests": session.prepared_requests}
        result["limits"] = [comm.max_size, comm.max_gather] if comm else None
        tp.destroy()
        result["closed"] = runtime.closed if runtime else None
        result["session_closed"] = session.closed if session else None
        result["destroyed"] = tp.destroyed
        result["live_after_destroy"] = len(hook._LIVE)
        restore_modules(saved)
    except Exception as exc:  # noqa: BLE001
        import traceback

        result["error"] = f"{exc!r}\n{traceback.format_exc()}"
    finally:
        Path(tmpdir, f"rank{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_world(scenario, **env_values):
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmpdir, env(**env_values):
        mp.spawn(_rank_main, args=(scenario, tmpdir), nprocs=WORLD, join=True)
        results = [json.loads(Path(tmpdir, f"rank{r}.json").read_text()) for r in range(WORLD)]
    for r in results:
        assert "error" not in r, r["error"]
    return results


def test_four_ranks_route_decode_sizes():
    results = _run_world("ok", INKLING_TP4_ROCE_REQUIRE=None, INKLING_TP4_ROCE_MAX_SIZE=None, INKLING_TP4_ROCE_GATHER_MAX_SIZE=None)
    for r in results:
        assert r["enabled"], r
        assert r["world_has_comm"] is False  # only the TP group is routed
        # <= 2 MiB on RoCE (bf16 and fp32), larger on the stock path
        assert json.loads(json.dumps(r["routed"])) == [[1, "bfloat16", True], [8, "bfloat16", True], [128, "bfloat16", True], [455, "bfloat16", True], [456, "bfloat16", False], [8, "float32", True], [128, "float32", True], [227, "float32", True], [228, "float32", False]], r
        assert r["int32_stock"] and r["big_gather_stock"] and r["gather_routed"]
        assert r["gather_shape"] == [8, WORLD * SHARD] and r["gather_ok"] and r["gather0_ok"]
        assert r["live"] == 1 and r["live_after_destroy"] == 0
        assert r["prepared"] and r["prepared"][0] == ["torch.bfloat16", "torch.float16", "torch.float32"]
        assert r["limits"] == [2 << 20, 16 << 20]
        assert r["closed"] and r["destroyed"]
    # rank-invariant: every rank took the same routing decisions
    assert len({json.dumps(r["routed"]) for r in results}) == 1


def test_four_ranks_one_rank_cannot_all_raise():
    results = _run_world("rank2_unsupported", INKLING_TP4_ROCE_REQUIRE=None)
    for r in results:
        assert "raised" in r and "rank 2" in r["raised"] and "INKLING_TP4_ROCE_REQUIRE=0" in r["raised"], r


def test_four_ranks_one_rank_cannot_lenient_all_nccl():
    results = _run_world("rank2_unsupported_lenient")
    for r in results:
        assert r["enabled"] is False and "rank 2" in (r["verdict"] or ""), r
        assert all(roce is False for _, _, roce in r["routed"]), r  # everything on the stock path


def test_four_ranks_limits_differ_all_raise():
    results = _run_world("limits_differ", INKLING_TP4_ROCE_REQUIRE=None)
    for r in results:
        assert "raised" in r and "differ" in r["raised"], r


def test_four_ranks_startup_policy_differs_nobody_routes():
    # One rank skipping the self-test (or not requiring RoCE) must not leave the others waiting
    # in the probe: the policy is voted, so no rank builds a runtime or issues a probe.
    for scenario in ("selftest_differs", "require_differs", "unsupported_and_require_differs"):
        results = _run_world(scenario, INKLING_TP4_ROCE_REQUIRE=None, INKLING_TP4_ROCE_SELFTEST=None)
        for r in results:
            text = r.get("raised") or r.get("verdict") or ""
            assert "start-up policy" in text and "differs" in text, (scenario, r)
            assert not r.get("enabled"), (scenario, r)
        # a policy mismatch is fatal on every rank, including one with REQUIRE=0
        assert all("raised" in r for r in results), (scenario, results)


def test_startup_policy_normalization():
    assert adapter_mod.startup_policy({}) == {"require": True, "selftest": True}
    assert adapter_mod.startup_policy({"INKLING_TP4_ROCE_SELFTEST": " 0 ", "INKLING_TP4_ROCE_REQUIRE": "0"}) == {
        "require": False, "selftest": False}
    assert adapter_mod.startup_policy({"INKLING_TP4_ROCE_SELFTEST": "yes"})["selftest"] is True


def test_four_ranks_plan_api_routes_and_prepares_once():
    """The image's b12x (prepared plans): same routing, one plan + session per runtime, no
    collective inside preparation, the self-test runs on the planned runtime, close() closes both."""
    results = _run_world("plan_ok", INKLING_TP4_ROCE_REQUIRE=None, INKLING_TP4_ROCE_MAX_SIZE=None,
                         INKLING_TP4_ROCE_GATHER_MAX_SIZE=None)
    for r in results:
        assert r["enabled"] and r["api"] == "plan", r
        assert json.loads(json.dumps(r["routed"])) == [[1, "bfloat16", True], [8, "bfloat16", True], [128, "bfloat16", True], [455, "bfloat16", True], [456, "bfloat16", False], [8, "float32", True], [128, "float32", True], [227, "float32", True], [228, "float32", False]], r
        assert r["gather_shape"] == [8, WORLD * SHARD] and r["gather_ok"] and r["gather0_ok"]
        assert r["prepared"] == [["torch.bfloat16", "torch.float16", "torch.float32"]], r
        assert r["prepared_after_capture"] == 1 and r["plans"] == 1, r
        assert r["session"] == {"autotune": False, "compile_workers": 0, "requests": ["inkling_roce"]}, r
        assert r["selftest_calls"] == 1, r  # the bf16 64-element probe (fp32 probe is 256 B)
        assert r["closed"] and r["session_closed"] and r["destroyed"]
    assert len({json.dumps(r["routed"]) for r in results}) == 1


def test_four_ranks_legacy_api_selected_without_plan():
    results = _run_world("ok", INKLING_TP4_ROCE_REQUIRE=None)
    for r in results:
        assert r["api"] == "legacy" and r["plans"] == 0 and r["session"] is None, r
        assert r["prepared_after_capture"] == 2, r  # legacy runtime: warm re-prepare (a no-op there)


def test_four_ranks_self_test_failure_all_raise():
    results = _run_world("selftest_bad", INKLING_TP4_ROCE_REQUIRE=None)
    for r in results:
        assert "raised" in r and "self-test failed" in r["raised"] and "rank 3" in r["raised"], r


def test_uses_plan_api_detection():
    assert adapter_mod.uses_plan_api(fake_roce_module(plan_api=True))
    assert not adapter_mod.uses_plan_api(fake_roce_module())
    assert adapter_mod.self_test_enabled({}) and not adapter_mod.self_test_enabled({"INKLING_TP4_ROCE_SELFTEST": "0"})


def test_four_ranks_single_node_skips_without_error():
    results = _run_world("single_node", INKLING_TP4_ROCE_REQUIRE=None)
    for r in results:
        assert r["enabled"] is False and "raised" not in r, r


# -- 3. the post-import patcher -----------------------------------------------------------


def test_boot_hook_patches_on_import_once(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    pkg = tmp / "inklingroce_fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "target.py").write_text("VALUE = 1\n")
    (pkg / "early.py").write_text("VALUE = 1\n")
    sys.path.insert(0, str(tmp))
    from inkling_roce.boot import PostImportPatcher, install_hooks

    calls = []
    import inklingroce_fakepkg.early  # noqa: F401  (imported before the hook exists)

    def patch(module):
        calls.append(module.__name__)
        module.VALUE = 2

    finder = install_hooks({"inklingroce_fakepkg.target": patch, "inklingroce_fakepkg.early": patch})
    try:
        assert isinstance(finder, PostImportPatcher)
        assert calls == ["inklingroce_fakepkg.early"]
        import inklingroce_fakepkg.target as t

        assert t.VALUE == 2 and calls == ["inklingroce_fakepkg.early", "inklingroce_fakepkg.target"]
        importlib.reload(t)  # a reload re-executes the module but does not patch again
        assert finder.pending() == []
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(tmp))
        for name in ("inklingroce_fakepkg", "inklingroce_fakepkg.target", "inklingroce_fakepkg.early"):
            sys.modules.pop(name, None)


def test_boot_hook_patch_failure_is_loud(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    pkg = tmp / "inklingroce_fakepkg2"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "target.py").write_text("VALUE = 1\n")
    sys.path.insert(0, str(tmp))
    from inkling_roce.boot import install_hooks

    def patch(module):
        raise AttributeError("CudaCommunicator")

    finder = install_hooks({"inklingroce_fakepkg2.target": patch})
    try:
        try:
            import inklingroce_fakepkg2.target  # noqa: F401
        except RuntimeError as exc:
            assert "patching inklingroce_fakepkg2.target failed" in str(exc)
        else:
            raise AssertionError("patch failure swallowed")
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(tmp))
        for name in ("inklingroce_fakepkg2", "inklingroce_fakepkg2.target"):
            sys.modules.pop(name, None)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                import traceback

                print(f"FAIL {name}: {exc!r}")
                traceback.print_exc()
    print("ALL PASS" if not failures else f"{failures} FAILED")
    sys.exit(1 if failures else 0)
