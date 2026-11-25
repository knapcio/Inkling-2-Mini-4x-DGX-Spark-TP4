"""CPU tests of the INKLING_TP4_ROCE hooks (tp4/inkling_sgl_roce.py). Needs torch, not SGLang or a GPU.

Fake GroupCoordinator / SchedulerBatchResultProcessor classes carry the image's method shapes; a fake
adapter records what reaches RoCE. The shared adapter itself (vote, self-test, plan API) is covered by
roce/tests/test_inkling_roce_cpu.py. With INKLING_SGL_IMAGE_SRC=<the image's sglang package dir, the
one holding srt/> the hook targets and the Inkling routing facts are also checked statically (AST and
text) against the image's own files.

  python3 tp4/tests/test_sgl_roce_cpu.py
  INKLING_SGL_IMAGE_SRC=/path/to/site-packages/sglang python3 tp4/tests/test_sgl_roce_cpu.py
In the image, with tp4/ and roce/ mounted apart, set INKLING_TP4_ROCE_PATH=/opt/inkling-roce.
"""
import ast
import os
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

import torch

TP4 = Path(__file__).resolve().parents[1]
ROOT = TP4.parent
ROCE = Path(os.environ.get("INKLING_TP4_ROCE_PATH") or ROOT / "roce")
sys.path[:0] = [str(TP4), str(ROCE)]
import inkling_sgl_roce as hook  # noqa: E402

HIDDEN = 2304          # Inkling-2-Mini
SHARD = 201024 // 4    # padded vocab per rank at TP4
MAX = 2 << 20          # default all-reduce limit


class FakeAdapter:
    def __init__(self, group, device_group, device, max_bytes=MAX, disabled=False):
        self.disabled, self.max_bytes, self.calls, self.events, self.health = disabled, max_bytes, [], [], 0

    def should_custom_ar(self, t):
        return t.dtype in (torch.bfloat16, torch.float32) and t.numel() * t.element_size() <= self.max_bytes

    def custom_all_reduce(self, t, out=None):
        self.calls.append(("ar", tuple(t.shape), t.dtype, out is t))
        res = t * 4
        if out is not None:
            out.copy_(res)
            return out
        return res

    def should_all_gather(self, t, dim):
        return dim == 0 and t.numel() * t.element_size() <= 16 << 20

    def all_gather(self, t, dim, out=None):
        self.calls.append(("ag", tuple(t.shape), out is not None))
        out.copy_(torch.cat([t] * 4, 0))
        return out

    @contextmanager
    def capture(self, stream=None):
        self.events.append(("capture_enter", stream))
        yield
        self.events.append(("capture_exit", stream))

    def check_health(self):
        self.health += 1

    def close(self):
        self.events.append("close")


def fake_parallel_state():
    m = types.ModuleType("fake_parallel_state")

    class Ctx:
        def __init__(self, stream):
            self.stream = stream

    class GroupCoordinator:
        def __init__(self, unique_name, world_size=4, device="cuda"):
            self.unique_name, self.world_size = unique_name, world_size
            self.device = types.SimpleNamespace(type=device)
            self.cpu_group, self.device_group, self.stock = object(), object(), []

        def _all_reduce_in_place(self, input_):
            self.stock.append("in_place")
            input_.mul_(4)

        def _all_reduce_out_place(self, input_, outplace_all_reduce_method):
            self.stock.append(("out_place", outplace_all_reduce_method))
            return input_ * 4

        def _all_gather_into_tensor(self, output, input):
            self.stock.append("gather")
            output.copy_(torch.cat([input] * 4, 0))

        @contextmanager
        def graph_capture(self, graph_capture_context=None, stream=None):
            self.stock.append("capture_enter")
            yield Ctx(stream or "new-stream")
            self.stock.append("capture_exit")

        def destroy(self):
            self.stock.append("destroy")

    m.GroupCoordinator = GroupCoordinator
    return m


def patched(**kw):
    m = fake_parallel_state()
    hook.patch_parallel_state(m, adapter_factory=lambda **a: FakeAdapter(**a, **kw))
    return m


def test_only_tp_group_gets_roce():
    hook._LIVE.clear()
    m = patched()
    assert m.GroupCoordinator("tp:0").inkling_roce_comm is not None
    for name in ("world:0", "pp:0", "attention_tp:0", "moe_ep:0", "moe_tp:0", "shared_experts_tp:0"):
        assert m.GroupCoordinator(name).inkling_roce_comm is None, name
    assert m.GroupCoordinator("tp:1", world_size=1).inkling_roce_comm is None
    assert m.GroupCoordinator("tp:2", device="cpu").inkling_roce_comm is None
    assert len(hook._LIVE) == 1


def test_routes_attention_moe_and_logits_sizes():
    m = patched()
    g = m.GroupCoordinator("tp:0")
    c = g.inkling_roce_comm
    attn = torch.ones(8, HIDDEN, dtype=torch.bfloat16)  # c1 x 8 verify rows, after wo_ud: 36 KiB
    g._all_reduce_in_place(attn)
    assert c.calls[-1] == ("ar", (8, HIDDEN), torch.bfloat16, True) and bool((attn == 4).all())
    moe = torch.ones(128, HIDDEN, dtype=torch.float32)  # c16 x 8 rows, FP32 Marlin combine: 1.13 MiB
    g._all_reduce_in_place(moe)
    assert c.calls[-1] == ("ar", (128, HIDDEN), torch.float32, True) and g.stock == []
    g._all_reduce_in_place(torch.ones(455, HIDDEN, dtype=torch.bfloat16))  # 2 MiB - 512 B: RoCE
    assert g.stock == []
    g._all_reduce_in_place(torch.ones(456, HIDDEN, dtype=torch.bfloat16))  # 2 MiB + 4 KiB -> NCCL
    g._all_reduce_in_place(torch.ones(228, HIDDEN, dtype=torch.float32))   # fp32 over 2 MiB -> NCCL
    g._all_reduce_in_place(torch.ones(8192, HIDDEN, dtype=torch.bfloat16))  # prefill chunk -> NCCL
    assert g.stock == ["in_place"] * 3
    x = torch.ones(8, HIDDEN, dtype=torch.bfloat16)
    assert bool((g._all_reduce_out_place(x, "auto") == 4).all()) and c.calls[-1][3] is False
    g._all_reduce_out_place(x, "pynccl")
    for method in ("ca", "pcie_ipc", "torch_symm_mem"):  # an explicit non-NCCL choice keeps its backend
        g._all_reduce_out_place(x, method)
        assert g.stock[-1] == ("out_place", method)
    ints = torch.ones(8, dtype=torch.int32)
    g._all_reduce_in_place(ints)
    assert g.stock[-1] == "in_place"
    # logits: FP32 shard [8, 50256] gathered as bf16 pairs [8, 100512] -> 1.6 MB per rank
    shard = torch.ones(8, SHARD, dtype=torch.float32).view(torch.bfloat16)
    out = torch.empty(32, 2 * SHARD, dtype=torch.bfloat16)
    g._all_gather_into_tensor(out, shard)
    assert c.calls[-1] == ("ag", (8, 2 * SHARD), True)
    assert bool((out.view(torch.float32) == 1).all())
    big = torch.ones(128, 2 * SHARD, dtype=torch.bfloat16)  # c16 x 8 rows: 25.7 MB -> NCCL
    g._all_gather_into_tensor(torch.empty(512, 2 * SHARD, dtype=torch.bfloat16), big)
    assert g.stock[-1] == "gather"


def test_disabled_adapter_is_stock():
    m = patched(disabled=True)
    g = m.GroupCoordinator("tp:0")
    x = torch.ones(4, 8, dtype=torch.bfloat16)
    g._all_reduce_in_place(x)
    g._all_gather_into_tensor(torch.empty(16, 8, dtype=torch.bfloat16), x)
    with g.graph_capture(stream="s"):
        pass
    assert g.stock == ["in_place", "gather", "capture_enter", "capture_exit"]


def test_graph_capture_uses_capture_stream_inside_stock_context():
    m = patched()
    g = m.GroupCoordinator("tp:0")
    with g.graph_capture(stream="cap") as ctx:
        assert ctx.stream == "cap"
        assert g.stock == ["capture_enter"] and g.inkling_roce_comm.events == [("capture_enter", "cap")]
    assert g.stock[-1] == "capture_exit" and g.inkling_roce_comm.events[-1] == ("capture_exit", "cap")
    with g.graph_capture():
        pass
    assert g.inkling_roce_comm.events[-1] == ("capture_exit", "new-stream")


def test_destroy_closes_runtime_first():
    hook._LIVE.clear()
    m = patched()
    g = m.GroupCoordinator("tp:0")
    c = g.inkling_roce_comm
    g.destroy()
    assert c.events == ["close"] and g.stock == ["destroy"] and hook._LIVE == []


def test_patch_is_idempotent():
    m = fake_parallel_state()
    built = []
    hook.patch_parallel_state(m, adapter_factory=lambda **a: built.append(1) or FakeAdapter(**a))
    hook.patch_parallel_state(m, adapter_factory=lambda **a: built.append(2) or FakeAdapter(**a))
    m.GroupCoordinator("tp:0")
    assert built == [1]


def test_result_processor_checks_before_consuming():
    order = []

    class Done:
        def synchronize(self):
            order.append("sync")

    class Proc:
        def process_batch_result_prefill(self, batch, result):
            order.append("orig")

        def process_batch_result_decode(self, batch, result):
            order.append("orig")

        def process_batch_result_idle(self, batch, result):
            order.append("orig")

    mod = types.SimpleNamespace(SchedulerBatchResultProcessor=Proc)
    comm = FakeAdapter(None, None, None)
    hook.patch_result_processor(mod, check=lambda r: hook.synchronize_and_check(r, live=[comm]))
    p = Proc()
    r = types.SimpleNamespace(copy_done=Done())
    p.process_batch_result_decode("b", r)
    p.process_batch_result_prefill("b", result=r)
    p.process_batch_result_idle(batch="b", result=r)
    assert order == ["sync", "orig"] * 3 and comm.health == 3
    hook.synchronize_and_check(types.SimpleNamespace(copy_done=None), live=[])  # no RoCE: no sync


def test_result_check_failure_stops_the_step():
    class Bad(FakeAdapter):
        def check_health(self):
            raise RuntimeError("RoCE collective on rank 1 timed out")

    done = types.SimpleNamespace(synchronize=lambda: None)
    try:
        hook.synchronize_and_check(types.SimpleNamespace(copy_done=done), live=[Bad(None, None, None)])
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("health failure swallowed")


def _run(env, code="pass", pythonpath=None):
    base = {k: v for k, v in os.environ.items() if not k.startswith(("INKLING_", "SGLANG_"))}
    base["PYTHONPATH"] = pythonpath or str(TP4)
    base.update(env)
    return subprocess.run([sys.executable, "-c", code], env=base, capture_output=True, text=True)


def test_sitecustomize_off_on_and_fail_closed():
    off = _run({}, "import sys; print('inkling_sgl_roce' in sys.modules)")
    assert off.returncode == 0 and off.stdout.split() == ["False"], (off.stdout, off.stderr)
    off0 = _run({"INKLING_TP4_ROCE": "0"}, "import sys; print('inkling_sgl_roce' in sys.modules)")
    assert off0.stdout.split() == ["False"], off0.stdout
    code = ("import sys; from inkling_roce.boot import PostImportPatcher;"
            "print(sum(isinstance(f, PostImportPatcher) for f in sys.meta_path), 'inkling_sgl_roce' in sys.modules)")
    on = _run({"INKLING_TP4_ROCE": "1", "INKLING_TP4_ROCE_PATH": str(ROCE)}, code)
    assert on.returncode == 0 and on.stdout.split() == ["1", "True"], (on.stdout, on.stderr)
    missing = _run({"INKLING_TP4_ROCE": "1", "INKLING_TP4_ROCE_PATH": "/nonexistent/inkling-roce"})
    assert missing.returncode == 78 and "is not mounted" in missing.stderr, missing.stderr
    conflict = _run({"INKLING_TP4_ROCE": "1", "INKLING_TP4_ROCE_PATH": str(ROCE),
                     "SGLANG_ENABLE_PCIE_IPC_ALLREDUCE": "1"})
    assert conflict.returncode == 78 and "SGLANG_ENABLE_PCIE_IPC_ALLREDUCE" in conflict.stderr


def test_sitecustomize_chains_to_the_next_one():
    with tempfile.TemporaryDirectory() as other:
        Path(other, "sitecustomize.py").write_text("import os\nos.environ['CHAINED_MARK'] = 'yes'\n")
        pp = os.pathsep.join([str(TP4), other])
        for env in ({}, {"INKLING_TP4_ROCE": "1", "INKLING_TP4_ROCE_PATH": str(ROCE)}):
            out = _run(env, "import os; print(os.environ.get('CHAINED_MARK'))", pythonpath=pp)
            assert out.returncode == 0 and out.stdout.split() == ["yes"], (env, out.stdout, out.stderr)


def test_route_facts_reported_when_missing():
    with tempfile.TemporaryDirectory() as d:
        problems = hook._verify_route_facts(d)
        assert len(problems) == len(hook.ROUTE_FACTS) and all("not readable" in p for p in problems)


def _method_params(tree, cls, meth):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == meth:
                    return [a.arg for a in f.args.args]
    return None


def test_image_sources_static():
    src = os.environ.get("INKLING_SGL_IMAGE_SRC")
    if not src:
        print("  (skipped: set INKLING_SGL_IMAGE_SRC)")
        return
    srt = Path(src) / "srt"
    ps_text = (srt / "distributed/parallel_state.py").read_text()
    ps = ast.parse(ps_text)
    want = {"_all_reduce_in_place": ["self", "input_"],
            "_all_reduce_out_place": ["self", "input_", "outplace_all_reduce_method"],
            "_all_gather_into_tensor": ["self", "output", "input"],
            "graph_capture": ["self", "graph_capture_context", "stream"], "destroy": ["self"]}
    for meth, params in want.items():
        assert _method_params(ps, "GroupCoordinator", meth) == params, meth
    assert "inplace_all_reduce(input_, group_name=self.unique_name)" in ps_text
    assert "self.all_gather_into_tensor(output_tensor, input_)" in ps_text
    brp_text = (srt / "managers/scheduler_components/batch_result_processor.py").read_text()
    brp = ast.parse(brp_text)
    for meth in hook.RESULT_METHODS:
        assert "result" in (_method_params(brp, "SchedulerBatchResultProcessor", meth) or []), meth
    assert brp_text.count("result.copy_done.synchronize()") >= 3
    assert hook._verify_route_facts(src) == [], hook._verify_route_facts(src)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(exc))
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
