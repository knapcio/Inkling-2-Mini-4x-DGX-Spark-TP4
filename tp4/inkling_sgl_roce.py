# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in (INKLING_TP4_ROCE=1): RoCEnante one-shot RDMA all-reduce/all-gather for the SGLang TP4 recipe.

Small tensor-parallel all-reduces and all-gathers of the ``tp`` group go to the image's
``b12x.comm.roce`` (FlashInfer 0.7.4 ``experimental/b12x``, prepared-plan API); NCCL keeps
everything above ``INKLING_TP4_ROCE_MAX_SIZE`` (2 MiB) / ``INKLING_TP4_ROCE_GATHER_MAX_SIZE``
(16 MiB per rank) and every other group.  The engine-agnostic part (vote, plan preparation,
self-test, size gate, route/health logging, fail-stop) is ``roce/inkling_roce/adapter.py``; this
file holds only the SGLang hook points.  ``roce/`` is put on ``sys.path`` from
``INKLING_TP4_ROCE_PATH`` (default ``/opt/inkling-roce``).

Port of our MiniMax-M3.1 TP4 recipe's ``sglang/mm31_sgl_roce.py``.  Hook points (sglang
``srt``), re-checked against ``lmsysorg/sglang:inkling-2-mini``
(sha256:efac853b..., SGLang nightly 03fde9cc); method signatures are unchanged from a6b0e4e:

1. ``GroupCoordinator.__init__``: for the group named ``tp`` (``tp:0``) build the adapter into
   ``self.inkling_roce_comm`` after the stock communicators (PyNccl, PCIe-IPC, ...) exist.  At TP4
   the Inkling override sets ``ep_size = 4``, so ``moe_ep`` *is* the ``tp`` object, and so is
   ``attn_tp`` (no DP attention); ``moe_tp`` is a size-1 group.
2. ``GroupCoordinator._all_reduce_in_place`` (the ``inplace_all_reduce`` custom op: eager and
   full CUDA graphs with custom all-reduce disabled) and ``_all_reduce_out_place`` (the
   ``outplace_all_reduce`` custom op: torch.compile and piecewise graphs): eligible tensors go to
   RoCE (in place via ``out=input``), the rest to the stock dispatch.  Inkling's two per-layer
   all-reduces both arrive here on SM12x: the attention output after ``wo_ud``
   (``group_all_reduce = is_sm120`` -> ``attn_tp_group.all_reduce``, bf16) and the Marlin MoE
   combine (``multicast_combine = not is_sm120`` -> ``moe_ep_group.all_reduce``, **fp32**).
3. ``GroupCoordinator._all_gather_into_tensor`` (``reg_all_gather_into_tensor``): dim-0 gather
   written straight into SGLang's output buffer.  The vocab-parallel logits gather reaches it
   through ``MultimemAllGatherer(enabled=False)`` (SM120) -> ``tensor_model_parallel_all_gather``
   -> ``all_gather(dim=-1)``; the FP32 head's shard travels as bf16 pairs.
4. ``GroupCoordinator.graph_capture``: the runtime's ``capture()`` around the stock context, on
   the capture stream; ``destroy`` closes the runtime first.
5. ``SchedulerBatchResultProcessor.process_batch_result_{prefill,decode,idle}``: the fail-stop
   health check once the step's results are on the host and before they are consumed (the
   placement local-inference-lab/vllm#597 uses in vLLM's worker, mapped to SGLang's result
   processor).

Not routed: the symmetric-memory branch of ``all_reduce`` (only with --enable-symm-mem), the
FlashInfer all-reduce-only mode (``_fi_workspace_hint``; auto-enabled on SM90/SM100 only), the
PCIe-IPC backend (``pcie_ipc``; only with SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1, which ``install``
refuses).  ``verify_targets`` checks all of this inside the image before a RoCE boot.

Credits: RoCEnante (``b12x.comm.roce``) by Local Inference Lab, Luke Alonso (@lukealonso) and Jason
Cook (@original-el8), local-inference-lab/b12x#295, shipped in FlashInfer ``experimental/b12x``;
the vLLM shim the adapter ports: Jason Cook, local-inference-lab/vllm#597; its upstream-shaped port
(all-gather switch, 300M spin limit): tonyd2wild; the idea of routing SGLang's TP all-reduce to
RoCEnante: rhys101 (SG17, DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8).

Provenance: written for our MiniMax-M3.1 recipe against SGLang a6b0e4e (Apache-2.0) and FlashInfer
0.7.2 b12x (Apache-2.0) APIs, re-targeted here at SGLang 03fde9cc / FlashInfer 0.7.4; the shared
adapter is the Apache-2.0 vLLM #597 port in roce/.  No code is copied from rhys101's overlay.
"""
from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
from contextlib import contextmanager, nullcontext

TAG = "INKLING_TP4_ROCE"
DEFAULT_ROCE_PATH = "/opt/inkling-roce"
MOD_PARALLEL_STATE = "sglang.srt.distributed.parallel_state"
MOD_RESULT_PROCESSOR = "sglang.srt.managers.scheduler_components.batch_result_processor"
RESULT_METHODS = ("process_batch_result_prefill", "process_batch_result_decode",
                  "process_batch_result_idle")
COMM_ATTR = "inkling_roce_comm"
_PATCHED = "_inkling_sgl_roce_patched"
_OFF = ("", "0", "off", "false", "no")
# Stock backends that would sit in front of the hooked paths or bypass them; refused when RoCE is on.
CONFLICTING_ENV = ("SGLANG_ENABLE_PCIE_IPC_ALLREDUCE",)

logger = logging.getLogger("sglang.inkling_roce")

# Every live adapter in this process (one per TP group; the scheduler process owns its rank's).
_LIVE: list = []


def enabled(environ=os.environ) -> bool:
    return environ.get(TAG, "0").strip().lower() not in _OFF


def _is_tp_group(unique_name) -> bool:
    return str(unique_name or "").split(":", 1)[0] == "tp"


def roce_comm_of(coordinator):
    comm = getattr(coordinator, COMM_ATTR, None)
    if comm is None or getattr(comm, "disabled", True):
        return None
    return comm


def _default_adapter(**kwargs):
    from inkling_roce import adapter

    adapter.logger = logger
    return adapter.InklingRoceAllReduce(**kwargs)


# -- 1-4: GroupCoordinator ---------------------------------------------------------------


def patch_parallel_state(module, adapter_factory=None) -> None:
    cls = module.GroupCoordinator
    if getattr(cls, _PATCHED, False):
        return
    make = adapter_factory or _default_adapter
    orig_init = cls.__init__
    orig_in_place = cls._all_reduce_in_place
    orig_out_place = cls._all_reduce_out_place
    orig_gather = cls._all_gather_into_tensor
    orig_graph_capture = cls.graph_capture
    orig_destroy = cls.destroy

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        setattr(self, COMM_ATTR, None)
        if not _is_tp_group(getattr(self, "unique_name", "")) or self.world_size <= 1:
            return
        if getattr(self.device, "type", "cuda") != "cuda":
            return
        comm = make(group=self.cpu_group, device_group=self.device_group, device=self.device)
        setattr(self, COMM_ATTR, comm)
        if not getattr(comm, "disabled", True):
            _LIVE.append(comm)

    @functools.wraps(orig_in_place)
    def _all_reduce_in_place(self, input_):
        comm = roce_comm_of(self)
        if comm is not None and comm.should_custom_ar(input_):
            comm.custom_all_reduce(input_, out=input_)
            return None
        return orig_in_place(self, input_)

    @functools.wraps(orig_out_place)
    def _all_reduce_out_place(self, input_, outplace_all_reduce_method):
        comm = roce_comm_of(self)
        # "auto" (torch.compile) and "pynccl" (piecewise graphs, or auto resolved to it) are the
        # methods the stock path would send to NCCL; ca/qr/pymscclpp/symm-mem/pcie_ipc keep theirs.
        if (outplace_all_reduce_method in ("auto", "pynccl") and comm is not None
                and comm.should_custom_ar(input_)):
            return comm.custom_all_reduce(input_)
        return orig_out_place(self, input_, outplace_all_reduce_method)

    @functools.wraps(orig_gather)
    def _all_gather_into_tensor(self, output, input):
        comm = roce_comm_of(self)
        if (comm is not None and output.is_contiguous()
                and comm.should_all_gather(input, 0)):
            comm.all_gather(input, 0, out=output)
            return None
        return orig_gather(self, output, input)

    @contextmanager
    def graph_capture(self, graph_capture_context=None, stream=None):
        comm = roce_comm_of(self)
        with orig_graph_capture(self, graph_capture_context, stream) as context:
            outer = comm.capture(context.stream) if comm is not None else nullcontext()
            with outer:
                yield context

    functools.update_wrapper(graph_capture, orig_graph_capture)

    @functools.wraps(orig_destroy)
    def destroy(self, *args, **kwargs):
        comm = getattr(self, COMM_ATTR, None)
        if comm is not None:
            if comm in _LIVE:
                _LIVE.remove(comm)
            comm.close()
            setattr(self, COMM_ATTR, None)
        return orig_destroy(self, *args, **kwargs)

    cls.__init__ = __init__
    cls._all_reduce_in_place = _all_reduce_in_place
    cls._all_reduce_out_place = _all_reduce_out_place
    cls._all_gather_into_tensor = _all_gather_into_tensor
    cls.graph_capture = graph_capture
    cls.destroy = destroy
    setattr(cls, _PATCHED, True)


# -- 5: result-boundary health check -------------------------------------------------------


def synchronize_and_check(result, live=_LIVE) -> None:
    """Fail-stop check at the result boundary (b12x: "call it after graph replays").

    ``check_health`` reads a host word the kernels write on a timed-out wait, so it is only
    meaningful once the step's kernels have finished: wait on SGLang's own D2H event when the
    scheduler recorded one, otherwise on the whole device (the overlap scheduler and the
    draft worker may have launched on several streams).  No RoCE runtime, no wait.
    """
    comms = list(live)
    if not comms:
        return
    event = getattr(result, "copy_done", None)
    if event is None:
        import torch

        torch.cuda.synchronize()
    else:
        event.synchronize()
    failures = []
    for comm in comms:
        try:
            comm.check_health()
        except Exception as exc:  # noqa: BLE001 - report every rail/runtime, then stop
            failures.append(exc)
    if failures:
        raise failures[0]


def patch_result_processor(module, check=synchronize_and_check) -> None:
    cls = module.SchedulerBatchResultProcessor
    if getattr(cls, _PATCHED, False):
        return
    for name in RESULT_METHODS:
        orig = getattr(cls, name)
        params = list(inspect.signature(orig).parameters)
        index = params.index("result")

        def make(orig=orig, index=index):
            @functools.wraps(orig)
            def method(self, *args, **kwargs):
                result = kwargs["result"] if "result" in kwargs else args[index - 1]
                check(result)  # the stock synchronize() that follows is then a no-op
                return orig(self, *args, **kwargs)

            return method

        setattr(cls, name, make())
    setattr(cls, _PATCHED, True)


PATCHES = {MOD_PARALLEL_STATE: patch_parallel_state, MOD_RESULT_PROCESSOR: patch_result_processor}


# -- in-image check (CPU) ------------------------------------------------------------------


def _params(fn):
    return list(inspect.signature(inspect.unwrap(fn)).parameters)


# Inkling routing facts the hooks rely on, checked as source text (the model modules import
# Triton/CuTe kernels, so they are read, not imported).  Path relative to the sglang package.
ROUTE_FACTS = (
    ("srt/arg_groups/model_overrides/inkling.py", 'overrides["ep_size"] = moe_ranks',
     "Inkling no longer forces ep_size = TP, so moe_ep may not be the tp group"),
    ("srt/distributed/parallel_state.py",
     "if moe_ep_size == tensor_model_parallel_size and not _is_npu:\n        _MOE_EP = _TP",
     "moe_ep is no longer the tp object at ep == tp"),
    ("srt/distributed/parallel_state.py",
     "if attn_tp_size == tensor_model_parallel_size:\n        _ATTN_TP = _TP",
     "attn_tp is no longer the tp object without DP attention"),
    ("srt/models/inkling_common/inkling2_attn.py", "self.group_all_reduce = get_platform().is_sm120",
     "attention no longer uses the group all-reduce on SM12x"),
    ("srt/models/inkling_common/inkling2_attn.py", "return parallel.attn_tp_group.all_reduce(output)",
     "attention group all-reduce call changed"),
    ("srt/models/inkling_common/inkling2_moe.py", "self.multicast_combine = not platform.is_sm120",
     "the MoE combine no longer avoids multicast on SM12x"),
    ("srt/models/inkling_common/inkling2_moe.py", "combined = group.all_reduce(",
     "the Marlin MoE combine no longer calls group.all_reduce"),
    ("srt/models/inkling2.py", "MultimemAllGatherer(max_tokens=0, enabled=False)",
     "the SM12x logits gather no longer bypasses multimem"),
    ("srt/distributed/device_communicators/triton_symm_mem_ag.py",
     "return tensor_model_parallel_all_gather(x, dim=-1)",
     "MultimemAllGatherer no longer falls back to tensor_model_parallel_all_gather"),
    ("srt/arg_groups/overrides.py", "and (get_platform().is_sm90 or get_platform().is_sm100)",
     "FlashInfer all-reduce fusion auto-enable may now cover SM12x (would bypass the hooks)"),
    ("srt/distributed/device_communicators/pcie_ipc_ar.py",
     "envs.SGLANG_ENABLE_PCIE_IPC_ALLREDUCE.get()",
     "the PCIe-IPC all-reduce is no longer opt-in"),
)


def _verify_route_facts(sglang_dir) -> list:
    problems = []
    for rel, needle, why in ROUTE_FACTS:
        path = os.path.join(sglang_dir, rel)
        try:
            with open(path) as f:
                text = f.read()
        except OSError as exc:
            problems.append(f"{rel}: not readable ({exc})")
            continue
        if needle not in text:
            problems.append(f"{rel}: {why}")
    return problems


def verify_targets() -> list:
    """Import the image's SGLang modules (CPU is enough) and return every mismatch found."""
    import importlib

    problems = []

    def need(cond, what):
        if not cond:
            problems.append(what)

    ps = importlib.import_module(MOD_PARALLEL_STATE)
    gc = getattr(ps, "GroupCoordinator", None)
    need(gc is not None, "GroupCoordinator missing")
    if gc is not None:
        need(_params(gc._all_reduce_in_place) == ["self", "input_"],
             f"_all_reduce_in_place{_params(gc._all_reduce_in_place)}")
        need(_params(gc._all_reduce_out_place) == ["self", "input_", "outplace_all_reduce_method"],
             f"_all_reduce_out_place{_params(gc._all_reduce_out_place)}")
        need(_params(gc._all_gather_into_tensor) == ["self", "output", "input"],
             f"_all_gather_into_tensor{_params(gc._all_gather_into_tensor)}")
        need(_params(gc.graph_capture)[:3] == ["self", "graph_capture_context", "stream"],
             f"graph_capture{_params(gc.graph_capture)}")
        need(callable(getattr(gc, "destroy", None)), "GroupCoordinator.destroy missing")
        init_src = inspect.getsource(inspect.unwrap(gc.__init__))
        for attr in ("cpu_group", "device_group", "unique_name", "world_size", "device"):
            need(f"self.{attr} = " in init_src, f"GroupCoordinator.__init__ no longer sets self.{attr}")
        ar_src = inspect.getsource(gc.all_reduce)
        need("inplace_all_reduce(input_, group_name=self.unique_name)" in ar_src,
             "all_reduce no longer falls back to the inplace_all_reduce op")
        ag_src = inspect.getsource(gc.all_gather)
        need("self.all_gather_into_tensor(output_tensor, input_)" in ag_src,
             "all_gather no longer goes through all_gather_into_tensor")
        agt_src = inspect.getsource(gc.all_gather_into_tensor)
        need("reg_all_gather_into_tensor(output, input, group_name=self.unique_name)" in agt_src,
             "all_gather_into_tensor no longer goes through the reg_all_gather_into_tensor op")
        out_src = inspect.getsource(inspect.unwrap(gc._all_reduce_out_place))
        need('outplace_all_reduce_method = "pynccl"' in out_src,
             "_all_reduce_out_place no longer resolves auto to pynccl")
    need(callable(getattr(ps, "in_the_same_node_as", None)), "in_the_same_node_as missing")
    for name in ("inplace_all_reduce", "outplace_all_reduce", "reg_all_gather_into_tensor"):
        need(callable(getattr(ps, name, None)), f"custom op {name} missing")
    brp = importlib.import_module(MOD_RESULT_PROCESSOR)
    proc = getattr(brp, "SchedulerBatchResultProcessor", None)
    need(proc is not None, "SchedulerBatchResultProcessor missing")
    if proc is not None:
        for name in RESULT_METHODS:
            fn = getattr(proc, name, None)
            need(fn is not None and "result" in _params(fn), f"{name} missing or has no result arg")
            if fn is not None:
                need("result.copy_done.synchronize()" in inspect.getsource(inspect.unwrap(fn)),
                     f"{name} no longer synchronizes result.copy_done")
    import sglang

    problems += _verify_route_facts(os.path.dirname(os.path.abspath(sglang.__file__)))
    return problems


def install(environ=os.environ):
    """Called from sitecustomize when INKLING_TP4_ROCE=1; returns the post-import finder."""
    for name in CONFLICTING_ENV:
        if environ.get(name, "").strip().lower() not in _OFF:
            raise RuntimeError(f"{TAG}=1 conflicts with {name}={environ[name]!r}")
    path = environ.get(f"{TAG}_PATH", DEFAULT_ROCE_PATH).strip()
    if path and path not in sys.path:
        if not os.path.isdir(path):
            raise RuntimeError(f"{TAG}=1 but {path} (roce/) is not mounted")
        sys.path.insert(0, path)
    from inkling_roce.boot import install_hooks

    return install_hooks(PATCHES)


if __name__ == "__main__":  # in-image preflight: python3 /opt/inkling-tp4/inkling_sgl_roce.py
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    path = os.environ.get(f"{TAG}_PATH", DEFAULT_ROCE_PATH)
    if path not in sys.path:
        sys.path.insert(0, path)
    issues = verify_targets()
    from b12x.comm import roce

    from inkling_roce.adapter import PREPARED_DTYPES, REQUIRED_B12X_ROCE_API_VERSION, uses_plan_api

    if getattr(roce, "API_VERSION", None) != REQUIRED_B12X_ROCE_API_VERSION:
        issues.append(f"image b12x.comm.roce API_VERSION {getattr(roce, 'API_VERSION', None)}")
    if not uses_plan_api(roce):
        issues.append("image b12x.comm.roce has no prepared-plan API")
    missing = [d for d in PREPARED_DTYPES if d not in tuple(getattr(roce, "SUPPORTED_DTYPES", ()))]
    if missing:
        issues.append(f"image b12x.comm.roce does not support {missing}")
    from b12x.comm.roce import _proxy

    lib = _proxy.load()  # the .so build_proxy.sh put in B12X_ROCE_CACHE_DIR must load here
    cache = os.environ.get("B12X_ROCE_CACHE_DIR", "")
    so = str(_proxy._build())
    if cache and not so.startswith(cache):
        issues.append(f"proxy {so} is not in B12X_ROCE_CACHE_DIR={cache}")
    print(f"{TAG} preflight: b12x {roce.__file__} API {roce.API_VERSION} proxy {so} "
          f"ABI {lib.roce_abi_version()}", file=sys.stderr)
    for issue in issues:
        print(f"{TAG} preflight: {issue}", file=sys.stderr)
    print(f"{TAG} preflight: {'FAIL' if issues else 'ok'}", file=sys.stderr)
    sys.exit(1 if issues else 0)
