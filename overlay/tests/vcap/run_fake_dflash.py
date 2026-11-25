"""Runs the toy DFLASH loop on the CPU stand-in package with the overlay armed the way the container arms it
(sitecustomize on PYTHONPATH). Saves the worker's reference history (torch.save) and prints one JSON line.

Usage (env): INKLING_VERIFY_CAP / INKLING_VERIFY_CAP_STEPLOG as in the container, FAKE_TEMPS=0,0.8,..(one per
request; 0 = greedy), FAKE_STEPS=n, FAKE_SKIP=i,j (steps that bypass _accept_block), FAKE_CAPACITY=n,
FAKE_HISTORY=path.

TP stand-in (fake_tp.py; one process per rank): FAKE_TP=4 FAKE_RANK=r FAKE_DIST=tcp://127.0.0.1:port.
FAKE_FULL_GATHER=1 resolves the vocab-sharded verify logits by the full gather instead of the cross-shard argmax;
FAKE_PERTURB_RANK=r halves rank r's walked-row max probabilities after every feature write (a rank-local feature
difference); FAKE_FAIL_FEATURES_RANK=r marks rank r's features failed after attach; FAKE_NO_DIST=1 / FAKE_ATTN_DP=1
for refusals that fire before any collective; FAKE_PLAN_RAISE_RANK=r makes rank r's adaptive length computation raise
(a rank-local planning failure); FAKE_BLOCK_VERIFY=1 sets the worker's block verification flag.
"""

import json
import os
from types import SimpleNamespace

import torch

torch.use_deterministic_algorithms(True)


def main() -> None:
    tp = int(os.environ.get("FAKE_TP", "1"))
    rank = int(os.environ.get("FAKE_RANK", "0"))
    group = None
    if tp > 1:
        import fake_tp

        group = fake_tp.install(rank, tp, os.environ.get("FAKE_DIST", ""),
                                backed=not os.environ.get("FAKE_NO_DIST"), attn_dp=bool(os.environ.get("FAKE_ATTN_DP")))
    from inkling_vcap.state import STATE

    if STATE.config is not None:
        from inkling_vcap import dflash

        cap = int(os.environ.get("FAKE_CAPACITY", "8"))
        dflash._graph_capacity = lambda: cap
        if os.environ.get("FAKE_FAIL_FLUSH"):
            from inkling_vcap import dfsteplog

            def broken(self, records):
                raise OSError(28, "No space left on device (test)")

            dfsteplog.DfStepLog._flush = broken
        if tp > 1 and os.environ.get("FAKE_PLAN_RAISE_RANK", "") == str(rank):
            from inkling_vcap import dfcap

            def boom(*args, **kwargs):
                raise RuntimeError("test: rank-local planning failure")

            dfcap.lens_policy = boom
            dfcap.lens_conf = boom
        if tp > 1 and os.environ.get("FAKE_PERTURB_RANK", "") == str(rank):
            orig_record = dflash.DflashFeatures.record

            def perturbed(self, **kwargs):
                orig_record(self, **kwargs)
                self.m.mul_(0.5)

            dflash.DflashFeatures.record = perturbed
    from sglang.srt.speculative import dflash_worker_v2 as w

    w.BLOCK_VERIFICATION = bool(os.environ.get("FAKE_BLOCK_VERIFY"))

    temps = [float(t) for t in os.environ.get("FAKE_TEMPS", "0,0.8,0").split(",")]
    bs = len(temps)
    worker = w.DFlashWorkerV2(seed=int(os.environ.get("FAKE_SEED", "0")))
    worker.skip_accept_steps = {int(s) for s in os.environ.get("FAKE_SKIP", "").split(",") if s}
    if os.environ.get("FAKE_FULL_GATHER"):
        worker._skip_verify_logits = False
    if tp > 1 and os.environ.get("FAKE_FAIL_FEATURES_RANK", "") == str(rank) and STATE.df is not None:
        STATE.df.failed = "test: rank-local feature failure"
    batch = SimpleNamespace(
        seq_lens=torch.tensor([17 + 5 * i for i in range(bs)], dtype=torch.int64),
        anchor=torch.tensor([3 + i for i in range(bs)], dtype=torch.int64),
        reqs=[SimpleNamespace(rid=f"req-{i}", origin_input_ids=list(range(5 + i, 22 + 6 * i))) for i in range(bs)],
        sampling_info=SimpleNamespace(temperatures=torch.tensor(temps, dtype=torch.float32)[:, None],
                                      has_custom_logit_processor=bool(os.environ.get("FAKE_CUSTOM_LP"))),
    )
    emitted = []
    for _ in range(int(os.environ.get("FAKE_STEPS", "40"))):
        emitted.append(worker.forward_batch_generation(batch).tolist())
        if tp > 1 and STATE.src_index is not None:
            worker.history[-1]["src_index"] = STATE.src_index[: bs * 8].clone()
        if os.environ.get("FAKE_STEP_SLEEP"):
            import time

            time.sleep(float(os.environ["FAKE_STEP_SLEEP"]))
    summary = {"emitted": emitted, "enabled": STATE.enabled, "applied": sorted(STATE.applied),
               "algorithm": STATE.algorithm}
    if STATE.steplog is not None:
        STATE.steplog.close()
        summary.update(steps=STATE.steplog.step, chunks=STATE.steplog.chunk, dropped=STATE.steplog.dropped,
                       written=STATE.steplog.written, failed=STATE.steplog.failed,
                       writer_error=repr(STATE.steplog._writer_error) if STATE.steplog._writer_error else None,
                       missing_routes=STATE.steplog.missing_routes, stale=STATE.steplog.stale_features,
                       run=STATE.steplog.run, skipped_walks=STATE.df.skipped)
    if tp > 1:
        sync = STATE.tp_sync
        summary.update(rank=rank, tp_size=STATE.tp_size, collectives=dict(group.counts),
                       tp_plan_errors=STATE.tp_plan_errors,
                       tp_sync=None if sync is None else dict(mode=sync.mode, calls=sync.calls,
                                                              mismatch_steps=sync.mismatch_steps,
                                                              mismatch_rows=sync.mismatch_rows))
    path = os.environ.get("FAKE_HISTORY")
    if path:
        torch.save(worker.history, path)
    if group is not None and group.backed:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
