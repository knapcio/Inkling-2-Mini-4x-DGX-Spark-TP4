"""Runs the toy DSpark loop on the CPU stand-in package, with the overlay armed the
same way the container arms it (sitecustomize on PYTHONPATH). Prints one JSON line.

Usage (env): INKLING_VERIFY_CAP=..., FAKE_PATH=eager|epilogue, FAKE_BS=n
"""

import json
import os
import sys

import torch

torch.use_deterministic_algorithms(True)


def main() -> None:
    from sglang.srt.speculative.dspark_components import dspark_worker_v2 as w

    from inkling_vcap.state import STATE

    bs = int(os.environ.get("FAKE_BS", "2"))
    prompts = [[3, 17, 5, 9], [11, 2, 30], [7, 7, 1, 22, 4]][:bs]
    worker = w.DSparkWorkerV2(bs=bs, prompts=prompts,
                              path=os.environ.get("FAKE_PATH", "eager"),
                              gen_tokens=int(os.environ.get("FAKE_TOKENS", "48")))
    seqs, steps = worker.run()
    unions = []
    for layer in worker.model.layers:
        unions.extend(len(t) for t in layer.experts.touched)
    if STATE.enabled:
        STATE.flush_stats()
    print(json.dumps({
        "seqs": seqs,
        "steps": steps,
        "verify_rows": worker.verify_rows,
        "enabled": STATE.enabled,
        "applied": sorted(STATE.applied),
        "remap_recorded": STATE.remap_recorded,
        "cutoff_injected": STATE.cutoff_injected,
        "static_accept_recorded": STATE.static_accept_recorded,
        "mean_union": sum(unions) / max(1, len(unions)),
    }))


if __name__ == "__main__":
    sys.exit(main())
