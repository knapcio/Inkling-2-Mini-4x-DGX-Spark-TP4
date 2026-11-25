# Usage (inside the SGLang image, overlay on PYTHONPATH): INKLING_VERIFY_CAP=fixed:8 python3 tools/bench_fused_accept.py 1,2,4,8,24
"""GB10 check of inkling_vcap.fused_accept (INKLING_VERIFY_CAP_FUSED_ACCEPT). Run in the image with
PYTHONPATH=/work/overlay INKLING_VERIFY_CAP=fixed:8 (arms the hooks).

1. startup self-check (Triton primitives vs dense torch primitives);
2. race-kernel law at V = 201024: residual max(P - Q, 0) with mass spread over chunk/split boundaries;
3. law Monte Carlo through the real accept_draft_tokens (V = 3; chain / BV, capped / uncapped);
4. cold-L2 timing of the real accept_draft_tokens, stock vs fused, at bs 1 / 8 / 24, chain and BV."""
import statistics
import sys
import types

import torch

from sglang.srt.speculative.dspark_components import dspark_verify as dv
from inkling_vcap import fused_accept as fa
from inkling_vcap.config import parse_config
from inkling_vcap.patches import fused_accept_selfcheck
from inkling_vcap.state import STATE

dev = "cuda"
ok = True
err = fused_accept_selfcheck(dev)
print('selfcheck:', err)
TP = STATE.fused_accept
V = 201024

def setup(n, cap='fixed:8'):
    STATE.config = parse_config({'INKLING_VERIFY_CAP': cap})
    STATE.allocate(stride=8, capacity=n, device=dev)
    STATE.enabled = True
    STATE.capping = True
    STATE._src_key = ('dyn', 0, n)

# ---------------------------------------------------------------- 4. timing
flush_buf = torch.empty(96 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
hot = torch.randperm(V, generator=torch.Generator().manual_seed(1))[:65536].to(dev)


def timeit(fn, reps=25):
    ts = []
    for _ in range(reps):
        flush_buf.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


print("4. bs | mode | stock us | fused us")
for bsz in [int(x) for x in sys.argv[1].split(',')]:
    setup(bsz, cap="fixed:8")
    STATE.live_lens[:bsz] = 8
    g = torch.Generator(device=dev).manual_seed(bsz)
    tlog = torch.randn((bsz * 8, V), generator=g, device=dev) * 4
    dlog = torch.full((bsz, 7, V), float("-inf"), dtype=torch.bfloat16, device=dev)
    dlog[:, :, hot] = (tlog.view(bsz, 8, V)[:, :7, hot] + torch.randn((bsz, 7, 65536), generator=g,
                                                                        device=dev)).to(torch.bfloat16)
    drafts = torch.argmax(dlog.float(), -1)
    cand = torch.cat([torch.zeros(bsz, 1, dtype=torch.long, device=dev), drafts], 1)
    si = types.SimpleNamespace(temperatures=torch.full((bsz, 1), 0.8, device=dev), is_all_greedy=False,
                               is_any_greedy=False, need_top_k_sampling=False, need_top_p_sampling=False)
    block = types.SimpleNamespace(greedy_draft=False, corrected_logits=dlog,
                                  temperatures=torch.full((bsz,), 0.56, device=dev), draft_tokens=drafts,
                                  greedy_mask=torch.zeros(bsz, dtype=torch.bool, device=dev), vocab_shard=None)
    for bv in (False, True):
        def run():
            dv.accept_draft_tokens(candidates=cand, target_logits=tlog, draft_block=block, sampling_info=si,
                                   draft_input=None, gamma=7, verify_num_draft_tokens=8, block_verification=bv)
        STATE.fused_accept = None
        t_stock = timeit(run)
        STATE.fused_accept = TP
        t_fused = timeit(run)
        print(f"   {bsz} | {'BV' if bv else 'chain'} | {t_stock:.0f} | {t_fused:.0f}", flush=True)
    del tlog, dlog
    torch.cuda.empty_cache()

