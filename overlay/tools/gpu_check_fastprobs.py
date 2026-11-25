# Usage (inside the SGLang image, overlay on PYTHONPATH): python3 tools/gpu_check_fastprobs.py
"""GB10 check of inkling_vcap.fastprobs (the image's split softmax kernels on CUDA) against the
references, plus cold-L2 timings. Run in the image with PYTHONPATH=/work/overlay."""
import statistics

import torch

from sglang.kernels.ops.speculative.dspark import dspark_accept as da
from inkling_vcap import fastprobs, logic
from inkling_vcap.patches import fastprobs_selfcheck

dev = "cuda"
V, G = 201024, 7
print("selfcheck:", fastprobs_selfcheck(dev), flush=True)
flush_buf = torch.empty(96 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
hot = torch.randperm(V, generator=torch.Generator().manual_seed(1))[:65536].to(dev)


def make(bs, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.full((bs, G, V), float("-inf"), dtype=torch.bfloat16, device=dev)
    vals = (torch.randn((bs, G, hot.numel()), generator=g, device=dev) * 3).to(torch.bfloat16)
    x[:, :, hot] = vals
    temps = torch.full((bs,), 0.8, device=dev)
    greedy = torch.zeros(bs, dtype=torch.bool, device=dev)
    greedy[::5] = True
    return x, temps, greedy


def timeit(fn, reps=25):
    ts = []
    for _ in range(reps):
        flush_buf.sum()  # cold L2: read > L2
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


print("bs | rows | m err | q err | ref m us | fast m us | flashinfer q us | fast q us | logits MB")
for bs in (1, 4, 8, 16, 24):
    x, temps, greedy = make(bs, bs)
    tok = torch.zeros(bs, G, dtype=torch.long, device=dev)
    ref_m = logic.draft_token_probs(x, tok, temperatures=temps, greedy_mask=greedy, use_max=True)
    got_m = fastprobs.max_prob(x, temperatures=temps, greedy_mask=greedy)
    flat = x.reshape(bs * G, V)
    ref_q = da.SoftmaxTemp.execute(logits=flat, temperatures=temps, rows_per_request=G)
    got_q = fastprobs.softmax_temp(flat, temps, G)
    em = float((ref_m - got_m).abs().max())
    eq = float((ref_q - got_q).abs().max())
    t_rm = timeit(lambda: logic.draft_token_probs(x, tok, temperatures=temps, greedy_mask=greedy, use_max=True))
    t_fm = timeit(lambda: fastprobs.max_prob(x, temperatures=temps, greedy_mask=greedy))
    t_rq = timeit(lambda: da.SoftmaxTemp.execute(logits=flat, temperatures=temps, rows_per_request=G))
    t_fq = timeit(lambda: fastprobs.softmax_temp(flat, temps, G))
    mb = flat.numel() * 2 / 1e6
    print(f"{bs} | {bs * G} | {em:.1e} | {eq:.1e} | {t_rm:.0f} | {t_fm:.0f} | {t_rq:.0f} | {t_fq:.0f} | {mb:.1f}",
          flush=True)
    del x, ref_q, got_q
    torch.cuda.empty_cache()
