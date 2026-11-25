# Inkling-2-Mini on four NVIDIA DGX Sparks (TP4)

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) across four DGX
Sparks (GB10, sm_121) with tensor parallelism 4 / expert parallelism 4 over the ConnectX-7 RoCE fabric, on the pinned
SGLang image, with the DFlash2 draft model
[incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), a one-shot RDMA all-reduce
(RoCEnante) and the TP-safe subset of the single-Spark recipe's overlays. The target model's weights and its output
distribution are unchanged.

**Status (2026-11-25, v0.3):** measured on one four-Spark fleet on one day. v0.3 adds the verify-width cap and the
sharded certified LM head (both ported to TP in this repository) and the single-Spark recipe's v2.9.2 prefill hoist
and prompt-end cache touch to v0.2.

This is the TP4 companion of the single-Spark recipe
[Inkling-2-Mini-1x-DGX-Spark](https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark) (v2.9.2); the overlays come
from there. What TP4 buys over one Spark is latency per request, long-context decode and a KV pool of 8.4M tokens. It
does not buy aggregate throughput: four independent single-Spark replicas deliver about as many tokens per second in
total.

One default profile: SGLang `lmsysorg/sglang:inkling-2-mini` (pinned by digest) on four Sparks, `--tp 4` (EP4 is set
by the image; the routed experts run on Marlin), the DFlash2 drafter at revision `4f9088ea` with a TP pre-patch, the
RoCEnante one-shot RDMA all-reduce / all-gather on the TP group, NCCL 2.30.7 for everything else, and the
single-Spark recipe's overlay levers that are exact, bit-identical or FP-order class at TP4, several of them ported
to TP here (see "The levers"). BF16 KV, context 262,144, 48 running requests, `MEM_FRACTION=0.80`.

The current column is one four-Spark fleet on 2026-11-25 with the release defaults (`results/REL2full/`); the stock
column is the stock engine at TP4 on the same fleet (`results/B1stock/`); the single-Spark columns are the
[1x recipe](https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark)'s published numbers (one Spark, v2.9 / v2.8.x).

| | 1 Spark, stock | 1 Spark, v2.9 | 4 Sparks, stock TP4 (DSpark) | **4 Sparks, TP4 v0.3 (this recipe)** |
|---|---:|---:|---:|---:|
| speedbench full, 73 prompts, c1, tok/s | 40.6 | 73.34 | - | **152.97** |
| speedbench quick, 30 prompts, c1, tok/s (prose) | 41.86 (33.18) | 70.61 (65.67) | 84.30 (68.17) | **158.4-160.5** (4 boots) |
| decode, prose, tok/s at c1 / c4 / c16 | 33.2 / 75.2 / 186.2 | 65.1 / 140.5 / 288.3 | 66.7 / 157.8 / 372.4 | **125.0 / 277.7 / 572.6** |
| decode, code, tok/s at c1 / c4 / c16 | 43.6 / 87.2 / 221.3 | 79.0 / 141.6 / 298.2 | 80.6 / 193.0 / 445.7 | **159.1 / 300.5 / 621.3** |
| thinking c8 / c24, aggregate tok/s (effort high, T=0.8)\* | - | 277.4 / 515.5 (v2.8) | - | **549.3 / 850.1** |
| decode at 32K / 128K context, c1, tok/s | 55.1 / 37.1 | 66.05 / 50.91 | - | **128.3 / 108.7** |
| cold prefill, 8K tokens, tok/s | 3,399 | 5,553 | - | **5,211** |
| KV pool, global-layer tokens | 843K | 2,371,200 | 6,601,344 | **8,377,216** |
| worst-case MemAvailable (thinking c24 + cold 240K needle), lowest rank | - | 4.6 GiB | - | **12.7 GiB** |

\* TP4: `bench/full.sh`, 16 / 48 requests x up to 4,096 tokens. Single Spark: as published by the 1x recipe, whose
row is labelled 48 x 2,048 tokens; not a like-for-like pair.

What the four Sparks buy is per-request speed and context: 2.09x the tuned single Spark on speedbench full, 1.9-2.0x
on single-request decode, ~2.1x at 128K context, and a KV pool of 8.4M tokens (about 32 requests at 256K). They do
not buy aggregate throughput: four single-Spark replicas at c4 each give 4 x 140.5 = 562 tok/s of prose, about TP4 at
c16 (573 tok/s), at half the per-request speed. Cold prefill is the weak spot: 5.2K tok/s at 8K is just below the
tuned single Spark (5.5K), because the routed experts run on Marlin at EP4 and the fused prefill attention of the 1x
recipe costs decode acceptance at TP4 (see "Open items").

### Before / after (same fleet, same harness, full runs)

| | v0.1 (`REL1full`) | v0.2 (`V02full`) | **v0.3 (`REL2full`)** | v0.3 vs v0.2 |
|---|---:|---:|---:|---:|
| what changed | DFlash2 + RoCEnante + exact / bit-identical levers | + prefill glue fusion | + verify-width cap, sharded certified head, hoist, cache touch | |
| speedbench full, all 73 prompts | 142.77 | 142.62 | **152.97** | +7.3 % |
| ... prose / code / structured | 125.28 / 157.10 / 153.79 | 125.00 / 156.77 / 153.52 | **135.10 / 173.12 / 162.46** | +8.1 / +10.4 / +5.8 % |
| ... agentic / multilingual / long context | 157.54 / 109.30 / 160.88 | 164.70 / 109.21 / 152.01 | **165.11 / 120.50 / 166.68** | +0.2 / +10.3 / +9.7 % |
| median step at c1 (speedbench full) | 21.87 ms | 21.89 ms | **20.09 ms** | -8.2 % |
| decode prose c1 / c4 / c16 | 125.6 / 275.5 / 544.4 | 125.7 / 272.1 / 551.9 | **125.0 / 277.7 / 572.6** | -0.6 / +2.1 / +3.8 % |
| decode code c1 / c4 / c16 | 143.1 / 274.4 / 557.0 | 142.8 / 274.8 / 571.3 | **159.1 / 300.5 / 621.3** | +11.4 / +9.4 / +8.8 % |
| decode structured / json c1 | 159.8 / 164.8 | 159.6 / 164.5 | **167.9 / 167.3** | +5.2 / +1.7 % |
| thinking c1 / c8 / c24 | 175.0 / 532.5 / 886.5 | 179.4 / 538.8 / 632.3\*\* | **165.3 / 549.3 / 850.1** | |
| cold prefill 1K / 8K / 32K / 128K | 3,929 / 4,616 / 4,539 / 4,087 | 4,239 / 5,163 / 5,067 / 4,513 | **4,264 / 5,211 / 5,152 / 4,504** | +0.6 / +0.9 / +1.7 / -0.2 % |
| decode at 32K / 128K | 110.1 / 111.4 | 134.8 / 111.6 | **128.3 / 108.7** | -4.8 / -2.6 % |
| KV pool (tokens) | 8,372,224 | - | **8,377,216** | |
| worst-case MemAvailable, ranks 0-3 (GiB) | 12.7 / 13.8 / 14.1 / 13.6 | 12.3 / 13.4 / 13.6 / 13.7 | **12.7 / 13.5 / 14.2 / 14.1** | |
| guard / sanity / needles 120K, 240K | 30 / 30, 11 / 11, pass | 30 / 30, 11 / 11, pass | **30 / 30, 11 / 11, pass** | |

\*\* v0.2's c24 run started with a 76-s admission stall (18 of the 24 requests reached the scheduler 76 s after
the first ones; per-request decode 41.3 tok/s against 42.9 / 41.6), see "Open items". Thinking runs are sampled
(T=0.8) with up to 4,096 tokens and vary by several per cent between runs; the long-context decode rows are two
documents each.

## Speed

Measured 2026-11-25 on four DGX Sparks (node0..node3) (GB10, kernel 6.17, GPU clock capped at 2200 MHz by each host's own timer,
one MikroTik CRS812 fabric), fresh start of all four containers, release defaults, first run after the boot discarded.
Harness: `bench/fullrun.sh` (the single-Spark recipe's `bench/full.sh` plus MemAvailable of all four ranks every 5 s).

### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories,
c1, temperature 0, 256 output tokens, reasoning effort low. Decode tok/s, geometric mean per category; 0 failed
prompts.

| | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TP4, v0.3** (`results/REL2full/`) | **135.10** | **173.12** | **162.46** | **165.11** | **120.50** | **166.68** | **152.97** |
| TP4, v0.2 (`results/V02full/`) | 125.00 | 156.77 | 153.52 | 164.70 | 109.21 | 152.01 | 142.62 |
| TP4, v0.1 (`results/REL1full/`) | 125.28 | 157.10 | 153.79 | 157.54 | 109.30 | 160.88 | 142.77 |
| TP4, DFlash2 + RoCE + exact levers only (B4, `results/B4full/`) | 113.37 | 154.77 | 143.03 | 149.48 | 103.12 | 141.10 | 133.03 |
| 1 Spark, v2.8.4 (1x recipe, box A) | 65.75 | 82.46 | 77.73 | 83.69 | 58.01 | 74.20 | 73.53 |
| 1 Spark, stock engine (1x recipe) | 32.6 | 48.0 | 44.3 | 49.3 | 25.3 | 51.0 | 40.6 |

v0.3 release run: median time to first token 82.3 ms, median step 20.09 ms, 2.94 tokens per step.

### Decode by prompt type and concurrency

T=0, 256 tokens, effort none; aggregate tok/s (per-stream median in brackets); c1 is per-stream. v0.3:

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 125.0 | 186.0 (101.4) | 277.7 (76.5) | 412.6 (54.8) | 572.6 (39.3) |
| code | 159.1 | 182.8 (102.4) | 300.5 (82.2) | 430.3 (62.6) | 621.3 (43.2) |
| structured | 167.9 | 223.6 (119.6) | 316.0 (86.4) | 446.7 (64.5) | 659.6 (46.9) |
| json | 167.3 | 220.4 (117.1) | 328.8 (89.9) | 463.7 (64.8) | 642.2 (48.8) |

### Thinking workloads

Effort high, T=0.8 (`bench/think_load.py`), v0.3: c1 3 x 2,048 tokens **165.3** tok/s; c8 16 x 4,096 tokens
**549.3** aggregate (79.9 per request); c24 48 x 4,096 tokens **850.1** aggregate (41.6 per request).

### Prefill and long context

speedbench longctx, 2 documents per length, v0.3:

| context | cold TTFT | cold prefill tok/s | cached TTFT | decode tok/s (512 tokens) |
|---|---:|---:|---:|---:|
| 1K | 0.24 s | 4,264 | 0.25 s | - |
| 8K | 1.56 s | 5,211 | 0.12 s | - |
| 32K | 6.35 s | 5,152 | 0.12 s | 128.3 |
| 128K | 29.1 s | 4,504 | 14.8 s | 108.7 |

The cached 128K TTFT re-prefills from the nearest hybrid-state checkpoint (as on one Spark).

### Agent loops

`bench/ag64.py` (the single-Spark recipe's AG64 harness): 24 agents, ~72K-token contexts, turn 1 cold then turn 2 on
the same conversations (`results/AGoff/`, v0.3 levers without the cache touch, `--enable-cache-report`): turn 1 447 s
for 1.73M prompt tokens; turn 2 85 s, 24 / 24 prefix-cache hits (~71.7K of 71.9K tokens cached), time to first token
~1.0 s. The second-turn misses the single-Spark recipe's v2.9.2 cache touch fixes do not occur at TP4 at this load.

### Memory and start-up

KV pool 8,377,216 global-layer tokens (BF16; the single Spark's packed 12-bit store refuses TP, and is not needed).
Lowest MemAvailable over the whole full run (thinking c24 and the cold 240K needle included): 12.7 / 13.5 / 14.2 /
14.1 GiB on ranks 0-3. Start-up ~2-3 min with warm caches. Requests for prompt-token logprobs of long prompts
(`return_logprob` with `logprob_start_len=0`) materialise full-vocabulary logits for every prompt token (~4 GB for a
5K-token prompt) and the allocator keeps that cache: after the logprob checks of this README, idle MemAvailable was
~8.8-10.7 GiB. The memory guard (3.5 GiB per rank) stays armed for the container's life.

### Quality checks

- speedbench quality guard (30 exact-answer items, T=0): 30 / 30 in v0.1, v0.2, v0.3 and every screen; sanity suite
  11 / 11 (chat, effort ladder, tool call and tool result, JSON schema), needles at 120K and 240K found.
- Certified head (v0.3): check mode on the fleet compared 6,145 certified verify steps (49,384 rows) with the full
  gather: 0 mismatches. Greedy output logprobs against stock TP4 with it on: 0.017 (as without).
- Prefill numerics: teacher-forced prompt-token logprobs on 30 prompts (15,680 tokens, `tools/tf_logprobs.py`), the
  release profile against the stock TP4 engine: mean |dlogprob| 0.017, NLL per token 2.5359 vs 2.5372 (the same-boot
  floor is 0.002-0.003; any floating-point-order change in the prefill path moves this model's prompt logprobs by
  ~0.5 mean |dlogprob| with an unchanged NLL, so this check separates bit-identical levers from the rest).
- Decode / verify numerics: greedy output logprobs on the common prefix (`tools/gen_logprobs.py`), release against
  stock TP4: mean |dlogprob| 0.016 at c1 and 0.021 at c8, the same as the stock engine's own c1-vs-c8 difference
  (0.021); greedy texts part at near-ties, as between batch layouts of the stock engine.
- Why the prefill check moves so much for FP-order changes: the stock TP4 engine with only its prefill CUDA graphs
  disabled (no overlay code; eager prefill sums in a different order) differs from itself with graphs by mean
  |dlogprob| 0.325 (NLL 2.5316 vs 2.5372) (`results/tf/chain-pfuse.log`). The NVFP4 activation quantisers turn
  last-bit differences into different quantisation buckets; NLL does not move. Levers in that class (pfuse) are
  judged by the quality guard, NLL and acceptance, not by this distance.
- TP4 against one Spark: not compared token by token here; TP changes the order of every reduction, so greedy texts
  differ from the single-Spark recipe at near-ties.

## The levers

Each row was added on top of the previous one in a separate boot of the four Sparks and screened with
`bench/tp4screen.sh` (speedbench quick: 30 prompts x 160 tokens, c1, T=0; decode table prose / code c1 / c4 / c16;
thinking c8: 16 x 2,048 tokens, T=0.8; speedbench quality guard). Screens of different boots differ in their greedy text
whenever a lever changes floating-point order, so the median decode step (ms) is the cleaner speed signal than tok/s;
the reproducer runs of nine boots with the final decode levers scored 149.6-152.1 on speedbench quick, three boots of
the drafter + RoCE stage 129.4-129.7. Raw files: `results/B*/`.

| # | lever | switch (default) | speedbench quick, c1 (median step) | output | credit |
|---|---|---|---:|---|---|
| B1 | stock engine at TP4: `--tp 4 --nnodes 4`, EP4 set by the image, experts on Marlin, the model's own DSpark draft, NCCL 2.30.7 over both PCIe halves of the 200G port | `SPEC=dspark OVERLAY=0 INKLING_TP4_ROCE=0` | 84.30 (29.2 ms) | stock | **SGLang** Inkling-2-Mini TP/EP support |
| B2 | DFlash2 drafter at TP4: incoai's compatibility patch refuses TP; `tp4/tp4_dflash2_prepatch.py` replaces that guard with a per-shard clamp of the real vocabulary (only the last shard is cut, to 49,290 rows; the union of the shards is the TP1 candidate set, CPU-tested) | `SPEC=dflash2` | 107.38 (28.9 ms, 3.00 tok/step) | draft proposals only | **incoai** drafter and patch; ours (TP clamp) |
| B3 | RoCEnante: the TP group's all-reduces up to 2 MiB (attention output BF16, Marlin MoE combine FP32) and all-gathers up to 16 MiB per rank (logits) go to the b12x one-shot RDMA kernel over both rails; NCCL keeps the rest | `INKLING_TP4_ROCE=1` | 129.62 (24.3 ms) | the one-shot kernel sums in a fixed rank order (identical bits on every rank); FP order differs from NCCL's ring; quality guard 30 / 30 | **Local Inference Lab** (Luke Alonso, Jason Cook), b12x #295 / vllm #597; port via our MiniMax recipe |
| B4 | exact sampled verification (reject / selector lane rules), block verification on sampled steps, prefix tail (2,048 tokens on prefix hits of prompts up to 4,096) and 8 draft-tail snapshots | `INKLING_RSFIX=1`, `BLOCK_VERIFY=1`, `INKLING_DFLASH_PREFIX_TAIL=2048`, `..._MAX_INPUT=4096`, `INKLING_DFLASH_TAIL_SNAPSHOTS=8` | 135.42 (24.4 ms, 3.12 tok/step); thinking c8 461.5 -> 480.9 | target law unchanged; draft-only state | ours (single-Spark recipe); block verification: **Sun et al.** |
| B5 | pico12 12-bit coded BF16 GEMV for the dense projections and the head shard (<= 16 rows), drafter MXFP8 layers (`mx8`) and b12x MXFP8 GEMM (`mx8w`), split-KV verify attention at every batch size, 17-256-row coded GEMM for verify batches | `INKLING_PICO12=dense,lmhead,stack1`, `INKLING_DFLASH_BYTES=mx8,mx8w`, `INKLING_VATTN=1`, `INKLING_WIDE=dense,lmhead` | 146.41 (23.0 ms) | weights exact per element (pico12, wide); FP32 order of the GEMVs differs; draft-only (mx8) | **Andy Shu** (pico / dsgb10 `bf16x`), **FlashInfer** b12x MXFP8 GEMM; ours |
| B5h | hot-vocabulary FP8 candidate plane for the drafter, ported to TP: each rank keeps the hot ids of its own shard (18,769 / 12,739 / 9,438 / 24,590 of 65,536), local top-16, then the stock K-wide gather; the shard plan is computed from the head's config on every rank (no new collective) | `INKLING_DFLASH_BYTES=mx8,mx8w,hot8` | 152.14 (22.2 ms); thinking c8 541.2 | draft proposals only (same candidate set as the TP1 hot plane, CPU-tested) | ours; hot list idea **FR-Spec** (Zhao et al.) |
| B7 | the two bit-identical prefill levers: relative bias written straight into FA4's sheared layout, fused SWA prologue | `INKLING_PREFILL=bias_layout`, `INKLING_PROLOGUE=1` | 152.08 (22.1 ms); prefill 8K / 32K 4,631 / 4,553 tok/s (+2.5 %); decode at 32K 130.2 | bit-identical (teacher-forced prompt logprobs at the same-boot floor) | ours (single-Spark recipe) |
| B8 | prefill glue fusion: the FP32 loop-message mixes, norms and residual updates of >= 512-row extends in fewer kernels | `INKLING_PFUSE=all` | 149.59 (22.2 ms, 3.21 tok/step = B7); prefill 8K / 32K 5,152 / 5,063 tok/s (+11.6 % over B7's 4,616 / 4,539); decode at 32K 133.6 | FP32-order class: prompt logprobs move like any prefill FP-order change (see Quality), NLL 2.534 vs stock 2.537; guard 30 / 30 | ours (single-Spark recipe) |
| B9 | verify-width cap on DFLASH, ported to TP: per request and step, verify only the first L of the 8 rows (stopping rule on the drafter's walked-row confidence); cut rows remapped to the anchor row's experts, which shrinks each rank's Marlin expert set. TP: whether a step caps is decided from rank-invariant inputs; the per-request lengths are all-gathered and every rank uses rank 0's; a settings digest (incl. the policy file) is compared at start-up. TP1 prices halved (`rowscale=0.5`; with TP1 prices the step shrank but tokens per step fell as much, flat) | `INKLING_VERIFY_CAP=dfpolicy:.../df-SLDF2x.json,bsmax=8,rowscale=0.5`, `INKLING_VERIFY_CAP_REMAP=1`, `INKLING_VERIFY_CAP_BV=mask` | 155.17 (21.0 ms, 3.15 tok/step), against ~150 for B8 (+3.4 %); prose c4 281.8, code c16 597.6 | target law unchanged (stopping rule; block verification with the sparse-q mask); the remap is FP-order class for live rows (remap on / off at `fixed:4`: c1 19 / 30 greedy texts identical, mean \|dlogprob\| 0.002, below the stock engine's own batch-layout noise of 0.021); no rank disagreement over a full screen (`TP_SYNC=check`) | ours (port of the single-Spark cap) |
| B10 | sharded certified greedy LM head: each rank screens its head shard with the 6-bit coded copy, one small all-gather makes the per-row bounds global, each rank runs the exact pass on the surviving tiles of its shard, and SGLang's deferred argmax replaces the logits all-gather on greedy verify steps (sampled / logprob steps keep the full gather). Start-up agreement and the bounds exchange are issued identically on every rank | `INKLING_CERT_HEAD=1`, `INKLING_CERT_TP=1` (+ `SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1`, set by the launcher) | 158.44 (20.5 ms, same 3.15 tok/step as B9); prose c16 587.6, code c16 637.4 | exact argmax by construction; check mode on the fleet: 6,145 steps / 49,384 rows certified, 0 mismatches against the full gather (NaN and cross-shard ties follow `torch.argmax`) | ours (the single-Spark certified head, sharded); pico's `bf16x` GEMV by **Andy Shu** for the exact pass |
| B11 | prefill loop-bank hoist over pfuse's loop mix (single-Spark v2.9.2): the loop controller's source projection prepared once per loop pass. TP: every rank votes its start-up self-test in one MIN all-reduce (gloo), so all ranks hoist or all run pfuse; `exact` refused at TP > 1 | `INKLING_PFUSE_HOIST=1` | 159.90 (20.5 ms); prefill 8K / 32K 5,224 / 5,147 tok/s (+1.2 / +1.6 %; 1x: +1.1 / +1.7 %) | FP32-order class: on 12 long documents (67K prompt tokens, teacher-forced) NLL 1.9435 with the hoist vs 1.9487 without, per-document differences of both signs (7 / 5) within the FP-order noise; tokens per step unchanged (3.15); guard 30 / 30; the 1x quality bracket also passed | ours (single-Spark recipe) |
| B12 | prompt-end cache touch (single-Spark v2.9.2): after a request finishes, its prompt path is re-touched with SGLang's `touch_prefix`, so the next turn of a long agent conversation keeps its prefix hit. TP: computed from scheduler state identical on every rank inside the consensus-checked checkpoint; arming agreed at start-up; a runtime overlay failure stops the server instead of one rank going inert | `INKLING_PROMPT_TOUCH=1` | no effect measured at TP4: with the touch OFF, the 24-agent second turn on ~72K-token contexts already hit the cache for 24 / 24 agents (~71.7K of 71.9K tokens cached, time to first token ~1.0 s; `results/AGoff/`), because the TP4 pools are much larger than one Spark's; kept for parity with the single-Spark recipe | none (cache LRU order only) | ours (single-Spark recipe); **SGLang** `touch_prefix` |

Versions: v0.1 = B7 (full run `results/REL1full/`), v0.2 = B8 (`results/V02full/`), v0.3 = B12, the release
defaults (`results/REL2full/`). The B9-B12 screens ran on v0.2 plus the levers above them in the table.

**Not in the profile:**

| lever | why |
|---|---|
| fused relative-bias prefill attention and sink gate (`INKLING_ATTN`, `INKLING_ATTN_GATE`) | prefill +24 % with the others below, but decode acceptance -5 to -7 % in two boots (3.00 tokens per step against 3.18-3.28 in every other arm, step time flat); cause not found |
| stacked Q/K/V split (`INKLING_PREFILL=qkvr_split`) | not bit-exact at TP4 shapes for 128-384-row prefill buckets: runs stock at TP > 1 |
| packed 12-bit KV store, fused sampled accept, the wide (17-128-row) certified head | TP1-only: they refuse or disarm at TP > 1 (the KV pool is 3.5x the single box's anyway) |
| expert-side levers (moe_tiles, MOE2, SCODES, NOEXP, PADX, prefill quant_once / partial_bf16) | inert: they patch the native SM120 MoE path, and at EP4 the image runs the experts on Marlin |

`scripts/launch-tp4.sh` refuses the rejected levers unless `ALLOW_TP4_UNVERIFIED=1`.

### Verify-width cap at TP4

The single-Spark recipe's verify-width cap (+13.0 % speedbench quick on one Spark) is ported in
`overlay/inkling_vcap/` (`dflash.py`, `dfcap.py`, `tpsync.py`; section "DFLASH at TP > 1" of its README):

- Whether a step caps is decided on the host from rank-invariant inputs. The per-request lengths of a capped step
  are all-gathered over the TP group and every rank uses rank 0's; a rank whose own proposal fails proposes full
  width and still joins the gather. At start-up the ranks compare a digest of their cap settings (including the
  policy file contents) and all refuse to start on a mismatch.
- `INKLING_VERIFY_CAP_BV=mask` is required at TP (block verification with the sparse-q mask).
- Output class: FP-order. The remap can change the FP32 reduction partition of a live row in Marlin's GEMM, like a
  different batch composition does; the target law is unchanged and greedy texts may differ at near-ties.
- The remap shrinks each rank's Marlin expert set (`fixed:4`: 137.4 with the remap against 122.1 without).
- Prices: `df-SLDF2x` was fitted on one Spark (step ~40 ms); at TP4 the step is ~21 ms and each rank reads a
  quarter of the expert union. With the TP1 prices (`results/C1cap/`) the step fell from 22.2 to 20.3 ms but tokens
  per step fell as much (speedbench quick 150.4, flat); `rowscale=0.5` (`results/C2cap05/`) gives 155.2. A refit
  from full-width step logs (`INKLING_VERIFY_CAP=steplog`, then the single-Spark recipe's `tools/dfreplay.py
  --per-mode --ctx-price --export-policy`) is left open.
- Dead (cut) rows are no longer counted in the capped accept's statistics (`cap_trim_lens`); the single-Spark recipe
  had the same miscount (tokens were never affected).

### Open items

- Fused prefill attention (`INKLING_ATTN`): +24 % cold prefill together with pfuse, but the global-layer path lowers
  drafter acceptance by 5-7 % at TP4 (the sink gate is not the cause; the sliding-window-only variant keeps
  acceptance). Root cause not found; this also blocks the single-Spark recipe's v2.9.1 global-layer gate, which
  lives in the same kernel's epilogue.
- `qkvr_split` at TP4 shapes: not bit-exact for 128-384-row prefill buckets; a per-bucket floor
  (`INKLING_PREFILL_QKVR_TP_MIN_M`) needs a GPU check.
- A refit of the verify-cap prices at TP4 (above), and the wide (17-128-row) certified head at TP.
- The c1 step (21 ms): routed experts on Marlin ~35 %, the one-shot collectives ~25 % (~188 per step, ~28 us each
  including rank skew), dense GEMVs ~20 % (`results/profile/`). Cold prefill (~5.2K tok/s at 8K) is the weak spot
  because the experts run on Marlin at EP4.
- One 76-s admission stall at the start of a thinking c24 run (v0.2 full run, `results/V02full/r0-server.log`):
  per-request decode was unchanged; not reproduced or explained.

## Running it

Requirements: four DGX Sparks on one ConnectX-7 fabric (here a MikroTik CRS812 with breakout DACs, one 200G port per
Spark seen as two PCIe halves `rocep1s0f0` / `roceP2p1s0f0`, GID index 3), kernel 6.17 (7.0.0-1019 breaks RDMA
memory registration on GB10), passwordless SSH from the head to the three workers on their fabric addresses, Docker
with the NVIDIA runtime, and on every node: the image, the weights, the drafter and NCCL 2.30.7 in `~/nccl-2.30.7`.

```bash
git clone https://github.com/knapcio/Inkling-2-Mini-4x-DGX-Spark-TP4 ~/inkling-tp4 && cd ~/inkling-tp4   # on the head
scripts/download.sh                      # on every node: weights + drafter at pinned revisions, sha256-checked
docker pull lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de   # every node
IPS="10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4" scripts/launch-tp4.sh up   # head first; the release profile
scripts/launch-tp4.sh watch &            # takes all ranks down if one of them dies
scripts/launch-tp4.sh status             # containers, MemAvailable per rank, API
python3 bench/sanity.py all              # chat, effort ladder, tool call, JSON schema, 120K needle
scripts/launch-tp4.sh down
```

`up` checks on every node that the weights and the drafter are the pinned revisions, that no other model server runs,
that MemAvailable is above 100 GiB and that both RDMA ports are ACTIVE. It syncs `overlay/`, `tp4/` and `roce/` to the
same path on every node and refuses to start if their bytes differ, builds the RoCE proxy once per node in a throwaway
CPU container, starts one container per rank with a memory guard (stops that rank below 3.5 GiB MemAvailable), and
finally compares the `INKLING_*`, `B12X_*`, `NCCL_*` and `PYTHONPATH` settings of the four containers: a lever that
differs between ranks makes their collectives diverge and deadlock, so a mismatch takes every rank down. A start
takes ~2-3 minutes with warm caches (~10 minutes the first time: CUDA graphs and overlay extensions).

The API is `http://127.0.0.1:8000/v1` on the head, model `inkling-2-mini`, loopback only; use
`ssh -L 8000:127.0.0.1:8000 <head>` from another machine. `PROFILE=1` adds a per-boot local admin key for SGLang's
`/start_profile` (kept in `~/.cache/inkling-tp4/admin-key`, used by `tools/profile_c1.sh`).

Settings are environment variables: `SPEC` (`dflash2` | `dspark` | `none`), `OVERLAY` (0 = no lever),
`INKLING_TP4_ROCE` (0 = NCCL only), each `INKLING_*` lever (an empty value switches it off), `MEM_FRACTION` (0.80),
`CTX` (262144), `MAX_RUNNING` (48), `SWA_RATIO` (0.05), `MIN_AVAIL_GIB` (3.5), `NCCL_DIR` (empty = the image's NCCL
2.28.3), `IPS`, `SSH_KEY`.

## Reproducing the numbers

```bash
git clone https://github.com/knapcio/speedbench ~/speedbench-6733653 && git -C ~/speedbench-6733653 checkout 6733653
scripts/launch-tp4.sh up && setsid nohup scripts/launch-tp4.sh watch >/dev/null 2>&1 &
bench/tp4screen.sh MYSCREEN              # ~8 min: speedbench quick, decode c1/c4/c16, thinking c8, guard; all-rank memory
bench/fullrun.sh MYRUN                   # ~80 min: bench/full.sh (speedbench full, decode table, thinking c1/c8/c24,
                                         # long context 1K-128K, guard, sanity, needles 120K/240K) + all-rank memory
bench/arm.sh LABEL [flags]               # down -> up -> watch -> 5-minute hang check -> screen, detached-safe
python3 tools/tf_logprobs.py dump A.json # teacher-forced prompt logprobs (prefill numerics); `compare A.json B.json`
python3 tools/gen_logprobs.py dump G.json --conc 8   # greedy output logprobs (decode / verify numerics); `compare`
```

Agent loop (AG64: 24 agents x ~72K-token contexts, two turns; needs `--enable-cache-report` for the API's
`cached_tokens`): `scripts/launch-tp4.sh up --enable-cache-report`, then `python3 bench/ag64.py --out results/AG1`
(`--phases long` skips the 1K reference). It prints per turn the hits / partial hits / whole-context misses
(cached tokens against the agent's previous prompt), TTFT and wall time, and writes `ag64.json`.

CPU tests: `tp4/tests/test_prepatch_cpu.py` (the DFlash2 shard clamp), `roce/tests/`, `tp4/tests/test_sgl_roce_cpu.py`
(RoCE hooks and adapter), `overlay/tests/dfbytes/test_dfbytes_tp.py` (TP hot plane),
`overlay/tests/vcap/test_dflash_cap_tp.py` (the verify cap at TP4: four gloo processes),
`overlay/tests/phoist/test_phoist_tp.py` and `overlay/tests/ptouch/test_ptouch_tp.py` (hoist and cache touch: same
decisions and identical cache trees on four gloo processes), plus the single-Spark recipe's
overlay tests.

## Licences

Repository-authored scripts, overlays, tests and documentation are MIT (see `LICENSE`). The overlays are those of the
single-Spark recipe, with the third-party code listed in `NOTICE`. `roce/` and `tp4/inkling_sgl_roce.py` port the
Apache-2.0 RoCEnante integration (see `NOTICE` and `CREDITS.md`). The serving image (SGLang) is Apache-2.0 and is used
as published, pinned by digest. The Inkling-2-Mini weights are Apache-2.0 from Thinking Machines Lab. The DFlash2
drafter and its SGLang compatibility patch (incoai/Inkling-2-Mini-DFlash2) are Apache-2.0; `scripts/download.sh`
fetches them at a pinned revision, and nothing of them is distributed here.

See [CREDITS.md](CREDITS.md) for the work this builds on.
