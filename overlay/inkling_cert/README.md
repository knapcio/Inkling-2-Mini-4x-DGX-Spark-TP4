# inkling_cert: certified greedy argmax of the target LM head

On greedy verify steps only the argmax of each row's 201,024 logits matters. This package reads a 6-bit copy of
the head (358 MB) instead of pico's coded head (715 MB), bounds every logit rigorously, and runs pico's exact head
GEMV only on the 16-row tiles that can hold the argmax. Every other logit is written as -inf. The stock softcap,
padding mask and argmax then run unchanged and return the same token as the full head, bit for bit.

Status (2026-11-21): written, CPU-tested (21 tests), and GPU-checked on GB10 with real weights:
`tools/gpu_check_cert.py` PASS, `tools/smoke_hooks.py` PASS (all hooks land on the image's sglang, cert outside
pico). Not yet run in a serving boot: the next step is `INKLING_CERT_HEAD=check` for a few thousand steps, then an
A/B against `=0`.

## GPU check (2026-11-21, GB10, real head, 5,983 real LM-head rows from the CPU oracle)

| gate | result |
|---|---|
| 6-bit copy decodes to the quantised head (by the kernel's rule); tables finite | ok |
| screen accumulation error / model (C_ACC ||D|| ||x||), 512 rows x 201,024 ids | max 0.0002 |
| pico accumulation error / model (C_ACC ||W|| ||x||) | max 0.0004 |
| s - B <= pico <= s + B for every logit | 0 outside; max abs(pico - s) / B = 0.13 |
| certified token == full-head token (stock softcap, mask, argmax), M = 8 | 5,976 rows, 0 mismatches, 747/747 steps certified |
| same, M = 16 | 5,968 rows, 0 mismatches, 373/373 certified |
| computed tiles bit-identical to pico's full head, others -inf | 0 bad steps |
| `on` = 0 equals pico's full head bit for bit | ok |
| near-tie rows (x along W_a + W_b, norms 150-800) | 2,048 rows, 0 mismatches |
| duplicate head rows (1 pair in the vocab): exact tie resolves to the lower id | ok |
| CUDA graph replay == eager, 40 steps with `on` toggled | ok |

Candidate tiles per step: M = 8 mean 57.6, p50 29, p90 96, p99 238, max 9,639 (one flat row; still exact, just
slower). M = 16: mean 105, p50 52, p99 433.

Microbench (L2 flushed with a 256 MiB read before each rep, CUDA events, median of 30):

| | M = 8 | M = 16 |
|---|---:|---:|
| pico full head (current `lmhead`) | 2.898 ms (247 GB/s) | 3.036 ms (236 GB/s) |
| certified path, median-candidate batch (18 / 30 tiles) | **1.671 ms** | **1.730 ms** |
| certified path, p90 batch (40 / 61 tiles) | 1.680 ms | 1.730 ms |
| screen alone (screen + rowmax + select) | 1.569 ms = 228 GB/s of 358 MB | 1.618 ms = 221 GB/s |
| `on` = 0 step (full head through the masked kernel) | 2.971 ms | 3.027 ms |

Final kernel (`gpu4.log`). Saving per certified step: -1.23 ms at c1 and -1.31 ms at c2. A persistent grid-stride screen (96-384 CTAs, x
loaded once per CTA) was tried and rejected: 140 GB/s at M = 8 and 202 at M = 16, because the x loads then go
ahead of the weight stream. A step that is not eligible pays about +90 us over
pico.

## Switch

| env | default | meaning |
|---|---|---|
| `INKLING_CERT_HEAD` | unset (off) | `1`: certified head on eligible steps; `check`: serve pico's full head, build the certified result beside it and count token mismatches (must stay 0); `0`: off |
| `INKLING_CERT_HEAD_LOG_EVERY` | 2000 | counter line every N verify steps (0 = never) |
| `INKLING_CERT_HEAD_CACHE` | `~/.cache/inkling-cert` | extension build cache |
| `INKLING_CERT_TP` | unset (off) | `1`: also arm the head at TP > 1, per vocab shard (see "Tensor parallel"); without it the package stays off at TP > 1 |

The package needs `INKLING_PICO12` with `lmhead`, installed first. pico's GEMV is both the exact pass and the
fallback, and the certified token must equal pico's.

Arming in `sitecustomize.py`, after the `INKLING_PICO12` block:

```python
    if _on("INKLING_CERT_HEAD"):  # after inkling_pico12: wraps its lm-head function and reads its coded head
        import inkling_cert
        inkling_cert.install()
```

`_on()` already treats `check` as on, and `install()` reads the mode itself. `scripts/launch.sh` passes the
variable through `SGLANG_ENV` (`-e INKLING_CERT_HEAD=1`); `scripts/launch.sh` sets it from `CERT_HEAD`.

## Measured before building (kill-check)

Real LM-head inputs: 5,983 rows from the CPU oracle (`engine/ref/inkling2_ref.py`, FP32). They come from
teacher-forcing the 16 speedbench greedy transcripts, rounded to BF16. The real unembedding was used. Script:
`results/kbench/round2/cert/cand_count.py`. The head rows have norm 1.00 and ||x|| is about 223. The top logit has a median of
50.7, and the gap to the second logit has a median of 13.6.

| screen | MB read | bound B (median) | candidate tiles / step c1 (mean, p99, max) | c2 (mean, p99) | est. ms |
|---|---:|---:|---|---|---:|
| INT8 per row | 471 | 2.0 | 13, 25, 26 | | 2.06 |
| **INT6, FP16 scale per 192** | **358** | **6.2** | **43, 236, 606** | **78, 328** | **1.59** |
| INT6, FP16 scale per 32 | 382 | 5.0 | 23, 92, 213 | 43, 138 | 1.68 |
| INT7, scale per 192 | 416 | 3.1 | 14, 37, 52 | 26, 57 | 1.82 |
| INT5, scale per 32 | 324 | 10.4 | 305, 1638, 2279 | | 1.52 (heavy tail) |
| INT4, scale per 32 | 290 | 21 | ~10k (everything) | | dead |

The estimate charges screen bytes at 235 GB/s, candidate tiles at 57 KB each at 200 GB/s, plus 50 us of small
kernels. pico's full head is 2.90 ms at 8 rows. The measured screen error is about 1.4 % of the bound, because the
Cauchy-Schwarz bound is far from tight. Only the very peaked logits make it work.

## Per step (M <= 16 verify rows, all in the verify CUDA graph)

| kernel | what |
|---|---|
| `k_screen` (12564 CTAs x 12 warps) | Computes s = x . D^T from the 6-bit copy with BF16 MMAs. Each warp covers one 192-value group and scales its partial by the group's FP16 scale. ||x_g|| is computed from the same registers. Epilogue: B, then lo/hi, then the max of lo and of hi per (row, 16-row tile) over the real vocab. A non-finite value sets a flag. |
| `k_rowmax` | T[m] = max lo, H[m] = max hi |
| `k_select` | Per tile: a mask byte, set when some row has hi >= T - delta in that tile. All tiles are set when the step is not eligible (`on` = 0), a value was non-finite, or a row fails the softcap range test. Also updates the counters. |
| `k_exact` | pico's `gemv16_body<kBF16X, mt 2>` on the masked CTAs. This is the code of pico's coded-head GEMV (the same file, compiled into this extension) and gives its bits. Every other CTA writes -inf. |

Then the stock code continues: `fused_softcap` in place, ids >= `vocab_size` set to -inf, and the folded greedy
accept's argmax.

## Why the token is the same (cert_math.py has the full statement)

1. **Bound.** |pico - <W_v, x>| <= C_ACC ||W_v|| ||x|| and |s - <D_v, x>| <= C_ACC ||D_v|| ||x||, where
   C_ACC = (K/16 * 17 + 32) 2^-23 is the tensor-core accumulation model (an assumption, measured by the GPU check;
   pico12's own check already gates pico at (K/16 + 32) 2^-23). Also |<W_v - D_v, x>| <= sum_g ||R_vg|| ||x_g||
   (Cauchy-Schwarz per group). Every table is rounded up, and every FP32 step is inflated, so lo <= l <= hi.
2. **The winner is computed.** Let w be the stock token and u the id with lo_u = T. If l_w >= l_u, then
   hi_w >= T. Otherwise f(l_w) >= f(l_u) with l_w < l_u, where f is the stock FP32 softcap, which is monotone only
   up to E = 2^-14. Then l_u - l_w <= delta = 2 * 2E / exp(-2L/30) on the range L checked on the device, so
   hi_w >= T - delta.
3. **No -inf beats it.** f(-inf) = -30, and f(l_w) >= 30 tanh(T/30) - E > -30 under the range test (|T|, |H| up
   to about 160; real logits are 50-70).
4. **Ties.** Computed ids hold the stock values. The set of maximisers among them therefore contains w, the lowest
   stock maximiser, and nothing below it. The stock argmax (lowest index) returns w.

x is finite, so every logit is finite and no NaN reaches the argmax. A non-finite screen value forces the full
head anyway.

## When the certified path runs (`on` = 1)

The device flag is written before each verify replay by the `TargetVerifyExecutor.run_non_compact` wrapper. It is
1 only when all of these hold:
- `begin_static_step(bs, armed)` was called with `armed = fold_eligible`. That means a static-mode step whose
  in-graph folded greedy accept is used: all requests greedy, `verify_logits_adjustments_are_noop` (no penalties,
  logit bias, grammar mask or custom processor), and no grammar.
- The batch has no `return_logprob`, no watermark candidates and no sampling-mask capture.
- None of the DSpark debug observers that read the verify logits is set (checked at load; the package disables
  itself if one is).

`begin_step` (compact mode, idle participation) sets it to 0. Eager forwards (no graph) always compute the full
head. The first eager call per row bucket goes through these kernels in full-fallback mode, and later eager calls
go straight to pico. Rows > 16 (c3 and up) never reach here: pico itself takes only 1-16.

### On the DFLASH worker (`dflash.py`, `SPEC=dflash2`, launch `CERT_DFLASH` / `CERT_WIDE_DFLASH`)

DFLASH has no verify epilogue: its accept runs eagerly after the target-verify graph, as the argmax of the verify
logits on greedy batches. A wrapper of `DFlashWorkerV2.forward_batch_generation` marks the step, and a wrapper of
`TpModelWorker.forward_batch_generation` sets `on` right before that step's target verify to
`batch_ok(batch)` and `DFlashWorkerV2._can_skip_verify_logits(batch, sampling_info)` (all greedy, no grammar,
logprobs, logit adjustments or selector sample), and back to 0 right after it, so no other replay reads a stale 1.
- Narrow head (<= 16 rows): it writes pico's exact logits on the candidate tiles and -inf elsewhere, so the stock
  argmax over the verify logits returns the certified token. Under the verify cap (`inkling_vcap` on DFLASH,
  `STATE.src_index`), dead rows are given their anchor's hidden row before the screen, selected on the device by
  `on`, so they add no candidate tiles; live rows keep their own rows and no accept reads a dead row's token.
- Wide head (17..`INKLING_CERT_WIDE_MAX_ROWS` rows): the certified IDs, which the wide head returns instead of
  writing the logits, are looked up after each target-verify replay (`DecodeCudaGraphRunner._process_output_after_replay`,
  from the replayed graph's own output) and passed to `DFlashWorkerV2._accept_block` as its target prediction. A wide
  replay whose IDs cannot be identified stops the server; the compact ragged verify layout is refused.
- Check mode (`CERT_DFLASH=check`): 34,808 rows, 0 mismatches.

## Patch points

| target | what |
|---|---|
| `models/inkling2.py: Inkling2LogitsProcessor._get_logits` | marks target-verify calls (`forward_mode.is_target_verify()`) |
| `models/inkling2.py: Inkling2ForConditionalGeneration.post_load_weights` | after pico coded the head: builds the 6-bit copy and the tables on the GPU (chunked, float64), runs the self-test (full fallback == pico bit for bit, certified argmax == full argmax on random rows at M = 8 and 16; this also loads every kernel before capture), and logs one line |
| `layers/quantization/unquant.py: sm12x_fp32_lm_head` | wraps pico's wrapper. It takes target-verify calls that pico would take, as long as the processor has softcap 30, no `logit_scale`, no TP gather, and `vocab_size` <= rows. Everything else goes to pico or stock. |
| `speculative/dspark_components/dspark_verify.py: DsparkVerifyEpilogue.begin_static_step / begin_step, TargetVerifyExecutor.run_non_compact` | the per-step device flag (above) and the periodic counter line |

Source strings are checked at `install()`, and attributes when each module is imported. A mismatch exits with 78.
TP > 1 disables the package with a log line, unless `INKLING_CERT_TP=1` (next section).

Coexistence:
- `inkling_pico12`: required, and its wrapper sits inside ours.
- `inkling_vcap`: under DSpark, independent: dead rows are still verify rows with real hidden states, and their
  logits are certified like the others. Under DFLASH, see above (dead rows take their anchor's row).
- `inkling_draftbytes`: the draft head is not touched.

## Tensor parallel (`INKLING_CERT_TP=1`, `certtp.py`)

Status (2026-11-25): written and CPU-tested (`tests/test_certtp.py`: the numpy model of 4 shards, `ops.Cert.run_tp`
with 4 ranks in threads over a barrier all-gather, the patched head including a bypassed head dispatch, the
deferral limit and NaN order, the load-time agreement with failing / disagreeing ranks). Review fixes after an
independent review of 250cbfa. Patch
targets confirmed in the image (sha256 of every file read equals our image-source copy). Not yet run on a GPU: next
is `tools/gpu_check_certtp.py`, then a TP4 boot with `INKLING_CERT_HEAD=check`, then an A/B.

At TP the head is vocab-parallel: rank r holds ids [r n, (r + 1) n), n = 50,256 at TP4, and the last shard has 49,290
real ids. The stock step computes every shard with pico, softcaps it, all-gathers the full [M, 201,024] FP32 logits
(0.30 ms median per c1 step on the RoCE all-gather, `results/profile`) and masks ids >= 200,058.

Per captured target-verify head call of 1-16 rows, on every rank:

| kernel / op | what |
|---|---|
| `k_screen` | the TP1 screen on the shard (its real ids only) |
| `k_rowmax_tp` | (T_r, H_r) of the shard into `send[M][2]`; +inf when `on` = 0 or a screen value was non-finite |
| all-gather | `send` over the TP group (`GroupCoordinator.all_gather_into_tensor`, so the RoCE hook takes it): [M, 2] FP32 |
| `k_select_tp` | T = max over ranks of T_r, H = max of H_r (the same bits on every rank); the TP1 range test, delta and tile selection on this shard against them; a failed test (any +inf) computes every tile |
| `k_exact` | unchanged: pico's GEMV on the selected tiles, -inf elsewhere |

Then the stock code continues on the shard. With `SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1` the DFLASH worker keeps the
verify logits as shards: a greedy step (`_can_skip_verify_logits`, the same condition that arms `on`) resolves them
with the stock `deferred_argmax`: the image's `softcap_row_argmax` on each shard and one [M, 2] all-gather, ties to the
lowest rank (lowest id); any other step runs the full gather eagerly (`finish_deferred_gather`). Without the env the
stock in-graph gather publishes the certified logits and the argmax is the same token.

Why the token is the same: the TP1 argument (cert_math.py) uses only lo_u = T for some real id u, l_u <= hi_u <= H and
the range test on (T, H). With T and H taken over every shard, the stock winner's tile is computed on the rank that
holds it, every uncomputed id is -inf (softcap -30) below it, and every computed id holds pico's value, so the lowest
maximiser over the real ids is the stock token, in the gathered row and in the per-rank (max, lowest id) merge. A
rank that computes its whole shard only adds stock values. Each id's interval comes from the rank that serves it, so
the argument does not need the ranks' hidden rows to be bit-identical. certtp_math.py has the full statement.

Why a global T: with each shard's own T the shard without the winner computes most of its tiles (real head, 5,983
real rows, 8-row steps: the slowest shard 2,668 of 3,141 tiles on average); with the global T, 27 on average (p50
15, p99 155, max 309) and 43 summed over the 4 shards, the TP1 count.

Rank invariance (a rank-divergent collective deadlocks the fleet):
- The (T, H) all-gather (and in check mode a second, [M, 4] one) is owed by every captured target-verify call of
  1-16 rows once TP mode is armed. `Inkling2LogitsProcessor._compute_lm_head` decides it, before SGLang picks a head
  kernel: if the certified head did not issue it (our head function never reached, e.g. `sm12x_fp32_lm_head_applies`
  false -> `torch.mm`; not eligible; disabled), it is issued there with +inf (`ops.Cert.skip_tp`), which makes every
  rank fall back for that step. Eager calls never issue it (the first eager call per row bucket loads the kernels
  with a local exchange).
- Arming is agreed at load by `certtp.agree`: one MIN all-reduce over the TP group's CPU group of (prepared ok, head
  mode, deferral setting), reached by every rank whatever its local outcome. An observer flag, a refused layout, a
  failed build or self-test vote no (no early return, no exit); all ranks arm only if all prepared, with the same
  `INKLING_CERT_HEAD` (1 / check: different collective counts) and `SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER`.
- `on` comes from batch metadata (`dflash.py`), the same on every rank; a rank with `on` = 0 sends +inf anyway.
- Whether a step is certified depends only on the gathered (T, H) and `on`: the same on every rank.

The deferral fixes (`certtp.patch_deferral`, source-checked), so that the deferred path returns what the non-deferred
one does:
- the image's `deferred_local_argmax` scans the padded 201,024 ids and `finish_deferred_gather` does not mask ids >=
  200,058 (the Inkling mask lands on the shard, where it is empty). Both are limited to the real vocabulary. Only
  reachable with all-negative real logits for the argmax (padded rows are zero); the CPU test builds such a case.
- NaN: the non-deferred accept is `torch.argmax` over the gathered row (the first NaN wins), while
  `softcap_row_argmax` skips NaNs. The local result reports a row's first NaN as (NaN, id), and `deferred_argmax`
  merges NaN first (lowest rank), then the largest maximum (lowest rank on ties) (`certtp.merge_shards`). Costs a few
  small eager kernels (isnan / any / argmax / where) per greedy step. CPU test: NaNs on any shard, exact ties, padding.

Dead rows under the verify cap: `dflash.dead_rows_substituted()` is true during a DFLASH step whose certified head
gave dead rows their anchor's hidden row. Accepted tokens never depend on it, but AcceptGreedy grades all rows before
`CapCorrectLen` clamps, so `cap_trim_lens` and the block acceptance statistics see the anchor's tokens on dead rows.
`inkling_vcap` should treat dead rows as ungraded there (a change in that package, not made here).

Not at TP: the wide head (17-128 rows, `INKLING_CERT_WIDE`; its reduce returns IDs without values, and its screen
compacts candidates in one launch with no point for the exchange), and DSpark (`on` stays 0; the collectives still
run). TP mode runs only on the DFLASH worker.

Enable (TP4, SPEC=dflash2): `INKLING_CERT_HEAD=1 INKLING_CERT_TP=1 SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1`, with
`INKLING_PICO12` holding `lmhead`. `INKLING_CERT_HEAD=check` serves pico's full shards and counts global-token
mismatches (the log line's `mismatches`, must stay 0).

Expected saving per c1 step (estimate, GB10): the shard head 0.71 ms (pico, profile) to about 0.42 ms (TP1 measured
ratio 0.58 plus about 30 us of small kernels and the [M, 2] gather), and with the deferral the 0.30 ms logits gather
to about 0.04 ms (softcap_row_argmax and an [M, 2] all-gather, eager): about 0.55 ms of 22.1 ms, 2.5 %. A step that is
not eligible pays the screen early-outs and the [M, 2] all-gather (about 20 us).

GPU check: `python3 inkling_cert/tools/gpu_check_certtp.py --model /model --hidden DIR` (one GPU, the real head in 4
shards, the exchange by copies): fallbacks bit for bit, real-row exactness by both resolutions with the same flag on
every shard, near ties and exact ties across shards, one shard off, CUDA graph, bench per shard.

## Cost and memory

- Extra memory is 358 MB (Q 347.4, scales 4.8, residual norms 4.8, row norms 0.8) plus about 1.7 MB of buffers.
- A step that is not eligible (`on` = 0: sampled, logprobs, ...) costs pico's head plus about 20-30 us. The screen
  and row-max kernels return at once, and the masked kernel runs every tile (pico's full head). An eligible step
  that fails the range test pays the screen and the full head; this is never seen on real rows.

## Reviews and logs

- GPU logs (GB10): `results/kbench/round2/cert/gpu2.log` (first PASS), `gpu3.log` (persistent variant, rejected),
  `gpu4.log` (final kernel).

## Files

| path | origin |
|---|---|
| `csrc/cert_ext.cu`, `*.py`, `tests/`, `tools/` | ours (ported in spirit from our GLM-5.3-Flash `glm_cert_head.py` / `cert_math.py`) |
| pico's `gemv16.cu`, `common.cuh`, `dsk.h`, `check.h` | not copied: `#include`d from `../inkling_pico12/third_party/pico` (MIT, Copyright (c) 2026 Andy Shu) |

## Checks

- CPU: `python3 -m unittest discover -s inkling_cert/tests -v` from `overlay/` (needs numpy and torch). It covers
  pack/unpack, the lane/MMA k-mapping of the packed layout, tables (torch vs numpy), bound coverage, adversarial
  screen errors, exact ties, softcap ties, padded ids, range fallback and the fill.
- GPU: `python3 inkling_cert/tools/gpu_check_cert.py --model /model --hidden DIR`, where DIR holds real LM-head
  input rows as `.npz`. It gates the packing, the accumulation model (screen and pico vs FP64), coverage
  (lo <= pico <= hi for every logit), and token identity on every real row at M = 8 and 16. It also gates near-tie
  rows, duplicate head rows, computed tiles bit-identical to pico, `on` = 0 equal to pico, and graph replay equal
  to eager. It then benches.
