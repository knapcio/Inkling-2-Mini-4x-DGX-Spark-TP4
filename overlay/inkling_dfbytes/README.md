# inkling_dfbytes: draft-side byte cuts for the DFlash2 drafter

Import-time overlay for the DFLASH worker (`SPEC=dflash2`). It changes only what the drafter proposes.

| switch | value | effect |
|---|---|---|
| `INKLING_DFLASH_BYTES` | unset / `0` | nothing is patched |
| | `mx8` | MXFP8 copies of the drafter's decoder linears, served at <= 16 rows (c1 / c2) |
| | `hot8` / `hot16` | the selector's top-16 candidates from a hot-vocabulary plane of the target head |
| `INKLING_DFLASH_HOT_N` | default 65536 | size of the hot plane (first N ids of `INKLING_DRAFT_HOTVOCAB_FILE`) |

`scripts/launch.sh` sets `DFLASH_BYTES=mx8,hot8` by default with `SPEC=dflash2` and the overlay on.

## Parts

- `mx8`: after `DFlashDraftModel.load_weights`, every BF16 unquantized linear under `model.layers` gets an MXFP8 copy
  (one E8M0 scale per 32 inputs) and `inkling_draftbytes`' W8 linear method. At <= 16 rows it runs the CUDA
  weight-streaming GEMV; larger batches keep BF16 (the Triton W8 kernel does not beat cuBLAS there).
  `inkling_draftbytes` itself is not armed: only its kernels are configured. The context K/V projections keep their
  BF16 weights.
- `hot8` / `hot16`: `DFlash2DraftModel.compute_candidates` takes the top-16 over a plane holding the target head's
  rows for the hot ids (FP8 row-scaled for `hot8`, 151 MB; BF16 for `hot16`, 302 MB) instead of the full BF16 head
  (922 MB). The plane is built from the target `lm_head` in `DFlashWorkerV2._maybe_build_draft_sampler`, before the
  draft graph is captured; it is skipped under the fused candidate merge and the megakernel head, and a moved head
  (online weight update) is refused.
- `hot8` / `hot16` under TP (vocab-sharded head): the same hot set (first N ids below the real vocabulary), split by
  vocab shard. Each rank keeps the rows of the hot ids inside its own shard `[org_vocab_start, start + num_org)`
  (clamped to the real vocabulary as `tp4/tp4_dflash2_prepatch.py` clamps the stock projection), quantized per row
  exactly as at TP1, takes its local top-16 over them and maps plane rows back to global ids; then the stock tail of
  `candidate_topk` runs unchanged (fused candidate all-gather if the image arms it, else NCCL all-gathers of the fp32
  values and int64 ids, global top-16). The shard ranges of every rank come from the head's configuration
  (`VocabParallelEmbedding._get_indices`), so all decisions are rank-invariant and no collective is added. If any
  shard holds fewer than 16 hot ids, every rank appends 16 pad rows whose logits are set to -inf (sentinel id: the
  smallest hot id), so each rank still sends exactly 16 candidates and no pad row can win. r2 list, 64K, TP4: per
  shard 18,769 / 12,739 / 9,438 / 24,590 ids = 43.3 / 29.4 / 21.8 / 56.8 MB FP8 against 231.6 / 231.6 / 231.6 / 227.1
  MB of BF16 head shard. CPU check: `tests/dfbytes/test_dfbytes_tp.py` (4 simulated shards against TP1, incl. pad).

## Exactness

Draft-only. The selector's q is computed from the candidates the path is drawn from, so the sampled verification
keeps the target law; greedy output is the target argmax chain. Proposals and acceptance change and were measured.

## Measured (GB10, 2026-11-24)

- `tools/bench_dfbytes.py`, cold L2 in a CUDA graph: the 20 drafter linears at 8 rows 3.17 -> 1.99 ms; the candidate
  projection at 7 / 56 / 168 rows 4.02 / 4.39 / 5.15 ms (stock) -> 0.68 / 0.77 / 1.40 ms (64K `hot8`). On 5,983
  captured target rows, 64K `hot8` holds 96.5 % of the stock top-16 and the stock top-1 in its top-16 on 99.6 %.
- Serving A/B against the stock drafter: speedbench quick +8.4 %, prose c1 +6.5 %, code c1 +4.4 %, prose c8 +6 %;
  greedy hashes equal except one known near-tie prompt.
