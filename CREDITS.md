# Credits

## TP4-specific

- **Luke Alonso (@lukealonso), Jason Cook (@original-el8), Local Inference Lab**: RoCEnante, the b12x one-shot RDMA
  all-reduce / all-gather and RDMA proxy ([b12x #295](https://github.com/local-inference-lab/b12x/pull/295),
  Apache-2.0), and the vLLM integration that `roce/inkling_roce` ports: the capability vote, the fail-stop health check
  ([local-inference-lab/vllm #597](https://github.com/local-inference-lab/vllm/pull/597)). The kernel sums the peer
  slots in a fixed rank order, which keeps replicated activations bit-identical on every rank.
- **FlashInfer** contributors: b12x ships inside FlashInfer 0.7.4 (`flashinfer/experimental/b12x`) in the serving
  image; the RoCE path uses that copy, unmodified.
- **tonyd2wild**: the port of #597 to an upstream-shaped vLLM tree (all-gather switch, the 300M spin limit).
- **rhys101**: the idea of routing SGLang's TP all-reduce to RoCEnante and the result-boundary health check pattern
  ([DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8)).
- **NCCL** 2.30.7 and the RoCE / NCCL settings (both PCIe halves of the 200G port, small pinned buffers, LL128 off)
  carried over from our own DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash TP4 recipes on the same four Sparks.
- **SGLang** project for Inkling-2-Mini's TP / EP support in the pinned image (head-sharded attention, EP4 on Marlin,
  the vocab-parallel head and the TP-aware DFLASH candidate gather).
- **incoai** for the DFlash2 drafter; its compatibility patch refuses TP, and `tp4/tp4_dflash2_prepatch.py` replaces
  that guard with a per-shard vocabulary clamp in a temporary copy (the pinned file is never edited).

## Inherited from the single-Spark recipe

- **Thinking Machines Lab** for Inkling-2-Mini, its NVFP4 / MXFP8 checkpoint and the bundled DSpark draft model
  (`mtp.safetensors`) ([model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini),
  [announcement](https://thinkingmachines.ai/news/inkling-2-mini/), Apache-2.0).
- **incoai**, [Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (Apache-2.0): the
  DFlash2 draft model that is this recipe's default drafter, and its SGLang compatibility patch
  (`sglang_inkling_dflash2.py`), both used as published at revision `4f9088ea`, fetched by `scripts/download.sh` and
  not distributed here. The drafter byte cuts, the DFLASH verify-width cap, the certified head on DFLASH steps and the
  selector fix are ours, on top of it.
- **The DFlash authors** ([z-lab/dflash](https://github.com/z-lab/dflash)): block-diffusion drafting for speculative
  decoding, the method behind the DFlash2 drafter, and SGLang's DFLASH worker that serves it.
- **SGLang** project for the Inkling-2-Mini support, the `lmsysorg/sglang:inkling-2-mini` image and the
  [Inkling-2-Mini cookbook](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini), whose DGX
  Spark DSpark cell (TP1/EP1, FA4, page 128, 8 draft tokens, the four `SGLANG_OPT_*` switches, `MAX_JOBS=4`, the
  `inkling` / `inkling2` parsers) is the base of the default profile.
- **vLLM** project for the Inkling-2-Mini support (vllm-project/vllm #69027, #69028, #69039 by Woosuk Kwon and
  contributors), the `vllm/vllm-openai:inkling-2-mini` image and the
  [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) behind the alternative profile.
- **Tech2Wild (tonyd2wild)**, [Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark):
  the first single-Spark vLLM recipe for this model. Reused here: the vLLM tier settings (`--gpu-memory-utilization 0.80`
  at 262K context with tool calling on), the hybrid KV-pool block model and the FP8-KV long-context tier with its memory
  budget (74 GiB of FP8 KV at 1M), the GB10 page-cache trap, and the advice to keep MemAvailable above ~6 GiB.
- **thefullctx**, [inkling-2-mini-sglang-dgx-spark](https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark) and
  [inkling-2-mini-vllm-dgx-spark](https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark): the first published
  SGLang vs vLLM head-to-head on one Spark (vLLM slightly ahead on short c1 decode, SGLang ~2x on long prefill, faster
  long-context decode and much lower memory use), which set the arms of our comparison.
- **Thien Tran (gau-nernst)**, vLLM PR #69168 (small-M GEMMs for Inkling's looped core): the split-K router /
  conv-projection and controller re-tiling we screened (not adopted, flat in serving).
- **sxuff**, [inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) (Apache-2.0): the vLLM + DSpark
  + INT8 draft recipe and benchmark used for the same-harness comparison, and the `--safetensors-load-strategy eager`
  result (127 -> 45 s) that prompted our one-pass checkpoint load with SGLang's own flag.
- **0xSero**, [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) (MIT):
  the method behind sxuff's eager-load finding.
- **tie-pilot-qxw**, [dspark-markov-walk](https://github.com/tie-pilot-qxw/dspark-markov-walk) (MIT), and
  [vLLM PR #62958](https://github.com/vllm-project/vllm/pull/62958): the one-kernel DSpark Markov walk that
  inspired `rnnhoist`. Our projection split is an independent implementation.
- **nick398** on the [NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/inkling-2-mini/389745): the
  first FP8-KV serving notes for this model (vLLM `--kv-cache-dtype fp8` on an RTX 5090: about half the KV memory per
  token, described there as a quality trade-off).
- **Andy Shu (@blaesus)**, [dsgb10](https://github.com/blaesus/dsgb10) (formerly dsgx10, MIT), as carried in
  [Picom-code/pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4) ("pico"): the exactly coded 12-bit BF16 GEMV
  (`bf16x` format and kernel) that `overlay/inkling_pico12/` builds and calls for the dense projections and the LM
  head; the sources are vendored unmodified in `overlay/inkling_pico12/third_party/pico/` with their licence.
- **Andy Shu's `bf16x` reader** is also copied (MIT, with its notice) into `overlay/inkling_wide/csrc/bf16x_format.cuh`,
  which reads pico12's coded weights in the 17-256-row GEMM, and `overlay/inkling_cert/` compiles pico's head GEMV
  body from the vendored copy for the exact pass of the certified LM head.
- **Tri Dao et al., FlashAttention-2** (algorithm): the online-softmax tiling of the fused relative-bias prefill
  attention in `overlay/inkling_attn/`.
- **SGLang Inkling2 kernels**: the split-KV relative-bias verify kernel that `overlay/inkling_vattn/` routes every batch
  size to (and copies, with the bias computed per key tile, above 32 rows), the split-KV decode kernel that
  `overlay/inkling_kv12/` follows with its K / V tile loads replaced, the split softmax kernels that
  `VCAP_FASTPROBS` calls, the block-verification accept kernel behind `--speculative-use-block-verification`, and the
  fused loop-controller kernels of sgl-project/sglang PR #51926, already in the image.
- **SGLang PR #51681** (DSpark Markov sampler: Philox words that rounded to u == 1) and **SGLang PR #51692** (blockwise
  rejection precision): their fixes are carried verbatim in `overlay/inkling_rsfix/` (`RSFIX`) until the image has
  them. The reject-sampler CDF fix and the DFLASH selector no-match fix in the same package are ours.
- **Sun et al.**, "Block Verification Accelerates Speculative Decoding" (ICLR 2025): the block-verification algorithm
  that sampled requests use (`VCAP_BV`, `DFLASH_BV`). Running it under the verify cap (dead drafts as a stopped draft) and its
  logits-only form in the fused accept are ours.
- **Xiao et al.**, StreamingLLM (2023): the attention sink kept by the optional drafter global-attention window
  (`dwin`, 128 sink tokens by default).
- **Lossless float compression by exponent coding** (ZipNN, Hershcovitch et al., 2024; SplitZip) and **Andy Shu's
  exactly coded 12-bit `bf16x` format**: the inspiration for the packed 12-bit global-layer KV in
  `overlay/inkling_kv12/` (sign + mantissa byte, 4-bit exponent code, escapes). The KV format, packer and attention
  kernel are our own code; nothing is copied from these projects.
- **Alexey Fateev / Project Maya**, [maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) (MIT): launch
  and "bytes in flight" notes on Inkling's small per-visit kernels that framed our L2-warming screen (not adopted, see
  docs/experiments.md).
- **FR-Spec** (Zhao et al., ACL 2025, frequency-ranked draft vocabulary) and SGLang's EAGLE `--speculative-token-map`:
  the idea behind the hot-vocabulary draft. The prompt corpus that ranks the shipped list comes from allenai/WildChat-1M
  (ODC-BY), OpenAssistant/oasst2 (Apache-2.0), NousResearch/hermes-function-calling-v1 (Apache-2.0),
  bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-BY), HuggingFaceH4/ultrachat_200k (MIT),
  ise-uiuc/Magicoder-OSS-Instruct-75K (MIT), openai/gsm8k (MIT), databricks/databricks-dolly-15k (CC-BY-SA-3.0),
  EleutherAI/hendrycks_math (MIT), abisee/cnn_dailymail (Apache-2.0), CohereLabs/aya_dataset (Apache-2.0),
  wikimedia/wikipedia (CC-BY-SA-4.0), juletxara/mgsm (CC-BY-SA-4.0) and llm-jp/oasst2-33k-ja (Apache-2.0); only token
  frequencies of the model's own outputs are shipped, no prompt text.
- **thefullctx**: the KV page-size 16 measurement we screened (not adopted here).
- **Christopher Owen** and B12X (MXFP4-CSF): the scale-code format idea behind `overlay/inkling_scodes/` (our
  implementation, in a modified FlashInfer / CUTLASS mainloop).
- **Christopher Owen, spark-ds41f** (Apache-2.0): the packed-BF16 vocabulary projection (b12x patch 0012) whose format
  we ported to Triton in a screen (not adopted, see docs/experiments.md).
- **FlashInfer** (CUTLASS / CuTe DSL grouped GEMMs, the GEMVs) and **CUTLASS**: the kernels the MoE tile choice
  and the GPU checks run on; since v2.8.4 also the b12x MXFP8 GEMM and quantiser (`mm_mxfp8`, `mxfp8_quantize`)
  that run the drafter's eligible linears (mx8w), used as shipped.
- **NVIDIA** for the DGX Spark platform.
- **speedbench** ([knapcio/speedbench](https://github.com/knapcio/speedbench)), our own benchmark, for the headline
  numbers and the long-context table; its frozen prompt set also feeds the decode table.

Our own earlier recipes contributed ideas reused here: the verify cap with dead-row expert remap (DeepSeek-V4.1
recipe), the batch-aware pay-for-rows rule and the certified greedy LM head (GLM-5.3-Flash recipe), FP8 / low-bit
draft heads (DeepSeek-V4.1), and our spark-tools and speedbench for measurement.

Every lever adopted from someone else's work gets a line here and an entry in the Credit column of the README.
