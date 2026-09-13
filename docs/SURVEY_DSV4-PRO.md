# DeepSeek V4 Pro — inference performance survey (public sources)

Survey of publicly reported inference performance for **deepseek-ai/DeepSeek-V4-Pro**
(1.6T total / ~49.6B active MoE, 61 layers, 384 routed experts, top-6, DSA sparse
attention + mHC compression, DSpark speculative stage). Compiled 2026-08-17 by the
dsv4-pro agent; host measurement is deferred (band unavailable), so this is
survey + plan only. All external figures are as-reported by the source; hardware
and quantization differ widely and are called out per row.

**Reading guide — the one caveat that governs this whole survey:** almost every
public headline number is (a) heavily quantized (NVFP4/FP4), (b) on datacenter
GPUs with 10-30x the memory bandwidth of our GB10 ring, and/or (c) high-concurrency
*aggregate* throughput, not B1 single-stream decode. Only two published numbers
are at or near our shipped precision (native MXFP4-E2M1 + FP8 + BF16): the
canada-quant "upstream MXFP4" B1 = 110.8 tok/s (8x B300), and the on-GB10
DSpark-Flash B1 ~= 50 tok/s (2x DGX Spark).

---

## 1. Results table

B1/B8/B16 = single-stream / batch-8 / batch-16 decode where reported; "c=N" is
concurrency (aggregate output tok/s). "MTP" = vLLM speculative MTP n=1; "DSpark"
= the GA 5-draft semi-autoregressive drafter (gamma=5).

| Reference | Source / date | Hardware | TP | Quantization | B1 tok/s | B8 / B16 (or c=N) | Key techniques |
|---|---|---|---|---|---|---|---|
| canada-quant "upstream MXFP4" (native checkpoint) | [HF canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) ~2026-05 | 8x B300 SXM6 | TP8+EP | native MXFP4-E2M1 trunk experts (g32) + FP8 attn/shared + FP8 KV | **110.8** | c16=491.4, c64=1699.2, c128=2806.7 | MTP n=1 + cuda graphs, deep_gemm mega-moe, expert-parallel |
| canada-quant NVFP4 artifact | [same repo](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) ~2026-05 | 8x B300 SXM6 | TP8+EP | NVFP4 trunk experts (g16, E4M3 scales) + FP8 attn/shared + FP8 KV | **139.3** | c16=672.6, c64=1927.3, c128=3004.8 | flashinfer_trtllm MoE backend (+25.7% B1 vs MXFP4), MTP n=1 + cuda graphs |
| Together AI (AA #1 output speed) | [ai-primer: Together AA latency/speed](https://www.ai-primer.com/engineer/stories/together-ai-ranks-deepseek-v4-pro-artificial-analysis-speed-latency) 2026-06-14 | HGX B200 | TP8+EP (n/a disclosed) | native MXFP4 + FP8 (assumed, not disclosed) | **171.3** | Lightning 147.5, Fireworks 111.9 | SWA-aware KV policy (1.2M->3.7M tok), CSA/HCA/SWA prefix reuse, endpoint profiles |
| AI Primer DSpark rollout | [ai-primer ~90 tok/s DSpark](https://www.ai-primer.com/engineer/stories/deepseek-v4-pro-benchmarks-90-tps-dspark) 2026-06-27 | (hosted, n/a) | n/a | mixed (n/a) | **~90** | one run 214s -> 116s (~1.84x) | DSpark speculative decode |
| drowzeys DSpark concurrency patch (FLASH model) | [GitHub drowzeys/Keys-Concurrency-Patch](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash) 2026-06-29 | **2x DGX Spark (GB10 / SM121)** | TP2 | MXFP4 experts + FP8 KV (kv-cache-dtype fp8) | **~50-54** | c4=122, c8=183, c16=290 (agg) | DSpark gamma=5, acceptance ~0.55-0.60, stable-slot + ragged-context patch |
| drowzeys 2-stack | [same repo](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash) 2026-06-29 | 4x DGX Spark | 2x TP2 | same | ~50 (per stack) | c32=375 (agg) | replica scale ~1.96x |
| InferenceX MI355X first-light (FP8) | [InferenceX/SemiAnalysis MI355X SGLang 110.5x](https://inferencex.semianalysis.com/blog/mi355x-deepseek-v4-pro-sglang-110x-in-26-days) 2026-05-26 | 8x MI355X | TP8 | FP8 (all) | n/a (agg) | c8=20.4 tok/s/GPU (2.43/user, TPOT 411ms) | launch FP8 path, torch FlashMLA fallback |
| InferenceX MI355X final (FP4) | [same article](https://inferencex.semianalysis.com/blog/mi355x-deepseek-v4-pro-sglang-110x-in-26-days) 2026-05-26 | 8x MI355X | TP8 | FP4 experts + FP8 attn | n/a (agg) | **2256 tok/s/GPU @ 9.4 tok/s/user** (8K/1K) | FP4 enable, TileLang indexer, Triton sparse MLA, fused RoPE/Hadamard, FlyDSL MoE, fused hash topk, AITER MHC, compressor fusions, num-continuous-decode-steps 4->8 |
| DSpark vs MTP-1 (paper) | [arXiv 2607.05147 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2607.05147) 2026-06 | (DeepSeek serving) | n/a | mixed | — | **+60-85% per-user generation over MTP-1** | semi-autoregressive draft (parallel + sequential module), confidence-scheduled verification |
| **SparkPipe Flash TP4 B1 (our own, local)** | qualification/dsv4/performance/tp4_b1_20260814/vllm-dsv4-b12x-tp4-b1.json 2026-08-14 | 4x GB10 ring | TP4 | MXFP4 + FP8 + BF16 KV, **no speculation** | **37.79** median decode | — | Flash h4096/l43/e256, 128-token decode |

### Key DSpark facts (the spec behind the numbers)
- DSpark ships **5 drafts** (block/gamma=5) per main step; Pro GA = 3 draft layers
  at taps {58,59,60}, markov rank 512, noise token 128799
  (our `model_contracts/dsv4_pro_authoritative.json:32-42`).
- Public "MTP" numbers (canada-quant) are **MTP n=1** (1 draft token), not the
  5-draft DSpark block; DSpark's reported 60-85% over MTP-1 is the relevant
  speculation uplift for our GA path.

---

## 2. Best reported result per quantization level (and what is comparable)

| Quantization level | Best reported | Hardware | Comparable to our BF16/FP8 baseline? |
|---|---|---|---|
| **NVFP4 trunk (g16) + FP8 + FP8 KV + MTP n=1** | 139.3 B1 / 3004.8 c128 | 8x B300 | **No** — NVFP4 group=16 is a *different, more aggressive* 4-bit scheme than our MXFP4-E2M1 group=32 |
| **Native MXFP4-E2M1 trunk + FP8 + FP8 KV + MTP n=1** | **110.8 B1** | 8x B300 | **YES — this is the precision-matched SOTA.** Same trunk expert codec (MXFP4-E2M1) and non-expert FP8 as our shipped pack; their FP8 KV + FP8 MoE activations are *at most equal* precision to our BF16 KV + BF16 activations (ours is slightly higher-fidelity, slightly slower) |
| **API single-stream (unknown codec, likely native MXFP4+FP8+MTP)** | 171.3 B1 | HGX B200 | **Partially** — codec undisclosed; treat as an upper bound on the native-codec serving ceiling |
| **FP8 (all weights, first-light)** | 20.4 tok/s/GPU @ c8 (aggregate) | 8x MI355X | **No** — ROCm launch path, aggregate not B1; this is the "accurate-slow" floor, not a target |
| **FP4 trunk + FP8 attn (mature)** | 2256 tok/s/GPU @ 9.4 tok/s/user (aggregate) | 8x MI355X | **No** — FP4 is below our precision; aggregate throughput at high concurrency |
| **On-GB10 DSpark (Flash, gamma=5)** | ~50-54 B1 | 2x DGX Spark | **Yes for hardware, No for model** — same GB10/SM121 silicon and same MXFP4+FP8(+FP8 KV) codec, but the *Flash* (h4096/l43/e256) model, ~2.2x smaller weight-per-token than Pro |

**Precision-matched number we should target.** There is no public *Pro-on-GB10*
number — that is exactly what we are building. The two anchors at our precision:

1. **Datacenter matched-precision SOTA: ~110 tok/s B1** (native MXFP4-E2M1 +
   FP8 + MTP n=1, 8x B300, canada-quant). We cannot match the *absolute* number
   on GB10 (B300 HBM3e is ~30x our measured 250-273 GB/s DRAM stream), but this
   is the per-token regime a matched-precision Pro driver should express.
2. **On-hardware matched-codec reference: ~50 tok/s B1** (DSpark gamma=5 on 2x
   GB10, *Flash* model). Scale that by Pro's ~2.2x weight-per-token and our
   16-rank TP4xPP4 interconnect, and our own estimate lands at **~45-60 tok/s
   B1 with DSpark gamma=5** (`tools/devcycle/dsv4_pro_performance_estimate.md:42-47`).

**So the precision-matched target to defend is ~45-60 tok/s B1 on the 16-spark
ring at MXFP4-E2M1 + FP8 + BF16 — i.e. match the on-GB10 DSpark regime, and
express the paper's +60-85% over MTP-1.** Anything below ~12-13 tok/s
(main-model-only, no draft) is "accurate-slow"; 45-60 is "match/exceed SOTA"
at matched precision on our silicon.

---

## 3. Ranked experiment ideas (informed by the survey)

### 1. Land the DSpark gamma=5 native pass (the one that buys the level)
- **Expected gain:** 12-13 -> ~45-60 tok/s B1 (2.5-4x); this is the paper's
  +60-85% over MTP-1 realized, and the GA checkpoint's design.
- **Why the survey supports it:** the only GB10 decode number that exists
  (drowzeys ~50 tok/s B1) *requires* DSpark gamma=5 with ~0.55-0.60 acceptance;
  every datacenter number uses some speculation. We have none landed
  (`spark_dsv4_resident_decode_stage_module.c:47-48`).
- **Code-size:** +900..1400 (kernels + module chain + acceptance + tests).
- **First step:** pin the still-zero draft shapes (heads/intermediate) in the
  contract, then file the kernel cards (`docs/KERNEL_CONTRACT_CARDS.md:27-49`).

### 2. Real FP8 KV cache bytes (not the quant-error sim)
- **Expected gain:** 1.75x KV capacity; ~neutral B1 decode (weight-bound), but
  unlocks 1M-token context and matches the reference act_quant layout.
- **Why the survey supports it:** every public recipe runs `kv-cache-dtype fp8`
  (drowzeys GB10, canada-quant B300, and our own API fingerprint
  `fp8_kvcache`); the reference itself quantizes KV to E4M3.
- **Code-size:** +150..250 (2 write sites + 3 read sites).
- **First step:** build the `kv_fp8` variant and gate the val4 slice
  (`tools/devcycle/dsv4_pro_kv_codec_plan.md:49-53`).

### 3. Confidence-scheduled verification (DSpark's adaptive verify length)
- **Expected gain:** reduced verification waste -> higher *effective* tokens/step
  under concurrency; the paper frames this as the load-aware half of DSpark.
- **Why:** our acceptance currently verifies a fixed 5-draft block; the paper
  tails verification length to per-request prefix-survival probability, which
  is what keeps throughput from collapsing at high concurrency.
- **Code-size:** +150..250 (module acceptance policy + client-side policy).
- **First step:** add a per-request verify-length cap driven by the confidence
  head we already load (`..._module.c:1074`), behind the existing acceptance gate.

### 4. Multi-token decode chains (num-continuous-decode-steps)
- **Expected gain:** +20-40% B1 control/collective overhead (halves per-token
  TCP + lets collectives cover 2 rows).
- **Why:** InferenceX measured +4.7% from 4->8 continuous decode steps; our
  module already validates `chain_step_count > 1` (`dsv4_pro_performance_plan.md:28-34`).
- **Code-size:** +50..150 (config + module chain path).
- **First step:** flip the P5 pacing knobs (config-only), then gate a 2-token
  chain behind a token-stream hash.

### 5. Fused RoPE/Hadamard + mHC pre/post on the *draft* path
- **Expected gain:** small per-step decode win + draft correctness vs the
  reference (our draft currently runs "no rotary", an approximation).
- **Why:** the MI355X 110.5x was ~half kernel fusions (#24727 RoPE/Hadamard,
  #24355/#26014 mHC pre/post); we already fuse query_rms_rope / hc_residual /
  indexer_post on the main path and should carry those into the 3 draft layers.
- **Code-size:** +80..150.
- **First step:** extend the existing fused kernels to the mtp.0..2 forward.

### 6. Expert-weight backend probe (MXFP4 g32 vs NVFP4 g16 / alternate MoE backend)
- **Expected gain:** ~0 on GB10 (we are DRAM *data*-bandwidth-bound, not
  scale-bound; canada-quant's +25.7% was a flashinfer-vs-deep_gemm kernel swap
  on B300, not a format win).
- **Why:** confirms our codec direction — the survey shows FP4 -> throughput,
  FP8 -> first-light-slow; keep MXFP4 as the throughput codec, do NOT chase FP8
  experts for speed.
- **Code-size:** +~100 (a pack variant + kernel variant probe), then decide.
- **First step:** A/B the existing SM121 MXFP4 fused-expert kernel against an
  NVFP4-g16 layout on a single layer, measure bytes streamed vs tok/s.

### 7. Concurrency-scaling validation + slot/dispatch tuning
- **Expected gain:** aggregate throughput ~3-5x at c16 (no B1 change).
- **Why:** drowzeys shows GB10 Flash DSpark goes 50 -> 290 tok/s c1->c16; our
  resident stage is 4-slot. Validate we capture the same curve and whether
  raising slot count / dispatch pacing helps.
- **Code-size:** ~0 (config + receipts); possibly +20-40 for slot bump.
- **First step:** once item 1 lands, sweep concurrency 1/4/8/16 and record
  per-stage service time (--profile-stages already wired).

### 8. cuda-graph / rolling-program robustness on the draft path
- **Expected gain:** correctness/robustness (not speed) — de-risk the F1
  graph-prewarm footgun on the draft path.
- **Why:** drowzeys runs ragged/mixed prefill+decode steps *eager* and only
  graphs the uniform decode-only path; our draft attention uses dynamic shapes
  and must take the same eager fallback.
- **Code-size:** +50..100 (eager-fallback guard).
- **First step:** add a dynamic-shape guard so the 5-draft attention launch is
  never graph-captured.

---

## 4. Summary line (the SOTA to match/beat)

> On GB10 at matched precision (MXFP4-E2M1 trunk experts + FP8 non-expert + BF16
> activations), the reference to match/beat is the on-hardware **~50 tok/s B1
> (DSpark gamma=5, 2x DGX Spark, Flash)** and the datacenter precision-matched
> **~110 tok/s B1 (native MXFP4 + MTP n=1, 8x B300)**; SparkPipe's GB10 Pro
> target is **~45-60 tok/s B1 with the GA 5-draft DSpark block** — match the
> GB10 DSpark regime and deliver the paper's **+60-85% over MTP-1**.

---

## Sources
- canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP — https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP
- DSpark paper (arXiv 2607.05147) — https://ar5iv.labs.arxiv.org/html/2607.05147
- MarkTechPost DSpark 60-85% over MTP-1 — https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/
- InferenceX/SemiAnalysis MI355X SGLang 110.5x — https://inferencex.semianalysis.com/blog/mi355x-deepseek-v4-pro-sglang-110x-in-26-days
- ai-primer: Together AI #1 on AA speed/latency — https://www.ai-primer.com/engineer/stories/together-ai-ranks-deepseek-v4-pro-artificial-analysis-speed-latency
- ai-primer: ~90 tok/s after DSpark — https://www.ai-primer.com/engineer/stories/deepseek-v4-pro-benchmarks-90-tps-dspark
- drowzeys DSpark concurrency patch (GB10) — https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- vLLM recipe (H200) — https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro?hardware=h200
- AMD MI355X DeepSeek inference — https://www.amd.com/en/developer/resources/technical-articles/2026/amd-instinct-mi355x-gpu-sets-a-new-bar-for-deepseek-inference.html
- Local anchors: qualification/dsv4/performance/tp4_b1_20260814/vllm-dsv4-b12x-tp4-b1.json; tools/devcycle/dsv4_pro_performance_estimate.md; model_contracts/dsv4_pro_authoritative.json
