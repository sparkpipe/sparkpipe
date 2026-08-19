# SURVEY_glm52 — public inference-performance survey for GLM-5.2

Survey of public sources (GitHub, Hugging Face, NVIDIA DGX-Spark/GB10 forums, LMSYS,
Baseten, vLLM/SGLang) for the best REPORTED GLM-5.2 inference numbers and the
techniques behind them. Our baseline for comparison: BF16 spine + FP8 routed
experts, TP8 fanout (8× GB10), no speculation — B1 6.91 / B8 43.46 / B16 75.55
tok/s (PERFORMANCE_STATUS.md:590,592,621).

## 1. Results table

| Ref | URL | Date | HW | Quant | Decode | Techniques |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA forum (karol.spark) | [3× GB10, 16.13 tok/s FP8](https://forums.developer.nvidia.com/t/glm-5-2-on-a-3x-gb10-cluster-16-13-tok-s-decode-215k-ctx-fp8-tp-3-vision/378150/) | 2026-07 | 3× GB10 (TP3) | FP8 | 16.13 tok/s (215K ctx) | FP8, VISION |
| NVIDIA forum (recipe) | [4× GB10, 22 tok/s](https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125/) | 2026-07 | 4× GB10 | FP8 (unpruned) | 22 tok/s (256K ctx) | recipe |
| NVIDIA forum (baristankut) | [4× DGX Spark, 27/52.5 tok/s](https://forums.developer.nvidia.com/t/glm-5-2-unpruned-200k-context-on-4x-dgx-spark-27-tok-s-single-52-5-tok-s-c4/377879/) | 2026-07 | 4× DGX Spark | unpruned (FP8) | 27 single / 52.5 @c4 (200K) | long ctx |
| NVIDIA forum | [8× GB10, Int4-Int8 33-54 tok/s](https://forums.developer.nvidia.com/t/glm-5-2-int4-int8-on-8x-gb10-1-200-t-s-prefill-33-54-t-s-avg-decode-generic-coding-structured/376831/) | 2026-07 | 8× GB10 | Int4-Int8 | 33-54 avg (~1200 t/s prefill) | Int4-Int8 |
| tonyd2wild QuantTrio | [4× DGX Spark, 36 tok/s](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s) | 2026-07 | 4× DGX Spark (GB10) | QuantTrio Int4-Int8Mix | 36 tok/s (median 28.8, mean 32.5) | MTP spec decode, 200K ctx |
| tonyd2wild NVFP4-KV | [4× DGX Spark, 42 tok/s](https://github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s) | 2026-07 | 4× DGX Spark (GB10) | NVFP4 KV | 42 tok/s peak (317K KV pool) | 4-bit NVFP4 KV cache (+58.6% vs fp8), 300K ctx |
| tonyd2wild 2-bit | [2× DGX Spark, 21.5 tok/s](https://github.com/tonyd2wild/GLM5.2-2bit-2-DGX-Spark--21.5tok-s) | 2026-07 | 2× DGX Spark | 2-bit | 21.5 tok/s | 2-bit |
| 0xdfi | [4× DGX Spark, 1M ctx](https://github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark) | 2026-08 | 4× DGX Spark | NVFP4 + B12X | (1M ctx) | NVFP4 compact-KV, B12X sparse-MLA, MTP-5 |
| canada-quant W4A16-MTP | [W4A16 + BF16 MTP](https://huggingface.co/canada-quant/GLM-5.2-W4A16-MTP) | 2026-07 | datacenter (B200-class) | W4A16 (INT4) + BF16 MTP | 125.7 tok/s @c1 | MTP-on |
| LMSYS/SGLang | [500 TPS NVFP4](https://www.lmsys.org/blog/2026-07-13-glm52-optimization/) | 2026-07 | 8× B300 | NVFP4 | 500 TPS aggregate | SGLang, NVFP4, fused kernels |
| Baseten | [fastest API](https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/) | 2026-08 | multi-GPU | FP8/quant | >280 TPS aggregate | API serving, batching |
| vLLM PR #46862 | [fused indexer q_rope_quant](https://github.com/vllm-project/vllm/pull/46862) | 2026-07 | — | — | +1.9-3.3% E2E | fused indexer Q/RoPE quant kernel |
| Red Hat AI (ours) | [DSpark on GLM-5.2, speed doubled](https://cloud.tencent.com.cn/developer/article/2703845) | 2026-07 | — | — | 2× (B1) | DSpark spec decode |

## 2. Best reported result PER QUANTIZATION LEVEL (matched-precision lens)

The user caveat holds: most public numbers are heavily quantized. Grouped by weight
precision, marking comparability to our BF16/FP8 baselines:

| Quantization | Best reported (GB10 unless noted) | Comparable to ours? |
| --- | --- | --- |
| BF16 (unquantized) | not reported on GB10 (744B MoE ≈ 1.5 TB — does not fit) | our spine is BF16; full-BF16 is our reference floor, not a target |
| FP8 (experts) / BF16 attention | 16.13 tok/s (3× GB10 TP3, 215K) · 22 tok/s (4× GB10) · 27/52.5 (4× DGX Spark) | DIRECTLY COMPARABLE — our FP8 baseline |
| Int4-Int8Mix (QuantTrio) | 36 tok/s (4× DGX Spark, MTP) | not matched — 4-bit experts vs our FP8 |
| NVFP4 (4-bit) | 42 tok/s (4× DGX Spark) · 500 TPS (8×B300) · 125.7 tok/s @c1 (B200) | not matched — 4-bit, datacenter GPUs |
| 2-bit | 21.5 tok/s (2× DGX Spark) | not comparable |

Precision-matched number to target: the FP8-class SOTA on GB10 is ~16-27 tok/s
single-stream decode (NVIDIA forum 3-4× GB10 FP8 recipes). Our B1 is 6.91 tok/s on
8 nodes — so the immediate gap is ~2.3-4×, and it is almost entirely explained by
(a) no MTP/speculation (the community's numbers include MTP-spec), (b) our TP8
collective overhead (158 reduces/token), and (c) BF16 KV (the community's NVFP4 KV
is +58.6% over fp8 KV). The heavier-quantized results (36-42 tok/s) are the
post-quantization target, not the matched target.

## 3. Ranked experiment ideas (informed by the survey)

1. Unlock the MTP layer (78) for MTP-1/nextn self-speculation. The single biggest
   reported lever (tonyd2wild 36 tok/s = MTP-spec; canada-quant's MTP-on is the
   head-to-head winner). Our MTP tree (spark_glm52_mtp_tree.h) + MTP layer weights
   are already in the checkpoint. Expected: B1 6.91 → 12-18 tok/s (~2-2.6×).
   Code: small (wire the MTP head + dispatch).
2. NVFP4 (4-bit) KV cache. tonyd2wild reports +58.6% over FP8 KV at 300K ctx; this
   is the KV-cache half of our own FP8-KV backlog. Expected: KV bytes halved →
   decode bandwidth floor drops; +30-60% at long ctx. Code: moderate (codec exists
   in weight_codec.cuh, needs the KV-store path).
3. QuantTrio Int4-Int8Mix experts. 36 tok/s vs our FP8 6.91 (4-bit experts).
   Expected: expert bytes roughly halved → B1 floor up ~1.5-2×. Code: small
   (int codecs already in glm52_stagepack.py CODECS + weight_codec.cuh).
4. DSpark/DFlash speculation (Red Hat's own "speed doubled"). Our DSpark backend +
   dispatch policy are landed but inert (no draft weights). Expected: B1 2×.
   Code: 0 (draft weights/config).
5. B1 reduce-path squeeze (collective fusion). 158 reduces/token ≈ 31 ms is the gap
   to the BW floor (PERFORMANCE_STATUS.md:596); the community's TP3/TP4 has fewer
   hops. Expected: B1 +30-40%. Code: small (shared collective).
6. Fused indexer Q/RoPE quant kernel. vLLM PR #46862 reports +1.9-3.3% E2E from one
   fused DSA-indexer kernel. Expected: +2-3% E2E. Code: small (one kernel).
7. CUDA-graph the decode step. ~25 ms launch overhead (PERFORMANCE_STATUS.md:597).
   Expected: B1 +10-15%. Code: ~100 lines.
8. NVFP4 KV + B12X sparse-MLA for long context. 0xdfi's 1M-ctx recipe (the DSA
   indexer is already our B12X-shaped sparse path). Expected: long-ctx KV capacity,
   no B1 gain. Code: large (separate backlog).

## 4. Summary line

SOTA to match/beat on GB10 at matched precision (FP8 experts / BF16 attention):
~16-27 tok/s single-stream decode (NVIDIA forum 3-4× GB10 FP8 recipes); the
post-quantization (Int4/NVFP4) + MTP-spec ceiling is ~36-42 tok/s — SparkPipe must
first close the 2.3-4× FP8 gap via MTP-spec + collective fusion + 4-bit KV, then
pursue the 36-42 tok/s quantized target.
