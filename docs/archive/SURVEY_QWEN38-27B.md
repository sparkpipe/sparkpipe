# Qwen 3.8 27B — public inference-performance survey

Lane: qwen38-27b. Survey of public sources (GitHub / Hugging Face / NVIDIA
forums / X.com) for the best REPORTED inference performance of the model and
the techniques behind it, focused on small-batch decode (B1/B8/B16),
speculation/MTP, and quantization. Hardware-matched numbers (DGX Spark /
GB10 / SM121 = our spark) are the comparison basis; other GPUs are listed for
technique only. Verified by fetching the cited sources on 2026-08-17.

**The model is `Qwen/Qwen3.8-27B`, a DENSE 27B hybrid** — 48 GatedDeltaNet
layers + 16 full-attention layers, untied embeddings (0xBakeer README: a dense
27B with hybrid attention, 48 GatedDeltaNet linear-attention layers and 16
full-attention layers). This is the 3.8 revision of the 27B we run; its
GDN/attention geometry is the 3.6 shape (hidden 5120 / 64 layers / 16
full-attn), re-checkpointed.

---

## 1. Results table

Single-GB10 (DGX Spark) rows are **hardware-matched** to our spark3. All tok/s
are single-stream decode (B1) unless marked aggregate.

| Ref (short) | Source URL | Date | HW | TP | Quant | B1 no-spec | B1 spec | Key techniques |
|---|---|---|---|---|---|---|---|---|
| tonyd2wild baseline | github.com/tonyd2wild/Qwen3.8-27B-NVFP4-DGX-Spark/blob/main/benchmarks/BASELINE.md | 2026-08 | DGX Spark GB10/SM121 | 1 | NVFP4 | **11.07** | — | vLLM, CUDA graphs, no eager |
| tonyd2wild exp001 | same repo | 2026-08-14 | DGX Spark | 1 | NVFP4 | — | **18.3** (MTP n=3) | MTP spec, 1.65x |
| 0xBakeer FP8 | github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark (RESULTS.md) | 2026-08 | DGX Spark GB10/SM121, 273 GB/s | 1 | FP8 | **7.88** | MTP k3 17.7-21.3; MTP k8 18.6-32.2; **DSpark k7 20.05-46.8; DSpark k14 58.5** | vLLM, MTP & DSpark spec, prefix cache 14-22x prefill |
| 0xBakeer DSpark c8 | same repo | 2026-08 | DGX Spark | 1 | FP8 | — | 208.7 (c8) / 256.1 (c16) | DSpark k7 aggregate |
| NVIDIA forum 380257 | forums.developer.nvidia.com/t/380257 | 2026-08 | DGX Spark | 1 | NVFP4 | — | 34-38 | SGLang + NVFP4 + DSpark |
| 0xBakeer tuned | weibo.com/2/detail/5332816097445649 | 2026-08 | DGX Spark | 1 | FP8 | — | 75 single / 256 (16-way) | DSpark k-tuned |

Different hardware — technique reference only (NOT comparable to our GB10):

| Ref | Source | HW (bandwidth) | Quant | B1 no-spec | B1 spec | Techniques |
|---|---|---|---|---|---|---|
| HF blog hexgridcloud | huggingface.co/blog/hexgridcloud/qwen3-6-27b-fp8-on-one-rtx-6000-ada-fast-ttft-668 | RTX 6000 Ada 48GB (~960 GB/s) | FP8 | — | 314 (aggregate) | vLLM FP8 |
| lastloop vllm-blackwell-guide | github.com/lastloop-ai/vllm-blackwell-guide | RTX PRO 6000 96GB (1792 GB/s) | INT4 / FP8 | 24 eager, 75 graphs | 100 (MTP n=3, INT4); 125 (MTP n=5); 170 (35B-A3B) | CUDA graphs + flashinfer + MTP, accept 0.87/0.72/0.60 |
| syv-ai rtx3090 | github.com/syv-ai/qwen38-27b-rtx3090 | RTX 3090 24GB (~936 GB/s) | W4A16 | — | 82 (MTP) / 417 (c64) | MTP spec, requantized embed+head |
| apideed/mtplx | github.com/apideed/mtplx | Apple Silicon | Q4 | — | 2.24x (MTP) | native MTP, no external drafter |
| ollama #17776 | github.com/ollama/ollama/issues/17776 | Apple Silicon | — | — | MTP **2x SLOWER** | naive MTP anti-pattern (per-draft lm_head) |

## 2. Best reported result per quantization level (single GB10, B1)

| Quantization | Best no-spec | Best with spec | Comparable to our BF16/FP8 baseline? |
|---|---|---|---|
| **BF16** (our baseline) | ~5.0 (bandwidth floor: 54.5 GB / 273 GB/s) | n/a (our spec is broken) | **YES — this is us** |
| **FP8** | **7.88** | **20.05 (DSpark k7) / 58.5 (DSpark k14)** | **YES — FP8 is our next rung** |
| NVFP4 (4-bit) | 11.07 | 18.3 (MTP n=3) / 34-38 (DSpark) | NO — 4-bit weights, quality loss |
| INT4 / W4A16 | — | 82 (RTX 3090, different HW) | NO — 4-bit + different HW |

**The precision-matched target to beat on one GB10 (B1, no speculation):**
`7.88 tok/s` at FP8, and `~5.0 tok/s` is the BF16 weight-stream floor. The
~20%-behind-SOTA gap the user reports is most consistent with comparing our
BF16 single-spark decode against the FP8 SOTA (7.88): at BF16 the physical
floor is ~5.0 tok/s, so **no BF16-only tuning closes a 20% gap — only
output-preserving speculation (MTP/DSpark) or a precision rung (FP8) does.**
This is the single most important survey conclusion.

## 3. Ranked experiment ideas (survey-informed)

1. **Fix, then replace, the broken MTP speculation.** Root cause is in the code
   (see HILLCLIMB doc): the build gate never exercises the MTP path and a
   chain-dead gate silently zeroes speculation. Fix the gate (~5 lines), then
   adopt the shared verifier/tree (~-60 lines). Expected: restore ~1.6-2.4x on
   B1 (SOTA MTP k3 = 2.4x over no-spec).
2. **Adopt DSpark (separate 5-layer drafter) over MTP.** Survey: MTP cost/
   draft-token 0.153 vs DSpark 0.046 (3.3x cheaper); DSpark k7 beats every MTP
   row and reaches 58.5 tok/s FP8 single-stream. Our in-checkpoint MTP re-runs
   a full lm_head argmax per draft (`spark_qwen38_27b_resident_decode_stage_module.c:1556-1576`),
   the exact anti-pattern the survey flags. New drafter ~+200 lines + contract
   cards; the highest ceiling.
3. **Land FP8 weights (vendor `Qwen/Qwen3.8-27B-FP8`, 28.5 GiB).** Halves the
   weight stream vs BF16 (54.5 GB), moving the single-spark floor from ~5.0 to
   ~9.5 tok/s and directly targeting the 7.88 SOTA. This is the 3.8-27B re-base
   from TOP10 #1/#6. +~150 lines (packer codec + kernel path).
4. **Enable prefix caching.** vLLM's default is OFF for hybrid models; on shared
   prefixes it is worth **14-22x on prefill** and costs nothing on decode
   (0xBakeer section 2). Our DRIVER_OWNS_KV path ignores page budgets today
   (PROPOSAL_KV_SEAM.md section 3.5); adopt the shared KV table. ~-50 lines.
5. **CUDA-graph the steady-state decode (no `--enforce-eager`).** The survey's
   24-to-75 tok/s (lastloop) and the ~55% eager-cost note map to our ~1000 eager
   launches/step (TOP10 #9). Expected: close launch overhead toward the weight
   floor. ~0 net lines (graph capture), high risk.
6. **Lossless BF16 weight codec + decompress-in-GEMM.** Our measured 1.52x
   entropy headroom (QWEN38_27B_TP4_PERF.md:35-36) is the output-preserving analogue
   of quantization — +25-30% at BF16 with zero quality loss. +~200 lines
   (CUDA-KERNELS card).
7. **Screen the B1 head (full-vocab argmax) + fuse GDN small kernels.** The last
   ~1-3 ms to the practical ceiling (QWEN38_27B_TP4_PERF.md:25-28). The FP8 SOTA's
   7.88-vs-9.5-floor shows the same ~17% overhead we carry; these trims are the
   same shape. ~0 net lines.
8. **Tune speculation depth k per workload.** SOTA: k=3 best on fresh generation,
   k=14 best on edit but -43% at c8. Our fixed D=2 leaves the adaptive-depth win
   on the table; make draft count a runtime program parameter
   (SPECULATION_AUDIT.md step 4). ~+40 lines.

## 4. Summary line

**SOTA to match/beat on one GB10 at matched precision: 7.88 tok/s single-stream
B1 (FP8, no speculation), 20-58 tok/s with DSpark speculation — and at BF16 the
ceiling is the ~5.0 tok/s weight-stream floor, so the gap closes via speculation
or FP8, not BF16 tuning.**

---
Sources: 0xBakeer FP8 RESULTS.md + README.md, tonyd2wild BASELINE.md, lastloop
vllm-blackwell-guide, syv-ai rtx3090, NVIDIA forum 380257, ollama issue 17776
(all fetched 2026-08-17; URLs in the tables above).
