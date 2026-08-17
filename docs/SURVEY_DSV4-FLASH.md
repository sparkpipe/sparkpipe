# Survey — DeepSeek V4 Flash inference performance (dsv4-flash lane)

Proposal/docs only. No commits/pushes. Fleet note: the TP4 band is wedged
(8 hosts), so host measurement is deferred; this doc is the survey + a
measurement plan for when the band returns. Public sources: GitHub, Hugging
Face, NVIDIA forums. Every number is as-reported by its author; "matched" is
about *weight precision*, and KV precision is flagged separately because the
community aggressively quantizes KV.

---

## 1. Results table

| Reference + source | Date | Hardware | TP/PP | Quantization (weights / KV) | B1 / B8 / B16 tok/s | Key techniques |
| --- | --- | --- | --- | --- | --- | --- |
| [joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark](https://github.com/joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark) | 2026-08 | 4× DGX Spark (GB10) | TP4 | official FP8/MXFP4 weights · **NVFP4 KV** (`nvfp4_ds_mla`) | **123.13** single-stream; k=8→115.2, k=10→102.6; c=4 per-stream 60.64; real agentic 57.9 | DSpark **k=7 + probabilistic draft** (34.3% vs 26.5% greedy), CUDA graphs FULL_DECODE_ONLY, capture=96, flashinfer b12x MoE + autotune, prefix cache, chunked prefill |
| [Verel-lab, NVIDIA forum](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-4x-dgx-spark-via-vllm-jasl-fork-tp-4-rdma-mtp-49-54-tok-s-single-stream-full-recipe-the-traps/373808) | 2026-06-18 | 4× DGX Spark (GB10) | TP4+EP | official FP8 · FP8 KV | **49.4** single-stream (54.4 peak reasoning); conc8 180 aggregate, 207 peak | **MTP n=2** (not DSpark), jasl vLLM fork + sm12x deep-gemm fallbacks, NCCL 2.30.4 (2.28.9 wedges) |
| [jeffery2011.jc, NVIDIA forum](https://forums.developer.nvidia.com/t/4-node-dgx-spark-cluster-with-deepseek-v4-flash-0731-dspark-benchmark-prefill-2-500-t-s-decode-90-t-s/378878) | 2026-08-01 | 4× ASUS/DGX Spark | TP4 | official FP8/MXFP4 (KV not stated) | **~90** decode (C=1); C=6 150K → 40.4/req; prefill ~2500, 193k effective on KV hit | DSpark; QRS812 fabric |
| [tonyd2wild/DeepSeek-v4-Flash-DSpark-60-tok-s-900K-ctx-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-60-tok-s-900K-ctx-2x-DGX-Spark) | 2026-08 | 2× DGX Spark (GB10) | TP2 | official FP8 · FP8 KV | **62.48** single-stream (acceptance 0.673, 3.36 tok/draft) | DSpark, MTP_NUM_TOKENS=5, fp8 KV pool ~962k tok |
| [tonyd615, NVIDIA forum](https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846) | ~2026-08 | 2× DGX Spark (GB10) | TP2 | official FP8 | **~60–67** single-stream + 1M ctx | DSpark, 1M context |
| [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx) | 2026-08-13 | 4× CMP 170HX (sm_80, **not GB10**) | **PP4** (TP1) | **MXFP4 experts + FP8 e4m3** (original ckpt) | plain 50.8 → **98.1 DSpark** (1.93×); code-gen 118.8 (2.16×); agg@64 = 712.8 | **DSpark under pipeline parallel** (vLLM does not support upstream), content-dependent acceptance (mean 3.03 of 6) |
| [canada-quant/DeepSeek-V4-Flash-W4A16-FP8](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) | 2026-05-26 | 2× DGX Spark / 2× RTX PRO 6000 | TP2 | **W4A16 INT4 experts + FP8 block attention** (heavily quantized) | 14–17 (DGX Spark) / 47–48 (RTX PRO 6000) bs=1; MTP successor 1.49× | W4A16 Marlin + FP8 block 128×128; MTP dropped at load (retained in [W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP)) |
| [hikarioyama/dsv4-flash-nvfp4-sm120](https://github.com/hikarioyama/dsv4-flash-nvfp4-sm120) | 2026-07 | 2× RTX PRO 6000 (SM120, **not GB10**) | TP2 | NVFP4 (full) | MTP **+38%** single-stream | MTP; fix: MTP draft experts are MXFP4, route to Mxfp4MoEMethod |
| [RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) | 2026 | B100/B200 (SM10.x) | — | NVFP4 experts + FP8 attention | (datacenter reference) | original mixed-precision topology |
| **SparkPipe baseline (ours)** | 2026-08-14 | 4× DGX Spark (GB10) | TP4 | **MXFP4 experts + FP8 linears + BF16 KV** | **33.55** no-spec (`PERFORMANCE_STATUS.md:322-355`); branch 38.18/40.46; target 50 | no speculation; 130 serial collectives/token floor |

---

## 2. Best reported result PER quantization level (matched-precision view)

Our weights baseline is the **official checkpoint** exactly:
`MXFP4_E2M1 experts + FP8_E4M3 linears` (`spark_dsv4_model.h:66-67`), KV
**BF16** (`spark_dsv4_model.h:68`). Community results use the same weights
(MXFP4/FP8) but almost always quantize KV harder than we do — that is the
single most important comparability caveat.

| Level (weights / KV) | Best reported | Source | Comparable to us? |
| --- | --- | --- | --- |
| MXFP4+FP8 / **BF16 KV** (ours, no spec) | 33.55 (merged-main) | `PERFORMANCE_STATUS.md:322-355` | **our baseline** |
| MXFP4+FP8 / FP8 KV, **MTP-only** | **49.4** (54.4 peak) | Verel-lab, 4× GB10 | KV slightly less accurate than ours |
| MXFP4+FP8 / FP8 KV, **DSpark** | **~90** (4-node) · 62.48 (2-node) | jeffery2011.jc · tonyd2wild | KV slightly less accurate than ours |
| MXFP4+FP8 / **NVFP4 KV**, DSpark | **123.13** | joesinvestments, 4× GB10 | KV 4-bit → **not** matched; upper bound |
| MXFP4+FP8 / BF16-or-FP8 KV, DSpark **under PP4** | **98.1** (code-gen 118.8) | allover326, 4× CMP 170HX | weights matched; **different silicon** (sm_80) |
| W4A16 INT4 + FP8 block (heavy) | 14–17 (DGX Spark TP2) / 47–48 (RTX PRO 6000) | canada-quant | **not** comparable (4-bit experts, MTP dropped) |
| NVFP4 full | MTP +38% (RTX PRO 6000) | hikarioyama | not comparable (4-bit weights, different HW) |

**Precision-matched number to target.** At our exact weights precision, the
GB10 TP4 SOTA to match/beat is **~123 tok/s** (joesinvestments, DSpark k=7 +
probabilistic), but that rides NVFP4 KV. The **matched-KV-precision** targets
are **~90 tok/s** (jeffery2011.jc, 4-node DSpark, FP8 KV) and **49.4 tok/s**
(Verel-lab, MTP n=2, FP8 KV). Our BF16-KV no-spec floor (33.55) should cross
the internal 50 target first, then the DSpark path should land **≥90 tok/s at
matched KV** and only claim 123 if we also adopt an NVFP4/FP8-KV *capacity*
experiment (correctness path stays BF16 — `DSPARK_DSV4_FLASH_DESIGN.md:66`).

---

## 3. Ranked experiment ideas (informed by the survey)

Expected gains are the author-reported deltas where available, else estimates;
code-size is for our tree (dsv4-flash lane).

1. **Greedy → probabilistic draft sampling.** joesinvestments' single biggest
   win: acceptance 26.5%→**34.3%**, tok/step 2.86→3.40, ~**+19% output speed**
   (RECIPE.md). Our verify path is greedy (`module.c:3582-3586`); the sampled
   kernel already exists (`inference/kernels/speculate.cuh:78`, V-02).
   *Expected: +15–20% on the DSpark path · code ≈ +60–80 L.*
2. **Measure DSpark at k=7, then sweep k=5/8/10.** The community k-sweep is
   decisive: k=7→123.1, k=8→115.2, k=10→102.6 (RECIPE.md) — k is a property of
   the *engine build*, not the model. Our driver is spec_step=7, implemented but
   **unmeasured** (`KERNEL_CONTRACT_CARDS.md:250-347`). *Expected: unlock the
   1.9–3× speculative level · code ≈ 0 (+ measurement harness).*
3. **TP4×PP4 stage-3 draft placement (DSpark under PP).** allover326: DSpark
   goes *negative* on TP above ~8 concurrent but **1.93× on PP** (50.8→98.1),
   because pipeline bubbles are exactly what the drafter fills. This is the
   SparkPipe differentiator vLLM lacks. *Expected: the PP4 batch/aggregate level
   · code ≈ +150–250 L* (`DSPARK_DSV4_FLASH_DESIGN.md:123-133`).
4. **Predeclared collective program (kill the 130-collective floor).** Our
   measured no-spec ceiling is ~5.5 ms/token of serial collective idle
   (`DSPARK_DSV4_FLASH_DESIGN.md:9-10`). Verel-lab's MTP-only 49.4 tok/s is the
   evidence that a clean control plane reaches ~50 no-DSpark. *Expected: 33.55 →
   ~50 no-spec · code ≈ −150 to −300 L.*
5. **NVFP4/FP8 KV-cache as a capacity+speed experiment (not the correctness
   path).** joesinvestments' 123.13 rides `nvfp4_ds_mla`; the gap between the
   90 (FP8 KV) and 123 (NVFP4 KV) numbers is the KV-quant headroom. *Expected:
   +KV capacity (toward 2.5 TB hot tier) + step-time; correctness path stays
   BF16 · code ≈ +100–200 L (a KV codec in the paged cache).*
6. **Correct CUDA-graph capture size + FULL_DECODE_ONLY.** capture =
   max_seqs × (k+1) = 12×8 = 96; under-sizing silently drops decode to eager
   (RECIPE.md). Our driver already graphs 130 collectives; verify the 8-row
   speculative bucket is fully captured. *Expected: no eager fallback under
   load · code ≈ 0 (config).*
7. **MoE backend parity (flashinfer b12x + autotune).** joesinvestments runs
   `--moe-backend flashinfer_b12x --enable-flashinfer-autotune`; our routed-
   expert compute is the largest kernel-time share (21%,
   `PERFORMANCE_STATUS.md:232`). *Expected: match their expert-GEMM efficiency
   · code ≈ kernel contract update (CUDA-KERNELS agent).*
8. **Warmup + tok/step telemetry protocol.** joesinvestments: warmup depth
   moved the result **11 tok/s** (111.86→123.13), and "aggregate acceptance is a
   trap" (compare tok/step). Adopt their `bst_parity.py` protocol (2 warmups +
   6 discarded + 10 measured) and report tok/step + per-position acceptance.
   *Expected: reproducible + comparable numbers · code ≈ +30–50 L (telemetry).*
9. **Prefix-cache + chunked-prefill admission budget.** max_num_batched_tokens
   = 8192 + (k−1)×max_seqs (RECIPE.md); 96–98% prompt-prefix hits on agentic
   traffic. Maps to our admission core + scheduler (shared, review-only for me).
   *Expected: TTFT + aggregate on agentic traffic · code ≈ config + scheduler
   table.*
10. **Content-aware k / acceptance instrumentation.** allover326's decode
    speedup is strongly content-dependent (code-gen 2.16× vs prose 1.70×);
    joesinvestments' per-position acceptance is 0.730/0.569/0.372/0.226/0.131.
    Instrument our driver to measure per-position acceptance before committing
    to a fixed k. *Expected: inform the k/acceptance decision · code ≈ +30–50 L.*

---

## 4. Summary line

On GB10 (4× DGX Spark, TP4) at our exact weights precision (MXFP4 experts +
FP8 linears), the SOTA to match/beat is **123.13 tok/s single-stream decode**
(joesinvestments, DSpark k=7 + probabilistic drafting) — but that rides NVFP4
KV, so the **matched-KV-precision target is ~90 tok/s** (jeffery2011.jc 4-node
DSpark, FP8 KV) and **49.4 tok/s** for MTP-only (Verel-lab); our BF16-KV
no-spec floor (33.55) must first cross 50, then DSpark should land ≥90 matched
and approach 123 only with an optional NVFP4/FP8-KV capacity experiment.

---

## Source URLs

- https://github.com/joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark (README + RECIPE.md)
- https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-4x-dgx-spark-via-vllm-jasl-fork-tp-4-rdma-mtp-49-54-tok-s-single-stream-full-recipe-the-traps/373808
- https://forums.developer.nvidia.com/t/4-node-dgx-spark-cluster-with-deepseek-v4-flash-0731-dspark-benchmark-prefill-2-500-t-s-decode-90-t-s/378878
- https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846
- https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-60-tok-s-900K-ctx-2x-DGX-Spark
- https://github.com/allover326/deepseek-v4-cmp170hx (README + RESULTS.md)
- https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8
- https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP
- https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8
- https://github.com/hikarioyama/dsv4-flash-nvfp4-sm120
- https://github.com/Weschera/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark
- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
- https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark
