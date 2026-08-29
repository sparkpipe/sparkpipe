# Survey: Kimi K3 reported inference performance (public sources)

Owner: K3 MODEL agent · Survey of public sources (GitHub, Hugging Face, NVIDIA
forums/docs, X/Twitter, vendor blogs) for the best REPORTED Kimi K3 serving
numbers and the techniques behind them, focused on small-batch decode,
speculation/MTP, and quantization variants. Sources fetched 2026-08-17; dates in
the table are the source's own publication date. Proposal/docs only.

**Model identity (from the sources, cross-checked against our clone):** Kimi K3 =
2.78T-parameter MoE (Moonshot AI), 16-of-896 experts active per token, 1M-token
context, multimodal, hybrid Kimi Delta Attention (linear) + MLA, native MXFP4
weights (`docs/K3_PERF.md`, thakicloud analysis). Our clone's driver matches:
896 experts / top-16 / latent 3584 (`inference/llms/kimi_k3/generated_config.h:7-12`).

---

## 1. Results table

| Reference (source · URL) | Date | Hardware | TP | Quantization | B1 tok/s (decode) | Key techniques |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM day-0, no speculation — [thakicloud](https://thakicloud.com/tech-blog/en/llmops/kimi-k3-dspark-bs1-decode/) (citing vLLM release notes) | 2026-07-27 | 16× GB300 NVL72 | TP16 | MXFP4 experts + BF16 dense, FP8 KV | **118** | FlashKDA, fused KDA decode/proj/conv, fused AttnRes, reimplemented MLA, SiTU MXFP4 MoE, DP16/EP16 |
| vLLM + DSpark — [thakicloud](https://thakicloud.com/tech-blog/en/llmops/kimi-k3-dspark-bs1-decode/) | 2026-07-27 | 16× GB300 NVL72 | TP16 | same | **370** (3.14x) | DSpark 5-layer block-diffusion drafter, gamma=7, non-causal attn, Markov head |
| vLLM + DSpark peak — [thakicloud](https://thakicloud.com/tech-blog/en/llmops/kimi-k3-dspark-bs1-decode/) | 2026-07-29 | 16× GB300 NVL72 | TP16 | same | **464** (3.93x implied) | acceptance ~0.81, 4.33 committed/step |
| GPUStack 8×B300, 64K/200K ctx — [GPUStack](https://www.cnblogs.com/gpustack/p/22073008.html) | 2026-07-30 | 8× B300 | TP8 | MXFP4 + FP8 KV + DSpark | e2e 100.5s (vLLM) / 150.8s (SGLang) @10-conc 64K | FlashInfer MLA, TRTLLM_RAGGED prefill, K3 latent-MoE tail fusion, dcp-size=8 |
| GPUStack 8×B300, 200K — same | 2026-07-30 | 8× B300 | TP8 | same | e2e 225.3s (SGLang) / 295.2s (vLLM) @10-conc | **decode context parallel (dcp=8)**: SGLang 1.25x degradation vs vLLM 3.29x |
| NVIDIA Dynamo K3 recipe — [NVIDIA](https://docs.nvidia.com/dynamo/dev/recipes/kimi-k3) | 2026-08 | 16× GB200 / 16× GB300 | TP16 / TP8 | MXFP4 + BF16, FP8 KV | — (deploy recipe, no tok/s) | KV-aware routing, FlashInfer MLA + TRT-LLM MoE backend, MNNVL |
| Qwen3-32B-FP8 (dense "32B-class" ref) — [aiconfigurator](https://github.com/ai-dynamo/aiconfigurator) | 2026 | 32× H200 | TP4 | FP8 | 684.79 tok/s/gpu, 21.9K aggregate, TPOT 9.97ms | disagg prefill/decode |
| Baseten suffix-automaton MTP — [baseten.co](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/) | 2026-05-05 | (DeepSeek V3.1 NVFP4) | — | NVFP4 | — | SA + MTP hybrid: **+40% throughput / −40% latency** vs MTP on coding; +30-33% accept length |
| Moonshot "8700 tok/s chip" — [wccftech](https://wccftech.com/kimi-k3-built-a-chip-in-48-hours-over-8700-tokens-s-as-china-delivers-2-8-trillion-ai-model/) | 2026-07 | 45nm @ 100MHz custom | — | — | 8700 (demo) | **Not a serving benchmark** — an AI-chip-design demo (K3 designing its own chip); excluded from SOTA |

Notes on the table:
- The **118 / 370 / 464** row is the canonical small-batch number: B1, TP16, low-entropy
  reasoning workload. Hardware wording varies across sources ("16 GB300 NVL72" / "4x4
  GB300" / "4×GB300 TP16"); thakicloud reads all three as **TP16 = 16 GPUs**. The 464
  peak is a different day from the 118 baseline, so "3.93x" is not a controlled A/B —
  the controlled 118→370 is **3.14x**.
- The **GPUStack** row is the only published TP8/long-context decode result; its
  headline is that decode-throughput degradation from 64K→200K context is the real
  battle (3.29x drop for vLLM vs 1.25x for SGLang with decode context-parallel), not
  prefill.
- The **Qwen3-32B-FP8** row is the "Qwen3-32B-style" dense active-parameter reference
  the brief asked for (K3 activates 16/896 experts ≈ a Qwen3-32B-class active size).
  It is not precision-matched to K3 (FP8 dense vs MXFP4 MoE) but is the TP4 reference.

---

## 2. Best reported result PER quantization level (comparability)

| Quantization level | Best reported | Source | Comparable to our baseline? |
| --- | --- | --- | --- |
| **BF16 (full)** | no published B1 number; needs 16× B200 (~3.1 TB) — [HTX](https://www.htx.com/news/kimi-k3-which-used-to-require-16-b200s-now-fits-on-just-8-am-y4XP20tK/) | HTX/AMD | Our BF16 dense spine is a *subset* (experts are MXFP4), not full-BF16 |
| **MXFP4 (native: 4-bit experts + BF16 dense) — THE reference** | **464 tok/s (DSpark) / 118 tok/s (no spec) B1, TP16** | vLLM via thakicloud | ✅ **MATCHED weight precision** — our pack V2 is exactly MXFP4 experts + BF16 dense (`docs/K3_WEIGHT_ONLY_MXFP4.md`) |
| **MXFP4 + FP8 KV** | same 118/464 (the reference stacks all use FP8 KV) | vLLM/Dynamo/GPUStack | ⚠️ our KV is BF16 (`K3_KV_BITS 16`); FP8 KV is the one precision delta |
| **NVFP4** (PatronusAI/kimi-k3-nvfp4) | no B1 number published | HF | ❌ NOT comparable (different 4-bit codec) |
| **MXFP4-q8 / REAP80** (pipenetwork MLX) | no B1 number published | HF | ❌ NOT comparable (mixed 4/8-bit, MLX) |
| **2.5-bit Cubic** (QuantTrio) | no B1 number published | HF | ❌ heavily quantized, below our floor |
| **GGUF derisked** (Blackfrost) | no B1 number published | HF | ❌ NOT comparable |

**Comparability verdict (the user's caveat):** the only published numbers that are
precision-matched to our MXFP4 weight-only pack are the **MXFP4-native** rows — and
the only one with a measured B1 decode number is the vLLM **118 / 370 / 464** set.
Everything else (NVFP4, 2.5-bit, GGUF, MXFP4-q8) is *more* heavily quantized than our
pack and is a different codec, so none of it is a valid target at matched precision.
The FP8 KV cache is the single real delta between the reference stack and ours (a KV
precision, not a weight precision) — our weight scheme already matches.

**Precision-matched number to target:** **118 tok/s B1 decode (no speculation) and
464 tok/s (with DSpark), MXFP4 experts + BF16 dense, TP16** — the vLLM reference on
16× GB300. On our GB10 fleet this does **not** translate 1:1 (see summary line), but
this is the absolute SOTA at our weight precision.

---

## 3. Ranked experiment ideas (informed by the survey)

Expected gains are quoted vs our measured 18.0 tok/s B1 (`docs/K3_PERF.md:36`).

1. **Land the K3 DSpark draft backend (block-7, GQA-16).** The survey's single
   biggest lever is software, not hardware: the controlled 118→370 is **3.14x**, the
   peak 464 is 3.93x, and batch throughput is +68% (matches our `dspark.h:96-101`
   SGLang citation). Our verifier (tap capture + `K3FoldAccepted`) already exists;
   only the drafter kernels are missing. **Expected:** 18 → ~55-65 tok/s B1.
   **Code:** ~80-100 lines net (neutralize the shared `glm52_dspark_draft_backend`
   + add GQA-16 attention/KV-scatter) — or ~1000 if duplicated. (= TOP10 #2.)
2. **KDA decode gate / latent-MoE tail fusion.** vLLM ships
   `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` and a ROCm "fuse Kimi-K3 KDA decode
   gate" PR (vllm#50634). We already fuse the KDA projections + AttnRes; the remaining
   epilogue is out_norm + output-gate + o_proj as separate launches.
   **Expected:** +5-15% decode (removes per-layer launches on the critical path).
   **Code:** ~40 lines.
3. **FP8 KV cache.** Every reference stack uses FP8 KV; ours is BF16
   (`K3_KV_BITS 16`). Halves the MLA KV arena and, on long context, the KV read term
   that GPUStack shows dominating (3.29x vLLM degradation vs 1.25x with SGLang dcp).
   **Expected:** +capacity (2x sequences) and modest long-context decode.
   **Code:** ~50 lines (KV store dtype + a fp8 quant kernel).
4. **Prefix caching / KV-aware routing.** The Dynamo Qwen3-32B trace shows 36.64%
   cache efficiency on shared-prefix conversation traffic; K3's Dynamo recipe is
   "KV-aware routing" first. For us this is the KV seam (TOP10 #3): paged KV + prefix
   reuse on the common arena. **Expected:** TTFT −30-40% on shared-prefix traffic
   (decode unchanged). **Code:** ~40 lines (already specced, PROPOSAL_KV_SEAM §3.6).
5. **Suffix-automaton (SA) + DSpark hybrid.** Baseten reports **+40% throughput /
   −40% latency** vs MTP alone on agentic coding (SA accept length 10+ on code),
   header-only POD CUDA, near-zero overhead. Orthogonal to #1 and applies to our
   DSpark path once it lands. **Expected:** +10-40% on code traffic, ~0 on chat.
   **Code:** ~150 lines (SA is the one real new kernel; rest is dispatch).
6. **BF16 KDA state.** The decode roofline flags the fp32 KDA state read/write as a
   bandwidth term (`docs/K3_PERF.md:37`); halving it to bf16 cuts that term and
   doubles resident sequences. **Expected:** toward the 20.6 tok/s roofline.
   **Code:** ~30 lines. (= TOP10 #5.)
7. **Reduce-scatter + all-gather fused AR.** Halves per-rank wire bytes for the
   TP4 fused 3×7168 exchange (`docs/K3_PERF.md:67`). **Expected:** incremental at B1
   (path is weight-bound), pays at batch. **Code:** ~50 lines. (= TOP10 #8.)
8. **Prefill query quantization + TRTLLM_RAGGED-style MLA prefill.** The reference
   sets `use_prefill_query_quantization: true` and a ragged MLA prefill backend.
   Our MLA prefill path is chunked decode; a ragged prefill + query-quantization
   would lift the measured 92→1537 tok/s prefill curve (`docs/K3_PERF.md:39`).
   **Expected:** prefill speedup (not B1 decode). **Code:** ~100 lines.
9. **Decode context parallelism (dcp) for the MLA layers.** SGLang's dcp-size=8 is
   the 200K-context win (1.31x over vLLM). Our KDA is already O(1) state (no win),
   but the 24 MLA layers grow KV with context; a context-parallel MLA decode would
   mirror SGLang's 200K result. **Expected:** long-context decode only (200K+).
   **Code:** large (~200+ lines, collective + KV sharding).
10. **Measure acceptance-first instrumentation.** thakicloud's conclusion: adoption
    is an acceptance-rate calculation, not a speedup hope; vLLM emits accepted/proposed
    token metrics. Wire the same counters into `K3EngineResolveVerify`
    (`engine.h:407`) so #1/#5 land with a measured per-workload acceptance rate.
    **Expected:** 0 perf, de-risks #1/#5. **Code:** ~15 lines.

---

## 4. Summary line (SOTA to match/beat on GB10 at matched precision)

> At matched weight precision (MXFP4 experts + BF16 dense), the SOTA to match is
> **vLLM's 464 tok/s B1 with DSpark / 118 tok/s without, TP16 on 16×GB300**
> (July 2026). On our 16-spark GB10 fleet the hardware-normalized target is to beat
> our own 20.6 tok/s roofline by replicating the *software* levers vLLM already ships
> (fused KDA + interleaved MXFP4 GEMM + fused AttnRes — which we have) **plus DSpark
> speculation at the reference 3.1-3.9x acceptance** (18 → ~55-65 tok/s B1) **plus
> FP8 KV cache** — the one precision delta still separating our stack from the
> reference.
