# DFlash2 / SGLang new release — research analysis for SparkPipe

Researched 2026-08-25. READ-ONLY research deliverable; every external claim carries its
source URL + access date. Builds on [`docs/DFLASH2_ADOPTION_SPEC.md`](DFLASH2_ADOPTION_SPEC.md)
(mined 2026-08-19), [`docs/DFLASH2_HANDOFF.md`](DFLASH2_HANDOFF.md) (2026-08-22 state),
[`docs/QWEN38_DFLASH2_RUNBOOK.md`](QWEN38_DFLASH2_RUNBOOK.md), and
[`docs/VLLM_SGLANG_SPEC_CONTRACTS.md`](VLLM_SGLANG_SPEC_CONTRACTS.md).

**What "the new release" actually is:** not a new drafter checkpoint — the drafter is
byte-unchanged since launch. It is (1) SGLang PR #35496 (merged 2026-08-20, commit
`1cf2b8c`) teaching the DFlash2 selector to run against a **quantized** target lm_head,
(2) a re-export of the `RadixArk/Qwen3.8-27B-NVFP4` target that keeps the **main model's
lm_head in dense BF16**, and (3) a wave of GB10 DGX Spark community measurements whose
headline is **~50 tok/s decode on Qwen3.8-27B**. vLLM's DFlash2 PR also merged in this
window, closing adoption-spec gate 1.

---

## 0. TL;DR verdicts

| Question | Verdict |
|---|---|
| What is the BF16 lm_head? | The **main model's** lm_head, in the RadixArk NVFP4 export (`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`). The DFlash2 drafter still ships **no** lm_head or embeddings of its own. |
| Main-model weight precision | MLP gate/up/down = NVFP4 W4A4 (group 16); attention projections = FP8; lm_head + MTP head + vision/embed = BF16. Not FP8-everything. |
| ~50 tok/s hardware | **Yes — DGX Spark GB10**, same class as SparkPipe's spark2. Model: Qwen3.8-27B, NVFP4 body + BF16 head, block 8. (A separate ~59 tok/s datapoint is a single datacenter A100.) |
| Drafter architecture delta vs our port | **None at the checkpoint level** (same tensors/shapes). Delta is engine-side: quantized-head selector path now upstream; vLLM merged; new GB10 acceptance/quality data. |
| Qwen3.8-27B | **GO** (drafter exists; precision story matches our packs). |
| DeepSeek-V4-Flash | **NO-GO for DFlash2** — no public drafter. DFlash **v1** drafters exist (`RedHatAI/DeepSeek-V4-Flash-speculator.dflash`). Stay on our DSpark port. |
| GLM 5.2 | **NO-GO** — nothing public. Closest artifact is `z-lab/GLM-5.1-FP8-DFlash` (**v1**, GLM **5.1**). |

---

## 1. Findings table — claim vs source vs applicability to SparkPipe

| # | Claim (as published) | Source (URL, accessed 2026-08-25) | Applicability to SparkPipe |
|---|---|---|---|
| F1 | SGLang merged "[Spec] Support quantized target lm_head in the DFlash2 selector", commit `1cf2b8c`, 2026-08-20. Routes the selector candidate projection through `lm_head.quant_method.apply` when `should_apply_lm_head_quant_method` admits the head; padded-vocab tail masked to −inf instead of cropped (flashinfer radix top-k contiguity); quantized and dense heads share the graph-folded `_SelectorDraftSampler`; TP communicates local top-k ids+values only. | <https://github.com/sgl-project/sglang/pull/35496> | **High.** Removes the engine-side dense-head hard wall our adoption spec recorded as a blocker. Validates SparkPipe's own W4 design (top-K over a quantized/screened head). Their measured NVFP4-packed-head GSM8K accept 4.85 @ RTX 5090 gives us an upstream acceptance anchor for quantized-head serving. |
| F2 | Same PR accuracy note: with spec vs target-only, greedy outputs "fork only at near-tie tokens and 22/28 completed chains reach the same final answer"; GSM8K 200q 95.5% spec vs 98.0% target-only (RTX 5090, NVFP4-packed head). | <https://github.com/sgl-project/sglang/pull/35496> | Medium. Quantified cost of packed-head candidate generation vs dense head on Blackwell-class consumer parts; bounds what we should expect if we ever de-BF16 our pack's head. Our handoff already treats sub-bf16 near-tie flips as a known numerics class. |
| F3 | Checkpoint update comment: "`RadixArk/Qwen3.8-27B-NVFP4` now ships a BF16 `lm_head` (original weights from Qwen/Qwen3.8-27B) … new revision `496d00f2`, pre-change revision `554ebba` remains in repo history"; motivation: dense head unblocks DFlash2/DSpark selector natively, "median accept len 3.85 vs 3.78"; cost "+1.7 GB" disk (~3.2 GB runtime); post-change measurements on 4×GB300: GSM8K 96.36%, Terminal-Bench 69.4%. Also names `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` as "the dense-lm_head export that will replace the current NVFP4 repo". | <https://github.com/sgl-project/sglang/pull/35825> (description + comments) | High for Q1. This is the origin of the "BF16 lm_head" talking point. ⚠️ Discrepancy: live HF refs show `RadixArk/Qwen3.8-27B-NVFP4` main at `319f741…` and its card still describing the lm_head as NVFP4 W4A4 — treat the separate `-BF16-LMHead` repo as the authoritative BF16-head artifact until the in-place update is confirmed by fetching revision `496d00f2` directly. |
| F4 | `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`: created 2026-08-20, current sha `009632fef96dd349150baa780c984e62e70e91fe`. `quantization_config`: group_0 num_bits 8 → all attention projections (linear_attn in_proj/out_proj + self_attn q/k/v/o across layers 0–63); group_1 num_bits 4 (NVFP4 W4A4) → all MLP gate/up/down; `ignore: ["mtp*", …]`. Safetensors census: BF16 3.454B params, F8_E4M3 7.214B, U8 8.556B, total 18.165B. lm_head appears in neither quant group ⇒ dense BF16. | <https://huggingface.co/api/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead>, <https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead> | **Definitive for Q1/Q-main-precision.** See §2. |
| F5 | `RadixArk/Qwen3.8-27B-NVFP4` card: "The MLP `gate_proj`, `up_proj`, and `down_proj` layers and `lm_head` use dynamic NVFP4 W4A4 quantization with group size 16. Attention weights use FP8, while MTP and vision tensors retain the source BF16 precision." Produced/validated on GB300; release 2026-08-14. | <https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4> | Confirms pre-update recipe (packed head) and that the *body* was never FP8 — it is NVFP4-experts + FP8-attention from day one. |
| F6 | MiaAI-Lab DGX Spark repo (GB10): DFlash2 code probe **50.9 tok/s** (n=5: 50.77–51.07), long essay 25.4, short chat 31.7/28.9/**66.6** (T0/T1/T1-thinking, streamed, counted from `completion_tokens`); concurrency ladder aggregate 56.6 → 227.6 tok/s at 16 streams (per-stream 56.6→28.2); default target `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`; draft pinned `z-lab/Qwen3.8-27B-DFlash2@50307d4`. Caveats stated by author: code deltas <15% are noise ("ties DSpark" at 51.5), SSE event-counting under-reads ~4×, box drifts under power cap. | <https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark> | **This is the "~50 tok/s decode" claim, on GB10.** Directly comparable hardware class to spark2. Note their DSpark control (51.5 code) — DFlash2's edge on GB10 shows up in chat/essay, not the code probe. |
| F7 | hasso5703 verified cookbook cell (#35860), ASUS Ascent GX10 (GB10, 128 GB): greedy batch-1 code 39.4 / reasoning 43.5 / free prose 20.2 tok/s (vs same-night DSpark 25.4/30.5/14.0); `bench.sh` greedy median **50.0 tok/s**; c8 aggregate 135–148; c32 **258 tok/s**; K sweep confirms 8 optimal, 9 collapses; GSM8K 188/200 exact parity with DSpark; tool-eval-bench 91 vs DSpark 93 attributed to near-tie flips under block verification; FP8-target variant 108 tok/s @ c8. Uses `--speculative-draft-model-quantization unquant`, mem-fraction 0.50, `--disable-flashinfer-autotune`. | <https://github.com/sgl-project/sglang/issues/35860> | **Best GB10 evidence**: independent second box, deterministic-stack methodology, quality parity study. Their c8/c32 numbers prove multi-stream DFlash2 stability on GB10 — relevant to our concurrency>1 concerns from the pre-merge vLLM crash report. |
| F8 | NVIDIA forum (Ama5u, 2026-08-22), one DGX Spark: decode tok/s by content — code 32.6, reasoning 47.3, prose ~21, math 55.8 (DFlash2) vs 19.8–25.4 (vLLM+MTP); quality 91–93/100 greedy ≈ thinking; SGLang cell needs commit ≥ `1cf2b8c` (#35496) because the released image refuses NVFP4 targets ("requires a dense FP16/BF16/FP32 target lm_head"); warns mem-fraction >0.50 risks locking unified memory; "DFlash2 doesn't play with YaRN". | <https://forums.developer.nvidia.com/t/qwen3-8-27b-benchmarking-on-one-dgx-spark-dflash2-beat-vllm-mtp-and-greedy-beat-the-thinking-sampler/380957> | Third independent GB10 confirmation of ~50-class tok/s on reasoning/math content. Operational warnings transfer verbatim to SparkPipe deployments (unified-memory guardrails). |
| F9 | mindstudio.ai blog (2026-08-18): single **A100**, baseline 28.9 → DFlash2 **59.1 tok/s** (~2×), one prompt ~72. | <https://www.mindstudio.ai/blog/dflash-2-speculative-decoding-qwen> | The only ~50+-tok/s datapoint that is NOT GB10. If anyone quotes "~59 tok/s" it is a datacenter A100, not Spark hardware. Do not mix the two populations. |
| F10 | vLLM PR #52816 "[Spec Decode] DFlash2: local convolution + candidate selector" — **merged 2026-08-21T05:27Z**, merge commit `b389ac29465b33f9e9c534df221ea3c129e9793f`, 14 files, +866/−44. | <https://api.github.com/repos/vllm-project/vllm/pulls/52816> | Closes adoption-spec gate 1: both major engines now carry DFlash2 upstream. The pre-merge sm120 concurrency crash report no longer represents shipped vLLM. |
| F11 | SGLang docs PRs #35786/#35825 re-measured the Qwen3.8-27B grids on one commit/image: RTX 5090 cells — DFlash2 bf16 accept 4.29, TPOT 4.92 ms; DSpark bf16 accept 2.59; EAGLE bf16 accept 3.02. RTX PRO 6000 — BF16+DFLASH2 accept 5.42 (TPOT 8.19 ms); NVFP4(BF16-LMHead)+DFLASH2 accept 4.19 (TPOT 5.36 ms). | <https://github.com/sgl-project/sglang/pull/35825> | Published acceptance anchors for quantized-target serving on small cards. NVFP4 accept 4.19–4.85 sits below BF16-target 5.42 — the quantization give-up is real but bounded (~20–23% on these cells), consistent with our GGUF-derived 1–3%-class estimate being optimistic for W4A4 bodies. |
| F12 | Drafter repos unchanged: `z-lab/Qwen3.8-27B-DFlash2` main = `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`; `incoai/Qwen3.8-27B-DFlash2` main = `dedf8df68adfb1afeaf7b7480c0a0243108177b4`; "byte-identical … same config.json and model.safetensors blobs" (#35860 note 6). z-lab author listing shows **no DFlash2 drafter newer than 2026-08-15/18** for any target beyond Qwen3.8-27B + Muse-Glimmer-30B. | <https://huggingface.co/api/models/z-lab/Qwen3.8-27B-DFlash2/refs>, <https://huggingface.co/api/models/incoai/Qwen3.8-27B-DFlash2/refs>, <https://github.com/sgl-project/sglang/issues/35860>, <https://huggingface.co/api/models?author=z-lab&sort=createdAt&direction=-1> | Pins Q3/Q4. Any "new DFlash2 drafter" rumor has no HF substance as of 2026-08-25. Pin `z-lab/Qwen3.8-27B-DFlash2@50307d4` (or incoai@dedf8df) in packs/scripts. |
| F13 | DeepSeek-V4-Flash drafter search: zero DFlash2 results. Existing DFlash **v1** artifacts: `RedHatAI/DeepSeek-V4-Flash-speculator.dflash` (2026-07-02, `speculators` library), `inference-optimization/dflash-DeepSeek-V4-Flash-all-swa-muon-speculators-50k` (2026-06-24), `inference-optimization/dflash-DeepSeek-V4-Flash-speculators-50k` (2026-05-29), plus a third-party GGUF of a DSpark-family drafter. | <https://huggingface.co/api/models?search=dflash2%20deepseek>, <https://huggingface.co/api/models?search=DFlash%20DeepSeek&limit=20> | Grounds the DSV4-Flash NO-GO. None of these are `DFlash2DraftModel` checkpoints; they would draft via the v1 path (no conv, no selector) — a different architecture than our landed qwen38 work. |
| F14 | GLM search: zero GLM DFlash2 results. Closest: `z-lab/GLM-5.1-FP8-DFlash` (created 2026-06-07) — DFlash **v1**, GLM **5.1**, not 5.2. | <https://huggingface.co/api/models?search=dflash2%20glm>, <https://huggingface.co/api/models?author=z-lab&sort=createdAt&direction=-1> | Grounds the GLM 5.2 NO-GO. Our `spark_dspark_drafter.h` GLM52 tables are DSpark-side constants; nothing changes for them. |

---

## 2. Q1 — What exactly is BF16, and what precision are the main weights?

**The BF16 lm_head is the MAIN MODEL'S lm_head.** The DFlash2 drafter still ships neither
`embed_tokens` nor `lm_head` (adoption-spec §(a): drafter borrows the target head; F12
shows the drafter repo unchanged). Two distinct artifacts got the "BF16 head" treatment:

1. **`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`** (the explicit export; created 2026-08-20,
   sha `009632fe…`): NVFP4-W4A4 body with the lm_head left **dense BF16** (it appears in
   neither ModelOpt quant group; param census BF16 3.45 B includes head ~1.27 B
   [248320×5120] + MTP + vision/embed). Source: F4.
2. **`RadixArk/Qwen3.8-27B-NVFP4`** in-place update to revision `496d00f2` (pre-change
   `554ebba`) per PR #35825 — claimed to swap the packed NVFP4 head for the original BF16
   head (+1.7 GB disk / ~3.2 GB runtime). ⚠️ Unverified against live Hub state (card still
   describes a packed head; branch tip differs). Verify before pinning. Source: F3.

**Main-model weight precision (both exports):**

| Component | Precision |
|---|---|
| MLP gate/up/down (all 64 layers) | NVFP4 W4A4, group 16 (dynamic activation quant) |
| Attention projections (GDN linear_attn in/out + full-attn q/k/v/o) | FP8 (num_bits 8) |
| lm_head | **BF16, dense** (post-update exports) / NVFP4-packed (pre-update `554ebba`) |
| MTP head, vision tower, embeddings | BF16 (ModelOpt `ignore` list) |

So "BF16 lm_head" does **not** mean an FP8/BF16 body flip — the body stays NVFP4+FP8 and
only the output head is held dense. Separately, SGLang #35496 means a *quantized* head no
longer refuses to boot: the dense head is now an accuracy/perf choice (acceptance median
3.78 → 3.85 per F3; NVFP4-packed-head accept 4.85 achievable via the quant path, F1/F11),
not an engine requirement.

---

## 3. Q2 — Hardware provenance of "~50 tok/s decode"

**Same hardware class as SparkPipe: yes.** All three primary ~50 tok/s claims are single
DGX Spark GB10 boxes running Qwen3.8-27B:

| Source | Box | Config | Numbers (decode, batch-1 unless noted) |
|---|---|---|---|
| MiaAI-Lab README (F6) | DGX Spark GB10 | NVFP4-BF16-head target + DFlash2 block 8 | code **50.9**, essay 25.4, chat 28.9–66.6; agg 227.6 @ c16 |
| hasso5703 #35860 (F7) | ASUS Ascent GX10 (GB10) | NVFP4 target (packed-head repo + #35496 path), DFLASH2 | bench median **50.0**; code 39.4 / reasoning 43.5 / prose 20.2; agg 258 @ c32 |
| NVIDIA forum Ama5u (F8) | DGX Spark GB10 | NVFP4 + #35496 build | reasoning **47.3** / math **55.8** / code 32.6 / prose ~21 |

The outlier is mindstudio.ai's **59.1 tok/s on one datacenter A100** (F9) — different
hardware population, do not blend. Datacenter references from the original release
(H200: 236 tok/s conc-1 GSM8K) remain the other population.

Context vs SparkPipe's own bar: our production qwen38 FP8 TP1 measures 24.5 tok/s wall /
28 tok/s decode on O512 (runbook §6, handoff §0g). The community GB10 cells land in the
39–56 band on code/reasoning/math content — i.e., roughly **1.4–2× our current decode**
on favorable content, achieved with: NVFP4 body + BF16 head, flashinfer attention,
torch.compile + decode graphs, K=8, chunked prefill 8192, autotune disabled. That gap is
engine/kernel maturity (they get ~245 GB/s-class GEMM behavior from flashinfer/triton on
NVFP4; our WS kernel measures 176–205 GB/s on MX-FP8), not drafter quality.

---

## 4. Q3 — Drafter architecture vs what SparkPipe already landed

**Checkpoint-level delta: none.** The drafter SparkPipe would fetch today is byte-identical
to the one our port targets (F12): 5 layers, hidden 5120, GQA 32/8 heads ×128, intermediate
17408, block 8 (anchor + 7 mask slots), `sliding_attention`×5 window 2048 non-causal,
target taps `[5, 19, 33, 47, 61]` of 64 via `fc` [5120, 25600], grouped dynamic conv
kernel 2 / group 16 (per-layer base_kernel [2,2,5120] + kernel_projection [1280,5120] ×
attn+mlp), selector rank 256 / top-K 16 (codebooks [248320,256]×2, hidden_projection
[256,5120]), mask token 248070, vocab 248320, rope default θ=1e7, all-BF16, no
embed/lm_head. Adoption-spec §(b)/(e) shape tables remain authoritative.

**Engine-level deltas since our port landed (this is the actual "release"):**

1. **Quantized-target-head selector path (SGLang #35496, merged).** Upstream can now serve
   DFlash2 against packed NVFP4 or compressed-tensors FP8 heads via `quant_method.apply`
   with padded-tail masking, folded into the draft CUDA graph (F1). SparkPipe's equivalent
   (W4: top-16 over our quantized, shadow-screened head through
   `SparkQwen36LaunchHeadScreenedArgmaxScore`) was designed before upstream had this; the
   design parity is now upstream-endorsed, and their padded-mask trick is worth copying for
   any TP-padded-vocab case.
2. **Selection rule unchanged — and our finding stands.** The serving path still selects
   drafts per-mask-row argmax over target-head logits (our handoff §0: vLLM loads the v1
   speculator; the dflash2 walk module is dead code in serving). Nothing in #35496/#35825
   re-enables the lattice walk for serving; #35496's `_SelectorDraftSampler` generates the
   top-K candidates the argmax screen consumes. Our `SPARK_QWEN36_DFLASH2_STATE_SELECT=1`
   rank-0 argmax remains faithful to the reference serving path.
3. **vLLM merged (F10)** — the two-engine contract table (VLLM_SGLANG_SPEC_CONTRACTS.md)
   gains a merged second implementation to diff against; the T>0 sampler divergence noted
   in adoption-spec §(a) persists (SGLang inverse-CDF vs vLLM Gumbel).
4. **New acceptance anchors on our hardware class** (F6/F7/F11): GB10 content-dependent;
   GSM8K parity with DSpark (188/200 both); published card accepts 4.29 (RTX 5090 bf16
   target) / 5.42 (PRO 6000 bf16) / 4.19–4.85 (NVFP4 paths). Compare against our measured
   E≈5.66 mean-committed/round (O512, k=8, FP8-MX target) only after normalizing metrics:
   upstream "acceptance length" excludes the bonus and averages per-request on GSM8K-style
   sets; ours counts committed tokens incl. bonus on one prompt. Directionally: our FP8
   target's acceptance is competitive with their NVFP4-body cells, below their BF16-target
   cells — matching the handoff §0 attribution (fp8 taps vs bf16 teacher).

---

## 5. Q4 — Go/no-go per SparkPipe target

### Qwen3.8-27B — GO

- **Drafter (public, pinned):**
  - `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (canonical)
  - `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df68adfb1afeaf7b7480c0a0243108177b4` (mirror,
    byte-identical, F12)
- **Target options, best first:**
  1. Keep our production `packs/qwen38-fp8.tp1.qwen36sp` — it already carries the LM head
     un-quantized (see §6 parity verdict), so it satisfies even the *old* dense-head rule
     with zero repack.
  2. Optional A/B mirroring RadixArk: keep body, guarantee head BF16 explicitly in the pack
     manifest (it already is) — no action.
  3. Later experiment (not now): quantized head via W4, upstream-precedented by #35496 +
     the llama.cpp Q4_K_M precedent; budget the F11-class acceptance give-up (~15–23%),
     not the GGUF 1–3% figure.
- Engine pins: SGLang ≥ `1cf2b8c` if we ever compare against upstream on-site; our own
  stack needs no version bump for this research.
- Expected outcome band on spark2 after landing the §7 plan: 39–51 tok/s decode-class on
  code/reasoning prompts (community GB10 band), vs our 28 today.

### DeepSeek-V4-Flash — NO-GO (for DFlash2)

- No `DFlash2DraftModel` checkpoint exists publicly (F13). DFlash **v1** drafters exist
  (`RedHatAI/DeepSeek-V4-Flash-speculator.dflash`,
  `inference-optimization/dflash-DeepSeek-V4-Flash-*-speculators-50k`) but are a different
  architecture (no conv, no selector) — adopting them would be a new port, not an extension
  of the landed DFlash2 work, and our DSV4 line already runs the DSpark resident stage
  (DSV4_TP4_B1_HANDOFF.md).
- Action: none. Re-check the z-lab collection monthly; the blog's "write to us" invitation
  (adoption-spec §(e) gate 2) is the documented path to commission a DSV4-Flash drafter.

### GLM 5.2 — NO-GO

- Zero GLM DFlash2 artifacts (F14). Nearest object is `z-lab/GLM-5.1-FP8-DFlash` — v1,
  prior-gen model. Our GLM52 support remains DSpark-constants-only
  (`spark_dspark_drafter.h` ABI v3); nothing to adopt.

---

## 6. Precision-parity verdict vs SparkPipe's packs

SparkPipe pack baselines (task framing; runbook/handoff figures): Flash TP4 =
MXFP4 experts + FP8 non-expert + BF16 spine/KV; qwen38-27b = mixed FP8 pack 29.9 GB
(`qwen38-fp8.tp1.qwen36sp`, runbook lists 29 GB) or BF16 pack 54.6 GB (runbook: 51 GB).

| Property | SGLang/RadixArk release config | SparkPipe qwen38 FP8 pack | Parity verdict |
|---|---|---|---|
| Expert/MLP body | NVFP4 W4A4 g16 | MX FP8 E4M3+E8M0/128 (fmt 6) | Different format, same "4/8-bit body" class. No DFlash2 dependency either way. |
| Non-expert / attention | FP8 | FP8 (non-expert FP8 per pack recipe) | ✅ parity |
| Spine/KV | BF16 (MTP/vision/embed ignored by quant) | BF16 spine/KV | ✅ parity |
| **Target lm_head** | **BF16 dense** (post-update) — the headline change | **Already un-quantized**: handoff floors section reads the head as 2.5 GB ≈ 248320×5120×2 B, and the perf addendum names "the 2.54 GiB BF16 head READ" | ✅ **Our production pack already satisfies the dense-BF16-head property the release made explicit. Zero repack needed.** The release validates a choice we had already made implicitly. |
| Drafter | BF16, 3.85 GB, no head | Our drafter pack `qwen38-dflash2-drafter.qwen36sp` 3.6 GB, 5-layer BF16 | ✅ same checkpoint family; sizes consistent |
| Selector/head interaction | top-16 over dense BF16 head (or quant head via #35496) | rank-0 argmax over screened head output | Functionally equivalent for serving-path selection (§4.2); our addendum's fused-cost measurement (17.1 ms @ 7 rows) remains the honest fused number vs their 0.04 ms materialized-logits figure |

**Verdict: no precision mismatch blocks anything.** The release's BF16-lm_head move is
RadixArk catching up to where SparkPipe's packs already were. The only genuinely new
information is quantitative: dense-vs-packed head is worth ~0.07 median acceptance
(3.78→3.83-class, F3) and NVFP4-body acceptance lands 4.19–4.85 on small cards (F11).

For **Flash TP4 (DSV4)**: moot until a DFlash2 drafter exists (Q4). When one does, check
where the DSV4 lm_head lives in the TP4 pack before assuming the qwen38 parity argument
transfers — the four docs reviewed do not pin the DSV4 head storage precision.

---

## 7. Port-plan sketch

### Qwen3.8-27B (GO) — increments on adoption-spec §(e) W1–W8, resequenced by this research

1. **Pin artifacts** (0.5 d): record drafter revision `50307d4` (z-lab) / `dedf8df`
   (incoai) + SHA256SUMS entry in the pack manifest; document the byte-identity attestation
   (F12) so either mirror satisfies the pin.
2. **No-op confirmations** (0.5 d): assert in the loader that the target head region of the
   `.qwen36sp` pack is BF16/unquantized (parity §6) — fail loudly if a future repack
   quantizes it; mirrors vLLM/SGLang's loud-fail philosophy (adoption-spec §(d)).
3. **Land W1/W8 first exactly as specced** (packer kinds + numpy oracle from the two vLLM
   unit tests) — unchanged by this research.
4. **W2/W3/W5 unchanged**; copy #35496's padded-tail masking detail into W3 notes for any
   future TP>1 drafter packing.
5. **W4 stays "top-K on our own head path"** — now double-precedented (#35496 + llama.cpp
   GGUF). Keep the perf-addendum plan (candidate-list reduction) as the follow-on lever.
6. **New benchmark matrix** (1 d): adopt the community protocol so our numbers become
   cross-comparable — code/reasoning/prose/math content buckets at batch-1 greedy
   (F7/F8 style), plus O512 regression; publish acceptance normalized to upstream's
   per-request-mean definition alongside our E.
7. **Optional head A/B** (deferred): BF16 head (status quo) vs W4-quantized head via W4;
   gate on F11-class give-up ≤ our threshold; do not schedule before the FFN/graph levers
   in handoff §0b/§0e, which dominate the throughput gap.
8. **Ops imports** (free): mem-fraction guardrail ≤0.50-equivalent for unified memory,
   `--disable-flashinfer-autotune`-style determinism flag analog, YaRN-off-for-spec rule,
   K=8-not-9 confirmation — all independently discovered by F7/F8 and matching our env
   surface (`BLOCK_KV=0`, window sweeps).

### DeepSeek-V4-Flash (NO-GO)

- No port. Keep DSpark TP4 line. Watch: z-lab collection + incoai org for a
  `*-DeepSeek-V4-Flash-DFlash2` drop; when it appears, re-run this analysis for tap-layer
  compatibility with the DSV4 resident stage (taps must map onto DSV4's layer indices; the
  qwen38 tap plumbing is not portable as-is).

### GLM 5.2 (NO-GO)

- No port. Keep GLM52 DSpark constants. Same watch condition; also watch for GLM-5.1
  artifacts being retro-fit to 5.2 by third parties (treat any such repo as proposal-grade
  until z-lab/incoai publish it — cf. how many of the 50+ third-party Qwen3.8 DFlash2
  quantizations in the HF listing are unvalidated derivatives).

---

## 8. Source register (all accessed 2026-08-25)

| ID | URL |
|---|---|
| F1/F2 | https://github.com/sgl-project/sglang/pull/35496 |
| F3/F11 | https://github.com/sgl-project/sglang/pull/35825 |
| F4 | https://huggingface.co/api/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead · https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead |
| F5 | https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4 |
| F6 | https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark |
| F7 | https://github.com/sgl-project/sglang/issues/35860 |
| F8 | https://forums.developer.nvidia.com/t/qwen3-8-27b-benchmarking-on-one-dgx-spark-dflash2-beat-vllm-mtp-and-greedy-beat-the-thinking-sampler/380957 |
| F9 | https://www.mindstudio.ai/blog/dflash-2-speculative-decoding-qwen |
| F10 | https://api.github.com/repos/vllm-project/vllm/pulls/52816 |
| F12 | https://huggingface.co/api/models/z-lab/Qwen3.8-27B-DFlash2/refs · https://huggingface.co/api/models/incoai/Qwen3.8-27B-DFlash2/refs · https://huggingface.co/api/models?author=z-lab&sort=createdAt&direction=-1 |
| F13 | https://huggingface.co/api/models?search=dflash2%20deepseek · https://huggingface.co/api/models?search=DFlash%20DeepSeek&limit=20 |
| F14 | https://huggingface.co/api/models?search=dflash2%20glm |
| Context | https://github.com/sgl-project/sglang/pull/35371 · https://inco.ai/blog/dflash2/ · https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2 · https://docs.sglang.io/advanced_features/speculative_decoding.html · https://api.github.com/repos/ggml-org/llama.cpp/pulls/27342 |
