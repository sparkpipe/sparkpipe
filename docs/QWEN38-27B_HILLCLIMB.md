# Qwen 3.8 27B — hill-climb mandate (speculation diagnosis, checkpoint pin, 1-spark plan)

Lane: qwen38-27b. Companion to SURVEY_QWEN38-27B.md. No host experiments run
yet (fleet_swap conversion in flight); all claims grep/read-verified against
this clone at HEAD afb43a8, and spark3 was read-only probed via ssh.

---

## A. Why speculation is broken (root cause + fix plan + code-size)

Speculation = the MTP 3-frame chain (decode-draft -> verify -> replay) in the
qwen38-27b driver (still the qwen38_27b module). Four findings, ranked:

**A1 — the build gate never exercises the speculation path (process root cause).**
The TP4 build publishes the module with MTP and GDN-snapshot DISABLED:
`qwen38_tp4_build.sh:27` passes `MTP_LAYER_COUNT=0 GDN_SNAPSHOT_SLOT_COUNT=0`,
which `modules/qwen38_27b_resident_decode_stage/Makefile:66-67` maps to
`SPARK_QWEN38_27B_STAGE_MTP=0` + `SPARK_QWEN38_27B_STAGE_GDN_SNAPSHOT_SLOTS=0` for the
publish/GPU-validator run. Deploy then arms both: `qwen38_tp4_deploy.sh:34`
sets `SPARK_QWEN38_27B_STAGE_MTP=1` and the serving adapter forces
`SPARK_QWEN38_27B_STAGE_MTP=1` + `SPARK_QWEN38_27B_STAGE_GDN_SNAPSHOT_SLOTS=8`
(`spark_qwen38_27b_serving_adapter.c:384-388`). So
`SparkQwen38_27bModuleRunMtpDraftChain` and `SparkQwen38_27bModuleGdnSnapshot`
(`spark_qwen38_27b_resident_decode_stage_module.c:1587,1445`) are the ONE code path
the GPU gate never runs — it ships unvalidated, and "broken" is the expected
outcome.

**A2 — the chain-dead self-consistency gate silently zeroes speculation.**
`spark_qwen38_27b_serving_adapter.c:1263` sets `chain_dead = draft_ids[0] !=
committed_ids[0]` — the MTP's first draft must EXACTLY reproduce the main
model's committed token. On any miss, `:1303-1304` force `min_accepted=0` and
the lane commits one token with zero credit. Because speculation is B1-only
(`:1414`, `active_sequence_count == 1u`), a single first-draft miss = no
speedup for the whole step. The intent (`:1259-1262`) is sound (a wrong
draft[0] poisons the verify), but it means the chain stands or falls on
draft[0]==C0, and there is no fallback re-draft.

**A3 — latent OOB + fragile hand-rolled bookkeeping.**
`spark_qwen38_27b_serving_adapter.c:1324` writes `replay_tokens[min_accepted+1u]`
into an 8-element array while `min_accepted` can reach `draft_count-1u` (up
to 8; `:1292`) -> index 8 out of bounds. Not hit at D=2 (the deploy config)
but proves the hand-rolled 3-frame chain (accept loop `:1280`) is fragile and
should be replaced, not patched.

**A4 — MTP is the wrong drafter, and ours is the worst variant.**
The survey's FP8 recipe (0xBakeer) measures MTP cost/draft-token 0.153 vs DSpark
0.046 (3.3x cheaper) because MTP re-runs a full lm_head projection per draft.
Our `SparkQwen38_27bModuleRunMtpArgmaxRow` does exactly that — a full
screened-argmax lm_head pass per draft step (`spark_qwen38_27b_resident_decode_stage_module.c:1556-1576`).
So even after A1-A3, our MTP D=2 ceiling is capped well below DSpark.

**Fix plan + code-size.**
1. (immediate, ~5 lines) Re-arm MTP + snapshots in the build gate
   (`qwen38_tp4_build.sh:27` -> `MTP_LAYER_COUNT=1 GDN_SNAPSHOT_SLOT_COUNT=8`)
   so the GPU validator exercises the chain; bound the A3 index.
2. (medium, ~-60 lines) Adopt the shared verifier + neutral tree
   (`inference/kernels/speculate.cuh` + `spark_speculation_tree.h`) to replace
   the hand-rolled accept loop and 3-frame bookkeeping (TOP10 #5).
3. (better, ~+200 lines) Add a DSpark (separate 5-layer) drafter — the
   survey's winning path (20-58 tok/s FP8 single-stream) — instead of
   deepening MTP.

**A5 — measured verdict (B1 TP1, memory-bound GB10, 2026-08-18).**
The async phase-one completion (option a: make the MTP_DRAFT_AFTER submit
non-blocking so decode-draft(N+1) overlaps verify(N)+replay(N) under the
residentd admit-first order) is CORRECT — token stream bit-identical — but it
is a NO-OP: spec D=2 = 3.46-3.47 tok/s vs 3.50 no-async vs 3.69 no-spec. The
GPU is memory-bound (rung-2 profile: FFN 65.4% + GDN 23.2% = 88.6%), so two
in-flight walks cost the same wall time as two serial walks: there is no
compute idle to reclaim and the host-side ~82ms/token is already pipelined.

VERDICT: the overlap/2-in-flight direction is DEAD for this hardware. MTP D=2
cannot pay — its intrinsic cost is ~3 full-model walks per ~3 committed tokens
(≈ 1 walk/token, same as no-spec) plus MTP + verify/replay orchestration. The
only spec path that pays is a HIGHER-ACCEPTANCE draft source (DSpark-class),
which is the rung-3 gate. The async changes stay UNCOMMITTED in the clone as
the record and are NOT landed. Lane redirects to rung 2 (tiled FP8 decode)
now: the 7.88 reference implies their FP8 kernels are ~2x ours, which is
exactly a tiled-dequant difference.

**A6 — rung-2 FP8 decode landed + spec D=2 re-measured (2026-08-18).**
Rung 2 (native e4m3x4 decode, landed 9cbe4d5): the FFN/GDN F32B128 dot row
called exp2f() per e4m3 byte, making it compute-bound; switching to the SM121
native `cvt.rn.f16x2.e4m3x2` (already shipped for the E8M0 path) removed it,
bit-losslessly (e4m3 subset of f16). Measured no-spec B1: 3.69 -> **8.00
tok/s** (HWM c729071), exceeding the 7.88 reference; token stream bit-identical,
validator bit_exact=1.

Spec D=2 re-armed on the 8.00 base: **7.16 tok/s** (3.35s/24, submitted=8 for
24 tokens = ~3 committed tokens/iteration, first_draft_miss=2 unchanged). So
D=2 rose 2.05x (3.50 -> 7.16) but is STILL BELOW no-spec 8.00 — the 3-walk
verdict HOLDS. Clarifying the arithmetic: "3 walks" is per ITERATION, and each
iteration commits ~3-4 accepted tokens, so the intrinsic cost is ~1 full-model
walk per committed token (same as no-spec); the MTP lm_head passes (a full
vocab-248320 screened argmax per draft, A4) + verify/replay orchestration are
what keep D=2 below plain decode. Only a higher-acceptance DSpark-class
drafter (rung 3) can make spec beat no-spec.

## B. Checkpoint id + revision (read from spark3, no downloads/launches)

spark3 (hostname `aitopatom-a18f`) runtime `/home/spark3/sparkdata/
qwen38.bf16.tp4` holds one pack: `packs/tp4-rank3.qwen38_27bsp` (16,059,167,232 B).
Its 120-byte header decodes to: magic 1347630673, format 3, tensor_count 866,
hidden 5120, layer_count 64, GDN 16 key/48 value heads, attention 24 query/4 KV
heads x 256, FFN intermediate 17408, vocab 248320, mtp_layer_count 1, tp 4/rank 3.

The rank config `config/qwen38_27b_tp4_rank3.json` pins `model_revision =
"bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1"`, which matches
`Makefile:161` (`QWEN38_27B_MODEL_REVISION`). The upstream model is
`Qwen/Qwen3.6-27B` (`tools/qwen38_27b_stagepack.py:2`;
`model_contracts/qwen38_27b_authoritative.json:3`).

**Conclusion — the pin is still OPEN, and the runtime is mis-named.**
The deployed weights are Qwen 3.6 27B BF16 (geometry above), NOT Qwen 3.8 27B.
No Hugging Face git revision is recorded anywhere in the deployment: the pack
header carries only geometry (no revision field), the rank config only the
internal string, and the checkpoint source dir is absent from spark3/spark0
(no `*qwen*` under `/home/spark{0,3}/extnvme`). The long-open checkpoint pin
requires capturing the HF revision of `Qwen/Qwen3.8-27B` at re-pack time
(the 3.8 27B checkpoint has never been packed for this runtime).

## C. 1-spark (spark3, TP1) hill-climb experiment plan

Target: close the ~20%-behind-SOTA B1 gap on one GB10. Reference floor: BF16
single-spark weight stream = 54.5 GB / 273 GB/s = ~5.0 tok/s; FP8 SOTA = 7.88
tok/s (survey). Order is baseline-first, then the cheapest output-preserving
win, then precision, then the drafter.

| # | Measure on spark3 (TP1) | Expected gain | Closes vs the 20% gap |
|---|---|---|---|
| 0 | B1 no-spec baseline (BF16, D=0). Record tok/s + per-phase ms (head, GDN, collectives, launches). | establishes the real gap | the anchor number |
| 1 | Re-arm the build gate (A1), re-run B1 with MTP D=2. | ~1.6-2.4x (SOTA MTP k3) -> ~8-12 tok/s | overshoots the gap if it works; verifies A2/A3 |
| 2 | Sweep MTP D=2..8; read acceptance per depth. | find the k knee (SOTA k3 fresh / k8 edit) | picks the right operating point |
| 3 | FP8 weights (vendor 28.5 GiB) -> re-pack + re-measure B1 no-spec. | floor ~5.0 -> ~9.5 tok/s; target 7.88 | closes the no-spec gap at matched precision |
| 4 | DSpark 5-layer drafter (A4) -> B1 spec. | 20-58 tok/s (survey) | the ceiling, output-preserving |
| 5 | Lossless BF16 codec + head screen + GDN fusions (TOP10 #7/#9). | +25-30% (codec) + ~1-3 ms (trims) | the BF16-only headroom |
| 6 | CUDA-graph steady-state decode (no eager). | launch overhead -> floor (lastloop 24->75) | riskiest, highest ceiling |

Gate after every step: the module GPU validator +
`Qwen38-TP4-E2E-PASS` (per-rank agreement, 64-token reference, bit-identical),
mirroring `QWEN38_27B_TP4_PERF.md:67-68`. Do not stack steps until the prior one
beats the retained baseline (the TECHDEBT acceptance rule).

## What I need
- spark3 cleared for a measured TP1 window (fleet_swap conversion first), per
  the mandate.
- The exact `Qwen/Qwen3.8-27B` HF checkpoint id+revision (from whoever holds
  the source) so B's pin closes at re-pack time.

---

## D. DSpark drafter parity — RESOLVED (kernel correct, numpy reference had a rope bug)

Standing bar: our 7 draft tokens == vLLM's 7 given the same taps. Status: MET.

The harness (`tools/qwen38_27b_dspark_parity.cu`, links the real `.cu` kernels)
produced `[220,16,92,198,12,328,82]`; the numpy oracle
(`tools/qwen38_27b_dspark_reference.py`) produced `[220,17,11,748,874,4799,13]`.
A multi-turn hunt (embedding gather, rope position, Q/K RMSNorm row indexing,
attention accumulation) read clean — all four were correct in the kernel.
A per-layer/per-position intermediate dump localized the first divergence to
the layer-0 ATTENTION output, and an independent numpy reimplementation of the
KERNEL's arithmetic matched the CUDA output to ~1 ULP. That proved the kernel
was right and the ORACLE was wrong.

**Deviant line** — `tools/qwen38_27b_dspark_reference.py` `apply_rope()` had an
in-place numpy view aliasing bug:

    xr = out[..., 0:ROPE_DIM:2]          # a VIEW, not a copy
    xi = out[..., 1:ROPE_DIM:2]
    out[..., 0:ROPE_DIM:2] = xr * c - xi * s   # overwrites even slots
    out[..., 1:ROPE_DIM:2] = xr * s + xi * c   # xr now reads ROTATED even values

The odd channels were computed from already-rotated even values. The CUDA
kernel is correct (reads re/im into registers before overwriting). 90-degree
probe: view-bug → [-2,-2,-4,-4]; correct → [-2,1,-4,3]. Fixed by `.copy()` on
`xr`/`xi`; the fixed reference now emits `[220,16,92,198,12,328,82]` == kernel.

Residual (not a bug): the kernel keeps fp32 after the q/k head-norm where the
reference truncates to BF16, so layer-0 attention still differs by ~1 ULP —
below the argmax. Position 1 is a genuine tie (tokens 17 and 16 both 19.625
base+markov), broken to 16 by argmax-first-max; not a margin flip.

vLLM oracle: not needed to settle this (optional structural confirmation still
on the table; blocked on the `Got unsupported ScalarType BFloat16` config leak).
Do NOT re-attempt HF `trust_remote_code` — post-fla/flashinfer install the
drafter's `dspark.py` import deadlocks (0% CPU, `Ss` state).

---

## E. DSpark-on-FP8 spec verdict — CANNOT PAY at B1 (final)

Measured on the FP8 target (29.9 GB, B1, O128, k=7) after three landed fixes:
C0 anchor (block[0]/Markov-prev consume the committed token, not the frame input),
draft remap (DSpark positions C0+i+1 folded into the MTP verify convention), and
the GDN-snapshot submit fix. No-spec HWM is 8.02 tok/s.

| config | tok/s | accepted/k | first-position |
|---|---|---|---|
| no-spec (FP8) | 8.02 | — | — |
| spec, C0-fix + remap (correct wiring) | 5.08 | 0.735 (10.5%) | 44% |
| spec, tap re-wire (capture on VERIFY row 0) | 4.58 | 0.342 (4.9%) | 21% |

VERDICT: **DSpark-on-the-FP8-target cannot pay at B1, same as MTP D=2.** The C0
anchor + draft remap were real fixes (5.3% -> 10.5% acceptance, 4.596 -> 5.078
tok/s), but the tap re-wire — capturing the target's hidden on the
SPECULATIVE_VERIFY frame's row 0 (the committed token) instead of the DRAFT frame's
input row — makes acceptance WORSE (10.5% -> 4.9%): the one-iteration-late tap is
further from the DSpark's trained input than the DRAFT-frame capture of the last
committed token's hidden. The ~10.5% accepted/k is the drafter's real quality on
real inputs, well short of the ~25%+ needed to beat 8.02 at k=7.

Provenance correction: the drafter (`RadixArk/Qwen3.8-27B-DSpark`) is trained
against Qwen3.8-27B-**FP8**, which IS our target — the earlier GPTQ-Int4 premise is
inverted, so there is no Int4 pack to obtain (Int4 is the perturbed-taps repack at
2.45-2.79 acceptance length, FP8 is the 3.39 canonical). The remaining lever is the
sampling method: the authors' 3.39 acceptance length is under PROBABILISTIC
sampling, while this port runs greedy argmax (their own data: greedy is slower).
The port stays env-gated (`SPEC_METHOD=dspark`, optional pack) as the platform for
that acceptance work.
