# MISSION.md — Q3F-T1 (mgr2 dispatch, 2026-09-15)

qwen3flash serves ≥14 tok/s correct tokens on the fleet under production law
(lazy, evictable, fail-closed). Its critical path: (1) the reference-engine
port (no T1 fixtures exist — donor pattern lane/wave-refs2 + lane/wave-t1-offline,
adapt the qwen family engine to qwen3flash geometry from the ADOPT-QF
llm_defines.h), fixtures + MANIFEST + negative controls + determinism proof
committed; (2) the fp8.tp8 stamp check (fresh placement, the qwen38max defect
class — verify stamp semantics before decode, one tensor hand-check K3A
pattern); (3) serving stack up (per-boot: the convergence-engine wedge after
one request per boot is known), decode both fixture prompts via the SHARED
weightsd, t1_reference_compare.py → T1 verdict; (4) receipt → PR lane/q3f-t1
→ main, then ONE measured B1 tok/s and STOP.

Measured anchors:
- **FOUND DEFECT (stamp check, BEFORE decode) MEASURED 09-15**: every placed
  qwen3flash.fp8.tp8 pack (16/16, also fp8.tp4pp4 stage3 ranks 12-15)
  stamps expert scales weight_format=4 (FP8_E4M3_F32B128) but the declared
  f32 scale planes contain u16->u32-widened source-safetensors HEADER TEXT
  (144/144 planes/rank, identical on all ranks; f4[0]=2.36e10, plane
  median 6.8e22). Directory stamp/sizing/geometry contract walk is CLEAN
  and the payload codes are sane e4m3 — only the scale planes are broken.
  The in-place repair pass (replaced 4690c141 -> 59872174, receipted after
  the pre-repair verify) corrupted them post-verify. Why the gate missed
  it: qwen4_flash_pack_verify.py forces nvfp4 sample coverage but not fp8;
  on an all-fp8 arm the 8 samples can miss every expert. Receipt:
  runs/q3f-t1/STAMP-CHECK-RECEIPT.md. Consequence: the packs load but
  every routed expert decodes with garbage scales — accuracy-fatal; T1
  proceeds per law (failure is a valid result).
- fp8.tp8 16/16 placed by the finisher wave (v2 format, chattr +i, verify
  gate PASS incl. streaming sha == receipt) — inherited; superseded by the
  finding above at the stamp-semantics level.
- Latent main break found MEASURED: _Static_assert inside the qwen4_flash
  CUDA source (never nvcc-built on main) fails `make archive`; fixed to
  static_assert on this lane (same class as the qwen38max #991 finding).
- qwen3flash ceiling 105.5 tok/s DERIVED (roofline; F7 12.87 GB replicated
  PLE tables).
- qwen3flash.fp8 weights-per-prompt 17.72 GB MEASURED (SMOKE-MD).
- T1 instrument: numpy reference engine (tools/t1_reference_qwen4_flash.py:
  HC 4-stream residual, GDN 36 layers, gated full attention 12 layers, PLE
  n-gram hash at layer 1, 512e top-10 MoE, fp8 expert dequant), family
  llm_defines pin, module T1 dump hook (streams/routes/head-score kernel),
  harness + 8-rank TP8 wave scripts; module archive + harness build green
  on spark7 (MEASURED).
- T1 gate: PENDING (fixtures waiting on the ceph window; decode after).
- Operator directive: accurate inference first, no hillclimb, slow-but-
  reliable collectives acceptable, no fabrication.

Behavioral rules (binding):
1. The critical path above is the default work; everything else justifies why
   it precedes this.
2. A slow-but-accurate state is a diagnostic reference, never a resting state.
3. A fast method proven accurate gets integration effort, not suspicion.
4. A window that doesn't advance the critical path says so explicitly.
5. Every claim graded MEASURED/DERIVED/ASSUMED; ungraded = not a fact.
6. Update MISSION.md + GOALS.md on every significant event — no exceptions.
7. ACCURATE INFERENCE FIRST: divergence = FIRST DIVERGENCE evidence, no
   loosening, no rerun-until-pass; failure is a valid result.
8. CO-RESIDENCY: 6 other accuracy lanes concurrent (~4 rank processes/node);
   touch nothing of theirs; no transport tuning; sparkcap + purges (MemoryMax
   ≥12G or stream on multi-GB pack reads).
