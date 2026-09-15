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
- fp8.tp8 16/16 placed by the finisher wave (v2 format, chattr +i, verify
  gate PASS incl. streaming sha == receipt) — inherited, to be re-verified
  at stamp-semantics level by this lane.
- qwen3flash ceiling 105.5 tok/s DERIVED (roofline; F7 12.87 GB replicated
  PLE tables).
- qwen3flash.fp8 weights-per-prompt 17.72 GB MEASURED (SMOKE-MD).
- T1 gate: PENDING this wave. No fixtures existed before this wave.
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
