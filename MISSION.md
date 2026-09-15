# MISSION.md — T1-GEMMA4 (gemma4-26b A4B bf16/TP4; retargeted from 31b 09-15)

gemma4-26b A4B serves correct tokens under production law; the fleet tok/s
target is NOT this wave's gate — the operator directive for this wave is
ACCURATE INFERENCE ONLY, slow-but-reliable allreduce acceptable, no perf
work. RETARGET (coordinator, operator correction): this lane is the 26b
A4B variant (the smoke-selected family, 0.93 GB/tok working set), NOT 31b.
Its critical path: (1) bring-up — the family has NEVER been served; locate
the ACC-2-verified 26b TP4 packs (rank0 stage2 FULL 134/134, rank3 66/134),
reconcile placement vs the TP4 replica law (rank = node % 4); (2) fixtures
— REF-PORTS-2 #1017 fixtures are 31b-pinned (MANIFEST checkpoint path
gemma-4-31b-it): port the 26b config through the same reference engine
(family llm_defines carries both variants) and regenerate 26b fixtures
before the compare; (3) T1 verdict — decode the two canonical prompts
through the real serving arm, ids/routing exact + streams in-band, FIRST
DIVERGENCE evidence on any mismatch, exact rates reported separately
(31b greedy chains split at position 2 on a near-tie — verify the same
for 26b); (4) ONE measured B1 tok/s vs the 26b ceiling, then STOP.

Measured anchors: 31b fixtures at lane/wave-refs2 (PR #1017 OPEN)
MEASURED present, MANIFEST pin: checkpoint gemma-4-31b-it, config
e967dd38, fixture SHAs cb80dfaa / 831d0252; OFFLINE GATES PASS MEASURED
09-15 (verify-manifest exit 0; corrupt-fixture negative control convicts
with FIRST DIVERGENCE); reference engine extended for the 26b MoE math
traced from module kernels + stagepack folds MEASURED (code), anchor-tier
validation PENDING warm reads; 26b packs ACC-2 checkpoint-faithful
MEASURED — the 2 verified stage2 packs PLACED 09-15 (spark8 sha
6105b20d..., sparkb sha b1eaa922..., destination==source), 14 packs
lease-gated; module COMPILES after 3 main-compile fixes MEASURED (commit
series 2a6e1b1..f0f51ee): shared combine kernels/timer + register-header
C-guard + validator rewire; GPU validator dataflow tier 28/28 PASS
MEASURED (runs/t1gemma4/validator_dataflow_28checks.log, spark2) incl.
router top-8 lowest-index ties + rope tables + keqv; chain tier SIGSEGV
INSIDE libcuda cuMemcpyDtoH during trivial copies MEASURED 4x (sparka,
spark2x2) — r3's documented quiet-GPU class, NOT fixed in-lane (one
diagnostic sync attempt: store clean, crash moved later = driver-class);
gemma4-26b roofline ceiling 230.4 tok/s DERIVED (R-wave ledger); B1 tok/s
and T1 serving compare: BLOCKED — the publish-gate SIGSEGV is
DRIVER-LEVEL, proven under the dedicated-GPU windows (co-residency
refuted window 1; sanitizer 0 device errors + the real one-element
context[1] validator bug found and FIXED in-lane window 2; crash
persisted post-fix at the same cuMemcpyDtoH site — coredev class per the
operator framing); warm batch queued behind HY4-T1's CEPH_LEASE MEASURED.

Behavioral rules (binding):
1. T1 accurate-inference receipt is the default work; nothing precedes it
   except the co-residency constraints (3 other accuracy lanes live: glm53
   flash TP8, qwen38max TP16, ling TP16 — touch nothing of theirs).
2. A slow-but-accurate state is a diagnostic reference, never a resting
   state — but THIS wave ends at T1 + one B1 measurement by directive.
3. A fast method proven accurate gets integration effort, not suspicion.
4. A window that doesn't advance the critical path says so explicitly.
5. Every claim graded MEASURED/DERIVED/ASSUMED; ungraded = not a fact.
6. Update MISSION.md + GOALS.md on every significant event — no exceptions.
7. A clean bring-up failure reported with evidence is an acceptable
   outcome: capture the exact defect, ONE in-lane fix attempt, then report
   either way. FAIL is honest; no tolerance loosening; no rerun-until-pass.
8. The known converged-engine defect (wedges after ~1 request) is worked
   around per-boot; it is coredev's to fix, reported with evidence.
