# MISSION — GFULL-T1 (mgr2 dispatch, 09-15)

glm53full (GLM-5.3 full, 78L MLA+DSA MoE-256 on the glm52 module) serves ≥14 tok/s
correct tokens on the fleet under production law (lazy, evictable, fail-closed).
Its critical path: (1) T1 fixture port — glm52-family reference engine on the
fp8.tp16 arm geometry, fixtures + MANIFEST + negative controls + determinism
committed; (2) serving bring-up — family gen deployment, 16 ranks, shared
weightsd lease, decode both fixture prompts; (3) t1_reference_compare verdict,
receipt on PR lane/gfull-t1, then ONE measured B1 tok/s.

Measured anchors: placed fp8.tp16 arm attachable-v2 MEASURED (SMOKE-MD lease
attach); content checkpoint-faithful MEASURED (ACC-1: bf16 rank5 full 3-leg
PASS, fp8 28-anchor plan-oracle 0 failures); per-rank SPINE 7,422,402,048 B +
EXPERTS 46,714,060,800 B lazy VMM-reserved MEASURED (R2 dispatch, 16/16);
glm53full WS 11.13 GB incl. spine MEASURED (SMOKE-MD capstone, first WS);
T1 fixture coverage for glm53full ASSUMED-0 until this wave commits it; glm52
module serving bring-up NEVER DONE (LoadDriver rc=3 gap = the bring-up risk).

Behavioral rules (binding):
1. T1 fixture port + serving decode is the default work; everything else
   justifies why it precedes this.
2. A slow-but-accurate state is a diagnostic reference, never a resting state.
3. A fast method proven accurate gets integration effort, not suspicion.
4. A window that doesn't advance the critical path says so explicitly.
5. Every claim graded MEASURED/DERIVED/ASSUMED; ungraded = not a fact.
6. Update MISSION.md + GOALS.md on every significant event — no exceptions.

Co-residency: 6 other accuracy lanes run concurrently. Touch nothing of theirs.
Slow-but-reliable collectives. No fabrication. No comments in code. All
git/gh through tools/sparkpipe_github_pat.sh (identity sparkpipe, verified).
