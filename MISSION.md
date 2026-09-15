# MISSION — GFULL-T1 (mgr2 dispatch, 09-15)

glm53full (GLM-5.3 full, 78L MLA+DSA MoE-256 on the glm52 module) serves ≥14 tok/s
correct tokens on the fleet under production law (lazy, evictable, fail-closed).
Its critical path: (1) T1 fixture port — glm52-family reference engine on the
fp8.tp16 arm geometry, fixtures + MANIFEST + negative controls + determinism
committed; (2) serving bring-up — DONE TO THE MESH WALL: rank0 reached
residentd READY full-init (first glm53full rank ever; pack gate, contract,
v2 manifest, lazy attach, 6.91 GiB spine all green), ranks ≥1 blocked at the
weightd-mesh client register (tp_device_collective.c:960 cudaHostRegister
EINVAL, lane≥1 only — fleet wall shared with Q3F-T1, routed to mgr2/coredev);
(3) on the mesh fix: relaunch, decode both fixture prompts, compare, ONE B1.

Measured anchors: placed fp8.tp16 arm attachable-v2 MEASURED (SMOKE-MD lease
attach); content checkpoint-faithful MEASURED (ACC-1: bf16 rank5 full 3-leg
PASS, fp8 28-anchor plan-oracle 0 failures; re-proven 09-15 by pack-header
recipe-digest match to /mnt/model-warm/glm-5.3-fp8); per-rank SPINE
7,422,402,048 B MEASURED; rank0 full init MEASURED 09-15 window 3;
T1 fixture coverage ASSUMED-0 → engine committed, fixtures generating;
T1 verdict NOT YET REACHED (blocked on the mesh register wall, no loosening).

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
