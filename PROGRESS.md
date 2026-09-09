# PROGRESS — gemma4 driver lane

Branch `lane/gemma4-driver` @ origin/main 8f3a6f2. DESIGN.md untracked (per brief).
Working dir: /Users/mac/batch-gemma4. Never pushed; manager owns GitHub.

## Rulings applied (from manager brief)

1. 31B = TP4xPP4 (16 sparks), 26B = TP4xPP2 (8 sparks), packer KV-head
   duplication on full-attention layers (per-rank KV_HEADS=1), port base 65100.
2. Zero frozen-file semantic edits. muse `LmHeadRmsNormKernel` and laguna
   `LmRopePerHeadKernel` +3 trailing params consumed; byte-exact copies when
   not yet on main.
3. Family-local kernels (gated gelu-tanh, scaled sharded embed gather) stay
   family-local.
4. ~90% qwen4_flash donor deletion IS the design.
5. nvfp4 arm = seam only.
6. Contracts ship pending-warm-download revision pins + fail-closed adapters.
7. port_family.py copied from /Users/mac/batch-ling (not on main).

## Shared-kernel landing status (surveyed)

- `LmHeadRmsNormKernel`: NOT on main norm.cuh (470 lines). On muse branch at
  inference/kernels/norm.cuh:497 with `weight_or_null` semantics
  (`weight_bf16 != 0` check) + bf16-round-then-multiply epilogue. Muse adds it
  AFTER LmCenteredRmsNormKernel; gemma copies ONLY LmHeadRmsNormKernel byte-exact
  (ruling: gemma does NOT need LmCenteredRmsNormKernel).
- `LmRopePerHeadKernel` extended signature: NOT on main. Main project.cuh:428 has
  6-arg; laguna branch project.cuh:428 adds `(const float *inv_freq_table = 0,
  float attention_scale = 1.0f, uint32_t rope_offset = 0xffffffffu)` and laguna
  attn.cuh adds defaulted `scale` params on LmRopePair/LmRopeRotate (required by
  the attention_scale epilogue). Both defaulted — existing main call sites
  compile unchanged. Copy byte-exact, converges at merge.
- DESIGN.md says attn.cuh:50; actual extended kernel lives in project.cuh:428 on
  the laguna branch. Coded against the real signature (design anticipated drift;
  one call site to rebase at merge).

## Acceptance criteria status

| AC | status | evidence |
|---|---|---|
| 1 port + deletion | IN PROGRESS | |
| 2 both arms compile | PENDING | |
| 3 header bindings test | PENDING | |
| 4 stagepack + synth | PENDING | |
| 5 oracle | PENDING | |
| 6 CUDA validation tier | PENDING | |
| 7 offline gates | PENDING | |
| 8 blocked-on-download | BLOCKED (by design) | warm dirs not landed yet |
