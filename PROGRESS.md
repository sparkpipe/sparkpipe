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

## Topology re-examination (operator ruling, mid-implementation)

TP16 preferred (weights must shard 16x or sparks get different RAM). Re-checked
both models with the muse kv-head replication precedent (muse DESIGN.md:186-188,
296-299: ranks 0-7 replicate kv head 0; all sharers bitwise-identical because the
residual stream is identical post-reduce and reduce order is deterministic).

Verdict per model at TP16:
- 31B: q 2/rank; sliding kv 16/16 = 1/rank exact (no replication); full kv 4
  replicated x4 (ranks 0-3 head 0 ...); per-rank KV_HEADS=1. hidden 336/rank,
  vocab 16384/rank, all exact. Switched: TP16 x PP1, single stage, 16 sparks.
- 26B: q 1/rank; sliding kv 8 replicated x2; full kv 2 replicated x8; experts
  128/16 = 8/rank; hidden 176/rank; dense 2112/16 = 132. Per-rank ~3.2 GiB
  uniform (fixes the TP4xPP2 12.6 GiB outlier).
- gqa.cuh (frozen, read): store static_assert kSlotBytes ==
  KV_HEADS*(HEAD_DIM+VALUE_DIM)*2 -> 2048 B (full, 512+512) / 1024 B (sliding,
  256+256) at KV_HEADS=1; decode guard heads>=KV_HEADS && heads%KV_HEADS==0
  trivially holds; kv_head degenerates to 0 = the single stored head.
- k_eq_v determinism: kraw = k_proj(h) column-parallel on bitwise-identical
  post-reduce residual -> identical kraw on sharers; v = v_norm(kraw) is a
  per-head-local no-weight RMS (fixed THREADS block sum, no atomics) ->
  identical v bytes. Replication identity holds exactly as muse's case. The
  v_norm-before-k_norm ordering is per-rank-local and identical across sharers;
  a wrong order is a correctness bug (design risk 2), not a determinism seam.
- 31B session-port cells: 65100 + src*16 + sink -> 65100..65355, inside uint16.

Port allocations: ALL ON HOLD — pending fleet renumber (manager escalation).
Verified constraint (ring/transport/tp_device_collective.c:29-31,601-615):
session cells carry route-kind offsets D2A +256 / TREE_ACK +512 / D2A_ACK
+768 with a hard 65535 rejection, so a degree-16 matrix needs ~1024 ports of
clearance. 31B@65200 and 26B@65460+ would overflow their ACK variants;
26B@62000 fits its own variants but its ACK/D2A zones graze k3's 62700 control
base and the legacy 62500 mark. ACTION TAKEN: every gemma session base is
env-driven with NO frozen default — the deployment generators fail closed when
the base env is unset; the module already required SPARK_GEMMA4_STAGE_TP_*
envs at configure (fail-closed, donor behavior). Port tables are marked
"pending fleet renumber"; topology (31B TP16xPP1, 26B TP4xPP4 fallback w/
TP16 table ready) is unchanged — only the table numbers wait. The region
above 64700 cannot host degree-16 fleets at all; coredev owns the renumber.

Width-aware ledger as known (base..end: fleet; session width = TP^2, plus
+256/+512/+768 route-kind zones for the tree backend): 60700/60710 transport
misc; 61500..61515 glm5_next TP4 session (61500..61755 if TP16); 62000..62255
gemma 26B TP16 candidate (ON HOLD: D2A zone grazes legacy 62500, ACK zone
grazes k3 62700); 62550..62805 glm5_next hc; 63500..63563 glm53full TP8;
63620 glm52 listen; 63640 glm5_next listen; 63700 glm53full listen; 64500
glm52 TP16 session zone + qwen4_flash listen/cells tangle (pre-existing);
64630 k3; 64800..64815 minimax TP4 (manager moving); 64800..64955 laguna TP8
session; 64900..65155 ling TP16 session; 65200+ structurally unusable for
degree-16 (ACK overflow past 65535).

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
