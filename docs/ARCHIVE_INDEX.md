# Archive index — docs/archive/ (the one active window into the archive)

Everything here is COMPLETED, SUPERSEDED, or HISTORICAL. Active docs
live in docs/ — if you are reading this to find whether something was
already decided, it probably was, here. (Move entries OUT only when a
doc becomes active again; add a one-liner here on every new archive.)

## Completed programs (their outcomes are in the tree + the ledger)
- CLEANUP_PROGRAM.md, CODEBASE_CLEANUP_PLAN.md, DRY_CONSOLIDATION_PLAN.md,
  HOUSECLEANING_PLAN.md — the audit-response waves: all merged, gates live.
- PERF_PROGRAM.md — v1; superseded by PERF_PROGRAM2 (active) after P3's
  measured-negative verdict corrected its premises.
- PACKER_CORE_PLAN.md — realized via the shared synthesize core (wave-1 DRY).

## Design docs whose implementations landed
- DSPARK_DSV4_FLASH_DESIGN.md — landed; the dspark gate + lease fixes.
- DSV4_FLASH_TP4_PP4.md, K3_PACK_FORMAT_V2.md, K3_WEIGHT_ONLY_MXFP4.md,
  K3_TP16_REPACK.md, K3_TP4PP4_PREP.md, K3_GATE_RECONCILIATION.md —
  landed (packs built/deployed; K3 gate state in the lane reports).
- GLM53_FLASH_KERNEL_ASSESSMENT.md — mission accomplished (glm5_next
  serving; the hunt's history is in the lane reports + coordinator log).
- PROPOSAL_KV_SEAM.md — superseded by JIT_KV_RESPONSE (active).
- PAIRED_DUAL_LINK_ALLREDUCE.md, SPARK_HOST_RDMA_DOORBELL.md (kept active
  in error? see name below if archived later) — transport experiments,
  outcomes in the collective's current shape.

## Measurements/receipts (historical records, one-time)
- DSPARK_* (5 receipts/runbooks), P1P2_*, RESIDENTD_B1_PROFILE,
  HOST_SIDE_B1_BREAKDOWN, QWEN36_TP4_PERF, QWEN38-27B_HILLCLIMB,
  QWEN38_MAX_* (the 8-audit series), SERIAL_TP16_K3* (3),
  TOP10_* (8), SURVEY_* (6), PERF_DASHBOARD, SPEC_DECODE_REFERENCE_*
  (later VLLM_SGLANG_*), serial_tp_replay.

## Superseded proposals (decision recorded, alternative chosen)
- PROPOSAL_ADMISSION_CORE, BOOT_UNBLOCK, DSV4_PRO_* (3), TREE_ADOPTION,
  GLM52_JIT_KV_MIGRATION, GLM52_PAGE_TABLE_DATAFLOW, RING_WINDOW_HOOK,
  RUNG3_DSPARK_ADOPTION, QWEN36_TO_QWEN38_RENAME, BACKLOG,
  CLIENT_B1_BUBBLE, DFLASH2_* (3), CODEX_RUNBOOK, ds4-parallel-pxe-*.

## Operational history (incident/playbooks superseded by the active
 INCIDENT_RECOVERY_PLAYBOOK + the coordinator log)
- GLM52_B12X_* (2), GLM52_SM121_*, QWEN38_B16_INCIDENT, TP1_RUNBOOK,
  plus everything already in docs/archive predating this index.
