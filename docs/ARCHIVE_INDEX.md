# Archive index — docs/archive/ (the one active window into the archive)

Everything here is COMPLETED, SUPERSEDED, or HISTORICAL. Active docs
live in docs/ and are indexed by [README.md](README.md) — if you are reading
this to find whether something was already decided, it probably was, here.
(Move entries OUT only when a doc becomes active again; add a one-liner here
on every new archive.)

The 2026-09-28 documentation cleanup moved 60 files here; each carries its
reason below. Two more were deleted instead of archived (end of this file).

## Retired root documents (2026-09-28)
- ARCHITECTURE.md — its unique content was merged into the root README.md.
  It contradicted README on topology (TP4xPP4 canonical), collectives
  (recursive doubling and split rings) and the model set.
- STRUCTURE.md — its corrected directory table is now README.md's Repository
  section. It listed a nonexistent `api/`.
- COORDINATION.md — the six-session era (2026-08-16): fleet_swap.sh, ring
  windows and port blocks are all obsolete, and its Qwen directive was
  corrupted by the rename in 33c69b6. The root COORDINATION.md is now a
  pointer to FLEET_RELEASE_RUNBOOK.md.

## Completed programs (their outcomes are in the tree and PERFORMANCE_STATUS.md)
- CLEANUP_PROGRAM.md, CODEBASE_CLEANUP_PLAN.md, DRY_CONSOLIDATION_PLAN.md,
  HOUSECLEANING_PLAN.md — the audit-response waves: all merged, gates live.
- PERF_PROGRAM.md, PERF_PROGRAM2.md — perf programs v1 and v2 (2026-08-30).
  R1-R3 and the correctness must-fixes landed (BUG_LEDGER.md); R4-R7 are
  open in TECHDEBT.md. PERF_PROGRAM2.md moved here on 2026-09-28.
- PACKER_CORE_PLAN.md — realized via the shared synthesize core (wave-1 DRY).
- BUG_LEDGER.md (2026-09-28) — superseded by TECHDEBT.md; the items still
  open on 2026-09-28 were carried there.
- CONSTANT_AUDIT.md (2026-09-28) — violations fixed in c3cc5d1; the law now
  lives in COMMON_MODULE_ARCHITECTURE.md §4.

## Design docs whose implementations landed
- DSPARK_DSV4_FLASH_DESIGN.md — landed; the dspark gate + lease fixes.
- DSV4_FLASH_TP4_PP4.md, K3_PACK_FORMAT_V2.md, K3_WEIGHT_ONLY_MXFP4.md,
  K3_TP16_REPACK.md, K3_TP4PP4_PREP.md, K3_GATE_RECONCILIATION.md —
  landed (packs built/deployed; K3 gate state in the lane reports).
- GLM53_FLASH_KERNEL_ASSESSMENT.md — mission accomplished (glm5_next
  serving; the hunt's history is in the lane reports + coordinator log).
- PROPOSAL_KV_SEAM.md — superseded by JIT_KV_RESPONSE (landed 2026-08-29
  as cache/kv_pager.c over cache/nvme_tier.c; archived).
- JIT_KV_DESIGN.md, JIT_KV_RESPONSE.md (2026-09-28) — landed in 2927f82,
  960e2ee, e258190, a26829a and 21406f9; W3 is still open in TECHDEBT.md.
- SPARKPIPE_SERVING_CHARTER.md (2026-09-28) — continuous batching (c4e7fb4),
  the native tokenizer, and model_api streaming and deadlines (1dd7bfa)
  landed.
- SCHEDULER_SUBSYSTEM_BOUNDARY.md (2026-09-28) — the admission core
  (4cfa095) and the shared ladder (7ecf1f5) landed; §2.4 moved to
  DRIVER_ACCEPTANCE.md; the obsolete fleet_swap sections were removed.
- WEIGHTD_MESH.md, WEIGHTD_MESH_IMPL.md (2026-09-28) — pre-implementation
  mesh plans; the shipped mesh is in WEIGHTD_DESIGN.md.
- WEIGHTD_RANGE_LEASE_WORK.md (2026-09-28) — merged; its contract is folded
  into WEIGHTD_DESIGN.md "Expert residency".
- WEIGHTD_RESIDENT_CACHE_DESIGN.md (2026-09-28) — S1 done, S2 partial, S3
  and S4 not implemented.
- GLM_KDA_STATE_LAYOUT.md (2026-09-28) — its formula is now in
  GLM5_NEXT_ROOFLINE.md's sharding table.
- GEMMA4_DRIVER_DESIGN.md (2026-09-28) — landed with a different placement:
  31B at TP16, 26B at TP4xPP4.
- DEPLOY_MULTI_DEV_PROPOSAL.md (2026-09-28) — implemented in the fleet agent
  and publish_local.
- CONSULT_multirow_kv_nondeterminism.md (2026-09-28) — resolved in 0a454a3;
  multi-row prefill serves (#1210).
- SPECULATION_SUBSYSTEM_BOUNDARY.md (2026-09-28) — the 08-17
  "unified"-branch inventory and agent-lane ownership; its steps landed or
  were superseded. The live inventory is SPECULATION_UNIFIED_DESIGN.md.

## Measurements/receipts (historical records, one-time)
- DSPARK_* (5 receipts/runbooks), P1P2_*, RESIDENTD_B1_PROFILE,
  HOST_SIDE_B1_BREAKDOWN, QWEN36_TP4_PERF, QWEN38-27B_HILLCLIMB,
  QWEN38_MAX_* (the 8-audit series), SERIAL_TP16_K3* (3),
  TOP10_* (8), SURVEY_* (6), PERF_DASHBOARD, SPEC_DECODE_REFERENCE_*
  (later VLLM_SGLANG_*), serial_tp_replay.
- T1_QMAX_RUN_RECEIPT.md (2026-09-28) — a Qwen 3.8 Max T1 run receipt, next
  to QWEN38_MAX_*; its TP16 blocker is fixed.
- STAGEPACK_AUDIT_2026-08-31.md (2026-09-28) — a dated audit: the
  qwen38.fp8.tp1 deletion and the pack identity baseline.
- PERFORMANCE_LEDGER.md (2026-09-28) — the August scoreboard ledger. Its
  rows with a measurement identity were folded into PERFORMANCE_STATUS.md on
  2026-09-28, and tools/perf_scoreboard.py was removed.
- PERFORMANCE_HISTORY_2026-08.md (2026-09-28) — moved out of
  PERFORMANCE_STATUS.md: the DSV4 Flash TP4 B1 hill-climb receipts, the TP4
  residentd-transport crossovers, the DSV4-based TP8/TP16 projection and the
  GLM 5.2 TP8 measurements.
- NCCL_16WIDE_RECEIPTS.md (2026-09-28) — 2026-08-31 NCCL 16-rank latency
  (~103 us at 14 KB).
- TP16_HARDWARE_PROFILE_20260922.md (2026-09-28) — 2026-09-22 TP16 CUPTI and
  serving receipts; the files are docs/receipts/glm5-next-tp16-perf-*.json.
- GLM_FLASH_HILLCLIMB.md (2026-09-28) — the GLM component qualification
  log, stopped 2026-09-09; superseded by GLM5_NEXT_ROOFLINE.md and
  GLM_PERFORMANCE_GATES.md. Its "Changes and lessons so far" table (PRs
  #842-#864, one lesson per fix) is still worth reading.
- GLM_KDA_REFERENCE.md (2026-09-28) — the PR #864 write-up.
- GLM_KV_PAYLOAD.md (2026-09-28) — the PR #862 write-up.
- GLM_NUMERICAL_GATES.md (2026-09-28) — the 2026-09-08 gates; the rule now
  lives in GLM_PERFORMANCE_GATES.md.

## Superseded proposals (decision recorded, alternative chosen)
- PROPOSAL_ADMISSION_CORE, BOOT_UNBLOCK, DSV4_PRO_* (3), TREE_ADOPTION,
  GLM52_JIT_KV_MIGRATION, GLM52_PAGE_TABLE_DATAFLOW, RING_WINDOW_HOOK,
  RUNG3_DSPARK_ADOPTION, QWEN36_TO_QWEN38_RENAME, BACKLOG,
  CLIENT_B1_BUBBLE, DFLASH2_* (3), CODEX_RUNBOOK, ds4-parallel-pxe-*.
- PAIRED_DUAL_LINK_ALLREDUCE.md, SPARK_HOST_RDMA_DOORBELL.md — transport
  designs never built as written; the weightd mesh (WEIGHTD_DESIGN.md, Mesh
  substrate and rendezvous) replaced them. SPARK_HOST_RDMA_DOORBELL.md moved
  here on 2026-09-28.
- NCCL_COLLECTIVE_PLANE.md (2026-09-28) — the NCCL backend was deleted in
  b31761e; only hidden_transport is accepted.
- UNIVERSAL_MESH.md (2026-09-28) — merged into WEIGHTD_DESIGN.md (Mesh
  substrate and rendezvous).
- ROADMAP_TP16_FLEET.md (2026-09-28) — superseded by ROADMAP.md; its
  placement lists from 2026-08-30 are stale.
- MESH_INVENTORY.md (2026-09-28) — a per-topology mesh design that was
  never built.
- FLEET_STARTUP_PROTOCOL.md (2026-09-28) — a registrar design; the
  production fleet-agent has no registrar.
- UNIVERSAL_PACKER.md (2026-09-28) — folded into DRY_PACKBUILDER_PROPOSAL.md.
- KVCACHE_SUBSYSTEM_BOUNDARY.md (2026-09-28) — the live KV contract is
  DRIVER_ACCEPTANCE.md plus the README KV section.
- SPECULATION_PROVIDER_DESIGN.md, SPECULATION_TREE_COMPOSITION_DESIGN.md
  (2026-09-28) — the provider slot and the node-level compositor were
  superseded by the speculation seam and the DFT3 draft bridge (#786/#788);
  the still-valid rationale is folded into SPECULATION_UNIFIED_DESIGN.md.
- KERNEL_CONTRACT_CARDS.md (2026-09-28) — a superseded speculation-kernel
  registry; the DSV4 Pro pin it tracked landed.
- HARDWARE_INDEPENDENCE.md (2026-09-28) — a 2026-08-27 snapshot, superseded
  by README and TECHDEBT §Hardware independence and INFERENCE_OS_DESIGN.md.

## Operational history (incidents, runbooks and handoffs superseded by the
 active INCIDENT_RECOVERY_PLAYBOOK.md and FLEET_RELEASE_RUNBOOK.md)
- GLM52_B12X_* (2), GLM52_SM121_*, QWEN38_B16_INCIDENT, TP1_RUNBOOK,
  plus everything already in docs/archive predating this index.
- FLEET_RELEASE.md, FLEET_RUNBOOK.md (2026-09-28) — folded into
  FLEET_RELEASE_RUNBOOK.md; their hub=sparkf, MESH_LEASE and coredev/mgr2
  rules are obsolete.
- DEPLOY_PROTOCOL_MAP.md (2026-09-28) — the 09-14 as-deployed snapshot.
- DEPLOY_ROLLOUT.md (2026-09-28) — the coredev/mgr2 rollout; do not run it.
- HILL_CLIMB_DEPLOY.md (2026-09-28) — the 09-10 narrative; its kill -9
  lesson is superseded by TERM drain.
- WARM_STORAGE_MODEL_POLICY.md (2026-09-28) — the 08-30 inventory; its
  drafter corpus list is still wanted by the speculation work.
- WEIGHTD_EXECUTE_STABILITY.md (2026-09-28) — the 2026-09-13 incident, fixed
  in 4ed8ec3.
- MULTI_DEV_SMOKE.md (2026-09-28) — the 2026-09-15 branch experiment.
- PARALLEL_RESIDENT_QUALIFICATION.md (2026-09-28) — the PR #1082 and
  shared-serving-20260922 campaigns, 8 lanes.
- WEIGHTSD.md (2026-09-28) — the retired weightsd channel; superseded by
  MULTIDEV_QUICKSTART.md.
- BATCH_CACHE_PARALLEL_STATUS.md (2026-09-28) — the 2026-09-22 qualification
  status; fleet-agent is production now. Its still-required items are in
  TECHDEBT.md.
- TP16_TAKEOVER_STATUS.md (2026-09-28) — the 2026-09-15 takeover handoff;
  its ops notes must not be followed; see FLEET_RELEASE_RUNBOOK.md.
- SPARKDEV_HANDOFF_GLM53FLASH.md (2026-09-28) — the PR #1027 handoff; its
  procedures must not be followed.
- HILLCLIMB_20260920.md (2026-09-28) — the allreduce session log of 09-20 to
  09-24; its wedge playbook and deploy facts must not be followed.
- K3_FLEET_LAUNCH_STATE.md (2026-09-28) — the 2026-08-29/30 lane snapshot;
  the bring-up is obsolete, use tools/k3_multidev_run_family.sh.
- QWEN38_DFLASH2_RUNBOOK.md (2026-09-28) — do not run it: spark2 is a GLM
  TP16 rank, and its launch contract SPARK_QWEN38_27B_SPECULATORS=0x4
  (tools/qwen38_27b_dflash2_serve.sh) fails adapter init with SCHEMA_ERROR
  (SPECULATION_UNIFIED_DESIGN.md). It is not a working procedure.
- MODEL_DEV_GUIDE.md (2026-09-28) — the pre-Queue-v2 guide; its rules moved
  to MULTIDEV_QUICKSTART.md "Model work rules".
- PR1077_SERVING_RELIABILITY.md (2026-09-28) — a PR write-up; its runtime
  contract moved to SERVING_FUZZ_COVERAGE.md and its timing fields to
  GLM5_NEXT_ROOFLINE.md.
- LITELLM_INTEGRATION.md (2026-09-28) — merged into LITELLM_FRONTEND.md.

## Deleted, not archived (2026-09-28)
- STATUS.md (root) — README.md's Documentation section now does its job.
- docs/SPARK_QUEUE_RUNBOOK.md — a stub; its content is in
  PARALLEL_DRIVER_DEBUG.md.
