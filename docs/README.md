# Maintained Technical Documentation

This is the index of the live documentation. Each file is the authority for
one thing. Superseded designs, handoffs and experiment logs are under
[`archive/`](archive/); [`ARCHIVE_INDEX.md`](ARCHIVE_INDEX.md) lists every
archived file with the reason it was archived. Nothing in the archive is
current authority.

## System authorities

- [`../README.md`](../README.md): the system authority, describing the
  system SparkPipe is built to be.
- [`../SPEC.md`](../SPEC.md): the firmware, module library and compiler
  contract.
- [`../sparkpipe_invariants.md`](../sparkpipe_invariants.md): the rules every
  driver must satisfy.
- [`../AGENTS.md`](../AGENTS.md): instructions for agents working in this
  repository.

## Architecture contracts

- [`HARDWARE_TOPOLOGY.md`](HARDWARE_TOPOLOGY.md): sixteen-Spark combined
  fabric and the Mac Studio pool design.
- [`TOPOLOGY_GUIDE.md`](TOPOLOGY_GUIDE.md): placement candidates (TP16,
  PP16, TP4 x PP4 and hybrids) and how to hill-climb them on measured numbers.
- [`MODEL_SUPPORT.md`](MODEL_SUPPORT.md): product model set: checkpoint
  contracts, owner direction, qualification contract.
- [`MODULE_MAP.md`](MODULE_MAP.md): source ownership boundaries.
- [`INFERENCE_OS_DESIGN.md`](INFERENCE_OS_DESIGN.md): device-layer memory
  model for hardware independence.
- [`EXPERT_GROUPED_SCHEDULING.md`](EXPERT_GROUPED_SCHEDULING.md):
  expert-grouped continuous batching.

## Implementation contracts

- [`WEIGHTD_DESIGN.md`](WEIGHTD_DESIGN.md): weightd daemon, production
  ownership, lanes, expert residency. Its
  [Mesh substrate and rendezvous](WEIGHTD_DESIGN.md#mesh-substrate-and-rendezvous)
  section is the TP transport contract.
- [`WEIGHTD_SUPERVISED_STARTUP.md`](WEIGHTD_SUPERVISED_STARTUP.md):
  supervised weightd startup contract.
- [`TP_STREAM_MEMOP_QUALIFICATION.md`](TP_STREAM_MEMOP_QUALIFICATION.md):
  hardware waits, their production status and qualification gaps
  (authoritative for device waits).
- [`EVENT_DRIVEN_PROGRESS.md`](EVENT_DRIVEN_PROGRESS.md): completion events
  and polling in the collective and the API worker.
- [`DRIVER_ACCEPTANCE.md`](DRIVER_ACCEPTANCE.md): driver contract, including
  page and position budgets, the known exception and per-driver gaps.
- [`COMMON_MODULE_ARCHITECTURE.md`](COMMON_MODULE_ARCHITECTURE.md): common
  modules and `llm_defines.h`.
- [`FAMILY_TEMPLATES.md`](FAMILY_TEMPLATES.md): code shared across forked
  model families, and the rule for sharing it.
- [`KERNEL_PLAYBOOK.md`](KERNEL_PLAYBOOK.md): shared-kernel techniques and
  contract template.
- [`STAGEPACK_NAMING.md`](STAGEPACK_NAMING.md): pack directory and
  runtime-root naming; `tools/stagepack_naming.json` is the machine source.
- [`DATAFILE_NAMING.md`](DATAFILE_NAMING.md): serving recipe datafile and KV
  geometry-hash naming.
- [`DRY_PACKBUILDER_PROPOSAL.md`](DRY_PACKBUILDER_PROPOSAL.md): packer
  consolidation plan.
- [`T1_REFERENCE_COMPARE.md`](T1_REFERENCE_COMPARE.md): T1 reference-decoder
  comparison contract.
- [`SERVING_FUZZ_COVERAGE.md`](SERVING_FUZZ_COVERAGE.md): serving coverage
  map and runtime contract.
- [`LITELLM_FRONTEND.md`](LITELLM_FRONTEND.md): the LiteLLM door.

## GLM 5.3 Flash

- [`GLM5_NEXT_ROOFLINE.md`](GLM5_NEXT_ROOFLINE.md): GLM 5.3 Flash
  performance, roofline and current measurements.
- [`GLM_PERFORMANCE_GATES.md`](GLM_PERFORMANCE_GATES.md): GLM acceptance
  gates (quality, parity, prefix, throughput receipts).
- [`GLM_LAZY_DRIVER_INTEGRATION.md`](GLM_LAZY_DRIVER_INTEGRATION.md): GLM
  expert residency and lazy loading, including the production pin-all mode.
- [`GLM_REAL_DRIVER_PROBE.md`](GLM_REAL_DRIVER_PROBE.md): single-rank driver
  probe.
- [`MIGRATION_GLM5_NEXT_COMMON.md`](MIGRATION_GLM5_NEXT_COMMON.md):
  glm5_next adoption of the common GLM modules.

## Other models

- [`K3_PERF.md`](K3_PERF.md): SparkPipe's only K3 numbers.
- [`K3_VS_GB10_VLLM.md`](K3_VS_GB10_VLLM.md): K3 against the gb10-vllm
  stack, lever by lever.

## Speculation

- [`SPECULATION_UNIFIED_DESIGN.md`](SPECULATION_UNIFIED_DESIGN.md):
  speculation seam, source mask, draft bridge, per-family state and
  direction.
- [`RTX5090_SPECULATION_NODE.md`](RTX5090_SPECULATION_NODE.md): rtx5090 host
  (fleet hub, GLM API, draft farm, drafter store).
- [`SPECULATION_PLAN.md`](SPECULATION_PLAN.md): speculation architecture,
  drafters per model, and the PR sequence.

## Operations and development

- [`FLEET_RELEASE_RUNBOOK.md`](FLEET_RELEASE_RUNBOOK.md): the single
  production fleet guide (hub, fleet-agent, releases, g53-api, bootstrap,
  triage). This is the release and operations procedure.
- [`MULTIDEV_QUICKSTART.md`](MULTIDEV_QUICKSTART.md): parallel development
  guide: lanes, tiers, queue, performance-evidence rule, model work rules.
- [`PARALLEL_DRIVER_DEBUG.md`](PARALLEL_DRIVER_DEBUG.md): queue v2 contract
  and PR evidence.
- [`DEVCYCLE.md`](DEVCYCLE.md): queue builds and the shared development
  cycle.
- [`INCIDENT_RECOVERY_PLAYBOOK.md`](INCIDENT_RECOVERY_PLAYBOOK.md): Spark boot
  and OOM recovery (PXE, then the brickproof controller).
- [`CX7_DMA_INVESTIGATION_2026-10.md`](CX7_DMA_INVESTIGATION_2026-10.md):
  the Lenovo ConnectX-7 completion-timeout and SMMU0 stall investigation
  (dated record: incidents, every crash run, mitigations, implications, open
  experiments).
- [`SPARK_MANAGEMENT_FAILOVER.md`](SPARK_MANAGEMENT_FAILOVER.md): management
  SSH route selection.
- [`TP_CUPTI_TRACE.md`](TP_CUPTI_TRACE.md): isolated per-rank CUPTI timing
  trace.
- [`SERVING_RELEASE_RECOVERY_20260924.md`](SERVING_RELEASE_RECOVERY_20260924.md):
  reproducing the `shared-serving-20260922` release (dated record; its
  replay needs an operator maintenance window).
- [`operations/managed-driver-station.md`](operations/managed-driver-station.md):
  the 2026-09-24 managed driver station layout (dated record; the fleet now
  serves GLM 5.3 Flash under fleet-agent, see the runbook).

## Changing status

- [`../TECHDEBT.md`](../TECHDEBT.md): unfinished implementation work only.
- [`ROADMAP.md`](ROADMAP.md): the order in which that work closes, with each
  milestone's exit criterion. It now includes M10 (speculation at B1-B64)
  and M11 (drivers in parallel).
- [`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md): measurements,
  projections, and target gates only. History:
  [`archive/PERFORMANCE_HISTORY_2026-08.md`](archive/PERFORMANCE_HISTORY_2026-08.md).
- [`GOALS.md`](GOALS.md): operator directives (topology by measurement, B1
  target, model direction, COMPSEC-17 gate).
- [`HANDOFF_DRIVERS_2026-09-28.md`](HANDOFF_DRIVERS_2026-09-28.md): milestone
  and driver status and open operator questions as of 2026-09-28.
- [`MARKETPLACE_PLAN.md`](MARKETPLACE_PLAN.md): business plan draft for the
  provider network.

## Evidence and research material

- [`receipts/`](receipts/): JSON receipts cited by the documents above.
- [`research-dflash2/`](research-dflash2/) and
  [`research-dspark/`](research-dspark/): copies of external papers, posts
  and code used by the speculation work; reference material, not SparkPipe
  authority.

## History

Earlier versions of this index presented eight archived documents as current
contracts: PAIRED_DUAL_LINK_ALLREDUCE, DSV4_FLASH_TP4_PP4, CODEX_RUNBOOK,
GLM52_B12X_PACK_WORKER_PROTOCOL, GLM52_B12X_RESIDENT_MOE_PACK,
GLM52_SM121_REQUIRED_CUDA_MODULE, K3_PACK_FORMAT_V2 and
K3_WEIGHT_ONLY_MXFP4. They are history; their entries are in
[`ARCHIVE_INDEX.md`](ARCHIVE_INDEX.md).
