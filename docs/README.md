# Maintained Technical Documentation

The live documentation set is intentionally small. These files describe the
selected system, not the sequence of experiments that produced it.

## Architecture contracts

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md): complete system architecture.
- [`../SPEC.md`](../SPEC.md): firmware, package, and runtime contract.
- [`HARDWARE_TOPOLOGY.md`](HARDWARE_TOPOLOGY.md): sixteen-Spark combined fabric.
- [`PAIRED_DUAL_LINK_ALLREDUCE.md`](PAIRED_DUAL_LINK_ALLREDUCE.md): adaptive
  two-rail collective.
- [`DSV4_FLASH_TP4_PP4.md`](DSV4_FLASH_TP4_PP4.md): canonical large-model
  resident layout.
- [`MODEL_SUPPORT.md`](MODEL_SUPPORT.md): product model set and qualification
  contract.
- [`MODULE_MAP.md`](MODULE_MAP.md): source ownership boundaries.

## Implementation contracts

- [`CODEX_RUNBOOK.md`](CODEX_RUNBOOK.md): clean-main release and qualification
  workflow.
- [`DATAFILE_NAMING.md`](DATAFILE_NAMING.md): immutable model artifact naming.
- [`GLM52_B12X_PACK_WORKER_PROTOCOL.md`](GLM52_B12X_PACK_WORKER_PROTOCOL.md):
  GLM pack worker protocol.
- [`GLM52_B12X_RESIDENT_MOE_PACK.md`](GLM52_B12X_RESIDENT_MOE_PACK.md): GLM
  resident MoE pack format.
- [`GLM52_SM121_REQUIRED_CUDA_MODULE.md`](GLM52_SM121_REQUIRED_CUDA_MODULE.md):
  required GLM CUDA boundary.
- [`K3_PACK_FORMAT_V2.md`](K3_PACK_FORMAT_V2.md): K3 resident pack format.
- [`K3_WEIGHT_ONLY_MXFP4.md`](K3_WEIGHT_ONLY_MXFP4.md): K3 precision contract.
- [`SPARK_HOST_RDMA_DOORBELL.md`](SPARK_HOST_RDMA_DOORBELL.md): host-RDMA data
  path.

## Changing status

- [`../TECHDEBT.md`](../TECHDEBT.md): unfinished implementation work only.
- [`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md): measurements,
  projections, and target gates only.

Superseded phase reports, handoffs, audits, experiment diaries, recovered
estimate documents, and validation snapshots are preserved under
[`archive/`](archive/). They are historical evidence, not current authority.
