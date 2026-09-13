# Phase 3 Validation Status

## Scope

This receipt covers the DSV4 exact cache allocator, grouped-MoE route construction, GLM sealed-batch expert queue, corrected GLM batch-plane estimator, restored model contracts, and the host-side compatibility repairs required to build their tests.

## Passed host checks

The retained targeted validation executes from an empty `build/` directory and passes:

- generated K3 contract check;
- generated DSV4 Flash and Pro contract check;
- mandatory deployment-target contract;
- grouped-MoE source contracts;
- DSV4 cache-plan and exact-arena allocation test;
- DSV4 cache reservation report tool;
- GLM low-latency and sealed-batch expert-queue test;
- corrected GLM batch-plane estimator;
- generic GLM stage-plan test;
- generic GLM TP-shard test;
- hidden-transport host contract test;
- GLM stagepack host test;
- single-ring and one-switch fabric-topology test;
- Qwen 3.6 work-control test;
- K3 engine, KV geometry, layer, pack, quantization, shard, shard-table and slice tests;
- router, K3 layer, K3 slice and shared layer host-CUDA harnesses;
- Python syntax compilation for all changed Python tools and tests.

The complete retained command output is in:

```text
qualification/phase3_targeted_validation.log
```

## Complete host-suite status

The repository-wide `make test` target is not green. It now progresses beyond four stale tests repaired in this phase and stops while compiling:

```text
tests/test_glm52_ring_runtime.c
```

That file contains malformed source and calls an obsolete ring-runtime rank-plan signature. The exact failure is retained in:

```text
qualification/phase3_full_make_test_failure.log
```

## CUDA and hardware boundary

The following have not been established:

- CUDA 13 compilation;
- `compute_121a` PTX generation;
- `sm_121a` `ptxas` acceptance;
- device linking;
- CUDA numerical correctness;
- memory and race safety;
- register, spill, shared-memory or occupancy results;
- ring or one-switch network performance;
- production readiness.

Status:

```text
HOST_TARGETED_VALIDATED
HOST_COMPLETE_SUITE_BLOCKED_BY_STALE_TEST
CUDA13_SM121A_COMPILE_NOT_RUN
BLACKWELL_EXECUTION_NOT_MEASURED
PRODUCTION_READY=false
```
