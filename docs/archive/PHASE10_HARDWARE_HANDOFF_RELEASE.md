# Phase 10 — Hardware Handoff Release

Phase 10 turns Spark hardware assumptions into an executable, fail-closed qualification system.

## Scope completed without hardware

- Thirty-three explicit hardware questions cover GB10 memory, launch behavior, resources, model kernels, NVMe, TCP/RDMA, ring/switch topology, PP degree, transport window, and stage placement.
- Exact run plans cover the five mandatory model packages, B1–B1024, retained context buckets, PP1–PP16 where topology permits, RDMA lane/window/CQ choices, and NVMe queue choices.
- Ring-debug and one-switch single-rail topologies have deterministic runner-configuration generation.
- Generic CUDA, NVMe, PMTU, model-kernel, transport, and topology probes are present.
- Model, transport, and topology probes require exact production-provider artifacts; synthetic providers cannot qualify production cells.
- Every plan cell has deterministic identity, bounded ownership, timeout handling, per-cell locking, and resume behavior.
- Runner configurations carry a content SHA-256 and are rejected after modification.
- A node preflight validates every command mapping and inventories exact executable/provider hashes before measurements begin.
- Policy compilation retains node and peer scope instead of selecting the best result across different machines.
- Policy compilation derives mapped-host versus explicit-copy decisions and CPU/GPU contention effects.
- Question closure proves the chain from question to exact cells, receipts, policy path, and source consumers.
- CUDA 13 compile-only qualification targets exact `compute_121a` and `sm_121a` and retains PTX, objects, `ptxas`, `cuobjdump`, and hashes.

## Deliberate qualification boundary

The release does not claim:

- CUDA 13 compilation was run in this environment;
- any production provider exists for an unfinished model executor;
- numerical correctness on GB10;
- race freedom;
- CUDA Graph correctness;
- physical ring or switch correctness;
- measured latency or throughput;
- production readiness.

Those claims can only be advanced by receipts produced from the exact archive on the Sparks.

## Mandatory targets

```text
Kimi K3
    MXFP4 routed-expert weights
    BF16 expert activations
    BF16 non-expert tensors
    FP32 accumulation

GLM 5.2
    FP8 E4M3 routed-expert weights
    BF16 expert activations
    BF16 non-expert tensors
    FP32 accumulation

Qwen 3.6 27B
    BF16 weights and activations

DeepSeek V4 Flash
    checkpoint-native mixed precision
    independent generated geometry and package

DeepSeek V4 Pro
    checkpoint-native mixed precision
    independent generated geometry and package
```

K3's native post-training recipe uses MXFP4 expert weights and MXFP8 expert activations. The BF16 expert-activation package is an intentional SparkPipe deployment variant and requires its own numerical receipts.

## Primary files

```text
model_contracts/spark_hardware_questions.json
model_contracts/spark_hardware_assumption_bindings.json
qualification/spark/probe_plan.json
qualification/spark/workload_profiles.json
qualification/spark/topologies/
qualification/spark/configs/
tools/hardware/spark_qualification_plan.py
tools/hardware/generate_runner_configs.py
tools/hardware/spark_handoff_preflight.py
tools/hardware/run_probe_job.py
tools/spark_hardware_qualify.py
tools/hardware/spark_policy.py
tools/hardware/spark_question_closure.py
tools/cuda13_sm121a_compile_gate.sh
```

See `docs/SPARK_HARDWARE_HANDOFF.md` for the exact operational sequence.
