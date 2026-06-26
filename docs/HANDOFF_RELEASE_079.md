# Iteration 079 handoff release

Iteration 079 finishes the SparkPipe-level firmware boundary and reshapes the CUDA modules for a target-hardware/Codex debug pass.

## SparkPipe level

The serving runtime remains a small OS-like layer:

```text
spark_driver_loader.o
spark_orchestrator.o
```

It does not contain JSON parsing, module publication, validation, hashing, code generation, CUDA planning, model graph execution, fallback routing, or LLM-specific structures.

The model driver ABI is now version 3. It supports:

```text
neutral admission
opaque dispatch slot
dispatch generation and cookies
runtime snapshot counters
zero host-staging/device-memcpy accounting
private pressure reporting
pre-resolved route submission
completion retirement
```

SparkPipe can schedule across heterogeneous replicas without knowing KV layout, MoE queues, CUDA graph topology, sparse-token policy, or MTP internals.

## CUDA module shape

Both active CUDA firmware modules now expose the same SOTA-oriented control shape:

```text
pre-bound resident node context
per-slot state machine
optional per-slot CUDA graph exec
per-slot graph capture/replay counters
configurable launch checking
no host-staged frame buffers
zero expected device memcpy
stream-ordered external completion
admission/snapshot symbols where scheduling pressure matters
```

The GLM resident decode-stage module additionally has:

```text
opaque dispatch-ticket validation
stale admission rejection
preselected sparse-index production path
debug-only serial DSA mode
parallel prefix sparse-index bring-up mode
phase-clock mode disabled by default
MXFP4-only MTP draft path
```

## What is still not claimed

No CUDA archive is hardware-validated in this environment. There is no `nvcc` and no SM121 Spark target here. The arithmetic kernels are shaped for handoff but several remain correctness-first.

The next developer pass should compile on target hardware, run the full-stage validator, inspect CUDA traces, and replace the projection/attention/MTP kernels that miss the latency ceiling. The architecture now localizes that work inside model-specific firmware archives instead of spreading it through SparkPipe.
