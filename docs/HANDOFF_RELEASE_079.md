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

## What is now hardware-validated

After import into the live Spark repo, both active CUDA firmware packages have
been built and validated on a GB10 / CUDA 13.0 / `sm_121` Spark node.

The decode-stage package target now goes past raw backend validation. After
`sparkpipe_model_compile` emits the generated `model_driver.so`, it runs an
orchestrator smoke test that attaches the generated driver, resolves the
resident decode route, admits one frame, submits through the driver into CUDA,
waits for stream-ordered completion, and checks runtime counters.

Latest live timings are recorded in:

```text
docs/GLM52_LLM_DRIVER_DEBUG.md
```

## What is still not claimed

The CUDA path is hardware-executed, but this is not yet a full GLM 5.2
inference pass. The decode-stage validator still uses deterministic smoke
tensors and mostly zero weights. Full inference requires deterministic nonzero
GLM tensor fixtures, reference comparisons for KV writes/cached attention,
restricted logits, MTP verify/commit behavior, and then real checkpoint tensor
loading.

The next developer pass should add that nonzero-fixture correctness mode before
performance tuning. The architecture now localizes that work inside
model-specific firmware archives instead of spreading it through SparkPipe.
