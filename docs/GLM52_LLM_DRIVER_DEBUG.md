# GLM 5.2 LLM driver debug log

This log records live Spark/CUDA driver gates that should not be lost in chat
history. A passing gate here is not a full GLM inference claim unless it says
so explicitly.

## Current verified gate

Hardware:

```text
GPU: NVIDIA GB10
CUDA: 13.0
Target: sm_121
```

Commands:

```sh
PATH=/usr/local/cuda-13.0/bin:$PATH make -j1 test
PATH=/usr/local/cuda-13.0/bin:$PATH make glm52_resident_sparse_mla_firmware_package MAX_STAGE_MICROSECONDS=1500
PATH=/usr/local/cuda-13.0/bin:$PATH make glm52_resident_decode_stage_firmware_package MAX_STAGE_MICROSECONDS=10000
```

The sparse-MLA package gate verifies:

```text
nvcc archive build
hardware sparse-MLA validator execution
module-library publication
model package compilation
generated driver loadability
```

Latest observed sparse-MLA timing:

```text
average_us=396.328
maximum_us=414.014
limit_us=1500.000
```

The decode-stage package gate verifies:

```text
nvcc archive build
hardware backend-submit validator execution
deterministic nonzero GLM-shaped smoke tensors
KV current-token write into the expected cache slot
cached attention over context length 4
restricted-vocabulary argmax
MTP draft accept/reject/commit counters
module-library publication
model package compilation
generated driver loadability
generated driver route resolution
orchestrator admission
orchestrator submit into CUDA backend
stream-ordered completion
runtime snapshot counters
zero host-staging bytes
zero device-memcpy bytes
```

Latest observed decode-stage timings:

```text
backend validator:
    fixture=nonzero_context4
    average_us=6343.243
    maximum_us=6652.352
    limit_us=10000.000

generated-driver/orchestrator validator:
    fixture=nonzero_context4
    elapsed_us=6656.384
    limit_us=10000.000
```

## What this proves

The generated GLM 5.2 decode-stage `model_driver.so` can be loaded by the
SparkPipe runtime, attached to an orchestrator route, admitted through the
driver scheduler, and submitted into the real CUDA backend on SM121 hardware.

It also proves the current resident decode-stage control path does not use
host-staged frame buffers or serving-path device copies for the submitted
frame. The counters are checked by the validator, not just printed.

## New nonzero fixture

The decode-stage validator now seeds deterministic nonzero tensors directly
into the resident device buffers. The pattern is intentionally small enough for
a host reference but still uses full-shape resident allocations and the real
CUDA chain.

The checked invariants are:

```text
cache slot 3 receives the current KV latent value
cache slot 3 receives the current key RoPE pair
query latent is nonzero
attention output[0] matches a host softmax reference over 4 cached tokens
restricted argmax selects token 1009
MTP drafts token 1011 twice
MTP accepts the first draft and rejects the second against token 1003
MTP event counters are accepted=1 rejected=1 committed=2 rollback=1 cancelled=0
phase completion marker is written
driver/orchestrator counters remain clean
```

## What this does not prove yet

This is not a full GLM 5.2 inference pass. The decode-stage validator now uses
deterministic nonzero smoke tensors, but it still does not load real checkpoint
weights or compare final logits against a known GLM artifact.

The next correctness gate must replace smoke tensors with deterministic
nonzero GLM fixtures and check:

```text
multiple positions
nonzero block-table remapping
more than one nonzero attention dimension/head
restricted-vocabulary logits against a richer host reference
MTP draft and verify/commit behavior with varied target patterns
runtime snapshot counters after real tensor work
```

## Next experiment hypothesis

Previous hypothesis:

```text
If deterministic nonzero GLM-shaped tensors are used, the current resident
decode-stage CUDA chain will preserve the same driver/orchestrator invariants
but expose numerical gaps in KV layout, sparse-index semantics, restricted
logits, or MTP verification before it exposes transport-level failures.
```

Result:

```text
The B1 context-length-4 nonzero fixture passed on GB10. KV write layout,
single-dimension cached attention, restricted argmax, and MTP accept/reject
matched the host reference.
```

Next hypothesis:

```text
The next likely gap is not the driver boundary. It is richer GLM tensor
semantics: block-table remapping beyond one physical block, multiple nonzero
attention dimensions/heads, and real checkpoint quantization layout.
```

If it fails:

```text
Do not tune constants.
Classify the failure as a layout, arithmetic, scheduling, or ownership bug.
Update this log with the failed assumption, then fix the model before rerunning.
```
