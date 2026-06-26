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
    average_us=5910.752
    maximum_us=6601.600
    limit_us=10000.000

generated-driver/orchestrator validator:
    elapsed_us=6392.128
    limit_us=10000.000
```

## What this proves

The generated GLM 5.2 decode-stage `model_driver.so` can be loaded by the
SparkPipe runtime, attached to an orchestrator route, admitted through the
driver scheduler, and submitted into the real CUDA backend on SM121 hardware.

It also proves the current resident decode-stage control path does not use
host-staged frame buffers or serving-path device copies for the submitted
frame. The counters are checked by the validator, not just printed.

## What this does not prove yet

This is not a full GLM 5.2 inference pass. The decode-stage validator still
uses deterministic smoke tensors and mostly zero weights. It proves the
hardware path and driver ownership model before it proves model logits.

The next correctness gate must replace smoke tensors with deterministic
nonzero GLM fixtures and check:

```text
KV write layout per position
cached attention reads over context length greater than one
block-table remapping
restricted-vocabulary logits
MTP draft and verify/commit behavior
runtime snapshot counters after real tensor work
```

## Next experiment hypothesis

Hypothesis:

```text
If deterministic nonzero GLM-shaped tensors are used, the current resident
decode-stage CUDA chain will preserve the same driver/orchestrator invariants
but expose numerical gaps in KV layout, sparse-index semantics, restricted
logits, or MTP verification before it exposes transport-level failures.
```

Experiment:

```text
Add a nonzero-fixture mode to the decode-stage validator.
Run B1 with context length 4, one remapped KV block, and fixed restricted vocab.
Compare device outputs against a small host reference for KV write, attention
output checksum, selected token, MTP accept mask, and committed token ids.
```

If it fails:

```text
Do not tune constants.
Classify the failure as a layout, arithmetic, scheduling, or ownership bug.
Update this log with the failed assumption, then fix the model before rerunning.
```
