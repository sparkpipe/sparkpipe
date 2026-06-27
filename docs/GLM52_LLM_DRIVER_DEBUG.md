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

Live GLM artifact geometry checked on Spark1:

```text
model_dirs=/home/spark1/models/hf/nvidia/GLM-5.2-NVFP4,/home/spark1/models/hf/lukealonso/GLM-5.2-NVFP4,/home/spark1/models/hf/zai-org/GLM-5.2-FP8
hidden_size=6144
num_attention_heads=64
kv_lora_rank=512
qk_rope_head_dim=64
index_topk=2048
num_hidden_layers=78
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
KV current-token write into a block-table-remapped cache slot
cached attention over context length 4 crossing two physical KV blocks
eight checked latent dimensions across four checked heads
four checked RoPE dimensions with non-identity rotation
attention output projection feeding restricted logits
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
    fixture=remapped_nonzero_context4_h4_d8_r4
    average_us=4570.848
    maximum_us=4821.600
    limit_us=10000.000

generated-driver/orchestrator validator:
    fixture=remapped_nonzero_context4_h4_d8_r4
    elapsed_us=5121.408
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
logical context tokens 61..64 resolve through physical block table [1,0]
cache slot 0 receives the current KV latent value
cache slot 0 receives non-identity-rotated current key RoPE values
query latent is nonzero for four checked heads and eight checked dimensions
rotated query RoPE is nonzero for four checked heads and four checked dimensions
attention outputs match a host softmax reference over remapped slots 125,126,127,0
restricted logits depend on attention output projection, not only input hidden
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
larger nonzero attention dimension/head coverage
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
The B1 context-length-4 remapped fixture passed on GB10. KV write layout,
two-block cached attention over slots 125,126,127,0, restricted argmax, and MTP
accept/reject matched the host reference through both direct backend submit and
generated-driver/orchestrator submit. The fixture was then broadened to four
heads, eight latent dimensions, four RoPE dimensions, non-identity key/query
RoPE, and attention-output-projection-fed restricted logits; that also passed
on Spark1 / GB10 under the 10000 us package ceiling.
```

Next hypothesis:

```text
The next likely gap is not the driver boundary or the small remapped-cache
layout. It is real GLM tensor semantics: varied positions, varied sparse-token
selection, multi-block KV checksums, and checkpoint quantization layout.
```

If it fails:

```text
Do not tune constants.
Classify the failure as a layout, arithmetic, scheduling, or ownership bug.
Update this log with the failed assumption, then fix the model before rerunning.
```
