# GLM-5.2 MTP throughput root cause and repair

Status: integrated on spark0, host tests and the SM121 archive and linked-driver
gates passed. GB10 ring performance has not yet been measured. Do not report a
speedup until the matched live experiment below is retained.

## Root cause

Layer-major speculative verification was reaching the six-layer stage as one
`logical_lanes * rows_per_lane` batch. FP8 projections and grouped MoE therefore
had an opportunity to reuse weights across verification rows. The final target
head did not.

`SparkGlm52ResidentDecodeStageFusedFinalTokenCandidateKernel` launched a
separate scalar CUDA-core dot product for every `(execution row, vocabulary
tile)`. The most recent optimization changed only the extra MTP draft head to
cuBLAS tensor cores. Target rows and all verifier rows still used the scalar
kernel. Thus the part that decides whether speculative tokens are accepted did
not use the intended row-batched tensor-core matmul.

The BF16 head contains:

```text
154880 * 6144 * 2 = 1,903,165,440 bytes (1.772 GiB)
```

The old kernel issued head loads independently for every execution row. L2 may
reuse some tiles depending on block scheduling, but the implementation neither
guaranteed one weight traversal nor used tensor-core accumulation. At B64 with
one draft, verification contains 128 rows. That is 128 logical head traversals
and scalar accumulation for 243.6 GB of load requests.

The draft-head patch also capped each cuBLAS call at 32 rows. A B1024 draft
therefore launched 32 GEMMs and traversed the head matrix 32 times per draft
depth.

## Repair

The final rank now allocates one bounded FP32 logits matrix and installs a
fail-closed final-token callback in the exact PP13 plan. Ordinary target rows,
layer-major verifier rows, and MTP draft rows all call the same BF16
tensor-core GEMM followed by device argmax.

The workspace row capacity is the smaller of the execution-row capacity and
`SPARK_STAGE_PLAN_MAX_BATCH_BUCKET`. For B1024 the additional logits
workspace is exactly 634,388,480 bytes (605 MiB). This makes ordinary B1024
decode and each B1024 MTP draft head one GEMM. A seven-row B1024 verification
uses seven 1024-row GEMMs instead of allocating 4.13 GiB for one 7168-row
logits matrix. B1 through B128 verification fits in one GEMM. The former
32-row chunk limit is gone.

Production startup must contain:

```text
pp13_full_vocab_head rank=12 backend=cublas_bf16_tensor_core maximum_rows=... logits_workspace_bytes=... fail_closed=1
```

Failure to allocate the workspace, create the cuBLAS handle, or bind the
callback now fails final-rank construction. It does not select the scalar path.
The scalar implementation remains available only for validators and plans that
do not come from the production builder.

## Required live gate

Build the CUDA targets on a Spark and first retain token-ID parity for plain
decode and MTP against the current accepted reference. Then run the same prompt
set, output length, release flags, and concurrency for plain and MTP at B1,
B16, B64, B128, and the largest resident bucket. Use at least 256 generated
tokens per point after warm-up.

Retain all of the following for each point:

- total token events/s and request latency;
- draft tokens, accepted drafts, rejected drafts, committed tokens, and verify
  dispatches;
- logical lanes, rows per lane, execution rows, and verify chunk count;
- final-rank CUDA kernel timings for the head, MTP layer, and complete stage;
- output token IDs, not only decoded text.

The trace must show a cuBLAS tensor-core GEMM for the target/verifier head and no
production launch of `SparkGlm52ResidentDecodeStageFusedFinalTokenCandidateKernel`.
If throughput remains flat after that is proven, the next measurement is the
final-rank critical path: repeated autoregressive MTP layers and heads are
serialized there and may require stage rebalancing or a fourteenth Spark. That
is a separate bottleneck; it should not be guessed before the head repair is
measured.

## Local validation completed

```text
make -j test tools glm52_pp13_service_backend hidden_transport_tcp_cuda
    glm52_pp13_node_context_builder: PASS
python3 tests/test_glm52_exact_pp13_prefill_hidden.py: PASS
git diff --check: PASS
SM121 archive exact-stage validation: PASS, 14.423410 ms
SM121 linked-driver exact-stage validation: PASS, 14.345318 ms
archive and linked output checksum64: 12536524931322045649
GB10 ring performance: NOT MEASURED
```
