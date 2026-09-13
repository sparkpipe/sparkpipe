# GLM-5.2 FP8 Scaled-GEMM Activation Receipt

This receipt records one deployment measurement. It does not certify model
accuracy, multi-lane serving, long-context behavior, or the theoretical
filled-pipeline ceiling. Current status terminology and accepted claims are in
`docs/GLM52_MEASURED_STATUS.md`.

## Provenance

```text
merged main: 9cc386a4ad1fa6827e7e36fba8fb1b4a7e16f00c
PR:          #286
release:     glm52-fp8-main-9cc386a-b16-scaled-gemm
generation:  20260712013500
bucket:      B16
```

The release was built from clean worktree `/tmp/sparkpipe-release-9cc386a` on
spark0. The live ring was updated only after the exact archive and linked
driver package both passed SM121 validation.

Installed hashes were identical on ranks 0 through 12:

```text
libglm52_pp13_node_context_builder.so  3e09169c90ffe89c...
model_driver.so                        c5aac8c4ce267419...
```

## CUDA Validation

The exact stage-0 six-layer FP8 gate used one whole-stage graph and real stage
packs. The archive and packaged driver produced the same nonzero checksum.

```text
archive timed_us:          15075.623
packaged driver timed_us:  14626.469
layer bodies:              6
graph captures:            1
graph replays:             2
nonzero bf16 outputs:      6144 / 6144
checksum64:                14772085175729278197
```

The previous live stage-0 receipt was approximately 64.8 ms. The scaled-GEMM
activation reduces that isolated stage by about 4.3x. The reciprocal, about
68.4 stages/s, is a theoretical filled-pipeline ceiling and is not measured
end-to-end throughput.

The focused FP8 GEMM gate also passed eager and graph modes with maximum
absolute error `0.017578`:

```text
B1  576->640 padded-tail eager   0.053235 ms
B1  576->640 padded-tail graph   0.051189 ms
B64 576->640 padded-tail eager   0.065526 ms
B64 576->640 padded-tail graph   0.071251 ms
B1  2048->16384 aligned graph    0.142880 ms
```

There is no FP8 WMMA fallback. An FP8 plan without the scaled-GEMM backend now
fails validation.

## End-To-End Smoke Observation

The deployed 13-rank ring returned the known greedy token:

```text
prompt:    Say OK. OK.
token id:  10397
text:      " OK"
elapsed:   0.409 s for one-token request
```

A distinct factual prompt produced:

```text
prompt:  Q: What is the capital of France?\nA:
output:  " Paris\nQ:"
```

Post-run health remained clean:

```text
backend_ready=1
pp13_ready=1
live_requests=0
queued_requests=0
event_backlog=0
dropped_events=0
first_blocker=""
```

## API Performance

Raw data is committed beside this receipt under:

```text
diagnostics/glm52_fp8_scaled_gemm_20260712/
```

B1, one request, eight generated tokens:

```text
elapsed:             2.571 s
generated tokens:    8
token events/s:      3.112
HTTP failures:       0
```

The 16-token check reached `3.272` token events/s. This is roughly 3.4x the
previous `0.93 tok/s` end-to-end receipt, but remains far below the isolated
stage ceiling.

Four simultaneous requests, four generated tokens each:

```text
request completion 1:  1.383 s
request completion 2:  2.760 s
request completion 3:  4.136 s
request completion 4:  5.545 s
aggregate tokens/s:     2.884
HTTP failures:          0
```

The completion staircase proves the gateway serializes clients. It enters
`SparkGlm52GatewayServeOne`, blocks in
`SparkGlm52GatewayDrainStreamResponse`, buffers the full SSE body, sends one
`Content-Length` response, closes that client, and only then accepts the next
client. The production scheduler cannot form a batch from requests it never
receives concurrently.

## Implemented But Not Active In The Live Path

The following are code-complete or substantially implemented but do not
contribute to this release's measured throughput:

1. Decode and prefill batch construction exists in the scheduler/request API,
   but the PP13 service backend and builder reject request, active-sequence,
   or lane counts other than one.
2. B16/B32/B64 and extended B128/B256/B512/B1024 bucket selection exists, but
   the live gateway presents one request to a one-lane backend dispatch.
3. Bulk/paged prefill plans exist, but the builder sets
   `maximum_prompt_token_count = 1` and serializes prompt tokens.
4. Q/KV branch streams and events are allocated, but the builder removes the
   `QKV_BRANCH_OVERLAP` capability.
5. JIT and async JIT KV-prefetch implementations exist, but the service
   backend omits both request-API configuration flags. Live counters remain
   zero.
6. MTP request/verify code exists, but the service backend omits the
   `MTP_COMMIT` configuration flag in this release.
7. DSpark code exists, but the release manifest does not pass `--dspark` and
   no draft weights are resident.
8. FP8 KV-cache support exists in lower CUDA contracts, but this builder
   allocates `mla_cache`, `key_nope_cache`, and `value_cache` as `uint16_t` and
   the service metadata declares two bytes per scalar.
9. The Spark-host RDMA transport exists, but the manifest selects
   `libhidden_transport_tcp_cuda.so`.
10. Real HTTP streaming is not active: the gateway buffers all SSE events and
    reports TTFT equal to total request latency.

The next highest-impact activation is the event-driven multi-client gateway
and multi-lane PP13 backend contract. Kernel tuning is no longer the first
bottleneck at B1.
