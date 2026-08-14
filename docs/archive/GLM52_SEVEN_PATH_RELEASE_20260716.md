# GLM-5.2 Seven-Path Release Gate

## Scope

This release candidate integrates bulk prefill, NVMe JIT KV, resident DSA block
prefetch, compressed FP8 KV, Q/KV overlap, persistent-doorbell verbs transport,
and memlink lane partitioning. Host validation is necessary but not sufficient.

## Build Gates

```sh
make -j test
make -j all tools
make cuda_glm52_resident_decode_stage
make glm52_pp13_node_context_builder
make hidden_transport_spark_host_rdma_verbs
```

The last three commands must build on an SM121 Spark host. A skip due to missing
`nvcc`, CUDA headers, or verbs headers is not a pass.

## Zero-Drift Release

1. Merge the PR to `main`.
2. Pull that exact commit into the live Spark checkout.
3. Build and publish from the pulled checkout.
4. Confirm every rank reports the same release and artifact hashes.
5. Restart the resident and agent roles in dependency order.
6. Run ring checks before inference.

## Ring Receipts

### Transport

Run 100 hidden-only laps and 100 laps with an 8 KiB sideband. Record average,
worst lap, each rank's hop time, and lane byte counts. Any payload mismatch or
timeout fails the release.

### Accuracy

Use the matched real prompt and greedy decoding with detailed hidden and cache
dumps enabled. Compare every stage against the serialized FP8 reference. A
streamed token without numerical parity is not an accuracy pass.

### Bulk Prefill

Run 64-token and 256-token prompts. Confirm the packet row counts, one embedding
gather per chunk, no resident IPC disconnect, and identical output to the
serialized reference.

### KV And DSA

Force GPU KV eviction to the `.jit` store, reload it, and verify byte parity.
Then run a context above 2,048 DSA candidates and record selected-token count,
selected-block count, prefetch duration, attention duration, and JIT counters.

### Performance

Record B1, B4, B16, B64, B256, and B1024 where memory permits. Report prompt
length, generated tokens, active experts, MTP mode, context length, cold and warm
latency, tokens per second, and per-stage device clocks. Compare against the last
measured release; do not infer speedup from capability activation.

## Fail Conditions

- Any fallback backend is selected.
- Any required capability is absent.
- SM121 or verbs compilation is skipped.
- Ring payload integrity fails.
- Serialized and ring accuracy diverge.
- NVMe records are recreated or truncated on every startup.
- A feature increases latency without a measured capacity or accuracy benefit.
