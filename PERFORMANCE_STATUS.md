# SparkPipe Performance Status

This file records measured end-to-end performance.  Analytical projections
remain in model-specific design documents and do not count as achieved
milestones here.

## Latest accepted milestone: DSV4 Flash TP4 B1

The latest retained measurement is **32.57 decode tokens/s mean** over four
runs, with a **32.81 tokens/s best run**.  This is one request, not eight or
sixteen concurrent requests and not aggregate batch throughput.

| Field | Measured configuration |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Checkpoint revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Kernel target | `cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16` |
| Topology | TP4 on `spark0` through `spark3`; no PP and no speculation |
| Workload | B1: one 128-token prompt, one request, 128 output tokens |
| KV state during timed region | Prompt prefill complete and its KV resident |
| Timed boundary | First emitted token to last emitted token: 127 full decode intervals |
| Included | Scheduler, model kernels, TP collectives, sampling, runtime IPC, and streamed output observation over SSH |
| Excluded | Initial process connection and prompt prefill/TTFT |
| Runtime limits | one in-flight submission, one active sequence, one input row, one resident sequence |
| Precision route | Exact target above; no speculative draft model |

The prompt is prefetched by the same request.  The benchmark starts its decode
clock at the first emitted token, after prefill has populated KV, and observes
every subsequent token at the client.  This is therefore end-to-end cached-KV
decode, not a kernel-only timer or a prefill-plus-decode blended rate.

### Four-run result

| Run | Decode tok/s | Median inter-token | p95 inter-token |
| ---: | ---: | ---: | ---: |
| 1 | 32.4299 | 30.6386 ms | 31.6988 ms |
| 2 | 32.6948 | 30.3928 ms | 31.4345 ms |
| 3 | **32.8075** | **30.2758 ms** | **31.1335 ms** |
| 4 | 32.3374 | 30.7020 ms | 32.2516 ms |
| Mean | **32.5674** | 30.5023 ms | - |

The four rates span 32.3374--32.8075 tok/s.  Their median is 32.5623 tok/s.
All four candidates emitted the same 128-token sequence, and that sequence is
byte-for-byte identical to the retained control output.  The comma-separated
token-id sequence hashes to
`211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`
with SHA-256.

### Identity and retained evidence

This milestone was measured from a scratch candidate based on
`main@7bf94d8bf087d5c9584e2627d03ea8408ce13c22`; it is a measured engineering
milestone, not yet a merged-main production qualification.  The candidate
fused the first recursive-doubling reduction with construction of the next
relay payload.  Its driver SHA-256 is
`212cee3f901e513936cea20b305305a6ea18df28d577b5adb6f27a25924c0a8e`.

The raw receipts remain outside Git because they include run-local paths and
full event streams.  Their retained paths and SHA-256 digests are:

| Receipt | SHA-256 |
| --- | --- |
| `/private/tmp/dsv4-combine-relay-run1.json` | `ba792c90f5f484bb49f6a92e95ef807d1e9efcd30d0dad06fb96b76481be2321` |
| `/private/tmp/dsv4-combine-relay-run2.json` | `5e7f8307bd29d42cea0aead9fd09d8edd63c02e459ce4e49a89cb75bbe9d32fc` |
| `/private/tmp/dsv4-combine-relay-run3.json` | `ff130bc6d751543b073e340772132457eff501b160588883fb198e18f19feaa8` |
| `/private/tmp/dsv4-combine-relay-run4.json` | `6e0e4ea9afdbd4c40eebea1c6d235062761c1811241b7fce28a69a771d2767ec` |
| `/private/tmp/dsv4-combine-relay-control2.json` | `ff9530527512f6ef1d128ae3e51dc17a1a481e430666742701482258be2d91b7` |
| `/private/tmp/dsv4-tp4-pp4-b1-compsec076-o128.json` | `e498f1fc88854044eafa64c41ce308b73d54f0a351fe156a513d7ff7ca630ead` |

Reproduce the measurement after deploying the exact driver to an otherwise
identical four-rank runtime:

```sh
python3 tools/model_stream_decode_benchmark.py \
  --output /private/tmp/dsv4-tp4-b1-o128.json \
  ssh spark0 /path/to/sparkpipe_model_batch \
  --deployment /path/to/model_resident.json \
  --runtime-root /path/to/runtime \
  --batch /path/to/b1-prompt128-output128.json
```

An accepted replacement milestone must retain at least three unprofiled runs,
use one real request for B1, time at least 127 post-prefill token intervals,
record exact source/driver/model identities, and prove emitted-token parity
with the accepted control.  Profiled, simulated, projected, kernel-only, or
multi-request aggregate figures must be reported separately.

## Current hill-climb boundary

The measured candidate still launches 130 graph islands per token.  A proposed
single graph exposed a real scheduling deadlock between an external host-RDMA
collective and CUDA graph execution; the old runtime had silently disabled
that path.  The silent downgrade is being removed.  No single-graph speedup is
claimed until it completes the exact workload above and preserves its token
sequence.
