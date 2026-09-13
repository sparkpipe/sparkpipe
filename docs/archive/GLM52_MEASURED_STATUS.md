# GLM-5.2 Measured Status

This file is the authoritative SparkPipe GLM-5.2 status ledger. A feature is
not working because code, a plan, a flag, a callback, or a test fixture exists.
It earns status only from a named release and retained measurement output.

## Status Vocabulary

- `MEASURED`: exact merged commit, release id, generation, hardware shape,
  command, raw output, and numerical result are retained.
- `OBSERVED`: the live runtime emitted the named event or counter. This proves
  activity, not correctness or performance.
- `NOT_MEASURED`: no acceptable retained measurement exists.
- `NOT_WORKING`: the active runtime rejects, omits, or cannot reach the path.

Compilation, host tests, isolated source coverage, and capability declarations
do not change any runtime status by themselves.

## Accepted Measurements

The latest retained matched B1 comparison is:

```text
commit:          f5364187449c45e14d5013a2e5b8243a2beabf4d
plain release:   glm52-fp8-main-f5364187-b1-plain-work-order
plain generation: 20260714043233
MTP release:     glm52-fp8-main-f5364187-b1-mtp1-work-order
MTP generation:  20260714042436
topology:        13 Spark ranks
live lanes:      1
transport:       host-staged TCP
```

Raw output is under `diagnostics/glm52_mtp_b1_f536418_20260714/`. The detailed
receipt is `docs/GLM52_MTP_B1_MEASUREMENT_20260714.md`.

| Surface | Status | Retained result |
| --- | --- | --- |
| B1 plain decode | `MEASURED` | 64 tokens in 17.236 s, 3.713 token events/s, 0.454 s average TTFT |
| B1 serialized one-draft MTP | `MEASURED` | 64 tokens in 15.835 s, 4.042 token events/s, 0.689 s average TTFT; advisor commit e10c0b0 was not deployed |
| Matched MTP gain | `MEASURED` | 1.089x token throughput, 8.135% lower average latency, 51.647% higher TTFT |
| Greedy output parity | `OBSERVED` | all four requests emitted the same 32 token IDs and text |
| MTP activity | `OBSERVED` | 22 drafts, 22 verifies, 16 accepted drafts, 6 rejected drafts, 48 decode dispatches |
| Model accuracy | `NOT_MEASURED` | no corpus, perplexity, long-context, or retained reference-equivalence score |
| DSpark correctness and throughput | `NOT_MEASURED` | no retained full-ring token, acceptance, or throughput receipt |
| B4/B16/B64 serving | `MEASURED` | 7.58 / 20.99 / 58.22 token events/s end-to-end; commit 1d7f176b, release glm52-fp8-main-1d7f176b-b64-perf, generation 20260713220928; raw output under `diagnostics/glm52_b64_api_performance_20260714/` |
| B256/B1024 serving | `NOT_MEASURED` | no retained end-to-end receipt |
| Ring occupancy | `MEASURED` | per-rank occupancy 86-95 ms against 15-19 ms kernels, reattributed to the F32 linear-plan M=1 launch loop; see `docs/GLM52_PP13_MULTIROW_LINEAR_PLAN_FIX_20260718.md` |

The prior retained full-ring performance measurement is:

```text
commit:      9cc386a4ad1fa6827e7e36fba8fb1b4a7e16f00c
release:     glm52-fp8-main-9cc386a-b16-scaled-gemm
generation:  20260712013500
topology:    13 Spark ranks
live lanes:  1
transport:   host-staged TCP
```

Raw output is under `diagnostics/glm52_fp8_scaled_gemm_20260712/`. The detailed
receipt is `docs/GLM52_FP8_SCALED_GEMM_ACTIVATION_20260712.md`.

| Surface | Status | Retained result |
| --- | --- | --- |
| B1 full-ring decode | `MEASURED` | 8 tokens in 2.571 s, 3.112 token events/s; 16-token run, 3.272 token events/s |
| Four concurrent clients | `MEASURED` | 16 total tokens in 5.545 s, 2.884 token events/s aggregate; staircase completion proves serialization |
| Stage-0 six-layer FP8 CUDA | `MEASURED` | 14.626 ms packaged-driver time, B1, one capture and two replays |
| End-to-end token path | `OBSERVED` | token 10397 for `Say OK. OK.` and ` Paris\nQ:` for a factual prompt |
| Model accuracy | `NOT_MEASURED` | no corpus, perplexity, long-context, or reference-equivalence score |
| B4/B16/B64/B256/B1024 serving | `NOT_WORKING` | measured backend accepts one request, active sequence, and lane |
| Multi-request GPU batching | `NOT_WORKING` | four clients were serialized before GPU admission |
| Bulk prefill | `NOT_WORKING` | prompt processing was token-serial with maximum prompt count one |
| Incremental HTTP streaming | `NOT_WORKING` | SSE body was buffered into one Content-Length response |
| JIT KV prefetch | `NOT_WORKING` | active request API omitted the JIT flags and counters remained zero |
| FP8 KV cache | `NOT_WORKING` | active builder allocated BF16 cache fields |
| DSA long-context inference | `NOT_MEASURED` | no retained official long-context parity or end-to-end throughput receipt |
| MTP throughput | `NOT_MEASURED` | no retained end-to-end token/s comparison against the same base release |
| DSpark correctness and throughput | `NOT_MEASURED` | no retained full-ring token, acceptance, or token/s receipt |
| Transport throughput | `NOT_MEASURED` | host-staged TCP identity was observed; no retained per-hop throughput result for this release |
| 32K/256K/1M contexts | `NOT_MEASURED` | no retained full-ring accuracy and performance matrix |
| Zero fallback execution | `NOT_MEASURED` | no retained runtime backend-identity and fallback-count receipt across every layer |
| Final LM-head throughput | `NOT_MEASURED` | no isolated or end-to-end retained timing for the active scalar BF16 head |

## Current Release Observation

The active measured release after the matched comparison is:

```text
commit:      f5364187449c45e14d5013a2e5b8243a2beabf4d
release:     glm52-fp8-main-f5364187-b1-mtp1-work-order
generation:  20260714042436
topology:    13 Spark ranks
live lanes:  1
```

Two sequential 32-token requests completed with identical output. Final health
reported no live request, no queue backlog, no dropped event, and
`mtp_status=OBSERVED`. Accuracy remains `NOT_MEASURED`; this result covers one
prompt and measures throughput, not model quality.

## Prior Release Observation

The honesty rewrite was deployed after the accepted performance measurement:

```text
commit:      e459d41df92f3bdaf2f8265e66151f97249c46f0
release:     glm52-fp8-main-e459d41-b1-measured-status
generation:  20260712063211
topology:    13 Spark ranks
live lanes:  1
```

All 13 residents reported ready and installed the manifest builder and driver
hashes. The first greedy `Say OK. OK.` request after restart returned token 16
(`1`). The immediate identical request returned token 10397 (` OK`). A distinct
factual prompt returned ` Paris\nQ:`. Therefore:

| Surface | Status | Result |
| --- | --- | --- |
| End-to-end event path | `OBSERVED` | accepted, token, done; queues drained |
| Maximum serving lanes | `OBSERVED` | prefill 1, decode 1 |
| Clean-start determinism | `NOT_WORKING` | identical requests returned different first tokens |
| Accuracy | `NOT_MEASURED` | smoke outputs are not an accuracy score |
| Performance | `NOT_MEASURED` | no timing run was accepted for this instrumented release |
| MTP execution | `OBSERVED` | 3 draft tokens, 1 verify dispatch, 1 accepted draft, 2 committed tokens, 2 rejected tokens; no throughput comparison |

The retained raw responses are under
`diagnostics/glm52_measured_status_e459d41_20260712/`.

The isolated stage result is not an end-to-end throughput claim. Dividing one
second by 14.626 ms gives a theoretical filled-pipeline ceiling, not measured
serving performance.

## Live Health Contract

`/health` uses schema `sparkpipe.runtime_observation.v1`. It reports exact
release identity, configured limits, bound interface information, local control
readiness, and counters observed since process start. Local readiness does not
claim that all 13 ranks are connected. It must always report accuracy and
performance as `NOT_MEASURED`; those properties require an external
reproducible benchmark.

`OBSERVED` in health means only that the named runtime event occurred. It must
never be rendered or described as validated, production-ready, accurate, fast,
or supported at an unobserved batch or context size.

## Measurement Admission

A new result may replace a row only when its receipt contains:

1. Full merged Git commit, immutable release id, and generation.
2. Installed artifact hashes on all 13 ranks.
3. Exact request, concurrency, prompt/context, token count, and temperature.
4. Actual maximum active-sequence and lane counters from the run.
5. Bound backend and transport identities plus fallback counters where relevant.
6. Raw timestamps, tokens, errors, and post-run queue health.
7. Separate first-token, steady decode, aggregate, prefill, stage, and transport
   timings when any of those numbers are claimed.

If any required evidence is absent, the status remains `NOT_MEASURED` or
`NOT_WORKING`.
