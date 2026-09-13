# GLM-5.2 MTP cycle decomposition receipt

## Scope

This receipt measures the B64 FP8 PP13 ring after PR #481 on merged main
`3d662d7cc1cad4bc3a1ed863a5be4cf806241585`. The profiling release was
`glm52-fp8-3d662d7-b64-cycle-timing-r1` with MTP enabled, 64 active lanes,
16,384 physical KV tokens, and 256 logical KV blocks.

The measured request used greedy decoding and the prompt:

```text
Explain why unsigned integer overflow can corrupt a C array-length calculation,
and show a safe checked-add helper.
```

It had 22 prompt tokens and requested 64 output tokens. One eight-token request
was completed first to warm the release. Historical logs were excluded using
per-host byte offsets recorded after warmup.

## End-to-end result

- 64 tokens completed in 19.583229 seconds: 3.268 tokens/second.
- The output token ID SHA-256 was
  `9a75ab164284854a2f35d4e24b9ad117727d7755fcbe26b6bb709d480a559990`.
- That hash exactly matches the retained pre-instrumentation MTP parity receipt.
- The request used 22 MTP verify dispatches.
- The gateway drained to zero live requests, queued requests, event backlog,
  and dropped events.
- This proves token-stream parity for this prompt. General model accuracy is
  not measured by this receipt.

## Gateway timing

| Dispatch | Count | Mean flight | P50 flight | P95 flight |
|---|---:|---:|---:|---:|
| MTP verify, six rows | 22 | 809.655 ms | 804.501 ms | 845.982 ms |
| Plain decode, one row | 2 | 234.598 ms | - | - |
| Initial MTP producer | 1 | 1,281.495 ms | - | - |

The verify completion-to-next-submit gap was 0.333 ms mean, 0.330 ms P50,
and 0.363 ms P95. Gateway cadence is not the verify bottleneck.

## Resident timing

Each value below is submit-to-completion for one six-row verify packet. Queue
wait is not added to execution time: it is the cumulative arrival offset in the
serial ring and therefore grows with rank as expected.

| Rank | Verify execute P50 | Verify execute P95 | Verify execute mean | Verify queue P50 |
|---:|---:|---:|---:|---:|
| 0 | 49.911 ms | 52.360 ms | 49.943 ms | 0.011 ms |
| 1 | 62.516 ms | 67.020 ms | 62.991 ms | 49.237 ms |
| 2 | 57.958 ms | 63.609 ms | 58.685 ms | 113.467 ms |
| 3 | 56.502 ms | 60.135 ms | 56.773 ms | 171.941 ms |
| 4 | 59.871 ms | 63.766 ms | 60.154 ms | 229.334 ms |
| 5 | 56.422 ms | 59.151 ms | 55.764 ms | 291.683 ms |
| 6 | 57.473 ms | 60.632 ms | 56.799 ms | 347.647 ms |
| 7 | 57.712 ms | 63.554 ms | 58.476 ms | 405.392 ms |
| 8 | 58.348 ms | 62.208 ms | 57.173 ms | 463.965 ms |
| 9 | 59.047 ms | 64.163 ms | 58.824 ms | 522.495 ms |
| 10 | 58.330 ms | 64.220 ms | 58.893 ms | 582.518 ms |
| 11 | 57.821 ms | 63.253 ms | 58.230 ms | 638.719 ms |
| 12 | 99.657 ms | 105.685 ms | 99.978 ms | 694.024 ms |

The sum of per-rank mean execution time was 792.685 ms for verify and
221.494 ms for the two measured plain rows, a 3.579x ratio. The verify sum is
only 16.970 ms below the 809.655 ms gateway flight mean. This closes the
latency ledger: the slowdown is resident verify execution across all stages,
with an additional final-rank verifier/head tail. It is not hidden gateway
polling or unexplained network delay.

## Corrected attribution

The measured result falsifies the prior estimate that a six-row verify should
cost only 1.2x to 1.5x a one-row pass. The receipt originally left graph replay
and the prepared F32 linear-plan row count as competing explanations. A source
audit now resolves that ambiguity.

`SparkGlm52ResidentDecodeStageLinearPlanRequiredPreparedActiveRows` returns one
prepared row for every F32-output linear plan, regardless of the active row
count. `SparkGlm52Sm121RequiredDecodeStageLaunchPreboundLinearPlan` then handles
the mismatch by setting `launch_count` to the active row count and issuing one
`cublasLtMatmul` for each row. A six-row verify therefore executes six separate
M=1 matmuls for every affected BF16-weight/F32-output plan instead of one
multirow matmul.

This is the source-proven cause of the multirow launch amplification. It
directly explains the measured 56-63 ms middle-rank verify times versus the
16-17 ms one-row stage range and the aggregate 3.579x ratio. Missing graph or
full-stage replay is no longer required to explain this receipt. The growing
queue-wait values remain cumulative ring arrival offsets, not hidden host or
network overhead.

The same mechanism is present at wider plain-decode batches: an active count of
64 can issue 64 M=1 launches for each affected plan. This is a source-confirmed
amplification path and a strong explanation for the retained B64 result, but its
causal share and the post-fix B64 throughput remain unmeasured until the hardware
A/B below is complete.

## Prepared correction

The prepared correction selects the required stage-plan batch bucket for every
BF16-row-major F32-output plan. The launch path requires the prepared row count
to equal that bucket, issues exactly one multirow `cublasLtMatmul`, and fails
closed on a mismatch. It does not restore a scalar or per-row fallback.

Rows between the active count and the prepared bucket are computed but ignored.
That is valid for this plan contract because buffers are sized to the maximum,
the matmul beta is zero, rows are independent, and downstream consumers use only
active rows. Stable bucket selection also avoids descriptor preparation and
autotuning on small cohort-width changes. FP8-E4M3 cublasLt plans are unchanged.
The graph signature already includes the descriptor pointer, so a bucket change
recaptures while steady-state launches replay the matching graph.

A companion control-path change posts the hidden-input receive before builder
preparation and marks the input as pre-received for the driver. This removes
repeated KV, upload, metadata, and linear-plan preparation on each BUSY retry.
It is a separate optimization, not the cause of the measured 3.579x execution
ratio. Builder and driver artifacts carrying this contract must be deployed
together.

The handoff reports passing host tests. SM121 compilation and live GB10
execution have not yet measured the correction, so no speedup is claimed here.

## Hardware acceptance gate

The corrected build must rerun the retained MTP verify and B64 benchmarks with
unchanged inputs and measurement settings. Acceptance requires:

- output token SHA-256 remains
  `9a75ab164284854a2f35d4e24b9ad117727d7755fcbe26b6bb709d480a559990`;
- middle-rank six-row verify execution is below 25 ms per rank; and
- B64 throughput exceeds 150 tokens/second.

The prepared-change projections are approximately 17-20 ms per middle-rank
verify, an MTP cycle near 300 ms, about 10 tokens/second for the retained B1 MTP
workload, and B64 toward the 260-330 tokens/second pipeline-model range. These
are acceptance hypotheses, not measured results.

`SPARKPIPE_MTP_CYCLE_PROFILE=1` was present in the resident process environment
but emitted no new epilogue lines in this run. The per-rank and gateway timing
still closes the full flight ledger, but the missing emission is a separate
instrumentation gap.

## Deployment finding

During the first profiling startup, the Spark2 rank daemon reached its resident
during the READY transition, exited with `status=8`, and was not retried by the
release manager. The resulting warmup was invalid and was discarded. Reapplying
the rank role after READY restored the ring. The benchmark above began only
after all 12 rank daemons were confirmed alive and the gateway was restarted at
zero work. Startup must wait for READY or retry this transient failure.

## Restored ring

After measurement, all timing variables were removed by deploying
`glm52-fp8-3d662d7-b64-clean-r1`. The clean release returned token ID 10397,
text ` OK`, for the one-token `Say OK. OK.` oracle in 0.822206 seconds and then
reported zero live requests, queued requests, backlog, dropped events, and
blockers.
