# TP16 hardware-wait and serving measurements

The isolated GLM5.3 Flash run from source
`606cc825ca77d6ca9c3f20828b8396871e0c0f54` produced 17 token events, then remained
in admission BUSY without a completed event. It is a partial reliability result,
not a successful request or a sustained throughput qualification. The flushed
rank-zero CUPTI trace has all 121 returned buffers, zero dropped activity
records and no malformed/incomplete GPU intervals.

The measured decode window starts at output token 0 and ends at output token
16: 16 intervals in 1.480802838 seconds, or 10.80495 tokens/s for that partial,
profiled window. Startup and prefill are excluded. Same-host monotonic bounds
are `576116547247054` and `576118028049892` ns. Profiling perturbs timing; completed trace-disabled measurements follow below.

| Interval measurement | Median ms | Mean ms |
| --- | ---: | ---: |
| Output-to-output wall time | 82.957 | 92.550 |
| GPU activity union | 56.078 | 56.384 |
| Compute kernels | 54.647 | 54.754 |
| Mesh kernels | 1.354 | 1.367 |
| Device memory copies | 0.053 | 0.242 |
| Device memory sets | 0.020 | 0.021 |
| Uncovered GPU timeline | 26.657 | 36.167 |

Coverage merges overlapping intervals, excludes host-to-host copies, and clips
activity to each token window. Category coverage and independently calculated
medians are not additive. Gaps are not automatically network time. CUPTI records
show nonzero graph IDs on all 28,379 kernels in the independently checked first
13 decode intervals, confirming that captured kernels actually executed.

The separate rank-zero device counters over its last 16 completed chains average
1.125 ms source wait, 27.474 ms peer wait, 0.233 ms copy and 0.230 ms combine.
Average chain wall time is 87.723 ms; output-to-output time is 92.550 ms because
the observation boundaries differ. Wait counters include scheduling and daemon
progress. Copy/combine counters describe work also present in the CUPTI mesh
category; adding both would double-count it.

## Compute finding

Across 16 intervals, the leading kernel totals are 223.203 ms for one FP8 GEMM
specialization, 205.079 ms for BF16 GEMM, 194.212 ms for `Glm5NextHcMixKernel`,
and 69.142 ms for `Glm5NextHcSplitSinkhornKernel`. The two HC kernels alone average
16.460 ms per token, from 90 calls each per token.

The trace records HcMix grid `(1,1,1)`, block `(256,1,1)` and 16 KiB dynamic shared
memory. In `modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh`, one
block processes the complete 16,384-element input for B1. Eight warps each compute
three of 24 mixing outputs, serially across four 4,096-element tiles. The observed
mix kernels take roughly 135 microseconds. Sinkhorn launches one 64-thread block
but maps a complete row to one thread; only thread zero performs B1's twenty
iterations. Its observed kernel duration is roughly 48 microseconds.

A bounded HcMix experiment can partition independent output mixes across blocks,
repeat the unchanged RMS reduction per block, and preserve the original warp
reduction and tile accumulation order. This needs no new scratch or extra kernel
launch. Three-block and 24-block variants require bitwise comparison with the
unchanged kernel and both repeated-weight and rotating-weight timing before a
production change. Sinkhorn is a separate optimization; it is not changed by
that experiment. DSV4 already has split-K HC machinery, but it reassociates the
reductions and is not an exact-order substitute.

## Reproduction and receipts

Remote root: `spark0:/tmp/sparkpipe-perf-606cc825ca77-graph-r2/`.
Raw `rank0.cupti.log` SHA256:
`a13f25a9ad5f76af7a04fe5086cd6abb58def929ab1f2189a53d3be74b89de23`.
`batch.events.jsonl` SHA256:
`1591546c3c3ba4a639102b4a7398a16cd78c493d2fec2d7d08a7ab8726bd43aa`.
The local receipt directory is
`/private/tmp/sparkpipe-pr1082-receipts/606cc825-r2-cupti/`; it contains the raw
files, `token0-to16.json` and `per-token.json`. Separate all-rank device counters
are in `/private/tmp/sparkpipe-pr1082-receipts/fleet-performance/sparkpipe-perf-606cc825ca77-graph-r2/collective-partial.json`.

```sh
python3 tools/tp_cupti_trace_report.py rank0.cupti.log --clock monotonic --start-ns 576116547247054 --end-ns 576118028049892 --output token0-to16.json
```

The profiler and timing tool are described in [TP_CUPTI_TRACE.md](TP_CUPTI_TRACE.md).

## Completed requests after the admission repair

The first fleet trace exposed a separate serving defect at submission 193:
zero-row `CACHE_PUBLISH` was accepted by the module, but its decision reported
zero available dispatch slots. The generated driver correctly kept returning
BUSY. Commit `bb94f3eba599a20e2eb7663ed9bdf24f7e75e821` reports the actual
slot availability before this control branch. The regression executes the real
GLM module through the generated admission wrapper; the pre-fix code fails.

A fresh coherent build of that commit passed CUDA compilation, retained GPU
module publication, driver linking/loading and artifact checks. Its bundle
SHA256 is `a98fc545b31f33b26872fb04a384618a3a4345acbb8b72636f0161aa6ad08528`;
the loaded driver artifact SHA256 is
`91e101615753403a428e175637e83a9a41156038470dc9b566d8caf427461ff2`.

All sixteen ranks initialized with hardware waits, graph mode, pinned experts,
FP8 weights, one active sequence, one execution row and the same 176-token
quality-fixture prompt. Context capacity was 512, logical/physical page counts
128, and explicit backing capacity 2 GiB. Each rank used an isolated weightd,
mesh records, socket, ports and cache root; existing services remained present
and were idle in the preflight snapshot. CUPTI was disabled for these runs.

| Request | Output tokens | Completion | Decode tok/s | Median interval | First token |
| --- | ---: | --- | ---: | ---: | ---: |
| First | 32 | status 0 | 11.384 | 84.352 ms | 17.071 s |
| Repeat | 32 | status 0 | 9.814 | 99.507 ms | 14.261 s |

Decode rate uses the 31 first-to-last output-token intervals. Engine timestamps
independently give 11.3835 and 9.8136 tok/s, agreeing with stdout arrival timing.
Both requests cross the formerly stuck cache boundary and complete their final
publication/release. Both produce the identical token-list SHA256
`6955a6afeadbb5b8a88399e8f20a275a22d633fcbafa578ed63e5fdc4e7f87da`.
This establishes repeatability, not comparison with a trusted model oracle.
Each CLI invocation creates a new coordinator engine and reports zero cached
prompt tokens, so this repeat is not a persistent-engine prefix-hit test.

The difference between the runs is retained, not averaged into a claimed
14 tok/s result. Stage-profile output on the first run reported 16 dropped
records; it cannot support a complete stage-by-stage decomposition. The earlier
CUPTI trace has separate zero-loss accounting and remains the compute attribution
source. All 32 owned fleet processes subsequently exited cleanly with no forced
kills; process absence was verified on every rank.

Retained local evidence is under
`/private/tmp/sparkpipe-pr1082-receipts/fleet-performance/sparkpipe-perf-bb94f3eba599-graph-r3/`,
with the first request in `pass1/` and the second at the root. Event-file SHA256s
are `6be8fc7af72383fd10a09d1a1a3ce0e8f0e9c498c37865762d21a11e8b6f25c7`
and `ed94ef6a887e246e871e988b557f034d6462dd75469f9dc2ffb0014d950d528d`.
The corresponding measurement JSON SHA256s are
`4b758fb8f6e9287adeca1c691c3a74f92dbb5b7ed7d4b08eaea7bbdb71f8a04e`
and `67b0b73b50a82034203dd92602b9a9333c0fa7921b38556925710730ab804777`.

## HC output partition and exact-kernel gate

Production commit `1a08cb81583c93e749707fff80504f0012f1097b` changes only five
added and two removed lines in `layer.cuh`: HcMix uses three blocks per row,
partitioning independent output mixes. Every block performs the same input
staging and RMS reduction; each selected warp retains its original element,
shuffle and tile accumulation order. No scratch, extra launch, runtime mode or
Sinkhorn change was added. The 24-block prototype was only about one percent
faster than three blocks while repeating staging/RMS eight times more often.

The retained `tests/test_glm5_next_hc_mix.cu` includes the actual complete GLM
CUDA translation unit and a frozen pre-change baseline function. After reversing
its symbol rename, the fixture is byte-for-byte equal to the unchanged function,
SHA256 `c87f38f32596a7ba4142db26ea898173ed2d9884c4c0ed0561885ea1ba30fede`.
It requires finite and bitwise-equal outputs for rows 1, 3 and 5 with signed,
exponent-varied and cancellation data. All nine cases passed on Spark0 at
source `012f16a4f25a89cf8ff1745b4bcdfed56b6bebcf`, explicitly LAZY/LAZY, in
0.890 seconds. Both baseline and actual production kernel use captured graphs
for the timing comparison.

| 90 B1 HC calls, eight samples after two warmups | Unchanged median | Three-block median |
| --- | ---: | ---: |
| One repeatedly reused weight matrix | 4.785056 ms | 1.906800 ms |
| 90 distinct weight matrices | 11.975744 ms | 4.367024 ms |

The rotating-weight microbenchmark improves 2.74 times and saves 7.609 ms per
90 calls. This is a same-process kernel comparison, not an end-to-end throughput
claim. The process exited 0, its PID disappeared, and the two existing GPU owners
and their allocations were unchanged. No service was restarted by the test.

Binary SHA256:
`2e0ad3c13b6d6972c19ed155d6a3b50f1c3f6379ac312d84822bd8f35eb1e325`.
Remote logs are in `spark0:/tmp/sparkpipe-hc-regression-012f16a4/`; the small
[receipt](receipts/glm5-next-hc-mix-012f16a4.json) retains source, binary, timings
and process cleanup. The [GPU inventory](SERVING_FUZZ_COVERAGE.md) lists the
explicit build/run target. Fleet output parity and throughput are measured separately below.


## Completed HC fleet and persistent API requests

An immutable serving bundle from production commit
`1a08cb81583c93e749707fff80504f0012f1097b` passed the same CUDA compilation,
GPU publication, driver linking/loading and artifact checks. Bundle SHA256:
`b3ab05e568a8fe1b3fef8672f4f0b62d0a8914600eb3cadc28ba3de97a94ff59`.
The TP16 geometry, FP8 packs, prompt, graph mode, pinned experts, B1 limits and
hardware waits match the preceding fleet run; CUPTI remained disabled.

| Request | Output tokens | Completion | Decode tok/s | Median interval | First token | Cached prompt tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CLI | 32 | status 0 | 11.590 | 80.792 ms | 17.646 s | 0 |
| Persistent API, first | 32 | status 0 | 12.811 | 77.420 ms | 13.356 s | 0 |
| Persistent API, second | 32 | status 0 | 11.770 | 82.354 ms | 3.944 s | 128 |

All three token lists match the previous completed baseline hash
`6955a6afeadbb5b8a88399e8f20a275a22d633fcbafa578ed63e5fdc4e7f87da`.
The two API requests used one engine in PID 3115283, distinct request IDs
100001/100002, HTTP 200, and actual engine completion. The second request reused
128 of the 176 prompt tokens and produced identical continuation tokens. This
qualifies this persistent-engine prefix-hit case; it does not qualify arbitrary
batch sizes or every cache eviction/movement scenario.

The HC microbenchmark establishes a kernel improvement, while these variable
end-to-end results do not establish a large throughput gain or 14 tok/s. The CLI
stage-profile summary again reported 16 dropped records, so it is not used as a
complete phase decomposition. All 32 owned fleet processes and the API exited 0
with their absence verified; no forced kill was needed.

Remote root: `spark0:/tmp/sparkpipe-perf-1a08cb81583c-graph-r4/`. Retained local
receipts are in `/private/tmp/sparkpipe-pr1082-receipts/fleet-performance/sparkpipe-perf-1a08cb81583c-graph-r4/`.
Event-file SHA256: `4f520d80d1fac007112ae503a9001ca12725b7ccb6208782c0f05056e14d04f5`.
CLI measurement SHA256: `39019456b3f74f698131ea5a4440597d7fc121639752843c76b9af92902e2d65`.
API JSONL SHA256: `077949adbaab537aaebed12d0c522b854c6d37c65fdfaeaab64f7c4930c95980`.
API summary SHA256: `0fc153483fcc6e0df8f33b31458e80fdecdc51b6b89e45712138297944710beb`.

## Useful-byte memory roofline

The rank-zero FP8 pack is 21,706,046,976 bytes, including every routed expert.
One token selects eight of 288 experts in each of 42 routed layers, totaling
544,997,376 bytes of expert FP8 weights and scales. Counting active weights,
recurrent state, window state and attention KV gives approximately 2.197–2.293 GB
of useful data per rank per token at context lengths 176–208.

[NVIDIA specifies 273 GB/s](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
for DGX Spark memory bandwidth. The useful-byte streaming ceiling is therefore
119–124 tokens/s; the measured 11.38 tokens/s uses roughly 9–10 percent of that
ideal. Dividing by the entire pack would wrongly count all inactive experts.
This is not measured DRAM utilization or an achievable serving target: actual
physical traffic can reread data, and compute, transport and scheduling add
time. Memory-controller counters are needed to claim a fraction of actual DRAM
bandwidth. At 12.81 tokens/s, the same useful-byte ratio is approximately
10–11 percent.

The reproducible rank-zero pack-directory accounting, read commands, per-group
formulas and metadata/source hashes are retained in
`reviews/sparkpipe-pr1077-20260922/implementation/roofline-20260922/` in the
review workspace. The source revision is
`84c6a6aa9497188e15a635ba793b0f95a79b1033`; the pack sidecar hash is identified
as a sidecar value, not a fresh hash of the entire 21.7 GB file.
