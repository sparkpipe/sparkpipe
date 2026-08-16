# SparkPipe Performance Status

This file is the only current performance ledger. Measurements include exact
scope and identity. Projections are kept in a separate section and never count
as achieved results.

## Measured fabric

### Eight pairwise direct links

On 2026-08-13 all eight direct pairs ran simultaneous traffic in both
directions. Every pair passed the 80 Gb/s-per-direction gate.

| Metric | Result |
| --- | ---: |
| Slowest observed direction | 91.669 Gb/s |
| Fastest observed direction | 105.907 Gb/s |
| Combined sixteen-direction throughput | 1.643 Tb/s |

These useful rates explain why the nominal 200 Gb/s direct links behave like
approximately 100 Gb/s links after the GB10 PCIe limit. A 100 Gb/s switched
port therefore does not reduce practical per-port throughput.

### Combined direct and switched rails

The isolated TP4 two-port characterization measured:

| Metric | Result |
| --- | ---: |
| Simultaneous two-port aggregate ceiling | 213.687 Gb/s |
| Best sustained 14 MiB split ring | 193.018 Gb/s |
| Ceiling utilization | 90.328% |

The sustained result used two counter-rotating rings, four streams, four
credits, and one verbs QP per route. It is an isolated transport measurement,
not model decode throughput.

### TP4 algorithm crossover

The retained BF16 profile now has three regions: direct all-to-all through
80 KiB, recursive doubling above 80 KiB and below 640 KiB, and the
counter-rotating split ring at and above 640 KiB.

The small-payload boundary was measured on 2026-08-14 with the production
selector and exact recursive-tree BF16 fused kernel on `spark4` through
`spark7`, one credit and one in-flight collective. Each value is the mean of
the four ranks.

| Payload | Direct all-to-all | Recursive doubling | Selected |
| ---: | ---: | ---: | --- |
| 8 KiB | **43.369 us** | 52.205 us | direct all-to-all |
| 80 KiB | **66.111 us** | 71.012 us | direct all-to-all |
| 96 KiB | 75.862 us | **75.244 us** | recursive doubling |

At 8 KiB the one-round path removes 8.836 us per collective, or 16.9%. With
130 such collectives per B1 token, its isolated upper bound is approximately
1.149 ms/token. This is measured collective latency, not an end-to-end decode
result. The candidate is based on `main@6cfd216bab72368923aff08819de08ea331d26b2`
and is not a merged-main production qualification.

Raw rank receipts are retained under
[`qualification/dsv4/performance/tp4_b1_20260814`](qualification/dsv4/performance/tp4_b1_20260814).
The fused BF16 kernel also matched the current two-step recursive accumulator
bit-for-bit for 1,048,576 deterministic finite inputs across all four TP ranks;
that receipt is retained at
[`qualification/dsv4/collectives/tp4_tree_bitwise_20260814/result.json`](qualification/dsv4/collectives/tp4_tree_bitwise_20260814/result.json).

### Hc row-adaptive boundary kernels

The same candidate tiles the Hc pre/post element space only when the active
row count would otherwise expose fewer than 16 blocks. Two hardware sweeps on
an exact `sm_121a` archive produced zero BF16 mismatches against the original
mapping at B1, B2, B4, B8, B16, B64, B128, B256, B512, and B1024. At B1,
pre-reduce fell from 6.154-6.155 us to 2.058-2.089 us and post fell from
8.199-8.208 us to 3.305-3.459 us. B16 and above retain one tile per row. The
B256-B1024 timings use only 30 iterations and are retained as correctness-sweep
data, not as performance claims.

The raw rows, timings, artifact hashes, and build metadata are retained in
[`hc-row-adaptive-bitwise.json`](qualification/dsv4/performance/tp4_b1_20260814/hc-row-adaptive-bitwise.json).
These are isolated kernel measurements, not an end-to-end decode result.

The larger-payload boundary remains:

| Payload | Recursive doubling | Split ring | Selected |
| ---: | ---: | ---: | --- |
| 576 KiB | 119.149 us | 121.229 us | recursive doubling |
| 640 KiB | 131.746 us | 122.962 us | split ring |
| 1 MiB | 198.642 us | 143.841 us | split ring |
| 14 MiB | 2.951 ms | 1.900 ms | split ring |

TP8 and TP16 require their own three-algorithm profiles. The TP4 threshold is
not a universal constant.

## Measured model performance

### Selected exact TP4 B1 stack: canonical Query plus fused KV and indexer

The 2026-08-14 and 2026-08-15 interleaved full-ring evidence selects the lean
stack with canonical separate Query arithmetic and the retained KV plus indexer
post fusions. Every experiment below used one real request, resident prompt KV,
128 generated tokens, no speculation, TP4 on `spark4` through `spark7`, and
127 timed decode intervals after the first emitted token.

| Experiment | Control mean tok/s | Candidate mean tok/s | Direct delta | Positive pairs | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| CSA active-slot | 38.0435 | 38.4009 | **+0.9395%** | 3/3 | Retain |
| Pre-post stack vs resident chain | 38.0019 | 40.2541 | **+5.9266%** | 3/3 | Retain |
| KV plus indexer post fusions | 40.1901 | 40.4007 | **+0.5241%** | 3/3 | Retain |
| Query fusion alone | 40.3437 | 40.0519 | **-0.7232%** | 0/3 | Reject |
| All-post vs lean selected stack | 40.3869 | 40.4596 | **+0.1800%** | 4/8 | Inconclusive; select lean |
| Projection-precollective overlap | 40.4164 | 40.1835 | **-0.5763%** | 0/3 | Reject production integration |
| Integrated lean live revalidation | 40.1503 | 40.4553 | **+0.7597%** | 3/3 | Retain observation; no new code attribution |
| Cooperative Hc finalize plus pre-reduce | 40.4753 | 40.3402 | **-0.3336%** | 0/3 | Reject and remove |

The CSA result is a repeatable bit-exact gain, not a failure: the raw receipt's
arbitrary one-percent acceptance threshold is explicitly superseded by the
adjudication. The Query fusion is rejected because its direct incremental test
regressed in all three pairs. Its all-post stack then won only four of eight
pairs by 0.1800% in aggregate. Under **Solutions/(codesize^2)**, that mixed
result does not justify retaining extra Query-fusion source, so canonical Query
is selected.

The pre-post stack's +5.9266% and the KV-plus-indexer increment's +0.5241%
compose multiplicatively to a **normalized +6.4817%**. That number is derived
from two separate direct experiments; it is **not** a directly measured total
for the integrated selected source. The CSA component is already represented
inside the pre-post stack and must not be added again.

All control and candidate O128 runs in the retained/rejected selection emitted
the same comma-separated, newline-terminated token vector, SHA-256
`211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`.
The available O24 gates matched
`18cd1dfc7c7fce04adf200916184cee952a185367412a1780c0d28486c42cbd8`.
Exact workload identities, driver identities, per-run rates, and raw receipt
SHA-256s are frozen in:

- [CSA active-slot adjudication](qualification/dsv4/performance/tp4_b1_20260814/csa_active_slot_adjudication_da7f/summary.json)
- [pre-post stack versus resident chain](qualification/dsv4/performance/tp4_b1_20260815/pre_post_stack_vs_resident_chain_a2bfd99f/summary.json)
- [KV plus indexer incremental result](qualification/dsv4/performance/tp4_b1_20260815/kv_indexer_runtime_inverse_07c63b06/summary.json)
- [standalone Query attribution](qualification/dsv4/performance/tp4_b1_20260815/query_runtime_inverse_e9116c07/summary.json)
- [lean versus all-post selection](qualification/dsv4/performance/tp4_b1_20260815/query_selection_kv_indexer_vs_all_post/summary.json)

The isolated projection-precollective
[PoC](qualification/dsv4/performance/tp4_b1_20260814/projection_precollective_overlap_da7f/poc_repeatability_receipt.json)
was positive, but the production full-ring A/B regressed by 0.6321%, 0.9574%,
and 0.1400% with all six O128 outputs exact. Production integration is
therefore rejected. The raw full-ring runs and aggregate decision are frozen in
the [rejection summary](qualification/dsv4/performance/tp4_b1_20260814/projection_precollective_overlap_da7f/full_ring_rejection_5e0a3b14/summary.json).

The cooperative Hc candidate was also positive in its isolated actual-shape
PoC: 93/93 exact comparisons and +13.1800%, saving 0.1606 ms/model step. The
real full-ring A/B nevertheless regressed in all three exact pairs by 0.3336%
in aggregate. Its production integration was removed; the PoC, rejected patch,
artifact identities, raw runs, and decision are frozen in the
[Hc rejection summary](qualification/dsv4/performance/tp4_b1_20260815/hc_cooperative_full_ring_rejection_a7ebd51c/summary.json).

The selected integrated lean source passed the GA validator, an exact O24
gate, and three exact interleaved O128 pairs. It measured 40.4553 tok/s mean
versus 40.1503 tok/s for the previous semantically identical lean build. The
positive 0.7597% observation is retained, but no separate optimization credit
is assigned because the production module source was byte-identical. The
[integrated-source receipt](qualification/dsv4/performance/tp4_b1_20260815/integrated_lean_3d962820/summary.json)
freezes the exact source and artifact hashes. The best contemporaneous lean
control mean is 40.4753 tok/s in the Hc rejection A/B. None of these branch
receipts is merged-main or zero-drift production qualification.

### Earlier retained milestone: exact combined plus LinearPair B1 wins

The unmerged candidate based on
`main@da7f91090c0d40729352b4e4180ad231971c90a2` produces **39.5202
decode tokens/s mean** over three interleaved TP4 B1 O128 pairs. The accepted
control measured 37.9797 tokens/s in the same sequence, so the candidate is
**4.0561% faster**. Every paired delta was positive. The normalized
increment over the preceding exact combined candidate is **1.3086%**; it is
retained because every repeatable bit-exact positive end-to-end delta counts,
with no minimum-gain threshold.

| Pair | Control tok/s | Candidate tok/s | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 37.9479 | 39.6750 | +4.5514% |
| 2 | 37.9765 | 39.2847 | +3.4448% |
| 3 | 38.0147 | 39.6008 | +4.1723% |
| Mean | **37.9797** | **39.5202** | **+4.0561%** |

This is one real request with resident prompt KV, 128 generated tokens, no
speculation, and TP4 on `spark4` through `spark7`. The timed boundary is the
127 decode intervals after the first emitted token, and includes scheduler,
model kernels, TP collectives, sampling, runtime IPC, and streamed client
observation. Mean end-to-end decode time fell from 26.3299 to 25.3035 ms/token,
a measured saving of 1.0263 ms/token.

The candidate combines active-slot CSA work, fused compressor ingest, explicit
B1 CTA policies, shared-dense W13, a certified head screen, and a common
runtime LinearPair work-shape selector. The selector lives in common kernel
code rather than a B1-specific driver fork. The target head, spine, KV, and
final rescore remain BF16: FP8 is used only to form a conservative head
candidate set, and every survivor is rescored against the untouched BF16 head
before the token and score are emitted. The screen has full rank-local
vocabulary capacity and dedicated scratch; it has no overflow fallback.

All six measured runs emitted the exact retained 128-token vector. Its
comma-separated token IDs plus trailing newline hash to
`211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`.
The exact archive passed the standard `sm_121a` GA B1 validator; the archive
SHA-256 is
`5615871cf7e38ecf4b85cdd7d11494455c6a6e248461b6be8ffbceae46481295`
and the measured driver SHA-256 is
`045b31d3a68c2cf41f79caa96af9471388b7410496d09c54108c4e31b2aad6fe`.

The raw client event streams, warmups, exact smoke, artifact identity, and
aggregate decision are retained under
[`combined_plus_linear_da7f`](qualification/dsv4/performance/tp4_b1_20260814/combined_plus_linear_da7f).
The aggregate receipt is
[`summary.json`](qualification/dsv4/performance/tp4_b1_20260814/combined_plus_linear_da7f/summary.json)
with SHA-256
`ad6959eddaa7e4a1c7e8dda3e8ed7d5141c3de6cfabfbd96fab8afdf47b5dc3f`.

The immediately preceding 39.0094 tok/s candidate was also profiled with
CUPTI. Its steady profiled decode period was 27.8437 ms: GPU busy-union was
22.0572 ms (79.22%) and exposed idle was 5.7865 ms (20.78%). Summed kernel
durations attribute 38.65% to dense/projection GEMVs, 21.08% to routed
experts, 10.81% to attention/KV, 9.48% to weight read-ahead, and 7.93% to
routing/gate. These category percentages can overlap and are not exclusive
wall time. Collective-shaped transitions account for approximately 4.47 ms
per model step and 73% of captured exposed gaps. The raw no-drop CUPTI log and
profile receipt are retained under
[`profile_da7f`](qualification/dsv4/performance/tp4_b1_20260814/profile_da7f).

| Profile category | Summed ms/model step | Share of summed kernel time |
| --- | ---: | ---: |
| Dense and projection GEMVs | 10.202 | 38.649% |
| Routed-expert compute | 5.564 | 21.077% |
| Attention and KV | 2.854 | 10.811% |
| Weight read-ahead | 2.504 | 9.484% |
| Routing and gate | 2.093 | 7.928% |
| Compressor and Hc | 1.349 | 5.110% |
| Output head | 0.801 | 3.036% |
| Collective local combine and pack | 0.584 | 2.214% |
| Norm and embedding | 0.410 | 1.554% |
| Setup and bookkeeping | 0.036 | 0.135% |

The first post-profile expert fast gate was rejected before production
integration. A routed-only cooperative W13-to-W2 schedule appeared 3.3829%
faster, but that test omitted the concurrent shared-expert branch. At the
required production overlap boundary, the same schedule regressed in all three
independent runs by 0.8871%, 0.8206%, and 0.7861% (0.8311% aggregate). A
non-grid-cooperative expert-cluster alternative regressed by 13.71%. Every
variant was bitwise exact; the rejection is strictly performance-based. No
expert change entered the candidate. The superseding source and raw receipt
are retained under
[`persistent_expert_probe_da7f`](qualification/dsv4/performance/tp4_b1_20260814/persistent_expert_probe_da7f).

The production-shaped dense/projection persistent bundle was also rejected.
It preserved exact FP32 accumulation order and was bitwise exact across all 41
non-SWA layers, nine CTA-grid choices, and seven deterministic inputs, but the
best 64-CTA choice lost 1.0338% in the first 15-round interleaved validation and
0.7840% in an independent replay. Smaller persistent grids were slower. The
accepted three-stream fork already overlaps FP8 q/kv with BF16 main/index work;
reusing the 8 KiB input did not repay the lost CTA parallelism and descriptor
overhead. No bundle change entered the candidate. Raw receipts and source are
retained under
[`projection_bundle_probe_da7f`](qualification/dsv4/performance/tp4_b1_20260814/projection_bundle_probe_da7f).

A predeclared persistent GPU queue for all 129 weight-read-ahead tasks was
also rejected at the production overlap boundary. It removed 128 kernel
launches per model step and remained exact for all 129 consumers, but its best
eight-CTA configuration regressed by 33.1519%, 32.8773%, and 33.1884% in three
independent interleaved runs: 13.4551 ms versus 9.0051 ms control, or 4.4500 ms
added per model step. One to four resident worker CTAs could not keep pace with
the 60 us collective windows; eight or more permanently occupied CTAs stole
capacity from the full-byte consumers. No queue change entered the candidate.
The exact actual-shape source and receipt are retained under
[`weight_readahead_queue_probe_da7f`](qualification/dsv4/performance/tp4_b1_20260814/weight_readahead_queue_probe_da7f).

This is branch evidence, not merged-main qualification. It remains below the
50 tokens/s target; CI and clean merged-main deployment qualification are
deferred until the retained hill-climb changes are finalized.

### Earlier retained milestone: resident decode chain

The unmerged candidate based on `main@a14c2e1caa519f2c337671699afb4f6a185bc09e`
produces **38.1059 decode tokens/s mean** over three TP4 B1 O128 runs, with a
**38.1757 tokens/s best run**. This is an 11.95% gain over the accepted
34.0383 tokens/s baseline and a 2.66% gain over the 37.1188 tokens/s
chain-only measurement. It is approximately 0.85% above the retained 37.7854
tokens/s plain-vLLM TP4 B1 reference measurement, but remains below the 50
tokens/s SparkPipe target. The vLLM comparison used no speculation and is a
throughput reference, not a token-parity claim between serving stacks.

| Run | Decode tok/s | Full-interval time |
| ---: | ---: | ---: |
| 1 | 38.0552 | 3.3373 s |
| 2 | **38.1757** | **3.3267 s** |
| 3 | 38.0869 | 3.3345 s |
| Mean | **38.1059** | 3.3328 s |

This is one request with resident prompt KV, 128 generated tokens, no
speculation, TP4 on `spark4` through `spark7`, and the same end-to-end client
boundary used by the retained B1 measurements. The mean full decode interval
is 26.2427 ms/token. The driver executes up to eight decode steps per resident
submission and emits lane-major token bursts; the scheduler caps each chain at
the output budget, context limit, and current KV-page boundary.

The combined candidate also retains the independently measured W2 N128/four-CTA
schedule, immutable shared direct-send buffer, and cache-streaming FP8 weight
loads. It does not change the BF16 spine, model weights, sampling, or KV
precision. All three runs emitted the exact retained 128-token sequence; the
canonical comma-separated sequence plus trailing newline hashes to
`211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`.

| Receipt | SHA-256 |
| --- | --- |
| [`resident-chain-combined-a14c2e1-run1.json`](qualification/dsv4/performance/tp4_b1_20260814/resident-chain-combined-a14c2e1-run1.json) | `687c9d25fcf102717ac45026340ab2053a93c41307feffe60be66c8fda91f02a` |
| [`resident-chain-combined-a14c2e1-run2.json`](qualification/dsv4/performance/tp4_b1_20260814/resident-chain-combined-a14c2e1-run2.json) | `6942d6844bdb798a28a6d5216d1aeb6d6df8dfcaf4c827cf001f7a5289c5e6ee` |
| [`resident-chain-combined-a14c2e1-run3.json`](qualification/dsv4/performance/tp4_b1_20260814/resident-chain-combined-a14c2e1-run3.json) | `13031f292a7bdc49388d082d19c753665a8d6fa7f8dcc8b47b0071940ef67a42` |
| [`vllm-dsv4-b12x-tp4-b1.json`](qualification/dsv4/performance/tp4_b1_20260814/vllm-dsv4-b12x-tp4-b1.json) | `9144842eff3c103d784df30bad3ab467e8c0633347d5704bc455546785617db5` |

This is branch evidence, not merged-main qualification. A clean merged-main
rebuild and three-run exact-output requalification remain required after merge.

### Latest merged-main qualification: DeepSeek V4 Flash TP4 B1

Merged `main@ed1d731dd72caa66ef9545464df6573d81dbbbb8` produces **33.55
decode tokens/s mean** over three runs, with a **33.65 tokens/s best run**.
This is one request, not eight or sixteen concurrent requests and not aggregate
batch throughput. It removes the regressed full-graph bridge and restores the
single control-plane execution path.

| Field | Measured configuration |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Checkpoint revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Kernel target | `cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16` |
| Topology | TP4 on `spark4` through `spark7`; no PP and no speculation |
| Workload | B1: one 128-token prompt, one request, 128 output tokens |
| KV state during timed region | Prompt prefill complete and its KV resident |
| Timed boundary | First emitted token to last emitted token: 127 full decode intervals |
| Included | Scheduler, model kernels, TP collectives, sampling, runtime IPC, and streamed output observation over SSH |
| Excluded | Initial process connection and prompt prefill/TTFT |
| Runtime limits | one in-flight submission, one active sequence, one input row, one resident sequence |
| Precision route | Exact target above; no speculative draft model |
| Continuation proof | 254 of 256 submissions used the fenced continuation lease; zero rejects and zero live leases after release |

All four ranks independently cloned, detached, and rebuilt the exact clean
merged-main commit with CUDA 13 for `sm_121a`. The exact Blackwell CI compile
also passed before merge.

| Run | Decode tok/s | Median inter-token | p95 inter-token |
| ---: | ---: | ---: | ---: |
| 1 | 33.5334 | 29.6790 ms | 30.8040 ms |
| 2 | **33.6468** | **29.5453 ms** | **30.6978 ms** |
| 3 | 33.4714 | 29.7585 ms | 31.5515 ms |
| Mean | **33.5505** | 29.6609 ms | - |

The retained pre-regression floor at `main@fc897a0` averaged 33.6647 tok/s on
the same workload. The restored current-main result is 0.34% lower, inside the
observed run-to-run range, and 20.74% faster than the regressed 27.7883 tok/s
bridge. All three runs emitted the same 128 tokens as the retained floor. The
comma-separated token-ID sequence plus a trailing newline hashes to
`211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`
with SHA-256.

The raw event streams and artifact identities are retained at:

| Receipt | SHA-256 |
| --- | --- |
| [`merged-main-ed1d731d-run1.json`](qualification/dsv4/performance/tp4_b1_20260814/merged-main-ed1d731d-run1.json) | `d5d6cc02cbdd470206a9f2baf453206e67d0be24de487bc430689285915057a4` |
| [`merged-main-ed1d731d-run2.json`](qualification/dsv4/performance/tp4_b1_20260814/merged-main-ed1d731d-run2.json) | `f013f26bb0f63f86d7c134051d47802c85fa3b63982ec61181a0ccb24d2e0d5b` |
| [`merged-main-ed1d731d-run3.json`](qualification/dsv4/performance/tp4_b1_20260814/merged-main-ed1d731d-run3.json) | `2caa49f9d0f152f0a8e0f64ab4dc3cfe4b88d89ea357c9d39199739bd29c0f1e` |
| [`merged-main-ed1d731d-summary.json`](qualification/dsv4/performance/tp4_b1_20260814/merged-main-ed1d731d-summary.json) | `fc875e4738960a389c00f43b50861630bbc5cbaee486d7425629b082035e7cc1` |

This remains below the 50 tok/s target. The 33.6647 tok/s pre-regression result
remains the performance floor for accepting the next optimization; current
merged main has independently restored it within 0.34% with exact token parity.

The package CUDA validator did not qualify this runtime package: it currently
hard-codes TP1 and a 64-graph maximum while this package is TP4 with 130 graphs.
Wrapper generation therefore used the explicitly named
`packaging-only.exact-live-inference-required.v1` recipe. Exact live inference
passed, but the static validator gap remains open and is not represented as a
validation success.

### Earlier retained milestone: device-predicated compressor emission

Candidate `de1beb2035ecbf40dd49f0fa822c266007baea9d`, based on merged
`main@ed1d731d`, produces **33.9911 decode tokens/s mean** over three TP4 B1
O128 runs, with a **34.1190 tokens/s best run**. This is 0.97% above the
retained 33.6647 tok/s floor and 1.31% above the current-main qualification.
It is a branch result until merged-main requalification.

| Run | Decode tok/s | Median inter-token | p95 inter-token |
| ---: | ---: | ---: | ---: |
| 1 | 33.8394 | 29.4739 ms | 30.3798 ms |
| 2 | 34.0149 | 29.1768 ms | **30.2899 ms** |
| 3 | **34.1190** | **29.0832 ms** | 30.4197 ms |
| Mean | **33.9911** | 29.2446 ms | - |

The change replaces the post-compressor RMSNorm, RoPE, optional Hadamard,
quantization, and cache-scatter launches with one device-predicated kernel.
Off-boundary CTAs return before touching weights, staging data, or cache. The
production post-compressor schedule falls from 269 to 62 launches per token;
there is no host token predicate and no legacy compatibility path.

An isolated `sm_121a` hardware probe compared the fused and standalone paths
for SWA, CSA attention, CSA index/Hadamard, and HCA. Emitted BF16 bytes and the
complete cache matched bit-for-bit in all four cases. All three end-to-end runs
then emitted the exact retained 128-token stream.

| Receipt | SHA-256 |
| --- | --- |
| [`device-predicated-compressor-de1beb2-run1.json`](qualification/dsv4/performance/tp4_b1_20260814/device-predicated-compressor-de1beb2-run1.json) | `3022be3101cbb2da889f015edb49c32cf6346c757349172b5a5e4e1b8e75122a` |
| [`device-predicated-compressor-de1beb2-run2.json`](qualification/dsv4/performance/tp4_b1_20260814/device-predicated-compressor-de1beb2-run2.json) | `668a5d6ae09d2541151b0cdd0cc98c8c9f711d24f20b4b4168659cb51ac740f8` |
| [`device-predicated-compressor-de1beb2-run3.json`](qualification/dsv4/performance/tp4_b1_20260814/device-predicated-compressor-de1beb2-run3.json) | `d1e7ee2409b41bd188c08945d279a50d45f5399a361572e15d38f51a26e8ffd0` |
| [`device-predicated-compressor-bitwise-sm121.json`](qualification/dsv4/performance/tp4_b1_20260814/device-predicated-compressor-bitwise-sm121.json) | `6537d968880ffd2effd4f971232e88701ac0ff1ea8f1c8c79f73c82c2eba4b77` |
| [`device-predicated-compressor-de1beb2-summary.json`](qualification/dsv4/performance/tp4_b1_20260814/device-predicated-compressor-de1beb2-summary.json) | `4605248320b7e6cbed55d478d207e5cd087530a696680beab436252d49145c38` |

### Previous scratch milestone: DeepSeek V4 Flash TP4 B1

The previous retained measurement was **32.57 decode tokens/s mean** over four
runs, with a **32.81 tokens/s best run**. It was one request, not aggregate
batch throughput.

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

The prompt is prefetched by the same request. The benchmark starts its decode
clock at the first emitted token, after prefill has populated KV, and observes
every subsequent token at the client. This is end-to-end cached-KV decode, not
a kernel-only timer or a prefill-plus-decode blended rate.

#### Four-run result

| Run | Decode tok/s | Median inter-token | p95 inter-token |
| ---: | ---: | ---: | ---: |
| 1 | 32.4299 | 30.6386 ms | 31.6988 ms |
| 2 | 32.6948 | 30.3928 ms | 31.4345 ms |
| 3 | **32.8075** | **30.2758 ms** | **31.1335 ms** |
| 4 | 32.3374 | 30.7020 ms | 32.2516 ms |
| Mean | **32.5674** | 30.5023 ms | - |

The four rates span 32.3374--32.8075 tok/s. Their median is 32.5623 tok/s. All
four candidates emitted the same 128-token sequence, and that sequence is
byte-for-byte identical to the retained control output. The comma-separated
token-ID sequence hashes to
`211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083`
with SHA-256.

#### Identity and retained evidence

This milestone was measured from a scratch candidate based on
`main@7bf94d8bf087d5c9584e2627d03ea8408ce13c22`; it is a measured engineering
milestone, not yet a merged-main production qualification. The candidate
fused the first recursive-doubling reduction with construction of the next
relay payload. Its driver SHA-256 is
`212cee3f901e513936cea20b305305a6ea18df28d577b5adb6f27a25924c0a8e`.

The raw receipts, full event streams, exact token IDs, and input batch are
retained in Git. Their repository paths and SHA-256 digests are:

| Receipt | SHA-256 |
| --- | --- |
| [`qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run1.json`](qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run1.json) | `ba792c90f5f484bb49f6a92e95ef807d1e9efcd30d0dad06fb96b76481be2321` |
| [`qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run2.json`](qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run2.json) | `5e7f8307bd29d42cea0aead9fd09d8edd63c02e459ce4e49a89cb75bbe9d32fc` |
| [`qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run3.json`](qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run3.json) | `ff130bc6d751543b073e340772132457eff501b160588883fb198e18f19feaa8` |
| [`qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run4.json`](qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-run4.json) | `6e0e4ea9afdbd4c40eebea1c6d235062761c1811241b7fce28a69a771d2767ec` |
| [`qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-control2.json`](qualification/dsv4/performance/tp4_b1_20260813/dsv4-combine-relay-control2.json) | `ff9530527512f6ef1d128ae3e51dc17a1a481e430666742701482258be2d91b7` |
| [`qualification/dsv4/performance/tp4_b1_20260813/dsv4-tp4-pp4-b1-compsec076-o128.json`](qualification/dsv4/performance/tp4_b1_20260813/dsv4-tp4-pp4-b1-compsec076-o128.json) | `e498f1fc88854044eafa64c41ce308b73d54f0a351fe156a513d7ff7ca630ead` |

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
record exact source, driver, model, topology, and workload identities, and
prove emitted-token parity with the accepted control. Profiled, simulated,
projected, kernel-only, and multi-request aggregate figures remain separate.

### Removed regression and current hill-climb boundary

Merged `main@07696e074d57e194e756b7f39f2ea7cbf6ca4413` replaced the
event-driven graph-island controller with a nominal full graph. After fixing
its recursive-U64 phase ownership bug, two exact B1 O128 runs measured 27.8426
and 27.7339 tok/s, or 27.7883 tok/s mean. The emitted token sequence is exactly
the accepted 128-token control. This is a 17.46% regression from the immediate
33.6647 tok/s predecessor.

There are 130 collectives per token. The full graph retained all 130 legacy
host submissions and completion callbacks, then added a mapped producer write,
host poll, mapped completion write, and graph wait at every boundary. The
6.2817 ms/token regression is 48.32 us per collective, matching the added
coherent rendezvous. The change did not alter the RDMA data plane; it doubled
the control plane around it.

| Receipt | Decode tok/s | SHA-256 |
| --- | ---: | --- |
| [`full-graph-phase-owned-run1.json`](qualification/dsv4/performance/tp4_b1_20260814/full-graph-phase-owned-run1.json) | 27.8426 | `a9e267ce13fcd1962d38c2105884f159b3ef9a7764435c128b4e79f6476cc098` |
| [`full-graph-phase-owned-run2.json`](qualification/dsv4/performance/tp4_b1_20260814/full-graph-phase-owned-run2.json) | 27.7339 | `f68236d2ae961ee55f142ae3188327240b71f2c8975c2a5ae162647578a3365e` |

The full-graph bridge was removed in merged `main@ed1d731d` rather than retained
behind a runtime switch. The clean merged-main rebuild and three-run
qualification above restored 33.5505 tok/s, 20.74% above the regressed bridge
and 0.34% below the 33.6647 tok/s floor. The working graph-island controller is
again the sole TP execution path until a predeclared collective program
replaces its 130 submissions and callbacks instead of wrapping them.

## Planning projections

The following values guide architecture choices and are not measurements.

If optimized TP4 reaches 38 tokens/s, with approximately 24.8 ms of remaining
local work and 85% of that work scaling with TP width, the planning points are:

| Layout | Projected step | Projected speed | Credible planning range |
| --- | ---: | ---: | ---: |
| TP8 | 16.56 ms | 60 tok/s | 55-65 tok/s |
| TP16 | 11.99 ms | 83 tok/s | 70-95 tok/s |

For large MoE residency, TP4 x PP4 remains the canonical operational choice:
B4 fills the pipeline, shared-prefix B8 can combine pipeline occupancy with
DSpark revisit slack, and high concurrency can vary stage-local microbatch
width without moving weights or KV. TP16 is a B1-latency or smaller-dense-model
layout, not a reason to reload the primary large model.

## Target gates

These are architecture requirements, not measured results:

| Gate | Target |
| --- | ---: |
| Promote a configured nonresident model to ready | at most 60 s |
| Read model shards from the external pooled tier | at least 20 Gb/s useful |
| Internal hot KV allocation | 2.5 TB per Spark |
| Internal active model-shard allocation | 1.0 TB per Spark |
| External direct model tier | at least 1.0 TB per Spark |
| Four- or eight-Station largest-model throughput | roughly 50% of matched DGX B300 workload |

Promotion timing includes shard access, verification, rank-local placement,
driver and communicator binding, prewarm, all-rank agreement, and atomic ready
publication. A copy-only storage benchmark does not close the model-promotion
gate.

The Station comparison requires the same checkpoint, precision, request shape,
context, output length, batching policy, and timing boundary on both systems.
No analytical memory-bandwidth ratio closes that gate.

### GLM 5.2 TP8 B1 single-stream + B8 multi-sequence (2026-08-16)

Merged-main accuracy fixes (#663 TP8 support, #665 head-offset + gate-up order)
deployed on the TP8 fanout band spark8-sparkf (8 ranks, BF16 spine + FP8
experts, recursive-doubling collective). Layer 0 and layer 3 outputs were
verified against a CPU torch reference of the BF16 master (0.2-0.9% RMS
agreement; layer 3 includes the router + correction-bias top-8 + FP8 expert
path). All runs emit the identical token stream.

| Field | Measured configuration |
| --- | --- |
| Model | zai-org/GLM-5.2 |
| Checkpoint revision | b4734de4facf877f85769a911abafc5283eab3d9 |
| Kernel target | cuda.sm121.glm52.resident_decode_stage.bf16.expert_fp8 |
| Topology | TP8 fanout on spark8 through sparkf; no PP, no speculation |
| Workload A (single-stream) | B1/B8: one 6-token prompt, one request, 128 output tokens |
| Workload B (serving aggregate) | B8: 8 concurrent sequences x 32 output tokens (256 total) |
| Timed boundary | Process start to terminal (includes prefill + connect) |
| Included | Scheduler, model kernels, TP collectives, runtime IPC |
| Precision route | Exact target above; no speculative draft model |

| Workload | Tokens | Wall time | Throughput |
| ---: | ---: | ---: | ---: |
| A: single-stream decode (B1, clean fleet) | 128 | 18.52 s | 6.91 tok/s |
| A: single-stream decode (B8 class, batched prefill) | 128 | 18.73 s | 6.83 tok/s |
| B: 8-sequence aggregate (B8 class) | 256 | 5.89 s | 43.46 tok/s |

The single-stream path is bandwidth-bound: per token it reads ~15.7 GB of FP8
expert weights plus ~6 GB of BF16 spine at the 273 GB/s LPDDR5x limit (~80 ms
hard floor), plus ~31 ms of TP collectives (158 reduces/token, median 912 us)
and ~25 ms of kernel launch + compute overhead. The B8 class amortizes the same
weight bytes across eight rows, which is what lifts the aggregate to 43.46
tok/s. Next: reduce-path and launch-overhead squeezes toward the single-stream
ceiling, and dspark speculative decode (draft weights must be trained first;
the base checkpoint ships none).

| Receipt | SHA-256 |
| --- | --- |
| [glm52-b128d.stderr](qualification/glm52/performance/tp8_b8_20260816/glm52-b128d.stderr) | 4519f1b8a5f4639916e81d374c4fdf7c92cd500fc0ffa8569636df0ad278fe06 |
| [glm52-toks.txt](qualification/glm52/performance/tp8_b8_20260816/glm52-toks.txt) | bc9edad1edcb6fcebf2aadb42c2dd1b85549bfdfc7ee8a0d6d2dca2a1d32b7c8 |
| [glm52-b8.stdout](qualification/glm52/performance/tp8_b8_20260816/glm52-b8.stdout) | 29bea03aad86124aae5e0521afdb34262b173f6e0fb8595f630f410b81a525f7 |
| [glm52-multi.stdout](qualification/glm52/performance/tp8_b8_20260816/glm52-multi.stdout) | 5740de2f83cdb93237fc5c12a4a96b6393e856c39793f682e8a2c2318846c0d9 |

### GLM 5.2 TP8 B16 16-sequence aggregate (2026-08-16)

The B16 batch class (16 rows per wave, resident_sequence_capacity 16, 8 GiB KV
backing) lifts the serving aggregate further: the same weight bytes amortize
across sixteen rows. Sequence 1 emits the identical retained token stream.

| Workload | Tokens | Wall time | Throughput |
| ---: | ---: | ---: | ---: |
| 16-sequence aggregate, run 1 | 512 | 6.74 s | 75.96 tok/s |
| 16-sequence aggregate, run 2 | 512 | 6.77 s | 75.63 tok/s |
| 16-sequence aggregate, run 3 | 512 | 6.82 s | 75.07 tok/s |
| Mean | 512 | 6.78 s | 75.55 tok/s |

| Receipt | SHA-256 |
| --- | --- |
| [glm52-b16.stdout](qualification/glm52/performance/tp8_b16_20260816/glm52-b16.stdout) | 0f74a5775de854ccadf57ac2b46b3e878e88aeecdef93d83298096f704652b44 |
| [glm52-b16.stderr](qualification/glm52/performance/tp8_b16_20260816/glm52-b16.stderr) | 2d606c22af465762952d0f9ba51c82a14c8e8885a7a30df4985cca403c260b21 |
| [glm52-b16-r1.stderr](qualification/glm52/performance/tp8_b16_20260816/glm52-b16-r1.stderr) | 9483ab63383a267903c771eb10caa182af3ff847543a7d080f198e512d51436a |
| [glm52-b16-r2.stderr](qualification/glm52/performance/tp8_b16_20260816/glm52-b16-r2.stderr) | cdf2d247e59adce83d8102741786860be4a0b5f87ba086ecd11d74f361b873a5 |
