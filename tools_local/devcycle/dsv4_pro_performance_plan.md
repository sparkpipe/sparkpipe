# DSV4 Pro TP4xPP4 performance improvement plan

Prioritized, with the expected win, the verification path, and the risk.
Baseline expectations from the path audit (tools/devcycle/
dsv4_pro_inference_path_audit.md): ~25-60 tok/s decode, dominated by expert
weight DRAM streaming (~8 GB/token of top-6 expert weights), then TP
collectives (~122/token), control plane, boundaries.

## Ring-day measurement prerequisites (do these first)

1. --profile-stages is already wired into first_decode_pro.sh: the receipt
   gets per-stage queue/service/elapsed ns for every submission.
2. Enable tp_collective profiling (the progress thread prints per-operation
   phase/timing every 250 ms when enabled) so the 122 collectives/token can
   be attributed. Check the exact env/flag before ring day.
3. Record the expert weight read volume per token (can be derived from the
   per-stage service time x expected 132 MB/layer top-6 reads).

## P1 - Prefill batching (DONE, pre-ring)

max_prefill_rows_per_submission 1 -> 128 (dsv4pro_o128_128row_batch.json).
Cuts prefill collectives 15,616 -> 122 and amortizes expert weight reads
~128x across prefill rows. TTFT expectation: several seconds -> well under
a second of collective overhead.

## P2 - Decode chains (biggest decode-side lever, needs design work)

The module already supports multi-step resident chains
(chain_step_count > 1 in SparkDsv4ModuleValidateResidentChain); the batch
engine issues one submission per token today. Chaining 2 tokens per
submission would halve the per-token control-plane traffic (32 TCP messages
-> 16/token) and let layer collectives cover 2 rows (2x payload, same
latency). The MTP layer in the pack (currently load-only, never executed)
is the natural 2-token generator; executing MTP requires the MTP kernels
(none exist in the .cu today) plus a verification policy for the second
token. Two paths:
  a. speculative 2-token chain without MTP (cheap, ~1.5-1.8x decode);
  b. real MTP execution (2x, but a kernel project).
Expected decode gain: 30-60 -> 45-110 tok/s. Risk: medium-high; needs a
token-stream gate comparison on the ring.

## P3 - GPUDIRECT transport variant (likely neutral on GB10)

The build already produces libhidden_transport_spark_gpudirect_rdma_verbs.so;
the mapped-host mode currently deployed is zero-copy on GB10's unified
memory, so the gain is expected to be small (removes the cudaHostAlloc
mapping indirection). Measure both on the ring; keep the faster. Low risk.

## P4 - Collective threshold tuning

The direct-all-to-all cutoff (81920 B) sends 3 point-to-point hops for the
14 KB decode payloads; recursive doubling uses 2 rounds. The split-ring
minimum (655360 B) only engages for batched rows. After P1's profiling
data: sweep the thresholds, measure per-collective latency at each size.
Low risk, config-only (stage JSON).

## P5 - Control-plane pacing

maximum_messages_per_rank_per_progress 8 and maximum_new_submissions_per
progress 1 serialize dispatch ~1 tick per token. Raising both (e.g. 16/4)
cuts the per-token dispatch tail. Config-only; measure on the ring. Low
risk.

## P6 - Progress-thread core pinning

The collective progress thread busy-spins one core per rank. Pin it off the
kernel-dispatch cores (taskset or the runtime's affinity hook) to keep
decode kernel launch latency stable. Ops-level change, no code. Low risk.

## P7 - Prefill wave batching (module change)

Prefill attention currently waves 1 row at a time for single-lane prompts
(round-major wave logic). The sparse-attn partial budget (48 SMs, 8
partials/row at TP4) allows 6 rows per launch: extend the wave grouping for
same-lane rows up to the capacity-derived bound under PRO_BUILD. Cuts
prefill attention launches 128 -> 22 per layer. Medium effort, medium risk;
validated via the prefill path on the ring (no single-spark path exists for
the prefill wave loop).

## P8 - Weight read-ahead generalization

The read-ahead mechanism currently overlaps only WQ_B with the attn-side
reduce. Extend it to prefetch the NEXT layer's expert tensors once the
router output is known (the gate runs before the FFN; the next layer's
experts are known at the end of the current layer's FFN). Hides DRAM
latency behind the FFN-side collective. Medium effort; needs the ring to
measure.

## P9 - Expert codec variants (accuracy/perf tradeoff)

With the expert-requant tooling in place (tools/dsv4_pro_expert_requant.py),
FP8-E4M3 expert packs become possible once the FP8 expert kernel variant
lands. FP8 experts double the expert DRAM traffic (1 B vs 0.5 B/element) so
they are an accuracy knob, NOT a speedup; MXFP4 stays the performance
codec. The kernel variant is the remaining work (see the selectability
doc).

## Order of operations on the ring

1. Baseline O128 run with --profile-stages + collective profiling.
2. P4/P5 config sweeps (cheap, same window).
3. P2a (2-token chains) behind the token-hash gate.
4. P6/P7/P8 as code lands.
