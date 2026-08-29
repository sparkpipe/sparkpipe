# Expert-Grouped Continuous Batching

## The idea

Stop thinking about batch sizes. Stop thinking about B1, B8, B64, B1024
as fixed operating points. Instead:

**Process every request that is ready to be processed, in one
expert-grouped pass.** The batch size is however many requests are
pending — from 1 to ∞ — and the MoE weight cost is the same regardless.

## Why this works: the weight-amortization crossover

For a MoE layer with E experts, top-k routing, and B pending requests:

| | Memory (weight stream) | Compute (expert math) |
|---|---|---|
| Traffic | E × bytes_per_expert (constant!) | B × top_k × FLOPs (linear in B) |
| Time | E × bytes / HBM_BW | B × top_k × FLOPs / compute_rate |

At small B, **memory dominates**: loading the expert weights costs more
than computing with them. At large B, **compute dominates**: the weights
are loaded once and the math is the cost.

The crossover B* where memory time = compute time:

```
E × bytes / BW = B × top_k × FLOPs / rate
B* = (E × bytes × rate) / (top_k × FLOPs × BW)
```

Below B*: adding tokens is nearly free (weight amortization).
Above B*: adding tokens costs compute (but that compute is useful work).

**The optimal operating point is B\* — full weight amortization, minimal
queuing latency beyond the compute itself.**

## Concrete numbers (K3: 896 experts, top-16, MXFP4, 93 layers)

- Per-expert weights: 25.7 MB (MXFP4)
- HBM bandwidth: 250 GB/s
- Per-layer weight stream: 896 × 25.7 MB = 23 GB → 92 ms

At B=1024 (1024 pending decode requests):
- Each expert activated ~18.3 times (1024 × 16 / 896)
- Weight traffic: **23 GB** (same as B=1!)
- Naive token-major would be 420 TB (18.3× re-read × 896 experts)
- **The grouped approach is 18× less traffic at B=1024, and the ratio
  grows with B**

At B=100,000:
- Each expert activated ~1,786 times
- Weight traffic: **still 23 GB**
- Hidden-state scratch: 100K × 7168 × 2 = 1.4 GB
- Compute: 100K × 16 × 2 × 7168 × 3584 FLOPs per layer = 8.2 TFLOP
  → at ~50 TFLOPS effective: 165 ms
- MoE HBM+compute: 92 + 165 = 257 ms per layer
- Per-token: 257/100000 = 2.6 µs per layer = 240 µs for 93 layers
- Aggregate: **~4,200 tok/s decode** (compute-bound, not memory-bound)

## The scheduling rule

```
ready_set = { requests with KV-prefill complete }
B = |ready_set|
if B == 0: wait for next prefill completion
if B > B*_compute: cap at B*_compute (oldest first — latency bound)
route = router(ready_set[0:B])
sorted = counting_sort(route)         // expert-major
for each expert E in sorted order:
    load E's weights
    process all tokens queued for E
scatter outputs back via route_source_token
advance every request one token
```

The "infinite" batch is capped at B*_compute because beyond that point:
- Weight amortization is already 100%
- Adding tokens only increases latency for the ones already queued
- The per-token marginal cost is pure compute

B*_compute is measurable: profile the MoE layer at increasing B and
find where per-token cost stops dropping. That's the knee.

## Why the B1/B8/B64/B1024 ladder becomes obsolete

The old model: pick a batch size, prefill that many, decode that many.
The throughput at each B was an operating point you tuned for.

The new model: the batch is **whatever the queue holds**. Throughput
scales with queue depth until the compute-bound knee, then plateaus.
There are no operating points — there is one continuous function from
queue depth to aggregate throughput, and the scheduler always operates
on the full ready set (or the compute-bound cap of it).

**The scoreboard changes**: instead of "tok/s at B=8", the metric is
"aggregate tok/s at queue depth Q" — or better, "tok/s at the
compute-bound knee" (the maximum sustainable throughput).

## What needs to change in the code

| Component | Current | Needed |
|---|---|---|
| Batch engine admission | KV-resident per microbatch | Admit ALL with KV resident |
| Decode batch assembly | row_count = current lanes | row_count = all pending |
| Route build input | rows = microbatch rows | rows = all pending rows |
| Grouped GEMM | Already handles any rows | Nothing |
| Output scatter | route_source_token already does it | Nothing |
| Hidden-state buffer | Per-microbatch | Persistent buffer sized to MAX_PENDING |
| MAX_ACTIVE_SEQUENCE_COUNT | 64–512 | Raise, or chunk into 512-slices with pipelined weight reuse |
| Scheduler | Fixed batch buckets | Ready-set admission + compute-bound cap |

The counting sort kernel needs zero changes. The grouped GEMM needs zero
changes. The route_source_token scatter needs zero changes. **All the
work is in the batch engine's admission and assembly logic.**

## Cross-layer dataflow (second-order optimization)

After expert E processes its queue at layer L, the outputs scatter back
to original positions. Layer L+1 re-routes and re-sorts. The cross-layer
optimization: while layer L computes, pre-run the router for layer L+1
on the scattered outputs and pre-sort, so layer L+1's grouped GEMM
starts immediately. This eliminates the sort latency between layers but
requires the router to run as a anticipatory kernel. Defer until the
base grouping is proven on all MoE models.

## Non-MoE layers

Dense FFN (27B), GDN, and attention layers scale per-token regardless.
They don't benefit from grouping (every token hits the same weights).
For hybrid models (Qwen 27B: 48 GDN + 16 attention, no MoE), this
innovation doesn't apply — the batch size still matters for dense
weight amortization. **This innovation is specifically for MoE models.**

## Filing

This is a SparkPipe design innovation: continuous expert-grouped
batching with compute-bound admission. The router + counting sort +
grouped GEMM infrastructure already exists (inference/kernels/route.cuh);
the innovation is the scheduling policy that feeds it the full ready set
instead of fixed-size microbatches.

## Addendum: B<∞ × PP bubbles; speculation interleave (2026-08-30)

PP BUBBLES: continuous batching AMORTIZES them at high B (bubble is
constant per step, compute grows with rows) but does NOT remove the
per-step fill/drain serialization. B=1 single-stream pays the full
chain (TP16 stays the latency topology; PP wins on capacity at load).
The remaining lever = STEP-LEVEL PIPELINING (double-buffered
activations, stage N+1 starts while stage N drains) — P1's async
completion is its prerequisite. Honest state: not built.

SPEC INTERLEAVE: a speculating row is a row with more positions per
step — heterogeneous steps (verify rows k+1 positions, plain rows 1,
prefill chunks their slices) all share one weight stream. Pattern per
tick: draft mini-step (tournament: N drafters race, tree built) →
verify step → accept/reject (KV only for accepted; re-draft from the
new prefix). Admission unchanged (new requests enter at step
boundaries regardless). Fairness: spec rows lengthen steps slightly;
low-acceptance rows get disarmed (the bandit's job). OPEN CORNER:
spec-under-PP — draft verify crosses the same serial stages; drafter
placement (first-stage vs head-owner node) unmeasured. Flag for the
TP4xPP4 spec cells.
