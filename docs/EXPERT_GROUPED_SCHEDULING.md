# Expert-Grouped Continuous Batching

## The idea

Stop thinking about batch sizes. Stop thinking about B1, B8, B64 and B1024
as fixed operating points.

**Process every request that is ready, in one expert-grouped pass.** The
batch size is however many requests are pending. Each expert's weights are
read once per step for all of the rows routed to it.

Weight traffic does not stay flat as B grows. It grows with the number of
**distinct** experts the step touches and saturates at E:

- At B1 only top_k experts per layer are read: 8 of 288 for GLM 5.3 Flash,
  16 of 896 for K3. That is why B1 is fast, and the TP16 B1 target depends
  on it.
- Once most experts are touched, traffic is roughly constant and extra rows
  cost compute only. README §Batching is the short, authoritative form of
  this rule.

## Why this works: the weight-amortization crossover

For a MoE layer with E experts, top-k routing and B pending rows:

| | Memory (weight stream) | Compute (expert math) |
|---|---|---|
| Traffic | D(B) × bytes_per_expert, with D(B) ≤ E | B × top_k × FLOPs_per_expert (linear in B) |
| Time | D(B) × bytes / BW | B × top_k × FLOPs / compute_rate |

D(B) is the number of distinct experts the step touches. Assuming uniform,
independent routing (arithmetic, not a measurement):

```
D(B) = E × (1 − (1 − top_k/E)^B)
```

For GLM 5.3 Flash (E=288, top-8) this gives D(1)=8, D(8)≈58 and
D(64)≈241. That matches README's "B64 already touches 240 experts".

**Small B: memory dominates.** Loading the touched experts costs more than
computing with them. Adding a row adds only the experts no other row had
already touched.

**Large B: compute dominates.** Once D(B) ≈ E, the weights are loaded once
and the math is the cost. The crossover B* is where D(B) × bytes / BW
equals B × top_k × FLOPs / rate.

**The optimal operating point is B\*.** There, weight amortization is full
and queuing latency adds little beyond the compute itself.

## Concrete numbers (K3: 896 experts, top-16, MXFP4, 93 layers)

All values below are arithmetic from the model definition. None is
measured.

**Model constants.**

- **Expert shape.** An expert maps the 3584-dim routed latent through a
  3072-dim intermediate:
  - `SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION` = 3584 and
    `SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION` = 3072
    (`model-families/k3/include/sparkpipe/spark_k3_llm_defines.h`);
  - the same values appear as `latent_dimension` and
    `expert_intermediate_dimension` in `model_contracts/k3_authoritative.json`.
- **Bytes per expert: 17,547,264**, about 17.5 MB
  (`full_model_expert_bytes` in `model-families/k3/smoke_experts.json`).
  That is 3 × 3584 × 3072 weights at 4.25 bits each: 4-bit E2M1, plus one
  E8M0 scale per 32 weights.
- **All experts of one layer:** 896 × 17.5 MB ≈ 15.7 GB.
- **Layers.** Layers 1-92 are routed (`SPARK_LLM_FIRST_ROUTED_LAYER` 1 of
  93), so all routed experts together are ≈ 1.45 TB.
- **Fleet bandwidth:** 16 × 273 GB/s ≈ 4.4 TB/s (`README.md`).

**Weight traffic per layer by batch size.** Grouped traffic assumes the
D(B) formula above.

| B | Distinct experts D(B) | Grouped (GB/layer) | Token-major (GB/layer) | Token-major ÷ grouped |
|---:|---:|---:|---:|---:|
| 1 | 16 | 0.28 | 0.28 | 1.0 |
| 8 | 120 | 2.1 | 2.2 | 1.06 |
| 64 | 613 | 10.8 | 18.0 | 1.7 |
| 256 | 887 | 15.6 | 71.9 | 4.6 |
| 1024 | 896 | 15.7 | 287.5 | 18.3 |

- **B1.** 16 experts × 17.5 MB × 92 layers ≈ 25.8 GB of expert weight per
  token.
- **Saturation.** Once D(B) ≈ E, a step streams all ≈ 1.45 TB of routed
  experts. At the full fleet bandwidth that takes about 0.33 s. This is an
  ideal: every Spark at 273 GB/s, with the experts evenly sharded.
- **B*_compute** must be measured on the fleet. Profile the MoE layer at
  increasing B and find where per-token cost stops dropping.

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

**The scoreboard changes.** Instead of "tok/s at B=8", report aggregate
tok/s at queue depth Q, and the maximum sustainable throughput at the
compute-bound knee. Report per-stream tok/s next to it. Per-stream speed
falls as B grows, because the bytes read per step grow with the number of
experts touched (see the GLM 5.3 Flash TP16 projection table in
`PERFORMANCE_STATUS.md`).

## What needs to change in the code

`model_batch_engine` is already a submission-wave continuous batcher. It
admits the whole ready set and interleaves chunked prefill with decode
(README §Batching). The GLM Flash scheduler already selects any count up to
its configured limit.

The remaining work is listed in TECHDEBT §Dynamic batching: qualifying
larger capacities, and bitwise B1 equality.

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
