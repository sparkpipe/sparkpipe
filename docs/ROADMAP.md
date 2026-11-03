# Roadmap

The goal is the system that [`README.md`](../README.md) describes, with every
section of [`TECHDEBT.md`](../TECHDEBT.md) closed and each claim backed by
receipts from a clean merged `main`. This file puts that work in order. Each
milestone has an exit criterion that a measurement decides. The measurements
themselves belong in [`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md).

Work goes one iteration at a time. An iteration changes one thing, says which
exit criterion it moves, and names the measurement that picks the next step.

The operator's current emphasis (2026-09-28), with the operator directives in
[`GOALS.md`](GOALS.md):

- drivers developed in parallel while common code improves (M11);
- GLM 5.3 Flash B1 without speculation at TP16, measured against public
  4-Spark results (M1);
- speculation that pays at B8 and B64 as well as B1 (M10);
- no fixed topology: TP16, PP16 and TP4 x PP4 compared on measured tok/s and
  latency (M5).

The state of each milestone and driver on 2026-09-28 is in
[`HANDOFF_DRIVERS_2026-09-28.md`](HANDOFF_DRIVERS_2026-09-28.md).

## M1. GLM 5.3 Flash at 80% of the memory roofline

Exit, on the sixteen-Spark fleet at TP16 and 1K context:

- a B8 decode step of at most 26 ms, which is 80% of the 21 ms floor (about
  300 tok/s across 8 streams);
- B1 at most 10 ms per token, which is 80% of the 8.0 ms floor;
- greedy tokens unchanged.

The operator's B1 target without speculation is 70-81 tok/s. That is 3 to 3.5
times the best public 4-Spark result, 23.2 tok/s (arithmetic; GOALS.md). The
10 ms exit (100 tok/s) sits above it.

Status on 2026-09-28 (PERFORMANCE_STATUS.md):

- B1 measured 36 tok/s end to end. Inside 8-step graph chains the step is
  23-24.5 ms/token. That run used `dd3526b` engines with every expert pinned
  and hardware wait.
- B8 best is 125.8 tok/s across 8 streams (committed), and 129.9 in the #1228
  thread.
- Neither exit is met.

In the committed 2026-09-27 rank-0 window, one 10 s interval ran 171 B8 waves
with `run_ms` 8835 and `peer_wait_ms` 2987. That is about 52 ms per wave, of
which 17.5 ms was spent waiting on peers (arithmetic). A 16-rank all-reduce
costs 167 us at p50. At 92 per step, that is about 15 ms of the B1 step
(arithmetic). The lead dev rates collective latency the dominant cost, so
step 3 below is the largest lever. M1 runs with every routed expert pinned;
M6 holds the work that lifts that.

Each step answers a measured cost:

1. **Prefill off the decode path.** Eager chains enqueue straight-line work,
   with no host round trip per collective round or per routed layer. After
   that: graph-captured prefill, or decode ordered ahead of prefill. Measured
   by `decode_wait_ms` in `G5N-WAVE-TIMING` and the eager `CHAIN-TIME` lines.
2. **Host work between waves at most 2 ms.** Decode several steps per
   submission, keep mesh activity and the chain epoch across waves, and take
   graph completion off residentd's thread. Resident decode chains, up to 8
   steps per decode frame, came first: they divide the per-frame cost by the
   chain length without changing it. Measured by `idle`, `key`, `setup` and
   `post` in `G5N-WAVE-TIMING`, divided by `steps/decode` from
   `tools/wave_timeline_report.py` for the cost per step.
3. **Collective waits at most 3 ms per wave.** GPU-initiated RDMA, and the
   pair link for the first reduction level. Measured by `peer_wait_ms`.
4. **Kernels at 80% of bandwidth or better** at B1 and B8. Measured by
   `make bench-glm5-next-batch`.

TECHDEBT: the glm5_next items under steady-state decode hot-path audit, and
mesh collectives.

## M2. Batch scale

Exit: B64 and B256 at 80% of the batch roofline in
[`GLM5_NEXT_ROOFLINE.md`](GLM5_NEXT_ROOFLINE.md) (about 750 and 2200 tok/s at
1K context), with arrivals, completions and cache pressure.

Work: the b16 and b32 variants; KV admitted by resident demand instead of a
full-context reservation per sequence; the replicated DSA state sharded
(context-parallel indexer on by default, then context-parallel latent
attention); one execution workspace per node; two micro-batches overlapping
compute and collectives; a measured reduce-scatter/all-gather crossover.

TECHDEBT: dynamic batching, KV sharding, mesh collectives.

## M3. Serving completeness

Exit: MTP speculation at TP16 raises tokens per second with unchanged output;
sampled rows run on graphs with the FP8 head and MTP; top-k, top-p and
logprobs work on every family; deadlines and priorities hold from the API
through the engine; a client failure drains every sequence slot.

MTP at TP16 needs the work in
[SPECULATION_UNIFIED_DESIGN.md](SPECULATION_UNIFIED_DESIGN.md#glm5_next-mtp-research-path):
a TP-sharded MTP head, draft and verify inside graph chains, and
per-sequence draft KV. glm5_next's MTP path runs at TP1 only today.

TECHDEBT: speculation, serving API, runtime completion.

## M4. One collective platform

Exit: one predeclared collective program per resident slot; one transport for
every family, with k3 moved onto the mesh; algorithms chosen from measured
TP4, TP8 and TP16 profiles; one topology schema for both rails.

TECHDEBT: TP collective control plane, adaptive all-reduce, dual-fabric
topology contract, mesh collectives.

## M5. Placement

Exit:

- TP16, PP16 and TP4 x PP4 each serve GLM 5.3 Flash with the same tokens as
  TP16;
- tok/s and latency are measured for each topology from B1 to B1024, and each
  batch range is served by the topology that measured best;
- a co-resident TP16 model is gang-scheduled without ordering hazards.

Today the glm5_next serving adapter accepts only TP degree 16
(`spark_glm5_next_serving_adapter.c:37`).

TECHDEBT: resident TP4 x PP4 execution.

## M6. Residency and multi-model serving

Exit: a nonresident model serves within 60 s; the pooled store reads at least
20 Gb/s; KV survives eviction and reactivation; lanes evict by priority,
queue requests behind a loading model, cap loader bandwidth and bound output
quanta; eight lanes and TP4/TP8 sub-lanes; MPS evaluated on GB10; GLM graph
decode runs on a bounded expert working set with unchanged greedy tokens.

Relocatable expert graphs come first. GLM whole-chain graphs need every
routed expert leased and pinned today: `SparkGlm5NextGraphClaimExperts`
refuses the graph otherwise (`78c2c21`). So experts can be neither evicted
nor loaded lazily while graphs are live. The design is to capture once,
record which kernel arguments are expert pointers, and patch them on load.
It is described in the messages of `8adebc6`, `360c0ee` and `1a674be`, and
it is not implemented.

TECHDEBT: model residency and storage, multi-model serving, topology-aware
lane sub-allocation, MPS evaluation on GB10.

## M7. Models and code health

Exit: contracts for MiniMax H3 and Qwen 3.8 Pro (Qwen 3.8 27B's is
`model_contracts/qwen38_27b_authoritative.json`); the K3 MXFP4 path;
numerical and service qualification for every driver; the near-duplicate driver code folded into common modules; one driver per model
with prewarmed B1-B1024 specializations; one packer; the DSV4 hot-path items;
one immutable qualification bundle per release.

TECHDEBT: model contracts, driver consolidation, packaging and provenance,
steady-state decode hot-path audit, production qualification.

## M8. Beyond CUDA and Sparks

Exit:

- modules run behind the device layer, with Metal and ROCm backends;
- eight Mac Studios serve the catalog as replicas and Thunderbolt 5 islands;
- a mixed fleet that prefills on Sparks and decodes on Studios serves a
  workload faster end to end than either pool alone;
- the fleet grows from 4 to 8 to 16 Sparks with rollback receipts.

The Studio pool needs the device layer and a Metal backend, not the GLM work
of M1-M3, so it can proceed alongside them once the Studios arrive. Pipeline
stages across the two classes also need M5's pipeline execution.

TECHDEBT: hardware independence, Mac Studio deployment, incremental
expansion.

## M9. Fleet-hours

Exit: a business rents a whole fleet without our help.

- It opens an account with its company name and a work email confirmed by a one-time code, and accepts the Hourly Terms with a box recorded against the texts' hashes.
- It pays a USDC or USDT invoice through the payment processor, and its hours appear once the payment clears.
- It reserves hours on a fleet's calendar while signed in, or through the API with an access token. A two-year customer can take an hourly reservation more than 6 hours ahead, and the hourly customer is told and gets the hour back.
- During its hours its access tokens reach the fleet, and nobody else's do.
- `sparkpipe.ai/availability.txt` shows live capacity, regenerated every 30 seconds.
- Every payment, issue of hours, reservation, displacement, credit and consumption is a line in the hash-chained ledger, whose daily root is published.
- Downtime credits are issued from the fleet's own records, without a claim.

Work:

- the account service: signed acceptance, business details, sanctions screening, the business check and the 720-hour limit;
- VAT invoices (USDC and USDT invoices through CoinPayments are built: `docs/PAYMENTS.md`);
- the ledger and its daily root;
- the reservation writer, the calendar, displacement and expiry;
- access tokens scoped to reserved hours, and the gate in front of the fleet;
- `availability.txt`;
- automatic credits.

It depends on:

- the legal review of the terms, which settles the business flow;
- the 24-hour retention of cached request state in production.

TECHDEBT: fleet-hours.

## M10. Speculation at B1 and above

Exit: on GLM 5.3 Flash at TP16, speculation raises tokens per second at B1,
B8 and B64. It is measured against the same build without speculation, with
greedy tokens unchanged.

Work:

- verify drafts inside decode chains, since MTP drafts skip multi-step
  chains today;
- a draft tree fed by several drafters, cut short or extended conditionally,
  that spends the compute left idle below the compute-bound batch size
  (GOALS.md);
- drafters on the rtx5090 hub. It is 22.6 us p50 from sparkf and about
  300 us p50 from the other Sparks, which reach it through sparkf
  (PERFORMANCE_STATUS.md, Measured fabric). GLM needs a glm5_next
  `DraftRemoteChain` call site first, with drafting pipelined one round
  ahead of verify;
- a DFlash2 capture inside the GLM graph engine, if a GLM 5.3 Flash DFlash2
  drafter is to be qualified.

Exact comparison at B8 and B64 needs M2's batch-invariant numerics: batched
serving changes tokens today (the 2026-09-28 COMPSEC-17 run report). This
milestone builds on M3's MTP work.

TECHDEBT: speculation, dynamic batching.

## M11. Drivers in parallel

Exit: every driver module meets three conditions on merged main. It builds
for sm_121a and links. It is releasable by
`tools/module_build_release.sh`. It has a functional receipt (COMPSEC-17 or
its model's equivalent) and a B1 measurement in PERFORMANCE_STATUS.md.

Drivers are taken in the order that GOALS.md's model direction sets.

Work: the no-hardware iterations i34-i41, then requalification on the fleet,
both listed in [`HANDOFF_DRIVERS_2026-09-28.md`](HANDOFF_DRIVERS_2026-09-28.md).

TECHDEBT: driver consolidation, production qualification.
