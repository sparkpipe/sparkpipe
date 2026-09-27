# Roadmap

The goal is the system that [`README.md`](../README.md) describes, with every
section of [`TECHDEBT.md`](../TECHDEBT.md) closed and each claim backed by
receipts from a clean merged `main`. This file puts that work in order. Each
milestone has an exit criterion that a measurement decides. The measurements
themselves belong in [`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md).

Work goes one iteration at a time. An iteration changes one thing, says which
exit criterion it moves, and names the measurement that picks the next step.

## M1. GLM 5.3 Flash at 80% of the memory roofline

Exit, on the sixteen-Spark fleet at TP16 and 1K context:

- a B8 decode step of at most 26 ms, which is 80% of the 21 ms floor (about
  300 tok/s across 8 streams);
- B1 at most 10 ms per token, which is 80% of the 8.0 ms floor;
- greedy tokens unchanged.

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

TECHDEBT: speculation, serving API, runtime completion.

## M4. One collective platform

Exit: one predeclared collective program per resident slot; one transport for
every family, with DSV4 moved onto the mesh; algorithms chosen from measured
TP4, TP8 and TP16 profiles; one topology schema for both rails.

TECHDEBT: TP collective control plane, adaptive all-reduce, dual-fabric
topology contract, mesh collectives.

## M5. Placement

Exit: TP4 × PP4 serves GLM 5.3 Flash with the same tokens as TP16, and a
co-resident TP16 model is gang-scheduled without ordering hazards.

TECHDEBT: resident TP4 x PP4 execution.

## M6. Residency and multi-model serving

Exit: a nonresident model serves within 60 s; the pooled store reads at least
20 Gb/s; KV survives eviction and reactivation; lanes evict by priority,
queue requests behind a loading model, cap loader bandwidth and bound output
quanta; eight lanes and TP4/TP8 sub-lanes; MPS evaluated on GB10.

TECHDEBT: model residency and storage, multi-model serving, topology-aware
lane sub-allocation, MPS evaluation on GB10.

## M7. Models and code health

Exit: contracts for MiniMax H3, Qwen 3.8 Pro and Qwen 3.8 27B; the K3 MXFP4
path; numerical and service qualification for every driver; the
near-duplicate driver code folded into common modules; one driver per model
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

## M9. The provider network

Exit, with an installation that sparkpipe.ai does not operate:

- it registers, joins the tailnet, offers a model at its own price and
  serves buyer traffic through the sparkpipe.ai router;
- one buyer's requests for a model run on at least two providers at once;
- the owner's own traffic takes over within one frame;
- at least 2% of served tokens are replayed on reference nodes, with exact
  greedy comparison and logprob comparison for sampled traffic, and a
  planted quantized provider is caught;
- payouts settle from signed receipts after the challenge window;
- the catalog, buyer console and provider dashboard are live on
  sparkpipe.ai.

Work:

- the tailnet: its control server, tagged keys and access rules;
- the provider agent and registration;
- an owner-first priority class, with preemption at frame boundaries;
- network requests that finish within their bound or resume elsewhere;
- signed completion receipts;
- the router on LiteLLM: one deployment per offer at the provider's price,
  routing by price, latency, load and verification record, and prefix
  affinity;
- the audit service (sampler, replay scheduler, comparator, challenge
  state), and reference nodes for each hardware type;
- bonds and payouts;
- the site's catalog, buyer console and provider dashboard (its static
  pages and playground are in `site/`).

It depends on:

- batch-invariant numerics, so replays compare bit for bit (M2);
- logprobs (M3);
- model promotion (M6).

Revenue per provider scales with batched throughput, so M2 matters to this
milestone as much as the network code does.

TECHDEBT: provider network.
