# SparkPipe Architecture

SparkPipe is a private, on-premises frontier-model serving engine for businesses,
teams, and individuals. It scales from four NVIDIA DGX Sparks through larger
Spark and DGX Station fabrics. Applications use its OpenAI-compatible API and
may add memory, tools, policy, routing, and user interfaces above it. SparkPipe
owns model execution, scheduling, transport, residency, storage, and evidence.

## Serving objective

One endpoint exposes the configured open-source frontier model catalog. The
active working set remains resident when capacity permits; a nonresident model
can be promoted from the model store in at most one minute, with substantially
lower latency expected for warm rank-local shards.

Callers may send one interactive request, a related batch, or a large dynamic
population of agent requests. They specify model, priority, deadline, and
request content. They do not select batch width, stage microbatch geometry,
collective algorithm, or physical placement.

The scheduler continuously chooses those execution details to maximize useful
hardware occupancy while honoring priorities. Interactive chat and throughput
agent work coexist: short high-priority quanta bound chat latency, and agent
work fills the remaining compute, memory, and network capacity.

## Physical fabric

Every Spark has two required data-plane rails:

1. A 100 Gb/s port connected to the CRS804 switch for all-to-all reach.
2. A nominal 200 Gb/s port connected directly to one paired Spark.

The direct port is limited by the GB10 PCIe path to about 110 Gb/s maximum
useful payload and normally operates near 100 Gb/s. The switched rail therefore
matches the practical per-port throughput of the direct rail. Neither rail is a
fallback for the other.

The direct pairs are:

```text
0 <-> 1    2 <-> 3    4 <-> 5    6 <-> 7
8 <-> 9    A <-> B    C <-> D    E <-> F
```

Communicator construction pins every switch edge to the CRS804 interface and
every direct edge to `rank XOR 1`. Failure to establish either route is fatal.
Control and management traffic are not inference transports.

## Deployment ladder

The serving contract is identical at every scale:

| Deployment | Primary use |
| --- | --- |
| 4 Sparks | Entry system, TP4 models, interactive and modest agent workloads |
| 8 Sparks | Larger resident set, TP8 or TP4 x PP2 model plans |
| 16 Sparks | Full TP4 x PP4 large-model pipeline and eight direct pairs |
| DGX Station | Higher-capacity standalone node or Spark-fabric enhancement |
| Multiple Stations | Four- or eight-node Station fabric for the largest models |

Four- and eight-Spark deployments use complete `rank XOR 1` direct pairs and
the same switched-plus-direct collective contract. Expansion adds nodes,
storage, and generated deployment plans without changing the API, model package
identity, scheduler semantics, or evidence rules.

The hardware strategy favors office-deployable prosumer systems with an active
secondary market. A business can add capacity incrementally as request volume,
model size, or latency requirements grow rather than replacing the serving
stack or committing immediately to datacenter infrastructure.

## Unified collective service

SparkPipe exposes one topology-aware all-reduce service. The immutable hardware
profile selects an algorithm in O(1) from communicator size, payload bytes,
datatype, and measured crossover tables.

### Small payloads

Topology-aware recursive doubling minimizes dependent network exchanges. The
first exchange uses `rank XOR 1` over the direct rail. Remaining XOR-distance
partners use the switched rail. TP8 completes in three dependency steps and
TP16 in four.

### Medium payloads

Topology-aware recursive halving/doubling performs a reduce-scatter followed by
an all-gather over the same XOR partner tree. It transfers fewer bytes than
recursive doubling and uses fewer dependent steps than a ring.

### Large payloads

Counter-rotating split rings divide aligned, alternating chunks into two
disjoint stripes. The forward ring alternates direct and switched edges; the
reverse ring traverses the same physical cycle in the opposite direction.

At steady state every Spark performs all four operations concurrently:

```text
direct TX + direct RX + switch TX + switch RX
```

For TP16, each physical direction carries `0.9375S` for a tensor of `S` bytes.
For TP8 it carries `0.875S`. This halves switch-facing traffic relative to a
one-port ring while retaining the switch as half of the active collective. The
ideal bandwidth gain is 2x, not 4x.

Forward and reverse stripes use independent queues, registered buffers,
credits, and progress state. The steady-state path performs no allocation,
registration, connection setup, CPU payload copy, device-wide synchronization,
or CPU-dispatched chunk transition.

The complete collective contract is in
[`docs/PAIRED_DUAL_LINK_ALLREDUCE.md`](docs/PAIRED_DUAL_LINK_ALLREDUCE.md).

## Resident model topology

On sixteen Sparks, large MoE models use `TP4 x PP4` as the canonical resident
layout:

| PP stage | TP ranks | Direct pairs |
| ---: | --- | --- |
| 0 | `0-3` | `0<->1`, `2<->3` |
| 1 | `4-7` | `4<->5`, `6<->7` |
| 2 | `8-B` | `8<->9`, `A<->B` |
| 3 | `C-F` | `C<->D`, `E<->F` |

Every TP group contains two complete direct pairs. Four PP stages limit
pipeline bubbles while providing enough pipeline depth for aggregate
throughput. B4 fills the pipeline exactly.

Weights, KV, communicators, CUDA modules, graphs, queues, and workspaces remain
resident. Runtime modes change scheduler policy, not placement:

- Interactive shared-prefix B8 uses independent suffix lanes and DSpark
  proposal work during pipeline revisit slack.
- Throughput mode continuously forms stage-local microbatches from ready
  requests. Global concurrency and stage-local microbatch width are independent.
- B1-B3 accept pipeline bubbles rather than redistributing the model into TP16.

A smaller dense model may remain resident with a global TP16 communicator.
Because both layouts consume all sixteen Sparks when executing, the scheduler
gang-schedules bounded all-rank quanta. It never interleaves an unrelated
collective inside another model's committed kernel sequence.

Model activation changes an execution plan and residency assignment, not the
public endpoint. Eviction preserves resumable KV and model artifacts in the
storage hierarchy. Promotion installs rank-local shards, binds stable pointers,
prewarms kernels and graphs, constructs communicators, and publishes readiness
atomically.

## Storage hierarchy

Every Spark in a 4-, 8-, or 16-node deployment has a 4 TB internal NVMe and
at least 4 TB of external NVMe.

The internal drive is partitioned by role:

| Capacity per Spark | Role |
| ---: | --- |
| 2.5 TB | Hot KV cache and resumable request state |
| 1.0 TB | Rank-local shards for the active model working set |
| 0.5 TB | Operating system, runtime, receipts, and bounded scratch |

The external drive contributes two model-storage tiers:

| Capacity per Spark | Role |
| ---: | --- |
| At least 1 TB | Direct rank-local model access and promotion staging |
| Remaining capacity | Striped RAID-like model-data pool across the fleet |

The pooled tier targets at least 20 Gb/s useful model-data access. Model
packages are immutable and content-addressed; the pool, direct external tier,
and internal active-shard tier are caches of the same verified package
identity. A promotion never converts or guesses a weight layout in the serving
path.

## Runtime ownership

One generic resident process runs on each rank. A model package binds one exact
adapter, driver, stage pack, weight format, KV contract, topology, and hardware
profile. Startup rejects any identity mismatch.

The common runtime owns:

- request admission, priorities, deadlines, cancellation, and event streaming;
- dynamic microbatch formation and fair queueing;
- resident process lifecycle and immutable deployment identity;
- transport progress, collective sequence identity, and operational counters;
- OpenAI-compatible request and streaming response boundaries.

Model drivers own:

- model geometry, layer execution, routing, sampling, and speculation;
- weight layout, stage-local KV, recurrent state, and graph geometry;
- model-specific kernels, buffers, and numerical qualification.

Batch width, speculation mode, and collective algorithm may change per dispatch
without reloading weights, drivers, KV, communicators, or resident processes.

## Model set

The product scope is all open-source frontier-level models that can be mapped to
the hardware. The initial model set is:

- DeepSeek V4 Flash and DeepSeek V4 Pro;
- GLM 5.2;
- Kimi K3;
- MiniMax H3;
- Qwen 3.8 Pro and Qwen 3.8 27B.

MiniMax 2.5 is not a support target. A model name in this list defines product
scope, not production readiness. Every exact checkpoint requires its own
contract, native kernels, numerical result, transport profile, and live service
receipt.

## Evidence contract

Source identity, host tests, CUDA compilation, transport measurements,
numerical correctness, end-to-end model performance, and hosted-reference
comparisons are separate evidence domains. A result states its exact domain and
cannot qualify another domain by implication.

The architecture changes only when the intended system changes. Open gaps are
maintained in [`TECHDEBT.md`](TECHDEBT.md). Measurements and projections are
maintained in [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md).

## DGX Station deployment class

SparkPipe also targets office-deployable DGX Station clusters. Four- and
eight-node Station systems may operate as a standalone fabric or augment the
Spark pipeline. The execution planner applies the same model-aware sharding,
residency, scheduling, and evidence contracts while using a Station-specific
hardware and collective profile.

The design objective is to aggregate enough memory bandwidth and model
parallelism to run the largest open models at roughly half the throughput of a
DGX B300 datacenter system. This is a product target, not a measured result.
The operational objective is ordinary-office deployment rather than the power,
cooling, and facilities requirements of a roughly 15 kW datacenter appliance.
