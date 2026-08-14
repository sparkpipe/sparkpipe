# SparkPipe

SparkPipe is a private, on-premises serving engine for open-source frontier
models. A business can start with four NVIDIA DGX Sparks, expand to eight and
sixteen, then add one or more DGX Stations without changing its serving API or
model packages. SparkPipe keeps a model working set resident, promotes other
configured models in at most one minute, and exposes one OpenAI-compatible
endpoint for interactive chat and large agent workloads.

Callers choose the model and specify request priority and deadline. SparkPipe
chooses placement, batch width, stage microbatch geometry, speculation policy,
and collective algorithm to maximize useful hardware occupancy while honoring
those priorities.

Applications may add memory, tools, routing, policy, and user interfaces above
the API. SparkPipe owns model execution, scheduling, transport, residency,
storage, and readiness evidence.

Measured engineering results are recorded separately from projections and
release claims in [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md). The latest
retained DSV4 Flash TP4 B1 result is 32.57 end-to-end cached-KV decode tok/s
mean over four runs (32.81 tok/s best), with one request, no speculation, and
exact emitted-token parity against the retained control. It is a scratch
candidate milestone based on `main`, not a merged-main production
qualification; the performance status records the exact timing boundary,
identities, raw-receipt hashes, and reproduction command.

## System shape

```text
OpenAI-compatible API
        |
priority and deadline scheduler
        |
resident model catalog + sub-minute promotion
        |
model execution plans
  large MoE: TP4 x PP4
  smaller dense: resident TP16 where appropriate
        |
adaptive collectives over two required fabrics
  100 Gb/s switched all-to-all
  nominal 200 Gb/s pairwise direct, PCIe-limited near 100 Gb/s useful
        |
sixteen Sparks with local KV and model-shard storage
```

The scheduler serves one request, a related batch, or a dynamic agent
population through the same interface. High-priority interactive work receives
short bounded quanta. Throughput work fills the remaining compute, memory, and
network capacity. Batch formation is automatic and never requires a caller to
pick a B-number.

## Incremental deployment

| Scale | Execution options |
| --- | --- |
| 4 Sparks | TP4 and model-specific entry plans |
| 8 Sparks | TP8 or TP4 x PP2 |
| 16 Sparks | canonical TP4 x PP4 for large MoE models |
| DGX Station | standalone capacity or Spark-fabric enhancement |
| 4-8 Stations | office-deployable fabric for the largest models |

Every Spark scale keeps complete direct pairs and the same combined-fabric
contract. Expansion preserves the endpoint, scheduler semantics, package
identity, storage hierarchy, and readiness rules. The hardware strategy uses
prosumer products with an active secondary market so capacity can follow actual
business demand.

## Combined dual fabric

Every Spark uses both data-plane rails:

- the CRS804 switched rail provides all-to-all reach at 100 Gb/s;
- the paired direct rail connects `rank XOR 1` at a nominal 200 Gb/s.

The direct rail is limited by the GB10 PCIe path to about 110 Gb/s maximum and
normally operates near 100 Gb/s useful payload. The switch therefore does not
reduce practical per-port throughput. It is not a fallback. Both routes are
pinned and mandatory.

SparkPipe selects one topology-aware all-reduce plan from communicator size,
payload bytes, datatype, and a measured hardware profile:

| Payload regime | Algorithm |
| --- | --- |
| Small | Recursive doubling over direct XOR-1, then switched XOR partners |
| Medium | Recursive halving/doubling over the same partner tree |
| Large | Counter-rotating split rings over alternating direct and switched edges |

The large-payload path splits alternating chunks into two disjoint stripes. At
steady state every Spark overlaps direct TX/RX with switched TX/RX. This halves
switch-facing traffic relative to a one-port ring while retaining the switch as
half of the active collective.

## Resident model topology

On sixteen Sparks, large MoE models use `TP4 x PP4` as the canonical resident
layout:

| PP stage | TP ranks | Direct pairs |
| ---: | --- | --- |
| 0 | `0-3` | `0<->1`, `2<->3` |
| 1 | `4-7` | `4<->5`, `6<->7` |
| 2 | `8-B` | `8<->9`, `A<->B` |
| 3 | `C-F` | `C<->D`, `E<->F` |

B4 fills all four stages. Shared-prefix B8, speculation, and high-concurrency
agent modes change scheduler policy and stage-local microbatch width without
moving weights or KV. B1-B3 accept pipeline bubbles instead of forcing model
redistribution.

A smaller dense model may stay resident with a global TP16 communicator. The
scheduler gang-schedules bounded all-rank quanta between co-resident execution
plans; it does not inject unrelated collectives into committed model work.

## Storage hierarchy

Every Spark in a 4-, 8-, or 16-node deployment has a 4 TB internal NVMe and at
least 4 TB of external NVMe.

| Tier | Per Spark | Purpose |
| --- | ---: | --- |
| Internal hot KV | 2.5 TB | KV cache and resumable request state |
| Internal active shards | 1.0 TB | rank-local working-set model data |
| Internal system | 0.5 TB | OS, runtime, receipts, bounded scratch |
| External direct | at least 1.0 TB | rank-local model access and promotion staging |
| External pooled | remaining capacity | striped RAID-like model store targeting 20 Gb/s useful reads |

Model packages are immutable and content-addressed across every tier. Model
promotion verifies exact package identity, installs rank-local shards, binds
stable pointers, prewarms kernels and graphs, constructs communicators, and
publishes readiness atomically.

## DGX Station path

The next deployment class combines four or eight DGX Stations, either as a
standalone fabric or as an enhancement to the Spark pipeline. SparkPipe uses
model sharding and a Station-specific collective profile to aggregate around
each Station's memory-bandwidth limit.

The target for the largest models is roughly half the throughput of a DGX B300
datacenter system, delivered in an ordinary office deployment. That target is a
projection until retained hardware measurements close it; it is not a current
benchmark claim.

## Model scope

SparkPipe targets all open-source frontier-level models that can be mapped to
the hardware. The initial product set is:

- DeepSeek V4 Flash and DeepSeek V4 Pro;
- GLM 5.2;
- Kimi K3;
- MiniMax H3;
- Qwen 3.8 Pro and Qwen 3.8 27B.

MiniMax 2.5 is not a support target. A product target is not a production-ready
claim: every exact checkpoint needs its own contract, kernels, numerical
result, transport profile, and live service receipt.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) is the intended final system and changes
  only when the goal changes.
- [`SPEC.md`](SPEC.md) is the firmware and package contract.
- [`TECHDEBT.md`](TECHDEBT.md) is the only list of unfinished implementation
  work; completed entries are removed.
- [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md) is the only performance
  ledger; measurements and projections are separated.
- [`docs/README.md`](docs/README.md) indexes maintained technical references.

Superseded designs, phase reports, handoffs, investigation notes, and old
validation snapshots are preserved under [`docs/archive/`](docs/archive/).
They are historical evidence, not part of the live documentation set.

## Build and test

```sh
make clean
make -j1 all
make test
sh tools/gates.sh
```

Target-hardware qualification additionally requires exact CUDA, topology,
transport, numerical, and live service receipts from a clean merged-main
release. Host or simulator success cannot stand in for those gates.

See [sparkpipe.ai](https://sparkpipe.ai/) for the public project overview.
