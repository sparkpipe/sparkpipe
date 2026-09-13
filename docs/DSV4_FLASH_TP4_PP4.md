# DeepSeek V4 Flash TP4 x PP4

DeepSeek V4 Flash uses the canonical sixteen-Spark large-model layout. Four
pipeline stages each contain a four-rank tensor-parallel communicator and two
complete direct pairs.

## Placement

```text
world_rank = pp_stage * 4 + tp_rank
```

| PP stage | Layers | TP ranks | Direct pairs |
| ---: | --- | --- | --- |
| 0 | 0-10 | `0-3` | `0<->1`, `2<->3` |
| 1 | 11-21 | `4-7` | `4<->5`, `6<->7` |
| 2 | 22-32 | `8-B` | `8<->9`, `A<->B` |
| 3 | 33-42 | `C-F` | `C<->D`, `E<->F` |

The four physical pipeline lanes are:

```text
0 -> 4 -> 8 -> C
1 -> 5 -> 9 -> D
2 -> 6 -> A -> E
3 -> 7 -> B -> F
```

Each stage owns only its layer slice. KV pages are stage-local and replicated
across the four TP ranks that execute the slice. TP does not divide the stage's
KV capacity ledger by four.

## Residency

Every rank keeps its weights, KV, communicators, registered transport arenas,
CUDA modules, graphs, and workspaces resident. The model package binds one
checkpoint, precision route, stage plan, topology profile, and collective
profile.

Switching batch width, speculation, or admission policy does not reload the
driver, model files, weights, KV, or communicator. Model eviction preserves
resumable KV in the internal hot tier and immutable model shards in the internal
and external model tiers.

## Scheduler modes

### Interactive shared-prefix B8

Eight suffix lanes may share one physical prefix. Four target stages remain
occupied while DSpark produces proposals during request revisit slack. The
scheduler prioritizes the oldest ready suffix subject to explicit request
priority and deadline.

### Throughput agents

The scheduler continuously forms stage-local microbatches from all ready
agents. Global concurrency may reach B128-B1024 while the best stage-local
microbatch remains smaller. Requests advance by token readiness rather than
whole-population lockstep.

### Low concurrency

B1-B3 accept pipeline bubbles. The resident model is not redistributed into a
TP16 layout solely to improve a transient low-concurrency request.

## Collectives

Every TP4 stage uses the combined switched-plus-direct collective service. The
profile selects recursive doubling, recursive halving/doubling, or
counter-rotating split rings from effective payload bytes and datatype. A
missing rail or invalid route fails readiness.

The collective contract is
[`PAIRED_DUAL_LINK_ALLREDUCE.md`](PAIRED_DUAL_LINK_ALLREDUCE.md). Measured
crossovers and model throughput live only in
[`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md).

## Package contract

The generated release contains one rank-local stage pack per physical rank.
Each pack records:

- exact checkpoint and contract hashes;
- PP stage, TP rank, and layer interval;
- weight, KV, and precision formats;
- model and topology profile hashes;
- expected model-shard and hot-KV storage roles; and
- immutable payload byte count and SHA-256.

All sixteen ranks agree on package generation, model contract, topology,
collective profile, and ready generation before the endpoint publishes the
model.

Implementation gaps and qualification work are maintained only in
[`../TECHDEBT.md`](../TECHDEBT.md).
