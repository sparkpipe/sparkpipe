# Mesh inventory: every topology weightd must wire

## Allreduce meshes (bidirectional tree)

| Topology | Routes per node | QPs per node | Notes |
|----------|----------------|--------------|-------|
| TP16 tree | 7 × 4 sessions = 28 | 28 | Primary: glm53flash, glm53full, dsv4 |
| 4× TP4 trees | 3 × 4 = 12 (within group) | 12 | Already deployed: dsv4flash.tp4pp4, glm53flash.fp8.tp4pp4 |
| 2× TP8 trees | 5 × 4 = 20 (within group) | 20 | Already deployed: glm53flash.fp8.tp8 |
| TP4 tree (single) | 3 × 4 = 12 | 12 | qwen27b, k3 |

## All-to-all exchange (B1 decode: every node sends to every peer in parallel)

For B1 allreduce, the fastest pattern is all-to-all: each node RDMA-writes
its partial vector (~8KB bf16 at 4096 hidden) to all 15 peers simultaneously
(the switched fabric handles full duplex on all 16 ports — 240 parallel
transfers at 8KB each = 1.9MB total, trivial for the switch). Each node then
locally sums 16 vectors. ONE hop, no tree stages, no fold kernels.

| Topology | Send QPs per node | Receive QPs per node | Total |
|----------|-------------------|---------------------|-------|
| TP16 all-to-all | 15 | 15 | 30 QPs |
| TP8 all-to-all (per group) | 7 | 7 | 14 QPs |
| TP4 all-to-all (per group) | 3 | 3 | 6 QPs |

These are ADDITIONAL to the tree allreduce QPs (the tree is better for B8+
where bandwidth matters more than latency; all-to-all is better at B1 where
the vector is small and latency dominates).

## Broadcast (tree down-pass for pipeline/logits)

The allreduce tree's down-pass handles one-to-many distribution (pipeline
scatter, token broadcast from rank 0). Same QPs as allreduce.

## Pipeline connections (point-to-point between groups)

| Topology | Routes per node | QPs per node | Notes |
|----------|----------------|--------------|-------|
| TP4×PP4 pipeline | 4 per node (to next stage's 4 ranks) | 4 | Stage 0→1→2→3, all ranks in stage connect to next |
| TP8×PP2 pipeline | 8 per node (to next stage's 8 ranks) | 8 | Stage 0→1 |
| PP16 pipeline | 16 per node (to next stage) | 1 | Pure pipeline: rank N → rank N+1 only |

## Total worst case per node

If ALL topologies are active simultaneously:
- Allreduce trees: 28 + 12 + 20 + 12 = 72 QPs
- All-to-all (B1): 30 + 14 + 6 = 50 QPs
- Pipeline: 4 + 8 + 1 = 13 QPs
- **Total: ~135 QPs per node**

## Lazy creation

Each mesh is created on first attach for its topology key:
`topology_key = hash(tp_degree, pipeline_degree, group_id, collective_kind)`

weightd stores a mesh table; the first residentd referencing a topology
triggers creation (15s hit). All subsequent attaches return the existing
mesh instantly.
