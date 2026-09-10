# Mesh inventory: every topology weightd must wire

## Allreduce meshes (bidirectional tree)

| Topology | Routes per node | QPs per node | Notes |
|----------|----------------|--------------|-------|
| TP16 tree | 7 × 4 sessions = 28 | 28 | Primary: glm53flash, glm53full, dsv4 |
| 4× TP4 trees | 3 × 4 = 12 (within group) | 12 | Already deployed: dsv4flash.tp4pp4, glm53flash.fp8.tp4pp4 |
| 2× TP8 trees | 5 × 4 = 20 (within group) | 20 | Already deployed: glm53flash.fp8.tp8 |
| TP4 tree (single) | 3 × 4 = 12 | 12 | qwen27b, k3 |

## Broadcast (same QPs as allreduce, down-pass only)

No separate mesh — the allreduce tree handles broadcast as its down-pass
stage (rank 0's data flows down the tree, 4 hops for TP16, intermediate
nodes fan out to their subtrees). The switched fabric means any node can
reach any node; the tree distributes send load instead of bottlennecking
at the root's NIC.

## Pipeline connections (point-to-point between groups)

| Topology | Routes per node | QPs per node | Notes |
|----------|----------------|--------------|-------|
| TP4×PP4 pipeline | 4 per node (to next stage's 4 ranks) | 4 | Stage 0→1→2→3, all ranks in stage connect to next |
| TP8×PP2 pipeline | 8 per node (to next stage's 8 ranks) | 8 | Stage 0→1 |
| PP16 pipeline | 16 per node (to next stage) | 1 | Pure pipeline: rank N → rank N+1 only |

## Total worst case per node

If ALL topologies are active simultaneously:
- Allreduce (broadcast shares these QPs): 28 + 12 + 20 + 12 = 72 QPs
- Pipeline: 4 + 8 + 1 = 13 QPs
- **Total: ~85 QPs per node**

## Lazy creation

Each mesh is created on first attach for its topology key:
`topology_key = hash(tp_degree, pipeline_degree, group_id, collective_kind)`

weightd stores a mesh table; the first residentd referencing a topology
triggers creation (15s hit). All subsequent attaches return the existing
mesh instantly.
