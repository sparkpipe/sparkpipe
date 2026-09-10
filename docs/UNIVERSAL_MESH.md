# The universal mesh: one all-to-all substrate

## The simplification

A complete graph (all-to-all) contains every topology as a subset:
- Tree allreduce: 7 of the 15 peer connections (tree neighbors)
- All-to-all: all 15 peer connections
- Pipeline: 1 connection (next stage)
- Broadcast down-pass: tree-neighbor connections (same as tree)
- 4× TP4: 3 connections within each group
- 2× TP8: 7 connections within each group

One mesh, 30 QPs per node (15 peers × send+receive). Every collective
algorithm is a routing decision on the same substrate. No per-topology
mesh management, no topology-aware wiring, no multiple mesh tables.

## What weightd does

At daemon start (once per node boot):
1. Open verbs device
2. Create 15 send QPs + 15 receive QPs (one pair per peer)
3. Register fixed receive buffers
4. Publish all QPNs to hub (HTTP POST)
5. Await all peers' QPNs (HTTP GET, 15s budget)
6. Transition all QPs to RTS
7. Done — the mesh serves every model, every topology, every collective

## What the residentd does

On lazy attach:
1. Get the mesh-ready flag + QP mapping from weightd
2. The module's collective code routes over the shared QPs:
   - Tree allreduce: fire sends to tree neighbors (7 of 15)
   - All-to-all at B1: fire sends to all 15 peers
   - Pipeline: fire send to next stage (1 of 15)
   - The crossover between tree and all-to-all is per-model, measured

## Deploy cycle

manifest → agent detects (1s) → kill -1 (instant) → start → warm attach
(100ms) → module load (1s) → ready. No QP creation, no record exchange,
no topology negotiation. **3-5 seconds.**

## The crossover measurement (per model)

For each model, measure tok/s with tree vs all-to-all at different batch
sizes. The crossover point (where all-to-all's single-hop latency advantage
is overcome by the tree's bandwidth efficiency) is different for each
model's hidden dimension and expert count. The module selects the strategy
at runtime based on the measured crossover.

## QP multiplexing

The current "4 sessions per route" (send, receive, ack_send, ack_receive)
can be multiplexed over 2 QPs per peer: the send QP carries both data and
ack writes (different buffer offsets), the receive QP handles both. This
halves the QP count and simplifies the mesh. The tag/offset in the RDMA
write header distinguishes data from ack.
