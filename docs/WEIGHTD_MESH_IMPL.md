# Weightd mesh implementation plan (Phase 1: daemon-start wiring)

## What weightd does at startup (once per node boot, 15s budget)

```
weightd main:
  ServerCreate (existing)
  MESH INIT (new):
    - Load transport DSO (hidden_transport.so) — same one residentds use
    - Open verbs device (already does for memory registration)
    - Read topology from the launch parameters the agent states
      (--mesh-rank --mesh-interface --mesh-sgid-index; rank is the
      deployment's roster index for this host, never derived from the
      hostname):
    - For each tree route (7 per node in TP16 tree topology):
      - Create 4 QPs (send, receive, ack_send, ack_receive)
      - Register fixed receive buffers
    - Publish all QPNs to hub via HTTP (one POST per route, or batch)
    - Await all peers' QPNs via HTTP GET (poll at 500ms, budget 15s)
    - Transition all QPs to RTS (ready to send)
    - Set mesh_ready = true
  ServerRun (existing event loop)
```

## What changes in the lazy attach

```
LazyAttachRequest gains:
  (nothing new — the mesh is per-node, not per-model)

LazyAttachResult gains:
  uint32_t mesh_ready;         // 1 if the daemon's mesh is wired
  uint64_t mesh_generation;    // changes when the mesh is rewired

The residentd checks mesh_ready:
  - If 0: the daemon is still wiring (first boot after daemon start)
    → the residentd can proceed with degraded routes (current behavior)
  - If 1: the mesh is wired, all QPs are ready
    → the residentd skips its own QP creation entirely
```

## What changes in the residentd transport

```
Current: residentd creates its own QPs + publishes own records + awaits peers
New:     residentd asks weightd for the mesh; if ready, uses weightd's QPs

The transport adapter (rdma.cu) gains a "proxy mode":
  - Send: write to weightd's mapped buffer, then signal weightd via
    the existing unix socket: "flush route R" (~2µs)
  - Weightd posts the RDMA write from its own QP (which is already wired)
  - Receive: peer writes to weightd's registered buffer (same mapped memory
    the residentd reads from — zero copy, no change to the data path)
```

## What stays per-model

- Memory registration for model-specific buffers (KV cache, activations)
- Credit tracking and slot management
- The tree collective algorithm (fold, combine, reduce)
- The module's compute kernels

## What becomes shared

- QPs (one set per node per TP degree)
- Peer connections (established once at weightd start)
- The rendezvous records (static per node boot — no re-exchange per deploy)

## Deploy cycle after this change

```
manifest change → agent detects (1s) → kill -9 (instant) → start residentd
→ residentd: weightd attach (100ms warm, no SHA, no spine) → module load (1s)
→ ready (no session opens, no record exchange, no HTTP fetch)
Total: ~3-5 seconds
```

## Implementation steps

1. weightd: load transport + create QPs + publish/await records at startup
   - Reuse the existing rdma.cu session-open code (embed, don't rewrite)
   - The topology comes from a config file the agent writes at node boot
2. Lazy attach result: add mesh_ready field
3. Transport proxy mode: send work requests through weightd's socket
4. Module: skip session opens when mesh_ready
5. Deploy: verify 5s cycle
