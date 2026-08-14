# Remaining Work After Phase 6

Phase 6 closes next-hop acknowledged replay and local asynchronous completion
ownership. The following work remains unfinished or unqualified.

## P0: system-wide distributed transaction

1. Add a real all-rank prepare/reserve phase before any rank performs a
   destructive model-state transition.
2. Require acceptance from every downstream rank, not only the immediate next
   hop, before execution is released.
3. Define one global commit decision after all ranks have completed the same
   request generation and step generation.
4. Define rollback or deterministic cancellation for every failure after one or
   more ranks have executed.
5. Carry the complete transaction identity natively in the resident IPC submit,
   submit-result, model-driver completion, and final-event ABIs. The rank
   daemon's request/sequence/position side map is a fail-closed bridge, not the
   final ABI.
6. Persist a restart epoch or durable terminal tombstones so reconnect/replay
   across process restart cannot be mistaken for a new operation.
7. Give cancellation and release the same acknowledged end-to-end transaction
   semantics as decode and prefill.
8. Integrate transport-window, resident-reservation, execution, and
   completion-ownership credits into the complete protocol. Only resident
   reservation is currently used in the backend hot path.
9. Verify that every acquired credit is returned exactly once on every success,
   retry, reconnect, cancellation, and failure path.
10. Add fault-injection tests for failure at every transaction boundary and
    every lane-completion order.

## P0: nonblocking resident submission

1. Replace the rank daemon's synchronous resident submit-result wait, which can
   block the event loop for up to 30 seconds, with an asynchronous state:

   ```text
   resident submit queued
   -> submit result pending
   -> accepted or rejected
   -> execution/completion ownership
   ```

2. Associate submit results with transaction identity rather than FIFO timing.
3. Continue pumping work input, acknowledgements, final events, and network
   progress while resident admission is pending.
4. Bound submit-result backlog and fail the owning transaction rather than the
   entire daemon when one result is malformed.

## P0: Kimi K3 execution closure

1. Connect the K3 resident-stage doorway to the shipping model-driver path.
2. Carry exact accepted-prefix replay slabs, request/step generations, bonus
   token ownership, and committed recurrent-state ownership through the global
   transaction protocol.
3. Transfer only newly produced Block AttnRes block state and the live partial
   at pipeline boundaries; never resend the complete bank.
4. Link KDA, Gated MLA, Block AttnRes, Stable LatentMoE, two shared experts, and
   the requested BF16-activation/MXFP4-weight routed path into one package.
5. Complete EAGLE-3 feature fusion and recurrent drafting.
6. Qualify the local BF16 expert-activation variant independently of the native
   MXFP8 expert-activation deployment recipe.

## P0: GLM 5.2 execution closure

1. Collapse the historical and first-party fragments into one link-complete
   FP8-expert/BF16-rest module.
2. Connect bulk prefill, decode, MTP, and DSpark to the same current entry points
   and transaction protocol.
3. Feed complete scheduler layer batches into the sealed expert queues.
4. Preserve one active-expert weight load and one grouped GEMM firing per expert
   per sealed layer batch, with weight-stationary continuation for experts that
   exceed one M tile.
5. Validate checkpoint-derived FP8 expert tensors, scale geometry, routing,
   route folding, and complete-layer output.
6. Prove all non-expert weights and activations remain BF16.

## P0: Qwen 3.6 27B execution closure

1. Connect restored work control to the shipping transaction and completion
   protocol.
2. Verify BF16 through packing, binding, attention, recurrent state, dense FFN,
   cache, prefix restore, final head, and streamed output.
3. Add checkpoint-derived complete-stage numerical vectors.

## P0: DeepSeek V4 Flash and Pro execution closure

1. Implement a complete Flash resident stage using class-exact sliding,
   compressed-history, and compressor/indexer arenas.
2. Implement Pro independently; never reuse Flash layer schedules or cache
   geometry.
3. Implement every generated sliding-attention, CSA, and HCA layer class.
4. Implement four-stream manifold-constrained hyper-connections, bootstrap
   hash-MoE, grouped low-rank output projection, and MTP.
5. Validate checkpoint FP4/FP8 formats and complete-stage output for both
   variants.

## P0: CUDA 13 and Blackwell qualification

1. Compile every owned CUDA translation unit for exact `compute_121a` PTX.
2. Assemble and device-link for exact `sm_121a`.
3. Retain `ptxas` register, spill, local-memory, and shared-memory reports.
4. Inspect final linked artifacts with `cuobjdump`.
5. Run independent BF16, FP8 E4M3, MXFP4 E2M1, UE4M3, and UE8M0 numerical
   vectors over multiple rows, experts, output tiles, and scale groups.
6. Run Compute Sanitizer memory, race, synchronization, and initialization
   checks on DGX Spark.
7. Retain exact-package B1 through B1024 latency, throughput, bandwidth,
   occupancy, power, and topology receipts.

## P1: acknowledgement-window throughput

1. Replace stop-and-wait forwarding with a bounded selective acknowledgement
   window.
2. Use transport-window credit to permit multiple packets in flight while
   preserving per-transaction order.
3. Add sequence numbers and selective replay so one lost acknowledgement does
   not stall unrelated work.
4. Replace per-packet backend `malloc`/`free` with a bounded reusable descriptor
   slab carrying slot generations.
5. Add fairness between prefill, decode, verify, cancellation, and release while
   preserving dependency order.

## P1: single-rail ring and one-switch networking

1. Match pipeline stage order to physical adjacency in direct-ring debug mode.
2. Bring up the one-switch 100 Gbit/s topology only after ring ordering and
   reconnect behavior are proven.
3. Pre-advertise persistent RDMA receive slots instead of negotiating every
   payload.
4. Add receive-slot generations so delayed writes cannot target reused memory.
5. Add bounded MR-cache eviction.
6. Poll completion queues in batches.
7. Add an autonomous network-progress worker.
8. Validate mapped-pinned and device-direct boundary buffers on the actual NIC,
   CUDA driver, peer-memory support, and IOMMU configuration.
9. Rotate independent B1 pipeline slots across QPs without reordering one
   request.
10. Keep dual-switch scheduling disabled until single-rail ownership, ordering,
    recovery, and performance are proven.

## P1: grouped-MoE and memory-bandwidth performance

1. Construct expert-major descriptors entirely on device.
2. Keep expert weights resident across every row tile in a sealed layer batch.
3. Overlap shared experts, route construction, expert-weight streaming,
   inter-stage transport, and route folding.
4. Retain real route-skew histograms and use them for queue/tile policy.
5. Keep token-centric small-row and grouped tensor-core large-row kernels as
   separately qualified modes.
6. Complete split-key grouped attention so KV reuse does not collapse B1 CTA
   parallelism.
7. Implement and qualify `sm_121a` TMA producer/consumer tensor schedules.

## Newly exposed or retained in Phase 6

1. Model-driver completion records do not yet contain request generation or
   transaction identity. The temporary side map therefore rejects concurrent
   in-flight completions sharing the same request/sequence/position tuple.
2. Completion callbacks can run synchronously during submit. Transaction state
   must be advanced only by the event-loop thread after submit returns.
3. Wake-pipe counters were a callback-thread data race and now use relaxed
   atomic operations.
4. A completion queue can be lossless only if final-event backpressure is
   checked before consuming its owner mapping.
5. Next-hop acceptance is not equivalent to all-rank reservation or global
   commit.
6. Stop-and-wait acknowledgements are useful for ring bring-up but impose
   head-of-line blocking and cannot be the final B256-B1024 protocol.
7. The DSPark draft result remains a serialized model-specific seam. Long term,
   draft payload identity belongs in the generic completion ABI.
