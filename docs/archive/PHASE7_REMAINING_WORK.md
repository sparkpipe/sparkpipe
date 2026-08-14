# Phase 7 Remaining Work

## P0 — model execution closure

1. Connect the K3 pack, binder, resident stage, KDA/MLA execution, exact speculative replay, Stable LatentMoE, incremental AttnRes state, draft model, output head, and completion into one shipping package.
2. Implement Qwen 3.6 Gated DeltaNet forget/write gates, full-attention output gating, multimodal MRoPE, a versioned BF16 pack, and an executing resident stage.
3. Complete independent DSV4 Flash and Pro executors, including their distinct layer schedules, sliding attention, CSA/HCA, hyper-connections, bootstrap hash-MoE, MTP, packs, and exact cache-arena integration.
4. Replace GLM-specific common-stage dispatch with a model-neutral driver boundary.
5. Produce one mandatory final-link artifact per model with no undefined symbols and hashes for every linked input.

## P0 — distributed correctness

1. Implement all-rank `PREPARE -> RESERVE -> EXECUTE -> COMMIT/CANCEL` rather than next-hop acceptance followed by local execution.
2. Add application-level final-event acknowledgement, reconnect replay deduplication, and durable tombstones or restart epochs.
3. Remove the blocking resident-admission wait from the rank event loop.
4. Carry transaction, request-generation, step-generation, and chunk identity natively through resident submission and completion.
5. Make missing required state terminal for the owning request across every model path.

## P0 — exact CUDA qualification

1. Compile every owned CUDA translation unit with CUDA 13 for `compute_121a`.
2. Assemble and device-link exact `sm_121a` artifacts.
3. Retain `ptxas` register, spill, local-memory, and shared-memory reports.
4. Inspect final artifacts with `cuobjdump`.
5. Run independent BF16, FP8 E4M3, MXFP4, UE4M3, and UE8M0 numerical vectors.
6. Run bounds, race, graph, recurrent replay, and transport checks on Spark hardware.

## P1 — memory-bandwidth and compute scheduling

1. Replace per-query-head GQA cache walks with split-key grouped-query attention that stages each K/V tile once and exactly merges partition-local online-softmax state.
2. Replace host-staged TP with a GPU-resident device-direct collective.
3. Use native Blackwell block-scaled MMA/TMA variants where measured faster, while retaining low-M token-centric weight streaming.
4. Build grouped-MoE prefixes and expert-major descriptors on device without host synchronization or full activation rematerialization.
5. Generate K3 tensor schema, packer, sharder, shard table, and binder from one authoritative checkpoint-derived schema.
6. Verify KDA convolution tap orientation against an independent reference.
7. Price K3 recurrent state and AttnRes state in admission.
8. Resolve K3 MTP/EAGLE-3 state and feature-fusion contracts.
9. Implement the declared KV quantization block and GLM DSA index normalization.
10. Continue removing semantic GLM assumptions from common headers and runtime code.

## P1 — single-rail ring and one-switch network

1. Match PP stage order to physical adjacency during direct-ring bring-up.
2. Pre-advertise RDMA receive slots and protect each slot with a generation.
3. Replace stop-and-wait forwarding with a bounded selective-acknowledgement window after correctness bring-up.
4. Add autonomous progress and batched completion processing.
5. Derive device and endpoint selection from topology rather than a fixed 13-node naming scheme.
6. Replace raw host wire structs with versioned endian-stable descriptors and unpredictable connection epochs.
7. Remove one control round trip per payload.
8. Measure and reduce per-send CUDA event/callback cost at B1.
9. Keep dual-switch mode disabled until single-rail ordering, reconnect, cancellation, and failure recovery are proven.

## Newly exposed integration work

- `LmKvView` now requires physical-page capacity and an error record. Existing opaque model buffer contracts must explicitly populate those fields before CUDA execution.
- Sparse summary/refinement kernels now require context lengths so an incomplete final block can be skipped without treating an interior missing page as optional.
- Arena allocation generations are now 64-bit because a 32-bit per-slot generation can wrap in a long-lived high-rate daemon.
- Source-package verification intentionally excludes raw qualification evidence; release tooling must publish a separately hashed evidence artifact.
