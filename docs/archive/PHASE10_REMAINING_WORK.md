# Phase 10 Remaining Work

## Requires CUDA 13 but not physical hardware

1. Run the complete `compute_121a`/`sm_121a` compile gate.
2. Repair every compile, device-link, architecture, register, spill, local-memory, or shared-memory failure.
3. Build exact production provider shared objects.
4. Bind provider artifact hashes and `ptxas` receipts to qualification results.

## Requires the Sparks

1. Run GB10 numerical vectors for BF16, FP8 E4M3, MXFP4 E2M1, UE4M3, and UE8M0.
2. Run Compute Sanitizer and race checks.
3. Validate CUDA Graph capture and replay.
4. Execute the complete ring plan with physical stage order matching ring adjacency.
5. Validate reconnect, timeout, cancellation, duplicate suppression, and completion ownership under injected failures.
6. Validate mapped-host and GPUDirect RDMA with the real NIC, IOMMU, kernel, CUDA driver, and visibility semantics.
7. Run the one-switch plan only after the ring closes.
8. Compile one policy per topology and compare them; never copy values between topologies.

## Model execution blockers outside the hardware suite

1. Complete one shipping K3 executor joining KDA, Gated MLA, Stable LatentMoE, exact replay, incremental AttnRes state, EAGLE-3 drafting, output head, and completion.
2. Complete Qwen 3.6 Gated DeltaNet gate projections, full-attention output gate, MRoPE, BF16 pack, and resident executor.
3. Complete independent DSV4 Flash and Pro packs and executors.
4. Produce one mandatory no-undefined-symbol GLM 5.2 final CUDA artifact.
5. Remove remaining semantic GLM assumptions from the common shipping stage.

## Distributed runtime blockers

1. Implement all-rank `PREPARE → EXECUTE → COMMIT/CANCEL`.
2. Add application-level final-event acknowledgement and reconnect deduplication.
3. Add restart epochs or durable terminal tombstones.
4. Remove blocking resident admission from the rank event loop.
5. Replace stop-and-wait work forwarding with a bounded selective-acknowledgement window after correctness bring-up.

## Future dual rail

Dual-switch mode remains disabled until single-rail correctness and recovery close. Future work must define independent rail ownership, ordering, failure domains, slot assignment, and policy measurements before enabling both MikroTik fabrics.
