# Phase 10 Change Manifest

## Hardware truth contracts

- Added and consolidated the 33-question registry and fail-closed assumption bindings.
- Added exact workload profiles for K3, GLM 5.2, Qwen 3.6 27B, DSV4 Flash, and DSV4 Pro.
- Added deterministic ring and one-switch qualification-plan generation.
- Added deterministic per-node runner-configuration generation and checked-in topology templates.

## Probes and provider ABIs

- Added GB10 CUDA and NVMe characterization probes.
- Added integrity-checked PMTU probing.
- Consolidated exact production model-kernel, transport, and topology provider ABIs.
- Added immutable local/peer provider identities and path-specific transport validation.

## Execution and policy

- Added exact cell identity, lock, timeout, resume, and failure receipts.
- Added runner configuration SHA-256 verification.
- Added per-node preflight with executable/provider artifact inventories.
- Fixed policy grouping so different nodes and peers are never collapsed into one best result.
- Added derived mapped-host/copy and CPU-contention policies.
- Added question closure from registry through source consumer.

## Build and validation

- Added hardware tools and host handoff tests to the Makefile.
- Added the hardware handoff gate to `tools/gates.sh`.
- Added exact CUDA 13 `compute_121a` and `sm_121a` compile coverage for the hardware CUDA probes.
- Added complete operational handoff documentation.
