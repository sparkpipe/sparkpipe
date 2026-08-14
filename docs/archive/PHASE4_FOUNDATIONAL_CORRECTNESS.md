# Phase 4 — Foundational CUDA and Model Correctness

## Scope

Phase 4 closes source-proven failures that would invalidate every later performance measurement. It does not claim CUDA compilation or GPU execution.

## Shared scale ABI

The shared GEMM now distinguishes unscaled, FP32, UE4M3, and UE8M0 scale planes. A scale descriptor includes element capacity and explicit expert/group, row-group, and K-group strides. The launch path validates those capacities before dispatch.

This prevents the earlier cross-model failure in which one-byte activation scales were cast to `const float *` and indexed without the row, expert, output, or K-group dimensions.

## TMA launch contract

Tensor-map descriptors are passed by value to the CUDA kernel. The producer contract elects one thread for the complete CTA, and the mbarrier expected-arrival count matches that single producer. Shared-memory opt-in state is device-specific and serialized.

## Race-free sub-byte writes

The old pair writer used overlapping 32-bit read-modify-write operations. Adjacent CUDA threads could lose each other's packed codes. The replacement gives one thread exactly eight codes and one disjoint byte range:

```text
4-bit: 8 codes -> 4 bytes
6-bit: 8 codes -> 6 bytes
7-bit: 8 codes -> 7 bytes
8-bit: 8 codes -> 8 bytes
```

The source-contract test independently round-trips all four widths.

## K3 exact accepted-prefix replay

Verification now writes the transformed FP32 retention and write-gate values directly into replay slabs. The recurrent verification kernel consumes those same slabs. Accepted-prefix folding therefore does not call the bounded-decay or sigmoid transforms a second time.

The fold also allocates the complete FP32 recurrent state tile in dynamic shared memory and validates replay pointers, layer ranges, sequence capacity, and slab row capacity before launching.

The host execution harness checks:

```text
committed one-token run
        ==
noncommitting two-token verification
+ fold of one accepted token
```

for both the recurrent state and all three causal-convolution windows.

## Model precision contracts

- K3: MXFP4 routed-expert weights with BF16 activations and BF16 non-experts.
- GLM 5.2: FP8 E4M3 routed-expert weights with BF16 activations and BF16 non-experts.
- Qwen 3.6 27B: BF16.
- DSV4 Flash/Pro: separate generated geometry and checkpoint-native mixed precision.

Package metadata forbids hidden fallback and runtime precision replacement.

## Build-system and audit corrections

- Model-specific Python pack logic was moved out of common runtime paths.
- The model-driver contract accepts explicit template instantiation or a templated launch binding, while still rejecting kernels defined inside model unity files.
- The authored-code-size metric excludes generated build output, making it stable before and after `make test`.
- Stale GLM model-description and firmware tests were rewritten against the current precision contract.
- DSV4 Pro gained an explicit compile surface so CUDA CI can no longer omit it silently.

## Validation boundary

Host tests prove source contracts, host control flow, deterministic reference behavior, and build integrity. They do not prove:

- CUDA 13 syntax and template instantiation;
- `compute_121a` PTX generation;
- `sm_121a` `ptxas` acceptance or device linking;
- register/spill/shared-memory resource use;
- GPU numerical equivalence;
- race freedom or out-of-bounds safety;
- CUDA Graph behavior;
- transport ordering;
- latency or throughput.
