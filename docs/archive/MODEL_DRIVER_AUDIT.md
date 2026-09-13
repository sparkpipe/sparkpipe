# Non-GLM model-driver audit

This audit covers the DSV4, K3, MiMo 2.5, and Qwen 3.6 resident-stage modules. It evaluates host contracts, ownership, failure behavior, package metadata, and publication gates. It does **not** claim GPU numerical correctness, latency, throughput, or production readiness; `nvcc`, target hardware, live weights, and retained GPU receipts were not available in this environment.

## Cross-family corrections

All four drivers now:

- initialize through firmware-module ABI 4 and validate descriptor sizes and reserved fields;
- expose model-driver ABI 7 descriptors and versioned create requests;
- reject unknown frame and admission flags;
- reject the generic dispatch-ticket flag and never use `driver_dispatch_slot` as a persistent model lane;
- validate exact stage-dependent host buffers;
- claim execution slots and model lanes atomically;
- release only lanes that the current request successfully claimed;
- enforce model-specific sequence continuity before reusing resident state;
- return `SPARK_STATUS_MODULE_NOT_VALIDATED` unless an explicit controlled-bring-up switch is set;
- use synchronous `submit_return` completion in their package descriptions;
- state `NOT_MEASURED`, `production_ready: false`, no fallback, and no runtime backend selection;
- publish only through common rules requiring an executable GPU validator and exact configuration identity.

The lane-release correction is material. Earlier DSV4, MiMo, and Qwen error paths could release a lane after another lane in the same request failed to claim. That could unlock state owned by a different request. Each driver now tracks whether its lane set was successfully claimed before any release.

## DSV4

### Contract now enforced

- Decode frames and round-major prefill wavefronts.
- Exact row count, lane arrays, sequence IDs, and positions.
- Stage-position buffer ownership.
- Required hidden transport for non-edge stages.
- Real execution-slot availability at admission.
- Lane/sequence continuity and failure invalidation.

### Deliberately unavailable

- Causal bulk-prefill execution within one sequence. Cross-request prefill rows
  execute together as a CUDA wave rather than as serialized B1 submissions.
- MTP execution.
- CUDA graph path as a qualified feature.

Those modes fail closed rather than selecting an alternate implementation. Loaded MTP-related tensors do not by themselves establish a usable or qualified MTP program.

## K3

K3 had the largest host-contract cleanup.

### Corrected hazards

- Persistent model lanes are carried only by K3 frame-context views; the generic dispatch ticket is no longer treated as a lane.
- `row_capacity` now names padded activation capacity; it is no longer mislabeled as active sequence capacity.
- Host token buffers are required only on embedding and final-head stages. Intermediate stages use hidden transport without fake token buffers.
- Decode and prefill block tables validate every logical block required by the attention range, not merely the current block.
- Allocation and staging use the shared checked ledger and rollback rules.
- Stage count, index, first layer, and layer count are part of the publication configuration.
- The CPU reference remains a developer tool and cannot serve as the publication validator.

### Status

Decode and bounded prefill shapes are structurally represented, but GPU execution and correctness remain `NOT_MEASURED`. The published example is controlled bring-up only.

## MiMo 2.5

### Contract now enforced

- Decode only.
- Unique lane use and exact sequence continuity.
- Exact stage-position buffers and transport.
- MTP state is canonical: when MTP is not armed, depth must be zero and the draft-token pointer null.
- When MTP is armed on the final-head stage, the advertised depth and draft-token storage must match the configured MTP contract.

### Deliberately unavailable

Prefill is rejected. MTP is not presented as production-qualified merely because structures and weights exist.

## Qwen 3.6

Qwen had the broadest existing frame contract. The audit retains that breadth while making ownership fail-closed.

Validated structurally on the host side:

- decode versus bulk-prefill exclusivity;
- decode-batch arrays and unique lanes;
- prefill ranges;
- KV block-table descriptors and capacities;
- GDN snapshot/restore descriptors;
- speculative verification and MTP draft descriptors;
- stage-position buffers and hidden transport;
- lane ownership and continuity.

The KV provider, GDN snapshots, MTP drafting, prefill path, and device execution remain unqualified without GPU receipts.

## Shared tensor-parallel collective

The shared TCP reference collective was also a model-driver foot gun because a failed peer could block connection, handshake, or payload I/O indefinitely. Its earlier loopback test compounded the defect: a rank that failed creation skipped the final barrier, so a normal connection error became a permanent test hang.

The replacement contract uses ABI 2 and now enforces:

- a nonzero deployment-supplied collective identifier shared by every rank;
- reciprocal handshakes binding magic, ABI, degree, rank, and collective identifier;
- nonblocking sockets and absolute monotonic deadlines for connect, accept, handshake, send, and receive;
- operation headers binding collective, sequence, step, operation kind, sender rank, and element count before payload transfer;
- fail-closed teardown after any transport or protocol failure;
- no retained listening socket after the group is established;
- numeric IPv4 endpoints rather than silently pretending arbitrary host names are supported;
- exact configuration validation, including degree-one canonical state, reserved fields, and unused peer entries;
- disjoint value and scratch memory ranges for multi-rank collectives;
- destroy-safe failed creation and idempotent teardown;
- explicit serial-operation semantics and homogeneous native-F32 wire assumptions.

The retained test now guarantees both barriers are reached after partial creation failure, selects currently unused loopback port ranges, serializes concurrent instances, verifies failed-create destruction and repeated destruction, rejects overlapping value/scratch storage, exercises missing-peer timeout and mismatched element counts, advances the operation sequence across consecutive collectives, and repeatedly validates recursive doubling through TP degree 16. This remains a reference TCP implementation, not a performance qualification of the intended fabric path.

## Shared resident-stage support

`model-families/common/src/spark_stage_module_common.c` centralizes:

- overflow-checked device allocation;
- allocation accounting and rollback;
- bounded pack staging;
- strict environment parsing;
- atomic slot/lane ownership;
- admission and snapshot initialization;
- bounded destroy-time quiescence.

It remains outside neutral core because these are resident model-module facilities, not scheduler ABI concepts.

## Publication safety

`modules/resident_decode_stage_rules.mk` is the common publication path. A module cannot publish unless:

- `nvcc` exists;
- the target is exactly `sm_121a`;
- the stage pack is readable;
- `GPU_VALIDATOR` names an executable;
- the configuration hash is incorporated into the validation recipe and validator arguments.

The module library independently hashes the validator executable and includes that identity in the immutable validation record. A CPU reference or source-level assertion cannot satisfy this gate.

## Remaining limitations

1. No CUDA compiler or target GPU was available, so CUDA translation units, linking, execution, numerical comparisons, and performance were not qualified.
2. Runtime configuration is still environment-derived. Publication binds a complete configuration hash, but production should pass immutable configuration JSON directly to the module.
3. Destroy has a `void` ABI. If quiescence cannot be proven within the bounded wait, the implementation preserves live state rather than freeing it unsafely; deployment must quiesce before destroy.
4. Host contract tests cannot prove kernel bounds, numerical equivalence, stream ordering, transport correctness, or actual completion timing.
5. Each family needs retained, exact-hardware receipts before any `MEASURED` or production-ready claim.
