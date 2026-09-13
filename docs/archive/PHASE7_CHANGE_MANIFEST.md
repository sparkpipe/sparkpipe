# Phase 7 Change Manifest

Phase 7 is a bounded repair phase against the findings in the audit of
`sparkpipe-main-10.zip` (`685c92333fd294daa734cc2795a7aa76cbadb19bcdf9d56ae483e15a5fb7d0bb`).
It closes the foundational storage, allocation, required-KV, sliding-window,
numerical-build-policy, estimator, and source-package identity defects listed
below. It does not claim completion of the model executors, all-rank commit, or
Blackwell qualification.

## Closed P0 findings

- **P0-009 — NVMe table overlap.** The pending-queue header, pending entries,
  staging ownership records, and record table now occupy independently sized,
  aligned regions. Initialization rejects undersized ownership regions.
- **P0-010 — cancelled DMA ownership.** A staging slot remains unavailable
  until `poll_read` proves cancellation terminal. A pending cancellation cannot
  be reassigned to another block.
- **P0-011 — publish-before-commit.** Writes use an explicit
  `ReserveWrite -> CommitWrite/AbortWrite` lifecycle. Reserved and writing
  records are not readable or hash-indexed.
- **P0-012 — runtime-arena ABA.** Allocation returns a generation-carrying
  handle; release validates arena class, slot, address, usable bytes, and a
  64-bit per-slot generation.
- **P0-016 — missing required KV silently ignored.** Required KV access now
  validates view, sequence, logical page, mapping, and physical page, publishes
  one structured device error, and fails the CUDA stream. Dense, selected,
  sparse-score, summary, and refinement paths use required semantics.
- **P0-018 — unverifiable source ZIP.** Source inventory, manifest, checksums,
  deterministic archive generation, and archive verification are Git-independent
  and require exact payload-set equality.
- **P0-019 — overstated status and incomplete aggregate gates.** `STATUS.md`
  is receipt-bound, and the architecture gate executes the complete host test
  inventory rather than relying on a dry run.

## Closed P1 findings

- **P1-023 — missing sliding-window position producer.** MiMo now builds the
  exact contiguous window position list on device.
- **P1-025 — invalid GQA geometry accepted.** GQA validates query-head/KV-head
  ordering and exact divisibility before use.
- **P1-026 — blanket CUDA fast math.** Global `--use_fast_math` was removed
  from all CUDA build surfaces. Approximate intrinsics must be local and
  package-qualified.
- **P1-047 — GLM estimator literals outside the contract.** The estimator
  derives GLM dimensions and chunking from `model_contracts/glm52.json`.
- **P1-049 — source and raw qualification evidence mixed.** The source package
  excludes raw evidence and transient logs; qualification is a separately
  hashed external artifact tied to the source archive SHA-256.

## Closed P2 findings

- **P2-051 — rounded class ordering.** Arena class ordering compares canonical
  rounded class sizes.
- **P2-052 — unchecked total-byte cast.** Arena sizing checks overflow and
  `SIZE_MAX` before allocation.
- **P2-053 — bandwidth-times-step overflow.** NVMe deadline arithmetic uses
  saturating multiplication and overflow-safe ceiling division.
- **P2-054 — O(n) pending queue and wrapping FIFO order.** The pending queue is
  a bounded binary min-heap ordered by deadline and a 64-bit FIFO sequence.
- **P2-055 — implicit alignment and ownership.** NVMe table/staging capacities,
  alignment, and the single-owner-thread contract are explicit in the public
  API.

## Additional hardening completed

- The optional KV accessor was removed because it could conceal an interior
  missing page. Tail handling now requires an explicit context-length check.
- KV failure publication uses a first-writer protocol so consumers cannot see a
  final error code before its detail record is complete.
- Arena generation exhaustion fails closed instead of wrapping.
- The package verifier rejects duplicate JSON keys, unsafe paths, duplicate
  archive members, links, devices, FIFOs, compiled artifacts, nested archives,
  raw evidence, and payload drift.
- Targeted arena, NVMe, and topology tests pass under AddressSanitizer and
  UndefinedBehaviorSanitizer.

## Deliberately not closed in Phase 7

The remaining model, queue, CUDA, and network blockers are tracked in
`docs/PHASE7_REMAINING_WORK.md` and remain release blockers.
