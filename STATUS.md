# SparkPipe Status

Current live DSV4 decode measurements are tracked in
[`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md), including the exact checkpoint,
runtime boundary, source and driver identities, receipt hashes, and reproduction
command for the latest accepted milestone.

SparkPipe status has two maintained sources:

- [`TECHDEBT.md`](TECHDEBT.md) lists unfinished work against the architecture.
- [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md) records measurements,
  projections, and target gates.

The intended system is defined by [`ARCHITECTURE.md`](ARCHITECTURE.md) and
[`SPEC.md`](SPEC.md). Those documents do not carry phase flags, implementation
chronology, handoff notes, or transient benchmark status.

Production readiness is evaluated per exact model checkpoint and deployment.
It requires matching source and package identity, host and CUDA gates, physical
route evidence, numerical correctness, end-to-end service results, and retained
receipts from a clean merged-main release.
