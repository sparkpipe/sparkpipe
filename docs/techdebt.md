# Current Technical Debt

This file lists only work that remains in the active generic stack. Completed
or deleted architectures belong in Git history, not in the live backlog.

## Package Provenance

Stage-pack headers record the model revision, model contract SHA-256, source
configuration SHA-256, and pack recipe SHA-256. They do not yet record the
upstream implementation commit used to interpret source tensors. Add that
field at the next stage-pack format revision; do not regenerate current packs
solely for this metadata addition.

## Deployment Specifications

The common deployment generator is model-neutral and DSV4 has a checked compact
specification. Add generated compact specifications for every released GLM
codec package when those package release IDs and rank-local pack paths are
final. Do not choose one codec as a default or generate runtime alternatives in
one deployment.

## Pack Environment

Model packers have explicit Python/CUDA dependencies. Publish and synchronize a
small pack-environment manifest across all Sparks so per-rank packing does not
depend on whichever Python environment happens to be active. The packer must
continue to require a CUDA device and exact package codec.

## Batch Client Shutdown

`sparkpipe_model_batch` preserves and flushes incremental token events. Add a
bounded cancellation/drain path for terminal local I/O failures so an active
batch can release resident sequence slots before the client exits. Normal
completion and fail-closed model errors are covered today.

## Qualification Receipts

Automate collection of one evidence bundle containing the merged commit,
release generation, package/driver/pack hashes, all-rank ready identities,
full token JSONL, accuracy result, throughput result, and drained queue state.
The individual gates exist; producing one immutable cross-gate receipt remains
manual.

## Live Model Qualification

Source contracts, host tests, and exact SM121 compilation are necessary but do
not establish model accuracy. GLM 5.2 and DSV4 GA Flash still require retained
live serial-versus-batch parity, accuracy, and performance receipts from clean
merged-main releases on the 13-Spark ring.
