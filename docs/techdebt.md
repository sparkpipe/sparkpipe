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

## Dynamic Batch Specialization

The common scheduler must form execution microbatches from compatible queued
requests and select the smallest validated power-of-two specialization that
fits the effective row count: B1, B2, B4, B8, B16, B32, B64, B128, B256,
B512, or B1024. The caller submits requests, priorities, and deadlines; it must
not choose a batch width or reproduce model-specific packing policy. Effective
rows means decode lanes for ordinary decode, actual packed rows for prefill,
and all verification rows for speculative decode, not merely the nominal
request count.

The existing common variant recipes correctly build every bucket from one
source tree, but DSV4 TP execution still publishes batch-specific module
identities and requires `rows == SPARK_BATCH_BUCKET`. Treat that exact-width
release shape as transitional debt. Production serving must expose one logical
model driver and one resident adapter/process. That resident may resolve
multiple AOT code objects and prewarm separate kernels or CUDA graphs per
bucket, but it must retain them together and switch at dispatch. Changing
bucket must not reload the driver, adapter, model files, weights, KV pages, or
resident process, and sequence/KV identity must survive every switch.

When an exact-width graph is fastest, inactive rows in the selected ceiling
must be represented explicitly and must not add material work proportional to
the padding, mutate KV, participate in routing, or emit tokens. Resource
pressure or a bounded latency deadline may force a smaller ready batch; the
scheduler must not wait indefinitely for the largest bucket. Requests above
B1024 fail closed or are partitioned by common scheduler policy rather than by
a model-specific caller.

Qualification must fuzz mixed arrivals, priorities, prompt lengths, shared
prefixes, cache pressure, cancellations, and speculative row multipliers from
one through more than 1024 queued requests. Receipts must prove smallest-fit
bucket selection at every boundary, starvation bounds, deterministic parity
across bucket switches, unchanged residency identities, no reloads, no work
for inactive rows, and measured latency and throughput for every B1--B1024
specialization.

## Payload-Aware Collectives

Collective choice is runtime policy, not model-package identity. Select it from
TP degree, actual payload bytes, datatype, and a measured hardware-topology
profile. Small messages should use the validated latency algorithm, expected
to be recursive doubling for TP4; intermediate messages should benchmark
recursive doubling against recursive halving/doubling; large messages should
use the validated bandwidth algorithm, expected to be counter-rotating split
rings when both physical links are available. Crossover thresholds must come
from hardware measurements rather than nominal batch labels.

The payload calculation must include all effective rows, so a nominal B8 with
eight speculative verification rows per request is treated as approximately
64 rows rather than B8. Switching collective algorithms must not reload or
redistribute model files, weights, or KV. Implement and qualify this selector
after the direct-pair network change; until then, keep collective time visible
in profiles but do not tune thresholds against the obsolete topology.

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
