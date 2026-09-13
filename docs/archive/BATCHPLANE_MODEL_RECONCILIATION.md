# Batch-plane model reconciliation with the 12-Spark GB10 performance model

This reconciles the batch-plane estimator and its host components against the
production `GLM52_12X_SPARK_PERFORMANCE_MODEL.md` and the
`GLM52_B1024_JIT_KV_INTEGRATION.md` handoff. Both are bandwidth-roofline models
on GB10; they describe different operating points, and stating the mapping keeps
the two design tracks from talking past each other. Neither has measured GB10
throughput; both are planning models pending the ring.

## The apparent hundredfold throughput gap is the expert-queue amortization

The 12-node model quotes 80 to 220 aggregate tokens per second; the batch-plane
sim quotes on the order of sixteen thousand committed tokens per second. These
are not the same quantity. The 12-node bands are per-token weight movement: a
single forward pass streams the active weights, roughly 25 to 35 GB per token at
FP8 or NVFP4, and the twelve-node aggregate roofline of 3.276 TB/s over 35 GB is
about 93 tokens per second. The batch-plane number is the amortized regime the
expert queue exists to create: when many tokens queue behind one expert load,
each 37.75 MB load serves on the order of a hundred rows, so the per-token weight
movement falls by that factor and throughput rises by it. Ninety-three tokens per
second times a hundredfold queue amortization is the same order as the batch
plane's committed figure. The two models are consistent exactly to the degree
that expert-queue amortization is real, which is the central unproven claim both
tracks depend on and which the ring measures directly through the routed-queue
depth.

## The per-rank stage anchors disagree and mine is the optimistic one

The 12-node model reports measured bring-up of 4.6 ms per checked layer on the
backend path, which is 27.6 ms for a six-layer rank stage. The batch-plane
calibration anchors to a 16.1 ms single-row rank stage. These conflict by about
1.7x in the direction that makes the batch-plane numbers optimistic. The likely
reconciliation is that 4.6 ms per layer is the current scalar correctness-kernel
bring-up, which the 12-node model itself labels as pre-optimization evidence and
not a target, while the 16.1 ms anchor is meant to represent an optimized
projection plan. Until the ring measures an optimized full-stage decode, the
honest posture is that the batch-plane throughput could be up to 1.7x high on the
stage-time axis alone, independent of the amortization question. This is now the
first constant to retire, ahead of the coverage and acceptance measurements.

## Production constants that confirm the host components

The B1024 integration handoff matches the batch-plane component constants: a KV
pool of 65,536 blocks per stage at 64 tokens per block, about 29 GiB, with NVMe
backing at 1,048,576 blocks per stage, and stage-local grouped FP8 experts with
no cross-node expert parallel. The JIT KV pool, the 64-token fragment, and the
per-rank residency assumption were built to these shapes and agree with them. MTP
at seven rows per lane matches the dspark ladder depth the arbitration work
assumed.

## Production constraints the batch-plane model was missing

Three constraints from the handoff tighten the model.

The GPU dispatch width is a hard 1,024 lanes per stage, separate from the
in-flight request capacity of 13,312. The sim admits sequences up to its own cap
and lets rows in flight grow with sequence count, but a decode kernel executes at
most 1,024 lanes at once; the B8 sequences must map onto those 1,024 physical
lanes per stage, so the throughput-bearing concurrency is bounded by lane width,
not by admitted sequence count. The sequence table's 16,384 admission capacity is
correct as backlog, but the firing-threshold controller should size against the
1,024-lane execution width, not the admitted population.

The KV budget is tight in a way the model glossed. Production runs about 4,089
context tokens plus seven MTP positions, exactly 64 blocks, and 4,096 context
crosses into a 65th block so only 1,008 lanes fit rather than 1,024. The
batch-plane model assumed 8,192 context freely; at the production 4,089 the pool
holds about 1,024 full sequences, which is why the pool size and the lane width
coincide. Attention byte terms in the model should use the production context,
which roughly halves them.

NVMe required is 370 to 429 GiB per node for the full 13,312-request backlog, so
JIT paging is mandatory, not optional, at production scale. This puts the JIT
pool and the dedup headroom work on the critical path rather than beside it.

## Dedup restated in production units

At the production 4,089-token context, 64 blocks per sequence, the 65,536-block
pool holds about 1,024 fully resident sequences, matching lane width. If 500
longmem sequences share a 1,500-token prefix, that is 23 shared blocks per
sequence; dedup collapses 500 copies to one and frees about 11,477 blocks, 5.06
GiB, which is roughly 180 additional resident lanes, an 18 percent concurrency
gain within the same physical pool with no quality change. This is the concrete
unconditional win in the units the production path uses: not tokens per second at
the current point, but resident lanes, which become tokens per second once the
plane is lane-bound or paging-bound.

## Boundary-rule compliance

The batch-plane components respect the driver boundary the 12-node model
mandates. The expert queue, JIT pool, dedup, and sequence table are firmware-side
structures that expose only admission, opaque slot identity, service and queue
estimates, residency match, and copy counters upward; none require SparkPipe to
learn layer partitions, expert placement, or KV ownership. The firing threshold,
fragment content hashing, and residency policy stay inside the firmware side of
that boundary.
