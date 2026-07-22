# TPx support in the sparkpipe device drivers: feasibility and work plan

Assessment of adding tensor-parallel decode shapes to the driver stack, for a
hybrid topology that selects parallelism by batch size: tensor parallel for
small batches where latency dominates, the existing PP pipeline for large
batches where the expert-queue batch plane amortizes weight movement, and mixed
TPxPP shapes between. All model constants below are read from the driver
headers, not assumed: hidden 6144, 78 layers of which 3 dense, 64 heads, MLA
latent 512 plus rope 64, MoE top-8 of 256 experts with intermediate 2048, dense
intermediate 12288.

## Why TP wins small batch and PP wins large batch

Decode is weight-bandwidth bound. PP13 walks one token through the whole model
serially, about 170 ms per token at the calibrated 174 GB/s effective; TP13
reads each layer's weights on thirteen ranks in parallel, about 13 ms of
compute plus collective time, a roughly 12.7x latency win at batch one. The
advantage inverts with batch because compute stays flat, the same weights are
read once and applied to every row, while all-reduce traffic grows linearly
with rows: at the measured activation size the collective traffic passes the
13 ms compute floor between batch eight and sixteen. Large batch belongs to the
pipeline and the expert queue; the crossover region is where the mixed shapes
live. This matches the proposed bucketing of TP-wide for one to sixteen,
mixed shapes for thirty-two to sixty-four, and PP for one twenty-eight up.

## Finding one: the transport is the blocker, and it is hardware-gated

The driver has no collective primitives. The hidden-state transport is a
pluggable send interface with a point-to-point persistent-ring implementation,
and the wire is TCP at about 29 us per hop. A token under TP13 needs 156
all-reduces (two per layer); even with recursive-doubling at eight steps per
collective, 29 us per step is 36 ms of pure latency per token, which makes TP
slower than the pipeline it is meant to beat. TP is therefore gated on the new
all-to-all switch fabric plus a low-latency path: the latency budget is about
2 us per step to keep collective latency near 2.5 ms against the 13 ms compute,
and 1 us to make it negligible. That means kernel-bypass or RDMA-class
networking, not the current socket path. The all-to-all topology is also what
enables the log-step recursive-doubling collective in the first place; on the
ring, all-reduce is linear in rank count and the latency math is far worse.
This transport work is the single largest item and cannot be validated in
software alone.

## Finding two: divisibility rules out TP13 and TP6 as literal shapes

Head-sharded attention needs the head count divisible by the TP degree, and
column-parallel MLP needs the intermediate divisible. With 64 heads, hidden
6144, MoE intermediate 2048, and dense intermediate 12288, the clean degrees
are TP2, TP4, and TP8. TP3, TP6, TP12, and TP13 all fail at least one
dimension (64 heads is not divisible by any of them; 13 divides none of the
four dimensions). So the proposed TP6xPP2 shape is not clean, and TP13 as
literal tensor parallel needs uneven shards or padding, both of which
complicate every kernel and the fail-closed binding contracts.

Two good alternatives preserve the intent. For the mid-batch shapes, TP4xPP3
is clean and uses twelve of thirteen ranks, and TP2xPP6 likewise. For the
thirteen-wide small-batch shape, expert parallelism fits the model better than
tensor parallelism: experts are independent, so 256 experts split unevenly
across thirteen ranks (twenty on some, nineteen on others) with no kernel
changes to the grouped expert path, while attention is small enough to
replicate outright, about 6 GB of FP8 attention weight per rank for all layers.
EP13 with replicated attention parallelizes the expert weight read the same way
TP would, uses all-to-all token dispatch (which the new fabric serves well)
instead of all-reduce, and sidesteps the divisibility problem entirely. Batch
one activates only the routed top-8 experts per layer, so per-layer expert read
is one expert per active rank, and the estimated latency lands in the same
low-teens-of-milliseconds band as ideal TP13.

## Finding three: storage is identical, KV capacity is not, and shape
switching costs a reload

Model storage per rank is the same 56 GB share under any of these shapes; that
part of the proposal is exactly right. KV is the asymmetry: under PP a rank
holds latent KV only for its six layers, 3456 bytes per token, while any
TP-wide or EP-wide rank runs every layer and must hold the full-depth latent,
44928 bytes per token, thirteen times less capacity from the same pool. MLA is
what makes this affordable at all; the 29 GiB pool still holds about 693K
tokens full-depth, comfortable for sixteen sequences at 4K context, and far
short of the 4.2M tokens that 1024 sequences need, which is consistent with
large batch staying on the pipeline. The KV pool and JIT paging need a
per-shape capacity configuration but no mechanical change.

The sharper operational cost is that a node's weight shard is shape-specific: a
PP13 shard is six full-width layers, a TP4xPP3 shard is twenty-six
quarter-width layers. Switching a node between shapes means loading a
different 56 GB image, about nine to ten seconds from NVMe at the current
rate. Different batch sizes having different optimal shapes therefore implies
one of: partitioning the fleet by shape, accepting a roughly ten-second
reconfiguration when the workload mix shifts, or storing both slicings on NVMe
(the space exists) and hot-swapping. This belongs in the serving-engine
scheduler as an explicit policy, not an implicit assumption.

## Work plan and effort

The driver structure is favorable in the places that matter: pipeline slicing
is already parameterized by first layer and layer count, every GEMM takes its
dimensions at runtime through the prebound plan machinery which re-prepares on
change, the attention grid is already (sequences, head groups) so head sharding
is a parameter, and the grouped expert kernels are unchanged under EP. The work
items, in dependency order, with honest sizes:

Collective transport: recursive-doubling all-reduce and all-to-all
dispatch-combine over the new fabric behind the existing pluggable transport
interface, with a latency target of at most 2 us per step. Two to four weeks
including fabric bring-up, and gated on the switch hardware existing.

Weight sharder: extend the stagepack to slice within layers, column-parallel
up projections, row-parallel down projections, head-range attention slices,
per-TP-rank images with the pack-to-kernel hash contracts extended to cover the
shard geometry so a mismatched shard fails closed. About one week.

Kernel plumbing: thread TP rank and degree through the stage descriptors, set
head groups to heads over TP, extend the fail-closed READY contract. Three to
five days given the runtime-dimension design.

MLA KV under full-depth residency: per-shape pool configuration and capacity
accounting; the pool mechanics are untouched. Two to three days.

EP13 path: expert-to-rank assignment tables in the stagepack, all-to-all
dispatch and combine calls in the MoE step, replicated attention images.
About one week, mostly reusing the grouped expert kernels as-is.

Hybrid orchestration: shape selection by batch bucket in the serving engine,
shape-aware admission, and the reconfiguration policy from finding three. One
to two weeks depending on whether hot-swap is in scope.

Total driver-side effort is roughly four to seven weeks, but the schedule is
dominated by the transport and the fabric. The right sequencing is TP2 first
on the direct pairwise links: clean divisibility, a two-node collective is
trivial, and it exercises the sharder, the collective interface, the KV
configuration, and the binding contracts end to end while the switches are
being installed. Then TP4xPP3 on the fabric, then the thirteen-wide EP shape.

## What must be measured before betting the topology

Switch port-to-port latency under load, since the entire TP budget hinges on
the 1 to 2 us per step number; the recursive-doubling all-reduce time for a
12 KB BF16 vector across thirteen nodes; whether FP8 activations survive
all-reduce numerically, which halves collective bytes and pushes the TP-to-PP
crossover from about batch sixteen toward batch thirty-two; and the actual
crossover batch on the real fabric. The compute-side numbers inherit the known
up-to-1.7x optimism on the per-rank stage anchor until the ring measures the
optimized stage.
