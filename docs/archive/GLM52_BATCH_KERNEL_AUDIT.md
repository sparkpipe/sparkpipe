# GLM-5.2 batch-size kernel audit

Question raised from the other developer's finding on a different model: a kernel
correct and acceptable at batch size one was reused unchanged for larger batches,
where a scalar one-row-per-block shape leaves the tensor cores idle and re-reads
the weight per row instead of tiling. This audit checks whether GLM-5.2's decode
stage does the same. It does not. The batch paths are tiled tensor-core kernels
and the scalar kernels are batch-parallel fallbacks that production refuses to
run on.

## Linear and QKVO projections

The dense projections do not use a single scalar kernel across batch sizes. The
primary path is a prebound plan, `MaybeLaunchPreboundLinearPlan`, which binds a
cublasLt scaled-GEMM backend for FP8 and a WMMA batch kernel for BF16. The FP8
path routes through `LaunchFp8E4m3ActivationWeightLinearScaledGemmBackend`, a
tensor-core scaled GEMM. The BF16 WMMA batch kernel,
`SupportedQuantizedBf16WmmaLinearBatchKernel`, launches a two-dimensional grid of
`output_dimension / 64` by `active_sequence_count / BATCH_ROWS`, so it tiles both
the output and the row dimension and processes `BATCH_ROWS` rows per tile through
tensor cores rather than one row per block. The `active_sequence_count`, the batch
dimension M, parameterizes the layout and is re-prepared when it changes, which is
the fix for the earlier M-equal-one plan pathology and is present and correct.

The scalar `Bf16LinearKernel` and `Fp8LinearKernel` still exist and are still
launched, but only as a fallback after `MaybeLaunchPreboundLinearPlan` reports it
did not fire, and even then their grid is `dim3(output_dimension,
active_sequence_count, 1)`, one block per output-and-row, which is still parallel
across the batch rather than a serialized per-row loop. The scalar path is a
correctness fallback, not the batch production path.

The B1024 integration contract closes the loop: the production resident refuses
READY state unless every FP8 linear plan binds the native scaled-GEMM backend and
all routed layers bind the grouped FlashInfer FP8 MoE backend, with no
BF16-WMMA fallback in that path. Production cannot silently run the scalar
fallback at batch; it fails closed instead.

## Attention

The absorbed MLA attention kernel is a batched flash-attention-style kernel, not
a per-row loop. It takes per-row `row_sequence_indices`, `block_table`,
`context_lengths`, and sparse token indices, tiles the latent cache into shared
memory, runs an online-softmax accumulation, and reads the FP8 or BF16 MLA cache.
It launches a grid of `(active_sequence_count, head_groups, 1)`, so it scales
across the batch on the grid x dimension with head groups on y. Batch size enters
the grid, not a loop.

## MoE experts

The routed expert path is grouped execution, not per-row. The packed-route
kernels reset, count, prefix-sum, and fill a per-expert routing table so that all
rows routed to one expert are processed as a group, which is the arithmetic-
intensity win the batch plane depends on. The production contract requires the
grouped FP8 MoE backend to bind before READY.

## Verdict

GLM-5.2's decode stage does not carry the batch-size defect. The projections,
attention, and expert paths are all batch-tiled or grouped, batch size enters the
launch geometry rather than a serial loop, and the scalar kernels are
batch-parallel fallbacks that the production fail-closed contract forbids at
batch. The regime-aware split the finding recommends, scalar for a single row and
tensor-core tiling at batch, is already the design: the single-row-friendly
scalar kernel exists as a fallback while the primary path is the tiled tensor-core
plan, and production binds the tiled plan or refuses to run.

One item worth a ring check rather than a code change: confirm on silicon that the
prebound tensor-core plan actually binds for every projection at every batch
bucket used, so the scalar fallback never silently carries batch traffic. This is
a plan-binding counter to read during the packet-timing session, the same place
the M-equal-one verify gate is measured, not a defect in the code as written.
