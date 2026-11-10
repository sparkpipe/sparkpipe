# SparkPipe Kernel Playbook

The lead dev maintains this playbook. Driver agents that tune kernels
update it when a technique or a hard constraint changes.

**Scope.** It covers the shared kernel code:

- `inference/kernels/`
- `model-families/common/include/sparkpipe/spark_lm_kernels.cuh`
- the kernels under `common/`

**Citations.** Every citation names a symbol, not a line number. Invariant
I48 forbids code comments, so this playbook is the only place the reasoning
behind a kernel is written down. Reasoning marked **(4187f90^)** comes from a
code comment that commit 4187f90 removed. Its numbers are the original
author's estimates and have not been re-measured.

**Numbers.** Measurements are not repeated here:

- `PERFORMANCE_STATUS.md` holds measurements, projections and target gates
  for every model.
- `docs/GLM5_NEXT_ROOFLINE.md` holds the GLM 5.3 Flash TP16 roofline and
  profiles.

---

# Part 1 — Kernel contract template

Fill in this block when you ask for kernel work. Every field is required
unless marked optional.

```
## Kernel contract

requestor:      <driver or subsystem>
op:             <one line: what the kernel computes, e.g. "grouped expert w1/w3
                (SiLU gate * up) GEMM", "RMSNorm + RoPE + FP8 QDQ fusion">
source:         <symbol of the caller / driver launch site, if any>

## Shapes
rows:           <batch rows, and whether B1/B8/B1024 decode or prefill>
input_dimension:<K, the contraction length; must state K-alignment>
output_dimension:<N for the GEMM; or per-op dims>
top_k:          <routed experts per token, 0 if dense>
experts:        <expert count, 0 if dense>
extra dims:     <heads, KV heads, head_dim, value_dim, rank, window, vocab, ...>

## dtypes
activation:     <BF16 | FP8 E4M3 (UE8M0, group __) | MXFP4 | ... ; stored bits>
weight:         <BF16 | FP8 E4M3 | MXFP4 E2M1 | NVFP4 | INT4/6/7/8; stored bits>
scale:          <none | UE8M0 | UE4M3 | F32-per-B128 | E8M0 ; group size>
accumulator:    <F32 | S32 (integer GEMM only)>
output:         <BF16 | FP8 codes+scale | ...>

## precision route (exact target string)
                e.g. cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.
                     expert_mxfp4.kv_bf16
rounding:       <where BF16 rounding is preserved; where fp32 is held>
bias/scale:     <which biases/scales fold in, and the exact fold order>
codec:          <ACTIVATION_CODEC or NONE; group size; QDQ in-place or not>

## memory / layout
strides:        <row strides, column offsets, group strides; explicit, no default>
layout:         <row/col-major, interleaved cell grid, expert-major, KV slot layout>
indirection:    <source_row_map / route_source_token? scale rows follow source? (yes/no)>
alignment:      <K % 128 == 0? width % TILE_N == 0? row pitch % 16 == 0?>

## target number
target:         <the number to beat, with unit: tok/s, us/layer, GB/s, ms/step>
baseline:       <measured baseline this must exceed, with its source>
constraints:    <static-shared <= 48 KB, occupancy, one-launch, DRY (no model name)>

## reference
ref:            <torch / flashinfer / vLLM / modeling file; exact identity>
tolerance:      <bit-exact, byte-exact, or RMS % + at what precision>
test gate:      <which test / validator / exact token hash must pass>
```

Rules for whoever implements the contract:

- **Name the precision route as a string.** The kernel-target rows of
  `PERFORMANCE_STATUS.md` give two examples:
  - DSV4 Flash: `cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16`
  - GLM 5.2: `cuda.sm121.glm52.resident_decode_stage.bf16.expert_fp8`
- **Misaligned K is refused, not rerouted.** Each of these returns
  `cudaErrorInvalidValue`:
  - `SparkLmValidateLinearContract`, when `row_count >= SPARK_LM_TILE` (16)
    and `input_dimension` is not a multiple of `SPARK_LM_TILE_K` (64).
  - `SparkLmValidateLinearContract` again, for NVFP4 when `input_dimension`
    is not a multiple of 16.
  - The native FP8 linear, when `input_dimension` is not a multiple of 128.

  State the alignment in the contract.
- **Row stride has no default.** `LmFusedResidualRmsNormKernel` takes an
  explicit `row_stride`. A stride that fills itself in is the "correct at
  rows==1, corrupt above" bug class.
- **Shared kernels never name a model** (`AGENTS.md`, `tests/test_dry_law.py`).
  Model facts go in tables and reach the kernel as parameters or template
  constants. `SparkLmSm121NativeDecodeShape` shows the pattern for a table of
  qualified shapes.

---

# Part 2 — Techniques

## 2.1 Hardware and ptxas facts

| Fact | Value | Source |
| --- | --- | --- |
| Target | `sm_121a` (GB10) | `TARGET` in `tests/test_ptx_capability_gate.py` |
| Memory | 128 GB unified, 273 GB/s per Spark | `README.md` |
| wgmma / tcgen05 | Not available on sm_121a; the gate requires these probes to fail | `tests/test_ptx_capability_gate.py` |
| NVFP4 block scale | `scale_vec::4X` with `ue4m3` assembles; `4X` with `ue8m0` is an sm_100 form and ptxas rejects it | `tests/test_ptx_capability_gate.py`, `LmMmaNvfp4` |
| Static shared memory | 48 KB per block (`LM_SMEM_STATIC_LIMIT`), out of 128 KB per SM (`LM_SMEM_SM_TOTAL`) | `inference/kernels/layout.cuh` |
| Linear-path compute | About 6.5 TFLOP/s, so compute binds only above about 30 FLOP per byte | Estimate (4187f90^, `project.cuh`) |

**The native gate.** `LM_SM121_NATIVE_COMPUTE_PTX` (`mma.cuh`) is 1 only when
`__CUDA_ARCH__ == 1210`. On any other architecture, `LmMmaMxf8Mxf4` and
`LmMmaMxf8Mxf8` execute `trap` rather than falling back to BF16 dequant,
which could be mistaken for the qualified route.

## 2.2 TMA, mbarrier and the staged pipeline (`tma.cuh`, `tile.cuh`, `layout.cuh`)

- **One thread issues TMA.** The producers `LmPipelineInitialise`,
  `LmPipelineProduceWeight` and `LmPipelineProduce` gate on
  `threadIdx.x == 0`. Using `elect.sync` would duplicate transactions and
  arrivals in a multi-warp CTA.
- **Declare exact byte totals.** `LmMbarrierArriveExpect` declares a byte
  total that must equal the sum of every box issued into the stage, or the
  barrier never flips. Byte totals come from `LmTileBytes`, never from
  literals.
- **Try-wait returns a predicate.** `LmMbarrierTryWait` returns a flag and
  `LmMbarrierWait` spins in C++. This avoids inline-asm label collisions when
  the function is inlined twice.
- **Tensor boxes.** Tensor-map loads (`LmTmaLoad2d`, `LmTmaLoad3d`) are
  bounds-checked and zero-fill out-of-range elements. In `LmTmaLoad3d`,
  coordinate 2 selects the expert, so one descriptor covers every expert.
- **Linear bulk copies.** `LmTmaLoadBulk1d` copies a linear address range
  with no hardware bounds check, so the caller must clamp the index. Do not
  use it for many small chunks: one CTA's bulk copies queue on the SM's TMA
  unit. The indirect A gather below implies about 20 ns per 16-byte copy,
  derived from kernel time rather than measured per copy.
- **Store side.** Stores use `LmTmaStore2d`, then `LmTmaStoreCommit`, then
  `LmTmaStoreWait<KEEP>`. The wait uses `wait_group.read`, which only
  guarantees that the staging buffer can be reused.
- **Fences.** `LmMbarrierInitFence` and `LmTmaStoreFence` issue
  `fence.proxy.async.shared::cta`. They order generic-proxy writes against
  async-proxy access.
- **Pipeline depth.** `LM_PIPELINE_STAGES` is 2 and
  `LM_PIPELINE_LOOKAHEAD` is 1. `LmPipelineStage` and `LmPipelineAhead` issue
  K tile `t + STAGES - 1` before consuming tile `t`.
  - **Why two stages (4187f90^).** By Little's Law, 218 GB/s at 400-600 ns
    of latency needs about 2.3 KB in flight per SM. Six stages of a 16-row
    NVFP4 tile take 110 KB, which fits one CTA per SM. Two stages take
    37 KB, which fits three.
  - The latency figure has never been measured.
- **Shared memory limits occupancy.** Going past `LM_SMEM_STATIC_LIMIT`
  needs dynamic shared memory and
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`.
  `LmPipelineSharedBytes` and `LmPipelineSharedBytesSplit` compute the
  budget.
- **Swizzle.** `LmSwizzleSpanFor` picks a 128-, 64- or 32-byte span from the
  row pitch. A span of 0 means the tile cannot be swizzled
  (`LmTileKIsSwizzleable`). The swizzle itself is applied only in
  `LmSwizzleChunk` and `LmSwizzledOffset`: the 16-byte chunk index is XORed
  with the row's 128-byte sector selector.
- **Interleaved weight staging.** In `LmPipelineProduceWeightInterleaved`,
  B is a grid of cells, not a `[neuron, k]` plane:
  - Each cell covers 16 neurons, per expert and per 128-element k tile.
  - A cell is 17 rows of 64 B: 16 payload rows, then one row of E8M0 group
    scales.
  - One rank-3 UINT8 tensor map covers the whole operand.
  - The K coordinate is in bytes, because the descriptors are UINT8.
- **Two-block A stage.** When `TILE_K` is 128 with interleaved B,
  `LmPipelineProduce` stages A as two boxes, one for k bytes 0-127 and one
  for 128-255.
- **Indirect A replaces the MoE gather.** `LmPipelineProduceIndirectA`
  stages packed row `p` straight from source row
  `LmRouteSourceRow(source_row_map, p)`, so no packed activation tensor is
  ever written.
  - Each thread copies its 16-byte chunks with `LmCopyAsync16`
    (`cp.async.cg`) and then calls `LmCopyAsyncArrive`
    (`cp.async.mbarrier.arrive.noinc`) once per stage. The producer thread
    expects only the weight bytes, and `LmGemmKernel` initialises the stage
    barriers with `1 + blockDim.x` arrivals on this path.
  - Before 2026-10-27 each chunk was its own `LmTmaLoadBulk1d`. A 64-row
    tile then issued 512 TMA copies per k-step, and the TMA unit serialised
    them. At 2048 rows the Solar Open 2 expert up projection took 5.7 ms
    on one GB10 (1.9 TFLOP/s, 60 GB/s). With `cp.async` it takes 1.4 ms
    (249 GB/s, near the memory roofline). K-EXAONE's shape (256 experts,
    hidden 6144, 128 rows per rank) went from 11.5 to 2.7 ms. The staged
    bytes and the MMA order are unchanged, so outputs are bit-identical.
    `build/expert_gemm_bench ROWS [gemm]` measures it. Its make target
    names the geometry in `EXPERT_GEMM_BENCH_SHAPE` (Solar's by default;
    `make EXPERT_GEMM_BENCH_SHAPE="-DBENCH_EXPERTS=256u -DBENCH_TOP_K=8u
    -DBENCH_HIDDEN=6144u -DBENCH_INTER=128u" build/expert_gemm_bench` for
    K-EXAONE).
- **Grouped skinny tasks cover real row batches.** `LmSkinnyGroupedKernel`
  scans `ceil(rows / ROWS)` per expert into a shared batch prefix
  (`LmSkinnyGroupedBatchScan`) and maps each task to one batch and one neuron
  group, so no launched task idles. Groups averaging more than eight rows
  take 16-row batches with two neurons per task, which reads each expert's
  weights once. A row's dot product depends only on `LANES`, which the input
  width fixes, so outputs do not depend on `ROWS` or `NPG`. On K-EXAONE's
  down projection at 512 rows this took the kernel from 8.9 to 3.8 ms, and
  at 256 rows from 4.9 to 2.8 ms, with identical digests.
  - Ragged tail rows are clamped to `row_base`.
  - A source row at or past `source_row_count` does not trap. The kernel
    reports `LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE` through
    `LmFrameErrorReport` and stages row 0 instead. The frame fails and the
    context survives (`tests/test_kernel_frame_error_source.py`).
- **Grouped tile scheduling.** `LmGroupOfTile` binary-searches the device
  tile prefix. `LmTotalTiles` reads the true total, so the host never
  launches empty groups on an average.

## 2.3 MMA atoms (`mma.cuh`)

- **Fragment mappings are tested.** Every mapping
  (`LmMmaAccumulator*`, `LmMma8Operand*`, `LmMma4Operand*`,
  `LmMma16Operand*`) is checked element by element, with a bijection check and
  a negative control, in `tests/test_mma_fragment_mapping.c`.
- **8-bit atoms, m16n8k32** (`LmMmaE4m3`, `LmMmaMxf8Mxf4`, `LmMmaMxf8Mxf8`).
  Each register holds four contiguous bytes, so operands use plain aligned
  32-bit loads, not `ldmatrix`.
- **4-bit atoms, m16n8k64.**
  - `LmMmaNvfp4`: `ue4m3` scales, one per 16 elements (`scale_vec::4X`).
  - `LmMmaMxfp4`: `ue8m0` scales, one per 32 elements (`scale_vec::2X`).
- **BF16 atom, m16n8k16** (`LmMmaBf16`). Each register holds two consecutive
  k values.
- **Free BF16 dequant.** `LmCodeToBf16Bits` ORs a code of at most 7 bits
  into the mantissa of `0x4300` (128.0). `LmCodeBias` folds the offset into
  the scale multiply. A `static_assert` enforces `BITS <= 7`: a wider code
  would spill into the exponent and double the value.
- **Code packing.** `LmStoreCodeOctet` writes 8 codes into a byte-aligned
  block that belongs to one thread. The earlier pair writer updated
  overlapping 32-bit words, so adjacent threads lost each other's bits.

## 2.4 Top-k, route, speculative verify, GQA, projections

**Top-k (`topk.cuh`, `topk_exact.cuh`)**

- **Small top-k.** `LmTopkSmallKernel` runs a bitonic sort in shared memory
  over up to `LM_TOPK_SMALL_LIMIT` (1024) scores. It is the router top-k.
- **Exact top-k.** `LmTopkExactKernel` finds the exact k-th key in
  `LM_TOPK_EXACT_PASSES` (4) radix passes of 8 bits each
  (`LmTopkExactPass`). `LmTopkExactEmit` then compacts with a block scan in
  index order: every key above the threshold, then the lowest-index ties. The
  result is exact and the same on every run.
- **Float order as integer order.** `LmTopkKey` flips the sign bit of a
  non-negative float and inverts a negative one. Unsigned comparison then
  matches float order.
- **Bias selects; it does not weigh.** `selection_bias` is added only to the
  sort key. Output values come from the unbiased `LmTopkScore`.
- **Score transform in the same pass.** `LmTopkScore<SCORE_TRANSFORM>`
  (identity, sigmoid or sqrt-softplus) is applied while the keys are built.
- **Grouped selection.** Each group is scored by the sum of its top two
  values, recovered through `LmTopkValue`. A group is kept when fewer than
  `TOP_GROUPS` groups beat it.
- **Renormalization** uses a block reduction.

**Route (`route.cuh`)**

- `LmRouteBuildKernel` is a counting sort over `EXPERTS` buckets: count,
  serial prefix, scatter. It also writes the tile prefixes.
- **Rows within an expert are unordered.** They appear in atomic-arrival
  order, so callers must not assume stable or sorted rows.
- **Scales follow the source row.** An indirect A read must index its scales
  by the source row as well (`LmRouteSourceRow`). Otherwise another token's
  scale is applied, with no fault.

**Speculative verify (`speculate.cuh`)**

- **Greedy.** `LmSpeculativeVerifyGreedyKernel` accepts the longest prefix
  where the draft matches the target argmax. At the divergence point it
  writes the target argmax (the bonus token) and sets
  `accepted_count = accepted + 1`. It then rolls `context_length` back by
  `draft_length - accepted`.
- **Sampled.** `LmSpeculativeVerifySampledKernel` accepts each draft token
  with probability `min(1, p_target / p_draft)`. On rejection it resamples
  from the positive part of `target - draft`.
- **Cache rollback.** A rejected token's KV has already been written. The
  context length is set from the accepted count, and nothing else.
- **Where the rest lives.** The speculation seam, tree and policy are in
  `include/sparkpipe/spark_speculation_*.h` and `src/spark_speculation_*.c`.

**GQA (`gqa.cuh`)**

`LmGqaKvStoreKernel` and `LmGqaAttentionDecodeKernel` both
`static_assert` the same slot layout, `[K: heads × head_dim][V: heads ×
value_dim]` in BF16. The decode accumulator is sized from `VALUE_DIM`.

**Context-split GQA (`gqa_shard.cuh`)**

- Each rank attends over its own share of the positions and sends every
  destination rank a partial record (unnormalized values, running max and
  sum per head); `LmGqaShardMergeKernel` combines them.
- `LmGqaShardTiledKernel` takes 128 query vectors of one KV head per block,
  flattened as (row, head) so any group size fills it, 16 per warp. 64-key
  K and V tiles stream through cp.async double buffers; scores and the
  online softmax stay in registers. P·V multiplies the bf16-rounded weights,
  as the reference models do (fp32 softmax, then bf16 before the matmul),
  while the denominator sums the f32 weights. Normalized values therefore
  stay within 2^-9 of the f32 scalar kernel for values in [-1, 1).
- A tile every query of the warp sees whole (no block mask) skips the
  per-key visibility test and only scales its scores. That gives the same
  bits as the masked path, so a row's result still does not depend on the
  rows beside it.
- Tile bounds are absolute and a fully masked tile leaves the state exact,
  so a row's result does not depend on the rows beside it. The launch
  therefore uses the tiled kernel for every row count of a geometry it
  fits (key heads up to 256 wide, value heads any multiple of 64): decode,
  verify and prefill rows compute the same bits.
- Keys up to 192 wide take 64-key tiles, wider keys 32-key tiles. Value
  heads wider than 128 split into 128-column parts over the grid's third
  dimension; each part recomputes the scores and only part 0 writes the
  running max and sum.
- On GB10, 64 heads over 8 KV heads of 128 and 8,192 local keys: 2,048 rows
  take 9.3 ms (58.9 TFLOP/s, about 65% of the measured 90 TFLOP/s BF16 peak)
  against 470 ms for the scalar kernel, and one decode row at 128K positions
  0.47 ms against 5.9 ms. With 24 heads over 4 KV heads of 256 and 4,096
  local keys: 2,048 rows take 5.6 ms (37.0 TFLOP/s counting each part's
  scores once) against 105 ms, and one decode row 0.32 ms against 2.2 ms.
  Splitting P into bf16 high and low parts (the previous version) cost 19% at
  the 128-wide shape; the whole-tile path is the other 1.28x.
**Latent shard attention (`attn_shard.cuh`, `attn_shard_prefill.cuh`)**

- A rank holds a share of every sequence's latent keys and computes, for
  every head, a partial record (max, sum, unnormalized latent output) over
  its own keys. `LmLatentShardMergeKernel` merges the ranks' records.
- `LmLatentShardPartialKernel` is the scalar kernel: one block per row and
  up to four heads, with every key re-read by each block.
- `LmLatentShardPrefillKernel` runs the same records on tensor cores: one
  row and 32 heads per block row, 32-key tiles through a two-buffer ring,
  QK and PV on `mma.sync`, one BF16 copy of each tile's weights against
  the running maximum (the publishers' flash form, Latent prefill below).
  - A block is 8 consumer warps and two producer warps, one block an SM,
    persistent over a grid stride of row blocks. The planner warp reads
    each row's sequence, positions and selection and fills a 4-row plan
    ring: tile range, key window, one key count per query and the first
    tile's split-list positions. It compacts in-kernel lists with warp
    ballots; a block's first row is compacted by the whole block. The
    issuer warp resolves slots one tile ahead, prefetches the next tile's
    page-table line into L1, loads each key row with one `cp.async.bulk`
    into an unswizzled row at a 16-byte padded pitch, zeroes null rows not
    already zero, and prefetches the next row's queries into L2. Full and
    empty mbarriers pace the two-buffer ring.
  - The consumers are two query groups of 4 warps (16 queries each) with
    their own named barriers. QK splits a group's tile into 2 k-halves x 2
    key pairs, the softmax gives each query 8 threads (keys k, k+8, k+16,
    k+24, summed in the 32-lane butterfly's pairing), and P·V gives each
    warp 128 value columns. A padded head group of at most 16 heads runs
    one query group. Queries for the next row load into registers after a
    row's last QK.
  - lenovo8 (GB10), against the one-row-per-block kernel this replaced
    (1,440 DSV4.1 calls of 410 µs in a 16K fleet prefill), ms a launch:

    | Wave | Before | Now |
    |---|---|---|
    | DSV4.1 landed keys, 2,048 rows, 4 heads, top 512 split, per source | 0.365 | 0.141 |
    | K3 dense, 2,048 rows, 96 heads, 8K / 64K / 256K | 5.93 / 39.6 / 173 | 4.00 / 28.1 / 138 |
    | K3 gather, 2,048 rows, 6 heads, 64K, per source | 2.38 | 1.63 |
    | Ling 3, 2,048 rows, 32 heads, 16K dense | 3.53 | 2.45 |
    | K3 top 2,048 compacted, 2,048 rows, 8K / 64K / 256K | 4.07 / 3.89 / 5.03 | 3.41 / 3.93 / 4.06 |
    | DSV4.1 64 heads, top 512 compacted, 2,048 rows | 1.90 | 2.01 |
    | K3 decode, 1 / 4 / 16 rows at 64K | 0.039 / 0.098 / 0.319 | 0.033 / 0.076 / 0.228 |
    | K3 top 2,048 decode, 1 / 4 / 16 rows at 64K | 0.027 / 0.027 / 0.029 | 0.022 / 0.021 / 0.023 |
    | DSV4.1 64 heads pair records, 1 / 16 rows at 64K | 0.0164 / 0.0205 | 0.0171 / 0.0209 |

    The dense and gather waves are bound by the two-buffer ring: with no
    QK, softmax or P·V the producers alone take 90% of the time, and a
    third 37 KB buffer does not fit. The compacted 96- and 64-head waves
    are bound by DRAM: they write 404 and 269 MB of FP32 records a launch
    (GB10 writes 198 GB/s) and read 226 and 134 MB of queries. A single
    compacted DSV4.1 row pays about 0.7 µs more than before for the
    planner and issuer hand-offs before its first tile. Records are main's
    bits in every case (`tests/cuda/latent_shard_prefill_bits_cuda.cu`
    keeps the old kernel).
  - Rows past the dense limit read their selected positions, either from
    a list compacted in shared memory or from the caller's split lists
    (`LmLatentShardSplitLaunch`). The two orders are the same, so both
    give the same records.
  - A row's record does not depend on the other rows of its block's
    stride: decode, verify frames and prefill waves agree bit for bit.
  - Pair-grouped records (`record_heads`, `record_copies`) are written as
    the scalar kernel writes them, and a row at position `0xffffffff` has no
    keys.
- `LmLatentShardAttendLaunch` and `LmLatentShardAttendHeadsLaunch` take the
  scalar launchers' arguments and use the tensor-core kernel when the width,
  the head count and the shared memory for the list fit; otherwise they run
  the scalar kernel. GLM 5.3 Flash, GLM 5.3 Full's scatter waves, A.X K2,
  Ling 3's MLA layers, Hy4 and both DSV4.1 attention paths call them.
- A head count that is a multiple of 32 fills each block. A head count below
  32 that divides 32 (`padded_heads`) runs one row per block row with its
  heads repeated across the query slots, and only the first copy writes
  records.
  MMA rows are independent, so each head computes exactly what it would in a
  full block. DSV4.1's landing path (4 heads per rank) therefore still equals
  its records path (64 heads) bit for bit (`tests/test_dsv41_flash_landing.cu`).
- lenovo7 (GB10), 64 heads over 16 ranks, one rank:

  | Shape | Scalar | Tensor cores |
  |---|---|---|
  | 4,096 rows, 16K context, 2,112 selected of each row, width 512 | 51.7 ms | 5.19 ms |
  | Same, width 576 | 40.2 ms | 5.00 ms |
  | 4,096 rows, 16K dense keys, width 576 | 99.0 ms | 8.73 ms |
  | Decode rows: 1 / 4 / 16 at 131K, 2,112 selected | 0.038 / 0.056 / 0.173 ms | 0.028 / 0.027 / 0.027 ms |
  | 4,096 rows, 16K context, 2,112 selected, 4 heads (padded) | 18.2 ms | 11.8 ms |

  The records are the flash form. On one rank every record lies closer to an
  f64 model of it (each 32-key tile's weights against the running maximum
  rounded to BF16 once, the sum unrounded) than to exact attention: 5e-5 to
  1.8e-4 of each head's peak from the model, 1.6e-3 to 1.7e-3 from exact.
  Against the FP32 scalar kernel the normalized outputs differ by up to 0.3%
  of each head's peak, which is that rounding
  (`tests/cuda/latent_shard_sparse_cuda.cu`). A low BF16 half (540dfd170)
  once matched the scalar kernel and the fleet's earlier greedy tokens; the
  publishers' kernels multiply one BF16 copy, the gates judge against their
  references, and without the half the kernel is 5 to 15% faster on the
  shapes above (the width-512 case is within run-to-run noise).

**Latent prefill (`attn_prefill.cuh`, `attn_shard_prefill.cuh`)**

- `LmLatentAttentionPrefillKernel` (GLM 5.3 Full and A.X K2, dense and
  DSA-sparse) and `LmLatentShardPrefillKernel` (GLM 5.3 Flash, GLM 5.3
  Full's scatter waves, A.X K2, Ling 3, Hy4, DSV4.1) take the
  same form as the GQA tiled kernel and the publishers' own kernels
  (FlashAttention-2, DeepSeek's `sparse_attn`): each 32-key tile's weights
  `exp(s - m)` against the running maximum are rounded to BF16 once for the
  P·V MMA, and the running sum adds the FP32 weights.
- The latent prefill used to split each weight into BF16 high and low parts
  and run two P·V MMAs per fragment. One part is 8 to 12% faster on GB10
  (4 heads, 512 + 64 wide): 2,048 rows after 30,720 cached positions take
  21.5 ms against 23.9 ms, 4,096 rows after 61,440 take 84 to 87 ms
  against 91 to 97 ms (three runs each, `tests/cuda/attn_prefill_cuda.cu
  --time` with longer cases). The kernel is not bound by its MMAs alone, so
  the gain is less than the third of its MMAs that the low part was.
- A block of `LmLatentAttentionPrefillKernel` holds `LM_PREFILL_ATTN_QUERIES`
  = 16 queries: 4 rows of 4 heads, all reading one latent key per position.
  The count is a template parameter; 32 (8 rows) builds and passes the same
  tests. 16 stays the default because it is bit-equal to the earlier kernel;
  32 changes tile boundaries over the block's union and waits for the fleet's
  measured 8-row union sizes.
  - The 16 queries of each MMA row tile are staged through the second key
    buffer into registers, 72 per thread (a k-half of 288 columns), as in
    `LmLatentShardPrefillKernel`. Two 32-key tiles still double-buffer in
    85,888 bytes of shared memory. 32 queries held in shared memory next to
    two key tiles would not fit.
  - A block loads each union key once for 8 rows instead of 4, which halves
    the key bytes per query when the rows' selections overlap.
  - Key slots resolve without stalling the block. Each tile loads the union
    positions three tiles ahead and prefetches the page-table entries two
    tiles ahead into L1. It resolves the slots two tiles ahead after its P·V.
    The old kernel looked the slot up at the top of every tile: two
    dependent global loads with branches, between barriers.
  - The softmax gives each query 8 threads (16 at 16 queries). A thread holds
    keys `k, k+8, k+16, k+24` and adds them in the order of the old 32-lane
    butterfly, so the running sums keep their bits.
- `LmPrefillSparseUnionKernel` marks each 131,072-position window of
  selected positions in a shared bitmap. It counts bits per word to give
  each position its union index, and ORs each row's bit into that index's
  membership byte. The union comes out ascending, as from the merge it
  replaced. That merge binary-searched every other row's list in global
  memory for every entry and took 0.87 ms of a 3.9 ms call. The bitmap takes
  0.06 to 0.3 ms.
- Bits: dense outputs are equal at 16 and 32 queries and equal to the
  16-query kernel before this change. So are sparse outputs at 16 queries,
  on every bench pattern at 64K and 300K context. At 32 queries a block's
  union covers 8 rows, so its 32-key tiles and their BF16 weights differ
  wherever two 4-row halves select different keys. Outputs then move by up
  to one BF16 step of the head's largest output
  (`tests/cuda/attn_prefill_sparse_cuda.cu`).
- lenovo8 (GB10), 2,048 rows of 4 heads after 65,536 cached positions
  (78 MB of keys), top 2,048 per row, times in ms including the union
  kernel. Rows share a selection in groups of 4 (the old bench) or 8:

  | Selections | Before (16 queries) | 16 queries (default) | 32 queries |
  |---|---|---|---|
  | Same set in groups of 4 / 8 | 3.97 / 3.48 | 2.80 / 1.96 | 1.99 / 1.53 |
  | 75% shared in groups of 4 / 8 | 5.35 / 5.17 | 3.31 / 3.15 | 2.93 / 2.42 |
  | 50% shared in groups of 4 / 8 | 6.33 / 6.36 | 3.56 / 3.58 | 4.01 / 3.66 |
  | Disjoint | 8.60 | 4.45 | 6.01 |
  | Most recent 2,048 | 2.66 | 1.27 | 0.99 |

  - Same set, 8-row groups: the 32-query attention takes 1.39 ms. A bare
    `cp.async` gather of the same random 1,152-byte rows takes 1.43 ms
    (420 GB/s). So the attention runs at the gather rate of this cache;
    L2-resident gathers reach 1.6 to 1.8 TB/s.
  - With the 4 or 8 rows' selections mostly disjoint, 32 queries do twice
    the MMA work per key that 16 do, for little saving in bytes.
  - After 300,000 cached positions (348 MB), 32 queries are faster in every
    case. Same set: 2.42 against 2.60 at 16 queries (groups of 8) and 4.04
    against 4.57 (groups of 4), from 3.86 and 5.38 before. Disjoint: 7.85
    against 11.2, from 13.1 before.

**Projections (`project.cuh`)**

- The low-rank and absorbed attention projections live in each model's
  own layer code; `project.cuh` keeps the shared split, rope and per-head
  kernels.
- **The per-head value projection stays unabsorbed**
  (`LmPerHeadProjectKernel`), for two reasons (4187f90^):
  - Elementwise output gating does not commute with the fold.
  - On GB10, absorbing the value half would grow the output projection
    from 8.30 GB to 19.55 GB across 24 layers. That costs about 55 ms per
    token to save a 302 MFLOP kernel that takes 46 µs.
- **Per-head projections in a row-exact step** (`project_chain.cuh`).
  `LmPerHeadProjectChainLaunch` runs 1 to 8 rows on the chain kernel and
  more rows on the tensor-core kernel (`LmPerHeadProjectMmaKernel`), whose
  summation order differs from the chain kernel's. A row-exact step (a
  verify frame, `commit = 0`) passes `row_exact`, and above 8 rows it then
  runs the CUDA-core tile kernel (`LmPerHeadProjectRowsLaunch`), which adds
  each output's products in the chain kernel's order. Every row of a verify
  frame so has the bits of its one-row decode
  (`tests/test_project_chain_cuda.py`, `tests/test_glm52_verify_frames.py`).

## 2.5 Norm and linear attention (`norm.cuh`, `linear_attn.cuh`)

- **Block reductions.** `LmBlockSum` and `LmBlockMax` reduce with warp
  shuffles first, then do one round through shared memory.
- **Fused residual and norm.** `LmFusedResidualRmsNormKernel` reads the
  residual once and writes both the normed row and the new residual. The
  sum is rounded to BF16 before the statistic (`SparkMathResidualSum`), as
  the publishers' eager `residual + hidden_states` rounds it, so the norm
  reads the residual it stores. `ROUND_FIRST` rounds the normed value
  before the gain (`SparkMathNormGainRounded`), as a publisher's
  `weight * x.to(bf16)` does; each geometry of the Gemma, Ling and GLM 5.2
  families chooses it (`kNormRoundFirst`).
- **Activation quantize.** `LmQuantiseRowsKernel` rounds each group scale
  to UE4M3 first and encodes the codes against that stored scale, so the
  consumer's `code * scale` is the nearest grid point. A nonzero group never
  stores the zero scale: it clamps to the smallest positive code.
- **Scale-group guard.** `LmQuantiseRowsKernel` asserts
  `Format::kScaleGroup > 0`. That catches the divide-by-zero class that
  shipped when an unquantized BF16 format reached the quantizer.
- **MoE finalize.** `LmMoeFinalizeKernel` folds `top_k` packed rows back to
  token-major order, weighted by the router gates. If it is left out, the
  output looks like a routing bug.
- **Bounded decay.** `LmBoundedDecay` computes
  `exp(minimum_log_decay × sigmoid(scaled))`, so the log decay stays in
  `(minimum_log_decay, 0)`. K3 sets the bound to -5
  (`SPARK_LLM_KDA_GATE_LOWER_BOUND`).
  - **Why the bound matters (4187f90^).** It keeps the chunkwise
    reciprocal inside BF16 range.
  - A softplus in place of the sigmoid, or dropping the bound, brings the
    overflow back.
- **Replay stores inputs, not state.** `LmReplayFoldKernel` folds the stored
  per-step inputs (`LmReplayStep`) instead of snapshots of the state. The
  fold must not recompute the gate (4187f90^).
- **One kernel for decode, prefill and verify.** `LmDeltaRuleKernel` and
  `LmCausalConvKernel` take a `commit` flag. With `commit == 0`, as in
  verify, the state is not written back. A run of T rows must be
  bit-identical to T one-row calls; `tests/host_cuda/k3_run_equivalence.cu`
  checks this.
- **Two reduction buffers.** `LmDeltaRuleKernel`'s `norm_reduction` holds two
  buffers. `LmBlockSum` reads `shared[0]` after its final barrier, so a
  second call back to back could overwrite it before another warp reads it.

## 2.6 `spark_lm_kernels.cuh`: expert tiles, native MXF kernels, Mloop, head

**Consumers.** Each consumer includes the header into its own CUDA unit.
The MXFP4 group size is the template parameter `GROUP_SIZE`, so a module
that disagrees on it fails to build. The consumers are:

- the modules dsv4, gemma4, glm5_next, laguna, ling, muse_glimmer,
  qwen38_27b, qwen38_max and qwen4_flash;
- `common/common_gdn_stage_kernels.cu`;
- `model-families/glm52/cuda_tree/spark_glm_cuda_layer.cuh`, for glm52;
- `spark_lm_certified_launch.h`.

**Kernels.**

- **Expert tile policy.** `SPARK_LM_EXPERT_TILE_POLICY` selects `ALL_WARPS`,
  `SOFTWARE_PIPELINED` or `AUTOMATIC`, the default. Under `AUTOMATIC`,
  `SparkLmExpertTileDispatch` runs `SparkLmExpertTileBodyAllWarps` when
  `input_dimension <= SPARK_LM_TILE_K`, and
  `SparkLmExpertTileBodySoftwarePipelined` otherwise.
- **Grouped scalar path.** `SparkLmGroupedScalarLinearKernel` is the grouped
  expert linear without tensor cores. It stages the source activation once
  in shared memory.
- **Native MXF8/MXF4.**
  - Activation staging: `SparkLmSm121StageMxf8` stages activations as E4M3,
    with an E8M0 scale taken from each block's amax.
  - Linear and dense kernels: `SparkLmSm121NativeLinearKernel` and
    `SparkLmSm121FusedDenseW13Kernel`.
  - Expert kernels: `SparkLmSm121FusedExpertW13Kernel` and
    `SparkLmSm121ExpertW2Kernel`.
  - E2M1 layout: for `kind::mxf8f6f4`, each E2M1 value must sit in register
    bits [5:2] of its 8-bit container (4187f90^).
- **B1 uses a GEMV.** At B1, `SparkLmSm121B1ExpertW13Kernel` and
  `SparkLmSm121B1ExpertW2Kernel` run a matrix-vector product. Padding B1 to
  the M16 atom would multiply activation work by sixteen (4187f90^).
  `SparkLmSm121NativeDecodeShape` qualifies the native path for rows 1, 5, 7,
  8, 16, 32, 64 and 1024.
- **Pair policies.** `SparkLmSm121B1Bf16LinearPairPolicy` and
  `SparkLmSm121B1Fp8LinearPairPolicy` choose `FLAT_16` only when both hold:
  - `row_count == 1`;
  - input × output is at least `SPARK_LM_BF16_PAIR_WIDE_MINIMUM_WORK`
    (524288) for BF16, or `SPARK_LM_FP8_PAIR_WIDE_MINIMUM_WORK` (1572864) for
    FP8.

  Otherwise they choose `FLAT_8`.
- **Mloop.**
  - `SparkLmExpertTileAllMloopKernel` (grouped experts) gives one CTA
    eight m-tiles (`M_GROUP`) of one n-tile. Each k-stage weight strip is
    staged once for all of them. Its dense BF16 sibling had no callers left
    and was removed with #927.
  - **Why (4187f90^).** At B=256 the plain grid launched about 122K mostly
    empty CTAs, at about 1.25 µs each. That was the measured collapse to
    233 GB/s.
- **MoE grouping.** `SparkLmMoeGroupKernel` runs a device histogram,
  prefix and scatter over up to `SPARK_LM_MOE_MAX_EXPERTS` (1024) experts.
  `SparkLmMoePairReduceKernel` folds the results back through the inverse
  map.
- **Weight read-ahead.** `SparkLmWeightReadAheadKernel` loads one `uint4`
  per 32-byte sector (`SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES`). It XORs
  each value into a per-thread sink so the compiler cannot drop the loads.
- **Head.**
  - `SparkLmHeadArgmaxKernel` fuses the matrix-vector product with a running
    argmax, so no logits tensor is written. Ties go to the lower candidate.
  - `SparkLmHeadCertifiedFp8QuantizeKernel` and
    `SparkLmHeadCertifiedFp8ScoreKernel` screen candidates with an FP8
    shadow of the head.
  - `SparkLmHostLaunchHeadScreenedArgmax` then rescores the survivors
    against the BF16 head.

---

### 2.6.1 NVFP4 layout, staging warps and TP windows

- **Format codes are kernel selectors.** The `SPARK_LM_WEIGHT_FORMAT_*`
  values pick a decoder. They are not stage-pack wire codes, and callers
  that hold a wire code translate it explicitly. Examples are
  `LmGdnStageExpertCodec` and `LmGdnStageLaunchGroupedExpertLinear` in
  `common/common_gdn_stage_kernels.cu`, and the qwen4_flash module.
  - Several values equal a wire code: BF16 0, F32 1, U32 2, MXFP4 3
    (`SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1` in
    `common/common_stagepack_format_ext.h`), FP8 with E8M0 block-128 scales
    6, and `SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1` 8
    (`SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED`).
  - FP8 block-128 with F32 scales is 5 here and 4 on the wire. Code 4 here
    is `SPARK_LM_WEIGHT_FORMAT_FP8_E4M3`.
  - The wire's `SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1_E8M0G32` (9) has no
    code here.
  - No static assertion ties any pair together.
- **NVFP4 decode.** NVFP4 here is the ModelOpt layout.
  `SparkLmDotRowNvfp4`, `SparkLmDotRowNvfp4Pair` and `SparkLmTileDecodeRun`
  decode one weight as e2m1 × e4m3 × global.
  - Payload: packed like MXFP4, eight e2m1 values per 32-bit word.
  - Scales: one e4m3 byte per 16 inputs, where MXFP4 has one E8M0 byte per
    `GROUP_SIZE` inputs. `SparkLmTileDecodeRun` indexes the plane by its
    template `GROUP_SIZE`, so `SparkLmHostLaunchBatchedLinear` and
    `SparkLmHostLaunchGroupedExpertTileMloop` instantiate the tile kernels
    with 16 for NVFP4.
  - Global: one F32 `weight_scale_2` per dense weight or per expert segment.
  - Besides `SparkLmValidateLinearContract` (Part 1),
    `SparkLmHostLaunchGroupedScalarLinear` also refuses NVFP4 when
    `input_dimension % 16 != 0`.
- **Where the NVFP4 global sits.**
  - Dense weights: the scale buffer is `[e4m3 plane][F32 global]`. These
    kernels read the global at byte offset
    `output_dimension * (input_dimension / 16)`, directly after the plane:
    `SparkLmLinearKernel`, `SparkLmGatherLinearKernel`,
    `SparkLmExpertTileBodyAllWarps` and
    `SparkLmExpertTileBodySoftwarePipelined`.
  - Expert segments: each expert's scale segment is
    `[e4m3 plane][F32 input scale][F32 weight_scale_2]`, that is, the plane
    plus `SPARK_STAGEPACK_NVFP4_GLOBAL_SCALE_BYTES` (8). The weight global is
    the last 4 bytes:
    - `SparkLmGroupedScalarLinearKernel` reads it at
      `scale_group_stride_bytes - 4`;
    - `SparkLmExpertTileAllMloopKernel` reads it at
      `scale_expert_stride_bytes - 4`;
    - `SparkLmSm121B1ExpertW13Task` and `SparkLmSm121B1ExpertW2Task` read it
      at plane + 4. They place each group's segment at
      `group * output_dimension * (input_dimension / 32)`, the MXFP4 stride.
      An NVFP4 segment is `output_dimension * (input_dimension / 16) + 8`
      bytes, so with NVFP4 every group after group 0 reads its plane and
      global from the wrong offset.

    The 4 bytes at plane + 0 are the input scale, not the weight global.
    `SparkLmExpertTileAllKernel` reaches the dense bodies through
    `SparkLmExpertTileDispatch`, so with NVFP4 it would read the input scale
    as the global. Nothing in the tree launches it.
- **Mloop warp roles.** `SparkLmExpertTileAllMloopKernel` runs 256 threads
  and works through each K tile in two phases.
  - Warps 0-3 run WMMA on the current buffers while warps 4-7 (threads
    128-255) stage the next input tile and the first half of the next weight
    tile.
  - Warps 4-7 then run WMMA while warps 0-3 stage the second half of the
    next weight tile.
  - Only warps 4-7 stage input. Their loop indexes the 16 × 64 tile from
    `threadIdx.x - 128` with stride 128. Indexing from `threadIdx.x` would
    start at entry 128, so rows 0-1 of every staged input tile would never
    be written.
- **TP-windowed grouped launch.** `SparkLmGroupedScalarLinearKernel` works
  on a window of the global route table.
  - `LmGdnStageLaunchGroupedExpertLinear` offsets `group_tile_prefix` and
    `group_row_offset` by `tp_rank * experts_per_rank` and passes
    `group_count = experts_per_rank`.
  - So `group_tile_prefix[0]` is this rank's first global tile index, and
    the rank's tasks are
    `[group_tile_prefix[0], group_tile_prefix[group_count])`. The task loop
    starts at `group_tile_prefix[0] + blockIdx.x` for this reason.
  - For a task below the window, `SparkLmGroupedScalarGroupOfTile` returns
    group 0 and the unsigned `in_group` underflows, so the kernel can read
    rows outside the group.
  - A window that starts at tile 0, as rank 0's does, cannot show this
    fault.

## 2.7 Dependent launch around mesh waits (`spark_tp_mesh_kernels.cuh`)

A mesh round runs publish, then arrival (one warp spinning on the peers'
tails while it prefetches the next weights), then combine. The combine is
the arrival's programmatic dependent, and it waits on the tails itself
(`own_wait`).

- **A spinning kernel releases its dependents only when they are small.**
  A released dependent becomes resident at once and spins in its own wait.
  - Large combines (up to 192 blocks) pinned the GPU this way: every rank
    of `tp_mesh_hardware_probe` shares one GPU, and from degree 8 the
    publish kernels could no longer run (#1630).
  - `SparkTpMeshHardwareDirectKernel` uses 128 registers per thread, so a
    GB10 holds only 2 of its 256-thread blocks per SM, 96 in all.
- **Decode-size combines are released before the wait.** A combine of at
  most `SPARK_TP_MESH_EARLY_ELEMENTS` (16,384) elements launches with at most
  `SPARK_TP_MESH_EARLY_BLOCKS` (4) blocks of 256 threads, and the arrival
  releases it before it spins (`SparkTpMeshCombineEarly`, `SparkTpMeshEarlyGrid`).
  - It reacts as soon as the data lands, instead of paying a block dispatch
    after the arrival exits on each of the ~300 collectives of a decode
    step.
  - Gathers loop over peers inside a block, so four blocks cover all sixteen.
  - The block width matters. With 128-thread blocks, a 7,168-element sum
    gives each thread two 8-element quads, and that cost as much as the early
    release saved. With 256 threads each thread sums one quad.
- **Larger combines are released after the wait**, so a resident dependent
  never holds more than four blocks while it waits.

# Part 3 — Finding kernels and their numbers

- **Kernel inventory.** Generate it instead of keeping a list by hand:
  ```sh
  grep -n '__global__' inference/kernels/*.cuh \
      model-families/common/include/sparkpipe/spark_lm_kernels.cuh \
      common/*.cu common/*/*.cuh
  ```
- **Numbers.** Look them up in `PERFORMANCE_STATUS.md` and
  `docs/GLM5_NEXT_ROOFLINE.md`. Never record a projection as a measurement.
- **Rejected optimizations.** `PERFORMANCE_STATUS.md` records optimizations
  that were bit-exact but regressed end to end. Do not propose them again
  without new evidence.

# Appendix — implementing against this playbook

1. **Work from the contract, not the caller.** If a field is missing, ask
   for it rather than inferring it. Inferring is how the "correct at
   rows==1" bug class happens.
2. **Reuse the shared pieces.** Use the staged pipeline in `tile.cuh` and
   the atoms in `mma.cuh`. Never write a second mbarrier protocol or a second
   fragment mapping.
3. **Respect the hard constraints:**
   - static shared memory ≤ 48 KB;
   - K alignment as refused in Part 1;
   - no wgmma, tcgen05 or 4X+ue8m0 on sm_121a;
   - native MXF atoms trap outside `__CUDA_ARCH__ == 1210`;
   - no model names in shared code.
4. **Measure at the right boundary.** An isolated fusion probe is the cheap
   first gate. The acceptance gate is the driver's end-to-end decode with
   exact token parity. A positive isolated result that regresses end to end
   is rejected.
5. **Record the measurement** in `PERFORMANCE_STATUS.md` or
   `docs/GLM5_NEXT_ROOFLINE.md`. Update this playbook if the technique or a
   constraint changed.
