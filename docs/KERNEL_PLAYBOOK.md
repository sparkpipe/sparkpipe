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
  - `SparkLmHostLaunchBatchedLinearMloop`, when `input_dimension` is not a
    multiple of `SPARK_LM_TILE_K`.

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
- **Linear bulk copies.** `LmTmaLoadBulk1d` copies linear addresses for
  indirect rows. It has no hardware bounds check, so the caller must clamp
  the index.
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
  `LmRouteSourceRow(source_row_map, p)`, one `LmTmaLoadBulk1d` per 16-byte
  chunk, so no packed activation tensor is ever written.
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

**Projections (`project.cuh`)**

- `LmLowRankProject` is the raw form and `LmAbsorbedProject` the absorbed
  form. At decode, absorbed avoids materializing per-head K and V.
- **The per-head value projection stays unabsorbed**
  (`LmPerHeadProjectKernel`), for two reasons (4187f90^):
  - Elementwise output gating does not commute with the fold.
  - On GB10, absorbing the value half would grow the output projection
    from 8.30 GB to 19.55 GB across 24 layers. That costs about 55 ms per
    token to save a 302 MFLOP kernel that takes 46 µs.

## 2.5 Norm and linear attention (`norm.cuh`, `linear_attn.cuh`)

- **Block reductions.** `LmBlockSum` and `LmBlockMax` reduce with warp
  shuffles first, then do one round through shared memory.
- **Fused residual and norm.** `LmFusedResidualRmsNormKernel` reads the
  residual once and writes both the normed row and the new residual.
  `LmFusedNormQuantiseKernel` normalizes and quantizes in one pass.
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
  - `SparkLmExpertTileMloopKernel` (dense BF16) gives one CTA
    `SPARK_LM_MLOOP_GROUP` (8) m-tiles of one n-tile. Each k-stage weight
    strip is staged once for all of them.
  - `SparkLmExpertTileAllMloopKernel` is the grouped-expert version.
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
