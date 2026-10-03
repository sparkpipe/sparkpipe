# minimax_resident_decode_stage

CUDA resident decode-stage firmware and serving adapter for the MiniMax H3
text tower (`MiniMaxAI/MiniMax-H3`): BF16 weights, 64 layers, hidden size
5120, served as one TP4 group.

## Implementation notes

### Static assertions

`spark_minimax_resident_decode_stage_firmware.h` is compiled as C and inside
the C++17 CUDA translation unit. Its include chain does not reach
`spark_driver_defines.h`, which defines `SPARK_LLM_STATIC_ASSERT`, so the
header defines its own `SPARK_MINIMAX_STATIC_ASSERT`. That macro maps to
`static_assert` in C++ and `_Static_assert` in C.

### Head norm and RoPE (`SparkMinimaxHeadNormRopeKernel`)

The per-head RMS norm runs before the rotary embedding. The rotation pairs
element i with element i +/- head_dim / 2. The partner must be the
normalized value, so each thread writes its normalized element to the shared
`exchange` array and reads its partner from there. The source buffer is not
modified until the final write: keys are written back in place as BF16, and
queries go to `query_roped_f32`.

### Vocabulary argmax

- `SparkMinimaxVocabArgmaxKernel` packs each candidate with
  `SparkMinimaxSortableScore` into a `uint64_t`: order-preserving float bits
  in the high word and the token id in the low word. It combines candidates
  with `atomicMax`.
- CUDA has no `atomicMax` overload for `uint64_t` (`unsigned long` on LP64).
  The kernel therefore reinterprets the cell as `unsigned long long`, which
  has the same width and representation.
- `SparkMinimaxArgmaxResolveKernel` takes the low 32 bits as the token.

### Serving descriptor

- The adapter is one hybrid TP group of four ranks (`parallel_group_size` 4,
  `stage_count` 4). Each of the four `stage_layer_counts` entries is the full
  64-layer count.
- `SparkDescriptorCheckStageLayerTotals` requires every entry to be nonzero.
  In hybrid mode it also requires the entries of a group to be equal and to
  sum, one entry per group, to the layer count.
- The descriptor claims neither `PREFIX_REUSE` nor `CACHE_PUBLISH`.
  `SparkDescriptorCheckRequiredCacheOperations` refuses such a descriptor
  with `SPARK_STATUS_UNSUPPORTED`. It runs whenever an adapter is loaded,
  through `SparkModelServingAdapterValidateInterface`.

### GPU component validator (`validation/spark_minimax_resident_decode_stage_cuda_validation.cu`)

- The validator compares these kernels against host mirrors at the model
  hidden and head dimensions: RMSNorm, SwiGLU, embedding gather, per-head
  norm with RoPE, vocabulary argmax with resolve, and the two TP combine
  kernels (`SparkTpLaunchSumRanksF32`, `SparkTpLaunchAccumU64Max`).
- Only the RMSNorm check launches twice and requires a bit-exact rerun.
- `SparkMinimaxLaunchResidualAdd` is declared but not exercised.
- It does not read a stage pack. It does not cover the linear projections,
  attention, the KV write path or TP collectives over a real transport. A
  PASS is a kernel-component result, not model or driver acceptance.
- `ValidationCloseBf16` allows two BF16 rounding steps between a BF16 output
  of fp32 device math and the double-precision host mirror. The bound is
  `2^-7 + 2^-7 * |expected|`.
- In the argmax check, the winner row of the test LM head is a copy of the
  input row. Its score is the input's squared norm, about 1.7e3 for 5120
  inputs uniform in [-1, 1). The other rows have weights scaled by 0.03 and
  score on the order of 1. This margin makes the result independent of fp32
  accumulation order.
- `validation/minimax_cpu_validate.c`, built by
  `build_minimax_cpu_validate.sh`, is a host tool that loads a stage pack
  through the module's `SparkMinimaxModuleLoadMappedPack`.
