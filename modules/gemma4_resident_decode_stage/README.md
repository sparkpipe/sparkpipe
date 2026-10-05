# gemma4_resident_decode_stage

CUDA resident decode-stage firmware and serving adapter for Gemma 4: `Makefile`
builds the 31B dense BF16 module and `Makefile.moe` builds the 26B-A4B MoE
module.

## Implementation notes

### Sliding-layer K and V buffers

- `SparkGemma4ModuleRunAttentionBody` splits the fused sliding-layer KV
  weight (`kv_fused`: K rows, then V rows) into two linear views at launch.
  It writes K and V into separate slot buffers, `sliding_kv_bf16` (K) and
  `sliding_value_bf16` (V), each `max_input_row_count *
  sliding_kv_heads_per_rank * SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION` BF16
  values (`SparkGemma4ModuleAllocateSlot`). K gets the key head RMS norm and
  RoPE. V gets a head RMS norm without a weight and no RoPE.
  `SparkGemma4LaunchKvStoreSliding` takes both buffers.
- A packed per-row `[K|V]` buffer cannot be used with these kernels.
  `LmHeadRmsNormKernel` and `LmRopePerHeadKernel` address row r at
  `r * heads * head_dimension`, and `LmGqaKvStoreKernel` takes separate key
  and value row pointers. On a packed buffer, row r + 1 would be read from
  row r's V half, which corrupts every row after the first.
- The validator's chain tier (`SparkGemma4ValChainDeviceStages`) uses the
  same split.

### Driver description hash

- `SparkServingAdapterTemplateLoadDriver` rejects a driver with
  `SPARK_STATUS_TARGET_MISMATCH` unless the request contract's
  `model_description_sha256` equals the driver descriptor's. The driver
  compiler sets that value to the SHA-256 of the firmware model-description
  file it compiled (`description->source_sha256`). That file is not the
  package contract, so the package contract hash does not match.
- `spark_qwen38_pp_serving_adapter_common.h` has no default for this value. It
  fails to compile unless the including adapter defines
  `SPARK_QWEN38_SERVING_ADAPTER_DRIVER_DESCRIPTION_SHA256`.
- gemma4 defines it as `GEMMA4_MODEL_DESCRIPTION_SHA256`. The module
  `Makefile` computes it from
  `examples/model_descriptions/gemma4_resident_decode_stage_bf16_firmware.json`.
  The root `Makefile` sets it from `GEMMA4_DESCRIPTION_SHA256` (the same
  file) for the dense TP16 and TP4 adapters, and from
  `GEMMA4_MOE_DESCRIPTION_SHA256`
  (`examples/model_descriptions/gemma4_26b_resident_decode_stage_firmware.json`)
  for the MoE adapter.

### GPU validator (`validation/spark_gemma4_resident_decode_stage_cuda_validation.cu`)

- Single-kernel checks (`SparkGemma4ValCompareBf16`) require relative L2 at
  most 5e-3 and cosine at least 0.999. The attention boundary of the chain
  (`chain_sliding_attention_dataflow`) uses the same bounds.
- The composed chain outputs `chain_sliding_hidden` and
  `chain_sliding_normed` are compared with
  `SparkGemma4ValCompareBf16Threshold` at relative L2 1e-2 (cosine still
  0.999). The difference against the host mirror accumulates over sliding
  attention, output projection, fused residual RMS norm, gate_up projection,
  gated GELU, down projection and a second fused residual RMS norm, each of
  which writes a BF16 result. The bound allows that accumulation and still
  fails on a layout or composition defect.
- `SparkGemma4ValMirrorDecode` takes its row count from the caller.
  Standalone checks pass `SPARK_GEMMA4_VAL_ROWS` (4) and the chain tier
  passes `SPARK_GEMMA4_VAL_CHAIN_ROWS` (2). The caller sizes the output
  buffer, so the row count must match that buffer.
- `gain_device` is shared. `SparkGemma4ValChainDeviceTail` overwrites it
  with the post-attention and post-feedforward gains, so
  `SparkGemma4ValChainDeviceStages` uploads the `input_ln` gains again on
  every run. Without that upload, the determinism rerun would normalize with
  the previous run's tail gains.
