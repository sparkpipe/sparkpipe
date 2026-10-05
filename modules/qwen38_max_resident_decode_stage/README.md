# qwen38_max_resident_decode_stage

CUDA resident decode-stage firmware and serving adapter for Qwen3.8 Max
(`Qwen/Qwen3.8-2.4T-A95B`): 92 layers (69 GDN linear attention, 23 full
attention), FP8 routed experts (`EXPERT_CODEC=fp8` is the only accepted
codec) and BF16 linear weights.

## Implementation notes

### Mesh kernels and the publish marker

- `spark_qwen38_max_resident_decode_stage_cuda.cu` includes
  `model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh`. That
  header defines the mesh launchers called by
  `ring/transport/tp_device_collective.c`, which the module links for its TP
  collectives.
- The header also embeds `SPARK_TP_MESH_KERNELS_MARKER` in the build.
- The `publish` target in `modules/resident_decode_stage_rules.mk` fails with
  `MESH-KERNELS-MARKER-MISSING` unless the module archive contains the marker
  string of the current common header. This catches a module built against a
  private or stale copy of the kernels.

### GDN decay and beta projections

`beta_pre_bf16` and `decay_pre_bf16` are allocated with the full GDN
value-head count per row, not the per-rank count. The decay and beta
projections are replicated, so the linear fills the whole row.
`LmGdnStageDecayBetaKernel` reads this rank's slice, because
`SPARK_LLM_GDN_DECAY_REPLICATED` is 1 for this family.

### Serving descriptor (TP16)

- The deployment has 16 nodes with `stage_index` equal to `rank_index`
  (`deployment/qwen38max-tp16-deploy`).
- `SparkModelResidentDeploymentValidateForAdapter` fails with
  `SPARK_STATUS_TARGET_MISMATCH` unless the deployment `node_count` equals the
  descriptor `stage_count`, so the descriptor has 16 stages.
- The 16 ranks form one parallel group (`parallel_group_size` 16, one PP
  stage) that runs the whole model with tensor-sharded weights. Every
  `stage_layer_counts` entry is the full layer count.
  `SparkDescriptorCheckStageLayerTotals` accepts this in hybrid mode: the 16
  entries of the group are equal and sum, one per group, to the layer count.
- `SparkDescriptorCheckParallelTransportHybridPairing` requires
  `PARALLEL_FANOUT | HIDDEN_TRANSPORT` to be set together with
  `HYBRID_TP_PP`.
- `SparkDescriptorCheckSpeculationPairing` requires `SPECULATION` to be set
  exactly when `max_speculative_token_count` is nonzero. The adapter sets
  `max_speculative_token_count` to the MTP layer count and sets `SPECULATION`
  only when that count is nonzero.
- `SparkSpeculationProviderValidate` rejects a provider whose
  `default_draft_token_count` is 0. The module `Makefile` defaults
  `MTP_LAYER_COUNT` to 0, and with a count of 0
  `SparkQwen38MaxServingBindFamily` binds neither the MTP provider nor the
  speculation seam. `SparkQwen38MaxServingUnbindFamily` calls
  `SparkSpeculationSeamDestroy`, which accepts a null seam.
- `SparkModelServingAdapterValidateInterface` requires non-null `prefetch`,
  `resolve_prefetch` and `reset`, among others:
  - `SparkQwen38MaxServingPrefetch` only validates the submissions;
  - `SparkQwen38MaxServingResolvePrefetch` only validates its arguments and
    the resolution value;
  - `SparkQwen38MaxServingReset` quiesces and then submits a driver admission
    with `SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET`.
- The descriptor claims neither `PREFIX_REUSE` nor `CACHE_PUBLISH`.
  `SparkDescriptorCheckRequiredCacheOperations` refuses such a descriptor
  with `SPARK_STATUS_UNSUPPORTED` when the adapter is loaded.

### Driver description hash

- `SparkServingAdapterTemplateLoadDriver` requires the request contract's
  `model_description_sha256` to equal the driver descriptor's, which the
  driver compiler sets to `description->source_sha256`, the SHA-256 of the
  firmware model-description file. That file is not the package contract.
- `spark_qwen38_pp_serving_adapter_common.h` has no default for this value.
  The adapter defines `SPARK_QWEN38_SERVING_ADAPTER_DRIVER_DESCRIPTION_SHA256`
  as `QWEN38_MAX_MODEL_DESCRIPTION_SHA256`.
- The module `Makefile` computes that hash from
  `examples/model_descriptions/qwen38_max_resident_decode_stage_firmware.json`
  and passes it in `ADAPTER_FLAGS`.

### GPU validator frame context

When the KV tier is active, `SparkQwen38MaxModuleKvPrepareFrame` fails with
`SPARK_STATUS_INVALID_ARGUMENT` if the frame context has no `decode_batch`.
It passes `decode_batch->row_sequence_ids` to `LmKvFramePrepareFrame`. The
validator's `SparkQwen38MaxValModuleExecute` therefore sets
`SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW`
and supplies a decode batch view with the row lane indices, positions and
sequence ids.
