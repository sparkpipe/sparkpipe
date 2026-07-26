#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_MODULE_ID \
    "spark.glm52.sm121.required_decode_stage.b12x_fused.v1"

SparkStatus SparkGlm52Sm121RequiredDecodeStageInitialize(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context);

SparkStatus SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedTensorCoreLinearPlan(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan);

uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateBlackwellNativeQuantizedTensorCoreWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan);

SparkStatus SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedProjectionPlans(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plans,
    uint32_t linear_plan_count);

SparkStatus SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedRegularLinearPlans(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plans,
    uint32_t linear_plan_count);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationQuantize(
    const void *input_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    float *output_amax_f32,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    void *cuda_stream);

uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8E4m3ActivationLinearWorkspaceBytes(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size);

#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_VIEW_ABI_VERSION 1u

typedef struct SparkGlm52Sm121RequiredDecodeStageFp8ActivationLinearWorkspaceView
{
    uint32_t abi_version;
    uint32_t maximum_active_sequence_count;
    uint32_t input_dimension;
    uint32_t scale_block_size;
    uint32_t scale_block_count;
    uint32_t reserved0;
    uint64_t required_workspace_bytes;
    uint8_t *activation_fp8_e4m3;
    float *activation_scale_f32;
    float *activation_amax_f32;
} SparkGlm52Sm121RequiredDecodeStageFp8ActivationLinearWorkspaceView;

SparkStatus SparkGlm52Sm121RequiredDecodeStageResolveFp8E4m3ActivationLinearWorkspace(
    void *workspace,
    uint64_t workspace_bytes,
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    SparkGlm52Sm121RequiredDecodeStageFp8ActivationLinearWorkspaceView *workspace_view_out);


#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_ARGUMENTS_ABI_VERSION 2u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_BACKEND_ABI_VERSION 1u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_STREAM_ORDERED 0x00000001u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_DEVICE_POINTERS 0x00000002u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_BLOCK_SCALES 0x00000004u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_BF16_OUTPUT 0x00000008u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_F32_OUTPUT 0x00000010u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_REQUIRED_CAPABILITIES \
    (SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_STREAM_ORDERED | \
     SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_DEVICE_POINTERS | \
     SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_BLOCK_SCALES)

typedef struct SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmArguments
{
    uint32_t abi_version;
    uint32_t active_sequence_count;
    uint32_t maximum_active_sequence_count;
    uint32_t input_dimension;
    uint32_t output_dimension;
    uint32_t scale_block_size;
    uint32_t output_is_f32;
    uint32_t activation_scale_block_count;
    uint32_t weight_input_scale_block_count;
    uint32_t weight_output_scale_block_count;
    uint32_t reserved0;
    uint32_t reserved1;
    const uint8_t *activation_fp8_e4m3;
    const float *activation_scale_f32;
    const float *activation_amax_f32;
    const uint8_t *weight_fp8_e4m3;
    const float *weight_scale_inv_f32;
    void *output;
    void *workspace;
    uint64_t workspace_bytes;
    void *opaque_state;
    void *cuda_stream;
} SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmArguments;

typedef SparkStatus (*SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmLaunchFunction)(
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmArguments *arguments);

typedef struct SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend
{
    uint32_t abi_version;
    uint32_t capability_flags;
    uint32_t cuda_architecture;
    uint32_t scale_block_size;
    uint32_t minimum_m_alignment;
    uint32_t minimum_n_alignment;
    uint32_t minimum_k_alignment;
    uint32_t reserved0;
    SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmLaunchFunction launch_function;
    void *opaque_state;
    uint64_t required_workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend;

#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_BUILTIN_FP8_SCALED_GEMM_STATE_ABI_VERSION 1u

typedef struct SparkGlm52Sm121RequiredDecodeStageBuiltinFp8ScaledGemmState
{
    uint32_t abi_version;
    uint32_t reserved0;
    void *workspace;
    uint64_t workspace_bytes;
} SparkGlm52Sm121RequiredDecodeStageBuiltinFp8ScaledGemmState;

uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateBuiltinFp8ScaledGemmWorkspaceBytes(void);

SparkStatus SparkGlm52Sm121RequiredDecodeStageInitializeBuiltinFp8ScaledGemmBackend(
    SparkGlm52Sm121RequiredDecodeStageBuiltinFp8ScaledGemmState *state,
    void *workspace,
    uint64_t workspace_bytes,
    SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend_out);

uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8E4m3ActivationLinearBackendWorkspaceBytes(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    uint64_t backend_workspace_bytes);

SparkStatus SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearScaledGemmBackend(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend);

SparkStatus SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearPlansScaledGemmBackend(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plans,
    uint32_t linear_plan_count,
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationWeightLinearScaledGemmBackend(
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend,
    const void *input_bf16,
    const uint8_t *weight_fp8_e4m3,
    const float *weight_scale_inv_f32,
    void *workspace,
    uint64_t workspace_bytes,
    void *output,
    uint32_t active_sequence_count,
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t scale_block_size,
    uint32_t output_is_f32,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3KvCacheStore(
    const void *input_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    uint32_t token_count,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3KvCacheLoad(
    const uint8_t *input_fp8_e4m3,
    const float *input_scale_f32,
    void *output_bf16,
    uint32_t token_count,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3MappedActiveRowsKvCacheStore(
    const void *input_bf16_active_rows,
    const uint32_t *slot_mapping,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    uint32_t active_sequence_count,
    uint32_t cache_token_capacity,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3MappedActiveRowsKvCacheStoreTriple(
    const void *mla_bf16_active_rows,
    const void *raw_kv_b_bf16_active_rows,
    const uint32_t *slot_mapping,
    uint8_t *mla_cache_fp8_e4m3,
    float *mla_cache_scale_f32,
    uint8_t *key_nope_cache_fp8_e4m3,
    float *key_nope_cache_scale_f32,
    uint8_t *value_cache_fp8_e4m3,
    float *value_cache_scale_f32,
    uint32_t active_sequence_count,
    uint32_t cache_token_capacity,
    uint32_t mla_element_count,
    uint32_t key_nope_element_count,
    uint32_t value_element_count,
    uint32_t raw_kv_b_element_count,
    uint32_t scale_block_size,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3MappedKvCacheStoreTriple(
    const void *mla_cache_bf16,
    const void *key_nope_cache_bf16,
    const void *value_cache_bf16,
    const uint32_t *slot_mapping,
    uint8_t *mla_cache_fp8_e4m3,
    float *mla_cache_scale_f32,
    uint8_t *key_nope_cache_fp8_e4m3,
    float *key_nope_cache_scale_f32,
    uint8_t *value_cache_fp8_e4m3,
    float *value_cache_scale_f32,
    uint32_t active_sequence_count,
    uint32_t cache_token_capacity,
    uint32_t mla_element_count,
    uint32_t key_nope_element_count,
    uint32_t value_element_count,
    uint32_t scale_block_size,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchRmsNormFp8E4m3ActivationQuantize(
    const void *input_bf16,
    const void *weight_bf16,
    void *output_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    float *output_amax_f32,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    float epsilon,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchSiluMulFp8E4m3ActivationQuantize(
    const void *gate_bf16,
    const void *up_bf16,
    void *output_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    float *output_amax_f32,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    void *cuda_stream);

#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_MOE_PACKED_ROUTE_VIEW_ABI_VERSION 2u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_MOE_PACKED_ROUTE_WORKSPACE_ALIGNMENT_BYTES 256ull

typedef struct SparkGlm52Sm121RequiredDecodeStageMoePackedRouteView
{
    uint32_t abi_version;
    uint32_t maximum_token_count;
    uint32_t active_sequence_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t maximum_route_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t *expert_route_offsets;
    uint32_t *expert_route_counts;
    uint32_t *expert_route_write_cursors;
    uint32_t *packed_expert_ids;
    uint32_t *packed_source_token_indices;
    uint32_t *packed_source_route_indices;
    uint32_t *packed_route_rows_by_token_route;
    float *packed_route_weights;
    uint32_t *packed_route_count;
} SparkGlm52Sm121RequiredDecodeStageMoePackedRouteView;

uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateMoePackedRouteWorkspaceBytes(
    uint32_t maximum_token_count,
    uint32_t top_k,
    uint32_t expert_count);

SparkStatus SparkGlm52Sm121RequiredDecodeStageResolveMoePackedRouteWorkspace(
    uint32_t maximum_token_count,
    uint32_t top_k,
    uint32_t expert_count,
    void *workspace,
    uint64_t workspace_bytes,
    SparkGlm52Sm121RequiredDecodeStageMoePackedRouteView *packed_route_view_out);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchMoePackedRouteBuild(
    const uint32_t *topk_expert_ids,
    const float *topk_weights,
    SparkGlm52Sm121RequiredDecodeStageMoePackedRouteView *packed_route_view,
    uint32_t active_sequence_count,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoePackedHiddenQuantize(
    const void *hidden_bf16,
    const uint32_t *packed_route_rows_by_token_route,
    uint8_t *packed_hidden_fp8_e4m3,
    float *packed_hidden_scale_f32,
    float *packed_hidden_amax_f32,
    uint32_t routed_row_count,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t scale_block_size,
    void *cuda_stream);


#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_ARGUMENTS_ABI_VERSION 2u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_ABI_VERSION 1u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_STREAM_ORDERED 0x00000001u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_DEVICE_POINTERS 0x00000002u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_EXPERT_MAJOR_PACKED_ROWS 0x00000004u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_DYNAMIC_ACTIVATION_SCALES 0x00000008u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_DETERMINISTIC_FINALIZE 0x00000010u
#define SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_REQUIRED_CAPABILITIES \
    (SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_STREAM_ORDERED | \
     SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_DEVICE_POINTERS | \
     SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_EXPERT_MAJOR_PACKED_ROWS | \
     SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_DYNAMIC_ACTIVATION_SCALES | \
     SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_CAPABILITY_DETERMINISTIC_FINALIZE)

typedef struct SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackendArguments
{
    uint32_t abi_version;
    uint32_t active_sequence_count;
    uint32_t routed_row_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t scale_block_size;
    uint32_t output_dtype;
    uint32_t hidden_scale_block_count;
    uint32_t intermediate_scale_block_count;
    uint32_t w1_output_scale_block_count;
    uint32_t w2_output_scale_block_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    const SparkGlm52Sm121RequiredDecodeStageMoePackedRouteView *packed_route_view;
    const uint8_t *packed_hidden_fp8_e4m3;
    const float *packed_hidden_scale_f32;
    const float *packed_hidden_amax_f32;
    const uint8_t *w1_weight_fp8_e4m3;
    const float *w1_scale_inv_f32;
    const uint8_t *w2_weight_fp8_e4m3;
    const float *w2_scale_inv_f32;
    void *output_bf16;
    void *workspace;
    uint64_t workspace_bytes;
    void *opaque_state;
    void *cuda_stream;
} SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackendArguments;

typedef SparkStatus (*SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackendLaunchFunction)(
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackendArguments *arguments);

typedef struct SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend
{
    uint32_t abi_version;
    uint32_t capability_flags;
    uint32_t cuda_architecture;
    uint32_t scale_block_size;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackendLaunchFunction launch_function;
    void *opaque_state;
    uint64_t required_workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend;

uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedExternalBackendWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *backend);

SparkStatus SparkGlm52Sm121RequiredDecodeStageBindFp8MoeGroupedExternalBackendPlan(
    SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *backend);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedExternalBackend(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan);

SparkStatus SparkGlm52Sm121RequiredDecodeStageBindFp8MoeGroupedReferencePlan(
    SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedReference(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchBlackwellQuantizedTensorCoreLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    const void *weight,
    void *output,
    uint32_t active_sequence_count,
    void *cuda_stream);


SparkStatus SparkGlm52Sm121RequiredDecodeStageBindDsaKvFragmentTransportPlan(
    SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan *transport_plan);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaSelectedBlockBuild(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const uint32_t *selected_token_indices,
    const uint32_t *context_lengths,
    const uint32_t *positions,
    const uint32_t *first_block_token_offsets,
    uint32_t *selected_block_indices,
    uint32_t *selected_block_counts,
    uint32_t *selection_epoch_by_layer,
    uint32_t layer_index,
    uint32_t active_sequence_count,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaSelectedKvFragmentPrefetch(
    const SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan *prefetch_plan,
    const uint32_t *selected_block_indices,
    const uint32_t *selected_block_counts,
    const uint32_t *block_table,
    uint32_t active_sequence_count,
    uint32_t selected_block_stride,
    uint32_t selected_block_capacity,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    void *producer_cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaSelectedKvFragmentSave(
    const SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan *save_plan,
    const uint32_t *selected_block_indices,
    const uint32_t *selected_block_counts,
    const uint32_t *block_table,
    uint32_t active_sequence_count,
    uint32_t selected_block_stride,
    uint32_t selected_block_capacity,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    void *producer_cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaKvFragmentSaveWrittenSlots(
    const SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan *save_plan,
    const uint32_t *positions,
    const uint32_t *first_block_token_offsets,
    const uint32_t *block_table,
    uint32_t active_sequence_count,
    uint32_t kv_block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    void *producer_cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageQueryDsaSelectedKvFragmentPrefetch(
    const SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan *prefetch_plan);

SparkStatus SparkGlm52Sm121RequiredDecodeStageWaitForDsaSelectedKvFragmentPrefetch(
    const SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan *prefetch_plan,
    void *consumer_cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareExportSelectedTokens(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t source_layer_index,
    void *selected_token_sideband,
    uint32_t active_sequence_count,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareImportSelectedTokens(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const void *selected_token_sideband,
    uint32_t source_layer_index,
    uint32_t active_sequence_count,
    void *cuda_stream);


SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaKeyIndexBlockSummaryBuild(
    const void *key_index_cache_bf16,
    const void *dirty_block_flags,
    void *key_index_block_min_bf16,
    void *key_index_block_max_bf16,
    uint32_t physical_block_count,
    uint32_t block_token_count,
    uint32_t cache_token_capacity,
    uint32_t index_head_dimension,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareBlockUpperBoundMask(
    const void *query_index_heads_bf16,
    const void *index_head_weights_bf16,
    const void *key_index_block_min_bf16,
    const void *key_index_block_max_bf16,
    const uint32_t *block_table,
    const uint32_t *context_lengths,
    const uint32_t *first_block_token_offsets,
    const float *minimum_required_scores_f32,
    float *block_upper_bounds_f32,
    uint8_t *candidate_block_flags_u8,
    uint32_t *candidate_block_counts,
    uint32_t active_sequence_count,
    uint32_t logical_block_capacity,
    uint32_t index_head_count,
    uint32_t index_head_dimension,
    uint32_t block_token_count,
    uint32_t kv_block_count,
    float index_softmax_scale,
    float conservative_score_epsilon,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareSelectTopkFromScores(
    float *dsa_token_scores,
    const uint32_t *context_lengths,
    uint32_t *sparse_token_indices,
    uint32_t active_sequence_count,
    uint32_t dsa_candidate_count,
    uint32_t selected_token_count,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareScoreTopk(
    const void *query_index_heads_bf16,
    const void *key_index_cache_bf16,
    const void *index_head_weights_bf16,
    const uint32_t *block_table,
    const uint32_t *context_lengths,
    const uint32_t *first_block_token_offsets,
    float *dsa_score_tiles_f32,
    uint32_t *sparse_token_indices,
    uint32_t active_sequence_count,
    uint32_t dsa_candidate_count,
    uint32_t dsa_score_row_capacity,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float index_softmax_scale,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareDecodeSelection(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaKeyIndexCacheStore(
    const void *raw_key_index_bf16,
    const void *key_norm_weight_bf16,
    const void *key_norm_bias_bf16,
    const uint32_t *positions,
    const uint32_t *slot_mapping,
    const float *cos_table,
    const float *sin_table,
    void *key_index_cache_bf16,
    uint32_t row_count,
    uint32_t position_count,
    uint32_t cache_token_capacity,
    float epsilon,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFullVocabGreedy(
    const void *hidden_bf16,
    const void *norm_weight_bf16,
    void *normalized_hidden_bf16,
    const void *lm_head_weight_bf16,
    const uint32_t *token_ids,
    float *logits_f32,
    uint32_t *selected_token_ids,
    float *selected_token_scores,
    void *workspace,
    uint64_t workspace_bytes,
    uint32_t active_sequence_count,
    uint32_t maximum_active_sequence_count,
    float rms_norm_epsilon,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunch(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchStageSlice(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    void *cuda_stream,
    void *backend_completion);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchExactPp13StageSlice(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    void *cuda_stream,
    void *backend_completion);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchBulkPrefill(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchPagedChunkPrefill(
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    void *cuda_stream);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchStageSliceBulkPrefill(
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    void *cuda_stream);

void SparkGlm52Sm121RequiredDecodeStageQuiesce(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context);

#ifdef __cplusplus
}
#endif
