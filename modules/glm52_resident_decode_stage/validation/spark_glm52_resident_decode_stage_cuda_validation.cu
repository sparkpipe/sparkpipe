#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>

#include <atomic>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "spark_glm52_resident_decode_stage_backend.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_b12x_moe_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_linear_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_orchestrator.h"
#include "sparkpipe/spark_status.h"

#ifndef SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT
#define SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT 1u
#endif
#if SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT == 0u
#error "SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT must be positive"
#endif
#define SPARK_VALIDATION_CACHE_TOKEN_CAPACITY 128u
#define SPARK_VALIDATION_KV_BLOCK_COUNT 2u
#define SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE 2u
#define SPARK_VALIDATION_POSITION_COUNT 128u
#define SPARK_VALIDATION_CONTEXT_LENGTH 4u
#define SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT 3u
#define SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX 3u
#define SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT 8u
#define SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT 6u
#define SPARK_VALIDATION_LAYER_COUNT 78u
#define SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET 61u
#define SPARK_VALIDATION_CURRENT_POSITION 64u
#define SPARK_VALIDATION_CURRENT_CACHE_SLOT 0u
#define SPARK_VALIDATION_REMAP_CACHE_SLOT0 125u
#define SPARK_VALIDATION_REMAP_CACHE_SLOT1 126u
#define SPARK_VALIDATION_REMAP_CACHE_SLOT2 127u
#define SPARK_VALIDATION_CHECKED_HEAD_COUNT 4u
#define SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT 8u
#define SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT 4u
#define SPARK_VALIDATION_MOE_BOUND_EXPERT_COUNT 8u
#define SPARK_VALIDATION_MOE_CHECKED_INTERMEDIATE 4u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID 233u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_0 233u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_1 41u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_2 166u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_3 174u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_4 186u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_5 37u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_6 117u
#define SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_7 223u
#define SPARK_VALIDATION_MOE_ROUTE_COUNT \
    (SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K)
#define SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN 1009u
#define SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN 1011u
#define SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN 1003u
#define SPARK_VALIDATION_MEASUREMENT_COUNT 3u
#define SPARK_VALIDATION_WARMUP_COUNT 1u
#define SPARK_VALIDATION_ATTENTION_TOLERANCE 0.030f
#define SPARK_VALIDATION_LOGIT_TOLERANCE 0.010f
#define SPARK_VALIDATION_REFERENCE_RELATIVE_TOLERANCE 0.020f
#define SPARK_VALIDATION_REFERENCE_ABSOLUTE_TOLERANCE 0.0625f
#define SPARK_VALIDATION_RESTRICTED_LM_HEAD_FIRST_TOKEN 1000u
#define SPARK_VALIDATION_LOGIT_REDUCTION_THREADS 256u
#define SPARK_VALIDATION_SAFETENSORS_HEADER_MAX_BYTES (128ull * 1024ull * 1024ull)
#define SPARK_VALIDATION_TENSOR_NAME_BYTES 256u
#define SPARK_VALIDATION_FP8_MOE_PACK_HEADER_BYTES 512u
#define SPARK_VALIDATION_FP8_MOE_PACK_REGION_COUNT 4u
#define SPARK_VALIDATION_FP8_MOE_COPY_CHUNK_BYTES (64ull * 1024ull * 1024ull)
#define SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_NVFP4 0u
#define SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_FP8 1u

typedef struct SparkValidationCompletionState
{
    std::atomic<uint32_t> completion_count;
} SparkValidationCompletionState;

typedef struct SparkValidationDriverCompletionState
{
    std::atomic<uint32_t> completion_count;
    SparkModelDriverCompletion completion;
} SparkValidationDriverCompletionState;

static bool SparkValidationPreflightRequiredFastPath(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context);

typedef struct SparkValidationRealLmHeadFixture
{
    uint16_t *restricted_rows_bf16;
    uint64_t restricted_row_bytes;
    uint32_t ready;
    uint32_t expected_selected_token;
    float expected_selected_score;
    float maximum_logit_error;
} SparkValidationRealLmHeadFixture;

typedef struct SparkValidationLayer0DenseBf16Fixture
{
    uint64_t copied_bytes;
    uint32_t ready;
} SparkValidationLayer0DenseBf16Fixture;

typedef struct SparkValidationLayer0AttentionBf16Fixture
{
    uint64_t copied_bytes;
    uint32_t ready;
} SparkValidationLayer0AttentionBf16Fixture;

typedef struct SparkValidationInputEmbeddingBf16Fixture
{
    uint64_t copied_bytes;
    uint32_t token_id;
    uint32_t ready;
} SparkValidationInputEmbeddingBf16Fixture;

typedef struct SparkValidationFinalNormBf16Fixture
{
    uint64_t copied_bytes;
    uint32_t ready;
} SparkValidationFinalNormBf16Fixture;

typedef struct SparkValidationPrefillKvBf16Fixture
{
    uint64_t copied_bytes;
    uint32_t first_token_id;
    uint32_t token_count;
    uint32_t ready;
} SparkValidationPrefillKvBf16Fixture;

typedef struct SparkValidationPromptTokenWindow
{
    uint32_t token_ids[SPARK_VALIDATION_CONTEXT_LENGTH];
    uint32_t token_count;
    uint32_t source_token_count;
    uint32_t ready;
} SparkValidationPromptTokenWindow;

typedef struct SparkValidationLayer3RouterBf16Fixture
{
    uint64_t copied_bytes;
    uint32_t ready;
} SparkValidationLayer3RouterBf16Fixture;

typedef struct SparkValidationLayer3SharedExpertBf16Fixture
{
    uint64_t copied_bytes;
    uint32_t ready;
} SparkValidationLayer3SharedExpertBf16Fixture;

typedef struct SparkValidationLayer3RoutedExpertNvfp4Fixture
{
    float gate_input_scale;
    float gate_weight_scale_2;
    float up_input_scale;
    float up_weight_scale_2;
    float down_input_scale;
    float down_weight_scale_2;
    float gate_input_scales[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float gate_weight_scale_2_values[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float up_input_scales[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float up_weight_scale_2_values[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float down_input_scales[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float down_weight_scale_2_values[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    uint64_t copied_bytes;
    uint32_t selected_expert_id;
    uint32_t bound_expert_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    uint32_t bound_expert_count;
    uint32_t ready;
} SparkValidationLayer3RoutedExpertNvfp4Fixture;

typedef struct SparkValidationDeviceBuffers
{
    uint16_t *input_hidden_bf16;
    uint16_t *normalized_hidden_bf16;
    uint16_t *query_latent_bf16;
    uint16_t *query_rope_input_bf16;
    uint16_t *key_rope_input_bf16;
    uint16_t *current_kv_latent_bf16;
    uint16_t *raw_query_a_bf16;
    uint16_t *raw_query_a_normalized_bf16;
    uint16_t *raw_query_b_bf16;
    uint16_t *raw_kv_a_bf16;
    uint16_t *raw_kv_a_normalized_bf16;
    uint16_t *raw_kv_b_bf16;
    uint16_t *mla_cache_bf16;
    uint16_t *key_nope_cache_bf16;
    uint16_t *value_cache_bf16;
    uint16_t *dense_layer_mla_cache_bf16[SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT];
    uint16_t *dense_layer_key_nope_cache_bf16[SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT];
    uint16_t *dense_layer_value_cache_bf16[SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT];
    uint16_t *routed_layer_mla_cache_bf16[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint16_t *routed_layer_key_nope_cache_bf16[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint16_t *routed_layer_value_cache_bf16[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint16_t *rotated_query_rope_bf16;
    uint16_t *attention_output_latent_bf16;
    uint16_t *attention_projected_hidden_bf16;
    uint16_t *post_attention_hidden_bf16;
    uint16_t *post_attention_normalized_hidden_bf16;
    uint16_t *moe_gate_bf16;
    uint16_t *moe_up_bf16;
    uint16_t *moe_intermediate_bf16;
    uint16_t *moe_route_output_bf16;
    uint16_t *layer_output_hidden_bf16;
    uint16_t *mtp_draft_hidden_bf16;
    uint16_t *attention_norm_weight_bf16;
    uint16_t *query_latent_weight_bf16;
    uint16_t *query_rope_weight_bf16;
    uint16_t *key_rope_weight_bf16;
    uint16_t *kv_latent_weight_bf16;
    uint16_t *raw_query_a_weight_bf16;
    uint16_t *raw_query_a_norm_weight_bf16;
    uint16_t *raw_query_b_weight_bf16;
    uint16_t *raw_kv_a_weight_bf16;
    uint16_t *raw_kv_a_norm_weight_bf16;
    uint16_t *raw_kv_b_weight_bf16;
    uint8_t *raw_query_a_weight_fp8_e4m3;
    float *raw_query_a_weight_scale_inv_f32;
    uint8_t *raw_query_b_weight_fp8_e4m3;
    float *raw_query_b_weight_scale_inv_f32;
    uint8_t *raw_kv_a_weight_fp8_e4m3;
    float *raw_kv_a_weight_scale_inv_f32;
    uint8_t *raw_kv_b_weight_fp8_e4m3;
    float *raw_kv_b_weight_scale_inv_f32;
    uint16_t *attention_output_weight_bf16;
    uint8_t *attention_output_weight_fp8_e4m3;
    float *attention_output_weight_scale_inv_f32;
    uint16_t *post_attention_norm_weight_bf16;
    uint16_t *dense_gate_weight_bf16;
    uint16_t *dense_up_weight_bf16;
    uint16_t *dense_down_weight_bf16;
    uint16_t *moe_router_weight_bf16;
    uint8_t *routed_gate_weight_payload_u8;
    uint8_t *routed_up_weight_payload_u8;
    uint8_t *routed_down_weight_payload_u8;
    uint8_t *routed_gate_weight_scale_e4m3;
    uint8_t *routed_up_weight_scale_e4m3;
    uint8_t *routed_down_weight_scale_e4m3;
    uint32_t *routed_bound_expert_ids;
    float *routed_gate_input_scale_f32;
    float *routed_gate_weight_scale_2_f32;
    float *routed_up_input_scale_f32;
    float *routed_up_weight_scale_2_f32;
    float *routed_down_input_scale_f32;
    float *routed_down_weight_scale_2_f32;
    uint16_t *final_norm_weight_bf16;
    uint16_t *restricted_lm_head_weight_bf16;
    uint8_t *mtp_mxfp4_weight_payload_u8;
    uint8_t *mtp_mxfp4_scale_e8m0_u8;
    float *cos_table;
    float *sin_table;
    float *dsa_token_scores;
    float *moe_router_score_bias_f32;
    float *moe_router_logits;
    float *moe_topk_weights;
    float *restricted_logits;
    float *restricted_selected_token_scores;
    float *mtp_draft_logits;
    uint32_t *positions;
    uint32_t *slot_mapping;
    uint32_t *block_table;
    uint32_t *context_lengths;
    uint32_t *first_block_token_offsets;
    uint32_t *sparse_token_indices;
    uint32_t *restricted_token_ids;
    uint32_t *moe_topk_expert_ids;
    uint32_t *restricted_selected_token_ids;
    uint32_t *mtp_draft_token_ids;
    uint32_t *mtp_target_token_ids;
    uint32_t *mtp_accept_mask;
    uint32_t *mtp_committed_token_ids;
    uint32_t *mtp_event_counters;
    uint64_t *phase_clock_cycles;
    SparkGlm52ResidentDecodeStageB12xMoeResidentBinding
        b12x_moe_bindings[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint32_t b12x_moe_binding_ready[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint32_t b12x_moe_binding_layer_indices[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    SparkGlm52ResidentDecodeStageFp8MoePlan
        fp8_moe_plans[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint8_t *fp8_moe_w1_weight_fp8_e4m3[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    float *fp8_moe_w1_scale_inv_f32[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint8_t *fp8_moe_w2_weight_fp8_e4m3[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    float *fp8_moe_w2_scale_inv_f32[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    void *fp8_moe_workspace[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint32_t fp8_moe_plan_ready[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint32_t fp8_moe_plan_layer_indices[SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT];
    uint32_t routed_layer_base_index;
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *linear_plan_binding;
} SparkValidationDeviceBuffers;

typedef struct SparkValidationFp8MoePackHeader
{
    uint8_t magic[16];
    uint32_t fields[16];
    uint64_t region_offsets[SPARK_VALIDATION_FP8_MOE_PACK_REGION_COUNT];
    uint64_t region_bytes[SPARK_VALIDATION_FP8_MOE_PACK_REGION_COUNT];
} SparkValidationFp8MoePackHeader;

typedef struct SparkValidationExactPp13StageSliceRuntime
{
    SparkValidationDeviceBuffers buffers[
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT];
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slots[
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT];
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState cuda_slot_states[
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT];
    SparkGlm52ResidentDecodeStageNodeContext node_contexts[
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT];
    const SparkGlm52ResidentDecodeStageNodeContext *node_context_pointers[
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT];
    SparkGlm52ResidentDecodeStageStageSlicePlan stage_slice_plan;
    SparkGlm52ResidentDecodeStageExactStageSlicePlan exact_stage_slice_plan;
    cudaStream_t query_branch_stream;
    cudaStream_t kv_branch_stream;
    cudaEvent_t branch_ready_event;
    cudaEvent_t query_branch_event;
    cudaEvent_t kv_branch_event;
    void *final_epilogue_workspace;
    uint64_t final_epilogue_workspace_bytes;
} SparkValidationExactPp13StageSliceRuntime;

static bool SparkValidationCudaSucceeded(
    cudaError_t cuda_status,
    const char *operation)
{
    if (cuda_status == cudaSuccess)
    {
        return true;
    }
    fprintf(
        stderr,
        "%s failed: %s\n",
        operation,
        cudaGetErrorString(cuda_status));
    return false;
}

static bool SparkValidationAllocateZeroed(
    void **device_pointer,
    uint64_t byte_count,
    const char *name)
{
    if (!SparkValidationCudaSucceeded(
            cudaMalloc(device_pointer, (size_t)byte_count),
            name))
    {
        return false;
    }
    return SparkValidationCudaSucceeded(
        cudaMemset(*device_pointer, 0, (size_t)byte_count),
        name);
}

static bool SparkValidationAllocateDeviceOnly(
    void **device_pointer,
    uint64_t byte_count,
    const char *name)
{
    return SparkValidationCudaSucceeded(
        cudaMalloc(device_pointer, (size_t)byte_count),
        name);
}

static bool SparkValidationCopyToDevice(
    void *device_pointer,
    const void *host_pointer,
    uint64_t byte_count,
    const char *name)
{
    return SparkValidationCudaSucceeded(
        cudaMemcpy(
            device_pointer,
            host_pointer,
            (size_t)byte_count,
            cudaMemcpyHostToDevice),
        name);
}

static bool SparkValidationCopyDeviceToDevice(
    void *destination_pointer,
    const void *source_pointer,
    uint64_t byte_count,
    const char *name)
{
    return SparkValidationCudaSucceeded(
        cudaMemcpy(
            destination_pointer,
            source_pointer,
            (size_t)byte_count,
            cudaMemcpyDeviceToDevice),
        name);
}

static uint16_t SparkValidationFloatToBf16(float value)
{
    union
    {
        uint32_t bits;
        float value;
    } conversion;
    uint32_t rounding_bias;

    conversion.value = value;
    rounding_bias = 0x7fffu + ((conversion.bits >> 16u) & 1u);
    return (uint16_t)((conversion.bits + rounding_bias) >> 16u);
}

static float SparkValidationBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t bits;
        float value;
    } conversion;

    conversion.bits = ((uint32_t)value) << 16u;
    return conversion.value;
}

static bool SparkValidationBuildModelPath(
    const char *model_directory,
    const char *leaf_name,
    char *path,
    uint32_t path_bytes)
{
    int written_bytes;

    written_bytes = snprintf(path, (size_t)path_bytes, "%s/%s", model_directory, leaf_name);
    return written_bytes >= 0 && (uint32_t)written_bytes < path_bytes;
}

static bool SparkValidationBuildLayerTensorName(
    char *tensor_name,
    uint32_t tensor_name_bytes,
    uint32_t layer_index,
    const char *suffix)
{
    int written_bytes;

    written_bytes = snprintf(
        tensor_name,
        (size_t)tensor_name_bytes,
        "model.layers.%u.%s",
        layer_index,
        suffix);
    return written_bytes >= 0 && (uint32_t)written_bytes < tensor_name_bytes;
}

static const char *SparkValidationStagePackDirectory(void)
{
    const char *stage_pack_directory;

    stage_pack_directory = getenv("GLM52_STAGE_PACK_DIR");
    if (stage_pack_directory == 0 || stage_pack_directory[0] == '\0')
        return 0;
    return stage_pack_directory;
}

static bool SparkValidationBuildStagePackPath(
    const char *stage_pack_directory,
    const char *leaf_name,
    char *path,
    uint32_t path_bytes)
{
    int written_bytes;

    written_bytes = snprintf(path, (size_t)path_bytes, "%s/%s", stage_pack_directory, leaf_name);
    return written_bytes >= 0 && (uint32_t)written_bytes < path_bytes;
}

static bool SparkValidationReadStagePackTypedTensorOffsets(
    const char *tensor_name,
    const char *expected_dtype,
    uint64_t bytes_per_element,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    char *tensor_path,
    uint32_t tensor_path_bytes,
    uint64_t *payload_file_offset,
    uint64_t *tensor_bytes)
{
    SparkJsonDocument document;
    const char *stage_pack_directory;
    char index_path[4096];
    char *file_name;
    uint64_t observed_bytes;
    uint64_t expected_bytes;
    uint32_t dimension_index;
    int32_t root_token_index;
    int32_t tensor_map_token_index;
    int32_t tensor_token_index;
    int32_t file_token_index;
    int32_t dtype_token_index;
    int32_t shape_token_index;
    int32_t offset_token_index;
    int32_t bytes_token_index;
    SparkStatus status;
    bool succeeded;

    stage_pack_directory = SparkValidationStagePackDirectory();
    if (stage_pack_directory == 0)
        return false;
    file_name = 0;
    if (!SparkValidationBuildStagePackPath(
            stage_pack_directory,
            "stagepack_index.json",
            index_path,
            sizeof(index_path)))
    {
        fprintf(stderr, "stage pack index path is too long\n");
        return false;
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonLoadFile(index_path, &document);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "could not load stage pack index %s: %s\n", index_path, SparkStatusToString(status));
        return false;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    tensor_map_token_index = SparkJsonFindObjectMember(&document, root_token_index, "tensor_map");
    tensor_token_index = tensor_map_token_index >= 0
        ? SparkJsonFindObjectMember(&document, tensor_map_token_index, tensor_name)
        : -1;
    file_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "file") : -1;
    dtype_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "dtype") : -1;
    shape_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "shape") : -1;
    offset_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "offset") : -1;
    bytes_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "bytes") : -1;
    succeeded = tensor_token_index >= 0 &&
        file_token_index >= 0 &&
        SparkJsonCopyString(&document, file_token_index, &file_name) == SPARK_STATUS_OK &&
        dtype_token_index >= 0 &&
        SparkJsonStringEquals(&document, dtype_token_index, expected_dtype) &&
        shape_token_index >= 0 &&
        SparkJsonGetArrayElementCount(&document, shape_token_index) == expected_rank &&
        offset_token_index >= 0 &&
        bytes_token_index >= 0 &&
        SparkJsonGetUInt64(&document, offset_token_index, payload_file_offset) == SPARK_STATUS_OK &&
        SparkJsonGetUInt64(&document, bytes_token_index, &observed_bytes) == SPARK_STATUS_OK;
    expected_bytes = bytes_per_element;
    for (dimension_index = 0u; succeeded && dimension_index < expected_rank; ++dimension_index)
    {
        uint64_t observed_dimension;

        succeeded =
            SparkJsonGetUInt64(
                &document,
                SparkJsonGetArrayElement(&document, shape_token_index, dimension_index),
                &observed_dimension) == SPARK_STATUS_OK &&
            observed_dimension == expected_shape[dimension_index] &&
            expected_shape[dimension_index] != 0u &&
            expected_bytes <= UINT64_MAX / expected_shape[dimension_index];
        if (succeeded)
            expected_bytes *= expected_shape[dimension_index];
    }
    succeeded = succeeded &&
        observed_bytes == expected_bytes &&
        SparkValidationBuildStagePackPath(
            stage_pack_directory,
            file_name != 0 ? file_name : "",
            tensor_path,
            tensor_path_bytes);
    free(file_name);
    SparkJsonDocumentDestroy(&document);
    if (!succeeded)
    {
        fprintf(stderr, "%s is missing or malformed in stage pack %s\n", tensor_name, index_path);
        return false;
    }
    *tensor_bytes = observed_bytes;
    return true;
}

static bool SparkValidationReadSafetensorsHeader(
    const char *path,
    char **header_text,
    uint64_t *header_bytes)
{
    FILE *file;
    uint8_t header_length_bytes[8];
    uint64_t header_length;
    uint32_t byte_index;

    *header_text = 0;
    *header_bytes = 0u;
    file = fopen(path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "could not open safetensors file %s\n", path);
        return false;
    }
    if (fread(header_length_bytes, 1u, sizeof(header_length_bytes), file) != sizeof(header_length_bytes))
    {
        fprintf(stderr, "could not read safetensors header length from %s\n", path);
        fclose(file);
        return false;
    }
    header_length = 0u;
    for (byte_index = 0u; byte_index < 8u; ++byte_index)
    {
        header_length |= ((uint64_t)header_length_bytes[byte_index]) << (8u * byte_index);
    }
    if (header_length == 0u || header_length > SPARK_VALIDATION_SAFETENSORS_HEADER_MAX_BYTES)
    {
        fprintf(stderr, "safetensors header length is unsupported for %s\n", path);
        fclose(file);
        return false;
    }
    *header_text = (char *)malloc((size_t)header_length + 1u);
    if (*header_text == 0)
    {
        fprintf(stderr, "could not allocate safetensors header buffer\n");
        fclose(file);
        return false;
    }
    if (fread(*header_text, 1u, (size_t)header_length, file) != (size_t)header_length)
    {
        fprintf(stderr, "could not read safetensors header from %s\n", path);
        free(*header_text);
        *header_text = 0;
        fclose(file);
        return false;
    }
    fclose(file);
    (*header_text)[header_length] = '\0';
    *header_bytes = header_length;
    return true;
}

static bool SparkValidationReadLmHeadShardName(
    const char *model_directory,
    char **file_name)
{
    SparkJsonDocument document;
    char index_path[4096];
    int32_t root_token_index;
    int32_t weight_map_token_index;
    int32_t file_token_index;
    SparkStatus status;
    bool succeeded;

    *file_name = 0;
    if (!SparkValidationBuildModelPath(model_directory, "model.safetensors.index.json", index_path, sizeof(index_path)))
    {
        fprintf(stderr, "safetensors index path is too long\n");
        return false;
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonLoadFile(index_path, &document);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "could not load safetensors index %s: %s\n", index_path, SparkStatusToString(status));
        return false;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    weight_map_token_index = SparkJsonFindObjectMember(&document, root_token_index, "weight_map");
    file_token_index = weight_map_token_index >= 0
        ? SparkJsonFindObjectMember(&document, weight_map_token_index, "lm_head.weight")
        : -1;
    succeeded = file_token_index >= 0 &&
        SparkJsonCopyString(&document, file_token_index, file_name) == SPARK_STATUS_OK;
    if (!succeeded)
    {
        fprintf(stderr, "lm_head.weight is missing from safetensors index\n");
    }
    SparkJsonDocumentDestroy(&document);
    return succeeded;
}

static bool SparkValidationReadTensorShardName(
    const char *model_directory,
    const char *tensor_name,
    char **file_name)
{
    SparkJsonDocument document;
    char index_path[4096];
    int32_t root_token_index;
    int32_t weight_map_token_index;
    int32_t file_token_index;
    SparkStatus status;
    bool succeeded;

    *file_name = 0;
    if (!SparkValidationBuildModelPath(model_directory, "model.safetensors.index.json", index_path, sizeof(index_path)))
    {
        fprintf(stderr, "safetensors index path is too long\n");
        return false;
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonLoadFile(index_path, &document);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "could not load safetensors index %s: %s\n", index_path, SparkStatusToString(status));
        return false;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    weight_map_token_index = SparkJsonFindObjectMember(&document, root_token_index, "weight_map");
    file_token_index = weight_map_token_index >= 0
        ? SparkJsonFindObjectMember(&document, weight_map_token_index, tensor_name)
        : -1;
    succeeded = file_token_index >= 0 &&
        SparkJsonCopyString(&document, file_token_index, file_name) == SPARK_STATUS_OK;
    if (!succeeded)
    {
        fprintf(stderr, "%s is missing from safetensors index\n", tensor_name);
    }
    SparkJsonDocumentDestroy(&document);
    return succeeded;
}

static bool SparkValidationReadTypedTensorOffsets(
    const char *tensor_path,
    const char *tensor_name,
    const char *expected_dtype,
    uint64_t bytes_per_element,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    uint64_t *payload_file_offset,
    uint64_t *tensor_bytes)
{
    SparkJsonDocument document;
    char *header_text;
    uint64_t header_bytes;
    uint64_t start_offset;
    uint64_t end_offset;
    uint64_t expected_bytes;
    int32_t root_token_index;
    int32_t tensor_token_index;
    int32_t dtype_token_index;
    int32_t shape_token_index;
    int32_t offsets_token_index;
    uint32_t dimension_index;
    SparkStatus status;
    bool succeeded;

    header_text = 0;
    if (!SparkValidationReadSafetensorsHeader(tensor_path, &header_text, &header_bytes))
    {
        return false;
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(header_text, (size_t)header_bytes, &document);
    free(header_text);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "could not parse safetensors header %s: %s\n", tensor_path, SparkStatusToString(status));
        return false;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    tensor_token_index = SparkJsonFindObjectMember(&document, root_token_index, tensor_name);
    dtype_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "dtype") : -1;
    shape_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "shape") : -1;
    offsets_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "data_offsets") : -1;
    succeeded = tensor_token_index >= 0 &&
        dtype_token_index >= 0 &&
        SparkJsonStringEquals(&document, dtype_token_index, expected_dtype) &&
        shape_token_index >= 0 &&
        SparkJsonGetArrayElementCount(&document, shape_token_index) == expected_rank &&
        offsets_token_index >= 0 &&
        SparkJsonGetArrayElementCount(&document, offsets_token_index) == 2u;
    expected_bytes = bytes_per_element;
    for (dimension_index = 0u; succeeded && dimension_index < expected_rank; ++dimension_index)
    {
        uint64_t observed_dimension;

        succeeded =
            SparkJsonGetUInt64(
                &document,
                SparkJsonGetArrayElement(&document, shape_token_index, dimension_index),
                &observed_dimension) == SPARK_STATUS_OK &&
            observed_dimension == expected_shape[dimension_index] &&
            expected_shape[dimension_index] != 0u &&
            expected_bytes <= UINT64_MAX / expected_shape[dimension_index];
        if (succeeded)
        {
            expected_bytes *= expected_shape[dimension_index];
        }
    }
    if (succeeded)
    {
        succeeded =
            SparkJsonGetUInt64(&document, SparkJsonGetArrayElement(&document, offsets_token_index, 0u), &start_offset) == SPARK_STATUS_OK &&
            SparkJsonGetUInt64(&document, SparkJsonGetArrayElement(&document, offsets_token_index, 1u), &end_offset) == SPARK_STATUS_OK &&
            end_offset >= start_offset &&
            end_offset - start_offset == expected_bytes;
    }
    if (!succeeded)
    {
        fprintf(stderr, "%s metadata is not the expected %s GLM shape\n", tensor_name, expected_dtype);
        SparkJsonDocumentDestroy(&document);
        return false;
    }
    *payload_file_offset = 8u + header_bytes + start_offset;
    *tensor_bytes = end_offset - start_offset;
    SparkJsonDocumentDestroy(&document);
    return true;
}

static bool SparkValidationReadBf16TensorOffsets(
    const char *tensor_path,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    uint64_t *payload_file_offset,
    uint64_t *tensor_bytes)
{
    return SparkValidationReadTypedTensorOffsets(
        tensor_path,
        tensor_name,
        "BF16",
        2u,
        expected_shape,
        expected_rank,
        payload_file_offset,
        tensor_bytes);
}

static bool SparkValidationReadF32TensorOffsets(
    const char *tensor_path,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    uint64_t *payload_file_offset,
    uint64_t *tensor_bytes)
{
    return SparkValidationReadTypedTensorOffsets(
        tensor_path,
        tensor_name,
        "F32",
        4u,
        expected_shape,
        expected_rank,
        payload_file_offset,
        tensor_bytes);
}

static bool SparkValidationReadU8TensorOffsets(
    const char *tensor_path,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    uint64_t *payload_file_offset,
    uint64_t *tensor_bytes)
{
    return SparkValidationReadTypedTensorOffsets(
        tensor_path,
        tensor_name,
        "U8",
        1u,
        expected_shape,
        expected_rank,
        payload_file_offset,
        tensor_bytes);
}

static bool SparkValidationReadF8E4m3TensorOffsets(
    const char *tensor_path,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    uint64_t *payload_file_offset,
    uint64_t *tensor_bytes)
{
    return SparkValidationReadTypedTensorOffsets(
        tensor_path,
        tensor_name,
        "F8_E4M3",
        1u,
        expected_shape,
        expected_rank,
        payload_file_offset,
        tensor_bytes);
}

static bool SparkValidationCopyTypedTensorToDevice(
    const char *model_directory,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    void *device_pointer,
    uint64_t *copied_bytes,
    bool (*read_offsets)(
        const char *,
        const char *,
        const uint64_t *,
        uint32_t,
        uint64_t *,
        uint64_t *))
{
    char *shard_name;
    char tensor_path[4096];
    uint64_t payload_file_offset;
    uint64_t tensor_bytes;
    uint64_t nonzero_bytes;
    uint8_t *host_tensor;
    FILE *file;
    size_t read_bytes;
    bool succeeded;

    shard_name = 0;
    host_tensor = 0;
    file = 0;
    if (SparkValidationStagePackDirectory() != 0)
    {
        succeeded = SparkValidationReadStagePackTypedTensorOffsets(
            tensor_name,
            read_offsets == SparkValidationReadBf16TensorOffsets ? "BF16" :
                read_offsets == SparkValidationReadF32TensorOffsets ? "F32" :
                read_offsets == SparkValidationReadU8TensorOffsets ? "U8" :
                "F8_E4M3",
            read_offsets == SparkValidationReadBf16TensorOffsets ? 2u :
                read_offsets == SparkValidationReadF32TensorOffsets ? 4u :
                1u,
            expected_shape,
            expected_rank,
            tensor_path,
            sizeof(tensor_path),
            &payload_file_offset,
            &tensor_bytes);
    }
    else
    {
        succeeded =
            SparkValidationReadTensorShardName(model_directory, tensor_name, &shard_name) &&
            SparkValidationBuildModelPath(model_directory, shard_name, tensor_path, sizeof(tensor_path)) &&
            read_offsets(tensor_path, tensor_name, expected_shape, expected_rank, &payload_file_offset, &tensor_bytes);
        free(shard_name);
    }
    if (!succeeded)
    {
        return false;
    }
    if (payload_file_offset > (uint64_t)LONG_MAX || tensor_bytes > (uint64_t)SIZE_MAX)
    {
        fprintf(stderr, "%s body span is unsupported by validator host\n", tensor_name);
        return false;
    }
    host_tensor = (uint8_t *)malloc((size_t)tensor_bytes);
    if (host_tensor == 0)
    {
        fprintf(stderr, "could not allocate host tensor for %s\n", tensor_name);
        return false;
    }
    file = fopen(tensor_path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "could not open tensor shard %s\n", tensor_path);
        free(host_tensor);
        return false;
    }
    if (fseek(file, (long)payload_file_offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "could not seek tensor %s\n", tensor_name);
        fclose(file);
        free(host_tensor);
        return false;
    }
    read_bytes = fread(host_tensor, 1u, (size_t)tensor_bytes, file);
    fclose(file);
    if (read_bytes != (size_t)tensor_bytes)
    {
        fprintf(stderr, "could not read tensor %s\n", tensor_name);
        free(host_tensor);
        return false;
    }
    nonzero_bytes = 0u;
    for (uint64_t byte_index = 0u; byte_index < tensor_bytes; ++byte_index)
    {
        if (host_tensor[byte_index] != 0u)
        {
            nonzero_bytes += 1u;
        }
    }
    if (nonzero_bytes == 0u)
    {
        fprintf(stderr, "%s tensor body is all zero\n", tensor_name);
        free(host_tensor);
        return false;
    }
    succeeded = SparkValidationCopyToDevice(
        device_pointer,
        host_tensor,
        tensor_bytes,
        tensor_name);
    free(host_tensor);
    if (!succeeded)
    {
        return false;
    }
    *copied_bytes += tensor_bytes;
    return true;
}

static bool SparkValidationCopyBf16TensorToDevice(
    const char *model_directory,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    void *device_pointer,
    uint64_t *copied_bytes)
{
    return SparkValidationCopyTypedTensorToDevice(
        model_directory,
        tensor_name,
        expected_shape,
        expected_rank,
        device_pointer,
        copied_bytes,
        SparkValidationReadBf16TensorOffsets);
}

static bool SparkValidationCopyF32TensorToDevice(
    const char *model_directory,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    void *device_pointer,
    uint64_t *copied_bytes)
{
    return SparkValidationCopyTypedTensorToDevice(
        model_directory,
        tensor_name,
        expected_shape,
        expected_rank,
        device_pointer,
        copied_bytes,
        SparkValidationReadF32TensorOffsets);
}

static bool SparkValidationCopyU8TensorToDevice(
    const char *model_directory,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    void *device_pointer,
    uint64_t *copied_bytes)
{
    return SparkValidationCopyTypedTensorToDevice(
        model_directory,
        tensor_name,
        expected_shape,
        expected_rank,
        device_pointer,
        copied_bytes,
        SparkValidationReadU8TensorOffsets);
}

static bool SparkValidationCopyF8E4m3TensorToDevice(
    const char *model_directory,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    void *device_pointer,
    uint64_t *copied_bytes)
{
    return SparkValidationCopyTypedTensorToDevice(
        model_directory,
        tensor_name,
        expected_shape,
        expected_rank,
        device_pointer,
        copied_bytes,
        SparkValidationReadF8E4m3TensorOffsets);
}

static bool SparkValidationReadScalarF32Tensor(
    const char *model_directory,
    const char *tensor_name,
    float *value)
{
    const uint64_t scalar_shape[1] = {1u};
    char *shard_name;
    char tensor_path[4096];
    uint64_t payload_file_offset;
    uint64_t tensor_bytes;
    FILE *file;
    float scalar_value;
    bool succeeded;

    shard_name = 0;
    file = 0;
    if (SparkValidationStagePackDirectory() != 0)
    {
        succeeded = SparkValidationReadStagePackTypedTensorOffsets(
            tensor_name,
            "F32",
            4u,
            scalar_shape,
            0u,
            tensor_path,
            sizeof(tensor_path),
            &payload_file_offset,
            &tensor_bytes);
    }
    else
    {
        succeeded =
            SparkValidationReadTensorShardName(model_directory, tensor_name, &shard_name) &&
            SparkValidationBuildModelPath(model_directory, shard_name, tensor_path, sizeof(tensor_path)) &&
            SparkValidationReadF32TensorOffsets(tensor_path, tensor_name, scalar_shape, 0u, &payload_file_offset, &tensor_bytes);
        free(shard_name);
    }
    if (!succeeded || tensor_bytes != sizeof(float))
    {
        fprintf(stderr, "%s scalar metadata is not F32\n", tensor_name);
        return false;
    }
    if (payload_file_offset > (uint64_t)LONG_MAX)
    {
        fprintf(stderr, "%s scalar offset is unsupported\n", tensor_name);
        return false;
    }
    file = fopen(tensor_path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "could not open scalar shard %s\n", tensor_path);
        return false;
    }
    if (fseek(file, (long)payload_file_offset, SEEK_SET) != 0 ||
        fread(&scalar_value, 1u, sizeof(scalar_value), file) !=
            sizeof(scalar_value))
    {
        fprintf(stderr, "could not read scalar tensor %s\n", tensor_name);
        fclose(file);
        return false;
    }
    fclose(file);
    *value = scalar_value;
    return true;
}

static bool SparkValidationCopyBf16TensorRowToDevice(
    const char *model_directory,
    const char *tensor_name,
    const uint64_t *expected_shape,
    uint32_t expected_rank,
    uint64_t row_index,
    void *device_pointer,
    uint64_t *copied_bytes)
{
    char *shard_name;
    char tensor_path[4096];
    uint64_t payload_file_offset;
    uint64_t tensor_bytes;
    uint64_t row_bytes;
    uint64_t row_offset;
    uint64_t absolute_offset;
    uint64_t nonzero_bytes;
    uint8_t *host_row;
    FILE *file;
    size_t read_bytes;
    bool succeeded;

    shard_name = 0;
    host_row = 0;
    file = 0;
    if (expected_rank != 2u || row_index >= expected_shape[0])
    {
        fprintf(stderr, "%s row request is outside expected shape\n", tensor_name);
        return false;
    }
    row_bytes = expected_shape[1] * 2u;
    row_offset = row_index * row_bytes;
    if (SparkValidationStagePackDirectory() != 0)
    {
        succeeded = SparkValidationReadStagePackTypedTensorOffsets(
            tensor_name,
            "BF16",
            2u,
            expected_shape,
            expected_rank,
            tensor_path,
            sizeof(tensor_path),
            &payload_file_offset,
            &tensor_bytes);
    }
    else
    {
        succeeded =
            SparkValidationReadTensorShardName(model_directory, tensor_name, &shard_name) &&
            SparkValidationBuildModelPath(model_directory, shard_name, tensor_path, sizeof(tensor_path)) &&
            SparkValidationReadBf16TensorOffsets(tensor_path, tensor_name, expected_shape, expected_rank, &payload_file_offset, &tensor_bytes);
        free(shard_name);
    }
    if (!succeeded)
    {
        return false;
    }
    if (row_offset > tensor_bytes || row_bytes > tensor_bytes - row_offset)
    {
        fprintf(stderr, "%s row span is outside tensor body\n", tensor_name);
        return false;
    }
    absolute_offset = payload_file_offset + row_offset;
    if (absolute_offset > (uint64_t)LONG_MAX || row_bytes > (uint64_t)SIZE_MAX)
    {
        fprintf(stderr, "%s row span is unsupported by validator host\n", tensor_name);
        return false;
    }
    host_row = (uint8_t *)malloc((size_t)row_bytes);
    if (host_row == 0)
    {
        fprintf(stderr, "could not allocate host row for %s\n", tensor_name);
        return false;
    }
    file = fopen(tensor_path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "could not open tensor shard %s\n", tensor_path);
        free(host_row);
        return false;
    }
    if (fseek(file, (long)absolute_offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "could not seek tensor row %s\n", tensor_name);
        fclose(file);
        free(host_row);
        return false;
    }
    read_bytes = fread(host_row, 1u, (size_t)row_bytes, file);
    fclose(file);
    if (read_bytes != (size_t)row_bytes)
    {
        fprintf(stderr, "could not read tensor row %s\n", tensor_name);
        free(host_row);
        return false;
    }
    nonzero_bytes = 0u;
    for (uint64_t byte_index = 0u; byte_index < row_bytes; ++byte_index)
    {
        if (host_row[byte_index] != 0u)
        {
            nonzero_bytes += 1u;
        }
    }
    if (nonzero_bytes == 0u)
    {
        fprintf(stderr, "%s row body is all zero\n", tensor_name);
        free(host_row);
        return false;
    }
    succeeded = SparkValidationCopyToDevice(
        device_pointer,
        host_row,
        row_bytes,
        tensor_name);
    free(host_row);
    if (!succeeded)
    {
        return false;
    }
    *copied_bytes += row_bytes;
    return true;
}

static bool SparkValidationCopyInputEmbeddingBf16Row(
    const char *model_directory,
    uint32_t token_id,
    void *device_pointer,
    uint64_t *copied_bytes)
{
    const uint64_t embedding_shape[2] = {
        154880u,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};

    return SparkValidationCopyBf16TensorRowToDevice(
        model_directory,
        "model.embed_tokens.weight",
        embedding_shape,
        2u,
        token_id,
        device_pointer,
        copied_bytes);
}

static bool SparkValidationLoadInputEmbeddingBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    uint32_t token_id,
    SparkValidationInputEmbeddingBf16Fixture *fixture)
{
    memset(fixture, 0, sizeof(*fixture));
    if (!SparkValidationCopyInputEmbeddingBf16Row(
            model_directory,
            token_id,
            buffers->input_hidden_bf16,
            &fixture->copied_bytes))
        return false;
    fixture->token_id = token_id;
    fixture->ready = 1u;
    fprintf(stderr, "input_embedding_bf16_fixture_ready=1 model_dir=%s token=%u bytes=%llu\n", model_directory, token_id, (unsigned long long)fixture->copied_bytes);
    return true;
}

static bool SparkValidationLoadFinalNormBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    SparkValidationFinalNormBf16Fixture *fixture)
{
    const uint64_t norm_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};

    memset(fixture, 0, sizeof(*fixture));
    if (!SparkValidationCopyBf16TensorToDevice(
            model_directory,
            "model.norm.weight",
            norm_shape,
            1u,
            buffers->final_norm_weight_bf16,
            &fixture->copied_bytes))
        return false;
    fixture->ready = 1u;
    fprintf(stderr, "real_final_norm_bf16_fixture_ready=1 model_dir=%s bytes=%llu\n", model_directory, (unsigned long long)fixture->copied_bytes);
    return true;
}

static bool SparkValidationLoadLayer0DenseBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    uint32_t layer_index,
    SparkValidationLayer0DenseBf16Fixture *fixture)
{
    const uint64_t norm_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t gate_up_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t down_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION};
    char norm_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char gate_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char up_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char down_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];

    memset(fixture, 0, sizeof(*fixture));
    if (layer_index >= SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT)
    {
        fprintf(stderr, "GLM52_DENSE_LAYER_INDEX must be 0..%u\n", SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT - 1u);
        return false;
    }
    if (!SparkValidationBuildLayerTensorName(norm_name, sizeof(norm_name), layer_index, "post_attention_layernorm.weight") ||
        !SparkValidationBuildLayerTensorName(gate_name, sizeof(gate_name), layer_index, "mlp.gate_proj.weight") ||
        !SparkValidationBuildLayerTensorName(up_name, sizeof(up_name), layer_index, "mlp.up_proj.weight") ||
        !SparkValidationBuildLayerTensorName(down_name, sizeof(down_name), layer_index, "mlp.down_proj.weight"))
    {
        fprintf(stderr, "dense layer tensor name is too long\n");
        return false;
    }
    if (!SparkValidationCopyBf16TensorToDevice(
            model_directory,
            norm_name,
            norm_shape,
            1u,
            buffers->post_attention_norm_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            gate_name,
            gate_up_shape,
            2u,
            buffers->dense_gate_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            up_name,
            gate_up_shape,
            2u,
            buffers->dense_up_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            down_name,
            down_shape,
            2u,
            buffers->dense_down_weight_bf16,
            &fixture->copied_bytes))
    {
        return false;
    }
    fixture->ready = 1u;
    fprintf(stderr, "layer0_dense_bf16_fixture_ready=1 model_dir=%s dense_layer_index=%u bytes=%llu\n", model_directory, layer_index, (unsigned long long)fixture->copied_bytes);
    return true;
}

static bool SparkValidationLoadLayer3SharedExpertBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    SparkValidationLayer3SharedExpertBf16Fixture *fixture)
{
    const uint64_t norm_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t gate_up_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t down_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION};
    char norm_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char gate_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char up_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char down_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];

    memset(fixture, 0, sizeof(*fixture));
    if (!SparkValidationBuildLayerTensorName(norm_name, sizeof(norm_name), 3u, "post_attention_layernorm.weight") ||
        !SparkValidationBuildLayerTensorName(gate_name, sizeof(gate_name), 3u, "mlp.shared_experts.gate_proj.weight") ||
        !SparkValidationBuildLayerTensorName(up_name, sizeof(up_name), 3u, "mlp.shared_experts.up_proj.weight") ||
        !SparkValidationBuildLayerTensorName(down_name, sizeof(down_name), 3u, "mlp.shared_experts.down_proj.weight"))
    {
        fprintf(stderr, "layer3 shared expert tensor name is too long\n");
        return false;
    }
    if (!SparkValidationCopyBf16TensorToDevice(
            model_directory,
            norm_name,
            norm_shape,
            1u,
            buffers->post_attention_norm_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            gate_name,
            gate_up_shape,
            2u,
            buffers->dense_gate_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            up_name,
            gate_up_shape,
            2u,
            buffers->dense_up_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            down_name,
            down_shape,
            2u,
            buffers->dense_down_weight_bf16,
            &fixture->copied_bytes))
    {
        return false;
    }
    fixture->ready = 1u;
    fprintf(stderr, "layer3_shared_expert_bf16_fixture_ready=1 model_dir=%s bytes=%llu\n", model_directory, (unsigned long long)fixture->copied_bytes);
    return true;
}

static bool SparkValidationLoadLayer0AttentionBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    uint32_t layer_index,
    SparkValidationLayer0AttentionBf16Fixture *fixture)
{
    const uint64_t hidden_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t query_a_norm_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION};
    const uint64_t kv_a_norm_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION};
    const uint64_t query_a_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t query_b_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION};
    const uint64_t kv_a_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t kv_b_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION};
    const uint64_t output_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION};
    char input_norm_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char q_a_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char q_a_norm_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char q_b_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char kv_a_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char kv_a_norm_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char kv_b_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char output_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];

    memset(fixture, 0, sizeof(*fixture));
    if (layer_index >= SPARK_VALIDATION_LAYER_COUNT)
    {
        fprintf(stderr, "GLM52 attention layer index must be 0..%u\n", SPARK_VALIDATION_LAYER_COUNT - 1u);
        return false;
    }
    if (!SparkValidationBuildLayerTensorName(input_norm_name, sizeof(input_norm_name), layer_index, "input_layernorm.weight") ||
        !SparkValidationBuildLayerTensorName(q_a_name, sizeof(q_a_name), layer_index, "self_attn.q_a_proj.weight") ||
        !SparkValidationBuildLayerTensorName(q_a_norm_name, sizeof(q_a_norm_name), layer_index, "self_attn.q_a_layernorm.weight") ||
        !SparkValidationBuildLayerTensorName(q_b_name, sizeof(q_b_name), layer_index, "self_attn.q_b_proj.weight") ||
        !SparkValidationBuildLayerTensorName(kv_a_name, sizeof(kv_a_name), layer_index, "self_attn.kv_a_proj_with_mqa.weight") ||
        !SparkValidationBuildLayerTensorName(kv_a_norm_name, sizeof(kv_a_norm_name), layer_index, "self_attn.kv_a_layernorm.weight") ||
        !SparkValidationBuildLayerTensorName(kv_b_name, sizeof(kv_b_name), layer_index, "self_attn.kv_b_proj.weight") ||
        !SparkValidationBuildLayerTensorName(output_name, sizeof(output_name), layer_index, "self_attn.o_proj.weight"))
    {
        fprintf(stderr, "attention layer tensor name is too long\n");
        return false;
    }
    if (!SparkValidationCopyBf16TensorToDevice(
            model_directory,
            input_norm_name,
            hidden_shape,
            1u,
            buffers->attention_norm_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            q_a_name,
            query_a_shape,
            2u,
            buffers->raw_query_a_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            q_a_norm_name,
            query_a_norm_shape,
            1u,
            buffers->raw_query_a_norm_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            q_b_name,
            query_b_shape,
            2u,
            buffers->raw_query_b_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            kv_a_name,
            kv_a_shape,
            2u,
            buffers->raw_kv_a_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            kv_a_norm_name,
            kv_a_norm_shape,
            1u,
            buffers->raw_kv_a_norm_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            kv_b_name,
            kv_b_shape,
            2u,
            buffers->raw_kv_b_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            output_name,
            output_shape,
            2u,
            buffers->attention_output_weight_bf16,
            &fixture->copied_bytes))
    {
        return false;
    }
    fixture->ready = 1u;
    fprintf(stderr, "layer0_attention_bf16_fixture_ready=1 model_dir=%s dense_layer_index=%u bytes=%llu\n", model_directory, layer_index, (unsigned long long)fixture->copied_bytes);
    return true;
}

static bool SparkValidationLoadRoutedLayerRouterBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    uint32_t layer_index,
    SparkValidationLayer3RouterBf16Fixture *fixture)
{
    const uint64_t hidden_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t router_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    const uint64_t bias_shape[1] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT};
    char norm_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char router_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char bias_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];

    memset(fixture, 0, sizeof(*fixture));
    if (layer_index < SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX ||
        layer_index >= SPARK_VALIDATION_LAYER_COUNT)
    {
        fprintf(stderr, "GLM52 routed layer index must be %u..%u\n", SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX, SPARK_VALIDATION_LAYER_COUNT - 1u);
        return false;
    }
    if (!SparkValidationBuildLayerTensorName(norm_name, sizeof(norm_name), layer_index, "post_attention_layernorm.weight") ||
        !SparkValidationBuildLayerTensorName(router_name, sizeof(router_name), layer_index, "mlp.gate.weight") ||
        !SparkValidationBuildLayerTensorName(bias_name, sizeof(bias_name), layer_index, "mlp.gate.e_score_correction_bias"))
    {
        fprintf(stderr, "routed layer router tensor name is too long layer=%u\n", layer_index);
        return false;
    }
    if (!SparkValidationCopyBf16TensorToDevice(
            model_directory,
            norm_name,
            hidden_shape,
            1u,
            buffers->post_attention_norm_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyBf16TensorToDevice(
            model_directory,
            router_name,
            router_shape,
            2u,
            buffers->moe_router_weight_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationCopyF32TensorToDevice(
            model_directory,
            bias_name,
            bias_shape,
            1u,
            buffers->moe_router_score_bias_f32,
            &fixture->copied_bytes))
    {
        return false;
    }
    fixture->ready = 1u;
    fprintf(stderr, "routed_layer_router_bf16_fixture_ready=1 model_dir=%s layer=%u bytes=%llu\n", model_directory, layer_index, (unsigned long long)fixture->copied_bytes);
    return true;
}

static bool SparkValidationLoadLayer3RouterBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    SparkValidationLayer3RouterBf16Fixture *fixture)
{
    return SparkValidationLoadRoutedLayerRouterBf16Fixture(
        buffers,
        model_directory,
        SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX,
        fixture);
}

static bool SparkValidationBuildRoutedLayerExpertTensorName(
    char *tensor_name,
    uint32_t tensor_name_bytes,
    uint32_t layer_index,
    uint32_t expert_id,
    const char *projection,
    const char *suffix)
{
    int written_bytes;

    written_bytes = snprintf(
        tensor_name,
        (size_t)tensor_name_bytes,
        "model.layers.%u.mlp.experts.%u.%s.%s",
        layer_index,
        expert_id,
        projection,
        suffix);
    return written_bytes >= 0 && (uint32_t)written_bytes < tensor_name_bytes;
}

static bool SparkValidationLoadLayer3RoutedExpertNvfp4FixtureForExperts(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    uint32_t layer_index,
    SparkValidationLayer3RoutedExpertNvfp4Fixture *fixture,
    const uint32_t *expert_ids,
    uint32_t expert_count)
{
    const uint64_t gate_up_weight_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u};
    const uint64_t gate_up_scale_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE};
    const uint64_t down_weight_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u};
    const uint64_t down_scale_shape[2] = {
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE};
    char gate_weight_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char gate_scale_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char gate_scale2_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char gate_input_scale_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char up_weight_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char up_scale_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char up_scale2_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char up_input_scale_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char down_weight_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char down_scale_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char down_scale2_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    char down_input_scale_name[SPARK_VALIDATION_TENSOR_NAME_BYTES];
    uint64_t gate_up_payload_stride;
    uint64_t gate_up_scale_stride;
    uint64_t down_payload_stride;
    uint64_t down_scale_stride;
    uint32_t expert_slot;

    if (layer_index < SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX ||
        layer_index >= SPARK_VALIDATION_LAYER_COUNT)
    {
        fprintf(stderr, "GLM52 routed expert layer index must be %u..%u\n", SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX, SPARK_VALIDATION_LAYER_COUNT - 1u);
        return false;
    }
    if (expert_ids == 0 ||
        expert_count == 0u ||
        expert_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K)
    {
        fprintf(stderr, "layer3 routed expert id list is invalid count=%u\n", expert_count);
        return false;
    }
    memset(fixture, 0, sizeof(*fixture));
    gate_up_payload_stride =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
    gate_up_scale_stride =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE);
    down_payload_stride =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
         2u);
    down_scale_stride =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE);
    fixture->bound_expert_count = expert_count;
    fixture->selected_expert_id = expert_ids[0];
    for (expert_slot = 0u;
         expert_slot < fixture->bound_expert_count;
         ++expert_slot)
    {
        uint32_t expert_id;

        expert_id = expert_ids[expert_slot];
        fixture->bound_expert_ids[expert_slot] = expert_id;
        if (!SparkValidationBuildRoutedLayerExpertTensorName(gate_weight_name, sizeof(gate_weight_name), layer_index, expert_id, "gate_proj", "weight") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(gate_scale_name, sizeof(gate_scale_name), layer_index, expert_id, "gate_proj", "weight_scale") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(gate_scale2_name, sizeof(gate_scale2_name), layer_index, expert_id, "gate_proj", "weight_scale_2") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(gate_input_scale_name, sizeof(gate_input_scale_name), layer_index, expert_id, "gate_proj", "input_scale") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(up_weight_name, sizeof(up_weight_name), layer_index, expert_id, "up_proj", "weight") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(up_scale_name, sizeof(up_scale_name), layer_index, expert_id, "up_proj", "weight_scale") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(up_scale2_name, sizeof(up_scale2_name), layer_index, expert_id, "up_proj", "weight_scale_2") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(up_input_scale_name, sizeof(up_input_scale_name), layer_index, expert_id, "up_proj", "input_scale") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(down_weight_name, sizeof(down_weight_name), layer_index, expert_id, "down_proj", "weight") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(down_scale_name, sizeof(down_scale_name), layer_index, expert_id, "down_proj", "weight_scale") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(down_scale2_name, sizeof(down_scale2_name), layer_index, expert_id, "down_proj", "weight_scale_2") ||
            !SparkValidationBuildRoutedLayerExpertTensorName(down_input_scale_name, sizeof(down_input_scale_name), layer_index, expert_id, "down_proj", "input_scale"))
        {
            fprintf(stderr, "layer3 routed expert tensor name is too long\n");
            return false;
        }
        if (!SparkValidationCopyU8TensorToDevice(model_directory, gate_weight_name, gate_up_weight_shape, 2u, buffers->routed_gate_weight_payload_u8 + (expert_slot * gate_up_payload_stride), &fixture->copied_bytes) ||
            !SparkValidationCopyF8E4m3TensorToDevice(model_directory, gate_scale_name, gate_up_scale_shape, 2u, buffers->routed_gate_weight_scale_e4m3 + (expert_slot * gate_up_scale_stride), &fixture->copied_bytes) ||
            !SparkValidationReadScalarF32Tensor(model_directory, gate_scale2_name, &fixture->gate_weight_scale_2_values[expert_slot]) ||
            !SparkValidationReadScalarF32Tensor(model_directory, gate_input_scale_name, &fixture->gate_input_scales[expert_slot]) ||
            !SparkValidationCopyU8TensorToDevice(model_directory, up_weight_name, gate_up_weight_shape, 2u, buffers->routed_up_weight_payload_u8 + (expert_slot * gate_up_payload_stride), &fixture->copied_bytes) ||
            !SparkValidationCopyF8E4m3TensorToDevice(model_directory, up_scale_name, gate_up_scale_shape, 2u, buffers->routed_up_weight_scale_e4m3 + (expert_slot * gate_up_scale_stride), &fixture->copied_bytes) ||
            !SparkValidationReadScalarF32Tensor(model_directory, up_scale2_name, &fixture->up_weight_scale_2_values[expert_slot]) ||
            !SparkValidationReadScalarF32Tensor(model_directory, up_input_scale_name, &fixture->up_input_scales[expert_slot]) ||
            !SparkValidationCopyU8TensorToDevice(model_directory, down_weight_name, down_weight_shape, 2u, buffers->routed_down_weight_payload_u8 + (expert_slot * down_payload_stride), &fixture->copied_bytes) ||
            !SparkValidationCopyF8E4m3TensorToDevice(model_directory, down_scale_name, down_scale_shape, 2u, buffers->routed_down_weight_scale_e4m3 + (expert_slot * down_scale_stride), &fixture->copied_bytes) ||
            !SparkValidationReadScalarF32Tensor(model_directory, down_scale2_name, &fixture->down_weight_scale_2_values[expert_slot]) ||
            !SparkValidationReadScalarF32Tensor(model_directory, down_input_scale_name, &fixture->down_input_scales[expert_slot]))
        {
            return false;
        }
    }
    fixture->gate_input_scale = fixture->gate_input_scales[0];
    fixture->gate_weight_scale_2 = fixture->gate_weight_scale_2_values[0];
    fixture->up_input_scale = fixture->up_input_scales[0];
    fixture->up_weight_scale_2 = fixture->up_weight_scale_2_values[0];
    fixture->down_input_scale = fixture->down_input_scales[0];
    fixture->down_weight_scale_2 = fixture->down_weight_scale_2_values[0];
    if (!SparkValidationCopyToDevice(buffers->routed_bound_expert_ids, fixture->bound_expert_ids, fixture->bound_expert_count * 4u, "copy nvfp4 bound expert ids") ||
        !SparkValidationCopyToDevice(buffers->routed_gate_input_scale_f32, fixture->gate_input_scales, fixture->bound_expert_count * 4u, "copy nvfp4 gate input scales") ||
        !SparkValidationCopyToDevice(buffers->routed_gate_weight_scale_2_f32, fixture->gate_weight_scale_2_values, fixture->bound_expert_count * 4u, "copy nvfp4 gate scale2") ||
        !SparkValidationCopyToDevice(buffers->routed_up_input_scale_f32, fixture->up_input_scales, fixture->bound_expert_count * 4u, "copy nvfp4 up input scales") ||
        !SparkValidationCopyToDevice(buffers->routed_up_weight_scale_2_f32, fixture->up_weight_scale_2_values, fixture->bound_expert_count * 4u, "copy nvfp4 up scale2") ||
        !SparkValidationCopyToDevice(buffers->routed_down_input_scale_f32, fixture->down_input_scales, fixture->bound_expert_count * 4u, "copy nvfp4 down input scales") ||
        !SparkValidationCopyToDevice(buffers->routed_down_weight_scale_2_f32, fixture->down_weight_scale_2_values, fixture->bound_expert_count * 4u, "copy nvfp4 down scale2"))
    {
        return false;
    }
    fixture->ready = 1u;
    fprintf(stderr, "routed_layer_expert_nvfp4_fixture_ready=1 model_dir=%s layer=%u expert=%u bound_count=%u ids=%u,%u,%u,%u,%u,%u,%u,%u bytes=%llu gate_scale2=%.9g up_scale2=%.9g down_scale2=%.9g\n", model_directory, layer_index, fixture->selected_expert_id, fixture->bound_expert_count, fixture->bound_expert_ids[0], fixture->bound_expert_ids[1], fixture->bound_expert_ids[2], fixture->bound_expert_ids[3], fixture->bound_expert_ids[4], fixture->bound_expert_ids[5], fixture->bound_expert_ids[6], fixture->bound_expert_ids[7], (unsigned long long)fixture->copied_bytes, (double)fixture->gate_weight_scale_2, (double)fixture->up_weight_scale_2, (double)fixture->down_weight_scale_2);
    return true;
}

static bool SparkValidationLoadLayer3RoutedExpertNvfp4Fixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    SparkValidationLayer3RoutedExpertNvfp4Fixture *fixture,
    uint32_t load_topk)
{
    const uint32_t topk_expert_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K] = {
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_0,
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_1,
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_2,
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_3,
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_4,
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_5,
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_6,
        SPARK_VALIDATION_LAYER3_ROUTED_EXPERT_ID_7};

    return SparkValidationLoadLayer3RoutedExpertNvfp4FixtureForExperts(
        buffers,
        model_directory,
        SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX,
        fixture,
        topk_expert_ids,
        load_topk != 0u
            ? SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K
            : 1u);
}

static bool SparkValidationReadLmHeadOffsets(
    const char *tensor_path,
    uint64_t *payload_file_offset,
    uint64_t *tensor_bytes)
{
    SparkJsonDocument document;
    char *header_text;
    uint64_t header_bytes;
    uint64_t start_offset;
    uint64_t end_offset;
    int32_t root_token_index;
    int32_t tensor_token_index;
    int32_t dtype_token_index;
    int32_t shape_token_index;
    int32_t offsets_token_index;
    SparkStatus status;
    bool succeeded;

    header_text = 0;
    if (!SparkValidationReadSafetensorsHeader(tensor_path, &header_text, &header_bytes))
    {
        return false;
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(header_text, (size_t)header_bytes, &document);
    free(header_text);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "could not parse safetensors header %s: %s\n", tensor_path, SparkStatusToString(status));
        return false;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    tensor_token_index = SparkJsonFindObjectMember(&document, root_token_index, "lm_head.weight");
    dtype_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "dtype") : -1;
    shape_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "shape") : -1;
    offsets_token_index = tensor_token_index >= 0 ? SparkJsonFindObjectMember(&document, tensor_token_index, "data_offsets") : -1;
    succeeded = tensor_token_index >= 0 &&
        dtype_token_index >= 0 &&
        SparkJsonStringEquals(&document, dtype_token_index, "BF16") &&
        shape_token_index >= 0 &&
        SparkJsonGetArrayElementCount(&document, shape_token_index) == 2u &&
        offsets_token_index >= 0 &&
        SparkJsonGetArrayElementCount(&document, offsets_token_index) == 2u;
    if (succeeded)
    {
        uint64_t row_count;
        uint64_t hidden_count;

        succeeded =
            SparkJsonGetUInt64(&document, SparkJsonGetArrayElement(&document, shape_token_index, 0u), &row_count) == SPARK_STATUS_OK &&
            SparkJsonGetUInt64(&document, SparkJsonGetArrayElement(&document, shape_token_index, 1u), &hidden_count) == SPARK_STATUS_OK &&
            row_count == 154880u &&
            hidden_count == SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION &&
            SparkJsonGetUInt64(&document, SparkJsonGetArrayElement(&document, offsets_token_index, 0u), &start_offset) == SPARK_STATUS_OK &&
            SparkJsonGetUInt64(&document, SparkJsonGetArrayElement(&document, offsets_token_index, 1u), &end_offset) == SPARK_STATUS_OK &&
            end_offset >= start_offset;
    }
    if (!succeeded)
    {
        fprintf(stderr, "lm_head.weight metadata is not the expected BF16 GLM shape\n");
        SparkJsonDocumentDestroy(&document);
        return false;
    }
    *payload_file_offset = 8u + header_bytes + start_offset;
    *tensor_bytes = end_offset - start_offset;
    SparkJsonDocumentDestroy(&document);
    return true;
}

static bool SparkValidationReadLmHeadRows(
    const char *tensor_path,
    uint64_t payload_file_offset,
    uint64_t tensor_bytes,
    SparkValidationRealLmHeadFixture *fixture)
{
    FILE *file;
    uint64_t sample_offset;
    uint64_t sample_bytes;
    uint64_t absolute_offset;
    uint64_t nonzero_bytes;
    size_t read_bytes;

    fixture->restricted_row_bytes =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u;
    sample_offset =
        (uint64_t)SPARK_VALIDATION_RESTRICTED_LM_HEAD_FIRST_TOKEN *
        fixture->restricted_row_bytes;
    sample_bytes =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT *
        fixture->restricted_row_bytes;
    if (sample_offset > tensor_bytes || sample_bytes > tensor_bytes - sample_offset)
    {
        fprintf(stderr, "restricted lm_head rows are outside tensor body\n");
        return false;
    }
    absolute_offset = payload_file_offset + sample_offset;
    if (absolute_offset > (uint64_t)LONG_MAX || sample_bytes > (uint64_t)SIZE_MAX)
    {
        fprintf(stderr, "restricted lm_head row span is unsupported by validator host\n");
        return false;
    }
    fixture->restricted_rows_bf16 = (uint16_t *)malloc((size_t)sample_bytes);
    if (fixture->restricted_rows_bf16 == 0)
    {
        fprintf(stderr, "could not allocate restricted lm_head rows\n");
        return false;
    }
    file = fopen(tensor_path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "could not open lm_head shard %s\n", tensor_path);
        return false;
    }
    if (fseek(file, (long)absolute_offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "could not seek restricted lm_head rows\n");
        fclose(file);
        return false;
    }
    read_bytes = fread(fixture->restricted_rows_bf16, 1u, (size_t)sample_bytes, file);
    fclose(file);
    if (read_bytes != (size_t)sample_bytes)
    {
        fprintf(stderr, "could not read restricted lm_head rows\n");
        return false;
    }
    nonzero_bytes = 0u;
    for (uint64_t byte_index = 0u; byte_index < sample_bytes; ++byte_index)
    {
        if (((const uint8_t *)fixture->restricted_rows_bf16)[byte_index] != 0u)
        {
            nonzero_bytes += 1u;
        }
    }
    if (nonzero_bytes == 0u)
    {
        fprintf(stderr, "restricted lm_head rows are all zero\n");
        return false;
    }
    return true;
}

static void SparkValidationDestroyRealLmHeadFixture(
    SparkValidationRealLmHeadFixture *fixture)
{
    free(fixture->restricted_rows_bf16);
    memset(fixture, 0, sizeof(*fixture));
}

static bool SparkValidationLoadRealLmHeadFixture(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    SparkValidationRealLmHeadFixture *fixture)
{
    char *shard_name;
    char tensor_path[4096];
    uint64_t payload_file_offset;
    uint64_t tensor_bytes;
    uint64_t copy_bytes;
    const uint64_t lm_head_shape[2] = {
        154880u,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION};
    bool succeeded;

    memset(fixture, 0, sizeof(*fixture));
    shard_name = 0;
    if (SparkValidationStagePackDirectory() != 0)
    {
        succeeded =
            SparkValidationReadStagePackTypedTensorOffsets(
                "lm_head.weight",
                "BF16",
                2u,
                lm_head_shape,
                2u,
                tensor_path,
                sizeof(tensor_path),
                &payload_file_offset,
                &tensor_bytes) &&
            SparkValidationReadLmHeadRows(tensor_path, payload_file_offset, tensor_bytes, fixture);
    }
    else
    {
        succeeded =
            SparkValidationReadLmHeadShardName(model_directory, &shard_name) &&
            SparkValidationBuildModelPath(model_directory, shard_name, tensor_path, sizeof(tensor_path)) &&
            SparkValidationReadLmHeadOffsets(tensor_path, &payload_file_offset, &tensor_bytes) &&
            SparkValidationReadLmHeadRows(tensor_path, payload_file_offset, tensor_bytes, fixture);
        free(shard_name);
    }
    if (!succeeded)
    {
        SparkValidationDestroyRealLmHeadFixture(fixture);
        return false;
    }
    copy_bytes =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT *
        fixture->restricted_row_bytes;
    if (!SparkValidationCopyToDevice(
            buffers->restricted_lm_head_weight_bf16,
            fixture->restricted_rows_bf16,
            copy_bytes,
            "copy real restricted lm_head rows"))
    {
        SparkValidationDestroyRealLmHeadFixture(fixture);
        return false;
    }
    fixture->ready = 1u;
    fprintf(stderr, "real_lm_head_fixture_ready=1 model_dir=%s bytes=%llu\n", model_directory, (unsigned long long)copy_bytes);
    return true;
}

static float SparkValidationFp8E4m3ToFloat(uint8_t value)
{
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    float decoded_value;

    sign = (uint32_t)(value >> 7u);
    exponent = (uint32_t)((value >> 3u) & 15u);
    mantissa = (uint32_t)(value & 7u);
    if ((value & 0x7fu) == 0u)
    {
        return 0.0f;
    }
    if (exponent == 0u)
    {
        decoded_value = ldexpf((float)mantissa / 8.0f, -6);
    }
    else
    {
        decoded_value =
            ldexpf(1.0f + ((float)mantissa / 8.0f), (int32_t)exponent - 7);
    }
    return sign != 0u ? -decoded_value : decoded_value;
}

static uint8_t SparkValidationFloatToFp8E4m3(float value)
{
    float best_error;
    uint8_t best_byte;
    uint32_t byte_value;

    best_error = fabsf(value);
    best_byte = 0u;
    for (byte_value = 1u; byte_value < 255u; ++byte_value)
    {
        float decoded_value;
        float error;

        decoded_value = SparkValidationFp8E4m3ToFloat((uint8_t)byte_value);
        error = fabsf(decoded_value - value);
        if (error < best_error)
        {
            best_error = error;
            best_byte = (uint8_t)byte_value;
        }
    }
    return best_byte;
}

static float SparkValidationSeedLatentValue(
    uint32_t token_index,
    uint32_t dimension_index)
{
    return (0.03125f * (float)(token_index + 1u)) +
        (0.0078125f * (float)(dimension_index + 1u));
}

static float SparkValidationSeedKeyNopeValue(
    uint32_t token_index,
    uint32_t head_index,
    uint32_t dimension_index)
{
    return (0.0234375f * (float)(token_index + 1u)) +
        (0.001953125f * (float)(head_index + 1u)) +
        (0.00048828125f * (float)(dimension_index + 1u));
}

static float SparkValidationSeedValueCacheValue(
    uint32_t token_index,
    uint32_t head_index,
    uint32_t dimension_index)
{
    return (0.046875f * (float)(token_index + 1u)) +
        (0.00390625f * (float)(head_index + 1u)) +
        (0.0009765625f * (float)(dimension_index + 1u));
}

static float SparkValidationSeedRopeValue(
    uint32_t token_index,
    uint32_t dimension_index)
{
    return (0.015625f * (float)(token_index + 1u)) +
        (0.00390625f * (float)(dimension_index + 1u));
}

static void SparkValidationSeedCacheSlot(
    uint16_t *cache_seed,
    uint32_t cache_slot_index,
    uint32_t token_index)
{
    uint64_t cache_offset;
    uint32_t dimension_index;

    cache_offset =
        (uint64_t)cache_slot_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        cache_seed[cache_offset + (uint64_t)dimension_index] =
            SparkValidationFloatToBf16(
                SparkValidationSeedLatentValue(token_index, dimension_index));
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         ++dimension_index)
    {
        cache_seed[
            cache_offset +
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
            (uint64_t)dimension_index] =
            SparkValidationFloatToBf16(
                SparkValidationSeedRopeValue(token_index, dimension_index));
    }
}

static bool SparkValidationCopyU8Value(
    uint8_t *device_pointer,
    uint64_t index,
    uint8_t value,
    const char *name)
{
    return SparkValidationCopyToDevice(
        &device_pointer[index],
        &value,
        sizeof(value),
        name);
}

static bool SparkValidationCopyU16Value(
    uint16_t *device_pointer,
    uint64_t index,
    uint16_t value,
    const char *name)
{
    return SparkValidationCopyToDevice(
        &device_pointer[index],
        &value,
        sizeof(value),
        name);
}

static bool SparkValidationCopyFp8FixtureValue(
    uint8_t *device_pointer,
    uint64_t index,
    float value,
    const char *name)
{
    return SparkValidationCopyU8Value(
        device_pointer,
        index,
        SparkValidationFloatToFp8E4m3(value),
        name);
}

static bool SparkValidationSeedKeyValueCache(
    SparkValidationDeviceBuffers *buffers,
    uint32_t cache_slot_index,
    uint32_t token_index)
{
    uint32_t head_index;
    uint32_t dimension_index;

    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t key_offset;
            uint64_t value_offset;

            key_offset =
                (((uint64_t)cache_slot_index *
                  (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                  (uint64_t)head_index) *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
                (uint64_t)dimension_index;
            value_offset =
                (((uint64_t)cache_slot_index *
                  (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                  (uint64_t)head_index) *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                (uint64_t)dimension_index;
            if (!SparkValidationCopyU16Value(
                    buffers->key_nope_cache_bf16,
                    key_offset,
                    SparkValidationFloatToBf16(
                        SparkValidationSeedKeyNopeValue(
                            token_index,
                            head_index,
                            dimension_index)),
                    "copy seeded key_nope cache") ||
                !SparkValidationCopyU16Value(
                    buffers->value_cache_bf16,
                    value_offset,
                    SparkValidationFloatToBf16(
                        SparkValidationSeedValueCacheValue(
                            token_index,
                            head_index,
                            dimension_index)),
                    "copy seeded value cache"))
            {
                return false;
            }
        }
    }
    return true;
}

static bool SparkValidationSeedBoundReferenceCache(
    SparkValidationDeviceBuffers *buffers)
{
    uint16_t cache_seed[
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS];

    if (buffers == 0 ||
        buffers->mla_cache_bf16 == 0 ||
        buffers->key_nope_cache_bf16 == 0 ||
        buffers->value_cache_bf16 == 0)
    {
        return false;
    }
    memset(cache_seed, 0, sizeof(cache_seed));
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT0,
        0u);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT1,
        1u);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT2,
        2u);
    return SparkValidationCopyToDevice(
            buffers->mla_cache_bf16,
            cache_seed,
            sizeof(cache_seed),
            "copy routed reference mla cache") &&
        SparkValidationSeedKeyValueCache(
            buffers,
            SPARK_VALIDATION_REMAP_CACHE_SLOT0,
            0u) &&
        SparkValidationSeedKeyValueCache(
            buffers,
            SPARK_VALIDATION_REMAP_CACHE_SLOT1,
            1u) &&
        SparkValidationSeedKeyValueCache(
            buffers,
            SPARK_VALIDATION_REMAP_CACHE_SLOT2,
            2u);
}

static bool SparkValidationSeedProjectionFixtureWeights(
    SparkValidationDeviceBuffers *buffers)
{
    uint32_t head_index;
    uint32_t dimension_index;
    uint32_t expert_index;

    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                (uint64_t)dimension_index;
            weight_value =
                (0.0030f * (float)(head_index + 1u)) +
                (0.0005f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->query_latent_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy query latent fixture"))
            {
                return false;
            }
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                (uint64_t)dimension_index;
            weight_value =
                (0.0020f * (float)(head_index + 1u)) +
                (0.00025f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->query_rope_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy query rope fixture"))
            {
                return false;
            }
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->kv_latent_weight_bf16,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0040f + (0.0005f * (float)dimension_index)),
                "copy kv latent fixture"))
        {
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->key_rope_weight_bf16,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0050f + (0.00075f * (float)dimension_index)),
                "copy key rope fixture"))
        {
            return false;
        }
    }
    if (!SparkValidationCopyU16Value(
            buffers->raw_query_a_weight_bf16,
            0u,
            SparkValidationFloatToBf16(0.0010f),
            "copy raw q_a fixture") ||
        !SparkValidationCopyU16Value(
            buffers->raw_kv_b_weight_bf16,
            0u,
            SparkValidationFloatToBf16(0.0010f),
            "copy raw kv_b fixture") ||
        !SparkValidationCopyFp8FixtureValue(
            buffers->raw_query_a_weight_fp8_e4m3,
            0u,
            0.0010f,
            "copy raw fp8 q_a fixture") ||
        !SparkValidationCopyFp8FixtureValue(
            buffers->raw_kv_b_weight_fp8_e4m3,
            0u,
            0.0010f,
            "copy raw fp8 kv_b fixture"))
    {
        return false;
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION) +
                (uint64_t)dimension_index;
            weight_value =
                (0.0030f * (float)(head_index + 1u)) +
                (0.0005f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->raw_query_b_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy raw q_b latent fixture") ||
                !SparkValidationCopyFp8FixtureValue(
                    buffers->raw_query_b_weight_fp8_e4m3,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    weight_value,
                    "copy raw fp8 q_b latent fixture"))
            {
                return false;
            }
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION) +
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                (uint64_t)dimension_index;
            weight_value =
                (0.0020f * (float)(head_index + 1u)) +
                (0.00025f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->raw_query_b_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy raw q_b rope fixture") ||
                !SparkValidationCopyFp8FixtureValue(
                    buffers->raw_query_b_weight_fp8_e4m3,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    weight_value,
                    "copy raw fp8 q_b rope fixture"))
            {
                return false;
            }
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->raw_kv_a_weight_bf16,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0040f + (0.0005f * (float)dimension_index)),
                "copy raw kv_a latent fixture") ||
            !SparkValidationCopyFp8FixtureValue(
                buffers->raw_kv_a_weight_fp8_e4m3,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                0.0040f + (0.0005f * (float)dimension_index),
                "copy raw fp8 kv_a latent fixture"))
        {
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->raw_kv_a_weight_bf16,
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
                 (uint64_t)dimension_index) *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0050f + (0.00075f * (float)dimension_index)),
                "copy raw kv_a rope fixture") ||
            !SparkValidationCopyFp8FixtureValue(
                buffers->raw_kv_a_weight_fp8_e4m3,
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
                 (uint64_t)dimension_index) *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                0.0050f + (0.00075f * (float)dimension_index),
                "copy raw fp8 kv_a rope fixture"))
        {
            return false;
        }
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t key_row_index;
            uint64_t value_row_index;
            float key_weight_value;
            float value_weight_value;

            key_row_index =
                ((uint64_t)head_index *
                 (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
                (uint64_t)dimension_index;
            value_row_index =
                ((uint64_t)head_index *
                 (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                (uint64_t)dimension_index;
            key_weight_value =
                0.0060f + (0.00025f * (float)(head_index + 1u)) +
                (0.000125f * (float)(dimension_index + 1u));
            value_weight_value =
                0.0090f + (0.00050f * (float)(head_index + 1u)) +
                (0.000250f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->raw_kv_b_weight_bf16,
                    key_row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                    SparkValidationFloatToBf16(key_weight_value),
                    "copy raw kv_b key_nope fixture") ||
                !SparkValidationCopyFp8FixtureValue(
                    buffers->raw_kv_b_weight_fp8_e4m3,
                    key_row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                    key_weight_value,
                    "copy raw fp8 kv_b key_nope fixture") ||
                !SparkValidationCopyU16Value(
                    buffers->raw_kv_b_weight_bf16,
                    value_row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                    SparkValidationFloatToBf16(value_weight_value),
                    "copy raw kv_b value fixture") ||
                !SparkValidationCopyFp8FixtureValue(
                    buffers->raw_kv_b_weight_fp8_e4m3,
                    value_row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                    value_weight_value,
                    "copy raw fp8 kv_b value fixture"))
            {
                return false;
            }
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        uint32_t hidden_output_index;
        uint64_t attention_input_index;

        hidden_output_index = 3u + dimension_index;
        attention_input_index = dimension_index;
        if (!SparkValidationCopyU16Value(
                buffers->attention_output_weight_bf16,
                ((uint64_t)hidden_output_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION) +
                    attention_input_index,
                SparkValidationFloatToBf16(
                    0.0200f + (0.0010f * (float)dimension_index)),
                "copy attention output fixture") ||
            !SparkValidationCopyU16Value(
                buffers->restricted_lm_head_weight_bf16,
                ((uint64_t)4u *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)hidden_output_index,
                SparkValidationFloatToBf16(0.0100f),
                "copy restricted logit fixture 4") ||
            !SparkValidationCopyU16Value(
                buffers->restricted_lm_head_weight_bf16,
                ((uint64_t)9u *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)hidden_output_index,
                SparkValidationFloatToBf16(0.0350f),
                "copy restricted logit fixture 9") ||
            !SparkValidationCopyFp8FixtureValue(
                buffers->attention_output_weight_fp8_e4m3,
                ((uint64_t)hidden_output_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION) +
                    attention_input_index,
                0.0200f + (0.0010f * (float)dimension_index),
                "copy fp8 attention output fixture"))
        {
            return false;
        }
    }
    for (expert_index = 0u;
         expert_index < SPARK_VALIDATION_MOE_BOUND_EXPERT_COUNT;
         ++expert_index)
    {
        uint32_t intermediate_index;

        for (intermediate_index = 0u;
             intermediate_index < SPARK_VALIDATION_MOE_CHECKED_INTERMEDIATE;
             ++intermediate_index)
        {
            uint64_t gate_offset;
            uint64_t down_offset;
            uint32_t hidden_output_index;

            hidden_output_index = 3u + intermediate_index;
            gate_offset =
                ((((uint64_t)expert_index *
                   (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
                  (uint64_t)intermediate_index) *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
            down_offset =
                ((((uint64_t)expert_index *
                   (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                  (uint64_t)hidden_output_index) *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
                (uint64_t)intermediate_index;
            if (!SparkValidationCopyU16Value(
                    buffers->dense_gate_weight_bf16,
                    gate_offset,
                    SparkValidationFloatToBf16(
                        0.2500f + (0.015625f * (float)expert_index)),
                    "copy moe gate fixture") ||
                !SparkValidationCopyU16Value(
                    buffers->dense_up_weight_bf16,
                    gate_offset,
                    SparkValidationFloatToBf16(
                        0.5000f + (0.0078125f * (float)intermediate_index)),
                    "copy moe up fixture") ||
                !SparkValidationCopyU16Value(
                    buffers->dense_down_weight_bf16,
                    down_offset,
                    SparkValidationFloatToBf16(
                        0.1250f + (0.00390625f * (float)expert_index)),
                    "copy moe down fixture"))
            {
                return false;
            }
        }
    }
    return
        SparkValidationCopyU8Value(
            buffers->mtp_mxfp4_weight_payload_u8,
            11u *
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u),
            7u,
            "copy mtp payload fixture") &&
        SparkValidationCopyU8Value(
            buffers->mtp_mxfp4_scale_e8m0_u8,
            11u *
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE),
            127u,
            "copy mtp scale fixture");
}

static void SparkValidationCompletion(void *completion_context)
{
    SparkValidationCompletionState *state;

    state = (SparkValidationCompletionState *)completion_context;
    if (state != 0)
    {
        state->completion_count.fetch_add(1u, std::memory_order_release);
    }
}

static void SparkValidationDriverCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkValidationDriverCompletionState *state;

    state = (SparkValidationDriverCompletionState *)completion_context;
    if (state != 0 && completion != 0)
    {
        state->completion = *completion;
        state->completion_count.fetch_add(1u, std::memory_order_release);
    }
}

static bool SparkValidationAllocateRoutedLayerCaches(
    SparkValidationDeviceBuffers *buffers,
    uint64_t cache_count,
    uint64_t key_nope_cache_count,
    uint64_t value_cache_count)
{
    char cache_name[96];
    uint32_t routed_layer_offset;

    for (routed_layer_offset = 0u;
         routed_layer_offset < SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT;
         ++routed_layer_offset)
    {
        snprintf(cache_name, sizeof(cache_name), "cudaMalloc routed_layer%u_mla_cache", routed_layer_offset);
        if (!SparkValidationAllocateZeroed((void **)&buffers->routed_layer_mla_cache_bf16[routed_layer_offset], cache_count * 2u, cache_name))
            return false;
        snprintf(cache_name, sizeof(cache_name), "cudaMalloc routed_layer%u_key_nope_cache", routed_layer_offset);
        if (!SparkValidationAllocateZeroed((void **)&buffers->routed_layer_key_nope_cache_bf16[routed_layer_offset], key_nope_cache_count * 2u, cache_name))
            return false;
        snprintf(cache_name, sizeof(cache_name), "cudaMalloc routed_layer%u_value_cache", routed_layer_offset);
        if (!SparkValidationAllocateZeroed((void **)&buffers->routed_layer_value_cache_bf16[routed_layer_offset], value_cache_count * 2u, cache_name))
            return false;
    }
    return true;
}

static bool SparkValidationAllocateDeviceBuffers(
    SparkValidationDeviceBuffers *buffers)
{
    uint64_t hidden_vector_count;
    uint64_t hidden_count;
    uint64_t query_latent_vector_count;
    uint64_t query_latent_count;
    uint64_t query_rope_vector_count;
    uint64_t query_rope_count;
    uint64_t raw_query_a_count;
    uint64_t raw_query_b_count;
    uint64_t raw_kv_a_count;
    uint64_t raw_kv_b_count;
    uint64_t cache_count;
    uint64_t key_nope_cache_count;
    uint64_t value_cache_count;
    uint64_t attention_output_count;
    uint64_t query_latent_weight_count;
    uint64_t attention_output_weight_count;
    uint64_t raw_query_a_weight_count;
    uint64_t raw_query_b_weight_count;
    uint64_t raw_kv_a_weight_count;
    uint64_t raw_kv_b_weight_count;
    uint64_t raw_query_a_scale_count;
    uint64_t raw_query_b_scale_count;
    uint64_t raw_kv_a_scale_count;
    uint64_t raw_kv_b_scale_count;
    uint64_t attention_output_scale_count;
    uint64_t moe_intermediate_count;
    uint64_t moe_route_hidden_count;
    uint64_t moe_gate_weight_count;
    uint64_t moe_down_weight_count;
    uint64_t moe_router_weight_count;
    uint64_t routed_gate_up_payload_count;
    uint64_t routed_gate_up_scale_count;
    uint64_t routed_down_payload_count;
    uint64_t routed_down_scale_count;
    uint64_t routed_bound_expert_capacity;
    uint64_t restricted_weight_count;
    uint64_t final_token_count;
    uint64_t mtp_token_count;
    uint64_t mtp_payload_count;
    uint64_t mtp_scale_count;
    uint64_t rope_table_count;

    memset(buffers, 0, sizeof(*buffers));
    hidden_vector_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    query_latent_vector_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION;
    query_rope_vector_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION;
    hidden_count =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        hidden_vector_count;
    query_latent_count =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        query_latent_vector_count;
    attention_output_count =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION;
    query_rope_count =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        query_rope_vector_count;
    raw_query_a_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION;
    raw_query_b_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION;
    raw_kv_a_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION;
    raw_kv_b_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION;
    cache_count =
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
    key_nope_cache_count =
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
    value_cache_count =
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
    query_latent_weight_count =
        query_latent_vector_count *
        hidden_vector_count;
    attention_output_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION *
        hidden_vector_count;
    raw_query_a_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    raw_query_b_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION;
    raw_kv_a_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    raw_kv_b_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
    raw_query_a_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    raw_query_b_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    raw_kv_a_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    raw_kv_b_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    attention_output_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    moe_intermediate_count =
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    moe_route_hidden_count =
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    moe_gate_weight_count =
        SPARK_VALIDATION_MOE_BOUND_EXPERT_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    moe_down_weight_count =
        SPARK_VALIDATION_MOE_BOUND_EXPERT_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    moe_router_weight_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    routed_bound_expert_capacity =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    routed_gate_up_payload_count =
        routed_bound_expert_capacity *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
    routed_gate_up_scale_count =
        routed_bound_expert_capacity *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE);
    routed_down_payload_count =
        routed_bound_expert_capacity *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
         2u);
    routed_down_scale_count =
        routed_bound_expert_capacity *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE);
    restricted_weight_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    final_token_count =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
    mtp_token_count =
        final_token_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
    mtp_payload_count = restricted_weight_count / 2u;
    mtp_scale_count =
        restricted_weight_count /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE;
    rope_table_count =
        SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u);
    return
        SparkValidationAllocateZeroed((void **)&buffers->input_hidden_bf16, hidden_count * 2u, "cudaMalloc input_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->normalized_hidden_bf16, hidden_count * 2u, "cudaMalloc normalized_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_latent_bf16, query_latent_count * 2u, "cudaMalloc query_latent") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_rope_input_bf16, query_rope_count * 2u, "cudaMalloc query_rope") &&
        SparkValidationAllocateZeroed((void **)&buffers->key_rope_input_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION * 2u, "cudaMalloc key_rope") &&
        SparkValidationAllocateZeroed((void **)&buffers->current_kv_latent_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "cudaMalloc current_kv") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * raw_query_a_count * 2u, "cudaMalloc raw_query_a") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_normalized_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * raw_query_a_count * 2u, "cudaMalloc raw_query_a_normalized") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * raw_query_b_count * 2u, "cudaMalloc raw_query_b") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * raw_kv_a_count * 2u, "cudaMalloc raw_kv_a") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_normalized_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "cudaMalloc raw_kv_a_normalized") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_bf16, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * raw_kv_b_count * 2u, "cudaMalloc raw_kv_b") &&
        SparkValidationAllocateZeroed((void **)&buffers->mla_cache_bf16, cache_count * 2u, "cudaMalloc mla_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->key_nope_cache_bf16, key_nope_cache_count * 2u, "cudaMalloc key_nope_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->value_cache_bf16, value_cache_count * 2u, "cudaMalloc value_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_layer_mla_cache_bf16[1], cache_count * 2u, "cudaMalloc dense_layer1_mla_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_layer_key_nope_cache_bf16[1], key_nope_cache_count * 2u, "cudaMalloc dense_layer1_key_nope_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_layer_value_cache_bf16[1], value_cache_count * 2u, "cudaMalloc dense_layer1_value_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_layer_mla_cache_bf16[2], cache_count * 2u, "cudaMalloc dense_layer2_mla_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_layer_key_nope_cache_bf16[2], key_nope_cache_count * 2u, "cudaMalloc dense_layer2_key_nope_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_layer_value_cache_bf16[2], value_cache_count * 2u, "cudaMalloc dense_layer2_value_cache") &&
        SparkValidationAllocateRoutedLayerCaches(buffers, cache_count, key_nope_cache_count, value_cache_count) &&
        SparkValidationAllocateZeroed((void **)&buffers->rotated_query_rope_bf16, query_rope_count * 2u, "cudaMalloc rotated_query_rope") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_output_latent_bf16, attention_output_count * 2u, "cudaMalloc attention_output_value") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_projected_hidden_bf16, hidden_count * 2u, "cudaMalloc attention_projected_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->post_attention_hidden_bf16, hidden_count * 2u, "cudaMalloc post_attention_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->post_attention_normalized_hidden_bf16, hidden_count * 2u, "cudaMalloc post_attention_normalized_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_gate_bf16, moe_intermediate_count * 2u, "cudaMalloc moe_gate") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_up_bf16, moe_intermediate_count * 2u, "cudaMalloc moe_up") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_intermediate_bf16, moe_intermediate_count * 2u, "cudaMalloc moe_intermediate") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_route_output_bf16, moe_route_hidden_count * 2u, "cudaMalloc moe_route_output") &&
        SparkValidationAllocateZeroed((void **)&buffers->layer_output_hidden_bf16, hidden_count * 2u, "cudaMalloc layer_output_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_draft_hidden_bf16, hidden_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * 2u, "cudaMalloc mtp_draft_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_norm_weight_bf16, hidden_vector_count * 2u, "cudaMalloc attention_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_latent_weight_bf16, query_latent_weight_count * 2u, "cudaMalloc query_latent_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_rope_weight_bf16, query_rope_vector_count * hidden_vector_count * 2u, "cudaMalloc query_rope_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->key_rope_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION * hidden_vector_count * 2u, "cudaMalloc key_rope_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->kv_latent_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * hidden_vector_count * 2u, "cudaMalloc kv_latent_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_weight_bf16, raw_query_a_weight_count * 2u, "cudaMalloc raw_query_a_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_norm_weight_bf16, raw_query_a_count * 2u, "cudaMalloc raw_query_a_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_weight_bf16, raw_query_b_weight_count * 2u, "cudaMalloc raw_query_b_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_weight_bf16, raw_kv_a_weight_count * 2u, "cudaMalloc raw_kv_a_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "cudaMalloc raw_kv_a_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_weight_bf16, raw_kv_b_weight_count * 2u, "cudaMalloc raw_kv_b_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_weight_fp8_e4m3, raw_query_a_weight_count, "cudaMalloc raw_query_a_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_weight_scale_inv_f32, raw_query_a_scale_count * 4u, "cudaMalloc raw_query_a_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_weight_fp8_e4m3, raw_query_b_weight_count, "cudaMalloc raw_query_b_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_weight_scale_inv_f32, raw_query_b_scale_count * 4u, "cudaMalloc raw_query_b_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_weight_fp8_e4m3, raw_kv_a_weight_count, "cudaMalloc raw_kv_a_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_weight_scale_inv_f32, raw_kv_a_scale_count * 4u, "cudaMalloc raw_kv_a_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_weight_fp8_e4m3, raw_kv_b_weight_count, "cudaMalloc raw_kv_b_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_weight_scale_inv_f32, raw_kv_b_scale_count * 4u, "cudaMalloc raw_kv_b_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_output_weight_bf16, attention_output_weight_count * 2u, "cudaMalloc attention_output_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_output_weight_fp8_e4m3, attention_output_weight_count, "cudaMalloc attention_output_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_output_weight_scale_inv_f32, attention_output_scale_count * 4u, "cudaMalloc attention_output_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->post_attention_norm_weight_bf16, hidden_vector_count * 2u, "cudaMalloc post_attention_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_gate_weight_bf16, moe_gate_weight_count * 2u, "cudaMalloc moe_gate_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_up_weight_bf16, moe_gate_weight_count * 2u, "cudaMalloc moe_up_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->dense_down_weight_bf16, moe_down_weight_count * 2u, "cudaMalloc moe_down_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_router_weight_bf16, moe_router_weight_count * 2u, "cudaMalloc moe_router_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_gate_weight_payload_u8, routed_gate_up_payload_count, "cudaMalloc routed_gate_weight_payload") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_up_weight_payload_u8, routed_gate_up_payload_count, "cudaMalloc routed_up_weight_payload") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_down_weight_payload_u8, routed_down_payload_count, "cudaMalloc routed_down_weight_payload") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_gate_weight_scale_e4m3, routed_gate_up_scale_count, "cudaMalloc routed_gate_weight_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_up_weight_scale_e4m3, routed_gate_up_scale_count, "cudaMalloc routed_up_weight_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_down_weight_scale_e4m3, routed_down_scale_count, "cudaMalloc routed_down_weight_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_bound_expert_ids, routed_bound_expert_capacity * 4u, "cudaMalloc routed_bound_expert_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_gate_input_scale_f32, routed_bound_expert_capacity * 4u, "cudaMalloc routed_gate_input_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_gate_weight_scale_2_f32, routed_bound_expert_capacity * 4u, "cudaMalloc routed_gate_weight_payload_scale_2") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_up_input_scale_f32, routed_bound_expert_capacity * 4u, "cudaMalloc routed_up_input_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_up_weight_scale_2_f32, routed_bound_expert_capacity * 4u, "cudaMalloc routed_up_weight_payload_scale_2") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_down_input_scale_f32, routed_bound_expert_capacity * 4u, "cudaMalloc routed_down_input_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->routed_down_weight_scale_2_f32, routed_bound_expert_capacity * 4u, "cudaMalloc routed_down_weight_payload_scale_2") &&
        SparkValidationAllocateZeroed((void **)&buffers->final_norm_weight_bf16, hidden_vector_count * 2u, "cudaMalloc final_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_lm_head_weight_bf16, restricted_weight_count * 2u, "cudaMalloc restricted_lm_head_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_mxfp4_weight_payload_u8, mtp_payload_count, "cudaMalloc mtp_payload") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_mxfp4_scale_e8m0_u8, mtp_scale_count, "cudaMalloc mtp_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->cos_table, rope_table_count * 4u, "cudaMalloc cos_table") &&
        SparkValidationAllocateZeroed((void **)&buffers->sin_table, rope_table_count * 4u, "cudaMalloc sin_table") &&
        SparkValidationAllocateZeroed((void **)&buffers->dsa_token_scores, 64u * 4u, "cudaMalloc dsa_scores") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_router_score_bias_f32, SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * 4u, "cudaMalloc moe_router_bias") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_router_logits, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * 4u, "cudaMalloc moe_router_logits") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_topk_weights, SPARK_VALIDATION_MOE_ROUTE_COUNT * 4u, "cudaMalloc moe_topk_weights") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_logits, final_token_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT * 4u, "cudaMalloc restricted_logits") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_selected_token_scores, final_token_count * 4u, "cudaMalloc selected_scores") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_draft_logits, mtp_token_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT * 4u, "cudaMalloc mtp_logits") &&
        SparkValidationAllocateZeroed((void **)&buffers->positions, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * 4u, "cudaMalloc positions") &&
        SparkValidationAllocateZeroed((void **)&buffers->slot_mapping, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * 4u, "cudaMalloc slot_mapping") &&
        SparkValidationAllocateZeroed((void **)&buffers->block_table, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE * 4u, "cudaMalloc block_table") &&
        SparkValidationAllocateZeroed((void **)&buffers->context_lengths, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * 4u, "cudaMalloc context_lengths") &&
        SparkValidationAllocateZeroed((void **)&buffers->first_block_token_offsets, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * 4u, "cudaMalloc first_block_token_offsets") &&
        SparkValidationAllocateZeroed((void **)&buffers->sparse_token_indices, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT * 4u, "cudaMalloc sparse_indices") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_token_ids, SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT * 4u, "cudaMalloc restricted_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->moe_topk_expert_ids, SPARK_VALIDATION_MOE_ROUTE_COUNT * 4u, "cudaMalloc moe_topk_expert_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_selected_token_ids, final_token_count * 4u, "cudaMalloc selected_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_draft_token_ids, mtp_token_count * 4u, "cudaMalloc mtp_draft_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_target_token_ids, mtp_token_count * 4u, "cudaMalloc mtp_target_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_accept_mask, mtp_token_count * 4u, "cudaMalloc mtp_accept_mask") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_committed_token_ids, mtp_token_count * 4u, "cudaMalloc mtp_committed_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_event_counters, SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT * 4u, "cudaMalloc mtp_event_counters") &&
        SparkValidationAllocateZeroed((void **)&buffers->phase_clock_cycles, SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT * 8u, "cudaMalloc phase_clocks");
}

static bool SparkValidationInitializeDeviceInputs(
    SparkValidationDeviceBuffers *buffers)
{
    uint16_t cache_seed[
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS];
    float cos_table[
        SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u)];
    float sin_table[
        SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u)];
    uint16_t one_bf16[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    float fp8_scale_inv[
        ((SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        ((SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK)];
    uint16_t input_hidden[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t mtp_draft_hidden[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint32_t restricted_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    uint32_t sparse_token_indices[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT];
    uint32_t context_length;
    uint32_t position;
    uint32_t slot_mapping;
    uint32_t block_table[SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE];
    uint32_t first_block_token_offset;
    uint32_t mtp_target_token_ids[
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t moe_topk_expert_ids[SPARK_VALIDATION_MOE_ROUTE_COUNT];
    float moe_topk_weights[SPARK_VALIDATION_MOE_ROUTE_COUNT];
    uint32_t index;
    uint32_t rope_pair_index;

    memset(cache_seed, 0, sizeof(cache_seed));
    memset(input_hidden, 0, sizeof(input_hidden));
    memset(mtp_draft_hidden, 0, sizeof(mtp_draft_hidden));
    for (index = 0u;
         index < SPARK_VALIDATION_POSITION_COUNT *
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u);
         ++index)
    {
        cos_table[index] = 1.0f;
        sin_table[index] = 0.0f;
    }
    for (rope_pair_index = 0u;
         rope_pair_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT / 2u;
         ++rope_pair_index)
    {
        uint64_t table_offset;
        float angle;

        table_offset =
            ((uint64_t)SPARK_VALIDATION_CURRENT_POSITION *
             (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u)) +
            (uint64_t)rope_pair_index;
        angle = 0.125f * (float)(rope_pair_index + 1u);
        cos_table[table_offset] = cosf(angle);
        sin_table[table_offset] = sinf(angle);
    }
    for (index = 0u;
         index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         ++index)
    {
        one_bf16[index] = SparkValidationFloatToBf16(1.0f);
    }
    for (index = 0u; index < sizeof(fp8_scale_inv) / sizeof(fp8_scale_inv[0]); ++index)
    {
        fp8_scale_inv[index] = 1.0f;
    }
    input_hidden[0] = SparkValidationFloatToBf16(1.0f);
    mtp_draft_hidden[0] = SparkValidationFloatToBf16(1.0f);
    mtp_draft_hidden[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION] =
        SparkValidationFloatToBf16(1.0f);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT0,
        0u);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT1,
        1u);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT2,
        2u);
    for (index = 0u;
         index < SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT;
         ++index)
    {
        restricted_token_ids[index] = 1000u + index;
    }
    for (index = 0u;
         index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
         ++index)
    {
        sparse_token_indices[index] = index < SPARK_VALIDATION_CONTEXT_LENGTH
            ? index
            : SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID;
    }
    for (index = 0u; index < SPARK_VALIDATION_MOE_ROUTE_COUNT; ++index)
    {
        moe_topk_expert_ids[index] = index %
            SPARK_VALIDATION_MOE_BOUND_EXPERT_COUNT;
        moe_topk_weights[index] =
            1.0f / (float)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    }
    context_length = SPARK_VALIDATION_CONTEXT_LENGTH;
    position = SPARK_VALIDATION_CURRENT_POSITION;
    slot_mapping = SPARK_VALIDATION_CURRENT_CACHE_SLOT;
    block_table[0] = 1u;
    block_table[1] = 0u;
    first_block_token_offset = SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET;
    for (index = 0u;
         index < SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
         ++index)
    {
        mtp_target_token_ids[
            index * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT] =
            SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN;
        mtp_target_token_ids[
            (index * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
            1u] = SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN;
    }
    return
        SparkValidationCopyToDevice(buffers->mla_cache_bf16, cache_seed, sizeof(cache_seed), "copy seeded cache") &&
        SparkValidationCopyToDevice(buffers->input_hidden_bf16, input_hidden, sizeof(input_hidden), "copy input_hidden") &&
        SparkValidationCopyToDevice(buffers->mtp_draft_hidden_bf16, mtp_draft_hidden, sizeof(mtp_draft_hidden), "copy mtp_draft_hidden") &&
        SparkValidationCopyToDevice(buffers->cos_table, cos_table, sizeof(cos_table), "copy cos_table") &&
        SparkValidationCopyToDevice(buffers->sin_table, sin_table, sizeof(sin_table), "copy sin_table") &&
        SparkValidationCopyToDevice(buffers->attention_norm_weight_bf16, one_bf16, sizeof(one_bf16), "copy attention_norm_weight") &&
        SparkValidationCopyToDevice(buffers->post_attention_norm_weight_bf16, one_bf16, sizeof(one_bf16), "copy post_attention_norm_weight") &&
        SparkValidationCopyToDevice(buffers->raw_query_a_norm_weight_bf16, one_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION * 2u, "copy raw_query_a_norm_weight") &&
        SparkValidationCopyToDevice(buffers->raw_kv_a_norm_weight_bf16, one_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "copy raw_kv_a_norm_weight") &&
        SparkValidationCopyToDevice(buffers->raw_query_a_weight_scale_inv_f32, fp8_scale_inv, 16u * 48u * 4u, "copy raw_query_a_scale") &&
        SparkValidationCopyToDevice(buffers->raw_query_b_weight_scale_inv_f32, fp8_scale_inv, 128u * 16u * 4u, "copy raw_query_b_scale") &&
        SparkValidationCopyToDevice(buffers->raw_kv_a_weight_scale_inv_f32, fp8_scale_inv, 5u * 48u * 4u, "copy raw_kv_a_scale") &&
        SparkValidationCopyToDevice(buffers->raw_kv_b_weight_scale_inv_f32, fp8_scale_inv, 224u * 4u * 4u, "copy raw_kv_b_scale") &&
        SparkValidationCopyToDevice(buffers->attention_output_weight_scale_inv_f32, fp8_scale_inv, 48u * 128u * 4u, "copy attention_output_scale") &&
        SparkValidationCopyToDevice(buffers->final_norm_weight_bf16, one_bf16, sizeof(one_bf16), "copy final_norm_weight") &&
        SparkValidationCopyToDevice(buffers->restricted_token_ids, restricted_token_ids, sizeof(restricted_token_ids), "copy restricted_token_ids") &&
        SparkValidationCopyToDevice(buffers->moe_topk_expert_ids, moe_topk_expert_ids, sizeof(moe_topk_expert_ids), "copy moe_topk_expert_ids") &&
        SparkValidationCopyToDevice(buffers->moe_topk_weights, moe_topk_weights, sizeof(moe_topk_weights), "copy moe_topk_weights") &&
        SparkValidationCopyToDevice(buffers->sparse_token_indices, sparse_token_indices, sizeof(sparse_token_indices), "copy sparse indices") &&
        SparkValidationCopyToDevice(buffers->context_lengths, &context_length, sizeof(context_length), "copy context_length") &&
        SparkValidationCopyToDevice(buffers->positions, &position, sizeof(position), "copy position") &&
        SparkValidationCopyToDevice(buffers->slot_mapping, &slot_mapping, sizeof(slot_mapping), "copy slot_mapping") &&
        SparkValidationCopyToDevice(buffers->block_table, block_table, sizeof(block_table), "copy block_table") &&
        SparkValidationCopyToDevice(buffers->first_block_token_offsets, &first_block_token_offset, sizeof(first_block_token_offset), "copy first_block_token_offset") &&
        SparkValidationCopyToDevice(buffers->mtp_target_token_ids, mtp_target_token_ids, sizeof(mtp_target_token_ids), "copy mtp targets") &&
        SparkValidationSeedKeyValueCache(buffers, SPARK_VALIDATION_REMAP_CACHE_SLOT0, 0u) &&
        SparkValidationSeedKeyValueCache(buffers, SPARK_VALIDATION_REMAP_CACHE_SLOT1, 1u) &&
        SparkValidationSeedKeyValueCache(buffers, SPARK_VALIDATION_REMAP_CACHE_SLOT2, 2u) &&
        SparkValidationSeedProjectionFixtureWeights(buffers);
}

static bool SparkValidationSetDecodeScalars(
    SparkValidationDeviceBuffers *buffers,
    uint32_t position,
    uint32_t slot_mapping,
    uint32_t context_length)
{
    uint32_t *positions;
    uint32_t *slot_mappings;
    uint32_t *context_lengths;
    uint32_t *first_block_token_offsets;
    uint32_t *block_table;
    uint32_t *sparse_token_indices;
    size_t sequence_bytes;
    size_t block_table_bytes;
    size_t sparse_token_indices_bytes;
    uint32_t sequence_index;
    uint32_t block_index;
    uint32_t sparse_index;
    bool copied;

    sequence_bytes = (size_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        sizeof(uint32_t);
    block_table_bytes = (size_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (size_t)SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE * sizeof(uint32_t);
    sparse_token_indices_bytes =
        (size_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (size_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT *
        sizeof(uint32_t);
    positions = (uint32_t *)malloc(sequence_bytes);
    slot_mappings = (uint32_t *)malloc(sequence_bytes);
    context_lengths = (uint32_t *)malloc(sequence_bytes);
    first_block_token_offsets = (uint32_t *)malloc(sequence_bytes);
    block_table = (uint32_t *)malloc(block_table_bytes);
    sparse_token_indices = (uint32_t *)malloc(sparse_token_indices_bytes);
    if (positions == 0 || slot_mappings == 0 || context_lengths == 0 ||
        first_block_token_offsets == 0 || block_table == 0 ||
        sparse_token_indices == 0)
    {
        free(positions);
        free(slot_mappings);
        free(context_lengths);
        free(first_block_token_offsets);
        free(block_table);
        free(sparse_token_indices);
        return false;
    }

    for (sequence_index = 0u;
         sequence_index < SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
         ++sequence_index)
    {
        positions[sequence_index] = position;
        slot_mappings[sequence_index] = slot_mapping + sequence_index;
        context_lengths[sequence_index] = context_length;
        first_block_token_offsets[sequence_index] =
            SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET;
        for (block_index = 0u;
             block_index < SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE;
             ++block_index)
        {
            block_table[
                (sequence_index * SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE) +
                block_index] = block_index == 0u ? 1u : 0u;
        }
        for (sparse_index = 0u;
             sparse_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
             ++sparse_index)
        {
            sparse_token_indices[
                (sequence_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT) +
                sparse_index] = sparse_index < context_length
                    ? sparse_index
                    : SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID;
        }
    }
    copied =
        SparkValidationCopyToDevice(buffers->positions, positions, sequence_bytes, "copy positions") &&
        SparkValidationCopyToDevice(buffers->slot_mapping, slot_mappings, sequence_bytes, "copy slot_mappings") &&
        SparkValidationCopyToDevice(buffers->context_lengths, context_lengths, sequence_bytes, "copy context_lengths") &&
        SparkValidationCopyToDevice(buffers->first_block_token_offsets, first_block_token_offsets, sequence_bytes, "copy first_block_token_offsets") &&
        SparkValidationCopyToDevice(buffers->block_table, block_table, block_table_bytes, "copy block_table") &&
        SparkValidationCopyToDevice(buffers->sparse_token_indices, sparse_token_indices, sparse_token_indices_bytes, "copy sparse_token_indices");
    free(positions);
    free(slot_mappings);
    free(context_lengths);
    free(first_block_token_offsets);
    free(block_table);
    free(sparse_token_indices);
    return copied;
}

static void SparkValidationConfigureNode(
    SparkValidationDeviceBuffers *buffers,
    cudaStream_t cuda_stream,
    SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t use_dense_mlp,
    uint32_t use_attention_bf16)
{
    memset(pipeline_slot, 0, sizeof(*pipeline_slot));
    memset(cuda_slot_state, 0, sizeof(*cuda_slot_state));
    memset(node_context, 0, sizeof(*node_context));
    pipeline_slot->cuda_stream = (void *)cuda_stream;
    pipeline_slot->input_hidden_bf16 = buffers->input_hidden_bf16;
    pipeline_slot->normalized_hidden_bf16 = buffers->normalized_hidden_bf16;
    pipeline_slot->query_latent_bf16 = buffers->query_latent_bf16;
    pipeline_slot->query_rope_input_bf16 = buffers->query_rope_input_bf16;
    pipeline_slot->key_rope_input_bf16 = buffers->key_rope_input_bf16;
    pipeline_slot->current_kv_latent_bf16 = buffers->current_kv_latent_bf16;
    pipeline_slot->raw_query_a_bf16 = buffers->raw_query_a_bf16;
    pipeline_slot->raw_query_a_normalized_bf16 =
        buffers->raw_query_a_normalized_bf16;
    pipeline_slot->raw_query_b_bf16 = buffers->raw_query_b_bf16;
    pipeline_slot->raw_kv_a_bf16 = buffers->raw_kv_a_bf16;
    pipeline_slot->raw_kv_a_normalized_bf16 =
        buffers->raw_kv_a_normalized_bf16;
    pipeline_slot->raw_kv_b_bf16 = buffers->raw_kv_b_bf16;
    pipeline_slot->positions = buffers->positions;
    pipeline_slot->slot_mapping = buffers->slot_mapping;
    pipeline_slot->block_table = buffers->block_table;
    pipeline_slot->context_lengths = buffers->context_lengths;
    pipeline_slot->first_block_token_offsets = buffers->first_block_token_offsets;
    pipeline_slot->dsa_token_scores = buffers->dsa_token_scores;
    pipeline_slot->sparse_token_indices = buffers->sparse_token_indices;
    pipeline_slot->rotated_query_rope_bf16 = buffers->rotated_query_rope_bf16;
    pipeline_slot->attention_output_latent_bf16 = buffers->attention_output_latent_bf16;
    pipeline_slot->attention_projected_hidden_bf16 = buffers->attention_projected_hidden_bf16;
    pipeline_slot->post_attention_hidden_bf16 = buffers->post_attention_hidden_bf16;
    pipeline_slot->post_attention_normalized_hidden_bf16 =
        buffers->post_attention_normalized_hidden_bf16;
    pipeline_slot->moe_router_logits = buffers->moe_router_logits;
    pipeline_slot->moe_topk_expert_ids = buffers->moe_topk_expert_ids;
    pipeline_slot->moe_topk_weights = buffers->moe_topk_weights;
    pipeline_slot->moe_gate_bf16 = buffers->moe_gate_bf16;
    pipeline_slot->moe_up_bf16 = buffers->moe_up_bf16;
    pipeline_slot->moe_intermediate_bf16 = buffers->moe_intermediate_bf16;
    pipeline_slot->moe_route_output_bf16 = buffers->moe_route_output_bf16;
    pipeline_slot->layer_output_hidden_bf16 = buffers->layer_output_hidden_bf16;
    pipeline_slot->mtp_draft_hidden_bf16 = buffers->mtp_draft_hidden_bf16;
    pipeline_slot->restricted_logits = buffers->restricted_logits;
    pipeline_slot->mtp_draft_logits = buffers->mtp_draft_logits;
    pipeline_slot->restricted_selected_token_ids = buffers->restricted_selected_token_ids;
    pipeline_slot->restricted_selected_token_scores = buffers->restricted_selected_token_scores;
    pipeline_slot->mtp_draft_token_ids = buffers->mtp_draft_token_ids;
    pipeline_slot->mtp_target_token_ids = buffers->mtp_target_token_ids;
    pipeline_slot->mtp_accept_mask = buffers->mtp_accept_mask;
    pipeline_slot->mtp_committed_token_ids = buffers->mtp_committed_token_ids;
    pipeline_slot->mtp_event_counters = buffers->mtp_event_counters;
    pipeline_slot->phase_clock_cycles = buffers->phase_clock_cycles;
    cuda_slot_state->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION;
    node_context->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
    node_context->pipeline_slot_count = 1u;
    node_context->max_active_sequence_count =
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
    node_context->cache_token_capacity = SPARK_VALIDATION_CACHE_TOKEN_CAPACITY;
    node_context->kv_block_count = SPARK_VALIDATION_KV_BLOCK_COUNT;
    node_context->max_blocks_per_sequence =
        SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE;
    node_context->position_count = SPARK_VALIDATION_POSITION_COUNT;
    node_context->dsa_candidate_count = 64u;
    node_context->qk_scale = 0.0416666679f;
    node_context->rms_norm_epsilon = 0.000001f;
    node_context->cos_table = buffers->cos_table;
    node_context->sin_table = buffers->sin_table;
    node_context->mla_cache_bf16 = buffers->mla_cache_bf16;
    node_context->key_nope_cache_bf16 = buffers->key_nope_cache_bf16;
    node_context->value_cache_bf16 = buffers->value_cache_bf16;
    node_context->attention_norm_weight_bf16 =
        buffers->attention_norm_weight_bf16;
    node_context->query_latent_weight_bf16 =
        buffers->query_latent_weight_bf16;
    node_context->query_rope_weight_bf16 = buffers->query_rope_weight_bf16;
    node_context->key_rope_weight_bf16 = buffers->key_rope_weight_bf16;
    node_context->kv_latent_weight_bf16 = buffers->kv_latent_weight_bf16;
    node_context->raw_query_a_weight_bf16 =
        buffers->raw_query_a_weight_bf16;
    node_context->raw_query_a_norm_weight_bf16 =
        buffers->raw_query_a_norm_weight_bf16;
    node_context->raw_query_b_weight_bf16 =
        buffers->raw_query_b_weight_bf16;
    node_context->raw_kv_a_weight_bf16 = buffers->raw_kv_a_weight_bf16;
    node_context->raw_kv_a_norm_weight_bf16 =
        buffers->raw_kv_a_norm_weight_bf16;
    node_context->raw_kv_b_weight_bf16 = buffers->raw_kv_b_weight_bf16;
    node_context->raw_query_a_weight_fp8_e4m3 =
        buffers->raw_query_a_weight_fp8_e4m3;
    node_context->raw_query_a_weight_scale_inv_f32 =
        buffers->raw_query_a_weight_scale_inv_f32;
    node_context->raw_query_b_weight_fp8_e4m3 =
        buffers->raw_query_b_weight_fp8_e4m3;
    node_context->raw_query_b_weight_scale_inv_f32 =
        buffers->raw_query_b_weight_scale_inv_f32;
    node_context->raw_kv_a_weight_fp8_e4m3 =
        buffers->raw_kv_a_weight_fp8_e4m3;
    node_context->raw_kv_a_weight_scale_inv_f32 =
        buffers->raw_kv_a_weight_scale_inv_f32;
    node_context->raw_kv_b_weight_fp8_e4m3 =
        buffers->raw_kv_b_weight_fp8_e4m3;
    node_context->raw_kv_b_weight_scale_inv_f32 =
        buffers->raw_kv_b_weight_scale_inv_f32;
    node_context->attention_output_weight_bf16 =
        buffers->attention_output_weight_bf16;
    node_context->attention_output_weight_fp8_e4m3 =
        buffers->attention_output_weight_fp8_e4m3;
    node_context->attention_output_weight_scale_inv_f32 =
        buffers->attention_output_weight_scale_inv_f32;
    node_context->post_attention_norm_weight_bf16 =
        buffers->post_attention_norm_weight_bf16;
    node_context->dense_gate_weight_bf16 = buffers->dense_gate_weight_bf16;
    node_context->dense_up_weight_bf16 = buffers->dense_up_weight_bf16;
    node_context->dense_down_weight_bf16 = buffers->dense_down_weight_bf16;
    node_context->final_norm_weight_bf16 = buffers->final_norm_weight_bf16;
    node_context->restricted_lm_head_weight_bf16 =
        buffers->restricted_lm_head_weight_bf16;
    node_context->mtp_mxfp4_weight_payload_u8 =
        buffers->mtp_mxfp4_weight_payload_u8;
    node_context->mtp_mxfp4_scale_e8m0_u8 =
        buffers->mtp_mxfp4_scale_e8m0_u8;
    node_context->restricted_token_ids = buffers->restricted_token_ids;
    node_context->pipeline_slots = pipeline_slot;
    node_context->cuda_pipeline_slot_states = cuda_slot_state;
    node_context->linear_plans = 0;
    node_context->linear_plan_count = 0u;
    node_context->projection_mode = use_attention_bf16 != 0u
        ? SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_BF16
        : SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3;
    node_context->layer_progression_mode = use_dense_mlp != 0u
        ? SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP
        : SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY;
    node_context->moe_expert_count = SPARK_VALIDATION_MOE_BOUND_EXPERT_COUNT;
    node_context->moe_first_bound_expert_id = 0u;
    node_context->moe_bound_expert_count =
        SPARK_VALIDATION_MOE_BOUND_EXPERT_COUNT;
    node_context->moe_top_k = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    node_context->moe_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    node_context->dense_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION;
    if (use_dense_mlp != 0u)
    {
        node_context->mlp_execution_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE;
    }
    node_context->sparse_index_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED;
    node_context->launch_check_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR;
    node_context->phase_clock_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64;
}

static void SparkValidationEnableLayer3RouterTopK(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    node_context->layer_progression_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY;
    node_context->moe_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_first_bound_expert_id = 0u;
    node_context->moe_bound_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_router_weight_bf16 = buffers->moe_router_weight_bf16;
    node_context->moe_router_score_bias_f32 =
        buffers->moe_router_score_bias_f32;
    node_context->moe_routed_scaling_factor = 2.5f;
    node_context->moe_norm_topk_prob = 1u;
}

static void SparkValidationEnableLayer3SharedExpertBf16(
    SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    node_context->layer_progression_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP;
    node_context->dense_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    node_context->mlp_execution_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE;
}

static void SparkValidationEnableLayer3RoutedExpertNvfp4(
    const SparkValidationLayer3RoutedExpertNvfp4Fixture *fixture,
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    (void)fixture;
    SparkValidationEnableLayer3RouterTopK(buffers, node_context);
    node_context->layer_progression_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK;
    node_context->mlp_execution_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE;
    node_context->moe_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_first_bound_expert_id = 0u;
    node_context->moe_bound_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_top_k =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    node_context->moe_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
}

static bool SparkValidationBuildB12xMoePackPath(
    char *pack_path,
    uint32_t pack_path_bytes,
    uint32_t layer_index)
{
    const char *single_pack_path;
    const char *pack_directory;
    int written_bytes;

    if (pack_path == 0 || pack_path_bytes == 0u)
    {
        return false;
    }

    single_pack_path = getenv("GLM52_B12X_MOE_PACK");
    if (single_pack_path != 0 && single_pack_path[0] != '\0')
    {
        written_bytes = snprintf(
            pack_path,
            (size_t)pack_path_bytes,
            "%s",
            single_pack_path);
        return written_bytes >= 0 && (uint32_t)written_bytes < pack_path_bytes;
    }

    pack_directory = getenv("GLM52_B12X_MOE_PACK_DIR");
    if (pack_directory == 0 || pack_directory[0] == '\0')
    {
        fprintf(stderr, "set GLM52_B12X_MOE_PACK_DIR to the B12x resident MoE pack directory\n");
        return false;
    }

    written_bytes = snprintf(
        pack_path,
        (size_t)pack_path_bytes,
        "%s/glm52_layer_%04u_b12x_moe.spb12x",
        pack_directory,
        layer_index);
    return written_bytes >= 0 && (uint32_t)written_bytes < pack_path_bytes;
}

static bool SparkValidationBindB12xMoePlanForLayer(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t layer_index)
{
    SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateInfo create_info;
    SparkStatus status;
    char pack_path[PATH_MAX];
    uint32_t binding_index;

    if (buffers == 0 || node_context == 0 ||
        layer_index < SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX ||
        layer_index < buffers->routed_layer_base_index)
    {
        return false;
    }

    binding_index = layer_index - buffers->routed_layer_base_index;
    if (binding_index >= SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT)
    {
        fprintf(stderr, "layer %u has no validation B12x binding slot base=%u\n", layer_index, buffers->routed_layer_base_index);
        return false;
    }

    if (buffers->b12x_moe_binding_ready[binding_index] == 0u ||
        buffers->b12x_moe_binding_layer_indices[binding_index] != layer_index)
    {
        if (!SparkValidationBuildB12xMoePackPath(
                pack_path,
                (uint32_t)sizeof(pack_path),
                layer_index))
        {
            return false;
        }

        SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(
            &buffers->b12x_moe_bindings[binding_index]);
        memset(&create_info, 0, sizeof(create_info));
        create_info.abi_version =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_ABI_VERSION;
        create_info.layer_index = layer_index;
        create_info.maximum_active_sequence_count =
            node_context->max_active_sequence_count;
        create_info.pack_path = pack_path;

        status = SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateFromPackFile(
            &buffers->b12x_moe_bindings[binding_index],
            &create_info);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(
                stderr,
                "failed to bind B12x resident MoE pack for layer %u path=%s status=%d\n",
                layer_index,
                pack_path,
                (int)status);
            return false;
        }
        buffers->b12x_moe_binding_ready[binding_index] = 1u;
        buffers->b12x_moe_binding_layer_indices[binding_index] = layer_index;
    }

    node_context->b12x_moe_dispatch_plan =
        &buffers->b12x_moe_bindings[binding_index].dispatch_plan;
    node_context->mlp_execution_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE;
    node_context->moe_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_first_bound_expert_id = 0u;
    node_context->moe_bound_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_top_k =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    node_context->moe_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    return true;
}

static uint64_t SparkValidationFp8MoePackExpectedRegionBytes(
    uint32_t region_index)
{
    if (region_index == 0u)
    {
        return
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT *
            (uint64_t)(2u * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    }
    if (region_index == 1u)
    {
        return
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT *
            32ull * 48ull * 4ull;
    }
    if (region_index == 2u)
    {
        return
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    }
    if (region_index == 3u)
    {
        return
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT *
            48ull * 16ull * 4ull;
    }
    return 0u;
}

static bool SparkValidationCopyFp8MoePackRegionToDevice(
    FILE *file,
    const char *pack_path,
    uint64_t region_offset,
    uint64_t region_bytes,
    void *device_pointer)
{
    uint8_t *host_buffer;
    uint8_t *device_bytes;
    uint64_t copied_bytes;
    uint64_t chunk_bytes;
    size_t read_bytes;

    if (file == 0 || pack_path == 0 || device_pointer == 0 ||
        region_bytes == 0u)
    {
        return false;
    }
    host_buffer = (uint8_t *)malloc((size_t)SPARK_VALIDATION_FP8_MOE_COPY_CHUNK_BYTES);
    if (host_buffer == 0)
    {
        fprintf(stderr, "failed to allocate FP8 pack copy buffer\n");
        return false;
    }
    if (fseeko(file, (off_t)region_offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "failed to seek FP8 pack region path=%s offset=%llu\n", pack_path, (unsigned long long)region_offset);
        free(host_buffer);
        return false;
    }
    device_bytes = (uint8_t *)device_pointer;
    copied_bytes = 0u;
    while (copied_bytes < region_bytes)
    {
        chunk_bytes = region_bytes - copied_bytes;
        if (chunk_bytes > SPARK_VALIDATION_FP8_MOE_COPY_CHUNK_BYTES)
        {
            chunk_bytes = SPARK_VALIDATION_FP8_MOE_COPY_CHUNK_BYTES;
        }
        read_bytes = fread(host_buffer, 1u, (size_t)chunk_bytes, file);
        if (read_bytes != (size_t)chunk_bytes)
        {
            fprintf(stderr, "short read from FP8 pack path=%s offset=%llu wanted=%llu got=%llu\n", pack_path, (unsigned long long)(region_offset + copied_bytes), (unsigned long long)chunk_bytes, (unsigned long long)read_bytes);
            free(host_buffer);
            return false;
        }
        if (!SparkValidationCopyToDevice(
                device_bytes + copied_bytes,
                host_buffer,
                chunk_bytes,
                "copy FP8 MoE pack region"))
        {
            free(host_buffer);
            return false;
        }
        copied_bytes += chunk_bytes;
    }
    free(host_buffer);
    return true;
}

static bool SparkValidationReadFp8MoePackHeader(
    FILE *file,
    const char *pack_path,
    uint32_t layer_index,
    SparkValidationFp8MoePackHeader *header)
{
    uint8_t header_bytes[SPARK_VALIDATION_FP8_MOE_PACK_HEADER_BYTES];
    uint32_t region_index;

    if (file == 0 || pack_path == 0 || header == 0)
    {
        return false;
    }
    if (fseeko(file, 0, SEEK_SET) != 0 ||
        fread(header_bytes, 1u, sizeof(header_bytes), file) != sizeof(header_bytes))
    {
        fprintf(stderr, "failed to read FP8 MoE pack header path=%s\n", pack_path);
        return false;
    }
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, header_bytes, sizeof(header->magic));
    memcpy(header->fields, header_bytes + 16u, sizeof(header->fields));
    memcpy(
        header->region_offsets,
        header_bytes + 80u,
        sizeof(header->region_offsets));
    memcpy(
        header->region_bytes,
        header_bytes + 80u + sizeof(header->region_offsets),
        sizeof(header->region_bytes));
    if (memcmp(header->magic, "SPARKGLM52FP8", 13u) != 0 ||
        header->fields[0] != 1u ||
        header->fields[1] != SPARK_VALIDATION_FP8_MOE_PACK_HEADER_BYTES ||
        header->fields[2] != layer_index ||
        header->fields[3] != SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE ||
        header->fields[4] != SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        header->fields[5] != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
        header->fields[6] != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
        header->fields[7] != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
        header->fields[8] != SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE ||
        header->fields[9] != SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
        header->fields[10] != SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR ||
        header->fields[11] != SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3 ||
        header->fields[12] != SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        header->fields[13] != 121u)
    {
        fprintf(stderr, "invalid FP8 MoE pack header path=%s layer=%u\n", pack_path, layer_index);
        return false;
    }
    for (region_index = 0u;
         region_index < SPARK_VALIDATION_FP8_MOE_PACK_REGION_COUNT;
         ++region_index)
    {
        if (header->region_offsets[region_index] == 0u ||
            header->region_bytes[region_index] !=
                SparkValidationFp8MoePackExpectedRegionBytes(region_index))
        {
            fprintf(stderr, "invalid FP8 MoE pack region path=%s region=%u offset=%llu bytes=%llu\n", pack_path, region_index, (unsigned long long)header->region_offsets[region_index], (unsigned long long)header->region_bytes[region_index]);
            return false;
        }
    }
    return true;
}

static bool SparkValidationBuildFp8MoePackPath(
    char *pack_path,
    uint32_t pack_path_bytes,
    uint32_t layer_index)
{
    const char *single_pack_path;
    const char *pack_directory;
    int written_bytes;

    if (pack_path == 0 || pack_path_bytes == 0u)
    {
        return false;
    }
    single_pack_path = getenv("GLM52_FP8_MOE_PACK");
    if (single_pack_path != 0 && single_pack_path[0] != '\0')
    {
        written_bytes = snprintf(
            pack_path,
            (size_t)pack_path_bytes,
            "%s",
            single_pack_path);
        return written_bytes >= 0 && (uint32_t)written_bytes < pack_path_bytes;
    }
    pack_directory = getenv("GLM52_FP8_MOE_PACK_DIR");
    if (pack_directory == 0 || pack_directory[0] == '\0')
    {
        fprintf(stderr, "set GLM52_FP8_MOE_PACK_DIR to the FP8 resident MoE pack directory\n");
        return false;
    }
    written_bytes = snprintf(
        pack_path,
        (size_t)pack_path_bytes,
        "%s/glm52_layer_%04u_fp8_moe.spfp8",
        pack_directory,
        layer_index);
    return written_bytes >= 0 && (uint32_t)written_bytes < pack_path_bytes;
}

static bool SparkValidationBindFp8MoePlanForLayer(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t layer_index)
{
    SparkValidationFp8MoePackHeader header;
    SparkGlm52ResidentDecodeStageFp8MoePlan *plan;
    FILE *file;
    SparkStatus status;
    char pack_path[PATH_MAX];
    uint32_t binding_index;

    if (buffers == 0 || node_context == 0 ||
        layer_index < SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX ||
        layer_index < buffers->routed_layer_base_index)
    {
        return false;
    }
    binding_index = layer_index - buffers->routed_layer_base_index;
    if (binding_index >= SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT)
    {
        fprintf(stderr, "layer %u has no validation FP8 binding slot base=%u\n", layer_index, buffers->routed_layer_base_index);
        return false;
    }
    plan = &buffers->fp8_moe_plans[binding_index];
    if (buffers->fp8_moe_plan_ready[binding_index] == 0u ||
        buffers->fp8_moe_plan_layer_indices[binding_index] != layer_index)
    {
        if (!SparkValidationBuildFp8MoePackPath(
                pack_path,
                (uint32_t)sizeof(pack_path),
                layer_index))
        {
            return false;
        }
        file = fopen(pack_path, "rb");
        if (file == 0)
        {
            fprintf(stderr, "failed to open FP8 MoE pack path=%s\n", pack_path);
            return false;
        }
        if (!SparkValidationReadFp8MoePackHeader(
                file,
                pack_path,
                layer_index,
                &header))
        {
            fclose(file);
            return false;
        }
        memset(plan, 0, sizeof(*plan));
        plan->abi_version =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION;
        plan->capability_flags =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES;
        plan->maximum_active_sequence_count =
            node_context->max_active_sequence_count;
        plan->maximum_token_count = node_context->max_active_sequence_count;
        plan->expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
        plan->top_k = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
        plan->hidden_dimension = SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
        plan->intermediate_dimension =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
        plan->output_dtype =
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16;
        plan->cuda_architecture = 121u;
        plan->gate_up_order =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE;
        plan->weight_layout =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR;
        plan->scale_layout =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR;
        plan->quant_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3;
        plan->scale_block_size =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE;
        plan->validated_maximum_latency_ns = 1000000000ull;
        if (!SparkValidationAllocateDeviceOnly(
                (void **)&buffers->fp8_moe_w1_weight_fp8_e4m3[binding_index],
                header.region_bytes[0],
                "cudaMalloc fp8 moe w1 weight") ||
            !SparkValidationAllocateDeviceOnly(
                (void **)&buffers->fp8_moe_w1_scale_inv_f32[binding_index],
                header.region_bytes[1],
                "cudaMalloc fp8 moe w1 scale") ||
            !SparkValidationAllocateDeviceOnly(
                (void **)&buffers->fp8_moe_w2_weight_fp8_e4m3[binding_index],
                header.region_bytes[2],
                "cudaMalloc fp8 moe w2 weight") ||
            !SparkValidationAllocateDeviceOnly(
                (void **)&buffers->fp8_moe_w2_scale_inv_f32[binding_index],
                header.region_bytes[3],
                "cudaMalloc fp8 moe w2 scale"))
        {
            fclose(file);
            return false;
        }
        plan->w1_weight_fp8_e4m3 =
            buffers->fp8_moe_w1_weight_fp8_e4m3[binding_index];
        plan->w1_scale_inv_f32 =
            buffers->fp8_moe_w1_scale_inv_f32[binding_index];
        plan->w2_weight_fp8_e4m3 =
            buffers->fp8_moe_w2_weight_fp8_e4m3[binding_index];
        plan->w2_scale_inv_f32 =
            buffers->fp8_moe_w2_scale_inv_f32[binding_index];
        plan->workspace_bytes =
            SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceBytes(
                plan);
        if (plan->workspace_bytes == 0u ||
            !SparkValidationAllocateDeviceOnly(
                &buffers->fp8_moe_workspace[binding_index],
                plan->workspace_bytes,
                "cudaMalloc fp8 moe workspace"))
        {
            fclose(file);
            return false;
        }
        plan->workspace = buffers->fp8_moe_workspace[binding_index];
        if (!SparkValidationCopyFp8MoePackRegionToDevice(
                file,
                pack_path,
                header.region_offsets[0],
                header.region_bytes[0],
                buffers->fp8_moe_w1_weight_fp8_e4m3[binding_index]) ||
            !SparkValidationCopyFp8MoePackRegionToDevice(
                file,
                pack_path,
                header.region_offsets[1],
                header.region_bytes[1],
                buffers->fp8_moe_w1_scale_inv_f32[binding_index]) ||
            !SparkValidationCopyFp8MoePackRegionToDevice(
                file,
                pack_path,
                header.region_offsets[2],
                header.region_bytes[2],
                buffers->fp8_moe_w2_weight_fp8_e4m3[binding_index]) ||
            !SparkValidationCopyFp8MoePackRegionToDevice(
                file,
                pack_path,
                header.region_offsets[3],
                header.region_bytes[3],
                buffers->fp8_moe_w2_scale_inv_f32[binding_index]))
        {
            fclose(file);
            return false;
        }
        fclose(file);
        status = SparkGlm52Sm121RequiredDecodeStageBindFp8MoeGroupedReferencePlan(
            plan);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(
                stderr,
                "failed to bind FP8 resident MoE pack for layer %u path=%s status=%d\n",
                layer_index,
                pack_path,
                (int)status);
            return false;
        }
        buffers->fp8_moe_plan_ready[binding_index] = 1u;
        buffers->fp8_moe_plan_layer_indices[binding_index] = layer_index;
    }
    node_context->fp8_moe_plan = plan;
    node_context->model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT;
    node_context->mlp_execution_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE;
    node_context->moe_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_first_bound_expert_id = 0u;
    node_context->moe_bound_expert_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    node_context->moe_top_k =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    node_context->moe_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    return true;
}

static void SparkValidationEnableLayer3RoutedExpertFp8(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    SparkValidationEnableLayer3RouterTopK(buffers, node_context);
    node_context->layer_progression_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK;
    node_context->model_quantization_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT;
    node_context->reserved_execution_flags |=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION;
    node_context->mlp_execution_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE;
}

static void SparkValidationReleaseLinearPlanBinding(
    SparkValidationDeviceBuffers *buffers)
{
    if (buffers == 0 || buffers->linear_plan_binding == 0)
    {
        return;
    }
    SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(
        buffers->linear_plan_binding);
    buffers->linear_plan_binding = 0;
}

static uint32_t SparkValidationCountInitializedLinearPlans(
    const SparkGlm52ResidentDecodeStageLinearPlan *plans,
    uint32_t plan_count)
{
    uint32_t plan_index;
    uint32_t initialized_count;

    initialized_count = 0u;
    if (plans == 0)
    {
        return 0u;
    }
    for (plan_index = 0u; plan_index < plan_count; ++plan_index)
    {
        if (plans[plan_index].abi_version ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION &&
            plans[plan_index].plan_kind !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED)
        {
            initialized_count += 1u;
        }
    }
    return initialized_count;
}

static bool SparkValidationRawAttentionFp8PlansAvailable(
    const SparkValidationDeviceBuffers *buffers)
{
    return buffers != 0 &&
        buffers->raw_query_a_weight_fp8_e4m3 != 0 &&
        buffers->raw_query_a_weight_scale_inv_f32 != 0 &&
        buffers->raw_query_b_weight_fp8_e4m3 != 0 &&
        buffers->raw_query_b_weight_scale_inv_f32 != 0 &&
        buffers->raw_kv_a_weight_fp8_e4m3 != 0 &&
        buffers->raw_kv_a_weight_scale_inv_f32 != 0 &&
        buffers->raw_kv_b_weight_fp8_e4m3 != 0 &&
        buffers->raw_kv_b_weight_scale_inv_f32 != 0 &&
        buffers->attention_output_weight_fp8_e4m3 != 0 &&
        buffers->attention_output_weight_scale_inv_f32 != 0;
}

static bool SparkValidationBindRequiredQuantizedProjectionPlans(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t required_plan_mask,
    uint32_t use_quantized_attention_plans)
{
    SparkGlm52ResidentDecodeStageLinearPlan *plans;
    SparkStatus status;
    uint32_t plan_count;

    if ((required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS) == 0u ||
        use_quantized_attention_plans == 0u)
    {
        return true;
    }
    if (buffers == 0 || buffers->linear_plan_binding == 0 ||
        node_context == 0)
    {
        return false;
    }
    plans = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingMutablePlans(
        buffers->linear_plan_binding,
        &plan_count);
    if (plans == 0 ||
        plan_count < SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT)
    {
        fprintf(stderr, "resident tensor-core linear binder returned no mutable plans\n");
        return false;
    }
    status = SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedProjectionPlans(
        plans,
        plan_count);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(
            stderr,
            "failed to bind required quantized Q/KV/O projection launcher status=%d\n",
            (int)status);
        return false;
    }
    node_context->linear_plans = plans;
    node_context->linear_plan_count = plan_count;
    return true;
}

static bool SparkValidationBindRequiredLinearPlans(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    uint32_t required_plan_mask,
    uint32_t use_quantized_attention_plans)
{
    SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo create_info;
    SparkStatus status;
    const SparkGlm52ResidentDecodeStageLinearPlan *plans;
    uint32_t plan_count;
    uint32_t initialized_plan_count;
    uint32_t dense_intermediate_dimension;
    uint32_t use_quantized_dense_plans;

    if (buffers == 0 || node_context == 0)
    {
        return false;
    }
    if (required_plan_mask == 0u)
    {
        return true;
    }
    if ((required_plan_mask &
         ~SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_REQUIRED_GLM52_PREFIX) != 0u)
    {
        fprintf(stderr, "invalid required linear plan mask=0x%08x\n", required_plan_mask);
        return false;
    }
    if ((required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS) != 0u &&
        use_quantized_attention_plans != 0u &&
        !SparkValidationRawAttentionFp8PlansAvailable(buffers))
    {
        fprintf(
            stderr,
            "required quantized attention projection plans missing FP8 weight/scale buffers\n");
        return false;
    }
    if (buffers->linear_plan_binding != 0)
    {
        plans = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingPlans(
            buffers->linear_plan_binding,
            &plan_count);
        if (plans != 0 &&
            plan_count >= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT)
        {
            node_context->linear_plans = plans;
            node_context->linear_plan_count = plan_count;
            if (!SparkValidationBindRequiredQuantizedProjectionPlans(
                    buffers,
                    node_context,
                    required_plan_mask,
                    use_quantized_attention_plans))
            {
                return false;
            }
            if ((required_plan_mask &
                 (SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
                  SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
                  SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN)) != 0u)
            {
                node_context->mlp_execution_mode =
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE;
            }
            if ((required_plan_mask &
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS) != 0u)
            {
                node_context->projection_backend_mode =
                    use_quantized_attention_plans != 0u
                        ? SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE
                        : SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT;
            }
            if ((required_plan_mask &
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RESTRICTED_LOGITS) != 0u)
            {
                node_context->reserved_execution_flags |=
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS;
            }
            return true;
        }
        SparkValidationReleaseLinearPlanBinding(buffers);
    }
    dense_intermediate_dimension = node_context->dense_intermediate_dimension != 0u
        ? node_context->dense_intermediate_dimension
        : SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION;
    memset(&create_info, 0, sizeof(create_info));
    create_info.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION;
    create_info.maximum_active_sequence_count =
        node_context->max_active_sequence_count;
    create_info.dense_intermediate_dimension = dense_intermediate_dimension;
    create_info.expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    create_info.required_plan_mask = required_plan_mask;
    create_info.autotune_warmup_iterations = 2u;
    create_info.autotune_measurement_iterations = 5u;
    create_info.workspace_limit_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_DEFAULT_WORKSPACE_BYTES;
    create_info.cuda_stream = (void *)cuda_stream;
    use_quantized_dense_plans =
        node_context->model_quantization_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT &&
        node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE;
    create_info.dense_input_bf16 =
        buffers->post_attention_normalized_hidden_bf16;
    create_info.dense_gate_weight_bf16 = buffers->dense_gate_weight_bf16;
    create_info.dense_gate_weight_fp8_e4m3 = use_quantized_dense_plans != 0u
        ? node_context->dense_gate_weight_fp8_e4m3
        : 0;
    create_info.dense_gate_weight_scale_inv_f32 = use_quantized_dense_plans != 0u
        ? node_context->dense_gate_weight_scale_inv_f32
        : 0;
    create_info.dense_up_weight_bf16 = buffers->dense_up_weight_bf16;
    create_info.dense_up_weight_fp8_e4m3 = use_quantized_dense_plans != 0u
        ? node_context->dense_up_weight_fp8_e4m3
        : 0;
    create_info.dense_up_weight_scale_inv_f32 = use_quantized_dense_plans != 0u
        ? node_context->dense_up_weight_scale_inv_f32
        : 0;
    create_info.dense_down_weight_bf16 = buffers->dense_down_weight_bf16;
    create_info.dense_down_weight_fp8_e4m3 = use_quantized_dense_plans != 0u
        ? node_context->dense_down_weight_fp8_e4m3
        : 0;
    create_info.dense_down_weight_scale_inv_f32 = use_quantized_dense_plans != 0u
        ? node_context->dense_down_weight_scale_inv_f32
        : 0;
    create_info.dense_gate_output_bf16 = buffers->moe_gate_bf16;
    create_info.dense_up_output_bf16 = buffers->moe_up_bf16;
    create_info.dense_intermediate_bf16 = buffers->moe_intermediate_bf16;
    create_info.dense_down_output_bf16 = buffers->layer_output_hidden_bf16;
    create_info.router_input_bf16 =
        buffers->post_attention_normalized_hidden_bf16;
    create_info.router_weight_bf16 = buffers->moe_router_weight_bf16;
    create_info.router_logits_f32 = buffers->moe_router_logits;
    create_info.raw_projection_input_bf16 =
        buffers->normalized_hidden_bf16;
    create_info.raw_query_a_weight_bf16 =
        buffers->raw_query_a_weight_bf16;
    create_info.raw_query_a_weight_fp8_e4m3 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_query_a_weight_fp8_e4m3
            : 0;
    create_info.raw_query_a_weight_scale_inv_f32 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_query_a_weight_scale_inv_f32
            : 0;
    create_info.raw_query_a_output_bf16 =
        buffers->raw_query_a_bf16;
    create_info.raw_query_b_input_bf16 =
        buffers->raw_query_a_normalized_bf16;
    create_info.raw_query_b_weight_bf16 =
        buffers->raw_query_b_weight_bf16;
    create_info.raw_query_b_weight_fp8_e4m3 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_query_b_weight_fp8_e4m3
            : 0;
    create_info.raw_query_b_weight_scale_inv_f32 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_query_b_weight_scale_inv_f32
            : 0;
    create_info.raw_query_b_output_bf16 =
        buffers->raw_query_b_bf16;
    create_info.raw_kv_a_weight_bf16 =
        buffers->raw_kv_a_weight_bf16;
    create_info.raw_kv_a_weight_fp8_e4m3 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_kv_a_weight_fp8_e4m3
            : 0;
    create_info.raw_kv_a_weight_scale_inv_f32 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_kv_a_weight_scale_inv_f32
            : 0;
    create_info.raw_kv_a_output_bf16 =
        buffers->raw_kv_a_bf16;
    create_info.raw_kv_b_input_bf16 =
        buffers->raw_kv_a_normalized_bf16;
    create_info.raw_kv_b_weight_bf16 =
        buffers->raw_kv_b_weight_bf16;
    create_info.raw_kv_b_weight_fp8_e4m3 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_kv_b_weight_fp8_e4m3
            : 0;
    create_info.raw_kv_b_weight_scale_inv_f32 =
        use_quantized_attention_plans != 0u
            ? buffers->raw_kv_b_weight_scale_inv_f32
            : 0;
    create_info.raw_kv_b_output_bf16 =
        buffers->raw_kv_b_bf16;
    create_info.attention_output_input_bf16 =
        buffers->attention_output_latent_bf16;
    create_info.attention_output_weight_bf16 =
        buffers->attention_output_weight_bf16;
    create_info.attention_output_weight_fp8_e4m3 =
        use_quantized_attention_plans != 0u
            ? buffers->attention_output_weight_fp8_e4m3
            : 0;
    create_info.attention_output_weight_scale_inv_f32 =
        use_quantized_attention_plans != 0u
            ? buffers->attention_output_weight_scale_inv_f32
            : 0;
    create_info.attention_output_bf16 =
        buffers->attention_projected_hidden_bf16;
    create_info.restricted_logits_input_bf16 =
        buffers->normalized_hidden_bf16;
    create_info.restricted_lm_head_weight_bf16 =
        buffers->restricted_lm_head_weight_bf16;
    create_info.restricted_logits_f32 =
        buffers->restricted_logits;
    status = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreate(
        &buffers->linear_plan_binding,
        &create_info);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(
            stderr,
            "failed to bind resident tensor-core linear plans mask=0x%08x dense_intermediate=%u status=%d\n",
            required_plan_mask,
            dense_intermediate_dimension,
            (int)status);
        return false;
    }
    plans = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingPlans(
        buffers->linear_plan_binding,
        &plan_count);
    if (plans == 0 ||
        plan_count < SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT)
    {
        fprintf(stderr, "resident tensor-core linear binder returned no plans\n");
        SparkValidationReleaseLinearPlanBinding(buffers);
        return false;
    }
    node_context->linear_plans = plans;
    node_context->linear_plan_count = plan_count;
    if (!SparkValidationBindRequiredQuantizedProjectionPlans(
            buffers,
            node_context,
            required_plan_mask,
            use_quantized_attention_plans))
    {
        SparkValidationReleaseLinearPlanBinding(buffers);
        return false;
    }
    initialized_plan_count = SparkValidationCountInitializedLinearPlans(
        plans,
        plan_count);
    if ((required_plan_mask &
         (SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
          SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
          SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN)) != 0u)
    {
        node_context->mlp_execution_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE;
    }
    if ((required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS) != 0u)
    {
        node_context->projection_backend_mode =
            use_quantized_attention_plans != 0u
                ? SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE
                : SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT;
    }
    if ((required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RESTRICTED_LOGITS) != 0u)
    {
        node_context->reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS;
    }
    fprintf(
        stderr,
        "resident_linear_plans_ready=1 mask=0x%08x dense_intermediate=%u initialized_plan_count=%u plan_slot_count=%u\n",
        required_plan_mask,
        dense_intermediate_dimension,
        initialized_plan_count,
        plan_count);
    return true;
}

static bool SparkValidationInitializeDenseLayerCacheAliases(
    SparkValidationDeviceBuffers *buffers)
{
    if (buffers == 0 ||
        buffers->mla_cache_bf16 == 0 ||
        buffers->key_nope_cache_bf16 == 0 ||
        buffers->value_cache_bf16 == 0)
    {
        return false;
    }
    buffers->dense_layer_mla_cache_bf16[0] = buffers->mla_cache_bf16;
    buffers->dense_layer_key_nope_cache_bf16[0] =
        buffers->key_nope_cache_bf16;
    buffers->dense_layer_value_cache_bf16[0] = buffers->value_cache_bf16;
    return true;
}

static bool SparkValidationBindDenseLayerCache(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t layer_index)
{
    if (buffers == 0 ||
        node_context == 0 ||
        layer_index >= SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT ||
        buffers->dense_layer_mla_cache_bf16[layer_index] == 0 ||
        buffers->dense_layer_key_nope_cache_bf16[layer_index] == 0 ||
        buffers->dense_layer_value_cache_bf16[layer_index] == 0)
    {
        fprintf(stderr, "dense layer cache binding is invalid layer=%u\n", layer_index);
        return false;
    }
    buffers->mla_cache_bf16 = buffers->dense_layer_mla_cache_bf16[layer_index];
    buffers->key_nope_cache_bf16 =
        buffers->dense_layer_key_nope_cache_bf16[layer_index];
    buffers->value_cache_bf16 =
        buffers->dense_layer_value_cache_bf16[layer_index];
    node_context->mla_cache_bf16 = buffers->mla_cache_bf16;
    node_context->key_nope_cache_bf16 = buffers->key_nope_cache_bf16;
    node_context->value_cache_bf16 = buffers->value_cache_bf16;
    return true;
}

static bool SparkValidationBindRoutedLayerCache(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t layer_index)
{
    uint16_t *mla_cache;
    uint16_t *key_nope_cache;
    uint16_t *value_cache;
    uint32_t routed_layer_offset;

    mla_cache = 0;
    key_nope_cache = 0;
    value_cache = 0;
    routed_layer_offset = (buffers != 0 &&
        layer_index >= buffers->routed_layer_base_index)
        ? layer_index - buffers->routed_layer_base_index
        : UINT32_MAX;
    if (buffers != 0 &&
        routed_layer_offset < SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT)
    {
        mla_cache = buffers->routed_layer_mla_cache_bf16[routed_layer_offset];
        key_nope_cache =
            buffers->routed_layer_key_nope_cache_bf16[routed_layer_offset];
        value_cache =
            buffers->routed_layer_value_cache_bf16[routed_layer_offset];
    }
    if (buffers == 0 ||
        node_context == 0 ||
        mla_cache == 0 ||
        key_nope_cache == 0 ||
        value_cache == 0)
    {
        fprintf(stderr, "routed layer cache binding is invalid layer=%u\n", layer_index);
        return false;
    }
    buffers->mla_cache_bf16 = mla_cache;
    buffers->key_nope_cache_bf16 = key_nope_cache;
    buffers->value_cache_bf16 = value_cache;
    node_context->mla_cache_bf16 = buffers->mla_cache_bf16;
    node_context->key_nope_cache_bf16 = buffers->key_nope_cache_bf16;
    node_context->value_cache_bf16 = buffers->value_cache_bf16;
    return SparkValidationSeedBoundReferenceCache(buffers);
}

static bool SparkValidationLinearPlanIsReady(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_is_f32)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;

    if (node_context == 0 ||
        node_context->linear_plans == 0 ||
        plan_index >= node_context->linear_plan_count)
    {
        return false;
    }
    linear_plan = &node_context->linear_plans[plan_index];
    return linear_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION &&
        linear_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED &&
        linear_plan->input_dimension == input_dimension &&
        linear_plan->output_dimension == output_dimension &&
        linear_plan->maximum_active_sequence_count >=
            node_context->max_active_sequence_count &&
        linear_plan->output_is_f32 == output_is_f32;
}

static bool SparkValidationPreflightRequiredFastPath(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context == 0)
    {
        fprintf(stderr, "required fast-path preflight missing node context\n");
        return false;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
    {
        if (node_context->mlp_execution_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE)
        {
            fprintf(stderr, "dense prefix requires prebound tensor-core MLP linear plans\n");
            return false;
        }
        if (!SparkValidationLinearPlanIsReady(
                node_context,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                node_context->dense_intermediate_dimension,
                0u) ||
            !SparkValidationLinearPlanIsReady(
                node_context,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                node_context->dense_intermediate_dimension,
                0u) ||
            !SparkValidationLinearPlanIsReady(
                node_context,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
                node_context->dense_intermediate_dimension,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                0u))
        {
            fprintf(stderr, "dense prefix missing resident prebound dense gate/up/down linear plans\n");
            return false;
        }
    }
    if (node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY ||
        node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        if (!SparkValidationLinearPlanIsReady(
                node_context,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                node_context->moe_expert_count,
                1u))
        {
            fprintf(stderr, "routed B12x path missing resident prebound router logits linear plan\n");
            return false;
        }
    }
    if (node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK &&
        (node_context->mlp_execution_mode !=
             SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE ||
         node_context->b12x_moe_dispatch_plan == 0))
    {
        fprintf(stderr, "routed B12x path missing FlashInfer B12x MoE dispatch plan\n");
        return false;
    }
    return true;
}

static bool SparkValidationRunOnce(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    float *elapsed_microseconds)
{
    SparkValidationCompletionState completion_state;
    SparkGlm52ResidentDecodeStageBackendCompletion completion;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    SparkStatus status;
    float elapsed_milliseconds;
    bool succeeded;

    completion_state.completion_count.store(0u, std::memory_order_release);
    completion.function = SparkValidationCompletion;
    completion.context = &completion_state;
    start_event = 0;
    stop_event = 0;
    succeeded =
        SparkValidationCudaSucceeded(cudaEventCreate(&start_event), "cudaEventCreate start") &&
        SparkValidationCudaSucceeded(cudaEventCreate(&stop_event), "cudaEventCreate stop") &&
        SparkValidationCudaSucceeded(cudaEventRecord(start_event, cuda_stream), "cudaEventRecord start");
    if (!succeeded)
    {
        return false;
    }
    if (!SparkValidationPreflightRequiredFastPath(node_context))
    {
        return false;
    }
    status = SparkGlm52ResidentDecodeStageBackendSubmit(
        node_context,
        0u,
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT,
        0,
        &completion);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "backend submit failed: %s\n", SparkStatusToString(status));
        return false;
    }
    if (!SparkValidationCudaSucceeded(cudaEventRecord(stop_event, cuda_stream), "cudaEventRecord stop") ||
        !SparkValidationCudaSucceeded(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop"))
    {
        return false;
    }
    if (completion_state.completion_count.load(std::memory_order_acquire) != 1u)
    {
        fprintf(stderr, "backend completion callback did not run\n");
        return false;
    }
    if (!SparkValidationCudaSucceeded(
            cudaEventElapsedTime(&elapsed_milliseconds, start_event, stop_event),
            "cudaEventElapsedTime"))
    {
        return false;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    *elapsed_microseconds = elapsed_milliseconds * 1000.0f;
    return true;
}

static bool SparkValidationPromptTokenDelimiter(int32_t character)
{
    return character == ',' ||
        character == ' ' ||
        character == '\n' ||
        character == '\r' ||
        character == '\t';
}

static bool SparkValidationPromptTokenPush(
    SparkValidationPromptTokenWindow *prompt_tokens,
    uint64_t token_id)
{
    uint32_t index;

    if (token_id >= 154880ull)
    {
        return false;
    }
    if (prompt_tokens->token_count < SPARK_VALIDATION_CONTEXT_LENGTH)
    {
        prompt_tokens->token_ids[prompt_tokens->token_count] =
            (uint32_t)token_id;
        prompt_tokens->token_count += 1u;
    }
    else
    {
        for (index = 1u; index < SPARK_VALIDATION_CONTEXT_LENGTH; ++index)
        {
            prompt_tokens->token_ids[index - 1u] =
                prompt_tokens->token_ids[index];
        }
        prompt_tokens->token_ids[SPARK_VALIDATION_CONTEXT_LENGTH - 1u] =
            (uint32_t)token_id;
    }
    prompt_tokens->source_token_count += 1u;
    return true;
}

static bool SparkValidationReadPromptTokenWindow(
    const char *path,
    SparkValidationPromptTokenWindow *prompt_tokens)
{
    FILE *file;
    uint64_t token_id;
    uint32_t has_digit;
    int32_t character;

    memset(prompt_tokens, 0, sizeof(*prompt_tokens));
    file = fopen(path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "unable to open GLM52_PREFILL_TOKEN_IDS_FILE: %s\n", path);
        return false;
    }
    token_id = 0u;
    has_digit = 0u;
    while ((character = fgetc(file)) != EOF)
    {
        if (character >= '0' && character <= '9')
        {
            token_id = ((token_id * 10ull) + (uint64_t)(character - '0'));
            has_digit = 1u;
        }
        else if (SparkValidationPromptTokenDelimiter(character))
        {
            if (has_digit != 0u)
            {
                if (!SparkValidationPromptTokenPush(prompt_tokens, token_id))
                {
                    fclose(file);
                    fprintf(stderr, "invalid prompt token id in %s\n", path);
                    return false;
                }
                token_id = 0u;
                has_digit = 0u;
            }
        }
        else
        {
            fclose(file);
            fprintf(stderr, "invalid character in GLM52_PREFILL_TOKEN_IDS_FILE: %s\n", path);
            return false;
        }
    }
    if (has_digit != 0u &&
        !SparkValidationPromptTokenPush(prompt_tokens, token_id))
    {
        fclose(file);
        fprintf(stderr, "invalid trailing prompt token id in %s\n", path);
        return false;
    }
    fclose(file);
    if (prompt_tokens->source_token_count < SPARK_VALIDATION_CONTEXT_LENGTH)
    {
        fprintf(stderr, "GLM52_PREFILL_TOKEN_IDS_FILE needs at least %u tokens for this validation context: %s\n", SPARK_VALIDATION_CONTEXT_LENGTH, path);
        return false;
    }
    prompt_tokens->ready = 1u;
    fprintf(stderr, "prompt_token_window_ready=1 path=%s source_tokens=%u context_tokens=%u,%u,%u,%u\n",
        path,
        prompt_tokens->source_token_count,
        prompt_tokens->token_ids[0],
        prompt_tokens->token_ids[1],
        prompt_tokens->token_ids[2],
        prompt_tokens->token_ids[3]);
    return true;
}

static uint32_t SparkValidationResolveContextToken(
    const SparkValidationPromptTokenWindow *prompt_tokens,
    uint32_t input_token_id,
    uint32_t context_index)
{
    if (prompt_tokens != 0 && prompt_tokens->ready != 0u)
    {
        return prompt_tokens->token_ids[context_index];
    }
    if (context_index + 1u == SPARK_VALIDATION_CONTEXT_LENGTH)
    {
        return input_token_id;
    }
    return input_token_id - (SPARK_VALIDATION_CONTEXT_LENGTH - 1u) +
        context_index;
}

static bool SparkValidationRunPrefillKvBf16Fixture(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *model_directory,
    uint32_t input_token_id,
    const SparkValidationPromptTokenWindow *prompt_tokens,
    SparkValidationPrefillKvBf16Fixture *fixture)
{
    uint32_t prefill_index;

    memset(fixture, 0, sizeof(*fixture));
    fixture->first_token_id = prompt_tokens != 0 && prompt_tokens->ready != 0u
        ? prompt_tokens->token_ids[0]
        : input_token_id - (SPARK_VALIDATION_CONTEXT_LENGTH - 1u);
    fixture->token_count = SPARK_VALIDATION_CONTEXT_LENGTH - 1u;
    for (prefill_index = 0u; prefill_index < fixture->token_count; ++prefill_index)
    {
        float elapsed_microseconds;
        uint32_t token_id;
        uint32_t position;
        uint32_t slot_mapping;
        uint32_t context_length;

        token_id = prompt_tokens != 0 && prompt_tokens->ready != 0u
            ? prompt_tokens->token_ids[prefill_index]
            : fixture->first_token_id + prefill_index;
        position = SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET + prefill_index;
        slot_mapping = SPARK_VALIDATION_REMAP_CACHE_SLOT0 + prefill_index;
        context_length = prefill_index + 1u;
        if (!SparkValidationCopyInputEmbeddingBf16Row(
                model_directory,
                token_id,
                buffers->input_hidden_bf16,
                &fixture->copied_bytes) ||
            !SparkValidationSetDecodeScalars(
                buffers,
                position,
                slot_mapping,
                context_length) ||
            !SparkValidationRunOnce(
                node_context,
                cuda_stream,
                &elapsed_microseconds) ||
            !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize prefill"))
            return false;
    }
    if (!SparkValidationCopyInputEmbeddingBf16Row(
            model_directory,
            prompt_tokens != 0 && prompt_tokens->ready != 0u
                ? prompt_tokens->token_ids[SPARK_VALIDATION_CONTEXT_LENGTH - 1u]
                : input_token_id,
            buffers->input_hidden_bf16,
            &fixture->copied_bytes) ||
        !SparkValidationSetDecodeScalars(
            buffers,
            SPARK_VALIDATION_CURRENT_POSITION,
            SPARK_VALIDATION_CURRENT_CACHE_SLOT,
            SPARK_VALIDATION_CONTEXT_LENGTH))
        return false;
    fixture->ready = 1u;
    fprintf(stderr, "prefill_kv_bf16_fixture_ready=1 model_dir=%s first_token=%u token_count=%u bytes=%llu\n", model_directory, fixture->first_token_id, fixture->token_count, (unsigned long long)fixture->copied_bytes);
    return true;
}

static float SparkValidationReferenceAttentionValue(
    const float *query_latent_values,
    const float *query_rope_values,
    const float cache_key_nope_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION],
    const float cache_rope_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION],
    const float cache_value_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT],
    uint32_t output_dimension)
{
    float scores[SPARK_VALIDATION_CONTEXT_LENGTH];
    float maximum_score;
    float exponential_sum;
    float expected_value;
    uint32_t token_index;
    uint32_t dimension_index;
    uint32_t rope_dimension_index;

    maximum_score = -1.0e30f;
    for (token_index = 0u;
         token_index < SPARK_VALIDATION_CONTEXT_LENGTH;
         ++token_index)
    {
        scores[token_index] = 0.0f;
        for (dimension_index = 0u;
             dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
             ++dimension_index)
        {
            scores[token_index] +=
                query_latent_values[dimension_index] *
                cache_key_nope_values[token_index][dimension_index];
        }
        for (rope_dimension_index = 0u;
             rope_dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
             ++rope_dimension_index)
        {
            scores[token_index] +=
                query_rope_values[rope_dimension_index] *
                cache_rope_values[token_index][rope_dimension_index];
        }
        scores[token_index] *= 0.0416666679f;
        if (scores[token_index] > maximum_score)
        {
            maximum_score = scores[token_index];
        }
    }
    exponential_sum = 0.0f;
    expected_value = 0.0f;
    for (token_index = 0u;
         token_index < SPARK_VALIDATION_CONTEXT_LENGTH;
         ++token_index)
    {
        float weight;

        weight = expf(scores[token_index] - maximum_score);
        exponential_sum += weight;
        expected_value +=
            weight * cache_value_values[token_index][output_dimension];
    }
    return expected_value / exponential_sum;
}

static float SparkValidationReferenceRestrictedLogit(
    const uint16_t *normalized_hidden_bf16,
    const uint16_t *lm_head_row_bf16)
{
    float partial_sums[SPARK_VALIDATION_LOGIT_REDUCTION_THREADS];
    uint32_t thread_index;
    uint32_t stride;

    memset(partial_sums, 0, sizeof(partial_sums));
    for (thread_index = 0u;
         thread_index < SPARK_VALIDATION_LOGIT_REDUCTION_THREADS;
         ++thread_index)
    {
        uint32_t hidden_index;

        for (hidden_index = thread_index;
             hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
             hidden_index += SPARK_VALIDATION_LOGIT_REDUCTION_THREADS)
        {
            partial_sums[thread_index] +=
                SparkValidationBf16ToFloat(normalized_hidden_bf16[hidden_index]) *
                SparkValidationBf16ToFloat(lm_head_row_bf16[hidden_index]);
        }
    }
    for (stride = SPARK_VALIDATION_LOGIT_REDUCTION_THREADS >> 1u;
         stride != 0u;
         stride >>= 1u)
    {
        for (thread_index = 0u; thread_index < stride; ++thread_index)
        {
            partial_sums[thread_index] += partial_sums[thread_index + stride];
        }
    }
    return partial_sums[0];
}

static float SparkValidationReferenceLinearRow(
    const uint16_t *input_bf16,
    const uint16_t *weight_row_bf16,
    uint32_t input_dimension)
{
    float partial_sums[SPARK_VALIDATION_LOGIT_REDUCTION_THREADS];
    uint32_t thread_index;
    uint32_t stride;

    memset(partial_sums, 0, sizeof(partial_sums));
    for (thread_index = 0u;
         thread_index < SPARK_VALIDATION_LOGIT_REDUCTION_THREADS;
         ++thread_index)
    {
        uint32_t input_index;

        for (input_index = thread_index;
             input_index < input_dimension;
             input_index += SPARK_VALIDATION_LOGIT_REDUCTION_THREADS)
        {
            partial_sums[thread_index] +=
                SparkValidationBf16ToFloat(input_bf16[input_index]) *
                SparkValidationBf16ToFloat(weight_row_bf16[input_index]);
        }
    }
    for (stride = SPARK_VALIDATION_LOGIT_REDUCTION_THREADS >> 1u;
         stride != 0u;
         stride >>= 1u)
    {
        for (thread_index = 0u; thread_index < stride; ++thread_index)
        {
            partial_sums[thread_index] += partial_sums[thread_index + stride];
        }
    }
    return partial_sums[0];
}

static float SparkValidationReferenceRmsNormValue(
    const uint16_t *input_bf16,
    const uint16_t *weight_bf16,
    uint32_t dimension,
    uint32_t output_index,
    float epsilon)
{
    float partial_sums[SPARK_VALIDATION_LOGIT_REDUCTION_THREADS];
    uint32_t thread_index;
    uint32_t stride;
    float inverse_rms;

    memset(partial_sums, 0, sizeof(partial_sums));
    for (thread_index = 0u;
         thread_index < SPARK_VALIDATION_LOGIT_REDUCTION_THREADS;
         ++thread_index)
    {
        uint32_t dimension_index;

        for (dimension_index = thread_index;
             dimension_index < dimension;
             dimension_index += SPARK_VALIDATION_LOGIT_REDUCTION_THREADS)
        {
            float value;

            value = SparkValidationBf16ToFloat(input_bf16[dimension_index]);
            partial_sums[thread_index] += value * value;
        }
    }
    for (stride = SPARK_VALIDATION_LOGIT_REDUCTION_THREADS >> 1u;
         stride != 0u;
         stride >>= 1u)
    {
        for (thread_index = 0u; thread_index < stride; ++thread_index)
        {
            partial_sums[thread_index] += partial_sums[thread_index + stride];
        }
    }
    inverse_rms = 1.0f / sqrtf((partial_sums[0] / (float)dimension) + epsilon);
    return SparkValidationBf16ToFloat(input_bf16[output_index]) *
        inverse_rms *
        SparkValidationBf16ToFloat(weight_bf16[output_index]);
}

static void SparkValidationReferenceRmsNormVector(
    uint16_t *output_bf16,
    const uint16_t *input_bf16,
    const uint16_t *weight_bf16,
    uint32_t dimension,
    float epsilon)
{
    float sum_square;
    float inverse_rms;
    uint32_t index;

    sum_square = 0.0f;
    for (index = 0u; index < dimension; ++index)
    {
        float value;

        value = SparkValidationBf16ToFloat(input_bf16[index]);
        sum_square += value * value;
    }
    inverse_rms = 1.0f / sqrtf((sum_square / (float)dimension) + epsilon);
    for (index = 0u; index < dimension; ++index)
    {
        output_bf16[index] = SparkValidationFloatToBf16(
            SparkValidationBf16ToFloat(input_bf16[index]) *
            inverse_rms *
            SparkValidationBf16ToFloat(weight_bf16[index]));
    }
}

static bool SparkValidationCheckBf16ReferenceValueTracked(
    const char *name,
    uint32_t index,
    uint16_t observed_bf16,
    float expected_value,
    float *maximum_error)
{
    float observed_value;
    float expected_rounded;
    float tolerance;
    float error;

    observed_value = SparkValidationBf16ToFloat(observed_bf16);
    expected_rounded = SparkValidationBf16ToFloat(
        SparkValidationFloatToBf16(expected_value));
    error = fabsf(observed_value - expected_rounded);
    if (maximum_error != 0 && error > *maximum_error)
    {
        *maximum_error = error;
    }
    tolerance = SPARK_VALIDATION_REFERENCE_ABSOLUTE_TOLERANCE +
        (fabsf(expected_rounded) * SPARK_VALIDATION_REFERENCE_RELATIVE_TOLERANCE);
    if (error > tolerance)
    {
        fprintf(
            stderr,
            "%s sampled reference mismatch index=%u observed=%.8f expected=%.8f error=%.8f tolerance=%.8f\n",
            name,
            index,
            observed_value,
            expected_rounded,
            error,
            tolerance);
        return false;
    }
    return true;
}

static bool SparkValidationCheckBf16ReferenceValue(
    const char *name,
    uint32_t index,
    uint16_t observed_bf16,
    float expected_value)
{
    return SparkValidationCheckBf16ReferenceValueTracked(
        name,
        index,
        observed_bf16,
        expected_value,
        0);
}

static bool SparkValidationCheckBf16ReferenceVector(
    const char *name,
    const uint16_t *observed_bf16,
    const uint16_t *expected_bf16,
    uint32_t element_count,
    float *maximum_error)
{
    uint32_t index;

    for (index = 0u; index < element_count; ++index)
    {
        if (!SparkValidationCheckBf16ReferenceValueTracked(
                name,
                index,
                observed_bf16[index],
                SparkValidationBf16ToFloat(expected_bf16[index]),
                maximum_error))
            return false;
    }
    return true;
}

static bool SparkValidationCopyDeviceBf16Vector(
    uint16_t *host_vector,
    const uint16_t *device_vector,
    uint64_t element_count,
    const char *name)
{
    return SparkValidationCudaSucceeded(
        cudaMemcpy(
            host_vector,
            device_vector,
            (size_t)(element_count * 2u),
            cudaMemcpyDeviceToHost),
        name);
}

static bool SparkValidationReadHiddenBf16File(
    SparkValidationDeviceBuffers *buffers,
    const char *path)
{
    uint16_t *host_hidden;
    FILE *file;
    size_t hidden_bytes;
    size_t read_bytes;
    int extra_byte;

    if (buffers == 0 || path == 0 || path[0] == '\0')
    {
        fprintf(stderr, "pipeline input hidden path is missing\n");
        return false;
    }
    hidden_bytes = (size_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (size_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
        sizeof(uint16_t);
    host_hidden = (uint16_t *)malloc(hidden_bytes);
    if (host_hidden == 0)
    {
        return false;
    }
    file = fopen(path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "could not open pipeline input hidden %s\n", path);
        free(host_hidden);
        return false;
    }
    read_bytes = fread(host_hidden, 1u, hidden_bytes, file);
    extra_byte = fgetc(file);
    fclose(file);
    if (read_bytes != hidden_bytes || extra_byte != EOF)
    {
        fprintf(stderr, "pipeline input hidden must be exactly %llu bytes for active_sequences=%u: %s\n", (unsigned long long)hidden_bytes, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT, path);
        free(host_hidden);
        return false;
    }
    if (!SparkValidationCopyToDevice(
        buffers->input_hidden_bf16,
        host_hidden,
        hidden_bytes,
        "copy pipeline input hidden"))
    {
        free(host_hidden);
        return false;
    }
    free(host_hidden);
    return true;
}

static bool SparkValidationWriteHiddenBf16File(
    const char *path,
    const uint16_t *device_hidden)
{
    uint16_t *host_hidden;
    FILE *file;
    size_t hidden_bytes;
    size_t hidden_elements;
    size_t written_bytes;
    uint64_t checksum;
    uint32_t dimension_index;
    uint32_t nonzero_count;

    if (path == 0 || path[0] == '\0')
    {
        return true;
    }
    hidden_elements = (size_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (size_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    hidden_bytes = hidden_elements * sizeof(uint16_t);
    host_hidden = (uint16_t *)malloc(hidden_bytes);
    if (host_hidden == 0)
    {
        return false;
    }
    if (!SparkValidationCopyDeviceBf16Vector(
            host_hidden,
            device_hidden,
            (uint32_t)hidden_elements,
            "copy pipeline output hidden"))
    {
        free(host_hidden);
        return false;
    }
    checksum = 1469598103934665603ull;
    nonzero_count = 0u;
    for (dimension_index = 0u; dimension_index < hidden_elements; ++dimension_index)
    {
        if (host_hidden[dimension_index] != 0u)
            nonzero_count += 1u;
        checksum ^= (uint64_t)host_hidden[dimension_index];
        checksum *= 1099511628211ull;
    }
    if (nonzero_count == 0u)
    {
        fprintf(stderr, "pipeline output hidden stayed zero path=%s\n", path);
        free(host_hidden);
        return false;
    }
    file = fopen(path, "wb");
    if (file == 0)
    {
        fprintf(stderr, "could not open pipeline output hidden %s\n", path);
        free(host_hidden);
        return false;
    }
    written_bytes = fwrite(host_hidden, 1u, hidden_bytes, file);
    if (fclose(file) != 0 || written_bytes != hidden_bytes)
    {
        fprintf(stderr, "could not write pipeline output hidden %s\n", path);
        free(host_hidden);
        return false;
    }
    fprintf(stderr, "pipeline_hidden_bf16_written=%s active_sequences=%u bytes=%llu nonzero=%u checksum64=%llu\n", path, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT, (unsigned long long)hidden_bytes, nonzero_count, (unsigned long long)checksum);
    free(host_hidden);
    return true;
}

static bool SparkValidationHashDeviceBytes(
    const void *device_pointer,
    uint64_t byte_count,
    const char *label,
    uint64_t *hash_out,
    uint64_t *nonzero_out)
{
    uint8_t host_chunk[65536u];
    const uint8_t *device_bytes;
    uint64_t offset;
    uint64_t hash;
    uint64_t nonzero_count;
    uint64_t copy_bytes;
    uint64_t byte_index;

    if (device_pointer == 0 || label == 0 || hash_out == 0 ||
        nonzero_out == 0 || byte_count == 0u)
    {
        fprintf(stderr, "invalid device hash request label=%s bytes=%llu\n", label != 0 ? label : "(null)", (unsigned long long)byte_count);
        return false;
    }
    device_bytes = (const uint8_t *)device_pointer;
    hash = 1469598103934665603ull;
    nonzero_count = 0u;
    offset = 0u;
    while (offset < byte_count)
    {
        copy_bytes = byte_count - offset;
        if (copy_bytes > sizeof(host_chunk))
            copy_bytes = sizeof(host_chunk);
        if (!SparkValidationCudaSucceeded(
                cudaMemcpy(
                    host_chunk,
                    device_bytes + offset,
                    (size_t)copy_bytes,
                    cudaMemcpyDeviceToHost),
                label))
        {
            return false;
        }
        for (byte_index = 0u; byte_index < copy_bytes; ++byte_index)
        {
            if (host_chunk[byte_index] != 0u)
                nonzero_count += 1u;
            hash ^= (uint64_t)host_chunk[byte_index];
            hash *= 1099511628211ull;
        }
        offset += copy_bytes;
    }
    *hash_out = hash;
    *nonzero_out = nonzero_count;
    return true;
}

static bool SparkValidationTraceBuffer(
    const SparkValidationDeviceBuffers *buffers,
    uint32_t layer_index,
    const char *name,
    const void *device_pointer,
    uint64_t byte_count)
{
    uint64_t hash;
    uint64_t nonzero_count;

    (void)buffers;
    if (!SparkValidationHashDeviceBytes(
            device_pointer,
            byte_count,
            name,
            &hash,
            &nonzero_count))
    {
        return false;
    }
    fprintf(stderr, "accuracy_trace_buffer layer=%u name=%s bytes=%llu nonzero=%llu hash64=%llu\n", layer_index, name, (unsigned long long)byte_count, (unsigned long long)nonzero_count, (unsigned long long)hash);
    return true;
}

static bool SparkValidationMaybeTraceRoutedBuffers(
    const SparkValidationDeviceBuffers *buffers,
    uint32_t layer_index)
{
    const char *trace_text;
    uint64_t active_hidden_bytes;
    uint64_t active_query_latent_bytes;
    uint64_t active_query_rope_bytes;
    uint64_t active_key_rope_bytes;
    uint64_t active_kv_latent_bytes;
    uint64_t active_raw_query_a_bytes;
    uint64_t active_raw_query_b_bytes;
    uint64_t active_raw_kv_a_bytes;
    uint64_t active_raw_kv_b_bytes;
    uint64_t active_attention_output_bytes;
    uint64_t active_route_hidden_bytes;
    uint64_t active_router_logits_bytes;
    uint64_t active_route_count;

    trace_text = getenv("GLM52_ACCURACY_TRACE_BUFFERS");
    if (trace_text == 0 || trace_text[0] == '\0' ||
        strcmp(trace_text, "0") == 0)
    {
        return true;
    }
    if (buffers == 0)
    {
        fprintf(stderr, "accuracy trace has no buffers\n");
        return false;
    }
    active_route_count =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    active_hidden_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u;
    active_query_latent_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION * 2u;
    active_query_rope_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION * 2u;
    active_key_rope_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION * 2u;
    active_kv_latent_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u;
    active_raw_query_a_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION * 2u;
    active_raw_query_b_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION * 2u;
    active_raw_kv_a_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION * 2u;
    active_raw_kv_b_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION * 2u;
    active_attention_output_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION * 2u;
    active_route_hidden_bytes =
        active_route_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u;
    active_router_logits_bytes =
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT * 4u;
    fprintf(stderr, "accuracy_trace_begin layer=%u active_sequences=%u\n", layer_index, SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT);
    return
        SparkValidationTraceBuffer(buffers, layer_index, "input_hidden_bf16", buffers->input_hidden_bf16, active_hidden_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "normalized_hidden_bf16", buffers->normalized_hidden_bf16, active_hidden_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "raw_query_a_bf16", buffers->raw_query_a_bf16, active_raw_query_a_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "raw_query_a_normalized_bf16", buffers->raw_query_a_normalized_bf16, active_raw_query_a_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "raw_query_b_bf16", buffers->raw_query_b_bf16, active_raw_query_b_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "raw_kv_a_bf16", buffers->raw_kv_a_bf16, active_raw_kv_a_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "raw_kv_a_normalized_bf16", buffers->raw_kv_a_normalized_bf16, active_kv_latent_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "raw_kv_b_bf16", buffers->raw_kv_b_bf16, active_raw_kv_b_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "query_latent_bf16", buffers->query_latent_bf16, active_query_latent_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "query_rope_input_bf16", buffers->query_rope_input_bf16, active_query_rope_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "key_rope_input_bf16", buffers->key_rope_input_bf16, active_key_rope_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "current_kv_latent_bf16", buffers->current_kv_latent_bf16, active_kv_latent_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "rotated_query_rope_bf16", buffers->rotated_query_rope_bf16, active_query_rope_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "attention_output_latent_bf16", buffers->attention_output_latent_bf16, active_attention_output_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "attention_projected_hidden_bf16", buffers->attention_projected_hidden_bf16, active_hidden_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "post_attention_hidden_bf16", buffers->post_attention_hidden_bf16, active_hidden_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "post_attention_normalized_hidden_bf16", buffers->post_attention_normalized_hidden_bf16, active_hidden_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "moe_router_logits_f32", buffers->moe_router_logits, active_router_logits_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "moe_topk_expert_ids_u32", buffers->moe_topk_expert_ids, active_route_count * 4u) &&
        SparkValidationTraceBuffer(buffers, layer_index, "moe_topk_weights_f32", buffers->moe_topk_weights, active_route_count * 4u) &&
        SparkValidationTraceBuffer(buffers, layer_index, "moe_route_output_bf16", buffers->moe_route_output_bf16, active_route_hidden_bytes) &&
        SparkValidationTraceBuffer(buffers, layer_index, "layer_output_hidden_bf16", buffers->layer_output_hidden_bf16, active_hidden_bytes);
}

static void SparkValidationSetOutputHiddenOnly(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t enabled)
{
    if (enabled != 0u)
    {
        node_context->reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY;
    }
    else
    {
        node_context->reserved_execution_flags &=
            ~SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY;
    }
}

static bool SparkValidationReadFinalTokenEvidence(
    SparkValidationDeviceBuffers *buffers,
    uint32_t *selected_token_id,
    uint32_t *mtp_draft_token_id,
    uint32_t *mtp_reject_token_id)
{
    uint32_t mtp_draft_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t mtp_committed_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];

    if (buffers == 0 ||
        selected_token_id == 0 ||
        mtp_draft_token_id == 0 ||
        mtp_reject_token_id == 0)
    {
        return false;
    }
    memset(mtp_draft_token_ids, 0, sizeof(mtp_draft_token_ids));
    memset(mtp_committed_token_ids, 0, sizeof(mtp_committed_token_ids));
    if (!SparkValidationCudaSucceeded(
            cudaMemcpy(
                selected_token_id,
                buffers->restricted_selected_token_ids,
                sizeof(*selected_token_id),
                cudaMemcpyDeviceToHost),
            "copy final restricted selected token") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_draft_token_ids,
                buffers->mtp_draft_token_ids,
                sizeof(mtp_draft_token_ids),
                cudaMemcpyDeviceToHost),
            "copy final mtp draft tokens") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_committed_token_ids,
                buffers->mtp_committed_token_ids,
                sizeof(mtp_committed_token_ids),
                cudaMemcpyDeviceToHost),
            "copy final mtp committed tokens"))
    {
        return false;
    }
    *mtp_draft_token_id = mtp_draft_token_ids[0];
    *mtp_reject_token_id = mtp_committed_token_ids[1];
    return true;
}

static bool SparkValidationCopyDeviceBf16Row(
    uint16_t *host_row,
    const uint16_t *device_rows,
    uint32_t row_index,
    uint32_t row_width,
    const char *name)
{
    return SparkValidationCudaSucceeded(
        cudaMemcpy(
            host_row,
            &device_rows[(uint64_t)row_index * (uint64_t)row_width],
            (size_t)((uint64_t)row_width * 2u),
            cudaMemcpyDeviceToHost),
        name);
}

static bool SparkValidationCheckSampledAttentionReferences(
    SparkValidationDeviceBuffers *buffers)
{
    static const uint32_t query_a_indices[4] = {0u, 17u, 511u, 2047u};
    static const uint32_t query_b_indices[4] = {0u, 191u, 256u, 16383u};
    static const uint32_t kv_a_indices[4] = {0u, 511u, 512u, 575u};
    static const uint32_t kv_b_indices[4] = {0u, 191u, 192u, 28671u};
    uint16_t input_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t attention_norm_weight[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t normalized_hidden_reference[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t raw_query_a[SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION];
    uint16_t raw_query_a_norm_weight[SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION];
    uint16_t raw_query_a_normalized[SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION];
    uint16_t raw_query_b[SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION];
    uint16_t raw_kv_a[SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION];
    uint16_t raw_kv_a_norm_weight[SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION];
    uint16_t raw_kv_a_normalized[SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION];
    uint16_t raw_kv_b[SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION];
    uint16_t row[SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION];
    uint32_t sample_index;
    uint32_t hidden_index;

    if (!SparkValidationCopyDeviceBf16Vector(input_hidden, buffers->input_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference input_hidden") ||
        !SparkValidationCopyDeviceBf16Vector(attention_norm_weight, buffers->attention_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference attention_norm_weight") ||
        !SparkValidationCopyDeviceBf16Vector(raw_query_a, buffers->raw_query_a_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION, "copy reference raw_query_a") ||
        !SparkValidationCopyDeviceBf16Vector(raw_query_a_norm_weight, buffers->raw_query_a_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION, "copy reference raw_query_a_norm_weight") ||
        !SparkValidationCopyDeviceBf16Vector(raw_query_a_normalized, buffers->raw_query_a_normalized_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION, "copy reference raw_query_a_normalized") ||
        !SparkValidationCopyDeviceBf16Vector(raw_query_b, buffers->raw_query_b_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION, "copy reference raw_query_b") ||
        !SparkValidationCopyDeviceBf16Vector(raw_kv_a, buffers->raw_kv_a_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION, "copy reference raw_kv_a") ||
        !SparkValidationCopyDeviceBf16Vector(raw_kv_a_norm_weight, buffers->raw_kv_a_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION, "copy reference raw_kv_a_norm_weight") ||
        !SparkValidationCopyDeviceBf16Vector(raw_kv_a_normalized, buffers->raw_kv_a_normalized_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION, "copy reference raw_kv_a_normalized") ||
        !SparkValidationCopyDeviceBf16Vector(raw_kv_b, buffers->raw_kv_b_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION, "copy reference raw_kv_b"))
        return false;
    for (hidden_index = 0u;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         ++hidden_index)
    {
        normalized_hidden_reference[hidden_index] =
            SparkValidationFloatToBf16(
                SparkValidationReferenceRmsNormValue(
                    input_hidden,
                    attention_norm_weight,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                    hidden_index,
                    0.000001f));
    }
    for (sample_index = 0u; sample_index < 4u; ++sample_index)
    {
        uint32_t query_a_index;
        uint32_t query_b_index;
        uint32_t kv_a_index;
        uint32_t kv_b_index;

        query_a_index = query_a_indices[sample_index];
        query_b_index = query_b_indices[sample_index];
        kv_a_index = kv_a_indices[sample_index];
        kv_b_index = kv_b_indices[sample_index];
        if (!SparkValidationCopyDeviceBf16Row(
                row,
                buffers->raw_query_a_weight_bf16,
                query_a_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                "copy reference q_a row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "q_a_proj",
                query_a_index,
                raw_query_a[query_a_index],
                SparkValidationReferenceLinearRow(
                    normalized_hidden_reference,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)) ||
            !SparkValidationCheckBf16ReferenceValue(
                "q_a_norm",
                query_a_index,
                raw_query_a_normalized[query_a_index],
                SparkValidationReferenceRmsNormValue(
                    raw_query_a,
                    raw_query_a_norm_weight,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    query_a_index,
                    0.000001f)) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->raw_query_b_weight_bf16,
                query_b_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                "copy reference q_b row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "q_b_proj",
                query_b_index,
                raw_query_b[query_b_index],
                SparkValidationReferenceLinearRow(
                    raw_query_a_normalized,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION)) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->raw_kv_a_weight_bf16,
                kv_a_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                "copy reference kv_a row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "kv_a_proj",
                kv_a_index,
                raw_kv_a[kv_a_index],
                SparkValidationReferenceLinearRow(
                    normalized_hidden_reference,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)))
            return false;
        if (kv_a_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION &&
            !SparkValidationCheckBf16ReferenceValue(
                "kv_a_norm",
                kv_a_index,
                raw_kv_a_normalized[kv_a_index],
                SparkValidationReferenceRmsNormValue(
                    raw_kv_a,
                    raw_kv_a_norm_weight,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                    kv_a_index,
                    0.000001f)))
            return false;
        if (!SparkValidationCopyDeviceBf16Row(
                row,
                buffers->raw_kv_b_weight_bf16,
                kv_b_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                "copy reference kv_b row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "kv_b_proj",
                kv_b_index,
                raw_kv_b[kv_b_index],
                SparkValidationReferenceLinearRow(
                    raw_kv_a_normalized,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION)))
            return false;
    }
    return true;
}

static bool SparkValidationCheckSampledOutputAndDenseReferences(
    SparkValidationDeviceBuffers *buffers)
{
    static const uint32_t hidden_indices[4] = {0u, 3u, 1024u, 6143u};
    static const uint32_t intermediate_indices[4] = {0u, 17u, 2047u, 12287u};
    uint16_t input_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t attention_output[SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION];
    uint16_t attention_projected_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_norm_weight[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_normalized[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t gate[SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION];
    uint16_t up[SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION];
    uint16_t intermediate[SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION];
    uint16_t layer_output[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t row[SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION];
    uint32_t sample_index;

    if (!SparkValidationCopyDeviceBf16Vector(input_hidden, buffers->input_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference output input_hidden") ||
        !SparkValidationCopyDeviceBf16Vector(attention_output, buffers->attention_output_latent_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION, "copy reference attention_output") ||
        !SparkValidationCopyDeviceBf16Vector(attention_projected_hidden, buffers->attention_projected_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference attention_projected") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_hidden, buffers->post_attention_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference post_attention") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_norm_weight, buffers->post_attention_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference post_attention_norm_weight") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_normalized, buffers->post_attention_normalized_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference post_attention_normalized") ||
        !SparkValidationCopyDeviceBf16Vector(gate, buffers->moe_gate_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION, "copy reference dense gate") ||
        !SparkValidationCopyDeviceBf16Vector(up, buffers->moe_up_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION, "copy reference dense up") ||
        !SparkValidationCopyDeviceBf16Vector(intermediate, buffers->moe_intermediate_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION, "copy reference dense intermediate") ||
        !SparkValidationCopyDeviceBf16Vector(layer_output, buffers->layer_output_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy reference layer output"))
        return false;
    for (sample_index = 0u; sample_index < 4u; ++sample_index)
    {
        uint32_t hidden_index;
        uint32_t intermediate_index;
        float gate_value;
        float up_value;
        float silu_value;

        hidden_index = hidden_indices[sample_index];
        intermediate_index = intermediate_indices[sample_index];
        if (!SparkValidationCopyDeviceBf16Row(
                row,
                buffers->attention_output_weight_bf16,
                hidden_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
                "copy reference o_proj row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "o_proj",
                hidden_index,
                attention_projected_hidden[hidden_index],
                SparkValidationReferenceLinearRow(
                    attention_output,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION)) ||
            !SparkValidationCheckBf16ReferenceValue(
                "attention_residual",
                hidden_index,
                post_attention_hidden[hidden_index],
                SparkValidationBf16ToFloat(input_hidden[hidden_index]) +
                    SparkValidationBf16ToFloat(attention_projected_hidden[hidden_index])) ||
            !SparkValidationCheckBf16ReferenceValue(
                "post_attention_norm",
                hidden_index,
                post_attention_normalized[hidden_index],
                SparkValidationReferenceRmsNormValue(
                    post_attention_hidden,
                    post_attention_norm_weight,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                    hidden_index,
                    0.000001f)) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->dense_gate_weight_bf16,
                intermediate_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                "copy reference dense gate row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "dense_gate",
                intermediate_index,
                gate[intermediate_index],
                SparkValidationReferenceLinearRow(
                    post_attention_normalized,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->dense_up_weight_bf16,
                intermediate_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                "copy reference dense up row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "dense_up",
                intermediate_index,
                up[intermediate_index],
                SparkValidationReferenceLinearRow(
                    post_attention_normalized,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)))
            return false;
        gate_value = SparkValidationBf16ToFloat(gate[intermediate_index]);
        up_value = SparkValidationBf16ToFloat(up[intermediate_index]);
        silu_value = gate_value / (1.0f + expf(-gate_value));
        if (!SparkValidationCheckBf16ReferenceValue(
                "dense_silu",
                intermediate_index,
                intermediate[intermediate_index],
                silu_value * up_value) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->dense_down_weight_bf16,
                hidden_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,
                "copy reference dense down row") ||
            !SparkValidationCheckBf16ReferenceValue(
                "dense_down_residual",
                hidden_index,
                layer_output[hidden_index],
                SparkValidationBf16ToFloat(post_attention_hidden[hidden_index]) +
                    SparkValidationReferenceLinearRow(
                        intermediate,
                        row,
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION)))
            return false;
    }
    return true;
}

static bool SparkValidationCheckFullOutputAndDenseReferences(
    SparkValidationDeviceBuffers *buffers,
    float *maximum_error)
{
    uint16_t input_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t attention_output[SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION];
    uint16_t attention_projected_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_norm_weight[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_normalized[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_normalized_reference[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t gate[SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION];
    uint16_t up[SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION];
    uint16_t intermediate[SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION];
    uint16_t layer_output[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t row[SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION];
    uint32_t index;

    *maximum_error = 0.0f;
    if (!SparkValidationCopyDeviceBf16Vector(input_hidden, buffers->input_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy full reference input_hidden") ||
        !SparkValidationCopyDeviceBf16Vector(attention_output, buffers->attention_output_latent_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION, "copy full reference attention_output") ||
        !SparkValidationCopyDeviceBf16Vector(attention_projected_hidden, buffers->attention_projected_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy full reference attention_projected") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_hidden, buffers->post_attention_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy full reference post_attention") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_norm_weight, buffers->post_attention_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy full reference post_attention_norm_weight") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_normalized, buffers->post_attention_normalized_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy full reference post_attention_normalized") ||
        !SparkValidationCopyDeviceBf16Vector(gate, buffers->moe_gate_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION, "copy full reference dense gate") ||
        !SparkValidationCopyDeviceBf16Vector(up, buffers->moe_up_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION, "copy full reference dense up") ||
        !SparkValidationCopyDeviceBf16Vector(intermediate, buffers->moe_intermediate_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION, "copy full reference dense intermediate") ||
        !SparkValidationCopyDeviceBf16Vector(layer_output, buffers->layer_output_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy full reference layer output"))
        return false;
    for (index = 0u; index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION; ++index)
    {
        if (!SparkValidationCopyDeviceBf16Row(row, buffers->attention_output_weight_bf16, index, SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION, "copy full reference o_proj row") ||
            !SparkValidationCheckBf16ReferenceValueTracked("o_proj_full", index, attention_projected_hidden[index], SparkValidationReferenceLinearRow(attention_output, row, SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION), maximum_error) ||
            !SparkValidationCheckBf16ReferenceValueTracked("attention_residual_full", index, post_attention_hidden[index], SparkValidationBf16ToFloat(input_hidden[index]) + SparkValidationBf16ToFloat(attention_projected_hidden[index]), maximum_error))
            return false;
    }
    SparkValidationReferenceRmsNormVector(post_attention_normalized_reference, post_attention_hidden, post_attention_norm_weight, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, 0.000001f);
    if (!SparkValidationCheckBf16ReferenceVector("post_attention_norm_full", post_attention_normalized, post_attention_normalized_reference, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, maximum_error))
        return false;
    for (index = 0u; index < SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION; ++index)
    {
        float gate_value;
        float up_value;
        float silu_value;

        gate_value = SparkValidationBf16ToFloat(gate[index]);
        up_value = SparkValidationBf16ToFloat(up[index]);
        silu_value = gate_value / (1.0f + expf(-gate_value));
        if (!SparkValidationCheckBf16ReferenceValueTracked("dense_silu_full", index, intermediate[index], silu_value * up_value, maximum_error))
            return false;
    }
    for (index = 0u; index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION; ++index)
    {
        if (!SparkValidationCopyDeviceBf16Row(row, buffers->dense_down_weight_bf16, index, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION, "copy full reference dense down row") ||
            !SparkValidationCheckBf16ReferenceValueTracked("dense_down_residual_full", index, layer_output[index], SparkValidationBf16ToFloat(post_attention_hidden[index]) + SparkValidationReferenceLinearRow(intermediate, row, SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION), maximum_error))
            return false;
    }
    return true;
}

static bool SparkValidationCheckLayer3SharedExpertReferences(
    SparkValidationDeviceBuffers *buffers,
    float *maximum_error)
{
    static const uint32_t hidden_indices[4] = {0u, 3u, 1024u, 6143u};
    static const uint32_t intermediate_indices[4] = {0u, 17u, 511u, 2047u};
    uint16_t post_attention_hidden[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_norm_weight[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t post_attention_normalized[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t gate[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION];
    uint16_t up[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION];
    uint16_t intermediate[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION];
    uint16_t layer_output[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t row[SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint32_t sample_index;

    *maximum_error = 0.0f;
    if (!SparkValidationCopyDeviceBf16Vector(post_attention_hidden, buffers->post_attention_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy shared expert post_attention") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_norm_weight, buffers->post_attention_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy shared expert norm weight") ||
        !SparkValidationCopyDeviceBf16Vector(post_attention_normalized, buffers->post_attention_normalized_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy shared expert normalized") ||
        !SparkValidationCopyDeviceBf16Vector(gate, buffers->moe_gate_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION, "copy shared expert gate") ||
        !SparkValidationCopyDeviceBf16Vector(up, buffers->moe_up_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION, "copy shared expert up") ||
        !SparkValidationCopyDeviceBf16Vector(intermediate, buffers->moe_intermediate_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION, "copy shared expert intermediate") ||
        !SparkValidationCopyDeviceBf16Vector(layer_output, buffers->layer_output_hidden_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, "copy shared expert layer output"))
    {
        return false;
    }
    for (sample_index = 0u; sample_index < 4u; ++sample_index)
    {
        uint32_t hidden_index;
        uint32_t intermediate_index;
        float gate_value;
        float up_value;
        float silu_value;

        hidden_index = hidden_indices[sample_index];
        intermediate_index = intermediate_indices[sample_index];
        if (!SparkValidationCheckBf16ReferenceValueTracked(
                "layer3_shared_post_attention_norm",
                hidden_index,
                post_attention_normalized[hidden_index],
                SparkValidationReferenceRmsNormValue(
                    post_attention_hidden,
                    post_attention_norm_weight,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                    hidden_index,
                    0.000001f),
                maximum_error) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->dense_gate_weight_bf16,
                intermediate_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                "copy shared expert gate row") ||
            !SparkValidationCheckBf16ReferenceValueTracked(
                "layer3_shared_gate",
                intermediate_index,
                gate[intermediate_index],
                SparkValidationReferenceLinearRow(
                    post_attention_normalized,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),
                maximum_error) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->dense_up_weight_bf16,
                intermediate_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                "copy shared expert up row") ||
            !SparkValidationCheckBf16ReferenceValueTracked(
                "layer3_shared_up",
                intermediate_index,
                up[intermediate_index],
                SparkValidationReferenceLinearRow(
                    post_attention_normalized,
                    row,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),
                maximum_error))
        {
            return false;
        }
        gate_value = SparkValidationBf16ToFloat(gate[intermediate_index]);
        up_value = SparkValidationBf16ToFloat(up[intermediate_index]);
        silu_value = gate_value / (1.0f + expf(-gate_value));
        if (!SparkValidationCheckBf16ReferenceValueTracked(
                "layer3_shared_silu",
                intermediate_index,
                intermediate[intermediate_index],
                silu_value * up_value,
                maximum_error) ||
            !SparkValidationCopyDeviceBf16Row(
                row,
                buffers->dense_down_weight_bf16,
                hidden_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,
                "copy shared expert down row") ||
            !SparkValidationCheckBf16ReferenceValueTracked(
                "layer3_shared_down_residual",
                hidden_index,
                layer_output[hidden_index],
                SparkValidationBf16ToFloat(post_attention_hidden[hidden_index]) +
                    SparkValidationReferenceLinearRow(
                        intermediate,
                        row,
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION),
                maximum_error))
        {
            return false;
        }
    }
    fprintf(stderr, "layer3_shared_expert_reference_ready=1 max_error=%.8f\n", (double)*maximum_error);
    return true;
}

static bool SparkValidationCheckSampledLayer0References(
    SparkValidationDeviceBuffers *buffers,
    uint32_t use_dense_mlp)
{
    if (!SparkValidationCheckSampledAttentionReferences(buffers))
        return false;
    if (use_dense_mlp != 0u &&
        !SparkValidationCheckSampledOutputAndDenseReferences(buffers))
        return false;
    return true;
}

static bool SparkValidationCheckFullLayer0References(
    SparkValidationDeviceBuffers *buffers,
    uint32_t use_dense_mlp,
    float *maximum_error)
{
    if (use_dense_mlp == 0u)
    {
        fprintf(stderr, "full layer0 reference requires dense BF16 layer0 MLP fixture\n");
        return false;
    }
    return SparkValidationCheckFullOutputAndDenseReferences(
        buffers,
        maximum_error);
}

static bool SparkValidationCheckLayer0References(
    SparkValidationDeviceBuffers *buffers,
    uint32_t use_dense_mlp,
    uint32_t check_sampled_reference,
    uint32_t check_full_reference,
    float *maximum_error)
{
    if ((check_sampled_reference != 0u || check_full_reference != 0u) &&
        !SparkValidationCheckSampledLayer0References(buffers, use_dense_mlp))
        return false;
    if (check_full_reference != 0u &&
        !SparkValidationCheckFullLayer0References(
            buffers,
            use_dense_mlp,
            maximum_error))
        return false;
    return true;
}

static bool SparkValidationCheckRealRestrictedLogits(
    SparkValidationDeviceBuffers *buffers,
    SparkValidationRealLmHeadFixture *fixture,
    uint32_t selected_token_id)
{
    uint16_t normalized_hidden[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    float restricted_logits[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    uint32_t token_index;
    uint32_t expected_token_id;
    float expected_score;
    float maximum_error;

    if (!SparkValidationCudaSucceeded(
            cudaMemcpy(
                normalized_hidden,
                buffers->normalized_hidden_bf16,
                sizeof(normalized_hidden),
                cudaMemcpyDeviceToHost),
            "copy normalized_hidden real lm_head") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                restricted_logits,
                buffers->restricted_logits,
                sizeof(restricted_logits),
                cudaMemcpyDeviceToHost),
            "copy restricted_logits real lm_head"))
    {
        return false;
    }
    expected_token_id = SPARK_VALIDATION_RESTRICTED_LM_HEAD_FIRST_TOKEN;
    expected_score = -FLT_MAX;
    maximum_error = 0.0f;
    for (token_index = 0u;
         token_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT;
         ++token_index)
    {
        const uint16_t *row_bf16;
        uint32_t candidate_token_id;
        float expected_logit;
        float observed_logit;
        float logit_error;

        row_bf16 = &fixture->restricted_rows_bf16[
            (uint64_t)token_index *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
        expected_logit = SparkValidationReferenceRestrictedLogit(
            normalized_hidden,
            row_bf16);
        observed_logit = restricted_logits[token_index];
        logit_error = fabsf(observed_logit - expected_logit);
        if (logit_error > maximum_error)
        {
            maximum_error = logit_error;
        }
        if (logit_error > SPARK_VALIDATION_LOGIT_TOLERANCE)
        {
            fprintf(
                stderr,
                "real lm_head logit mismatch token=%u observed=%.8f expected=%.8f error=%.8f\n",
                SPARK_VALIDATION_RESTRICTED_LM_HEAD_FIRST_TOKEN + token_index,
                observed_logit,
                expected_logit,
                logit_error);
            return false;
        }
        candidate_token_id =
            SPARK_VALIDATION_RESTRICTED_LM_HEAD_FIRST_TOKEN + token_index;
        if (expected_logit > expected_score ||
            (expected_logit == expected_score &&
             candidate_token_id < expected_token_id))
        {
            expected_score = expected_logit;
            expected_token_id = candidate_token_id;
        }
    }
    fixture->expected_selected_token = expected_token_id;
    fixture->expected_selected_score = expected_score;
    fixture->maximum_logit_error = maximum_error;
    if (selected_token_id != expected_token_id)
    {
        fprintf(
            stderr,
            "real lm_head argmax mismatch observed=%u expected=%u score=%.8f max_logit_error=%.8f\n",
            selected_token_id,
            expected_token_id,
            expected_score,
            maximum_error);
        return false;
    }
    return true;
}

static bool SparkValidationCheckLayer3RouterTopK(
    SparkValidationDeviceBuffers *buffers)
{
    uint16_t *hidden_bf16;
    uint16_t *router_weight_bf16;
    float router_bias[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    float choice_scores[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    float route_weights[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    uint32_t observed_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float observed_weights[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    uint32_t expected_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float expected_weights[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    uint64_t router_weight_count;
    uint32_t expert_index;
    uint32_t topk_index;
    bool succeeded;

    hidden_bf16 = 0;
    router_weight_bf16 = 0;
    router_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    hidden_bf16 = (uint16_t *)malloc(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u);
    router_weight_bf16 = (uint16_t *)malloc(router_weight_count * 2u);
    if (hidden_bf16 == 0 || router_weight_bf16 == 0)
    {
        fprintf(stderr, "could not allocate router reference buffers\n");
        free(hidden_bf16);
        free(router_weight_bf16);
        return false;
    }
    succeeded =
        SparkValidationCudaSucceeded(
            cudaMemcpy(
                hidden_bf16,
                buffers->post_attention_normalized_hidden_bf16,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u,
                cudaMemcpyDeviceToHost),
            "copy router hidden") &&
        SparkValidationCudaSucceeded(
            cudaMemcpy(
                router_weight_bf16,
                buffers->moe_router_weight_bf16,
                (size_t)(router_weight_count * 2u),
                cudaMemcpyDeviceToHost),
            "copy router weight") &&
        SparkValidationCudaSucceeded(
            cudaMemcpy(
                router_bias,
                buffers->moe_router_score_bias_f32,
                sizeof(router_bias),
                cudaMemcpyDeviceToHost),
            "copy router bias") &&
        SparkValidationCudaSucceeded(
            cudaMemcpy(
                observed_ids,
                buffers->moe_topk_expert_ids,
                sizeof(observed_ids),
                cudaMemcpyDeviceToHost),
            "copy router topk ids") &&
        SparkValidationCudaSucceeded(
            cudaMemcpy(
                observed_weights,
                buffers->moe_topk_weights,
                sizeof(observed_weights),
                cudaMemcpyDeviceToHost),
            "copy router topk weights");
    if (!succeeded)
    {
        free(hidden_bf16);
        free(router_weight_bf16);
        return false;
    }
    for (expert_index = 0u;
         expert_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
         ++expert_index)
    {
        float router_logit;
        uint32_t hidden_index;

        router_logit = 0.0f;
        for (hidden_index = 0u;
             hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
             ++hidden_index)
        {
            router_logit +=
                SparkValidationBf16ToFloat(hidden_bf16[hidden_index]) *
                SparkValidationBf16ToFloat(
                    router_weight_bf16[
                        ((uint64_t)expert_index *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                        (uint64_t)hidden_index]);
        }
        route_weights[expert_index] =
            1.0f / (1.0f + expf(-router_logit));
        choice_scores[expert_index] =
            route_weights[expert_index] + router_bias[expert_index];
    }
    for (topk_index = 0u;
         topk_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
         ++topk_index)
    {
        float best_score;
        uint32_t best_expert;

        best_score = -FLT_MAX;
        best_expert = UINT32_MAX;
        for (expert_index = 0u;
             expert_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
             ++expert_index)
        {
            if (choice_scores[expert_index] > best_score ||
                (choice_scores[expert_index] == best_score &&
                 expert_index < best_expert))
            {
                best_score = choice_scores[expert_index];
                best_expert = expert_index;
            }
        }
        expected_ids[topk_index] = best_expert;
        expected_weights[topk_index] = route_weights[best_expert];
        choice_scores[best_expert] = -FLT_MAX;
    }
    {
        float weight_sum;

        weight_sum = 0.0f;
        for (topk_index = 0u;
             topk_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
             ++topk_index)
            weight_sum += expected_weights[topk_index];
        for (topk_index = 0u;
             topk_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
             ++topk_index)
        {
            if (weight_sum > 0.0f)
                expected_weights[topk_index] /= weight_sum;
            expected_weights[topk_index] *= 2.5f;
        }
    }
    for (topk_index = 0u;
         topk_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
         ++topk_index)
    {
        if (observed_ids[topk_index] != expected_ids[topk_index] ||
            fabsf(observed_weights[topk_index] - expected_weights[topk_index]) >
                0.0001f)
        {
            fprintf(
                stderr,
                "layer3 router topk mismatch slot=%u observed=(%u,%.8f) expected=(%u,%.8f)\n",
                topk_index,
                observed_ids[topk_index],
                observed_weights[topk_index],
                expected_ids[topk_index],
                expected_weights[topk_index]);
            free(hidden_bf16);
            free(router_weight_bf16);
            return false;
        }
    }
    fprintf(stderr, "layer3_router_topk_reference_ready=1 first_expert=%u first_weight=%.8f ids=%u,%u,%u,%u,%u,%u,%u,%u weights=%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", observed_ids[0], observed_weights[0], observed_ids[0], observed_ids[1], observed_ids[2], observed_ids[3], observed_ids[4], observed_ids[5], observed_ids[6], observed_ids[7], observed_weights[0], observed_weights[1], observed_weights[2], observed_weights[3], observed_weights[4], observed_weights[5], observed_weights[6], observed_weights[7]);
    free(hidden_bf16);
    free(router_weight_bf16);
    return true;
}

static bool SparkValidationCheckLayer3RoutedExpertNvfp4(
    SparkValidationDeviceBuffers *buffers,
    const SparkValidationLayer3RoutedExpertNvfp4Fixture *fixture)
{
    uint16_t gate_output[
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION];
    uint16_t up_output[
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION];
    uint16_t intermediate_output[
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION];
    uint16_t route_output[
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t layer_output[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint8_t intermediate_fp4_scratch[
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u) +
        SPARK_VALIDATION_MOE_ROUTE_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE)];
    uint32_t observed_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    uint32_t index;
    uint32_t nonzero_intermediate_payload_count;
    uint32_t nonzero_intermediate_scale_count;
    uint32_t nonzero_gate_count;
    uint32_t nonzero_up_count;
    uint32_t nonzero_intermediate_count;
    uint32_t nonzero_route_count;
    float up_checksum;
    float intermediate_checksum;
    float route_checksum;
    float layer_checksum;
    float up_maximum;
    float intermediate_maximum;
    float route_maximum;
    uint32_t route_count;
    uint64_t intermediate_value_count;
    uint64_t route_hidden_value_count;
    uint64_t intermediate_payload_bytes;
    uint64_t intermediate_scratch_bytes;

    route_count = fixture->bound_expert_count > 1u
        ? fixture->bound_expert_count
        : 1u;
    intermediate_value_count =
        (uint64_t)route_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
    route_hidden_value_count =
        (uint64_t)route_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    intermediate_payload_bytes =
        (uint64_t)route_count *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u);
    intermediate_scratch_bytes =
        intermediate_payload_bytes +
        ((uint64_t)route_count *
         (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
          SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE));
    if (!SparkValidationCudaSucceeded(
            cudaMemcpy(
                observed_ids,
                buffers->moe_topk_expert_ids,
                sizeof(observed_ids),
                cudaMemcpyDeviceToHost),
            "copy routed nvfp4 topk ids") ||
        !SparkValidationCopyDeviceBf16Vector(
            gate_output,
            buffers->moe_gate_bf16,
            (uint32_t)intermediate_value_count,
            "copy routed nvfp4 gate output") ||
        !SparkValidationCopyDeviceBf16Vector(
            up_output,
            buffers->moe_up_bf16,
            (uint32_t)intermediate_value_count,
            "copy routed nvfp4 up output") ||
        !SparkValidationCopyDeviceBf16Vector(
            intermediate_output,
            buffers->moe_intermediate_bf16,
            (uint32_t)intermediate_value_count,
            "copy routed nvfp4 intermediate output") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                intermediate_fp4_scratch,
                buffers->moe_gate_bf16,
                (size_t)intermediate_scratch_bytes,
                cudaMemcpyDeviceToHost),
            "copy routed nvfp4 intermediate fp4 scratch") ||
        !SparkValidationCopyDeviceBf16Vector(
            route_output,
            buffers->moe_route_output_bf16,
            (uint32_t)route_hidden_value_count,
            "copy routed nvfp4 route output") ||
        !SparkValidationCopyDeviceBf16Vector(
            layer_output,
            buffers->layer_output_hidden_bf16,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            "copy routed nvfp4 layer output"))
    {
        return false;
    }
    if (observed_ids[0] != fixture->selected_expert_id)
    {
        fprintf(
            stderr,
            "layer3 routed nvfp4 expected expert %u but router selected %u\n",
            fixture->selected_expert_id,
            observed_ids[0]);
        return false;
    }
    if (route_count > 1u)
    {
        uint32_t route_order_changed;

        route_order_changed = 0u;
        for (index = 0u; index < route_count; ++index)
        {
            uint32_t observed_index;
            uint32_t found_expert;

            found_expert = 0u;
            if (observed_ids[index] != fixture->bound_expert_ids[index])
                route_order_changed = 1u;
            for (observed_index = 0u; observed_index < route_count; ++observed_index)
            {
                if (observed_ids[observed_index] == fixture->bound_expert_ids[index])
                    found_expert = 1u;
            }
            if (found_expert == 0u)
            {
                fprintf(
                    stderr,
                    "layer3 routed nvfp4 missing expected route expert %u\n",
                    fixture->bound_expert_ids[index]);
                return false;
            }
        }
        if (route_order_changed != 0u)
        {
            fprintf(stderr, "layer3 routed nvfp4 route order changed but expert set matched\n");
        }
    }
    nonzero_gate_count = 0u;
    nonzero_up_count = 0u;
    nonzero_intermediate_count = 0u;
    nonzero_intermediate_payload_count = 0u;
    nonzero_intermediate_scale_count = 0u;
    up_checksum = 0.0f;
    intermediate_checksum = 0.0f;
    up_maximum = 0.0f;
    intermediate_maximum = 0.0f;
    for (index = 0u;
         index < intermediate_value_count;
         ++index)
    {
        float up_value;
        float intermediate_value;

        up_value = SparkValidationBf16ToFloat(up_output[index]);
        intermediate_value =
            SparkValidationBf16ToFloat(intermediate_output[index]);
        if (!isfinite(up_value) ||
            !isfinite(intermediate_value))
        {
            fprintf(stderr, "layer3 routed nvfp4 produced nonfinite expert activation at intermediate=%u\n", index);
            return false;
        }
        if (gate_output[index] != 0u)
        {
            nonzero_gate_count += 1u;
        }
        if (up_output[index] != 0u)
        {
            nonzero_up_count += 1u;
        }
        if (intermediate_output[index] != 0u)
        {
            nonzero_intermediate_count += 1u;
        }
        up_maximum = fmaxf(up_maximum, fabsf(up_value));
        intermediate_maximum =
            fmaxf(intermediate_maximum, fabsf(intermediate_value));
        if (index < 64u)
        {
            up_checksum += up_value * (float)(index + 1u);
            intermediate_checksum += intermediate_value * (float)(index + 1u);
        }
    }
    for (index = 0u; index < intermediate_scratch_bytes; ++index)
    {
        if (intermediate_fp4_scratch[index] != 0u)
        {
            if (index < intermediate_payload_bytes)
            {
                nonzero_intermediate_payload_count += 1u;
            }
            else
            {
                nonzero_intermediate_scale_count += 1u;
            }
        }
    }
    nonzero_route_count = 0u;
    route_checksum = 0.0f;
    layer_checksum = 0.0f;
    route_maximum = 0.0f;
    for (index = 0u;
         index < route_hidden_value_count;
         ++index)
    {
        float route_value;

        route_value = SparkValidationBf16ToFloat(route_output[index]);
        if (!isfinite(route_value))
        {
            fprintf(stderr, "layer3 routed nvfp4 produced nonfinite route output at hidden=%u\n", index);
            return false;
        }
        if (route_output[index] != 0u)
        {
            nonzero_route_count += 1u;
        }
        route_maximum = fmaxf(route_maximum, fabsf(route_value));
        if (index < 64u)
        {
            route_checksum += route_value * (float)(index + 1u);
        }
    }
    for (index = 0u;
         index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         ++index)
    {
        float layer_value;

        layer_value = SparkValidationBf16ToFloat(layer_output[index]);
        if (!isfinite(layer_value))
        {
            fprintf(stderr, "layer3 routed nvfp4 produced nonfinite layer output at hidden=%u\n", index);
            return false;
        }
        if (index < 64u)
        {
            layer_checksum += layer_value * (float)(index + 1u);
        }
    }
    if (nonzero_route_count == 0u ||
        fabsf(route_checksum) < 0.000001f ||
        fabsf(layer_checksum) < 0.000001f)
    {
        fprintf(stderr, "layer3 routed nvfp4 output was degenerate route_count=%u gate_scratch_nonzero=%u up_nonzero=%u intermediate_nonzero=%u intermediate_fp4_payload_nonzero=%u intermediate_fp4_scale_nonzero=%u route_nonzero=%u up_max=%.8g intermediate_max=%.8g route_max=%.8g up_checksum64=%.8f intermediate_checksum64=%.8f route_checksum64=%.8f layer_checksum64=%.8f\n", route_count, nonzero_gate_count, nonzero_up_count, nonzero_intermediate_count, nonzero_intermediate_payload_count, nonzero_intermediate_scale_count, nonzero_route_count, (double)up_maximum, (double)intermediate_maximum, (double)route_maximum, up_checksum, intermediate_checksum, route_checksum, layer_checksum);
        return false;
    }
    fprintf(stderr, "layer3_routed_expert_nvfp4_reference_ready=1 expert=%u route_count=%u gate_scratch_nonzero=%u up_nonzero=%u intermediate_nonzero=%u intermediate_fp4_payload_nonzero=%u intermediate_fp4_scale_nonzero=%u route_nonzero=%u up_max=%.8g intermediate_max=%.8g route_max=%.8g up_checksum64=%.8f intermediate_checksum64=%.8f route_checksum64=%.8f layer_checksum64=%.8f\n", fixture->selected_expert_id, route_count, nonzero_gate_count, nonzero_up_count, nonzero_intermediate_count, nonzero_intermediate_payload_count, nonzero_intermediate_scale_count, nonzero_route_count, (double)up_maximum, (double)intermediate_maximum, (double)route_maximum, (double)up_checksum, (double)intermediate_checksum, (double)route_checksum, (double)layer_checksum);
    return true;
}

static bool SparkValidationReadReferenceCacheFromDevice(
    SparkValidationDeviceBuffers *buffers,
    float cache_key_nope_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION],
    float cache_rope_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION],
    float cache_value_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT])
{
    uint16_t key_value[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION];
    uint16_t value_value[
        SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t rope_value[SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION];
    uint32_t token_index;
    uint32_t head_index;
    uint32_t dimension_index;

    for (token_index = 0u; token_index < SPARK_VALIDATION_CONTEXT_LENGTH; ++token_index)
    {
        uint32_t cache_slot_index;

        cache_slot_index = token_index == SPARK_VALIDATION_CONTEXT_LENGTH - 1u
            ? SPARK_VALIDATION_CURRENT_CACHE_SLOT
            : SPARK_VALIDATION_REMAP_CACHE_SLOT0 + token_index;
        if (!SparkValidationCudaSucceeded(
                cudaMemcpy(
                    rope_value,
                    &buffers->mla_cache_bf16[
                        ((uint64_t)cache_slot_index *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION],
                    sizeof(rope_value),
                    cudaMemcpyDeviceToHost),
                "copy reference rope cache"))
            return false;
        for (dimension_index = 0u; dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION; ++dimension_index)
            cache_rope_values[token_index][dimension_index] =
                SparkValidationBf16ToFloat(rope_value[dimension_index]);
        for (head_index = 0u; head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT; ++head_index)
        {
            uint32_t key_nonzero_count;
            uint32_t value_nonzero_count;

            key_nonzero_count = 0u;
            value_nonzero_count = 0u;
            if (!SparkValidationCudaSucceeded(
                    cudaMemcpy(
                        key_value,
                        &buffers->key_nope_cache_bf16[
                            (((uint64_t)cache_slot_index *
                              (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                              (uint64_t)head_index) *
                             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION)],
                        sizeof(key_value),
                        cudaMemcpyDeviceToHost),
                    "copy reference key cache") ||
                !SparkValidationCudaSucceeded(
                    cudaMemcpy(
                        value_value,
                        &buffers->value_cache_bf16[
                            (((uint64_t)cache_slot_index *
                              (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                              (uint64_t)head_index) *
                             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)],
                        sizeof(value_value),
                        cudaMemcpyDeviceToHost),
                    "copy reference value cache"))
                return false;
            for (dimension_index = 0u; dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION; ++dimension_index)
            {
                if (key_value[dimension_index] != 0u)
                    key_nonzero_count += 1u;
                cache_key_nope_values[head_index][token_index][dimension_index] =
                    SparkValidationBf16ToFloat(key_value[dimension_index]);
            }
            for (dimension_index = 0u; dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT; ++dimension_index)
            {
                if (value_value[dimension_index] != 0u)
                    value_nonzero_count += 1u;
                cache_value_values[head_index][token_index][dimension_index] =
                    SparkValidationBf16ToFloat(value_value[dimension_index]);
            }
            if (key_nonzero_count == 0u || value_nonzero_count == 0u)
            {
                fprintf(stderr, "prefill reference cache stayed zero token=%u head=%u key_nonzero=%u value_nonzero=%u\n", token_index, head_index, key_nonzero_count, value_nonzero_count);
                return false;
            }
        }
    }
    return true;
}

static bool SparkValidationCheckOutputs(
    SparkValidationDeviceBuffers *buffers,
    SparkValidationRealLmHeadFixture *real_lm_head,
    uint32_t require_moe_route_output,
    uint32_t use_prefill_kv,
    uint32_t require_phase_clock)
{
    uint16_t current_kv_value[
        SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t cached_kv_value[
        SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t cached_key_nope_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION];
    uint16_t cached_value_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t key_rope_value[SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION];
    uint16_t cached_rope_value[SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION];
    uint16_t query_latent_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION];
    uint16_t query_rope_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION];
    uint16_t attention_output_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t moe_route_output_value[SPARK_VALIDATION_MOE_CHECKED_INTERMEDIATE];
    uint16_t layer_output_value[SPARK_VALIDATION_MOE_CHECKED_INTERMEDIATE];
    uint32_t selected_token_id;
    uint32_t mtp_draft_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t mtp_accept_mask[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t mtp_committed_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t counters[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT];
    uint64_t phase_clocks[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT];
    float query_latent_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION];
    float query_rope_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION];
    float cache_key_nope_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION];
    float cache_value_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    float cache_rope_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION];
    uint32_t head_index;
    uint32_t dimension_index;
    uint32_t token_index;
    uint32_t expected_mtp_event_count;

    memset(current_kv_value, 0, sizeof(current_kv_value));
    memset(cached_kv_value, 0, sizeof(cached_kv_value));
    memset(cached_key_nope_value, 0, sizeof(cached_key_nope_value));
    memset(cached_value_value, 0, sizeof(cached_value_value));
    memset(key_rope_value, 0, sizeof(key_rope_value));
    memset(cached_rope_value, 0, sizeof(cached_rope_value));
    memset(query_latent_value, 0, sizeof(query_latent_value));
    memset(query_rope_value, 0, sizeof(query_rope_value));
    memset(attention_output_value, 0, sizeof(attention_output_value));
    memset(moe_route_output_value, 0, sizeof(moe_route_output_value));
    memset(layer_output_value, 0, sizeof(layer_output_value));
    selected_token_id = 0u;
    memset(mtp_draft_token_ids, 0, sizeof(mtp_draft_token_ids));
    memset(mtp_accept_mask, 0, sizeof(mtp_accept_mask));
    memset(mtp_committed_token_ids, 0, sizeof(mtp_committed_token_ids));
    memset(counters, 0, sizeof(counters));
    memset(phase_clocks, 0, sizeof(phase_clocks));
    expected_mtp_event_count =
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
    if (!SparkValidationCudaSucceeded(
            cudaMemcpy(
                current_kv_value,
                buffers->current_kv_latent_bf16,
                sizeof(current_kv_value),
                cudaMemcpyDeviceToHost),
            "copy current_kv") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                cached_kv_value,
                &buffers->mla_cache_bf16[
                    (SPARK_VALIDATION_CURRENT_CACHE_SLOT *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                    0u],
                sizeof(cached_kv_value),
                cudaMemcpyDeviceToHost),
            "copy cached current kv") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                key_rope_value,
                buffers->key_rope_input_bf16,
                sizeof(key_rope_value),
                cudaMemcpyDeviceToHost),
            "copy key_rope_input") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                cached_rope_value,
                &buffers->mla_cache_bf16[
                    (SPARK_VALIDATION_CURRENT_CACHE_SLOT *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION],
                sizeof(cached_rope_value),
                cudaMemcpyDeviceToHost),
            "copy cached current rope") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                moe_route_output_value,
                &buffers->moe_route_output_bf16[3u],
                sizeof(moe_route_output_value),
                cudaMemcpyDeviceToHost),
            "copy moe route output") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                layer_output_value,
                &buffers->layer_output_hidden_bf16[3u],
                sizeof(layer_output_value),
                cudaMemcpyDeviceToHost),
            "copy layer output") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                &selected_token_id,
                buffers->restricted_selected_token_ids,
                sizeof(selected_token_id),
                cudaMemcpyDeviceToHost),
            "copy selected_token") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_draft_token_ids,
                buffers->mtp_draft_token_ids,
                sizeof(mtp_draft_token_ids),
                cudaMemcpyDeviceToHost),
            "copy mtp_draft_token_ids") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_accept_mask,
                buffers->mtp_accept_mask,
                sizeof(mtp_accept_mask),
                cudaMemcpyDeviceToHost),
            "copy mtp_accept_mask") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_committed_token_ids,
                buffers->mtp_committed_token_ids,
                sizeof(mtp_committed_token_ids),
                cudaMemcpyDeviceToHost),
            "copy mtp_committed_token_ids") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                counters,
                buffers->mtp_event_counters,
                sizeof(counters),
                cudaMemcpyDeviceToHost),
            "copy mtp_event_counters") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                phase_clocks,
                buffers->phase_clock_cycles,
                sizeof(phase_clocks),
                cudaMemcpyDeviceToHost),
            "copy phase_clock_cycles"))
    {
        return false;
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        if (!SparkValidationCudaSucceeded(
                cudaMemcpy(
                    query_latent_value[head_index],
                    &buffers->query_latent_bf16[
                        (uint64_t)head_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION],
                    sizeof(query_latent_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy query_latent checked head") ||
            !SparkValidationCudaSucceeded(
                cudaMemcpy(
                    query_rope_value[head_index],
                    &buffers->rotated_query_rope_bf16[
                        (uint64_t)head_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION],
                    sizeof(query_rope_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy rotated_query_rope checked head") ||
            !SparkValidationCudaSucceeded(
                cudaMemcpy(
                    attention_output_value[head_index],
                    &buffers->attention_output_latent_bf16[
                        (uint64_t)head_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION],
                    sizeof(attention_output_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy attention_output checked head") ||
            !SparkValidationCudaSucceeded(
                cudaMemcpy(
                    cached_key_nope_value[head_index],
                    &buffers->key_nope_cache_bf16[
                        (((uint64_t)SPARK_VALIDATION_CURRENT_CACHE_SLOT *
                          (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                          (uint64_t)head_index) *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION)],
                    sizeof(cached_key_nope_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy current key_nope cache checked head") ||
            !SparkValidationCudaSucceeded(
                cudaMemcpy(
                    cached_value_value[head_index],
                    &buffers->value_cache_bf16[
                        (((uint64_t)SPARK_VALIDATION_CURRENT_CACHE_SLOT *
                          (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                          (uint64_t)head_index) *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)],
                    sizeof(cached_value_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy current value cache checked head"))
        {
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (current_kv_value[dimension_index] == 0u ||
            cached_kv_value[dimension_index] != current_kv_value[dimension_index])
        {
            fprintf(
                stderr,
                "current KV value was not written to expected cache slot dim=%u\n",
                dimension_index);
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         dimension_index += 2u)
    {
        float first_input;
        float second_input;
        float cosine;
        float sine;
        float expected_first;
        float expected_second;
        float observed_first;
        float observed_second;

        first_input = SparkValidationBf16ToFloat(
            key_rope_value[dimension_index]);
        second_input = SparkValidationBf16ToFloat(
            key_rope_value[dimension_index + 1u]);
        cosine = cosf(0.125f * (float)((dimension_index / 2u) + 1u));
        sine = sinf(0.125f * (float)((dimension_index / 2u) + 1u));
        expected_first = (first_input * cosine) - (second_input * sine);
        expected_second = (first_input * sine) + (second_input * cosine);
        observed_first = SparkValidationBf16ToFloat(
            cached_rope_value[dimension_index]);
        observed_second = SparkValidationBf16ToFloat(
            cached_rope_value[dimension_index + 1u]);
        if (first_input == 0.0f ||
            second_input == 0.0f ||
            fabsf(observed_first - expected_first) >
                SPARK_VALIDATION_ATTENTION_TOLERANCE ||
            fabsf(observed_second - expected_second) >
                SPARK_VALIDATION_ATTENTION_TOLERANCE)
        {
            fprintf(
                stderr,
                "current key RoPE rotation mismatch pair=%u observed=(%.6f,%.6f) expected=(%.6f,%.6f)\n",
                dimension_index / 2u,
                observed_first,
                observed_second,
                expected_first,
                expected_second);
            return false;
        }
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        uint32_t query_latent_nonzero_count;
        uint32_t query_rope_nonzero_count;
        uint32_t cached_key_nonzero_count;
        uint32_t cached_value_nonzero_count;

        query_latent_nonzero_count = 0u;
        query_rope_nonzero_count = 0u;
        cached_key_nonzero_count = 0u;
        cached_value_nonzero_count = 0u;
        for (dimension_index = 0u;
             dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
             ++dimension_index)
        {
            if (query_latent_value[head_index][dimension_index] != 0u)
                query_latent_nonzero_count += 1u;
            query_latent_values[head_index][dimension_index] =
                SparkValidationBf16ToFloat(
                    query_latent_value[head_index][dimension_index]);
            if (cached_key_nope_value[head_index][dimension_index] != 0u)
                cached_key_nonzero_count += 1u;
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
             ++dimension_index)
        {
            if (query_rope_value[head_index][dimension_index] != 0u)
                query_rope_nonzero_count += 1u;
            query_rope_values[head_index][dimension_index] =
                SparkValidationBf16ToFloat(
                    query_rope_value[head_index][dimension_index]);
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            if (cached_value_value[head_index][dimension_index] != 0u)
                cached_value_nonzero_count += 1u;
        }
        if (query_latent_nonzero_count == 0u ||
            query_rope_nonzero_count == 0u ||
            cached_key_nonzero_count == 0u ||
            cached_value_nonzero_count == 0u)
        {
            fprintf(stderr, "attention fixture stayed zero head=%u query_latent=%u query_rope=%u key=%u value=%u\n", head_index, query_latent_nonzero_count, query_rope_nonzero_count, cached_key_nonzero_count, cached_value_nonzero_count);
            return false;
        }
    }
    if (use_prefill_kv != 0u)
    {
        if (!SparkValidationReadReferenceCacheFromDevice(
                buffers,
                cache_key_nope_values,
                cache_rope_values,
                cache_value_values))
            return false;
    }
    else
    {
        for (token_index = 0u;
             token_index < SPARK_VALIDATION_CONTEXT_LENGTH - 1u;
             ++token_index)
        {
            for (head_index = 0u;
                 head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
                 ++head_index)
            {
                for (dimension_index = 0u;
                     dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
                     ++dimension_index)
                {
                    cache_key_nope_values[head_index][token_index][dimension_index] =
                        SparkValidationBf16ToFloat(
                            SparkValidationFloatToBf16(
                                SparkValidationSeedKeyNopeValue(
                                    token_index,
                                    head_index,
                                    dimension_index)));
                }
                for (dimension_index = 0u;
                     dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
                     ++dimension_index)
                {
                    cache_value_values[head_index][token_index][dimension_index] =
                        SparkValidationBf16ToFloat(
                            SparkValidationFloatToBf16(
                                SparkValidationSeedValueCacheValue(
                                    token_index,
                                    head_index,
                                    dimension_index)));
                }
            }
            for (dimension_index = 0u;
                 dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
                 ++dimension_index)
            {
                cache_rope_values[token_index][dimension_index] =
                    SparkValidationBf16ToFloat(
                        SparkValidationFloatToBf16(
                            SparkValidationSeedRopeValue(
                                token_index,
                                dimension_index)));
            }
        }
        for (head_index = 0u;
             head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
             ++head_index)
        {
            for (dimension_index = 0u;
                 dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
                 ++dimension_index)
            {
                cache_key_nope_values[
                    head_index]
                    [SPARK_VALIDATION_CONTEXT_LENGTH - 1u]
                    [dimension_index] =
                    SparkValidationBf16ToFloat(
                        cached_key_nope_value[head_index][dimension_index]);
            }
            for (dimension_index = 0u;
                 dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
                 ++dimension_index)
            {
                cache_value_values[
                    head_index]
                    [SPARK_VALIDATION_CONTEXT_LENGTH - 1u]
                    [dimension_index] =
                    SparkValidationBf16ToFloat(
                        cached_value_value[head_index][dimension_index]);
            }
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
             ++dimension_index)
        {
            cache_rope_values[
                SPARK_VALIDATION_CONTEXT_LENGTH - 1u][dimension_index] =
                SparkValidationBf16ToFloat(cached_rope_value[dimension_index]);
        }
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            float expected_attention;
            float observed_attention;

            expected_attention = SparkValidationReferenceAttentionValue(
                query_latent_values[head_index],
                query_rope_values[head_index],
                cache_key_nope_values[head_index],
                cache_rope_values,
                cache_value_values[head_index],
                dimension_index);
            observed_attention = SparkValidationBf16ToFloat(
                attention_output_value[head_index][dimension_index]);
            if (fabsf(observed_attention - expected_attention) >
                SPARK_VALIDATION_ATTENTION_TOLERANCE)
            {
                fprintf(
                    stderr,
                    "attention reference mismatch head=%u dim=%u observed=%.6f expected=%.6f\n",
                    head_index,
                    dimension_index,
                    observed_attention,
                    expected_attention);
                return false;
            }
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_MOE_CHECKED_INTERMEDIATE;
         ++dimension_index)
    {
        if (require_moe_route_output != 0u &&
            moe_route_output_value[dimension_index] == 0u)
        {
            fprintf(stderr, "required routed MoE output stayed zero\n");
            return false;
        }
        if (layer_output_value[dimension_index] == 0u)
        {
            fprintf(stderr, "layer output stayed zero\n");
            return false;
        }
    }
    if (real_lm_head != 0 && real_lm_head->ready != 0u)
    {
        if (!SparkValidationCheckRealRestrictedLogits(
                buffers,
                real_lm_head,
                selected_token_id))
        {
            return false;
        }
    }
    else if (selected_token_id != SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN)
    {
        fprintf(
            stderr,
            "restricted argmax mismatch observed=%u expected=%u\n",
            selected_token_id,
            SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN);
        return false;
    }
    if (mtp_draft_token_ids[0] != SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN ||
        mtp_draft_token_ids[1] != SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN ||
        mtp_accept_mask[0] != 1u ||
        mtp_accept_mask[1] != 0u ||
        mtp_committed_token_ids[0] != SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN ||
        mtp_committed_token_ids[1] != SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN)
    {
        fprintf(stderr, "MTP draft/verify fixture did not produce expected tokens\n");
        return false;
    }
    if (require_phase_clock != 0u &&
        phase_clocks[
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY] == 0u)
    {
        fprintf(stderr, "phase clock completion marker was not written\n");
        return false;
    }
    if (counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED] == 0u)
    {
        fprintf(stderr, "MTP commit counter was not incremented\n");
        return false;
    }
    if ((counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED] +
         counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED]) !=
            expected_mtp_event_count ||
        (counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED] +
         counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED]) !=
            expected_mtp_event_count ||
        counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK] !=
            counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED] ||
        counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED] <
            SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT ||
        counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED] >
            expected_mtp_event_count)
    {
        fprintf(stderr, "MTP event counters do not match aggregate invariants accepted=%u rejected=%u committed=%u rollback=%u cancelled=%u expected_events=%u\n", counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED], counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED], counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED], counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK], counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED], expected_mtp_event_count);
        return false;
    }
    return true;
}

static bool SparkValidationRunDriverOnce(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    float *elapsed_microseconds)
{
    SparkValidationDriverCompletionState completion_state;
    SparkOrchestratorConfiguration orchestrator_configuration;
    SparkOrchestrator *orchestrator;
    SparkOrchestratorNodeHandle node_handle;
    SparkOrchestratorDriverHandle driver_handle;
    SparkOrchestratorRouteHandle route_handle;
    SparkModelDriverRuntimeSnapshot runtime_snapshot;
    SparkModelDriverFrame frame;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    SparkStatus status;
    char error_buffer[1024];
    float elapsed_milliseconds;

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&orchestrator_configuration, 0, sizeof(orchestrator_configuration));
    memset(&frame, 0, sizeof(frame));
    orchestrator_configuration.node_capacity = 1u;
    orchestrator_configuration.driver_capacity = 1u;
    orchestrator_configuration.route_capacity = 1u;
    orchestrator_configuration.route_endpoint_capacity = 1u;
    orchestrator_configuration.completion_function =
        SparkValidationDriverCompletion;
    orchestrator_configuration.completion_context = &completion_state;
    orchestrator = 0;
    start_event = 0;
    stop_event = 0;
    status = SparkCreateOrchestrator(
        &orchestrator_configuration,
        &orchestrator);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkCreateOrchestrator failed: %s\n", SparkStatusToString(status));
        return false;
    }
    status = SparkOrchestratorAddNode(
        orchestrator,
        "cuda-node-0",
        SPARK_GLM52_RESIDENT_DECODE_STAGE_TARGET,
        node_context,
        &node_handle);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorAddNode failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    status = SparkOrchestratorAttachDriver(
        orchestrator,
        node_handle,
        driver_path,
        &driver_handle,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorAttachDriver failed: %s: %s\n", SparkStatusToString(status), error_buffer);
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    status = SparkOrchestratorResolveRoute(
        orchestrator,
        "zai.glm-5.2.resident-decode-stage-firmware",
        "bf16-h6144-h64-d512-r64-k2048-b1024-rv256-mtp2-v1",
        "resident_decode",
        "decode",
        &route_handle);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorResolveRoute failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (!SparkValidationCudaSucceeded(cudaEventCreate(&start_event), "cudaEventCreate start") ||
        !SparkValidationCudaSucceeded(cudaEventCreate(&stop_event), "cudaEventCreate stop") ||
        !SparkValidationCudaSucceeded(cudaEventRecord(start_event, cuda_stream), "cudaEventRecord start"))
    {
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    frame.request_id = 9001u;
    frame.sequence_id = 70001u;
    frame.sequence_position = 17u;
    frame.active_slot_count = SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
    frame.new_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u;
    frame.residency.owner = 1u;
    if (!SparkValidationPreflightRequiredFastPath(node_context))
    {
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    status = SparkOrchestratorSubmit(orchestrator, route_handle, &frame);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorSubmit failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if ((frame.flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u ||
        frame.driver_dispatch_slot != 0u)
    {
        fprintf(stderr, "driver dispatch slot was not assigned by admission\n");
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (!SparkValidationCudaSucceeded(cudaEventRecord(stop_event, cuda_stream), "cudaEventRecord stop") ||
        !SparkValidationCudaSucceeded(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop"))
    {
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (completion_state.completion_count.load(std::memory_order_acquire) != 1u ||
        completion_state.completion.request_id != frame.request_id ||
        completion_state.completion.status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "orchestrator completion did not match submitted frame\n");
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    status = SparkOrchestratorGetDriverProgramSnapshot(
        orchestrator,
        driver_handle,
        "decode",
        &runtime_snapshot);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorGetDriverProgramSnapshot failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (runtime_snapshot.submitted_count != 1u ||
        runtime_snapshot.completed_count != 1u ||
        runtime_snapshot.active_submission_count != 0u ||
        runtime_snapshot.host_callback_completion_count != 1u ||
        runtime_snapshot.host_staging_bytes_per_submit != 0u ||
        runtime_snapshot.device_memcpy_bytes_per_submit != 0u)
    {
        fprintf(stderr, "orchestrator snapshot counters are not clean\n");
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (!SparkValidationCudaSucceeded(
            cudaEventElapsedTime(&elapsed_milliseconds, start_event, stop_event),
            "cudaEventElapsedTime"))
    {
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    *elapsed_microseconds = elapsed_milliseconds * 1000.0f;
    SparkDestroyOrchestrator(orchestrator);
    return true;
}

static bool SparkValidationRunSubmitOnce(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    float *elapsed_microseconds)
{
    if (driver_path != 0)
    {
        return SparkValidationRunDriverOnce(
            node_context,
            cuda_stream,
            driver_path,
            elapsed_microseconds);
    }
    return SparkValidationRunOnce(
        node_context,
        cuda_stream,
        elapsed_microseconds);
}

static bool SparkValidationLoadDenseLayerBf16Fixtures(
    SparkValidationDeviceBuffers *buffers,
    const char *model_directory,
    uint32_t layer_index)
{
    SparkValidationLayer0DenseBf16Fixture dense_fixture;
    SparkValidationLayer0AttentionBf16Fixture attention_fixture;

    return SparkValidationLoadLayer0AttentionBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &attention_fixture) &&
        SparkValidationLoadLayer0DenseBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &dense_fixture);
}

static bool SparkValidationCopyLayerOutputToInput(
    SparkValidationDeviceBuffers *buffers)
{
    return SparkValidationCopyDeviceToDevice(
        buffers->input_hidden_bf16,
        buffers->layer_output_hidden_bf16,
        (uint64_t)SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u,
        "copy chained layer output to input");
}

static bool SparkValidationRunChainedDenseLayer(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    const char *model_directory,
    SparkValidationRealLmHeadFixture *real_lm_head,
    uint32_t layer_index,
    uint32_t position,
    uint32_t slot_mapping,
    uint32_t context_length,
    uint32_t check_outputs,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count,
    float *maximum_reference_error)
{
    float elapsed_microseconds;
    float layer_reference_error;

    layer_reference_error = 0.0f;
    node_context->layer_progression_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP;
    node_context->dense_intermediate_dimension =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION;
    node_context->mlp_execution_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE;
    if (!SparkValidationLoadDenseLayerBf16Fixtures(
            buffers,
            model_directory,
            layer_index) ||
        !SparkValidationBindDenseLayerCache(
            buffers,
            node_context,
            layer_index) ||
        !SparkValidationSetDecodeScalars(
            buffers,
            position,
            slot_mapping,
            context_length) ||
        !SparkValidationRunSubmitOnce(
            node_context,
            cuda_stream,
            driver_path,
            &elapsed_microseconds) ||
        !SparkValidationCudaSucceeded(
            cudaStreamSynchronize(cuda_stream),
            "cudaStreamSynchronize chained layer"))
    {
        return false;
    }
    *total_microseconds += (double)elapsed_microseconds;
    if ((double)elapsed_microseconds > *maximum_observed_microseconds)
    {
        *maximum_observed_microseconds = (double)elapsed_microseconds;
    }
    *submission_count += 1u;
    if (check_outputs != 0u &&
        (!SparkValidationCheckOutputs(buffers, real_lm_head, 0u, 1u, 1u) ||
         !SparkValidationCheckLayer0References(
            buffers,
            1u,
            1u,
            1u,
            &layer_reference_error)))
    {
        return false;
    }
    if (layer_reference_error > *maximum_reference_error)
        *maximum_reference_error = layer_reference_error;
    if (layer_index + 1u < SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT &&
        !SparkValidationCopyLayerOutputToInput(buffers))
        return false;
    return true;
}

static bool SparkValidationRunChainedDenseLayers(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    const char *model_directory,
    uint32_t input_token_id,
    const SparkValidationPromptTokenWindow *prompt_tokens,
    SparkValidationRealLmHeadFixture *real_lm_head,
    uint32_t current_token_only,
    uint32_t check_current_outputs,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count,
    float *maximum_reference_error)
{
    uint32_t prefill_index;
    uint32_t layer_index;
    uint64_t copied_bytes;

    *total_microseconds = 0.0;
    *maximum_observed_microseconds = 0.0;
    *submission_count = 0u;
    *maximum_reference_error = 0.0f;
    copied_bytes = 0u;
    if (current_token_only == 0u)
    {
        for (prefill_index = 0u;
             prefill_index < SPARK_VALIDATION_CONTEXT_LENGTH - 1u;
             ++prefill_index)
        {
            uint32_t token_id;
            uint32_t position;
            uint32_t slot_mapping;
            uint32_t context_length;

            token_id = SparkValidationResolveContextToken(
                prompt_tokens,
                input_token_id,
                prefill_index);
            position = SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET + prefill_index;
            slot_mapping = SPARK_VALIDATION_REMAP_CACHE_SLOT0 + prefill_index;
            context_length = prefill_index + 1u;
            if (!SparkValidationCopyInputEmbeddingBf16Row(
                    model_directory,
                    token_id,
                    buffers->input_hidden_bf16,
                    &copied_bytes))
                return false;
            for (layer_index = 0u;
                 layer_index < SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT;
                 ++layer_index)
            {
                if (!SparkValidationRunChainedDenseLayer(
                        buffers,
                        node_context,
                        cuda_stream,
                        driver_path,
                        model_directory,
                        real_lm_head,
                        layer_index,
                        position,
                        slot_mapping,
                        context_length,
                        0u,
                        total_microseconds,
                        maximum_observed_microseconds,
                        submission_count,
                        maximum_reference_error))
                    return false;
            }
        }
    }
    if (!SparkValidationCopyInputEmbeddingBf16Row(
            model_directory,
            SparkValidationResolveContextToken(
                prompt_tokens,
                input_token_id,
                SPARK_VALIDATION_CONTEXT_LENGTH - 1u),
            buffers->input_hidden_bf16,
            &copied_bytes))
        return false;
    for (layer_index = 0u;
         layer_index < SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT;
         ++layer_index)
    {
        if (!SparkValidationRunChainedDenseLayer(
                buffers,
                node_context,
                cuda_stream,
                driver_path,
                model_directory,
                real_lm_head,
                layer_index,
                SPARK_VALIDATION_CURRENT_POSITION,
                SPARK_VALIDATION_CURRENT_CACHE_SLOT,
                SPARK_VALIDATION_CONTEXT_LENGTH,
                check_current_outputs,
                total_microseconds,
                maximum_observed_microseconds,
                submission_count,
                maximum_reference_error))
            return false;
    }
    fprintf(stderr, "dense_chain_embedding_bf16_bytes=%llu\n", (unsigned long long)copied_bytes);
    return true;
}

static bool SparkValidationReadLayer3TopKExpertIds(
    SparkValidationDeviceBuffers *buffers,
    uint32_t *expert_ids)
{
    if (buffers == 0 ||
        expert_ids == 0)
    {
        fprintf(stderr, "layer3 top-k expert id read was given null input\n");
        return false;
    }
    return SparkValidationCudaSucceeded(
        cudaMemcpy(
            expert_ids,
            buffers->moe_topk_expert_ids,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K * 4u,
            cudaMemcpyDeviceToHost),
        "copy dynamic layer3 topk expert ids");
}

static bool SparkValidationRunLayer3RouterTopKLayer(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    const char *model_directory,
    uint32_t layer_index,
    uint32_t position,
    uint32_t slot_mapping,
    uint32_t context_length,
    uint32_t check_outputs,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count)
{
    SparkValidationLayer0AttentionBf16Fixture attention_fixture;
    SparkValidationLayer3RouterBf16Fixture router_fixture;
    float elapsed_microseconds;

    if (!SparkValidationLoadLayer0AttentionBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &attention_fixture) ||
        !SparkValidationLoadRoutedLayerRouterBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &router_fixture) ||
        !SparkValidationBindRoutedLayerCache(
            buffers,
            node_context,
            layer_index) ||
        !SparkValidationSetDecodeScalars(
            buffers,
            position,
            slot_mapping,
            context_length))
    {
        return false;
    }
    SparkValidationEnableLayer3RouterTopK(buffers, node_context);
    if (!SparkValidationRunSubmitOnce(
            node_context,
            cuda_stream,
            driver_path,
            &elapsed_microseconds) ||
        !SparkValidationCudaSucceeded(
            cudaStreamSynchronize(cuda_stream),
            "cudaStreamSynchronize chained routed layer router"))
    {
        return false;
    }
    *total_microseconds += (double)elapsed_microseconds;
    if ((double)elapsed_microseconds > *maximum_observed_microseconds)
    {
        *maximum_observed_microseconds = (double)elapsed_microseconds;
    }
    *submission_count += 1u;
    if (check_outputs != 0u &&
        !SparkValidationCheckLayer3RouterTopK(buffers))
    {
        return false;
    }
    return true;
}

static bool SparkValidationRunLayer3RoutedTopKLayer(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    const char *model_directory,
    SparkValidationRealLmHeadFixture *real_lm_head,
    const SparkValidationLayer3RoutedExpertNvfp4Fixture *layer3_routed_expert,
    uint32_t layer_index,
    uint32_t position,
    uint32_t slot_mapping,
    uint32_t context_length,
    uint32_t check_outputs,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count)
{
    SparkValidationLayer0AttentionBf16Fixture attention_fixture;
    float elapsed_microseconds;

    if (layer3_routed_expert == 0 ||
        layer3_routed_expert->ready == 0u)
    {
        fprintf(stderr, "layer3 routed top-k fixture is not ready\n");
        return false;
    }
    if (!SparkValidationLoadLayer0AttentionBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &attention_fixture) ||
        !SparkValidationBindRoutedLayerCache(
            buffers,
            node_context,
            layer_index) ||
        !SparkValidationSetDecodeScalars(
            buffers,
            position,
            slot_mapping,
            context_length) ||
        !SparkValidationBindB12xMoePlanForLayer(
            buffers,
            node_context,
            layer_index))
    {
        return false;
    }
    SparkValidationEnableLayer3RoutedExpertNvfp4(
        layer3_routed_expert,
        buffers,
        node_context);
    if (!SparkValidationRunSubmitOnce(
            node_context,
            cuda_stream,
            driver_path,
            &elapsed_microseconds) ||
        !SparkValidationCudaSucceeded(
            cudaStreamSynchronize(cuda_stream),
            "cudaStreamSynchronize chained routed layer"))
    {
        return false;
    }
    *total_microseconds += (double)elapsed_microseconds;
    if ((double)elapsed_microseconds > *maximum_observed_microseconds)
    {
        *maximum_observed_microseconds = (double)elapsed_microseconds;
    }
    *submission_count += 1u;
    if (check_outputs != 0u &&
        (!SparkValidationCheckLayer3RouterTopK(buffers) ||
         !SparkValidationCheckLayer3RoutedExpertNvfp4(
            buffers,
            layer3_routed_expert) ||
         !SparkValidationCheckOutputs(buffers, real_lm_head, 1u, 1u, 1u)))
    {
        return false;
    }
    return true;
}

static bool SparkValidationRunRoutedLayerProductionB12x(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    const char *model_directory,
    SparkValidationRealLmHeadFixture *real_lm_head,
    SparkValidationLayer3RoutedExpertNvfp4Fixture *layer3_routed_expert,
    uint32_t layer_index,
    uint32_t position,
    uint32_t slot_mapping,
    uint32_t context_length,
    uint32_t check_outputs,
    uint32_t require_phase_clock,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count)
{
    SparkValidationLayer0AttentionBf16Fixture attention_fixture;
    SparkValidationLayer3RouterBf16Fixture router_fixture;
    uint32_t topk_expert_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    float elapsed_microseconds;

    if (!SparkValidationLoadLayer0AttentionBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &attention_fixture) ||
        !SparkValidationLoadRoutedLayerRouterBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &router_fixture) ||
        !SparkValidationBindRoutedLayerCache(
            buffers,
            node_context,
            layer_index) ||
        !SparkValidationSetDecodeScalars(
            buffers,
            position,
            slot_mapping,
            context_length) ||
        !SparkValidationBindB12xMoePlanForLayer(
            buffers,
            node_context,
            layer_index))
    {
        return false;
    }
    SparkValidationEnableLayer3RoutedExpertNvfp4(
        0,
        buffers,
        node_context);
    if (!SparkValidationRunSubmitOnce(
            node_context,
            cuda_stream,
            driver_path,
            &elapsed_microseconds) ||
        !SparkValidationCudaSucceeded(
            cudaStreamSynchronize(cuda_stream),
            "cudaStreamSynchronize production routed B12x layer") ||
        !SparkValidationMaybeTraceRoutedBuffers(buffers, layer_index) ||
        !SparkValidationReadLayer3TopKExpertIds(buffers, topk_expert_ids))
    {
        return false;
    }
    *total_microseconds += (double)elapsed_microseconds;
    if ((double)elapsed_microseconds > *maximum_observed_microseconds)
    {
        *maximum_observed_microseconds = (double)elapsed_microseconds;
    }
    *submission_count += 1u;
    fprintf(stderr, "routed_layer_production_topk_ids layer=%u elapsed_us=%.3f ids=%u,%u,%u,%u,%u,%u,%u,%u\n", layer_index, (double)elapsed_microseconds, topk_expert_ids[0], topk_expert_ids[1], topk_expert_ids[2], topk_expert_ids[3], topk_expert_ids[4], topk_expert_ids[5], topk_expert_ids[6], topk_expert_ids[7]);
    if (layer3_routed_expert != 0)
    {
        memset(layer3_routed_expert, 0, sizeof(*layer3_routed_expert));
        layer3_routed_expert->ready = 1u;
        layer3_routed_expert->selected_expert_id = topk_expert_ids[0];
        layer3_routed_expert->bound_expert_count =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
    }
    if (check_outputs != 0u &&
        (!SparkValidationCheckLayer3RouterTopK(buffers) ||
         !SparkValidationCheckOutputs(
            buffers,
            real_lm_head,
            1u,
            1u,
            require_phase_clock)))
    {
        return false;
    }
    return true;
}

static bool SparkValidationRunDenseChainLayer3RoutedTopK(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    const char *model_directory,
    uint32_t input_token_id,
    const SparkValidationPromptTokenWindow *prompt_tokens,
    SparkValidationRealLmHeadFixture *real_lm_head,
    SparkValidationLayer3RoutedExpertNvfp4Fixture *layer3_routed_expert,
    uint32_t first_routed_layer_index,
    uint32_t routed_chain_layer_count,
    uint32_t current_token_only,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count,
    float *maximum_reference_error)
{
    uint32_t prefill_index;
    uint32_t layer_index;
    uint32_t routed_layer_offset;
    uint64_t copied_bytes;

    if (first_routed_layer_index < SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX ||
        first_routed_layer_index >= SPARK_VALIDATION_LAYER_COUNT)
    {
        fprintf(stderr, "first routed chain layer is invalid layer=%u\n", first_routed_layer_index);
        return false;
    }
    if (routed_chain_layer_count == 0u ||
        routed_chain_layer_count > SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT ||
        routed_chain_layer_count >
            (SPARK_VALIDATION_LAYER_COUNT - first_routed_layer_index))
    {
        fprintf(stderr, "routed chain layer count is invalid count=%u\n", routed_chain_layer_count);
        return false;
    }
    *total_microseconds = 0.0;
    *maximum_observed_microseconds = 0.0;
    *submission_count = 0u;
    *maximum_reference_error = 0.0f;
    copied_bytes = 0u;
    if (current_token_only == 0u)
    {
        for (prefill_index = 0u;
             prefill_index < SPARK_VALIDATION_CONTEXT_LENGTH - 1u;
             ++prefill_index)
        {
            uint32_t token_id;
            uint32_t position;
            uint32_t slot_mapping;
            uint32_t context_length;

            token_id = SparkValidationResolveContextToken(
                prompt_tokens,
                input_token_id,
                prefill_index);
            position = SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET + prefill_index;
            slot_mapping = SPARK_VALIDATION_REMAP_CACHE_SLOT0 + prefill_index;
            context_length = prefill_index + 1u;
            if (!SparkValidationCopyInputEmbeddingBf16Row(
                    model_directory,
                    token_id,
                    buffers->input_hidden_bf16,
                    &copied_bytes))
                return false;
            for (layer_index = 0u;
                 layer_index < SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT;
                 ++layer_index)
            {
                if (!SparkValidationRunChainedDenseLayer(
                        buffers,
                        node_context,
                        cuda_stream,
                        driver_path,
                        model_directory,
                        real_lm_head,
                        layer_index,
                        position,
                        slot_mapping,
                        context_length,
                        0u,
                        total_microseconds,
                        maximum_observed_microseconds,
                        submission_count,
                        maximum_reference_error))
                    return false;
            }
            if (!SparkValidationCopyLayerOutputToInput(buffers))
                return false;
            for (routed_layer_offset = 0u;
                 routed_layer_offset < routed_chain_layer_count;
                 ++routed_layer_offset)
            {
                if (!SparkValidationRunRoutedLayerProductionB12x(
                        buffers,
                        node_context,
                        cuda_stream,
                        driver_path,
                        model_directory,
                        real_lm_head,
                        layer3_routed_expert,
                        first_routed_layer_index + routed_layer_offset,
                        position,
                        slot_mapping,
                        context_length,
                        0u,
                        1u,
                        total_microseconds,
                        maximum_observed_microseconds,
                        submission_count))
                    return false;
                if (routed_layer_offset + 1u < routed_chain_layer_count &&
                    !SparkValidationCopyLayerOutputToInput(buffers))
                    return false;
            }
        }
    }
    if (!SparkValidationCopyInputEmbeddingBf16Row(
            model_directory,
            SparkValidationResolveContextToken(
                prompt_tokens,
                input_token_id,
                SPARK_VALIDATION_CONTEXT_LENGTH - 1u),
            buffers->input_hidden_bf16,
            &copied_bytes))
        return false;
    for (layer_index = 0u;
         layer_index < SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT;
         ++layer_index)
    {
        if (!SparkValidationRunChainedDenseLayer(
                buffers,
                node_context,
                cuda_stream,
                driver_path,
                model_directory,
                real_lm_head,
                layer_index,
                SPARK_VALIDATION_CURRENT_POSITION,
                SPARK_VALIDATION_CURRENT_CACHE_SLOT,
                SPARK_VALIDATION_CONTEXT_LENGTH,
                0u,
                total_microseconds,
                maximum_observed_microseconds,
                submission_count,
                maximum_reference_error))
            return false;
    }
    if (!SparkValidationCopyLayerOutputToInput(buffers))
        return false;
    for (routed_layer_offset = 0u;
         routed_layer_offset < routed_chain_layer_count;
         ++routed_layer_offset)
    {
        if (!SparkValidationRunRoutedLayerProductionB12x(
                buffers,
                node_context,
                cuda_stream,
                driver_path,
                model_directory,
                real_lm_head,
                layer3_routed_expert,
                first_routed_layer_index + routed_layer_offset,
                SPARK_VALIDATION_CURRENT_POSITION,
                SPARK_VALIDATION_CURRENT_CACHE_SLOT,
                SPARK_VALIDATION_CONTEXT_LENGTH,
                0u,
                1u,
                total_microseconds,
                maximum_observed_microseconds,
                submission_count))
            return false;
        if (routed_layer_offset + 1u < routed_chain_layer_count &&
            !SparkValidationCopyLayerOutputToInput(buffers))
            return false;
    }
    fprintf(stderr, "dense_chain_layer3_embedding_bf16_bytes=%llu\n", (unsigned long long)copied_bytes);
    return true;
}

static bool SparkValidationRunRoutedChainFromHidden(
    SparkValidationDeviceBuffers *buffers,
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    const char *model_directory,
    SparkValidationRealLmHeadFixture *real_lm_head,
    SparkValidationLayer3RoutedExpertNvfp4Fixture *layer3_routed_expert,
    uint32_t first_routed_layer_index,
    uint32_t routed_chain_layer_count,
    uint32_t final_token_stage,
    uint32_t require_phase_clock,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count)
{
    uint32_t routed_layer_offset;

    if (first_routed_layer_index < SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX ||
        first_routed_layer_index >= SPARK_VALIDATION_LAYER_COUNT)
    {
        fprintf(stderr, "first routed pipeline layer is invalid layer=%u\n", first_routed_layer_index);
        return false;
    }
    if (routed_chain_layer_count == 0u ||
        routed_chain_layer_count > SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT ||
        routed_chain_layer_count >
            (SPARK_VALIDATION_LAYER_COUNT - first_routed_layer_index))
    {
        fprintf(stderr, "routed pipeline layer count is invalid count=%u\n", routed_chain_layer_count);
        return false;
    }
    *total_microseconds = 0.0;
    *maximum_observed_microseconds = 0.0;
    *submission_count = 0u;
    for (routed_layer_offset = 0u;
         routed_layer_offset < routed_chain_layer_count;
         ++routed_layer_offset)
    {
        uint32_t run_final_outputs;

        run_final_outputs = final_token_stage != 0u &&
            routed_layer_offset + 1u == routed_chain_layer_count;
        SparkValidationSetOutputHiddenOnly(
            node_context,
            run_final_outputs == 0u);
        if (!SparkValidationRunRoutedLayerProductionB12x(
                buffers,
                node_context,
                cuda_stream,
                driver_path,
                model_directory,
                real_lm_head,
                layer3_routed_expert,
                first_routed_layer_index + routed_layer_offset,
                SPARK_VALIDATION_CURRENT_POSITION,
                SPARK_VALIDATION_CURRENT_CACHE_SLOT,
                SPARK_VALIDATION_CONTEXT_LENGTH,
                run_final_outputs,
                require_phase_clock,
                total_microseconds,
                maximum_observed_microseconds,
                submission_count))
        {
            SparkValidationSetOutputHiddenOnly(
                node_context,
                final_token_stage == 0u);
            return false;
        }
        if (routed_layer_offset + 1u < routed_chain_layer_count &&
            !SparkValidationCopyLayerOutputToInput(buffers))
        {
            SparkValidationSetOutputHiddenOnly(
                node_context,
                final_token_stage == 0u);
            return false;
        }
    }
    SparkValidationSetOutputHiddenOnly(
        node_context,
        final_token_stage == 0u);
    return true;
}

static uint32_t SparkValidationExactPp13BucketForActiveCount(void)
{
    return SparkGlm52StagePlanSelectBatchBucketValue(
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT);
}

static bool SparkValidationAllocateExactFinalEpilogueWorkspace(
    SparkValidationExactPp13StageSliceRuntime *runtime,
    uint32_t bucket)
{
    uint64_t candidate_count;
    uint64_t workspace_bytes;

    runtime->final_epilogue_workspace = 0;
    runtime->final_epilogue_workspace_bytes = 0u;
    candidate_count =
        (uint64_t)bucket *
        (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u) *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT;
    workspace_bytes =
        (candidate_count * (uint64_t)sizeof(float)) +
        (candidate_count * (uint64_t)sizeof(uint32_t));
    if (workspace_bytes == 0u)
        return false;
    if (!SparkValidationAllocateZeroed(
            &runtime->final_epilogue_workspace,
            workspace_bytes,
            "cudaMalloc exact_pp13_final_epilogue_workspace"))
        return false;
    runtime->final_epilogue_workspace_bytes = workspace_bytes;
    return true;
}

static bool SparkValidationInitializeExactPp13StageSlicePlan(
    SparkValidationExactPp13StageSliceRuntime *runtime,
    uint32_t first_layer_index,
    uint32_t final_token_stage)
{
    uint32_t bucket;
    uint32_t stage_index;
    uint32_t capability_flags;

    if (runtime == 0 ||
        first_layer_index % SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT != 0u ||
        first_layer_index + SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT >
            SPARK_VALIDATION_LAYER_COUNT)
    {
        return false;
    }
    bucket = SparkValidationExactPp13BucketForActiveCount();
    if (bucket == 0u)
    {
        fprintf(
            stderr,
            "exact PP13 stage-slice validation supports B<=%u active sequences\n",
            SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET);
        return false;
    }
    stage_index = first_layer_index /
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT;
    capability_flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PRODUCTION_PP13_CAPABILITIES;
    if (!SparkValidationCudaSucceeded(
            cudaStreamCreate(&runtime->query_branch_stream),
            "cudaStreamCreate exact_pp13_query_branch") ||
        !SparkValidationCudaSucceeded(
            cudaStreamCreate(&runtime->kv_branch_stream),
            "cudaStreamCreate exact_pp13_kv_branch") ||
        !SparkValidationCudaSucceeded(
            cudaEventCreate(&runtime->branch_ready_event),
            "cudaEventCreate exact_pp13_branch_ready") ||
        !SparkValidationCudaSucceeded(
            cudaEventCreate(&runtime->query_branch_event),
            "cudaEventCreate exact_pp13_query_done") ||
        !SparkValidationCudaSucceeded(
            cudaEventCreate(&runtime->kv_branch_event),
            "cudaEventCreate exact_pp13_kv_done") ||
        !SparkValidationAllocateExactFinalEpilogueWorkspace(runtime, bucket))
    {
        return false;
    }
    memset(&runtime->exact_stage_slice_plan, 0, sizeof(runtime->exact_stage_slice_plan));
    runtime->exact_stage_slice_plan.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_ABI_VERSION;
    runtime->exact_stage_slice_plan.descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_DESCRIPTOR_BYTES;
    runtime->exact_stage_slice_plan.stage_index = stage_index;
    runtime->exact_stage_slice_plan.first_layer_index = first_layer_index;
    runtime->exact_stage_slice_plan.layer_count =
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT;
    runtime->exact_stage_slice_plan.batch_bucket = bucket;
    runtime->exact_stage_slice_plan.maximum_active_sequence_count = bucket;
    runtime->exact_stage_slice_plan.capability_flags = capability_flags;
    runtime->exact_stage_slice_plan.query_branch_stream =
        runtime->query_branch_stream;
    runtime->exact_stage_slice_plan.kv_branch_stream =
        runtime->kv_branch_stream;
    runtime->exact_stage_slice_plan.branch_ready_event =
        runtime->branch_ready_event;
    runtime->exact_stage_slice_plan.query_branch_event =
        runtime->query_branch_event;
    runtime->exact_stage_slice_plan.kv_branch_event =
        runtime->kv_branch_event;
    runtime->exact_stage_slice_plan.workspace =
        runtime->final_epilogue_workspace;
    runtime->exact_stage_slice_plan.workspace_bytes =
        runtime->final_epilogue_workspace_bytes;
    runtime->exact_stage_slice_plan.validated_maximum_latency_ns =
        1000000000ull;
    memset(&runtime->stage_slice_plan, 0, sizeof(runtime->stage_slice_plan));
    runtime->stage_slice_plan.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PLAN_ABI_VERSION;
    runtime->stage_slice_plan.maximum_active_sequence_count = bucket;
    runtime->stage_slice_plan.maximum_layer_count =
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT;
    runtime->stage_slice_plan.capability_flags = capability_flags;
    runtime->stage_slice_plan.opaque_state =
        &runtime->exact_stage_slice_plan;
    runtime->stage_slice_plan.workspace =
        runtime->final_epilogue_workspace;
    runtime->stage_slice_plan.workspace_bytes =
        runtime->final_epilogue_workspace_bytes;
    runtime->stage_slice_plan.validated_maximum_latency_ns = 1000000000ull;
    if (final_token_stage != 0u &&
        first_layer_index + SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT !=
            SPARK_VALIDATION_LAYER_COUNT)
    {
        return false;
    }
    return true;
}

static bool SparkValidationPrepareExactPp13StageSliceLayer(
    SparkValidationExactPp13StageSliceRuntime *runtime,
    const char *model_directory,
    cudaStream_t cuda_stream,
    uint32_t first_layer_index,
    uint32_t layer_offset,
    uint32_t final_token_stage,
    uint32_t disable_graph_replay,
    uint32_t model_quantization)
{
    SparkValidationLayer0AttentionBf16Fixture attention_fixture;
    SparkValidationLayer0DenseBf16Fixture dense_fixture;
    SparkValidationLayer3RouterBf16Fixture router_fixture;
    SparkValidationFinalNormBf16Fixture final_norm_fixture;
    SparkValidationRealLmHeadFixture real_lm_head_fixture;
    SparkValidationDeviceBuffers *buffers;
    SparkGlm52ResidentDecodeStageNodeContext *node_context;
    SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;
    uint32_t layer_index;
    uint32_t required_linear_plan_mask;
    uint32_t use_dense_mlp;

    layer_index = first_layer_index + layer_offset;
    use_dense_mlp = layer_index < SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX;
    buffers = &runtime->buffers[layer_offset];
    pipeline_slot = &runtime->pipeline_slots[layer_offset];
    node_context = &runtime->node_contexts[layer_offset];
    if (!SparkValidationAllocateDeviceBuffers(buffers) ||
        !SparkValidationInitializeDenseLayerCacheAliases(buffers) ||
        !SparkValidationInitializeDeviceInputs(buffers))
    {
        return false;
    }
    buffers->routed_layer_base_index = layer_index;
    SparkValidationConfigureNode(
        buffers,
        cuda_stream,
        pipeline_slot,
        &runtime->cuda_slot_states[layer_offset],
        node_context,
        use_dense_mlp,
        0u);
    pipeline_slot->input_hidden_bf16 =
        layer_offset == 0u
        ? buffers->input_hidden_bf16
        : runtime->buffers[layer_offset - 1u].layer_output_hidden_bf16;
    if (getenv("GLM52_EXACT_PP13_DEBUG_LAUNCH_CHECK") != 0)
    {
        node_context->launch_check_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR;
        node_context->phase_clock_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64;
    }
    else
    {
        node_context->launch_check_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE;
        node_context->phase_clock_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED;
        node_context->reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION;
    }
    if (getenv("GLM52_EXACT_PP13_PHASE_PROFILE") != 0)
    {
        node_context->phase_clock_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64;
    }
    if (disable_graph_replay == 0u)
    {
        node_context->enable_cuda_graph_replay = 1u;
        node_context->reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY;
    }
    required_linear_plan_mask =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS;
    if (!SparkValidationLoadLayer0AttentionBf16Fixture(
            buffers,
            model_directory,
            layer_index,
            &attention_fixture) ||
        !SparkValidationSetDecodeScalars(
            buffers,
            SPARK_VALIDATION_CURRENT_POSITION,
            SPARK_VALIDATION_CURRENT_CACHE_SLOT,
            SPARK_VALIDATION_CONTEXT_LENGTH))
    {
        return false;
    }
    if (use_dense_mlp != 0u)
    {
        if (!SparkValidationLoadLayer0DenseBf16Fixture(
                buffers,
                model_directory,
                layer_index,
                &dense_fixture) ||
            !SparkValidationBindDenseLayerCache(
                buffers,
                node_context,
                layer_index))
        {
            return false;
        }
        required_linear_plan_mask |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN;
    }
    else
    {
        if (!SparkValidationLoadRoutedLayerRouterBf16Fixture(
                buffers,
                model_directory,
                layer_index,
                &router_fixture) ||
            !SparkValidationBindRoutedLayerCache(
                buffers,
                node_context,
                layer_index))
        {
            return false;
        }
        if (model_quantization ==
            SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_FP8)
        {
            if (!SparkValidationBindFp8MoePlanForLayer(
                    buffers,
                    node_context,
                    layer_index))
            {
                return false;
            }
            SparkValidationEnableLayer3RoutedExpertFp8(
                buffers,
                node_context);
        }
        else
        {
            if (!SparkValidationBindB12xMoePlanForLayer(
                    buffers,
                    node_context,
                    layer_index))
            {
                return false;
            }
            SparkValidationEnableLayer3RoutedExpertNvfp4(
                0,
                buffers,
                node_context);
        }
        required_linear_plan_mask |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ROUTER_LOGITS;
    }
    if (final_token_stage != 0u &&
        layer_offset + 1u == SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT)
    {
        if (!SparkValidationLoadFinalNormBf16Fixture(
                buffers,
                model_directory,
                &final_norm_fixture) ||
            !SparkValidationLoadRealLmHeadFixture(
                buffers,
                model_directory,
                &real_lm_head_fixture))
        {
            return false;
        }
    }
    if (!SparkValidationBindRequiredLinearPlans(
            buffers,
            node_context,
            cuda_stream,
            required_linear_plan_mask,
            1u))
    {
        return false;
    }
    runtime->node_context_pointers[layer_offset] = node_context;
    return true;
}

static bool SparkValidationRunExactPp13StageSliceSubmit(
    SparkValidationExactPp13StageSliceRuntime *runtime,
    cudaStream_t cuda_stream,
    uint32_t final_token_stage,
    float *elapsed_microseconds)
{
    SparkValidationCompletionState completion_state;
    SparkGlm52ResidentDecodeStageBackendCompletion completion;
    struct timespec host_start_time;
    struct timespec host_stop_time;
    uint64_t host_elapsed_nanoseconds;
    SparkStatus status;

    host_elapsed_nanoseconds = 0u;
    memset(&completion_state, 0, sizeof(completion_state));
    memset(&completion, 0, sizeof(completion));
    completion.function = SparkValidationCompletion;
    completion.context = &completion_state;
    if (clock_gettime(CLOCK_MONOTONIC, &host_start_time) != 0)
    {
        fprintf(stderr, "clock_gettime exact_pp13_start failed\n");
        return false;
    }
    status = SparkGlm52ResidentDecodeStageBackendSubmitStageSlice(
        &runtime->stage_slice_plan,
        runtime->node_context_pointers,
        SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT,
        0u,
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT,
        final_token_stage,
        0,
        0,
        &completion);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "exact PP13 stage-slice submit failed status=%d\n", (int)status);
        return false;
    }
    if (!SparkValidationCudaSucceeded(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize exact_pp13_timing"))
    {
        return false;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &host_stop_time) != 0)
    {
        fprintf(stderr, "clock_gettime exact_pp13_stop failed\n");
        return false;
    }
    if (completion_state.completion_count.load(std::memory_order_acquire) != 1u)
    {
        fprintf(stderr, "exact PP13 stage-slice completion callback did not fire\n");
        return false;
    }
    if (host_stop_time.tv_nsec >= host_start_time.tv_nsec)
    {
        host_elapsed_nanoseconds =
            ((uint64_t)(host_stop_time.tv_sec - host_start_time.tv_sec) *
             1000000000ull) +
            (uint64_t)(host_stop_time.tv_nsec - host_start_time.tv_nsec);
    }
    else
    {
        host_elapsed_nanoseconds =
            ((uint64_t)(host_stop_time.tv_sec - host_start_time.tv_sec - 1) *
             1000000000ull) +
            (uint64_t)(host_stop_time.tv_nsec + 1000000000 -
                       host_start_time.tv_nsec);
    }
    *elapsed_microseconds =
        (float)((double)host_elapsed_nanoseconds / 1000.0);
    return true;
}

static uint64_t SparkValidationPhaseDelta(
    const uint64_t *phase_clocks,
    uint32_t first_phase,
    uint32_t second_phase)
{
    if (phase_clocks == 0 ||
        first_phase >= SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT ||
        second_phase >= SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT ||
        phase_clocks[first_phase] == 0u ||
        phase_clocks[second_phase] == 0u ||
        phase_clocks[second_phase] < phase_clocks[first_phase])
    {
        return 0u;
    }
    return phase_clocks[second_phase] - phase_clocks[first_phase];
}

static bool SparkValidationPrintExactPp13PhaseProfile(
    SparkValidationExactPp13StageSliceRuntime *runtime,
    uint32_t first_layer_index)
{
    uint64_t phase_clocks[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT];
    uint32_t layer_offset;
    uint32_t layer_index;

    if (getenv("GLM52_EXACT_PP13_PHASE_PROFILE") == 0)
    {
        return true;
    }
    if (runtime == 0)
    {
        return false;
    }
    for (layer_offset = 0u;
         layer_offset < SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT;
         ++layer_offset)
    {
        memset(phase_clocks, 0, sizeof(phase_clocks));
        if (!SparkValidationCudaSucceeded(
                cudaMemcpy(
                    phase_clocks,
                    runtime->buffers[layer_offset].phase_clock_cycles,
                    sizeof(phase_clocks),
                    cudaMemcpyDeviceToHost),
                "copy exact pp13 phase profile"))
        {
            return false;
        }
        layer_index = first_layer_index + layer_offset;
        fprintf(
            stderr,
            "exact_pp13_phase_profile layer=%u attention_norm_ns=%llu attention_projection_ns=%llu dsa_selection_ns=%llu rope_kv_write_ns=%llu mla_attention_ns=%llu output_projection_ns=%llu post_attention_norm_ns=%llu local_moe_ns=%llu completion_tail_ns=%llu total_ns=%llu\n",
            layer_index,
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_SUBMITTED,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_NORM),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_NORM,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_PROJECTION),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_PROJECTION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_DSA_SELECTION),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_DSA_SELECTION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ROPE_KV_WRITE),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ROPE_KV_WRITE,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MLA_ATTENTION),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MLA_ATTENTION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_OUTPUT_PROJECTION),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_OUTPUT_PROJECTION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_POST_ATTENTION_NORM),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_POST_ATTENTION_NORM,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY),
            (unsigned long long)SparkValidationPhaseDelta(
                phase_clocks,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_SUBMITTED,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY));
    }
    return true;
}

static bool SparkValidationRunExactPp13StageSliceFromHidden(
    SparkValidationExactPp13StageSliceRuntime *runtime,
    cudaStream_t cuda_stream,
    const char *model_directory,
    const char *pipeline_input_hidden_path,
    SparkValidationRealLmHeadFixture *real_lm_head,
    uint32_t first_layer_index,
    uint32_t final_token_stage,
    uint32_t disable_graph_replay,
    uint32_t model_quantization,
    double *total_microseconds,
    double *maximum_observed_microseconds,
    uint32_t *submission_count)
{
    SparkValidationDeviceBuffers *last_buffers;
    float elapsed_microseconds;
    float warmup_microseconds;
    uint32_t layer_offset;

    memset(runtime, 0, sizeof(*runtime));
    *total_microseconds = 0.0;
    *maximum_observed_microseconds = 0.0;
    *submission_count = 0u;
    if (!SparkValidationInitializeExactPp13StageSlicePlan(
            runtime,
            first_layer_index,
            final_token_stage))
    {
        return false;
    }
    for (layer_offset = 0u;
         layer_offset < SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT;
         ++layer_offset)
    {
        if (!SparkValidationPrepareExactPp13StageSliceLayer(
                runtime,
                model_directory,
                cuda_stream,
                first_layer_index,
                layer_offset,
                final_token_stage,
                disable_graph_replay,
                model_quantization))
        {
            return false;
        }
    }
    if (!SparkValidationReadHiddenBf16File(
            &runtime->buffers[0],
            pipeline_input_hidden_path))
    {
        return false;
    }
    if (!SparkValidationRunExactPp13StageSliceSubmit(
            runtime,
            cuda_stream,
            final_token_stage,
            &warmup_microseconds) ||
        !SparkValidationRunExactPp13StageSliceSubmit(
            runtime,
            cuda_stream,
            final_token_stage,
            &elapsed_microseconds))
    {
        return false;
    }
    if (!SparkValidationPrintExactPp13PhaseProfile(runtime, first_layer_index))
    {
        return false;
    }
    *total_microseconds = (double)elapsed_microseconds;
    *maximum_observed_microseconds = (double)elapsed_microseconds;
    *submission_count = 1u;
    last_buffers =
        &runtime->buffers[SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT - 1u];
    fprintf(
        stderr,
        "exact_pp13_stage_slice_warmup_us=%.3f timed_us=%.3f\n",
        (double)warmup_microseconds,
        (double)elapsed_microseconds);
    if (final_token_stage != 0u)
    {
        uint32_t selected_token_id;
        uint32_t mtp_draft_token_id;
        uint32_t mtp_reject_token_id;

        if (!SparkValidationReadFinalTokenEvidence(
                last_buffers,
                &selected_token_id,
                &mtp_draft_token_id,
                &mtp_reject_token_id))
        {
            return false;
        }
        if (real_lm_head == 0 || real_lm_head->ready == 0u)
        {
            fprintf(stderr, "exact PP13 final stage requires real lm_head fixture\n");
            return false;
        }
        if (!SparkValidationCheckRealRestrictedLogits(
                last_buffers,
                real_lm_head,
                selected_token_id))
        {
            return false;
        }
        fprintf(
            stderr,
            "exact_pp13_stage_slice_final_evidence restricted_token=%u expected_token=%u mtp_draft=%u mtp_reject=%u real_lm_head=1 real_lm_head_max_logit_error=%.8f built_in_final_epilogue=1\n",
            selected_token_id,
            real_lm_head->expected_selected_token,
            mtp_draft_token_id,
            mtp_reject_token_id,
            real_lm_head->maximum_logit_error);
    }
    return true;
}

int main(int argc, char **argv)
{
    SparkValidationDeviceBuffers buffers;
    SparkValidationExactPp13StageSliceRuntime exact_stage_slice_runtime;
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slot;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState cuda_slot_state;
    SparkGlm52ResidentDecodeStageNodeContext node_context;
    SparkValidationRealLmHeadFixture real_lm_head;
    SparkValidationLayer0DenseBf16Fixture layer0_dense;
    SparkValidationLayer0AttentionBf16Fixture layer0_attention;
    SparkValidationInputEmbeddingBf16Fixture input_embedding;
    SparkValidationFinalNormBf16Fixture final_norm;
    SparkValidationPrefillKvBf16Fixture prefill_kv;
    SparkValidationPromptTokenWindow prompt_tokens;
    SparkValidationLayer3RouterBf16Fixture layer3_router;
    SparkValidationLayer3SharedExpertBf16Fixture layer3_shared_expert;
    SparkValidationLayer3RoutedExpertNvfp4Fixture layer3_routed_expert;
    cudaStream_t cuda_stream;
    const char *model_directory;
    const char *stage_pack_directory;
    const char *load_layer0_dense;
    const char *load_layer0_attention;
    const char *input_token_text;
    const char *write_input_embedding_hidden_text;
    const char *dense_layer_index_text;
    const char *prefill_kv_text;
    const char *prefill_token_ids_path;
    const char *check_layer0_reference_text;
    const char *check_layer0_full_reference_text;
    const char *chain_dense_layers_text;
    const char *load_layer3_router_text;
    const char *load_layer3_shared_expert_text;
    const char *load_layer3_routed_expert_text;
    const char *load_layer3_routed_expert_topk_text;
    const char *chain_dense_layer3_routed_expert_topk_text;
    const char *dense_prefix_current_token_only_text;
    const char *chain_routed_from_hidden_text;
    const char *chain_routed_from_hidden_final_text;
    const char *exact_pp13_stage_slice_text;
    const char *exact_pp13_stage_slice_final_text;
    const char *exact_pp13_disable_graph_replay_text;
    const char *exact_pp13_model_quantization_text;
    const char *pipeline_input_hidden_path;
    const char *pipeline_output_hidden_path;
    const char *routed_chain_first_layer_text;
    const char *routed_chain_layer_count_text;
    const char *enable_graph_replay_text;
    const char *production_timing_text;
    const char *dense_chain_current_token_only_text;
    double maximum_stage_microseconds;
    double total_microseconds;
    double maximum_observed_microseconds;
    float layer0_full_reference_max_error;
    uint32_t iteration;
    uint32_t use_dense_mlp;
    uint32_t use_attention_bf16;
    uint32_t use_input_embedding;
    uint32_t write_input_embedding_hidden;
    uint32_t use_prefill_kv;
    uint32_t check_layer0_reference;
    uint32_t check_layer0_full_reference;
    uint32_t use_dense_chain;
    uint32_t use_layer3_router;
    uint32_t use_layer3_shared_expert;
    uint32_t use_layer3_routed_expert;
    uint32_t use_layer3_routed_expert_topk;
    uint32_t use_dense_chain_layer3_routed_expert_topk;
    uint32_t dense_prefix_current_token_only;
    uint32_t use_routed_chain_from_hidden;
    uint32_t use_routed_chain_from_hidden_final;
    uint32_t use_exact_pp13_stage_slice;
    uint32_t use_exact_pp13_stage_slice_final;
    uint32_t disable_exact_pp13_graph_replay;
    uint32_t exact_pp13_model_quantization;
    uint32_t routed_chain_first_layer_index;
    uint32_t routed_chain_layer_count;
    uint32_t enable_graph_replay;
    uint32_t production_timing;
    uint32_t dense_chain_current_token_only;
    uint32_t required_linear_plan_mask;
    uint32_t input_token_id;
    uint32_t dense_layer_index;

    if (argc != 2 && argc != 3)
    {
        fprintf(stderr, "usage: %s MAX_STAGE_MICROSECONDS [DRIVER_SO]\n", argv[0]);
        return 2;
    }
    maximum_stage_microseconds = atof(argv[1]);
    if (maximum_stage_microseconds <= 0.0)
    {
        fprintf(stderr, "MAX_STAGE_MICROSECONDS must be positive\n");
        return 2;
    }
    memset(&real_lm_head, 0, sizeof(real_lm_head));
    memset(&layer0_dense, 0, sizeof(layer0_dense));
    memset(&layer0_attention, 0, sizeof(layer0_attention));
    memset(&input_embedding, 0, sizeof(input_embedding));
    memset(&final_norm, 0, sizeof(final_norm));
    memset(&prefill_kv, 0, sizeof(prefill_kv));
    memset(&prompt_tokens, 0, sizeof(prompt_tokens));
    memset(&layer3_router, 0, sizeof(layer3_router));
    memset(&layer3_shared_expert, 0, sizeof(layer3_shared_expert));
    memset(&layer3_routed_expert, 0, sizeof(layer3_routed_expert));
    model_directory = getenv("GLM52_MODEL_DIR");
    if (model_directory == 0)
        model_directory = "";
    stage_pack_directory = SparkValidationStagePackDirectory();
    load_layer0_dense = getenv("GLM52_LOAD_LAYER0_DENSE_BF16");
    load_layer0_attention = getenv("GLM52_LOAD_LAYER0_ATTENTION_BF16");
    input_token_text = getenv("GLM52_INPUT_TOKEN_ID");
    write_input_embedding_hidden_text =
        getenv("GLM52_WRITE_INPUT_EMBEDDING_HIDDEN_BF16");
    dense_layer_index_text = getenv("GLM52_DENSE_LAYER_INDEX");
    prefill_kv_text = getenv("GLM52_PREFILL_KV_FROM_EMBEDDINGS");
    prefill_token_ids_path = getenv("GLM52_PREFILL_TOKEN_IDS_FILE");
    check_layer0_reference_text = getenv("GLM52_CHECK_LAYER0_REFERENCE");
    check_layer0_full_reference_text =
        getenv("GLM52_CHECK_LAYER0_FULL_REFERENCE");
    chain_dense_layers_text = getenv("GLM52_CHAIN_DENSE_LAYERS");
    load_layer3_router_text = getenv("GLM52_LOAD_LAYER3_ROUTER_BF16");
    load_layer3_shared_expert_text =
        getenv("GLM52_LOAD_LAYER3_SHARED_EXPERT_BF16");
    load_layer3_routed_expert_text =
        getenv("GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4");
    load_layer3_routed_expert_topk_text =
        getenv("GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4_TOPK");
    chain_dense_layer3_routed_expert_topk_text =
        getenv("GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK");
    dense_prefix_current_token_only_text =
        getenv("GLM52_DENSE_PREFIX_CURRENT_TOKEN_ONLY");
    chain_routed_from_hidden_text =
        getenv("GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16");
    chain_routed_from_hidden_final_text =
        getenv("GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN");
    exact_pp13_stage_slice_text =
        getenv("GLM52_EXACT_PP13_STAGE_SLICE");
    exact_pp13_stage_slice_final_text =
        getenv("GLM52_EXACT_PP13_STAGE_SLICE_FINAL_TOKEN");
    exact_pp13_disable_graph_replay_text =
        getenv("GLM52_EXACT_PP13_DISABLE_GRAPH_REPLAY");
    exact_pp13_model_quantization_text =
        getenv("GLM52_EXACT_PP13_MODEL_QUANTIZATION");
    pipeline_input_hidden_path =
        getenv("GLM52_PIPELINE_INPUT_HIDDEN_BF16");
    pipeline_output_hidden_path =
        getenv("GLM52_PIPELINE_OUTPUT_HIDDEN_BF16");
    routed_chain_first_layer_text =
        getenv("GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX");
    routed_chain_layer_count_text =
        getenv("GLM52_ROUTED_CHAIN_LAYER_COUNT");
    enable_graph_replay_text =
        getenv("GLM52_ENABLE_CUDA_GRAPH_REPLAY");
    production_timing_text =
        getenv("GLM52_PRODUCTION_TIMING");
    dense_chain_current_token_only_text =
        getenv("GLM52_DENSE_CHAIN_CURRENT_TOKEN_ONLY");
    use_dense_mlp = load_layer0_dense != 0 && load_layer0_dense[0] != '\0' &&
        strcmp(load_layer0_dense, "0") != 0;
    use_attention_bf16 =
        load_layer0_attention != 0 && load_layer0_attention[0] != '\0' &&
        strcmp(load_layer0_attention, "0") != 0;
    use_input_embedding = input_token_text != 0 && input_token_text[0] != '\0';
    write_input_embedding_hidden =
        write_input_embedding_hidden_text != 0 &&
        write_input_embedding_hidden_text[0] != '\0' &&
        strcmp(write_input_embedding_hidden_text, "0") != 0;
    use_prefill_kv = prefill_kv_text != 0 && prefill_kv_text[0] != '\0' &&
        strcmp(prefill_kv_text, "0") != 0;
    check_layer0_reference =
        check_layer0_reference_text != 0 &&
        check_layer0_reference_text[0] != '\0' &&
        strcmp(check_layer0_reference_text, "0") != 0;
    check_layer0_full_reference =
        check_layer0_full_reference_text != 0 &&
        check_layer0_full_reference_text[0] != '\0' &&
        strcmp(check_layer0_full_reference_text, "0") != 0;
    use_dense_chain =
        chain_dense_layers_text != 0 &&
        chain_dense_layers_text[0] != '\0' &&
        strcmp(chain_dense_layers_text, "0") != 0;
    use_layer3_router =
        load_layer3_router_text != 0 &&
        load_layer3_router_text[0] != '\0' &&
        strcmp(load_layer3_router_text, "0") != 0;
    use_layer3_shared_expert =
        load_layer3_shared_expert_text != 0 &&
        load_layer3_shared_expert_text[0] != '\0' &&
        strcmp(load_layer3_shared_expert_text, "0") != 0;
    use_layer3_routed_expert =
        load_layer3_routed_expert_text != 0 &&
        load_layer3_routed_expert_text[0] != '\0' &&
        strcmp(load_layer3_routed_expert_text, "0") != 0;
    use_layer3_routed_expert_topk =
        load_layer3_routed_expert_topk_text != 0 &&
        load_layer3_routed_expert_topk_text[0] != '\0' &&
        strcmp(load_layer3_routed_expert_topk_text, "0") != 0;
    use_dense_chain_layer3_routed_expert_topk =
        chain_dense_layer3_routed_expert_topk_text != 0 &&
        chain_dense_layer3_routed_expert_topk_text[0] != '\0' &&
        strcmp(chain_dense_layer3_routed_expert_topk_text, "0") != 0;
    dense_prefix_current_token_only =
        dense_prefix_current_token_only_text != 0 &&
        dense_prefix_current_token_only_text[0] != '\0' &&
        strcmp(dense_prefix_current_token_only_text, "0") != 0;
    use_routed_chain_from_hidden =
        chain_routed_from_hidden_text != 0 &&
        chain_routed_from_hidden_text[0] != '\0' &&
        strcmp(chain_routed_from_hidden_text, "0") != 0;
    use_routed_chain_from_hidden_final =
        chain_routed_from_hidden_final_text != 0 &&
        chain_routed_from_hidden_final_text[0] != '\0' &&
        strcmp(chain_routed_from_hidden_final_text, "0") != 0;
    use_exact_pp13_stage_slice_final =
        exact_pp13_stage_slice_final_text != 0 &&
        exact_pp13_stage_slice_final_text[0] != '\0' &&
        strcmp(exact_pp13_stage_slice_final_text, "0") != 0;
    use_exact_pp13_stage_slice =
        use_exact_pp13_stage_slice_final != 0u ||
        (exact_pp13_stage_slice_text != 0 &&
         exact_pp13_stage_slice_text[0] != '\0' &&
         strcmp(exact_pp13_stage_slice_text, "0") != 0);
    disable_exact_pp13_graph_replay =
        exact_pp13_disable_graph_replay_text != 0 &&
        exact_pp13_disable_graph_replay_text[0] != '\0' &&
        strcmp(exact_pp13_disable_graph_replay_text, "0") != 0;
    exact_pp13_model_quantization =
        SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_NVFP4;
    if (exact_pp13_model_quantization_text != 0 &&
        exact_pp13_model_quantization_text[0] != '\0')
    {
        if (strcmp(exact_pp13_model_quantization_text, "fp8") == 0 ||
            strcmp(exact_pp13_model_quantization_text, "fp8_e4m3") == 0 ||
            strcmp(exact_pp13_model_quantization_text, "8") == 0)
        {
            exact_pp13_model_quantization =
                SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_FP8;
        }
        else if (strcmp(exact_pp13_model_quantization_text, "nvfp4") == 0 ||
                 strcmp(exact_pp13_model_quantization_text, "4") == 0)
        {
            exact_pp13_model_quantization =
                SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_NVFP4;
        }
        else
        {
            fprintf(stderr, "unknown GLM52_EXACT_PP13_MODEL_QUANTIZATION=%s\n", exact_pp13_model_quantization_text);
            return 2;
        }
    }
    enable_graph_replay =
        enable_graph_replay_text != 0 &&
        enable_graph_replay_text[0] != '\0' &&
        strcmp(enable_graph_replay_text, "0") != 0;
    if (enable_graph_replay == 0u)
    {
        disable_exact_pp13_graph_replay = 1u;
    }
    production_timing =
        production_timing_text != 0 &&
        production_timing_text[0] != '\0' &&
        strcmp(production_timing_text, "0") != 0;
    dense_chain_current_token_only =
        dense_chain_current_token_only_text != 0 &&
        dense_chain_current_token_only_text[0] != '\0' &&
        strcmp(dense_chain_current_token_only_text, "0") != 0;
    if (use_layer3_routed_expert_topk != 0u)
    {
        use_layer3_routed_expert = 1u;
    }
    if (use_dense_chain_layer3_routed_expert_topk != 0u)
    {
        use_dense_mlp = 1u;
        use_attention_bf16 = 1u;
        use_layer3_routed_expert = 1u;
        use_layer3_routed_expert_topk = 1u;
    }
    if (use_routed_chain_from_hidden != 0u ||
        use_routed_chain_from_hidden_final != 0u)
    {
        use_attention_bf16 = 1u;
        use_layer3_routed_expert = 1u;
        use_layer3_routed_expert_topk = 1u;
    }
    if (use_dense_chain != 0u)
    {
        use_dense_mlp = 1u;
        use_attention_bf16 = 1u;
    }
    if (use_layer3_router != 0u)
    {
        use_dense_mlp = 0u;
        use_attention_bf16 = 1u;
    }
    if (use_layer3_shared_expert != 0u)
    {
        use_dense_mlp = 1u;
        use_attention_bf16 = 1u;
    }
    if (use_layer3_routed_expert != 0u &&
        use_dense_chain_layer3_routed_expert_topk == 0u)
    {
        use_dense_mlp = 0u;
        use_attention_bf16 = 1u;
    }
    layer0_full_reference_max_error = 0.0f;
    input_token_id = 0u;
    dense_layer_index = 0u;
    routed_chain_first_layer_index = SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX;
    routed_chain_layer_count = 1u;
    if (use_input_embedding != 0u)
    {
        char *end_pointer;
        unsigned long parsed_token_id;

        end_pointer = 0;
        parsed_token_id = strtoul(input_token_text, &end_pointer, 10);
        if (end_pointer == input_token_text ||
            *end_pointer != '\0' ||
            parsed_token_id >= 154880ul)
        {
            fprintf(stderr, "GLM52_INPUT_TOKEN_ID is invalid\n");
            return 2;
        }
        input_token_id = (uint32_t)parsed_token_id;
    }
    if (prefill_token_ids_path != 0 &&
        prefill_token_ids_path[0] != '\0' &&
        !SparkValidationReadPromptTokenWindow(
            prefill_token_ids_path,
            &prompt_tokens))
    {
        return 2;
    }
    if (prompt_tokens.ready != 0u &&
        use_input_embedding != 0u &&
        input_token_id !=
            prompt_tokens.token_ids[SPARK_VALIDATION_CONTEXT_LENGTH - 1u])
    {
        fprintf(stderr, "GLM52_INPUT_TOKEN_ID does not match prompt current token input=%u current=%u\n",
            input_token_id,
            prompt_tokens.token_ids[SPARK_VALIDATION_CONTEXT_LENGTH - 1u]);
        return 2;
    }
    if (write_input_embedding_hidden != 0u &&
        (use_input_embedding == 0u ||
         pipeline_output_hidden_path == 0 ||
         pipeline_output_hidden_path[0] == '\0' ||
         (stage_pack_directory == 0 &&
          (model_directory == 0 || model_directory[0] == '\0')) ||
         use_prefill_kv != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u ||
         use_dense_chain != 0u ||
         use_dense_chain_layer3_routed_expert_topk != 0u ||
         use_routed_chain_from_hidden != 0u ||
         use_routed_chain_from_hidden_final != 0u ||
         use_exact_pp13_stage_slice != 0u ||
         use_layer3_router != 0u ||
         use_layer3_shared_expert != 0u ||
         use_layer3_routed_expert != 0u))
    {
        fprintf(stderr, "GLM52_WRITE_INPUT_EMBEDDING_HIDDEN_BF16 requires only GLM52_INPUT_TOKEN_ID, GLM52_STAGE_PACK_DIR or GLM52_MODEL_DIR, and GLM52_PIPELINE_OUTPUT_HIDDEN_BF16\n");
        return 2;
    }
    if (dense_layer_index_text != 0 && dense_layer_index_text[0] != '\0')
    {
        char *end_pointer;
        unsigned long parsed_layer_index;

        end_pointer = 0;
        parsed_layer_index = strtoul(dense_layer_index_text, &end_pointer, 10);
        if (end_pointer == dense_layer_index_text ||
            *end_pointer != '\0' ||
            parsed_layer_index >= SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT)
        {
            fprintf(stderr, "GLM52_DENSE_LAYER_INDEX is invalid; expected 0..%u\n", SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT - 1u);
            return 2;
        }
        dense_layer_index = (uint32_t)parsed_layer_index;
    }
    if (routed_chain_layer_count_text != 0 &&
        routed_chain_layer_count_text[0] != '\0')
    {
        char *end_pointer;
        unsigned long parsed_layer_count;

        end_pointer = 0;
        parsed_layer_count = strtoul(routed_chain_layer_count_text, &end_pointer, 10);
        if (end_pointer == routed_chain_layer_count_text ||
            *end_pointer != '\0' ||
            parsed_layer_count == 0ul ||
            parsed_layer_count > (unsigned long)SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT)
        {
            fprintf(stderr, "GLM52_ROUTED_CHAIN_LAYER_COUNT is invalid; expected 1..%u\n", SPARK_VALIDATION_ROUTED_CHAIN_LAYER_LIMIT);
            return 2;
        }
        routed_chain_layer_count = (uint32_t)parsed_layer_count;
    }
    if (routed_chain_first_layer_text != 0 &&
        routed_chain_first_layer_text[0] != '\0')
    {
        char *end_pointer;
        unsigned long parsed_layer_index;

        end_pointer = 0;
        parsed_layer_index = strtoul(routed_chain_first_layer_text, &end_pointer, 10);
        if (end_pointer == routed_chain_first_layer_text ||
            *end_pointer != '\0' ||
            parsed_layer_index <
                (use_exact_pp13_stage_slice != 0u
                    ? 0ul
                    : (unsigned long)SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX) ||
            parsed_layer_index >= (unsigned long)SPARK_VALIDATION_LAYER_COUNT)
        {
            fprintf(stderr, "GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX is invalid; expected %u..%u\n", use_exact_pp13_stage_slice != 0u ? 0u : SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX, SPARK_VALIDATION_LAYER_COUNT - 1u);
            return 2;
        }
        routed_chain_first_layer_index = (uint32_t)parsed_layer_index;
    }
    if (routed_chain_layer_count >
        (SPARK_VALIDATION_LAYER_COUNT - routed_chain_first_layer_index))
    {
        fprintf(stderr, "GLM52 routed chain slice exceeds final layer first=%u count=%u\n", routed_chain_first_layer_index, routed_chain_layer_count);
        return 2;
    }
    if (use_prefill_kv != 0u &&
        (use_input_embedding == 0u ||
         use_attention_bf16 == 0u ||
         use_dense_mlp == 0u))
    {
        fprintf(stderr, "GLM52_PREFILL_KV_FROM_EMBEDDINGS requires input embedding, layer0 attention, and layer0 dense fixtures\n");
        return 2;
    }
    if (check_layer0_reference != 0u &&
        (use_input_embedding == 0u ||
         use_prefill_kv == 0u ||
         use_attention_bf16 == 0u ||
         use_dense_mlp == 0u))
    {
        fprintf(stderr, "GLM52_CHECK_LAYER0_REFERENCE requires input embedding, prefilled KV, layer0 attention, and layer0 dense fixtures\n");
        return 2;
    }
    if (check_layer0_full_reference != 0u &&
        (use_input_embedding == 0u ||
         use_prefill_kv == 0u ||
         use_attention_bf16 == 0u ||
         use_dense_mlp == 0u))
    {
        fprintf(stderr, "GLM52_CHECK_LAYER0_FULL_REFERENCE requires input embedding, prefilled KV, layer0 attention, and layer0 dense fixtures\n");
        return 2;
    }
    if (use_dense_chain != 0u &&
        (use_input_embedding == 0u ||
         use_prefill_kv != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_CHAIN_DENSE_LAYERS requires GLM52_INPUT_TOKEN_ID and owns its per-layer KV/reference checks\n");
        return 2;
    }
    if (use_dense_chain_layer3_routed_expert_topk != 0u &&
        (use_input_embedding == 0u ||
         use_prefill_kv != 0u ||
         use_dense_chain != 0u ||
         use_routed_chain_from_hidden != 0u ||
         use_routed_chain_from_hidden_final != 0u ||
         use_layer3_router != 0u ||
         use_layer3_shared_expert != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK requires GLM52_INPUT_TOKEN_ID and owns dense-prefix plus layer3 top-k checks\n");
        return 2;
    }
    if (use_routed_chain_from_hidden != 0u &&
        use_routed_chain_from_hidden_final != 0u)
    {
        fprintf(stderr, "choose only one routed-from-hidden mode: intermediate or final-token\n");
        return 2;
    }
    if (use_exact_pp13_stage_slice != 0u &&
        (pipeline_input_hidden_path == 0 ||
         pipeline_input_hidden_path[0] == '\0' ||
         pipeline_output_hidden_path == 0 ||
         pipeline_output_hidden_path[0] == '\0' ||
         (stage_pack_directory == 0 &&
          (model_directory == 0 || model_directory[0] == '\0')) ||
         routed_chain_layer_count !=
            SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT ||
         (routed_chain_first_layer_index %
            SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT) != 0u ||
         use_input_embedding != 0u ||
         use_prefill_kv != 0u ||
         use_dense_chain != 0u ||
         use_dense_chain_layer3_routed_expert_topk != 0u ||
         use_routed_chain_from_hidden != 0u ||
         use_routed_chain_from_hidden_final != 0u ||
         use_layer3_router != 0u ||
         use_layer3_shared_expert != 0u ||
         use_layer3_routed_expert != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_EXACT_PP13_STAGE_SLICE requires stage pack or model dir, input/output hidden, first layer multiple of 6, count=6, and owns production slice timing\n");
        return 2;
    }
    if (use_routed_chain_from_hidden != 0u &&
        (pipeline_input_hidden_path == 0 ||
         pipeline_input_hidden_path[0] == '\0' ||
         use_input_embedding != 0u ||
         use_prefill_kv != 0u ||
         use_dense_chain != 0u ||
         use_dense_chain_layer3_routed_expert_topk != 0u ||
         use_layer3_router != 0u ||
         use_layer3_shared_expert != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16 requires GLM52_PIPELINE_INPUT_HIDDEN_BF16 and owns routed slice checks\n");
        return 2;
    }
    if (use_routed_chain_from_hidden_final != 0u &&
        (pipeline_input_hidden_path == 0 ||
         pipeline_input_hidden_path[0] == '\0' ||
         use_input_embedding != 0u ||
         use_prefill_kv != 0u ||
         use_dense_chain != 0u ||
         use_dense_chain_layer3_routed_expert_topk != 0u ||
         use_layer3_router != 0u ||
         use_layer3_shared_expert != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN requires GLM52_PIPELINE_INPUT_HIDDEN_BF16 and owns final routed slice checks\n");
        return 2;
    }
    if (use_layer3_router != 0u &&
        (use_input_embedding == 0u ||
         use_prefill_kv != 0u ||
         use_dense_chain != 0u ||
         use_layer3_shared_expert != 0u ||
         use_layer3_routed_expert != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_LOAD_LAYER3_ROUTER_BF16 requires GLM52_INPUT_TOKEN_ID and owns its router top-k checks\n");
        return 2;
    }
    if (use_layer3_shared_expert != 0u &&
        (use_input_embedding == 0u ||
         use_prefill_kv != 0u ||
         use_dense_chain != 0u ||
         use_layer3_router != 0u ||
         use_layer3_routed_expert != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_LOAD_LAYER3_SHARED_EXPERT_BF16 requires GLM52_INPUT_TOKEN_ID and owns its shared expert checks\n");
        return 2;
    }
    if (use_layer3_routed_expert != 0u &&
        use_dense_chain_layer3_routed_expert_topk == 0u &&
        use_routed_chain_from_hidden == 0u &&
        use_routed_chain_from_hidden_final == 0u &&
        (use_input_embedding == 0u ||
         use_prefill_kv != 0u ||
         use_dense_chain != 0u ||
         use_layer3_router != 0u ||
         use_layer3_shared_expert != 0u ||
         check_layer0_reference != 0u ||
         check_layer0_full_reference != 0u))
    {
        fprintf(stderr, "GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4 requires GLM52_INPUT_TOKEN_ID and owns its routed expert checks\n");
        return 2;
    }
    if (use_prefill_kv != 0u &&
        prompt_tokens.ready == 0u &&
        input_token_id < SPARK_VALIDATION_CONTEXT_LENGTH - 1u)
    {
        fprintf(stderr, "GLM52_INPUT_TOKEN_ID is too small for prefill fixture\n");
        return 2;
    }
    if (use_dense_chain != 0u &&
        prompt_tokens.ready == 0u &&
        input_token_id < SPARK_VALIDATION_CONTEXT_LENGTH - 1u)
    {
        fprintf(stderr, "GLM52_INPUT_TOKEN_ID is too small for dense chain fixture\n");
        return 2;
    }
    if (use_dense_chain_layer3_routed_expert_topk != 0u &&
        prompt_tokens.ready == 0u &&
        input_token_id < SPARK_VALIDATION_CONTEXT_LENGTH - 1u)
    {
        fprintf(stderr, "GLM52_INPUT_TOKEN_ID is too small for dense+layer3 fixture\n");
        return 2;
    }
    cuda_stream = 0;
    if (!SparkValidationCudaSucceeded(cudaSetDevice(0), "cudaSetDevice") ||
        !SparkValidationCudaSucceeded(
            cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking),
            "cudaStreamCreate") ||
        !SparkValidationAllocateDeviceBuffers(&buffers) ||
        !SparkValidationInitializeDenseLayerCacheAliases(&buffers) ||
        !SparkValidationInitializeDeviceInputs(&buffers))
    {
        return 2;
    }
    buffers.routed_layer_base_index = routed_chain_first_layer_index;
    if ((use_dense_mlp != 0u ||
         use_attention_bf16 != 0u ||
         use_input_embedding != 0u) &&
        stage_pack_directory == 0 &&
        (model_directory == 0 || model_directory[0] == '\0'))
    {
        fprintf(stderr, "checkpoint fixtures require GLM52_STAGE_PACK_DIR or GLM52_MODEL_DIR\n");
        return 2;
    }
    if (use_input_embedding != 0u &&
        !SparkValidationLoadInputEmbeddingBf16Fixture(
            &buffers,
            model_directory,
            input_token_id,
            &input_embedding))
    {
        return 2;
    }
    if (write_input_embedding_hidden != 0u)
    {
        if (!SparkValidationWriteHiddenBf16File(
                pipeline_output_hidden_path,
                buffers.input_hidden_bf16))
        {
            return 2;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=input_embedding_hidden token=%u total_submissions=0 total_us=0.000 maximum_us=0.000 pipeline_output_hidden=%s bytes=%u\n",
            input_token_id,
            pipeline_output_hidden_path,
            SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
                2u);
        return 0;
    }
    if (use_attention_bf16 != 0u &&
        !SparkValidationLoadLayer0AttentionBf16Fixture(
            &buffers,
            model_directory,
            (use_layer3_router != 0u ||
             use_layer3_shared_expert != 0u ||
             use_layer3_routed_expert != 0u) ? 3u : dense_layer_index,
            &layer0_attention))
    {
        return 2;
    }
    if (use_dense_mlp != 0u &&
        use_layer3_shared_expert == 0u &&
        !SparkValidationLoadLayer0DenseBf16Fixture(
            &buffers,
            model_directory,
            dense_layer_index,
            &layer0_dense))
    {
        return 2;
    }
    if (use_layer3_shared_expert != 0u &&
        !SparkValidationLoadLayer3SharedExpertBf16Fixture(
            &buffers,
            model_directory,
            &layer3_shared_expert))
    {
        return 2;
    }
    if ((use_layer3_router != 0u ||
         use_layer3_routed_expert != 0u) &&
        !SparkValidationLoadLayer3RouterBf16Fixture(
            &buffers,
            model_directory,
            &layer3_router))
    {
        return 2;
    }
    if (use_layer3_routed_expert != 0u &&
        use_dense_chain_layer3_routed_expert_topk == 0u &&
        use_routed_chain_from_hidden == 0u &&
        use_routed_chain_from_hidden_final == 0u &&
        !SparkValidationLoadLayer3RoutedExpertNvfp4Fixture(
            &buffers,
            model_directory,
            &layer3_routed_expert,
            use_layer3_routed_expert_topk))
    {
        return 2;
    }
    if (use_routed_chain_from_hidden == 0u &&
        use_exact_pp13_stage_slice == 0u &&
        use_dense_chain_layer3_routed_expert_topk == 0u &&
        (stage_pack_directory != 0 ||
         (model_directory != 0 && model_directory[0] != '\0')) &&
        !SparkValidationLoadFinalNormBf16Fixture(
            &buffers,
            model_directory,
            &final_norm))
    {
        return 2;
    }
    if (use_routed_chain_from_hidden == 0u &&
        use_exact_pp13_stage_slice == 0u &&
        use_dense_chain_layer3_routed_expert_topk == 0u &&
        (stage_pack_directory != 0 ||
         (model_directory != 0 && model_directory[0] != '\0')) &&
        !SparkValidationLoadRealLmHeadFixture(
            &buffers,
            model_directory,
            &real_lm_head))
    {
        return 2;
    }
    if (use_exact_pp13_stage_slice != 0u &&
        use_exact_pp13_stage_slice_final != 0u &&
        (stage_pack_directory != 0 ||
         (model_directory != 0 && model_directory[0] != '\0')) &&
        !SparkValidationLoadRealLmHeadFixture(
            &buffers,
            model_directory,
            &real_lm_head))
    {
        return 2;
    }
    SparkValidationConfigureNode(
        &buffers,
        cuda_stream,
        &pipeline_slot,
        &cuda_slot_state,
        &node_context,
        use_dense_mlp,
        use_attention_bf16);
    if (use_routed_chain_from_hidden != 0u ||
        use_dense_chain_layer3_routed_expert_topk != 0u ||
        (use_dense_chain != 0u && production_timing != 0u))
    {
        node_context.reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY;
    }
    if (production_timing != 0u)
    {
        node_context.launch_check_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE;
        node_context.phase_clock_mode =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED;
        node_context.reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION;
    }
    if (enable_graph_replay != 0u)
    {
        node_context.enable_cuda_graph_replay = 1u;
        node_context.reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY;
    }
    if (use_layer3_router != 0u)
    {
        SparkValidationEnableLayer3RouterTopK(&buffers, &node_context);
    }
    if (use_layer3_shared_expert != 0u)
    {
        SparkValidationEnableLayer3SharedExpertBf16(&node_context);
    }
    if (use_layer3_routed_expert != 0u &&
        use_dense_chain_layer3_routed_expert_topk == 0u &&
        use_routed_chain_from_hidden == 0u &&
        use_routed_chain_from_hidden_final == 0u)
    {
        if (!SparkValidationBindB12xMoePlanForLayer(
                &buffers,
                &node_context,
                SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX))
        {
            return 2;
        }
        SparkValidationEnableLayer3RoutedExpertNvfp4(
            &layer3_routed_expert,
            &buffers,
            &node_context);
    }
    required_linear_plan_mask = 0u;
    if (use_dense_mlp != 0u ||
        use_dense_chain != 0u ||
        use_layer3_shared_expert != 0u ||
        use_dense_chain_layer3_routed_expert_topk != 0u)
    {
        required_linear_plan_mask |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN;
    }
    if (use_layer3_router != 0u ||
        use_layer3_routed_expert != 0u ||
        use_dense_chain_layer3_routed_expert_topk != 0u ||
        use_routed_chain_from_hidden != 0u ||
        use_routed_chain_from_hidden_final != 0u)
    {
        required_linear_plan_mask |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ROUTER_LOGITS;
    }
    if (use_attention_bf16 != 0u)
    {
        required_linear_plan_mask |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS;
    }
    if (use_routed_chain_from_hidden == 0u &&
        use_dense_chain_layer3_routed_expert_topk == 0u &&
        (stage_pack_directory != 0 ||
         (model_directory != 0 && model_directory[0] != '\0')))
    {
        required_linear_plan_mask |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RESTRICTED_LOGITS;
    }
    if (required_linear_plan_mask != 0u &&
        !SparkValidationBindRequiredLinearPlans(
            &buffers,
            &node_context,
            cuda_stream,
            required_linear_plan_mask,
            (check_layer0_reference != 0u ||
             check_layer0_full_reference != 0u) ? 0u : 1u))
    {
        return 2;
    }
    if ((use_routed_chain_from_hidden != 0u ||
         use_routed_chain_from_hidden_final != 0u) &&
        !SparkValidationReadHiddenBf16File(
            &buffers,
            pipeline_input_hidden_path))
    {
        return 2;
    }
    if (use_prefill_kv != 0u &&
        !SparkValidationRunPrefillKvBf16Fixture(
            &buffers,
            &node_context,
            cuda_stream,
            model_directory,
            input_token_id,
            &prompt_tokens,
            &prefill_kv))
    {
        return 2;
    }
    if (use_exact_pp13_stage_slice != 0u)
    {
        SparkGlm52ResidentDecodeStageCudaPipelineSlotState *exact_slot_state;
        uint32_t selected_token_id;
        uint32_t mtp_draft_token_id;
        uint32_t mtp_reject_token_id;
        uint32_t submission_count;
        uint32_t expected_layer_body_success_count;
        uint32_t expected_b12x_moe_success_count;
        uint32_t routed_layer_count;
        uint32_t layer_offset;
        uint64_t layer_body_success_count;
        uint64_t b12x_moe_success_count;

        selected_token_id = 0u;
        mtp_draft_token_id = 0u;
        mtp_reject_token_id = 0u;
        routed_layer_count = 0u;
        layer_body_success_count = 0u;
        b12x_moe_success_count = 0u;
        if (!SparkValidationRunExactPp13StageSliceFromHidden(
                &exact_stage_slice_runtime,
                cuda_stream,
                model_directory,
                pipeline_input_hidden_path,
                &real_lm_head,
                routed_chain_first_layer_index,
                use_exact_pp13_stage_slice_final,
                disable_exact_pp13_graph_replay,
                exact_pp13_model_quantization,
                &total_microseconds,
                &maximum_observed_microseconds,
                &submission_count) ||
            !SparkValidationWriteHiddenBf16File(
                pipeline_output_hidden_path,
                exact_stage_slice_runtime
                    .buffers[SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT - 1u]
                    .layer_output_hidden_bf16))
        {
            return 2;
        }
        if (use_exact_pp13_stage_slice_final != 0u &&
            !SparkValidationReadFinalTokenEvidence(
                &exact_stage_slice_runtime
                    .buffers[SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT - 1u],
                &selected_token_id,
                &mtp_draft_token_id,
                &mtp_reject_token_id))
        {
            return 2;
        }
        exact_slot_state = &exact_stage_slice_runtime.cuda_slot_states[0];
        for (layer_offset = 0u;
             layer_offset < SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT;
             ++layer_offset)
        {
            layer_body_success_count += exact_stage_slice_runtime
                .cuda_slot_states[layer_offset].layer_body_success_count;
            b12x_moe_success_count += exact_stage_slice_runtime
                .cuda_slot_states[layer_offset].b12x_moe_success_count;
            if (routed_chain_first_layer_index + layer_offset >=
                SPARK_VALIDATION_FIRST_ROUTED_LAYER_INDEX)
            {
                routed_layer_count += 1u;
            }
        }
        expected_layer_body_success_count =
            disable_exact_pp13_graph_replay != 0u
            ? (2u * SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT)
            : SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT;
        if (layer_body_success_count < expected_layer_body_success_count)
        {
            fprintf(
                stderr,
                "exact PP13 stage-slice expected at least %u successful layer bodies, saw %llu\n",
                expected_layer_body_success_count,
                (unsigned long long)layer_body_success_count);
            return 2;
        }
        expected_b12x_moe_success_count =
            exact_pp13_model_quantization ==
                SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_FP8
            ? 0u
            : (disable_exact_pp13_graph_replay != 0u
                ? (2u * routed_layer_count)
                : routed_layer_count);
        if (expected_b12x_moe_success_count != 0u &&
            b12x_moe_success_count < expected_b12x_moe_success_count)
        {
            fprintf(
                stderr,
                "exact PP13 stage-slice expected at least %u B12x MoE launches, saw %llu\n",
                expected_b12x_moe_success_count,
                (unsigned long long)b12x_moe_success_count);
            return 2;
        }
        if (disable_exact_pp13_graph_replay == 0u &&
            (exact_slot_state->graph_capture_count != 1u ||
             exact_slot_state->graph_replay_count != 2u))
        {
            fprintf(stderr, "exact PP13 stage-slice expected one graph capture, one warmup replay, and one timed replay; captures=%llu replays=%llu\n", (unsigned long long)exact_slot_state->graph_capture_count, (unsigned long long)exact_slot_state->graph_replay_count);
            return 2;
        }
        if (disable_exact_pp13_graph_replay != 0u &&
            (exact_slot_state->graph_capture_count != 0u ||
             exact_slot_state->graph_replay_count != 0u))
        {
            fprintf(stderr, "exact PP13 diagnostic graph-disable expected no graph capture or replay; captures=%llu replays=%llu\n", (unsigned long long)exact_slot_state->graph_capture_count, (unsigned long long)exact_slot_state->graph_replay_count);
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(
            &exact_stage_slice_runtime.node_contexts[0]);
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage exact PP13 stage-slice validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                submission_count);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=local_hidden_handoff exact_pp13_stage_slice=1 model_quantization=%s intermediate_stage=%u final_stage=%u stage_index=%u first_layer=%u layer_count=%u total_submissions=%u total_us=%.3f maximum_us=%.3f limit_us=%.3f pipeline_input_hidden=%s pipeline_output_hidden=%s restricted_token=%u expected_token=%u mtp_draft=%u mtp_reject=%u real_lm_head=%u real_lm_head_max_logit_error=%.8f built_in_exact_pp13_aot=1 built_in_final_epilogue=%u layer_bodies=%llu b12x_moe_launches=%llu launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            exact_pp13_model_quantization ==
                SPARK_VALIDATION_EXACT_PP13_MODEL_QUANTIZATION_FP8
                ? "fp8"
                : "nvfp4",
            use_exact_pp13_stage_slice_final == 0u ? 1u : 0u,
            use_exact_pp13_stage_slice_final,
            routed_chain_first_layer_index /
                SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT,
            routed_chain_first_layer_index,
            SPARK_VALIDATION_EXACT_PP13_STAGE_LAYER_COUNT,
            submission_count,
            total_microseconds,
            maximum_observed_microseconds,
            maximum_stage_microseconds,
            pipeline_input_hidden_path,
            pipeline_output_hidden_path,
            selected_token_id,
            real_lm_head.ready != 0u
                ? real_lm_head.expected_selected_token
                : SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN,
            mtp_draft_token_id,
            mtp_reject_token_id,
            real_lm_head.ready,
            real_lm_head.maximum_logit_error,
            use_exact_pp13_stage_slice_final,
            (unsigned long long)layer_body_success_count,
            (unsigned long long)b12x_moe_success_count,
            (unsigned long long)exact_slot_state->launch_chain_count,
            (unsigned long long)exact_slot_state->graph_capture_count,
            (unsigned long long)exact_slot_state->graph_replay_count);
        return 0;
    }
    if (argc == 3)
    {
        float elapsed_microseconds;

        if (use_dense_chain_layer3_routed_expert_topk != 0u)
        {
            uint32_t submission_count;
            bool chain_succeeded;

            chain_succeeded = SparkValidationRunDenseChainLayer3RoutedTopK(
                &buffers,
                &node_context,
                cuda_stream,
                argv[2],
                model_directory,
                input_token_id,
                &prompt_tokens,
                &real_lm_head,
                &layer3_routed_expert,
                routed_chain_first_layer_index,
                routed_chain_layer_count,
                dense_prefix_current_token_only,
                &total_microseconds,
                &maximum_observed_microseconds,
                &submission_count,
                &layer0_full_reference_max_error);
            if (!chain_succeeded)
            {
                return 2;
            }
            if (!SparkValidationWriteHiddenBf16File(
                    pipeline_output_hidden_path,
                    buffers.layer_output_hidden_bf16))
            {
                return 2;
            }
            if (enable_graph_replay != 0u &&
                cuda_slot_state.graph_capture_count == 0u &&
                cuda_slot_state.graph_replay_count == 0u)
            {
                fprintf(stderr, "GLM52_ENABLE_CUDA_GRAPH_REPLAY requested but no graph capture/replay was observed\n");
                return 2;
            }
            SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
            if (maximum_observed_microseconds > maximum_stage_microseconds)
            {
                fprintf(
                    stderr,
                    "glm52_resident_decode_stage dense+layer3 validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                    total_microseconds,
                    maximum_observed_microseconds,
                    maximum_stage_microseconds,
                    submission_count);
                return 1;
            }
            printf(
                "glm52_resident_decode_stage orchestrator validation passed fixture=remapped_nonzero_context4_h4_d8_r4 dense_prefix_routed_pipeline=1 intermediate_stage=1 production_b12x=1 dense_chain_layers=%u first_routed_layer=%u routed_chain_layers=%u total_submissions=%u total_us=%.3f maximum_us=%.3f limit_us=%.3f pipeline_output_hidden=%s input_embedding_token=%u prompt_prefill=%u prompt_source_tokens=%u prompt_context_tokens=%u,%u,%u,%u layer3_selected_expert=%u layer3_bound_experts=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
                SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT,
                routed_chain_first_layer_index,
                routed_chain_layer_count,
                submission_count,
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                pipeline_output_hidden_path != 0 ? pipeline_output_hidden_path : "",
                input_token_id,
                prompt_tokens.ready,
                prompt_tokens.source_token_count,
                prompt_tokens.token_ids[0],
                prompt_tokens.token_ids[1],
                prompt_tokens.token_ids[2],
                prompt_tokens.token_ids[3],
                layer3_routed_expert.selected_expert_id,
                layer3_routed_expert.bound_expert_count,
                (unsigned long long)cuda_slot_state.launch_chain_count,
                (unsigned long long)cuda_slot_state.graph_capture_count,
                (unsigned long long)cuda_slot_state.graph_replay_count);
            return 0;
        }
        if (use_routed_chain_from_hidden != 0u)
        {
            uint32_t submission_count;

            if (!SparkValidationRunRoutedChainFromHidden(
                    &buffers,
                    &node_context,
                    cuda_stream,
                    argv[2],
                    model_directory,
                    &real_lm_head,
                    &layer3_routed_expert,
                    routed_chain_first_layer_index,
                    routed_chain_layer_count,
                    0u,
                    1u,
                    &total_microseconds,
                    &maximum_observed_microseconds,
                    &submission_count) ||
                !SparkValidationWriteHiddenBf16File(
                    pipeline_output_hidden_path,
                    buffers.layer_output_hidden_bf16))
            {
                return 2;
            }
            SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
            if (maximum_observed_microseconds > maximum_stage_microseconds)
            {
                fprintf(
                    stderr,
                    "glm52_resident_decode_stage routed pipeline orchestrator validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                    total_microseconds,
                    maximum_observed_microseconds,
                    maximum_stage_microseconds,
                    submission_count);
                return 1;
            }
            printf(
                "glm52_resident_decode_stage orchestrator validation passed fixture=local_hidden_handoff routed_pipeline_from_hidden=1 intermediate_stage=1 first_routed_layer=%u routed_chain_layers=%u total_submissions=%u total_us=%.3f maximum_us=%.3f limit_us=%.3f pipeline_input_hidden=%s pipeline_output_hidden=%s layer3_selected_expert=%u layer3_bound_experts=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
                routed_chain_first_layer_index,
                routed_chain_layer_count,
                submission_count,
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                pipeline_input_hidden_path,
                pipeline_output_hidden_path != 0 ? pipeline_output_hidden_path : "",
                layer3_routed_expert.selected_expert_id,
                layer3_routed_expert.bound_expert_count,
                (unsigned long long)cuda_slot_state.launch_chain_count,
                (unsigned long long)cuda_slot_state.graph_capture_count,
                (unsigned long long)cuda_slot_state.graph_replay_count);
            return 0;
        }
        if (use_routed_chain_from_hidden_final != 0u)
        {
            uint32_t submission_count;
            uint32_t selected_token_id;
            uint32_t mtp_draft_token_id;
            uint32_t mtp_reject_token_id;

            if (!SparkValidationRunRoutedChainFromHidden(
                    &buffers,
                    &node_context,
                    cuda_stream,
                    argv[2],
                    model_directory,
                    &real_lm_head,
                    &layer3_routed_expert,
                    routed_chain_first_layer_index,
                    routed_chain_layer_count,
                    1u,
                    production_timing == 0u,
                    &total_microseconds,
                    &maximum_observed_microseconds,
                    &submission_count) ||
                !SparkValidationWriteHiddenBf16File(
                    pipeline_output_hidden_path,
                    buffers.layer_output_hidden_bf16) ||
                !SparkValidationReadFinalTokenEvidence(
                    &buffers,
                    &selected_token_id,
                    &mtp_draft_token_id,
                    &mtp_reject_token_id))
            {
                return 2;
            }
            if (enable_graph_replay != 0u &&
                cuda_slot_state.graph_capture_count == 0u &&
                cuda_slot_state.graph_replay_count == 0u)
            {
                fprintf(stderr, "GLM52_ENABLE_CUDA_GRAPH_REPLAY requested but no graph capture/replay was observed\n");
                return 2;
            }
            SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
            if (maximum_observed_microseconds > maximum_stage_microseconds)
            {
                fprintf(
                    stderr,
                    "glm52_resident_decode_stage final routed pipeline orchestrator validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                    total_microseconds,
                    maximum_observed_microseconds,
                    maximum_stage_microseconds,
                    submission_count);
                return 1;
            }
            printf(
                "glm52_resident_decode_stage orchestrator validation passed fixture=local_hidden_handoff routed_pipeline_from_hidden_final=1 final_stage=1 first_routed_layer=%u routed_chain_layers=%u total_submissions=%u total_us=%.3f maximum_us=%.3f limit_us=%.3f pipeline_input_hidden=%s pipeline_output_hidden=%s restricted_token=%u mtp_draft=%u mtp_reject=%u layer3_selected_expert=%u layer3_bound_experts=%u real_lm_head=%u real_lm_head_max_logit_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
                routed_chain_first_layer_index,
                routed_chain_layer_count,
                submission_count,
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                pipeline_input_hidden_path,
                pipeline_output_hidden_path != 0 ? pipeline_output_hidden_path : "",
                selected_token_id,
                mtp_draft_token_id,
                mtp_reject_token_id,
                layer3_routed_expert.selected_expert_id,
                layer3_routed_expert.bound_expert_count,
                real_lm_head.ready,
                real_lm_head.maximum_logit_error,
                (unsigned long long)cuda_slot_state.launch_chain_count,
                (unsigned long long)cuda_slot_state.graph_capture_count,
                (unsigned long long)cuda_slot_state.graph_replay_count);
            return 0;
        }
        if (use_dense_chain != 0u)
        {
            uint32_t submission_count;

            if (!SparkValidationRunChainedDenseLayers(
                    &buffers,
                    &node_context,
                    cuda_stream,
                    argv[2],
                    model_directory,
                    input_token_id,
                    &prompt_tokens,
                    &real_lm_head,
                    dense_chain_current_token_only,
                    production_timing == 0u,
                    &total_microseconds,
                    &maximum_observed_microseconds,
                    &submission_count,
                    &layer0_full_reference_max_error))
            {
                return 2;
            }
            SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
            if (maximum_observed_microseconds > maximum_stage_microseconds)
            {
                fprintf(
                    stderr,
                    "glm52_resident_decode_stage dense chain orchestrator validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                    total_microseconds,
                    maximum_observed_microseconds,
                    maximum_stage_microseconds,
                    submission_count);
                return 1;
            }
            printf(
                "glm52_resident_decode_stage orchestrator validation passed fixture=remapped_nonzero_context4_h4_d8_r4 dense_chain_layers=%u dense_chain_submissions=%u dense_chain_total_us=%.3f maximum_us=%.3f limit_us=%.3f restricted_token=%u mtp_draft=%u mtp_reject=%u input_embedding_bf16=%u input_embedding_token=%u layer0_reference_full=1 layer0_reference_full_max_error=%.8f real_lm_head=%u real_lm_head_max_logit_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
                SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT,
                submission_count,
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                real_lm_head.ready != 0u
                    ? real_lm_head.expected_selected_token
                    : SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN,
                SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN,
                SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN,
                input_embedding.ready,
                input_embedding.token_id,
                (double)layer0_full_reference_max_error,
                real_lm_head.ready,
                real_lm_head.maximum_logit_error,
                (unsigned long long)cuda_slot_state.launch_chain_count,
                (unsigned long long)cuda_slot_state.graph_capture_count,
                (unsigned long long)cuda_slot_state.graph_replay_count);
            return 0;
        }
        if (use_layer3_router != 0u)
        {
            if (!SparkValidationRunDriverOnce(
                    &node_context,
                    cuda_stream,
                    argv[2],
                    &elapsed_microseconds) ||
                !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize router") ||
                !SparkValidationCheckLayer3RouterTopK(&buffers))
            {
                return 2;
            }
            SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
            if ((double)elapsed_microseconds > maximum_stage_microseconds)
            {
                fprintf(
                    stderr,
                    "glm52_resident_decode_stage layer3 router orchestrator validation failed elapsed_us=%.3f limit_us=%.3f\n",
                    (double)elapsed_microseconds,
                    maximum_stage_microseconds);
                return 1;
            }
            printf(
                "glm52_resident_decode_stage orchestrator validation passed fixture=remapped_nonzero_context4_h4_d8_r4 layer3_router_bf16=1 elapsed_us=%.3f limit_us=%.3f input_embedding_bf16=%u input_embedding_token=%u layer3_router_bf16_bytes=%llu launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
                (double)elapsed_microseconds,
                maximum_stage_microseconds,
                input_embedding.ready,
                input_embedding.token_id,
                (unsigned long long)layer3_router.copied_bytes,
                (unsigned long long)cuda_slot_state.launch_chain_count,
                (unsigned long long)cuda_slot_state.graph_capture_count,
                (unsigned long long)cuda_slot_state.graph_replay_count);
            return 0;
        }
        if (use_layer3_shared_expert != 0u)
        {
            if (!SparkValidationRunDriverOnce(
                    &node_context,
                    cuda_stream,
                    argv[2],
                    &elapsed_microseconds) ||
                !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize shared expert") ||
                !SparkValidationCheckLayer3SharedExpertReferences(
                    &buffers,
                    &layer0_full_reference_max_error))
            {
                return 2;
            }
            SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
            if ((double)elapsed_microseconds > maximum_stage_microseconds)
            {
                fprintf(
                    stderr,
                    "glm52_resident_decode_stage layer3 shared expert orchestrator validation failed elapsed_us=%.3f limit_us=%.3f\n",
                    (double)elapsed_microseconds,
                    maximum_stage_microseconds);
                return 1;
            }
            printf(
                "glm52_resident_decode_stage orchestrator validation passed fixture=remapped_nonzero_context4_h4_d8_r4 layer3_shared_expert_bf16=1 elapsed_us=%.3f limit_us=%.3f input_embedding_bf16=%u input_embedding_token=%u layer3_shared_expert_bf16_bytes=%llu layer3_shared_expert_max_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
                (double)elapsed_microseconds,
                maximum_stage_microseconds,
                input_embedding.ready,
                input_embedding.token_id,
                (unsigned long long)layer3_shared_expert.copied_bytes,
                (double)layer0_full_reference_max_error,
                (unsigned long long)cuda_slot_state.launch_chain_count,
                (unsigned long long)cuda_slot_state.graph_capture_count,
                (unsigned long long)cuda_slot_state.graph_replay_count);
            return 0;
        }
        if (use_layer3_routed_expert != 0u)
        {
            if (!SparkValidationRunDriverOnce(
                    &node_context,
                    cuda_stream,
                    argv[2],
                    &elapsed_microseconds) ||
                !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize routed nvfp4") ||
                !SparkValidationCheckLayer3RouterTopK(&buffers) ||
                !SparkValidationCheckLayer3RoutedExpertNvfp4(
                    &buffers,
                    &layer3_routed_expert))
            {
                return 2;
            }
            SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
            if ((double)elapsed_microseconds > maximum_stage_microseconds)
            {
                fprintf(
                    stderr,
                    "glm52_resident_decode_stage layer3 routed nvfp4 orchestrator validation failed elapsed_us=%.3f limit_us=%.3f\n",
                    (double)elapsed_microseconds,
                    maximum_stage_microseconds);
                return 1;
            }
            printf(
                "glm52_resident_decode_stage orchestrator validation passed fixture=remapped_nonzero_context4_h4_d8_r4 layer3_routed_expert_nvfp4=1 elapsed_us=%.3f limit_us=%.3f input_embedding_bf16=%u input_embedding_token=%u layer3_router_bf16_bytes=%llu layer3_routed_expert_nvfp4_bytes=%llu selected_expert=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
                (double)elapsed_microseconds,
                maximum_stage_microseconds,
                input_embedding.ready,
                input_embedding.token_id,
                (unsigned long long)layer3_router.copied_bytes,
                (unsigned long long)layer3_routed_expert.copied_bytes,
                layer3_routed_expert.selected_expert_id,
                (unsigned long long)cuda_slot_state.launch_chain_count,
                (unsigned long long)cuda_slot_state.graph_capture_count,
                (unsigned long long)cuda_slot_state.graph_replay_count);
            return 0;
        }
        if (!SparkValidationRunDriverOnce(
                &node_context,
                cuda_stream,
                argv[2],
                &elapsed_microseconds) ||
            !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize") ||
            !SparkValidationCheckOutputs(&buffers, &real_lm_head, 0u, use_prefill_kv, 1u) ||
            !SparkValidationCheckLayer0References(
                &buffers,
                use_dense_mlp,
                check_layer0_reference,
                check_layer0_full_reference,
                &layer0_full_reference_max_error))
        {
            return 2;
        }
        if ((double)elapsed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage orchestrator validation failed elapsed_us=%.3f limit_us=%.3f\n",
                (double)elapsed_microseconds,
                maximum_stage_microseconds);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage orchestrator validation passed fixture=remapped_nonzero_context4_h4_d8_r4 elapsed_us=%.3f limit_us=%.3f restricted_token=%u mtp_draft=%u mtp_reject=%u dense_layer_index=%u input_embedding_bf16=%u input_embedding_token=%u input_embedding_bf16_bytes=%llu prefill_kv_bf16=%u prefill_kv_tokens=%u prefill_kv_bytes=%llu layer0_reference_sampled=%u layer0_reference_full=%u layer0_reference_full_max_error=%.8f real_lm_head=%u layer0_attention_bf16=%u layer0_attention_bf16_bytes=%llu layer0_dense_bf16=%u layer0_dense_bf16_bytes=%llu real_lm_head_max_logit_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            (double)elapsed_microseconds,
            maximum_stage_microseconds,
            real_lm_head.ready != 0u
                ? real_lm_head.expected_selected_token
                : SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN,
            SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN,
            SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN,
            dense_layer_index,
            input_embedding.ready,
            input_embedding.token_id,
            (unsigned long long)input_embedding.copied_bytes,
            prefill_kv.ready,
            prefill_kv.token_count,
            (unsigned long long)prefill_kv.copied_bytes,
            check_layer0_reference,
            check_layer0_full_reference,
            (double)layer0_full_reference_max_error,
            real_lm_head.ready,
            layer0_attention.ready,
            (unsigned long long)layer0_attention.copied_bytes,
            layer0_dense.ready,
            (unsigned long long)layer0_dense.copied_bytes,
            real_lm_head.maximum_logit_error,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    total_microseconds = 0.0;
    maximum_observed_microseconds = 0.0;
    if (use_dense_chain_layer3_routed_expert_topk != 0u)
    {
        uint32_t submission_count;

        if (!SparkValidationRunDenseChainLayer3RoutedTopK(
                &buffers,
                &node_context,
                cuda_stream,
                0,
                model_directory,
                input_token_id,
                &prompt_tokens,
                &real_lm_head,
                &layer3_routed_expert,
                routed_chain_first_layer_index,
                routed_chain_layer_count,
                dense_prefix_current_token_only,
                &total_microseconds,
                &maximum_observed_microseconds,
                &submission_count,
                &layer0_full_reference_max_error))
        {
            return 2;
        }
        if (!SparkValidationWriteHiddenBf16File(
                pipeline_output_hidden_path,
                buffers.layer_output_hidden_bf16))
        {
            return 2;
        }
        if (enable_graph_replay != 0u &&
            cuda_slot_state.graph_capture_count == 0u &&
            cuda_slot_state.graph_replay_count == 0u)
        {
            fprintf(stderr, "GLM52_ENABLE_CUDA_GRAPH_REPLAY requested but no graph capture/replay was observed\n");
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage dense+layer3 validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                submission_count);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 dense_prefix_routed_pipeline=1 intermediate_stage=1 production_b12x=1 dense_chain_layers=%u first_routed_layer=%u routed_chain_layers=%u total_submissions=%u total_us=%.3f maximum_us=%.3f limit_us=%.3f pipeline_output_hidden=%s input_embedding_token=%u prompt_prefill=%u prompt_source_tokens=%u prompt_context_tokens=%u,%u,%u,%u layer3_selected_expert=%u layer3_bound_experts=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT,
            routed_chain_first_layer_index,
            routed_chain_layer_count,
            submission_count,
            total_microseconds,
            maximum_observed_microseconds,
            maximum_stage_microseconds,
            pipeline_output_hidden_path != 0 ? pipeline_output_hidden_path : "",
            input_token_id,
            prompt_tokens.ready,
            prompt_tokens.source_token_count,
            prompt_tokens.token_ids[0],
            prompt_tokens.token_ids[1],
            prompt_tokens.token_ids[2],
            prompt_tokens.token_ids[3],
            layer3_routed_expert.selected_expert_id,
            layer3_routed_expert.bound_expert_count,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    if (use_routed_chain_from_hidden != 0u)
    {
        uint32_t submission_count;

        if (!SparkValidationRunRoutedChainFromHidden(
                &buffers,
                &node_context,
                cuda_stream,
                0,
                model_directory,
                &real_lm_head,
                &layer3_routed_expert,
                routed_chain_first_layer_index,
                routed_chain_layer_count,
                0u,
                1u,
                &total_microseconds,
                &maximum_observed_microseconds,
                &submission_count) ||
            !SparkValidationWriteHiddenBf16File(
                pipeline_output_hidden_path,
                buffers.layer_output_hidden_bf16))
        {
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage routed pipeline validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                submission_count);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=local_hidden_handoff routed_pipeline_from_hidden=1 intermediate_stage=1 first_routed_layer=%u routed_chain_layers=%u total_submissions=%u total_us=%.3f maximum_us=%.3f limit_us=%.3f pipeline_input_hidden=%s pipeline_output_hidden=%s layer3_selected_expert=%u layer3_bound_experts=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            routed_chain_first_layer_index,
            routed_chain_layer_count,
            submission_count,
            total_microseconds,
            maximum_observed_microseconds,
            maximum_stage_microseconds,
            pipeline_input_hidden_path,
            pipeline_output_hidden_path != 0 ? pipeline_output_hidden_path : "",
            layer3_routed_expert.selected_expert_id,
            layer3_routed_expert.bound_expert_count,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    if (use_routed_chain_from_hidden_final != 0u)
    {
        uint32_t submission_count;
        uint32_t selected_token_id;
        uint32_t mtp_draft_token_id;
        uint32_t mtp_reject_token_id;

        if (!SparkValidationRunRoutedChainFromHidden(
                &buffers,
                &node_context,
                cuda_stream,
                0,
                model_directory,
                &real_lm_head,
                &layer3_routed_expert,
                routed_chain_first_layer_index,
                routed_chain_layer_count,
                1u,
                production_timing == 0u,
                &total_microseconds,
                &maximum_observed_microseconds,
                &submission_count) ||
            !SparkValidationWriteHiddenBf16File(
                pipeline_output_hidden_path,
                buffers.layer_output_hidden_bf16) ||
            !SparkValidationReadFinalTokenEvidence(
                &buffers,
                &selected_token_id,
                &mtp_draft_token_id,
                &mtp_reject_token_id))
        {
            return 2;
        }
        if (enable_graph_replay != 0u &&
            cuda_slot_state.graph_capture_count == 0u &&
            cuda_slot_state.graph_replay_count == 0u)
        {
            fprintf(stderr, "GLM52_ENABLE_CUDA_GRAPH_REPLAY requested but no graph capture/replay was observed\n");
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage final routed pipeline validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                submission_count);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=local_hidden_handoff routed_pipeline_from_hidden_final=1 final_stage=1 first_routed_layer=%u routed_chain_layers=%u total_submissions=%u total_us=%.3f maximum_us=%.3f limit_us=%.3f pipeline_input_hidden=%s pipeline_output_hidden=%s restricted_token=%u mtp_draft=%u mtp_reject=%u layer3_selected_expert=%u layer3_bound_experts=%u real_lm_head=%u real_lm_head_max_logit_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            routed_chain_first_layer_index,
            routed_chain_layer_count,
            submission_count,
            total_microseconds,
            maximum_observed_microseconds,
            maximum_stage_microseconds,
            pipeline_input_hidden_path,
            pipeline_output_hidden_path != 0 ? pipeline_output_hidden_path : "",
            selected_token_id,
            mtp_draft_token_id,
            mtp_reject_token_id,
            layer3_routed_expert.selected_expert_id,
            layer3_routed_expert.bound_expert_count,
            real_lm_head.ready,
            real_lm_head.maximum_logit_error,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    if (use_dense_chain != 0u)
    {
        uint32_t submission_count;

        if (!SparkValidationRunChainedDenseLayers(
                &buffers,
                &node_context,
                cuda_stream,
                0,
                model_directory,
                input_token_id,
                &prompt_tokens,
                &real_lm_head,
                dense_chain_current_token_only,
                production_timing == 0u,
                &total_microseconds,
                &maximum_observed_microseconds,
                &submission_count,
                &layer0_full_reference_max_error))
        {
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage dense chain validation failed total_us=%.3f maximum_us=%.3f limit_us=%.3f submissions=%u\n",
                total_microseconds,
                maximum_observed_microseconds,
                maximum_stage_microseconds,
                submission_count);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 dense_chain_layers=%u dense_chain_submissions=%u dense_chain_total_us=%.3f maximum_us=%.3f limit_us=%.3f restricted_token=%u mtp_draft=%u mtp_reject=%u input_embedding_bf16=%u input_embedding_token=%u layer0_reference_full=1 layer0_reference_full_max_error=%.8f real_lm_head=%u real_lm_head_max_logit_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            SPARK_VALIDATION_FIRST_DENSE_LAYER_COUNT,
            submission_count,
            total_microseconds,
            maximum_observed_microseconds,
            maximum_stage_microseconds,
            real_lm_head.ready != 0u
                ? real_lm_head.expected_selected_token
                : SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN,
            SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN,
            SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN,
            input_embedding.ready,
            input_embedding.token_id,
            (double)layer0_full_reference_max_error,
            real_lm_head.ready,
            real_lm_head.maximum_logit_error,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    if (use_layer3_router != 0u)
    {
        float elapsed_microseconds;

        if (!SparkValidationRunOnce(
                &node_context,
                cuda_stream,
                &elapsed_microseconds) ||
            !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize router") ||
            !SparkValidationCheckLayer3RouterTopK(&buffers))
        {
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
        maximum_observed_microseconds = (double)elapsed_microseconds;
        total_microseconds = (double)elapsed_microseconds;
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage layer3 router validation failed elapsed_us=%.3f limit_us=%.3f\n",
                maximum_observed_microseconds,
                maximum_stage_microseconds);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 layer3_router_bf16=1 elapsed_us=%.3f limit_us=%.3f input_embedding_bf16=%u input_embedding_token=%u layer3_router_bf16_bytes=%llu launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            total_microseconds,
            maximum_stage_microseconds,
            input_embedding.ready,
            input_embedding.token_id,
            (unsigned long long)layer3_router.copied_bytes,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    if (use_layer3_shared_expert != 0u)
    {
        float elapsed_microseconds;

        if (!SparkValidationRunOnce(
                &node_context,
                cuda_stream,
                &elapsed_microseconds) ||
            !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize shared expert") ||
            !SparkValidationCheckLayer3SharedExpertReferences(
                &buffers,
                &layer0_full_reference_max_error))
        {
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
        maximum_observed_microseconds = (double)elapsed_microseconds;
        total_microseconds = (double)elapsed_microseconds;
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage layer3 shared expert validation failed elapsed_us=%.3f limit_us=%.3f\n",
                maximum_observed_microseconds,
                maximum_stage_microseconds);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 layer3_shared_expert_bf16=1 elapsed_us=%.3f limit_us=%.3f input_embedding_bf16=%u input_embedding_token=%u layer3_shared_expert_bf16_bytes=%llu layer3_shared_expert_max_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            total_microseconds,
            maximum_stage_microseconds,
            input_embedding.ready,
            input_embedding.token_id,
            (unsigned long long)layer3_shared_expert.copied_bytes,
            (double)layer0_full_reference_max_error,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    if (use_layer3_routed_expert != 0u)
    {
        float elapsed_microseconds;

        if (!SparkValidationRunOnce(
                &node_context,
                cuda_stream,
                &elapsed_microseconds) ||
            !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize routed nvfp4") ||
            !SparkValidationCheckLayer3RouterTopK(&buffers) ||
            !SparkValidationCheckLayer3RoutedExpertNvfp4(
                &buffers,
                &layer3_routed_expert))
        {
            return 2;
        }
        SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
        maximum_observed_microseconds = (double)elapsed_microseconds;
        total_microseconds = (double)elapsed_microseconds;
        if (maximum_observed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage layer3 routed nvfp4 validation failed elapsed_us=%.3f limit_us=%.3f\n",
                maximum_observed_microseconds,
                maximum_stage_microseconds);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 layer3_routed_expert_nvfp4=1 elapsed_us=%.3f limit_us=%.3f input_embedding_bf16=%u input_embedding_token=%u layer3_router_bf16_bytes=%llu layer3_routed_expert_nvfp4_bytes=%llu selected_expert=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            total_microseconds,
            maximum_stage_microseconds,
            input_embedding.ready,
            input_embedding.token_id,
            (unsigned long long)layer3_router.copied_bytes,
            (unsigned long long)layer3_routed_expert.copied_bytes,
            layer3_routed_expert.selected_expert_id,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    for (iteration = 0u;
         iteration < SPARK_VALIDATION_WARMUP_COUNT + SPARK_VALIDATION_MEASUREMENT_COUNT;
         ++iteration)
    {
        float elapsed_microseconds;

        if (!SparkValidationRunOnce(
                &node_context,
                cuda_stream,
                &elapsed_microseconds))
        {
            return 2;
        }
        if (iteration >= SPARK_VALIDATION_WARMUP_COUNT)
        {
            total_microseconds += (double)elapsed_microseconds;
            if ((double)elapsed_microseconds > maximum_observed_microseconds)
            {
                maximum_observed_microseconds = (double)elapsed_microseconds;
            }
        }
    }
    if (!SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize") ||
        !SparkValidationCheckOutputs(&buffers, &real_lm_head, 0u, use_prefill_kv, 1u) ||
        !SparkValidationCheckLayer0References(
            &buffers,
            use_dense_mlp,
            check_layer0_reference,
            check_layer0_full_reference,
            &layer0_full_reference_max_error))
    {
        return 2;
    }
    SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
    if (maximum_observed_microseconds > maximum_stage_microseconds)
    {
        fprintf(
            stderr,
            "glm52_resident_decode_stage validation failed average_us=%.3f maximum_us=%.3f limit_us=%.3f\n",
            total_microseconds / SPARK_VALIDATION_MEASUREMENT_COUNT,
            maximum_observed_microseconds,
            maximum_stage_microseconds);
        return 1;
    }
    printf(
        "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 average_us=%.3f maximum_us=%.3f limit_us=%.3f restricted_token=%u mtp_draft=%u mtp_reject=%u dense_layer_index=%u input_embedding_bf16=%u input_embedding_token=%u input_embedding_bf16_bytes=%llu prefill_kv_bf16=%u prefill_kv_tokens=%u prefill_kv_bytes=%llu layer0_reference_sampled=%u layer0_reference_full=%u layer0_reference_full_max_error=%.8f real_lm_head=%u layer0_attention_bf16=%u layer0_attention_bf16_bytes=%llu layer0_dense_bf16=%u layer0_dense_bf16_bytes=%llu real_lm_head_max_logit_error=%.8f launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
        total_microseconds / SPARK_VALIDATION_MEASUREMENT_COUNT,
        maximum_observed_microseconds,
        maximum_stage_microseconds,
        real_lm_head.ready != 0u
            ? real_lm_head.expected_selected_token
            : SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN,
        SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN,
        SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN,
        dense_layer_index,
        input_embedding.ready,
        input_embedding.token_id,
        (unsigned long long)input_embedding.copied_bytes,
        prefill_kv.ready,
        prefill_kv.token_count,
        (unsigned long long)prefill_kv.copied_bytes,
        check_layer0_reference,
        check_layer0_full_reference,
        (double)layer0_full_reference_max_error,
        real_lm_head.ready,
        layer0_attention.ready,
        (unsigned long long)layer0_attention.copied_bytes,
        layer0_dense.ready,
        (unsigned long long)layer0_dense.copied_bytes,
        real_lm_head.maximum_logit_error,
        (unsigned long long)cuda_slot_state.launch_chain_count,
        (unsigned long long)cuda_slot_state.graph_capture_count,
        (unsigned long long)cuda_slot_state.graph_replay_count);
    return 0;
}
