#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_SPARSE_MLA_FIRMWARE_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_SPARSE_MLA_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_SPARSE_MLA_NODE_CONTEXT_ABI_VERSION 5u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT 64u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION 512u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION 64u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS 576u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_BLOCK_TOKENS 64u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT 2048u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_MAX_PIPELINE_SLOT_COUNT 64u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_PIPELINE_SLOT_SCALAR_INDEX 0u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_SLOT_STATE_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_PLAN_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_TOKEN_INDEX UINT32_MAX

#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION 0x00000001u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_GRAPH_REPLAY 0x00000002u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION 0x00000004u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH 0x00000008u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_VALIDATED_LATENCY 0x00000010u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_CUSTOM_ATTENTION_PLAN 0x00000020u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_SOTA_FAST_PATH \
    (SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_GRAPH_REPLAY | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_VALIDATED_LATENCY | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_CUSTOM_ATTENTION_PLAN)
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_KNOWN_FLAGS \
    SPARK_GLM52_RESIDENT_SPARSE_MLA_EXECUTION_REQUIRE_SOTA_FAST_PATH

#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_STREAM_ORDERED 0x00000001u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_CUDA_GRAPH_REPLAY 0x00000002u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_FUSED_ROPE_KV_WRITE 0x00000004u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_TILED_ONLINE_SOFTMAX 0x00000008u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_ZERO_HOST_STAGING 0x00000010u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_ZERO_DEVICE_MEMCPY 0x00000020u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_SOTA_CAPABILITIES \
    (SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_STREAM_ORDERED | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_CUDA_GRAPH_REPLAY | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_FUSED_ROPE_KV_WRITE | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_TILED_ONLINE_SOFTMAX | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_ZERO_HOST_STAGING | \
     SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_CAPABILITY_ZERO_DEVICE_MEMCPY)


#define SPARK_GLM52_RESIDENT_SPARSE_MLA_MODULE_ID \
    "spark.glm52.resident_sparse_mla.bf16.h64.d512.r64.k2048.b64.rope_adjacent.v1"
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_TARGET \
    "cuda.sm121.glm52.resident_sparse_mla.bf16"

typedef enum SparkGlm52ResidentSparseMlaLaunchCheckMode
{
    SPARK_GLM52_RESIDENT_SPARSE_MLA_LAUNCH_CHECK_NONE = 0,
    SPARK_GLM52_RESIDENT_SPARSE_MLA_LAUNCH_CHECK_PEEK = 1,
    SPARK_GLM52_RESIDENT_SPARSE_MLA_LAUNCH_CHECK_SYNC_ON_ERROR = 2
} SparkGlm52ResidentSparseMlaLaunchCheckMode;


typedef enum SparkGlm52ResidentSparseMlaAttentionExecutionMode
{
    SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_EXECUTION_SCORE_CACHE_REFERENCE = 0,
    SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX = 1
} SparkGlm52ResidentSparseMlaAttentionExecutionMode;


typedef struct SparkGlm52ResidentSparseMlaAttentionPlan
{
    uint32_t abi_version;
    uint32_t maximum_active_sequence_count;
    uint32_t capability_flags;
    uint32_t reserved;
    void *launch_function;
    void *opaque_state;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentSparseMlaAttentionPlan;

typedef struct SparkGlm52ResidentSparseMlaCudaPipelineSlotState
{
    uint32_t abi_version;
    uint32_t graph_active_sequence_count;
    void *cuda_graph_exec;
    uint64_t graph_capture_count;
    uint64_t graph_replay_count;
    uint64_t launch_chain_count;
    uint64_t launch_error_count;
} SparkGlm52ResidentSparseMlaCudaPipelineSlotState;

typedef struct SparkGlm52ResidentSparseMlaPipelineSlot
{
    void *cuda_stream;
    const void *query_latent_bf16;
    const void *query_rope_input_bf16;
    const void *key_rope_input_bf16;
    const void *current_kv_latent_bf16;
    const uint32_t *positions;
    const uint32_t *slot_mapping;
    const uint32_t *block_table;
    const uint32_t *context_lengths;
    const uint32_t *first_block_token_offsets;
    const uint32_t *sparse_token_indices;
    void *rotated_query_rope_bf16;
    void *output_latent_bf16;
} SparkGlm52ResidentSparseMlaPipelineSlot;

typedef struct SparkGlm52ResidentSparseMlaNodeContext
{
    uint32_t abi_version;
    uint32_t pipeline_slot_count;
    uint32_t max_active_sequence_count;
    uint32_t cache_token_capacity;
    uint32_t kv_block_count;
    uint32_t max_blocks_per_sequence;
    uint32_t position_count;
    uint32_t reserved;
    float qk_scale;
    uint32_t reserved_1;
    const float *cos_table;
    const float *sin_table;
    void *mla_cache_bf16;
    const SparkGlm52ResidentSparseMlaPipelineSlot *pipeline_slots;
    SparkGlm52ResidentSparseMlaCudaPipelineSlotState *cuda_pipeline_slot_states;
    uint32_t launch_check_mode;
    uint32_t enable_cuda_graph_replay;
    uint32_t attention_execution_mode;
    uint32_t execution_flags;
    const SparkGlm52ResidentSparseMlaAttentionPlan *attention_plan;
    uint64_t validated_stage_latency_ns;
    uint64_t estimated_service_time_ns;
} SparkGlm52ResidentSparseMlaNodeContext;

SparkStatus SparkGlm52ResidentSparseMlaInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);

SparkStatus SparkGlm52ResidentSparseMlaExecute(
    void *module_state,
    SparkModelDriverFrame *frame);

SparkStatus SparkGlm52ResidentSparseMlaAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

SparkStatus SparkGlm52ResidentSparseMlaSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot);

void SparkGlm52ResidentSparseMlaDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
