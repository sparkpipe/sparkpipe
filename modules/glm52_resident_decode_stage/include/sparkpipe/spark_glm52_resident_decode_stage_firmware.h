#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 7u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION 6144u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION 512u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION 2048u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION 16384u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION 576u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION 28672u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION 192u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K 8u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION 2048u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION 12288u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK 128u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS 576u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT * \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT * \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION 4096u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT 2048u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT 2u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE 32u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT 5u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX 0u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CANCELLED_TOKEN_ID UINT32_MAX

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MODULE_ID \
    "spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048.b64.rv256.mtp2.v1"
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_TARGET \
    "cuda.sm121.glm52.resident_decode_stage.bf16"

typedef enum SparkGlm52ResidentDecodeStagePhase
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_SUBMITTED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_NORM = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_PROJECTION = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_DSA_SELECTION = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ROPE_KV_WRITE = 4,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MLA_ATTENTION = 5,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_OUTPUT_PROJECTION = 6,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_POST_ATTENTION_NORM = 7,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE = 8,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_RESTRICTED_LOGITS = 9,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MTP_DRAFT = 10,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MTP_VERIFY = 11,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY = 12
} SparkGlm52ResidentDecodeStagePhase;

typedef enum SparkGlm52ResidentDecodeStageMtpCounter
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED = 4
} SparkGlm52ResidentDecodeStageMtpCounter;

typedef enum SparkGlm52ResidentDecodeStageSparseIndexMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_COPY_CONTEXT_PREFIX = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DEBUG_SERIAL_TOPK = 2
} SparkGlm52ResidentDecodeStageSparseIndexMode;

typedef enum SparkGlm52ResidentDecodeStageLaunchCheckMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_PEEK = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR = 2
} SparkGlm52ResidentDecodeStageLaunchCheckMode;

typedef enum SparkGlm52ResidentDecodeStagePhaseClockMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64 = 1
} SparkGlm52ResidentDecodeStagePhaseClockMode;

typedef enum SparkGlm52ResidentDecodeStageProjectionMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16 = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_BF16 = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3 = 2
} SparkGlm52ResidentDecodeStageProjectionMode;

typedef enum SparkGlm52ResidentDecodeStageLayerProgressionMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_PRESELECTED_BF16_LOCAL_MOE = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY = 3
} SparkGlm52ResidentDecodeStageLayerProgressionMode;

typedef struct SparkGlm52ResidentDecodeStageCudaPipelineSlotState
{
    uint32_t abi_version;
    uint32_t graph_active_sequence_count;
    void *cuda_graph_exec;
    uint64_t graph_capture_count;
    uint64_t graph_replay_count;
    uint64_t launch_chain_count;
    uint64_t launch_error_count;
} SparkGlm52ResidentDecodeStageCudaPipelineSlotState;

typedef struct SparkGlm52ResidentDecodeStagePipelineSlot
{
    void *cuda_stream;
    const void *input_hidden_bf16;
    void *normalized_hidden_bf16;
    void *query_latent_bf16;
    void *query_rope_input_bf16;
    void *key_rope_input_bf16;
    void *current_kv_latent_bf16;
    void *raw_query_a_bf16;
    void *raw_query_a_normalized_bf16;
    void *raw_query_b_bf16;
    void *raw_kv_a_bf16;
    void *raw_kv_a_normalized_bf16;
    void *raw_kv_b_bf16;
    const uint32_t *positions;
    const uint32_t *slot_mapping;
    const uint32_t *block_table;
    const uint32_t *context_lengths;
    const uint32_t *first_block_token_offsets;
    const float *dsa_token_scores;
    uint32_t *sparse_token_indices;
    void *rotated_query_rope_bf16;
    void *attention_output_latent_bf16;
    void *attention_projected_hidden_bf16;
    void *post_attention_hidden_bf16;
    void *post_attention_normalized_hidden_bf16;
    uint32_t *moe_topk_expert_ids;
    float *moe_topk_weights;
    void *moe_gate_bf16;
    void *moe_up_bf16;
    void *moe_intermediate_bf16;
    void *moe_route_output_bf16;
    void *layer_output_hidden_bf16;
    const void *mtp_draft_hidden_bf16;
    float *restricted_logits;
    float *mtp_draft_logits;
    uint32_t *restricted_selected_token_ids;
    float *restricted_selected_token_scores;
    uint32_t *mtp_draft_token_ids;
    const uint32_t *mtp_target_token_ids;
    uint32_t *mtp_accept_mask;
    uint32_t *mtp_committed_token_ids;
    uint32_t *mtp_event_counters;
    uint64_t *phase_clock_cycles;
} SparkGlm52ResidentDecodeStagePipelineSlot;

typedef struct SparkGlm52ResidentDecodeStageNodeContext
{
    uint32_t abi_version;
    uint32_t pipeline_slot_count;
    uint32_t max_active_sequence_count;
    uint32_t cache_token_capacity;
    uint32_t kv_block_count;
    uint32_t max_blocks_per_sequence;
    uint32_t position_count;
    uint32_t dsa_candidate_count;
    float qk_scale;
    float rms_norm_epsilon;
    const float *cos_table;
    const float *sin_table;
    void *mla_cache_bf16;
    void *key_nope_cache_bf16;
    void *value_cache_bf16;
    const void *attention_norm_weight_bf16;
    const void *query_latent_weight_bf16;
    const void *query_rope_weight_bf16;
    const void *key_rope_weight_bf16;
    const void *kv_latent_weight_bf16;
    const void *raw_query_a_weight_bf16;
    const void *raw_query_a_norm_weight_bf16;
    const void *raw_query_b_weight_bf16;
    const void *raw_kv_a_weight_bf16;
    const void *raw_kv_a_norm_weight_bf16;
    const void *raw_kv_b_weight_bf16;
    const uint8_t *raw_query_a_weight_fp8_e4m3;
    const float *raw_query_a_weight_scale_inv_f32;
    const uint8_t *raw_query_b_weight_fp8_e4m3;
    const float *raw_query_b_weight_scale_inv_f32;
    const uint8_t *raw_kv_a_weight_fp8_e4m3;
    const float *raw_kv_a_weight_scale_inv_f32;
    const uint8_t *raw_kv_b_weight_fp8_e4m3;
    const float *raw_kv_b_weight_scale_inv_f32;
    const void *attention_output_weight_bf16;
    const uint8_t *attention_output_weight_fp8_e4m3;
    const float *attention_output_weight_scale_inv_f32;
    const void *post_attention_norm_weight_bf16;
    const void *moe_gate_weight_bf16;
    const void *moe_up_weight_bf16;
    const void *moe_down_weight_bf16;
    const void *dense_gate_weight_bf16;
    const void *dense_up_weight_bf16;
    const void *dense_down_weight_bf16;
    const void *final_norm_weight_bf16;
    const void *restricted_lm_head_weight_bf16;
    const uint8_t *mtp_mxfp4_weight_payload_u8;
    const uint8_t *mtp_mxfp4_scale_e8m0_u8;
    const uint32_t *restricted_token_ids;
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slots;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_pipeline_slot_states;
    uint32_t projection_mode;
    uint32_t layer_progression_mode;
    uint32_t moe_expert_count;
    uint32_t moe_first_bound_expert_id;
    uint32_t moe_bound_expert_count;
    uint32_t moe_top_k;
    uint32_t moe_intermediate_dimension;
    uint32_t dense_intermediate_dimension;
    uint32_t sparse_index_mode;
    uint32_t launch_check_mode;
    uint32_t phase_clock_mode;
    uint32_t enable_cuda_graph_replay;
    const void *moe_router_weight_bf16;
    const float *moe_router_score_bias_f32;
    float moe_routed_scaling_factor;
    uint32_t moe_norm_topk_prob;
} SparkGlm52ResidentDecodeStageNodeContext;

SparkStatus SparkGlm52ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);

SparkStatus SparkGlm52ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame);

SparkStatus SparkGlm52ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

SparkStatus SparkGlm52ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot);

void SparkGlm52ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
