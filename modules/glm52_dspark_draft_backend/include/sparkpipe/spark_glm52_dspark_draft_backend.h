#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSPARK_DRAFT_BACKEND_ABI_VERSION 3u
#define SPARK_DSPARK_DRAFT_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkDraftBackendConfiguration))
#define SPARK_DSPARK_DRAFT_BACKEND_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkDraftBackend))
#define SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT 1024u
#define SPARK_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT 64u
#define SPARK_DSPARK_DRAFT_BACKEND_ATTENTION_DIMENSION \
    (SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT * \
     SPARK_DSPARK_DRAFT_HEAD_DIMENSION)
#define SPARK_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION \
    (SPARK_DSPARK_AUX_LAYER_COUNT * SPARK_DSPARK_HIDDEN_DIMENSION)
#define SPARK_DSPARK_DRAFT_BACKEND_MAX_TAP_ROW_COUNT \
    (SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT * \
     SPARK_DSPARK_BLOCK_SIZE)
#define SPARK_DSPARK_DRAFT_BACKEND_PENDING_NONE 0u
#define SPARK_DSPARK_DRAFT_BACKEND_PENDING_STAGE 1u
#define SPARK_DSPARK_DRAFT_BACKEND_PENDING_DRAFT 2u

typedef struct SparkGlm52DsparkDraftBackendConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t maximum_lane_count;
    uint32_t maximum_context_token_count;
    uint32_t restricted_vocabulary_count;
    const uint32_t *restricted_token_ids;
    const char *manifest_path;
    const char *config_path;
    const char *safetensors_path;
    void *cuda_stream;
} SparkGlm52DsparkDraftBackendConfiguration;

typedef struct SparkGlm52DsparkDraftBackendLaneState
{
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t tap_generation;
    uint32_t last_token_id;
    uint32_t context_token_count;
    uint32_t staged;
    uint32_t reserved;
} SparkGlm52DsparkDraftBackendLaneState;

typedef struct SparkGlm52DsparkDraftBackendStage
{
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t tap_generation;
    uint32_t tap_row_index;
    uint32_t backend_lane_index;
    uint32_t token_id;
    uint32_t reserved;
} SparkGlm52DsparkDraftBackendStage;

typedef struct SparkGlm52DsparkDraftBackend
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t maximum_lane_count;
    uint32_t maximum_context_token_count;
    uint32_t restricted_vocabulary_count;
    uint32_t owns_cuda_stream;
    uint32_t weight_count;
    uint32_t maximum_tap_row_count;
    uint32_t pending_operation_kind;
    uint32_t pending_draft_lane_count;
    SparkGlm52DsparkModelContract contract;
    void *cuda_stream;
    /* Shared-GEMM stack state (runtime/gemm.cuh): a two-word device scratch
     * the dense launcher requires for its group-row offset, and the SM count
     * captured once at Initialize so launches need no per-call device query.
     * This slot replaced the cuBLAS handle: the drafter's GEMMs are plain
     * dense BF16 projections and must not carry a second GEMM stack. */
    void *gemm_group_scratch;
    uint32_t multiprocessor_count;
    void *device_weights[SPARK_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT];
    uint32_t *device_restricted_token_ids;
    uint16_t *device_tap_arena_bf16;
    uint64_t tap_arena_lane_stride_bytes;
    uint16_t *device_stage_tap_bf16;
    uint16_t *device_target_hidden_bf16;
    uint16_t *device_context_key_bf16;
    uint16_t *device_context_value_bf16;
    uint16_t *device_block_hidden_bf16;
    uint16_t *device_block_normed_bf16;
    uint16_t *device_block_attention_bf16;
    uint16_t *device_block_query_bf16;
    uint16_t *device_block_key_bf16;
    uint16_t *device_block_value_bf16;
    /* Gate and up projections share one buffer, interleaved per row (gate
     * first): that is the layout LmSiluMulKernel consumes, so the two MLP
     * GEMMs write their columns in place and no repack pass exists. */
    uint16_t *device_block_gate_up_bf16;
    uint16_t *device_block_mlp_bf16;
    uint16_t *device_block_final_bf16;
    uint16_t *device_block_logits_bf16;
    uint16_t *device_markov_logits_bf16;
    uint32_t *device_argmax_u32;
    float *device_confidence_f32;
    uint32_t *device_tap_row_indices;
    uint32_t *device_backend_lane_indices;
    uint32_t *device_sequence_positions;
    uint32_t *device_context_token_counts;
    uint32_t *device_last_token_ids;
    uint32_t *host_argmax_u32;
    float *host_confidence_f32;
    void *completion_event;
    uint32_t host_tap_row_indices[
        SPARK_DSPARK_DRAFT_BACKEND_MAX_TAP_ROW_COUNT];
    uint32_t host_backend_lane_indices[
        SPARK_DSPARK_DRAFT_BACKEND_MAX_TAP_ROW_COUNT];
    uint32_t host_sequence_positions[
        SPARK_DSPARK_DRAFT_BACKEND_MAX_TAP_ROW_COUNT];
    uint32_t host_context_token_counts[
        SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT];
    uint32_t host_last_token_ids[
        SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT];
    uint32_t host_requested_token_counts[
        SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT];
    SparkGlm52DsparkDraftBackendLaneState
        lane_states[SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT];
    SparkGlm52DsparkDraftBackendLaneState
        validation_lane_states[SPARK_DSPARK_DRAFT_BACKEND_MAX_LANE_COUNT];
} SparkGlm52DsparkDraftBackend;

SparkStatus SparkGlm52DsparkDraftBackendInitialize(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendConfiguration *configuration);

SparkStatus SparkGlm52DsparkDraftBackendTeardown(
    SparkGlm52DsparkDraftBackend *backend);

SparkStatus SparkGlm52DsparkDraftBackendModelContract(
    const SparkGlm52DsparkDraftBackend *backend,
    SparkGlm52DsparkModelContract *contract_out);

SparkStatus SparkGlm52DsparkDraftBackendTapOutputPointers(
    SparkGlm52DsparkDraftBackend *backend,
    uint32_t lane_index,
    void *tap_output_bf16[SPARK_DSPARK_AUX_LAYER_COUNT],
    uint64_t *lane_stride_bytes_out);

SparkStatus SparkGlm52DsparkDraftBackendStageBatch(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftBackendStage *stages,
    uint32_t stage_count);

SparkStatus SparkGlm52DsparkDraftBackendLaunchDraftBatch(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftRequest *requests,
    uint32_t lane_count);

SparkStatus SparkGlm52DsparkDraftBackendTakeBatchResults(
    SparkGlm52DsparkDraftBackend *backend,
    SparkGlm52DsparkDraftResult *results,
    uint32_t result_capacity,
    uint32_t *result_count);

/* Forget a lane's staged context (sequence ended / resident slot reassigned).
 * The next StageBatch for that lane must start at sequence_position 1; the
 * drafter's context K/V beyond the reset counter is never read. */
SparkStatus SparkGlm52DsparkDraftBackendResetLanes(
    SparkGlm52DsparkDraftBackend *backend,
    const uint32_t *lane_indices,
    uint32_t lane_count);

#ifdef __cplusplus
}

#endif
