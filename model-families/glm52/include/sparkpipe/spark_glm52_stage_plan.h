#ifndef SPARKPIPE_SPARK_GLM52_STAGE_PLAN_H
#define SPARKPIPE_SPARK_GLM52_STAGE_PLAN_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_STAGE_PLAN_ABI_VERSION 1u
#define SPARK_GLM52_STAGE_PLAN_LAYER_COUNT SPARK_GLM52_MODEL_LAYER_COUNT
#define SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER \
    SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER
#define SPARK_GLM52_STAGE_PLAN_ROUTED_LAYER_COUNT \
    (SPARK_GLM52_STAGE_PLAN_LAYER_COUNT - \
     SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER)
#define SPARK_GLM52_STAGE_PLAN_MAX_ROUTED_LAYERS_PER_STAGE 8u
#define SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT 13u
#define SPARK_GLM52_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY \
    (SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT * \
     SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET)
#define SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT \
    SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_GLM52_STAGE_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52StagePlan))
#define SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701 20260701u

#define SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO 0u
#define SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT 1u
#define SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT 2u

#define SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN 0x00000001u
#define SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN 0x00000002u
#define SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN 0x00000004u
#define SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX 0x00000008u
#define SPARK_GLM52_STAGE_PLAN_STAGE_KNOWN_FLAGS \
    (SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN | \
     SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN | \
     SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN | \
     SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX)

#define SPARK_GLM52_STAGE_PLAN_BUCKET_B16 16u
#define SPARK_GLM52_STAGE_PLAN_BUCKET_B32 32u
#define SPARK_GLM52_STAGE_PLAN_BUCKET_B64 64u
#define SPARK_GLM52_STAGE_PLAN_BUCKET_B128 128u
#define SPARK_GLM52_STAGE_PLAN_BUCKET_B256 256u
#define SPARK_GLM52_STAGE_PLAN_BUCKET_B512 512u
#define SPARK_GLM52_STAGE_PLAN_BUCKET_B1024 1024u
#define SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B1024
#define SPARK_GLM52_STAGE_PLAN_BATCH_BUCKETS \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B16, \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B32, \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B64, \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B128, \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B256, \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B512, \
    SPARK_GLM52_STAGE_PLAN_BUCKET_B1024

static inline uint32_t SparkGlm52StagePlanBatchBucketIsSupported(
    uint32_t batch_bucket)
{
    if (batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B16 ||
        batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B32 ||
        batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64 ||
        batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B128 ||
        batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B256 ||
        batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B512 ||
        batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B1024)
    {
        return 1u;
    }
    return 0u;
}

static inline uint32_t SparkGlm52StagePlanSelectBatchBucketValue(
    uint32_t active_sequence_count)
{
    if (active_sequence_count == 0u ||
        active_sequence_count > SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET)
    {
        return 0u;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B16)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B32)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B32;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B64)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B64;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B128)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B128;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B256)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B256;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B512)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B512;
    }
    return SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
}

typedef struct SparkGlm52StagePlanStage
{
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t flags;
    uint32_t reserved;
} SparkGlm52StagePlanStage;

typedef struct SparkGlm52StagePlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t stage_count;
    uint32_t reserved;
    SparkGlm52StagePlanStage stages[SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT];
} SparkGlm52StagePlan;

SparkStatus SparkGlm52StagePlanValidate(
    const SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanBuildFromLayerCounts(
    const uint32_t *layer_counts,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanBuildUniform(
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanBuildBalanced(
    const uint64_t *layer_cost_ns,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanBuildBalancedWithFinalCost(
    const uint64_t *layer_cost_ns,
    uint64_t final_stage_extra_cost_ns,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanLoadMeasuredCostProfile(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out);

SparkStatus SparkGlm52StagePlanLoadMeasuredCostProfileForQuantization(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out);

SparkStatus SparkGlm52StagePlanBuildMeasuredBalanced(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanBuildMeasuredBalancedForQuantization(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanBuildCurrentSparkMeasuredBalanced(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52StagePlanSelectBatchBucket(
    uint32_t active_sequence_count,
    uint32_t *bucket_out);

SparkStatus SparkGlm52StagePlanExecutionChunkShape(
    uint32_t logical_sequence_count,
    uint32_t rows_per_sequence,
    uint32_t execution_row_capacity,
    uint32_t *maximum_sequences_per_chunk_out,
    uint32_t *chunk_count_out);

#ifdef __cplusplus
}
#endif

#endif
