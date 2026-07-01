#ifndef SPARKPIPE_SPARK_GLM52_STAGE_PLAN_H
#define SPARKPIPE_SPARK_GLM52_STAGE_PLAN_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_STAGE_PLAN_ABI_VERSION 1u
#define SPARK_GLM52_STAGE_PLAN_LAYER_COUNT 78u
#define SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER 3u
#define SPARK_GLM52_STAGE_PLAN_ROUTED_LAYER_COUNT \
    (SPARK_GLM52_STAGE_PLAN_LAYER_COUNT - \
     SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER)
#define SPARK_GLM52_STAGE_PLAN_MAX_ROUTED_LAYERS_PER_STAGE 8u
#define SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT 13u
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

#ifdef __cplusplus
}
#endif

#endif
