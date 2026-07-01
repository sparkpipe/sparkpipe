#ifndef SPARKPIPE_SPARK_GLM52_SCHEDULER_H
#define SPARKPIPE_SPARK_GLM52_SCHEDULER_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_stage_plan.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SCHEDULER_ABI_VERSION 1u
#define SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52Scheduler))
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerConfiguration))
#define SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerRequest))
#define SPARK_GLM52_SCHEDULER_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerDecision))
#define SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT \
    SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_GLM52_SCHEDULER_DEFAULT_QUEUE_DEPTH_PER_SPARK 1u
#define SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS 16u

#define SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE 0x00000001u
#define SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL 0x00000002u
#define SPARK_GLM52_SCHEDULER_REQUEST_KNOWN_FLAGS \
    (SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE | \
     SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL)

typedef struct SparkGlm52SchedulerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t spark_count;
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;
} SparkGlm52SchedulerConfiguration;

typedef struct SparkGlm52SchedulerRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t active_sequence_count;
    uint32_t prompt_token_count;
    uint32_t flags;
    uint32_t reserved;
} SparkGlm52SchedulerRequest;

typedef struct SparkGlm52SchedulerDispatchStage
{
    uint32_t spark_index;
    uint32_t batch_bucket;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t stage_flags;
    uint32_t reserved;
    uint64_t estimated_service_time_ns;
} SparkGlm52SchedulerDispatchStage;

typedef struct SparkGlm52SchedulerDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t batch_bucket;
    uint32_t quantization_mode;
    uint32_t spark_count;
    uint32_t stage_count;
    uint32_t rejected_status;
    uint64_t estimated_critical_path_ns;
    SparkGlm52StagePlan stage_plan;
    SparkGlm52SchedulerDispatchStage dispatch_stages[
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT];
} SparkGlm52SchedulerDecision;

typedef struct SparkGlm52Scheduler
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t spark_count;
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t spark_inflight_counts[SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT];
    uint64_t admitted_count;
    uint64_t rejected_count;
    uint64_t completed_count;
} SparkGlm52Scheduler;

SparkStatus SparkGlm52SchedulerInitialize(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerConfiguration *configuration);

SparkStatus SparkGlm52SchedulerAdmit(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request,
    SparkGlm52SchedulerDecision *decision);

SparkStatus SparkGlm52SchedulerComplete(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision);

#ifdef __cplusplus
}
#endif

#endif
