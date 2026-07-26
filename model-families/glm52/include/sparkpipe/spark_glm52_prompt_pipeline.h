#ifndef SPARKPIPE_SPARK_GLM52_PROMPT_PIPELINE_H
#define SPARKPIPE_SPARK_GLM52_PROMPT_PIPELINE_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_request_api.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION 1u
#define SPARK_GLM52_PROMPT_PIPELINE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PromptPipelineConfiguration))
#define SPARK_GLM52_PROMPT_PIPELINE_PREFILL_DISPATCH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PromptPipelinePrefillDispatch))
#define SPARK_GLM52_PROMPT_PIPELINE_RUN_STATS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52PromptPipelineRunStats))

#define SPARK_GLM52_PROMPT_PIPELINE_RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH \
    0x00000001u
#define SPARK_GLM52_PROMPT_PIPELINE_RUN_KNOWN_FLAGS \
    SPARK_GLM52_PROMPT_PIPELINE_RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH
#define SPARK_GLM52_PROMPT_PIPELINE_DEFAULT_MAX_DISPATCH_STEPS 4096u

typedef struct SparkGlm52PromptPipelinePrefillDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t step_index;
    uint32_t dispatch_kind;
    uint32_t active_sequence_count;
    uint32_t lane_count;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t prompt_token_stride;
    uint32_t host_token_stride;
    uint32_t reserved0;
    uint32_t reserved1;
    const SparkGlm52RequestApiDispatch *request_dispatch;
    const SparkGlm52RequestApiPrefillDispatchView *prefill_view;
    const uint32_t *host_token_ids;
    const SparkGlm52KvBlockTableView *kv_block_table_view;
} SparkGlm52PromptPipelinePrefillDispatch;

typedef SparkStatus (*SparkGlm52PromptPipelinePrefillFunction)(
    void *context,
    const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch);

typedef SparkStatus (*SparkGlm52PromptPipelineDecodeFunction)(
    void *context,
    const SparkGlm52RequestApiDispatch *decode_dispatch);

typedef struct SparkGlm52PromptPipelineConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t run_flags;
    uint32_t max_dispatch_steps;
    SparkGlm52RequestApi *request_api;
    uint32_t *host_prefill_token_ids;
    uint32_t host_prefill_token_stride;
    uint32_t host_prefill_lane_capacity;
    uint32_t *host_physical_block_indices;
    const uint32_t *execution_physical_block_indices;
    uint32_t kv_block_lane_stride;
    uint32_t kv_block_lane_capacity;
    uint32_t *lane_physical_block_counts;
    uint32_t lane_count_capacity;
    SparkGlm52PromptPipelinePrefillFunction prefill_function;
    SparkGlm52PromptPipelineDecodeFunction decode_function;
    void *callback_context;
} SparkGlm52PromptPipelineConfiguration;

typedef struct SparkGlm52PromptPipelineRunStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t completed_dispatch_count;
    uint32_t prefill_dispatch_count;
    uint32_t decode_dispatch_count;
    uint32_t speculative_verify_dispatch_count;
    uint32_t prefill_token_count;
    uint32_t maximum_prefill_token_count;
    uint32_t maximum_prefill_lane_count;
    uint32_t last_dispatch_kind;
    uint32_t reached_decode_dispatch;
    uint32_t reserved0;
} SparkGlm52PromptPipelineRunStats;

void SparkGlm52PromptPipelineInitializeRunStats(
    SparkGlm52PromptPipelineRunStats *stats);

SparkStatus SparkGlm52PromptPipelineRun(
    const SparkGlm52PromptPipelineConfiguration *configuration,
    SparkGlm52PromptPipelineRunStats *stats);

#ifdef __cplusplus
}
#endif

#endif
