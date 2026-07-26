#ifndef SPARKPIPE_SPARK_GLM52_SCHEDULER_H
#define SPARKPIPE_SPARK_GLM52_SCHEDULER_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_prefix_cache.h"
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
#define SPARK_GLM52_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerBatchRequest))
#define SPARK_GLM52_SCHEDULER_PREFILL_BATCH_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerPrefillBatchRequest))
#define SPARK_GLM52_SCHEDULER_PACKED_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerPackedRequest))
#define SPARK_GLM52_SCHEDULER_PREFILL_BATCH_LANE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerPrefillBatchLane))
#define SPARK_GLM52_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerBatchDecision))
#define SPARK_GLM52_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52SchedulerPrefillBatchDecision))
#define SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT \
    SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT \
    SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET
#define SPARK_GLM52_SCHEDULER_BATCH_REQUEST_CAPACITY_FACTOR 2u
#define SPARK_GLM52_SCHEDULER_MAX_BATCH_REQUEST_COUNT \
    (SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET * \
     SPARK_GLM52_SCHEDULER_BATCH_REQUEST_CAPACITY_FACTOR)
#define SPARK_GLM52_SCHEDULER_DEFAULT_QUEUE_DEPTH_PER_SPARK 1u
#define SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS 16u
#define SPARK_GLM52_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP \
    SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
#define SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS \
    SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY \
    (SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS / \
     SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS)

#define SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL 0x00000001u
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE 0x00000002u
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CUDAGRAPH_PADDING 0x00000004u
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE 0x00000008u
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_KV_CACHE_REQUIRED 0x00000010u
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION 0x00000020u
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE 0x00000040u
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS \
    (SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL | \
     SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE | \
     SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CUDAGRAPH_PADDING | \
     SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE | \
     SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_KV_CACHE_REQUIRED | \
     SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION | \
     SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE)
#define SPARK_GLM52_SCHEDULER_CONFIGURATION_KNOWN_FLAGS \
    SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS

#define SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE 0x00000001u
#define SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL 0x00000002u
#define SPARK_GLM52_SCHEDULER_REQUEST_KNOWN_FLAGS \
    (SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE | \
     SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL)

#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK 0x00000001u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK 0x00000002u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED 0x00000004u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING 0x00000008u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_STEP 0x00000010u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_STEP 0x00000020u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT 0x00000040u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL 0x00000080u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK 0x00000100u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK 0x00000200u
#define SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET 0x00000400u

#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE 0x00000001u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL 0x00000002u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_CHUNK 0x00000004u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_FINAL_CHUNK 0x00000008u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_CUDAGRAPH_PADDING 0x00000010u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_RESERVED_DECODE_SLOT 0x00000020u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL 0x00000040u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK 0x00000080u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_PREFILL_PACK 0x00000100u
#define SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET 0x00000200u

#define SPARK_GLM52_SCHEDULER_NO_PROMPT_LIMIT 0u

typedef struct SparkGlm52SchedulerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t spark_count;
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;
    uint32_t max_prefill_tokens_per_step;
    uint32_t prefix_cache_block_tokens;
    uint32_t configuration_flags;
    uint32_t reserved;
    SparkGlm52PrefixCache *prefix_cache;
} SparkGlm52SchedulerConfiguration;

typedef struct SparkGlm52SchedulerRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t active_sequence_count;
    uint32_t prompt_token_count;
    uint32_t flags;
    uint32_t computed_prompt_token_count;
    uint32_t cached_prefix_token_count;
    uint32_t max_scheduled_prompt_token_count;
    uint32_t reserved;
    uint64_t sequence_id;
    const uint32_t *prompt_token_ids;
} SparkGlm52SchedulerRequest;

typedef struct SparkGlm52SchedulerBatchRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_count;
    uint32_t reserved;
    const SparkGlm52SchedulerRequest *requests;
} SparkGlm52SchedulerBatchRequest;

typedef struct SparkGlm52SchedulerPrefillBatchRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_count;
    uint32_t reserved;
    const SparkGlm52SchedulerRequest *requests;
} SparkGlm52SchedulerPrefillBatchRequest;

typedef struct SparkGlm52SchedulerPackedRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_index;
    uint32_t active_sequence_offset;
    uint32_t active_sequence_count;
    uint32_t request_flags;
    uint32_t scheduled_token_count;
    uint32_t reserved;
    uint64_t total_scheduled_token_count;
} SparkGlm52SchedulerPackedRequest;

typedef struct SparkGlm52SchedulerDispatchStage
{
    uint32_t spark_index;
    uint32_t batch_bucket;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t stage_flags;
    uint32_t dispatch_flags;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t scheduled_prompt_token_offset;
    uint32_t scheduled_prompt_token_count;
    uint32_t cached_prefix_token_count;
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
    uint32_t decision_flags;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t prompt_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t cached_prefix_token_count;
    uint32_t prefix_cache_block_count;
    uint32_t scheduled_prompt_token_offset;
    uint32_t scheduled_prompt_token_count;
    uint32_t remaining_prompt_token_count_after_step;
    uint32_t prefill_block_count;
    uint32_t cache_commit_token_count_after_step;
    uint32_t kv_block_token_count;
    uint32_t kv_physical_block_count;
    uint32_t kv_cached_physical_block_count;
    uint32_t kv_pending_physical_block_count;
    uint32_t kv_block_table_token_count;
    uint64_t prefix_cache_reservation_epoch;
    uint64_t prefix_cache_parent_hash;
    uint64_t prefix_cache_result_hash;
    uint64_t sequence_id;
    const uint32_t *prompt_token_ids;
    uint64_t total_scheduled_token_count;
    uint64_t estimated_critical_path_ns;
    uint32_t kv_physical_block_indices[
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
    SparkGlm52StagePlan stage_plan;
    SparkGlm52SchedulerDispatchStage dispatch_stages[
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT];
} SparkGlm52SchedulerDecision;

typedef struct SparkGlm52SchedulerPrefillBatchLane
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_index;
    uint32_t active_sequence_offset;
    uint32_t active_sequence_count;
    uint32_t prompt_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t cached_prefix_token_count;
    uint32_t scheduled_prompt_token_offset;
    uint32_t scheduled_prompt_token_count;
    uint32_t remaining_prompt_token_count_after_step;
    uint32_t cache_commit_token_count_after_step;
    uint32_t prefix_cache_block_count;
    uint32_t kv_block_token_count;
    uint32_t kv_physical_block_count;
    uint32_t kv_cached_physical_block_count;
    uint32_t kv_pending_physical_block_count;
    uint32_t kv_block_table_token_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t prefix_cache_reservation_epoch;
    uint64_t prefix_cache_parent_hash;
    uint64_t prefix_cache_result_hash;
    uint64_t sequence_id;
    const uint32_t *prompt_token_ids;
} SparkGlm52SchedulerPrefillBatchLane;

typedef struct SparkGlm52SchedulerBatchDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t rejected_status;
    uint32_t source_request_count;
    uint32_t packed_request_count;
    uint32_t batch_bucket;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t decision_flags;
    uint32_t reserved;
    uint64_t total_scheduled_token_count;
    uint64_t estimated_critical_path_ns;
    SparkGlm52SchedulerDecision stage_decision;
    SparkGlm52SchedulerPackedRequest packed_requests[
        SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT];
} SparkGlm52SchedulerBatchDecision;

typedef struct SparkGlm52SchedulerPrefillBatchDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t rejected_status;
    uint32_t source_request_count;
    uint32_t packed_request_count;
    uint32_t batch_bucket;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t decision_flags;
    uint32_t maximum_scheduled_prompt_token_count;
    uint64_t total_scheduled_token_count;
    uint64_t estimated_critical_path_ns;
    SparkGlm52SchedulerDecision stage_decision;
    SparkGlm52SchedulerPrefillBatchLane lanes[
        SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT];
} SparkGlm52SchedulerPrefillBatchDecision;

typedef struct SparkGlm52Scheduler
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t spark_count;
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;
    uint32_t max_prefill_tokens_per_step;
    uint32_t prefix_cache_block_tokens;
    uint32_t configuration_flags;
    uint32_t reserved;
    SparkGlm52PrefixCache *prefix_cache;
    uint32_t spark_inflight_counts[SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT];
    uint32_t prefill_demand;
    uint64_t admitted_count;
    uint64_t rejected_count;
    uint64_t completed_count;
    uint64_t scheduled_decode_token_count;
    uint64_t scheduled_prefill_token_count;
    uint64_t prefix_cache_hit_token_count;
    uint64_t chunked_prefill_count;
    uint64_t interleaved_prefill_admission_count;
    uint64_t decode_bypass_admission_count;
    uint64_t adaptive_decode_pack_admission_count;
    uint64_t adaptive_decode_pack_request_count;
    uint64_t adaptive_decode_pack_padding_token_count;
    uint64_t adaptive_prefill_pack_admission_count;
    uint64_t adaptive_prefill_pack_request_count;
    uint64_t adaptive_prefill_pack_padding_token_count;
    uint64_t measured_decode_bucket_selection_count;
    uint64_t measured_decode_bucket_padding_token_count;
    uint64_t kv_block_reservation_count;
    uint64_t kv_block_reservation_token_count;
    uint64_t kv_block_cancel_count;
} SparkGlm52Scheduler;

SparkStatus SparkGlm52SchedulerInitialize(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerConfiguration *configuration);

uint32_t SparkGlm52SchedulerSelectPipelineBatchWidth(
    const SparkGlm52Scheduler *scheduler,
    uint32_t active_request_count,
    uint32_t batch_capacity);

SparkStatus SparkGlm52SchedulerEstimateDecodeWorkNs(
    const SparkGlm52Scheduler *scheduler,
    uint32_t logical_sequence_count,
    uint32_t rows_per_sequence,
    uint32_t execution_row_capacity,
    uint64_t *estimated_work_ns_out);

void SparkGlm52SchedulerSetPrefillDemand(
    SparkGlm52Scheduler *scheduler,
    uint32_t prefill_demand);
SparkStatus SparkGlm52SchedulerAdmit(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request,
    SparkGlm52SchedulerDecision *decision);

SparkStatus SparkGlm52SchedulerComplete(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision);

SparkStatus SparkGlm52SchedulerAdmitDecodeBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerBatchRequest *batch_request,
    SparkGlm52SchedulerBatchDecision *batch_decision);

SparkStatus SparkGlm52SchedulerAdmitPrefillBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchRequest *batch_request,
    SparkGlm52SchedulerPrefillBatchDecision *batch_decision);

SparkStatus SparkGlm52SchedulerCompleteDecodeBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerBatchDecision *batch_decision);

SparkStatus SparkGlm52SchedulerCancelDecodeBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerBatchDecision *batch_decision);

SparkStatus SparkGlm52SchedulerCompletePrefillBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchDecision *batch_decision);

SparkStatus SparkGlm52SchedulerCancel(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision);

SparkStatus SparkGlm52SchedulerCancelPrefillBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchDecision *batch_decision);

SparkStatus SparkGlm52SchedulerBuildKvBlockTable(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *physical_block_count_out);

SparkStatus SparkGlm52SchedulerBuildPrefillBatchKvBlockTables(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchDecision *batch_decision,
    uint32_t *physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity);

SparkStatus SparkGlm52SchedulerReleaseSequence(
    SparkGlm52Scheduler *scheduler,
    uint64_t sequence_id);

#ifdef __cplusplus
}
#endif

#endif
