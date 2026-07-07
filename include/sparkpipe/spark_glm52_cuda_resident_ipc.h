#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"
#include "sparkpipe/spark_glm52_pp13_work_control.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_CUDA_RESIDENT_IPC_ABI_VERSION 2u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_MAGIC 0x52445543u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS \
    (SPARK_GLM52_KV_CONTEXT_TOKENS / SPARK_GLM52_KV_BLOCK_TOKENS)
#define SPARK_GLM52_CUDA_RESIDENT_IPC_HEADER_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcHeader))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_HELLO_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcHello))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcSubmitWork))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcCompletion))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcSubmitResult))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_QUERY_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcQuery))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_STATS_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcStats))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_PREFILL_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcSubmitPrefill))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcSubmitDecode))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcAnyPayload))

#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO 1u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO_ACK 2u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK 3u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT 4u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_COMPLETION 5u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_QUERY 6u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_STATS 7u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SHUTDOWN 8u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_ERROR 9u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_PREFILL 10u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_DECODE 11u

#define SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_EMPTY 0u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_LOADING 1u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_READY 2u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_DRAINING 3u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_FAILED 4u

#define SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_DRIVER_RESIDENT 0x00000001u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_BUILDER_RESIDENT 0x00000002u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_TRANSPORT_RESIDENT 0x00000004u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_CUDA_STATE_RESIDENT 0x00000008u

#define SPARK_GLM52_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES 160u

typedef struct SparkGlm52CudaResidentIpcHeader
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t payload_bytes;
    uint32_t rank_index;
    uint64_t sequence_number;
} SparkGlm52CudaResidentIpcHeader;

typedef struct SparkGlm52CudaResidentIpcHello
{
    uint32_t descriptor_bytes;
    uint32_t rank_index;
    uint32_t rank_count;
    uint32_t expected_cuda_generation;
    uint64_t control_generation;
    uint64_t process_id;
} SparkGlm52CudaResidentIpcHello;

typedef struct SparkGlm52CudaResidentIpcSubmitWork
{
    uint32_t descriptor_bytes;
    uint32_t reserved0;
    SparkGlm52Pp13WorkControlPacket work_packet;
} SparkGlm52CudaResidentIpcSubmitWork;

typedef struct SparkGlm52CudaResidentIpcCompletion
{
    uint32_t descriptor_bytes;
    uint32_t reserved0;
    SparkModelDriverCompletion completion;
} SparkGlm52CudaResidentIpcCompletion;

typedef struct SparkGlm52CudaResidentIpcQuery
{
    uint32_t descriptor_bytes;
    uint32_t reserved0;
} SparkGlm52CudaResidentIpcQuery;

typedef struct SparkGlm52CudaResidentIpcStats
{
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t capability_flags;
    uint32_t rank_index;
    uint32_t max_active_sequence_count;
    uint32_t active_submission_count;
    uint32_t available_dispatch_slot_count;
    uint32_t private_queue_pressure;
    uint64_t submitted_count;
    uint64_t completed_count;
    uint64_t rejected_count;
    uint64_t resident_sequence_count;
    uint64_t resident_token_count;
    uint64_t cuda_generation;
    uint64_t control_generation;
    char blocker[SPARK_GLM52_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES];
} SparkGlm52CudaResidentIpcStats;

typedef struct SparkGlm52CudaResidentIpcSubmitResult
{
    uint32_t descriptor_bytes;
    uint32_t status;
    SparkGlm52CudaResidentIpcStats stats;
} SparkGlm52CudaResidentIpcSubmitResult;

typedef struct SparkGlm52CudaResidentIpcSubmitPrefill
{
    uint32_t descriptor_bytes;
    uint32_t highest_priority;
    uint64_t request_id;
    uint64_t sequence_id;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t kv_block_token_count;
    uint32_t kv_lane_block_count;
    uint32_t prompt_token_ids[SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS];
    uint32_t kv_physical_block_indices[SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS];
} SparkGlm52CudaResidentIpcSubmitPrefill;

typedef struct SparkGlm52CudaResidentIpcSubmitDecode
{
    uint32_t descriptor_bytes;
    uint32_t highest_priority;
    uint64_t request_id;
    uint64_t sequence_id;
    uint32_t request_flags;
    uint32_t mtp_draft_token_budget;
    uint32_t input_token_id;
    uint32_t sequence_position;
    uint32_t context_token_count;
    uint32_t kv_block_token_count;
    uint32_t kv_lane_block_count;
    uint32_t reserved0;
    uint32_t kv_physical_block_indices[SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS];
} SparkGlm52CudaResidentIpcSubmitDecode;

typedef union SparkGlm52CudaResidentIpcAnyPayload
{
    SparkGlm52CudaResidentIpcHello hello;
    SparkGlm52CudaResidentIpcSubmitWork submit_work;
    SparkGlm52CudaResidentIpcSubmitResult submit_result;
    SparkGlm52CudaResidentIpcCompletion completion;
    SparkGlm52CudaResidentIpcQuery query;
    SparkGlm52CudaResidentIpcStats stats;
    SparkGlm52CudaResidentIpcSubmitPrefill submit_prefill;
    SparkGlm52CudaResidentIpcSubmitDecode submit_decode;
} SparkGlm52CudaResidentIpcAnyPayload;

void SparkGlm52CudaResidentIpcInitializeHeader(
    SparkGlm52CudaResidentIpcHeader *header,
    uint32_t kind,
    uint32_t rank_index,
    uint64_t sequence_number,
    uint32_t payload_bytes);
SparkStatus SparkGlm52CudaResidentIpcValidateHeader(
    const SparkGlm52CudaResidentIpcHeader *header,
    uint32_t expected_kind,
    uint32_t maximum_payload_bytes);

#ifdef __cplusplus
}
#endif
