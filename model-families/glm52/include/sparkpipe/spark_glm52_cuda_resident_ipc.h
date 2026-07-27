#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_ring_node_context_builder.h"
#include "sparkpipe/spark_glm52_ring_work_control.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_CUDA_RESIDENT_IPC_ABI_VERSION 23u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_MAGIC 0x52445543u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS \
    (SPARK_GLM52_KV_CONTEXT_TOKENS / SPARK_GLM52_KV_BLOCK_TOKENS)
#define SPARK_GLM52_CUDA_RESIDENT_IPC_HEADER_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcHeader))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_HELLO_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcHello))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES \
    ((uint32_t)sizeof(SparkGlm52CudaResidentIpcSubmitWork))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES \
	((uint32_t)offsetof(SparkGlm52CudaResidentIpcSubmitWork,work_packet))
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
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_PREFILL_PREFIX_BYTES \
	((uint32_t)offsetof(SparkGlm52CudaResidentIpcSubmitPrefill,work_packet))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_BYTES \
    SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES \
    ((uint32_t)offsetof(SparkGlm52CudaResidentIpcSubmitDecode, \
        kv_physical_block_indices))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES \
    (SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES + \
     (SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT * \
      SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS * (uint32_t)sizeof(uint32_t)))
#define SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES \
    SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES

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

#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY \
    0x00000001u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_KNOWN_FLAGS \
    SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT \
    0x00000001u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_KNOWN_FLAGS \
    SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT

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

typedef struct SparkGlm52CudaResidentIpcReader
{
    SparkGlm52CudaResidentIpcHeader header;
    uint32_t header_offset;
    uint32_t payload_offset;
    uint32_t header_ready;
} SparkGlm52CudaResidentIpcReader;

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
    uint32_t flags;
    SparkGlm52RingWorkControlPacket work_packet;
} SparkGlm52CudaResidentIpcSubmitWork;

typedef struct SparkGlm52CudaResidentIpcCompletion
{
    uint32_t descriptor_bytes;
    uint32_t flags;
    SparkModelDriverCompletion completion;
	SparkGlm52DsparkDraftResult dspark_draft;
} SparkGlm52CudaResidentIpcCompletion;

#define SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT 0x00000001u
#define SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_KNOWN_FLAGS \
	SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT

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
    uint32_t kv_nvme_enabled;
    uint32_t kv_physical_block_capacity;
    uint32_t kv_logical_block_capacity;
    uint32_t kv_nvme_mode;
    uint64_t kv_logical_block_count;
    uint64_t kv_resident_block_count;
    uint64_t kv_swapped_block_count;
    uint64_t kv_nvme_record_bytes;
    uint64_t kv_nvme_store_count;
    uint64_t kv_nvme_load_count;
    uint64_t kv_nvme_write_bytes;
    uint64_t kv_nvme_read_bytes;
    uint64_t kv_nvme_synchronous_wait_count;
    uint64_t kv_nvme_batch_flush_count;
    uint64_t kv_nvme_maximum_batch_operation_count;
    uint64_t kv_resident_bytes_per_token;
    uint64_t kv_resident_pool_bytes;
    uint64_t kv_nvme_capacity_bytes;
    uint64_t kv_compact_selected_mla_working_set_bytes;
    uint32_t kv_nvme_batch_block_capacity;
    uint32_t kv_nvme_pending_store_count;
    uint32_t kv_nvme_pending_load_count;
    uint32_t kv_nvme_clean_evict_count;
    uint32_t work_queue_depth;
    uint32_t work_queue_capacity;
    uint32_t builder_pending_work;
    uint32_t resident_driver_inflight;
    uint64_t work_queue_accepted_count;
    uint64_t work_queue_submit_count;
    uint64_t work_queue_error_count;
    uint64_t asynchronous_submit_count;
    uint64_t asynchronous_completion_count;
    uint64_t asynchronous_failure_count;
    uint32_t logical_lane_capacity;
    uint32_t execution_row_capacity;
    uint32_t last_layer_major_logical_lane_count;
    uint32_t last_layer_major_rows_per_lane;
    uint32_t last_layer_major_execution_row_count;
    uint32_t moe_backend_kind;
    uint32_t moe_bound_layer_count;
    uint32_t moe_expected_layer_count;
    uint32_t fp8_scaled_gemm_bound_plan_count;
    uint32_t fp8_scaled_gemm_expected_plan_count;
    uint32_t model_quantization_mode;
    uint64_t layer_major_submit_count;
    uint64_t layer_major_completion_count;
    uint64_t layer_major_failure_count;
	uint64_t cuda_total_bytes;
	uint64_t cuda_initial_free_bytes;
	uint64_t cuda_current_free_bytes;
	uint64_t cuda_consumed_bytes;
	uint64_t cuda_builder_allocation_bytes;
	uint64_t cuda_largest_allocation_bytes;
	uint64_t host_mapped_allocation_bytes;
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
    uint32_t request_flags;
    SparkGlm52RingWorkControlPacket work_packet;
} SparkGlm52CudaResidentIpcSubmitPrefill;

typedef struct SparkGlm52CudaResidentIpcDecodeLane
{
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint32_t request_slot_index;
    uint32_t context_token_count;
    uint32_t input_token_id;
    uint32_t mtp_draft_token_budget;
    uint32_t speculative_token_count;
    uint8_t mtp_resolution_proposed_token_count;
    uint8_t mtp_resolution_accepted_token_count;
    uint16_t mtp_resolution_path_id;
    uint32_t kv_block_offset;
    uint32_t kv_block_count;
    uint32_t speculative_draft_token_ids[
        SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkGlm52CudaResidentIpcDecodeLane;

typedef struct SparkGlm52CudaResidentIpcSubmitDecode
{
    uint32_t descriptor_bytes;
    uint32_t highest_priority;
    uint32_t request_flags;
    uint32_t dispatch_kind;
    uint32_t lane_count;
    uint32_t active_sequence_count;
    uint32_t execution_batch_bucket;
    uint32_t speculative_token_count;
    uint32_t kv_block_token_count;
    uint32_t kv_block_index_count;
    uint32_t resident_flags;
    uint64_t control_generation;
    SparkGlm52CudaResidentIpcDecodeLane
        lanes[SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT];
    uint32_t kv_physical_block_indices[];
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
void SparkGlm52CudaResidentIpcReaderReset(
    SparkGlm52CudaResidentIpcReader *reader);
SparkStatus SparkGlm52CudaResidentIpcReadHeader(
    SparkGlm52CudaResidentIpcReader *reader,
    int32_t fd,
    uint32_t maximum_payload_bytes);
SparkStatus SparkGlm52CudaResidentIpcReadPayload(
    SparkGlm52CudaResidentIpcReader *reader,
    int32_t fd,
    uint8_t *payload,
    uint32_t payload_capacity);
uint32_t SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(
	const SparkGlm52RingWorkControlPacket *work_packet);
SparkStatus SparkGlm52CudaResidentIpcInitializeSubmitWork(
    SparkGlm52CudaResidentIpcSubmitWork *message,
    const SparkGlm52RingWorkControlPacket *work_packet,
    uint32_t flags);
SparkStatus SparkGlm52CudaResidentIpcValidateSubmitWork(
    const SparkGlm52CudaResidentIpcSubmitWork *message,
    uint32_t payload_bytes);
uint32_t SparkGlm52CudaResidentIpcCalculateSubmitPrefillBytes(
	const SparkGlm52RingWorkControlPacket *work_packet);
SparkStatus SparkGlm52CudaResidentIpcDecodePayloadBytes(
    uint32_t kv_block_index_count,
    uint32_t *payload_bytes_out);
SparkStatus SparkGlm52CudaResidentIpcValidateSubmitDecode(
    const SparkGlm52CudaResidentIpcSubmitDecode *message,
    uint32_t payload_bytes,
    uint32_t maximum_lane_count);

#ifdef __cplusplus
}
#endif
