#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION 1u
#define SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC 0x35574350u
#define SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES \
	((uint32_t)sizeof(SparkGlm52Pp13WorkControlPacket))
#define SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES \
	((uint32_t)sizeof(SparkGlm52Pp13WorkControlKvState))

#define SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL 0x00000001u
#define SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP 0x00000002u
#define SPARK_GLM52_PP13_WORK_CONTROL_KNOWN_FLAGS \
	(SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL | \
	 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP)

typedef struct SparkGlm52Pp13WorkControlPacket
{
	uint32_t magic;
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t deadline_time_ns;
	uint32_t active_sequence_count;
	uint32_t new_token_count;
	uint32_t pipeline_slot;
	uint32_t priority;
	uint32_t block_token_count;
	uint32_t kv_block_table_token_count;
	uint32_t max_blocks_per_sequence;
	uint32_t mtp_draft_token_count;
} SparkGlm52Pp13WorkControlPacket;

typedef struct SparkGlm52Pp13WorkControlKvState
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_capacity;
	uint32_t lane_stride;
	uint32_t block_token_count;
	uint32_t physical_block_capacity;
	uint32_t *physical_block_indices;
	uint32_t *lane_physical_block_counts;
} SparkGlm52Pp13WorkControlKvState;

SparkStatus SparkGlm52Pp13WorkControlValidatePacket(
	const SparkGlm52Pp13WorkControlPacket *packet,
	uint32_t max_active_sequence_count,
	uint32_t max_pipeline_slot_count);
SparkStatus SparkGlm52Pp13WorkControlInitializeKvState(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t lane_capacity,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t *physical_block_indices,
	uint32_t *lane_physical_block_counts);
uint32_t SparkGlm52Pp13WorkControlBlockCount(
	uint32_t token_count,
	uint32_t block_token_count);
SparkStatus SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state,
	SparkGlm52KvBlockTableView *view);

#ifdef __cplusplus
}
#endif
