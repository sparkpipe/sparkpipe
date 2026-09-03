#ifndef SPARKPIPE_SPARK_QWEN38_MAX_WORK_CONTROL_H
#define SPARKPIPE_SPARK_QWEN38_MAX_WORK_CONTROL_H

#include <stdint.h>

#include "sparkpipe/spark_stage_kv_client.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QWEN38_MAX_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE 0u
#define SPARK_QWEN38_MAX_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE 1u

#define SPARK_QWEN38_MAX_WORK_CONTROL_BATCH_IDLE 0u
#define SPARK_QWEN38_MAX_WORK_CONTROL_BATCH_SUBMITTED 1u
#define SPARK_QWEN38_MAX_WORK_CONTROL_BATCH_READY 2u

typedef struct SparkQwen38MaxWorkControlKvPlanConfig
{
	uint64_t model_fingerprint;
	uint64_t cache_layout_fingerprint;
	uint32_t rank_index;
	uint32_t block_record_bytes;
	uint32_t gdn_record_bytes;
	uint32_t lookahead_packet_count;
	uint32_t physical_block_capacity;
	uint32_t allocated_physical_block_count;
	uint32_t staging_block_capacity;
} SparkQwen38MaxWorkControlKvPlanConfig;

typedef struct SparkQwen38MaxWorkControlPendingLane
{
	uint64_t sequence_id;
	const uint32_t *nonresident_blocks;
	uint32_t nonresident_block_count;
	uint32_t gdn_nonresident;
} SparkQwen38MaxWorkControlPendingLane;

typedef struct SparkQwen38MaxWorkControlKvBatchState
{
	uint64_t batch_id;
	SparkStatus status;
	uint32_t state;
	uint32_t submitted_block_count;
} SparkQwen38MaxWorkControlKvBatchState;

typedef struct SparkQwen38MaxWorkControlKvState
{
	SparkQwen38MaxWorkControlKvBatchState restore;
	SparkQwen38MaxWorkControlKvBatchState evict;
} SparkQwen38MaxWorkControlKvState;

uint32_t SparkQwen38MaxWorkControlGdnBlockEquivalents(
	uint32_t gdn_record_bytes,
	uint32_t block_record_bytes);
SparkStatus SparkQwen38MaxWorkControlCumulativeNonresident(
	const SparkQwen38MaxWorkControlPendingLane *pending_lanes,
	uint32_t pending_lane_count,
	const uint32_t *packet_lane_counts,
	uint32_t packet_count,
	uint32_t gdn_block_equivalents,
	uint32_t *cumulative_nonresident_block_counts);
uint32_t SparkQwen38MaxWorkControlSelectRestorePackets(
	const SparkQwen38MaxWorkControlKvPlanConfig *configuration,
	uint32_t packet_count,
	const uint32_t *cumulative_nonresident_block_counts);
SparkStatus SparkQwen38MaxWorkControlBuildRestoreBatch(
	const SparkQwen38MaxWorkControlKvPlanConfig *configuration,
	const SparkQwen38MaxWorkControlPendingLane *pending_lanes,
	uint32_t pending_lane_count,
	const uint32_t *packet_lane_counts,
	uint32_t packet_count,
	void *block_staging,
	uint32_t block_staging_record_capacity,
	void *gdn_staging,
	uint32_t gdn_staging_record_capacity,
	SparkKvStoreBlock *blocks,
	uint32_t block_capacity,
	uint32_t *block_count,
	uint32_t *lanes_built);
SparkStatus SparkQwen38MaxWorkControlBuildEvictBatch(
	const SparkQwen38MaxWorkControlKvPlanConfig *configuration,
	uint64_t sequence_id,
	const uint32_t *resident_blocks,
	uint32_t resident_block_count,
	uint32_t include_gdn_state,
	const void *block_staging,
	const void *gdn_staging,
	SparkKvStoreBlock *blocks,
	uint32_t block_capacity,
	uint32_t *block_count);
SparkStatus SparkQwen38MaxWorkControlSubmit(
	SparkStageKvClient *client,
	SparkQwen38MaxWorkControlKvBatchState *batch_state,
	uint32_t operation,
	const SparkKvStoreBlock *blocks,
	uint32_t block_count,
	uint32_t priority);
SparkStatus SparkQwen38MaxWorkControlProgress(
	SparkStageKvClient *client,
	SparkQwen38MaxWorkControlKvState *state);
SparkStatus SparkQwen38MaxWorkControlAcknowledge(
	SparkQwen38MaxWorkControlKvBatchState *batch_state);

#ifdef __cplusplus
}
#endif

#endif
