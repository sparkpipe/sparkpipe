#ifndef SPARKPIPE_SPARK_QWEN38_27B_WORK_CONTROL_H
#define SPARKPIPE_SPARK_QWEN38_27B_WORK_CONTROL_H

#include <stdint.h>

#include "sparkpipe/spark_stage_kv_client.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QWEN38_27B_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE 0u
#define SPARK_QWEN38_27B_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE 1u

#define SPARK_QWEN38_27B_WORK_CONTROL_BATCH_IDLE 0u
#define SPARK_QWEN38_27B_WORK_CONTROL_BATCH_SUBMITTED 1u
#define SPARK_QWEN38_27B_WORK_CONTROL_BATCH_READY 2u

typedef struct SparkQwen38_27bWorkControlKvPlanConfig
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
} SparkQwen38_27bWorkControlKvPlanConfig;

typedef struct SparkQwen38_27bWorkControlPendingLane
{
	uint64_t sequence_id;
	const uint32_t *nonresident_blocks;
	uint32_t nonresident_block_count;
	uint32_t gdn_nonresident;
} SparkQwen38_27bWorkControlPendingLane;

typedef struct SparkQwen38_27bWorkControlKvBatchState
{
	uint64_t batch_id;
	SparkStatus status;
	uint32_t state;
	uint32_t submitted_block_count;
} SparkQwen38_27bWorkControlKvBatchState;

typedef struct SparkQwen38_27bWorkControlKvState
{
	SparkQwen38_27bWorkControlKvBatchState restore;
	SparkQwen38_27bWorkControlKvBatchState evict;
} SparkQwen38_27bWorkControlKvState;

uint32_t SparkQwen38_27bWorkControlGdnBlockEquivalents(
	uint32_t gdn_record_bytes,
	uint32_t block_record_bytes);
SparkStatus SparkQwen38_27bWorkControlCumulativeNonresident(
	const SparkQwen38_27bWorkControlPendingLane *pending_lanes,
	uint32_t pending_lane_count,
	const uint32_t *packet_lane_counts,
	uint32_t packet_count,
	uint32_t gdn_block_equivalents,
	uint32_t *cumulative_nonresident_block_counts);
uint32_t SparkQwen38_27bWorkControlSelectRestorePackets(
	const SparkQwen38_27bWorkControlKvPlanConfig *configuration,
	uint32_t packet_count,
	const uint32_t *cumulative_nonresident_block_counts);
SparkStatus SparkQwen38_27bWorkControlBuildRestoreBatch(
	const SparkQwen38_27bWorkControlKvPlanConfig *configuration,
	const SparkQwen38_27bWorkControlPendingLane *pending_lanes,
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
SparkStatus SparkQwen38_27bWorkControlBuildEvictBatch(
	const SparkQwen38_27bWorkControlKvPlanConfig *configuration,
	uint64_t sequence_id,
	const uint32_t *resident_blocks,
	uint32_t resident_block_count,
	uint32_t include_gdn_state,
	const void *block_staging,
	const void *gdn_staging,
	SparkKvStoreBlock *blocks,
	uint32_t block_capacity,
	uint32_t *block_count);
SparkStatus SparkQwen38_27bWorkControlSubmit(
	SparkStageKvClient *client,
	SparkQwen38_27bWorkControlKvBatchState *batch_state,
	uint32_t operation,
	const SparkKvStoreBlock *blocks,
	uint32_t block_count,
	uint32_t priority);
SparkStatus SparkQwen38_27bWorkControlProgress(
	SparkStageKvClient *client,
	SparkQwen38_27bWorkControlKvState *state);
SparkStatus SparkQwen38_27bWorkControlAcknowledge(
	SparkQwen38_27bWorkControlKvBatchState *batch_state);

#ifdef __cplusplus
}
#endif

#endif
