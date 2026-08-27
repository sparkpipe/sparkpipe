#ifndef SPARKPIPE_SPARK_QWEN4_FLASH_WORK_CONTROL_H
#define SPARKPIPE_SPARK_QWEN4_FLASH_WORK_CONTROL_H

#include <stdint.h>

#include "sparkpipe/spark_stage_kv_client.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QWEN4_FLASH_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE 0u
#define SPARK_QWEN4_FLASH_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE 1u

#define SPARK_QWEN4_FLASH_WORK_CONTROL_BATCH_IDLE 0u
#define SPARK_QWEN4_FLASH_WORK_CONTROL_BATCH_SUBMITTED 1u
#define SPARK_QWEN4_FLASH_WORK_CONTROL_BATCH_READY 2u

typedef struct SparkQwen4FlashWorkControlKvPlanConfig
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
} SparkQwen4FlashWorkControlKvPlanConfig;

typedef struct SparkQwen4FlashWorkControlPendingLane
{
	uint64_t sequence_id;
	const uint32_t *nonresident_blocks;
	uint32_t nonresident_block_count;
	uint32_t gdn_nonresident;
} SparkQwen4FlashWorkControlPendingLane;

typedef struct SparkQwen4FlashWorkControlKvBatchState
{
	uint64_t batch_id;
	SparkStatus status;
	uint32_t state;
	uint32_t submitted_block_count;
} SparkQwen4FlashWorkControlKvBatchState;

typedef struct SparkQwen4FlashWorkControlKvState
{
	SparkQwen4FlashWorkControlKvBatchState restore;
	SparkQwen4FlashWorkControlKvBatchState evict;
} SparkQwen4FlashWorkControlKvState;

uint32_t SparkQwen4FlashWorkControlGdnBlockEquivalents(
	uint32_t gdn_record_bytes,
	uint32_t block_record_bytes);
SparkStatus SparkQwen4FlashWorkControlCumulativeNonresident(
	const SparkQwen4FlashWorkControlPendingLane *pending_lanes,
	uint32_t pending_lane_count,
	const uint32_t *packet_lane_counts,
	uint32_t packet_count,
	uint32_t gdn_block_equivalents,
	uint32_t *cumulative_nonresident_block_counts);
uint32_t SparkQwen4FlashWorkControlSelectRestorePackets(
	const SparkQwen4FlashWorkControlKvPlanConfig *configuration,
	uint32_t packet_count,
	const uint32_t *cumulative_nonresident_block_counts);
SparkStatus SparkQwen4FlashWorkControlBuildRestoreBatch(
	const SparkQwen4FlashWorkControlKvPlanConfig *configuration,
	const SparkQwen4FlashWorkControlPendingLane *pending_lanes,
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
SparkStatus SparkQwen4FlashWorkControlBuildEvictBatch(
	const SparkQwen4FlashWorkControlKvPlanConfig *configuration,
	uint64_t sequence_id,
	const uint32_t *resident_blocks,
	uint32_t resident_block_count,
	uint32_t include_gdn_state,
	const void *block_staging,
	const void *gdn_staging,
	SparkKvStoreBlock *blocks,
	uint32_t block_capacity,
	uint32_t *block_count);
SparkStatus SparkQwen4FlashWorkControlSubmit(
	SparkStageKvClient *client,
	SparkQwen4FlashWorkControlKvBatchState *batch_state,
	uint32_t operation,
	const SparkKvStoreBlock *blocks,
	uint32_t block_count,
	uint32_t priority);
SparkStatus SparkQwen4FlashWorkControlProgress(
	SparkStageKvClient *client,
	SparkQwen4FlashWorkControlKvState *state);
SparkStatus SparkQwen4FlashWorkControlAcknowledge(
	SparkQwen4FlashWorkControlKvBatchState *batch_state);

#ifdef __cplusplus
}
#endif

#endif
