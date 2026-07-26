#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_kv_store.h"
#include "sparkpipe/spark_stage_kv_client.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Qwen 3.6 stage residency planning over the KV tier.
 *
 * The runtime owns the block table, the lane map and the eviction policy;
 * this layer owns the transfer plan: how many pending decode packets to
 * restore ahead of execution under physical-block and staging pressure,
 * the batch that fetches a lane back (its attention block records plus,
 * uniquely to this hybrid, the lane's GDN recurrence record), the batch
 * that stores an evicted lane out, and a single-inflight-per-direction
 * state machine over spark_stage_kv_client. A lane is never split across
 * batches: a restore either carries every nonresident record of a lane or
 * leaves the whole lane for the next batch, so a completed batch is
 * immediately actionable. Packet zero is the frame about to execute:
 * its restore submits at priority zero, everything speculative at one.
 * A BUSY submit changes nothing - the frame simply stays queued and the
 * caller retries after progress.
 *
 * The GDN record has no physical KV block behind it but does consume
 * staging, so pressure math charges it in block equivalents; the physical
 * axis is thereby conservative, never overcommitted. Its key uses the
 * reserved logical block below under the ordinary key scheme.
 */

#define SPARK_QWEN36_WORK_CONTROL_GDN_RECORD_BLOCK 0xFFFFFFFFu
#define SPARK_QWEN36_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE 0u
#define SPARK_QWEN36_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE 1u

#define SPARK_QWEN36_WORK_CONTROL_BATCH_IDLE 0u
#define SPARK_QWEN36_WORK_CONTROL_BATCH_WAIT 1u
#define SPARK_QWEN36_WORK_CONTROL_BATCH_READY 2u

typedef struct SparkQwen36WorkControlPendingLane
{
	uint64_t sequence_id;
	uint32_t lane_index;
	uint32_t nonresident_block_count;
	const uint32_t *nonresident_blocks;
	uint32_t gdn_nonresident;
	uint32_t reserved0;
} SparkQwen36WorkControlPendingLane;

typedef struct SparkQwen36WorkControlKvPlanConfig
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
	uint32_t reserved0;
} SparkQwen36WorkControlKvPlanConfig;

typedef struct SparkQwen36WorkControlKvBatchState
{
	uint64_t batch_id;
	uint32_t state;
	SparkStatus status;
} SparkQwen36WorkControlKvBatchState;

typedef struct SparkQwen36WorkControlKvState
{
	SparkQwen36WorkControlKvBatchState restore;
	SparkQwen36WorkControlKvBatchState evict;
} SparkQwen36WorkControlKvState;

uint32_t SparkQwen36WorkControlGdnBlockEquivalents(uint32_t gdn_record_bytes, uint32_t block_record_bytes);
SparkStatus SparkQwen36WorkControlCumulativeNonresident(const SparkQwen36WorkControlPendingLane *pending, uint32_t lane_count, const uint32_t *packet_lane_counts, uint32_t packet_count, uint32_t gdn_block_equivalents, uint32_t *cumulative_out);
uint32_t SparkQwen36WorkControlSelectRestorePackets(const SparkQwen36WorkControlKvPlanConfig *config, uint32_t packet_count, const uint32_t *cumulative_nonresident);
SparkStatus SparkQwen36WorkControlBuildRestoreBatch(const SparkQwen36WorkControlKvPlanConfig *config, const SparkQwen36WorkControlPendingLane *pending, uint32_t lane_count, const uint32_t *packet_lane_counts, uint32_t selected_packet_count, void *block_staging, uint32_t block_staging_capacity, void *gdn_staging, uint32_t gdn_staging_capacity, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count_out, uint32_t *lanes_built_out);
SparkStatus SparkQwen36WorkControlBuildEvictBatch(const SparkQwen36WorkControlKvPlanConfig *config, uint64_t sequence_id, const uint32_t *resident_blocks, uint32_t resident_block_count, uint32_t gdn_present, void *block_staging, void *gdn_staging, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count_out);
SparkStatus SparkQwen36WorkControlSubmit(SparkStageKvClient *client, SparkQwen36WorkControlKvBatchState *batch_state, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority);
SparkStatus SparkQwen36WorkControlProgress(SparkStageKvClient *client, SparkQwen36WorkControlKvState *kv_state);
SparkStatus SparkQwen36WorkControlAcknowledge(SparkQwen36WorkControlKvBatchState *batch_state);

#ifdef __cplusplus
}
#endif
