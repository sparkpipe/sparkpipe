#include <stdint.h>

#include "sparkpipe/spark_stage_kv_client.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SPARK_WORK_CONTROL_TYPE(KvPlanConfig)
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
} SPARK_WORK_CONTROL_TYPE(KvPlanConfig);

typedef struct SPARK_WORK_CONTROL_TYPE(PendingLane)
{
	uint64_t sequence_id;
	const uint32_t *nonresident_blocks;
	uint32_t nonresident_block_count;
	uint32_t gdn_nonresident;
} SPARK_WORK_CONTROL_TYPE(PendingLane);

typedef struct SPARK_WORK_CONTROL_TYPE(KvBatchState)
{
	uint64_t batch_id;
	SparkStatus status;
	uint32_t state;
	uint32_t submitted_block_count;
} SPARK_WORK_CONTROL_TYPE(KvBatchState);

typedef struct SPARK_WORK_CONTROL_TYPE(KvState)
{
	SPARK_WORK_CONTROL_TYPE(KvBatchState) restore;
	SPARK_WORK_CONTROL_TYPE(KvBatchState) evict;
} SPARK_WORK_CONTROL_TYPE(KvState);

uint32_t SPARK_WORK_CONTROL_FN(GdnBlockEquivalents)(
	uint32_t gdn_record_bytes,
	uint32_t block_record_bytes);
SparkStatus SPARK_WORK_CONTROL_FN(CumulativeNonresident)(
	const SPARK_WORK_CONTROL_TYPE(PendingLane) *pending_lanes,
	uint32_t pending_lane_count,
	const uint32_t *packet_lane_counts,
	uint32_t packet_count,
	uint32_t gdn_block_equivalents,
	uint32_t *cumulative_nonresident_block_counts);
uint32_t SPARK_WORK_CONTROL_FN(SelectRestorePackets)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration,
	uint32_t packet_count,
	const uint32_t *cumulative_nonresident_block_counts);
SparkStatus SPARK_WORK_CONTROL_FN(BuildRestoreBatch)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration,
	const SPARK_WORK_CONTROL_TYPE(PendingLane) *pending_lanes,
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
SparkStatus SPARK_WORK_CONTROL_FN(BuildEvictBatch)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration,
	uint64_t sequence_id,
	const uint32_t *resident_blocks,
	uint32_t resident_block_count,
	uint32_t include_gdn_state,
	const void *block_staging,
	const void *gdn_staging,
	SparkKvStoreBlock *blocks,
	uint32_t block_capacity,
	uint32_t *block_count);
SparkStatus SPARK_WORK_CONTROL_FN(Submit)(
	SparkStageKvClient *client,
	SPARK_WORK_CONTROL_TYPE(KvBatchState) *batch_state,
	uint32_t operation,
	const SparkKvStoreBlock *blocks,
	uint32_t block_count,
	uint32_t priority);
SparkStatus SPARK_WORK_CONTROL_FN(Progress)(
	SparkStageKvClient *client,
	SPARK_WORK_CONTROL_TYPE(KvState) *state);
SparkStatus SPARK_WORK_CONTROL_FN(Acknowledge)(
	SPARK_WORK_CONTROL_TYPE(KvBatchState) *batch_state);

#ifdef __cplusplus
}
#endif

#undef SPARK_WORK_CONTROL_FN
#undef SPARK_WORK_CONTROL_TYPE
