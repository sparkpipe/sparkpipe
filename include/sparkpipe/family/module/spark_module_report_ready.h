#pragma once

static SparkStatus SPARK_FAMILY(ModuleKvFrameBuildRestoreBatch)(const LmKvFramePlanConfig *configuration, const LmKvFramePendingLane *pending_lanes, uint32_t pending_lane_count, const uint32_t *packet_lane_counts, uint32_t packet_count, void *block_staging, uint32_t block_staging_record_capacity, void *gdn_staging, uint32_t gdn_staging_record_capacity, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count, uint32_t *lanes_built)
{
	_Static_assert(sizeof(LmKvFramePlanConfig) == sizeof(SPARK_FAMILY(WorkControlKvPlanConfig)), "kv plan layout");
	_Static_assert(sizeof(LmKvFramePendingLane) == sizeof(SPARK_FAMILY(WorkControlPendingLane)), "kv pending lane layout");
	return(SPARK_FAMILY(WorkControlBuildRestoreBatch)((const SPARK_FAMILY(WorkControlKvPlanConfig) *)configuration,(const SPARK_FAMILY(WorkControlPendingLane) *)pending_lanes,pending_lane_count,packet_lane_counts,packet_count,block_staging,block_staging_record_capacity,gdn_staging,gdn_staging_record_capacity,blocks,block_capacity,block_count,lanes_built));
}

static SparkStatus SPARK_FAMILY(ModuleKvFrameBuildEvictBatch)(const LmKvFramePlanConfig *configuration, uint64_t sequence_id, const uint32_t *resident_blocks, uint32_t resident_block_count, uint32_t include_gdn_state, const void *block_staging, const void *gdn_staging, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count)
{
	return(SPARK_FAMILY(WorkControlBuildEvictBatch)((const SPARK_FAMILY(WorkControlKvPlanConfig) *)configuration,sequence_id,resident_blocks,resident_block_count,include_gdn_state,block_staging,gdn_staging,blocks,block_capacity,block_count));
}

static SparkStatus SPARK_FAMILY(ModuleKvFrameSubmit)(SparkStageKvClient *client, LmKvFrameBatchState *batch_state, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority)
{
	_Static_assert(sizeof(LmKvFrameBatchState) == sizeof(SPARK_FAMILY(WorkControlKvBatchState)), "kv batch state layout");
	return(SPARK_FAMILY(WorkControlSubmit)(client,(SPARK_FAMILY(WorkControlKvBatchState) *)batch_state,operation,blocks,block_count,priority));
}

static SparkStatus SPARK_FAMILY(ModuleKvFrameProgress)(SparkStageKvClient *client, LmKvFrameWorkState *work)
{
	_Static_assert(sizeof(LmKvFrameWorkState) == sizeof(SPARK_FAMILY(WorkControlKvState)), "kv work state layout");
	return(SPARK_FAMILY(WorkControlProgress)(client,(SPARK_FAMILY(WorkControlKvState) *)work));
}

static SparkStatus SPARK_FAMILY(ModuleKvFrameAcknowledge)(LmKvFrameBatchState *batch_state)
{
	return(SPARK_FAMILY(WorkControlAcknowledge)((SPARK_FAMILY(WorkControlKvBatchState) *)batch_state));
}

static void SPARK_FAMILY(ModuleReportReady)(void *module_state)
{
	SPARK_FAMILY(ModuleState) *state = (SPARK_FAMILY(ModuleState) *)module_state;
	fprintf(stderr,"%s initialize ok slice=%u+%u gdn=%u attn=%u owns_embedding=%u owns_head=%u\n",SPARK_FAMILY_CONST(MODULE_TAG),state->first_layer_index,state->layer_count,state->gdn_layer_count,state->attn_layer_count,state->owns_embedding,state->owns_final_head);
}
