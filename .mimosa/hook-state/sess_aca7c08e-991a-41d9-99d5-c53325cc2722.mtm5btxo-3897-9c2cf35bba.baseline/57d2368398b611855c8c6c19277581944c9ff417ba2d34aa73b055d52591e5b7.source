#include "sparkpipe/spark_dsv4_cache_arena.h"

#include <stddef.h>
#include <string.h>

static void *SparkDsv4CacheArenaOffsetPointer(
	void *arena,
	uint64_t arena_bytes,
	uint64_t offset_bytes,
	uint64_t region_bytes)
{
	if ( region_bytes == 0u )
		return(0);
	if ( arena == 0 || offset_bytes > arena_bytes ||
		region_bytes > arena_bytes - offset_bytes )
		return(0);
	return((unsigned char *)arena + offset_bytes);
}

SparkStatus SparkDsv4CacheArenaAllocate(
	const SparkDsv4CachePlanConfiguration *configuration,
	SparkStageModuleLedger *ledger,
	SparkDsv4CacheArena *arena)
{
	SparkDsv4CacheArena candidate;
	SparkStatus status;
	uint32_t allocation_checkpoint;

	if ( configuration == 0 || ledger == 0 || arena == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&candidate,0,sizeof(candidate));
	allocation_checkpoint = ledger->device_allocation_count;
	status = SparkDsv4CachePlanBuild(configuration,&candidate.plan);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( candidate.plan.sliding_arena_bytes != 0u )
	{
		status = SparkStageModuleDeviceAllocateZeroed(
			ledger,candidate.plan.sliding_arena_bytes,&candidate.sliding_arena);
		if ( status != SPARK_STATUS_OK )
			goto rollback;
	}
	if ( candidate.plan.compressed_history_arena_bytes != 0u )
	{
		status = SparkStageModuleDeviceAllocateZeroed(
			ledger,candidate.plan.compressed_history_arena_bytes,
			&candidate.compressed_history_arena);
		if ( status != SPARK_STATUS_OK )
			goto rollback;
	}
	if ( candidate.plan.compressor_state_arena_bytes != 0u )
	{
		status = SparkStageModuleDeviceAllocateZeroed(
			ledger,candidate.plan.compressor_state_arena_bytes,
			&candidate.compressor_state_arena);
		if ( status != SPARK_STATUS_OK )
			goto rollback;
	}
	*arena = candidate;
	return(SPARK_STATUS_OK);

rollback:
	SparkStageModuleLedgerRollback(ledger,allocation_checkpoint);
	return(status);
}

SparkStatus SparkDsv4CacheArenaLayerView(
	const SparkDsv4CacheArena *arena,
	uint32_t planned_layer_index,
	SparkDsv4LayerCacheArenaView *view)
{
	const SparkDsv4LayerCachePlan *layer_plan;

	if ( arena == 0 || view == 0 ||
		planned_layer_index >= arena->plan.planned_layer_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(view,0,sizeof(*view));
	layer_plan = &arena->plan.layers[planned_layer_index];
	view->plan = layer_plan;
	view->sliding_arena = SparkDsv4CacheArenaOffsetPointer(
		arena->sliding_arena,
		arena->plan.sliding_arena_bytes,
		layer_plan->sliding_arena_offset_bytes,
		layer_plan->sliding_arena_bytes);
	view->compressed_history_arena = SparkDsv4CacheArenaOffsetPointer(
		arena->compressed_history_arena,
		arena->plan.compressed_history_arena_bytes,
		layer_plan->compressed_history_arena_offset_bytes,
		layer_plan->compressed_history_arena_bytes);
	view->compressor_state_arena = SparkDsv4CacheArenaOffsetPointer(
		arena->compressor_state_arena,
		arena->plan.compressor_state_arena_bytes,
		layer_plan->compressor_state_arena_offset_bytes,
		layer_plan->compressor_state_arena_bytes);
	if ( (layer_plan->sliding_arena_bytes != 0u && view->sliding_arena == 0) ||
		(layer_plan->compressed_history_arena_bytes != 0u &&
		 view->compressed_history_arena == 0) ||
		(layer_plan->compressor_state_arena_bytes != 0u &&
		 view->compressor_state_arena == 0) )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}
