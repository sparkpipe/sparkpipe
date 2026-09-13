#ifndef SPARKPIPE_SPARK_DSV4_CACHE_ARENA_H
#define SPARKPIPE_SPARK_DSV4_CACHE_ARENA_H

#include <stdint.h>

#include "sparkpipe/spark_dsv4_cache_plan.h"
#include "sparkpipe/spark_stage_module_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkDsv4CacheArena
{
	SparkDsv4CachePlan plan;
	void *sliding_arena;
	void *compressed_history_arena;
	void *compressor_state_arena;
}
SparkDsv4CacheArena;

typedef struct SparkDsv4LayerCacheArenaView
{
	const SparkDsv4LayerCachePlan *plan;
	void *sliding_arena;
	void *compressed_history_arena;
	void *compressor_state_arena;
}
SparkDsv4LayerCacheArenaView;

SparkStatus SparkDsv4CacheArenaAllocate(
	const SparkDsv4CachePlanConfiguration *configuration,
	SparkStageModuleLedger *ledger,
	SparkDsv4CacheArena *arena);
SparkStatus SparkDsv4CacheArenaLayerView(
	const SparkDsv4CacheArena *arena,
	uint32_t planned_layer_index,
	SparkDsv4LayerCacheArenaView *view);

#ifdef __cplusplus
}
#endif

#endif
