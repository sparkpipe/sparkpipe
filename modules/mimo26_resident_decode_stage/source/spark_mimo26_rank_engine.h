#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MIMO26_RANK_ENGINE_MODE_EAGER 0u
#define SPARK_MIMO26_RANK_ENGINE_MODE_GRAPH 1u

typedef struct SparkMimo26RankEngineConfig
{
	uint32_t rank;
	uint32_t lane_count;
	uint32_t max_positions;
	uint32_t mode;
	uint64_t expert_pool_bytes;
	uint64_t spine_budget_bytes;
	uint64_t wait_ns;
	const char *pack_path;
	const char *pack_sha256;
	const char *dump_directory;
} SparkMimo26RankEngineConfig;

typedef struct SparkMimo26RankEngineStats
{
	uint64_t steps;
	uint64_t graph_steps;
	uint64_t step_ns;
	uint64_t collectives;
	uint64_t spine_bytes;
	uint64_t expert_pool_bytes;
	uint32_t pinned_experts;
	uint32_t kv_error;
} SparkMimo26RankEngineStats;

typedef struct SparkMimo26RankEngine SparkMimo26RankEngine;

SparkStatus SparkMimo26RankEngineCreate(const SparkMimo26RankEngineConfig *config, SparkMimo26RankEngine **out);
SparkStatus SparkMimo26RankEngineStep(SparkMimo26RankEngine *engine, uint32_t lane, uint32_t token, uint32_t position, uint32_t *next_token, float *score);
SparkStatus SparkMimo26RankEngineReadStats(SparkMimo26RankEngine *engine, SparkMimo26RankEngineStats *stats);
SparkStatus SparkMimo26RankEngineDestroy(SparkMimo26RankEngine *engine);

#ifdef __cplusplus
}
#endif
