#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_DRAFTER_MIX_PRIMARY 0u
#define SPARK_SPECULATION_DRAFTER_MIX_FALLBACK 1u
#define SPARK_SPECULATION_DRAFTER_MIX_SOURCES 2u
#define SPARK_SPECULATION_DRAFTER_MIX_NONE UINT32_MAX

typedef struct SparkSpeculationDrafterMixCounters
{
	uint64_t rounds;
	uint64_t proposed;
	uint64_t accepted;
	uint64_t declined;
} SparkSpeculationDrafterMixCounters;

typedef struct SparkSpeculationDrafterMix
{
	SparkSpeculationDraftFunction functions[SPARK_SPECULATION_DRAFTER_MIX_SOURCES];
	void *contexts[SPARK_SPECULATION_DRAFTER_MIX_SOURCES];
	uint32_t primary_min_tokens;
	uint32_t lane_count;
	uint32_t cap_max;
	uint32_t *caps;
	uint64_t *sequence_ids;
	uint32_t *last_source;
	uint32_t *last_proposed;
	SparkSpeculationDrafterMixCounters counters[SPARK_SPECULATION_DRAFTER_MIX_SOURCES];
} SparkSpeculationDrafterMix;

SparkStatus SparkSpeculationDrafterMixInitialize(SparkSpeculationDrafterMix *mix,SparkSpeculationDraftFunction primary,void *primary_context,SparkSpeculationDraftFunction fallback,void *fallback_context,uint32_t primary_min_tokens,uint32_t lane_count,uint32_t cap_max);
void SparkSpeculationDrafterMixDestroy(SparkSpeculationDrafterMix *mix);
SparkStatus SparkSpeculationDrafterMixTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result);
SparkStatus SparkSpeculationDrafterMixObserve(SparkSpeculationDrafterMix *mix,uint32_t lane,uint32_t proposed,uint32_t accepted);
uint32_t SparkSpeculationDrafterMixLastSource(const SparkSpeculationDrafterMix *mix,uint32_t lane);

#ifdef __cplusplus
}
#endif
