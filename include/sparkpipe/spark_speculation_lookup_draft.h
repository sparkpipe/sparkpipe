#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_LOOKUP_UNKNOWN_TOKEN UINT32_MAX

typedef struct SparkSpeculationLookupDraft
{
	uint32_t lane_count;
	uint32_t lane_capacity;
	uint32_t min_match;
	uint32_t max_match;
	uint32_t *tokens;
	uint32_t *lengths;
	uint64_t *sequence_ids;
} SparkSpeculationLookupDraft;

typedef struct SparkSpeculationLookupMatch
{
	uint32_t match_length;
	uint32_t source_end;
} SparkSpeculationLookupMatch;

SparkStatus SparkSpeculationLookupDraftInitialize(SparkSpeculationLookupDraft *draft,uint32_t lane_count,uint32_t lane_capacity,uint32_t min_match,uint32_t max_match);
void SparkSpeculationLookupDraftDestroy(SparkSpeculationLookupDraft *draft);
SparkStatus SparkSpeculationLookupDraftObserve(SparkSpeculationLookupDraft *draft,uint32_t lane,uint64_t sequence_id,uint64_t position,const uint32_t *token_ids,uint32_t token_count);
SparkStatus SparkSpeculationLookupDraftFind(const SparkSpeculationLookupDraft *draft,uint32_t lane,uint64_t anchor_position,SparkSpeculationLookupMatch *match_out);
SparkStatus SparkSpeculationLookupDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result);

#ifdef __cplusplus
}
#endif
