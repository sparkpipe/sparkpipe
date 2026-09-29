#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_NGRAM_MATCH_LIMIT 16u
#define SPARK_SPECULATION_NGRAM_NO_POSITION UINT32_MAX

typedef struct SparkSpeculationNgramDraft
{
	uint32_t lane_count;
	uint32_t lane_capacity;
	uint32_t min_match;
	uint32_t max_match;
	uint32_t scan_limit;
	uint32_t bucket_mask;
	uint32_t *tokens;
	uint32_t *lengths;
	uint64_t *sequence_ids;
	uint32_t *chain;
	uint32_t *heads;
} SparkSpeculationNgramDraft;

typedef struct SparkSpeculationNgramChoice
{
	uint32_t token_id;
	uint32_t match_length;
	uint32_t count;
	uint32_t scanned;
} SparkSpeculationNgramChoice;

SparkStatus SparkSpeculationNgramDraftInitialize(SparkSpeculationNgramDraft *draft,uint32_t lane_count,uint32_t lane_capacity,uint32_t min_match,uint32_t max_match,uint32_t scan_limit);
void SparkSpeculationNgramDraftDestroy(SparkSpeculationNgramDraft *draft);
SparkStatus SparkSpeculationNgramDraftObserve(SparkSpeculationNgramDraft *draft,uint32_t lane,uint64_t sequence_id,uint64_t position,const uint32_t *token_ids,uint32_t token_count);
SparkStatus SparkSpeculationNgramDraftNext(const SparkSpeculationNgramDraft *draft,uint32_t lane,const uint32_t *suffix,uint32_t suffix_count,SparkSpeculationNgramChoice *choice_out);
SparkStatus SparkSpeculationNgramDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result);

#ifdef __cplusplus
}
#endif
