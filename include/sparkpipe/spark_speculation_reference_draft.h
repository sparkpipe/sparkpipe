#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkSpeculationReferenceMode
{
	SPARK_SPECULATION_REFERENCE_ORACLE = 1,
	SPARK_SPECULATION_REFERENCE_ADVERSARY = 2
} SparkSpeculationReferenceMode;

typedef struct SparkSpeculationReferenceDraft
{
	SparkSpeculationReferenceMode mode;
	uint32_t vocab_size;
	uint64_t first_position;
	const uint32_t *token_ids;
	uint32_t token_count;
} SparkSpeculationReferenceDraft;

SparkStatus SparkSpeculationReferenceDraftInitialize(SparkSpeculationReferenceDraft *draft,SparkSpeculationReferenceMode mode,uint32_t vocab_size,uint64_t first_position,const uint32_t *token_ids,uint32_t token_count);
SparkStatus SparkSpeculationReferenceDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result);

#ifdef __cplusplus
}
#endif
