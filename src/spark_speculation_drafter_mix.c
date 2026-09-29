#include "sparkpipe/spark_speculation_drafter_mix.h"

#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_speculation_depth.h"

SparkStatus SparkSpeculationDrafterMixInitialize(SparkSpeculationDrafterMix *mix,SparkSpeculationDraftFunction primary,void *primary_context,SparkSpeculationDraftFunction fallback,void *fallback_context,uint32_t primary_min_tokens,uint32_t lane_count,uint32_t cap_max)
{
	uint32_t lane;
	if ( mix == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(mix,0,sizeof(*mix));
	if ( primary == 0 || fallback == 0 || lane_count == 0u || cap_max == 0u || cap_max > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT || primary_min_tokens == 0u || primary_min_tokens > cap_max )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	mix->caps = (uint32_t *)calloc((size_t)lane_count * SPARK_SPECULATION_DRAFTER_MIX_SOURCES,sizeof(uint32_t));
	mix->sequence_ids = (uint64_t *)calloc(lane_count,sizeof(uint64_t));
	mix->last_source = (uint32_t *)calloc(lane_count,sizeof(uint32_t));
	mix->last_proposed = (uint32_t *)calloc(lane_count,sizeof(uint32_t));
	if ( mix->caps == 0 || mix->sequence_ids == 0 || mix->last_source == 0 || mix->last_proposed == 0 )
	{
		SparkSpeculationDrafterMixDestroy(mix);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	mix->functions[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY] = primary;
	mix->contexts[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY] = primary_context;
	mix->functions[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK] = fallback;
	mix->contexts[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK] = fallback_context;
	mix->primary_min_tokens = primary_min_tokens;
	mix->lane_count = lane_count;
	mix->cap_max = cap_max;
	for (lane=0u; lane<lane_count; lane++)
	{
		mix->caps[lane * SPARK_SPECULATION_DRAFTER_MIX_SOURCES + SPARK_SPECULATION_DRAFTER_MIX_PRIMARY] = cap_max;
		mix->caps[lane * SPARK_SPECULATION_DRAFTER_MIX_SOURCES + SPARK_SPECULATION_DRAFTER_MIX_FALLBACK] = cap_max;
		mix->last_source[lane] = SPARK_SPECULATION_DRAFTER_MIX_NONE;
	}
	return(SPARK_STATUS_OK);
}

void SparkSpeculationDrafterMixDestroy(SparkSpeculationDrafterMix *mix)
{
	if ( mix == 0 )
		return;
	free(mix->caps);
	free(mix->sequence_ids);
	free(mix->last_source);
	free(mix->last_proposed);
	memset(mix,0,sizeof(*mix));
}

static SparkStatus SparkSpeculationDrafterMixAsk(SparkSpeculationDrafterMix *mix,uint32_t source,const SparkSpeculationPolicyDraftRequest *request,uint32_t cap,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationPolicyDraftRequest sub;
	SparkStatus status;
	sub = *request;
	sub.requested_token_count = request->requested_token_count < cap ? request->requested_token_count : cap;
	memset(result,0,sizeof(*result));
	result->abi_version = SPARK_SPECULATION_ABI_VERSION;
	result->descriptor_bytes = SPARK_SPECULATION_DRAFT_RESULT_DESCRIPTOR_BYTES;
	status = mix->functions[source](mix->contexts[source],&sub,result);
	if ( status == SPARK_STATUS_NOT_FOUND )
	{
		result->token_count = 0u;
		return(SPARK_STATUS_OK);
	}
	if ( status == SPARK_STATUS_OK && result->token_count > sub.requested_token_count )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	SPARK_RETURN(status);
}

SparkStatus SparkSpeculationDrafterMixTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationDrafterMix *mix = (SparkSpeculationDrafterMix *)context;
	uint32_t lane,*caps,need;
	SparkStatus status;
	if ( mix == 0 || mix->caps == 0 || request == 0 || result == 0 || request->requested_token_count == 0u || request->requested_token_count > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT || request->active_sequence_index >= mix->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	lane = request->active_sequence_index;
	caps = mix->caps + (uint64_t)lane * SPARK_SPECULATION_DRAFTER_MIX_SOURCES;
	if ( mix->sequence_ids[lane] != request->sequence_id )
	{
		mix->sequence_ids[lane] = request->sequence_id;
		caps[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY] = mix->cap_max;
		caps[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK] = mix->cap_max;
	}
	mix->last_source[lane] = SPARK_SPECULATION_DRAFTER_MIX_NONE;
	mix->last_proposed[lane] = 0u;
	status = SparkSpeculationDrafterMixAsk(mix,SPARK_SPECULATION_DRAFTER_MIX_PRIMARY,request,caps[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY],result);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	need = request->requested_token_count < mix->primary_min_tokens ? request->requested_token_count : mix->primary_min_tokens;
	if ( result->token_count != 0u && result->token_count >= need )
	{
		mix->last_source[lane] = SPARK_SPECULATION_DRAFTER_MIX_PRIMARY;
		mix->last_proposed[lane] = result->token_count;
		return(SPARK_STATUS_OK);
	}
	if ( result->token_count != 0u )
		mix->counters[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY].declined++;
	status = SparkSpeculationDrafterMixAsk(mix,SPARK_SPECULATION_DRAFTER_MIX_FALLBACK,request,caps[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK],result);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( result->token_count == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	mix->last_source[lane] = SPARK_SPECULATION_DRAFTER_MIX_FALLBACK;
	mix->last_proposed[lane] = result->token_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationDrafterMixObserve(SparkSpeculationDrafterMix *mix,uint32_t lane,uint32_t proposed,uint32_t accepted)
{
	SparkSpeculationDrafterMixCounters *counters;
	uint32_t source,*cap,next;
	if ( mix == 0 || mix->caps == 0 || lane >= mix->lane_count || accepted > proposed )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	source = mix->last_source[lane];
	if ( source >= SPARK_SPECULATION_DRAFTER_MIX_SOURCES || proposed != mix->last_proposed[lane] )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	counters = &mix->counters[source];
	counters->rounds++;
	counters->proposed += proposed;
	counters->accepted += accepted;
	cap = &mix->caps[(uint64_t)lane * SPARK_SPECULATION_DRAFTER_MIX_SOURCES + source];
	next = SparkSpeculationDepthCapNext(*cap,proposed,accepted,mix->cap_max);
	if ( source == SPARK_SPECULATION_DRAFTER_MIX_PRIMARY && next < mix->primary_min_tokens )
		next = mix->primary_min_tokens;
	*cap = next;
	mix->last_source[lane] = SPARK_SPECULATION_DRAFTER_MIX_NONE;
	mix->last_proposed[lane] = 0u;
	return(SPARK_STATUS_OK);
}

uint32_t SparkSpeculationDrafterMixLastSource(const SparkSpeculationDrafterMix *mix,uint32_t lane)
{
	if ( mix == 0 || mix->last_source == 0 || lane >= mix->lane_count )
		return(SPARK_SPECULATION_DRAFTER_MIX_NONE);
	return(mix->last_source[lane]);
}
