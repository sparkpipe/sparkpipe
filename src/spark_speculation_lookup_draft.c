#include "sparkpipe/spark_speculation_lookup_draft.h"

#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"

#define SPARK_SPECULATION_LOOKUP_MATCH_LIMIT 64u

SparkStatus SparkSpeculationLookupDraftInitialize(SparkSpeculationLookupDraft *draft,uint32_t lane_count,uint32_t lane_capacity,uint32_t min_match,uint32_t max_match)
{
	if ( draft == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft,0,sizeof(*draft));
	if ( lane_count == 0u || lane_capacity < 2u || min_match == 0u || min_match > max_match || max_match > SPARK_SPECULATION_LOOKUP_MATCH_LIMIT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (uint64_t)lane_count * lane_capacity > (uint64_t)SIZE_MAX / sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	draft->tokens = (uint32_t *)malloc((size_t)lane_count * lane_capacity * sizeof(uint32_t));
	draft->lengths = (uint32_t *)calloc(lane_count,sizeof(uint32_t));
	draft->sequence_ids = (uint64_t *)calloc(lane_count,sizeof(uint64_t));
	if ( draft->tokens == 0 || draft->lengths == 0 || draft->sequence_ids == 0 )
	{
		SparkSpeculationLookupDraftDestroy(draft);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	draft->lane_count = lane_count;
	draft->lane_capacity = lane_capacity;
	draft->min_match = min_match;
	draft->max_match = max_match;
	return(SPARK_STATUS_OK);
}

void SparkSpeculationLookupDraftDestroy(SparkSpeculationLookupDraft *draft)
{
	if ( draft == 0 )
		return;
	free(draft->tokens);
	free(draft->lengths);
	free(draft->sequence_ids);
	memset(draft,0,sizeof(*draft));
}

SparkStatus SparkSpeculationLookupDraftObserve(SparkSpeculationLookupDraft *draft,uint32_t lane,uint64_t sequence_id,uint64_t position,const uint32_t *token_ids,uint32_t token_count)
{
	uint32_t *history;
	uint32_t index;
	if ( draft == 0 || draft->tokens == 0 || token_ids == 0 || token_count == 0u || lane >= draft->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( position > draft->lane_capacity || token_count > draft->lane_capacity - position )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<token_count; index++)
		if ( token_ids[index] == SPARK_SPECULATION_LOOKUP_UNKNOWN_TOKEN )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	history = draft->tokens + (uint64_t)lane * draft->lane_capacity;
	if ( draft->sequence_ids[lane] != sequence_id || position == 0u )
	{
		draft->sequence_ids[lane] = sequence_id;
		draft->lengths[lane] = 0u;
	}
	for (index=draft->lengths[lane]; index<position; index++)
		history[index] = SPARK_SPECULATION_LOOKUP_UNKNOWN_TOKEN;
	memcpy(history + position,token_ids,(size_t)token_count * sizeof(uint32_t));
	draft->lengths[lane] = (uint32_t)position + token_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationLookupDraftFind(const SparkSpeculationLookupDraft *draft,uint32_t lane,uint64_t anchor_position,SparkSpeculationLookupMatch *match_out)
{
	const uint32_t *history;
	uint32_t anchor,end,length,best_length,best_end;
	if ( draft == 0 || draft->tokens == 0 || match_out == 0 || lane >= draft->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	match_out->match_length = 0u;
	match_out->source_end = 0u;
	history = draft->tokens + (uint64_t)lane * draft->lane_capacity;
	if ( anchor_position >= draft->lengths[lane] || history[anchor_position] == SPARK_SPECULATION_LOOKUP_UNKNOWN_TOKEN )
		return(SPARK_STATUS_NOT_FOUND);
	anchor = (uint32_t)anchor_position;
	best_length = 0u;
	best_end = 0u;
	for (end=anchor; end>0u && best_length<draft->max_match; end--)
	{
		length = 0u;
		while ( length < draft->max_match && length < end && history[end - 1u - length] == history[anchor - length] && history[anchor - length] != SPARK_SPECULATION_LOOKUP_UNKNOWN_TOKEN )
			length++;
		if ( length > best_length )
		{
			best_length = length;
			best_end = end - 1u;
		}
	}
	if ( best_length >= draft->min_match )
	{
		match_out->match_length = best_length;
		match_out->source_end = best_end;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationLookupDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	const SparkSpeculationLookupDraft *draft = (const SparkSpeculationLookupDraft *)context;
	SparkSpeculationLookupMatch match;
	const uint32_t *history;
	uint32_t count,token,confidence;
	uint64_t source;
	SparkStatus status;
	if ( draft == 0 || request == 0 || result == 0 || request->requested_token_count == 0u || request->requested_token_count > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT || request->active_sequence_index >= draft->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	result->token_count = 0u;
	if ( draft->sequence_ids[request->active_sequence_index] != request->sequence_id )
		return(SPARK_STATUS_NOT_FOUND);
	status = SparkSpeculationLookupDraftFind(draft,request->active_sequence_index,request->sequence_position,&match);
	if ( status == SPARK_STATUS_NOT_FOUND )
		return(status);
	if ( status != SPARK_STATUS_OK || match.match_length == 0u )
		SPARK_RETURN(status);
	history = draft->tokens + (uint64_t)request->active_sequence_index * draft->lane_capacity;
	confidence = SPARK_SPECULATION_CONFIDENCE_MILLI_ONE * match.match_length / draft->max_match;
	for (count=0u; count<request->requested_token_count; count++)
	{
		source = (uint64_t)match.source_end + 1u + count;
		if ( source > request->sequence_position )
			break;
		token = history[source];
		if ( token == SPARK_SPECULATION_LOOKUP_UNKNOWN_TOKEN )
			break;
		result->token_ids[count] = token;
		result->confidence_milli[count] = confidence;
	}
	result->token_count = count;
	return(SPARK_STATUS_OK);
}
