#include "sparkpipe/spark_speculation_ngram_draft.h"

#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"

#define SPARK_SPECULATION_NGRAM_SCAN_LIMIT 1024u

static uint32_t NgramOrders(const SparkSpeculationNgramDraft *draft)
{
	return(draft->max_match - draft->min_match + 1u);
}

static uint64_t NgramHash(const uint32_t *tokens,uint32_t count)
{
	uint64_t hash = UINT64_C(0xcbf29ce484222325);
	uint32_t index;
	for (index=0u; index<count; index++)
	{
		hash ^= tokens[index];
		hash *= UINT64_C(0x100000001b3);
		hash ^= hash >> 29;
	}
	return(hash);
}

static uint32_t *NgramChain(const SparkSpeculationNgramDraft *draft,uint32_t lane,uint32_t order)
{
	return(draft->chain + ((uint64_t)lane * NgramOrders(draft) + (order - draft->min_match)) * draft->lane_capacity);
}

static uint32_t *NgramHeads(const SparkSpeculationNgramDraft *draft,uint32_t lane,uint32_t order)
{
	return(draft->heads + ((uint64_t)lane * NgramOrders(draft) + (order - draft->min_match)) * ((uint64_t)draft->bucket_mask + 1u));
}

SparkStatus SparkSpeculationNgramDraftInitialize(SparkSpeculationNgramDraft *draft,uint32_t lane_count,uint32_t lane_capacity,uint32_t min_match,uint32_t max_match,uint32_t scan_limit)
{
	uint64_t buckets = 1u,orders,chain_entries,head_entries;
	if ( draft == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft,0,sizeof(*draft));
	if ( lane_count == 0u || lane_capacity < 2u || lane_capacity == SPARK_SPECULATION_NGRAM_NO_POSITION || min_match == 0u || min_match > max_match || max_match > SPARK_SPECULATION_NGRAM_MATCH_LIMIT || scan_limit == 0u || scan_limit > SPARK_SPECULATION_NGRAM_SCAN_LIMIT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	while ( buckets < 2u * (uint64_t)lane_capacity )
		buckets <<= 1;
	orders = max_match - min_match + 1u;
	chain_entries = (uint64_t)lane_count * orders * lane_capacity;
	head_entries = (uint64_t)lane_count * orders * buckets;
	if ( buckets > UINT32_MAX || chain_entries > SIZE_MAX / sizeof(uint32_t) || head_entries > SIZE_MAX / sizeof(uint32_t) || (uint64_t)lane_count * lane_capacity > SIZE_MAX / sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	draft->tokens = (uint32_t *)malloc((size_t)lane_count * lane_capacity * sizeof(uint32_t));
	draft->lengths = (uint32_t *)calloc(lane_count,sizeof(uint32_t));
	draft->sequence_ids = (uint64_t *)calloc(lane_count,sizeof(uint64_t));
	draft->chain = (uint32_t *)malloc((size_t)chain_entries * sizeof(uint32_t));
	draft->heads = (uint32_t *)malloc((size_t)head_entries * sizeof(uint32_t));
	if ( draft->tokens == 0 || draft->lengths == 0 || draft->sequence_ids == 0 || draft->chain == 0 || draft->heads == 0 )
	{
		SparkSpeculationNgramDraftDestroy(draft);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	memset(draft->heads,0xff,(size_t)head_entries * sizeof(uint32_t));
	draft->lane_count = lane_count;
	draft->lane_capacity = lane_capacity;
	draft->min_match = min_match;
	draft->max_match = max_match;
	draft->scan_limit = scan_limit;
	draft->bucket_mask = (uint32_t)(buckets - 1u);
	return(SPARK_STATUS_OK);
}

void SparkSpeculationNgramDraftDestroy(SparkSpeculationNgramDraft *draft)
{
	if ( draft == 0 )
		return;
	free(draft->tokens);
	free(draft->lengths);
	free(draft->sequence_ids);
	free(draft->chain);
	free(draft->heads);
	memset(draft,0,sizeof(*draft));
}

static void NgramResetLane(SparkSpeculationNgramDraft *draft,uint32_t lane)
{
	uint32_t order;
	for (order=draft->min_match; order<=draft->max_match; order++)
		memset(NgramHeads(draft,lane,order),0xff,((size_t)draft->bucket_mask + 1u) * sizeof(uint32_t));
	draft->lengths[lane] = 0u;
}

SparkStatus SparkSpeculationNgramDraftObserve(SparkSpeculationNgramDraft *draft,uint32_t lane,uint64_t sequence_id,uint64_t position,const uint32_t *token_ids,uint32_t token_count)
{
	uint32_t *history,*heads,*chain;
	uint32_t index,order,end;
	uint64_t bucket;
	if ( draft == 0 || draft->tokens == 0 || token_ids == 0 || token_count == 0u || lane >= draft->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( draft->sequence_ids[lane] != sequence_id || position == 0u )
	{
		NgramResetLane(draft,lane);
		draft->sequence_ids[lane] = sequence_id;
	}
	if ( position != draft->lengths[lane] )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( position > draft->lane_capacity || token_count > draft->lane_capacity - position )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	history = draft->tokens + (uint64_t)lane * draft->lane_capacity;
	for (index=0u; index<token_count; index++)
	{
		end = (uint32_t)position + index;
		history[end] = token_ids[index];
		for (order=draft->min_match; order<=draft->max_match && order<=end + 1u; order++)
		{
			heads = NgramHeads(draft,lane,order);
			chain = NgramChain(draft,lane,order);
			bucket = NgramHash(history + end + 1u - order,order) & draft->bucket_mask;
			chain[end] = heads[bucket];
			heads[bucket] = end;
		}
	}
	draft->lengths[lane] = (uint32_t)position + token_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationNgramDraftNext(const SparkSpeculationNgramDraft *draft,uint32_t lane,const uint32_t *suffix,uint32_t suffix_count,SparkSpeculationNgramChoice *choice_out)
{
	uint32_t seen_tokens[SPARK_SPECULATION_NGRAM_SCAN_LIMIT],seen_counts[SPARK_SPECULATION_NGRAM_SCAN_LIMIT];
	const uint32_t *history,*context;
	uint32_t order,length,end,scanned,distinct,index,best;
	if ( draft == 0 || draft->tokens == 0 || suffix == 0 || suffix_count == 0u || lane >= draft->lane_count || choice_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(choice_out,0,sizeof(*choice_out));
	history = draft->tokens + (uint64_t)lane * draft->lane_capacity;
	length = draft->lengths[lane];
	order = suffix_count < draft->max_match ? suffix_count : draft->max_match;
	for (; order>=draft->min_match; order--)
	{
		context = suffix + suffix_count - order;
		end = NgramHeads(draft,lane,order)[NgramHash(context,order) & draft->bucket_mask];
		scanned = 0u;
		distinct = 0u;
		while ( end != SPARK_SPECULATION_NGRAM_NO_POSITION && scanned < draft->scan_limit )
		{
			if ( end + 1u < length && memcmp(history + end + 1u - order,context,(size_t)order * sizeof(uint32_t)) == 0 )
			{
				for (index=0u; index<distinct && seen_tokens[index]!=history[end + 1u]; index++)
					;
				if ( index == distinct )
				{
					seen_tokens[distinct] = history[end + 1u];
					seen_counts[distinct++] = 0u;
				}
				seen_counts[index]++;
				scanned++;
			}
			end = NgramChain(draft,lane,order)[end];
		}
		if ( scanned == 0u )
			continue;
		best = 0u;
		for (index=1u; index<distinct; index++)
			if ( seen_counts[index] > seen_counts[best] )
				best = index;
		choice_out->token_id = seen_tokens[best];
		choice_out->match_length = order;
		choice_out->count = seen_counts[best];
		choice_out->scanned = scanned;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkSpeculationNgramDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	const SparkSpeculationNgramDraft *draft = (const SparkSpeculationNgramDraft *)context;
	uint32_t suffix[SPARK_SPECULATION_NGRAM_MATCH_LIMIT + SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT];
	SparkSpeculationNgramChoice choice;
	const uint32_t *history;
	uint32_t lane,length,take,count;
	SparkStatus status;
	if ( draft == 0 || request == 0 || result == 0 || request->requested_token_count == 0u || request->requested_token_count > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT || request->active_sequence_index >= draft->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	result->token_count = 0u;
	lane = request->active_sequence_index;
	length = draft->lengths[lane];
	if ( draft->sequence_ids[lane] != request->sequence_id || length == 0u || request->sequence_position + 1u != length )
		return(SPARK_STATUS_NOT_FOUND);
	history = draft->tokens + (uint64_t)lane * draft->lane_capacity;
	take = length < draft->max_match ? length : draft->max_match;
	memcpy(suffix,history + length - take,(size_t)take * sizeof(uint32_t));
	for (count=0u; count<request->requested_token_count; count++)
	{
		status = SparkSpeculationNgramDraftNext(draft,lane,suffix + count,take,&choice);
		if ( status == SPARK_STATUS_NOT_FOUND )
			break;
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		result->token_ids[count] = choice.token_id;
		result->confidence_milli[count] = SPARK_SPECULATION_CONFIDENCE_MILLI_ONE * choice.count / choice.scanned;
		suffix[take + count] = choice.token_id;
	}
	result->token_count = count;
	return(count == 0u ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_OK);
}
