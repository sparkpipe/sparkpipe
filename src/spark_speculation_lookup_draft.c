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

static inline uint32_t SparkSpeculationLookupHash3(uint32_t t0,uint32_t t1,uint32_t t2,uint32_t mask)
{
	uint64_t hash = UINT64_C(0xcbf29ce484222325);
	hash ^= t0;
	hash *= UINT64_C(0x100000001b3);
	hash ^= hash >> 29;
	hash ^= t1;
	hash *= UINT64_C(0x100000001b3);
	hash ^= hash >> 29;
	hash ^= t2;
	hash *= UINT64_C(0x100000001b3);
	hash ^= hash >> 29;
	return((uint32_t)hash & mask);
}

static uint32_t SparkSpeculationLookupIndexBucket(const SparkSpeculationLookupIndex *index,const uint32_t *gram_tokens)
{
	uint64_t hash = UINT64_C(0xcbf29ce484222325);
	uint32_t token;
	if ( index->gram == 3u )
		return(SparkSpeculationLookupHash3(gram_tokens[0],gram_tokens[1],gram_tokens[2],index->bucket_mask));
	for (token=0u; token<index->gram; token++)
	{
		hash ^= gram_tokens[token];
		hash *= UINT64_C(0x100000001b3);
		hash ^= hash >> 29;
	}
	return((uint32_t)hash & index->bucket_mask);
}

static inline uint32_t SparkSpeculationLookupGramEqual(const uint32_t *a,const uint32_t *b,uint32_t gram)
{
	uint32_t i;
	for (i=gram; i-- > 0u;)
		if ( a[i] != b[i] )
			return(0u);
	return(1u);
}

SparkStatus SparkSpeculationLookupIndexBind(SparkSpeculationLookupIndex *index,uint32_t *heads,uint32_t bucket_count,uint32_t *chain,uint32_t capacity,uint32_t gram)
{
	if ( index == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(index,0,sizeof(*index));
	if ( heads == 0 || chain == 0 || bucket_count == 0u || (bucket_count & (bucket_count - 1u)) != 0u || capacity < 2u || capacity == SPARK_SPECULATION_LOOKUP_NO_POSITION || gram == 0u || gram > SPARK_SPECULATION_LOOKUP_MATCH_LIMIT || gram >= capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	index->gram = gram;
	index->bucket_mask = bucket_count - 1u;
	index->capacity = capacity;
	index->heads = heads;
	index->chain = chain;
	SparkSpeculationLookupIndexReset(index);
	return(SPARK_STATUS_OK);
}

void SparkSpeculationLookupIndexReset(SparkSpeculationLookupIndex *index)
{
	if ( index == 0 || index->heads == 0 )
		return;
	memset(index->heads,0xff,((size_t)index->bucket_mask + 1u) * sizeof(uint32_t));
	index->length = 0u;
}

SparkStatus SparkSpeculationLookupIndexAppend(SparkSpeculationLookupIndex *index,const uint32_t *history,uint32_t length)
{
	uint32_t end,bucket;
	uint32_t *heads,*chain;
	uint32_t mask,gram;
	if ( index == 0 || index->heads == 0 || history == 0 || length < index->length )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( length > index->capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	heads = index->heads;
	chain = index->chain;
	mask = index->bucket_mask;
	gram = index->gram;
	if ( gram == 3u )
	{
		uint32_t start = index->length > 2u ? index->length : 2u;
		if ( start < length )
		{
			uint32_t t0 = history[start - 2u];
			uint32_t t1 = history[start - 1u];
			for (end=start; end<length; end++)
			{
				const uint32_t t2 = history[end];
				bucket = SparkSpeculationLookupHash3(t0,t1,t2,mask);
				chain[end] = heads[bucket];
				heads[bucket] = end;
				t0 = t1;
				t1 = t2;
			}
		}
		index->length = length;
		return(SPARK_STATUS_OK);
	}
	for (end=index->length; end<length; end++)
	{
		if ( end + 1u < gram )
			continue;
		bucket = SparkSpeculationLookupIndexBucket(index,history + end + 1u - gram);
		chain[end] = heads[bucket];
		heads[bucket] = end;
	}
	index->length = length;
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkSpeculationLookupTailToken(const uint32_t *history,uint32_t history_length,const uint32_t *tail,uint32_t tail_length,uint32_t back)
{
	return(back < tail_length ? tail[tail_length - 1u - back] : history[history_length - 1u - (back - tail_length)]);
}

SparkStatus SparkSpeculationLookupIndexFindTail(const SparkSpeculationLookupIndex *index,const uint32_t *history,const uint32_t *tail,uint32_t tail_length,
	uint32_t max_match,uint32_t candidate_limit,SparkSpeculationLookupMatch *match_out)
{
	uint32_t last[SPARK_SPECULATION_LOOKUP_MATCH_LIMIT];
	uint32_t end,length,best_length,best_end,steps,token,combined,gram,probe_token = 0u,g0 = 0u,g1 = 0u,g2 = 0u;
	const uint32_t *chain;
	if ( index == 0 || index->heads == 0 || history == 0 || (tail == 0 && tail_length != 0u) || match_out == 0 || max_match < index->gram ||
		max_match > SPARK_SPECULATION_LOOKUP_MATCH_LIMIT || candidate_limit == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	match_out->match_length = 0u;
	match_out->source_end = 0u;
	gram = index->gram;
	combined = index->length + tail_length;
	if ( index->length <= gram || combined < gram )
		return(SPARK_STATUS_NOT_FOUND);
	for (token=0u; token<gram; token++)
		last[token] = SparkSpeculationLookupTailToken(history,index->length,tail,tail_length,gram - 1u - token);
	if ( gram == 3u )
		{ g0 = last[0]; g1 = last[1]; g2 = last[2]; }
	best_length = 0u;
	best_end = 0u;
	chain = index->chain;
	end = index->heads[SparkSpeculationLookupIndexBucket(index,last)];
	for (steps=0u; end != SPARK_SPECULATION_LOOKUP_NO_POSITION && steps < candidate_limit && best_length < max_match; steps++, end=chain[end])
	{
		if ( end + 1u >= index->length )
			continue;
		if ( best_length >= gram && (best_length > end || best_length >= combined || history[end - best_length] != probe_token) )
			continue;
		if ( gram == 3u ? (history[end] != g2 || history[end - 1u] != g1 || history[end - 2u] != g0) :
			SparkSpeculationLookupGramEqual(history + end + 1u - gram,last,gram) == 0u )
			continue;
		length = gram;
		while ( length < max_match && length <= end && length < combined &&
			history[end - length] == SparkSpeculationLookupTailToken(history,index->length,tail,tail_length,length) )
			length++;
		if ( length > best_length )
		{
			best_length = length;
			best_end = end;
			if ( best_length < max_match && best_length < combined )
				probe_token = SparkSpeculationLookupTailToken(history,index->length,tail,tail_length,best_length);
		}
	}
	if ( best_length == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	match_out->match_length = best_length;
	match_out->source_end = best_end;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationLookupIndexFindRanked(const SparkSpeculationLookupIndex *index,const uint32_t *history,uint32_t max_match,uint32_t candidate_limit,
	SparkSpeculationLookupMatch *matches_out,uint32_t match_capacity,uint32_t *match_count_out)
{
	uint32_t anchor,end,length,steps,count = 0u,slot,place,gram,min_len = 0u,probe_token = 0u,g0 = 0u,g1 = 0u,g2 = 0u;
	const uint32_t *chain,*anchor_gram;
	if ( index == 0 || index->heads == 0 || history == 0 || matches_out == 0 || match_count_out == 0 || match_capacity == 0u || max_match < index->gram ||
		max_match > SPARK_SPECULATION_LOOKUP_MATCH_LIMIT || candidate_limit == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*match_count_out = 0u;
	gram = index->gram;
	if ( index->length <= gram )
		return(SPARK_STATUS_NOT_FOUND);
	anchor = index->length - 1u;
	anchor_gram = history + anchor + 1u - gram;
	if ( gram == 3u )
		{ g0 = anchor_gram[0]; g1 = anchor_gram[1]; g2 = anchor_gram[2]; }
	chain = index->chain;
	end = index->heads[SparkSpeculationLookupIndexBucket(index,anchor_gram)];
	for (steps=0u; end != SPARK_SPECULATION_LOOKUP_NO_POSITION && steps < candidate_limit; steps++, end=chain[end])
	{
		if ( end >= anchor )
			continue;
		if ( count == match_capacity )
		{
			if ( min_len >= max_match )
				break;
			if ( min_len > end || history[end - min_len] != probe_token )
				continue;
		}
		if ( gram == 3u ? (history[end] != g2 || history[end - 1u] != g1 || history[end - 2u] != g0) :
			SparkSpeculationLookupGramEqual(history + end + 1u - gram,anchor_gram,gram) == 0u )
			continue;
		length = gram;
		while ( length < max_match && length <= end && history[end - length] == history[anchor - length] )
			length++;
		for (place=0u; place<count && matches_out[place].match_length >= length; place++)
			;
		if ( place >= match_capacity )
			continue;
		for (slot=(count < match_capacity ? count : match_capacity - 1u); slot>place; slot--)
			matches_out[slot] = matches_out[slot - 1u];
		matches_out[place].match_length = length;
		matches_out[place].source_end = end;
		if ( count < match_capacity )
			count++;
		if ( count == match_capacity )
		{
			min_len = matches_out[match_capacity - 1u].match_length;
			if ( min_len < max_match )
				probe_token = history[anchor - min_len];
		}
	}
	*match_count_out = count;
	return(count != 0u ? SPARK_STATUS_OK : SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkSpeculationLookupIndexFind(const SparkSpeculationLookupIndex *index,const uint32_t *history,uint32_t max_match,uint32_t candidate_limit,SparkSpeculationLookupMatch *match_out)
{
	uint32_t anchor,end,length,best_length,best_end,steps,gram,probe_token = 0u,g0 = 0u,g1 = 0u,g2 = 0u;
	const uint32_t *chain,*anchor_gram;
	if ( index == 0 || index->heads == 0 || history == 0 || match_out == 0 || max_match < index->gram || max_match > SPARK_SPECULATION_LOOKUP_MATCH_LIMIT || candidate_limit == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	match_out->match_length = 0u;
	match_out->source_end = 0u;
	gram = index->gram;
	if ( index->length <= gram )
		return(SPARK_STATUS_NOT_FOUND);
	anchor = index->length - 1u;
	anchor_gram = history + anchor + 1u - gram;
	if ( gram == 3u )
		{ g0 = anchor_gram[0]; g1 = anchor_gram[1]; g2 = anchor_gram[2]; }
	chain = index->chain;
	end = index->heads[SparkSpeculationLookupIndexBucket(index,anchor_gram)];
	best_length = 0u;
	best_end = 0u;
	for (steps=0u; end != SPARK_SPECULATION_LOOKUP_NO_POSITION && steps < candidate_limit && best_length < max_match; steps++, end=chain[end])
	{
		if ( end >= anchor )
			continue;
		if ( best_length >= gram && (best_length > end || history[end - best_length] != probe_token) )
			continue;
		if ( gram == 3u ? (history[end] != g2 || history[end - 1u] != g1 || history[end - 2u] != g0) :
			SparkSpeculationLookupGramEqual(history + end + 1u - gram,anchor_gram,gram) == 0u )
			continue;
		length = gram;
		while ( length < max_match && length <= end && history[end - length] == history[anchor - length] )
			length++;
		if ( length > best_length )
		{
			best_length = length;
			best_end = end;
			if ( best_length < max_match )
				probe_token = history[anchor - best_length];
		}
	}
	if ( best_length == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	match_out->match_length = best_length;
	match_out->source_end = best_end;
	return(SPARK_STATUS_OK);
}
