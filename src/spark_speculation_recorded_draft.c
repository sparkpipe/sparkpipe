#include "sparkpipe/spark_speculation_recorded_draft.h"

#include <string.h>

#include "sparkpipe/spark_error_site.h"

static uint32_t SparkSpeculationRecordedGet32(const uint8_t *bytes)
{
	return((uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24);
}

static uint64_t SparkSpeculationRecordedGet64(const uint8_t *bytes)
{
	return((uint64_t)SparkSpeculationRecordedGet32(bytes) | (uint64_t)SparkSpeculationRecordedGet32(bytes + 4) << 32);
}

static int SparkSpeculationRecordedCompare(const uint8_t *entry,uint64_t sequence_id,uint64_t position)
{
	uint64_t entry_sequence = SparkSpeculationRecordedGet64(entry),entry_position = SparkSpeculationRecordedGet64(entry + 8);
	if ( entry_sequence != sequence_id )
		return(entry_sequence < sequence_id ? -1 : 1);
	if ( entry_position != position )
		return(entry_position < position ? -1 : 1);
	return(0);
}

SparkStatus SparkSpeculationRecordedDraftInitialize(SparkSpeculationRecordedDraft *draft,const uint8_t *bytes,uint64_t byte_count,uint32_t vocab_size)
{
	uint64_t index,count;
	uint32_t depth,entry_bytes,token_count,slot;
	const uint8_t *entry,*previous;
	if ( draft == 0 || bytes == 0 || vocab_size < 2u || byte_count < SPARK_SPECULATION_RECORDED_HEADER_BYTES )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft,0,sizeof(*draft));
	if ( SparkSpeculationRecordedGet32(bytes) != SPARK_SPECULATION_RECORDED_MAGIC || SparkSpeculationRecordedGet32(bytes + 4) != SPARK_SPECULATION_RECORDED_VERSION )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	depth = SparkSpeculationRecordedGet32(bytes + 8);
	if ( depth == 0u || depth > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT || SparkSpeculationRecordedGet32(bytes + 12) != vocab_size || SparkSpeculationRecordedGet64(bytes + 24) != 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkSpeculationRecordedGet64(bytes + 16);
	entry_bytes = SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES + 4u * depth;
	if ( count == 0u || count > (byte_count - SPARK_SPECULATION_RECORDED_HEADER_BYTES) / entry_bytes || byte_count != SPARK_SPECULATION_RECORDED_HEADER_BYTES + count * entry_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	previous = 0;
	for (index=0u; index<count; index++)
	{
		entry = bytes + SPARK_SPECULATION_RECORDED_HEADER_BYTES + index * entry_bytes;
		token_count = SparkSpeculationRecordedGet32(entry + 16);
		if ( token_count == 0u || token_count > depth || SparkSpeculationRecordedGet32(entry + 20) != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( previous != 0 && SparkSpeculationRecordedCompare(previous,SparkSpeculationRecordedGet64(entry),SparkSpeculationRecordedGet64(entry + 8)) >= 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for (slot=0u; slot<depth; slot++)
		{
			uint32_t token = SparkSpeculationRecordedGet32(entry + SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES + 4u * slot);
			if ( (slot < token_count && token >= vocab_size) || (slot >= token_count && token != 0u) )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		previous = entry;
	}
	draft->entries = bytes + SPARK_SPECULATION_RECORDED_HEADER_BYTES;
	draft->entry_count = count;
	draft->depth = depth;
	draft->entry_bytes = entry_bytes;
	draft->vocab_size = vocab_size;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRecordedDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationRecordedDraft *draft = (SparkSpeculationRecordedDraft *)context;
	uint64_t low,high,middle;
	uint32_t index,count;
	const uint8_t *entry;
	int order;
	if ( draft == 0 || draft->entries == 0 || request == 0 || result == 0 || request->requested_token_count == 0u || request->requested_token_count > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	low = 0u;
	high = draft->entry_count;
	while ( low < high )
	{
		middle = low + (high - low) / 2u;
		entry = draft->entries + middle * draft->entry_bytes;
		order = SparkSpeculationRecordedCompare(entry,request->sequence_id,request->sequence_position);
		if ( order == 0 )
		{
			count = SparkSpeculationRecordedGet32(entry + 16);
			count = count < request->requested_token_count ? count : request->requested_token_count;
			for (index=0u; index<count; index++)
			{
				result->token_ids[index] = SparkSpeculationRecordedGet32(entry + SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES + 4u * index);
				result->confidence_milli[index] = SPARK_SPECULATION_CONFIDENCE_MILLI_ONE;
			}
			result->token_count = count;
			draft->hits++;
			return(SPARK_STATUS_OK);
		}
		if ( order < 0 )
			low = middle + 1u;
		else
			high = middle;
	}
	draft->misses++;
	return(SPARK_STATUS_NOT_FOUND);
}
