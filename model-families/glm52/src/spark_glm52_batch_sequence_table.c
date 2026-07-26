#include "sparkpipe/spark_glm52_batch_sequence_table.h"

#include <string.h>

SparkStatus SparkGlm52BatchSequenceTableInitialize(SparkGlm52BatchSequenceTable *table,const SparkGlm52BatchSequenceTableConfiguration *configuration)
{
	if ( table == 0 || configuration == 0 ||
		configuration->abi_version != SPARK_GLM52_BATCH_SEQUENCE_ABI_VERSION ||
		configuration->sequence_capacity == 0u ||
		configuration->sequence_capacity > SPARK_GLM52_BATCH_SEQUENCE_MAX_SEQUENCES ||
		configuration->lane_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(table,0,sizeof(*table));
	table->abi_version = SPARK_GLM52_BATCH_SEQUENCE_ABI_VERSION;
	table->sequence_capacity = configuration->sequence_capacity;
	table->lane_count = configuration->lane_count;
	table->free_head = UINT32_MAX;
	table->free_high_water = 0u;
	return(SPARK_STATUS_OK);
}

static SparkGlm52BatchSequence *SparkGlm52BatchSequenceTableResolveHandle(SparkGlm52BatchSequenceTable *table,uint32_t sequence_handle,uint32_t *sequence_index_out)
{
	uint32_t sequence_index = (sequence_handle & SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK);
	uint32_t generation = (sequence_handle >> SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_BITS);
	if ( sequence_index >= table->sequence_capacity ||
		table->sequences[sequence_index].generation != generation )
		return(0);
	*sequence_index_out = sequence_index;
	return(&table->sequences[sequence_index]);
}

SparkStatus SparkGlm52BatchSequenceTableAdmit(SparkGlm52BatchSequenceTable *table,uint64_t sequence_id,uint32_t context_tokens,uint32_t fragment_base,uint32_t fragment_count,uint32_t *sequence_handle_out)
{
	SparkGlm52BatchSequence *sequence;
	uint32_t sequence_index;
	if ( table == 0 || sequence_handle_out == 0 || fragment_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( table->free_head != UINT32_MAX )
	{
		sequence_index = table->free_head;
		table->free_head = table->sequences[sequence_index].free_next;
	}
	else if ( table->free_high_water < table->sequence_capacity )
	{
		sequence_index = table->free_high_water;
		table->free_high_water += 1u;
	}
	else
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	sequence = &table->sequences[sequence_index];
	sequence->sequence_id = sequence_id;
	sequence->state = SPARK_GLM52_BATCH_SEQUENCE_STATE_ACTIVE;
	sequence->lane_count = table->lane_count;
	sequence->exchange_number = 0u;
	sequence->context_tokens = context_tokens;
	sequence->fragment_base = fragment_base;
	sequence->fragment_count = fragment_count;
	table->active_count += 1u;
	table->exchange_count += 1u;
	*sequence_handle_out = (sequence_index |
		(sequence->generation << SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_BITS));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52BatchSequenceTableBeginExchange(SparkGlm52BatchSequenceTable *table,uint32_t sequence_handle,uint32_t appended_context_tokens)
{
	SparkGlm52BatchSequence *sequence;
	uint32_t sequence_index;
	if ( table == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = SparkGlm52BatchSequenceTableResolveHandle(table,sequence_handle,&sequence_index);
	if ( sequence == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( sequence->state != SPARK_GLM52_BATCH_SEQUENCE_STATE_AWAITING_TOOL )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence->state = SPARK_GLM52_BATCH_SEQUENCE_STATE_ACTIVE;
	sequence->exchange_number += 1u;
	sequence->context_tokens += appended_context_tokens;
	table->awaiting_tool_count -= 1u;
	table->active_count += 1u;
	table->exchange_count += 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52BatchSequenceTablePauseForTool(SparkGlm52BatchSequenceTable *table,uint32_t sequence_handle)
{
	SparkGlm52BatchSequence *sequence;
	uint32_t sequence_index;
	if ( table == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = SparkGlm52BatchSequenceTableResolveHandle(table,sequence_handle,&sequence_index);
	if ( sequence == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( sequence->state != SPARK_GLM52_BATCH_SEQUENCE_STATE_ACTIVE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence->state = SPARK_GLM52_BATCH_SEQUENCE_STATE_AWAITING_TOOL;
	table->active_count -= 1u;
	table->awaiting_tool_count += 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52BatchSequenceTableComplete(SparkGlm52BatchSequenceTable *table,uint32_t sequence_handle)
{
	SparkGlm52BatchSequence *sequence;
	uint32_t sequence_index;
	if ( table == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = SparkGlm52BatchSequenceTableResolveHandle(table,sequence_handle,&sequence_index);
	if ( sequence == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( sequence->state != SPARK_GLM52_BATCH_SEQUENCE_STATE_ACTIVE &&
		sequence->state != SPARK_GLM52_BATCH_SEQUENCE_STATE_AWAITING_TOOL )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( sequence->state == SPARK_GLM52_BATCH_SEQUENCE_STATE_ACTIVE )
		table->active_count -= 1u;
	else
		table->awaiting_tool_count -= 1u;
	sequence->state = SPARK_GLM52_BATCH_SEQUENCE_STATE_COMPLETE;
	sequence->generation += 1u;
	sequence->free_next = table->free_head;
	table->free_head = sequence_index;
	table->complete_count += 1u;
	return(SPARK_STATUS_OK);
}

uint32_t SparkGlm52BatchSequenceTableFiringThreshold(const SparkGlm52BatchSequenceTable *table,uint32_t topk,uint32_t expert_count,uint32_t threshold_cap)
{
	uint32_t threshold;
	if ( table == 0 || expert_count == 0u || threshold_cap == 0u )
		return(1u);
	threshold = ((table->active_count * table->lane_count * topk) / expert_count);
	if ( threshold == 0u )
		threshold = 1u;
	if ( threshold > threshold_cap )
		threshold = threshold_cap;
	return(threshold);
}
