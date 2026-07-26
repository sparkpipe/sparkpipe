#include "sparkpipe/spark_glm52_batch_sequence_table.h"
#include "sparkpipe/spark_glm52_expert_queue.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static SparkGlm52ExpertQueue test_queue;
static SparkGlm52BatchSequenceTable test_table;

static void SparkTestExpertQueueThresholdDeadlineAndOrder(void)
{
	SparkGlm52ExpertQueueConfiguration configuration;
	SparkGlm52ExpertQueueFiring firing;
	uint32_t row_index;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_EXPERT_QUEUE_ABI_VERSION;
	configuration.layer_count = 6u;
	configuration.expert_count = 256u;
	configuration.firing_threshold_rows = 4u;
	configuration.firing_deadline_ns = 1000000u;
	assert(SparkGlm52ExpertQueueInitialize(&test_queue,&configuration) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,0u,&firing) == SPARK_STATUS_NOT_FOUND);
	for (row_index=0u; row_index<4u; row_index++)
		assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,2u,17u,1000u + row_index,100u + row_index) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,1u,200u,2000u,50u) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,2u,3u,3000u,60u) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,200u,&firing) == SPARK_STATUS_OK);
	assert(firing.layer_index == 2u && firing.expert_index == 17u && firing.row_count == 4u);
	assert(firing.row_ids[0u] == 1000u && firing.row_ids[3u] == 1003u);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,200u,&firing) == SPARK_STATUS_NOT_FOUND);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,50u + 1000000u,&firing) == SPARK_STATUS_OK);
	assert(firing.layer_index == 1u && firing.expert_index == 200u && firing.row_count == 1u);
	assert(firing.row_ids[0u] == 2000u);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,60u + 1000000u,&firing) == SPARK_STATUS_OK);
	assert(firing.layer_index == 2u && firing.expert_index == 3u && firing.row_count == 1u);
	assert(test_queue.enqueued_row_count == 0u);
	assert(test_queue.firing_count == 3u && test_queue.fired_row_count == 6u);
	// Lazy free-list correctness: allocate past a fired-and-recycled batch and
	// confirm rows keep flowing without the eager init that used to touch 1M
	// entries. Overfill one slot beyond the emit cap and verify the cap holds
	// and the remainder stays queued with a correct advanced oldest arrival.
	{
		uint32_t bulk_index;
		SparkGlm52ExpertQueueFiring bulk;
		for (bulk_index=0u; bulk_index<SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS + 200u; ++bulk_index)
			assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,0u,0u,7000u + bulk_index,10u + bulk_index) == SPARK_STATUS_OK);
		assert(SparkGlm52ExpertQueueNextFiring(&test_queue,0u,&bulk) == SPARK_STATUS_OK);
		assert(bulk.row_count == SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS);
		assert(bulk.row_ids[0u] == 7000u);
		assert(test_queue.slots[0u][0u].count == 200u);
		assert(test_queue.slots[0u][0u].oldest_arrival_ns == 10u + SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS);
	}
}

static void SparkTestBatchSequenceTableLifecycleAndThreshold(void)
{
	SparkGlm52BatchSequenceTableConfiguration configuration;
	uint32_t first_handle,second_handle,first_index;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_BATCH_SEQUENCE_ABI_VERSION;
	configuration.sequence_capacity = 4u;
	configuration.lane_count = 8u;
	assert(SparkGlm52BatchSequenceTableInitialize(&test_table,&configuration) == SPARK_STATUS_OK);
	assert(SparkGlm52BatchSequenceTableAdmit(&test_table,900u,8192u,0u,128u,&first_handle) == SPARK_STATUS_OK);
	assert(SparkGlm52BatchSequenceTableAdmit(&test_table,901u,8192u,128u,128u,&second_handle) == SPARK_STATUS_OK);
	assert(test_table.active_count == 2u);
	assert(SparkGlm52BatchSequenceTableFiringThreshold(&test_table,8u,256u,1024u) == 1u);
	{
		uint32_t fill_index,scratch;
		for (fill_index=2u; fill_index<4u; fill_index++)
			assert(SparkGlm52BatchSequenceTableAdmit(&test_table,900u + fill_index,8192u,fill_index * 128u,128u,&scratch) == SPARK_STATUS_OK);
		assert(SparkGlm52BatchSequenceTableAdmit(&test_table,999u,8192u,512u,128u,&scratch) == SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	assert(SparkGlm52BatchSequenceTableFiringThreshold(&test_table,8u,256u,1024u) == 1u);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,first_handle) == SPARK_STATUS_OK);
	assert(test_table.active_count == 3u && test_table.awaiting_tool_count == 1u);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,first_handle) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52BatchSequenceTableBeginExchange(&test_table,first_handle,192u) == SPARK_STATUS_OK);
	first_index = (first_handle & SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK);
	assert(test_table.sequences[first_index].exchange_number == 1u);
	assert(test_table.sequences[first_index].context_tokens == 8384u);
	assert(test_table.active_count == 4u && test_table.exchange_count == 5u);
	assert(SparkGlm52BatchSequenceTableComplete(&test_table,second_handle) == SPARK_STATUS_OK);
	assert(test_table.active_count == 3u && test_table.complete_count == 1u);
	// The stale handle now fails handle resolution, not just the state check:
	// the generation moved when the slot was freed, so a holdover handle can
	// never act on the slot's next occupant.
	assert(SparkGlm52BatchSequenceTableComplete(&test_table,second_handle) == SPARK_STATUS_NOT_FOUND);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,second_handle) == SPARK_STATUS_NOT_FOUND);
	// The completed slot must be reclaimable: a capacity-4 table that has seen
	// completions keeps admitting under churn instead of leaking slots forever.
	{
		uint32_t churn_index,recycled,previous = second_handle;
		for (churn_index=0u; churn_index<64u; ++churn_index)
		{
			assert(SparkGlm52BatchSequenceTableAdmit(&test_table,5000u + churn_index,4089u,0u,64u,&recycled) == SPARK_STATUS_OK);
			assert((recycled & SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK) ==
				(second_handle & SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK));
			assert(recycled != previous);
			previous = recycled;
			assert(SparkGlm52BatchSequenceTableComplete(&test_table,recycled) == SPARK_STATUS_OK);
		}
	}
}

int main(void)
{
	SparkTestExpertQueueThresholdDeadlineAndOrder();
	SparkTestBatchSequenceTableLifecycleAndThreshold();
	printf("test_glm52_batch_plane PASS\n");
	return(0);
}
