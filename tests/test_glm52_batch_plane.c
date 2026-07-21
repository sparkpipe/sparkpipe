#include "sparkpipe/spark_glm52_batch_sequence_table.h"
#include "sparkpipe/spark_glm52_expert_queue.h"
#include "sparkpipe/spark_glm52_jit_kv_pool.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static SparkGlm52ExpertQueue test_queue;
static SparkGlm52JitKvPool test_pool;
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
}

static void SparkTestJitKvPoolPrefetchEvictionAndLateness(void)
{
	SparkGlm52JitKvPoolConfiguration configuration;
	uint32_t require_ids[2u];
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	configuration.fragment_capacity = 8u;
	configuration.dram_fragment_capacity = 2u;
	configuration.fragment_bytes = 1000000u;
	configuration.nvme_bytes_per_second = 1000000000u;
	assert(SparkGlm52JitKvPoolInitialize(&test_pool,&configuration) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,0u,10u,0u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,1u,10u,1u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,2u,11u,0u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,3u,11u,1u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) == SPARK_STATUS_OK);
	require_ids[0u] = 0u;
	require_ids[1u] = 1u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,0u,require_ids,2u,5000000u) == SPARK_STATUS_OK);
	assert(test_pool.hit_count == 2u && test_pool.miss_count == 0u);
	require_ids[0u] = 2u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,0u,require_ids,1u,3000000u) == SPARK_STATUS_OK);
	assert(test_pool.miss_count == 1u && test_pool.stage_in_count == 1u && test_pool.stage_out_count == 1u);
	assert(test_pool.staging_in_count == 1u && test_pool.staging_out_count == 1u);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,2u) == 0u);
	assert(SparkGlm52JitKvPoolTick(&test_pool,3000000u) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,2u) == 1u);
	assert(test_pool.dram_resident_count == 2u && test_pool.late_count == 0u);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,0u) == 1u);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,1u) == 0u);
	require_ids[0u] = 3u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,3000000u,require_ids,1u,3500000u) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolTick(&test_pool,10000000u) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,3u) == 1u);
	assert(test_pool.late_count == 1u);
	require_ids[0u] = 0u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,10000000u,require_ids,1u,10500000u) == SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void SparkTestBatchSequenceTableLifecycleAndThreshold(void)
{
	SparkGlm52BatchSequenceTableConfiguration configuration;
	uint32_t first_index,second_index;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_BATCH_SEQUENCE_ABI_VERSION;
	configuration.sequence_capacity = 4u;
	configuration.lane_count = 8u;
	assert(SparkGlm52BatchSequenceTableInitialize(&test_table,&configuration) == SPARK_STATUS_OK);
	assert(SparkGlm52BatchSequenceTableAdmit(&test_table,900u,8192u,0u,128u,&first_index) == SPARK_STATUS_OK);
	assert(SparkGlm52BatchSequenceTableAdmit(&test_table,901u,8192u,128u,128u,&second_index) == SPARK_STATUS_OK);
	assert(test_table.active_count == 2u);
	assert(SparkGlm52BatchSequenceTableFiringThreshold(&test_table,8u,256u,1024u) == 1u);
	{
		uint32_t fill_index,scratch;
		for (fill_index=2u; fill_index<4u; fill_index++)
			assert(SparkGlm52BatchSequenceTableAdmit(&test_table,900u + fill_index,8192u,fill_index * 128u,128u,&scratch) == SPARK_STATUS_OK);
		assert(SparkGlm52BatchSequenceTableAdmit(&test_table,999u,8192u,512u,128u,&scratch) == SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	assert(SparkGlm52BatchSequenceTableFiringThreshold(&test_table,8u,256u,1024u) == 1u);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,first_index) == SPARK_STATUS_OK);
	assert(test_table.active_count == 3u && test_table.awaiting_tool_count == 1u);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,first_index) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52BatchSequenceTableBeginExchange(&test_table,first_index,192u) == SPARK_STATUS_OK);
	assert(test_table.sequences[first_index].exchange_number == 1u);
	assert(test_table.sequences[first_index].context_tokens == 8384u);
	assert(test_table.active_count == 4u && test_table.exchange_count == 5u);
	assert(SparkGlm52BatchSequenceTableComplete(&test_table,second_index) == SPARK_STATUS_OK);
	assert(test_table.active_count == 3u && test_table.complete_count == 1u);
	assert(SparkGlm52BatchSequenceTableComplete(&test_table,second_index) == SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkGlm52JitKvPool stress_pool;

static void SparkTestJitKvPoolScaleAndBurst(void)
{
	SparkGlm52JitKvPoolConfiguration configuration;
	uint32_t fragment_index,require_id;
	uint64_t now_ns = 0u;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	configuration.fragment_capacity = 200000u;
	configuration.dram_fragment_capacity = 100000u;
	configuration.fragment_bytes = 442368u;
	configuration.nvme_bytes_per_second = 6000000000u;
	assert(SparkGlm52JitKvPoolInitialize(&stress_pool,&configuration) == SPARK_STATUS_OK);
	for (fragment_index=0u; fragment_index<200000u; fragment_index++)
		assert(SparkGlm52JitKvPoolAdmitFragment(&stress_pool,fragment_index,fragment_index / 32u,fragment_index % 32u,fragment_index < 100000u ? SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM : SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) == SPARK_STATUS_OK);
	assert(stress_pool.eviction_heap_count == 100000u);
	// Burst far beyond the transfer ring without ticking: inline overflow drain
	// must keep every require succeeding rather than hard-failing.
	for (fragment_index=0u; fragment_index<10000u; fragment_index++)
	{
		require_id = 100000u + fragment_index;
		assert(SparkGlm52JitKvPoolRequireByEta(&stress_pool,now_ns,&require_id,1u,now_ns + 20000000000u) == SPARK_STATUS_OK);
	}
	assert(stress_pool.overflow_drain_count != 0u);
	assert(stress_pool.transfer_count <= SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS);
	// Heap root is always the farthest-future need among residents.
	{
		uint32_t root = stress_pool.eviction_heap[0u],child;
		for (child=1u; child<stress_pool.eviction_heap_count && child<7u; child++)
			assert(stress_pool.fragments[root].next_need_ns >= stress_pool.fragments[stress_pool.eviction_heap[child]].next_need_ns);
	}
}

int main(void)
{
	SparkTestExpertQueueThresholdDeadlineAndOrder();
	SparkTestJitKvPoolPrefetchEvictionAndLateness();
	SparkTestBatchSequenceTableLifecycleAndThreshold();
	SparkTestJitKvPoolScaleAndBurst();
	printf("test_glm52_batch_plane PASS\n");
	return(0);
}
