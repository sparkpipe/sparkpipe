#include "sparkpipe/spark_glm52_jit_kv_pool.h"

#include <string.h>

// Max-heap over DRAM-resident fragments keyed on next_need_ns: the root is the
// farthest-future need, the correct eviction victim by Belady. heap_position
// tracks each fragment's slot so an ETA change re-sifts in place. Transfers are
// a ring buffer; on overflow the oldest in-flight transfer is completed inline
// rather than refusing the require, so a prefetch burst never hard-fails.

static void SparkGlm52JitKvPoolHeapSwap(SparkGlm52JitKvPool *pool,uint32_t a,uint32_t b)
{
	uint32_t fragment_a = pool->eviction_heap[a],fragment_b = pool->eviction_heap[b];
	pool->eviction_heap[a] = fragment_b;
	pool->eviction_heap[b] = fragment_a;
	pool->fragments[fragment_a].heap_position = b;
	pool->fragments[fragment_b].heap_position = a;
}

// Ordering: farthest next_need_ns is the max (root, evicted first); on equal
// need the larger fragment_id is the max, so ties evict highest id and keep
// lowest, deterministic for the ring SHA gates.
static uint32_t SparkGlm52JitKvPoolHeapGreater(const SparkGlm52JitKvPool *pool,uint32_t heap_left,uint32_t heap_right)
{
	uint32_t fragment_left = pool->eviction_heap[heap_left],fragment_right = pool->eviction_heap[heap_right];
	uint64_t need_left = pool->fragments[fragment_left].next_need_ns,need_right = pool->fragments[fragment_right].next_need_ns;
	if ( need_left != need_right )
		return(need_left > need_right ? 1u : 0u);
	return(fragment_left > fragment_right ? 1u : 0u);
}

static void SparkGlm52JitKvPoolHeapSiftUp(SparkGlm52JitKvPool *pool,uint32_t position)
{
	while (position != 0u)
	{
		uint32_t parent = ((position - 1u) / 2u);
		if ( !SparkGlm52JitKvPoolHeapGreater(pool,position,parent) )
			break;
		SparkGlm52JitKvPoolHeapSwap(pool,position,parent);
		position = parent;
	}
}

static void SparkGlm52JitKvPoolHeapSiftDown(SparkGlm52JitKvPool *pool,uint32_t position)
{
	for (;;)
	{
		uint32_t left = (2u * position + 1u),right = (2u * position + 2u),largest = position;
		if ( left < pool->eviction_heap_count && SparkGlm52JitKvPoolHeapGreater(pool,left,largest) )
			largest = left;
		if ( right < pool->eviction_heap_count && SparkGlm52JitKvPoolHeapGreater(pool,right,largest) )
			largest = right;
		if ( largest == position )
			break;
		SparkGlm52JitKvPoolHeapSwap(pool,position,largest);
		position = largest;
	}
}

static void SparkGlm52JitKvPoolHeapInsert(SparkGlm52JitKvPool *pool,uint32_t fragment_id)
{
	uint32_t position = pool->eviction_heap_count;
	pool->eviction_heap[position] = fragment_id;
	pool->fragments[fragment_id].heap_position = position;
	pool->eviction_heap_count += 1u;
	SparkGlm52JitKvPoolHeapSiftUp(pool,position);
}

static void SparkGlm52JitKvPoolHeapRemove(SparkGlm52JitKvPool *pool,uint32_t fragment_id)
{
	uint32_t position = pool->fragments[fragment_id].heap_position,last;
	if ( position >= pool->eviction_heap_count || pool->eviction_heap[position] != fragment_id )
		return;
	last = (pool->eviction_heap_count - 1u);
	pool->eviction_heap_count -= 1u;
	if ( position == last )
		return;
	SparkGlm52JitKvPoolHeapSwap(pool,position,last);
	SparkGlm52JitKvPoolHeapSiftDown(pool,position);
	SparkGlm52JitKvPoolHeapSiftUp(pool,position);
}

static void SparkGlm52JitKvPoolHeapUpdate(SparkGlm52JitKvPool *pool,uint32_t fragment_id)
{
	uint32_t position = pool->fragments[fragment_id].heap_position;
	if ( position >= pool->eviction_heap_count || pool->eviction_heap[position] != fragment_id )
		return;
	SparkGlm52JitKvPoolHeapSiftDown(pool,position);
	SparkGlm52JitKvPoolHeapSiftUp(pool,position);
}

SparkStatus SparkGlm52JitKvPoolInitialize(SparkGlm52JitKvPool *pool,const SparkGlm52JitKvPoolConfiguration *configuration)
{
	if ( pool == 0 || configuration == 0 ||
		configuration->abi_version != SPARK_GLM52_JIT_KV_POOL_ABI_VERSION ||
		configuration->fragment_capacity == 0u ||
		configuration->fragment_capacity > SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS ||
		configuration->dram_fragment_capacity == 0u ||
		configuration->dram_fragment_capacity > configuration->fragment_capacity ||
		configuration->fragment_bytes == 0u ||
		configuration->nvme_bytes_per_second == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pool,0,sizeof(*pool));
	pool->abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	pool->fragment_capacity = configuration->fragment_capacity;
	pool->dram_fragment_capacity = configuration->dram_fragment_capacity;
	pool->fragment_bytes = configuration->fragment_bytes;
	pool->nvme_bytes_per_second = configuration->nvme_bytes_per_second;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52JitKvPoolAdmitFragment(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint64_t sequence_id,uint32_t fragment_index_in_sequence,uint32_t initial_state)
{
	SparkGlm52JitKvFragment *fragment;
	if ( pool == 0 || fragment_id >= pool->fragment_capacity ||
		(initial_state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME &&
		 initial_state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	fragment = &pool->fragments[fragment_id];
	if ( fragment->state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_FREE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( initial_state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM &&
		pool->dram_resident_count >= pool->dram_fragment_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	fragment->sequence_id = sequence_id;
	fragment->fragment_index_in_sequence = fragment_index_in_sequence;
	fragment->state = initial_state;
	fragment->next_need_ns = UINT64_MAX;
	if ( initial_state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM )
	{
		pool->dram_resident_count += 1u;
		SparkGlm52JitKvPoolHeapInsert(pool,fragment_id);
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm52JitKvPoolCompleteTransfer(SparkGlm52JitKvPool *pool,uint32_t ring_index,uint64_t now_ns)
{
	SparkGlm52JitKvTransfer *transfer = &pool->transfers[ring_index];
	SparkGlm52JitKvFragment *fragment = &pool->fragments[transfer->fragment_id];
	uint64_t effective_now = (now_ns > transfer->done_ns ? now_ns : transfer->done_ns);
	if ( transfer->direction_in != 0u )
	{
		fragment->state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM;
		pool->staging_in_count -= 1u;
		pool->dram_resident_count += 1u;
		SparkGlm52JitKvPoolHeapInsert(pool,transfer->fragment_id);
		if ( transfer->done_ns > fragment->next_need_ns )
			pool->late_count += 1u;
	}
	else
	{
		fragment->state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME;
		fragment->next_need_ns = UINT64_MAX;
		pool->staging_out_count -= 1u;
	}
	(void)effective_now;
}

static SparkStatus SparkGlm52JitKvPoolQueueTransfer(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint32_t direction_in,uint64_t now_ns)
{
	SparkGlm52JitKvTransfer *transfer;
	uint64_t start_ns,duration_ns;
	uint32_t tail;
	if ( pool->transfer_count >= SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS )
	{
		SparkGlm52JitKvPoolCompleteTransfer(pool,pool->transfer_head,now_ns);
		pool->transfer_head = ((pool->transfer_head + 1u) % SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS);
		pool->transfer_count -= 1u;
		pool->overflow_drain_count += 1u;
	}
	start_ns = (pool->nvme_busy_until_ns > now_ns ? pool->nvme_busy_until_ns : now_ns);
	duration_ns = ((pool->fragment_bytes * 1000000000u) / pool->nvme_bytes_per_second);
	if ( duration_ns == 0u )
		duration_ns = 1u;
	tail = ((pool->transfer_head + pool->transfer_count) % SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS);
	transfer = &pool->transfers[tail];
	transfer->fragment_id = fragment_id;
	transfer->direction_in = direction_in;
	transfer->start_ns = start_ns;
	transfer->done_ns = (start_ns + duration_ns);
	pool->transfer_count += 1u;
	pool->nvme_busy_until_ns = transfer->done_ns;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52JitKvPoolStageIn(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint64_t now_ns)
{
	SparkGlm52JitKvFragment *fragment = &pool->fragments[fragment_id];
	SparkStatus status;
	if ( pool->dram_resident_count + pool->staging_in_count >= pool->dram_fragment_capacity )
	{
		uint32_t victim;
		if ( pool->eviction_heap_count == 0u )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		victim = pool->eviction_heap[0];
		if ( pool->fragments[victim].next_need_ns <= fragment->next_need_ns )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		SparkGlm52JitKvPoolHeapRemove(pool,victim);
		status = SparkGlm52JitKvPoolQueueTransfer(pool,victim,0u,now_ns);
		if ( status != SPARK_STATUS_OK )
			return(status);
		pool->fragments[victim].state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_OUT;
		pool->dram_resident_count -= 1u;
		pool->staging_out_count += 1u;
		pool->stage_out_count += 1u;
	}
	status = SparkGlm52JitKvPoolQueueTransfer(pool,fragment_id,1u,now_ns);
	if ( status != SPARK_STATUS_OK )
		return(status);
	fragment->state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_IN;
	pool->staging_in_count += 1u;
	pool->stage_in_count += 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52JitKvPoolRequireByEta(SparkGlm52JitKvPool *pool,uint64_t now_ns,const uint32_t *fragment_ids,uint32_t fragment_count,uint64_t need_ns)
{
	SparkGlm52JitKvFragment *fragment;
	uint32_t request_index;
	SparkStatus status;
	if ( pool == 0 || (fragment_ids == 0 && fragment_count != 0u) || need_ns < now_ns )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (request_index=0u; request_index<fragment_count; request_index++)
	{
		if ( fragment_ids[request_index] >= pool->fragment_capacity )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		fragment = &pool->fragments[fragment_ids[request_index]];
		if ( fragment->state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_FREE )
			return(SPARK_STATUS_NOT_FOUND);
		if ( need_ns < fragment->next_need_ns )
		{
			fragment->next_need_ns = need_ns;
			if ( fragment->state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM )
				SparkGlm52JitKvPoolHeapUpdate(pool,fragment_ids[request_index]);
		}
		if ( fragment->state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM ||
			fragment->state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_IN )
		{
			pool->hit_count += 1u;
			continue;
		}
		pool->miss_count += 1u;
		status = SparkGlm52JitKvPoolStageIn(pool,fragment_ids[request_index],now_ns);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52JitKvPoolTick(SparkGlm52JitKvPool *pool,uint64_t now_ns)
{
	if ( pool == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	while (pool->transfer_count != 0u)
	{
		SparkGlm52JitKvTransfer *transfer = &pool->transfers[pool->transfer_head];
		if ( transfer->done_ns > now_ns )
			break;
		SparkGlm52JitKvPoolCompleteTransfer(pool,pool->transfer_head,now_ns);
		pool->transfer_head = ((pool->transfer_head + 1u) % SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS);
		pool->transfer_count -= 1u;
	}
	return(SPARK_STATUS_OK);
}

uint32_t SparkGlm52JitKvPoolFragmentIsResident(const SparkGlm52JitKvPool *pool,uint32_t fragment_id)
{
	if ( pool == 0 || fragment_id >= pool->fragment_capacity )
		return(0u);
	return(pool->fragments[fragment_id].state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM ? 1u : 0u);
}
