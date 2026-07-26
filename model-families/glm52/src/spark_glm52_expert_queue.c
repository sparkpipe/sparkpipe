#include "sparkpipe/spark_glm52_expert_queue.h"

#include <stddef.h>
#include <string.h>

SparkStatus SparkGlm52ExpertQueueInitialize(SparkGlm52ExpertQueue *queue,const SparkGlm52ExpertQueueConfiguration *configuration)
{
	uint32_t layer_index,expert_index;
	if ( queue == 0 || configuration == 0 ||
		configuration->abi_version != SPARK_GLM52_EXPERT_QUEUE_ABI_VERSION ||
		configuration->layer_count == 0u ||
		configuration->layer_count > SPARK_GLM52_EXPERT_QUEUE_MAX_LAYERS ||
		configuration->expert_count == 0u ||
		configuration->expert_count > SPARK_GLM52_EXPERT_QUEUE_MAX_EXPERTS ||
		configuration->firing_threshold_rows == 0u ||
		configuration->firing_threshold_rows > SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS ||
		configuration->firing_deadline_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(queue,0,offsetof(SparkGlm52ExpertQueue,rows));
	queue->abi_version = SPARK_GLM52_EXPERT_QUEUE_ABI_VERSION;
	queue->layer_count = configuration->layer_count;
	queue->expert_count = configuration->expert_count;
	queue->firing_threshold_rows = configuration->firing_threshold_rows;
	queue->firing_deadline_ns = configuration->firing_deadline_ns;
	queue->free_head = UINT32_MAX;
	queue->free_high_water = 0u;
	for (layer_index=0u; layer_index<queue->layer_count; layer_index++)
		for (expert_index=0u; expert_index<queue->expert_count; expert_index++)
		{
			queue->slots[layer_index][expert_index].head = UINT32_MAX;
			queue->slots[layer_index][expert_index].tail = UINT32_MAX;
		}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm52ExpertQueueAllocateRow(SparkGlm52ExpertQueue *queue)
{
	uint32_t row_index;
	if ( queue->free_head != UINT32_MAX )
	{
		row_index = queue->free_head;
		queue->free_head = queue->rows[row_index].list_next;
		return(row_index);
	}
	if ( queue->free_high_water < SPARK_GLM52_EXPERT_QUEUE_MAX_ROWS )
	{
		row_index = queue->free_high_water;
		queue->free_high_water += 1u;
		return(row_index);
	}
	return(UINT32_MAX);
}

static void SparkGlm52ExpertQueueReleaseRow(SparkGlm52ExpertQueue *queue,uint32_t row_index)
{
	queue->rows[row_index].list_next = queue->free_head;
	queue->free_head = row_index;
}

SparkStatus SparkGlm52ExpertQueueEnqueueRow(SparkGlm52ExpertQueue *queue,uint32_t layer_index,uint32_t expert_index,uint64_t row_id,uint64_t arrival_ns)
{
	SparkGlm52ExpertQueueSlot *slot;
	uint32_t row_index;
	if ( queue == 0 || layer_index >= queue->layer_count || expert_index >= queue->expert_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	row_index = SparkGlm52ExpertQueueAllocateRow(queue);
	if ( row_index == UINT32_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	queue->rows[row_index].row_id = row_id;
	queue->rows[row_index].arrival_ns = arrival_ns;
	queue->rows[row_index].list_next = UINT32_MAX;
	slot = &queue->slots[layer_index][expert_index];
	if ( slot->count == 0u )
	{
		slot->head = row_index;
		slot->oldest_arrival_ns = arrival_ns;
	}
	else
		queue->rows[slot->tail].list_next = row_index;
	slot->tail = row_index;
	slot->count += 1u;
	queue->enqueued_row_count += 1u;
	queue->layer_enqueued_row_count[layer_index] += 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52ExpertQueueSetFiringThreshold(SparkGlm52ExpertQueue *queue,uint32_t firing_threshold_rows)
{
	if ( queue == 0 || firing_threshold_rows == 0u ||
		firing_threshold_rows > SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	queue->firing_threshold_rows = firing_threshold_rows;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm52ExpertQueueSlotShouldFire(const SparkGlm52ExpertQueue *queue,const SparkGlm52ExpertQueueSlot *slot,uint64_t now_ns)
{
	if ( slot->count == 0u )
		return(0u);
	if ( slot->count >= queue->firing_threshold_rows )
		return(1u);
	if ( now_ns >= slot->oldest_arrival_ns &&
		(now_ns - slot->oldest_arrival_ns) >= queue->firing_deadline_ns )
		return(1u);
	return(0u);
}

SparkStatus SparkGlm52ExpertQueueNextFiring(SparkGlm52ExpertQueue *queue,uint64_t now_ns,SparkGlm52ExpertQueueFiring *firing_out)
{
	SparkGlm52ExpertQueueSlot *slot;
	uint32_t layer_index,expert_index,row_index,emit_count;
	if ( queue == 0 || firing_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (layer_index=0u; layer_index<queue->layer_count; layer_index++)
		for (expert_index=0u; expert_index<queue->expert_count; expert_index++)
		{
			slot = &queue->slots[layer_index][expert_index];
			if ( !SparkGlm52ExpertQueueSlotShouldFire(queue,slot,now_ns) )
				continue;
			emit_count = (slot->count < SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS ? slot->count : SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS);
			firing_out->layer_index = layer_index;
			firing_out->expert_index = expert_index;
			firing_out->row_count = emit_count;
			row_index = slot->head;
			for (uint32_t emit_index=0u; emit_index<emit_count; emit_index++)
			{
				uint32_t next = queue->rows[row_index].list_next;
				firing_out->row_ids[emit_index] = queue->rows[row_index].row_id;
				SparkGlm52ExpertQueueReleaseRow(queue,row_index);
				row_index = next;
			}
			slot->head = row_index;
			if ( row_index == UINT32_MAX )
				slot->tail = UINT32_MAX;
			slot->count -= emit_count;
			if ( slot->count != 0u )
				slot->oldest_arrival_ns = queue->rows[slot->head].arrival_ns;
			queue->enqueued_row_count -= emit_count;
			queue->layer_enqueued_row_count[layer_index] -= emit_count;
			queue->firing_count += 1u;
			queue->fired_row_count += emit_count;
			return(SPARK_STATUS_OK);
		}
	return(SPARK_STATUS_NOT_FOUND);
}
