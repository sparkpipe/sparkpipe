#pragma once

#include <stdint.h>

#define SPARK_HYBRID_KV_TENSORS_PER_SLOT 2u
#define SPARK_HYBRID_ORDINAL_UNASSIGNED UINT32_MAX

#define SPARK_HYBRID_LAYER_IS_PHASE(layer_index,period,phase) \
	(((layer_index) % (period)) == (phase))
#define SPARK_HYBRID_PHASE_CLASS_COUNT(layer_count,period) \
	((layer_count) / (period))
#define SPARK_HYBRID_WINDOW_CLASS_COUNT(layer_count,period) \
	((layer_count) - SPARK_HYBRID_PHASE_CLASS_COUNT(layer_count,period))

#define SPARK_HYBRID_KV_HEAD_SLOT_BYTES(head_dimension,element_bytes) \
	((uint64_t)(head_dimension) * SPARK_HYBRID_KV_TENSORS_PER_SLOT * (uint64_t)(element_bytes))
#define SPARK_HYBRID_KV_SLOT_BYTES(kv_head_count,head_dimension,element_bytes) \
	((uint64_t)(kv_head_count) * SPARK_HYBRID_KV_HEAD_SLOT_BYTES(head_dimension,element_bytes))
#define SPARK_HYBRID_KV_POOL_BYTES(block_count,block_tokens,slot_bytes) \
	((uint64_t)(block_count) * (uint64_t)(block_tokens) * (slot_bytes))

#define SPARK_HYBRID_KDA_STATE_BYTES_PER_LAYER(head_count,key_dimension,value_dimension,state_element_bytes) \
	((uint64_t)(head_count) * (uint64_t)(key_dimension) * (uint64_t)(value_dimension) * (uint64_t)(state_element_bytes))
#define SPARK_HYBRID_KDA_CONV_BYTES_PER_LAYER(head_count,key_dimension,value_dimension,conv_kernel,conv_element_bytes) \
	(((uint64_t)(head_count) * ((uint64_t)(key_dimension) * SPARK_HYBRID_KV_TENSORS_PER_SLOT + (uint64_t)(value_dimension))) * (uint64_t)(conv_kernel) * (uint64_t)(conv_element_bytes))
#define SPARK_HYBRID_KDA_SLOT_BYTES(layer_count,head_count,key_dimension,value_dimension,conv_kernel,state_element_bytes,conv_element_bytes) \
	((uint64_t)(layer_count) * (SPARK_HYBRID_KDA_STATE_BYTES_PER_LAYER(head_count,key_dimension,value_dimension,state_element_bytes) + SPARK_HYBRID_KDA_CONV_BYTES_PER_LAYER(head_count,key_dimension,value_dimension,conv_kernel,conv_element_bytes)))

static inline void SparkHybridStateBuildOrdinals(uint32_t *phase_ordinal_by_layer, uint32_t *window_ordinal_by_layer, uint32_t *phase_count, uint32_t *window_count, uint32_t total_layer_count, uint32_t first_layer_index, uint32_t layer_count, uint32_t period, uint32_t phase)
{
	uint32_t layer;
	for (layer = 0u; layer < total_layer_count; layer++)
	{
		phase_ordinal_by_layer[layer] = SPARK_HYBRID_ORDINAL_UNASSIGNED;
		window_ordinal_by_layer[layer] = SPARK_HYBRID_ORDINAL_UNASSIGNED;
	}
	for (layer = first_layer_index; layer < first_layer_index + layer_count; layer++)
	{
		if ( SPARK_HYBRID_LAYER_IS_PHASE(layer,period,phase) != 0u )
			phase_ordinal_by_layer[layer] = (*phase_count)++;
		else
			window_ordinal_by_layer[layer] = (*window_count)++;
	}
}
