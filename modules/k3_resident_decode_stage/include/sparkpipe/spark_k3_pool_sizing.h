#ifndef SPARKPIPE_SPARK_K3_POOL_SIZING_H
#define SPARKPIPE_SPARK_K3_POOL_SIZING_H

#include <stdint.h>


typedef struct SparkK3PoolSizing
{
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t mla_layer_count;
	uint32_t kda_layer_count;
	uint64_t kda_slot_bytes_per_sequence;
	uint64_t mla_bytes_per_token;
} SparkK3PoolSizing;

static inline uint32_t SparkK3LayerIsMla(uint32_t layer_index)
{
	return((layer_index % 4u) == 3u || layer_index == 92u);
}

static inline uint32_t SparkK3MlaLayersInSlice(uint32_t first_layer,
	uint32_t layer_count)
{
	uint32_t count = 0u;
	for ( uint32_t layer = first_layer; layer < first_layer + layer_count; layer++ )
		if ( SparkK3LayerIsMla(layer) )
			count++;
	return(count);
}

static inline void SparkK3PoolSizingForSlice(uint32_t first_layer,
	uint32_t layer_count, SparkK3PoolSizing *sizing)
{
	uint32_t mla = SparkK3MlaLayersInSlice(first_layer, layer_count);
	uint32_t kda = layer_count - mla;
	const uint64_t kda_state_per_layer = 96ull * 128u * 128u * 4u;
	const uint64_t kda_conv_per_layer =
		((2ull * 96u * 128u) + (96ull * 128u)) * 4u * 2u;
	const uint64_t mla_entry_per_layer = (512u + 64u) * 2u;
	sizing->first_layer = first_layer;
	sizing->layer_count = layer_count;
	sizing->mla_layer_count = mla;
	sizing->kda_layer_count = kda;
	sizing->kda_slot_bytes_per_sequence =
		(uint64_t)kda * (kda_state_per_layer + kda_conv_per_layer);
	sizing->mla_bytes_per_token = (uint64_t)mla * mla_entry_per_layer;
}

#endif
