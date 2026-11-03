#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_pack_window.h"

uint64_t SparkPackWindowRegionBytes(uint32_t ranks, uint32_t dense_buffers, uint64_t dense_capacity, uint32_t expert_slots, uint64_t expert_capacity)
{
	return (uint64_t)dense_buffers * ranks * dense_capacity + ((uint64_t)expert_slots + 1u) * expert_capacity;
}

SparkStatus SparkPackWindowInitialize(SparkPackWindow *window, const SparkPackWindowConfig *config)
{
	uint64_t cursor = 0u;
	uint32_t buffer,slot;
	if ( window == 0 || config == 0 || config->stream == 0 || config->region == 0 || config->ranks < 2u || config->rank >= config->ranks ||
		config->dense_buffers == 0u || config->dense_buffers > SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS ||
		(config->expert_slots == 0u ? config->expert_capacity != 0u : config->expert_slots < 2u || config->expert_slots > SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS ||
			config->expert_capacity == 0u || (config->expert_capacity % 2u) != 0u || config->read_partner == 0) ||
		config->dense_capacity == 0u || config->read_own == 0 ||
		config->region_bytes < SparkPackWindowRegionBytes(config->ranks,config->dense_buffers,config->dense_capacity,config->expert_slots,config->expert_capacity) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(window,0,sizeof(*window));
	window->config = *config;
	for (buffer=0u; buffer<config->dense_buffers; buffer++)
	{
		window->dense_base[buffer] = cursor;
		cursor += (uint64_t)config->ranks * config->dense_capacity;
	}
	for (slot=0u; slot<config->expert_slots; slot++)
	{
		window->expert_slot[slot] = cursor;
		cursor += config->expert_capacity;
	}
	window->own_staging = cursor;
	cursor += config->expert_capacity / 2u;
	window->partner_staging = cursor;
	return SPARK_STATUS_OK;
}

uint64_t SparkPackWindowDenseBytes(const SparkPackWindowLayer *layer)
{
	uint64_t bytes = 0u;
	uint32_t range;
	for (range=0u; layer != 0 && range<layer->dense_count && range<SPARK_PACK_WINDOW_MAX_DENSE_RANGES; range++)
		bytes += layer->dense[range].bytes;
	return bytes;
}

SparkStatus SparkPackWindowDense(SparkPackWindow *window, const SparkPackWindowLayer *layer, uint32_t buffer, uint32_t tag, uint8_t **rank_base)
{
	const uint64_t bytes = SparkPackWindowDenseBytes(layer);
	uint64_t placed = 0u,own;
	uint32_t range,rank;
	SparkStatus status;
	if ( window == 0 || layer == 0 || rank_base == 0 || buffer >= window->config.dense_buffers || layer->dense_count == 0u || layer->dense_count > SPARK_PACK_WINDOW_MAX_DENSE_RANGES ||
		bytes == 0u || bytes > window->config.dense_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	own = window->dense_base[buffer] + (uint64_t)window->config.rank * bytes;
	for (range=0u; range<layer->dense_count; range++)
	{
		status = window->config.read_own(window->config.own_context,layer->dense[range].offset,layer->dense[range].bytes,
			window->config.region + own + placed);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		placed += layer->dense[range].bytes;
	}
	status = SparkWeightStreamBroadcast(window->config.stream,own,bytes,window->dense_base[buffer],tag);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (rank=0u; rank<window->config.ranks; rank++)
		rank_base[rank] = window->config.region + window->dense_base[buffer] + (uint64_t)rank * bytes;
	return SPARK_STATUS_OK;
}

SparkStatus SparkPackWindowExpertStep(SparkPackWindow *window, const SparkPackWindowLayer *layer, uint32_t step, uint32_t slot_index, uint32_t tag, uint32_t *source,
	const uint8_t **block)
{
	SparkWeightStreamBlock arrived;
	uint64_t half,slot;
	SparkStatus status;
	if ( window == 0 || layer == 0 || source == 0 || block == 0 || slot_index >= window->config.expert_slots || layer->experts.bytes == 0u || (layer->experts.bytes % 2u) != 0u ||
		layer->experts.bytes > window->config.expert_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	half = layer->experts.bytes / 2u;
	if ( window->staged == 0u || window->staged_layer_offset != layer->experts.offset )
	{
		status = window->config.read_own(window->config.own_context,layer->experts.offset,half,window->config.region + window->own_staging);
		if ( status == SPARK_STATUS_OK )
			status = window->config.read_partner(window->config.partner_context,layer->experts.offset + half,half,
				window->config.region + window->partner_staging);
		if ( status != SPARK_STATUS_OK )
		{
			window->staged = 0u;
			SPARK_RETURN(status);
		}
		window->staged = 1u;
		window->staged_layer_offset = layer->experts.offset;
	}
	slot = window->expert_slot[slot_index];
	status = SparkWeightStreamRotate(window->config.stream,step,window->own_staging,window->partner_staging - half,layer->experts.bytes,slot,tag,&arrived);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	*source = arrived.source;
	if ( arrived.source == window->config.rank )
	{
		*block = 0;
		return SPARK_STATUS_OK;
	}
	if ( arrived.half_offset[1] != slot + half )
		memcpy(window->config.region + slot + half,window->config.region + arrived.half_offset[1],half);
	*block = window->config.region + slot;
	return SPARK_STATUS_OK;
}
