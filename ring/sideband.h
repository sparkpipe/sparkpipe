#pragma once


#include <stdint.h>

#define LM_SIDEBAND_OK 0
#define LM_SIDEBAND_ERR_SHAPE (-91)
#define LM_SIDEBAND_ERR_ABSENT (-92)
#define LM_SIDEBAND_ERR_STALE (-93)
#define LM_SIDEBAND_ERR_CAPACITY (-94)

typedef enum LmSidebandKind
{
	LM_SIDEBAND_NONE = 0,
	LM_SIDEBAND_INDEX_SHARE = 1,
	LM_SIDEBAND_HIDDEN_TAP = 2,
	LM_SIDEBAND_PREFIX_INDICES = 3
}
LmSidebandKind;

typedef struct LmSidebandHeader
{
	uint32_t kind;
	uint32_t source_rank;
	uint32_t source_layer;
	uint32_t generation;
	uint32_t row_count;
	uint32_t element_count;
	uint32_t element_bytes;
	uint32_t reserved;
}
LmSidebandHeader;

typedef struct LmTapPlan
{
	uint32_t tap_layer[8];
	uint32_t tap_count;
	uint32_t consumer_rank;
}
LmTapPlan;

static int32_t LmSidebandPayloadBytes(const LmSidebandHeader *header, uint64_t *bytes_out)
{
	uint64_t bytes;
	if ( header == 0 || bytes_out == 0 )
		return(LM_SIDEBAND_ERR_SHAPE);
	if ( header->element_bytes == 0u || header->element_count == 0u )
		return(LM_SIDEBAND_ERR_SHAPE);
	bytes = (uint64_t)header->row_count * header->element_count * header->element_bytes;
	if ( bytes > (uint64_t)1u << 32 )
		return(LM_SIDEBAND_ERR_CAPACITY);
	*bytes_out = bytes;
	return(LM_SIDEBAND_OK);
}

static int32_t LmSidebandAcceptable(const LmSidebandHeader *header, uint32_t want_kind, uint32_t current_generation, uint32_t consumer_layer, uint32_t share_group_layers)
{
	if ( header == 0 || header->kind == LM_SIDEBAND_NONE )
		return(LM_SIDEBAND_ERR_ABSENT);
	if ( header->kind != want_kind )
		return(LM_SIDEBAND_ERR_SHAPE);
	if ( header->generation != current_generation )
		return(LM_SIDEBAND_ERR_STALE);
	if ( header->kind == LM_SIDEBAND_INDEX_SHARE && share_group_layers != 0u )
	{
		if ( (header->source_layer / share_group_layers)
			!= (consumer_layer / share_group_layers) )
			return(LM_SIDEBAND_ERR_STALE);
	}
	return(LM_SIDEBAND_OK);
}

static int32_t LmSidebandShouldExportIndexShare(uint32_t last_layer_on_rank, uint32_t share_group_layers)
{
	if ( share_group_layers == 0u )
		return(0);
	return(((last_layer_on_rank + 1u) % share_group_layers) != 0u);
}

static uint32_t LmSidebandTapsOnRank(const LmTapPlan *plan, uint32_t first_layer, uint32_t layer_count, uint32_t *local_out, uint32_t capacity)
{
	uint32_t index,found = 0u;
	if ( plan == 0 || local_out == 0 )
		return(0u);
	for (index = 0u; index < plan->tap_count && index < 8u; ++index)
	{
		uint32_t layer = plan->tap_layer[index];
		if ( layer < first_layer || layer >= first_layer + layer_count )
			continue;
		if ( found >= capacity )
			break;
		local_out[found++] = layer - first_layer;
	}
	return(found);
}

static int32_t LmSidebandRankDrafts(const LmTapPlan *plan, uint32_t rank)
{
	return(plan != 0 && plan->consumer_rank == rank);
}

static void LmSidebandInitialiseHeader(LmSidebandHeader *header, uint32_t kind, uint32_t rank, uint32_t layer, uint32_t generation, uint32_t rows, uint32_t elements, uint32_t element_bytes)
{
	header->kind = kind;
	header->source_rank = rank;
	header->source_layer = layer;
	header->generation = generation;
	header->row_count = rows;
	header->element_count = elements;
	header->element_bytes = element_bytes;
	header->reserved = 0u;
}
