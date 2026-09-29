#pragma once

#include <stdint.h>

#include "sparkpipe/spark_k3_llm_defines.h"

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
	return(SPARK_K3_MODEL_LAYER_IS_MLA(layer_index));
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

#define SPARK_K3_SLICE_NO_LAYER UINT32_MAX

typedef struct SparkK3SliceKindLayers
{
	uint32_t kda;
	uint32_t mla;
	uint32_t routed;
	uint32_t dense;
} SparkK3SliceKindLayers;

static inline void SparkK3SliceKindLayersFor(uint32_t first_layer,
	uint32_t layer_count, SparkK3SliceKindLayers *kinds)
{
	kinds->kda = SPARK_K3_SLICE_NO_LAYER;
	kinds->mla = SPARK_K3_SLICE_NO_LAYER;
	kinds->routed = SPARK_K3_SLICE_NO_LAYER;
	kinds->dense = SPARK_K3_SLICE_NO_LAYER;
	for ( uint32_t layer = first_layer; layer < first_layer + layer_count; layer++ )
	{
		if ( SparkK3LayerIsMla(layer) )
		{
			if ( kinds->mla == SPARK_K3_SLICE_NO_LAYER )
				kinds->mla = layer;
		}
		else if ( kinds->kda == SPARK_K3_SLICE_NO_LAYER )
			kinds->kda = layer;
		if ( layer < SPARK_K3_MODEL_FIRST_ROUTED_LAYER )
		{
			if ( kinds->dense == SPARK_K3_SLICE_NO_LAYER )
				kinds->dense = layer;
		}
		else if ( kinds->routed == SPARK_K3_SLICE_NO_LAYER )
			kinds->routed = layer;
	}
}

static inline void SparkK3PoolSizingForSlice(uint32_t first_layer,
	uint32_t layer_count, SparkK3PoolSizing *sizing)
{
	uint32_t mla = SparkK3MlaLayersInSlice(first_layer, layer_count);
	uint32_t kda = layer_count - mla;
	const uint64_t kda_state_per_layer = SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER;
	const uint64_t kda_conv_per_layer =
		SPARK_K3_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER;
	const uint64_t mla_entry_per_layer =
		(uint64_t)SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS *
		SPARK_K3_KV_BYTES_PER_SCALAR;
	sizing->first_layer = first_layer;
	sizing->layer_count = layer_count;
	sizing->mla_layer_count = mla;
	sizing->kda_layer_count = kda;
	sizing->kda_slot_bytes_per_sequence =
		(uint64_t)kda * (kda_state_per_layer + kda_conv_per_layer);
	sizing->mla_bytes_per_token = (uint64_t)mla * mla_entry_per_layer;
}

#define SPARK_K3_SLOT_POOL_STATE 0u
#define SPARK_K3_SLOT_POOL_Q_WINDOW 1u
#define SPARK_K3_SLOT_POOL_K_WINDOW 2u
#define SPARK_K3_SLOT_POOL_V_WINDOW 3u
#define SPARK_K3_SLOT_POOLS 4u
#define SPARK_K3_SLOT_RESET_SPANS_MAX \
	(SPARK_K3_MODEL_KDA_LAYER_COUNT * SPARK_K3_SLOT_POOLS)

typedef struct SparkK3SlotResetSpan
{
	uint32_t pool;
	uint64_t offset;
	uint64_t bytes;
} SparkK3SlotResetSpan;

static inline uint32_t SparkK3SlotResetSpans(uint32_t kda_layer_count,
	uint32_t sequences, uint32_t tp_degree, uint32_t slot,
	SparkK3SlotResetSpan *spans, uint32_t capacity)
{
	const uint64_t state_slot = SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER;
	const uint64_t kernel = SPARK_K3_MODEL_KDA_CONV_KERNEL;
	const uint64_t scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	uint64_t rank_qk, rank_v, qk_layer, v_layer;
	uint32_t count = 0u;
	if ( spans == 0 || sequences == 0u || slot >= sequences || tp_degree == 0u ||
		SPARK_K3_MODEL_KDA_HEAD_COUNT % tp_degree != 0u ||
		kda_layer_count > SPARK_K3_MODEL_KDA_LAYER_COUNT ||
		capacity < kda_layer_count * SPARK_K3_SLOT_POOLS )
		return(0u);
	rank_qk = (uint64_t)(SPARK_K3_MODEL_KDA_HEAD_COUNT / tp_degree) *
		SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION;
	rank_v = (uint64_t)(SPARK_K3_MODEL_KDA_HEAD_COUNT / tp_degree) *
		SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION;
	qk_layer = (uint64_t)sequences * SPARK_K3_MODEL_KDA_QK_DIMENSION * kernel * scalar;
	v_layer = (uint64_t)sequences * SPARK_K3_MODEL_KDA_VALUE_DIMENSION * kernel * scalar;
	for ( uint32_t layer = 0u; layer < kda_layer_count; layer++ )
	{
		spans[count].pool = SPARK_K3_SLOT_POOL_STATE;
		spans[count].offset = ((uint64_t)layer * sequences + slot) * state_slot;
		spans[count++].bytes = state_slot;
		spans[count].pool = SPARK_K3_SLOT_POOL_Q_WINDOW;
		spans[count].offset = (uint64_t)layer * qk_layer + (uint64_t)slot * rank_qk * kernel * scalar;
		spans[count++].bytes = rank_qk * kernel * scalar;
		spans[count].pool = SPARK_K3_SLOT_POOL_K_WINDOW;
		spans[count].offset = (uint64_t)layer * qk_layer + (uint64_t)slot * rank_qk * kernel * scalar;
		spans[count++].bytes = rank_qk * kernel * scalar;
		spans[count].pool = SPARK_K3_SLOT_POOL_V_WINDOW;
		spans[count].offset = (uint64_t)layer * v_layer + (uint64_t)slot * rank_v * kernel * scalar;
		spans[count++].bytes = rank_v * kernel * scalar;
	}
	return(count);
}
