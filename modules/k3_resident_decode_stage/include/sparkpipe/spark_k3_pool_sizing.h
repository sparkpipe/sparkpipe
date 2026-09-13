#ifndef SPARKPIPE_SPARK_K3_POOL_SIZING_H
#define SPARKPIPE_SPARK_K3_POOL_SIZING_H

#include <stdint.h>

/*
 * K3 per-rank pool sizing for a PP slice. The full-model constants live in
 * model-families/k3/spark_k3_kv_geometry.h; this header scales them by the
 * rank's layer slice so every spark allocates only its 1/16 share (PP4
 * quarters the layer counts, and the MLA latent replicates inside the TP
 * group by design - the rank-local saving is the slice, not a TP split).
 */

/*
 * THE SLICE GEOMETRY, SPELLED ONCE (driver-audit D2). The backbone length
 * and the PP4 stage split used to exist as independent literal sets (the
 * runner tables, the adapter descriptor, the module validation, docs); the
 * layer-kind divergence that bit layer 92 came from exactly this
 * duplication class. Consumers derive from these two names:
 */
#define SPARK_K3_TOTAL_LAYERS 93u
#define SPARK_K3_PP4_STAGE_LAYER_COUNTS_TUPLE 24u, 23u, 23u, 23u

/* Layer count of PP stage (stage_index % 4) under the TP4xPP4 split. */
static inline uint32_t SparkK3StageLayerCount(uint32_t stage_index)
{
	static const uint32_t counts[4] =
		{ SPARK_K3_PP4_STAGE_LAYER_COUNTS_TUPLE };
	return(counts[stage_index % 4u]);
}

/* First global layer index of PP stage stage_index - the prefix sum of
 * the same table, so no second copy of the offsets can drift. */
static inline uint32_t SparkK3StageFirstLayer(uint32_t stage_index)
{
	uint32_t first = 0u;
	uint32_t s;
	for (s = 0u; s < stage_index && s < 4u; ++s)
		first += SparkK3StageLayerCount(s);
	return(first);
}

typedef struct SparkK3PoolSizing
{
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t mla_layer_count;
	uint32_t kda_layer_count;
	uint64_t kda_slot_bytes_per_sequence;   /* the slice's KDA slabs */
	uint64_t mla_bytes_per_token;           /* the slice's MLA arena entry */
} SparkK3PoolSizing;

static inline uint32_t SparkK3LayerIsMla(uint32_t layer_index)
{
	/* 3:1 period puts MLA at 3, 7, 11, ... AND the backbone's trailing
	 * layer 92 is MLA (checkpoint full_attn_layers includes 93 one-indexed). */
	return((layer_index % 4u) == 3u || layer_index == 92u);
}

/* Counts per slice: 24 MLA (23 periodic + the trailing 92) and 69 KDA. */
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
	/* Per-layer sizes are the kv-geometry constants; the drift gate
	 * (tests/test_k3_kv_geometry.py) holds them equal to the kernel config. */
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
