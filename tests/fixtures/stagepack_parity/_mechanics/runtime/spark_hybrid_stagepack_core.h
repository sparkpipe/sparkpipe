#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stage-pack mechanics shared by the hybrid GDN/attention stage families,
 * whose packs share one design: a slice-first header restating compiled
 * geometry, a directory of (kind, layer) entries with declared byte counts,
 * computed inventories over a 3:1 attention period map, and the
 * global/every-layer/GDN/attention layer-class system.
 *
 * The WIRE FORMATS stay per family (own magics, header layouts, entry
 * structs and weight-format enum values); what lives here is only what the
 * adopting loaders accept IDENTICALLY today: byte accounting for their six
 * weight classes (the e8m0-tiled fp8 class arrived with a family's MX serving
 * pack), the scale-group rule, the period arithmetic behind the
 * inventory counts, the layer-class resolution tail with its exact refusal
 * codes, and u32-prefix header comparison. A family adopts a piece by
 * forwarding its historical function name; where a family's on-disk
 * semantics genuinely diverge (an unchecked compressed-text class), the
 * local policy stays local - this header never widens or narrows a check a
 * driver performed.
 */

#define SPARK_HYBRID_STAGEPACK_CLASS_GLOBAL 0u
#define SPARK_HYBRID_STAGEPACK_CLASS_EVERY_LAYER 1u
#define SPARK_HYBRID_STAGEPACK_CLASS_GDN_LAYER 2u
#define SPARK_HYBRID_STAGEPACK_CLASS_ATTN_LAYER 3u

/* Normalized weight classes of the two families' entry formats. */
#define SPARK_HYBRID_STAGEPACK_WEIGHT_BF16 0u
#define SPARK_HYBRID_STAGEPACK_WEIGHT_F32 1u
#define SPARK_HYBRID_STAGEPACK_WEIGHT_U32 2u
#define SPARK_HYBRID_STAGEPACK_WEIGHT_MXFP4_E2M1 3u
#define SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_F32B128 4u
#define SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_E8M0B128 5u

/* Full-attention layers among the first n of the stack: one phase in the
 * shared period puts them at period-1, 2*period-1, ..., so the count is
 * simply n / period. Both inventories are built from this difference. */
static inline uint32_t SparkHybridStagePackFullAttentionLayersBelow(uint32_t attention_period,uint32_t layer_count)
{
	return(layer_count / attention_period);
}

/* Payload bytes for one tensor of the normalized class. MXFP4 packs two
 * nibbles per byte; fp8 is one byte per element; f32/u32 four; bf16 two. */
static inline uint64_t SparkHybridStagePackPayloadBytes(uint32_t weight_class,uint32_t rows,uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_MXFP4_E2M1 )
		return(elements / 2u);
	if ( weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_F32 || weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_U32 )
		return(elements * 4u);
	if ( weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_F32B128 ||
		weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_E8M0B128 )
		return(elements);
	return(elements * 2u);
}

/* Scale-plane bytes: mxfp4 carries one e8m0 byte per 32-element group; fp8
 * e4m3 carries one f32 scale per 128x128 tile of the 2-D plane; the e8m0-
 * tiled fp8 class carries one e8m0 byte per row per 128-column group.
 * Anything else travels scale-free. */
static inline uint64_t SparkHybridStagePackScaleBytes(uint32_t weight_class,uint32_t rows,uint32_t columns)
{
	if ( weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / 32u);
	if ( weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_F32B128 )
		return(((uint64_t)rows / 128u) * ((uint64_t)columns / 128u) * 4u);
	if ( weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_E8M0B128 )
		return((uint64_t)rows * ((uint64_t)columns / 128u));
	return(0u);
}

/* The entry's declared scale_group_size must equal the class constant:
 * 32 for mxfp4, 128 for fp8 tiles, zero for scale-free classes. */
static inline uint32_t SparkHybridStagePackScaleGroupSizeOk(uint32_t weight_class,uint32_t declared_group_size)
{
	uint32_t expected = weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_MXFP4_E2M1 ? 32u : ((weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_F32B128 || weight_class == SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_E8M0B128) ? 128u : 0u);
	return(declared_group_size == expected ? 1u : 0u);
}

/*
 * Layer-class resolution tail, shared verbatim by both families' resolvers.
 * Precedence and codes are the contract: at the reserved MTP marker only
 * every-layer and attention kinds may ride per-layer (else -6); a global
 * kind must carry the global marker and vice versa (-2); globals pass;
 * per-layer kinds sit inside the stack (-3) and their class must agree
 * with the hybrid layer map (-4 GDN, -5 attention).
 */
static inline int32_t SparkHybridStagePackResolveLayerClass(uint32_t layer_class,uint32_t is_global,uint32_t at_mtp_marker,uint32_t layer_index,uint32_t total_layer_count,uint32_t layer_is_gdn)
{
	if ( at_mtp_marker != 0u )
		return((is_global == 0u && (layer_class == SPARK_HYBRID_STAGEPACK_CLASS_EVERY_LAYER || layer_class == SPARK_HYBRID_STAGEPACK_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (layer_class == SPARK_HYBRID_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= total_layer_count )
		return(-3);
	if ( layer_class == SPARK_HYBRID_STAGEPACK_CLASS_GDN_LAYER && layer_is_gdn == 0u )
		return(-4);
	if ( layer_class == SPARK_HYBRID_STAGEPACK_CLASS_ATTN_LAYER && layer_is_gdn != 0u )
		return(-5);
	return(0);
}

/*
 * Header comparison as one walk over the contiguous u32 prefix (every field
 * up to the trailing u64 offsets): first mismatch wins with code -(i+1),
 * so each field owns a unique negative code naming exactly what disagreed.
 * Field-name tables stay per family alongside their headers.
 */
static inline int32_t SparkHybridStagePackHeaderFieldsMatch(const void *file_header,const void *expected_header,uint32_t u32_field_count)
{
	const uint32_t *file_fields = (const uint32_t *)file_header;
	const uint32_t *expected_fields = (const uint32_t *)expected_header;
	uint32_t i;
	for ( i = 0u; i < u32_field_count; i++ )
		if ( file_fields[i] != expected_fields[i] )
			return(-(int32_t)(i + 1u));
	return(0);
}

#ifdef __cplusplus
}
#endif
