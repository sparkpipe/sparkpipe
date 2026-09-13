#pragma once

#include <stdint.h>

#include "sparkpipe/spark_weight_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Model-neutral stage-pack reader mechanics.
 *
 * Every resident decode stage loads a content-addressed pack through the
 * same chain: header check, directory read, per-entry shape resolution and
 * byte accounting, duplicate-bitmask bookkeeping, range-overlap and bounds
 * checks, inventory comparison. The WIRE FORMATS stay per model (each
 * family's header/entry structs are its own contract); what is shared here
 * is the mechanics around them, parameterized by a normalized tensor shape
 * and by the model's geometry table, which resolves a (tensor kind, layer)
 * pair into that shape plus its row/column sharding policy.
 *
 * A model adopts the piece that matches its behavior exactly. Where a
 * family's on-disk semantics genuinely differ - scale-plane geometry,
 * unchecked payload classes, computed-vs-declared inventories - the local
 * policy stays local until a format-generation change makes the behaviors
 * converge; this header never widens a check a driver did not perform.
 */

#define SPARK_STAGE_PACK_PAYLOAD_BF16 1u
#define SPARK_STAGE_PACK_PAYLOAD_F32 2u
#define SPARK_STAGE_PACK_PAYLOAD_U32 3u
#define SPARK_STAGE_PACK_PAYLOAD_PACKED_WEIGHT 4u

/* The normalized shape every geometry table emits. */
typedef struct SparkStagePackShape
{
	uint32_t payload_type;
	uint32_t weight_codec;      /* SparkWeightCodec; meaningful when packed */
	uint32_t scale_encoding;    /* SparkWeightScaleEncoding; NONE unless packed */
	uint32_t group_count;
	uint32_t rows;
	uint32_t columns;
} SparkStagePackShape;

static inline void SparkStagePackShapeBf16(SparkStagePackShape *shape,uint32_t groups,uint32_t rows,uint32_t columns)
{
	shape->payload_type = SPARK_STAGE_PACK_PAYLOAD_BF16;
	shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
	shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
	shape->group_count = groups;
	shape->rows = rows;
	shape->columns = columns;
}

static inline void SparkStagePackShapeWords(SparkStagePackShape *shape,uint32_t payload_type,uint32_t groups,uint32_t rows,uint32_t columns)
{
	shape->payload_type = payload_type;
	shape->weight_codec = SPARK_WEIGHT_CODEC_BF16;
	shape->scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_NONE;
	shape->group_count = groups;
	shape->rows = rows;
	shape->columns = columns;
}

/* Packed-expert shape: codec supplies scale encoding and grouping. */
static inline int32_t SparkStagePackShapePacked(SparkStagePackShape *shape,uint32_t codec,uint32_t groups,uint32_t rows,uint32_t columns)
{
	if ( SparkWeightCodecIsKnown(codec) == 0u || codec == SPARK_WEIGHT_CODEC_BF16 || groups == 0u || rows == 0u || columns == 0u )
		return(-1);
	shape->payload_type = SPARK_STAGE_PACK_PAYLOAD_PACKED_WEIGHT;
	shape->weight_codec = codec;
	shape->scale_encoding = SparkWeightCodecScaleEncoding(codec);
	shape->group_count = groups;
	shape->rows = rows;
	shape->columns = columns;
	return(0);
}

static inline uint64_t SparkStagePackPayloadBytes(const SparkStagePackShape *shape)
{
	uint64_t elements;
	if ( shape == 0 || shape->group_count == 0u || shape->rows == 0u || shape->columns == 0u || shape->group_count > UINT64_MAX / shape->rows || (uint64_t)shape->group_count * shape->rows > UINT64_MAX / shape->columns )
		return(0u);
	elements = (uint64_t)shape->group_count * shape->rows * shape->columns;
	switch ( shape->payload_type )
	{
	case SPARK_STAGE_PACK_PAYLOAD_BF16: return(elements > UINT64_MAX / 2u ? 0u : elements * 2u);
	case SPARK_STAGE_PACK_PAYLOAD_F32:
	case SPARK_STAGE_PACK_PAYLOAD_U32: return(elements > UINT64_MAX / 4u ? 0u : elements * 4u);
	default: return(shape->payload_type == SPARK_STAGE_PACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecPayloadBytes(shape->weight_codec,(uint64_t)shape->group_count * shape->rows,shape->columns) : 0u);
	}
}

static inline uint64_t SparkStagePackScaleBytes(const SparkStagePackShape *shape)
{
	return(shape != 0 && shape->payload_type == SPARK_STAGE_PACK_PAYLOAD_PACKED_WEIGHT ? SparkWeightCodecScaleBytes(shape->weight_codec,shape->group_count,shape->rows,shape->columns) : 0u);
}

/* Row/column sharding policy application: a table declares which kinds
 * shrink along which axis for a TP degree, and the mechanics enforce
 * divisibility before dividing. Returns 0 or a negative schema code. */
#define SPARK_STAGE_PACK_SHARD_NONE 0u
#define SPARK_STAGE_PACK_SHARD_ROWS 1u
#define SPARK_STAGE_PACK_SHARD_COLUMNS 2u

static inline int32_t SparkStagePackApplyShard(SparkStagePackShape *shape,uint32_t shard_policy,uint32_t tp_degree)
{
	if ( shape == 0 || tp_degree == 0u )
		return(-1);
	if ( shard_policy == SPARK_STAGE_PACK_SHARD_NONE )
		return(0);
	if ( shard_policy == SPARK_STAGE_PACK_SHARD_ROWS )
	{
		if ( shape->rows == 0u || shape->rows % tp_degree != 0u )
			return(-2);
		shape->rows /= tp_degree;
		return(0);
	}
	if ( shard_policy == SPARK_STAGE_PACK_SHARD_COLUMNS )
	{
		if ( shape->columns == 0u || shape->columns % tp_degree != 0u )
			return(-2);
		shape->columns /= tp_degree;
		return(0);
	}
	return(-3);
}

/* Duplicate detection over the seen bitmaps: returns 1 when the kind was
 * already recorded (a duplicate entry), and records it otherwise. */
static inline uint32_t SparkStagePackSeenHas(uint64_t seen,uint32_t tensor_kind)
{
	return(tensor_kind < 64u && (seen & (UINT64_C(1) << tensor_kind)) != 0u ? 1u : 0u);
}

static inline void SparkStagePackSeenMark(uint64_t *seen,uint32_t tensor_kind)
{
	if ( seen != 0 && tensor_kind < 64u )
		*seen |= UINT64_C(1) << tensor_kind;
}

/* Pairwise range overlap between two [offset, offset+bytes) spans. Empty
 * spans never overlap. Written in subtraction form so hostile u64 offsets
 * and byte counts can never wrap the additions into a false "no overlap":
 * for every input where both spans fit inside any common window (the only
 * case callers reach here, after file-fit validation) this returns exactly
 * what the naive addition form returns. */
static inline uint32_t SparkStagePackRangesOverlap(uint64_t left_offset,uint64_t left_bytes,uint64_t right_offset,uint64_t right_bytes)
{
	if ( left_bytes == 0u || right_bytes == 0u )
		return(0u);
	if ( left_offset <= right_offset )
		return(right_offset - left_offset < left_bytes ? 1u : 0u);
	return(left_offset - right_offset < right_bytes ? 1u : 0u);
}

/*
 * Entry placement against one file window: both the payload and the scale
 * span must be aligned, start at or after `minimum_offset` (the directory
 * end), and fit inside `file_bytes`. A zero-byte scale must sit at offset
 * zero. This is the strictest placement discipline in the tree; a driver
 * whose wire format relaxes any clause keeps its local checker.
 */
static inline int32_t SparkStagePackCheckEntryBounds(uint32_t alignment_bytes,const SparkStagePackShape *shape,uint64_t minimum_offset,uint64_t file_bytes,uint64_t payload_offset,uint64_t payload_bytes,uint64_t scale_offset,uint64_t scale_bytes)
{
	uint64_t expected_payload,expected_scale;
	if ( alignment_bytes == 0u || shape == 0 || file_bytes < minimum_offset )
		return(-1);
	expected_payload = SparkStagePackPayloadBytes(shape);
	expected_scale = SparkStagePackScaleBytes(shape);
	if ( expected_payload == 0u || payload_bytes != expected_payload || scale_bytes != expected_scale )
		return(-2);
	if ( payload_offset % alignment_bytes != 0u || payload_offset < minimum_offset || payload_offset > file_bytes || payload_bytes > file_bytes - payload_offset )
		return(-3);
	if ( expected_scale == 0u )
	{
		if ( scale_offset != 0u )
			return(-4);
	}
	else if ( scale_offset % alignment_bytes != 0u || scale_offset < minimum_offset || scale_offset > file_bytes || scale_bytes > file_bytes - scale_offset )
		return(-5);
	return(0);
}

/*
 * Expected-inventory mask construction: for one layer, set the bit of every
 * kind whose geometry table resolves. `expected_shape` is the model's table
 * entry point; `context` carries whatever the table reads (codec, degree).
 * Kinds at or beyond 64 are skipped (they cannot ride a u64 mask); every
 * current family's kind space fits far below that bound.
 */
typedef int32_t (*SparkStagePackShapeResolver)(void *context,uint32_t tensor_kind,uint32_t layer_index,SparkStagePackShape *shape);

static inline uint64_t SparkStagePackExpectedLayerMask(void *resolver_context,SparkStagePackShapeResolver expected_shape,uint32_t first_tensor_kind,uint32_t tensor_kind_count,uint32_t layer_index)
{
	SparkStagePackShape shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind = first_tensor_kind; kind < tensor_kind_count && kind < 64u; kind++)
		if ( expected_shape(resolver_context,kind,layer_index,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

/*
 * Flat row-by-column entry accounting - the second pack dialect beside the
 * packed shapes above: directory entries carry (rows,columns,wire format),
 * one payload density class plus one scale-plane geometry per format, both
 * family-table DATA. Classes keep the flat dialect's ids (BF16 2 B/element,
 * F32/U32 4, FP8 1, NIBBLE 2 elements/B). The wrapping u64 arithmetic and
 * absent guards are the dialect's historical hostile-entry contract.
 */
#define SPARK_STAGE_PACK_WEIGHT_BF16 0u
#define SPARK_STAGE_PACK_WEIGHT_F32 1u
#define SPARK_STAGE_PACK_WEIGHT_U32 2u
#define SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1 3u
#define SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128 4u
#define SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128 5u
#define SPARK_STAGE_PACK_WEIGHT_FP8_E4M3 6u

/* Scale planes: NONE scale-free; FLAT_GROUP32 one e8m0 per 32 flattened
 * elements; TILE_128X128_F32 one f32 per 128x128 plane tile; PER_ROW_BLOCK
 * one byte per block-column group of every row, floor or ceiling. Blocks
 * ride as log2 so a table row stays plain bytes. */
#define SPARK_STAGE_PACK_SCALE_PLANE_NONE 0u
#define SPARK_STAGE_PACK_SCALE_PLANE_FLAT_GROUP32 1u
#define SPARK_STAGE_PACK_SCALE_PLANE_TILE_128X128_F32 2u
#define SPARK_STAGE_PACK_SCALE_PLANE_PER_ROW_BLOCK 3u
#define SPARK_STAGE_PACK_SCALE_ROUND_FLOOR 0u
#define SPARK_STAGE_PACK_SCALE_ROUND_CEIL 1u

typedef struct SparkStagePackWeightAccounting
{
	uint8_t payload_class;     /* SPARK_STAGE_PACK_WEIGHT_* density */
	uint8_t scale_plane;       /* SPARK_STAGE_PACK_SCALE_PLANE_* */
	uint8_t scale_block_log2;  /* PER_ROW_BLOCK block = 1 << log2 */
	uint8_t scale_round;       /* SPARK_STAGE_PACK_SCALE_ROUND_* */
	uint8_t scale_group_size;  /* scale_group_size value the loader requires */
} SparkStagePackWeightAccounting;

static inline uint64_t SparkStagePackWeightPayloadBytes(uint32_t payload_class,uint32_t rows,uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( payload_class == SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1 )
		return(elements / 2u);
	if ( payload_class == SPARK_STAGE_PACK_WEIGHT_F32 || payload_class == SPARK_STAGE_PACK_WEIGHT_U32 )
		return(elements * 4u);
	if ( payload_class == SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128 || payload_class == SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128 || payload_class == SPARK_STAGE_PACK_WEIGHT_FP8_E4M3 )
		return(elements);
	return(elements * 2u);
}

static inline uint64_t SparkStagePackWeightScaleBytes(const SparkStagePackWeightAccounting *accounting,uint32_t rows,uint32_t columns)
{
	if ( accounting == 0 || accounting->scale_plane == SPARK_STAGE_PACK_SCALE_PLANE_NONE )
		return(0u);
	if ( accounting->scale_plane == SPARK_STAGE_PACK_SCALE_PLANE_FLAT_GROUP32 )
		return(((uint64_t)rows * (uint64_t)columns) / 32u);
	if ( accounting->scale_plane == SPARK_STAGE_PACK_SCALE_PLANE_TILE_128X128_F32 )
		return(((uint64_t)rows / 128u) * ((uint64_t)columns / 128u) * 4u);
	if ( accounting->scale_round == SPARK_STAGE_PACK_SCALE_ROUND_CEIL )
		return((uint64_t)rows * ((columns + ((1u << accounting->scale_block_log2) - 1u)) >> accounting->scale_block_log2));
	return((uint64_t)rows * (columns >> accounting->scale_block_log2));
}

/* The declared scale_group_size check as table data: equality against the
 * row's required value (the row of a scale-free class requires zero). */
static inline uint32_t SparkStagePackWeightScaleGroupSizeOk(const SparkStagePackWeightAccounting *accounting,uint32_t declared_group_size)
{
	return(accounting != 0 && (uint32_t)accounting->scale_group_size == declared_group_size ? 1u : 0u);
}

/* Class-id form of the scale-plane rule for callers still speaking bare
 * weight-class ids: the three historical scale-carrying classes map onto
 * their planes; everything else travels scale-free. */
static inline const SparkStagePackWeightAccounting *SparkStagePackWeightClassAccounting(uint32_t weight_class)
{
	static const SparkStagePackWeightAccounting classes[6] =
	{
		[SPARK_STAGE_PACK_WEIGHT_BF16] = {SPARK_STAGE_PACK_WEIGHT_BF16,SPARK_STAGE_PACK_SCALE_PLANE_NONE,0u,SPARK_STAGE_PACK_SCALE_ROUND_FLOOR,0u},
		[SPARK_STAGE_PACK_WEIGHT_F32] = {SPARK_STAGE_PACK_WEIGHT_F32,SPARK_STAGE_PACK_SCALE_PLANE_NONE,0u,SPARK_STAGE_PACK_SCALE_ROUND_FLOOR,0u},
		[SPARK_STAGE_PACK_WEIGHT_U32] = {SPARK_STAGE_PACK_WEIGHT_U32,SPARK_STAGE_PACK_SCALE_PLANE_NONE,0u,SPARK_STAGE_PACK_SCALE_ROUND_FLOOR,0u},
		[SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1] = {SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1,SPARK_STAGE_PACK_SCALE_PLANE_FLAT_GROUP32,5u,SPARK_STAGE_PACK_SCALE_ROUND_FLOOR,32u},
		[SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128] = {SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128,SPARK_STAGE_PACK_SCALE_PLANE_TILE_128X128_F32,7u,SPARK_STAGE_PACK_SCALE_ROUND_FLOOR,128u},
		[SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128] = {SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128,SPARK_STAGE_PACK_SCALE_PLANE_PER_ROW_BLOCK,7u,SPARK_STAGE_PACK_SCALE_ROUND_FLOOR,128u}
	};
	return(weight_class < 6u ? &classes[weight_class] : 0);
}

static inline uint64_t SparkStagePackWeightClassScaleBytes(uint32_t weight_class,uint32_t rows,uint32_t columns)
{
	return(SparkStagePackWeightScaleBytes(SparkStagePackWeightClassAccounting(weight_class),rows,columns));
}

static inline uint32_t SparkStagePackWeightClassScaleGroupSizeOk(uint32_t weight_class,uint32_t declared_group_size)
{
	return(SparkStagePackWeightScaleGroupSizeOk(SparkStagePackWeightClassAccounting(weight_class),declared_group_size));
}

/* Full-attention layers among the first n layers of a periodic stack: one
 * phase in the shared period puts them at period-1, 2*period-1, ..., so the
 * count is simply n / period. Both periodic inventories are built from this
 * difference. */
static inline uint32_t SparkStagePackFullAttentionLayersBelow(uint32_t attention_period,uint32_t layer_count)
{
	return(layer_count / attention_period);
}

/*
 * Periodic slice inventory: `per_layer_base` tensors on every layer plus
 * `gdn_add`/`attn_add` on recurrent/full-attention layers, the embedding on
 * stage zero, and the head-stage tail: `last_stage_fixed_add` globals plus
 * `last_stage_split_embed_add` more on a split stack (the untied second
 * embedding copy). Serves the two families differing only in constants.
 */
static inline uint32_t SparkStagePackPeriodicSliceInventory(uint32_t first_layer_index,uint32_t layer_count,uint32_t total_layer_count,uint32_t attention_period,uint32_t per_layer_base,uint32_t gdn_add,uint32_t attn_add,uint32_t last_stage_fixed_add,uint32_t last_stage_split_embed_add)
{
	uint32_t full = SparkStagePackFullAttentionLayersBelow(attention_period,first_layer_index + layer_count) - SparkStagePackFullAttentionLayersBelow(attention_period,first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * per_layer_base) + (gdn * gdn_add) + (full * attn_add);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == total_layer_count )
		tensors += last_stage_fixed_add + (first_layer_index != 0u ? last_stage_split_embed_add : 0u);
	return(tensors);
}

/*
 * Class-summed inventory over a layer range for stacks without a uniform
 * attention period: `class_adds[class_of(layer)]` tensors per layer. The
 * compressor/indexer families sum their per-kind adds through this one
 * loop; stage extras and ownership adjustments stay family-local tails.
 * The class values are the family's contract - a class_of that can emit a
 * sentinel maps it onto a real row before returning (the reader does not
 * guess a fallback).
 */
typedef uint32_t (*SparkStagePackLayerClassOf)(void *context,uint32_t layer_index);

static inline uint32_t SparkStagePackClassSummedRange(void *context,SparkStagePackLayerClassOf class_of,const uint32_t *class_adds,uint32_t first_layer_index,uint32_t layer_count)
{
	uint32_t layer,tensors = 0u;
	for (layer = first_layer_index; layer < first_layer_index + layer_count; layer++)
		tensors += class_adds[class_of(context,layer)];
	return(tensors);
}

/*
 * Header comparison as one walk over the contiguous u32 prefix (every field
 * up to the trailing u64 offsets): first mismatch wins with code -(i+1), so
 * each field owns a unique negative code naming exactly what disagreed.
 * Field-name tables stay per family alongside their headers.
 */
static inline int32_t SparkStagePackHeaderFieldsMatch(const void *file_header,const void *expected_header,uint32_t u32_field_count)
{
	const uint32_t *file_fields = (const uint32_t *)file_header;
	const uint32_t *expected_fields = (const uint32_t *)expected_header;
	uint32_t i;
	for ( i = 0u; i < u32_field_count; i++ )
		if ( file_fields[i] != expected_fields[i] )
			return(-(int32_t)(i + 1u));
	return(0);
}

/*
 * Flat-dialect layer-class resolution tail, shared verbatim by the families
 * that reserve an MTP marker inside the global namespace. Precedence and
 * codes are the contract: at the marker only every-layer/attention kinds
 * ride per-layer (-6); global kinds carry the marker and vice versa (-2);
 * globals pass; per-layer kinds sit inside the stack (-3) and agree with
 * the layer map (-4 recurrent, -5 attention).
 */
#define SPARK_STAGE_PACK_LAYER_CLASS_GLOBAL 0u
#define SPARK_STAGE_PACK_LAYER_CLASS_EVERY_LAYER 1u
#define SPARK_STAGE_PACK_LAYER_CLASS_GDN_LAYER 2u
#define SPARK_STAGE_PACK_LAYER_CLASS_ATTN_LAYER 3u

static inline int32_t SparkStagePackResolveLayerClass(uint32_t layer_class,uint32_t is_global,uint32_t at_mtp_marker,uint32_t layer_index,uint32_t total_layer_count,uint32_t layer_is_gdn)
{
	if ( at_mtp_marker != 0u )
		return((is_global == 0u && (layer_class == SPARK_STAGE_PACK_LAYER_CLASS_EVERY_LAYER || layer_class == SPARK_STAGE_PACK_LAYER_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (layer_class == SPARK_STAGE_PACK_LAYER_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= total_layer_count )
		return(-3);
	if ( layer_class == SPARK_STAGE_PACK_LAYER_CLASS_GDN_LAYER && layer_is_gdn == 0u )
		return(-4);
	if ( layer_class == SPARK_STAGE_PACK_LAYER_CLASS_ATTN_LAYER && layer_is_gdn != 0u )
		return(-5);
	return(0);
}

/*
 * Format-generation dispatch.
 *
 * Packs carry a format_version. Generation 3 is the historical per-model
 * dialect set whose mechanics the tables above already parameterize; its
 * behavior is FROZEN CONTRACT - none of the functions above may change
 * semantics, so an already-published generation-3 pack keeps loading
 * bit-identically for as long as it exists. Retirement of the v3 path is
 * not a code decision: it requires an explicit republish ruling backed by a
 * fleet inventory proving no generation-3 pack remains in service.
 *
 * Generation 4 is the converged dialect - one scale-plane algebra, an
 * accounted extent for EVERY payload class, overflow-checked arithmetic,
 * and the strict placement discipline made mandatory for every loader:
 *
 *  - Block scale planes always round the block-column count UP and carry
 *    their own scale element width; the tile plane and the floor rounding
 *    no longer exist. A ragged row count therefore changes the accepted
 *    scale extent under v4 (the historical planes silently rounded it to
 *    zero) - that acceptance-set change is exactly what the version bump
 *    guards.
 *  - The group-32 plane refuses a non-divisible element extent instead of
 *    truncating it.
 *  - Entropy-coded payloads stop being byte-check-exempt: they are bound-
 *    checked against their uncompressed extent (non-empty, must fit), the
 *    strongest law expressible without a wire change; a declared-exact
 *    stream size or a stream checksum needs new entry fields and stays a
 *    coordinator decision.
 *  - Payload arithmetic that historically wrapped now refuses overflow.
 *
 * Dispatch belongs at the call site: a loader reads its wire version,
 * refuses unknown generations loudly, and selects the v4 chain below only
 * for version 4. Family adoption rides owner patches; nothing here widens
 * a check a generation-3 loader performed.
 */
#define SPARK_STAGE_PACK_GENERATION_V3 3u
#define SPARK_STAGE_PACK_GENERATION_V4 4u

static inline uint32_t SparkStagePackGenerationKnown(uint32_t format_version)
{
	return(format_version == SPARK_STAGE_PACK_GENERATION_V3 || format_version == SPARK_STAGE_PACK_GENERATION_V4 ? 1u : 0u);
}

/* Entropy-coded bf16 streams: accounted by BOUND (the uncompressed extent),
 * never byte-exact. Scale-free by definition of the stream format. */
#define SPARK_STAGE_PACK_WEIGHT_BF16_RANS 7u

/* The group-32 scale plane's element divisor under generation 4. */
#define SPARK_STAGE_PACK_V4_GROUP32_DIVISOR 32u

/*
 * The ONE generation-4 accounting table. Same payload density classes as
 * generation 3; every row states its whole scale law (plane, block, group
 * size, element width) and ceiling is implied - there is no floor row.
 */
typedef struct SparkStagePackV4Accounting
{
	uint8_t payload_class;
	uint8_t scale_plane;       /* NONE | FLAT_GROUP32 | PER_ROW_BLOCK only */
	uint8_t scale_block_log2;  /* PER_ROW_BLOCK block = 1 << log2 */
	uint8_t scale_group_size;  /* declared scale_group_size the row requires */
	uint8_t scale_element_bytes;
} SparkStagePackV4Accounting;

static inline const SparkStagePackV4Accounting *SparkStagePackV4ClassAccounting(uint32_t weight_class)
{
	static const SparkStagePackV4Accounting classes[8] =
	{
		[SPARK_STAGE_PACK_WEIGHT_BF16] = {SPARK_STAGE_PACK_WEIGHT_BF16,SPARK_STAGE_PACK_SCALE_PLANE_NONE,0u,0u,0u},
		[SPARK_STAGE_PACK_WEIGHT_F32] = {SPARK_STAGE_PACK_WEIGHT_F32,SPARK_STAGE_PACK_SCALE_PLANE_NONE,0u,0u,0u},
		[SPARK_STAGE_PACK_WEIGHT_U32] = {SPARK_STAGE_PACK_WEIGHT_U32,SPARK_STAGE_PACK_SCALE_PLANE_NONE,0u,0u,0u},
		[SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1] = {SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1,SPARK_STAGE_PACK_SCALE_PLANE_FLAT_GROUP32,5u,32u,1u},
		[SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128] = {SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128,SPARK_STAGE_PACK_SCALE_PLANE_PER_ROW_BLOCK,7u,128u,4u},
		[SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128] = {SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128,SPARK_STAGE_PACK_SCALE_PLANE_PER_ROW_BLOCK,7u,128u,1u},
		/* Plain fp8 with one e8m0 per block-column of every row: the same
		 * law as the e8m0b128 class, carried under its own historical id. */
		[SPARK_STAGE_PACK_WEIGHT_FP8_E4M3] = {SPARK_STAGE_PACK_WEIGHT_FP8_E4M3,SPARK_STAGE_PACK_SCALE_PLANE_PER_ROW_BLOCK,7u,128u,1u},
		[SPARK_STAGE_PACK_WEIGHT_BF16_RANS] = {SPARK_STAGE_PACK_WEIGHT_BF16_RANS,SPARK_STAGE_PACK_SCALE_PLANE_NONE,0u,0u,0u}
	};
	return(weight_class < 8u ? &classes[weight_class] : 0);
}

/* Overflow-checked payload extents; 0 marks an invalid or overflowing
 * extent (rows and columns are nonzero by the caller's contract). An
 * entropy-coded class yields its BOUND - the uncompressed bf16 extent. */
static inline uint64_t SparkStagePackV4PayloadBytes(const SparkStagePackV4Accounting *accounting,uint32_t rows,uint32_t columns)
{
	uint64_t elements;
	if ( accounting == 0 || rows == 0u || columns == 0u || rows > UINT64_MAX / columns )
		return(0u);
	elements = (uint64_t)rows * columns;
	switch ( accounting->payload_class )
	{
	case SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1: return(elements / 2u);
	case SPARK_STAGE_PACK_WEIGHT_F32:
	case SPARK_STAGE_PACK_WEIGHT_U32: return(elements > UINT64_MAX / 4u ? 0u : elements * 4u);
	case SPARK_STAGE_PACK_WEIGHT_BF16_RANS:
	case SPARK_STAGE_PACK_WEIGHT_BF16: return(elements > UINT64_MAX / 2u ? 0u : elements * 2u);
	default: return(elements > UINT64_MAX ? 0u : elements);
	}
}

/* Generation-4 scale extents: NONE is zero; the group-32 plane demands a
 * divisible element extent (callers refuse 0-with-nonzero-elements as the
 * divisibility refusal); block planes round UP and multiply their width. */
static inline uint64_t SparkStagePackV4ScaleBytes(const SparkStagePackV4Accounting *accounting,uint32_t rows,uint32_t columns)
{
	uint64_t elements,block_count;
	if ( accounting == 0 )
		return(0u);
	if ( accounting->scale_plane == SPARK_STAGE_PACK_SCALE_PLANE_FLAT_GROUP32 )
	{
		if ( rows == 0u || columns == 0u || rows > UINT64_MAX / columns )
			return(UINT64_MAX);
		elements = (uint64_t)rows * columns;
		return(elements % SPARK_STAGE_PACK_V4_GROUP32_DIVISOR != 0u ? UINT64_MAX : elements / SPARK_STAGE_PACK_V4_GROUP32_DIVISOR);
	}
	if ( accounting->scale_plane != SPARK_STAGE_PACK_SCALE_PLANE_PER_ROW_BLOCK )
		return(0u);
	block_count = ((uint64_t)columns + (1u << accounting->scale_block_log2) - 1u) >> accounting->scale_block_log2;
	if ( block_count == 0u || rows > UINT64_MAX / block_count || (uint64_t)rows * block_count > UINT64_MAX / accounting->scale_element_bytes )
		return(UINT64_MAX);
	return((uint64_t)rows * block_count * accounting->scale_element_bytes);
}

/*
 * The ONE normalized entry check generation 4 mandates, in refusal order -
 * each cause owns a unique negative code:
 *   -1 null table / unknown class / zero rows-columns-alignment / file
 *      window smaller than the minimum offset / unaccountable extent
 *   -2 declared payload_bytes wrong (entropy-coded: empty or over its
 *      uncompressed bound)
 *   -3 declared scale_bytes wrong (group-32 extent not divisible included)
 *   -4 declared scale_group_size != the class's required value
 *   -5 payload placement against the strict discipline
 *   -6 scale placement against the strict discipline (a zero-byte scale at
 *      a nonzero offset included)
 * The strict discipline restates CheckEntryBounds clause-for-clause so the
 * generation-3 function above remains untouched frozen contract.
 */
static inline int32_t SparkStagePackV4ValidateEntryExtent(const SparkStagePackV4Accounting *accounting,uint32_t rows,uint32_t columns,uint64_t minimum_offset,uint32_t alignment_bytes,uint64_t file_bytes,uint64_t payload_offset,uint64_t payload_bytes,uint64_t scale_offset,uint64_t scale_bytes,uint32_t declared_scale_group_size)
{
	uint64_t expected_payload,expected_scale;
	if ( accounting == 0 || alignment_bytes == 0u || rows == 0u || columns == 0u || minimum_offset > file_bytes )
		return(-1);
	expected_payload = SparkStagePackV4PayloadBytes(accounting,rows,columns);
	if ( expected_payload == 0u )
		return(-1);
	if ( accounting->payload_class == SPARK_STAGE_PACK_WEIGHT_BF16_RANS )
	{
		if ( payload_bytes == 0u || payload_bytes > expected_payload )
			return(-2);
	}
	else if ( payload_bytes != expected_payload )
		return(-2);
	expected_scale = SparkStagePackV4ScaleBytes(accounting,rows,columns);
	if ( expected_scale == UINT64_MAX || scale_bytes != expected_scale )
		return(-3);
	if ( declared_scale_group_size != (uint32_t)accounting->scale_group_size )
		return(-4);
	if ( payload_offset % alignment_bytes != 0u || payload_offset < minimum_offset || payload_offset > file_bytes || payload_bytes > file_bytes - payload_offset )
		return(-5);
	if ( expected_scale == 0u )
	{
		if ( scale_offset != 0u )
			return(-6);
	}
	else if ( scale_offset % alignment_bytes != 0u || scale_offset < minimum_offset || scale_offset > file_bytes || scale_bytes > file_bytes - scale_offset )
		return(-6);
	return(0);
}

/*
 * The ONE inventory rule generation 4 mandates: the seen mask over a
 * layer's kinds must EQUAL the mask computed from the geometry tables -
 * neither missing kinds nor extra kinds are tolerated, and a declared
 * count field is advisory only (it can never rescue a mismatch).
 */
static inline uint32_t SparkStagePackV4InventoryComplete(uint64_t expected_mask,uint64_t seen_mask)
{
	return(expected_mask == seen_mask ? 1u : 0u);
}

#ifdef __cplusplus
}
#endif
