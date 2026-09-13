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
 * spans never overlap. */
static inline uint32_t SparkStagePackRangesOverlap(uint64_t left_offset,uint64_t left_bytes,uint64_t right_offset,uint64_t right_bytes)
{
	return(left_bytes != 0u && right_bytes != 0u && left_offset < right_offset + right_bytes && right_offset < left_offset + left_bytes ? 1u : 0u);
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
 * Kinds beyond 64 bits do not exist in any current family.
 */
typedef int32_t (*SparkStagePackShapeResolver)(void *context,uint32_t tensor_kind,uint32_t layer_index,SparkStagePackShape *shape);

static inline uint64_t SparkStagePackExpectedLayerMask(void *resolver_context,SparkStagePackShapeResolver expected_shape,uint32_t first_tensor_kind,uint32_t tensor_kind_count,uint32_t layer_index)
{
	SparkStagePackShape shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind = first_tensor_kind; kind < tensor_kind_count; kind++)
		if ( expected_shape(resolver_context,kind,layer_index,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

#ifdef __cplusplus
}
#endif
