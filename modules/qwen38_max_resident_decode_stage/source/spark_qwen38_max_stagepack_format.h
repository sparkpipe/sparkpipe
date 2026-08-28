#pragma once

#include <stdint.h>

#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"

/*
 * Qwen 3.8 Max stage pack: a single file holding every tensor one pipeline
 * STAGE makes resident, plus the geometry the tensors were produced for.
 * The header restates the model geometry and the layer slice; every field is
 * compared against the compiled constants at load and a mismatch is a hard
 * failure with the offending field named. Same discipline as the qwen38_27b pack.
 *
 * Layout: [header][directory: tensor_count entries][payload bytes].
 * All offsets are absolute file offsets. A tensor's payload is contiguous;
 * MXFP4 tensors append their E8M0 scale plane immediately after the payload.
 *
 * Routed experts are flattened: w1/w3 are [expert_count * intermediate, H]
 * and w2 is [expert_count * H, intermediate]. The checkpoint's fused
 * gate_up_proj is split at pack time into w1 (rows 0..I) and w3 (rows I..2I).
 */

#define SPARK_QWEN38_MAX_STAGEPACK_MAGIC 0x50533851u /* 'Q8SP' little endian */
#define SPARK_QWEN38_MAX_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_QWEN38_MAX_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_QWEN38_MAX_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_QWEN38_MAX_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkQwen38MaxStagePackTensorKind
{
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTENTION_NORM = 3,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MLP_NORM = 4,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_GATE = 5,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1 = 6,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W3 = 7,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN = 8,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_GATE = 9,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_UP = 10,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_DOWN = 11,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT = 12,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_QKV = 13,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_GATE = 14,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_BETA = 15,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_DECAY = 16,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_OUTPUT = 17,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_CONV_WEIGHT = 18,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_A_LOG = 19,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_DT_BIAS = 20,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_NORM = 21,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY = 22,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY = 23,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_VALUE = 24,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT = 25,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY_NORM = 26,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY_NORM = 27,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FC = 28,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_EMBED_NORM = 29,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_HIDDEN_NORM = 30,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FINAL_NORM = 31,
	SPARK_QWEN38_MAX_STAGEPACK_TENSOR_KIND_COUNT = 32
} SparkQwen38MaxStagePackTensorKind;

#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_GLOBAL 0u
#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_EVERY_LAYER 1u
#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_GDN_LAYER 2u
#define SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER 3u

typedef struct SparkQwen38MaxStagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t tensor_count;
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t first_layer_index;
	uint32_t total_layer_count;
	uint32_t attention_period;
	uint32_t full_attention_phase;
	uint32_t gdn_key_head_count;
	uint32_t gdn_value_head_count;
	uint32_t gdn_head_key_dimension;
	uint32_t gdn_head_value_dimension;
	uint32_t gdn_conv_kernel;
	uint32_t attn_query_head_count;
	uint32_t attn_kv_head_count;
	uint32_t attn_head_dimension;
	uint32_t attn_rope_dimension;
	uint32_t routed_expert_count;
	uint32_t experts_per_token;
	uint32_t expert_intermediate_dimension;
	uint32_t output_vocab_count;
	uint32_t mxfp4_group_size;
	uint32_t mtp_layer_count;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkQwen38MaxStagePackHeader;

typedef struct SparkQwen38MaxStagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint32_t scale_group_size;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} SparkQwen38MaxStagePackEntry;

/*
 * Fixed wire sizes: the structs are ordered so natural alignment produces no
 * padding on the LP64 targets this module builds for; the asserts make that a
 * compile error rather than a silent format drift.
 */
#define SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES 120u
#define SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkQwen38MaxStagePackHeader) == SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES,"qwen38 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkQwen38MaxStagePackEntry) == SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES,"qwen38 stage pack directory entry must be 56 wire bytes");

// Model-geometry compile-time proofs.
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_LAYER_COUNT + SPARK_QWEN38_MAX_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_QWEN38_MAX_MODEL_LAYER_COUNT,"qwen38 layer split must cover the stack");
_Static_assert((SPARK_QWEN38_MAX_MODEL_LAYER_COUNT % SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD) == 0u,"qwen38 layer count must be whole periods");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_LAYER_COUNT == (SPARK_QWEN38_MAX_MODEL_LAYER_COUNT / SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD) * (SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD - 1u),"qwen38 gdn count must match the 3:1 period");
_Static_assert((SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT % SPARK_QWEN38_MAX_MODEL_GDN_KEY_HEAD_COUNT) == 0u,"qwen38 value heads must group evenly onto key heads");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD == 8u,"qwen38 grouped-value ratio is eight per config");
_Static_assert((SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"qwen38 query heads must group evenly onto kv heads");
_Static_assert(SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION == SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION / 4u,"qwen38 rope covers a quarter of the head");
_Static_assert((SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"qwen38 rope dimension must pair");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS == 20480u,"qwen38 conv width is q+k+v concatenated");
_Static_assert(SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION == 2048u && SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION == 16384u,"qwen38 gdn projection widths per config");
_Static_assert(SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION == 16384u && SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION == 1024u,"qwen38 attention projection widths per config");
_Static_assert(SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE == 32u,"qwen38 mxfp4 group size must be 32");
_Static_assert((SPARK_QWEN38_MAX_MODEL_EXPERT_INTERMEDIATE_DIMENSION % SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 expert intermediate must tile for mxfp4 groups");
_Static_assert((SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION % SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 hidden must tile for mxfp4 groups");

static inline uint32_t SparkQwen38MaxStagePackFullAttentionLayersBelow(uint32_t layer_count)
{
	return(layer_count / SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD);
}

/*
 * The tensor inventory of a slice, computed, never declared: ten tensors on
 * every layer (two norms and the eight MoE tensors), nine more on a GDN
 * layer, six more on a full-attention layer, the embedding on stage zero and
 * the final norm, LM head, four MTP globals, sixteen MTP layer tensors and
 * (multi-stage only) a second embedding copy on the last stage.
 */
static inline uint32_t SparkQwen38MaxStagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t full = SparkQwen38MaxStagePackFullAttentionLayersBelow(first_layer_index + layer_count) - SparkQwen38MaxStagePackFullAttentionLayersBelow(first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * 10u) + (gdn * 9u) + (full * 6u);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_QWEN38_MAX_MODEL_LAYER_COUNT )
		tensors += 2u + 4u + 16u + (first_layer_index != 0u ? 1u : 0u);
	return(tensors);
}

static inline void SparkQwen38MaxStagePackExpectedGeometry(SparkQwen38MaxStagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	header->magic = SPARK_QWEN38_MAX_STAGEPACK_MAGIC;
	header->format_version = SPARK_QWEN38_MAX_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkQwen38MaxStagePackExpectedTensorCount(first_layer_index,layer_count);
	header->hidden_dimension = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_QWEN38_MAX_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_QWEN38_MAX_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_QWEN38_MAX_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = SPARK_QWEN38_MAX_MODEL_GDN_KEY_HEAD_COUNT;
	header->gdn_value_head_count = SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT;
	header->gdn_head_key_dimension = SPARK_QWEN38_MAX_MODEL_GDN_HEAD_KEY_DIMENSION;
	header->gdn_head_value_dimension = SPARK_QWEN38_MAX_MODEL_GDN_HEAD_VALUE_DIMENSION;
	header->gdn_conv_kernel = SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL;
	header->attn_query_head_count = SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION;
	header->routed_expert_count = SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT;
	header->experts_per_token = SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN;
	header->expert_intermediate_dimension = SPARK_QWEN38_MAX_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_QWEN38_MAX_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE;
	header->mtp_layer_count = SPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

/* Field-by-field comparison; returns 0 on match, nonzero on any drift.
 * The comparison is the library's; the layout proof is this family's
 * compile-time admission ticket to it. */
SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkQwen38MaxStagePackHeader);
static inline int32_t SparkQwen38MaxStagePackHeaderMatches(const SparkQwen38MaxStagePackHeader *file_header, const SparkQwen38MaxStagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}

/* The shape algebra and header comparison are the stagepack format
 * library's; this family states its geometry as data and keeps only the
 * tensors that are genuinely its own. The static asserts pin the family's
 * ABI codes to the shared ones so neither side can drift silently. */
typedef SparkStagePackTensorShape SparkQwen38MaxStagePackTensorShape;

_Static_assert(SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,"qwen38 bf16 weight code must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 == SPARK_STAGEPACK_FORMAT_WEIGHT_F32,"qwen38 f32 weight code must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128,"qwen38 fp8 weight code must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_STAGEPACK_CLASS_GLOBAL == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL,"qwen38 global class must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_STAGEPACK_CLASS_EVERY_LAYER == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER,"qwen38 every-layer class must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_STAGEPACK_CLASS_GDN_LAYER == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER,"qwen38 gdn class must match the shared format");
_Static_assert(SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER,"qwen38 attn class must match the shared format");

static const SparkStagePackGeometryTable SparkQwen38MaxStagePackGeometry =
{
	.norm_width = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,
	.hidden_dimension = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,
	.routed_expert_count = SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT,
	.expert_intermediate_dimension = SPARK_QWEN38_MAX_MODEL_EXPERT_INTERMEDIATE_DIMENSION,
	.gdn_conv_channels = SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS,
	.gdn_value_dimension = SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION,
	.gdn_value_head_count = SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
	.gdn_head_value_dimension = SPARK_QWEN38_MAX_MODEL_GDN_HEAD_VALUE_DIMENSION,
	.gdn_conv_kernel = SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL
};

static inline void SparkQwen38MaxStagePackShapeInit(SparkQwen38MaxStagePackTensorShape *shape)
{
	SparkStagePackShapeInit(shape);
}

static inline int32_t SparkQwen38MaxStagePackShapeGlobal(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN38_MAX_STAGEPACK_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_QWEN38_MAX_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FC:
		shape->rows = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		shape->columns = 2u * SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_EMBED_NORM:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_HIDDEN_NORM:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen38MaxStagePackShapeEveryLayer(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	/* The whole every-layer inventory of this family is the shared axis. */
	return(SparkStagePackShapeEveryLayerCommon(tensor_kind,
		&SparkQwen38MaxStagePackGeometry,shape));
}

static inline int32_t SparkQwen38MaxStagePackShapeGdn(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	/* The whole GDN inventory of this family is the shared axis. */
	return(SparkStagePackShapeGdnCommon(tensor_kind,
		&SparkQwen38MaxStagePackGeometry,shape));
}

static inline int32_t SparkQwen38MaxStagePackShapeAttn(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = 2u * SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows = SPARK_QWEN38_MAX_MODEL_ATTN_KV_DIMENSION;
		shape->columns = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION;
		return(0);
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen38MaxStagePackTensorShapeOf(uint32_t tensor_kind, SparkQwen38MaxStagePackTensorShape *shape)
{
	SparkQwen38MaxStagePackShapeInit(shape);
	if ( SparkQwen38MaxStagePackShapeGlobal(tensor_kind,shape) == 0 )
		return(0);
	SparkQwen38MaxStagePackShapeInit(shape);
	if ( SparkQwen38MaxStagePackShapeEveryLayer(tensor_kind,shape) == 0 )
		return(0);
	SparkQwen38MaxStagePackShapeInit(shape);
	if ( SparkQwen38MaxStagePackShapeGdn(tensor_kind,shape) == 0 )
		return(0);
	SparkQwen38MaxStagePackShapeInit(shape);
	if ( SparkQwen38MaxStagePackShapeAttn(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

/*
 * Kind resolved against a concrete layer: a global kind must carry the global
 * layer marker, a per-layer kind must sit inside the total layer space, and
 * the GDN/attention classes must agree with the hybrid layer map. The MTP
 * decoder is geometry-identical to a full-attention layer, so its sixteen
 * layer-shaped tensors REUSE the per-layer kinds at the reserved MTP marker.
 */
static inline int32_t SparkQwen38MaxStagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkQwen38MaxStagePackTensorShape *shape)
{
	if ( SparkQwen38MaxStagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	if ( layer_index == SPARK_QWEN38_MAX_STAGEPACK_MTP_LAYER )
		return((is_global == 0u && (shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_EVERY_LAYER || shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= SPARK_QWEN38_MAX_MODEL_LAYER_COUNT )
		return(-3);
	if ( shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_GDN_LAYER && SPARK_QWEN38_MAX_MODEL_LAYER_IS_GDN(layer_index) == 0u )
		return(-4);
	if ( shape->layer_class == SPARK_QWEN38_MAX_STAGEPACK_CLASS_ATTN_LAYER && SPARK_QWEN38_MAX_MODEL_LAYER_IS_GDN(layer_index) != 0u )
		return(-5);
	return(0);
}

static inline uint64_t SparkQwen38MaxStagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(elements / 2u);
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(elements);
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	return(elements * (uint64_t)SPARK_QWEN38_MAX_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkQwen38MaxStagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE);
	if ( weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(((uint64_t)rows / 128u) * ((uint64_t)columns / 128u) * 4u);
	return(0u);
}
