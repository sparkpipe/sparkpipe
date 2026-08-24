#pragma once

#include <stdint.h>

#include "runtime/spark_hybrid_stagepack_core.h"
#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_status.h"

/*
 * Qwen 3.6 27B stage pack: a single file holding every tensor one pipeline
 * STAGE makes resident, plus the geometry the tensors were produced for.
 *
 * The header restates the model geometry and the layer slice; every field is
 * compared against the compiled constants at load and a mismatch is a hard
 * failure with the offending field named. Unlike the K3 v1 pack, the slice is
 * first-class: first_layer_index and layer_count describe this stage's layers
 * and the expected tensor count is COMPUTED from the slice, so a pack cannot
 * misdeclare its own inventory. The whole-stack case is first_layer_index 0
 * with layer_count 64, which is simply the one-stage pipeline.
 *
 * The MODELING-PIN pass against transformers main modeling_qwen3_5 (2026-07)
 * settled the checkpoint layout this table now states: the GDN q|k|v
 * projection is ONE fused tensor in conv channel order, beta and decay are
 * separate 48-row projections, the depthwise conv carries NO bias, and the
 * attention query projection fuses a per-head output gate (each head's 512
 * columns are 256 query then 256 gate, applied as sigmoid before o_proj).
 *
 * Layout: [header][directory: tensor_count entries][payload bytes].
 * All offsets are absolute file offsets. Payload of a tensor is contiguous.
 */

#define SPARK_QWEN36_STAGEPACK_MAGIC 0x50533651u /* 'Q6SP' little endian */
#define SPARK_QWEN36_STAGEPACK_FORMAT_VERSION 3u
/* v3 added tp_degree/tp_rank. A v2 pack read into the v3 struct leaves the
 * two TP fields zero; the loader treats degree 0 as degree 1 (no tensor
 * parallelism), so v2 PP packs stay loadable. */
#define SPARK_QWEN36_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_QWEN36_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_QWEN36_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkQwen36StagePackTensorKind
{
	SPARK_QWEN36_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_QWEN36_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_QWEN36_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM = 3,
	SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM = 4,
	SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE = 5,
	SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP = 6,
	SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN = 7,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_QKV = 8,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_GATE = 9,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_BETA = 10,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DECAY = 11,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_OUTPUT = 12,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_CONV_WEIGHT = 13,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_A_LOG = 14,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DT_BIAS = 15,
	SPARK_QWEN36_STAGEPACK_TENSOR_GDN_NORM = 16,
	SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY = 17,
	SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY = 18,
	SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE = 19,
	SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT = 20,
	SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM = 21,
	SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM = 22,
	SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC = 23,
	SPARK_QWEN36_STAGEPACK_TENSOR_MTP_EMBED_NORM = 24,
	SPARK_QWEN36_STAGEPACK_TENSOR_MTP_HIDDEN_NORM = 25,
	SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM = 26,
	SPARK_QWEN36_STAGEPACK_TENSOR_KIND_COUNT = 27
} SparkQwen36StagePackTensorKind;

// A kind belongs to a layer class; the resolver enforces class against the
// hybrid layer map, so a GDN tensor on a full-attention layer is a schema
// error at load, not a stray pointer at launch. The class values are the
// shared hybrid core's (runtime/spark_hybrid_stagepack_core.h).
#define SPARK_QWEN36_STAGEPACK_CLASS_GLOBAL SPARK_HYBRID_STAGEPACK_CLASS_GLOBAL
#define SPARK_QWEN36_STAGEPACK_CLASS_EVERY_LAYER SPARK_HYBRID_STAGEPACK_CLASS_EVERY_LAYER
#define SPARK_QWEN36_STAGEPACK_CLASS_GDN_LAYER SPARK_HYBRID_STAGEPACK_CLASS_GDN_LAYER
#define SPARK_QWEN36_STAGEPACK_CLASS_ATTN_LAYER SPARK_HYBRID_STAGEPACK_CLASS_ATTN_LAYER

typedef struct SparkQwen36StagePackHeader
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
	uint32_t ffn_intermediate_dimension;
	uint32_t output_vocab_count;
	uint32_t mxfp4_group_size;
	uint32_t mtp_layer_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkQwen36StagePackHeader;

typedef struct SparkQwen36StagePackEntry
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
} SparkQwen36StagePackEntry;

/*
 * Fixed wire sizes. The structs are ordered so natural alignment produces no
 * padding on the LP64 targets this module builds for; the asserts make that a
 * compile error rather than a silent format drift.
 */
#define SPARK_QWEN36_STAGEPACK_HEADER_BYTES 120u
#define SPARK_QWEN36_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkQwen36StagePackHeader) == SPARK_QWEN36_STAGEPACK_HEADER_BYTES,"qwen36 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkQwen36StagePackEntry) == SPARK_QWEN36_STAGEPACK_ENTRY_BYTES,"qwen36 stage pack directory entry must be 56 wire bytes");

// Model-geometry compile-time proofs live here, in the C-only pack header,
// because the CUDA translation unit includes the model header and the C++
// front end does not accept _Static_assert.
_Static_assert(SPARK_QWEN36_MODEL_GDN_LAYER_COUNT + SPARK_QWEN36_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_QWEN36_MODEL_LAYER_COUNT,"qwen36 layer split must cover the stack");
_Static_assert((SPARK_QWEN36_MODEL_LAYER_COUNT % SPARK_QWEN36_MODEL_ATTENTION_PERIOD) == 0u,"qwen36 layer count must be whole periods");
_Static_assert(SPARK_QWEN36_MODEL_GDN_LAYER_COUNT == (SPARK_QWEN36_MODEL_LAYER_COUNT / SPARK_QWEN36_MODEL_ATTENTION_PERIOD) * (SPARK_QWEN36_MODEL_ATTENTION_PERIOD - 1u),"qwen36 gdn count must match the 3:1 period");
_Static_assert((SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT % SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT) == 0u,"qwen36 value heads must group evenly onto key heads");
_Static_assert(SPARK_QWEN36_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD == 3u,"qwen36 grouped-value ratio is three per config");
_Static_assert((SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"qwen36 query heads must group evenly onto kv heads");
_Static_assert(SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION == SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION / 4u,"qwen36 rope covers a quarter of the head");
_Static_assert((SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"qwen36 rope dimension must pair");
_Static_assert(SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS == 10240u,"qwen36 conv width is q+k+v concatenated");
_Static_assert(SPARK_QWEN36_MODEL_GDN_QK_DIMENSION == 2048u && SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION == 6144u,"qwen36 gdn projection widths per config");
_Static_assert(SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION == 6144u && SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION == 1024u,"qwen36 attention projection widths per config");
_Static_assert((SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS % 16u) == 0u,"qwen36 chunk must tile for wmma");

// Full-attention layers among the first n of the stack: phase 3 in period 4
// puts them at 3, 7, 11, ..., so the count is simply n / 4. Mechanics live
// in the shared hybrid core.
static inline uint32_t SparkQwen36StagePackFullAttentionLayersBelow(uint32_t layer_count)
{
	return(SparkHybridStagePackFullAttentionLayersBelow(SPARK_QWEN36_MODEL_ATTENTION_PERIOD,layer_count));
}

/*
 * The tensor inventory of a slice, computed, never declared: five tensors on
 * every layer (two norms and the SwiGLU triple), nine more on a GDN layer,
 * six more on a full-attention layer, the embedding on stage zero and the
 * final norm plus LM head on the last stage.
 */
/*
 * The head stage's MTP draft chain embeds its own draft tokens, and the
 * vocabulary is untied, so on a multi-stage split the head pack carries a
 * second copy of the embedding table. A whole-stack pack already holds it
 * as the stage-zero global, so the whole-stack tensor count is unchanged.
 */
static inline uint32_t SparkQwen36StagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t full = SparkQwen36StagePackFullAttentionLayersBelow(first_layer_index + layer_count) - SparkQwen36StagePackFullAttentionLayersBelow(first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * 5u) + (gdn * 9u) + (full * 6u);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_QWEN36_MODEL_LAYER_COUNT )
		tensors += 2u + 4u + 11u + (first_layer_index != 0u ? 1u : 0u);
	return(tensors);
}

static inline void SparkQwen36StagePackExpectedGeometry(SparkQwen36StagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	header->magic = SPARK_QWEN36_STAGEPACK_MAGIC;
	header->format_version = SPARK_QWEN36_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_QWEN36_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_QWEN36_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkQwen36StagePackExpectedTensorCount(first_layer_index,layer_count);
	header->hidden_dimension = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_QWEN36_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_QWEN36_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_QWEN36_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT;
	header->gdn_value_head_count = SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT;
	header->gdn_head_key_dimension = SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION;
	header->gdn_head_value_dimension = SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION;
	header->gdn_conv_kernel = SPARK_QWEN36_MODEL_GDN_CONV_KERNEL;
	header->attn_query_head_count = SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION;
	header->ffn_intermediate_dimension = SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = 32u;
	header->mtp_layer_count = SPARK_QWEN36_MODEL_MTP_LAYER_COUNT;
	header->tp_degree = 1u;
	header->tp_rank = 0u;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

// Field-by-field comparison; each field owns a unique negative code so the
// failing load names exactly which dimension the pack disagrees on. The
// walk over the contiguous u32 prefix (everything up to the trailing u64
// offsets) lives in the shared hybrid core; the name table below maps the
// codes back to fields.
#define SPARK_QWEN36_STAGEPACK_COMPARE_U32_FIELDS \
	((sizeof(SparkQwen36StagePackHeader) - 2u * sizeof(uint64_t)) / sizeof(uint32_t))

static inline int32_t SparkQwen36StagePackCompareGeometry(const SparkQwen36StagePackHeader *file_header, const SparkQwen36StagePackHeader *expected)
{
	return(SparkHybridStagePackHeaderFieldsMatch(file_header,expected,(uint32_t)SPARK_QWEN36_STAGEPACK_COMPARE_U32_FIELDS));
}

static inline const char *SparkQwen36StagePackGeometryFieldName(int32_t compare_result)
{
	static const char *names[25] =
	{
		"ok","magic","format_version","header_bytes","directory_entry_bytes",
		"tensor_count","hidden_dimension","layer_count","first_layer_index",
		"total_layer_count","attention_period","full_attention_phase",
		"gdn_key_head_count","gdn_value_head_count","gdn_head_key_dimension",
		"gdn_head_value_dimension","gdn_conv_kernel","attn_query_head_count",
		"attn_kv_head_count","attn_head_dimension","attn_rope_dimension",
		"ffn_intermediate_dimension","output_vocab_count","mxfp4_group_size",
		"mtp_layer_count"
	};
	uint32_t index = (uint32_t)(-compare_result);
	if ( compare_result > 0 || index > 24u )
		return("unknown");
	return(names[index]);
}

typedef struct SparkQwen36StagePackTensorShape
{
	uint32_t rows;
	uint32_t columns;
	uint32_t natural_format;
	uint32_t quantizable;
	uint32_t layer_class;
} SparkQwen36StagePackTensorShape;

static inline int32_t SparkQwen36StagePackShapeGlobal(uint32_t tensor_kind, SparkQwen36StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN36_STAGEPACK_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_EMBEDDING:
		shape->rows = SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC:
		shape->rows = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->columns = 2u * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_EMBED_NORM:
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_HIDDEN_NORM:
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen36StagePackShapeEveryLayer(uint32_t tensor_kind, SparkQwen36StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN36_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM:
	case SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE:
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP:
		shape->rows = SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN:
		shape->rows = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen36StagePackShapeGdn(uint32_t tensor_kind, SparkQwen36StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN36_STAGEPACK_CLASS_GDN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_QKV:
		shape->rows = SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_GATE:
		shape->rows = SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_BETA:
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DECAY:
		shape->rows = SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_OUTPUT:
		shape->rows = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_CONV_WEIGHT:
		shape->rows = SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS;
		shape->columns = SPARK_QWEN36_MODEL_GDN_CONV_KERNEL;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_A_LOG:
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DT_BIAS:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT;
		shape->natural_format = SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen36StagePackShapeAttn(uint32_t tensor_kind, SparkQwen36StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN36_STAGEPACK_CLASS_ATTN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = 2u * SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows = SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION;
		shape->columns = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION;
		shape->quantizable = 1u;
		return(0);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen36StagePackTensorShapeOf(uint32_t tensor_kind, SparkQwen36StagePackTensorShape *shape)
{
	shape->natural_format = SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	shape->quantizable = 0u;
	if ( SparkQwen36StagePackShapeGlobal(tensor_kind,shape) == 0 )
		return(0);
	if ( SparkQwen36StagePackShapeEveryLayer(tensor_kind,shape) == 0 )
		return(0);
	if ( SparkQwen36StagePackShapeGdn(tensor_kind,shape) == 0 )
		return(0);
	if ( SparkQwen36StagePackShapeAttn(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

/*
 * Kind resolved against a concrete layer: a global kind must carry the global
 * layer marker, a per-layer kind must sit inside the total layer space, and
 * the GDN/attention classes must agree with the hybrid layer map.
 */
/*
 * The MTP decoder is geometry-identical to a full-attention layer, so its
 * eleven layer-shaped tensors REUSE the per-layer kinds at the reserved MTP
 * layer marker; only the four MTP globals (fc, the two pre-fc norms, the
 * final norm) are new kinds. Pinned from the checkpoint safetensors index.
 */
/* TP packs store one rank's row/column window of each shardable tensor;
 * replicated kinds (norms, conv, gates, beta/decay, MTP) are unchanged.
 * The fused GDN q|k|v projection is stitched q|k|v per rank, so its row
 * count is (2*qk + v) / degree. */
static inline void SparkQwen36StagePackApplyTpShard(uint32_t tensor_kind, uint32_t tp_degree, SparkQwen36StagePackTensorShape *shape)
{
	if ( tp_degree <= 1u )
		return;
	switch ( tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_EMBEDDING:
		/* Replicated: no collective broadcast primitive yet, so every rank
		 * gathers from the full table (matches the packer's TP plan). */
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE:
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN:
		shape->columns /= tp_degree;
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_QKV:
		shape->rows = (2u * SPARK_QWEN36_MODEL_GDN_QK_DIMENSION +
			SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION) / tp_degree;
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_OUTPUT:
		shape->columns /= tp_degree;
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows /= tp_degree;
		break;
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->columns /= tp_degree;
		break;
	default:
		break;
	}
}

static inline int32_t SparkQwen36StagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, uint32_t tp_degree, SparkQwen36StagePackTensorShape *shape)
{
	if ( SparkQwen36StagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	/* The MTP decoder's attention/FFN tensors shard exactly like main
	 * layers (the fc and the three norms live at the GLOBAL layer and keep
	 * full shapes via the default case below). */
	SparkQwen36StagePackApplyTpShard(tensor_kind,tp_degree,shape);
	/* The resolution tail - MTP marker admission, global/class agreement,
	 * stack bound, hybrid-map class check - is the shared core's, with its
	 * exact refusal codes (-6,-2,-3,-4,-5). */
	return(SparkHybridStagePackResolveLayerClass(shape->layer_class,is_global,
		layer_index == SPARK_QWEN36_STAGEPACK_MTP_LAYER ? 1u : 0u,
		layer_index,SPARK_QWEN36_MODEL_LAYER_COUNT,
		SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer_index)));
}

/* Byte accounting is the shared hybrid core's; this forward maps the
 * family's wire format enum onto the normalized weight classes. */
static inline uint32_t SparkQwen36StagePackWeightClass(uint32_t weight_format)
{
	if ( weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_MXFP4_E2M1);
	if ( weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_F32B128);
	/* Merge consolidation: origin/main's MX serving format (e8m0-tiled fp8)
	 * joins the forward map; its byte accounting lives in the shared hybrid
	 * core (payload 1 B/element, scale rows x columns/128 e8m0). */
	if ( weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_E8M0B128);
	if ( weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_F32);
	if ( weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_U32);
	return(SPARK_HYBRID_STAGEPACK_WEIGHT_BF16);
}

static inline uint64_t SparkQwen36StagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	return(SparkHybridStagePackPayloadBytes(SparkQwen36StagePackWeightClass(weight_format),rows,columns));
}

static inline uint64_t SparkQwen36StagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	return(SparkHybridStagePackScaleBytes(SparkQwen36StagePackWeightClass(weight_format),rows,columns));
}
