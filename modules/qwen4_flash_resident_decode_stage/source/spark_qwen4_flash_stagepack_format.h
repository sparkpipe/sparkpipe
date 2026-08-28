#pragma once

#include <stdint.h>

#include "sparkpipe/spark_qwen4_flash_resident_decode_stage_firmware.h"
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

#define SPARK_QWEN4_FLASH_STAGEPACK_MAGIC 0x50533451u /* 'Q4SP' little endian: the Flash family pack format (own magic, not the qwen38 'Q8SP') */
/* Version 2: the hyper-connection residual (full 4-stream hc_norm on the
 * norm slots plus the per-sublayer mixers), the attention indexer, the PLE
 * n-gram block and the global/mtp readout mixers join the inventory; the
 * norm-slot widths change 2560 -> 10240, so v1 packs fail closed at load. */
#define SPARK_QWEN4_FLASH_STAGEPACK_FORMAT_VERSION 2u
#define SPARK_QWEN4_FLASH_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_QWEN4_FLASH_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_QWEN4_FLASH_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkQwen4FlashStagePackTensorKind
{
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTENTION_NORM = 3,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_NORM = 4,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_GATE = 5,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W1 = 6,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W3 = 7,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_DOWN = 8,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE = 9,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_UP = 10,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_DOWN = 11,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT = 12,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_QKV = 13,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_GATE = 14,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_BETA = 15,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DECAY = 16,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_OUTPUT = 17,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_CONV_WEIGHT = 18,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_A_LOG = 19,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DT_BIAS = 20,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_NORM = 21,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY = 22,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY = 23,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE = 24,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT = 25,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY_NORM = 26,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY_NORM = 27,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FC = 28,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_EMBED_NORM = 29,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_HIDDEN_NORM = 30,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FINAL_NORM = 31,
	/* v2: hyper-connection residual, per sublayer (attn and mlp sets),
	 * the attention indexer (attn-layer class), the readout mixers and
	 * the PLE block (layer PLE_LAYER_INDEX only). */
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_DOWN = 32,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_UP = 33,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_INJECT = 34,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_DOWN = 35,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_UP = 36,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_INJECT = 37,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_QK = 38,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_Q_NORM = 39,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM = 40,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_DOWN = 41,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_UP = 42,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_DOWN = 43,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_UP = 44,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_KEY = 45,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_VALUE = 46,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_KEY = 47,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_QUERY = 48,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_CONV = 49,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_CONV = 50,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_MULTIPLIERS = 51,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_VOCABS = 52,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_OFFSETS = 53,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NGRAM = 54,
	SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_KIND_COUNT = 55
} SparkQwen4FlashStagePackTensorKind;

#define SPARK_QWEN4_FLASH_STAGEPACK_CLASS_GLOBAL 0u
#define SPARK_QWEN4_FLASH_STAGEPACK_CLASS_EVERY_LAYER 1u
#define SPARK_QWEN4_FLASH_STAGEPACK_CLASS_GDN_LAYER 2u
#define SPARK_QWEN4_FLASH_STAGEPACK_CLASS_ATTN_LAYER 3u
#define SPARK_QWEN4_FLASH_STAGEPACK_CLASS_PLE_LAYER 4u

typedef struct SparkQwen4FlashStagePackHeader
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
} SparkQwen4FlashStagePackHeader;

typedef struct SparkQwen4FlashStagePackEntry
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
} SparkQwen4FlashStagePackEntry;

/*
 * Fixed wire sizes: the structs are ordered so natural alignment produces no
 * padding on the LP64 targets this module builds for; the asserts make that a
 * compile error rather than a silent format drift.
 */
#define SPARK_QWEN4_FLASH_STAGEPACK_HEADER_BYTES 120u
#define SPARK_QWEN4_FLASH_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkQwen4FlashStagePackHeader) == SPARK_QWEN4_FLASH_STAGEPACK_HEADER_BYTES,"qwen38 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkQwen4FlashStagePackEntry) == SPARK_QWEN4_FLASH_STAGEPACK_ENTRY_BYTES,"qwen38 stage pack directory entry must be 56 wire bytes");

// Model-geometry compile-time proofs.
_Static_assert(SPARK_QWEN4_FLASH_MODEL_GDN_LAYER_COUNT + SPARK_QWEN4_FLASH_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT,"qwen38 layer split must cover the stack");
_Static_assert((SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT % SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD) == 0u,"qwen38 layer count must be whole periods");
_Static_assert(SPARK_QWEN4_FLASH_MODEL_GDN_LAYER_COUNT == (SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT / SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD) * (SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD - 1u),"qwen38 gdn count must match the 3:1 period");
_Static_assert((SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT % SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT) == 0u,"qwen38 value heads must group evenly onto key heads");
_Static_assert(SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD == 3u,"qwen4_flash grouped-value ratio is three per config (48 value heads over 16 key heads)");
_Static_assert((SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"qwen38 query heads must group evenly onto kv heads");
_Static_assert(SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION == SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION / 4u,"qwen38 rope covers a quarter of the head");
_Static_assert((SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"qwen38 rope dimension must pair");
_Static_assert(SPARK_QWEN4_FLASH_MODEL_GDN_CONV_CHANNELS == 10240u,"qwen4_flash conv width is q+k+v concatenated");
_Static_assert(SPARK_QWEN4_FLASH_MODEL_GDN_QK_DIMENSION == 2048u && SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION == 6144u,"qwen4_flash gdn projection widths per config");
_Static_assert(SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION == 6144u && SPARK_QWEN4_FLASH_MODEL_ATTN_KV_DIMENSION == 512u,"qwen4_flash attention projection widths per config");
_Static_assert(SPARK_QWEN4_FLASH_MODEL_MXFP4_GROUP_SIZE == 32u,"qwen38 mxfp4 group size must be 32");
_Static_assert((SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION % SPARK_QWEN4_FLASH_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 expert intermediate must tile for mxfp4 groups");
_Static_assert((SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION % SPARK_QWEN4_FLASH_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 hidden must tile for mxfp4 groups");

static inline uint32_t SparkQwen4FlashStagePackFullAttentionLayersBelow(uint32_t layer_count)
{
	return(layer_count / SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD);
}

/*
 * The tensor inventory of a slice, computed, never declared: sixteen tensors
 * on every layer (two hc norms and the eight MoE tensors plus the six hc
 * mixer tensors), nine more on a GDN layer, nine more on a full-attention
 * layer (attn tensors + indexer), ten PLE tensors on the PLE layer, the
 * embedding on stage zero and the final norm, LM head, mixer pair, four MTP
 * globals plus the MTP mixer pair, twenty-five MTP layer tensors and
 * (multi-stage only) a second embedding copy on the last stage.
 */
static inline uint32_t SparkQwen4FlashStagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t full = SparkQwen4FlashStagePackFullAttentionLayersBelow(first_layer_index + layer_count) - SparkQwen4FlashStagePackFullAttentionLayersBelow(first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * 16u) + (gdn * 9u) + (full * 9u);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT )
		tensors += 2u + 4u + 4u + 25u + (first_layer_index != 0u ? 1u : 0u);
	return(tensors);
}

/* Real packs always carry the ten PLE tensors when the slice covers the PLE
 * layer; synthesized mid-pipeline test packs may omit the whole block (the
 * 23.8 GiB n-gram table is not synthesizable at true shape), which the
 * module accepts ONLY under the explicit allow-missing-ple env gate. */
static inline uint32_t SparkQwen4FlashStagePackExpectedTensorCountWithPle(uint32_t first_layer_index, uint32_t layer_count, uint32_t include_ple)
{
	uint32_t tensors = SparkQwen4FlashStagePackExpectedTensorCount(first_layer_index,layer_count);
	if ( include_ple != 0u && first_layer_index <= SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX && first_layer_index + layer_count > SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX )
		tensors += 10u;
	return(tensors);
}

static inline void SparkQwen4FlashStagePackExpectedGeometry(SparkQwen4FlashStagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count, uint32_t include_ple)
{
	header->magic = SPARK_QWEN4_FLASH_STAGEPACK_MAGIC;
	header->format_version = SPARK_QWEN4_FLASH_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_QWEN4_FLASH_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_QWEN4_FLASH_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkQwen4FlashStagePackExpectedTensorCountWithPle(first_layer_index,layer_count,include_ple);
	header->hidden_dimension = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_QWEN4_FLASH_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT;
	header->gdn_value_head_count = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT;
	header->gdn_head_key_dimension = SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_KEY_DIMENSION;
	header->gdn_head_value_dimension = SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION;
	header->gdn_conv_kernel = SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL;
	header->attn_query_head_count = SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION;
	header->routed_expert_count = SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT;
	header->experts_per_token = SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN;
	header->expert_intermediate_dimension = SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = SPARK_QWEN4_FLASH_MODEL_MXFP4_GROUP_SIZE;
	header->mtp_layer_count = SPARK_QWEN4_FLASH_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

/* Field-by-field comparison; returns 0 on match, nonzero on any drift.
 * The comparison is the library's; the layout proof is this family's
 * compile-time admission ticket to it. */
SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkQwen4FlashStagePackHeader);
static inline int32_t SparkQwen4FlashStagePackHeaderMatches(const SparkQwen4FlashStagePackHeader *file_header, const SparkQwen4FlashStagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}

/* The shape algebra and header comparison are the stagepack format
 * library's; this family states its geometry as data and keeps only the
 * v2 tensors that are genuinely its own (hyper-connections, indexer,
 * mixers, PLE). The static asserts pin the family's ABI codes to the
 * shared ones so neither side can drift silently. */
typedef SparkStagePackTensorShape SparkQwen4FlashStagePackTensorShape;

_Static_assert(SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,"qwen4 bf16 weight code must match the shared format");
_Static_assert(SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 == SPARK_STAGEPACK_FORMAT_WEIGHT_F32,"qwen4 f32 weight code must match the shared format");
_Static_assert(SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128,"qwen4 fp8 weight code must match the shared format");
_Static_assert(SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 == SPARK_STAGEPACK_FORMAT_WEIGHT_I64,"qwen4 i64 weight code must match the shared format");
_Static_assert(SPARK_QWEN4_FLASH_STAGEPACK_CLASS_GLOBAL == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL,"qwen4 global class must match the shared format");
_Static_assert(SPARK_QWEN4_FLASH_STAGEPACK_CLASS_EVERY_LAYER == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER,"qwen4 every-layer class must match the shared format");
_Static_assert(SPARK_QWEN4_FLASH_STAGEPACK_CLASS_GDN_LAYER == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER,"qwen4 gdn class must match the shared format");
_Static_assert(SPARK_QWEN4_FLASH_STAGEPACK_CLASS_ATTN_LAYER == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER,"qwen4 attn class must match the shared format");

static const SparkStagePackGeometryTable SparkQwen4FlashStagePackGeometry =
{
	.norm_width = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH,
	.hidden_dimension = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION,
	.routed_expert_count = SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT,
	.expert_intermediate_dimension = SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION,
	.gdn_conv_channels = SPARK_QWEN4_FLASH_MODEL_GDN_CONV_CHANNELS,
	.gdn_value_dimension = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION,
	.gdn_value_head_count = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT,
	.gdn_head_value_dimension = SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION,
	.gdn_conv_kernel = SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL
};

static inline void SparkQwen4FlashStagePackShapeInit(SparkQwen4FlashStagePackTensorShape *shape)
{
	SparkStagePackShapeInit(shape);
}

static inline int32_t SparkQwen4FlashStagePackShapeGlobal(uint32_t tensor_kind, SparkQwen4FlashStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN4_FLASH_STAGEPACK_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FC:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		shape->columns = 2u * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_DOWN:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_DOWN:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_UP:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_UP:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_EMBED_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_HIDDEN_NORM:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen4FlashStagePackShapeEveryLayer(uint32_t tensor_kind, SparkQwen4FlashStagePackTensorShape *shape)
{
	/* Norms and the MoE set are the shared axis; the hyper-connection
	 * residual pair (v2) is this family's own. */
	if ( SparkStagePackShapeEveryLayerCommon(tensor_kind,
		&SparkQwen4FlashStagePackGeometry,shape) == 0 )
		return(0);
	shape->layer_class = SPARK_QWEN4_FLASH_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_DOWN:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_DOWN:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_UP:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_UP:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_INJECT:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_INJECT:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen4FlashStagePackShapeGdn(uint32_t tensor_kind, SparkQwen4FlashStagePackTensorShape *shape)
{
	/* The whole GDN inventory of this family is the shared axis. */
	return(SparkStagePackShapeGdnCommon(tensor_kind,
		&SparkQwen4FlashStagePackGeometry,shape));
}

static inline int32_t SparkQwen4FlashStagePackShapeAttn(uint32_t tensor_kind, SparkQwen4FlashStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN4_FLASH_STAGEPACK_CLASS_ATTN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = 2u * SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_ATTN_KV_DIMENSION;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_QK:
		shape->rows = (SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + SPARK_QWEN4_FLASH_MODEL_INDEXER_KV_HEAD_COUNT) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_Q_NORM:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

/* PLE block shapes; only valid on the PLE layer (layer PLE_LAYER_INDEX).
 * The I64 metadata rows are exact hash constants - they travel as raw
 * little-endian int64, never converted. */
static inline int32_t SparkQwen4FlashStagePackShapePle(uint32_t tensor_kind, SparkQwen4FlashStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_QWEN4_FLASH_STAGEPACK_CLASS_PLE_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_KEY:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_VALUE:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_KEY:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_QUERY:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_CONV:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_CONV:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_MULTIPLIERS:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_SIZE;
		shape->natural_format = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_VOCABS:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_OFFSETS:
		shape->rows = 1u;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_COUNT;
		shape->natural_format = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64;
		return(0);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NGRAM:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_ROW_COUNT;
		shape->columns = SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkQwen4FlashStagePackTensorShapeOf(uint32_t tensor_kind, SparkQwen4FlashStagePackTensorShape *shape)
{
	SparkQwen4FlashStagePackShapeInit(shape);
	if ( SparkQwen4FlashStagePackShapeGlobal(tensor_kind,shape) == 0 )
		return(0);
	SparkQwen4FlashStagePackShapeInit(shape);
	if ( SparkQwen4FlashStagePackShapeEveryLayer(tensor_kind,shape) == 0 )
		return(0);
	SparkQwen4FlashStagePackShapeInit(shape);
	if ( SparkQwen4FlashStagePackShapeGdn(tensor_kind,shape) == 0 )
		return(0);
	SparkQwen4FlashStagePackShapeInit(shape);
	if ( SparkQwen4FlashStagePackShapeAttn(tensor_kind,shape) == 0 )
		return(0);
	SparkQwen4FlashStagePackShapeInit(shape);
	if ( SparkQwen4FlashStagePackShapePle(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

/*
 * Rank-local entry geometry: the per-kind narrowing the TP pack plan
 * applies (mirrored by tools/qwen4_flash_stagepack.py shard_ref and
 * tools/qwen4_flash_pack_verify.py - three statements of one plan, so a
 * drift fails closed at load). KV projections replicate whole when the kv
 * heads do not divide across ranks; norms, scalars and the MTP globals
 * replicate; everything else narrows along the sharded axis.
 */
static inline void SparkQwen4FlashStagePackNarrowShape(SparkQwen4FlashStagePackTensorShape *shape, uint32_t tensor_kind, uint32_t tp_degree, uint32_t tp_rank)
{
	uint32_t key_heads, value_heads, experts;
	if ( tp_degree <= 1u )
		return;
	key_heads = SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT / tp_degree;
	value_heads = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
	experts = SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	(void)tp_rank;
	switch ( tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY:
		shape->rows = (SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree) * 2u * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE:
		if ( (SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u )
			break; /* replicated */
		shape->rows = (SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT:
		shape->columns = SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION / tp_degree;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_QKV:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_CONV_WEIGHT:
		shape->rows = (2u * (key_heads * SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_KEY_DIMENSION))
			+ (value_heads * SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION);
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_GATE:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION / tp_degree;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_BETA:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DECAY:
		shape->rows = value_heads;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_A_LOG:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DT_BIAS:
		shape->columns = value_heads;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_OUTPUT:
		shape->columns = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION / tp_degree;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_GATE:
		shape->rows = experts;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W1:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W3:
		shape->rows = experts * SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_DOWN:
		shape->rows = experts * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_UP:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION / tp_degree;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		shape->columns = SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION / tp_degree;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT / tp_degree;
		break;
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NGRAM:
		/* Vocab-sharded n-gram table (the decided plan): rank r holds rows
		 * [r*rows/tp, (r+1)*rows/tp) of the head-major concatenated row
		 * space; out-of-shard ids gather zero and the bf16 all-reduce
		 * completes the embedding, exactly the main embedding pattern. */
		shape->rows = SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_ROW_COUNT / tp_degree;
		break;
	default:
		break; /* replicated: norms, scalars, MTP globals */
	}
}

/*
 * Kind resolved against a concrete layer: a global kind must carry the global
 * layer marker, a per-layer kind must sit inside the total layer space, and
 * the GDN/attention classes must agree with the hybrid layer map. The MTP
 * decoder is geometry-identical to a full-attention layer, so its sixteen
 * layer-shaped tensors REUSE the per-layer kinds at the reserved MTP marker.
 */
static inline int32_t SparkQwen4FlashStagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkQwen4FlashStagePackTensorShape *shape)
{
	if ( SparkQwen4FlashStagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	if ( layer_index == SPARK_QWEN4_FLASH_STAGEPACK_MTP_LAYER )
		return((is_global == 0u && (shape->layer_class == SPARK_QWEN4_FLASH_STAGEPACK_CLASS_EVERY_LAYER || shape->layer_class == SPARK_QWEN4_FLASH_STAGEPACK_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (shape->layer_class == SPARK_QWEN4_FLASH_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT )
		return(-3);
	if ( shape->layer_class == SPARK_QWEN4_FLASH_STAGEPACK_CLASS_GDN_LAYER && SPARK_QWEN4_FLASH_MODEL_LAYER_IS_GDN(layer_index) == 0u )
		return(-4);
	if ( shape->layer_class == SPARK_QWEN4_FLASH_STAGEPACK_CLASS_ATTN_LAYER && SPARK_QWEN4_FLASH_MODEL_LAYER_IS_GDN(layer_index) != 0u )
		return(-5);
	if ( shape->layer_class == SPARK_QWEN4_FLASH_STAGEPACK_CLASS_PLE_LAYER && layer_index != SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX )
		return(-7);
	return(0);
}

static inline uint64_t SparkQwen4FlashStagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(elements / 2u);
	if ( weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ||
		weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return(elements);
	if ( weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	if ( weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 )
		return(elements * 8u);
	return(elements * (uint64_t)SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkQwen4FlashStagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	if ( weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / SPARK_QWEN4_FLASH_MODEL_MXFP4_GROUP_SIZE);
	if ( weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(((uint64_t)rows / 128u) * ((uint64_t)columns / 128u) * 4u);
	/* E8M0B128: one exponent byte per (row, 128-column block) - the 27b form. */
	if ( weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return((uint64_t)rows * ((uint64_t)columns / 128u));
	return(0u);
}
