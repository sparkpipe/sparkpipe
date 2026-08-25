#pragma once

#include <stdint.h>

#include "runtime/spark_hybrid_stagepack_core.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_status.h"

/*
 * Qwen 3.8 Max stage pack: a single file holding every tensor one pipeline
 * STAGE makes resident, plus the geometry the tensors were produced for.
 * The header restates the model geometry and the layer slice; every field is
 * compared against the compiled constants at load and a mismatch is a hard
 * failure with the offending field named. Same discipline as the qwen36 pack.
 *
 * Layout: [header][directory: tensor_count entries][payload bytes].
 * All offsets are absolute file offsets. A tensor's payload is contiguous;
 * MXFP4 tensors append their E8M0 scale plane immediately after the payload.
 *
 * Routed experts are flattened: w1/w3 are [expert_count * intermediate, H]
 * and w2 is [expert_count * H, intermediate]. The checkpoint's fused
 * gate_up_proj is split at pack time into w1 (rows 0..I) and w3 (rows I..2I).
 */

#define SPARK_QWEN38_STAGEPACK_MAGIC 0x50533851u /* 'Q8SP' little endian */
#define SPARK_QWEN38_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_QWEN38_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_QWEN38_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_QWEN38_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkQwen38StagePackTensorKind
{
	SPARK_QWEN38_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_QWEN38_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_QWEN38_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_QWEN38_STAGEPACK_TENSOR_ATTENTION_NORM = 3,
	SPARK_QWEN38_STAGEPACK_TENSOR_MLP_NORM = 4,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_GATE = 5,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W1 = 6,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W3 = 7,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_DOWN = 8,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE = 9,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_UP = 10,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_DOWN = 11,
	SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT = 12,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_QKV = 13,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_GATE = 14,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_BETA = 15,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DECAY = 16,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_OUTPUT = 17,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_CONV_WEIGHT = 18,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_A_LOG = 19,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DT_BIAS = 20,
	SPARK_QWEN38_STAGEPACK_TENSOR_GDN_NORM = 21,
	SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY = 22,
	SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY = 23,
	SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_VALUE = 24,
	SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_OUTPUT = 25,
	SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM = 26,
	SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM = 27,
	SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC = 28,
	SPARK_QWEN38_STAGEPACK_TENSOR_MTP_EMBED_NORM = 29,
	SPARK_QWEN38_STAGEPACK_TENSOR_MTP_HIDDEN_NORM = 30,
	SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FINAL_NORM = 31,
	SPARK_QWEN38_STAGEPACK_TENSOR_KIND_COUNT = 32
} SparkQwen38StagePackTensorKind;

/* Layer-class values are the shared hybrid core's
 * (runtime/spark_hybrid_stagepack_core.h), under this family's names. */
#define SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL SPARK_HYBRID_STAGEPACK_CLASS_GLOBAL
#define SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER SPARK_HYBRID_STAGEPACK_CLASS_EVERY_LAYER
#define SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER SPARK_HYBRID_STAGEPACK_CLASS_GDN_LAYER
#define SPARK_QWEN38_STAGEPACK_CLASS_ATTN_LAYER SPARK_HYBRID_STAGEPACK_CLASS_ATTN_LAYER

typedef struct SparkQwen38StagePackHeader
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
} SparkQwen38StagePackHeader;

typedef struct SparkQwen38StagePackEntry
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
} SparkQwen38StagePackEntry;

/*
 * Fixed wire sizes: the structs are ordered so natural alignment produces no
 * padding on the LP64 targets this module builds for; the asserts make that a
 * compile error rather than a silent format drift.
 */
#define SPARK_QWEN38_STAGEPACK_HEADER_BYTES 120u
#define SPARK_QWEN38_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkQwen38StagePackHeader) == SPARK_QWEN38_STAGEPACK_HEADER_BYTES,"qwen38 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkQwen38StagePackEntry) == SPARK_QWEN38_STAGEPACK_ENTRY_BYTES,"qwen38 stage pack directory entry must be 56 wire bytes");

// Model-geometry compile-time proofs.
_Static_assert(SPARK_QWEN38_MODEL_GDN_LAYER_COUNT + SPARK_QWEN38_MODEL_FULL_ATTENTION_LAYER_COUNT == SPARK_QWEN38_MODEL_LAYER_COUNT,"qwen38 layer split must cover the stack");
_Static_assert((SPARK_QWEN38_MODEL_LAYER_COUNT % SPARK_QWEN38_MODEL_ATTENTION_PERIOD) == 0u,"qwen38 layer count must be whole periods");
_Static_assert(SPARK_QWEN38_MODEL_GDN_LAYER_COUNT == (SPARK_QWEN38_MODEL_LAYER_COUNT / SPARK_QWEN38_MODEL_ATTENTION_PERIOD) * (SPARK_QWEN38_MODEL_ATTENTION_PERIOD - 1u),"qwen38 gdn count must match the 3:1 period");
_Static_assert((SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT % SPARK_QWEN38_MODEL_GDN_KEY_HEAD_COUNT) == 0u,"qwen38 value heads must group evenly onto key heads");
_Static_assert(SPARK_QWEN38_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD == 8u,"qwen38 grouped-value ratio is eight per config");
_Static_assert((SPARK_QWEN38_MODEL_ATTN_QUERY_HEAD_COUNT % SPARK_QWEN38_MODEL_ATTN_KV_HEAD_COUNT) == 0u,"qwen38 query heads must group evenly onto kv heads");
_Static_assert(SPARK_QWEN38_MODEL_ATTN_ROPE_DIMENSION == SPARK_QWEN38_MODEL_ATTN_HEAD_DIMENSION / 4u,"qwen38 rope covers a quarter of the head");
_Static_assert((SPARK_QWEN38_MODEL_ATTN_ROPE_DIMENSION % 2u) == 0u,"qwen38 rope dimension must pair");
_Static_assert(SPARK_QWEN38_MODEL_GDN_CONV_CHANNELS == 20480u,"qwen38 conv width is q+k+v concatenated");
_Static_assert(SPARK_QWEN38_MODEL_GDN_QK_DIMENSION == 2048u && SPARK_QWEN38_MODEL_GDN_VALUE_DIMENSION == 16384u,"qwen38 gdn projection widths per config");
_Static_assert(SPARK_QWEN38_MODEL_ATTN_QUERY_DIMENSION == 16384u && SPARK_QWEN38_MODEL_ATTN_KV_DIMENSION == 1024u,"qwen38 attention projection widths per config");
_Static_assert(SPARK_QWEN38_MODEL_MXFP4_GROUP_SIZE == 32u,"qwen38 mxfp4 group size must be 32");
_Static_assert((SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION % SPARK_QWEN38_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 expert intermediate must tile for mxfp4 groups");
_Static_assert((SPARK_QWEN38_MODEL_HIDDEN_DIMENSION % SPARK_QWEN38_MODEL_MXFP4_GROUP_SIZE) == 0u,"qwen38 hidden must tile for mxfp4 groups");

static inline uint32_t SparkQwen38StagePackFullAttentionLayersBelow(uint32_t layer_count)
{
	return(SparkHybridStagePackFullAttentionLayersBelow(SPARK_QWEN38_MODEL_ATTENTION_PERIOD,layer_count));
}

/*
 * The tensor inventory of a slice, computed, never declared: ten tensors on
 * every layer (two norms and the eight MoE tensors), nine more on a GDN
 * layer, six more on a full-attention layer, the embedding on stage zero and
 * the final norm, LM head, four MTP globals, sixteen MTP layer tensors and
 * (multi-stage only) a second embedding copy on the last stage.
 */
static inline uint32_t SparkQwen38StagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t full = SparkQwen38StagePackFullAttentionLayersBelow(first_layer_index + layer_count) - SparkQwen38StagePackFullAttentionLayersBelow(first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * 10u) + (gdn * 9u) + (full * 6u);
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_QWEN38_MODEL_LAYER_COUNT )
		tensors += 2u + 4u + 16u + (first_layer_index != 0u ? 1u : 0u);
	return(tensors);
}

static inline void SparkQwen38StagePackExpectedGeometry(SparkQwen38StagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	header->magic = SPARK_QWEN38_STAGEPACK_MAGIC;
	header->format_version = SPARK_QWEN38_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_QWEN38_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_QWEN38_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkQwen38StagePackExpectedTensorCount(first_layer_index,layer_count);
	header->hidden_dimension = SPARK_QWEN38_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_QWEN38_MODEL_LAYER_COUNT;
	header->attention_period = SPARK_QWEN38_MODEL_ATTENTION_PERIOD;
	header->full_attention_phase = SPARK_QWEN38_MODEL_FULL_ATTENTION_PHASE;
	header->gdn_key_head_count = SPARK_QWEN38_MODEL_GDN_KEY_HEAD_COUNT;
	header->gdn_value_head_count = SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT;
	header->gdn_head_key_dimension = SPARK_QWEN38_MODEL_GDN_HEAD_KEY_DIMENSION;
	header->gdn_head_value_dimension = SPARK_QWEN38_MODEL_GDN_HEAD_VALUE_DIMENSION;
	header->gdn_conv_kernel = SPARK_QWEN38_MODEL_GDN_CONV_KERNEL;
	header->attn_query_head_count = SPARK_QWEN38_MODEL_ATTN_QUERY_HEAD_COUNT;
	header->attn_kv_head_count = SPARK_QWEN38_MODEL_ATTN_KV_HEAD_COUNT;
	header->attn_head_dimension = SPARK_QWEN38_MODEL_ATTN_HEAD_DIMENSION;
	header->attn_rope_dimension = SPARK_QWEN38_MODEL_ATTN_ROPE_DIMENSION;
	header->routed_expert_count = SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT;
	header->experts_per_token = SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN;
	header->expert_intermediate_dimension = SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT;
	header->mxfp4_group_size = SPARK_QWEN38_MODEL_MXFP4_GROUP_SIZE;
	header->mtp_layer_count = SPARK_QWEN38_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

/* Field-by-field comparison; the walk over the contiguous u32 prefix
 * (everything up to the trailing u64 offsets) is the shared core's and
 * gives each field its own negative code, so a refusal names the first
 * field that disagreed. */
#define SPARK_QWEN38_STAGEPACK_COMPARE_U32_FIELDS \
	((sizeof(SparkQwen38StagePackHeader) - 2u * sizeof(uint64_t)) / sizeof(uint32_t))

static inline int32_t SparkQwen38StagePackHeaderMatches(const SparkQwen38StagePackHeader *file_header, const SparkQwen38StagePackHeader *expected)
{
	return(SparkHybridStagePackHeaderFieldsMatch(file_header,expected,(uint32_t)SPARK_QWEN38_STAGEPACK_COMPARE_U32_FIELDS));
}

typedef struct SparkQwen38StagePackTensorShape
{
	uint32_t rows;
	uint32_t columns;
	uint32_t natural_format;
	uint32_t layer_class;
} SparkQwen38StagePackTensorShape;

/*
 * The per-kind geometry table - the model-specific part of this family, as
 * data rather than as four switches of case lines. One row per tensor kind:
 * its layer class, its natural weight format, and its unpacked shape. The
 * reader mechanics around these rows live in runtime/spark_stagepack_reader.h.
 */
typedef struct SparkQwen38StagePackKindGeometry
{
	uint32_t rows;
	uint32_t columns;
	uint32_t natural_format;
	uint32_t layer_class;
}
SparkQwen38StagePackKindGeometry;

#define SPARK_QWEN38_GEOM_ROW(kind,class,format,rows_expression,columns_expression) \
	[kind] = {rows_expression,columns_expression,format,class}

static const SparkQwen38StagePackKindGeometry SPARK_QWEN38_STAGEPACK_KIND_GEOMETRY[SPARK_QWEN38_STAGEPACK_TENSOR_KIND_COUNT] =
{
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_EMBEDDING,SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_FINAL_NORM,SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_LM_HEAD,SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_ATTENTION_NORM,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MLP_NORM,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_GATE,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W1,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128,SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT * SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W3,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128,SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT * SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_DOWN,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128,SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT * SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_UP,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_DOWN,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT,SPARK_QWEN38_STAGEPACK_CLASS_EVERY_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_QKV,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_GDN_CONV_CHANNELS,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_GATE,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_GDN_VALUE_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_BETA,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DECAY,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_OUTPUT,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_GDN_VALUE_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_CONV_WEIGHT,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_GDN_CONV_CHANNELS,SPARK_QWEN38_MODEL_GDN_CONV_KERNEL),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_A_LOG,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32,1u,SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DT_BIAS,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32,1u,SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_GDN_NORM,SPARK_QWEN38_STAGEPACK_CLASS_GDN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_GDN_HEAD_VALUE_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY,SPARK_QWEN38_STAGEPACK_CLASS_ATTN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,2u * SPARK_QWEN38_MODEL_ATTN_QUERY_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY,SPARK_QWEN38_STAGEPACK_CLASS_ATTN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_ATTN_KV_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_VALUE,SPARK_QWEN38_STAGEPACK_CLASS_ATTN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_ATTN_KV_DIMENSION,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_OUTPUT,SPARK_QWEN38_STAGEPACK_CLASS_ATTN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_ATTN_QUERY_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM,SPARK_QWEN38_STAGEPACK_CLASS_ATTN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_ATTN_HEAD_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM,SPARK_QWEN38_STAGEPACK_CLASS_ATTN_LAYER,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_ATTN_HEAD_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC,SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,2u * SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MTP_EMBED_NORM,SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MTP_HIDDEN_NORM,SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),
	SPARK_QWEN38_GEOM_ROW(SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FINAL_NORM,SPARK_QWEN38_STAGEPACK_CLASS_GLOBAL,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,1u,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION)
};

#undef SPARK_QWEN38_GEOM_ROW

/* Every kind has a row, so the only failure is a kind outside the enum. */
static inline int32_t SparkQwen38StagePackTensorShapeOf(uint32_t tensor_kind, SparkQwen38StagePackTensorShape *shape)
{
	const SparkQwen38StagePackKindGeometry *geometry;
	if ( tensor_kind >= SPARK_QWEN38_STAGEPACK_TENSOR_KIND_COUNT )
		return(-1);
	geometry = &SPARK_QWEN38_STAGEPACK_KIND_GEOMETRY[tensor_kind];
	shape->rows = geometry->rows;
	shape->columns = geometry->columns;
	shape->natural_format = geometry->natural_format;
	shape->layer_class = geometry->layer_class;
	return(0);
}

/*
 * Kind resolved against a concrete layer: a global kind must carry the global
 * layer marker, a per-layer kind must sit inside the total layer space, and
 * the GDN/attention classes must agree with the hybrid layer map. The MTP
 * decoder is geometry-identical to a full-attention layer, so its sixteen
 * layer-shaped tensors REUSE the per-layer kinds at the reserved MTP marker.
 */
static inline int32_t SparkQwen38StagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkQwen38StagePackTensorShape *shape)
{
	if ( SparkQwen38StagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	/* The resolution tail - MTP marker admission, global/class agreement,
	 * stack bound, hybrid-map class check - is the shared core's, with its
	 * exact refusal codes (-6,-2,-3,-4,-5). */
	return(SparkHybridStagePackResolveLayerClass(shape->layer_class,is_global,
		layer_index == SPARK_QWEN38_STAGEPACK_MTP_LAYER ? 1u : 0u,
		layer_index,SPARK_QWEN38_MODEL_LAYER_COUNT,
		SPARK_QWEN38_MODEL_LAYER_IS_GDN(layer_index)));
}

/* Byte accounting is the shared hybrid core's; this forward maps the
 * family's wire format enum onto the normalized weight classes. */
static inline uint32_t SparkQwen38StagePackWeightClass(uint32_t weight_format)
{
	if ( weight_format == SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_MXFP4_E2M1);
	if ( weight_format == SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_FP8_E4M3_F32B128);
	if ( weight_format == SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_F32);
	if ( weight_format == SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(SPARK_HYBRID_STAGEPACK_WEIGHT_U32);
	return(SPARK_HYBRID_STAGEPACK_WEIGHT_BF16);
}

static inline uint64_t SparkQwen38StagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	return(SparkHybridStagePackPayloadBytes(SparkQwen38StagePackWeightClass(weight_format),rows,columns));
}

static inline uint64_t SparkQwen38StagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	return(SparkHybridStagePackScaleBytes(SparkQwen38StagePackWeightClass(weight_format),rows,columns));
}
