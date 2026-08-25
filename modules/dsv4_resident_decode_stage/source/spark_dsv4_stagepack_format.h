#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_weight_codec.h"

/*
 * DeepSeek V4 stage pack format, one table for BOTH variants: this header
 * deliberately includes NO model header. A translation unit selects Flash
 * or Pro by including spark_dsv4_model.h or spark_dsv4_pro_model.h FIRST
 * (they share an include guard, so mixing is structurally impossible), or
 * the build passes -include for the variant; every shape below reads the
 * variant's macros. The variant identifier in the header is the layer
 * count itself - 43 and 61 cannot collide.
 *
 * Directory economics: the routed experts are STACKED - one tensor kind
 * carries all n_routed_experts of a projection (rows = experts x out), so
 * the directory stays O(layers), not O(layers x experts). Scales travel in
 * the entry beside their payload exactly as in the qwen38_27b format: fp8
 * weights carry one e8m0 byte per 128-column block per row, fp4 experts
 * one e8m0 byte per 32-column block per row. The hash router's tid2eid
 * table is a u32 tensor like any other. MTP kinds remain reserved for a
 * future native DSpark implementation. Flash GA baseline packs set
 * mtp_layer_count to zero and carry no mtp.* tensors; all three checkpoint
 * DSpark layers are excluded. The GA head pack therefore carries no old
 * MTP-only embedding copy; the embedding stays solely on stage zero.
 */

#define SPARK_DSV4_STAGEPACK_MAGIC 0x34565344u
#define SPARK_DSV4_STAGEPACK_FORMAT_VERSION 4u
#define SPARK_DSV4_STAGEPACK_GLOBAL_LAYER UINT32_MAX
/* Three DSpark draft layers (checkpoint namespace mtp.0..2, attached to main
 * layers 40-42). Encoded as layer indices MTP_LAYER_FIRST+stage so the
 * standard per-layer tensor kinds carry the draft transformer weights; the
 * MTP_* kinds below carry only the draft-only extras. */
#define SPARK_DSV4_STAGEPACK_MTP_LAYER_FIRST (UINT32_MAX - 4u)
#define SPARK_DSV4_STAGEPACK_MTP_LAYER_LAST (UINT32_MAX - 2u)
#define SPARK_DSV4_STAGEPACK_MTP_LAYER_COUNT_MAX 3u
#define SPARK_DSV4_STAGEPACK_MTP_LAYER(stage) (SPARK_DSV4_STAGEPACK_MTP_LAYER_FIRST + (stage))
#define SPARK_DSV4_STAGEPACK_HEADER_BYTES ((uint32_t)sizeof(SparkDsv4StagePackHeader))
#define SPARK_DSV4_STAGEPACK_ENTRY_BYTES ((uint32_t)sizeof(SparkDsv4StagePackEntry))

#define SPARK_DSV4_STAGEPACK_WEIGHT_BF16 0u
#define SPARK_DSV4_STAGEPACK_WEIGHT_F32 1u
#define SPARK_DSV4_STAGEPACK_WEIGHT_U32 2u
/* Keep these wire values identical to spark_lm_kernels.cuh. */
#define SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1 3u
#define SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3 4u

#define SPARK_DSV4_STAGEPACK_FP8_SCALE_BLOCK 128u
#define SPARK_DSV4_STAGEPACK_FP4_SCALE_BLOCK 32u

#define SPARK_DSV4_STAGEPACK_CLASS_GLOBAL 0u
#define SPARK_DSV4_STAGEPACK_CLASS_EVERY_LAYER 1u
#define SPARK_DSV4_STAGEPACK_CLASS_COMPRESS_LAYER 2u
#define SPARK_DSV4_STAGEPACK_CLASS_CSA_LAYER 3u

typedef enum SparkDsv4StagePackTensorKind
{
	SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK = 0,
	SPARK_DSV4_STAGEPACK_TENSOR_WQ_A = 1,
	SPARK_DSV4_STAGEPACK_TENSOR_Q_NORM = 2,
	SPARK_DSV4_STAGEPACK_TENSOR_WQ_B = 3,
	SPARK_DSV4_STAGEPACK_TENSOR_WKV = 4,
	SPARK_DSV4_STAGEPACK_TENSOR_KV_NORM = 5,
	SPARK_DSV4_STAGEPACK_TENSOR_WO_A = 6,
	SPARK_DSV4_STAGEPACK_TENSOR_WO_B = 7,
	SPARK_DSV4_STAGEPACK_TENSOR_ATTN_NORM = 8,
	SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM = 9,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_FN = 10,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_FN = 11,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_BASE = 12,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_BASE = 13,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_SCALE = 14,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_SCALE = 15,
	SPARK_DSV4_STAGEPACK_TENSOR_GATE_WEIGHT = 16,
	SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS = 17,
	SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID = 18,
	SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1 = 19,
	SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2 = 20,
	SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3 = 21,
	SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W1 = 22,
	SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W2 = 23,
	SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3 = 24,
	SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE = 25,
	SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV = 26,
	SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE = 27,
	SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM = 28,
	SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B = 29,
	SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS = 30,
	SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE = 31,
	SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV = 32,
	SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE = 33,
	SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM = 34,
	SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING = 35,
	SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM = 36,
	SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD = 37,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_FN = 38,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_BASE = 39,
	SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE = 40,
	/* Draft-stage extras (reference: inference/model.py DSparkBlock).
	 * Stage 0 carries the target-hidden projection; stage 2 carries the
	 * output norm, the 5-token hc head, the markov logits-bias head, and
	 * the acceptance confidence head. The draft transformer tensors ride
	 * the standard kinds above under layer MTP_LAYER_FIRST+stage. */
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_PROJ = 41,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_NORM = 42,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_FINAL_NORM = 43,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_FN = 44,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_BASE = 45,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE = 46,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_MARKOV_W1 = 47,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_MARKOV_W2 = 48,
	SPARK_DSV4_STAGEPACK_TENSOR_MTP_CONFIDENCE_PROJ = 49,
	SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT = 50
} SparkDsv4StagePackTensorKind;

typedef struct SparkDsv4StagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t codec_abi_version;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t kv_cache_codec;
	uint32_t tensor_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t total_layer_count;
	uint32_t hidden_dimension;
	uint32_t vocab_count;
	uint32_t routed_expert_count;
	uint32_t mtp_layer_count;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkDsv4StagePackHeader;

typedef struct SparkDsv4StagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint32_t reserved0;
	uint64_t payload_offset;
	uint64_t scale_offset;
} SparkDsv4StagePackEntry;

typedef struct SparkDsv4StagePackTensorShape
{
	uint32_t rows;
	uint32_t columns;
	uint32_t weight_format;
	uint32_t layer_class;
} SparkDsv4StagePackTensorShape;

static inline uint32_t SparkDsv4StagePackLayerIsMtp(uint32_t layer_index)
{
	return(layer_index >= SPARK_DSV4_STAGEPACK_MTP_LAYER_FIRST &&
		layer_index <= SPARK_DSV4_STAGEPACK_MTP_LAYER_LAST ? 1u : 0u);
}

static inline uint32_t SparkDsv4StagePackMtpStage(uint32_t layer_index)
{
	return(layer_index - SPARK_DSV4_STAGEPACK_MTP_LAYER_FIRST);
}

// The MoE routing split, pinned: layers below HASH_ROUTED_LAYER_COUNT ship
// the tid2eid lookup and NO balancer bias; all others the reverse. The MTP
// layers sit past the hash range, so they score-route.
static inline uint32_t SparkDsv4StagePackLayerIsHashRouted(uint32_t layer_index)
{
	return(SparkDsv4StagePackLayerIsMtp(layer_index) == 0u &&
		layer_index < SPARK_DSV4_MODEL_HASH_ROUTED_LAYER_COUNT ? 1u : 0u);
}

static inline uint32_t SparkDsv4StagePackLayerKind(uint32_t layer_index)
{
	if ( SparkDsv4StagePackLayerIsMtp(layer_index) != 0u )
		return(SPARK_DSV4_MODEL_MTP_LAYER_KIND);
	return(SparkDsv4ModelLayerKind(layer_index));
}

// Shapes for kinds whose geometry is layer-independent. Kinds that vary
// with the layer's attention kind (the compressor's overlap doubling) are
// resolved in SparkDsv4StagePackResolvedShape against a concrete layer.
static inline int32_t SparkDsv4StagePackShapeOfGlobal(uint32_t tensor_kind, SparkDsv4StagePackTensorShape *shape);

static inline int32_t SparkDsv4StagePackShapeOfLayer(uint32_t tensor_kind, SparkDsv4StagePackTensorShape *shape)
{
	memset(shape,0,sizeof(*shape));
	shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_BF16;
	shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_A: shape->rows = SPARK_DSV4_MODEL_QUERY_LORA_RANK; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_Q_NORM: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_QUERY_LORA_RANK; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_B: shape->rows = SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION; shape->columns = SPARK_DSV4_MODEL_QUERY_LORA_RANK; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_WKV: shape->rows = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_KV_NORM: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_A: shape->rows = SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK; shape->columns = SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_B: shape->rows = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->columns = SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_NORM:
	case SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_FN:
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_FN: shape->rows = SPARK_DSV4_MODEL_HC_MIX_ROWS; shape->columns = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_BASE:
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_BASE: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_HC_MIX_ROWS; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_SCALE:
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_SCALE: shape->rows = 1u; shape->columns = 3u; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_WEIGHT: shape->rows = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID: shape->rows = SPARK_DSV4_MODEL_VOCAB_COUNT; shape->columns = SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_U32; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1:
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3: shape->rows = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2: shape->rows = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->columns = SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W1:
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3: shape->rows = SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W2: shape->rows = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->columns = SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_COMPRESS_LAYER; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B: shape->rows = SPARK_DSV4_MODEL_INDEX_DIMENSION; shape->columns = SPARK_DSV4_MODEL_QUERY_LORA_RANK; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_CSA_LAYER; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS: shape->rows = SPARK_DSV4_MODEL_INDEX_HEAD_COUNT; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_CSA_LAYER; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE: shape->rows = SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO; shape->columns = SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_CSA_LAYER; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV:
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE: shape->rows = SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_CSA_LAYER; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_CSA_LAYER; return(0);
	default:
		return(SparkDsv4StagePackShapeOfGlobal(tensor_kind,shape));
	}
}

static inline int32_t SparkDsv4StagePackShapeOf(uint32_t tensor_kind, SparkDsv4StagePackTensorShape *shape)
{
	if ( tensor_kind < SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING )
		return(SparkDsv4StagePackShapeOfLayer(tensor_kind,shape));
	return(SparkDsv4StagePackShapeOfGlobal(tensor_kind,shape));
}

static inline int32_t SparkDsv4StagePackShapeOfGlobal(uint32_t tensor_kind, SparkDsv4StagePackTensorShape *shape)
{
	memset(shape,0,sizeof(*shape));
	shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_BF16;
	shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD: shape->rows = SPARK_DSV4_MODEL_VOCAB_COUNT; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM:
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_NORM:
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_FINAL_NORM: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_FN:
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_FN: shape->rows = SPARK_DSV4_MODEL_HC_STREAM_COUNT; shape->columns = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_BASE:
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_BASE: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_HC_STREAM_COUNT; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE:
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE: shape->rows = 1u; shape->columns = 1u; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_F32; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_PROJ: shape->rows = SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->columns = SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION; shape->weight_format = SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MARKOV_W1:
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MARKOV_W2: shape->rows = SPARK_DSV4_MODEL_VOCAB_COUNT; shape->columns = SPARK_DSV4_MODEL_DSPARK_MARKOV_RANK; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_CONFIDENCE_PROJ: shape->rows = 1u; shape->columns = SPARK_DSV4_MODEL_HIDDEN_DIMENSION + SPARK_DSV4_MODEL_DSPARK_MARKOV_RANK; shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_GLOBAL; return(0);
	default:
		return(-1);
	}
}

/*
 * Kind resolved against a concrete layer. The compressor's APE/WKV/WGATE
 * widths depend on the layer's attention kind: the ratio-4 compressor
 * overlaps and doubles its channels, the ratio-128 one does not - so those
 * three kinds resolve here, everything else defers to the flat table. The
 * routing tensors split by the hash pin, the MTP marker admits every-layer
 * and compress-independent kinds only (its kind is SWA), and the classes
 * must agree with the variant's layer-kind map.
 */
static inline int32_t SparkDsv4StagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkDsv4StagePackTensorShape *shape)
{
	uint32_t kind = SparkDsv4StagePackLayerKind(layer_index),ratio,overlap;
	if ( is_global == 0u && (tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE || tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV || tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE) )
	{
		if ( SparkDsv4StagePackLayerIsMtp(layer_index) == 0u && layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT )
			return(-2);
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
			return(-3);
		ratio = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
		overlap = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR : 1u;
		memset(shape,0,sizeof(*shape));
		shape->layer_class = SPARK_DSV4_STAGEPACK_CLASS_COMPRESS_LAYER;
		shape->weight_format = tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE ? SPARK_DSV4_STAGEPACK_WEIGHT_F32 : SPARK_DSV4_STAGEPACK_WEIGHT_BF16;
		shape->rows = tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE ? ratio : overlap * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
		shape->columns = tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE ? overlap * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION : SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
		return(0);
	}
	if ( SparkDsv4StagePackShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	if ( (shape->layer_class == SPARK_DSV4_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-4);
	if ( is_global != 0u )
		return(0);
	if ( SparkDsv4StagePackLayerIsMtp(layer_index) == 0u && layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT )
		return(-5);
	if ( tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS && SparkDsv4StagePackLayerIsHashRouted(layer_index) != 0u )
		return(-6);
	if ( tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID && SparkDsv4StagePackLayerIsHashRouted(layer_index) == 0u )
		return(-7);
	if ( shape->layer_class == SPARK_DSV4_STAGEPACK_CLASS_COMPRESS_LAYER && kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		return(-8);
	if ( shape->layer_class == SPARK_DSV4_STAGEPACK_CLASS_CSA_LAYER && kind != SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		return(-9);
	return(0);
}

static inline uint64_t SparkDsv4StagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1 )
		return(elements / 2u);
	if ( weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_F32 || weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_U32 )
		return(elements * 4u);
	if ( weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3 )
		return(elements);
	return(elements * (uint64_t)SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES);
}

// e8m0 scales, one byte per block per row: fp8 weights block 128 columns,
// fp4 experts block 32, everything else scale-free.
static inline uint64_t SparkDsv4StagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	if ( weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3 )
		return((uint64_t)rows * ((columns + SPARK_DSV4_STAGEPACK_FP8_SCALE_BLOCK - 1u) / SPARK_DSV4_STAGEPACK_FP8_SCALE_BLOCK));
	if ( weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1 )
		return((uint64_t)rows * ((columns + SPARK_DSV4_STAGEPACK_FP4_SCALE_BLOCK - 1u) / SPARK_DSV4_STAGEPACK_FP4_SCALE_BLOCK));
	return(0u);
}

// Per-layer tensor population: 23 shared kinds (attention core, hc pair,
// norms, gate weight, the three stacked expert projections, the shared
// expert) plus exactly one routing tensor, plus the compressor quartet off
// SWA, plus the indexer sextet on CSA only.
static inline uint32_t SparkDsv4StagePackLayerTensorCount(uint32_t layer_index)
{
	uint32_t kind = SparkDsv4StagePackLayerKind(layer_index);
	uint32_t tensors = 23u + 1u;
	if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		tensors += 4u;
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		tensors += 6u;
	return(tensors);
}

static inline uint32_t SparkDsv4StagePackExpectedTensorCountForOwnership(
	uint32_t first_layer_index,
	uint32_t layer_count,
	uint32_t include_embedding,
	uint32_t include_final_globals)
{
	uint32_t layer,tensors = 0u;
	for (layer = first_layer_index; layer < first_layer_index + layer_count; layer++)
		tensors += SparkDsv4StagePackLayerTensorCount(layer);
	if ( include_embedding != 0u )
		tensors += 1u;
	if ( SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u )
	{
		uint32_t stage;
		/* DSpark draft: every rank's pack carries the full draft layers
		 * (they ride the standard per-layer kinds under the MTP range)
		 * plus the 9 draft-only extras, so every geometry admits them. */
		for (stage = 0u; stage < SPARK_DSV4_MODEL_MTP_LAYER_COUNT; stage++)
			tensors += SparkDsv4StagePackLayerTensorCount(SPARK_DSV4_STAGEPACK_MTP_LAYER(stage));
		tensors += 9u;
	}
	if ( include_final_globals != 0u )
	{
		tensors += 5u;
		if ( include_embedding == 0u )
			tensors += 1u;
	}
	return(tensors);
}

static inline uint32_t SparkDsv4StagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	return(SparkDsv4StagePackExpectedTensorCountForOwnership(
		first_layer_index,
		layer_count,
		first_layer_index == 0u ? 1u : 0u,
		first_layer_index + layer_count == SPARK_DSV4_MODEL_LAYER_COUNT ? 1u : 0u));
}

static inline void SparkDsv4StagePackExpectedGeometry(SparkDsv4StagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	memset(header,0,sizeof(*header));
	header->magic = SPARK_DSV4_STAGEPACK_MAGIC;
	header->format_version = SPARK_DSV4_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_DSV4_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_DSV4_STAGEPACK_ENTRY_BYTES;
	header->codec_abi_version = SPARK_WEIGHT_CODEC_ABI_VERSION;
	header->linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC;
	header->expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC;
	header->kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC;
	header->tensor_count = SparkDsv4StagePackExpectedTensorCount(first_layer_index,layer_count);
	header->first_layer_index = first_layer_index;
	header->layer_count = layer_count;
	header->total_layer_count = SPARK_DSV4_MODEL_LAYER_COUNT;
	header->hidden_dimension = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	header->vocab_count = SPARK_DSV4_MODEL_VOCAB_COUNT;
	header->routed_expert_count = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT;
	header->mtp_layer_count = SPARK_DSV4_MODEL_MTP_LAYER_COUNT;
	header->directory_offset = SPARK_DSV4_STAGEPACK_HEADER_BYTES;
}

// Field-by-field comparison; each field owns a unique negative code so the
// loader's refusal names exactly what disagreed.
static inline int32_t SparkDsv4StagePackCompareGeometry(const SparkDsv4StagePackHeader *file_header, const SparkDsv4StagePackHeader *expected)
{
	if ( file_header->magic != expected->magic )
		return(-1);
	if ( file_header->format_version != expected->format_version )
		return(-2);
	if ( file_header->header_bytes != expected->header_bytes )
		return(-3);
	if ( file_header->directory_entry_bytes != expected->directory_entry_bytes )
		return(-4);
	if ( file_header->codec_abi_version != expected->codec_abi_version )
		return(-5);
	if ( file_header->linear_weight_codec != expected->linear_weight_codec )
		return(-6);
	if ( file_header->expert_weight_codec != expected->expert_weight_codec )
		return(-7);
	if ( file_header->kv_cache_codec != expected->kv_cache_codec )
		return(-8);
	if ( file_header->tensor_count != expected->tensor_count )
		return(-9);
	if ( file_header->first_layer_index != expected->first_layer_index )
		return(-10);
	if ( file_header->layer_count != expected->layer_count )
		return(-11);
	if ( file_header->total_layer_count != expected->total_layer_count )
		return(-12);
	if ( file_header->hidden_dimension != expected->hidden_dimension )
		return(-13);
	if ( file_header->vocab_count != expected->vocab_count )
		return(-14);
	if ( file_header->routed_expert_count != expected->routed_expert_count )
		return(-15);
	if ( file_header->mtp_layer_count != expected->mtp_layer_count )
		return(-16);
	return(0);
}

static inline const char *SparkDsv4StagePackGeometryFieldName(int32_t compare_code)
{
	static const char *names[16] =
	{
		"magic","format_version","header_bytes","directory_entry_bytes","codec_abi_version","linear_weight_codec",
		"expert_weight_codec","kv_cache_codec","tensor_count","first_layer_index","layer_count","total_layer_count",
		"hidden_dimension","vocab_count","routed_expert_count","mtp_layer_count"
	};
	return(compare_code <= -1 && compare_code >= -16 ? names[(-compare_code) - 1] : "unknown");
}
