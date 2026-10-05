#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"

#define SPARK_MIMO26_STAGEPACK_MAGIC 0x5036324Du
#define SPARK_MIMO26_STAGEPACK_FORMAT_VERSION 2u
#define SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_MIMO26_STAGEPACK_PAYLOAD_ALIGNMENT 256u
#define SPARK_MIMO26_STAGEPACK_HEADER_BYTES 120u
#define SPARK_MIMO26_STAGEPACK_ENTRY_BYTES 56u

typedef enum SparkMimo26StagePackTensorKind
{
	SPARK_MIMO26_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_MIMO26_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_MIMO26_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_MIMO26_STAGEPACK_TENSOR_ATTENTION_NORM = 3,
	SPARK_MIMO26_STAGEPACK_TENSOR_MLP_NORM = 4,
	SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE = 5,
	SPARK_MIMO26_STAGEPACK_TENSOR_SINK_BIAS = 22,
	SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE_BIAS = 23,
	SPARK_MIMO26_STAGEPACK_TENSOR_QKV = 24,
	SPARK_MIMO26_STAGEPACK_TENSOR_O_PROJ = 27,
	SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_GATE = 28,
	SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_UP = 29,
	SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_DOWN = 30,
	SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_GATE = 31,
	SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_UP = 32,
	SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_DOWN = 33,
	SPARK_MIMO26_STAGEPACK_TENSOR_KIND_COUNT = 34
} SparkMimo26StagePackTensorKind;

#define SPARK_MIMO26_STAGEPACK_WEIGHT_BF16 SPARK_STAGEPACK_FORMAT_WEIGHT_BF16
#define SPARK_MIMO26_STAGEPACK_WEIGHT_F32 SPARK_STAGEPACK_FORMAT_WEIGHT_F32
#define SPARK_MIMO26_STAGEPACK_WEIGHT_FP8_E4M3_F32B128 SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128
#define SPARK_MIMO26_STAGEPACK_WEIGHT_MXFP4_E2M1_E8M0G32 SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1_E8M0G32

typedef struct SparkMimo26StagePackHeader
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
} SparkMimo26StagePackHeader;

typedef struct SparkMimo26StagePackEntry
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
} SparkMimo26StagePackEntry;

_Static_assert(sizeof(SparkMimo26StagePackHeader) == SPARK_MIMO26_STAGEPACK_HEADER_BYTES,"mimo26 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkMimo26StagePackEntry) == SPARK_MIMO26_STAGEPACK_ENTRY_BYTES,"mimo26 stage pack directory entry must be 56 wire bytes");
SPARK_STAGEPACK_HEADER_LAYOUT_PROOF(SparkMimo26StagePackHeader);

static inline uint32_t SparkMimo26StagePackIsGlobal(uint32_t tensor_kind)
{
	return(tensor_kind <= SPARK_MIMO26_STAGEPACK_TENSOR_LM_HEAD ? 1u : 0u);
}

static inline uint32_t SparkMimo26StagePackIsExpert(uint32_t tensor_kind)
{
	return(tensor_kind >= SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_GATE
	       && tensor_kind <= SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_DOWN ? 1u : 0u);
}

static inline uint32_t SparkMimo26StagePackExpectedTensorCount(uint32_t layer_count, uint32_t swa_layer_count, uint32_t moe_layer_count)
{
	return(3u + layer_count * 4u + swa_layer_count + moe_layer_count * 5u + (layer_count - moe_layer_count) * 3u);
}

static inline uint32_t SparkMimo26StagePackQkvRankRows(uint32_t query_heads, uint32_t kv_heads, uint32_t head_dimension, uint32_t value_dimension, uint32_t ranks)
{
	return((query_heads / ranks) * head_dimension + (kv_heads / ranks) * (head_dimension + value_dimension));
}

static inline uint32_t SparkMimo26StagePackQkvRankScaleRows(uint32_t rank_rows)
{
	return((rank_rows + 127u) / 128u);
}

static inline int32_t SparkMimo26StagePackHeaderMatches(const SparkMimo26StagePackHeader *file_header,
	const SparkMimo26StagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}
