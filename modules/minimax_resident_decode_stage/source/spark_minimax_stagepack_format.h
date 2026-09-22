#pragma once

#include <stdint.h>
#include <stddef.h>

#include "sparkpipe/spark_minimax_model.h"
#include "sparkpipe/spark_status.h"

#define SPARK_MINIMAX_STAGEPACK_MAGIC 0x58544E4Du
#define SPARK_MINIMAX_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_MINIMAX_STAGEPACK_HEADER_BYTES 120u
#define SPARK_MINIMAX_STAGEPACK_ENTRY_BYTES 56u
#define SPARK_MINIMAX_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_MINIMAX_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkMinimaxStagePackTensorKind
{
	SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_MINIMAX_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD = 2,
	SPARK_MINIMAX_STAGEPACK_TENSOR_INPUT_NORM = 3,
	SPARK_MINIMAX_STAGEPACK_TENSOR_POST_ATTENTION_NORM = 4,
	SPARK_MINIMAX_STAGEPACK_TENSOR_Q_NORM = 5,
	SPARK_MINIMAX_STAGEPACK_TENSOR_K_NORM = 6,
	SPARK_MINIMAX_STAGEPACK_TENSOR_QUERY = 7,
	SPARK_MINIMAX_STAGEPACK_TENSOR_KEY = 8,
	SPARK_MINIMAX_STAGEPACK_TENSOR_VALUE = 9,
	SPARK_MINIMAX_STAGEPACK_TENSOR_OUTPUT = 10,
	SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_GATE = 11,
	SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_UP = 12,
	SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_DOWN = 13,
	SPARK_MINIMAX_STAGEPACK_TENSOR_KIND_COUNT = 14
} SparkMinimaxStagePackTensorKind;

#define SPARK_MINIMAX_STAGEPACK_WEIGHT_BF16 0u

typedef struct SparkMinimaxStagePackHeader
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
	uint32_t query_head_count;
	uint32_t kv_head_count;
	uint32_t head_dimension;
	uint32_t ffn_intermediate_dimension;
	uint32_t output_vocab_count;
	uint32_t mtp_layer_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t reserved0;
	uint32_t reserved1;
	uint32_t reserved2;
	uint32_t reserved3;
	uint32_t reserved4;
	uint32_t reserved5;
	uint32_t reserved6;
	uint32_t reserved7;
	uint32_t reserved8;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkMinimaxStagePackHeader;

typedef struct SparkMinimaxStagePackEntry
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
} SparkMinimaxStagePackEntry;

_Static_assert(sizeof(SparkMinimaxStagePackHeader) == SPARK_MINIMAX_STAGEPACK_HEADER_BYTES,"minimax stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkMinimaxStagePackEntry) == SPARK_MINIMAX_STAGEPACK_ENTRY_BYTES,"minimax stage pack directory entry must be 56 wire bytes");
_Static_assert(offsetof(SparkMinimaxStagePackHeader,directory_offset) == 104u,"minimax pack directory offset must sit at wire word 26");
_Static_assert(offsetof(SparkMinimaxStagePackHeader,file_bytes) == 112u,"minimax pack file bytes must sit at wire word 27");

_Static_assert((SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION % 2u) == 0u,"minimax hidden must pair for bf16 words");
_Static_assert((SPARK_MINIMAX_TEXT_LAYER_COUNT % 4u) == 0u,"minimax layer count must cover whole 3:1 sliding periods");
_Static_assert((SPARK_MINIMAX_TEXT_ATTENTION_HEAD_COUNT % SPARK_MINIMAX_TEXT_KV_HEAD_COUNT) == 0u,"minimax query heads must group evenly onto kv heads");
_Static_assert(SPARK_MINIMAX_TEXT_QUERY_DIMENSION == 8192u,"minimax query projection width is 64 heads by 128");
_Static_assert(SPARK_MINIMAX_TEXT_KV_DIMENSION == 1024u,"minimax kv projection width is 8 heads by 128");
_Static_assert((SPARK_MINIMAX_TEXT_DENSE_INTERMEDIATE_DIMENSION % 256u) == 0u,"minimax ffn intermediate must tile the payload alignment");

typedef struct SparkMinimaxStagePackTensorShape
{
	uint32_t rows;
	uint32_t columns;
	uint32_t row_base;
	uint32_t column_base;
} SparkMinimaxStagePackTensorShape;

static inline void SparkMinimaxStagePackFullShape(uint32_t tensor_kind,SparkMinimaxStagePackTensorShape *shape)
{
	shape->row_base = 0u;
	shape->column_base = 0u;
	switch ( tensor_kind )
	{
	case SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING:
	case SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD:
		shape->rows = SPARK_MINIMAX_TEXT_VOCAB_COUNT;
		shape->columns = SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION;
		return;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FINAL_NORM:
	case SPARK_MINIMAX_STAGEPACK_TENSOR_INPUT_NORM:
	case SPARK_MINIMAX_STAGEPACK_TENSOR_POST_ATTENTION_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION;
		return;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_Q_NORM:
	case SPARK_MINIMAX_STAGEPACK_TENSOR_K_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_MINIMAX_TEXT_HEAD_DIMENSION;
		return;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_QUERY:
		shape->rows = SPARK_MINIMAX_TEXT_QUERY_DIMENSION;
		shape->columns = SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION;
		return;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_KEY:
	case SPARK_MINIMAX_STAGEPACK_TENSOR_VALUE:
		shape->rows = SPARK_MINIMAX_TEXT_KV_DIMENSION;
		shape->columns = SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION;
		return;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_OUTPUT:
		shape->rows = SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_TEXT_QUERY_DIMENSION;
		return;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_GATE:
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_UP:
		shape->rows = SPARK_MINIMAX_TEXT_DENSE_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION;
		return;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_DOWN:
		shape->rows = SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION;
		shape->columns = SPARK_MINIMAX_TEXT_DENSE_INTERMEDIATE_DIMENSION;
		return;
	default:
		shape->rows = 0u;
		shape->columns = 0u;
		return;
	}
}

static inline uint32_t SparkMinimaxStagePackKindIsRowSharded(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_QUERY ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_KEY ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_VALUE ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_GATE ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_UP ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD);
}

static inline uint32_t SparkMinimaxStagePackKindIsColumnSharded(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_OUTPUT ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_DOWN);
}

static inline uint32_t SparkMinimaxStagePackKindIsGlobal(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_FINAL_NORM ||
		tensor_kind == SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD);
}

static inline uint32_t SparkMinimaxStagePackLayerTensorCount(void)
{
	return(SPARK_MINIMAX_STAGEPACK_TENSOR_KIND_COUNT - 3u);
}

static inline uint64_t SparkMinimaxStagePackPayloadBytes(uint32_t rows,uint32_t columns)
{
	return((uint64_t)rows * (uint64_t)columns * SPARK_MINIMAX_TEXT_BF16_ELEMENT_BYTES);
}

static inline void SparkMinimaxStagePackNarrowShape(uint32_t tensor_kind,uint32_t tp_degree,uint32_t tp_rank,SparkMinimaxStagePackTensorShape *shape)
{
	SparkMinimaxStagePackFullShape(tensor_kind,shape);
	if ( tp_degree <= 1u )
		return;
	if ( SparkMinimaxStagePackKindIsRowSharded(tensor_kind) != 0u )
	{
		shape->rows /= tp_degree;
		shape->row_base = tp_rank * shape->rows;
	}
	if ( SparkMinimaxStagePackKindIsColumnSharded(tensor_kind) != 0u )
	{
		shape->columns /= tp_degree;
		shape->column_base = tp_rank * shape->columns;
	}
}

static inline uint32_t SparkMinimaxStagePackExpectedTensorCount(uint32_t layer_count)
{
	return(3u + layer_count * SparkMinimaxStagePackLayerTensorCount());
}
