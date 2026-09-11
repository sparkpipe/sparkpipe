#pragma once

#include <stdint.h>

#include "spark_hy4_stagepack_format.h"

#define SPARK_HY4_FP8_SCALE_GROUP SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE
#define SPARK_HY4_FP8_KV_B_HEAD_ROWS \
	(SPARK_HY4_MODEL_QK_NOPE_HEAD_DIMENSION + \
	SPARK_HY4_MODEL_V_HEAD_DIMENSION)
#define SPARK_HY4_FP8_Q_B_GLOBAL_ROWS \
	(SPARK_HY4_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_HY4_MODEL_QK_HEAD_DIMENSION)
#define SPARK_HY4_FP8_KV_B_GLOBAL_ROWS \
	(SPARK_HY4_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_HY4_FP8_KV_B_HEAD_ROWS)
#define SPARK_HY4_FP8_WQ_B_GLOBAL_ROWS \
	(SPARK_HY4_MODEL_INDEX_HEAD_COUNT * SPARK_HY4_MODEL_INDEX_HEAD_DIMENSION)
#define SPARK_HY4_FP8_O_PROJ_GLOBAL_COLUMNS \
	(SPARK_HY4_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_HY4_MODEL_V_HEAD_DIMENSION)

typedef enum SparkHy4Fp8ScaleRule
{
	SPARK_HY4_FP8_SCALE_ALIGNED = 0,
	SPARK_HY4_FP8_SCALE_REPLICATED_ROWS = 1,
	SPARK_HY4_FP8_SCALE_REPLICATED_GROUPS = 2
} SparkHy4Fp8ScaleRule;

typedef struct SparkHy4Fp8ScaleContract
{
	uint32_t rule;
	uint32_t payload_rows;
	uint32_t payload_columns;
	uint32_t scale_rows;
	uint32_t scale_groups;
	uint32_t rank_row_offset;
	uint32_t rank_group_offset;
} SparkHy4Fp8ScaleContract;

static inline int32_t SparkHy4Fp8ScaleRuleOf(uint32_t tensor_kind,
	SparkHy4Fp8ScaleRule *rule, uint32_t *global_rows,
	uint32_t *global_columns)
{
	*global_rows = 0u;
	*global_columns = 0u;
	switch ( tensor_kind )
	{
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A:
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A:
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_WK:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_W1:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_DOWN:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_GATE:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_UP:
	case SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		*rule = SPARK_HY4_FP8_SCALE_ALIGNED;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_B:
		*rule = SPARK_HY4_FP8_SCALE_REPLICATED_ROWS;
		*global_rows = SPARK_HY4_FP8_Q_B_GLOBAL_ROWS;
		*global_columns = SPARK_HY4_MODEL_QUERY_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_K_B:
		*rule = SPARK_HY4_FP8_SCALE_REPLICATED_ROWS;
		*global_rows = SPARK_HY4_FP8_KV_B_GLOBAL_ROWS;
		*global_columns = SPARK_HY4_MODEL_KV_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_INDEX_WQ_B:
		*rule = SPARK_HY4_FP8_SCALE_REPLICATED_ROWS;
		*global_rows = SPARK_HY4_FP8_WQ_B_GLOBAL_ROWS;
		*global_columns = SPARK_HY4_MODEL_QUERY_LORA_RANK;
		return 0;
	case SPARK_HY4_STAGEPACK_TENSOR_ATTN_OUTPUT:
		*rule = SPARK_HY4_FP8_SCALE_REPLICATED_GROUPS;
		*global_rows = SPARK_HY4_MODEL_HIDDEN_DIMENSION;
		*global_columns = SPARK_HY4_FP8_O_PROJ_GLOBAL_COLUMNS;
		return 0;
	default:
		return -1;
	}
}

static inline int32_t SparkHy4Fp8ScaleContractOf(uint32_t tensor_kind,
	uint32_t rank, uint32_t ranks, uint32_t payload_rows,
	uint32_t payload_columns, SparkHy4Fp8ScaleContract *contract)
{
	SparkHy4Fp8ScaleRule rule;
	uint32_t global_rows, global_columns, local_groups;
	if ( contract == 0 )
		return -5;
	if ( SparkHy4Fp8ScaleRuleOf(tensor_kind,&rule,&global_rows,
		&global_columns) != 0 )
		return -1;
	if ( ranks == 0u || rank >= ranks )
		return -2;
	if ( payload_rows == 0u || payload_columns == 0u ||
	    (payload_columns % SPARK_HY4_FP8_SCALE_GROUP) != 0u )
		return -3;
	local_groups = payload_columns / SPARK_HY4_FP8_SCALE_GROUP;
	contract->rule = (uint32_t)rule;
	contract->payload_rows = payload_rows;
	contract->payload_columns = payload_columns;
	contract->scale_rows = payload_rows;
	contract->scale_groups = local_groups;
	contract->rank_row_offset = 0u;
	contract->rank_group_offset = 0u;
	if ( rule == SPARK_HY4_FP8_SCALE_ALIGNED )
		return 0;
	if ( rule == SPARK_HY4_FP8_SCALE_REPLICATED_ROWS )
	{
		if ( (global_rows % ranks) != 0u ||
		    payload_rows != (global_rows / ranks) ||
		    payload_columns != global_columns )
			return -6;
		contract->scale_rows = global_rows;
		contract->rank_row_offset = rank * payload_rows;
		return 0;
	}
	if ( (global_columns % ranks) != 0u ||
	    payload_columns != (global_columns / ranks) ||
	    payload_rows != global_rows )
		return -7;
	contract->scale_groups = global_columns / SPARK_HY4_FP8_SCALE_GROUP;
	contract->rank_group_offset = rank * local_groups;
	return 0;
}

static inline int32_t SparkHy4Fp8ScaleContractValidate(uint32_t tensor_kind,
	uint32_t rank, uint32_t ranks, uint32_t payload_rows,
	uint32_t payload_columns, uint32_t scale_rows, uint32_t scale_groups,
	SparkHy4Fp8ScaleContract *contract)
{
	SparkHy4Fp8ScaleContract local;
	int32_t status;
	if ( contract == 0 )
		contract = &local;
	status = SparkHy4Fp8ScaleContractOf(tensor_kind,rank,ranks,
	    payload_rows,payload_columns,contract);
	if ( status != 0 )
		return status;
	if ( contract->scale_rows != scale_rows ||
	    contract->scale_groups != scale_groups )
		return -4;
	return 0;
}

static inline uint32_t SparkHy4Fp8ScaleZeroCopy(
	const SparkHy4Fp8ScaleContract *contract)
{
	return contract->scale_groups ==
	    (contract->payload_columns / SPARK_HY4_FP8_SCALE_GROUP) ? 1u : 0u;
}

static inline uint64_t SparkHy4Fp8ScaleByteIndex(
	const SparkHy4Fp8ScaleContract *contract, uint32_t local_row,
	uint32_t local_group)
{
	return ((uint64_t)contract->rank_row_offset + (uint64_t)local_row) *
	    (uint64_t)contract->scale_groups +
	    (uint64_t)contract->rank_group_offset + (uint64_t)local_group;
}
