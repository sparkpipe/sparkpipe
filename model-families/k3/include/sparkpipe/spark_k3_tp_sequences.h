#pragma once

#include <stdint.h>

#include "sparkpipe/spark_k3_kv_shard.h"

#define SPARK_K3_TP_ROW_ELEMENTS SPARK_K3_MODEL_HIDDEN_DIMENSION
#define SPARK_K3_TP_GATE_UP_ROW_ELEMENTS \
	(SPARK_K3_MODEL_MOE_TOP_K * SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION * 2u)

SPARK_K3_KV_SHARD_FN uint32_t SparkK3TpSequences(uint64_t elements)
{
	return((uint32_t)((elements + SPARK_K3_TP_ROW_ELEMENTS - 1u) / SPARK_K3_TP_ROW_ELEMENTS));
}

SPARK_K3_KV_SHARD_FN uint32_t SparkK3TpSequenceCapacity(uint32_t rows,uint32_t degree)
{
	uint32_t capacity = SparkK3KvShardSequenceCapacity(rows,degree);
	uint32_t gate_up = SparkK3TpSequences((uint64_t)rows * SPARK_K3_TP_GATE_UP_ROW_ELEMENTS);
	uint32_t shared = SparkK3TpSequences((uint64_t)rows * SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION);
	capacity = gate_up > capacity ? gate_up : capacity;
	return(shared > capacity ? shared : capacity);
}
