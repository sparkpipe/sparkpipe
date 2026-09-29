#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_tp_mesh_round_control.h"

#if defined(__CUDACC__)
#define SPARK_GLM5_NEXT_KV_SHARD_FN static inline __host__ __device__
#else
#define SPARK_GLM5_NEXT_KV_SHARD_FN static inline
#endif

#define SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_WIDE_UNIT \
	(SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT)
#define SPARK_GLM5_NEXT_KV_SHARD_RECORD_FLOATS \
	(SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION + 2u)
#define SPARK_GLM5_NEXT_KV_SHARD_GATHER_KEY_BYTES SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES
#define SPARK_GLM5_NEXT_KV_SHARD_SEQUENCE_BYTES (SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT * 2u)

typedef struct SparkGlm5NextKvShardRoundPlan
{
	uint32_t scatter_rounds;
	uint32_t gather_rounds;
	uint32_t gather_sequences;
	uint32_t gather;
	uint64_t scatter_wire_bytes;
	uint64_t gather_wire_bytes;
}
SparkGlm5NextKvShardRoundPlan;

SPARK_GLM5_NEXT_KV_SHARD_FN SparkKvShard SparkGlm5NextKvShardLatent(uint32_t rank,uint32_t degree)
{
	SparkKvShard shard;
	shard.degree = degree;
	shard.rank = rank;
	shard.grain = 1u;
	return(shard);
}

SPARK_GLM5_NEXT_KV_SHARD_FN SparkKvShard SparkGlm5NextKvShardIndex(uint32_t rank,uint32_t degree)
{
	SparkKvShard shard;
	shard.degree = degree;
	shard.rank = rank;
	shard.grain = SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL;
	return(shard);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardHeads(uint32_t degree)
{
	return(degree == 0u ? 0u : SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT / degree);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardQuerySequences(uint32_t rows,uint32_t degree)
{
	uint64_t elements = (uint64_t)rows * SparkGlm5NextKvShardHeads(degree) * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION;
	return((uint32_t)((elements + SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT - 1u) / SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint64_t SparkGlm5NextKvShardPartialElements(uint32_t rows,uint32_t degree)
{
	return((uint64_t)rows * SparkGlm5NextKvShardHeads(degree) * SPARK_GLM5_NEXT_KV_SHARD_RECORD_FLOATS * 2u);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardPartialWide(uint32_t rows,uint32_t degree,uint32_t capacity)
{
	uint64_t elements = SparkGlm5NextKvShardPartialElements(rows,degree);
	return((elements + SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT - 1u) / SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT > capacity ? 1u : 0u);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardPartialUnit(uint32_t rows,uint32_t degree,uint32_t capacity)
{
	return(SparkGlm5NextKvShardPartialWide(rows,degree,capacity) != 0u ? SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_WIDE_UNIT : SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardPartialSequences(uint32_t rows,uint32_t degree,uint32_t capacity)
{
	uint64_t unit = SparkGlm5NextKvShardPartialUnit(rows,degree,capacity);
	return((uint32_t)((SparkGlm5NextKvShardPartialElements(rows,degree) + unit - 1u) / unit));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint64_t SparkGlm5NextKvShardQueryStride(uint32_t rows,uint32_t degree)
{
	return((uint64_t)SparkGlm5NextKvShardQuerySequences(rows,degree) * SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint64_t SparkGlm5NextKvShardPartialStride(uint32_t rows,uint32_t degree,uint32_t capacity)
{
	return((uint64_t)SparkGlm5NextKvShardPartialSequences(rows,degree,capacity) * (SparkGlm5NextKvShardPartialUnit(rows,degree,capacity) / 2u));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint64_t SparkGlm5NextKvShardPartialStrideCapacity(uint32_t degree,uint32_t capacity)
{
	uint64_t elements = SparkGlm5NextKvShardPartialElements(capacity,degree);
	uint64_t narrow = (elements + SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT - 1u) / SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT * (SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT / 2u);
	uint64_t wide = (elements + SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_WIDE_UNIT - 1u) / SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_WIDE_UNIT * (SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_WIDE_UNIT / 2u);
	return(narrow > wide ? narrow : wide);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardFits(uint32_t degree,uint32_t rows)
{
	if ( degree < 2u || degree > SPARK_KV_SHARD_MAX_DEGREE || rows == 0u || SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT % degree != 0u ||
		SparkKvShardValid(SparkGlm5NextKvShardLatent(0u,degree),SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS) == 0u ||
		SparkKvShardValid(SparkGlm5NextKvShardIndex(0u,degree),SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS) == 0u )
		return(0u);
	return(SparkGlm5NextKvShardQuerySequences(rows,degree) <= rows && SparkGlm5NextKvShardPartialSequences(rows,degree,rows) <= rows ? 1u : 0u);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardGatherSequences(uint64_t keys)
{
	return((uint32_t)((keys * SPARK_GLM5_NEXT_KV_SHARD_GATHER_KEY_BYTES + SPARK_GLM5_NEXT_KV_SHARD_SEQUENCE_BYTES - 1u) / SPARK_GLM5_NEXT_KV_SHARD_SEQUENCE_BYTES));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint64_t SparkGlm5NextKvShardGatherStride(uint32_t sequences)
{
	return((uint64_t)sequences * SPARK_GLM5_NEXT_KV_SHARD_SEQUENCE_BYTES);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardGatherCapacity(uint32_t degree,uint32_t capacity)
{
	uint64_t send = (uint64_t)degree * SparkGlm5NextKvShardQueryStride(capacity,degree) / SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT;
	uint64_t receive = SparkGlm5NextKvShardPartialStrideCapacity(degree,capacity) * 2u / SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT;
	uint64_t fit = send < receive ? send : receive;
	return((uint32_t)(fit < capacity ? fit : capacity));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardPlanRounds(uint32_t rows,uint32_t degree,uint32_t capacity,uint64_t gather_keys,uint64_t slot_bytes,SparkGlm5NextKvShardRoundPlan *plan)
{
	uint64_t query_elements,partial_elements,gather_elements;
	if ( plan == 0 || rows == 0u || degree < 2u || SparkGlm5NextKvShardFits(degree,capacity) == 0u || rows > capacity )
		return(0u);
	query_elements = (uint64_t)SparkGlm5NextKvShardQuerySequences(rows,degree) * SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT;
	partial_elements = (uint64_t)SparkGlm5NextKvShardPartialSequences(rows,degree,capacity) * SparkGlm5NextKvShardPartialUnit(rows,degree,capacity);
	plan->scatter_rounds = (uint32_t)(SparkTpMeshDirectChunks(query_elements * degree,degree,SPARK_TP_MESH_OPERATION_ALL_GATHER,slot_bytes) + SparkTpMeshAllToAllChunks(partial_elements,degree,slot_bytes));
	plan->scatter_wire_bytes = (uint64_t)(degree - 1u) * (query_elements + partial_elements) * 2u;
	plan->gather_sequences = SparkGlm5NextKvShardGatherSequences(gather_keys);
	gather_elements = (uint64_t)plan->gather_sequences * SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT;
	plan->gather_rounds = gather_elements == 0u ? 0u : (uint32_t)SparkTpMeshDirectChunks(gather_elements * degree,degree,SPARK_TP_MESH_OPERATION_ALL_GATHER,slot_bytes);
	plan->gather_wire_bytes = (uint64_t)(degree - 1u) * gather_elements * 2u;
	plan->gather = plan->gather_sequences != 0u && plan->gather_sequences <= SparkGlm5NextKvShardGatherCapacity(degree,capacity) &&
		SparkKvShardExchangeCostNs(plan->gather_rounds,plan->gather_wire_bytes) < SparkKvShardExchangeCostNs(plan->scatter_rounds,plan->scatter_wire_bytes) ? 1u : 0u;
	return(1u);
}
