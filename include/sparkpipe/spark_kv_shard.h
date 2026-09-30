#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define SPARK_KV_SHARD_FN static inline __host__ __device__
#else
#define SPARK_KV_SHARD_FN static inline
#endif

#define SPARK_KV_SHARD_MAX_DEGREE 16u
#define SPARK_KV_SHARD_SLICE_ALIGN 16u
#define SPARK_KV_SHARD_ROUND_NS 40000u
#define SPARK_KV_SHARD_WIRE_BYTES_PER_US 25000u

typedef struct SparkKvShard
{
	uint32_t degree;
	uint32_t rank;
	uint32_t grain;
}
SparkKvShard;

typedef struct SparkKvShardExchangePlan
{
	uint64_t query_bytes_per_rank;
	uint64_t query_wire_bytes_per_rank;
	uint32_t query_rounds;
	uint64_t partial_bytes_per_peer;
	uint64_t partial_slice_bytes;
	uint32_t partial_slice_rounds;
	uint64_t partial_slice_wire_bytes_per_rank;
	uint32_t partial_gather_rounds;
	uint64_t partial_gather_wire_bytes_per_rank;
}
SparkKvShardExchangePlan;

SPARK_KV_SHARD_FN uint32_t SparkKvShardValid(SparkKvShard shard,uint32_t page_slots)
{
	uint32_t span;
	if ( shard.degree == 0u || shard.degree > SPARK_KV_SHARD_MAX_DEGREE || shard.rank >= shard.degree || shard.grain == 0u || page_slots == 0u )
		return(0u);
	if ( shard.grain > page_slots / shard.degree )
		return(0u);
	span = shard.grain * shard.degree;
	return(page_slots % span == 0u ? 1u : 0u);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardOwner(SparkKvShard shard,uint32_t position)
{
	return((position / shard.grain) % shard.degree);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardOwns(SparkKvShard shard,uint32_t position)
{
	return(SparkKvShardOwner(shard,position) == shard.rank ? 1u : 0u);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardPageSlots(SparkKvShard shard,uint32_t page_slots)
{
	return(page_slots / shard.degree);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardSlotInPage(SparkKvShard shard,uint32_t page_slots,uint32_t position)
{
	uint32_t in_page = position % page_slots;
	return((in_page / (shard.grain * shard.degree)) * shard.grain + in_page % shard.grain);
}

SPARK_KV_SHARD_FN uint64_t SparkKvShardPageBytes(SparkKvShard shard,uint32_t page_slots,uint64_t slot_bytes)
{
	return((uint64_t)SparkKvShardPageSlots(shard,page_slots) * slot_bytes);
}

SPARK_KV_SHARD_FN uint64_t SparkKvShardPoolBytes(SparkKvShard shard,uint32_t page_slots,uint64_t slot_bytes,uint64_t pages)
{
	return(pages * SparkKvShardPageBytes(shard,page_slots,slot_bytes));
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardLocalKeys(SparkKvShard shard,uint32_t keys)
{
	uint32_t span = shard.grain * shard.degree,rest,first;
	rest = keys % span;
	first = shard.rank * shard.grain;
	return((keys / span) * shard.grain + (rest > first ? (rest - first < shard.grain ? rest - first : shard.grain) : 0u));
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardLocalPosition(SparkKvShard shard,uint32_t local)
{
	return(((local / shard.grain) * shard.degree + shard.rank) * shard.grain + local % shard.grain);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardLocalIndex(SparkKvShard shard,uint32_t position)
{
	return((position / (shard.grain * shard.degree)) * shard.grain + position % shard.grain);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardGatherKeys(SparkKvShard shard,uint32_t keys)
{
	uint32_t span = shard.grain * shard.degree;
	return(span == 0u ? 0u : (uint32_t)(((uint64_t)keys + span - 1u) / span) * shard.grain);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardGatherPlan(
	SparkKvShard shard,
	const uint32_t *row_sequence,
	const uint32_t *row_position,
	uint32_t rows,
	uint32_t *list,
	uint32_t *key_offset,
	uint32_t *key_context,
	uint64_t *total_keys,
	uint32_t *most_keys)
{
	uint32_t row,other,context,keys,first,count = 0u,most = 0u;
	uint64_t total = 0u;
	for (row=0u; row<rows; row++)
	{
		first = 1u;
		for (other=0u; other<row && first != 0u; other++)
			if ( row_sequence[other] == row_sequence[row] )
				first = 0u;
		if ( first == 0u )
			continue;
		context = 0u;
		for (other=row; other<rows; other++)
			if ( row_sequence[other] == row_sequence[row] && row_position[other] + 1u > context )
				context = row_position[other] + 1u;
		keys = SparkKvShardGatherKeys(shard,context);
		if ( list != 0 && key_offset != 0 && key_context != 0 )
		{
			list[count] = row_sequence[row];
			key_offset[row_sequence[row]] = (uint32_t)total;
			key_context[row_sequence[row]] = context;
		}
		total += keys;
		most = keys > most ? keys : most;
		count++;
	}
	*total_keys = total;
	*most_keys = most;
	return(count);
}

SPARK_KV_SHARD_FN uint64_t SparkKvShardExchangeCostNs(uint64_t rounds,uint64_t wire_bytes)
{
	return(rounds * SPARK_KV_SHARD_ROUND_NS + wire_bytes * 1000u / SPARK_KV_SHARD_WIRE_BYTES_PER_US);
}

SPARK_KV_SHARD_FN uint64_t SparkKvShardCeil(uint64_t value,uint64_t unit)
{
	return(unit == 0u ? 0u : (value + unit - 1u) / unit);
}

SPARK_KV_SHARD_FN uint32_t SparkKvShardExchangePlanBuild(
	uint32_t rows,
	uint32_t heads_per_rank,
	uint32_t query_width,
	uint32_t value_width,
	uint32_t degree,
	uint64_t slot_payload_bytes,
	SparkKvShardExchangePlan *plan)
{
	uint64_t slice;
	if ( plan == 0 || rows == 0u || heads_per_rank == 0u || query_width == 0u || value_width == 0u || degree < 2u || degree > SPARK_KV_SHARD_MAX_DEGREE )
		return(0u);
	slice = (slot_payload_bytes / degree) / SPARK_KV_SHARD_SLICE_ALIGN * SPARK_KV_SHARD_SLICE_ALIGN;
	if ( slice == 0u )
		return(0u);
	plan->query_bytes_per_rank = (uint64_t)rows * heads_per_rank * query_width * 2u;
	plan->query_wire_bytes_per_rank = plan->query_bytes_per_rank * (degree - 1u);
	plan->query_rounds = (uint32_t)SparkKvShardCeil(plan->query_bytes_per_rank,slot_payload_bytes);
	plan->partial_bytes_per_peer = (uint64_t)rows * heads_per_rank * (value_width + 2u) * 4u;
	plan->partial_slice_bytes = slice;
	plan->partial_slice_rounds = (uint32_t)SparkKvShardCeil(plan->partial_bytes_per_peer,slice);
	plan->partial_slice_wire_bytes_per_rank = plan->partial_bytes_per_peer * (degree - 1u);
	plan->partial_gather_rounds = (uint32_t)SparkKvShardCeil(plan->partial_bytes_per_peer * degree,slot_payload_bytes);
	plan->partial_gather_wire_bytes_per_rank = plan->partial_bytes_per_peer * degree * (degree - 1u);
	return(1u);
}
