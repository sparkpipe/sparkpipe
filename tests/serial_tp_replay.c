#include "serial_tp_replay.h"

#include <string.h>

/* bf16 is a 16-bit value; a host float is the bf16 bits in the high half. */
float spark_serial_tp_bf16_to_f32(uint16_t b)
{
	uint32_t u = ((uint32_t)b) << 16;
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

uint16_t spark_serial_tp_f32_to_bf16(float f)
{
	uint32_t u;
	memcpy(&u, &f, sizeof(u));
	/* round-to-nearest-even on the low 16 bits being dropped */
	uint32_t lsb = (u >> 16) & 1u;
	uint32_t rounded = u + 0x7fffu + lsb;
	return (uint16_t)(rounded >> 16);
}

int spark_serial_tp_sweep(
	uint32_t tp_degree,
	uint64_t partial_elements,
	const uint16_t *input_bf16,
	uint64_t input_elements,
	uint16_t *partials_bf16,
	const SparkSerialTpRankHooks *hooks,
	void *context,
	SparkSerialTpDeviceBudget *budget)
{
	uint32_t rank;
	if (tp_degree == 0u || partial_elements == 0u || partials_bf16 == 0 ||
		hooks == 0 || hooks->load_shard == 0 || hooks->free_shard == 0 ||
		hooks->shard_device_bytes == 0 || hooks->run_rank == 0 || budget == 0)
		return -1;
	if (budget->held_bytes > budget->cap_bytes)
		return -1;
	for (rank = 0u; rank < tp_degree; ++rank)
	{
		uint64_t shard_bytes = hooks->shard_device_bytes(rank, context);
		if (hooks->load_shard(rank, context) != 0)
			return -2;
		if (shard_bytes > budget->cap_bytes - budget->held_bytes)
		{
			(void)hooks->free_shard(rank, context);
			return -3; /* budget overflow: shard does not fit the cap */
		}
		budget->held_bytes += shard_bytes;
		if (budget->held_bytes > budget->peak_held_bytes)
			budget->peak_held_bytes = budget->held_bytes;
		if (hooks->run_rank(rank, input_bf16,
			partials_bf16 + ((uint64_t)rank * partial_elements),
			input_elements, partial_elements, context) != 0)
		{
			budget->held_bytes -= shard_bytes;
			(void)hooks->free_shard(rank, context);
			return -4;
		}
		(void)hooks->free_shard(rank, context);
		budget->held_bytes -= shard_bytes;
	}
	return 0;
}

void spark_serial_tp_all_reduce_sum_bf16(const uint16_t *partials, uint32_t tp,
	uint64_t n, uint16_t *out)
{
	uint64_t i;
	for (i = 0u; i < n; ++i)
	{
		float sum = 0.0f;
		uint32_t r;
		for (r = 0u; r < tp; ++r)
			sum += spark_serial_tp_bf16_to_f32(partials[((uint64_t)r * n) + i]);
		out[i] = spark_serial_tp_f32_to_bf16(sum);
	}
}

void spark_serial_tp_reduce_scatter_bf16(const uint16_t *partials, uint32_t tp,
	uint64_t n, uint32_t rank, uint16_t *out)
{
	uint64_t shard = n / (uint64_t)tp;
	uint64_t j;
	for (j = 0u; j < shard; ++j)
	{
		float sum = 0.0f;
		uint32_t r;
		for (r = 0u; r < tp; ++r)
			sum += spark_serial_tp_bf16_to_f32(
				partials[((uint64_t)r * n) + ((uint64_t)rank * shard) + j]);
		out[j] = spark_serial_tp_f32_to_bf16(sum);
	}
}

void spark_serial_tp_u64_maxloc(const uint64_t *partials, uint32_t tp,
	uint64_t n, uint64_t *out_values, uint32_t *out_ranks)
{
	uint64_t i;
	for (i = 0u; i < n; ++i)
	{
		uint64_t best = partials[i];
		uint32_t best_rank = 0u, r;
		for (r = 1u; r < tp; ++r)
		{
			uint64_t v = partials[((uint64_t)r * n) + i];
			if (v > best)
			{
				best = v;
				best_rank = r;
			}
		}
		out_values[i] = best;
		out_ranks[i] = best_rank;
	}
}

int spark_serial_tp_compare_exact(const uint16_t *output_bf16,
	uint64_t element_count, void *context)
{
	return memcmp(output_bf16, context,
		(size_t)element_count * sizeof(uint16_t)) == 0 ? 0 : -1;
}

uint64_t spark_serial_tp_hash_elements(const void *data, uint64_t element_count,
	uint32_t element_bytes)
{
	const uint8_t *bytes = (const uint8_t *)data;
	uint64_t count = element_count * (uint64_t)element_bytes;
	uint64_t hash = 1469598103934665603ull; /* FNV-1a offset basis */
	uint64_t i;
	for (i = 0u; i < count; ++i)
	{
		hash ^= bytes[i];
		hash *= 1099511628211ull; /* FNV-1a prime */
	}
	return hash;
}
