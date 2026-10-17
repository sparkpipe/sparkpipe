#pragma once

#include "inference/kernels/attn_shard.cuh"

#define LM_EXPANDED_PREFILL_ROWS 32u
#define LM_EXPANDED_PREFILL_HEADS 2u
#define LM_EXPANDED_PREFILL_TILE 16u

template<uint32_t LATENT, uint32_t ROW_DIM, uint32_t VALUE>
__global__ void LmLatentValueExpandKernel(
	const uint8_t *__restrict__ keys,
	uint64_t key_source_stride,
	uint32_t keys_per_source,
	const uint16_t *__restrict__ weight_bf16,
	uint32_t heads,
	uint8_t *__restrict__ values,
	uint64_t value_source_stride)
{
	const uint32_t key = blockIdx.x, head = blockIdx.y, source = blockIdx.z;
	const uint16_t *row = (const uint16_t *)(keys + (uint64_t)source * key_source_stride + (uint64_t)key * ROW_DIM * 2u);
	uint16_t *target = (uint16_t *)(values + (uint64_t)source * value_source_stride) + ((uint64_t)key * heads + head) * VALUE;
	uint32_t output, element;
	if (key >= keys_per_source || head >= heads)
		return;
	for (output = threadIdx.x; output < VALUE; output += blockDim.x)
	{
		float total = 0.0f;
		for (element = 0u; element < LATENT; element++)
			total += LmBf16ToFloat(row[element]) * LmBf16ToFloat(weight_bf16[((uint64_t)head * VALUE + output) * LATENT + element]);
		target[output] = LmFloatToBf16(total);
	}
}

template<uint32_t LATENT, uint32_t ROW_DIM, uint32_t VALUE>
static inline cudaError_t LmLatentValueExpandLaunch(
	const uint8_t *keys,
	uint64_t key_source_stride,
	uint32_t sources,
	uint32_t keys_per_source,
	const uint16_t *weight_bf16,
	uint32_t heads,
	uint8_t *values,
	uint64_t value_source_stride,
	cudaStream_t stream)
{
	(void)stream;
	if (keys == 0 || weight_bf16 == 0 || values == 0 || sources == 0u || heads == 0u ||
		(key_source_stride % 16u) != 0u || (value_source_stride % 16u) != 0u ||
		(sources > 1u && (key_source_stride < (uint64_t)keys_per_source * ROW_DIM * 2u ||
			value_source_stride < (uint64_t)keys_per_source * heads * VALUE * 2u)))
		return cudaErrorInvalidValue;
	if (keys_per_source == 0u)
		return cudaSuccess;
	LM_HOST_LAUNCH(dim3(keys_per_source, heads, sources), (LmLatentValueExpandKernel<LATENT, ROW_DIM, VALUE>(keys, key_source_stride, keys_per_source,
		weight_bf16, heads, values, value_source_stride)));
	return cudaSuccess;
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t VALUE>
__global__ void LmExpandedGatherPrefillKernel(
	LmKvShardGatherView cache,
	const uint8_t *__restrict__ values,
	const uint16_t *__restrict__ query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	float *__restrict__ partials,
	uint32_t rows)
{
	constexpr uint32_t DIM = LATENT + ROPE;
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint16_t *query = query_bf16 + ((uint64_t)row * heads_per_rank + head) * DIM;
	float *record = partials + ((uint64_t)row * heads_per_rank + head) * LM_LATENT_SHARD_RECORD_FLOATS(VALUE);
	float maximum = -INFINITY, sum = 0.0f, output[VALUE];
	uint32_t sequence, dense, count, limit, local, element;
	if (row >= rows || head >= heads_per_rank)
		return;
	sequence = sequence_of_row[row];
	if (!LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count)
	{
		LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, 0xffffffffu, 0xffffffffu);
		return;
	}
	dense = context_length[sequence];
	if (row_position[row] < dense)
		dense = row_position[row] + 1u;
	count = SparkKvShardLocalKeys(cache.shard, dense);
	if (count > SparkKvShardLocalKeys(cache.shard, cache.key_context[sequence]))
	{
		LmKvReportRequiredAccessFailure(cache.pages, LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence,
			SparkKvShardLocalPosition(cache.shard, count - 1u), Geometry::PageOf(SparkKvShardLocalPosition(cache.shard, count - 1u)));
		return;
	}
	limit = count < cache.key_end ? count : cache.key_end;
	for (element = 0u; element < VALUE; element++)
		output[element] = 0.0f;
	for (local = cache.key_begin; local < limit; local++)
	{
		const uint8_t *slot = LmKvShardSlotRequired<Geometry>(cache, sequence, SparkKvShardLocalPosition(cache.shard, local), row, LM_KV_ACCESS_READ);
		const uint16_t *key = (const uint16_t *)slot, *value;
		float score = 0.0f, next, scale, weight;
		if (slot == 0)
			return;
		value = (const uint16_t *)(values + (uint64_t)(slot - cache.keys) / Geometry::kSlotBytes * heads_per_rank * VALUE * 2u) + (uint64_t)head * VALUE;
		for (element = 0u; element < DIM; element++)
			score += LmBf16ToFloat(query[element]) * LmBf16ToFloat(key[element]);
		score *= qk_scale;
		next = fmaxf(maximum, score);
		scale = maximum == -INFINITY ? 0.0f : __expf(maximum - next);
		weight = __expf(score - next);
		sum = sum * scale + weight;
		for (element = 0u; element < VALUE; element++)
			output[element] = output[element] * scale + weight * LmBf16ToFloat(value[element]);
		maximum = next;
	}
	record[0] = maximum;
	record[1] = sum;
	for (element = 0u; element < VALUE; element++)
		record[2u + element] = output[element];
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t VALUE>
static inline cudaError_t LmExpandedGatherPrefillLaunch(
	LmKvShardGatherView cache,
	const uint8_t *values,
	const uint16_t *query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint32_t rows,
	cudaStream_t stream)
{
	(void)stream;
	if (rows == 0u || heads_per_rank == 0u || (heads_per_rank % LM_EXPANDED_PREFILL_HEADS) != 0u || values == 0 || query_bf16 == 0 ||
		partials == 0 || row_position == 0 || cache.keys == 0 || cache.key_offset == 0 || cache.key_context == 0 ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u || (cache.key_begin % LM_EXPANDED_PREFILL_TILE) != 0u)
		return cudaErrorInvalidValue;
	LM_HOST_LAUNCH(dim3(rows, heads_per_rank, 1u), (LmExpandedGatherPrefillKernel<Geometry, LATENT, ROPE, VALUE>(cache, values, query_bf16, heads_per_rank,
		sequence_of_row, context_length, row_position, qk_scale, partials, rows)));
	return cudaSuccess;
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t VALUE>
static inline cudaError_t LmExpandedGatherPrefillSourcesLaunch(
	LmKvShardGatherView cache,
	uint32_t sources,
	uint64_t key_source_stride,
	const uint8_t *values,
	uint64_t value_source_stride,
	const uint16_t *query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint64_t partial_source_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	uint32_t source;
	cudaError_t status;
	if (sources == 0u || (uint64_t)cache.shard.rank + sources > cache.shard.degree ||
		(sources > 1u && (key_source_stride == 0u || value_source_stride == 0u ||
			partial_source_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(VALUE))))
		return cudaErrorInvalidValue;
	for (source = 0u; source < sources; source++)
	{
		LmKvShardGatherView view = cache;
		view.shard.rank = cache.shard.rank + source;
		view.keys = cache.keys + (uint64_t)source * key_source_stride;
		status = LmExpandedGatherPrefillLaunch<Geometry, LATENT, ROPE, VALUE>(view, values + (uint64_t)source * value_source_stride, query_bf16,
			heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials + (uint64_t)source * partial_source_stride, rows, stream);
		if (status != cudaSuccess)
			return status;
	}
	return cudaSuccess;
}
