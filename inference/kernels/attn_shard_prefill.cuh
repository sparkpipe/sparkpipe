#pragma once

#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/attn_shard.cuh"

#define LM_LATENT_SHARD_PREFILL_MIN_ROWS 17u

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1)
void LmLatentShardPrefillKernel(
	Pages cache,
	const uint16_t *__restrict__ query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	float *__restrict__ partials,
	uint64_t partial_rank_stride)
{
	constexpr uint32_t DIM = LATENT + ROPE;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES;
	constexpr uint32_t TILE = LM_PREFILL_ATTN_TILE;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t TILE_CHUNKS = TILE * CHUNKS;
	static_assert(DIM % 64u == 0u && K_STEPS % 2u == 0u, "the query-key width must split into two halves of 16-wide steps");
	static_assert(LATENT % (LM_PREFILL_ATTN_WARPS * 16u) == 0u, "each warp owns whole 16-column value slices");
	static_assert(Geometry::kSlotBytes >= ROW_BYTES, "the slot holds the latent and its rope part");
	static_assert(LmPrefillAttnSharedBytes<DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_shard_prefill_shared[];
	uint8_t *query = lm_shard_prefill_shared;
	uint8_t *keys = query + QUERIES * ROW_BYTES;
	float *score_parts = (float *)(keys + 2u * TILE * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE);
	uint16_t *weight_low = weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE;
	float *running_max = (float *)(weight_low + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + QUERIES);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.x, group = blockIdx.y;
	const uint32_t query_base = (uint32_t)__cvta_generic_to_shared(query);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	float output[COLUMN_FRAGS][4];
	uint32_t sequence, dense, count, tiles, tile, index, chunk, step, frag, q, head;
	uint64_t base;

	sequence = sequence_of_row[row];
	if (!LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, 0xffffffffu, 0xffffffffu);
		return;
	}
	dense = context_length[sequence];
	if (row_position[row] < dense)
		dense = row_position[row] + 1u;
	count = SparkKvShardLocalKeys(cache.shard, dense);
	tiles = (count + TILE - 1u) / TILE;

	for (index = threadIdx.x; index < QUERIES * CHUNKS; index += LM_PREFILL_ATTN_THREADS)
	{
		q = index / CHUNKS;
		chunk = index % CHUNKS;
		head = group * QUERIES + q;
		LmPrefillCopy(query_base + q * ROW_BYTES + LmPrefillSwizzle(q,chunk) * 16u,
			query_bf16 + (uint64_t)(head / heads_per_rank) * query_rank_stride + ((uint64_t)row * heads_per_rank + head % heads_per_rank) * DIM + chunk * 8u, 16u);
	}
	if (threadIdx.x < QUERIES)
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
	for (frag = 0u; frag < COLUMN_FRAGS; frag++)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;

	auto issue = [&](uint32_t which)
	{
		uint32_t buffer = which & 1u, local, slot_index, item, piece;
		if (threadIdx.x < TILE)
		{
			local = which * TILE + threadIdx.x;
			slots[buffer * TILE + threadIdx.x] = local < count
				? LmKvShardSlotRequired<Geometry>(cache, sequence, SparkKvShardLocalPosition(cache.shard, local), row, LM_KV_ACCESS_READ) : 0;
		}
		__syncthreads();
		for (item = threadIdx.x; item < TILE_CHUNKS; item += LM_PREFILL_ATTN_THREADS)
		{
			slot_index = item / CHUNKS;
			piece = item % CHUNKS;
			const uint8_t *slot = slots[buffer * TILE + slot_index];
			LmPrefillCopy(key_base + (buffer * TILE + slot_index) * ROW_BYTES + LmPrefillSwizzle(slot_index,piece) * 16u,
				slot != 0 ? (const void *)(slot + piece * 16u) : (const void *)query_bf16, slot != 0 ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	if (tiles != 0u)
		issue(0u);
	else
	{
		asm volatile("cp.async.commit_group;\n" ::: "memory");
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	}
	for (tile = 0u; tile < tiles; tile++)
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_base = key_base + buffer * TILE * ROW_BYTES;
		if (tile + 1u < tiles)
		{
			issue(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		{
			const uint32_t n_frag = warp & 3u, half = warp >> 2u;
			float score[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			uint32_t a[4], b[2];
			for (step = half * HALF_STEPS; step < (half + 1u) * HALF_STEPS; step++)
			{
				const uint32_t matrix = lane >> 3u, within = lane & 7u;
				const uint32_t a_row = (matrix & 1u) * 8u + within, a_chunk = step * 2u + (matrix >> 1u);
				const uint32_t b_row = n_frag * 8u + within, b_chunk = step * 2u + (matrix & 1u);
				LmPrefillLoadX4(query_base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row,a_chunk) * 16u, a);
				LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u, b);
				LmMmaBf16(score, a, b);
			}
			for (index = 0u; index < 4u; index++)
				score_parts[(half * QUERIES + (lane >> 2u) + (index >> 1u) * 8u) * LM_PREFILL_ATTN_SCORE_STRIDE + n_frag * 8u + (lane & 3u) * 2u + (index & 1u)] = score[index];
		}
		__syncthreads();
		for (q = warp * 2u; q < warp * 2u + 2u; q++)
		{
			const uint32_t local = tile * TILE + lane;
			float value = (score_parts[q * LM_PREFILL_ATTN_SCORE_STRIDE + lane] + score_parts[(QUERIES + q) * LM_PREFILL_ATTN_SCORE_STRIDE + lane]) * qk_scale;
			float tile_max, previous_max, next_max, weight, total, scale;
			uint16_t high;
			value = local < count ? value : -INFINITY;
			tile_max = value;
			for (index = 16u; index > 0u; index >>= 1u)
				tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, index));
			previous_max = running_max[q];
			next_max = fmaxf(previous_max, tile_max);
			weight = next_max == -INFINITY || value == -INFINITY ? 0.0f : __expf(value - next_max);
			scale = previous_max == -INFINITY ? 0.0f : __expf(previous_max - next_max);
			total = weight;
			for (index = 16u; index > 0u; index >>= 1u)
				total += __shfl_xor_sync(0xffffffffu, total, index);
			high = LmFloatToBf16(weight);
			weight_high[q * LM_PREFILL_ATTN_WEIGHT_STRIDE + lane] = high;
			weight_low[q * LM_PREFILL_ATTN_WEIGHT_STRIDE + lane] = LmFloatToBf16(weight - LmBf16ToFloat(high));
			__syncwarp();
			if (lane == 0u)
			{
				running_sum[q] = running_sum[q] * scale + total;
				running_max[q] = next_max;
				rescale[q] = scale;
			}
		}
		__syncthreads();
		{
			const uint32_t matrix = lane >> 3u, within = lane & 7u;
			const float low_scale = rescale[lane >> 2u], high_scale = rescale[(lane >> 2u) + 8u];
			uint32_t a_high[4], a_low[4], b[4];
			for (frag = 0u; frag < COLUMN_FRAGS; frag++)
			{
				output[frag][0] *= low_scale;
				output[frag][1] *= low_scale;
				output[frag][2] *= high_scale;
				output[frag][3] *= high_scale;
			}
			for (step = 0u; step < TILE / 16u; step++)
			{
				const uint32_t a_row = (matrix & 1u) * 8u + within, a_column = step * 16u + (matrix >> 1u) * 8u;
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_column), a_high);
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_low + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_column), a_low);
				for (frag = 0u; frag < COLUMN_FRAGS; frag += 2u)
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
					const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
					uint32_t first[2], second[2];
					LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(output[frag], a_high, first);
					LmMmaBf16(output[frag], a_low, first);
					LmMmaBf16(output[frag + 1u], a_high, second);
					LmMmaBf16(output[frag + 1u], a_low, second);
				}
			}
		}
		__syncthreads();
	}
	for (frag = 0u; frag < COLUMN_FRAGS; frag++)
		for (index = 0u; index < 4u; index += 2u)
		{
			q = (lane >> 2u) + (index >> 1u) * 8u;
			head = group * QUERIES + q;
			base = (uint64_t)(head / heads_per_rank) * partial_rank_stride + ((uint64_t)row * heads_per_rank + head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
			*(float2 *)(partials + base + 2u + warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u) = make_float2(output[frag][index], output[frag][index + 1u]);
		}
	if (threadIdx.x < QUERIES)
	{
		head = group * QUERIES + threadIdx.x;
		base = (uint64_t)(head / heads_per_rank) * partial_rank_stride + ((uint64_t)row * heads_per_rank + head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
		partials[base] = running_max[threadIdx.x];
		partials[base + 1u] = running_sum[threadIdx.x];
	}
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardPrefillLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	constexpr uint32_t shared = LmPrefillAttnSharedBytes<LATENT + ROPE>();
	const uint32_t total_heads = heads_per_rank * cache.shard.degree;
	static bool granted = false;
	if (rows == 0u || heads_per_rank == 0u || query_bf16 == 0 || partials == 0 || row_position == 0 ||
		(total_heads % LM_PREFILL_ATTN_QUERIES) != 0u ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u ||
		query_rank_stride < (uint64_t)rows * heads_per_rank * (LATENT + ROPE) ||
		partial_rank_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT) ||
		(partial_rank_stride % 2u) != 0u)
		return cudaErrorInvalidValue;
	if (!granted)
	{
		cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
		if (error != cudaSuccess)
			return error;
		granted = true;
	}
	LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE><<<dim3(rows, total_heads / LM_PREFILL_ATTN_QUERIES), LM_PREFILL_ATTN_THREADS, shared, stream>>>(
		cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride);
	return cudaPeekAtLastError();
}
