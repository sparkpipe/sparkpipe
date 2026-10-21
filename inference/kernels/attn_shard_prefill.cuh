#pragma once

#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/attn_shard.cuh"

#define LM_LATENT_SHARD_PREFILL_MIN_ROWS 17u

#define LM_LATENT_SHARD_QUERIES 32u

template<uint32_t DIM>
static __host__ __device__ constexpr uint32_t LmLatentShardPrefillSharedBytes(void)
{
	return(2u * LM_PREFILL_ATTN_TILE * DIM * 2u
		+ 2u * LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE * 4u
		+ 2u * LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE * 2u
		+ 3u * LM_LATENT_SHARD_QUERIES * 4u
		+ 2u * LM_PREFILL_ATTN_TILE * 8u);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t ROWS>
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
	uint64_t partial_rank_stride,
	uint32_t rows,
	uint32_t split_count,
	uint64_t split_stride)
{
	constexpr uint32_t DIM = LATENT + ROPE;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t QUERIES = LM_LATENT_SHARD_QUERIES;
	constexpr uint32_t TILE = LM_PREFILL_ATTN_TILE;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t TILE_CHUNKS = TILE * CHUNKS;
	constexpr uint32_t BLOCK_HEADS = QUERIES / ROWS;
	constexpr uint32_t WARP_QUERIES = QUERIES / LM_PREFILL_ATTN_WARPS;
	static_assert(ROWS >= 1u && QUERIES % ROWS == 0u, "a block holds whole heads of each of its rows");
	static_assert(QUERIES == 32u && LM_PREFILL_ATTN_WARPS == 8u && TILE == 32u, "two 16-query tiles, a k-half and an 8-key pair per warp, one key per lane");
	static_assert(DIM % 64u == 0u && K_STEPS % 2u == 0u, "the query-key width must split into two halves of 16-wide steps");
	static_assert(LATENT % (LM_PREFILL_ATTN_WARPS * 16u) == 0u, "each warp owns whole 16-column value slices");
	static_assert(Geometry::kSlotBytes >= ROW_BYTES, "the slot holds the latent and its rope part");
	static_assert(QUERIES * ROW_BYTES <= TILE * ROW_BYTES, "the queries stage through the second key buffer while the first tile loads");
	static_assert(LmLatentShardPrefillSharedBytes<DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_shard_prefill_shared[];
	uint8_t *keys = lm_shard_prefill_shared;
	float *score_parts = (float *)(keys + 2u * TILE * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE);
	uint16_t *weight_low = weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE;
	float *running_max = (float *)(weight_low + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + QUERIES);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t row = blockIdx.x * ROWS, group = blockIdx.y;
	const uint32_t last_row = (row + ROWS < rows ? row + ROWS : rows) - 1u;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t qk_tile = warp >> 2u, qk_half = (warp >> 1u) & 1u, qk_pair = warp & 1u;
	uint32_t query_frag[HALF_STEPS][4];
	float output[2][COLUMN_FRAGS][4];
	uint32_t sequence, dense, count, window_begin, limit, tiles, tile_low, span, tile, tile_first, tile_end, index, chunk, step, frag, q, head, query_row, m, shift, warp_count[WARP_QUERIES];
	uint64_t base;

	partials += (uint64_t)blockIdx.z * split_stride;

	sequence = sequence_of_row[row];
	for (query_row = row + 1u; query_row <= last_row && ROWS > 1u; query_row++)
		if (sequence_of_row[query_row] != sequence)
			sequence = cache.pages.sequence_count;
	if (!LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
		return;
	}
	count = 0u;
	for (query_row = row; query_row <= last_row; query_row++)
	{
		dense = context_length[sequence];
		if (row_position[query_row] < dense)
			dense = row_position[query_row] + 1u;
		dense = SparkKvShardLocalKeys(cache.shard, dense);
		count = dense > count ? dense : count;
	}
	window_begin = LmKvKeyWindowBegin(cache);
	limit = LmKvKeyWindowEnd(cache);
	limit = count < limit ? count : limit;
	tile_low = window_begin / TILE;
	tiles = limit > window_begin ? (limit + TILE - 1u) / TILE : tile_low;
	span = tiles - tile_low;
	tile_first = tile_low + (span + split_count - 1u) / split_count * blockIdx.z;
	tile_end = tile_first + (span + split_count - 1u) / split_count;
	tile_first = tile_first < tiles ? tile_first : tiles;
	tile_end = tile_end < tiles ? tile_end : tiles;
#pragma unroll
	for (index = 0u; index < WARP_QUERIES; index++)
	{
		query_row = row + (warp * WARP_QUERIES + index) / BLOCK_HEADS;
		dense = context_length[sequence];
		if (query_row <= last_row && row_position[query_row] < dense)
			dense = row_position[query_row] + 1u;
		warp_count[index] = query_row <= last_row ? SparkKvShardLocalKeys(cache.shard, dense) : 0u;
	}

	auto lookup = [&](uint32_t which) -> const uint8_t *
	{
		const uint32_t local = which * TILE + threadIdx.x;
		return local < limit && local >= window_begin ? LmKvShardSlotRequired<Geometry>(cache, sequence, SparkKvShardLocalPosition(cache.shard, local), row, LM_KV_ACCESS_READ) : 0;
	};
	auto copy = [&](uint32_t which)
	{
		uint32_t buffer = which & 1u, slot_index, item, piece;
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

	const uint32_t query_base = key_base + ((tile_first + 1u) & 1u) * TILE * ROW_BYTES;
	for (index = threadIdx.x; index < QUERIES * CHUNKS; index += LM_PREFILL_ATTN_THREADS)
	{
		q = index / CHUNKS;
		chunk = index % CHUNKS;
		head = group * BLOCK_HEADS + q % BLOCK_HEADS;
		query_row = row + q / BLOCK_HEADS;
		query_row = query_row <= last_row ? query_row : last_row;
		LmPrefillCopy(query_base + q * ROW_BYTES + LmPrefillSwizzle(q,chunk) * 16u,
			query_bf16 + (uint64_t)(head / heads_per_rank) * query_rank_stride + ((uint64_t)query_row * heads_per_rank + head % heads_per_rank) * DIM + chunk * 8u, 16u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	if (threadIdx.x < TILE && tile_first < tile_end)
	{
		slots[(tile_first & 1u) * TILE + threadIdx.x] = lookup(tile_first);
		if (tile_first + 1u < tile_end)
			slots[((tile_first + 1u) & 1u) * TILE + threadIdx.x] = lookup(tile_first + 1u);
	}
	__syncthreads();
	if (tile_first < tile_end)
	{
		copy(tile_first);
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
	}
	else
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	{
		const uint32_t a_row = qk_tile * 16u + (matrix & 1u) * 8u + within;
#pragma unroll
		for (step = 0u; step < HALF_STEPS; step++)
			LmPrefillLoadX4(query_base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row,(qk_half * HALF_STEPS + step) * 2u + (matrix >> 1u)) * 16u, query_frag[step]);
	}
	if (threadIdx.x < QUERIES)
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for (m = 0u; m < 2u; m++)
#pragma unroll
		for (frag = 0u; frag < COLUMN_FRAGS; frag++)
			output[m][frag][0] = output[m][frag][1] = output[m][frag][2] = output[m][frag][3] = 0.0f;
	__syncthreads();
	for (tile = tile_first; tile < tile_end; tile++)
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_base = key_base + buffer * TILE * ROW_BYTES;
		const uint8_t *ahead = 0;
		if (threadIdx.x < TILE && tile + 2u < tile_end)
			ahead = lookup(tile + 2u);
		if (tile + 1u < tile_end)
		{
			copy(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		{
			float score[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
			uint32_t b[4], first[2], second[2];
			const uint32_t b_row = (qk_pair * 2u + (matrix >> 1u)) * 8u + within;
#pragma unroll
			for (step = 0u; step < HALF_STEPS; step++)
			{
				const uint32_t b_chunk = (qk_half * HALF_STEPS + step) * 2u + (matrix & 1u);
				LmPrefillLoadX4(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(score[0], query_frag[step], first);
				LmMmaBf16(score[1], query_frag[step], second);
			}
#pragma unroll
			for (m = 0u; m < 2u; m++)
#pragma unroll
				for (index = 0u; index < 4u; index++)
					score_parts[(qk_half * QUERIES + qk_tile * 16u + (lane >> 2u) + (index >> 1u) * 8u) * LM_PREFILL_ATTN_SCORE_STRIDE + (qk_pair * 2u + m) * 8u + (lane & 3u) * 2u + (index & 1u)] = score[m][index];
		}
		__syncthreads();
		{
			const uint32_t local = tile * TILE + lane;
			float value[WARP_QUERIES], peak[WARP_QUERIES], previous_max[WARP_QUERIES], next_max[WARP_QUERIES], weight[WARP_QUERIES], scale[WARP_QUERIES], total[WARP_QUERIES];
			uint16_t high;
#pragma unroll
			for (index = 0u; index < WARP_QUERIES; index++)
			{
				q = warp * WARP_QUERIES + index;
				value[index] = (score_parts[q * LM_PREFILL_ATTN_SCORE_STRIDE + lane] + score_parts[(QUERIES + q) * LM_PREFILL_ATTN_SCORE_STRIDE + lane]) * qk_scale;
				value[index] = local < warp_count[index] && local < limit && local >= window_begin ? value[index] : -INFINITY;
				peak[index] = value[index];
			}
#pragma unroll
			for (shift = 16u; shift > 0u; shift >>= 1u)
#pragma unroll
				for (index = 0u; index < WARP_QUERIES; index++)
					peak[index] = fmaxf(peak[index], __shfl_xor_sync(0xffffffffu, peak[index], shift));
#pragma unroll
			for (index = 0u; index < WARP_QUERIES; index++)
			{
				previous_max[index] = running_max[warp * WARP_QUERIES + index];
				next_max[index] = fmaxf(previous_max[index], peak[index]);
				weight[index] = next_max[index] == -INFINITY || value[index] == -INFINITY ? 0.0f : __expf(value[index] - next_max[index]);
				scale[index] = previous_max[index] == -INFINITY ? 0.0f : __expf(previous_max[index] - next_max[index]);
				total[index] = weight[index];
			}
#pragma unroll
			for (shift = 16u; shift > 0u; shift >>= 1u)
#pragma unroll
				for (index = 0u; index < WARP_QUERIES; index++)
					total[index] += __shfl_xor_sync(0xffffffffu, total[index], shift);
#pragma unroll
			for (index = 0u; index < WARP_QUERIES; index++)
			{
				q = warp * WARP_QUERIES + index;
				high = LmFloatToBf16(weight[index]);
				weight_high[q * LM_PREFILL_ATTN_WEIGHT_STRIDE + lane] = high;
			}
			__syncwarp();
			if (lane == 0u)
#pragma unroll
				for (index = 0u; index < WARP_QUERIES; index++)
				{
					q = warp * WARP_QUERIES + index;
					running_sum[q] = running_sum[q] * scale[index] + total[index];
					running_max[q] = next_max[index];
					rescale[q] = scale[index];
				}
		}
		__syncthreads();
		{
			uint32_t a_high[2][4], b[4], first[2], second[2];
#pragma unroll
			for (m = 0u; m < 2u; m++)
			{
				const float low_scale = rescale[m * 16u + (lane >> 2u)], high_scale = rescale[m * 16u + (lane >> 2u) + 8u];
#pragma unroll
				for (frag = 0u; frag < COLUMN_FRAGS; frag++)
				{
					output[m][frag][0] *= low_scale;
					output[m][frag][1] *= low_scale;
					output[m][frag][2] *= high_scale;
					output[m][frag][3] *= high_scale;
				}
			}
#pragma unroll
			for (step = 0u; step < TILE / 16u; step++)
			{
#pragma unroll
				for (m = 0u; m < 2u; m++)
				{
					const uint32_t a_row = m * 16u + (matrix & 1u) * 8u + within, a_column = step * 16u + (matrix >> 1u) * 8u;
					LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_column), a_high[m]);
				}
#pragma unroll
				for (frag = 0u; frag < COLUMN_FRAGS; frag += 2u)
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
					const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
					LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
#pragma unroll
					for (m = 0u; m < 2u; m++)
					{
						LmMmaBf16(output[m][frag], a_high[m], first);
						LmMmaBf16(output[m][frag + 1u], a_high[m], second);
					}
				}
			}
		}
		if (threadIdx.x < TILE && tile + 2u < tile_end)
			slots[(tile & 1u) * TILE + threadIdx.x] = ahead;
		__syncthreads();
	}
#pragma unroll
	for (m = 0u; m < 2u; m++)
#pragma unroll
		for (frag = 0u; frag < COLUMN_FRAGS; frag++)
#pragma unroll
			for (index = 0u; index < 4u; index += 2u)
			{
				q = m * 16u + (lane >> 2u) + (index >> 1u) * 8u;
				head = group * BLOCK_HEADS + q % BLOCK_HEADS;
				query_row = row + q / BLOCK_HEADS;
				if (query_row > last_row)
					continue;
				base = (uint64_t)(head / heads_per_rank) * partial_rank_stride + ((uint64_t)query_row * heads_per_rank + head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
				*(float2 *)(partials + base + 2u + warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u) = make_float2(output[m][frag][index], output[m][frag][index + 1u]);
			}
	if (threadIdx.x < QUERIES && row + threadIdx.x / BLOCK_HEADS <= last_row)
	{
		head = group * BLOCK_HEADS + threadIdx.x % BLOCK_HEADS;
		query_row = row + threadIdx.x / BLOCK_HEADS;
		base = (uint64_t)(head / heads_per_rank) * partial_rank_stride + ((uint64_t)query_row * heads_per_rank + head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
		partials[base] = running_max[threadIdx.x];
		partials[base + 1u] = running_sum[threadIdx.x];
	}
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t ROWS>
static inline cudaError_t LmLatentShardPrefillRowsLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	uint32_t total_heads,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	uint32_t split_count,
	uint64_t split_stride,
	cudaStream_t stream)
{
	constexpr uint32_t shared = LmLatentShardPrefillSharedBytes<LATENT + ROPE>();
	constexpr uint32_t block_heads = LM_LATENT_SHARD_QUERIES / ROWS;
	static bool granted = false;
	if (rows == 0u || heads_per_rank == 0u || query_bf16 == 0 || partials == 0 || row_position == 0 ||
		total_heads == 0u || (total_heads % block_heads) != 0u ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u ||
		query_rank_stride < (uint64_t)rows * heads_per_rank * (LATENT + ROPE) ||
		partial_rank_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT) ||
		(partial_rank_stride % 2u) != 0u)
		return cudaErrorInvalidValue;
	if (!granted)
	{
		cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE, ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
		if (error != cudaSuccess)
			return error;
		granted = true;
	}
	LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE, ROWS><<<dim3((rows + ROWS - 1u) / ROWS, total_heads / block_heads, split_count), LM_PREFILL_ATTN_THREADS, shared, stream>>>(
		cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows,
		split_count, split_stride);
	return cudaPeekAtLastError();
}

template<uint32_t LATENT>
__global__ __launch_bounds__(LM_LATENT_SHARD_THREADS)
void LmLatentShardSplitMergeKernel(
	const float *__restrict__ split_partials,
	uint64_t split_rank_stride,
	uint64_t split_stride,
	uint32_t split_count,
	float *__restrict__ partials,
	uint64_t partial_rank_stride,
	uint32_t heads_per_rank)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t record = ((uint64_t)row * heads_per_rank + head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const uint64_t source = (uint64_t)(head / heads_per_rank) * split_rank_stride + record;
	const uint64_t base = (uint64_t)(head / heads_per_rank) * partial_rank_stride + record;
	float maximum = -INFINITY, sum = 0.0f, weight, value;
	uint32_t split, element;
	for (split = 0u; split < split_count; split++)
		maximum = fmaxf(maximum, split_partials[(uint64_t)split * split_stride + source]);
	for (split = 0u; split < split_count; split++)
	{
		const float local = split_partials[(uint64_t)split * split_stride + source];
		weight = maximum == -INFINITY || local == -INFINITY ? 0.0f : __expf(local - maximum);
		sum = fmaf(split_partials[(uint64_t)split * split_stride + source + 1u], weight, sum);
	}
	for (element = threadIdx.x; element < LATENT; element += LM_LATENT_SHARD_THREADS)
	{
		value = 0.0f;
		for (split = 0u; split < split_count; split++)
		{
			const float local = split_partials[(uint64_t)split * split_stride + source];
			weight = maximum == -INFINITY || local == -INFINITY ? 0.0f : __expf(local - maximum);
			value = fmaf(split_partials[(uint64_t)split * split_stride + source + 2u + element], weight, value);
		}
		partials[base + 2u + element] = value;
	}
	if (threadIdx.x == 0u)
	{
		partials[base] = maximum;
		partials[base + 1u] = sum;
	}
}

static inline uint32_t LmLatentShardDecodeSplits(uint32_t rows)
{
	return rows >= 16u ? 1u : 16u / rows;
}

static inline uint64_t LmLatentShardDecodeSplitFloats(uint32_t rows, uint32_t total_heads, uint32_t latent)
{
	return (uint64_t)LmLatentShardDecodeSplits(rows) * rows * total_heads * LM_LATENT_SHARD_RECORD_FLOATS(latent);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardSplitLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *split_partials,
	uint64_t split_capacity_floats,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	uint32_t splits,
	cudaStream_t stream)
{
	const uint32_t total_heads = heads_per_rank * cache.shard.degree;
	const uint64_t row_floats = (uint64_t)splits * total_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	uint32_t first, count, chunk;
	cudaError_t error;
	if (rows == 0u || splits == 0u || total_heads == 0u || total_heads % LM_LATENT_SHARD_QUERIES != 0u)
		return cudaErrorInvalidValue;
	if (splits == 1u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream);
	if (split_partials == 0 || split_capacity_floats < row_floats)
		return cudaErrorInvalidValue;
	chunk = split_capacity_floats / row_floats < rows ? (uint32_t)(split_capacity_floats / row_floats) : rows;
	for (first = 0u; first < rows; first += count)
	{
		count = rows - first < chunk ? rows - first : chunk;
		const uint64_t split_rank_stride = (uint64_t)count * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
		const uint64_t split_stride = (uint64_t)cache.shard.degree * split_rank_stride;
		const uint64_t records = (uint64_t)first * heads_per_rank;
		error = LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16 + records * (LATENT + ROPE), query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row + first, context_length, row_position + first, qk_scale, split_partials, split_rank_stride, count, splits, split_stride, stream);
		if (error != cudaSuccess)
			return error;
		LmLatentShardSplitMergeKernel<LATENT><<<dim3(count, total_heads), LM_LATENT_SHARD_THREADS, 0, stream>>>(split_partials, split_rank_stride, split_stride, splits,
			partials + records * LM_LATENT_SHARD_RECORD_FLOATS(LATENT), partial_rank_stride, heads_per_rank);
		error = cudaPeekAtLastError();
		if (error != cudaSuccess)
			return error;
	}
	return cudaSuccess;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardDecodeLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *split_partials,
	uint64_t split_capacity_floats,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	const uint32_t total_heads = heads_per_rank * cache.shard.degree, splits = rows != 0u ? LmLatentShardDecodeSplits(rows) : 1u;
	if (rows == 0u || rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS || total_heads % LM_LATENT_SHARD_QUERIES != 0u)
		return cudaErrorInvalidValue;
	if (splits > 1u && (split_partials == 0 || split_capacity_floats < LmLatentShardDecodeSplitFloats(rows, total_heads, LATENT)))
		return cudaErrorInvalidValue;
	return LmLatentShardSplitLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length,
		row_position, qk_scale, split_partials, split_capacity_floats, partials, partial_rank_stride, rows, splits, stream);
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
	if ((heads_per_rank * cache.shard.degree) % LM_LATENT_SHARD_QUERIES != 0u)
		return cudaErrorInvalidValue;
	return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank,
		heads_per_rank * cache.shard.degree, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentGatherPrefillLaunch(
	Pages cache,
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
	const uint64_t query_stride = (uint64_t)rows * heads_per_rank * (LATENT + ROPE);
	const uint64_t partial_stride = (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	if (heads_per_rank % 32u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream);
	if (heads_per_rank % 16u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 2u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream);
	if (heads_per_rank % 8u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 4u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream);
	if (heads_per_rank % 4u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 8u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream);
	if (heads_per_rank % 2u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 16u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream);
	return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 32u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream);
}
