#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"

template<uint32_t DIM>
static __host__ __device__ constexpr uint32_t ShardBitsReferenceSharedBytes(void)
{
	return(2u * LM_PREFILL_ATTN_TILE * DIM * 2u
		+ 2u * LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE * 4u
		+ LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE * 2u
		+ 3u * LM_LATENT_SHARD_QUERIES * 4u
		+ 2u * LM_PREFILL_ATTN_TILE * 8u);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t ROWS>
__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1)
void ShardBitsReferenceKernel(
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
	uint64_t split_stride,
	LmLatentShardSelection selection)
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
	static_assert(ShardBitsReferenceSharedBytes<DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t shard_bits_reference_shared[];
	uint8_t *keys = shard_bits_reference_shared;
	float *score_parts = (float *)(keys + 2u * TILE * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE);
	float *running_max = (float *)(weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + QUERIES);
	uint32_t *list_shared = (uint32_t *)(slots + 2u * TILE);
	uint32_t *list_scan = list_shared + selection.selected_count;
	const uint32_t *list_global = 0;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t row = blockIdx.x * ROWS, group = blockIdx.y;
	const uint32_t last_row = (row + ROWS < rows ? row + ROWS : rows) - 1u;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t qk_tile = warp >> 2u, qk_half = (warp >> 1u) & 1u, qk_pair = warp & 1u;
	uint32_t query_frag[HALF_STEPS][4];
	float output[2][COLUMN_FRAGS][4];
	uint32_t sequence, dense, count, window_begin, limit, tiles, tile_low, span, tile, tile_first, tile_end, index, chunk, step, frag, q, head, query_row, m, shift, warp_count[WARP_QUERIES], listed = 0u, copy_index;
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
	if (ROWS == 1u && selection.selected_positions != 0)
	{
		dense = context_length[sequence];
		if (selection.empty_rows != 0u && row_position[row] == 0xffffffffu)
			dense = 0u;
		else if (row_position[row] < dense)
			dense = row_position[row] + 1u;
		listed = dense > selection.dense_limit ? 1u : 0u;
	}
	count = 0u;
	if (listed != 0u && selection.split_positions != 0)
	{
		const uint32_t *offsets = selection.split_offsets + (uint64_t)row * (cache.shard.degree + 1u) + cache.shard.rank;
		list_global = selection.split_positions + (uint64_t)row * selection.selected_count + offsets[0];
		count = offsets[1] - offsets[0];
	}
	else if (listed != 0u)
		count = LmLatentShardCompact(cache.shard, selection.selected_positions + (uint64_t)row * selection.selected_count, selection.selected_count,
			row_position[row], 1u, list_shared, list_scan);
	for (query_row = row; listed == 0u && query_row <= last_row; query_row++)
	{
		dense = context_length[sequence];
		if (selection.empty_rows != 0u && row_position[query_row] == 0xffffffffu)
			dense = 0u;
		else if (row_position[query_row] < dense)
			dense = row_position[query_row] + 1u;
		dense = SparkKvShardLocalKeys(cache.shard, dense);
		count = dense > count ? dense : count;
	}
	window_begin = listed != 0u ? 0u : LmKvKeyWindowBegin(cache);
	limit = listed != 0u ? count : LmKvKeyWindowEnd(cache);
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
		if (query_row <= last_row && selection.empty_rows != 0u && row_position[query_row] == 0xffffffffu)
			dense = 0u;
		else if (query_row <= last_row && row_position[query_row] < dense)
			dense = row_position[query_row] + 1u;
		warp_count[index] = query_row > last_row ? 0u : listed != 0u ? count : SparkKvShardLocalKeys(cache.shard, dense);
	}

	auto lookup = [&](uint32_t which) -> const uint8_t *
	{
		const uint32_t local = which * TILE + threadIdx.x;
		if (local >= limit || local < window_begin)
			return 0;
		return LmKvShardSlotRequired<Geometry>(cache, sequence, listed == 0u ? SparkKvShardLocalPosition(cache.shard, local) :
			list_global != 0 ? list_global[local] : list_shared[local], row, LM_KV_ACCESS_READ);
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
		head = selection.padded_heads != 0u ? (q % BLOCK_HEADS) % selection.padded_heads : group * BLOCK_HEADS + q % BLOCK_HEADS;
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
				weight_high[q * LM_PREFILL_ATTN_WEIGHT_STRIDE + lane] = LmFloatToBf16(weight[index]);
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
				if (query_row > last_row || (selection.padded_heads != 0u && q % BLOCK_HEADS >= selection.padded_heads))
					continue;
				for (copy_index = 0u; copy_index < selection.record_copies; copy_index++)
				{
					base = ((uint64_t)(head / selection.record_heads) * selection.record_copies + copy_index) * partial_rank_stride +
						((uint64_t)query_row * selection.record_heads + head % selection.record_heads) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
					*(float2 *)(partials + base + 2u + warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u) = make_float2(output[m][frag][index], output[m][frag][index + 1u]);
				}
			}
	if (threadIdx.x < QUERIES && row + threadIdx.x / BLOCK_HEADS <= last_row &&
		(selection.padded_heads == 0u || threadIdx.x % BLOCK_HEADS < selection.padded_heads))
	{
		head = group * BLOCK_HEADS + threadIdx.x % BLOCK_HEADS;
		query_row = row + threadIdx.x / BLOCK_HEADS;
		for (copy_index = 0u; copy_index < selection.record_copies; copy_index++)
		{
			base = ((uint64_t)(head / selection.record_heads) * selection.record_copies + copy_index) * partial_rank_stride +
				((uint64_t)query_row * selection.record_heads + head % selection.record_heads) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
			partials[base] = running_max[threadIdx.x];
			partials[base + 1u] = running_sum[threadIdx.x];
		}
	}
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t ROWS>
static inline cudaError_t ShardBitsReferenceRowsLaunch(
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
	cudaStream_t stream,
	LmLatentShardSelection selection)
{
	constexpr uint32_t block_heads = LM_LATENT_SHARD_QUERIES / ROWS;
	const uint64_t shared = ShardBitsReferenceSharedBytes<LATENT + ROPE>() +
		(selection.selected_positions != 0 && selection.split_positions == 0 ? LmLatentShardListBytes(selection.selected_count) : 0u);
	static bool granted = false;
	if (rows == 0u || heads_per_rank == 0u || query_bf16 == 0 || partials == 0 || row_position == 0 ||
		total_heads == 0u || (selection.padded_heads != 0u ? ROWS != 1u || selection.padded_heads != total_heads || total_heads >= block_heads :
			(total_heads % block_heads) != 0u) ||
		selection.record_heads == 0u || selection.record_copies == 0u || total_heads % selection.record_heads != 0u ||
		(selection.selected_positions != 0 && (ROWS != 1u || split_count != 1u || (selection.split_positions == 0) != (selection.split_offsets == 0))) ||
		shared > LM_PREFILL_ATTN_SHARED_LIMIT ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u ||
		query_rank_stride < (uint64_t)rows * heads_per_rank * (LATENT + ROPE) ||
		partial_rank_stride < (uint64_t)rows * selection.record_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT) ||
		(partial_rank_stride % 2u) != 0u)
		return cudaErrorInvalidValue;
	if (!granted)
	{
		cudaError_t error = cudaFuncSetAttribute((const void *)ShardBitsReferenceKernel<Geometry, Pages, LATENT, ROPE, ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)LM_PREFILL_ATTN_SHARED_LIMIT);
		if (error != cudaSuccess)
			return error;
		granted = true;
	}
	ShardBitsReferenceKernel<Geometry, Pages, LATENT, ROPE, ROWS><<<dim3((rows + ROWS - 1u) / ROWS, selection.padded_heads != 0u ? 1u : total_heads / block_heads, split_count), LM_PREFILL_ATTN_THREADS, shared, stream>>>(
		cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows,
		split_count, split_stride, selection);
	return cudaPeekAtLastError();
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t ShardBitsReferenceSplitLaunch(
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
		return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (split_partials == 0 || split_capacity_floats < row_floats)
		return cudaErrorInvalidValue;
	chunk = split_capacity_floats / row_floats < rows ? (uint32_t)(split_capacity_floats / row_floats) : rows;
	for (first = 0u; first < rows; first += count)
	{
		count = rows - first < chunk ? rows - first : chunk;
		const uint64_t split_rank_stride = (uint64_t)count * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
		const uint64_t split_stride = (uint64_t)cache.shard.degree * split_rank_stride;
		const uint64_t records = (uint64_t)first * heads_per_rank;
		error = ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16 + records * (LATENT + ROPE), query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row + first, context_length, row_position + first, qk_scale, split_partials, split_rank_stride, count, splits, split_stride, stream, LmLatentShardDense(heads_per_rank));
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
static inline cudaError_t ShardBitsReferenceDecodeLaunch(
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
	return ShardBitsReferenceSplitLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length,
		row_position, qk_scale, split_partials, split_capacity_floats, partials, partial_rank_stride, rows, splits, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t ShardBitsReferencePrefillLaunch(
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
	return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank,
		heads_per_rank * cache.shard.degree, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t ShardBitsReferenceGatherLaunch(
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
		return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 16u == 0u)
		return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 2u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 8u == 0u)
		return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 4u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 4u == 0u)
		return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 8u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 2u == 0u)
		return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 16u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 32u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE>
static __host__ __device__ constexpr bool ShardBitsReferenceTensorFits(void)
{
	return (LATENT + ROPE) % 64u == 0u && LATENT % (LM_PREFILL_ATTN_WARPS * 16u) == 0u && Geometry::kSlotBytes >= (LATENT + ROPE) * 2u &&
		ShardBitsReferenceSharedBytes<LATENT + ROPE>() <= LM_PREFILL_ATTN_SHARED_LIMIT;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t ShardBitsReferenceAttendHeadsLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	uint32_t total_heads,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t dense_limit,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t record_heads,
	uint32_t record_copies,
	uint32_t rows,
	cudaStream_t stream,
	const uint32_t *split_positions = 0,
	const uint32_t *split_offsets = 0,
	uint32_t heads_per_block_limit = 4u)
{
	if constexpr (ShardBitsReferenceTensorFits<Geometry, LATENT, ROPE>())
	{
		const uint64_t list = selected_positions != 0 && split_positions == 0 ? LmLatentShardListBytes(selected_count) : 0u;
		const uint32_t padded = total_heads < LM_LATENT_SHARD_QUERIES ? total_heads : 0u;
		if ( (total_heads % LM_LATENT_SHARD_QUERIES == 0u || padded != 0u) && row_position != 0 && (partial_rank_stride % 2u) == 0u &&
			ShardBitsReferenceSharedBytes<LATENT + ROPE>() + list <= LM_PREFILL_ATTN_SHARED_LIMIT)
		{
			const LmLatentShardSelection selection = { selected_positions, split_positions, split_offsets, selected_count, dense_limit, record_heads, record_copies, 1u, padded };
			return ShardBitsReferenceRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads,
				sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, selection);
		}
	}
	return LmLatentShardPartialHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads, sequence_of_row,
		context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, record_heads, record_copies, rows,
		stream, split_positions, split_offsets, heads_per_block_limit);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t ShardBitsReferenceAttendLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t dense_limit,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	return ShardBitsReferenceAttendHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, heads_per_rank * cache.shard.degree,
		sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, heads_per_rank, 1u,
		rows, stream);
}

#define BITS_CUDA(call) do { cudaError_t bits_error = (call); if ( bits_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(bits_error)); exit(1); } } while (0)

static int bits_failures;

__global__ void BitsFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint64_t z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
		z ^= z >> 31u;
		values[index] = LmFloatToBf16((float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 512.0f);
	}
}

static uint64_t BitsMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

template<class T>
static T *BitsDevice(const std::vector<T> &host)
{
	T *device;
	BITS_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T) + 16u));
	BITS_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static void BitsRandom(uint16_t *device,uint64_t count,uint64_t salt)
{
	BitsFillKernel<<<512,256>>>(device,count,salt);
	BITS_CUDA(cudaGetLastError());
}

struct BitsOutputs
{
	float *mine;
	float *reference;
	LmKvAccessError *mine_error;
	LmKvAccessError *reference_error;
	uint64_t floats;
};

static BitsOutputs BitsOutputsMake(uint64_t floats)
{
	BitsOutputs outputs;
	LmKvAccessError error;
	outputs.floats = floats;
	BITS_CUDA(cudaMalloc((void **)&outputs.mine,floats * sizeof(float)));
	BITS_CUDA(cudaMalloc((void **)&outputs.reference,floats * sizeof(float)));
	BITS_CUDA(cudaMalloc((void **)&outputs.mine_error,sizeof(LmKvAccessError)));
	BITS_CUDA(cudaMalloc((void **)&outputs.reference_error,sizeof(LmKvAccessError)));
	BITS_CUDA(cudaMemset(outputs.mine,0x7d,floats * sizeof(float)));
	BITS_CUDA(cudaMemset(outputs.reference,0x7d,floats * sizeof(float)));
	LmKvAccessErrorReset(&error);
	BITS_CUDA(cudaMemcpy(outputs.mine_error,&error,sizeof(error),cudaMemcpyHostToDevice));
	BITS_CUDA(cudaMemcpy(outputs.reference_error,&error,sizeof(error),cudaMemcpyHostToDevice));
	return outputs;
}

static void BitsOutputsFree(BitsOutputs *outputs)
{
	cudaFree(outputs->mine);
	cudaFree(outputs->reference);
	cudaFree(outputs->mine_error);
	cudaFree(outputs->reference_error);
}

static void BitsCheck(const char *label,const BitsOutputs &outputs,uint32_t expect_error)
{
	std::vector<uint32_t> mine(outputs.floats),reference(outputs.floats);
	LmKvAccessError mine_error,reference_error;
	uint64_t index,differ = 0u,written = 0u;
	BITS_CUDA(cudaDeviceSynchronize());
	BITS_CUDA(cudaMemcpy(mine.data(),outputs.mine,outputs.floats * sizeof(float),cudaMemcpyDeviceToHost));
	BITS_CUDA(cudaMemcpy(reference.data(),outputs.reference,outputs.floats * sizeof(float),cudaMemcpyDeviceToHost));
	BITS_CUDA(cudaMemcpy(&mine_error,outputs.mine_error,sizeof(mine_error),cudaMemcpyDeviceToHost));
	BITS_CUDA(cudaMemcpy(&reference_error,outputs.reference_error,sizeof(reference_error),cudaMemcpyDeviceToHost));
	for (index=0u; index<outputs.floats; index++)
	{
		differ += mine[index] != reference[index] ? 1u : 0u;
		written += reference[index] != 0x7d7d7d7du ? 1u : 0u;
	}
	const int passed = differ == 0u && written != 0u && mine_error.error_code == reference_error.error_code &&
		(expect_error != 0u) == (reference_error.error_code != LM_FRAME_ERROR_NONE);
	if ( !passed )
		bits_failures++;
	printf("%s %s: %llu of %llu words differ, %llu written, error codes %u and %u\n",passed ? "ok  " : "FAIL",label,(unsigned long long)differ,
		(unsigned long long)outputs.floats,(unsigned long long)written,mine_error.error_code,reference_error.error_code);
}

template<class Kv>
struct BitsPool
{
	std::vector<uint8_t *> pools;
	uint32_t *table;
	uint32_t pages;
	uint32_t sequences;
	void Make(const std::vector<uint32_t> &contexts,uint32_t degree,uint64_t salt)
	{
		uint32_t sequence,page,most = 0u;
		sequences = (uint32_t)contexts.size();
		for (sequence=0u; sequence<sequences; sequence++)
			most = contexts[sequence] > most ? contexts[sequence] : most;
		pages = (most + Kv::kPageSlots - 1u) / Kv::kPageSlots + 1u;
		std::vector<uint32_t> host((uint64_t)sequences * pages);
		for (sequence=0u; sequence<sequences; sequence++)
			for (page=0u; page<pages; page++)
				host[(uint64_t)sequence * pages + page] = (uint32_t)(((uint64_t)page * sequences + sequence) * 7u % ((uint64_t)pages * sequences));
		table = BitsDevice(host);
		pools.resize(degree);
		for (uint32_t rank=0u; rank<degree; rank++)
		{
			const uint64_t bytes = (uint64_t)pages * sequences * Kv::kPageBytes / degree;
			BITS_CUDA(cudaMalloc((void **)&pools[rank],bytes));
			BitsRandom((uint16_t *)pools[rank],bytes / 2u,salt + rank * 131u);
		}
	}
	LmKvShardView View(uint32_t degree,uint32_t rank,uint32_t grain,LmKvAccessError *error) const
	{
		LmKvShardView view;
		SparkKvShard shard = {degree,rank,grain};
		if ( LmKvShardViewInitialize<Kv>(&view,pools[rank],table,pages,sequences,pages * sequences,error,shard) != 0 )
		{
			printf("FAIL shard view\n");
			exit(1);
		}
		return view;
	}
	void Free()
	{
		for (uint8_t *pool : pools)
			cudaFree(pool);
		cudaFree(table);
	}
};

struct BitsRows
{
	std::vector<uint32_t> sequence_of_row;
	std::vector<uint32_t> row_position;
	std::vector<uint32_t> contexts;
	uint32_t *sequence_device;
	uint32_t *position_device;
	uint32_t *context_device;
	uint32_t rows;
	void Upload()
	{
		rows = (uint32_t)sequence_of_row.size();
		sequence_device = BitsDevice(sequence_of_row);
		position_device = BitsDevice(row_position);
		context_device = BitsDevice(contexts);
	}
	void Free()
	{
		cudaFree(sequence_device);
		cudaFree(position_device);
		cudaFree(context_device);
	}
};

static BitsRows BitsRowsMake(const std::vector<uint32_t> &contexts,const std::vector<uint32_t> &rows_per_sequence,uint32_t spread,uint32_t empty_every)
{
	BitsRows rows;
	uint32_t sequence,row;
	rows.contexts = contexts;
	for (sequence=0u; sequence<contexts.size(); sequence++)
		for (row=0u; row<rows_per_sequence[sequence]; row++)
		{
			uint32_t position = spread != 0u ? (uint32_t)((uint64_t)contexts[sequence] * (row + 1u) / rows_per_sequence[sequence]) - 1u
				: contexts[sequence] - rows_per_sequence[sequence] + row;
			if ( empty_every != 0u && (rows.sequence_of_row.size() + 1u) % empty_every == 0u )
				position = 0xffffffffu;
			rows.sequence_of_row.push_back(sequence);
			rows.row_position.push_back(position);
		}
	rows.Upload();
	return rows;
}

static uint32_t *BitsSelection(const BitsRows &rows,uint32_t selected_count,uint64_t salt)
{
	std::vector<uint32_t> selected((uint64_t)rows.rows * selected_count);
	for (uint32_t row=0u; row<rows.rows; row++)
	{
		const uint32_t context = rows.contexts[rows.sequence_of_row[row]];
		const uint32_t visible = rows.row_position[row] == 0xffffffffu ? context : rows.row_position[row] + 1u;
		for (uint32_t entry=0u; entry<selected_count; entry++)
		{
			const uint64_t z = BitsMix(salt ^ ((uint64_t)row << 32u) ^ entry);
			uint32_t position = entry % 7u == 3u ? visible - 1u - entry % visible : (uint32_t)(z % (visible + visible / 8u + 1u));
			if ( entry % 97u == 5u )
				position = 0xffffffffu;
			selected[(uint64_t)row * selected_count + entry] = position;
		}
	}
	return BitsDevice(selected);
}

template<uint32_t SLOT,uint32_t PAGE,uint32_t LATENT,uint32_t ROPE>
static void BitsScatter(const char *name,const std::vector<uint32_t> &contexts,const std::vector<uint32_t> &rows_per_sequence,uint32_t spread,
	uint32_t degree,uint32_t grain,uint32_t heads_per_rank,const std::vector<uint32_t> &ranks)
{
	using Kv = LmKvGeometry<SLOT,PAGE,true>;
	constexpr uint32_t DIM = LATENT + ROPE;
	BitsPool<Kv> pool;
	BitsRows rows = BitsRowsMake(contexts,rows_per_sequence,spread,0u);
	const uint64_t query_stride = (uint64_t)rows.rows * heads_per_rank * DIM,rank_stride = (uint64_t)rows.rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const float scale = 1.0f / sqrtf((float)DIM);
	uint16_t *query;
	char label[192];
	pool.Make(contexts,degree,11u);
	BITS_CUDA(cudaMalloc((void **)&query,(uint64_t)degree * query_stride * 2u));
	BitsRandom(query,(uint64_t)degree * query_stride,7u);
	for (uint32_t rank : ranks)
	{
		BitsOutputs outputs = BitsOutputsMake((uint64_t)degree * rank_stride);
		const LmKvShardView mine = pool.View(degree,rank,grain,outputs.mine_error),reference = pool.View(degree,rank,grain,outputs.reference_error);
		snprintf(label,sizeof(label),"%s dense prefill rank %u of %u grain %u heads %u rows %u",name,rank,degree,grain,heads_per_rank,rows.rows);
		BITS_CUDA((LmLatentShardPrefillLaunch<Kv,LmKvShardView,LATENT,ROPE>(mine,query,query_stride,heads_per_rank,rows.sequence_device,rows.context_device,
			rows.position_device,scale,outputs.mine,rank_stride,rows.rows,0)));
		BITS_CUDA((ShardBitsReferencePrefillLaunch<Kv,LmKvShardView,LATENT,ROPE>(reference,query,query_stride,heads_per_rank,rows.sequence_device,rows.context_device,
			rows.position_device,scale,outputs.reference,rank_stride,rows.rows,0)));
		BitsCheck(label,outputs,0u);
		BitsOutputsFree(&outputs);
	}
	cudaFree(query);
	rows.Free();
	pool.Free();
}

template<uint32_t SLOT,uint32_t PAGE,uint32_t LATENT,uint32_t ROPE>
static void BitsDecode(const char *name,uint32_t context,uint32_t rows_count,uint32_t degree,uint32_t heads_per_rank,uint32_t chunk_rows,const std::vector<uint32_t> &ranks)
{
	using Kv = LmKvGeometry<SLOT,PAGE,true>;
	constexpr uint32_t DIM = LATENT + ROPE;
	BitsPool<Kv> pool;
	BitsRows rows = BitsRowsMake(std::vector<uint32_t>(1u,context),std::vector<uint32_t>(1u,rows_count),0u,0u);
	const uint32_t heads = heads_per_rank * degree;
	const uint64_t query_stride = (uint64_t)rows.rows * heads_per_rank * DIM,rank_stride = (uint64_t)rows.rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const uint64_t split_floats = rows_count < LM_LATENT_SHARD_PREFILL_MIN_ROWS ? LmLatentShardDecodeSplitFloats(rows_count,heads,LATENT)
		: (uint64_t)chunk_rows * LmLatentShardDecodeSplits(1u) * heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const float scale = 1.0f / sqrtf((float)DIM);
	uint16_t *query;
	float *mine_split,*reference_split;
	char label[192];
	pool.Make(rows.contexts,degree,23u);
	BITS_CUDA(cudaMalloc((void **)&query,(uint64_t)degree * query_stride * 2u));
	BITS_CUDA(cudaMalloc((void **)&mine_split,split_floats * sizeof(float)));
	BITS_CUDA(cudaMalloc((void **)&reference_split,split_floats * sizeof(float)));
	BitsRandom(query,(uint64_t)degree * query_stride,9u);
	for (uint32_t rank : ranks)
	{
		BitsOutputs outputs = BitsOutputsMake((uint64_t)degree * rank_stride);
		const LmKvShardView mine = pool.View(degree,rank,1u,outputs.mine_error),reference = pool.View(degree,rank,1u,outputs.reference_error);
		if ( rows_count < LM_LATENT_SHARD_PREFILL_MIN_ROWS )
		{
			snprintf(label,sizeof(label),"%s decode rank %u rows %u splits %u context %u",name,rank,rows.rows,LmLatentShardDecodeSplits(rows_count),context);
			BITS_CUDA((LmLatentShardDecodeLaunch<Kv,LmKvShardView,LATENT,ROPE>(mine,query,query_stride,heads_per_rank,rows.sequence_device,rows.context_device,
				rows.position_device,scale,mine_split,split_floats,outputs.mine,rank_stride,rows.rows,0)));
			BITS_CUDA((ShardBitsReferenceDecodeLaunch<Kv,LmKvShardView,LATENT,ROPE>(reference,query,query_stride,heads_per_rank,rows.sequence_device,rows.context_device,
				rows.position_device,scale,reference_split,split_floats,outputs.reference,rank_stride,rows.rows,0)));
		}
		else
		{
			snprintf(label,sizeof(label),"%s one-row splits rank %u rows %u in chunks of %u context %u",name,rank,rows.rows,chunk_rows,context);
			BITS_CUDA((LmLatentShardSplitLaunch<Kv,LmKvShardView,LATENT,ROPE>(mine,query,query_stride,heads_per_rank,rows.sequence_device,rows.context_device,
				rows.position_device,scale,mine_split,split_floats,outputs.mine,rank_stride,rows.rows,LmLatentShardDecodeSplits(1u),0)));
			BITS_CUDA((ShardBitsReferenceSplitLaunch<Kv,LmKvShardView,LATENT,ROPE>(reference,query,query_stride,heads_per_rank,rows.sequence_device,rows.context_device,
				rows.position_device,scale,reference_split,split_floats,outputs.reference,rank_stride,rows.rows,LmLatentShardDecodeSplits(1u),0)));
		}
		BitsCheck(label,outputs,0u);
		BitsOutputsFree(&outputs);
	}
	cudaFree(query);
	cudaFree(mine_split);
	cudaFree(reference_split);
	rows.Free();
	pool.Free();
}

template<uint32_t SLOT,uint32_t PAGE,uint32_t LATENT,uint32_t ROPE>
static void BitsSelected(const char *name,const std::vector<uint32_t> &contexts,const std::vector<uint32_t> &rows_per_sequence,uint32_t degree,uint32_t grain,
	uint32_t heads_per_rank,uint32_t total_heads,uint32_t selected_count,uint32_t dense_limit,uint32_t record_heads,uint32_t record_copies,uint32_t split,
	const std::vector<uint32_t> &ranks)
{
	using Kv = LmKvGeometry<SLOT,PAGE,true>;
	constexpr uint32_t DIM = LATENT + ROPE;
	BitsPool<Kv> pool;
	BitsRows rows = BitsRowsMake(contexts,rows_per_sequence,1u,11u);
	const uint64_t query_stride = (uint64_t)rows.rows * heads_per_rank * DIM,rank_stride = (uint64_t)rows.rows * record_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const uint64_t floats = (uint64_t)(total_heads + record_heads - 1u) / record_heads * record_copies * rank_stride;
	const float scale = 1.0f / sqrtf((float)DIM);
	uint32_t *selected = BitsSelection(rows,selected_count,41u),*split_positions = 0,*split_offsets = 0;
	uint16_t *query;
	char label[192];
	pool.Make(contexts,degree,31u);
	BITS_CUDA(cudaMalloc((void **)&query,(uint64_t)(total_heads + heads_per_rank - 1u) / heads_per_rank * query_stride * 2u));
	BitsRandom(query,(uint64_t)(total_heads + heads_per_rank - 1u) / heads_per_rank * query_stride,13u);
	if ( split != 0u )
	{
		BITS_CUDA(cudaMalloc((void **)&split_positions,(uint64_t)rows.rows * selected_count * 4u));
		BITS_CUDA(cudaMalloc((void **)&split_offsets,(uint64_t)rows.rows * (degree + 1u) * 4u));
		SparkKvShard shard = {degree,0u,grain};
		BITS_CUDA(LmLatentShardSplitLaunch(shard,selected,selected_count,rows.position_device,split_positions,split_offsets,rows.rows,0));
	}
	for (uint32_t rank : ranks)
	{
		BitsOutputs outputs = BitsOutputsMake(floats);
		const LmKvShardView mine = pool.View(degree,rank,grain,outputs.mine_error),reference = pool.View(degree,rank,grain,outputs.reference_error);
		snprintf(label,sizeof(label),"%s selected %u limit %u rank %u of %u grain %u heads %u/%u records %ux%u%s rows %u",name,selected_count,dense_limit,rank,degree,
			grain,heads_per_rank,total_heads,record_heads,record_copies,split != 0u ? " split lists" : "",rows.rows);
		BITS_CUDA((LmLatentShardAttendHeadsLaunch<Kv,LmKvShardView,LATENT,ROPE>(mine,query,query_stride,heads_per_rank,total_heads,rows.sequence_device,
			rows.context_device,rows.position_device,selected,selected_count,dense_limit,scale,outputs.mine,rank_stride,record_heads,record_copies,rows.rows,0,
			split_positions,split_offsets)));
		BITS_CUDA((ShardBitsReferenceAttendHeadsLaunch<Kv,LmKvShardView,LATENT,ROPE>(reference,query,query_stride,heads_per_rank,total_heads,rows.sequence_device,
			rows.context_device,rows.position_device,selected,selected_count,dense_limit,scale,outputs.reference,rank_stride,record_heads,record_copies,rows.rows,0,
			split_positions,split_offsets)));
		BitsCheck(label,outputs,0u);
		BitsOutputsFree(&outputs);
	}
	cudaFree(query);
	cudaFree(selected);
	if ( split_positions != 0 )
	{
		cudaFree(split_positions);
		cudaFree(split_offsets);
	}
	rows.Free();
	pool.Free();
}

template<uint32_t SLOT,uint32_t PAGE,uint32_t LATENT,uint32_t ROPE>
static void BitsGather(const char *name,const std::vector<uint32_t> &contexts,const std::vector<uint32_t> &rows_per_sequence,uint32_t spread,uint32_t degree,
	uint32_t grain,uint32_t heads,uint32_t selected_count,uint32_t window_begin,uint32_t window_end,const std::vector<uint32_t> &sources,uint32_t expect_error)
{
	using Kv = LmKvGeometry<SLOT,PAGE,true>;
	constexpr uint32_t DIM = LATENT + ROPE;
	BitsPool<Kv> pool;
	BitsRows rows = BitsRowsMake(contexts,rows_per_sequence,spread,selected_count != 0u ? 13u : 0u);
	const SparkKvShard probe = {degree,0u,grain};
	const uint64_t query_stride = (uint64_t)rows.rows * heads * DIM,rank_stride = (uint64_t)rows.rows * heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const float scale = 1.0f / sqrtf((float)DIM);
	std::vector<uint32_t> offsets(contexts.size());
	uint32_t *offset_device,*selected = 0,*split_positions = 0,*split_offsets = 0;
	uint64_t keys = 0u;
	uint16_t *query;
	uint8_t *landing;
	char label[224];
	for (uint32_t sequence=0u; sequence<contexts.size(); sequence++)
	{
		offsets[sequence] = (uint32_t)keys;
		keys += SparkKvShardGatherKeys(probe,contexts[sequence]);
	}
	offset_device = BitsDevice(offsets);
	pool.Make(contexts,degree,37u);
	BITS_CUDA(cudaMalloc((void **)&query,query_stride * 2u));
	BITS_CUDA(cudaMalloc((void **)&landing,(uint64_t)degree * keys * Kv::kSlotBytes));
	BitsRandom(query,query_stride,17u);
	BitsRandom((uint16_t *)landing,(uint64_t)degree * keys * Kv::kSlotBytes / 2u,19u);
	if ( selected_count != 0u )
	{
		selected = BitsSelection(rows,selected_count,43u);
		BITS_CUDA(cudaMalloc((void **)&split_positions,(uint64_t)rows.rows * selected_count * 4u));
		BITS_CUDA(cudaMalloc((void **)&split_offsets,(uint64_t)rows.rows * (degree + 1u) * 4u));
		BITS_CUDA(LmLatentShardSplitLaunch(probe,selected,selected_count,rows.position_device,split_positions,split_offsets,rows.rows,0));
	}
	for (uint32_t source : sources)
		for (uint32_t list_mode=0u; list_mode<(selected_count != 0u ? 2u : 1u); list_mode++)
		{
			BitsOutputs outputs = BitsOutputsMake(rank_stride);
			const LmKvShardView pages_mine = pool.View(degree,source,grain,outputs.mine_error),pages_reference = pool.View(degree,source,grain,outputs.reference_error);
			SparkKvShard shard = {degree,source,grain};
			LmKvShardGatherView mine,reference;
			if ( LmKvShardGatherViewInitialize<Kv>(&mine,pages_mine.pages,shard,landing + (uint64_t)source * keys * Kv::kSlotBytes,offset_device,rows.context_device) != 0 ||
				LmKvShardGatherViewInitialize<Kv>(&reference,pages_reference.pages,shard,landing + (uint64_t)source * keys * Kv::kSlotBytes,offset_device,rows.context_device) != 0 ||
				(window_end != 0u && (LmKvShardGatherViewWindow(&mine,window_begin,window_end) != 0 || LmKvShardGatherViewWindow(&reference,window_begin,window_end) != 0)) )
			{
				printf("FAIL gather view\n");
				exit(1);
			}
			if ( window_end != 0u )
			{
				mine.keys += (uint64_t)window_begin * Kv::kSlotBytes;
				reference.keys += (uint64_t)window_begin * Kv::kSlotBytes;
			}
			if ( selected_count == 0u )
			{
				snprintf(label,sizeof(label),"%s gather source %u of %u grain %u heads %u rows %u window %u-%u",name,source,degree,grain,heads,rows.rows,window_begin,window_end);
				BITS_CUDA((LmLatentGatherPrefillLaunch<Kv,LmKvShardGatherView,LATENT,ROPE>(mine,query,heads,rows.sequence_device,rows.context_device,rows.position_device,
					scale,outputs.mine,rows.rows,0)));
				BITS_CUDA((ShardBitsReferenceGatherLaunch<Kv,LmKvShardGatherView,LATENT,ROPE>(reference,query,heads,rows.sequence_device,rows.context_device,
					rows.position_device,scale,outputs.reference,rows.rows,0)));
			}
			else
			{
				snprintf(label,sizeof(label),"%s landed source %u of %u grain %u heads %u selected %u%s rows %u",name,source,degree,grain,heads,selected_count,
					list_mode != 0u ? " split lists" : "",rows.rows);
				BITS_CUDA((LmLatentShardAttendHeadsLaunch<Kv,LmKvShardGatherView,LATENT,ROPE>(mine,query,query_stride,heads,heads,rows.sequence_device,
					rows.context_device,rows.position_device,selected,selected_count,selected_count,scale,outputs.mine,rank_stride,heads,1u,rows.rows,0,
					list_mode != 0u ? split_positions : 0,list_mode != 0u ? split_offsets : 0,1u)));
				BITS_CUDA((ShardBitsReferenceAttendHeadsLaunch<Kv,LmKvShardGatherView,LATENT,ROPE>(reference,query,query_stride,heads,heads,rows.sequence_device,
					rows.context_device,rows.position_device,selected,selected_count,selected_count,scale,outputs.reference,rank_stride,heads,1u,rows.rows,0,
					list_mode != 0u ? split_positions : 0,list_mode != 0u ? split_offsets : 0,1u)));
			}
			BitsCheck(label,outputs,expect_error);
			BitsOutputsFree(&outputs);
		}
	cudaFree(query);
	cudaFree(landing);
	cudaFree(offset_device);
	if ( selected != 0 )
	{
		cudaFree(selected);
		cudaFree(split_positions);
		cudaFree(split_offsets);
	}
	rows.Free();
	pool.Free();
}

int main(void)
{
	BitsScatter<1152u,64u,512u,64u>("576-wide",{5000u,777u},{200u,100u},0u,16u,1u,6u,{0u,7u,15u});
	BitsScatter<1152u,64u,512u,64u>("576-wide",{3000u},{97u},1u,16u,1u,2u,{0u,15u});
	BitsScatter<1152u,64u,512u,64u>("576-wide",{9000u,40u,12345u},{150u,40u,250u},1u,4u,4u,8u,{1u,3u});
	BitsScatter<1280u,128u,512u,0u>("512-wide",{8192u},{300u},1u,16u,8u,4u,{0u,9u});
	BitsDecode<1152u,64u,512u,64u>("576-wide",9001u,1u,16u,6u,0u,{0u,9u});
	BitsDecode<1152u,64u,512u,64u>("576-wide",40u,3u,16u,6u,0u,{4u});
	BitsDecode<1152u,64u,512u,64u>("576-wide",20000u,16u,16u,6u,0u,{0u});
	BitsDecode<1152u,64u,512u,64u>("576-wide",20000u,40u,16u,6u,7u,{0u,13u});
	BitsSelected<1152u,64u,512u,64u>("576-wide",{10000u,2500u},{90u,30u},16u,1u,6u,96u,2048u,2048u,6u,1u,0u,{0u,11u});
	BitsSelected<1152u,64u,512u,64u>("576-wide",{6000u},{64u},1u,1u,3u,3u,2048u,2048u,3u,1u,0u,{0u});
	BitsSelected<1280u,128u,512u,0u>("512-wide",{8192u},{200u},16u,8u,4u,64u,512u,512u,4u,1u,0u,{0u,13u});
	BitsSelected<1280u,128u,512u,0u>("512-wide",{8192u},{200u},16u,8u,4u,64u,512u,512u,8u,2u,0u,{5u});
	BitsSelected<1280u,128u,512u,0u>("512-wide",{8192u},{200u},16u,8u,4u,64u,512u,512u,4u,1u,1u,{2u});
	BitsSelected<1024u,64u,512u,0u>("512-wide",{12000u},{150u},16u,1u,4u,64u,2051u,2048u,4u,1u,0u,{0u,15u});
	BitsGather<1152u,64u,512u,64u>("576-wide",{4099u},{203u},0u,16u,1u,6u,0u,0u,0u,{0u,5u,15u},0u);
	BitsGather<1152u,64u,512u,64u>("576-wide",{4099u,1500u},{203u,77u},0u,16u,1u,6u,0u,0u,0u,{3u},1u);
	BitsGather<1152u,64u,512u,64u>("576-wide",{3000u},{100u},0u,16u,1u,6u,0u,96u,192u,{1u,14u},0u);
	BitsGather<1152u,64u,512u,64u>("576-wide",{2048u},{64u},1u,8u,1u,12u,0u,0u,0u,{0u,7u},0u);
	BitsGather<1152u,64u,512u,64u>("576-wide",{6000u},{150u},1u,16u,1u,4u,0u,0u,0u,{2u},0u);
	BitsGather<1152u,64u,512u,64u>("576-wide",{6000u},{150u},1u,16u,1u,1u,0u,0u,0u,{2u},0u);
	BitsGather<1152u,64u,512u,64u>("576-wide",{6000u},{150u},1u,16u,1u,16u,0u,0u,0u,{2u},0u);
	BitsGather<1152u,64u,512u,64u>("576-wide",{6000u},{150u},1u,16u,1u,32u,0u,0u,0u,{2u},0u);
	BitsGather<1280u,128u,512u,0u>("512-wide",{8192u},{300u},1u,16u,8u,4u,512u,0u,0u,{0u,6u,15u},0u);
	BitsGather<1280u,128u,512u,0u>("512-wide",{3000u,9000u},{40u,260u},1u,16u,8u,4u,512u,0u,0u,{11u},0u);
	if ( bits_failures != 0 )
	{
		printf("latent shard prefill bits: %d failure(s)\n",bits_failures);
		return 1;
	}
	printf("latent shard prefill bits: PASS\n");
	return 0;
}
