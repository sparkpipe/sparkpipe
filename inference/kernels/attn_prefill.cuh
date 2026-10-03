#pragma once

#include "inference/kernels/attn.cuh"
#include "inference/kernels/mma.cuh"
#include <math.h>
#include <stdint.h>

#define LM_PREFILL_ATTN_THREADS 256u
#define LM_PREFILL_ATTN_WARPS 8u
#define LM_PREFILL_ATTN_QUERIES 16u
#define LM_PREFILL_ATTN_TILE 32u
#define LM_PREFILL_ATTN_SCORE_STRIDE (LM_PREFILL_ATTN_TILE + 1u)
#define LM_PREFILL_ATTN_WEIGHT_STRIDE (LM_PREFILL_ATTN_TILE + 8u)
#define LM_PREFILL_ATTN_SHARED_LIMIT 101376u
#define LM_PREFILL_ATTN_COUNT_SHIFT 24u
#define LM_PREFILL_ATTN_ROW_MASK ((1u << LM_PREFILL_ATTN_COUNT_SHIFT) - 1u)
#define LM_PREFILL_ATTN_EMPTY_BLOCK 0xffffffffu

template<uint32_t DIM>
static __host__ __device__ constexpr uint32_t LmPrefillAttnSharedBytes(void)
{
	return(LM_PREFILL_ATTN_QUERIES * DIM * 2u
		+ 2u * LM_PREFILL_ATTN_TILE * DIM * 2u
		+ 2u * LM_PREFILL_ATTN_QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE * 4u
		+ 2u * LM_PREFILL_ATTN_QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE * 2u
		+ 3u * LM_PREFILL_ATTN_QUERIES * 4u
		+ 2u * LM_PREFILL_ATTN_TILE * 8u);
}

static __device__ __forceinline__ uint32_t LmPrefillSwizzle(uint32_t row, uint32_t chunk)
{
	return(chunk ^ (row & 7u));
}

static __device__ __forceinline__ void LmPrefillLoadX4(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void LmPrefillLoadX4Trans(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void LmPrefillLoadX2(uint32_t address, uint32_t r[2])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
		: "=r"(r[0]), "=r"(r[1]) : "r"(address));
}

static __device__ __forceinline__ void LmPrefillCopy(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

#define LM_PREFILL_UNION_THREADS 256u

static __device__ __forceinline__ uint32_t LmPrefillCountBelow(const uint32_t *__restrict__ list, uint32_t length, uint32_t value, uint32_t inclusive)
{
	uint32_t low = 0u,high = length,middle;
	while (low < high)
	{
		middle = (low + high) >> 1u;
		if (list[middle] < value || (inclusive != 0u && list[middle] == value))
			low = middle + 1u;
		else
			high = middle;
	}
	return(low);
}

template<uint32_t ROWS_PER_BLOCK, uint32_t SELECTED>
__global__ __launch_bounds__(LM_PREFILL_UNION_THREADS)
void LmPrefillSparseUnionKernel(
	const uint32_t *__restrict__ selected,
	const uint32_t *__restrict__ row_position,
	uint32_t rows,
	const uint32_t *__restrict__ block_table,
	uint32_t union_capacity,
	uint32_t *__restrict__ union_positions,
	uint8_t *__restrict__ union_masks,
	uint32_t *__restrict__ union_counts)
{
	constexpr uint32_t MERGED = ROWS_PER_BLOCK * SELECTED;
	constexpr uint32_t PER_THREAD = (MERGED + LM_PREFILL_UNION_THREADS - 1u) / LM_PREFILL_UNION_THREADS;
	static_assert(ROWS_PER_BLOCK <= 8u, "one membership bit per row of a block");
	__shared__ uint32_t merged[MERGED];
	__shared__ uint8_t source[MERGED];
	__shared__ uint32_t length[ROWS_PER_BLOCK];
	__shared__ uint32_t scan[LM_PREFILL_UNION_THREADS];
	const uint32_t table = blockIdx.x;
	const uint32_t entry = block_table != 0 ? block_table[table] : 0u;
	uint32_t first_row,row_end,rows_here,row,index,other,slot,total,flags[PER_THREAD],run,inclusive,base,count,mask;
	if (block_table != 0 && entry == LM_PREFILL_ATTN_EMPTY_BLOCK)
	{
		if (threadIdx.x == 0u)
			union_counts[table] = 0u;
		return;
	}
	first_row = block_table != 0 ? (entry & LM_PREFILL_ATTN_ROW_MASK) : table * ROWS_PER_BLOCK;
	row_end = block_table != 0 ? min(rows, first_row + (entry >> LM_PREFILL_ATTN_COUNT_SHIFT)) : min(rows, first_row + ROWS_PER_BLOCK);
	rows_here = row_end > first_row ? row_end - first_row : 0u;
	if (threadIdx.x < ROWS_PER_BLOCK)
		length[threadIdx.x] = threadIdx.x < rows_here
			? LmPrefillCountBelow(selected + (uint64_t)(first_row + threadIdx.x) * SELECTED, SELECTED, row_position[first_row + threadIdx.x], 1u) : 0u;
	__syncthreads();
	total = 0u;
	for (row = 0u; row < ROWS_PER_BLOCK; row++)
		total += length[row];
	for (index = threadIdx.x; index < MERGED; index += LM_PREFILL_UNION_THREADS)
	{
		row = index / SELECTED;
		other = index % SELECTED;
		if (row >= rows_here || other >= length[row])
			continue;
		const uint32_t value = selected[(uint64_t)(first_row + row) * SELECTED + other];
		slot = other;
		for (uint32_t list = 0u; list < rows_here; list++)
			if (list != row)
				slot += LmPrefillCountBelow(selected + (uint64_t)(first_row + list) * SELECTED, length[list], value, list < row ? 1u : 0u);
		merged[slot] = value;
		source[slot] = (uint8_t)row;
	}
	__syncthreads();
	run = 0u;
	for (index = 0u; index < PER_THREAD; index++)
	{
		slot = threadIdx.x * PER_THREAD + index;
		flags[index] = slot < total && (slot == 0u || merged[slot] != merged[slot - 1u]) ? 1u : 0u;
		run += flags[index];
	}
	scan[threadIdx.x] = run;
	__syncthreads();
	for (other = 1u; other < LM_PREFILL_UNION_THREADS; other <<= 1u)
	{
		inclusive = threadIdx.x >= other ? scan[threadIdx.x - other] : 0u;
		__syncthreads();
		scan[threadIdx.x] += inclusive;
		__syncthreads();
	}
	base = scan[threadIdx.x] - run;
	count = scan[LM_PREFILL_UNION_THREADS - 1u];
	for (index = 0u; index < PER_THREAD; index++)
	{
		slot = threadIdx.x * PER_THREAD + index;
		if (flags[index] == 0u)
			continue;
		mask = 0u;
		for (other = slot; other < total && merged[other] == merged[slot]; other++)
			mask |= 1u << source[other];
		if (base < union_capacity)
		{
			union_positions[(uint64_t)table * union_capacity + base] = merged[slot];
			union_masks[(uint64_t)table * union_capacity + base] = (uint8_t)mask;
		}
		base++;
	}
	if (threadIdx.x == 0u)
		union_counts[table] = count < union_capacity ? count : union_capacity;
}

template<class Geometry, uint32_t HEADS, uint32_t LATENT, uint32_t ROPE, bool SPARSE>
__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1)
void LmLatentAttentionPrefillKernel(
	const uint16_t *__restrict__ query_latent_bf16,
	const uint16_t *__restrict__ query_rope_bf16,
	LmKvView cache,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	uint16_t *__restrict__ output_bf16,
	uint32_t rows,
	const uint32_t *__restrict__ block_table,
	const uint32_t *__restrict__ union_positions,
	const uint8_t *__restrict__ union_masks,
	const uint32_t *__restrict__ union_counts,
	uint32_t union_capacity)
{
	constexpr uint32_t DIM = LATENT + ROPE;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t ROWS_PER_BLOCK = LM_PREFILL_ATTN_QUERIES / HEADS;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t TILE_CHUNKS = LM_PREFILL_ATTN_TILE * CHUNKS;
	static_assert(LM_PREFILL_ATTN_QUERIES % HEADS == 0u && HEADS <= LM_PREFILL_ATTN_QUERIES, "a block holds whole rows of heads");
	static_assert(DIM % 64u == 0u && K_STEPS % 2u == 0u, "the query-key width must split into two halves of 16-wide steps");
	static_assert(LATENT % (LM_PREFILL_ATTN_WARPS * 16u) == 0u, "each warp owns whole 16-column value slices");
	static_assert(LmPrefillAttnSharedBytes<DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_prefill_shared[];
	uint8_t *query = lm_prefill_shared;
	uint8_t *keys = query + LM_PREFILL_ATTN_QUERIES * ROW_BYTES;
	float *score_parts = (float *)(keys + 2u * LM_PREFILL_ATTN_TILE * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * LM_PREFILL_ATTN_QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE);
	uint16_t *weight_low = weight_high + LM_PREFILL_ATTN_QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE;
	float *running_max = (float *)(weight_low + LM_PREFILL_ATTN_QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + LM_PREFILL_ATTN_QUERIES;
	float *rescale = running_sum + LM_PREFILL_ATTN_QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + LM_PREFILL_ATTN_QUERIES);
	const uint32_t warp = threadIdx.x >> 5u,lane = threadIdx.x & 31u;
	const uint32_t table = gridDim.x - 1u - blockIdx.x;
	const uint32_t entry = block_table != 0 ? block_table[table] : 0u;
	const uint32_t *tile_positions = SPARSE ? union_positions + (uint64_t)table * union_capacity : 0;
	const uint8_t *tile_masks = SPARSE ? union_masks + (uint64_t)table * union_capacity : 0;
	const uint32_t first_row = block_table != 0 ? (entry & LM_PREFILL_ATTN_ROW_MASK) : (gridDim.x - 1u - blockIdx.x) * ROWS_PER_BLOCK;
	const uint32_t row_end = block_table != 0 ? min(rows, first_row + (entry >> LM_PREFILL_ATTN_COUNT_SHIFT)) : min(rows, first_row + ROWS_PER_BLOCK);
	const uint32_t query_base = (uint32_t)__cvta_generic_to_shared(query);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	float output[COLUMN_FRAGS][4];
	uint32_t sequence,last_position,tiles,tile,row,index,chunk,column,step,frag,q;

	if (block_table != 0 && entry == LM_PREFILL_ATTN_EMPTY_BLOCK)
		return;
	sequence = sequence_of_row[first_row];
	last_position = 0u;
	for (row = first_row; row < row_end; row++)
	{
		if (sequence_of_row[row] != sequence)
		{
			if (threadIdx.x == 0u)
				LmKvReportRequiredAccessFailure(cache, LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
			return;
		}
		last_position = row_position[row] > last_position ? row_position[row] : last_position;
	}
	if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache, !LmKvViewIsConfigured(cache) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, first_row, sequence, 0xffffffffu, 0xffffffffu);
		return;
	}
	tiles = SPARSE ? (union_counts[table] + LM_PREFILL_ATTN_TILE - 1u) / LM_PREFILL_ATTN_TILE : last_position / LM_PREFILL_ATTN_TILE + 1u;

	for (index = threadIdx.x; index < LM_PREFILL_ATTN_QUERIES * CHUNKS; index += LM_PREFILL_ATTN_THREADS)
	{
		q = index / CHUNKS;
		chunk = index % CHUNKS;
		row = first_row + q / HEADS;
		const uint16_t *source = row >= row_end ? query_latent_bf16
			: chunk * 8u < LATENT ? query_latent_bf16 + ((uint64_t)row * HEADS + q % HEADS) * LATENT + chunk * 8u
			: query_rope_bf16 + ((uint64_t)row * HEADS + q % HEADS) * ROPE + (chunk * 8u - LATENT);
		LmPrefillCopy(query_base + q * ROW_BYTES + LmPrefillSwizzle(q,chunk) * 16u, source, row >= row_end ? 0u : 16u);
	}
	if (threadIdx.x < LM_PREFILL_ATTN_QUERIES)
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
	for (frag = 0u; frag < COLUMN_FRAGS; frag++)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;

	auto issue = [&](uint32_t which)
	{
		uint32_t buffer = which & 1u,position,slot_index,item,piece;
		if (threadIdx.x < LM_PREFILL_ATTN_TILE)
		{
			position = which * LM_PREFILL_ATTN_TILE + threadIdx.x;
			if (SPARSE)
				position = position < union_counts[table] ? tile_positions[position] : 0xffffffffu;
			slots[buffer * LM_PREFILL_ATTN_TILE + threadIdx.x] = position <= last_position
				? LmKvSlotRequired<Geometry>(cache, sequence, position, first_row, LM_KV_ACCESS_READ) : 0;
		}
		__syncthreads();
		for (item = threadIdx.x; item < TILE_CHUNKS; item += LM_PREFILL_ATTN_THREADS)
		{
			slot_index = item / CHUNKS;
			piece = item % CHUNKS;
			const uint8_t *slot = slots[buffer * LM_PREFILL_ATTN_TILE + slot_index];
			LmPrefillCopy(key_base + (buffer * LM_PREFILL_ATTN_TILE + slot_index) * ROW_BYTES + LmPrefillSwizzle(slot_index,piece) * 16u,
				slot != 0 ? (const void *)(slot + piece * 16u) : (const void *)query_latent_bf16, slot != 0 ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	issue(0u);
	for (tile = 0u; tile < tiles; tile++)
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_base = key_base + buffer * LM_PREFILL_ATTN_TILE * ROW_BYTES;
		if (tile + 1u < tiles)
		{
			issue(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		{
			const uint32_t n_frag = warp & 3u,half = warp >> 2u;
			float score[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			uint32_t a[4],b[2];
			for (step = half * HALF_STEPS; step < (half + 1u) * HALF_STEPS; step++)
			{
				const uint32_t matrix = lane >> 3u,within = lane & 7u;
				const uint32_t a_row = (matrix & 1u) * 8u + within,a_chunk = step * 2u + (matrix >> 1u);
				const uint32_t b_row = n_frag * 8u + within,b_chunk = step * 2u + (matrix & 1u);
				LmPrefillLoadX4(query_base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row,a_chunk) * 16u, a);
				LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u, b);
				LmMmaBf16(score, a, b);
			}
			for (index = 0u; index < 4u; index++)
				score_parts[(half * LM_PREFILL_ATTN_QUERIES + (lane >> 2u) + (index >> 1u) * 8u) * LM_PREFILL_ATTN_SCORE_STRIDE + n_frag * 8u + (lane & 3u) * 2u + (index & 1u)] = score[index];
		}
		__syncthreads();
		for (q = warp * 2u; q < warp * 2u + 2u; q++)
		{
			const uint32_t listed = tile * LM_PREFILL_ATTN_TILE + lane;
			const uint32_t query_row = first_row + q / HEADS;
			const uint32_t member = !SPARSE || (listed < union_counts[table] && ((tile_masks[listed] >> (q / HEADS)) & 1u) != 0u);
			const uint32_t position = SPARSE ? (member != 0u ? tile_positions[listed] : 0xffffffffu) : listed;
			const uint32_t limit = query_row < row_end ? row_position[query_row] : 0u;
			float value = (score_parts[q * LM_PREFILL_ATTN_SCORE_STRIDE + lane] + score_parts[(LM_PREFILL_ATTN_QUERIES + q) * LM_PREFILL_ATTN_SCORE_STRIDE + lane]) * qk_scale;
			float tile_max,previous_max,next_max,weight,total,scale;
			uint16_t high;
			value = member != 0u && position <= limit ? value : -INFINITY;
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
			const uint32_t matrix = lane >> 3u,within = lane & 7u;
			const float low_scale = rescale[lane >> 2u],high_scale = rescale[(lane >> 2u) + 8u];
			uint32_t a_high[4],a_low[4],b[4];
			for (frag = 0u; frag < COLUMN_FRAGS; frag++)
			{
				output[frag][0] *= low_scale;
				output[frag][1] *= low_scale;
				output[frag][2] *= high_scale;
				output[frag][3] *= high_scale;
			}
			for (step = 0u; step < LM_PREFILL_ATTN_TILE / 16u; step++)
			{
				const uint32_t a_row = (matrix & 1u) * 8u + within,a_column = step * 16u + (matrix >> 1u) * 8u;
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_column), a_high);
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_low + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_column), a_low);
				for (frag = 0u; frag < COLUMN_FRAGS; frag += 2u)
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
					const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
					uint32_t first[2],second[2];
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
			row = first_row + q / HEADS;
			if (row >= row_end)
				continue;
			column = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
			const float inverse = 1.0f / running_sum[q];
			*(uint32_t *)(output_bf16 + ((uint64_t)row * HEADS + q % HEADS) * LATENT + column) =
				(uint32_t)LmFloatToBf16(output[frag][index] * inverse) | ((uint32_t)LmFloatToBf16(output[frag][index + 1u] * inverse) << 16u);
		}
}

template<uint32_t ROWS_PER_BLOCK>
__global__ void LmPrefillBlockTableKernel(
	LmKvView cache,
	const uint32_t *__restrict__ sequence_of_row,
	uint32_t rows,
	uint32_t *__restrict__ block_table,
	uint32_t table_blocks,
	uint32_t *__restrict__ segment_ends,
	uint32_t segment_capacity)
{
	uint32_t count = 0u,row = 0u,taken,sequence,segments = 0u;
	if (threadIdx.x != 0u || blockIdx.x != 0u)
		return;
	while (row < rows && count < table_blocks)
	{
		sequence = sequence_of_row[row];
		for (taken = 1u; row + taken < rows && taken < ROWS_PER_BLOCK && sequence_of_row[row + taken] == sequence; taken++)
		{
		}
		block_table[count++] = row | (taken << LM_PREFILL_ATTN_COUNT_SHIFT);
		row += taken;
		if (segment_ends != 0 && (row == rows || sequence_of_row[row] != sequence))
		{
			if (segments == segment_capacity)
			{
				LmKvReportRequiredAccessFailure(cache, LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row - 1u, sequence, 0xffffffffu, 0xffffffffu);
				break;
			}
			segment_ends[segments++] = row - 1u;
		}
	}
	if (row < rows)
		LmKvReportRequiredAccessFailure(cache, LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
	for (; count < table_blocks; count++)
		block_table[count] = LM_PREFILL_ATTN_EMPTY_BLOCK;
	for (; segment_ends != 0 && segments < segment_capacity; segments++)
		segment_ends[segments] = rows - 1u;
}

template<uint32_t HEADS>
static inline cudaError_t LmPrefillBlockTableLaunch(
	LmKvView cache,
	const uint32_t *sequence_of_row,
	uint32_t rows,
	uint32_t *block_table,
	uint32_t table_blocks,
	uint32_t *segment_ends,
	uint32_t segment_capacity,
	cudaStream_t stream)
{
	constexpr uint32_t ROWS_PER_BLOCK = LM_PREFILL_ATTN_QUERIES / HEADS;
	if (rows == 0u || sequence_of_row == 0 || block_table == 0 || table_blocks < (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK)
		return cudaErrorInvalidValue;
	if (segment_ends != 0 && segment_capacity == 0u)
		return cudaErrorInvalidValue;
	LmPrefillBlockTableKernel<ROWS_PER_BLOCK><<<1, 32, 0, stream>>>(cache, sequence_of_row, rows, block_table, table_blocks, segment_ends, segment_capacity);
	return cudaPeekAtLastError();
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentAttentionPrefillLaunch(
	const uint16_t *query_latent_bf16,
	const uint16_t *query_rope_bf16,
	LmKvView cache,
	const uint32_t *sequence_of_row,
	const uint32_t *row_position,
	uint32_t heads,
	float qk_scale,
	uint16_t *output_bf16,
	uint32_t rows,
	const uint32_t *block_table,
	uint32_t table_blocks,
	cudaStream_t stream)
{
	constexpr uint32_t shared = LmPrefillAttnSharedBytes<LATENT + ROPE>();
	if (rows == 0u || query_latent_bf16 == 0 || query_rope_bf16 == 0 || sequence_of_row == 0 || row_position == 0 || output_bf16 == 0 ||
		(block_table != 0 && table_blocks == 0u))
		return cudaErrorInvalidValue;
	if (heads == 4u)
	{
		static bool granted = false;
		if (!granted)
		{
			cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
			if (error != cudaSuccess)
				return error;
			granted = true;
		}
		LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, false><<<block_table != 0 ? table_blocks : (rows + LM_PREFILL_ATTN_QUERIES / 4u - 1u) / (LM_PREFILL_ATTN_QUERIES / 4u), LM_PREFILL_ATTN_THREADS, shared, stream>>>(
			query_latent_bf16, query_rope_bf16, cache, sequence_of_row, row_position, qk_scale, output_bf16, rows, block_table, 0, 0, 0, 0u);
		return cudaPeekAtLastError();
	}
	return cudaErrorInvalidValue;
}

static __host__ __device__ __forceinline__ uint64_t LmPrefillSparseUnionEntries(uint32_t table_blocks, uint32_t heads, uint32_t selected_count)
{
	return(heads == 0u || heads > LM_PREFILL_ATTN_QUERIES ? 0u : (uint64_t)table_blocks * (LM_PREFILL_ATTN_QUERIES / heads) * selected_count);
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t SELECTED>
static inline cudaError_t LmLatentAttentionSparsePrefillLaunch(
	const uint16_t *query_latent_bf16,
	const uint16_t *query_rope_bf16,
	LmKvView cache,
	const uint32_t *sequence_of_row,
	const uint32_t *row_position,
	uint32_t heads,
	float qk_scale,
	uint16_t *output_bf16,
	uint32_t rows,
	const uint32_t *block_table,
	uint32_t table_blocks,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t *union_positions,
	uint8_t *union_masks,
	uint32_t *union_counts,
	uint64_t union_entries,
	cudaStream_t stream)
{
	constexpr uint32_t shared = LmPrefillAttnSharedBytes<LATENT + ROPE>();
	const uint32_t blocks = block_table != 0 ? table_blocks : (rows + LM_PREFILL_ATTN_QUERIES / 4u - 1u) / (LM_PREFILL_ATTN_QUERIES / 4u);
	const uint32_t capacity = (LM_PREFILL_ATTN_QUERIES / 4u) * selected_count;
	if (heads != 4u || rows == 0u || query_latent_bf16 == 0 || query_rope_bf16 == 0 || sequence_of_row == 0 || row_position == 0 || output_bf16 == 0 ||
		selected_positions == 0 || selected_count == 0u || union_positions == 0 || union_masks == 0 || union_counts == 0 ||
		(block_table != 0 && table_blocks == 0u) || (uint64_t)blocks * capacity > union_entries)
		return cudaErrorInvalidValue;
	{
		static bool granted = false;
		if (!granted)
		{
			cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
			if (error != cudaSuccess)
				return error;
			granted = true;
		}
	}
	if (selected_count != SELECTED)
		return cudaErrorInvalidValue;
	LmPrefillSparseUnionKernel<LM_PREFILL_ATTN_QUERIES / 4u, SELECTED><<<blocks, LM_PREFILL_UNION_THREADS, 0, stream>>>(
		selected_positions, row_position, rows, block_table, capacity, union_positions, union_masks, union_counts);
	LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, true><<<blocks, LM_PREFILL_ATTN_THREADS, shared, stream>>>(
		query_latent_bf16, query_rope_bf16, cache, sequence_of_row, row_position, qk_scale, output_bf16, rows, block_table, union_positions, union_masks, union_counts, capacity);
	return cudaPeekAtLastError();
}
