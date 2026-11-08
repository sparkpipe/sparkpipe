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
#define LM_PREFILL_ATTN_BLOCK_ROWS(heads) (LM_PREFILL_ATTN_QUERIES / (heads))
#define LM_PREFILL_ATTN_PARTIAL_STRIDE (LM_PREFILL_ATTN_TILE + 8u)

template<uint32_t DIM, uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES>
static __host__ __device__ constexpr uint32_t LmPrefillAttnSharedBytes(void)
{
	return(2u * LM_PREFILL_ATTN_TILE * DIM * 2u
		+ 2u * QUERIES * LM_PREFILL_ATTN_PARTIAL_STRIDE * 4u
		+ QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE * 2u
		+ 3u * QUERIES * 4u
		+ 2u * LM_PREFILL_ATTN_TILE * 8u
		+ 2u * LM_PREFILL_ATTN_TILE * 4u);
}

static __device__ __forceinline__ uint32_t LmPrefillSwizzle(uint32_t row, uint32_t chunk)
{
	return(chunk ^ (row & 7u));
}

static __device__ __forceinline__ uint32_t LmPrefillMin(uint32_t a, uint32_t b)
{
	return(a < b ? a : b);
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
#define LM_PREFILL_UNION_WINDOW 131072u

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
	constexpr uint32_t WORDS = LM_PREFILL_UNION_WINDOW / 32u;
	constexpr uint32_t WARPS = LM_PREFILL_UNION_THREADS / 32u;
	constexpr uint32_t WARP_WORDS = WORDS / WARPS;
	constexpr uint32_t MERGED = ROWS_PER_BLOCK * SELECTED;
	static_assert(ROWS_PER_BLOCK >= 1u && ROWS_PER_BLOCK <= 8u, "one membership bit per row of a block");
	static_assert(WORDS % (WARPS * 32u) == 0u, "each warp scans whole 32-word rounds of the window");
	static_assert(MERGED < 65536u, "a window's union index fits 16 bits");
	__shared__ uint32_t present[WORDS];
	__shared__ uint16_t below[WORDS];
	__shared__ uint32_t member[(MERGED + 3u) / 4u];
	__shared__ uint32_t length[ROWS_PER_BLOCK];
	__shared__ uint32_t begin[ROWS_PER_BLOCK];
	__shared__ uint32_t end[ROWS_PER_BLOCK];
	__shared__ uint32_t warp_total[WARPS];
	__shared__ uint32_t span[2];
	const uint32_t table = blockIdx.x;
	const uint32_t entry = block_table != 0 ? block_table[table] : 0u;
	const uint32_t warp = threadIdx.x >> 5u,lane = threadIdx.x & 31u;
	uint32_t first_row,row_end,rows_here,row,index,word,offset,bits,count,window_count,base,carry,inclusive,neighbour,shift;
	uint64_t window,window_end;
	if (block_table != 0 && entry == LM_PREFILL_ATTN_EMPTY_BLOCK)
	{
		if (threadIdx.x == 0u)
			union_counts[table] = 0u;
		return;
	}
	first_row = block_table != 0 ? (entry & LM_PREFILL_ATTN_ROW_MASK) : table * ROWS_PER_BLOCK;
	row_end = block_table != 0 ? LmPrefillMin(rows, first_row + (entry >> LM_PREFILL_ATTN_COUNT_SHIFT)) : LmPrefillMin(rows, first_row + ROWS_PER_BLOCK);
	rows_here = row_end > first_row ? row_end - first_row : 0u;
	if (threadIdx.x < ROWS_PER_BLOCK)
	{
		length[threadIdx.x] = threadIdx.x < rows_here
			? LmPrefillCountBelow(selected + (uint64_t)(first_row + threadIdx.x) * SELECTED, SELECTED, row_position[first_row + threadIdx.x], 1u) : 0u;
		end[threadIdx.x] = 0u;
	}
	__syncthreads();
	if (threadIdx.x == 0u)
	{
		span[0] = 0xffffffffu;
		span[1] = 0u;
		for (row = 0u; row < ROWS_PER_BLOCK; row++)
			if (length[row] != 0u)
			{
				span[0] = LmPrefillMin(span[0], selected[(uint64_t)(first_row + row) * SELECTED]);
				span[1] = selected[(uint64_t)(first_row + row) * SELECTED + length[row] - 1u] > span[1] ? selected[(uint64_t)(first_row + row) * SELECTED + length[row] - 1u] : span[1];
			}
	}
	__syncthreads();
	count = 0u;
	for (window = span[0] - span[0] % LM_PREFILL_UNION_WINDOW; span[0] <= span[1] && window <= span[1]; window += LM_PREFILL_UNION_WINDOW)
	{
		window_end = window + LM_PREFILL_UNION_WINDOW;
		if (threadIdx.x < ROWS_PER_BLOCK)
		{
			const uint32_t *list = selected + (uint64_t)(first_row + threadIdx.x) * SELECTED;
			begin[threadIdx.x] = end[threadIdx.x];
			end[threadIdx.x] = length[threadIdx.x] == 0u || list[length[threadIdx.x] - 1u] < window_end ? length[threadIdx.x]
				: LmPrefillCountBelow(list, length[threadIdx.x], (uint32_t)window_end, 0u);
		}
		for (word = threadIdx.x; word < WORDS; word += LM_PREFILL_UNION_THREADS)
			present[word] = 0u;
		for (word = threadIdx.x; word < (MERGED + 3u) / 4u; word += LM_PREFILL_UNION_THREADS)
			member[word] = 0u;
		__syncthreads();
		for (row = 0u; row < ROWS_PER_BLOCK; row++)
			for (index = begin[row] + threadIdx.x; index < end[row]; index += LM_PREFILL_UNION_THREADS)
			{
				offset = selected[(uint64_t)(first_row + row) * SELECTED + index] - (uint32_t)window;
				atomicOr(&present[offset >> 5u], 1u << (offset & 31u));
			}
		__syncthreads();
		carry = 0u;
		for (index = 0u; index < WARP_WORDS / 32u; index++)
		{
			word = warp * WARP_WORDS + index * 32u + lane;
			bits = __popc(present[word]);
			inclusive = bits;
			for (shift = 1u; shift < 32u; shift <<= 1u)
			{
				neighbour = __shfl_up_sync(0xffffffffu, inclusive, shift);
				inclusive += lane >= shift ? neighbour : 0u;
			}
			below[word] = (uint16_t)(carry + inclusive - bits);
			carry += __shfl_sync(0xffffffffu, inclusive, 31u);
		}
		if (lane == 0u)
			warp_total[warp] = carry;
		__syncthreads();
		base = 0u;
		window_count = 0u;
		for (index = 0u; index < WARPS; index++)
		{
			base += index < warp ? warp_total[index] : 0u;
			window_count += warp_total[index];
		}
		for (index = 0u; index < WARP_WORDS / 32u; index++)
			below[warp * WARP_WORDS + index * 32u + lane] += (uint16_t)base;
		__syncthreads();
		for (row = 0u; row < ROWS_PER_BLOCK; row++)
			for (index = begin[row] + threadIdx.x; index < end[row]; index += LM_PREFILL_UNION_THREADS)
			{
				offset = selected[(uint64_t)(first_row + row) * SELECTED + index] - (uint32_t)window;
				word = offset >> 5u;
				bits = below[word] + __popc(present[word] & ((1u << (offset & 31u)) - 1u));
				atomicOr(&member[bits >> 2u], (1u << row) << ((bits & 3u) * 8u));
			}
		__syncthreads();
		for (index = 0u; index < WARP_WORDS / 32u; index++)
		{
			word = warp * WARP_WORDS + index * 32u + lane;
			bits = present[word];
			offset = below[word];
			while (bits != 0u)
			{
				const uint32_t bit = __ffs(bits) - 1u;
				bits &= bits - 1u;
				if (count + offset < union_capacity)
				{
					union_positions[(uint64_t)table * union_capacity + count + offset] = (uint32_t)window + word * 32u + bit;
					union_masks[(uint64_t)table * union_capacity + count + offset] = (uint8_t)(member[offset >> 2u] >> ((offset & 3u) * 8u));
				}
				offset++;
			}
		}
		count += window_count;
		__syncthreads();
	}
	if (threadIdx.x == 0u)
		union_counts[table] = count < union_capacity ? count : union_capacity;
}

template<class Geometry, uint32_t HEADS, uint32_t LATENT, uint32_t ROPE, bool SPARSE, uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES>
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
	constexpr uint32_t TILE = LM_PREFILL_ATTN_TILE;
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / HEADS;
	constexpr uint32_t QUERY_TILES = QUERIES / 16u;
	constexpr uint32_t KEY_FRAGS = QUERY_TILES;
	constexpr uint32_t KEY_GROUPS = TILE / 8u / KEY_FRAGS;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / LM_PREFILL_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t TILE_CHUNKS = TILE * CHUNKS;
	static_assert((QUERIES == 16u || QUERIES == 32u) && LM_PREFILL_ATTN_WARPS == 8u && TILE == 32u, "one or two 16-query tiles, a k-half and one or two 8-key fragments per warp");
	static_assert(QUERIES % HEADS == 0u && HEADS <= QUERIES && ROWS_PER_BLOCK <= 32u, "a block holds whole rows of heads, one row bit each");
	static_assert(DIM % 64u == 0u && K_STEPS % 2u == 0u, "the query-key width must split into two halves of 16-wide steps");
	static_assert(LATENT % (LM_PREFILL_ATTN_WARPS * 16u) == 0u, "each warp owns whole 16-column value slices");
	static_assert(LmPrefillAttnSharedBytes<DIM, QUERIES>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_prefill_shared[];
	uint8_t *keys = lm_prefill_shared;
	float *score_parts = (float *)(keys + 2u * TILE * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_PARTIAL_STRIDE);
	float *running_max = (float *)(weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + QUERIES);
	uint32_t *key_rows = (uint32_t *)(slots + 2u * TILE);
	const uint32_t warp = threadIdx.x >> 5u,lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u,within = lane & 7u;
	const uint32_t query_tile = warp / (2u * KEY_GROUPS),half = (warp / KEY_GROUPS) & 1u,key_group = warp % KEY_GROUPS;
	const uint32_t table = gridDim.x - 1u - blockIdx.x;
	const uint32_t entry = block_table != 0 ? block_table[table] : 0u;
	const uint32_t *tile_positions = SPARSE ? union_positions + (uint64_t)table * union_capacity : 0;
	const uint8_t *tile_masks = SPARSE ? union_masks + (uint64_t)table * union_capacity : 0;
	const uint32_t first_row = block_table != 0 ? (entry & LM_PREFILL_ATTN_ROW_MASK) : table * ROWS_PER_BLOCK;
	const uint32_t row_end = block_table != 0 ? LmPrefillMin(rows, first_row + (entry >> LM_PREFILL_ATTN_COUNT_SHIFT)) : LmPrefillMin(rows, first_row + ROWS_PER_BLOCK);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t query_base = key_base + TILE * ROW_BYTES;
	uint32_t query_frag[HALF_STEPS][4];
	float output[QUERY_TILES][COLUMN_FRAGS][4];
	uint32_t row_limit[ROWS_PER_BLOCK];
	uint32_t sequence,last_position,listed_keys,tiles,tile,row,index,chunk,column,step,frag,q,m,shift;
	uint32_t next_position = 0xffffffffu,next_members = 0u;

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
	listed_keys = SPARSE ? union_counts[table] : last_position + 1u;
	tiles = (listed_keys + TILE - 1u) / TILE;
#pragma unroll
	for (row = 0u; row < ROWS_PER_BLOCK; row++)
		row_limit[row] = first_row + row < row_end ? row_position[first_row + row] : 0u;

	auto listing = [&](uint32_t which, uint32_t &position, uint32_t &members)
	{
		const uint32_t listed = which * TILE + threadIdx.x;
		position = !SPARSE ? listed : listed < listed_keys ? tile_positions[listed] : 0xffffffffu;
		members = !SPARSE ? 0xffffffffu : listed < listed_keys ? (uint32_t)tile_masks[listed] : 0u;
	};
	auto prefetch = [&](uint32_t position)
	{
		const uint32_t page = Geometry::PageOf(position);
		if (position <= last_position && page < cache.page_table_stride)
			asm volatile("prefetch.global.L1 [%0];\n" :: "l"(cache.page_table + (uint64_t)sequence * cache.page_table_stride + page));
	};
	auto resolve = [&](uint32_t position, uint32_t members, uint32_t &valid) -> const uint8_t *
	{
		uint32_t r;
		valid = 0u;
#pragma unroll
		for (r = 0u; r < ROWS_PER_BLOCK; r++)
			valid |= (first_row + r < row_end && ((members >> r) & 1u) != 0u && position <= row_limit[r] ? 1u : 0u) << r;
		return position <= last_position ? LmKvSlotRequired<Geometry>(cache, sequence, position, first_row, LM_KV_ACCESS_READ) : 0;
	};
	auto copy = [&](uint32_t which)
	{
		const uint32_t buffer = which & 1u;
		uint32_t item,slot_index,piece;
		for (item = threadIdx.x; item < TILE_CHUNKS; item += LM_PREFILL_ATTN_THREADS)
		{
			slot_index = item / CHUNKS;
			piece = item % CHUNKS;
			const uint8_t *slot = slots[buffer * TILE + slot_index];
			LmPrefillCopy(key_base + (buffer * TILE + slot_index) * ROW_BYTES + LmPrefillSwizzle(slot_index,piece) * 16u,
				slot != 0 ? (const void *)(slot + piece * 16u) : (const void *)query_latent_bf16, slot != 0 ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	for (index = threadIdx.x; index < QUERIES * CHUNKS; index += LM_PREFILL_ATTN_THREADS)
	{
		q = index / CHUNKS;
		chunk = index % CHUNKS;
		row = first_row + q / HEADS;
		const uint16_t *source = row >= row_end ? query_latent_bf16
			: chunk * 8u < LATENT ? query_latent_bf16 + ((uint64_t)row * HEADS + q % HEADS) * LATENT + chunk * 8u
			: query_rope_bf16 + ((uint64_t)row * HEADS + q % HEADS) * ROPE + (chunk * 8u - LATENT);
		LmPrefillCopy(query_base + q * ROW_BYTES + LmPrefillSwizzle(q,chunk) * 16u, source, row >= row_end ? 0u : 16u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	if (threadIdx.x < TILE)
	{
		uint32_t valid,first_position,first_members,second_position = 0xffffffffu,second_members = 0u;
		if (tiles > 0u)
			listing(0u, first_position, first_members);
		if (tiles > 1u)
			listing(1u, second_position, second_members);
		if (tiles > 2u)
			listing(2u, next_position, next_members);
		if (tiles > 0u)
		{
			slots[threadIdx.x] = resolve(first_position, first_members, valid);
			key_rows[threadIdx.x] = valid;
		}
		if (tiles > 1u)
		{
			slots[TILE + threadIdx.x] = resolve(second_position, second_members, valid);
			key_rows[TILE + threadIdx.x] = valid;
		}
	}
	__syncthreads();
	if (tiles > 0u)
	{
		copy(0u);
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
	}
	else
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	{
		const uint32_t a_row = query_tile * 16u + (matrix & 1u) * 8u + within;
#pragma unroll
		for (step = 0u; step < HALF_STEPS; step++)
			LmPrefillLoadX4(query_base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row,(half * HALF_STEPS + step) * 2u + (matrix >> 1u)) * 16u, query_frag[step]);
	}
	if (threadIdx.x < QUERIES)
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for (m = 0u; m < QUERY_TILES; m++)
#pragma unroll
		for (frag = 0u; frag < COLUMN_FRAGS; frag++)
			output[m][frag][0] = output[m][frag][1] = output[m][frag][2] = output[m][frag][3] = 0.0f;
	__syncthreads();

	for (tile = 0u; tile < tiles; tile++)
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_base = key_base + buffer * TILE * ROW_BYTES;
		uint32_t ahead_position = 0xffffffffu,ahead_members = 0u;
		if (threadIdx.x < TILE && tile + 2u < tiles)
			prefetch(next_position);
		if (threadIdx.x < TILE && tile + 3u < tiles)
			listing(tile + 3u, ahead_position, ahead_members);
		if (tile + 1u < tiles)
		{
			copy(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		{
			float score[KEY_FRAGS][4];
			uint32_t b[4];
			const uint32_t b_row = (key_group * KEY_FRAGS + (KEY_FRAGS == 2u ? matrix >> 1u : 0u)) * 8u + within;
#pragma unroll
			for (frag = 0u; frag < KEY_FRAGS; frag++)
				score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
#pragma unroll
			for (step = 0u; step < HALF_STEPS; step++)
			{
				const uint32_t b_chunk = (half * HALF_STEPS + step) * 2u + (matrix & 1u);
				if constexpr (KEY_FRAGS == 2u)
				{
					uint32_t first[2],second[2];
					LmPrefillLoadX4(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(score[0], query_frag[step], first);
					LmMmaBf16(score[1], query_frag[step], second);
				}
				else
				{
					LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u, b);
					LmMmaBf16(score[0], query_frag[step], b);
				}
			}
#pragma unroll
			for (frag = 0u; frag < KEY_FRAGS; frag++)
#pragma unroll
				for (index = 0u; index < 4u; index++)
					score_parts[(half * QUERIES + query_tile * 16u + (lane >> 2u) + (index >> 1u) * 8u) * LM_PREFILL_ATTN_PARTIAL_STRIDE
						+ (key_group * KEY_FRAGS + frag) * 8u + (lane & 3u) * 2u + (index & 1u)] = score[frag][index];
		}
		__syncthreads();
		{
			const uint32_t own = threadIdx.x / QUERY_THREADS,part = threadIdx.x % QUERY_THREADS;
			float value[QUERY_KEYS],weight[QUERY_KEYS],peak,previous_max,next_max,scale,total;
#pragma unroll
			for (index = 0u; index < QUERY_KEYS; index++)
			{
				const uint32_t key = part + index * QUERY_THREADS;
				value[index] = (score_parts[own * LM_PREFILL_ATTN_PARTIAL_STRIDE + key] + score_parts[(QUERIES + own) * LM_PREFILL_ATTN_PARTIAL_STRIDE + key]) * qk_scale;
				value[index] = ((key_rows[buffer * TILE + key] >> (own / HEADS)) & 1u) != 0u ? value[index] : -INFINITY;
			}
			peak = value[0];
#pragma unroll
			for (index = 1u; index < QUERY_KEYS; index++)
				peak = fmaxf(peak, value[index]);
#pragma unroll
			for (shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u)
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
			previous_max = running_max[own];
			next_max = fmaxf(previous_max, peak);
			scale = previous_max == -INFINITY ? 0.0f : __expf(previous_max - next_max);
#pragma unroll
			for (index = 0u; index < QUERY_KEYS; index++)
			{
				weight[index] = next_max == -INFINITY || value[index] == -INFINITY ? 0.0f : __expf(value[index] - next_max);
				weight_high[own * LM_PREFILL_ATTN_WEIGHT_STRIDE + part + index * QUERY_THREADS] = LmFloatToBf16(weight[index]);
			}
#pragma unroll
			for (step = QUERY_KEYS / 2u; step > 0u; step >>= 1u)
#pragma unroll
				for (index = 0u; index < step; index++)
					weight[index] += weight[index + step];
			total = weight[0];
#pragma unroll
			for (shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u)
				total += __shfl_xor_sync(0xffffffffu, total, shift);
			__syncwarp();
			if (part == 0u)
			{
				running_sum[own] = running_sum[own] * scale + total;
				running_max[own] = next_max;
				rescale[own] = scale;
			}
		}
		__syncthreads();
		{
			uint32_t a_high[QUERY_TILES][4],b[4],first[2],second[2];
#pragma unroll
			for (m = 0u; m < QUERY_TILES; m++)
			{
				const float low_scale = rescale[m * 16u + (lane >> 2u)],high_scale = rescale[m * 16u + (lane >> 2u) + 8u];
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
				for (m = 0u; m < QUERY_TILES; m++)
				{
					const uint32_t a_row = m * 16u + (matrix & 1u) * 8u + within,a_column = step * 16u + (matrix >> 1u) * 8u;
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
					for (m = 0u; m < QUERY_TILES; m++)
					{
						LmMmaBf16(output[m][frag], a_high[m], first);
						LmMmaBf16(output[m][frag + 1u], a_high[m], second);
					}
				}
			}
		}
		if (threadIdx.x < TILE && tile + 2u < tiles)
		{
			uint32_t valid;
			slots[buffer * TILE + threadIdx.x] = resolve(next_position, next_members, valid);
			key_rows[buffer * TILE + threadIdx.x] = valid;
		}
		next_position = ahead_position;
		next_members = ahead_members;
		__syncthreads();
	}
#pragma unroll
	for (m = 0u; m < QUERY_TILES; m++)
#pragma unroll
		for (frag = 0u; frag < COLUMN_FRAGS; frag++)
#pragma unroll
			for (index = 0u; index < 4u; index += 2u)
			{
				q = m * 16u + (lane >> 2u) + (index >> 1u) * 8u;
				row = first_row + q / HEADS;
				if (row >= row_end)
					continue;
				column = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
				const float inverse = 1.0f / running_sum[q];
				*(uint32_t *)(output_bf16 + ((uint64_t)row * HEADS + q % HEADS) * LATENT + column) =
					(uint32_t)LmFloatToBf16(output[m][frag][index] * inverse) | ((uint32_t)LmFloatToBf16(output[m][frag][index + 1u] * inverse) << 16u);
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

template<uint32_t HEADS, uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES>
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
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / HEADS;
	static_assert(QUERIES % HEADS == 0u && ROWS_PER_BLOCK < (1u << (32u - LM_PREFILL_ATTN_COUNT_SHIFT)), "a block's row count fits its table entry");
	if (rows == 0u || sequence_of_row == 0 || block_table == 0 || table_blocks < (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK)
		return cudaErrorInvalidValue;
	if (segment_ends != 0 && segment_capacity == 0u)
		return cudaErrorInvalidValue;
	LmPrefillBlockTableKernel<ROWS_PER_BLOCK><<<1, 32, 0, stream>>>(cache, sequence_of_row, rows, block_table, table_blocks, segment_ends, segment_capacity);
	return cudaPeekAtLastError();
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES>
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
	constexpr uint32_t shared = LmPrefillAttnSharedBytes<LATENT + ROPE, QUERIES>();
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / 4u;
	if (rows == 0u || query_latent_bf16 == 0 || query_rope_bf16 == 0 || sequence_of_row == 0 || row_position == 0 || output_bf16 == 0 ||
		(block_table != 0 && table_blocks == 0u))
		return cudaErrorInvalidValue;
	if (heads == 4u)
	{
		static bool granted = false;
		if (!granted)
		{
			cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, false, QUERIES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
			if (error != cudaSuccess)
				return error;
			granted = true;
		}
		LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, false, QUERIES><<<block_table != 0 ? table_blocks : (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK, LM_PREFILL_ATTN_THREADS, shared, stream>>>(
			query_latent_bf16, query_rope_bf16, cache, sequence_of_row, row_position, qk_scale, output_bf16, rows, block_table, 0, 0, 0, 0u);
		return cudaPeekAtLastError();
	}
	return cudaErrorInvalidValue;
}

template<uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES>
static __host__ __device__ __forceinline__ uint64_t LmPrefillSparseUnionEntries(uint32_t table_blocks, uint32_t heads, uint32_t selected_count)
{
	return(heads == 0u || heads > QUERIES ? 0u : (uint64_t)table_blocks * (QUERIES / heads) * selected_count);
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t SELECTED, uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES>
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
	constexpr uint32_t shared = LmPrefillAttnSharedBytes<LATENT + ROPE, QUERIES>();
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / 4u;
	const uint32_t blocks = block_table != 0 ? table_blocks : (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK;
	const uint32_t capacity = ROWS_PER_BLOCK * selected_count;
	if (heads != 4u || rows == 0u || query_latent_bf16 == 0 || query_rope_bf16 == 0 || sequence_of_row == 0 || row_position == 0 || output_bf16 == 0 ||
		selected_positions == 0 || selected_count == 0u || union_positions == 0 || union_masks == 0 || union_counts == 0 ||
		(block_table != 0 && table_blocks == 0u) || (uint64_t)blocks * capacity > union_entries)
		return cudaErrorInvalidValue;
	{
		static bool granted = false;
		if (!granted)
		{
			cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, true, QUERIES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
			if (error != cudaSuccess)
				return error;
			granted = true;
		}
	}
	if (selected_count != SELECTED)
		return cudaErrorInvalidValue;
	LmPrefillSparseUnionKernel<ROWS_PER_BLOCK, SELECTED><<<blocks, LM_PREFILL_UNION_THREADS, 0, stream>>>(
		selected_positions, row_position, rows, block_table, capacity, union_positions, union_masks, union_counts);
	LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE, true, QUERIES><<<blocks, LM_PREFILL_ATTN_THREADS, shared, stream>>>(
		query_latent_bf16, query_rope_bf16, cache, sequence_of_row, row_position, qk_scale, output_bf16, rows, block_table, union_positions, union_masks, union_counts, capacity);
	return cudaPeekAtLastError();
}
