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

template<class Geometry, uint32_t HEADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1)
void LmLatentAttentionPrefillKernel(
	const uint16_t *__restrict__ query_latent_bf16,
	const uint16_t *__restrict__ query_rope_bf16,
	LmKvView cache,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	uint16_t *__restrict__ output_bf16,
	uint32_t rows)
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
	const uint32_t first_row = (gridDim.x - 1u - blockIdx.x) * ROWS_PER_BLOCK;
	const uint32_t query_base = (uint32_t)__cvta_generic_to_shared(query);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	float output[COLUMN_FRAGS][4];
	uint32_t sequence,last_position,tiles,tile,row,index,chunk,column,step,frag,q;

	sequence = sequence_of_row[first_row];
	last_position = 0u;
	for (row = first_row; row < first_row + ROWS_PER_BLOCK && row < rows; row++)
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
	tiles = last_position / LM_PREFILL_ATTN_TILE + 1u;

	for (index = threadIdx.x; index < LM_PREFILL_ATTN_QUERIES * CHUNKS; index += LM_PREFILL_ATTN_THREADS)
	{
		q = index / CHUNKS;
		chunk = index % CHUNKS;
		row = first_row + q / HEADS;
		const uint16_t *source = row >= rows ? query_latent_bf16
			: chunk * 8u < LATENT ? query_latent_bf16 + ((uint64_t)row * HEADS + q % HEADS) * LATENT + chunk * 8u
			: query_rope_bf16 + ((uint64_t)row * HEADS + q % HEADS) * ROPE + (chunk * 8u - LATENT);
		LmPrefillCopy(query_base + q * ROW_BYTES + LmPrefillSwizzle(q,chunk) * 16u, source, row >= rows ? 0u : 16u);
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
			const uint32_t position = tile * LM_PREFILL_ATTN_TILE + lane;
			const uint32_t query_row = first_row + q / HEADS;
			const uint32_t limit = query_row < rows ? row_position[query_row] : 0u;
			float value = (score_parts[q * LM_PREFILL_ATTN_SCORE_STRIDE + lane] + score_parts[(LM_PREFILL_ATTN_QUERIES + q) * LM_PREFILL_ATTN_SCORE_STRIDE + lane]) * qk_scale;
			float tile_max,previous_max,next_max,weight,total,scale;
			uint16_t high;
			value = position <= limit ? value : -INFINITY;
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
			if (row >= rows)
				continue;
			column = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
			const float inverse = 1.0f / running_sum[q];
			*(uint32_t *)(output_bf16 + ((uint64_t)row * HEADS + q % HEADS) * LATENT + column) =
				(uint32_t)LmFloatToBf16(output[frag][index] * inverse) | ((uint32_t)LmFloatToBf16(output[frag][index + 1u] * inverse) << 16u);
		}
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
	cudaStream_t stream)
{
	constexpr uint32_t shared = LmPrefillAttnSharedBytes<LATENT + ROPE>();
	if (rows == 0u || query_latent_bf16 == 0 || query_rope_bf16 == 0 || sequence_of_row == 0 || row_position == 0 || output_bf16 == 0)
		return cudaErrorInvalidValue;
	if (heads == 4u)
	{
		static bool granted = false;
		if (!granted)
		{
			cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
			if (error != cudaSuccess)
				return error;
			granted = true;
		}
		LmLatentAttentionPrefillKernel<Geometry, 4u, LATENT, ROPE><<<(rows + LM_PREFILL_ATTN_QUERIES / 4u - 1u) / (LM_PREFILL_ATTN_QUERIES / 4u), LM_PREFILL_ATTN_THREADS, shared, stream>>>(
			query_latent_bf16, query_rope_bf16, cache, sequence_of_row, row_position, qk_scale, output_bf16, rows);
		return cudaPeekAtLastError();
	}
	return cudaErrorInvalidValue;
}
