#pragma once

#include "runtime/launch.h"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/activation.cuh"
#include <math.h>
#include <stdint.h>

#define LM_INDEX_SCORE_WARPS 8u
#define LM_INDEX_SCORE_THREADS (LM_INDEX_SCORE_WARPS * LM_WARP_LANES)
#define LM_INDEX_SCORE_TILE_POSITIONS 8u
#define LM_INDEX_SCORE_BLOCK_TILES 32u
#define LM_INDEX_SCORE_BLOCK_POSITIONS (LM_INDEX_SCORE_BLOCK_TILES * LM_INDEX_SCORE_TILE_POSITIONS)
#define LM_INDEX_SCORE_SHARED_ROWS 8u
#define LM_INDEX_SCORE_SHARED_MIN_ROWS 64u
#define LM_INDEX_SCORE_STAGED_KEYS 128u
#define LM_INDEX_SCORE_STAGED_ROWS 64u
#define LM_INDEX_SCORE_STAGED_PAD 16u
#define LM_INDEX_FP8_KEY_FLOOR 1e-4f
#define LM_INDEX_FP8_QUERY_FLOOR 1e-10f

struct LmIndexScorePackedView
{
	LmKvView pages;
	SparkKvShard shard;
	const uint8_t *keys;
};

static __host__ __device__ __forceinline__ uint32_t LmIndexScoreBlocks(uint32_t positions)
{
	return((positions + LM_INDEX_SCORE_BLOCK_POSITIONS - 1u) / LM_INDEX_SCORE_BLOCK_POSITIONS);
}

static __host__ __device__ __forceinline__ uint32_t LmIndexScoreRowsPerBlock(uint32_t rows)
{
	return(rows >= LM_INDEX_SCORE_SHARED_MIN_ROWS ? LM_INDEX_SCORE_SHARED_ROWS : 1u);
}

static __host__ __device__ __forceinline__ const LmKvView &LmIndexScorePages(const LmKvView &view)
{
	return(view);
}

static __host__ __device__ __forceinline__ const LmKvView &LmIndexScorePages(const LmKvShardView &view)
{
	return(view.pages);
}

static __host__ __device__ __forceinline__ const LmKvView &LmIndexScorePages(const LmIndexScorePackedView &view)
{
	return(view.pages);
}

static __device__ __forceinline__ uint32_t LmIndexScoreKeys(const LmKvView &view, uint32_t bound)
{
	(void)view;
	return(bound);
}

static __device__ __forceinline__ uint32_t LmIndexScoreKeys(const LmKvShardView &view, uint32_t bound)
{
	return(SparkKvShardLocalKeys(view.shard, bound));
}

static __device__ __forceinline__ uint32_t LmIndexScoreKeys(const LmIndexScorePackedView &view, uint32_t bound)
{
	return(SparkKvShardLocalKeys(view.shard, bound));
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmIndexScoreSlot(const LmIndexScorePackedView &view, uint32_t sequence, uint32_t key, uint32_t row)
{
	(void)sequence;
	(void)row;
	return(view.keys + (uint64_t)key * Geometry::kSlotBytes);
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmIndexScoreSlot(const LmKvView &view, uint32_t sequence, uint32_t key, uint32_t row)
{
	return(LmKvSlotRequired<Geometry>(view, sequence, key, row, LM_KV_ACCESS_READ));
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmIndexScoreSlot(const LmKvShardView &view, uint32_t sequence, uint32_t key, uint32_t row)
{
	return(LmKvShardSlotRequired<Geometry>(view, sequence, SparkKvShardLocalPosition(view.shard, key), row, LM_KV_ACCESS_READ));
}

template<class Geometry, class Pages, uint32_t INDEX_HEADS, uint32_t INDEX_DIM, uint32_t ROWS_PER_BLOCK>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void LmWeightedSparseScoreKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight_bf16, Pages index_view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores)
{
	const LmKvView &index_cache = LmIndexScorePages(index_view);
	constexpr uint32_t HEAD_TILES = INDEX_HEADS / 16u;
	static_assert(INDEX_HEADS % 16u == 0u && HEAD_TILES >= 2u, "whole 16-head operand tiles, at least two");
	static_assert(INDEX_DIM % 16u == 0u, "whole 16-wide k steps");
	static_assert(LM_INDEX_SCORE_WARPS % ROWS_PER_BLOCK == 0u, "every row gets the same number of warps");
	const uint32_t steps = INDEX_DIM / 16u;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t group = lane / 4u,pair = lane % 4u;
	const uint32_t row = blockIdx.y * ROWS_PER_BLOCK + warp % ROWS_PER_BLOCK;
	const uint32_t tile_stride = LM_INDEX_SCORE_WARPS / ROWS_PER_BLOCK;
	uint32_t a[HEAD_TILES][INDEX_DIM / 16u][4];
	float weight[2u * HEAD_TILES];
	uint32_t sequence,bound,tile,step,head_tile,entry,valid;
	if ( scores == 0 || row >= rows )
		return;
	valid = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight_bf16 != 0 && LmKvViewIsConfigured(index_cache) ? 1u : 0u;
	sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
	bound = 0u;
	if ( valid == 0u || sequence >= index_cache.sequence_count )
	{
		if ( lane == 0u && blockIdx.x == 0u && warp < ROWS_PER_BLOCK )
			LmKvReportRequiredAccessFailure(index_cache,valid == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,LM_KV_ACCESS_READ,row,sequence,blockIdx.x * LM_INDEX_SCORE_BLOCK_POSITIONS,0xffffffffu);
		valid = 0u;
	}
	else
	{
		const uint16_t *query = index_query_bf16 + (uint64_t)row * INDEX_HEADS * INDEX_DIM;
		bound = context_length[sequence];
		if ( row_position != 0 && row_position[row] + 1u < bound )
			bound = row_position[row] + 1u;
		bound = LmIndexScoreKeys(index_view, bound);
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
			for (step = 0u; step < steps; step++)
			{
				const uint16_t *top = query + (uint64_t)(head_tile * 16u + group) * INDEX_DIM + step * 16u + pair * 2u;
				const uint16_t *bottom = top + 8u * INDEX_DIM;
				a[head_tile][step][0] = *(const uint32_t *)top;
				a[head_tile][step][1] = *(const uint32_t *)bottom;
				a[head_tile][step][2] = *(const uint32_t *)(top + 8u);
				a[head_tile][step][3] = *(const uint32_t *)(bottom + 8u);
			}
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
		{
			weight[2u * head_tile] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group]);
			weight[2u * head_tile + 1u] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group + 8u]);
		}
	}
	for (tile = warp / ROWS_PER_BLOCK; tile < LM_INDEX_SCORE_BLOCK_TILES; tile += tile_stride)
	{
		const uint32_t first = blockIdx.x * LM_INDEX_SCORE_BLOCK_POSITIONS + tile * LM_INDEX_SCORE_TILE_POSITIONS;
		const uint32_t mine = first + group;
		const uint32_t written = first + pair * 2u;
		float accumulator[HEAD_TILES][4] = {};
		float low,high;
		if ( first >= score_stride )
			break;
		if ( valid != 0u && first < bound )
		{
			const uint8_t *slot = mine < bound ? LmIndexScoreSlot<Geometry>(index_view,sequence,mine,row) : 0;
			for (step = 0u; step < steps; step++)
			{
				uint32_t b[2];
				b[0] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u) * sizeof(uint16_t)) : 0u;
				b[1] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u + 8u) * sizeof(uint16_t)) : 0u;
				for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
					LmMmaBf16(accumulator[head_tile],a[head_tile][step],b);
			}
		}
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
			for (entry = 0u; entry < 4u; entry++)
				accumulator[head_tile][entry] = fmaxf(accumulator[head_tile][entry], 0.0f);
		low = __fmul_rn(weight[0], accumulator[0][0]);
		high = __fmul_rn(weight[0], accumulator[0][1]);
		low = __fmaf_rn(weight[1], accumulator[0][2], low);
		high = __fmaf_rn(weight[1], accumulator[0][3], high);
		for (head_tile = 1u; head_tile < HEAD_TILES; head_tile++)
		{
			low = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][0], low);
			high = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][1], high);
			low = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][2], low);
			high = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][3], high);
		}
		for (entry = 4u; entry < LM_WARP_LANES; entry <<= 1u)
		{
			low += __shfl_xor_sync(0xffffffffu,low,(int)entry);
			high += __shfl_xor_sync(0xffffffffu,high,(int)entry);
		}
		if ( group == 0u )
		{
			if ( written < score_stride )
				scores[(uint64_t)row * score_stride + written] = valid != 0u && written < bound ? low * qk_scale : -INFINITY;
			if ( written + 1u < score_stride )
				scores[(uint64_t)row * score_stride + written + 1u] = valid != 0u && written + 1u < bound ? high * qk_scale : -INFINITY;
		}
	}
}

template<class Geometry, uint32_t INDEX_HEADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void LmWeightedSparseScoreStagedKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight_bf16, LmIndexScorePackedView index_view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores)
{
	constexpr uint32_t HEAD_TILES = INDEX_HEADS / 16u;
	constexpr uint32_t STEPS = INDEX_DIM / 16u;
	constexpr uint32_t KEY_BYTES = INDEX_DIM * (uint32_t)sizeof(uint16_t);
	constexpr uint32_t PITCH = KEY_BYTES + LM_INDEX_SCORE_STAGED_PAD;
	constexpr uint32_t TILES = LM_INDEX_SCORE_STAGED_KEYS / LM_INDEX_SCORE_TILE_POSITIONS;
	constexpr uint32_t ROWS_PER_WARP = LM_INDEX_SCORE_STAGED_ROWS / LM_INDEX_SCORE_WARPS;
	static_assert(INDEX_HEADS % 16u == 0u && HEAD_TILES >= 2u, "whole 16-head operand tiles, at least two");
	static_assert(INDEX_DIM % 16u == 0u && KEY_BYTES % 16u == 0u, "whole 16-wide k steps and 16-byte key chunks");
	static_assert(LM_INDEX_SCORE_STAGED_ROWS % LM_INDEX_SCORE_WARPS == 0u, "every warp gets the same number of rows");
	extern __shared__ __align__(16) uint8_t staged[];
	__shared__ uint32_t block_bound;
	const LmKvView &index_cache = index_view.pages;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t group = lane / 4u,pair = lane % 4u;
	const uint32_t key_first = blockIdx.x * LM_INDEX_SCORE_STAGED_KEYS;
	const uint32_t row_first = blockIdx.y * LM_INDEX_SCORE_STAGED_ROWS;
	const uint32_t configured = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight_bf16 != 0 && LmKvViewIsConfigured(index_cache) ? 1u : 0u;
	uint32_t local,tile,step,head_tile,entry;
	if ( scores == 0 )
		return;
	if ( threadIdx.x == 0u )
		block_bound = 0u;
	__syncthreads();
	if ( threadIdx.x < LM_INDEX_SCORE_STAGED_ROWS && configured != 0u && row_first + threadIdx.x < rows )
	{
		const uint32_t row = row_first + threadIdx.x,sequence = sequence_of_row[row];
		if ( sequence < index_cache.sequence_count )
		{
			uint32_t bound = context_length[sequence];
			if ( row_position != 0 && row_position[row] + 1u < bound )
				bound = row_position[row] + 1u;
			atomicMax(&block_bound,LmIndexScoreKeys(index_view,bound));
		}
	}
	__syncthreads();
	for (uint32_t chunk = threadIdx.x; chunk < LM_INDEX_SCORE_STAGED_KEYS * (KEY_BYTES / 16u); chunk += LM_INDEX_SCORE_THREADS)
	{
		const uint32_t key = chunk / (KEY_BYTES / 16u),part = chunk % (KEY_BYTES / 16u);
		uint4 value = make_uint4(0u,0u,0u,0u);
		if ( key_first + key < block_bound && key_first + key < score_stride )
			value = *(const uint4 *)(LmIndexScoreSlot<Geometry>(index_view,0u,key_first + key,row_first) + part * 16u);
		*(uint4 *)(staged + key * PITCH + part * 16u) = value;
	}
	__syncthreads();
	for (local = 0u; local < ROWS_PER_WARP; local++)
	{
		const uint32_t row = row_first + warp * ROWS_PER_WARP + local;
		uint32_t a[HEAD_TILES][STEPS][4];
		float weight[2u * HEAD_TILES];
		uint32_t sequence,bound,valid;
		if ( row >= rows )
			break;
		valid = configured;
		sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
		bound = 0u;
		if ( valid == 0u || sequence >= index_cache.sequence_count )
		{
			if ( lane == 0u && blockIdx.x == 0u )
				LmKvReportRequiredAccessFailure(index_cache,valid == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,LM_KV_ACCESS_READ,row,sequence,0u,0xffffffffu);
			valid = 0u;
		}
		else
		{
			const uint16_t *query = index_query_bf16 + (uint64_t)row * INDEX_HEADS * INDEX_DIM;
			bound = context_length[sequence];
			if ( row_position != 0 && row_position[row] + 1u < bound )
				bound = row_position[row] + 1u;
			bound = LmIndexScoreKeys(index_view, bound);
			for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
				for (step = 0u; step < STEPS; step++)
				{
					const uint16_t *top = query + (uint64_t)(head_tile * 16u + group) * INDEX_DIM + step * 16u + pair * 2u;
					const uint16_t *bottom = top + 8u * INDEX_DIM;
					a[head_tile][step][0] = *(const uint32_t *)top;
					a[head_tile][step][1] = *(const uint32_t *)bottom;
					a[head_tile][step][2] = *(const uint32_t *)(top + 8u);
					a[head_tile][step][3] = *(const uint32_t *)(bottom + 8u);
				}
			for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
			{
				weight[2u * head_tile] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group]);
				weight[2u * head_tile + 1u] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group + 8u]);
			}
		}
		for (tile = 0u; tile < TILES; tile++)
		{
			const uint32_t first = key_first + tile * LM_INDEX_SCORE_TILE_POSITIONS;
			const uint32_t mine = first + group;
			const uint32_t written = first + pair * 2u;
			float accumulator[HEAD_TILES][4] = {};
			float low,high;
			if ( first >= score_stride )
				break;
			if ( valid != 0u && first < bound )
			{
				const uint8_t *slot = mine < bound ? staged + (tile * LM_INDEX_SCORE_TILE_POSITIONS + group) * PITCH : 0;
				for (step = 0u; step < STEPS; step++)
				{
					uint32_t b[2];
					b[0] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u) * sizeof(uint16_t)) : 0u;
					b[1] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u + 8u) * sizeof(uint16_t)) : 0u;
					for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
						LmMmaBf16(accumulator[head_tile],a[head_tile][step],b);
				}
			}
			for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
				for (entry = 0u; entry < 4u; entry++)
					accumulator[head_tile][entry] = fmaxf(accumulator[head_tile][entry], 0.0f);
			low = __fmul_rn(weight[0], accumulator[0][0]);
			high = __fmul_rn(weight[0], accumulator[0][1]);
			low = __fmaf_rn(weight[1], accumulator[0][2], low);
			high = __fmaf_rn(weight[1], accumulator[0][3], high);
			for (head_tile = 1u; head_tile < HEAD_TILES; head_tile++)
			{
				low = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][0], low);
				high = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][1], high);
				low = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][2], low);
				high = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][3], high);
			}
			for (entry = 4u; entry < LM_WARP_LANES; entry <<= 1u)
			{
				low += __shfl_xor_sync(0xffffffffu,low,(int)entry);
				high += __shfl_xor_sync(0xffffffffu,high,(int)entry);
			}
			if ( group == 0u )
			{
				if ( written < score_stride )
					scores[(uint64_t)row * score_stride + written] = valid != 0u && written < bound ? low * qk_scale : -INFINITY;
				if ( written + 1u < score_stride )
					scores[(uint64_t)row * score_stride + written + 1u] = valid != 0u && written + 1u < bound ? high * qk_scale : -INFINITY;
			}
		}
	}
}

struct LmIndexScoreFp8Keys
{
	LmKvView pages;
	const uint8_t *values;
	const float *scales;
};

template<uint32_t INDEX_DIM>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void LmIndexKeysFp8Kernel(const uint16_t *__restrict__ keys_bf16, uint32_t count, uint8_t *__restrict__ keys_fp8, float *__restrict__ key_scales)
{
	constexpr uint32_t PER_LANE = INDEX_DIM / LM_WARP_LANES;
	static_assert(INDEX_DIM % (4u * LM_WARP_LANES) == 0u, "whole four-byte groups a lane");
	const uint32_t lane = threadIdx.x % LM_WARP_LANES;
	const uint64_t key = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / LM_WARP_LANES;
	float values[PER_LANE],amax = 0.0f,scale;
	uint32_t index,offset;
	if ( key >= count )
		return;
	for (index = 0u; index < PER_LANE; index++)
	{
		values[index] = LmBf16ToFloat(keys_bf16[key * INDEX_DIM + lane * PER_LANE + index]);
		amax = fmaxf(amax,fabsf(values[index]));
	}
	for (offset = LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
		amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,(int)offset));
	scale = LmActivationFp8Scale(amax,LM_INDEX_FP8_KEY_FLOOR);
	for (index = 0u; index < PER_LANE; index += 4u)
		*(uint32_t *)(keys_fp8 + key * INDEX_DIM + lane * PER_LANE + index) = (uint32_t)LmFloatPairToE4m3(values[index] / scale,values[index + 1u] / scale) |
			((uint32_t)LmFloatPairToE4m3(values[index + 2u] / scale,values[index + 3u] / scale) << 16u);
	if ( lane == 0u )
		key_scales[key] = scale;
}

template<uint32_t INDEX_HEADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void LmIndexQueryFp8Kernel(const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ head_weight_bf16, uint32_t rows, float weight_scale,
	uint8_t *__restrict__ query_fp8, float *__restrict__ head_weight)
{
	constexpr uint32_t PER_LANE = INDEX_DIM / LM_WARP_LANES;
	static_assert(INDEX_DIM % (4u * LM_WARP_LANES) == 0u, "whole four-byte groups a lane");
	const uint32_t lane = threadIdx.x % LM_WARP_LANES;
	const uint64_t pair = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / LM_WARP_LANES;
	float values[PER_LANE],amax = 0.0f,scale;
	uint32_t index,offset;
	if ( pair >= (uint64_t)rows * INDEX_HEADS )
		return;
	for (index = 0u; index < PER_LANE; index++)
	{
		values[index] = LmBf16ToFloat(query_bf16[pair * INDEX_DIM + lane * PER_LANE + index]);
		amax = fmaxf(amax,fabsf(values[index]));
	}
	for (offset = LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
		amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,(int)offset));
	scale = LmActivationFp8Scale(amax,LM_INDEX_FP8_QUERY_FLOOR);
	for (index = 0u; index < PER_LANE; index++)
		values[index] = fminf(fmaxf(values[index] / scale,-LM_E4M3_MAX),LM_E4M3_MAX);
	for (index = 0u; index < PER_LANE; index += 4u)
		*(uint32_t *)(query_fp8 + pair * INDEX_DIM + lane * PER_LANE + index) = (uint32_t)LmFloatPairToE4m3(values[index],values[index + 1u]) |
			((uint32_t)LmFloatPairToE4m3(values[index + 2u],values[index + 3u]) << 16u);
	if ( lane == 0u )
		head_weight[pair] = LmBf16ToFloat(head_weight_bf16[pair]) * scale * weight_scale;
}

template<uint32_t INDEX_HEADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void LmWeightedSparseScoreFp8Kernel(const uint8_t *__restrict__ query_fp8, const float *__restrict__ head_weight, LmIndexScoreFp8Keys keys,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t rows,
	uint32_t score_stride, float *__restrict__ scores)
{
	constexpr uint32_t HEAD_TILES = INDEX_HEADS / 16u;
	constexpr uint32_t STEPS = INDEX_DIM / 32u;
	constexpr uint32_t PITCH = INDEX_DIM + LM_INDEX_SCORE_STAGED_PAD;
	constexpr uint32_t TILES = LM_INDEX_SCORE_STAGED_KEYS / LM_INDEX_SCORE_TILE_POSITIONS;
	constexpr uint32_t ROWS_PER_WARP = LM_INDEX_SCORE_STAGED_ROWS / LM_INDEX_SCORE_WARPS;
	static_assert(INDEX_HEADS % 16u == 0u && HEAD_TILES >= 2u, "whole 16-head operand tiles, at least two");
	static_assert(INDEX_DIM % 32u == 0u && INDEX_DIM % 16u == 0u, "whole 32-wide k steps and 16-byte key chunks");
	static_assert(LM_INDEX_SCORE_STAGED_ROWS % LM_INDEX_SCORE_WARPS == 0u, "every warp gets the same number of rows");
	__shared__ __align__(16) uint8_t staged[LM_INDEX_SCORE_STAGED_KEYS * PITCH];
	__shared__ float staged_scales[LM_INDEX_SCORE_STAGED_KEYS];
	__shared__ uint32_t block_bound;
	const LmKvView &index_cache = keys.pages;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t group = lane / 4u,pair = lane % 4u;
	const uint32_t key_first = blockIdx.x * LM_INDEX_SCORE_STAGED_KEYS;
	const uint32_t row_first = blockIdx.y * LM_INDEX_SCORE_STAGED_ROWS;
	const uint32_t configured = sequence_of_row != 0 && context_length != 0 && query_fp8 != 0 && head_weight != 0 && keys.values != 0 && keys.scales != 0 &&
		LmKvViewIsConfigured(index_cache) ? 1u : 0u;
	uint32_t local,tile,step,head_tile,entry;
	if ( scores == 0 )
		return;
	if ( threadIdx.x == 0u )
		block_bound = 0u;
	__syncthreads();
	if ( threadIdx.x < LM_INDEX_SCORE_STAGED_ROWS && configured != 0u && row_first + threadIdx.x < rows )
	{
		const uint32_t row = row_first + threadIdx.x,sequence = sequence_of_row[row];
		if ( sequence < index_cache.sequence_count )
		{
			uint32_t bound = context_length[sequence];
			if ( row_position != 0 && row_position[row] + 1u < bound )
				bound = row_position[row] + 1u;
			atomicMax(&block_bound,bound);
		}
	}
	__syncthreads();
	for (uint32_t chunk = threadIdx.x; chunk < LM_INDEX_SCORE_STAGED_KEYS * (INDEX_DIM / 16u); chunk += LM_INDEX_SCORE_THREADS)
	{
		const uint32_t key = chunk / (INDEX_DIM / 16u),part = chunk % (INDEX_DIM / 16u);
		uint4 value = make_uint4(0u,0u,0u,0u);
		if ( key_first + key < block_bound && key_first + key < score_stride )
			value = *(const uint4 *)(keys.values + (uint64_t)(key_first + key) * INDEX_DIM + part * 16u);
		*(uint4 *)(staged + key * PITCH + part * 16u) = value;
	}
	for (uint32_t key = threadIdx.x; key < LM_INDEX_SCORE_STAGED_KEYS; key += LM_INDEX_SCORE_THREADS)
		staged_scales[key] = key_first + key < block_bound && key_first + key < score_stride ? keys.scales[key_first + key] : 0.0f;
	__syncthreads();
	for (local = 0u; local < ROWS_PER_WARP; local++)
	{
		const uint32_t row = row_first + warp * ROWS_PER_WARP + local;
		uint32_t a[HEAD_TILES][STEPS][4];
		float weight[2u * HEAD_TILES];
		uint32_t sequence,bound,valid;
		if ( row >= rows )
			break;
		valid = configured;
		sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
		bound = 0u;
		if ( valid == 0u || sequence >= index_cache.sequence_count )
		{
			if ( lane == 0u && blockIdx.x == 0u )
				LmKvReportRequiredAccessFailure(index_cache,valid == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,LM_KV_ACCESS_READ,row,sequence,0u,0xffffffffu);
			valid = 0u;
		}
		else
		{
			const uint8_t *query = query_fp8 + (uint64_t)row * INDEX_HEADS * INDEX_DIM;
			bound = context_length[sequence];
			if ( row_position != 0 && row_position[row] + 1u < bound )
				bound = row_position[row] + 1u;
			for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
				for (step = 0u; step < STEPS; step++)
				{
					const uint8_t *top = query + (uint64_t)(head_tile * 16u + group) * INDEX_DIM + step * 32u + pair * 4u;
					const uint8_t *bottom = top + 8u * INDEX_DIM;
					a[head_tile][step][0] = *(const uint32_t *)top;
					a[head_tile][step][1] = *(const uint32_t *)bottom;
					a[head_tile][step][2] = *(const uint32_t *)(top + 16u);
					a[head_tile][step][3] = *(const uint32_t *)(bottom + 16u);
				}
			for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
			{
				weight[2u * head_tile] = head_weight[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group];
				weight[2u * head_tile + 1u] = head_weight[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group + 8u];
			}
		}
		for (tile = 0u; tile < TILES; tile++)
		{
			const uint32_t first = key_first + tile * LM_INDEX_SCORE_TILE_POSITIONS;
			const uint32_t mine = first + group;
			const uint32_t written = first + pair * 2u;
			float accumulator[HEAD_TILES][4] = {};
			float low,high;
			if ( first >= score_stride )
				break;
			if ( valid != 0u && first < bound )
			{
				const uint8_t *slot = mine < bound ? staged + (tile * LM_INDEX_SCORE_TILE_POSITIONS + group) * PITCH : 0;
				for (step = 0u; step < STEPS; step++)
				{
					uint32_t b[2];
					b[0] = slot != 0 ? *(const uint32_t *)(slot + step * 32u + pair * 4u) : 0u;
					b[1] = slot != 0 ? *(const uint32_t *)(slot + step * 32u + 16u + pair * 4u) : 0u;
					for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
						LmMmaE4m3(accumulator[head_tile],a[head_tile][step],b);
				}
			}
			for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
				for (entry = 0u; entry < 4u; entry++)
					accumulator[head_tile][entry] = fmaxf(accumulator[head_tile][entry], 0.0f);
			low = __fmul_rn(weight[0], accumulator[0][0]);
			high = __fmul_rn(weight[0], accumulator[0][1]);
			low = __fmaf_rn(weight[1], accumulator[0][2], low);
			high = __fmaf_rn(weight[1], accumulator[0][3], high);
			for (head_tile = 1u; head_tile < HEAD_TILES; head_tile++)
			{
				low = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][0], low);
				high = __fmaf_rn(weight[2u * head_tile], accumulator[head_tile][1], high);
				low = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][2], low);
				high = __fmaf_rn(weight[2u * head_tile + 1u], accumulator[head_tile][3], high);
			}
			for (entry = 4u; entry < LM_WARP_LANES; entry <<= 1u)
			{
				low += __shfl_xor_sync(0xffffffffu,low,(int)entry);
				high += __shfl_xor_sync(0xffffffffu,high,(int)entry);
			}
			if ( group == 0u )
			{
				if ( written < score_stride )
					scores[(uint64_t)row * score_stride + written] = valid != 0u && written < bound ? low * staged_scales[tile * LM_INDEX_SCORE_TILE_POSITIONS + pair * 2u] :
						-INFINITY;
				if ( written + 1u < score_stride )
					scores[(uint64_t)row * score_stride + written + 1u] = valid != 0u && written + 1u < bound ?
						high * staged_scales[tile * LM_INDEX_SCORE_TILE_POSITIONS + pair * 2u + 1u] : -INFINITY;
			}
		}
	}
}

template<uint32_t INDEX_DIM>
static inline cudaError_t LmIndexKeysFp8Launch(const uint16_t *keys_bf16, uint32_t count, uint8_t *keys_fp8, float *key_scales, cudaStream_t stream)
{
	const uint32_t warps = LM_INDEX_SCORE_THREADS / LM_WARP_LANES;
	if ( keys_bf16 == 0 || keys_fp8 == 0 || key_scales == 0 )
		return(cudaErrorInvalidValue);
	if ( count == 0u )
		return(cudaSuccess);
	LM_LAUNCH((LmIndexKeysFp8Kernel<INDEX_DIM>), (count + warps - 1u) / warps, LM_INDEX_SCORE_THREADS, 0, stream, keys_bf16, count, keys_fp8, key_scales);
	return(cudaPeekAtLastError());
}

template<uint32_t INDEX_HEADS, uint32_t INDEX_DIM>
static inline cudaError_t LmIndexQueryFp8Launch(const uint16_t *query_bf16, const uint16_t *head_weight_bf16, uint32_t rows, float weight_scale, uint8_t *query_fp8,
	float *head_weight, cudaStream_t stream)
{
	const uint32_t warps = LM_INDEX_SCORE_THREADS / LM_WARP_LANES;
	const uint64_t pairs = (uint64_t)rows * INDEX_HEADS;
	if ( query_bf16 == 0 || head_weight_bf16 == 0 || query_fp8 == 0 || head_weight == 0 )
		return(cudaErrorInvalidValue);
	if ( rows == 0u )
		return(cudaSuccess);
	LM_LAUNCH((LmIndexQueryFp8Kernel<INDEX_HEADS,INDEX_DIM>), (uint32_t)((pairs + warps - 1u) / warps), LM_INDEX_SCORE_THREADS, 0, stream, query_bf16,
		head_weight_bf16, rows, weight_scale, query_fp8, head_weight);
	return(cudaPeekAtLastError());
}

template<uint32_t INDEX_HEADS, uint32_t INDEX_DIM>
static inline cudaError_t LmWeightedSparseScoreFp8Launch(const uint8_t *query_fp8, const float *head_weight, LmIndexScoreFp8Keys keys,
	const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position, uint32_t rows, uint32_t score_stride, float *scores,
	cudaStream_t stream)
{
	if ( rows == 0u || score_stride == 0u )
		return(cudaSuccess);
	LM_LAUNCH((LmWeightedSparseScoreFp8Kernel<INDEX_HEADS,INDEX_DIM>), dim3((score_stride + LM_INDEX_SCORE_STAGED_KEYS - 1u) / LM_INDEX_SCORE_STAGED_KEYS,
		(rows + LM_INDEX_SCORE_STAGED_ROWS - 1u) / LM_INDEX_SCORE_STAGED_ROWS), LM_INDEX_SCORE_THREADS, 0, stream, query_fp8, head_weight, keys,
		sequence_of_row, context_length, row_position, rows, score_stride, scores);
	return(cudaPeekAtLastError());
}

template<class Pages>
static constexpr uint32_t LmIndexScoreStagedView(void)
{
	return(0u);
}

template<>
constexpr uint32_t LmIndexScoreStagedView<LmIndexScorePackedView>(void)
{
	return(1u);
}

template<class Geometry, uint32_t INDEX_HEADS, uint32_t INDEX_DIM, class Pages>
static inline cudaError_t LmWeightedSparseScoreLaunch(const uint16_t *index_query_bf16, const uint16_t *head_weight_bf16, Pages index_view, const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *scores, cudaStream_t stream)
{
	if constexpr ( LmIndexScoreStagedView<Pages>() != 0u )
	{
		LM_LAUNCH((LmWeightedSparseScoreStagedKernel<Geometry,INDEX_HEADS,INDEX_DIM>), dim3((score_stride + LM_INDEX_SCORE_STAGED_KEYS - 1u) / LM_INDEX_SCORE_STAGED_KEYS,
			(rows + LM_INDEX_SCORE_STAGED_ROWS - 1u) / LM_INDEX_SCORE_STAGED_ROWS), LM_INDEX_SCORE_THREADS,
			LM_INDEX_SCORE_STAGED_KEYS * (INDEX_DIM * (uint32_t)sizeof(uint16_t) + LM_INDEX_SCORE_STAGED_PAD), stream,
			index_query_bf16,head_weight_bf16,index_view,sequence_of_row,context_length,row_position,rows,score_stride,qk_scale,scores);
		return(cudaPeekAtLastError());
	}
	if ( LmIndexScoreRowsPerBlock(rows) == LM_INDEX_SCORE_SHARED_ROWS )
		LM_LAUNCH((LmWeightedSparseScoreKernel<Geometry,Pages,INDEX_HEADS,INDEX_DIM,LM_INDEX_SCORE_SHARED_ROWS>), dim3(LmIndexScoreBlocks(score_stride),(rows + LM_INDEX_SCORE_SHARED_ROWS - 1u) / LM_INDEX_SCORE_SHARED_ROWS), LM_INDEX_SCORE_THREADS, 0, stream,
			index_query_bf16,head_weight_bf16,index_view,sequence_of_row,context_length,row_position,rows,score_stride,qk_scale,scores);
	else
		LM_LAUNCH((LmWeightedSparseScoreKernel<Geometry,Pages,INDEX_HEADS,INDEX_DIM,1u>), dim3(LmIndexScoreBlocks(score_stride),rows), LM_INDEX_SCORE_THREADS, 0, stream,
			index_query_bf16,head_weight_bf16,index_view,sequence_of_row,context_length,row_position,rows,score_stride,qk_scale,scores);
	return(cudaPeekAtLastError());
}
