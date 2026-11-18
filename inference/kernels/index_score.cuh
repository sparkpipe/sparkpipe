#pragma once

#include "runtime/launch.h"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/mma.cuh"
#include <math.h>
#include <stdint.h>

#define LM_INDEX_SCORE_WARPS 8u
#define LM_INDEX_SCORE_THREADS (LM_INDEX_SCORE_WARPS * LM_WARP_LANES)
#define LM_INDEX_SCORE_TILE_POSITIONS 8u
#define LM_INDEX_SCORE_BLOCK_TILES 32u
#define LM_INDEX_SCORE_BLOCK_POSITIONS (LM_INDEX_SCORE_BLOCK_TILES * LM_INDEX_SCORE_TILE_POSITIONS)
#define LM_INDEX_SCORE_SHARED_ROWS 8u
#define LM_INDEX_SCORE_SHARED_MIN_ROWS 64u

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

template<uint32_t STEPS>
static __device__ __forceinline__ void LmIndexScoreKeyFragments(const uint8_t *slot, uint32_t pair, uint32_t b[STEPS][2])
{
	uint32_t step;
	for (step = 0u; step < STEPS; step++)
	{
		b[step][0] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u) * sizeof(uint16_t)) : 0u;
		b[step][1] = slot != 0 ? *(const uint32_t *)(slot + (step * 16u + pair * 2u + 8u) * sizeof(uint16_t)) : 0u;
	}
}

template<class Geometry, class Pages, uint32_t INDEX_HEADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void LmWeightedSparseScoreSlotsKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight_bf16, Pages index_view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores)
{
	__shared__ const uint8_t *block_slots[LM_INDEX_SCORE_BLOCK_POSITIONS];
	__shared__ uint32_t block_sequence,block_bound,block_mixed;
	const LmKvView &index_cache = LmIndexScorePages(index_view);
	constexpr uint32_t HEAD_TILES = INDEX_HEADS / 16u,STEPS = INDEX_DIM / 16u;
	static_assert(INDEX_HEADS % 16u == 0u && HEAD_TILES >= 2u, "whole 16-head operand tiles, at least two");
	static_assert(INDEX_DIM % 16u == 0u, "whole 16-wide k steps");
	static_assert(LM_INDEX_SCORE_THREADS == LM_INDEX_SCORE_BLOCK_POSITIONS, "one thread finds each position's slot");
	const uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t group = lane / 4u,pair = lane % 4u;
	const uint32_t row = blockIdx.y * LM_INDEX_SCORE_WARPS + warp;
	const uint32_t block_first = blockIdx.x * LM_INDEX_SCORE_BLOCK_POSITIONS;
	const uint32_t live = scores != 0 && row < rows ? 1u : 0u;
	uint32_t a[HEAD_TILES][STEPS][4];
	uint32_t b[STEPS][2],next[STEPS][2];
	float weight[2u * HEAD_TILES];
	uint32_t sequence = 0xffffffffu,bound = 0u,valid = 0u,tile,step,head_tile,entry;
	if ( threadIdx.x == 0u )
	{
		block_sequence = 0xffffffffu;
		block_bound = 0u;
		block_mixed = 0u;
	}
	__syncthreads();
	if ( live != 0u )
	{
		valid = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight_bf16 != 0 && LmKvViewIsConfigured(index_cache) ? 1u : 0u;
		sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
		if ( valid == 0u || sequence >= index_cache.sequence_count )
		{
			if ( lane == 0u && blockIdx.x == 0u )
				LmKvReportRequiredAccessFailure(index_cache,valid == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,LM_KV_ACCESS_READ,row,sequence,block_first,0xffffffffu);
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
	}
	if ( lane == 0u && valid != 0u )
		(void)atomicCAS(&block_sequence,0xffffffffu,sequence);
	__syncthreads();
	if ( lane == 0u && valid != 0u )
	{
		if ( block_sequence != sequence )
			block_mixed = 1u;
		(void)atomicMax(&block_bound,bound);
	}
	__syncthreads();
	const uint32_t shared_slots = block_mixed == 0u && block_sequence != 0xffffffffu ? 1u : 0u;
	if ( shared_slots != 0u )
	{
		const uint32_t position = block_first + threadIdx.x;
		block_slots[threadIdx.x] = position < block_bound && position < score_stride ?
			LmIndexScoreSlot<Geometry>(index_view,block_sequence,position,blockIdx.y * LM_INDEX_SCORE_WARPS) : 0;
	}
	__syncthreads();
	if ( live == 0u )
		return;
	LmIndexScoreKeyFragments<STEPS>(valid != 0u && block_first + group < bound && block_first < score_stride ? (shared_slots != 0u ? block_slots[group] :
		LmIndexScoreSlot<Geometry>(index_view,sequence,block_first + group,row)) : 0,pair,next);
	for (tile = 0u; tile < LM_INDEX_SCORE_BLOCK_TILES; tile++)
	{
		const uint32_t first = block_first + tile * LM_INDEX_SCORE_TILE_POSITIONS;
		const uint32_t ahead = first + LM_INDEX_SCORE_TILE_POSITIONS + group;
		const uint32_t written = first + pair * 2u;
		float accumulator[HEAD_TILES][4] = {};
		float low,high;
		if ( first >= score_stride )
			break;
		for (step = 0u; step < STEPS; step++)
		{
			b[step][0] = next[step][0];
			b[step][1] = next[step][1];
		}
		LmIndexScoreKeyFragments<STEPS>(valid != 0u && tile + 1u < LM_INDEX_SCORE_BLOCK_TILES && ahead < bound && first + LM_INDEX_SCORE_TILE_POSITIONS < score_stride ?
			(shared_slots != 0u ? block_slots[(tile + 1u) * LM_INDEX_SCORE_TILE_POSITIONS + group] : LmIndexScoreSlot<Geometry>(index_view,sequence,ahead,row)) : 0,pair,next);
		if ( valid != 0u && first < bound )
			for (step = 0u; step < STEPS; step++)
				for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
					LmMmaBf16(accumulator[head_tile],a[head_tile][step],b[step]);
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

template<class Geometry, uint32_t INDEX_HEADS, uint32_t INDEX_DIM, class Pages>
static inline cudaError_t LmWeightedSparseScoreLaunch(const uint16_t *index_query_bf16, const uint16_t *head_weight_bf16, Pages index_view, const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *scores, cudaStream_t stream)
{
	static_assert(LM_INDEX_SCORE_SHARED_ROWS == LM_INDEX_SCORE_WARPS, "the block-slot kernel gives every warp of a block its own row");
	if ( LmIndexScoreRowsPerBlock(rows) == LM_INDEX_SCORE_SHARED_ROWS )
		LM_LAUNCH((LmWeightedSparseScoreSlotsKernel<Geometry,Pages,INDEX_HEADS,INDEX_DIM>), dim3(LmIndexScoreBlocks(score_stride),(rows + LM_INDEX_SCORE_SHARED_ROWS - 1u) / LM_INDEX_SCORE_SHARED_ROWS), LM_INDEX_SCORE_THREADS, 0, stream,
			index_query_bf16,head_weight_bf16,index_view,sequence_of_row,context_length,row_position,rows,score_stride,qk_scale,scores);
	else
		LM_LAUNCH((LmWeightedSparseScoreKernel<Geometry,Pages,INDEX_HEADS,INDEX_DIM,1u>), dim3(LmIndexScoreBlocks(score_stride),rows), LM_INDEX_SCORE_THREADS, 0, stream,
			index_query_bf16,head_weight_bf16,index_view,sequence_of_row,context_length,row_position,rows,score_stride,qk_scale,scores);
	return(cudaPeekAtLastError());
}
