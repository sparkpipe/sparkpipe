#pragma once


#include "inference/kernels/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"
#include <stdint.h>

template<uint32_t THREADS, uint32_t TILE>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCandidateKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.y,tile = blockIdx.x,index,element,stride;
	float best = -INFINITY;
	uint32_t best_token = 0u;
	for (index = tile * TILE + threadIdx.x; index < (tile + 1u) * TILE && index < vocabulary; index += THREADS)
	{
		uint32_t token = token_ids != 0 ? token_ids[index] : index;
		float total = 0.0f;
		for (element = 0u; element < hidden; ++element)
			total += LmBf16ToFloat(normed_bf16[((uint64_t)row * hidden) + element])
				* LmBf16ToFloat(head_weight_bf16[((uint64_t)token * hidden) + element]);
		if ( total > best )
		{
			best = total;
			best_token = token;
		}
	}
	shared_score[threadIdx.x] = best;
	shared_token[threadIdx.x] = best_token;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride && shared_score[threadIdx.x + stride] > shared_score[threadIdx.x] )
		{
			shared_score[threadIdx.x] = shared_score[threadIdx.x + stride];
			shared_token[threadIdx.x] = shared_token[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		candidate_score[(row * gridDim.x) + tile] = shared_score[0];
		candidate_token[(row * gridDim.x) + tile] = shared_token[0];
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCommitKernel(const float *__restrict__ candidate_score, const uint32_t *__restrict__ candidate_token, uint32_t tiles, uint32_t *__restrict__ token_out, float *__restrict__ score_out, uint32_t rows)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.x,index,stride;
	float best = -INFINITY;
	uint32_t best_token = 0xffffffffu;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < tiles; index += THREADS)
	{
		float score = candidate_score[(row * tiles) + index];
		uint32_t token = candidate_token[(row * tiles) + index];
		if ( score > best || (score == best && token < best_token) )
		{
			best = score;
			best_token = token;
		}
	}
	shared_score[threadIdx.x] = best;
	shared_token[threadIdx.x] = best_token;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
		{
			float other = shared_score[threadIdx.x + stride];
			uint32_t other_token = shared_token[threadIdx.x + stride];
			if ( other > shared_score[threadIdx.x]
				|| (other == shared_score[threadIdx.x] && other_token < shared_token[threadIdx.x]) )
			{
				shared_score[threadIdx.x] = other;
				shared_token[threadIdx.x] = other_token;
			}
		}
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		token_out[row] = shared_token[0];
		if ( score_out != 0 )
			score_out[row] = shared_score[0];
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSoftmaxKernel(float *__restrict__ logits, uint32_t rows, uint32_t vocabulary, float temperature)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint64_t base = (uint64_t)blockIdx.x * vocabulary;
	uint32_t index;
	float local = -INFINITY,maximum,total = 0.0f,inverse_temperature;
	if ( blockIdx.x >= rows )
		return;
	inverse_temperature = 1.0f / fmaxf(temperature,1.0e-4f);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		local = fmaxf(local,logits[base + index] * inverse_temperature);
	maximum = LmBlockMax<THREADS>(local,reduction);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
	{
		float value = __expf((logits[base + index] * inverse_temperature) - maximum);
		logits[base + index] = value;
		total += value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		logits[base + index] /= fmaxf(total,1.0e-20f);
}


template<uint32_t TILE, uint32_t K>
static inline uint32_t LmHeadTopkCandidatePairs(uint32_t rows, uint32_t vocabulary)
{
	return(rows * ((vocabulary + TILE - 1u) / TILE) * K);
}

template<uint32_t THREADS, uint32_t TILE, uint32_t K>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadTopkCandidateKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	__shared__ uint32_t taken[K];
	#define LM_HEAD_TOPK_LOCAL ((TILE + THREADS - 1u) / THREADS)
	float local_score[LM_HEAD_TOPK_LOCAL];
	uint32_t local_token[LM_HEAD_TOPK_LOCAL];
	uint32_t row = blockIdx.y,tile = blockIdx.x,index,element,round,pick,stride;
	uint32_t tile_begin = tile * TILE,tile_end = tile_begin + TILE,count = 0u;
	if ( tile_end > vocabulary )
		tile_end = vocabulary;
	for (index = tile_begin + threadIdx.x; index < tile_end; index += THREADS)
	{
		uint32_t token = token_ids != 0 ? token_ids[index] : index;
		float total = 0.0f;
		for (element = 0u; element < hidden; ++element)
			total += LmBf16ToFloat(normed_bf16[((uint64_t)row * hidden) + element])
				* LmBf16ToFloat(head_weight_bf16[((uint64_t)token * hidden) + element]);
		local_score[count] = total;
		local_token[count] = token;
		++count;
	}
	for (round = 0u; round < K; ++round)
	{
		float best = -INFINITY;
		uint32_t best_token = 0xffffffffu;
		for (pick = 0u; pick < count; ++pick)
		{
			uint32_t held = 0u;
			for (index = 0u; index < round; ++index)
				if ( taken[index] == local_token[pick] )
					held = 1u;
			if ( held != 0u )
				continue;
			if ( local_score[pick] > best ||
				(local_score[pick] == best && local_token[pick] < best_token) )
			{
				best = local_score[pick];
				best_token = local_token[pick];
			}
		}
		shared_score[threadIdx.x] = best;
		shared_token[threadIdx.x] = best_token;
		__syncthreads();
		for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
		{
			if ( threadIdx.x < stride )
			{
				float other = shared_score[threadIdx.x + stride];
				uint32_t other_token = shared_token[threadIdx.x + stride];
				if ( other > shared_score[threadIdx.x]
					|| (other == shared_score[threadIdx.x] && other_token < shared_token[threadIdx.x]) )
				{
					shared_score[threadIdx.x] = other;
					shared_token[threadIdx.x] = other_token;
				}
			}
			__syncthreads();
		}
		if ( threadIdx.x == 0u )
		{
			candidate_score[((row * gridDim.x) + tile) * K + round] = shared_score[0];
			candidate_token[((row * gridDim.x) + tile) * K + round] = shared_token[0];
			taken[round] = shared_token[0];
		}
		__syncthreads();
	}
	#undef LM_HEAD_TOPK_LOCAL
}

template<uint32_t THREADS, uint32_t K>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadTopkCommitKernel(const float *__restrict__ candidate_score, const uint32_t *__restrict__ candidate_token, uint32_t tiles, uint32_t *__restrict__ token_out, float *__restrict__ score_out, uint32_t rows)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	__shared__ uint32_t taken[K];
	uint32_t row = blockIdx.x,candidates = tiles * K,index,round,stride,prior;
	if ( row >= rows )
		return;
	for (round = 0u; round < K; ++round)
	{
		float best = -INFINITY;
		uint32_t best_token = 0xffffffffu;
		for (index = threadIdx.x; index < candidates; index += THREADS)
		{
			float score = candidate_score[(row * candidates) + index];
			uint32_t token = candidate_token[(row * candidates) + index];
			uint32_t held = 0u;
			for (prior = 0u; prior < round; ++prior)
				if ( taken[prior] == token )
					held = 1u;
			if ( held != 0u )
				continue;
			if ( score > best || (score == best && token < best_token) )
			{
				best = score;
				best_token = token;
			}
		}
		shared_score[threadIdx.x] = best;
		shared_token[threadIdx.x] = best_token;
		__syncthreads();
		for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
		{
			if ( threadIdx.x < stride )
			{
				float other = shared_score[threadIdx.x + stride];
				uint32_t other_token = shared_token[threadIdx.x + stride];
				if ( other > shared_score[threadIdx.x]
					|| (other == shared_score[threadIdx.x] && other_token < shared_token[threadIdx.x]) )
				{
					shared_score[threadIdx.x] = other;
					shared_token[threadIdx.x] = other_token;
				}
			}
			__syncthreads();
		}
		if ( threadIdx.x == 0u )
		{
			token_out[(row * K) + round] = shared_token[0];
			if ( score_out != 0 )
				score_out[(row * K) + round] = shared_score[0];
			taken[round] = shared_token[0];
		}
		__syncthreads();
	}
}

template<uint32_t THREADS, uint32_t TILE, uint32_t K>
static int32_t LmHeadTopk(
	const uint16_t *normed_bf16,
	const uint16_t *head_weight_bf16,
	const uint32_t *token_ids,
	float *candidate_score,
	uint32_t *candidate_token,
	uint32_t *token_out,
	float *score_out,
	uint32_t rows,
	uint32_t hidden,
	uint32_t vocabulary,
	cudaStream_t stream)
{
	uint32_t tiles;

	if ( normed_bf16 == 0 || head_weight_bf16 == 0 ||
		candidate_score == 0 || candidate_token == 0 || token_out == 0 ||
		rows == 0u || hidden == 0u || vocabulary == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	tiles = (vocabulary + TILE - 1u) / TILE;
	LM_LAUNCH((LmHeadTopkCandidateKernel<THREADS,TILE,K>), dim3(tiles,rows), THREADS, 0, stream,
		normed_bf16,head_weight_bf16,token_ids,candidate_score,candidate_token,rows,hidden,vocabulary);
	LM_LAUNCH((LmHeadTopkCommitKernel<THREADS,K>), rows, THREADS, 0, stream,
		candidate_score,candidate_token,tiles,token_out,score_out,rows);
	return(cudaPeekAtLastError() == cudaSuccess
		? LM_LAUNCH_OK
		: LM_LAUNCH_ERR_LAUNCH);
}
