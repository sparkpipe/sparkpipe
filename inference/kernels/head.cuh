#pragma once


#include "inference/kernels/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/rows_tile.cuh"
#include "include/sparkpipe/spark_sampling.h"
#include "runtime/launch.h"
#include <stdint.h>

template<uint32_t ROWS>
static __device__ __forceinline__ void LmHeadRowsReduce(float (*best_score)[LM_ROWS_TILE_N], uint32_t (*best_token)[LM_ROWS_TILE_N], uint32_t threads)
{
	uint32_t stride, element, row, lane;
	for (stride = LM_ROWS_TILE_N / 2u; stride > 0u; stride >>= 1u)
	{
		for (element = threadIdx.x; element < ROWS * stride; element += threads)
		{
			row = element / stride;
			lane = element % stride;
			if ( best_score[row][lane + stride] > best_score[row][lane] || (best_score[row][lane + stride] == best_score[row][lane] && best_token[row][lane + stride] < best_token[row][lane]) )
			{
				best_score[row][lane] = best_score[row][lane + stride];
				best_token[row][lane] = best_token[row][lane + stride];
			}
		}
		__syncthreads();
	}
}

#define LM_SPLITMIX64_INCREMENT 0x9e3779b97f4a7c15ull
#define LM_SPLITMIX64_MULTIPLIER_A 0xbf58476d1ce4e5b9ull
#define LM_SPLITMIX64_MULTIPLIER_B 0x94d049bb133111ebull
#define LM_GUMBEL_UNIFORM_BITS 24u

static __host__ __device__ __forceinline__ uint64_t LmSplitMix64(uint64_t value)
{
	value += LM_SPLITMIX64_INCREMENT;
	value = (value ^ (value >> 30u)) * LM_SPLITMIX64_MULTIPLIER_A;
	value = (value ^ (value >> 27u)) * LM_SPLITMIX64_MULTIPLIER_B;
	return(value ^ (value >> 31u));
}

static __host__ __device__ __forceinline__ float LmGumbelNoise(uint64_t seed, uint32_t position, uint32_t token)
{
	const uint64_t bits = LmSplitMix64(seed ^ LmSplitMix64(((uint64_t)position << 32u) | token));
	const float uniform = ((float)(uint32_t)(bits >> (64u - LM_GUMBEL_UNIFORM_BITS)) + 0.5f) / (float)(1u << LM_GUMBEL_UNIFORM_BITS);
	return(-logf(-logf(uniform)));
}

typedef struct LmHeadSampling
{
	const SparkRowSampling *rules;
	const uint32_t *positions;
	uint32_t rows;
	uint32_t token_offset;
}
LmHeadSampling;

struct LmHeadGreedy
{
	__device__ __forceinline__ float operator()(float logit, uint32_t, uint32_t) const
	{
		return(logit);
	}
};

struct LmHeadSampler
{
	LmHeadSampling sampling;
	__device__ __forceinline__ float operator()(float logit, uint32_t row, uint32_t token) const
	{
		float inverse_temperature;
		if ( row >= sampling.rows )
			return(logit);
		inverse_temperature = sampling.rules[row].inverse_temperature;
		return(inverse_temperature == 0.0f ? logit : fmaf(logit,inverse_temperature,LmGumbelNoise(sampling.rules[row].seed,sampling.positions[row],sampling.token_offset + token)));
	}
};

template<uint32_t THREADS, uint32_t TILE, uint32_t ROWS, class Score>
static __device__ __forceinline__ void LmHeadCandidateRowsBody(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary, const Score &score)
{
	constexpr uint32_t per_thread = ROWS / (THREADS / LM_ROWS_TILE_N);
	static_assert(TILE % LM_ROWS_TILE_N == 0u, "head tiles hold whole token groups");
	__shared__ float weight_tile[LM_ROWS_TILE_N][LM_ROWS_TILE_K + 1u];
	__shared__ float input_tile[ROWS][LM_ROWS_TILE_K];
	__shared__ float best_score[ROWS][LM_ROWS_TILE_N];
	__shared__ uint32_t best_token[ROWS][LM_ROWS_TILE_N];
	const LmRowsTileOperand operand = {normed_bf16, head_weight_bf16, token_ids, hidden, hidden, rows, vocabulary, hidden};
	const uint32_t tile = blockIdx.x, first_row = blockIdx.y * ROWS, lane = threadIdx.x % LM_ROWS_TILE_N, group = threadIdx.x / LM_ROWS_TILE_N;
	uint32_t sub, j, index, best_id[per_thread];
	float total[per_thread], best[per_thread];
	for (j = 0u; j < per_thread; ++j)
	{
		best[j] = -INFINITY;
		best_id[j] = 0xffffffffu;
	}
	for (sub = 0u; sub < TILE; sub += LM_ROWS_TILE_N)
	{
		index = tile * TILE + sub + lane;
		LmRowsTileDot<THREADS, ROWS>(&operand, weight_tile, input_tile, tile * TILE + sub, first_row, total);
		for (j = 0u; j < per_thread; ++j)
			if ( index < vocabulary && score(total[j], first_row + group * per_thread + j, index) > best[j] )
			{
				best[j] = score(total[j], first_row + group * per_thread + j, index);
				best_id[j] = token_ids != 0 ? token_ids[index] : index;
			}
	}
	for (j = 0u; j < per_thread; ++j)
	{
		best_score[group * per_thread + j][lane] = best[j];
		best_token[group * per_thread + j][lane] = best_id[j];
	}
	__syncthreads();
	LmHeadRowsReduce<ROWS>(best_score, best_token, THREADS);
	if ( threadIdx.x < ROWS && first_row + threadIdx.x < rows )
	{
		candidate_score[((first_row + threadIdx.x) * gridDim.x) + tile] = best_score[threadIdx.x][0];
		candidate_token[((first_row + threadIdx.x) * gridDim.x) + tile] = best_token[threadIdx.x][0];
	}
}

template<uint32_t THREADS, uint32_t TILE, uint32_t ROWS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCandidateRowsKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary)
{
	LmHeadCandidateRowsBody<THREADS, TILE, ROWS>(normed_bf16, head_weight_bf16, token_ids, candidate_score, candidate_token, rows, hidden, vocabulary, LmHeadGreedy());
}

template<uint32_t THREADS, uint32_t TILE, uint32_t ROWS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSampledCandidateRowsKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary, const LmHeadSampling sampling)
{
	LmHeadCandidateRowsBody<THREADS, TILE, ROWS>(normed_bf16, head_weight_bf16, 0, candidate_score, candidate_token, rows, hidden, vocabulary, LmHeadSampler{sampling});
}

template<uint32_t THREADS, uint32_t TILE, class Score>
static __device__ __forceinline__ void LmHeadCandidateBody(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t hidden, uint32_t vocabulary, const Score &score)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.y,tile = blockIdx.x,index,element,stride;
	float best = -INFINITY,value;
	uint32_t best_token = 0u;
	for (index = tile * TILE + threadIdx.x; index < (tile + 1u) * TILE && index < vocabulary; index += THREADS)
	{
		uint32_t token = token_ids != 0 ? token_ids[index] : index;
		float total = 0.0f;
		for (element = 0u; element < hidden; ++element)
			total += LmBf16ToFloat(normed_bf16[((uint64_t)row * hidden) + element])
				* LmBf16ToFloat(head_weight_bf16[((uint64_t)token * hidden) + element]);
		value = score(total, row, index);
		if ( value > best )
		{
			best = value;
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

template<uint32_t THREADS, uint32_t TILE>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCandidateKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary)
{
	(void)rows;
	LmHeadCandidateBody<THREADS, TILE>(normed_bf16, head_weight_bf16, token_ids, candidate_score, candidate_token, hidden, vocabulary, LmHeadGreedy());
}

template<uint32_t THREADS, uint32_t TILE>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSampledCandidateKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t hidden, uint32_t vocabulary, const LmHeadSampling sampling)
{
	LmHeadCandidateBody<THREADS, TILE>(normed_bf16, head_weight_bf16, 0, candidate_score, candidate_token, hidden, vocabulary, LmHeadSampler{sampling});
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
