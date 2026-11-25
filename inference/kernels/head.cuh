#pragma once


#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/gemm.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/rows_tile.cuh"
#include "include/sparkpipe/spark_sampling.h"
#include "runtime/launch.h"
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"

template<bool TIE_BY_TOKEN>
static __device__ __forceinline__ bool LmHeadBetter(float other_score, uint32_t other_token, float best_score, uint32_t best_token)
{
	return(other_score > best_score || (TIE_BY_TOKEN && other_score == best_score && other_token < best_token));
}

template<uint32_t THREADS, bool TIE_BY_TOKEN>
static __device__ __forceinline__ void LmHeadBlockArgMax(float best, uint32_t best_token, float *shared_score, uint32_t *shared_token, float *out_score, uint32_t *out_token)
{
#if defined(__CUDA_ARCH__)
	constexpr uint32_t lanes = THREADS < 32u ? THREADS : 32u;
	constexpr uint32_t warps = THREADS / lanes;
	const uint32_t lane = threadIdx.x & (lanes - 1u);
	const uint32_t warp = threadIdx.x >> 5u;
	#pragma unroll
	for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u)
	{
		const float other_score = __shfl_down_sync(0xffffffffu, best, offset);
		const uint32_t other_token = __shfl_down_sync(0xffffffffu, best_token, offset);
		if ( LmHeadBetter<TIE_BY_TOKEN>(other_score, other_token, best, best_token) )
		{
			best = other_score;
			best_token = other_token;
		}
	}
	if constexpr ( warps == 1u )
	{
		*out_score = best;
		*out_token = best_token;
		return;
	}
	if ( lane == 0u )
	{
		shared_score[warp] = best;
		shared_token[warp] = best_token;
	}
	__syncthreads();
	if ( warp == 0u )
	{
		best = lane < warps ? shared_score[lane] : -INFINITY;
		best_token = lane < warps ? shared_token[lane] : 0xffffffffu;
		constexpr uint32_t start = warps > 16u ? 16u : (warps > 8u ? 8u : (warps > 4u ? 4u : (warps > 2u ? 2u : 1u)));
		#pragma unroll
		for (uint32_t offset = start; offset > 0u; offset >>= 1u)
		{
			const float other_score = __shfl_down_sync(0xffffffffu, best, offset);
			const uint32_t other_token = __shfl_down_sync(0xffffffffu, best_token, offset);
			if ( LmHeadBetter<TIE_BY_TOKEN>(other_score, other_token, best, best_token) )
			{
				best = other_score;
				best_token = other_token;
			}
		}
	}
	*out_score = best;
	*out_token = best_token;
#else
	shared_score[threadIdx.x] = best;
	shared_token[threadIdx.x] = best_token;
	__syncthreads();
	for (uint32_t stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
		{
			const float other_score = shared_score[threadIdx.x + stride];
			const uint32_t other_token = shared_token[threadIdx.x + stride];
			if ( LmHeadBetter<TIE_BY_TOKEN>(other_score, other_token, shared_score[threadIdx.x], shared_token[threadIdx.x]) )
			{
				shared_score[threadIdx.x] = other_score;
				shared_token[threadIdx.x] = other_token;
			}
		}
		__syncthreads();
	}
	*out_score = shared_score[0];
	*out_token = shared_token[0];
#endif
}

template<uint32_t ROWS>
static __device__ __forceinline__ void LmHeadRowsReduce(float (*best_score)[LM_ROWS_TILE_N], uint32_t (*best_token)[LM_ROWS_TILE_N], uint32_t threads)
{
#if defined(__CUDA_ARCH__)
	const uint32_t warps = threads / 32u;
	const uint32_t warp = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	for (uint32_t row = warp; row < ROWS; row += warps)
	{
		float s0 = best_score[row][lane];
		uint32_t t0 = best_token[row][lane];
		const float s1 = best_score[row][lane + 32u];
		const uint32_t t1 = best_token[row][lane + 32u];
		if ( LmHeadBetter<true>(s1, t1, s0, t0) )
		{
			s0 = s1;
			t0 = t1;
		}
		#pragma unroll
		for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
		{
			const float os = __shfl_down_sync(0xffffffffu, s0, offset);
			const uint32_t ot = __shfl_down_sync(0xffffffffu, t0, offset);
			if ( LmHeadBetter<true>(os, ot, s0, t0) )
			{
				s0 = os;
				t0 = ot;
			}
		}
		if ( lane == 0u )
		{
			best_score[row][0] = s0;
			best_token[row][0] = t0;
		}
	}
	__syncthreads();
#else
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
#endif
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
		return(inverse_temperature == 0.0f ? logit : fmaf(logit,inverse_temperature,SparkMathGumbelNoise(sampling.rules[row].seed,sampling.positions[row],sampling.token_offset + token)));
	}
};

#define LM_HEAD_ROWS_VEC_K (LM_ROWS_TILE_K / 8u)
#define LM_HEAD_ROWS_PITCH (LM_ROWS_TILE_K / 2u + 1u)
#define LM_HEAD_ROWS_IN_PITCH (LM_ROWS_TILE_K / 2u)

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmHeadRowsFastFetch(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, uint32_t rows, uint32_t hidden, uint32_t vocabulary, uint32_t first_index, uint32_t first_row, uint32_t k0, uint4 *w_reg, uint4 *x_reg)
{
	constexpr uint32_t w_vecs = (LM_ROWS_TILE_N * LM_HEAD_ROWS_VEC_K + THREADS - 1u) / THREADS;
	constexpr uint32_t x_vecs = (ROWS * LM_HEAD_ROWS_VEC_K + THREADS - 1u) / THREADS;
	#pragma unroll
	for (uint32_t s = 0u; s < w_vecs; ++s)
	{
		const uint32_t slot = threadIdx.x + s * THREADS;
		const uint32_t n = slot / LM_HEAD_ROWS_VEC_K;
		const uint32_t v = slot % LM_HEAD_ROWS_VEC_K;
		const uint32_t idx = first_index + n;
		w_reg[s] = make_uint4(0u, 0u, 0u, 0u);
		if ( slot < LM_ROWS_TILE_N * LM_HEAD_ROWS_VEC_K && idx < vocabulary )
		{
			const uint32_t token = token_ids != 0 ? token_ids[idx] : idx;
			w_reg[s] = __ldcs((const uint4 *)(head_weight_bf16 + (uint64_t)token * hidden + k0 + v * 8u));
		}
	}
	#pragma unroll
	for (uint32_t s = 0u; s < x_vecs; ++s)
	{
		const uint32_t slot = threadIdx.x + s * THREADS;
		const uint32_t r = slot / LM_HEAD_ROWS_VEC_K;
		const uint32_t v = slot % LM_HEAD_ROWS_VEC_K;
		const uint32_t row = first_row + r;
		x_reg[s] = make_uint4(0u, 0u, 0u, 0u);
		if ( slot < ROWS * LM_HEAD_ROWS_VEC_K && row < rows )
			x_reg[s] = __ldg((const uint4 *)(normed_bf16 + (uint64_t)row * hidden + k0 + v * 8u));
	}
}

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmHeadRowsFastStore(uint32_t (*w_tile)[LM_HEAD_ROWS_PITCH], uint32_t (*x_tile)[LM_HEAD_ROWS_IN_PITCH], const uint4 *w_reg, const uint4 *x_reg)
{
	constexpr uint32_t w_vecs = (LM_ROWS_TILE_N * LM_HEAD_ROWS_VEC_K + THREADS - 1u) / THREADS;
	constexpr uint32_t x_vecs = (ROWS * LM_HEAD_ROWS_VEC_K + THREADS - 1u) / THREADS;
	#pragma unroll
	for (uint32_t s = 0u; s < w_vecs; ++s)
	{
		const uint32_t slot = threadIdx.x + s * THREADS;
		if ( slot < LM_ROWS_TILE_N * LM_HEAD_ROWS_VEC_K )
		{
			const uint32_t n = slot / LM_HEAD_ROWS_VEC_K;
			const uint32_t off = (slot % LM_HEAD_ROWS_VEC_K) * 4u;
			w_tile[n][off] = w_reg[s].x;
			w_tile[n][off + 1u] = w_reg[s].y;
			w_tile[n][off + 2u] = w_reg[s].z;
			w_tile[n][off + 3u] = w_reg[s].w;
		}
	}
	#pragma unroll
	for (uint32_t s = 0u; s < x_vecs; ++s)
	{
		const uint32_t slot = threadIdx.x + s * THREADS;
		if ( slot < ROWS * LM_HEAD_ROWS_VEC_K )
		{
			const uint32_t r = slot / LM_HEAD_ROWS_VEC_K;
			const uint32_t off = (slot % LM_HEAD_ROWS_VEC_K) * 4u;
			x_tile[r][off] = x_reg[s].x;
			x_tile[r][off + 1u] = x_reg[s].y;
			x_tile[r][off + 2u] = x_reg[s].z;
			x_tile[r][off + 3u] = x_reg[s].w;
		}
	}
}

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmHeadRowsFastDot(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, uint32_t (*w_stage)[LM_ROWS_TILE_N][LM_HEAD_ROWS_PITCH], uint32_t (*x_stage)[ROWS][LM_HEAD_ROWS_IN_PITCH], uint32_t rows, uint32_t hidden, uint32_t vocabulary, uint32_t first_index, uint32_t first_row, float *total)
{
	constexpr uint32_t per_thread = ROWS / (THREADS / LM_ROWS_TILE_N);
	constexpr uint32_t w_vecs = (LM_ROWS_TILE_N * LM_HEAD_ROWS_VEC_K + THREADS - 1u) / THREADS;
	constexpr uint32_t x_vecs = (ROWS * LM_HEAD_ROWS_VEC_K + THREADS - 1u) / THREADS;
	const uint32_t lane = threadIdx.x % LM_ROWS_TILE_N, group = threadIdx.x / LM_ROWS_TILE_N;
	const uint32_t chunks = hidden / LM_ROWS_TILE_K;
	uint4 w_reg[w_vecs], x_reg[x_vecs];
	#pragma unroll
	for (uint32_t j = 0u; j < per_thread; ++j)
		total[j] = 0.0f;
	if ( chunks == 0u )
		return;
	LmHeadRowsFastFetch<THREADS, ROWS>(normed_bf16, head_weight_bf16, token_ids, rows, hidden, vocabulary, first_index, first_row, 0u, w_reg, x_reg);
	LmHeadRowsFastStore<THREADS, ROWS>(w_stage[0], x_stage[0], w_reg, x_reg);
	__syncthreads();
	for (uint32_t chunk = 0u; chunk < chunks; ++chunk)
	{
		const uint32_t buf = chunk & 1u;
		if ( chunk + 1u < chunks )
			LmHeadRowsFastFetch<THREADS, ROWS>(normed_bf16, head_weight_bf16, token_ids, rows, hidden, vocabulary, first_index, first_row, (chunk + 1u) * LM_ROWS_TILE_K, w_reg, x_reg);
		#pragma unroll 8
		for (uint32_t pair = 0u; pair < LM_HEAD_ROWS_IN_PITCH; ++pair)
		{
			const uint32_t ww = w_stage[buf][lane][pair];
			const float w0 = __uint_as_float(ww << 16u);
			const float w1 = __uint_as_float(ww & 0xffff0000u);
			#pragma unroll
			for (uint32_t j = 0u; j < per_thread; ++j)
			{
				const uint32_t xx = x_stage[buf][group * per_thread + j][pair];
				total[j] = fmaf(__uint_as_float(xx << 16u), w0, total[j]);
				total[j] = fmaf(__uint_as_float(xx & 0xffff0000u), w1, total[j]);
			}
		}
		if ( chunk + 1u < chunks )
			LmHeadRowsFastStore<THREADS, ROWS>(w_stage[buf ^ 1u], x_stage[buf ^ 1u], w_reg, x_reg);
		__syncthreads();
	}
}

template<uint32_t THREADS, uint32_t TILE, uint32_t ROWS, class Score>
static __device__ __forceinline__ void LmHeadCandidateRowsBody(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary, const Score &score)
{
	constexpr uint32_t per_thread = ROWS / (THREADS / LM_ROWS_TILE_N);
	static_assert(TILE % LM_ROWS_TILE_N == 0u, "head tiles hold whole token groups");
	LmDependentWait();
	const uint32_t tile = blockIdx.x, first_row = blockIdx.y * ROWS, lane = threadIdx.x % LM_ROWS_TILE_N, group = threadIdx.x / LM_ROWS_TILE_N;
	__shared__ float best_score[ROWS][LM_ROWS_TILE_N];
	__shared__ uint32_t best_token[ROWS][LM_ROWS_TILE_N];
	uint32_t sub, j, index, best_id[per_thread];
	float total[per_thread], best[per_thread];
	for (j = 0u; j < per_thread; ++j)
	{
		best[j] = -INFINITY;
		best_id[j] = 0xffffffffu;
	}
	constexpr bool fast_smem_ok = (2u * LM_ROWS_TILE_N * LM_HEAD_ROWS_PITCH + 2u * ROWS * LM_HEAD_ROWS_IN_PITCH) * 4u <= 36864u && (THREADS % LM_HEAD_ROWS_VEC_K) == 0u;
	if constexpr ( fast_smem_ok )
	{
		if ( (hidden % LM_ROWS_TILE_K) == 0u && LmIsAligned16(normed_bf16) && LmIsAligned16(head_weight_bf16) )
		{
			__shared__ uint32_t w_stage[2][LM_ROWS_TILE_N][LM_HEAD_ROWS_PITCH];
			__shared__ uint32_t x_stage[2][ROWS][LM_HEAD_ROWS_IN_PITCH];
			for (sub = 0u; sub < TILE; sub += LM_ROWS_TILE_N)
			{
				index = tile * TILE + sub + lane;
				LmHeadRowsFastDot<THREADS, ROWS>(normed_bf16, head_weight_bf16, token_ids, w_stage, x_stage, rows, hidden, vocabulary, tile * TILE + sub, first_row, total);
				if ( index < vocabulary )
				{
					const uint32_t tok = token_ids != 0 ? token_ids[index] : index;
					#pragma unroll
					for (j = 0u; j < per_thread; ++j)
					{
						const float sc = score(total[j], first_row + group * per_thread + j, index);
						if ( sc > best[j] )
						{
							best[j] = sc;
							best_id[j] = tok;
						}
					}
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
			LmDependentRelease();
			return;
		}
	}
	__shared__ float weight_tile[LM_ROWS_TILE_N][LM_ROWS_TILE_K + 1u];
	__shared__ float input_tile[ROWS][LM_ROWS_TILE_K];
	const LmRowsTileOperand operand = {normed_bf16, head_weight_bf16, token_ids, hidden, hidden, rows, vocabulary, hidden};
	for (sub = 0u; sub < TILE; sub += LM_ROWS_TILE_N)
	{
		index = tile * TILE + sub + lane;
		LmRowsTileDot<THREADS, ROWS>(&operand, weight_tile, input_tile, tile * TILE + sub, first_row, total);
		if ( index < vocabulary )
		{
			const uint32_t tok = token_ids != 0 ? token_ids[index] : index;
			for (j = 0u; j < per_thread; ++j)
			{
				const float sc = score(total[j], first_row + group * per_thread + j, index);
				if ( sc > best[j] )
				{
					best[j] = sc;
					best_id[j] = tok;
				}
			}
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
	LmDependentRelease();
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

template<uint32_t THREADS, uint32_t TILE, uint32_t ROWS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadLogitsRowsKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, float *__restrict__ logits, uint32_t rows, uint32_t hidden, uint32_t vocabulary, uint32_t logit_stride)
{
	constexpr uint32_t per_thread = ROWS / (THREADS / LM_ROWS_TILE_N);
	static_assert(TILE % LM_ROWS_TILE_N == 0u, "head tiles hold whole token groups");
	LmDependentWait();
	const uint32_t tile = blockIdx.x, first_row = blockIdx.y * ROWS, lane = threadIdx.x % LM_ROWS_TILE_N, group = threadIdx.x / LM_ROWS_TILE_N;
	uint32_t sub, j, index, row;
	float total[per_thread];
	constexpr bool fast_smem_ok = (2u * LM_ROWS_TILE_N * LM_HEAD_ROWS_PITCH + 2u * ROWS * LM_HEAD_ROWS_IN_PITCH) * 4u <= 36864u && (THREADS % LM_HEAD_ROWS_VEC_K) == 0u;
	if constexpr ( fast_smem_ok )
	{
		if ( (hidden % LM_ROWS_TILE_K) == 0u && LmIsAligned16(normed_bf16) && LmIsAligned16(head_weight_bf16) )
		{
			__shared__ uint32_t w_stage[2][LM_ROWS_TILE_N][LM_HEAD_ROWS_PITCH];
			__shared__ uint32_t x_stage[2][ROWS][LM_HEAD_ROWS_IN_PITCH];
			for (sub = 0u; sub < TILE; sub += LM_ROWS_TILE_N)
			{
				index = tile * TILE + sub + lane;
				LmHeadRowsFastDot<THREADS, ROWS>(normed_bf16, head_weight_bf16, 0, w_stage, x_stage, rows, hidden, vocabulary, tile * TILE + sub, first_row, total);
				if ( index < vocabulary )
				{
					#pragma unroll
					for (j = 0u; j < per_thread; ++j)
					{
						row = first_row + group * per_thread + j;
						if ( row < rows )
							logits[(uint64_t)row * logit_stride + index] = total[j];
					}
				}
			}
			LmDependentRelease();
			return;
		}
	}
	__shared__ float weight_tile[LM_ROWS_TILE_N][LM_ROWS_TILE_K + 1u];
	__shared__ float input_tile[ROWS][LM_ROWS_TILE_K];
	const LmRowsTileOperand operand = {normed_bf16, head_weight_bf16, 0, hidden, hidden, rows, vocabulary, hidden};
	for (sub = 0u; sub < TILE; sub += LM_ROWS_TILE_N)
	{
		index = tile * TILE + sub + lane;
		LmRowsTileDot<THREADS, ROWS>(&operand, weight_tile, input_tile, tile * TILE + sub, first_row, total);
		for (j = 0u; j < per_thread; ++j)
		{
			row = first_row + group * per_thread + j;
			if ( index < vocabulary && row < rows )
				logits[(uint64_t)row * logit_stride + index] = total[j];
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t TILE, class Score>
static __device__ __forceinline__ void LmHeadCandidateScalarBody(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t hidden, uint32_t vocabulary, const Score &score)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.y,tile = blockIdx.x,index,element;
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
	float final_score;
	uint32_t final_token;
	LmHeadBlockArgMax<THREADS, false>(best, best_token, shared_score, shared_token, &final_score, &final_token);
	if ( threadIdx.x == 0u )
	{
		candidate_score[(row * gridDim.x) + tile] = final_score;
		candidate_token[(row * gridDim.x) + tile] = final_token;
	}
}

#define LM_HEAD_STAGE_ELEMENTS 64u
#define LM_HEAD_STAGE_PITCH (LM_HEAD_STAGE_ELEMENTS / 2u + 1u)
#define LM_HEAD_STAGE_VECTORS (LM_HEAD_STAGE_ELEMENTS / 8u)
#define LM_HEAD_STAGE_SUB_MAX 128u

template<uint32_t THREADS, uint32_t TILE>
struct LmHeadStage
{
	static constexpr uint32_t kSubTile = TILE <= LM_HEAD_STAGE_SUB_MAX ? TILE : LM_HEAD_STAGE_SUB_MAX;
	static constexpr uint32_t kVectors = (kSubTile * LM_HEAD_STAGE_VECTORS + THREADS - 1u) / THREADS;
	static constexpr bool kSupported = kSubTile <= THREADS && (TILE % kSubTile) == 0u && (THREADS % LM_HEAD_STAGE_VECTORS) == 0u && 2u * kSubTile * LM_HEAD_STAGE_PITCH * 4u <= 40960u;
};

template<uint32_t THREADS, uint32_t SUB_TILE>
static __device__ __forceinline__ void LmHeadStageLoadBase(const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, uint32_t base_index, uint32_t hidden, uint32_t vocabulary, uint32_t first, uint4 *staged)
{
	constexpr uint32_t kVectors = (SUB_TILE * LM_HEAD_STAGE_VECTORS + THREADS - 1u) / THREADS;
	uint32_t vector, slot, token, index;
	#pragma unroll
	for (vector = 0u; vector < kVectors; vector++)
	{
		slot = threadIdx.x + vector * THREADS;
		index = base_index + slot / LM_HEAD_STAGE_VECTORS;
		staged[vector] = make_uint4(0u, 0u, 0u, 0u);
		if ( slot < SUB_TILE * LM_HEAD_STAGE_VECTORS && index < vocabulary )
		{
			token = token_ids != 0 ? token_ids[index] : index;
			staged[vector] = __ldcs((const uint4 *)(head_weight_bf16 + (uint64_t)token * hidden + first + (slot % LM_HEAD_STAGE_VECTORS) * 8u));
		}
	}
}

template<uint32_t THREADS, uint32_t SUB_TILE>
static __device__ __forceinline__ void LmHeadStageStoreBase(uint32_t *stage, const uint4 *staged)
{
	constexpr uint32_t kVectors = (SUB_TILE * LM_HEAD_STAGE_VECTORS + THREADS - 1u) / THREADS;
	uint32_t vector, slot, word;
	#pragma unroll
	for (vector = 0u; vector < kVectors; vector++)
	{
		slot = threadIdx.x + vector * THREADS;
		if ( slot >= SUB_TILE * LM_HEAD_STAGE_VECTORS )
			continue;
		word = (slot / LM_HEAD_STAGE_VECTORS) * LM_HEAD_STAGE_PITCH + (slot % LM_HEAD_STAGE_VECTORS) * 4u;
		stage[word] = staged[vector].x;
		stage[word + 1u] = staged[vector].y;
		stage[word + 2u] = staged[vector].z;
		stage[word + 3u] = staged[vector].w;
	}
}

template<uint32_t THREADS, uint32_t TILE>
static __device__ __forceinline__ void LmHeadStageLoad(const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, uint32_t tile, uint32_t hidden, uint32_t vocabulary, uint32_t first, uint4 *staged)
{
	LmHeadStageLoadBase<THREADS, LmHeadStage<THREADS, TILE>::kSubTile>(head_weight_bf16, token_ids, tile * TILE, hidden, vocabulary, first, staged);
}

template<uint32_t THREADS, uint32_t TILE>
static __device__ __forceinline__ void LmHeadStageStore(uint32_t *stage, const uint4 *staged)
{
	LmHeadStageStoreBase<THREADS, LmHeadStage<THREADS, TILE>::kSubTile>(stage, staged);
}

template<uint32_t THREADS, uint32_t TILE, class Score>
static __device__ __forceinline__ void LmHeadCandidateBody(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t hidden, uint32_t vocabulary, const Score &score)
{
	LmDependentWait();
	if constexpr ( !LmHeadStage<THREADS, TILE>::kSupported )
	{
		LmHeadCandidateScalarBody<THREADS, TILE>(normed_bf16, head_weight_bf16, token_ids, candidate_score, candidate_token, hidden, vocabulary, score);
		LmDependentRelease();
	}
	else
	{
		constexpr uint32_t SUB_TILE = LmHeadStage<THREADS, TILE>::kSubTile;
		__shared__ uint32_t stage[2][SUB_TILE * LM_HEAD_STAGE_PITCH];
		__shared__ float input[2][LM_HEAD_STAGE_ELEMENTS];
		__shared__ float shared_score[THREADS];
		__shared__ uint32_t shared_token[THREADS];
		const uint32_t row = blockIdx.y, tile = blockIdx.x, chunks = hidden / LM_HEAD_STAGE_ELEMENTS;
		const uint16_t *activation = normed_bf16 + (uint64_t)row * hidden;
		uint4 staged[LmHeadStage<THREADS, TILE>::kVectors];
		uint32_t chunk, element, word, buffer, best_token = 0u;
		float best = -INFINITY, value;
		if ( (hidden % LM_HEAD_STAGE_ELEMENTS) != 0u || ((uintptr_t)head_weight_bf16 % 16u) != 0u )
		{
			LmHeadCandidateScalarBody<THREADS, TILE>(normed_bf16, head_weight_bf16, token_ids, candidate_score, candidate_token, hidden, vocabulary, score);
			LmDependentRelease();
			return;
		}
		for (uint32_t sub = 0u; sub < TILE; sub += SUB_TILE)
		{
			const uint32_t base_index = tile * TILE + sub;
			if ( base_index >= vocabulary )
				break;
			const uint32_t index = base_index + threadIdx.x;
			float total = 0.0f;
			LmHeadStageLoadBase<THREADS, SUB_TILE>(head_weight_bf16, token_ids, base_index, hidden, vocabulary, 0u, staged);
			LmHeadStageStoreBase<THREADS, SUB_TILE>(stage[0], staged);
			if ( threadIdx.x < LM_HEAD_STAGE_ELEMENTS )
				input[0][threadIdx.x] = LmBf16ToFloat(__ldg(activation + threadIdx.x));
			__syncthreads();
			for (chunk = 0u; chunk < chunks; chunk++)
			{
				buffer = chunk & 1u;
				if ( chunk + 1u < chunks )
					LmHeadStageLoadBase<THREADS, SUB_TILE>(head_weight_bf16, token_ids, base_index, hidden, vocabulary, (chunk + 1u) * LM_HEAD_STAGE_ELEMENTS, staged);
				if ( threadIdx.x < SUB_TILE )
				{
					#pragma unroll 8
					for (element = 0u; element < LM_HEAD_STAGE_ELEMENTS; element += 2u)
					{
						word = stage[buffer][threadIdx.x * LM_HEAD_STAGE_PITCH + element / 2u];
						total = fmaf(input[buffer][element], __uint_as_float(word << 16u), total);
						total = fmaf(input[buffer][element + 1u], __uint_as_float(word & 0xffff0000u), total);
					}
				}
				if ( chunk + 1u < chunks )
				{
					LmHeadStageStoreBase<THREADS, SUB_TILE>(stage[buffer ^ 1u], staged);
					if ( threadIdx.x < LM_HEAD_STAGE_ELEMENTS )
						input[buffer ^ 1u][threadIdx.x] = LmBf16ToFloat(__ldg(activation + (chunk + 1u) * LM_HEAD_STAGE_ELEMENTS + threadIdx.x));
				}
				__syncthreads();
			}
			if ( threadIdx.x < SUB_TILE && index < vocabulary )
			{
				value = score(total, row, index);
				if ( value > best )
				{
					best = value;
					best_token = token_ids != 0 ? token_ids[index] : index;
				}
			}
		}
		float final_score;
		uint32_t final_token;
		LmHeadBlockArgMax<THREADS, false>(best, best_token, shared_score, shared_token, &final_score, &final_token);
		if ( threadIdx.x == 0u )
		{
			candidate_score[(row * gridDim.x) + tile] = final_score;
			candidate_token[(row * gridDim.x) + tile] = final_token;
		}
		LmDependentRelease();
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
	uint32_t row = blockIdx.x, index;
	float best = -INFINITY;
	uint32_t best_token = 0xffffffffu;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
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
	float final_score;
	uint32_t final_token;
	LmHeadBlockArgMax<THREADS, true>(best, best_token, shared_score, shared_token, &final_score, &final_token);
	if ( threadIdx.x == 0u )
	{
		token_out[row] = final_token;
		if ( score_out != 0 )
			score_out[row] = final_score;
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSoftmaxKernel(float *__restrict__ logits, uint32_t rows, uint32_t vocabulary, float temperature)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint64_t base = (uint64_t)blockIdx.x * vocabulary;
	uint32_t index;
	float local = -INFINITY, maximum, total = 0.0f, inverse_temperature, denom;
	LmDependentWait();
	if ( blockIdx.x >= rows )
	{
		LmDependentRelease();
		return;
	}
	inverse_temperature = 1.0f / fmaxf(temperature,1.0e-4f);
	float *row_ptr = logits + base;
	if ( ((vocabulary & 3u) == 0u) && LmIsAligned16(row_ptr) )
	{
		const uint32_t vecs = vocabulary >> 2u;
		float4 *v4 = (float4 *)row_ptr;
		for (index = threadIdx.x; index < vecs; index += THREADS)
		{
			const float4 v = v4[index];
			local = fmaxf(local, v.x * inverse_temperature);
			local = fmaxf(local, v.y * inverse_temperature);
			local = fmaxf(local, v.z * inverse_temperature);
			local = fmaxf(local, v.w * inverse_temperature);
		}
		maximum = LmBlockMax<THREADS>(local, reduction);
		for (index = threadIdx.x; index + 3u * THREADS < vocabulary; index += 4u * THREADS)
		{
			const float v0 = row_ptr[index];
			const float v1 = row_ptr[index + THREADS];
			const float v2 = row_ptr[index + 2u * THREADS];
			const float v3 = row_ptr[index + 3u * THREADS];
			const float e0 = __expf((v0 * inverse_temperature) - maximum);
			const float e1 = __expf((v1 * inverse_temperature) - maximum);
			const float e2 = __expf((v2 * inverse_temperature) - maximum);
			const float e3 = __expf((v3 * inverse_temperature) - maximum);
			row_ptr[index] = e0;
			row_ptr[index + THREADS] = e1;
			row_ptr[index + 2u * THREADS] = e2;
			row_ptr[index + 3u * THREADS] = e3;
			total += e0;
			total += e1;
			total += e2;
			total += e3;
		}
		for (; index < vocabulary; index += THREADS)
		{
			const float value = __expf((row_ptr[index] * inverse_temperature) - maximum);
			row_ptr[index] = value;
			total += value;
		}
		total = LmBlockSum<THREADS>(total, reduction);
		denom = fmaxf(total, 1.0e-20f);
		for (index = threadIdx.x; index < vecs; index += THREADS)
		{
			const float4 v = v4[index];
			v4[index] = {v.x / denom, v.y / denom, v.z / denom, v.w / denom};
		}
		LmDependentRelease();
		return;
	}
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		local = fmaxf(local,row_ptr[index] * inverse_temperature);
	maximum = LmBlockMax<THREADS>(local,reduction);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
	{
		float value = __expf((row_ptr[index] * inverse_temperature) - maximum);
		row_ptr[index] = value;
		total += value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	denom = fmaxf(total,1.0e-20f);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		row_ptr[index] /= denom;
	LmDependentRelease();
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
	constexpr uint32_t SUB_TILE = LmHeadStage<THREADS, TILE>::kSubTile;
	constexpr uint32_t BASE_LOCAL = (TILE + THREADS - 1u) / THREADS;
	constexpr uint32_t STAGE_LOCAL = LmHeadStage<THREADS, TILE>::kSupported ? (TILE / SUB_TILE) : 1u;
	constexpr uint32_t LM_HEAD_TOPK_LOCAL = STAGE_LOCAL > BASE_LOCAL ? STAGE_LOCAL : BASE_LOCAL;
	float local_score[LM_HEAD_TOPK_LOCAL];
	uint32_t local_token[LM_HEAD_TOPK_LOCAL];
	uint32_t row = blockIdx.y, tile = blockIdx.x, index, element, round, pick;
	uint32_t tile_begin = tile * TILE, tile_end = tile_begin + TILE, count = 0u;
	LmDependentWait();
	if ( tile_end > vocabulary )
		tile_end = vocabulary;
	if constexpr ( LmHeadStage<THREADS, TILE>::kSupported )
	{
		if ( (hidden % LM_HEAD_STAGE_ELEMENTS) == 0u && LmIsAligned16(head_weight_bf16) )
		{
			__shared__ uint32_t stage[2][SUB_TILE * LM_HEAD_STAGE_PITCH];
			__shared__ float input[2][LM_HEAD_STAGE_ELEMENTS];
			const uint32_t chunks = hidden / LM_HEAD_STAGE_ELEMENTS;
			const uint16_t *activation = normed_bf16 + (uint64_t)row * hidden;
			uint4 staged[LmHeadStage<THREADS, TILE>::kVectors];
			for (uint32_t sub = 0u; sub < TILE; sub += SUB_TILE)
			{
				const uint32_t base_index = tile_begin + sub;
				if ( base_index >= tile_end )
					break;
				float total = 0.0f;
				LmHeadStageLoadBase<THREADS, SUB_TILE>(head_weight_bf16, token_ids, base_index, hidden, vocabulary, 0u, staged);
				LmHeadStageStoreBase<THREADS, SUB_TILE>(stage[0], staged);
				if ( threadIdx.x < LM_HEAD_STAGE_ELEMENTS )
					input[0][threadIdx.x] = LmBf16ToFloat(__ldg(activation + threadIdx.x));
				__syncthreads();
				for (uint32_t chunk = 0u; chunk < chunks; ++chunk)
				{
					const uint32_t buffer = chunk & 1u;
					if ( chunk + 1u < chunks )
						LmHeadStageLoadBase<THREADS, SUB_TILE>(head_weight_bf16, token_ids, base_index, hidden, vocabulary, (chunk + 1u) * LM_HEAD_STAGE_ELEMENTS, staged);
					if ( threadIdx.x < SUB_TILE )
					{
						#pragma unroll 8
						for (element = 0u; element < LM_HEAD_STAGE_ELEMENTS; element += 2u)
						{
							const uint32_t word = stage[buffer][threadIdx.x * LM_HEAD_STAGE_PITCH + element / 2u];
							total = fmaf(input[buffer][element], __uint_as_float(word << 16u), total);
							total = fmaf(input[buffer][element + 1u], __uint_as_float(word & 0xffff0000u), total);
						}
					}
					if ( chunk + 1u < chunks )
					{
						LmHeadStageStoreBase<THREADS, SUB_TILE>(stage[buffer ^ 1u], staged);
						if ( threadIdx.x < LM_HEAD_STAGE_ELEMENTS )
							input[buffer ^ 1u][threadIdx.x] = LmBf16ToFloat(__ldg(activation + (chunk + 1u) * LM_HEAD_STAGE_ELEMENTS + threadIdx.x));
					}
					__syncthreads();
				}
				if ( threadIdx.x < SUB_TILE && base_index + threadIdx.x < tile_end )
				{
					const uint32_t idx = base_index + threadIdx.x;
					local_score[count] = total;
					local_token[count] = token_ids != 0 ? token_ids[idx] : idx;
					++count;
				}
			}
			goto select_topk;
		}
	}
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
select_topk:
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
		float final_score;
		uint32_t final_token;
		LmHeadBlockArgMax<THREADS, true>(best, best_token, shared_score, shared_token, &final_score, &final_token);
		if ( threadIdx.x == 0u )
		{
			candidate_score[((row * gridDim.x) + tile) * K + round] = final_score;
			candidate_token[((row * gridDim.x) + tile) * K + round] = final_token;
			taken[round] = final_token;
		}
		__syncthreads();
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t K>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadTopkCommitKernel(const float *__restrict__ candidate_score, const uint32_t *__restrict__ candidate_token, uint32_t tiles, uint32_t *__restrict__ token_out, float *__restrict__ score_out, uint32_t rows)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	__shared__ uint32_t taken[K];
	uint32_t row = blockIdx.x, candidates = tiles * K, index, round, prior;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
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
		float final_score;
		uint32_t final_token;
		LmHeadBlockArgMax<THREADS, true>(best, best_token, shared_score, shared_token, &final_score, &final_token);
		if ( threadIdx.x == 0u )
		{
			token_out[(row * K) + round] = final_token;
			if ( score_out != 0 )
				score_out[(row * K) + round] = final_score;
			taken[round] = final_token;
		}
		__syncthreads();
	}
	LmDependentRelease();
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
	SparkGpuStream stream)
{
	uint32_t tiles;

	if ( normed_bf16 == 0 || head_weight_bf16 == 0 ||
		candidate_score == 0 || candidate_token == 0 || token_out == 0 ||
		rows == 0u || hidden == 0u || vocabulary == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	tiles = (vocabulary + TILE - 1u) / TILE;
	LM_LAUNCH_DEPENDENT((LmHeadTopkCandidateKernel<THREADS,TILE,K>), dim3(tiles,rows), THREADS, 0, stream,
		normed_bf16,head_weight_bf16,token_ids,candidate_score,candidate_token,rows,hidden,vocabulary);
	LM_LAUNCH_DEPENDENT((LmHeadTopkCommitKernel<THREADS,K>), rows, THREADS, 0, stream,
		candidate_score,candidate_token,tiles,token_out,score_out,rows);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS
		? LM_LAUNCH_OK
		: LM_LAUNCH_ERR_LAUNCH);
}
