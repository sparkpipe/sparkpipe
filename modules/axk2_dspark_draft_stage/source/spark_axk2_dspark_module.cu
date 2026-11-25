#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_draft_tree_pack.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "runtime/launch.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AXK2_HIDDEN 7168u
#define AXK2_TAPS 3u
#define AXK2_TAP_WIDTH (AXK2_TAPS * AXK2_HIDDEN)
#define AXK2_LAYERS 3u
#define AXK2_HEADS 64u
#define AXK2_KV_HEADS 64u
#define AXK2_HEAD_DIM 112u
#define AXK2_HALF (AXK2_HEAD_DIM / 2u)
#define AXK2_HEAD_THREADS 128u
#define AXK2_Q_WIDTH (AXK2_HEADS * AXK2_HEAD_DIM)
#define AXK2_KV_WIDTH (AXK2_KV_HEADS * AXK2_HEAD_DIM)
#define AXK2_MLP 18432u
#define AXK2_BLOCK 5u
#define AXK2_SLIDING 128u
#define AXK2_WINDOW (AXK2_SLIDING - AXK2_BLOCK)
#define AXK2_COLUMNS (AXK2_WINDOW + AXK2_BLOCK)
#define AXK2_VOCAB 163840u
#define AXK2_DRAFT_VOCAB 32768u
#define AXK2_MARKOV 256u
#define AXK2_CONFIDENCE (AXK2_HIDDEN + AXK2_MARKOV)
#define AXK2_CANDIDATES 2048u
#define AXK2_SAMPLED 32u
#define AXK2_BRANCHES 2u
#define AXK2_BRANCH_LEN 4u
#define AXK2_INGEST_ROWS 256u
#define AXK2_THREADS 256u
#define AXK2_WARP 32u
#define AXK2_WARPS (AXK2_THREADS / AXK2_WARP)
#define AXK2_PICK_THREADS 1024u
#define AXK2_TOPK_THREADS 512u
#define AXK2_ATTN_TILE 32u
#define AXK2_ATTN_THREADS (AXK2_BLOCK * AXK2_WARP)
#define AXK2_ATTN_VEC4S (AXK2_HEAD_DIM / 8u)
#define AXK2_SEQUENCES_MAX 256u
#define AXK2_LAYER_KV ((uint64_t)AXK2_KV_HEADS * AXK2_COLUMNS * AXK2_HEAD_DIM)
#define AXK2_SEQUENCE_ELEMENTS (2u * AXK2_LAYERS * AXK2_LAYER_KV)
#define AXK2_SEQUENCE_BYTES (AXK2_SEQUENCE_ELEMENTS * sizeof(uint16_t))
#define AXK2_PACK_MAGIC 0x4B505244u
#define AXK2_PACK_VERSION 1u
#define AXK2_PACK_ALIGNMENT 128u
#define AXK2_PI 3.14159265358979323846
#define AXK2_INVERSE_PI 0.31830988618379067154

typedef struct Axk2Layer
{
	const uint16_t *input_norm;
	const uint16_t *post_norm;
	const uint16_t *q;
	const uint16_t *k;
	const uint16_t *v;
	const uint16_t *o;
	const uint16_t *q_norm;
	const uint16_t *k_norm;
	const uint16_t *gate;
	const uint16_t *up;
	const uint16_t *down;
} Axk2Layer;

typedef struct Axk2Sequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t last_used;
	uint32_t live;
	uint32_t retired;
	int64_t positions[AXK2_WINDOW];
	uint32_t tokens[AXK2_WINDOW];
	uint16_t *cache;
} Axk2Sequence;

typedef struct Axk2Draft
{
	uint64_t seed;
	uint32_t positions[AXK2_BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint32_t start;
	uint32_t block_at;
	uint32_t masked;
	uint8_t valid[AXK2_WINDOW];
} Axk2Draft;

typedef struct Axk2Picks
{
	uint32_t tokens[AXK2_BLOCK];
	float scores[AXK2_BLOCK];
	uint32_t alt_tokens[AXK2_BLOCK];
	float alt_scores[AXK2_BLOCK];
	uint32_t branch_fork[AXK2_BRANCHES];
	uint32_t branch_len[AXK2_BRANCHES];
	uint32_t branch_tokens[AXK2_BRANCHES][AXK2_BRANCH_LEN];
	float branch_scores[AXK2_BRANCHES][AXK2_BRANCH_LEN];
} Axk2Picks;

typedef struct Axk2Forward
{
	Axk2Draft *host_draft;
	Axk2Draft *draft;
	Axk2Picks *host_picks;
	Axk2Picks *picks;
	uint16_t *hidden;
	uint16_t *x;
	uint16_t *q;
	uint16_t *q_roped;
	uint16_t *k;
	uint16_t *v;
	uint16_t *attended;
	uint16_t *projected;
	uint16_t *residual;
	uint16_t *gate;
	uint16_t *up;
	uint16_t *act;
	uint16_t *mlp;
	uint16_t *y;
	uint16_t *logits;
	float *logits_f32;
	uint32_t *candidates;
	float *markov_row;
} Axk2Forward;

typedef struct Axk2State
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	uint32_t mask_token;
	float epsilon;
	float rope_gain;
	float softmax_scale;
	float *rope_inverse;
	const uint16_t *fc;
	const uint16_t *hidden_norm;
	const uint16_t *norm;
	const uint16_t *embed;
	const uint16_t *head;
	const int64_t *d2t;
	const uint16_t *markov_previous;
	const uint16_t *markov_next;
	const uint16_t *confidence_weight;
	const uint16_t *confidence_bias;
	Axk2Layer layers[AXK2_LAYERS];
	Axk2Forward forward;
	uint16_t *host_taps;
	uint32_t *host_positions;
	uint16_t *taps;
	uint32_t *positions;
	uint16_t *features_raw;
	uint16_t *features;
	uint16_t *keys_raw;
	uint16_t *values_raw;
	uint16_t *caches;
	uint32_t capacity;
	uint64_t clock;
	Axk2Sequence *sequences;
	uint64_t frames;
	uint64_t rows;
	uint64_t gaps;
} Axk2State;

static __device__ __forceinline__ void Axk2Rotation(uint32_t position, float inverse, float gain, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * AXK2_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	sincospif(reduced, sine, cosine);
	*cosine *= gain;
	*sine *= gain;
}

static __device__ __forceinline__ uint32_t Axk2DraftToTarget(const int64_t *__restrict__ d2t, uint32_t draft_id)
{
	const int64_t mapped = (int64_t)draft_id + d2t[draft_id];
	return mapped >= 0 && mapped < (int64_t)AXK2_VOCAB ? (uint32_t)mapped : 0u;
}

static __device__ __forceinline__ float Axk2WarpSum(float value)
{
	for ( uint32_t mask = AXK2_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ float Axk2WarpMax(float value)
{
	for ( uint32_t mask = AXK2_WARP / 2u; mask > 0u; mask >>= 1u )
		value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
	return value;
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2RopeTableKernel(float *__restrict__ inverse, double theta, double factor, double original, double beta_fast, double beta_slow)
{
	if ( threadIdx.x < AXK2_HALF )
	{
		const double dimension = (double)AXK2_HEAD_DIM;
		double low = floor(dimension * log(original / (beta_fast * 2.0 * AXK2_PI)) / (2.0 * log(theta)));
		double high = ceil(dimension * log(original / (beta_slow * 2.0 * AXK2_PI)) / (2.0 * log(theta)));
		const double extrapolated = 1.0 / pow(theta, (2.0 * (double)threadIdx.x) / dimension);
		double ramp;
		low = low < 0.0 ? 0.0 : low;
		high = high > dimension - 1.0 ? dimension - 1.0 : high;
		if ( low == high )
			high += 0.001;
		ramp = ((double)threadIdx.x - low) / (high - low);
		ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
		inverse[threadIdx.x] = (float)(extrapolated / factor * ramp + extrapolated * (1.0 - ramp));
	}
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2RmsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[AXK2_WARPS];
	const uint64_t row = blockIdx.x;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += AXK2_THREADS )
	{
		const float value = LmBf16ToFloat(input[row * dimension + index]);
		total += value * value;
	}
	total = LmBlockSum<AXK2_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += AXK2_THREADS )
	{
		const float normalized = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(input[row * dimension + index]) * scale));
		output[row * dimension + index] = LmFloatToBf16(normalized * LmBf16ToFloat(weight[index]));
	}
}

__global__ __launch_bounds__(AXK2_HEAD_THREADS, 1)
void Axk2HeadNormRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, float rope_gain, uint16_t *__restrict__ output, uint16_t *__restrict__ cache_keys,
	uint32_t block_at, uint32_t use_ring)
{
	__shared__ float reduction[AXK2_HEAD_THREADS / AXK2_WARP];
	__shared__ float normed[AXK2_HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = lane < AXK2_HEAD_DIM ? LmBf16ToFloat(input[((uint64_t)row * heads + head) * AXK2_HEAD_DIM + lane]) : 0.0f;
	const float scale = rsqrtf(LmBlockSum<AXK2_HEAD_THREADS>(value * value, reduction) / (float)AXK2_HEAD_DIM + epsilon);
	if ( lane < AXK2_HEAD_DIM )
		normed[lane] = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value * scale)) * LmBf16ToFloat(weight[lane])));
	__syncthreads();
	if ( lane < AXK2_HEAD_DIM )
	{
		float cosine, sine, rotated;
		Axk2Rotation(position, inverse[lane % AXK2_HALF], rope_gain, &cosine, &sine);
		rotated = lane < AXK2_HALF ? normed[lane] * cosine - normed[lane + AXK2_HALF] * sine : normed[lane] * cosine + normed[lane - AXK2_HALF] * sine;
		if ( cache_keys != 0 )
		{
			const uint32_t column = use_ring != 0u ? position % AXK2_WINDOW : block_at + row;
			cache_keys[((uint64_t)head * AXK2_COLUMNS + column) * AXK2_HEAD_DIM + lane] = LmFloatToBf16(rotated);
		}
		else
			output[((uint64_t)row * heads + head) * AXK2_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	}
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2StoreValuesKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions, uint32_t block_at, uint16_t *__restrict__ cache)
{
	const uint32_t row = blockIdx.x, column = positions != 0 ? positions[row] % AXK2_WINDOW : block_at + row;
	for ( uint32_t index = threadIdx.x; index < AXK2_KV_WIDTH; index += AXK2_THREADS )
		cache[((uint64_t)(index / AXK2_HEAD_DIM) * AXK2_COLUMNS + column) * AXK2_HEAD_DIM + index % AXK2_HEAD_DIM] = values[(uint64_t)row * AXK2_KV_WIDTH + index];
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2EmbedKernel(const uint16_t *__restrict__ embed, const Axk2Draft *__restrict__ draft, uint16_t *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	for ( uint32_t index = threadIdx.x; index < AXK2_HIDDEN; index += AXK2_THREADS )
		hidden[(uint64_t)row * AXK2_HIDDEN + index] = embed[(uint64_t)token * AXK2_HIDDEN + index];
}

__global__ __launch_bounds__(AXK2_ATTN_THREADS, 1)
void Axk2AttentionKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ keys, const uint16_t *__restrict__ values,
	const Axk2Draft *__restrict__ draft, float scale, uint16_t *__restrict__ output)
{
	__shared__ uint32_t s_k[AXK2_ATTN_TILE][AXK2_HALF + 1u];
	__shared__ uint32_t s_v[AXK2_ATTN_TILE][AXK2_HALF + 1u];
	__shared__ float s_scores[AXK2_BLOCK][AXK2_ATTN_TILE];
	__shared__ uint8_t s_vis[AXK2_ATTN_TILE];
	const uint32_t head = blockIdx.x;
	const uint32_t warp = threadIdx.x / AXK2_WARP, lane = threadIdx.x % AXK2_WARP;
	const uint32_t start = draft->start, block_at = draft->block_at, masked = draft->masked;
	const uint32_t tiles = start < block_at ? (block_at - start + AXK2_ATTN_TILE - 1u) / AXK2_ATTN_TILE : 0u;
	const uint16_t *head_keys = keys + (uint64_t)head * AXK2_COLUMNS * AXK2_HEAD_DIM;
	const uint16_t *head_values = values + (uint64_t)head * AXK2_COLUMNS * AXK2_HEAD_DIM;
	const uint16_t *q_row = q + ((uint64_t)warp * AXK2_HEADS + head) * AXK2_HEAD_DIM;
	const uint32_t qp0 = *(const uint32_t *)(q_row + 2u * lane);
	const float q0 = LmBf16ToFloat((uint16_t)(qp0 & 0xffffu));
	const float q1 = LmBf16ToFloat((uint16_t)(qp0 >> 16u));
	float q2 = 0.0f, q3 = 0.0f;
	float running_max = -INFINITY, running_sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
	uint16_t *out_row = output + (uint64_t)warp * AXK2_Q_WIDTH + (uint64_t)head * AXK2_HEAD_DIM;
	if ( lane < AXK2_HALF - AXK2_WARP )
	{
		const uint32_t qp1 = *(const uint32_t *)(q_row + 2u * (AXK2_WARP + lane));
		q2 = LmBf16ToFloat((uint16_t)(qp1 & 0xffffu));
		q3 = LmBf16ToFloat((uint16_t)(qp1 >> 16u));
	}
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t tile_first = start + tile * AXK2_ATTN_TILE;
		const uint32_t tile_keys = block_at - tile_first < AXK2_ATTN_TILE ? block_at - tile_first : AXK2_ATTN_TILE;
		for ( uint32_t idx = threadIdx.x; idx < tile_keys * AXK2_ATTN_VEC4S; idx += AXK2_ATTN_THREADS )
		{
			const uint32_t load_key = idx / AXK2_ATTN_VEC4S, load_chunk = idx % AXK2_ATTN_VEC4S;
			const uint32_t col = tile_first + load_key;
			const uint4 kv_k = *(const uint4 *)(head_keys + (uint64_t)col * AXK2_HEAD_DIM + load_chunk * 8u);
			const uint4 kv_v = *(const uint4 *)(head_values + (uint64_t)col * AXK2_HEAD_DIM + load_chunk * 8u);
			s_k[load_key][load_chunk * 4u + 0u] = kv_k.x;
			s_k[load_key][load_chunk * 4u + 1u] = kv_k.y;
			s_k[load_key][load_chunk * 4u + 2u] = kv_k.z;
			s_k[load_key][load_chunk * 4u + 3u] = kv_k.w;
			s_v[load_key][load_chunk * 4u + 0u] = kv_v.x;
			s_v[load_key][load_chunk * 4u + 1u] = kv_v.y;
			s_v[load_key][load_chunk * 4u + 2u] = kv_v.z;
			s_v[load_key][load_chunk * 4u + 3u] = kv_v.w;
		}
		if ( threadIdx.x < AXK2_ATTN_TILE )
			s_vis[threadIdx.x] = threadIdx.x < tile_keys && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
		__syncthreads();
		for ( uint32_t t = 0u; t < tile_keys; t++ )
		{
			const uint32_t kp0 = s_k[t][lane];
			float dot = fmaf(q0, LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1 * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
			if ( lane < AXK2_HALF - AXK2_WARP )
			{
				const uint32_t kp1 = s_k[t][AXK2_WARP + lane];
				dot = fmaf(q2, LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), fmaf(q3, LmBf16ToFloat((uint16_t)(kp1 >> 16u)), dot));
			}
			dot = Axk2WarpSum(dot);
			if ( lane == 0u )
				s_scores[warp][t] = s_vis[t] != 0u ? dot * scale : -INFINITY;
		}
		__syncwarp();
		{
			const float sc = lane < tile_keys ? s_scores[warp][lane] : -INFINITY;
			const float tile_max = Axk2WarpMax(sc);
			const float next_max = fmaxf(running_max, tile_max);
			const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
			const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
			const float tile_sum = Axk2WarpSum(weight);
			running_sum = running_sum * rescale + tile_sum;
			running_max = next_max;
			acc0 *= rescale;
			acc1 *= rescale;
			acc2 *= rescale;
			acc3 *= rescale;
			s_scores[warp][lane] = weight;
		}
		__syncwarp();
		for ( uint32_t t = 0u; t < tile_keys; t++ )
		{
			const float wt = s_scores[warp][t];
			const uint32_t vp0 = s_v[t][lane];
			acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0);
			acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1);
			if ( lane < AXK2_HALF - AXK2_WARP )
			{
				const uint32_t vp1 = s_v[t][AXK2_WARP + lane];
				acc2 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2);
				acc3 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3);
			}
		}
		__syncthreads();
	}
	if ( threadIdx.x < AXK2_BLOCK * AXK2_ATTN_VEC4S )
	{
		const uint32_t load_key = threadIdx.x / AXK2_ATTN_VEC4S, load_chunk = threadIdx.x % AXK2_ATTN_VEC4S;
		const uint32_t col = block_at + load_key;
		const uint4 kv_k = *(const uint4 *)(head_keys + (uint64_t)col * AXK2_HEAD_DIM + load_chunk * 8u);
		const uint4 kv_v = *(const uint4 *)(head_values + (uint64_t)col * AXK2_HEAD_DIM + load_chunk * 8u);
		s_k[load_key][load_chunk * 4u + 0u] = kv_k.x;
		s_k[load_key][load_chunk * 4u + 1u] = kv_k.y;
		s_k[load_key][load_chunk * 4u + 2u] = kv_k.z;
		s_k[load_key][load_chunk * 4u + 3u] = kv_k.w;
		s_v[load_key][load_chunk * 4u + 0u] = kv_v.x;
		s_v[load_key][load_chunk * 4u + 1u] = kv_v.y;
		s_v[load_key][load_chunk * 4u + 2u] = kv_v.z;
		s_v[load_key][load_chunk * 4u + 3u] = kv_v.w;
	}
	__syncthreads();
	for ( uint32_t t = 0u; t <= warp; t++ )
	{
		const uint32_t kp0 = s_k[t][lane];
		float dot = fmaf(q0, LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1 * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
		if ( lane < AXK2_HALF - AXK2_WARP )
		{
			const uint32_t kp1 = s_k[t][AXK2_WARP + lane];
			dot = fmaf(q2, LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), fmaf(q3, LmBf16ToFloat((uint16_t)(kp1 >> 16u)), dot));
		}
		dot = Axk2WarpSum(dot);
		if ( lane == 0u )
			s_scores[warp][t] = dot * scale;
	}
	__syncwarp();
	{
		const float sc = lane <= warp ? s_scores[warp][lane] : -INFINITY;
		const float tile_max = Axk2WarpMax(sc);
		const float next_max = fmaxf(running_max, tile_max);
		const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
		const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
		const float tile_sum = Axk2WarpSum(weight);
		running_sum = running_sum * rescale + tile_sum;
		running_max = next_max;
		acc0 *= rescale;
		acc1 *= rescale;
		acc2 *= rescale;
		acc3 *= rescale;
		s_scores[warp][lane] = weight;
	}
	__syncwarp();
	for ( uint32_t t = 0u; t <= warp; t++ )
	{
		const float wt = s_scores[warp][t];
		const uint32_t vp0 = s_v[t][lane];
		acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0);
		acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1);
		if ( lane < AXK2_HALF - AXK2_WARP )
		{
			const uint32_t vp1 = s_v[t][AXK2_WARP + lane];
			acc2 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2);
			acc3 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3);
		}
	}
	out_row[2u * lane] = LmFloatToBf16(acc0 / running_sum);
	out_row[2u * lane + 1u] = LmFloatToBf16(acc1 / running_sum);
	if ( lane < AXK2_HALF - AXK2_WARP )
	{
		out_row[2u * (AXK2_WARP + lane)] = LmFloatToBf16(acc2 / running_sum);
		out_row[2u * (AXK2_WARP + lane) + 1u] = LmFloatToBf16(acc3 / running_sum);
	}
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2ResidualKernel(const uint16_t *__restrict__ base, const uint16_t *__restrict__ addend, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * AXK2_THREADS + threadIdx.x; index < count; index += gridDim.x * AXK2_THREADS )
		output[index] = LmFloatToBf16(LmBf16ToFloat(base[index]) + LmBf16ToFloat(addend[index]));
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2SwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * AXK2_THREADS + threadIdx.x; index < count; index += gridDim.x * AXK2_THREADS )
	{
		const float g = LmBf16ToFloat(gate[index]);
		output[index] = LmFloatToBf16(g / (1.0f + expf(-g)) * LmBf16ToFloat(up[index]));
	}
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2WidenKernel(const uint16_t *__restrict__ input, float *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * AXK2_THREADS + threadIdx.x; index < count; index += gridDim.x * AXK2_THREADS )
		output[index] = LmBf16ToFloat(input[index]);
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2MarkovRowKernel(uint32_t step, uint32_t anchor, const uint32_t *__restrict__ tokens, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ row)
{
	__shared__ float vector[AXK2_MARKOV];
	const uint32_t previous = step == 0u ? anchor : tokens[step - 1u];
	const uint32_t warp = threadIdx.x / AXK2_WARP, lane = threadIdx.x % AXK2_WARP;
	for ( uint32_t index = threadIdx.x; index < AXK2_MARKOV; index += AXK2_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * AXK2_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * AXK2_WARPS + warp; slot < AXK2_CANDIDATES; slot += gridDim.x * AXK2_WARPS )
	{
		const uint32_t candidate = candidates[step * AXK2_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * AXK2_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < AXK2_MARKOV; element += AXK2_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = Axk2WarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * AXK2_DRAFT_VOCAB + candidate] + sum;
	}
}

static __device__ __forceinline__ unsigned long long Axk2RowKey(const float *row, uint32_t slot)
{
	return ((unsigned long long)SparkMathSampleOrdered(row[slot]) << 32u) | (unsigned long long)(0xffffffffu - slot);
}

static __device__ __forceinline__ uint32_t Axk2KeySlot(unsigned long long key)
{
	return 0xffffffffu - (uint32_t)key;
}

static __device__ __forceinline__ void Axk2Top2Insert(unsigned long long candidate, unsigned long long *first, unsigned long long *second)
{
	if ( candidate > *first )
	{
		*second = *first;
		*first = candidate;
	}
	else if ( candidate > *second && candidate != *first )
		*second = candidate;
}

__global__ __launch_bounds__(AXK2_PICK_THREADS, 1)
void Axk2MarkovPickKernel(uint32_t step, uint32_t anchor, uint32_t position, unsigned long long seed, float inverse_temperature, float nucleus,
	uint32_t top_k, const float *__restrict__ row, const uint32_t *__restrict__ candidates, const int64_t *__restrict__ d2t,
	const uint16_t *__restrict__ y, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ confidence_weight,
	const uint16_t *__restrict__ confidence_bias, Axk2Picks *__restrict__ picks)
{
	__shared__ unsigned long long keys[AXK2_CANDIDATES];
	__shared__ unsigned long long best[AXK2_PICK_THREADS / AXK2_WARP];
	__shared__ unsigned long long second_best[AXK2_PICK_THREADS / AXK2_WARP];
	__shared__ float reduction[AXK2_PICK_THREADS / AXK2_WARP];
	__shared__ uint32_t chosen;
	__shared__ uint32_t alt_chosen;
	__shared__ float peak_scaled;
	const uint32_t previous = step == 0u ? anchor : picks->tokens[step - 1u];
	const uint32_t warp = threadIdx.x / AXK2_WARP, lane = threadIdx.x % AXK2_WARP;
	const float scale = inverse_temperature > 0.0f ? inverse_temperature : 1.0f;
	float dot = 0.0f, denom = 0.0f;
	if ( inverse_temperature > 0.0f )
	{
		for ( uint32_t index = threadIdx.x; index < AXK2_CANDIDATES; index += AXK2_PICK_THREADS )
			keys[index] = Axk2RowKey(row, index);
		__syncthreads();
		for ( uint32_t size = 2u; size <= AXK2_CANDIDATES; size <<= 1u )
			for ( uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u )
			{
				for ( uint32_t index = threadIdx.x; index < AXK2_CANDIDATES; index += AXK2_PICK_THREADS )
				{
					const uint32_t other = index ^ stride;
					if ( other > index )
					{
						const unsigned long long a = keys[index], b = keys[other];
						if ( ((index & size) == 0u) ? a < b : a > b )
						{
							keys[index] = b;
							keys[other] = a;
						}
					}
				}
				__syncthreads();
			}
		if ( warp == 0u )
		{
			const uint32_t slot = Axk2KeySlot(keys[lane]);
			const uint32_t token = Axk2DraftToTarget(d2t, candidates[step * AXK2_CANDIDATES + slot]);
			const float scaled = row[slot] * inverse_temperature;
			float peak = Axk2WarpMax(scaled), probability = expf(scaled - peak), total = Axk2WarpSum(probability), running, keyed;
			uint32_t winner, runner;
			unsigned long long key;
			probability /= total;
			running = probability;
			for ( uint32_t offset = 1u; offset < AXK2_SAMPLED; offset <<= 1u )
			{
				const float earlier = __shfl_up_sync(0xffffffffu, running, offset);
				if ( lane >= offset )
					running += earlier;
			}
			keyed = (((nucleus <= 0.0f) || (running - probability < nucleus)) && (top_k == 0u || lane < top_k)) || lane == 0u
				? scaled + SparkMathGumbelNoise(seed, position + step, token) : -INFINITY;
			key = ((unsigned long long)SparkMathSampleOrdered(keyed) << 32u) | (unsigned long long)(0xffffffffu - lane);
			for ( uint32_t mask = AXK2_WARP / 2u; mask > 0u; mask >>= 1u )
			{
				const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, mask);
				key = other > key ? other : key;
			}
			winner = 0xffffffffu - (uint32_t)key;
			runner = winner == 0u ? 1u : 0u;
			if ( lane == winner )
				chosen = slot;
			if ( lane == runner )
				alt_chosen = slot;
			if ( lane == 0u )
				peak_scaled = peak;
		}
	}
	else
	{
		unsigned long long first = 0ull, second = 0ull;
		for ( uint32_t index = threadIdx.x; index < AXK2_CANDIDATES; index += AXK2_PICK_THREADS )
			Axk2Top2Insert(Axk2RowKey(row, index), &first, &second);
		for ( uint32_t mask = AXK2_WARP / 2u; mask > 0u; mask >>= 1u )
		{
			const unsigned long long other_first = __shfl_xor_sync(0xffffffffu, first, mask);
			const unsigned long long other_second = __shfl_xor_sync(0xffffffffu, second, mask);
			Axk2Top2Insert(other_first, &first, &second);
			Axk2Top2Insert(other_second, &first, &second);
		}
		if ( lane == 0u )
		{
			best[warp] = first;
			second_best[warp] = second;
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			first = best[0];
			second = second_best[0];
			for ( uint32_t index = 1u; index < AXK2_PICK_THREADS / AXK2_WARP; index++ )
			{
				Axk2Top2Insert(best[index], &first, &second);
				Axk2Top2Insert(second_best[index], &first, &second);
			}
			chosen = Axk2KeySlot(first);
			alt_chosen = Axk2KeySlot(second);
			peak_scaled = row[chosen];
		}
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < AXK2_CANDIDATES; index += AXK2_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<AXK2_PICK_THREADS>(denom, reduction);
	for ( uint32_t index = threadIdx.x; index < AXK2_CONFIDENCE; index += AXK2_PICK_THREADS )
	{
		const float feature = index < AXK2_HIDDEN ? LmBf16ToFloat(y[(uint64_t)step * AXK2_HIDDEN + index])
			: LmBf16ToFloat(markov_previous[(uint64_t)previous * AXK2_MARKOV + index - AXK2_HIDDEN]);
		dot += LmBf16ToFloat(confidence_weight[index]) * feature;
	}
	dot = LmBlockSum<AXK2_PICK_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->tokens[step] = Axk2DraftToTarget(d2t, candidates[step * AXK2_CANDIDATES + chosen]);
		picks->scores[step] = 1.0f / (1.0f + expf(-(dot + LmBf16ToFloat(confidence_bias[0]))));
		picks->alt_tokens[step] = Axk2DraftToTarget(d2t, candidates[step * AXK2_CANDIDATES + alt_chosen]);
		picks->alt_scores[step] = denom > 0.0f ? expf(row[alt_chosen] * scale - peak_scaled) / denom : 0.0f;
	}
}

__global__ __launch_bounds__(AXK2_WARP, 1)
void Axk2BranchSelectKernel(Axk2Picks *__restrict__ picks, uint32_t drafts, float min_score)
{
	uint32_t used = 0u;
	if ( threadIdx.x != 0u )
		return;
	for ( uint32_t branch = 0u; branch < AXK2_BRANCHES; branch++ )
	{
		picks->branch_fork[branch] = AXK2_BLOCK;
		picks->branch_len[branch] = 0u;
	}
	for ( uint32_t step = 0u; step + 1u < drafts; step++ )
	{
		const float score = picks->alt_scores[step];
		uint32_t pos = used;
		if ( picks->alt_tokens[step] == picks->tokens[step] || !(score >= min_score) )
			continue;
		while ( pos > 0u && score > picks->branch_scores[pos - 1u][0] )
		{
			if ( pos < AXK2_BRANCHES )
			{
				picks->branch_fork[pos] = picks->branch_fork[pos - 1u];
				picks->branch_len[pos] = picks->branch_len[pos - 1u];
				picks->branch_tokens[pos][0] = picks->branch_tokens[pos - 1u][0];
				picks->branch_scores[pos][0] = picks->branch_scores[pos - 1u][0];
			}
			pos--;
		}
		if ( pos < AXK2_BRANCHES )
		{
			picks->branch_fork[pos] = step;
			picks->branch_len[pos] = 1u;
			picks->branch_tokens[pos][0] = picks->alt_tokens[step];
			picks->branch_scores[pos][0] = score;
			if ( used < AXK2_BRANCHES )
				used++;
		}
	}
}

__global__ __launch_bounds__(AXK2_THREADS, 1)
void Axk2BranchRowKernel(uint32_t offset, uint32_t drafts, const Axk2Picks *__restrict__ picks, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ rows)
{
	__shared__ float vector[AXK2_MARKOV];
	const uint32_t branch = blockIdx.y, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / AXK2_WARP, lane = threadIdx.x % AXK2_WARP;
	uint32_t previous;
	float *row;
	if ( fork >= drafts || step >= drafts )
		return;
	previous = picks->branch_tokens[branch][offset - 1u];
	row = rows + (uint64_t)branch * AXK2_CANDIDATES;
	for ( uint32_t index = threadIdx.x; index < AXK2_MARKOV; index += AXK2_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * AXK2_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * AXK2_WARPS + warp; slot < AXK2_CANDIDATES; slot += gridDim.x * AXK2_WARPS )
	{
		const uint32_t candidate = candidates[step * AXK2_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * AXK2_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < AXK2_MARKOV; element += AXK2_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = Axk2WarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * AXK2_DRAFT_VOCAB + candidate] + sum;
	}
}

__global__ __launch_bounds__(AXK2_PICK_THREADS, 1)
void Axk2BranchPickKernel(uint32_t offset, uint32_t drafts, float scale, const float *__restrict__ rows, const uint32_t *__restrict__ candidates,
	const int64_t *__restrict__ d2t, Axk2Picks *__restrict__ picks)
{
	__shared__ unsigned long long best[AXK2_PICK_THREADS / AXK2_WARP];
	__shared__ float reduction[AXK2_PICK_THREADS / AXK2_WARP];
	__shared__ uint32_t chosen;
	__shared__ float peak_scaled;
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / AXK2_WARP, lane = threadIdx.x % AXK2_WARP;
	const float *row = rows + (uint64_t)branch * AXK2_CANDIDATES;
	unsigned long long first = 0ull;
	float denom = 0.0f;
	if ( fork >= drafts || step >= drafts )
		return;
	for ( uint32_t index = threadIdx.x; index < AXK2_CANDIDATES; index += AXK2_PICK_THREADS )
	{
		const unsigned long long key = Axk2RowKey(row, index);
		if ( key > first )
			first = key;
	}
	for ( uint32_t mask = AXK2_WARP / 2u; mask > 0u; mask >>= 1u )
	{
		const unsigned long long other = __shfl_xor_sync(0xffffffffu, first, mask);
		if ( other > first )
			first = other;
	}
	if ( lane == 0u )
		best[warp] = first;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		first = best[0];
		for ( uint32_t index = 1u; index < AXK2_PICK_THREADS / AXK2_WARP; index++ )
			if ( best[index] > first )
				first = best[index];
		chosen = Axk2KeySlot(first);
		peak_scaled = row[chosen] * scale;
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < AXK2_CANDIDATES; index += AXK2_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<AXK2_PICK_THREADS>(denom, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->branch_tokens[branch][offset] = Axk2DraftToTarget(d2t, candidates[step * AXK2_CANDIDATES + chosen]);
		picks->branch_scores[branch][offset] = denom > 0.0f ? 1.0f / denom : 0.0f;
		picks->branch_len[branch] = offset + 1u;
	}
}

static SparkStatus Axk2Refuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	SPARK_FAIL(status);
}

static SparkStatus Axk2LoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
{
	uint8_t header[16];
	uint32_t magic, version;
	uint64_t bytes;
	char *text;
	SparkStatus status;
	FILE *file = fopen(pack_path, "rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( fread(header, 1u, sizeof(header), file) != sizeof(header) )
	{
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	memcpy(&magic, header, sizeof(magic));
	memcpy(&version, header + sizeof(magic), sizeof(version));
	memcpy(&bytes, header + sizeof(magic) + sizeof(version), sizeof(bytes));
	if ( magic != AXK2_PACK_MAGIC || version != AXK2_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
	{
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	text = (char *)malloc((size_t)bytes);
	if ( text == 0 || fread(text, 1u, (size_t)bytes, file) != (size_t)bytes )
	{
		free(text);
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	(void)fclose(file);
	status = SparkJsonParseText(text, (size_t)bytes, document);
	free(text);
	*payload_base = (sizeof(header) + bytes + AXK2_PACK_ALIGNMENT - 1u) / AXK2_PACK_ALIGNMENT * AXK2_PACK_ALIGNMENT;
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2TensorBytes(const Axk2State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name,
	uint64_t expected_bytes, const void **pointer)
{
	const int32_t tensors = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "tensors");
	const int32_t entry = tensors < 0 ? -1 : SparkJsonFindObjectMember(document, tensors, name);
	const int32_t offset_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "offset");
	const int32_t bytes_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "bytes");
	uint64_t offset = 0u, bytes = 0u;
	const void *slice = 0;
	if ( offset_token < 0 || bytes_token < 0 || SparkJsonGetUInt64(document, offset_token, &offset) != SPARK_STATUS_OK ||
		SparkJsonGetUInt64(document, bytes_token, &bytes) != SPARK_STATUS_OK || bytes != expected_bytes )
	{
		fprintf(stderr, "axk2 dspark: pack tensor %s is missing or is not %llu bytes\n", name, (unsigned long long)expected_bytes);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*pointer = slice;
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2Tensor(const Axk2State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t expected,
	const uint16_t **pointer)
{
	const void *slice = 0;
	const SparkStatus status = Axk2TensorBytes(state, document, payload_base, name, expected * sizeof(uint16_t), &slice);
	if ( status == SPARK_STATUS_OK )
		*pointer = (const uint16_t *)slice;
	return status;
}

static int32_t Axk2Member(const SparkJsonDocument *document, int32_t object, const char *name)
{
	return object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
}

static double Axk2YarnScale(double scale, double weight)
{
	return scale <= 1.0 || weight == 0.0 ? 1.0 : 0.1 * weight * log(scale) + 1.0;
}

static SparkStatus Axk2Shape(Axk2State *state, const SparkJsonDocument *document)
{
	const int32_t config = Axk2Member(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = Axk2Member(document, config, "checkpoint_config");
	const int32_t transformer = Axk2Member(document, checkpoint, "transformer_layer_config");
	const int32_t rope = Axk2Member(document, transformer, "rope_parameters");
	const int32_t aux = Axk2Member(document, checkpoint, "aux_hidden_state_layer_ids");
	const int32_t layer_types = Axk2Member(document, transformer, "layer_types");
	const struct
	{
		int32_t object;
		const char *name;
		uint64_t value;
	} expected[] = {
		{checkpoint, "block_size", AXK2_BLOCK},
		{checkpoint, "draft_vocab_size", AXK2_DRAFT_VOCAB},
		{checkpoint, "markov_rank", AXK2_MARKOV},
		{transformer, "hidden_size", AXK2_HIDDEN},
		{transformer, "intermediate_size", AXK2_MLP},
		{transformer, "num_hidden_layers", AXK2_LAYERS},
		{transformer, "num_attention_heads", AXK2_HEADS},
		{transformer, "num_key_value_heads", AXK2_KV_HEADS},
		{transformer, "head_dim", AXK2_HEAD_DIM},
		{transformer, "sliding_window", AXK2_SLIDING},
		{transformer, "vocab_size", AXK2_VOCAB},
	};
	const struct
	{
		int32_t object;
		const char *name;
		bool value;
	} flags[] = {
		{checkpoint, "sample_from_anchor", true},
		{checkpoint, "enable_confidence_head", true},
		{checkpoint, "confidence_head_with_markov", true},
		{checkpoint, "sliding_window_non_causal", false},
		{transformer, "use_sliding_window", true},
	};
	uint64_t value = 0u, original = 0u;
	float theta = 0.0f, factor = 1.0f, beta_fast = 32.0f, beta_slow = 1.0f, mscale = 1.0f, mscale_all = 1.0f;
	int32_t token;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		token = Axk2Member(document, expected[index].object, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "axk2 dspark: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	for ( uint32_t index = 0u; index < sizeof(flags) / sizeof(flags[0]); index++ )
	{
		bool flag = false;
		token = Axk2Member(document, flags[index].object, flags[index].name);
		if ( token < 0 || SparkJsonGetBoolean(document, token, &flag) != SPARK_STATUS_OK || flag != flags[index].value )
		{
			fprintf(stderr, "axk2 dspark: checkpoint_config %s does not match module contract\n", flags[index].name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	token = Axk2Member(document, checkpoint, "markov_head_type");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "vanilla") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( aux < 0 || SparkJsonGetArrayElementCount(document, aux) != AXK2_TAPS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( layer_types < 0 || SparkJsonGetArrayElementCount(document, layer_types) != AXK2_LAYERS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t layer = 0u; layer < AXK2_LAYERS; layer++ )
	{
		const int32_t elem = SparkJsonGetArrayElement(document, layer_types, layer);
		if ( elem < 0 || !SparkJsonStringEquals(document, elem, "sliding_attention") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	token = Axk2Member(document, checkpoint, "mask_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= AXK2_VOCAB )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->mask_token = (uint32_t)value;
	token = Axk2Member(document, transformer, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "rope_type");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "yarn") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "rope_theta");
	if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK || theta <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "factor");
	if ( token < 0 || SparkJsonGetFloat(document, token, &factor) != SPARK_STATUS_OK || factor < 1.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "original_max_position_embeddings");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &original) != SPARK_STATUS_OK || original == 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "beta_fast");
	if ( token < 0 || SparkJsonGetFloat(document, token, &beta_fast) != SPARK_STATUS_OK || beta_fast <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "beta_slow");
	if ( token < 0 || SparkJsonGetFloat(document, token, &beta_slow) != SPARK_STATUS_OK || beta_slow <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "mscale");
	if ( token >= 0 && SparkJsonGetFloat(document, token, &mscale) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Axk2Member(document, rope, "mscale_all_dim");
	if ( token >= 0 && SparkJsonGetFloat(document, token, &mscale_all) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->rope_gain = (float)(Axk2YarnScale((double)factor, (double)mscale) / Axk2YarnScale((double)factor, (double)mscale_all));
	state->softmax_scale = (float)(1.0 / sqrt((double)AXK2_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, AXK2_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	LM_LAUNCH((Axk2RopeTableKernel), 1u, AXK2_THREADS, 0, state->stream, state->rope_inverse, (double)theta, (double)factor, (double)original,
		(double)beta_fast, (double)beta_slow);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2Weights(Axk2State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	const void *d2t_slice = 0;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = Axk2LoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = Axk2Shape(state, &document);
	{
		const struct
		{
			const char *name;
			uint64_t elements;
			const uint16_t **pointer;
		} plain[] = {
			{"fc.weight", (uint64_t)AXK2_HIDDEN * AXK2_TAP_WIDTH, &state->fc},
			{"hidden_norm.weight", AXK2_HIDDEN, &state->hidden_norm},
			{"norm.weight", AXK2_HIDDEN, &state->norm},
			{"target.embed", (uint64_t)AXK2_VOCAB * AXK2_HIDDEN, &state->embed},
			{"target.head", (uint64_t)AXK2_DRAFT_VOCAB * AXK2_HIDDEN, &state->head},
			{"markov_head.markov_w1.weight", (uint64_t)AXK2_VOCAB * AXK2_MARKOV, &state->markov_previous},
			{"markov_head.markov_w2.weight", (uint64_t)AXK2_DRAFT_VOCAB * AXK2_MARKOV, &state->markov_next},
			{"confidence_head.proj.weight", AXK2_CONFIDENCE, &state->confidence_weight},
			{"confidence_head.proj.bias", 1u, &state->confidence_bias},
		};
		for ( uint32_t index = 0u; index < sizeof(plain) / sizeof(plain[0]) && status == SPARK_STATUS_OK; index++ )
			status = Axk2Tensor(state, &document, payload_base, plain[index].name, plain[index].elements, plain[index].pointer);
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = Axk2TensorBytes(state, &document, payload_base, "d2t", (uint64_t)AXK2_DRAFT_VOCAB * sizeof(int64_t), &d2t_slice);
		if ( status == SPARK_STATUS_OK )
			state->d2t = (const int64_t *)d2t_slice;
	}
	for ( uint32_t layer = 0u; layer < AXK2_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		Axk2Layer *weights = &state->layers[layer];
		const struct
		{
			const char *suffix;
			uint64_t elements;
			const uint16_t **pointer;
		} tensors[] = {
			{"input_layernorm.weight", AXK2_HIDDEN, &weights->input_norm},
			{"post_attention_layernorm.weight", AXK2_HIDDEN, &weights->post_norm},
			{"self_attn.q_proj.weight", (uint64_t)AXK2_Q_WIDTH * AXK2_HIDDEN, &weights->q},
			{"self_attn.k_proj.weight", (uint64_t)AXK2_KV_WIDTH * AXK2_HIDDEN, &weights->k},
			{"self_attn.v_proj.weight", (uint64_t)AXK2_KV_WIDTH * AXK2_HIDDEN, &weights->v},
			{"self_attn.o_proj.weight", (uint64_t)AXK2_HIDDEN * AXK2_Q_WIDTH, &weights->o},
			{"self_attn.q_norm.weight", AXK2_HEAD_DIM, &weights->q_norm},
			{"self_attn.k_norm.weight", AXK2_HEAD_DIM, &weights->k_norm},
			{"mlp.gate_proj.weight", (uint64_t)AXK2_MLP * AXK2_HIDDEN, &weights->gate},
			{"mlp.up_proj.weight", (uint64_t)AXK2_MLP * AXK2_HIDDEN, &weights->up},
			{"mlp.down_proj.weight", (uint64_t)AXK2_HIDDEN * AXK2_MLP, &weights->down},
		};
		for ( uint32_t index = 0u; index < sizeof(tensors) / sizeof(tensors[0]) && status == SPARK_STATUS_OK; index++ )
		{
			char name[128];
			snprintf(name, sizeof(name), "layers.%u.%s", layer, tensors[index].suffix);
			status = Axk2Tensor(state, &document, payload_base, name, tensors[index].elements, tensors[index].pointer);
		}
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define AXK2_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus Axk2Scratch(Axk2State *state, uint64_t context_bytes)
{
	Axk2Forward *forward = &state->forward;
	int multiprocessors = 0;
	uint64_t capacity = context_bytes / AXK2_SEQUENCE_BYTES;
	capacity = capacity < 1u ? 1u : capacity > AXK2_SEQUENCES_MAX ? AXK2_SEQUENCES_MAX : capacity;
	state->capacity = (uint32_t)capacity;
	state->sequences = (Axk2Sequence *)calloc(capacity, sizeof(Axk2Sequence));
	if ( state->sequences == 0 || SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)AXK2_INGEST_ROWS * AXK2_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)AXK2_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(Axk2Draft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(Axk2Picks)) != SPARK_GPU_SUCCESS ||
		!AXK2_DEVICE(state->caches, capacity * AXK2_SEQUENCE_ELEMENTS) || !AXK2_DEVICE(state->taps, (uint64_t)AXK2_INGEST_ROWS * AXK2_TAP_WIDTH) ||
		!AXK2_DEVICE(state->positions, AXK2_INGEST_ROWS) || !AXK2_DEVICE(state->features_raw, (uint64_t)AXK2_INGEST_ROWS * AXK2_HIDDEN) ||
		!AXK2_DEVICE(state->features, (uint64_t)AXK2_INGEST_ROWS * AXK2_HIDDEN) || !AXK2_DEVICE(state->keys_raw, (uint64_t)AXK2_INGEST_ROWS * AXK2_KV_WIDTH) ||
		!AXK2_DEVICE(state->values_raw, (uint64_t)AXK2_INGEST_ROWS * AXK2_KV_WIDTH) || !AXK2_DEVICE(forward->draft, 1u) || !AXK2_DEVICE(forward->picks, 1u) ||
		!AXK2_DEVICE(forward->hidden, AXK2_BLOCK * AXK2_HIDDEN) || !AXK2_DEVICE(forward->x, AXK2_BLOCK * AXK2_HIDDEN) ||
		!AXK2_DEVICE(forward->q, AXK2_BLOCK * AXK2_Q_WIDTH) || !AXK2_DEVICE(forward->q_roped, AXK2_BLOCK * AXK2_Q_WIDTH) ||
		!AXK2_DEVICE(forward->k, AXK2_BLOCK * AXK2_KV_WIDTH) || !AXK2_DEVICE(forward->v, AXK2_BLOCK * AXK2_KV_WIDTH) ||
		!AXK2_DEVICE(forward->attended, AXK2_BLOCK * AXK2_Q_WIDTH) || !AXK2_DEVICE(forward->projected, AXK2_BLOCK * AXK2_HIDDEN) ||
		!AXK2_DEVICE(forward->residual, AXK2_BLOCK * AXK2_HIDDEN) || !AXK2_DEVICE(forward->gate, AXK2_BLOCK * AXK2_MLP) ||
		!AXK2_DEVICE(forward->up, AXK2_BLOCK * AXK2_MLP) || !AXK2_DEVICE(forward->act, AXK2_BLOCK * AXK2_MLP) ||
		!AXK2_DEVICE(forward->mlp, AXK2_BLOCK * AXK2_HIDDEN) || !AXK2_DEVICE(forward->y, AXK2_BLOCK * AXK2_HIDDEN) ||
		!AXK2_DEVICE(forward->logits, (uint64_t)AXK2_BLOCK * AXK2_DRAFT_VOCAB) || !AXK2_DEVICE(forward->logits_f32, (uint64_t)AXK2_BLOCK * AXK2_DRAFT_VOCAB) ||
		!AXK2_DEVICE(forward->candidates, AXK2_BLOCK * AXK2_CANDIDATES) || !AXK2_DEVICE(forward->markov_row, AXK2_BRANCHES * AXK2_CANDIDATES) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		state->sequences[index].cache = state->caches + (uint64_t)index * AXK2_SEQUENCE_ELEMENTS;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static uint16_t *Axk2Keys(const Axk2Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer) * AXK2_LAYER_KV;
}

static uint16_t *Axk2Values(const Axk2Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer + 1u) * AXK2_LAYER_KV;
}

static void Axk2Reset(Axk2Sequence *sequence)
{
	for ( uint32_t slot = 0u; slot < AXK2_WINDOW; slot++ )
	{
		sequence->positions[slot] = -1;
		sequence->tokens[slot] = 0u;
	}
	sequence->retired = 0u;
}

static Axk2Sequence *Axk2Live(Axk2State *state, uint64_t id)
{
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		if ( state->sequences[index].live != 0u && state->sequences[index].retired == 0u && state->sequences[index].id == id )
			return &state->sequences[index];
	return 0;
}

static Axk2Sequence *Axk2Claim(Axk2State *state)
{
	Axk2Sequence *oldest = 0;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
	{
		Axk2Sequence *sequence = &state->sequences[index];
		if ( sequence->live == 0u )
			return sequence;
		if ( oldest == 0 || sequence->last_used < oldest->last_used )
			oldest = sequence;
	}
	return oldest;
}

static uint32_t Axk2Held(const Axk2Sequence *sequence, uint64_t position)
{
	return sequence->positions[position % AXK2_WINDOW] == (int64_t)position ? 1u : 0u;
}

static SparkStatus Axk2Gemm(Axk2State *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2IngestRows(Axk2State *state, Axk2Sequence *sequence, uint32_t rows)
{
	SparkStatus status = Axk2Gemm(state, state->fc, state->taps, state->features_raw, rows, AXK2_TAP_WIDTH, AXK2_HIDDEN);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((Axk2RmsKernel), rows, AXK2_THREADS, 0, state->stream, state->features_raw, state->hidden_norm, state->features, AXK2_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < AXK2_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = Axk2Gemm(state, state->layers[layer].k, state->features, state->keys_raw, rows, AXK2_HIDDEN, AXK2_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Axk2Gemm(state, state->layers[layer].v, state->features, state->values_raw, rows, AXK2_HIDDEN, AXK2_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Axk2HeadNormRopeKernel), rows * AXK2_KV_HEADS, AXK2_HEAD_THREADS, 0, state->stream, state->keys_raw, state->layers[layer].k_norm,
			state->rope_inverse, state->positions, AXK2_KV_HEADS, state->epsilon, state->rope_gain, (uint16_t *)0, Axk2Keys(sequence, layer), 0u, 1u);
		LM_LAUNCH((Axk2StoreValuesKernel), rows, AXK2_THREADS, 0, state->stream, state->values_raw, state->positions, 0u, Axk2Values(sequence, layer));
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static SparkStatus Axk2Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	Axk2State *state = (Axk2State *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	const uint32_t skip = rows > AXK2_WINDOW ? rows - AXK2_WINDOW : 0u;
	const uint64_t first = frame->header.first_position + skip;
	Axk2Sequence *sequence;
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != AXK2_TAPS || hidden != AXK2_HIDDEN || frame->rows == 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t tap = 0u; tap < AXK2_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * AXK2_HIDDEN > frame->row_elements )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( frame->header.first_position + rows > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = Axk2Live(state, frame->header.sequence_id);
	if ( sequence != 0 && frame->header.first_position != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence->last_used = ++state->clock;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = Axk2Claim(state);
		Axk2Reset(sequence);
		sequence->live = 1u;
		sequence->id = frame->header.sequence_id;
	}
	else if ( frame->header.first_position == 0u )
		Axk2Reset(sequence);
	sequence->control_generation = frame->header.control_generation;
	sequence->request_id = frame->header.request_id;
	sequence->last_used = ++state->clock;
	for ( uint32_t row = skip; row < rows; row++ )
		sequence->positions[(frame->header.first_position + row) % AXK2_WINDOW] = -1;
	for ( uint32_t done = skip; done < rows && status == SPARK_STATUS_OK; done += AXK2_INGEST_ROWS )
	{
		const uint32_t part = rows - done < AXK2_INGEST_ROWS ? rows - done : AXK2_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < AXK2_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * AXK2_TAP_WIDTH + (uint64_t)tap * AXK2_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * AXK2_HIDDEN, AXK2_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(frame->header.first_position + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * AXK2_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t),
			SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		{
			(void)SparkGpuStreamSynchronize(state->stream);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		status = Axk2IngestRows(state, sequence, part);
	}
	if ( status == SPARK_STATUS_OK )
	{
		for ( uint32_t slot = 0u; slot < AXK2_WINDOW; slot++ )
			if ( sequence->positions[slot] >= (int64_t)first )
				sequence->positions[slot] = -1;
		for ( uint32_t row = skip; row < rows; row++ )
		{
			const uint64_t position = frame->header.first_position + row;
			sequence->positions[position % AXK2_WINDOW] = (int64_t)position;
			sequence->tokens[position % AXK2_WINDOW] = frame->tokens != 0 ? frame->tokens[row] : 0u;
		}
	}
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus Axk2CopySlots(Axk2State *state, Axk2Sequence *sequence, const Axk2Sequence *donor, uint32_t slot, uint32_t count)
{
	for ( uint32_t layer = 0u; layer < AXK2_LAYERS; layer++ )
		for ( uint32_t head = 0u; head < AXK2_KV_HEADS; head++ )
		{
			const uint64_t at = ((uint64_t)head * AXK2_COLUMNS + slot) * AXK2_HEAD_DIM;
			const size_t bytes = (size_t)count * AXK2_HEAD_DIM * sizeof(uint16_t);
			if ( SparkGpuMemcpyAsync(Axk2Keys(sequence, layer) + at, Axk2Keys(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) !=
				SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(Axk2Values(sequence, layer) + at, Axk2Values(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	return SPARK_STATUS_OK;
}

static uint32_t Axk2Usable(const Axk2Sequence *sequence, const Axk2Sequence *donor, const SparkDraftServerRequest *request, uint64_t position)
{
	return Axk2Held(sequence, position) == 0u && Axk2Held(donor, position) != 0u && position < request->committed_token_count &&
		donor->tokens[position % AXK2_WINDOW] == request->committed_token_ids[position] ? 1u : 0u;
}

static SparkStatus Axk2FillFromDonors(Axk2State *state, Axk2Sequence *sequence, const SparkDraftServerRequest *request, uint64_t first)
{
	uint8_t tried[AXK2_SEQUENCES_MAX];
	memset(tried, 0, sizeof(tried));
	for ( ;; )
	{
		Axk2Sequence *donor = 0;
		uint32_t chosen = 0u, missing = 0u;
		uint64_t position = first;
		for ( uint64_t at = first; at < request->position; at++ )
			missing += Axk2Held(sequence, at) == 0u ? 1u : 0u;
		if ( missing == 0u )
			return SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < state->capacity; index++ )
		{
			Axk2Sequence *other = &state->sequences[index];
			if ( other->live != 0u && other != sequence && tried[index] == 0u && (donor == 0 || other->last_used > donor->last_used) )
			{
				donor = other;
				chosen = index;
			}
		}
		if ( donor == 0 )
			return SPARK_STATUS_OK;
		tried[chosen] = 1u;
		while ( position < request->position )
		{
			uint64_t stop = position;
			if ( Axk2Usable(sequence, donor, request, position) == 0u )
			{
				position++;
				continue;
			}
			while ( stop < request->position && Axk2Usable(sequence, donor, request, stop) != 0u && (stop == position || stop % AXK2_WINDOW != 0u) )
				stop++;
			if ( Axk2CopySlots(state, sequence, donor, (uint32_t)(position % AXK2_WINDOW), (uint32_t)(stop - position)) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			for ( uint64_t at = position; at < stop; at++ )
			{
				sequence->positions[at % AXK2_WINDOW] = (int64_t)at;
				sequence->tokens[at % AXK2_WINDOW] = donor->tokens[at % AXK2_WINDOW];
			}
			position = stop;
		}
	}
}

static void Axk2Span(Axk2Sequence *sequence, uint64_t checked, uint64_t position, Axk2Draft *draft)
{
	const uint64_t count = position - checked;
	const uint32_t first = count != 0u ? (uint32_t)(checked % AXK2_WINDOW) : 0u, end = first + (uint32_t)count;
	uint32_t newer = 0u;
	memset(draft->valid, 0, sizeof(draft->valid));
	draft->masked = 0u;
	if ( count == AXK2_WINDOW )
	{
		draft->start = 0u;
		draft->block_at = AXK2_WINDOW;
		return;
	}
	for ( uint32_t slot = end; slot < AXK2_WINDOW && slot < end + AXK2_BLOCK; slot++ )
		newer |= count != 0u && sequence->positions[slot] > (int64_t)(position - 1u) ? 1u : 0u;
	if ( end <= AXK2_WINDOW && newer == 0u )
	{
		for ( uint32_t slot = end; slot < AXK2_WINDOW && slot < end + AXK2_BLOCK; slot++ )
			sequence->positions[slot] = -1;
		draft->start = first;
		draft->block_at = end;
		return;
	}
	for ( uint64_t at = checked; at < position; at++ )
		draft->valid[at % AXK2_WINDOW] = 1u;
	draft->masked = 1u;
	draft->start = 0u;
	draft->block_at = AXK2_WINDOW;
}

static SparkStatus Axk2Prepare(Axk2State *state, Axk2Sequence *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
	const SparkRowSampling *sampling)
{
	Axk2Draft *draft = state->forward.host_draft;
	if ( anchor >= AXK2_VOCAB || position + AXK2_BLOCK > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft, 0, offsetof(Axk2Draft, valid));
	for ( uint32_t row = 0u; row < AXK2_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	draft->anchor = anchor;
	draft->mask = state->mask_token;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	Axk2Span(sequence, checked, position, draft);
	if ( SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2RunForward(Axk2State *state, const Axk2Sequence *sequence, uint32_t layers)
{
	Axk2Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(Axk2Draft, positions));
	const uint32_t block_at = state->forward.host_draft->block_at;
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((Axk2EmbedKernel), AXK2_BLOCK, AXK2_THREADS, 0, stream, state->embed, forward->draft, forward->hidden);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const Axk2Layer *weights = &state->layers[layer];
		LM_LAUNCH((Axk2RmsKernel), AXK2_BLOCK, AXK2_THREADS, 0, stream, forward->hidden, weights->input_norm, forward->x, AXK2_HIDDEN, state->epsilon);
		status = Axk2Gemm(state, weights->q, forward->x, forward->q, AXK2_BLOCK, AXK2_HIDDEN, AXK2_Q_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Axk2Gemm(state, weights->k, forward->x, forward->k, AXK2_BLOCK, AXK2_HIDDEN, AXK2_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Axk2Gemm(state, weights->v, forward->x, forward->v, AXK2_BLOCK, AXK2_HIDDEN, AXK2_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Axk2HeadNormRopeKernel), AXK2_BLOCK * AXK2_HEADS, AXK2_HEAD_THREADS, 0, stream, forward->q, weights->q_norm, state->rope_inverse, positions,
			AXK2_HEADS, state->epsilon, state->rope_gain, forward->q_roped, (uint16_t *)0, 0u, 0u);
		LM_LAUNCH((Axk2HeadNormRopeKernel), AXK2_BLOCK * AXK2_KV_HEADS, AXK2_HEAD_THREADS, 0, stream, forward->k, weights->k_norm, state->rope_inverse,
			positions, AXK2_KV_HEADS, state->epsilon, state->rope_gain, (uint16_t *)0, Axk2Keys(sequence, layer), block_at, 0u);
		LM_LAUNCH((Axk2StoreValuesKernel), AXK2_BLOCK, AXK2_THREADS, 0, stream, forward->v, (const uint32_t *)0, block_at, Axk2Values(sequence, layer));
		LM_LAUNCH((Axk2AttentionKernel), AXK2_HEADS, AXK2_ATTN_THREADS, 0, stream, forward->q_roped, Axk2Keys(sequence, layer), Axk2Values(sequence, layer),
			forward->draft, state->softmax_scale, forward->attended);
		status = Axk2Gemm(state, weights->o, forward->attended, forward->projected, AXK2_BLOCK, AXK2_Q_WIDTH, AXK2_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Axk2ResidualKernel), 2u * state->multiprocessors, AXK2_THREADS, 0, stream, forward->hidden, forward->projected, forward->residual,
			AXK2_BLOCK * AXK2_HIDDEN);
		LM_LAUNCH((Axk2RmsKernel), AXK2_BLOCK, AXK2_THREADS, 0, stream, forward->residual, weights->post_norm, forward->x, AXK2_HIDDEN, state->epsilon);
		status = Axk2Gemm(state, weights->gate, forward->x, forward->gate, AXK2_BLOCK, AXK2_HIDDEN, AXK2_MLP);
		if ( status == SPARK_STATUS_OK )
			status = Axk2Gemm(state, weights->up, forward->x, forward->up, AXK2_BLOCK, AXK2_HIDDEN, AXK2_MLP);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Axk2SwigluKernel), 2u * state->multiprocessors, AXK2_THREADS, 0, stream, forward->gate, forward->up, forward->act, AXK2_BLOCK * AXK2_MLP);
		status = Axk2Gemm(state, weights->down, forward->act, forward->mlp, AXK2_BLOCK, AXK2_MLP, AXK2_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Axk2ResidualKernel), 2u * state->multiprocessors, AXK2_THREADS, 0, stream, forward->residual, forward->mlp, forward->hidden,
			AXK2_BLOCK * AXK2_HIDDEN);
	}
	if ( status != SPARK_STATUS_OK || layers < AXK2_LAYERS )
		return status;
	LM_LAUNCH((Axk2RmsKernel), AXK2_BLOCK, AXK2_THREADS, 0, stream, forward->hidden, state->norm, forward->y, AXK2_HIDDEN, state->epsilon);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2Heads(Axk2State *state, uint32_t anchor_token, uint64_t position, const SparkRowSampling *sampling, uint32_t drafts, uint32_t tree)
{
	Axk2Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const float nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	SparkStatus status = Axk2Gemm(state, state->head, forward->y, forward->logits, AXK2_BLOCK, AXK2_HIDDEN, AXK2_DRAFT_VOCAB);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Axk2WidenKernel), 4u * state->multiprocessors, AXK2_THREADS, 0, stream, forward->logits, forward->logits_f32, AXK2_BLOCK * AXK2_DRAFT_VOCAB);
	if ( LmTopkExactRowsLaunch<AXK2_TOPK_THREADS>(forward->logits_f32, AXK2_BLOCK, AXK2_DRAFT_VOCAB, AXK2_CANDIDATES, forward->candidates, stream) !=
		SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t step = 0u; step < AXK2_BLOCK; step++ )
	{
		LM_LAUNCH((Axk2MarkovRowKernel), AXK2_CANDIDATES / AXK2_WARPS, AXK2_THREADS, 0, stream, step, anchor_token, forward->picks->tokens,
			forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
		LM_LAUNCH((Axk2MarkovPickKernel), 1u, AXK2_PICK_THREADS, 0, stream, step, anchor_token, (uint32_t)position, (unsigned long long)sampling->seed,
			sampling->inverse_temperature, nucleus, sampling->top_k, forward->markov_row, forward->candidates, state->d2t, forward->y,
			state->markov_previous, state->confidence_weight, state->confidence_bias, forward->picks);
	}
	if ( tree != 0u && drafts >= 2u )
	{
		const float scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
		LM_LAUNCH((Axk2BranchSelectKernel), 1u, AXK2_WARP, 0, stream, forward->picks, drafts, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE);
		for ( uint32_t offset = 1u; offset < AXK2_BRANCH_LEN && offset < drafts; offset++ )
		{
			LM_LAUNCH((Axk2BranchRowKernel), dim3(AXK2_CANDIDATES / AXK2_WARPS, AXK2_BRANCHES), AXK2_THREADS, 0, stream, offset, drafts,
				forward->picks, forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
			LM_LAUNCH((Axk2BranchPickKernel), AXK2_BRANCHES, AXK2_PICK_THREADS, 0, stream, offset, drafts, scale, forward->markov_row, forward->candidates,
				state->d2t, forward->picks);
		}
	}
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(Axk2Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2Settle(Axk2State *state, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static void Axk2ForwardFree(Axk2Forward *forward)
{
	void *buffers[] = {
		forward->draft, forward->picks, forward->hidden, forward->x, forward->q, forward->q_roped, forward->k, forward->v,
		forward->attended, forward->projected, forward->residual, forward->gate, forward->up, forward->act, forward->mlp, forward->y,
		forward->logits, forward->logits_f32, forward->candidates, forward->markov_row,
	};
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		(void)SparkGpuFree(buffers[index]);
	if ( forward->host_draft != 0 )
		(void)SparkGpuFreeHost(forward->host_draft);
	if ( forward->host_picks != 0 )
		(void)SparkGpuFreeHost(forward->host_picks);
}

static SparkStatus Axk2Close(void *opaque)
{
	Axk2State *state = (Axk2State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	Axk2ForwardFree(&state->forward);
	{
		void *buffers[] = {
			state->caches, state->taps, state->positions, state->features_raw, state->features, state->keys_raw, state->values_raw, state->rope_inverse,
		};
		for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
			(void)SparkGpuFree(buffers[index]);
	}
	if ( state->host_taps != 0 )
		(void)SparkGpuFreeHost(state->host_taps);
	if ( state->host_positions != 0 )
		(void)SparkGpuFreeHost(state->host_positions);
	if ( state->stream != 0 )
		(void)SparkGpuStreamDestroy(state->stream);
	if ( state->pack != 0 )
		status = SparkWeightdLazyPackDestroy(state->pack);
	free(state->sequences);
	free(state);
	return status;
}

static SparkStatus Axk2Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	Axk2State *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return Axk2Refuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	{
		const SparkStatus device = SparkDeviceInitialize(0);
		if ( (device != SPARK_STATUS_OK && device != SPARK_STATUS_DUPLICATE) || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
			return Axk2Refuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
	}
	bind->device_free_bytes = (uint64_t)free_bytes;
	bind->device_total_bytes = (uint64_t)total_bytes;
	need = lane->pack_bytes + 255u + lane->context_bytes;
	if ( need > (uint64_t)free_bytes )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s needs %llu bytes of device memory (pack %llu + context cap %llu), %llu of %llu free",
			lane->lane_name, (unsigned long long)need, (unsigned long long)lane->pack_bytes, (unsigned long long)lane->context_bytes,
			(unsigned long long)free_bytes, (unsigned long long)total_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( lane->context_bytes < AXK2_SEQUENCE_BYTES )
		return Axk2Refuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one sequence's window", lane->lane_name);
	state = (Axk2State *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&request, 0, sizeof(request));
	memset(&options, 0, sizeof(options));
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = lane->pack_bytes;
	snprintf(request.identity.model, sizeof(request.identity.model), "%s", lane->model);
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s", lane->revision);
	snprintf(request.identity.pack_sha256, sizeof(request.identity.pack_sha256), "%s", lane->pack_sha256);
	snprintf(request.pack_path, sizeof(request.pack_path), "%s", lane->pack_path);
	options.timeout_nanoseconds = lane->timeout_ns;
	started = SparkMonotonicNs();
	status = SparkWeightdLazyPackCreate(lane->weightd_socket, &request, &options, &state->pack);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s did not bind", lane->lane_name, lane->pack_path);
		free(state);
		SPARK_FAIL(status);
	}
	status = SparkGpuStreamCreateWithFlags(&state->stream, SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = Axk2Weights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = Axk2Scratch(state, lane->context_bytes);
	if ( status == SPARK_STATUS_OK )
		status = Axk2Settle(state, SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not an A.X-K2 DSpark pack this module reads", lane->lane_name, lane->pack_path);
		(void)Axk2Close(state);
		SPARK_FAIL(status);
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s with %u sequence windows", lane->lane_name, lane->pack_path, state->capacity);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus Axk2Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	Axk2State *state = (Axk2State *)opaque;
	Axk2Sequence *sequence = Axk2Live(state, request->sequence_id);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > AXK2_WINDOW ? request->position - AXK2_WINDOW : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( sequence == 0 || (request->position != 0u && Axk2Held(sequence, request->position - 1u) == 0u) )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	status = Axk2FillFromDonors(state, sequence, request, first);
	if ( status != SPARK_STATUS_OK )
		return Axk2Settle(state, status);
	for ( uint64_t position = first; position < request->position; position++ )
		if ( Axk2Held(sequence, position) == 0u )
			checked = position + 1u;
	if ( checked != first )
		state->gaps++;
	if ( request->committed_token_count < request->position )
		return Axk2Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	for ( uint64_t position = checked; position < request->position; position++ )
		if ( sequence->tokens[position % AXK2_WINDOW] != request->committed_token_ids[position] )
			return Axk2Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	drafts = drafts < AXK2_BLOCK ? drafts : AXK2_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		SparkDraftTreeBranch branches[AXK2_BLOCK];
		uint32_t branch_count = 0u;
		status = Axk2Prepare(state, sequence, request->anchor_token_id, request->position, checked, &request->sampling);
		if ( status == SPARK_STATUS_OK )
			status = Axk2RunForward(state, sequence, AXK2_LAYERS);
		if ( status == SPARK_STATUS_OK )
			status = Axk2Heads(state, request->anchor_token_id, request->position, &request->sampling, drafts, tree);
		status = Axk2Settle(state, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				uint32_t matched = AXK2_BRANCHES;
				for ( uint32_t b = 0u; b < AXK2_BRANCHES; b++ )
					if ( state->forward.host_picks->branch_len[b] > 0u && state->forward.host_picks->branch_fork[b] == step )
						matched = b;
				branches[branch_count].fork_step = step;
				if ( matched < AXK2_BRANCHES )
				{
					const uint32_t len = state->forward.host_picks->branch_len[matched];
					branches[branch_count].length = len;
					for ( uint32_t k = 0u; k < len; k++ )
					{
						branches[branch_count].tokens[k] = state->forward.host_picks->branch_tokens[matched][k];
						branches[branch_count].scores[k] = state->forward.host_picks->branch_scores[matched][k];
					}
				}
				else
				{
					branches[branch_count].length = 1u;
					branches[branch_count].tokens[0] = state->forward.host_picks->alt_tokens[step];
					branches[branch_count].scores[0] = state->forward.host_picks->alt_scores[step];
				}
				branch_count++;
			}
		*count = SparkDraftTreePackBranches(request->anchor_token_id, source, state->forward.host_picks->tokens, state->forward.host_picks->scores,
			branches, branch_count, drafts, requested, limit, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	}
	return SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS;
}

static const SparkDrafterModuleInterface axk2_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"axk2-dspark",
	Axk2Open,
	Axk2Ingest,
	Axk2Propose,
	Axk2Close,
};

static SparkStatus Axk2Readback(void *host, const void *device, size_t bytes)
{
	if ( SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &axk2_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_axk2_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	Axk2State *state = (Axk2State *)opaque;
	const Axk2Sequence *sequence = state != 0 ? Axk2Live(state, sequence_id) : 0;
	SparkStatus status;
	if ( sequence == 0 || layer >= AXK2_LAYERS || keys == 0 || values == 0 || positions == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memcpy(positions, sequence->positions, sizeof(sequence->positions));
	status = Axk2Readback(keys, Axk2Keys(sequence, layer), AXK2_LAYER_KV * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = Axk2Readback(values, Axk2Values(sequence, layer), AXK2_LAYER_KV * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_axk2_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	Axk2State *state = (Axk2State *)opaque;
	Axk2Sequence *sequence = state != 0 ? Axk2Live(state, sequence_id) : 0;
	const uint64_t started = SparkMonotonicNs(), first = position > AXK2_WINDOW ? position - AXK2_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	for ( uint64_t at = first; at < position; at++ )
		if ( Axk2Held(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = Axk2Prepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Axk2RunForward(state, sequence, AXK2_LAYERS);
	status = Axk2Settle(state, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = Axk2Readback(y, state->forward.y, (size_t)AXK2_BLOCK * AXK2_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_axk2_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	Axk2State *state = (Axk2State *)opaque;
	Axk2Sequence *sequence = state != 0 ? Axk2Live(state, sequence_id) : 0;
	const uint64_t first = position > AXK2_WINDOW ? position - AXK2_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 || hidden == 0 || layers == 0u || layers > AXK2_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint64_t at = first; at < position; at++ )
		if ( Axk2Held(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = Axk2Prepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Axk2RunForward(state, sequence, layers);
	status = Axk2Settle(state, status);
	if ( status == SPARK_STATUS_OK )
		status = Axk2Readback(hidden, state->forward.hidden, (size_t)AXK2_BLOCK * AXK2_HIDDEN * sizeof(uint16_t));
	return status;
}
