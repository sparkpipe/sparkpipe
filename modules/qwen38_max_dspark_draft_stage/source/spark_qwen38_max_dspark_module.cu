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

#define Q38M_HIDDEN 8192u
#define Q38M_TAPS 6u
#define Q38M_TAP_WIDTH (Q38M_TAPS * Q38M_HIDDEN)
#define Q38M_LAYERS 5u
#define Q38M_HEADS 64u
#define Q38M_KV_HEADS 8u
#define Q38M_GROUP (Q38M_HEADS / Q38M_KV_HEADS)
#define Q38M_HEAD_DIM 128u
#define Q38M_HALF (Q38M_HEAD_DIM / 2u)
#define Q38M_Q_WIDTH (Q38M_HEADS * Q38M_HEAD_DIM)
#define Q38M_KV_WIDTH (Q38M_KV_HEADS * Q38M_HEAD_DIM)
#define Q38M_MLP 16384u
#define Q38M_BLOCK 7u
#define Q38M_SLIDING 8192u
#define Q38M_WINDOW (Q38M_SLIDING - Q38M_BLOCK)
#define Q38M_COLUMNS (Q38M_WINDOW + Q38M_BLOCK)
#define Q38M_VOCAB 248320u
#define Q38M_MARKOV 256u
#define Q38M_CONFIDENCE (Q38M_HIDDEN + Q38M_MARKOV)
#define Q38M_CANDIDATES 2048u
#define Q38M_SAMPLED 32u
#define Q38M_BRANCHES 2u
#define Q38M_BRANCH_LEN 4u
#define Q38M_INGEST_ROWS 256u
#define Q38M_THREADS 256u
#define Q38M_WARP 32u
#define Q38M_WARPS (Q38M_THREADS / Q38M_WARP)
#define Q38M_PICK_THREADS 1024u
#define Q38M_TOPK_THREADS 512u
#define Q38M_SPLIT_KEYS 256u
#define Q38M_SPLITS ((Q38M_WINDOW + Q38M_SPLIT_KEYS - 1u) / Q38M_SPLIT_KEYS)
#define Q38M_ATTN_TILE 32u
#define Q38M_ATTN_THREADS (Q38M_BLOCK * Q38M_WARP)
#define Q38M_SEQUENCES_MAX 256u
#define Q38M_LAYER_KV ((uint64_t)Q38M_KV_HEADS * Q38M_COLUMNS * Q38M_HEAD_DIM)
#define Q38M_SEQUENCE_ELEMENTS (2u * Q38M_LAYERS * Q38M_LAYER_KV)
#define Q38M_SEQUENCE_BYTES (Q38M_SEQUENCE_ELEMENTS * sizeof(uint16_t))
#define Q38M_PACK_MAGIC 0x4B505244u
#define Q38M_PACK_VERSION 1u
#define Q38M_PACK_ALIGNMENT 128u
#define Q38M_PI 3.14159265358979323846
#define Q38M_INVERSE_PI 0.31830988618379067154

typedef struct Q38MLayer
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
} Q38MLayer;

typedef struct Q38MSequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t last_used;
	uint32_t live;
	uint32_t retired;
	int64_t positions[Q38M_WINDOW];
	uint32_t tokens[Q38M_WINDOW];
	uint16_t *cache;
} Q38MSequence;

typedef struct Q38MDraft
{
	uint64_t seed;
	uint32_t positions[Q38M_BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint32_t start;
	uint32_t block_at;
	uint32_t masked;
	uint8_t valid[Q38M_WINDOW];
} Q38MDraft;

typedef struct Q38MPicks
{
	uint32_t tokens[Q38M_BLOCK];
	float scores[Q38M_BLOCK];
	uint32_t alt_tokens[Q38M_BLOCK];
	float alt_scores[Q38M_BLOCK];
	uint32_t branch_fork[Q38M_BRANCHES];
	uint32_t branch_len[Q38M_BRANCHES];
	uint32_t branch_tokens[Q38M_BRANCHES][Q38M_BRANCH_LEN];
	float branch_scores[Q38M_BRANCHES][Q38M_BRANCH_LEN];
} Q38MPicks;

typedef struct Q38MForward
{
	Q38MDraft *host_draft;
	Q38MDraft *draft;
	Q38MPicks *host_picks;
	Q38MPicks *picks;
	uint16_t *hidden;
	uint16_t *x;
	uint16_t *q;
	uint16_t *q_roped;
	uint16_t *k;
	uint16_t *v;
	float *partials;
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
} Q38MForward;

typedef struct Q38MState
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
	const uint16_t *markov_previous;
	const uint16_t *markov_next;
	const uint16_t *confidence_weight;
	const uint16_t *confidence_bias;
	Q38MLayer layers[Q38M_LAYERS];
	Q38MForward forward;
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
	Q38MSequence *sequences;
	uint64_t frames;
	uint64_t rows;
	uint64_t gaps;
} Q38MState;

static __device__ __forceinline__ void Q38MRotation(uint32_t position, float inverse, float gain, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * Q38M_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	float s, c;
	sincospif(reduced, &s, &c);
	*cosine = c * gain;
	*sine = s * gain;
}

static __device__ __forceinline__ float Q38MWarpSum(float value)
{
	for ( uint32_t mask = Q38M_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ float Q38MWarpMax(float value)
{
	for ( uint32_t mask = Q38M_WARP / 2u; mask > 0u; mask >>= 1u )
		value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
	return value;
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MYarnTableKernel(float *__restrict__ inverse, double base, double factor, double original, double fast, double slow)
{
	if ( threadIdx.x < Q38M_HALF )
	{
		const double dimension = (double)Q38M_HEAD_DIM;
		double low = floor(dimension * log(original / (fast * 2.0 * Q38M_PI)) / (2.0 * log(base)));
		double high = ceil(dimension * log(original / (slow * 2.0 * Q38M_PI)) / (2.0 * log(base)));
		const double extrapolated = 1.0 / pow(base, (2.0 * (double)threadIdx.x) / dimension);
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

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MRmsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[Q38M_WARPS];
	const uint64_t row = blockIdx.x;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += Q38M_THREADS )
	{
		const float value = LmBf16ToFloat(input[row * dimension + index]);
		total += value * value;
	}
	total = LmBlockSum<Q38M_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += Q38M_THREADS )
	{
		const float normalized = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(input[row * dimension + index]) * scale));
		output[row * dimension + index] = LmFloatToBf16(normalized * LmBf16ToFloat(weight[index]));
	}
}

__global__ __launch_bounds__(Q38M_HEAD_DIM, 1)
void Q38MHeadNormRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse, float rope_gain,
	const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, uint16_t *__restrict__ output, uint16_t *__restrict__ cache_keys,
	uint32_t block_at, uint32_t use_ring)
{
	__shared__ float reduction[Q38M_HEAD_DIM / Q38M_WARP];
	__shared__ float normed[Q38M_HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = LmBf16ToFloat(input[((uint64_t)row * heads + head) * Q38M_HEAD_DIM + lane]);
	float scale, cosine, sine, rotated;
	scale = rsqrtf(LmBlockSum<Q38M_HEAD_DIM>(value * value, reduction) / (float)Q38M_HEAD_DIM + epsilon);
	normed[lane] = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value * scale)) * LmBf16ToFloat(weight[lane])));
	__syncthreads();
	Q38MRotation(position, inverse[lane % Q38M_HALF], rope_gain, &cosine, &sine);
	rotated = lane < Q38M_HALF ? normed[lane] * cosine - normed[lane + Q38M_HALF] * sine : normed[lane] * cosine + normed[lane - Q38M_HALF] * sine;
	if ( cache_keys != 0 )
	{
		const uint32_t column = use_ring != 0u ? position % Q38M_WINDOW : block_at + row;
		cache_keys[((uint64_t)head * Q38M_COLUMNS + column) * Q38M_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	}
	else
		output[((uint64_t)row * heads + head) * Q38M_HEAD_DIM + lane] = LmFloatToBf16(rotated);
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MStoreValuesKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions, uint32_t block_at, uint16_t *__restrict__ cache)
{
	const uint32_t row = blockIdx.x, column = positions != 0 ? positions[row] % Q38M_WINDOW : block_at + row;
	for ( uint32_t index = threadIdx.x; index < Q38M_KV_WIDTH; index += Q38M_THREADS )
		cache[((uint64_t)(index / Q38M_HEAD_DIM) * Q38M_COLUMNS + column) * Q38M_HEAD_DIM + index % Q38M_HEAD_DIM] = values[(uint64_t)row * Q38M_KV_WIDTH + index];
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MEmbedKernel(const uint16_t *__restrict__ embed, const Q38MDraft *__restrict__ draft, uint16_t *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	for ( uint32_t index = threadIdx.x; index < Q38M_HIDDEN; index += Q38M_THREADS )
		hidden[(uint64_t)row * Q38M_HIDDEN + index] = embed[(uint64_t)token * Q38M_HIDDEN + index];
}

__global__ __launch_bounds__(Q38M_ATTN_THREADS, 1)
void Q38MAttentionSplitKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ keys, const uint16_t *__restrict__ values,
	const Q38MDraft *__restrict__ draft, float scale, float *__restrict__ partials)
{
	__shared__ uint32_t s_k[Q38M_ATTN_TILE][Q38M_HALF + 1u];
	__shared__ uint32_t s_v[Q38M_ATTN_TILE][Q38M_HALF + 1u];
	__shared__ float s_scores[Q38M_BLOCK][Q38M_ATTN_TILE];
	__shared__ uint8_t s_vis[Q38M_ATTN_TILE];
	const uint32_t head = blockIdx.x, kv_head = head / Q38M_GROUP, split = blockIdx.y;
	const uint32_t warp = threadIdx.x / Q38M_WARP, lane = threadIdx.x % Q38M_WARP;
	const uint32_t first = draft->start + split * Q38M_SPLIT_KEYS, end = draft->block_at;
	const uint32_t stop = first + Q38M_SPLIT_KEYS < end ? first + Q38M_SPLIT_KEYS : end;
	const uint32_t tiles = first < stop ? (stop - first + Q38M_ATTN_TILE - 1u) / Q38M_ATTN_TILE : 0u;
	const uint32_t masked = draft->masked;
	const uint16_t *head_keys = keys + (uint64_t)kv_head * Q38M_COLUMNS * Q38M_HEAD_DIM;
	const uint16_t *head_values = values + (uint64_t)kv_head * Q38M_COLUMNS * Q38M_HEAD_DIM;
	const uint16_t *q_row = q + ((uint64_t)warp * Q38M_HEADS + head) * Q38M_HEAD_DIM;
	const uint32_t qp0 = *(const uint32_t *)(q_row + 2u * lane);
	const uint32_t qp1 = *(const uint32_t *)(q_row + Q38M_HALF + 2u * lane);
	const float q0 = LmBf16ToFloat((uint16_t)(qp0 & 0xffffu));
	const float q1 = LmBf16ToFloat((uint16_t)(qp0 >> 16u));
	const float q2 = LmBf16ToFloat((uint16_t)(qp1 & 0xffffu));
	const float q3 = LmBf16ToFloat((uint16_t)(qp1 >> 16u));
	float running_max = -INFINITY, running_sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
	float *out = partials + (((uint64_t)head * Q38M_SPLITS + split) * Q38M_BLOCK + warp) * (Q38M_HEAD_DIM + 2u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t tile_first = first + tile * Q38M_ATTN_TILE;
		for ( uint32_t item = threadIdx.x; item < Q38M_ATTN_TILE * 16u; item += Q38M_ATTN_THREADS )
		{
			const uint32_t load_key = item >> 4u, load_chunk = item & 15u;
			const uint32_t col = tile_first + load_key;
			if ( col < stop )
			{
				const uint4 kv_k = *(const uint4 *)(head_keys + (uint64_t)col * Q38M_HEAD_DIM + load_chunk * 8u);
				const uint4 kv_v = *(const uint4 *)(head_values + (uint64_t)col * Q38M_HEAD_DIM + load_chunk * 8u);
				s_k[load_key][load_chunk * 4u + 0u] = kv_k.x;
				s_k[load_key][load_chunk * 4u + 1u] = kv_k.y;
				s_k[load_key][load_chunk * 4u + 2u] = kv_k.z;
				s_k[load_key][load_chunk * 4u + 3u] = kv_k.w;
				s_v[load_key][load_chunk * 4u + 0u] = kv_v.x;
				s_v[load_key][load_chunk * 4u + 1u] = kv_v.y;
				s_v[load_key][load_chunk * 4u + 2u] = kv_v.z;
				s_v[load_key][load_chunk * 4u + 3u] = kv_v.w;
			}
			else
			{
				s_k[load_key][load_chunk * 4u + 0u] = 0u;
				s_k[load_key][load_chunk * 4u + 1u] = 0u;
				s_k[load_key][load_chunk * 4u + 2u] = 0u;
				s_k[load_key][load_chunk * 4u + 3u] = 0u;
				s_v[load_key][load_chunk * 4u + 0u] = 0u;
				s_v[load_key][load_chunk * 4u + 1u] = 0u;
				s_v[load_key][load_chunk * 4u + 2u] = 0u;
				s_v[load_key][load_chunk * 4u + 3u] = 0u;
			}
		}
		if ( threadIdx.x < Q38M_ATTN_TILE )
			s_vis[threadIdx.x] = tile_first + threadIdx.x < stop && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
		__syncthreads();
		for ( uint32_t t = 0u; t < Q38M_ATTN_TILE; t++ )
		{
			const uint32_t kp0 = s_k[t][lane];
			const uint32_t kp1 = s_k[t][Q38M_WARP + lane];
			const float d0 = fmaf(q0, LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1 * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
			const float d1 = fmaf(q2, LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), q3 * LmBf16ToFloat((uint16_t)(kp1 >> 16u)));
			const float dot = Q38MWarpSum(d0 + d1);
			if ( lane == 0u )
				s_scores[warp][t] = s_vis[t] != 0u ? dot * scale : -INFINITY;
		}
		__syncwarp();
		{
			const float sc = s_scores[warp][lane];
			const float tile_max = Q38MWarpMax(sc);
			const float next_max = fmaxf(running_max, tile_max);
			const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
			const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
			const float tile_sum = Q38MWarpSum(weight);
			running_sum = running_sum * rescale + tile_sum;
			running_max = next_max;
			acc0 *= rescale;
			acc1 *= rescale;
			acc2 *= rescale;
			acc3 *= rescale;
			s_scores[warp][lane] = weight;
		}
		__syncwarp();
		for ( uint32_t t = 0u; t < Q38M_ATTN_TILE; t++ )
		{
			const float wt = s_scores[warp][t];
			const uint32_t vp0 = s_v[t][lane];
			const uint32_t vp1 = s_v[t][Q38M_WARP + lane];
			acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0);
			acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1);
			acc2 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2);
			acc3 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3);
		}
		__syncthreads();
	}
	out[2u * lane] = acc0;
	out[2u * lane + 1u] = acc1;
	out[Q38M_HALF + 2u * lane] = acc2;
	out[Q38M_HALF + 2u * lane + 1u] = acc3;
	if ( lane == 0u )
	{
		out[Q38M_HEAD_DIM] = running_max;
		out[Q38M_HEAD_DIM + 1u] = running_sum;
	}
}

__global__ __launch_bounds__(Q38M_HEAD_DIM, 1)
void Q38MAttentionMergeKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ keys, const uint16_t *__restrict__ values,
	const Q38MDraft *__restrict__ draft, float scale, const float *__restrict__ partials, uint32_t splits, uint16_t *__restrict__ output)
{
	__shared__ float reduction[Q38M_HEAD_DIM / Q38M_WARP];
	const uint32_t row = blockIdx.x / Q38M_HEADS, head = blockIdx.x % Q38M_HEADS, kv_head = head / Q38M_GROUP, lane = threadIdx.x;
	const uint32_t block_at = draft->block_at;
	const float own = LmBf16ToFloat(q[((uint64_t)row * Q38M_HEADS + head) * Q38M_HEAD_DIM + lane]);
	const uint16_t *head_keys = keys + (uint64_t)kv_head * Q38M_COLUMNS * Q38M_HEAD_DIM;
	const uint16_t *head_values = values + (uint64_t)kv_head * Q38M_COLUMNS * Q38M_HEAD_DIM;
	float top = -INFINITY, sum = 0.0f, accumulated = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part = partials + (((uint64_t)head * Q38M_SPLITS + split) * Q38M_BLOCK + row) * (Q38M_HEAD_DIM + 2u);
		const float part_top = part[Q38M_HEAD_DIM], next = fmaxf(top, part_top);
		float carry, weight;
		if ( part_top == -INFINITY )
			continue;
		carry = top == -INFINITY ? 0.0f : expf(top - next);
		weight = expf(part_top - next);
		sum = sum * carry + part[Q38M_HEAD_DIM + 1u] * weight;
		accumulated = accumulated * carry + part[lane] * weight;
		top = next;
	}
	for ( uint32_t key = 0u; key < Q38M_BLOCK; key++ )
	{
		const float score = LmBlockSum<Q38M_HEAD_DIM>(own * LmBf16ToFloat(head_keys[(uint64_t)(block_at + key) * Q38M_HEAD_DIM + lane]), reduction) * scale;
		const float next = fmaxf(top, score), carry = top == -INFINITY ? 0.0f : expf(top - next), weight = expf(score - next);
		sum = sum * carry + weight;
		accumulated = accumulated * carry + weight * LmBf16ToFloat(head_values[(uint64_t)(block_at + key) * Q38M_HEAD_DIM + lane]);
		top = next;
	}
	output[(uint64_t)row * Q38M_Q_WIDTH + head * Q38M_HEAD_DIM + lane] = LmFloatToBf16(accumulated / sum);
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MResidualKernel(const uint16_t *__restrict__ base, const uint16_t *__restrict__ addend, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * Q38M_THREADS + threadIdx.x; index < count; index += gridDim.x * Q38M_THREADS )
		output[index] = LmFloatToBf16(LmBf16ToFloat(base[index]) + LmBf16ToFloat(addend[index]));
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MSwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * Q38M_THREADS + threadIdx.x; index < count; index += gridDim.x * Q38M_THREADS )
	{
		const float g = LmBf16ToFloat(gate[index]);
		output[index] = LmFloatToBf16(g / (1.0f + expf(-g)) * LmBf16ToFloat(up[index]));
	}
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MWidenKernel(const uint16_t *__restrict__ input, float *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * Q38M_THREADS + threadIdx.x; index < count; index += gridDim.x * Q38M_THREADS )
		output[index] = LmBf16ToFloat(input[index]);
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MMarkovRowKernel(uint32_t step, uint32_t anchor, const uint32_t *__restrict__ tokens, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ row)
{
	__shared__ float vector[Q38M_MARKOV];
	const uint32_t previous = step == 0u ? anchor : tokens[step - 1u];
	const uint32_t warp = threadIdx.x / Q38M_WARP, lane = threadIdx.x % Q38M_WARP;
	for ( uint32_t index = threadIdx.x; index < Q38M_MARKOV; index += Q38M_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * Q38M_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * Q38M_WARPS + warp; slot < Q38M_CANDIDATES; slot += gridDim.x * Q38M_WARPS )
	{
		const uint32_t candidate = candidates[step * Q38M_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * Q38M_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < Q38M_MARKOV; element += Q38M_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = Q38MWarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * Q38M_VOCAB + candidate] + sum;
	}
}

static __device__ __forceinline__ unsigned long long Q38MRowKey(const float *row, uint32_t slot)
{
	return ((unsigned long long)SparkMathSampleOrdered(row[slot]) << 32u) | (unsigned long long)(0xffffffffu - slot);
}

static __device__ __forceinline__ uint32_t Q38MKeySlot(unsigned long long key)
{
	return 0xffffffffu - (uint32_t)key;
}

static __device__ __forceinline__ void Q38MTop2Insert(unsigned long long candidate, unsigned long long *first, unsigned long long *second)
{
	if ( candidate > *first )
	{
		*second = *first;
		*first = candidate;
	}
	else if ( candidate > *second && candidate != *first )
		*second = candidate;
}

__global__ __launch_bounds__(Q38M_PICK_THREADS, 1)
void Q38MMarkovPickKernel(uint32_t step, uint32_t anchor, uint32_t position, unsigned long long seed, float inverse_temperature, float nucleus,
	uint32_t top_k, const float *__restrict__ row, const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ y,
	const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ confidence_weight, const uint16_t *__restrict__ confidence_bias,
	Q38MPicks *__restrict__ picks)
{
	__shared__ unsigned long long keys[Q38M_CANDIDATES];
	__shared__ unsigned long long best[Q38M_PICK_THREADS / Q38M_WARP];
	__shared__ unsigned long long second_best[Q38M_PICK_THREADS / Q38M_WARP];
	__shared__ float reduction[Q38M_PICK_THREADS / Q38M_WARP];
	__shared__ uint32_t chosen;
	__shared__ uint32_t alt_chosen;
	__shared__ float peak_scaled;
	const uint32_t previous = step == 0u ? anchor : picks->tokens[step - 1u];
	const uint32_t warp = threadIdx.x / Q38M_WARP, lane = threadIdx.x % Q38M_WARP;
	const float scale = inverse_temperature > 0.0f ? inverse_temperature : 1.0f;
	float dot = 0.0f, denom = 0.0f;
	if ( inverse_temperature > 0.0f )
	{
		for ( uint32_t index = threadIdx.x; index < Q38M_CANDIDATES; index += Q38M_PICK_THREADS )
			keys[index] = Q38MRowKey(row, index);
		__syncthreads();
		for ( uint32_t size = 2u; size <= Q38M_CANDIDATES; size <<= 1u )
			for ( uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u )
			{
				for ( uint32_t index = threadIdx.x; index < Q38M_CANDIDATES; index += Q38M_PICK_THREADS )
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
			const uint32_t slot = Q38MKeySlot(keys[lane]);
			const uint32_t token = candidates[step * Q38M_CANDIDATES + slot];
			const float scaled = row[slot] * inverse_temperature;
			float peak = Q38MWarpMax(scaled), probability = expf(scaled - peak), total = Q38MWarpSum(probability), running, keyed;
			uint32_t winner, runner;
			unsigned long long key;
			probability /= total;
			running = probability;
			for ( uint32_t offset = 1u; offset < Q38M_SAMPLED; offset <<= 1u )
			{
				const float earlier = __shfl_up_sync(0xffffffffu, running, offset);
				if ( lane >= offset )
					running += earlier;
			}
			keyed = (((nucleus <= 0.0f) || (running - probability < nucleus)) && (top_k == 0u || lane < top_k)) || lane == 0u
				? scaled + SparkMathGumbelNoise(seed, position + step, token) : -INFINITY;
			key = ((unsigned long long)SparkMathSampleOrdered(keyed) << 32u) | (unsigned long long)(0xffffffffu - lane);
			for ( uint32_t mask = Q38M_WARP / 2u; mask > 0u; mask >>= 1u )
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
		for ( uint32_t index = threadIdx.x; index < Q38M_CANDIDATES; index += Q38M_PICK_THREADS )
			Q38MTop2Insert(Q38MRowKey(row, index), &first, &second);
		for ( uint32_t mask = Q38M_WARP / 2u; mask > 0u; mask >>= 1u )
		{
			const unsigned long long other_first = __shfl_xor_sync(0xffffffffu, first, mask);
			const unsigned long long other_second = __shfl_xor_sync(0xffffffffu, second, mask);
			Q38MTop2Insert(other_first, &first, &second);
			Q38MTop2Insert(other_second, &first, &second);
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
			for ( uint32_t index = 1u; index < Q38M_PICK_THREADS / Q38M_WARP; index++ )
			{
				Q38MTop2Insert(best[index], &first, &second);
				Q38MTop2Insert(second_best[index], &first, &second);
			}
			chosen = Q38MKeySlot(first);
			alt_chosen = Q38MKeySlot(second);
			peak_scaled = row[chosen];
		}
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < Q38M_CANDIDATES; index += Q38M_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<Q38M_PICK_THREADS>(denom, reduction);
	for ( uint32_t index = threadIdx.x; index < Q38M_CONFIDENCE; index += Q38M_PICK_THREADS )
	{
		const float feature = index < Q38M_HIDDEN ? LmBf16ToFloat(y[(uint64_t)step * Q38M_HIDDEN + index])
			: LmBf16ToFloat(markov_previous[(uint64_t)previous * Q38M_MARKOV + index - Q38M_HIDDEN]);
		dot += LmBf16ToFloat(confidence_weight[index]) * feature;
	}
	dot = LmBlockSum<Q38M_PICK_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->tokens[step] = candidates[step * Q38M_CANDIDATES + chosen];
		picks->scores[step] = 1.0f / (1.0f + expf(-(dot + LmBf16ToFloat(confidence_bias[0]))));
		picks->alt_tokens[step] = candidates[step * Q38M_CANDIDATES + alt_chosen];
		picks->alt_scores[step] = denom > 0.0f ? expf(row[alt_chosen] * scale - peak_scaled) / denom : 0.0f;
	}
}

__global__ __launch_bounds__(Q38M_WARP, 1)
void Q38MBranchSelectKernel(Q38MPicks *__restrict__ picks, uint32_t drafts, float min_score)
{
	uint32_t used = 0u;
	if ( threadIdx.x != 0u )
		return;
	for ( uint32_t branch = 0u; branch < Q38M_BRANCHES; branch++ )
	{
		picks->branch_fork[branch] = Q38M_BLOCK;
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
			if ( pos < Q38M_BRANCHES )
			{
				picks->branch_fork[pos] = picks->branch_fork[pos - 1u];
				picks->branch_len[pos] = picks->branch_len[pos - 1u];
				picks->branch_tokens[pos][0] = picks->branch_tokens[pos - 1u][0];
				picks->branch_scores[pos][0] = picks->branch_scores[pos - 1u][0];
			}
			pos--;
		}
		if ( pos < Q38M_BRANCHES )
		{
			picks->branch_fork[pos] = step;
			picks->branch_len[pos] = 1u;
			picks->branch_tokens[pos][0] = picks->alt_tokens[step];
			picks->branch_scores[pos][0] = score;
			if ( used < Q38M_BRANCHES )
				used++;
		}
	}
}

__global__ __launch_bounds__(Q38M_THREADS, 1)
void Q38MBranchRowKernel(uint32_t offset, uint32_t drafts, const Q38MPicks *__restrict__ picks, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ rows)
{
	__shared__ float vector[Q38M_MARKOV];
	const uint32_t branch = blockIdx.y, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / Q38M_WARP, lane = threadIdx.x % Q38M_WARP;
	uint32_t previous;
	float *row;
	if ( fork >= drafts || step >= drafts )
		return;
	previous = picks->branch_tokens[branch][offset - 1u];
	row = rows + (uint64_t)branch * Q38M_CANDIDATES;
	for ( uint32_t index = threadIdx.x; index < Q38M_MARKOV; index += Q38M_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * Q38M_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * Q38M_WARPS + warp; slot < Q38M_CANDIDATES; slot += gridDim.x * Q38M_WARPS )
	{
		const uint32_t candidate = candidates[step * Q38M_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * Q38M_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < Q38M_MARKOV; element += Q38M_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = Q38MWarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * Q38M_VOCAB + candidate] + sum;
	}
}

__global__ __launch_bounds__(Q38M_PICK_THREADS, 1)
void Q38MBranchPickKernel(uint32_t offset, uint32_t drafts, float scale, const float *__restrict__ rows, const uint32_t *__restrict__ candidates,
	Q38MPicks *__restrict__ picks)
{
	__shared__ unsigned long long best[Q38M_PICK_THREADS / Q38M_WARP];
	__shared__ float reduction[Q38M_PICK_THREADS / Q38M_WARP];
	__shared__ uint32_t chosen;
	__shared__ float peak_scaled;
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / Q38M_WARP, lane = threadIdx.x % Q38M_WARP;
	const float *row = rows + (uint64_t)branch * Q38M_CANDIDATES;
	unsigned long long first = 0ull;
	float denom = 0.0f;
	if ( fork >= drafts || step >= drafts )
		return;
	for ( uint32_t index = threadIdx.x; index < Q38M_CANDIDATES; index += Q38M_PICK_THREADS )
	{
		const unsigned long long key = Q38MRowKey(row, index);
		if ( key > first )
			first = key;
	}
	for ( uint32_t mask = Q38M_WARP / 2u; mask > 0u; mask >>= 1u )
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
		for ( uint32_t index = 1u; index < Q38M_PICK_THREADS / Q38M_WARP; index++ )
			if ( best[index] > first )
				first = best[index];
		chosen = Q38MKeySlot(first);
		peak_scaled = row[chosen] * scale;
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < Q38M_CANDIDATES; index += Q38M_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<Q38M_PICK_THREADS>(denom, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->branch_tokens[branch][offset] = candidates[step * Q38M_CANDIDATES + chosen];
		picks->branch_scores[branch][offset] = denom > 0.0f ? 1.0f / denom : 0.0f;
		picks->branch_len[branch] = offset + 1u;
	}
}

static SparkStatus Q38MRefuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	SPARK_FAIL(status);
}

static SparkStatus Q38MLoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
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
	if ( magic != Q38M_PACK_MAGIC || version != Q38M_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
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
	*payload_base = (sizeof(header) + bytes + Q38M_PACK_ALIGNMENT - 1u) / Q38M_PACK_ALIGNMENT * Q38M_PACK_ALIGNMENT;
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38MTensor(const Q38MState *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t expected,
	const uint16_t **pointer)
{
	const int32_t tensors = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "tensors");
	const int32_t entry = tensors < 0 ? -1 : SparkJsonFindObjectMember(document, tensors, name);
	const int32_t offset_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "offset");
	const int32_t bytes_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "bytes");
	uint64_t offset = 0u, bytes = 0u;
	const void *slice = 0;
	if ( offset_token < 0 || bytes_token < 0 || SparkJsonGetUInt64(document, offset_token, &offset) != SPARK_STATUS_OK ||
		SparkJsonGetUInt64(document, bytes_token, &bytes) != SPARK_STATUS_OK || bytes != expected * sizeof(uint16_t) )
	{
		fprintf(stderr, "qwen38-max dspark: pack tensor %s is missing or is not %llu bf16 elements\n", name, (unsigned long long)expected);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*pointer = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static int32_t Q38MMember(const SparkJsonDocument *document, int32_t object, const char *name)
{
	return object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
}

static SparkStatus Q38MShape(Q38MState *state, const SparkJsonDocument *document)
{
	const int32_t config = Q38MMember(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = Q38MMember(document, config, "checkpoint_config");
	const int32_t dflash = Q38MMember(document, checkpoint, "dflash_config");
	const int32_t rope = Q38MMember(document, checkpoint, "rope_parameters");
	const int32_t taps = Q38MMember(document, dflash, "target_layer_ids");
	const struct
	{
		int32_t object;
		const char *name;
		uint64_t value;
	} expected[] = {
		{checkpoint, "block_size", Q38M_BLOCK},
		{checkpoint, "hidden_size", Q38M_HIDDEN},
		{checkpoint, "intermediate_size", Q38M_MLP},
		{checkpoint, "num_hidden_layers", Q38M_LAYERS},
		{checkpoint, "num_attention_heads", Q38M_HEADS},
		{checkpoint, "num_key_value_heads", Q38M_KV_HEADS},
		{checkpoint, "head_dim", Q38M_HEAD_DIM},
		{checkpoint, "vocab_size", Q38M_VOCAB},
		{checkpoint, "markov_rank", Q38M_MARKOV},
	};
	const struct
	{
		int32_t object;
		const char *name;
		bool value;
	} flags[] = {
		{checkpoint, "enable_confidence_head", true},
		{checkpoint, "confidence_head_with_markov", true},
		{checkpoint, "use_sliding_window", false},
	};
	uint64_t value = 0u;
	float theta = 0.0f, factor = 0.0f, original = 0.0f, fast = 0.0f, slow = 0.0f, attention_factor = 0.0f;
	int32_t token;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		token = Q38MMember(document, expected[index].object, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "qwen38-max dspark: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	for ( uint32_t index = 0u; index < sizeof(flags) / sizeof(flags[0]); index++ )
	{
		bool flag = false;
		token = Q38MMember(document, flags[index].object, flags[index].name);
		if ( token < 0 || SparkJsonGetBoolean(document, token, &flag) != SPARK_STATUS_OK || flag != flags[index].value )
		{
			fprintf(stderr, "qwen38-max dspark: checkpoint_config %s does not match module contract\n", flags[index].name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	token = Q38MMember(document, checkpoint, "markov_head_type");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "vanilla") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Q38MMember(document, dflash, "projector_type");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "dspark") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Q38MMember(document, dflash, "attention_mode");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "gqa") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( taps < 0 || SparkJsonGetArrayElementCount(document, taps) != Q38M_TAPS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Q38MMember(document, dflash, "mask_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= Q38M_VOCAB )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->mask_token = (uint32_t)value;
	token = Q38MMember(document, checkpoint, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK || state->epsilon <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Q38MMember(document, rope, "rope_type");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "yarn") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( Q38MMember(document, rope, "rope_theta") < 0 || SparkJsonGetFloat(document, Q38MMember(document, rope, "rope_theta"), &theta) != SPARK_STATUS_OK ||
		theta <= 0.0f || Q38MMember(document, rope, "factor") < 0 || SparkJsonGetFloat(document, Q38MMember(document, rope, "factor"), &factor) != SPARK_STATUS_OK ||
		factor <= 0.0f || Q38MMember(document, rope, "original_max_position_embeddings") < 0 ||
		SparkJsonGetFloat(document, Q38MMember(document, rope, "original_max_position_embeddings"), &original) != SPARK_STATUS_OK || original <= 0.0f ||
		Q38MMember(document, rope, "beta_fast") < 0 || SparkJsonGetFloat(document, Q38MMember(document, rope, "beta_fast"), &fast) != SPARK_STATUS_OK ||
		fast <= 0.0f || Q38MMember(document, rope, "beta_slow") < 0 ||
		SparkJsonGetFloat(document, Q38MMember(document, rope, "beta_slow"), &slow) != SPARK_STATUS_OK || slow <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = Q38MMember(document, rope, "attention_factor");
	if ( token >= 0 && SparkJsonGetFloat(document, token, &attention_factor) == SPARK_STATUS_OK && attention_factor > 0.0f )
		state->rope_gain = attention_factor;
	else
		state->rope_gain = (float)(factor > 1.0f ? 0.1 * log((double)factor) + 1.0 : 1.0);
	state->softmax_scale = (float)(1.0 / sqrt((double)Q38M_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, Q38M_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	LM_LAUNCH((Q38MYarnTableKernel), 1u, Q38M_THREADS, 0, state->stream, state->rope_inverse, (double)theta, (double)factor, (double)original,
		(double)fast, (double)slow);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38MWeights(Q38MState *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = Q38MLoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = Q38MShape(state, &document);
	{
		const struct
		{
			const char *name;
			uint64_t elements;
			const uint16_t **pointer;
		} plain[] = {
			{"fc.weight", (uint64_t)Q38M_HIDDEN * Q38M_TAP_WIDTH, &state->fc},
			{"hidden_norm.weight", Q38M_HIDDEN, &state->hidden_norm},
			{"norm.weight", Q38M_HIDDEN, &state->norm},
			{"target.embed", (uint64_t)Q38M_VOCAB * Q38M_HIDDEN, &state->embed},
			{"target.head", (uint64_t)Q38M_VOCAB * Q38M_HIDDEN, &state->head},
			{"markov_head.markov_w1.weight", (uint64_t)Q38M_VOCAB * Q38M_MARKOV, &state->markov_previous},
			{"markov_head.markov_w2.weight", (uint64_t)Q38M_VOCAB * Q38M_MARKOV, &state->markov_next},
			{"confidence_head.proj.weight", Q38M_CONFIDENCE, &state->confidence_weight},
			{"confidence_head.proj.bias", 1u, &state->confidence_bias},
		};
		for ( uint32_t index = 0u; index < sizeof(plain) / sizeof(plain[0]) && status == SPARK_STATUS_OK; index++ )
			status = Q38MTensor(state, &document, payload_base, plain[index].name, plain[index].elements, plain[index].pointer);
	}
	for ( uint32_t layer = 0u; layer < Q38M_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		Q38MLayer *weights = &state->layers[layer];
		const struct
		{
			const char *suffix;
			uint64_t elements;
			const uint16_t **pointer;
		} tensors[] = {
			{"input_layernorm.weight", Q38M_HIDDEN, &weights->input_norm},
			{"post_attention_layernorm.weight", Q38M_HIDDEN, &weights->post_norm},
			{"self_attn.q_proj.weight", (uint64_t)Q38M_Q_WIDTH * Q38M_HIDDEN, &weights->q},
			{"self_attn.k_proj.weight", (uint64_t)Q38M_KV_WIDTH * Q38M_HIDDEN, &weights->k},
			{"self_attn.v_proj.weight", (uint64_t)Q38M_KV_WIDTH * Q38M_HIDDEN, &weights->v},
			{"self_attn.o_proj.weight", (uint64_t)Q38M_HIDDEN * Q38M_Q_WIDTH, &weights->o},
			{"self_attn.q_norm.weight", Q38M_HEAD_DIM, &weights->q_norm},
			{"self_attn.k_norm.weight", Q38M_HEAD_DIM, &weights->k_norm},
			{"mlp.gate_proj.weight", (uint64_t)Q38M_MLP * Q38M_HIDDEN, &weights->gate},
			{"mlp.up_proj.weight", (uint64_t)Q38M_MLP * Q38M_HIDDEN, &weights->up},
			{"mlp.down_proj.weight", (uint64_t)Q38M_HIDDEN * Q38M_MLP, &weights->down},
		};
		for ( uint32_t index = 0u; index < sizeof(tensors) / sizeof(tensors[0]) && status == SPARK_STATUS_OK; index++ )
		{
			char name[128];
			snprintf(name, sizeof(name), "layers.%u.%s", layer, tensors[index].suffix);
			status = Q38MTensor(state, &document, payload_base, name, tensors[index].elements, tensors[index].pointer);
		}
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define Q38M_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus Q38MScratch(Q38MState *state, uint64_t context_bytes)
{
	Q38MForward *forward = &state->forward;
	int multiprocessors = 0;
	uint64_t capacity = context_bytes / Q38M_SEQUENCE_BYTES;
	capacity = capacity < 1u ? 1u : capacity > Q38M_SEQUENCES_MAX ? Q38M_SEQUENCES_MAX : capacity;
	state->capacity = (uint32_t)capacity;
	state->sequences = (Q38MSequence *)calloc(capacity, sizeof(Q38MSequence));
	if ( state->sequences == 0 || SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)Q38M_INGEST_ROWS * Q38M_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)Q38M_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(Q38MDraft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(Q38MPicks)) != SPARK_GPU_SUCCESS ||
		!Q38M_DEVICE(state->caches, capacity * Q38M_SEQUENCE_ELEMENTS) || !Q38M_DEVICE(state->taps, (uint64_t)Q38M_INGEST_ROWS * Q38M_TAP_WIDTH) ||
		!Q38M_DEVICE(state->positions, Q38M_INGEST_ROWS) || !Q38M_DEVICE(state->features_raw, (uint64_t)Q38M_INGEST_ROWS * Q38M_HIDDEN) ||
		!Q38M_DEVICE(state->features, (uint64_t)Q38M_INGEST_ROWS * Q38M_HIDDEN) || !Q38M_DEVICE(state->keys_raw, (uint64_t)Q38M_INGEST_ROWS * Q38M_KV_WIDTH) ||
		!Q38M_DEVICE(state->values_raw, (uint64_t)Q38M_INGEST_ROWS * Q38M_KV_WIDTH) || !Q38M_DEVICE(forward->draft, 1u) || !Q38M_DEVICE(forward->picks, 1u) ||
		!Q38M_DEVICE(forward->hidden, Q38M_BLOCK * Q38M_HIDDEN) || !Q38M_DEVICE(forward->x, Q38M_BLOCK * Q38M_HIDDEN) ||
		!Q38M_DEVICE(forward->q, Q38M_BLOCK * Q38M_Q_WIDTH) || !Q38M_DEVICE(forward->q_roped, Q38M_BLOCK * Q38M_Q_WIDTH) ||
		!Q38M_DEVICE(forward->k, Q38M_BLOCK * Q38M_KV_WIDTH) || !Q38M_DEVICE(forward->v, Q38M_BLOCK * Q38M_KV_WIDTH) ||
		!Q38M_DEVICE(forward->partials, (uint64_t)Q38M_HEADS * Q38M_SPLITS * Q38M_BLOCK * (Q38M_HEAD_DIM + 2u)) ||
		!Q38M_DEVICE(forward->attended, Q38M_BLOCK * Q38M_Q_WIDTH) || !Q38M_DEVICE(forward->projected, Q38M_BLOCK * Q38M_HIDDEN) ||
		!Q38M_DEVICE(forward->residual, Q38M_BLOCK * Q38M_HIDDEN) || !Q38M_DEVICE(forward->gate, Q38M_BLOCK * Q38M_MLP) ||
		!Q38M_DEVICE(forward->up, Q38M_BLOCK * Q38M_MLP) || !Q38M_DEVICE(forward->act, Q38M_BLOCK * Q38M_MLP) ||
		!Q38M_DEVICE(forward->mlp, Q38M_BLOCK * Q38M_HIDDEN) || !Q38M_DEVICE(forward->y, Q38M_BLOCK * Q38M_HIDDEN) ||
		!Q38M_DEVICE(forward->logits, (uint64_t)Q38M_BLOCK * Q38M_VOCAB) || !Q38M_DEVICE(forward->logits_f32, (uint64_t)Q38M_BLOCK * Q38M_VOCAB) ||
		!Q38M_DEVICE(forward->candidates, Q38M_BLOCK * Q38M_CANDIDATES) || !Q38M_DEVICE(forward->markov_row, Q38M_BRANCHES * Q38M_CANDIDATES) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		state->sequences[index].cache = state->caches + (uint64_t)index * Q38M_SEQUENCE_ELEMENTS;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static uint16_t *Q38MKeys(const Q38MSequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer) * Q38M_LAYER_KV;
}

static uint16_t *Q38MValues(const Q38MSequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer + 1u) * Q38M_LAYER_KV;
}

static void Q38MReset(Q38MSequence *sequence)
{
	for ( uint32_t slot = 0u; slot < Q38M_WINDOW; slot++ )
	{
		sequence->positions[slot] = -1;
		sequence->tokens[slot] = 0u;
	}
	sequence->retired = 0u;
}

static Q38MSequence *Q38MLive(Q38MState *state, uint64_t id)
{
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		if ( state->sequences[index].live != 0u && state->sequences[index].retired == 0u && state->sequences[index].id == id )
			return &state->sequences[index];
	return 0;
}

static Q38MSequence *Q38MClaim(Q38MState *state)
{
	Q38MSequence *oldest = 0;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
	{
		Q38MSequence *sequence = &state->sequences[index];
		if ( sequence->live == 0u )
			return sequence;
		if ( oldest == 0 || sequence->last_used < oldest->last_used )
			oldest = sequence;
	}
	return oldest;
}

static uint32_t Q38MHeld(const Q38MSequence *sequence, uint64_t position)
{
	return sequence->positions[position % Q38M_WINDOW] == (int64_t)position ? 1u : 0u;
}

static SparkStatus Q38MGemm(Q38MState *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38MIngestRows(Q38MState *state, Q38MSequence *sequence, uint32_t rows)
{
	SparkStatus status = Q38MGemm(state, state->fc, state->taps, state->features_raw, rows, Q38M_TAP_WIDTH, Q38M_HIDDEN);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((Q38MRmsKernel), rows, Q38M_THREADS, 0, state->stream, state->features_raw, state->hidden_norm, state->features, Q38M_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < Q38M_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = Q38MGemm(state, state->layers[layer].k, state->features, state->keys_raw, rows, Q38M_HIDDEN, Q38M_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38MGemm(state, state->layers[layer].v, state->features, state->values_raw, rows, Q38M_HIDDEN, Q38M_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38MHeadNormRopeKernel), rows * Q38M_KV_HEADS, Q38M_HEAD_DIM, 0, state->stream, state->keys_raw, state->layers[layer].k_norm,
			state->rope_inverse, state->rope_gain, state->positions, Q38M_KV_HEADS, state->epsilon, (uint16_t *)0, Q38MKeys(sequence, layer), 0u, 1u);
		LM_LAUNCH((Q38MStoreValuesKernel), rows, Q38M_THREADS, 0, state->stream, state->values_raw, state->positions, 0u, Q38MValues(sequence, layer));
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static SparkStatus Q38MIngest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	Q38MState *state = (Q38MState *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	const uint32_t skip = rows > Q38M_WINDOW ? rows - Q38M_WINDOW : 0u;
	const uint64_t first = frame->header.first_position + skip;
	Q38MSequence *sequence;
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != Q38M_TAPS || hidden != Q38M_HIDDEN || frame->rows == 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t tap = 0u; tap < Q38M_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * Q38M_HIDDEN > frame->row_elements )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( frame->header.first_position + rows > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = Q38MLive(state, frame->header.sequence_id);
	if ( sequence != 0 && frame->header.first_position != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence->last_used = ++state->clock;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = Q38MClaim(state);
		Q38MReset(sequence);
		sequence->live = 1u;
		sequence->id = frame->header.sequence_id;
	}
	else if ( frame->header.first_position == 0u )
		Q38MReset(sequence);
	sequence->control_generation = frame->header.control_generation;
	sequence->request_id = frame->header.request_id;
	sequence->last_used = ++state->clock;
	for ( uint32_t row = skip; row < rows; row++ )
		sequence->positions[(frame->header.first_position + row) % Q38M_WINDOW] = -1;
	for ( uint32_t done = skip; done < rows && status == SPARK_STATUS_OK; done += Q38M_INGEST_ROWS )
	{
		const uint32_t part = rows - done < Q38M_INGEST_ROWS ? rows - done : Q38M_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < Q38M_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * Q38M_TAP_WIDTH + (uint64_t)tap * Q38M_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * Q38M_HIDDEN, Q38M_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(frame->header.first_position + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * Q38M_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t),
			SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		{
			(void)SparkGpuStreamSynchronize(state->stream);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		status = Q38MIngestRows(state, sequence, part);
	}
	if ( status == SPARK_STATUS_OK )
	{
		for ( uint32_t slot = 0u; slot < Q38M_WINDOW; slot++ )
			if ( sequence->positions[slot] >= (int64_t)first )
				sequence->positions[slot] = -1;
		for ( uint32_t row = skip; row < rows; row++ )
		{
			const uint64_t position = frame->header.first_position + row;
			sequence->positions[position % Q38M_WINDOW] = (int64_t)position;
			sequence->tokens[position % Q38M_WINDOW] = frame->tokens != 0 ? frame->tokens[row] : 0u;
		}
	}
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus Q38MCopySlots(Q38MState *state, Q38MSequence *sequence, const Q38MSequence *donor, uint32_t slot, uint32_t count)
{
	for ( uint32_t layer = 0u; layer < Q38M_LAYERS; layer++ )
		for ( uint32_t head = 0u; head < Q38M_KV_HEADS; head++ )
		{
			const uint64_t at = ((uint64_t)head * Q38M_COLUMNS + slot) * Q38M_HEAD_DIM;
			const size_t bytes = (size_t)count * Q38M_HEAD_DIM * sizeof(uint16_t);
			if ( SparkGpuMemcpyAsync(Q38MKeys(sequence, layer) + at, Q38MKeys(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) !=
				SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(Q38MValues(sequence, layer) + at, Q38MValues(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	return SPARK_STATUS_OK;
}

static uint32_t Q38MUsable(const Q38MSequence *sequence, const Q38MSequence *donor, const SparkDraftServerRequest *request, uint64_t position)
{
	return Q38MHeld(sequence, position) == 0u && Q38MHeld(donor, position) != 0u && position < request->committed_token_count &&
		donor->tokens[position % Q38M_WINDOW] == request->committed_token_ids[position] ? 1u : 0u;
}

static SparkStatus Q38MFillFromDonors(Q38MState *state, Q38MSequence *sequence, const SparkDraftServerRequest *request, uint64_t first)
{
	uint8_t tried[Q38M_SEQUENCES_MAX];
	memset(tried, 0, sizeof(tried));
	for ( ;; )
	{
		Q38MSequence *donor = 0;
		uint32_t chosen = 0u, missing = 0u;
		uint64_t position = first;
		for ( uint64_t at = first; at < request->position; at++ )
			missing += Q38MHeld(sequence, at) == 0u ? 1u : 0u;
		if ( missing == 0u )
			return SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < state->capacity; index++ )
		{
			Q38MSequence *other = &state->sequences[index];
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
			if ( Q38MUsable(sequence, donor, request, position) == 0u )
			{
				position++;
				continue;
			}
			while ( stop < request->position && Q38MUsable(sequence, donor, request, stop) != 0u && (stop == position || stop % Q38M_WINDOW != 0u) )
				stop++;
			if ( Q38MCopySlots(state, sequence, donor, (uint32_t)(position % Q38M_WINDOW), (uint32_t)(stop - position)) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			for ( uint64_t at = position; at < stop; at++ )
			{
				sequence->positions[at % Q38M_WINDOW] = (int64_t)at;
				sequence->tokens[at % Q38M_WINDOW] = donor->tokens[at % Q38M_WINDOW];
			}
			position = stop;
		}
	}
}

static void Q38MSpan(Q38MSequence *sequence, uint64_t checked, uint64_t position, Q38MDraft *draft)
{
	const uint64_t count = position - checked;
	const uint32_t first = count != 0u ? (uint32_t)(checked % Q38M_WINDOW) : 0u, end = first + (uint32_t)count;
	uint32_t newer = 0u;
	memset(draft->valid, 0, sizeof(draft->valid));
	draft->masked = 0u;
	if ( count == Q38M_WINDOW )
	{
		draft->start = 0u;
		draft->block_at = Q38M_WINDOW;
		return;
	}
	for ( uint32_t slot = end; slot < Q38M_WINDOW && slot < end + Q38M_BLOCK; slot++ )
		newer |= count != 0u && sequence->positions[slot] > (int64_t)(position - 1u) ? 1u : 0u;
	if ( end <= Q38M_WINDOW && newer == 0u )
	{
		for ( uint32_t slot = end; slot < Q38M_WINDOW && slot < end + Q38M_BLOCK; slot++ )
			sequence->positions[slot] = -1;
		draft->start = first;
		draft->block_at = end;
		return;
	}
	for ( uint64_t at = checked; at < position; at++ )
		draft->valid[at % Q38M_WINDOW] = 1u;
	draft->masked = 1u;
	draft->start = 0u;
	draft->block_at = Q38M_WINDOW;
}

static SparkStatus Q38MPrepare(Q38MState *state, Q38MSequence *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
	const SparkRowSampling *sampling)
{
	Q38MDraft *draft = state->forward.host_draft;
	if ( anchor >= Q38M_VOCAB || position + Q38M_BLOCK > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft, 0, offsetof(Q38MDraft, valid));
	for ( uint32_t row = 0u; row < Q38M_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	draft->anchor = anchor;
	draft->mask = state->mask_token;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	Q38MSpan(sequence, checked, position, draft);
	if ( SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38MRunForward(Q38MState *state, const Q38MSequence *sequence, uint32_t layers)
{
	Q38MForward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(Q38MDraft, positions));
	const uint32_t block_at = state->forward.host_draft->block_at;
	const uint32_t span = block_at - state->forward.host_draft->start;
	const uint32_t splits = (span + Q38M_SPLIT_KEYS - 1u) / Q38M_SPLIT_KEYS;
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((Q38MEmbedKernel), Q38M_BLOCK, Q38M_THREADS, 0, stream, state->embed, forward->draft, forward->hidden);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const Q38MLayer *weights = &state->layers[layer];
		LM_LAUNCH((Q38MRmsKernel), Q38M_BLOCK, Q38M_THREADS, 0, stream, forward->hidden, weights->input_norm, forward->x, Q38M_HIDDEN, state->epsilon);
		status = Q38MGemm(state, weights->q, forward->x, forward->q, Q38M_BLOCK, Q38M_HIDDEN, Q38M_Q_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38MGemm(state, weights->k, forward->x, forward->k, Q38M_BLOCK, Q38M_HIDDEN, Q38M_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38MGemm(state, weights->v, forward->x, forward->v, Q38M_BLOCK, Q38M_HIDDEN, Q38M_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38MHeadNormRopeKernel), Q38M_BLOCK * Q38M_HEADS, Q38M_HEAD_DIM, 0, stream, forward->q, weights->q_norm, state->rope_inverse,
			state->rope_gain, positions, Q38M_HEADS, state->epsilon, forward->q_roped, (uint16_t *)0, 0u, 0u);
		LM_LAUNCH((Q38MHeadNormRopeKernel), Q38M_BLOCK * Q38M_KV_HEADS, Q38M_HEAD_DIM, 0, stream, forward->k, weights->k_norm, state->rope_inverse,
			state->rope_gain, positions, Q38M_KV_HEADS, state->epsilon, (uint16_t *)0, Q38MKeys(sequence, layer), block_at, 0u);
		LM_LAUNCH((Q38MStoreValuesKernel), Q38M_BLOCK, Q38M_THREADS, 0, stream, forward->v, (const uint32_t *)0, block_at, Q38MValues(sequence, layer));
		if ( splits != 0u )
			LM_LAUNCH((Q38MAttentionSplitKernel), dim3(Q38M_HEADS, splits), Q38M_ATTN_THREADS, 0, stream, forward->q_roped, Q38MKeys(sequence, layer),
				Q38MValues(sequence, layer), forward->draft, state->softmax_scale, forward->partials);
		LM_LAUNCH((Q38MAttentionMergeKernel), Q38M_BLOCK * Q38M_HEADS, Q38M_HEAD_DIM, 0, stream, forward->q_roped, Q38MKeys(sequence, layer),
			Q38MValues(sequence, layer), forward->draft, state->softmax_scale, forward->partials, splits, forward->attended);
		status = Q38MGemm(state, weights->o, forward->attended, forward->projected, Q38M_BLOCK, Q38M_Q_WIDTH, Q38M_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38MResidualKernel), 2u * state->multiprocessors, Q38M_THREADS, 0, stream, forward->hidden, forward->projected, forward->residual,
			Q38M_BLOCK * Q38M_HIDDEN);
		LM_LAUNCH((Q38MRmsKernel), Q38M_BLOCK, Q38M_THREADS, 0, stream, forward->residual, weights->post_norm, forward->x, Q38M_HIDDEN, state->epsilon);
		status = Q38MGemm(state, weights->gate, forward->x, forward->gate, Q38M_BLOCK, Q38M_HIDDEN, Q38M_MLP);
		if ( status == SPARK_STATUS_OK )
			status = Q38MGemm(state, weights->up, forward->x, forward->up, Q38M_BLOCK, Q38M_HIDDEN, Q38M_MLP);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38MSwigluKernel), 2u * state->multiprocessors, Q38M_THREADS, 0, stream, forward->gate, forward->up, forward->act, Q38M_BLOCK * Q38M_MLP);
		status = Q38MGemm(state, weights->down, forward->act, forward->mlp, Q38M_BLOCK, Q38M_MLP, Q38M_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38MResidualKernel), 2u * state->multiprocessors, Q38M_THREADS, 0, stream, forward->residual, forward->mlp, forward->hidden,
			Q38M_BLOCK * Q38M_HIDDEN);
	}
	if ( status != SPARK_STATUS_OK || layers < Q38M_LAYERS )
		return status;
	LM_LAUNCH((Q38MRmsKernel), Q38M_BLOCK, Q38M_THREADS, 0, stream, forward->hidden, state->norm, forward->y, Q38M_HIDDEN, state->epsilon);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38MHeads(Q38MState *state, uint32_t anchor_token, uint64_t position, const SparkRowSampling *sampling, uint32_t drafts, uint32_t tree)
{
	Q38MForward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const float nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	SparkStatus status = Q38MGemm(state, state->head, forward->y, forward->logits, Q38M_BLOCK, Q38M_HIDDEN, Q38M_VOCAB);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Q38MWidenKernel), 4u * state->multiprocessors, Q38M_THREADS, 0, stream, forward->logits, forward->logits_f32, Q38M_BLOCK * Q38M_VOCAB);
	if ( LmTopkExactRowsLaunch<Q38M_TOPK_THREADS>(forward->logits_f32, Q38M_BLOCK, Q38M_VOCAB, Q38M_CANDIDATES, forward->candidates, stream) !=
		SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t step = 0u; step < Q38M_BLOCK; step++ )
	{
		LM_LAUNCH((Q38MMarkovRowKernel), Q38M_CANDIDATES / Q38M_WARPS, Q38M_THREADS, 0, stream, step, anchor_token, forward->picks->tokens,
			forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
		LM_LAUNCH((Q38MMarkovPickKernel), 1u, Q38M_PICK_THREADS, 0, stream, step, anchor_token, (uint32_t)position, (unsigned long long)sampling->seed,
			sampling->inverse_temperature, nucleus, sampling->top_k, forward->markov_row, forward->candidates, forward->y, state->markov_previous,
			state->confidence_weight, state->confidence_bias, forward->picks);
	}
	if ( tree != 0u && drafts >= 2u )
	{
		const float scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
		LM_LAUNCH((Q38MBranchSelectKernel), 1u, Q38M_WARP, 0, stream, forward->picks, drafts, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE);
		for ( uint32_t offset = 1u; offset < Q38M_BRANCH_LEN && offset < drafts; offset++ )
		{
			LM_LAUNCH((Q38MBranchRowKernel), dim3(Q38M_CANDIDATES / Q38M_WARPS, Q38M_BRANCHES), Q38M_THREADS, 0, stream, offset, drafts,
				forward->picks, forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
			LM_LAUNCH((Q38MBranchPickKernel), Q38M_BRANCHES, Q38M_PICK_THREADS, 0, stream, offset, drafts, scale, forward->markov_row, forward->candidates,
				forward->picks);
		}
	}
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(Q38MPicks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38MSettle(Q38MState *state, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static void Q38MForwardFree(Q38MForward *forward)
{
	void *buffers[] = {
		forward->draft, forward->picks, forward->hidden, forward->x, forward->q, forward->q_roped, forward->k, forward->v, forward->partials,
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

static SparkStatus Q38MClose(void *opaque)
{
	Q38MState *state = (Q38MState *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	Q38MForwardFree(&state->forward);
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

static SparkStatus Q38MOpen(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	Q38MState *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return Q38MRefuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	{
		const SparkStatus device = SparkDeviceInitialize(0);
		if ( (device != SPARK_STATUS_OK && device != SPARK_STATUS_DUPLICATE) || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
			return Q38MRefuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
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
	if ( lane->context_bytes < Q38M_SEQUENCE_BYTES )
		return Q38MRefuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one sequence's window", lane->lane_name);
	state = (Q38MState *)calloc(1u, sizeof(*state));
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
		status = Q38MWeights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = Q38MScratch(state, lane->context_bytes);
	if ( status == SPARK_STATUS_OK )
		status = Q38MSettle(state, SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a Qwen3.8-Max DSpark pack this module reads", lane->lane_name, lane->pack_path);
		(void)Q38MClose(state);
		SPARK_FAIL(status);
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s with %u sequence windows", lane->lane_name, lane->pack_path, state->capacity);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus Q38MPropose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	Q38MState *state = (Q38MState *)opaque;
	Q38MSequence *sequence = Q38MLive(state, request->sequence_id);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > Q38M_WINDOW ? request->position - Q38M_WINDOW : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( sequence == 0 || (request->position != 0u && Q38MHeld(sequence, request->position - 1u) == 0u) )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	status = Q38MFillFromDonors(state, sequence, request, first);
	if ( status != SPARK_STATUS_OK )
		return Q38MSettle(state, status);
	for ( uint64_t position = first; position < request->position; position++ )
		if ( Q38MHeld(sequence, position) == 0u )
			checked = position + 1u;
	if ( checked != first )
		state->gaps++;
	if ( request->committed_token_count < request->position )
		return Q38MSettle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	for ( uint64_t position = checked; position < request->position; position++ )
		if ( sequence->tokens[position % Q38M_WINDOW] != request->committed_token_ids[position] )
			return Q38MSettle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	drafts = drafts < Q38M_BLOCK ? drafts : Q38M_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		SparkDraftTreeBranch branches[Q38M_BLOCK];
		uint32_t branch_count = 0u;
		status = Q38MPrepare(state, sequence, request->anchor_token_id, request->position, checked, &request->sampling);
		if ( status == SPARK_STATUS_OK )
			status = Q38MRunForward(state, sequence, Q38M_LAYERS);
		if ( status == SPARK_STATUS_OK )
			status = Q38MHeads(state, request->anchor_token_id, request->position, &request->sampling, drafts, tree);
		status = Q38MSettle(state, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				uint32_t matched = Q38M_BRANCHES;
				for ( uint32_t b = 0u; b < Q38M_BRANCHES; b++ )
					if ( state->forward.host_picks->branch_len[b] > 0u && state->forward.host_picks->branch_fork[b] == step )
						matched = b;
				branches[branch_count].fork_step = step;
				if ( matched < Q38M_BRANCHES )
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

static const SparkDrafterModuleInterface q38m_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"qwen38-max-dspark",
	Q38MOpen,
	Q38MIngest,
	Q38MPropose,
	Q38MClose,
};

static SparkStatus Q38MReadback(void *host, const void *device, size_t bytes)
{
	if ( SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &q38m_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_max_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	Q38MState *state = (Q38MState *)opaque;
	const Q38MSequence *sequence = state != 0 ? Q38MLive(state, sequence_id) : 0;
	SparkStatus status;
	if ( sequence == 0 || layer >= Q38M_LAYERS || keys == 0 || values == 0 || positions == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memcpy(positions, sequence->positions, sizeof(sequence->positions));
	status = Q38MReadback(keys, Q38MKeys(sequence, layer), Q38M_LAYER_KV * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = Q38MReadback(values, Q38MValues(sequence, layer), Q38M_LAYER_KV * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_max_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	Q38MState *state = (Q38MState *)opaque;
	Q38MSequence *sequence = state != 0 ? Q38MLive(state, sequence_id) : 0;
	const uint64_t started = SparkMonotonicNs(), first = position > Q38M_WINDOW ? position - Q38M_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	for ( uint64_t at = first; at < position; at++ )
		if ( Q38MHeld(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = Q38MPrepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Q38MRunForward(state, sequence, Q38M_LAYERS);
	status = Q38MSettle(state, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = Q38MReadback(y, state->forward.y, (size_t)Q38M_BLOCK * Q38M_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_max_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	Q38MState *state = (Q38MState *)opaque;
	Q38MSequence *sequence = state != 0 ? Q38MLive(state, sequence_id) : 0;
	const uint64_t first = position > Q38M_WINDOW ? position - Q38M_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 || hidden == 0 || layers == 0u || layers > Q38M_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint64_t at = first; at < position; at++ )
		if ( Q38MHeld(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = Q38MPrepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Q38MRunForward(state, sequence, layers);
	status = Q38MSettle(state, status);
	if ( status == SPARK_STATUS_OK )
		status = Q38MReadback(hidden, state->forward.hidden, (size_t)Q38M_BLOCK * Q38M_HIDDEN * sizeof(uint16_t));
	return status;
}
