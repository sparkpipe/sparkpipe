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
#include "model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G53F_HIDDEN 4096u
#define G53F_TAPS 5u
#define G53F_TAP_WIDTH (G53F_TAPS * G53F_HIDDEN)
#define G53F_LAYERS 5u
#define G53F_HEADS 32u
#define G53F_KV_HEADS 8u
#define G53F_GROUP (G53F_HEADS / G53F_KV_HEADS)
#define G53F_HEAD_DIM 128u
#define G53F_HALF (G53F_HEAD_DIM / 2u)
#define G53F_Q_WIDTH (G53F_HEADS * G53F_HEAD_DIM)
#define G53F_KV_WIDTH (G53F_KV_HEADS * G53F_HEAD_DIM)
#define G53F_MLP 12288u
#define G53F_BLOCK 8u
#define G53F_DRAFTS (G53F_BLOCK - 1u)
#define G53F_SLIDING 2048u
#define G53F_WINDOW 2048u
#define G53F_COLUMNS (G53F_WINDOW + G53F_BLOCK)
#define G53F_VOCAB SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT
#define G53F_CONV_GROUP 16u
#define G53F_CONV_GROUPS (G53F_HIDDEN / G53F_CONV_GROUP)
#define G53F_CONV_TAPS 2u
#define G53F_CONV_WIDTH (2u * G53F_CONV_TAPS * G53F_CONV_GROUPS)
#define G53F_SELECTOR_RANK 256u
#define G53F_TOP_K 16u
#define G53F_BRANCH_LEN 4u
#define G53F_INGEST_ROWS 256u
#define G53F_THREADS 256u
#define G53F_WARP 32u
#define G53F_WARPS (G53F_THREADS / G53F_WARP)
#define G53F_TOPK_THREADS 512u
#define G53F_SPLIT_KEYS 256u
#define G53F_SPLITS ((G53F_WINDOW + G53F_SPLIT_KEYS - 1u) / G53F_SPLIT_KEYS)
#define G53F_ATTN_TILE 32u
#define G53F_ATTN_THREADS (G53F_BLOCK * G53F_WARP)
#define G53F_SEQUENCES_MAX 256u
#define G53F_LAYER_KV ((uint64_t)G53F_KV_HEADS * G53F_COLUMNS * G53F_HEAD_DIM)
#define G53F_SEQUENCE_ELEMENTS (2u * G53F_LAYERS * G53F_LAYER_KV)
#define G53F_SEQUENCE_BYTES (G53F_SEQUENCE_ELEMENTS * sizeof(uint16_t))
#define G53F_PACK_MAGIC 0x4B505244u
#define G53F_PACK_VERSION 1u
#define G53F_PACK_ALIGNMENT 128u
#define G53F_INVERSE_PI 0.31830988618379067154

typedef struct G53FLayer
{
	const uint16_t *input_norm;
	const uint16_t *post_norm;
	const uint16_t *attn_conv_base;
	const uint16_t *attn_conv_proj;
	const uint16_t *q;
	const uint16_t *k;
	const uint16_t *v;
	const uint16_t *o;
	const uint16_t *q_norm;
	const uint16_t *k_norm;
	const uint16_t *mlp_conv_base;
	const uint16_t *mlp_conv_proj;
	const uint16_t *gate;
	const uint16_t *up;
	const uint16_t *down;
} G53FLayer;

typedef struct G53FSequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t last_used;
	uint32_t live;
	uint32_t retired;
	int64_t positions[G53F_WINDOW];
	uint32_t tokens[G53F_WINDOW];
	uint16_t *cache;
} G53FSequence;

typedef struct G53FDraft
{
	uint64_t seed;
	uint32_t positions[G53F_BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t greedy;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint32_t start;
	uint32_t block_at;
	uint32_t masked;
	uint32_t reserved;
	uint8_t valid[G53F_WINDOW];
} G53FDraft;

typedef struct G53FPicks
{
	uint32_t tokens[G53F_DRAFTS];
	float scores[G53F_DRAFTS];
	uint32_t branch_tokens[G53F_DRAFTS][G53F_BRANCH_LEN];
	float branch_scores[G53F_DRAFTS][G53F_BRANCH_LEN];
} G53FPicks;

typedef struct G53FForward
{
	G53FDraft *host_draft;
	G53FDraft *draft;
	G53FPicks *host_picks;
	G53FPicks *picks;
	float *hidden;
	uint16_t *x;
	uint16_t *conv_coeff;
	uint16_t *prepared;
	uint16_t *q;
	uint16_t *q_roped;
	uint16_t *k;
	uint16_t *v;
	float *partials;
	uint16_t *attended;
	uint16_t *projected;
	uint16_t *gate;
	uint16_t *up;
	uint16_t *act;
	uint16_t *mlp;
	uint16_t *y;
	uint16_t *logits;
	float *logits_f32;
	uint16_t *sel_hidden;
	uint32_t *candidates;
	float *unary;
	float *tail;
	float *edges;
} G53FForward;

typedef struct G53FState
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	uint32_t mask_token;
	float epsilon;
	float softmax_scale;
	float *rope_inverse;
	const uint16_t *fc;
	const uint16_t *hidden_norm;
	const uint16_t *norm;
	const uint16_t *embed;
	const uint16_t *head;
	const uint16_t *selector_hidden;
	const uint16_t *predecessors;
	const uint16_t *successors;
	G53FLayer layers[G53F_LAYERS];
	G53FForward forward;
	uint16_t *host_taps;
	uint32_t *host_positions;
	uint16_t *taps;
	uint32_t *positions;
	float *features_raw;
	uint16_t *features;
	uint16_t *keys_raw;
	uint16_t *values_raw;
	uint16_t *caches;
	uint32_t capacity;
	uint64_t clock;
	G53FSequence *sequences;
	uint64_t frames;
	uint64_t rows;
	uint64_t gaps;
} G53FState;

static __device__ __forceinline__ void G53FRotation(uint32_t position, float inverse, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * G53F_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	sincospif(reduced, sine, cosine);
}

static __device__ __forceinline__ float G53FWarpSum(float value)
{
	for ( uint32_t mask = G53F_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ float G53FWarpMax(float value)
{
	for ( uint32_t mask = G53F_WARP / 2u; mask > 0u; mask >>= 1u )
		value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
	return value;
}

static __device__ __forceinline__ float G53FLogAddExp(float a, float b)
{
	const float hi = fmaxf(a, b), lo = fminf(a, b);
	if ( hi == -INFINITY )
		return -INFINITY;
	return hi + log1pf(expf(lo - hi));
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FRopeTableKernel(float *__restrict__ inverse, float theta)
{
	if ( threadIdx.x < G53F_HALF )
		inverse[threadIdx.x] = 1.0f / powf(theta, (float)(2u * threadIdx.x) / (float)G53F_HEAD_DIM);
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FRmsF32Kernel(const float *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[G53F_WARPS];
	const uint64_t row = blockIdx.x;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += G53F_THREADS )
	{
		const float value = input[row * dimension + index];
		total += value * value;
	}
	total = LmBlockSum<G53F_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += G53F_THREADS )
		output[row * dimension + index] = LmFloatToBf16(input[row * dimension + index] * scale * LmBf16ToFloat(weight[index]));
}

__global__ __launch_bounds__(G53F_HEAD_DIM, 1)
void G53FHeadNormRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, uint16_t *__restrict__ output, uint16_t *__restrict__ cache_keys,
	uint32_t block_at, uint32_t use_ring)
{
	__shared__ float reduction[G53F_HEAD_DIM / G53F_WARP];
	__shared__ float normed[G53F_HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = LmBf16ToFloat(input[((uint64_t)row * heads + head) * G53F_HEAD_DIM + lane]);
	float scale, cosine, sine, rotated;
	scale = rsqrtf(LmBlockSum<G53F_HEAD_DIM>(value * value, reduction) / (float)G53F_HEAD_DIM + epsilon);
	normed[lane] = value * scale * LmBf16ToFloat(weight[lane]);
	__syncthreads();
	G53FRotation(position, inverse[lane % G53F_HALF], &cosine, &sine);
	rotated = lane < G53F_HALF ? normed[lane] * cosine - normed[lane + G53F_HALF] * sine : normed[lane] * cosine + normed[lane - G53F_HALF] * sine;
	if ( cache_keys != 0 )
	{
		const uint32_t column = use_ring != 0u ? position % G53F_WINDOW : block_at + row;
		cache_keys[((uint64_t)head * G53F_COLUMNS + column) * G53F_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	}
	else
		output[((uint64_t)row * heads + head) * G53F_HEAD_DIM + lane] = LmFloatToBf16(rotated);
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FStoreValuesKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions, uint32_t block_at, uint16_t *__restrict__ cache)
{
	const uint32_t row = blockIdx.x, column = positions != 0 ? positions[row] % G53F_WINDOW : block_at + row;
	for ( uint32_t index = threadIdx.x; index < G53F_KV_WIDTH; index += G53F_THREADS )
		cache[((uint64_t)(index / G53F_HEAD_DIM) * G53F_COLUMNS + column) * G53F_HEAD_DIM + index % G53F_HEAD_DIM] = values[(uint64_t)row * G53F_KV_WIDTH + index];
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FEmbedKernel(const uint16_t *__restrict__ embed, const G53FDraft *__restrict__ draft, float *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	for ( uint32_t index = threadIdx.x; index < G53F_HIDDEN; index += G53F_THREADS )
		hidden[(uint64_t)row * G53F_HIDDEN + index] = LmBf16ToFloat(embed[(uint64_t)token * G53F_HIDDEN + index]);
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FConvPrepareKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ base, const uint16_t *__restrict__ delta,
	uint16_t *__restrict__ output)
{
	const uint32_t row = blockIdx.x;
	const uint16_t *base0 = base, *base1 = base + G53F_HIDDEN;
	const uint16_t *delta0 = delta + (uint64_t)row * G53F_CONV_WIDTH;
	const uint16_t *delta1 = delta0 + G53F_CONV_GROUPS;
	for ( uint32_t col = threadIdx.x; col < G53F_HIDDEN; col += G53F_THREADS )
	{
		const uint32_t group = col / G53F_CONV_GROUP;
		const float c0 = LmBf16ToFloat(base0[col]) + LmBf16ToFloat(delta0[group]);
		float acc = c0 * LmBf16ToFloat(input[(uint64_t)row * G53F_HIDDEN + col]);
		if ( row > 0u )
		{
			const float c1 = LmBf16ToFloat(base1[col]) + LmBf16ToFloat(delta1[group]);
			acc = fmaf(c1, LmBf16ToFloat(input[(uint64_t)(row - 1u) * G53F_HIDDEN + col]), acc);
		}
		output[(uint64_t)row * G53F_HIDDEN + col] = LmFloatToBf16(acc);
	}
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FConvFinishResidualKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ base, const uint16_t *__restrict__ delta,
	float *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x;
	const uint16_t *base0 = base + 2u * G53F_HIDDEN, *base1 = base + 3u * G53F_HIDDEN;
	const uint16_t *delta0 = delta + (uint64_t)row * G53F_CONV_WIDTH + 2u * G53F_CONV_GROUPS;
	const uint16_t *delta1 = delta0 + G53F_CONV_GROUPS;
	for ( uint32_t col = threadIdx.x; col < G53F_HIDDEN; col += G53F_THREADS )
	{
		const uint32_t group = col / G53F_CONV_GROUP;
		const float c0 = LmBf16ToFloat(base0[col]) + LmBf16ToFloat(delta0[group]);
		float acc = c0 * LmBf16ToFloat(input[(uint64_t)row * G53F_HIDDEN + col]);
		if ( row > 0u )
		{
			const float c1 = LmBf16ToFloat(base1[col]) + LmBf16ToFloat(delta1[group]);
			acc = fmaf(c1, LmBf16ToFloat(input[(uint64_t)(row - 1u) * G53F_HIDDEN + col]), acc);
		}
		hidden[(uint64_t)row * G53F_HIDDEN + col] += acc;
	}
}

__global__ __launch_bounds__(G53F_ATTN_THREADS, 1)
void G53FAttentionSplitKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ keys, const uint16_t *__restrict__ values,
	const G53FDraft *__restrict__ draft, float scale, float *__restrict__ partials)
{
	__shared__ uint32_t s_k[G53F_ATTN_TILE][G53F_HALF + 1u];
	__shared__ uint32_t s_v[G53F_ATTN_TILE][G53F_HALF + 1u];
	__shared__ float s_scores[G53F_BLOCK][G53F_ATTN_TILE];
	__shared__ uint8_t s_vis[G53F_ATTN_TILE];
	const uint32_t head = blockIdx.x, kv_head = head / G53F_GROUP, split = blockIdx.y;
	const uint32_t warp = threadIdx.x / G53F_WARP, lane = threadIdx.x % G53F_WARP;
	const uint32_t warp_bit = 1u << warp;
	const uint32_t first = draft->start + split * G53F_SPLIT_KEYS, end = draft->block_at;
	const uint32_t stop = first + G53F_SPLIT_KEYS < end ? first + G53F_SPLIT_KEYS : end;
	const uint32_t tiles = first < stop ? (stop - first + G53F_ATTN_TILE - 1u) / G53F_ATTN_TILE : 0u;
	const uint32_t masked = draft->masked;
	const uint16_t *head_keys = keys + (uint64_t)kv_head * G53F_COLUMNS * G53F_HEAD_DIM;
	const uint16_t *head_values = values + (uint64_t)kv_head * G53F_COLUMNS * G53F_HEAD_DIM;
	const uint16_t *q_row = q + ((uint64_t)warp * G53F_HEADS + head) * G53F_HEAD_DIM;
	const uint32_t qp0 = *(const uint32_t *)(q_row + 2u * lane);
	const uint32_t qp1 = *(const uint32_t *)(q_row + G53F_HALF + 2u * lane);
	const float q0 = LmBf16ToFloat((uint16_t)(qp0 & 0xffffu));
	const float q1 = LmBf16ToFloat((uint16_t)(qp0 >> 16u));
	const float q2 = LmBf16ToFloat((uint16_t)(qp1 & 0xffffu));
	const float q3 = LmBf16ToFloat((uint16_t)(qp1 >> 16u));
	float running_max = -INFINITY, running_sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
	float *out = partials + (((uint64_t)head * G53F_SPLITS + split) * G53F_BLOCK + warp) * (G53F_HEAD_DIM + 2u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t tile_first = first + tile * G53F_ATTN_TILE;
		for ( uint32_t item = threadIdx.x; item < G53F_ATTN_TILE * 16u; item += G53F_ATTN_THREADS )
		{
			const uint32_t load_key = item >> 4u, load_chunk = item & 15u;
			const uint32_t col = tile_first + load_key;
			if ( col < stop )
			{
				const uint4 kv_k = *(const uint4 *)(head_keys + (uint64_t)col * G53F_HEAD_DIM + load_chunk * 8u);
				const uint4 kv_v = *(const uint4 *)(head_values + (uint64_t)col * G53F_HEAD_DIM + load_chunk * 8u);
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
		if ( threadIdx.x < G53F_ATTN_TILE )
		{
			const uint32_t col = tile_first + threadIdx.x;
			s_vis[threadIdx.x] = col < stop ? (masked == 0u ? (uint8_t)0xffu : draft->valid[col]) : (uint8_t)0u;
		}
		__syncthreads();
		for ( uint32_t t = 0u; t < G53F_ATTN_TILE; t++ )
		{
			const uint32_t kp0 = s_k[t][lane];
			const uint32_t kp1 = s_k[t][G53F_WARP + lane];
			const float d0 = fmaf(q0, LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1 * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
			const float d1 = fmaf(q2, LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), q3 * LmBf16ToFloat((uint16_t)(kp1 >> 16u)));
			const float dot = G53FWarpSum(d0 + d1);
			if ( lane == 0u )
				s_scores[warp][t] = (s_vis[t] & warp_bit) != 0u ? dot * scale : -INFINITY;
		}
		__syncwarp();
		{
			const float sc = s_scores[warp][lane];
			const float tile_max = G53FWarpMax(sc);
			const float next_max = fmaxf(running_max, tile_max);
			const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
			const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
			const float tile_sum = G53FWarpSum(weight);
			running_sum = running_sum * rescale + tile_sum;
			running_max = next_max;
			acc0 *= rescale;
			acc1 *= rescale;
			acc2 *= rescale;
			acc3 *= rescale;
			s_scores[warp][lane] = weight;
		}
		__syncwarp();
		for ( uint32_t t = 0u; t < G53F_ATTN_TILE; t++ )
		{
			const float wt = s_scores[warp][t];
			const uint32_t vp0 = s_v[t][lane];
			const uint32_t vp1 = s_v[t][G53F_WARP + lane];
			acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0);
			acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1);
			acc2 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2);
			acc3 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3);
		}
		__syncthreads();
	}
	out[2u * lane] = acc0;
	out[2u * lane + 1u] = acc1;
	out[G53F_HALF + 2u * lane] = acc2;
	out[G53F_HALF + 2u * lane + 1u] = acc3;
	if ( lane == 0u )
	{
		out[G53F_HEAD_DIM] = running_max;
		out[G53F_HEAD_DIM + 1u] = running_sum;
	}
}

__global__ __launch_bounds__(G53F_HEAD_DIM, 1)
void G53FAttentionMergeKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ keys, const uint16_t *__restrict__ values,
	const G53FDraft *__restrict__ draft, float scale, const float *__restrict__ partials, uint32_t splits, uint16_t *__restrict__ output)
{
	__shared__ float reduction[G53F_HEAD_DIM / G53F_WARP];
	const uint32_t row = blockIdx.x / G53F_HEADS, head = blockIdx.x % G53F_HEADS, kv_head = head / G53F_GROUP, lane = threadIdx.x;
	const uint32_t block_at = draft->block_at;
	const float own = LmBf16ToFloat(q[((uint64_t)row * G53F_HEADS + head) * G53F_HEAD_DIM + lane]);
	const uint16_t *head_keys = keys + (uint64_t)kv_head * G53F_COLUMNS * G53F_HEAD_DIM;
	const uint16_t *head_values = values + (uint64_t)kv_head * G53F_COLUMNS * G53F_HEAD_DIM;
	float top = -INFINITY, sum = 0.0f, accumulated = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part = partials + (((uint64_t)head * G53F_SPLITS + split) * G53F_BLOCK + row) * (G53F_HEAD_DIM + 2u);
		const float part_top = part[G53F_HEAD_DIM], next = fmaxf(top, part_top);
		float carry, weight;
		if ( part_top == -INFINITY )
			continue;
		carry = top == -INFINITY ? 0.0f : expf(top - next);
		weight = expf(part_top - next);
		sum = sum * carry + part[G53F_HEAD_DIM + 1u] * weight;
		accumulated = accumulated * carry + part[lane] * weight;
		top = next;
	}
	for ( uint32_t key = 0u; key < G53F_BLOCK; key++ )
	{
		const float score = LmBlockSum<G53F_HEAD_DIM>(own * LmBf16ToFloat(head_keys[(uint64_t)(block_at + key) * G53F_HEAD_DIM + lane]), reduction) * scale;
		const float next = fmaxf(top, score), carry = top == -INFINITY ? 0.0f : expf(top - next), weight = expf(score - next);
		sum = sum * carry + weight;
		accumulated = accumulated * carry + weight * LmBf16ToFloat(head_values[(uint64_t)(block_at + key) * G53F_HEAD_DIM + lane]);
		top = next;
	}
	output[(uint64_t)row * G53F_Q_WIDTH + head * G53F_HEAD_DIM + lane] = LmFloatToBf16(accumulated / sum);
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FSwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * G53F_THREADS + threadIdx.x; index < count; index += gridDim.x * G53F_THREADS )
	{
		const float g = LmBf16ToFloat(gate[index]);
		output[index] = LmFloatToBf16(g / (1.0f + expf(-g)) * LmBf16ToFloat(up[index]));
	}
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FWidenKernel(const uint16_t *__restrict__ input, float *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * G53F_THREADS + threadIdx.x; index < count; index += gridDim.x * G53F_THREADS )
		output[index] = LmBf16ToFloat(input[index]);
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FLogitsSummaryKernel(const float *__restrict__ logits, uint32_t *__restrict__ candidates, float *__restrict__ unary, float *__restrict__ tail)
{
	__shared__ float s_max[G53F_WARPS];
	__shared__ float s_sum[G53F_WARPS];
	__shared__ uint32_t s_tok[G53F_TOP_K];
	__shared__ float s_val[G53F_TOP_K];
	const uint32_t step = blockIdx.x, warp = threadIdx.x / G53F_WARP, lane = threadIdx.x % G53F_WARP;
	const float *row = logits + (uint64_t)step * G53F_VOCAB;
	float local_max = -INFINITY, local_sum = 0.0f, row_max, row_sum, total;
	for ( uint32_t index = threadIdx.x; index < G53F_VOCAB; index += G53F_THREADS )
		local_max = fmaxf(local_max, row[index]);
	local_max = G53FWarpMax(local_max);
	if ( lane == 0u )
		s_max[warp] = local_max;
	__syncthreads();
	if ( warp == 0u )
	{
		float m = lane < G53F_WARPS ? s_max[lane] : -INFINITY;
		m = G53FWarpMax(m);
		if ( lane == 0u )
			s_max[0] = m;
	}
	__syncthreads();
	row_max = s_max[0];
	for ( uint32_t index = threadIdx.x; index < G53F_VOCAB; index += G53F_THREADS )
		local_sum += expf(row[index] - row_max);
	row_sum = LmBlockSum<G53F_THREADS>(local_sum, s_sum);
	total = row_max + logf(row_sum);
	if ( threadIdx.x < G53F_TOP_K )
	{
		const uint32_t tok = candidates[step * G53F_TOP_K + threadIdx.x];
		s_tok[threadIdx.x] = tok;
		s_val[threadIdx.x] = row[tok];
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float top_max = -INFINITY, top_sum = 0.0f, lse_unary, ratio;
		for ( uint32_t i = 0u; i < G53F_TOP_K; i++ )
		{
			uint32_t best = i;
			for ( uint32_t j = i + 1u; j < G53F_TOP_K; j++ )
				if ( s_val[j] > s_val[best] || (s_val[j] == s_val[best] && s_tok[j] < s_tok[best]) )
					best = j;
			if ( best != i )
			{
				const float tv = s_val[i];
				const uint32_t tt = s_tok[i];
				s_val[i] = s_val[best];
				s_tok[i] = s_tok[best];
				s_val[best] = tv;
				s_tok[best] = tt;
			}
		}
		for ( uint32_t i = 0u; i < G53F_TOP_K; i++ )
		{
			candidates[step * G53F_TOP_K + i] = s_tok[i];
			unary[step * G53F_TOP_K + i] = s_val[i];
			top_max = fmaxf(top_max, s_val[i]);
		}
		for ( uint32_t i = 0u; i < G53F_TOP_K; i++ )
			top_sum += expf(s_val[i] - top_max);
		lse_unary = top_max + logf(top_sum);
		ratio = fminf(expf(lse_unary - total), 1.0f - 1e-7f);
		tail[step] = total + log1pf(-ratio);
	}
}

__global__ __launch_bounds__(G53F_THREADS, 1)
void G53FLatticeKernel(const G53FDraft *__restrict__ draft, const uint32_t *__restrict__ candidates, const float *__restrict__ unary,
	const uint16_t *__restrict__ sel_hidden, const uint16_t *__restrict__ predecessors, const uint16_t *__restrict__ successors, float *__restrict__ edges)
{
	__shared__ float gate[G53F_SELECTOR_RANK];
	const uint32_t step = blockIdx.x / G53F_TOP_K, prev_slot = blockIdx.x % G53F_TOP_K;
	const uint32_t warp = threadIdx.x / G53F_WARP, lane = threadIdx.x % G53F_WARP;
	uint32_t prev_token;
	if ( step == 0u && prev_slot != 0u )
		return;
	prev_token = step == 0u ? draft->anchor : candidates[(step - 1u) * G53F_TOP_K + prev_slot];
	for ( uint32_t r = threadIdx.x; r < G53F_SELECTOR_RANK; r += G53F_THREADS )
		gate[r] = LmBf16ToFloat(predecessors[(uint64_t)prev_token * G53F_SELECTOR_RANK + r]) *
			LmBf16ToFloat(sel_hidden[(uint64_t)step * G53F_SELECTOR_RANK + r]);
	__syncthreads();
	for ( uint32_t c = warp; c < G53F_TOP_K; c += G53F_WARPS )
	{
		const uint32_t succ_token = candidates[step * G53F_TOP_K + c];
		const uint16_t *succ_row = successors + (uint64_t)succ_token * G53F_SELECTOR_RANK;
		float dot = 0.0f;
		for ( uint32_t r = lane; r < G53F_SELECTOR_RANK; r += G53F_WARP )
			dot = fmaf(gate[r], LmBf16ToFloat(succ_row[r]), dot);
		dot = G53FWarpSum(dot);
		if ( lane == 0u )
			edges[(step * G53F_TOP_K + prev_slot) * G53F_TOP_K + c] = unary[step * G53F_TOP_K + c] + dot;
	}
}

__global__ __launch_bounds__(G53F_WARP, 1)
void G53FWalkKernel(const G53FDraft *__restrict__ draft, const uint32_t *__restrict__ candidates, const float *__restrict__ tail,
	const float *__restrict__ edges, uint32_t tree, G53FPicks *__restrict__ picks)
{
	const uint32_t lane = threadIdx.x;
	const uint32_t active = (1u << G53F_TOP_K) - 1u;
	uint32_t previous = 0u;
	if ( lane >= G53F_TOP_K )
		return;
	for ( uint32_t step = 0u; step < G53F_DRAFTS; step++ )
	{
		const float row_val = edges[(step * G53F_TOP_K + previous) * G53F_TOP_K + lane];
		const uint32_t cand_tok = candidates[step * G53F_TOP_K + lane];
		float row_max = row_val, row_sum, lse_row, normalizer;
		uint32_t choice = 0u, runner = 0u;
		float chosen_val, runner_val;
		for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
			row_max = fmaxf(row_max, __shfl_xor_sync(active, row_max, mask));
		row_sum = expf(row_val - row_max);
		for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
			row_sum += __shfl_xor_sync(active, row_sum, mask);
		lse_row = row_max + logf(row_sum);
		normalizer = G53FLogAddExp(lse_row, tail[step]);
		if ( draft->greedy != 0u )
		{
			unsigned long long key = ((unsigned long long)SparkMathSampleOrdered(row_val) << 32u) | (unsigned long long)(0xffffffffu - lane);
			for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
			{
				const unsigned long long other = __shfl_xor_sync(active, key, mask);
				key = other > key ? other : key;
			}
			choice = 0xffffffffu - (uint32_t)key;
		}
		else
		{
			const float scaled = row_val * draft->inverse_temperature;
			float s_max = scaled, prob, s_sum, before = 0.0f;
			uint32_t rank = 0u, winner_rank;
			unsigned long long key;
			for ( uint32_t other = 0u; other < G53F_TOP_K; other++ )
			{
				const float ov = __shfl_sync(active, scaled, other);
				rank += ov > scaled || (ov == scaled && other < lane) ? 1u : 0u;
			}
			for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
				s_max = fmaxf(s_max, __shfl_xor_sync(active, s_max, mask));
			prob = expf(scaled - s_max);
			s_sum = prob;
			for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
				s_sum += __shfl_xor_sync(active, s_sum, mask);
			prob /= s_sum;
			for ( uint32_t other = 0u; other < G53F_TOP_K; other++ )
			{
				const float op = __shfl_sync(active, prob, other);
				const float ov = __shfl_sync(active, scaled, other);
				if ( ov > scaled || (ov == scaled && other < lane) )
					before += op;
			}
			key = (((draft->nucleus <= 0.0f) || (before < draft->nucleus)) && (draft->top_k == 0u || rank < draft->top_k)) || rank == 0u
				? ((unsigned long long)SparkMathSampleOrdered(scaled + SparkMathGumbelNoise(draft->seed, draft->positions[0] + step, cand_tok)) << 32u) |
					(unsigned long long)(0xffffffffu - rank)
				: 0ull;
			for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
			{
				const unsigned long long other = __shfl_xor_sync(active, key, mask);
				key = other > key ? other : key;
			}
			winner_rank = 0xffffffffu - (uint32_t)key;
			for ( uint32_t other = 0u; other < G53F_TOP_K; other++ )
				if ( __shfl_sync(active, rank, other) == winner_rank )
					choice = other;
		}
		{
			const float masked_val = lane == choice ? -INFINITY : row_val;
			unsigned long long rkey = ((unsigned long long)SparkMathSampleOrdered(masked_val) << 32u) | (unsigned long long)(0xffffffffu - lane);
			for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
			{
				const unsigned long long other = __shfl_xor_sync(active, rkey, mask);
				rkey = other > rkey ? other : rkey;
			}
			runner = 0xffffffffu - (uint32_t)rkey;
		}
		chosen_val = __shfl_sync(active, row_val, choice);
		runner_val = __shfl_sync(active, row_val, runner);
		if ( lane == 0u )
		{
			picks->tokens[step] = candidates[step * G53F_TOP_K + choice];
			picks->scores[step] = expf(chosen_val - normalizer);
			picks->branch_tokens[step][0] = candidates[step * G53F_TOP_K + runner];
			picks->branch_scores[step][0] = expf(runner_val - normalizer);
		}
		if ( tree != 0u )
		{
			uint32_t branch_lane = runner;
			for ( uint32_t k = 1u; k < G53F_BRANCH_LEN && step + k < G53F_DRAFTS; k++ )
			{
				const uint32_t next_step = step + k;
				const float bval = edges[(next_step * G53F_TOP_K + branch_lane) * G53F_TOP_K + lane];
				float bmax = bval, bsum, blse, bnorm, best_bval;
				unsigned long long bkey = ((unsigned long long)SparkMathSampleOrdered(bval) << 32u) | (unsigned long long)(0xffffffffu - lane);
				for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
				{
					const unsigned long long other = __shfl_xor_sync(active, bkey, mask);
					bkey = other > bkey ? other : bkey;
					bmax = fmaxf(bmax, __shfl_xor_sync(active, bmax, mask));
				}
				bsum = expf(bval - bmax);
				for ( uint32_t mask = G53F_TOP_K / 2u; mask > 0u; mask >>= 1u )
					bsum += __shfl_xor_sync(active, bsum, mask);
				blse = bmax + logf(bsum);
				bnorm = G53FLogAddExp(blse, tail[next_step]);
				branch_lane = 0xffffffffu - (uint32_t)bkey;
				best_bval = __shfl_sync(active, bval, branch_lane);
				if ( lane == 0u )
				{
					picks->branch_tokens[step][k] = candidates[next_step * G53F_TOP_K + branch_lane];
					picks->branch_scores[step][k] = expf(best_bval - bnorm);
				}
			}
		}
		previous = choice;
	}
}

static SparkStatus G53FRefuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	SPARK_FAIL(status);
}

static SparkStatus G53FLoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
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
	if ( magic != G53F_PACK_MAGIC || version != G53F_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
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
	*payload_base = (sizeof(header) + bytes + G53F_PACK_ALIGNMENT - 1u) / G53F_PACK_ALIGNMENT * G53F_PACK_ALIGNMENT;
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FTensor(const G53FState *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t expected,
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
		fprintf(stderr, "glm53 flash dflash2: pack tensor %s is missing or is not %llu bf16 elements\n", name, (unsigned long long)expected);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*pointer = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static int32_t G53FMember(const SparkJsonDocument *document, int32_t object, const char *name)
{
	return object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
}

static SparkStatus G53FShape(G53FState *state, const SparkJsonDocument *document)
{
	const int32_t config = G53FMember(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = G53FMember(document, config, "checkpoint_config");
	const int32_t dflash = G53FMember(document, checkpoint, "dflash_config");
	const int32_t rope = G53FMember(document, checkpoint, "rope_parameters");
	const int32_t layers = G53FMember(document, dflash, "target_layer_ids");
	const int32_t layer_types = G53FMember(document, checkpoint, "layer_types");
	const struct
	{
		int32_t object;
		const char *name;
		uint64_t value;
	} expected[] = {
		{checkpoint, "hidden_size", G53F_HIDDEN},
		{checkpoint, "intermediate_size", G53F_MLP},
		{checkpoint, "num_hidden_layers", G53F_LAYERS},
		{checkpoint, "num_attention_heads", G53F_HEADS},
		{checkpoint, "num_key_value_heads", G53F_KV_HEADS},
		{checkpoint, "head_dim", G53F_HEAD_DIM},
		{checkpoint, "sliding_window", G53F_SLIDING},
		{checkpoint, "vocab_size", G53F_VOCAB},
		{dflash, "block_size", G53F_BLOCK},
		{dflash, "conv_group_size", G53F_CONV_GROUP},
		{dflash, "conv_kernel_size", G53F_CONV_TAPS},
		{dflash, "selector_rank", G53F_SELECTOR_RANK},
		{dflash, "selector_top_k", G53F_TOP_K},
	};
	uint64_t value = 0u;
	float theta = 0.0f;
	bool is_causal = true;
	int32_t token;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		token = G53FMember(document, expected[index].object, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "glm53 flash dflash2: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	token = G53FMember(document, checkpoint, "is_causal");
	if ( token < 0 || SparkJsonGetBoolean(document, token, &is_causal) != SPARK_STATUS_OK || is_causal )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( layers < 0 || SparkJsonGetArrayElementCount(document, layers) != G53F_TAPS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( layer_types < 0 || SparkJsonGetArrayElementCount(document, layer_types) != G53F_LAYERS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t index = 0u; index < G53F_LAYERS; index++ )
	{
		const int32_t item = SparkJsonGetArrayElement(document, layer_types, (int32_t)index);
		if ( item < 0 || !SparkJsonStringEquals(document, item, "sliding_attention") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	token = G53FMember(document, dflash, "mask_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= G53F_VOCAB )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->mask_token = (uint32_t)value;
	token = G53FMember(document, checkpoint, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = G53FMember(document, rope, "rope_type");
	if ( token >= 0 && !SparkJsonStringEquals(document, token, "default") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = G53FMember(document, rope, "rope_theta");
	if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK || theta <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->softmax_scale = (float)(1.0 / sqrt((double)G53F_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, G53F_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	LM_LAUNCH((G53FRopeTableKernel), 1u, G53F_THREADS, 0, state->stream, state->rope_inverse, theta);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FWeights(G53FState *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = G53FLoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = G53FShape(state, &document);
	{
		const struct
		{
			const char *name;
			uint64_t elements;
			const uint16_t **pointer;
		} plain[] = {
			{"fc.weight", (uint64_t)G53F_HIDDEN * G53F_TAP_WIDTH, &state->fc},
			{"hidden_norm.weight", G53F_HIDDEN, &state->hidden_norm},
			{"norm.weight", G53F_HIDDEN, &state->norm},
			{"target.embed", (uint64_t)G53F_VOCAB * G53F_HIDDEN, &state->embed},
			{"target.head", (uint64_t)G53F_VOCAB * G53F_HIDDEN, &state->head},
			{"candidate_selector.hidden_projection.weight", (uint64_t)G53F_SELECTOR_RANK * G53F_HIDDEN, &state->selector_hidden},
			{"candidate_selector.predecessor_codebook", (uint64_t)G53F_VOCAB * G53F_SELECTOR_RANK, &state->predecessors},
			{"candidate_selector.successor_codebook", (uint64_t)G53F_VOCAB * G53F_SELECTOR_RANK, &state->successors},
		};
		for ( uint32_t index = 0u; index < sizeof(plain) / sizeof(plain[0]) && status == SPARK_STATUS_OK; index++ )
			status = G53FTensor(state, &document, payload_base, plain[index].name, plain[index].elements, plain[index].pointer);
	}
	for ( uint32_t layer = 0u; layer < G53F_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		G53FLayer *weights = &state->layers[layer];
		const struct
		{
			const char *suffix;
			uint64_t elements;
			const uint16_t **pointer;
		} tensors[] = {
			{"input_layernorm.weight", G53F_HIDDEN, &weights->input_norm},
			{"post_attention_layernorm.weight", G53F_HIDDEN, &weights->post_norm},
			{"attention_conv.base_kernel", (uint64_t)2u * G53F_CONV_TAPS * G53F_HIDDEN, &weights->attn_conv_base},
			{"attention_conv.kernel_projection.weight", (uint64_t)G53F_CONV_WIDTH * G53F_HIDDEN, &weights->attn_conv_proj},
			{"self_attn.q_proj.weight", (uint64_t)G53F_Q_WIDTH * G53F_HIDDEN, &weights->q},
			{"self_attn.k_proj.weight", (uint64_t)G53F_KV_WIDTH * G53F_HIDDEN, &weights->k},
			{"self_attn.v_proj.weight", (uint64_t)G53F_KV_WIDTH * G53F_HIDDEN, &weights->v},
			{"self_attn.o_proj.weight", (uint64_t)G53F_HIDDEN * G53F_Q_WIDTH, &weights->o},
			{"self_attn.q_norm.weight", G53F_HEAD_DIM, &weights->q_norm},
			{"self_attn.k_norm.weight", G53F_HEAD_DIM, &weights->k_norm},
			{"mlp_conv.base_kernel", (uint64_t)2u * G53F_CONV_TAPS * G53F_HIDDEN, &weights->mlp_conv_base},
			{"mlp_conv.kernel_projection.weight", (uint64_t)G53F_CONV_WIDTH * G53F_HIDDEN, &weights->mlp_conv_proj},
			{"mlp.gate_proj.weight", (uint64_t)G53F_MLP * G53F_HIDDEN, &weights->gate},
			{"mlp.up_proj.weight", (uint64_t)G53F_MLP * G53F_HIDDEN, &weights->up},
			{"mlp.down_proj.weight", (uint64_t)G53F_HIDDEN * G53F_MLP, &weights->down},
		};
		for ( uint32_t index = 0u; index < sizeof(tensors) / sizeof(tensors[0]) && status == SPARK_STATUS_OK; index++ )
		{
			char name[128];
			snprintf(name, sizeof(name), "layers.%u.%s", layer, tensors[index].suffix);
			status = G53FTensor(state, &document, payload_base, name, tensors[index].elements, tensors[index].pointer);
		}
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define G53F_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus G53FScratch(G53FState *state, uint64_t context_bytes)
{
	G53FForward *forward = &state->forward;
	int multiprocessors = 0;
	uint64_t capacity = context_bytes / G53F_SEQUENCE_BYTES;
	capacity = capacity < 1u ? 1u : capacity > G53F_SEQUENCES_MAX ? G53F_SEQUENCES_MAX : capacity;
	state->capacity = (uint32_t)capacity;
	state->sequences = (G53FSequence *)calloc(capacity, sizeof(G53FSequence));
	if ( state->sequences == 0 || SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)G53F_INGEST_ROWS * G53F_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)G53F_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(G53FDraft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(G53FPicks)) != SPARK_GPU_SUCCESS ||
		!G53F_DEVICE(state->caches, capacity * G53F_SEQUENCE_ELEMENTS) || !G53F_DEVICE(state->taps, (uint64_t)G53F_INGEST_ROWS * G53F_TAP_WIDTH) ||
		!G53F_DEVICE(state->positions, G53F_INGEST_ROWS) || !G53F_DEVICE(state->features_raw, (uint64_t)G53F_INGEST_ROWS * G53F_HIDDEN) ||
		!G53F_DEVICE(state->features, (uint64_t)G53F_INGEST_ROWS * G53F_HIDDEN) || !G53F_DEVICE(state->keys_raw, (uint64_t)G53F_INGEST_ROWS * G53F_KV_WIDTH) ||
		!G53F_DEVICE(state->values_raw, (uint64_t)G53F_INGEST_ROWS * G53F_KV_WIDTH) || !G53F_DEVICE(forward->draft, 1u) || !G53F_DEVICE(forward->picks, 1u) ||
		!G53F_DEVICE(forward->hidden, G53F_BLOCK * G53F_HIDDEN) || !G53F_DEVICE(forward->x, G53F_BLOCK * G53F_HIDDEN) ||
		!G53F_DEVICE(forward->conv_coeff, G53F_BLOCK * G53F_CONV_WIDTH) || !G53F_DEVICE(forward->prepared, G53F_BLOCK * G53F_HIDDEN) ||
		!G53F_DEVICE(forward->q, G53F_BLOCK * G53F_Q_WIDTH) || !G53F_DEVICE(forward->q_roped, G53F_BLOCK * G53F_Q_WIDTH) ||
		!G53F_DEVICE(forward->k, G53F_BLOCK * G53F_KV_WIDTH) || !G53F_DEVICE(forward->v, G53F_BLOCK * G53F_KV_WIDTH) ||
		!G53F_DEVICE(forward->partials, (uint64_t)G53F_HEADS * G53F_SPLITS * G53F_BLOCK * (G53F_HEAD_DIM + 2u)) ||
		!G53F_DEVICE(forward->attended, G53F_BLOCK * G53F_Q_WIDTH) || !G53F_DEVICE(forward->projected, G53F_BLOCK * G53F_HIDDEN) ||
		!G53F_DEVICE(forward->gate, G53F_BLOCK * G53F_MLP) || !G53F_DEVICE(forward->up, G53F_BLOCK * G53F_MLP) ||
		!G53F_DEVICE(forward->act, G53F_BLOCK * G53F_MLP) || !G53F_DEVICE(forward->mlp, G53F_BLOCK * G53F_HIDDEN) ||
		!G53F_DEVICE(forward->y, G53F_BLOCK * G53F_HIDDEN) || !G53F_DEVICE(forward->logits, (uint64_t)G53F_DRAFTS * G53F_VOCAB) ||
		!G53F_DEVICE(forward->logits_f32, (uint64_t)G53F_DRAFTS * G53F_VOCAB) || !G53F_DEVICE(forward->sel_hidden, G53F_DRAFTS * G53F_SELECTOR_RANK) ||
		!G53F_DEVICE(forward->candidates, G53F_DRAFTS * G53F_TOP_K) || !G53F_DEVICE(forward->unary, G53F_DRAFTS * G53F_TOP_K) ||
		!G53F_DEVICE(forward->tail, G53F_DRAFTS) || !G53F_DEVICE(forward->edges, G53F_DRAFTS * G53F_TOP_K * G53F_TOP_K) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		state->sequences[index].cache = state->caches + (uint64_t)index * G53F_SEQUENCE_ELEMENTS;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static uint16_t *G53FKeys(const G53FSequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer) * G53F_LAYER_KV;
}

static uint16_t *G53FValues(const G53FSequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer + 1u) * G53F_LAYER_KV;
}

static void G53FReset(G53FSequence *sequence)
{
	for ( uint32_t slot = 0u; slot < G53F_WINDOW; slot++ )
	{
		sequence->positions[slot] = -1;
		sequence->tokens[slot] = 0u;
	}
	sequence->retired = 0u;
}

static G53FSequence *G53FLive(G53FState *state, uint64_t id)
{
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		if ( state->sequences[index].live != 0u && state->sequences[index].retired == 0u && state->sequences[index].id == id )
			return &state->sequences[index];
	return 0;
}

static G53FSequence *G53FClaim(G53FState *state)
{
	G53FSequence *oldest = 0;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
	{
		G53FSequence *sequence = &state->sequences[index];
		if ( sequence->live == 0u )
			return sequence;
		if ( oldest == 0 || sequence->last_used < oldest->last_used )
			oldest = sequence;
	}
	return oldest;
}

static uint32_t G53FHeld(const G53FSequence *sequence, uint64_t position)
{
	return sequence->positions[position % G53F_WINDOW] == (int64_t)position ? 1u : 0u;
}

static SparkStatus G53FGemm(G53FState *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FGemmF32(G53FState *state, const uint16_t *weight, const uint16_t *input, float *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, (uint16_t *)0, output, rows, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FIngestRows(G53FState *state, G53FSequence *sequence, uint32_t rows)
{
	SparkStatus status = G53FGemmF32(state, state->fc, state->taps, state->features_raw, rows, G53F_TAP_WIDTH, G53F_HIDDEN);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((G53FRmsF32Kernel), rows, G53F_THREADS, 0, state->stream, state->features_raw, state->hidden_norm, state->features, G53F_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < G53F_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = G53FGemm(state, state->layers[layer].k, state->features, state->keys_raw, rows, G53F_HIDDEN, G53F_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = G53FGemm(state, state->layers[layer].v, state->features, state->values_raw, rows, G53F_HIDDEN, G53F_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FHeadNormRopeKernel), rows * G53F_KV_HEADS, G53F_HEAD_DIM, 0, state->stream, state->keys_raw, state->layers[layer].k_norm,
			state->rope_inverse, state->positions, G53F_KV_HEADS, state->epsilon, (uint16_t *)0, G53FKeys(sequence, layer), 0u, 1u);
		LM_LAUNCH((G53FStoreValuesKernel), rows, G53F_THREADS, 0, state->stream, state->values_raw, state->positions, 0u, G53FValues(sequence, layer));
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static SparkStatus G53FIngest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	G53FState *state = (G53FState *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	const uint32_t skip = rows > G53F_WINDOW ? rows - G53F_WINDOW : 0u;
	const uint64_t first = frame->header.first_position + skip;
	G53FSequence *sequence;
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != G53F_TAPS || hidden != G53F_HIDDEN || frame->rows == 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t tap = 0u; tap < G53F_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * G53F_HIDDEN > frame->row_elements )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( frame->header.first_position + rows > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = G53FLive(state, frame->header.sequence_id);
	if ( sequence != 0 && frame->header.first_position != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence->last_used = ++state->clock;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = G53FClaim(state);
		G53FReset(sequence);
		sequence->live = 1u;
		sequence->id = frame->header.sequence_id;
	}
	else if ( frame->header.first_position == 0u )
		G53FReset(sequence);
	sequence->control_generation = frame->header.control_generation;
	sequence->request_id = frame->header.request_id;
	sequence->last_used = ++state->clock;
	for ( uint32_t row = skip; row < rows; row++ )
		sequence->positions[(frame->header.first_position + row) % G53F_WINDOW] = -1;
	for ( uint32_t done = skip; done < rows && status == SPARK_STATUS_OK; done += G53F_INGEST_ROWS )
	{
		const uint32_t part = rows - done < G53F_INGEST_ROWS ? rows - done : G53F_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < G53F_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * G53F_TAP_WIDTH + (uint64_t)tap * G53F_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * G53F_HIDDEN, G53F_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(frame->header.first_position + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * G53F_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t),
			SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		{
			(void)SparkGpuStreamSynchronize(state->stream);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		status = G53FIngestRows(state, sequence, part);
	}
	if ( status == SPARK_STATUS_OK )
	{
		for ( uint32_t slot = 0u; slot < G53F_WINDOW; slot++ )
			if ( sequence->positions[slot] >= (int64_t)first )
				sequence->positions[slot] = -1;
		for ( uint32_t row = skip; row < rows; row++ )
		{
			const uint64_t position = frame->header.first_position + row;
			sequence->positions[position % G53F_WINDOW] = (int64_t)position;
			sequence->tokens[position % G53F_WINDOW] = frame->tokens != 0 ? frame->tokens[row] : 0u;
		}
	}
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus G53FCopySlots(G53FState *state, G53FSequence *sequence, const G53FSequence *donor, uint32_t slot, uint32_t count)
{
	for ( uint32_t layer = 0u; layer < G53F_LAYERS; layer++ )
		for ( uint32_t head = 0u; head < G53F_KV_HEADS; head++ )
		{
			const uint64_t at = ((uint64_t)head * G53F_COLUMNS + slot) * G53F_HEAD_DIM;
			const size_t bytes = (size_t)count * G53F_HEAD_DIM * sizeof(uint16_t);
			if ( SparkGpuMemcpyAsync(G53FKeys(sequence, layer) + at, G53FKeys(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) !=
				SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(G53FValues(sequence, layer) + at, G53FValues(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	return SPARK_STATUS_OK;
}

static uint32_t G53FUsable(const G53FSequence *sequence, const G53FSequence *donor, const SparkDraftServerRequest *request, uint64_t position)
{
	return G53FHeld(sequence, position) == 0u && G53FHeld(donor, position) != 0u && position < request->committed_token_count &&
		donor->tokens[position % G53F_WINDOW] == request->committed_token_ids[position] ? 1u : 0u;
}

static SparkStatus G53FFillFromDonors(G53FState *state, G53FSequence *sequence, const SparkDraftServerRequest *request, uint64_t first)
{
	uint8_t tried[G53F_SEQUENCES_MAX];
	memset(tried, 0, sizeof(tried));
	for ( ;; )
	{
		G53FSequence *donor = 0;
		uint32_t chosen = 0u, missing = 0u;
		uint64_t position = first;
		for ( uint64_t at = first; at < request->position; at++ )
			missing += G53FHeld(sequence, at) == 0u ? 1u : 0u;
		if ( missing == 0u )
			return SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < state->capacity; index++ )
		{
			G53FSequence *other = &state->sequences[index];
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
			if ( G53FUsable(sequence, donor, request, position) == 0u )
			{
				position++;
				continue;
			}
			while ( stop < request->position && G53FUsable(sequence, donor, request, stop) != 0u && (stop == position || stop % G53F_WINDOW != 0u) )
				stop++;
			if ( G53FCopySlots(state, sequence, donor, (uint32_t)(position % G53F_WINDOW), (uint32_t)(stop - position)) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			for ( uint64_t at = position; at < stop; at++ )
			{
				sequence->positions[at % G53F_WINDOW] = (int64_t)at;
				sequence->tokens[at % G53F_WINDOW] = donor->tokens[at % G53F_WINDOW];
			}
			position = stop;
		}
	}
}

static void G53FSpan(G53FSequence *sequence, uint64_t checked, uint64_t position, G53FDraft *draft)
{
	const uint64_t count = position - checked;
	const uint32_t first = count != 0u ? (uint32_t)(checked % G53F_WINDOW) : 0u, end = first + (uint32_t)count;
	const uint32_t need_mask = position > G53F_SLIDING - G53F_BLOCK ? 1u : 0u;
	uint32_t newer = 0u;
	memset(draft->valid, 0, sizeof(draft->valid));
	draft->masked = 0u;
	for ( uint32_t slot = end; slot < G53F_WINDOW && slot < end + G53F_BLOCK; slot++ )
		newer |= count != 0u && sequence->positions[slot] > (int64_t)(position - 1u) ? 1u : 0u;
	if ( need_mask == 0u && end <= G53F_WINDOW && newer == 0u )
	{
		for ( uint32_t slot = end; slot < G53F_WINDOW && slot < end + G53F_BLOCK; slot++ )
			sequence->positions[slot] = -1;
		draft->start = first;
		draft->block_at = end;
		return;
	}
	for ( uint64_t at = checked; at < position; at++ )
	{
		const uint64_t gap = position - at;
		const uint32_t max_row = gap <= G53F_SLIDING - G53F_BLOCK ? G53F_BLOCK - 1u : (uint32_t)((G53F_SLIDING - 1u) - gap);
		draft->valid[at % G53F_WINDOW] = (uint8_t)((1u << (max_row + 1u)) - 1u);
	}
	draft->masked = 1u;
	draft->start = 0u;
	draft->block_at = G53F_WINDOW;
}

static SparkStatus G53FPrepare(G53FState *state, G53FSequence *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
	const SparkRowSampling *sampling)
{
	G53FDraft *draft = state->forward.host_draft;
	if ( anchor >= G53F_VOCAB || position + G53F_BLOCK > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft, 0, offsetof(G53FDraft, valid));
	for ( uint32_t row = 0u; row < G53F_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	draft->anchor = anchor;
	draft->mask = state->mask_token;
	draft->greedy = sampling->inverse_temperature > 0.0f ? 0u : 1u;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	G53FSpan(sequence, checked, position, draft);
	if ( SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FRunForward(G53FState *state, const G53FSequence *sequence, uint32_t layers)
{
	G53FForward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(G53FDraft, positions));
	const uint32_t block_at = state->forward.host_draft->block_at;
	const uint32_t span = block_at - state->forward.host_draft->start;
	const uint32_t splits = (span + G53F_SPLIT_KEYS - 1u) / G53F_SPLIT_KEYS;
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((G53FEmbedKernel), G53F_BLOCK, G53F_THREADS, 0, stream, state->embed, forward->draft, forward->hidden);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const G53FLayer *weights = &state->layers[layer];
		LM_LAUNCH((G53FRmsF32Kernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->hidden, weights->input_norm, forward->x, G53F_HIDDEN, state->epsilon);
		status = G53FGemm(state, weights->attn_conv_proj, forward->x, forward->conv_coeff, G53F_BLOCK, G53F_HIDDEN, G53F_CONV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FConvPrepareKernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->x, weights->attn_conv_base, forward->conv_coeff, forward->prepared);
		status = G53FGemm(state, weights->q, forward->prepared, forward->q, G53F_BLOCK, G53F_HIDDEN, G53F_Q_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = G53FGemm(state, weights->k, forward->prepared, forward->k, G53F_BLOCK, G53F_HIDDEN, G53F_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = G53FGemm(state, weights->v, forward->prepared, forward->v, G53F_BLOCK, G53F_HIDDEN, G53F_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FHeadNormRopeKernel), G53F_BLOCK * G53F_HEADS, G53F_HEAD_DIM, 0, stream, forward->q, weights->q_norm, state->rope_inverse, positions,
			G53F_HEADS, state->epsilon, forward->q_roped, (uint16_t *)0, 0u, 0u);
		LM_LAUNCH((G53FHeadNormRopeKernel), G53F_BLOCK * G53F_KV_HEADS, G53F_HEAD_DIM, 0, stream, forward->k, weights->k_norm, state->rope_inverse, positions,
			G53F_KV_HEADS, state->epsilon, (uint16_t *)0, G53FKeys(sequence, layer), block_at, 0u);
		LM_LAUNCH((G53FStoreValuesKernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->v, (const uint32_t *)0, block_at, G53FValues(sequence, layer));
		if ( splits != 0u )
			LM_LAUNCH((G53FAttentionSplitKernel), dim3(G53F_HEADS, splits), G53F_ATTN_THREADS, 0, stream, forward->q_roped, G53FKeys(sequence, layer),
				G53FValues(sequence, layer), forward->draft, state->softmax_scale, forward->partials);
		LM_LAUNCH((G53FAttentionMergeKernel), G53F_BLOCK * G53F_HEADS, G53F_HEAD_DIM, 0, stream, forward->q_roped, G53FKeys(sequence, layer),
			G53FValues(sequence, layer), forward->draft, state->softmax_scale, forward->partials, splits, forward->attended);
		status = G53FGemm(state, weights->o, forward->attended, forward->projected, G53F_BLOCK, G53F_Q_WIDTH, G53F_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FConvFinishResidualKernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->projected, weights->attn_conv_base, forward->conv_coeff,
			forward->hidden);
		LM_LAUNCH((G53FRmsF32Kernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->hidden, weights->post_norm, forward->x, G53F_HIDDEN, state->epsilon);
		status = G53FGemm(state, weights->mlp_conv_proj, forward->x, forward->conv_coeff, G53F_BLOCK, G53F_HIDDEN, G53F_CONV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FConvPrepareKernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->x, weights->mlp_conv_base, forward->conv_coeff, forward->prepared);
		status = G53FGemm(state, weights->gate, forward->prepared, forward->gate, G53F_BLOCK, G53F_HIDDEN, G53F_MLP);
		if ( status == SPARK_STATUS_OK )
			status = G53FGemm(state, weights->up, forward->prepared, forward->up, G53F_BLOCK, G53F_HIDDEN, G53F_MLP);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FSwigluKernel), 2u * state->multiprocessors, G53F_THREADS, 0, stream, forward->gate, forward->up, forward->act, G53F_BLOCK * G53F_MLP);
		status = G53FGemm(state, weights->down, forward->act, forward->mlp, G53F_BLOCK, G53F_MLP, G53F_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FConvFinishResidualKernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->mlp, weights->mlp_conv_base, forward->conv_coeff, forward->hidden);
	}
	if ( status != SPARK_STATUS_OK || layers < G53F_LAYERS )
		return status;
	LM_LAUNCH((G53FRmsF32Kernel), G53F_BLOCK, G53F_THREADS, 0, stream, forward->hidden, state->norm, forward->y, G53F_HIDDEN, state->epsilon);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FHeads(G53FState *state, uint32_t tree)
{
	G53FForward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint16_t *rows = forward->y + G53F_HIDDEN;
	SparkStatus status = G53FGemm(state, state->head, rows, forward->logits, G53F_DRAFTS, G53F_HIDDEN, G53F_VOCAB);
	if ( status == SPARK_STATUS_OK )
		status = G53FGemm(state, state->selector_hidden, rows, forward->sel_hidden, G53F_DRAFTS, G53F_HIDDEN, G53F_SELECTOR_RANK);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((G53FWidenKernel), 4u * state->multiprocessors, G53F_THREADS, 0, stream, forward->logits, forward->logits_f32, G53F_DRAFTS * G53F_VOCAB);
	if ( LmTopkExactRowsLaunch<G53F_TOPK_THREADS>(forward->logits_f32, G53F_DRAFTS, G53F_VOCAB, G53F_TOP_K, forward->candidates, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LM_LAUNCH((G53FLogitsSummaryKernel), G53F_DRAFTS, G53F_THREADS, 0, stream, forward->logits_f32, forward->candidates, forward->unary, forward->tail);
	LM_LAUNCH((G53FLatticeKernel), G53F_DRAFTS * G53F_TOP_K, G53F_THREADS, 0, stream, forward->draft, forward->candidates, forward->unary, forward->sel_hidden,
		state->predecessors, state->successors, forward->edges);
	LM_LAUNCH((G53FWalkKernel), 1u, G53F_WARP, 0, stream, forward->draft, forward->candidates, forward->tail, forward->edges, tree, forward->picks);
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(G53FPicks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FSettle(G53FState *state, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static void G53FForwardFree(G53FForward *forward)
{
	void *buffers[] = {
		forward->draft, forward->picks, forward->hidden, forward->x, forward->conv_coeff, forward->prepared, forward->q, forward->q_roped, forward->k,
		forward->v, forward->partials, forward->attended, forward->projected, forward->gate, forward->up, forward->act, forward->mlp, forward->y,
		forward->logits, forward->logits_f32, forward->sel_hidden, forward->candidates, forward->unary, forward->tail, forward->edges,
	};
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		(void)SparkGpuFree(buffers[index]);
	if ( forward->host_draft != 0 )
		(void)SparkGpuFreeHost(forward->host_draft);
	if ( forward->host_picks != 0 )
		(void)SparkGpuFreeHost(forward->host_picks);
}

static SparkStatus G53FClose(void *opaque)
{
	G53FState *state = (G53FState *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	G53FForwardFree(&state->forward);
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

static SparkStatus G53FOpen(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	G53FState *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return G53FRefuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	{
		const SparkStatus device = SparkDeviceInitialize(0);
		if ( (device != SPARK_STATUS_OK && device != SPARK_STATUS_DUPLICATE) || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
			return G53FRefuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
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
	if ( lane->context_bytes < G53F_SEQUENCE_BYTES )
		return G53FRefuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one sequence's window", lane->lane_name);
	state = (G53FState *)calloc(1u, sizeof(*state));
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
		status = G53FWeights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = G53FScratch(state, lane->context_bytes);
	if ( status == SPARK_STATUS_OK )
		status = G53FSettle(state, SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a GLM-5.3-Flash DFlash2 pack this module reads", lane->lane_name, lane->pack_path);
		(void)G53FClose(state);
		SPARK_FAIL(status);
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s with %u sequence windows", lane->lane_name, lane->pack_path, state->capacity);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus G53FPropose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	G53FState *state = (G53FState *)opaque;
	G53FSequence *sequence = G53FLive(state, request->sequence_id);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > G53F_SLIDING - 1u ? request->position - (G53F_SLIDING - 1u) : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( sequence == 0 || (request->position != 0u && G53FHeld(sequence, request->position - 1u) == 0u) )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	status = G53FFillFromDonors(state, sequence, request, first);
	if ( status != SPARK_STATUS_OK )
		return G53FSettle(state, status);
	for ( uint64_t position = first; position < request->position; position++ )
		if ( G53FHeld(sequence, position) == 0u )
			checked = position + 1u;
	if ( checked != first )
		state->gaps++;
	if ( request->committed_token_count < request->position )
		return G53FSettle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	for ( uint64_t position = checked; position < request->position; position++ )
		if ( sequence->tokens[position % G53F_WINDOW] != request->committed_token_ids[position] )
			return G53FSettle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	drafts = drafts < G53F_DRAFTS ? drafts : G53F_DRAFTS;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		SparkDraftTreeBranch branches[G53F_DRAFTS];
		uint32_t branch_count = 0u;
		status = G53FPrepare(state, sequence, request->anchor_token_id, request->position, checked, &request->sampling);
		if ( status == SPARK_STATUS_OK )
			status = G53FRunForward(state, sequence, G53F_LAYERS);
		if ( status == SPARK_STATUS_OK )
			status = G53FHeads(state, tree);
		status = G53FSettle(state, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				const uint32_t len = drafts - step < G53F_BRANCH_LEN ? drafts - step : G53F_BRANCH_LEN;
				branches[branch_count].fork_step = step;
				branches[branch_count].length = len;
				for ( uint32_t k = 0u; k < len; k++ )
				{
					branches[branch_count].tokens[k] = state->forward.host_picks->branch_tokens[step][k];
					branches[branch_count].scores[k] = state->forward.host_picks->branch_scores[step][k];
				}
				branch_count++;
			}
		*count = SparkDraftTreePackBranches(request->anchor_token_id, source, state->forward.host_picks->tokens, state->forward.host_picks->scores,
			branches, branch_count, drafts, requested, limit, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE, nodes);
	}
	return SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS;
}

static const SparkDrafterModuleInterface g53f_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"glm53-flash-dflash2",
	G53FOpen,
	G53FIngest,
	G53FPropose,
	G53FClose,
};

static SparkStatus G53FReadback(void *host, const void *device, size_t bytes)
{
	if ( SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &g53f_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_flash_dflash2_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	G53FState *state = (G53FState *)opaque;
	const G53FSequence *sequence = state != 0 ? G53FLive(state, sequence_id) : 0;
	SparkStatus status;
	if ( sequence == 0 || layer >= G53F_LAYERS || keys == 0 || values == 0 || positions == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memcpy(positions, sequence->positions, sizeof(sequence->positions));
	status = G53FReadback(keys, G53FKeys(sequence, layer), G53F_LAYER_KV * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = G53FReadback(values, G53FValues(sequence, layer), G53F_LAYER_KV * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_flash_dflash2_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	G53FState *state = (G53FState *)opaque;
	G53FSequence *sequence = state != 0 ? G53FLive(state, sequence_id) : 0;
	const uint64_t started = SparkMonotonicNs(), first = position > G53F_SLIDING - 1u ? position - (G53F_SLIDING - 1u) : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	for ( uint64_t at = first; at < position; at++ )
		if ( G53FHeld(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = G53FPrepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = G53FRunForward(state, sequence, G53F_LAYERS);
	status = G53FSettle(state, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = G53FReadback(y, state->forward.y, (size_t)G53F_BLOCK * G53F_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_flash_dflash2_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, float *hidden)
{
	G53FState *state = (G53FState *)opaque;
	G53FSequence *sequence = state != 0 ? G53FLive(state, sequence_id) : 0;
	const uint64_t first = position > G53F_SLIDING - 1u ? position - (G53F_SLIDING - 1u) : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 || hidden == 0 || layers == 0u || layers > G53F_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint64_t at = first; at < position; at++ )
		if ( G53FHeld(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = G53FPrepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = G53FRunForward(state, sequence, layers);
	status = G53FSettle(state, status);
	if ( status == SPARK_STATUS_OK )
		status = G53FReadback(hidden, state->forward.hidden, (size_t)G53F_BLOCK * G53F_HIDDEN * sizeof(float));
	return status;
}
