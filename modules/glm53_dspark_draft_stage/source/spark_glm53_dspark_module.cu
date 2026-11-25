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

#define G53_HIDDEN 6144u
#define G53_TAPS 5u
#define G53_TAP_WIDTH (G53_TAPS * G53_HIDDEN)
#define G53_LAYERS 3u
#define G53_HEADS 64u
#define G53_KV_HEADS 64u
#define G53_HEAD_DIM 64u
#define G53_HALF (G53_HEAD_DIM / 2u)
#define G53_Q_WIDTH (G53_HEADS * G53_HEAD_DIM)
#define G53_KV_WIDTH (G53_KV_HEADS * G53_HEAD_DIM)
#define G53_MLP 12288u
#define G53_BLOCK 8u
#define G53_SLIDING 2048u
#define G53_WINDOW (G53_SLIDING - G53_BLOCK)
#define G53_COLUMNS (G53_WINDOW + G53_BLOCK)
#define G53_VOCAB 154880u
#define G53_MARKOV 256u
#define G53_CONFIDENCE (G53_HIDDEN + G53_MARKOV)
#define G53_CANDIDATES 2048u
#define G53_SAMPLED 32u
#define G53_BRANCHES 2u
#define G53_BRANCH_LEN 4u
#define G53_INGEST_ROWS 256u
#define G53_THREADS 256u
#define G53_WARP 32u
#define G53_WARPS (G53_THREADS / G53_WARP)
#define G53_PICK_THREADS 1024u
#define G53_TOPK_THREADS 512u
#define G53_SPLIT_KEYS 256u
#define G53_SPLITS ((G53_WINDOW + G53_SPLIT_KEYS - 1u) / G53_SPLIT_KEYS)
#define G53_ATTN_TILE 32u
#define G53_ATTN_THREADS (G53_BLOCK * G53_WARP)
#define G53_SEQUENCES_MAX 256u
#define G53_LAYER_KV ((uint64_t)G53_KV_HEADS * G53_COLUMNS * G53_HEAD_DIM)
#define G53_SEQUENCE_ELEMENTS (2u * G53_LAYERS * G53_LAYER_KV)
#define G53_SEQUENCE_BYTES (G53_SEQUENCE_ELEMENTS * sizeof(uint16_t))
#define G53_PACK_MAGIC 0x4B505244u
#define G53_PACK_VERSION 1u
#define G53_PACK_ALIGNMENT 128u
#define G53_INVERSE_PI 0.31830988618379067154

typedef struct G53Layer
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
} G53Layer;

typedef struct G53Sequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t last_used;
	uint32_t live;
	uint32_t retired;
	int64_t positions[G53_WINDOW];
	uint32_t tokens[G53_WINDOW];
	uint16_t *cache;
} G53Sequence;

typedef struct G53Draft
{
	uint64_t seed;
	uint32_t positions[G53_BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint32_t start;
	uint32_t block_at;
	uint32_t masked;
	uint8_t valid[G53_WINDOW];
} G53Draft;

typedef struct G53Picks
{
	uint32_t tokens[G53_BLOCK];
	float scores[G53_BLOCK];
	uint32_t alt_tokens[G53_BLOCK];
	float alt_scores[G53_BLOCK];
	uint32_t branch_fork[G53_BRANCHES];
	uint32_t branch_len[G53_BRANCHES];
	uint32_t branch_tokens[G53_BRANCHES][G53_BRANCH_LEN];
	float branch_scores[G53_BRANCHES][G53_BRANCH_LEN];
} G53Picks;

typedef struct G53Forward
{
	G53Draft *host_draft;
	G53Draft *draft;
	G53Picks *host_picks;
	G53Picks *picks;
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
} G53Forward;

typedef struct G53State
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
	const uint16_t *markov_previous;
	const uint16_t *markov_next;
	const uint16_t *confidence_weight;
	const uint16_t *confidence_bias;
	G53Layer layers[G53_LAYERS];
	G53Forward forward;
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
	G53Sequence *sequences;
	uint64_t frames;
	uint64_t rows;
	uint64_t gaps;
} G53State;

static __device__ __forceinline__ void G53Rotation(uint32_t position, float inverse, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * G53_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	sincospif(reduced, sine, cosine);
}

static __device__ __forceinline__ float G53WarpSum(float value)
{
	for ( uint32_t mask = G53_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ float G53WarpMax(float value)
{
	for ( uint32_t mask = G53_WARP / 2u; mask > 0u; mask >>= 1u )
		value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
	return value;
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53RopeTableKernel(float *__restrict__ inverse, float theta)
{
	if ( threadIdx.x < G53_HALF )
		inverse[threadIdx.x] = 1.0f / powf(theta, (float)(2u * threadIdx.x) / (float)G53_HEAD_DIM);
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53RmsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[G53_WARPS];
	const uint64_t row = blockIdx.x;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += G53_THREADS )
	{
		const float value = LmBf16ToFloat(input[row * dimension + index]);
		total += value * value;
	}
	total = LmBlockSum<G53_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += G53_THREADS )
	{
		const float normalized = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(input[row * dimension + index]) * scale));
		output[row * dimension + index] = LmFloatToBf16(normalized * LmBf16ToFloat(weight[index]));
	}
}

__global__ __launch_bounds__(G53_HEAD_DIM, 1)
void G53HeadNormRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, uint16_t *__restrict__ output, uint16_t *__restrict__ cache_keys,
	uint32_t block_at, uint32_t use_ring)
{
	__shared__ float reduction[G53_HEAD_DIM / G53_WARP];
	__shared__ float normed[G53_HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = LmBf16ToFloat(input[((uint64_t)row * heads + head) * G53_HEAD_DIM + lane]);
	float scale, cosine, sine, rotated;
	scale = rsqrtf(LmBlockSum<G53_HEAD_DIM>(value * value, reduction) / (float)G53_HEAD_DIM + epsilon);
	normed[lane] = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value * scale)) * LmBf16ToFloat(weight[lane])));
	__syncthreads();
	G53Rotation(position, inverse[lane % G53_HALF], &cosine, &sine);
	rotated = lane < G53_HALF ? normed[lane] * cosine - normed[lane + G53_HALF] * sine : normed[lane] * cosine + normed[lane - G53_HALF] * sine;
	if ( cache_keys != 0 )
	{
		const uint32_t column = use_ring != 0u ? position % G53_WINDOW : block_at + row;
		cache_keys[((uint64_t)head * G53_COLUMNS + column) * G53_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	}
	else
		output[((uint64_t)row * heads + head) * G53_HEAD_DIM + lane] = LmFloatToBf16(rotated);
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53StoreValuesKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions, uint32_t block_at, uint16_t *__restrict__ cache)
{
	const uint32_t row = blockIdx.x, column = positions != 0 ? positions[row] % G53_WINDOW : block_at + row;
	for ( uint32_t index = threadIdx.x; index < G53_KV_WIDTH; index += G53_THREADS )
		cache[((uint64_t)(index / G53_HEAD_DIM) * G53_COLUMNS + column) * G53_HEAD_DIM + index % G53_HEAD_DIM] = values[(uint64_t)row * G53_KV_WIDTH + index];
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53EmbedKernel(const uint16_t *__restrict__ embed, const G53Draft *__restrict__ draft, uint16_t *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	for ( uint32_t index = threadIdx.x; index < G53_HIDDEN; index += G53_THREADS )
		hidden[(uint64_t)row * G53_HIDDEN + index] = embed[(uint64_t)token * G53_HIDDEN + index];
}

__global__ __launch_bounds__(G53_ATTN_THREADS, 1)
void G53AttentionSplitKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ keys, const uint16_t *__restrict__ values,
	const G53Draft *__restrict__ draft, float scale, float *__restrict__ partials)
{
	__shared__ uint32_t s_k[G53_ATTN_TILE][G53_HALF + 1u];
	__shared__ uint32_t s_v[G53_ATTN_TILE][G53_HALF + 1u];
	__shared__ float s_scores[G53_BLOCK][G53_ATTN_TILE];
	__shared__ uint8_t s_vis[G53_ATTN_TILE];
	const uint32_t head = blockIdx.x, split = blockIdx.y;
	const uint32_t warp = threadIdx.x / G53_WARP, lane = threadIdx.x % G53_WARP;
	const uint32_t first = draft->start + split * G53_SPLIT_KEYS, end = draft->block_at;
	const uint32_t stop = first + G53_SPLIT_KEYS < end ? first + G53_SPLIT_KEYS : end;
	const uint32_t tiles = first < stop ? (stop - first + G53_ATTN_TILE - 1u) / G53_ATTN_TILE : 0u;
	const uint32_t masked = draft->masked;
	const uint16_t *head_keys = keys + (uint64_t)head * G53_COLUMNS * G53_HEAD_DIM;
	const uint16_t *head_values = values + (uint64_t)head * G53_COLUMNS * G53_HEAD_DIM;
	const uint32_t q_pair = *(const uint32_t *)(q + ((uint64_t)warp * G53_HEADS + head) * G53_HEAD_DIM + 2u * lane);
	const float q0 = LmBf16ToFloat((uint16_t)(q_pair & 0xffffu));
	const float q1 = LmBf16ToFloat((uint16_t)(q_pair >> 16u));
	float running_max = -INFINITY, running_sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
	float *out = partials + (((uint64_t)head * G53_SPLITS + split) * G53_BLOCK + warp) * (G53_HEAD_DIM + 2u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t tile_first = first + tile * G53_ATTN_TILE;
		const uint32_t load_key = threadIdx.x >> 3u, load_chunk = threadIdx.x & 7u;
		const uint32_t col = tile_first + load_key;
		if ( col < stop )
		{
			const uint4 kv_k = *(const uint4 *)(head_keys + (uint64_t)col * G53_HEAD_DIM + load_chunk * 8u);
			const uint4 kv_v = *(const uint4 *)(head_values + (uint64_t)col * G53_HEAD_DIM + load_chunk * 8u);
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
		if ( threadIdx.x < G53_ATTN_TILE )
			s_vis[threadIdx.x] = tile_first + threadIdx.x < stop && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
		__syncthreads();
		for ( uint32_t t = 0u; t < G53_ATTN_TILE; t++ )
		{
			const uint32_t kp = s_k[t][lane];
			const float k0 = LmBf16ToFloat((uint16_t)(kp & 0xffffu));
			const float k1 = LmBf16ToFloat((uint16_t)(kp >> 16u));
			const float dot = G53WarpSum(fmaf(q0, k0, q1 * k1));
			if ( lane == 0u )
				s_scores[warp][t] = s_vis[t] != 0u ? dot * scale : -INFINITY;
		}
		__syncwarp();
		{
			const float sc = s_scores[warp][lane];
			const float tile_max = G53WarpMax(sc);
			const float next_max = fmaxf(running_max, tile_max);
			const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
			const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
			const float tile_sum = G53WarpSum(weight);
			running_sum = running_sum * rescale + tile_sum;
			running_max = next_max;
			acc0 *= rescale;
			acc1 *= rescale;
			s_scores[warp][lane] = weight;
		}
		__syncwarp();
		for ( uint32_t t = 0u; t < G53_ATTN_TILE; t++ )
		{
			const float wt = s_scores[warp][t];
			const uint32_t vp = s_v[t][lane];
			acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp & 0xffffu)), acc0);
			acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp >> 16u)), acc1);
		}
		__syncthreads();
	}
	out[2u * lane] = acc0;
	out[2u * lane + 1u] = acc1;
	if ( lane == 0u )
	{
		out[G53_HEAD_DIM] = running_max;
		out[G53_HEAD_DIM + 1u] = running_sum;
	}
}

__global__ __launch_bounds__(G53_HEAD_DIM, 1)
void G53AttentionMergeKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ keys, const uint16_t *__restrict__ values,
	const G53Draft *__restrict__ draft, float scale, const float *__restrict__ partials, uint32_t splits, uint16_t *__restrict__ output)
{
	__shared__ float reduction[G53_HEAD_DIM / G53_WARP];
	const uint32_t row = blockIdx.x / G53_HEADS, head = blockIdx.x % G53_HEADS, lane = threadIdx.x;
	const uint32_t block_at = draft->block_at;
	const float own = LmBf16ToFloat(q[((uint64_t)row * G53_HEADS + head) * G53_HEAD_DIM + lane]);
	const uint16_t *head_keys = keys + (uint64_t)head * G53_COLUMNS * G53_HEAD_DIM;
	const uint16_t *head_values = values + (uint64_t)head * G53_COLUMNS * G53_HEAD_DIM;
	float top = -INFINITY, sum = 0.0f, accumulated = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part = partials + (((uint64_t)head * G53_SPLITS + split) * G53_BLOCK + row) * (G53_HEAD_DIM + 2u);
		const float part_top = part[G53_HEAD_DIM], next = fmaxf(top, part_top);
		float carry, weight;
		if ( part_top == -INFINITY )
			continue;
		carry = top == -INFINITY ? 0.0f : expf(top - next);
		weight = expf(part_top - next);
		sum = sum * carry + part[G53_HEAD_DIM + 1u] * weight;
		accumulated = accumulated * carry + part[lane] * weight;
		top = next;
	}
	for ( uint32_t key = 0u; key <= row; key++ )
	{
		const float score = LmBlockSum<G53_HEAD_DIM>(own * LmBf16ToFloat(head_keys[(uint64_t)(block_at + key) * G53_HEAD_DIM + lane]), reduction) * scale;
		const float next = fmaxf(top, score), carry = top == -INFINITY ? 0.0f : expf(top - next), weight = expf(score - next);
		sum = sum * carry + weight;
		accumulated = accumulated * carry + weight * LmBf16ToFloat(head_values[(uint64_t)(block_at + key) * G53_HEAD_DIM + lane]);
		top = next;
	}
	output[(uint64_t)row * G53_Q_WIDTH + head * G53_HEAD_DIM + lane] = LmFloatToBf16(accumulated / sum);
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53ResidualKernel(const uint16_t *__restrict__ base, const uint16_t *__restrict__ addend, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * G53_THREADS + threadIdx.x; index < count; index += gridDim.x * G53_THREADS )
		output[index] = LmFloatToBf16(LmBf16ToFloat(base[index]) + LmBf16ToFloat(addend[index]));
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53SwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * G53_THREADS + threadIdx.x; index < count; index += gridDim.x * G53_THREADS )
	{
		const float g = LmBf16ToFloat(gate[index]);
		output[index] = LmFloatToBf16(g / (1.0f + expf(-g)) * LmBf16ToFloat(up[index]));
	}
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53WidenKernel(const uint16_t *__restrict__ input, float *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * G53_THREADS + threadIdx.x; index < count; index += gridDim.x * G53_THREADS )
		output[index] = LmBf16ToFloat(input[index]);
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53MarkovRowKernel(uint32_t step, uint32_t anchor, const uint32_t *__restrict__ tokens, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ row)
{
	__shared__ float vector[G53_MARKOV];
	const uint32_t previous = step == 0u ? anchor : tokens[step - 1u];
	const uint32_t warp = threadIdx.x / G53_WARP, lane = threadIdx.x % G53_WARP;
	for ( uint32_t index = threadIdx.x; index < G53_MARKOV; index += G53_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * G53_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * G53_WARPS + warp; slot < G53_CANDIDATES; slot += gridDim.x * G53_WARPS )
	{
		const uint32_t candidate = candidates[step * G53_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * G53_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < G53_MARKOV; element += G53_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = G53WarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * G53_VOCAB + candidate] + sum;
	}
}

static __device__ __forceinline__ unsigned long long G53RowKey(const float *row, uint32_t slot)
{
	return ((unsigned long long)SparkMathSampleOrdered(row[slot]) << 32u) | (unsigned long long)(0xffffffffu - slot);
}

static __device__ __forceinline__ uint32_t G53KeySlot(unsigned long long key)
{
	return 0xffffffffu - (uint32_t)key;
}

static __device__ __forceinline__ void G53Top2Insert(unsigned long long candidate, unsigned long long *first, unsigned long long *second)
{
	if ( candidate > *first )
	{
		*second = *first;
		*first = candidate;
	}
	else if ( candidate > *second && candidate != *first )
		*second = candidate;
}

__global__ __launch_bounds__(G53_PICK_THREADS, 1)
void G53MarkovPickKernel(uint32_t step, uint32_t anchor, uint32_t position, unsigned long long seed, float inverse_temperature, float nucleus,
	uint32_t top_k, const float *__restrict__ row, const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ y,
	const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ confidence_weight, const uint16_t *__restrict__ confidence_bias,
	G53Picks *__restrict__ picks)
{
	__shared__ unsigned long long keys[G53_CANDIDATES];
	__shared__ unsigned long long best[G53_PICK_THREADS / G53_WARP];
	__shared__ unsigned long long second_best[G53_PICK_THREADS / G53_WARP];
	__shared__ float reduction[G53_PICK_THREADS / G53_WARP];
	__shared__ uint32_t chosen;
	__shared__ uint32_t alt_chosen;
	__shared__ float peak_scaled;
	const uint32_t previous = step == 0u ? anchor : picks->tokens[step - 1u];
	const uint32_t warp = threadIdx.x / G53_WARP, lane = threadIdx.x % G53_WARP;
	const float scale = inverse_temperature > 0.0f ? inverse_temperature : 1.0f;
	float dot = 0.0f, denom = 0.0f;
	if ( inverse_temperature > 0.0f )
	{
		for ( uint32_t index = threadIdx.x; index < G53_CANDIDATES; index += G53_PICK_THREADS )
			keys[index] = G53RowKey(row, index);
		__syncthreads();
		for ( uint32_t size = 2u; size <= G53_CANDIDATES; size <<= 1u )
			for ( uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u )
			{
				for ( uint32_t index = threadIdx.x; index < G53_CANDIDATES; index += G53_PICK_THREADS )
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
			const uint32_t slot = G53KeySlot(keys[lane]);
			const uint32_t token = candidates[step * G53_CANDIDATES + slot];
			const float scaled = row[slot] * inverse_temperature;
			float peak = G53WarpMax(scaled), probability = expf(scaled - peak), total = G53WarpSum(probability), running, keyed;
			uint32_t winner, runner;
			unsigned long long key;
			probability /= total;
			running = probability;
			for ( uint32_t offset = 1u; offset < G53_SAMPLED; offset <<= 1u )
			{
				const float earlier = __shfl_up_sync(0xffffffffu, running, offset);
				if ( lane >= offset )
					running += earlier;
			}
			keyed = (((nucleus <= 0.0f) || (running - probability < nucleus)) && (top_k == 0u || lane < top_k)) || lane == 0u
				? scaled + SparkMathGumbelNoise(seed, position + step, token) : -INFINITY;
			key = ((unsigned long long)SparkMathSampleOrdered(keyed) << 32u) | (unsigned long long)(0xffffffffu - lane);
			for ( uint32_t mask = G53_WARP / 2u; mask > 0u; mask >>= 1u )
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
		for ( uint32_t index = threadIdx.x; index < G53_CANDIDATES; index += G53_PICK_THREADS )
			G53Top2Insert(G53RowKey(row, index), &first, &second);
		for ( uint32_t mask = G53_WARP / 2u; mask > 0u; mask >>= 1u )
		{
			const unsigned long long other_first = __shfl_xor_sync(0xffffffffu, first, mask);
			const unsigned long long other_second = __shfl_xor_sync(0xffffffffu, second, mask);
			G53Top2Insert(other_first, &first, &second);
			G53Top2Insert(other_second, &first, &second);
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
			for ( uint32_t index = 1u; index < G53_PICK_THREADS / G53_WARP; index++ )
			{
				G53Top2Insert(best[index], &first, &second);
				G53Top2Insert(second_best[index], &first, &second);
			}
			chosen = G53KeySlot(first);
			alt_chosen = G53KeySlot(second);
			peak_scaled = row[chosen];
		}
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < G53_CANDIDATES; index += G53_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<G53_PICK_THREADS>(denom, reduction);
	for ( uint32_t index = threadIdx.x; index < G53_CONFIDENCE; index += G53_PICK_THREADS )
	{
		const float feature = index < G53_HIDDEN ? LmBf16ToFloat(y[(uint64_t)step * G53_HIDDEN + index])
			: LmBf16ToFloat(markov_previous[(uint64_t)previous * G53_MARKOV + index - G53_HIDDEN]);
		dot += LmBf16ToFloat(confidence_weight[index]) * feature;
	}
	dot = LmBlockSum<G53_PICK_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->tokens[step] = candidates[step * G53_CANDIDATES + chosen];
		picks->scores[step] = 1.0f / (1.0f + expf(-(dot + LmBf16ToFloat(confidence_bias[0]))));
		picks->alt_tokens[step] = candidates[step * G53_CANDIDATES + alt_chosen];
		picks->alt_scores[step] = denom > 0.0f ? expf(row[alt_chosen] * scale - peak_scaled) / denom : 0.0f;
	}
}

__global__ __launch_bounds__(G53_WARP, 1)
void G53BranchSelectKernel(G53Picks *__restrict__ picks, uint32_t drafts, float min_score)
{
	uint32_t used = 0u;
	if ( threadIdx.x != 0u )
		return;
	for ( uint32_t branch = 0u; branch < G53_BRANCHES; branch++ )
	{
		picks->branch_fork[branch] = G53_BLOCK;
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
			if ( pos < G53_BRANCHES )
			{
				picks->branch_fork[pos] = picks->branch_fork[pos - 1u];
				picks->branch_len[pos] = picks->branch_len[pos - 1u];
				picks->branch_tokens[pos][0] = picks->branch_tokens[pos - 1u][0];
				picks->branch_scores[pos][0] = picks->branch_scores[pos - 1u][0];
			}
			pos--;
		}
		if ( pos < G53_BRANCHES )
		{
			picks->branch_fork[pos] = step;
			picks->branch_len[pos] = 1u;
			picks->branch_tokens[pos][0] = picks->alt_tokens[step];
			picks->branch_scores[pos][0] = score;
			if ( used < G53_BRANCHES )
				used++;
		}
	}
}

__global__ __launch_bounds__(G53_THREADS, 1)
void G53BranchRowKernel(uint32_t offset, uint32_t drafts, const G53Picks *__restrict__ picks, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ rows)
{
	__shared__ float vector[G53_MARKOV];
	const uint32_t branch = blockIdx.y, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / G53_WARP, lane = threadIdx.x % G53_WARP;
	uint32_t previous;
	float *row;
	if ( fork >= drafts || step >= drafts )
		return;
	previous = picks->branch_tokens[branch][offset - 1u];
	row = rows + (uint64_t)branch * G53_CANDIDATES;
	for ( uint32_t index = threadIdx.x; index < G53_MARKOV; index += G53_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * G53_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * G53_WARPS + warp; slot < G53_CANDIDATES; slot += gridDim.x * G53_WARPS )
	{
		const uint32_t candidate = candidates[step * G53_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * G53_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < G53_MARKOV; element += G53_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = G53WarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * G53_VOCAB + candidate] + sum;
	}
}

__global__ __launch_bounds__(G53_PICK_THREADS, 1)
void G53BranchPickKernel(uint32_t offset, uint32_t drafts, float scale, const float *__restrict__ rows, const uint32_t *__restrict__ candidates,
	G53Picks *__restrict__ picks)
{
	__shared__ unsigned long long best[G53_PICK_THREADS / G53_WARP];
	__shared__ float reduction[G53_PICK_THREADS / G53_WARP];
	__shared__ uint32_t chosen;
	__shared__ float peak_scaled;
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / G53_WARP, lane = threadIdx.x % G53_WARP;
	const float *row = rows + (uint64_t)branch * G53_CANDIDATES;
	unsigned long long first = 0ull;
	float denom = 0.0f;
	if ( fork >= drafts || step >= drafts )
		return;
	for ( uint32_t index = threadIdx.x; index < G53_CANDIDATES; index += G53_PICK_THREADS )
	{
		const unsigned long long key = G53RowKey(row, index);
		if ( key > first )
			first = key;
	}
	for ( uint32_t mask = G53_WARP / 2u; mask > 0u; mask >>= 1u )
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
		for ( uint32_t index = 1u; index < G53_PICK_THREADS / G53_WARP; index++ )
			if ( best[index] > first )
				first = best[index];
		chosen = G53KeySlot(first);
		peak_scaled = row[chosen] * scale;
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < G53_CANDIDATES; index += G53_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<G53_PICK_THREADS>(denom, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->branch_tokens[branch][offset] = candidates[step * G53_CANDIDATES + chosen];
		picks->branch_scores[branch][offset] = denom > 0.0f ? 1.0f / denom : 0.0f;
		picks->branch_len[branch] = offset + 1u;
	}
}

static SparkStatus G53Refuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	SPARK_FAIL(status);
}

static SparkStatus G53LoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
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
	if ( magic != G53_PACK_MAGIC || version != G53_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
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
	*payload_base = (sizeof(header) + bytes + G53_PACK_ALIGNMENT - 1u) / G53_PACK_ALIGNMENT * G53_PACK_ALIGNMENT;
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return SPARK_STATUS_OK;
}

static SparkStatus G53Tensor(const G53State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t expected,
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
		fprintf(stderr, "glm53 dspark: pack tensor %s is missing or is not %llu bf16 elements\n", name, (unsigned long long)expected);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*pointer = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static int32_t G53Member(const SparkJsonDocument *document, int32_t object, const char *name)
{
	return object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
}

static SparkStatus G53Shape(G53State *state, const SparkJsonDocument *document)
{
	const int32_t config = G53Member(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = G53Member(document, config, "checkpoint_config");
	const int32_t transformer = G53Member(document, checkpoint, "transformer_layer_config");
	const int32_t rope = G53Member(document, transformer, "rope_parameters");
	const int32_t aux = G53Member(document, checkpoint, "aux_hidden_state_layer_ids");
	const struct
	{
		int32_t object;
		const char *name;
		uint64_t value;
	} expected[] = {
		{checkpoint, "block_size", G53_BLOCK},
		{checkpoint, "draft_vocab_size", G53_VOCAB},
		{checkpoint, "markov_rank", G53_MARKOV},
		{transformer, "hidden_size", G53_HIDDEN},
		{transformer, "intermediate_size", G53_MLP},
		{transformer, "num_hidden_layers", G53_LAYERS},
		{transformer, "num_attention_heads", G53_HEADS},
		{transformer, "num_key_value_heads", G53_KV_HEADS},
		{transformer, "head_dim", G53_HEAD_DIM},
		{transformer, "sliding_window", G53_SLIDING},
		{transformer, "vocab_size", G53_VOCAB},
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
	};
	uint64_t value = 0u;
	float theta = 0.0f;
	int32_t token;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		token = G53Member(document, expected[index].object, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "glm53 dspark: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	for ( uint32_t index = 0u; index < sizeof(flags) / sizeof(flags[0]); index++ )
	{
		bool flag = false;
		token = G53Member(document, flags[index].object, flags[index].name);
		if ( token < 0 || SparkJsonGetBoolean(document, token, &flag) != SPARK_STATUS_OK || flag != flags[index].value )
		{
			fprintf(stderr, "glm53 dspark: checkpoint_config %s does not match module contract\n", flags[index].name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	token = G53Member(document, checkpoint, "markov_head_type");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "vanilla") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( aux < 0 || SparkJsonGetArrayElementCount(document, aux) != G53_TAPS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = G53Member(document, checkpoint, "mask_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= G53_VOCAB )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->mask_token = (uint32_t)value;
	token = G53Member(document, transformer, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = G53Member(document, rope, "rope_theta");
	if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK || theta <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->softmax_scale = (float)(1.0 / sqrt((double)G53_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, G53_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	LM_LAUNCH((G53RopeTableKernel), 1u, G53_THREADS, 0, state->stream, state->rope_inverse, theta);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53Weights(G53State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = G53LoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = G53Shape(state, &document);
	{
		const struct
		{
			const char *name;
			uint64_t elements;
			const uint16_t **pointer;
		} plain[] = {
			{"fc.weight", (uint64_t)G53_HIDDEN * G53_TAP_WIDTH, &state->fc},
			{"hidden_norm.weight", G53_HIDDEN, &state->hidden_norm},
			{"norm.weight", G53_HIDDEN, &state->norm},
			{"target.embed", (uint64_t)G53_VOCAB * G53_HIDDEN, &state->embed},
			{"target.head", (uint64_t)G53_VOCAB * G53_HIDDEN, &state->head},
			{"markov_head.markov_w1.weight", (uint64_t)G53_VOCAB * G53_MARKOV, &state->markov_previous},
			{"markov_head.markov_w2.weight", (uint64_t)G53_VOCAB * G53_MARKOV, &state->markov_next},
			{"confidence_head.proj.weight", G53_CONFIDENCE, &state->confidence_weight},
			{"confidence_head.proj.bias", 1u, &state->confidence_bias},
		};
		for ( uint32_t index = 0u; index < sizeof(plain) / sizeof(plain[0]) && status == SPARK_STATUS_OK; index++ )
			status = G53Tensor(state, &document, payload_base, plain[index].name, plain[index].elements, plain[index].pointer);
	}
	for ( uint32_t layer = 0u; layer < G53_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		G53Layer *weights = &state->layers[layer];
		const struct
		{
			const char *suffix;
			uint64_t elements;
			const uint16_t **pointer;
		} tensors[] = {
			{"input_layernorm.weight", G53_HIDDEN, &weights->input_norm},
			{"post_attention_layernorm.weight", G53_HIDDEN, &weights->post_norm},
			{"self_attn.q_proj.weight", (uint64_t)G53_Q_WIDTH * G53_HIDDEN, &weights->q},
			{"self_attn.k_proj.weight", (uint64_t)G53_KV_WIDTH * G53_HIDDEN, &weights->k},
			{"self_attn.v_proj.weight", (uint64_t)G53_KV_WIDTH * G53_HIDDEN, &weights->v},
			{"self_attn.o_proj.weight", (uint64_t)G53_HIDDEN * G53_Q_WIDTH, &weights->o},
			{"self_attn.q_norm.weight", G53_HEAD_DIM, &weights->q_norm},
			{"self_attn.k_norm.weight", G53_HEAD_DIM, &weights->k_norm},
			{"mlp.gate_proj.weight", (uint64_t)G53_MLP * G53_HIDDEN, &weights->gate},
			{"mlp.up_proj.weight", (uint64_t)G53_MLP * G53_HIDDEN, &weights->up},
			{"mlp.down_proj.weight", (uint64_t)G53_HIDDEN * G53_MLP, &weights->down},
		};
		for ( uint32_t index = 0u; index < sizeof(tensors) / sizeof(tensors[0]) && status == SPARK_STATUS_OK; index++ )
		{
			char name[128];
			snprintf(name, sizeof(name), "layers.%u.%s", layer, tensors[index].suffix);
			status = G53Tensor(state, &document, payload_base, name, tensors[index].elements, tensors[index].pointer);
		}
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define G53_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus G53Scratch(G53State *state, uint64_t context_bytes)
{
	G53Forward *forward = &state->forward;
	int multiprocessors = 0;
	uint64_t capacity = context_bytes / G53_SEQUENCE_BYTES;
	capacity = capacity < 1u ? 1u : capacity > G53_SEQUENCES_MAX ? G53_SEQUENCES_MAX : capacity;
	state->capacity = (uint32_t)capacity;
	state->sequences = (G53Sequence *)calloc(capacity, sizeof(G53Sequence));
	if ( state->sequences == 0 || SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)G53_INGEST_ROWS * G53_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)G53_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(G53Draft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(G53Picks)) != SPARK_GPU_SUCCESS ||
		!G53_DEVICE(state->caches, capacity * G53_SEQUENCE_ELEMENTS) || !G53_DEVICE(state->taps, (uint64_t)G53_INGEST_ROWS * G53_TAP_WIDTH) ||
		!G53_DEVICE(state->positions, G53_INGEST_ROWS) || !G53_DEVICE(state->features_raw, (uint64_t)G53_INGEST_ROWS * G53_HIDDEN) ||
		!G53_DEVICE(state->features, (uint64_t)G53_INGEST_ROWS * G53_HIDDEN) || !G53_DEVICE(state->keys_raw, (uint64_t)G53_INGEST_ROWS * G53_KV_WIDTH) ||
		!G53_DEVICE(state->values_raw, (uint64_t)G53_INGEST_ROWS * G53_KV_WIDTH) || !G53_DEVICE(forward->draft, 1u) || !G53_DEVICE(forward->picks, 1u) ||
		!G53_DEVICE(forward->hidden, G53_BLOCK * G53_HIDDEN) || !G53_DEVICE(forward->x, G53_BLOCK * G53_HIDDEN) ||
		!G53_DEVICE(forward->q, G53_BLOCK * G53_Q_WIDTH) || !G53_DEVICE(forward->q_roped, G53_BLOCK * G53_Q_WIDTH) ||
		!G53_DEVICE(forward->k, G53_BLOCK * G53_KV_WIDTH) || !G53_DEVICE(forward->v, G53_BLOCK * G53_KV_WIDTH) ||
		!G53_DEVICE(forward->partials, (uint64_t)G53_HEADS * G53_SPLITS * G53_BLOCK * (G53_HEAD_DIM + 2u)) ||
		!G53_DEVICE(forward->attended, G53_BLOCK * G53_Q_WIDTH) || !G53_DEVICE(forward->projected, G53_BLOCK * G53_HIDDEN) ||
		!G53_DEVICE(forward->residual, G53_BLOCK * G53_HIDDEN) || !G53_DEVICE(forward->gate, G53_BLOCK * G53_MLP) ||
		!G53_DEVICE(forward->up, G53_BLOCK * G53_MLP) || !G53_DEVICE(forward->act, G53_BLOCK * G53_MLP) ||
		!G53_DEVICE(forward->mlp, G53_BLOCK * G53_HIDDEN) || !G53_DEVICE(forward->y, G53_BLOCK * G53_HIDDEN) ||
		!G53_DEVICE(forward->logits, (uint64_t)G53_BLOCK * G53_VOCAB) || !G53_DEVICE(forward->logits_f32, (uint64_t)G53_BLOCK * G53_VOCAB) ||
		!G53_DEVICE(forward->candidates, G53_BLOCK * G53_CANDIDATES) || !G53_DEVICE(forward->markov_row, G53_BRANCHES * G53_CANDIDATES) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		state->sequences[index].cache = state->caches + (uint64_t)index * G53_SEQUENCE_ELEMENTS;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static uint16_t *G53Keys(const G53Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer) * G53_LAYER_KV;
}

static uint16_t *G53Values(const G53Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer + 1u) * G53_LAYER_KV;
}

static void G53Reset(G53Sequence *sequence)
{
	for ( uint32_t slot = 0u; slot < G53_WINDOW; slot++ )
	{
		sequence->positions[slot] = -1;
		sequence->tokens[slot] = 0u;
	}
	sequence->retired = 0u;
}

static G53Sequence *G53Live(G53State *state, uint64_t id)
{
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		if ( state->sequences[index].live != 0u && state->sequences[index].retired == 0u && state->sequences[index].id == id )
			return &state->sequences[index];
	return 0;
}

static G53Sequence *G53Claim(G53State *state)
{
	G53Sequence *oldest = 0;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
	{
		G53Sequence *sequence = &state->sequences[index];
		if ( sequence->live == 0u )
			return sequence;
		if ( oldest == 0 || sequence->last_used < oldest->last_used )
			oldest = sequence;
	}
	return oldest;
}

static uint32_t G53Held(const G53Sequence *sequence, uint64_t position)
{
	return sequence->positions[position % G53_WINDOW] == (int64_t)position ? 1u : 0u;
}

static SparkStatus G53Gemm(G53State *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53IngestRows(G53State *state, G53Sequence *sequence, uint32_t rows)
{
	SparkStatus status = G53Gemm(state, state->fc, state->taps, state->features_raw, rows, G53_TAP_WIDTH, G53_HIDDEN);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((G53RmsKernel), rows, G53_THREADS, 0, state->stream, state->features_raw, state->hidden_norm, state->features, G53_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < G53_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = G53Gemm(state, state->layers[layer].k, state->features, state->keys_raw, rows, G53_HIDDEN, G53_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = G53Gemm(state, state->layers[layer].v, state->features, state->values_raw, rows, G53_HIDDEN, G53_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53HeadNormRopeKernel), rows * G53_KV_HEADS, G53_HEAD_DIM, 0, state->stream, state->keys_raw, state->layers[layer].k_norm,
			state->rope_inverse, state->positions, G53_KV_HEADS, state->epsilon, (uint16_t *)0, G53Keys(sequence, layer), 0u, 1u);
		LM_LAUNCH((G53StoreValuesKernel), rows, G53_THREADS, 0, state->stream, state->values_raw, state->positions, 0u, G53Values(sequence, layer));
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static SparkStatus G53Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	G53State *state = (G53State *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	const uint32_t skip = rows > G53_WINDOW ? rows - G53_WINDOW : 0u;
	const uint64_t first = frame->header.first_position + skip;
	G53Sequence *sequence;
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != G53_TAPS || hidden != G53_HIDDEN || frame->rows == 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t tap = 0u; tap < G53_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * G53_HIDDEN > frame->row_elements )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( frame->header.first_position + rows > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = G53Live(state, frame->header.sequence_id);
	if ( sequence != 0 && frame->header.first_position != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence->last_used = ++state->clock;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = G53Claim(state);
		G53Reset(sequence);
		sequence->live = 1u;
		sequence->id = frame->header.sequence_id;
	}
	else if ( frame->header.first_position == 0u )
		G53Reset(sequence);
	sequence->control_generation = frame->header.control_generation;
	sequence->request_id = frame->header.request_id;
	sequence->last_used = ++state->clock;
	for ( uint32_t row = skip; row < rows; row++ )
		sequence->positions[(frame->header.first_position + row) % G53_WINDOW] = -1;
	for ( uint32_t done = skip; done < rows && status == SPARK_STATUS_OK; done += G53_INGEST_ROWS )
	{
		const uint32_t part = rows - done < G53_INGEST_ROWS ? rows - done : G53_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < G53_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * G53_TAP_WIDTH + (uint64_t)tap * G53_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * G53_HIDDEN, G53_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(frame->header.first_position + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * G53_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t),
			SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		{
			(void)SparkGpuStreamSynchronize(state->stream);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		status = G53IngestRows(state, sequence, part);
	}
	if ( status == SPARK_STATUS_OK )
	{
		for ( uint32_t slot = 0u; slot < G53_WINDOW; slot++ )
			if ( sequence->positions[slot] >= (int64_t)first )
				sequence->positions[slot] = -1;
		for ( uint32_t row = skip; row < rows; row++ )
		{
			const uint64_t position = frame->header.first_position + row;
			sequence->positions[position % G53_WINDOW] = (int64_t)position;
			sequence->tokens[position % G53_WINDOW] = frame->tokens != 0 ? frame->tokens[row] : 0u;
		}
	}
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus G53CopySlots(G53State *state, G53Sequence *sequence, const G53Sequence *donor, uint32_t slot, uint32_t count)
{
	for ( uint32_t layer = 0u; layer < G53_LAYERS; layer++ )
		for ( uint32_t head = 0u; head < G53_KV_HEADS; head++ )
		{
			const uint64_t at = ((uint64_t)head * G53_COLUMNS + slot) * G53_HEAD_DIM;
			const size_t bytes = (size_t)count * G53_HEAD_DIM * sizeof(uint16_t);
			if ( SparkGpuMemcpyAsync(G53Keys(sequence, layer) + at, G53Keys(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) !=
				SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(G53Values(sequence, layer) + at, G53Values(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	return SPARK_STATUS_OK;
}

static uint32_t G53Usable(const G53Sequence *sequence, const G53Sequence *donor, const SparkDraftServerRequest *request, uint64_t position)
{
	return G53Held(sequence, position) == 0u && G53Held(donor, position) != 0u && position < request->committed_token_count &&
		donor->tokens[position % G53_WINDOW] == request->committed_token_ids[position] ? 1u : 0u;
}

static SparkStatus G53FillFromDonors(G53State *state, G53Sequence *sequence, const SparkDraftServerRequest *request, uint64_t first)
{
	uint8_t tried[G53_SEQUENCES_MAX];
	memset(tried, 0, sizeof(tried));
	for ( ;; )
	{
		G53Sequence *donor = 0;
		uint32_t chosen = 0u, missing = 0u;
		uint64_t position = first;
		for ( uint64_t at = first; at < request->position; at++ )
			missing += G53Held(sequence, at) == 0u ? 1u : 0u;
		if ( missing == 0u )
			return SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < state->capacity; index++ )
		{
			G53Sequence *other = &state->sequences[index];
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
			if ( G53Usable(sequence, donor, request, position) == 0u )
			{
				position++;
				continue;
			}
			while ( stop < request->position && G53Usable(sequence, donor, request, stop) != 0u && (stop == position || stop % G53_WINDOW != 0u) )
				stop++;
			if ( G53CopySlots(state, sequence, donor, (uint32_t)(position % G53_WINDOW), (uint32_t)(stop - position)) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			for ( uint64_t at = position; at < stop; at++ )
			{
				sequence->positions[at % G53_WINDOW] = (int64_t)at;
				sequence->tokens[at % G53_WINDOW] = donor->tokens[at % G53_WINDOW];
			}
			position = stop;
		}
	}
}

static void G53Span(G53Sequence *sequence, uint64_t checked, uint64_t position, G53Draft *draft)
{
	const uint64_t count = position - checked;
	const uint32_t first = count != 0u ? (uint32_t)(checked % G53_WINDOW) : 0u, end = first + (uint32_t)count;
	uint32_t newer = 0u;
	memset(draft->valid, 0, sizeof(draft->valid));
	draft->masked = 0u;
	if ( count == G53_WINDOW )
	{
		draft->start = 0u;
		draft->block_at = G53_WINDOW;
		return;
	}
	for ( uint32_t slot = end; slot < G53_WINDOW && slot < end + G53_BLOCK; slot++ )
		newer |= count != 0u && sequence->positions[slot] > (int64_t)(position - 1u) ? 1u : 0u;
	if ( end <= G53_WINDOW && newer == 0u )
	{
		for ( uint32_t slot = end; slot < G53_WINDOW && slot < end + G53_BLOCK; slot++ )
			sequence->positions[slot] = -1;
		draft->start = first;
		draft->block_at = end;
		return;
	}
	for ( uint64_t at = checked; at < position; at++ )
		draft->valid[at % G53_WINDOW] = 1u;
	draft->masked = 1u;
	draft->start = 0u;
	draft->block_at = G53_WINDOW;
}

static SparkStatus G53Prepare(G53State *state, G53Sequence *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
	const SparkRowSampling *sampling)
{
	G53Draft *draft = state->forward.host_draft;
	if ( anchor >= G53_VOCAB || position + G53_BLOCK > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft, 0, offsetof(G53Draft, valid));
	for ( uint32_t row = 0u; row < G53_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	draft->anchor = anchor;
	draft->mask = state->mask_token;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	G53Span(sequence, checked, position, draft);
	if ( SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53RunForward(G53State *state, const G53Sequence *sequence, uint32_t layers)
{
	G53Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(G53Draft, positions));
	const uint32_t block_at = state->forward.host_draft->block_at;
	const uint32_t span = block_at - state->forward.host_draft->start;
	const uint32_t splits = (span + G53_SPLIT_KEYS - 1u) / G53_SPLIT_KEYS;
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((G53EmbedKernel), G53_BLOCK, G53_THREADS, 0, stream, state->embed, forward->draft, forward->hidden);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const G53Layer *weights = &state->layers[layer];
		LM_LAUNCH((G53RmsKernel), G53_BLOCK, G53_THREADS, 0, stream, forward->hidden, weights->input_norm, forward->x, G53_HIDDEN, state->epsilon);
		status = G53Gemm(state, weights->q, forward->x, forward->q, G53_BLOCK, G53_HIDDEN, G53_Q_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = G53Gemm(state, weights->k, forward->x, forward->k, G53_BLOCK, G53_HIDDEN, G53_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = G53Gemm(state, weights->v, forward->x, forward->v, G53_BLOCK, G53_HIDDEN, G53_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53HeadNormRopeKernel), G53_BLOCK * G53_HEADS, G53_HEAD_DIM, 0, stream, forward->q, weights->q_norm, state->rope_inverse, positions,
			G53_HEADS, state->epsilon, forward->q_roped, (uint16_t *)0, 0u, 0u);
		LM_LAUNCH((G53HeadNormRopeKernel), G53_BLOCK * G53_KV_HEADS, G53_HEAD_DIM, 0, stream, forward->k, weights->k_norm, state->rope_inverse, positions,
			G53_KV_HEADS, state->epsilon, (uint16_t *)0, G53Keys(sequence, layer), block_at, 0u);
		LM_LAUNCH((G53StoreValuesKernel), G53_BLOCK, G53_THREADS, 0, stream, forward->v, (const uint32_t *)0, block_at, G53Values(sequence, layer));
		if ( splits != 0u )
			LM_LAUNCH((G53AttentionSplitKernel), dim3(G53_HEADS, splits), G53_ATTN_THREADS, 0, stream, forward->q_roped, G53Keys(sequence, layer),
				G53Values(sequence, layer), forward->draft, state->softmax_scale, forward->partials);
		LM_LAUNCH((G53AttentionMergeKernel), G53_BLOCK * G53_HEADS, G53_HEAD_DIM, 0, stream, forward->q_roped, G53Keys(sequence, layer),
			G53Values(sequence, layer), forward->draft, state->softmax_scale, forward->partials, splits, forward->attended);
		status = G53Gemm(state, weights->o, forward->attended, forward->projected, G53_BLOCK, G53_Q_WIDTH, G53_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53ResidualKernel), 2u * state->multiprocessors, G53_THREADS, 0, stream, forward->hidden, forward->projected, forward->residual,
			G53_BLOCK * G53_HIDDEN);
		LM_LAUNCH((G53RmsKernel), G53_BLOCK, G53_THREADS, 0, stream, forward->residual, weights->post_norm, forward->x, G53_HIDDEN, state->epsilon);
		status = G53Gemm(state, weights->gate, forward->x, forward->gate, G53_BLOCK, G53_HIDDEN, G53_MLP);
		if ( status == SPARK_STATUS_OK )
			status = G53Gemm(state, weights->up, forward->x, forward->up, G53_BLOCK, G53_HIDDEN, G53_MLP);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53SwigluKernel), 2u * state->multiprocessors, G53_THREADS, 0, stream, forward->gate, forward->up, forward->act, G53_BLOCK * G53_MLP);
		status = G53Gemm(state, weights->down, forward->act, forward->mlp, G53_BLOCK, G53_MLP, G53_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53ResidualKernel), 2u * state->multiprocessors, G53_THREADS, 0, stream, forward->residual, forward->mlp, forward->hidden,
			G53_BLOCK * G53_HIDDEN);
	}
	if ( status != SPARK_STATUS_OK || layers < G53_LAYERS )
		return status;
	LM_LAUNCH((G53RmsKernel), G53_BLOCK, G53_THREADS, 0, stream, forward->hidden, state->norm, forward->y, G53_HIDDEN, state->epsilon);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53Heads(G53State *state, uint32_t anchor_token, uint64_t position, const SparkRowSampling *sampling, uint32_t drafts, uint32_t tree)
{
	G53Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const float nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	SparkStatus status = G53Gemm(state, state->head, forward->y, forward->logits, G53_BLOCK, G53_HIDDEN, G53_VOCAB);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((G53WidenKernel), 4u * state->multiprocessors, G53_THREADS, 0, stream, forward->logits, forward->logits_f32, G53_BLOCK * G53_VOCAB);
	if ( LmTopkExactRowsLaunch<G53_TOPK_THREADS>(forward->logits_f32, G53_BLOCK, G53_VOCAB, G53_CANDIDATES, forward->candidates, stream) !=
		SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t step = 0u; step < G53_BLOCK; step++ )
	{
		LM_LAUNCH((G53MarkovRowKernel), G53_CANDIDATES / G53_WARPS, G53_THREADS, 0, stream, step, anchor_token, forward->picks->tokens,
			forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
		LM_LAUNCH((G53MarkovPickKernel), 1u, G53_PICK_THREADS, 0, stream, step, anchor_token, (uint32_t)position, (unsigned long long)sampling->seed,
			sampling->inverse_temperature, nucleus, sampling->top_k, forward->markov_row, forward->candidates, forward->y, state->markov_previous,
			state->confidence_weight, state->confidence_bias, forward->picks);
	}
	if ( tree != 0u && drafts >= 2u )
	{
		const float scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
		LM_LAUNCH((G53BranchSelectKernel), 1u, G53_WARP, 0, stream, forward->picks, drafts, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE);
		for ( uint32_t offset = 1u; offset < G53_BRANCH_LEN && offset < drafts; offset++ )
		{
			LM_LAUNCH((G53BranchRowKernel), dim3(G53_CANDIDATES / G53_WARPS, G53_BRANCHES), G53_THREADS, 0, stream, offset, drafts,
				forward->picks, forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
			LM_LAUNCH((G53BranchPickKernel), G53_BRANCHES, G53_PICK_THREADS, 0, stream, offset, drafts, scale, forward->markov_row, forward->candidates,
				forward->picks);
		}
	}
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(G53Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53Settle(G53State *state, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static void G53ForwardFree(G53Forward *forward)
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

static SparkStatus G53Close(void *opaque)
{
	G53State *state = (G53State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	G53ForwardFree(&state->forward);
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

static SparkStatus G53Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	G53State *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return G53Refuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	{
		const SparkStatus device = SparkDeviceInitialize(0);
		if ( (device != SPARK_STATUS_OK && device != SPARK_STATUS_DUPLICATE) || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
			return G53Refuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
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
	if ( lane->context_bytes < G53_SEQUENCE_BYTES )
		return G53Refuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one sequence's window", lane->lane_name);
	state = (G53State *)calloc(1u, sizeof(*state));
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
		status = G53Weights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = G53Scratch(state, lane->context_bytes);
	if ( status == SPARK_STATUS_OK )
		status = G53Settle(state, SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a GLM-5.3 DSpark pack this module reads", lane->lane_name, lane->pack_path);
		(void)G53Close(state);
		SPARK_FAIL(status);
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s with %u sequence windows", lane->lane_name, lane->pack_path, state->capacity);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus G53Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	G53State *state = (G53State *)opaque;
	G53Sequence *sequence = G53Live(state, request->sequence_id);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > G53_WINDOW ? request->position - G53_WINDOW : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( sequence == 0 || (request->position != 0u && G53Held(sequence, request->position - 1u) == 0u) )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	status = G53FillFromDonors(state, sequence, request, first);
	if ( status != SPARK_STATUS_OK )
		return G53Settle(state, status);
	for ( uint64_t position = first; position < request->position; position++ )
		if ( G53Held(sequence, position) == 0u )
			checked = position + 1u;
	if ( checked != first )
		state->gaps++;
	if ( request->committed_token_count < request->position )
		return G53Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	for ( uint64_t position = checked; position < request->position; position++ )
		if ( sequence->tokens[position % G53_WINDOW] != request->committed_token_ids[position] )
			return G53Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	drafts = drafts < G53_BLOCK ? drafts : G53_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		SparkDraftTreeBranch branches[G53_BLOCK];
		uint32_t branch_count = 0u;
		status = G53Prepare(state, sequence, request->anchor_token_id, request->position, checked, &request->sampling);
		if ( status == SPARK_STATUS_OK )
			status = G53RunForward(state, sequence, G53_LAYERS);
		if ( status == SPARK_STATUS_OK )
			status = G53Heads(state, request->anchor_token_id, request->position, &request->sampling, drafts, tree);
		status = G53Settle(state, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				uint32_t matched = G53_BRANCHES;
				for ( uint32_t b = 0u; b < G53_BRANCHES; b++ )
					if ( state->forward.host_picks->branch_len[b] > 0u && state->forward.host_picks->branch_fork[b] == step )
						matched = b;
				branches[branch_count].fork_step = step;
				if ( matched < G53_BRANCHES )
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

static const SparkDrafterModuleInterface g53_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"glm53-dspark",
	G53Open,
	G53Ingest,
	G53Propose,
	G53Close,
};

static SparkStatus G53Readback(void *host, const void *device, size_t bytes)
{
	if ( SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &g53_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	G53State *state = (G53State *)opaque;
	const G53Sequence *sequence = state != 0 ? G53Live(state, sequence_id) : 0;
	SparkStatus status;
	if ( sequence == 0 || layer >= G53_LAYERS || keys == 0 || values == 0 || positions == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memcpy(positions, sequence->positions, sizeof(sequence->positions));
	status = G53Readback(keys, G53Keys(sequence, layer), G53_LAYER_KV * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = G53Readback(values, G53Values(sequence, layer), G53_LAYER_KV * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	G53State *state = (G53State *)opaque;
	G53Sequence *sequence = state != 0 ? G53Live(state, sequence_id) : 0;
	const uint64_t started = SparkMonotonicNs(), first = position > G53_WINDOW ? position - G53_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	for ( uint64_t at = first; at < position; at++ )
		if ( G53Held(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = G53Prepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = G53RunForward(state, sequence, G53_LAYERS);
	status = G53Settle(state, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = G53Readback(y, state->forward.y, (size_t)G53_BLOCK * G53_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	G53State *state = (G53State *)opaque;
	G53Sequence *sequence = state != 0 ? G53Live(state, sequence_id) : 0;
	const uint64_t first = position > G53_WINDOW ? position - G53_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 || hidden == 0 || layers == 0u || layers > G53_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint64_t at = first; at < position; at++ )
		if ( G53Held(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	status = G53Prepare(state, sequence, anchor, position, checked, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = G53RunForward(state, sequence, layers);
	status = G53Settle(state, status);
	if ( status == SPARK_STATUS_OK )
		status = G53Readback(hidden, state->forward.hidden, (size_t)G53_BLOCK * G53_HIDDEN * sizeof(uint16_t));
	return status;
}
