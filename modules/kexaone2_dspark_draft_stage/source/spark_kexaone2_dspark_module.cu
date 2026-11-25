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

#define KX2_HIDDEN 6144u
#define KX2_TAPS 5u
#define KX2_TAP_WIDTH (KX2_TAPS * KX2_HIDDEN)
#define KX2_LAYERS 5u
#define KX2_TARGET_LAYERS 78u
#define KX2_HEADS 64u
#define KX2_KV_HEADS 8u
#define KX2_GQA_GROUP (KX2_HEADS / KX2_KV_HEADS)
#define KX2_HEAD_DIM 128u
#define KX2_HALF (KX2_HEAD_DIM / 2u)
#define KX2_Q_WIDTH (KX2_HEADS * KX2_HEAD_DIM)
#define KX2_KV_WIDTH (KX2_KV_HEADS * KX2_HEAD_DIM)
#define KX2_MLP 18432u
#define KX2_BLOCK 7u
#define KX2_VOCAB 153600u
#define KX2_MARKOV 256u
#define KX2_CONFIDENCE (KX2_HIDDEN + KX2_MARKOV)
#define KX2_CANDIDATES 2048u
#define KX2_SAMPLED 32u
#define KX2_BRANCHES 2u
#define KX2_BRANCH_LEN 4u
#define KX2_INGEST_ROWS 256u
#define KX2_THREADS 256u
#define KX2_WARP 32u
#define KX2_WARPS (KX2_THREADS / KX2_WARP)
#define KX2_PICK_THREADS 1024u
#define KX2_TOPK_THREADS 512u
#define KX2_ATTN_TILE 32u
#define KX2_ATTN_THREADS (KX2_GQA_GROUP * KX2_WARP)
#define KX2_SPLITS_MAX 16u
#define KX2_SPLIT_ROWS_MIN 256u
#define KX2_RUNS_MAX 256u
#define KX2_PAGE_POSITIONS 1024u
#define KX2_PAGES_MAX 1024u
#define KX2_SEQUENCES_MAX 64u
#define KX2_PAGE_HALF_LAYER ((uint64_t)KX2_KV_HEADS * KX2_PAGE_POSITIONS * KX2_HEAD_DIM)
#define KX2_PAGE_LAYER (2u * KX2_PAGE_HALF_LAYER)
#define KX2_PAGE_ELEMENTS ((uint64_t)KX2_LAYERS * KX2_PAGE_LAYER)
#define KX2_PAGE_BYTES (KX2_PAGE_ELEMENTS * sizeof(uint16_t))
#define KX2_PACK_MAGIC 0x4B505244u
#define KX2_PACK_VERSION 1u
#define KX2_PACK_ALIGNMENT 128u
#define KX2_INVERSE_PI 0.31830988618379067154

typedef struct KX2Layer
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
} KX2Layer;

typedef struct KX2Sequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t held_end;
	uint64_t verified;
	uint64_t last_used;
	uint32_t retired;
	uint32_t page_count;
	uint16_t **pages;
	uint8_t *held;
	uint32_t *tokens;
} KX2Sequence;

typedef struct KX2Draft
{
	uint32_t positions[KX2_BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t runs;
	uint32_t run_first[KX2_RUNS_MAX];
	uint32_t run_position[KX2_RUNS_MAX];
} KX2Draft;

typedef struct KX2Picks
{
	uint32_t tokens[KX2_BLOCK];
	float scores[KX2_BLOCK];
	uint32_t alt_tokens[KX2_BLOCK];
	float alt_scores[KX2_BLOCK];
	uint32_t branch_fork[KX2_BRANCHES];
	uint32_t branch_len[KX2_BRANCHES];
	uint32_t branch_tokens[KX2_BRANCHES][KX2_BRANCH_LEN];
	float branch_scores[KX2_BRANCHES][KX2_BRANCH_LEN];
} KX2Picks;

typedef struct KX2Forward
{
	KX2Draft *host_draft;
	KX2Draft *draft;
	uint16_t **host_pages;
	uint16_t **pages;
	KX2Picks *host_picks;
	KX2Picks *picks;
	uint16_t *hidden;
	uint16_t *x;
	uint16_t *q;
	uint16_t *q_roped;
	uint16_t *k;
	uint16_t *k_roped;
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
} KX2Forward;

typedef struct KX2State
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
	KX2Layer layers[KX2_LAYERS];
	KX2Forward forward;
	uint16_t *host_taps;
	uint32_t *host_positions;
	uint16_t *taps;
	uint32_t *positions;
	uint16_t *features_raw;
	uint16_t *features;
	uint16_t *keys_raw;
	uint16_t *values_raw;
	uint64_t context_bytes;
	uint64_t pages_used;
	uint64_t clock;
	KX2Sequence sequences[KX2_SEQUENCES_MAX];
	uint64_t frames;
	uint64_t rows;
} KX2State;

static __device__ __forceinline__ void KX2Rotation(uint32_t position, float inverse, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * KX2_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	sincospif(reduced, sine, cosine);
}

static __device__ __forceinline__ float KX2WarpSum(float value)
{
	for ( uint32_t mask = KX2_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ float KX2WarpMax(float value)
{
	for ( uint32_t mask = KX2_WARP / 2u; mask > 0u; mask >>= 1u )
		value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
	return value;
}

static __device__ __forceinline__ const uint16_t *KX2KeyRow(uint16_t *const *__restrict__ pages, uint64_t layer_head_offset,
	const uint32_t *__restrict__ run_first, const uint32_t *__restrict__ run_position, uint32_t runs, uint32_t index, uint32_t end)
{
	uint32_t low = 0u, high = runs, position;
	if ( index >= end )
		return 0;
	while ( high - low > 1u )
	{
		const uint32_t middle = (low + high) >> 1u;
		if ( run_first[middle] <= index )
			low = middle;
		else
			high = middle;
	}
	position = run_position[low] + (index - run_first[low]);
	return pages[position / KX2_PAGE_POSITIONS] + layer_head_offset + (uint64_t)(position % KX2_PAGE_POSITIONS) * KX2_HEAD_DIM;
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2RopeTableKernel(float *__restrict__ inverse, float theta)
{
	if ( threadIdx.x < KX2_HALF )
		inverse[threadIdx.x] = 1.0f / powf(theta, (float)(2u * threadIdx.x) / (float)KX2_HEAD_DIM);
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2RmsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[KX2_WARPS];
	const uint64_t row = blockIdx.x;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += KX2_THREADS )
	{
		const float value = LmBf16ToFloat(input[row * dimension + index]);
		total += value * value;
	}
	total = LmBlockSum<KX2_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += KX2_THREADS )
		output[row * dimension + index] = LmFloatToBf16(LmBf16ToFloat(input[row * dimension + index]) * scale * LmBf16ToFloat(weight[index]));
}

__global__ __launch_bounds__(KX2_HEAD_DIM, 1)
void KX2BlockHeadNormRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, uint16_t *__restrict__ output)
{
	__shared__ float reduction[KX2_HEAD_DIM / KX2_WARP];
	__shared__ float normed[KX2_HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = LmBf16ToFloat(input[((uint64_t)row * heads + head) * KX2_HEAD_DIM + lane]);
	float scale, cosine, sine, rotated;
	scale = rsqrtf(LmBlockSum<KX2_HEAD_DIM>(value * value, reduction) / (float)KX2_HEAD_DIM + epsilon);
	normed[lane] = LmBf16ToFloat(LmFloatToBf16(value * scale * LmBf16ToFloat(weight[lane])));
	__syncthreads();
	KX2Rotation(position, inverse[lane % KX2_HALF], &cosine, &sine);
	rotated = lane < KX2_HALF ? normed[lane] * cosine - normed[lane + KX2_HALF] * sine : normed[lane] * cosine + normed[lane - KX2_HALF] * sine;
	output[((uint64_t)row * heads + head) * KX2_HEAD_DIM + lane] = LmFloatToBf16(rotated);
}

__global__ __launch_bounds__(KX2_HEAD_DIM, 1)
void KX2IngestKeyRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, float epsilon, uint16_t *const *__restrict__ pages, uint32_t layer)
{
	__shared__ float reduction[KX2_HEAD_DIM / KX2_WARP];
	__shared__ float normed[KX2_HEAD_DIM];
	const uint32_t row = blockIdx.x / KX2_KV_HEADS, head = blockIdx.x % KX2_KV_HEADS, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = LmBf16ToFloat(input[((uint64_t)row * KX2_KV_HEADS + head) * KX2_HEAD_DIM + lane]);
	float scale, cosine, sine, rotated;
	uint16_t *page = pages[position / KX2_PAGE_POSITIONS];
	scale = rsqrtf(LmBlockSum<KX2_HEAD_DIM>(value * value, reduction) / (float)KX2_HEAD_DIM + epsilon);
	normed[lane] = LmBf16ToFloat(LmFloatToBf16(value * scale * LmBf16ToFloat(weight[lane])));
	__syncthreads();
	KX2Rotation(position, inverse[lane % KX2_HALF], &cosine, &sine);
	rotated = lane < KX2_HALF ? normed[lane] * cosine - normed[lane + KX2_HALF] * sine : normed[lane] * cosine + normed[lane - KX2_HALF] * sine;
	page[(uint64_t)layer * KX2_PAGE_LAYER + ((uint64_t)head * KX2_PAGE_POSITIONS + (position % KX2_PAGE_POSITIONS)) * KX2_HEAD_DIM + lane] =
		LmFloatToBf16(rotated);
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2IngestStoreValuesKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions,
	uint16_t *const *__restrict__ pages, uint32_t layer)
{
	const uint32_t row = blockIdx.x, position = positions[row];
	uint16_t *dst = pages[position / KX2_PAGE_POSITIONS] + (uint64_t)layer * KX2_PAGE_LAYER + KX2_PAGE_HALF_LAYER;
	const uint32_t slot = position % KX2_PAGE_POSITIONS;
	for ( uint32_t index = threadIdx.x; index < KX2_KV_WIDTH; index += KX2_THREADS )
		dst[((uint64_t)(index / KX2_HEAD_DIM) * KX2_PAGE_POSITIONS + slot) * KX2_HEAD_DIM + index % KX2_HEAD_DIM] =
			values[(uint64_t)row * KX2_KV_WIDTH + index];
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2EmbedKernel(const uint16_t *__restrict__ embed, const KX2Draft *__restrict__ draft, uint16_t *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	for ( uint32_t index = threadIdx.x; index < KX2_HIDDEN; index += KX2_THREADS )
		hidden[(uint64_t)row * KX2_HIDDEN + index] = embed[(uint64_t)token * KX2_HIDDEN + index];
}

__global__ __launch_bounds__(KX2_ATTN_THREADS, 1)
void KX2ContextAttentionKernel(const uint16_t *__restrict__ q, uint16_t *const *__restrict__ pages, uint32_t layer,
	const KX2Draft *__restrict__ draft, uint32_t held, uint32_t split_rows, float scale, float *__restrict__ partials)
{
	__shared__ const uint16_t *s_ptrs[KX2_ATTN_TILE];
	__shared__ uint32_t s_k[KX2_ATTN_TILE][KX2_HALF + 1u];
	__shared__ uint32_t s_v[KX2_ATTN_TILE][KX2_HALF + 1u];
	__shared__ float s_scores[KX2_GQA_GROUP][KX2_ATTN_TILE];
	const uint32_t kv_head = blockIdx.x / KX2_BLOCK, row = blockIdx.x % KX2_BLOCK, split = blockIdx.y;
	const uint32_t warp = threadIdx.x / KX2_WARP, lane = threadIdx.x % KX2_WARP;
	const uint32_t q_head = kv_head * KX2_GQA_GROUP + warp;
	const uint64_t layer_head_offset = (uint64_t)layer * KX2_PAGE_LAYER + (uint64_t)kv_head * KX2_PAGE_POSITIONS * KX2_HEAD_DIM;
	const uint32_t begin = split * split_rows;
	const uint32_t end = begin + split_rows < held ? begin + split_rows : held;
	const uint32_t tiles = begin < end ? (end - begin + KX2_ATTN_TILE - 1u) / KX2_ATTN_TILE : 0u;
	const uint2 q_quad = *(const uint2 *)(q + ((uint64_t)row * KX2_HEADS + q_head) * KX2_HEAD_DIM + 4u * lane);
	const float q0 = LmBf16ToFloat((uint16_t)(q_quad.x & 0xffffu));
	const float q1 = LmBf16ToFloat((uint16_t)(q_quad.x >> 16u));
	const float q2 = LmBf16ToFloat((uint16_t)(q_quad.y & 0xffffu));
	const float q3 = LmBf16ToFloat((uint16_t)(q_quad.y >> 16u));
	float running_max = -INFINITY, running_sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
	float *out = partials + (((uint64_t)q_head * KX2_SPLITS_MAX + split) * KX2_BLOCK + row) * (KX2_HEAD_DIM + 2u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t tile_first = begin + tile * KX2_ATTN_TILE;
		const uint32_t tile_keys = end - tile_first < KX2_ATTN_TILE ? end - tile_first : KX2_ATTN_TILE;
		const uint32_t load_key = threadIdx.x >> 3u, load_chunk = threadIdx.x & 7u;
		if ( threadIdx.x < KX2_ATTN_TILE )
			s_ptrs[threadIdx.x] = KX2KeyRow(pages, layer_head_offset, draft->run_first, draft->run_position, draft->runs,
				tile_first + threadIdx.x, end);
		__syncthreads();
		{
			const uint16_t *src_k = s_ptrs[load_key];
			if ( src_k != 0 )
			{
				const uint16_t *src_v = src_k + KX2_PAGE_HALF_LAYER;
				const uint4 k0 = *(const uint4 *)(src_k + load_chunk * 16u);
				const uint4 k1 = *(const uint4 *)(src_k + load_chunk * 16u + 8u);
				const uint4 v0 = *(const uint4 *)(src_v + load_chunk * 16u);
				const uint4 v1 = *(const uint4 *)(src_v + load_chunk * 16u + 8u);
				s_k[load_key][load_chunk * 8u + 0u] = k0.x;
				s_k[load_key][load_chunk * 8u + 1u] = k0.y;
				s_k[load_key][load_chunk * 8u + 2u] = k0.z;
				s_k[load_key][load_chunk * 8u + 3u] = k0.w;
				s_k[load_key][load_chunk * 8u + 4u] = k1.x;
				s_k[load_key][load_chunk * 8u + 5u] = k1.y;
				s_k[load_key][load_chunk * 8u + 6u] = k1.z;
				s_k[load_key][load_chunk * 8u + 7u] = k1.w;
				s_v[load_key][load_chunk * 8u + 0u] = v0.x;
				s_v[load_key][load_chunk * 8u + 1u] = v0.y;
				s_v[load_key][load_chunk * 8u + 2u] = v0.z;
				s_v[load_key][load_chunk * 8u + 3u] = v0.w;
				s_v[load_key][load_chunk * 8u + 4u] = v1.x;
				s_v[load_key][load_chunk * 8u + 5u] = v1.y;
				s_v[load_key][load_chunk * 8u + 6u] = v1.z;
				s_v[load_key][load_chunk * 8u + 7u] = v1.w;
			}
			else
			{
				for ( uint32_t w = 0u; w < 8u; w++ )
				{
					s_k[load_key][load_chunk * 8u + w] = 0u;
					s_v[load_key][load_chunk * 8u + w] = 0u;
				}
			}
		}
		__syncthreads();
		for ( uint32_t t = 0u; t < tile_keys; t++ )
		{
			const uint32_t kp0 = s_k[t][2u * lane];
			const uint32_t kp1 = s_k[t][2u * lane + 1u];
			float dot = fmaf(q0, LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)),
				fmaf(q1, LmBf16ToFloat((uint16_t)(kp0 >> 16u)),
					fmaf(q2, LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), q3 * LmBf16ToFloat((uint16_t)(kp1 >> 16u)))));
			dot = KX2WarpSum(dot);
			if ( lane == 0u )
				s_scores[warp][t] = dot * scale;
		}
		__syncwarp();
		{
			const float sc = lane < tile_keys ? s_scores[warp][lane] : -INFINITY;
			const float tile_max = KX2WarpMax(sc);
			const float next_max = fmaxf(running_max, tile_max);
			const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
			const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
			const float tile_sum = KX2WarpSum(weight);
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
			const uint32_t vp0 = s_v[t][2u * lane];
			const uint32_t vp1 = s_v[t][2u * lane + 1u];
			acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0);
			acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1);
			acc2 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2);
			acc3 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3);
		}
		__syncthreads();
	}
	out[4u * lane + 0u] = acc0;
	out[4u * lane + 1u] = acc1;
	out[4u * lane + 2u] = acc2;
	out[4u * lane + 3u] = acc3;
	if ( lane == 0u )
	{
		out[KX2_HEAD_DIM] = running_max;
		out[KX2_HEAD_DIM + 1u] = running_sum;
	}
}

__global__ __launch_bounds__(KX2_HEAD_DIM, 1)
void KX2AttentionMergeKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ block_k, const uint16_t *__restrict__ block_v,
	float scale, const float *__restrict__ partials, uint32_t splits, uint16_t *__restrict__ output)
{
	__shared__ float reduction[KX2_HEAD_DIM / KX2_WARP];
	const uint32_t row = blockIdx.x / KX2_HEADS, head = blockIdx.x % KX2_HEADS, kv_head = head / KX2_GQA_GROUP, lane = threadIdx.x;
	const float own = LmBf16ToFloat(q[((uint64_t)row * KX2_HEADS + head) * KX2_HEAD_DIM + lane]);
	float top = -INFINITY, sum = 0.0f, accumulated = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part = partials + (((uint64_t)head * KX2_SPLITS_MAX + split) * KX2_BLOCK + row) * (KX2_HEAD_DIM + 2u);
		const float part_top = part[KX2_HEAD_DIM], next = fmaxf(top, part_top);
		float carry, weight;
		if ( part_top == -INFINITY )
			continue;
		carry = top == -INFINITY ? 0.0f : expf(top - next);
		weight = expf(part_top - next);
		sum = sum * carry + part[KX2_HEAD_DIM + 1u] * weight;
		accumulated = accumulated * carry + part[lane] * weight;
		top = next;
	}
	for ( uint32_t key = 0u; key < KX2_BLOCK; key++ )
	{
		const uint64_t kv_at = ((uint64_t)key * KX2_KV_HEADS + kv_head) * KX2_HEAD_DIM + lane;
		const float score = LmBlockSum<KX2_HEAD_DIM>(own * LmBf16ToFloat(block_k[kv_at]), reduction) * scale;
		const float next = fmaxf(top, score), carry = top == -INFINITY ? 0.0f : expf(top - next), weight = expf(score - next);
		sum = sum * carry + weight;
		accumulated = accumulated * carry + weight * LmBf16ToFloat(block_v[kv_at]);
		top = next;
	}
	output[(uint64_t)row * KX2_Q_WIDTH + head * KX2_HEAD_DIM + lane] = LmFloatToBf16(accumulated / sum);
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2ResidualKernel(const uint16_t *__restrict__ base, const uint16_t *__restrict__ addend, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * KX2_THREADS + threadIdx.x; index < count; index += gridDim.x * KX2_THREADS )
		output[index] = LmFloatToBf16(LmBf16ToFloat(base[index]) + LmBf16ToFloat(addend[index]));
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2SwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * KX2_THREADS + threadIdx.x; index < count; index += gridDim.x * KX2_THREADS )
	{
		const float g = LmBf16ToFloat(gate[index]);
		output[index] = LmFloatToBf16(g / (1.0f + expf(-g)) * LmBf16ToFloat(up[index]));
	}
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2WidenKernel(const uint16_t *__restrict__ input, float *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * KX2_THREADS + threadIdx.x; index < count; index += gridDim.x * KX2_THREADS )
		output[index] = LmBf16ToFloat(input[index]);
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2MarkovRowKernel(uint32_t step, uint32_t anchor, const uint32_t *__restrict__ tokens, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ row)
{
	__shared__ float vector[KX2_MARKOV];
	const uint32_t previous = step == 0u ? anchor : tokens[step - 1u];
	const uint32_t warp = threadIdx.x / KX2_WARP, lane = threadIdx.x % KX2_WARP;
	for ( uint32_t index = threadIdx.x; index < KX2_MARKOV; index += KX2_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * KX2_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * KX2_WARPS + warp; slot < KX2_CANDIDATES; slot += gridDim.x * KX2_WARPS )
	{
		const uint32_t candidate = candidates[step * KX2_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * KX2_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < KX2_MARKOV; element += KX2_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = KX2WarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * KX2_VOCAB + candidate] + sum;
	}
}

static __device__ __forceinline__ unsigned long long KX2RowKey(const float *row, uint32_t slot)
{
	return ((unsigned long long)SparkMathSampleOrdered(row[slot]) << 32u) | (unsigned long long)(0xffffffffu - slot);
}

static __device__ __forceinline__ uint32_t KX2KeySlot(unsigned long long key)
{
	return 0xffffffffu - (uint32_t)key;
}

static __device__ __forceinline__ void KX2Top2Insert(unsigned long long candidate, unsigned long long *first, unsigned long long *second)
{
	if ( candidate > *first )
	{
		*second = *first;
		*first = candidate;
	}
	else if ( candidate > *second && candidate != *first )
		*second = candidate;
}

__global__ __launch_bounds__(KX2_PICK_THREADS, 1)
void KX2MarkovPickKernel(uint32_t step, uint32_t anchor, uint32_t position, unsigned long long seed, float inverse_temperature, float nucleus,
	uint32_t top_k, const float *__restrict__ row, const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ y,
	const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ confidence_weight, const uint16_t *__restrict__ confidence_bias,
	KX2Picks *__restrict__ picks)
{
	__shared__ unsigned long long keys[KX2_CANDIDATES];
	__shared__ unsigned long long best[KX2_PICK_THREADS / KX2_WARP];
	__shared__ unsigned long long second_best[KX2_PICK_THREADS / KX2_WARP];
	__shared__ float reduction[KX2_PICK_THREADS / KX2_WARP];
	__shared__ uint32_t chosen;
	__shared__ uint32_t alt_chosen;
	__shared__ float peak_scaled;
	const uint32_t previous = step == 0u ? anchor : picks->tokens[step - 1u];
	const uint32_t warp = threadIdx.x / KX2_WARP, lane = threadIdx.x % KX2_WARP;
	const float scale = inverse_temperature > 0.0f ? inverse_temperature : 1.0f;
	float dot = 0.0f, denom = 0.0f;
	if ( inverse_temperature > 0.0f )
	{
		for ( uint32_t index = threadIdx.x; index < KX2_CANDIDATES; index += KX2_PICK_THREADS )
			keys[index] = KX2RowKey(row, index);
		__syncthreads();
		for ( uint32_t size = 2u; size <= KX2_CANDIDATES; size <<= 1u )
			for ( uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u )
			{
				for ( uint32_t index = threadIdx.x; index < KX2_CANDIDATES; index += KX2_PICK_THREADS )
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
			const uint32_t slot = KX2KeySlot(keys[lane]);
			const uint32_t token = candidates[step * KX2_CANDIDATES + slot];
			const float scaled = row[slot] * inverse_temperature;
			float peak = KX2WarpMax(scaled), probability = expf(scaled - peak), total = KX2WarpSum(probability), running, keyed;
			uint32_t winner, runner;
			unsigned long long key;
			probability /= total;
			running = probability;
			for ( uint32_t offset = 1u; offset < KX2_SAMPLED; offset <<= 1u )
			{
				const float earlier = __shfl_up_sync(0xffffffffu, running, offset);
				if ( lane >= offset )
					running += earlier;
			}
			keyed = (((nucleus <= 0.0f) || (running - probability < nucleus)) && (top_k == 0u || lane < top_k)) || lane == 0u
				? scaled + SparkMathGumbelNoise(seed, position + step, token) : -INFINITY;
			key = ((unsigned long long)SparkMathSampleOrdered(keyed) << 32u) | (unsigned long long)(0xffffffffu - lane);
			for ( uint32_t mask = KX2_WARP / 2u; mask > 0u; mask >>= 1u )
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
		for ( uint32_t index = threadIdx.x; index < KX2_CANDIDATES; index += KX2_PICK_THREADS )
			KX2Top2Insert(KX2RowKey(row, index), &first, &second);
		for ( uint32_t mask = KX2_WARP / 2u; mask > 0u; mask >>= 1u )
		{
			const unsigned long long other_first = __shfl_xor_sync(0xffffffffu, first, mask);
			const unsigned long long other_second = __shfl_xor_sync(0xffffffffu, second, mask);
			KX2Top2Insert(other_first, &first, &second);
			KX2Top2Insert(other_second, &first, &second);
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
			for ( uint32_t index = 1u; index < KX2_PICK_THREADS / KX2_WARP; index++ )
			{
				KX2Top2Insert(best[index], &first, &second);
				KX2Top2Insert(second_best[index], &first, &second);
			}
			chosen = KX2KeySlot(first);
			alt_chosen = KX2KeySlot(second);
			peak_scaled = row[chosen];
		}
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < KX2_CANDIDATES; index += KX2_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<KX2_PICK_THREADS>(denom, reduction);
	for ( uint32_t index = threadIdx.x; index < KX2_CONFIDENCE; index += KX2_PICK_THREADS )
	{
		const float feature = index < KX2_HIDDEN ? LmBf16ToFloat(y[(uint64_t)step * KX2_HIDDEN + index])
			: LmBf16ToFloat(markov_previous[(uint64_t)previous * KX2_MARKOV + index - KX2_HIDDEN]);
		dot += LmBf16ToFloat(confidence_weight[index]) * feature;
	}
	dot = LmBlockSum<KX2_PICK_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->tokens[step] = candidates[step * KX2_CANDIDATES + chosen];
		picks->scores[step] = 1.0f / (1.0f + expf(-(dot + LmBf16ToFloat(confidence_bias[0]))));
		picks->alt_tokens[step] = candidates[step * KX2_CANDIDATES + alt_chosen];
		picks->alt_scores[step] = denom > 0.0f ? expf(row[alt_chosen] * scale - peak_scaled) / denom : 0.0f;
	}
}

__global__ __launch_bounds__(KX2_WARP, 1)
void KX2BranchSelectKernel(KX2Picks *__restrict__ picks, uint32_t drafts, float min_score)
{
	uint32_t used = 0u;
	if ( threadIdx.x != 0u )
		return;
	for ( uint32_t branch = 0u; branch < KX2_BRANCHES; branch++ )
	{
		picks->branch_fork[branch] = KX2_BLOCK;
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
			if ( pos < KX2_BRANCHES )
			{
				picks->branch_fork[pos] = picks->branch_fork[pos - 1u];
				picks->branch_len[pos] = picks->branch_len[pos - 1u];
				picks->branch_tokens[pos][0] = picks->branch_tokens[pos - 1u][0];
				picks->branch_scores[pos][0] = picks->branch_scores[pos - 1u][0];
			}
			pos--;
		}
		if ( pos < KX2_BRANCHES )
		{
			picks->branch_fork[pos] = step;
			picks->branch_len[pos] = 1u;
			picks->branch_tokens[pos][0] = picks->alt_tokens[step];
			picks->branch_scores[pos][0] = score;
			if ( used < KX2_BRANCHES )
				used++;
		}
	}
}

__global__ __launch_bounds__(KX2_THREADS, 1)
void KX2BranchRowKernel(uint32_t offset, uint32_t drafts, const KX2Picks *__restrict__ picks, const uint32_t *__restrict__ candidates,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ rows)
{
	__shared__ float vector[KX2_MARKOV];
	const uint32_t branch = blockIdx.y, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / KX2_WARP, lane = threadIdx.x % KX2_WARP;
	uint32_t previous;
	float *row;
	if ( fork >= drafts || step >= drafts )
		return;
	previous = picks->branch_tokens[branch][offset - 1u];
	row = rows + (uint64_t)branch * KX2_CANDIDATES;
	for ( uint32_t index = threadIdx.x; index < KX2_MARKOV; index += KX2_THREADS )
		vector[index] = LmBf16ToFloat(markov_previous[(uint64_t)previous * KX2_MARKOV + index]);
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * KX2_WARPS + warp; slot < KX2_CANDIDATES; slot += gridDim.x * KX2_WARPS )
	{
		const uint32_t candidate = candidates[step * KX2_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * KX2_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < KX2_MARKOV; element += KX2_WARP )
			sum += LmBf16ToFloat(edge[element]) * vector[element];
		sum = KX2WarpSum(sum);
		if ( lane == 0u )
			row[slot] = logits[(uint64_t)step * KX2_VOCAB + candidate] + sum;
	}
}

__global__ __launch_bounds__(KX2_PICK_THREADS, 1)
void KX2BranchPickKernel(uint32_t offset, uint32_t drafts, float scale, const float *__restrict__ rows, const uint32_t *__restrict__ candidates,
	KX2Picks *__restrict__ picks)
{
	__shared__ unsigned long long best[KX2_PICK_THREADS / KX2_WARP];
	__shared__ float reduction[KX2_PICK_THREADS / KX2_WARP];
	__shared__ uint32_t chosen;
	__shared__ float peak_scaled;
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / KX2_WARP, lane = threadIdx.x % KX2_WARP;
	const float *row = rows + (uint64_t)branch * KX2_CANDIDATES;
	unsigned long long first = 0ull;
	float denom = 0.0f;
	if ( fork >= drafts || step >= drafts )
		return;
	for ( uint32_t index = threadIdx.x; index < KX2_CANDIDATES; index += KX2_PICK_THREADS )
	{
		const unsigned long long key = KX2RowKey(row, index);
		if ( key > first )
			first = key;
	}
	for ( uint32_t mask = KX2_WARP / 2u; mask > 0u; mask >>= 1u )
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
		for ( uint32_t index = 1u; index < KX2_PICK_THREADS / KX2_WARP; index++ )
			if ( best[index] > first )
				first = best[index];
		chosen = KX2KeySlot(first);
		peak_scaled = row[chosen] * scale;
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < KX2_CANDIDATES; index += KX2_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<KX2_PICK_THREADS>(denom, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->branch_tokens[branch][offset] = candidates[step * KX2_CANDIDATES + chosen];
		picks->branch_scores[branch][offset] = denom > 0.0f ? 1.0f / denom : 0.0f;
		picks->branch_len[branch] = offset + 1u;
	}
}

static SparkStatus KX2Refuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	SPARK_FAIL(status);
}

static SparkStatus KX2LoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
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
	if ( magic != KX2_PACK_MAGIC || version != KX2_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
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
	*payload_base = (sizeof(header) + bytes + KX2_PACK_ALIGNMENT - 1u) / KX2_PACK_ALIGNMENT * KX2_PACK_ALIGNMENT;
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return SPARK_STATUS_OK;
}

static SparkStatus KX2Tensor(const KX2State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t expected,
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
		fprintf(stderr, "kexaone2 dspark: pack tensor %s is missing or is not %llu bf16 elements\n", name, (unsigned long long)expected);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*pointer = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static int32_t KX2Member(const SparkJsonDocument *document, int32_t object, const char *name)
{
	return object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
}

static SparkStatus KX2Shape(KX2State *state, const SparkJsonDocument *document)
{
	const int32_t config = KX2Member(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = KX2Member(document, config, "checkpoint_config");
	const int32_t rope = KX2Member(document, checkpoint, "rope_parameters");
	const int32_t taps = KX2Member(document, checkpoint, "dspark_target_layer_ids");
	const int32_t layer_types = KX2Member(document, checkpoint, "mtp_layer_types");
	const struct
	{
		const char *name;
		uint64_t value;
	} expected[] = {
		{"dspark_block_size", KX2_BLOCK},
		{"dspark_num_hidden_layers", KX2_LAYERS},
		{"dspark_num_target_layers", KX2_TARGET_LAYERS},
		{"dspark_markov_rank", KX2_MARKOV},
		{"hidden_size", KX2_HIDDEN},
		{"intermediate_size", KX2_MLP},
		{"num_attention_heads", KX2_HEADS},
		{"num_key_value_heads", KX2_KV_HEADS},
		{"head_dim", KX2_HEAD_DIM},
		{"vocab_size", KX2_VOCAB},
	};
	const struct
	{
		const char *name;
		bool value;
	} flags[] = {
		{"dspark_enable_confidence_head", true},
		{"dspark_confidence_head_with_markov", true},
	};
	uint64_t value = 0u;
	float theta = 0.0f;
	int32_t token;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		token = KX2Member(document, checkpoint, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "kexaone2 dspark: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	for ( uint32_t index = 0u; index < sizeof(flags) / sizeof(flags[0]); index++ )
	{
		bool flag = false;
		token = KX2Member(document, checkpoint, flags[index].name);
		if ( token < 0 || SparkJsonGetBoolean(document, token, &flag) != SPARK_STATUS_OK || flag != flags[index].value )
		{
			fprintf(stderr, "kexaone2 dspark: checkpoint_config %s does not match module contract\n", flags[index].name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	token = KX2Member(document, checkpoint, "dspark_markov_head_type");
	if ( token < 0 || !SparkJsonStringEquals(document, token, "vanilla") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( taps < 0 || SparkJsonGetArrayElementCount(document, taps) != KX2_TAPS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( layer_types < 0 || SparkJsonGetArrayElementCount(document, layer_types) != KX2_LAYERS )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t layer = 0u; layer < KX2_LAYERS; layer++ )
	{
		const int32_t elem = SparkJsonGetArrayElement(document, layer_types, layer);
		if ( elem < 0 || !SparkJsonStringEquals(document, elem, "full_attention") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	token = KX2Member(document, checkpoint, "dspark_noise_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= KX2_VOCAB )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->mask_token = (uint32_t)value;
	token = KX2Member(document, checkpoint, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = KX2Member(document, rope, "rope_theta");
	if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK || theta <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->softmax_scale = (float)(1.0 / sqrt((double)KX2_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, KX2_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	LM_LAUNCH((KX2RopeTableKernel), 1u, KX2_THREADS, 0, state->stream, state->rope_inverse, theta);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus KX2Weights(KX2State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = KX2LoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = KX2Shape(state, &document);
	{
		const struct
		{
			const char *name;
			uint64_t elements;
			const uint16_t **pointer;
		} plain[] = {
			{"mtp.fc.weight", (uint64_t)KX2_HIDDEN * KX2_TAP_WIDTH, &state->fc},
			{"mtp.hidden_norm.weight", KX2_HIDDEN, &state->hidden_norm},
			{"mtp.norm.weight", KX2_HIDDEN, &state->norm},
			{"target.embed", (uint64_t)KX2_VOCAB * KX2_HIDDEN, &state->embed},
			{"target.head", (uint64_t)KX2_VOCAB * KX2_HIDDEN, &state->head},
			{"mtp.markov_head.markov_w1.weight", (uint64_t)KX2_VOCAB * KX2_MARKOV, &state->markov_previous},
			{"mtp.markov_head.markov_w2.weight", (uint64_t)KX2_VOCAB * KX2_MARKOV, &state->markov_next},
			{"mtp.confidence_head.proj.weight", KX2_CONFIDENCE, &state->confidence_weight},
			{"mtp.confidence_head.proj.bias", 1u, &state->confidence_bias},
		};
		for ( uint32_t index = 0u; index < sizeof(plain) / sizeof(plain[0]) && status == SPARK_STATUS_OK; index++ )
			status = KX2Tensor(state, &document, payload_base, plain[index].name, plain[index].elements, plain[index].pointer);
	}
	for ( uint32_t layer = 0u; layer < KX2_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		KX2Layer *weights = &state->layers[layer];
		const struct
		{
			const char *suffix;
			uint64_t elements;
			const uint16_t **pointer;
		} tensors[] = {
			{"input_layernorm.weight", KX2_HIDDEN, &weights->input_norm},
			{"post_attention_layernorm.weight", KX2_HIDDEN, &weights->post_norm},
			{"self_attn.q_proj.weight", (uint64_t)KX2_Q_WIDTH * KX2_HIDDEN, &weights->q},
			{"self_attn.k_proj.weight", (uint64_t)KX2_KV_WIDTH * KX2_HIDDEN, &weights->k},
			{"self_attn.v_proj.weight", (uint64_t)KX2_KV_WIDTH * KX2_HIDDEN, &weights->v},
			{"self_attn.o_proj.weight", (uint64_t)KX2_HIDDEN * KX2_Q_WIDTH, &weights->o},
			{"self_attn.q_norm.weight", KX2_HEAD_DIM, &weights->q_norm},
			{"self_attn.k_norm.weight", KX2_HEAD_DIM, &weights->k_norm},
			{"mlp.gate_proj.weight", (uint64_t)KX2_MLP * KX2_HIDDEN, &weights->gate},
			{"mlp.up_proj.weight", (uint64_t)KX2_MLP * KX2_HIDDEN, &weights->up},
			{"mlp.down_proj.weight", (uint64_t)KX2_HIDDEN * KX2_MLP, &weights->down},
		};
		for ( uint32_t index = 0u; index < sizeof(tensors) / sizeof(tensors[0]) && status == SPARK_STATUS_OK; index++ )
		{
			char name[128];
			snprintf(name, sizeof(name), "mtp.layers.%u.%s", layer, tensors[index].suffix);
			status = KX2Tensor(state, &document, payload_base, name, tensors[index].elements, tensors[index].pointer);
		}
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define KX2_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus KX2Scratch(KX2State *state)
{
	KX2Forward *forward = &state->forward;
	int multiprocessors = 0;
	if ( SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)KX2_INGEST_ROWS * KX2_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)KX2_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(KX2Draft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_pages, (size_t)KX2_PAGES_MAX * sizeof(*forward->host_pages)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(KX2Picks)) != SPARK_GPU_SUCCESS ||
		!KX2_DEVICE(state->taps, (uint64_t)KX2_INGEST_ROWS * KX2_TAP_WIDTH) || !KX2_DEVICE(state->positions, KX2_INGEST_ROWS) ||
		!KX2_DEVICE(state->features_raw, (uint64_t)KX2_INGEST_ROWS * KX2_HIDDEN) ||
		!KX2_DEVICE(state->features, (uint64_t)KX2_INGEST_ROWS * KX2_HIDDEN) ||
		!KX2_DEVICE(state->keys_raw, (uint64_t)KX2_INGEST_ROWS * KX2_KV_WIDTH) ||
		!KX2_DEVICE(state->values_raw, (uint64_t)KX2_INGEST_ROWS * KX2_KV_WIDTH) ||
		!KX2_DEVICE(forward->draft, 1u) || !KX2_DEVICE(forward->pages, KX2_PAGES_MAX) || !KX2_DEVICE(forward->picks, 1u) ||
		!KX2_DEVICE(forward->hidden, KX2_BLOCK * KX2_HIDDEN) || !KX2_DEVICE(forward->x, KX2_BLOCK * KX2_HIDDEN) ||
		!KX2_DEVICE(forward->q, KX2_BLOCK * KX2_Q_WIDTH) || !KX2_DEVICE(forward->q_roped, KX2_BLOCK * KX2_Q_WIDTH) ||
		!KX2_DEVICE(forward->k, KX2_BLOCK * KX2_KV_WIDTH) || !KX2_DEVICE(forward->k_roped, KX2_BLOCK * KX2_KV_WIDTH) ||
		!KX2_DEVICE(forward->v, KX2_BLOCK * KX2_KV_WIDTH) ||
		!KX2_DEVICE(forward->partials, (uint64_t)KX2_HEADS * KX2_SPLITS_MAX * KX2_BLOCK * (KX2_HEAD_DIM + 2u)) ||
		!KX2_DEVICE(forward->attended, KX2_BLOCK * KX2_Q_WIDTH) || !KX2_DEVICE(forward->projected, KX2_BLOCK * KX2_HIDDEN) ||
		!KX2_DEVICE(forward->residual, KX2_BLOCK * KX2_HIDDEN) || !KX2_DEVICE(forward->gate, KX2_BLOCK * KX2_MLP) ||
		!KX2_DEVICE(forward->up, KX2_BLOCK * KX2_MLP) || !KX2_DEVICE(forward->act, KX2_BLOCK * KX2_MLP) ||
		!KX2_DEVICE(forward->mlp, KX2_BLOCK * KX2_HIDDEN) || !KX2_DEVICE(forward->y, KX2_BLOCK * KX2_HIDDEN) ||
		!KX2_DEVICE(forward->logits, (uint64_t)KX2_BLOCK * KX2_VOCAB) || !KX2_DEVICE(forward->logits_f32, (uint64_t)KX2_BLOCK * KX2_VOCAB) ||
		!KX2_DEVICE(forward->candidates, KX2_BLOCK * KX2_CANDIDATES) || !KX2_DEVICE(forward->markov_row, KX2_BRANCHES * KX2_CANDIDATES) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static void KX2SequenceFree(KX2State *state, KX2Sequence *sequence)
{
	for ( uint32_t page = 0u; page < sequence->page_count; page++ )
		if ( sequence->pages[page] != 0 )
		{
			(void)SparkGpuFree(sequence->pages[page]);
			state->pages_used--;
		}
	free(sequence->pages);
	free(sequence->held);
	free(sequence->tokens);
	memset(sequence, 0, sizeof(*sequence));
}

static KX2Sequence *KX2SequenceFind(KX2State *state, uint64_t id, uint32_t create)
{
	KX2Sequence *empty = 0, *oldest = 0;
	for ( uint32_t index = 0u; index < KX2_SEQUENCES_MAX; index++ )
	{
		KX2Sequence *sequence = &state->sequences[index];
		if ( sequence->pages != 0 && sequence->retired == 0u && sequence->id == id )
		{
			sequence->last_used = ++state->clock;
			return sequence;
		}
		if ( sequence->pages == 0 && empty == 0 )
			empty = sequence;
		if ( sequence->pages != 0 && (oldest == 0 || sequence->last_used < oldest->last_used) )
			oldest = sequence;
	}
	if ( create == 0u )
		return 0;
	if ( empty == 0 )
	{
		KX2SequenceFree(state, oldest);
		empty = oldest;
	}
	empty->pages = (uint16_t **)calloc(KX2_PAGES_MAX, sizeof(uint16_t *));
	if ( empty->pages == 0 )
		return 0;
	empty->id = id;
	empty->last_used = ++state->clock;
	return empty;
}

static KX2Sequence *KX2SequenceOldestBut(KX2State *state, const KX2Sequence *keep, const KX2Sequence *donor)
{
	KX2Sequence *oldest = 0;
	for ( uint32_t index = 0u; index < KX2_SEQUENCES_MAX; index++ )
	{
		KX2Sequence *sequence = &state->sequences[index];
		if ( sequence->pages != 0 && sequence != keep && sequence != donor && (oldest == 0 || sequence->last_used < oldest->last_used) )
			oldest = sequence;
	}
	return oldest;
}

static SparkStatus KX2SequencePages(KX2State *state, KX2Sequence *sequence, uint64_t end, const KX2Sequence *donor)
{
	const uint64_t needed = (end + KX2_PAGE_POSITIONS - 1u) / KX2_PAGE_POSITIONS;
	const uint64_t had = (uint64_t)sequence->page_count * KX2_PAGE_POSITIONS;
	uint8_t *held;
	uint32_t *tokens;
	if ( needed > KX2_PAGES_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( needed <= sequence->page_count )
		return SPARK_STATUS_OK;
	held = (uint8_t *)realloc(sequence->held, (size_t)(needed * KX2_PAGE_POSITIONS));
	if ( held == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	sequence->held = held;
	memset(held + had, 0, (size_t)(needed * KX2_PAGE_POSITIONS - had));
	tokens = (uint32_t *)realloc(sequence->tokens, (size_t)(needed * KX2_PAGE_POSITIONS) * sizeof(uint32_t));
	if ( tokens == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	sequence->tokens = tokens;
	memset(tokens + had, 0, (size_t)(needed * KX2_PAGE_POSITIONS - had) * sizeof(uint32_t));
	for ( uint64_t page = sequence->page_count; page < needed; page++ )
	{
		while ( (state->pages_used + 1u) * KX2_PAGE_BYTES > state->context_bytes )
		{
			KX2Sequence *oldest = KX2SequenceOldestBut(state, sequence, donor);
			if ( oldest == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			KX2SequenceFree(state, oldest);
		}
		if ( SparkGpuMalloc((void **)&sequence->pages[page], (size_t)KX2_PAGE_BYTES) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->pages_used++;
		sequence->page_count = (uint32_t)page + 1u;
	}
	return SPARK_STATUS_OK;
}

static uint16_t *KX2KeySlotPtr(const KX2Sequence *sequence, uint32_t layer, uint32_t kv_head, uint64_t position)
{
	uint16_t *page = sequence->pages[position / KX2_PAGE_POSITIONS];
	return page + (uint64_t)layer * KX2_PAGE_LAYER + ((uint64_t)kv_head * KX2_PAGE_POSITIONS + (position % KX2_PAGE_POSITIONS)) * KX2_HEAD_DIM;
}

static uint16_t *KX2ValSlotPtr(const KX2Sequence *sequence, uint32_t layer, uint32_t kv_head, uint64_t position)
{
	return KX2KeySlotPtr(sequence, layer, kv_head, position) + KX2_PAGE_HALF_LAYER;
}

static void KX2Forget(KX2Sequence *sequence, uint64_t first)
{
	const uint64_t capacity = (uint64_t)sequence->page_count * KX2_PAGE_POSITIONS;
	if ( first < capacity )
		memset(sequence->held + first, 0, (size_t)(capacity - first));
	if ( sequence->held_end > first )
		sequence->held_end = first;
	if ( sequence->verified > first )
		sequence->verified = first;
}

static SparkStatus KX2Gemm(KX2State *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus KX2IngestRows(KX2State *state, uint32_t rows)
{
	SparkStatus status = KX2Gemm(state, state->fc, state->taps, state->features_raw, rows, KX2_TAP_WIDTH, KX2_HIDDEN);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((KX2RmsKernel), rows, KX2_THREADS, 0, state->stream, state->features_raw, state->hidden_norm, state->features, KX2_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < KX2_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = KX2Gemm(state, state->layers[layer].k, state->features, state->keys_raw, rows, KX2_HIDDEN, KX2_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = KX2Gemm(state, state->layers[layer].v, state->features, state->values_raw, rows, KX2_HIDDEN, KX2_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((KX2IngestKeyRopeKernel), rows * KX2_KV_HEADS, KX2_HEAD_DIM, 0, state->stream, state->keys_raw, state->layers[layer].k_norm,
			state->rope_inverse, state->positions, state->epsilon, state->forward.pages, layer);
		LM_LAUNCH((KX2IngestStoreValuesKernel), rows, KX2_THREADS, 0, state->stream, state->values_raw, state->positions, state->forward.pages, layer);
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static SparkStatus KX2Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	KX2State *state = (KX2State *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t first = frame->header.first_position, end = first + rows;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	KX2Sequence *sequence;
	SparkStatus status;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != KX2_TAPS || hidden != KX2_HIDDEN || frame->rows == 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for ( uint32_t tap = 0u; tap < KX2_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * KX2_HIDDEN > frame->row_elements )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( end > (uint64_t)KX2_PAGES_MAX * KX2_PAGE_POSITIONS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = KX2SequenceFind(state, frame->header.sequence_id, 0u);
	if ( sequence != 0 && first != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = KX2SequenceFind(state, frame->header.sequence_id, 1u);
		if ( sequence == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		sequence->control_generation = frame->header.control_generation;
		sequence->request_id = frame->header.request_id;
	}
	status = KX2SequencePages(state, sequence, end, 0);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( first == 0u )
	{
		KX2Forget(sequence, 0u);
		sequence->control_generation = frame->header.control_generation;
		sequence->request_id = frame->header.request_id;
	}
	KX2Forget(sequence, first);
	for ( uint32_t page = 0u; page < sequence->page_count; page++ )
		state->forward.host_pages[page] = sequence->pages[page];
	if ( SparkGpuMemcpyAsync(state->forward.pages, state->forward.host_pages, (size_t)sequence->page_count * sizeof(*state->forward.host_pages),
		SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
	{
		(void)SparkGpuStreamSynchronize(state->stream);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	for ( uint32_t done = 0u; done < rows && status == SPARK_STATUS_OK; done += KX2_INGEST_ROWS )
	{
		const uint32_t part = rows - done < KX2_INGEST_ROWS ? rows - done : KX2_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < KX2_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * KX2_TAP_WIDTH + (uint64_t)tap * KX2_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * KX2_HIDDEN, KX2_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(first + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * KX2_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
		{
			(void)SparkGpuStreamSynchronize(state->stream);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		status = KX2IngestRows(state, part);
	}
	if ( status == SPARK_STATUS_OK )
	{
		memset(sequence->held + first, 1, rows);
		if ( frame->tokens != 0 )
			memcpy(sequence->tokens + first, frame->tokens, (size_t)rows * sizeof(uint32_t));
		sequence->held_end = end;
	}
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus KX2DraftRuns(const KX2Sequence *sequence, uint64_t position, KX2Draft *draft, uint32_t *held_out)
{
	const uint64_t capacity = sequence != 0 ? (uint64_t)sequence->page_count * KX2_PAGE_POSITIONS : 0u;
	const uint64_t limit = position < capacity ? position : capacity;
	uint64_t at = 0u, held = 0u;
	draft->runs = 0u;
	while ( at < limit )
	{
		const uint8_t *start = (const uint8_t *)memchr(sequence->held + at, 1, (size_t)(limit - at));
		const uint8_t *stop;
		uint64_t begin, finish;
		if ( start == 0 )
			break;
		begin = (uint64_t)(start - sequence->held);
		stop = (const uint8_t *)memchr(start, 0, (size_t)(limit - begin));
		finish = stop != 0 ? (uint64_t)(stop - sequence->held) : limit;
		if ( draft->runs == KX2_RUNS_MAX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		draft->run_first[draft->runs] = (uint32_t)held;
		draft->run_position[draft->runs] = (uint32_t)begin;
		draft->runs++;
		held += finish - begin;
		at = finish;
	}
	*held_out = (uint32_t)held;
	return SPARK_STATUS_OK;
}

static uint32_t KX2Covers(const KX2Sequence *sequence, uint64_t position)
{
	if ( position == 0u )
		return 1u;
	return sequence != 0 && position - 1u < (uint64_t)sequence->page_count * KX2_PAGE_POSITIONS && sequence->held[position - 1u] != 0u ? 1u : 0u;
}

static uint32_t KX2Complete(const KX2Sequence *sequence, uint64_t position)
{
	return position <= (uint64_t)sequence->page_count * KX2_PAGE_POSITIONS && memchr(sequence->held, 0, (size_t)position) == 0 ? 1u : 0u;
}

static SparkStatus KX2CopyRows(KX2State *state, KX2Sequence *sequence, const KX2Sequence *donor, uint64_t first, uint64_t end)
{
	while ( first < end )
	{
		const uint64_t page_end = (first / KX2_PAGE_POSITIONS + 1u) * KX2_PAGE_POSITIONS, stop = end < page_end ? end : page_end;
		const size_t bytes = (size_t)(stop - first) * KX2_HEAD_DIM * sizeof(uint16_t);
		for ( uint32_t layer = 0u; layer < KX2_LAYERS; layer++ )
			for ( uint32_t head = 0u; head < KX2_KV_HEADS; head++ )
				if ( SparkGpuMemcpyAsync(KX2KeySlotPtr(sequence, layer, head, first), KX2KeySlotPtr(donor, layer, head, first), bytes,
					SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
					SparkGpuMemcpyAsync(KX2ValSlotPtr(sequence, layer, head, first), KX2ValSlotPtr(donor, layer, head, first), bytes,
						SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		memset(sequence->held + first, 1, (size_t)(stop - first));
		memcpy(sequence->tokens + first, donor->tokens + first, (size_t)(stop - first) * sizeof(uint32_t));
		first = stop;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus KX2CopyPrefix(KX2State *state, KX2Sequence *sequence, const KX2Sequence *donor, const SparkDraftServerRequest *request)
{
	const uint64_t donor_capacity = (uint64_t)donor->page_count * KX2_PAGE_POSITIONS;
	const uint64_t shared = request->position < donor_capacity ? request->position : donor_capacity;
	uint64_t prefix = 0u, at = 0u;
	SparkStatus status;
	while ( prefix < shared && donor->tokens[prefix] == request->committed_token_ids[prefix] )
		prefix++;
	if ( prefix == 0u )
		return SPARK_STATUS_OK;
	status = KX2SequencePages(state, sequence, prefix, donor);
	while ( status == SPARK_STATUS_OK && at < prefix )
	{
		uint64_t stop;
		if ( donor->held[at] == 0u || sequence->held[at] != 0u )
		{
			at++;
			continue;
		}
		stop = at + 1u;
		while ( stop < prefix && donor->held[stop] != 0u && sequence->held[stop] == 0u )
			stop++;
		status = KX2CopyRows(state, sequence, donor, at, stop);
		at = stop;
	}
	return status;
}

static SparkStatus KX2FillFromDonors(KX2State *state, KX2Sequence *sequence, const SparkDraftServerRequest *request)
{
	static_assert(KX2_SEQUENCES_MAX <= 64u, "one bit per sequence marks the donors tried");
	uint64_t tried = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	while ( status == SPARK_STATUS_OK && KX2Complete(sequence, request->position) == 0u )
	{
		KX2Sequence *donor = 0;
		uint32_t chosen = KX2_SEQUENCES_MAX;
		for ( uint32_t index = 0u; index < KX2_SEQUENCES_MAX; index++ )
		{
			KX2Sequence *other = &state->sequences[index];
			if ( other->pages != 0 && other != sequence && ((tried >> index) & 1u) == 0u && (donor == 0 || other->last_used > donor->last_used) )
			{
				donor = other;
				chosen = index;
			}
		}
		if ( donor == 0 )
			break;
		tried |= UINT64_C(1) << chosen;
		status = KX2CopyPrefix(state, sequence, donor, request);
	}
	return status;
}

static uint32_t KX2TokensMatch(KX2Sequence *sequence, const SparkDraftServerRequest *request)
{
	const uint64_t capacity = (uint64_t)sequence->page_count * KX2_PAGE_POSITIONS;
	const uint64_t end = request->position < capacity ? request->position : capacity;
	uint32_t complete = end == request->position ? 1u : 0u;
	if ( (uint64_t)request->committed_token_count < request->position )
		return 0u;
	for ( uint64_t at = sequence->verified; at < end; at++ )
	{
		if ( sequence->held[at] == 0u )
			complete = 0u;
		else if ( sequence->tokens[at] != request->committed_token_ids[at] )
			return 0u;
	}
	if ( complete != 0u && end > sequence->verified )
		sequence->verified = end;
	return 1u;
}

static SparkStatus KX2RunForward(KX2State *state, const KX2Sequence *sequence, uint32_t anchor, uint64_t position, uint32_t layers)
{
	KX2Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t pages = sequence != 0 ? sequence->page_count : 0u;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(KX2Draft, positions));
	uint32_t held = 0u, splits = 0u, split_rows = 0u;
	SparkStatus status;
	if ( anchor >= KX2_VOCAB || position + KX2_BLOCK > (uint64_t)KX2_PAGES_MAX * KX2_PAGE_POSITIONS || layers == 0u || layers > KX2_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(forward->host_draft, 0, sizeof(*forward->host_draft));
	status = KX2DraftRuns(sequence, position, forward->host_draft, &held);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t row = 0u; row < KX2_BLOCK; row++ )
		forward->host_draft->positions[row] = (uint32_t)(position + row);
	forward->host_draft->anchor = anchor;
	forward->host_draft->mask = state->mask_token;
	for ( uint32_t page = 0u; page < pages; page++ )
		forward->host_pages[page] = sequence->pages[page];
	if ( held != 0u )
	{
		splits = (held + KX2_SPLIT_ROWS_MIN - 1u) / KX2_SPLIT_ROWS_MIN;
		splits = splits > KX2_SPLITS_MAX ? KX2_SPLITS_MAX : splits;
		split_rows = ((held + splits - 1u) / splits + KX2_ATTN_TILE - 1u) / KX2_ATTN_TILE * KX2_ATTN_TILE;
		splits = (held + split_rows - 1u) / split_rows;
	}
	if ( SparkGpuMemcpyAsync(forward->draft, forward->host_draft, sizeof(KX2Draft), SPARK_GPU_COPY_HOST_TO_DEVICE, stream) != SPARK_GPU_SUCCESS ||
		(pages != 0u && SparkGpuMemcpyAsync(forward->pages, forward->host_pages, (size_t)pages * sizeof(*forward->host_pages),
			SPARK_GPU_COPY_HOST_TO_DEVICE, stream) != SPARK_GPU_SUCCESS) )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	LM_LAUNCH((KX2EmbedKernel), KX2_BLOCK, KX2_THREADS, 0, stream, state->embed, forward->draft, forward->hidden);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const KX2Layer *weights = &state->layers[layer];
		LM_LAUNCH((KX2RmsKernel), KX2_BLOCK, KX2_THREADS, 0, stream, forward->hidden, weights->input_norm, forward->x, KX2_HIDDEN, state->epsilon);
		status = KX2Gemm(state, weights->q, forward->x, forward->q, KX2_BLOCK, KX2_HIDDEN, KX2_Q_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = KX2Gemm(state, weights->k, forward->x, forward->k, KX2_BLOCK, KX2_HIDDEN, KX2_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = KX2Gemm(state, weights->v, forward->x, forward->v, KX2_BLOCK, KX2_HIDDEN, KX2_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((KX2BlockHeadNormRopeKernel), KX2_BLOCK * KX2_HEADS, KX2_HEAD_DIM, 0, stream, forward->q, weights->q_norm, state->rope_inverse,
			positions, KX2_HEADS, state->epsilon, forward->q_roped);
		LM_LAUNCH((KX2BlockHeadNormRopeKernel), KX2_BLOCK * KX2_KV_HEADS, KX2_HEAD_DIM, 0, stream, forward->k, weights->k_norm, state->rope_inverse,
			positions, KX2_KV_HEADS, state->epsilon, forward->k_roped);
		if ( splits != 0u )
			LM_LAUNCH((KX2ContextAttentionKernel), dim3(KX2_KV_HEADS * KX2_BLOCK, splits), KX2_ATTN_THREADS, 0, stream, forward->q_roped,
				forward->pages, layer, forward->draft, held, split_rows, state->softmax_scale, forward->partials);
		LM_LAUNCH((KX2AttentionMergeKernel), KX2_BLOCK * KX2_HEADS, KX2_HEAD_DIM, 0, stream, forward->q_roped, forward->k_roped, forward->v,
			state->softmax_scale, forward->partials, splits, forward->attended);
		status = KX2Gemm(state, weights->o, forward->attended, forward->projected, KX2_BLOCK, KX2_Q_WIDTH, KX2_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((KX2ResidualKernel), 2u * state->multiprocessors, KX2_THREADS, 0, stream, forward->hidden, forward->projected, forward->residual,
			KX2_BLOCK * KX2_HIDDEN);
		LM_LAUNCH((KX2RmsKernel), KX2_BLOCK, KX2_THREADS, 0, stream, forward->residual, weights->post_norm, forward->x, KX2_HIDDEN, state->epsilon);
		status = KX2Gemm(state, weights->gate, forward->x, forward->gate, KX2_BLOCK, KX2_HIDDEN, KX2_MLP);
		if ( status == SPARK_STATUS_OK )
			status = KX2Gemm(state, weights->up, forward->x, forward->up, KX2_BLOCK, KX2_HIDDEN, KX2_MLP);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((KX2SwigluKernel), 2u * state->multiprocessors, KX2_THREADS, 0, stream, forward->gate, forward->up, forward->act, KX2_BLOCK * KX2_MLP);
		status = KX2Gemm(state, weights->down, forward->act, forward->mlp, KX2_BLOCK, KX2_MLP, KX2_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((KX2ResidualKernel), 2u * state->multiprocessors, KX2_THREADS, 0, stream, forward->residual, forward->mlp, forward->hidden,
			KX2_BLOCK * KX2_HIDDEN);
	}
	if ( status != SPARK_STATUS_OK || layers < KX2_LAYERS )
		return status;
	LM_LAUNCH((KX2RmsKernel), KX2_BLOCK, KX2_THREADS, 0, stream, forward->hidden, state->norm, forward->y, KX2_HIDDEN, state->epsilon);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus KX2Heads(KX2State *state, uint32_t anchor_token, uint64_t position, const SparkRowSampling *sampling, uint32_t drafts, uint32_t tree)
{
	KX2Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const float nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	SparkStatus status = KX2Gemm(state, state->head, forward->y, forward->logits, KX2_BLOCK, KX2_HIDDEN, KX2_VOCAB);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((KX2WidenKernel), 4u * state->multiprocessors, KX2_THREADS, 0, stream, forward->logits, forward->logits_f32, KX2_BLOCK * KX2_VOCAB);
	if ( LmTopkExactRowsLaunch<KX2_TOPK_THREADS>(forward->logits_f32, KX2_BLOCK, KX2_VOCAB, KX2_CANDIDATES, forward->candidates, stream) !=
		SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t step = 0u; step < KX2_BLOCK; step++ )
	{
		LM_LAUNCH((KX2MarkovRowKernel), KX2_CANDIDATES / KX2_WARPS, KX2_THREADS, 0, stream, step, anchor_token, forward->picks->tokens,
			forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
		LM_LAUNCH((KX2MarkovPickKernel), 1u, KX2_PICK_THREADS, 0, stream, step, anchor_token, (uint32_t)position, (unsigned long long)sampling->seed,
			sampling->inverse_temperature, nucleus, sampling->top_k, forward->markov_row, forward->candidates, forward->y, state->markov_previous,
			state->confidence_weight, state->confidence_bias, forward->picks);
	}
	if ( tree != 0u && drafts >= 2u )
	{
		const float scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
		LM_LAUNCH((KX2BranchSelectKernel), 1u, KX2_WARP, 0, stream, forward->picks, drafts, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE);
		for ( uint32_t offset = 1u; offset < KX2_BRANCH_LEN && offset < drafts; offset++ )
		{
			LM_LAUNCH((KX2BranchRowKernel), dim3(KX2_CANDIDATES / KX2_WARPS, KX2_BRANCHES), KX2_THREADS, 0, stream, offset, drafts,
				forward->picks, forward->candidates, forward->logits_f32, state->markov_previous, state->markov_next, forward->markov_row);
			LM_LAUNCH((KX2BranchPickKernel), KX2_BRANCHES, KX2_PICK_THREADS, 0, stream, offset, drafts, scale, forward->markov_row, forward->candidates,
				forward->picks);
		}
	}
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(KX2Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus KX2Settle(KX2State *state, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return status;
}

static void KX2ForwardFree(KX2Forward *forward)
{
	void *buffers[] = {
		forward->draft, forward->pages, forward->picks, forward->hidden, forward->x, forward->q, forward->q_roped, forward->k, forward->k_roped,
		forward->v, forward->partials, forward->attended, forward->projected, forward->residual, forward->gate, forward->up, forward->act, forward->mlp,
		forward->y, forward->logits, forward->logits_f32, forward->candidates, forward->markov_row,
	};
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		(void)SparkGpuFree(buffers[index]);
	if ( forward->host_draft != 0 )
		(void)SparkGpuFreeHost(forward->host_draft);
	if ( forward->host_pages != 0 )
		(void)SparkGpuFreeHost((void *)forward->host_pages);
	if ( forward->host_picks != 0 )
		(void)SparkGpuFreeHost(forward->host_picks);
}

static SparkStatus KX2Close(void *opaque)
{
	KX2State *state = (KX2State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	for ( uint32_t index = 0u; index < KX2_SEQUENCES_MAX; index++ )
		if ( state->sequences[index].pages != 0 )
			KX2SequenceFree(state, &state->sequences[index]);
	KX2ForwardFree(&state->forward);
	{
		void *buffers[] = {
			state->taps, state->positions, state->features_raw, state->features, state->keys_raw, state->values_raw, state->rope_inverse,
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
	free(state);
	return status;
}

static SparkStatus KX2Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	KX2State *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return KX2Refuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	{
		const SparkStatus device = SparkDeviceInitialize(0);
		if ( (device != SPARK_STATUS_OK && device != SPARK_STATUS_DUPLICATE) || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
			return KX2Refuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
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
	if ( lane->context_bytes < KX2_PAGE_BYTES )
		return KX2Refuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one context page", lane->lane_name);
	state = (KX2State *)calloc(1u, sizeof(*state));
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
	state->context_bytes = lane->context_bytes;
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
		status = KX2Weights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = KX2Scratch(state);
	if ( status == SPARK_STATUS_OK )
		status = KX2Settle(state, SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a K-EXAONE 2.0 DSpark pack this module reads", lane->lane_name,
			lane->pack_path);
		(void)KX2Close(state);
		SPARK_FAIL(status);
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s", lane->lane_name, lane->pack_path);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus KX2Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	KX2State *state = (KX2State *)opaque;
	KX2Sequence *sequence = KX2SequenceFind(state, request->sequence_id, 0u);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( KX2Covers(sequence, request->position) == 0u )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	if ( sequence != 0 && KX2Complete(sequence, request->position) == 0u )
	{
		status = KX2FillFromDonors(state, sequence, request);
		if ( status != SPARK_STATUS_OK )
			return KX2Settle(state, status);
	}
	if ( sequence != 0 && KX2TokensMatch(sequence, request) == 0u )
		return KX2Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	drafts = drafts < KX2_BLOCK ? drafts : KX2_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		SparkDraftTreeBranch branches[KX2_BLOCK];
		uint32_t branch_count = 0u;
		status = KX2RunForward(state, sequence, request->anchor_token_id, request->position, KX2_LAYERS);
		if ( status == SPARK_STATUS_OK )
			status = KX2Heads(state, request->anchor_token_id, request->position, &request->sampling, drafts, tree);
		status = KX2Settle(state, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				uint32_t matched = KX2_BRANCHES;
				for ( uint32_t b = 0u; b < KX2_BRANCHES; b++ )
					if ( state->forward.host_picks->branch_len[b] > 0u && state->forward.host_picks->branch_fork[b] == step )
						matched = b;
				branches[branch_count].fork_step = step;
				if ( matched < KX2_BRANCHES )
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

static const SparkDrafterModuleInterface kx2_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"kexaone2-dspark",
	KX2Open,
	KX2Ingest,
	KX2Propose,
	KX2Close,
};

static SparkStatus KX2Readback(void *host, const void *device, size_t bytes)
{
	if ( SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &kx2_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_kexaone2_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint64_t first, uint32_t count, uint16_t *keys, uint16_t *values)
{
	KX2State *state = (KX2State *)opaque;
	const KX2Sequence *sequence = state != 0 ? KX2SequenceFind(state, sequence_id, 0u) : 0;
	uint64_t at = first, end = first + count;
	if ( sequence == 0 || layer >= KX2_LAYERS || keys == 0 || values == 0 || end > sequence->held_end )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	while ( at < end )
	{
		const uint64_t page_end = (at / KX2_PAGE_POSITIONS + 1u) * KX2_PAGE_POSITIONS, stop = end < page_end ? end : page_end;
		const size_t bytes = (size_t)(stop - at) * KX2_HEAD_DIM * sizeof(uint16_t);
		for ( uint32_t head = 0u; head < KX2_KV_HEADS; head++ )
		{
			const uint64_t dst = ((uint64_t)head * count + (at - first)) * KX2_HEAD_DIM;
			if ( KX2Readback(keys + dst, KX2KeySlotPtr(sequence, layer, head, at), bytes) != SPARK_STATUS_OK ||
				KX2Readback(values + dst, KX2ValSlotPtr(sequence, layer, head, at), bytes) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		at = stop;
	}
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_kexaone2_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	KX2State *state = (KX2State *)opaque;
	const KX2Sequence *sequence = state != 0 ? KX2SequenceFind(state, sequence_id, 0u) : 0;
	const uint64_t started = SparkMonotonicNs();
	SparkStatus status;
	if ( state == 0 || sequence == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	status = KX2RunForward(state, sequence, anchor, position, KX2_LAYERS);
	status = KX2Settle(state, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = KX2Readback(y, state->forward.y, (size_t)KX2_BLOCK * KX2_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_kexaone2_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	KX2State *state = (KX2State *)opaque;
	const KX2Sequence *sequence = state != 0 ? KX2SequenceFind(state, sequence_id, 0u) : 0;
	SparkStatus status;
	if ( state == 0 || sequence == 0 || layers == 0u || layers > KX2_LAYERS || hidden == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = KX2RunForward(state, sequence, anchor, position, layers);
	status = KX2Settle(state, status);
	if ( status == SPARK_STATUS_OK )
		status = KX2Readback(hidden, state->forward.hidden, (size_t)KX2_BLOCK * KX2_HIDDEN * sizeof(uint16_t));
	return status;
}
