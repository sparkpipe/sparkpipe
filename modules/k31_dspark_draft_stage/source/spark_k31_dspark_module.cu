#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_draft_tree_pack.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/drafter_blackwell.cuh"
#include "runtime/launch.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K31_HIDDEN 7168u
#define K31_TAPS 5u
#define K31_TAP_WIDTH (K31_TAPS * K31_HIDDEN)
#define K31_LAYERS 5u
#define K31_KV_RANK 512u
#define K31_ROPE 64u
#define K31_LATENT (K31_KV_RANK + K31_ROPE)
#define K31_HEADS 96u
#define K31_NOPE 128u
#define K31_VALUE 128u
#define K31_QK (K31_NOPE + K31_ROPE)
#define K31_QUERY_RANK 1536u
#define K31_MLP 14336u
#define K31_BLOCK 7u
#define K31_BRANCHES 2u
#define K31_BRANCH_LEN 4u
#define K31_VOCAB 163840u
#define K31_QUERIES (K31_BLOCK * K31_HEADS)
#define K31_MARKOV 512u
#define K31_CANDIDATES 2048u
#define K31_SAMPLED 32u
#define K31_PICK_THREADS 1024u
#define K31_TOPK_THREADS 512u
#define K31_CONFIDENCE (K31_HIDDEN + K31_MARKOV)
#define K31_PAGE_POSITIONS 4096u
#define K31_PAGE_BYTES ((uint64_t)K31_LAYERS * K31_PAGE_POSITIONS * K31_LATENT * sizeof(uint16_t))
#define K31_PAGES_MAX 4096u
#define K31_SEQUENCES_MAX 64u
#define K31_INGEST_ROWS 256u
#define K31_THREADS 256u
#define K31_WARP 32u
#define K31_CHUNK_BYTES 16u
#define K31_CHUNK_ELEMENTS (K31_CHUNK_BYTES / sizeof(uint16_t))
#define K31_ROW_BYTES (K31_LATENT * sizeof(uint16_t))
#define K31_ATTN_THREADS 256u
#define K31_ATTN_QUERIES 32u
#define K31_ATTN_TILE 32u
#define K31_ATTN_STRIDE (K31_ATTN_TILE + 8u)
#define K31_ATTN_TILES (K31_QUERIES / K31_ATTN_QUERIES)
#define K31_ATTN_SHARED_BYTES (2u * K31_ATTN_TILE * K31_ROW_BYTES + 2u * K31_ATTN_QUERIES * K31_ATTN_STRIDE * sizeof(float) + \
	K31_ATTN_QUERIES * K31_ATTN_STRIDE * sizeof(uint16_t) + 3u * K31_ATTN_QUERIES * sizeof(float) + 2u * K31_ATTN_TILE * sizeof(void *))
#define K31_MERGE_THREADS 128u
#define K31_SPLITS_MAX 8u
#define K31_SPLIT_ROWS_MIN 1024u
#define K31_RUNS_MAX 256u
#define K31_LOG2E 1.4426950408889634
#define K31_PACK_MAGIC 0x4B505244u
#define K31_PACK_VERSION 1u
#define K31_PACK_ALIGNMENT 128u
#define K31_PI 3.14159265358979323846
#define K31_INVERSE_PI 0.31830988618379067154

typedef struct K31Layer
{
	const uint16_t *kv_a;
	const uint16_t *kv_a_norm;
	const uint16_t *input_norm;
	const uint16_t *post_norm;
	const uint16_t *q_a;
	const uint16_t *q_a_norm;
	const uint16_t *q_b;
	const uint16_t *kv_b;
	const uint16_t *o_proj;
	const uint16_t *gate;
	const uint16_t *up;
	const uint16_t *down;
} K31Layer;

typedef struct K31Sequence
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
} K31Sequence;

typedef struct K31Draft
{
	uint64_t seed;
	uint32_t anchor;
	uint32_t mask;
	uint32_t position;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	float scale;
	uint32_t held;
	uint32_t splits;
	uint32_t split_rows;
	uint32_t drafts;
	uint32_t positions[K31_BLOCK];
	uint32_t runs;
	uint32_t run_first[K31_RUNS_MAX];
	uint32_t run_position[K31_RUNS_MAX];
} K31Draft;

typedef struct K31Picks
{
	uint32_t tokens[K31_BLOCK];
	float scores[K31_BLOCK];
	uint32_t alt_tokens[K31_BLOCK];
	float alt_scores[K31_BLOCK];
	uint32_t branch_fork[K31_BRANCHES];
	uint32_t branch_len[K31_BRANCHES];
	uint32_t branch_tokens[K31_BRANCHES][K31_BRANCH_LEN];
	float branch_scores[K31_BRANCHES][K31_BRANCH_LEN];
} K31Picks;

typedef struct K31Forward
{
	K31Draft *host_draft;
	K31Draft *draft;
	const uint16_t **host_pages;
	const uint16_t **pages;
	uint16_t **block_destinations;
	float *residual;
	uint16_t *x;
	uint16_t *q_a_raw;
	uint16_t *q_a;
	uint16_t *q;
	uint16_t *latent_raw;
	uint16_t *block_rows;
	uint16_t *queries;
	float *queries_f32;
	float *partials;
	float *stats;
	float *latent;
	uint16_t *values;
	uint16_t *attention;
	uint16_t *gate;
	uint16_t *up;
	uint16_t *act;
	uint16_t *mlp;
	uint16_t *y;
	uint16_t *logits;
	SparkDrafterTopkBf16Workspace topk_ws;
	uint32_t *candidates;
	float *markov_row;
	K31Picks *picks;
	K31Picks *host_picks;
} K31Forward;

typedef struct K31State
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	uint32_t mask_token;
	float epsilon;
	float rope_gain;
	float attention_scale;
	float *rope_inverse;
	const uint16_t *context_proj;
	const uint16_t *context_norm;
	const uint16_t *embed;
	const uint16_t *final_norm;
	const uint16_t *head;
	const uint16_t *markov_previous;
	const uint16_t *markov_next;
	const uint16_t *confidence_weight;
	const uint16_t *confidence_bias;
	K31Layer layers[K31_LAYERS];
	K31Forward forward;
	SparkDrafterGraphCache graphs;
	uint16_t **ingest_rows;
	uint16_t *host_taps;
	uint16_t *taps;
	uint16_t *features_raw;
	uint16_t *features;
	uint16_t *latent_raw;
	uint32_t *positions;
	uint32_t *host_positions;
	uint64_t context_bytes;
	uint64_t pages_used;
	uint64_t clock;
	K31Sequence sequences[K31_SEQUENCES_MAX];
	uint64_t frames;
	uint64_t rows;
} K31State;

__global__ __launch_bounds__(K31_THREADS, 2)
void K31RmsRowsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[K31_THREADS / K31_WARP];
	const uint16_t *row = input + (uint64_t)blockIdx.x * dimension;
	uint16_t *out = output + (uint64_t)blockIdx.x * dimension;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += K31_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(row + index));
		total += value * value;
	}
	total = LmBlockSum<K31_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += K31_THREADS )
		out[index] = LmFloatToBf16(LmBf16ToFloat(__ldg(row + index)) * scale * LmBf16ToFloat(__ldg(weight + index)));
	LmDependentRelease();
}

static __device__ __forceinline__ void K31Rotation(uint32_t position, float inverse, float gain, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * K31_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	float s, c;
	sincospif(reduced, &s, &c);
	*cosine = c * gain;
	*sine = s * gain;
}

__global__ __launch_bounds__(K31_THREADS, 2)
void K31LatentRowsKernel(const uint16_t *__restrict__ latent, const uint16_t *__restrict__ norm, const float *__restrict__ inverse, float gain,
	const uint32_t *__restrict__ positions, uint16_t *const *__restrict__ destinations, float epsilon)
{
	LmDependentWait();
	__shared__ float reduction[K31_THREADS / K31_WARP];
	const uint16_t *row = latent + (uint64_t)blockIdx.x * K31_LATENT;
	uint16_t *out = destinations[blockIdx.x];
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < K31_KV_RANK; index += K31_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(row + index));
		total += value * value;
	}
	total = LmBlockSum<K31_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)K31_KV_RANK + epsilon);
	for ( uint32_t index = threadIdx.x; index < K31_KV_RANK; index += K31_THREADS )
		out[index] = LmFloatToBf16(LmBf16ToFloat(__ldg(row + index)) * scale * LmBf16ToFloat(__ldg(norm + index)));
	if ( threadIdx.x < K31_ROPE / 2u )
	{
		const uint32_t pair = threadIdx.x;
		float cosine, sine;
		K31Rotation(positions[blockIdx.x], inverse[pair], gain, &cosine, &sine);
		const float even = LmBf16ToFloat(__ldg(row + K31_KV_RANK + 2u * pair)), odd = LmBf16ToFloat(__ldg(row + K31_KV_RANK + 2u * pair + 1u));
		out[K31_KV_RANK + 2u * pair] = LmFloatToBf16(even * cosine - odd * sine);
		out[K31_KV_RANK + 2u * pair + 1u] = LmFloatToBf16(even * sine + odd * cosine);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_THREADS, 2)
void K31EmbedKernel(const uint16_t *__restrict__ embed, const K31Draft *__restrict__ draft, float *__restrict__ residual)
{
	const uint32_t token = blockIdx.x == 0u ? draft->anchor : draft->mask;
	const uint4 *src = (const uint4 *)(embed + (uint64_t)token * K31_HIDDEN);
	float4 *dst = (float4 *)(residual + (uint64_t)blockIdx.x * K31_HIDDEN);
	for ( uint32_t index = threadIdx.x; index < K31_HIDDEN / 8u; index += K31_THREADS )
	{
		const uint4 p = __ldg(src + index);
		dst[2u * index] = make_float4(__uint_as_float(p.x << 16u), __uint_as_float(p.x & 0xffff0000u),
			__uint_as_float(p.y << 16u), __uint_as_float(p.y & 0xffff0000u));
		dst[2u * index + 1u] = make_float4(__uint_as_float(p.z << 16u), __uint_as_float(p.z & 0xffff0000u),
			__uint_as_float(p.w << 16u), __uint_as_float(p.w & 0xffff0000u));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_THREADS, 2)
void K31ResidualNormKernel(float *__restrict__ residual, const uint16_t *__restrict__ addend, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ output, float epsilon)
{
	__shared__ float reduction[K31_THREADS / K31_WARP];
	float *row = residual + (uint64_t)blockIdx.x * K31_HIDDEN;
	const uint16_t *add = addend != 0 ? addend + (uint64_t)blockIdx.x * K31_HIDDEN : (const uint16_t *)0;
	uint16_t *out = output + (uint64_t)blockIdx.x * K31_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < K31_HIDDEN; index += K31_THREADS )
	{
		float value = row[index];
		if ( add != 0 )
		{
			value += LmBf16ToFloat(__ldg(add + index));
			row[index] = value;
		}
		total += value * value;
	}
	total = LmBlockSum<K31_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)K31_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < K31_HIDDEN; index += K31_THREADS )
		out[index] = LmFloatToBf16(row[index] * scale * LmBf16ToFloat(__ldg(weight + index)));
	LmDependentRelease();
}

__global__ __launch_bounds__(64u, 4)
void K31QueryKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ kv_b, const float *__restrict__ inverse, float gain,
	const uint32_t *__restrict__ positions, uint16_t *__restrict__ queries, float *__restrict__ queries_f32)
{
	__shared__ float nope[K31_BLOCK][K31_NOPE];
	const uint32_t head = blockIdx.x >> 2u, part = blockIdx.x & 3u;
	const uint32_t col0 = part * 128u + 2u * threadIdx.x;
	float acc0[K31_BLOCK], acc1[K31_BLOCK];
	for ( uint32_t index = threadIdx.x; index < K31_BLOCK * K31_NOPE; index += 64u )
		nope[index / K31_NOPE][index % K31_NOPE] = LmBf16ToFloat(__ldg(q + (uint64_t)(index / K31_NOPE) * K31_HEADS * K31_QK + head * K31_QK + index % K31_NOPE));
#pragma unroll
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
	{
		acc0[row] = 0.0f;
		acc1[row] = 0.0f;
	}
	__syncthreads();
	const uint16_t *base_w = kv_b + (uint64_t)head * (K31_NOPE + K31_VALUE) * K31_KV_RANK + col0;
	uint32_t next_pair = __ldg((const uint32_t *)base_w);
	for ( uint32_t n = 0u; n < K31_NOPE; n++ )
	{
		const uint32_t pair = next_pair;
		if ( n + 1u < K31_NOPE )
			next_pair = __ldg((const uint32_t *)(base_w + (uint64_t)(n + 1u) * K31_KV_RANK));
		const float w0 = __uint_as_float(pair << 16u);
		const float w1 = __uint_as_float(pair & 0xffff0000u);
#pragma unroll
		for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
		{
			const float x = nope[row][n];
			acc0[row] += x * w0;
			acc1[row] += x * w1;
		}
	}
#pragma unroll
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
	{
		const uint64_t at = ((uint64_t)row * K31_HEADS + head) * K31_LATENT + col0;
		*(float2 *)(queries_f32 + at) = make_float2(acc0[row], acc1[row]);
		*(uint32_t *)(queries + at) = (uint32_t)LmFloatToBf16(acc0[row]) | ((uint32_t)LmFloatToBf16(acc1[row]) << 16u);
	}
	{
		const uint32_t idx = part * 64u + threadIdx.x;
		const uint32_t row = idx / (K31_ROPE / 2u), pair = idx % (K31_ROPE / 2u);
		const uint16_t *rope = q + (uint64_t)row * K31_HEADS * K31_QK + head * K31_QK + K31_NOPE;
		float cosine, sine;
		K31Rotation(positions[row], inverse[pair], gain, &cosine, &sine);
		const float even = LmBf16ToFloat(__ldg(rope + 2u * pair)), odd = LmBf16ToFloat(__ldg(rope + 2u * pair + 1u));
		const float first = even * cosine - odd * sine, second = even * sine + odd * cosine;
		const uint64_t at = ((uint64_t)row * K31_HEADS + head) * K31_LATENT + K31_KV_RANK + 2u * pair;
		*(float2 *)(queries_f32 + at) = make_float2(first, second);
		*(uint32_t *)(queries + at) = (uint32_t)LmFloatToBf16(first) | ((uint32_t)LmFloatToBf16(second) << 16u);
	}
	LmDependentRelease();
}

static __device__ __forceinline__ uint32_t K31Swizzle(uint32_t row, uint32_t chunk)
{
	return chunk ^ (row & 7u);
}

static __device__ __forceinline__ void K31LoadX4(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void K31LoadX4Trans(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void K31Stage(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ const uint16_t *K31KeyRow(const uint16_t *const *__restrict__ pages, uint64_t layer_offset,
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
	return pages[position / K31_PAGE_POSITIONS] + layer_offset + (uint64_t)(position % K31_PAGE_POSITIONS) * K31_LATENT;
}

__global__ __launch_bounds__(K31_ATTN_THREADS, 1)
void K31ContextAttentionKernel(const uint16_t *__restrict__ queries, const uint16_t *const *__restrict__ pages, uint64_t layer_offset,
	const K31Draft *__restrict__ draft, float scale, float *__restrict__ partials, float *__restrict__ stats)
{
	LmDependentWait();
	const uint32_t *run_first = draft->run_first;
	const uint32_t *run_position = draft->run_position;
	const uint32_t runs = draft->runs, held = draft->held, split_rows = draft->split_rows;
	constexpr uint32_t CHUNKS = K31_LATENT / K31_CHUNK_ELEMENTS;
	constexpr uint32_t TILE = K31_ATTN_TILE;
	constexpr uint32_t QUERIES = K31_ATTN_QUERIES;
	constexpr uint32_t QUERY_TILES = QUERIES / 16u;
	constexpr uint32_t KEY_FRAGS = 2u;
	constexpr uint32_t KEY_GROUPS = TILE / 8u / KEY_FRAGS;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / K31_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	constexpr uint32_t HALF_STEPS = K31_LATENT / 32u;
	constexpr uint32_t WARP_COLUMNS = K31_KV_RANK / (K31_ATTN_THREADS / K31_WARP);
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	static_assert(QUERY_TILES * 2u * KEY_GROUPS == K31_ATTN_THREADS / K31_WARP, "each warp owns one query tile, one k-half and one key group");
	static_assert(K31_QUERIES % QUERIES == 0u && K31_LATENT % 32u == 0u && COLUMN_FRAGS % 2u == 0u, "the tiles cover the queries and the row");
	extern __shared__ __align__(128) uint8_t k31_attention_shared[];
	uint8_t *keys = k31_attention_shared;
	float *score_parts = (float *)(keys + 2u * TILE * K31_ROW_BYTES);
	uint16_t *weights = (uint16_t *)(score_parts + 2u * QUERIES * K31_ATTN_STRIDE);
	float *running_max = (float *)(weights + QUERIES * K31_ATTN_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint16_t **slots = (const uint16_t **)(rescale + QUERIES);
	const uint32_t warp = threadIdx.x / K31_WARP, lane = threadIdx.x % K31_WARP, matrix = lane >> 3u, within = lane & 7u;
	const uint32_t query_tile = warp / (2u * KEY_GROUPS), half = (warp / KEY_GROUPS) & 1u, key_group = warp % KEY_GROUPS;
	const uint32_t first_query = blockIdx.x * QUERIES, split = blockIdx.y;
	const uint32_t begin = split * split_rows, end = begin + split_rows < held ? begin + split_rows : held;
	const uint32_t tiles = (end - begin + TILE - 1u) / TILE;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys), query_base = key_base + TILE * K31_ROW_BYTES;
	uint32_t query_frag[HALF_STEPS][4];
	float output[QUERY_TILES][COLUMN_FRAGS][4];

	auto stage = [&](uint32_t which)
	{
		const uint32_t buffer = which & 1u;
		for ( uint32_t item = threadIdx.x; item < TILE * CHUNKS; item += K31_ATTN_THREADS )
		{
			const uint32_t slot_index = item / CHUNKS, piece = item % CHUNKS;
			const uint16_t *slot = slots[buffer * TILE + slot_index];
			const void *source = slot != 0 ? (const void *)(slot + piece * K31_CHUNK_ELEMENTS) : (const void *)queries;
			const uint32_t destination = key_base + (buffer * TILE + slot_index) * K31_ROW_BYTES + K31Swizzle(slot_index, piece) * K31_CHUNK_BYTES;
			K31Stage(destination, source, slot != 0 ? K31_CHUNK_BYTES : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	for ( uint32_t index = threadIdx.x; index < QUERIES * CHUNKS; index += K31_ATTN_THREADS )
	{
		const uint32_t q = index / CHUNKS, chunk = index % CHUNKS;
		const uint16_t *source = queries + (uint64_t)(first_query + q) * K31_LATENT + chunk * K31_CHUNK_ELEMENTS;
		K31Stage(query_base + q * K31_ROW_BYTES + K31Swizzle(q, chunk) * K31_CHUNK_BYTES, source, K31_CHUNK_BYTES);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	if ( threadIdx.x < TILE )
	{
		slots[threadIdx.x] = K31KeyRow(pages, layer_offset, run_first, run_position, runs, begin + threadIdx.x, end);
		slots[TILE + threadIdx.x] = K31KeyRow(pages, layer_offset, run_first, run_position, runs, begin + TILE + threadIdx.x, end);
	}
	__syncthreads();
	stage(0u);
	asm volatile("cp.async.wait_group 1;\n" ::: "memory");
	__syncthreads();
	{
		const uint32_t a_row = query_tile * 16u + (matrix & 1u) * 8u + within;
#pragma unroll
		for ( uint32_t step = 0u; step < HALF_STEPS; step++ )
			K31LoadX4(query_base + a_row * K31_ROW_BYTES + K31Swizzle(a_row, (half * HALF_STEPS + step) * 2u + (matrix >> 1u)) * K31_CHUNK_BYTES,
				query_frag[step]);
	}
	if ( threadIdx.x < QUERIES )
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++ )
			output[m][frag][0] = output[m][frag][1] = output[m][frag][2] = output[m][frag][3] = 0.0f;
	__syncthreads();

	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t buffer = tile & 1u, tile_base = key_base + buffer * TILE * K31_ROW_BYTES, tile_first = begin + tile * TILE;
		if ( tile + 1u < tiles )
		{
			stage(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		{
			float score[KEY_FRAGS][4];
			uint32_t b[4], first[2], second[2];
			const uint32_t b_row = (key_group * KEY_FRAGS + (matrix >> 1u)) * 8u + within;
#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
				score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
#pragma unroll
			for ( uint32_t step = 0u; step < HALF_STEPS; step++ )
			{
				const uint32_t b_chunk = (half * HALF_STEPS + step) * 2u + (matrix & 1u);
				K31LoadX4(tile_base + b_row * K31_ROW_BYTES + K31Swizzle(b_row, b_chunk) * K31_CHUNK_BYTES, b);
				first[0] = b[0];
				first[1] = b[1];
				second[0] = b[2];
				second[1] = b[3];
				LmMmaBf16(score[0], query_frag[step], first);
				LmMmaBf16(score[1], query_frag[step], second);
			}
#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
#pragma unroll
				for ( uint32_t index = 0u; index < 4u; index++ )
					score_parts[(half * QUERIES + query_tile * 16u + (lane >> 2u) + (index >> 1u) * 8u) * K31_ATTN_STRIDE +
						(key_group * KEY_FRAGS + frag) * 8u + (lane & 3u) * 2u + (index & 1u)] = score[frag][index];
		}
		__syncthreads();
		{
			const uint32_t own = threadIdx.x / QUERY_THREADS, part = threadIdx.x % QUERY_THREADS;
			float value[QUERY_KEYS], peak = -INFINITY, total = 0.0f, previous, next, carry;
#pragma unroll
			for ( uint32_t index = 0u; index < QUERY_KEYS; index++ )
			{
				const uint32_t key = part + index * QUERY_THREADS;
				value[index] = tile_first + key < end
					? (score_parts[own * K31_ATTN_STRIDE + key] + score_parts[(QUERIES + own) * K31_ATTN_STRIDE + key]) * scale : -INFINITY;
				peak = fmaxf(peak, value[index]);
			}
#pragma unroll
			for ( uint32_t shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
			previous = running_max[own];
			next = fmaxf(previous, peak);
			carry = previous == -INFINITY ? 0.0f : exp2f(previous - next);
#pragma unroll
			for ( uint32_t index = 0u; index < QUERY_KEYS; index++ )
			{
				const float weight = value[index] == -INFINITY ? 0.0f : exp2f(value[index] - next);
				weights[own * K31_ATTN_STRIDE + part + index * QUERY_THREADS] = LmFloatToBf16(weight);
				total += weight;
			}
#pragma unroll
			for ( uint32_t shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u )
				total += __shfl_xor_sync(0xffffffffu, total, shift);
			__syncwarp();
			if ( part == 0u )
			{
				running_sum[own] = running_sum[own] * carry + total;
				running_max[own] = next;
				rescale[own] = carry;
			}
		}
		__syncthreads();
		{
			uint32_t a[QUERY_TILES][4], b[4], first[2], second[2];
#pragma unroll
			for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
			{
				const float low = rescale[m * 16u + (lane >> 2u)], high = rescale[m * 16u + (lane >> 2u) + 8u];
#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++ )
				{
					output[m][frag][0] *= low;
					output[m][frag][1] *= low;
					output[m][frag][2] *= high;
					output[m][frag][3] *= high;
				}
			}
#pragma unroll
			for ( uint32_t step = 0u; step < TILE / 16u; step++ )
			{
#pragma unroll
				for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
				{
					const uint32_t a_row = m * 16u + (matrix & 1u) * 8u + within, a_column = step * 16u + (matrix >> 1u) * 8u;
					K31LoadX4((uint32_t)__cvta_generic_to_shared(weights + a_row * K31_ATTN_STRIDE + a_column), a[m]);
				}
#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag += 2u )
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
					const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
					K31LoadX4Trans(tile_base + b_row * K31_ROW_BYTES + K31Swizzle(b_row, b_chunk) * K31_CHUNK_BYTES, b);
					first[0] = b[0];
					first[1] = b[1];
					second[0] = b[2];
					second[1] = b[3];
#pragma unroll
					for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
					{
						LmMmaBf16(output[m][frag], a[m], first);
						LmMmaBf16(output[m][frag + 1u], a[m], second);
					}
				}
			}
		}
		if ( threadIdx.x < TILE && tile + 2u < tiles )
			slots[buffer * TILE + threadIdx.x] = K31KeyRow(pages, layer_offset, run_first, run_position, runs, tile_first + 2u * TILE + threadIdx.x, end);
		__syncthreads();
	}
#pragma unroll
	for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++ )
#pragma unroll
			for ( uint32_t index = 0u; index < 4u; index += 2u )
			{
				const uint32_t q = m * 16u + (lane >> 2u) + (index >> 1u) * 8u;
				const uint32_t column = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
				*(float2 *)(partials + ((uint64_t)split * K31_QUERIES + first_query + q) * K31_KV_RANK + column) =
					make_float2(output[m][frag][index], output[m][frag][index + 1u]);
			}
	if ( threadIdx.x < QUERIES )
	{
		stats[((uint64_t)split * K31_QUERIES + first_query + threadIdx.x) * 2u] = running_max[threadIdx.x];
		stats[((uint64_t)split * K31_QUERIES + first_query + threadIdx.x) * 2u + 1u] = running_sum[threadIdx.x];
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_MERGE_THREADS, 2)
void K31AttentionMergeKernel(const float *__restrict__ partials, const float *__restrict__ stats, const K31Draft *__restrict__ draft,
	const float *__restrict__ queries, const uint16_t *__restrict__ block_rows, float scale, float *__restrict__ latent)
{
	LmDependentWait();
	constexpr uint32_t PER = K31_KV_RANK / K31_MERGE_THREADS;
	__shared__ float reduction[K31_MERGE_THREADS / K31_WARP];
	const uint32_t splits = draft->splits;
	const uint32_t query = blockIdx.x, row = query / K31_HEADS;
	const float *own = queries + (uint64_t)query * K31_LATENT;
	float accumulated[PER], scores[K31_BLOCK], best = -INFINITY, total = 0.0f, peak = -INFINITY, next, keep, denominator;
#pragma unroll
	for ( uint32_t column = 0u; column < PER; column++ )
		accumulated[column] = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
		best = fmaxf(best, stats[((uint64_t)split * K31_QUERIES + query) * 2u]);
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part = stats + ((uint64_t)split * K31_QUERIES + query) * 2u;
		const float weight = exp2f(part[0] - best);
		total += weight * part[1];
#pragma unroll
		for ( uint32_t column = 0u; column < PER; column++ )
			accumulated[column] += weight * partials[((uint64_t)split * K31_QUERIES + query) * K31_KV_RANK + column * K31_MERGE_THREADS + threadIdx.x];
	}
#pragma unroll
	for ( uint32_t key = 0u; key < K31_BLOCK; key++ )
	{
		float dot = 0.0f;
		scores[key] = -INFINITY;
		if ( key > row )
			continue;
		for ( uint32_t element = threadIdx.x; element < K31_LATENT; element += K31_MERGE_THREADS )
			dot += own[element] * LmBf16ToFloat(__ldg(block_rows + (uint64_t)key * K31_LATENT + element));
		scores[key] = LmBlockSum<K31_MERGE_THREADS>(dot, reduction) * scale;
		peak = fmaxf(peak, scores[key]);
	}
	next = fmaxf(best, peak);
	keep = exp2f(best - next);
	denominator = total * keep;
#pragma unroll
	for ( uint32_t column = 0u; column < PER; column++ )
		accumulated[column] *= keep;
#pragma unroll
	for ( uint32_t key = 0u; key < K31_BLOCK; key++ )
	{
		if ( key > row )
			continue;
		const float weight = exp2f(scores[key] - next);
		denominator += weight;
#pragma unroll
		for ( uint32_t column = 0u; column < PER; column++ )
			accumulated[column] += weight * LmBf16ToFloat(__ldg(block_rows + (uint64_t)key * K31_LATENT + column * K31_MERGE_THREADS + threadIdx.x));
	}
#pragma unroll
	for ( uint32_t column = 0u; column < PER; column++ )
		latent[(uint64_t)query * K31_KV_RANK + column * K31_MERGE_THREADS + threadIdx.x] = accumulated[column] / denominator;
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 2)
void K31ValueKernel(const float *__restrict__ latent, const uint16_t *__restrict__ kv_b, uint16_t *__restrict__ values)
{
	LmDependentWait();
	__shared__ float rows[K31_BLOCK][K31_KV_RANK];
	const uint32_t head = blockIdx.x >> 2u, part = blockIdx.x & 3u;
	const uint32_t warp = threadIdx.x / K31_WARP, lane = threadIdx.x % K31_WARP;
	for ( uint32_t index = threadIdx.x; index < K31_BLOCK * (K31_KV_RANK / 4u); index += 128u )
	{
		const uint32_t r = index / (K31_KV_RANK / 4u), c = (index % (K31_KV_RANK / 4u)) * 4u;
		*(float4 *)(rows[r] + c) = __ldg((const float4 *)(latent + ((uint64_t)r * K31_HEADS + head) * K31_KV_RANK + c));
	}
	__syncthreads();
	const uint32_t val_begin = part * (K31_VALUE / 4u), val_end = val_begin + (K31_VALUE / 4u);
	for ( uint32_t value = val_begin + warp; value < val_end; value += 4u )
	{
		const uint16_t *weight = kv_b + ((uint64_t)head * (K31_NOPE + K31_VALUE) + K31_NOPE + value) * K31_KV_RANK;
		float sum[K31_BLOCK];
#pragma unroll
		for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
			sum[row] = 0.0f;
		for ( uint32_t element = lane; element < K31_KV_RANK; element += K31_WARP )
		{
			const float w = LmBf16ToFloat(__ldg(weight + element));
#pragma unroll
			for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
				sum[row] += rows[row][element] * w;
		}
#pragma unroll
		for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
#pragma unroll
			for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
				sum[row] += __shfl_xor_sync(0xffffffffu, sum[row], mask);
		if ( lane == 0u )
#pragma unroll
			for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
				values[(uint64_t)row * K31_HEADS * K31_VALUE + head * K31_VALUE + value] = LmFloatToBf16(sum[row]);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_THREADS, 2)
void K31SwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * K31_THREADS + threadIdx.x; index < count; index += gridDim.x * K31_THREADS )
	{
		const float g = LmBf16ToFloat(__ldg(gate + index));
		output[index] = LmFloatToBf16(g / (1.0f + expf(-g)) * LmBf16ToFloat(__ldg(up + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_THREADS, 2)
void K31MarkovRowKernel(uint32_t step, const K31Draft *__restrict__ draft, const uint32_t *__restrict__ tokens, const uint32_t *__restrict__ candidates,
	const uint16_t *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ row)
{
	LmDependentWait();
	__shared__ float vector[K31_MARKOV];
	const uint32_t previous = step == 0u ? draft->anchor : tokens[step - 1u];
	const uint32_t warp = threadIdx.x / K31_WARP, lane = threadIdx.x % K31_WARP;
	for ( uint32_t index = threadIdx.x; index < K31_MARKOV; index += K31_THREADS )
		vector[index] = LmBf16ToFloat(__ldg(markov_previous + (uint64_t)previous * K31_MARKOV + index));
	__syncthreads();
	for ( uint32_t slot = blockIdx.x * (K31_THREADS / K31_WARP) + warp; slot < K31_CANDIDATES; slot += gridDim.x * (K31_THREADS / K31_WARP) )
	{
		const uint32_t candidate = candidates[step * K31_CANDIDATES + slot];
		const uint16_t *edge = markov_next + (uint64_t)candidate * K31_MARKOV;
		float sum = 0.0f;
		for ( uint32_t element = lane; element < K31_MARKOV; element += K31_WARP )
			sum += LmBf16ToFloat(__ldg(edge + element)) * vector[element];
		for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
			sum += __shfl_xor_sync(0xffffffffu, sum, mask);
		if ( lane == 0u )
			row[slot] = LmBf16ToFloat(__ldg(logits + (uint64_t)step * K31_VOCAB + candidate)) + sum;
	}
	LmDependentRelease();
}

static __device__ __forceinline__ unsigned long long K31RowKey(const float *row, uint32_t slot)
{
	return ((unsigned long long)SparkMathSampleOrdered(row[slot]) << 32u) | (unsigned long long)(0xffffffffu - slot);
}

static __device__ __forceinline__ uint32_t K31KeySlot(unsigned long long key)
{
	return 0xffffffffu - (uint32_t)key;
}

static __device__ __forceinline__ void K31Top2Insert(unsigned long long candidate, unsigned long long *first, unsigned long long *second)
{
	if ( candidate > *first )
	{
		*second = *first;
		*first = candidate;
	}
	else if ( candidate > *second && candidate != *first )
		*second = candidate;
}

__global__ __launch_bounds__(K31_PICK_THREADS, 1)
void K31MarkovPickKernel(uint32_t step, const K31Draft *__restrict__ draft, const float *__restrict__ row, const uint32_t *__restrict__ candidates,
	const uint16_t *__restrict__ y, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ confidence_weight,
	const uint16_t *__restrict__ confidence_bias, K31Picks *__restrict__ picks)
{
	LmDependentWait();
	__shared__ unsigned long long keys[K31_CANDIDATES];
	__shared__ unsigned long long best[K31_PICK_THREADS / K31_WARP];
	__shared__ unsigned long long second_best[K31_PICK_THREADS / K31_WARP];
	__shared__ float reduction[K31_PICK_THREADS / K31_WARP];
	__shared__ uint32_t chosen;
	__shared__ uint32_t alt_chosen;
	__shared__ float peak_scaled;
	const uint32_t anchor = draft->anchor, position = draft->position, top_k = draft->top_k;
	const unsigned long long seed = (unsigned long long)draft->seed;
	const float inverse_temperature = draft->inverse_temperature, nucleus = draft->nucleus, scale = draft->scale;
	const uint32_t previous = step == 0u ? anchor : picks->tokens[step - 1u];
	const uint32_t warp = threadIdx.x / K31_WARP, lane = threadIdx.x % K31_WARP;
	float dot = 0.0f, denom = 0.0f;
	if ( inverse_temperature > 0.0f )
	{
		for ( uint32_t index = threadIdx.x; index < K31_CANDIDATES; index += K31_PICK_THREADS )
			keys[index] = K31RowKey(row, index);
		__syncthreads();
		for ( uint32_t size = 2u; size <= K31_CANDIDATES; size <<= 1u )
			for ( uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u )
			{
				for ( uint32_t index = threadIdx.x; index < K31_CANDIDATES; index += K31_PICK_THREADS )
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
			const uint32_t slot = K31KeySlot(keys[lane]);
			const uint32_t token = candidates[step * K31_CANDIDATES + slot];
			const float scaled = row[slot] * inverse_temperature;
			float peak = scaled, total, probability, running, keyed;
			uint32_t winner, runner;
			unsigned long long key;
			for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, mask));
			probability = expf(scaled - peak);
			total = probability;
			for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
				total += __shfl_xor_sync(0xffffffffu, total, mask);
			probability /= total;
			running = probability;
			for ( uint32_t offset = 1u; offset < K31_SAMPLED; offset <<= 1u )
			{
				const float earlier = __shfl_up_sync(0xffffffffu, running, offset);
				if ( lane >= offset )
					running += earlier;
			}
			keyed = (((nucleus <= 0.0f) || (running - probability < nucleus)) && (top_k == 0u || lane < top_k)) || lane == 0u
				? scaled + SparkMathGumbelNoise(seed, position + step, token) : -INFINITY;
			key = ((unsigned long long)SparkMathSampleOrdered(keyed) << 32u) | (unsigned long long)(0xffffffffu - lane);
			for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
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
		for ( uint32_t index = threadIdx.x; index < K31_CANDIDATES; index += K31_PICK_THREADS )
			K31Top2Insert(K31RowKey(row, index), &first, &second);
		for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
		{
			const unsigned long long other_first = __shfl_xor_sync(0xffffffffu, first, mask);
			const unsigned long long other_second = __shfl_xor_sync(0xffffffffu, second, mask);
			K31Top2Insert(other_first, &first, &second);
			K31Top2Insert(other_second, &first, &second);
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
			for ( uint32_t index = 1u; index < K31_PICK_THREADS / K31_WARP; index++ )
			{
				K31Top2Insert(best[index], &first, &second);
				K31Top2Insert(second_best[index], &first, &second);
			}
			chosen = K31KeySlot(first);
			alt_chosen = K31KeySlot(second);
			peak_scaled = row[chosen];
		}
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < K31_CANDIDATES; index += K31_PICK_THREADS )
		denom += expf(row[index] * scale - peak_scaled);
	denom = LmBlockSum<K31_PICK_THREADS>(denom, reduction);
	for ( uint32_t index = threadIdx.x; index < K31_CONFIDENCE; index += K31_PICK_THREADS )
	{
		const float feature = index < K31_HIDDEN ? LmBf16ToFloat(__ldg(y + (uint64_t)step * K31_HIDDEN + index))
			: LmBf16ToFloat(__ldg(markov_previous + (uint64_t)previous * K31_MARKOV + index - K31_HIDDEN));
		dot += LmBf16ToFloat(__ldg(confidence_weight + index)) * feature;
	}
	dot = LmBlockSum<K31_PICK_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
	{
		picks->tokens[step] = candidates[step * K31_CANDIDATES + chosen];
		picks->scores[step] = 1.0f / (1.0f + expf(-(dot + LmBf16ToFloat(confidence_bias[0]))));
		picks->alt_tokens[step] = candidates[step * K31_CANDIDATES + alt_chosen];
		picks->alt_scores[step] = denom > 0.0f ? expf(row[alt_chosen] * scale - peak_scaled) / denom : 0.0f;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_WARP, 1)
void K31BranchSelectKernel(K31Picks *__restrict__ picks, const K31Draft *__restrict__ draft, float min_score)
{
	LmDependentWait();
	const uint32_t drafts = draft->drafts;
	uint32_t used = 0u;
	if ( threadIdx.x == 0u )
	{
		for ( uint32_t branch = 0u; branch < K31_BRANCHES; branch++ )
		{
			picks->branch_fork[branch] = K31_BLOCK;
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
				if ( pos < K31_BRANCHES )
				{
					picks->branch_fork[pos] = picks->branch_fork[pos - 1u];
					picks->branch_len[pos] = picks->branch_len[pos - 1u];
					picks->branch_tokens[pos][0] = picks->branch_tokens[pos - 1u][0];
					picks->branch_scores[pos][0] = picks->branch_scores[pos - 1u][0];
				}
				pos--;
			}
			if ( pos < K31_BRANCHES )
			{
				picks->branch_fork[pos] = step;
				picks->branch_len[pos] = 1u;
				picks->branch_tokens[pos][0] = picks->alt_tokens[step];
				picks->branch_scores[pos][0] = score;
				if ( used < K31_BRANCHES )
					used++;
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_THREADS, 2)
void K31BranchRowKernel(uint32_t offset, const K31Draft *__restrict__ draft, const K31Picks *__restrict__ picks, const uint32_t *__restrict__ candidates,
	const uint16_t *__restrict__ logits, const uint16_t *__restrict__ markov_previous, const uint16_t *__restrict__ markov_next, float *__restrict__ rows)
{
	LmDependentWait();
	__shared__ float vector[K31_MARKOV];
	const uint32_t drafts = draft->drafts;
	const uint32_t branch = blockIdx.y, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / K31_WARP, lane = threadIdx.x % K31_WARP;
	if ( fork < drafts && step < drafts )
	{
		const uint32_t previous = picks->branch_tokens[branch][offset - 1u];
		float *row = rows + (uint64_t)branch * K31_CANDIDATES;
		for ( uint32_t index = threadIdx.x; index < K31_MARKOV; index += K31_THREADS )
			vector[index] = LmBf16ToFloat(__ldg(markov_previous + (uint64_t)previous * K31_MARKOV + index));
		__syncthreads();
		for ( uint32_t slot = blockIdx.x * (K31_THREADS / K31_WARP) + warp; slot < K31_CANDIDATES; slot += gridDim.x * (K31_THREADS / K31_WARP) )
		{
			const uint32_t candidate = candidates[step * K31_CANDIDATES + slot];
			const uint16_t *edge = markov_next + (uint64_t)candidate * K31_MARKOV;
			float sum = 0.0f;
			for ( uint32_t element = lane; element < K31_MARKOV; element += K31_WARP )
				sum += LmBf16ToFloat(__ldg(edge + element)) * vector[element];
			for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
				sum += __shfl_xor_sync(0xffffffffu, sum, mask);
			if ( lane == 0u )
				row[slot] = LmBf16ToFloat(__ldg(logits + (uint64_t)step * K31_VOCAB + candidate)) + sum;
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_PICK_THREADS, 1)
void K31BranchPickKernel(uint32_t offset, const K31Draft *__restrict__ draft, const float *__restrict__ rows, const uint32_t *__restrict__ candidates,
	K31Picks *__restrict__ picks)
{
	LmDependentWait();
	__shared__ unsigned long long best[K31_PICK_THREADS / K31_WARP];
	__shared__ float reduction[K31_PICK_THREADS / K31_WARP];
	__shared__ uint32_t chosen;
	__shared__ float peak_scaled;
	const uint32_t drafts = draft->drafts;
	const float scale = draft->scale;
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	const uint32_t warp = threadIdx.x / K31_WARP, lane = threadIdx.x % K31_WARP;
	if ( fork < drafts && step < drafts )
	{
		const float *row = rows + (uint64_t)branch * K31_CANDIDATES;
		unsigned long long first = 0ull;
		float denom = 0.0f;
		for ( uint32_t index = threadIdx.x; index < K31_CANDIDATES; index += K31_PICK_THREADS )
		{
			const unsigned long long key = K31RowKey(row, index);
			if ( key > first )
				first = key;
		}
		for ( uint32_t mask = K31_WARP / 2u; mask > 0u; mask >>= 1u )
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
			for ( uint32_t index = 1u; index < K31_PICK_THREADS / K31_WARP; index++ )
				if ( best[index] > first )
					first = best[index];
			chosen = K31KeySlot(first);
			peak_scaled = row[chosen] * scale;
		}
		__syncthreads();
		for ( uint32_t index = threadIdx.x; index < K31_CANDIDATES; index += K31_PICK_THREADS )
			denom += expf(row[index] * scale - peak_scaled);
		denom = LmBlockSum<K31_PICK_THREADS>(denom, reduction);
		if ( threadIdx.x == 0u )
		{
			picks->branch_tokens[branch][offset] = candidates[step * K31_CANDIDATES + chosen];
			picks->branch_scores[branch][offset] = denom > 0.0f ? 1.0f / denom : 0.0f;
			picks->branch_len[branch] = offset + 1u;
		}
	}
	LmDependentRelease();
}

static SparkStatus K31DeviceOpen(void)
{
	const SparkStatus status = SparkDeviceInitialize(0);
	return status == SPARK_STATUS_DUPLICATE ? SPARK_STATUS_OK : status;
}

static SparkStatus K31Refuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	return status;
}

static SparkStatus K31LoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
{
	uint8_t header[16];
	uint32_t magic, version;
	uint64_t bytes;
	char *text;
	SparkStatus status;
	FILE *file = fopen(pack_path, "rb");
	if ( file == 0 )
		return SPARK_STATUS_NOT_FOUND;
	if ( fread(header, 1u, sizeof(header), file) != sizeof(header) )
	{
		(void)fclose(file);
		return SPARK_STATUS_IO_ERROR;
	}
	memcpy(&magic, header, sizeof(magic));
	memcpy(&version, header + sizeof(magic), sizeof(version));
	memcpy(&bytes, header + sizeof(magic) + sizeof(version), sizeof(bytes));
	if ( magic != K31_PACK_MAGIC || version != K31_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
	{
		(void)fclose(file);
		return SPARK_STATUS_VALIDATION_FAILED;
	}
	text = (char *)malloc((size_t)bytes);
	if ( text == 0 || fread(text, 1u, (size_t)bytes, file) != (size_t)bytes )
	{
		free(text);
		(void)fclose(file);
		return SPARK_STATUS_IO_ERROR;
	}
	(void)fclose(file);
	status = SparkJsonParseText(text, (size_t)bytes, document);
	free(text);
	*payload_base = (sizeof(header) + bytes + K31_PACK_ALIGNMENT - 1u) / K31_PACK_ALIGNMENT * K31_PACK_ALIGNMENT;
	return status;
}

static SparkStatus K31Tensor(const K31State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t elements,
	const uint16_t **pointer)
{
	const int32_t tensors = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "tensors");
	const int32_t entry = tensors < 0 ? -1 : SparkJsonFindObjectMember(document, tensors, name);
	const int32_t offset_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "offset");
	const int32_t bytes_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "bytes");
	uint64_t offset = 0u, bytes = 0u;
	const void *slice = 0;
	if ( offset_token < 0 || bytes_token < 0 || SparkJsonGetUInt64(document, offset_token, &offset) != SPARK_STATUS_OK ||
		SparkJsonGetUInt64(document, bytes_token, &bytes) != SPARK_STATUS_OK || bytes != elements * sizeof(uint16_t) )
	{
		fprintf(stderr, "k31 dspark: pack tensor %s is missing or is not %llu bf16 elements\n", name, (unsigned long long)elements);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	*pointer = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static float K31ConfigFloat(const SparkJsonDocument *document, int32_t object, const char *name, float fallback)
{
	const int32_t token = object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
	float value = fallback;
	if ( token >= 0 )
		(void)SparkJsonGetFloat(document, token, &value);
	return value;
}

static int32_t K31Checkpoint(const SparkJsonDocument *document)
{
	const int32_t config = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "config");
	return config < 0 ? -1 : SparkJsonFindObjectMember(document, config, "checkpoint_config");
}

static SparkStatus K31Shape(K31State *state, const SparkJsonDocument *document)
{
	static const struct
	{
		const char *name;
		uint64_t value;
	} expected[] = {
		{"hidden_size", K31_HIDDEN}, {"num_attention_heads", K31_HEADS}, {"q_lora_rank", K31_QUERY_RANK}, {"kv_lora_rank", K31_KV_RANK},
		{"qk_nope_head_dim", K31_NOPE}, {"qk_rope_head_dim", K31_ROPE}, {"v_head_dim", K31_VALUE}, {"intermediate_size", K31_MLP},
		{"num_hidden_layers", K31_LAYERS}, {"num_target_layers", K31_TAPS}, {"block_size", K31_BLOCK}, {"vocab_size", K31_VOCAB},
		{"draft_vocab_size", K31_VOCAB}, {"markov_rank", K31_MARKOV},
	};
	const int32_t checkpoint = K31Checkpoint(document);
	const int32_t mask = checkpoint < 0 ? -1 : SparkJsonFindObjectMember(document, checkpoint, "mask_token_id");
	uint64_t value = 0u;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		const int32_t token = checkpoint < 0 ? -1 : SparkJsonFindObjectMember(document, checkpoint, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "k31 dspark: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			return SPARK_STATUS_SCHEMA_ERROR;
		}
	}
	if ( mask < 0 || SparkJsonGetUInt64(document, mask, &value) != SPARK_STATUS_OK || value >= K31_VOCAB )
	{
		fprintf(stderr, "k31 dspark: checkpoint_config mask_token_id is missing or outside the vocabulary\n");
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	state->mask_token = (uint32_t)value;
	return SPARK_STATUS_OK;
}

static double K31YarnScale(double scale, double weight)
{
	return scale <= 1.0 || weight == 0.0 ? 1.0 : 0.1 * weight * log(scale) + 1.0;
}

static SparkStatus K31Rope(K31State *state, const SparkJsonDocument *document)
{
	const int32_t checkpoint = K31Checkpoint(document);
	const int32_t rope = checkpoint < 0 ? -1 : SparkJsonFindObjectMember(document, checkpoint, "rope_parameters");
	const double base = K31ConfigFloat(document, rope, "rope_theta", 10000.0f), factor = K31ConfigFloat(document, rope, "factor", 1.0f);
	const double original = K31ConfigFloat(document, rope, "original_max_position_embeddings", 4096.0f);
	const double fast = K31ConfigFloat(document, rope, "beta_fast", 32.0f), slow = K31ConfigFloat(document, rope, "beta_slow", 1.0f);
	const double mscale = K31ConfigFloat(document, rope, "mscale", 1.0f), mscale_all = K31ConfigFloat(document, rope, "mscale_all_dim", 0.0f);
	const double attention = K31ConfigFloat(document, rope, "attn_factor", 1.0f);
	const double dimension = (double)K31_ROPE;
	float host[K31_ROPE / 2u];
	double low, high;
	if ( rope < 0 || checkpoint < 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->epsilon = K31ConfigFloat(document, checkpoint, "rms_norm_eps", 1e-6f);
	low = floor(dimension * log(original / (fast * 2.0 * K31_PI)) / (2.0 * log(base)));
	high = ceil(dimension * log(original / (slow * 2.0 * K31_PI)) / (2.0 * log(base)));
	low = low < 0.0 ? 0.0 : low;
	high = high > dimension - 1.0 ? dimension - 1.0 : high;
	if ( low == high )
		high += 0.001;
	for ( uint32_t pair = 0u; pair < K31_ROPE / 2u; pair++ )
	{
		const double extrapolated = 1.0 / pow(base, (2.0 * pair) / dimension);
		double ramp = ((double)pair - low) / (high - low);
		ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
		host[pair] = (float)(extrapolated / factor * ramp + extrapolated * (1.0 - ramp));
	}
	state->rope_gain = (float)(K31YarnScale(factor, mscale) / K31YarnScale(factor, mscale_all) * attention);
	state->attention_scale = (float)(K31YarnScale(factor, mscale_all) * K31YarnScale(factor, mscale_all) / sqrt((double)K31_QK) * K31_LOG2E);
	if ( SparkGpuMalloc((void **)&state->rope_inverse, sizeof(host)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(state->rope_inverse, host, sizeof(host), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus K31LayerTensor(K31State *state, const SparkJsonDocument *document, uint64_t payload_base, uint32_t layer, const char *suffix,
	uint64_t elements, const uint16_t **pointer)
{
	char name[128];
	snprintf(name, sizeof(name), "layers.%u.%s", layer, suffix);
	return K31Tensor(state, document, payload_base, name, elements, pointer);
}

static SparkStatus K31Weights(K31State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = K31LoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = K31Shape(state, &document);
	if ( status == SPARK_STATUS_OK )
		status = K31Rope(state, &document);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "context_proj.weight", (uint64_t)K31_HIDDEN * K31_TAP_WIDTH, &state->context_proj);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "context_norm.weight", K31_HIDDEN, &state->context_norm);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "final_norm.weight", K31_HIDDEN, &state->final_norm);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "target.embed", (uint64_t)K31_VOCAB * K31_HIDDEN, &state->embed);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "target.head", (uint64_t)K31_VOCAB * K31_HIDDEN, &state->head);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "markov_head.markov_w1.weight", (uint64_t)K31_VOCAB * K31_MARKOV, &state->markov_previous);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "markov_head.markov_w2.weight", (uint64_t)K31_VOCAB * K31_MARKOV, &state->markov_next);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "confidence_head.proj.weight", K31_CONFIDENCE, &state->confidence_weight);
	if ( status == SPARK_STATUS_OK )
		status = K31Tensor(state, &document, payload_base, "confidence_head.proj.bias", 1u, &state->confidence_bias);
	for ( uint32_t layer = 0u; layer < K31_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		K31Layer *weights = &state->layers[layer];
		const struct
		{
			const char *suffix;
			uint64_t elements;
			const uint16_t **pointer;
		} tensors[] = {
			{"self_attn.kv_a_proj_with_mqa.weight", (uint64_t)K31_LATENT * K31_HIDDEN, &weights->kv_a},
			{"self_attn.kv_a_layernorm.weight", K31_KV_RANK, &weights->kv_a_norm},
			{"input_layernorm.weight", K31_HIDDEN, &weights->input_norm},
			{"post_attention_layernorm.weight", K31_HIDDEN, &weights->post_norm},
			{"self_attn.q_a_proj.weight", (uint64_t)K31_QUERY_RANK * K31_HIDDEN, &weights->q_a},
			{"self_attn.q_a_layernorm.weight", K31_QUERY_RANK, &weights->q_a_norm},
			{"self_attn.q_b_proj.weight", (uint64_t)K31_HEADS * K31_QK * K31_QUERY_RANK, &weights->q_b},
			{"self_attn.kv_b_proj.weight", (uint64_t)K31_HEADS * (K31_NOPE + K31_VALUE) * K31_KV_RANK, &weights->kv_b},
			{"self_attn.o_proj.weight", (uint64_t)K31_HIDDEN * K31_HEADS * K31_VALUE, &weights->o_proj},
			{"mlp.gate_proj.weight", (uint64_t)K31_MLP * K31_HIDDEN, &weights->gate},
			{"mlp.up_proj.weight", (uint64_t)K31_MLP * K31_HIDDEN, &weights->up},
			{"mlp.down_proj.weight", (uint64_t)K31_HIDDEN * K31_MLP, &weights->down},
		};
		for ( uint32_t index = 0u; index < sizeof(tensors) / sizeof(tensors[0]) && status == SPARK_STATUS_OK; index++ )
			status = K31LayerTensor(state, &document, payload_base, layer, tensors[index].suffix, tensors[index].elements, tensors[index].pointer);
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define K31_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus K31Scratch(K31State *state)
{
	K31Forward *forward = &state->forward;
	uint16_t *destinations[K31_BLOCK];
	int multiprocessors = 0;
	if ( SparkGpuStreamCreateWithFlags(&state->stream, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuFuncSetAttribute(K31ContextAttentionKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)K31_ATTN_SHARED_BYTES) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)K31_INGEST_ROWS * K31_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)K31_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(K31Draft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_pages, (size_t)K31_PAGES_MAX * sizeof(*forward->host_pages)) != SPARK_GPU_SUCCESS ||
		!K31_DEVICE(state->taps, (uint64_t)K31_INGEST_ROWS * K31_TAP_WIDTH) || !K31_DEVICE(state->features_raw, (uint64_t)K31_INGEST_ROWS * K31_HIDDEN) ||
		!K31_DEVICE(state->features, (uint64_t)K31_INGEST_ROWS * K31_HIDDEN) || !K31_DEVICE(state->latent_raw, (uint64_t)K31_INGEST_ROWS * K31_LATENT) ||
		!K31_DEVICE(state->positions, K31_INGEST_ROWS) || !K31_DEVICE(state->ingest_rows, (uint64_t)K31_LAYERS * K31_INGEST_ROWS) ||
		!K31_DEVICE(forward->draft, 1u) || !K31_DEVICE(forward->pages, K31_PAGES_MAX) ||
		!K31_DEVICE(forward->block_destinations, K31_BLOCK) || !K31_DEVICE(forward->residual, K31_BLOCK * K31_HIDDEN) ||
		!K31_DEVICE(forward->x, K31_BLOCK * K31_HIDDEN) || !K31_DEVICE(forward->q_a_raw, K31_BLOCK * K31_QUERY_RANK) ||
		!K31_DEVICE(forward->q_a, K31_BLOCK * K31_QUERY_RANK) || !K31_DEVICE(forward->q, K31_BLOCK * K31_HEADS * K31_QK) ||
		!K31_DEVICE(forward->latent_raw, K31_BLOCK * K31_LATENT) || !K31_DEVICE(forward->block_rows, K31_BLOCK * K31_LATENT) ||
		!K31_DEVICE(forward->queries, (uint64_t)K31_QUERIES * K31_LATENT) || !K31_DEVICE(forward->queries_f32, (uint64_t)K31_QUERIES * K31_LATENT) ||
		!K31_DEVICE(forward->partials, (uint64_t)K31_SPLITS_MAX * K31_QUERIES * K31_KV_RANK) ||
		!K31_DEVICE(forward->stats, (uint64_t)K31_SPLITS_MAX * K31_QUERIES * 2u) || !K31_DEVICE(forward->latent, (uint64_t)K31_QUERIES * K31_KV_RANK) ||
		!K31_DEVICE(forward->values, K31_BLOCK * K31_HEADS * K31_VALUE) || !K31_DEVICE(forward->attention, K31_BLOCK * K31_HIDDEN) ||
		!K31_DEVICE(forward->gate, K31_BLOCK * K31_MLP) || !K31_DEVICE(forward->up, K31_BLOCK * K31_MLP) || !K31_DEVICE(forward->act, K31_BLOCK * K31_MLP) ||
		!K31_DEVICE(forward->mlp, K31_BLOCK * K31_HIDDEN) || !K31_DEVICE(forward->y, K31_BLOCK * K31_HIDDEN) ||
		!K31_DEVICE(forward->logits, (uint64_t)K31_BLOCK * K31_VOCAB) || SparkDrafterTopkBf16Alloc(&forward->topk_ws, K31_BLOCK) != SPARK_STATUS_OK ||
		!K31_DEVICE(forward->candidates, K31_BLOCK * K31_CANDIDATES) || !K31_DEVICE(forward->markov_row, K31_BRANCHES * K31_CANDIDATES) ||
		!K31_DEVICE(forward->picks, 1u) ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(K31Picks)) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
		destinations[row] = forward->block_rows + (uint64_t)row * K31_LATENT;
	if ( SparkGpuMemcpy(forward->block_destinations, destinations, sizeof(destinations), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static void K31SequenceFree(K31State *state, K31Sequence *sequence)
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

static K31Sequence *K31SequenceFind(K31State *state, uint64_t id, uint32_t create)
{
	K31Sequence *empty = 0, *oldest = 0;
	for ( uint32_t index = 0u; index < K31_SEQUENCES_MAX; index++ )
	{
		K31Sequence *sequence = &state->sequences[index];
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
		K31SequenceFree(state, oldest);
		empty = oldest;
	}
	empty->pages = (uint16_t **)calloc(K31_PAGES_MAX, sizeof(uint16_t *));
	if ( empty->pages == 0 )
		return 0;
	empty->id = id;
	empty->last_used = ++state->clock;
	return empty;
}

static K31Sequence *K31SequenceOldestBut(K31State *state, const K31Sequence *keep, const K31Sequence *donor)
{
	K31Sequence *oldest = 0;
	for ( uint32_t index = 0u; index < K31_SEQUENCES_MAX; index++ )
	{
		K31Sequence *sequence = &state->sequences[index];
		if ( sequence->pages != 0 && sequence != keep && sequence != donor && (oldest == 0 || sequence->last_used < oldest->last_used) )
			oldest = sequence;
	}
	return oldest;
}

static SparkStatus K31SequencePages(K31State *state, K31Sequence *sequence, uint64_t end, const K31Sequence *donor)
{
	const uint64_t needed = (end + K31_PAGE_POSITIONS - 1u) / K31_PAGE_POSITIONS;
	const uint64_t had = (uint64_t)sequence->page_count * K31_PAGE_POSITIONS;
	uint8_t *held;
	uint32_t *tokens;
	if ( needed > K31_PAGES_MAX )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( needed <= sequence->page_count )
		return SPARK_STATUS_OK;
	held = (uint8_t *)realloc(sequence->held, (size_t)(needed * K31_PAGE_POSITIONS));
	if ( held == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	sequence->held = held;
	memset(held + had, 0, (size_t)(needed * K31_PAGE_POSITIONS - had));
	tokens = (uint32_t *)realloc(sequence->tokens, (size_t)(needed * K31_PAGE_POSITIONS) * sizeof(uint32_t));
	if ( tokens == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	sequence->tokens = tokens;
	memset(tokens + had, 0, (size_t)(needed * K31_PAGE_POSITIONS - had) * sizeof(uint32_t));
	for ( uint64_t page = sequence->page_count; page < needed; page++ )
	{
		while ( (state->pages_used + 1u) * K31_PAGE_BYTES > state->context_bytes )
		{
			K31Sequence *oldest = K31SequenceOldestBut(state, sequence, donor);
			if ( oldest == 0 )
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			K31SequenceFree(state, oldest);
		}
		if ( SparkGpuMalloc((void **)&sequence->pages[page], (size_t)K31_PAGE_BYTES) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->pages_used++;
		sequence->page_count = (uint32_t)page + 1u;
	}
	return SPARK_STATUS_OK;
}

static uint16_t *K31Row(const K31Sequence *sequence, uint32_t layer, uint64_t position)
{
	uint16_t *page = sequence->pages[position / K31_PAGE_POSITIONS];
	return page + ((uint64_t)layer * K31_PAGE_POSITIONS + position % K31_PAGE_POSITIONS) * K31_LATENT;
}

static SparkStatus K31WriteRows(K31State *state, K31Sequence *sequence, uint64_t first, uint32_t rows)
{
	uint16_t *host_rows[K31_LAYERS * K31_INGEST_ROWS];
	SparkStatus status = SPARK_STATUS_OK;
	for ( uint32_t layer = 0u; layer < K31_LAYERS; layer++ )
		for ( uint32_t row = 0u; row < rows; row++ )
			host_rows[(uint64_t)layer * rows + row] = K31Row(sequence, layer, first + row);
	if ( SparkGpuMemcpyAsync(state->ingest_rows, host_rows, (size_t)K31_LAYERS * rows * sizeof(uint16_t *), SPARK_GPU_COPY_HOST_TO_DEVICE,
		state->stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	if ( LmDenseRows<LmBf16Format>(state->context_proj, LmScaleTensorNone(), state->taps, state->features_raw, (float *)0, rows, K31_TAP_WIDTH,
		K31_HIDDEN, 0u, state->multiprocessors, state->stream) != LM_LAUNCH_OK )
		return SPARK_STATUS_INTERNAL_ERROR;
	LM_LAUNCH((K31RmsRowsKernel), rows, K31_THREADS, 0, state->stream, state->features_raw, state->context_norm, state->features, K31_HIDDEN,
		state->epsilon);
	for ( uint32_t layer = 0u; layer < K31_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		if ( LmDenseRows<LmBf16Format>(state->layers[layer].kv_a, LmScaleTensorNone(), state->features, state->latent_raw, (float *)0, rows, K31_HIDDEN,
			K31_LATENT, 0u, state->multiprocessors, state->stream) != LM_LAUNCH_OK )
			status = SPARK_STATUS_INTERNAL_ERROR;
		else
			LM_LAUNCH((K31LatentRowsKernel), rows, K31_THREADS, 0, state->stream, state->latent_raw, state->layers[layer].kv_a_norm,
				state->rope_inverse, state->rope_gain, state->positions, state->ingest_rows + (uint64_t)layer * rows, state->epsilon);
	}
	if ( status == SPARK_STATUS_OK && SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_IO_ERROR;
	return status;
}

static void K31Forget(K31Sequence *sequence, uint64_t first)
{
	const uint64_t capacity = (uint64_t)sequence->page_count * K31_PAGE_POSITIONS;
	if ( first < capacity )
		memset(sequence->held + first, 0, (size_t)(capacity - first));
	if ( sequence->held_end > first )
		sequence->held_end = first;
	if ( sequence->verified > first )
		sequence->verified = first;
}

static SparkStatus K31Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	K31State *state = (K31State *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t first = frame->header.first_position, end = first + rows;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	K31Sequence *sequence;
	SparkStatus status;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != K31_TAPS || hidden != K31_HIDDEN || frame->rows == 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	for ( uint32_t tap = 0u; tap < K31_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * K31_HIDDEN > frame->row_elements )
			return SPARK_STATUS_SCHEMA_ERROR;
	sequence = K31SequenceFind(state, frame->header.sequence_id, 0u);
	if ( sequence != 0 && first != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = K31SequenceFind(state, frame->header.sequence_id, 1u);
		if ( sequence == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		sequence->control_generation = frame->header.control_generation;
		sequence->request_id = frame->header.request_id;
	}
	status = K31SequencePages(state, sequence, end, 0);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( first == 0u )
	{
		K31Forget(sequence, 0u);
		sequence->control_generation = frame->header.control_generation;
		sequence->request_id = frame->header.request_id;
	}
	K31Forget(sequence, first);
	for ( uint32_t done = 0u; done < rows && status == SPARK_STATUS_OK; done += K31_INGEST_ROWS )
	{
		const uint32_t part = rows - done < K31_INGEST_ROWS ? rows - done : K31_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < K31_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * K31_TAP_WIDTH + (uint64_t)tap * K31_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * K31_HIDDEN, K31_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(first + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * K31_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
			status = SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = K31WriteRows(state, sequence, first + done, part);
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

static SparkStatus K31DraftRuns(const K31Sequence *sequence, uint64_t position, K31Draft *draft, uint32_t *held_out)
{
	const uint64_t capacity = sequence != 0 ? (uint64_t)sequence->page_count * K31_PAGE_POSITIONS : 0u;
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
		if ( draft->runs == K31_RUNS_MAX )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		draft->run_first[draft->runs] = (uint32_t)held;
		draft->run_position[draft->runs] = (uint32_t)begin;
		draft->runs++;
		held += finish - begin;
		at = finish;
	}
	*held_out = (uint32_t)held;
	return SPARK_STATUS_OK;
}

static SparkStatus K31Gemm(K31State *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t input_dimension,
	uint32_t output_dimension)
{
	return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, K31_BLOCK, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) == LM_LAUNCH_OK ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus K31LayerForward(K31State *state, uint32_t layer, uint32_t splits)
{
	K31Forward *forward = &state->forward;
	const K31Layer *weights = &state->layers[layer];
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(K31Draft, positions));
	const SparkGpuStream stream = state->stream;
	SparkStatus status;
	LM_LAUNCH((K31ResidualNormKernel), K31_BLOCK, K31_THREADS, 0, stream, forward->residual,
		layer == 0u ? (const uint16_t *)0 : forward->mlp, weights->input_norm, forward->x, state->epsilon);
	status = K31Gemm(state, weights->q_a, forward->x, forward->q_a_raw, K31_HIDDEN, K31_QUERY_RANK);
	if ( status == SPARK_STATUS_OK )
		status = K31Gemm(state, weights->kv_a, forward->x, forward->latent_raw, K31_HIDDEN, K31_LATENT);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((K31RmsRowsKernel), K31_BLOCK, K31_THREADS, 0, stream, forward->q_a_raw, weights->q_a_norm, forward->q_a, K31_QUERY_RANK,
		state->epsilon);
	status = K31Gemm(state, weights->q_b, forward->q_a, forward->q, K31_QUERY_RANK, K31_HEADS * K31_QK);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((K31QueryKernel), K31_HEADS * 4u, 64u, 0, stream, forward->q, weights->kv_b, state->rope_inverse, state->rope_gain,
		positions, forward->queries, forward->queries_f32);
	if ( LM_LAUNCH_DEPENDENT((K31LatentRowsKernel), K31_BLOCK, K31_THREADS, 0, stream, forward->latent_raw, weights->kv_a_norm, state->rope_inverse,
			state->rope_gain, positions, forward->block_destinations, state->epsilon) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( splits != 0u &&
		LM_LAUNCH_DEPENDENT((K31ContextAttentionKernel), dim3(K31_ATTN_TILES, splits), K31_ATTN_THREADS, K31_ATTN_SHARED_BYTES, stream, forward->queries,
			forward->pages, (uint64_t)layer * K31_PAGE_POSITIONS * K31_LATENT, forward->draft, state->attention_scale, forward->partials,
			forward->stats) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( LM_LAUNCH_DEPENDENT((K31AttentionMergeKernel), K31_QUERIES, K31_MERGE_THREADS, 0, stream, forward->partials, forward->stats, forward->draft,
		forward->queries_f32, forward->block_rows, state->attention_scale, forward->latent) != SPARK_GPU_SUCCESS ||
		LM_LAUNCH_DEPENDENT((K31ValueKernel), K31_HEADS * 4u, 128u, 0, stream, forward->latent, weights->kv_b, forward->values) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = K31Gemm(state, weights->o_proj, forward->values, forward->attention, K31_HEADS * K31_VALUE, K31_HIDDEN);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((K31ResidualNormKernel), K31_BLOCK, K31_THREADS, 0, stream, forward->residual, forward->attention, weights->post_norm,
		forward->x, state->epsilon);
	status = K31Gemm(state, weights->gate, forward->x, forward->gate, K31_HIDDEN, K31_MLP);
	if ( status == SPARK_STATUS_OK )
		status = K31Gemm(state, weights->up, forward->x, forward->up, K31_HIDDEN, K31_MLP);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((K31SwigluKernel), state->multiprocessors, K31_THREADS, 0, stream, forward->gate, forward->up, forward->act,
		K31_BLOCK * K31_MLP);
	return K31Gemm(state, weights->down, forward->act, forward->mlp, K31_MLP, K31_HIDDEN);
}

static SparkStatus K31ForwardReplay(K31State *state, uint32_t layers, uint32_t splits)
{
	K31Forward *forward = &state->forward;
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((K31EmbedKernel), K31_BLOCK, K31_THREADS, 0, state->stream, state->embed, forward->draft,
		forward->residual);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
		status = K31LayerForward(state, layer, splits);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( layers == K31_LAYERS )
		LM_LAUNCH((K31ResidualNormKernel), K31_BLOCK, K31_THREADS, 0, state->stream, forward->residual, forward->mlp, state->final_norm,
			forward->y, state->epsilon);
	return SPARK_STATUS_OK;
}

static SparkStatus K31PrepareForward(K31State *state, const K31Sequence *sequence, uint32_t anchor_token, uint64_t position, uint32_t layers,
	const SparkRowSampling *sampling, uint32_t drafts, uint32_t *splits_out)
{
	K31Forward *forward = &state->forward;
	const uint32_t pages = sequence != 0 ? sequence->page_count : 0u;
	uint32_t held = 0u, splits = 0u, split_rows = 0u;
	SparkStatus status;
	if ( anchor_token >= K31_VOCAB || position + K31_BLOCK > (uint64_t)K31_PAGES_MAX * K31_PAGE_POSITIONS || layers == 0u || layers > K31_LAYERS )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = K31DraftRuns(sequence, position, forward->host_draft, &held);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
		forward->host_draft->positions[row] = (uint32_t)(position + row);
	for ( uint32_t page = 0u; page < pages; page++ )
		forward->host_pages[page] = sequence->pages[page];
	if ( held != 0u )
	{
		splits = (held + K31_SPLIT_ROWS_MIN - 1u) / K31_SPLIT_ROWS_MIN;
		splits = splits > K31_SPLITS_MAX ? K31_SPLITS_MAX : splits;
		split_rows = ((held + splits - 1u) / splits + K31_ATTN_TILE - 1u) / K31_ATTN_TILE * K31_ATTN_TILE;
		splits = (held + split_rows - 1u) / split_rows;
	}
	forward->host_draft->anchor = anchor_token;
	forward->host_draft->mask = state->mask_token;
	forward->host_draft->position = (uint32_t)position;
	forward->host_draft->held = held;
	forward->host_draft->splits = splits;
	forward->host_draft->split_rows = split_rows;
	forward->host_draft->drafts = drafts;
	if ( sampling != 0 )
	{
		forward->host_draft->seed = sampling->seed;
		forward->host_draft->top_k = sampling->top_k;
		forward->host_draft->inverse_temperature = sampling->inverse_temperature;
		forward->host_draft->nucleus = sampling->nucleus;
		forward->host_draft->scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
	}
	else
	{
		forward->host_draft->seed = 0u;
		forward->host_draft->top_k = 0u;
		forward->host_draft->inverse_temperature = 0.0f;
		forward->host_draft->nucleus = 0.0f;
		forward->host_draft->scale = 1.0f;
	}
	if ( SparkGpuMemcpyAsync(forward->draft, forward->host_draft, sizeof(K31Draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
		(pages != 0u && SparkGpuMemcpyAsync(forward->pages, forward->host_pages, (size_t)pages * sizeof(*forward->host_pages),
			SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS) )
		return SPARK_STATUS_IO_ERROR;
	*splits_out = splits;
	return SPARK_STATUS_OK;
}

static SparkStatus K31ForwardRun(K31State *state, const K31Sequence *sequence, uint32_t anchor_token, uint64_t position, uint32_t layers)
{
	uint32_t splits = 0u;
	SparkStatus status = K31PrepareForward(state, sequence, anchor_token, position, layers, 0, K31_BLOCK, &splits);
	if ( status == SPARK_STATUS_OK )
		status = K31ForwardReplay(state, layers, splits);
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	return status;
}

static SparkStatus K31HeadsReplay(K31State *state, uint32_t branch_steps, uint32_t tree)
{
	K31Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	SparkStatus status;
	if ( LmDenseRows<LmBf16Format>(state->head, LmScaleTensorNone(), forward->y, forward->logits, (float *)0, K31_BLOCK, K31_HIDDEN, K31_VOCAB, 0u,
		state->multiprocessors, stream) != LM_LAUNCH_OK )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = SparkDrafterTopkBf16Launch(forward->logits, K31_BLOCK, K31_VOCAB, K31_CANDIDATES, &forward->topk_ws, forward->candidates, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t step = 0u; step < K31_BLOCK; step++ )
	{
		if ( LM_LAUNCH_DEPENDENT((K31MarkovRowKernel), K31_CANDIDATES / (K31_THREADS / K31_WARP), K31_THREADS, 0, stream, step, forward->draft,
			forward->picks->tokens, forward->candidates, forward->logits, state->markov_previous, state->markov_next,
			forward->markov_row) != SPARK_GPU_SUCCESS ||
			LM_LAUNCH_DEPENDENT((K31MarkovPickKernel), 1u, K31_PICK_THREADS, 0, stream, step, forward->draft, forward->markov_row, forward->candidates,
				forward->y, state->markov_previous, state->confidence_weight, state->confidence_bias, forward->picks) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
	}
	if ( tree != 0u && branch_steps >= 2u )
	{
		if ( LM_LAUNCH_DEPENDENT((K31BranchSelectKernel), 1u, K31_WARP, 0, stream, forward->picks, forward->draft,
			SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		for ( uint32_t offset = 1u; offset < branch_steps; offset++ )
		{
			if ( LM_LAUNCH_DEPENDENT((K31BranchRowKernel), dim3(K31_CANDIDATES / (K31_THREADS / K31_WARP), K31_BRANCHES), K31_THREADS, 0, stream, offset,
				forward->draft, forward->picks, forward->candidates, forward->logits, state->markov_previous, state->markov_next,
				forward->markov_row) != SPARK_GPU_SUCCESS ||
				LM_LAUNCH_DEPENDENT((K31BranchPickKernel), K31_BRANCHES, K31_PICK_THREADS, 0, stream, offset, forward->draft, forward->markov_row,
					forward->candidates, forward->picks) != SPARK_GPU_SUCCESS )
				return SPARK_STATUS_INTERNAL_ERROR;
		}
	}
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(K31Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static uint32_t K31Covers(const K31Sequence *sequence, uint64_t position)
{
	if ( position == 0u )
		return 1u;
	return sequence != 0 && position - 1u < (uint64_t)sequence->page_count * K31_PAGE_POSITIONS && sequence->held[position - 1u] != 0u ? 1u : 0u;
}

static uint32_t K31Complete(const K31Sequence *sequence, uint64_t position)
{
	return position <= (uint64_t)sequence->page_count * K31_PAGE_POSITIONS && memchr(sequence->held, 0, (size_t)position) == 0 ? 1u : 0u;
}

static SparkStatus K31CopyRows(K31State *state, K31Sequence *sequence, const K31Sequence *donor, uint64_t first, uint64_t end)
{
	while ( first < end )
	{
		const uint64_t page_end = (first / K31_PAGE_POSITIONS + 1u) * K31_PAGE_POSITIONS, stop = end < page_end ? end : page_end;
		for ( uint32_t layer = 0u; layer < K31_LAYERS; layer++ )
			if ( SparkGpuMemcpyAsync(K31Row(sequence, layer, first), K31Row(donor, layer, first), (size_t)(stop - first) * K31_ROW_BYTES,
				SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
				return SPARK_STATUS_IO_ERROR;
		memset(sequence->held + first, 1, (size_t)(stop - first));
		memcpy(sequence->tokens + first, donor->tokens + first, (size_t)(stop - first) * sizeof(uint32_t));
		first = stop;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus K31CopyPrefix(K31State *state, K31Sequence *sequence, const K31Sequence *donor, const SparkDraftServerRequest *request)
{
	const uint64_t donor_capacity = (uint64_t)donor->page_count * K31_PAGE_POSITIONS;
	const uint64_t shared = request->position < donor_capacity ? request->position : donor_capacity;
	uint64_t prefix = 0u, at = 0u;
	SparkStatus status;
	while ( prefix < shared && donor->tokens[prefix] == request->committed_token_ids[prefix] )
		prefix++;
	if ( prefix == 0u )
		return SPARK_STATUS_OK;
	status = K31SequencePages(state, sequence, prefix, donor);
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
		status = K31CopyRows(state, sequence, donor, at, stop);
		at = stop;
	}
	return status;
}

static SparkStatus K31FillFromDonors(K31State *state, K31Sequence *sequence, const SparkDraftServerRequest *request)
{
	static_assert(K31_SEQUENCES_MAX <= 64u, "one bit per sequence marks the donors tried");
	uint64_t tried = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	while ( status == SPARK_STATUS_OK && K31Complete(sequence, request->position) == 0u )
	{
		K31Sequence *donor = 0;
		uint32_t chosen = K31_SEQUENCES_MAX;
		for ( uint32_t index = 0u; index < K31_SEQUENCES_MAX; index++ )
		{
			K31Sequence *other = &state->sequences[index];
			if ( other->pages != 0 && other != sequence && ((tried >> index) & 1u) == 0u && (donor == 0 || other->last_used > donor->last_used) )
			{
				donor = other;
				chosen = index;
			}
		}
		if ( donor == 0 )
			break;
		tried |= UINT64_C(1) << chosen;
		status = K31CopyPrefix(state, sequence, donor, request);
	}
	return status;
}

static uint32_t K31TokensMatch(K31Sequence *sequence, const SparkDraftServerRequest *request)
{
	const uint64_t capacity = (uint64_t)sequence->page_count * K31_PAGE_POSITIONS;
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

static void K31ForwardFree(K31Forward *forward)
{
	void *buffers[] = {
		forward->draft, forward->pages, forward->block_destinations, forward->residual, forward->x, forward->q_a_raw, forward->q_a, forward->q,
		forward->latent_raw, forward->block_rows, forward->queries, forward->queries_f32, forward->partials, forward->stats, forward->latent,
		forward->values, forward->attention, forward->gate, forward->up, forward->act, forward->mlp, forward->y, forward->logits,
		forward->candidates, forward->markov_row, forward->picks,
	};
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		(void)SparkGpuFree(buffers[index]);
	SparkDrafterTopkBf16Free(&forward->topk_ws);
	if ( forward->host_draft != 0 )
		(void)SparkGpuFreeHost(forward->host_draft);
	if ( forward->host_picks != 0 )
		(void)SparkGpuFreeHost(forward->host_picks);
	if ( forward->host_pages != 0 )
		(void)SparkGpuFreeHost((void *)forward->host_pages);
}

static SparkStatus K31Close(void *opaque)
{
	K31State *state = (K31State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	SparkDrafterGraphCacheDestroy(&state->graphs);
	for ( uint32_t index = 0u; index < K31_SEQUENCES_MAX; index++ )
		if ( state->sequences[index].pages != 0 )
			K31SequenceFree(state, &state->sequences[index]);
	K31ForwardFree(&state->forward);
	(void)SparkGpuFree(state->taps);
	(void)SparkGpuFree(state->features_raw);
	(void)SparkGpuFree(state->features);
	(void)SparkGpuFree(state->latent_raw);
	(void)SparkGpuFree(state->positions);
	(void)SparkGpuFree(state->ingest_rows);
	(void)SparkGpuFree(state->rope_inverse);
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

static SparkStatus K31Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	K31State *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return K31Refuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	if ( K31DeviceOpen() != SPARK_STATUS_OK || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
		return K31Refuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
	bind->device_free_bytes = (uint64_t)free_bytes;
	bind->device_total_bytes = (uint64_t)total_bytes;
	need = lane->pack_bytes + 255u + lane->context_bytes;
	if ( need > (uint64_t)free_bytes )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s needs %llu bytes of device memory (pack %llu + context cap %llu), %llu of %llu free",
			lane->lane_name, (unsigned long long)need, (unsigned long long)lane->pack_bytes, (unsigned long long)lane->context_bytes,
			(unsigned long long)free_bytes, (unsigned long long)total_bytes);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( lane->context_bytes < K31_PAGE_BYTES )
		return K31Refuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one context page", lane->lane_name);
	state = (K31State *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
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
		return status;
	}
	status = K31Weights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = K31Scratch(state);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a K3.1 Pro DSpark pack this module reads", lane->lane_name, lane->pack_path);
		(void)K31Close(state);
		return status;
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s", lane->lane_name, lane->pack_path);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus K31Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	K31State *state = (K31State *)opaque;
	K31Sequence *sequence = K31SequenceFind(state, request->sequence_id, 0u);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( K31Covers(sequence, request->position) == 0u )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	if ( sequence != 0 && K31Complete(sequence, request->position) == 0u && K31FillFromDonors(state, sequence, request) != SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	if ( sequence != 0 && K31TokensMatch(sequence, request) == 0u )
		return SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT;
	drafts = drafts < K31_BLOCK ? drafts : K31_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		const uint32_t branch_steps = tree != 0u ? (drafts < K31_BRANCH_LEN ? drafts : K31_BRANCH_LEN) : 0u;
		uint32_t splits = 0u;
		SparkDraftTreeBranch branches[K31_BLOCK];
		uint32_t branch_count = 0u;
		status = K31PrepareForward(state, sequence, request->anchor_token_id, request->position, K31_LAYERS, &request->sampling, drafts, &splits);
		if ( status == SPARK_STATUS_OK )
		{
			const uint64_t key = ((uint64_t)splits << 8u) | ((uint64_t)tree << 4u) | (uint64_t)branch_steps;
			status = SparkDrafterGraphRun(&state->graphs, state->stream, key, [&]() -> SparkStatus
			{
				SparkStatus inner = K31ForwardReplay(state, K31_LAYERS, splits);
				if ( inner == SPARK_STATUS_OK )
					inner = K31HeadsReplay(state, branch_steps, tree);
				return inner;
			});
		}
		if ( status == SPARK_STATUS_OK && SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS )
			status = SPARK_STATUS_IO_ERROR;
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				uint32_t matched = K31_BRANCHES;
				for ( uint32_t b = 0u; b < K31_BRANCHES; b++ )
					if ( state->forward.host_picks->branch_len[b] > 0u && state->forward.host_picks->branch_fork[b] == step )
						matched = b;
				branches[branch_count].fork_step = step;
				if ( matched < K31_BRANCHES )
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

static const SparkDrafterModuleInterface k31_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"k31-dspark",
	K31Open,
	K31Ingest,
	K31Propose,
	K31Close,
};

static SparkStatus K31Readback(void *host, const void *device, size_t bytes)
{
	return SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &k31_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_k31_dspark_context_rows(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint64_t first, uint32_t count, uint16_t *host)
{
	K31State *state = (K31State *)opaque;
	K31Sequence *sequence = state != 0 ? K31SequenceFind(state, sequence_id, 0u) : 0;
	if ( sequence == 0 || layer >= K31_LAYERS || first + count > sequence->held_end )
		return SPARK_STATUS_NOT_FOUND;
	for ( uint32_t row = 0u; row < count; row++ )
		if ( K31Readback(host + (uint64_t)row * K31_LATENT, K31Row(sequence, layer, first + row), K31_LATENT * sizeof(uint16_t)) != SPARK_STATUS_OK )
			return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_k31_dspark_block_rows(void *opaque, uint64_t sequence_id, uint32_t anchor_token,
	uint64_t position, uint16_t *host, uint64_t *elapsed_ns)
{
	K31State *state = (K31State *)opaque;
	const K31Sequence *sequence = state != 0 ? K31SequenceFind(state, sequence_id, 0u) : 0;
	const uint64_t started = SparkMonotonicNs();
	SparkStatus status;
	if ( state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = K31ForwardRun(state, sequence, anchor_token, position, K31_LAYERS);
	if ( status == SPARK_STATUS_OK && SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_IO_ERROR;
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && host != 0 )
		status = K31Readback(host, state->forward.y, (size_t)K31_BLOCK * K31_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_k31_dspark_block_trace(void *opaque, uint64_t sequence_id, uint32_t anchor_token,
	uint64_t position, uint32_t layers, float *queries, float *latent, uint16_t *block)
{
	K31State *state = (K31State *)opaque;
	const K31Sequence *sequence = state != 0 ? K31SequenceFind(state, sequence_id, 0u) : 0;
	SparkStatus status;
	if ( state == 0 || queries == 0 || latent == 0 || block == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = K31ForwardRun(state, sequence, anchor_token, position, layers);
	if ( status == SPARK_STATUS_OK && SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = K31Readback(queries, state->forward.queries_f32, (size_t)K31_QUERIES * K31_LATENT * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = K31Readback(latent, state->forward.latent, (size_t)K31_QUERIES * K31_KV_RANK * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = K31Readback(block, state->forward.block_rows, (size_t)K31_BLOCK * K31_LATENT * sizeof(uint16_t));
	return status;
}
