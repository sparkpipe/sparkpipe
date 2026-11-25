#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_draft_tree_pack.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/drafter_blackwell.cuh"
#include "runtime/launch.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q38_HIDDEN 5120u
#define Q38_TAPS 5u
#define Q38_TAP_WIDTH (Q38_TAPS * Q38_HIDDEN)
#define Q38_LAYERS 5u
#define Q38_HEADS 32u
#define Q38_KV_HEADS 8u
#define Q38_HEAD_DIM 128u
#define Q38_HALF (Q38_HEAD_DIM / 2u)
#define Q38_GROUP (Q38_HEADS / Q38_KV_HEADS)
#define Q38_Q_WIDTH (Q38_HEADS * Q38_HEAD_DIM)
#define Q38_KV_WIDTH (Q38_KV_HEADS * Q38_HEAD_DIM)
#define Q38_MLP 17408u
#define Q38_BLOCK 16u
#define Q38_DRAFTS (Q38_BLOCK - 1u)
#define Q38_SLIDING 8192u
#define Q38_WINDOW (Q38_SLIDING - Q38_BLOCK)
#define Q38_COLUMNS (Q38_WINDOW + Q38_BLOCK)
#define Q38_VOCAB 248320u
#define Q38_CODE 1024u
#define Q38_RANK 512u
#define Q38_PACKED (3u * Q38_RANK)
#define Q38_LANES 16u
#define Q38_BRANCH_LEN 4u
#define Q38_POOL 64u
#define Q38_EARLY (Q38_LAYERS - 1u)
#define Q38_INGEST_ROWS 256u
#define Q38_THREADS 256u
#define Q38_WARP 32u
#define Q38_WARPS (Q38_THREADS / Q38_WARP)
#define Q38_SPLIT_KEYS 256u
#define Q38_SPLITS ((Q38_WINDOW + Q38_SPLIT_KEYS - 1u) / Q38_SPLIT_KEYS)
#define Q38_QUERIES (Q38_GROUP * Q38_BLOCK)
#define Q38_ATTENTION_TILE 32u
#define Q38_STREAM_OUTPUTS 4096u
#define Q38_ATTENTION_THREADS (Q38_GROUP * Q38_WARP)
#define Q38_TOPK_THREADS 512u
#define Q38_TOPK_CHUNK 2048u
#define Q38_SEQUENCES_MAX 256u
#define Q38_LAYER_KV ((uint64_t)Q38_KV_HEADS * Q38_COLUMNS * Q38_HEAD_DIM)
#define Q38_SEQUENCE_ELEMENTS (2u * Q38_LAYERS * Q38_LAYER_KV)
#define Q38_SEQUENCE_BYTES (Q38_SEQUENCE_ELEMENTS * sizeof(uint16_t))
#define Q38_PACK_MAGIC 0x4B505244u
#define Q38_PACK_VERSION 1u
#define Q38_PACK_ALIGNMENT 128u
#define Q38_INVERSE_PI 0.31830988618379067154
#define Q38_CONFIDENCE_FLOOR 1e-6f
#define Q38_TINY 1.1754943508222875e-38f

typedef struct Q38Layer
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
	const uint16_t *read;
	const uint16_t *write;
} Q38Layer;

typedef struct Q38Sequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t last_used;
	uint32_t live;
	uint32_t retired;
	int64_t positions[Q38_WINDOW];
	uint32_t tokens[Q38_WINDOW];
	uint16_t *cache;
} Q38Sequence;

typedef struct Q38Draft
{
	uint16_t *cache;
	uint64_t seed;
	uint32_t positions[Q38_BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t greedy;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint32_t start;
	uint32_t block_at;
	uint32_t masked;
	uint32_t splits;
	uint32_t tree;
	uint8_t valid[Q38_WINDOW];
} Q38Draft;

typedef struct Q38Picks
{
	uint32_t tokens[Q38_DRAFTS];
	float scores[Q38_DRAFTS];
	uint32_t branch_tokens[Q38_DRAFTS][Q38_BRANCH_LEN];
	float branch_scores[Q38_DRAFTS][Q38_BRANCH_LEN];
} Q38Picks;

static __device__ __forceinline__ void Q38Rotation(uint32_t position, float inverse, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * Q38_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	sincospif(reduced, sine, cosine);
}

static __device__ __forceinline__ float Q38WarpSum(float value)
{
	for ( uint32_t mask = Q38_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ float Q38Sigmoid(float value)
{
	return 1.0f / (1.0f + expf(-value));
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38RmsKernel(const uint16_t *__restrict__ input, const float *__restrict__ input_f32, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ output, uint32_t dimension, float epsilon, uint32_t fused)
{
	__shared__ float reduction[Q38_WARPS];
	const uint64_t row = blockIdx.x;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += Q38_THREADS )
	{
		const float value = input_f32 != 0 ? input_f32[row * dimension + index] : LmBf16ToFloat(__ldg(input + row * dimension + index));
		total += value * value;
	}
	total = LmBlockSum<Q38_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += Q38_THREADS )
	{
		const float value = input_f32 != 0 ? input_f32[row * dimension + index] : LmBf16ToFloat(__ldg(input + row * dimension + index));
		const float normalized = fused != 0u ? LmBf16ToFloat(LmFloatToBf16(value * scale)) : value * scale;
		output[row * dimension + index] = LmFloatToBf16(normalized * LmBf16ToFloat(__ldg(weight + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38DualRmsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ out_fused,
	uint16_t *__restrict__ out_plain, float epsilon)
{
	LmDependentWait();
	__shared__ float reduction[Q38_WARPS];
	const uint64_t row = blockIdx.x;
	const uint16_t *src = input + row * Q38_HIDDEN;
	uint16_t *dst_f = out_fused + row * Q38_HIDDEN;
	uint16_t *dst_p = out_plain + row * Q38_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(src + index));
		total += value * value;
	}
	total = LmBlockSum<Q38_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)Q38_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(src + index)) * scale;
		const float w = LmBf16ToFloat(__ldg(weight + index));
		dst_f[index] = LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value)) * w);
		dst_p[index] = LmFloatToBf16(value * w);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38ResidualRmsKernel(const float *__restrict__ hidden_f32, const uint16_t *__restrict__ hidden, const uint16_t *__restrict__ addend,
	const uint16_t *__restrict__ weight, uint16_t *__restrict__ sum_out, uint16_t *__restrict__ norm_out, float epsilon)
{
	__shared__ float reduction[Q38_WARPS];
	const uint64_t row = blockIdx.x;
	const float *src_f32 = hidden_f32 != 0 ? hidden_f32 + row * Q38_HIDDEN : (const float *)0;
	const uint16_t *src_h = hidden + row * Q38_HIDDEN;
	const uint16_t *src_a = addend + row * Q38_HIDDEN;
	uint16_t *dst_sum = sum_out + row * Q38_HIDDEN;
	uint16_t *dst_norm = norm_out + row * Q38_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
	{
		const float base = src_f32 != 0 ? LmBf16ToFloat(LmFloatToBf16(src_f32[index])) : LmBf16ToFloat(__ldg(src_h + index));
		const uint16_t sum_bf16 = LmFloatToBf16(base + LmBf16ToFloat(__ldg(src_a + index)));
		const float value = LmBf16ToFloat(sum_bf16);
		dst_sum[index] = sum_bf16;
		total += value * value;
	}
	total = LmBlockSum<Q38_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)Q38_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
		dst_norm[index] = LmFloatToBf16(LmBf16ToFloat(dst_sum[index]) * scale * LmBf16ToFloat(__ldg(weight + index)));
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_HEAD_DIM, 4)
void Q38HeadNormRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, uint16_t *__restrict__ output, const Q38Draft *__restrict__ draft,
	uint64_t layer_offset)
{
	if ( output == 0 )
		LmDependentWait();
	__shared__ float reduction[Q38_HEAD_DIM / Q38_WARP];
	__shared__ float normed[Q38_HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = LmBf16ToFloat(__ldg(input + ((uint64_t)row * heads + head) * Q38_HEAD_DIM + lane));
	float scale, cosine, sine, rotated;
	scale = rsqrtf(LmBlockSum<Q38_HEAD_DIM>(value * value, reduction) / (float)Q38_HEAD_DIM + epsilon);
	normed[lane] = LmBf16ToFloat(LmFloatToBf16(value * scale * LmBf16ToFloat(__ldg(weight + lane))));
	__syncthreads();
	Q38Rotation(position, __ldg(inverse + (lane % Q38_HALF)), &cosine, &sine);
	rotated = lane < Q38_HALF ? normed[lane] * cosine - normed[lane + Q38_HALF] * sine : normed[lane] * cosine + normed[lane - Q38_HALF] * sine;
	if ( output == 0 )
	{
		uint16_t *cache_keys = draft->cache + layer_offset;
		cache_keys[((uint64_t)head * Q38_COLUMNS + draft->block_at + row) * Q38_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	}
	else
		output[((uint64_t)row * heads + head) * Q38_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38IngestKeysKernel(const uint16_t *__restrict__ raw, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, float epsilon, uint16_t *__restrict__ keys)
{
	__shared__ float reduction[Q38_THREADS / Q38_WARP];
	__shared__ float normed[Q38_KV_HEADS][Q38_HEAD_DIM];
	const uint32_t row = blockIdx.x, position = positions[row], slot = position % Q38_WINDOW;
	for ( uint32_t head = 0u; head < Q38_KV_HEADS; head += Q38_THREADS / Q38_HEAD_DIM )
	{
		const uint32_t own = head + threadIdx.x / Q38_HEAD_DIM, lane = threadIdx.x % Q38_HEAD_DIM;
		const float value = LmBf16ToFloat(__ldg(raw + ((uint64_t)row * Q38_KV_HEADS + own) * Q38_HEAD_DIM + lane));
		float total = Q38WarpSum(value * value);
		if ( threadIdx.x % Q38_WARP == 0u )
			reduction[threadIdx.x / Q38_WARP] = total;
		__syncthreads();
		total = 0.0f;
		for ( uint32_t warp = 0u; warp < Q38_HEAD_DIM / Q38_WARP; warp++ )
			total += reduction[(threadIdx.x / Q38_HEAD_DIM) * (Q38_HEAD_DIM / Q38_WARP) + warp];
		normed[own][lane] = LmBf16ToFloat(LmFloatToBf16(value * rsqrtf(total / (float)Q38_HEAD_DIM + epsilon) * LmBf16ToFloat(__ldg(weight + lane))));
		__syncthreads();
	}
	for ( uint32_t index = threadIdx.x; index < Q38_KV_WIDTH; index += Q38_THREADS )
	{
		const uint32_t head = index / Q38_HEAD_DIM, lane = index % Q38_HEAD_DIM;
		float cosine, sine, rotated;
		Q38Rotation(position, __ldg(inverse + (lane % Q38_HALF)), &cosine, &sine);
		rotated = lane < Q38_HALF ? normed[head][lane] * cosine - normed[head][lane + Q38_HALF] * sine :
			normed[head][lane] * cosine + normed[head][lane - Q38_HALF] * sine;
		keys[((uint64_t)head * Q38_COLUMNS + slot) * Q38_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	}
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38StoreValuesKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions, const Q38Draft *__restrict__ draft,
	uint64_t layer_offset, uint16_t *__restrict__ cache)
{
	if ( draft != 0 )
		LmDependentWait();
	const uint32_t row = blockIdx.x, column = positions != 0 ? positions[row] % Q38_WINDOW : draft->block_at + row;
	uint16_t *dst = cache != 0 ? cache : draft->cache + layer_offset;
	const uint4 *vsrc = (const uint4 *)(values + (uint64_t)row * Q38_KV_WIDTH);
	for ( uint32_t v = threadIdx.x; v < Q38_KV_WIDTH / 8u; v += Q38_THREADS )
	{
		const uint32_t head = v >> 4u, chunk = v & 15u;
		*(uint4 *)(dst + ((uint64_t)head * Q38_COLUMNS + column) * Q38_HEAD_DIM + chunk * 8u) = __ldg(vsrc + v);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38EmbedKernel(const uint16_t *__restrict__ embed, const Q38Draft *__restrict__ draft, float *__restrict__ hidden)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->mask;
	const uint4 *src = (const uint4 *)(embed + (uint64_t)token * Q38_HIDDEN);
	float4 *dst = (float4 *)(hidden + (uint64_t)row * Q38_HIDDEN);
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN / 8u; index += Q38_THREADS )
	{
		const uint4 p = __ldg(src + index);
		dst[2u * index] = make_float4(__uint_as_float(p.x << 16u), __uint_as_float(p.x & 0xffff0000u),
			__uint_as_float(p.y << 16u), __uint_as_float(p.y & 0xffff0000u));
		dst[2u * index + 1u] = make_float4(__uint_as_float(p.z << 16u), __uint_as_float(p.z & 0xffff0000u),
			__uint_as_float(p.w << 16u), __uint_as_float(p.w & 0xffff0000u));
	}
	LmDependentRelease();
}

static __device__ __forceinline__ uint32_t Q38Swizzle(uint32_t row, uint32_t chunk)
{
	return chunk ^ (row & 7u);
}

static __device__ __forceinline__ void Q38LoadX4(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void Q38LoadX4Trans(uint32_t address, uint32_t r[4])
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void Q38Stage(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

__global__ __launch_bounds__(Q38_ATTENTION_THREADS, 1)
void Q38AttentionSplitKernel(const uint16_t *__restrict__ q, uint64_t keys_offset, uint64_t values_offset,
	const Q38Draft *__restrict__ draft, float scale, float *__restrict__ partials)
{
	LmDependentWait();
	constexpr uint32_t TILE = Q38_ATTENTION_TILE, ROW_BYTES = Q38_HEAD_DIM * 2u, CHUNKS = Q38_HEAD_DIM / 8u, K_STEPS = Q38_HEAD_DIM / 16u;
	constexpr uint32_t KEY_FRAGS = TILE / 8u, P_STEPS = TILE / 16u, VALUE_FRAGS = Q38_HEAD_DIM / 8u;
	__shared__ __align__(128) uint8_t key_tiles[2u * TILE * ROW_BYTES];
	__shared__ __align__(128) uint8_t value_tiles[2u * TILE * ROW_BYTES];
	__shared__ uint8_t visible[2u * TILE];
	const uint32_t group = blockIdx.x, split = blockIdx.y, warp = threadIdx.x / Q38_WARP, lane = threadIdx.x % Q38_WARP;
	const uint32_t matrix = lane >> 3u, within = lane & 7u, head = group * Q38_GROUP + warp, masked = draft->masked;
	const uint32_t first = draft->start + split * Q38_SPLIT_KEYS, end = draft->block_at;
	const uint32_t stop = first + Q38_SPLIT_KEYS < end ? first + Q38_SPLIT_KEYS : end, tiles = first < stop ? (stop - first + TILE - 1u) / TILE : 0u;
	const uint16_t *group_keys = draft->cache + keys_offset + (uint64_t)group * Q38_COLUMNS * Q38_HEAD_DIM;
	const uint16_t *group_values = draft->cache + values_offset + (uint64_t)group * Q38_COLUMNS * Q38_HEAD_DIM;
	const float scale_log2 = scale * 1.4426950408889634f;
	float running_max[2] = { -INFINITY, -INFINITY }, running_sum[2] = { 0.0f, 0.0f }, output[VALUE_FRAGS][4];
	uint32_t query[K_STEPS][4];
	float *base = partials + ((uint64_t)group * Q38_SPLITS + split) * Q38_QUERIES * (Q38_HEAD_DIM + 2u);
#pragma unroll
	for ( uint32_t step = 0u; step < K_STEPS; step++ )
#pragma unroll
		for ( uint32_t index = 0u; index < 4u; index++ )
			query[step][index] = *(const uint32_t *)(q + ((uint64_t)((lane >> 2u) + 8u * (index % 2u)) * Q38_HEADS + head) * Q38_HEAD_DIM + step * 16u +
				2u * (lane & 3u) + 8u * (index / 2u));
#pragma unroll
	for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag++ )
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	auto load = [&](uint32_t tile, uint32_t buffer)
	{
		const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(key_tiles + buffer * TILE * ROW_BYTES);
		const uint32_t value_base = (uint32_t)__cvta_generic_to_shared(value_tiles + buffer * TILE * ROW_BYTES);
		const uint32_t tile_first = first + tile * TILE;
		for ( uint32_t item = threadIdx.x; item < TILE * 2u * CHUNKS; item += Q38_ATTENTION_THREADS )
		{
			const uint32_t key = item / (2u * CHUNKS), piece = item % (2u * CHUNKS), column = tile_first + key, inside = column < stop ? 1u : 0u;
			const uint64_t row = (uint64_t)(inside != 0u ? column : first) * Q38_HEAD_DIM;
			if ( piece < CHUNKS )
				Q38Stage(key_base + key * ROW_BYTES + Q38Swizzle(key, piece) * 16u, group_keys + row + piece * 8u, inside != 0u ? 16u : 0u);
			else
				Q38Stage(value_base + key * ROW_BYTES + Q38Swizzle(key, piece - CHUNKS) * 16u, group_values + row + (piece - CHUNKS) * 8u,
					inside != 0u ? 16u : 0u);
		}
		if ( threadIdx.x < TILE )
			visible[buffer * TILE + threadIdx.x] = tile_first + threadIdx.x < stop && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	if ( tiles != 0u )
		load(0u, 0u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_keys = (uint32_t)__cvta_generic_to_shared(key_tiles + buffer * TILE * ROW_BYTES);
		const uint32_t tile_values = (uint32_t)__cvta_generic_to_shared(value_tiles + buffer * TILE * ROW_BYTES);
		float score[KEY_FRAGS][4], peak[2], rescale[2], shift[2];
		if ( tile + 1u < tiles )
		{
			load(tile + 1u, buffer ^ 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
#pragma unroll
		for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
			score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
#pragma unroll
		for ( uint32_t step = 0u; step < K_STEPS; step++ )
#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag += 2u )
			{
				const uint32_t b_row = (frag + (matrix >> 1u)) * 8u + within, b_chunk = 2u * step + (matrix & 1u);
				uint32_t b[4];
				Q38LoadX4(tile_keys + b_row * ROW_BYTES + Q38Swizzle(b_row, b_chunk) * 16u, b);
				LmMmaBf16(score[frag], query[step], b);
				LmMmaBf16(score[frag + 1u], query[step], b + 2);
			}
		peak[0] = peak[1] = -INFINITY;
#pragma unroll
		for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
#pragma unroll
			for ( uint32_t entry = 0u; entry < 4u; entry++ )
			{
				const uint32_t key = frag * 8u + 2u * (lane & 3u) + (entry & 1u);
				score[frag][entry] = visible[buffer * TILE + key] != 0u ? score[frag][entry] : -INFINITY;
				peak[entry / 2u] = fmaxf(peak[entry / 2u], score[frag][entry]);
			}
#pragma unroll
		for ( uint32_t side = 0u; side < 2u; side++ )
		{
			float next;
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 1u));
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 2u));
			next = fmaxf(running_max[side], peak[side]);
			rescale[side] = running_max[side] == -INFINITY ? 0.0f : SparkMathExp2Fast((running_max[side] - next) * scale_log2);
			running_max[side] = next;
			shift[side] = next == -INFINITY ? 0.0f : next * scale_log2;
			running_sum[side] *= rescale[side];
		}
#pragma unroll
		for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag++ )
		{
			output[frag][0] *= rescale[0];
			output[frag][1] *= rescale[0];
			output[frag][2] *= rescale[1];
			output[frag][3] *= rescale[1];
		}
#pragma unroll
		for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
#pragma unroll
			for ( uint32_t entry = 0u; entry < 4u; entry++ )
			{
				const float weight = SparkMathExp2Fast(fmaf(score[frag][entry], scale_log2, -shift[entry / 2u]));
				running_sum[entry / 2u] += weight;
				score[frag][entry] = weight;
			}
#pragma unroll
		for ( uint32_t step = 0u; step < P_STEPS; step++ )
		{
			uint32_t weights[4];
#pragma unroll
			for ( uint32_t index = 0u; index < 4u; index++ )
				weights[index] = LmPackBf16Pair(score[2u * step + index / 2u][2u * (index % 2u)], score[2u * step + index / 2u][2u * (index % 2u) + 1u]);
#pragma unroll
			for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag += 2u )
			{
				const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within, b_chunk = frag + (matrix >> 1u);
				uint32_t b[4];
				Q38LoadX4Trans(tile_values + b_row * ROW_BYTES + Q38Swizzle(b_row, b_chunk) * 16u, b);
				LmMmaBf16(output[frag], weights, b);
				LmMmaBf16(output[frag + 1u], weights, b + 2);
			}
		}
		__syncthreads();
	}
#pragma unroll
	for ( uint32_t side = 0u; side < 2u; side++ )
	{
		float *out = base + (uint64_t)(warp * Q38_BLOCK + (lane >> 2u) + 8u * side) * (Q38_HEAD_DIM + 2u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 2u);
#pragma unroll
		for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag++ )
		{
			out[frag * 8u + 2u * (lane & 3u)] = output[frag][2u * side];
			out[frag * 8u + 2u * (lane & 3u) + 1u] = output[frag][2u * side + 1u];
		}
		if ( (lane & 3u) == 0u )
		{
			out[Q38_HEAD_DIM] = running_max[side] == -INFINITY ? -INFINITY : running_max[side] * scale;
			out[Q38_HEAD_DIM + 1u] = running_sum[side];
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_HEAD_DIM, 4)
void Q38AttentionMergeKernel(const uint16_t *__restrict__ q, uint64_t keys_offset, uint64_t values_offset,
	const Q38Draft *__restrict__ draft, float scale, const float *__restrict__ partials, uint16_t *__restrict__ output)
{
	LmDependentWait();
	__shared__ float reduction[Q38_HEAD_DIM / Q38_WARP];
	const uint32_t row = blockIdx.x / Q38_HEADS, head = blockIdx.x % Q38_HEADS, group = head / Q38_GROUP, lane = threadIdx.x;
	const uint32_t query_index = (head % Q38_GROUP) * Q38_BLOCK + row, block_at = draft->block_at, splits = draft->splits;
	const float own = LmBf16ToFloat(__ldg(q + ((uint64_t)row * Q38_HEADS + head) * Q38_HEAD_DIM + lane));
	const uint16_t *group_keys = draft->cache + keys_offset + (uint64_t)group * Q38_COLUMNS * Q38_HEAD_DIM;
	const uint16_t *group_values = draft->cache + values_offset + (uint64_t)group * Q38_COLUMNS * Q38_HEAD_DIM;
	float top = -INFINITY, sum = 0.0f, accumulated = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part = partials + (((uint64_t)group * Q38_SPLITS + split) * Q38_QUERIES + query_index) * (Q38_HEAD_DIM + 2u);
		const float part_top = part[Q38_HEAD_DIM], next = fmaxf(top, part_top);
		float carry, weight;
		if ( part_top == -INFINITY )
			continue;
		carry = top == -INFINITY ? 0.0f : expf(top - next);
		weight = expf(part_top - next);
		sum = sum * carry + part[Q38_HEAD_DIM + 1u] * weight;
		accumulated = accumulated * carry + part[lane] * weight;
		top = next;
	}
	for ( uint32_t key = 0u; key <= row; key++ )
	{
		const float score = LmBlockSum<Q38_HEAD_DIM>(own * LmBf16ToFloat(__ldg(group_keys + (uint64_t)(block_at + key) * Q38_HEAD_DIM + lane)), reduction) * scale;
		const float next = fmaxf(top, score), carry = top == -INFINITY ? 0.0f : expf(top - next), weight = expf(score - next);
		sum = sum * carry + weight;
		accumulated = accumulated * carry + weight * LmBf16ToFloat(__ldg(group_values + (uint64_t)(block_at + key) * Q38_HEAD_DIM + lane));
		top = next;
	}
	output[(uint64_t)row * Q38_Q_WIDTH + head * Q38_HEAD_DIM + lane] = LmFloatToBf16(accumulated / sum);
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38ResidualKernel(const float *__restrict__ hidden_f32, const uint16_t *__restrict__ hidden, const uint16_t *__restrict__ addend,
	uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * Q38_THREADS + threadIdx.x; index < count; index += gridDim.x * Q38_THREADS )
	{
		const float base = hidden_f32 != 0 ? LmBf16ToFloat(LmFloatToBf16(hidden_f32[index])) : LmBf16ToFloat(__ldg(hidden + index));
		output[index] = LmFloatToBf16(base + LmBf16ToFloat(__ldg(addend + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38SwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t count)
{
	for ( uint32_t index = blockIdx.x * Q38_THREADS + threadIdx.x; index < count; index += gridDim.x * Q38_THREADS )
	{
		const float g = LmBf16ToFloat(__ldg(gate + index));
		output[index] = LmFloatToBf16(g / (1.0f + expf(-g)) * LmBf16ToFloat(__ldg(up + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38ResidualBridgeScaleKernel(const uint16_t *__restrict__ residual, const uint16_t *__restrict__ mlp, uint16_t *__restrict__ raw,
	float *__restrict__ rms, uint16_t *__restrict__ normalized, float epsilon)
{
	__shared__ float reduction[Q38_WARPS];
	const uint64_t row = blockIdx.x;
	const uint16_t *src_r = residual + row * Q38_HIDDEN;
	const uint16_t *src_m = mlp + row * Q38_HIDDEN;
	uint16_t *dst_raw = raw + row * Q38_HIDDEN;
	uint16_t *dst_norm = normalized + row * Q38_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
	{
		const uint16_t sum_bf16 = LmFloatToBf16(LmBf16ToFloat(__ldg(src_r + index)) + LmBf16ToFloat(__ldg(src_m + index)));
		const float value = LmBf16ToFloat(sum_bf16);
		dst_raw[index] = sum_bf16;
		total += value * value;
	}
	total = LmBlockSum<Q38_THREADS>(total, reduction);
	scale = sqrtf(fmaxf(total / (float)Q38_HIDDEN, Q38_TINY));
	if ( threadIdx.x == 0u )
		rms[row] = scale;
	const float denom = scale + epsilon;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
		dst_norm[index] = LmFloatToBf16(LmBf16ToFloat(dst_raw[index]) / denom);
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38BridgeCodesKernel(const uint16_t *__restrict__ projected, const uint16_t *__restrict__ codebook, const Q38Draft *__restrict__ draft,
	float epsilon, uint16_t *__restrict__ codes)
{
	__shared__ float reduction[Q38_WARPS];
	const uint64_t row = blockIdx.x;
	const float root = sqrtf((float)Q38_CODE);
	float total = 0.0f, length;
	if ( row == 0u )
	{
		for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += Q38_THREADS )
			codes[index] = LmFloatToBf16(root * LmBf16ToFloat(__ldg(codebook + (uint64_t)draft->anchor * Q38_CODE + index)));
		LmDependentRelease();
		return;
	}
	for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += Q38_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(projected + row * Q38_CODE + index));
		total += value * value;
	}
	length = sqrtf(LmBlockSum<Q38_THREADS>(total, reduction));
	for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += Q38_THREADS )
		codes[row * Q38_CODE + index] = LmFloatToBf16(root * LmBf16ToFloat(__ldg(projected + row * Q38_CODE + index)) / (length + epsilon));
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38BridgeMessageKernel(const uint16_t *__restrict__ packed, const uint16_t *__restrict__ bias, uint16_t *__restrict__ message)
{
	const uint32_t row = blockIdx.x;
	for ( uint32_t index = threadIdx.x; index < Q38_RANK; index += Q38_THREADS )
	{
		float value = 0.0f;
		if ( row != 0u )
		{
			const float query = LmBf16ToFloat(__ldg(packed + (uint64_t)row * Q38_PACKED + index));
			const float sender = LmBf16ToFloat(__ldg(packed + (uint64_t)(row - 1u) * Q38_PACKED + Q38_RANK + index));
			const float prior = LmBf16ToFloat(__ldg(packed + (uint64_t)(row - 1u) * Q38_PACKED + 2u * Q38_RANK + index));
			value = Q38Sigmoid(query + sender + LmBf16ToFloat(__ldg(bias + index))) * prior;
		}
		message[(uint64_t)row * Q38_RANK + index] = LmFloatToBf16(value);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38BridgeOutKernel(const uint16_t *__restrict__ raw, const float *__restrict__ rms, const uint16_t *__restrict__ delta, uint16_t *__restrict__ output)
{
	const uint64_t row = blockIdx.x;
	const float scale = rms[row];
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(raw + row * Q38_HIDDEN + index));
		output[row * Q38_HIDDEN + index] = LmFloatToBf16(row == 0u ? value : value + scale * LmBf16ToFloat(__ldg(delta + row * Q38_HIDDEN + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38PoolKernel(const uint32_t *__restrict__ pooled, const uint16_t *__restrict__ normed, const uint16_t *__restrict__ head,
	uint32_t *__restrict__ candidates)
{
	LmDependentWait();
	__shared__ uint32_t tokens[Q38_POOL];
	__shared__ uint32_t sorted[Q38_POOL];
	__shared__ float exact[Q38_POOL];
	__shared__ uint32_t chosen[Q38_LANES];
	const uint32_t row = blockIdx.x, warp = threadIdx.x / Q38_WARP, lane = threadIdx.x % Q38_WARP;
	const uint16_t *own = normed + (uint64_t)(row + 1u) * Q38_HIDDEN;
	if ( threadIdx.x < Q38_POOL )
		tokens[threadIdx.x] = pooled[(uint64_t)row * Q38_POOL + threadIdx.x];
	__syncthreads();
	if ( threadIdx.x < Q38_POOL )
	{
		uint32_t rank = 0u;
		for ( uint32_t other = 0u; other < Q38_POOL; other++ )
			rank += tokens[other] < tokens[threadIdx.x] ? 1u : 0u;
		sorted[rank] = tokens[threadIdx.x];
	}
	__syncthreads();
	for ( uint32_t item = warp; item < Q38_POOL; item += Q38_WARPS )
	{
		const uint16_t *weights = head + (uint64_t)sorted[item] * Q38_HIDDEN;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_HIDDEN; index += Q38_WARP )
			dot += LmBf16ToFloat(__ldg(own + index)) * LmBf16ToFloat(__ldg(weights + index));
		dot = Q38WarpSum(dot);
		if ( lane == 0u )
			exact[item] = LmBf16ToFloat(LmFloatToBf16(dot));
	}
	__syncthreads();
	if ( threadIdx.x < Q38_POOL )
	{
		const float value = exact[threadIdx.x];
		uint32_t rank = 0u;
		for ( uint32_t other = 0u; other < Q38_POOL; other++ )
			rank += exact[other] > value || (exact[other] == value && other < threadIdx.x) ? 1u : 0u;
		if ( rank < Q38_LANES )
			chosen[rank] = sorted[threadIdx.x];
	}
	__syncthreads();
	if ( threadIdx.x < Q38_LANES )
	{
		uint32_t rank = 0u;
		for ( uint32_t other = 0u; other < Q38_LANES; other++ )
			rank += chosen[other] < chosen[threadIdx.x] ? 1u : 0u;
		candidates[(uint64_t)row * Q38_LANES + rank] = chosen[threadIdx.x];
	}
	LmDependentRelease();
}

static __device__ __forceinline__ uint32_t Q38Sender(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, uint32_t row,
	uint32_t lane)
{
	return row == 0u ? draft->anchor : candidates[(uint64_t)(row - 1u) * Q38_LANES + lane];
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38GreedyTableKernel(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ raw,
	const float *__restrict__ rms, const uint16_t *__restrict__ query, const uint16_t *__restrict__ norm, const uint16_t *__restrict__ head,
	const uint16_t *__restrict__ tg, const uint16_t *__restrict__ tsv, const uint16_t *__restrict__ bias, uint32_t *__restrict__ table,
	uint32_t *__restrict__ alt_table, float *__restrict__ confidence, float *__restrict__ alt_confidence)
{
	LmDependentWait();
	__shared__ float base[Q38_LANES];
	__shared__ float message[Q38_LANES][Q38_RANK];
	__shared__ float scores[Q38_LANES][Q38_LANES];
	const uint32_t row = blockIdx.x, warp = threadIdx.x / Q38_WARP, lane = threadIdx.x % Q38_WARP, source = row + 1u;
	const float scale = rms[source];
	for ( uint32_t item = warp; item < Q38_LANES; item += Q38_WARPS )
	{
		const uint16_t *weights = head + (uint64_t)candidates[(uint64_t)row * Q38_LANES + item] * Q38_HIDDEN;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_HIDDEN; index += Q38_WARP )
			dot += LmBf16ToFloat(__ldg(weights + index)) * (LmBf16ToFloat(__ldg(raw + (uint64_t)source * Q38_HIDDEN + index)) * LmBf16ToFloat(__ldg(norm + index)));
		dot = Q38WarpSum(dot);
		if ( lane == 0u )
			base[item] = dot;
	}
	for ( uint32_t index = threadIdx.x; index < Q38_LANES * Q38_RANK; index += Q38_THREADS )
	{
		const uint32_t sender = index / Q38_RANK, column = index % Q38_RANK;
		const uint16_t *pair = tsv + (uint64_t)Q38Sender(draft, candidates, row, sender) * (2u * Q38_RANK);
		message[sender][column] = Q38Sigmoid(LmBf16ToFloat(__ldg(query + (uint64_t)source * Q38_PACKED + column)) + LmBf16ToFloat(__ldg(pair + column)) +
			LmBf16ToFloat(__ldg(bias + column))) * LmBf16ToFloat(__ldg(pair + Q38_RANK + column));
	}
	__syncthreads();
	for ( uint32_t item = warp; item < Q38_LANES * Q38_LANES; item += Q38_WARPS )
	{
		const uint32_t sender = item / Q38_LANES, choice = item % Q38_LANES;
		const uint16_t *weights = tg + (uint64_t)candidates[(uint64_t)row * Q38_LANES + choice] * Q38_RANK;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_RANK; index += Q38_WARP )
			dot += message[sender][index] * LmBf16ToFloat(__ldg(weights + index));
		dot = Q38WarpSum(dot);
		if ( lane == 0u )
			scores[sender][choice] = base[choice] + scale * dot;
	}
	__syncthreads();
	if ( threadIdx.x < Q38_LANES )
	{
		const uint32_t sender = threadIdx.x;
		const float divisor = fmaxf(scale, Q38_CONFIDENCE_FLOOR);
		float best = scores[sender][0], second = -INFINITY, total = 0.0f;
		uint32_t pick = 0u, alt_pick = 1u;
		for ( uint32_t choice = 1u; choice < Q38_LANES; choice++ )
		{
			const float value = scores[sender][choice];
			if ( value > best )
			{
				second = best;
				alt_pick = pick;
				best = value;
				pick = choice;
			}
			else if ( value > second )
			{
				second = value;
				alt_pick = choice;
			}
		}
		for ( uint32_t choice = 0u; choice < Q38_LANES; choice++ )
			total += expf((scores[sender][choice] - best) / divisor);
		table[(uint64_t)row * Q38_LANES + sender] = pick;
		alt_table[(uint64_t)row * Q38_LANES + sender] = alt_pick;
		confidence[(uint64_t)row * Q38_LANES + sender] = 1.0f / total;
		alt_confidence[(uint64_t)row * Q38_LANES + sender] = expf((second - best) / divisor) / total;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_THREADS, 2)
void Q38RefinedKernel(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ raw,
	const float *__restrict__ rms, const uint16_t *__restrict__ query, const uint16_t *__restrict__ codebook, const uint16_t *__restrict__ sender_weight,
	const uint16_t *__restrict__ value_weight, const uint16_t *__restrict__ bias, const uint16_t *__restrict__ write, const uint16_t *__restrict__ norm,
	const uint16_t *__restrict__ head, float epsilon, float *__restrict__ refined_table)
{
	LmDependentWait();
	__shared__ float codes[Q38_CODE];
	__shared__ float message[Q38_RANK];
	__shared__ float refined[Q38_HIDDEN];
	__shared__ float reduction[Q38_WARPS];
	const uint32_t row = blockIdx.x / Q38_LANES, sender = blockIdx.x % Q38_LANES, source = row + 1u;
	if ( row == 0u && sender != 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t warp = threadIdx.x / Q38_WARP, lane = threadIdx.x % Q38_WARP, token = Q38Sender(draft, candidates, row, sender);
	const float root = sqrtf((float)Q38_CODE), scale = rms[source];
	float total = 0.0f, inverse;
	for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += Q38_THREADS )
		codes[index] = root * LmBf16ToFloat(__ldg(codebook + (uint64_t)token * Q38_CODE + index));
	__syncthreads();
	for ( uint32_t output = warp; output < Q38_RANK; output += Q38_WARPS )
	{
		const uint16_t *gate_row = sender_weight + (uint64_t)output * Q38_CODE, *value_row = value_weight + (uint64_t)output * Q38_CODE;
		float gate = 0.0f, value = 0.0f;
		for ( uint32_t index = lane; index < Q38_CODE; index += Q38_WARP )
		{
			const float c = codes[index];
			gate += c * LmBf16ToFloat(__ldg(gate_row + index));
			value += c * LmBf16ToFloat(__ldg(value_row + index));
		}
		gate = Q38WarpSum(gate);
		value = Q38WarpSum(value);
		if ( lane == 0u )
			message[output] = Q38Sigmoid(LmBf16ToFloat(__ldg(query + (uint64_t)source * Q38_PACKED + output)) + gate + LmBf16ToFloat(__ldg(bias + output))) * value;
	}
	__syncthreads();
	for ( uint32_t output = warp; output < Q38_HIDDEN; output += Q38_WARPS )
	{
		const uint16_t *write_row = write + (uint64_t)output * Q38_RANK;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_RANK; index += Q38_WARP )
			dot += message[index] * LmBf16ToFloat(__ldg(write_row + index));
		dot = Q38WarpSum(dot);
		if ( lane == 0u )
			refined[output] = LmBf16ToFloat(__ldg(raw + (uint64_t)source * Q38_HIDDEN + output)) + scale * dot;
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
		total += refined[index] * refined[index];
	inverse = 1.0f / sqrtf(LmBlockSum<Q38_THREADS>(total, reduction) / (float)Q38_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += Q38_THREADS )
		refined[index] = refined[index] * inverse * LmBf16ToFloat(__ldg(norm + index));
	__syncthreads();
	for ( uint32_t choice = warp; choice < Q38_LANES; choice += Q38_WARPS )
	{
		const uint16_t *weights = head + (uint64_t)candidates[(uint64_t)row * Q38_LANES + choice] * Q38_HIDDEN;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_HIDDEN; index += Q38_WARP )
			dot += refined[index] * LmBf16ToFloat(__ldg(weights + index));
		dot = Q38WarpSum(dot);
		if ( lane == 0u )
			refined_table[((uint64_t)row * Q38_LANES + sender) * Q38_LANES + choice] = dot;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(Q38_WARP, 1)
void Q38ChainKernel(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, const uint32_t *__restrict__ table,
	const uint32_t *__restrict__ alt_table, const float *__restrict__ confidence, const float *__restrict__ alt_confidence,
	const float *__restrict__ refined_table, Q38Picks *__restrict__ picks)
{
	LmDependentWait();
	const uint32_t lane = threadIdx.x;
	const uint32_t tree = draft->tree;
	uint32_t choice = 0u, previous = 0u, sampled_lane = 0u;
	if ( draft->greedy != 0u || lane < Q38_LANES )
	{
		if ( lane == 0u || draft->greedy == 0u )
			choice = table[0];
		for ( uint32_t row = 0u; row < Q38_DRAFTS; row++ )
		{
			const uint32_t greedy_token = candidates[(uint64_t)row * Q38_LANES + choice];
			const float score = confidence[(uint64_t)row * Q38_LANES + previous];
			uint32_t token = greedy_token;
			uint32_t alt_lane = alt_table[(uint64_t)row * Q38_LANES + previous];
			uint32_t alt_token = candidates[(uint64_t)row * Q38_LANES + alt_lane];
			float alt_score = alt_confidence[(uint64_t)row * Q38_LANES + previous];
			if ( draft->greedy == 0u )
			{
				const float value = refined_table[((uint64_t)row * Q38_LANES + sampled_lane) * Q38_LANES + lane] * draft->inverse_temperature;
				const uint32_t candidate = candidates[(uint64_t)row * Q38_LANES + lane];
				const uint32_t active = (1u << Q38_LANES) - 1u;
				float top = value, sum, probability, before = 0.0f;
				uint32_t rank = 0u, winner_rank, runner_rank;
				float alt_prob = 0.0f;
				unsigned long long key;
				alt_lane = 0u;
				for ( uint32_t other = 0u; other < Q38_LANES; other++ )
				{
					const float other_value = __shfl_sync(active, value, other);
					rank += other_value > value || (other_value == value && other < lane) ? 1u : 0u;
				}
				for ( uint32_t mask = Q38_LANES / 2u; mask > 0u; mask >>= 1u )
					top = fmaxf(top, __shfl_xor_sync(active, top, mask));
				probability = expf(value - top);
				sum = probability;
				for ( uint32_t mask = Q38_LANES / 2u; mask > 0u; mask >>= 1u )
					sum += __shfl_xor_sync(active, sum, mask);
				probability /= sum;
				for ( uint32_t other = 0u; other < Q38_LANES; other++ )
				{
					const float other_probability = __shfl_sync(active, probability, other);
					const float other_value = __shfl_sync(active, value, other);
					if ( other_value > value || (other_value == value && other < lane) )
						before += other_probability;
				}
				key = (((draft->nucleus <= 0.0f) || (before < draft->nucleus)) && (draft->top_k == 0u || rank < draft->top_k)) || rank == 0u ?
					((unsigned long long)SparkMathSampleOrdered(value + SparkMathGumbelNoise(draft->seed, draft->positions[0] + row, candidate)) << 32u) |
					(unsigned long long)(0xffffffffu - rank) : 0ull;
				for ( uint32_t mask = Q38_LANES / 2u; mask > 0u; mask >>= 1u )
				{
					const unsigned long long other = __shfl_xor_sync(active, key, mask);
					key = other > key ? other : key;
				}
				winner_rank = 0xffffffffu - (uint32_t)key;
				runner_rank = winner_rank == 0u ? 1u : 0u;
				for ( uint32_t other = 0u; other < Q38_LANES; other++ )
				{
					const uint32_t other_rank = __shfl_sync(active, rank, other);
					const float other_prob = __shfl_sync(active, probability, other);
					if ( other_rank == winner_rank )
						sampled_lane = other;
					if ( other_rank == runner_rank )
					{
						alt_lane = other;
						alt_prob = other_prob;
					}
				}
				token = candidates[(uint64_t)row * Q38_LANES + sampled_lane];
				alt_token = candidates[(uint64_t)row * Q38_LANES + alt_lane];
				alt_score = alt_prob;
			}
			if ( lane == 0u )
			{
				picks->tokens[row] = token;
				picks->scores[row] = score;
				picks->branch_tokens[row][0] = alt_token;
				picks->branch_scores[row][0] = alt_score;
			}
			if ( tree != 0u )
			{
				uint32_t branch_lane = alt_lane;
				for ( uint32_t k = 1u; k < Q38_BRANCH_LEN && row + k < Q38_DRAFTS; k++ )
				{
					if ( draft->greedy != 0u )
					{
						if ( lane == 0u )
						{
							const uint64_t next_at = (uint64_t)(row + k) * Q38_LANES + branch_lane;
							branch_lane = table[next_at];
							picks->branch_tokens[row][k] = candidates[(uint64_t)(row + k) * Q38_LANES + branch_lane];
							picks->branch_scores[row][k] = confidence[next_at];
						}
					}
					else
					{
						const uint32_t active = (1u << Q38_LANES) - 1u;
						const float value = refined_table[((uint64_t)(row + k) * Q38_LANES + branch_lane) * Q38_LANES + lane] * draft->inverse_temperature;
						float top = value, denom;
						unsigned long long key = ((unsigned long long)SparkMathSampleOrdered(value) << 32u) | (unsigned long long)(0xffffffffu - lane);
						for ( uint32_t mask = Q38_LANES / 2u; mask > 0u; mask >>= 1u )
						{
							const unsigned long long other = __shfl_xor_sync(active, key, mask);
							key = other > key ? other : key;
							top = fmaxf(top, __shfl_xor_sync(active, top, mask));
						}
						denom = expf(value - top);
						for ( uint32_t mask = Q38_LANES / 2u; mask > 0u; mask >>= 1u )
							denom += __shfl_xor_sync(active, denom, mask);
						branch_lane = 0xffffffffu - (uint32_t)key;
						if ( lane == 0u )
						{
							picks->branch_tokens[row][k] = candidates[(uint64_t)(row + k) * Q38_LANES + branch_lane];
							picks->branch_scores[row][k] = denom > 0.0f ? 1.0f / denom : 0.0f;
						}
					}
				}
			}
			previous = choice;
			if ( row + 1u < Q38_DRAFTS )
				choice = table[(uint64_t)(row + 1u) * Q38_LANES + choice];
		}
	}
	LmDependentRelease();
}

typedef struct Q38Forward
{
	Q38Draft *host_draft;
	Q38Draft *draft;
	Q38Picks *host_picks;
	Q38Picks *picks;
	float *hidden_f32;
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
	uint16_t *raw;
	float *rms;
	uint16_t *bridge_normed;
	uint16_t *codes_projected;
	uint16_t *codes;
	uint16_t *packed;
	uint16_t *message;
	uint16_t *delta;
	uint16_t *normed;
	uint16_t *y;
	uint16_t *base;
	SparkDrafterTopkBf16Workspace topk_ws;
	uint32_t *pooled;
	uint32_t *candidates;
	uint32_t *table;
	uint32_t *alt_table;
	float *confidence;
	float *alt_confidence;
	float *refined_table;
} Q38Forward;

typedef struct Q38State
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	uint32_t mask_token;
	float epsilon;
	float bridge_epsilon;
	float softmax_scale;
	float *rope_inverse;
	const uint16_t *fc;
	const uint16_t *hidden_norm;
	const uint16_t *norm;
	const uint16_t *embed;
	const uint16_t *head;
	const uint16_t *codebook;
	const uint16_t *gate_bias;
	const uint16_t *bridge_gate;
	const uint16_t *bridge_sender;
	const uint16_t *bridge_value;
	Q38Layer layers[Q38_LAYERS];
	uint16_t *packed_weight;
	uint16_t *write_transposed;
	uint16_t *tg;
	uint16_t *tsv;
	Q38Forward forward;
	SparkDrafterGraphCache graphs;
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
	Q38Sequence *sequences;
	uint64_t frames;
	uint64_t rows;
	uint64_t gaps;
} Q38State;

__global__ __launch_bounds__(Q38_THREADS, 1)
void Q38ScaledRowsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, float factor, uint32_t width, uint64_t count,
	uint16_t *__restrict__ output)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * Q38_THREADS + threadIdx.x; index < count; index += (uint64_t)gridDim.x * Q38_THREADS )
		output[index] = LmFloatToBf16(LmBf16ToFloat(input[index]) * (weight != 0 ? LmBf16ToFloat(weight[index % width]) : factor));
}

__global__ __launch_bounds__(Q38_THREADS, 1)
void Q38TransposeKernel(const uint16_t *__restrict__ input, uint32_t rows, uint32_t columns, uint16_t *__restrict__ output)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * Q38_THREADS + threadIdx.x; index < (uint64_t)rows * columns; index += (uint64_t)gridDim.x * Q38_THREADS )
		output[(index % columns) * rows + index / columns] = input[index];
}

__global__ __launch_bounds__(Q38_THREADS, 1)
void Q38RopeTableKernel(float *__restrict__ inverse, float theta)
{
	if ( threadIdx.x < Q38_HALF )
		inverse[threadIdx.x] = 1.0f / powf(theta, (float)(2u * threadIdx.x) / (float)Q38_HEAD_DIM);
}

static SparkStatus Q38Refuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	return status;
}

static SparkStatus Q38LoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
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
	if ( magic != Q38_PACK_MAGIC || version != Q38_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
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
	*payload_base = (sizeof(header) + bytes + Q38_PACK_ALIGNMENT - 1u) / Q38_PACK_ALIGNMENT * Q38_PACK_ALIGNMENT;
	return status;
}

static SparkStatus Q38Tensor(const Q38State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t expected,
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
		fprintf(stderr, "qwen38 dspine: pack tensor %s is missing or is not %llu bf16 elements\n", name, (unsigned long long)expected);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	*pointer = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static int32_t Q38Member(const SparkJsonDocument *document, int32_t object, const char *name)
{
	return object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
}

static SparkStatus Q38Shape(Q38State *state, const SparkJsonDocument *document)
{
	static const struct
	{
		const char *name;
		uint64_t value;
	} expected[] = {
		{"hidden_size", Q38_HIDDEN}, {"num_hidden_layers", Q38_LAYERS}, {"num_attention_heads", Q38_HEADS}, {"num_key_value_heads", Q38_KV_HEADS},
		{"head_dim", Q38_HEAD_DIM}, {"intermediate_size", Q38_MLP}, {"block_size", Q38_BLOCK}, {"sliding_window", Q38_SLIDING},
		{"vocab_size", Q38_VOCAB},
	};
	const int32_t config = Q38Member(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = Q38Member(document, config, "checkpoint_config");
	const int32_t dflash = Q38Member(document, checkpoint, "dflash_config");
	const int32_t bridge = Q38Member(document, dflash, "selected_prefix_code_bridge");
	const int32_t second = Q38Member(document, checkpoint, "code_bridge_second_pass");
	const int32_t rope = Q38Member(document, checkpoint, "rope_parameters");
	const struct
	{
		int32_t object;
		const char *name;
		uint64_t value;
	} nested[] = {
		{bridge, "code_size", Q38_CODE}, {bridge, "rank", Q38_RANK}, {second, "head_topk", Q38_LANES}, {second, "first_exact", Q38_LAYERS - 1u},
	};
	uint64_t value = 0u;
	float theta = 0.0f;
	int32_t token;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		token = Q38Member(document, checkpoint, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "qwen38 dspine: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			return SPARK_STATUS_SCHEMA_ERROR;
		}
	}
	for ( uint32_t index = 0u; index < sizeof(nested) / sizeof(nested[0]); index++ )
	{
		token = Q38Member(document, nested[index].object, nested[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != nested[index].value )
		{
			fprintf(stderr, "qwen38 dspine: the code bridge's %s is not %llu\n", nested[index].name, (unsigned long long)nested[index].value);
			return SPARK_STATUS_SCHEMA_ERROR;
		}
	}
	token = Q38Member(document, dflash, "mask_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= Q38_VOCAB )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->mask_token = (uint32_t)value;
	token = Q38Member(document, checkpoint, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	token = Q38Member(document, bridge, "eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->bridge_epsilon) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	token = Q38Member(document, rope, "rope_theta");
	if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->softmax_scale = (float)(1.0 / sqrt((double)Q38_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, Q38_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	LM_LAUNCH((Q38RopeTableKernel), 1u, Q38_THREADS, 0, state->stream, state->rope_inverse, theta);
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Q38Weights(Q38State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	const uint16_t *early = 0, *last_read = 0, *writes = 0;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = Q38LoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = Q38Shape(state, &document);
	{
		const struct
		{
			const char *name;
			uint64_t elements;
			const uint16_t **pointer;
		} plain[] = {
			{"fc.weight", (uint64_t)Q38_HIDDEN * Q38_TAP_WIDTH, &state->fc}, {"hidden_norm.weight", Q38_HIDDEN, &state->hidden_norm},
			{"norm.weight", Q38_HIDDEN, &state->norm}, {"target.embed", (uint64_t)Q38_VOCAB * Q38_HIDDEN, &state->embed},
			{"target.head", (uint64_t)Q38_VOCAB * Q38_HIDDEN, &state->head},
			{"selected_prefix_bridge.codebook", (uint64_t)Q38_VOCAB * Q38_CODE, &state->codebook},
			{"selected_prefix_bridge.early_read_weights", (uint64_t)Q38_EARLY * Q38_CODE * Q38_HIDDEN, &early},
			{"selected_prefix_bridge.read_weight", (uint64_t)Q38_CODE * Q38_HIDDEN, &last_read},
			{"selected_prefix_bridge.feature_write_weights", (uint64_t)Q38_LAYERS * Q38_HIDDEN * Q38_RANK, &writes},
			{"selected_prefix_bridge.gate.weight", (uint64_t)Q38_RANK * Q38_CODE, &state->bridge_gate},
			{"selected_prefix_bridge.gate.bias", Q38_RANK, &state->gate_bias},
			{"selected_prefix_bridge.predecessor_gate.weight", (uint64_t)Q38_RANK * Q38_CODE, &state->bridge_sender},
			{"selected_prefix_bridge.prev.weight", (uint64_t)Q38_RANK * Q38_CODE, &state->bridge_value},
		};
		for ( uint32_t index = 0u; index < sizeof(plain) / sizeof(plain[0]) && status == SPARK_STATUS_OK; index++ )
			status = Q38Tensor(state, &document, payload_base, plain[index].name, plain[index].elements, plain[index].pointer);
	}
	for ( uint32_t layer = 0u; layer < Q38_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		Q38Layer *weights = &state->layers[layer];
		const struct
		{
			const char *suffix;
			uint64_t elements;
			const uint16_t **pointer;
		} tensors[] = {
			{"input_layernorm.weight", Q38_HIDDEN, &weights->input_norm}, {"post_attention_layernorm.weight", Q38_HIDDEN, &weights->post_norm},
			{"self_attn.q_proj.weight", (uint64_t)Q38_Q_WIDTH * Q38_HIDDEN, &weights->q}, {"self_attn.k_proj.weight", (uint64_t)Q38_KV_WIDTH * Q38_HIDDEN, &weights->k},
			{"self_attn.v_proj.weight", (uint64_t)Q38_KV_WIDTH * Q38_HIDDEN, &weights->v}, {"self_attn.o_proj.weight", (uint64_t)Q38_HIDDEN * Q38_Q_WIDTH, &weights->o},
			{"self_attn.q_norm.weight", Q38_HEAD_DIM, &weights->q_norm}, {"self_attn.k_norm.weight", Q38_HEAD_DIM, &weights->k_norm},
			{"mlp.gate_proj.weight", (uint64_t)Q38_MLP * Q38_HIDDEN, &weights->gate}, {"mlp.up_proj.weight", (uint64_t)Q38_MLP * Q38_HIDDEN, &weights->up},
			{"mlp.down_proj.weight", (uint64_t)Q38_HIDDEN * Q38_MLP, &weights->down},
		};
		for ( uint32_t index = 0u; index < sizeof(tensors) / sizeof(tensors[0]) && status == SPARK_STATUS_OK; index++ )
		{
			char name[128];
			snprintf(name, sizeof(name), "layers.%u.%s", layer, tensors[index].suffix);
			status = Q38Tensor(state, &document, payload_base, name, tensors[index].elements, tensors[index].pointer);
		}
		if ( status == SPARK_STATUS_OK )
		{
			weights->read = layer < Q38_EARLY ? early + (uint64_t)layer * Q38_CODE * Q38_HIDDEN : last_read;
			weights->write = writes + (uint64_t)layer * Q38_HIDDEN * Q38_RANK;
		}
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define Q38_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus Q38Gemm(Q38State *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( rows == Q38_BLOCK && output_dimension >= Q38_STREAM_OUTPUTS )
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u, 0u,
			state->multiprocessors, state->stream) == LM_LAUNCH_OK ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u,
		state->multiprocessors, state->stream) == LM_LAUNCH_OK ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Q38Tables(Q38State *state)
{
	const uint32_t chunk = 8192u;
	uint16_t *scaled = 0;
	SparkStatus status = SPARK_STATUS_OK;
	if ( !Q38_DEVICE(state->packed_weight, (uint64_t)Q38_PACKED * Q38_CODE) || !Q38_DEVICE(state->write_transposed, (uint64_t)Q38_RANK * Q38_HIDDEN) ||
		!Q38_DEVICE(state->tg, (uint64_t)Q38_VOCAB * Q38_RANK) || !Q38_DEVICE(state->tsv, (uint64_t)Q38_VOCAB * 2u * Q38_RANK) ||
		!Q38_DEVICE(scaled, (uint64_t)chunk * Q38_HIDDEN) )
	{
		(void)SparkGpuFree(scaled);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( SparkGpuMemcpyAsync(state->packed_weight, state->bridge_gate, (size_t)Q38_RANK * Q38_CODE * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE,
		state->stream) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpyAsync(state->packed_weight + (uint64_t)Q38_RANK * Q38_CODE, state->bridge_sender, (size_t)Q38_RANK * Q38_CODE * sizeof(uint16_t),
		SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpyAsync(state->packed_weight + 2u * (uint64_t)Q38_RANK * Q38_CODE, state->bridge_value, (size_t)Q38_RANK * Q38_CODE * sizeof(uint16_t),
		SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((Q38TransposeKernel), 4u * state->multiprocessors, Q38_THREADS, 0, state->stream, state->layers[Q38_LAYERS - 1u].write, Q38_HIDDEN, Q38_RANK,
			state->write_transposed);
	for ( uint32_t first = 0u; first < Q38_VOCAB && status == SPARK_STATUS_OK; first += chunk )
	{
		const uint32_t rows = Q38_VOCAB - first < chunk ? Q38_VOCAB - first : chunk;
		LM_LAUNCH((Q38ScaledRowsKernel), 4u * state->multiprocessors, Q38_THREADS, 0, state->stream, state->head + (uint64_t)first * Q38_HIDDEN, state->norm,
			1.0f, Q38_HIDDEN, (uint64_t)rows * Q38_HIDDEN, scaled);
		status = Q38Gemm(state, state->write_transposed, scaled, state->tg + (uint64_t)first * Q38_RANK, rows, Q38_HIDDEN, Q38_RANK);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38ScaledRowsKernel), 4u * state->multiprocessors, Q38_THREADS, 0, state->stream, state->codebook + (uint64_t)first * Q38_CODE,
			(const uint16_t *)0, sqrtf((float)Q38_CODE), Q38_CODE, (uint64_t)rows * Q38_CODE, scaled);
		status = Q38Gemm(state, state->packed_weight + (uint64_t)Q38_RANK * Q38_CODE, scaled, state->tsv + (uint64_t)first * 2u * Q38_RANK, rows, Q38_CODE,
			2u * Q38_RANK);
	}
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	(void)SparkGpuFree(scaled);
	return status;
}

static SparkStatus Q38Scratch(Q38State *state, uint64_t context_bytes)
{
	Q38Forward *forward = &state->forward;
	int multiprocessors = 0;
	uint64_t capacity = context_bytes / Q38_SEQUENCE_BYTES;
	capacity = capacity < 1u ? 1u : capacity > Q38_SEQUENCES_MAX ? Q38_SEQUENCES_MAX : capacity;
	state->capacity = (uint32_t)capacity;
	state->sequences = (Q38Sequence *)calloc(capacity, sizeof(Q38Sequence));
	if ( state->sequences == 0 || SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)Q38_INGEST_ROWS * Q38_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)Q38_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(Q38Draft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(Q38Picks)) != SPARK_GPU_SUCCESS ||
		!Q38_DEVICE(state->caches, capacity * Q38_SEQUENCE_ELEMENTS) || !Q38_DEVICE(state->taps, (uint64_t)Q38_INGEST_ROWS * Q38_TAP_WIDTH) ||
		!Q38_DEVICE(state->positions, Q38_INGEST_ROWS) || !Q38_DEVICE(state->features_raw, (uint64_t)Q38_INGEST_ROWS * Q38_HIDDEN) ||
		!Q38_DEVICE(state->features, (uint64_t)Q38_INGEST_ROWS * Q38_HIDDEN) || !Q38_DEVICE(state->keys_raw, (uint64_t)Q38_INGEST_ROWS * Q38_KV_WIDTH) ||
		!Q38_DEVICE(state->values_raw, (uint64_t)Q38_INGEST_ROWS * Q38_KV_WIDTH) || !Q38_DEVICE(forward->draft, 1u) || !Q38_DEVICE(forward->picks, 1u) ||
		!Q38_DEVICE(forward->hidden_f32, Q38_BLOCK * Q38_HIDDEN) || !Q38_DEVICE(forward->hidden, Q38_BLOCK * Q38_HIDDEN) ||
		!Q38_DEVICE(forward->x, Q38_BLOCK * Q38_HIDDEN) || !Q38_DEVICE(forward->q, Q38_BLOCK * Q38_Q_WIDTH) ||
		!Q38_DEVICE(forward->q_roped, Q38_BLOCK * Q38_Q_WIDTH) || !Q38_DEVICE(forward->k, Q38_BLOCK * Q38_KV_WIDTH) ||
		!Q38_DEVICE(forward->v, Q38_BLOCK * Q38_KV_WIDTH) || !Q38_DEVICE(forward->partials, (uint64_t)Q38_KV_HEADS * Q38_SPLITS * Q38_QUERIES * (Q38_HEAD_DIM + 2u)) ||
		!Q38_DEVICE(forward->attended, Q38_BLOCK * Q38_Q_WIDTH) || !Q38_DEVICE(forward->projected, Q38_BLOCK * Q38_HIDDEN) ||
		!Q38_DEVICE(forward->residual, Q38_BLOCK * Q38_HIDDEN) || !Q38_DEVICE(forward->gate, Q38_BLOCK * Q38_MLP) ||
		!Q38_DEVICE(forward->up, Q38_BLOCK * Q38_MLP) || !Q38_DEVICE(forward->act, Q38_BLOCK * Q38_MLP) || !Q38_DEVICE(forward->mlp, Q38_BLOCK * Q38_HIDDEN) ||
		!Q38_DEVICE(forward->raw, Q38_BLOCK * Q38_HIDDEN) || !Q38_DEVICE(forward->rms, Q38_BLOCK) || !Q38_DEVICE(forward->bridge_normed, Q38_BLOCK * Q38_HIDDEN) ||
		!Q38_DEVICE(forward->codes_projected, Q38_BLOCK * Q38_CODE) || !Q38_DEVICE(forward->codes, Q38_BLOCK * Q38_CODE) ||
		!Q38_DEVICE(forward->packed, Q38_BLOCK * Q38_PACKED) || !Q38_DEVICE(forward->message, Q38_BLOCK * Q38_RANK) ||
		!Q38_DEVICE(forward->delta, Q38_BLOCK * Q38_HIDDEN) || !Q38_DEVICE(forward->normed, Q38_BLOCK * Q38_HIDDEN) ||
		!Q38_DEVICE(forward->y, Q38_BLOCK * Q38_HIDDEN) || !Q38_DEVICE(forward->base, (uint64_t)Q38_BLOCK * Q38_VOCAB) ||
		SparkDrafterTopkBf16Alloc(&forward->topk_ws, Q38_DRAFTS) != SPARK_STATUS_OK || !Q38_DEVICE(forward->pooled, Q38_DRAFTS * Q38_POOL) ||
		!Q38_DEVICE(forward->candidates, Q38_DRAFTS * Q38_LANES) || !Q38_DEVICE(forward->table, Q38_DRAFTS * Q38_LANES) ||
		!Q38_DEVICE(forward->alt_table, Q38_DRAFTS * Q38_LANES) || !Q38_DEVICE(forward->confidence, Q38_DRAFTS * Q38_LANES) ||
		!Q38_DEVICE(forward->alt_confidence, Q38_DRAFTS * Q38_LANES) || !Q38_DEVICE(forward->refined_table, Q38_DRAFTS * Q38_LANES * Q38_LANES) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		state->sequences[index].cache = state->caches + (uint64_t)index * Q38_SEQUENCE_ELEMENTS;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static uint16_t *Q38Keys(const Q38Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer) * Q38_LAYER_KV;
}

static uint16_t *Q38Values(const Q38Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer + 1u) * Q38_LAYER_KV;
}

static void Q38Reset(Q38Sequence *sequence)
{
	for ( uint32_t slot = 0u; slot < Q38_WINDOW; slot++ )
	{
		sequence->positions[slot] = -1;
		sequence->tokens[slot] = 0u;
	}
	sequence->retired = 0u;
}

static Q38Sequence *Q38Live(Q38State *state, uint64_t id)
{
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		if ( state->sequences[index].live != 0u && state->sequences[index].retired == 0u && state->sequences[index].id == id )
			return &state->sequences[index];
	return 0;
}

static Q38Sequence *Q38Claim(Q38State *state)
{
	Q38Sequence *oldest = 0;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
	{
		Q38Sequence *sequence = &state->sequences[index];
		if ( sequence->live == 0u )
			return sequence;
		if ( oldest == 0 || sequence->last_used < oldest->last_used )
			oldest = sequence;
	}
	return oldest;
}

static uint32_t Q38Held(const Q38Sequence *sequence, uint64_t position)
{
	return sequence->positions[position % Q38_WINDOW] == (int64_t)position ? 1u : 0u;
}

static SparkStatus Q38IngestRows(Q38State *state, Q38Sequence *sequence, uint32_t rows)
{
	SparkStatus status = Q38Gemm(state, state->fc, state->taps, state->features_raw, rows, Q38_TAP_WIDTH, Q38_HIDDEN);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((Q38RmsKernel), rows, Q38_THREADS, 0, state->stream, state->features_raw, (const float *)0, state->hidden_norm, state->features, Q38_HIDDEN,
			state->epsilon, 0u);
	for ( uint32_t layer = 0u; layer < Q38_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = Q38Gemm(state, state->layers[layer].k, state->features, state->keys_raw, rows, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, state->layers[layer].v, state->features, state->values_raw, rows, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38IngestKeysKernel), rows, Q38_THREADS, 0, state->stream, state->keys_raw, state->layers[layer].k_norm, state->rope_inverse, state->positions,
			state->epsilon, Q38Keys(sequence, layer));
		LM_LAUNCH((Q38StoreValuesKernel), rows, Q38_THREADS, 0, state->stream, state->values_raw, state->positions, (const Q38Draft *)0, 0u,
			Q38Values(sequence, layer));
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	return status;
}

static SparkStatus Q38Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	Q38State *state = (Q38State *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	const uint32_t skip = rows > Q38_WINDOW ? rows - Q38_WINDOW : 0u;
	const uint64_t first = frame->header.first_position + skip;
	Q38Sequence *sequence;
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != Q38_TAPS || hidden != Q38_HIDDEN || frame->rows == 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	for ( uint32_t tap = 0u; tap < Q38_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * Q38_HIDDEN > frame->row_elements )
			return SPARK_STATUS_SCHEMA_ERROR;
	if ( frame->header.first_position + rows > UINT32_MAX )
		return SPARK_STATUS_INVALID_ARGUMENT;
	sequence = Q38Live(state, frame->header.sequence_id);
	if ( sequence != 0 && frame->header.first_position != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence->last_used = ++state->clock;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = Q38Claim(state);
		Q38Reset(sequence);
		sequence->live = 1u;
		sequence->id = frame->header.sequence_id;
	}
	else if ( frame->header.first_position == 0u )
		Q38Reset(sequence);
	sequence->control_generation = frame->header.control_generation;
	sequence->request_id = frame->header.request_id;
	sequence->last_used = ++state->clock;
	for ( uint32_t row = skip; row < rows; row++ )
		sequence->positions[(frame->header.first_position + row) % Q38_WINDOW] = -1;
	for ( uint32_t done = skip; done < rows && status == SPARK_STATUS_OK; done += Q38_INGEST_ROWS )
	{
		const uint32_t part = rows - done < Q38_INGEST_ROWS ? rows - done : Q38_INGEST_ROWS;
		for ( uint32_t row = 0u; row < part; row++ )
		{
			const uint16_t *source = frame->rows + (uint64_t)(done + row) * frame->row_elements;
			for ( uint32_t tap = 0u; tap < Q38_TAPS; tap++ )
				memcpy(state->host_taps + (uint64_t)row * Q38_TAP_WIDTH + (uint64_t)tap * Q38_HIDDEN,
					source + (uint64_t)frame->required_index[tap] * Q38_HIDDEN, Q38_HIDDEN * sizeof(uint16_t));
			state->host_positions[row] = (uint32_t)(frame->header.first_position + done + row);
		}
		if ( SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)part * Q38_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)part * sizeof(uint32_t),
			SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		{
			(void)SparkGpuStreamSynchronize(state->stream);
			status = SPARK_STATUS_IO_ERROR;
			break;
		}
		status = Q38IngestRows(state, sequence, part);
	}
	if ( status == SPARK_STATUS_OK )
	{
		for ( uint32_t slot = 0u; slot < Q38_WINDOW; slot++ )
			if ( sequence->positions[slot] >= (int64_t)first )
				sequence->positions[slot] = -1;
		for ( uint32_t row = skip; row < rows; row++ )
		{
			const uint64_t position = frame->header.first_position + row;
			sequence->positions[position % Q38_WINDOW] = (int64_t)position;
			sequence->tokens[position % Q38_WINDOW] = frame->tokens != 0 ? frame->tokens[row] : 0u;
		}
	}
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus Q38CopySlots(Q38State *state, Q38Sequence *sequence, const Q38Sequence *donor, uint32_t slot, uint32_t count)
{
	for ( uint32_t layer = 0u; layer < Q38_LAYERS; layer++ )
		for ( uint32_t head = 0u; head < Q38_KV_HEADS; head++ )
		{
			const uint64_t at = ((uint64_t)head * Q38_COLUMNS + slot) * Q38_HEAD_DIM;
			const size_t bytes = (size_t)count * Q38_HEAD_DIM * sizeof(uint16_t);
			if ( SparkGpuMemcpyAsync(Q38Keys(sequence, layer) + at, Q38Keys(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) !=
				SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(Q38Values(sequence, layer) + at, Q38Values(donor, layer) + at, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE,
				state->stream) != SPARK_GPU_SUCCESS )
				return SPARK_STATUS_IO_ERROR;
		}
	return SPARK_STATUS_OK;
}

static uint32_t Q38Usable(const Q38Sequence *sequence, const Q38Sequence *donor, const SparkDraftServerRequest *request, uint64_t position)
{
	return Q38Held(sequence, position) == 0u && Q38Held(donor, position) != 0u && position < request->committed_token_count &&
		donor->tokens[position % Q38_WINDOW] == request->committed_token_ids[position] ? 1u : 0u;
}

static SparkStatus Q38FillFromDonors(Q38State *state, Q38Sequence *sequence, const SparkDraftServerRequest *request, uint64_t first)
{
	uint8_t tried[Q38_SEQUENCES_MAX];
	memset(tried, 0, sizeof(tried));
	for ( ;; )
	{
		Q38Sequence *donor = 0;
		uint32_t chosen = 0u, missing = 0u;
		uint64_t position = first;
		for ( uint64_t at = first; at < request->position; at++ )
			missing += Q38Held(sequence, at) == 0u ? 1u : 0u;
		if ( missing == 0u )
			return SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < state->capacity; index++ )
		{
			Q38Sequence *other = &state->sequences[index];
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
			if ( Q38Usable(sequence, donor, request, position) == 0u )
			{
				position++;
				continue;
			}
			while ( stop < request->position && Q38Usable(sequence, donor, request, stop) != 0u && (stop == position || stop % Q38_WINDOW != 0u) )
				stop++;
			if ( Q38CopySlots(state, sequence, donor, (uint32_t)(position % Q38_WINDOW), (uint32_t)(stop - position)) != SPARK_STATUS_OK )
				return SPARK_STATUS_IO_ERROR;
			for ( uint64_t at = position; at < stop; at++ )
			{
				sequence->positions[at % Q38_WINDOW] = (int64_t)at;
				sequence->tokens[at % Q38_WINDOW] = donor->tokens[at % Q38_WINDOW];
			}
			position = stop;
		}
	}
}

static void Q38Span(Q38Sequence *sequence, uint64_t checked, uint64_t position, Q38Draft *draft)
{
	const uint64_t count = position - checked;
	const uint32_t first = count != 0u ? (uint32_t)(checked % Q38_WINDOW) : 0u, end = first + (uint32_t)count;
	uint32_t newer = 0u;
	memset(draft->valid, 0, sizeof(draft->valid));
	draft->masked = 0u;
	if ( count == Q38_WINDOW )
	{
		draft->start = 0u;
		draft->block_at = Q38_WINDOW;
		return;
	}
	for ( uint32_t slot = end; slot < Q38_WINDOW && slot < end + Q38_BLOCK; slot++ )
		newer |= count != 0u && sequence->positions[slot] > (int64_t)(position - 1u) ? 1u : 0u;
	if ( end <= Q38_WINDOW && newer == 0u )
	{
		for ( uint32_t slot = end; slot < Q38_WINDOW && slot < end + Q38_BLOCK; slot++ )
			sequence->positions[slot] = -1;
		draft->start = first;
		draft->block_at = end;
		return;
	}
	for ( uint64_t at = checked; at < position; at++ )
		draft->valid[at % Q38_WINDOW] = 1u;
	draft->masked = 1u;
	draft->start = 0u;
	draft->block_at = Q38_WINDOW;
}

static SparkStatus Q38Prepare(Q38State *state, Q38Sequence *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
	const SparkRowSampling *sampling, uint32_t tree, uint32_t *splits_out)
{
	Q38Draft *draft = state->forward.host_draft;
	uint32_t splits;
	if ( anchor >= Q38_VOCAB || position + Q38_BLOCK > UINT32_MAX )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(draft, 0, offsetof(Q38Draft, valid));
	draft->cache = sequence->cache;
	for ( uint32_t row = 0u; row < Q38_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	draft->anchor = anchor;
	draft->mask = state->mask_token;
	draft->greedy = sampling->inverse_temperature > 0.0f ? 0u : 1u;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	draft->tree = tree;
	Q38Span(sequence, checked, position, draft);
	splits = (draft->block_at - draft->start + Q38_SPLIT_KEYS - 1u) / Q38_SPLIT_KEYS;
	draft->splits = splits;
	if ( splits_out != 0 )
		*splits_out = splits;
	return SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) == SPARK_GPU_SUCCESS ?
		SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus Q38Bridge(Q38State *state, uint32_t layer, uint16_t *output)
{
	Q38Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	SparkStatus status;
	LM_LAUNCH((Q38ResidualBridgeScaleKernel), Q38_BLOCK, Q38_THREADS, 0, stream, forward->residual, forward->mlp, forward->raw, forward->rms,
		forward->bridge_normed, state->bridge_epsilon);
	status = Q38Gemm(state, state->layers[layer].read, forward->bridge_normed, forward->codes_projected, Q38_BLOCK, Q38_HIDDEN, Q38_CODE);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Q38BridgeCodesKernel), Q38_BLOCK, Q38_THREADS, 0, stream, forward->codes_projected, state->codebook, forward->draft, state->bridge_epsilon,
		forward->codes);
	status = Q38Gemm(state, state->packed_weight, forward->codes, forward->packed, Q38_BLOCK, Q38_CODE, Q38_PACKED);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Q38BridgeMessageKernel), Q38_BLOCK, Q38_THREADS, 0, stream, forward->packed, state->gate_bias, forward->message);
	status = Q38Gemm(state, state->layers[layer].write, forward->message, forward->delta, Q38_BLOCK, Q38_RANK, Q38_HIDDEN);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Q38BridgeOutKernel), Q38_BLOCK, Q38_THREADS, 0, stream, forward->raw, forward->rms, forward->delta, output);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38RunForward(Q38State *state, uint32_t layers, uint32_t splits)
{
	Q38Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(Q38Draft, positions));
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((Q38EmbedKernel), Q38_BLOCK, Q38_THREADS, 0, stream, state->embed, forward->draft, forward->hidden_f32);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const Q38Layer *weights = &state->layers[layer];
		const uint64_t keys_offset = (uint64_t)(2u * layer) * Q38_LAYER_KV;
		const uint64_t values_offset = (uint64_t)(2u * layer + 1u) * Q38_LAYER_KV;
		LM_LAUNCH((Q38RmsKernel), Q38_BLOCK, Q38_THREADS, 0, stream, forward->hidden, layer == 0u ? forward->hidden_f32 : (const float *)0, weights->input_norm,
			forward->x, Q38_HIDDEN, state->epsilon, layer == 0u ? 0u : 1u);
		status = Q38Gemm(state, weights->q, forward->x, forward->q, Q38_BLOCK, Q38_HIDDEN, Q38_Q_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, weights->k, forward->x, forward->k, Q38_BLOCK, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, weights->v, forward->x, forward->v, Q38_BLOCK, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38HeadNormRopeKernel), Q38_BLOCK * Q38_HEADS, Q38_HEAD_DIM, 0, stream, forward->q, weights->q_norm, state->rope_inverse, positions,
			Q38_HEADS, state->epsilon, forward->q_roped, forward->draft, 0u);
		if ( LM_LAUNCH_DEPENDENT((Q38HeadNormRopeKernel), Q38_BLOCK * Q38_KV_HEADS, Q38_HEAD_DIM, 0, stream, forward->k, weights->k_norm,
			state->rope_inverse, positions, Q38_KV_HEADS, state->epsilon, (uint16_t *)0, forward->draft, keys_offset) != SPARK_GPU_SUCCESS ||
			LM_LAUNCH_DEPENDENT((Q38StoreValuesKernel), Q38_BLOCK, Q38_THREADS, 0, stream, forward->v, (const uint32_t *)0, forward->draft,
				values_offset, (uint16_t *)0) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		if ( splits != 0u &&
			LM_LAUNCH_DEPENDENT((Q38AttentionSplitKernel), dim3(Q38_KV_HEADS, splits), Q38_ATTENTION_THREADS, 0, stream, forward->q_roped, keys_offset,
				values_offset, forward->draft, state->softmax_scale, forward->partials) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		if ( LM_LAUNCH_DEPENDENT((Q38AttentionMergeKernel), Q38_BLOCK * Q38_HEADS, Q38_HEAD_DIM, 0, stream, forward->q_roped, keys_offset,
			values_offset, forward->draft, state->softmax_scale, forward->partials, forward->attended) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		status = Q38Gemm(state, weights->o, forward->attended, forward->projected, Q38_BLOCK, Q38_Q_WIDTH, Q38_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38ResidualRmsKernel), Q38_BLOCK, Q38_THREADS, 0, stream, layer == 0u ? forward->hidden_f32 : (const float *)0,
			forward->hidden, forward->projected, weights->post_norm, forward->residual, forward->x, state->epsilon);
		status = Q38Gemm(state, weights->gate, forward->x, forward->gate, Q38_BLOCK, Q38_HIDDEN, Q38_MLP);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, weights->up, forward->x, forward->up, Q38_BLOCK, Q38_HIDDEN, Q38_MLP);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38SwigluKernel), 2u * state->multiprocessors, Q38_THREADS, 0, stream, forward->gate, forward->up, forward->act, Q38_BLOCK * Q38_MLP);
		status = Q38Gemm(state, weights->down, forward->act, forward->mlp, Q38_BLOCK, Q38_MLP, Q38_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		status = Q38Bridge(state, layer, forward->hidden);
	}
	if ( status != SPARK_STATUS_OK || layers < Q38_LAYERS )
		return status;
	if ( LM_LAUNCH_DEPENDENT((Q38DualRmsKernel), Q38_BLOCK, Q38_THREADS, 0, stream, forward->hidden, state->norm, forward->normed, forward->y,
		state->epsilon) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = Q38Gemm(state, state->head, forward->y, forward->base, Q38_BLOCK, Q38_HIDDEN, Q38_VOCAB);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Q38Heads(Q38State *state, uint32_t greedy)
{
	Q38Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	SparkStatus status = SparkDrafterTopkBf16Launch(forward->base + Q38_VOCAB, Q38_DRAFTS, Q38_VOCAB, Q38_POOL, &forward->topk_ws, forward->pooled, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( LM_LAUNCH_DEPENDENT((Q38PoolKernel), Q38_DRAFTS, Q38_THREADS, 0, stream, forward->pooled, forward->normed, state->head,
		forward->candidates) != SPARK_GPU_SUCCESS ||
		LM_LAUNCH_DEPENDENT((Q38GreedyTableKernel), Q38_DRAFTS, Q38_THREADS, 0, stream, forward->draft, forward->candidates, forward->raw, forward->rms,
			forward->packed, state->norm, state->head, state->tg, state->tsv, state->gate_bias, forward->table, forward->alt_table, forward->confidence,
			forward->alt_confidence) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( greedy == 0u &&
		LM_LAUNCH_DEPENDENT((Q38RefinedKernel), Q38_DRAFTS * Q38_LANES, Q38_THREADS, 0, stream, forward->draft, forward->candidates, forward->raw, forward->rms,
			forward->packed, state->codebook, state->bridge_sender, state->bridge_value, state->gate_bias, state->layers[Q38_LAYERS - 1u].write, state->norm,
			state->head, state->epsilon, forward->refined_table) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( LM_LAUNCH_DEPENDENT((Q38ChainKernel), 1u, Q38_WARP, 0, stream, forward->draft, forward->candidates, forward->table, forward->alt_table,
		forward->confidence, forward->alt_confidence, forward->refined_table, forward->picks) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(Q38Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Q38Settle(Q38State *state, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	return status;
}

static void Q38ForwardFree(Q38Forward *forward)
{
	void *buffers[] = {
		forward->draft, forward->picks, forward->hidden_f32, forward->hidden, forward->x, forward->q, forward->q_roped, forward->k, forward->v,
		forward->partials, forward->attended, forward->projected, forward->residual, forward->gate, forward->up, forward->act, forward->mlp, forward->raw,
		forward->rms, forward->bridge_normed, forward->codes_projected, forward->codes, forward->packed, forward->message, forward->delta, forward->normed,
		forward->y, forward->base, forward->pooled, forward->candidates, forward->table,
		forward->alt_table, forward->confidence, forward->alt_confidence, forward->refined_table,
	};
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		(void)SparkGpuFree(buffers[index]);
	SparkDrafterTopkBf16Free(&forward->topk_ws);
	if ( forward->host_draft != 0 )
		(void)SparkGpuFreeHost(forward->host_draft);
	if ( forward->host_picks != 0 )
		(void)SparkGpuFreeHost(forward->host_picks);
}

static SparkStatus Q38Close(void *opaque)
{
	Q38State *state = (Q38State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	SparkDrafterGraphCacheDestroy(&state->graphs);
	Q38ForwardFree(&state->forward);
	{
		void *buffers[] = {
			state->caches, state->taps, state->positions, state->features_raw, state->features, state->keys_raw, state->values_raw, state->packed_weight,
			state->write_transposed, state->tg, state->tsv, state->rope_inverse,
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

static SparkStatus Q38Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	Q38State *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return Q38Refuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	{
		const SparkStatus device = SparkDeviceInitialize(0);
		if ( (device != SPARK_STATUS_OK && device != SPARK_STATUS_DUPLICATE) || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
			return Q38Refuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
	}
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
	if ( lane->context_bytes < Q38_SEQUENCE_BYTES )
		return Q38Refuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one sequence's window", lane->lane_name);
	state = (Q38State *)calloc(1u, sizeof(*state));
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
	started = SparkMonotonicNs();
	status = SparkWeightdLazyPackCreate(lane->weightd_socket, &request, &options, &state->pack);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s did not bind", lane->lane_name, lane->pack_path);
		free(state);
		return status;
	}
	status = SparkGpuStreamCreateWithFlags(&state->stream, SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = Q38Weights(state, lane->pack_path);
	if ( status == SPARK_STATUS_OK )
		status = Q38Scratch(state, lane->context_bytes);
	if ( status == SPARK_STATUS_OK )
		status = Q38Tables(state);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a Qwen3.8-27B DSpine pack this module reads", lane->lane_name, lane->pack_path);
		(void)Q38Close(state);
		return status;
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s with %u sequence windows", lane->lane_name, lane->pack_path, state->capacity);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus Q38Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	Q38State *state = (Q38State *)opaque;
	Q38Sequence *sequence = Q38Live(state, request->sequence_id);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > Q38_WINDOW ? request->position - Q38_WINDOW : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( sequence == 0 || (request->position != 0u && Q38Held(sequence, request->position - 1u) == 0u) )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	status = Q38FillFromDonors(state, sequence, request, first);
	if ( status != SPARK_STATUS_OK )
		return Q38Settle(state, status);
	for ( uint64_t position = first; position < request->position; position++ )
		if ( Q38Held(sequence, position) == 0u )
			checked = position + 1u;
	if ( checked != first )
		state->gaps++;
	if ( request->committed_token_count < request->position )
		return Q38Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	for ( uint64_t position = checked; position < request->position; position++ )
		if ( sequence->tokens[position % Q38_WINDOW] != request->committed_token_ids[position] )
			return Q38Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	drafts = drafts < Q38_DRAFTS ? drafts : Q38_DRAFTS;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		const uint32_t greedy = request->sampling.inverse_temperature > 0.0f ? 0u : 1u;
		uint32_t splits = 0u;
		SparkDraftTreeBranch branches[Q38_DRAFTS];
		uint32_t branch_count = 0u;
		status = Q38Prepare(state, sequence, request->anchor_token_id, request->position, checked, &request->sampling, tree, &splits);
		if ( status == SPARK_STATUS_OK )
		{
			const uint64_t key = ((uint64_t)splits << 1u) | (uint64_t)greedy;
			status = SparkDrafterGraphRun(&state->graphs, state->stream, key, [&]() -> SparkStatus
			{
				SparkStatus inner = Q38RunForward(state, Q38_LAYERS, splits);
				if ( inner == SPARK_STATUS_OK )
					inner = Q38Heads(state, greedy);
				return inner;
			});
		}
		status = Q38Settle(state, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				const uint32_t len = drafts - step < Q38_BRANCH_LEN ? drafts - step : Q38_BRANCH_LEN;
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

static const SparkDrafterModuleInterface q38_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"qwen38-dspine",
	Q38Open,
	Q38Ingest,
	Q38Propose,
	Q38Close,
};

static SparkStatus Q38Readback(void *host, const void *device, size_t bytes)
{
	return SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &q38_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_dspine_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	Q38State *state = (Q38State *)opaque;
	const Q38Sequence *sequence = state != 0 ? Q38Live(state, sequence_id) : 0;
	SparkStatus status;
	if ( sequence == 0 || layer >= Q38_LAYERS || keys == 0 || values == 0 || positions == 0 )
		return SPARK_STATUS_NOT_FOUND;
	memcpy(positions, sequence->positions, sizeof(sequence->positions));
	status = Q38Readback(keys, Q38Keys(sequence, layer), Q38_LAYER_KV * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = Q38Readback(values, Q38Values(sequence, layer), Q38_LAYER_KV * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_dspine_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint16_t *normed, uint32_t *candidates, uint64_t *elapsed_ns)
{
	Q38State *state = (Q38State *)opaque;
	Q38Sequence *sequence = state != 0 ? Q38Live(state, sequence_id) : 0;
	const uint64_t started = SparkMonotonicNs(), first = position > Q38_WINDOW ? position - Q38_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		return SPARK_STATUS_NOT_FOUND;
	for ( uint64_t at = first; at < position; at++ )
		if ( Q38Held(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	{
		uint32_t splits = 0u;
		status = Q38Prepare(state, sequence, anchor, position, checked, &greedy, 0u, &splits);
		if ( status == SPARK_STATUS_OK )
			status = Q38RunForward(state, Q38_LAYERS, splits);
		if ( status == SPARK_STATUS_OK )
			status = Q38Heads(state, 1u);
	}
	status = Q38Settle(state, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = Q38Readback(y, state->forward.y, (size_t)Q38_BLOCK * Q38_HIDDEN * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && normed != 0 )
		status = Q38Readback(normed, state->forward.normed, (size_t)Q38_BLOCK * Q38_HIDDEN * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && candidates != 0 )
		status = Q38Readback(candidates, state->forward.candidates, (size_t)Q38_DRAFTS * Q38_LANES * sizeof(uint32_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_dspine_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	Q38State *state = (Q38State *)opaque;
	Q38Sequence *sequence = state != 0 ? Q38Live(state, sequence_id) : 0;
	const uint64_t first = position > Q38_WINDOW ? position - Q38_WINDOW : 0u;
	uint64_t checked = first;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 || hidden == 0 || layers == 0u || layers > Q38_LAYERS )
		return SPARK_STATUS_INVALID_ARGUMENT;
	for ( uint64_t at = first; at < position; at++ )
		if ( Q38Held(sequence, at) == 0u )
			checked = at + 1u;
	memset(&greedy, 0, sizeof(greedy));
	{
		uint32_t splits = 0u;
		status = Q38Prepare(state, sequence, anchor, position, checked, &greedy, 0u, &splits);
		if ( status == SPARK_STATUS_OK )
			status = Q38RunForward(state, layers, splits);
	}
	status = Q38Settle(state, status);
	if ( status == SPARK_STATUS_OK )
		status = Q38Readback(hidden, state->forward.hidden, (size_t)Q38_BLOCK * Q38_HIDDEN * sizeof(uint16_t));
	return status;
}
