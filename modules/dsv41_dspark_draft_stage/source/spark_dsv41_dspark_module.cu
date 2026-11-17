#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_draft_tree_pack.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "runtime/launch.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS41_HIDDEN 5120u
#define DS41_TAPS 3u
#define DS41_TAP_WIDTH (DS41_TAPS * DS41_HIDDEN)
#define DS41_STAGES 3u
#define DS41_LAYERS 40u
#define DS41_HEADS 64u
#define DS41_HEAD_DIM 512u
#define DS41_ROPE 64u
#define DS41_NOPE (DS41_HEAD_DIM - DS41_ROPE)
#define DS41_QUERY_RANK 1280u
#define DS41_QUERY_WIDTH (DS41_HEADS * DS41_HEAD_DIM)
#define DS41_GROUPS 8u
#define DS41_GROUP_WIDTH (DS41_QUERY_WIDTH / DS41_GROUPS)
#define DS41_O_RANK 1024u
#define DS41_O_WIDTH (DS41_GROUPS * DS41_O_RANK)
#define DS41_WINDOW 128u
#define DS41_BLOCK 5u
#define DS41_BRANCHES 2u
#define DS41_BRANCH_LEN 3u
#define DS41_VOCAB 129280u
#define DS41_MARKOV 256u
#define DS41_EXPERTS 128u
#define DS41_ACTIVE 3u
#define DS41_PAIRS (DS41_BLOCK * DS41_ACTIVE)
#define DS41_MOE 2304u
#define DS41_HC 4u
#define DS41_HC_MIX ((2u + DS41_HC) * DS41_HC)
#define DS41_HC_FLAT (DS41_HC * DS41_HIDDEN)
#define DS41_SINKHORN 20u
#define DS41_CONFIDENCE (DS41_HIDDEN + DS41_MARKOV)
#define DS41_SAMPLED 64u
#define DS41_QDQ 32u
#define DS41_INGEST_ROWS DS41_WINDOW
#define DS41_LINEAR_ROWS LM_SKINNY_ROWS_WIDE
#define DS41_THREADS 256u
#define DS41_WARP 32u
#define DS41_WARPS (DS41_THREADS / DS41_WARP)
#define DS41_PICK_THREADS 1024u
#define DS41_TOPK_THREADS 512u
#define DS41_TOPK_CHUNK 2048u
#define DS41_EXPERT_NEURONS 64u
#define DS41_UP_SHARED_BYTES (DS41_BLOCK * DS41_HIDDEN * sizeof(uint16_t))
#define DS41_DOWN_SHARED_BYTES (DS41_BLOCK * DS41_MOE * sizeof(uint16_t))
#define DS41_SEQUENCES_MAX 1024u
#define DS41_RING_ELEMENTS ((uint64_t)DS41_STAGES * DS41_WINDOW * DS41_HEAD_DIM)
#define DS41_RING_BYTES (DS41_RING_ELEMENTS * sizeof(uint16_t))
#define DS41_STAGE_RING (DS41_WINDOW * DS41_HEAD_DIM)
#define DS41_EXPERT_PARTS 6u
#define DS41_W1_PART 0u
#define DS41_W2_PART 2u
#define DS41_W3_PART 4u
#define DS41_PACK_MAGIC 0x4B505244u
#define DS41_PACK_VERSION 1u
#define DS41_PACK_ALIGNMENT 128u
#define DS41_INVERSE_PI 0.31830988618379067154
#define DS41_ROUTE_EPSILON 1e-20f
#define DS41_SOFTPLUS_THRESHOLD 20.0f

static_assert(DS41_ACTIVE == 3u, "the gate keeps three experts a row");

typedef struct Ds41Matrix
{
	const uint8_t *weight;
	const uint8_t *scale;
	uint8_t *rows_scale;
	uint32_t rows;
	uint32_t columns;
} Ds41Matrix;

typedef struct Ds41Stage
{
	const float *sink;
	const uint16_t *q_norm;
	const uint16_t *kv_norm;
	const uint16_t *attn_norm;
	const uint16_t *ffn_norm;
	const uint16_t *gate;
	const float *gate_bias;
	const float *hc_attn_fn;
	const float *hc_attn_base;
	const float *hc_attn_scale;
	const float *hc_ffn_fn;
	const float *hc_ffn_base;
	const float *hc_ffn_scale;
	Ds41Matrix wq_a;
	Ds41Matrix wq_b;
	Ds41Matrix wkv;
	Ds41Matrix wo_a;
	Ds41Matrix wo_b;
	Ds41Matrix shared_w1;
	Ds41Matrix shared_w2;
	Ds41Matrix shared_w3;
	const uint8_t *const *experts;
} Ds41Stage;

typedef struct Ds41Sequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t last_used;
	uint32_t live;
	uint32_t retired;
	int64_t positions[DS41_WINDOW];
	uint32_t tokens[DS41_WINDOW];
	uint16_t *ring;
} Ds41Sequence;

typedef struct Ds41Draft
{
	uint64_t seed;
	uint32_t positions[DS41_BLOCK];
	uint32_t anchor;
	uint32_t noise;
	uint32_t greedy;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint8_t valid[DS41_WINDOW];
} Ds41Draft;

typedef struct Ds41Picks
{
	unsigned long long keys[DS41_BLOCK];
	unsigned long long alt_keys[DS41_BLOCK];
	unsigned long long branch_keys[DS41_BRANCHES];
	uint32_t tokens[DS41_BLOCK];
	float scores[DS41_BLOCK];
	uint32_t alt_tokens[DS41_BLOCK];
	float alt_scores[DS41_BLOCK];
	uint32_t branch_fork[DS41_BRANCHES];
	uint32_t branch_len[DS41_BRANCHES];
	uint32_t branch_tokens[DS41_BRANCHES][DS41_BRANCH_LEN];
	float branch_scores[DS41_BRANCHES][DS41_BRANCH_LEN];
} Ds41Picks;

typedef struct Ds41Forward
{
	Ds41Draft *host_draft;
	Ds41Draft *draft;
	Ds41Picks *host_picks;
	Ds41Picks *picks;
	uint16_t *streams[2];
	float *mix_partials;
	float *pre_initial;
	float *attn_pre;
	float *attn_post;
	float *attn_comb;
	float *ffn_pre[2];
	float *ffn_post;
	float *ffn_comb;
	uint16_t *collapsed;
	uint16_t *x;
	uint16_t *xq;
	uint16_t *q_raw;
	uint16_t *q_normed;
	uint16_t *q_quant;
	uint16_t *q;
	uint16_t *kv_raw;
	uint16_t *kv;
	uint16_t *attended;
	uint16_t *grouped;
	uint16_t *projected;
	uint16_t *projected_quant;
	uint16_t *attn_out;
	float *route_weight;
	uint32_t *route_expert;
	float *gate_scores;
	uint16_t *pair_gate;
	uint16_t *pair_up;
	uint16_t *pair_hidden;
	uint16_t *pair_down;
	uint16_t *shared_gate;
	uint16_t *shared_up;
	uint16_t *shared_hidden;
	uint16_t *shared_out;
	uint16_t *moe_out;
	uint16_t *hidden;
	uint16_t *normed;
	float *logits;
	float *row;
	float *softmax_partials;
	float *topk_values;
	uint32_t *topk_positions;
	uint32_t *candidates;
} Ds41Forward;

typedef struct Ds41State
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	uint32_t noise_token;
	float epsilon;
	float hc_epsilon;
	float route_scale;
	float swiglu_limit;
	float rope_theta;
	float softmax_scale;
	float *rope_inverse;
	const uint16_t *embed;
	const uint16_t *head;
	const uint16_t *final_norm;
	const uint16_t *markov_embed;
	const uint16_t *markov_head;
	const uint16_t *confidence;
	const uint16_t *main_norm;
	Ds41Matrix main_proj;
	Ds41Stage stages[DS41_STAGES];
	const uint8_t **expert_table;
	uint8_t *rows_scales;
	uint64_t rows_scales_bytes;
	Ds41Forward forward;
	uint16_t *host_taps;
	uint32_t *host_positions;
	uint16_t *taps;
	uint16_t *taps_quant;
	uint32_t *positions;
	uint16_t *main_raw;
	uint16_t *main;
	uint16_t *main_quant;
	uint16_t *key_raw;
	uint16_t *rings;
	uint32_t capacity;
	uint64_t clock;
	Ds41Sequence *sequences;
	uint64_t frames;
	uint64_t rows;
	uint64_t donor_fills;
	uint64_t gaps;
} Ds41State;

static __device__ __forceinline__ void Ds41Rotation(uint32_t position, float inverse, float *cosine, float *sine)
{
	const float angle = (float)position * inverse;
	const double half_turns = (double)angle * DS41_INVERSE_PI;
	const float reduced = (float)(half_turns - 2.0 * rint(0.5 * half_turns));
	sincospif(reduced, sine, cosine);
}

static __device__ __forceinline__ float Ds41WarpMax(float value)
{
	for ( uint32_t mask = DS41_WARP / 2u; mask > 0u; mask >>= 1u )
		value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
	return value;
}

static __device__ __forceinline__ float Ds41WarpSum(float value)
{
	for ( uint32_t mask = DS41_WARP / 2u; mask > 0u; mask >>= 1u )
		value += __shfl_xor_sync(0xffffffffu, value, mask);
	return value;
}

static __device__ __forceinline__ unsigned long long Ds41WarpKey(unsigned long long key)
{
	for ( uint32_t mask = DS41_WARP / 2u; mask > 0u; mask >>= 1u )
	{
		const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, mask);
		key = other > key ? other : key;
	}
	return key;
}

static __device__ __forceinline__ uint16_t Ds41QdqLane(float value)
{
	const float amax = Ds41WarpMax(fabsf(value));
	return LmFloatToBf16(SparkMathFp8Qdq(value, SparkMathFp8QdqScale(amax)));
}

static __device__ __forceinline__ void Ds41SoftmaxMerge(float *top, float *sum, float other_top, float other_sum)
{
	const float joined = fmaxf(*top, other_top);
	*sum = (*top == -INFINITY ? 0.0f : *sum * expf(*top - joined)) + (other_top == -INFINITY ? 0.0f : other_sum * expf(other_top - joined));
	*top = joined;
}

static __device__ __forceinline__ uint32_t Ds41KeyToken(unsigned long long key)
{
	return 0xffffffffu - (uint32_t)key;
}

static __device__ __forceinline__ float Ds41KeyValue(unsigned long long key)
{
	const uint32_t ordered = (uint32_t)(key >> 32u);
	return __uint_as_float(ordered ^ ((ordered & 0x80000000u) != 0u ? 0x80000000u : 0xffffffffu));
}

static __device__ __forceinline__ void Ds41Top2Insert(unsigned long long candidate, unsigned long long *first, unsigned long long *second)
{
	if ( candidate > *first )
	{
		*second = *first;
		*first = candidate;
	}
	else if ( candidate > *second && candidate != *first )
		*second = candidate;
}

static __device__ __forceinline__ uint32_t Ds41Previous(const Ds41Draft *draft, const Ds41Picks *picks, uint32_t step)
{
	if ( step == 0u )
		return draft->anchor;
	return draft->greedy != 0u ? Ds41KeyToken(picks->keys[step - 1u]) : picks->tokens[step - 1u];
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41ExpandScalesKernel(const uint8_t *__restrict__ block, uint8_t *__restrict__ rows_scale, uint32_t rows, uint32_t groups)
{
	const uint64_t total = (uint64_t)rows * groups;
	for ( uint64_t index = (uint64_t)blockIdx.x * DS41_THREADS + threadIdx.x; index < total; index += (uint64_t)gridDim.x * DS41_THREADS )
	{
		const uint64_t row = index / groups, group = index % groups;
		rows_scale[index] = block[(row / DS41_QDQ) * groups + group];
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41RopeTableKernel(float *__restrict__ inverse, float theta)
{
	if ( threadIdx.x < DS41_ROPE / 2u )
		inverse[threadIdx.x] = 1.0f / powf(theta, (float)(2u * threadIdx.x) / (float)DS41_ROPE);
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41QdqKernel(const uint16_t *__restrict__ input, uint16_t *__restrict__ output, uint32_t rows, uint32_t parts, uint32_t part_width,
	uint32_t to_part_major)
{
	const uint64_t groups = (uint64_t)rows * parts * part_width / DS41_QDQ;
	const uint32_t lane = threadIdx.x % DS41_WARP;
	for ( uint64_t group = (uint64_t)blockIdx.x * DS41_WARPS + threadIdx.x / DS41_WARP; group < groups; group += (uint64_t)gridDim.x * DS41_WARPS )
	{
		const uint64_t element = group * DS41_QDQ + lane;
		const uint64_t row = element / ((uint64_t)parts * part_width), part = (element / part_width) % parts, column = element % part_width;
		const uint64_t row_major = (row * parts + part) * part_width + column, part_major = (part * rows + row) * part_width + column;
		const uint64_t source = to_part_major != 0u ? row_major : part_major, target = to_part_major != 0u ? part_major : row_major;
		output[target] = Ds41QdqLane(LmBf16ToFloat(input[source]));
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41RmsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output, uint32_t dimension,
	float epsilon)
{
	__shared__ float reduction[DS41_WARPS];
	const uint16_t *row = input + (uint64_t)blockIdx.x * dimension;
	uint16_t *out = output + (uint64_t)blockIdx.x * dimension;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += DS41_THREADS )
	{
		const float value = LmBf16ToFloat(row[index]);
		total += value * value;
	}
	total = LmBlockSum<DS41_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += DS41_THREADS )
		out[index] = LmFloatToBf16(LmBf16ToFloat(row[index]) * scale * LmBf16ToFloat(weight[index]));
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41KeyRowsKernel(const uint16_t *__restrict__ raw, const uint16_t *__restrict__ norm, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, float epsilon, uint16_t *__restrict__ ring, uint16_t *__restrict__ rows_out)
{
	__shared__ float reduction[DS41_WARPS];
	__shared__ float key[DS41_HEAD_DIM];
	const uint16_t *row = raw + (uint64_t)blockIdx.x * DS41_HEAD_DIM;
	const uint32_t position = positions[blockIdx.x];
	uint16_t *out = ring != 0 ? ring + (uint64_t)(position % DS41_WINDOW) * DS41_HEAD_DIM : rows_out + (uint64_t)blockIdx.x * DS41_HEAD_DIM;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DS41_THREADS )
	{
		const float value = LmBf16ToFloat(row[index]);
		total += value * value;
	}
	total = LmBlockSum<DS41_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)DS41_HEAD_DIM + epsilon);
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DS41_THREADS )
		key[index] = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(row[index]) * scale * LmBf16ToFloat(norm[index])));
	__syncthreads();
	if ( threadIdx.x < DS41_ROPE / 2u )
	{
		const uint32_t pair = threadIdx.x, at = DS41_NOPE + 2u * pair;
		const float even = key[at], odd = key[at + 1u];
		float cosine, sine;
		Ds41Rotation(position, inverse[pair], &cosine, &sine);
		key[at] = LmBf16ToFloat(LmFloatToBf16(even * cosine - odd * sine));
		key[at + 1u] = LmBf16ToFloat(LmFloatToBf16(even * sine + odd * cosine));
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DS41_THREADS )
		out[index] = Ds41QdqLane(key[index]);
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41RopeKernel(uint16_t *__restrict__ data, const Ds41Draft *__restrict__ draft, const float *__restrict__ inverse, uint32_t heads,
	float direction)
{
	const uint32_t pair = threadIdx.x % (DS41_ROPE / 2u), item = blockIdx.x * (DS41_THREADS / (DS41_ROPE / 2u)) + threadIdx.x / (DS41_ROPE / 2u);
	if ( item >= DS41_BLOCK * heads )
		return;
	{
		uint16_t *row = data + (uint64_t)item * DS41_HEAD_DIM + DS41_NOPE + 2u * pair;
		const float even = LmBf16ToFloat(row[0]), odd = LmBf16ToFloat(row[1]);
		float cosine, sine;
		Ds41Rotation(draft->positions[item / heads], inverse[pair], &cosine, &sine);
		sine *= direction;
		row[0] = LmFloatToBf16(even * cosine - odd * sine);
		row[1] = LmFloatToBf16(even * sine + odd * cosine);
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41EmbedKernel(const uint16_t *__restrict__ embed, const Ds41Draft *__restrict__ draft, uint16_t *__restrict__ streams)
{
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->noise;
	for ( uint32_t index = threadIdx.x; index < DS41_HC_FLAT; index += DS41_THREADS )
		streams[(uint64_t)row * DS41_HC_FLAT + index] = embed[(uint64_t)token * DS41_HIDDEN + index % DS41_HIDDEN];
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41HcPostKernel(const uint16_t *__restrict__ output, const uint16_t *__restrict__ residual, const float *__restrict__ post_rows,
	const float *__restrict__ comb_rows, uint16_t *__restrict__ streams)
{
	__shared__ float post[DS41_HC];
	__shared__ float comb[DS41_HC * DS41_HC];
	const uint32_t row = blockIdx.y;
	if ( threadIdx.x < DS41_HC )
		post[threadIdx.x] = post_rows[(uint64_t)row * DS41_HC + threadIdx.x];
	if ( threadIdx.x < DS41_HC * DS41_HC )
		comb[threadIdx.x] = comb_rows[(uint64_t)row * DS41_HC * DS41_HC + threadIdx.x];
	__syncthreads();
	for ( uint32_t element = blockIdx.x * DS41_THREADS + threadIdx.x; element < DS41_HIDDEN; element += gridDim.x * DS41_THREADS )
	{
		const float y = LmBf16ToFloat(output[(uint64_t)row * DS41_HIDDEN + element]);
		float source[DS41_HC];
#pragma unroll
		for ( uint32_t stream = 0u; stream < DS41_HC; stream++ )
			source[stream] = LmBf16ToFloat(residual[((uint64_t)row * DS41_HC + stream) * DS41_HIDDEN + element]);
#pragma unroll
		for ( uint32_t stream = 0u; stream < DS41_HC; stream++ )
		{
			float mixed = 0.0f;
#pragma unroll
			for ( uint32_t from = 0u; from < DS41_HC; from++ )
				mixed += comb[from * DS41_HC + stream] * source[from];
			streams[((uint64_t)row * DS41_HC + stream) * DS41_HIDDEN + element] = LmFloatToBf16(post[stream] * y + mixed);
		}
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41AttentionKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ ring, const uint16_t *__restrict__ block_keys,
	const Ds41Draft *__restrict__ draft, const float *__restrict__ sink, float scale, uint16_t *__restrict__ output)
{
	constexpr uint32_t KEYS = DS41_WINDOW + DS41_BLOCK;
	constexpr uint32_t PER = DS41_HEAD_DIM / DS41_WARP;
	__shared__ float query[DS41_HEAD_DIM];
	__shared__ float scores[KEYS];
	__shared__ float reduction[DS41_WARPS];
	const uint32_t head = blockIdx.x, row = blockIdx.y, warp = threadIdx.x / DS41_WARP, lane = threadIdx.x % DS41_WARP;
	const uint16_t *own = q + ((uint64_t)row * DS41_HEADS + head) * DS41_HEAD_DIM;
	float top = -INFINITY, total = 0.0f, low = 0.0f, high = 0.0f;
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DS41_THREADS )
		query[index] = LmBf16ToFloat(own[index]);
	__syncthreads();
	for ( uint32_t key = warp; key < KEYS; key += DS41_WARPS )
	{
		const uint16_t *source = key < DS41_WINDOW ? ring + (uint64_t)key * DS41_HEAD_DIM : block_keys + (uint64_t)(key - DS41_WINDOW) * DS41_HEAD_DIM;
		float dot = 0.0f;
		if ( key < DS41_WINDOW && draft->valid[key] == 0u )
		{
			if ( lane == 0u )
				scores[key] = -INFINITY;
			continue;
		}
#pragma unroll
		for ( uint32_t part = 0u; part < PER; part++ )
			dot += query[part * DS41_WARP + lane] * LmBf16ToFloat(source[part * DS41_WARP + lane]);
		dot = Ds41WarpSum(dot);
		if ( lane == 0u )
			scores[key] = dot * scale;
	}
	__syncthreads();
	for ( uint32_t key = threadIdx.x; key < KEYS; key += DS41_THREADS )
		top = fmaxf(top, scores[key]);
	top = Ds41WarpMax(top);
	if ( lane == 0u )
		reduction[warp] = top;
	__syncthreads();
	top = -INFINITY;
	for ( uint32_t index = 0u; index < DS41_WARPS; index++ )
		top = fmaxf(top, reduction[index]);
	top = fmaxf(top, -1e30f);
	__syncthreads();
	for ( uint32_t key = threadIdx.x; key < KEYS; key += DS41_THREADS )
	{
		const float weight = expf(scores[key] - top);
		scores[key] = weight;
		total += weight;
	}
	total = LmBlockSum<DS41_THREADS>(total, reduction);
	total += expf(sink[head] - top);
	for ( uint32_t key = 0u; key < KEYS; key++ )
	{
		const float weight = LmBf16ToFloat(LmFloatToBf16(scores[key]));
		const uint16_t *source = key < DS41_WINDOW ? ring + (uint64_t)key * DS41_HEAD_DIM : block_keys + (uint64_t)(key - DS41_WINDOW) * DS41_HEAD_DIM;
		if ( weight == 0.0f )
			continue;
		low += weight * LmBf16ToFloat(source[threadIdx.x]);
		high += weight * LmBf16ToFloat(source[threadIdx.x + DS41_THREADS]);
	}
	output[((uint64_t)row * DS41_HEADS + head) * DS41_HEAD_DIM + threadIdx.x] = LmFloatToBf16(low / total);
	output[((uint64_t)row * DS41_HEADS + head) * DS41_HEAD_DIM + threadIdx.x + DS41_THREADS] = LmFloatToBf16(high / total);
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41GateScoreKernel(const uint16_t *__restrict__ x, const uint16_t *__restrict__ gate, float *__restrict__ scores)
{
	constexpr uint32_t STEPS = DS41_HIDDEN / (DS41_WARP * 8u);
	const uint32_t warp = threadIdx.x / DS41_WARP, lane = threadIdx.x % DS41_WARP, expert = blockIdx.x * DS41_WARPS + warp;
	const uint16_t *weights = gate + (uint64_t)expert * DS41_HIDDEN;
	float dot[DS41_BLOCK];
#pragma unroll
	for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
		dot[row] = 0.0f;
	for ( uint32_t step = 0u; step < STEPS; step++ )
	{
		const uint32_t at = (step * DS41_WARP + lane) * 8u;
		const uint4 packed = __ldg((const uint4 *)(weights + at));
		const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
		for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
		{
			const uint4 input = __ldg((const uint4 *)(x + (uint64_t)row * DS41_HIDDEN + at));
			const uint32_t values[4] = {input.x, input.y, input.z, input.w};
#pragma unroll
			for ( uint32_t word = 0u; word < 4u; word++ )
			{
				dot[row] += __uint_as_float(values[word] << 16u) * __uint_as_float(words[word] << 16u);
				dot[row] += __uint_as_float(values[word] & 0xffff0000u) * __uint_as_float(words[word] & 0xffff0000u);
			}
		}
	}
#pragma unroll
	for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
	{
		const float total = Ds41WarpSum(dot[row]);
		if ( lane == 0u )
			scores[row * DS41_EXPERTS + expert] = sqrtf(total > DS41_SOFTPLUS_THRESHOLD ? total : log1pf(expf(total)));
	}
}

__global__ __launch_bounds__(DS41_WARP, 1)
void Ds41GateTopKernel(const float *__restrict__ scores_rows, const float *__restrict__ bias, float route_scale, uint32_t *__restrict__ route_expert,
	float *__restrict__ route_weight)
{
	const uint32_t row = blockIdx.x, lane = threadIdx.x;
	const float *scores = scores_rows + (uint64_t)row * DS41_EXPERTS;
	uint32_t first = 0u, second = 0u, third = 0u;
#pragma unroll
	for ( uint32_t slot = 0u; slot < DS41_ACTIVE; slot++ )
	{
		unsigned long long key = 0ull;
		for ( uint32_t expert = lane; expert < DS41_EXPERTS; expert += DS41_WARP )
			if ( !(slot > 0u && expert == first) && !(slot > 1u && expert == second) )
			{
				const unsigned long long candidate = ((unsigned long long)SparkMathSampleOrdered(scores[expert] + bias[expert]) << 32u) |
					(unsigned long long)(0xffffffffu - expert);
				key = candidate > key ? candidate : key;
			}
		key = Ds41WarpKey(key);
		if ( slot == 0u )
			first = Ds41KeyToken(key);
		else if ( slot == 1u )
			second = Ds41KeyToken(key);
		else
			third = Ds41KeyToken(key);
	}
	if ( lane < DS41_ACTIVE )
	{
		const float a = scores[first], b = scores[second], c = scores[third];
		const uint32_t expert = lane == 0u ? first : lane == 1u ? second : third;
		const float picked = lane == 0u ? a : lane == 1u ? b : c;
		route_expert[row * DS41_ACTIVE + lane] = expert;
		route_weight[row * DS41_ACTIVE + lane] = picked / (a + b + c + DS41_ROUTE_EPSILON) * route_scale;
	}
}

static __device__ __forceinline__ void Ds41Mxfp4Decode(uint4 packed, float *weights)
{
	const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
	for ( uint32_t word = 0u; word < 4u; word++ )
#pragma unroll
		for ( uint32_t byte = 0u; byte < 4u; byte++ )
		{
			const uint32_t code = (words[word] >> (8u * byte)) & 0xffu;
			weights[8u * word + 2u * byte] = LmE2m1ToFloatPure(code & 15u);
			weights[8u * word + 2u * byte + 1u] = LmE2m1ToFloatPure(code >> 4u);
		}
}

static __device__ __forceinline__ float Ds41Mxfp4Dot(const float *weights, const uint16_t *input)
{
	float total = 0.0f;
#pragma unroll
	for ( uint32_t word = 0u; word < 4u; word++ )
	{
		const uint4 loaded = *(const uint4 *)(input + 8u * word);
		const uint32_t values[4] = {loaded.x, loaded.y, loaded.z, loaded.w};
#pragma unroll
		for ( uint32_t byte = 0u; byte < 4u; byte++ )
			total += weights[8u * word + 2u * byte] * __uint_as_float(values[byte] << 16u) +
				weights[8u * word + 2u * byte + 1u] * __uint_as_float(values[byte] & 0xffff0000u);
	}
	return total;
}

static __device__ __forceinline__ uint32_t Ds41ExpertPairs(const uint32_t *__restrict__ route_expert, uint32_t unique_index, uint32_t *pairs,
	uint32_t *expert_out)
{
	uint32_t unique = 0u, count = 0u, expert = 0xffffffffu;
	for ( uint32_t pair = 0u; pair < DS41_PAIRS; pair++ )
	{
		uint32_t seen = 0u;
		for ( uint32_t earlier = 0u; earlier < pair; earlier++ )
			seen |= route_expert[earlier] == route_expert[pair] ? 1u : 0u;
		if ( seen == 0u && unique++ == unique_index )
			expert = route_expert[pair];
	}
	for ( uint32_t pair = 0u; pair < DS41_PAIRS && expert != 0xffffffffu; pair++ )
		if ( route_expert[pair] == expert )
			pairs[count++] = pair;
	*expert_out = expert;
	return count;
}

template<uint32_t INPUT, uint32_t OUTPUTS, uint32_t ROW_DIVISOR>
static __device__ __forceinline__ void Ds41ExpertRows(const uint16_t *__restrict__ input, const uint8_t *const *__restrict__ table,
	const uint32_t *__restrict__ route_expert, uint32_t first_part, uint32_t second_part, uint16_t *first_out, uint16_t *second_out)
{
	constexpr uint32_t CHUNKS = INPUT / DS41_QDQ;
	constexpr uint32_t PER_LANE = (CHUNKS + DS41_WARP - 1u) / DS41_WARP;
	extern __shared__ __align__(16) uint16_t ds41_expert_rows[];
	__shared__ uint32_t pairs[DS41_BLOCK];
	__shared__ uint32_t shared_count;
	__shared__ uint32_t shared_expert;
	const uint32_t warp = threadIdx.x / DS41_WARP, lane = threadIdx.x % DS41_WARP;
	uint32_t count, expert;
	if ( threadIdx.x == 0u )
		shared_count = Ds41ExpertPairs(route_expert, blockIdx.x, pairs, &shared_expert);
	__syncthreads();
	count = shared_count;
	expert = shared_expert;
	if ( count == 0u )
		return;
	for ( uint32_t index = threadIdx.x; index < count * (INPUT / 8u); index += DS41_THREADS )
	{
		const uint32_t row = index / (INPUT / 8u), column = index % (INPUT / 8u);
		((uint4 *)ds41_expert_rows)[index] = __ldg((const uint4 *)(input + (uint64_t)(pairs[row] / ROW_DIVISOR) * INPUT) + column);
	}
	__syncthreads();
#pragma unroll 1
	for ( uint32_t item = warp; item < OUTPUTS * DS41_EXPERT_NEURONS; item += DS41_WARPS )
	{
		const uint32_t which = item / DS41_EXPERT_NEURONS, neuron = blockIdx.y * DS41_EXPERT_NEURONS + item % DS41_EXPERT_NEURONS;
		const uint8_t *const *parts = table + (uint64_t)expert * DS41_EXPERT_PARTS + (which == 0u ? first_part : second_part);
		uint16_t *output = which == 0u ? first_out : second_out;
		const uint8_t *weight = parts[1] + (uint64_t)neuron * (INPUT / 2u), *scale = parts[0] + (uint64_t)neuron * CHUNKS;
		float dot[DS41_BLOCK];
		uint4 next = lane < CHUNKS ? __ldg((const uint4 *)(weight + (uint64_t)lane * (DS41_QDQ / 2u))) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
		for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
			dot[row] = 0.0f;
#pragma unroll 1
		for ( uint32_t step = 0u; step < PER_LANE; step++ )
		{
			const uint32_t chunk = lane + step * DS41_WARP, following = chunk + DS41_WARP;
			const uint4 packed = next;
			float weights[DS41_QDQ], factor;
			if ( chunk >= CHUNKS )
				break;
			if ( following < CHUNKS )
				next = __ldg((const uint4 *)(weight + (uint64_t)following * (DS41_QDQ / 2u)));
			factor = LmUe8m0ToFloat(scale[chunk]);
			Ds41Mxfp4Decode(packed, weights);
#pragma unroll
			for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
				if ( row < count )
					dot[row] += Ds41Mxfp4Dot(weights, ds41_expert_rows + (uint64_t)row * INPUT + chunk * DS41_QDQ) * factor;
		}
#pragma unroll
		for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
			if ( row < count )
			{
				const float total = Ds41WarpSum(dot[row]);
				if ( lane == 0u )
					output[(uint64_t)pairs[row] * (OUTPUTS == 2u ? DS41_MOE : DS41_HIDDEN) + neuron] = LmFloatToBf16(total);
			}
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41ExpertUpKernel(const uint16_t *__restrict__ xq, const uint8_t *const *__restrict__ table, const uint32_t *__restrict__ route_expert,
	uint16_t *__restrict__ gate_out, uint16_t *__restrict__ up_out)
{
	Ds41ExpertRows<DS41_HIDDEN, 2u, DS41_ACTIVE>(xq, table, route_expert, DS41_W1_PART, DS41_W3_PART, gate_out, up_out);
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41ExpertDownKernel(const uint16_t *__restrict__ hidden, const uint8_t *const *__restrict__ table, const uint32_t *__restrict__ route_expert,
	uint16_t *__restrict__ down_out)
{
	Ds41ExpertRows<DS41_MOE, 1u, 1u>(hidden, table, route_expert, DS41_W2_PART, DS41_W2_PART, down_out, down_out);
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41SwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ route_weight, float limit,
	uint32_t rows, uint16_t *__restrict__ output)
{
	const uint64_t groups = (uint64_t)rows * DS41_MOE / DS41_QDQ;
	const uint32_t lane = threadIdx.x % DS41_WARP;
	for ( uint64_t group = (uint64_t)blockIdx.x * DS41_WARPS + threadIdx.x / DS41_WARP; group < groups; group += (uint64_t)gridDim.x * DS41_WARPS )
	{
		const uint64_t element = group * DS41_QDQ + lane, row = element / DS41_MOE;
		const float g = fminf(LmBf16ToFloat(gate[element]), limit);
		const float u = fminf(fmaxf(LmBf16ToFloat(up[element]), -limit), limit);
		float value = g / (1.0f + expf(-g)) * u;
		if ( route_weight != 0 )
			value = route_weight[row] * value;
		output[element] = Ds41QdqLane(LmBf16ToFloat(LmFloatToBf16(value)));
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41MoeSumKernel(const uint16_t *__restrict__ pair_down, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * DS41_THREADS + threadIdx.x; element < DS41_HIDDEN; element += gridDim.x * DS41_THREADS )
	{
		float total = 0.0f;
#pragma unroll
		for ( uint32_t slot = 0u; slot < DS41_ACTIVE; slot++ )
			total += LmBf16ToFloat(pair_down[((uint64_t)row * DS41_ACTIVE + slot) * DS41_HIDDEN + element]);
		total += LmBf16ToFloat(shared[(uint64_t)row * DS41_HIDDEN + element]);
		output[(uint64_t)row * DS41_HIDDEN + element] = LmFloatToBf16(total);
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41MarkovRowKernel(uint32_t step, const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks, const float *__restrict__ logits,
	const uint16_t *__restrict__ markov_embed, const uint16_t *__restrict__ markov_head, float *__restrict__ row, float *__restrict__ partials)
{
	constexpr uint32_t PER = DS41_MARKOV / DS41_WARP;
	__shared__ float embedding[DS41_MARKOV];
	__shared__ unsigned long long best[DS41_WARPS];
	__shared__ unsigned long long second_best[DS41_WARPS];
	__shared__ float tops[DS41_WARPS];
	__shared__ float sums[DS41_WARPS];
	const uint32_t warp = threadIdx.x / DS41_WARP, lane = threadIdx.x % DS41_WARP, previous = Ds41Previous(draft, picks, step);
	const float scale = draft->greedy != 0u ? 1.0f : draft->inverse_temperature;
	unsigned long long first = 0ull, second = 0ull;
	float top = -INFINITY, sum = 0.0f;
	for ( uint32_t index = threadIdx.x; index < DS41_MARKOV; index += DS41_THREADS )
		embedding[index] = LmBf16ToFloat(markov_embed[(uint64_t)previous * DS41_MARKOV + index]);
	__syncthreads();
	for ( uint32_t token = blockIdx.x * DS41_WARPS + warp; token < DS41_VOCAB; token += gridDim.x * DS41_WARPS )
	{
		const uint4 packed = __ldg((const uint4 *)(markov_head + (uint64_t)token * DS41_MARKOV + lane * PER));
		const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
		float dot = 0.0f, value;
#pragma unroll
		for ( uint32_t word = 0u; word < 4u; word++ )
		{
			dot += __uint_as_float(words[word] << 16u) * embedding[lane * PER + 2u * word];
			dot += __uint_as_float(words[word] & 0xffff0000u) * embedding[lane * PER + 2u * word + 1u];
		}
		dot = Ds41WarpSum(dot);
		value = logits[(uint64_t)step * DS41_VOCAB + token] + dot;
		if ( lane == 0u )
		{
			const unsigned long long candidate = ((unsigned long long)SparkMathSampleOrdered(value) << 32u) | (unsigned long long)(0xffffffffu - token);
			row[token] = value;
			if ( draft->greedy != 0u )
				Ds41Top2Insert(candidate, &first, &second);
			Ds41SoftmaxMerge(&top, &sum, value * scale, 1.0f);
		}
	}
	if ( lane == 0u )
	{
		best[warp] = first;
		second_best[warp] = second;
		tops[warp] = top;
		sums[warp] = sum;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		for ( uint32_t index = 1u; index < DS41_WARPS; index++ )
		{
			Ds41SoftmaxMerge(&top, &sum, tops[index], sums[index]);
			if ( draft->greedy != 0u )
			{
				Ds41Top2Insert(best[index], &first, &second);
				Ds41Top2Insert(second_best[index], &first, &second);
			}
		}
		partials[((uint64_t)step * gridDim.x + blockIdx.x) * 2u] = top;
		partials[((uint64_t)step * gridDim.x + blockIdx.x) * 2u + 1u] = sum;
		if ( draft->greedy != 0u )
		{
			const unsigned long long prev = atomicMax(&picks->keys[step], first);
			const unsigned long long loser = prev > first ? first : prev;
			const unsigned long long runner = loser > second ? loser : second;
			if ( runner != 0ull )
				atomicMax(&picks->alt_keys[step], runner);
		}
	}
}

__global__ __launch_bounds__(DS41_PICK_THREADS, 1)
void Ds41SampleKernel(uint32_t step, const Ds41Draft *__restrict__ draft, const float *__restrict__ row, const uint32_t *__restrict__ candidates,
	const float *__restrict__ partials, uint32_t blocks, Ds41Picks *__restrict__ picks)
{
	__shared__ float values[DS41_SAMPLED];
	__shared__ uint32_t tokens[DS41_SAMPLED];
	__shared__ float sorted_values[DS41_SAMPLED];
	__shared__ uint32_t sorted_tokens[DS41_SAMPLED];
	__shared__ float reduction[DS41_PICK_THREADS / DS41_WARP];
	__shared__ float before[DS41_SAMPLED];
	__shared__ unsigned long long best[DS41_SAMPLED / DS41_WARP];
	const float inverse_temperature = draft->inverse_temperature, nucleus = draft->nucleus;
	const uint32_t warp = threadIdx.x / DS41_WARP, lane = threadIdx.x % DS41_WARP, position = draft->positions[0] + step;
	float top, sum = 0.0f;
	if ( threadIdx.x < DS41_SAMPLED )
	{
		tokens[threadIdx.x] = candidates[threadIdx.x];
		values[threadIdx.x] = row[candidates[threadIdx.x]] * inverse_temperature;
	}
	__syncthreads();
	if ( threadIdx.x < DS41_SAMPLED )
	{
		const float value = values[threadIdx.x];
		const uint32_t token = tokens[threadIdx.x];
		uint32_t rank = 0u;
		for ( uint32_t other = 0u; other < DS41_SAMPLED; other++ )
			rank += values[other] > value || (values[other] == value && tokens[other] < token) ? 1u : 0u;
		sorted_values[rank] = value;
		sorted_tokens[rank] = token;
	}
	__syncthreads();
	top = sorted_values[0];
	for ( uint32_t block = threadIdx.x; block < blocks; block += DS41_PICK_THREADS )
		if ( partials[2u * block] != -INFINITY )
			sum += partials[2u * block + 1u] * expf(partials[2u * block] - top);
	sum = LmBlockSum<DS41_PICK_THREADS>(sum, reduction);
	if ( threadIdx.x == 0u )
	{
		float running = 0.0f;
		for ( uint32_t rank = 0u; rank < DS41_SAMPLED; rank++ )
		{
			const float probability = expf(sorted_values[rank] - top) / sum;
			running += probability;
			before[rank] = running - probability;
		}
	}
	__syncthreads();
	if ( warp < DS41_SAMPLED / DS41_WARP )
	{
		const uint32_t rank = warp * DS41_WARP + lane;
		const uint32_t keep = (((nucleus <= 0.0f) || (before[rank] < nucleus)) && (draft->top_k == 0u || rank < draft->top_k)) || rank == 0u ? 1u : 0u;
		const float score = keep != 0u ? sorted_values[rank] + SparkMathGumbelNoise(draft->seed, position, sorted_tokens[rank]) : -INFINITY;
		unsigned long long key = ((unsigned long long)SparkMathSampleOrdered(score) << 32u) | (unsigned long long)(0xffffffffu - rank);
		key = Ds41WarpKey(key);
		if ( lane == 0u )
			best[warp] = key;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		unsigned long long key = best[0];
		uint32_t winner, runner;
		for ( uint32_t index = 1u; index < DS41_SAMPLED / DS41_WARP; index++ )
			key = best[index] > key ? best[index] : key;
		winner = Ds41KeyToken(key);
		runner = winner == 0u ? 1u : 0u;
		picks->tokens[step] = sorted_tokens[winner];
		picks->alt_tokens[step] = sorted_tokens[runner];
		picks->alt_scores[step] = sum > 0.0f ? expf(sorted_values[runner] - top) / sum : 0.0f;
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41ConfidenceKernel(const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks, const uint16_t *__restrict__ hidden,
	const uint16_t *__restrict__ markov_embed, const uint16_t *__restrict__ weight, const float *__restrict__ partials, uint32_t blocks)
{
	__shared__ float reduction[DS41_WARPS];
	const uint32_t slot = blockIdx.x, previous = Ds41Previous(draft, picks, slot);
	float dot = 0.0f;
	for ( uint32_t index = threadIdx.x; index < DS41_CONFIDENCE; index += DS41_THREADS )
	{
		const float feature = index < DS41_HIDDEN ? LmBf16ToFloat(hidden[(uint64_t)slot * DS41_HIDDEN + index])
			: LmBf16ToFloat(markov_embed[(uint64_t)previous * DS41_MARKOV + index - DS41_HIDDEN]);
		dot += feature * LmBf16ToFloat(weight[index]);
	}
	dot = LmBlockSum<DS41_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
		picks->scores[slot] = 1.0f / (1.0f + expf(-dot));
	if ( draft->greedy != 0u )
	{
		const float top = Ds41KeyValue(picks->keys[slot]);
		const float alt = Ds41KeyValue(picks->alt_keys[slot]);
		const float *step_partials = partials + (uint64_t)slot * 2u * blocks;
		float sum = 0.0f;
		for ( uint32_t block = threadIdx.x; block < blocks; block += DS41_THREADS )
			if ( step_partials[2u * block] != -INFINITY )
				sum += step_partials[2u * block + 1u] * expf(step_partials[2u * block] - top);
		sum = LmBlockSum<DS41_THREADS>(sum, reduction);
		if ( threadIdx.x == 0u )
		{
			picks->tokens[slot] = Ds41KeyToken(picks->keys[slot]);
			picks->alt_tokens[slot] = Ds41KeyToken(picks->alt_keys[slot]);
			picks->alt_scores[slot] = sum > 0.0f ? expf(alt - top) / sum : 0.0f;
		}
	}
}

__global__ __launch_bounds__(DS41_WARP, 1)
void Ds41BranchSelectKernel(Ds41Picks *__restrict__ picks, uint32_t drafts, float min_score)
{
	uint32_t used = 0u;
	if ( threadIdx.x != 0u )
		return;
	for ( uint32_t branch = 0u; branch < DS41_BRANCHES; branch++ )
	{
		picks->branch_keys[branch] = 0ull;
		picks->branch_fork[branch] = DS41_BLOCK;
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
			if ( pos < DS41_BRANCHES )
			{
				picks->branch_fork[pos] = picks->branch_fork[pos - 1u];
				picks->branch_len[pos] = picks->branch_len[pos - 1u];
				picks->branch_tokens[pos][0] = picks->branch_tokens[pos - 1u][0];
				picks->branch_scores[pos][0] = picks->branch_scores[pos - 1u][0];
			}
			pos--;
		}
		if ( pos < DS41_BRANCHES )
		{
			picks->branch_fork[pos] = step;
			picks->branch_len[pos] = 1u;
			picks->branch_tokens[pos][0] = picks->alt_tokens[step];
			picks->branch_scores[pos][0] = score;
			if ( used < DS41_BRANCHES )
				used++;
		}
	}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41BranchRowKernel(uint32_t offset, uint32_t drafts, const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_embed, const uint16_t *__restrict__ markov_head, float *__restrict__ partials)
{
	constexpr uint32_t PER = DS41_MARKOV / DS41_WARP;
	__shared__ float embedding[DS41_BRANCHES][DS41_MARKOV];
	__shared__ unsigned long long best[DS41_BRANCHES][DS41_WARPS];
	__shared__ float tops[DS41_BRANCHES][DS41_WARPS];
	__shared__ float sums[DS41_BRANCHES][DS41_WARPS];
	const uint32_t warp = threadIdx.x / DS41_WARP, lane = threadIdx.x % DS41_WARP;
	const float scale = draft->greedy != 0u ? 1.0f : draft->inverse_temperature;
	uint32_t steps[DS41_BRANCHES], any = 0u;
	unsigned long long first[DS41_BRANCHES];
	float top[DS41_BRANCHES], sum[DS41_BRANCHES];
#pragma unroll
	for ( uint32_t b = 0u; b < DS41_BRANCHES; b++ )
	{
		const uint32_t fork = picks->branch_fork[b];
		steps[b] = fork < drafts && fork + offset < drafts ? fork + offset : DS41_BLOCK;
		first[b] = 0ull;
		top[b] = -INFINITY;
		sum[b] = 0.0f;
		if ( steps[b] < DS41_BLOCK )
		{
			const uint32_t previous = picks->branch_tokens[b][offset - 1u];
			any = 1u;
			for ( uint32_t index = threadIdx.x; index < DS41_MARKOV; index += DS41_THREADS )
				embedding[b][index] = LmBf16ToFloat(markov_embed[(uint64_t)previous * DS41_MARKOV + index]);
		}
	}
	if ( any == 0u )
		return;
	__syncthreads();
	for ( uint32_t token = blockIdx.x * DS41_WARPS + warp; token < DS41_VOCAB; token += gridDim.x * DS41_WARPS )
	{
		const uint4 packed = __ldg((const uint4 *)(markov_head + (uint64_t)token * DS41_MARKOV + lane * PER));
		const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
		for ( uint32_t b = 0u; b < DS41_BRANCHES; b++ )
		{
			if ( steps[b] >= DS41_BLOCK )
				continue;
			float dot = 0.0f;
#pragma unroll
			for ( uint32_t word = 0u; word < 4u; word++ )
			{
				dot += __uint_as_float(words[word] << 16u) * embedding[b][lane * PER + 2u * word];
				dot += __uint_as_float(words[word] & 0xffff0000u) * embedding[b][lane * PER + 2u * word + 1u];
			}
			dot = Ds41WarpSum(dot);
			if ( lane == 0u )
			{
				const float value = logits[(uint64_t)steps[b] * DS41_VOCAB + token] + dot;
				const unsigned long long candidate = ((unsigned long long)SparkMathSampleOrdered(value) << 32u) | (unsigned long long)(0xffffffffu - token);
				if ( candidate > first[b] )
					first[b] = candidate;
				Ds41SoftmaxMerge(&top[b], &sum[b], value * scale, 1.0f);
			}
		}
	}
	if ( lane == 0u )
#pragma unroll
		for ( uint32_t b = 0u; b < DS41_BRANCHES; b++ )
		{
			best[b][warp] = first[b];
			tops[b][warp] = top[b];
			sums[b][warp] = sum[b];
		}
	__syncthreads();
	if ( threadIdx.x == 0u )
#pragma unroll
		for ( uint32_t b = 0u; b < DS41_BRANCHES; b++ )
		{
			if ( steps[b] >= DS41_BLOCK )
				continue;
			for ( uint32_t index = 1u; index < DS41_WARPS; index++ )
			{
				Ds41SoftmaxMerge(&top[b], &sum[b], tops[b][index], sums[b][index]);
				if ( best[b][index] > first[b] )
					first[b] = best[b][index];
			}
			partials[((uint64_t)b * gridDim.x + blockIdx.x) * 2u] = top[b];
			partials[((uint64_t)b * gridDim.x + blockIdx.x) * 2u + 1u] = sum[b];
			atomicMax(&picks->branch_keys[b], first[b]);
		}
}

__global__ __launch_bounds__(DS41_THREADS, 1)
void Ds41BranchReduceKernel(uint32_t offset, uint32_t drafts, const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks,
	const float *__restrict__ partials, uint32_t blocks)
{
	__shared__ float reduction[DS41_WARPS];
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	if ( fork >= drafts || step >= drafts )
		return;
	{
		const float scale = draft->greedy != 0u ? 1.0f : draft->inverse_temperature;
		const unsigned long long key = picks->branch_keys[branch];
		const float top = Ds41KeyValue(key) * scale;
		const float *branch_partials = partials + (uint64_t)branch * 2u * blocks;
		float sum = 0.0f;
		for ( uint32_t block = threadIdx.x; block < blocks; block += DS41_THREADS )
			if ( branch_partials[2u * block] != -INFINITY )
				sum += branch_partials[2u * block + 1u] * expf(branch_partials[2u * block] - top);
		sum = LmBlockSum<DS41_THREADS>(sum, reduction);
		if ( threadIdx.x == 0u )
		{
			picks->branch_tokens[branch][offset] = Ds41KeyToken(key);
			picks->branch_scores[branch][offset] = sum > 0.0f ? 1.0f / sum : 0.0f;
			picks->branch_len[branch] = offset + 1u;
			picks->branch_keys[branch] = 0ull;
		}
	}
}

static SparkStatus Ds41DeviceOpen(void)
{
	const SparkStatus status = SparkDeviceInitialize(0);
	return status == SPARK_STATUS_DUPLICATE ? SPARK_STATUS_OK : status;
}

static SparkStatus Ds41Refuse(SparkDrafterModuleBind *bind, SparkStatus status, const char *format, const char *detail)
{
	if ( bind != 0 )
		snprintf(bind->text, sizeof(bind->text), format, detail);
	return status;
}

static SparkStatus Ds41LoadManifest(const char *pack_path, SparkJsonDocument *document, uint64_t *payload_base)
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
	if ( magic != DS41_PACK_MAGIC || version != DS41_PACK_VERSION || bytes == 0u || bytes > (UINT64_C(64) << 20) )
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
	*payload_base = (sizeof(header) + bytes + DS41_PACK_ALIGNMENT - 1u) / DS41_PACK_ALIGNMENT * DS41_PACK_ALIGNMENT;
	return status;
}

static SparkStatus Ds41Tensor(const Ds41State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *name, uint64_t expected,
	const void **pointer)
{
	const int32_t tensors = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "tensors");
	const int32_t entry = tensors < 0 ? -1 : SparkJsonFindObjectMember(document, tensors, name);
	const int32_t offset_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "offset");
	const int32_t bytes_token = entry < 0 ? -1 : SparkJsonFindObjectMember(document, entry, "bytes");
	uint64_t offset = 0u, bytes = 0u;
	const void *slice = 0;
	if ( offset_token < 0 || bytes_token < 0 || SparkJsonGetUInt64(document, offset_token, &offset) != SPARK_STATUS_OK ||
		SparkJsonGetUInt64(document, bytes_token, &bytes) != SPARK_STATUS_OK || bytes != expected )
	{
		fprintf(stderr, "dsv41 dspark: pack tensor %s is missing or is not %llu bytes\n", name, (unsigned long long)expected);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( SparkWeightdLazyPackSlice(state->pack, payload_base + offset, bytes, &slice) != SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	*pointer = slice;
	return SPARK_STATUS_OK;
}

static int32_t Ds41Checkpoint(const SparkJsonDocument *document)
{
	const int32_t config = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "config");
	return config < 0 ? -1 : SparkJsonFindObjectMember(document, config, "checkpoint_config");
}

static SparkStatus Ds41ConfigFloat(const SparkJsonDocument *document, int32_t object, const char *name, float *value)
{
	const int32_t token = object < 0 ? -1 : SparkJsonFindObjectMember(document, object, name);
	if ( token < 0 || SparkJsonGetFloat(document, token, value) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "dsv41 dspark: checkpoint_config %s is missing\n", name);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus Ds41Shape(Ds41State *state, const SparkJsonDocument *document)
{
	static const struct
	{
		const char *name;
		uint64_t value;
	} expected[] = {
		{"dim", DS41_HIDDEN}, {"n_heads", DS41_HEADS}, {"head_dim", DS41_HEAD_DIM}, {"rope_head_dim", DS41_ROPE}, {"q_lora_rank", DS41_QUERY_RANK},
		{"o_groups", DS41_GROUPS}, {"o_lora_rank", DS41_O_RANK}, {"window_size", DS41_WINDOW}, {"dspark_block_size", DS41_BLOCK},
		{"vocab_size", DS41_VOCAB}, {"dspark_markov_rank", DS41_MARKOV}, {"dspark_n_routed_experts", DS41_EXPERTS},
		{"dspark_n_activated_experts", DS41_ACTIVE}, {"moe_inter_dim", DS41_MOE}, {"hc_mult", DS41_HC}, {"hc_sinkhorn_iters", DS41_SINKHORN},
		{"n_mtp_layers", DS41_STAGES}, {"n_layers", DS41_LAYERS},
	};
	const int32_t checkpoint = Ds41Checkpoint(document);
	const int32_t noise = checkpoint < 0 ? -1 : SparkJsonFindObjectMember(document, checkpoint, "dspark_noise_token_id");
	uint64_t value = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	for ( uint32_t index = 0u; index < sizeof(expected) / sizeof(expected[0]); index++ )
	{
		const int32_t token = checkpoint < 0 ? -1 : SparkJsonFindObjectMember(document, checkpoint, expected[index].name);
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value != expected[index].value )
		{
			fprintf(stderr, "dsv41 dspark: checkpoint_config %s is not %llu, the shape this module is built for\n", expected[index].name,
				(unsigned long long)expected[index].value);
			return SPARK_STATUS_SCHEMA_ERROR;
		}
	}
	if ( noise < 0 || SparkJsonGetUInt64(document, noise, &value) != SPARK_STATUS_OK || value >= DS41_VOCAB )
	{
		fprintf(stderr, "dsv41 dspark: checkpoint_config dspark_noise_token_id is missing or outside the vocabulary\n");
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	state->noise_token = (uint32_t)value;
	status = Ds41ConfigFloat(document, checkpoint, "norm_eps", &state->epsilon);
	if ( status == SPARK_STATUS_OK )
		status = Ds41ConfigFloat(document, checkpoint, "hc_eps", &state->hc_epsilon);
	if ( status == SPARK_STATUS_OK )
		status = Ds41ConfigFloat(document, checkpoint, "route_scale", &state->route_scale);
	if ( status == SPARK_STATUS_OK )
		status = Ds41ConfigFloat(document, checkpoint, "swiglu_limit", &state->swiglu_limit);
	if ( status == SPARK_STATUS_OK )
		status = Ds41ConfigFloat(document, checkpoint, "rope_theta", &state->rope_theta);
	state->softmax_scale = (float)(1.0 / sqrt((double)DS41_HEAD_DIM));
	return status;
}

static SparkStatus Ds41Named(Ds41State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *format, uint32_t stage,
	const char *suffix, uint64_t bytes, const void **pointer)
{
	char name[160];
	snprintf(name, sizeof(name), format, stage, suffix);
	return Ds41Tensor(state, document, payload_base, name, bytes, pointer);
}

static SparkStatus Ds41MatrixLoad(Ds41State *state, const SparkJsonDocument *document, uint64_t payload_base, uint32_t stage, const char *prefix,
	uint32_t rows, uint32_t columns, Ds41Matrix *matrix)
{
	char weight[128], scale[128];
	const void *pointer = 0;
	SparkStatus status;
	snprintf(weight, sizeof(weight), "%s.weight", prefix);
	snprintf(scale, sizeof(scale), "%s.scale", prefix);
	matrix->rows = rows;
	matrix->columns = columns;
	status = Ds41Named(state, document, payload_base, "mtp.%u.%s", stage, weight, (uint64_t)rows * columns, &pointer);
	matrix->weight = (const uint8_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = Ds41Named(state, document, payload_base, "mtp.%u.%s", stage, scale, (uint64_t)(rows / DS41_QDQ) * (columns / DS41_QDQ), &pointer);
	matrix->scale = (const uint8_t *)pointer;
	state->rows_scales_bytes += (uint64_t)rows * (columns / DS41_QDQ);
	return status;
}

static SparkStatus Ds41StageLoad(Ds41State *state, const SparkJsonDocument *document, uint64_t payload_base, uint32_t stage, const uint8_t **table)
{
	Ds41Stage *s = &state->stages[stage];
	const struct
	{
		const char *suffix;
		uint64_t bytes;
		const void **pointer;
	} plain[] = {
		{"attn.attn_sink", DS41_HEADS * sizeof(float), (const void **)&s->sink},
		{"attn.q_norm.weight", DS41_QUERY_RANK * sizeof(uint16_t), (const void **)&s->q_norm},
		{"attn.kv_norm.weight", DS41_HEAD_DIM * sizeof(uint16_t), (const void **)&s->kv_norm},
		{"attn_norm.weight", DS41_HIDDEN * sizeof(uint16_t), (const void **)&s->attn_norm},
		{"ffn_norm.weight", DS41_HIDDEN * sizeof(uint16_t), (const void **)&s->ffn_norm},
		{"ffn.gate.weight", (uint64_t)DS41_EXPERTS * DS41_HIDDEN * sizeof(uint16_t), (const void **)&s->gate},
		{"ffn.gate.bias", DS41_EXPERTS * sizeof(float), (const void **)&s->gate_bias},
		{"hc_attn_fn", (uint64_t)DS41_HC_MIX * DS41_HC_FLAT * sizeof(float), (const void **)&s->hc_attn_fn},
		{"hc_attn_base", DS41_HC_MIX * sizeof(float), (const void **)&s->hc_attn_base},
		{"hc_attn_scale", 3u * sizeof(float), (const void **)&s->hc_attn_scale},
		{"hc_ffn_fn", (uint64_t)DS41_HC_MIX * DS41_HC_FLAT * sizeof(float), (const void **)&s->hc_ffn_fn},
		{"hc_ffn_base", DS41_HC_MIX * sizeof(float), (const void **)&s->hc_ffn_base},
		{"hc_ffn_scale", 3u * sizeof(float), (const void **)&s->hc_ffn_scale},
	};
	const struct
	{
		const char *prefix;
		uint32_t rows;
		uint32_t columns;
		Ds41Matrix *matrix;
	} matrices[] = {
		{"attn.wq_a", DS41_QUERY_RANK, DS41_HIDDEN, &s->wq_a}, {"attn.wq_b", DS41_QUERY_WIDTH, DS41_QUERY_RANK, &s->wq_b},
		{"attn.wkv", DS41_HEAD_DIM, DS41_HIDDEN, &s->wkv}, {"attn.wo_a", DS41_O_WIDTH, DS41_GROUP_WIDTH, &s->wo_a},
		{"attn.wo_b", DS41_HIDDEN, DS41_O_WIDTH, &s->wo_b}, {"ffn.shared_experts.w1", DS41_MOE, DS41_HIDDEN, &s->shared_w1},
		{"ffn.shared_experts.w2", DS41_HIDDEN, DS41_MOE, &s->shared_w2}, {"ffn.shared_experts.w3", DS41_MOE, DS41_HIDDEN, &s->shared_w3},
	};
	const struct
	{
		const char *name;
		uint32_t rows;
		uint32_t columns;
	} parts[DS41_EXPERT_PARTS / 2u] = {{"w1", DS41_MOE, DS41_HIDDEN}, {"w2", DS41_HIDDEN, DS41_MOE}, {"w3", DS41_MOE, DS41_HIDDEN}};
	SparkStatus status = SPARK_STATUS_OK;
	for ( uint32_t index = 0u; index < sizeof(plain) / sizeof(plain[0]) && status == SPARK_STATUS_OK; index++ )
		status = Ds41Named(state, document, payload_base, "mtp.%u.%s", stage, plain[index].suffix, plain[index].bytes, plain[index].pointer);
	for ( uint32_t index = 0u; index < sizeof(matrices) / sizeof(matrices[0]) && status == SPARK_STATUS_OK; index++ )
		status = Ds41MatrixLoad(state, document, payload_base, stage, matrices[index].prefix, matrices[index].rows, matrices[index].columns,
			matrices[index].matrix);
	for ( uint32_t expert = 0u; expert < DS41_EXPERTS && status == SPARK_STATUS_OK; expert++ )
		for ( uint32_t part = 0u; part < DS41_EXPERT_PARTS / 2u && status == SPARK_STATUS_OK; part++ )
		{
			char name[128];
			const void *pointer = 0;
			snprintf(name, sizeof(name), "ffn.experts.%u.%s.scale", expert, parts[part].name);
			status = Ds41Named(state, document, payload_base, "mtp.%u.%s", stage, name,
				(uint64_t)parts[part].rows * (parts[part].columns / DS41_QDQ), &pointer);
			table[((uint64_t)stage * DS41_EXPERTS + expert) * DS41_EXPERT_PARTS + 2u * part] = (const uint8_t *)pointer;
			snprintf(name, sizeof(name), "ffn.experts.%u.%s.weight", expert, parts[part].name);
			if ( status == SPARK_STATUS_OK )
				status = Ds41Named(state, document, payload_base, "mtp.%u.%s", stage, name, (uint64_t)parts[part].rows * (parts[part].columns / 2u),
					&pointer);
			table[((uint64_t)stage * DS41_EXPERTS + expert) * DS41_EXPERT_PARTS + 2u * part + 1u] = (const uint8_t *)pointer;
		}
	return status;
}

static SparkStatus Ds41Weights(Ds41State *state, const char *pack_path, const uint8_t **table)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = Ds41LoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Shape(state, &document);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Tensor(state, &document, payload_base, "target.embed", (uint64_t)DS41_VOCAB * DS41_HIDDEN * sizeof(uint16_t),
			(const void **)&state->embed);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Tensor(state, &document, payload_base, "target.head", (uint64_t)DS41_VOCAB * DS41_HIDDEN * sizeof(uint16_t),
			(const void **)&state->head);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Named(state, &document, payload_base, "mtp.%u.%s", DS41_STAGES - 1u, "norm.weight", DS41_HIDDEN * sizeof(uint16_t),
			(const void **)&state->final_norm);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Named(state, &document, payload_base, "mtp.%u.%s", DS41_STAGES - 1u, "markov_head.embed.weight",
			(uint64_t)DS41_VOCAB * DS41_MARKOV * sizeof(uint16_t), (const void **)&state->markov_embed);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Named(state, &document, payload_base, "mtp.%u.%s", DS41_STAGES - 1u, "markov_head.head.weight",
			(uint64_t)DS41_VOCAB * DS41_MARKOV * sizeof(uint16_t), (const void **)&state->markov_head);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Named(state, &document, payload_base, "mtp.%u.%s", DS41_STAGES - 1u, "confidence_head.proj.weight",
			DS41_CONFIDENCE * sizeof(uint16_t), (const void **)&state->confidence);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Named(state, &document, payload_base, "mtp.%u.%s", 0u, "main_norm.weight", DS41_HIDDEN * sizeof(uint16_t),
			(const void **)&state->main_norm);
	if ( status == SPARK_STATUS_OK )
		status = Ds41MatrixLoad(state, &document, payload_base, 0u, "main_proj", DS41_HIDDEN, DS41_TAP_WIDTH, &state->main_proj);
	for ( uint32_t stage = 0u; stage < DS41_STAGES && status == SPARK_STATUS_OK; stage++ )
		status = Ds41StageLoad(state, &document, payload_base, stage, table);
	SparkJsonDocumentDestroy(&document);
	return status;
}

#define DS41_DEVICE(field, count) (SparkGpuMalloc((void **)&(field), (size_t)(count) * sizeof(*(field))) == SPARK_GPU_SUCCESS)

static SparkStatus Ds41Expand(Ds41State *state, Ds41Matrix *matrix, uint64_t *used)
{
	const uint32_t groups = matrix->columns / DS41_QDQ;
	matrix->rows_scale = state->rows_scales + *used;
	*used += (uint64_t)matrix->rows * groups;
	LM_LAUNCH((Ds41ExpandScalesKernel), state->multiprocessors, DS41_THREADS, 0, state->stream, matrix->scale, matrix->rows_scale, matrix->rows, groups);
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Ds41Tables(Ds41State *state, const uint8_t **table)
{
	static const float initial[DS41_BLOCK * DS41_HC] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f, 0.0f};
	uint64_t used = 0u;
	SparkStatus status;
	if ( !DS41_DEVICE(state->rows_scales, state->rows_scales_bytes) || !DS41_DEVICE(state->expert_table, (uint64_t)DS41_STAGES * DS41_EXPERTS * DS41_EXPERT_PARTS) ||
		!DS41_DEVICE(state->rope_inverse, DS41_ROPE / 2u) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( SparkGpuMemcpy(state->expert_table, table, (size_t)DS41_STAGES * DS41_EXPERTS * DS41_EXPERT_PARTS * sizeof(*table), SPARK_GPU_COPY_HOST_TO_DEVICE) !=
		SPARK_GPU_SUCCESS || SparkGpuMemcpy(state->forward.pre_initial, initial, sizeof(initial), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	status = Ds41Expand(state, &state->main_proj, &used);
	for ( uint32_t stage = 0u; stage < DS41_STAGES && status == SPARK_STATUS_OK; stage++ )
	{
		Ds41Stage *s = &state->stages[stage];
		Ds41Matrix *matrices[] = {&s->wq_a, &s->wq_b, &s->wkv, &s->wo_a, &s->wo_b, &s->shared_w1, &s->shared_w2, &s->shared_w3};
		s->experts = (const uint8_t *const *)(state->expert_table + (uint64_t)stage * DS41_EXPERTS * DS41_EXPERT_PARTS);
		for ( uint32_t index = 0u; index < sizeof(matrices) / sizeof(matrices[0]) && status == SPARK_STATUS_OK; index++ )
			status = Ds41Expand(state, matrices[index], &used);
	}
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((Ds41RopeTableKernel), 1u, DS41_THREADS, 0, state->stream, state->rope_inverse, state->rope_theta);
	if ( status == SPARK_STATUS_OK && (SparkGpuGetLastError() != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
		status = SPARK_STATUS_INTERNAL_ERROR;
	return status;
}

static SparkStatus Ds41Scratch(Ds41State *state, uint64_t context_bytes)
{
	Ds41Forward *forward = &state->forward;
	int multiprocessors = 0;
	uint64_t capacity = context_bytes / DS41_RING_BYTES;
	capacity = capacity < 1u ? 1u : capacity > DS41_SEQUENCES_MAX ? DS41_SEQUENCES_MAX : capacity;
	state->capacity = (uint32_t)capacity;
	state->sequences = (Ds41Sequence *)calloc(capacity, sizeof(Ds41Sequence));
	if ( state->sequences == 0 ||
		SparkGpuStreamCreateWithFlags(&state->stream, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuFuncSetAttribute(Ds41ExpertUpKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)DS41_UP_SHARED_BYTES) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_taps, (size_t)DS41_INGEST_ROWS * DS41_TAP_WIDTH * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&state->host_positions, (size_t)DS41_INGEST_ROWS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_draft, sizeof(Ds41Draft)) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost((void **)&forward->host_picks, sizeof(Ds41Picks)) != SPARK_GPU_SUCCESS ||
		!DS41_DEVICE(state->rings, capacity * DS41_RING_ELEMENTS) || !DS41_DEVICE(state->taps, (uint64_t)DS41_INGEST_ROWS * DS41_TAP_WIDTH) ||
		!DS41_DEVICE(state->taps_quant, (uint64_t)DS41_INGEST_ROWS * DS41_TAP_WIDTH) || !DS41_DEVICE(state->positions, DS41_INGEST_ROWS) ||
		!DS41_DEVICE(state->main_raw, (uint64_t)DS41_INGEST_ROWS * DS41_HIDDEN) || !DS41_DEVICE(state->main, (uint64_t)DS41_INGEST_ROWS * DS41_HIDDEN) ||
		!DS41_DEVICE(state->main_quant, (uint64_t)DS41_INGEST_ROWS * DS41_HIDDEN) || !DS41_DEVICE(state->key_raw, (uint64_t)DS41_INGEST_ROWS * DS41_HEAD_DIM) ||
		!DS41_DEVICE(forward->draft, 1u) || !DS41_DEVICE(forward->picks, 1u) || !DS41_DEVICE(forward->streams[0], DS41_BLOCK * DS41_HC_FLAT) ||
		!DS41_DEVICE(forward->streams[1], DS41_BLOCK * DS41_HC_FLAT) ||
		!DS41_DEVICE(forward->mix_partials, DS41_BLOCK * LmHcSinkhornShape<DS41_HC>::kPartials) ||
		!DS41_DEVICE(forward->pre_initial, DS41_BLOCK * DS41_HC) || !DS41_DEVICE(forward->attn_pre, DS41_BLOCK * DS41_HC) ||
		!DS41_DEVICE(forward->attn_post, DS41_BLOCK * DS41_HC) || !DS41_DEVICE(forward->attn_comb, DS41_BLOCK * DS41_HC * DS41_HC) ||
		!DS41_DEVICE(forward->ffn_pre[0], DS41_BLOCK * DS41_HC) || !DS41_DEVICE(forward->ffn_pre[1], DS41_BLOCK * DS41_HC) ||
		!DS41_DEVICE(forward->ffn_post, DS41_BLOCK * DS41_HC) || !DS41_DEVICE(forward->ffn_comb, DS41_BLOCK * DS41_HC * DS41_HC) ||
		!DS41_DEVICE(forward->collapsed, DS41_BLOCK * DS41_HIDDEN) || !DS41_DEVICE(forward->x, DS41_BLOCK * DS41_HIDDEN) ||
		!DS41_DEVICE(forward->xq, DS41_BLOCK * DS41_HIDDEN) || !DS41_DEVICE(forward->q_raw, DS41_BLOCK * DS41_QUERY_RANK) ||
		!DS41_DEVICE(forward->q_normed, DS41_BLOCK * DS41_QUERY_RANK) || !DS41_DEVICE(forward->q_quant, DS41_BLOCK * DS41_QUERY_RANK) ||
		!DS41_DEVICE(forward->q, (uint64_t)DS41_BLOCK * DS41_QUERY_WIDTH) || !DS41_DEVICE(forward->kv_raw, DS41_BLOCK * DS41_HEAD_DIM) ||
		!DS41_DEVICE(forward->kv, DS41_BLOCK * DS41_HEAD_DIM) || !DS41_DEVICE(forward->attended, (uint64_t)DS41_BLOCK * DS41_QUERY_WIDTH) ||
		!DS41_DEVICE(forward->grouped, (uint64_t)DS41_BLOCK * DS41_QUERY_WIDTH) || !DS41_DEVICE(forward->projected, DS41_BLOCK * DS41_O_WIDTH) ||
		!DS41_DEVICE(forward->projected_quant, DS41_BLOCK * DS41_O_WIDTH) || !DS41_DEVICE(forward->attn_out, DS41_BLOCK * DS41_HIDDEN) ||
		!DS41_DEVICE(forward->route_weight, DS41_PAIRS) || !DS41_DEVICE(forward->route_expert, DS41_PAIRS) ||
		!DS41_DEVICE(forward->gate_scores, DS41_BLOCK * DS41_EXPERTS) ||
		!DS41_DEVICE(forward->pair_gate, DS41_PAIRS * DS41_MOE) || !DS41_DEVICE(forward->pair_up, DS41_PAIRS * DS41_MOE) ||
		!DS41_DEVICE(forward->pair_hidden, DS41_PAIRS * DS41_MOE) || !DS41_DEVICE(forward->pair_down, DS41_PAIRS * DS41_HIDDEN) ||
		!DS41_DEVICE(forward->shared_gate, DS41_BLOCK * DS41_MOE) || !DS41_DEVICE(forward->shared_up, DS41_BLOCK * DS41_MOE) ||
		!DS41_DEVICE(forward->shared_hidden, DS41_BLOCK * DS41_MOE) || !DS41_DEVICE(forward->shared_out, DS41_BLOCK * DS41_HIDDEN) ||
		!DS41_DEVICE(forward->moe_out, DS41_BLOCK * DS41_HIDDEN) || !DS41_DEVICE(forward->hidden, DS41_BLOCK * DS41_HIDDEN) ||
		!DS41_DEVICE(forward->normed, DS41_BLOCK * DS41_HIDDEN) || !DS41_DEVICE(forward->logits, (uint64_t)DS41_BLOCK * DS41_VOCAB) ||
		!DS41_DEVICE(forward->row, DS41_VOCAB) || !DS41_DEVICE(forward->candidates, DS41_SAMPLED) ||
		!DS41_DEVICE(forward->softmax_partials, (uint64_t)DS41_BLOCK * 8u * (uint64_t)multiprocessors) ||
		!DS41_DEVICE(forward->topk_values, 2u * LmTopkExactCandidateEntries(DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK)) ||
		!DS41_DEVICE(forward->topk_positions, 2u * LmTopkExactCandidateEntries(DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK)) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		state->sequences[index].ring = state->rings + (uint64_t)index * DS41_RING_ELEMENTS;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static void Ds41Reset(Ds41Sequence *sequence)
{
	for ( uint32_t slot = 0u; slot < DS41_WINDOW; slot++ )
	{
		sequence->positions[slot] = -1;
		sequence->tokens[slot] = 0u;
	}
	sequence->retired = 0u;
}

static Ds41Sequence *Ds41Live(Ds41State *state, uint64_t id)
{
	for ( uint32_t index = 0u; index < state->capacity; index++ )
		if ( state->sequences[index].live != 0u && state->sequences[index].retired == 0u && state->sequences[index].id == id )
			return &state->sequences[index];
	return 0;
}

static Ds41Sequence *Ds41Claim(Ds41State *state)
{
	Ds41Sequence *oldest = 0;
	for ( uint32_t index = 0u; index < state->capacity; index++ )
	{
		Ds41Sequence *sequence = &state->sequences[index];
		if ( sequence->live == 0u )
			return sequence;
		if ( oldest == 0 || sequence->last_used < oldest->last_used )
			oldest = sequence;
	}
	return oldest;
}

static uint32_t Ds41Held(const Ds41Sequence *sequence, uint64_t position)
{
	return sequence->positions[position % DS41_WINDOW] == (int64_t)position ? 1u : 0u;
}

static SparkStatus Ds41Linear(Ds41State *state, const Ds41Matrix *matrix, uint64_t row_offset, uint32_t outputs, const uint16_t *input, uint16_t *output,
	uint32_t rows)
{
	const uint32_t groups = matrix->columns / DS41_QDQ;
	for ( uint32_t first = 0u; first < rows; first += DS41_LINEAR_ROWS )
	{
		LmSkinnyArguments args;
		memset(&args, 0, sizeof(args));
		args.weight = matrix->weight + row_offset * matrix->columns;
		args.scale = LmScaleTensorRowsUe8m0(matrix->rows_scale + row_offset * groups, outputs, matrix->columns, DS41_QDQ);
		args.activation = input + (uint64_t)first * matrix->columns;
		args.output_bf16 = output + (uint64_t)first * outputs;
		args.rows = rows - first < DS41_LINEAR_ROWS ? rows - first : DS41_LINEAR_ROWS;
		args.input_dimension = matrix->columns;
		args.output_dimension = outputs;
		if ( LmSkinnyLaunch<LmMxfp8>(&args, state->stream) != LM_LAUNCH_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
	}
	return SPARK_STATUS_OK;
}

static void Ds41Qdq(Ds41State *state, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t parts, uint32_t part_width, uint32_t to_part_major)
{
	const uint64_t groups = (uint64_t)rows * parts * part_width / DS41_QDQ;
	const uint64_t blocks = (groups + DS41_WARPS - 1u) / DS41_WARPS;
	const uint32_t grid = blocks < 4u * state->multiprocessors ? (uint32_t)blocks : 4u * state->multiprocessors;
	LM_LAUNCH((Ds41QdqKernel), grid, DS41_THREADS, 0, state->stream, input, output, rows, parts, part_width, to_part_major);
}

static SparkStatus Ds41IngestRows(Ds41State *state, Ds41Sequence *sequence, uint32_t rows)
{
	SparkStatus status;
	Ds41Qdq(state, state->taps, state->taps_quant, rows, 1u, DS41_TAP_WIDTH, 0u);
	status = Ds41Linear(state, &state->main_proj, 0u, DS41_HIDDEN, state->taps_quant, state->main_raw, rows);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Ds41RmsKernel), rows, DS41_THREADS, 0, state->stream, state->main_raw, state->main_norm, state->main, DS41_HIDDEN, state->epsilon);
	Ds41Qdq(state, state->main, state->main_quant, rows, 1u, DS41_HIDDEN, 0u);
	for ( uint32_t stage = 0u; stage < DS41_STAGES && status == SPARK_STATUS_OK; stage++ )
	{
		status = Ds41Linear(state, &state->stages[stage].wkv, 0u, DS41_HEAD_DIM, state->main_quant, state->key_raw, rows);
		if ( status == SPARK_STATUS_OK )
			LM_LAUNCH((Ds41KeyRowsKernel), rows, DS41_THREADS, 0, state->stream, state->key_raw, state->stages[stage].kv_norm, state->rope_inverse,
				state->positions, state->epsilon, sequence->ring + (uint64_t)stage * DS41_STAGE_RING, (uint16_t *)0);
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	return status;
}

static SparkStatus Ds41Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	Ds41State *state = (Ds41State *)opaque;
	const uint32_t rows = frame->header.row_count;
	const uint64_t first = frame->header.first_position;
	const uint64_t hidden = frame->hello != 0 ? frame->hello->hidden : 0u;
	const uint32_t skip = rows > DS41_WINDOW ? rows - DS41_WINDOW : 0u;
	Ds41Sequence *sequence;
	SparkStatus status;
	if ( rows == 0u )
		return SPARK_STATUS_OK;
	if ( frame->required_count != DS41_TAPS || hidden != DS41_HIDDEN || frame->rows == 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	for ( uint32_t tap = 0u; tap < DS41_TAPS; tap++ )
		if ( ((uint64_t)frame->required_index[tap] + 1u) * DS41_HIDDEN > frame->row_elements )
			return SPARK_STATUS_SCHEMA_ERROR;
	if ( first + rows > UINT32_MAX )
		return SPARK_STATUS_INVALID_ARGUMENT;
	sequence = Ds41Live(state, frame->header.sequence_id);
	if ( sequence != 0 && first != 0u &&
		(sequence->control_generation != frame->header.control_generation || sequence->request_id != frame->header.request_id) )
	{
		sequence->retired = 1u;
		sequence->last_used = ++state->clock;
		sequence = 0;
	}
	if ( sequence == 0 )
	{
		sequence = Ds41Claim(state);
		Ds41Reset(sequence);
		sequence->live = 1u;
		sequence->id = frame->header.sequence_id;
	}
	else if ( first == 0u )
		Ds41Reset(sequence);
	sequence->control_generation = frame->header.control_generation;
	sequence->request_id = frame->header.request_id;
	sequence->last_used = ++state->clock;
	for ( uint32_t slot = 0u; slot < DS41_WINDOW; slot++ )
		if ( sequence->positions[slot] >= (int64_t)first )
			sequence->positions[slot] = -1;
	for ( uint32_t row = skip; row < rows; row++ )
	{
		const uint16_t *source = frame->rows + (uint64_t)row * frame->row_elements;
		for ( uint32_t tap = 0u; tap < DS41_TAPS; tap++ )
			memcpy(state->host_taps + (uint64_t)(row - skip) * DS41_TAP_WIDTH + (uint64_t)tap * DS41_HIDDEN,
				source + (uint64_t)frame->required_index[tap] * DS41_HIDDEN, DS41_HIDDEN * sizeof(uint16_t));
		state->host_positions[row - skip] = (uint32_t)(first + row);
	}
	status = SparkGpuMemcpyAsync(state->taps, state->host_taps, (size_t)(rows - skip) * DS41_TAP_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE,
		state->stream) == SPARK_GPU_SUCCESS && SparkGpuMemcpyAsync(state->positions, state->host_positions, (size_t)(rows - skip) * sizeof(uint32_t),
		SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = Ds41IngestRows(state, sequence, rows - skip);
	else
		(void)SparkGpuStreamSynchronize(state->stream);
	if ( status == SPARK_STATUS_OK )
		for ( uint32_t row = skip; row < rows; row++ )
		{
			const uint64_t position = first + row;
			sequence->positions[position % DS41_WINDOW] = (int64_t)position;
			sequence->tokens[position % DS41_WINDOW] = frame->tokens != 0 ? frame->tokens[row] : 0u;
		}
	state->frames++;
	state->rows += rows;
	return status;
}

static SparkStatus Ds41FillFromDonors(Ds41State *state, Ds41Sequence *sequence, const SparkDraftServerRequest *request, uint64_t first)
{
	uint8_t tried[DS41_SEQUENCES_MAX];
	memset(tried, 0, sizeof(tried));
	for ( ;; )
	{
		Ds41Sequence *donor = 0;
		uint32_t chosen = 0u, missing = 0u;
		for ( uint64_t position = first; position < request->position; position++ )
			missing += Ds41Held(sequence, position) == 0u ? 1u : 0u;
		if ( missing == 0u )
			return SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < state->capacity; index++ )
		{
			Ds41Sequence *other = &state->sequences[index];
			if ( other->live != 0u && other != sequence && tried[index] == 0u && (donor == 0 || other->last_used > donor->last_used) )
			{
				donor = other;
				chosen = index;
			}
		}
		if ( donor == 0 )
			return SPARK_STATUS_OK;
		tried[chosen] = 1u;
		for ( uint64_t position = first; position < request->position; position++ )
		{
			const uint32_t slot = (uint32_t)(position % DS41_WINDOW);
			if ( Ds41Held(sequence, position) != 0u || Ds41Held(donor, position) == 0u || position >= request->committed_token_count ||
				donor->tokens[slot] != request->committed_token_ids[position] )
				continue;
			for ( uint32_t stage = 0u; stage < DS41_STAGES; stage++ )
				if ( SparkGpuMemcpyAsync(sequence->ring + (uint64_t)stage * DS41_STAGE_RING + (uint64_t)slot * DS41_HEAD_DIM,
					donor->ring + (uint64_t)stage * DS41_STAGE_RING + (uint64_t)slot * DS41_HEAD_DIM, DS41_HEAD_DIM * sizeof(uint16_t),
					SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
					return SPARK_STATUS_IO_ERROR;
			sequence->positions[slot] = (int64_t)position;
			sequence->tokens[slot] = donor->tokens[slot];
			state->donor_fills++;
		}
	}
}

static SparkStatus Ds41Attention(Ds41State *state, uint32_t stage, const uint16_t *ring)
{
	Ds41Forward *forward = &state->forward;
	const Ds41Stage *s = &state->stages[stage];
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(Ds41Draft, positions));
	SparkStatus status;
	Ds41Qdq(state, forward->x, forward->xq, DS41_BLOCK, 1u, DS41_HIDDEN, 0u);
	status = Ds41Linear(state, &s->wq_a, 0u, DS41_QUERY_RANK, forward->xq, forward->q_raw, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Ds41RmsKernel), DS41_BLOCK, DS41_THREADS, 0, stream, forward->q_raw, s->q_norm, forward->q_normed, DS41_QUERY_RANK, state->epsilon);
	Ds41Qdq(state, forward->q_normed, forward->q_quant, DS41_BLOCK, 1u, DS41_QUERY_RANK, 0u);
	status = Ds41Linear(state, &s->wq_b, 0u, DS41_QUERY_WIDTH, forward->q_quant, forward->q, DS41_BLOCK);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Linear(state, &s->wkv, 0u, DS41_HEAD_DIM, forward->xq, forward->kv_raw, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Ds41RopeKernel), (DS41_BLOCK * DS41_HEADS * (DS41_ROPE / 2u) + DS41_THREADS - 1u) / DS41_THREADS, DS41_THREADS, 0, stream, forward->q,
		forward->draft, state->rope_inverse, DS41_HEADS, 1.0f);
	LM_LAUNCH((Ds41KeyRowsKernel), DS41_BLOCK, DS41_THREADS, 0, stream, forward->kv_raw, s->kv_norm, state->rope_inverse, positions, state->epsilon,
		(uint16_t *)0, forward->kv);
	LM_LAUNCH((Ds41AttentionKernel), dim3(DS41_HEADS, DS41_BLOCK), DS41_THREADS, 0, stream, forward->q, ring, forward->kv, forward->draft, s->sink,
		state->softmax_scale, forward->attended);
	LM_LAUNCH((Ds41RopeKernel), (DS41_BLOCK * DS41_HEADS * (DS41_ROPE / 2u) + DS41_THREADS - 1u) / DS41_THREADS, DS41_THREADS, 0, stream,
		forward->attended, forward->draft, state->rope_inverse, DS41_HEADS, -1.0f);
	Ds41Qdq(state, forward->attended, forward->grouped, DS41_BLOCK, DS41_GROUPS, DS41_GROUP_WIDTH, 1u);
	for ( uint32_t group = 0u; group < DS41_GROUPS && status == SPARK_STATUS_OK; group++ )
		status = Ds41Linear(state, &s->wo_a, (uint64_t)group * DS41_O_RANK, DS41_O_RANK, forward->grouped + (uint64_t)group * DS41_BLOCK * DS41_GROUP_WIDTH,
			forward->projected + (uint64_t)group * DS41_BLOCK * DS41_O_RANK, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	Ds41Qdq(state, forward->projected, forward->projected_quant, DS41_BLOCK, DS41_GROUPS, DS41_O_RANK, 0u);
	return Ds41Linear(state, &s->wo_b, 0u, DS41_HIDDEN, forward->projected_quant, forward->attn_out, DS41_BLOCK);
}

static SparkStatus Ds41Moe(Ds41State *state, uint32_t stage)
{
	Ds41Forward *forward = &state->forward;
	const Ds41Stage *s = &state->stages[stage];
	const SparkGpuStream stream = state->stream;
	const uint32_t swiglu_blocks = (DS41_PAIRS * DS41_MOE / DS41_QDQ + DS41_WARPS - 1u) / DS41_WARPS;
	SparkStatus status;
	LM_LAUNCH((Ds41GateScoreKernel), DS41_EXPERTS / DS41_WARPS, DS41_THREADS, 0, stream, forward->x, s->gate, forward->gate_scores);
	LM_LAUNCH((Ds41GateTopKernel), DS41_BLOCK, DS41_WARP, 0, stream, forward->gate_scores, s->gate_bias, state->route_scale, forward->route_expert,
		forward->route_weight);
	Ds41Qdq(state, forward->x, forward->xq, DS41_BLOCK, 1u, DS41_HIDDEN, 0u);
	LM_LAUNCH((Ds41ExpertUpKernel), dim3(DS41_PAIRS, DS41_MOE / DS41_EXPERT_NEURONS), DS41_THREADS, DS41_UP_SHARED_BYTES, stream, forward->xq,
		s->experts, forward->route_expert, forward->pair_gate, forward->pair_up);
	LM_LAUNCH((Ds41SwigluKernel), swiglu_blocks, DS41_THREADS, 0, stream, forward->pair_gate, forward->pair_up, forward->route_weight, state->swiglu_limit,
		DS41_PAIRS, forward->pair_hidden);
	LM_LAUNCH((Ds41ExpertDownKernel), dim3(DS41_PAIRS, DS41_HIDDEN / DS41_EXPERT_NEURONS), DS41_THREADS, DS41_DOWN_SHARED_BYTES, stream,
		forward->pair_hidden, s->experts, forward->route_expert, forward->pair_down);
	status = Ds41Linear(state, &s->shared_w1, 0u, DS41_MOE, forward->xq, forward->shared_gate, DS41_BLOCK);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Linear(state, &s->shared_w3, 0u, DS41_MOE, forward->xq, forward->shared_up, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Ds41SwigluKernel), (DS41_BLOCK * DS41_MOE / DS41_QDQ + DS41_WARPS - 1u) / DS41_WARPS, DS41_THREADS, 0, stream, forward->shared_gate,
		forward->shared_up, (const float *)0, state->swiglu_limit, DS41_BLOCK, forward->shared_hidden);
	status = Ds41Linear(state, &s->shared_w2, 0u, DS41_HIDDEN, forward->shared_hidden, forward->shared_out, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Ds41MoeSumKernel), dim3(DS41_HIDDEN / DS41_THREADS, DS41_BLOCK), DS41_THREADS, 0, stream, forward->pair_down, forward->shared_out,
		forward->moe_out);
	return SPARK_STATUS_OK;
}

static SparkStatus Ds41Mixes(Ds41State *state, const uint16_t *streams, const float *fn, const float *scale, const float *base, float *pre, float *post,
	float *comb)
{
	return LmHcSinkhornMixes<DS41_HC>(streams, fn, scale, base, state->forward.mix_partials, pre, post, comb, DS41_BLOCK, DS41_HIDDEN, state->epsilon,
		DS41_SINKHORN, state->hc_epsilon, state->stream) == LM_LAUNCH_OK ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static void Ds41Collapse(Ds41State *state, const uint16_t *streams, const float *pre, uint16_t *output)
{
	LM_LAUNCH((LmHcCollapseKernel<DS41_HC>), dim3(DS41_HIDDEN / DS41_THREADS, DS41_BLOCK), DS41_THREADS, 0, state->stream, streams, pre, output, DS41_HIDDEN);
}

static SparkStatus Ds41RunForward(Ds41State *state, const uint16_t *ring_base, uint32_t stages, const uint16_t **streams_out, const float **pre_out)
{
	Ds41Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const float *pre = forward->pre_initial;
	uint32_t current = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((Ds41EmbedKernel), DS41_BLOCK, DS41_THREADS, 0, stream, state->embed, forward->draft, forward->streams[0]);
	for ( uint32_t stage = 0u; stage < stages && status == SPARK_STATUS_OK; stage++ )
	{
		const Ds41Stage *s = &state->stages[stage];
		status = Ds41Mixes(state, forward->streams[current], s->hc_attn_fn, s->hc_attn_scale, s->hc_attn_base, forward->attn_pre, forward->attn_post,
			forward->attn_comb);
		if ( status != SPARK_STATUS_OK )
			break;
		Ds41Collapse(state, forward->streams[current], pre, forward->collapsed);
		LM_LAUNCH((Ds41RmsKernel), DS41_BLOCK, DS41_THREADS, 0, stream, forward->collapsed, s->attn_norm, forward->x, DS41_HIDDEN, state->epsilon);
		status = Ds41Attention(state, stage, ring_base + (uint64_t)stage * DS41_STAGE_RING);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Ds41HcPostKernel), dim3(DS41_HIDDEN / DS41_THREADS, DS41_BLOCK), DS41_THREADS, 0, stream, forward->attn_out, forward->streams[current],
			forward->attn_post, forward->attn_comb, forward->streams[1u - current]);
		current = 1u - current;
		status = Ds41Mixes(state, forward->streams[current], s->hc_ffn_fn, s->hc_ffn_scale, s->hc_ffn_base, forward->ffn_pre[stage % 2u], forward->ffn_post,
			forward->ffn_comb);
		if ( status != SPARK_STATUS_OK )
			break;
		Ds41Collapse(state, forward->streams[current], forward->attn_pre, forward->collapsed);
		LM_LAUNCH((Ds41RmsKernel), DS41_BLOCK, DS41_THREADS, 0, stream, forward->collapsed, s->ffn_norm, forward->x, DS41_HIDDEN, state->epsilon);
		status = Ds41Moe(state, stage);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Ds41HcPostKernel), dim3(DS41_HIDDEN / DS41_THREADS, DS41_BLOCK), DS41_THREADS, 0, stream, forward->moe_out, forward->streams[current],
			forward->ffn_post, forward->ffn_comb, forward->streams[1u - current]);
		current = 1u - current;
		pre = forward->ffn_pre[stage % 2u];
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( streams_out != 0 )
		*streams_out = forward->streams[current];
	if ( pre_out != 0 )
		*pre_out = pre;
	if ( stages < DS41_STAGES )
		return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	Ds41Collapse(state, forward->streams[current], pre, forward->hidden);
	LM_LAUNCH((Ds41RmsKernel), DS41_BLOCK, DS41_THREADS, 0, stream, forward->hidden, state->final_norm, forward->normed, DS41_HIDDEN, state->epsilon);
	if ( LmDenseRows<LmBf16Format>(state->head, LmScaleTensorNone(), forward->normed, (uint16_t *)0, forward->logits, DS41_BLOCK, DS41_HIDDEN, DS41_VOCAB, 0u,
		state->multiprocessors, stream) != LM_LAUNCH_OK )
		return SPARK_STATUS_INTERNAL_ERROR;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Ds41Heads(Ds41State *state, uint32_t greedy, uint32_t drafts, uint32_t tree)
{
	Ds41Forward *forward = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t blocks = 4u * state->multiprocessors;
	const uint64_t entries = 2u * LmTopkExactCandidateEntries(DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK);
	if ( SparkGpuMemsetAsync(forward->picks, 0, sizeof(Ds41Picks), stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	for ( uint32_t step = 0u; step < DS41_BLOCK; step++ )
	{
		LM_LAUNCH((Ds41MarkovRowKernel), blocks, DS41_THREADS, 0, stream, step, forward->draft, forward->picks, forward->logits, state->markov_embed,
			state->markov_head, forward->row, forward->softmax_partials);
		if ( greedy != 0u )
			continue;
		if ( LmTopkExactLaunch<DS41_TOPK_THREADS>(forward->row, 1u, DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS,
			forward->topk_values, forward->topk_positions, entries, forward->candidates, stream) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		LM_LAUNCH((Ds41SampleKernel), 1u, DS41_PICK_THREADS, 0, stream, step, forward->draft, forward->row, forward->candidates,
			forward->softmax_partials + (uint64_t)step * 2u * blocks, blocks, forward->picks);
	}
	LM_LAUNCH((Ds41ConfidenceKernel), DS41_BLOCK, DS41_THREADS, 0, stream, forward->draft, forward->picks, forward->hidden, state->markov_embed,
		state->confidence, forward->softmax_partials, blocks);
	if ( tree != 0u && drafts >= 2u )
	{
		LM_LAUNCH((Ds41BranchSelectKernel), 1u, DS41_WARP, 0, stream, forward->picks, drafts, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE);
		for ( uint32_t offset = 1u; offset < DS41_BRANCH_LEN && offset < drafts; offset++ )
		{
			LM_LAUNCH((Ds41BranchRowKernel), blocks, DS41_THREADS, 0, stream, offset, drafts, forward->draft, forward->picks, forward->logits,
				state->markov_embed, state->markov_head, forward->softmax_partials);
			LM_LAUNCH((Ds41BranchReduceKernel), DS41_BRANCHES, DS41_THREADS, 0, stream, offset, drafts, forward->draft, forward->picks,
				forward->softmax_partials, blocks);
		}
	}
	if ( SparkGpuMemcpyAsync(forward->host_picks, forward->picks, sizeof(Ds41Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Ds41Prepare(Ds41State *state, const Ds41Sequence *sequence, uint32_t anchor, uint64_t position, const SparkRowSampling *sampling)
{
	Ds41Draft *draft = state->forward.host_draft;
	if ( anchor >= DS41_VOCAB || position + DS41_BLOCK > UINT32_MAX )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(draft, 0, sizeof(*draft));
	for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	for ( uint32_t slot = 0u; slot < DS41_WINDOW; slot++ )
	{
		const int64_t held = sequence->positions[slot];
		draft->valid[slot] = held >= 0 && held < (int64_t)position && held >= (int64_t)position - (int64_t)DS41_WINDOW ? 1u : 0u;
	}
	draft->anchor = anchor;
	draft->noise = state->noise_token;
	draft->greedy = sampling->inverse_temperature > 0.0f ? 0u : 1u;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	return SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) == SPARK_GPU_SUCCESS ?
		SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus Ds41Settle(Ds41State *state, SparkStatus status)
{
	if ( SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	return status;
}

static void Ds41ForwardFree(Ds41Forward *forward)
{
	void *buffers[] = {
		forward->draft, forward->picks, forward->streams[0], forward->streams[1], forward->mix_partials, forward->pre_initial, forward->attn_pre,
		forward->attn_post, forward->attn_comb, forward->ffn_pre[0], forward->ffn_pre[1], forward->ffn_post, forward->ffn_comb, forward->collapsed,
		forward->x, forward->xq, forward->q_raw, forward->q_normed, forward->q_quant, forward->q, forward->kv_raw, forward->kv, forward->attended,
		forward->grouped, forward->projected, forward->projected_quant, forward->attn_out, forward->route_weight, forward->route_expert, forward->gate_scores,
		forward->pair_gate, forward->pair_up, forward->pair_hidden, forward->pair_down, forward->shared_gate, forward->shared_up, forward->shared_hidden,
		forward->shared_out, forward->moe_out, forward->hidden, forward->normed, forward->logits, forward->row, forward->candidates,
		forward->softmax_partials, forward->topk_values, forward->topk_positions,
	};
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		(void)SparkGpuFree(buffers[index]);
	if ( forward->host_draft != 0 )
		(void)SparkGpuFreeHost(forward->host_draft);
	if ( forward->host_picks != 0 )
		(void)SparkGpuFreeHost(forward->host_picks);
}

static SparkStatus Ds41Close(void *opaque)
{
	Ds41State *state = (Ds41State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	void *buffers[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	Ds41ForwardFree(&state->forward);
	buffers[0] = state->rings;
	buffers[1] = state->taps;
	buffers[2] = state->taps_quant;
	buffers[3] = state->positions;
	buffers[4] = state->main_raw;
	buffers[5] = state->main;
	buffers[6] = state->main_quant;
	buffers[7] = state->key_raw;
	buffers[8] = state->rows_scales;
	buffers[9] = (void *)state->expert_table;
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); index++ )
		(void)SparkGpuFree(buffers[index]);
	(void)SparkGpuFree(state->rope_inverse);
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

static SparkStatus Ds41Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	Ds41State *state;
	const uint8_t **table;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
		return Ds41Refuse(bind, SPARK_STATUS_ABI_MISMATCH, "the lane descriptor is not drafter ABI %s", "1");
	if ( Ds41DeviceOpen() != SPARK_STATUS_OK || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
		return Ds41Refuse(bind, SPARK_STATUS_IO_ERROR, "lane %s cannot read the device's free memory", lane->lane_name);
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
	if ( lane->context_bytes < DS41_RING_BYTES )
		return Ds41Refuse(bind, SPARK_STATUS_INVALID_ARGUMENT, "lane %s has a context cap below one sequence's window", lane->lane_name);
	state = (Ds41State *)calloc(1u, sizeof(*state));
	table = (const uint8_t **)calloc((size_t)DS41_STAGES * DS41_EXPERTS * DS41_EXPERT_PARTS, sizeof(*table));
	if ( state == 0 || table == 0 )
	{
		free(state);
		free(table);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
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
		free(table);
		free(state);
		return status;
	}
	status = Ds41Weights(state, lane->pack_path, table);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Scratch(state, lane->context_bytes);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Tables(state, table);
	free(table);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s is not a DeepSeek V4.1 Flash DSpark pack this module reads", lane->lane_name,
			lane->pack_path);
		(void)Ds41Close(state);
		return status;
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s with %u sequence windows", lane->lane_name, lane->pack_path, state->capacity);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus Ds41Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	Ds41State *state = (Ds41State *)opaque;
	Ds41Sequence *sequence = Ds41Live(state, request->sequence_id);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > DS41_WINDOW ? request->position - DS41_WINDOW : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( sequence == 0 || (request->position != 0u && Ds41Held(sequence, request->position - 1u) == 0u) )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	status = Ds41FillFromDonors(state, sequence, request, first);
	if ( status != SPARK_STATUS_OK )
		return Ds41Settle(state, status);
	for ( uint64_t position = first; position < request->position; position++ )
		if ( Ds41Held(sequence, position) == 0u )
			checked = position + 1u;
	if ( checked != first )
		state->gaps++;
	if ( request->committed_token_count < request->position )
		return Ds41Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	for ( uint64_t position = checked; position < request->position; position++ )
		if ( sequence->tokens[position % DS41_WINDOW] != request->committed_token_ids[position] )
			return Ds41Settle(state, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
	drafts = drafts < DS41_BLOCK ? drafts : DS41_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		SparkDraftTreeBranch branches[DS41_BLOCK];
		uint32_t branch_count = 0u;
		status = Ds41Prepare(state, sequence, request->anchor_token_id, request->position, &request->sampling);
		if ( status == SPARK_STATUS_OK )
			status = Ds41RunForward(state, sequence->ring, DS41_STAGES, (const uint16_t **)0, (const float **)0);
		if ( status == SPARK_STATUS_OK )
			status = Ds41Heads(state, request->sampling.inverse_temperature > 0.0f ? 0u : 1u, drafts, tree);
		status = Ds41Settle(state, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( tree != 0u )
			for ( uint32_t step = 0u; step < drafts; step++ )
			{
				uint32_t matched = DS41_BRANCHES;
				for ( uint32_t b = 0u; b < DS41_BRANCHES; b++ )
					if ( state->forward.host_picks->branch_len[b] > 0u && state->forward.host_picks->branch_fork[b] == step )
						matched = b;
				branches[branch_count].fork_step = step;
				if ( matched < DS41_BRANCHES )
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

static const SparkDrafterModuleInterface ds41_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"dsv41-dspark",
	Ds41Open,
	Ds41Ingest,
	Ds41Propose,
	Ds41Close,
};

static SparkStatus Ds41Readback(void *host, const void *device, size_t bytes)
{
	return SparkGpuMemcpy(host, device, bytes, SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &ds41_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_dsv41_dspark_ring(void *opaque, uint64_t sequence_id, uint16_t *keys,
	int64_t *positions)
{
	Ds41State *state = (Ds41State *)opaque;
	const Ds41Sequence *sequence = state != 0 ? Ds41Live(state, sequence_id) : 0;
	if ( sequence == 0 || keys == 0 || positions == 0 )
		return SPARK_STATUS_NOT_FOUND;
	memcpy(positions, sequence->positions, sizeof(sequence->positions));
	return Ds41Readback(keys, sequence->ring, DS41_RING_BYTES);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_dsv41_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *hidden, float *logits, uint64_t *elapsed_ns)
{
	Ds41State *state = (Ds41State *)opaque;
	const Ds41Sequence *sequence = state != 0 ? Ds41Live(state, sequence_id) : 0;
	const uint64_t started = SparkMonotonicNs();
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		return SPARK_STATUS_NOT_FOUND;
	memset(&greedy, 0, sizeof(greedy));
	status = Ds41Prepare(state, sequence, anchor, position, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Ds41RunForward(state, sequence->ring, DS41_STAGES, (const uint16_t **)0, (const float **)0);
	status = Ds41Settle(state, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && hidden != 0 )
		status = Ds41Readback(hidden, state->forward.hidden, (size_t)DS41_BLOCK * DS41_HIDDEN * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && logits != 0 )
		status = Ds41Readback(logits, state->forward.logits, (size_t)DS41_BLOCK * DS41_VOCAB * sizeof(float));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_dsv41_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t stages, uint16_t *streams, float *pre)
{
	Ds41State *state = (Ds41State *)opaque;
	const Ds41Sequence *sequence = state != 0 ? Ds41Live(state, sequence_id) : 0;
	const uint16_t *streams_at = 0;
	const float *pre_at = 0;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 || streams == 0 || pre == 0 || stages > DS41_STAGES )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&greedy, 0, sizeof(greedy));
	status = Ds41Prepare(state, sequence, anchor, position, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Ds41RunForward(state, sequence->ring, stages, &streams_at, &pre_at);
	status = Ds41Settle(state, status);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Readback(streams, streams_at, (size_t)DS41_BLOCK * DS41_HC_FLAT * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = Ds41Readback(pre, pre_at, (size_t)DS41_BLOCK * DS41_HC * sizeof(float));
	return status;
}
