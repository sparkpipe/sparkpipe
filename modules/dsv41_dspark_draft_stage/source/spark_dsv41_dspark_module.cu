#include "inference/kernels/dspark_draft_stage_common.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/topk_exact.cuh"

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
#define DS41_TOPK_THREADS 512u
#define DS41_TOPK_CHUNK 2048u
#define DS41_EXPERT_NEURONS 64u
#define DS41_DOWN_SHARED_BYTES (DS41_BLOCK * DS41_MOE * sizeof(uint16_t))
#define DS41_SEQUENCES_MAX 1024u
#define DS41_RING_ELEMENTS ((uint64_t)DS41_STAGES * DS41_WINDOW * DS41_HEAD_DIM)
#define DS41_RING_BYTES (DS41_RING_ELEMENTS * sizeof(uint16_t))
#define DS41_STAGE_RING (DS41_WINDOW * DS41_HEAD_DIM)
#define DS41_EXPERT_PARTS 6u
#define DS41_W1_PART 0u
#define DS41_W2_PART 2u
#define DS41_W3_PART 4u
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

using Ds41Sequence = DsparkRingSequence<DS41_WINDOW>;
using Ds41RingPool = DsparkRingPool<DS41_WINDOW, DS41_SEQUENCES_MAX>;

typedef struct Ds41Draft
{
	const uint16_t *ring;
	uint64_t seed;
	uint32_t positions[DS41_BLOCK];
	uint32_t anchor;
	uint32_t noise;
	uint32_t greedy;
	uint32_t tree;
	uint32_t drafts;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint8_t valid[DS41_WINDOW];
} Ds41Draft;

typedef struct Ds41RoutePlan
{
	uint32_t expert[DS41_PAIRS];
	uint32_t count[DS41_PAIRS];
	uint32_t pairs[DS41_PAIRS][DS41_BLOCK];
} Ds41RoutePlan;

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
	Ds41RoutePlan *route_plan;
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
	SparkDrafterGraphCache graphs;
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
	Ds41RingPool ring;
	uint64_t frames;
	uint64_t rows;
} Ds41State;

static __device__ __forceinline__ uint16_t Ds41QdqLane(float value)
{
	const float amax = DsparkWarpMax(fabsf(value));
	return LmFloatToBf16(SparkMathFp8Qdq(value, SparkMathFp8QdqScale(amax)));
}

static __device__ __forceinline__ void Ds41SoftmaxMerge(float *top, float *sum, float other_top, float other_sum)
{
	if ( other_top == -INFINITY )
		return;
	if ( *top == -INFINITY )
	{
		*top = other_top;
		*sum = other_sum;
		return;
	}
	if ( other_top <= *top )
		*sum += other_sum * expf(other_top - *top);
	else
	{
		*sum = *sum * expf(*top - other_top) + other_sum;
		*top = other_top;
	}
}

static __device__ __forceinline__ float Ds41KeyValue(unsigned long long key)
{
	const uint32_t ordered = (uint32_t)(key >> 32u);
	return __uint_as_float(ordered ^ ((ordered & 0x80000000u) != 0u ? 0x80000000u : 0xffffffffu));
}

static __device__ __forceinline__ uint32_t Ds41Previous(const Ds41Draft *draft, const Ds41Picks *picks, uint32_t step)
{
	if ( step == 0u )
		return draft->anchor;
	return draft->greedy != 0u ? DsparkKeySlot(picks->keys[step - 1u]) : picks->tokens[step - 1u];
}

__global__ __launch_bounds__(DSPARK_THREADS, 1)
void Ds41ExpandScalesKernel(const uint8_t *__restrict__ block, uint8_t *__restrict__ rows_scale, uint32_t rows, uint32_t groups)
{
	const uint64_t total = (uint64_t)rows * groups;
	for ( uint64_t index = (uint64_t)blockIdx.x * DSPARK_THREADS + threadIdx.x; index < total; index += (uint64_t)gridDim.x * DSPARK_THREADS )
	{
		const uint64_t row = index / groups, group = index % groups;
		rows_scale[index] = block[(row / DS41_QDQ) * groups + group];
	}
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41QdqKernel(const uint16_t *__restrict__ input, uint16_t *__restrict__ output, uint32_t rows, uint32_t parts, uint32_t part_width,
	uint32_t to_part_major)
{
	LmDependentWait();
	const uint64_t groups = (uint64_t)rows * parts * part_width / DS41_QDQ;
	const uint32_t lane = threadIdx.x % DSPARK_WARP;
	for ( uint64_t group = (uint64_t)blockIdx.x * DSPARK_WARPS + threadIdx.x / DSPARK_WARP; group < groups; group += (uint64_t)gridDim.x * DSPARK_WARPS )
	{
		const uint64_t element = group * DS41_QDQ + lane;
		const uint64_t row = element / ((uint64_t)parts * part_width), part = (element / part_width) % parts, column = element % part_width;
		const uint64_t row_major = (row * parts + part) * part_width + column, part_major = (part * rows + row) * part_width + column;
		const uint64_t source = to_part_major != 0u ? row_major : part_major, target = to_part_major != 0u ? part_major : row_major;
		output[target] = Ds41QdqLane(LmBf16ToFloat(__ldg(input + source)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41RmsQdqKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ output_qdq, uint32_t dimension,
	float epsilon)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	const uint16_t *row = input + (uint64_t)blockIdx.x * dimension;
	uint16_t *out = output_qdq + (uint64_t)blockIdx.x * dimension;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < dimension; index += DSPARK_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(row + index));
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += DSPARK_THREADS )
	{
		const float normed = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(__ldg(row + index)) * scale * LmBf16ToFloat(__ldg(weight + index))));
		out[index] = Ds41QdqLane(normed);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41CollapseRmsQdqKernel(const uint16_t *__restrict__ streams, const float *__restrict__ pre_rows, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ out_x, uint16_t *__restrict__ out_xq, float epsilon)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	__shared__ float pre[DS41_HC];
	const uint32_t row = blockIdx.x;
	const uint16_t *src = streams + (uint64_t)row * DS41_HC_FLAT;
	if ( threadIdx.x < DS41_HC )
		pre[threadIdx.x] = pre_rows[(uint64_t)row * DS41_HC + threadIdx.x];
	__syncthreads();
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < DS41_HIDDEN; index += DSPARK_THREADS )
	{
		float collapsed = 0.0f;
#pragma unroll
		for ( uint32_t s = 0u; s < DS41_HC; s++ )
			collapsed = fmaf(pre[s], LmBf16ToFloat(__ldg(src + (uint64_t)s * DS41_HIDDEN + index)), collapsed);
		const float cbf = LmBf16ToFloat(LmFloatToBf16(collapsed));
		total += cbf * cbf;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)DS41_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < DS41_HIDDEN; index += DSPARK_THREADS )
	{
		float collapsed = 0.0f;
#pragma unroll
		for ( uint32_t s = 0u; s < DS41_HC; s++ )
			collapsed = fmaf(pre[s], LmBf16ToFloat(__ldg(src + (uint64_t)s * DS41_HIDDEN + index)), collapsed);
		const float cbf = LmBf16ToFloat(LmFloatToBf16(collapsed));
		const uint16_t normed_bf = LmFloatToBf16(cbf * scale * LmBf16ToFloat(__ldg(weight + index)));
		if ( out_x != 0 )
			out_x[(uint64_t)row * DS41_HIDDEN + index] = normed_bf;
		if ( out_xq != 0 )
			out_xq[(uint64_t)row * DS41_HIDDEN + index] = Ds41QdqLane(LmBf16ToFloat(normed_bf));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41CollapseRmsKernel(const uint16_t *__restrict__ streams, const float *__restrict__ pre_rows, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ out_hidden, uint16_t *__restrict__ out_normed, float epsilon)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	__shared__ float pre[DS41_HC];
	const uint32_t row = blockIdx.x;
	const uint16_t *src = streams + (uint64_t)row * DS41_HC_FLAT;
	if ( threadIdx.x < DS41_HC )
		pre[threadIdx.x] = pre_rows[(uint64_t)row * DS41_HC + threadIdx.x];
	__syncthreads();
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < DS41_HIDDEN; index += DSPARK_THREADS )
	{
		float collapsed = 0.0f;
#pragma unroll
		for ( uint32_t s = 0u; s < DS41_HC; s++ )
			collapsed = fmaf(pre[s], LmBf16ToFloat(__ldg(src + (uint64_t)s * DS41_HIDDEN + index)), collapsed);
		const uint16_t cbf_u16 = LmFloatToBf16(collapsed);
		out_hidden[(uint64_t)row * DS41_HIDDEN + index] = cbf_u16;
		const float cbf = LmBf16ToFloat(cbf_u16);
		total += cbf * cbf;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)DS41_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < DS41_HIDDEN; index += DSPARK_THREADS )
	{
		const float cbf = LmBf16ToFloat(out_hidden[(uint64_t)row * DS41_HIDDEN + index]);
		out_normed[(uint64_t)row * DS41_HIDDEN + index] = LmFloatToBf16(cbf * scale * LmBf16ToFloat(__ldg(weight + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41KeyRowsKernel(const uint16_t *__restrict__ raw, const uint16_t *__restrict__ norm, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, float epsilon, uint16_t *__restrict__ ring, uint16_t *__restrict__ rows_out)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	__shared__ float key[DS41_HEAD_DIM];
	const uint16_t *row = raw + (uint64_t)blockIdx.x * DS41_HEAD_DIM;
	const uint32_t position = positions[blockIdx.x];
	uint16_t *out = ring != 0 ? ring + (uint64_t)(position % DS41_WINDOW) * DS41_HEAD_DIM : rows_out + (uint64_t)blockIdx.x * DS41_HEAD_DIM;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DSPARK_THREADS )
	{
		const float value = LmBf16ToFloat(row[index]);
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)DS41_HEAD_DIM + epsilon);
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DSPARK_THREADS )
		key[index] = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(row[index]) * scale * LmBf16ToFloat(norm[index])));
	__syncthreads();
	if ( threadIdx.x < DS41_ROPE / 2u )
	{
		const uint32_t pair = threadIdx.x, at = DS41_NOPE + 2u * pair;
		const float even = key[at], odd = key[at + 1u];
		float cosine, sine;
		DsparkRotationUnit(position, inverse[pair], &cosine, &sine);
		key[at] = LmBf16ToFloat(LmFloatToBf16(even * cosine - odd * sine));
		key[at + 1u] = LmBf16ToFloat(LmFloatToBf16(even * sine + odd * cosine));
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DSPARK_THREADS )
		out[index] = Ds41QdqLane(key[index]);
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41RopeKernel(uint16_t *__restrict__ data, const Ds41Draft *__restrict__ draft, const float *__restrict__ inverse, uint32_t heads,
	float direction)
{
	LmDependentWait();
	const uint32_t pair = threadIdx.x % (DS41_ROPE / 2u), item = blockIdx.x * (DSPARK_THREADS / (DS41_ROPE / 2u)) + threadIdx.x / (DS41_ROPE / 2u);
	if ( item < DS41_BLOCK * heads )
	{
		uint16_t *row = data + (uint64_t)item * DS41_HEAD_DIM + DS41_NOPE + 2u * pair;
		const float even = LmBf16ToFloat(row[0]), odd = LmBf16ToFloat(row[1]);
		float cosine, sine;
		DsparkRotationUnit(draft->positions[item / heads], inverse[pair], &cosine, &sine);
		sine *= direction;
		row[0] = LmFloatToBf16(even * cosine - odd * sine);
		row[1] = LmFloatToBf16(even * sine + odd * cosine);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41EmbedKernel(const uint16_t *__restrict__ embed, const Ds41Draft *__restrict__ draft, uint16_t *__restrict__ streams)
{
	LmDependentWait();
	const uint32_t row = blockIdx.x, token = row == 0u ? draft->anchor : draft->noise;
	const uint4 *src = (const uint4 *)(embed + (uint64_t)token * DS41_HIDDEN);
	uint4 *dst = (uint4 *)(streams + (uint64_t)row * DS41_HC_FLAT);
	constexpr uint32_t VECS = DS41_HIDDEN / 8u;
	for ( uint32_t index = threadIdx.x; index < VECS; index += DSPARK_THREADS )
	{
		const uint4 v = __ldg(src + index);
#pragma unroll
		for ( uint32_t s = 0u; s < DS41_HC; s++ )
			dst[(uint64_t)s * VECS + index] = v;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41HcPostKernel(const uint16_t *__restrict__ output, const uint16_t *__restrict__ residual, const float *__restrict__ post_rows,
	const float *__restrict__ comb_rows, uint16_t *__restrict__ streams)
{
	LmDependentWait();
	__shared__ float post[DS41_HC];
	__shared__ float comb[DS41_HC * DS41_HC];
	const uint32_t row = blockIdx.y;
	if ( threadIdx.x < DS41_HC )
		post[threadIdx.x] = post_rows[(uint64_t)row * DS41_HC + threadIdx.x];
	if ( threadIdx.x < DS41_HC * DS41_HC )
		comb[threadIdx.x] = comb_rows[(uint64_t)row * DS41_HC * DS41_HC + threadIdx.x];
	__syncthreads();
	for ( uint32_t element = blockIdx.x * DSPARK_THREADS + threadIdx.x; element < DS41_HIDDEN; element += gridDim.x * DSPARK_THREADS )
	{
		const float y = LmBf16ToFloat(__ldg(output + (uint64_t)row * DS41_HIDDEN + element));
		float source[DS41_HC];
#pragma unroll
		for ( uint32_t stream = 0u; stream < DS41_HC; stream++ )
			source[stream] = LmBf16ToFloat(__ldg(residual + ((uint64_t)row * DS41_HC + stream) * DS41_HIDDEN + element));
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
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41WoAKernel(const uint8_t *__restrict__ weight, const uint8_t *__restrict__ rows_scale, const uint16_t *__restrict__ grouped,
	uint16_t *__restrict__ projected)
{
	LmDependentWait();
	constexpr uint32_t CHUNKS = DS41_GROUP_WIDTH / 16u;
	constexpr uint32_t GROUPS = DS41_GROUP_WIDTH / DS41_QDQ;
	constexpr uint32_t STEPS = CHUNKS / DSPARK_WARP;
	const uint32_t group = blockIdx.y, warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint32_t local_row = blockIdx.x * DSPARK_WARPS + warp;
	const uint64_t global_row = (uint64_t)group * DS41_O_RANK + local_row;
	const uint8_t *w_row = weight + global_row * DS41_GROUP_WIDTH;
	const uint8_t *s_row = rows_scale + global_row * GROUPS;
	const uint16_t *in_group = grouped + (uint64_t)group * DS41_BLOCK * DS41_GROUP_WIDTH;
	float dot[DS41_BLOCK];
#pragma unroll
	for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
		dot[row] = 0.0f;
	uint4 next_w = __ldg((const uint4 *)(w_row + (uint64_t)lane * 16u));
	uint8_t next_s = __ldg(s_row + (lane >> 1u));
#pragma unroll 1
	for ( uint32_t step = 0u; step < STEPS; step++ )
	{
		const uint32_t chunk = lane + step * DSPARK_WARP, following = chunk + DSPARK_WARP;
		const uint4 packed = next_w;
		const uint8_t sc_code = next_s;
		if ( step + 1u < STEPS )
		{
			next_w = __ldg((const uint4 *)(w_row + (uint64_t)following * 16u));
			next_s = __ldg(s_row + (following >> 1u));
		}
		const float factor = LmUe8m0ToFloat(sc_code);
		const LmSkinnyFormat<LmMxfp8>::Decoded decoded = LmSkinnyFormat<LmMxfp8>::Decode(packed);
#pragma unroll
		for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
		{
			const LmSkinnyFormat<LmMxfp8>::Activation act = LmSkinnyFormat<LmMxfp8>::Load(in_group + (uint64_t)row * DS41_GROUP_WIDTH + (uint64_t)chunk * 16u);
			dot[row] += LmSkinnyFormat<LmMxfp8>::ChunkDecoded(decoded, act) * factor;
		}
	}
#pragma unroll
	for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
	{
		const float total = DsparkWarpSum(dot[row]);
		if ( lane == 0u )
			projected[((uint64_t)group * DS41_BLOCK + row) * DS41_O_RANK + local_row] = LmFloatToBf16(total);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41AttentionKernel(const uint16_t *__restrict__ q, uint32_t stage, const uint16_t *__restrict__ block_keys,
	const Ds41Draft *__restrict__ draft, const float *__restrict__ sink, float scale, uint16_t *__restrict__ output)
{
	LmDependentWait();
	constexpr uint32_t KEYS = DS41_WINDOW + DS41_BLOCK;
	constexpr uint32_t PER = DS41_HEAD_DIM / DSPARK_WARP;
	__shared__ float query[DS41_HEAD_DIM];
	__shared__ float scores[KEYS];
	__shared__ float reduction[DSPARK_WARPS];
	const uint16_t *ring = draft->ring + (uint64_t)stage * DS41_STAGE_RING;
	const uint32_t head = blockIdx.x, row = blockIdx.y, warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint16_t *own = q + ((uint64_t)row * DS41_HEADS + head) * DS41_HEAD_DIM;
	float top = -INFINITY, total = 0.0f, low = 0.0f, high = 0.0f;
	for ( uint32_t index = threadIdx.x; index < DS41_HEAD_DIM; index += DSPARK_THREADS )
		query[index] = LmBf16ToFloat(__ldg(own + index));
	__syncthreads();
	for ( uint32_t key = warp; key < KEYS; key += DSPARK_WARPS )
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
			dot += query[part * DSPARK_WARP + lane] * LmBf16ToFloat(__ldg(source + part * DSPARK_WARP + lane));
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
			scores[key] = dot * scale;
	}
	__syncthreads();
	for ( uint32_t key = threadIdx.x; key < KEYS; key += DSPARK_THREADS )
		top = fmaxf(top, scores[key]);
	top = DsparkWarpMax(top);
	if ( lane == 0u )
		reduction[warp] = top;
	__syncthreads();
	top = -INFINITY;
	for ( uint32_t index = 0u; index < DSPARK_WARPS; index++ )
		top = fmaxf(top, reduction[index]);
	top = fmaxf(top, -1e30f);
	__syncthreads();
	for ( uint32_t key = threadIdx.x; key < KEYS; key += DSPARK_THREADS )
	{
		const float weight = expf(scores[key] - top);
		scores[key] = LmBf16ToFloat(LmFloatToBf16(weight));
		total += weight;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	total += expf(sink[head] - top);
	for ( uint32_t key = 0u; key < KEYS; key++ )
	{
		const float weight = scores[key];
		const uint16_t *source = key < DS41_WINDOW ? ring + (uint64_t)key * DS41_HEAD_DIM : block_keys + (uint64_t)(key - DS41_WINDOW) * DS41_HEAD_DIM;
		if ( weight == 0.0f )
			continue;
		low += weight * LmBf16ToFloat(__ldg(source + threadIdx.x));
		high += weight * LmBf16ToFloat(__ldg(source + threadIdx.x + DSPARK_THREADS));
	}
	output[((uint64_t)row * DS41_HEADS + head) * DS41_HEAD_DIM + threadIdx.x] = LmFloatToBf16(low / total);
	output[((uint64_t)row * DS41_HEADS + head) * DS41_HEAD_DIM + threadIdx.x + DSPARK_THREADS] = LmFloatToBf16(high / total);
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41GateScoreKernel(const uint16_t *__restrict__ x, const uint16_t *__restrict__ gate, float *__restrict__ scores)
{
	LmDependentWait();
	constexpr uint32_t STEPS = DS41_HIDDEN / (DSPARK_WARP * 8u);
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP, expert = blockIdx.x * DSPARK_WARPS + warp;
	const uint16_t *weights = gate + (uint64_t)expert * DS41_HIDDEN;
	float dot[DS41_BLOCK];
#pragma unroll
	for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
		dot[row] = 0.0f;
	for ( uint32_t step = 0u; step < STEPS; step++ )
	{
		const uint32_t at = (step * DSPARK_WARP + lane) * 8u;
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
		const float total = DsparkWarpSum(dot[row]);
		if ( lane == 0u )
			scores[row * DS41_EXPERTS + expert] = sqrtf(total > DS41_SOFTPLUS_THRESHOLD ? total : log1pf(expf(total)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DS41_BLOCK * DSPARK_WARP, 1)
void Ds41GateTopKernel(const float *__restrict__ scores_rows, const float *__restrict__ bias, float route_scale, uint32_t *__restrict__ route_expert,
	float *__restrict__ route_weight, Ds41RoutePlan *__restrict__ plan)
{
	LmDependentWait();
	__shared__ uint32_t sh_expert[DS41_PAIRS];
	const uint32_t row = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const float *scores = scores_rows + (uint64_t)row * DS41_EXPERTS;
	uint32_t first = 0u, second = 0u, third = 0u;
#pragma unroll
	for ( uint32_t slot = 0u; slot < DS41_ACTIVE; slot++ )
	{
		unsigned long long key = 0ull;
		for ( uint32_t expert = lane; expert < DS41_EXPERTS; expert += DSPARK_WARP )
			if ( !(slot > 0u && expert == first) && !(slot > 1u && expert == second) )
			{
				const unsigned long long candidate = ((unsigned long long)SparkMathSampleOrdered(scores[expert] + bias[expert]) << 32u) |
					(unsigned long long)(0xffffffffu - expert);
				key = candidate > key ? candidate : key;
			}
		key = DsparkWarpKey(key);
		if ( slot == 0u )
			first = DsparkKeySlot(key);
		else if ( slot == 1u )
			second = DsparkKeySlot(key);
		else
			third = DsparkKeySlot(key);
	}
	if ( lane < DS41_ACTIVE )
	{
		const float a = scores[first], b = scores[second], c = scores[third];
		const uint32_t expert = lane == 0u ? first : lane == 1u ? second : third;
		const float picked = lane == 0u ? a : lane == 1u ? b : c;
		route_expert[row * DS41_ACTIVE + lane] = expert;
		sh_expert[row * DS41_ACTIVE + lane] = expert;
		route_weight[row * DS41_ACTIVE + lane] = picked / (a + b + c + DS41_ROUTE_EPSILON) * route_scale;
	}
	__syncthreads();
	if ( threadIdx.x < DS41_PAIRS )
	{
		const uint32_t unique_index = threadIdx.x;
		uint32_t unique = 0u, count = 0u, expert = 0xffffffffu;
		for ( uint32_t pair = 0u; pair < DS41_PAIRS; pair++ )
		{
			uint32_t seen = 0u;
			for ( uint32_t earlier = 0u; earlier < pair; earlier++ )
				seen |= sh_expert[earlier] == sh_expert[pair] ? 1u : 0u;
			if ( seen == 0u && unique++ == unique_index )
				expert = sh_expert[pair];
		}
		for ( uint32_t pair = 0u; pair < DS41_PAIRS && expert != 0xffffffffu; pair++ )
			if ( sh_expert[pair] == expert )
				plan->pairs[unique_index][count++] = pair;
		plan->expert[unique_index] = expert;
		plan->count[unique_index] = count;
	}
	LmDependentRelease();
}

template<uint32_t INPUT, uint32_t OUTPUTS, uint32_t ROW_DIVISOR, bool USE_SMEM>
static __device__ __forceinline__ void Ds41ExpertRows(const uint16_t *__restrict__ input, const uint8_t *const *__restrict__ table,
	const Ds41RoutePlan *__restrict__ plan, uint32_t first_part, uint32_t second_part, uint16_t *first_out, uint16_t *second_out)
{
	LmDependentWait();
	constexpr uint32_t CHUNKS = INPUT / DS41_QDQ;
	constexpr uint32_t PER_LANE = (CHUNKS + DSPARK_WARP - 1u) / DSPARK_WARP;
	extern __shared__ __align__(16) uint16_t ds41_expert_rows[];
	__shared__ uint32_t s_pairs[DS41_BLOCK];
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint32_t count = plan->count[blockIdx.x];
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t expert = plan->expert[blockIdx.x];
	if ( threadIdx.x < count )
		s_pairs[threadIdx.x] = plan->pairs[blockIdx.x][threadIdx.x];
	__syncthreads();
	if constexpr ( USE_SMEM )
	{
		for ( uint32_t index = threadIdx.x; index < count * (INPUT / 8u); index += DSPARK_THREADS )
		{
			const uint32_t row = index / (INPUT / 8u), column = index % (INPUT / 8u);
			((uint4 *)ds41_expert_rows)[index] = __ldg((const uint4 *)(input + (uint64_t)(s_pairs[row] / ROW_DIVISOR) * INPUT) + column);
		}
		__syncthreads();
	}
#pragma unroll 1
	for ( uint32_t item = warp; item < OUTPUTS * DS41_EXPERT_NEURONS; item += DSPARK_WARPS )
	{
		const uint32_t which = item / DS41_EXPERT_NEURONS, neuron = blockIdx.y * DS41_EXPERT_NEURONS + item % DS41_EXPERT_NEURONS;
		const uint8_t *const *parts = table + (uint64_t)expert * DS41_EXPERT_PARTS + (which == 0u ? first_part : second_part);
		uint16_t *output = which == 0u ? first_out : second_out;
		const uint8_t *weight = parts[1] + (uint64_t)neuron * (INPUT / 2u), *scale = parts[0] + (uint64_t)neuron * CHUNKS;
		float dot[DS41_BLOCK];
		uint4 next = lane < CHUNKS ? __ldg((const uint4 *)(weight + (uint64_t)lane * (DS41_QDQ / 2u))) : make_uint4(0u, 0u, 0u, 0u);
		uint8_t next_scale = lane < CHUNKS ? __ldg(scale + lane) : 0u;
#pragma unroll
		for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
			dot[row] = 0.0f;
#pragma unroll 1
		for ( uint32_t step = 0u; step < PER_LANE; step++ )
		{
			const uint32_t chunk = lane + step * DSPARK_WARP, following = chunk + DSPARK_WARP;
			const uint4 packed = next;
			const uint8_t sc_code = next_scale;
			if ( chunk >= CHUNKS )
				break;
			if ( following < CHUNKS )
			{
				next = __ldg((const uint4 *)(weight + (uint64_t)following * (DS41_QDQ / 2u)));
				next_scale = __ldg(scale + following);
			}
			const float factor = LmUe8m0ToFloat(sc_code);
			const LmSkinnyE2m1HalfDecoded d0 = LmSkinnyDecodeE2m1Half(packed.x, packed.y);
			const LmSkinnyE2m1HalfDecoded d1 = LmSkinnyDecodeE2m1Half(packed.z, packed.w);
#pragma unroll
			for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
				if ( row < count )
				{
					const uint16_t *src_row = USE_SMEM ? (ds41_expert_rows + (uint64_t)row * INPUT)
						: (input + (uint64_t)(s_pairs[row] / ROW_DIVISOR) * INPUT);
					const uint16_t *at = src_row + chunk * DS41_QDQ;
					const uint4 a0 = USE_SMEM ? *((const uint4 *)at) : __ldg((const uint4 *)at);
					const uint4 a1 = USE_SMEM ? *((const uint4 *)at + 1) : __ldg((const uint4 *)at + 1);
					const uint4 a2 = USE_SMEM ? *((const uint4 *)at + 2) : __ldg((const uint4 *)at + 2);
					const uint4 a3 = USE_SMEM ? *((const uint4 *)at + 3) : __ldg((const uint4 *)at + 3);
					const uint32_t v0[4] = {a0.x, a0.y, a0.z, a0.w}, v1[4] = {a1.x, a1.y, a1.z, a1.w};
					const uint32_t v2[4] = {a2.x, a2.y, a2.z, a2.w}, v3[4] = {a3.x, a3.y, a3.z, a3.w};
					float part = 0.0f;
#pragma unroll
					for ( uint32_t b = 0u; b < 4u; b++ )
						part += d0.pair[b].x * __uint_as_float(v0[b] << 16u) + d0.pair[b].y * __uint_as_float(v0[b] & 0xffff0000u);
#pragma unroll
					for ( uint32_t b = 0u; b < 4u; b++ )
						part += d0.pair[4u + b].x * __uint_as_float(v1[b] << 16u) + d0.pair[4u + b].y * __uint_as_float(v1[b] & 0xffff0000u);
#pragma unroll
					for ( uint32_t b = 0u; b < 4u; b++ )
						part += d1.pair[b].x * __uint_as_float(v2[b] << 16u) + d1.pair[b].y * __uint_as_float(v2[b] & 0xffff0000u);
#pragma unroll
					for ( uint32_t b = 0u; b < 4u; b++ )
						part += d1.pair[4u + b].x * __uint_as_float(v3[b] << 16u) + d1.pair[4u + b].y * __uint_as_float(v3[b] & 0xffff0000u);
					dot[row] += part * factor;
				}
		}
#pragma unroll
		for ( uint32_t row = 0u; row < DS41_BLOCK; row++ )
			if ( row < count )
			{
				const float total = DsparkWarpSum(dot[row]);
				if ( lane == 0u )
					output[(uint64_t)s_pairs[row] * (OUTPUTS == 2u ? DS41_MOE : DS41_HIDDEN) + neuron] = LmFloatToBf16(total);
			}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41ExpertUpKernel(const uint16_t *__restrict__ xq, const uint8_t *const *__restrict__ table, const Ds41RoutePlan *__restrict__ plan,
	uint16_t *__restrict__ gate_out, uint16_t *__restrict__ up_out)
{
	Ds41ExpertRows<DS41_HIDDEN, 2u, DS41_ACTIVE, false>(xq, table, plan, DS41_W1_PART, DS41_W3_PART, gate_out, up_out);
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41ExpertDownKernel(const uint16_t *__restrict__ hidden, const uint8_t *const *__restrict__ table, const Ds41RoutePlan *__restrict__ plan,
	uint16_t *__restrict__ down_out)
{
	Ds41ExpertRows<DS41_MOE, 1u, 1u, true>(hidden, table, plan, DS41_W2_PART, DS41_W2_PART, down_out, down_out);
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41SwigluKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ route_weight, float limit,
	uint32_t rows, uint16_t *__restrict__ output)
{
	LmDependentWait();
	const uint64_t groups = (uint64_t)rows * DS41_MOE / DS41_QDQ;
	const uint32_t lane = threadIdx.x % DSPARK_WARP;
	for ( uint64_t group = (uint64_t)blockIdx.x * DSPARK_WARPS + threadIdx.x / DSPARK_WARP; group < groups; group += (uint64_t)gridDim.x * DSPARK_WARPS )
	{
		const uint64_t element = group * DS41_QDQ + lane, row = element / DS41_MOE;
		const float g = fminf(LmBf16ToFloat(__ldg(gate + element)), limit);
		const float u = fminf(fmaxf(LmBf16ToFloat(__ldg(up + element)), -limit), limit);
		float value = g / (1.0f + expf(-g)) * u;
		if ( route_weight != 0 )
			value = route_weight[row] * value;
		output[element] = Ds41QdqLane(LmBf16ToFloat(LmFloatToBf16(value)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41MoeSumKernel(const uint16_t *__restrict__ pair_down, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * DSPARK_THREADS + threadIdx.x; element < DS41_HIDDEN; element += gridDim.x * DSPARK_THREADS )
	{
		float total = 0.0f;
#pragma unroll
		for ( uint32_t slot = 0u; slot < DS41_ACTIVE; slot++ )
			total += LmBf16ToFloat(__ldg(pair_down + ((uint64_t)row * DS41_ACTIVE + slot) * DS41_HIDDEN + element));
		total += LmBf16ToFloat(__ldg(shared + (uint64_t)row * DS41_HIDDEN + element));
		output[(uint64_t)row * DS41_HIDDEN + element] = LmFloatToBf16(total);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41MarkovRowKernel(uint32_t step, const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks, const float *__restrict__ logits,
	const uint16_t *__restrict__ markov_embed, const uint16_t *__restrict__ markov_head, float *__restrict__ row, float *__restrict__ partials)
{
	LmDependentWait();
	constexpr uint32_t PER = DS41_MARKOV / DSPARK_WARP;
	__shared__ float embedding[DS41_MARKOV];
	__shared__ unsigned long long best[DSPARK_WARPS];
	__shared__ unsigned long long second_best[DSPARK_WARPS];
	__shared__ float tops[DSPARK_WARPS];
	__shared__ float sums[DSPARK_WARPS];
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP, previous = Ds41Previous(draft, picks, step);
	const uint32_t greedy = draft->greedy, need_softmax = greedy == 0u || draft->tree != 0u ? 1u : 0u;
	const float scale = greedy != 0u ? 1.0f : draft->inverse_temperature;
	unsigned long long first = 0ull, second = 0ull;
	float top = -INFINITY, sum = 0.0f;
	for ( uint32_t index = threadIdx.x; index < DS41_MARKOV; index += DSPARK_THREADS )
		embedding[index] = LmBf16ToFloat(__ldg(markov_embed + (uint64_t)previous * DS41_MARKOV + index));
	__syncthreads();
	for ( uint32_t token = blockIdx.x * DSPARK_WARPS + warp; token < DS41_VOCAB; token += gridDim.x * DSPARK_WARPS )
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
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
		{
			value = __ldg(logits + (uint64_t)step * DS41_VOCAB + token) + dot;
			if ( greedy == 0u )
				row[token] = value;
			else
			{
				const unsigned long long candidate = ((unsigned long long)SparkMathSampleOrdered(value) << 32u) | (unsigned long long)(0xffffffffu - token);
				if ( need_softmax != 0u )
					DsparkTop2Insert(candidate, &first, &second);
				else if ( candidate > first )
					first = candidate;
			}
			if ( need_softmax != 0u )
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
		for ( uint32_t index = 1u; index < DSPARK_WARPS; index++ )
		{
			if ( need_softmax != 0u )
				Ds41SoftmaxMerge(&top, &sum, tops[index], sums[index]);
			if ( greedy != 0u )
			{
				if ( need_softmax != 0u )
				{
					DsparkTop2Insert(best[index], &first, &second);
					DsparkTop2Insert(second_best[index], &first, &second);
				}
				else if ( best[index] > first )
					first = best[index];
			}
		}
		if ( need_softmax != 0u )
		{
			partials[((uint64_t)step * gridDim.x + blockIdx.x) * 2u] = top;
			partials[((uint64_t)step * gridDim.x + blockIdx.x) * 2u + 1u] = sum;
		}
		if ( greedy != 0u )
		{
			const unsigned long long prev = atomicMax(&picks->keys[step], first);
			if ( need_softmax != 0u )
			{
				const unsigned long long loser = prev > first ? first : prev;
				const unsigned long long runner = loser > second ? loser : second;
				if ( runner != 0ull )
					atomicMax(&picks->alt_keys[step], runner);
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_PICK_THREADS, 1)
void Ds41SampleKernel(uint32_t step, const Ds41Draft *__restrict__ draft, const float *__restrict__ row, const uint32_t *__restrict__ candidates,
	const float *__restrict__ partials, uint32_t blocks, Ds41Picks *__restrict__ picks)
{
	LmDependentWait();
	__shared__ float values[DS41_SAMPLED];
	__shared__ uint32_t tokens[DS41_SAMPLED];
	__shared__ float sorted_values[DS41_SAMPLED];
	__shared__ uint32_t sorted_tokens[DS41_SAMPLED];
	__shared__ float reduction[DSPARK_PICK_THREADS / DSPARK_WARP];
	__shared__ float before[DS41_SAMPLED];
	__shared__ unsigned long long best[DS41_SAMPLED / DSPARK_WARP];
	const float inverse_temperature = draft->inverse_temperature, nucleus = draft->nucleus;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP, position = draft->positions[0] + step;
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
	for ( uint32_t block = threadIdx.x; block < blocks; block += DSPARK_PICK_THREADS )
		if ( partials[2u * block] != -INFINITY )
			sum += partials[2u * block + 1u] * expf(partials[2u * block] - top);
	sum = LmBlockSum<DSPARK_PICK_THREADS>(sum, reduction);
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
	if ( warp < DS41_SAMPLED / DSPARK_WARP )
	{
		const uint32_t rank = warp * DSPARK_WARP + lane;
		const uint32_t keep = (((nucleus <= 0.0f) || (before[rank] < nucleus)) && (draft->top_k == 0u || rank < draft->top_k)) || rank == 0u ? 1u : 0u;
		const float score = keep != 0u ? sorted_values[rank] + SparkMathGumbelNoise(draft->seed, position, sorted_tokens[rank]) : -INFINITY;
		unsigned long long key = DsparkWarpKey(((unsigned long long)SparkMathSampleOrdered(score) << 32u) | (unsigned long long)(0xffffffffu - rank));
		if ( lane == 0u )
			best[warp] = key;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		unsigned long long key = best[0];
		uint32_t winner, runner;
		for ( uint32_t index = 1u; index < DS41_SAMPLED / DSPARK_WARP; index++ )
			key = best[index] > key ? best[index] : key;
		winner = DsparkKeySlot(key);
		runner = winner == 0u ? 1u : 0u;
		picks->tokens[step] = sorted_tokens[winner];
		picks->alt_tokens[step] = sorted_tokens[runner];
		picks->alt_scores[step] = sum > 0.0f ? expf(sorted_values[runner] - top) / sum : 0.0f;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41ConfidenceKernel(const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks, const uint16_t *__restrict__ hidden,
	const uint16_t *__restrict__ markov_embed, const uint16_t *__restrict__ weight, const float *__restrict__ partials, uint32_t blocks)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	const uint32_t slot = blockIdx.x, previous = Ds41Previous(draft, picks, slot);
	float dot = 0.0f;
	for ( uint32_t index = threadIdx.x; index < DS41_CONFIDENCE; index += DSPARK_THREADS )
	{
		const float feature = index < DS41_HIDDEN ? LmBf16ToFloat(__ldg(hidden + (uint64_t)slot * DS41_HIDDEN + index))
			: LmBf16ToFloat(__ldg(markov_embed + (uint64_t)previous * DS41_MARKOV + index - DS41_HIDDEN));
		dot += feature * LmBf16ToFloat(__ldg(weight + index));
	}
	dot = LmBlockSum<DSPARK_THREADS>(dot, reduction);
	if ( threadIdx.x == 0u )
		picks->scores[slot] = 1.0f / (1.0f + expf(-dot));
	if ( draft->greedy != 0u )
	{
		if ( draft->tree != 0u )
		{
			const float top = Ds41KeyValue(picks->keys[slot]);
			const float alt = Ds41KeyValue(picks->alt_keys[slot]);
			const float *step_partials = partials + (uint64_t)slot * 2u * blocks;
			float sum = 0.0f;
			for ( uint32_t block = threadIdx.x; block < blocks; block += DSPARK_THREADS )
				if ( step_partials[2u * block] != -INFINITY )
					sum += step_partials[2u * block + 1u] * expf(step_partials[2u * block] - top);
			sum = LmBlockSum<DSPARK_THREADS>(sum, reduction);
			if ( threadIdx.x == 0u )
			{
				picks->tokens[slot] = DsparkKeySlot(picks->keys[slot]);
				picks->alt_tokens[slot] = DsparkKeySlot(picks->alt_keys[slot]);
				picks->alt_scores[slot] = sum > 0.0f ? expf(alt - top) / sum : 0.0f;
			}
		}
		else if ( threadIdx.x == 0u )
		{
			picks->tokens[slot] = DsparkKeySlot(picks->keys[slot]);
			picks->alt_tokens[slot] = 0u;
			picks->alt_scores[slot] = 0.0f;
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41BranchRowKernel(uint32_t offset, uint32_t drafts, const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks,
	const float *__restrict__ logits, const uint16_t *__restrict__ markov_embed, const uint16_t *__restrict__ markov_head, float *__restrict__ partials)
{
	LmDependentWait();
	constexpr uint32_t PER = DS41_MARKOV / DSPARK_WARP;
	__shared__ float embedding[DS41_BRANCHES][DS41_MARKOV];
	__shared__ unsigned long long best[DS41_BRANCHES][DSPARK_WARPS];
	__shared__ float tops[DS41_BRANCHES][DSPARK_WARPS];
	__shared__ float sums[DS41_BRANCHES][DSPARK_WARPS];
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
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
			for ( uint32_t index = threadIdx.x; index < DS41_MARKOV; index += DSPARK_THREADS )
				embedding[b][index] = LmBf16ToFloat(__ldg(markov_embed + (uint64_t)previous * DS41_MARKOV + index));
		}
	}
	if ( any == 0u )
	{
		LmDependentRelease();
		return;
	}
	__syncthreads();
	for ( uint32_t token = blockIdx.x * DSPARK_WARPS + warp; token < DS41_VOCAB; token += gridDim.x * DSPARK_WARPS )
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
			dot = DsparkWarpSum(dot);
			if ( lane == 0u )
			{
				const float value = __ldg(logits + (uint64_t)steps[b] * DS41_VOCAB + token) + dot;
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
			for ( uint32_t index = 1u; index < DSPARK_WARPS; index++ )
			{
				Ds41SoftmaxMerge(&top[b], &sum[b], tops[b][index], sums[b][index]);
				if ( best[b][index] > first[b] )
					first[b] = best[b][index];
			}
			partials[((uint64_t)b * gridDim.x + blockIdx.x) * 2u] = top[b];
			partials[((uint64_t)b * gridDim.x + blockIdx.x) * 2u + 1u] = sum[b];
			atomicMax(&picks->branch_keys[b], first[b]);
		}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Ds41BranchReduceKernel(uint32_t offset, uint32_t drafts, const Ds41Draft *__restrict__ draft, Ds41Picks *__restrict__ picks,
	const float *__restrict__ partials, uint32_t blocks)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	const uint32_t branch = blockIdx.x, fork = picks->branch_fork[branch], step = fork + offset;
	if ( fork < drafts && step < drafts )
	{
		const float scale = draft->greedy != 0u ? 1.0f : draft->inverse_temperature;
		const unsigned long long key = picks->branch_keys[branch];
		const float top = Ds41KeyValue(key) * scale;
		const float *branch_partials = partials + (uint64_t)branch * 2u * blocks;
		float sum = 0.0f;
		for ( uint32_t block = threadIdx.x; block < blocks; block += DSPARK_THREADS )
			if ( branch_partials[2u * block] != -INFINITY )
				sum += branch_partials[2u * block + 1u] * expf(branch_partials[2u * block] - top);
		sum = LmBlockSum<DSPARK_THREADS>(sum, reduction);
		if ( threadIdx.x == 0u )
		{
			picks->branch_tokens[branch][offset] = DsparkKeySlot(key);
			picks->branch_scores[branch][offset] = sum > 0.0f ? 1.0f / sum : 0.0f;
			picks->branch_len[branch] = offset + 1u;
			picks->branch_keys[branch] = 0ull;
		}
	}
	LmDependentRelease();
}

static SparkStatus Ds41ConfigFloat(const SparkJsonDocument *document, int32_t object, const char *name, float *value)
{
	const int32_t token = DsparkMember(document, object, name);
	if ( token < 0 || SparkJsonGetFloat(document, token, value) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "dsv41 dspark: checkpoint_config %s is missing\n", name);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus Ds41Shape(Ds41State *state, const SparkJsonDocument *document)
{
	const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
	const int32_t noise = DsparkMember(document, checkpoint, "dspark_noise_token_id");
	const DsparkExpectedU64 expected[] = {
		{checkpoint, "dim", DS41_HIDDEN}, {checkpoint, "n_heads", DS41_HEADS}, {checkpoint, "head_dim", DS41_HEAD_DIM},
		{checkpoint, "rope_head_dim", DS41_ROPE}, {checkpoint, "q_lora_rank", DS41_QUERY_RANK}, {checkpoint, "o_groups", DS41_GROUPS},
		{checkpoint, "o_lora_rank", DS41_O_RANK}, {checkpoint, "window_size", DS41_WINDOW}, {checkpoint, "dspark_block_size", DS41_BLOCK},
		{checkpoint, "vocab_size", DS41_VOCAB}, {checkpoint, "dspark_markov_rank", DS41_MARKOV}, {checkpoint, "dspark_n_routed_experts", DS41_EXPERTS},
		{checkpoint, "dspark_n_activated_experts", DS41_ACTIVE}, {checkpoint, "moe_inter_dim", DS41_MOE}, {checkpoint, "hc_mult", DS41_HC},
		{checkpoint, "hc_sinkhorn_iters", DS41_SINKHORN}, {checkpoint, "n_mtp_layers", DS41_STAGES}, {checkpoint, "n_layers", DS41_LAYERS},
	};
	uint64_t value = 0u;
	SparkStatus status = DsparkCheckU64Table(document, "dsv41 dspark", expected, (uint32_t)(sizeof(expected) / sizeof(expected[0])));
	if ( status != SPARK_STATUS_OK )
		return status;
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

static SparkStatus Ds41Named(const Ds41State *state, const SparkJsonDocument *document, uint64_t payload_base, const char *format, uint32_t stage,
	const char *suffix, uint64_t bytes, const void **pointer)
{
	char name[160];
	snprintf(name, sizeof(name), format, stage, suffix);
	return DsparkTensorBytes(state->pack, document, payload_base, "dsv41 dspark", name, bytes, pointer);
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
	status = DsparkLoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Shape(state, &document);
	if ( status == SPARK_STATUS_OK )
		status = DsparkTensorBf16(state->pack, &document, payload_base, "dsv41 dspark", "target.embed", (uint64_t)DS41_VOCAB * DS41_HIDDEN,
			&state->embed);
	if ( status == SPARK_STATUS_OK )
		status = DsparkTensorBf16(state->pack, &document, payload_base, "dsv41 dspark", "target.head", (uint64_t)DS41_VOCAB * DS41_HIDDEN,
			&state->head);
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

static SparkStatus Ds41Expand(Ds41State *state, Ds41Matrix *matrix, uint64_t *used)
{
	const uint32_t groups = matrix->columns / DS41_QDQ;
	matrix->rows_scale = state->rows_scales + *used;
	*used += (uint64_t)matrix->rows * groups;
	LM_LAUNCH((Ds41ExpandScalesKernel), state->multiprocessors, DSPARK_THREADS, 0, state->stream, matrix->scale, matrix->rows_scale, matrix->rows,
		groups);
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Ds41Tables(Ds41State *state, const uint8_t **table)
{
	static const float initial[DS41_BLOCK * DS41_HC] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
	uint64_t used = 0u;
	SparkStatus status;
	if ( !DSPARK_DEVICE(state->rows_scales, state->rows_scales_bytes) ||
		!DSPARK_DEVICE(state->expert_table, (uint64_t)DS41_STAGES * DS41_EXPERTS * DS41_EXPERT_PARTS) ||
		!DSPARK_DEVICE(state->rope_inverse, DS41_ROPE / 2u) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( SparkGpuMemcpy(state->expert_table, table, (size_t)DS41_STAGES * DS41_EXPERTS * DS41_EXPERT_PARTS * sizeof(*table),
			SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(state->forward.pre_initial, initial, sizeof(initial), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
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
		LM_LAUNCH((DsparkRopeTableKernel<DS41_ROPE / 2u, DS41_ROPE>), 1u, DSPARK_THREADS, 0, state->stream, state->rope_inverse, state->rope_theta);
	if ( status == SPARK_STATUS_OK && (SparkGpuGetLastError() != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(state->stream) != SPARK_GPU_SUCCESS) )
		status = SPARK_STATUS_INTERNAL_ERROR;
	return status;
}

static SparkStatus Ds41Scratch(Ds41State *state, uint64_t context_bytes)
{
	Ds41Forward *f = &state->forward;
	int multiprocessors = 0;
	if ( !state->ring.Init(context_bytes, DS41_RING_ELEMENTS, &state->rings) ||
		SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		!DsparkAllocHost(&state->host_taps, (size_t)DS41_INGEST_ROWS * DS41_TAP_WIDTH) ||
		!DsparkAllocHost(&state->host_positions, DS41_INGEST_ROWS) || !DsparkAllocHost(&f->host_draft, 1u) ||
		!DsparkAllocHost(&f->host_picks, 1u) ||
		!DSPARK_DEVICE(state->taps, (uint64_t)DS41_INGEST_ROWS * DS41_TAP_WIDTH) ||
		!DSPARK_DEVICE(state->taps_quant, (uint64_t)DS41_INGEST_ROWS * DS41_TAP_WIDTH) || !DSPARK_DEVICE(state->positions, DS41_INGEST_ROWS) ||
		!DSPARK_DEVICE(state->main_raw, (uint64_t)DS41_INGEST_ROWS * DS41_HIDDEN) ||
		!DSPARK_DEVICE(state->main, (uint64_t)DS41_INGEST_ROWS * DS41_HIDDEN) ||
		!DSPARK_DEVICE(state->main_quant, (uint64_t)DS41_INGEST_ROWS * DS41_HIDDEN) ||
		!DSPARK_DEVICE(state->key_raw, (uint64_t)DS41_INGEST_ROWS * DS41_HEAD_DIM) || !DSPARK_DEVICE(f->draft, 1u) || !DSPARK_DEVICE(f->picks, 1u) ||
		!DSPARK_DEVICE(f->streams[0], DS41_BLOCK * DS41_HC_FLAT) || !DSPARK_DEVICE(f->streams[1], DS41_BLOCK * DS41_HC_FLAT) ||
		!DSPARK_DEVICE(f->mix_partials, DS41_BLOCK * LmHcSinkhornShape<DS41_HC>::kPartials) ||
		!DSPARK_DEVICE(f->pre_initial, DS41_BLOCK * DS41_HC) || !DSPARK_DEVICE(f->attn_pre, DS41_BLOCK * DS41_HC) ||
		!DSPARK_DEVICE(f->attn_post, DS41_BLOCK * DS41_HC) || !DSPARK_DEVICE(f->attn_comb, DS41_BLOCK * DS41_HC * DS41_HC) ||
		!DSPARK_DEVICE(f->ffn_pre[0], DS41_BLOCK * DS41_HC) || !DSPARK_DEVICE(f->ffn_pre[1], DS41_BLOCK * DS41_HC) ||
		!DSPARK_DEVICE(f->ffn_post, DS41_BLOCK * DS41_HC) || !DSPARK_DEVICE(f->ffn_comb, DS41_BLOCK * DS41_HC * DS41_HC) ||
		!DSPARK_DEVICE(f->collapsed, DS41_BLOCK * DS41_HIDDEN) || !DSPARK_DEVICE(f->x, DS41_BLOCK * DS41_HIDDEN) ||
		!DSPARK_DEVICE(f->xq, DS41_BLOCK * DS41_HIDDEN) || !DSPARK_DEVICE(f->q_raw, DS41_BLOCK * DS41_QUERY_RANK) ||
		!DSPARK_DEVICE(f->q_normed, DS41_BLOCK * DS41_QUERY_RANK) || !DSPARK_DEVICE(f->q_quant, DS41_BLOCK * DS41_QUERY_RANK) ||
		!DSPARK_DEVICE(f->q, (uint64_t)DS41_BLOCK * DS41_QUERY_WIDTH) || !DSPARK_DEVICE(f->kv_raw, DS41_BLOCK * DS41_HEAD_DIM) ||
		!DSPARK_DEVICE(f->kv, DS41_BLOCK * DS41_HEAD_DIM) || !DSPARK_DEVICE(f->attended, (uint64_t)DS41_BLOCK * DS41_QUERY_WIDTH) ||
		!DSPARK_DEVICE(f->grouped, (uint64_t)DS41_BLOCK * DS41_QUERY_WIDTH) || !DSPARK_DEVICE(f->projected, DS41_BLOCK * DS41_O_WIDTH) ||
		!DSPARK_DEVICE(f->projected_quant, DS41_BLOCK * DS41_O_WIDTH) || !DSPARK_DEVICE(f->attn_out, DS41_BLOCK * DS41_HIDDEN) ||
		!DSPARK_DEVICE(f->route_weight, DS41_PAIRS) || !DSPARK_DEVICE(f->route_expert, DS41_PAIRS) || !DSPARK_DEVICE(f->route_plan, 1u) ||
		!DSPARK_DEVICE(f->gate_scores, DS41_BLOCK * DS41_EXPERTS) || !DSPARK_DEVICE(f->pair_gate, DS41_PAIRS * DS41_MOE) ||
		!DSPARK_DEVICE(f->pair_up, DS41_PAIRS * DS41_MOE) || !DSPARK_DEVICE(f->pair_hidden, DS41_PAIRS * DS41_MOE) ||
		!DSPARK_DEVICE(f->pair_down, DS41_PAIRS * DS41_HIDDEN) || !DSPARK_DEVICE(f->shared_gate, DS41_BLOCK * DS41_MOE) ||
		!DSPARK_DEVICE(f->shared_up, DS41_BLOCK * DS41_MOE) || !DSPARK_DEVICE(f->shared_hidden, DS41_BLOCK * DS41_MOE) ||
		!DSPARK_DEVICE(f->shared_out, DS41_BLOCK * DS41_HIDDEN) || !DSPARK_DEVICE(f->moe_out, DS41_BLOCK * DS41_HIDDEN) ||
		!DSPARK_DEVICE(f->hidden, DS41_BLOCK * DS41_HIDDEN) || !DSPARK_DEVICE(f->normed, DS41_BLOCK * DS41_HIDDEN) ||
		!DSPARK_DEVICE(f->logits, (uint64_t)DS41_BLOCK * DS41_VOCAB) || !DSPARK_DEVICE(f->row, DS41_VOCAB) ||
		!DSPARK_DEVICE(f->candidates, DS41_SAMPLED) ||
		!DSPARK_DEVICE(f->softmax_partials, (uint64_t)DS41_BLOCK * 8u * (uint64_t)multiprocessors) ||
		!DSPARK_DEVICE(f->topk_values, 2u * LmTopkExactCandidateEntries(DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK)) ||
		!DSPARK_DEVICE(f->topk_positions, 2u * LmTopkExactCandidateEntries(DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK)) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static SparkStatus Ds41Linear(Ds41State *state, const Ds41Matrix *matrix, uint64_t row_offset, uint32_t outputs, const uint16_t *input,
	uint16_t *output, uint32_t rows)
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

static void Ds41Qdq(Ds41State *state, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t parts, uint32_t part_width,
	uint32_t to_part_major)
{
	const uint64_t groups = (uint64_t)rows * parts * part_width / DS41_QDQ;
	const uint64_t blocks = (groups + DSPARK_WARPS - 1u) / DSPARK_WARPS;
	const uint32_t grid = blocks < 4u * state->multiprocessors ? (uint32_t)blocks : 4u * state->multiprocessors;
	(void)LM_LAUNCH_DEPENDENT((Ds41QdqKernel), grid, DSPARK_THREADS, 0, state->stream, input, output, rows, parts, part_width, to_part_major);
}

static SparkStatus Ds41IngestRows(Ds41State *state, Ds41Sequence *sequence, uint32_t rows)
{
	SparkStatus status;
	Ds41Qdq(state, state->taps, state->taps_quant, rows, 1u, DS41_TAP_WIDTH, 0u);
	status = Ds41Linear(state, &state->main_proj, 0u, DS41_HIDDEN, state->taps_quant, state->main_raw, rows);
	if ( status != SPARK_STATUS_OK )
		return status;
	(void)LM_LAUNCH_DEPENDENT((Ds41RmsQdqKernel), rows, DSPARK_THREADS, 0, state->stream, state->main_raw, state->main_norm, state->main_quant,
		DS41_HIDDEN, state->epsilon);
	for ( uint32_t stage = 0u; stage < DS41_STAGES && status == SPARK_STATUS_OK; stage++ )
	{
		status = Ds41Linear(state, &state->stages[stage].wkv, 0u, DS41_HEAD_DIM, state->main_quant, state->key_raw, rows);
		if ( status == SPARK_STATUS_OK )
			(void)LM_LAUNCH_DEPENDENT((Ds41KeyRowsKernel), rows, DSPARK_THREADS, 0, state->stream, state->key_raw, state->stages[stage].kv_norm,
				state->rope_inverse, state->positions, state->epsilon, sequence->cache + (uint64_t)stage * DS41_STAGE_RING, (uint16_t *)0);
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	return DsparkSettle(state->stream, status);
}

static SparkStatus Ds41Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	Ds41State *state = (Ds41State *)opaque;
	return state->ring.IngestFrame<DS41_TAPS, DS41_HIDDEN, DS41_INGEST_ROWS>(frame, state->host_taps, state->host_positions, state->taps,
		state->positions, state->stream, &state->frames, &state->rows,
		[&](Ds41Sequence *sequence, uint32_t part) { return Ds41IngestRows(state, sequence, part); });
}

static SparkStatus Ds41Attention(Ds41State *state, uint32_t stage)
{
	Ds41Forward *f = &state->forward;
	const Ds41Stage *s = &state->stages[stage];
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)f->draft + offsetof(Ds41Draft, positions));
	SparkStatus status = Ds41Linear(state, &s->wq_a, 0u, DS41_QUERY_RANK, f->xq, f->q_raw, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	(void)LM_LAUNCH_DEPENDENT((Ds41RmsQdqKernel), DS41_BLOCK, DSPARK_THREADS, 0, stream, f->q_raw, s->q_norm, f->q_quant, DS41_QUERY_RANK,
		state->epsilon);
	status = Ds41Linear(state, &s->wq_b, 0u, DS41_QUERY_WIDTH, f->q_quant, f->q, DS41_BLOCK);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Linear(state, &s->wkv, 0u, DS41_HEAD_DIM, f->xq, f->kv_raw, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	(void)LM_LAUNCH_DEPENDENT((Ds41RopeKernel), (DS41_BLOCK * DS41_HEADS * (DS41_ROPE / 2u) + DSPARK_THREADS - 1u) / DSPARK_THREADS, DSPARK_THREADS,
		0, stream, f->q, f->draft, state->rope_inverse, DS41_HEADS, 1.0f);
	(void)LM_LAUNCH_DEPENDENT((Ds41KeyRowsKernel), DS41_BLOCK, DSPARK_THREADS, 0, stream, f->kv_raw, s->kv_norm, state->rope_inverse, positions,
		state->epsilon, (uint16_t *)0, f->kv);
	(void)LM_LAUNCH_DEPENDENT((Ds41AttentionKernel), dim3(DS41_HEADS, DS41_BLOCK), DSPARK_THREADS, 0, stream, f->q, stage, f->kv, f->draft, s->sink,
		state->softmax_scale, f->attended);
	(void)LM_LAUNCH_DEPENDENT((Ds41RopeKernel), (DS41_BLOCK * DS41_HEADS * (DS41_ROPE / 2u) + DSPARK_THREADS - 1u) / DSPARK_THREADS, DSPARK_THREADS,
		0, stream, f->attended, f->draft, state->rope_inverse, DS41_HEADS, -1.0f);
	Ds41Qdq(state, f->attended, f->grouped, DS41_BLOCK, DS41_GROUPS, DS41_GROUP_WIDTH, 1u);
	if ( LM_LAUNCH_DEPENDENT((Ds41WoAKernel), dim3(DS41_O_RANK / DSPARK_WARPS, DS41_GROUPS), DSPARK_THREADS, 0, stream, s->wo_a.weight,
			s->wo_a.rows_scale, f->grouped, f->projected) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	Ds41Qdq(state, f->projected, f->projected_quant, DS41_BLOCK, DS41_GROUPS, DS41_O_RANK, 0u);
	return Ds41Linear(state, &s->wo_b, 0u, DS41_HIDDEN, f->projected_quant, f->attn_out, DS41_BLOCK);
}

static SparkStatus Ds41Moe(Ds41State *state, uint32_t stage)
{
	Ds41Forward *f = &state->forward;
	const Ds41Stage *s = &state->stages[stage];
	const SparkGpuStream stream = state->stream;
	const uint32_t swiglu_blocks = (DS41_PAIRS * DS41_MOE / DS41_QDQ + DSPARK_WARPS - 1u) / DSPARK_WARPS;
	SparkStatus status;
	(void)LM_LAUNCH_DEPENDENT((Ds41GateScoreKernel), DS41_EXPERTS / DSPARK_WARPS, DSPARK_THREADS, 0, stream, f->x, s->gate, f->gate_scores);
	(void)LM_LAUNCH_DEPENDENT((Ds41GateTopKernel), 1u, DS41_BLOCK * DSPARK_WARP, 0, stream, f->gate_scores, s->gate_bias, state->route_scale,
		f->route_expert, f->route_weight, f->route_plan);
	(void)LM_LAUNCH_DEPENDENT((Ds41ExpertUpKernel), dim3(DS41_PAIRS, DS41_MOE / DS41_EXPERT_NEURONS), DSPARK_THREADS, 0, stream, f->xq, s->experts,
		f->route_plan, f->pair_gate, f->pair_up);
	(void)LM_LAUNCH_DEPENDENT((Ds41SwigluKernel), swiglu_blocks, DSPARK_THREADS, 0, stream, f->pair_gate, f->pair_up, f->route_weight,
		state->swiglu_limit, DS41_PAIRS, f->pair_hidden);
	(void)LM_LAUNCH_DEPENDENT((Ds41ExpertDownKernel), dim3(DS41_PAIRS, DS41_HIDDEN / DS41_EXPERT_NEURONS), DSPARK_THREADS, DS41_DOWN_SHARED_BYTES,
		stream, f->pair_hidden, s->experts, f->route_plan, f->pair_down);
	status = Ds41Linear(state, &s->shared_w1, 0u, DS41_MOE, f->xq, f->shared_gate, DS41_BLOCK);
	if ( status == SPARK_STATUS_OK )
		status = Ds41Linear(state, &s->shared_w3, 0u, DS41_MOE, f->xq, f->shared_up, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	(void)LM_LAUNCH_DEPENDENT((Ds41SwigluKernel), (DS41_BLOCK * DS41_MOE / DS41_QDQ + DSPARK_WARPS - 1u) / DSPARK_WARPS, DSPARK_THREADS, 0, stream,
		f->shared_gate, f->shared_up, (const float *)0, state->swiglu_limit, DS41_BLOCK, f->shared_hidden);
	status = Ds41Linear(state, &s->shared_w2, 0u, DS41_HIDDEN, f->shared_hidden, f->shared_out, DS41_BLOCK);
	if ( status != SPARK_STATUS_OK )
		return status;
	(void)LM_LAUNCH_DEPENDENT((Ds41MoeSumKernel), dim3(DS41_HIDDEN / DSPARK_THREADS, DS41_BLOCK), DSPARK_THREADS, 0, stream, f->pair_down,
		f->shared_out, f->moe_out);
	return SPARK_STATUS_OK;
}

static SparkStatus Ds41Mixes(Ds41State *state, const uint16_t *streams, const float *fn, const float *scale, const float *base, float *pre,
	float *post, float *comb)
{
	return LmHcSinkhornMixes<DS41_HC>(streams, fn, scale, base, state->forward.mix_partials, pre, post, comb, DS41_BLOCK, DS41_HIDDEN, state->epsilon,
		DS41_SINKHORN, state->hc_epsilon, state->stream) == LM_LAUNCH_OK ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Ds41RunForward(Ds41State *state, uint32_t stages, const uint16_t **streams_out, const float **pre_out)
{
	Ds41Forward *f = &state->forward;
	const SparkGpuStream stream = state->stream;
	const float *pre = f->pre_initial;
	uint32_t current = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	(void)LM_LAUNCH_DEPENDENT((Ds41EmbedKernel), DS41_BLOCK, DSPARK_THREADS, 0, stream, state->embed, f->draft, f->streams[0]);
	for ( uint32_t stage = 0u; stage < stages && status == SPARK_STATUS_OK; stage++ )
	{
		const Ds41Stage *s = &state->stages[stage];
		status = Ds41Mixes(state, f->streams[current], s->hc_attn_fn, s->hc_attn_scale, s->hc_attn_base, f->attn_pre, f->attn_post, f->attn_comb);
		if ( status != SPARK_STATUS_OK )
			break;
		(void)LM_LAUNCH_DEPENDENT((Ds41CollapseRmsQdqKernel), DS41_BLOCK, DSPARK_THREADS, 0, stream, f->streams[current], pre, s->attn_norm,
			(uint16_t *)0, f->xq, state->epsilon);
		status = Ds41Attention(state, stage);
		if ( status != SPARK_STATUS_OK )
			break;
		(void)LM_LAUNCH_DEPENDENT((Ds41HcPostKernel), dim3(DS41_HIDDEN / DSPARK_THREADS, DS41_BLOCK), DSPARK_THREADS, 0, stream, f->attn_out,
			f->streams[current], f->attn_post, f->attn_comb, f->streams[1u - current]);
		current = 1u - current;
		status = Ds41Mixes(state, f->streams[current], s->hc_ffn_fn, s->hc_ffn_scale, s->hc_ffn_base, f->ffn_pre[stage % 2u], f->ffn_post,
			f->ffn_comb);
		if ( status != SPARK_STATUS_OK )
			break;
		(void)LM_LAUNCH_DEPENDENT((Ds41CollapseRmsQdqKernel), DS41_BLOCK, DSPARK_THREADS, 0, stream, f->streams[current], f->attn_pre, s->ffn_norm,
			f->x, f->xq, state->epsilon);
		status = Ds41Moe(state, stage);
		if ( status != SPARK_STATUS_OK )
			break;
		(void)LM_LAUNCH_DEPENDENT((Ds41HcPostKernel), dim3(DS41_HIDDEN / DSPARK_THREADS, DS41_BLOCK), DSPARK_THREADS, 0, stream, f->moe_out,
			f->streams[current], f->ffn_post, f->ffn_comb, f->streams[1u - current]);
		current = 1u - current;
		pre = f->ffn_pre[stage % 2u];
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( streams_out != 0 )
		*streams_out = f->streams[current];
	if ( pre_out != 0 )
		*pre_out = pre;
	if ( stages < DS41_STAGES )
		return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	(void)LM_LAUNCH_DEPENDENT((Ds41CollapseRmsKernel), DS41_BLOCK, DSPARK_THREADS, 0, stream, f->streams[current], pre, state->final_norm, f->hidden,
		f->normed, state->epsilon);
	status = DsparkGemmF32(state->head, f->normed, f->logits, DS41_BLOCK, DS41_HIDDEN, DS41_VOCAB, state->multiprocessors, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Ds41Heads(Ds41State *state, uint32_t greedy, uint32_t drafts, uint32_t tree)
{
	Ds41Forward *f = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t blocks = 4u * state->multiprocessors;
	const uint64_t entries = 2u * LmTopkExactCandidateEntries(DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK);
	if ( SparkGpuMemsetAsync(f->picks, 0, sizeof(Ds41Picks), stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	for ( uint32_t step = 0u; step < DS41_BLOCK; step++ )
	{
		(void)LM_LAUNCH_DEPENDENT((Ds41MarkovRowKernel), blocks, DSPARK_THREADS, 0, stream, step, f->draft, f->picks, f->logits, state->markov_embed,
			state->markov_head, f->row, f->softmax_partials);
		if ( greedy != 0u )
			continue;
		if ( LmTopkExactLaunch<DS41_TOPK_THREADS>(f->row, 1u, DS41_VOCAB, DS41_SAMPLED, DS41_TOPK_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, f->topk_values,
				f->topk_positions, entries, f->candidates, stream) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		(void)LM_LAUNCH_DEPENDENT((Ds41SampleKernel), 1u, DSPARK_PICK_THREADS, 0, stream, step, f->draft, f->row, f->candidates,
			f->softmax_partials + (uint64_t)step * 2u * blocks, blocks, f->picks);
	}
	(void)LM_LAUNCH_DEPENDENT((Ds41ConfidenceKernel), DS41_BLOCK, DSPARK_THREADS, 0, stream, f->draft, f->picks, f->hidden, state->markov_embed,
		state->confidence, f->softmax_partials, blocks);
	if ( tree != 0u && drafts >= 2u )
	{
		(void)LM_LAUNCH_DEPENDENT((DsparkBranchSelectKernel<DS41_BLOCK, DS41_BRANCHES, Ds41Draft, Ds41Picks>), 1u, DSPARK_WARP, 0, stream, f->picks,
			f->draft, SPARK_DRAFT_TREE_DEFAULT_MIN_SCORE);
		for ( uint32_t offset = 1u; offset < DS41_BRANCH_LEN && offset < drafts; offset++ )
		{
			(void)LM_LAUNCH_DEPENDENT((Ds41BranchRowKernel), blocks, DSPARK_THREADS, 0, stream, offset, drafts, f->draft, f->picks, f->logits,
				state->markov_embed, state->markov_head, f->softmax_partials);
			(void)LM_LAUNCH_DEPENDENT((Ds41BranchReduceKernel), DS41_BRANCHES, DSPARK_THREADS, 0, stream, offset, drafts, f->draft, f->picks,
				f->softmax_partials, blocks);
		}
	}
	if ( SparkGpuMemcpyAsync(f->host_picks, f->picks, sizeof(Ds41Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Ds41FillDraft(Ds41State *state, const Ds41Sequence *sequence, uint32_t anchor, uint64_t position, uint32_t tree, uint32_t drafts,
	const SparkRowSampling *sampling)
{
	Ds41Draft *draft = state->forward.host_draft;
	if ( anchor >= DS41_VOCAB || position + DS41_BLOCK > UINT32_MAX )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(draft, 0, sizeof(*draft));
	draft->ring = sequence->cache;
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
	draft->tree = tree;
	draft->drafts = drafts;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	return SPARK_STATUS_OK;
}

static SparkStatus Ds41Prepare(Ds41State *state, const Ds41Sequence *sequence, uint32_t anchor, uint64_t position, const SparkRowSampling *sampling)
{
	const SparkStatus status = Ds41FillDraft(state, sequence, anchor, position, 1u, DS41_BLOCK, sampling);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkGpuMemcpyAsync(state->forward.draft, state->forward.host_draft, sizeof(Ds41Draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) ==
		SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus Ds41Close(void *opaque)
{
	Ds41State *state = (Ds41State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	SparkDrafterGraphCacheDestroy(&state->graphs);
	{
		Ds41Forward *f = &state->forward;
		void *buffers[] = {
			f->draft, f->picks, f->streams[0], f->streams[1], f->mix_partials, f->pre_initial, f->attn_pre, f->attn_post, f->attn_comb,
			f->ffn_pre[0], f->ffn_pre[1], f->ffn_post, f->ffn_comb, f->collapsed, f->x, f->xq, f->q_raw, f->q_normed, f->q_quant, f->q, f->kv_raw,
			f->kv, f->attended, f->grouped, f->projected, f->projected_quant, f->attn_out, f->route_weight, f->route_expert, f->route_plan,
			f->gate_scores, f->pair_gate, f->pair_up, f->pair_hidden, f->pair_down, f->shared_gate, f->shared_up, f->shared_hidden, f->shared_out,
			f->moe_out, f->hidden, f->normed, f->logits, f->row, f->candidates, f->softmax_partials, f->topk_values, f->topk_positions,
			state->rings, state->taps, state->taps_quant, state->positions, state->main_raw, state->main, state->main_quant, state->key_raw,
			state->rows_scales, (void *)state->expert_table, state->rope_inverse,
		};
		void *host_buffers[] = { f->host_draft, f->host_picks, state->host_taps, state->host_positions };
		DsparkFreeDeviceBuffers(buffers, (uint32_t)(sizeof(buffers) / sizeof(buffers[0])));
		DsparkFreeHostBuffers(host_buffers, (uint32_t)(sizeof(host_buffers) / sizeof(host_buffers[0])));
	}
	if ( state->stream != 0 )
		(void)SparkGpuStreamDestroy(state->stream);
	if ( state->pack != 0 )
		status = SparkWeightdLazyPackDestroy(state->pack);
	free(state->ring.sequences);
	free(state);
	return status;
}

static SparkStatus Ds41Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	return DsparkDrafterOpen<Ds41State>(lane, DS41_RING_BYTES, "lane %s has a context cap below one sequence's window",
		"lane %s: the pack %s is not a DeepSeek V4.1 Flash DSpark pack this module reads", true, state_out, bind, Ds41Close,
		[&](Ds41State *state) -> SparkStatus {
			const uint8_t **table = (const uint8_t **)calloc((size_t)DS41_STAGES * DS41_EXPERTS * DS41_EXPERT_PARTS, sizeof(*table));
			SparkStatus status;
			if ( table == 0 )
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			status = Ds41Weights(state, lane->pack_path, table);
			if ( status == SPARK_STATUS_OK )
				status = Ds41Scratch(state, lane->context_bytes);
			if ( status == SPARK_STATUS_OK )
				status = Ds41Tables(state, table);
			free(table);
			return status;
		},
		[](const Ds41State *state) { return state->ring.capacity; });
}

static SparkStatus Ds41Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	Ds41State *state = (Ds41State *)opaque;
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > DS41_WINDOW ? request->position - DS41_WINDOW : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	Ds41Sequence *sequence = 0;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = state->ring.PrepareProposeContext(request, first, state->stream, &sequence, &checked,
		[&](Ds41Sequence *dst, const Ds41Sequence *src, uint32_t slot, uint32_t n) -> SparkStatus {
			const size_t bytes = (size_t)n * DS41_HEAD_DIM * sizeof(uint16_t);
			for ( uint32_t stage = 0u; stage < DS41_STAGES; stage++ )
				if ( SparkGpuMemcpyAsync(dst->cache + (uint64_t)stage * DS41_STAGE_RING + (uint64_t)slot * DS41_HEAD_DIM,
						src->cache + (uint64_t)stage * DS41_STAGE_RING + (uint64_t)slot * DS41_HEAD_DIM, bytes, SPARK_GPU_COPY_DEVICE_TO_DEVICE,
						state->stream) != SPARK_GPU_SUCCESS )
					return SPARK_STATUS_IO_ERROR;
			return SPARK_STATUS_OK;
		});
	if ( status != SPARK_STATUS_OK )
		return status;
	drafts = drafts < DS41_BLOCK ? drafts : DS41_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		const uint32_t greedy = request->sampling.inverse_temperature > 0.0f ? 0u : 1u;
		const uint64_t graph_key = ((uint64_t)greedy << 32u) | ((uint64_t)tree << 16u) | (uint64_t)drafts;
		status = Ds41FillDraft(state, sequence, request->anchor_token_id, request->position, tree, drafts, &request->sampling);
		if ( status == SPARK_STATUS_OK )
			status = SparkDrafterGraphRun(&state->graphs, state->stream, graph_key, [&]() -> SparkStatus {
				if ( SparkGpuMemcpyAsync(state->forward.draft, state->forward.host_draft, sizeof(Ds41Draft), SPARK_GPU_COPY_HOST_TO_DEVICE,
						state->stream) != SPARK_GPU_SUCCESS )
					return SPARK_STATUS_IO_ERROR;
				SparkStatus inner = Ds41RunForward(state, DS41_STAGES, (const uint16_t **)0, (const float **)0);
				if ( inner == SPARK_STATUS_OK )
					inner = Ds41Heads(state, greedy, drafts, tree);
				return inner;
			});
		status = DsparkSettle(state->stream, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		*count = DsparkPackMarkovTree<DS41_BLOCK, DS41_BRANCHES>(request->anchor_token_id, source, state->forward.host_picks, tree, drafts, requested,
			limit, nodes);
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

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &ds41_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_dsv41_dspark_ring(void *opaque, uint64_t sequence_id, uint16_t *keys,
	int64_t *positions)
{
	Ds41State *state = (Ds41State *)opaque;
	const Ds41Sequence *sequence = state != 0 ? state->ring.Live(sequence_id) : 0;
	if ( sequence == 0 || keys == 0 || positions == 0 )
		return SPARK_STATUS_NOT_FOUND;
	memcpy(positions, sequence->positions, sizeof(sequence->positions));
	return DsparkReadback(keys, sequence->cache, DS41_RING_BYTES);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_dsv41_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *hidden, float *logits, uint64_t *elapsed_ns)
{
	Ds41State *state = (Ds41State *)opaque;
	const Ds41Sequence *sequence = state != 0 ? state->ring.Live(sequence_id) : 0;
	const uint64_t started = SparkMonotonicNs();
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		return SPARK_STATUS_NOT_FOUND;
	memset(&greedy, 0, sizeof(greedy));
	status = Ds41Prepare(state, sequence, anchor, position, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Ds41RunForward(state, DS41_STAGES, (const uint16_t **)0, (const float **)0);
	status = DsparkSettle(state->stream, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && hidden != 0 )
		status = DsparkReadback(hidden, state->forward.hidden, (size_t)DS41_BLOCK * DS41_HIDDEN * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && logits != 0 )
		status = DsparkReadback(logits, state->forward.logits, (size_t)DS41_BLOCK * DS41_VOCAB * sizeof(float));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_dsv41_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t stages, uint16_t *streams, float *pre)
{
	Ds41State *state = (Ds41State *)opaque;
	const Ds41Sequence *sequence = state != 0 ? state->ring.Live(sequence_id) : 0;
	const uint16_t *streams_at = 0;
	const float *pre_at = 0;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 || streams == 0 || pre == 0 || stages > DS41_STAGES )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&greedy, 0, sizeof(greedy));
	status = Ds41Prepare(state, sequence, anchor, position, &greedy);
	if ( status == SPARK_STATUS_OK )
		status = Ds41RunForward(state, stages, &streams_at, &pre_at);
	status = DsparkSettle(state->stream, status);
	if ( status == SPARK_STATUS_OK )
		status = DsparkReadback(streams, streams_at, (size_t)DS41_BLOCK * DS41_HC_FLAT * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = DsparkReadback(pre, pre_at, (size_t)DS41_BLOCK * DS41_HC * sizeof(float));
	return status;
}
