#include "sparkpipe/spark_glm52_model.h"
#include "inference/kernels/dspark_draft_stage_common.cuh"

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
#define G53F_MLP SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define G53F_BLOCK 8u
#define G53F_DRAFTS (G53F_BLOCK - 1u)
#define G53F_SLIDING 2048u
#define G53F_WINDOW 2048u
#define G53F_COLUMNS (G53F_WINDOW + G53F_BLOCK)
#define G53F_VOCAB SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT
#define G53F_CONV_GROUP 16u
#define G53F_CONV_GROUPS (G53F_HIDDEN / G53F_CONV_GROUP)
#define G53F_CONV_TAPS 2u
#define G53F_CONV_WIDTH (2u * G53F_CONV_TAPS * G53F_CONV_GROUPS)
#define G53F_SELECTOR_RANK 256u
#define G53F_TOP_K 16u
#define G53F_BRANCH_LEN 4u
#define G53F_INGEST_ROWS 256u
#define G53F_SPLIT_KEYS 256u
#define G53F_SPLITS ((G53F_WINDOW + G53F_SPLIT_KEYS - 1u) / G53F_SPLIT_KEYS)
#define G53F_ATTN_TILE 32u
#define G53F_ATTN_THREADS (G53F_BLOCK * DSPARK_WARP)
#define G53F_SEQUENCES_MAX 256u
#define G53F_LAYER_KV ((uint64_t)G53F_KV_HEADS * G53F_COLUMNS * G53F_HEAD_DIM)
#define G53F_SEQUENCE_ELEMENTS (2u * G53F_LAYERS * G53F_LAYER_KV)
#define G53F_SEQUENCE_BYTES (G53F_SEQUENCE_ELEMENTS * sizeof(uint16_t))

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

using G53FSequence = DsparkRingSequence<G53F_WINDOW>;
using G53FRingPool = DsparkRingPool<G53F_WINDOW, G53F_SEQUENCES_MAX>;

typedef struct G53FDraft
{
	uint16_t *cache;
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
	uint32_t splits;
	uint32_t tree;
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
	SparkDrafterTopkBf16Workspace topk_ws;
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
	SparkDrafterGraphCache graphs;
	uint16_t *host_taps;
	uint32_t *host_positions;
	uint16_t *taps;
	uint32_t *positions;
	float *features_raw;
	uint16_t *features;
	uint16_t *keys_raw;
	uint16_t *values_raw;
	uint16_t *caches;
	G53FRingPool ring;
	uint64_t frames;
	uint64_t rows;
} G53FState;

static __device__ __forceinline__ float G53FLogAddExp(float a, float b)
{
	const float hi = fmaxf(a, b), lo = fminf(a, b);
	if ( hi == -INFINITY )
		return -INFINITY;
	return hi + log1pf(expf(lo - hi));
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void G53FConvPrepareKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ base, const uint16_t *__restrict__ delta,
	uint16_t *__restrict__ output)
{
	const uint32_t row = blockIdx.x;
	const uint16_t *base0 = base, *base1 = base + G53F_HIDDEN;
	const uint16_t *delta0 = delta + (uint64_t)row * G53F_CONV_WIDTH;
	const uint16_t *delta1 = delta0 + G53F_CONV_GROUPS;
	const uint16_t *in_cur = input + (uint64_t)row * G53F_HIDDEN;
	const uint16_t *in_prev = row > 0u ? input + (uint64_t)(row - 1u) * G53F_HIDDEN : (const uint16_t *)0;
	uint16_t *out_row = output + (uint64_t)row * G53F_HIDDEN;
	for ( uint32_t col = threadIdx.x; col < G53F_HIDDEN; col += DSPARK_THREADS )
	{
		const uint32_t group = col / G53F_CONV_GROUP;
		const float c0 = LmBf16ToFloat(__ldg(base0 + col)) + LmBf16ToFloat(__ldg(delta0 + group));
		float acc = c0 * LmBf16ToFloat(__ldg(in_cur + col));
		if ( in_prev != 0 )
		{
			const float c1 = LmBf16ToFloat(__ldg(base1 + col)) + LmBf16ToFloat(__ldg(delta1 + group));
			acc = fmaf(c1, LmBf16ToFloat(__ldg(in_prev + col)), acc);
		}
		out_row[col] = LmFloatToBf16(acc);
	}
	LmDependentRelease();
}

template<bool DO_RMS>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void G53FConvFinishKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ base, const uint16_t *__restrict__ delta,
	float *__restrict__ hidden, const uint16_t *__restrict__ weight, uint16_t *__restrict__ normed, float epsilon)
{
	__shared__ float reduction[DSPARK_WARPS];
	const uint32_t row = blockIdx.x;
	const uint16_t *base0 = base + 2u * G53F_HIDDEN, *base1 = base + 3u * G53F_HIDDEN;
	const uint16_t *delta0 = delta + (uint64_t)row * G53F_CONV_WIDTH + 2u * G53F_CONV_GROUPS;
	const uint16_t *delta1 = delta0 + G53F_CONV_GROUPS;
	const uint16_t *in_cur = input + (uint64_t)row * G53F_HIDDEN;
	const uint16_t *in_prev = row > 0u ? input + (uint64_t)(row - 1u) * G53F_HIDDEN : (const uint16_t *)0;
	float *hid_row = hidden + (uint64_t)row * G53F_HIDDEN;
	float total = 0.0f;
	for ( uint32_t col = threadIdx.x; col < G53F_HIDDEN; col += DSPARK_THREADS )
	{
		const uint32_t group = col / G53F_CONV_GROUP;
		const float c0 = LmBf16ToFloat(__ldg(base0 + col)) + LmBf16ToFloat(__ldg(delta0 + group));
		float acc = c0 * LmBf16ToFloat(__ldg(in_cur + col));
		if ( in_prev != 0 )
		{
			const float c1 = LmBf16ToFloat(__ldg(base1 + col)) + LmBf16ToFloat(__ldg(delta1 + group));
			acc = fmaf(c1, LmBf16ToFloat(__ldg(in_prev + col)), acc);
		}
		const float value = hid_row[col] + acc;
		hid_row[col] = value;
		if constexpr ( DO_RMS )
			total += value * value;
	}
	if constexpr ( DO_RMS )
	{
		const float scale = rsqrtf(LmBlockSum<DSPARK_THREADS>(total, reduction) / (float)G53F_HIDDEN + epsilon);
		uint16_t *out_row = normed + (uint64_t)row * G53F_HIDDEN;
		for ( uint32_t col = threadIdx.x; col < G53F_HIDDEN; col += DSPARK_THREADS )
			out_row[col] = LmFloatToBf16(hid_row[col] * scale * LmBf16ToFloat(__ldg(weight + col)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(G53F_ATTN_THREADS, 2)
void G53FAttentionSplitKernel(const uint16_t *__restrict__ q, uint64_t keys_offset, uint64_t values_offset,
	const G53FDraft *__restrict__ draft, float scale, float *__restrict__ partials)
{
	LmDependentWait();
	constexpr uint32_t STRIDE = G53F_HALF + 4u;
	__shared__ __align__(16) uint32_t s_k[2][G53F_ATTN_TILE][STRIDE];
	__shared__ __align__(16) uint32_t s_v[2][G53F_ATTN_TILE][STRIDE];
	__shared__ float s_scores[G53F_BLOCK][G53F_ATTN_TILE];
	__shared__ uint8_t s_vis[2][G53F_ATTN_TILE];
	const uint32_t kv_head = blockIdx.x, split = blockIdx.y;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint32_t warp_bit = 1u << warp;
	const uint32_t first = draft->start + split * G53F_SPLIT_KEYS, end = draft->block_at;
	const uint32_t stop = first + G53F_SPLIT_KEYS < end ? first + G53F_SPLIT_KEYS : end;
	const uint32_t tiles = first < stop ? (stop - first + G53F_ATTN_TILE - 1u) / G53F_ATTN_TILE : 0u;
	const uint32_t masked = draft->masked;
	const uint16_t *head_keys = draft->cache + keys_offset + (uint64_t)kv_head * G53F_COLUMNS * G53F_HEAD_DIM;
	const uint16_t *head_values = draft->cache + values_offset + (uint64_t)kv_head * G53F_COLUMNS * G53F_HEAD_DIM;
	float q0[G53F_GROUP], q1[G53F_GROUP], q2[G53F_GROUP], q3[G53F_GROUP];
	float running_max[G53F_GROUP], running_sum[G53F_GROUP];
	float acc0[G53F_GROUP], acc1[G53F_GROUP], acc2[G53F_GROUP], acc3[G53F_GROUP];
#pragma unroll
	for ( uint32_t g = 0u; g < G53F_GROUP; g++ )
	{
		const uint32_t head = kv_head * G53F_GROUP + g;
		const uint16_t *q_row = q + ((uint64_t)warp * G53F_HEADS + head) * G53F_HEAD_DIM;
		const uint32_t qp0 = __ldg((const uint32_t *)(q_row + 2u * lane));
		const uint32_t qp1 = __ldg((const uint32_t *)(q_row + G53F_HALF + 2u * lane));
		q0[g] = LmBf16ToFloat((uint16_t)(qp0 & 0xffffu));
		q1[g] = LmBf16ToFloat((uint16_t)(qp0 >> 16u));
		q2[g] = LmBf16ToFloat((uint16_t)(qp1 & 0xffffu));
		q3[g] = LmBf16ToFloat((uint16_t)(qp1 >> 16u));
		running_max[g] = -INFINITY;
		running_sum[g] = 0.0f;
		acc0[g] = acc1[g] = acc2[g] = acc3[g] = 0.0f;
	}
	auto stage = [&](uint32_t tile_idx, uint32_t buf)
	{
		const uint32_t tile_first = first + tile_idx * G53F_ATTN_TILE;
#pragma unroll
		for ( uint32_t item = threadIdx.x; item < G53F_ATTN_TILE * 16u; item += G53F_ATTN_THREADS )
		{
			const uint32_t load_key = item >> 4u, load_chunk = item & 15u;
			const uint32_t col = tile_first + load_key;
			const uint32_t bytes = col < stop ? 16u : 0u;
			const uint32_t safe_col = col < stop ? col : 0u;
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_k[buf][load_key][load_chunk * 4u]),
				head_keys + (uint64_t)safe_col * G53F_HEAD_DIM + load_chunk * 8u, bytes);
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_v[buf][load_key][load_chunk * 4u]),
				head_values + (uint64_t)safe_col * G53F_HEAD_DIM + load_chunk * 8u, bytes);
		}
		SparkDrafterCommitGroup();
		if ( threadIdx.x < G53F_ATTN_TILE )
		{
			const uint32_t col = tile_first + threadIdx.x;
			s_vis[buf][threadIdx.x] = col < stop ? (masked == 0u ? (uint8_t)0xffu : draft->valid[col]) : (uint8_t)0u;
		}
	};
	if ( tiles != 0u )
		stage(0u, 0u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t buf = tile & 1u;
		if ( tile + 1u < tiles )
		{
			stage(tile + 1u, buf ^ 1u);
			SparkDrafterWaitGroup<1u>();
		}
		else
			SparkDrafterWaitGroup<0u>();
		__syncthreads();
#pragma unroll
		for ( uint32_t g = 0u; g < G53F_GROUP; g++ )
		{
			for ( uint32_t t = 0u; t < G53F_ATTN_TILE; t++ )
			{
				const uint32_t kp0 = s_k[buf][t][lane];
				const uint32_t kp1 = s_k[buf][t][DSPARK_WARP + lane];
				const float d0 = fmaf(q0[g], LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1[g] * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
				const float d1 = fmaf(q2[g], LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), q3[g] * LmBf16ToFloat((uint16_t)(kp1 >> 16u)));
				const float dot = DsparkWarpSum(d0 + d1);
				if ( lane == 0u )
					s_scores[warp][t] = (s_vis[buf][t] & warp_bit) != 0u ? dot * scale : -INFINITY;
			}
			__syncwarp();
			{
				const float sc = s_scores[warp][lane];
				const float tile_max = DsparkWarpMax(sc);
				const float next_max = fmaxf(running_max[g], tile_max);
				const float rescale = running_max[g] == -INFINITY ? 0.0f : expf(running_max[g] - next_max);
				const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
				const float tile_sum = DsparkWarpSum(weight);
				running_sum[g] = running_sum[g] * rescale + tile_sum;
				running_max[g] = next_max;
				acc0[g] *= rescale;
				acc1[g] *= rescale;
				acc2[g] *= rescale;
				acc3[g] *= rescale;
				s_scores[warp][lane] = weight;
			}
			__syncwarp();
			for ( uint32_t t = 0u; t < G53F_ATTN_TILE; t++ )
			{
				const float wt = s_scores[warp][t];
				const uint32_t vp0 = s_v[buf][t][lane];
				const uint32_t vp1 = s_v[buf][t][DSPARK_WARP + lane];
				acc0[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0[g]);
				acc1[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1[g]);
				acc2[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2[g]);
				acc3[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3[g]);
			}
			__syncwarp();
		}
		__syncthreads();
	}
#pragma unroll
	for ( uint32_t g = 0u; g < G53F_GROUP; g++ )
	{
		const uint32_t head = kv_head * G53F_GROUP + g;
		float *out = partials + (((uint64_t)head * G53F_SPLITS + split) * G53F_BLOCK + warp) * (G53F_HEAD_DIM + 2u);
		out[2u * lane] = acc0[g];
		out[2u * lane + 1u] = acc1[g];
		out[G53F_HALF + 2u * lane] = acc2[g];
		out[G53F_HALF + 2u * lane + 1u] = acc3[g];
		if ( lane == 0u )
		{
			out[G53F_HEAD_DIM] = running_max[g];
			out[G53F_HEAD_DIM + 1u] = running_sum[g];
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void G53FLogitsSummaryKernel(const uint16_t *__restrict__ logits, uint32_t *__restrict__ candidates, float *__restrict__ unary, float *__restrict__ tail)
{
	LmDependentWait();
	__shared__ float s_max[DSPARK_WARPS];
	__shared__ float s_sum[DSPARK_WARPS];
	__shared__ uint32_t s_tok[G53F_TOP_K];
	__shared__ float s_val[G53F_TOP_K];
	const uint32_t step = blockIdx.x, warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint16_t *row = logits + (uint64_t)step * G53F_VOCAB;
	float local_max = -INFINITY, local_sum = 0.0f, row_max, row_sum, total;
	for ( uint32_t index = threadIdx.x; index < G53F_VOCAB; index += DSPARK_THREADS )
		local_max = fmaxf(local_max, LmBf16ToFloat(__ldg(row + index)));
	local_max = DsparkWarpMax(local_max);
	if ( lane == 0u )
		s_max[warp] = local_max;
	__syncthreads();
	if ( warp == 0u )
	{
		float m = lane < DSPARK_WARPS ? s_max[lane] : -INFINITY;
		m = DsparkWarpMax(m);
		if ( lane == 0u )
			s_max[0] = m;
	}
	__syncthreads();
	row_max = s_max[0];
	for ( uint32_t index = threadIdx.x; index < G53F_VOCAB; index += DSPARK_THREADS )
		local_sum += expf(LmBf16ToFloat(__ldg(row + index)) - row_max);
	row_sum = LmBlockSum<DSPARK_THREADS>(local_sum, s_sum);
	total = row_max + logf(row_sum);
	if ( threadIdx.x < G53F_TOP_K )
	{
		const uint32_t tok = candidates[step * G53F_TOP_K + threadIdx.x];
		s_tok[threadIdx.x] = tok;
		s_val[threadIdx.x] = LmBf16ToFloat(__ldg(row + tok));
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
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void G53FLatticeKernel(const G53FDraft *__restrict__ draft, const uint32_t *__restrict__ candidates, const float *__restrict__ unary,
	const uint16_t *__restrict__ sel_hidden, const uint16_t *__restrict__ predecessors, const uint16_t *__restrict__ successors, float *__restrict__ edges)
{
	LmDependentWait();
	__shared__ float gate[G53F_SELECTOR_RANK];
	const uint32_t step = blockIdx.x / G53F_TOP_K, prev_slot = blockIdx.x % G53F_TOP_K;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	if ( step != 0u || prev_slot == 0u )
	{
		const uint32_t prev_token = step == 0u ? draft->anchor : candidates[(step - 1u) * G53F_TOP_K + prev_slot];
		for ( uint32_t r = threadIdx.x; r < G53F_SELECTOR_RANK; r += DSPARK_THREADS )
			gate[r] = LmBf16ToFloat(__ldg(predecessors + (uint64_t)prev_token * G53F_SELECTOR_RANK + r)) *
				LmBf16ToFloat(__ldg(sel_hidden + (uint64_t)step * G53F_SELECTOR_RANK + r));
		__syncthreads();
		for ( uint32_t c = warp; c < G53F_TOP_K; c += DSPARK_WARPS )
		{
			const uint32_t succ_token = candidates[step * G53F_TOP_K + c];
			const uint16_t *succ_row = successors + (uint64_t)succ_token * G53F_SELECTOR_RANK;
			float dot = 0.0f;
			for ( uint32_t r = lane; r < G53F_SELECTOR_RANK; r += DSPARK_WARP )
				dot = fmaf(gate[r], LmBf16ToFloat(__ldg(succ_row + r)), dot);
			dot = DsparkWarpSum(dot);
			if ( lane == 0u )
				edges[(step * G53F_TOP_K + prev_slot) * G53F_TOP_K + c] = unary[step * G53F_TOP_K + c] + dot;
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_WARP, 1)
void G53FWalkKernel(const G53FDraft *__restrict__ draft, const uint32_t *__restrict__ candidates, const float *__restrict__ tail,
	const float *__restrict__ edges, G53FPicks *__restrict__ picks)
{
	LmDependentWait();
	const uint32_t lane = threadIdx.x;
	const uint32_t active = (1u << G53F_TOP_K) - 1u;
	const uint32_t tree = draft->tree;
	uint32_t previous = 0u;
	if ( lane < G53F_TOP_K )
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
				DsparkWarpSampleSubwarp<G53F_TOP_K>(row_val * draft->inverse_temperature, cand_tok, lane, draft->seed, draft->positions[0] + step,
					draft->nucleus, draft->top_k, &choice, (uint32_t *)0, (float *)0);
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
	LmDependentRelease();
}

static SparkStatus G53FShape(G53FState *state, const SparkJsonDocument *document)
{
	const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
	const int32_t dflash = DsparkMember(document, checkpoint, "dflash_config");
	const int32_t rope = DsparkMember(document, checkpoint, "rope_parameters");
	const int32_t layers = DsparkMember(document, dflash, "target_layer_ids");
	const int32_t layer_types = DsparkMember(document, checkpoint, "layer_types");
	const DsparkExpectedU64 expected[] = {
		{checkpoint, "hidden_size", G53F_HIDDEN}, {checkpoint, "intermediate_size", G53F_MLP}, {checkpoint, "num_hidden_layers", G53F_LAYERS},
		{checkpoint, "num_attention_heads", G53F_HEADS}, {checkpoint, "num_key_value_heads", G53F_KV_HEADS}, {checkpoint, "head_dim", G53F_HEAD_DIM},
		{checkpoint, "sliding_window", G53F_SLIDING}, {checkpoint, "vocab_size", G53F_VOCAB}, {dflash, "block_size", G53F_BLOCK},
		{dflash, "conv_group_size", G53F_CONV_GROUP}, {dflash, "conv_kernel_size", G53F_CONV_TAPS}, {dflash, "selector_rank", G53F_SELECTOR_RANK},
		{dflash, "selector_top_k", G53F_TOP_K},
	};
	uint64_t value = 0u;
	float theta = 0.0f;
	bool is_causal = true;
	int32_t token;
	SparkStatus status = DsparkCheckU64Table(document, "glm53 flash dflash2", expected, (uint32_t)(sizeof(expected) / sizeof(expected[0])));
	if ( status != SPARK_STATUS_OK )
		return status;
	token = DsparkMember(document, checkpoint, "is_causal");
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
	token = DsparkMember(document, dflash, "mask_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= G53F_VOCAB )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->mask_token = (uint32_t)value;
	token = DsparkMember(document, checkpoint, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = DsparkMember(document, rope, "rope_type");
	if ( token >= 0 && !SparkJsonStringEquals(document, token, "default") )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = DsparkMember(document, rope, "rope_theta");
	if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK || theta <= 0.0f )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->softmax_scale = (float)(1.0 / sqrt((double)G53F_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, G53F_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	LM_LAUNCH((DsparkRopeTableKernel<G53F_HALF, G53F_HEAD_DIM>), 1u, DSPARK_THREADS, 0, state->stream, state->rope_inverse, theta);
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
	status = DsparkLoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = G53FShape(state, &document);
	if ( status == SPARK_STATUS_OK )
	{
		const DsparkBf16Spec plain[] = {
			{"fc.weight", (uint64_t)G53F_HIDDEN * G53F_TAP_WIDTH, &state->fc},
			{"hidden_norm.weight", G53F_HIDDEN, &state->hidden_norm},
			{"norm.weight", G53F_HIDDEN, &state->norm},
			{"target.embed", (uint64_t)G53F_VOCAB * G53F_HIDDEN, &state->embed},
			{"target.head", (uint64_t)G53F_VOCAB * G53F_HIDDEN, &state->head},
			{"candidate_selector.hidden_projection.weight", (uint64_t)G53F_SELECTOR_RANK * G53F_HIDDEN, &state->selector_hidden},
			{"candidate_selector.predecessor_codebook", (uint64_t)G53F_VOCAB * G53F_SELECTOR_RANK, &state->predecessors},
			{"candidate_selector.successor_codebook", (uint64_t)G53F_VOCAB * G53F_SELECTOR_RANK, &state->successors},
		};
		status = DsparkLoadBf16Table(state->pack, &document, payload_base, "glm53 flash dflash2", plain, (uint32_t)(sizeof(plain) / sizeof(plain[0])));
	}
	for ( uint32_t layer = 0u; layer < G53F_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		G53FLayer *w = &state->layers[layer];
		const DsparkBf16Spec tensors[] = {
			{"input_layernorm.weight", G53F_HIDDEN, &w->input_norm},
			{"post_attention_layernorm.weight", G53F_HIDDEN, &w->post_norm},
			{"attention_conv.base_kernel", (uint64_t)2u * G53F_CONV_TAPS * G53F_HIDDEN, &w->attn_conv_base},
			{"attention_conv.kernel_projection.weight", (uint64_t)G53F_CONV_WIDTH * G53F_HIDDEN, &w->attn_conv_proj},
			{"self_attn.q_proj.weight", (uint64_t)G53F_Q_WIDTH * G53F_HIDDEN, &w->q},
			{"self_attn.k_proj.weight", (uint64_t)G53F_KV_WIDTH * G53F_HIDDEN, &w->k},
			{"self_attn.v_proj.weight", (uint64_t)G53F_KV_WIDTH * G53F_HIDDEN, &w->v},
			{"self_attn.o_proj.weight", (uint64_t)G53F_HIDDEN * G53F_Q_WIDTH, &w->o},
			{"self_attn.q_norm.weight", G53F_HEAD_DIM, &w->q_norm},
			{"self_attn.k_norm.weight", G53F_HEAD_DIM, &w->k_norm},
			{"mlp_conv.base_kernel", (uint64_t)2u * G53F_CONV_TAPS * G53F_HIDDEN, &w->mlp_conv_base},
			{"mlp_conv.kernel_projection.weight", (uint64_t)G53F_CONV_WIDTH * G53F_HIDDEN, &w->mlp_conv_proj},
			{"mlp.gate_proj.weight", (uint64_t)G53F_MLP * G53F_HIDDEN, &w->gate},
			{"mlp.up_proj.weight", (uint64_t)G53F_MLP * G53F_HIDDEN, &w->up},
			{"mlp.down_proj.weight", (uint64_t)G53F_HIDDEN * G53F_MLP, &w->down},
		};
		status = DsparkLoadPrefixedBf16Table(state->pack, &document, payload_base, "glm53 flash dflash2", "layers.%u.%s", layer, tensors,
			(uint32_t)(sizeof(tensors) / sizeof(tensors[0])));
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

static SparkStatus G53FScratch(G53FState *state, uint64_t context_bytes)
{
	G53FForward *f = &state->forward;
	int multiprocessors = 0;
	if ( !state->ring.Init(context_bytes, G53F_SEQUENCE_ELEMENTS, &state->caches) ||
		SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		!DsparkAllocHost(&state->host_taps, (size_t)G53F_INGEST_ROWS * G53F_TAP_WIDTH) ||
		!DsparkAllocHost(&state->host_positions, G53F_INGEST_ROWS) || !DsparkAllocHost(&f->host_draft, 1u) ||
		!DsparkAllocHost(&f->host_picks, 1u) ||
		!DSPARK_DEVICE(state->taps, (uint64_t)G53F_INGEST_ROWS * G53F_TAP_WIDTH) || !DSPARK_DEVICE(state->positions, G53F_INGEST_ROWS) ||
		!DSPARK_DEVICE(state->features_raw, (uint64_t)G53F_INGEST_ROWS * G53F_HIDDEN) ||
		!DSPARK_DEVICE(state->features, (uint64_t)G53F_INGEST_ROWS * G53F_HIDDEN) ||
		!DSPARK_DEVICE(state->keys_raw, (uint64_t)G53F_INGEST_ROWS * G53F_KV_WIDTH) ||
		!DSPARK_DEVICE(state->values_raw, (uint64_t)G53F_INGEST_ROWS * G53F_KV_WIDTH) || !DSPARK_DEVICE(f->draft, 1u) ||
		!DSPARK_DEVICE(f->picks, 1u) || !DSPARK_DEVICE(f->hidden, G53F_BLOCK * G53F_HIDDEN) || !DSPARK_DEVICE(f->x, G53F_BLOCK * G53F_HIDDEN) ||
		!DSPARK_DEVICE(f->conv_coeff, G53F_BLOCK * G53F_CONV_WIDTH) || !DSPARK_DEVICE(f->prepared, G53F_BLOCK * G53F_HIDDEN) ||
		!DSPARK_DEVICE(f->q, G53F_BLOCK * G53F_Q_WIDTH) || !DSPARK_DEVICE(f->q_roped, G53F_BLOCK * G53F_Q_WIDTH) ||
		!DSPARK_DEVICE(f->k, G53F_BLOCK * G53F_KV_WIDTH) || !DSPARK_DEVICE(f->v, G53F_BLOCK * G53F_KV_WIDTH) ||
		!DSPARK_DEVICE(f->partials, (uint64_t)G53F_HEADS * G53F_SPLITS * G53F_BLOCK * (G53F_HEAD_DIM + 2u)) ||
		!DSPARK_DEVICE(f->attended, G53F_BLOCK * G53F_Q_WIDTH) || !DSPARK_DEVICE(f->projected, G53F_BLOCK * G53F_HIDDEN) ||
		!DSPARK_DEVICE(f->gate, G53F_BLOCK * G53F_MLP) || !DSPARK_DEVICE(f->up, G53F_BLOCK * G53F_MLP) ||
		!DSPARK_DEVICE(f->act, G53F_BLOCK * G53F_MLP) || !DSPARK_DEVICE(f->mlp, G53F_BLOCK * G53F_HIDDEN) ||
		!DSPARK_DEVICE(f->y, G53F_BLOCK * G53F_HIDDEN) || !DSPARK_DEVICE(f->logits, (uint64_t)G53F_DRAFTS * G53F_VOCAB) ||
		SparkDrafterTopkBf16Alloc(&f->topk_ws, G53F_DRAFTS) != SPARK_STATUS_OK ||
		!DSPARK_DEVICE(f->sel_hidden, G53F_DRAFTS * G53F_SELECTOR_RANK) || !DSPARK_DEVICE(f->candidates, G53F_DRAFTS * G53F_TOP_K) ||
		!DSPARK_DEVICE(f->unary, G53F_DRAFTS * G53F_TOP_K) || !DSPARK_DEVICE(f->tail, G53F_DRAFTS) ||
		!DSPARK_DEVICE(f->edges, G53F_DRAFTS * G53F_TOP_K * G53F_TOP_K) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
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

static SparkStatus G53FIngestRows(G53FState *state, G53FSequence *sequence, uint32_t rows)
{
	SparkStatus status = DsparkGemmF32(state->fc, state->taps, state->features_raw, rows, G53F_TAP_WIDTH, G53F_HIDDEN, state->multiprocessors,
		state->stream);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((DsparkRmsKernel<false, false, float>), rows, DSPARK_THREADS, 0, state->stream, state->features_raw, state->hidden_norm,
			state->features, G53F_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < G53F_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = DsparkGemm(state->layers[layer].k, state->features, state->keys_raw, rows, G53F_HIDDEN, G53F_KV_WIDTH, state->multiprocessors,
			state->stream);
		if ( status == SPARK_STATUS_OK )
			status = DsparkGemm(state->layers[layer].v, state->features, state->values_raw, rows, G53F_HIDDEN, G53F_KV_WIDTH, state->multiprocessors,
				state->stream);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((DsparkHeadNormRopeRingKernel<G53F_HEAD_DIM, G53F_HEAD_DIM, G53F_WINDOW, G53F_COLUMNS, DSPARK_NORM_F32, false, G53FDraft>),
			rows * G53F_KV_HEADS, G53F_HEAD_DIM, 0, state->stream, state->keys_raw, state->layers[layer].k_norm, state->rope_inverse, 1.0f,
			state->positions, G53F_KV_HEADS, state->epsilon, (uint16_t *)0, G53FKeys(sequence, layer), (const G53FDraft *)0, 0u, 1u);
		LM_LAUNCH((DsparkStoreValuesRingKernel<G53F_KV_WIDTH, G53F_HEAD_DIM, G53F_WINDOW, G53F_COLUMNS, G53FDraft>), rows, DSPARK_THREADS, 0,
			state->stream, state->values_raw, state->positions, (const G53FDraft *)0, 0u, G53FValues(sequence, layer));
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	return DsparkSettle(state->stream, status);
}

static SparkStatus G53FIngest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	G53FState *state = (G53FState *)opaque;
	return state->ring.IngestFrame<G53F_TAPS, G53F_HIDDEN, G53F_INGEST_ROWS>(frame, state->host_taps, state->host_positions, state->taps,
		state->positions, state->stream, &state->frames, &state->rows,
		[&](G53FSequence *sequence, uint32_t part) { return G53FIngestRows(state, sequence, part); });
}

static SparkStatus G53FPrepare(G53FState *state, G53FSequence *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
	const SparkRowSampling *sampling, uint32_t tree, uint32_t *splits_out)
{
	G53FDraft *draft = state->forward.host_draft;
	uint32_t span, splits;
	if ( anchor >= G53F_VOCAB || position + G53F_BLOCK > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(draft, 0, offsetof(G53FDraft, valid));
	draft->cache = sequence->cache;
	for ( uint32_t row = 0u; row < G53F_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	draft->anchor = anchor;
	draft->mask = state->mask_token;
	draft->greedy = sampling->inverse_temperature > 0.0f ? 0u : 1u;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	draft->tree = tree;
	G53FRingPool::Span<G53F_BLOCK, G53F_SLIDING, true>(sequence, checked, position, draft);
	span = draft->block_at - draft->start;
	splits = (span + G53F_SPLIT_KEYS - 1u) / G53F_SPLIT_KEYS;
	draft->splits = splits;
	if ( splits_out != 0 )
		*splits_out = splits;
	if ( SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FRunForward(G53FState *state, uint32_t layers, uint32_t splits)
{
	G53FForward *f = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)f->draft + offsetof(G53FDraft, positions));
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((DsparkEmbedF32Kernel<G53F_HIDDEN, G53FDraft>), G53F_BLOCK, DSPARK_THREADS, 0, stream, state->embed, f->draft, f->hidden);
	LM_LAUNCH((DsparkRmsKernel<false, false, float>), G53F_BLOCK, DSPARK_THREADS, 0, stream, f->hidden, state->layers[0].input_norm, f->x,
		G53F_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const G53FLayer *w = &state->layers[layer];
		const uint64_t keys_offset = (uint64_t)(2u * layer) * G53F_LAYER_KV;
		const uint64_t values_offset = (uint64_t)(2u * layer + 1u) * G53F_LAYER_KV;
		status = DsparkGemm(w->attn_conv_proj, f->x, f->conv_coeff, G53F_BLOCK, G53F_HIDDEN, G53F_CONV_WIDTH, state->multiprocessors, stream);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FConvPrepareKernel), G53F_BLOCK, DSPARK_THREADS, 0, stream, f->x, w->attn_conv_base, f->conv_coeff, f->prepared);
		status = DsparkGemm(w->q, f->prepared, f->q, G53F_BLOCK, G53F_HIDDEN, G53F_Q_WIDTH, state->multiprocessors, stream);
		if ( status == SPARK_STATUS_OK )
			status = DsparkGemm(w->k, f->prepared, f->k, G53F_BLOCK, G53F_HIDDEN, G53F_KV_WIDTH, state->multiprocessors, stream);
		if ( status == SPARK_STATUS_OK )
			status = DsparkGemm(w->v, f->prepared, f->v, G53F_BLOCK, G53F_HIDDEN, G53F_KV_WIDTH, state->multiprocessors, stream);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((DsparkHeadNormRopeRingKernel<G53F_HEAD_DIM, G53F_HEAD_DIM, G53F_WINDOW, G53F_COLUMNS, DSPARK_NORM_F32, false, G53FDraft>),
			G53F_BLOCK * G53F_HEADS, G53F_HEAD_DIM, 0, stream, f->q, w->q_norm, state->rope_inverse, 1.0f, positions, G53F_HEADS, state->epsilon,
			f->q_roped, (uint16_t *)0, f->draft, 0u, 0u);
		if ( LM_LAUNCH_DEPENDENT((DsparkHeadNormRopeRingKernel<G53F_HEAD_DIM, G53F_HEAD_DIM, G53F_WINDOW, G53F_COLUMNS, DSPARK_NORM_F32, false,
					G53FDraft>),
				G53F_BLOCK * G53F_KV_HEADS, G53F_HEAD_DIM, 0, stream, f->k, w->k_norm, state->rope_inverse, 1.0f, positions, G53F_KV_HEADS,
				state->epsilon, (uint16_t *)0, (uint16_t *)0, f->draft, keys_offset, 0u) != SPARK_GPU_SUCCESS ||
			LM_LAUNCH_DEPENDENT((DsparkStoreValuesRingKernel<G53F_KV_WIDTH, G53F_HEAD_DIM, G53F_WINDOW, G53F_COLUMNS, G53FDraft>), G53F_BLOCK,
				DSPARK_THREADS, 0, stream, f->v, (const uint32_t *)0, f->draft, values_offset, (uint16_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( splits != 0u &&
			LM_LAUNCH_DEPENDENT((G53FAttentionSplitKernel), dim3(G53F_KV_HEADS, splits), G53F_ATTN_THREADS, 0, stream, f->q_roped, keys_offset,
				values_offset, f->draft, state->softmax_scale, f->partials) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( LM_LAUNCH_DEPENDENT((DsparkAttentionMergeKernel<G53F_HEADS, G53F_KV_HEADS, G53F_BLOCK, G53F_COLUMNS, G53F_HEAD_DIM, G53F_SPLITS, false,
					false, false, G53FDraft>),
				G53F_BLOCK * G53F_HEADS, G53F_HEAD_DIM, 0, stream, f->q_roped, (const uint16_t *)0, (const uint16_t *)0, keys_offset, values_offset,
				f->draft, state->softmax_scale, f->partials, f->attended) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = DsparkGemm(w->o, f->attended, f->projected, G53F_BLOCK, G53F_Q_WIDTH, G53F_HIDDEN, state->multiprocessors, stream);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FConvFinishKernel<true>), G53F_BLOCK, DSPARK_THREADS, 0, stream, f->projected, w->attn_conv_base, f->conv_coeff, f->hidden,
			w->post_norm, f->x, state->epsilon);
		status = DsparkGemm(w->mlp_conv_proj, f->x, f->conv_coeff, G53F_BLOCK, G53F_HIDDEN, G53F_CONV_WIDTH, state->multiprocessors, stream);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((G53FConvPrepareKernel), G53F_BLOCK, DSPARK_THREADS, 0, stream, f->x, w->mlp_conv_base, f->conv_coeff, f->prepared);
		status = DsparkGemm(w->gate, f->prepared, f->gate, G53F_BLOCK, G53F_HIDDEN, G53F_MLP, state->multiprocessors, stream);
		if ( status == SPARK_STATUS_OK )
			status = DsparkGemm(w->up, f->prepared, f->up, G53F_BLOCK, G53F_HIDDEN, G53F_MLP, state->multiprocessors, stream);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((DsparkSwigluKernel), 2u * state->multiprocessors, DSPARK_THREADS, 0, stream, f->gate, f->up, f->act, G53F_BLOCK * G53F_MLP);
		status = DsparkGemm(w->down, f->act, f->mlp, G53F_BLOCK, G53F_MLP, G53F_HIDDEN, state->multiprocessors, stream);
		if ( status != SPARK_STATUS_OK )
			break;
		if ( layer + 1u < layers )
			LM_LAUNCH((G53FConvFinishKernel<true>), G53F_BLOCK, DSPARK_THREADS, 0, stream, f->mlp, w->mlp_conv_base, f->conv_coeff, f->hidden,
				state->layers[layer + 1u].input_norm, f->x, state->epsilon);
		else if ( layers == G53F_LAYERS )
			LM_LAUNCH((G53FConvFinishKernel<true>), G53F_BLOCK, DSPARK_THREADS, 0, stream, f->mlp, w->mlp_conv_base, f->conv_coeff, f->hidden,
				state->norm, f->y, state->epsilon);
		else
			LM_LAUNCH((G53FConvFinishKernel<false>), G53F_BLOCK, DSPARK_THREADS, 0, stream, f->mlp, w->mlp_conv_base, f->conv_coeff, f->hidden,
				(const uint16_t *)0, (uint16_t *)0, 0.0f);
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FHeads(G53FState *state)
{
	G53FForward *f = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint16_t *rows = f->y + G53F_HIDDEN;
	SparkStatus status = DsparkGemm(state->head, rows, f->logits, G53F_DRAFTS, G53F_HIDDEN, G53F_VOCAB, state->multiprocessors, stream);
	if ( status == SPARK_STATUS_OK )
		status = DsparkGemm(state->selector_hidden, rows, f->sel_hidden, G53F_DRAFTS, G53F_HIDDEN, G53F_SELECTOR_RANK, state->multiprocessors,
			stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkDrafterTopkBf16Launch(f->logits, G53F_DRAFTS, G53F_VOCAB, G53F_TOP_K, &f->topk_ws, f->candidates, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( LM_LAUNCH_DEPENDENT((G53FLogitsSummaryKernel), G53F_DRAFTS, DSPARK_THREADS, 0, stream, f->logits, f->candidates, f->unary, f->tail) !=
			SPARK_GPU_SUCCESS ||
		LM_LAUNCH_DEPENDENT((G53FLatticeKernel), G53F_DRAFTS * G53F_TOP_K, DSPARK_THREADS, 0, stream, f->draft, f->candidates, f->unary,
			f->sel_hidden, state->predecessors, state->successors, f->edges) != SPARK_GPU_SUCCESS ||
		LM_LAUNCH_DEPENDENT((G53FWalkKernel), 1u, DSPARK_WARP, 0, stream, f->draft, f->candidates, f->tail, f->edges, f->picks) !=
			SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( SparkGpuMemcpyAsync(f->host_picks, f->picks, sizeof(G53FPicks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus G53FClose(void *opaque)
{
	G53FState *state = (G53FState *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	SparkDrafterGraphCacheDestroy(&state->graphs);
	{
		G53FForward *f = &state->forward;
		void *buffers[] = {
			f->draft, f->picks, f->hidden, f->x, f->conv_coeff, f->prepared, f->q, f->q_roped, f->k, f->v, f->partials, f->attended, f->projected,
			f->gate, f->up, f->act, f->mlp, f->y, f->logits, f->sel_hidden, f->candidates, f->unary, f->tail, f->edges, state->caches, state->taps,
			state->positions, state->features_raw, state->features, state->keys_raw, state->values_raw, state->rope_inverse,
		};
		void *host_buffers[] = { f->host_draft, f->host_picks, state->host_taps, state->host_positions };
		DsparkFreeDeviceBuffers(buffers, (uint32_t)(sizeof(buffers) / sizeof(buffers[0])));
		SparkDrafterTopkBf16Free(&f->topk_ws);
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

static SparkStatus G53FOpen(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	return DsparkDrafterOpen<G53FState>(lane, G53F_SEQUENCE_BYTES, "lane %s has a context cap below one sequence's window",
		"lane %s: the pack %s is not a GLM-5.3-Flash DFlash2 pack this module reads", true, state_out, bind, G53FClose,
		[&](G53FState *state) -> SparkStatus {
			SparkStatus status = G53FWeights(state, lane->pack_path);
			if ( status == SPARK_STATUS_OK )
				status = G53FScratch(state, lane->context_bytes);
			if ( status == SPARK_STATUS_OK )
				status = DsparkSettle(state->stream, SPARK_STATUS_OK);
			return status;
		},
		[](const G53FState *state) { return state->ring.capacity; });
}

static SparkStatus G53FPropose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	G53FState *state = (G53FState *)opaque;
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > G53F_SLIDING - 1u ? request->position - (G53F_SLIDING - 1u) : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested, splits = 0u;
	G53FSequence *sequence = 0;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = state->ring.PrepareProposeContext(request, first, state->stream, &sequence, &checked,
		[&](G53FSequence *dst, const G53FSequence *src, uint32_t slot, uint32_t n) {
			return G53FRingPool::CopyKvRange<G53F_LAYERS, G53F_KV_HEADS, G53F_COLUMNS, G53F_HEAD_DIM>(dst, src, slot, n, state->stream);
		});
	if ( status != SPARK_STATUS_OK )
		return status;
	drafts = drafts < G53F_DRAFTS ? drafts : G53F_DRAFTS;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		status = G53FPrepare(state, sequence, request->anchor_token_id, request->position, checked, &request->sampling, tree, &splits);
		if ( status == SPARK_STATUS_OK )
			status = SparkDrafterGraphRun(&state->graphs, state->stream, (uint64_t)splits, [&]() -> SparkStatus {
				SparkStatus inner = G53FRunForward(state, G53F_LAYERS, splits);
				if ( inner == SPARK_STATUS_OK )
					inner = G53FHeads(state);
				return inner;
			});
		status = DsparkSettle(state->stream, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		*count = DsparkPackGridTree<G53F_DRAFTS, G53F_BRANCH_LEN>(request->anchor_token_id, source, state->forward.host_picks, tree, drafts,
			requested, limit, nodes);
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

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &g53f_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_flash_dflash2_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	G53FState *state = (G53FState *)opaque;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	return state->ring.CacheReadback<G53F_LAYERS, G53F_LAYER_KV>(sequence_id, layer, keys, values, positions);
}

static SparkStatus G53FRunDiag(G53FState *state, uint64_t sequence_id, uint32_t anchor, uint64_t position, uint32_t layers, SparkStatus missing)
{
	G53FSequence *sequence = state != 0 ? state->ring.Live(sequence_id) : 0;
	const uint64_t first = position > G53F_SLIDING - 1u ? position - (G53F_SLIDING - 1u) : 0u;
	uint32_t splits = 0u;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		SPARK_FAIL(missing);
	memset(&greedy, 0, sizeof(greedy));
	status = G53FPrepare(state, sequence, anchor, position, G53FRingPool::CheckedFrom(sequence, first, position), &greedy, 0u, &splits);
	if ( status == SPARK_STATUS_OK )
		status = G53FRunForward(state, layers, splits);
	return DsparkSettle(state->stream, status);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_flash_dflash2_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	G53FState *state = (G53FState *)opaque;
	const uint64_t started = SparkMonotonicNs();
	SparkStatus status = G53FRunDiag(state, sequence_id, anchor, position, G53F_LAYERS, SPARK_STATUS_NOT_FOUND);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = DsparkReadback(y, state->forward.y, (size_t)G53F_BLOCK * G53F_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_flash_dflash2_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, float *hidden)
{
	G53FState *state = (G53FState *)opaque;
	SparkStatus status;
	if ( state == 0 || hidden == 0 || layers == 0u || layers > G53F_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = G53FRunDiag(state, sequence_id, anchor, position, layers, SPARK_STATUS_INVALID_ARGUMENT);
	if ( status == SPARK_STATUS_OK )
		status = DsparkReadback(hidden, state->forward.hidden, (size_t)G53F_BLOCK * G53F_HIDDEN * sizeof(float));
	return status;
}
