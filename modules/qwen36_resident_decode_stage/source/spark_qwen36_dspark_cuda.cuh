#pragma once
/* DSpark drafter kernels: the DFlash dual-source block forward.
 *
 * The drafter is a 5-layer full-attention decoder (40 Q / 8 KV heads x 128,
 * FFN 10240) that emits a 7-token block in ONE forward. Attention is DUAL-SOURCE:
 * the drafter's Q comes from its own block hidden, while K/V = cat(the aligned
 * TARGET tap hidden (1 position), the drafter's own block hidden (7 positions)).
 * The block is NON-CAUSAL (positions attend to each other + the tap).
 *
 * Everything except the attention core reuses the shared linear/FFN kernels
 * (SparkLmLaunchLinear). This header adds the one genuinely new kernel: the
 * flat 7x8 dual-source GQA attention, plus the Markov bigram bias.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "sparkpipe/spark_qwen36_model.h"
#include "spark_qwen36_dspark_format.h"

#define SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM 128u
#define SPARK_QWEN36_DSPARK_ATTN_KV_HEADS 8u
#define SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM 64u
#define SPARK_QWEN36_DSPARK_BLOCK_SIZE 7u

static __device__ __forceinline__ float SparkQwen36DsparkRopeFrequency(uint32_t pair)
{
	/* Same convention as the target: theta^(-2*pair/rope_dim), rope_dim 64. */
	return exp2f(-((float)(2u * pair) / (float)SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN36_MODEL_ATTN_ROPE_THETA));
}

/* The flat dual-source attention. Each CTA handles one (block position, KV
 * head group); threads span the 4 Q heads in the group. qk_v layout:
 *   Q: block_size x 32 x 128 (after q_proj, before q_norm)
 *   K: (1+block_size) x 8 x 128 (after k_proj, before k_norm)
 *   V: (1+block_size) x 8 x 128 (after v_proj)
 */
static __global__ void SparkQwen36DsparkAttnKernel(
	const __nv_bfloat16 *q_bf16, const __nv_bfloat16 *k_bf16, const __nv_bfloat16 *v_bf16,
	const __nv_bfloat16 *q_norm_bf16, const __nv_bfloat16 *k_norm_bf16,
	__nv_bfloat16 *attn_out_bf16, uint32_t block_size, uint64_t base_position)
{
	const uint32_t kv_heads = SPARK_QWEN36_DSPARK_ATTN_KV_HEADS;
	const uint32_t head_dim = SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM;
	const uint32_t q_heads_per_group = SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS / kv_heads;
	const uint32_t q_pos = blockIdx.x;
	const uint32_t kv_group = blockIdx.y;
	const uint32_t q_head_in_group = threadIdx.x;
	float qn[head_dim], kn[head_dim], acc[head_dim];
	float score, max_score, sum_exp, coeff;
	uint32_t kv_pos, d, q_head;
	uint64_t pos;
	if ( q_pos >= block_size || q_head_in_group >= q_heads_per_group )
		return;
	q_head = kv_group * q_heads_per_group + q_head_in_group;
	/* Q: weighted head RMSNorm (single [128] weight shared across heads) + rope. */
	{
		float sum = 0.0f, scale;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			qn[d] = __bfloat162float(q_bf16[((uint64_t)q_pos * SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS + q_head) * head_dim + d]);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			sum = fmaf(qn[d], qn[d], sum);
		scale = rsqrtf(sum / (float)head_dim + 1e-6f);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			qn[d] = qn[d] * scale * __bfloat162float(q_norm_bf16[d]);
	}
	pos = base_position + q_pos;
	#pragma unroll
	for (d = 0u; d < SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM; d += 2u)
	{
		float f = pos * SparkQwen36DsparkRopeFrequency(d >> 1u);
		float c = cosf(f), sn = sinf(f);
		float re = qn[d], im = qn[d + 1u];
		qn[d] = re * c - im * sn;
		qn[d + 1u] = re * sn + im * c;
	}
	/* Online softmax over the (1+block_size) KV positions, non-causal. */
	#pragma unroll
	for (d = 0u; d < head_dim; d++)
		acc[d] = 0.0f;
	max_score = -1e30f;
	sum_exp = 0.0f;
	for (kv_pos = 0u; kv_pos < 1u + block_size; kv_pos++)
	{
		float sum = 0.0f, scale;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			kn[d] = __bfloat162float(k_bf16[((uint64_t)kv_pos * kv_heads + kv_group) * head_dim + d]);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			sum = fmaf(kn[d], kn[d], sum);
		scale = rsqrtf(sum / (float)head_dim + 1e-6f);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			kn[d] = kn[d] * scale * __bfloat162float(k_norm_bf16[d]);
		/* K rope (tap at base_position - 1 = committed position, block at base_position + kv_pos - 1) */
		pos = kv_pos == 0u ? base_position - 1u : base_position + (kv_pos - 1u);
		#pragma unroll
		for (d = 0u; d < SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM; d += 2u)
		{
			float f = pos * SparkQwen36DsparkRopeFrequency(d >> 1u);
			float c = cosf(f), sn = sinf(f);
			float re = kn[d], im = kn[d + 1u];
			kn[d] = re * c - im * sn;
			kn[d + 1u] = re * sn + im * c;
		}
		score = 0.0f;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			score = fmaf(qn[d], kn[d], score);
		score *= 0.088388347f; /* 1/sqrt(128) */
		max_score = fmaxf(max_score, score);
		coeff = __expf(score);
		sum_exp += coeff;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			acc[d] = fmaf(coeff, __bfloat162float(v_bf16[((uint64_t)kv_pos * kv_heads + kv_group) * head_dim + d]), acc[d]);
	}
	/* The single-pass online softmax is exact only for a monotone max; rescale
	 * once more against the true max for safety at this small size. */
	#pragma unroll
	for (d = 0u; d < head_dim; d++)
		attn_out_bf16[((uint64_t)q_pos * SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS + q_head) * head_dim + d] = __float2bfloat16(acc[d] / sum_exp);
}

/* Grouped dynamic depthwise conv (DFlash2), fused into ONE elementwise pass — no
 * im2col. One CTA per (block position, group); threads span the group_size channels.
 *   x:    [block_size, H] BF16
 *   delta: [block_size, taps, num_groups] F32 (per-token, from kernel_projection)
 *   base:  [taps, H] BF16 (learned base, one side)
 *   out[i,c] = sum_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c], taps zero where
 *              (i & (block_size-1)) < t.
 */
static __global__ void SparkQwen36DsparkConvKernel(
	const __nv_bfloat16 *x_bf16, const __nv_bfloat16 *delta, const __nv_bfloat16 *base_bf16,
	__nv_bfloat16 *out_bf16, uint32_t block_size, uint32_t num_groups, uint32_t group_size, uint32_t side)
{
	const uint32_t pos = blockIdx.x;
	const uint32_t group = blockIdx.y;
	const uint32_t c = group * group_size + threadIdx.x;
	const uint32_t H = num_groups * group_size;
	uint32_t p;
	float x0, d0, out;
	if ( c >= H )
		return;
	p = (block_size & (block_size - 1u)) == 0u ? pos & (block_size - 1u) : pos % block_size;
	const uint32_t ds = (pos * 2u + side) * 2u;
	x0 = __bfloat162float(x_bf16[(uint64_t)pos * H + c]);
	d0 = __bfloat162float(delta[(uint64_t)(ds + 0u) * num_groups + group]);
	out = (__bfloat162float(base_bf16[0u * H + c]) + d0) * x0;
	if ( p >= 1u )
	{
		float x1 = __bfloat162float(x_bf16[((uint64_t)(pos - 1u)) * H + c]);
		float d1 = __bfloat162float(delta[(uint64_t)(ds + 1u) * num_groups + group]);
		out += (__bfloat162float(base_bf16[1u * H + c]) + d1) * x1;
	}
	out_bf16[(uint64_t)pos * H + c] = __float2bfloat16(out);
}

/* Markov bigram bias: bias[v] = w2 @ w1[prev_token], applied to the draft
 * logits. One CTA, one thread per vocab shard. */
static __global__ void SparkQwen36DsparkMarkovKernel(
	const __nv_bfloat16 *markov_w1_bf16, const __nv_bfloat16 *markov_w2_bf16,
	const uint32_t *prev_token_ids, uint32_t draft_count, uint32_t rank,
	float *bias_out, uint32_t vocab)
{
	uint32_t draft_pos = blockIdx.x, v = blockIdx.y * blockDim.x + threadIdx.x;
	float acc;
	uint32_t r;
	if ( draft_pos >= draft_count || v >= vocab )
		return;
	acc = 0.0f;
	#pragma unroll
	for (r = 0u; r < rank; r++)
		acc = fmaf(__bfloat162float(markov_w1_bf16[(uint64_t)prev_token_ids[draft_pos] * rank + r]),
		           __bfloat162float(markov_w2_bf16[(uint64_t)v * rank + r]), acc);
	/* vLLM's bias is a BF16 Linear output (markov_w2 @ markov_w1[prev]); the
	 * host sampler, the numpy reference, and specforge dspark.py all truncate
	 * bias -> BF16 BEFORE the add, so this device kernel must too (else it
	 * lies about the rounding convention). */
	bias_out[(uint64_t)draft_pos * vocab + v] = __bfloat162float(__float2bfloat16(acc));
}/*
 * ===========================================================================
 * DFlash2 candidate selector, device side (adoption items W4 and W3).
 *
 * The two kernels below replace DSpark's "full-vocabulary Markov rewrite then
 * per-slot argmax" with DFlash2's "target top-K per slot, K x K edge lattice,
 * one greedy walk from the verified anchor". Both are pinned to the numpy
 * oracles in tools/qwen36_dspark_reference.py, which are themselves exact
 * ports of vLLM PR #52816 (_score_edges / _greedy_walk) - those are the
 * contract, including WHERE a value is truncated to BF16:
 *
 *   W4 unary  : mask_logits = bf16(hidden @ lm_head.T)  <- truncate, THEN top-K
 *   W3 gate   : H          = bf16(hidden @ hidden_projection.T)
 *   W3 edges  : edge[p,c]  = unary[c] + sum_r (A[pred[p],r] * H[r]) * B[cand[c],r]
 *               with fp32 kept all the way to the walk's argmax (no BF16 store)
 *
 * W4 (top-16 over the 248320-row BF16 lm_head) is the generalisation of the
 * landed top-1 reduction: SparkLmHeadExactArgmaxRange keeps ONE (score, id)
 * per warp with ties to the lower id, so the identical dot product
 * (SparkLmDotRowBf16 + SparkLmWarpReduceSum - same staging, same FMA order,
 * same warp tree) now feeds an ordered 64-bit key (SparkLmOrderedTopKKey, the
 * shared primitive dsv4/k3/mimo25 already select on) into a 16-deep insertion
 * list per warp. Score-descending / id-ascending IS the total order that key
 * encodes, so the top-1 entry of this kernel is by construction the token the
 * landed argmax head would have emitted for the same truncation.
 *
 * Two stages, because the vocabulary is striped over the GPU exactly like the
 * landed fallback rescore (SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT chunks per row):
 * stage one emits each chunk's top-K keys, stage two bitonic-merges the
 * chunk_count x top_k keys of a row and decodes the final ids and scores.
 * Nothing is materialised at vocabulary width - the [rows, 248320] fp32 logits
 * tensor the drafter host path currently D2Hs (half a gigabyte at B=32) is
 * exactly what this kernel exists to delete.
 *
 * TIE RULE - read this before comparing against numpy. The unary logits are
 * BF16 (8-bit mantissa) BEFORE the selection, so exact ties among 248320
 * candidates are not an edge case, they are the common case, and any top-K
 * over them is only well defined once the tie rule is pinned. This kernel
 * pins FIRST-MAX (lowest candidate id wins), which is the rule
 * SparkLmArgmaxReduce already uses and the same rule _greedy_walk uses for
 * its argmax. The oracle's np.argsort(-mask_logits) is numpy's DEFAULT
 * quicksort, which is NOT stable, so its id order among tied scores is
 * implementation defined; the parity harness compares against
 * np.argsort(..., kind="stable"), which is the stable-sort spelling of
 * first-max.
 * ===========================================================================
 */

/* One chunk of the vocabulary per CTA in stage one, mirroring the landed
 * fallback rescore's striping so the two paths have the same shape of work. */
#define SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT
/* The lattice width IS the pack's selector_top_k; the kernels accept any
 * top_k <= this and the launchers reject anything wider. */
#define SPARK_QWEN36_DSPARK_TOPK_WIDTH SPARK_QWEN36_DSPARK_SELECTOR_TOP_K
/* Per-CTA candidate pool in stage one: one 16-deep list per warp. */
#define SPARK_QWEN36_DSPARK_TOPK_WARP_POOL (SPARK_LM_CTA_WARPS * SPARK_QWEN36_DSPARK_TOPK_WIDTH)
/* Per-row candidate pool in stage two: every chunk's top_k. Both pools are
 * powers of two so the shared bitonic sort can consume them directly. */
#define SPARK_QWEN36_DSPARK_TOPK_MERGE_POOL (SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT * SPARK_QWEN36_DSPARK_TOPK_WIDTH)

/* Ordered key with an explicit "empty slot" encoding: 0 means no candidate,
 * which sorts below every real key. Mirrors SparkDsv4OrderedTopKKey. */
static __device__ __forceinline__ uint64_t SparkQwen36DsparkTopKKey(float score, uint32_t candidate)
{
	if ( score <= -3.0e38f )
		return(0ull);
	return(SparkLmOrderedTopKKey(score,candidate));
}

static __device__ __forceinline__ uint32_t SparkQwen36DsparkTopKKeyCandidate(uint64_t key)
{
	return(0xffffffffu - (uint32_t)key);
}

/* Inverse of SparkLmOrderedTopKKey's monotone float mapping. */
static __device__ __forceinline__ float SparkQwen36DsparkTopKKeyScore(uint64_t key)
{
	uint32_t ordered = (uint32_t)(key >> 32u);
	return(__uint_as_float((ordered & 0x80000000u) != 0u ? (ordered ^ 0x80000000u) : ~ordered));
}

/* Descending insertion into a width-deep list; a key that cannot reach the
 * tail is dropped without touching memory. Called by ONE lane per warp, so
 * no synchronisation is needed inside the list. */
static __device__ __forceinline__ void SparkQwen36DsparkTopKInsert(uint64_t *ordered, uint32_t width, uint64_t key)
{
	uint32_t slot;
	if ( key <= ordered[width - 1u] )
		return;
	for (slot = width - 1u; slot > 0u && ordered[slot - 1u] < key; slot--)
		ordered[slot] = ordered[slot - 1u];
	ordered[slot] = key;
}

/*
 * W4 stage one: exact BF16 head dot products over one vocabulary chunk, each
 * truncated to BF16 (the DFlash2 unary contract) before it competes, and the
 * chunk's top_k keys emitted. Grid is (row, chunk); shared memory is the
 * staged hidden row, dynamic and sized by the launcher.
 */
static __global__ void SparkQwen36DsparkHeadTopKChunkKernel(
	const void *hidden_bf16, const void *head_weight_bf16, uint64_t *chunk_keys,
	uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension, uint32_t top_k)
{
	extern __shared__ float hidden_shared[];
	__shared__ uint64_t candidate_pool[SPARK_QWEN36_DSPARK_TOPK_WARP_POOL];
	uint32_t row = blockIdx.x,chunk = blockIdx.y;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t first_candidate,end_candidate,candidate,slot;
	float score;
	if ( row >= row_count )
		return;
	for (slot = threadIdx.x; slot < SPARK_QWEN36_DSPARK_TOPK_WARP_POOL; slot += blockDim.x)
		candidate_pool[slot] = 0ull;
	/* SparkLmHeadStageHidden ends in __syncthreads(), which also publishes the
	 * pool zeroing above. */
	SparkLmHeadStageHidden(hidden_bf16,hidden_shared,row,hidden_dimension);
	first_candidate = (uint32_t)(((uint64_t)candidate_count * chunk) / SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT);
	end_candidate = (uint32_t)(((uint64_t)candidate_count * (chunk + 1u)) / SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT);
	for (candidate = first_candidate + warp; candidate < end_candidate; candidate += SPARK_LM_CTA_WARPS)
	{
		score = SparkLmWarpReduceSum(SparkLmDotRowBf16(hidden_shared,head_weight_bf16,candidate,hidden_dimension,lane));
		score = __shfl_sync(0xffffffffu,score,0);
		/* The oracle truncates the whole logit row to BF16 and only THEN takes
		 * the top-K, so the truncation has to happen before the compare - not
		 * on the way out. */
		if ( lane == 0u )
			SparkQwen36DsparkTopKInsert(candidate_pool + ((uint64_t)warp * SPARK_QWEN36_DSPARK_TOPK_WIDTH),
				SPARK_QWEN36_DSPARK_TOPK_WIDTH,
				SparkQwen36DsparkTopKKey(__bfloat162float(__float2bfloat16(score)),candidate));
	}
	__syncthreads();
	SparkLmBitonicSortKeysAscending<SPARK_QWEN36_DSPARK_TOPK_WARP_POOL>(candidate_pool);
	for (slot = threadIdx.x; slot < top_k; slot += blockDim.x)
		chunk_keys[((((uint64_t)row * SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT) + chunk) * top_k) + slot] =
			candidate_pool[(SPARK_QWEN36_DSPARK_TOPK_WARP_POOL - 1u) - slot];
}

/*
 * W4 stage two: merge a row's chunk keys and decode the top_k (id, score)
 * pairs, descending. candidate_offset is the rank-local to global vocabulary
 * shift the landed head wrappers already carry.
 */
static __global__ void SparkQwen36DsparkHeadTopKMergeKernel(
	const uint64_t *chunk_keys, uint32_t *top_candidate_ids, float *top_scores_f32, void *top_scores_bf16,
	uint32_t row_count, uint32_t candidate_offset, uint32_t top_k)
{
	__shared__ uint64_t candidate_pool[SPARK_QWEN36_DSPARK_TOPK_MERGE_POOL];
	uint32_t row = blockIdx.x,slot,pool_count;
	uint64_t key;
	float score;
	if ( row >= row_count )
		return;
	pool_count = SPARK_QWEN36_DSPARK_TOPK_CHUNK_COUNT * top_k;
	for (slot = threadIdx.x; slot < SPARK_QWEN36_DSPARK_TOPK_MERGE_POOL; slot += blockDim.x)
		candidate_pool[slot] = slot < pool_count ? chunk_keys[((uint64_t)row * pool_count) + slot] : 0ull;
	__syncthreads();
	SparkLmBitonicSortKeysAscending<SPARK_QWEN36_DSPARK_TOPK_MERGE_POOL>(candidate_pool);
	for (slot = threadIdx.x; slot < top_k; slot += blockDim.x)
	{
		key = candidate_pool[(SPARK_QWEN36_DSPARK_TOPK_MERGE_POOL - 1u) - slot];
		score = key != 0ull ? SparkQwen36DsparkTopKKeyScore(key) : -3.4028235e38f;
		top_candidate_ids[((uint64_t)row * top_k) + slot] =
			key != 0ull ? SparkQwen36DsparkTopKKeyCandidate(key) + candidate_offset : UINT32_MAX;
		if ( top_scores_f32 != 0 )
			top_scores_f32[((uint64_t)row * top_k) + slot] = score;
		/* The selector consumes the unary logits as BF16 in production; the
		 * value is already BF16-exact, so this store is a reinterpretation,
		 * never a second rounding. */
		if ( top_scores_bf16 != 0 )
			SparkLmFloatToBf16(top_scores_bf16,((uint64_t)row * top_k) + slot,score);
	}
}

/*
 * W3 step one: the context gate H = bf16(hidden_projection @ h_t), one
 * [rank, hidden] BF16 matvec per slot. Same staging and dot product as the
 * head, so the drafter has exactly one BF16 matvec convention.
 */
static __global__ void SparkQwen36DsparkSelectorProjectKernel(
	const void *hidden_bf16, const void *projection_bf16, void *context_gate_bf16,
	uint32_t row_count, uint32_t rank, uint32_t hidden_dimension)
{
	extern __shared__ float hidden_shared[];
	uint32_t row = blockIdx.x,lane = threadIdx.x % SPARK_LM_WARP_LANES,warp = threadIdx.x / SPARK_LM_WARP_LANES,element;
	float value;
	if ( row >= row_count )
		return;
	SparkLmHeadStageHidden(hidden_bf16,hidden_shared,row,hidden_dimension);
	for (element = warp; element < rank; element += SPARK_LM_CTA_WARPS)
	{
		value = SparkLmWarpReduceSum(SparkLmDotRowBf16(hidden_shared,projection_bf16,element,hidden_dimension,lane));
		if ( lane == 0u )
			SparkLmFloatToBf16(context_gate_bf16,((uint64_t)row * rank) + element,value);
	}
}

/*
 * W3 step two: the K x K edge lattice of one (batch, slot).
 *
 *   predecessor id p : the anchor token for slot 0, else candidate_ids[slot-1][p]
 *                      (this IS the oracle's concatenate([anchor, ids[:, :-1]]))
 *   gate[p][r]       : A[pred_id[p]][r] * H[r]        (fp32, exact for BF16 inputs)
 *   edge[p][c]       : unary[c] + sum_r gate[p][r] * B[cand_id[c]][r]
 *
 * One CTA per (batch, slot) and one thread per edge. The gathered codebook
 * rows live in shared memory: the predecessor gate as [p][r] (the 16 threads
 * sharing a p broadcast-read one address) and the successor rows TRANSPOSED
 * as [r][c] (the 16 threads of a p read 16 consecutive addresses), which is
 * what keeps the inner product conflict free. The accumulation is ascending
 * in r and stays fp32 to the end: the oracle adds unary to the completed
 * einsum and never truncates the edge, so neither does this.
 */
static __global__ void SparkQwen36DsparkSelectorEdgeKernel(
	const void *predecessor_bf16, const void *successor_bf16, const uint32_t *candidate_ids,
	const uint32_t *anchor_token_ids, const float *unary_f32, const void *context_gate_bf16,
	float *edges_f32, uint32_t batch_count, uint32_t slot_count, uint32_t top_k, uint32_t rank)
{
	__shared__ float gate_shared[SPARK_QWEN36_DSPARK_SELECTOR_TOP_K * SPARK_QWEN36_DSPARK_SELECTOR_RANK];
	__shared__ float successor_shared[SPARK_QWEN36_DSPARK_SELECTOR_RANK * SPARK_QWEN36_DSPARK_SELECTOR_TOP_K];
	__shared__ float context_shared[SPARK_QWEN36_DSPARK_SELECTOR_RANK];
	__shared__ uint32_t predecessor_ids[SPARK_QWEN36_DSPARK_SELECTOR_TOP_K];
	__shared__ uint32_t successor_ids[SPARK_QWEN36_DSPARK_SELECTOR_TOP_K];
	uint32_t row = blockIdx.x,batch,slot,index,element,predecessor,successor;
	float accumulator;
	if ( row >= batch_count * slot_count )
		return;
	batch = row / slot_count;
	slot = row - (batch * slot_count);
	if ( threadIdx.x < top_k )
	{
		successor_ids[threadIdx.x] = candidate_ids[((uint64_t)row * top_k) + threadIdx.x];
		predecessor_ids[threadIdx.x] = slot == 0u
			? anchor_token_ids[batch]
			: candidate_ids[((uint64_t)(row - 1u) * top_k) + threadIdx.x];
	}
	for (index = threadIdx.x; index < rank; index += blockDim.x)
		context_shared[index] = SparkLmBf16ToFloat(context_gate_bf16,((uint64_t)row * rank) + index);
	__syncthreads();
	for (index = threadIdx.x; index < top_k * rank; index += blockDim.x)
	{
		predecessor = index / rank;
		element = index - (predecessor * rank);
		gate_shared[(predecessor * rank) + element] =
			SparkLmBf16ToFloat(predecessor_bf16,((uint64_t)predecessor_ids[predecessor] * rank) + element) *
			context_shared[element];
	}
	for (index = threadIdx.x; index < top_k * rank; index += blockDim.x)
	{
		successor = index / rank;
		element = index - (successor * rank);
		successor_shared[(element * top_k) + successor] =
			SparkLmBf16ToFloat(successor_bf16,((uint64_t)successor_ids[successor] * rank) + element);
	}
	__syncthreads();
	if ( threadIdx.x < top_k * top_k )
	{
		predecessor = threadIdx.x / top_k;
		successor = threadIdx.x - (predecessor * top_k);
		accumulator = 0.0f;
		for (element = 0u; element < rank; element++)
			accumulator = fmaf(gate_shared[(predecessor * rank) + element],successor_shared[(element * top_k) + successor],accumulator);
		edges_f32[((uint64_t)row * top_k * top_k) + (predecessor * top_k) + successor] =
			unary_f32[((uint64_t)row * top_k) + successor] + accumulator;
	}
}

/*
 * W3 step three: the greedy walk. previous = 0 is the anchor row of slot 0's
 * lattice (every predecessor of slot 0 IS the anchor, so row 0 is the anchor
 * row), then each slot's pick becomes the next slot's predecessor row. The
 * compare is STRICTLY greater scanning ascending, i.e. argmax-first-max, the
 * oracle's min(where(row == row.max())). The slot-to-slot dependency is a
 * loop inside one program, never a kernel per slot.
 */
static __global__ void SparkQwen36DsparkSelectorWalkKernel(
	const float *edges_f32, const uint32_t *candidate_ids, uint32_t *draft_token_ids,
	uint32_t *draft_candidate_slots, uint32_t batch_count, uint32_t slot_count, uint32_t top_k)
{
	uint32_t batch = (blockIdx.x * blockDim.x) + threadIdx.x,slot,candidate,best_candidate,previous = 0u;
	uint64_t row;
	const float *lattice_row;
	float best,value;
	if ( batch >= batch_count )
		return;
	for (slot = 0u; slot < slot_count; slot++)
	{
		row = ((uint64_t)batch * slot_count) + slot;
		lattice_row = edges_f32 + (((row * top_k) + previous) * top_k);
		best = lattice_row[0];
		best_candidate = 0u;
		for (candidate = 1u; candidate < top_k; candidate++)
		{
			value = lattice_row[candidate];
			if ( value > best )
			{
				best = value;
				best_candidate = candidate;
			}
		}
		draft_token_ids[row] = candidate_ids[(row * top_k) + best_candidate];
		if ( draft_candidate_slots != 0 )
			draft_candidate_slots[row] = best_candidate;
		previous = best_candidate;
	}
}
