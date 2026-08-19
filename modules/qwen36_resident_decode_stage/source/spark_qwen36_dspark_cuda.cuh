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

static __device__ __forceinline__ float SparkQwen36DsparkRopeFrequency(uint32_t pair)
{
	/* Same convention as the target: theta^(-2*pair/rope_dim), rope_dim 64. */
	return exp2f(-((float)(2u * pair) / (float)SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN36_MODEL_ATTN_ROPE_THETA));
}

/* Store one tap layer's per-position hiddens into the tap history:
 * taps[position(row), tap_index, :] = hidden[row, :]. One thread per row
 * stride; the row->position map comes from the device row_positions. */
static __global__ void SparkQwen36DsparkTapStoreKernel(
	const __nv_bfloat16 *hidden_bf16, const uint64_t *row_positions,
	__nv_bfloat16 *taps_bf16, uint32_t rows, uint32_t tap_index,
	uint32_t hidden_dim, uint32_t tap_layers)
{
	const uint32_t row = blockIdx.x;
	const uint32_t c = threadIdx.x;
	const uint64_t pos = row_positions[row];
	if ( c >= hidden_dim )
		return;
	taps_bf16[((pos * (uint64_t)tap_layers + tap_index) * (uint64_t)hidden_dim) + c] = hidden_bf16[(uint64_t)row * hidden_dim + c];
}

/* K-row preparation for the drafter's context-KV cache and block K/V:
 * per (row, kv head): RMSNorm with the layer's k_norm then RoPE at the row's
 * absolute position, IN PLACE. V rows pass through untouched (upstream: no
 * norm, no rope on V). */
static __global__ void SparkQwen36DsparkKPrepKernel(
	__nv_bfloat16 *k_bf16, const __nv_bfloat16 *k_norm_bf16,
	const uint64_t *positions, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	const uint32_t head = blockIdx.y;
	const uint32_t d = threadIdx.x;
	const uint32_t head_dim = SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM;
	const uint32_t kv_heads = SPARK_QWEN36_DSPARK_ATTN_KV_HEADS;
	float kn[SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM];
	float sum = 0.0f, scale;
	uint64_t pos;
	if ( d >= head_dim )
		return;
	{
		float v = __bfloat162float(k_bf16[((uint64_t)row * kv_heads + head) * head_dim + d]);
		kn[d] = v;
		sum = fmaf(v, v, sum);
	}
	/* the whole head's threads must contribute to sum: head_dim == blockDim */
	/* full-block reduce (blockDim == head_dim == 128) */
	__shared__ float total;
	if ( d == 0u )
		total = 0.0f;
	__syncthreads();
	atomicAdd(&total, sum);
	__syncthreads();
	scale = rsqrtf(total / (float)head_dim + 1e-6f);
	pos = positions[row];
	{
		float gated = kn[d] * scale * __bfloat162float(k_norm_bf16[d]);
		if ( d < SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM )
		{
			uint32_t pair = d >> 1u;
			uint32_t is_im = d & 1u;
			float f = (float)pos * SparkQwen36DsparkRopeFrequency(pair);
			float c = cosf(f), sn = sinf(f);
			float other = kn[d ^ 1u] * scale * __bfloat162float(k_norm_bf16[d ^ 1u]);
			k_bf16[((uint64_t)row * kv_heads + head) * head_dim + d] = __float2bfloat16(is_im == 0u ? gated * c - other * sn : gated * sn + other * c);
		}
		else
			k_bf16[((uint64_t)row * kv_heads + head) * head_dim + d] = __float2bfloat16(gated);
	}
}

/* Q-row preparation: per (row, query head) RMSNorm with q_norm + RoPE, in
 * place. q layout [rows, 32*128]. */
static __global__ void SparkQwen36DsparkQPrepKernel(
	__nv_bfloat16 *q_bf16, const __nv_bfloat16 *q_norm_bf16,
	const uint64_t *positions, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	const uint32_t head = blockIdx.y;
	const uint32_t d = threadIdx.x;
	const uint32_t head_dim = SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM;
	const uint32_t q_heads = SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS;
	float qn[SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM];
	float sum = 0.0f;
	uint64_t pos;
	if ( d >= head_dim )
		return;
	{
		float v = __bfloat162float(q_bf16[((uint64_t)row * q_heads + head) * head_dim + d]);
		qn[d] = v;
		sum = fmaf(v, v, sum);
	}
	__shared__ float total;
	if ( d == 0u )
		total = 0.0f;
	__syncthreads();
	atomicAdd(&total, sum);
	__syncthreads();
	{
		float scale = rsqrtf(total / (float)head_dim + 1e-6f);
		float gated = qn[d] * scale * __bfloat162float(q_norm_bf16[d]);
		pos = positions[row];
		if ( d < SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM )
		{
			uint32_t pair = d >> 1u;
			uint32_t is_im = d & 1u;
			float f = (float)pos * SparkQwen36DsparkRopeFrequency(pair);
			float c = cosf(f), sn = sinf(f);
			float other = qn[d ^ 1u] * scale * __bfloat162float(q_norm_bf16[d ^ 1u]);
			q_bf16[((uint64_t)row * q_heads + head) * head_dim + d] = __float2bfloat16(is_im == 0u ? gated * c - other * sn : gated * sn + other * c);
		}
		else
			q_bf16[((uint64_t)row * q_heads + head) * head_dim + d] = __float2bfloat16(gated);
	}
}

/* Cache-based drafter attention. One CTA per (block row, query head),
 * 128 threads over head_dim. K/V live in one staged array:
 *   [0 .. ctx_len-1]   = context cache (from the target taps, pre-roped)
 *   [ctx_len .. nkv-1] = the block's own rows (anchor first, pre-roped)
 * Q arrives pre-normed/pre-roped. Online softmax: scores to shared, max/exp
 * reduce, then per-dim weighted V accumulation. */
static __global__ void SparkQwen36DsparkCacheAttnKernel(
	const __nv_bfloat16 *q_bf16, const __nv_bfloat16 *k_bf16, const __nv_bfloat16 *v_bf16,
	__nv_bfloat16 *out_bf16, uint32_t nkv)
{
	extern __shared__ float scores[];
	const uint32_t row = blockIdx.x;
	const uint32_t head = blockIdx.y;
	const uint32_t d = threadIdx.x;
	const uint32_t head_dim = SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM;
	const uint32_t kv_heads = SPARK_QWEN36_DSPARK_ATTN_KV_HEADS;
	const uint32_t q_heads = SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS;
	const uint32_t kv_group = head / (q_heads / kv_heads);
	float q[SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM];
	float acc, max_score, sum_exp, coeff;
	__shared__ float total_max, total_sum;
	uint32_t kv;
	if ( d >= head_dim )
		return;
	for (kv = 0u; kv < head_dim; kv++)
		q[kv] = __bfloat162float(q_bf16[((uint64_t)row * q_heads + head) * head_dim + kv]);
	/* scores: each thread handles rows strided by blockDim */
	for (kv = d; kv < nkv; kv += head_dim)
	{
		const __nv_bfloat16 *krow = k_bf16 + ((uint64_t)kv * kv_heads + kv_group) * head_dim;
		float s = 0.0f;
		uint32_t e;
		for (e = 0u; e < head_dim; e++)
			s = fmaf(q[e], __bfloat162float(krow[e]), s);
		scores[kv] = s * 0.088388347f; /* 1/sqrt(128) */
	}
	__syncthreads();
	/* max + softmax sum (block reduce through shared scalars) */
	if ( d == 0u )
	{
		total_max = -3.4028235e38f;
		total_sum = 0.0f;
	}
	__syncthreads();
	if ( d == 0u )
	{
		for (kv = 0u; kv < nkv; kv++)
			total_max = fmaxf(total_max, scores[kv]);
	}
	__syncthreads();
	{
		float local_sum = 0.0f;
		for (kv = d; kv < nkv; kv += head_dim)
		{
			scores[kv] = __expf(scores[kv] - total_max);
			local_sum += scores[kv];
		}
		atomicAdd(&total_sum, local_sum);
	}
	__syncthreads();
	/* weighted V: thread d accumulates its dim over all rows */
	acc = 0.0f;
	(void)max_score;
	(void)sum_exp;
	(void)coeff;
	for (kv = 0u; kv < nkv; kv++)
		acc = fmaf(scores[kv], __bfloat162float(v_bf16[((uint64_t)kv * kv_heads + kv_group) * head_dim + d]), acc);
	out_bf16[((uint64_t)row * q_heads + head) * head_dim + d] = __float2bfloat16(acc / total_sum);
}

/* Grouped dynamic depthwise conv (DFlash2), fused into ONE elementwise pass — no
 * im2col. One CTA per (block position, group); threads span the group_size channels.
 *   x:    [block_size, H] BF16
 *   delta: the kernel_projection's fused [block_size, sides, taps, num_groups] BF16
 *          output, read directly: row stride is sides*taps*num_groups and `side`
 *          selects the prepare (0) or finish (1) plane. (A caller-side pointer
 *          offset cannot work: the row stride is the FULL width, not one side.)
 *   base:  [taps, H] BF16 (learned base, ONE side; caller offsets by side*taps*H)
 *   out[i,c] = sum_t (base[t,c] + delta[i,side,t,g(c)]) * x[i-t,c], taps zero where
 *              (i & (block_size-1)) < t.
 */
static __global__ void SparkQwen36DsparkConvKernel(
	const __nv_bfloat16 *x_bf16, const __nv_bfloat16 *delta_bf16, const __nv_bfloat16 *base_bf16,
	__nv_bfloat16 *out_bf16, uint32_t block_size, uint32_t num_groups, uint32_t group_size, uint32_t side)
{
	const uint32_t pos = blockIdx.x;
	const uint32_t group = blockIdx.y;
	const uint32_t c = group * group_size + threadIdx.x;
	const uint32_t H = num_groups * group_size;
	const uint64_t plane = (uint64_t)pos * 4u * num_groups + (uint64_t)side * 2u * num_groups;
	uint32_t p;
	float x0, d0, out;
	if ( c >= H )
		return;
	p = (block_size & (block_size - 1u)) == 0u ? pos & (block_size - 1u) : pos % block_size;
	x0 = __bfloat162float(x_bf16[(uint64_t)pos * H + c]);
	d0 = __bfloat162float(delta_bf16[plane + 0u * num_groups + group]);
	out = (__bfloat162float(base_bf16[0u * H + c]) + d0) * x0;
	if ( p >= 1u )
	{
		float x1 = __bfloat162float(x_bf16[((uint64_t)(pos - 1u)) * H + c]);
		float d1 = __bfloat162float(delta_bf16[plane + 1u * num_groups + group]);
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
}