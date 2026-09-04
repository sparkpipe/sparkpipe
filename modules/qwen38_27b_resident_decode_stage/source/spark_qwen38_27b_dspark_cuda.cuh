#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "sparkpipe/spark_qwen38_27b_model.h"
#include "spark_qwen38_27b_dspark_format.h"

#define SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIM 128u
#define SPARK_QWEN38_27B_DSPARK_ATTN_KV_HEADS 8u
#define SPARK_QWEN38_27B_DSPARK_ATTN_ROPE_DIM 64u

static __device__ __forceinline__ float SparkQwen38_27bDsparkRopeFrequencyNeoX(uint32_t pair)
{
	return exp2f(-((float)(2u * pair) / (float)SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIM) * log2f((float)SPARK_QWEN38_27B_MODEL_ATTN_ROPE_THETA));
}

static __device__ __forceinline__ float SparkQwen38_27bDsparkRopeFrequency(uint32_t pair)
{
	return exp2f(-((float)(2u * pair) / (float)SPARK_QWEN38_27B_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN38_27B_MODEL_ATTN_ROPE_THETA));
}

static __global__ void SparkQwen38_27bDsparkTapStoreKernel(
	const __nv_bfloat16 *hidden_bf16, const uint64_t *row_positions,
	__nv_bfloat16 *taps_bf16, uint32_t rows, uint32_t tap_index,
	uint32_t hidden_dim, uint32_t tap_layers)
{
	const uint32_t row = blockIdx.x;
	const uint32_t c = (blockIdx.y * blockDim.x) + threadIdx.x;
	const uint64_t pos = row_positions[row];
	if ( c >= hidden_dim )
		return;
	taps_bf16[((pos * (uint64_t)tap_layers + tap_index) * (uint64_t)hidden_dim) + c] = hidden_bf16[(uint64_t)row * hidden_dim + c];
}

static __global__ void SparkQwen38_27bDsparkKPrepKernel(
	__nv_bfloat16 *k_bf16, const __nv_bfloat16 *k_norm_bf16,
	const uint64_t *positions, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	const uint32_t head = blockIdx.y;
	const uint32_t d = threadIdx.x;
	const uint32_t head_dim = SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIM;
	const uint32_t kv_heads = SPARK_QWEN38_27B_DSPARK_ATTN_KV_HEADS;
	float sum = 0.0f, scale;
	uint64_t pos;
	if ( d >= head_dim )
		return;
	{
		float v = __bfloat162float(k_bf16[((uint64_t)row * kv_heads + head) * head_dim + d]);
		sum = fmaf(v, v, sum);
	}
	__shared__ float total;
	if ( d == 0u )
		total = 0.0f;
	__syncthreads();
	atomicAdd(&total, sum);
	__syncthreads();
	scale = rsqrtf(total / (float)head_dim + 1e-6f);
	pos = positions[row];
	if ( (d & 1u) != 0u && d < SPARK_QWEN38_27B_DSPARK_ATTN_ROPE_DIM )
		return;
	{
		uint32_t dp = d ^ 1u;
		float re = __bfloat162float(k_bf16[((uint64_t)row * kv_heads + head) * head_dim + d]);
		float im = __bfloat162float(k_bf16[((uint64_t)row * kv_heads + head) * head_dim + dp]);
		re *= scale * __bfloat162float(k_norm_bf16[d]);
		im *= scale * __bfloat162float(k_norm_bf16[dp]);
		if ( d < SPARK_QWEN38_27B_DSPARK_ATTN_ROPE_DIM )
		{
			float f = (float)pos * SparkQwen38_27bDsparkRopeFrequency(d >> 1u);
			float c = cosf(f), sn = sinf(f);
			float ro = re * c - im * sn;
			float io = re * sn + im * c;
			re = ro;
			im = io;
		}
		k_bf16[((uint64_t)row * kv_heads + head) * head_dim + d] = __float2bfloat16(re);
		if ( dp != d )
			k_bf16[((uint64_t)row * kv_heads + head) * head_dim + dp] = __float2bfloat16(im);
	}
}

static __global__ void SparkQwen38_27bDsparkQPrepKernel(
	__nv_bfloat16 *q_bf16, const __nv_bfloat16 *q_norm_bf16,
	const uint64_t *positions, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	const uint32_t head = blockIdx.y;
	const uint32_t d = threadIdx.x;
	const uint32_t head_dim = SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIM;
	const uint32_t q_heads = SPARK_QWEN38_27B_DSPARK_ATTN_QUERY_HEADS;
	float sum = 0.0f;
	uint64_t pos;
	if ( d >= head_dim )
		return;
	{
		float v = __bfloat162float(q_bf16[((uint64_t)row * q_heads + head) * head_dim + d]);
		sum = fmaf(v, v, sum);
	}
	__shared__ float total;
	if ( d == 0u )
		total = 0.0f;
	__syncthreads();
	atomicAdd(&total, sum);
	__syncthreads();
	if ( (d & 1u) != 0u && d < SPARK_QWEN38_27B_DSPARK_ATTN_ROPE_DIM )
		return;
	{
		float scale = rsqrtf(total / (float)head_dim + 1e-6f);
		uint32_t dp = d ^ 1u;
		float re = __bfloat162float(q_bf16[((uint64_t)row * q_heads + head) * head_dim + d]);
		float im = __bfloat162float(q_bf16[((uint64_t)row * q_heads + head) * head_dim + dp]);
		re *= scale * __bfloat162float(q_norm_bf16[d]);
		im *= scale * __bfloat162float(q_norm_bf16[dp]);
		pos = positions[row];
		if ( d < SPARK_QWEN38_27B_DSPARK_ATTN_ROPE_DIM )
		{
			float f = (float)pos * SparkQwen38_27bDsparkRopeFrequency(d >> 1u);
			float c = cosf(f), sn = sinf(f);
			float ro = re * c - im * sn;
			float io = re * sn + im * c;
			re = ro;
			im = io;
		}
		q_bf16[((uint64_t)row * q_heads + head) * head_dim + d] = __float2bfloat16(re);
		if ( dp != d )
			q_bf16[((uint64_t)row * q_heads + head) * head_dim + dp] = __float2bfloat16(im);
	}
}


static __global__ void SparkQwen38_27bDsparkCacheAttnKernel(
	const __nv_bfloat16 *q_bf16, const __nv_bfloat16 *k_bf16, const __nv_bfloat16 *v_bf16,
	const __nv_bfloat16 *q_norm_bf16, const __nv_bfloat16 *k_norm_bf16,
	const uint64_t *positions, __nv_bfloat16 *out_bf16, uint32_t block_rows,
	uint32_t nkv, uint32_t window)
{
	extern __shared__ float scores[];
	const uint32_t row = blockIdx.x;
	const uint32_t head = blockIdx.y;
	const uint32_t d = threadIdx.x;
	const uint32_t head_dim = SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIM;
	const uint32_t kv_heads = SPARK_QWEN38_27B_DSPARK_ATTN_KV_HEADS;
	const uint32_t q_heads = SPARK_QWEN38_27B_DSPARK_ATTN_QUERY_HEADS;
	const uint32_t kv_group = head / (q_heads / kv_heads);
	float q[SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIM];
	float sum, scale, f, c, sn, other;
	__shared__ float total_sum;
	uint32_t kv, e;
	uint64_t q_pos, k_pos;
	if ( d >= head_dim )
		return;
	q_pos = positions[window + row];
	sum = 0.0f;
	for (kv = 0u; kv < head_dim; kv++)
	{
		q[kv] = __bfloat162float(q_bf16[((uint64_t)row * q_heads + head) * head_dim + kv]);
		sum = fmaf(q[kv], q[kv], sum);
	}
	scale = rsqrtf(sum / (float)head_dim + 1e-6f);
	for (kv = 0u; kv < head_dim; kv++)
		q[kv] *= scale * __bfloat162float(q_norm_bf16[kv]);
	for (kv = 0u; kv < head_dim / 2u; kv++)
	{
		f = (float)q_pos * SparkQwen38_27bDsparkRopeFrequencyNeoX(kv);
		c = cosf(f);
		sn = sinf(f);
		{
			float re = q[kv], im = q[kv + head_dim / 2u];
			q[kv] = re * c - im * sn;
			q[kv + head_dim / 2u] = re * sn + im * c;
		}
	}
	for (kv = d; kv < nkv; kv += head_dim)
	{
		const __nv_bfloat16 *krow = k_bf16 + ((uint64_t)kv * kv_heads + kv_group) * head_dim;
		float kn[SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIM];
		float s = 0.0f;
		sum = 0.0f;
		for (e = 0u; e < head_dim; e++)
		{
			kn[e] = __bfloat162float(krow[e]);
			sum = fmaf(kn[e], kn[e], sum);
		}
		scale = rsqrtf(sum / (float)head_dim + 1e-6f);
		k_pos = positions[kv];
		for (e = 0u; e < head_dim; e++)
			kn[e] *= scale * __bfloat162float(k_norm_bf16[e]);
		for (e = 0u; e < head_dim / 2u; e++)
		{
			f = (float)k_pos * SparkQwen38_27bDsparkRopeFrequencyNeoX(e);
			c = cosf(f);
			sn = sinf(f);
			{
				float re = kn[e], im = kn[e + head_dim / 2u];
				kn[e] = re * c - im * sn;
				kn[e + head_dim / 2u] = re * sn + im * c;
			}
		}
		(void)other;
		for (e = 0u; e < head_dim; e++)
			s = fmaf(q[e], kn[e], s);
		scores[kv] = s * 0.088388347f;
	}
	__syncthreads();
	if ( d == 0u )
	{
		float m = -3.4028235e38f;
		for (kv = 0u; kv < nkv; kv++)
			m = fmaxf(m, scores[kv]);
		total_sum = 0.0f;
		{
			float local_sum = 0.0f;
			float mx = m;
			for (kv = 0u; kv < nkv; kv++)
			{
				scores[kv] = __expf(scores[kv] - mx);
				local_sum += scores[kv];
			}
			total_sum = local_sum;
		}
	}
	__syncthreads();
	{
		float acc = 0.0f;
		for (kv = 0u; kv < nkv; kv++)
			acc = fmaf(scores[kv], __bfloat162float(v_bf16[((uint64_t)kv * kv_heads + kv_group) * head_dim + d]), acc);
		out_bf16[((uint64_t)row * q_heads + head) * head_dim + d] = __float2bfloat16(acc / total_sum);
	}
}

static __global__ void SparkQwen38_27bDsparkConvKernel(
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

static __global__ void SparkQwen38_27bDsparkMarkovKernel(
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
	bias_out[(uint64_t)draft_pos * vocab + v] = __bfloat162float(__float2bfloat16(acc));
}

#define SPARK_QWEN38_27B_DSPARK_SEL_THREADS 128u

static __global__ void SparkQwen38_27bDsparkSelectKernel(
	const __nv_bfloat16 *logits, const __nv_bfloat16 *hidden, const __nv_bfloat16 *hproj_w,
	uint32_t *out_ids, float *out_scores, float *out_hproj,
	uint32_t vocab, uint32_t hidden_dim, uint32_t rank, uint32_t top_k)
{
	const uint32_t slot = blockIdx.x;
	const uint32_t tid = threadIdx.x;
	const uint32_t nt = blockDim.x;
	const uint32_t K = top_k;
	const __nv_bfloat16 *row = logits + (uint64_t)(slot + 1u) * vocab;
	__shared__ float s_val[SPARK_QWEN38_27B_DSPARK_SEL_THREADS * 16u];
	__shared__ uint32_t s_idx[SPARK_QWEN38_27B_DSPARK_SEL_THREADS * 16u];
	__shared__ uint32_t s_head[SPARK_QWEN38_27B_DSPARK_SEL_THREADS];
	__shared__ unsigned long long s_best;
	float lv[16];
	uint32_t li[16];
	uint32_t ln = 0u;
	uint32_t out;
	for (uint32_t v = tid; v < vocab; v += nt)
	{
		const float value = __bfloat162float(row[v]);
		uint32_t insert = ln < K ? ln : K;
		uint32_t shift;
		while ( insert > 0u && (value > lv[insert - 1u] || (value == lv[insert - 1u] && v < li[insert - 1u])) )
			insert--;
		if ( insert == K )
			continue;
		shift = ln < K ? ln : K - 1u;
		for ( ; shift > insert; shift-- )
		{
			lv[shift] = lv[shift - 1u];
			li[shift] = li[shift - 1u];
		}
		if ( ln < K )
			ln++;
		lv[insert] = value;
		li[insert] = v;
	}
	for (out = 0u; out < ln; out++)
	{
		s_val[tid * 16u + out] = lv[out];
		s_idx[tid * 16u + out] = li[out];
	}
	s_head[tid] = 0u;
	if ( tid == 0u )
		s_best = 0ull;
	__syncthreads();
	for (out = 0u; out < K; out++)
	{
		unsigned long long mine = 0ull;
		if ( s_head[tid] < ln )
		{
			const float v0 = s_val[tid * 16u + s_head[tid]];
			const uint32_t i0 = s_idx[tid * 16u + s_head[tid]];
			uint32_t sb = __float_as_uint(v0);
			sb ^= (sb >> 31u) != 0u ? 0xFFFFFFFFu : 0x80000000u;
			mine = ((unsigned long long)sb << 20) | (unsigned long long)(0xFFFFFu - (i0 & 0xFFFFFu));
		}
		atomicMax(&s_best, mine);
		__syncthreads();
		if ( mine != 0ull && mine == s_best )
		{
			out_ids[(uint64_t)slot * K + out] = s_idx[tid * 16u + s_head[tid]];
			out_scores[(uint64_t)slot * K + out] = s_val[tid * 16u + s_head[tid]];
			s_head[tid]++;
		}
		if ( tid == 0u )
			s_best = 0ull;
		__syncthreads();
	}
	for (uint32_t rr = tid; rr < rank; rr += nt)
	{
		const __nv_bfloat16 *hrow = hidden + (uint64_t)(slot + 1u) * hidden_dim;
		const __nv_bfloat16 *wrow = hproj_w + (uint64_t)rr * hidden_dim;
		float acc = 0.0f;
		uint32_t c;
		for (c = 0u; c < hidden_dim; c++)
			acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(hrow[c]), __bfloat162float(wrow[c])));
		out_hproj[(uint64_t)slot * rank + rr] = __bfloat162float(__float2bfloat16(acc));
	}
}
