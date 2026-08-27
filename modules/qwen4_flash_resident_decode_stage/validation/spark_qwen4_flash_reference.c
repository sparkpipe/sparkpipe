#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * CPU reference oracles for the Qwen 3.6 27B stage, encoding the PINNED
 * forms from modeling_qwen4_exp (transformers main, 2026-07) in plain fp32 C:
 *
 * - recurrent gated delta rule: q,k L2-normalized per head (eps 1e-6), query
 *   scaled by 1/sqrt(dk); per step S *= exp(g), delta = beta * (v - S^T k),
 *   S += outer(k, delta), o = S^T q
 * - depthwise causal conv, kernel 4, NO bias, silu on the conv output, with
 *   the carried tail state the driver hands across dispatches
 * - gated RMSNorm per value head: fp32 norm, times weight, times silu(z)
 * - full attention with the FUSED per-head query|gate projection layout,
 *   per-head q/k RMSNorm, partial RoPE on the first 64 dims (HF rotate-half
 *   pairing), GQA sharing, and sigmoid(gate) on the attention output
 *
 * The tests are self-contained on synthetic data and assert closed-form or
 * invariance properties, plus the state-carry contract: a sequence processed
 * in one pass must equal the same sequence processed in two carried passes,
 * which is exactly the boundary the resident driver crosses every dispatch.
 * The chunk-vs-recurrence oracle lands together with the CUDA chunk kernel,
 * mirroring its algorithm, as the K3 module did.
 *
 * These are dimension-generic; the tests run small head counts with the real
 * per-head widths where the width matters (rope pairing, gated norm).
 */

static uint64_t SparkQwen4FlashRefNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static float SparkQwen4FlashRefUniform(uint64_t *state)
{
	return(((float)(SparkQwen4FlashRefNext(state) & 0xffffffu) / 8388608.0f) - 1.0f);
}

static void SparkQwen4FlashRefFill(float *buffer, uint64_t count, uint64_t *state)
{
	uint64_t index;
	for (index = 0; index < count; index++)
		buffer[index] = SparkQwen4FlashRefUniform(state) * 0.5f;
}

static float SparkQwen4FlashRefSilu(float value)
{
	return(value / (1.0f + expf(-value)));
}

static float SparkQwen4FlashRefSigmoid(float value)
{
	return(1.0f / (1.0f + expf(-value)));
}

static void SparkQwen4FlashRefL2Norm(const float *input, float *output, uint32_t dimension)
{
	uint32_t element;
	float total = 0.0f;
	for (element = 0; element < dimension; element++)
		total += (input[element] * input[element]);
	total = 1.0f / sqrtf(total + 1e-6f);
	for (element = 0; element < dimension; element++)
		output[element] = input[element] * total;
}

/*
 * Recurrent gated delta rule for ONE head, sequence-major. state is dk x dv
 * row-major and is read and written, which is the carry contract. q and k
 * arrive un-normalized; the L2 norm and the query scale live here, matching
 * torch_recurrent_gated_delta_rule with use_qk_l2norm_in_kernel.
 */
static void SparkQwen4FlashRefGdnRecurrence(const float *q, const float *k, const float *v, const float *g, const float *beta, float *state, float *output, uint32_t tokens, uint32_t dk, uint32_t dv)
{
	float qn[256],kn[256],delta[256];
	float scale = 1.0f / sqrtf((float)dk),decay,kv_mem;
	uint32_t token,row,column;
	for (token = 0; token < tokens; token++)
	{
		SparkQwen4FlashRefL2Norm(q + ((uint64_t)token * dk),qn,dk);
		SparkQwen4FlashRefL2Norm(k + ((uint64_t)token * dk),kn,dk);
		for (row = 0; row < dk; row++)
			qn[row] *= scale;
		decay = expf(g[token]);
		for (row = 0; row < dk; row++)
			for (column = 0; column < dv; column++)
				state[(row * dv) + column] *= decay;
		for (column = 0; column < dv; column++)
		{
			kv_mem = 0.0f;
			for (row = 0; row < dk; row++)
				kv_mem += (state[(row * dv) + column] * kn[row]);
			delta[column] = (v[((uint64_t)token * dv) + column] - kv_mem) * beta[token];
		}
		for (row = 0; row < dk; row++)
			for (column = 0; column < dv; column++)
				state[(row * dv) + column] += (kn[row] * delta[column]);
		for (column = 0; column < dv; column++)
		{
			kv_mem = 0.0f;
			for (row = 0; row < dk; row++)
				kv_mem += (state[(row * dv) + column] * qn[row]);
			output[((uint64_t)token * dv) + column] = kv_mem;
		}
	}
}

// Depthwise causal conv over one channel: kernel 4, no bias, silu output.
// tail[3] carries the last three raw inputs across calls.
static void SparkQwen4FlashRefConvChannel(const float *input, const float *weight, float *tail, float *output, uint32_t tokens)
{
	float window[4];
	uint32_t token,tap;
	float accumulator;
	for (token = 0; token < tokens; token++)
	{
		window[0] = token >= 3u ? input[token - 3u] : tail[token];
		window[1] = token >= 2u ? input[token - 2u] : tail[token + 1u];
		window[2] = token >= 1u ? input[token - 1u] : tail[token + 2u];
		window[3] = input[token];
		accumulator = 0.0f;
		for (tap = 0; tap < 4u; tap++)
			accumulator += (window[tap] * weight[tap]);
		output[token] = SparkQwen4FlashRefSilu(accumulator);
	}
	tail[0] = tokens >= 3u ? input[tokens - 3u] : (tokens == 2u ? tail[2] : (tokens == 1u ? tail[1] : tail[0]));
	tail[1] = tokens >= 2u ? input[tokens - 2u] : (tokens == 1u ? tail[2] : tail[1]);
	tail[2] = tokens >= 1u ? input[tokens - 1u] : tail[2];
}

static void SparkQwen4FlashRefGatedNorm(const float *input, const float *z, const float *weight, float *output, uint32_t dimension, float epsilon)
{
	float variance = 0.0f,inverse;
	uint32_t element;
	for (element = 0; element < dimension; element++)
		variance += (input[element] * input[element]);
	inverse = 1.0f / sqrtf((variance / (float)dimension) + epsilon);
	for (element = 0; element < dimension; element++)
		output[element] = (input[element] * inverse) * weight[element] * SparkQwen4FlashRefSilu(z[element]);
}

// HF rotate-half partial RoPE on the first rope_dim dims of one head vector.
static void SparkQwen4FlashRefRope(float *vector, uint32_t rope_dim, uint32_t position, float theta)
{
	uint32_t pair,half = rope_dim / 2u;
	float frequency,angle,cosine,sine,low,high;
	for (pair = 0; pair < half; pair++)
	{
		frequency = powf(theta,-((float)(2u * pair) / (float)rope_dim));
		angle = (float)position * frequency;
		cosine = cosf(angle);
		sine = sinf(angle);
		low = vector[pair];
		high = vector[pair + half];
		vector[pair] = (low * cosine) - (high * sine);
		vector[pair + half] = (high * cosine) + (low * sine);
	}
}

/*
 * Full attention for one query row at position (tokens - 1) against a cache
 * of `tokens` positions, one kv head shared by `group` query heads, with the
 * fused per-head [query|gate] input layout and sigmoid gating on the output.
 * q_fused is group heads x (2 x head_dim); k/v caches are tokens x head_dim,
 * already normalized and roped by the caller for all but the newest row.
 */
static void SparkQwen4FlashRefAttention(const float *q_fused, const float *k_cache, const float *v_cache, const float *q_norm_weight, float *output, uint32_t group, uint32_t head_dim, uint32_t rope_dim, uint32_t tokens, float epsilon)
{
	float qh[512],scores[128],probability;
	float scale = 1.0f / sqrtf((float)head_dim),maximum,total,variance,inverse;
	uint32_t head,element,token;
	for (head = 0; head < group; head++)
	{
		const float *fused = q_fused + ((uint64_t)head * 2u * head_dim);
		variance = 0.0f;
		for (element = 0; element < head_dim; element++)
			variance += (fused[element] * fused[element]);
		inverse = 1.0f / sqrtf((variance / (float)head_dim) + epsilon);
		for (element = 0; element < head_dim; element++)
			qh[element] = fused[element] * inverse * q_norm_weight[element];
		SparkQwen4FlashRefRope(qh,rope_dim,tokens - 1u,10000000.0f);
		maximum = -3.0e38f;
		for (token = 0; token < tokens; token++)
		{
			probability = 0.0f;
			for (element = 0; element < head_dim; element++)
				probability += (qh[element] * k_cache[((uint64_t)token * head_dim) + element]);
			scores[token] = probability * scale;
			if ( scores[token] > maximum )
				maximum = scores[token];
		}
		total = 0.0f;
		for (token = 0; token < tokens; token++)
		{
			scores[token] = expf(scores[token] - maximum);
			total += scores[token];
		}
		for (element = 0; element < head_dim; element++)
		{
			probability = 0.0f;
			for (token = 0; token < tokens; token++)
				probability += ((scores[token] / total) * v_cache[((uint64_t)token * head_dim) + element]);
			output[((uint64_t)head * head_dim) + element] = probability * SparkQwen4FlashRefSigmoid(fused[head_dim + element]);
		}
	}
}

/*
 * Chunked gated delta rule for one head, chunk 64, matching
 * torch_chunk_gated_delta_rule exactly: intra-chunk cumulative decay, the
 * forward-substitution UT transform T = (I - A)^-1 applied to beta-scaled
 * values and decayed keys, then per chunk out = (q o e^G) S + (q k^T o D) v_new
 * with v_new = T v_beta - (T (k_beta o e^G)) S and the state carried as
 * S <- S e^G_last + (k o e^(G_last - G))^T v_new. tokens must be a multiple
 * of the chunk; state is read and written, the same carry contract as the
 * recurrence. This is the CPU mirror the wmma prefill kernel is checked
 * against.
 */
#define SPARK_QWEN4_FLASH_REF_CHUNK 64u

static void SparkQwen4FlashRefChunkPrepare(const float *q, const float *k, const float *beta, const float *g, float *qn, float *kn, float *cum_g, float *decay, float *attn, uint32_t dk)
{
	uint32_t row,column,element;
	float scale = 1.0f / sqrtf((float)dk),running = 0.0f,product;
	for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
	{
		SparkQwen4FlashRefL2Norm(q + ((uint64_t)row * dk),qn + ((uint64_t)row * dk),dk);
		SparkQwen4FlashRefL2Norm(k + ((uint64_t)row * dk),kn + ((uint64_t)row * dk),dk);
		for (element = 0; element < dk; element++)
			qn[(row * dk) + element] *= scale;
		running += g[row];
		cum_g[row] = running;
	}
	for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
		for (column = 0; column < SPARK_QWEN4_FLASH_REF_CHUNK; column++)
			decay[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + column] = column <= row ? expf(cum_g[row] - cum_g[column]) : 0.0f;
	for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
		for (column = 0; column < SPARK_QWEN4_FLASH_REF_CHUNK; column++)
		{
			product = 0.0f;
			for (element = 0; element < dk && column < row; element++)
				product += (kn[(row * dk) + element] * beta[row] * kn[(column * dk) + element]);
			attn[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + column] = column < row ? -(product * decay[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + column]) : 0.0f;
		}
}

static void SparkQwen4FlashRefChunkForwardSubstitute(float *attn)
{
	uint32_t row,column,element;
	float accumulator;
	for (row = 1; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
		for (column = 0; column < row; column++)
		{
			accumulator = attn[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + column];
			for (element = 0; element < row; element++)
				accumulator += (attn[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + element] * attn[(element * SPARK_QWEN4_FLASH_REF_CHUNK) + column]);
			attn[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + column] = accumulator;
		}
	for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
		attn[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + row] += 1.0f;
}

// w = T (v o beta); kg = T (k o beta o e^G): the two transformed operands.
static void SparkQwen4FlashRefChunkTransform(const float *attn, const float *kn, const float *v, const float *beta, const float *cum_g, float *w, float *kg, uint32_t dk, uint32_t dv)
{
	uint32_t row,column,element;
	float accumulator;
	for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
	{
		for (column = 0; column < dv; column++)
		{
			accumulator = 0.0f;
			for (element = 0; element < SPARK_QWEN4_FLASH_REF_CHUNK; element++)
				accumulator += (attn[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + element] * v[(element * dv) + column] * beta[element]);
			w[(row * dv) + column] = accumulator;
		}
		for (column = 0; column < dk; column++)
		{
			accumulator = 0.0f;
			for (element = 0; element < SPARK_QWEN4_FLASH_REF_CHUNK; element++)
				accumulator += (attn[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + element] * kn[(element * dk) + column] * beta[element] * expf(cum_g[element]));
			kg[(row * dk) + column] = accumulator;
		}
	}
}

static void SparkQwen4FlashRefChunkStep(const float *qn, const float *kn, const float *w, const float *kg, const float *cum_g, const float *decay, float *state, float *output, uint32_t dk, uint32_t dv)
{
	float v_new[SPARK_QWEN4_FLASH_REF_CHUNK * 256u],score;
	uint32_t row,column,element;
	float g_last = cum_g[SPARK_QWEN4_FLASH_REF_CHUNK - 1u],carry;
	for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
		for (column = 0; column < dv; column++)
		{
			score = 0.0f;
			for (element = 0; element < dk; element++)
				score += (kg[(row * dk) + element] * state[(element * dv) + column]);
			v_new[(row * dv) + column] = w[(row * dv) + column] - score;
		}
	for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
		for (column = 0; column < dv; column++)
		{
			score = 0.0f;
			for (element = 0; element < dk; element++)
				score += (qn[(row * dk) + element] * expf(cum_g[row]) * state[(element * dv) + column]);
			for (element = 0; element <= row; element++)
			{
				float dot = 0.0f;
				uint32_t inner;
				for (inner = 0; inner < dk; inner++)
					dot += (qn[(row * dk) + inner] * kn[(element * dk) + inner]);
				score += (dot * decay[(row * SPARK_QWEN4_FLASH_REF_CHUNK) + element] * v_new[(element * dv) + column]);
			}
			output[(row * dv) + column] = score;
		}
	for (element = 0; element < dk; element++)
		for (column = 0; column < dv; column++)
		{
			carry = state[(element * dv) + column] * expf(g_last);
			for (row = 0; row < SPARK_QWEN4_FLASH_REF_CHUNK; row++)
				carry += (kn[(row * dk) + element] * expf(g_last - cum_g[row]) * v_new[(row * dv) + column]);
			state[(element * dv) + column] = carry;
		}
}

static void SparkQwen4FlashRefGdnChunk(const float *q, const float *k, const float *v, const float *g, const float *beta, float *state, float *output, uint32_t tokens, uint32_t dk, uint32_t dv)
{
	float qn[SPARK_QWEN4_FLASH_REF_CHUNK * 256u],kn[SPARK_QWEN4_FLASH_REF_CHUNK * 256u];
	float cum_g[SPARK_QWEN4_FLASH_REF_CHUNK],decay[SPARK_QWEN4_FLASH_REF_CHUNK * SPARK_QWEN4_FLASH_REF_CHUNK],attn[SPARK_QWEN4_FLASH_REF_CHUNK * SPARK_QWEN4_FLASH_REF_CHUNK];
	float w[SPARK_QWEN4_FLASH_REF_CHUNK * 256u],kg[SPARK_QWEN4_FLASH_REF_CHUNK * 256u];
	uint32_t chunk,base;
	for (chunk = 0; chunk < tokens / SPARK_QWEN4_FLASH_REF_CHUNK; chunk++)
	{
		base = chunk * SPARK_QWEN4_FLASH_REF_CHUNK;
		SparkQwen4FlashRefChunkPrepare(q + ((uint64_t)base * dk),k + ((uint64_t)base * dk),beta + base,g + base,qn,kn,cum_g,decay,attn,dk);
		SparkQwen4FlashRefChunkForwardSubstitute(attn);
		SparkQwen4FlashRefChunkTransform(attn,kn,v + ((uint64_t)base * dv),beta + base,cum_g,w,kg,dk,dv);
		SparkQwen4FlashRefChunkStep(qn,kn,w,kg,cum_g,decay,state,output + ((uint64_t)base * dv),dk,dv);
	}
}

static int32_t SparkQwen4FlashRefTestChunkAgainstRecurrence(void)
{
	float q[192u * 32u],k[192u * 32u],v[192u * 32u],g[192u],beta[192u];
	float state_rec[32u * 32u],state_chunk[32u * 32u],out_rec[192u * 32u],out_chunk[192u * 32u];
	uint64_t noise = 0xc4a1c4a1u;
	float difference = 0.0f,state_difference = 0.0f,delta;
	uint32_t index;
	SparkQwen4FlashRefFill(q,192u * 32u,&noise);
	SparkQwen4FlashRefFill(k,192u * 32u,&noise);
	SparkQwen4FlashRefFill(v,192u * 32u,&noise);
	for (index = 0; index < 192u; index++)
	{
		g[index] = -0.05f - (0.4f * fabsf(SparkQwen4FlashRefUniform(&noise)));
		beta[index] = 0.2f + (0.6f * fabsf(SparkQwen4FlashRefUniform(&noise)));
	}
	// Warm a nonzero state with 64 recurrent tokens, then run the NEXT 128
	// tokens both ways from that shared state: the initial-state path is
	// exactly what mid-sequence chunked prefill needs.
	memset(state_rec,0,sizeof(state_rec));
	SparkQwen4FlashRefGdnRecurrence(q,k,v,g,beta,state_rec,out_rec,64u,32u,32u);
	memcpy(state_chunk,state_rec,sizeof(state_rec));
	SparkQwen4FlashRefGdnRecurrence(q + (64u * 32u),k + (64u * 32u),v + (64u * 32u),g + 64u,beta + 64u,state_rec,out_rec + (64u * 32u),128u,32u,32u);
	SparkQwen4FlashRefGdnChunk(q + (64u * 32u),k + (64u * 32u),v + (64u * 32u),g + 64u,beta + 64u,state_chunk,out_chunk + (64u * 32u),128u,32u,32u);
	for (index = 64u * 32u; index < 192u * 32u; index++)
	{
		delta = fabsf(out_rec[index] - out_chunk[index]);
		if ( delta > difference )
			difference = delta;
	}
	for (index = 0; index < 32u * 32u; index++)
	{
		delta = fabsf(state_rec[index] - state_chunk[index]);
		if ( delta > state_difference )
			state_difference = delta;
	}
	printf("chunk_vs_recurrence out_max_abs_diff=%.3e state_max_abs_diff=%.3e\n",(double)difference,(double)state_difference);
	return((difference < 1e-4f && state_difference < 1e-4f) ? 0 : -1);
}

static int32_t SparkQwen4FlashRefTestCarry(void)
{
	float q[128u * 32u],k[128u * 32u],v[128u * 32u],g[128u],beta[128u];
	float state_one[32u * 32u],state_two[32u * 32u],out_one[128u * 32u],out_two[128u * 32u];
	uint64_t noise = 0x51363636u;
	float difference = 0.0f,delta;
	uint32_t index;
	SparkQwen4FlashRefFill(q,128u * 32u,&noise);
	SparkQwen4FlashRefFill(k,128u * 32u,&noise);
	SparkQwen4FlashRefFill(v,128u * 32u,&noise);
	for (index = 0; index < 128u; index++)
	{
		g[index] = -0.05f - (0.4f * fabsf(SparkQwen4FlashRefUniform(&noise)));
		beta[index] = 0.2f + (0.6f * fabsf(SparkQwen4FlashRefUniform(&noise)));
	}
	memset(state_one,0,sizeof(state_one));
	memset(state_two,0,sizeof(state_two));
	SparkQwen4FlashRefGdnRecurrence(q,k,v,g,beta,state_one,out_one,128u,32u,32u);
	SparkQwen4FlashRefGdnRecurrence(q,k,v,g,beta,state_two,out_two,64u,32u,32u);
	SparkQwen4FlashRefGdnRecurrence(q + (64u * 32u),k + (64u * 32u),v + (64u * 32u),g + 64u,beta + 64u,state_two,out_two + (64u * 32u),64u,32u,32u);
	for (index = 0; index < 128u * 32u; index++)
	{
		delta = fabsf(out_one[index] - out_two[index]);
		if ( delta > difference )
			difference = delta;
	}
	printf("carry max_abs_diff=%.3e\n",(double)difference);
	return(difference < 1e-6f ? 0 : -1);
}

static int32_t SparkQwen4FlashRefTestSaturatedDecay(void)
{
	float q[32u],k[32u],v[32u],qn[32u],kn[32u],state[32u * 32u],output[32u];
	float g = -30.0f,beta = 0.7f,expected,alignment = 0.0f,difference = 0.0f,delta;
	uint64_t noise = 0xbeef51u;
	uint32_t index;
	SparkQwen4FlashRefFill(q,32u,&noise);
	SparkQwen4FlashRefFill(k,32u,&noise);
	SparkQwen4FlashRefFill(v,32u,&noise);
	SparkQwen4FlashRefFill(state,32u * 32u,&noise);
	SparkQwen4FlashRefGdnRecurrence(q,k,v,&g,&beta,state,output,1u,32u,32u);
	// With exp(-30) the prior state is annihilated: o = <q_n, k_n> * beta * v
	// with q_n scaled by 1/sqrt(dk).
	SparkQwen4FlashRefL2Norm(q,qn,32u);
	SparkQwen4FlashRefL2Norm(k,kn,32u);
	for (index = 0; index < 32u; index++)
		alignment += (qn[index] * kn[index]);
	alignment /= sqrtf(32.0f);
	for (index = 0; index < 32u; index++)
	{
		expected = alignment * beta * v[index];
		delta = fabsf(output[index] - expected);
		if ( delta > difference )
			difference = delta;
	}
	printf("saturated_decay max_abs_diff=%.3e\n",(double)difference);
	return(difference < 1e-5f ? 0 : -1);
}

static int32_t SparkQwen4FlashRefTestConv(void)
{
	float weight[4] = {0.3f,-0.5f,0.8f,0.4f};
	float input[24u],one_pass[24u],two_pass[24u],tail_one[3],tail_two[3];
	uint64_t noise = 0xc0471u ^ 0x1234u;
	float difference = 0.0f,delta,impulse_in[8u] = {1.0f,0,0,0,0,0,0,0},impulse_out[8u],tail_zero[3] = {0,0,0};
	uint32_t index;
	SparkQwen4FlashRefConvChannel(impulse_in,weight,tail_zero,impulse_out,8u);
	for (index = 0; index < 4u; index++)
	{
		delta = fabsf(impulse_out[index] - SparkQwen4FlashRefSilu(weight[3u - index]));
		if ( delta > difference )
			difference = delta;
	}
	SparkQwen4FlashRefFill(input,24u,&noise);
	memset(tail_one,0,sizeof(tail_one));
	memset(tail_two,0,sizeof(tail_two));
	SparkQwen4FlashRefConvChannel(input,weight,tail_one,one_pass,24u);
	SparkQwen4FlashRefConvChannel(input,weight,tail_two,two_pass,10u);
	SparkQwen4FlashRefConvChannel(input + 10u,weight,tail_two,two_pass + 10u,14u);
	for (index = 0; index < 24u; index++)
	{
		delta = fabsf(one_pass[index] - two_pass[index]);
		if ( delta > difference )
			difference = delta;
	}
	printf("conv max_abs_diff=%.3e\n",(double)difference);
	return(difference < 1e-6f ? 0 : -1);
}

static int32_t SparkQwen4FlashRefTestGatedNorm(void)
{
	float input[128u],z[128u],weight[128u],output[128u];
	uint64_t noise = 0x9a9a9au;
	float variance = 0.0f,inverse,expected,difference = 0.0f,delta;
	uint32_t index;
	SparkQwen4FlashRefFill(input,128u,&noise);
	SparkQwen4FlashRefFill(z,128u,&noise);
	for (index = 0; index < 128u; index++)
		weight[index] = 1.0f + (0.1f * SparkQwen4FlashRefUniform(&noise));
	SparkQwen4FlashRefGatedNorm(input,z,weight,output,128u,1e-6f);
	for (index = 0; index < 128u; index++)
		variance += (input[index] * input[index]);
	inverse = 1.0f / sqrtf((variance / 128.0f) + 1e-6f);
	for (index = 0; index < 128u; index++)
	{
		expected = input[index] * inverse * weight[index] * SparkQwen4FlashRefSilu(z[index]);
		delta = fabsf(output[index] - expected);
		if ( delta > difference )
			difference = delta;
	}
	memset(z,0,sizeof(z));
	SparkQwen4FlashRefGatedNorm(input,z,weight,output,128u,1e-6f);
	for (index = 0; index < 128u; index++)
		if ( fabsf(output[index]) > difference )
			difference = fabsf(output[index]);
	printf("gated_norm max_abs_diff=%.3e\n",(double)difference);
	return(difference < 1e-6f ? 0 : -1);
}

static int32_t SparkQwen4FlashRefTestAttention(void)
{
	float q_fused[2u * 2u * 256u],k_cache[48u * 256u],v_cache[48u * 256u],q_norm[256u],output[2u * 256u];
	float head[256u],rotated_q[256u],rotated_k[256u],dot_before,dot_after,difference = 0.0f,delta;
	uint64_t noise = 0xa77e0u;
	uint32_t index,token;
	SparkQwen4FlashRefFill(q_fused,2u * 2u * 256u,&noise);
	SparkQwen4FlashRefFill(k_cache,48u * 256u,&noise);
	for (index = 0; index < 256u; index++)
		q_norm[index] = 1.0f;
	// Constant value cache: softmax mixing must reproduce it exactly, so the
	// output equals sigmoid(gate) times the constant per element.
	for (token = 0; token < 48u; token++)
		for (index = 0; index < 256u; index++)
			v_cache[(token * 256u) + index] = 0.25f;
	SparkQwen4FlashRefAttention(q_fused,k_cache,v_cache,q_norm,output,2u,256u,64u,48u,1e-6f);
	for (index = 0; index < 2u * 256u; index++)
	{
		const float *fused = q_fused + (((index / 256u) * 2u * 256u) + 256u);
		delta = fabsf(output[index] - (0.25f * SparkQwen4FlashRefSigmoid(fused[index % 256u])));
		if ( delta > difference )
			difference = delta;
	}
	// RoPE inner-product invariance at equal positions on the roped span.
	SparkQwen4FlashRefFill(head,256u,&noise);
	memcpy(rotated_q,head,sizeof(head));
	SparkQwen4FlashRefFill(head,256u,&noise);
	memcpy(rotated_k,head,sizeof(head));
	dot_before = 0.0f;
	for (index = 0; index < 64u; index++)
		dot_before += (rotated_q[index] * rotated_k[index]);
	SparkQwen4FlashRefRope(rotated_q,64u,17u,10000000.0f);
	SparkQwen4FlashRefRope(rotated_k,64u,17u,10000000.0f);
	dot_after = 0.0f;
	for (index = 0; index < 64u; index++)
		dot_after += (rotated_q[index] * rotated_k[index]);
	delta = fabsf(dot_before - dot_after);
	if ( delta > difference )
		difference = delta;
	printf("attention max_abs_diff=%.3e\n",(double)difference);
	return(difference < 1e-4f ? 0 : -1);
}

int main(void)
{
	if ( SparkQwen4FlashRefTestCarry() != 0 )
		return(1);
	if ( SparkQwen4FlashRefTestChunkAgainstRecurrence() != 0 )
		return(6);
	if ( SparkQwen4FlashRefTestSaturatedDecay() != 0 )
		return(2);
	if ( SparkQwen4FlashRefTestConv() != 0 )
		return(3);
	if ( SparkQwen4FlashRefTestGatedNorm() != 0 )
		return(4);
	if ( SparkQwen4FlashRefTestAttention() != 0 )
		return(5);
	printf("PASS all cpu oracles agree\n");
	return(0);
}
