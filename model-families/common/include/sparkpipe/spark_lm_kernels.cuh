#pragma once

#include <stdio.h>

#include <cuda_bf16.h>

#include <cuda_fp16.h>

#include <cuda_runtime.h>

#include <math.h>

#include <stdint.h>

#include "inference/kernels/activation.cuh"

#include "sparkpipe/spark_head_screen.h"

#if defined(__CUDACC__)
#define SPARK_LM_LAUNCH(launch_statement) launch_statement
#else
#define SPARK_LM_LAUNCH(launch_statement)
#endif

#define SPARK_LM_WARP_LANES 32u

#define SPARK_LM_CTA_THREADS 256u

#define SPARK_LM_CTA_WARPS (SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES)

static __device__ __forceinline__ float SparkLmBf16ToFloat(const void *source, uint64_t index)
{
	return(__bfloat162float(((const __nv_bfloat16 *)source)[index]));
}

static __device__ __forceinline__ float2 SparkLmLoadBf16Pair(const void *source, uint64_t pair_index)
{
	uint32_t raw = __ldg(((const uint32_t *)source) + pair_index);
	return(__bfloat1622float2(*(const __nv_bfloat162 *)&raw));
}

static __device__ __forceinline__ float SparkLmWarpReduceSum(float value)
{
	uint32_t offset;
	for (offset = SPARK_LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	return(value);
}

static __device__ __forceinline__ float SparkLmDotRowBf16(const float *shared_input, const void *weight_payload, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t pair_row = ((uint64_t)neuron * input_dimension) >> 1u;
	uint32_t pair_count = input_dimension >> 1u,pair,element;
	float accumulator = 0.0f;
	float2 pair_value;
	#pragma unroll 4
	for (pair = lane; pair < pair_count; pair += SPARK_LM_WARP_LANES)
	{
		pair_value = SparkLmLoadBf16Pair(weight_payload,pair_row + pair);
		accumulator = fmaf(shared_input[pair << 1u],pair_value.x,accumulator);
		accumulator = fmaf(shared_input[(pair << 1u) + 1u],pair_value.y,accumulator);
	}
	for (element = (pair_count << 1u) + lane; element < input_dimension; element += SPARK_LM_WARP_LANES)
		accumulator += (shared_input[element] * SparkLmBf16ToFloat(weight_payload,((uint64_t)neuron * input_dimension) + element));
	return(accumulator);
}

static __device__ void SparkLmArgmaxReduce(float running_best, uint32_t running_candidate, float *best_score, uint32_t *best_candidate)
{
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,offset,candidate,shuffle_candidate;
	float shuffle_score;
	for (offset = SPARK_LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
	{
		shuffle_score = __shfl_down_sync(0xffffffffu,running_best,offset);
		shuffle_candidate = __shfl_down_sync(0xffffffffu,running_candidate,offset);
		if ( shuffle_score > running_best || (shuffle_score == running_best && shuffle_candidate < running_candidate) )
		{
			running_best = shuffle_score;
			running_candidate = shuffle_candidate;
		}
	}
	if ( lane == 0u )
	{
		best_score[warp] = running_best;
		best_candidate[warp] = running_candidate;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		for (candidate = 1; candidate < SPARK_LM_CTA_WARPS; candidate++)
			if ( best_score[candidate] > best_score[0] || (best_score[candidate] == best_score[0] && best_candidate[candidate] < best_candidate[0]) )
			{
				best_score[0] = best_score[candidate];
				best_candidate[0] = best_candidate[candidate];
			}
	}
	__syncthreads();
}

static __device__ void SparkLmHeadStageHidden(const void *hidden_bf16, float *hidden_shared, uint32_t row, uint32_t hidden_dimension)
{
	uint32_t element;
	float2 pair;
	for (element=threadIdx.x; element<(hidden_dimension >> 1u); element+=blockDim.x)
	{
		pair = SparkLmLoadBf16Pair(hidden_bf16,(((uint64_t)row * hidden_dimension) >> 1u) + element);
		hidden_shared[element << 1u] = pair.x;
		hidden_shared[(element << 1u) + 1u] = pair.y;
	}
	__syncthreads();
}

#define SPARK_LM_HEAD_CERTIFIED_FP8_THREADS 1024u

#define SPARK_LM_HEAD_CERTIFIED_FP8_WARPS 32u

static __device__ __forceinline__ double SparkLmHeadWarpReduceDouble(double value)
{
	uint32_t offset;
	for (offset=SPARK_LM_WARP_LANES / 2u; offset!=0u; offset>>=1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	return(value);
}

static __device__ __forceinline__ float SparkLmHeadWarpReduceMax(float value)
{
	uint32_t offset;
	float other;
	for (offset=SPARK_LM_WARP_LANES / 2u; offset!=0u; offset>>=1u)
	{
		other = __shfl_down_sync(0xffffffffu,value,offset);
		value = other > value ? other : value;
	}
	return(value);
}

static __device__ __forceinline__ float SparkLmHeadRoundDoubleUp(double value)
{
	float rounded = (float)value;
	if ( (double)rounded < value )
		rounded = nextafterf(rounded,INFINITY);
	return(rounded);
}

static __device__ __forceinline__ double SparkLmHeadFloatGamma(uint32_t operations)
{
	double unit = 0x1p-24;
	return(((double)operations * unit) / (1.0 - ((double)operations * unit)));
}

static __device__ __forceinline__ uint32_t SparkLmHeadUint4Word(uint4 value,uint32_t index)
{
	if ( index == 0u ) return(value.x);
	if ( index == 1u ) return(value.y);
	if ( index == 2u ) return(value.z);
	return(value.w);
}

static __global__ void SparkLmHeadCertifiedFp8QuantizeKernel(const void *head_bf16,uint8_t *shadow_payload,float *shadow_scale_f32,float *cert_norm_f32,uint32_t candidate_count,uint32_t hidden_dimension)
{
	uint32_t neuron = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,group,groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	uint64_t group_base,row_base = (uint64_t)neuron * hidden_dimension;
	double delta,error_square,error_norm,shadow_norm,shadow_square,weight_norm,weight_square,cert;
	double exact_gamma = SparkLmHeadFloatGamma((hidden_dimension / SPARK_LM_WARP_LANES) + 5u),shadow_gamma = SparkLmHeadFloatGamma((2u * hidden_dimension / SPARK_LM_WARP_LANES) + 5u);
	float absmax,exact,quantized,scale;
	if ( neuron >= candidate_count ) return;
	for (group=0u; group<groups; group++)
	{
		group_base = row_base + ((uint64_t)group * SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE);
		exact = SparkLmBf16ToFloat(head_bf16,group_base + lane);
		absmax = SparkLmHeadWarpReduceMax(fabsf(exact));
		absmax = __shfl_sync(0xffffffffu,absmax,0u);
		scale = absmax > 0.0f ? absmax / LM_E4M3_MAX : 1.0f;
		shadow_payload[group_base + lane] = LmFloatToE4m3(exact / scale);
		quantized = LmE4m3ToFloat(shadow_payload[group_base + lane]) * scale;
		delta = (double)exact - (double)quantized;
		error_square = SparkLmHeadWarpReduceDouble(delta * delta);
		weight_square = SparkLmHeadWarpReduceDouble((double)exact * (double)exact);
		shadow_square = SparkLmHeadWarpReduceDouble((double)quantized * (double)quantized);
		if ( lane == 0u )
		{
			error_norm = sqrt(error_square * (1.0 + 1.0e-12));
			weight_norm = sqrt(weight_square * (1.0 + 1.0e-12));
			shadow_norm = sqrt(shadow_square * (1.0 + 1.0e-12));
			cert = (error_norm + (exact_gamma * weight_norm) + (shadow_gamma * shadow_norm)) * (1.0 + 1.0e-12);
			shadow_scale_f32[((uint64_t)neuron * groups) + group] = scale;
			cert_norm_f32[((uint64_t)neuron * groups) + group] = SparkLmHeadRoundDoubleUp(cert);
		}
	}
}

static __global__ void SparkLmHeadCertifiedHiddenNormKernel(const void *hidden_bf16,float *hidden_norm_f32,uint32_t hidden_dimension)
{
	uint32_t group = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	double value,sum;
	if ( group >= groups ) return;
	value = (double)SparkLmBf16ToFloat(hidden_bf16,((uint64_t)group * SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE) + lane);
	sum = SparkLmHeadWarpReduceDouble(value * value);
	if ( lane == 0u )
		hidden_norm_f32[group] = SparkLmHeadRoundDoubleUp(sqrt(sum * (1.0 + 1.0e-12)));
}

static __global__ void SparkLmHeadCertifiedFp8ScoreKernel(const void *hidden_bf16,const uint8_t *shadow_payload,const float *shadow_scale_f32,const float *cert_norm_f32,const float *hidden_norm_f32,float *coarse_scores,float *bounds,uint32_t candidate_count,uint32_t hidden_dimension)
{
	uint32_t neuron = (blockIdx.x * SPARK_LM_HEAD_CERTIFIED_FP8_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,group,pair,tile,tiles = hidden_dimension / 512u,word;
	uint64_t norm_base = (uint64_t)neuron * (hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE),shadow_base = ((uint64_t)neuron * hidden_dimension) / sizeof(uint4);
	float accumulator = 0.0f,bound = 0.0f,scale;
	float2 hidden_pair,shadow_pair;
	uint4 hidden_first,hidden_second,shadow_vector;
	uint2 shadow_half;
	if ( neuron >= candidate_count ) return;
	for (tile=0u; tile<tiles; tile++)
	{
		group = (tile * 16u) + (lane >> 1u);
		scale = (lane & 1u) == 0u ? __ldg(shadow_scale_f32 + norm_base + group) : 0.0f;
		scale = __shfl_sync(0xffffffffu,scale,lane & ~1u);
		shadow_vector = __ldg(((const uint4 *)shadow_payload) + shadow_base + ((uint64_t)tile * SPARK_LM_WARP_LANES) + lane);
		hidden_first = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)tile * 64u) + ((uint64_t)lane * 2u));
		hidden_second = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)tile * 64u) + ((uint64_t)lane * 2u) + 1u);
		#pragma unroll
		for (pair=0u; pair<8u; pair++)
		{
			word = SparkLmHeadUint4Word(shadow_vector,pair >> 1u);
			shadow_pair = LmE4m3PairToFloat2((uint16_t)(word >> ((pair & 1u) << 4u)));
			word = pair < 4u ? SparkLmHeadUint4Word(hidden_first,pair) : SparkLmHeadUint4Word(hidden_second,pair - 4u);
			hidden_pair = __bfloat1622float2(*(const __nv_bfloat162 *)&word);
			accumulator = fmaf(hidden_pair.x,shadow_pair.x * scale,accumulator);
			accumulator = fmaf(hidden_pair.y,shadow_pair.y * scale,accumulator);
		}
		if ( (lane & 1u) == 0u )
			bound = fmaf(__ldg(hidden_norm_f32 + group),__ldg(cert_norm_f32 + norm_base + group),bound);
	}
	if ( (hidden_dimension & 256u) != 0u )
	{
		group = (tiles * 16u) + (lane >> 2u);
		scale = __ldg(shadow_scale_f32 + norm_base + group);
		shadow_half = __ldg(((const uint2 *)shadow_payload) + (shadow_base * 2u) + ((uint64_t)tiles * 64u) + lane);
		hidden_first = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)tiles * 64u) + lane);
		#pragma unroll
		for (pair=0u; pair<4u; pair++)
		{
			word = pair < 2u ? shadow_half.x : shadow_half.y;
			shadow_pair = LmE4m3PairToFloat2((uint16_t)(word >> ((pair & 1u) << 4u)));
			word = SparkLmHeadUint4Word(hidden_first,pair);
			hidden_pair = __bfloat1622float2(*(const __nv_bfloat162 *)&word);
			accumulator = fmaf(hidden_pair.x,shadow_pair.x * scale,accumulator);
			accumulator = fmaf(hidden_pair.y,shadow_pair.y * scale,accumulator);
		}
		if ( (lane & 3u) == 0u )
			bound = fmaf(__ldg(hidden_norm_f32 + group),__ldg(cert_norm_f32 + norm_base + group),bound);
	}
	accumulator = SparkLmWarpReduceSum(accumulator);
	bound = SparkLmWarpReduceSum(bound);
	if ( lane == 0u )
	{
		coarse_scores[neuron] = accumulator;
		bound *= 1.0f + (float)SparkLmHeadFloatGamma(tiles + ((hidden_dimension & 256u) != 0u ? 1u : 0u) + 5u);
		bounds[neuron] = nextafterf(bound,INFINITY);
	}
}

#define SPARK_LM_HEAD_CERTIFIED_FP8_ROWS 8u

static __global__ void SparkLmHeadCertifiedFp8ScoreRowsKernel(const void *hidden_bf16,const uint8_t *shadow_payload,const float *shadow_scale_f32,const float *cert_norm_f32,const float *hidden_norm_f32,float *coarse_scores,float *bounds,uint64_t row_stride,uint32_t rows,uint32_t candidate_count,uint32_t hidden_dimension)
{
	uint32_t neuron = (blockIdx.x * SPARK_LM_HEAD_CERTIFIED_FP8_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,group,pair,tile,tiles = hidden_dimension / 512u,word,row;
	uint64_t norm_base = (uint64_t)neuron * (hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE),shadow_base = ((uint64_t)neuron * hidden_dimension) / sizeof(uint4);
	float accumulator[SPARK_LM_HEAD_CERTIFIED_FP8_ROWS],bound[SPARK_LM_HEAD_CERTIFIED_FP8_ROWS],scale,cert;
	float2 hidden_pair,shadow_pair[8];
	uint4 hidden_first,hidden_second,shadow_vector;
	uint2 shadow_half;
	if ( neuron >= candidate_count ) return;
	#pragma unroll
	for (row=0u; row<SPARK_LM_HEAD_CERTIFIED_FP8_ROWS; row++)
	{
		accumulator[row] = 0.0f;
		bound[row] = 0.0f;
	}
	for (tile=0u; tile<tiles; tile++)
	{
		group = (tile * 16u) + (lane >> 1u);
		scale = (lane & 1u) == 0u ? __ldg(shadow_scale_f32 + norm_base + group) : 0.0f;
		scale = __shfl_sync(0xffffffffu,scale,lane & ~1u);
		cert = (lane & 1u) == 0u ? __ldg(cert_norm_f32 + norm_base + group) : 0.0f;
		shadow_vector = __ldg(((const uint4 *)shadow_payload) + shadow_base + ((uint64_t)tile * SPARK_LM_WARP_LANES) + lane);
		#pragma unroll
		for (pair=0u; pair<8u; pair++)
		{
			word = SparkLmHeadUint4Word(shadow_vector,pair >> 1u);
			shadow_pair[pair] = LmE4m3PairToFloat2((uint16_t)(word >> ((pair & 1u) << 4u)));
		}
		#pragma unroll
		for (row=0u; row<SPARK_LM_HEAD_CERTIFIED_FP8_ROWS; row++)
		{
			if ( row >= rows ) break;
			hidden_first = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)row * hidden_dimension) / 8u + ((uint64_t)tile * 64u) + ((uint64_t)lane * 2u));
			hidden_second = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)row * hidden_dimension) / 8u + ((uint64_t)tile * 64u) + ((uint64_t)lane * 2u) + 1u);
			#pragma unroll
			for (pair=0u; pair<8u; pair++)
			{
				word = pair < 4u ? SparkLmHeadUint4Word(hidden_first,pair) : SparkLmHeadUint4Word(hidden_second,pair - 4u);
				hidden_pair = __bfloat1622float2(*(const __nv_bfloat162 *)&word);
				accumulator[row] = fmaf(hidden_pair.x,shadow_pair[pair].x * scale,accumulator[row]);
				accumulator[row] = fmaf(hidden_pair.y,shadow_pair[pair].y * scale,accumulator[row]);
			}
			if ( (lane & 1u) == 0u )
				bound[row] = fmaf(__ldg(hidden_norm_f32 + (uint64_t)row * row_stride + group),cert,bound[row]);
		}
	}
	if ( (hidden_dimension & 256u) != 0u )
	{
		group = (tiles * 16u) + (lane >> 2u);
		scale = __ldg(shadow_scale_f32 + norm_base + group);
		cert = __ldg(cert_norm_f32 + norm_base + group);
		shadow_half = __ldg(((const uint2 *)shadow_payload) + (shadow_base * 2u) + ((uint64_t)tiles * 64u) + lane);
		#pragma unroll
		for (pair=0u; pair<4u; pair++)
		{
			word = pair < 2u ? shadow_half.x : shadow_half.y;
			shadow_pair[pair] = LmE4m3PairToFloat2((uint16_t)(word >> ((pair & 1u) << 4u)));
		}
		#pragma unroll
		for (row=0u; row<SPARK_LM_HEAD_CERTIFIED_FP8_ROWS; row++)
		{
			if ( row >= rows ) break;
			hidden_first = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)row * hidden_dimension) / 8u + ((uint64_t)tiles * 64u) + lane);
			#pragma unroll
			for (pair=0u; pair<4u; pair++)
			{
				word = SparkLmHeadUint4Word(hidden_first,pair);
				hidden_pair = __bfloat1622float2(*(const __nv_bfloat162 *)&word);
				accumulator[row] = fmaf(hidden_pair.x,shadow_pair[pair].x * scale,accumulator[row]);
				accumulator[row] = fmaf(hidden_pair.y,shadow_pair[pair].y * scale,accumulator[row]);
			}
			if ( (lane & 3u) == 0u )
				bound[row] = fmaf(__ldg(hidden_norm_f32 + (uint64_t)row * row_stride + group),cert,bound[row]);
		}
	}
	#pragma unroll
	for (row=0u; row<SPARK_LM_HEAD_CERTIFIED_FP8_ROWS; row++)
	{
		if ( row >= rows ) break;
		accumulator[row] = SparkLmWarpReduceSum(accumulator[row]);
		bound[row] = SparkLmWarpReduceSum(bound[row]);
		if ( lane == 0u )
		{
			coarse_scores[(uint64_t)row * row_stride + neuron] = accumulator[row];
			bound[row] *= 1.0f + (float)SparkLmHeadFloatGamma(tiles + ((hidden_dimension & 256u) != 0u ? 1u : 0u) + 5u);
			bounds[(uint64_t)row * row_stride + neuron] = nextafterf(bound[row],INFINITY);
		}
	}
}

static __global__ void SparkLmHeadCertifiedScreenKernel(const float *coarse_scores,const float *bounds,uint32_t *candidate_ids,uint32_t *candidate_count,uint32_t vocabulary_count)
{
	__shared__ float scratch[SPARK_LM_CTA_WARPS];
	__shared__ float best_lower;
	__shared__ uint32_t cursor;
	uint32_t candidate,lane = threadIdx.x % SPARK_LM_WARP_LANES,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	float lower = -3.0e38f,other,upper;
	for (candidate=threadIdx.x; candidate<vocabulary_count; candidate+=blockDim.x)
	{
		other = nextafterf(coarse_scores[candidate] - bounds[candidate],-INFINITY);
		lower = other > lower ? other : lower;
	}
	lower = SparkLmHeadWarpReduceMax(lower);
	if ( lane == 0u ) scratch[warp] = lower;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		best_lower = scratch[0];
		for (candidate=1u; candidate<SPARK_LM_CTA_WARPS; candidate++) best_lower = scratch[candidate] > best_lower ? scratch[candidate] : best_lower;
		cursor = 0u;
	}
	__syncthreads();
	for (candidate=threadIdx.x; candidate<vocabulary_count; candidate+=blockDim.x)
	{
		upper = nextafterf(coarse_scores[candidate] + bounds[candidate],INFINITY);
		if ( upper >= best_lower ) candidate_ids[atomicAdd(&cursor,1u)] = candidate;
	}
	__syncthreads();
	if ( threadIdx.x == 0u ) *candidate_count = cursor;
}

static __global__ void SparkLmHeadCertifiedRescoreKernel(const void *hidden_bf16,const void *head_weight_bf16,const uint32_t *candidate_ids,const uint32_t *candidate_count,float *partial_scores,uint32_t *partial_candidates,uint32_t hidden_dimension)
{
	extern __shared__ float hidden_shared[];
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t count = *candidate_count,lane = threadIdx.x % SPARK_LM_WARP_LANES,slot = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),candidate,running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	SparkLmHeadStageHidden(hidden_bf16,hidden_shared,0u,hidden_dimension);
	for (; slot<count; slot+=SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT * SPARK_LM_CTA_WARPS)
	{
		candidate = candidate_ids[slot];
		score = SparkLmWarpReduceSum(SparkLmDotRowBf16(hidden_shared,head_weight_bf16,candidate,hidden_dimension,lane));
		score = __shfl_sync(0xffffffffu,score,0u);
		if ( lane == 0u && (score > running_best || (score == running_best && candidate < running_candidate)) )
		{
			running_best = score;
			running_candidate = candidate;
		}
	}
	SparkLmArgmaxReduce(running_best,running_candidate,best_score,best_candidate);
	if ( threadIdx.x == 0u )
	{
		partial_scores[blockIdx.x] = best_score[0];
		partial_candidates[blockIdx.x] = best_candidate[0];
	}
}

static __global__ void SparkLmHeadCertifiedReduceKernel(const float *partial_scores,const uint32_t *partial_candidates,uint32_t *output_token_id,float *output_score,uint32_t candidate_offset)
{
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t partial,running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	for (partial=threadIdx.x; partial<SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT; partial+=blockDim.x)
	{
		score = partial_scores[partial];
		if ( score > running_best || (score == running_best && partial_candidates[partial] < running_candidate) )
		{
			running_best = score;
			running_candidate = partial_candidates[partial];
		}
	}
	SparkLmArgmaxReduce(running_best,running_candidate,best_score,best_candidate);
	if ( threadIdx.x == 0u )
	{
		*output_token_id = best_candidate[0] + candidate_offset;
		*output_score = best_score[0];
	}
}

#include <mma.h>

static inline cudaError_t SparkLmHostLaunchHeadCertifiedFp8Quantize(
	cudaStream_t stream,const void *head_bf16,uint8_t *shadow_payload,
	float *shadow_scale_f32,float *cert_norm_f32,uint32_t candidate_count,
	uint32_t hidden_dimension)
{
	uint32_t blocks;
	if ( head_bf16 == 0 || shadow_payload == 0 || shadow_scale_f32 == 0 ||
		cert_norm_f32 == 0 || candidate_count == 0u || hidden_dimension == 0u ||
		(hidden_dimension % SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE) != 0u )
		return(cudaErrorInvalidValue);
	blocks = (candidate_count + SPARK_LM_CTA_WARPS - 1u) /
		SPARK_LM_CTA_WARPS;
	SPARK_LM_LAUNCH((
	SparkLmHeadCertifiedFp8QuantizeKernel<<<blocks,SPARK_LM_CTA_THREADS,0u,
			stream>>>(head_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,
			candidate_count,hidden_dimension)
	));
	return(cudaGetLastError());
}

typedef struct SparkLmHeadCertifiedFp8Scratch
{
	float *coarse_scores;
	float *bounds;
	float *hidden_norms;
	float *partial_scores;
	uint32_t *partial_candidates;
} SparkLmHeadCertifiedFp8Scratch;

static inline SparkLmHeadCertifiedFp8Scratch SparkLmHeadCertifiedFp8ScratchView(
	void *scratch,uint32_t vocabulary_count,uint32_t hidden_dimension)
{
	SparkLmHeadCertifiedFp8Scratch view;
	view.coarse_scores = (float *)scratch;
	view.bounds = view.coarse_scores + vocabulary_count;
	view.hidden_norms = view.bounds + vocabulary_count;
	view.partial_scores = view.hidden_norms +
		(hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE);
	view.partial_candidates = (uint32_t *)(view.partial_scores +
		SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT);
	return(view);
}

static inline cudaError_t SparkLmHostLaunchHeadCertifiedFp8B1WithScore(
	cudaStream_t stream,const void *hidden_bf16,const void *head_weight_bf16,
	const uint8_t *shadow_payload,const float *shadow_scale_f32,
	const float *cert_norm_f32,void *scratch,uint32_t *candidate_ids,
	uint32_t *candidate_count,uint32_t *output_token_id,float *output_score,
	uint32_t candidate_offset,uint32_t row_count,uint32_t vocabulary_count,
	uint32_t hidden_dimension)
{
	uint32_t groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	uint32_t norm_blocks = (groups + SPARK_LM_CTA_WARPS - 1u) /
		SPARK_LM_CTA_WARPS;
	uint32_t score_blocks = (vocabulary_count +
		SPARK_LM_HEAD_CERTIFIED_FP8_WARPS - 1u) /
		SPARK_LM_HEAD_CERTIFIED_FP8_WARPS;
	SparkLmHeadCertifiedFp8Scratch view;
	if ( hidden_bf16 == 0 || head_weight_bf16 == 0 || shadow_payload == 0 ||
		shadow_scale_f32 == 0 || cert_norm_f32 == 0 || scratch == 0 ||
		candidate_ids == 0 || candidate_count == 0 || output_token_id == 0 ||
		output_score == 0 || row_count != 1u || vocabulary_count == 0u ||
		hidden_dimension == 0u || (hidden_dimension % 256u) != 0u )
		return(cudaErrorInvalidValue);
	view = SparkLmHeadCertifiedFp8ScratchView(scratch,vocabulary_count,
		hidden_dimension);
	SPARK_LM_LAUNCH((
	SparkLmHeadCertifiedHiddenNormKernel<<<norm_blocks,SPARK_LM_CTA_THREADS,0u,
			stream>>>(hidden_bf16,view.hidden_norms,hidden_dimension)
	));
	SPARK_LM_LAUNCH((
	SparkLmHeadCertifiedFp8ScoreKernel<<<score_blocks,
			SPARK_LM_HEAD_CERTIFIED_FP8_THREADS,0u,stream>>>(hidden_bf16,
			shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
			view.coarse_scores,view.bounds,vocabulary_count,hidden_dimension)
	));
	SPARK_LM_LAUNCH((
	SparkLmHeadCertifiedScreenKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
			view.coarse_scores,view.bounds,candidate_ids,candidate_count,
			vocabulary_count)
	));
	SPARK_LM_LAUNCH((
	SparkLmHeadCertifiedRescoreKernel<<<SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT,
			SPARK_LM_CTA_THREADS,hidden_dimension * sizeof(float),stream>>>(
			hidden_bf16,head_weight_bf16,candidate_ids,candidate_count,
			view.partial_scores,view.partial_candidates,hidden_dimension)
	));
	SPARK_LM_LAUNCH((
	SparkLmHeadCertifiedReduceKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
			view.partial_scores,view.partial_candidates,output_token_id,output_score,
			candidate_offset)
	));
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(
	cudaStream_t stream,const void *hidden_bf16,const void *head_weight_bf16,
	const uint8_t *shadow_payload,const float *shadow_scale_f32,
	const float *cert_norm_f32,void *scratch,uint32_t *candidate_ids,
	uint32_t *candidate_count,uint32_t *output_token_id,float *output_score,
	uint32_t candidate_offset,uint32_t row_count,uint32_t vocabulary_count,
	uint32_t hidden_dimension)
{
	uint32_t groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	uint32_t norm_blocks = (groups + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS;
	uint32_t score_blocks = (vocabulary_count + SPARK_LM_HEAD_CERTIFIED_FP8_WARPS - 1u) /
		SPARK_LM_HEAD_CERTIFIED_FP8_WARPS;
	uint64_t row_floats = SparkHeadCertifiedFp8ScratchBytes(vocabulary_count,hidden_dimension) / sizeof(float);
	uint32_t first,rows,row;
	SparkLmHeadCertifiedFp8Scratch view;
	if ( row_count == 1u )
		return(SparkLmHostLaunchHeadCertifiedFp8B1WithScore(stream,hidden_bf16,head_weight_bf16,shadow_payload,
			shadow_scale_f32,cert_norm_f32,scratch,candidate_ids,candidate_count,output_token_id,output_score,
			candidate_offset,1u,vocabulary_count,hidden_dimension));
	if ( hidden_bf16 == 0 || head_weight_bf16 == 0 || shadow_payload == 0 ||
		shadow_scale_f32 == 0 || cert_norm_f32 == 0 || scratch == 0 ||
		candidate_ids == 0 || candidate_count == 0 || output_token_id == 0 ||
		output_score == 0 || row_count == 0u || vocabulary_count == 0u ||
		hidden_dimension == 0u || (hidden_dimension % 256u) != 0u )
		return(cudaErrorInvalidValue);
	for (row=0u; row<row_count; row++)
	{
		view = SparkLmHeadCertifiedFp8ScratchView((float *)scratch + row * row_floats,vocabulary_count,hidden_dimension);
		SPARK_LM_LAUNCH((
		SparkLmHeadCertifiedHiddenNormKernel<<<norm_blocks,SPARK_LM_CTA_THREADS,0u,
				stream>>>((const uint16_t *)hidden_bf16 + (uint64_t)row * hidden_dimension,view.hidden_norms,hidden_dimension)
		));
	}
	for (first=0u; first<row_count; first+=rows)
	{
		rows = row_count - first < SPARK_LM_HEAD_CERTIFIED_FP8_ROWS ? row_count - first : SPARK_LM_HEAD_CERTIFIED_FP8_ROWS;
		view = SparkLmHeadCertifiedFp8ScratchView((float *)scratch + first * row_floats,vocabulary_count,hidden_dimension);
		SPARK_LM_LAUNCH((
		SparkLmHeadCertifiedFp8ScoreRowsKernel<<<score_blocks,
				SPARK_LM_HEAD_CERTIFIED_FP8_THREADS,0u,stream>>>((const uint16_t *)hidden_bf16 + (uint64_t)first * hidden_dimension,
				shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
				view.coarse_scores,view.bounds,row_floats,rows,vocabulary_count,hidden_dimension)
		));
	}
	for (row=0u; row<row_count; row++)
	{
		const void *hidden_row = (const uint16_t *)hidden_bf16 + (uint64_t)row * hidden_dimension;
		uint32_t *row_candidates = candidate_ids + (uint64_t)row * vocabulary_count;
		view = SparkLmHeadCertifiedFp8ScratchView((float *)scratch + row * row_floats,vocabulary_count,hidden_dimension);
		SPARK_LM_LAUNCH((
		SparkLmHeadCertifiedScreenKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
				view.coarse_scores,view.bounds,row_candidates,candidate_count + row,
				vocabulary_count)
		));
		SPARK_LM_LAUNCH((
		SparkLmHeadCertifiedRescoreKernel<<<SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT,
				SPARK_LM_CTA_THREADS,hidden_dimension * sizeof(float),stream>>>(
				hidden_row,head_weight_bf16,row_candidates,candidate_count + row,
				view.partial_scores,view.partial_candidates,hidden_dimension)
		));
		SPARK_LM_LAUNCH((
		SparkLmHeadCertifiedReduceKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
				view.partial_scores,view.partial_candidates,output_token_id + row,output_score + row,
				candidate_offset)
		));
	}
	return(cudaGetLastError());
}
