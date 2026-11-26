#pragma once
#include <stdio.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "inference/kernels/activation.cuh"
#include "inference/kernels/vec.cuh"
#include "sparkpipe/spark_head_screen.h"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"
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

static __device__ __forceinline__ float SparkLmHeadCertifiedNorm(double error_square,double weight_square,double shadow_square,double exact_gamma,double shadow_gamma)
{
	return(SparkLmHeadRoundDoubleUp((sqrt(error_square * (1.0 + 1.0e-12)) + (exact_gamma * sqrt(weight_square * (1.0 + 1.0e-12))) + (shadow_gamma * sqrt(shadow_square * (1.0 + 1.0e-12)))) * (1.0 + 1.0e-12)));
}

static __global__ void SparkLmHeadCertifiedFp8QuantizeKernel(const void *head_bf16,uint8_t *shadow_payload,float *shadow_scale_f32,float *cert_norm_f32,uint32_t candidate_count,uint32_t hidden_dimension)
{
	uint32_t neuron = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,group = 0u,groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	uint64_t group_base,row_base = (uint64_t)neuron * hidden_dimension;
	double delta,error_square,shadow_square,weight_square;
	double exact_gamma = SparkLmHeadFloatGamma((hidden_dimension / SPARK_LM_WARP_LANES) + 5u),shadow_gamma = SparkLmHeadFloatGamma((2u * hidden_dimension / SPARK_LM_WARP_LANES) + 5u);
	float absmax,exact,quantized,scale;
	if ( neuron >= candidate_count ) return;
	if ( LmIsAligned16(head_bf16) && LmIsAligned8(shadow_payload) )
		for (; group + 8u <= groups; group += 8u)
		{
			uint64_t vec_index = ((row_base + ((uint64_t)group * SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE)) >> 3u) + lane;
			float exact_vec[8],other;
			uint32_t packed_lo = 0u,packed_hi = 0u,k;
			LmLoadBf16x8AsFloat(__ldg(((const uint4 *)head_bf16) + vec_index),exact_vec);
			absmax = 0.0f;
			#pragma unroll
			for (k=0u; k<8u; k++) { float m = fabsf(exact_vec[k]); absmax = m > absmax ? m : absmax; }
			other = __shfl_xor_sync(0xffffffffu,absmax,2u,4u); absmax = other > absmax ? other : absmax;
			other = __shfl_xor_sync(0xffffffffu,absmax,1u,4u); absmax = other > absmax ? other : absmax;
			scale = absmax > 0.0f ? absmax / LM_E4M3_MAX : 1.0f;
			error_square = 0.0; weight_square = 0.0; shadow_square = 0.0;
			#pragma unroll
			for (k=0u; k<8u; k++)
			{
				uint8_t code = LmFloatToE4m3(exact_vec[k] / scale);
				if ( k < 4u ) packed_lo |= (uint32_t)code << (k << 3u); else packed_hi |= (uint32_t)code << ((k - 4u) << 3u);
				quantized = LmE4m3ToFloat(code) * scale;
				delta = (double)exact_vec[k] - (double)quantized;
				error_square += delta * delta;
				weight_square += (double)exact_vec[k] * (double)exact_vec[k];
				shadow_square += (double)quantized * (double)quantized;
			}
			((uint2 *)shadow_payload)[vec_index] = make_uint2(packed_lo,packed_hi);
			error_square += __shfl_down_sync(0xffffffffu,error_square,2u,4u); error_square += __shfl_down_sync(0xffffffffu,error_square,1u,4u);
			weight_square += __shfl_down_sync(0xffffffffu,weight_square,2u,4u); weight_square += __shfl_down_sync(0xffffffffu,weight_square,1u,4u);
			shadow_square += __shfl_down_sync(0xffffffffu,shadow_square,2u,4u); shadow_square += __shfl_down_sync(0xffffffffu,shadow_square,1u,4u);
			if ( (lane & 3u) == 0u )
			{
				uint64_t target = ((uint64_t)neuron * groups) + group + (lane >> 2u);
				shadow_scale_f32[target] = scale;
				cert_norm_f32[target] = SparkLmHeadCertifiedNorm(error_square,weight_square,shadow_square,exact_gamma,shadow_gamma);
			}
		}
	for (; group<groups; group++)
	{
		uint8_t code;
		group_base = row_base + ((uint64_t)group * SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE);
		exact = SparkLmBf16ToFloat(head_bf16,group_base + lane);
		absmax = __shfl_sync(0xffffffffu,SparkLmHeadWarpReduceMax(fabsf(exact)),0u);
		scale = absmax > 0.0f ? absmax / LM_E4M3_MAX : 1.0f;
		code = LmFloatToE4m3(exact / scale);
		shadow_payload[group_base + lane] = code;
		quantized = LmE4m3ToFloat(code) * scale;
		delta = (double)exact - (double)quantized;
		error_square = SparkLmHeadWarpReduceDouble(delta * delta);
		weight_square = SparkLmHeadWarpReduceDouble((double)exact * (double)exact);
		shadow_square = SparkLmHeadWarpReduceDouble((double)quantized * (double)quantized);
		if ( lane == 0u )
		{
			uint64_t target = ((uint64_t)neuron * groups) + group;
			shadow_scale_f32[target] = scale;
			cert_norm_f32[target] = SparkLmHeadCertifiedNorm(error_square,weight_square,shadow_square,exact_gamma,shadow_gamma);
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

static inline SparkGpuError SparkLmHostLaunchHeadCertifiedFp8Quantize(
	SparkGpuStream stream,const void *head_bf16,uint8_t *shadow_payload,
	float *shadow_scale_f32,float *cert_norm_f32,uint32_t candidate_count,
	uint32_t hidden_dimension)
{
	uint32_t blocks;
	if ( head_bf16 == 0 || shadow_payload == 0 || shadow_scale_f32 == 0 ||
		cert_norm_f32 == 0 || candidate_count == 0u || hidden_dimension == 0u ||
		(hidden_dimension % SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE) != 0u )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	blocks = (candidate_count + SPARK_LM_CTA_WARPS - 1u) /
		SPARK_LM_CTA_WARPS;
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedFp8QuantizeKernel), blocks,SPARK_LM_CTA_THREADS,0u,
			stream, head_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,
			candidate_count,hidden_dimension)
	));
	return(SparkGpuGetLastError());
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

static inline SparkGpuError SparkLmHostLaunchHeadCertifiedFp8B1WithScore(
	SparkGpuStream stream,const void *hidden_bf16,const void *head_weight_bf16,
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
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	view = SparkLmHeadCertifiedFp8ScratchView(scratch,vocabulary_count,
		hidden_dimension);
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedHiddenNormKernel), norm_blocks,SPARK_LM_CTA_THREADS,0u,
			stream, hidden_bf16,view.hidden_norms,hidden_dimension)
	));
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedFp8ScoreKernel), score_blocks,
			SPARK_LM_HEAD_CERTIFIED_FP8_THREADS,0u,stream, hidden_bf16,
			shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
			view.coarse_scores,view.bounds,vocabulary_count,hidden_dimension)
	));
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedScreenKernel),1u,SPARK_LM_CTA_THREADS,0u,stream,
			view.coarse_scores,view.bounds,candidate_ids,candidate_count,
			vocabulary_count)
	));
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedRescoreKernel), SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT,
			SPARK_LM_CTA_THREADS,hidden_dimension * sizeof(float),stream,
			hidden_bf16,head_weight_bf16,candidate_ids,candidate_count,
			view.partial_scores,view.partial_candidates,hidden_dimension)
	));
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedReduceKernel),1u,SPARK_LM_CTA_THREADS,0u,stream,
			view.partial_scores,view.partial_candidates,output_token_id,output_score,
			candidate_offset)
	));
	return(SparkGpuGetLastError());
}

static __device__ __forceinline__ uint32_t SparkLmHeadCertifiedE4m3x2ToBf16x2(uint16_t pair)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	uint32_t f16x2,lo,hi,lo_b,hi_b;
	asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(pair));
	lo = f16x2 & 0xffffu;
	hi = f16x2 >> 16u;
	lo_b = (lo & 0x7fffu) == 0u ? lo : ((lo & 0x8000u) | (((lo & 0x7fffu) >> 3u) + 0x3800u));
	hi_b = (hi & 0x7fffu) == 0u ? hi : ((hi & 0x8000u) | (((hi & 0x7fffu) >> 3u) + 0x3800u));
	return(lo_b | (hi_b << 16u));
#else
	float2 f = LmE4m3PairToFloat2(pair);
	return(LmPackBf16Pair(f.x,f.y));
#endif
}

static __device__ __forceinline__ uint32_t SparkLmHeadCertifiedSwizzle8(uint32_t row,uint32_t chunk)
{
	return(chunk ^ (row & 7u));
}

static __global__ void SparkLmHeadCertifiedHiddenNormBatchedKernel(const void *hidden_bf16,float *hidden_norm_f32,uint64_t row_stride,uint32_t hidden_dimension,uint32_t rows)
{
	uint32_t row = blockIdx.y;
	uint32_t group = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	double value,sum;
	if ( row >= rows || group >= groups ) return;
	value = (double)SparkLmBf16ToFloat(hidden_bf16,((uint64_t)row * hidden_dimension) + ((uint64_t)group * SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE) + lane);
	sum = SparkLmHeadWarpReduceDouble(value * value);
	if ( lane == 0u )
		hidden_norm_f32[(uint64_t)row * row_stride + group] = SparkLmHeadRoundDoubleUp(sqrt(sum * (1.0 + 1.0e-12)));
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M>
__global__ __launch_bounds__(THREADS, 2)
static void SparkLmHeadCertifiedFp8ScoreMmaKernel(const uint16_t *__restrict__ hidden_bf16,const uint8_t *__restrict__ shadow_payload,const float *__restrict__ shadow_scale_f32,const float *__restrict__ cert_norm_f32,const float *__restrict__ hidden_norm_f32,float *__restrict__ coarse_scores,float *__restrict__ bounds,uint64_t row_stride,uint32_t rows,uint32_t candidate_count,uint32_t hidden_dimension)
{
	constexpr uint32_t TILE_K = 64u;
	constexpr uint32_t GROUPS_PER_TILE = 2u;
	constexpr uint32_t A_ROW_BYTES = 128u;
	constexpr uint32_t B_ROW_BYTES = 64u;
	constexpr uint32_t WARPS = THREADS / 32u;
	constexpr uint32_t WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M;
	constexpr uint32_t WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u;
	constexpr uint32_t NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES;
	constexpr uint32_t B_BYTES = TILE_N * B_ROW_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * 8u;
	constexpr uint32_t B_CHUNKS = TILE_N * 4u;
	__shared__ __align__(128) uint8_t smem_a[2][A_BYTES];
	__shared__ __align__(128) uint8_t smem_b[2][B_BYTES];
	__shared__ __align__(16) float smem_sc[2][TILE_N][GROUPS_PER_TILE];
	__shared__ __align__(16) float smem_cn[2][TILE_N][GROUPS_PER_TILE];
	__shared__ __align__(16) float smem_hn[2][TILE_M][GROUPS_PER_TILE];
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks;
	const uint32_t m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M;
	const uint32_t col0 = n_block * TILE_N;
	if ( row0 >= rows || col0 >= candidate_count ) return;
	const uint32_t warp = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t quad_row = lane >> 2u;
	const uint32_t quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N;
	const uint32_t warp_n = warp % WARPS_N;
	const uint32_t k_tiles = hidden_dimension / TILE_K;
	const uint32_t total_groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	const uint32_t a_base0 = (uint32_t)__cvta_generic_to_shared(smem_a[0]);
	const uint32_t a_base1 = (uint32_t)__cvta_generic_to_shared(smem_a[1]);
	const uint32_t b_base0 = (uint32_t)__cvta_generic_to_shared(smem_b[0]);
	const uint32_t b_base1 = (uint32_t)__cvta_generic_to_shared(smem_b[1]);
	float acc[MF][NF][4],bnd[MF][NF][4];
	#pragma unroll
	for (uint32_t i = 0u; i < MF; ++i)
		#pragma unroll
		for (uint32_t j = 0u; j < NF; ++j)
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] =
			bnd[i][j][0] = bnd[i][j][1] = bnd[i][j][2] = bnd[i][j][3] = 0.0f;
	auto issue_stage = [&](uint32_t st,uint32_t kt) {
		const uint32_t ab = st != 0u ? a_base1 : a_base0;
		const uint32_t bb = st != 0u ? b_base1 : b_base0;
		#pragma unroll
		for (uint32_t idx = threadIdx.x; idx < A_CHUNKS; idx += THREADS)
		{
			const uint32_t r = idx >> 3u,c = idx & 7u;
			const uint32_t dst = ab + r * A_ROW_BYTES + SparkLmHeadCertifiedSwizzle8(r,c) * 16u;
			if ( row0 + r < rows )
			{
				const uint16_t *src = hidden_bf16 + (uint64_t)(row0 + r) * hidden_dimension + kt * TILE_K + c * 8u;
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
			}
			else
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" :: "r"(dst), "l"(hidden_bf16) : "memory");
		}
		#pragma unroll
		for (uint32_t idx = threadIdx.x; idx < B_CHUNKS; idx += THREADS)
		{
			const uint32_t n = idx >> 2u,c = idx & 3u;
			const uint32_t dst = bb + n * B_ROW_BYTES + c * 16u;
			if ( col0 + n < candidate_count )
			{
				const uint8_t *src = shadow_payload + (uint64_t)(col0 + n) * hidden_dimension + kt * TILE_K + c * 16u;
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
			}
			else
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" :: "r"(dst), "l"(shadow_payload) : "memory");
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
		const uint32_t g0 = kt * 2u;
		for (uint32_t n = threadIdx.x; n < TILE_N; n += THREADS)
		{
			if ( col0 + n < candidate_count )
			{
				const uint64_t off = (uint64_t)(col0 + n) * total_groups + g0;
				const float2 sc2 = *(const float2 *)(shadow_scale_f32 + off);
				const float2 cn2 = *(const float2 *)(cert_norm_f32 + off);
				smem_sc[st][n][0] = sc2.x;
				smem_sc[st][n][1] = sc2.y;
				smem_cn[st][n][0] = cn2.x;
				smem_cn[st][n][1] = cn2.y;
			}
			else
			{
				smem_sc[st][n][0] = smem_sc[st][n][1] = 0.0f;
				smem_cn[st][n][0] = smem_cn[st][n][1] = 0.0f;
			}
		}
		for (uint32_t r = threadIdx.x; r < TILE_M; r += THREADS)
		{
			if ( row0 + r < rows )
			{
				const uint64_t hoff = (uint64_t)(row0 + r) * row_stride + g0;
				smem_hn[st][r][0] = __ldg(hidden_norm_f32 + hoff);
				smem_hn[st][r][1] = __ldg(hidden_norm_f32 + hoff + 1u);
			}
			else
				smem_hn[st][r][0] = smem_hn[st][r][1] = 0.0f;
		}
	};
	issue_stage(0u,0u);
	uint32_t cur = 0u;
	for (uint32_t kt = 0u; kt < k_tiles; ++kt)
	{
		if ( kt + 1u < k_tiles )
		{
			issue_stage(cur ^ 1u,kt + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		const uint32_t ab = cur != 0u ? a_base1 : a_base0;
		const uint32_t bb = cur != 0u ? b_base1 : b_base0;
		#pragma unroll
		for (uint32_t g = 0u; g < 2u; ++g)
		{
			float gdot[MF][NF][4];
			#pragma unroll
			for (uint32_t i = 0u; i < MF; ++i)
				#pragma unroll
				for (uint32_t j = 0u; j < NF; ++j)
					gdot[i][j][0] = gdot[i][j][1] = gdot[i][j][2] = gdot[i][j][3] = 0.0f;
			#pragma unroll
			for (uint32_t ks = 0u; ks < 2u; ++ks)
			{
				const uint32_t kstep = g * 2u + ks;
				uint32_t a_frag[MF][4];
				#pragma unroll
				for (uint32_t i = 0u; i < MF; ++i)
				{
					const uint32_t a_row = warp_m * WARP_M + i * 16u + (lane & 15u);
					const uint32_t a_col = kstep * 2u + ((lane >> 4u) & 1u);
					const uint32_t a_addr = ab + a_row * A_ROW_BYTES + SparkLmHeadCertifiedSwizzle8(a_row,a_col) * 16u;
					asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
						: "=r"(a_frag[i][0]), "=r"(a_frag[i][1]), "=r"(a_frag[i][2]), "=r"(a_frag[i][3]) : "r"(a_addr));
				}
				#pragma unroll
				for (uint32_t j = 0u; j < NF; ++j)
				{
					const uint32_t n_local = warp_n * WARP_N + j * 8u + quad_row;
					const uint32_t b_off = bb + n_local * B_ROW_BYTES + kstep * 16u + quad * 2u;
					uint16_t raw0,raw1;
					uint32_t b_frag[2];
					asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(raw0) : "r"(b_off));
					asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(raw1) : "r"(b_off + 8u));
					b_frag[0] = SparkLmHeadCertifiedE4m3x2ToBf16x2(raw0);
					b_frag[1] = SparkLmHeadCertifiedE4m3x2ToBf16x2(raw1);
					#pragma unroll
					for (uint32_t i = 0u; i < MF; ++i)
						asm volatile(
							"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
							: "+f"(gdot[i][j][0]), "+f"(gdot[i][j][1]), "+f"(gdot[i][j][2]), "+f"(gdot[i][j][3])
							: "r"(a_frag[i][0]), "r"(a_frag[i][1]), "r"(a_frag[i][2]), "r"(a_frag[i][3]),
							  "r"(b_frag[0]), "r"(b_frag[1]));
				}
			}
			#pragma unroll
			for (uint32_t j = 0u; j < NF; ++j)
			{
				const uint32_t n0 = warp_n * WARP_N + j * 8u + quad * 2u;
				const float sc0 = smem_sc[cur][n0][g],sc1 = smem_sc[cur][n0 + 1u][g];
				const float cn0 = smem_cn[cur][n0][g],cn1 = smem_cn[cur][n0 + 1u][g];
				#pragma unroll
				for (uint32_t i = 0u; i < MF; ++i)
				{
					const uint32_t r0 = warp_m * WARP_M + i * 16u + quad_row;
					const uint32_t r1 = r0 + 8u;
					const float hn0 = smem_hn[cur][r0][g],hn1 = smem_hn[cur][r1][g];
					acc[i][j][0] = fmaf(gdot[i][j][0],sc0,acc[i][j][0]);
					acc[i][j][1] = fmaf(gdot[i][j][1],sc1,acc[i][j][1]);
					acc[i][j][2] = fmaf(gdot[i][j][2],sc0,acc[i][j][2]);
					acc[i][j][3] = fmaf(gdot[i][j][3],sc1,acc[i][j][3]);
					bnd[i][j][0] = fmaf(cn0,hn0,bnd[i][j][0]);
					bnd[i][j][1] = fmaf(cn1,hn0,bnd[i][j][1]);
					bnd[i][j][2] = fmaf(cn0,hn1,bnd[i][j][2]);
					bnd[i][j][3] = fmaf(cn1,hn1,bnd[i][j][3]);
				}
			}
		}
		cur ^= 1u;
		if ( kt + 1u < k_tiles ) __syncthreads();
	}
	const float bnd_scale = 1.0f + (float)SparkLmHeadFloatGamma(total_groups + 6u);
	#pragma unroll
	for (uint32_t i = 0u; i < MF; ++i)
	{
		const uint32_t r0 = row0 + warp_m * WARP_M + i * 16u + quad_row;
		const uint32_t r1 = r0 + 8u;
		#pragma unroll
		for (uint32_t j = 0u; j < NF; ++j)
		{
			const uint32_t c = col0 + warp_n * WARP_N + j * 8u + quad * 2u;
			const float b0 = nextafterf(bnd[i][j][0] * bnd_scale,INFINITY);
			const float b1 = nextafterf(bnd[i][j][1] * bnd_scale,INFINITY);
			const float b2 = nextafterf(bnd[i][j][2] * bnd_scale,INFINITY);
			const float b3 = nextafterf(bnd[i][j][3] * bnd_scale,INFINITY);
			if ( c + 1u < candidate_count )
			{
				if ( r0 < rows )
				{
					const uint64_t off = (uint64_t)r0 * row_stride + c;
					coarse_scores[off] = acc[i][j][0];
					coarse_scores[off + 1u] = acc[i][j][1];
					bounds[off] = b0;
					bounds[off + 1u] = b1;
				}
				if ( r1 < rows )
				{
					const uint64_t off = (uint64_t)r1 * row_stride + c;
					coarse_scores[off] = acc[i][j][2];
					coarse_scores[off + 1u] = acc[i][j][3];
					bounds[off] = b2;
					bounds[off + 1u] = b3;
				}
			}
			else if ( c < candidate_count )
			{
				if ( r0 < rows )
				{
					coarse_scores[(uint64_t)r0 * row_stride + c] = acc[i][j][0];
					bounds[(uint64_t)r0 * row_stride + c] = b0;
				}
				if ( r1 < rows )
				{
					coarse_scores[(uint64_t)r1 * row_stride + c] = acc[i][j][2];
					bounds[(uint64_t)r1 * row_stride + c] = b2;
				}
			}
		}
	}
}

__global__ __launch_bounds__(512u, 2)
static void SparkLmHeadCertifiedScreenBatchedKernel(const float *__restrict__ coarse_scores,const float *__restrict__ bounds,uint64_t row_stride,uint32_t *__restrict__ candidate_ids,uint32_t *__restrict__ candidate_count,uint32_t vocabulary_count,uint32_t rows)
{
	const uint32_t row = blockIdx.y;
	if ( row >= rows ) return;
	const float *cs = coarse_scores + (uint64_t)row * row_stride;
	const float *bn = bounds + (uint64_t)row * row_stride;
	uint32_t *out_ids = candidate_ids + (uint64_t)row * vocabulary_count;
	__shared__ float warp_low[16];
	__shared__ float best_low_s;
	__shared__ uint32_t shared_count;
	const uint32_t tid = threadIdx.x,warp = tid >> 5u,lane = tid & 31u;
	float best_low = -3.0e38f;
	const bool aligned16 = ((((uintptr_t)cs | (uintptr_t)bn) & 15u) == 0u);
	const uint32_t vec_n = aligned16 ? (vocabulary_count >> 2u) : 0u;
	const float4 *cs4 = (const float4 *)cs;
	const float4 *bn4 = (const float4 *)bn;
	for (uint32_t v = tid; v < vec_n; v += 512u)
	{
		const float4 c = cs4[v],b = bn4[v];
		const float l0 = nextafterf(c.x - b.x,-INFINITY);
		const float l1 = nextafterf(c.y - b.y,-INFINITY);
		const float l2 = nextafterf(c.z - b.z,-INFINITY);
		const float l3 = nextafterf(c.w - b.w,-INFINITY);
		best_low = fmaxf(best_low,fmaxf(fmaxf(l0,l1),fmaxf(l2,l3)));
	}
	for (uint32_t idx = (vec_n << 2u) + tid; idx < vocabulary_count; idx += 512u)
		best_low = fmaxf(best_low,nextafterf(cs[idx] - bn[idx],-INFINITY));
	#pragma unroll
	for (uint32_t off = 16u; off > 0u; off >>= 1u)
		best_low = fmaxf(best_low,__shfl_xor_sync(0xffffffffu,best_low,off));
	if ( lane == 0u ) warp_low[warp] = best_low;
	__syncthreads();
	if ( warp == 0u )
	{
		float v = lane < 16u ? warp_low[lane] : -3.0e38f;
		#pragma unroll
		for (uint32_t off = 8u; off > 0u; off >>= 1u)
			v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,off));
		if ( lane == 0u )
		{
			best_low_s = v;
			shared_count = 0u;
		}
	}
	__syncthreads();
	const float thresh = best_low_s;
	for (uint32_t base = 0u; base < vec_n; base += 512u)
	{
		const uint32_t v = base + tid;
		uint32_t flags = 0u;
		if ( v < vec_n )
		{
			const float4 c = cs4[v],b = bn4[v];
			if ( nextafterf(c.x + b.x,INFINITY) >= thresh ) flags |= 1u;
			if ( nextafterf(c.y + b.y,INFINITY) >= thresh ) flags |= 2u;
			if ( nextafterf(c.z + b.z,INFINITY) >= thresh ) flags |= 4u;
			if ( nextafterf(c.w + b.w,INFINITY) >= thresh ) flags |= 8u;
		}
		const uint32_t any_mask = __ballot_sync(0xffffffffu,flags != 0u);
		if ( any_mask != 0u )
		{
			const uint32_t my_cnt = __popc(flags);
			uint32_t prefix = my_cnt;
			#pragma unroll
			for (uint32_t off = 1u; off < 32u; off <<= 1u)
			{
				const uint32_t n = __shfl_up_sync(0xffffffffu,prefix,off);
				if ( lane >= off ) prefix += n;
			}
			const uint32_t warp_tot = __shfl_sync(0xffffffffu,prefix,31u);
			uint32_t warp_base = 0u;
			if ( lane == 0u ) warp_base = atomicAdd(&shared_count,warp_tot);
			warp_base = __shfl_sync(0xffffffffu,warp_base,0u);
			uint32_t dst = warp_base + prefix - my_cnt;
			const uint32_t i0 = v << 2u;
			if ( (flags & 1u) != 0u ) out_ids[dst++] = i0;
			if ( (flags & 2u) != 0u ) out_ids[dst++] = i0 + 1u;
			if ( (flags & 4u) != 0u ) out_ids[dst++] = i0 + 2u;
			if ( (flags & 8u) != 0u ) out_ids[dst++] = i0 + 3u;
		}
	}
	for (uint32_t idx = (vec_n << 2u) + tid; idx < vocabulary_count; idx += 512u)
		if ( nextafterf(cs[idx] + bn[idx],INFINITY) >= thresh )
			out_ids[atomicAdd(&shared_count,1u)] = idx;
	__syncthreads();
	if ( tid == 0u ) candidate_count[row] = shared_count;
}

static __global__ void SparkLmHeadCertifiedRescoreBatchedKernel(const void *hidden_bf16,const void *head_weight_bf16,const uint32_t *candidate_ids,const uint32_t *candidate_count,float *partial_scores,uint32_t *partial_candidates,uint64_t row_stride,uint32_t vocabulary_count,uint32_t hidden_dimension,uint32_t rows)
{
	extern __shared__ float hidden_shared[];
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	const uint32_t row = blockIdx.y;
	if ( row >= rows ) return;
	uint32_t count = candidate_count[row],lane = threadIdx.x % SPARK_LM_WARP_LANES,slot = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),candidate,running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	const uint32_t *row_candidates = candidate_ids + (uint64_t)row * vocabulary_count;
	SparkLmHeadStageHidden(hidden_bf16,hidden_shared,row,hidden_dimension);
	for (; slot < count; slot += gridDim.x * SPARK_LM_CTA_WARPS)
	{
		candidate = row_candidates[slot];
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
		partial_scores[(uint64_t)row * row_stride + blockIdx.x] = best_score[0];
		partial_candidates[(uint64_t)row * row_stride + blockIdx.x] = best_candidate[0];
	}
}

static __global__ void SparkLmHeadCertifiedReduceBatchedKernel(const float *partial_scores,const uint32_t *partial_candidates,uint64_t row_stride,uint32_t *output_token_id,float *output_score,uint32_t candidate_offset,uint32_t partial_count,uint32_t rows)
{
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	const uint32_t row = blockIdx.y;
	if ( row >= rows ) return;
	const float *row_scores = partial_scores + (uint64_t)row * row_stride;
	const uint32_t *row_candidates = partial_candidates + (uint64_t)row * row_stride;
	uint32_t partial,running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	for (partial = threadIdx.x; partial < partial_count; partial += blockDim.x)
	{
		score = row_scores[partial];
		if ( score > running_best || (score == running_best && row_candidates[partial] < running_candidate) )
		{
			running_best = score;
			running_candidate = row_candidates[partial];
		}
	}
	SparkLmArgmaxReduce(running_best,running_candidate,best_score,best_candidate);
	if ( threadIdx.x == 0u )
	{
		output_token_id[row] = best_candidate[0] + candidate_offset;
		output_score[row] = best_score[0];
	}
}

static inline SparkGpuError SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(
	SparkGpuStream stream,const void *hidden_bf16,const void *head_weight_bf16,
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
	uint32_t partial_count = row_count <= 4u ? SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT :
		(row_count <= 16u ? 32u : 16u);
	uint64_t row_floats = SparkHeadCertifiedFp8ScratchBytes(vocabulary_count,hidden_dimension) / sizeof(float);
	uint32_t first,rows;
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
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	view = SparkLmHeadCertifiedFp8ScratchView(scratch,vocabulary_count,hidden_dimension);
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedHiddenNormBatchedKernel), dim3(norm_blocks,row_count),SPARK_LM_CTA_THREADS,0u,
			stream, hidden_bf16,view.hidden_norms,row_floats,hidden_dimension,row_count)
	));
	if ( row_count >= 4u && (hidden_dimension % 64u) == 0u )
	{
		uint32_t nb = (vocabulary_count + 63u) / 64u;
		if ( row_count <= 16u )
			SPARK_LM_LAUNCH((
			LM_LAUNCH((SparkLmHeadCertifiedFp8ScoreMmaKernel<16u,64u,128u,1u>), ((row_count + 15u) / 16u) * nb,128u,0u,
					stream, (const uint16_t *)hidden_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
					view.coarse_scores,view.bounds,row_floats,row_count,vocabulary_count,hidden_dimension)
			));
		else if ( row_count <= 32u )
			SPARK_LM_LAUNCH((
			LM_LAUNCH((SparkLmHeadCertifiedFp8ScoreMmaKernel<32u,64u,256u,2u>), ((row_count + 31u) / 32u) * nb,256u,0u,
					stream, (const uint16_t *)hidden_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
					view.coarse_scores,view.bounds,row_floats,row_count,vocabulary_count,hidden_dimension)
			));
		else
			SPARK_LM_LAUNCH((
			LM_LAUNCH((SparkLmHeadCertifiedFp8ScoreMmaKernel<64u,64u,256u,4u>), ((row_count + 63u) / 64u) * nb,256u,0u,
					stream, (const uint16_t *)hidden_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
					view.coarse_scores,view.bounds,row_floats,row_count,vocabulary_count,hidden_dimension)
			));
	}
	else
	{
		for (first=0u; first<row_count; first+=rows)
		{
			rows = row_count - first < SPARK_LM_HEAD_CERTIFIED_FP8_ROWS ? row_count - first : SPARK_LM_HEAD_CERTIFIED_FP8_ROWS;
			view = SparkLmHeadCertifiedFp8ScratchView((float *)scratch + first * row_floats,vocabulary_count,hidden_dimension);
			SPARK_LM_LAUNCH((
			LM_LAUNCH((SparkLmHeadCertifiedFp8ScoreRowsKernel), score_blocks,
					SPARK_LM_HEAD_CERTIFIED_FP8_THREADS,0u,stream, (const uint16_t *)hidden_bf16 + (uint64_t)first * hidden_dimension,
					shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
					view.coarse_scores,view.bounds,row_floats,rows,vocabulary_count,hidden_dimension)
			));
		}
		view = SparkLmHeadCertifiedFp8ScratchView(scratch,vocabulary_count,hidden_dimension);
	}
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedScreenBatchedKernel), dim3(1u,row_count),512u,0u,stream,
			view.coarse_scores,view.bounds,row_floats,candidate_ids,candidate_count,
			vocabulary_count,row_count)
	));
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedRescoreBatchedKernel), dim3(partial_count,row_count),
			SPARK_LM_CTA_THREADS,hidden_dimension * sizeof(float),stream,
			hidden_bf16,head_weight_bf16,candidate_ids,candidate_count,
			view.partial_scores,view.partial_candidates,row_floats,vocabulary_count,hidden_dimension,row_count)
	));
	SPARK_LM_LAUNCH((
	LM_LAUNCH((SparkLmHeadCertifiedReduceBatchedKernel), dim3(1u,row_count),SPARK_LM_CTA_THREADS,0u,stream,
			view.partial_scores,view.partial_candidates,row_floats,output_token_id,output_score,
			candidate_offset,partial_count,row_count)
	));
	return(SparkGpuGetLastError());
}
