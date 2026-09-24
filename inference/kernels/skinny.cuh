#pragma once

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/scale.cuh"
#include "runtime/launch.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define LM_SKINNY_ROWS 4u
#define LM_SKINNY_THREADS 256u
#define LM_SKINNY_CHUNK_BYTES 16u

typedef struct LmSkinnyArguments
{
	const uint8_t *weight;
	const uint16_t *activation;
	uint16_t *output_bf16;
	float *output_f32;
	const uint32_t *route_expert;
	const uint32_t *route_packed_row;
	LmScaleTensor scale;
	uint64_t weight_group_bytes;
	uint32_t rows,pairs,top_k,activation_packed;
	uint32_t input_dimension,output_dimension,output_row_stride,output_column_offset;
}
LmSkinnyArguments;

static __device__ __forceinline__ float LmSkinnyBf16Pairs(uint32_t weight, uint32_t activation, float accumulator)
{
	accumulator = fmaf(__uint_as_float(weight << 16u),__uint_as_float(activation << 16u),accumulator);
	return(fmaf(__uint_as_float(weight & 0xffff0000u),__uint_as_float(activation & 0xffff0000u),accumulator));
}

static __device__ __forceinline__ float LmSkinnyBf16Chunk(uint4 weight, uint4 activation, float accumulator)
{
	accumulator = LmSkinnyBf16Pairs(weight.x,activation.x,accumulator);
	accumulator = LmSkinnyBf16Pairs(weight.y,activation.y,accumulator);
	accumulator = LmSkinnyBf16Pairs(weight.z,activation.z,accumulator);
	return(LmSkinnyBf16Pairs(weight.w,activation.w,accumulator));
}

static __device__ __forceinline__ float LmSkinnyFp8Word(uint32_t weight, uint4 activation_half, uint32_t word, float accumulator)
{
	float2 low = LmE4m3PairToFloat2((uint16_t)weight), high = LmE4m3PairToFloat2((uint16_t)(weight >> 16u));
	uint32_t first = word == 0u ? activation_half.x : activation_half.z, second = word == 0u ? activation_half.y : activation_half.w;
	accumulator = fmaf(low.x,__uint_as_float(first << 16u),accumulator);
	accumulator = fmaf(low.y,__uint_as_float(first & 0xffff0000u),accumulator);
	accumulator = fmaf(high.x,__uint_as_float(second << 16u),accumulator);
	return(fmaf(high.y,__uint_as_float(second & 0xffff0000u),accumulator));
}

template<class Format>
struct LmSkinnyFormat
{
	static constexpr bool kSupported = false;
};

typedef struct LmSkinnyWide
{
	uint4 low,high;
}
LmSkinnyWide;

template<>
struct LmSkinnyFormat<LmBf16Format>
{
	using Activation = uint4;
	static constexpr bool kSupported = true;
	static constexpr uint32_t kElements = 8u;
	static __device__ __forceinline__ Activation Load(const uint16_t *activation)
	{
		return(__ldg((const uint4 *)activation));
	}
	static __device__ __forceinline__ float Chunk(uint4 weight, Activation activation)
	{
		return(LmSkinnyBf16Chunk(weight,activation,0.0f));
	}
};

template<>
struct LmSkinnyFormat<LmFp8>
{
	using Activation = LmSkinnyWide;
	static constexpr bool kSupported = true;
	static constexpr uint32_t kElements = 16u;
	static __device__ __forceinline__ Activation Load(const uint16_t *activation)
	{
		Activation value;
		value.low = __ldg((const uint4 *)activation);
		value.high = __ldg((const uint4 *)activation + 1);
		return(value);
	}
	static __device__ __forceinline__ float Chunk(uint4 weight, Activation activation)
	{
		float accumulator = 0.0f;
		accumulator = LmSkinnyFp8Word(weight.x,activation.low,0u,accumulator);
		accumulator = LmSkinnyFp8Word(weight.y,activation.low,1u,accumulator);
		accumulator = LmSkinnyFp8Word(weight.z,activation.high,0u,accumulator);
		return(LmSkinnyFp8Word(weight.w,activation.high,1u,accumulator));
	}
};

template<class Format>
static __device__ __forceinline__ float LmSkinnyScale(const LmScaleTensor *scale, uint32_t group, uint32_t neuron, uint32_t k)
{
	if constexpr ( Format::kScaleGroup == 0u )
		return(1.0f);
	else
		return(LmScaleTensorLoad(scale,group,neuron,k));
}

template<class Format, uint32_t LANES, uint32_t ROWS>
__global__ __launch_bounds__(LM_SKINNY_THREADS) void LmSkinnyKernel(const __grid_constant__ LmSkinnyArguments args)
{
	constexpr uint32_t unroll = ROWS == 1u ? 8u : 4u, elements = LmSkinnyFormat<Format>::kElements;
	const uint32_t sub = threadIdx.x % LANES, task = (blockIdx.x * LM_SKINNY_THREADS + threadIdx.x) / LANES;
	const uint32_t total = (args.route_expert != 0 ? args.pairs : 1u) * args.output_dimension, live = task < total ? 1u : 0u;
	const uint32_t chunks = live != 0u ? args.input_dimension / elements : 0u, pair = live != 0u ? task / args.output_dimension : 0u, neuron = live != 0u ? task % args.output_dimension : 0u;
	uint32_t group = 0u, source = 0u, target = 0u, base, u, r, c, step;
	float accumulator[ROWS], scale;
	uint4 weight[unroll];
	typename LmSkinnyFormat<Format>::Activation staged[unroll][ROWS];
	const uint4 *row;
	const uint16_t *activation[ROWS];
	if ( live != 0u && args.route_expert != 0 )
	{
		group = args.route_expert[pair];
		target = args.route_packed_row != 0 ? args.route_packed_row[pair] : pair;
		source = args.activation_packed != 0u ? target : pair / args.top_k;
	}
	row = (const uint4 *)(args.weight + (uint64_t)group * args.weight_group_bytes + (uint64_t)neuron * args.input_dimension * (LM_SKINNY_CHUNK_BYTES / elements));
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
	{
		accumulator[r] = 0.0f;
		activation[r] = args.activation + (uint64_t)(source + (r < args.rows ? r : 0u)) * args.input_dimension;
	}
	for ( base = sub; base < chunks; base += LANES * unroll )
	{
		#pragma unroll
		for ( u = 0u; u < unroll; u++ )
		{
			c = base + u * LANES < chunks ? base + u * LANES : 0u;
			weight[u] = base + u * LANES < chunks ? __ldcs(row + c) : make_uint4(0u,0u,0u,0u);
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				staged[u][r] = LmSkinnyFormat<Format>::Load(activation[r] + c * elements);
		}
		#pragma unroll
		for ( u = 0u; u < unroll; u++ )
		{
			scale = LmSkinnyScale<Format>(&args.scale,group,neuron,(base + u * LANES) * elements);
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				accumulator[r] = fmaf(scale,LmSkinnyFormat<Format>::Chunk(weight[u],staged[u][r]),accumulator[r]);
		}
	}
	#pragma unroll
	for ( step = LANES / 2u; step > 0u; step >>= 1u )
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			accumulator[r] += __shfl_xor_sync(0xffffffffu,accumulator[r],step,LANES);
	if ( live == 0u || sub != 0u )
		return;
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
		if ( r < args.rows && args.output_f32 != 0 )
			args.output_f32[(uint64_t)(target + r) * args.output_row_stride + args.output_column_offset + neuron] = accumulator[r];
		else if ( r < args.rows )
			args.output_bf16[(uint64_t)(target + r) * args.output_row_stride + args.output_column_offset + neuron] = LmFloatToBf16(accumulator[r]);
}

template<class Format, uint32_t LANES, uint32_t ROWS>
static int32_t LmSkinnyLaunchShape(const LmSkinnyArguments *args, uint64_t tasks, cudaStream_t stream)
{
	LmSkinnyKernel<Format,LANES,ROWS><<<(uint32_t)((tasks * LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS),LM_SKINNY_THREADS,0u,stream>>>(*args);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format, uint32_t ROWS>
static int32_t LmSkinnyLaunchRows(const LmSkinnyArguments *args, uint64_t tasks, uint32_t chunks, cudaStream_t stream)
{
	if ( chunks >= 32u )
		return(LmSkinnyLaunchShape<Format,32u,ROWS>(args,tasks,stream));
	if ( chunks >= 16u )
		return(LmSkinnyLaunchShape<Format,16u,ROWS>(args,tasks,stream));
	return(LmSkinnyLaunchShape<Format,8u,ROWS>(args,tasks,stream));
}

static uint32_t LmSkinnyAligned(const void *pointer)
{
	return(((uintptr_t)pointer % LM_SKINNY_CHUNK_BYTES) == 0u ? 1u : 0u);
}

template<class Format>
static int32_t LmSkinnyValidate(const LmSkinnyArguments *args)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	uint32_t grouped = args->route_expert != 0 ? 1u : 0u;
	if ( args->weight == 0 || args->activation == 0 || (args->output_bf16 == 0) == (args->output_f32 == 0) )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->rows == 0u || args->rows > LM_SKINNY_ROWS || (grouped != 0u && (args->rows != 1u || args->pairs == 0u || args->top_k == 0u || args->pairs > LM_SKINNY_ROWS * args->top_k)) )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->input_dimension == 0u || args->output_dimension == 0u || (args->input_dimension % elements) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( LmSkinnyAligned(args->weight) == 0u || LmSkinnyAligned(args->activation) == 0u || (args->weight_group_bytes % LM_SKINNY_CHUNK_BYTES) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->output_column_offset > args->output_row_stride || args->output_dimension > args->output_row_stride - args->output_column_offset )
		return(LM_LAUNCH_ERR_OUTPUT);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t LmSkinnyLaunch(LmSkinnyArguments *args, cudaStream_t stream)
{
	uint64_t tasks;
	int32_t status;
	if constexpr ( !LmSkinnyFormat<Format>::kSupported )
		return(LM_LAUNCH_ERR_SHAPE);
	else
	{
		if ( args->output_row_stride == 0u )
			args->output_row_stride = args->output_dimension;
		status = LmSkinnyValidate<Format>(args);
		if ( status != LM_LAUNCH_OK )
			return(status);
		tasks = (uint64_t)(args->route_expert != 0 ? args->pairs : 1u) * args->output_dimension;
		if ( args->rows == 1u )
			return(LmSkinnyLaunchRows<Format,1u>(args,tasks,args->input_dimension / LmSkinnyFormat<Format>::kElements,stream));
		return(LmSkinnyLaunchRows<Format,LM_SKINNY_ROWS>(args,tasks,args->input_dimension / LmSkinnyFormat<Format>::kElements,stream));
	}
}

template<class Format>
static int32_t LmSkinnyDense(const void *weight, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, cudaStream_t stream)
{
	LmSkinnyArguments args;
	memset(&args,0,sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.output_f32 = output_f32;
	args.scale = LmScaleTensorNone();
	args.rows = rows;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_row_stride;
	args.output_column_offset = output_column_offset;
	return(LmSkinnyLaunch<Format>(&args,stream));
}

template<class Format>
static int32_t LmSkinnyExperts(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *route_expert, const uint32_t *route_packed_row, uint32_t pairs, uint32_t top_k, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, cudaStream_t stream)
{
	LmSkinnyArguments args;
	memset(&args,0,sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.route_expert = route_expert;
	args.route_packed_row = route_packed_row;
	args.scale = scale;
	args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * (Format::kStoredBits / 8u);
	args.rows = 1u;
	args.pairs = pairs;
	args.top_k = top_k;
	args.activation_packed = activation_packed;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	return(LmSkinnyLaunch<Format>(&args,stream));
}
