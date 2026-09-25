#pragma once

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/scale.cuh"
#include "runtime/launch.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define LM_SKINNY_ROWS 8u
#define LM_SKINNY_ROWS_MID 4u
#define LM_SKINNY_THREADS 256u
#define LM_SKINNY_CHUNK_BYTES 16u
#define LM_SKINNY_GROUPED_MAX_MEAN_ROWS 16u
#define LM_SKINNY_GROUPED_NEURONS 4u

typedef struct LmSkinnyArguments
{
	const uint8_t *weight;
	const uint16_t *activation;
	uint16_t *output_bf16;
	float *output_f32;
	const uint32_t *route_expert;
	const uint32_t *route_packed_row;
	const uint32_t *group_row_offset;
	const uint32_t *route_source_token;
	LmScaleTensor scale;
	uint64_t weight_group_bytes;
	uint32_t rows,pairs,top_k,activation_packed,groups;
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
	float2 low = LmE4m3PairToFloat2Pure((uint16_t)weight), high = LmE4m3PairToFloat2Pure((uint16_t)(weight >> 16u));
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
	float value;
	if constexpr ( Format::kScaleGroup == 0u )
		return(1.0f);
	else
	{
		value = LmScaleTensorLoadByIndex(scale,((uint64_t)group * scale->group_stride_entries) + ((uint64_t)neuron * scale->row_group_stride_entries) + (k / Format::kScaleGroup));
		return(scale->encoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL ? value * ((const float *)scale->global_data)[group] : value);
	}
}

template<class Format>
static uint32_t LmSkinnyScaleLayout(const LmScaleTensor *scale)
{
	if constexpr ( Format::kScaleGroup == 0u )
		return(1u);
	else
		return(scale->encoding == LM_SCALE_ENCODING_NONE || (scale->reserved == 0u && scale->data != 0 && scale->row_group_size == 1u && scale->k_group_size == Format::kScaleGroup) ? 1u : 0u);
}

static __device__ __forceinline__ void LmSkinnyResolve(const LmSkinnyArguments &args, uint32_t pair, uint32_t *group, uint32_t *source, uint32_t *target)
{
	*group = 0u;
	*source = 0u;
	*target = 0u;
	if ( args.route_expert == 0 )
		return;
	*group = args.route_expert[pair];
	*target = args.route_packed_row != 0 ? args.route_packed_row[pair] : pair;
	*source = args.activation_packed != 0u ? *target : pair / args.top_k;
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyAccumulate(const LmSkinnyArguments &args, const uint4 *const *rows, const uint16_t *const *activation, uint32_t group, uint32_t neuron, uint32_t sub, uint32_t chunks, float (*accumulator)[ROWS])
{
	constexpr uint32_t depth = ROWS == 1u ? 8u : ROWS <= LM_SKINNY_ROWS_MID ? 4u : 2u, unroll = depth / NPG != 0u ? depth / NPG : 1u, elements = LmSkinnyFormat<Format>::kElements;
	uint32_t base, u, n, r, c;
	float scale;
	uint4 weight[unroll][NPG];
	typename LmSkinnyFormat<Format>::Activation staged[unroll][ROWS];
	for ( base = sub; base < chunks; base += LANES * unroll )
	{
		#pragma unroll
		for ( u = 0u; u < unroll; u++ )
		{
			c = base + u * LANES < chunks ? base + u * LANES : 0u;
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
				weight[u][n] = base + u * LANES < chunks ? __ldcs(rows[n] + c) : make_uint4(0u,0u,0u,0u);
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				staged[u][r] = LmSkinnyFormat<Format>::Load(activation[r] + c * elements);
		}
		#pragma unroll
		for ( u = 0u; u < unroll; u++ )
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				scale = base + u * LANES < chunks ? LmSkinnyScale<Format>(&args.scale,group,neuron + n < args.output_dimension ? neuron + n : neuron,(base + u * LANES) * elements) : 1.0f;
				#pragma unroll
				for ( r = 0u; r < ROWS; r++ )
					accumulator[n][r] = fmaf(scale,LmSkinnyFormat<Format>::Chunk(weight[u][n],staged[u][r]),accumulator[n][r]);
			}
	}
}

template<uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyStore(const LmSkinnyArguments &args, float (*accumulator)[ROWS], uint32_t sub, uint32_t live, uint32_t target, uint32_t neuron, uint32_t rows)
{
	uint32_t step, n, r;
	uint64_t index;
	#pragma unroll
	for ( step = LANES / 2u; step > 0u; step >>= 1u )
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				accumulator[n][r] += __shfl_xor_sync(0xffffffffu,accumulator[n][r],step,LANES);
	if ( live == 0u || sub != 0u )
		return;
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
		{
			if ( r >= rows || neuron + n >= args.output_dimension )
				continue;
			index = (uint64_t)(target + r) * args.output_row_stride + args.output_column_offset + neuron + n;
			if ( args.output_f32 != 0 )
				args.output_f32[index] = accumulator[n][r];
			else
				args.output_bf16[index] = LmFloatToBf16(accumulator[n][r]);
		}
}

template<class Format, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyWeightRows(const LmSkinnyArguments &args, uint32_t group, uint32_t neuron, const uint4 **rows)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	uint32_t n;
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
		rows[n] = (const uint4 *)(args.weight + (uint64_t)group * args.weight_group_bytes + (uint64_t)(neuron + n < args.output_dimension ? neuron + n : neuron) * args.input_dimension * (LM_SKINNY_CHUNK_BYTES / elements));
}

template<uint32_t NPG, uint32_t ROWS>
static __device__ __forceinline__ void LmSkinnyClear(float (*accumulator)[ROWS])
{
	uint32_t n, r;
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			accumulator[n][r] = 0.0f;
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
__global__ __launch_bounds__(LM_SKINNY_THREADS) void LmSkinnyKernel(const __grid_constant__ LmSkinnyArguments args)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	const uint32_t sub = threadIdx.x % LANES, task = (blockIdx.x * LM_SKINNY_THREADS + threadIdx.x) / LANES, per_pair = (args.output_dimension + NPG - 1u) / NPG;
	const uint32_t live = task < (args.route_expert != 0 ? args.pairs : 1u) * per_pair ? 1u : 0u;
	const uint32_t chunks = live != 0u ? args.input_dimension / elements : 0u, pair = live != 0u ? task / per_pair : 0u, neuron = live != 0u ? (task % per_pair) * NPG : 0u;
	uint32_t group, source, target, r;
	float accumulator[NPG][ROWS];
	const uint4 *rows[NPG];
	const uint16_t *activation[ROWS];
	LmSkinnyResolve(args,pair,&group,&source,&target);
	LmSkinnyWeightRows<Format,NPG>(args,group,neuron,rows);
	LmSkinnyClear<NPG,ROWS>(accumulator);
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
		activation[r] = args.activation + (uint64_t)(source + (r < args.rows ? r : 0u)) * args.input_dimension;
	LmSkinnyAccumulate<Format,LANES,ROWS,NPG>(args,rows,activation,group,neuron,sub,chunks,accumulator);
	LmSkinnyStore<LANES,ROWS,NPG>(args,accumulator,sub,live,target,neuron,args.rows);
}

template<uint32_t ROWS>
static __device__ __forceinline__ void LmSkinnyGroupedActivation(const LmSkinnyArguments &args, uint32_t row, uint32_t count, const uint16_t **activation)
{
	uint32_t r, packed;
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
	{
		packed = row + (r < count ? r : 0u);
		activation[r] = args.activation + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * args.input_dimension;
	}
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
__global__ __launch_bounds__(LM_SKINNY_THREADS) void LmSkinnyGroupedKernel(const __grid_constant__ LmSkinnyArguments args)
{
	const uint32_t sub = threadIdx.x % LANES, task = (blockIdx.x * LM_SKINNY_THREADS + threadIdx.x) / LANES, per_group = (args.output_dimension + NPG - 1u) / NPG;
	const uint32_t live = task < args.groups * per_group ? 1u : 0u, group = live != 0u ? task / per_group : 0u, neuron = live != 0u ? (task % per_group) * NPG : 0u;
	const uint32_t chunks = args.input_dimension / LmSkinnyFormat<Format>::kElements, end = live != 0u ? args.group_row_offset[group + 1u] : 0u;
	uint32_t row, count;
	float accumulator[NPG][ROWS];
	const uint4 *weights[NPG];
	const uint16_t *activation[ROWS];
	LmSkinnyWeightRows<Format,NPG>(args,group,neuron,weights);
	for ( row = live != 0u ? args.group_row_offset[group] : 0u; row < end; row += ROWS )
	{
		count = end - row < ROWS ? end - row : ROWS;
		LmSkinnyGroupedActivation<ROWS>(args,row,count,activation);
		LmSkinnyClear<NPG,ROWS>(accumulator);
		LmSkinnyAccumulate<Format,LANES,ROWS,NPG>(args,weights,activation,group,neuron,sub,chunks,accumulator);
		LmSkinnyStore<LANES,ROWS,NPG>(args,accumulator,sub,1u,row,neuron,count);
	}
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static int32_t LmSkinnyLaunchShape(const LmSkinnyArguments *args, cudaStream_t stream)
{
	uint64_t tasks = (uint64_t)(args->route_expert != 0 ? args->pairs : 1u) * ((args->output_dimension + NPG - 1u) / NPG);
	LM_LAUNCH((LmSkinnyKernel<Format,LANES,ROWS,NPG>),(uint32_t)((tasks * LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS),LM_SKINNY_THREADS,0u,stream,*args);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static int32_t LmSkinnyGroupedShape(const LmSkinnyArguments *args, cudaStream_t stream)
{
	const uint32_t per_group = (args->output_dimension + NPG - 1u) / NPG;
	uint64_t tasks = (uint64_t)args->groups * per_group;
	if ( (per_group % (32u / LANES)) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((LmSkinnyGroupedKernel<Format,LANES,ROWS,NPG>),(uint32_t)((tasks * LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS),LM_SKINNY_THREADS,0u,stream,*args);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format, uint32_t ROWS, uint32_t NPG>
static int32_t LmSkinnyGroupedLanes(const LmSkinnyArguments *args, uint32_t chunks, cudaStream_t stream)
{
	if ( chunks >= 32u )
		return(LmSkinnyGroupedShape<Format,32u,ROWS,NPG>(args,stream));
	if ( chunks >= 16u )
		return(LmSkinnyGroupedShape<Format,16u,ROWS,NPG>(args,stream));
	return(LmSkinnyGroupedShape<Format,8u,ROWS,NPG>(args,stream));
}

template<class Format, uint32_t LANES>
static int32_t LmSkinnyLaunchGroup(const LmSkinnyArguments *args, uint32_t chunks, cudaStream_t stream)
{
	uint32_t per_lane = (chunks + LANES - 1u) / LANES;
	if ( args->rows > LM_SKINNY_ROWS_MID )
		return(LmSkinnyLaunchShape<Format,LANES,LM_SKINNY_ROWS,1u>(args,stream));
	if ( args->rows != 1u )
		return(LmSkinnyLaunchShape<Format,LANES,LM_SKINNY_ROWS_MID,1u>(args,stream));
	if ( per_lane >= 8u )
		return(LmSkinnyLaunchShape<Format,LANES,1u,1u>(args,stream));
	if ( per_lane >= 4u )
		return(LmSkinnyLaunchShape<Format,LANES,1u,2u>(args,stream));
	if ( per_lane >= 2u )
		return(LmSkinnyLaunchShape<Format,LANES,1u,4u>(args,stream));
	return(LmSkinnyLaunchShape<Format,LANES,1u,8u>(args,stream));
}

template<class Format>
static int32_t LmSkinnyLaunchRows(const LmSkinnyArguments *args, uint32_t chunks, cudaStream_t stream)
{
	if ( chunks >= 32u )
		return(LmSkinnyLaunchGroup<Format,32u>(args,chunks,stream));
	if ( chunks >= 16u )
		return(LmSkinnyLaunchGroup<Format,16u>(args,chunks,stream));
	return(LmSkinnyLaunchGroup<Format,8u>(args,chunks,stream));
}

static uint32_t LmSkinnyAligned(const void *pointer)
{
	return(((uintptr_t)pointer % LM_SKINNY_CHUNK_BYTES) == 0u ? 1u : 0u);
}

template<class Format>
static int32_t LmSkinnyValidateOperands(const LmSkinnyArguments *args)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	if ( args->weight == 0 || args->activation == 0 || (args->output_bf16 == 0) == (args->output_f32 == 0) )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->input_dimension == 0u || args->output_dimension == 0u || (args->input_dimension % elements) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( LmSkinnyAligned(args->weight) == 0u || LmSkinnyAligned(args->activation) == 0u || (args->weight_group_bytes % LM_SKINNY_CHUNK_BYTES) != 0u || LmSkinnyScaleLayout<Format>(&args->scale) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t LmSkinnyValidate(const LmSkinnyArguments *args)
{
	uint32_t grouped = args->route_expert != 0 ? 1u : 0u;
	if ( LmSkinnyValidateOperands<Format>(args) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->rows == 0u || args->rows > LM_SKINNY_ROWS || (grouped != 0u && (args->rows != 1u || args->pairs == 0u || args->top_k == 0u || args->pairs > LM_SKINNY_ROWS * args->top_k)) )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( args->output_column_offset > args->output_row_stride || args->output_dimension > args->output_row_stride - args->output_column_offset )
		return(LM_LAUNCH_ERR_OUTPUT);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t LmSkinnyLaunch(LmSkinnyArguments *args, cudaStream_t stream)
{
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
		return(LmSkinnyLaunchRows<Format>(args,args->input_dimension / LmSkinnyFormat<Format>::kElements,stream));
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

template<class Format, uint32_t NPG>
static int32_t LmSkinnyGroupedExpertsWith(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t pairs, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, cudaStream_t stream)
{
	LmSkinnyArguments args;
	if constexpr ( !LmSkinnyFormat<Format>::kSupported )
		return(LM_LAUNCH_ERR_SHAPE);
	else
	{
		memset(&args,0,sizeof(args));
		args.weight = (const uint8_t *)weight;
		args.activation = activation;
		args.output_bf16 = output_bf16;
		args.group_row_offset = group_row_offset;
		args.route_source_token = route_source_token;
		args.scale = scale;
		args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * (Format::kStoredBits / 8u);
		args.rows = 1u;
		args.pairs = pairs;
		args.activation_packed = activation_packed;
		args.groups = groups;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		args.output_row_stride = output_dimension;
		if ( LmSkinnyValidateOperands<Format>(&args) != LM_LAUNCH_OK || group_row_offset == 0 || (activation_packed == 0u && route_source_token == 0) || groups == 0u || pairs == 0u || pairs > LM_SKINNY_GROUPED_MAX_MEAN_ROWS * groups )
			return(LM_LAUNCH_ERR_SHAPE);
		if ( pairs <= 2u * groups )
			return(LmSkinnyGroupedLanes<Format,LM_SKINNY_ROWS_MID,NPG>(&args,input_dimension / LmSkinnyFormat<Format>::kElements,stream));
		return(LmSkinnyGroupedLanes<Format,LM_SKINNY_ROWS,NPG>(&args,input_dimension / LmSkinnyFormat<Format>::kElements,stream));
	}
}

template<class Format>
static int32_t LmSkinnyGroupedExperts(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t pairs, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, cudaStream_t stream)
{
	return(LmSkinnyGroupedExpertsWith<Format,LM_SKINNY_GROUPED_NEURONS>(weight,scale,activation,output_bf16,group_row_offset,route_source_token,groups,pairs,activation_packed,input_dimension,output_dimension,stream));
}
