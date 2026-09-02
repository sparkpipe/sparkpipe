#pragma once


#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include <stdint.h>

template<uint32_t THREADS>
static __device__ float LmBlockSum(float value, float *shared)
{
	const uint32_t warps = THREADS / LM_WARP_LANES;
	uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	uint32_t offset;
	for (offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	if ( lane == 0u )
		shared[warp] = value;
	__syncthreads();
	value = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
	if ( warp == 0u )
		for (offset = warps / 2u; offset > 0u; offset >>= 1u)
			value += __shfl_down_sync(0xffffffffu,value,offset);
	if ( threadIdx.x == 0u )
		shared[0] = value;
	__syncthreads();
	return(shared[0]);
}

template<uint32_t THREADS>
static __device__ float LmBlockMax(float value, float *shared)
{
	const uint32_t warps = THREADS / LM_WARP_LANES;
	uint32_t lane = threadIdx.x % LM_WARP_LANES,warp = threadIdx.x / LM_WARP_LANES;
	uint32_t offset;
	for (offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	if ( lane == 0u )
		shared[warp] = value;
	__syncthreads();
	value = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
	if ( warp == 0u )
		for (offset = warps / 2u; offset > 0u; offset >>= 1u)
			value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	if ( threadIdx.x == 0u )
		shared[0] = value;
	__syncthreads();
	return(shared[0]);
}

template<uint32_t THREADS, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmFusedResidualRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared;
	float *reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index;
	float total = 0.0f,scale;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[base + index]);
		if ( residual_bf16 != 0 )
			value += LmBf16ToFloat(residual_bf16[base + index]);
		row[index] = value;
		total += value * value;
		if ( residual_out_bf16 != 0 )
			residual_out_bf16[base + index] = LmFloatToBf16(value);
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = rsqrtf((total / (float)dimension) + epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		output_bf16[base + index] =
			LmFloatToBf16(row[index] * scale * LmScalarToFloat(weight[index]));
}

template<uint32_t THREADS, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmLayerNormKernel(const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, const Weight *__restrict__ bias, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared;
	float *reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index;
	float value,total = 0.0f,total_squared = 0.0f,mean,variance,inverse;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]);
		row[index] = value;
		total += value;
		total_squared += value * value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	total_squared = LmBlockSum<THREADS>(total_squared,reduction);
	mean = total / (float)dimension;
	variance = fmaxf((total_squared / (float)dimension) - (mean * mean),0.0f);
	inverse = rsqrtf(variance + epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		output_bf16[base + index] = LmFloatToBf16(((row[index] - mean) * inverse * LmScalarToFloat(weight[index])) + LmScalarToFloat(bias[index]));
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? index : dimension + index)]);
		float up = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? dimension + index : index)]);
		output_bf16[out_base + index] =
			LmFloatToBf16((gate / (1.0f + __expf(-gate))) * up);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSituMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, float beta, float linear_beta)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_up_bf16[base + index]);
		float up = LmBf16ToFloat(gate_up_bf16[base + dimension + index]);
		float activated = beta * tanhf(gate / beta) *
			(1.0f / (1.0f + __expf(-gate)));
		if (linear_beta > 0.0f)
			up = linear_beta * tanhf(up / linear_beta);
		output_bf16[out_base + index] = LmFloatToBf16(activated * up);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSigmoidRowsKernel(const uint16_t *__restrict__ logits_bf16, float *__restrict__ scores, uint32_t width)
{
	uint64_t base = (uint64_t)blockIdx.x * width;
	uint32_t index;
	for (index = threadIdx.x; index < width; index += THREADS)
	{
		float value = LmBf16ToFloat(logits_bf16[base + index]);
		scores[base + index] = 1.0f / (1.0f + __expf(-value));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_bf16[base + index]);
		float value = LmBf16ToFloat(output_bf16[base + index]);
		output_bf16[base + index] =
			LmFloatToBf16(value * (1.0f / (1.0f + __expf(-gate))));
	}
}

template<class Format, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmQuantiseRowsKernel(const uint16_t *__restrict__ input_bf16, const uint32_t *__restrict__ source_row_map, uint8_t *__restrict__ output_codes, uint8_t *__restrict__ output_scales, uint32_t row_count, uint32_t dimension)
{
	static_assert(Format::kScaleGroup > 0u,
		"an unquantised format has no scale groups and must not reach a "
		"quantise; the caller should skip it, not divide by zero sizing the grid");
	extern __shared__ float lm_quant_shared[];
	float *row = lm_quant_shared;
	float *reduction = lm_quant_shared + Format::kScaleGroup;
	const uint32_t groups = dimension / Format::kScaleGroup;
	uint32_t destination = blockIdx.x,group = blockIdx.y,index;
	uint64_t source;
	float absmax = 0.0f,scale,inverse;
	if ( destination >= row_count || group >= groups )
		return;
	source = (uint64_t)(source_row_map != 0 ? source_row_map[destination] : destination)
		* (uint64_t)dimension + (group * Format::kScaleGroup);
	for (index = threadIdx.x; index < Format::kScaleGroup; index += THREADS)
	{
		row[index] = LmBf16ToFloat(input_bf16[source + index]);
		absmax = fmaxf(absmax,fabsf(row[index]));
	}
	absmax = LmBlockMax<THREADS>(absmax,reduction);
	scale = fmaxf(absmax / Format::kMax,1.0e-8f);
	inverse = 1.0f / scale;
	if ( threadIdx.x == 0u )
		output_scales[((uint64_t)destination * groups) + group] = LmFloatToUe4m3(scale);
	__syncthreads();
	static_assert(
		(Format::kScaleGroup % 8u) == 0u,
		"quantization groups must contain complete eight-code blocks");
	for (index = threadIdx.x * 8u;
		index < Format::kScaleGroup;
		index += THREADS * 8u)
	{
		float values[8];
		uint32_t value_index;
		uint64_t bit =
			(((uint64_t)destination * dimension) +
			 (group * Format::kScaleGroup) + index) *
			Format::kStoredBits;

		for (value_index = 0u; value_index < 8u; ++value_index)
		{
			values[value_index] = row[index + value_index] * inverse;
		}
		LmStoreCodeOctet<Format>(output_codes, bit, values);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmMoeFinalizeKernel(const uint16_t *__restrict__ packed_bf16, const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight, uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t top_k, uint32_t dimension)
{
	uint32_t token = blockIdx.y;
	uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	uint32_t route;
	float total = 0.0f;
	if ( token >= tokens || element >= dimension )
		return;
	for (route = 0u; route < top_k; ++route)
	{
		uint32_t packed_row = packed_row_of_token_route[(token * top_k) + route];
		total += LmBf16ToFloat(packed_bf16[((uint64_t)packed_row * dimension) + element])
			* route_weight[(token * top_k) + route];
	}
	output_bf16[((uint64_t)token * dimension) + element] = LmFloatToBf16(total);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmAddRowsKernel(const uint16_t *__restrict__ a_bf16, const uint16_t *__restrict__ b_bf16, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t dimension)
{
	uint32_t row = blockIdx.y,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint64_t index;
	if ( row >= rows || element >= dimension )
		return;
	index = ((uint64_t)row * dimension) + element;
	out_bf16[index] = LmFloatToBf16(LmBf16ToFloat(a_bf16[index]) + LmBf16ToFloat(b_bf16[index]));
}

template<class Format, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmFusedNormQuantiseKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const uint16_t *__restrict__ weight_bf16, const uint32_t *__restrict__ source_row_map, uint16_t *__restrict__ residual_out_bf16, uint8_t *__restrict__ output_codes, uint8_t *__restrict__ output_scales, uint32_t rows, uint32_t dimension, float epsilon)
{
	extern __shared__ float lm_fused_shared[];
	float *row = lm_fused_shared;
	float *reduction = lm_fused_shared + dimension;
	const uint32_t groups = dimension / Format::kScaleGroup;
	uint32_t destination = blockIdx.x,index,group;
	uint64_t source;
	float total = 0.0f,scale;
	if ( destination >= rows )
		return;
	source = (uint64_t)(source_row_map != 0 ? source_row_map[destination] : destination) * dimension;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[source + index]);
		if ( residual_bf16 != 0 )
			value += LmBf16ToFloat(residual_bf16[source + index]);
		row[index] = value;
		total += value * value;
		if ( residual_out_bf16 != 0 )
			residual_out_bf16[source + index] = LmFloatToBf16(value);
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = rsqrtf((total / (float)dimension) + epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		row[index] = row[index] * scale * LmBf16ToFloat(weight_bf16[index]);
	__syncthreads();
	for (group = 0u; group < groups; ++group)
	{
		float absmax = 0.0f,inverse;
		for (index = threadIdx.x; index < Format::kScaleGroup; index += THREADS)
			absmax = fmaxf(absmax,fabsf(row[(group * Format::kScaleGroup) + index]));
		absmax = LmBlockMax<THREADS>(absmax,reduction);
		inverse = 1.0f / fmaxf(absmax / Format::kMax,1.0e-8f);
		if ( threadIdx.x == 0u )
			output_scales[((uint64_t)destination * groups) + group] =
				LmFloatToUe4m3(fmaxf(absmax / Format::kMax,1.0e-8f));
		static_assert(
			(Format::kScaleGroup % 8u) == 0u,
			"quantization groups must contain complete eight-code blocks");
		for (index = threadIdx.x * 8u;
			index < Format::kScaleGroup;
			index += THREADS * 8u)
		{
			float values[8];
			uint32_t value_index;
			uint64_t bit =
				(((uint64_t)destination * dimension) +
				 (group * Format::kScaleGroup) + index) *
				Format::kStoredBits;

			for (value_index = 0u; value_index < 8u; ++value_index)
			{
				values[value_index] =
					row[(group * Format::kScaleGroup) + index + value_index] *
					inverse;
			}
			LmStoreCodeOctet<Format>(output_codes, bit, values);
		}
		__syncthreads();
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmCopyRowsKernel(const uint16_t *__restrict__ source_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension)
{
	uint32_t row = blockIdx.y,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint64_t index;
	if ( row >= rows || element >= dimension )
		return;
	index = ((uint64_t)row * dimension) + element;
	destination_bf16[index] = source_bf16[index];
}

template<uint32_t THREADS, uint32_t MAX_SOURCES>
__global__ __launch_bounds__(THREADS, 1)
void LmAttnResKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16, const uint16_t *__restrict__ score_weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t sources, uint32_t rows, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float score[MAX_SOURCES];
	__shared__ float weight[MAX_SOURCES];
	uint32_t row = blockIdx.x,source,index;
	float running_max = -INFINITY,running_sum = 0.0f;
	for (source = 0u; source < sources; ++source)
	{
		const uint16_t *values = source + 1u == sources
			? partial_bf16 + ((uint64_t)row * dimension)
			: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
		float square = 0.0f,dot = 0.0f,inverse;
		for (index = threadIdx.x; index < dimension; index += THREADS)
		{
			float value = LmBf16ToFloat(values[index]);
			square += value * value;
		}
		square = LmBlockSum<THREADS>(square,reduction);
		inverse = rsqrtf((square / (float)dimension) + epsilon);
		for (index = threadIdx.x; index < dimension; index += THREADS)
			dot += LmBf16ToFloat(values[index]) * inverse
				* LmBf16ToFloat(score_weight_bf16[index]);
		dot = LmBlockSum<THREADS>(dot,reduction);
		if ( threadIdx.x == 0u )
			score[source] = dot;
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		for (source = 0u; source < sources; ++source)
			running_max = fmaxf(running_max,score[source]);
		for (source = 0u; source < sources; ++source)
		{
			weight[source] = __expf(score[source] - running_max);
			running_sum += weight[source];
		}
		for (source = 0u; source < sources; ++source)
			weight[source] /= running_sum;
	}
	__syncthreads();
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float total = 0.0f;
		for (source = 0u; source < sources; ++source)
		{
			const uint16_t *values = source + 1u == sources
				? partial_bf16 + ((uint64_t)row * dimension)
				: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
			total += weight[source] * LmBf16ToFloat(values[index]);
		}
		output_bf16[((uint64_t)row * dimension) + index] = LmFloatToBf16(total);
	}
}

template<uint32_t THREADS>
__global__ void LmGatherRowsKernel(const uint16_t *__restrict__ source_bf16, const uint32_t *__restrict__ source_row_map, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension)
{
	uint32_t row = blockIdx.y,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint32_t source_row;
	uint64_t from,to;
	if ( row >= rows || element >= dimension )
		return;
	source_row = source_row_map != 0 ? source_row_map[row] : row;
	from = ((uint64_t)source_row * dimension) + element;
	to = ((uint64_t)row * dimension) + element;
	destination_bf16[to] = source_bf16[from];
}
