#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <math.h>
#include <stdint.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define SPARK_MINIMAX_H3_CUDA_ATTENTION_THREADS 128u
#define SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS 256u

static __device__ float SparkMinimaxH3BlockReduceMax(float value, float *scratch)
{
	uint32_t lane = threadIdx.x & 31u;
	uint32_t warp = threadIdx.x >> 5u;
	uint32_t offset;
	for (offset=16u; offset>0u; offset>>=1u)
		value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
	if ( lane == 0u )
		scratch[warp] = value;
	__syncthreads();
	uint32_t warps = (blockDim.x + 31u) >> 5u;
	if ( threadIdx.x == 0u )
	{
		float reduced = scratch[0];
		for (offset=1u; offset<warps; offset++)
			reduced = fmaxf(reduced, scratch[offset]);
		scratch[0] = reduced;
	}
	__syncthreads();
	return(scratch[0]);
}

static __device__ float SparkMinimaxH3BlockReduceSumShared(float value, float *scratch)
{
	uint32_t lane = threadIdx.x & 31u;
	uint32_t warp = threadIdx.x >> 5u;
	uint32_t offset;
	for (offset=16u; offset>0u; offset>>=1u)
		value += __shfl_down_sync(0xffffffffu, value, offset);
	if ( lane == 0u )
		scratch[warp] = value;
	__syncthreads();
	uint32_t warps = (blockDim.x + 31u) >> 5u;
	if ( threadIdx.x == 0u )
	{
		float reduced = 0.0f;
		for (offset=0u; offset<warps; offset++)
			reduced += scratch[offset];
		scratch[0] = reduced;
	}
	__syncthreads();
	return(scratch[0]);
}

static __global__ void SparkMinimaxH3Rope3dKernel(const __nv_bfloat16 *input,
	const float *cos_angles, const float *sin_angles, uint32_t heads,
	uint32_t head_dim, uint32_t rope_dim, __nv_bfloat16 *output)
{
	uint32_t row = blockIdx.y;
	uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t head = slot / head_dim;
	uint32_t element = slot - head * head_dim;
	uint64_t base = ((uint64_t)row * heads + head) * head_dim;
	if ( head >= heads )
		return;
	if ( element >= rope_dim )
	{
		output[base + element] = input[base + element];
		return;
	}
	uint32_t half = rope_dim >> 1u;
	float self = __bfloat162float(input[base + element]);
	float partner = __bfloat162float(element < half ?
		input[base + half + element] : input[base + element - half]);
	float rotated = element < half ? -partner : partner;
	output[base + element] = __float2bfloat16(
		self * cos_angles[(uint64_t)row * rope_dim + element] +
		rotated * sin_angles[(uint64_t)row * rope_dim + element]);
}

static __global__ void SparkMinimaxH3DenseAttentionKernel(const __nv_bfloat16 *queries,
	const __nv_bfloat16 *keys, const __nv_bfloat16 *values, uint32_t seq,
	uint32_t heads, uint32_t head_dim, __nv_bfloat16 *output)
{
	extern __shared__ float shared[];
	float *query_shared = shared;
	float *accumulator = shared + head_dim;
	float *probabilities = accumulator + head_dim;
	float *scratch = probabilities + blockDim.x;
	uint32_t head = blockIdx.x;
	uint32_t row = blockIdx.y;
	uint32_t thread = threadIdx.x;
	uint32_t start,element;
	const __nv_bfloat16 *query_row =
		queries + ((uint64_t)row * heads + head) * head_dim;
	if ( thread < head_dim )
	{
		query_shared[thread] = __bfloat162float(query_row[thread]);
		accumulator[thread] = 0.0f;
	}
	__syncthreads();
	float running_max = -INFINITY;
	float running_sum = 0.0f;
	float scale = rsqrtf((float)head_dim);
	for (start=0u; start<seq; start+=blockDim.x)
	{
		uint32_t column = start + thread;
		float score = -INFINITY;
		if ( column < seq )
		{
			const __nv_bfloat16 *key_column =
				keys + ((uint64_t)column * heads + head) * head_dim;
			float dot = 0.0f;
			for (element=0u; element<head_dim; element++)
				dot += query_shared[element] * __bfloat162float(key_column[element]);
			score = dot * scale;
		}
		float tile_max = SparkMinimaxH3BlockReduceMax(score,scratch);
		float tile_sum;
		float rescale = expf(running_max - tile_max);
		float weight;
		if ( thread < head_dim )
			accumulator[thread] *= rescale;
		running_sum *= rescale;
		running_max = tile_max;
		weight = column < seq ? expf(score - running_max) : 0.0f;
		probabilities[thread] = weight;
		tile_sum = SparkMinimaxH3BlockReduceSumShared(weight,scratch);
		running_sum += tile_sum;
		for (element = thread; element < head_dim; element += blockDim.x)
		{
			float total = 0.0f;
			uint32_t tile;
			for (tile=0u; tile<blockDim.x; tile++)
			{
				if ( start + tile < seq )
				{
					const __nv_bfloat16 *value_column =
						values + ((uint64_t)(start + tile) * heads + head) * head_dim;
					total += probabilities[tile] *
						__bfloat162float(value_column[element]);
				}
			}
			accumulator[element] += total;
		}
		__syncthreads();
	}
	if ( thread < head_dim )
	{
		__nv_bfloat16 *out_row =
			output + ((uint64_t)row * heads + head) * head_dim;
		out_row[thread] = __float2bfloat16(accumulator[thread] / running_sum);
	}
}

static __global__ void SparkMinimaxH3AdaLNAffineKernel(const __nv_bfloat16 *input,
	const __nv_bfloat16 *scale, const __nv_bfloat16 *shift, uint32_t width,
	uint64_t count, __nv_bfloat16 *output)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= count )
		return;
	uint32_t column = (uint32_t)(index % (uint64_t)width);
	float value = __bfloat162float(input[index]);
	float scale_value = __bfloat162float(scale[column]);
	float shift_value = __bfloat162float(shift[column]);
	output[index] = __float2bfloat16(value * (1.0f + scale_value) + shift_value);
}

static __global__ void SparkMinimaxH3GateResidualKernel(const __nv_bfloat16 *value,
	const __nv_bfloat16 *gate, const __nv_bfloat16 *residual, uint32_t width,
	uint64_t count, __nv_bfloat16 *output)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= count )
		return;
	uint32_t column = (uint32_t)(index % (uint64_t)width);
	float gate_value = __bfloat162float(gate[column]);
	float residual_value = __bfloat162float(residual[index]);
	float value = __bfloat162float(value[index]);
	output[index] = __float2bfloat16(residual_value + gate_value * value);
}

static __global__ void SparkMinimaxH3SiluMulKernel(const __nv_bfloat16 *gate_up,
	uint32_t ffn, uint32_t count, __nv_bfloat16 *output)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= count )
		return;
	uint32_t row = index / ffn;
	uint32_t column = index - row * ffn;
	uint64_t base = (uint64_t)row * (2u * ffn);
	float gate = __bfloat162float(gate_up[base + column]);
	float up = __bfloat162float(gate_up[base + ffn + column]);
	output[index] = __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
}

static __device__ uint32_t SparkMinimaxH3ReflectIndex(int32_t position, uint32_t extent)
{
	if ( position < 0 )
		return((uint32_t)(-position));
	if ( position >= (int32_t)extent )
		return((uint32_t)(2 * (int32_t)extent - 2 - position));
	return((uint32_t)position);
}

static __global__ void SparkMinimaxH3Conv2d3x3ReflectKernel(const __nv_bfloat16 *input,
	const __nv_bfloat16 *weight, const __nv_bfloat16 *bias, uint32_t in_channels,
	uint32_t out_channels, uint32_t height, uint32_t width, __nv_bfloat16 *output)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t plane = (uint64_t)height * width;
	if ( index >= (uint64_t)out_channels * plane )
		return;
	uint32_t out_channel = (uint32_t)(index / plane);
	uint64_t spatial = index - (uint64_t)out_channel * plane;
	uint32_t y = (uint32_t)(spatial / width);
	uint32_t x = (uint32_t)(spatial - (uint64_t)y * width);
	float accumulator = bias != 0 ? __bfloat162float(bias[out_channel]) : 0.0f;
	uint32_t in_channel,ky,kx;
	for (in_channel=0u; in_channel<in_channels; in_channel++)
	{
		const __nv_bfloat16 *in_plane = input + (uint64_t)in_channel * plane;
		for (ky=0u; ky<3u; ky++)
		{
			uint32_t sy = SparkMinimaxH3ReflectIndex((int32_t)y + (int32_t)ky - 1, height);
			for (kx=0u; kx<3u; kx++)
			{
				uint32_t sx = SparkMinimaxH3ReflectIndex((int32_t)x + (int32_t)kx - 1, width);
				float sample = __bfloat162float(in_plane[(uint64_t)sy * width + sx]);
				float tap = __bfloat162float(weight[
					(((uint64_t)out_channel * in_channels + in_channel) * 3u + ky) * 3u + kx]);
				accumulator += sample * tap;
			}
		}
	}
	output[index] = __float2bfloat16(accumulator);
}

static __global__ void SparkMinimaxH3Conv1dDilatedKernel(const __nv_bfloat16 *input,
	const __nv_bfloat16 *weight, const __nv_bfloat16 *bias, uint32_t channels,
	uint32_t length, uint32_t dilation, __nv_bfloat16 *output)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= (uint64_t)channels * length )
		return;
	uint32_t channel = (uint32_t)(index / length);
	uint32_t position = (uint32_t)(index - (uint64_t)channel * length);
	float accumulator = bias != 0 ? __bfloat162float(bias[channel]) : 0.0f;
	uint32_t tap;
	for (tap=0u; tap<3u; tap++)
	{
		int32_t source = (int32_t)position + (int32_t)(tap * dilation) - (int32_t)dilation;
		if ( source >= 0 && source < (int32_t)length )
		{
			accumulator += __bfloat162float(
				input[(uint64_t)channel * length + (uint32_t)source]) *
				__bfloat162float(weight[(uint64_t)channel * 3u + tap]);
		}
	}
	output[index] = __float2bfloat16(accumulator);
}

static __global__ void SparkMinimaxH3SnakeKernel(const __nv_bfloat16 *input,
	const __nv_bfloat16 *alpha, uint32_t length, uint64_t count, __nv_bfloat16 *output)
{
	uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= count )
		return;
	uint32_t channel = (uint32_t)(index / length);
	float value = __bfloat162float(input[index]);
	float alpha_value = __bfloat162float(alpha[channel]);
	float sine = sinf(alpha_value * value);
	output[index] = __float2bfloat16(value + sine * sine / alpha_value);
}

extern "C" cudaError_t SparkMinimaxH3Rope3d(cudaStream_t stream, const void *input_bf16,
	const float *cos_angles, const float *sin_angles, uint32_t rows, uint32_t heads,
	uint32_t head_dim, uint32_t rope_dim, void *output_bf16)
{
	uint32_t slots = heads * head_dim;
	uint32_t threads = slots < SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS ?
		slots : SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS;
	dim3 grid((slots + threads - 1u) / threads, rows);
	if ( input_bf16 == 0 || cos_angles == 0 || sin_angles == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3Rope3dKernel<<<grid,threads,0,stream>>>(
		(const __nv_bfloat16 *)input_bf16,cos_angles,sin_angles,heads,head_dim,
		rope_dim,(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxH3DenseAttention(cudaStream_t stream,
	const void *queries_bf16, const void *keys_bf16, const void *values_bf16,
	uint32_t seq, uint32_t heads, uint32_t head_dim, void *output_bf16)
{
	uint32_t threads = SPARK_MINIMAX_H3_CUDA_ATTENTION_THREADS;
	size_t shared = ((size_t)head_dim * 2u + (size_t)threads * 2u) * sizeof(float);
	dim3 grid(heads,seq);
	if ( queries_bf16 == 0 || keys_bf16 == 0 || values_bf16 == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	if ( head_dim > threads )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3DenseAttentionKernel<<<grid,threads,shared,stream>>>(
		(const __nv_bfloat16 *)queries_bf16,(const __nv_bfloat16 *)keys_bf16,
		(const __nv_bfloat16 *)values_bf16,seq,heads,head_dim,
		(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxH3AdaLNAffine(cudaStream_t stream,
	const void *input_bf16, const void *scale_bf16, const void *shift_bf16,
	uint32_t rows, uint32_t width, void *output_bf16)
{
	uint64_t count = (uint64_t)rows * width;
	dim3 grid((uint32_t)((count + SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS - 1u) /
		SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS));
	if ( input_bf16 == 0 || scale_bf16 == 0 || shift_bf16 == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3AdaLNAffineKernel<<<grid,SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)input_bf16,(const __nv_bfloat16 *)scale_bf16,
		(const __nv_bfloat16 *)shift_bf16,width,count,
		(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxH3GateResidual(cudaStream_t stream,
	const void *value_bf16, const void *gate_bf16, const void *residual_bf16,
	uint32_t rows, uint32_t width, void *output_bf16)
{
	uint64_t count = (uint64_t)rows * width;
	dim3 grid((uint32_t)((count + SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS - 1u) /
		SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS));
	if ( value_bf16 == 0 || gate_bf16 == 0 || residual_bf16 == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3GateResidualKernel<<<grid,SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)value_bf16,(const __nv_bfloat16 *)gate_bf16,
		(const __nv_bfloat16 *)residual_bf16,width,count,
		(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxH3SiluMul(cudaStream_t stream,
	const void *gate_up_bf16, uint32_t rows, uint32_t ffn, void *output_bf16)
{
	uint32_t count = rows * ffn;
	dim3 grid((count + SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS - 1u) /
		SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS);
	if ( gate_up_bf16 == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3SiluMulKernel<<<grid,SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)gate_up_bf16,ffn,count,(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxH3Conv2d3x3Reflect(cudaStream_t stream,
	const void *input_bf16, const void *weight_bf16, const void *bias_bf16,
	uint32_t in_channels, uint32_t out_channels, uint32_t height, uint32_t width,
	void *output_bf16)
{
	uint64_t count = (uint64_t)out_channels * height * width;
	dim3 grid((uint32_t)((count + SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS - 1u) /
		SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS));
	if ( input_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3Conv2d3x3ReflectKernel<<<grid,SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)input_bf16,(const __nv_bfloat16 *)weight_bf16,
		(const __nv_bfloat16 *)bias_bf16,in_channels,out_channels,height,width,
		(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxH3Conv1dDilated(cudaStream_t stream,
	const void *input_bf16, const void *weight_bf16, const void *bias_bf16,
	uint32_t channels, uint32_t length, uint32_t dilation, void *output_bf16)
{
	uint64_t count = (uint64_t)channels * length;
	dim3 grid((uint32_t)((count + SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS - 1u) /
		SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS));
	if ( input_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3Conv1dDilatedKernel<<<grid,SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)input_bf16,(const __nv_bfloat16 *)weight_bf16,
		(const __nv_bfloat16 *)bias_bf16,channels,length,dilation,
		(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMinimaxH3Snake(cudaStream_t stream, const void *input_bf16,
	const void *alpha_bf16, uint32_t channels, uint32_t length, void *output_bf16)
{
	uint64_t count = (uint64_t)channels * length;
	dim3 grid((uint32_t)((count + SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS - 1u) /
		SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS));
	if ( input_bf16 == 0 || alpha_bf16 == 0 || output_bf16 == 0 )
		return(cudaErrorInvalidValue);
	SparkMinimaxH3SnakeKernel<<<grid,SPARK_MINIMAX_H3_CUDA_ELEMENTWISE_THREADS,0,stream>>>(
		(const __nv_bfloat16 *)input_bf16,(const __nv_bfloat16 *)alpha_bf16,length,
		count,(__nv_bfloat16 *)output_bf16);
	return(cudaGetLastError());
}
