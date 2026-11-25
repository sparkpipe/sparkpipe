#pragma once

#include <stdint.h>

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

__global__ __launch_bounds__(256u, 2) void LmDifferentialGateKernel(const uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ lambda_bf16, uint32_t lambda_stride,
	const uint16_t *__restrict__ gate_bf16, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t groups, uint32_t signal, uint32_t dim)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y, width = groups * signal * dim;
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const uint32_t head = element / dim, d = element % dim, group = head / signal, member = head % signal;
		const uint16_t *heads = heads_bf16 + (uint64_t)row * groups * (signal + 1u) * dim + (uint64_t)group * (signal + 1u) * dim;
		const float sig = LmBf16ToFloat(__ldg(heads + (uint64_t)member * dim + d)), noise = LmBf16ToFloat(__ldg(heads + (uint64_t)signal * dim + d));
		const float combined = SparkMathDifferentialCombine(sig, noise, LmBf16ToFloat(__ldg(lambda_bf16 + (uint64_t)row * lambda_stride + head)));
		out_bf16[(uint64_t)row * width + element] = LmFloatToBf16(combined * SparkMathRoundBf16(SparkMathSigmoid(LmBf16ToFloat(__ldg(gate_bf16 + (uint64_t)row * width + element)))));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(64u, 4) void LmDifferentialGateVecKernel(const uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ lambda_bf16, uint32_t lambda_stride,
	const uint16_t *__restrict__ gate_bf16, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t groups, uint32_t signal, uint32_t dim)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y, dim_vecs = dim >> 3u, total_vecs = groups * signal * dim_vecs;
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const uint64_t row_heads_base = (uint64_t)row * groups * (signal + 1u) * dim;
	const uint64_t row_out_base = (uint64_t)row * groups * signal * dim;
	for ( uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vecs; v += gridDim.x * blockDim.x )
	{
		const uint32_t head = v / dim_vecs, d = (v % dim_vecs) << 3u, group = head / signal, member = head % signal;
		const uint16_t *heads = heads_bf16 + row_heads_base + (uint64_t)group * (signal + 1u) * dim;
		const float lambda = LmBf16ToFloat(__ldg(lambda_bf16 + (uint64_t)row * lambda_stride + head));
		float sig[8], noise[8], gate[8], out[8];
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(heads + (uint64_t)member * dim + d)), sig);
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(heads + (uint64_t)signal * dim + d)), noise);
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(gate_bf16 + row_out_base + ((uint64_t)v << 3u))), gate);
#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
		{
			const float combined = SparkMathDifferentialCombine(sig[k], noise[k], lambda);
			out[k] = combined * SparkMathRoundBf16(SparkMathSigmoid(gate[k]));
		}
		*(uint4 *)(out_bf16 + row_out_base + ((uint64_t)v << 3u)) = LmPackUint4FloatToBf16(out);
	}
	LmDependentRelease();
}

static SparkGpuError LmDifferentialGateLaunch(const uint16_t *heads, const uint16_t *lambda, uint32_t lambda_stride, const uint16_t *gate, uint16_t *out,
	uint32_t rows, uint32_t groups, uint32_t signal, uint32_t dim, SparkGpuStream stream)
{
	if ( rows == 0u )
		return SPARK_GPU_SUCCESS;
	if ( (dim & 7u) == 0u && LmIsAligned16(heads) && LmIsAligned16(gate) && LmIsAligned16(out) )
	{
		const uint32_t total_vecs = groups * signal * (dim >> 3u);
		LM_LAUNCH_DEPENDENT((LmDifferentialGateVecKernel), dim3((total_vecs + 63u) / 64u, rows), 64u, 0, stream, heads, lambda, lambda_stride, gate, out, rows, groups, signal, dim);
		return SparkGpuPeekAtLastError();
	}
	const uint32_t width = groups * signal * dim;
	LM_LAUNCH_DEPENDENT((LmDifferentialGateKernel), dim3((width + 255u) / 256u, rows), 256u, 0, stream, heads, lambda, lambda_stride, gate, out, rows, groups, signal, dim);
	return SparkGpuPeekAtLastError();
}
