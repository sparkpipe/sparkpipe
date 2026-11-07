#pragma once

#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kernel_math.cuh"

__global__ void LmDifferentialGateKernel(const uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ lambda_bf16, uint32_t lambda_stride,
	const uint16_t *__restrict__ gate_bf16, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t groups, uint32_t signal, uint32_t dim)
{
	const uint32_t row = blockIdx.y, width = groups * signal * dim;
	if ( row >= rows )
		return;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const uint32_t head = element / dim, d = element % dim, group = head / signal, member = head % signal;
		const uint16_t *heads = heads_bf16 + (uint64_t)row * groups * (signal + 1u) * dim + (uint64_t)group * (signal + 1u) * dim;
		const float sig = LmBf16ToFloat(heads[(uint64_t)member * dim + d]), noise = LmBf16ToFloat(heads[(uint64_t)signal * dim + d]);
		const float combined = SparkMathDifferentialCombine(sig, noise, LmBf16ToFloat(lambda_bf16[(uint64_t)row * lambda_stride + head]));
		out_bf16[(uint64_t)row * width + element] = LmFloatToBf16(combined * SparkMathRoundBf16(SparkMathSigmoid(LmBf16ToFloat(gate_bf16[(uint64_t)row * width + element]))));
	}
}

static cudaError_t LmDifferentialGateLaunch(const uint16_t *heads, const uint16_t *lambda, uint32_t lambda_stride, const uint16_t *gate, uint16_t *out,
	uint32_t rows, uint32_t groups, uint32_t signal, uint32_t dim, cudaStream_t stream)
{
	if ( rows == 0u )
		return cudaSuccess;
	const uint32_t width = groups * signal * dim;
	LmDifferentialGateKernel<<<dim3((width + 255u) / 256u, rows), 256u, 0, stream>>>(heads, lambda, lambda_stride, gate, out, rows, groups, signal, dim);
	return cudaPeekAtLastError();
}
