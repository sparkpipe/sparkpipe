#pragma once

#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"

#define LM_POLY_NORM_PARAMS 4u

static __device__ __forceinline__ float LmPolyNormClamp(float value, float limit)
{
	return fminf(fmaxf(value, -limit), limit);
}

static __device__ __forceinline__ float LmPolyNormSigmoid(float value)
{
	return 1.0f / (1.0f + expf(-value));
}

template<uint32_t THREADS, bool ROUTED>
__global__ __launch_bounds__(THREADS) void LmPolyNormMulKernel(const uint16_t *__restrict__ gate_up_bf16, const float *__restrict__ params_f32,
	const uint32_t *__restrict__ param_of_row, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t width, float clamp, float bias_clamp,
	float out_scale, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
		return;
	const uint16_t *gate = gate_up_bf16 + (uint64_t)row * 2u * width, *up = gate + width;
	const float *params = params_f32 + (uint64_t)LM_POLY_NORM_PARAMS * (param_of_row != 0 ? param_of_row[row] : 0u);
	float square = 0.0f, fourth = 0.0f, sixth = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float g = LmPolyNormClamp(LmBf16ToFloat(gate[index]), clamp), g2 = g * g, g3 = g2 * g;
		square += g * g;
		fourth += g2 * g2;
		sixth += g3 * g3;
	}
	square = LmBlockSum<THREADS>(square, reduction);
	fourth = LmBlockSum<THREADS>(fourth, reduction);
	sixth = LmBlockSum<THREADS>(sixth, reduction);
	const float count = (float)width;
	const float d1 = sqrtf(square / count + epsilon), d2 = sqrtf(fourth / count + epsilon), d3 = sqrtf(sixth / count + epsilon);
	const float w0 = LmPolyNormSigmoid(params[0]), w1 = LmPolyNormSigmoid(params[1]), w2 = LmPolyNormSigmoid(params[2]);
	const float bias = ROUTED ? LmPolyNormClamp(params[3], bias_clamp) : params[3];
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float g = LmPolyNormClamp(LmBf16ToFloat(gate[index]), clamp), u = LmPolyNormClamp(LmBf16ToFloat(up[index]), clamp);
		const float poly = w0 * (g * g * g / d3) + w1 * (g * g / d2) + w2 * (g / d1) + bias;
		const float value = poly * u;
		out_bf16[(uint64_t)row * width + index] = ROUTED ? LmFloatToBf16(LmPolyNormClamp(value, clamp) * out_scale)
			: LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value)) * out_scale);
	}
}

template<bool ROUTED>
static cudaError_t LmPolyNormMulLaunch(const uint16_t *gate_up, const float *params, const uint32_t *param_of_row, uint16_t *out, uint32_t rows,
	uint32_t width, float clamp, float bias_clamp, float out_scale, float epsilon, cudaStream_t stream)
{
	if ( rows == 0u )
		return cudaSuccess;
	LmPolyNormMulKernel<256u, ROUTED><<<rows, 256u, 0, stream>>>(gate_up, params, param_of_row, out, rows, width, clamp, bias_clamp, out_scale, epsilon);
	return cudaPeekAtLastError();
}
