#pragma once

#include <stdint.h>

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#define LM_POLY_NORM_PARAMS 4u
#define LM_POLY_NORM_STAGED_VECS 6u

template<uint32_t THREADS>
static __device__ __forceinline__ void LmPolyNormBlockSum3(float *s1, float *s2, float *s3, float (*reduction)[THREADS / LM_WARP_LANES])
{
	constexpr uint32_t WARPS = THREADS / LM_WARP_LANES;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp = threadIdx.x / LM_WARP_LANES;
	float v1 = *s1, v2 = *s2, v3 = *s3;
#pragma unroll
	for ( uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u )
	{
		v1 += __shfl_down_sync(0xffffffffu, v1, offset);
		v2 += __shfl_down_sync(0xffffffffu, v2, offset);
		v3 += __shfl_down_sync(0xffffffffu, v3, offset);
	}
	if constexpr ( WARPS == 1u )
	{
		*s1 = __shfl_sync(0xffffffffu, v1, 0u);
		*s2 = __shfl_sync(0xffffffffu, v2, 0u);
		*s3 = __shfl_sync(0xffffffffu, v3, 0u);
		return;
	}
	if ( lane == 0u )
	{
		reduction[0][warp] = v1;
		reduction[1][warp] = v2;
		reduction[2][warp] = v3;
	}
	__syncthreads();
	if ( warp == 0u )
	{
		v1 = lane < WARPS ? reduction[0][lane] : 0.0f;
		v2 = lane < WARPS ? reduction[1][lane] : 0.0f;
		v3 = lane < WARPS ? reduction[2][lane] : 0.0f;
#pragma unroll
		for ( uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u )
		{
			v1 += __shfl_down_sync(0xffffffffu, v1, offset);
			v2 += __shfl_down_sync(0xffffffffu, v2, offset);
			v3 += __shfl_down_sync(0xffffffffu, v3, offset);
		}
		if ( lane == 0u )
		{
			reduction[0][0] = v1;
			reduction[1][0] = v2;
			reduction[2][0] = v3;
		}
	}
	__syncthreads();
	*s1 = reduction[0][0];
	*s2 = reduction[1][0];
	*s3 = reduction[2][0];
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint4 LmPolyNormWaveAccum(const uint16_t *gate, uint16_t *raw_wave, uint32_t wave_base, uint32_t width, float clamp, float *square, float *fourth, float *sixth)
{
	const uint32_t offset = wave_base + threadIdx.x * 8u;
	uint4 vec = make_uint4(0u, 0u, 0u, 0u);
	if ( offset < width )
	{
		vec = __ldg((const uint4 *)(gate + offset));
		*(uint4 *)(raw_wave + threadIdx.x * 8u) = vec;
	}
	__syncthreads();
#pragma unroll
	for ( uint32_t s = 0u; s < 8u; ++s )
	{
		if ( wave_base + s * THREADS + threadIdx.x < width )
		{
			const float g = SparkMathClamp(LmBf16ToFloat(raw_wave[s * THREADS + threadIdx.x]), clamp), g2 = g * g, g3 = g2 * g;
			*square += g * g;
			*fourth += g2 * g2;
			*sixth += g3 * g3;
		}
	}
	if ( wave_base + THREADS * 8u < width )
		__syncthreads();
	return vec;
}

template<bool ROUTED>
static __device__ __forceinline__ void LmPolyNormStore8(uint4 gate_vec, uint4 up_vec, float clamp, float d1, float d2, float d3, float w0, float w1, float w2, float bias, float out_scale, uint16_t *dst)
{
	float gv[8], uv[8], out[8];
	LmLoadBf16x8AsFloat(gate_vec, gv);
	LmLoadBf16x8AsFloat(up_vec, uv);
#pragma unroll
	for ( uint32_t k = 0u; k < 8u; ++k )
	{
		const float g = SparkMathClamp(gv[k], clamp), u = SparkMathClamp(uv[k], clamp);
		const float value = SparkMathPolyNorm(g, d1, d2, d3, w0, w1, w2, bias) * u;
		out[k] = ROUTED ? (SparkMathClamp(value, clamp) * out_scale) : (LmBf16ToFloat(LmFloatToBf16(value)) * out_scale);
	}
	*(uint4 *)dst = LmPackUint4FloatToBf16(out);
}

template<uint32_t THREADS, bool ROUTED>
__global__ __launch_bounds__(THREADS, 2) void LmPolyNormMulKernel(const uint16_t *__restrict__ gate_up_bf16, const float *__restrict__ params_f32,
	const uint32_t *__restrict__ param_of_row, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t width, float clamp, float bias_clamp,
	float out_scale, float epsilon)
{
	__shared__ float reduction[3u][THREADS / LM_WARP_LANES];
	__shared__ __align__(16) uint16_t raw_wave[THREADS * 8u];
	LmDependentWait();
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *gate = gate_up_bf16 + (uint64_t)row * 2u * width, *up = gate + width;
	const float *params = params_f32 + (uint64_t)LM_POLY_NORM_PARAMS * (param_of_row != 0 ? __ldg(param_of_row + row) : 0u);
	float square = 0.0f, fourth = 0.0f, sixth = 0.0f;
	const float count = (float)width;
	const float w0 = SparkMathSigmoid(__ldg(params + 0u)), w1 = SparkMathSigmoid(__ldg(params + 1u)), w2 = SparkMathSigmoid(__ldg(params + 2u));
	const float p3 = __ldg(params + 3u);
	const float bias = ROUTED ? SparkMathClamp(p3, bias_clamp) : p3;
	if ( ((width & 7u) == 0u) && LmIsAligned16(gate) && LmIsAligned16(up) && LmIsAligned16(out_bf16 + (uint64_t)row * width) )
	{
		if ( width <= THREADS * 8u * LM_POLY_NORM_STAGED_VECS )
		{
			uint4 staged_g[LM_POLY_NORM_STAGED_VECS];
#pragma unroll
			for ( uint32_t slot = 0u; slot < LM_POLY_NORM_STAGED_VECS; ++slot )
			{
				const uint32_t wave_base = slot * THREADS * 8u;
				if ( wave_base >= width )
					staged_g[slot] = make_uint4(0u, 0u, 0u, 0u);
				else
					staged_g[slot] = LmPolyNormWaveAccum<THREADS>(gate, raw_wave, wave_base, width, clamp, &square, &fourth, &sixth);
			}
			LmPolyNormBlockSum3<THREADS>(&square, &fourth, &sixth, reduction);
			const float d1 = sqrtf(square / count + epsilon), d2 = sqrtf(fourth / count + epsilon), d3 = sqrtf(sixth / count + epsilon);
			uint16_t *row_out = out_bf16 + (uint64_t)row * width;
#pragma unroll
			for ( uint32_t slot = 0u; slot < LM_POLY_NORM_STAGED_VECS; ++slot )
			{
				const uint32_t off = (slot * THREADS + threadIdx.x) * 8u;
				if ( off < width )
					LmPolyNormStore8<ROUTED>(staged_g[slot], __ldg((const uint4 *)(up + off)), clamp, d1, d2, d3, w0, w1, w2, bias, out_scale, row_out + off);
			}
			LmDependentRelease();
			return;
		}
		for ( uint32_t wave_base = 0u; wave_base < width; wave_base += THREADS * 8u )
			(void)LmPolyNormWaveAccum<THREADS>(gate, raw_wave, wave_base, width, clamp, &square, &fourth, &sixth);
		LmPolyNormBlockSum3<THREADS>(&square, &fourth, &sixth, reduction);
		const float d1 = sqrtf(square / count + epsilon), d2 = sqrtf(fourth / count + epsilon), d3 = sqrtf(sixth / count + epsilon);
		uint16_t *row_out = out_bf16 + (uint64_t)row * width;
		for ( uint32_t off = threadIdx.x * 8u; off < width; off += THREADS * 8u )
			LmPolyNormStore8<ROUTED>(__ldg((const uint4 *)(gate + off)), __ldg((const uint4 *)(up + off)), clamp, d1, d2, d3, w0, w1, w2, bias, out_scale, row_out + off);
		LmDependentRelease();
		return;
	}
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float g = SparkMathClamp(LmBf16ToFloat(__ldg(gate + index)), clamp), g2 = g * g, g3 = g2 * g;
		square += g * g;
		fourth += g2 * g2;
		sixth += g3 * g3;
	}
	LmPolyNormBlockSum3<THREADS>(&square, &fourth, &sixth, reduction);
	const float d1 = sqrtf(square / count + epsilon), d2 = sqrtf(fourth / count + epsilon), d3 = sqrtf(sixth / count + epsilon);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float g = SparkMathClamp(LmBf16ToFloat(__ldg(gate + index)), clamp), u = SparkMathClamp(LmBf16ToFloat(__ldg(up + index)), clamp);
		const float poly = SparkMathPolyNorm(g, d1, d2, d3, w0, w1, w2, bias);
		const float value = poly * u;
		out_bf16[(uint64_t)row * width + index] = ROUTED ? LmFloatToBf16(SparkMathClamp(value, clamp) * out_scale)
			: LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value)) * out_scale);
	}
	LmDependentRelease();
}

template<bool ROUTED>
static SparkGpuError LmPolyNormMulLaunch(const uint16_t *gate_up, const float *params, const uint32_t *param_of_row, uint16_t *out, uint32_t rows,
	uint32_t width, float clamp, float bias_clamp, float out_scale, float epsilon, SparkGpuStream stream)
{
	if ( rows == 0u )
		return SPARK_GPU_SUCCESS;
	LM_LAUNCH_DEPENDENT((LmPolyNormMulKernel<256u, ROUTED>), rows, 256u, 0, stream, gate_up, params, param_of_row, out, rows, width, clamp, bias_clamp, out_scale, epsilon);
	return SparkGpuPeekAtLastError();
}
