#include <metal_stdlib>
#include "inference/kernels/metal/kernel_params.h"
#include "inference/kernels/kernel_math.cuh"

using namespace metal;

static inline float4 SparkBf16x4(ushort4 bits)
{
	return float4(SparkMathBf16ToFloat(bits.x), SparkMathBf16ToFloat(bits.y), SparkMathBf16ToFloat(bits.z), SparkMathBf16ToFloat(bits.w));
}

static inline float SparkGroupSum(float value, threadgroup float *scratch, uint lane, uint simd, uint simds)
{
	value = simd_sum(value);
	if ( lane == 0u )
		scratch[simd] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float total = lane < simds ? scratch[lane] : 0.0f;
	total = simd_sum(total);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return total;
}

static inline float SparkGroupMax(float value, threadgroup float *scratch, uint lane, uint simd, uint simds)
{
	value = simd_max(value);
	if ( lane == 0u )
		scratch[simd] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float total = lane < simds ? scratch[lane] : -INFINITY;
	total = simd_max(total);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return total;
}

template <bool MX>
static inline float SparkFp8Block(constant SparkGemvParams &p, device const uchar *scale, ulong row, uint k)
{
	return(MX ? SparkMathUe8m0ToFloat(scale[row * p.scale_stride + k / SPARK_MXFP8_GROUP]) :
		((device const float *)scale)[row * p.scale_stride + k / SPARK_FP8_SCALE_K]);
}
