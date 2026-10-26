#include <metal_stdlib>
#include "inference/kernels/metal/kernel_params.h"

using namespace metal;

static inline float SparkBf16ToFloat(ushort bits)
{
	return as_type<float>(uint(bits) << 16);
}

static inline ushort SparkFloatToBf16(float value)
{
	const uint bits = as_type<uint>(value);
	if ( (bits & 0x7fffffffu) > 0x7f800000u )
		return ushort((bits >> 16) | 0x0040u);
	return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static inline float SparkRoundBf16(float value)
{
	return SparkBf16ToFloat(SparkFloatToBf16(value));
}

static inline float SparkSilu(float value)
{
	return value / (1.0f + exp(-value));
}

static inline float SparkSigmoid(float value)
{
	return 1.0f / (1.0f + exp(-value));
}

static inline float4 SparkBf16x4(ushort4 bits)
{
	return float4(SparkBf16ToFloat(bits.x), SparkBf16ToFloat(bits.y), SparkBf16ToFloat(bits.z), SparkBf16ToFloat(bits.w));
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
