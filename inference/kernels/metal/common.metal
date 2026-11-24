#include <metal_stdlib>
#include "inference/kernels/metal/kernel_params.h"
#include "inference/kernels/kernel_math.cuh"

using namespace metal;

static inline float4 SparkBf16x4(ushort4 bits)
{
	return float4(SparkMathBf16ToFloat(bits.x), SparkMathBf16ToFloat(bits.y), SparkMathBf16ToFloat(bits.z), SparkMathBf16ToFloat(bits.w));
}

static inline float4 SparkUnpack4(uint2 bits)
{
	return float4(as_type<float>(bits.x << 16u), as_type<float>(bits.x & 0xffff0000u), as_type<float>(bits.y << 16u), as_type<float>(bits.y & 0xffff0000u));
}

static inline void SparkUnpack8(uint4 bits, thread float4 &low, thread float4 &high)
{
	low = float4(as_type<float>(bits.x << 16u), as_type<float>(bits.x & 0xffff0000u), as_type<float>(bits.y << 16u), as_type<float>(bits.y & 0xffff0000u));
	high = float4(as_type<float>(bits.z << 16u), as_type<float>(bits.z & 0xffff0000u), as_type<float>(bits.w << 16u), as_type<float>(bits.w & 0xffff0000u));
}

static inline uint SparkPack2(float a, float b)
{
	return uint(SparkMathFloatToBf16(a)) | (uint(SparkMathFloatToBf16(b)) << 16u);
}

static inline uint2 SparkPack4(float4 v)
{
	return uint2(SparkPack2(v.x, v.y), SparkPack2(v.z, v.w));
}

static inline uint4 SparkPack8(float4 low, float4 high)
{
	return uint4(SparkPack2(low.x, low.y), SparkPack2(low.z, low.w), SparkPack2(high.x, high.y), SparkPack2(high.z, high.w));
}

static inline float4 SparkRoundBf16x4(float4 v)
{
	return float4(SparkMathRoundBf16(v.x), SparkMathRoundBf16(v.y), SparkMathRoundBf16(v.z), SparkMathRoundBf16(v.w));
}

static inline float4 SparkResidualSum4(float4 x, float4 a)
{
	return SparkRoundBf16x4(x + a);
}

static inline float4 SparkNormGainRounded4(float4 v, float4 w)
{
	return SparkRoundBf16x4(v) * w;
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
