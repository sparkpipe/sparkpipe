#ifndef SPARKPIPE_INFERENCE_KERNELS_KERNEL_MATH_CUH
#define SPARKPIPE_INFERENCE_KERNELS_KERNEL_MATH_CUH

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#define SPARK_MATH_FN static inline
typedef uint SparkMathU32;
typedef ushort SparkMathU16;
#define SPARK_MATH_AS_FLOAT(bits) as_type<float>(bits)
#define SPARK_MATH_AS_BITS(value) as_type<uint>(value)
#define SPARK_MATH_EXP(x) metal::exp(x)
#define SPARK_MATH_EXP_FAST(x) metal::exp(x)
#define SPARK_MATH_LOG(x) metal::log(x)
#define SPARK_MATH_LOG_FAST(x) metal::log(x)
#define SPARK_MATH_TANH(x) metal::tanh(x)
#elif defined(__CUDACC__) || defined(LM_HOST_CUDA_PRELUDE)
#include <stdint.h>
#if defined(__CUDACC__)
#include <cuda_bf16.h>
#endif
#define SPARK_MATH_FN static __device__ __forceinline__
typedef uint32_t SparkMathU32;
typedef uint16_t SparkMathU16;
#define SPARK_MATH_AS_FLOAT(bits) __uint_as_float(bits)
#define SPARK_MATH_AS_BITS(value) __float_as_uint(value)
#define SPARK_MATH_EXP(x) expf(x)
#define SPARK_MATH_EXP_FAST(x) __expf(x)
#define SPARK_MATH_LOG(x) logf(x)
#define SPARK_MATH_LOG_FAST(x) __logf(x)
#define SPARK_MATH_TANH(x) tanhf(x)
#else
#error "kernel_math.cuh is kernel code for the CUDA and Metal compilers"
#endif

SPARK_MATH_FN float SparkMathBf16ToFloat(SparkMathU16 bits)
{
	return(SPARK_MATH_AS_FLOAT((SparkMathU32)bits << 16u));
}

SPARK_MATH_FN SparkMathU16 SparkMathFloatToBf16(float value)
{
#if defined(__CUDA_ARCH__)
	return(__bfloat16_as_ushort(__float2bfloat16_rn(value)));
#else
	const SparkMathU32 bits = SPARK_MATH_AS_BITS(value);
	return((bits & 0x7fffffffu) > 0x7f800000u ? (SparkMathU16)0x7fffu : (SparkMathU16)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
#endif
}

SPARK_MATH_FN float SparkMathRoundBf16(float value)
{
	return(SparkMathBf16ToFloat(SparkMathFloatToBf16(value)));
}

SPARK_MATH_FN float SparkMathExp2Fast(float value)
{
#if defined(__CUDA_ARCH__)
	float result;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
	return(result);
#elif defined(__METAL_VERSION__)
	return(metal::exp2(value));
#else
	return(exp2f(value));
#endif
}

SPARK_MATH_FN float SparkMathScaledSigmoid(float scale, float value)
{
	return(scale / (1.0f + SPARK_MATH_EXP(-value)));
}

SPARK_MATH_FN float SparkMathScaledSigmoidFast(float scale, float value)
{
	return(scale / (1.0f + SPARK_MATH_EXP_FAST(-value)));
}

SPARK_MATH_FN float SparkMathSigmoid(float value)
{
	return(SparkMathScaledSigmoid(1.0f,value));
}

SPARK_MATH_FN float SparkMathSigmoidFast(float value)
{
	return(SparkMathScaledSigmoidFast(1.0f,value));
}

SPARK_MATH_FN float SparkMathSilu(float value)
{
	return(SparkMathScaledSigmoid(value,value));
}

SPARK_MATH_FN float SparkMathSiluFast(float value)
{
	return(SparkMathScaledSigmoidFast(value,value));
}

SPARK_MATH_FN float SparkMathSoftplus(float value)
{
	return(value > 20.0f ? value : SPARK_MATH_LOG(1.0f + SPARK_MATH_EXP(value)));
}

SPARK_MATH_FN float SparkMathSoftplusFast(float value)
{
	return(value > 20.0f ? value : SPARK_MATH_LOG_FAST(1.0f + SPARK_MATH_EXP_FAST(value)));
}

SPARK_MATH_FN float SparkMathGatedDeltaDecay(float log_scale, float shifted)
{
	const float softplus = SparkMathSoftplus(shifted);
	return(SPARK_MATH_EXP(-SPARK_MATH_EXP(log_scale) * softplus));
}

SPARK_MATH_FN float SparkMathGatedDeltaDecayFast(float log_scale, float shifted)
{
	const float softplus = SparkMathSoftplusFast(shifted);
	return(SPARK_MATH_EXP_FAST(-SPARK_MATH_EXP_FAST(log_scale) * softplus));
}

SPARK_MATH_FN float SparkMathGeluTanh(float value)
{
	const float cube = value * value * value;
	return(0.5f * value * (1.0f + SPARK_MATH_TANH(0.7978845608028654f * (value + 0.044715f * cube))));
}

SPARK_MATH_FN float SparkMathSoftcap(float value, float cap)
{
	return(cap * SPARK_MATH_TANH(value / cap));
}

#endif
