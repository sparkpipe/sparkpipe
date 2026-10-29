#ifndef SPARKPIPE_INFERENCE_KERNELS_KERNEL_MATH_CUH
#define SPARKPIPE_INFERENCE_KERNELS_KERNEL_MATH_CUH

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#define SPARK_MATH_FN static inline
typedef uint SparkMathU32;
typedef ushort SparkMathU16;
typedef uchar SparkMathU8;
#define SPARK_MATH_AS_FLOAT(bits) as_type<float>(bits)
#define SPARK_MATH_AS_BITS(value) as_type<uint>(value)
#define SPARK_MATH_EXP(x) metal::exp(x)
#define SPARK_MATH_EXP_FAST(x) metal::exp(x)
#define SPARK_MATH_LOG(x) metal::log(x)
#define SPARK_MATH_LOG_FAST(x) metal::log(x)
#define SPARK_MATH_TANH(x) metal::tanh(x)
#define SPARK_MATH_LDEXP(x, e) metal::ldexp(x, e)
#define SPARK_MATH_SQRT(x) metal::sqrt(x)
#define SPARK_MATH_RSQRT(x) metal::rsqrt(x)
#define SPARK_MATH_FABS(x) metal::fabs(x)
#define SPARK_MATH_FMAX(a, b) metal::fmax(a, b)
#elif defined(__CUDACC__) || defined(LM_HOST_CUDA_PRELUDE)
#include <stdint.h>
#if defined(__CUDACC__)
#include <cuda_bf16.h>
#endif
#define SPARK_MATH_FN static __device__ __forceinline__
typedef uint32_t SparkMathU32;
typedef uint16_t SparkMathU16;
typedef uint8_t SparkMathU8;
#define SPARK_MATH_AS_FLOAT(bits) __uint_as_float(bits)
#define SPARK_MATH_AS_BITS(value) __float_as_uint(value)
#define SPARK_MATH_EXP(x) expf(x)
#define SPARK_MATH_EXP_FAST(x) __expf(x)
#define SPARK_MATH_LOG(x) logf(x)
#define SPARK_MATH_LOG_FAST(x) __logf(x)
#define SPARK_MATH_TANH(x) tanhf(x)
#define SPARK_MATH_LDEXP(x, e) ldexpf(x, e)
#define SPARK_MATH_SQRT(x) sqrtf(x)
#define SPARK_MATH_RSQRT(x) rsqrtf(x)
#define SPARK_MATH_FABS(x) fabsf(x)
#define SPARK_MATH_FMAX(a, b) fmaxf(a, b)
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

SPARK_MATH_FN float SparkMathRmsScale(float total, float count, float epsilon)
{
	return(SPARK_MATH_RSQRT(total / count + epsilon));
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

SPARK_MATH_FN float SparkMathSwishFast(float value)
{
	return(value * SparkMathSigmoidFast(value));
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

SPARK_MATH_FN float SparkMathGeluTanhScaleFirst(float value)
{
	return(0.5f * value * (1.0f + SPARK_MATH_TANH(0.7978845608028654f * (value + 0.044715f * value * value * value))));
}

SPARK_MATH_FN float SparkMathSoftcap(float value, float cap)
{
	return(cap * SPARK_MATH_TANH(value / cap));
}

SPARK_MATH_FN float SparkMathHyperGate(float value, float divisor)
{
	const float scaled = SparkMathRoundBf16(value / divisor);
	return(scaled * SparkMathSigmoid(scaled));
}

SPARK_MATH_FN float SparkMathHyperMixGate(float up)
{
	return(SparkMathRoundBf16(SparkMathSigmoid(up)));
}

SPARK_MATH_FN float SparkMathHyperMixTerm(float gate, float normed)
{
	return(SparkMathRoundBf16(gate * normed));
}

SPARK_MATH_FN float SparkMathHyperInjectWeight(float logit, float streams)
{
	return(SparkMathRoundBf16(2.0f * SparkMathRoundBf16(SparkMathSigmoid(SparkMathRoundBf16(logit / streams)))));
}

SPARK_MATH_FN float SparkMathHyperInjected(float output, float weight)
{
	return(SparkMathRoundBf16(output * weight));
}

SPARK_MATH_FN float SparkMathHyperKeyGate(float total, SparkMathU32 width)
{
	const float score = SparkMathRoundBf16(SparkMathRoundBf16(total) / SPARK_MATH_SQRT((float)width));
	const float magnitude = SparkMathRoundBf16(SPARK_MATH_SQRT(SPARK_MATH_FMAX(SPARK_MATH_FABS(score),1e-6f)));
	return(SparkMathRoundBf16(SparkMathSigmoid(score < 0.0f ? -magnitude : (score > 0.0f ? magnitude : 0.0f))));
}

SPARK_MATH_FN float SparkMathHyperAddend(float gated, float conv)
{
	return(SparkMathRoundBf16(gated + conv));
}

SPARK_MATH_FN float SparkMathE4m3ToFloat(SparkMathU32 value)
{
	const SparkMathU32 sign = (value & 0x80u) << 24u, exponent = (value >> 3u) & 15u, mantissa = value & 7u;
	if ( (value & 0x7fu) == 0x7fu )
		return(SPARK_MATH_AS_FLOAT(sign | 0x7fc00000u));
	if ( exponent == 0u )
		return(SPARK_MATH_AS_FLOAT(sign | SPARK_MATH_AS_BITS((float)mantissa * 0.001953125f)));
	return(SPARK_MATH_AS_FLOAT(sign | ((exponent + 120u) << 23u) | (mantissa << 20u)));
}

SPARK_MATH_FN float SparkMathE2m1ToFloat(SparkMathU32 nibble)
{
	SparkMathU32 exponent = (nibble >> 1u) & 3u, sign = (nibble & 8u) << 28u;
	SparkMathU32 magnitude = exponent != 0u ? ((exponent + 126u) << 23u) | ((nibble & 1u) << 22u) : ((nibble & 1u) != 0u ? 0x3f000000u : 0u);
	return(SPARK_MATH_AS_FLOAT(sign | magnitude));
}

SPARK_MATH_FN float SparkMathUe8m0ToFloat(SparkMathU8 value)
{
	return(SPARK_MATH_AS_FLOAT(value == 0u ? 0x00400000u : (value == 0xffu ? 0x7fffffffu : (SparkMathU32)value << 23u)));
}

SPARK_MATH_FN float SparkMathUe4m3ToFloat(SparkMathU8 value)
{
	SparkMathU32 biased,mantissa;
	value &= 0x7fu;
	biased = (SparkMathU32)value >> 3u;
	mantissa = (SparkMathU32)value & 7u;
	if ( biased == 0u )
		return(SPARK_MATH_LDEXP((float)mantissa,-9));
	if ( biased == 15u && mantissa == 7u )
		return(448.0f);
	return(SPARK_MATH_LDEXP(1.0f + ((float)mantissa / 8.0f),(int)biased - 7));
}

#endif
