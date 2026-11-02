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
#define SPARK_MATH_COS_FAST(x) metal::cos(x)
#define SPARK_MATH_SIN_FAST(x) metal::sin(x)
#define SPARK_MATH_TANH(x) metal::tanh(x)
#define SPARK_MATH_LDEXP(x, e) metal::ldexp(x, e)
#define SPARK_MATH_SQRT(x) metal::sqrt(x)
#define SPARK_MATH_RSQRT(x) metal::rsqrt(x)
#define SPARK_MATH_FABS(x) metal::fabs(x)
#define SPARK_MATH_FMAX(a, b) metal::fmax(a, b)
#define SPARK_MATH_FMIN(a, b) metal::fmin(a, b)
#define SPARK_MATH_RINT(x) metal::rint(x)
#define SPARK_MATH_COPYSIGN(x, y) metal::copysign(x, y)
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
#define SPARK_MATH_COS_FAST(x) __cosf(x)
#define SPARK_MATH_SIN_FAST(x) __sinf(x)
#define SPARK_MATH_TANH(x) tanhf(x)
#define SPARK_MATH_LDEXP(x, e) ldexpf(x, e)
#define SPARK_MATH_SQRT(x) sqrtf(x)
#define SPARK_MATH_RSQRT(x) rsqrtf(x)
#define SPARK_MATH_FABS(x) fabsf(x)
#define SPARK_MATH_FMAX(a, b) fmaxf(a, b)
#define SPARK_MATH_FMIN(a, b) fminf(a, b)
#define SPARK_MATH_RINT(x) rintf(x)
#define SPARK_MATH_COPYSIGN(x, y) copysignf(x, y)
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

SPARK_MATH_FN float SparkMathSwishFast(float value)
{
	return(value * SparkMathSigmoidFast(value));
}

SPARK_MATH_FN float SparkMathSoftplus(float value)
{
	return(value > 20.0f ? value : SPARK_MATH_LOG(1.0f + SPARK_MATH_EXP(value)));
}

SPARK_MATH_FN float SparkMathRopeCos(float angle, float scale)
{
	return(SPARK_MATH_COS_FAST(angle) * scale);
}

SPARK_MATH_FN float SparkMathRopeSin(float angle, float scale)
{
	return(SPARK_MATH_SIN_FAST(angle) * scale);
}

SPARK_MATH_FN float SparkMathRopeLow(float low, float high, float cosine, float sine)
{
	return((low * cosine) - (high * sine));
}

SPARK_MATH_FN float SparkMathRopeHigh(float low, float high, float cosine, float sine)
{
	return((low * sine) + (high * cosine));
}

SPARK_MATH_FN float SparkMathLog1p(float value)
{
#if defined(__METAL_VERSION__)
	const float shifted = 1.0f + value;
	return(shifted == 1.0f ? value : metal::log(shifted) * (value / (shifted - 1.0f)));
#else
	return(log1pf(value));
#endif
}

SPARK_MATH_FN float SparkMathSoftplusGate(float logit)
{
	return(SparkMathRoundBf16(logit > 20.0f ? logit : SparkMathLog1p(SPARK_MATH_EXP(logit))));
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

SPARK_MATH_FN float SparkMathBoundedDecay(float logit, float bias, float log_scale, float minimum_log_decay)
{
	const float scaled = SPARK_MATH_EXP_FAST(log_scale) * (logit + bias);
	return(SPARK_MATH_EXP_FAST(minimum_log_decay * SparkMathSigmoidFast(scaled)));
}

SPARK_MATH_FN float SparkMathClampedGlu(float activated, float up, float limit)
{
	return(SPARK_MATH_FMIN(activated, limit) * SPARK_MATH_FMIN(SPARK_MATH_FMAX(up, -limit), limit));
}

SPARK_MATH_FN float SparkMathGateLimitedGlu(float gate, float up, float limit)
{
	const float limited = SPARK_MATH_FMIN(gate, limit);
	const float clamped = SPARK_MATH_FMIN(SPARK_MATH_FMAX(up, -limit), limit);
	return(SparkMathSilu(limited) * clamped);
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

SPARK_MATH_FN float SparkMathSitu(float gate, float up, float beta, float linear_beta)
{
	const float activated = beta * SPARK_MATH_TANH(gate / beta) * SparkMathSigmoidFast(gate);
	return(activated * (linear_beta > 0.0f ? linear_beta * SPARK_MATH_TANH(up / linear_beta) : up));
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

SPARK_MATH_FN float SparkMathSqrtSoftplus(float value)
{
	const float softplus = value > 20.0f ? value : SparkMathLog1p(SPARK_MATH_EXP(value));
	return(SPARK_MATH_SQRT(softplus > 0.0f ? softplus : 0.0f));
}

SPARK_MATH_FN float SparkMathSignedSqrtGate(float dot, float left_square, float right_square, SparkMathU32 dimension, float epsilon, float floor)
{
	const float rstd = (1.0f / SPARK_MATH_SQRT(left_square / (float)dimension + epsilon)) * (1.0f / SPARK_MATH_SQRT(right_square / (float)dimension + epsilon));
	const float scaled = dot * rstd * (1.0f / SPARK_MATH_SQRT((float)dimension));
	const float magnitude = SPARK_MATH_SQRT(SPARK_MATH_FMAX(SPARK_MATH_FABS(scaled),floor));
	return(SparkMathSigmoid(SPARK_MATH_COPYSIGN(magnitude,scaled)));
}

SPARK_MATH_FN float SparkMathSoftmaxPool2(float first, float first_score, float second, float second_score)
{
	const float top = SPARK_MATH_FMAX(first_score,second_score);
	const float e0 = SPARK_MATH_EXP(first_score - top), e1 = SPARK_MATH_EXP(second_score - top);
	return((first * e0 + second * e1) / (e0 + e1));
}

SPARK_MATH_FN float SparkMathPow2CeilScale(float amax, float inverse_max)
{
	const SparkMathU32 bits = SPARK_MATH_AS_BITS(amax * inverse_max);
	const int exponent = (int)((bits >> 23u) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u ? 1 : 0);
	return(SPARK_MATH_AS_FLOAT((SparkMathU32)(exponent + 127) << 23u));
}

SPARK_MATH_FN float SparkMathRoundMinifloat(float value, int mantissa_bits, int minimum_exponent, float maximum)
{
	int exponent = (int)((SPARK_MATH_AS_BITS(value) >> 23u) & 0xffu) - 127;
	exponent = exponent < minimum_exponent ? minimum_exponent : exponent;
	const float step = SPARK_MATH_AS_FLOAT((SparkMathU32)(exponent - mantissa_bits + 127) << 23u);
	const float inverse = SPARK_MATH_AS_FLOAT((SparkMathU32)(mantissa_bits - exponent + 127) << 23u);
	return(SPARK_MATH_FMIN(SPARK_MATH_FMAX(SPARK_MATH_RINT(value * inverse) * step,-maximum),maximum));
}

SPARK_MATH_FN float SparkMathRoundE4m3(float value)
{
	return(SparkMathRoundMinifloat(value,3,-6,448.0f));
}

SPARK_MATH_FN float SparkMathRoundE2m1(float value)
{
	return(SparkMathRoundMinifloat(value,1,0,6.0f));
}

SPARK_MATH_FN float SparkMathFp8QdqScale(float amax)
{
	return(SparkMathPow2CeilScale(SPARK_MATH_FMAX(amax,1e-4f),1.0f / 448.0f));
}

SPARK_MATH_FN float SparkMathFp8Qdq(float value, float scale)
{
	return(SparkMathRoundE4m3(SPARK_MATH_FMIN(SPARK_MATH_FMAX(value / scale,-448.0f),448.0f)) * scale);
}

SPARK_MATH_FN float SparkMathFp4QdqScale(float amax)
{
	return(SparkMathRoundE4m3(SPARK_MATH_FMAX(amax,6.0f * 0.001953125f) / 6.0f));
}

SPARK_MATH_FN float SparkMathFp4Pow2QdqScale(float amax)
{
	return(SparkMathPow2CeilScale(SPARK_MATH_FMAX(amax,6.0f * 1.1754943508222875e-38f),1.0f / 6.0f));
}

SPARK_MATH_FN float SparkMathFp4Qdq(float value, float scale)
{
	return(SparkMathRoundE2m1(SPARK_MATH_FMIN(SPARK_MATH_FMAX(value / scale,-6.0f),6.0f)) * scale);
}

SPARK_MATH_FN float SparkMathClamp(float value, float limit)
{
	return(SPARK_MATH_FMIN(SPARK_MATH_FMAX(value,-limit),limit));
}

SPARK_MATH_FN float SparkMathPolyNorm(float gate, float d1, float d2, float d3, float w0, float w1, float w2, float bias)
{
	return(w0 * (gate * gate * gate / d3) + w1 * (gate * gate / d2) + w2 * (gate / d1) + bias);
}

SPARK_MATH_FN float SparkMathDifferentialGate(float signal, float noise, float lambda, float gate)
{
	const float scaled = SparkMathRoundBf16(SparkMathRoundBf16(SparkMathSigmoid(lambda)) * noise);
	return(SparkMathRoundBf16(signal - scaled) * SparkMathRoundBf16(SparkMathSigmoid(gate)));
}

#endif
