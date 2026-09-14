#include <cuda_runtime.h>

#include "sparkpipe/spark_hy4_resident_decode_stage_firmware.h"
#include "runtime/launch.h"

#define SPARK_HY4_CUDA_HIDDEN SPARK_HY4_MODEL_HIDDEN_DIMENSION
#define SPARK_HY4_CUDA_HC SPARK_HY4_MODEL_HC_STREAM_COUNT
#define SPARK_HY4_CUDA_HC_FLAT (SPARK_HY4_CUDA_HC * SPARK_HY4_CUDA_HIDDEN)
#define SPARK_HY4_CUDA_HC_ROWS SPARK_HY4_MODEL_HC_FN_OUTPUT_ROWS
#define SPARK_HY4_CUDA_QUERY_LORA SPARK_HY4_MODEL_QUERY_LORA_RANK
#define SPARK_HY4_CUDA_KV_LORA SPARK_HY4_MODEL_KV_LORA_RANK
#define SPARK_HY4_CUDA_NOPE SPARK_HY4_MODEL_QK_NOPE_HEAD_DIMENSION
#define SPARK_HY4_CUDA_ROPE SPARK_HY4_MODEL_QK_ROPE_HEAD_DIMENSION
#define SPARK_HY4_CUDA_QK SPARK_HY4_MODEL_QK_HEAD_DIMENSION
#define SPARK_HY4_CUDA_V_HEAD SPARK_HY4_MODEL_V_HEAD_DIMENSION
#define SPARK_HY4_CUDA_LOCAL_HEADS SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK
#define SPARK_HY4_CUDA_LOCAL_EXPERTS SPARK_HY4_MODEL_EXPERTS_PER_RANK
#define SPARK_HY4_CUDA_EXPERT_INTER SPARK_HY4_MODEL_EXPERT_INTERMEDIATE_DIMENSION
#define SPARK_HY4_CUDA_SCALE_GROUP SPARK_HY4_MODEL_EXPERT_SCALE_GROUP_SIZE
#define SPARK_HY4_CUDA_ROUTE_MAX \
	(SPARK_HY4_MODEL_EXPERTS_PER_TOKEN * SPARK_HY4_MODEL_HC_STREAM_COUNT)

static __device__ __forceinline__ float SparkHy4E8m0ToFloat(uint8_t raw)
{
	return exp2f((float)raw - 127.0f);
}

static __device__ __forceinline__ float SparkHy4Fp8ByteToFloat(uint32_t raw)
{
	uint32_t sign = (raw & 0x80u) << 24;
	uint32_t exponent = (raw >> 3) & 0x0fu;
	uint32_t mantissa = raw & 0x07u;
	if ( exponent == 0x0fu && mantissa == 0x07u )
		return NAN;
	if ( exponent == 0u )
		return __uint_as_float(sign |
		    __float_as_uint((float)mantissa * 1.953125e-3f));
	return __uint_as_float(sign | ((exponent + 120u) << 23) |
	    (mantissa << 20));
}

__global__ void SparkHy4GemvFp8GroupedKernel(const uint8_t *weights,
	const uint8_t *scales, const float *x, float *y, int rows,
	int columns, int scale_stride)
{
	int row = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
	int lane = (int)(threadIdx.x & 31u);
	if ( row >= rows )
		return;
	const uint8_t *row_weights = weights + (size_t)row * (size_t)columns;
	const uint8_t *row_scales = scales + (size_t)row * (size_t)scale_stride;
	float accumulator = 0.0f;
	int vectorized = columns & ~127;
	for (int base = lane * 16; base < vectorized; base += 512)
	{
		uint4 codes = *(const uint4 *)(row_weights + base);
		float scale = SparkHy4E8m0ToFloat(row_scales[base >> 5]);
		accumulator += scale * (
		    SparkHy4Fp8ByteToFloat(codes.x & 0xffu) * x[base] +
		    SparkHy4Fp8ByteToFloat((codes.x >> 8) & 0xffu) * x[base + 1] +
		    SparkHy4Fp8ByteToFloat((codes.x >> 16) & 0xffu) * x[base + 2] +
		    SparkHy4Fp8ByteToFloat((codes.x >> 24) & 0xffu) * x[base + 3] +
		    SparkHy4Fp8ByteToFloat(codes.y & 0xffu) * x[base + 4] +
		    SparkHy4Fp8ByteToFloat((codes.y >> 8) & 0xffu) * x[base + 5] +
		    SparkHy4Fp8ByteToFloat((codes.y >> 16) & 0xffu) * x[base + 6] +
		    SparkHy4Fp8ByteToFloat((codes.y >> 24) & 0xffu) * x[base + 7] +
		    SparkHy4Fp8ByteToFloat(codes.z & 0xffu) * x[base + 8] +
		    SparkHy4Fp8ByteToFloat((codes.z >> 8) & 0xffu) * x[base + 9] +
		    SparkHy4Fp8ByteToFloat((codes.z >> 16) & 0xffu) * x[base + 10] +
		    SparkHy4Fp8ByteToFloat((codes.z >> 24) & 0xffu) * x[base + 11] +
		    SparkHy4Fp8ByteToFloat(codes.w & 0xffu) * x[base + 12] +
		    SparkHy4Fp8ByteToFloat((codes.w >> 8) & 0xffu) * x[base + 13] +
		    SparkHy4Fp8ByteToFloat((codes.w >> 16) & 0xffu) * x[base + 14] +
		    SparkHy4Fp8ByteToFloat((codes.w >> 24) & 0xffu) * x[base + 15]);
	}
	float remainder_total = 0.0f;
	for (int index = vectorized + lane; index < columns; index += 32)
	{
		int group = index >> 5;
		remainder_total += SparkHy4E8m0ToFloat(row_scales[group]) *
		    SparkHy4Fp8ByteToFloat(row_weights[index]) * x[index];
	}
	accumulator += remainder_total;
	for (unsigned offset = 16u; offset != 0u; offset >>= 1)
		accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
	if ( lane == 0 )
		y[row] = accumulator;
}

__global__ void SparkHy4GemvKernel(const float *weights, const float *x,
	float *y, int rows, int columns)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows)
		return;
	const float *row_pointer = weights + (size_t)row * (size_t)columns;
	float accumulator = 0.0f;
	for (int index = 0; index < columns; ++index)
		accumulator = fmaf(row_pointer[index], x[index], accumulator);
	y[row] = accumulator;
}

__global__ void SparkHy4RmsSquaredKernel(const float *x, float *sum_out,
	int n)
{
	__shared__ float partial[256];
	int tid = threadIdx.x;
	float sum = 0.0f;
	for (int k = tid; k < n; k += blockDim.x)
		sum = fmaf(x[k], x[k], sum);
	partial[tid] = sum;
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		__syncthreads();
		if (tid < offset)
			partial[tid] += partial[tid + offset];
	}
	if (tid == 0)
		sum_out[0] = partial[0];
}

__global__ void SparkHy4RmsScaleKernel(const float *x, const float *weight,
	float *y, int n, float epsilon, const float *sum_in)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float inverse = rsqrtf(sum_in[0] / (float)n + epsilon);
	if (index >= n)
		return;
	y[index] = x[index] * inverse * (weight != 0 ? weight[index] : 1.0f);
}

__global__ void SparkHy4RopeKernel(float *v, int rot, float position)
{
	int pair = threadIdx.x;
	if (pair * 2 + 1 >= rot)
		return;
	float angle = position * powf(SPARK_HY4_MODEL_ROPE_THETA,
	    -(float)(2 * pair) / (float)rot);
	float cos_value = cosf(angle);
	float sin_value = sinf(angle);
	float a = v[pair * 2];
	float b = v[pair * 2 + 1];
	v[pair * 2] = a * cos_value - b * sin_value;
	v[pair * 2 + 1] = a * sin_value + b * cos_value;
}

__global__ void SparkHy4HyperGatesKernel(const float *mixes,
	const float *scale, const float *base, float epsilon,
	float magnitude, float *pre, float *post)
{
	int index = threadIdx.x;
	if (index < SPARK_HY4_CUDA_HC)
		pre[index] = 1.0f / (1.0f +
		    expf(-(mixes[index] * scale[0] + base[index]))) + epsilon;
	else
	{
		int j = index - SPARK_HY4_CUDA_HC;
		post[j] = magnitude / (1.0f +
		    expf(-(mixes[SPARK_HY4_CUDA_HC + j] * scale[1] +
		    base[SPARK_HY4_CUDA_HC + j]))) + epsilon;
	}
}

__global__ void SparkHy4HyperReduceKernel(const float *streams,
	const float *pre, float *out, int hidden, int hc)
{
	int element = blockIdx.x * blockDim.x + threadIdx.x;
	float accumulator = 0.0f;
	if (element >= hidden)
		return;
	for (int s = 0; s < hc; ++s)
		accumulator += streams[(size_t)s * (size_t)hidden + element] *
		    pre[s];
	out[element] = accumulator;
}

__global__ void SparkHy4HyperDistributeKernel(float *streams,
	const float *branch, const float *post, int hidden, int hc)
{
	int element = blockIdx.x * blockDim.x + threadIdx.x;
	if (element >= hidden)
		return;
	for (int s = 0; s < hc; ++s)
		streams[(size_t)s * (size_t)hidden + element] +=
		    branch[element] * post[s];
}

__global__ void SparkHy4SwigluClampedKernel(const float *gate,
	const float *up, float *out, int n, float limit)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	float g = gate[index];
	float u = up[index];
	if (g > limit)
		g = limit;
	if (u > limit)
		u = limit;
	if (u < -limit)
		u = -limit;
	out[index] = g / (1.0f + expf(-g)) * u;
}

__global__ void SparkHy4SigmoidKernel(float *v, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n)
		v[index] = 1.0f / (1.0f + expf(-v[index]));
}

__global__ void SparkHy4AxpyKernel(float *acc, const float *h, float w,
	int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n)
		acc[index] += h[index] * w;
}

__global__ void SparkHy4AttnHeadKernel(const float *q_absorbed,
	const float *q_pe, const float *k_latent, const float *k_pe,
	float sink, int context, float scale, float *value_latent)
{
	float scores[SPARK_HY4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS];
	float maximum = sink;
	for (int t = 0; t < context; ++t)
	{
		float s = 0.0f;
		for (int i = 0; i < SPARK_HY4_CUDA_KV_LORA; ++i)
			s = fmaf(q_absorbed[i],
			    k_latent[(size_t)t * SPARK_HY4_CUDA_KV_LORA + i],
			    s);
		for (int i = 0; i < SPARK_HY4_CUDA_ROPE; ++i)
			s = fmaf(q_pe[i],
			    k_pe[(size_t)t * SPARK_HY4_CUDA_ROPE + i], s);
		scores[t] = s * scale;
		if (scores[t] > maximum)
			maximum = scores[t];
	}
	float denominator = expf(sink - maximum);
	for (int t = 0; t < context; ++t)
		denominator += expf(scores[t] - maximum);
	for (int i = 0; i < SPARK_HY4_CUDA_KV_LORA; ++i)
	{
		float accumulator = 0.0f;
		for (int t = 0; t < context; ++t)
			accumulator += expf(scores[t] - maximum) /
			    denominator *
			    k_latent[(size_t)t * SPARK_HY4_CUDA_KV_LORA + i];
		value_latent[i] = accumulator;
	}
}

__global__ void SparkHy4RouteKernel(const float *router_logits,
	const float *router_bias, uint32_t *selected, float *weights,
	int routed_experts, int experts_per_token)
{
	extern __shared__ float keys[];
	float *probabilities = keys + routed_experts;
	int index = threadIdx.x;
	if (index < routed_experts)
	{
		float probability =
		    1.0f / (1.0f + expf(-router_logits[index]));
		probabilities[index] = probability;
		keys[index] = probability + router_bias[index];
	}
	__syncthreads();
	if (index != 0)
		return;
	float weight_sum = 0.0f;
	for (int k = 0; k < experts_per_token; ++k)
	{
		int best = -1;
		float best_value = -3.4e38f;
		for (int e = 0; e < routed_experts; ++e)
		{
			if (keys[e] > best_value)
			{
				best_value = keys[e];
				best = e;
			}
		}
		keys[best] = -3.4e38f;
		selected[k] = (uint32_t)best;
		weights[k] = probabilities[best];
		weight_sum += probabilities[best];
	}
	if (weight_sum < 6.1e-5f)
		weight_sum = 6.1e-5f;
	for (int k = 0; k < experts_per_token; ++k)
		weights[k] = weights[k] / weight_sum *
		    SPARK_HY4_MODEL_ROUTED_SCALING_FACTOR;
}
