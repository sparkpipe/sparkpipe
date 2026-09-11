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

static __device__ __forceinline__ float SparkHy4Fp8ToFloat(uint8_t raw)
{
	float sign = (raw & 0x80u) != 0u ? -1.0f : 1.0f;
	uint32_t exponent = ((uint32_t)raw >> 3) & 0x0fu;
	uint32_t mantissa = (uint32_t)raw & 0x07u;
	if ( exponent == 0u )
		return sign * ldexpf((float)mantissa,-9);
	if ( exponent == 0x0fu && mantissa == 0x07u )
		return NAN;
	return sign * ldexpf(1.0f + (float)mantissa * 0.125f,
	    (int)exponent - 7);
}

static __device__ __forceinline__ float SparkHy4E8m0ToFloat(uint8_t raw)
{
	return exp2f((float)raw - 127.0f);
}

static __device__ __forceinline__ float SparkHy4DotFp8Grouped(
	const uint8_t *payload, const uint8_t *scales, const float *query,
	int columns)
{
	float total = 0.0f;
	for (int group = 0; group * SPARK_HY4_CUDA_SCALE_GROUP < columns;
	    ++group)
	{
		float group_partial = 0.0f;
		int base = group * SPARK_HY4_CUDA_SCALE_GROUP;
		int end = base + SPARK_HY4_CUDA_SCALE_GROUP;
		if (end > columns)
			end = columns;
		for (int index = base; index < end; ++index)
			group_partial = fmaf(
			    SparkHy4Fp8ToFloat(payload[index]), query[index],
			    group_partial);
		total += group_partial *
		    SparkHy4E8m0ToFloat(scales[group]);
	}
	return total;
}

static __device__ __forceinline__ float2 SparkHy4LoadBf16Pair(
    const void *base, uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float(
	    (int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return pair;
}

static __device__ __forceinline__ void SparkHy4StoreBf16Pair(
    void *base, uint64_t element, float x, float y)
{
	uint32_t packed = (__float_as_uint(y) & UINT32_C(0xffff0000)) |
	    (__float_as_uint(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

static __global__ void SparkHy4AccumAddBf16Kernel(void *destination_bf16,
    const void *source_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair;
	float2 source_pair;
	uint32_t element;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u);
	    element += blockDim.x)
	{
		destination_pair =
		    SparkHy4LoadBf16Pair(destination_bf16, offset + element);
		source_pair =
		    SparkHy4LoadBf16Pair(source_bf16, offset + element);
		SparkHy4StoreBf16Pair(destination_bf16, offset + element,
		    destination_pair.x + source_pair.x,
		    destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkHy4AccumU64MaxKernel(uint64_t *destination,
    const uint64_t *source, uint32_t element_count)
{
	uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkHy4LaunchAccumAddBf16(cudaStream_t stream,
    void *destination_bf16, const void *source_bf16, uint32_t row_count,
    uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u ||
	    width == 0u || (width & 1u) != 0u )
		return cudaErrorInvalidValue;
	SparkHy4AccumAddBf16Kernel<<<row_count, 256u, 0u, stream>>>(
	    destination_bf16, source_bf16, row_count, width);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkHy4LaunchAccumU64Max(cudaStream_t stream,
    uint64_t *destination, const uint64_t *source, uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return cudaErrorInvalidValue;
	SparkHy4AccumU64MaxKernel<<<(element_count + 255u) / 256u, 256u, 0u,
	    stream>>>(destination, source, element_count);
	return cudaPeekAtLastError();
}
