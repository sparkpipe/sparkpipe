#include <stdint.h>

#include "inference/kernels/norm.cuh"
#include "tests/nvrm/fixture_kernels.h"

extern "C" __global__ void SparkNvrmFirstLight(uint32_t *output, uint32_t value)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	output[index] = value ^ index;
}

extern "C" __global__ void __launch_bounds__(128) SparkNvrmParameterMix(uint8_t small, const uint64_t *table, uint16_t lanes,
	SparkNvrmFixtureShape shape, float scale, int8_t sign, double *output)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	output[index] = (double)table[index % shape.rows] * scale * shape.weight + small + lanes + sign + shape.lanes + shape.flag;
}

extern "C" __global__ void SparkNvrmStaticShared(float *output, uint32_t count)
{
	__shared__ float tile[SPARK_NVRM_FIXTURE_TILE];
	uint32_t index;
	for (index = threadIdx.x; index < SPARK_NVRM_FIXTURE_TILE; index += blockDim.x)
		tile[index] = (float)(index * count);
	__syncthreads();
	output[blockIdx.x * blockDim.x + threadIdx.x] = tile[(threadIdx.x * 7u) % SPARK_NVRM_FIXTURE_TILE];
}

extern "C" __global__ void SparkNvrmLocalStack(uint32_t *output, uint32_t pick)
{
	volatile uint32_t scratch[SPARK_NVRM_FIXTURE_SCRATCH];
	uint32_t index;
	for (index = 0u; index < SPARK_NVRM_FIXTURE_SCRATCH; index++)
		scratch[(index * pick) % SPARK_NVRM_FIXTURE_SCRATCH] = output[index] + threadIdx.x;
	output[threadIdx.x] = scratch[(pick + threadIdx.x) % SPARK_NVRM_FIXTURE_SCRATCH];
}

extern "C" __global__ void SparkNvrmLargeParameter(SparkNvrmFixtureBlock block, uint32_t *output)
{
	output[threadIdx.x] = block.words[threadIdx.x % SPARK_NVRM_FIXTURE_BLOCK_WORDS];
}

template __global__ void LmBf16RmsNormKernel<256u>(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t dimension,
	uint32_t row_stride, float epsilon);
template __global__ void LmFusedResidualRmsNormKernel<256u, uint16_t, false>(const uint16_t *__restrict__ input_bf16,
	const uint16_t *__restrict__ residual_bf16, const uint16_t *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16,
	uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon);
