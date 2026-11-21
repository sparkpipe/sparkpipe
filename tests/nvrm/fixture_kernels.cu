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

extern "C" __global__ void __launch_bounds__(256) SparkNvrmStreamRead(const uint4 *source, uint64_t count, uint32_t *sink)
{
	uint4 total = make_uint4(0u, 0u, 0u, 0u);
	uint64_t index;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		const uint4 value = source[index];
		total.x ^= value.x;
		total.y ^= value.y;
		total.z ^= value.z;
		total.w ^= value.w;
	}
	if ( (total.x ^ total.y ^ total.z ^ total.w) == SPARK_NVRM_FIXTURE_SINK_MATCH )
		sink[0] = blockIdx.x;
}

extern "C" __global__ void __launch_bounds__(256) SparkNvrmStreamCopy(const uint4 *source, uint4 *destination, uint64_t count)
{
	uint64_t index;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
		destination[index] = source[index];
}

extern "C" __global__ void __launch_bounds__(256) SparkNvrmStreamFill(uint4 *destination, uint64_t count, uint32_t value)
{
	uint64_t index;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
		destination[index] = make_uint4(value, value ^ 1u, value ^ 2u, value ^ (uint32_t)index);
}
