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

extern "C" __global__ void SparkNvrmHugeParameter(uint32_t *output, SparkNvrmFixtureHuge block)
{
	uint32_t index;
	for (index = blockIdx.x * blockDim.x + threadIdx.x; index < SPARK_NVRM_FIXTURE_HUGE_WORDS; index += gridDim.x * blockDim.x)
		output[index] = block.words[index];
}

namespace spark_nvrm_fixture_long_names
{
struct SparkNvrmFixtureValue { uint32_t value; };
struct SparkNvrmFixtureTagAlpha {};
struct SparkNvrmFixtureTagBravo {};

template<typename Value, typename First, typename Second>
__global__ void SparkNvrmLongTemplate(uint32_t *output, Value value)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	output[index] = value.value + index;
}

template __global__ void SparkNvrmLongTemplate<SparkNvrmFixtureValue, SparkNvrmFixtureTagAlpha, SparkNvrmFixtureTagBravo>(uint32_t *output,
	SparkNvrmFixtureValue value);
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

static __device__ uint32_t spark_nvrm_fixture_word;
__constant__ uint32_t spark_nvrm_fixture_table[SPARK_NVRM_FIXTURE_TABLE_WORDS] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};

extern "C" __global__ void SparkNvrmGlobalSwap(uint32_t *output, uint32_t value)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
	{
		output[0] = spark_nvrm_fixture_word;
		spark_nvrm_fixture_word = value;
	}
}

extern "C" __global__ void SparkNvrmConstantTable(uint32_t *output)
{
	output[threadIdx.x] = spark_nvrm_fixture_table[threadIdx.x % SPARK_NVRM_FIXTURE_TABLE_WORDS];
}

__device__ __noinline__ uint32_t SparkNvrmFixtureDeep(uint32_t pick, uint32_t lane)
{
	volatile uint32_t scratch[SPARK_NVRM_FIXTURE_DEEP_WORDS];
	uint32_t index,total = 0u;
	for (index = 0u; index < SPARK_NVRM_FIXTURE_DEEP_WORDS; index++)
		scratch[index] = lane * SPARK_NVRM_FIXTURE_DEEP_STRIDE + index;
	for (index = 0u; index < SPARK_NVRM_FIXTURE_DEEP_WORDS; index++)
		total += scratch[(index * pick) % SPARK_NVRM_FIXTURE_DEEP_WORDS];
	return(total);
}

typedef uint32_t (*SparkNvrmFixtureDeepFunction)(uint32_t pick, uint32_t lane);
__device__ SparkNvrmFixtureDeepFunction spark_nvrm_fixture_deep = SparkNvrmFixtureDeep;

extern "C" __global__ void SparkNvrmCallStack(uint32_t *output, uint32_t pick)
{
	const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
	output[lane] = spark_nvrm_fixture_deep(pick,lane);
}

extern "C" __global__ void SparkNvrmDeepFrame(uint32_t *output, uint32_t pick)
{
	volatile uint32_t scratch[SPARK_NVRM_FIXTURE_FRAME_WORDS];
	const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t index,total = 0u;
	for (index = 0u; index < SPARK_NVRM_FIXTURE_FRAME_WORDS; index++)
		scratch[index] = lane * SPARK_NVRM_FIXTURE_DEEP_STRIDE + index;
	for (index = 0u; index < SPARK_NVRM_FIXTURE_FRAME_WORDS; index++)
		total += scratch[(index * pick) % SPARK_NVRM_FIXTURE_FRAME_WORDS];
	output[lane] = total;
}

extern "C" __global__ void SparkNvrmNoParameters(void)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		spark_nvrm_fixture_word ^= 1u;
}

extern "C" __global__ void SparkNvrmProbeDimensions(uint32_t *output)
{
	if ( threadIdx.x == 0u && threadIdx.y == 0u && threadIdx.z == 0u && blockIdx.x == 0u && blockIdx.y == 0u && blockIdx.z == 0u )
	{
		output[0] = blockDim.x;
		output[1] = blockDim.y;
		output[2] = blockDim.z;
		output[3] = gridDim.x;
		output[4] = gridDim.y;
		output[5] = gridDim.z;
	}
}
