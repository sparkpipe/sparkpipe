#include "spark_glm52_sota_decode_common.cuh"

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaDenseMlpEpilogueKernel(
    SparkGlm52SotaDenseMlpArguments arguments)
{
    uint32_t index = threadIdx.x + (blockIdx.x * blockDim.x);
    uint32_t value_count = arguments.active_token_count * arguments.intermediate_dimension;

    while (index < value_count)
    {
        float gate = SparkGlm52SotaBf16ToFloat(arguments.gate_bf16[index]);
        float up = SparkGlm52SotaBf16ToFloat(arguments.up_bf16[index]);
        float value = SparkGlm52SotaSilu(gate) * up;

        arguments.activation_bf16[index] = SparkGlm52SotaFloatToBf16(value);
        index += gridDim.x * blockDim.x;
    }
}

extern "C" cudaError_t SparkGlm52SotaDenseMlpEpilogueSm121(
    const SparkGlm52SotaDenseMlpArguments *arguments,
    cudaStream_t stream)
{
    uint32_t value_count;

    if (arguments == 0 || arguments->gate_bf16 == 0 || arguments->up_bf16 == 0 || arguments->activation_bf16 == 0 ||
        arguments->active_token_count == 0u || arguments->intermediate_dimension == 0u)
    {
        return cudaErrorInvalidValue;
    }
    value_count = arguments->active_token_count * arguments->intermediate_dimension;
    SparkGlm52SotaDenseMlpEpilogueKernel<<<SparkGlm52SotaCeilDivU32(value_count, 256u), 256u, 0u, stream>>>(*arguments);
    return cudaGetLastError();
}
