#include "spark_glm52_sota_decode_common.cuh"

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaRmsNormKernel(
    SparkGlm52SotaNormArguments arguments,
    uint32_t add_residual)
{
    __shared__ float shared_inv_rms;
    uint32_t token_index = blockIdx.x;
    uint32_t lane_index = threadIdx.x;
    uint32_t hidden_dimension = arguments.hidden_dimension;
    float sum_squares = 0.0f;

    if (token_index >= arguments.active_token_count)
    {
        return;
    }

    for (uint32_t column = lane_index * 2u; column + 1u < hidden_dimension; column += blockDim.x * 2u)
    {
        uint64_t offset = ((uint64_t)token_index * hidden_dimension) + column;
        __nv_bfloat162 input_pair = *reinterpret_cast<const __nv_bfloat162 *>(&arguments.input_bf16[offset]);
        float2 input_f32 = __bfloat1622float2(input_pair);

        if (add_residual != 0u && arguments.residual_bf16 != 0)
        {
            __nv_bfloat162 residual_pair = *reinterpret_cast<const __nv_bfloat162 *>(&arguments.residual_bf16[offset]);
            float2 residual_f32 = __bfloat1622float2(residual_pair);

            input_f32.x += residual_f32.x;
            input_f32.y += residual_f32.y;
        }
        sum_squares = fmaf(input_f32.x, input_f32.x, sum_squares);
        sum_squares = fmaf(input_f32.y, input_f32.y, sum_squares);
    }
    sum_squares = SparkGlm52SotaBlockReduceSum(sum_squares);
    if (lane_index == 0u)
    {
        shared_inv_rms = rsqrtf((sum_squares / (float)hidden_dimension) + arguments.epsilon);
    }
    __syncthreads();

    for (uint32_t column = lane_index * 2u; column + 1u < hidden_dimension; column += blockDim.x * 2u)
    {
        uint64_t offset = ((uint64_t)token_index * hidden_dimension) + column;
        __nv_bfloat162 input_pair = *reinterpret_cast<const __nv_bfloat162 *>(&arguments.input_bf16[offset]);
        __nv_bfloat162 weight_pair = *reinterpret_cast<const __nv_bfloat162 *>(&arguments.weight_bf16[column]);
        float2 input_f32 = __bfloat1622float2(input_pair);
        float2 weight_f32 = __bfloat1622float2(weight_pair);

        if (add_residual != 0u && arguments.residual_bf16 != 0)
        {
            __nv_bfloat162 residual_pair = *reinterpret_cast<const __nv_bfloat162 *>(&arguments.residual_bf16[offset]);
            float2 residual_f32 = __bfloat1622float2(residual_pair);

            input_f32.x += residual_f32.x;
            input_f32.y += residual_f32.y;
            if (arguments.residual_output_bf16 != 0)
            {
                __nv_bfloat162 residual_output_pair;
                residual_output_pair = __floats2bfloat162_rn(input_f32.x, input_f32.y);
                *reinterpret_cast<__nv_bfloat162 *>(&arguments.residual_output_bf16[offset]) = residual_output_pair;
            }
        }
        input_f32.x = input_f32.x * shared_inv_rms * weight_f32.x;
        input_f32.y = input_f32.y * shared_inv_rms * weight_f32.y;
        *reinterpret_cast<__nv_bfloat162 *>(&arguments.output_bf16[offset]) = __floats2bfloat162_rn(input_f32.x, input_f32.y);
    }
}

extern "C" cudaError_t SparkGlm52SotaRmsNormSm121(
    const SparkGlm52SotaNormArguments *arguments,
    cudaStream_t stream)
{
    if (arguments == 0 || arguments->input_bf16 == 0 || arguments->weight_bf16 == 0 || arguments->output_bf16 == 0)
    {
        return cudaErrorInvalidValue;
    }
    SparkGlm52SotaRmsNormKernel<<<arguments->active_token_count, 256u, 0u, stream>>>(*arguments, 0u);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkGlm52SotaResidualAddRmsNormSm121(
    const SparkGlm52SotaNormArguments *arguments,
    cudaStream_t stream)
{
    if (arguments == 0 || arguments->input_bf16 == 0 || arguments->residual_bf16 == 0 ||
        arguments->weight_bf16 == 0 || arguments->output_bf16 == 0)
    {
        return cudaErrorInvalidValue;
    }
    SparkGlm52SotaRmsNormKernel<<<arguments->active_token_count, 256u, 0u, stream>>>(*arguments, 1u);
    return cudaGetLastError();
}
