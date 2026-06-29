#include "spark_glm52_sota_decode_common.cuh"

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaRestrictedBf16LogitsWarpRowsKernel(
    SparkGlm52SotaRestrictedLogitsArguments arguments)
{
    uint32_t token_index = blockIdx.y;
    uint32_t candidate_base = blockIdx.x * SPARK_GLM52_SOTA_ROUTER_EXPERTS_PER_CTA;
    uint32_t warp_index = threadIdx.x >> 5u;
    uint32_t lane_index = threadIdx.x & 31u;
    uint32_t candidate_offset = candidate_base + warp_index;
    float accumulator = 0.0f;

    if (token_index >= arguments.active_token_count || candidate_offset >= arguments.restricted_token_count)
    {
        return;
    }

    uint32_t token_id = arguments.token_ids[candidate_offset];
    for (uint32_t column = lane_index * 2u; column + 1u < arguments.hidden_dimension; column += 64u)
    {
        __nv_bfloat162 hidden_pair = *reinterpret_cast<const __nv_bfloat162 *>(
            &arguments.hidden_bf16[((uint64_t)token_index * arguments.hidden_dimension) + column]);
        __nv_bfloat162 weight_pair = *reinterpret_cast<const __nv_bfloat162 *>(
            &arguments.lm_head_bf16[((uint64_t)token_id * arguments.vocab_stride_elements) + column]);
        float2 hidden_f32 = __bfloat1622float2(hidden_pair);
        float2 weight_f32 = __bfloat1622float2(weight_pair);

        accumulator = fmaf(hidden_f32.x, weight_f32.x, accumulator);
        accumulator = fmaf(hidden_f32.y, weight_f32.y, accumulator);
    }
    accumulator = SparkGlm52SotaWarpReduceSum(accumulator);
    if (lane_index == 0u)
    {
        arguments.logits_f32[((uint64_t)token_index * arguments.restricted_token_count) + candidate_offset] = accumulator;
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaRestrictedArgmaxKernel(
    SparkGlm52SotaRestrictedLogitsArguments arguments)
{
    __shared__ float shared_scores[256];
    __shared__ uint32_t shared_offsets[256];
    uint32_t token_index = blockIdx.x;
    uint32_t lane_index = threadIdx.x;
    float score = -CUDART_INF_F;
    uint32_t offset = lane_index;

    if (token_index >= arguments.active_token_count)
    {
        return;
    }
    if (lane_index < arguments.restricted_token_count)
    {
        score = arguments.logits_f32[((uint64_t)token_index * arguments.restricted_token_count) + lane_index];
    }
    shared_scores[lane_index] = score;
    shared_offsets[lane_index] = offset;
    __syncthreads();

    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (lane_index < stride)
        {
            float other_score = shared_scores[lane_index + stride];
            uint32_t other_offset = shared_offsets[lane_index + stride];

            if (other_score > shared_scores[lane_index])
            {
                shared_scores[lane_index] = other_score;
                shared_offsets[lane_index] = other_offset;
            }
        }
        __syncthreads();
    }

    if (lane_index == 0u)
    {
        uint32_t best_offset = shared_offsets[0];

        arguments.selected_token_offsets[token_index] = best_offset;
        arguments.selected_token_ids[token_index] = arguments.token_ids[best_offset];
    }
}

extern "C" cudaError_t SparkGlm52SotaRestrictedLogitsSamplerSm121(
    const SparkGlm52SotaRestrictedLogitsArguments *arguments,
    cudaStream_t stream)
{
    dim3 grid;
    cudaError_t error;

    if (arguments == 0 || arguments->hidden_bf16 == 0 || arguments->token_ids == 0 ||
        arguments->logits_f32 == 0 || arguments->selected_token_ids == 0 || arguments->selected_token_offsets == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (arguments->restricted_token_count == 0u || arguments->restricted_token_count > 256u)
    {
        return cudaErrorInvalidValue;
    }

    if (arguments->restricted_projection_plan.launch != 0)
    {
        error = arguments->restricted_projection_plan.launch(arguments->restricted_projection_plan.plan_context, stream);
        if (error != cudaSuccess)
        {
            return error;
        }
    }
    else
    {
        if (arguments->require_plan != 0u || arguments->lm_head_bf16 == 0)
        {
            return cudaErrorInvalidValue;
        }
        grid.x = SparkGlm52SotaCeilDivU32(arguments->restricted_token_count, SPARK_GLM52_SOTA_ROUTER_EXPERTS_PER_CTA);
        grid.y = arguments->active_token_count;
        grid.z = 1u;
        SparkGlm52SotaRestrictedBf16LogitsWarpRowsKernel<<<grid, 256u, 0u, stream>>>(*arguments);
    }

    SparkGlm52SotaRestrictedArgmaxKernel<<<arguments->active_token_count, 256u, 0u, stream>>>(*arguments);
    return cudaGetLastError();
}
