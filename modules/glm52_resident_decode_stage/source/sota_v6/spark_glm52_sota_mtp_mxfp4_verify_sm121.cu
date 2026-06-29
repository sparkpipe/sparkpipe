#include "spark_glm52_sota_decode_common.cuh"

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaMtpMxfp4DraftWarpRowsKernel(
    SparkGlm52SotaMtpArguments arguments)
{
    uint32_t token_index = blockIdx.y;
    uint32_t candidate_base = blockIdx.x * SPARK_GLM52_SOTA_ROUTER_EXPERTS_PER_CTA;
    uint32_t warp_index = threadIdx.x >> 5u;
    uint32_t lane_index = threadIdx.x & 31u;
    uint32_t candidate_offset = candidate_base + warp_index;
    float accumulator = 0.0f;

    if (token_index >= arguments.active_token_count || candidate_offset >= arguments.candidate_token_count)
    {
        return;
    }

    uint32_t token_id = arguments.candidate_token_ids[candidate_offset];
    for (uint32_t column = lane_index; column < arguments.hidden_dimension; column += SPARK_GLM52_SOTA_WARP_LANES)
    {
        float hidden = SparkGlm52SotaBf16ToFloat(
            arguments.hidden_bf16[((uint64_t)token_index * arguments.hidden_dimension) + column]);
        float weight = SparkGlm52SotaLoadMxfp4(
            arguments.draft_weight_mxfp4.payload_u8,
            arguments.draft_weight_mxfp4.scale_e8m0_u8,
            arguments.draft_weight_mxfp4.packed_row_stride_bytes,
            arguments.draft_weight_mxfp4.scale_row_stride_bytes,
            token_id,
            column);

        accumulator = fmaf(hidden, weight, accumulator);
    }
    accumulator = SparkGlm52SotaWarpReduceSum(accumulator);
    if (lane_index == 0u)
    {
        arguments.draft_logits_f32[((uint64_t)token_index * arguments.candidate_token_count) + candidate_offset] = accumulator;
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaMtpArgmaxVerifyKernel(
    SparkGlm52SotaMtpArguments arguments)
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
    if (lane_index < arguments.candidate_token_count)
    {
        score = arguments.draft_logits_f32[((uint64_t)token_index * arguments.candidate_token_count) + lane_index];
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
        uint32_t predicted_token_id = arguments.candidate_token_ids[best_offset];
        uint32_t verify_token_id = arguments.verify_token_ids != 0 ? arguments.verify_token_ids[token_index] : predicted_token_id;

        arguments.draft_token_ids[token_index] = predicted_token_id;
        if (predicted_token_id == verify_token_id)
        {
            atomicAdd(arguments.accepted_count, 1u);
        }
        else
        {
            atomicAdd(arguments.rejected_count, 1u);
            atomicAdd(arguments.rollback_count, 1u);
        }
    }
}

extern "C" cudaError_t SparkGlm52SotaMtpMxfp4VerifySm121(
    const SparkGlm52SotaMtpArguments *arguments,
    cudaStream_t stream)
{
    dim3 grid;
    cudaError_t error;

    if (arguments == 0 || arguments->hidden_bf16 == 0 || arguments->candidate_token_ids == 0 ||
        arguments->draft_logits_f32 == 0 || arguments->draft_token_ids == 0 ||
        arguments->accepted_count == 0 || arguments->rejected_count == 0 || arguments->rollback_count == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (arguments->candidate_token_count == 0u || arguments->candidate_token_count > 256u)
    {
        return cudaErrorInvalidValue;
    }
    if (arguments->mtp_launch != 0)
    {
        return arguments->mtp_launch(arguments->mtp_plan_context, stream);
    }
    if (arguments->require_plan != 0u || arguments->draft_weight_mxfp4.payload_u8 == 0 ||
        arguments->draft_weight_mxfp4.scale_e8m0_u8 == 0)
    {
        return cudaErrorInvalidValue;
    }

    error = cudaMemsetAsync(arguments->accepted_count, 0, sizeof(uint32_t), stream);
    if (error != cudaSuccess)
    {
        return error;
    }
    error = cudaMemsetAsync(arguments->rejected_count, 0, sizeof(uint32_t), stream);
    if (error != cudaSuccess)
    {
        return error;
    }
    error = cudaMemsetAsync(arguments->rollback_count, 0, sizeof(uint32_t), stream);
    if (error != cudaSuccess)
    {
        return error;
    }

    grid.x = SparkGlm52SotaCeilDivU32(arguments->candidate_token_count, SPARK_GLM52_SOTA_ROUTER_EXPERTS_PER_CTA);
    grid.y = arguments->active_token_count;
    grid.z = 1u;
    SparkGlm52SotaMtpMxfp4DraftWarpRowsKernel<<<grid, 256u, 0u, stream>>>(*arguments);
    SparkGlm52SotaMtpArgmaxVerifyKernel<<<arguments->active_token_count, 256u, 0u, stream>>>(*arguments);
    return cudaGetLastError();
}
