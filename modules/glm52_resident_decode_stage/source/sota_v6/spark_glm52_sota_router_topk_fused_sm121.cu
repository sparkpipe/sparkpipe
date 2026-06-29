#include "spark_glm52_sota_production_plan_sm121.cuh"

struct SparkGlm52SotaRouterTop8PlanSm121
{
    uint32_t abi_version;
    uint32_t active_token_count;
    uint32_t normalize_topk_prob;
    uint64_t capability_flags;
    float routed_scaling_factor;
    SparkGlm52SotaBf16LinearPlanSm121 *router_projection;
    const float *bias_f32;
    float *router_logits_f32;
    uint32_t *topk_expert_ids;
    float *topk_weights;
};

static __device__ __forceinline__ void SparkGlm52SotaTop8Insert(
    float candidate_value,
    uint32_t candidate_index,
    float (&top_values)[SPARK_GLM52_SOTA_TOP_K],
    uint32_t (&top_indices)[SPARK_GLM52_SOTA_TOP_K])
{
    #pragma unroll
    for (uint32_t index = 0u; index < SPARK_GLM52_SOTA_TOP_K; ++index)
    {
        if (candidate_value > top_values[index])
        {
            float old_value = top_values[index];
            uint32_t old_index = top_indices[index];
            top_values[index] = candidate_value;
            top_indices[index] = candidate_index;
            candidate_value = old_value;
            candidate_index = old_index;
        }
    }
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52SotaFusedBiasSigmoidTop8Kernel(SparkGlm52SotaRouterTop8PlanSm121 plan)
{
    __shared__ float shared_values[SPARK_GLM52_SOTA_EXPERT_COUNT];
    __shared__ uint32_t shared_indices[SPARK_GLM52_SOTA_EXPERT_COUNT];
    uint32_t token_index = blockIdx.x;
    uint32_t lane = threadIdx.x;
    float top_values[SPARK_GLM52_SOTA_TOP_K];
    uint32_t top_indices[SPARK_GLM52_SOTA_TOP_K];
    float sum = 0.0f;

    #pragma unroll
    for (uint32_t index = 0u; index < SPARK_GLM52_SOTA_TOP_K; ++index)
    {
        top_values[index] = -CUDART_INF_F;
        top_indices[index] = SPARK_GLM52_SOTA_INVALID_U32;
    }

    if (token_index >= plan.active_token_count)
    {
        return;
    }

    if (lane < SPARK_GLM52_SOTA_EXPERT_COUNT)
    {
        float value = plan.router_logits_f32[((uint64_t)token_index * SPARK_GLM52_SOTA_EXPERT_COUNT) + lane];
        float bias = plan.bias_f32 != 0 ? plan.bias_f32[lane] : 0.0f;
        value = 1.0f / (1.0f + __expf(-(value + bias)));
        shared_values[lane] = value;
        shared_indices[lane] = lane;
    }
    __syncthreads();

    for (uint32_t expert_index = lane; expert_index < SPARK_GLM52_SOTA_EXPERT_COUNT; expert_index += blockDim.x)
    {
        SparkGlm52SotaTop8Insert(shared_values[expert_index], shared_indices[expert_index], top_values, top_indices);
    }

    #pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
    {
        #pragma unroll
        for (uint32_t rank = 0u; rank < SPARK_GLM52_SOTA_TOP_K; ++rank)
        {
            float other_value = __shfl_down_sync(0xffffffffu, top_values[rank], offset);
            uint32_t other_index = __shfl_down_sync(0xffffffffu, top_indices[rank], offset);
            SparkGlm52SotaTop8Insert(other_value, other_index, top_values, top_indices);
        }
    }

    if ((lane & 31u) == 0u)
    {
        uint32_t warp_index = lane >> 5u;
        #pragma unroll
        for (uint32_t rank = 0u; rank < SPARK_GLM52_SOTA_TOP_K; ++rank)
        {
            shared_values[(warp_index * SPARK_GLM52_SOTA_TOP_K) + rank] = top_values[rank];
            shared_indices[(warp_index * SPARK_GLM52_SOTA_TOP_K) + rank] = top_indices[rank];
        }
    }
    __syncthreads();

    if (lane < SPARK_GLM52_SOTA_TOP_K)
    {
        #pragma unroll
        for (uint32_t rank = 0u; rank < SPARK_GLM52_SOTA_TOP_K; ++rank)
        {
            top_values[rank] = -CUDART_INF_F;
            top_indices[rank] = SPARK_GLM52_SOTA_INVALID_U32;
        }
        #pragma unroll
        for (uint32_t item = 0u; item < 8u * SPARK_GLM52_SOTA_TOP_K; ++item)
        {
            SparkGlm52SotaTop8Insert(shared_values[item], shared_indices[item], top_values, top_indices);
        }
        sum = 0.0f;
        #pragma unroll
        for (uint32_t rank = 0u; rank < SPARK_GLM52_SOTA_TOP_K; ++rank)
        {
            sum += top_values[rank];
        }
        float value = top_values[lane];
        if (plan.normalize_topk_prob != 0u)
        {
            value = value / fmaxf(sum, 1.0e-20f);
        }
        value *= plan.routed_scaling_factor;
        plan.topk_expert_ids[((uint64_t)token_index * SPARK_GLM52_SOTA_TOP_K) + lane] = top_indices[lane];
        plan.topk_weights[((uint64_t)token_index * SPARK_GLM52_SOTA_TOP_K) + lane] = value;
    }
}

extern "C" cudaError_t SparkGlm52SotaLaunchRouterTop8Sm121(SparkGlm52SotaRouterTop8PlanSm121 *plan, cudaStream_t stream)
{
    cudaError_t error;
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI || plan->active_token_count == 0u || plan->active_token_count > 128u || plan->router_projection == 0 || plan->router_logits_f32 == 0 || plan->topk_expert_ids == 0 || plan->topk_weights == 0)
    {
        return cudaErrorInvalidValue;
    }
    if ((plan->capability_flags & SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROUTER_TOPK) == 0u)
    {
        return cudaErrorInvalidValue;
    }
    plan->router_projection->d = plan->router_logits_f32;
    plan->router_projection->m = plan->active_token_count;
    error = SparkGlm52SotaLaunchBf16CublasLtLinearSm121(plan->router_projection, stream);
    if (error != cudaSuccess)
    {
        return error;
    }
    SparkGlm52SotaFusedBiasSigmoidTop8Kernel<<<plan->active_token_count, 256u, 0u, stream>>>(*plan);
    return cudaGetLastError();
}
