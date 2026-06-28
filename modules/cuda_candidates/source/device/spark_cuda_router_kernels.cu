#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_router_kernels.h"

#define SPARK_CUDA_ROUTER_THREADS 256u

static __device__ float SparkCudaRouterBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static uint64_t SparkCudaRouterScoreCountHost(const SparkCudaRouterRequest *request)
{
    return (uint64_t)request->row_count * (uint64_t)request->expert_count;
}

static uint64_t SparkCudaRouterTopKValueCountHost(const SparkCudaRouterRequest *request)
{
    return (uint64_t)request->row_count * (uint64_t)request->top_k;
}

static void SparkCudaRouterFillReportShape(const SparkCudaRouterRequest *request, SparkCudaRouterReport *report)
{
    report->score_count = SparkCudaRouterScoreCountHost(request);
    report->topk_value_count = SparkCudaRouterTopKValueCountHost(request);
}

static __device__ float SparkCudaRouterWarpMax(float value)
{
    uint32_t offset;
    float other;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        other = __shfl_down_sync(0xffffffffu, value, offset);
        value = other > value ? other : value;
    }
    return value;
}

static __device__ float SparkCudaRouterWarpSum(float value)
{
    uint32_t offset;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

static __device__ float SparkCudaRouterBlockMax(float value, float *warp_values)
{
    uint32_t lane;
    uint32_t warp;

    lane = threadIdx.x & 31u;
    warp = threadIdx.x >> 5u;
    value = SparkCudaRouterWarpMax(value);
    if (lane == 0u)
    {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8u ? warp_values[threadIdx.x] : -FLT_MAX;
    if (warp == 0u)
    {
        value = SparkCudaRouterWarpMax(value);
    }
    if (threadIdx.x == 0u)
    {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

static __device__ float SparkCudaRouterBlockSum(float value, float *warp_values)
{
    uint32_t lane;
    uint32_t warp;

    lane = threadIdx.x & 31u;
    warp = threadIdx.x >> 5u;
    value = SparkCudaRouterWarpSum(value);
    if (lane == 0u)
    {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8u ? warp_values[threadIdx.x] : 0.0f;
    if (warp == 0u)
    {
        value = SparkCudaRouterWarpSum(value);
    }
    if (threadIdx.x == 0u)
    {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

static __device__ uint32_t SparkCudaRouterGroupSelected(const uint32_t *selected_groups, uint32_t top_k_group, uint32_t group_index)
{
    uint32_t selected_index;

    for (selected_index = 0u; selected_index < top_k_group; ++selected_index)
    {
        if (selected_groups[selected_index] == group_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static __device__ uint32_t SparkCudaRouterExpertSelected(const uint32_t *selected_experts, uint32_t selected_count, uint32_t expert_index)
{
    uint32_t selected_index;

    for (selected_index = 0u; selected_index < selected_count; ++selected_index)
    {
        if (selected_experts[selected_index] == expert_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static __device__ uint32_t SparkCudaRouterBetterChoice(float next_value, uint32_t next_id, float best_value, uint32_t best_id)
{
    if (next_value > best_value)
    {
        return 1u;
    }
    if (next_value == best_value && next_id < best_id)
    {
        return 1u;
    }
    return 0u;
}

static __device__ void SparkCudaRouterWarpBestPair(float *value, uint32_t *id)
{
    uint32_t offset;
    float other_value;
    uint32_t other_id;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        other_value = __shfl_down_sync(0xffffffffu, *value, offset);
        other_id = __shfl_down_sync(0xffffffffu, *id, offset);
        if (SparkCudaRouterBetterChoice(other_value, other_id, *value, *id) != 0u)
        {
            *value = other_value;
            *id = other_id;
        }
    }
}

static __device__ void SparkCudaRouterWarpTop8(float candidate_value, uint32_t candidate_id, float *warp_values, uint32_t *warp_ids)
{
    uint32_t lane;
    uint32_t warp;
    uint32_t selected_index;
    float best_value;
    uint32_t best_id;

    lane = threadIdx.x & 31u;
    warp = threadIdx.x >> 5u;
    for (selected_index = 0u; selected_index < 8u; ++selected_index)
    {
        best_value = candidate_value;
        best_id = candidate_id;
        SparkCudaRouterWarpBestPair(&best_value, &best_id);
        best_value = __shfl_sync(0xffffffffu, best_value, 0);
        best_id = __shfl_sync(0xffffffffu, best_id, 0);
        if (lane == 0u)
        {
            warp_values[(warp * 8u) + selected_index] = best_value;
            warp_ids[(warp * 8u) + selected_index] = best_id;
        }
        if (candidate_id == best_id)
        {
            candidate_value = -FLT_MAX;
            candidate_id = UINT32_MAX;
        }
    }
}

static __device__ void SparkCudaRouterBlockBest(float local_value, uint32_t local_id, float *best_values, uint32_t *best_ids)
{
    uint32_t stride;

    best_values[threadIdx.x] = local_value;
    best_ids[threadIdx.x] = local_id;
    __syncthreads();
    for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            if (SparkCudaRouterBetterChoice(best_values[threadIdx.x + stride], best_ids[threadIdx.x + stride], best_values[threadIdx.x], best_ids[threadIdx.x]) != 0u)
            {
                best_values[threadIdx.x] = best_values[threadIdx.x + stride];
                best_ids[threadIdx.x] = best_ids[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
}

static __device__ uint32_t SparkCudaRouterFastGroupSelected(const uint32_t *selected_groups, uint32_t group_index)
{
    uint32_t selected_index;

    for (selected_index = 0u; selected_index < 4u; ++selected_index)
    {
        if (selected_groups[selected_index] == group_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static __device__ float SparkCudaRouterSigmoid(float value)
{
    return 1.0f / (1.0f + __expf(-value));
}

static __device__ void SparkCudaRouterFastEmitTop8(SparkCudaRouterRequest request, const uint16_t *logits, float *candidate_values, uint32_t *candidate_ids, float *topk_weights, uint32_t *topk_ids, uint32_t *token_expert_indices)
{
    uint32_t selected_index;
    uint32_t candidate_index;
    uint32_t best_candidate;
    uint32_t best_id;
    uint64_t logit_offset;
    uint64_t output_offset;
    float best_value;
    float top_max;
    float weight_sum;
    float value;

    if (threadIdx.x != 0u)
    {
        return;
    }
    top_max = -FLT_MAX;
    weight_sum = 0.0f;
    for (selected_index = 0u; selected_index < 8u; ++selected_index)
    {
        best_candidate = 0u;
        best_value = -FLT_MAX;
        best_id = UINT32_MAX;
        for (candidate_index = 0u; candidate_index < 64u; ++candidate_index)
        {
            if (SparkCudaRouterBetterChoice(candidate_values[candidate_index], candidate_ids[candidate_index], best_value, best_id) != 0u)
            {
                best_candidate = candidate_index;
                best_value = candidate_values[candidate_index];
                best_id = candidate_ids[candidate_index];
            }
        }
        candidate_values[best_candidate] = -FLT_MAX;
        output_offset = ((uint64_t)blockIdx.x * 8u) + (uint64_t)selected_index;
        topk_ids[output_offset] = best_id;
        if (best_value > top_max)
        {
            top_max = best_value;
        }
    }
    for (selected_index = 0u; selected_index < 8u; ++selected_index)
    {
        output_offset = ((uint64_t)blockIdx.x * 8u) + (uint64_t)selected_index;
        best_id = topk_ids[output_offset];
        logit_offset = ((uint64_t)blockIdx.x * 256u) + (uint64_t)best_id;
        value = SparkCudaRouterBf16ToFloat(logits[logit_offset]);
        if (request.score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_SOFTMAX)
        {
            value = __expf(value - top_max);
        }
        else
        {
            value = SparkCudaRouterSigmoid(value);
        }
        topk_weights[output_offset] = value;
        weight_sum += value;
    }
    for (selected_index = 0u; selected_index < 8u; ++selected_index)
    {
        output_offset = ((uint64_t)blockIdx.x * 8u) + (uint64_t)selected_index;
        if (request.renormalize != 0u && weight_sum > 0.0f)
        {
            topk_weights[output_offset] = topk_weights[output_offset] / weight_sum;
        }
        topk_weights[output_offset] = topk_weights[output_offset] * request.routed_scaling_factor;
        token_expert_indices[output_offset] = (selected_index * request.row_count) + blockIdx.x;
    }
}

static __device__ void SparkCudaRouterFastEmitGroupedTop8(SparkCudaRouterRequest request, const float *score_values, float *candidate_values, uint32_t *candidate_ids, float *reduce_values, uint32_t *reduce_ids, float *weight_sum_slot, float *topk_weights, uint32_t *topk_ids, uint32_t *token_expert_indices)
{
    uint32_t selected_index;
    uint32_t best_id;
    uint32_t lane;
    uint32_t warp;
    uint64_t output_offset;
    float best_value;
    float selected_weight;

    lane = threadIdx.x & 31u;
    warp = threadIdx.x >> 5u;
    if (threadIdx.x == 0u)
    {
        weight_sum_slot[0] = 0.0f;
    }
    __syncthreads();
    for (selected_index = 0u; selected_index < 8u; ++selected_index)
    {
        best_value = threadIdx.x < 64u ? candidate_values[threadIdx.x] : -FLT_MAX;
        best_id = threadIdx.x < 64u ? candidate_ids[threadIdx.x] : UINT32_MAX;
        if (threadIdx.x < 64u)
        {
            SparkCudaRouterWarpBestPair(&best_value, &best_id);
        }
        if (lane == 0u && warp < 2u)
        {
            reduce_values[warp] = best_value;
            reduce_ids[warp] = best_id;
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            if (SparkCudaRouterBetterChoice(reduce_values[1u], reduce_ids[1u], reduce_values[0u], reduce_ids[0u]) != 0u)
            {
                reduce_values[0u] = reduce_values[1u];
                reduce_ids[0u] = reduce_ids[1u];
            }
            best_id = reduce_ids[0u];
            selected_weight = score_values[best_id];
            output_offset = ((uint64_t)blockIdx.x * 8u) + (uint64_t)selected_index;
            topk_ids[output_offset] = best_id;
            topk_weights[output_offset] = selected_weight;
            weight_sum_slot[0] += selected_weight;
        }
        __syncthreads();
        if (threadIdx.x < 64u && candidate_ids[threadIdx.x] == reduce_ids[0u])
        {
            candidate_values[threadIdx.x] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x < 8u)
    {
        output_offset = ((uint64_t)blockIdx.x * 8u) + (uint64_t)threadIdx.x;
        if (request.renormalize != 0u && weight_sum_slot[0] > 0.0f)
        {
            topk_weights[output_offset] = topk_weights[output_offset] / weight_sum_slot[0];
        }
        topk_weights[output_offset] = topk_weights[output_offset] * request.routed_scaling_factor;
        token_expert_indices[output_offset] = (threadIdx.x * request.row_count) + blockIdx.x;
    }
}

static __device__ void SparkCudaRouterScoreLogitRow(SparkCudaRouterRequest request, const uint16_t *logits, float *choice_scores)
{
    uint32_t expert_index;

    for (expert_index = threadIdx.x; expert_index < request.expert_count; expert_index += blockDim.x)
    {
        choice_scores[expert_index] = SparkCudaRouterBf16ToFloat(logits[((uint64_t)blockIdx.x * (uint64_t)request.expert_count) + expert_index]);
    }
}

static __device__ void SparkCudaRouterScoreSoftmaxRow(SparkCudaRouterRequest request, const uint16_t *logits, const float *bias, float *scores, float *choice_scores, float *reduce_values)
{
    uint32_t expert_index;
    float local_max;
    float row_max;
    float local_sum;
    float row_sum;
    float value;

    local_max = -FLT_MAX;
    for (expert_index = threadIdx.x; expert_index < request.expert_count; expert_index += blockDim.x)
    {
        value = SparkCudaRouterBf16ToFloat(logits[((uint64_t)blockIdx.x * (uint64_t)request.expert_count) + expert_index]);
        local_max = value > local_max ? value : local_max;
    }
    row_max = SparkCudaRouterBlockMax(local_max, reduce_values);
    local_sum = 0.0f;
    for (expert_index = threadIdx.x; expert_index < request.expert_count; expert_index += blockDim.x)
    {
        value = __expf(SparkCudaRouterBf16ToFloat(logits[((uint64_t)blockIdx.x * (uint64_t)request.expert_count) + expert_index]) - row_max);
        scores[expert_index] = value;
        local_sum += value;
    }
    row_sum = SparkCudaRouterBlockSum(local_sum, reduce_values);
    for (expert_index = threadIdx.x; expert_index < request.expert_count; expert_index += blockDim.x)
    {
        value = scores[expert_index] / row_sum;
        scores[expert_index] = value;
        choice_scores[expert_index] = request.use_bias != 0u ? value + bias[expert_index] : value;
    }
}

static __device__ void SparkCudaRouterScoreSigmoidRow(SparkCudaRouterRequest request, const uint16_t *logits, const float *bias, float *scores, float *choice_scores)
{
    uint32_t expert_index;
    float value;

    for (expert_index = threadIdx.x; expert_index < request.expert_count; expert_index += blockDim.x)
    {
        value = SparkCudaRouterBf16ToFloat(logits[((uint64_t)blockIdx.x * (uint64_t)request.expert_count) + expert_index]);
        value = SparkCudaRouterSigmoid(value);
        scores[expert_index] = value;
        choice_scores[expert_index] = request.use_bias != 0u ? value + bias[expert_index] : value;
    }
}

static __device__ void SparkCudaRouterSelectGroups(SparkCudaRouterRequest request, const float *choice_scores, float *group_scores, uint32_t *selected_groups)
{
    uint32_t group_index;
    uint32_t expert_index;
    uint32_t selected_index;
    uint32_t best_group;
    uint32_t group_size;
    float best0;
    float best1;
    float value;
    float best_value;

    group_size = request.expert_count / request.expert_group_count;
    if (threadIdx.x < request.expert_group_count)
    {
        best0 = -FLT_MAX;
        best1 = -FLT_MAX;
        for (expert_index = 0u; expert_index < group_size; ++expert_index)
        {
            value = choice_scores[(threadIdx.x * group_size) + expert_index];
            if (value > best0)
            {
                best1 = best0;
                best0 = value;
            }
            else if (value > best1)
            {
                best1 = value;
            }
        }
        group_scores[threadIdx.x] = best0 + best1;
    }
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        for (selected_index = 0u; selected_index < request.top_k_group; ++selected_index)
        {
            best_value = -FLT_MAX;
            best_group = 0u;
            for (group_index = 0u; group_index < request.expert_group_count; ++group_index)
            {
                value = group_scores[group_index];
                if (SparkCudaRouterGroupSelected(selected_groups, selected_index, group_index) == 0u && (value > best_value || (value == best_value && group_index < best_group)))
                {
                    best_value = value;
                    best_group = group_index;
                }
            }
            selected_groups[selected_index] = best_group;
        }
    }
}

static __device__ void SparkCudaRouterSelectExpertIds(SparkCudaRouterRequest request, const float *choice_scores, const uint32_t *selected_groups, uint32_t *selected_experts, float *best_values, uint32_t *best_ids)
{
    uint32_t selected_index;
    uint32_t expert_index;
    uint32_t grouped;
    uint32_t group_size;
    uint32_t group_index;
    uint32_t local_id;
    float local_value;
    float value;

    grouped = request.score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_GROUPED_SIGMOID ? 1u : 0u;
    group_size = grouped != 0u ? request.expert_count / request.expert_group_count : request.expert_count;
    if (threadIdx.x < request.top_k)
    {
        selected_experts[threadIdx.x] = UINT32_MAX;
    }
    __syncthreads();
    for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
    {
        local_value = -FLT_MAX;
        local_id = UINT32_MAX;
        for (expert_index = threadIdx.x; expert_index < request.expert_count; expert_index += blockDim.x)
        {
            group_index = grouped != 0u ? expert_index / group_size : 0u;
            if (SparkCudaRouterExpertSelected(selected_experts, selected_index, expert_index) != 0u)
            {
                continue;
            }
            if (grouped != 0u && SparkCudaRouterGroupSelected(selected_groups, request.top_k_group, group_index) == 0u)
            {
                continue;
            }
            value = choice_scores[expert_index];
            if (SparkCudaRouterBetterChoice(value, expert_index, local_value, local_id) != 0u)
            {
                local_value = value;
                local_id = expert_index;
            }
        }
        SparkCudaRouterBlockBest(local_value, local_id, best_values, best_ids);
        if (threadIdx.x == 0u)
        {
            selected_experts[selected_index] = best_ids[0];
        }
        __syncthreads();
    }
}

static __device__ void SparkCudaRouterSelectExpertsFromLogits(SparkCudaRouterRequest request, const float *choice_scores, uint32_t *selected_experts, float *best_values, uint32_t *best_ids, float *topk_weights, uint32_t *topk_ids, uint32_t *token_expert_indices)
{
    uint32_t selected_index;
    float best_value;
    float value;
    float top_max;
    float weight_sum;
    uint64_t output_offset;

    SparkCudaRouterSelectExpertIds(request, choice_scores, 0, selected_experts, best_values, best_ids);
    if (threadIdx.x != 0u)
    {
        return;
    }
    top_max = -FLT_MAX;
    weight_sum = 0.0f;
    for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
    {
        best_value = choice_scores[selected_experts[selected_index]];
        top_max = best_value > top_max ? best_value : top_max;
    }
    for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
    {
        value = choice_scores[selected_experts[selected_index]];
        if (request.score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_SOFTMAX)
        {
            value = __expf(value - top_max);
        }
        else
        {
            value = SparkCudaRouterSigmoid(value);
        }
        output_offset = ((uint64_t)blockIdx.x * (uint64_t)request.top_k) + selected_index;
        topk_ids[output_offset] = selected_experts[selected_index];
        topk_weights[output_offset] = value;
        weight_sum += value;
    }
    for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
    {
        output_offset = ((uint64_t)blockIdx.x * (uint64_t)request.top_k) + selected_index;
        if (request.renormalize != 0u && weight_sum > 0.0f)
        {
            topk_weights[output_offset] = topk_weights[output_offset] / weight_sum;
        }
        topk_weights[output_offset] = topk_weights[output_offset] * request.routed_scaling_factor;
        token_expert_indices[output_offset] = (selected_index * request.row_count) + blockIdx.x;
    }
}

static __device__ void SparkCudaRouterSelectExperts(SparkCudaRouterRequest request, const float *scores, const float *choice_scores, const uint32_t *selected_groups, uint32_t *selected_experts, float *best_values, uint32_t *best_ids, float *topk_weights, uint32_t *topk_ids, uint32_t *token_expert_indices)
{
    uint32_t selected_index;
    uint32_t best_expert;
    float weight_sum;
    uint64_t output_offset;

    SparkCudaRouterSelectExpertIds(request, choice_scores, selected_groups, selected_experts, best_values, best_ids);
    if (threadIdx.x != 0u)
    {
        return;
    }
    weight_sum = 0.0f;
    for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
    {
        best_expert = selected_experts[selected_index];
        output_offset = ((uint64_t)blockIdx.x * (uint64_t)request.top_k) + selected_index;
        topk_ids[output_offset] = best_expert;
        topk_weights[output_offset] = scores[best_expert];
        weight_sum += scores[best_expert];
    }
    if (request.renormalize != 0u && weight_sum > 0.0f)
    {
        for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
        {
            output_offset = ((uint64_t)blockIdx.x * (uint64_t)request.top_k) + selected_index;
            topk_weights[output_offset] = topk_weights[output_offset] / weight_sum;
        }
    }
    if (request.routed_scaling_factor != 1.0f)
    {
        for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
        {
            output_offset = ((uint64_t)blockIdx.x * (uint64_t)request.top_k) + selected_index;
            topk_weights[output_offset] = topk_weights[output_offset] * request.routed_scaling_factor;
        }
    }
    for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
    {
        output_offset = ((uint64_t)blockIdx.x * (uint64_t)request.top_k) + selected_index;
        token_expert_indices[output_offset] = (selected_index * request.row_count) + blockIdx.x;
    }
}

static __global__ void SparkCudaRouterTopKKernel(SparkCudaRouterRequest request, const uint16_t *logits, const float *bias, float *topk_weights, uint32_t *topk_ids, uint32_t *token_expert_indices)
{
    __shared__ float scores[SPARKPIPE_CUDA_ROUTER_MAX_EXPERTS];
    __shared__ float choice_scores[SPARKPIPE_CUDA_ROUTER_MAX_EXPERTS];
    __shared__ float group_scores[SPARKPIPE_CUDA_ROUTER_MAX_GROUPS];
    __shared__ float reduce_values[8u];
    __shared__ float best_values[SPARK_CUDA_ROUTER_THREADS];
    __shared__ uint32_t selected_groups[SPARKPIPE_CUDA_ROUTER_MAX_GROUPS];
    __shared__ uint32_t selected_experts[SPARKPIPE_CUDA_ROUTER_MAX_TOP_K];
    __shared__ uint32_t best_ids[SPARK_CUDA_ROUTER_THREADS];

    if (threadIdx.x < SPARKPIPE_CUDA_ROUTER_MAX_GROUPS)
    {
        selected_groups[threadIdx.x] = UINT32_MAX;
        group_scores[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();
    if ((request.score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_SOFTMAX || request.score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_SIGMOID) && request.use_bias == 0u)
    {
        SparkCudaRouterScoreLogitRow(request, logits, choice_scores);
        __syncthreads();
        SparkCudaRouterSelectExpertsFromLogits(request, choice_scores, selected_experts, best_values, best_ids, topk_weights, topk_ids, token_expert_indices);
        return;
    }
    if (request.score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_SOFTMAX)
    {
        SparkCudaRouterScoreSoftmaxRow(request, logits, bias, scores, choice_scores, reduce_values);
    }
    else
    {
        SparkCudaRouterScoreSigmoidRow(request, logits, bias, scores, choice_scores);
    }
    __syncthreads();
    if (request.score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_GROUPED_SIGMOID)
    {
        SparkCudaRouterSelectGroups(request, choice_scores, group_scores, selected_groups);
    }
    __syncthreads();
    SparkCudaRouterSelectExperts(request, scores, choice_scores, selected_groups, selected_experts, best_values, best_ids, topk_weights, topk_ids, token_expert_indices);
}

static __global__ void SparkCudaRouterTopK256Kernel(SparkCudaRouterRequest request, const uint16_t *logits, const float *bias, float *topk_weights, uint32_t *topk_ids, uint32_t *token_expert_indices)
{
    __shared__ float warp_values[64u];
    __shared__ float score_values[256u];
    __shared__ float emit_reduce_values[2u];
    __shared__ float emit_weight_sum[1u];
    __shared__ uint32_t warp_ids[64u];
    __shared__ uint32_t emit_reduce_ids[2u];
    __shared__ float group_scores[8u];
    __shared__ uint32_t selected_groups[4u];
    uint32_t expert_index;
    uint32_t group_index;
    uint32_t selected_index;
    uint32_t best_group;
    uint64_t logit_offset;
    float logit_value;
    float score_value;
    float choice_value;
    float best0;
    float best1;
    uint32_t best0_id;
    uint32_t best1_id;
    float best_group_value;

    expert_index = threadIdx.x;
    group_index = threadIdx.x >> 5u;
    logit_offset = ((uint64_t)blockIdx.x * 256u) + (uint64_t)expert_index;
    logit_value = SparkCudaRouterBf16ToFloat(logits[logit_offset]);
    if (request.score_kind != (uint32_t)SPARK_CUDA_ROUTER_SCORE_GROUPED_SIGMOID)
    {
        SparkCudaRouterWarpTop8(logit_value, expert_index, warp_values, warp_ids);
        __syncthreads();
        SparkCudaRouterFastEmitTop8(request, logits, warp_values, warp_ids, topk_weights, topk_ids, token_expert_indices);
        return;
    }
    if (threadIdx.x < 4u)
    {
        selected_groups[threadIdx.x] = UINT32_MAX;
    }
    score_value = SparkCudaRouterSigmoid(logit_value);
    score_values[expert_index] = score_value;
    choice_value = score_value + bias[expert_index];
    best0 = choice_value;
    best0_id = expert_index;
    SparkCudaRouterWarpBestPair(&best0, &best0_id);
    best0 = __shfl_sync(0xffffffffu, best0, 0);
    best0_id = __shfl_sync(0xffffffffu, best0_id, 0);
    if (expert_index == best0_id)
    {
        choice_value = -FLT_MAX;
    }
    best1 = choice_value;
    best1_id = expert_index;
    SparkCudaRouterWarpBestPair(&best1, &best1_id);
    best1 = __shfl_sync(0xffffffffu, best1, 0);
    if ((threadIdx.x & 31u) == 0u)
    {
        group_scores[group_index] = best0 + best1;
    }
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        for (selected_index = 0u; selected_index < 4u; ++selected_index)
        {
            best_group = 0u;
            best_group_value = -FLT_MAX;
            for (group_index = 0u; group_index < 8u; ++group_index)
            {
                if (SparkCudaRouterFastGroupSelected(selected_groups, group_index) == 0u && (group_scores[group_index] > best_group_value || (group_scores[group_index] == best_group_value && group_index < best_group)))
                {
                    best_group = group_index;
                    best_group_value = group_scores[group_index];
                }
            }
            selected_groups[selected_index] = best_group;
        }
    }
    __syncthreads();
    choice_value = SparkCudaRouterFastGroupSelected(selected_groups, threadIdx.x >> 5u) != 0u ? (score_value + bias[expert_index]) : -FLT_MAX;
    SparkCudaRouterWarpTop8(choice_value, expert_index, warp_values, warp_ids);
    __syncthreads();
    SparkCudaRouterFastEmitGroupedTop8(request, score_values, warp_values, warp_ids, emit_reduce_values, emit_reduce_ids, emit_weight_sum, topk_weights, topk_ids, token_expert_indices);
}

static bool SparkCudaRouterCanUseTopK256Host(const SparkCudaRouterRequest *request)
{
    if (request->expert_count != 256u || request->top_k != 8u)
    {
        return false;
    }
    if (request->score_kind == (uint32_t)SPARK_CUDA_ROUTER_SCORE_GROUPED_SIGMOID && request->use_bias != 0u && request->expert_group_count == 8u && request->top_k_group == 4u)
    {
        return true;
    }
    return false;
}

extern "C" SparkStatus SparkRunCudaRouterTopK(const SparkCudaRouterRequest *request, const void *device_logits_bf16, const float *device_bias, float *device_topk_weights, uint32_t *device_topk_ids, uint32_t *device_token_expert_indices, SparkCudaRouterReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRouterRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_logits_bf16 == 0 || device_topk_weights == 0 || device_topk_ids == 0 || device_token_expert_indices == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->use_bias != 0u && device_bias == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaRouterFillReportShape(request, report);
    if (SparkCudaRouterCanUseTopK256Host(request))
    {
        SparkCudaRouterTopK256Kernel<<<request->row_count, SPARK_CUDA_ROUTER_THREADS>>>(*request, (const uint16_t *)device_logits_bf16, device_bias, device_topk_weights, device_topk_ids, device_token_expert_indices);
    }
    else
    {
        SparkCudaRouterTopKKernel<<<request->row_count, SPARK_CUDA_ROUTER_THREADS>>>(*request, (const uint16_t *)device_logits_bf16, device_bias, device_topk_weights, device_topk_ids, device_token_expert_indices);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->router_kernel_count = 1u;
    return SPARK_STATUS_OK;
}
