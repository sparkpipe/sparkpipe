#include "sparkpipe/spark_glm52_sm121_b12x_generated_kernel_table.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeCreate(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out);

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeLaunch(
    void *state_pointer,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments);

extern "C" void SparkFlashInferB12xCompiledMoeDestroy(void *state_pointer);

typedef struct SparkFlashInferB12xCompiledMoeState
{
    SparkGlm52Sm121FlashInferB12xMoeRecipe recipe;
    uint32_t bucket_count;
    SparkGlm52Sm121B12xGeneratedWorkspace *workspaces;
} SparkFlashInferB12xCompiledMoeState;

static size_t SparkGlm52B12xAlignUpSize(
    size_t value,
    size_t alignment)
{
    return ((value + alignment - 1u) / alignment) * alignment;
}

static SparkStatus SparkGlm52B12xCheckedMultiplySize(
    size_t left,
    size_t right,
    size_t *product_out)
{
    if (product_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (left != 0u && right > ((size_t)-1) / left)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *product_out = left * right;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xCudaToSparkStatus(cudaError_t cuda_status)
{
    if (cuda_status == cudaSuccess)
    {
        return SPARK_STATUS_OK;
    }
    if (cuda_status == cudaErrorMemoryAllocation)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52B12xCudaMalloc(
    void **device_pointer_out,
    size_t byte_count)
{
    cudaError_t cuda_status;

    if (device_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *device_pointer_out = 0;
    if (byte_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    cuda_status = cudaMalloc(device_pointer_out, byte_count);
    return SparkGlm52B12xCudaToSparkStatus(cuda_status);
}

static void SparkGlm52B12xCudaFree(void *device_pointer)
{
    if (device_pointer != 0)
    {
        cudaFree(device_pointer);
    }
}

static SparkStatus SparkGlm52B12xAllocateDeviceArray(
    void **device_pointer_out,
    size_t element_count,
    size_t element_size)
{
    SparkStatus status;
    size_t byte_count;

    status = SparkGlm52B12xCheckedMultiplySize(
        element_count,
        element_size,
        &byte_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xCudaMalloc(device_pointer_out, byte_count);
}

static void SparkGlm52B12xReleaseGeneratedWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace)
{
    if (workspace == 0)
    {
        return;
    }
    SparkGlm52B12xCudaFree(workspace->row_counts_i32);
    SparkGlm52B12xCudaFree(workspace->token_map_i32);
    SparkGlm52B12xCudaFree(workspace->token_weights_fp32);
    SparkGlm52B12xCudaFree(workspace->packed_input_u8);
    SparkGlm52B12xCudaFree(workspace->packed_input_scale_u8);
    SparkGlm52B12xCudaFree(workspace->barrier_count_i32);
    SparkGlm52B12xCudaFree(workspace->barrier_epoch_i32);
    SparkGlm52B12xCudaFree(workspace->active_expert_count_i32);
    SparkGlm52B12xCudaFree(workspace->weight_expert_ids_i32);
    SparkGlm52B12xCudaFree(workspace->global_to_local_expert_i32);
    SparkGlm52B12xCudaFree(workspace->compact_topk_ids_i32);
    SparkGlm52B12xCudaFree(workspace->expert_write_rows_i32);
    SparkGlm52B12xCudaFree(workspace->expert_tile_base_i32);
    SparkGlm52B12xCudaFree(workspace->pair_head_i32);
    SparkGlm52B12xCudaFree(workspace->producers_done_count_i32);
    SparkGlm52B12xCudaFree(workspace->all_work_published_i32);
    SparkGlm52B12xCudaFree(workspace->task_head_i32);
    SparkGlm52B12xCudaFree(workspace->task_tail_i32);
    SparkGlm52B12xCudaFree(workspace->task_ready_i32);
    SparkGlm52B12xCudaFree(workspace->task_expert_i32);
    SparkGlm52B12xCudaFree(workspace->task_m_tile_i32);
    SparkGlm52B12xCudaFree(workspace->task_slice_begin_i32);
    SparkGlm52B12xCudaFree(workspace->task_slice_count_i32);
    SparkGlm52B12xCudaFree(workspace->task_valid_rows_i32);
    SparkGlm52B12xCudaFree(workspace->tile_write_count_i32);
    SparkGlm52B12xCudaFree(workspace->route_output_bf16);
    memset(workspace, 0, sizeof(*workspace));
}

static SparkStatus SparkGlm52B12xInitializeIdentityMap(
    void *device_pointer,
    uint32_t element_count)
{
    int32_t *host_values;
    uint32_t index;
    cudaError_t cuda_status;

    if (device_pointer == 0 || element_count == 0u)
    {
        return SPARK_STATUS_OK;
    }

    host_values = (int32_t *)malloc(sizeof(*host_values) * element_count);
    if (host_values == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    for (index = 0u; index < element_count; ++index)
    {
        host_values[index] = (int32_t)index;
    }

    cuda_status = cudaMemcpy(
        device_pointer,
        host_values,
        sizeof(*host_values) * element_count,
        cudaMemcpyHostToDevice);
    free(host_values);
    return SparkGlm52B12xCudaToSparkStatus(cuda_status);
}

__global__ void SparkGlm52B12xPrepareMicroTopKKernel(
    const int32_t *topk_ids,
    int32_t *compact_topk_ids,
    int32_t *weight_expert_ids,
    int32_t *active_expert_count,
    uint32_t route_count)
{
    uint32_t route_index;
    uint32_t active_count;

    if (threadIdx.x != 0u || blockIdx.x != 0u)
    {
        return;
    }
    active_count = 0u;
    for (route_index = 0u; route_index < route_count; ++route_index)
    {
        int32_t expert_id;
        uint32_t compact_index;
        uint32_t found;

        expert_id = topk_ids[route_index];
        found = 0u;
        compact_index = 0u;
        while (compact_index < active_count)
        {
            if (weight_expert_ids[compact_index] == expert_id)
            {
                found = 1u;
                break;
            }
            ++compact_index;
        }
        if (found == 0u)
        {
            compact_index = active_count;
            weight_expert_ids[active_count] = expert_id;
            ++active_count;
        }
        compact_topk_ids[route_index] = (int32_t)compact_index;
    }
    active_expert_count[0] = (int32_t)active_count;
}


__global__ void SparkGlm52B12xPrepareRouterTopKParallelKernel(
    const float *router_logits,
    const float *router_score_bias,
    int32_t *topk_ids,
    float *topk_weights,
    uint32_t token_count,
    uint32_t expert_count,
    uint32_t top_k,
    uint32_t norm_topk_prob,
    float routed_scaling_factor)
{
    __shared__ float shared_choice_scores[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT];
    __shared__ float shared_route_weights[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT];
    __shared__ int32_t shared_selected_ids[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K];
    __shared__ float shared_selected_weights[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K];
    uint32_t token_index;
    uint32_t expert_index;

    token_index = blockIdx.x;
    expert_index = threadIdx.x;
    if (token_index >= token_count ||
        expert_index >= SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT)
    {
        return;
    }

    if (expert_index < expert_count)
    {
        float router_logit;
        float router_score;

        router_logit = router_logits[
            ((uint64_t)token_index *
             (uint64_t)SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT) +
            (uint64_t)expert_index];
        router_score = 1.0f / (1.0f + __expf(-router_logit));
        shared_route_weights[expert_index] = router_score;
        shared_choice_scores[expert_index] =
            router_score + router_score_bias[expert_index];
    }
    else
    {
        shared_route_weights[expert_index] = 0.0f;
        shared_choice_scores[expert_index] = -FLT_MAX;
    }
    __syncthreads();

    if (threadIdx.x == 0u)
    {
        float selected_weight_sum;
        uint32_t selected_index;

        selected_weight_sum = 0.0f;
        for (selected_index = 0u; selected_index < top_k; ++selected_index)
        {
            float best_score;
            uint32_t best_expert;
            uint32_t candidate_expert;

            best_score = -FLT_MAX;
            best_expert = 0u;
            for (candidate_expert = 0u;
                 candidate_expert < expert_count;
                 ++candidate_expert)
            {
                float candidate_score;

                candidate_score = shared_choice_scores[candidate_expert];
                if (candidate_score > best_score ||
                    (candidate_score == best_score &&
                     candidate_expert < best_expert))
                {
                    best_score = candidate_score;
                    best_expert = candidate_expert;
                }
            }
            shared_selected_ids[selected_index] = (int32_t)best_expert;
            shared_selected_weights[selected_index] =
                shared_route_weights[best_expert];
            selected_weight_sum += shared_selected_weights[selected_index];
            shared_choice_scores[best_expert] = -FLT_MAX;
        }
        for (selected_index = 0u; selected_index < top_k; ++selected_index)
        {
            float selected_weight;
            uint64_t route_index;

            selected_weight = shared_selected_weights[selected_index];
            if (norm_topk_prob != 0u && selected_weight_sum > 0.0f)
            {
                selected_weight /= selected_weight_sum;
            }
            selected_weight *= routed_scaling_factor;
            route_index =
                ((uint64_t)token_index * (uint64_t)top_k) +
                (uint64_t)selected_index;
            topk_ids[route_index] = shared_selected_ids[selected_index];
            topk_weights[route_index] = selected_weight;
        }
    }
}

__global__ void SparkGlm52B12xPrepareRouterTopKMicroKernel(
    const float *router_logits,
    const float *router_score_bias,
    int32_t *topk_ids,
    float *topk_weights,
    int32_t *compact_topk_ids,
    int32_t *weight_expert_ids,
    int32_t *active_expert_count,
    uint32_t token_count,
    uint32_t expert_count,
    uint32_t top_k,
    uint32_t norm_topk_prob,
    float routed_scaling_factor)
{
    __shared__ float shared_choice_scores[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT];
    __shared__ float shared_route_weights[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT];
    __shared__ int32_t shared_selected_ids[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K];
    __shared__ float shared_selected_weights[
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K];
    __shared__ uint32_t shared_active_expert_count;
    uint32_t token_index;
    uint32_t expert_index;

    if (threadIdx.x == 0u)
    {
        shared_active_expert_count = 0u;
    }
    __syncthreads();

    expert_index = threadIdx.x;
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        if (expert_index < expert_count)
        {
            float router_logit;
            float router_score;

            router_logit = router_logits[
                ((uint64_t)token_index *
                 (uint64_t)SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT) +
                (uint64_t)expert_index];
            router_score = 1.0f / (1.0f + __expf(-router_logit));
            shared_route_weights[expert_index] = router_score;
            shared_choice_scores[expert_index] =
                router_score + router_score_bias[expert_index];
        }
        else if (expert_index < SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT)
        {
            shared_route_weights[expert_index] = 0.0f;
            shared_choice_scores[expert_index] = -FLT_MAX;
        }
        __syncthreads();

        if (threadIdx.x == 0u)
        {
            float selected_weight_sum;
            uint32_t selected_index;

            selected_weight_sum = 0.0f;
            for (selected_index = 0u; selected_index < top_k; ++selected_index)
            {
                float best_score;
                uint32_t best_expert;
                uint32_t candidate_expert;

                best_score = -FLT_MAX;
                best_expert = 0u;
                for (candidate_expert = 0u;
                     candidate_expert < expert_count;
                     ++candidate_expert)
                {
                    float candidate_score;

                    candidate_score = shared_choice_scores[candidate_expert];
                    if (candidate_score > best_score ||
                        (candidate_score == best_score &&
                         candidate_expert < best_expert))
                    {
                        best_score = candidate_score;
                        best_expert = candidate_expert;
                    }
                }
                shared_selected_ids[selected_index] = (int32_t)best_expert;
                shared_selected_weights[selected_index] =
                    shared_route_weights[best_expert];
                selected_weight_sum += shared_selected_weights[selected_index];
                shared_choice_scores[best_expert] = -FLT_MAX;
            }

            for (selected_index = 0u; selected_index < top_k; ++selected_index)
            {
                uint64_t route_index;
                uint32_t compact_index;
                uint32_t found;
                float selected_weight;
                int32_t selected_expert;

                route_index =
                    ((uint64_t)token_index * (uint64_t)top_k) +
                    (uint64_t)selected_index;
                selected_expert = shared_selected_ids[selected_index];
                selected_weight = shared_selected_weights[selected_index];
                if (norm_topk_prob != 0u && selected_weight_sum > 0.0f)
                {
                    selected_weight /= selected_weight_sum;
                }
                selected_weight *= routed_scaling_factor;
                topk_ids[route_index] = selected_expert;
                topk_weights[route_index] = selected_weight;

                found = 0u;
                compact_index = 0u;
                while (compact_index < shared_active_expert_count)
                {
                    if (weight_expert_ids[compact_index] == selected_expert)
                    {
                        found = 1u;
                        break;
                    }
                    ++compact_index;
                }
                if (found == 0u)
                {
                    compact_index = shared_active_expert_count;
                    weight_expert_ids[compact_index] = selected_expert;
                    ++shared_active_expert_count;
                }
                compact_topk_ids[route_index] = (int32_t)compact_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u)
    {
        active_expert_count[0] = (int32_t)shared_active_expert_count;
    }
}


__global__ void SparkGlm52B12xBenchmarkForceExpertCoverageKernel(
    int32_t *topk_ids,
    float *topk_weights,
    int32_t *compact_topk_ids,
    int32_t *weight_expert_ids,
    int32_t *active_expert_count,
    uint32_t token_count,
    uint32_t expert_count,
    uint32_t top_k,
    uint32_t forced_expert_count,
    float routed_scaling_factor)
{
    uint64_t route_index;
    uint64_t route_count;

    route_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    route_count = (uint64_t)token_count * (uint64_t)top_k;
    if (topk_ids == 0 || topk_weights == 0 || route_index >= route_count ||
        token_count == 0u || expert_count == 0u || top_k == 0u ||
        forced_expert_count == 0u)
    {
        return;
    }
    if (forced_expert_count > expert_count)
    {
        forced_expert_count = expert_count;
    }
    topk_ids[route_index] = (int32_t)(route_index % forced_expert_count);
    topk_weights[route_index] = routed_scaling_factor / (float)top_k;
    if (compact_topk_ids != 0)
    {
        compact_topk_ids[route_index] = topk_ids[route_index];
    }
    if (route_index < (uint64_t)forced_expert_count && weight_expert_ids != 0)
    {
        weight_expert_ids[route_index] = (int32_t)route_index;
    }
    if (route_index == 0u && active_expert_count != 0)
    {
        active_expert_count[0] = (int32_t)forced_expert_count;
    }
}

static uint32_t SparkGlm52B12xBenchmarkForcedExpertCoverage(
    uint32_t token_count,
    uint32_t expert_count,
    uint32_t top_k)
{
    const char *text;
    char *end_pointer;
    unsigned long value;
    uint64_t route_count;

    text = getenv("GLM52_BENCHMARK_FORCE_EXPERT_COVERAGE");
    if (text == 0 || text[0] == '\0' || token_count == 0u ||
        expert_count == 0u || top_k == 0u)
    {
        return 0u;
    }
    value = strtoul(text, &end_pointer, 10);
    if (end_pointer == text || value == 0ul)
    {
        return 0u;
    }
    route_count = (uint64_t)token_count * (uint64_t)top_k;
    if (route_count > (uint64_t)UINT32_MAX)
    {
        route_count = (uint64_t)UINT32_MAX;
    }
    if (value > (unsigned long)expert_count)
    {
        value = (unsigned long)expert_count;
    }
    if (value > (unsigned long)route_count)
    {
        value = (unsigned long)route_count;
    }
    return (uint32_t)value;
}

static SparkStatus SparkGlm52B12xMaybeForceBenchmarkExpertCoverage(
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments,
    const int32_t **launch_topk_ids_inout)
{
    cudaStream_t cuda_stream;
    cudaError_t cuda_status;
    uint32_t forced_expert_count;
    uint32_t route_count;
    int32_t *compact_topk_ids;
    int32_t *weight_expert_ids;
    int32_t *active_expert_count;

    if (bucket == 0 || workspace == 0 || arguments == 0 ||
        launch_topk_ids_inout == 0 || arguments->topk_ids_i32 == 0 ||
        arguments->topk_weights_fp32 == 0 || arguments->top_k == 0u ||
        arguments->token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    forced_expert_count = SparkGlm52B12xBenchmarkForcedExpertCoverage(
        arguments->token_count,
        arguments->expert_count,
        arguments->top_k);
    if (forced_expert_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    route_count = arguments->token_count * arguments->top_k;
    if (route_count > bucket->max_rows)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    compact_topk_ids = 0;
    weight_expert_ids = 0;
    active_expert_count = 0;
    if (bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_MICRO)
    {
        if (workspace->compact_topk_ids_i32 == 0 ||
            workspace->weight_expert_ids_i32 == 0 ||
            workspace->active_expert_count_i32 == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        compact_topk_ids = (int32_t *)workspace->compact_topk_ids_i32;
        weight_expert_ids = (int32_t *)workspace->weight_expert_ids_i32;
        active_expert_count = (int32_t *)workspace->active_expert_count_i32;
        *launch_topk_ids_inout = (const int32_t *)workspace->compact_topk_ids_i32;
    }
    cuda_stream = (cudaStream_t)arguments->cuda_stream;
    SparkGlm52B12xBenchmarkForceExpertCoverageKernel<<<
        (route_count + 255u) / 256u,
        256u,
        0u,
        cuda_stream>>>(
        (int32_t *)arguments->topk_ids_i32,
        arguments->topk_weights_fp32,
        compact_topk_ids,
        weight_expert_ids,
        active_expert_count,
        arguments->token_count,
        arguments->expert_count,
        arguments->top_k,
        forced_expert_count,
        arguments->router_routed_scaling_factor);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SparkGlm52B12xCudaToSparkStatus(cuda_status);
    }
    fprintf(
        stderr,
        "b12x_forced_expert_coverage tokens=%u active_experts=%u of=%u routes=%u\n",
        arguments->token_count,
        forced_expert_count,
        arguments->expert_count,
        route_count);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xPrepareMicroTopK(
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    uint32_t route_count;
    cudaStream_t cuda_stream;

    if (bucket == 0 || workspace == 0 || arguments == 0 ||
        arguments->topk_ids_i32 == 0 ||
        workspace->compact_topk_ids_i32 == 0 ||
        workspace->weight_expert_ids_i32 == 0 ||
        workspace->active_expert_count_i32 == 0 ||
        arguments->top_k == 0u ||
        arguments->token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route_count = arguments->token_count * arguments->top_k;
    if (route_count > bucket->max_rows)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    cuda_stream = (cudaStream_t)arguments->cuda_stream;
    SparkGlm52B12xPrepareMicroTopKKernel<<<1u, 1u, 0u, cuda_stream>>>(
        arguments->topk_ids_i32,
        (int32_t *)workspace->compact_topk_ids_i32,
        (int32_t *)workspace->weight_expert_ids_i32,
        (int32_t *)workspace->active_expert_count_i32,
        route_count);
    return SparkGlm52B12xCudaToSparkStatus(cudaGetLastError());
}


static SparkStatus SparkGlm52B12xPrepareRouterTopK(
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments,
    const int32_t **launch_topk_ids_out)
{
    uint32_t route_count;
    cudaStream_t cuda_stream;

    if (bucket == 0 || workspace == 0 || arguments == 0 ||
        launch_topk_ids_out == 0 ||
        arguments->router_logits_f32 == 0 ||
        arguments->router_score_bias_f32 == 0 ||
        arguments->topk_ids_i32 == 0 ||
        arguments->topk_weights_fp32 == 0 ||
        arguments->top_k == 0u ||
        arguments->token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route_count = arguments->token_count * arguments->top_k;
    if (route_count > bucket->max_rows)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    cuda_stream = (cudaStream_t)arguments->cuda_stream;
    if (bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_MICRO)
    {
        if (workspace->compact_topk_ids_i32 == 0 ||
            workspace->weight_expert_ids_i32 == 0 ||
            workspace->active_expert_count_i32 == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        SparkGlm52B12xPrepareRouterTopKMicroKernel<<<
            1u,
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT,
            0u,
            cuda_stream>>>(
            arguments->router_logits_f32,
            arguments->router_score_bias_f32,
            arguments->topk_ids_i32,
            arguments->topk_weights_fp32,
            (int32_t *)workspace->compact_topk_ids_i32,
            (int32_t *)workspace->weight_expert_ids_i32,
            (int32_t *)workspace->active_expert_count_i32,
            arguments->token_count,
            arguments->expert_count,
            arguments->top_k,
            arguments->router_norm_topk_prob,
            arguments->router_routed_scaling_factor);
        *launch_topk_ids_out =
            (const int32_t *)workspace->compact_topk_ids_i32;
    }
    else
    {
        SparkGlm52B12xPrepareRouterTopKParallelKernel<<<
            arguments->token_count,
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT,
            0u,
            cuda_stream>>>(
            arguments->router_logits_f32,
            arguments->router_score_bias_f32,
            arguments->topk_ids_i32,
            arguments->topk_weights_fp32,
            arguments->token_count,
            arguments->expert_count,
            arguments->top_k,
            arguments->router_norm_topk_prob,
            arguments->router_routed_scaling_factor);
        *launch_topk_ids_out = arguments->topk_ids_i32;
    }
    return SparkGlm52B12xCudaToSparkStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52B12xAllocateCommonControlWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    uint32_t expert_count)
{
    SparkStatus status;

    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->row_counts_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->barrier_count_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->barrier_epoch_i32,
        1u,
        sizeof(int32_t));
    return status;
}

static SparkStatus SparkGlm52B12xAllocateStaticWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket)
{
    SparkStatus status;
    size_t expert_count;
    size_t max_rows;
    size_t hidden_dimension;
    size_t rows_pad_k;
    size_t cols_pad_k;
    size_t expert_rows;
    size_t packed_input_elements;
    size_t packed_scale_elements;
    size_t compact_count;

    expert_count = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT;
    hidden_dimension = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION;
    max_rows = bucket->max_rows;
    rows_pad_k = SparkGlm52B12xAlignUpSize(max_rows, 128u);
    cols_pad_k = SparkGlm52B12xAlignUpSize(
        hidden_dimension / SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_NVFP4_GROUP_SIZE,
        4u);

    status = SparkGlm52B12xCheckedMultiplySize(
        expert_count,
        max_rows,
        &expert_rows);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xCheckedMultiplySize(
        expert_rows,
        hidden_dimension / 2u,
        &packed_input_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xCheckedMultiplySize(
        expert_count,
        rows_pad_k,
        &packed_scale_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xCheckedMultiplySize(
        packed_scale_elements,
        cols_pad_k,
        &packed_scale_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    compact_count = expert_count;
    if (compact_count < max_rows)
    {
        compact_count = max_rows;
    }

    status = SparkGlm52B12xAllocateCommonControlWorkspace(workspace, expert_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_map_i32,
        expert_rows,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_weights_fp32,
        expert_rows,
        sizeof(float));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_u8,
        packed_input_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_scale_u8,
        packed_scale_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->active_expert_count_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->weight_expert_ids_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->global_to_local_expert_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->compact_topk_ids_i32,
        compact_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xInitializeIdentityMap(
        workspace->weight_expert_ids_i32,
        (uint32_t)expert_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xInitializeIdentityMap(
        workspace->global_to_local_expert_i32,
        (uint32_t)expert_count);
}

static SparkStatus SparkGlm52B12xAllocateDynamicWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket)
{
    SparkStatus status;
    size_t expert_count;
    size_t max_rows;
    size_t hidden_dimension;
    size_t cols_pad_k;
    size_t packed_input_elements;
    size_t packed_scale_elements;
    size_t task_count;

    expert_count = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT;
    hidden_dimension = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION;
    max_rows = bucket->max_rows;
    task_count = bucket->task_capacity;
    cols_pad_k = SparkGlm52B12xAlignUpSize(
        hidden_dimension / SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_NVFP4_GROUP_SIZE,
        4u);

    status = SparkGlm52B12xCheckedMultiplySize(
        max_rows,
        hidden_dimension / 2u,
        &packed_input_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xCheckedMultiplySize(
        max_rows,
        cols_pad_k,
        &packed_scale_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xAllocateCommonControlWorkspace(workspace, expert_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_map_i32,
        max_rows,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_weights_fp32,
        max_rows,
        sizeof(float));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_u8,
        packed_input_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_scale_u8,
        packed_scale_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->expert_write_rows_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->expert_tile_base_i32,
        expert_count + 1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->pair_head_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->producers_done_count_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->all_work_published_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_head_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_tail_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_ready_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_expert_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_m_tile_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_slice_begin_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_slice_count_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_valid_rows_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xAllocateDeviceArray(
        &workspace->tile_write_count_i32,
        bucket->physical_tile_capacity,
        sizeof(int32_t));
}

static uint32_t SparkGlm52B12xGetRouteOutputSliceCount(
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    uint32_t intermediate_dimension)
{
    uint32_t slice_tile_n;

    if (bucket == 0 || intermediate_dimension == 0u)
    {
        return 0u;
    }
    if (bucket->route_output_slice_count != 0u)
    {
        return bucket->route_output_slice_count;
    }
    slice_tile_n = bucket->static_mma_tile_n;
    if (slice_tile_n == 0u)
    {
        slice_tile_n = 128u;
    }
    return (intermediate_dimension + slice_tile_n - 1u) / slice_tile_n;
}

static SparkStatus SparkGlm52B12xAllocateDeterministicFinalizeWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket);

static SparkStatus SparkGlm52B12xAllocateGeneratedWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket)
{
    SparkStatus status;

    if (workspace == 0 || bucket == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(workspace, 0, sizeof(*workspace));
    workspace->abi_version = SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION;
    workspace->backend_kind = bucket->backend_kind;
    workspace->routed_rows_capacity = bucket->routed_rows_capacity;
    workspace->max_rows = bucket->max_rows;
    workspace->physical_tile_capacity = bucket->physical_tile_capacity;
    workspace->task_capacity = bucket->task_capacity;

    if (bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_MICRO ||
        bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_STATIC)
    {
        status = SparkGlm52B12xAllocateStaticWorkspace(workspace, bucket);
    }
    else if (bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_DYNAMIC)
    {
        status = SparkGlm52B12xAllocateDynamicWorkspace(workspace, bucket);
    }
    else
    {
        status = SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xAllocateDeterministicFinalizeWorkspace(
            workspace,
            bucket);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52B12xReleaseGeneratedWorkspace(workspace);
    }
    return status;
}

static SparkStatus SparkGlm52B12xMemsetAsyncIfPresent(
    void *device_pointer,
    size_t byte_count,
    cudaStream_t cuda_stream)
{
    cudaError_t cuda_status;

    if (device_pointer == 0 || byte_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    cuda_status = cudaMemsetAsync(device_pointer, 0, byte_count, cuda_stream);
    return SparkGlm52B12xCudaToSparkStatus(cuda_status);
}

static SparkStatus SparkGlm52B12xAllocateDeterministicFinalizeWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket)
{
    SparkStatus status;
    size_t output_element_count;

    if (workspace == 0 || bucket == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52B12xCheckedMultiplySize(
        (size_t)bucket->max_rows,
        (size_t)SparkGlm52B12xGetRouteOutputSliceCount(
            bucket,
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION),
        &output_element_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xCheckedMultiplySize(
        output_element_count,
        (size_t)SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION,
        &output_element_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xAllocateDeviceArray(
        &workspace->route_output_bf16,
        output_element_count,
        sizeof(uint16_t));
}

__global__ void SparkGlm52B12xDeterministicFc2FinalizeKernel(
    const uint16_t *route_output_bf16,
    uint16_t *output_bf16,
    uint32_t token_count,
    uint32_t top_k,
    uint32_t route_output_slice_count,
    uint32_t hidden_dimension)
{
    uint64_t element_index;
    uint64_t element_stride;
    uint64_t total_elements;

    element_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    element_stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    total_elements = (uint64_t)token_count * (uint64_t)hidden_dimension;

    while (element_index < total_elements)
    {
        uint32_t token_index;
        uint32_t hidden_index;
        uint32_t route_index;
        float sum;

        token_index = (uint32_t)(element_index / (uint64_t)hidden_dimension);
        hidden_index = (uint32_t)(element_index -
            ((uint64_t)token_index * (uint64_t)hidden_dimension));
        sum = 0.0f;
        for (route_index = 0u; route_index < top_k; ++route_index)
        {
            uint32_t slice_index;

            for (slice_index = 0u; slice_index < route_output_slice_count; ++slice_index)
            {
                uint64_t route_row;
                uint64_t route_slice_row;
                uint64_t route_offset;
                __nv_bfloat16 route_value;

                route_row =
                    ((uint64_t)token_index * (uint64_t)top_k) +
                    (uint64_t)route_index;
                route_slice_row =
                    (route_row * (uint64_t)route_output_slice_count) +
                    (uint64_t)slice_index;
                route_offset =
                    (route_slice_row * (uint64_t)hidden_dimension) +
                    (uint64_t)hidden_index;
                route_value = *reinterpret_cast<const __nv_bfloat16 *>(
                    route_output_bf16 + route_offset);
                sum += __bfloat162float(route_value);
            }
        }
        *reinterpret_cast<__nv_bfloat16 *>(output_bf16 + element_index) =
            __float2bfloat16_rn(sum);
        element_index += element_stride;
    }
}

static SparkStatus SparkGlm52B12xLaunchDeterministicFc2Finalize(
    const SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    cudaStream_t cuda_stream;
    uint32_t block_count;
    uint32_t route_output_slice_count;
    uint64_t total_elements;

    if (workspace == 0 || bucket == 0 || arguments == 0 ||
        workspace->route_output_bf16 == 0 || arguments->output_bf16 == 0 ||
        arguments->cuda_stream == 0 || arguments->token_count == 0u ||
        arguments->top_k == 0u || arguments->hidden_dimension == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    route_output_slice_count = SparkGlm52B12xGetRouteOutputSliceCount(
        bucket,
        arguments->intermediate_dimension);
    if (route_output_slice_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    total_elements =
        (uint64_t)arguments->token_count *
        (uint64_t)arguments->hidden_dimension;
    block_count = (uint32_t)((total_elements + 255u) / 256u);
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 2048u)
    {
        block_count = 2048u;
    }
    cuda_stream = (cudaStream_t)arguments->cuda_stream;
    SparkGlm52B12xDeterministicFc2FinalizeKernel<<<
        block_count,
        256u,
        0u,
        cuda_stream>>>(
        (const uint16_t *)workspace->route_output_bf16,
        (uint16_t *)arguments->output_bf16,
        arguments->token_count,
        arguments->top_k,
        route_output_slice_count,
        arguments->hidden_dimension);
    return SparkGlm52B12xCudaToSparkStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52B12xResetLaunchState(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    cudaStream_t cuda_stream;
    size_t expert_count;
    size_t task_count;
    size_t output_element_count;
    SparkStatus status;

    if (workspace == 0 || bucket == 0 || arguments == 0 ||
        workspace->route_output_bf16 == 0 ||
        arguments->output_bf16 == 0 || arguments->cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cuda_stream = (cudaStream_t)arguments->cuda_stream;
    expert_count = (size_t)arguments->expert_count;
    task_count = (size_t)bucket->task_capacity;

    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->row_counts_i32,
        expert_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->barrier_count_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->barrier_epoch_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->active_expert_count_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->expert_write_rows_i32,
        expert_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->expert_tile_base_i32,
        (expert_count + 1u) * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->pair_head_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->producers_done_count_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->all_work_published_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_head_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_tail_i32,
        sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_ready_i32,
        task_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_expert_i32,
        task_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_m_tile_i32,
        task_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_slice_begin_i32,
        task_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_slice_count_i32,
        task_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->task_valid_rows_i32,
        task_count * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->tile_write_count_i32,
        (size_t)bucket->physical_tile_capacity * sizeof(int32_t),
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xCheckedMultiplySize(
        (size_t)arguments->token_count,
        (size_t)arguments->top_k,
        &output_element_count);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xCheckedMultiplySize(
        output_element_count,
        (size_t)SparkGlm52B12xGetRouteOutputSliceCount(
            bucket,
            arguments->intermediate_dimension),
        &output_element_count);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52B12xCheckedMultiplySize(
        output_element_count,
        (size_t)arguments->hidden_dimension,
        &output_element_count);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkGlm52B12xMemsetAsyncIfPresent(
        workspace->route_output_bf16,
        output_element_count * sizeof(uint16_t),
        cuda_stream);
}

static SparkStatus SparkGlm52B12xValidateRecipeAgainstGeneratedManifest(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe)
{
    const SparkGlm52Sm121B12xGeneratedManifest *manifest;

    if (recipe == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    manifest = &SparkGlm52Sm121B12xGeneratedManifestInstance;
    if (manifest->abi_version !=
        SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION ||
        manifest->bucket_count == 0u ||
        manifest->buckets == 0 ||
        (manifest->manifest_flags &
            SPARK_GLM52_SM121_B12X_GENERATED_MANIFEST_REQUIRED_FLAGS) !=
            SPARK_GLM52_SM121_B12X_GENERATED_MANIFEST_REQUIRED_FLAGS)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (manifest->hidden_dimension != recipe->hidden_dimension ||
        manifest->intermediate_dimension != recipe->intermediate_dimension ||
        manifest->expert_count != recipe->expert_count ||
        manifest->top_k != recipe->top_k ||
        manifest->maximum_token_count == 0u ||
        recipe->maximum_token_count == 0u ||
        manifest->maximum_token_count != recipe->maximum_token_count ||
        manifest->cuda_architecture != recipe->cuda_architecture)
    {
        return SPARK_STATUS_TARGET_MISMATCH;
    }
    if (manifest->manifest_hash_low64 != recipe->kernel_manifest_hash_low64)
    {
        return SPARK_STATUS_HASH_MISMATCH;
    }
    return SPARK_STATUS_OK;
}

static const SparkGlm52Sm121B12xGeneratedKernelBucket *SparkGlm52B12xSelectBucket(
    const SparkFlashInferB12xCompiledMoeState *state,
    uint32_t token_count,
    uint32_t *bucket_index_out)
{
    const SparkGlm52Sm121B12xGeneratedManifest *manifest;
    uint32_t bucket_index;

    if (state == 0 || bucket_index_out == 0)
    {
        return 0;
    }

    manifest = &SparkGlm52Sm121B12xGeneratedManifestInstance;
    for (bucket_index = 0u; bucket_index < manifest->bucket_count; ++bucket_index)
    {
        if (token_count == manifest->buckets[bucket_index].token_upper_bound)
        {
            *bucket_index_out = bucket_index;
            return &manifest->buckets[bucket_index];
        }
    }
    return 0;
}

static const SparkGlm52Sm121B12xGeneratedKernelBucket *
SparkGlm52B12xSelectLargestBucketAtMost(
    const SparkFlashInferB12xCompiledMoeState *state,
    uint32_t token_count,
    uint32_t *bucket_index_out)
{
    const SparkGlm52Sm121B12xGeneratedManifest *manifest;
    const SparkGlm52Sm121B12xGeneratedKernelBucket *best_bucket;
    uint32_t best_index;
    uint32_t bucket_index;

    if (state == 0 || token_count == 0u || bucket_index_out == 0)
    {
        return 0;
    }

    manifest = &SparkGlm52Sm121B12xGeneratedManifestInstance;
    best_bucket = 0;
    best_index = 0u;
    for (bucket_index = 0u; bucket_index < manifest->bucket_count; ++bucket_index)
    {
        if (manifest->buckets[bucket_index].token_upper_bound <= token_count &&
            (best_bucket == 0 ||
             manifest->buckets[bucket_index].token_upper_bound >
                best_bucket->token_upper_bound))
        {
            best_bucket = &manifest->buckets[bucket_index];
            best_index = bucket_index;
        }
    }
    if (best_bucket == 0)
    {
        return 0;
    }
    *bucket_index_out = best_index;
    return best_bucket;
}

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeCreate(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out)
{
    SparkFlashInferB12xCompiledMoeState *state;
    SparkStatus status;
    uint32_t bucket_index;

    if (state_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *state_out = 0;

    status = SparkGlm52B12xValidateRecipeAgainstGeneratedManifest(recipe);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    state = (SparkFlashInferB12xCompiledMoeState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    state->recipe = *recipe;
    state->bucket_count = SparkGlm52Sm121B12xGeneratedManifestInstance.bucket_count;
    state->workspaces = (SparkGlm52Sm121B12xGeneratedWorkspace *)calloc(
        state->bucket_count,
        sizeof(*state->workspaces));
    if (state->workspaces == 0)
    {
        free(state);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    for (bucket_index = 0u; bucket_index < state->bucket_count; ++bucket_index)
    {
        status = SparkGlm52B12xAllocateGeneratedWorkspace(
            &state->workspaces[bucket_index],
            &SparkGlm52Sm121B12xGeneratedManifestInstance.buckets[bucket_index]);
        if (status != SPARK_STATUS_OK)
        {
            SparkFlashInferB12xCompiledMoeDestroy(state);
            return status;
        }
    }

    *state_out = state;
    return SPARK_STATUS_OK;
}

extern "C" uint64_t SparkFlashInferB12xCompiledMoeActiveManifestHashLow64(void)
{
    return SparkGlm52Sm121B12xGeneratedManifestInstance.manifest_hash_low64;
}

static SparkStatus SparkGlm52B12xMaybeLogExpertCoverage(const SparkGlm52Sm121B12xGeneratedWorkspace *workspace, const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments, uint32_t chunk_token_count)
{
    cudaStream_t cuda_stream;
    cudaStreamCaptureStatus capture_status;
    cudaError_t cuda_status;
    int32_t active_expert_count = -1;

    if (getenv("GLM52_LOG_EXPERT_COVERAGE") == 0)
        return SPARK_STATUS_OK;
    cuda_stream = (cudaStream_t)arguments->cuda_stream;
    cuda_status = cudaStreamIsCapturing(cuda_stream, &capture_status);
    if (cuda_status != cudaSuccess)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (capture_status != cudaStreamCaptureStatusNone)
    {
        fprintf(stderr, "b12x_expert_coverage skipped: stream is graph-capturing\n");
        return SPARK_STATUS_OK;
    }
    cuda_status = cudaMemcpyAsync(&active_expert_count, workspace->active_expert_count_i32, sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (cuda_status != cudaSuccess)
        return SPARK_STATUS_INTERNAL_ERROR;
    cuda_status = cudaStreamSynchronize(cuda_stream);
    if (cuda_status != cudaSuccess)
        return SPARK_STATUS_INTERNAL_ERROR;
    fprintf(stderr, "b12x_expert_coverage tokens=%u active_experts=%d of=%u\n", chunk_token_count, active_expert_count, arguments->expert_count);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xLaunchSelectedBucket(
    SparkFlashInferB12xCompiledMoeState *state,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    uint32_t bucket_index)
{
    SparkGlm52Sm121B12xGeneratedLaunchArguments generated_arguments;
    const int32_t *launch_topk_ids;
    SparkStatus status;

    if (state == 0 || arguments == 0 || bucket == 0 ||
        bucket_index >= state->bucket_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52B12xResetLaunchState(
        &state->workspaces[bucket_index],
        bucket,
        arguments);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    launch_topk_ids = arguments->topk_ids_i32;
    if ((arguments->argument_flags &
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_ROUTER_LOGITS) != 0u)
    {
        status = SparkGlm52B12xPrepareRouterTopK(
            bucket,
            &state->workspaces[bucket_index],
            arguments,
            &launch_topk_ids);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    else if (bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_MICRO)
    {
        status = SparkGlm52B12xPrepareMicroTopK(
            bucket,
            &state->workspaces[bucket_index],
            arguments);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        launch_topk_ids =
            (const int32_t *)state->workspaces[bucket_index].compact_topk_ids_i32;
    }

    status = SparkGlm52B12xMaybeForceBenchmarkExpertCoverage(
        bucket,
        &state->workspaces[bucket_index],
        arguments,
        &launch_topk_ids);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&generated_arguments, 0, sizeof(generated_arguments));
    generated_arguments.abi_version =
        SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION;
    generated_arguments.token_count = arguments->token_count;
    generated_arguments.maximum_token_count = arguments->maximum_token_count;
    generated_arguments.expert_count = arguments->expert_count;
    generated_arguments.top_k = arguments->top_k;
    generated_arguments.hidden_dimension = arguments->hidden_dimension;
    generated_arguments.intermediate_dimension = arguments->intermediate_dimension;
    generated_arguments.hidden_bf16 = arguments->hidden_bf16;
    generated_arguments.topk_ids_i32 = launch_topk_ids;
    generated_arguments.topk_weights_fp32 = arguments->topk_weights_fp32;
    generated_arguments.w1_weight_fp4_static_view = arguments->w1_weight_fp4_static_view;
    generated_arguments.w1_scale_static_storage_ue4m3 = arguments->w1_scale_static_storage_ue4m3;
    generated_arguments.w1_alpha_fp32_by_expert = arguments->w1_alpha_fp32_by_expert;
    generated_arguments.fc2_input_scale_fp32_by_expert = arguments->fc2_input_scale_fp32_by_expert;
    generated_arguments.w2_weight_fp4_static_view = arguments->w2_weight_fp4_static_view;
    generated_arguments.w2_scale_static_storage_ue4m3 = arguments->w2_scale_static_storage_ue4m3;
    generated_arguments.w2_alpha_fp32_by_expert = arguments->w2_alpha_fp32_by_expert;
    generated_arguments.output_bf16 =
        state->workspaces[bucket_index].route_output_bf16;
    generated_arguments.user_workspace = arguments->workspace;
    generated_arguments.user_workspace_bytes = arguments->workspace_bytes;
    generated_arguments.generated_workspace = &state->workspaces[bucket_index];
    generated_arguments.cuda_stream = arguments->cuda_stream;

    status = SparkGlm52Sm121B12xGeneratedLaunch(bucket, &generated_arguments);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xMaybeLogExpertCoverage(
        &state->workspaces[bucket_index],
        arguments,
        arguments->token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xLaunchDeterministicFc2Finalize(
        &state->workspaces[bucket_index],
        bucket,
        arguments);
}

static const void *SparkGlm52B12xOffsetConstElements(
    const void *pointer,
    uint32_t token_offset,
    uint32_t element_count,
    uint32_t element_bytes)
{
    return (const void *)((const uint8_t *)pointer +
        ((size_t)token_offset * (size_t)element_count * (size_t)element_bytes));
}

static void *SparkGlm52B12xOffsetElements(
    void *pointer,
    uint32_t token_offset,
    uint32_t element_count,
    uint32_t element_bytes)
{
    return (void *)((uint8_t *)pointer +
        ((size_t)token_offset * (size_t)element_count * (size_t)element_bytes));
}

static SparkStatus SparkGlm52B12xLaunchChunked(
    SparkFlashInferB12xCompiledMoeState *state,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    SparkGlm52Sm121FlashInferB12xMoeArguments chunk_arguments;
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket;
    SparkStatus status;
    uint32_t bucket_index;
    uint32_t token_offset;
    uint32_t remaining_tokens;

    if (state == 0 || arguments == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    token_offset = 0u;
    while (token_offset < arguments->token_count)
    {
        remaining_tokens = arguments->token_count - token_offset;
        bucket = SparkGlm52B12xSelectLargestBucketAtMost(
            state,
            remaining_tokens,
            &bucket_index);
        if (bucket == 0)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        chunk_arguments = *arguments;
        chunk_arguments.token_count = bucket->token_upper_bound;
        chunk_arguments.hidden_bf16 = SparkGlm52B12xOffsetConstElements(
            arguments->hidden_bf16,
            token_offset,
            arguments->hidden_dimension,
            sizeof(uint16_t));
        chunk_arguments.topk_ids_i32 =
            (int32_t *)SparkGlm52B12xOffsetElements(
                arguments->topk_ids_i32,
                token_offset,
                arguments->top_k,
                sizeof(int32_t));
        chunk_arguments.topk_weights_fp32 =
            (float *)SparkGlm52B12xOffsetElements(
                arguments->topk_weights_fp32,
                token_offset,
                arguments->top_k,
                sizeof(float));
        if (arguments->router_logits_f32 != 0)
        {
            chunk_arguments.router_logits_f32 =
                (const float *)SparkGlm52B12xOffsetConstElements(
                    arguments->router_logits_f32,
                    token_offset,
                    arguments->expert_count,
                    sizeof(float));
        }
        chunk_arguments.output_bf16 = SparkGlm52B12xOffsetElements(
            arguments->output_bf16,
            token_offset,
            arguments->hidden_dimension,
            sizeof(uint16_t));
        status = SparkGlm52B12xLaunchSelectedBucket(
            state,
            &chunk_arguments,
            bucket,
            bucket_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        token_offset += bucket->token_upper_bound;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeLaunch(
    void *state_pointer,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    SparkFlashInferB12xCompiledMoeState *state;
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket;
    uint32_t bucket_index;

    if (state_pointer == 0 || arguments == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    state = (SparkFlashInferB12xCompiledMoeState *)state_pointer;
    if (arguments->token_count > state->recipe.maximum_token_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    bucket = SparkGlm52B12xSelectBucket(
        state,
        arguments->token_count,
        &bucket_index);
    if (bucket == 0)
    {
        return SparkGlm52B12xLaunchChunked(state, arguments);
    }
    return SparkGlm52B12xLaunchSelectedBucket(
        state,
        arguments,
        bucket,
        bucket_index);
}

extern "C" void SparkFlashInferB12xCompiledMoeDestroy(void *state_pointer)
{
    SparkFlashInferB12xCompiledMoeState *state;
    uint32_t bucket_index;

    if (state_pointer == 0)
    {
        return;
    }

    state = (SparkFlashInferB12xCompiledMoeState *)state_pointer;
    if (state->workspaces != 0)
    {
        for (bucket_index = 0u; bucket_index < state->bucket_count; ++bucket_index)
        {
            SparkGlm52B12xReleaseGeneratedWorkspace(&state->workspaces[bucket_index]);
        }
        free(state->workspaces);
    }
    free(state);
}
