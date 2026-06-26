#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_sampler_kernels.h"

#define SPARK_CUDA_SAMPLER_THREADS 1024u
#define SPARK_CUDA_SAMPLER_NEG_INF -3.4028234663852886e+38f

static uint64_t SparkCudaSamplerLogitCountHost(const SparkCudaSamplerRequest *request)
{
    return (uint64_t)request->row_count * (uint64_t)request->vocab_size;
}

static void SparkCudaSamplerFillReportShape(const SparkCudaSamplerRequest *request, SparkCudaSamplerReport *report)
{
    report->logit_count = SparkCudaSamplerLogitCountHost(request);
    report->candidate_count = (uint64_t)request->row_count * (uint64_t)request->top_k;
    report->error_counter_count = SPARKPIPE_CUDA_SAMPLER_ERROR_COUNTERS;
}

static __device__ bool SparkCudaSamplerCandidateBetter(float left_score, uint32_t left_id, float right_score, uint32_t right_id)
{
    if (left_score > right_score)
    {
        return true;
    }
    if (left_score == right_score && left_id < right_id)
    {
        return true;
    }
    return false;
}

static __device__ bool SparkCudaSamplerCandidateAfterPrevious(float score, uint32_t token_id, float previous_score, uint32_t previous_id, uint32_t rank)
{
    if (rank == 0u)
    {
        return true;
    }
    if (score < previous_score)
    {
        return true;
    }
    if (score == previous_score && token_id > previous_id)
    {
        return true;
    }
    return false;
}

static __device__ void SparkCudaSamplerInsertTop2(float score, uint32_t token_id, float *best_score, uint32_t *best_id, float *second_score, uint32_t *second_id)
{
    if (SparkCudaSamplerCandidateBetter(score, token_id, *best_score, *best_id))
    {
        *second_score = *best_score;
        *second_id = *best_id;
        *best_score = score;
        *best_id = token_id;
    }
    else if (SparkCudaSamplerCandidateBetter(score, token_id, *second_score, *second_id))
    {
        *second_score = score;
        *second_id = token_id;
    }
}

static __device__ void SparkCudaSamplerWarpReduceBest(float *score, uint32_t *token_id)
{
    float other_score;
    uint32_t other_id;
    uint32_t offset;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        other_score = __shfl_down_sync(0xffffffffu, *score, offset);
        other_id = __shfl_down_sync(0xffffffffu, *token_id, offset);
        if (SparkCudaSamplerCandidateBetter(other_score, other_id, *score, *token_id))
        {
            *score = other_score;
            *token_id = other_id;
        }
    }
}

static __device__ uint32_t SparkCudaSamplerWarpReduceSumU32(uint32_t value)
{
    uint32_t offset;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

static __device__ void SparkCudaSamplerWarpReduceTop2(float *best_score, uint32_t *best_id, float *second_score, uint32_t *second_id)
{
    float other_best_score;
    float other_second_score;
    uint32_t other_best_id;
    uint32_t other_second_id;
    uint32_t offset;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        other_best_score = __shfl_down_sync(0xffffffffu, *best_score, offset);
        other_best_id = __shfl_down_sync(0xffffffffu, *best_id, offset);
        other_second_score = __shfl_down_sync(0xffffffffu, *second_score, offset);
        other_second_id = __shfl_down_sync(0xffffffffu, *second_id, offset);
        if (other_best_id != UINT32_MAX)
        {
            SparkCudaSamplerInsertTop2(other_best_score, other_best_id, best_score, best_id, second_score, second_id);
        }
        if (other_second_id != UINT32_MAX)
        {
            SparkCudaSamplerInsertTop2(other_second_score, other_second_id, best_score, best_id, second_score, second_id);
        }
    }
}

static __global__ void SparkCudaSamplerClearKernel(SparkCudaSamplerRequest request, uint32_t *error_counters)
{
    if (blockIdx.x != 0u || threadIdx.x >= SPARKPIPE_CUDA_SAMPLER_ERROR_COUNTERS)
    {
        return;
    }
    error_counters[threadIdx.x] = 0u;
    if (threadIdx.x == 0u && request.sentinel != SPARKPIPE_CUDA_SAMPLER_SENTINEL)
    {
        error_counters[SPARK_CUDA_SAMPLER_ERROR_SENTINEL] = 1u;
    }
}

static __global__ void SparkCudaSamplerDominantMinPKernel(SparkCudaSamplerRequest request, const float *logits, const float *uniforms, uint32_t *output_token_ids, float *output_probabilities, uint32_t *error_counters)
{
    __shared__ float warp_best_scores[SPARK_CUDA_SAMPLER_THREADS / 32u];
    __shared__ float warp_second_scores[SPARK_CUDA_SAMPLER_THREADS / 32u];
    __shared__ uint32_t warp_best_ids[SPARK_CUDA_SAMPLER_THREADS / 32u];
    __shared__ uint32_t warp_second_ids[SPARK_CUDA_SAMPLER_THREADS / 32u];
    __shared__ uint32_t warp_nonfinite_counts[SPARK_CUDA_SAMPLER_THREADS / 32u];
    uint32_t row_index;
    uint32_t token_id;
    uint32_t nonfinite_count;
    uint32_t best_id;
    uint32_t second_id;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;
    float score;
    float best_score;
    float second_score;
    float second_probability;
    float uniform_value;

    row_index = blockIdx.x;
    if (row_index >= request.row_count)
    {
        return;
    }
    lane_index = threadIdx.x & 31u;
    warp_index = threadIdx.x >> 5u;
    warp_count = blockDim.x >> 5u;
    best_score = SPARK_CUDA_SAMPLER_NEG_INF;
    second_score = SPARK_CUDA_SAMPLER_NEG_INF;
    best_id = UINT32_MAX;
    second_id = UINT32_MAX;
    nonfinite_count = 0u;
    for (token_id = threadIdx.x; token_id < request.vocab_size; token_id += blockDim.x)
    {
        score = logits[((uint64_t)row_index * (uint64_t)request.vocab_size) + token_id];
        if (!isfinite(score))
        {
            nonfinite_count += 1u;
            continue;
        }
        SparkCudaSamplerInsertTop2(score, token_id, &best_score, &best_id, &second_score, &second_id);
    }
    SparkCudaSamplerWarpReduceTop2(&best_score, &best_id, &second_score, &second_id);
    nonfinite_count = SparkCudaSamplerWarpReduceSumU32(nonfinite_count);
    if (lane_index == 0u)
    {
        warp_best_scores[warp_index] = best_score;
        warp_second_scores[warp_index] = second_score;
        warp_best_ids[warp_index] = best_id;
        warp_second_ids[warp_index] = second_id;
        warp_nonfinite_counts[warp_index] = nonfinite_count;
    }
    __syncthreads();
    if (warp_index == 0u)
    {
        best_score = lane_index < warp_count ? warp_best_scores[lane_index] : SPARK_CUDA_SAMPLER_NEG_INF;
        second_score = lane_index < warp_count ? warp_second_scores[lane_index] : SPARK_CUDA_SAMPLER_NEG_INF;
        best_id = lane_index < warp_count ? warp_best_ids[lane_index] : UINT32_MAX;
        second_id = lane_index < warp_count ? warp_second_ids[lane_index] : UINT32_MAX;
        nonfinite_count = lane_index < warp_count ? warp_nonfinite_counts[lane_index] : 0u;
        SparkCudaSamplerWarpReduceTop2(&best_score, &best_id, &second_score, &second_id);
        nonfinite_count = SparkCudaSamplerWarpReduceSumU32(nonfinite_count);
    }
    if (threadIdx.x != 0u)
    {
        return;
    }
    if (nonfinite_count != 0u)
    {
        atomicAdd(&error_counters[SPARK_CUDA_SAMPLER_ERROR_NONFINITE_LOGIT], nonfinite_count);
    }
    if (best_id == UINT32_MAX || second_id == UINT32_MAX)
    {
        output_token_ids[row_index] = UINT32_MAX;
        output_probabilities[row_index] = 0.0f;
        return;
    }
    second_probability = expf((second_score - best_score) / request.temperature);
    if (second_probability >= request.min_p)
    {
        output_token_ids[row_index] = UINT32_MAX;
        output_probabilities[row_index] = 0.0f;
        return;
    }
    uniform_value = uniforms[row_index];
    if (!isfinite(uniform_value) || uniform_value < 0.0f || uniform_value >= 1.0f)
    {
        atomicAdd(&error_counters[SPARK_CUDA_SAMPLER_ERROR_BAD_UNIFORM], 1u);
    }
    output_token_ids[row_index] = best_id;
    output_probabilities[row_index] = 1.0f;
}

static __global__ void SparkCudaSamplerKernel(SparkCudaSamplerRequest request, const float *logits, const float *uniforms, uint32_t *output_token_ids, float *output_probabilities, uint32_t *error_counters, uint32_t dominant_prefilter)
{
    __shared__ float warp_scores[SPARK_CUDA_SAMPLER_THREADS / 32u];
    __shared__ uint32_t warp_ids[SPARK_CUDA_SAMPLER_THREADS / 32u];
    __shared__ float top_scores[SPARKPIPE_CUDA_SAMPLER_MAX_TOP_K];
    __shared__ uint32_t top_ids[SPARKPIPE_CUDA_SAMPLER_MAX_TOP_K];
    uint32_t row_index;
    uint32_t rank;
    uint32_t token_id;
    uint32_t selected_count;
    uint32_t candidate_index;
    uint32_t selected_token;
    float previous_score;
    uint32_t previous_id;
    float candidate_score;
    float best_score;
    uint32_t best_id;
    uint32_t stride;
    float max_score;
    float probability;
    float full_probability;
    float total_probability;
    float cumulative_probability;
    float threshold;
    float uniform_value;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;

    row_index = blockIdx.x;
    if (row_index >= request.row_count)
    {
        return;
    }
    if (dominant_prefilter != 0u && output_token_ids[row_index] != UINT32_MAX)
    {
        return;
    }
    lane_index = threadIdx.x & 31u;
    warp_index = threadIdx.x >> 5u;
    warp_count = blockDim.x >> 5u;
    previous_score = 0.0f;
    previous_id = 0u;
    for (rank = 0u; rank < request.top_k; ++rank)
    {
        best_score = SPARK_CUDA_SAMPLER_NEG_INF;
        best_id = UINT32_MAX;
        stride = blockDim.x;
        for (token_id = threadIdx.x; token_id < request.vocab_size; token_id += stride)
        {
            candidate_score = logits[((uint64_t)row_index * (uint64_t)request.vocab_size) + token_id];
            if (!isfinite(candidate_score))
            {
                if (rank == 0u && dominant_prefilter == 0u)
                {
                    atomicAdd(&error_counters[SPARK_CUDA_SAMPLER_ERROR_NONFINITE_LOGIT], 1u);
                }
                continue;
            }
            if (SparkCudaSamplerCandidateAfterPrevious(candidate_score, token_id, previous_score, previous_id, rank) && SparkCudaSamplerCandidateBetter(candidate_score, token_id, best_score, best_id))
            {
                best_score = candidate_score;
                best_id = token_id;
            }
        }
        SparkCudaSamplerWarpReduceBest(&best_score, &best_id);
        if (lane_index == 0u)
        {
            warp_scores[warp_index] = best_score;
            warp_ids[warp_index] = best_id;
        }
        __syncthreads();
        if (warp_index == 0u)
        {
            best_score = lane_index < warp_count ? warp_scores[lane_index] : SPARK_CUDA_SAMPLER_NEG_INF;
            best_id = lane_index < warp_count ? warp_ids[lane_index] : UINT32_MAX;
            SparkCudaSamplerWarpReduceBest(&best_score, &best_id);
            if (lane_index == 0u)
            {
                top_scores[rank] = best_score;
                top_ids[rank] = best_id;
            }
        }
        __syncthreads();
        previous_score = top_scores[rank];
        previous_id = top_ids[rank];
    }
    if (threadIdx.x != 0u)
    {
        return;
    }
    if (top_ids[0] == UINT32_MAX)
    {
        output_token_ids[row_index] = 0u;
        output_probabilities[row_index] = 0.0f;
        atomicAdd(&error_counters[SPARK_CUDA_SAMPLER_ERROR_EMPTY_FILTER], 1u);
        return;
    }
    max_score = top_scores[0];
    full_probability = 0.0f;
    total_probability = 0.0f;
    selected_count = 0u;
    for (candidate_index = 0u; candidate_index < request.top_k; ++candidate_index)
    {
        if (top_ids[candidate_index] == UINT32_MAX)
        {
            continue;
        }
        probability = expf((top_scores[candidate_index] - max_score) / request.temperature);
        if (request.min_p == 0.0f || probability >= request.min_p)
        {
            full_probability += probability;
        }
    }
    cumulative_probability = 0.0f;
    for (candidate_index = 0u; candidate_index < request.top_k; ++candidate_index)
    {
        if (top_ids[candidate_index] == UINT32_MAX)
        {
            continue;
        }
        probability = expf((top_scores[candidate_index] - max_score) / request.temperature);
        if (request.min_p > 0.0f && probability < request.min_p)
        {
            continue;
        }
        cumulative_probability += probability;
        if (request.top_p < 1.0f && selected_count > 0u && cumulative_probability > (request.top_p * full_probability))
        {
            break;
        }
        selected_count += 1u;
        total_probability = cumulative_probability;
    }
    if (selected_count == 0u || total_probability <= 0.0f)
    {
        output_token_ids[row_index] = 0u;
        output_probabilities[row_index] = 0.0f;
        atomicAdd(&error_counters[SPARK_CUDA_SAMPLER_ERROR_EMPTY_FILTER], 1u);
        return;
    }
    uniform_value = uniforms[row_index];
    if (!isfinite(uniform_value) || uniform_value < 0.0f || uniform_value >= 1.0f)
    {
        uniform_value = 0.0f;
        atomicAdd(&error_counters[SPARK_CUDA_SAMPLER_ERROR_BAD_UNIFORM], 1u);
    }
    threshold = uniform_value * total_probability;
    cumulative_probability = 0.0f;
    selected_token = top_ids[0];
    probability = 0.0f;
    for (candidate_index = 0u; candidate_index < selected_count; ++candidate_index)
    {
        probability = expf((top_scores[candidate_index] - max_score) / request.temperature);
        cumulative_probability += probability;
        selected_token = top_ids[candidate_index];
        if (threshold <= cumulative_probability)
        {
            break;
        }
    }
    output_token_ids[row_index] = selected_token;
    output_probabilities[row_index] = probability / total_probability;
}

extern "C" SparkStatus SparkRunCudaSamplerTopKTopPMinP(const SparkCudaSamplerRequest *request, const float *device_logits, const float *device_uniforms, uint32_t *device_output_token_ids, float *device_output_probabilities, uint32_t *device_error_counters, SparkCudaSamplerReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint32_t dominant_prefilter;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaSamplerRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_logits == 0 || device_uniforms == 0 || device_output_token_ids == 0 || device_output_probabilities == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaSamplerFillReportShape(request, report);
    dominant_prefilter = request->min_p > 0.0f ? 1u : 0u;
    SparkCudaSamplerClearKernel<<<1u, SPARKPIPE_CUDA_SAMPLER_ERROR_COUNTERS>>>(*request, device_error_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (dominant_prefilter != 0u)
    {
        SparkCudaSamplerDominantMinPKernel<<<request->row_count, SPARK_CUDA_SAMPLER_THREADS>>>(*request, device_logits, device_uniforms, device_output_token_ids, device_output_probabilities, device_error_counters);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }
    SparkCudaSamplerKernel<<<request->row_count, SPARK_CUDA_SAMPLER_THREADS>>>(*request, device_logits, device_uniforms, device_output_token_ids, device_output_probabilities, device_error_counters, dominant_prefilter);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->clear_kernel_count = 1u;
    report->sampler_kernel_count = dominant_prefilter != 0u ? 2u : 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}
