#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_spec_verify_kernels.h"

static uint64_t SparkCudaSpecVerifyTokenCountHost(const SparkCudaSpecVerifyRequest *request)
{
    return (uint64_t)request->sequence_count * (uint64_t)request->draft_count;
}

static void SparkCudaSpecVerifyFillReportShape(const SparkCudaSpecVerifyRequest *request, SparkCudaSpecVerifyReport *report)
{
    report->token_count = SparkCudaSpecVerifyTokenCountHost(request);
    report->error_counter_count = SPARKPIPE_CUDA_SPEC_VERIFY_ERROR_COUNTERS;
}

static __global__ void SparkCudaSpecVerifyClearKernel(SparkCudaSpecVerifyRequest request, uint32_t *error_counters)
{
    if (blockIdx.x != 0u || threadIdx.x >= SPARKPIPE_CUDA_SPEC_VERIFY_ERROR_COUNTERS)
    {
        return;
    }
    error_counters[threadIdx.x] = 0u;
    if (threadIdx.x == 0u && request.sentinel != SPARKPIPE_CUDA_SPEC_VERIFY_SENTINEL)
    {
        error_counters[SPARK_CUDA_SPEC_VERIFY_ERROR_SENTINEL] = 1u;
    }
}

static __global__ void SparkCudaSpecVerifyKernel(SparkCudaSpecVerifyRequest request, const uint32_t *draft_token_ids, const uint32_t *target_token_ids, uint32_t *accepted_mask, uint32_t *accepted_prefix_lengths)
{
    uint32_t sequence_index;
    uint32_t draft_index;
    uint32_t accept_prefix;
    uint32_t prefix_length;
    uint64_t token_index;

    sequence_index = blockIdx.x;
    if (sequence_index >= request.sequence_count || threadIdx.x != 0u)
    {
        return;
    }
    accept_prefix = 1u;
    prefix_length = 0u;
    for (draft_index = 0u; draft_index < request.draft_count; ++draft_index)
    {
        token_index = ((uint64_t)sequence_index * (uint64_t)request.draft_count) + draft_index;
        if (accept_prefix != 0u && draft_token_ids[token_index] == target_token_ids[token_index])
        {
            accepted_mask[token_index] = 1u;
            prefix_length += 1u;
        }
        else
        {
            accepted_mask[token_index] = 0u;
            accept_prefix = 0u;
        }
    }
    accepted_prefix_lengths[sequence_index] = prefix_length;
}

extern "C" SparkStatus SparkRunCudaSpecVerify(const SparkCudaSpecVerifyRequest *request, const uint32_t *device_draft_token_ids, const uint32_t *device_target_token_ids, uint32_t *device_accepted_mask, uint32_t *device_accepted_prefix_lengths, uint32_t *device_error_counters, SparkCudaSpecVerifyReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaSpecVerifyRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_draft_token_ids == 0 || device_target_token_ids == 0 || device_accepted_mask == 0 || device_accepted_prefix_lengths == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaSpecVerifyFillReportShape(request, report);
    SparkCudaSpecVerifyClearKernel<<<1u, SPARKPIPE_CUDA_SPEC_VERIFY_ERROR_COUNTERS>>>(*request, device_error_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkCudaSpecVerifyKernel<<<request->sequence_count, 1u>>>(*request, device_draft_token_ids, device_target_token_ids, device_accepted_mask, device_accepted_prefix_lengths);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->clear_kernel_count = 1u;
    report->verify_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}
