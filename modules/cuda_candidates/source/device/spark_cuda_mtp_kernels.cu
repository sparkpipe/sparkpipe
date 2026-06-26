#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_mtp_kernels.h"

#define SPARK_CUDA_MTP_LOGIT_THREADS 256u
#define SPARK_CUDA_MTP_ARGMAX_THREADS 256u

static __device__ float SparkCudaMtpBf16ToFloat(uint16_t value)
{
    uint32_t bits;

    bits = ((uint32_t)value) << 16u;
    return __uint_as_float(bits);
}

static __device__ float SparkCudaMtpDecodeE2m1(uint8_t nibble)
{
    float value;

    switch (nibble & 7u)
    {
        case 1u:
        {
            value = 0.5f;
            break;
        }
        case 2u:
        {
            value = 1.0f;
            break;
        }
        case 3u:
        {
            value = 1.5f;
            break;
        }
        case 4u:
        {
            value = 2.0f;
            break;
        }
        case 5u:
        {
            value = 3.0f;
            break;
        }
        case 6u:
        {
            value = 4.0f;
            break;
        }
        case 7u:
        {
            value = 6.0f;
            break;
        }
        default:
        {
            value = 0.0f;
            break;
        }
    }
    return (nibble & 8u) != 0u ? -value : value;
}

static __device__ float SparkCudaMtpDecodeE8m0(uint8_t value)
{
    if (value == 0u)
    {
        return 0.0f;
    }
    return ldexpf(1.0f, (int)value - 127);
}

static __device__ float SparkCudaMtpDecodeFp4Mxfp4Weight(const SparkCudaMtpDraftVerifyRequest request, const uint8_t *weight_payload, const uint8_t *weight_scale, uint32_t vocabulary_index, uint32_t hidden_index)
{
    uint64_t packed_index;
    uint64_t scale_index;
    uint8_t packed_value;
    uint8_t nibble;
    float scale_value;

    packed_index = ((uint64_t)vocabulary_index * ((uint64_t)request.hidden_size >> 1u)) + ((uint64_t)hidden_index >> 1u);
    scale_index = ((uint64_t)vocabulary_index * ((uint64_t)request.hidden_size / (uint64_t)SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK)) + ((uint64_t)hidden_index / (uint64_t)SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK);
    packed_value = weight_payload[packed_index];
    nibble = (hidden_index & 1u) == 0u ? (packed_value & 0x0fu) : (packed_value >> 4u);
    scale_value = SparkCudaMtpDecodeE8m0(weight_scale[scale_index]);
    return SparkCudaMtpDecodeE2m1(nibble) * scale_value;
}

static __global__ void SparkCudaMtpDraftVerifyClearEventCountersKernel(uint32_t *event_counters)
{
    if (blockIdx.x != 0u || threadIdx.x >= SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_EVENT_COUNTERS)
    {
        return;
    }
    event_counters[threadIdx.x] = 0u;
}

static __global__ void SparkCudaMtpDraftLogitsKernel(SparkCudaMtpDraftVerifyRequest request, const uint16_t *hidden_bf16, const uint8_t *weight_payload, const uint8_t *weight_scale, float *draft_logits)
{
    uint64_t logit_index;
    uint64_t total_logits;

    total_logits = (uint64_t)request.batch_count * (uint64_t)request.draft_token_count * (uint64_t)request.vocabulary_size;
    logit_index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
    while (logit_index < total_logits)
    {
        uint32_t vocabulary_index;
        uint32_t draft_index;
        uint32_t batch_index;
        uint32_t hidden_index;
        float sum;

        vocabulary_index = (uint32_t)(logit_index % request.vocabulary_size);
        draft_index = (uint32_t)((logit_index / request.vocabulary_size) % request.draft_token_count);
        batch_index = (uint32_t)(logit_index / ((uint64_t)request.vocabulary_size * (uint64_t)request.draft_token_count));
        sum = 0.0f;
        for (hidden_index = 0u; hidden_index < request.hidden_size; ++hidden_index)
        {
            uint64_t hidden_offset;
            float activation_value;
            float weight_value;

            hidden_offset = (((uint64_t)batch_index * (uint64_t)request.draft_token_count + (uint64_t)draft_index) * (uint64_t)request.hidden_size) + hidden_index;
            activation_value = SparkCudaMtpBf16ToFloat(hidden_bf16[hidden_offset]);
            weight_value = SparkCudaMtpDecodeFp4Mxfp4Weight(request, weight_payload, weight_scale, vocabulary_index, hidden_index);
            sum += activation_value * weight_value;
        }
        draft_logits[logit_index] = sum;
        logit_index += (uint64_t)gridDim.x * blockDim.x;
    }
}

static __global__ void SparkCudaMtpDraftArgmaxKernel(SparkCudaMtpDraftVerifyRequest request, const float *draft_logits, uint32_t *draft_token_ids)
{
    __shared__ float shared_scores[SPARK_CUDA_MTP_ARGMAX_THREADS];
    __shared__ uint32_t shared_tokens[SPARK_CUDA_MTP_ARGMAX_THREADS];
    uint32_t row_index;
    uint32_t vocabulary_index;
    float best_score;
    uint32_t best_token;
    uint32_t stride;

    row_index = blockIdx.x;
    if (row_index >= request.batch_count * request.draft_token_count)
    {
        return;
    }
    best_score = -CUDART_INF_F;
    best_token = 0u;
    for (vocabulary_index = threadIdx.x; vocabulary_index < request.vocabulary_size; vocabulary_index += blockDim.x)
    {
        float score;

        score = draft_logits[((uint64_t)row_index * request.vocabulary_size) + vocabulary_index];
        if (score > best_score || (score == best_score && vocabulary_index < best_token))
        {
            best_score = score;
            best_token = vocabulary_index;
        }
    }
    shared_scores[threadIdx.x] = best_score;
    shared_tokens[threadIdx.x] = best_token;
    __syncthreads();
    for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            float other_score;
            uint32_t other_token;

            other_score = shared_scores[threadIdx.x + stride];
            other_token = shared_tokens[threadIdx.x + stride];
            if (other_score > shared_scores[threadIdx.x] || (other_score == shared_scores[threadIdx.x] && other_token < shared_tokens[threadIdx.x]))
            {
                shared_scores[threadIdx.x] = other_score;
                shared_tokens[threadIdx.x] = other_token;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        draft_token_ids[row_index] = shared_tokens[0];
    }
}

static __global__ void SparkCudaMtpVerifyCommitKernel(SparkCudaMtpDraftVerifyRequest request, const uint32_t *target_token_ids, const uint32_t *draft_token_ids, uint32_t *accept_mask, uint32_t *committed_token_ids, uint32_t *event_counters)
{
    uint32_t batch_index;
    uint32_t accepting;
    uint32_t draft_index;

    batch_index = blockIdx.x;
    if (threadIdx.x != 0u || batch_index >= request.batch_count)
    {
        return;
    }
    accepting = 1u;
    for (draft_index = 0u; draft_index < request.draft_token_count; ++draft_index)
    {
        uint32_t row_index;
        uint32_t accepted;

        row_index = (batch_index * request.draft_token_count) + draft_index;
        accepted = accepting != 0u && draft_token_ids[row_index] == target_token_ids[row_index] ? 1u : 0u;
        accept_mask[row_index] = accepted;
        if (accepted != 0u)
        {
            committed_token_ids[row_index] = draft_token_ids[row_index];
            atomicAdd(&event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_ACCEPTED], 1u);
            atomicAdd(&event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_COMMITTED], 1u);
        }
        else if (accepting != 0u)
        {
            uint32_t rejected_suffix_count;
            uint32_t cancelled_suffix_count;

            rejected_suffix_count = request.draft_token_count - draft_index;
            cancelled_suffix_count = rejected_suffix_count - 1u;
            committed_token_ids[row_index] = target_token_ids[row_index];
            atomicAdd(&event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_REJECTED], rejected_suffix_count);
            atomicAdd(&event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_COMMITTED], 1u);
            atomicAdd(&event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_ROLLBACK], rejected_suffix_count);
            atomicAdd(&event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_CANCELLED], cancelled_suffix_count);
            accepting = 0u;
        }
        else
        {
            committed_token_ids[row_index] = SPARKPIPE_CUDA_MTP_CANCELLED_TOKEN_ID;
        }
    }
}

static void SparkCudaMtpFillDraftVerifyReportShapeHost(const SparkCudaMtpDraftVerifyRequest *request, SparkCudaMtpDraftVerifyReport *report)
{
    memset(report, 0, sizeof(*report));
    report->batch_count = request->batch_count;
    report->draft_token_count = request->draft_token_count;
    report->hidden_size = request->hidden_size;
    report->vocabulary_size = request->vocabulary_size;
    report->draft_logit_count = (uint64_t)request->batch_count * (uint64_t)request->draft_token_count * (uint64_t)request->vocabulary_size;
    report->event_counter_count = SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_EVENT_COUNTERS;
    report->clear_kernel_count = 1u;
    report->draft_logits_kernel_count = 1u;
    report->draft_argmax_kernel_count = 1u;
    report->verify_kernel_count = 1u;
    report->commit_kernel_count = 1u;
    report->route_mxfp4_count = 1u;
    report->hot_path_allocation_count = 0u;
}

extern "C" SparkStatus SparkRunCudaMtpDraftVerify(const SparkCudaMtpDraftVerifyRequest *request,
                                                  const void *device_hidden_bf16,
                                                  const void *device_weight_payload_u8,
                                                  const void *device_weight_scale_u8,
                                                  const uint32_t *device_target_token_ids,
                                                  float *device_draft_logits,
                                                  uint32_t *device_draft_token_ids,
                                                  uint32_t *device_accept_mask,
                                                  uint32_t *device_committed_token_ids,
                                                  uint32_t *device_event_counters,
                                                  SparkCudaMtpDraftVerifyReport *report)
{
    SparkStatus status;
    cudaError_t cuda_status;
    uint64_t logit_count;
    uint64_t logit_grid_count_64;
    uint32_t logit_grid_count;
    uint32_t host_event_counters[SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_EVENT_COUNTERS];
    SparkCudaMtpDraftVerifyReport host_report;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMtpDraftVerifyRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_hidden_bf16 == 0 || device_weight_payload_u8 == 0 || device_weight_scale_u8 == 0 || device_target_token_ids == 0 || device_draft_logits == 0 || device_draft_token_ids == 0 || device_accept_mask == 0 || device_committed_token_ids == 0 || device_event_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMtpFillDraftVerifyReportShapeHost(request, &host_report);
    logit_count = host_report.draft_logit_count;
    logit_grid_count_64 = (logit_count + SPARK_CUDA_MTP_LOGIT_THREADS - 1u) / SPARK_CUDA_MTP_LOGIT_THREADS;
    if (logit_grid_count_64 == 0u || logit_grid_count_64 > 2147483647u)
    {
        host_report.unsupported_shape_count = 1u;
        *report = host_report;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    logit_grid_count = (uint32_t)logit_grid_count_64;
    SparkCudaMtpDraftVerifyClearEventCountersKernel<<<1u, SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_EVENT_COUNTERS>>>(device_event_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status == cudaSuccess)
    {
        SparkCudaMtpDraftLogitsKernel<<<logit_grid_count, SPARK_CUDA_MTP_LOGIT_THREADS>>>(*request, (const uint16_t *)device_hidden_bf16, (const uint8_t *)device_weight_payload_u8, (const uint8_t *)device_weight_scale_u8, device_draft_logits);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkCudaMtpDraftArgmaxKernel<<<request->batch_count * request->draft_token_count, SPARK_CUDA_MTP_ARGMAX_THREADS>>>(*request, device_draft_logits, device_draft_token_ids);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkCudaMtpVerifyCommitKernel<<<request->batch_count, 1u>>>(*request, device_target_token_ids, device_draft_token_ids, device_accept_mask, device_committed_token_ids, device_event_counters);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(host_event_counters, device_event_counters, sizeof(host_event_counters), cudaMemcpyDeviceToHost);
    }
    if (cuda_status != cudaSuccess)
    {
        *report = host_report;
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    host_report.accepted_token_count = host_event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_ACCEPTED];
    host_report.rejected_token_count = host_event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_REJECTED];
    host_report.committed_token_count = host_event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_COMMITTED];
    host_report.rollback_token_count = host_event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_ROLLBACK];
    host_report.cancelled_token_count = host_event_counters[SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_CANCELLED];
    *report = host_report;
    return SPARK_STATUS_OK;
}
