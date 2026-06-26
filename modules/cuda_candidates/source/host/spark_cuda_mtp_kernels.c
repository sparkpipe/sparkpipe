#include "sparkpipe/spark_cuda_mtp_kernels.h"

#include <string.h>

static SparkMtpDraftCudaKernelKind SparkCudaMtpExpectedKernel(SparkFp4StorageFormatKind draft_weight_format)
{
    if (draft_weight_format == SPARK_FP4_STORAGE_FORMAT_MXFP4_E2M1_E8M0)
    {
        return SPARK_MTP_DRAFT_CUDA_KERNEL_MXFP4_E2M1_E8M0;
    }
    return SPARK_MTP_DRAFT_CUDA_KERNEL_UNKNOWN;
}

static void SparkCudaMtpDraftVerifyReportReset(SparkCudaMtpDraftVerifyReport *report)
{
    if (report == 0)
    {
        return;
    }
    memset(report, 0, sizeof(*report));
}

static void SparkCudaMtpFillDraftVerifyReportShape(const SparkCudaMtpDraftVerifyRequest *request, SparkCudaMtpDraftVerifyReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->batch_count = request->batch_count;
    report->draft_token_count = request->draft_token_count;
    report->hidden_size = request->hidden_size;
    report->vocabulary_size = request->vocabulary_size;
    report->draft_logit_count = (uint64_t)request->batch_count * (uint64_t)request->draft_token_count * (uint64_t)request->vocabulary_size;
    report->event_counter_count = SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_EVENT_COUNTERS;
    if (request->draft_weight_format == SPARK_FP4_STORAGE_FORMAT_MXFP4_E2M1_E8M0)
    {
        report->route_mxfp4_count = 1u;
    }
    if (request->draft_weight_format == SPARK_FP4_STORAGE_FORMAT_NVFP4_E2M1_E4M3)
    {
        report->route_nvfp4_rejected_count = 1u;
    }
}

void SparkCudaMtpDraftVerifyRequestReset(SparkCudaMtpDraftVerifyRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->draft_group_size = SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK;
    request->sentinel = SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_SENTINEL;
}

SparkStatus SparkValidateCudaMtpDraftVerifyRequest(const SparkCudaMtpDraftVerifyRequest *request)
{
    SparkMtpDraftCudaKernelKind expected_kernel;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->batch_count == 0u || request->draft_token_count == 0u || request->hidden_size == 0u || request->vocabulary_size == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->draft_token_count > SPARKPIPE_CUDA_MTP_MAX_DRAFT_TOKENS)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (request->draft_group_size != SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK || (request->hidden_size % SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK) != 0u || (request->hidden_size & 1u) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    expected_kernel = SparkCudaMtpExpectedKernel(request->draft_weight_format);
    if (expected_kernel == SPARK_MTP_DRAFT_CUDA_KERNEL_UNKNOWN || request->draft_kernel != expected_kernel)
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaMtpDraftVerify(const SparkCudaMtpDraftVerifyRequest *request,
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

    SparkCudaMtpDraftVerifyReportReset(report);
    status = SparkValidateCudaMtpDraftVerifyRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_hidden_bf16 == 0 || device_weight_payload_u8 == 0 || device_weight_scale_u8 == 0 || device_target_token_ids == 0 || device_draft_logits == 0 || device_draft_token_ids == 0 || device_accept_mask == 0 || device_committed_token_ids == 0 || device_event_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMtpFillDraftVerifyReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
