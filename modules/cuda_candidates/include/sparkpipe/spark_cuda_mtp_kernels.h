#ifndef SPARKPIPE_SPARK_CUDA_MTP_KERNELS_H
#define SPARKPIPE_SPARK_CUDA_MTP_KERNELS_H

#include <stdint.h>

#include "sparkpipe/spark_mtp_draft.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_SENTINEL 0x535043554D545044ull
#define SPARKPIPE_CUDA_MTP_MAX_DRAFT_TOKENS SPARKPIPE_MTP_DRAFT_MAX_DRAFT_TOKENS
#define SPARKPIPE_CUDA_MTP_DRAFT_VERIFY_EVENT_COUNTERS 5u
#define SPARKPIPE_CUDA_MTP_CANCELLED_TOKEN_ID 0xffffffffu

typedef enum SparkCudaMtpDraftVerifyEventCounter
{
    SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_ACCEPTED = 0,
    SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_REJECTED = 1,
    SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_COMMITTED = 2,
    SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_ROLLBACK = 3,
    SPARK_CUDA_MTP_DRAFT_VERIFY_EVENT_CANCELLED = 4
} SparkCudaMtpDraftVerifyEventCounter;

typedef struct SparkCudaMtpDraftVerifyRequest
{
    uint32_t batch_count;
    uint32_t draft_token_count;
    uint32_t hidden_size;
    uint32_t vocabulary_size;
    SparkFp4StorageFormatKind draft_weight_format;
    SparkMtpDraftCudaKernelKind draft_kernel;
    uint32_t draft_group_size;
    uint32_t reserved0;
    uint64_t sentinel;
} SparkCudaMtpDraftVerifyRequest;

typedef struct SparkCudaMtpDraftVerifyReport
{
    uint64_t draft_logit_count;
    uint64_t draft_logit_checksum;
    uint32_t batch_count;
    uint32_t draft_token_count;
    uint32_t hidden_size;
    uint32_t vocabulary_size;
    uint32_t clear_kernel_count;
    uint32_t draft_logits_kernel_count;
    uint32_t draft_argmax_kernel_count;
    uint32_t verify_kernel_count;
    uint32_t commit_kernel_count;
    uint32_t event_counter_count;
    uint32_t accepted_token_count;
    uint32_t rejected_token_count;
    uint32_t committed_token_count;
    uint32_t rollback_token_count;
    uint32_t cancelled_token_count;
    uint32_t route_mxfp4_count;
    uint32_t route_nvfp4_rejected_count;
    uint32_t hot_path_allocation_count;
    uint32_t unsupported_shape_count;
    uint32_t sentinel_violation_count;
} SparkCudaMtpDraftVerifyReport;

void SparkCudaMtpDraftVerifyRequestReset(SparkCudaMtpDraftVerifyRequest *request);
SparkStatus SparkValidateCudaMtpDraftVerifyRequest(const SparkCudaMtpDraftVerifyRequest *request);
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
                                       SparkCudaMtpDraftVerifyReport *report);

#ifdef __cplusplus
}
#endif

#endif
