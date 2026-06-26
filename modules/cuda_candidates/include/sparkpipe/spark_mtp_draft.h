#ifndef SPARKPIPE_SPARK_MTP_DRAFT_H
#define SPARKPIPE_SPARK_MTP_DRAFT_H

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_fp4_format.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_MTP_DRAFT_STAGEPACK_METADATA_SENTINEL 0x53504D5450444D45ull
#define SPARKPIPE_MTP_DRAFT_MAX_DRAFT_TOKENS 8u
#define SPARKPIPE_MTP_DRAFT_MIN_METADATA_VERSION 1u
#define SPARKPIPE_MTP_DRAFT_CURRENT_METADATA_VERSION 1u
#define SPARKPIPE_MTP_DRAFT_ROUTE_BLOCKER_BYTES 160u

typedef enum SparkMtpDraftMetadataSourceKind
{
    SPARK_MTP_DRAFT_METADATA_SOURCE_UNKNOWN = 0,
    SPARK_MTP_DRAFT_METADATA_SOURCE_STAGEPACK_EXPERT = 1,
    SPARK_MTP_DRAFT_METADATA_SOURCE_MODEL_LEVEL_QUANT_NAME = 2
} SparkMtpDraftMetadataSourceKind;

typedef enum SparkMtpDraftCudaKernelKind
{
    SPARK_MTP_DRAFT_CUDA_KERNEL_UNKNOWN = 0,
    SPARK_MTP_DRAFT_CUDA_KERNEL_MXFP4_E2M1_E8M0 = 1,
    SPARK_MTP_DRAFT_CUDA_KERNEL_NVFP4_E2M1_E4M3 = 2
} SparkMtpDraftCudaKernelKind;

typedef struct SparkMtpDraftStagePackMetadata
{
    uint64_t sentinel;
    uint32_t metadata_version;
    uint32_t stage_id;
    uint32_t expert_id;
    uint32_t hidden_size;
    uint32_t intermediate_rows;
    uint32_t vocabulary_size;
    uint32_t draft_token_count;
    SparkMtpDraftMetadataSourceKind metadata_source;
    SparkFp4StorageFormatKind weight_format;
    SparkFp4ValueEncodingKind value_encoding;
    SparkFp4ScaleEncodingKind scale_encoding;
    uint32_t values_per_scale_block;
    uint64_t gate_payload_bytes;
    uint64_t gate_scale_bytes;
    uint64_t up_payload_bytes;
    uint64_t up_scale_bytes;
    uint64_t down_payload_bytes;
    uint64_t down_scale_bytes;
    uint64_t metadata_checksum;
} SparkMtpDraftStagePackMetadata;

typedef struct SparkMtpDraftRoute
{
    bool ready;
    uint32_t stage_id;
    uint32_t expert_id;
    uint32_t hidden_size;
    uint32_t intermediate_rows;
    uint32_t vocabulary_size;
    uint32_t draft_token_count;
    SparkMtpDraftMetadataSourceKind metadata_source;
    SparkFp4StorageFormatKind weight_format;
    SparkMtpDraftCudaKernelKind cuda_kernel;
    uint32_t stagepack_metadata_count;
    uint32_t model_level_quant_name_count;
    uint32_t model_default_format_ignored_count;
    uint32_t ambiguous_metadata_count;
    uint32_t mxfp4_route_count;
    uint32_t nvfp4_rejected_count;
    uint32_t blocker_count;
    char first_blocker[SPARKPIPE_MTP_DRAFT_ROUTE_BLOCKER_BYTES];
    uint64_t metadata_checksum;
    uint64_t route_checksum;
} SparkMtpDraftRoute;

const char *SparkMtpDraftMetadataSourceKindToString(SparkMtpDraftMetadataSourceKind metadata_source);
const char *SparkMtpDraftCudaKernelKindToString(SparkMtpDraftCudaKernelKind cuda_kernel);
void SparkMtpDraftStagePackMetadataReset(SparkMtpDraftStagePackMetadata *metadata);
void SparkMtpDraftRouteReset(SparkMtpDraftRoute *route);
uint64_t SparkComputeMtpDraftStagePackMetadataChecksum(const SparkMtpDraftStagePackMetadata *metadata);
uint64_t SparkComputeMtpDraftRouteChecksum(const SparkMtpDraftRoute *route);
SparkStatus SparkFinalizeMtpDraftStagePackMetadata(SparkMtpDraftStagePackMetadata *metadata);
SparkStatus SparkValidateMtpDraftStagePackMetadata(const SparkMtpDraftStagePackMetadata *metadata);
SparkStatus SparkRouteMtpDraftStagePackMetadata(const SparkMtpDraftStagePackMetadata *metadata, SparkFp4StorageFormatKind model_default_format, SparkMtpDraftRoute *route);

#ifdef __cplusplus
}
#endif

#endif
