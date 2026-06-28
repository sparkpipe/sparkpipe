#include "sparkpipe/spark_mtp_draft.h"

#include "sparkpipe/spark_common.h"

#include <stdio.h>
#include <string.h>

static bool SparkMtpDraftPayloadSizesArePresent(const SparkMtpDraftStagePackMetadata *metadata)
{
    return metadata->gate_payload_bytes != 0u &&
           metadata->gate_scale_bytes != 0u &&
           metadata->up_payload_bytes != 0u &&
           metadata->up_scale_bytes != 0u &&
           metadata->down_payload_bytes != 0u &&
           metadata->down_scale_bytes != 0u;
}

static SparkStatus SparkMtpDraftExpectedFormatContract(const SparkMtpDraftStagePackMetadata *metadata, SparkFp4FormatContract *contract)
{
    SparkStatus status;

    if (metadata == 0 || contract == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkDescribeFp4Format(metadata->weight_format, contract);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (metadata->value_encoding != contract->value_encoding || metadata->scale_encoding != contract->scale_encoding || metadata->values_per_scale_block != contract->values_per_scale_block)
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkMtpDraftCudaKernelKind SparkMtpDraftCudaKernelForFormat(SparkFp4StorageFormatKind weight_format)
{
    switch (weight_format)
    {
        case SPARK_FP4_STORAGE_FORMAT_MXFP4_E2M1_E8M0:
        {
            return SPARK_MTP_DRAFT_CUDA_KERNEL_MXFP4_E2M1_E8M0;
        }
        case SPARK_FP4_STORAGE_FORMAT_NVFP4_E2M1_E4M3:
        {
            return SPARK_MTP_DRAFT_CUDA_KERNEL_NVFP4_E2M1_E4M3;
        }
        default:
        {
            return SPARK_MTP_DRAFT_CUDA_KERNEL_UNKNOWN;
        }
    }
}

static void SparkMtpDraftRouteSetBlocker(SparkMtpDraftRoute *route, const char *blocker)
{
    if (route == 0 || blocker == 0 || route->blocker_count != 0u)
    {
        if (route != 0 && blocker != 0)
        {
            route->blocker_count += 1u;
        }
        return;
    }
    snprintf(route->first_blocker, sizeof(route->first_blocker), "%s", blocker);
    route->blocker_count = 1u;
}

const char *SparkMtpDraftMetadataSourceKindToString(SparkMtpDraftMetadataSourceKind metadata_source)
{
    switch (metadata_source)
    {
        case SPARK_MTP_DRAFT_METADATA_SOURCE_STAGEPACK_EXPERT:
        {
            return "stagepack_expert_metadata";
        }
        case SPARK_MTP_DRAFT_METADATA_SOURCE_MODEL_LEVEL_QUANT_NAME:
        {
            return "model_level_quant_name";
        }
        default:
        {
            return "unknown_mtp_draft_metadata_source";
        }
    }
}

const char *SparkMtpDraftCudaKernelKindToString(SparkMtpDraftCudaKernelKind cuda_kernel)
{
    switch (cuda_kernel)
    {
        case SPARK_MTP_DRAFT_CUDA_KERNEL_MXFP4_E2M1_E8M0:
        {
            return "mtp_draft_mxfp4_e2m1_e8m0";
        }
        case SPARK_MTP_DRAFT_CUDA_KERNEL_NVFP4_E2M1_E4M3:
        {
            return "mtp_draft_nvfp4_e2m1_e4m3";
        }
        default:
        {
            return "unknown_mtp_draft_cuda_kernel";
        }
    }
}

void SparkMtpDraftStagePackMetadataReset(SparkMtpDraftStagePackMetadata *metadata)
{
    if (metadata == 0)
    {
        return;
    }
    memset(metadata, 0, sizeof(*metadata));
    metadata->sentinel = SPARKPIPE_MTP_DRAFT_STAGEPACK_METADATA_SENTINEL;
    metadata->metadata_version = SPARKPIPE_MTP_DRAFT_CURRENT_METADATA_VERSION;
    metadata->metadata_source = SPARK_MTP_DRAFT_METADATA_SOURCE_UNKNOWN;
}

void SparkMtpDraftRouteReset(SparkMtpDraftRoute *route)
{
    if (route == 0)
    {
        return;
    }
    memset(route, 0, sizeof(*route));
}

uint64_t SparkComputeMtpDraftStagePackMetadataChecksum(const SparkMtpDraftStagePackMetadata *metadata)
{
    uint64_t checksum;

    if (metadata == 0)
    {
        return 0u;
    }
    checksum = SPARKPIPE_MTP_DRAFT_STAGEPACK_METADATA_SENTINEL;
    checksum = SparkMixU64(checksum, metadata->metadata_version);
    checksum = SparkMixU64(checksum, metadata->stage_id);
    checksum = SparkMixU64(checksum, metadata->expert_id);
    checksum = SparkMixU64(checksum, metadata->hidden_size);
    checksum = SparkMixU64(checksum, metadata->intermediate_rows);
    checksum = SparkMixU64(checksum, metadata->vocabulary_size);
    checksum = SparkMixU64(checksum, metadata->draft_token_count);
    checksum = SparkMixU64(checksum, (uint32_t)metadata->metadata_source);
    checksum = SparkMixU64(checksum, (uint32_t)metadata->weight_format);
    checksum = SparkMixU64(checksum, (uint32_t)metadata->value_encoding);
    checksum = SparkMixU64(checksum, (uint32_t)metadata->scale_encoding);
    checksum = SparkMixU64(checksum, metadata->values_per_scale_block);
    checksum = SparkMixU64(checksum, metadata->gate_payload_bytes);
    checksum = SparkMixU64(checksum, metadata->gate_scale_bytes);
    checksum = SparkMixU64(checksum, metadata->up_payload_bytes);
    checksum = SparkMixU64(checksum, metadata->up_scale_bytes);
    checksum = SparkMixU64(checksum, metadata->down_payload_bytes);
    checksum = SparkMixU64(checksum, metadata->down_scale_bytes);
    return checksum;
}

uint64_t SparkComputeMtpDraftRouteChecksum(const SparkMtpDraftRoute *route)
{
    uint64_t checksum;

    if (route == 0)
    {
        return 0u;
    }
    checksum = 0x53504D5450525445ull;
    checksum = SparkMixU64(checksum, route->ready ? 1u : 0u);
    checksum = SparkMixU64(checksum, route->stage_id);
    checksum = SparkMixU64(checksum, route->expert_id);
    checksum = SparkMixU64(checksum, route->hidden_size);
    checksum = SparkMixU64(checksum, route->intermediate_rows);
    checksum = SparkMixU64(checksum, route->vocabulary_size);
    checksum = SparkMixU64(checksum, route->draft_token_count);
    checksum = SparkMixU64(checksum, (uint32_t)route->metadata_source);
    checksum = SparkMixU64(checksum, (uint32_t)route->weight_format);
    checksum = SparkMixU64(checksum, (uint32_t)route->cuda_kernel);
    checksum = SparkMixU64(checksum, route->stagepack_metadata_count);
    checksum = SparkMixU64(checksum, route->model_level_quant_name_count);
    checksum = SparkMixU64(checksum, route->model_default_format_ignored_count);
    checksum = SparkMixU64(checksum, route->ambiguous_metadata_count);
    checksum = SparkMixU64(checksum, route->mxfp4_route_count);
    checksum = SparkMixU64(checksum, route->nvfp4_rejected_count);
    checksum = SparkMixU64(checksum, route->metadata_checksum);
    return checksum;
}

SparkStatus SparkFinalizeMtpDraftStagePackMetadata(SparkMtpDraftStagePackMetadata *metadata)
{
    if (metadata == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    metadata->metadata_checksum = SparkComputeMtpDraftStagePackMetadataChecksum(metadata);
    return SparkValidateMtpDraftStagePackMetadata(metadata);
}

SparkStatus SparkValidateMtpDraftStagePackMetadata(const SparkMtpDraftStagePackMetadata *metadata)
{
    SparkFp4FormatContract contract;

    if (metadata == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (metadata->sentinel != SPARKPIPE_MTP_DRAFT_STAGEPACK_METADATA_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (metadata->metadata_version < SPARKPIPE_MTP_DRAFT_MIN_METADATA_VERSION || metadata->metadata_version > SPARKPIPE_MTP_DRAFT_CURRENT_METADATA_VERSION)
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    if (metadata->metadata_source != SPARK_MTP_DRAFT_METADATA_SOURCE_STAGEPACK_EXPERT)
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    if (metadata->hidden_size == 0u || metadata->intermediate_rows == 0u || metadata->vocabulary_size == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (metadata->draft_token_count == 0u || metadata->draft_token_count > SPARKPIPE_MTP_DRAFT_MAX_DRAFT_TOKENS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkMtpDraftPayloadSizesArePresent(metadata))
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    if (SparkMtpDraftExpectedFormatContract(metadata, &contract) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    if (!SparkFp4FormatContractIsValid(&contract))
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    if (SparkMtpDraftCudaKernelForFormat(metadata->weight_format) == SPARK_MTP_DRAFT_CUDA_KERNEL_UNKNOWN)
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    if (metadata->weight_format != SPARK_FP4_STORAGE_FORMAT_MXFP4_E2M1_E8M0)
    {
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }
    if (metadata->metadata_checksum == 0u || metadata->metadata_checksum != SparkComputeMtpDraftStagePackMetadataChecksum(metadata))
    {
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRouteMtpDraftStagePackMetadata(const SparkMtpDraftStagePackMetadata *metadata, SparkFp4StorageFormatKind model_default_format, SparkMtpDraftRoute *route)
{
    SparkStatus status;

    if (metadata == 0 || route == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkMtpDraftRouteReset(route);
    route->metadata_source = metadata->metadata_source;
    route->weight_format = metadata->weight_format;
    route->cuda_kernel = SparkMtpDraftCudaKernelForFormat(metadata->weight_format);
    route->metadata_checksum = metadata->metadata_checksum;
    if (model_default_format != SPARK_FP4_STORAGE_FORMAT_UNKNOWN && model_default_format != metadata->weight_format)
    {
        route->model_default_format_ignored_count = 1u;
    }
    if (metadata->metadata_source == SPARK_MTP_DRAFT_METADATA_SOURCE_STAGEPACK_EXPERT)
    {
        route->stagepack_metadata_count = 1u;
    }
    else if (metadata->metadata_source == SPARK_MTP_DRAFT_METADATA_SOURCE_MODEL_LEVEL_QUANT_NAME)
    {
        route->model_level_quant_name_count = 1u;
        route->ambiguous_metadata_count = 1u;
        SparkMtpDraftRouteSetBlocker(route, "MTP draft expert quantization came from model-level quant name; StagePack expert metadata is required");
    }
    else
    {
        route->ambiguous_metadata_count = 1u;
        SparkMtpDraftRouteSetBlocker(route, "MTP draft expert quantization metadata is missing or ambiguous");
    }
    if (metadata->weight_format == SPARK_FP4_STORAGE_FORMAT_NVFP4_E2M1_E4M3)
    {
        route->nvfp4_rejected_count = 1u;
        SparkMtpDraftRouteSetBlocker(route, "MTP draft expert metadata resolved to NVFP4 E2M1/E4M3; DSV4 draft experts require MXFP4 E2M1/E8M0");
    }
    status = SparkValidateMtpDraftStagePackMetadata(metadata);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    route->ready = true;
    route->stage_id = metadata->stage_id;
    route->expert_id = metadata->expert_id;
    route->hidden_size = metadata->hidden_size;
    route->intermediate_rows = metadata->intermediate_rows;
    route->vocabulary_size = metadata->vocabulary_size;
    route->draft_token_count = metadata->draft_token_count;
    route->mxfp4_route_count = 1u;
    route->first_blocker[0] = '\0';
    route->blocker_count = 0u;
    route->route_checksum = SparkComputeMtpDraftRouteChecksum(route);
    return SPARK_STATUS_OK;
}
