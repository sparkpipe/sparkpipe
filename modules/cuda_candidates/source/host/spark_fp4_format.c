#include "sparkpipe/spark_fp4_format.h"

#include <math.h>
#include <string.h>

static bool SparkFp4StrideIsAligned(uint64_t stride_bytes)
{
    return (stride_bytes % SPARKPIPE_FP4_STRIDE_ALIGNMENT_BYTES) == 0u;
}

static uint64_t SparkFp4ScaleBlockCount(uint32_t row_count, uint32_t column_count, uint32_t values_per_scale_block)
{
    if (row_count == 0u || column_count == 0u || values_per_scale_block == 0u)
    {
        return 0u;
    }
    return (uint64_t)row_count * ((uint64_t)column_count / (uint64_t)values_per_scale_block);
}

static void SparkFp4TensorGeometryReportReset(SparkFp4TensorGeometryReport *report)
{
    if (report == 0)
    {
        return;
    }
    memset(report, 0, sizeof(*report));
}

static SparkStatus SparkFillFp4TensorGeometryReport(uint32_t row_count, uint32_t column_count, uint32_t values_per_scale_block, uint64_t payload_stride_bytes, uint64_t scale_stride_bytes, SparkFp4TensorGeometryReport *report)
{
    uint64_t packed_row_bytes;
    uint64_t scale_row_bytes;

    if (report == 0)
    {
        return SPARK_STATUS_OK;
    }
    SparkFp4TensorGeometryReportReset(report);
    packed_row_bytes = SparkFp4PackedRowBytes(column_count);
    scale_row_bytes = SparkFp4ScaleRowBytes(column_count, values_per_scale_block);
    report->element_count = (uint64_t)row_count * (uint64_t)column_count;
    report->packed_row_bytes = packed_row_bytes;
    report->scale_row_bytes = scale_row_bytes;
    report->payload_byte_count = SparkFp4TensorPayloadBytes(row_count, payload_stride_bytes, packed_row_bytes);
    report->scale_byte_count = SparkFp4TensorScaleBytes(row_count, scale_stride_bytes, scale_row_bytes);
    report->scale_block_count = SparkFp4ScaleBlockCount(row_count, column_count, values_per_scale_block);
    report->row_count = row_count;
    report->column_count = column_count;
    report->values_per_scale_block = values_per_scale_block;
    return SPARK_STATUS_OK;
}

const char *SparkFp4StorageFormatKindToString(SparkFp4StorageFormatKind storage_format)
{
    switch (storage_format)
    {
        case SPARK_FP4_STORAGE_FORMAT_MXFP4_E2M1_E8M0:
        {
            return "mxfp4_e2m1_e8m0";
        }
        case SPARK_FP4_STORAGE_FORMAT_NVFP4_E2M1_E4M3:
        {
            return "nvfp4_e2m1_e4m3";
        }
        default:
        {
            return "unknown_fp4_storage_format";
        }
    }
}

const char *SparkFp4ValueEncodingKindToString(SparkFp4ValueEncodingKind value_encoding)
{
    switch (value_encoding)
    {
        case SPARK_FP4_VALUE_ENCODING_E2M1:
        {
            return "e2m1";
        }
        default:
        {
            return "unknown_fp4_value_encoding";
        }
    }
}

const char *SparkFp4ScaleEncodingKindToString(SparkFp4ScaleEncodingKind scale_encoding)
{
    switch (scale_encoding)
    {
        case SPARK_FP4_SCALE_ENCODING_E8M0:
        {
            return "e8m0";
        }
        case SPARK_FP4_SCALE_ENCODING_E4M3:
        {
            return "e4m3";
        }
        default:
        {
            return "unknown_fp4_scale_encoding";
        }
    }
}

SparkStatus SparkDescribeFp4Format(SparkFp4StorageFormatKind storage_format, SparkFp4FormatContract *contract)
{
    if (contract == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(contract, 0, sizeof(*contract));
    switch (storage_format)
    {
        case SPARK_FP4_STORAGE_FORMAT_MXFP4_E2M1_E8M0:
        {
            contract->storage_format = storage_format;
            contract->value_encoding = SPARK_FP4_VALUE_ENCODING_E2M1;
            contract->scale_encoding = SPARK_FP4_SCALE_ENCODING_E8M0;
            contract->values_per_scale_block = SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK;
            contract->scale_bytes_per_block = 1u;
            contract->requires_global_scale = false;
            contract->ready = true;
            return SPARK_STATUS_OK;
        }
        case SPARK_FP4_STORAGE_FORMAT_NVFP4_E2M1_E4M3:
        {
            contract->storage_format = storage_format;
            contract->value_encoding = SPARK_FP4_VALUE_ENCODING_E2M1;
            contract->scale_encoding = SPARK_FP4_SCALE_ENCODING_E4M3;
            contract->values_per_scale_block = SPARKPIPE_NVFP4_E2M1_E4M3_SCALE_BLOCK;
            contract->scale_bytes_per_block = 1u;
            contract->requires_global_scale = true;
            contract->ready = true;
            return SPARK_STATUS_OK;
        }
        default:
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
}

bool SparkFp4FormatContractIsValid(const SparkFp4FormatContract *contract)
{
    SparkFp4FormatContract expected_contract;

    if (contract == 0 || !contract->ready)
    {
        return false;
    }
    if (SparkDescribeFp4Format(contract->storage_format, &expected_contract) != SPARK_STATUS_OK)
    {
        return false;
    }
    return contract->value_encoding == expected_contract.value_encoding &&
           contract->scale_encoding == expected_contract.scale_encoding &&
           contract->values_per_scale_block == expected_contract.values_per_scale_block &&
           contract->scale_bytes_per_block == expected_contract.scale_bytes_per_block &&
           contract->requires_global_scale == expected_contract.requires_global_scale;
}

void SparkMxfp4E2m1E8m0TensorDescriptorReset(SparkMxfp4E2m1E8m0TensorDescriptor *descriptor)
{
    if (descriptor == 0)
    {
        return;
    }
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->values_per_scale_block = SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK;
    descriptor->sentinel = SPARKPIPE_MXFP4_E2M1_E8M0_SENTINEL;
}

void SparkNvfp4E2m1E4m3TensorDescriptorReset(SparkNvfp4E2m1E4m3TensorDescriptor *descriptor)
{
    if (descriptor == 0)
    {
        return;
    }
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->values_per_scale_block = SPARKPIPE_NVFP4_E2M1_E4M3_SCALE_BLOCK;
    descriptor->global_scale = 1.0f;
    descriptor->sentinel = SPARKPIPE_NVFP4_E2M1_E4M3_SENTINEL;
}

uint64_t SparkFp4PackedRowBytes(uint32_t column_count)
{
    if (column_count == 0u)
    {
        return 0u;
    }
    return ((uint64_t)column_count + (uint64_t)SPARKPIPE_FP4_VALUES_PER_BYTE - 1u) / (uint64_t)SPARKPIPE_FP4_VALUES_PER_BYTE;
}

uint64_t SparkFp4ScaleRowBytes(uint32_t column_count, uint32_t values_per_scale_block)
{
    if (column_count == 0u || values_per_scale_block == 0u || (column_count % values_per_scale_block) != 0u)
    {
        return 0u;
    }
    return (uint64_t)column_count / (uint64_t)values_per_scale_block;
}

uint64_t SparkFp4TensorPayloadBytes(uint32_t row_count, uint64_t payload_stride_bytes, uint64_t packed_row_bytes)
{
    if (row_count == 0u || payload_stride_bytes == 0u || packed_row_bytes == 0u || payload_stride_bytes < packed_row_bytes)
    {
        return 0u;
    }
    return ((uint64_t)(row_count - 1u) * payload_stride_bytes) + packed_row_bytes;
}

uint64_t SparkFp4TensorScaleBytes(uint32_t row_count, uint64_t scale_stride_bytes, uint64_t scale_row_bytes)
{
    if (row_count == 0u || scale_stride_bytes == 0u || scale_row_bytes == 0u || scale_stride_bytes < scale_row_bytes)
    {
        return 0u;
    }
    return ((uint64_t)(row_count - 1u) * scale_stride_bytes) + scale_row_bytes;
}

SparkStatus SparkValidateMxfp4E2m1E8m0TensorDescriptor(const SparkMxfp4E2m1E8m0TensorDescriptor *descriptor, SparkFp4TensorGeometryReport *report)
{
    uint64_t packed_row_bytes;
    uint64_t scale_row_bytes;

    SparkFp4TensorGeometryReportReset(report);
    if (descriptor == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->sentinel != SPARKPIPE_MXFP4_E2M1_E8M0_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->row_count == 0u || descriptor->column_count == 0u || descriptor->values_per_scale_block != SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((descriptor->column_count % SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    packed_row_bytes = SparkFp4PackedRowBytes(descriptor->column_count);
    scale_row_bytes = SparkFp4ScaleRowBytes(descriptor->column_count, descriptor->values_per_scale_block);
    if (descriptor->payload_stride_bytes < packed_row_bytes || descriptor->scale_stride_bytes < scale_row_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkFp4StrideIsAligned(descriptor->payload_stride_bytes) || !SparkFp4StrideIsAligned(descriptor->scale_stride_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkFillFp4TensorGeometryReport(descriptor->row_count, descriptor->column_count, descriptor->values_per_scale_block, descriptor->payload_stride_bytes, descriptor->scale_stride_bytes, report);
}

SparkStatus SparkValidateNvfp4E2m1E4m3TensorDescriptor(const SparkNvfp4E2m1E4m3TensorDescriptor *descriptor, SparkFp4TensorGeometryReport *report)
{
    uint64_t packed_row_bytes;
    uint64_t scale_row_bytes;

    SparkFp4TensorGeometryReportReset(report);
    if (descriptor == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->sentinel != SPARKPIPE_NVFP4_E2M1_E4M3_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->row_count == 0u || descriptor->column_count == 0u || descriptor->values_per_scale_block != SPARKPIPE_NVFP4_E2M1_E4M3_SCALE_BLOCK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((descriptor->column_count % SPARKPIPE_NVFP4_E2M1_E4M3_SCALE_BLOCK) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(descriptor->global_scale) || descriptor->global_scale <= 0.0f)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    packed_row_bytes = SparkFp4PackedRowBytes(descriptor->column_count);
    scale_row_bytes = SparkFp4ScaleRowBytes(descriptor->column_count, descriptor->values_per_scale_block);
    if (descriptor->payload_stride_bytes < packed_row_bytes || descriptor->scale_stride_bytes < scale_row_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkFp4StrideIsAligned(descriptor->payload_stride_bytes) || !SparkFp4StrideIsAligned(descriptor->scale_stride_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkFillFp4TensorGeometryReport(descriptor->row_count, descriptor->column_count, descriptor->values_per_scale_block, descriptor->payload_stride_bytes, descriptor->scale_stride_bytes, report);
}
