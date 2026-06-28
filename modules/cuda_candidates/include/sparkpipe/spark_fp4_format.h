#ifndef SPARKPIPE_SPARK_FP4_FORMAT_H
#define SPARKPIPE_SPARK_FP4_FORMAT_H

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_MXFP4_E2M1_E8M0_SENTINEL 0x535046504D584634ull
#define SPARKPIPE_NVFP4_E2M1_E4M3_SENTINEL 0x535046504E564634ull
#define SPARKPIPE_MXFP4_E2M1_E8M0_SCALE_BLOCK 32u
#define SPARKPIPE_NVFP4_E2M1_E4M3_SCALE_BLOCK 16u
#define SPARKPIPE_FP4_VALUES_PER_BYTE 2u
#define SPARKPIPE_FP4_STRIDE_ALIGNMENT_BYTES 16u

typedef enum SparkFp4StorageFormatKind
{
    SPARK_FP4_STORAGE_FORMAT_UNKNOWN = 0,
    SPARK_FP4_STORAGE_FORMAT_MXFP4_E2M1_E8M0 = 1,
    SPARK_FP4_STORAGE_FORMAT_NVFP4_E2M1_E4M3 = 2
} SparkFp4StorageFormatKind;

typedef enum SparkFp4ValueEncodingKind
{
    SPARK_FP4_VALUE_ENCODING_UNKNOWN = 0,
    SPARK_FP4_VALUE_ENCODING_E2M1 = 1
} SparkFp4ValueEncodingKind;

typedef enum SparkFp4ScaleEncodingKind
{
    SPARK_FP4_SCALE_ENCODING_UNKNOWN = 0,
    SPARK_FP4_SCALE_ENCODING_E8M0 = 1,
    SPARK_FP4_SCALE_ENCODING_E4M3 = 2
} SparkFp4ScaleEncodingKind;

typedef struct SparkFp4FormatContract
{
    SparkFp4StorageFormatKind storage_format;
    SparkFp4ValueEncodingKind value_encoding;
    SparkFp4ScaleEncodingKind scale_encoding;
    uint32_t values_per_scale_block;
    uint32_t scale_bytes_per_block;
    bool requires_global_scale;
    bool ready;
} SparkFp4FormatContract;

typedef struct SparkMxfp4E2m1E8m0TensorDescriptor
{
    uint32_t row_count;
    uint32_t column_count;
    uint32_t values_per_scale_block;
    uint32_t reserved0;
    uint64_t payload_stride_bytes;
    uint64_t scale_stride_bytes;
    uint64_t sentinel;
} SparkMxfp4E2m1E8m0TensorDescriptor;

typedef struct SparkNvfp4E2m1E4m3TensorDescriptor
{
    uint32_t row_count;
    uint32_t column_count;
    uint32_t values_per_scale_block;
    uint32_t reserved0;
    uint64_t payload_stride_bytes;
    uint64_t scale_stride_bytes;
    float global_scale;
    uint32_t reserved1;
    uint64_t sentinel;
} SparkNvfp4E2m1E4m3TensorDescriptor;

typedef struct SparkFp4TensorGeometryReport
{
    uint64_t element_count;
    uint64_t packed_row_bytes;
    uint64_t scale_row_bytes;
    uint64_t payload_byte_count;
    uint64_t scale_byte_count;
    uint64_t scale_block_count;
    uint32_t row_count;
    uint32_t column_count;
    uint32_t values_per_scale_block;
} SparkFp4TensorGeometryReport;

const char *SparkFp4StorageFormatKindToString(SparkFp4StorageFormatKind storage_format);
const char *SparkFp4ValueEncodingKindToString(SparkFp4ValueEncodingKind value_encoding);
const char *SparkFp4ScaleEncodingKindToString(SparkFp4ScaleEncodingKind scale_encoding);
SparkStatus SparkDescribeFp4Format(SparkFp4StorageFormatKind storage_format, SparkFp4FormatContract *contract);
bool SparkFp4FormatContractIsValid(const SparkFp4FormatContract *contract);
void SparkMxfp4E2m1E8m0TensorDescriptorReset(SparkMxfp4E2m1E8m0TensorDescriptor *descriptor);
void SparkNvfp4E2m1E4m3TensorDescriptorReset(SparkNvfp4E2m1E4m3TensorDescriptor *descriptor);
uint64_t SparkFp4PackedRowBytes(uint32_t column_count);
uint64_t SparkFp4ScaleRowBytes(uint32_t column_count, uint32_t values_per_scale_block);
uint64_t SparkFp4TensorPayloadBytes(uint32_t row_count, uint64_t payload_stride_bytes, uint64_t packed_row_bytes);
uint64_t SparkFp4TensorScaleBytes(uint32_t row_count, uint64_t scale_stride_bytes, uint64_t scale_row_bytes);
SparkStatus SparkValidateMxfp4E2m1E8m0TensorDescriptor(const SparkMxfp4E2m1E8m0TensorDescriptor *descriptor, SparkFp4TensorGeometryReport *report);
SparkStatus SparkValidateNvfp4E2m1E4m3TensorDescriptor(const SparkNvfp4E2m1E4m3TensorDescriptor *descriptor, SparkFp4TensorGeometryReport *report);

#ifdef __cplusplus
}
#endif

#endif
