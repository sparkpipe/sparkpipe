#ifndef SPARKPIPE_SPARK_ACTIVATION_LAYOUT_H
#define SPARKPIPE_SPARK_ACTIVATION_LAYOUT_H

#include "sparkpipe/spark_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkActivationElementKind
{
    SPARK_ACTIVATION_ELEMENT_BF16 = 0,
    SPARK_ACTIVATION_ELEMENT_FP16 = 1,
    SPARK_ACTIVATION_ELEMENT_FP8 = 2
} SparkActivationElementKind;

typedef struct SparkActivationLayout
{
    SparkPhysicalProfileId profile_id;
    SparkActivationElementKind element_kind;
    uint32_t physical_slot_count;
    uint32_t hidden_size;
    uint32_t element_size_bytes;
    uint32_t alignment_bytes;
    uint64_t row_bytes;
    uint64_t row_stride_bytes;
    uint64_t payload_bytes;
    uint64_t aligned_payload_bytes;
} SparkActivationLayout;

typedef struct SparkActivationBuffer
{
    bool initialized;
    uint32_t buffer_id;
    SparkActivationLayout layout;
    SparkActiveSlotMask active_slot_mask;
    uint64_t simulated_bytes_used;
    uint64_t payload_checksum;
    uint8_t simulated_bytes[SPARKPIPE_SIMULATED_ACTIVATION_BUFFER_BYTES];
} SparkActivationBuffer;

uint32_t SparkGetActivationElementSizeBytes(SparkActivationElementKind element_kind);
SparkStatus SparkBuildActivationLayout(SparkActivationLayout *activation_layout, SparkPhysicalProfileId profile_id, uint32_t hidden_size, SparkActivationElementKind element_kind);
SparkStatus SparkValidateActivationLayout(const SparkActivationLayout *activation_layout);
SparkStatus SparkInitializeActivationBuffer(SparkActivationBuffer *activation_buffer, uint32_t buffer_id, const SparkActivationLayout *activation_layout, const SparkActiveSlotMask *active_slot_mask, uint64_t seed_checksum);
uint64_t SparkComputeActivationBufferChecksum(const SparkActivationBuffer *activation_buffer);

#ifdef __cplusplus
}
#endif

#endif
