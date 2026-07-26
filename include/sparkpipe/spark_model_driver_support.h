#ifndef SPARKPIPE_SPARK_MODEL_DRIVER_SUPPORT_H
#define SPARKPIPE_SPARK_MODEL_DRIVER_SUPPORT_H

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_model_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t SparkModelDriverRangeFitsWithinCapacity(
    uint64_t range_start,
    uint64_t range_length,
    uint64_t capacity)
{
    if (range_start > capacity)
    {
        return 0u;
    }
    return range_length <= capacity - range_start ? 1u : 0u;
}

static inline void SparkModelDriverInitializeAdmissionDecision(
    SparkModelDriverAdmissionDecision *decision)
{
    if (decision == 0)
    {
        return;
    }
    memset(decision, 0, sizeof(*decision));
    decision->descriptor_bytes = (uint32_t)sizeof(*decision);
    decision->rejection_reason =
        SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
}

static inline SparkStatus SparkModelDriverRejectAdmission(
    SparkModelDriverAdmissionDecision *decision,
    SparkModelDriverAdmissionRejection rejection_reason,
    uint32_t available_dispatch_slot_count)
{
    if (decision == 0 ||
        rejection_reason < SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY ||
        rejection_reason > SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkModelDriverInitializeAdmissionDecision(decision);
    decision->rejection_reason = (uint32_t)rejection_reason;
    decision->available_dispatch_slot_count = available_dispatch_slot_count;
    return SPARK_STATUS_OK;
}

static inline void SparkModelDriverInitializeRuntimeSnapshot(
    SparkModelDriverRuntimeSnapshot *snapshot,
    uint32_t program_id)
{
    if (snapshot == 0)
    {
        return;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->descriptor_bytes = (uint32_t)sizeof(*snapshot);
    snapshot->program_id = program_id;
}

static inline SparkStatus SparkModelDriverValidateBuffer(
    const SparkModelDriverFrame *frame,
    uint32_t buffer_index,
    uint32_t logical_slot,
    uint32_t required_flags,
    uint64_t minimum_bytes)
{
    const SparkModelDriverBuffer *buffer;
    const uint32_t known_flags =
        SPARK_MODEL_DRIVER_BUFFER_FLAG_READ |
        SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;

    if (frame == 0 || frame->buffers == 0 || buffer_index >= frame->buffer_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    buffer = &frame->buffers[buffer_index];
    if (buffer->slot != logical_slot ||
        (buffer->flags & required_flags) != required_flags ||
        (buffer->flags & ~known_flags) != 0u ||
        buffer->address == 0 ||
        buffer->bytes < minimum_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifdef __cplusplus
}
#endif

#endif
