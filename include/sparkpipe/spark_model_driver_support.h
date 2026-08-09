#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_model_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void SparkModelDriverInitializeCreateRequest(
    SparkModelDriverCreateRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_MODEL_DRIVER_ABI_VERSION;
    request->descriptor_bytes = SPARK_MODEL_DRIVER_CREATE_REQUEST_BYTES;
}

static inline uint32_t SparkModelDriverCreateRequestIsValid(
    const SparkModelDriverCreateRequest *request)
{
    if (request == 0 ||
        request->abi_version != SPARK_MODEL_DRIVER_ABI_VERSION ||
        request->descriptor_bytes != SPARK_MODEL_DRIVER_CREATE_REQUEST_BYTES ||
        request->flags != 0u || request->reserved0 != 0u ||
        request->node_id == 0 || request->node_id[0] == '\0' ||
        request->node_target == 0 || request->node_target[0] == '\0' ||
        request->reserved[0] != 0u || request->reserved[1] != 0u)
    {
        return 0u;
    }
    return 1u;
}

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

static inline uint32_t SparkModelDriverProgramSupportsRuntimeLimits(
    const SparkModelDriverProgramDescriptor *program,
    uint32_t required_flags,
    uint32_t max_inflight,
    uint32_t max_active_slots,
    uint32_t max_new_tokens,
    uint32_t max_resident_sequences)
{
    const SparkModelDriverProgramProfile *profile;
    if (program == 0 || program->profile == 0 ||
        program->profile->descriptor_bytes < sizeof(*program->profile) ||
        (program->flags & required_flags) != required_flags)
    {
        return 0u;
    }
    profile = program->profile;
    if (program->max_inflight < max_inflight ||
        profile->max_inflight < max_inflight ||
        profile->max_active_slots < max_active_slots ||
        profile->max_new_tokens < max_new_tokens ||
        profile->max_resident_sequences < max_resident_sequences)
    {
        return 0u;
    }
    return 1u;
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

static inline uint32_t SparkModelDriverAdmissionDecisionIsValid(
    const SparkModelDriverAdmissionDecision *decision)
{
    uint32_t has_dispatch_slot;
    if (decision == 0 || decision->descriptor_bytes < sizeof(*decision) ||
        decision->accepted > 1u ||
        decision->rejection_reason >
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE)
        return 0u;
    if ((decision->accepted != 0u) !=
        (decision->rejection_reason == SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED))
        return 0u;
    has_dispatch_slot = decision->driver_dispatch_slot !=
        SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT ? 1u : 0u;
    if (decision->accepted == 0u && has_dispatch_slot != 0u)
        return 0u;
    if (has_dispatch_slot == 0u &&
        (decision->driver_dispatch_generation != 0u ||
         decision->driver_dispatch_cookie0 != 0u ||
         decision->driver_dispatch_cookie1 != 0u))
        return 0u;
    if (has_dispatch_slot != 0u &&
        decision->driver_dispatch_generation == 0u)
        return 0u;
    return 1u;
}

static inline SparkStatus SparkModelDriverApplyAdmissionDecision(
    const SparkModelDriverAdmissionDecision *decision,
    SparkModelDriverFrame *frame)
{
    if (frame == 0 || SparkModelDriverAdmissionDecisionIsValid(decision) == 0u)
        return SPARK_STATUS_ABI_MISMATCH;
    if (decision->accepted == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (decision->driver_dispatch_slot ==
        SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
        return SPARK_STATUS_OK;
    frame->flags |= SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
    frame->driver_dispatch_slot = decision->driver_dispatch_slot;
    frame->driver_dispatch_generation = decision->driver_dispatch_generation;
    frame->driver_dispatch_cookie0 = decision->driver_dispatch_cookie0;
    frame->driver_dispatch_cookie1 = decision->driver_dispatch_cookie1;
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
