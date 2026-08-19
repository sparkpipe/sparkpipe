#include <string.h>

#include "sparkpipe/spark_admission.h"

SparkStatus SparkAdmissionRequestFromSubmission(
    uint32_t program_id,
    const SparkModelServingSubmission *submission,
    const SparkModelDriverCacheLane *cache_lanes,
    uint32_t admission_flags,
    SparkModelDriverAdmissionRequest *request)
{
    if (submission == 0 || request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(request, 0, sizeof(*request));
    request->descriptor_bytes = (uint32_t)sizeof(*request);
    request->program_id = program_id;
    request->submission_id = submission->submission_id;
    request->control_generation = submission->control_generation;
    request->transaction_id = submission->transaction_id;
    request->request_generation = submission->request_generation;
    request->step_generation = submission->step_generation;
    request->request_id = submission->request_id;
    request->sequence_id = submission->sequence_id;
    request->sequence_position = submission->sequence_position;
    request->deadline_time_ns = submission->deadline_time_ns;
    request->active_slot_count = submission->active_sequence_count;
    request->new_token_count = submission->new_token_count;
    request->priority = submission->priority;
    request->frame_flags =
        submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL
            ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL
            : submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE
                ? SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE
                : 0u;
    request->admission_flags = admission_flags;
    request->cache_lane_count =
        cache_lanes != 0 ? submission->active_sequence_count : 0u;
    request->cache_lanes = cache_lanes;
    request->residency = submission->residency;
    return SPARK_STATUS_OK;
}

SparkStatus SparkAdmissionRequestFromFrame(
    uint32_t program_id,
    const SparkModelDriverFrame *frame,
    const SparkModelDriverCacheLane *cache_lanes,
    uint32_t admission_flags,
    SparkModelDriverAdmissionRequest *request)
{
    if (frame == 0 || request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(request, 0, sizeof(*request));
    request->descriptor_bytes = (uint32_t)sizeof(*request);
    request->program_id = program_id;
    request->request_id = frame->request_id;
    request->sequence_id = frame->sequence_id;
    request->sequence_position = frame->sequence_position;
    request->deadline_time_ns = frame->deadline_time_ns;
    request->active_slot_count = frame->active_slot_count;
    request->new_token_count =
        frame->new_token_count != 0u ? frame->new_token_count : 1u;
    request->priority = frame->priority;
    request->frame_flags = frame->flags;
    request->admission_flags = admission_flags;
    request->cache_lane_count = frame->cache_lane_count;
    request->cache_lanes = cache_lanes != 0 ? cache_lanes : frame->cache_lanes;
    request->residency = frame->residency;
    return SPARK_STATUS_OK;
}

SparkStatus SparkAdmissionEvaluate(
    const SparkModelDriverInterface *driver_interface,
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    return SparkModelDriverEvaluateAdmission(
        driver_interface, driver_instance, request, decision);
}

SparkStatus SparkAdmissionEvaluateAndApply(
    const SparkModelDriverInterface *driver_interface,
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverFrame *frame,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkStatus status;

    status = SparkAdmissionEvaluate(
        driver_interface, driver_instance, request, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkModelDriverApplyAdmissionDecision(decision, frame);
}

uint64_t SparkAdmissionDecisionCost(
    const SparkModelDriverAdmissionDecision *decision)
{
    uint64_t cost;

    if (decision == 0)
    {
        return 0u;
    }
    cost = decision->endpoint_cost;
    cost += decision->estimated_queue_delay_ns;
    cost += decision->estimated_service_time_ns;
    cost += ((uint64_t)decision->private_queue_pressure) << 20u;
    cost += decision->host_staging_bytes << 2u;
    cost += decision->device_memcpy_bytes;
    if (decision->residency_match_score > cost)
    {
        return 0u;
    }
    return cost - decision->residency_match_score;
}

static uint64_t SparkAdmissionSaturatingAddU64(uint64_t left, uint64_t right)
{
    if (UINT64_MAX - left < right)
    {
        return UINT64_MAX;
    }
    return left + right;
}

SparkStatus SparkAdmissionMergeDecision(
    SparkModelDriverAdmissionDecision *destination,
    const SparkModelDriverAdmissionDecision *source)
{
    if (destination == 0 || source == 0 ||
        SparkModelDriverAdmissionDecisionIsValid(source) == 0u ||
        source->accepted == 0u)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }

    if (source->driver_dispatch_slot != SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
    {
        if (destination->driver_dispatch_slot == SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
        {
            destination->driver_dispatch_slot = source->driver_dispatch_slot;
            destination->driver_dispatch_generation = source->driver_dispatch_generation;
            destination->driver_dispatch_cookie0 = source->driver_dispatch_cookie0;
            destination->driver_dispatch_cookie1 = source->driver_dispatch_cookie1;
        }
        else if (destination->driver_dispatch_slot != source->driver_dispatch_slot ||
                 destination->driver_dispatch_generation != source->driver_dispatch_generation ||
                 destination->driver_dispatch_cookie0 != source->driver_dispatch_cookie0 ||
                 destination->driver_dispatch_cookie1 != source->driver_dispatch_cookie1)
        {
            return SPARK_STATUS_ABI_MISMATCH;
        }
    }

    if (source->estimated_queue_delay_ns > destination->estimated_queue_delay_ns)
    {
        destination->estimated_queue_delay_ns = source->estimated_queue_delay_ns;
    }
    destination->estimated_service_time_ns = SparkAdmissionSaturatingAddU64(
        destination->estimated_service_time_ns, source->estimated_service_time_ns);
    destination->endpoint_cost = SparkAdmissionSaturatingAddU64(
        destination->endpoint_cost, source->endpoint_cost);
    if (source->residency_match_score < destination->residency_match_score)
    {
        destination->residency_match_score = source->residency_match_score;
    }
    destination->device_memcpy_bytes = SparkAdmissionSaturatingAddU64(
        destination->device_memcpy_bytes, source->device_memcpy_bytes);
    destination->host_staging_bytes = SparkAdmissionSaturatingAddU64(
        destination->host_staging_bytes, source->host_staging_bytes);
    if (source->private_queue_pressure > destination->private_queue_pressure)
    {
        destination->private_queue_pressure = source->private_queue_pressure;
    }
    if (source->available_dispatch_slot_count < destination->available_dispatch_slot_count)
    {
        destination->available_dispatch_slot_count = source->available_dispatch_slot_count;
    }
    return SPARK_STATUS_OK;
}
