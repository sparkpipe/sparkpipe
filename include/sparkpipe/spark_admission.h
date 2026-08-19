#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_ADMISSION_ABI_VERSION 1u

/*
 * Per-model policy. The table is model-neutral: every ceiling, slot rule, and
 * tail is a data field. Model-specific logic lives only in the optional
 * predicate and cost callbacks, whose implementations stay in the model
 * directories and may name their model.
 */
typedef uint32_t SparkAdmissionPolicyFlags;
#define SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT 0x00000001u
#define SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS  0x00000002u
#define SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG  0x00000004u
#define SPARK_ADMISSION_POLICY_KNOWN_FLAGS \
    (SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT | \
     SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS | \
     SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG)

/* Model-specific cost tail (staging/memcpy/service-time fields). */
typedef void (*SparkAdmissionCostFunction)(
    void *context,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

/* Model-specific capacity/shape tail (for example JIT-KV prepare/commit/abort). */
typedef SparkStatus (*SparkAdmissionPredicateFunction)(
    void *context,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

typedef struct SparkAdmissionPolicyTable
{
    uint32_t abi_version;               /* SPARK_ADMISSION_ABI_VERSION */
    uint32_t descriptor_bytes;          /* sizeof(table)               */
    uint32_t max_active_sequence_count; /* active_slot_count ceiling; 0 = none  */
    uint32_t max_input_row_count;       /* new_token_count ceiling;   0 = none  */
    uint64_t max_sequence_positions;    /* KV sequence_position bound; 0 = none */
    SparkAdmissionPolicyFlags flags;    /* prefill/decode slot rules            */
    SparkAdmissionPredicateFunction predicate; /* NULL = shape/capacity accepted */
    void *predicate_context;
    SparkAdmissionCostFunction cost;           /* NULL = no cost fields          */
    void *cost_context;
} SparkAdmissionPolicyTable;

/* --- builders --- */
SparkStatus SparkAdmissionRequestFromSubmission(
    uint32_t program_id,
    const SparkModelServingSubmission *submission,
    const SparkModelDriverCacheLane *cache_lanes,
    uint32_t admission_flags,
    SparkModelDriverAdmissionRequest *request);

SparkStatus SparkAdmissionRequestFromFrame(
    uint32_t program_id,
    const SparkModelDriverFrame *frame,
    const SparkModelDriverCacheLane *cache_lanes,
    uint32_t admission_flags,
    SparkModelDriverAdmissionRequest *request);

/* --- evaluate/apply --- */
SparkStatus SparkAdmissionEvaluate(
    const SparkModelDriverInterface *driver_interface,
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

SparkStatus SparkAdmissionEvaluateAndApply(
    const SparkModelDriverInterface *driver_interface,
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverFrame *frame,
    SparkModelDriverAdmissionDecision *decision);

/* --- cost + merge --- */
uint64_t SparkAdmissionDecisionCost(
    const SparkModelDriverAdmissionDecision *decision);

SparkStatus SparkAdmissionMergeDecision(
    SparkModelDriverAdmissionDecision *destination,
    const SparkModelDriverAdmissionDecision *source);

/*
 * --- module-side table-driven gate ---
 * The common ladder each resident-decode stage admit repeats: descriptor/ABI
 * check, frame-flag whitelist, active/new-token ceilings, the two slot-rule
 * flags, the KV sequence-position bound, the optional predicate tail (which
 * runs before the BUSY check so a JIT-KV prepare or release short-circuits),
 * the BUSY check on a zero free-slot count, the optional cost tail, and
 * accept. Inlined so a module archive can call it without linking the core.
 */
static inline SparkStatus SparkAdmissionEvaluateShape(
    const SparkAdmissionPolicyTable *table,
    uint32_t available_dispatch_slot_count,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    uint32_t known_frame_flags;
    uint32_t is_prefill;
    SparkStatus status;

    if (table == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (table->abi_version != SPARK_ADMISSION_ABI_VERSION ||
        table->descriptor_bytes != (uint32_t)sizeof(*table) ||
        (table->flags & ~SPARK_ADMISSION_POLICY_KNOWN_FLAGS) != 0u)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }

    SparkModelDriverInitializeAdmissionDecision(decision);
    decision->available_dispatch_slot_count = available_dispatch_slot_count;

    if (request->descriptor_bytes < (uint32_t)sizeof(*request) ||
        request->program_id == 0u ||
        SparkModelDriverAdmissionRequestIsValid(request) == 0u)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }

    is_prefill = (request->frame_flags &
        SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
    known_frame_flags =
        SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL |
        SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
    if ((table->flags & SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG) != 0u)
    {
        known_frame_flags |= SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
    }

    if ((request->frame_flags & ~known_frame_flags) != 0u ||
        request->active_slot_count == 0u ||
        (table->max_active_sequence_count != 0u &&
         request->active_slot_count > table->max_active_sequence_count) ||
        request->new_token_count == 0u ||
        (table->max_input_row_count != 0u &&
         request->new_token_count > table->max_input_row_count) ||
        ((table->flags & SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT) != 0u &&
         is_prefill != 0u && request->active_slot_count != 1u) ||
        ((table->flags & SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS) != 0u &&
         is_prefill == 0u && request->new_token_count != request->active_slot_count))
    {
        SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE,
            available_dispatch_slot_count);
        return SPARK_STATUS_OK;
    }

    if (table->max_sequence_positions != 0u &&
        request->sequence_position >= table->max_sequence_positions)
    {
        SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY,
            available_dispatch_slot_count);
        return SPARK_STATUS_OK;
    }

    if (table->predicate != 0)
    {
        status = table->predicate(table->predicate_context, request, decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        /* A predicate may accept (short-circuit) or reject by editing the
         * decision; both are terminal for the ladder. */
        if (decision->accepted != 0u ||
            decision->rejection_reason != SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED)
        {
            return SPARK_STATUS_OK;
        }
    }

    if (available_dispatch_slot_count == 0u)
    {
        SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY,
            available_dispatch_slot_count);
        return SPARK_STATUS_OK;
    }

    if (table->cost != 0)
    {
        table->cost(table->cost_context, request, decision);
    }

    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    return SPARK_STATUS_OK;
}

#ifdef __cplusplus
}
#endif
