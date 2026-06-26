#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spark_glm52_resident_decode_stage_backend.h"

typedef struct SparkGlm52ResidentDecodeStageState SparkGlm52ResidentDecodeStageState;

typedef struct SparkGlm52ResidentDecodeStagePendingCompletion
{
    atomic_uint state;
    SparkGlm52ResidentDecodeStageState *owner;
    SparkGlm52ResidentDecodeStageBackendCompletion backend_completion;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint32_t program_id;
    uint32_t driver_dispatch_slot;
    uint32_t accepted_token_count;
    atomic_uint_fast64_t dispatch_generation;
    SparkModelDriverResidencyToken residency;
} SparkGlm52ResidentDecodeStagePendingCompletion;

enum
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE = 0u,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_RESERVED = 1u,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_SUBMITTED = 2u
};

struct SparkGlm52ResidentDecodeStageState
{
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
    const SparkGlm52ResidentDecodeStageNodeContext *node_context;
    uint32_t pipeline_slot_count;
    atomic_uint_fast64_t submitted_count;
    atomic_uint_fast64_t completed_count;
    atomic_uint_fast64_t rejected_count;
    atomic_uint_fast64_t host_callback_completion_count;
    atomic_uint_fast64_t stale_admission_count;
    SparkGlm52ResidentDecodeStagePendingCompletion pending_completions[];
};

static bool SparkGlm52ResidentDecodeStagePointerIsAligned(
    const void *pointer,
    uintptr_t required_alignment)
{
    return pointer != 0 &&
        ((uintptr_t)pointer % required_alignment) == 0u;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStagePipelineSlot(
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (pipeline_slot == 0 || pipeline_slot->cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->input_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->normalized_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->query_latent_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->query_rope_input_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->key_rope_input_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->current_kv_latent_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->positions,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->slot_mapping,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->block_table,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->context_lengths,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->first_block_token_offsets,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->dsa_token_scores,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->sparse_token_indices,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->rotated_query_rope_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->attention_output_latent_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->attention_projected_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->post_attention_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_hidden_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_logits,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_logits,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_selected_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->restricted_selected_token_scores,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_draft_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_target_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_accept_mask,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_committed_token_ids,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->mtp_event_counters,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            pipeline_slot->phase_clock_cycles,
            8u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateGlm52ResidentDecodeStageNodeContext(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    uint64_t represented_token_capacity;
    uint32_t pipeline_slot_index;

    if (node_context == 0 ||
        node_context->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (node_context->pipeline_slot_count == 0u ||
        node_context->pipeline_slot_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT ||
        node_context->max_active_sequence_count == 0u ||
        node_context->cache_token_capacity == 0u ||
        node_context->kv_block_count == 0u ||
        node_context->max_blocks_per_sequence == 0u ||
        node_context->position_count == 0u ||
        node_context->dsa_candidate_count == 0u ||
        node_context->sparse_index_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DEBUG_SERIAL_TOPK ||
        node_context->launch_check_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR ||
        node_context->phase_clock_mode >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64 ||
        !isfinite(node_context->qk_scale) ||
        node_context->qk_scale <= 0.0f ||
        !isfinite(node_context->rms_norm_epsilon) ||
        node_context->rms_norm_epsilon <= 0.0f ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->cos_table,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->sin_table,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->mla_cache_bf16,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->attention_norm_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->query_latent_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->query_rope_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->key_rope_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->kv_latent_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->attention_output_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->final_norm_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->restricted_lm_head_weight_bf16,
            2u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->mtp_mxfp4_weight_payload_u8,
            1u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->mtp_mxfp4_scale_e8m0_u8,
            1u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAligned(
            node_context->restricted_token_ids,
            4u) ||
        node_context->pipeline_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    represented_token_capacity =
        (uint64_t)node_context->kv_block_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
    if ((uint64_t)node_context->cache_token_capacity >
            represented_token_capacity ||
        node_context->max_blocks_per_sequence >
            node_context->kv_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (node_context->enable_cuda_graph_replay != 0u)
    {
        if (node_context->cuda_pipeline_slot_states == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        SparkStatus status;

        status = SparkValidateGlm52ResidentDecodeStagePipelineSlot(
            &node_context->pipeline_slots[pipeline_slot_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (node_context->cuda_pipeline_slot_states != 0 &&
            node_context->cuda_pipeline_slot_states[pipeline_slot_index].abi_version !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION)
        {
            return SPARK_STATUS_ABI_MISMATCH;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52ResidentDecodeStageReleaseSlot(
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion)
{
    atomic_fetch_add_explicit(
        &pending_completion->dispatch_generation,
        1u,
        memory_order_release);
    atomic_store_explicit(
        &pending_completion->state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE,
        memory_order_release);
}

static void SparkGlm52ResidentDecodeStageComplete(void *completion_context)
{
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion;
    SparkGlm52ResidentDecodeStageState *state;
    SparkModelDriverCompletion completion;

    pending_completion =
        (SparkGlm52ResidentDecodeStagePendingCompletion *)completion_context;
    if (pending_completion == 0 || pending_completion->owner == 0)
    {
        return;
    }
    state = pending_completion->owner;
    if (atomic_load_explicit(
            &pending_completion->state,
            memory_order_acquire) !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_SUBMITTED)
    {
        return;
    }

    memset(&completion, 0, sizeof(completion));
    completion.request_id = pending_completion->request_id;
    completion.sequence_id = pending_completion->sequence_id;
    completion.sequence_position = pending_completion->sequence_position;
    completion.program_id = pending_completion->program_id;
    completion.driver_dispatch_slot = pending_completion->driver_dispatch_slot;
    completion.accepted_token_count = pending_completion->accepted_token_count;
    completion.status = SPARK_STATUS_OK;
    completion.residency = pending_completion->residency;
    completion.device_memcpy_bytes = 0u;
    completion.host_staging_bytes = 0u;

    SparkGlm52ResidentDecodeStageReleaseSlot(pending_completion);
    atomic_fetch_add_explicit(
        &state->completed_count,
        1u,
        memory_order_relaxed);
    atomic_fetch_add_explicit(
        &state->host_callback_completion_count,
        1u,
        memory_order_relaxed);
    state->completion_function(state->completion_context, &completion);
}

SparkStatus SparkGlm52ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
    const SparkGlm52ResidentDecodeStageNodeContext *node_context;
    SparkGlm52ResidentDecodeStageState *state;
    size_t allocation_bytes;
    uint32_t pipeline_slot_index;
    SparkStatus status;

    if (configuration == 0 || host_services == 0 || module_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *module_state = 0;
    if (configuration->abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION ||
        configuration->reserved != 0u)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (host_services->completion_function == 0 ||
        host_services->node_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    node_context = (const SparkGlm52ResidentDecodeStageNodeContext *)
        host_services->node_context;
    status = SparkValidateGlm52ResidentDecodeStageNodeContext(node_context);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    allocation_bytes = sizeof(*state) +
        ((size_t)node_context->pipeline_slot_count *
         sizeof(state->pending_completions[0]));
    state = (SparkGlm52ResidentDecodeStageState *)calloc(1u, allocation_bytes);
    if (state == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->completion_function = host_services->completion_function;
    state->completion_context = host_services->completion_context;
    state->node_context = node_context;
    state->pipeline_slot_count = node_context->pipeline_slot_count;
    atomic_init(&state->submitted_count, 0u);
    atomic_init(&state->completed_count, 0u);
    atomic_init(&state->rejected_count, 0u);
    atomic_init(&state->host_callback_completion_count, 0u);
    atomic_init(&state->stale_admission_count, 0u);
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < state->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion;

        pending_completion = &state->pending_completions[pipeline_slot_index];
        atomic_init(
            &pending_completion->state,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE);
        atomic_init(&pending_completion->dispatch_generation, 1u);
        pending_completion->owner = state;
        pending_completion->backend_completion.function =
            SparkGlm52ResidentDecodeStageComplete;
        pending_completion->backend_completion.context = pending_completion;
    }

    *module_state = state;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52ResidentDecodeStageCountSlotsInState(
    const SparkGlm52ResidentDecodeStageState *state,
    unsigned int slot_state)
{
    uint32_t slot_index;
    uint32_t matching_slot_count;

    matching_slot_count = 0u;
    if (state == 0)
    {
        return 0u;
    }
    for (slot_index = 0u; slot_index < state->pipeline_slot_count; ++slot_index)
    {
        if (atomic_load_explicit(
                &state->pending_completions[slot_index].state,
                memory_order_acquire) == slot_state)
        {
            matching_slot_count += 1u;
        }
    }
    return matching_slot_count;
}

static uint32_t SparkGlm52ResidentDecodeStageFindAvailableSlot(
    const SparkGlm52ResidentDecodeStageState *state)
{
    uint32_t slot_index;

    if (state == 0)
    {
        return SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    }
    for (slot_index = 0u; slot_index < state->pipeline_slot_count; ++slot_index)
    {
        if (atomic_load_explicit(
                &state->pending_completions[slot_index].state,
                memory_order_acquire) ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE)
        {
            return slot_index;
        }
    }
    return SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
}

static bool SparkGlm52ResidentDecodeStageFrameShapeIsSupported(
    const SparkGlm52ResidentDecodeStageState *state,
    const SparkModelDriverFrame *frame)
{
    if (state == 0 || frame == 0)
    {
        return false;
    }
    if (frame->active_slot_count == 0u ||
        frame->active_slot_count > state->node_context->max_active_sequence_count ||
        frame->new_token_count >
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u) ||
        frame->program_id == 0u ||
        frame->buffer_count != 0u ||
        frame->buffers != 0)
    {
        return false;
    }
    return true;
}

SparkStatus SparkGlm52ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
    SparkGlm52ResidentDecodeStageState *state;
    SparkGlm52ResidentDecodeStagePendingCompletion *pending_completion;
    uint64_t pipeline_slot_value;
    uint32_t pipeline_slot_index;
    uint64_t current_dispatch_generation;
    unsigned int expected_state;
    SparkStatus status;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ResidentDecodeStageFrameShapeIsSupported(state, frame))
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u)
    {
        pipeline_slot_value = frame->driver_dispatch_slot;
    }
    else
    {
        pipeline_slot_value =
            frame->scalar[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX];
    }
    if (pipeline_slot_value >= state->pipeline_slot_count)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot_index = (uint32_t)pipeline_slot_value;
    pending_completion = &state->pending_completions[pipeline_slot_index];
    if ((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u)
    {
        current_dispatch_generation = atomic_load_explicit(
            &pending_completion->dispatch_generation,
            memory_order_acquire);
        if (frame->driver_dispatch_generation != current_dispatch_generation)
        {
            atomic_fetch_add_explicit(
                &state->stale_admission_count,
                1u,
                memory_order_relaxed);
            return SPARK_STATUS_BUSY;
        }
    }
    expected_state = SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE;
    if (!atomic_compare_exchange_strong_explicit(
            &pending_completion->state,
            &expected_state,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_RESERVED,
            memory_order_acq_rel,
            memory_order_relaxed))
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SPARK_STATUS_BUSY;
    }

    pending_completion->request_id = frame->request_id;
    pending_completion->sequence_id = frame->sequence_id;
    pending_completion->sequence_position = frame->sequence_position;
    pending_completion->program_id = frame->program_id;
    pending_completion->driver_dispatch_slot = pipeline_slot_index;
    pending_completion->accepted_token_count =
        frame->new_token_count != 0u ? frame->new_token_count : 1u;
    pending_completion->residency = frame->residency;
    atomic_store_explicit(
        &pending_completion->state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_SUBMITTED,
        memory_order_release);
    status = SparkGlm52ResidentDecodeStageBackendSubmit(
        state->node_context,
        pipeline_slot_index,
        frame->active_slot_count,
        &pending_completion->backend_completion);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageReleaseSlot(pending_completion);
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return status;
    }

    atomic_fetch_add_explicit(
        &state->submitted_count,
        1u,
        memory_order_relaxed);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkGlm52ResidentDecodeStageState *state;
    uint32_t available_slot_count;
    uint32_t active_submission_count;
    uint32_t selected_slot;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(decision, 0, sizeof(*decision));
    decision->descriptor_bytes = sizeof(*decision);
    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;

    if (request->descriptor_bytes < sizeof(*request) ||
        request->program_id == 0u ||
        request->active_slot_count == 0u ||
        request->active_slot_count > state->node_context->max_active_sequence_count ||
        request->new_token_count >
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u))
    {
        decision->rejection_reason =
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SPARK_STATUS_OK;
    }

    selected_slot = SparkGlm52ResidentDecodeStageFindAvailableSlot(state);
    available_slot_count = SparkGlm52ResidentDecodeStageCountSlotsInState(
        state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE);
    active_submission_count = state->pipeline_slot_count - available_slot_count;
    decision->available_dispatch_slot_count = available_slot_count;
    decision->private_queue_pressure = state->pipeline_slot_count != 0u ?
        (uint32_t)(((uint64_t)active_submission_count * 1024u) /
                  (uint64_t)state->pipeline_slot_count) :
        1024u;
    decision->endpoint_cost =
        ((uint64_t)decision->private_queue_pressure << 32u) |
        (uint64_t)active_submission_count;
    decision->device_memcpy_bytes = 0u;
    decision->host_staging_bytes = 0u;

    if (selected_slot == SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
    {
        decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY;
        return SPARK_STATUS_OK;
    }

    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    decision->driver_dispatch_slot = selected_slot;
    decision->driver_dispatch_generation = atomic_load_explicit(
        &state->pending_completions[selected_slot].dispatch_generation,
        memory_order_acquire);
    decision->driver_dispatch_cookie0 =
        ((uint64_t)selected_slot << 32u) ^ decision->driver_dispatch_generation;
    decision->driver_dispatch_cookie1 =
        request->sequence_id ^ request->sequence_position;
    decision->residency_match_score =
        request->residency.owner != 0u ? UINT64_MAX : 0u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkGlm52ResidentDecodeStageState *state;
    uint32_t available_slot_count;
    uint32_t active_submission_count;
    uint32_t pipeline_slot_index;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0 || snapshot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    available_slot_count = SparkGlm52ResidentDecodeStageCountSlotsInState(
        state,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SLOT_AVAILABLE);
    active_submission_count = state->pipeline_slot_count - available_slot_count;

    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->descriptor_bytes = sizeof(*snapshot);
    snapshot->program_id = program_id;
    snapshot->active_submission_count = active_submission_count;
    snapshot->available_dispatch_slot_count = available_slot_count;
    snapshot->submitted_count = atomic_load_explicit(
        &state->submitted_count,
        memory_order_relaxed);
    snapshot->completed_count = atomic_load_explicit(
        &state->completed_count,
        memory_order_relaxed);
    snapshot->rejected_count = atomic_load_explicit(
        &state->rejected_count,
        memory_order_relaxed);
    snapshot->resident_sequence_count = state->node_context->max_active_sequence_count;
    snapshot->resident_token_count = state->node_context->cache_token_capacity;
    snapshot->kv_token_capacity = state->node_context->cache_token_capacity;
    snapshot->device_memcpy_bytes_per_submit = 0u;
    snapshot->host_staging_bytes_per_submit = 0u;
    snapshot->host_callback_completion_count = atomic_load_explicit(
        &state->host_callback_completion_count,
        memory_order_relaxed);
    snapshot->stale_admission_count = atomic_load_explicit(
        &state->stale_admission_count,
        memory_order_relaxed);
    if (state->node_context->cuda_pipeline_slot_states != 0)
    {
        for (pipeline_slot_index = 0u;
             pipeline_slot_index < state->pipeline_slot_count;
             ++pipeline_slot_index)
        {
            const SparkGlm52ResidentDecodeStageCudaPipelineSlotState *slot_state;

            slot_state = &state->node_context->cuda_pipeline_slot_states[
                pipeline_slot_index];
            snapshot->cuda_graph_capture_count += slot_state->graph_capture_count;
            snapshot->cuda_graph_replay_count += slot_state->graph_replay_count;
        }
    }
    snapshot->private_queue_pressure = state->pipeline_slot_count != 0u ?
        (uint32_t)(((uint64_t)active_submission_count * 1024u) /
                  (uint64_t)state->pipeline_slot_count) :
        1024u;
    return SPARK_STATUS_OK;
}

void SparkGlm52ResidentDecodeStageDestroy(void *module_state)
{
    SparkGlm52ResidentDecodeStageState *state;

    state = (SparkGlm52ResidentDecodeStageState *)module_state;
    if (state == 0)
    {
        return;
    }
    SparkGlm52ResidentDecodeStageBackendQuiesce(state->node_context);
    free(state);
}
