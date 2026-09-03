#include "sparkpipe/spark_runtime_completion.h"

#include <stddef.h>
#include <string.h>

static SparkStatus SparkRuntimeValidateParticipant(
    const SparkRuntimeParticipant *participant,
    uint32_t expected_participant_index)
{
    if (participant == 0 ||
        participant->descriptor_bytes != SPARK_RUNTIME_PARTICIPANT_BYTES ||
        participant->participant_index != expected_participant_index ||
        (participant->flags & SPARK_RUNTIME_PARTICIPANT_REQUIRED_FLAGS) !=
            SPARK_RUNTIME_PARTICIPANT_REQUIRED_FLAGS ||
        (participant->flags & ~SPARK_RUNTIME_PARTICIPANT_KNOWN_FLAGS) != 0u ||
        participant->restart_epoch == 0u ||
        participant->participant_name == 0 ||
        participant->participant_name[0] == '\0' ||
        participant->prepare == 0 ||
        participant->execute == 0 ||
        participant->commit == 0 ||
        participant->cancel == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRuntimeValidateFinalEventQueue(
    const SparkRuntimeFinalEventQueue *queue)
{
    if (queue == 0 ||
        queue->abi_version != SPARK_RUNTIME_COMPLETION_ABI_VERSION ||
        queue->descriptor_bytes != SPARK_RUNTIME_FINAL_EVENT_QUEUE_BYTES ||
        queue->capacity == 0u ||
        queue->pending_count > queue->capacity ||
        queue->next_event_generation == 0u ||
        queue->events == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRuntimeValidateTransmissionWindow(
    const SparkRuntimeTransmissionWindow *window)
{
    if (window == 0 ||
        window->abi_version != SPARK_RUNTIME_COMPLETION_ABI_VERSION ||
        window->descriptor_bytes != SPARK_RUNTIME_TRANSMISSION_WINDOW_BYTES ||
        window->capacity == 0u ||
        window->in_use_count > window->capacity ||
        window->next_slot_generation == 0u ||
        window->slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkRuntimeFinalEvent *SparkRuntimeFindReusableFinalEvent(
    SparkRuntimeFinalEventQueue *queue)
{
    SparkRuntimeFinalEvent *oldest_acknowledged;
    uint64_t oldest_generation;
    uint32_t event_index;

    oldest_acknowledged = 0;
    oldest_generation = UINT64_MAX;
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        SparkRuntimeFinalEvent *event;

        event = &queue->events[event_index];
        if (event->state == SPARK_RUNTIME_FINAL_EVENT_STATE_FREE)
        {
            return event;
        }
        if (event->state == SPARK_RUNTIME_FINAL_EVENT_STATE_ACKNOWLEDGED &&
            event->event_generation < oldest_generation)
        {
            oldest_acknowledged = event;
            oldest_generation = event->event_generation;
        }
    }
    return oldest_acknowledged;
}

static SparkStatus SparkRuntimeEnqueueFinalEvent(
    SparkRuntimeFinalEventQueue *queue,
    const SparkRuntimeTransactionRequest *request,
    uint64_t payload_fingerprint,
    SparkStatus status,
    uint64_t *event_generation_out)
{
    SparkRuntimeFinalEvent *event;

    if (event_generation_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *event_generation_out = 0u;
    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK ||
        request == 0 ||
        payload_fingerprint == 0u ||
        queue->pending_count >= queue->capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    event = SparkRuntimeFindReusableFinalEvent(queue);
    if (event == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memset(event,0,sizeof(*event));
    event->magic = SPARK_RUNTIME_COMPLETION_FINAL_EVENT_MAGIC;
    event->abi_version = SPARK_RUNTIME_COMPLETION_ABI_VERSION;
    event->descriptor_bytes = SPARK_RUNTIME_FINAL_EVENT_BYTES;
    event->state = SPARK_RUNTIME_FINAL_EVENT_STATE_PENDING;
    event->event_generation = queue->next_event_generation;
    queue->next_event_generation += 1u;
    if (queue->next_event_generation == 0u)
    {
        queue->next_event_generation = 1u;
    }
    event->identity = request->identity;
    event->payload_fingerprint = payload_fingerprint;
    event->request_id = request->request_id;
    event->sequence_id = request->sequence_id;
    event->sequence_position = request->sequence_position;
    event->status = status;
    queue->pending_count += 1u;
    *event_generation_out = event->event_generation;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRuntimeFindFinalEvent(
    const SparkRuntimeFinalEventQueue *queue,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    const SparkRuntimeFinalEvent **event_out)
{
    uint32_t event_index;

    if (event_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *event_out = 0;
    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        const SparkRuntimeFinalEvent *event;

        event = &queue->events[event_index];
        if (event->state != SPARK_RUNTIME_FINAL_EVENT_STATE_FREE &&
            SparkWorkTransactionIdentitiesMatch(
                &event->identity,
                identity) != 0u &&
            event->payload_fingerprint == payload_fingerprint)
        {
            *event_out = event;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}

static SparkStatus SparkRuntimeAcquireCredit(
    SparkRuntimeController *controller,
    uint32_t domain)
{
    SparkStatus status;

    status = SparkWorkTransactionAcquireCredits(
        &controller->credit_ledger,
        domain,
        1u);
    return status == SPARK_STATUS_CAPACITY_EXCEEDED ?
        SPARK_STATUS_BUSY : status;
}

static void SparkRuntimeReleaseCreditIfHeld(
    SparkRuntimeController *controller,
    uint32_t domain,
    uint32_t *held)
{
    if (*held != 0u)
    {
        (void)SparkWorkTransactionReleaseCredits(
            &controller->credit_ledger,
            domain,
            1u);
        *held = 0u;
    }
}

static void SparkRuntimeCancelPreparedParticipants(
    SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    uint32_t prepared_participant_count,
    SparkStatus reason,
    SparkRuntimeTransactionResult *result)
{
    while (prepared_participant_count != 0u)
    {
        SparkRuntimeParticipant *participant;

        prepared_participant_count -= 1u;
        participant = &controller->participants[prepared_participant_count];
        participant->cancel(
            participant->participant_context,
            request,
            reason);
        result->cancelled_participant_count += 1u;
    }
}

SparkStatus SparkRuntimeInitializeFinalEventQueue(
    SparkRuntimeFinalEventQueue *queue,
    SparkRuntimeFinalEvent *events,
    uint32_t capacity)
{
    if (queue == 0 || events == 0 || capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(queue,0,sizeof(*queue));
    memset(events,0,(size_t)capacity * sizeof(events[0u]));
    queue->abi_version = SPARK_RUNTIME_COMPLETION_ABI_VERSION;
    queue->descriptor_bytes = SPARK_RUNTIME_FINAL_EVENT_QUEUE_BYTES;
    queue->capacity = capacity;
    queue->next_event_generation = 1u;
    queue->events = events;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRuntimePeekFinalEvent(
    const SparkRuntimeFinalEventQueue *queue,
    const SparkRuntimeFinalEvent **event_out)
{
    const SparkRuntimeFinalEvent *oldest_event;
    uint64_t oldest_generation;
    uint32_t event_index;

    if (event_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *event_out = 0;
    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    oldest_event = 0;
    oldest_generation = UINT64_MAX;
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        const SparkRuntimeFinalEvent *event;

        event = &queue->events[event_index];
        if (event->state == SPARK_RUNTIME_FINAL_EVENT_STATE_PENDING &&
            event->event_generation < oldest_generation)
        {
            oldest_event = event;
            oldest_generation = event->event_generation;
        }
    }
    if (oldest_event == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    *event_out = oldest_event;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRuntimeAcknowledgeFinalEvent(
    SparkRuntimeFinalEventQueue *queue,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint)
{
    uint32_t event_index;

    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        event_generation == 0u ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        SparkRuntimeFinalEvent *event;

        event = &queue->events[event_index];
        if (event->event_generation == event_generation &&
            SparkWorkTransactionIdentitiesMatch(
                &event->identity,
                identity) != 0u)
        {
            if (event->payload_fingerprint != payload_fingerprint)
            {
                return SPARK_STATUS_VALIDATION_FAILED;
            }
            if (event->state == SPARK_RUNTIME_FINAL_EVENT_STATE_ACKNOWLEDGED)
            {
                return SPARK_STATUS_DUPLICATE;
            }
            if (event->state != SPARK_RUNTIME_FINAL_EVENT_STATE_PENDING ||
                queue->pending_count == 0u)
            {
                return SPARK_STATUS_VALIDATION_FAILED;
            }
            event->state = SPARK_RUNTIME_FINAL_EVENT_STATE_ACKNOWLEDGED;
            queue->pending_count -= 1u;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}

SparkStatus SparkRuntimeAcknowledgeControllerFinalEvent(
    SparkRuntimeController *controller,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint)
{
    SparkStatus status;

    if (controller == 0 ||
        controller->abi_version != SPARK_RUNTIME_COMPLETION_ABI_VERSION ||
        controller->descriptor_bytes != SPARK_RUNTIME_CONTROLLER_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkRuntimeAcknowledgeFinalEvent(
        &controller->final_event_queue,
        identity,
        event_generation,
        payload_fingerprint);
    if (status == SPARK_STATUS_DUPLICATE)
    {
        return status;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkWorkTransactionReleaseCredits(
        &controller->credit_ledger,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COMPLETION_OWNERSHIP,
        1u);
}

SparkStatus SparkRuntimeInitializeTransmissionWindow(
    SparkRuntimeTransmissionWindow *window,
    SparkRuntimeTransmissionSlot *slots,
    uint32_t capacity)
{
    uint32_t slot_index;

    if (window == 0 || slots == 0 || capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(window,0,sizeof(*window));
    memset(slots,0,(size_t)capacity * sizeof(slots[0u]));
    window->abi_version = SPARK_RUNTIME_COMPLETION_ABI_VERSION;
    window->descriptor_bytes = SPARK_RUNTIME_TRANSMISSION_WINDOW_BYTES;
    window->capacity = capacity;
    window->next_slot_generation = 1u;
    window->slots = slots;
    for (slot_index = 0u; slot_index < capacity; ++slot_index)
    {
        slots[slot_index].slot_index = slot_index;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRuntimeReserveTransmissionSlot(
    SparkRuntimeTransmissionWindow *window,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out)
{
    uint32_t slot_index;

    if (slot_index_out == 0 || slot_generation_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *slot_index_out = SPARK_RUNTIME_COMPLETION_INVALID_INDEX;
    *slot_generation_out = 0u;
    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (window->in_use_count >= window->capacity)
    {
        return SPARK_STATUS_BUSY;
    }
    for (slot_index = 0u; slot_index < window->capacity; ++slot_index)
    {
        SparkRuntimeTransmissionSlot *slot;

        slot = &window->slots[slot_index];
        if (slot->state == SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_FREE ||
            slot->state == SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_ACKNOWLEDGED)
        {
            uint64_t generation;

            generation = window->next_slot_generation;
            window->next_slot_generation += 1u;
            if (window->next_slot_generation == 0u)
            {
                window->next_slot_generation = 1u;
            }
            memset(slot,0,sizeof(*slot));
            slot->state = SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_RESERVED;
            slot->slot_index = slot_index;
            slot->slot_generation = generation;
            slot->identity = *identity;
            slot->payload_fingerprint = payload_fingerprint;
            window->in_use_count += 1u;
            *slot_index_out = slot_index;
            *slot_generation_out = generation;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

SparkStatus SparkRuntimeMarkTransmissionSent(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation)
{
    SparkRuntimeTransmissionSlot *slot;

    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK ||
        slot_index >= window->capacity ||
        slot_generation == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    slot = &window->slots[slot_index];
    if (slot->slot_generation != slot_generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (slot->state == SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_SENT)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    if (slot->state != SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_RESERVED)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    window->send_epoch += 1u;
    if (window->send_epoch == 0u)
    {
        window->send_epoch = 1u;
    }
    slot->state = SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_SENT;
    slot->last_send_epoch = window->send_epoch;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRuntimeAcknowledgeTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint)
{
    SparkRuntimeTransmissionSlot *slot;

    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK ||
        slot_index >= window->capacity ||
        slot_generation == 0u ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    slot = &window->slots[slot_index];
    if (slot->slot_generation != slot_generation ||
        SparkWorkTransactionIdentitiesMatch(&slot->identity,identity) == 0u ||
        slot->payload_fingerprint != payload_fingerprint)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (slot->state == SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_ACKNOWLEDGED)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    if (slot->state != SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_SENT ||
        window->in_use_count == 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    slot->state = SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_ACKNOWLEDGED;
    window->in_use_count -= 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRuntimeNextReplayTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out)
{
    SparkRuntimeTransmissionSlot *oldest_slot;
    uint64_t oldest_epoch;
    uint32_t slot_index;

    if (slot_index_out == 0 || slot_generation_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *slot_index_out = SPARK_RUNTIME_COMPLETION_INVALID_INDEX;
    *slot_generation_out = 0u;
    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    oldest_slot = 0;
    oldest_epoch = UINT64_MAX;
    for (slot_index = 0u; slot_index < window->capacity; ++slot_index)
    {
        SparkRuntimeTransmissionSlot *slot;

        slot = &window->slots[slot_index];
        if (slot->state == SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_SENT &&
            slot->last_send_epoch < oldest_epoch)
        {
            oldest_slot = slot;
            oldest_epoch = slot->last_send_epoch;
        }
    }
    if (oldest_slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    window->send_epoch += 1u;
    if (window->send_epoch == 0u)
    {
        window->send_epoch = 1u;
    }
    oldest_slot->last_send_epoch = window->send_epoch;
    oldest_slot->retry_count += 1u;
    if (oldest_slot->retry_count == 0u)
    {
        oldest_slot->retry_count = UINT32_MAX;
    }
    *slot_index_out = oldest_slot->slot_index;
    *slot_generation_out = oldest_slot->slot_generation;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRuntimeInitializeController(
    SparkRuntimeController *controller,
    SparkRuntimeParticipant *participants,
    uint32_t participant_count,
    SparkWorkTransactionEntry *transaction_entries,
    uint32_t transaction_capacity,
    SparkRuntimeFinalEvent *final_events,
    uint32_t final_event_capacity,
    const uint32_t credit_capacities[
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT])
{
    SparkStatus status;
    uint32_t participant_index;

    if (controller == 0 || participants == 0 ||
        participant_count == 0u ||
        participant_count > SPARK_RUNTIME_COMPLETION_MAX_PARTICIPANTS ||
        transaction_entries == 0 || transaction_capacity == 0u ||
        final_events == 0 || final_event_capacity == 0u ||
        credit_capacities == 0 ||
        credit_capacities[
            SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COMPLETION_OWNERSHIP] !=
            final_event_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (participant_index = 0u;
         participant_index < participant_count;
         ++participant_index)
    {
        status = SparkRuntimeValidateParticipant(
            &participants[participant_index],
            participant_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    memset(controller,0,sizeof(*controller));
    controller->abi_version = SPARK_RUNTIME_COMPLETION_ABI_VERSION;
    controller->descriptor_bytes = SPARK_RUNTIME_CONTROLLER_BYTES;
    controller->participant_count = participant_count;
    controller->participants = participants;
    status = SparkWorkTransactionInitializeLedger(
        &controller->transaction_ledger,
        transaction_entries,
        transaction_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWorkTransactionInitializeCreditLedger(
        &controller->credit_ledger,
        credit_capacities);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkRuntimeInitializeFinalEventQueue(
        &controller->final_event_queue,
        final_events,
        final_event_capacity);
}

SparkStatus SparkRuntimeValidateTransactionRequest(
    const SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    uint64_t *payload_fingerprint_out)
{
    uint64_t payload_fingerprint;

    if (payload_fingerprint_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *payload_fingerprint_out = 0u;
    if (controller == 0 ||
        controller->abi_version != SPARK_RUNTIME_COMPLETION_ABI_VERSION ||
        controller->descriptor_bytes != SPARK_RUNTIME_CONTROLLER_BYTES ||
        controller->participant_count == 0u ||
        controller->participants == 0 ||
        request == 0 ||
        request->descriptor_bytes != SPARK_RUNTIME_TRANSACTION_REQUEST_BYTES ||
        request->participant_count != controller->participant_count ||
        SparkWorkTransactionValidateIdentity(&request->identity) !=
            SPARK_STATUS_OK ||
        request->request_id == 0u ||
        request->sequence_id == 0u ||
        request->payload == 0 ||
        request->payload_bytes == 0u ||
        request->reserved0 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    {
        uint32_t participant_index;

        for (participant_index = 0u;
             participant_index < controller->participant_count;
             ++participant_index)
        {
            if (request->identity.control_generation <
                controller->participants[participant_index].restart_epoch)
            {
                return SPARK_STATUS_VALIDATION_FAILED;
            }
        }
    }
    payload_fingerprint = SparkWorkTransactionFingerprintBytes(
        request->payload,
        request->payload_bytes);
    if (payload_fingerprint == 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    *payload_fingerprint_out = payload_fingerprint;
    return SPARK_STATUS_OK;
}

SparkStatus SparkRuntimeRunTransaction(
    SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    SparkRuntimeTransactionResult *result)
{
    SparkWorkTransactionEntry *transaction_entry;
    const SparkRuntimeFinalEvent *existing_event;
    uint64_t payload_fingerprint;
    uint32_t observation;
    uint32_t participant_index;
    uint32_t prepared_participant_count;
    uint32_t transport_credit_held;
    uint32_t reservation_credit_held;
    uint32_t execution_credit_held;
    uint32_t completion_credit_held;
    uint32_t terminal_state;
    SparkStatus status;

    if (result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(result,0,sizeof(*result));
    result->descriptor_bytes = SPARK_RUNTIME_TRANSACTION_RESULT_BYTES;
    result->status = SPARK_STATUS_PENDING;
    status = SparkRuntimeValidateTransactionRequest(
        controller,
        request,
        &payload_fingerprint);
    if (status != SPARK_STATUS_OK)
    {
        result->status = status;
        return status;
    }
    result->payload_fingerprint = payload_fingerprint;
    status = SparkWorkTransactionObserve(
        &controller->transaction_ledger,
        &request->identity,
        payload_fingerprint,
        &transaction_entry,
        &observation);
    if (status != SPARK_STATUS_OK)
    {
        result->status = status;
        return status;
    }
    if (observation != SPARK_WORK_TRANSACTION_OBSERVATION_NEW)
    {
        result->replayed = 1u;
        if (observation == SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_ACTIVE)
        {
            result->status = SPARK_STATUS_BUSY;
            return SPARK_STATUS_BUSY;
        }
        result->status = (SparkStatus)transaction_entry->terminal_status;
        if (SparkRuntimeFindFinalEvent(
                &controller->final_event_queue,
                &request->identity,
                payload_fingerprint,
                &existing_event) == SPARK_STATUS_OK)
        {
            result->final_event_generation =
                existing_event->event_generation;
        }
        return SPARK_STATUS_DUPLICATE;
    }

    prepared_participant_count = 0u;
    transport_credit_held = 0u;
    reservation_credit_held = 0u;
    execution_credit_held = 0u;
    completion_credit_held = 0u;
    terminal_state = SPARK_WORK_TRANSACTION_STATE_FAILED;

    status = SparkRuntimeAcquireCredit(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = SPARK_WORK_TRANSACTION_STATE_CANCELLED;
        goto fail_transaction;
    }
    transport_credit_held = 1u;

    status = SparkRuntimeAcquireCredit(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = SPARK_WORK_TRANSACTION_STATE_CANCELLED;
        goto fail_transaction;
    }
    reservation_credit_held = 1u;

    status = SparkRuntimeAcquireCredit(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COMPLETION_OWNERSHIP);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = SPARK_WORK_TRANSACTION_STATE_CANCELLED;
        goto fail_transaction;
    }
    completion_credit_held = 1u;

    for (participant_index = 0u;
         participant_index < controller->participant_count;
         ++participant_index)
    {
        SparkRuntimeParticipant *participant;

        participant = &controller->participants[participant_index];
        status = participant->prepare(
            participant->participant_context,
            request);
        if (status != SPARK_STATUS_OK)
        {
            goto fail_transaction;
        }
        prepared_participant_count += 1u;
    }

    status = SparkWorkTransactionTransition(
        &controller->transaction_ledger,
        &request->identity,
        SPARK_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING);
    if (status != SPARK_STATUS_OK)
    {
        goto fail_transaction;
    }

    status = SparkRuntimeAcquireCredit(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = SPARK_WORK_TRANSACTION_STATE_CANCELLED;
        goto fail_transaction;
    }
    execution_credit_held = 1u;

    status = SparkWorkTransactionTransition(
        &controller->transaction_ledger,
        &request->identity,
        SPARK_WORK_TRANSACTION_STATE_EXECUTING,
        SPARK_STATUS_PENDING);
    if (status != SPARK_STATUS_OK)
    {
        goto fail_transaction;
    }

    for (participant_index = 0u;
         participant_index < controller->participant_count;
         ++participant_index)
    {
        SparkRuntimeParticipant *participant;

        participant = &controller->participants[participant_index];
        status = participant->execute(
            participant->participant_context,
            request);
        if (status != SPARK_STATUS_OK)
        {
            goto fail_transaction;
        }
    }

    for (participant_index = 0u;
         participant_index < controller->participant_count;
         ++participant_index)
    {
        SparkRuntimeParticipant *participant;

        participant = &controller->participants[participant_index];
        participant->commit(
            participant->participant_context,
            request);
        result->committed_participant_count += 1u;
    }
    status = SparkWorkTransactionTransition(
        &controller->transaction_ledger,
        &request->identity,
        SPARK_WORK_TRANSACTION_STATE_COMMITTED,
        SPARK_STATUS_OK);
    if (status != SPARK_STATUS_OK)
    {
        result->status = SPARK_STATUS_INTERNAL_ERROR;
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION,
            &execution_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
            &reservation_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW,
            &transport_credit_held);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkRuntimeEnqueueFinalEvent(
        &controller->final_event_queue,
        request,
        payload_fingerprint,
        SPARK_STATUS_OK,
        &result->final_event_generation);
    if (status != SPARK_STATUS_OK)
    {
        result->status = SPARK_STATUS_INTERNAL_ERROR;
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION,
            &execution_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
            &reservation_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW,
            &transport_credit_held);
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    SparkRuntimeReleaseCreditIfHeld(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION,
        &execution_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
        &reservation_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW,
        &transport_credit_held);
    result->status = SPARK_STATUS_OK;
    return SPARK_STATUS_OK;

fail_transaction:
    SparkRuntimeCancelPreparedParticipants(
        controller,
        request,
        prepared_participant_count,
        status,
        result);
    if (SparkWorkTransactionTransition(
            &controller->transaction_ledger,
            &request->identity,
            terminal_state,
            status) != SPARK_STATUS_OK)
    {
        status = SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION,
        &execution_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COMPLETION_OWNERSHIP,
        &completion_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
        &reservation_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW,
        &transport_credit_held);
    result->status = status;
    return status;
}
