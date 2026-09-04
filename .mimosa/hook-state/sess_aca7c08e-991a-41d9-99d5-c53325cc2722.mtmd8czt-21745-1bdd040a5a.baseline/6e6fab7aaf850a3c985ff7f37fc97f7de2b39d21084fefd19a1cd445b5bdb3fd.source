#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_runtime_completion.h"

#define TEST_PARTICIPANT_COUNT 3u
#define TEST_TRANSACTION_CAPACITY 16u
#define TEST_FINAL_EVENT_CAPACITY 4u
#define TEST_TRANSMISSION_CAPACITY 4u

typedef struct TestParticipantContext
{
    uint32_t prepare_count;
    uint32_t execute_count;
    uint32_t commit_count;
    uint32_t cancel_count;
    SparkStatus prepare_status;
    SparkStatus execute_status;
    SparkStatus last_cancel_reason;
} TestParticipantContext;

static int TestExpect(
    int condition,
    const char *message)
{
    if (!condition)
    {
        fprintf(stderr,"FAIL: %s\n",message);
        return 0;
    }
    return 1;
}

static SparkStatus TestParticipantPrepare(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request)
{
    TestParticipantContext *context;

    context = (TestParticipantContext *)participant_context;
    if (context == 0 || request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context->prepare_count += 1u;
    return context->prepare_status;
}

static SparkStatus TestParticipantExecute(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request)
{
    TestParticipantContext *context;

    context = (TestParticipantContext *)participant_context;
    if (context == 0 || request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context->execute_count += 1u;
    return context->execute_status;
}

static void TestParticipantCommit(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request)
{
    TestParticipantContext *context;

    context = (TestParticipantContext *)participant_context;
    if (context != 0 && request != 0)
    {
        context->commit_count += 1u;
    }
}

static void TestParticipantCancel(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request,
    SparkStatus reason)
{
    TestParticipantContext *context;

    context = (TestParticipantContext *)participant_context;
    if (context != 0 && request != 0)
    {
        context->cancel_count += 1u;
        context->last_cancel_reason = reason;
    }
}

static void TestInitializeIdentity(
    SparkWorkTransactionIdentity *identity,
    uint64_t transaction_id)
{
    memset(identity,0,sizeof(*identity));
    identity->control_generation = 7u;
    identity->transaction_id = transaction_id;
    identity->dispatch_generation = 11u;
    identity->request_generation = 13u;
    identity->step_generation = transaction_id;
    identity->step_chunk_index = 0u;
    identity->step_chunk_count = 1u;
    identity->phase = SPARK_WORK_TRANSACTION_PHASE_DECODE;
}

static void TestInitializeRequest(
    SparkRuntimeTransactionRequest *request,
    uint64_t transaction_id,
    const void *payload,
    uint32_t payload_bytes)
{
    memset(request,0,sizeof(*request));
    request->descriptor_bytes = SPARK_RUNTIME_TRANSACTION_REQUEST_BYTES;
    request->participant_count = TEST_PARTICIPANT_COUNT;
    TestInitializeIdentity(&request->identity,transaction_id);
    request->request_id = 101u + transaction_id;
    request->sequence_id = 201u + transaction_id;
    request->sequence_position = transaction_id;
    request->payload = payload;
    request->payload_bytes = payload_bytes;
}

static int TestInitializeController(
    SparkRuntimeController *controller,
    SparkRuntimeParticipant participants[TEST_PARTICIPANT_COUNT],
    TestParticipantContext contexts[TEST_PARTICIPANT_COUNT],
    SparkWorkTransactionEntry transaction_entries[TEST_TRANSACTION_CAPACITY],
    SparkRuntimeFinalEvent final_events[TEST_FINAL_EVENT_CAPACITY])
{
    static const char *participant_names[TEST_PARTICIPANT_COUNT] =
    {
        "rank0",
        "rank1",
        "rank2"
    };
    uint32_t credit_capacities[SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT];
    uint32_t participant_index;
    SparkStatus status;

    memset(contexts,0,
        sizeof(contexts[0u]) * TEST_PARTICIPANT_COUNT);
    memset(participants,0,
        sizeof(participants[0u]) * TEST_PARTICIPANT_COUNT);
    for (participant_index = 0u;
         participant_index < TEST_PARTICIPANT_COUNT;
         ++participant_index)
    {
        contexts[participant_index].prepare_status = SPARK_STATUS_OK;
        contexts[participant_index].execute_status = SPARK_STATUS_OK;
        participants[participant_index].descriptor_bytes =
            SPARK_RUNTIME_PARTICIPANT_BYTES;
        participants[participant_index].participant_index = participant_index;
        participants[participant_index].flags =
            SPARK_RUNTIME_PARTICIPANT_REQUIRED_FLAGS;
        participants[participant_index].restart_epoch = 1u;
        participants[participant_index].participant_name =
            participant_names[participant_index];
        participants[participant_index].participant_context =
            &contexts[participant_index];
        participants[participant_index].prepare = TestParticipantPrepare;
        participants[participant_index].execute = TestParticipantExecute;
        participants[participant_index].commit = TestParticipantCommit;
        participants[participant_index].cancel = TestParticipantCancel;
    }
    credit_capacities[SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW] =
        TEST_FINAL_EVENT_CAPACITY;
    credit_capacities[
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION] =
        TEST_FINAL_EVENT_CAPACITY;
    credit_capacities[SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION] =
        TEST_FINAL_EVENT_CAPACITY;
    credit_capacities[
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COMPLETION_OWNERSHIP] =
        TEST_FINAL_EVENT_CAPACITY;
    status = SparkRuntimeInitializeController(
        controller,
        participants,
        TEST_PARTICIPANT_COUNT,
        transaction_entries,
        TEST_TRANSACTION_CAPACITY,
        final_events,
        TEST_FINAL_EVENT_CAPACITY,
        credit_capacities);
    return TestExpect(status == SPARK_STATUS_OK,"controller initialization");
}

static int TestSuccessfulTransactionAndReplay(void)
{
    static const uint8_t payload[] = {1u,2u,3u,4u};
    SparkRuntimeController controller;
    SparkRuntimeParticipant participants[TEST_PARTICIPANT_COUNT];
    TestParticipantContext contexts[TEST_PARTICIPANT_COUNT];
    SparkWorkTransactionEntry transaction_entries[TEST_TRANSACTION_CAPACITY];
    SparkRuntimeFinalEvent final_events[TEST_FINAL_EVENT_CAPACITY];
    SparkRuntimeTransactionRequest request;
    SparkRuntimeTransactionResult result;
    const SparkRuntimeFinalEvent *event;
    SparkStatus status;
    uint32_t participant_index;

    if (!TestInitializeController(
            &controller,
            participants,
            contexts,
            transaction_entries,
            final_events))
    {
        return 0;
    }
    TestInitializeRequest(
        &request,
        1u,
        payload,
        (uint32_t)sizeof(payload));
    status = SparkRuntimeRunTransaction(&controller,&request,&result);
    if (!TestExpect(status == SPARK_STATUS_OK,"successful transaction") ||
        !TestExpect(result.status == SPARK_STATUS_OK,"successful result") ||
        !TestExpect(result.committed_participant_count ==
            TEST_PARTICIPANT_COUNT,"all participants committed") ||
        !TestExpect(result.final_event_generation != 0u,
            "final event generation"))
    {
        return 0;
    }
    for (participant_index = 0u;
         participant_index < TEST_PARTICIPANT_COUNT;
         ++participant_index)
    {
        if (!TestExpect(contexts[participant_index].prepare_count == 1u,
                "participant prepared once") ||
            !TestExpect(contexts[participant_index].execute_count == 1u,
                "participant executed once") ||
            !TestExpect(contexts[participant_index].commit_count == 1u,
                "participant committed once") ||
            !TestExpect(contexts[participant_index].cancel_count == 0u,
                "participant not cancelled"))
        {
            return 0;
        }
    }
    status = SparkRuntimePeekFinalEvent(
        &controller.final_event_queue,
        &event);
    if (!TestExpect(status == SPARK_STATUS_OK,"peek final event") ||
        !TestExpect(event->event_generation == result.final_event_generation,
            "final event identity") ||
        !TestExpect(event->status == SPARK_STATUS_OK,"final event status"))
    {
        return 0;
    }
    status = SparkRuntimeRunTransaction(&controller,&request,&result);
    if (!TestExpect(status == SPARK_STATUS_DUPLICATE,
            "terminal replay is duplicate") ||
        !TestExpect(result.replayed == 1u,"replay marked") ||
        !TestExpect(result.status == SPARK_STATUS_OK,
            "replay retains terminal status"))
    {
        return 0;
    }
    for (participant_index = 0u;
         participant_index < TEST_PARTICIPANT_COUNT;
         ++participant_index)
    {
        if (!TestExpect(contexts[participant_index].execute_count == 1u,
                "replay did not execute twice"))
        {
            return 0;
        }
    }
    status = SparkRuntimeAcknowledgeControllerFinalEvent(
        &controller,
        &request.identity,
        event->event_generation,
        event->payload_fingerprint);
    if (!TestExpect(status == SPARK_STATUS_OK,"final event acknowledgement") ||
        !TestExpect(controller.final_event_queue.pending_count == 0u,
            "final event ownership released"))
    {
        return 0;
    }
    status = SparkRuntimeAcknowledgeControllerFinalEvent(
        &controller,
        &request.identity,
        event->event_generation,
        event->payload_fingerprint);
    return TestExpect(status == SPARK_STATUS_DUPLICATE,
        "duplicate final event acknowledgement");
}

static int TestPrepareFailureCancelsPreparedRanks(void)
{
    static const uint8_t payload[] = {9u,8u,7u};
    SparkRuntimeController controller;
    SparkRuntimeParticipant participants[TEST_PARTICIPANT_COUNT];
    TestParticipantContext contexts[TEST_PARTICIPANT_COUNT];
    SparkWorkTransactionEntry transaction_entries[TEST_TRANSACTION_CAPACITY];
    SparkRuntimeFinalEvent final_events[TEST_FINAL_EVENT_CAPACITY];
    SparkRuntimeTransactionRequest request;
    SparkRuntimeTransactionResult result;
    SparkStatus status;

    if (!TestInitializeController(
            &controller,
            participants,
            contexts,
            transaction_entries,
            final_events))
    {
        return 0;
    }
    contexts[1u].prepare_status = SPARK_STATUS_CAPACITY_EXCEEDED;
    TestInitializeRequest(
        &request,
        2u,
        payload,
        (uint32_t)sizeof(payload));
    status = SparkRuntimeRunTransaction(&controller,&request,&result);
    return TestExpect(status == SPARK_STATUS_CAPACITY_EXCEEDED,
            "prepare failure returned") &&
        TestExpect(contexts[0u].cancel_count == 1u,
            "prepared rank cancelled") &&
        TestExpect(contexts[1u].cancel_count == 0u,
            "failed prepare owns its cleanup") &&
        TestExpect(contexts[2u].prepare_count == 0u,
            "later rank not prepared") &&
        TestExpect(controller.final_event_queue.pending_count == 0u,
            "failed transaction emitted no final event");
}

static int TestExecuteFailureCancelsAllPreparedRanks(void)
{
    static const uint8_t payload[] = {5u,4u,3u,2u,1u};
    SparkRuntimeController controller;
    SparkRuntimeParticipant participants[TEST_PARTICIPANT_COUNT];
    TestParticipantContext contexts[TEST_PARTICIPANT_COUNT];
    SparkWorkTransactionEntry transaction_entries[TEST_TRANSACTION_CAPACITY];
    SparkRuntimeFinalEvent final_events[TEST_FINAL_EVENT_CAPACITY];
    SparkRuntimeTransactionRequest request;
    SparkRuntimeTransactionResult result;
    SparkStatus status;
    uint32_t participant_index;

    if (!TestInitializeController(
            &controller,
            participants,
            contexts,
            transaction_entries,
            final_events))
    {
        return 0;
    }
    contexts[1u].execute_status = SPARK_STATUS_IO_ERROR;
    TestInitializeRequest(
        &request,
        3u,
        payload,
        (uint32_t)sizeof(payload));
    status = SparkRuntimeRunTransaction(&controller,&request,&result);
    if (!TestExpect(status == SPARK_STATUS_IO_ERROR,
            "execute failure returned") ||
        !TestExpect(result.cancelled_participant_count ==
            TEST_PARTICIPANT_COUNT,"all staged ranks cancelled"))
    {
        return 0;
    }
    for (participant_index = 0u;
         participant_index < TEST_PARTICIPANT_COUNT;
         ++participant_index)
    {
        if (!TestExpect(contexts[participant_index].cancel_count == 1u,
                "each staged rank cancelled") ||
            !TestExpect(contexts[participant_index].commit_count == 0u,
                "no partial commit"))
        {
            return 0;
        }
    }
    return 1;
}

static int TestRejectsStaleControlGeneration(void)
{
    static const uint8_t payload[] = {4u,2u};
    SparkRuntimeController controller;
    SparkRuntimeParticipant participants[TEST_PARTICIPANT_COUNT];
    TestParticipantContext contexts[TEST_PARTICIPANT_COUNT];
    SparkWorkTransactionEntry transaction_entries[TEST_TRANSACTION_CAPACITY];
    SparkRuntimeFinalEvent final_events[TEST_FINAL_EVENT_CAPACITY];
    SparkRuntimeTransactionRequest request;
    SparkRuntimeTransactionResult result;
    SparkStatus status;

    if (!TestInitializeController(
            &controller,
            participants,
            contexts,
            transaction_entries,
            final_events))
    {
        return 0;
    }
    participants[2u].restart_epoch = 8u;
    TestInitializeRequest(
        &request,
        4u,
        payload,
        (uint32_t)sizeof(payload));
    status = SparkRuntimeRunTransaction(&controller,&request,&result);
    return TestExpect(status == SPARK_STATUS_VALIDATION_FAILED,
        "pre-restart transaction rejected");
}

static int TestTransmissionWindow(void)
{
    static const uint8_t payload[] = {6u,6u,6u};
    SparkRuntimeTransmissionWindow window;
    SparkRuntimeTransmissionSlot slots[TEST_TRANSMISSION_CAPACITY];
    SparkWorkTransactionIdentity identity;
    uint64_t payload_fingerprint;
    uint64_t slot_generation;
    uint64_t replay_generation;
    uint32_t slot_index;
    uint32_t replay_index;
    SparkStatus status;

    TestInitializeIdentity(&identity,9u);
    payload_fingerprint = SparkWorkTransactionFingerprintBytes(
        payload,
        (uint32_t)sizeof(payload));
    status = SparkRuntimeInitializeTransmissionWindow(
        &window,
        slots,
        TEST_TRANSMISSION_CAPACITY);
    if (!TestExpect(status == SPARK_STATUS_OK,"window initialization"))
    {
        return 0;
    }
    status = SparkRuntimeReserveTransmissionSlot(
        &window,
        &identity,
        payload_fingerprint,
        &slot_index,
        &slot_generation);
    if (!TestExpect(status == SPARK_STATUS_OK,"reserve transmission") ||
        !TestExpect(slot_generation != 0u,"slot generation"))
    {
        return 0;
    }
    status = SparkRuntimeMarkTransmissionSent(
        &window,
        slot_index,
        slot_generation);
    if (!TestExpect(status == SPARK_STATUS_OK,"mark sent"))
    {
        return 0;
    }
    status = SparkRuntimeNextReplayTransmission(
        &window,
        &replay_index,
        &replay_generation);
    if (!TestExpect(status == SPARK_STATUS_OK,"select replay") ||
        !TestExpect(replay_index == slot_index,"replay slot") ||
        !TestExpect(replay_generation == slot_generation,
            "replay generation"))
    {
        return 0;
    }
    status = SparkRuntimeAcknowledgeTransmission(
        &window,
        slot_index,
        slot_generation,
        &identity,
        payload_fingerprint);
    if (!TestExpect(status == SPARK_STATUS_OK,"ack transmission") ||
        !TestExpect(window.in_use_count == 0u,"window ownership released"))
    {
        return 0;
    }
    status = SparkRuntimeAcknowledgeTransmission(
        &window,
        slot_index,
        slot_generation,
        &identity,
        payload_fingerprint);
    return TestExpect(status == SPARK_STATUS_DUPLICATE,
        "duplicate transmission acknowledgement");
}

int main(void)
{
    if (!TestSuccessfulTransactionAndReplay() ||
        !TestPrepareFailureCancelsPreparedRanks() ||
        !TestExecuteFailureCancelsAllPreparedRanks() ||
        !TestRejectsStaleControlGeneration() ||
        !TestTransmissionWindow())
    {
        return 1;
    }
    printf("PASS runtime completion controller and selective ACK window\n");
    return 0;
}
