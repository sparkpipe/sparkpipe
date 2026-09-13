#ifndef SPARKPIPE_SPARK_RUNTIME_COMPLETION_H
#define SPARKPIPE_SPARK_RUNTIME_COMPLETION_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_work_transaction.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_RUNTIME_COMPLETION_ABI_VERSION 1u
#define SPARK_RUNTIME_COMPLETION_MAX_PARTICIPANTS 32u
#define SPARK_RUNTIME_COMPLETION_FINAL_EVENT_MAGIC UINT32_C(0x544E5645)
#define SPARK_RUNTIME_COMPLETION_INVALID_INDEX UINT32_MAX

#define SPARK_RUNTIME_PARTICIPANT_FLAG_TRANSACTIONAL_STAGING UINT32_C(0x00000001)
#define SPARK_RUNTIME_PARTICIPANT_FLAG_COMMIT_INFALLIBLE UINT32_C(0x00000002)
#define SPARK_RUNTIME_PARTICIPANT_FLAG_CANCEL_IDEMPOTENT UINT32_C(0x00000004)
#define SPARK_RUNTIME_PARTICIPANT_REQUIRED_FLAGS \
    (SPARK_RUNTIME_PARTICIPANT_FLAG_TRANSACTIONAL_STAGING | \
     SPARK_RUNTIME_PARTICIPANT_FLAG_COMMIT_INFALLIBLE | \
     SPARK_RUNTIME_PARTICIPANT_FLAG_CANCEL_IDEMPOTENT)
#define SPARK_RUNTIME_PARTICIPANT_KNOWN_FLAGS \
    SPARK_RUNTIME_PARTICIPANT_REQUIRED_FLAGS

#define SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_FREE 0u
#define SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_RESERVED 1u
#define SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_SENT 2u
#define SPARK_RUNTIME_TRANSMISSION_SLOT_STATE_ACKNOWLEDGED 3u

#define SPARK_RUNTIME_FINAL_EVENT_STATE_FREE 0u
#define SPARK_RUNTIME_FINAL_EVENT_STATE_PENDING 1u
#define SPARK_RUNTIME_FINAL_EVENT_STATE_ACKNOWLEDGED 2u

typedef struct SparkRuntimeTransactionRequest
{
    uint32_t descriptor_bytes;
    uint32_t participant_count;
    SparkWorkTransactionIdentity identity;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    const void *payload;
    uint32_t payload_bytes;
    uint32_t reserved0;
} SparkRuntimeTransactionRequest;

typedef SparkStatus (*SparkRuntimeParticipantPrepareFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request);
typedef SparkStatus (*SparkRuntimeParticipantExecuteFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request);
typedef void (*SparkRuntimeParticipantCommitFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request);
typedef void (*SparkRuntimeParticipantCancelFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request,
    SparkStatus reason);

typedef struct SparkRuntimeParticipant
{
    uint32_t descriptor_bytes;
    uint32_t participant_index;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t restart_epoch;
    const char *participant_name;
    void *participant_context;
    SparkRuntimeParticipantPrepareFunction prepare;
    SparkRuntimeParticipantExecuteFunction execute;
    SparkRuntimeParticipantCommitFunction commit;
    SparkRuntimeParticipantCancelFunction cancel;
} SparkRuntimeParticipant;

typedef struct SparkRuntimeFinalEvent
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint64_t event_generation;
    SparkWorkTransactionIdentity identity;
    uint64_t payload_fingerprint;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    SparkStatus status;
    uint32_t reserved0;
} SparkRuntimeFinalEvent;

typedef struct SparkRuntimeFinalEventQueue
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capacity;
    uint32_t pending_count;
    uint64_t next_event_generation;
    SparkRuntimeFinalEvent *events;
} SparkRuntimeFinalEventQueue;

typedef struct SparkRuntimeTransmissionSlot
{
    uint32_t state;
    uint32_t slot_index;
    uint64_t slot_generation;
    SparkWorkTransactionIdentity identity;
    uint64_t payload_fingerprint;
    uint64_t last_send_epoch;
    uint32_t retry_count;
    uint32_t reserved0;
} SparkRuntimeTransmissionSlot;

typedef struct SparkRuntimeTransmissionWindow
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capacity;
    uint32_t in_use_count;
    uint64_t next_slot_generation;
    uint64_t send_epoch;
    SparkRuntimeTransmissionSlot *slots;
} SparkRuntimeTransmissionWindow;

typedef struct SparkRuntimeController
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t participant_count;
    uint32_t reserved0;
    SparkRuntimeParticipant *participants;
    SparkWorkTransactionLedger transaction_ledger;
    SparkWorkTransactionCreditLedger credit_ledger;
    SparkRuntimeFinalEventQueue final_event_queue;
} SparkRuntimeController;

typedef struct SparkRuntimeTransactionResult
{
    uint32_t descriptor_bytes;
    uint32_t replayed;
    uint32_t committed_participant_count;
    uint32_t cancelled_participant_count;
    uint64_t payload_fingerprint;
    uint64_t final_event_generation;
    SparkStatus status;
    uint32_t reserved0;
} SparkRuntimeTransactionResult;

#define SPARK_RUNTIME_TRANSACTION_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkRuntimeTransactionRequest))
#define SPARK_RUNTIME_PARTICIPANT_BYTES \
    ((uint32_t)sizeof(SparkRuntimeParticipant))
#define SPARK_RUNTIME_FINAL_EVENT_BYTES \
    ((uint32_t)sizeof(SparkRuntimeFinalEvent))
#define SPARK_RUNTIME_FINAL_EVENT_QUEUE_BYTES \
    ((uint32_t)sizeof(SparkRuntimeFinalEventQueue))
#define SPARK_RUNTIME_TRANSMISSION_SLOT_BYTES \
    ((uint32_t)sizeof(SparkRuntimeTransmissionSlot))
#define SPARK_RUNTIME_TRANSMISSION_WINDOW_BYTES \
    ((uint32_t)sizeof(SparkRuntimeTransmissionWindow))
#define SPARK_RUNTIME_CONTROLLER_BYTES \
    ((uint32_t)sizeof(SparkRuntimeController))
#define SPARK_RUNTIME_TRANSACTION_RESULT_BYTES \
    ((uint32_t)sizeof(SparkRuntimeTransactionResult))

SparkStatus SparkRuntimeInitializeFinalEventQueue(
    SparkRuntimeFinalEventQueue *queue,
    SparkRuntimeFinalEvent *events,
    uint32_t capacity);
SparkStatus SparkRuntimePeekFinalEvent(
    const SparkRuntimeFinalEventQueue *queue,
    const SparkRuntimeFinalEvent **event_out);
SparkStatus SparkRuntimeAcknowledgeFinalEvent(
    SparkRuntimeFinalEventQueue *queue,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint);
SparkStatus SparkRuntimeAcknowledgeControllerFinalEvent(
    SparkRuntimeController *controller,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint);

SparkStatus SparkRuntimeInitializeTransmissionWindow(
    SparkRuntimeTransmissionWindow *window,
    SparkRuntimeTransmissionSlot *slots,
    uint32_t capacity);
SparkStatus SparkRuntimeReserveTransmissionSlot(
    SparkRuntimeTransmissionWindow *window,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out);
SparkStatus SparkRuntimeMarkTransmissionSent(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation);
SparkStatus SparkRuntimeAcknowledgeTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint);
SparkStatus SparkRuntimeNextReplayTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out);

SparkStatus SparkRuntimeInitializeController(
    SparkRuntimeController *controller,
    SparkRuntimeParticipant *participants,
    uint32_t participant_count,
    SparkWorkTransactionEntry *transaction_entries,
    uint32_t transaction_capacity,
    SparkRuntimeFinalEvent *final_events,
    uint32_t final_event_capacity,
    const uint32_t credit_capacities[
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT]);
SparkStatus SparkRuntimeValidateTransactionRequest(
    const SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    uint64_t *payload_fingerprint_out);
SparkStatus SparkRuntimeRunTransaction(
    SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    SparkRuntimeTransactionResult *result);

#ifdef __cplusplus
}
#endif

#endif
