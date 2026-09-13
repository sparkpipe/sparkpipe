#pragma once

#include "sparkpipe/spark_work_transaction.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DISTRIBUTED_WORK_PROTOCOL_ABI_VERSION \
    SPARK_WORK_TRANSACTION_ABI_VERSION
#define SPARK_DISTRIBUTED_WORK_ACK_MAGIC \
    SPARK_WORK_TRANSACTION_ACK_MAGIC
#define SPARK_DISTRIBUTED_WORK_INVALID_INDEX \
    SPARK_WORK_TRANSACTION_INVALID_INDEX

#define SPARK_DISTRIBUTED_WORK_PHASE_PREFILL \
    SPARK_WORK_TRANSACTION_PHASE_PREFILL
#define SPARK_DISTRIBUTED_WORK_PHASE_DECODE \
    SPARK_WORK_TRANSACTION_PHASE_DECODE
#define SPARK_DISTRIBUTED_WORK_PHASE_VERIFY \
    SPARK_WORK_TRANSACTION_PHASE_VERIFY
#define SPARK_DISTRIBUTED_WORK_PHASE_RELEASE \
    SPARK_WORK_TRANSACTION_PHASE_RELEASE
#define SPARK_DISTRIBUTED_WORK_PHASE_CANCEL \
    SPARK_WORK_TRANSACTION_PHASE_CANCEL
#define SPARK_DISTRIBUTED_WORK_PHASE_KNOWN_MAX \
    SPARK_WORK_TRANSACTION_PHASE_KNOWN_MAX

#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FREE \
    SPARK_WORK_TRANSACTION_STATE_EMPTY
#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_PREPARED \
    SPARK_WORK_TRANSACTION_STATE_PREPARED
#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED \
    SPARK_WORK_TRANSACTION_STATE_ACCEPTED
#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FORWARDING \
    SPARK_WORK_TRANSACTION_STATE_ACCEPTED
#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING \
    SPARK_WORK_TRANSACTION_STATE_EXECUTING
#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED \
    SPARK_WORK_TRANSACTION_STATE_COMMITTED
#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FAILED \
    SPARK_WORK_TRANSACTION_STATE_FAILED
#define SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_CANCELLED \
    SPARK_WORK_TRANSACTION_STATE_CANCELLED

#define SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_TRANSPORT_WINDOW \
    SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW
#define SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_RESIDENT_RESERVATION \
    SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION
#define SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_EXECUTION \
    SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION
#define SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_COMPLETION_OWNERSHIP \
    SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COMPLETION_OWNERSHIP
#define SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_COUNT \
    SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT

#define SPARK_DISTRIBUTED_WORK_ACKNOWLEDGEMENT_BYTES \
    SPARK_WORK_TRANSACTION_ACKNOWLEDGEMENT_BYTES

typedef SparkWorkTransactionIdentity SparkDistributedWorkIdentity;
typedef SparkWorkTransactionAcknowledgement \
    SparkDistributedWorkAcknowledgement;
typedef SparkWorkTransactionEntry SparkDistributedWorkTransactionEntry;
typedef SparkWorkTransactionLedger SparkDistributedWorkTransactionLedger;
typedef SparkWorkTransactionCreditDomain SparkDistributedWorkCreditDomain;
typedef SparkWorkTransactionCreditLedger SparkDistributedWorkCreditLedger;

static inline uint32_t SparkDistributedWorkPhaseIsValid(
    uint32_t transaction_phase)
{
    return SparkWorkTransactionPhaseIsValid(transaction_phase);
}

static inline uint64_t SparkDistributedWorkHashBytes(
    const void *data,
    uint32_t data_bytes)
{
    return SparkWorkTransactionFingerprintBytes(data,data_bytes);
}

static inline uint32_t SparkDistributedWorkIdentityIsValid(
    const SparkDistributedWorkIdentity *identity)
{
    return SparkWorkTransactionValidateIdentity(identity) == SPARK_STATUS_OK;
}

static inline uint32_t SparkDistributedWorkIdentityMatches(
    const SparkDistributedWorkIdentity *left,
    const SparkDistributedWorkIdentity *right)
{
    return SparkWorkTransactionIdentitiesMatch(left,right);
}

static inline void SparkDistributedWorkInitializeAcknowledgement(
    SparkDistributedWorkAcknowledgement *acknowledgement,
    const SparkDistributedWorkIdentity *identity,
    uint64_t packet_hash,
    SparkStatus status)
{
    uint32_t transaction_state;

    transaction_state = status == SPARK_STATUS_OK ||
        status == SPARK_STATUS_DUPLICATE ?
        SPARK_WORK_TRANSACTION_STATE_ACCEPTED :
        SPARK_WORK_TRANSACTION_STATE_FAILED;
    SparkWorkTransactionInitializeAcknowledgement(
        acknowledgement,
        identity,
        packet_hash,
        transaction_state,
        status);
}

static inline SparkStatus SparkDistributedWorkValidateAcknowledgement(
    const SparkDistributedWorkAcknowledgement *acknowledgement,
    const SparkDistributedWorkIdentity *expected_identity,
    uint64_t expected_packet_hash)
{
    return SparkWorkTransactionValidateAcknowledgement(
        acknowledgement,
        expected_identity,
        expected_packet_hash);
}

static inline SparkStatus SparkDistributedWorkInitializeTransactionLedger(
    SparkDistributedWorkTransactionLedger *ledger,
    SparkDistributedWorkTransactionEntry *entries,
    uint32_t *hash_heads,
    uint32_t *hash_next,
    uint32_t capacity)
{
    (void)hash_heads;
    (void)hash_next;
    return SparkWorkTransactionInitializeLedger(ledger,entries,capacity);
}

static inline SparkStatus SparkDistributedWorkObserveTransaction(
    SparkDistributedWorkTransactionLedger *ledger,
    const SparkDistributedWorkIdentity *identity,
    uint64_t packet_hash,
    uint32_t *existing_state_out,
    SparkStatus *terminal_status_out)
{
    SparkWorkTransactionEntry *entry;
    SparkStatus status;
    uint32_t observation;

    status = SparkWorkTransactionObserve(
        ledger,
        identity,
        packet_hash,
        &entry,
        &observation);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (existing_state_out != 0)
    {
        *existing_state_out = observation ==
            SPARK_WORK_TRANSACTION_OBSERVATION_NEW ?
            SPARK_WORK_TRANSACTION_STATE_EMPTY : entry->state;
    }
    if (terminal_status_out != 0)
    {
        *terminal_status_out = (SparkStatus)entry->terminal_status;
    }
    return observation == SPARK_WORK_TRANSACTION_OBSERVATION_NEW ?
        SPARK_STATUS_OK : SPARK_STATUS_DUPLICATE;
}

static inline SparkStatus SparkDistributedWorkTransitionTransaction(
    SparkDistributedWorkTransactionLedger *ledger,
    const SparkDistributedWorkIdentity *identity,
    uint64_t packet_hash,
    uint32_t next_state,
    SparkStatus terminal_status)
{
    SparkWorkTransactionEntry *entry;
    SparkStatus status;

    status = SparkWorkTransactionFind(ledger,identity,&entry);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (entry->payload_fingerprint != packet_hash)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    status = SparkWorkTransactionTransition(
        ledger,
        identity,
        next_state,
        terminal_status);
    return status == SPARK_STATUS_INVALID_ARGUMENT ?
        SPARK_STATUS_VALIDATION_FAILED : status;
}

static inline SparkStatus SparkDistributedWorkFindTransaction(
    const SparkDistributedWorkTransactionLedger *ledger,
    const SparkDistributedWorkIdentity *identity,
    uint64_t packet_hash,
    const SparkDistributedWorkTransactionEntry **entry_out)
{
    SparkWorkTransactionEntry *entry;
    SparkStatus status;

    if (entry_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkWorkTransactionFind(
        (SparkWorkTransactionLedger *)ledger,
        identity,
        &entry);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (entry->payload_fingerprint != packet_hash)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    *entry_out = entry;
    return SPARK_STATUS_OK;
}

static inline SparkStatus SparkDistributedWorkInitializeCreditLedger(
    SparkDistributedWorkCreditLedger *ledger,
    const uint32_t capacities[SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_COUNT])
{
    return SparkWorkTransactionInitializeCreditLedger(ledger,capacities);
}

static inline SparkStatus SparkDistributedWorkAcquireCredits(
    SparkDistributedWorkCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count)
{
    SparkStatus status;

    status = SparkWorkTransactionAcquireCredits(ledger,domain,credit_count);
    return status == SPARK_STATUS_CAPACITY_EXCEEDED ?
        SPARK_STATUS_BUSY : status;
}

static inline SparkStatus SparkDistributedWorkReleaseCredits(
    SparkDistributedWorkCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count)
{
    return SparkWorkTransactionReleaseCredits(ledger,domain,credit_count);
}

static inline uint32_t SparkDistributedWorkAvailableCredits(
    const SparkDistributedWorkCreditLedger *ledger,
    uint32_t domain)
{
    return SparkWorkTransactionAvailableCredits(ledger,domain);
}

#ifdef __cplusplus
}
#endif
