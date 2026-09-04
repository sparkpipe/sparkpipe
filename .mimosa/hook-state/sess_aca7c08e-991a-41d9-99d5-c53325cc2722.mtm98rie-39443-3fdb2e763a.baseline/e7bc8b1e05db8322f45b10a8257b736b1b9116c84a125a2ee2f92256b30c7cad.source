#ifndef SPARKPIPE_SPARK_WORK_TRANSACTION_H
#define SPARKPIPE_SPARK_WORK_TRANSACTION_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WORK_TRANSACTION_ABI_VERSION 2u
#define SPARK_WORK_TRANSACTION_ACK_MAGIC UINT32_C(0x4B415457)
#define SPARK_WORK_TRANSACTION_INVALID_INDEX UINT32_MAX

#define SPARK_WORK_TRANSACTION_PHASE_PREFILL 1u
#define SPARK_WORK_TRANSACTION_PHASE_DECODE 2u
#define SPARK_WORK_TRANSACTION_PHASE_VERIFY 3u
#define SPARK_WORK_TRANSACTION_PHASE_RELEASE 4u
#define SPARK_WORK_TRANSACTION_PHASE_CANCEL 5u
#define SPARK_WORK_TRANSACTION_PHASE_KNOWN_MAX \
    SPARK_WORK_TRANSACTION_PHASE_CANCEL

#define SPARK_WORK_TRANSACTION_STATE_EMPTY 0u
#define SPARK_WORK_TRANSACTION_STATE_PREPARED 1u
#define SPARK_WORK_TRANSACTION_STATE_ACCEPTED 2u
#define SPARK_WORK_TRANSACTION_STATE_EXECUTING 3u
#define SPARK_WORK_TRANSACTION_STATE_COMMITTED 4u
#define SPARK_WORK_TRANSACTION_STATE_CANCELLED 5u
#define SPARK_WORK_TRANSACTION_STATE_FAILED 6u
#define SPARK_WORK_TRANSACTION_STATE_TOMBSTONE 7u

#define SPARK_WORK_TRANSACTION_OBSERVATION_NEW 1u
#define SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_ACTIVE 2u
#define SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_TERMINAL 3u
#define SPARK_WORK_TRANSACTION_OBSERVATION_CONFLICT 4u

#define SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_TRANSPORT_WINDOW 0u
#define SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION 1u
#define SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_EXECUTION 2u
#define SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COMPLETION_OWNERSHIP 3u
#define SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT 4u

#define SPARK_WORK_TRANSACTION_FINGERPRINT_OFFSET \
    UINT64_C(14695981039346656037)
#define SPARK_WORK_TRANSACTION_FINGERPRINT_PRIME \
    UINT64_C(1099511628211)

typedef struct SparkWorkTransactionIdentity
{
    uint64_t control_generation;
    uint64_t transaction_id;
    uint64_t dispatch_generation;
    uint64_t request_generation;
    uint64_t step_generation;
    uint32_t step_chunk_index;
    uint32_t step_chunk_count;
    union
    {
        uint32_t phase;
        uint32_t transaction_phase;
    };
    uint32_t reserved0;
} SparkWorkTransactionIdentity;

typedef struct SparkWorkTransactionAcknowledgement
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t status;
    uint32_t transaction_state;
    uint32_t reserved0;
    SparkWorkTransactionIdentity identity;
    uint64_t packet_fingerprint;
    uint64_t reserved1;
} SparkWorkTransactionAcknowledgement;

typedef struct SparkWorkTransactionEntry
{
    SparkWorkTransactionIdentity identity;
    union
    {
        uint64_t payload_fingerprint;
        uint64_t packet_hash;
    };
    uint64_t last_observed_epoch;
    uint32_t state;
    uint32_t terminal_status;
} SparkWorkTransactionEntry;

typedef struct SparkWorkTransactionLedger
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t entry_capacity;
    uint32_t entry_count;
    uint32_t active_entry_count;
    uint32_t terminal_entry_count;
    uint32_t tombstone_entry_count;
    uint32_t reserved0;
    uint64_t epoch;
    uint64_t active_control_generation;
    SparkWorkTransactionEntry *entries;
} SparkWorkTransactionLedger;

typedef struct SparkWorkTransactionCreditDomain
{
    uint32_t capacity;
    uint32_t in_use;
} SparkWorkTransactionCreditDomain;

typedef struct SparkWorkTransactionCreditLedger
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkWorkTransactionCreditDomain domains[
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT];
} SparkWorkTransactionCreditLedger;

#define SPARK_WORK_TRANSACTION_IDENTITY_BYTES \
    ((uint32_t)sizeof(SparkWorkTransactionIdentity))
#define SPARK_WORK_TRANSACTION_ACKNOWLEDGEMENT_BYTES \
    ((uint32_t)sizeof(SparkWorkTransactionAcknowledgement))
#define SPARK_WORK_TRANSACTION_ENTRY_BYTES \
    ((uint32_t)sizeof(SparkWorkTransactionEntry))
#define SPARK_WORK_TRANSACTION_LEDGER_BYTES \
    ((uint32_t)sizeof(SparkWorkTransactionLedger))

uint32_t SparkWorkTransactionPhaseIsValid(
    uint32_t phase);
uint32_t SparkWorkTransactionStateIsTerminal(
    uint32_t state);
uint64_t SparkWorkTransactionFingerprintBytes(
    const void *data,
    uint32_t data_bytes);
uint32_t SparkWorkTransactionIdentitiesMatch(
    const SparkWorkTransactionIdentity *left,
    const SparkWorkTransactionIdentity *right);
SparkStatus SparkWorkTransactionValidateIdentity(
    const SparkWorkTransactionIdentity *identity);
SparkStatus SparkWorkTransactionInitializeLedger(
    SparkWorkTransactionLedger *ledger,
    SparkWorkTransactionEntry *entries,
    uint32_t entry_capacity);
SparkStatus SparkWorkTransactionObserve(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    SparkWorkTransactionEntry **entry_out,
    uint32_t *observation_out);
SparkStatus SparkWorkTransactionTransition(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    uint32_t target_state,
    SparkStatus terminal_status);
SparkStatus SparkWorkTransactionFind(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    SparkWorkTransactionEntry **entry_out);
void SparkWorkTransactionInitializeAcknowledgement(
    SparkWorkTransactionAcknowledgement *acknowledgement,
    const SparkWorkTransactionIdentity *identity,
    uint64_t packet_fingerprint,
    uint32_t transaction_state,
    SparkStatus status);
SparkStatus SparkWorkTransactionValidateAcknowledgement(
    const SparkWorkTransactionAcknowledgement *acknowledgement,
    const SparkWorkTransactionIdentity *expected_identity,
    uint64_t expected_packet_fingerprint);
SparkStatus SparkWorkTransactionInitializeCreditLedger(
    SparkWorkTransactionCreditLedger *ledger,
    const uint32_t capacities[SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT]);
SparkStatus SparkWorkTransactionAcquireCredits(
    SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count);
SparkStatus SparkWorkTransactionReleaseCredits(
    SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count);
uint32_t SparkWorkTransactionAvailableCredits(
    const SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain);

#ifdef __cplusplus
}
#endif

#endif
