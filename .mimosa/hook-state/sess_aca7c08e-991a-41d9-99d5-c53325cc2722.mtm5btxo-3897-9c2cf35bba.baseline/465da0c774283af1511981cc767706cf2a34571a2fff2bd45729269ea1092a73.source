#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_work_transaction.h"

static SparkWorkTransactionIdentity SparkTestIdentity(
    uint64_t transaction_id,
    uint64_t request_generation,
    uint64_t step_generation,
    uint32_t chunk_index)
{
    SparkWorkTransactionIdentity identity;

    memset(&identity,0,sizeof(identity));
    identity.control_generation = 17u;
    identity.transaction_id = transaction_id;
    identity.dispatch_generation = transaction_id + 1000u;
    identity.request_generation = request_generation;
    identity.step_generation = step_generation;
    identity.step_chunk_index = chunk_index;
    identity.step_chunk_count = chunk_index + 1u;
    identity.phase = SPARK_WORK_TRANSACTION_PHASE_DECODE;
    return identity;
}

static void SparkTestAdvanceToCommitted(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity)
{
    assert(SparkWorkTransactionTransition(
        ledger,
        identity,
        SPARK_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkWorkTransactionTransition(
        ledger,
        identity,
        SPARK_WORK_TRANSACTION_STATE_EXECUTING,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkWorkTransactionTransition(
        ledger,
        identity,
        SPARK_WORK_TRANSACTION_STATE_COMMITTED,
        SPARK_STATUS_OK) == SPARK_STATUS_OK);
}

static void SparkTestReplayAndConflict(void)
{
    SparkWorkTransactionEntry entries[4];
    SparkWorkTransactionIdentity identity;
    SparkWorkTransactionLedger ledger;
    SparkWorkTransactionEntry *entry;
    uint32_t observation;

    identity = SparkTestIdentity(101u,201u,301u,0u);
    assert(SparkWorkTransactionInitializeLedger(
        &ledger,entries,4u) == SPARK_STATUS_OK);
    assert(SparkWorkTransactionObserve(
        &ledger,&identity,401u,&entry,&observation) == SPARK_STATUS_OK);
    assert(observation == SPARK_WORK_TRANSACTION_OBSERVATION_NEW);
    SparkTestAdvanceToCommitted(&ledger,&identity);
    assert(SparkWorkTransactionObserve(
        &ledger,&identity,401u,&entry,&observation) == SPARK_STATUS_OK);
    assert(observation ==
        SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_TERMINAL);
    assert(SparkWorkTransactionObserve(
        &ledger,&identity,402u,&entry,&observation) ==
        SPARK_STATUS_VALIDATION_FAILED);
    assert(observation == SPARK_WORK_TRANSACTION_OBSERVATION_CONFLICT);
}

static void SparkTestTombstoneDoesNotBreakProbeChain(void)
{
    enum
    {
        SparkTestCapacity = 4,
        SparkTestIdentityCount = 5
    };
    SparkWorkTransactionEntry entries[SparkTestCapacity];
    SparkWorkTransactionIdentity identities[SparkTestIdentityCount];
    SparkWorkTransactionLedger ledger;
    SparkWorkTransactionEntry *entry;
    uint32_t bucket;
    uint32_t candidate;
    uint32_t identity_count;
    uint32_t identity_index;
    uint32_t observation;

    bucket = UINT32_MAX;
    identity_count = 0u;
    for (candidate = 1u;
         candidate < 100000u && identity_count < SparkTestIdentityCount;
         ++candidate)
    {
        SparkWorkTransactionIdentity identity;
        uint32_t candidate_bucket;

        identity = SparkTestIdentity(
            1000u + candidate,
            2000u + candidate,
            3000u + candidate,
            candidate);
        candidate_bucket = (uint32_t)(SparkWorkTransactionFingerprintBytes(
            &identity,
            (uint32_t)sizeof(identity)) % SparkTestCapacity);
        if (bucket == UINT32_MAX)
        {
            bucket = candidate_bucket;
        }
        if (candidate_bucket == bucket)
        {
            identities[identity_count] = identity;
            identity_count += 1u;
        }
    }
    assert(identity_count == SparkTestIdentityCount);
    assert(SparkWorkTransactionInitializeLedger(
        &ledger,entries,SparkTestCapacity) == SPARK_STATUS_OK);
    for (identity_index = 0u;
         identity_index < SparkTestCapacity;
         ++identity_index)
    {
        assert(SparkWorkTransactionObserve(
            &ledger,
            &identities[identity_index],
            5000u + identity_index,
            &entry,
            &observation) == SPARK_STATUS_OK);
    }
    assert(SparkWorkTransactionTransition(
        &ledger,
        &identities[0u],
        SPARK_WORK_TRANSACTION_STATE_FAILED,
        SPARK_STATUS_INTERNAL_ERROR) == SPARK_STATUS_OK);
    assert(SparkWorkTransactionObserve(
        &ledger,
        &identities[4u],
        5004u,
        &entry,
        &observation) == SPARK_STATUS_OK);
    for (identity_index = 1u;
         identity_index < SparkTestIdentityCount;
         ++identity_index)
    {
        assert(SparkWorkTransactionFind(
            &ledger,
            &identities[identity_index],
            &entry) == SPARK_STATUS_OK);
    }
}

static void SparkTestAcknowledgementAndCredits(void)
{
    static const uint32_t capacities[
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT] =
    {
        2u,
        2u,
        2u,
        2u
    };
    SparkWorkTransactionAcknowledgement acknowledgement;
    SparkWorkTransactionCreditLedger credit_ledger;
    SparkWorkTransactionIdentity identity;

    identity = SparkTestIdentity(701u,801u,901u,0u);
    SparkWorkTransactionInitializeAcknowledgement(
        &acknowledgement,
        &identity,
        1001u,
        SPARK_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_OK);
    assert(SparkWorkTransactionValidateAcknowledgement(
        &acknowledgement,&identity,1001u) == SPARK_STATUS_OK);
    acknowledgement.identity.step_generation += 1u;
    assert(SparkWorkTransactionValidateAcknowledgement(
        &acknowledgement,&identity,1001u) ==
        SPARK_STATUS_VALIDATION_FAILED);

    assert(SparkWorkTransactionInitializeCreditLedger(
        &credit_ledger,capacities) == SPARK_STATUS_OK);
    assert(SparkWorkTransactionAcquireCredits(
        &credit_ledger,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
        2u) == SPARK_STATUS_OK);
    assert(SparkWorkTransactionAcquireCredits(
        &credit_ledger,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
        1u) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(SparkWorkTransactionReleaseCredits(
        &credit_ledger,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
        2u) == SPARK_STATUS_OK);
    assert(SparkWorkTransactionReleaseCredits(
        &credit_ledger,
        SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_RESIDENT_RESERVATION,
        1u) == SPARK_STATUS_VALIDATION_FAILED);
}

int main(void)
{
    SparkTestReplayAndConflict();
    SparkTestTombstoneDoesNotBreakProbeChain();
    SparkTestAcknowledgementAndCredits();
    return 0;
}
