#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_distributed_work.h"

#define SPARK_TEST_LEDGER_CAPACITY 8u

static SparkDistributedWorkIdentity SparkTestIdentity(
    uint64_t transaction_id,
    uint64_t dispatch_generation,
    uint64_t request_generation,
    uint64_t step_generation,
    uint32_t chunk_index,
    uint32_t phase)
{
    SparkDistributedWorkIdentity identity;

    memset(&identity,0,sizeof(identity));
    identity.control_generation = 17u;
    identity.transaction_id = transaction_id;
    identity.dispatch_generation = dispatch_generation;
    identity.request_generation = request_generation;
    identity.step_generation = step_generation;
    identity.step_chunk_index = chunk_index;
    identity.step_chunk_count = chunk_index + 1u;
    identity.phase = phase;
    return identity;
}

static void SparkTestInitializeLedger(
    SparkDistributedWorkTransactionLedger *ledger,
    SparkDistributedWorkTransactionEntry entries[SPARK_TEST_LEDGER_CAPACITY])
{
    assert(SparkDistributedWorkInitializeTransactionLedger(
        ledger,
        entries,
        0,
        0,
        SPARK_TEST_LEDGER_CAPACITY) == SPARK_STATUS_OK);
}

static void SparkTestExactReplayIsIdempotent(void)
{
    SparkDistributedWorkTransactionEntry entries[SPARK_TEST_LEDGER_CAPACITY];
    SparkDistributedWorkTransactionLedger ledger;
    SparkDistributedWorkIdentity identity;
    uint32_t existing_state;
    SparkStatus terminal_status;

    SparkTestInitializeLedger(&ledger,entries);
    identity = SparkTestIdentity(
        31u,
        41u,
        45u,
        51u,
        0u,
        SPARK_DISTRIBUTED_WORK_PHASE_DECODE);
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &identity,
        101u,
        &existing_state,
        &terminal_status) == SPARK_STATUS_OK);
    assert(existing_state == SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FREE);
    assert(ledger.entry_count == 1u);
    assert(ledger.active_entry_count == 1u);
    assert(ledger.terminal_entry_count == 0u);
    assert(SparkDistributedWorkTransitionTransaction(
        &ledger,
        &identity,
        101u,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkTransitionTransaction(
        &ledger,
        &identity,
        101u,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkTransitionTransaction(
        &ledger,
        &identity,
        101u,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED,
        SPARK_STATUS_OK) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &identity,
        101u,
        &existing_state,
        &terminal_status) == SPARK_STATUS_DUPLICATE);
    assert(existing_state ==
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED);
    assert(terminal_status == SPARK_STATUS_OK);
    assert(ledger.active_entry_count == 0u);
    assert(ledger.terminal_entry_count == 1u);
}

static void SparkTestConflictingReplayIsRejected(void)
{
    SparkDistributedWorkTransactionEntry entries[SPARK_TEST_LEDGER_CAPACITY];
    SparkDistributedWorkTransactionLedger ledger;
    SparkDistributedWorkIdentity identity;

    SparkTestInitializeLedger(&ledger,entries);
    identity = SparkTestIdentity(
        32u,
        42u,
        46u,
        52u,
        1u,
        SPARK_DISTRIBUTED_WORK_PHASE_PREFILL);
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &identity,
        201u,
        0,
        0) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &identity,
        202u,
        0,
        0) == SPARK_STATUS_VALIDATION_FAILED);
}

static void SparkTestIllegalTransitionIsRejected(void)
{
    SparkDistributedWorkTransactionEntry entries[SPARK_TEST_LEDGER_CAPACITY];
    SparkDistributedWorkTransactionLedger ledger;
    SparkDistributedWorkIdentity identity;

    SparkTestInitializeLedger(&ledger,entries);
    identity = SparkTestIdentity(
        33u,
        43u,
        47u,
        53u,
        0u,
        SPARK_DISTRIBUTED_WORK_PHASE_VERIFY);
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &identity,
        301u,
        0,
        0) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkTransitionTransaction(
        &ledger,
        &identity,
        301u,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED,
        SPARK_STATUS_OK) == SPARK_STATUS_VALIDATION_FAILED);
    assert(SparkDistributedWorkTransitionTransaction(
        &ledger,
        &identity,
        301u,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkTransitionTransaction(
        &ledger,
        &identity,
        301u,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_PREPARED,
        SPARK_STATUS_PENDING) == SPARK_STATUS_VALIDATION_FAILED);
}

static void SparkTestGenerationChangeWaitsForActiveTransactions(void)
{
    SparkDistributedWorkTransactionEntry entries[SPARK_TEST_LEDGER_CAPACITY];
    SparkDistributedWorkTransactionLedger ledger;
    SparkDistributedWorkIdentity first;
    SparkDistributedWorkIdentity second;

    SparkTestInitializeLedger(&ledger,entries);
    first = SparkTestIdentity(
        34u,
        44u,
        48u,
        54u,
        0u,
        SPARK_DISTRIBUTED_WORK_PHASE_DECODE);
    second = SparkTestIdentity(
        35u,
        45u,
        49u,
        55u,
        0u,
        SPARK_DISTRIBUTED_WORK_PHASE_DECODE);
    second.control_generation = 18u;
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &first,
        401u,
        0,
        0) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &second,
        402u,
        0,
        0) == SPARK_STATUS_BUSY);
    assert(SparkDistributedWorkTransitionTransaction(
        &ledger,
        &first,
        401u,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FAILED,
        SPARK_STATUS_INTERNAL_ERROR) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkObserveTransaction(
        &ledger,
        &second,
        402u,
        0,
        0) == SPARK_STATUS_OK);
    assert(ledger.active_control_generation == 18u);
    assert(ledger.entry_count == 1u);
}

static void SparkTestAcknowledgementBindsIdentityAndBytes(void)
{
    SparkDistributedWorkAcknowledgement acknowledgement;
    SparkDistributedWorkIdentity identity;

    identity = SparkTestIdentity(
        36u,
        46u,
        50u,
        56u,
        0u,
        SPARK_DISTRIBUTED_WORK_PHASE_RELEASE);
    SparkDistributedWorkInitializeAcknowledgement(
        &acknowledgement,
        &identity,
        501u,
        SPARK_STATUS_OK);
    assert(SparkDistributedWorkValidateAcknowledgement(
        &acknowledgement,
        &identity,
        501u) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkValidateAcknowledgement(
        &acknowledgement,
        &identity,
        502u) == SPARK_STATUS_VALIDATION_FAILED);
    acknowledgement.identity.step_generation += 1u;
    assert(SparkDistributedWorkValidateAcknowledgement(
        &acknowledgement,
        &identity,
        501u) == SPARK_STATUS_VALIDATION_FAILED);
}

static void SparkTestCreditDomainsCannotDoubleRelease(void)
{
    static const uint32_t capacities[
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_COUNT] =
    {
        2u,
        3u,
        4u,
        5u
    };
    SparkDistributedWorkCreditLedger ledger;

    assert(SparkDistributedWorkInitializeCreditLedger(
        &ledger,
        capacities) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkAcquireCredits(
        &ledger,
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_TRANSPORT_WINDOW,
        2u) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkAcquireCredits(
        &ledger,
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_TRANSPORT_WINDOW,
        1u) == SPARK_STATUS_BUSY);
    assert(SparkDistributedWorkReleaseCredits(
        &ledger,
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_TRANSPORT_WINDOW,
        2u) == SPARK_STATUS_OK);
    assert(SparkDistributedWorkReleaseCredits(
        &ledger,
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_TRANSPORT_WINDOW,
        1u) == SPARK_STATUS_VALIDATION_FAILED);
    assert(SparkDistributedWorkAvailableCredits(
        &ledger,
        SPARK_DISTRIBUTED_WORK_CREDIT_DOMAIN_TRANSPORT_WINDOW) == 2u);
}

int main(void)
{
    SparkTestExactReplayIsIdempotent();
    SparkTestConflictingReplayIsRejected();
    SparkTestIllegalTransitionIsRejected();
    SparkTestGenerationChangeWaitsForActiveTransactions();
    SparkTestAcknowledgementBindsIdentityAndBytes();
    SparkTestCreditDomainsCannotDoubleRelease();
    return 0;
}
