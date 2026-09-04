#include "sparkpipe/spark_work_transaction.h"

#include <string.h>

static SparkStatus SparkWorkTransactionValidateLedger(
    const SparkWorkTransactionLedger *ledger)
{
    if (ledger == 0 ||
        ledger->abi_version != SPARK_WORK_TRANSACTION_ABI_VERSION ||
        ledger->descriptor_bytes != SPARK_WORK_TRANSACTION_LEDGER_BYTES ||
        ledger->entry_capacity == 0u ||
        ledger->entries == 0 ||
        ledger->entry_count > ledger->entry_capacity ||
        ledger->active_entry_count > ledger->entry_count ||
        ledger->terminal_entry_count > ledger->entry_count ||
        ledger->tombstone_entry_count > ledger->entry_capacity ||
        ledger->active_entry_count + ledger->terminal_entry_count !=
            ledger->entry_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkWorkTransactionStateIsStored(
    uint32_t state)
{
    return state >= SPARK_WORK_TRANSACTION_STATE_PREPARED &&
        state <= SPARK_WORK_TRANSACTION_STATE_FAILED;
}

static uint32_t SparkWorkTransactionTransitionIsLegal(
    uint32_t current_state,
    uint32_t target_state)
{
    if (current_state == target_state)
    {
        return 1u;
    }
    if (current_state == SPARK_WORK_TRANSACTION_STATE_PREPARED)
    {
        return target_state == SPARK_WORK_TRANSACTION_STATE_ACCEPTED ||
            target_state == SPARK_WORK_TRANSACTION_STATE_CANCELLED ||
            target_state == SPARK_WORK_TRANSACTION_STATE_FAILED;
    }
    if (current_state == SPARK_WORK_TRANSACTION_STATE_ACCEPTED)
    {
        return target_state == SPARK_WORK_TRANSACTION_STATE_EXECUTING ||
            target_state == SPARK_WORK_TRANSACTION_STATE_CANCELLED ||
            target_state == SPARK_WORK_TRANSACTION_STATE_FAILED;
    }
    if (current_state == SPARK_WORK_TRANSACTION_STATE_EXECUTING)
    {
        return target_state == SPARK_WORK_TRANSACTION_STATE_COMMITTED ||
            target_state == SPARK_WORK_TRANSACTION_STATE_CANCELLED ||
            target_state == SPARK_WORK_TRANSACTION_STATE_FAILED;
    }
    return 0u;
}

static uint32_t SparkWorkTransactionIdentityStartIndex(
    const SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity)
{
    uint64_t fingerprint;

    fingerprint = SparkWorkTransactionFingerprintBytes(
        identity,
        SPARK_WORK_TRANSACTION_IDENTITY_BYTES);
    return (uint32_t)(fingerprint % ledger->entry_capacity);
}

static SparkStatus SparkWorkTransactionFindInternal(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    SparkWorkTransactionEntry **entry_out,
    uint32_t *entry_index_out)
{
    uint32_t entry_index;
    uint32_t probe_index;
    uint32_t start_index;

    start_index = SparkWorkTransactionIdentityStartIndex(ledger,identity);
    for (probe_index = 0u;
         probe_index < ledger->entry_capacity;
         ++probe_index)
    {
        SparkWorkTransactionEntry *entry;

        entry_index = (start_index + probe_index) % ledger->entry_capacity;
        entry = &ledger->entries[entry_index];
        if (entry->state == SPARK_WORK_TRANSACTION_STATE_EMPTY)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        if (entry->state != SPARK_WORK_TRANSACTION_STATE_TOMBSTONE &&
            SparkWorkTransactionIdentitiesMatch(
                &entry->identity,
                identity) != 0u)
        {
            if (entry_out != 0)
            {
                *entry_out = entry;
            }
            if (entry_index_out != 0)
            {
                *entry_index_out = entry_index;
            }
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}

static uint32_t SparkWorkTransactionHasActiveEntries(
    const SparkWorkTransactionLedger *ledger)
{
    return ledger->active_entry_count != 0u;
}

static void SparkWorkTransactionResetLedger(
    SparkWorkTransactionLedger *ledger,
    uint64_t control_generation)
{
    memset(
        ledger->entries,
        0,
        (size_t)ledger->entry_capacity * sizeof(ledger->entries[0u]));
    ledger->entry_count = 0u;
    ledger->active_entry_count = 0u;
    ledger->terminal_entry_count = 0u;
    ledger->tombstone_entry_count = 0u;
    ledger->epoch = 0u;
    ledger->active_control_generation = control_generation;
}

static SparkStatus SparkWorkTransactionPrepareControlGeneration(
    SparkWorkTransactionLedger *ledger,
    uint64_t control_generation)
{
    if (ledger->active_control_generation == 0u)
    {
        SparkWorkTransactionResetLedger(ledger,control_generation);
        return SPARK_STATUS_OK;
    }
    if (control_generation == ledger->active_control_generation)
    {
        return SPARK_STATUS_OK;
    }
    if (control_generation < ledger->active_control_generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (SparkWorkTransactionHasActiveEntries(ledger) != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    SparkWorkTransactionResetLedger(ledger,control_generation);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWorkTransactionRecycleOldestTerminalEntry(
    SparkWorkTransactionLedger *ledger)
{
    uint64_t oldest_epoch;
    uint32_t entry_index;
    uint32_t oldest_index;

    oldest_epoch = UINT64_MAX;
    oldest_index = SPARK_WORK_TRANSACTION_INVALID_INDEX;
    for (entry_index = 0u;
         entry_index < ledger->entry_capacity;
         ++entry_index)
    {
        SparkWorkTransactionEntry *entry;

        entry = &ledger->entries[entry_index];
        if (SparkWorkTransactionStateIsTerminal(entry->state) != 0u &&
            entry->last_observed_epoch < oldest_epoch)
        {
            oldest_epoch = entry->last_observed_epoch;
            oldest_index = entry_index;
        }
    }
    if (oldest_index == SPARK_WORK_TRANSACTION_INVALID_INDEX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memset(&ledger->entries[oldest_index],0,
        sizeof(ledger->entries[oldest_index]));
    ledger->entries[oldest_index].state =
        SPARK_WORK_TRANSACTION_STATE_TOMBSTONE;
    ledger->entry_count -= 1u;
    ledger->terminal_entry_count -= 1u;
    ledger->tombstone_entry_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWorkTransactionSelectInsertionEntry(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    SparkWorkTransactionEntry **entry_out)
{
    uint32_t entry_index;
    uint32_t first_tombstone_index;
    uint32_t probe_index;
    uint32_t start_index;

    first_tombstone_index = SPARK_WORK_TRANSACTION_INVALID_INDEX;
    start_index = SparkWorkTransactionIdentityStartIndex(ledger,identity);
    for (probe_index = 0u;
         probe_index < ledger->entry_capacity;
         ++probe_index)
    {
        SparkWorkTransactionEntry *entry;

        entry_index = (start_index + probe_index) % ledger->entry_capacity;
        entry = &ledger->entries[entry_index];
        if (entry->state == SPARK_WORK_TRANSACTION_STATE_TOMBSTONE &&
            first_tombstone_index == SPARK_WORK_TRANSACTION_INVALID_INDEX)
        {
            first_tombstone_index = entry_index;
            continue;
        }
        if (entry->state == SPARK_WORK_TRANSACTION_STATE_EMPTY)
        {
            if (first_tombstone_index !=
                SPARK_WORK_TRANSACTION_INVALID_INDEX)
            {
                entry = &ledger->entries[first_tombstone_index];
                ledger->tombstone_entry_count -= 1u;
            }
            *entry_out = entry;
            return SPARK_STATUS_OK;
        }
    }
    if (first_tombstone_index != SPARK_WORK_TRANSACTION_INVALID_INDEX)
    {
        ledger->tombstone_entry_count -= 1u;
        *entry_out = &ledger->entries[first_tombstone_index];
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}

uint32_t SparkWorkTransactionPhaseIsValid(
    uint32_t phase)
{
    return phase >= SPARK_WORK_TRANSACTION_PHASE_PREFILL &&
        phase <= SPARK_WORK_TRANSACTION_PHASE_KNOWN_MAX;
}

uint32_t SparkWorkTransactionStateIsTerminal(
    uint32_t state)
{
    return state == SPARK_WORK_TRANSACTION_STATE_COMMITTED ||
        state == SPARK_WORK_TRANSACTION_STATE_CANCELLED ||
        state == SPARK_WORK_TRANSACTION_STATE_FAILED;
}

uint64_t SparkWorkTransactionFingerprintBytes(
    const void *data,
    uint32_t data_bytes)
{
    const uint8_t *bytes;
    uint64_t hash;
    uint32_t byte_index;

    if (data == 0 || data_bytes == 0u)
    {
        return 0u;
    }
    bytes = (const uint8_t *)data;
    hash = SPARK_WORK_TRANSACTION_FINGERPRINT_OFFSET;
    for (byte_index = 0u; byte_index < data_bytes; ++byte_index)
    {
        hash ^= bytes[byte_index];
        hash *= SPARK_WORK_TRANSACTION_FINGERPRINT_PRIME;
    }
    return hash == 0u ? 1u : hash;
}

uint32_t SparkWorkTransactionIdentitiesMatch(
    const SparkWorkTransactionIdentity *left,
    const SparkWorkTransactionIdentity *right)
{
    return left != 0 && right != 0 &&
        left->control_generation == right->control_generation &&
        left->transaction_id == right->transaction_id &&
        left->dispatch_generation == right->dispatch_generation &&
        left->request_generation == right->request_generation &&
        left->step_generation == right->step_generation &&
        left->step_chunk_index == right->step_chunk_index &&
        left->step_chunk_count == right->step_chunk_count &&
        left->phase == right->phase &&
        left->reserved0 == right->reserved0;
}

SparkStatus SparkWorkTransactionValidateIdentity(
    const SparkWorkTransactionIdentity *identity)
{
    if (identity == 0 ||
        identity->control_generation == 0u ||
        identity->transaction_id == 0u ||
        identity->dispatch_generation == 0u ||
        identity->request_generation == 0u ||
        identity->step_generation == 0u ||
        identity->step_chunk_count == 0u ||
        identity->step_chunk_index >= identity->step_chunk_count ||
        SparkWorkTransactionPhaseIsValid(identity->phase) == 0u ||
        identity->reserved0 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWorkTransactionInitializeLedger(
    SparkWorkTransactionLedger *ledger,
    SparkWorkTransactionEntry *entries,
    uint32_t entry_capacity)
{
    if (ledger == 0 || entries == 0 || entry_capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(ledger,0,sizeof(*ledger));
    memset(entries,0,(size_t)entry_capacity * sizeof(entries[0u]));
    ledger->abi_version = SPARK_WORK_TRANSACTION_ABI_VERSION;
    ledger->descriptor_bytes = SPARK_WORK_TRANSACTION_LEDGER_BYTES;
    ledger->entry_capacity = entry_capacity;
    ledger->entries = entries;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWorkTransactionObserve(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    SparkWorkTransactionEntry **entry_out,
    uint32_t *observation_out)
{
    SparkWorkTransactionEntry *entry;
    SparkStatus status;

    if (entry_out == 0 || observation_out == 0 ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *entry_out = 0;
    *observation_out = 0u;
    status = SparkWorkTransactionValidateLedger(ledger);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWorkTransactionValidateIdentity(identity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWorkTransactionPrepareControlGeneration(
        ledger,
        identity->control_generation);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    ledger->epoch += 1u;
    if (ledger->epoch == 0u)
    {
        ledger->epoch = 1u;
    }
    status = SparkWorkTransactionFindInternal(ledger,identity,&entry,0);
    if (status == SPARK_STATUS_OK)
    {
        entry->last_observed_epoch = ledger->epoch;
        *entry_out = entry;
        if (entry->payload_fingerprint != payload_fingerprint)
        {
            *observation_out = SPARK_WORK_TRANSACTION_OBSERVATION_CONFLICT;
            return SPARK_STATUS_VALIDATION_FAILED;
        }
        *observation_out = SparkWorkTransactionStateIsTerminal(entry->state)
            ? SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_TERMINAL
            : SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_ACTIVE;
        return SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_NOT_FOUND)
    {
        return status;
    }
    status = SparkWorkTransactionSelectInsertionEntry(
        ledger,
        identity,
        &entry);
    if (status == SPARK_STATUS_CAPACITY_EXCEEDED)
    {
        status = SparkWorkTransactionRecycleOldestTerminalEntry(ledger);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkWorkTransactionSelectInsertionEntry(
                ledger,
                identity,
                &entry);
        }
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    memset(entry,0,sizeof(*entry));
    entry->identity = *identity;
    entry->payload_fingerprint = payload_fingerprint;
    entry->last_observed_epoch = ledger->epoch;
    entry->state = SPARK_WORK_TRANSACTION_STATE_PREPARED;
    entry->terminal_status = (uint32_t)SPARK_STATUS_PENDING;
    ledger->entry_count += 1u;
    ledger->active_entry_count += 1u;
    *entry_out = entry;
    *observation_out = SPARK_WORK_TRANSACTION_OBSERVATION_NEW;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWorkTransactionTransition(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    uint32_t target_state,
    SparkStatus terminal_status)
{
    SparkWorkTransactionEntry *entry;
    uint32_t target_is_terminal;
    SparkStatus status;

    status = SparkWorkTransactionValidateLedger(ledger);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWorkTransactionValidateIdentity(identity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkWorkTransactionStateIsStored(target_state) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkWorkTransactionFindInternal(ledger,identity,&entry,0);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkWorkTransactionStateIsTerminal(entry->state) != 0u)
    {
        if (entry->state == target_state &&
            entry->terminal_status == (uint32_t)terminal_status)
        {
            return SPARK_STATUS_DUPLICATE;
        }
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkWorkTransactionTransitionIsLegal(
            entry->state,
            target_state) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    ledger->epoch += 1u;
    if (ledger->epoch == 0u)
    {
        ledger->epoch = 1u;
    }
    entry->last_observed_epoch = ledger->epoch;
    target_is_terminal = SparkWorkTransactionStateIsTerminal(target_state);
    if (target_is_terminal != 0u)
    {
        ledger->active_entry_count -= 1u;
        ledger->terminal_entry_count += 1u;
    }
    entry->state = target_state;
    entry->terminal_status = target_is_terminal != 0u ?
        (uint32_t)terminal_status : (uint32_t)SPARK_STATUS_PENDING;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWorkTransactionFind(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    SparkWorkTransactionEntry **entry_out)
{
    SparkStatus status;

    if (entry_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *entry_out = 0;
    status = SparkWorkTransactionValidateLedger(ledger);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWorkTransactionValidateIdentity(identity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkWorkTransactionFindInternal(ledger,identity,entry_out,0);
}

void SparkWorkTransactionInitializeAcknowledgement(
    SparkWorkTransactionAcknowledgement *acknowledgement,
    const SparkWorkTransactionIdentity *identity,
    uint64_t packet_fingerprint,
    uint32_t transaction_state,
    SparkStatus status)
{
    if (acknowledgement == 0)
    {
        return;
    }
    memset(acknowledgement,0,sizeof(*acknowledgement));
    acknowledgement->magic = SPARK_WORK_TRANSACTION_ACK_MAGIC;
    acknowledgement->abi_version = SPARK_WORK_TRANSACTION_ABI_VERSION;
    acknowledgement->descriptor_bytes =
        SPARK_WORK_TRANSACTION_ACKNOWLEDGEMENT_BYTES;
    acknowledgement->status = (uint32_t)status;
    acknowledgement->transaction_state = transaction_state;
    if (identity != 0)
    {
        acknowledgement->identity = *identity;
    }
    acknowledgement->packet_fingerprint = packet_fingerprint;
}

SparkStatus SparkWorkTransactionValidateAcknowledgement(
    const SparkWorkTransactionAcknowledgement *acknowledgement,
    const SparkWorkTransactionIdentity *expected_identity,
    uint64_t expected_packet_fingerprint)
{
    if (acknowledgement == 0 || expected_identity == 0 ||
        expected_packet_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (acknowledgement->magic != SPARK_WORK_TRANSACTION_ACK_MAGIC ||
        acknowledgement->abi_version != SPARK_WORK_TRANSACTION_ABI_VERSION ||
        acknowledgement->descriptor_bytes !=
            SPARK_WORK_TRANSACTION_ACKNOWLEDGEMENT_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (SparkWorkTransactionValidateIdentity(
            &acknowledgement->identity) != SPARK_STATUS_OK ||
        SparkWorkTransactionIdentitiesMatch(
            &acknowledgement->identity,
            expected_identity) == 0u ||
        acknowledgement->packet_fingerprint !=
            expected_packet_fingerprint ||
        acknowledgement->status > (uint32_t)SPARK_STATUS_UNSUPPORTED ||
        SparkWorkTransactionStateIsStored(
            acknowledgement->transaction_state) == 0u ||
        acknowledgement->reserved0 != 0u ||
        acknowledgement->reserved1 != 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    return (SparkStatus)acknowledgement->status;
}

SparkStatus SparkWorkTransactionInitializeCreditLedger(
    SparkWorkTransactionCreditLedger *ledger,
    const uint32_t capacities[SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT])
{
    uint32_t domain;

    if (ledger == 0 || capacities == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(ledger,0,sizeof(*ledger));
    ledger->abi_version = SPARK_WORK_TRANSACTION_ABI_VERSION;
    ledger->descriptor_bytes = (uint32_t)sizeof(*ledger);
    for (domain = 0u;
         domain < SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT;
         ++domain)
    {
        ledger->domains[domain].capacity = capacities[domain];
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWorkTransactionAcquireCredits(
    SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count)
{
    SparkWorkTransactionCreditDomain *credit_domain;

    if (ledger == 0 ||
        ledger->abi_version != SPARK_WORK_TRANSACTION_ABI_VERSION ||
        ledger->descriptor_bytes != (uint32_t)sizeof(*ledger) ||
        domain >= SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT ||
        credit_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    credit_domain = &ledger->domains[domain];
    if (credit_domain->in_use > credit_domain->capacity ||
        credit_count > credit_domain->capacity - credit_domain->in_use)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    credit_domain->in_use += credit_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWorkTransactionReleaseCredits(
    SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count)
{
    SparkWorkTransactionCreditDomain *credit_domain;

    if (ledger == 0 ||
        ledger->abi_version != SPARK_WORK_TRANSACTION_ABI_VERSION ||
        ledger->descriptor_bytes != (uint32_t)sizeof(*ledger) ||
        domain >= SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT ||
        credit_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    credit_domain = &ledger->domains[domain];
    if (credit_count > credit_domain->in_use)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    credit_domain->in_use -= credit_count;
    return SPARK_STATUS_OK;
}

uint32_t SparkWorkTransactionAvailableCredits(
    const SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain)
{
    if (ledger == 0 ||
        ledger->abi_version != SPARK_WORK_TRANSACTION_ABI_VERSION ||
        ledger->descriptor_bytes != (uint32_t)sizeof(*ledger) ||
        domain >= SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_COUNT ||
        ledger->domains[domain].in_use >
            ledger->domains[domain].capacity)
    {
        return 0u;
    }
    return ledger->domains[domain].capacity -
        ledger->domains[domain].in_use;
}
