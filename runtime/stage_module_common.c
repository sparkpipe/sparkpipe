#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "sparkpipe/spark_stage_module_common.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

static const char *SparkStageModuleSafeText(const char *text)
{
    return text != 0 ? text : "unknown";
}

static SparkStatus SparkStageModuleValidateLedger(
    const SparkStageModuleLedger *ledger)
{
    if (ledger == 0 || ledger->module_tag == 0 || ledger->module_tag[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (ledger->device_allocation_count > SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static void SparkStageModuleReleaseLastAllocation(
    SparkStageModuleLedger *ledger,
    void *allocation)
{
    uint32_t allocation_index;
    uint64_t allocation_bytes;

    if (ledger == 0 || allocation == 0 || ledger->device_allocation_count == 0u)
    {
        return;
    }

    allocation_index = ledger->device_allocation_count - 1u;
    if (ledger->device_allocations[allocation_index] != allocation)
    {
        return;
    }

    allocation_bytes = ledger->device_allocation_bytes[allocation_index];
    ledger->device_allocations[allocation_index] = 0;
    ledger->device_allocation_bytes[allocation_index] = 0u;
    ledger->device_allocation_count = allocation_index;
    if (allocation_bytes <= ledger->device_bytes_resident)
    {
        ledger->device_bytes_resident -= allocation_bytes;
    }
    else
    {
        ledger->device_bytes_resident = 0u;
    }
    (void)cudaFree(allocation);
}

static SparkStatus SparkStageModuleRecordAllocation(
    SparkStageModuleLedger *ledger,
    void *allocation,
    uint64_t bytes)
{
    SparkStatus status;

    status = SparkStageModuleValidateLedger(ledger);
    if (status != SPARK_STATUS_OK || allocation == 0 || bytes == 0u)
    {
        return status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (ledger->device_allocation_count >= SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS)
    {
        fprintf(
            stderr,
            "%s allocation_ledger_full count=%u\n",
            ledger->module_tag,
            ledger->device_allocation_count);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (UINT64_MAX - ledger->device_bytes_resident < bytes)
    {
        fprintf(stderr, "%s allocation_byte_count_overflow\n", ledger->module_tag);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    ledger->device_allocations[ledger->device_allocation_count] = allocation;
    ledger->device_allocation_bytes[ledger->device_allocation_count] = bytes;
    ledger->device_allocation_count++;
    ledger->device_bytes_resident += bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkStageModuleParseUnsigned64(
    const char *module_tag,
    const char *name,
    const char *text,
    uint64_t minimum,
    uint64_t maximum,
    uint64_t *value)
{
    unsigned long long parsed;
    char *end;

    if (module_tag == 0 || name == 0 || text == 0 || value == 0 ||
        minimum > maximum || text[0] < '0' || text[0] > '9')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    errno = 0;
    end = 0;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || end == 0 || *end != '\0' ||
        parsed < minimum || parsed > maximum)
    {
        fprintf(
            stderr,
            "%s config_invalid name=%s value=%s allowed=[%llu,%llu]\n",
            module_tag,
            name,
            text,
            (unsigned long long)minimum,
            (unsigned long long)maximum);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *value = (uint64_t)parsed;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleCudaStatus(
    const char *module_tag,
    cudaError_t error,
    const char *site)
{
    if (error == cudaSuccess)
    {
        return SPARK_STATUS_OK;
    }

    fprintf(
        stderr,
        "%s cuda_error site=%s error=%s\n",
        SparkStageModuleSafeText(module_tag),
        SparkStageModuleSafeText(site),
        SparkStageModuleSafeText(cudaGetErrorString(error)));
    if (error == cudaErrorMemoryAllocation)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

void SparkStageModuleCudaForkDestroy(SparkStageModuleCudaFork *fork)
{
    if (fork == 0)
    {
        return;
    }
    if (fork->join_event != 0)
    {
        (void)cudaEventDestroy(fork->join_event);
    }
    if (fork->milestone_event != 0)
    {
        (void)cudaEventDestroy(fork->milestone_event);
    }
    if (fork->fork_event != 0)
    {
        (void)cudaEventDestroy(fork->fork_event);
    }
    if (fork->auxiliary_stream != 0)
    {
        (void)cudaStreamDestroy(fork->auxiliary_stream);
    }
    memset(fork, 0, sizeof(*fork));
}

SparkStatus SparkStageModuleCudaForkInitialize(
    const char *module_tag,
    SparkStageModuleCudaFork *fork)
{
    cudaError_t error;
    if (module_tag == 0 || module_tag[0] == '\0' || fork == 0 ||
        fork->auxiliary_stream != 0 || fork->fork_event != 0 ||
        fork->milestone_event != 0 || fork->join_event != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    error = cudaStreamCreateWithFlags(
        &fork->auxiliary_stream, cudaStreamNonBlocking);
    if (error == cudaSuccess)
    {
        error = cudaEventCreateWithFlags(
            &fork->fork_event, cudaEventDisableTiming);
    }
    if (error == cudaSuccess)
    {
        error = cudaEventCreateWithFlags(
            &fork->milestone_event, cudaEventDisableTiming);
    }
    if (error == cudaSuccess)
    {
        error = cudaEventCreateWithFlags(
            &fork->join_event, cudaEventDisableTiming);
    }
    if (error != cudaSuccess)
    {
        SparkStageModuleCudaForkDestroy(fork);
    }
    return SparkStageModuleCudaStatus(module_tag, error, "cuda_fork_initialize");
}

SparkStatus SparkStageModuleEnvironmentText(
    const char *module_tag,
    const char *name,
    const char **value)
{
    const char *text;

    if (module_tag == 0 || name == 0 || name[0] == '\0' || value == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *value = 0;
    text = getenv(name);
    if (text == 0 || text[0] == '\0')
    {
        fprintf(stderr, "%s config_missing name=%s\n", module_tag, name);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *value = text;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleEnvironmentUnsigned(
    const char *module_tag,
    const char *name,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t *value)
{
    const char *text;
    SparkStatus status;
    uint64_t parsed;

    if (value == 0 || minimum > maximum)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkStageModuleEnvironmentText(module_tag, name, &text);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkStageModuleParseUnsigned64(
        module_tag,
        name,
        text,
        minimum,
        maximum,
        &parsed);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    *value = (uint32_t)parsed;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleEnvironmentUnsigned64(
    const char *module_tag,
    const char *name,
    uint64_t minimum,
    uint64_t maximum,
    uint64_t *value)
{
    const char *text;
    SparkStatus status;

    if (value == 0 || minimum > maximum)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkStageModuleEnvironmentText(module_tag, name, &text);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkStageModuleParseUnsigned64(
        module_tag,
        name,
        text,
        minimum,
        maximum,
        value);
}

SparkStatus SparkStageModuleDeviceAllocate(
    SparkStageModuleLedger *ledger,
    uint64_t bytes,
    void **pointer)
{
    void *allocation;
    SparkStatus status;

    if (pointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *pointer = 0;
    status = SparkStageModuleValidateLedger(ledger);
    if (status != SPARK_STATUS_OK || bytes == 0u || bytes > (uint64_t)SIZE_MAX)
    {
        return status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (ledger->device_allocation_count >= SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS ||
        UINT64_MAX - ledger->device_bytes_resident < bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    allocation = 0;
    status = SparkStageModuleCudaStatus(
        ledger->module_tag,
        cudaMalloc(&allocation, (size_t)bytes),
        "cudaMalloc");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkStageModuleRecordAllocation(ledger, allocation, bytes);
    if (status != SPARK_STATUS_OK)
    {
        (void)cudaFree(allocation);
        return status;
    }

    *pointer = allocation;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleDeviceAllocateZeroed(
    SparkStageModuleLedger *ledger,
    uint64_t bytes,
    void **pointer)
{
    SparkStatus status;

    status = SparkStageModuleDeviceAllocate(ledger, bytes, pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkStageModuleCudaStatus(
        ledger->module_tag,
        cudaMemset(*pointer, 0, (size_t)bytes),
        "cudaMemset");
    if (status != SPARK_STATUS_OK)
    {
        SparkStageModuleReleaseLastAllocation(ledger, *pointer);
        *pointer = 0;
    }
    return status;
}

void SparkStageModuleLedgerRollback(
    SparkStageModuleLedger *ledger,
    uint32_t allocation_count)
{
    uint32_t current_count;

    if (ledger == 0 || allocation_count > ledger->device_allocation_count)
    {
        return;
    }
    current_count = ledger->device_allocation_count;
    while (current_count > allocation_count)
    {
        uint32_t current_index;
        void *allocation;
        uint64_t allocation_bytes;

        current_index = current_count - 1u;
        allocation = ledger->device_allocations[current_index];
        allocation_bytes = ledger->device_allocation_bytes[current_index];
        if (allocation != 0)
        {
            (void)cudaFree(allocation);
        }
        ledger->device_allocations[current_index] = 0;
        ledger->device_allocation_bytes[current_index] = 0u;
        if (allocation_bytes <= ledger->device_bytes_resident)
        {
            ledger->device_bytes_resident -= allocation_bytes;
        }
        else
        {
            ledger->device_bytes_resident = 0u;
        }
        current_count = current_index;
    }
    ledger->device_allocation_count = allocation_count;
}

void SparkStageModuleLedgerRelease(SparkStageModuleLedger *ledger)
{
    uint32_t allocation_index;

    if (ledger == 0)
    {
        return;
    }

    for (allocation_index = ledger->device_allocation_count;
         allocation_index > 0u;
         allocation_index--)
    {
        uint32_t current_index = allocation_index - 1u;
        if (ledger->device_allocations[current_index] != 0)
        {
            (void)cudaFree(ledger->device_allocations[current_index]);
        }
        ledger->device_allocations[current_index] = 0;
        ledger->device_allocation_bytes[current_index] = 0u;
    }
    ledger->device_allocation_count = 0u;
    ledger->device_bytes_resident = 0u;
}

SparkStatus SparkStageModulePackRead(
    const char *module_tag,
    FILE *file,
    uint64_t offset,
    void *destination,
    uint64_t bytes)
{
    size_t read_bytes;

    if (module_tag == 0 || file == 0 || destination == 0 || bytes == 0u ||
        bytes > (uint64_t)SIZE_MAX || offset > (uint64_t)INT64_MAX)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (fseeko(file, (off_t)offset, SEEK_SET) != 0)
    {
        fprintf(
            stderr,
            "%s pack_seek_failed offset=%llu\n",
            module_tag,
            (unsigned long long)offset);
        return SPARK_STATUS_IO_ERROR;
    }

    read_bytes = fread(destination, 1u, (size_t)bytes, file);
    if (read_bytes != (size_t)bytes)
    {
        fprintf(
            stderr,
            "%s pack_read_failed offset=%llu bytes=%llu read=%llu\n",
            module_tag,
            (unsigned long long)offset,
            (unsigned long long)bytes,
            (unsigned long long)read_bytes);
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleLoadDeviceRegion(
    SparkStageModuleLedger *ledger,
    FILE *file,
    uint64_t offset,
    uint64_t bytes,
    void **pointer)
{
    void *device;
    void *staging;
    uint64_t moved;
    size_t staging_bytes;
    SparkStatus status;

    if (pointer == 0 || file == 0 || bytes == 0u ||
        bytes > (uint64_t)SIZE_MAX || offset > UINT64_MAX - bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *pointer = 0;
    device = 0;
    status = SparkStageModuleDeviceAllocate(ledger, bytes, &device);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    staging_bytes = (size_t)(bytes < SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES
        ? bytes
        : SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES);
    staging = malloc(staging_bytes);
    if (staging == 0)
    {
        SparkStageModuleReleaseLastAllocation(ledger, device);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    status = SPARK_STATUS_OK;
    moved = 0u;
    while (moved < bytes)
    {
        uint64_t chunk = bytes - moved;
        if (chunk > SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES)
        {
            chunk = SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES;
        }
        status = SparkStageModulePackRead(
            ledger->module_tag,
            file,
            offset + moved,
            staging,
            chunk);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkStageModuleCudaStatus(
                ledger->module_tag,
                cudaMemcpy(
                    (uint8_t *)device + moved,
                    staging,
                    (size_t)chunk,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy_h2d");
        }
        if (status != SPARK_STATUS_OK)
        {
            break;
        }
        moved += chunk;
    }

    free(staging);
    if (status != SPARK_STATUS_OK)
    {
        SparkStageModuleReleaseLastAllocation(ledger, device);
        return status;
    }

    *pointer = device;
    return SPARK_STATUS_OK;
}

static uint64_t SparkStageModuleMonotonicNanoseconds(void)
{
    struct timespec current_time;

    if (clock_gettime(CLOCK_MONOTONIC, &current_time) != 0)
    {
        return 0u;
    }
    return (uint64_t)current_time.tv_sec * 1000000000ull +
        (uint64_t)current_time.tv_nsec;
}

static int SparkStageModuleAdmissionRejectionIsValid(
    SparkModelDriverAdmissionRejection rejection_reason)
{
    return rejection_reason >= SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY &&
        rejection_reason <= SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
}

void SparkStageModuleAdmissionDecisionInitialize(
    SparkModelDriverAdmissionDecision *decision,
    uint32_t available_dispatch_slot_count)
{
    if (decision == 0)
    {
        return;
    }
    memset(decision, 0, sizeof(*decision));
    decision->descriptor_bytes = (uint32_t)sizeof(*decision);
    decision->rejection_reason =
        SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    decision->available_dispatch_slot_count = available_dispatch_slot_count;
}

void SparkStageModuleAdmissionDecisionAccept(
    SparkModelDriverAdmissionDecision *decision)
{
    if (decision == 0)
    {
        return;
    }
    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
}

void SparkStageModuleAdmissionDecisionReject(
    SparkModelDriverAdmissionDecision *decision,
    SparkModelDriverAdmissionRejection rejection_reason)
{
    if (decision == 0)
    {
        return;
    }
    decision->accepted = 0u;
    decision->rejection_reason = SparkStageModuleAdmissionRejectionIsValid(
        rejection_reason)
        ? (uint32_t)rejection_reason
        : (uint32_t)SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE;
    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    decision->driver_dispatch_generation = 0u;
    decision->driver_dispatch_cookie0 = 0u;
    decision->driver_dispatch_cookie1 = 0u;
}

void SparkStageModuleRuntimeSnapshotInitialize(
    SparkModelDriverRuntimeSnapshot *snapshot,
    uint32_t program_id,
    const atomic_uint *slot_states,
    uint32_t slot_count)
{
    uint32_t available_slot_count;

    if (snapshot == 0)
    {
        return;
    }
    available_slot_count = SparkStageModuleSlotAvailableCount(
        slot_states,
        slot_count);
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->descriptor_bytes = (uint32_t)sizeof(*snapshot);
    snapshot->program_id = program_id;
    snapshot->available_dispatch_slot_count = available_slot_count;
    snapshot->active_submission_count = slot_count - available_slot_count;
}

SparkStatus SparkStageModuleSlotClaim(
    atomic_uint *slot_states,
    uint32_t slot_count,
    uint32_t *slot_index)
{
    uint32_t index;

    if (slot_states == 0 || slot_index == 0 || slot_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *slot_index = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    for (index = 0u; index < slot_count; index++)
    {
        unsigned int expected = SPARK_STAGE_MODULE_SLOT_FREE;
        if (atomic_compare_exchange_strong_explicit(
                &slot_states[index],
                &expected,
                SPARK_STAGE_MODULE_SLOT_CLAIMED,
                memory_order_acquire,
                memory_order_relaxed))
        {
            *slot_index = index;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_BUSY;
}

SparkStatus SparkStageModuleIndexSetClaim(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count)
{
    uint32_t claimed_count;

    if (index_states == 0 || indices == 0 || index_capacity == 0u ||
        index_count == 0u || index_count > index_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (claimed_count = 0u; claimed_count < index_count; claimed_count++)
    {
        unsigned int expected_state;
        uint32_t index;
        uint32_t previous_index;
        SparkStatus status;

        index = indices[claimed_count];
        if (index >= index_capacity)
        {
            SparkStageModuleIndexSetRelease(
                index_states,
                index_capacity,
                indices,
                claimed_count);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        expected_state = SPARK_STAGE_MODULE_SLOT_FREE;
        if (!atomic_compare_exchange_strong_explicit(
                &index_states[index],
                &expected_state,
                claimed_count + 1u,
                memory_order_acquire,
                memory_order_relaxed))
        {
            status = SPARK_STATUS_BUSY;
            for (previous_index = 0u; previous_index < claimed_count; previous_index++)
            {
                if (indices[previous_index] == index)
                {
                    status = SPARK_STATUS_INVALID_ARGUMENT;
                    break;
                }
            }
            SparkStageModuleIndexSetRelease(
                index_states,
                index_capacity,
                indices,
                claimed_count);
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleIndexClaimOrdinal(
    const atomic_uint *index_states,
    uint32_t index_capacity,
    uint32_t index,
    uint32_t *ordinal_out)
{
    unsigned int claim;

    if (index_states == 0 || ordinal_out == 0 || index >= index_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *ordinal_out = UINT32_MAX;
    claim = atomic_load_explicit(&index_states[index], memory_order_acquire);
    if (claim == SPARK_STAGE_MODULE_SLOT_FREE)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (claim > index_capacity)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    *ordinal_out = claim - 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleIndexSetClaimAndPrepare(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count,
    SparkStageModuleClaimedIndexPrepareFunction prepare_function,
    void *prepare_context)
{
    SparkStatus status;

    if (prepare_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkStageModuleIndexSetClaim(
        index_states,
        index_capacity,
        indices,
        index_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = prepare_function(prepare_context);
    if (status != SPARK_STATUS_OK)
    {
        SparkStageModuleIndexSetRelease(
            index_states,
            index_capacity,
            indices,
            index_count);
    }
    return status;
}

void SparkStageModuleIndexSetRelease(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count)
{
    uint32_t index_offset;

    if (index_states == 0 || indices == 0)
    {
        return;
    }
    for (index_offset = 0u; index_offset < index_count; index_offset++)
    {
        uint32_t index;

        index = indices[index_offset];
        if (index < index_capacity)
        {
            atomic_store_explicit(
                &index_states[index],
                SPARK_STAGE_MODULE_SLOT_FREE,
                memory_order_release);
        }
    }
}

void SparkStageModuleAtomicStateArrayInitialize(
    atomic_uint *states,
    uint32_t state_count)
{
    uint32_t state_index;

    if (states == 0)
    {
        return;
    }
    for (state_index = 0u; state_index < state_count; state_index++)
    {
        atomic_init(&states[state_index], SPARK_STAGE_MODULE_SLOT_FREE);
    }
}

uint32_t SparkStageModuleSlotAvailableCount(
    const atomic_uint *slot_states,
    uint32_t slot_count)
{
    uint32_t available_count;
    uint32_t index;

    if (slot_states == 0)
    {
        return 0u;
    }

    available_count = 0u;
    for (index = 0u; index < slot_count; index++)
    {
        if (atomic_load_explicit(&slot_states[index], memory_order_relaxed) ==
            SPARK_STAGE_MODULE_SLOT_FREE)
        {
            available_count++;
        }
    }
    return available_count;
}

uint32_t SparkStageModuleSlotCountFree(
    const atomic_uint *slot_states,
    uint32_t slot_count)
{
    return SparkStageModuleSlotAvailableCount(slot_states, slot_count);
}

SparkStatus SparkStageModuleWaitForSlots(
    const char *module_tag,
    const atomic_uint *slot_states,
    uint32_t slot_count,
    uint64_t timeout_nanoseconds)
{
    const struct timespec sleep_interval = {0, 1000000};
    uint64_t start_time;
    uint64_t current_time;

    if (module_tag == 0 || slot_states == 0 || slot_count == 0u ||
        timeout_nanoseconds == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    start_time = SparkStageModuleMonotonicNanoseconds();
    if (start_time == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    while (SparkStageModuleSlotAvailableCount(slot_states, slot_count) != slot_count)
    {
        current_time = SparkStageModuleMonotonicNanoseconds();
        if (current_time == 0u || current_time - start_time >= timeout_nanoseconds)
        {
            fprintf(
                stderr,
                "%s destroy_quiesce_timeout active_slots=%u\n",
                module_tag,
                slot_count - SparkStageModuleSlotAvailableCount(
                    slot_states,
                    slot_count));
            return SPARK_STATUS_BUSY;
        }
        (void)nanosleep(&sleep_interval, 0);
    }
    return SPARK_STATUS_OK;
}

void SparkStageModuleSlotRelease(
    atomic_uint *slot_states,
    uint32_t slot_index)
{
    if (slot_states == 0 || slot_index == SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
    {
        return;
    }
    atomic_store_explicit(
        &slot_states[slot_index],
        SPARK_STAGE_MODULE_SLOT_FREE,
        memory_order_release);
}

void SparkStageModuleCompleteAndReleaseClaims(
    SparkModelDriverCompletionFunction completion_function,
    void *completion_context,
    const SparkModelDriverCompletion *completion,
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count,
    atomic_uint *slot_states,
    uint32_t slot_index)
{
    if (completion_function != 0)
    {
        completion_function(completion_context, completion);
    }
    SparkStageModuleIndexSetRelease(
        index_states,
        index_capacity,
        indices,
        index_count);
    SparkStageModuleSlotRelease(slot_states, slot_index);
}
