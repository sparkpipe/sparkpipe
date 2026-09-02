#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_weightd_attach.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

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
    uint32_t branch;
    if (fork == 0)
    {
        return;
    }
    for (branch = 0u; branch < SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES;
         ++branch)
    {
        if (fork->join_events[branch] != 0)
        {
            (void)cudaEventDestroy(fork->join_events[branch]);
        }
    }
    if (fork->milestone_event != 0)
    {
        (void)cudaEventDestroy(fork->milestone_event);
    }
    if (fork->fork_event != 0)
    {
        (void)cudaEventDestroy(fork->fork_event);
    }
    for (branch = 0u; branch < SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES;
         ++branch)
    {
        if (fork->auxiliary_streams[branch] != 0)
        {
            (void)cudaStreamDestroy(fork->auxiliary_streams[branch]);
        }
    }
    memset(fork, 0, sizeof(*fork));
}

SparkStatus SparkStageModuleCudaForkInitialize(
    const char *module_tag,
    SparkStageModuleCudaFork *fork)
{
    cudaError_t error;
    uint32_t branch;
    if (module_tag == 0 || module_tag[0] == '\0' || fork == 0 ||
        fork->fork_event != 0 || fork->milestone_event != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (branch = 0u; branch < SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES;
         ++branch)
    {
        if (fork->auxiliary_streams[branch] != 0 ||
            fork->join_events[branch] != 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    error = cudaSuccess;
    for (branch = 0u; branch < SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES &&
         error == cudaSuccess; ++branch)
    {
        error = cudaStreamCreateWithFlags(
            &fork->auxiliary_streams[branch], cudaStreamNonBlocking);
    }
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
    for (branch = 0u; branch < SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES &&
         error == cudaSuccess; ++branch)
    {
        error = cudaEventCreateWithFlags(
            &fork->join_events[branch], cudaEventDisableTiming);
    }
    if (error != cudaSuccess)
    {
        SparkStageModuleCudaForkDestroy(fork);
    }
    return SparkStageModuleCudaStatus(module_tag, error, "cuda_fork_initialize");
}

cudaError_t SparkStageModuleCudaForkBegin(
    SparkStageModuleCudaFork *fork,
    cudaStream_t primary_stream,
    uint32_t branch_count)
{
    cudaError_t error;
    uint32_t branch;
    if (fork == 0 || primary_stream == 0 || branch_count == 0u ||
        branch_count > SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES ||
        fork->fork_event == 0)
    {
        return cudaErrorInvalidValue;
    }
    error = cudaEventRecord(fork->fork_event, primary_stream);
    for (branch = 0u; branch < branch_count && error == cudaSuccess; ++branch)
    {
        if (fork->auxiliary_streams[branch] == 0)
        {
            return cudaErrorInvalidValue;
        }
        error = cudaStreamWaitEvent(
            fork->auxiliary_streams[branch], fork->fork_event, 0u);
    }
    return error;
}

cudaError_t SparkStageModuleCudaForkJoin(
    SparkStageModuleCudaFork *fork,
    cudaStream_t primary_stream,
    uint32_t branch_count)
{
    cudaError_t error;
    uint32_t branch;
    if (fork == 0 || primary_stream == 0 || branch_count == 0u ||
        branch_count > SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES)
    {
        return cudaErrorInvalidValue;
    }
    error = cudaSuccess;
    for (branch = 0u; branch < branch_count && error == cudaSuccess; ++branch)
    {
        if (fork->auxiliary_streams[branch] == 0 ||
            fork->join_events[branch] == 0)
        {
            return cudaErrorInvalidValue;
        }
        error = cudaEventRecord(fork->join_events[branch],
            fork->auxiliary_streams[branch]);
        if (error == cudaSuccess)
        {
            error = cudaStreamWaitEvent(primary_stream,
                fork->join_events[branch], 0u);
        }
    }
    return error;
}

void SparkStageModuleCudaReadAheadDestroy(
    SparkStageModuleCudaReadAhead *read_ahead)
{
    if (read_ahead == 0)
        return;
    if (read_ahead->stream != 0)
        (void)cudaStreamDestroy(read_ahead->stream);
    if (read_ahead->completion_event != 0)
        (void)cudaEventDestroy(read_ahead->completion_event);
    if (read_ahead->source_ready_event != 0)
        (void)cudaEventDestroy(read_ahead->source_ready_event);
    memset(read_ahead, 0, sizeof(*read_ahead));
}

SparkStatus SparkStageModuleCudaReadAheadInitialize(
    const char *module_tag,
    SparkStageModuleLedger *ledger,
    SparkStageModuleCudaReadAhead *read_ahead,
    uint32_t sink_word_capacity)
{
    cudaError_t error;
    SparkStatus status;
    if (module_tag == 0 || module_tag[0] == '\0' || ledger == 0 ||
        read_ahead == 0 || sink_word_capacity == 0u || read_ahead->stream != 0 ||
        read_ahead->source_ready_event != 0 || read_ahead->completion_event != 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    atomic_init(&read_ahead->state, SPARK_STAGE_MODULE_CUDA_READ_AHEAD_IDLE);
    read_ahead->sink_word_capacity = sink_word_capacity;
    status = SparkStageModuleDeviceAllocate(ledger,
        (uint64_t)sink_word_capacity * sizeof(uint32_t),
        (void **)&read_ahead->sink_u32);
    error = status == SPARK_STATUS_OK ? cudaStreamCreateWithFlags(
        &read_ahead->stream, cudaStreamNonBlocking) : cudaSuccess;
    if (status == SPARK_STATUS_OK && error == cudaSuccess)
        error = cudaEventCreateWithFlags(&read_ahead->source_ready_event,
            cudaEventDisableTiming);
    if (status == SPARK_STATUS_OK && error == cudaSuccess)
        error = cudaEventCreateWithFlags(&read_ahead->completion_event,
            cudaEventDisableTiming);
    if (status == SPARK_STATUS_OK && error != cudaSuccess)
        status = SparkStageModuleCudaStatus(module_tag, error,
            "cuda_read_ahead_initialize");
    if (status != SPARK_STATUS_OK)
        SparkStageModuleCudaReadAheadDestroy(read_ahead);
    return status;
}

SparkStatus SparkStageModuleCudaReadAheadArm(
    const char *module_tag,
    SparkStageModuleCudaReadAhead *read_ahead,
    cudaStream_t primary_stream,
    SparkStageModuleCudaReadAheadLaunchFunction launch_function,
    void *launch_context)
{
    unsigned int expected;
    cudaError_t error;
    if (module_tag == 0 || read_ahead == 0 || read_ahead->stream == 0 ||
        read_ahead->source_ready_event == 0 || read_ahead->completion_event == 0 ||
        read_ahead->sink_u32 == 0 || read_ahead->sink_word_capacity == 0u ||
        primary_stream == 0 || launch_function == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    expected = SPARK_STAGE_MODULE_CUDA_READ_AHEAD_IDLE;
    if (!atomic_compare_exchange_strong_explicit(&read_ahead->state, &expected,
            SPARK_STAGE_MODULE_CUDA_READ_AHEAD_BUILDING,
            memory_order_acq_rel, memory_order_acquire))
        return SPARK_STATUS_BUSY;
    error = cudaEventRecord(read_ahead->source_ready_event, primary_stream);
    if (error == cudaSuccess)
        error = cudaStreamWaitEvent(read_ahead->stream,
            read_ahead->source_ready_event, 0u);
    if (error == cudaSuccess)
        error = launch_function(read_ahead->stream, read_ahead->sink_u32,
            read_ahead->sink_word_capacity, launch_context);
    if (error == cudaSuccess)
        error = cudaEventRecord(read_ahead->completion_event, read_ahead->stream);
    if (error == cudaSuccess)
        atomic_store_explicit(&read_ahead->state,
            SPARK_STAGE_MODULE_CUDA_READ_AHEAD_ARMED, memory_order_release);
    else
    {
        (void)cudaStreamSynchronize(read_ahead->stream);
        atomic_store_explicit(&read_ahead->state,
            SPARK_STAGE_MODULE_CUDA_READ_AHEAD_IDLE, memory_order_release);
    }
    return SparkStageModuleCudaStatus(module_tag, error, "cuda_read_ahead_arm");
}

SparkStatus SparkStageModuleCudaReadAheadJoin(
    const char *module_tag,
    SparkStageModuleCudaReadAhead *read_ahead,
    cudaStream_t primary_stream)
{
    unsigned int state;
    cudaError_t error;
    if (module_tag == 0 || read_ahead == 0 || primary_stream == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state = atomic_load_explicit(&read_ahead->state, memory_order_acquire);
    if (state == SPARK_STAGE_MODULE_CUDA_READ_AHEAD_IDLE)
        return SPARK_STATUS_OK;
    if (state != SPARK_STAGE_MODULE_CUDA_READ_AHEAD_ARMED ||
        read_ahead->completion_event == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    error = cudaStreamWaitEvent(primary_stream, read_ahead->completion_event, 0u);
    if (error != cudaSuccess)
        (void)cudaStreamSynchronize(read_ahead->stream);
    atomic_store_explicit(&read_ahead->state,
        SPARK_STAGE_MODULE_CUDA_READ_AHEAD_IDLE, memory_order_release);
    return SparkStageModuleCudaStatus(module_tag, error, "cuda_read_ahead_join");
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

SparkStatus SparkStageModuleEnvironmentUnsignedOrDefault(
    const char *module_tag,
    const char *name,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t fallback,
    uint32_t *value)
{
    const char *text;
    SparkStatus status;
    uint64_t parsed;

    if (module_tag == 0 || name == 0 || name[0] == '\0' || value == 0 ||
        minimum > maximum || fallback < minimum || fallback > maximum)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    text = getenv(name);
    if (text == 0 || text[0] == '\0')
    {
        *value = fallback;
        return SPARK_STATUS_OK;
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

static void SparkStageModulePackArenaRelease(SparkStageModuleLedger *ledger);

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
    SparkStageModulePackArenaRelease(ledger);
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


typedef struct SparkStageModulePackArena
{
	SparkWeightdAttachOutcome outcome;
	uint64_t pack_bytes;
	char pack_path[SPARK_WEIGHTD_PATH_BYTES];
	int failed;
} SparkStageModulePackArena;

static void SparkStageModulePackArenaRelease(SparkStageModuleLedger *ledger)
{
	SparkStageModulePackArena *arena;
	if (ledger == 0 || ledger->pack_arena == 0)
		return;
	arena = (SparkStageModulePackArena *)ledger->pack_arena;
	if (arena->outcome.client != 0)
		(void)SparkWeightdAttachRelease(&arena->outcome);
	free(arena);
	ledger->pack_arena = 0;
}

static int SparkStageModulePackArenaEnsure(
	SparkStageModuleLedger *ledger,
	FILE *file)
{
	SparkStageModulePackArena *arena;
	SparkWeightdPackSlice slice;
	struct stat pack_stat;
	char fd_path[64];
	char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
	ssize_t path_bytes;
	SparkStatus status;

	if (ledger == 0 || file == 0)
		return 0;
	if (ledger->pack_arena != 0)
	{
		arena = (SparkStageModulePackArena *)ledger->pack_arena;
		return arena->outcome.client != 0 && arena->outcome.map_base != 0;
	}
	if (SparkWeightdAttachRequested() != SPARK_STATUS_OK)
		return 0;
	arena = (SparkStageModulePackArena *)calloc(1u,sizeof(*arena));
	if (arena == 0)
		return 0;
	ledger->pack_arena = arena;
	path_bytes = -1;
	(void)snprintf(fd_path,sizeof(fd_path),"/proc/self/fd/%d",fileno(file));
	path_bytes = readlink(fd_path,arena->pack_path,sizeof(arena->pack_path) - 1u);
#ifdef __APPLE__
#ifndef F_GETPATH
#define F_GETPATH 50
#endif
	if (path_bytes <= 0)
	{
		char fcntl_path[1024];
		if (fcntl(fileno(file),F_GETPATH,fcntl_path) >= 0)
			path_bytes = (ssize_t)snprintf(arena->pack_path,
				sizeof(arena->pack_path),"%s",fcntl_path);
	}
#endif
	if (path_bytes <= 0 || fstat(fileno(file),&pack_stat) != 0 ||
		pack_stat.st_size <= 0)
	{
		arena->failed = 1;
		return 0;
	}
	arena->pack_path[path_bytes] = '\0';
	arena->pack_bytes = (uint64_t)pack_stat.st_size;
	memset(&slice,0,sizeof(slice));
	slice.model = ledger->module_tag;
	slice.pack_bytes = arena->pack_bytes;
	status = SparkWeightdAttachPack(&slice,arena->pack_path,
		SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,&arena->outcome,reason);
	if (status != SPARK_STATUS_OK || arena->outcome.client == 0)
	{
		fprintf(stderr,"stage-module weightd fallback: status=%s reason=%s\n",
			SparkStatusToString(status),reason);
		arena->failed = 1;
		return 0;
	}
	status = SparkWeightdAttachImportMap(&arena->outcome,arena->pack_bytes,
		SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,reason);
	if (status != SPARK_STATUS_OK || arena->outcome.map_base == 0)
	{
		if (arena->outcome.client != 0)
			(void)SparkWeightdAttachRelease(&arena->outcome);
		memset(&arena->outcome,0,sizeof(arena->outcome));
		arena->failed = 1;
		return 0;
	}
	return 1;
}

static int SparkStageModulePackArenaSlice(
	SparkStageModuleLedger *ledger,
	FILE *file,
	uint64_t offset,
	uint64_t bytes,
	void **pointer)
{
	SparkStageModulePackArena *arena;
	if ((offset & UINT64_C(0xff)) != 0u)
		return 0;
	if (SparkStageModulePackArenaEnsure(ledger,file) == 0)
		return 0;
	arena = (SparkStageModulePackArena *)ledger->pack_arena;
	if (offset > arena->pack_bytes || bytes > arena->pack_bytes - offset)
		return 0;
	*pointer = (uint8_t *)arena->outcome.map_base + offset;
	return 1;
}

static SparkStatus SparkStageModuleLoadRegionSynchronous(
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

#define SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS 2u

typedef struct SparkStageModuleLoadChunk
{
    uint64_t file_offset;
    uint64_t bytes;
    uint8_t *device_address;
    SparkStatus status;
} SparkStageModuleLoadChunk;

struct SparkStageModuleLoadPipeline
{
    const char *module_tag;
    int file_descriptor;
    pthread_t worker_thread;
    int worker_started;
    int worker_joined;
    pthread_mutex_t mutex;
    pthread_cond_t progress;
    void *slot_staging[SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS];
    cudaEvent_t slot_copy_events[SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS];
    cudaStream_t upload_stream;
    uint64_t slot_bytes;
    SparkStageModuleLoadChunk chunk_ring[SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS];
    uint64_t enqueued_count;
    uint64_t read_done_count;
    uint64_t copy_issued_count;
    int abort_requested;
    SparkStatus failure;
};

SparkStatus SparkStageModuleLoadPipelineRequested(void)
{
    const char *text = getenv("SPARK_STAGE_MODULE_LOAD_PIPELINE");
    return (text != 0 && text[0] == '0' && text[1] == '\0')
        ? SPARK_STATUS_BUSY
        : SPARK_STATUS_OK;
}

static SparkStatus SparkStageModuleLoadPipelineWaitForSlotCopyIssued(
    SparkStageModuleLoadPipeline *pipeline,
    uint64_t chunk_index)
{
    pthread_mutex_lock(&pipeline->mutex);
    while (pipeline->copy_issued_count + 1u + SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS <=
               chunk_index + 1u &&
           pipeline->failure == SPARK_STATUS_OK &&
           !pipeline->abort_requested)
    {
        pthread_cond_wait(&pipeline->progress, &pipeline->mutex);
    }
    pthread_mutex_unlock(&pipeline->mutex);
    if (chunk_index >= SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS)
    {
        cudaError_t error = cudaEventSynchronize(
            pipeline->slot_copy_events[
                chunk_index % SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS]);
        if (error != cudaSuccess)
        {
            return SparkStageModuleCudaStatus(
                pipeline->module_tag, error, "load_pipeline_slot_event");
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkStageModuleLoadPipelineReadChunk(
    SparkStageModuleLoadPipeline *pipeline,
    uint64_t file_offset,
    void *staging,
    uint64_t bytes)
{
    uint64_t moved = 0u;
    while (moved < bytes)
    {
        ssize_t read_bytes = pread(
            pipeline->file_descriptor,
            (uint8_t *)staging + moved,
            (size_t)(bytes - moved),
            (off_t)(file_offset + moved));
        if (read_bytes <= 0)
        {
            if (read_bytes < 0 && (errno == EINTR))
            {
                continue;
            }
            fprintf(
                stderr,
                "%s pack_read_failed offset=%llu bytes=%llu read=%llu\n",
                pipeline->module_tag,
                (unsigned long long)(file_offset + moved),
                (unsigned long long)bytes,
                (unsigned long long)moved);
            return SPARK_STATUS_IO_ERROR;
        }
        moved += (uint64_t)read_bytes;
    }
    return SPARK_STATUS_OK;
}

static void *SparkStageModuleLoadPipelineWorker(void *argument)
{
    SparkStageModuleLoadPipeline *pipeline =
        (SparkStageModuleLoadPipeline *)argument;

    for (;;)
    {
        uint64_t chunk_index;
        SparkStageModuleLoadChunk chunk;
        SparkStatus status;

        pthread_mutex_lock(&pipeline->mutex);
        while (pipeline->read_done_count == pipeline->enqueued_count &&
            pipeline->failure == SPARK_STATUS_OK &&
            !pipeline->abort_requested)
        {
            pthread_cond_wait(&pipeline->progress, &pipeline->mutex);
        }
        if (pipeline->read_done_count == pipeline->enqueued_count)
        {
            pthread_mutex_unlock(&pipeline->mutex);
            break;
        }
        chunk_index = pipeline->read_done_count;
        chunk = pipeline->chunk_ring[
            chunk_index % SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS];
        pthread_mutex_unlock(&pipeline->mutex);

        status = SparkStageModuleLoadPipelineWaitForSlotCopyIssued(
            pipeline, chunk_index);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkStageModuleLoadPipelineReadChunk(
                pipeline,
                chunk.file_offset,
                pipeline->slot_staging[
                    chunk_index % SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS],
                chunk.bytes);
        }

        pthread_mutex_lock(&pipeline->mutex);
        pipeline->chunk_ring[
            chunk_index % SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS].status =
            status;
        pipeline->read_done_count = chunk_index + 1u;
        if (status != SPARK_STATUS_OK &&
            pipeline->failure == SPARK_STATUS_OK)
        {
            pipeline->failure = status;
        }
        pthread_cond_broadcast(&pipeline->progress);
        pthread_mutex_unlock(&pipeline->mutex);
    }
    return 0;
}

SparkStatus SparkStageModuleLoadPipelineCreate(
    const char *module_tag,
    FILE *file,
    SparkStageModuleLoadPipeline **pipeline_pointer)
{
    SparkStageModuleLoadPipeline *pipeline;
    cudaError_t error;
    uint32_t slot;

    if (module_tag == 0 || module_tag[0] == '\0' || file == 0 ||
        pipeline_pointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *pipeline_pointer = 0;
    pipeline = (SparkStageModuleLoadPipeline *)calloc(1u, sizeof(*pipeline));
    if (pipeline == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    pipeline->module_tag = module_tag;
    pipeline->failure = SPARK_STATUS_OK;
    pipeline->file_descriptor = fileno(file);
    if (pipeline->file_descriptor < 0)
    {
        free(pipeline);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    pipeline->slot_bytes = SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES;
    if (pthread_mutex_init(&pipeline->mutex, 0) != 0)
    {
        free(pipeline);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (pthread_cond_init(&pipeline->progress, 0) != 0)
    {
        pthread_mutex_destroy(&pipeline->mutex);
        free(pipeline);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    error = cudaStreamCreateWithFlags(
        &pipeline->upload_stream, cudaStreamNonBlocking);
    for (slot = 0u; error == cudaSuccess &&
        slot < SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS; ++slot)
    {
        error = cudaEventCreateWithFlags(
            &pipeline->slot_copy_events[slot], cudaEventDisableTiming);
    }
    for (slot = 0u; error == cudaSuccess &&
        slot < SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS; ++slot)
    {
        error = cudaHostAlloc(
            &pipeline->slot_staging[slot],
            (size_t)pipeline->slot_bytes,
            cudaHostAllocDefault);
    }
    if (error != cudaSuccess)
    {
        SparkStageModuleCudaStatus(module_tag, error, "load_pipeline_create");
        SparkStageModuleLoadPipelineDestroy(pipeline);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (pthread_create(
            &pipeline->worker_thread, 0,
            SparkStageModuleLoadPipelineWorker, pipeline) != 0)
    {
        fprintf(stderr, "%s load_pipeline_worker_spawn_failed\n", module_tag);
        SparkStageModuleLoadPipelineDestroy(pipeline);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    pipeline->worker_started = 1;
    *pipeline_pointer = pipeline;
    return SPARK_STATUS_OK;
}

static int SparkStageModuleLoadPipelineIssueNext(
    SparkStageModuleLoadPipeline *pipeline)
{
    SparkStageModuleLoadChunk *chunk;
    cudaError_t error;
    if (pipeline->copy_issued_count >= pipeline->read_done_count)
    {
        return 0;
    }
    chunk = &pipeline->chunk_ring[
        pipeline->copy_issued_count % SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS];
    error = cudaMemcpyAsync(
        chunk->device_address,
        pipeline->slot_staging[
            pipeline->copy_issued_count % SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS],
        (size_t)chunk->bytes,
        cudaMemcpyHostToDevice,
        pipeline->upload_stream);
    if (error == cudaSuccess)
    {
        error = cudaEventRecord(
            pipeline->slot_copy_events[
                pipeline->copy_issued_count %
                SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS],
            pipeline->upload_stream);
    }
    if (error != cudaSuccess)
    {
        pipeline->failure = SparkStageModuleCudaStatus(
            pipeline->module_tag, error, "load_pipeline_h2d");
        return 0;
    }
    pipeline->copy_issued_count++;
    pthread_cond_broadcast(&pipeline->progress);
    return 1;
}

static SparkStatus SparkStageModuleLoadPipelineDrain(
    SparkStageModuleLoadPipeline *pipeline,
    int blocking)
{
    SparkStatus status = SPARK_STATUS_OK;
    pthread_mutex_lock(&pipeline->mutex);
    for (;;)
    {
        if (pipeline->failure != SPARK_STATUS_OK)
        {
            status = pipeline->failure;
            break;
        }
        if (SparkStageModuleLoadPipelineIssueNext(pipeline) != 0)
        {
            continue;
        }
        if (!blocking || pipeline->copy_issued_count == pipeline->enqueued_count)
        {
            break;
        }
        pthread_cond_wait(&pipeline->progress, &pipeline->mutex);
    }
    pthread_mutex_unlock(&pipeline->mutex);
    return status;
}

SparkStatus SparkStageModuleLoadPipelineRegion(
    SparkStageModuleLoadPipeline *pipeline,
    SparkStageModuleLedger *ledger,
    uint64_t offset,
    uint64_t bytes,
    void **pointer)
{
    void *device = 0;
    SparkStatus status;
    uint64_t moved;

    if (pipeline == 0 || ledger == 0 || pointer == 0 || bytes == 0u ||
        bytes > (uint64_t)SIZE_MAX || offset > UINT64_MAX - bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *pointer = 0;
    if (pipeline->failure != SPARK_STATUS_OK)
    {
        return pipeline->failure;
    }
    status = SparkStageModuleDeviceAllocate(ledger, bytes, &device);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SPARK_STATUS_OK;
    moved = 0u;
    while (status == SPARK_STATUS_OK && moved < bytes)
    {
        uint64_t chunk_bytes = bytes - moved;
        uint64_t slot;
        if (chunk_bytes > pipeline->slot_bytes)
        {
            chunk_bytes = pipeline->slot_bytes;
        }
        pthread_mutex_lock(&pipeline->mutex);
        while (pipeline->failure == SPARK_STATUS_OK &&
            pipeline->enqueued_count - pipeline->copy_issued_count >=
                SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS)
        {
            if (SparkStageModuleLoadPipelineIssueNext(pipeline) != 0)
            {
                continue;
            }
            pthread_cond_wait(&pipeline->progress, &pipeline->mutex);
        }
        if (pipeline->failure != SPARK_STATUS_OK)
        {
            pthread_mutex_unlock(&pipeline->mutex);
            status = pipeline->failure;
            break;
        }
        slot = pipeline->enqueued_count % SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS;
        pipeline->chunk_ring[slot].file_offset = offset + moved;
        pipeline->chunk_ring[slot].bytes = chunk_bytes;
        pipeline->chunk_ring[slot].device_address = (uint8_t *)device + moved;
        pipeline->chunk_ring[slot].status = SPARK_STATUS_INTERNAL_ERROR;
        pipeline->enqueued_count++;
        pthread_cond_broadcast(&pipeline->progress);
        pthread_mutex_unlock(&pipeline->mutex);
        moved += chunk_bytes;
    }
    if (status == SPARK_STATUS_OK)
        status = SparkStageModuleLoadPipelineDrain(pipeline, 0);
    if (status != SPARK_STATUS_OK)
    {
        SparkStageModuleReleaseLastAllocation(ledger, device);
        return status;
    }
    *pointer = device;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleLoadPipelineFinish(
    SparkStageModuleLoadPipeline *pipeline)
{
    SparkStatus status;
    cudaError_t error;

    if (pipeline == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkStageModuleLoadPipelineDrain(pipeline, 1);
    error = cudaStreamSynchronize(pipeline->upload_stream);
    if (error != cudaSuccess)
    {
        status = SparkStageModuleCudaStatus(
            pipeline->module_tag, error, "load_pipeline_finish");
    }
    pthread_mutex_lock(&pipeline->mutex);
    pipeline->abort_requested = 1;
    pthread_cond_broadcast(&pipeline->progress);
    pthread_mutex_unlock(&pipeline->mutex);
    if (pipeline->worker_started && !pipeline->worker_joined)
    {
        if (pthread_join(pipeline->worker_thread, 0) == 0)
        {
            pipeline->worker_joined = 1;
        }
    }
    if (status == SPARK_STATUS_OK && pipeline->failure != SPARK_STATUS_OK)
    {
        status = pipeline->failure;
    }
    return status;
}

void SparkStageModuleLoadPipelineDestroy(
    SparkStageModuleLoadPipeline *pipeline)
{
    uint32_t slot;
    if (pipeline == 0)
    {
        return;
    }
    if (pipeline->worker_started && !pipeline->worker_joined)
    {
        pthread_mutex_lock(&pipeline->mutex);
        pipeline->abort_requested = 1;
        pthread_cond_broadcast(&pipeline->progress);
        pthread_mutex_unlock(&pipeline->mutex);
        if (pthread_join(pipeline->worker_thread, 0) == 0)
        {
            pipeline->worker_joined = 1;
        }
    }
    if (pipeline->upload_stream != 0)
    {
        (void)cudaStreamSynchronize(pipeline->upload_stream);
        (void)cudaStreamDestroy(pipeline->upload_stream);
    }
    for (slot = 0u; slot < SPARK_STAGE_MODULE_LOAD_PIPELINE_SLOTS; ++slot)
    {
        if (pipeline->slot_copy_events[slot] != 0)
        {
            (void)cudaEventDestroy(pipeline->slot_copy_events[slot]);
        }
        if (pipeline->slot_staging[slot] != 0)
        {
            (void)cudaFreeHost(pipeline->slot_staging[slot]);
        }
    }
    pthread_cond_destroy(&pipeline->progress);
    pthread_mutex_destroy(&pipeline->mutex);
    free(pipeline);
}

SparkStatus SparkStageModuleLoadDeviceRegion(
    SparkStageModuleLedger *ledger,
    FILE *file,
    uint64_t offset,
    uint64_t bytes,
    void **pointer)
{
    SparkStageModuleLoadPipeline *pipeline = 0;
    SparkStatus status;

    if (SparkStageModulePackArenaSlice(ledger,file,offset,bytes,pointer) != 0)
        return SPARK_STATUS_OK;
    if (SparkStageModuleLoadPipelineRequested() == SPARK_STATUS_OK &&
        bytes >= SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES)
    {
        status = SparkStageModuleLoadPipelineCreate(
            ledger->module_tag, file, &pipeline);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkStageModuleLoadPipelineRegion(
                pipeline, ledger, offset, bytes, pointer);
            if (status == SPARK_STATUS_OK)
            {
                status = SparkStageModuleLoadPipelineFinish(pipeline);
            }
            SparkStageModuleLoadPipelineDestroy(pipeline);
            return status;
        }
    }
    return SparkStageModuleLoadRegionSynchronous(
        ledger, file, offset, bytes, pointer);
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
