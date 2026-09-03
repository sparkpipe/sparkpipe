#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_tp_collective.h"

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SPARK_TEST_TPC_MAX_DEGREE 16u
#define SPARK_TEST_TPC_STANDARD_ELEMENTS (6u * 1024u)
#define SPARK_TEST_TPC_LARGE_ELEMENTS (16u * SPARK_TEST_TPC_STANDARD_ELEMENTS)
#define SPARK_TEST_TPC_CONNECT_TIMEOUT_MILLI 2000u
#define SPARK_TEST_TPC_OPERATION_TIMEOUT_MILLI 10000u
#define SPARK_TEST_TPC_FAILURE_TIMEOUT_MILLI 200u
#define SPARK_TEST_TPC_PORT_MINIMUM 24000u
#define SPARK_TEST_TPC_PORT_MAXIMUM 60000u
#define SPARK_TEST_TPC_PORT_SEARCH_ATTEMPTS 4096u
#define SPARK_TEST_TPC_PORT_SEARCH_STRIDE 17u
#define SPARK_TEST_TPC_PROCESS_LOCK_PATH "/tmp/sparkpipe-test-tp-collective.lock"

typedef struct SparkTestTpcBarrier
{
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    uint32_t participant_count;
    uint32_t arrived_count;
    uint32_t generation;
} SparkTestTpcBarrier;

static void SparkTestTpcBarrierInitialize(
    SparkTestTpcBarrier *barrier,
    uint32_t participant_count)
{
    assert(barrier != NULL);
    assert(participant_count > 0u);
    memset(barrier, 0, sizeof(*barrier));
    assert(pthread_mutex_init(&barrier->mutex, NULL) == 0);
    assert(pthread_cond_init(&barrier->condition, NULL) == 0);
    barrier->participant_count = participant_count;
}

static void SparkTestTpcBarrierWait(SparkTestTpcBarrier *barrier)
{
    uint32_t generation;

    assert(barrier != NULL);
    assert(pthread_mutex_lock(&barrier->mutex) == 0);
    generation = barrier->generation;
    barrier->arrived_count += 1u;
    if (barrier->arrived_count == barrier->participant_count)
    {
        barrier->arrived_count = 0u;
        barrier->generation += 1u;
        assert(pthread_cond_broadcast(&barrier->condition) == 0);
    }
    else
    {
        while (generation == barrier->generation)
        {
            assert(pthread_cond_wait(
                       &barrier->condition,
                       &barrier->mutex) == 0);
        }
    }
    assert(pthread_mutex_unlock(&barrier->mutex) == 0);
}

static void SparkTestTpcBarrierDestroy(SparkTestTpcBarrier *barrier)
{
    assert(barrier != NULL);
    assert(barrier->arrived_count == 0u);
    assert(pthread_cond_destroy(&barrier->condition) == 0);
    assert(pthread_mutex_destroy(&barrier->mutex) == 0);
}

typedef struct SparkTestTpcThread
{
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint16_t port_base;
    uint64_t collective_identifier;
    uint64_t element_count;
    uint32_t operation_count;
    uint32_t use_bf16;
    float *values;
    float *scratch;
    uint16_t *values_bf16;
    uint16_t *scratch_bf16;
    SparkStatus status;
    SparkTestTpcBarrier *barrier;
} SparkTestTpcThread;

static int SparkTestTpcAcquireProcessLock(void)
{
    struct flock process_lock;
    int lock_descriptor;
    int lock_status;

    lock_descriptor = open(
        SPARK_TEST_TPC_PROCESS_LOCK_PATH,
        O_CREAT | O_RDWR | O_CLOEXEC,
        0600);
    assert(lock_descriptor >= 0);

    memset(&process_lock, 0, sizeof(process_lock));
    process_lock.l_type = F_WRLCK;
    process_lock.l_whence = SEEK_SET;
    do
    {
        lock_status = fcntl(lock_descriptor, F_SETLKW, &process_lock);
    }
    while (lock_status != 0 && errno == EINTR);
    assert(lock_status == 0);

    return lock_descriptor;
}

static void SparkTestTpcReleaseProcessLock(int lock_descriptor)
{
    struct flock process_lock;
    int unlock_status;

    assert(lock_descriptor >= 0);
    memset(&process_lock, 0, sizeof(process_lock));
    process_lock.l_type = F_UNLCK;
    process_lock.l_whence = SEEK_SET;
    do
    {
        unlock_status = fcntl(lock_descriptor, F_SETLK, &process_lock);
    }
    while (unlock_status != 0 && errno == EINTR);
    assert(unlock_status == 0);
    assert(close(lock_descriptor) == 0);
}

static uint64_t SparkTestTpcMonotonicNanoseconds(void)
{
    struct timespec current_time;

    assert(clock_gettime(CLOCK_MONOTONIC, &current_time) == 0);
    return ((uint64_t)current_time.tv_sec * 1000000000u) +
        (uint64_t)current_time.tv_nsec;
}

static uint64_t SparkTestTpcNextCollectiveIdentifier(void)
{
    static uint64_t identifier_sequence = 1u;
    uint64_t identifier;

    identifier = ((uint64_t)(uint32_t)getpid() << 32u) ^
        SparkTestTpcMonotonicNanoseconds() ^
        identifier_sequence;
    identifier_sequence += 1u;
    if (identifier == 0u)
    {
        identifier = identifier_sequence;
    }

    return identifier;
}

static int SparkTestTpcPortRangeIsAvailable(
    uint16_t port_base,
    uint32_t port_count)
{
    int reservation_sockets[SPARK_TEST_TPC_MAX_DEGREE];
    struct sockaddr_in reservation_address;
    uint32_t port_index;
    int range_is_available;

    assert(port_count <= SPARK_TEST_TPC_MAX_DEGREE);
    for (port_index = 0u; port_index < SPARK_TEST_TPC_MAX_DEGREE; ++port_index)
    {
        reservation_sockets[port_index] = -1;
    }

    range_is_available = 1;
    for (port_index = 0u; port_index < port_count; ++port_index)
    {
        reservation_sockets[port_index] = socket(AF_INET, SOCK_STREAM, 0);
        if (reservation_sockets[port_index] < 0)
        {
            range_is_available = 0;
            break;
        }

        memset(&reservation_address, 0, sizeof(reservation_address));
        reservation_address.sin_family = AF_INET;
        assert(inet_pton(
                   AF_INET,
                   "127.0.0.1",
                   &reservation_address.sin_addr) == 1);
        reservation_address.sin_port = htons((uint16_t)(port_base + port_index));
        if (bind(
                reservation_sockets[port_index],
                (struct sockaddr *)&reservation_address,
                sizeof(reservation_address)) != 0)
        {
            range_is_available = 0;
            break;
        }
    }

    for (port_index = 0u; port_index < SPARK_TEST_TPC_MAX_DEGREE; ++port_index)
    {
        if (reservation_sockets[port_index] >= 0)
        {
            close(reservation_sockets[port_index]);
        }
    }

    return range_is_available;
}

static uint16_t SparkTestTpcFindAvailablePortBase(uint32_t port_count)
{
    static uint32_t search_sequence = 0u;
    uint32_t candidate_span;
    uint32_t first_candidate;
    uint32_t attempt_index;

    assert(port_count > 0u && port_count <= SPARK_TEST_TPC_MAX_DEGREE);
    candidate_span = SPARK_TEST_TPC_PORT_MAXIMUM -
        SPARK_TEST_TPC_PORT_MINIMUM -
        SPARK_TEST_TPC_MAX_DEGREE;
    first_candidate = SPARK_TEST_TPC_PORT_MINIMUM +
        (((uint32_t)getpid() + search_sequence * SPARK_TEST_TPC_PORT_SEARCH_STRIDE) %
         candidate_span);
    search_sequence += 1u;

    for (attempt_index = 0u;
         attempt_index < SPARK_TEST_TPC_PORT_SEARCH_ATTEMPTS;
         ++attempt_index)
    {
        uint32_t candidate_port;

        candidate_port = SPARK_TEST_TPC_PORT_MINIMUM +
            ((first_candidate - SPARK_TEST_TPC_PORT_MINIMUM +
              attempt_index * SPARK_TEST_TPC_PORT_SEARCH_STRIDE) %
             candidate_span);
        if (SparkTestTpcPortRangeIsAvailable(
                (uint16_t)candidate_port,
                port_count))
        {
            return (uint16_t)candidate_port;
        }
    }

    assert(!"no available TCP port range for tensor-parallel collective test");
    return 0u;
}

static void SparkTestTpcPopulateConfig(
    SparkTpCollectiveConfig *config,
    uint32_t tp_degree,
    uint32_t tp_rank,
    uint16_t port_base,
    uint64_t collective_identifier,
    uint32_t connect_timeout_milli,
    uint32_t operation_timeout_milli)
{
    uint32_t step_index;

    memset(config, 0, sizeof(*config));
    config->abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
    config->tp_degree = tp_degree;
    config->tp_rank = tp_rank;
    config->listen_port = (uint16_t)(port_base + tp_rank);
    config->connect_timeout_milli = connect_timeout_milli;
    config->operation_timeout_milli = operation_timeout_milli;
    config->collective_identifier = collective_identifier;

    for (step_index = 0u;
         (tp_degree >> (step_index + 1u)) != 0u;
         ++step_index)
    {
        uint32_t partner_rank;

        partner_rank = tp_rank ^ (1u << step_index);
        snprintf(
            config->peers[step_index].host_name,
            sizeof(config->peers[step_index].host_name),
            "%s",
            "127.0.0.1");
        config->peers[step_index].port =
            (uint16_t)(port_base + partner_rank);
    }
}

static void SparkTestTpcFill(
    float *values,
    uint64_t element_count,
    uint32_t tp_rank)
{
    uint64_t element_index;

    for (element_index = 0u; element_index < element_count; ++element_index)
    {
        values[element_index] =
            (float)((tp_rank + 1u) * (element_index % 7u + 1u));
    }
}

static uint16_t SparkTestTpcBf16FromUint(uint32_t value)
{
    float as_float;
    uint32_t bits;

    as_float = (float)value;
    memcpy(&bits, &as_float, sizeof(bits));
    return (uint16_t)(bits >> 16u);
}

static void SparkTestTpcFillBf16(
    uint16_t *values_bf16,
    uint64_t element_count,
    uint32_t tp_rank)
{
    uint64_t element_index;

    for (element_index = 0u; element_index < element_count; ++element_index)
    {
        values_bf16[element_index] = SparkTestTpcBf16FromUint(
            (tp_rank + 1u) * (uint32_t)(element_index % 2u + 1u));
    }
}

static void *SparkTestTpcMain(void *argument)
{
    SparkTestTpcThread *thread;
    SparkTpCollectiveConfig config;
    SparkTpCollective collective;
    int collective_was_created;

    thread = (SparkTestTpcThread *)argument;
    SparkTestTpcPopulateConfig(
        &config,
        thread->tp_degree,
        thread->tp_rank,
        thread->port_base,
        thread->collective_identifier,
        SPARK_TEST_TPC_CONNECT_TIMEOUT_MILLI,
        SPARK_TEST_TPC_OPERATION_TIMEOUT_MILLI);

    thread->status = SparkTpCollectiveCreate(&config, &collective);
    collective_was_created = thread->status == SPARK_STATUS_OK;
    SparkTestTpcBarrierWait(thread->barrier);

    if (collective_was_created)
    {
        uint32_t operation_index;

        for (operation_index = 0u;
             operation_index < thread->operation_count;
             ++operation_index)
        {
            if (thread->use_bf16 != 0u)
            {
                SparkTestTpcFillBf16(
                    thread->values_bf16,
                    thread->element_count,
                    thread->tp_rank);
                thread->status = SparkTpCollectiveAllReduceSumBf16(
                    &collective,
                    thread->values_bf16,
                    thread->element_count,
                    thread->scratch_bf16);
            }
            else
            {
                SparkTestTpcFill(
                    thread->values,
                    thread->element_count,
                    thread->tp_rank);
                thread->status = SparkTpCollectiveAllReduceSumF32(
                    &collective,
                    thread->values,
                    thread->element_count,
                    thread->scratch);
            }
            if (thread->status != SPARK_STATUS_OK)
            {
                break;
            }
        }
    }

    SparkTestTpcBarrierWait(thread->barrier);
    if (collective_was_created)
    {
        SparkTpCollectiveDestroy(&collective);
    }

    return NULL;
}

static void SparkTestTpcReportUnexpectedStatus(
    uint32_t rank_index,
    SparkStatus actual_status,
    SparkStatus expected_status)
{
    if (actual_status != expected_status)
    {
        fprintf(
            stderr,
            "tensor-parallel rank %u returned status %d; expected %d\n",
            rank_index,
            (int)actual_status,
            (int)expected_status);
    }
}

static void SparkTestTpcRun(
    uint32_t tp_degree,
    uint64_t element_count)
{
    pthread_t threads[SPARK_TEST_TPC_MAX_DEGREE];
    SparkTestTpcThread contexts[SPARK_TEST_TPC_MAX_DEGREE];
    SparkTestTpcBarrier barrier;
    float *storage;
    float *scratch_storage;
    uint16_t port_base;
    uint64_t collective_identifier;
    uint32_t rank_index;
    uint64_t element_index;

    assert(tp_degree <= SPARK_TEST_TPC_MAX_DEGREE);
    storage = (float *)malloc(
        (size_t)tp_degree * (size_t)element_count * sizeof(float));
    scratch_storage = (float *)malloc(
        (size_t)tp_degree * (size_t)element_count * sizeof(float));
    assert(storage != NULL && scratch_storage != NULL);

    port_base = SparkTestTpcFindAvailablePortBase(tp_degree);
    collective_identifier = SparkTestTpcNextCollectiveIdentifier();
    SparkTestTpcBarrierInitialize(&barrier, tp_degree);
    for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
    {
        memset(&contexts[rank_index], 0, sizeof(contexts[rank_index]));
        contexts[rank_index].tp_degree = tp_degree;
        contexts[rank_index].tp_rank = rank_index;
        contexts[rank_index].port_base = port_base;
        contexts[rank_index].collective_identifier = collective_identifier;
        contexts[rank_index].element_count = element_count;
        contexts[rank_index].operation_count = 2u;
        contexts[rank_index].values = storage +
            ((size_t)rank_index * (size_t)element_count);
        contexts[rank_index].scratch = scratch_storage +
            ((size_t)rank_index * (size_t)element_count);
        contexts[rank_index].status = SPARK_STATUS_INTERNAL_ERROR;
        contexts[rank_index].barrier = &barrier;
        assert(pthread_create(
            &threads[rank_index],
            NULL,
            SparkTestTpcMain,
            &contexts[rank_index]) == 0);
    }

    for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
    {
        assert(pthread_join(threads[rank_index], NULL) == 0);
    }
    SparkTestTpcBarrierDestroy(&barrier);

    for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
    {
        SparkTestTpcReportUnexpectedStatus(
            rank_index,
            contexts[rank_index].status,
            SPARK_STATUS_OK);
        assert(contexts[rank_index].status == SPARK_STATUS_OK);
    }

    for (element_index = 0u; element_index < element_count; ++element_index)
    {
        float expected_value;

        expected_value = (float)((element_index % 7u + 1u) *
            (tp_degree * (tp_degree + 1u) / 2u));
        assert(storage[element_index] == expected_value);
    }

    for (rank_index = 1u; rank_index < tp_degree; ++rank_index)
    {
        assert(memcmp(
            storage,
            storage + ((size_t)rank_index * (size_t)element_count),
            (size_t)element_count * sizeof(float)) == 0);
    }

    free(storage);
    free(scratch_storage);
}

static void SparkTestTpcRunBf16(
    uint32_t tp_degree,
    uint64_t element_count)
{
    pthread_t threads[SPARK_TEST_TPC_MAX_DEGREE];
    SparkTestTpcThread contexts[SPARK_TEST_TPC_MAX_DEGREE];
    SparkTestTpcBarrier barrier;
    uint16_t *storage;
    uint16_t *scratch_storage;
    uint16_t port_base;
    uint64_t collective_identifier;
    uint32_t rank_index;
    uint64_t element_index;

    assert(tp_degree <= SPARK_TEST_TPC_MAX_DEGREE);
    storage = (uint16_t *)malloc(
        (size_t)tp_degree * (size_t)element_count * sizeof(uint16_t));
    scratch_storage = (uint16_t *)malloc(
        (size_t)tp_degree * (size_t)element_count * sizeof(uint16_t));
    assert(storage != NULL && scratch_storage != NULL);

    port_base = SparkTestTpcFindAvailablePortBase(tp_degree);
    collective_identifier = SparkTestTpcNextCollectiveIdentifier();
    SparkTestTpcBarrierInitialize(&barrier, tp_degree);
    for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
    {
        memset(&contexts[rank_index], 0, sizeof(contexts[rank_index]));
        contexts[rank_index].tp_degree = tp_degree;
        contexts[rank_index].tp_rank = rank_index;
        contexts[rank_index].port_base = port_base;
        contexts[rank_index].collective_identifier = collective_identifier;
        contexts[rank_index].element_count = element_count;
        contexts[rank_index].operation_count = 2u;
        contexts[rank_index].use_bf16 = 1u;
        contexts[rank_index].values_bf16 = storage +
            ((size_t)rank_index * (size_t)element_count);
        contexts[rank_index].scratch_bf16 = scratch_storage +
            ((size_t)rank_index * (size_t)element_count);
        contexts[rank_index].status = SPARK_STATUS_INTERNAL_ERROR;
        contexts[rank_index].barrier = &barrier;
        assert(pthread_create(
            &threads[rank_index],
            NULL,
            SparkTestTpcMain,
            &contexts[rank_index]) == 0);
    }

    for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
    {
        assert(pthread_join(threads[rank_index], NULL) == 0);
    }
    SparkTestTpcBarrierDestroy(&barrier);

    for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
    {
        SparkTestTpcReportUnexpectedStatus(
            rank_index,
            contexts[rank_index].status,
            SPARK_STATUS_OK);
        assert(contexts[rank_index].status == SPARK_STATUS_OK);
    }

    for (element_index = 0u; element_index < element_count; ++element_index)
    {
        uint16_t expected_value;

        expected_value = SparkTestTpcBf16FromUint(
            (uint32_t)(element_index % 2u + 1u) *
            (tp_degree * (tp_degree + 1u) / 2u));
        assert(storage[element_index] == expected_value);
    }

    for (rank_index = 1u; rank_index < tp_degree; ++rank_index)
    {
        assert(memcmp(
            storage,
            storage + ((size_t)rank_index * (size_t)element_count),
            (size_t)element_count * sizeof(uint16_t)) == 0);
    }

    free(storage);
    free(scratch_storage);
}
static void SparkTestTpcDegreeOneNoOp(void)
{
    SparkTpCollectiveConfig config;
    SparkTpCollective collective;
    float value;
    uint16_t value_bf16;

    memset(&config, 0, sizeof(config));
    config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
    config.tp_degree = 1u;
    value = 42.0f;
    assert(SparkTpCollectiveCreate(&config, &collective) == SPARK_STATUS_OK);
    assert(SparkTpCollectiveAllReduceSumF32(
        &collective,
        &value,
        1u,
        NULL) == SPARK_STATUS_OK);
    assert(value == 42.0f);
    value_bf16 = SparkTestTpcBf16FromUint(42u);
    assert(SparkTpCollectiveAllReduceSumBf16(
        &collective,
        &value_bf16,
        1u,
        NULL) == SPARK_STATUS_OK);
    assert(value_bf16 == SparkTestTpcBf16FromUint(42u));
    SparkTpCollectiveDestroy(&collective);
    SparkTpCollectiveDestroy(&collective);
}

static void SparkTestTpcFailedCreateLeavesDestroySafeObject(void)
{
    SparkTpCollectiveConfig config;
    SparkTpCollective collective;

    memset(&config, 0, sizeof(config));
    memset(&collective, 0xa5, sizeof(collective));
    config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
    config.tp_degree = 3u;
    assert(SparkTpCollectiveCreate(
        &config,
        &collective) == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTpCollectiveDestroy(&collective);
    SparkTpCollectiveDestroy(&collective);
}

static void SparkTestTpcRejectsInvalidConfiguration(void)
{
    SparkTpCollectiveConfig config;
    SparkTpCollective collective;
    uint16_t port_base;

    memset(&config, 0, sizeof(config));
    config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
    config.tp_degree = 3u;
    assert(SparkTpCollectiveCreate(
        &config,
        &collective) == SPARK_STATUS_INVALID_ARGUMENT);

    config.tp_degree = 4u;
    config.tp_rank = 4u;
    assert(SparkTpCollectiveCreate(
        &config,
        &collective) == SPARK_STATUS_INVALID_ARGUMENT);

    port_base = SparkTestTpcFindAvailablePortBase(2u);
    SparkTestTpcPopulateConfig(
        &config,
        2u,
        0u,
        port_base,
        SparkTestTpcNextCollectiveIdentifier(),
        SPARK_TEST_TPC_CONNECT_TIMEOUT_MILLI,
        SPARK_TEST_TPC_OPERATION_TIMEOUT_MILLI);
    config.operation_timeout_milli = 0u;
    assert(SparkTpCollectiveCreate(
        &config,
        &collective) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestTpcPopulateConfig(
        &config,
        2u,
        0u,
        port_base,
        0u,
        SPARK_TEST_TPC_CONNECT_TIMEOUT_MILLI,
        SPARK_TEST_TPC_OPERATION_TIMEOUT_MILLI);
    assert(SparkTpCollectiveCreate(
        &config,
        &collective) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestTpcMissingPeerTimesOut(void)
{
    SparkTpCollectiveConfig config;
    SparkTpCollective collective;
    uint16_t port_base;
    uint64_t begin_nanoseconds;
    uint64_t elapsed_milliseconds;

    port_base = SparkTestTpcFindAvailablePortBase(2u);
    SparkTestTpcPopulateConfig(
        &config,
        2u,
        0u,
        port_base,
        SparkTestTpcNextCollectiveIdentifier(),
        SPARK_TEST_TPC_FAILURE_TIMEOUT_MILLI,
        SPARK_TEST_TPC_FAILURE_TIMEOUT_MILLI);

    begin_nanoseconds = SparkTestTpcMonotonicNanoseconds();
    assert(SparkTpCollectiveCreate(
        &config,
        &collective) == SPARK_STATUS_IO_ERROR);
    elapsed_milliseconds =
        (SparkTestTpcMonotonicNanoseconds() - begin_nanoseconds) / 1000000u;
    assert(elapsed_milliseconds <
        (uint64_t)SPARK_TEST_TPC_CONNECT_TIMEOUT_MILLI);
}

static void SparkTestTpcRejectsMismatchedElementCounts(void)
{
    pthread_t threads[2];
    SparkTestTpcThread contexts[2];
    SparkTestTpcBarrier barrier;
    float values[3];
    float scratch[3];
    uint16_t port_base;
    uint64_t collective_identifier;
    uint32_t rank_index;

    port_base = SparkTestTpcFindAvailablePortBase(2u);
    collective_identifier = SparkTestTpcNextCollectiveIdentifier();
    SparkTestTpcBarrierInitialize(&barrier, 2u);
    memset(values, 0, sizeof(values));
    memset(scratch, 0, sizeof(scratch));

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        memset(&contexts[rank_index], 0, sizeof(contexts[rank_index]));
        contexts[rank_index].tp_degree = 2u;
        contexts[rank_index].tp_rank = rank_index;
        contexts[rank_index].port_base = port_base;
        contexts[rank_index].collective_identifier = collective_identifier;
        contexts[rank_index].element_count = rank_index + 1u;
        contexts[rank_index].operation_count = 1u;
        contexts[rank_index].values = values + rank_index;
        contexts[rank_index].scratch = scratch + rank_index;
        contexts[rank_index].status = SPARK_STATUS_INTERNAL_ERROR;
        contexts[rank_index].barrier = &barrier;
        assert(pthread_create(
            &threads[rank_index],
            NULL,
            SparkTestTpcMain,
            &contexts[rank_index]) == 0);
    }

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        assert(pthread_join(threads[rank_index], NULL) == 0);
    }
    SparkTestTpcBarrierDestroy(&barrier);

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        SparkTestTpcReportUnexpectedStatus(
            rank_index,
            contexts[rank_index].status,
            SPARK_STATUS_VALIDATION_FAILED);
        assert(contexts[rank_index].status == SPARK_STATUS_VALIDATION_FAILED);
    }
}

static void SparkTestTpcRejectsMixedWireKinds(void)
{
    pthread_t threads[2];
    SparkTestTpcThread contexts[2];
    SparkTestTpcBarrier barrier;
    float values[2];
    float scratch[2];
    uint16_t values_bf16[2];
    uint16_t scratch_bf16[2];
    uint16_t port_base;
    uint64_t collective_identifier;
    uint32_t rank_index;

    port_base = SparkTestTpcFindAvailablePortBase(2u);
    collective_identifier = SparkTestTpcNextCollectiveIdentifier();
    SparkTestTpcBarrierInitialize(&barrier, 2u);
    memset(values, 0, sizeof(values));
    memset(scratch, 0, sizeof(scratch));
    memset(values_bf16, 0, sizeof(values_bf16));
    memset(scratch_bf16, 0, sizeof(scratch_bf16));

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        memset(&contexts[rank_index], 0, sizeof(contexts[rank_index]));
        contexts[rank_index].tp_degree = 2u;
        contexts[rank_index].tp_rank = rank_index;
        contexts[rank_index].port_base = port_base;
        contexts[rank_index].collective_identifier = collective_identifier;
        contexts[rank_index].element_count = 1u;
        contexts[rank_index].operation_count = 1u;
        contexts[rank_index].use_bf16 = rank_index;
        contexts[rank_index].values = values + rank_index;
        contexts[rank_index].scratch = scratch + rank_index;
        contexts[rank_index].values_bf16 = values_bf16 + rank_index;
        contexts[rank_index].scratch_bf16 = scratch_bf16 + rank_index;
        contexts[rank_index].status = SPARK_STATUS_INTERNAL_ERROR;
        contexts[rank_index].barrier = &barrier;
        assert(pthread_create(
            &threads[rank_index],
            NULL,
            SparkTestTpcMain,
            &contexts[rank_index]) == 0);
    }

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        assert(pthread_join(threads[rank_index], NULL) == 0);
    }
    SparkTestTpcBarrierDestroy(&barrier);

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        SparkTestTpcReportUnexpectedStatus(
            rank_index,
            contexts[rank_index].status,
            SPARK_STATUS_VALIDATION_FAILED);
        assert(contexts[rank_index].status == SPARK_STATUS_VALIDATION_FAILED);
    }
}

static void SparkTestTpcRejectsOverlappingBuffers(void)
{
    pthread_t threads[2];
    SparkTestTpcThread contexts[2];
    SparkTestTpcBarrier barrier;
    float values[2];
    uint16_t port_base;
    uint64_t collective_identifier;
    uint32_t rank_index;

    port_base = SparkTestTpcFindAvailablePortBase(2u);
    collective_identifier = SparkTestTpcNextCollectiveIdentifier();
    SparkTestTpcBarrierInitialize(&barrier, 2u);
    memset(values, 0, sizeof(values));

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        memset(&contexts[rank_index], 0, sizeof(contexts[rank_index]));
        contexts[rank_index].tp_degree = 2u;
        contexts[rank_index].tp_rank = rank_index;
        contexts[rank_index].port_base = port_base;
        contexts[rank_index].collective_identifier = collective_identifier;
        contexts[rank_index].element_count = 1u;
        contexts[rank_index].operation_count = 1u;
        contexts[rank_index].values = &values[rank_index];
        contexts[rank_index].scratch = &values[rank_index];
        contexts[rank_index].status = SPARK_STATUS_INTERNAL_ERROR;
        contexts[rank_index].barrier = &barrier;
        assert(pthread_create(
            &threads[rank_index],
            NULL,
            SparkTestTpcMain,
            &contexts[rank_index]) == 0);
    }

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        assert(pthread_join(threads[rank_index], NULL) == 0);
    }
    SparkTestTpcBarrierDestroy(&barrier);

    for (rank_index = 0u; rank_index < 2u; ++rank_index)
    {
        SparkTestTpcReportUnexpectedStatus(
            rank_index,
            contexts[rank_index].status,
            SPARK_STATUS_INVALID_ARGUMENT);
        assert(contexts[rank_index].status == SPARK_STATUS_INVALID_ARGUMENT);
    }
}

int main(void)
{
    int process_lock_descriptor;

    process_lock_descriptor = SparkTestTpcAcquireProcessLock();
    SparkTestTpcDegreeOneNoOp();
    SparkTestTpcFailedCreateLeavesDestroySafeObject();
    SparkTestTpcRejectsInvalidConfiguration();
    SparkTestTpcMissingPeerTimesOut();
    SparkTestTpcRejectsMismatchedElementCounts();
    SparkTestTpcRejectsMixedWireKinds();
    SparkTestTpcRejectsOverlappingBuffers();
    SparkTestTpcRun(2u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcRun(4u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcRun(8u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcRun(16u, SPARK_TEST_TPC_LARGE_ELEMENTS);
    SparkTestTpcRunBf16(2u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcRunBf16(4u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcRunBf16(16u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcRun(8u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcRun(8u, SPARK_TEST_TPC_STANDARD_ELEMENTS);
    SparkTestTpcReleaseProcessLock(process_lock_descriptor);
    return 0;
}
