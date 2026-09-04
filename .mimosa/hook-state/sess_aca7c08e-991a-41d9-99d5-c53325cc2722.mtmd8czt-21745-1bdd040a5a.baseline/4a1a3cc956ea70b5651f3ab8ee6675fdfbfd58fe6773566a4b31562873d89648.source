#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_memlink.h"
#include "sparkpipe/spark_status.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define SPARK_MEMLINK_WIRE_MAGIC 0x534d454d4c494e4bull
#define SPARK_MEMLINK_WIRE_VERSION 1u
#define SPARK_MEMLINK_OP_PUT 1u
#define SPARK_MEMLINK_OP_GET 2u
#define SPARK_MEMLINK_OP_STAT 3u
#define SPARK_MEMLINK_STATUS_OK 0u
#define SPARK_MEMLINK_STATUS_ERROR 1u
#define SPARK_MEMLINK_STATUS_NOT_FOUND 2u
#define SPARK_MEMLINK_STATUS_BUSY 3u
#define SPARK_MEMLINK_STATUS_CAPACITY 4u
#define SPARK_MEMLINK_STATUS_INVALID 5u

typedef struct SparkMemlinkWireHeader
{
    uint64_t magic;
    uint32_t version;
    uint32_t operation;
    uint32_t status;
    uint32_t lane_count;
    uint32_t lane_index;
    uint32_t key_bytes;
    uint64_t transfer_id;
    uint64_t total_bytes;
    uint64_t offset;
    uint64_t byte_count;
} SparkMemlinkWireHeader;

typedef struct SparkMemlinkStoredObject
{
    char key[SPARK_MEMLINK_MAX_KEY_BYTES + 1u];
    uint8_t *data;
    uint64_t total_bytes;
    uint64_t received_bytes;
    uint64_t transfer_id;
    uint64_t fifo_sequence;
    uint32_t active_readers;
    uint32_t active_writers;
    int complete;
    struct SparkMemlinkStoredObject *next;
} SparkMemlinkStoredObject;

typedef struct SparkMemlinkDaemonState
{
    pthread_mutex_t mutex;
    SparkMemlinkStoredObject *objects;
    uint64_t store_capacity_bytes;
    uint64_t current_bytes;
    uint64_t next_fifo_sequence;
    char bind_host[SPARK_MEMLINK_MAX_HOST_BYTES];
    uint16_t base_port;
    uint32_t lane_count;
} SparkMemlinkDaemonState;

typedef struct SparkMemlinkLaneServer
{
    SparkMemlinkDaemonState *state;
    uint32_t lane_index;
} SparkMemlinkLaneServer;

typedef struct SparkMemlinkConnection
{
    SparkMemlinkDaemonState *state;
    int socket_fd;
    uint32_t lane_index;
} SparkMemlinkConnection;

typedef struct SparkMemlinkClientTransfer
{
    SparkMemlinkEndpoint endpoint;
    const char *key;
    const uint8_t *source_data;
    uint8_t *destination_data;
    uint64_t total_bytes;
    uint64_t transfer_id;
    uint32_t operation;
    volatile int failed;
} SparkMemlinkClientTransfer;

typedef struct SparkMemlinkClientLane
{
    SparkMemlinkClientTransfer *transfer;
    SparkMemlinkTransferPartition partition;
} SparkMemlinkClientLane;

typedef struct SparkMemlinkCommandOptions
{
    const char *command;
    const char *host;
    const char *bind_host;
    const char *host_template;
    const char *key;
    const char *input_path;
    const char *output_path;
    uint16_t base_port;
    uint32_t lane_count;
    uint32_t rank;
    uint32_t rank_count;
    uint64_t store_bytes;
    uint64_t generated_bytes;
} SparkMemlinkCommandOptions;

static uint64_t SparkMemlinkNowNanoseconds(void)
{
    struct timespec time_value;

    if (clock_gettime(CLOCK_MONOTONIC, &time_value) != 0)
    {
        return 0ull;
    }

    return ((uint64_t)time_value.tv_sec * 1000000000ull) + (uint64_t)time_value.tv_nsec;
}

static uint64_t SparkMemlinkMakeTransferId(void)
{
    uint64_t now;
    uint64_t pid_value;

    now = SparkMemlinkNowNanoseconds();
    pid_value = (uint64_t)(uint32_t)getpid();

    return now ^ (pid_value << 32u) ^ 0x9e3779b97f4a7c15ull;
}

static int SparkMemlinkParseUint32(const char *text, uint32_t *value)
{
    char *end_pointer;
    unsigned long parsed;

    if (text == NULL || value == NULL || text[0] == '\0')
    {
        return 0;
    }

    errno = 0;
    parsed = strtoul(text, &end_pointer, 10);
    if (errno != 0 || end_pointer == text || *end_pointer != '\0' || parsed > UINT32_MAX)
    {
        return 0;
    }

    *value = (uint32_t)parsed;
    return 1;
}

static int SparkMemlinkParseUint16(const char *text, uint16_t *value)
{
    uint32_t parsed;

    if (!SparkMemlinkParseUint32(text, &parsed) || parsed > UINT16_MAX)
    {
        return 0;
    }

    *value = (uint16_t)parsed;
    return 1;
}

static int SparkMemlinkParseSize(const char *text, uint64_t *value)
{
    char *end_pointer;
    unsigned long long parsed;
    uint64_t multiplier;

    if (text == NULL || value == NULL || text[0] == '\0')
    {
        return 0;
    }

    errno = 0;
    parsed = strtoull(text, &end_pointer, 10);
    if (errno != 0 || end_pointer == text)
    {
        return 0;
    }

    multiplier = 1ull;
    if (*end_pointer != '\0')
    {
        if ((end_pointer[1] != '\0' && end_pointer[1] != 'b' && end_pointer[1] != 'B') ||
            (end_pointer[1] != '\0' && end_pointer[2] != '\0'))
        {
            return 0;
        }

        switch (*end_pointer)
        {
            case 'k':
            case 'K':
                multiplier = 1024ull;
                break;
            case 'm':
            case 'M':
                multiplier = 1024ull * 1024ull;
                break;
            case 'g':
            case 'G':
                multiplier = 1024ull * 1024ull * 1024ull;
                break;
            case 't':
            case 'T':
                multiplier = 1024ull * 1024ull * 1024ull * 1024ull;
                break;
            default:
                return 0;
        }
    }

    if (parsed > ULLONG_MAX / multiplier)
    {
        return 0;
    }

    *value = (uint64_t)(parsed * multiplier);
    return 1;
}

static int SparkMemlinkSetSocketOptions(int socket_fd)
{
    int enabled;
    int buffer_bytes;

    enabled = 1;
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
#ifdef SO_REUSEPORT
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_REUSEPORT, &enabled, sizeof(enabled));
#endif
    (void)setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
#ifdef SO_NOSIGPIPE
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif

    buffer_bytes = 32 * 1024 * 1024;
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes));
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes));

    return 0;
}

static int SparkMemlinkReadExact(int socket_fd, void *buffer, size_t byte_count)
{
    uint8_t *cursor;
    size_t remaining;

    cursor = (uint8_t *)buffer;
    remaining = byte_count;

    while (remaining > 0u)
    {
        ssize_t result;

        result = recv(socket_fd, cursor, remaining, 0);
        if (result == 0)
        {
            return 0;
        }
        if (result < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return 0;
        }

        cursor += (size_t)result;
        remaining -= (size_t)result;
    }

    return 1;
}

static int SparkMemlinkWriteExact(int socket_fd, const void *buffer, size_t byte_count)
{
    const uint8_t *cursor;
    size_t remaining;

    cursor = (const uint8_t *)buffer;
    remaining = byte_count;

    while (remaining > 0u)
    {
        ssize_t result;

        result = send(socket_fd, cursor, remaining, MSG_NOSIGNAL);
        if (result < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return 0;
        }

        cursor += (size_t)result;
        remaining -= (size_t)result;
    }

    return 1;
}

static int SparkMemlinkReadFile(const char *path, uint8_t **data, uint64_t *byte_count)
{
    FILE *file;
    struct stat file_stat;
    uint8_t *buffer;
    size_t read_count;

    if (path == NULL || data == NULL || byte_count == NULL)
    {
        return 0;
    }

    if (strcmp(path, "-") == 0)
    {
        size_t capacity;
        size_t used;

        capacity = 1024u * 1024u;
        used = 0u;
        buffer = (uint8_t *)malloc(capacity);
        if (buffer == NULL)
        {
            return 0;
        }

        for (;;)
        {
            size_t available;
            size_t result;

            if (used == capacity)
            {
                uint8_t *new_buffer;
                size_t new_capacity;

                if (capacity > ((size_t)-1) / 2u)
                {
                    free(buffer);
                    return 0;
                }

                new_capacity = capacity * 2u;
                new_buffer = (uint8_t *)realloc(buffer, new_capacity);
                if (new_buffer == NULL)
                {
                    free(buffer);
                    return 0;
                }

                buffer = new_buffer;
                capacity = new_capacity;
            }

            available = capacity - used;
            result = fread(buffer + used, 1u, available, stdin);
            used += result;

            if (result < available)
            {
                if (ferror(stdin))
                {
                    free(buffer);
                    return 0;
                }
                break;
            }
        }

        *data = buffer;
        *byte_count = (uint64_t)used;
        return 1;
    }

    if (stat(path, &file_stat) != 0 || file_stat.st_size < 0)
    {
        return 0;
    }

    file = fopen(path, "rb");
    if (file == NULL)
    {
        return 0;
    }

    buffer = NULL;
    if (file_stat.st_size > 0)
    {
        buffer = (uint8_t *)malloc((size_t)file_stat.st_size);
        if (buffer == NULL)
        {
            fclose(file);
            return 0;
        }

        read_count = fread(buffer, 1u, (size_t)file_stat.st_size, file);
        if (read_count != (size_t)file_stat.st_size)
        {
            free(buffer);
            fclose(file);
            return 0;
        }
    }
    else
    {
        buffer = (uint8_t *)malloc(1u);
        if (buffer == NULL)
        {
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    *data = buffer;
    *byte_count = (uint64_t)file_stat.st_size;
    return 1;
}

static int SparkMemlinkWriteFile(const char *path, const uint8_t *data, uint64_t byte_count)
{
    FILE *file;
    size_t written;

    if (path == NULL || data == NULL)
    {
        return 0;
    }

    if (strcmp(path, "-") == 0)
    {
        file = stdout;
    }
    else
    {
        file = fopen(path, "wb");
        if (file == NULL)
        {
            return 0;
        }
    }

    written = fwrite(data, 1u, (size_t)byte_count, file);
    if (written != (size_t)byte_count)
    {
        if (file != stdout)
        {
            fclose(file);
        }
        return 0;
    }

    if (file != stdout)
    {
        fclose(file);
    }

    return 1;
}

static uint8_t *SparkMemlinkGeneratePattern(uint64_t byte_count)
{
    uint8_t *data;
    uint64_t index;

    data = (uint8_t *)malloc(byte_count == 0ull ? 1u : (size_t)byte_count);
    if (data == NULL)
    {
        return NULL;
    }

    for (index = 0; index < byte_count; ++index)
    {
        data[index] = (uint8_t)((index * 131ull + 17ull) & 0xffu);
    }

    return data;
}

static SparkMemlinkStoredObject *SparkMemlinkFindObject(SparkMemlinkDaemonState *state, const char *key)
{
    SparkMemlinkStoredObject *object;

    object = state->objects;
    while (object != NULL)
    {
        if (strcmp(object->key, key) == 0)
        {
            return object;
        }
        object = object->next;
    }

    return NULL;
}

static void SparkMemlinkRemoveObjectLocked(SparkMemlinkDaemonState *state, SparkMemlinkStoredObject *target)
{
    SparkMemlinkStoredObject **link;

    link = &state->objects;
    while (*link != NULL)
    {
        if (*link == target)
        {
            *link = target->next;
            state->current_bytes -= target->total_bytes;
            free(target->data);
            free(target);
            return;
        }
        link = &(*link)->next;
    }
}

static int SparkMemlinkEvictForCapacityLocked(
    SparkMemlinkDaemonState *state,
    uint64_t required_bytes,
    const char *protected_key)
{
    while (state->current_bytes + required_bytes > state->store_capacity_bytes)
    {
        SparkMemlinkStoredObject *best_object;
        SparkMemlinkStoredObject *object;

        best_object = NULL;
        object = state->objects;
        while (object != NULL)
        {
            if (object->complete && object->active_readers == 0u && object->active_writers == 0u &&
                (protected_key == NULL || strcmp(object->key, protected_key) != 0))
            {
                if (best_object == NULL || object->fifo_sequence < best_object->fifo_sequence)
                {
                    best_object = object;
                }
            }
            object = object->next;
        }

        if (best_object == NULL)
        {
            return 0;
        }

        SparkMemlinkRemoveObjectLocked(state, best_object);
    }

    return 1;
}

static SparkMemlinkStoredObject *SparkMemlinkCreateObjectLocked(
    SparkMemlinkDaemonState *state,
    const char *key,
    uint64_t total_bytes,
    uint64_t transfer_id)
{
    SparkMemlinkStoredObject *object;

    if (total_bytes > state->store_capacity_bytes)
    {
        return NULL;
    }

    if (!SparkMemlinkEvictForCapacityLocked(state, total_bytes, key))
    {
        return NULL;
    }

    object = (SparkMemlinkStoredObject *)calloc(1u, sizeof(*object));
    if (object == NULL)
    {
        return NULL;
    }

    object->data = (uint8_t *)malloc(total_bytes == 0ull ? 1u : (size_t)total_bytes);
    if (object->data == NULL)
    {
        free(object);
        return NULL;
    }

    snprintf(object->key, sizeof(object->key), "%s", key);
    object->total_bytes = total_bytes;
    object->transfer_id = transfer_id;
    object->fifo_sequence = state->next_fifo_sequence++;
    object->next = state->objects;
    state->objects = object;
    state->current_bytes += total_bytes;

    return object;
}

static uint32_t SparkMemlinkMapStatus(SparkStatus status)
{
    if (status == SPARK_STATUS_OK)
    {
        return SPARK_MEMLINK_STATUS_OK;
    }
    if (status == SPARK_STATUS_NOT_FOUND)
    {
        return SPARK_MEMLINK_STATUS_NOT_FOUND;
    }
    if (status == SPARK_STATUS_BUSY)
    {
        return SPARK_MEMLINK_STATUS_BUSY;
    }
    if (status == SPARK_STATUS_CAPACITY_EXCEEDED)
    {
        return SPARK_MEMLINK_STATUS_CAPACITY;
    }
    return SPARK_MEMLINK_STATUS_ERROR;
}

static SparkStatus SparkMemlinkServerReadKey(int socket_fd, const SparkMemlinkWireHeader *header, char *key, size_t key_capacity)
{
    if (header->key_bytes == 0u || header->key_bytes > SPARK_MEMLINK_MAX_KEY_BYTES || key_capacity <= header->key_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (!SparkMemlinkReadExact(socket_fd, key, header->key_bytes))
    {
        return SPARK_STATUS_IO_ERROR;
    }

    key[header->key_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static int SparkMemlinkSendStatus(int socket_fd, const SparkMemlinkWireHeader *request_header, SparkStatus status, uint64_t total_bytes)
{
    SparkMemlinkWireHeader response_header;

    memset(&response_header, 0, sizeof(response_header));
    response_header.magic = SPARK_MEMLINK_WIRE_MAGIC;
    response_header.version = SPARK_MEMLINK_WIRE_VERSION;
    response_header.operation = request_header->operation;
    response_header.status = SparkMemlinkMapStatus(status);
    response_header.lane_count = request_header->lane_count;
    response_header.lane_index = request_header->lane_index;
    response_header.transfer_id = request_header->transfer_id;
    response_header.total_bytes = total_bytes;
    response_header.offset = request_header->offset;
    response_header.byte_count = request_header->byte_count;

    return SparkMemlinkWriteExact(socket_fd, &response_header, sizeof(response_header));
}

static SparkStatus SparkMemlinkHandlePut(
    SparkMemlinkDaemonState *state,
    int socket_fd,
    const SparkMemlinkWireHeader *header,
    const char *key)
{
    SparkMemlinkStoredObject *object;
    SparkStatus status;
    uint64_t end_offset;

    if (header->lane_count == 0u || header->lane_count > SPARK_MEMLINK_MAX_LANE_COUNT ||
        header->lane_index >= header->lane_count || header->offset > header->total_bytes ||
        header->byte_count > header->total_bytes - header->offset)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    end_offset = header->offset + header->byte_count;
    pthread_mutex_lock(&state->mutex);
    object = SparkMemlinkFindObject(state, key);
    if (object != NULL && object->transfer_id != header->transfer_id)
    {
        if (!object->complete || object->active_readers != 0u || object->active_writers != 0u)
        {
            pthread_mutex_unlock(&state->mutex);
            return SPARK_STATUS_BUSY;
        }
        SparkMemlinkRemoveObjectLocked(state, object);
        object = NULL;
    }

    if (object == NULL)
    {
        object = SparkMemlinkCreateObjectLocked(state, key, header->total_bytes, header->transfer_id);
        if (object == NULL)
        {
            pthread_mutex_unlock(&state->mutex);
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
    }

    if (object->total_bytes != header->total_bytes || end_offset > object->total_bytes)
    {
        pthread_mutex_unlock(&state->mutex);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    object->active_writers++;
    pthread_mutex_unlock(&state->mutex);

    status = SPARK_STATUS_OK;
    if (!SparkMemlinkReadExact(socket_fd, object->data + header->offset, (size_t)header->byte_count))
    {
        status = SPARK_STATUS_IO_ERROR;
    }

    pthread_mutex_lock(&state->mutex);
    if (status == SPARK_STATUS_OK)
    {
        object->received_bytes += header->byte_count;
        if (object->received_bytes >= object->total_bytes)
        {
            object->complete = 1;
            object->fifo_sequence = state->next_fifo_sequence++;
            fprintf(stderr, "memlink object complete key=%s bytes=%" PRIu64 "\n", object->key, object->total_bytes);
        }
    }
    object->active_writers--;
    pthread_mutex_unlock(&state->mutex);

    return status;
}

static SparkStatus SparkMemlinkHandleGet(
    SparkMemlinkDaemonState *state,
    int socket_fd,
    const SparkMemlinkWireHeader *header,
    const char *key)
{
    SparkMemlinkStoredObject *object;
    SparkStatus status;
    uint64_t end_offset;

    if (header->lane_count == 0u || header->lane_count > SPARK_MEMLINK_MAX_LANE_COUNT ||
        header->lane_index >= header->lane_count || header->offset > header->total_bytes ||
        header->byte_count > header->total_bytes - header->offset)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    end_offset = header->offset + header->byte_count;
    pthread_mutex_lock(&state->mutex);
    object = SparkMemlinkFindObject(state, key);
    if (object == NULL || !object->complete)
    {
        pthread_mutex_unlock(&state->mutex);
        return SPARK_STATUS_NOT_FOUND;
    }

    if (object->total_bytes != header->total_bytes || end_offset > object->total_bytes)
    {
        pthread_mutex_unlock(&state->mutex);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    object->active_readers++;
    pthread_mutex_unlock(&state->mutex);

    if (!SparkMemlinkSendStatus(socket_fd, header, SPARK_STATUS_OK, object->total_bytes))
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    else if (!SparkMemlinkWriteExact(socket_fd, object->data + header->offset, (size_t)header->byte_count))
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    else
    {
        status = SPARK_STATUS_OK;
    }

    pthread_mutex_lock(&state->mutex);
    object->active_readers--;
    pthread_mutex_unlock(&state->mutex);

    return status;
}

static SparkStatus SparkMemlinkHandleStat(
    SparkMemlinkDaemonState *state,
    int socket_fd,
    const SparkMemlinkWireHeader *header,
    const char *key)
{
    SparkMemlinkStoredObject *object;
    uint64_t total_bytes;

    pthread_mutex_lock(&state->mutex);
    object = SparkMemlinkFindObject(state, key);
    if (object == NULL || !object->complete)
    {
        pthread_mutex_unlock(&state->mutex);
        return SPARK_STATUS_NOT_FOUND;
    }
    total_bytes = object->total_bytes;
    pthread_mutex_unlock(&state->mutex);

    if (!SparkMemlinkSendStatus(socket_fd, header, SPARK_STATUS_OK, total_bytes))
    {
        return SPARK_STATUS_IO_ERROR;
    }

    return SPARK_STATUS_OK;
}

static void *SparkMemlinkConnectionThread(void *argument)
{
    SparkMemlinkConnection *connection;
    SparkMemlinkWireHeader header;
    char key[SPARK_MEMLINK_MAX_KEY_BYTES + 1u];
    SparkStatus status;

    connection = (SparkMemlinkConnection *)argument;
    status = SPARK_STATUS_OK;

    if (!SparkMemlinkReadExact(connection->socket_fd, &header, sizeof(header)))
    {
        close(connection->socket_fd);
        free(connection);
        return NULL;
    }

    if (header.magic != SPARK_MEMLINK_WIRE_MAGIC || header.version != SPARK_MEMLINK_WIRE_VERSION)
    {
        status = SPARK_STATUS_INVALID_ARGUMENT;
    }
    else
    {
        status = SparkMemlinkServerReadKey(connection->socket_fd, &header, key, sizeof(key));
    }

    if (status == SPARK_STATUS_OK)
    {
        if (header.operation == SPARK_MEMLINK_OP_PUT)
        {
            status = SparkMemlinkHandlePut(connection->state, connection->socket_fd, &header, key);
            (void)SparkMemlinkSendStatus(connection->socket_fd, &header, status, header.total_bytes);
        }
        else if (header.operation == SPARK_MEMLINK_OP_GET)
        {
            status = SparkMemlinkHandleGet(connection->state, connection->socket_fd, &header, key);
            if (status != SPARK_STATUS_OK)
            {
                (void)SparkMemlinkSendStatus(connection->socket_fd, &header, status, 0ull);
            }
        }
        else if (header.operation == SPARK_MEMLINK_OP_STAT)
        {
            status = SparkMemlinkHandleStat(connection->state, connection->socket_fd, &header, key);
            if (status != SPARK_STATUS_OK)
            {
                (void)SparkMemlinkSendStatus(connection->socket_fd, &header, status, 0ull);
            }
        }
        else
        {
            (void)SparkMemlinkSendStatus(connection->socket_fd, &header, SPARK_STATUS_INVALID_ARGUMENT, 0ull);
        }
    }

    close(connection->socket_fd);
    free(connection);
    return NULL;
}

static int SparkMemlinkCreateListenSocket(const char *bind_host, uint16_t port)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    char port_text[16];
    int socket_fd;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    snprintf(port_text, sizeof(port_text), "%u", (unsigned)port);

    if (getaddrinfo(bind_host, port_text, &hints, &result) != 0)
    {
        return -1;
    }

    socket_fd = -1;
    entry = result;
    while (entry != NULL)
    {
        socket_fd = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
        if (socket_fd >= 0)
        {
            SparkMemlinkSetSocketOptions(socket_fd);
            if (bind(socket_fd, entry->ai_addr, entry->ai_addrlen) == 0 && listen(socket_fd, 1024) == 0)
            {
                break;
            }
            close(socket_fd);
            socket_fd = -1;
        }
        entry = entry->ai_next;
    }

    freeaddrinfo(result);
    return socket_fd;
}

static void *SparkMemlinkLaneAcceptThread(void *argument)
{
    SparkMemlinkLaneServer *lane_server;
    int listen_socket;
    uint16_t port;

    lane_server = (SparkMemlinkLaneServer *)argument;
    port = (uint16_t)(lane_server->state->base_port + lane_server->lane_index);
    listen_socket = SparkMemlinkCreateListenSocket(lane_server->state->bind_host, port);
    if (listen_socket < 0)
    {
        fprintf(stderr, "memlink lane listen failed lane=%u port=%u\n", lane_server->lane_index, (unsigned)port);
        return NULL;
    }

    fprintf(stderr, "memlink lane listening lane=%u host=%s port=%u\n", lane_server->lane_index, lane_server->state->bind_host, (unsigned)port);

    for (;;)
    {
        SparkMemlinkConnection *connection;
        pthread_t thread;
        int client_socket;

        client_socket = accept(listen_socket, NULL, NULL);
        if (client_socket < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        SparkMemlinkSetSocketOptions(client_socket);
        connection = (SparkMemlinkConnection *)calloc(1u, sizeof(*connection));
        if (connection == NULL)
        {
            close(client_socket);
            continue;
        }

        connection->state = lane_server->state;
        connection->socket_fd = client_socket;
        connection->lane_index = lane_server->lane_index;

        if (pthread_create(&thread, NULL, SparkMemlinkConnectionThread, connection) != 0)
        {
            close(client_socket);
            free(connection);
            continue;
        }
        pthread_detach(thread);
    }

    close(listen_socket);
    return NULL;
}

static int SparkMemlinkConnect(const char *host, uint16_t port)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    char port_text[16];
    int socket_fd;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_text, sizeof(port_text), "%u", (unsigned)port);

    if (getaddrinfo(host, port_text, &hints, &result) != 0)
    {
        return -1;
    }

    socket_fd = -1;
    entry = result;
    while (entry != NULL)
    {
        socket_fd = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
        if (socket_fd >= 0)
        {
            SparkMemlinkSetSocketOptions(socket_fd);
            if (connect(socket_fd, entry->ai_addr, entry->ai_addrlen) == 0)
            {
                break;
            }
            close(socket_fd);
            socket_fd = -1;
        }
        entry = entry->ai_next;
    }

    freeaddrinfo(result);
    return socket_fd;
}

static int SparkMemlinkClientSendRequestHeader(
    int socket_fd,
    const SparkMemlinkClientTransfer *transfer,
    const SparkMemlinkTransferPartition *partition)
{
    SparkMemlinkWireHeader header;
    size_t key_bytes;

    key_bytes = strlen(transfer->key);
    if (key_bytes == 0u || key_bytes > SPARK_MEMLINK_MAX_KEY_BYTES)
    {
        return 0;
    }

    memset(&header, 0, sizeof(header));
    header.magic = SPARK_MEMLINK_WIRE_MAGIC;
    header.version = SPARK_MEMLINK_WIRE_VERSION;
    header.operation = transfer->operation;
    header.status = 0u;
    header.lane_count = transfer->endpoint.lane_count;
    header.lane_index = partition->lane_index;
    header.key_bytes = (uint32_t)key_bytes;
    header.transfer_id = transfer->transfer_id;
    header.total_bytes = transfer->total_bytes;
    header.offset = partition->offset;
    header.byte_count = partition->byte_count;

    if (!SparkMemlinkWriteExact(socket_fd, &header, sizeof(header)))
    {
        return 0;
    }

    return SparkMemlinkWriteExact(socket_fd, transfer->key, key_bytes);
}

static int SparkMemlinkClientReadStatus(int socket_fd, uint32_t expected_operation, uint64_t *total_bytes)
{
    SparkMemlinkWireHeader response_header;

    if (!SparkMemlinkReadExact(socket_fd, &response_header, sizeof(response_header)))
    {
        return 0;
    }

    if (response_header.magic != SPARK_MEMLINK_WIRE_MAGIC ||
        response_header.version != SPARK_MEMLINK_WIRE_VERSION ||
        response_header.operation != expected_operation ||
        response_header.status != SPARK_MEMLINK_STATUS_OK)
    {
        return 0;
    }

    if (total_bytes != NULL)
    {
        *total_bytes = response_header.total_bytes;
    }

    return 1;
}

static void *SparkMemlinkClientLaneThread(void *argument)
{
    SparkMemlinkClientLane *lane;
    SparkMemlinkClientTransfer *transfer;
    int socket_fd;
    uint16_t port;

    lane = (SparkMemlinkClientLane *)argument;
    transfer = lane->transfer;
    port = (uint16_t)(transfer->endpoint.base_port + lane->partition.lane_index);
    socket_fd = SparkMemlinkConnect(transfer->endpoint.host, port);
    if (socket_fd < 0)
    {
        transfer->failed = 1;
        return NULL;
    }

    if (!SparkMemlinkClientSendRequestHeader(socket_fd, transfer, &lane->partition))
    {
        transfer->failed = 1;
        close(socket_fd);
        return NULL;
    }

    if (transfer->operation == SPARK_MEMLINK_OP_PUT)
    {
        if (!SparkMemlinkWriteExact(
                socket_fd,
                transfer->source_data + lane->partition.offset,
                (size_t)lane->partition.byte_count) ||
            !SparkMemlinkClientReadStatus(socket_fd, SPARK_MEMLINK_OP_PUT, NULL))
        {
            transfer->failed = 1;
        }
    }
    else if (transfer->operation == SPARK_MEMLINK_OP_GET)
    {
        if (!SparkMemlinkClientReadStatus(socket_fd, SPARK_MEMLINK_OP_GET, NULL) ||
            !SparkMemlinkReadExact(
                socket_fd,
                transfer->destination_data + lane->partition.offset,
                (size_t)lane->partition.byte_count))
        {
            transfer->failed = 1;
        }
    }
    else
    {
        transfer->failed = 1;
    }

    close(socket_fd);
    return NULL;
}

static int SparkMemlinkClientRunParallelTransfer(SparkMemlinkClientTransfer *transfer)
{
    pthread_t threads[SPARK_MEMLINK_MAX_LANE_COUNT];
    SparkMemlinkClientLane lanes[SPARK_MEMLINK_MAX_LANE_COUNT];
    uint32_t lane_index;
    int created_count;

    if (SparkMemlinkValidateLaneCount(transfer->endpoint.lane_count) != SPARK_STATUS_OK)
    {
        return 0;
    }

    created_count = 0;
    for (lane_index = 0; lane_index < transfer->endpoint.lane_count; ++lane_index)
    {
        if (SparkMemlinkBuildTransferPartition(
                transfer->total_bytes,
                transfer->endpoint.lane_count,
                lane_index,
                &lanes[lane_index].partition) != SPARK_STATUS_OK)
        {
            transfer->failed = 1;
            break;
        }

        lanes[lane_index].transfer = transfer;
        if (pthread_create(&threads[lane_index], NULL, SparkMemlinkClientLaneThread, &lanes[lane_index]) != 0)
        {
            transfer->failed = 1;
            break;
        }
        created_count++;
    }

    for (lane_index = 0; lane_index < (uint32_t)created_count; ++lane_index)
    {
        pthread_join(threads[lane_index], NULL);
    }

    return transfer->failed == 0;
}

static int SparkMemlinkClientStat(const SparkMemlinkEndpoint *endpoint, const char *key, uint64_t *total_bytes)
{
    SparkMemlinkClientTransfer transfer;
    SparkMemlinkTransferPartition partition;
    int socket_fd;
    uint16_t port;

    memset(&transfer, 0, sizeof(transfer));
    transfer.endpoint = *endpoint;
    transfer.key = key;
    transfer.operation = SPARK_MEMLINK_OP_STAT;
    transfer.transfer_id = SparkMemlinkMakeTransferId();

    partition.lane_index = 0u;
    partition.lane_count = endpoint->lane_count;
    partition.offset = 0ull;
    partition.byte_count = 0ull;

    port = endpoint->base_port;
    socket_fd = SparkMemlinkConnect(endpoint->host, port);
    if (socket_fd < 0)
    {
        return 0;
    }

    if (!SparkMemlinkClientSendRequestHeader(socket_fd, &transfer, &partition) ||
        !SparkMemlinkClientReadStatus(socket_fd, SPARK_MEMLINK_OP_STAT, total_bytes))
    {
        close(socket_fd);
        return 0;
    }

    close(socket_fd);
    return 1;
}

static void SparkMemlinkInitializeOptions(SparkMemlinkCommandOptions *options)
{
    memset(options, 0, sizeof(*options));
    options->bind_host = "0.0.0.0";
    options->host_template = "spark%x";
    options->base_port = (uint16_t)SPARK_MEMLINK_DEFAULT_BASE_PORT;
    options->lane_count = SPARK_MEMLINK_DEFAULT_LANE_COUNT;
    options->rank_count = 13u;
    options->store_bytes = SPARK_MEMLINK_DEFAULT_STORE_BYTES;
}

static void SparkMemlinkPrintUsage(FILE *stream)
{
    fprintf(stream,
        "usage:\n"
        "  sparkpipe_memlink daemon [--bind 0.0.0.0] [--base-port 55200] [--lanes 8] [--store-bytes 64G]\n"
        "  sparkpipe_memlink put --host HOST --key KEY (--input PATH | --bytes SIZE) [--base-port P] [--lanes N]\n"
        "  sparkpipe_memlink get --host HOST --key KEY --output PATH [--base-port P] [--lanes N]\n"
        "  sparkpipe_memlink prevcp --rank R --key KEY (--input PATH | --bytes SIZE) [--host-template spark%%x]\n"
        "  sparkpipe_memlink nextcp --rank R --key KEY (--input PATH | --bytes SIZE) [--host-template spark%%x]\n"
        "  sparkpipe_memlink stat --host HOST --key KEY [--base-port P]\n");
}

static int SparkMemlinkParseOptions(int argc, char **argv, SparkMemlinkCommandOptions *options)
{
    int index;

    SparkMemlinkInitializeOptions(options);
    if (argc < 2)
    {
        return 0;
    }

    options->command = argv[1];
    index = 2;
    while (index < argc)
    {
        const char *name;
        const char *value;

        name = argv[index++];
        if (index >= argc)
        {
            return 0;
        }
        value = argv[index++];

        if (strcmp(name, "--host") == 0)
        {
            options->host = value;
        }
        else if (strcmp(name, "--bind") == 0)
        {
            options->bind_host = value;
        }
        else if (strcmp(name, "--host-template") == 0)
        {
            options->host_template = value;
        }
        else if (strcmp(name, "--key") == 0)
        {
            options->key = value;
        }
        else if (strcmp(name, "--input") == 0)
        {
            options->input_path = value;
        }
        else if (strcmp(name, "--output") == 0)
        {
            options->output_path = value;
        }
        else if (strcmp(name, "--base-port") == 0)
        {
            if (!SparkMemlinkParseUint16(value, &options->base_port))
            {
                return 0;
            }
        }
        else if (strcmp(name, "--lanes") == 0)
        {
            if (!SparkMemlinkParseUint32(value, &options->lane_count))
            {
                return 0;
            }
        }
        else if (strcmp(name, "--rank") == 0)
        {
            if (!SparkMemlinkParseUint32(value, &options->rank))
            {
                return 0;
            }
        }
        else if (strcmp(name, "--rank-count") == 0)
        {
            if (!SparkMemlinkParseUint32(value, &options->rank_count))
            {
                return 0;
            }
        }
        else if (strcmp(name, "--store-bytes") == 0)
        {
            if (!SparkMemlinkParseSize(value, &options->store_bytes))
            {
                return 0;
            }
        }
        else if (strcmp(name, "--bytes") == 0)
        {
            if (!SparkMemlinkParseSize(value, &options->generated_bytes))
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }

    return SparkMemlinkValidateLaneCount(options->lane_count) == SPARK_STATUS_OK;
}

static int SparkMemlinkRunDaemon(const SparkMemlinkCommandOptions *options)
{
    SparkMemlinkDaemonState state;
    SparkMemlinkLaneServer lane_servers[SPARK_MEMLINK_MAX_LANE_COUNT];
    pthread_t lane_threads[SPARK_MEMLINK_MAX_LANE_COUNT];
    uint32_t lane_index;

    memset(&state, 0, sizeof(state));
    pthread_mutex_init(&state.mutex, NULL);
    snprintf(state.bind_host, sizeof(state.bind_host), "%s", options->bind_host);
    state.base_port = options->base_port;
    state.lane_count = options->lane_count;
    state.store_capacity_bytes = options->store_bytes;
    state.next_fifo_sequence = 1ull;

    fprintf(stderr,
        "memlink daemon start bind=%s base_port=%u lanes=%u store_bytes=%" PRIu64 "\n",
        state.bind_host,
        (unsigned)state.base_port,
        state.lane_count,
        state.store_capacity_bytes);

    for (lane_index = 0; lane_index < state.lane_count; ++lane_index)
    {
        lane_servers[lane_index].state = &state;
        lane_servers[lane_index].lane_index = lane_index;
        if (pthread_create(&lane_threads[lane_index], NULL, SparkMemlinkLaneAcceptThread, &lane_servers[lane_index]) != 0)
        {
            fprintf(stderr, "memlink failed to start lane=%u\n", lane_index);
            return 1;
        }
    }

    for (lane_index = 0; lane_index < state.lane_count; ++lane_index)
    {
        pthread_join(lane_threads[lane_index], NULL);
    }

    return 0;
}

static int SparkMemlinkBuildEndpointFromOptions(
    const SparkMemlinkCommandOptions *options,
    SparkMemlinkNeighborDirection *optional_direction,
    SparkMemlinkEndpoint *endpoint)
{
    SparkStatus status;

    if (optional_direction != NULL)
    {
        status = SparkMemlinkResolveNeighborEndpoint(
            options->rank,
            options->rank_count,
            *optional_direction,
            options->host_template,
            options->base_port,
            options->lane_count,
            endpoint);
        return status == SPARK_STATUS_OK;
    }

    if (options->host == NULL)
    {
        return 0;
    }

    snprintf(endpoint->host, sizeof(endpoint->host), "%s", options->host);
    endpoint->base_port = options->base_port;
    endpoint->lane_count = options->lane_count;
    return 1;
}

static int SparkMemlinkRunPutLike(
    const SparkMemlinkCommandOptions *options,
    SparkMemlinkNeighborDirection *optional_direction)
{
    SparkMemlinkEndpoint endpoint;
    SparkMemlinkClientTransfer transfer;
    uint8_t *data;
    uint64_t byte_count;
    uint64_t start_time;
    uint64_t end_time;
    double seconds;
    double gib_per_second;

    if (options->key == NULL || (options->input_path == NULL && options->generated_bytes == 0ull))
    {
        return 1;
    }

    if (!SparkMemlinkBuildEndpointFromOptions(options, optional_direction, &endpoint))
    {
        return 1;
    }

    data = NULL;
    byte_count = 0ull;
    if (options->input_path != NULL)
    {
        if (!SparkMemlinkReadFile(options->input_path, &data, &byte_count))
        {
            fprintf(stderr, "memlink failed to read input\n");
            return 1;
        }
    }
    else
    {
        byte_count = options->generated_bytes;
        data = SparkMemlinkGeneratePattern(byte_count);
        if (data == NULL)
        {
            fprintf(stderr, "memlink failed to allocate generated buffer\n");
            return 1;
        }
    }

    memset(&transfer, 0, sizeof(transfer));
    transfer.endpoint = endpoint;
    transfer.key = options->key;
    transfer.source_data = data;
    transfer.total_bytes = byte_count;
    transfer.transfer_id = SparkMemlinkMakeTransferId();
    transfer.operation = SPARK_MEMLINK_OP_PUT;

    start_time = SparkMemlinkNowNanoseconds();
    if (!SparkMemlinkClientRunParallelTransfer(&transfer))
    {
        free(data);
        fprintf(stderr, "memlink put failed host=%s key=%s\n", endpoint.host, options->key);
        return 1;
    }
    end_time = SparkMemlinkNowNanoseconds();

    seconds = (double)(end_time - start_time) / 1000000000.0;
    gib_per_second = seconds > 0.0 ? ((double)byte_count / (1024.0 * 1024.0 * 1024.0)) / seconds : 0.0;

    fprintf(stderr,
        "memlink put complete host=%s key=%s bytes=%" PRIu64 " lanes=%u seconds=%.6f GiBps=%.3f\n",
        endpoint.host,
        options->key,
        byte_count,
        endpoint.lane_count,
        seconds,
        gib_per_second);

    free(data);
    return 0;
}

static int SparkMemlinkRunGet(const SparkMemlinkCommandOptions *options)
{
    SparkMemlinkEndpoint endpoint;
    SparkMemlinkClientTransfer transfer;
    uint8_t *data;
    uint64_t byte_count;
    uint64_t start_time;
    uint64_t end_time;
    double seconds;
    double gib_per_second;

    if (options->key == NULL || options->output_path == NULL || !SparkMemlinkBuildEndpointFromOptions(options, NULL, &endpoint))
    {
        return 1;
    }

    if (!SparkMemlinkClientStat(&endpoint, options->key, &byte_count))
    {
        fprintf(stderr, "memlink stat before get failed host=%s key=%s\n", endpoint.host, options->key);
        return 1;
    }

    data = (uint8_t *)malloc(byte_count == 0ull ? 1u : (size_t)byte_count);
    if (data == NULL)
    {
        return 1;
    }

    memset(&transfer, 0, sizeof(transfer));
    transfer.endpoint = endpoint;
    transfer.key = options->key;
    transfer.destination_data = data;
    transfer.total_bytes = byte_count;
    transfer.transfer_id = SparkMemlinkMakeTransferId();
    transfer.operation = SPARK_MEMLINK_OP_GET;

    start_time = SparkMemlinkNowNanoseconds();
    if (!SparkMemlinkClientRunParallelTransfer(&transfer))
    {
        free(data);
        fprintf(stderr, "memlink get failed host=%s key=%s\n", endpoint.host, options->key);
        return 1;
    }
    end_time = SparkMemlinkNowNanoseconds();

    if (!SparkMemlinkWriteFile(options->output_path, data, byte_count))
    {
        free(data);
        fprintf(stderr, "memlink failed to write output\n");
        return 1;
    }

    seconds = (double)(end_time - start_time) / 1000000000.0;
    gib_per_second = seconds > 0.0 ? ((double)byte_count / (1024.0 * 1024.0 * 1024.0)) / seconds : 0.0;

    fprintf(stderr,
        "memlink get complete host=%s key=%s bytes=%" PRIu64 " lanes=%u seconds=%.6f GiBps=%.3f\n",
        endpoint.host,
        options->key,
        byte_count,
        endpoint.lane_count,
        seconds,
        gib_per_second);

    free(data);
    return 0;
}

static int SparkMemlinkRunStat(const SparkMemlinkCommandOptions *options)
{
    SparkMemlinkEndpoint endpoint;
    uint64_t byte_count;

    if (options->key == NULL || !SparkMemlinkBuildEndpointFromOptions(options, NULL, &endpoint))
    {
        return 1;
    }

    if (!SparkMemlinkClientStat(&endpoint, options->key, &byte_count))
    {
        fprintf(stderr, "memlink stat failed host=%s key=%s\n", endpoint.host, options->key);
        return 1;
    }

    printf("key=%s bytes=%" PRIu64 "\n", options->key, byte_count);
    return 0;
}

int main(int argc, char **argv)
{
    SparkMemlinkCommandOptions options;
    SparkMemlinkNeighborDirection direction;
#ifdef SPARK_MEMLINK_FIXED_COMMAND
    char **shifted_argv;
    int shifted_argc;
    int shifted_index;
#endif

    signal(SIGPIPE, SIG_IGN);

#ifdef SPARK_MEMLINK_FIXED_COMMAND
    shifted_argv = NULL;
    if (argc < 2 || argv[1][0] == '-')
    {
        shifted_argc = argc + 1;
        shifted_argv = (char **)calloc((size_t)shifted_argc + 1u, sizeof(char *));
        if (shifted_argv == NULL)
        {
            return 2;
        }
        shifted_argv[0] = argv[0];
        shifted_argv[1] = (char *)SPARK_MEMLINK_FIXED_COMMAND;
        for (shifted_index = 1; shifted_index < argc; ++shifted_index)
        {
            shifted_argv[shifted_index + 1] = argv[shifted_index];
        }
        argc = shifted_argc;
        argv = shifted_argv;
    }
#endif

    if (!SparkMemlinkParseOptions(argc, argv, &options))
    {
        SparkMemlinkPrintUsage(stderr);
#ifdef SPARK_MEMLINK_FIXED_COMMAND
        free(shifted_argv);
#endif
        return 2;
    }

    if (strcmp(options.command, "daemon") == 0)
    {
        return SparkMemlinkRunDaemon(&options);
    }

    if (strcmp(options.command, "put") == 0)
    {
        return SparkMemlinkRunPutLike(&options, NULL);
    }

    if (strcmp(options.command, "get") == 0)
    {
        return SparkMemlinkRunGet(&options);
    }

    if (strcmp(options.command, "stat") == 0)
    {
        return SparkMemlinkRunStat(&options);
    }

    if (strcmp(options.command, "prevcp") == 0)
    {
        direction = SPARK_MEMLINK_NEIGHBOR_PREVIOUS;
        return SparkMemlinkRunPutLike(&options, &direction);
    }

    if (strcmp(options.command, "nextcp") == 0)
    {
        direction = SPARK_MEMLINK_NEIGHBOR_NEXT;
        return SparkMemlinkRunPutLike(&options, &direction);
    }

    SparkMemlinkPrintUsage(stderr);
    return 2;
}
