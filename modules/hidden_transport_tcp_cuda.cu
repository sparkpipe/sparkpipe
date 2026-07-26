#include "sparkpipe/spark_hidden_transport.h"
#include <cuda_runtime_api.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SPARK_HIDDEN_TCP_CUDA_MAGIC 0x53484354u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_BASE 55400u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_OFFSET 100u
#define SPARK_HIDDEN_TCP_CUDA_HOST_BYTES 64u
#define SPARK_HIDDEN_TCP_CUDA_NO_PENDING 0xffffffffu
#define SPARK_HIDDEN_TCP_CUDA_PENDING_DEPTH 8u
#define SPARK_HIDDEN_TCP_CUDA_PENDING_HASH_LOAD_FACTOR 2u
#define SPARK_HIDDEN_TCP_CUDA_OUTGOING_DEPTH 8u
#define SPARK_HIDDEN_TCP_CUDA_CONNECT_RETRY_MS 50u
#define SPARK_HIDDEN_TCP_CUDA_DESTROY_DRAIN_TIMEOUT_MS 5000u

typedef struct SparkHiddenTcpCudaHeader
{
    uint32_t magic;
    uint32_t active_sequence_count;
    uint32_t hidden_dimension;
    uint32_t bytes_per_sequence;
    uint64_t sequence_id;
    uint64_t token_index;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
} SparkHiddenTcpCudaHeader;

typedef struct SparkHiddenTcpCudaPendingPacket
{
    uint32_t active;
    uint32_t next_pending_index;
    uint32_t next_free_index;
    SparkHiddenTcpCudaHeader header;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint8_t *payload;
} SparkHiddenTcpCudaPendingPacket;

typedef struct SparkHiddenTcpCudaOutgoingSlot
{
    SparkHiddenTcpCudaHeader header;
    uint64_t bytes;
    uint8_t *payload;
} SparkHiddenTcpCudaOutgoingSlot;

typedef struct SparkHiddenTcpCudaState
{
    SparkHiddenTransportEndpoint endpoint;
    int32_t local_rank;
    int32_t source_rank;
    int32_t sink_rank;
    uint32_t port_base;
    uint32_t is_sender;
    int listen_fd;
    int socket_fd;
    int event_fd;
    uint32_t stop;
    uint32_t thread_running;
    pthread_t io_thread;
    pthread_mutex_t lock;
    pthread_cond_t outgoing_cond;
    pthread_cond_t pending_free_cond;
    SparkHiddenTcpCudaOutgoingSlot outgoing_slots[SPARK_HIDDEN_TCP_CUDA_OUTGOING_DEPTH];
    uint32_t outgoing_head;
    uint32_t outgoing_count;
    SparkHiddenTcpCudaPendingPacket *pending_packets;
    uint32_t *pending_hash_heads;
    uint32_t pending_hash_slots;
    uint32_t pending_depth;
    uint32_t free_pending_head;
    uint64_t slot_payload_bytes;
    SparkHiddenTransportCompletionQueue completion_queue;
    uint32_t debug_enabled;
    char source_host[SPARK_HIDDEN_TCP_CUDA_HOST_BYTES];
    char sink_host[SPARK_HIDDEN_TCP_CUDA_HOST_BYTES];
} SparkHiddenTcpCudaState;

static SparkStatus SparkHiddenTcpCudaPushCompletionLocked(
    SparkHiddenTcpCudaState *state,
    const SparkHiddenTcpCudaHeader *header,
    SparkStatus status,
    uint64_t transfer_bytes)
{
    SparkHiddenTransportCompletion completion;
    memset(&completion,0,sizeof(completion));
    completion.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    completion.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    completion.status = status;
    completion.active_sequence_count = header->active_sequence_count;
    completion.sequence_id = header->sequence_id;
    completion.token_index = header->token_index;
    completion.transfer_bytes = transfer_bytes;
    return SparkHiddenTransportCompletionQueuePush(
        &state->completion_queue,&completion);
}

static uint64_t SparkHiddenTcpCudaHashBytes(
    const void *data,
    uint64_t bytes,
    uint64_t seed)
{
    const uint8_t *cursor;
    uint64_t hash;
    uint64_t offset;
    cursor = (const uint8_t *)data;
    hash = seed ^ 0xcbf29ce484222325ull;
    for (offset = 0u; offset < bytes; ++offset)
    {
        hash ^= (uint64_t)cursor[offset];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

static void SparkHiddenTcpCudaMaybeDumpPayload(
    int32_t local_rank,
    const char *direction,
    uint64_t sequence_id,
    uint64_t token_index,
    const void *payload,
    uint64_t bytes)
{
    static int32_t resolved = 0;
    static const char *dump_dir = 0;
    char path[256];
    FILE *file;
    int32_t close_status;
    int32_t written_bytes;
    uint64_t written_payload_bytes;
    if (resolved == 0)
    {
        dump_dir = getenv("SPARKPIPE_HIDDEN_DUMP_DIR");
        if (dump_dir != 0 && mkdir(dump_dir,0755) != 0 && errno != EEXIST)
        {
            fprintf(stderr,"hidden_dump_directory_failed path=%s errno=%d\n",
                dump_dir,errno);
            dump_dir = 0;
        }
        resolved = 1;
    }
    if (dump_dir == 0 || payload == 0 || bytes == 0u)
        return;
    written_bytes = snprintf(path,sizeof(path),
        "%s/rank%d_%s_seq%llu_tok%llu.bin",
        dump_dir,local_rank,direction,
        (unsigned long long)sequence_id,
        (unsigned long long)token_index);
    if (written_bytes < 0 || (uint32_t)written_bytes >= sizeof(path))
    {
        fprintf(stderr,"hidden_dump_path_too_long rank=%d seq=%llu token=%llu\n",
            local_rank,(unsigned long long)sequence_id,
            (unsigned long long)token_index);
        return;
    }
    file = fopen(path,"wb");
    if (file == 0)
    {
        fprintf(stderr,"hidden_dump_open_failed path=%s errno=%d\n",path,errno);
        return;
    }
    written_payload_bytes = (uint64_t)fwrite(payload,1u,(size_t)bytes,file);
    close_status = fclose(file);
    if (written_payload_bytes != bytes || close_status != 0)
        fprintf(stderr,"hidden_dump_write_failed path=%s errno=%d\n",path,errno);
}

static uint64_t SparkHiddenTcpCudaPayloadHash(
    const void *payload,
    uint64_t bytes)
{
    if (payload == 0 || bytes == 0u)
        return 0u;
    return SparkHiddenTcpCudaHashBytes(payload,bytes,0u);
}

static uint32_t SparkHiddenTcpCudaKeyHash(
    uint64_t sequence_id,
    uint64_t token_index)
{
    uint64_t hash;
    hash = SparkHiddenTcpCudaHashBytes(&sequence_id,sizeof(sequence_id),
        token_index);
    return (uint32_t)(hash ^ (hash >> 32));
}

static uint32_t SparkHiddenTcpCudaHeaderHash(
    const SparkHiddenTcpCudaHeader *header)
{
    return SparkHiddenTcpCudaKeyHash(header->sequence_id,header->token_index);
}

static uint32_t SparkHiddenTcpCudaPacketHash(
    const SparkHiddenTransportPacket *packet)
{
    return SparkHiddenTcpCudaKeyHash(packet->sequence_id,packet->token_index);
}

static uint32_t SparkHiddenTcpCudaHeaderMatchesPacket(
    const SparkHiddenTcpCudaHeader *header,
    const SparkHiddenTransportPacket *packet)
{
    if (header == 0 || packet == 0)
        return 0u;
    return header->magic == SPARK_HIDDEN_TCP_CUDA_MAGIC &&
        header->sequence_id == packet->sequence_id &&
        header->token_index == packet->token_index &&
        header->active_sequence_count == packet->active_sequence_count &&
        header->hidden_dimension == packet->hidden_dimension &&
        header->bytes_per_sequence == packet->bytes_per_sequence &&
        header->sideband_kind == packet->sideband_kind &&
        header->sideband_bytes_per_sequence ==
            packet->sideband_bytes_per_sequence;
}

static uint32_t SparkHiddenTcpCudaParseUintEnv(
    const char *name,
    uint32_t fallback)
{
    const char *text;
    char *end;
    unsigned long value;
    text = getenv(name);
    if (text == 0 || text[0] == '\0')
        return fallback;
    end = 0;
    value = strtoul(text,&end,10);
    if (end == 0 || *end != '\0')
        return fallback;
    return (uint32_t)value;
}

static int32_t SparkHiddenTcpCudaRankFromHost(const char *host)
{
    uint32_t tail;
    char extra;
    if (host == 0 || host[0] != 's' || host[1] != 'p' || host[2] != 'a' ||
        host[3] != 'r' || host[4] != 'k' || host[5] == '\0')
    {
        extra = '\0';
        if (sscanf(host,"10.10.100.%u%c",&tail,&extra) == 1 &&
            tail >= 10u && tail <= 22u)
            return (int32_t)(tail - 10u);
        return -1;
    }
    if (host[5] >= '0' && host[5] <= '9' && host[6] == '\0')
        return (int32_t)(host[5] - '0');
    if (host[5] >= 'a' && host[5] <= 'c' && host[6] == '\0')
        return (int32_t)(10 + (host[5] - 'a'));
    return -1;
}

static SparkStatus SparkHiddenTcpCudaParseRoute(
    const char *route_name,
    char *source_host,
    char *sink_host)
{
    const char *middle;
    const char *suffix;
    uint64_t source_bytes;
    uint64_t sink_bytes;
    if (route_name == 0 || source_host == 0 || sink_host == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    middle = strstr(route_name,"_to_");
    suffix = strstr(route_name,"_hidden");
    if (middle == 0 || suffix == 0 || middle >= suffix)
        return SPARK_STATUS_INVALID_ARGUMENT;
    source_bytes = (uint64_t)(middle - route_name);
    sink_bytes = (uint64_t)(suffix - (middle + 4));
    if (source_bytes == 0u || sink_bytes == 0u ||
        source_bytes >= SPARK_HIDDEN_TCP_CUDA_HOST_BYTES ||
        sink_bytes >= SPARK_HIDDEN_TCP_CUDA_HOST_BYTES)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    memcpy(source_host,route_name,(size_t)source_bytes);
    source_host[source_bytes] = '\0';
    memcpy(sink_host,middle + 4,(size_t)sink_bytes);
    sink_host[sink_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static int SparkHiddenTcpCudaCloseFd(int fd)
{
    if (fd >= 0)
        (void)close(fd);
    return -1;
}

static void SparkHiddenTcpCudaBuildDeadline(
    struct timespec *deadline,
    uint32_t timeout_ms)
{
    if (clock_gettime(CLOCK_REALTIME,deadline) != 0)
    {
        deadline->tv_sec = 0;
        deadline->tv_nsec = 0;
        return;
    }
    deadline->tv_sec += (time_t)(timeout_ms / 1000u);
    deadline->tv_nsec += (long)((timeout_ms % 1000u) * 1000000u);
    if (deadline->tv_nsec >= 1000000000L)
    {
        deadline->tv_sec += 1;
        deadline->tv_nsec -= 1000000000L;
    }
}

static void SparkHiddenTcpCudaWaitForOutgoingDrain(
    SparkHiddenTcpCudaState *state)
{
    struct timespec deadline;
    int wait_status;
    if (state->is_sender == 0u)
        return;
    SparkHiddenTcpCudaBuildDeadline(&deadline,
        SPARK_HIDDEN_TCP_CUDA_DESTROY_DRAIN_TIMEOUT_MS);
    (void)pthread_mutex_lock(&state->lock);
    while (state->outgoing_count != 0u && state->stop == 0u)
    {
        wait_status = pthread_cond_timedwait(&state->outgoing_cond,
            &state->lock,&deadline);
        if (wait_status == ETIMEDOUT)
            break;
    }
    (void)pthread_mutex_unlock(&state->lock);
}

static void SparkHiddenTcpCudaConfigureSocket(int fd)
{
    int value;
    if (fd < 0)
        return;
    value = 1;
    (void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&value,sizeof(value));
    (void)setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&value,sizeof(value));
}

static int SparkHiddenTcpCudaListen(uint32_t port)
{
    struct sockaddr_in address;
    int fd;
    int value;
    fd = socket(AF_INET,SOCK_STREAM,0);
    if (fd < 0)
        return -1;
    value = 1;
    setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&value,sizeof(value));
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((uint16_t)port);
    if (bind(fd,(struct sockaddr *)&address,sizeof(address)) < 0 ||
        listen(fd,16) < 0)
        return SparkHiddenTcpCudaCloseFd(fd);
    return fd;
}

static void SparkHiddenTcpCudaSignalEvent(SparkHiddenTcpCudaState *state)
{
    uint64_t one;
    one = 1u;
    if (state->event_fd >= 0)
        (void)write(state->event_fd,&one,sizeof(one));
}

static void SparkHiddenTcpCudaDrainEvent(SparkHiddenTcpCudaState *state)
{
    uint64_t counter;
    if (state->event_fd < 0)
        return;
    while (read(state->event_fd,&counter,sizeof(counter)) ==
        (ssize_t)sizeof(counter))
        continue;
}

static SparkStatus SparkHiddenTcpCudaReadFull(
    SparkHiddenTcpCudaState *state,
    void *buffer,
    uint64_t bytes)
{
    uint8_t *cursor;
    uint64_t offset;
    ssize_t got;
    cursor = (uint8_t *)buffer;
    offset = 0u;
    while (offset < bytes)
    {
        got = recv(state->socket_fd,cursor + offset,
            (size_t)(bytes - offset),0);
        if (got < 0 && errno == EINTR)
            continue;
        if (got <= 0)
            return SPARK_STATUS_IO_ERROR;
        offset += (uint64_t)got;
        if (state->stop != 0u)
            return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaWriteFull(
    SparkHiddenTcpCudaState *state,
    const void *buffer,
    uint64_t bytes)
{
    const uint8_t *cursor;
    uint64_t offset;
    ssize_t wrote;
    cursor = (const uint8_t *)buffer;
    offset = 0u;
    while (offset < bytes)
    {
        wrote = send(state->socket_fd,cursor + offset,
            (size_t)(bytes - offset),MSG_NOSIGNAL);
        if (wrote < 0 && errno == EINTR)
            continue;
        if (wrote <= 0)
            return SPARK_STATUS_IO_ERROR;
        offset += (uint64_t)wrote;
        if (state->stop != 0u)
            return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenTcpCudaInitializePendingHash(SparkHiddenTcpCudaState *state)
{
    uint32_t hash_index;
    for (hash_index = 0u; hash_index < state->pending_hash_slots; ++hash_index)
        state->pending_hash_heads[hash_index] = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
}

static void SparkHiddenTcpCudaResetPendingSlots(SparkHiddenTcpCudaState *state)
{
    uint32_t pending_index;
    SparkHiddenTcpCudaPendingPacket *pending;
    state->free_pending_head = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    for (pending_index = state->pending_depth; pending_index > 0u; --pending_index)
    {
        pending = &state->pending_packets[pending_index - 1u];
        pending->active = 0u;
        pending->next_pending_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
        pending->next_free_index = state->free_pending_head;
        state->free_pending_head = pending_index - 1u;
    }
}

static SparkHiddenTcpCudaPendingPacket *SparkHiddenTcpCudaReservePendingPacketLocked(
    SparkHiddenTcpCudaState *state)
{
    SparkHiddenTcpCudaPendingPacket *pending;
    if (state->free_pending_head == SPARK_HIDDEN_TCP_CUDA_NO_PENDING)
        return 0;
    pending = &state->pending_packets[state->free_pending_head];
    state->free_pending_head = pending->next_free_index;
    pending->next_pending_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    pending->active = 0u;
    return pending;
}

static void SparkHiddenTcpCudaReleasePendingPacketLocked(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTcpCudaPendingPacket *pending)
{
    pending->active = 0u;
    pending->next_free_index = state->free_pending_head;
    state->free_pending_head =
        (uint32_t)(pending - state->pending_packets);
    (void)pthread_cond_signal(&state->pending_free_cond);
}

static void SparkHiddenTcpCudaInsertPendingPacketLocked(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTcpCudaPendingPacket *pending)
{
    uint32_t hash_slot;
    uint32_t pending_index;
    pending_index = (uint32_t)(pending - state->pending_packets);
    hash_slot = SparkHiddenTcpCudaHeaderHash(&pending->header) %
        state->pending_hash_slots;
    pending->next_pending_index = state->pending_hash_heads[hash_slot];
    state->pending_hash_heads[hash_slot] = pending_index;
    pending->active = 1u;
}

static SparkHiddenTcpCudaPendingPacket *SparkHiddenTcpCudaDetachPendingPacketLocked(
    SparkHiddenTcpCudaState *state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaPendingPacket *pending;
    uint32_t pending_index;
    uint32_t previous_index;
    uint32_t hash_slot;
    hash_slot = SparkHiddenTcpCudaPacketHash(packet) % state->pending_hash_slots;
    previous_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    pending_index = state->pending_hash_heads[hash_slot];
    while (pending_index != SPARK_HIDDEN_TCP_CUDA_NO_PENDING)
    {
        pending = &state->pending_packets[pending_index];
        if (pending->active != 0u &&
            SparkHiddenTcpCudaHeaderMatchesPacket(&pending->header,packet) != 0u)
        {
            if (previous_index == SPARK_HIDDEN_TCP_CUDA_NO_PENDING)
                state->pending_hash_heads[hash_slot] =
                    pending->next_pending_index;
            else
                state->pending_packets[previous_index].next_pending_index =
                    pending->next_pending_index;
            pending->next_pending_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
            return pending;
        }
        previous_index = pending_index;
        pending_index = pending->next_pending_index;
    }
    return 0;
}

static SparkStatus SparkHiddenTcpCudaCopyPayloadToPacket(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTransportPacket *packet,
    const uint8_t *payload,
    uint64_t hidden_bytes,
    uint64_t sideband_bytes)
{
    uint64_t hidden_hash;
    uint64_t sideband_hash;
    if (state == 0 || packet == 0 || payload == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (cudaMemcpyAsync((void *)packet->hidden_bf16,payload,
            (size_t)hidden_bytes,cudaMemcpyHostToDevice,
            (cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (sideband_bytes != 0u &&
        cudaMemcpyAsync((void *)packet->sideband_payload,
            payload + hidden_bytes,(size_t)sideband_bytes,
            cudaMemcpyHostToDevice,
            (cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    SparkHiddenTcpCudaMaybeDumpPayload(state->local_rank,"rx",
        packet->sequence_id,packet->token_index,payload,hidden_bytes);
    if (state->debug_enabled != 0u)
    {
        hidden_hash = SparkHiddenTcpCudaPayloadHash(payload,hidden_bytes);
        sideband_hash = SparkHiddenTcpCudaPayloadHash(
            payload + hidden_bytes,sideband_bytes);
        fprintf(stderr,
            "hidden_tcp_deliver seq=%llu token=%llu active=%u sideband_kind=%u hidden_hash=%016llx sideband_hash=%016llx hidden_bytes=%llu sideband_bytes=%llu\n",
            (unsigned long long)packet->sequence_id,
            (unsigned long long)packet->token_index,
            packet->active_sequence_count,
            packet->sideband_kind,
            (unsigned long long)hidden_hash,
            (unsigned long long)sideband_hash,
            (unsigned long long)hidden_bytes,
            (unsigned long long)sideband_bytes);
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenTcpCudaSenderConnectBlocking(SparkHiddenTcpCudaState *state)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    char port_text[16];
    const char *host;
    int fd;
    while (state->stop == 0u && state->socket_fd < 0)
    {
        host = getenv("SPARKPIPE_PP13_TRANSPORT_HOST_OVERRIDE");
        if (host == 0 || host[0] == '\0')
            host = state->sink_host;
        memset(&hints,0,sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        snprintf(port_text,sizeof(port_text),"%u",
            state->port_base + (uint32_t)state->sink_rank);
        result = 0;
        if (getaddrinfo(host,port_text,&hints,&result) != 0 || result == 0)
        {
            (void)poll(0,0,SPARK_HIDDEN_TCP_CUDA_CONNECT_RETRY_MS);
            continue;
        }
        for (entry = result; entry != 0 && state->socket_fd < 0;
             entry = entry->ai_next)
        {
            fd = socket(entry->ai_family,entry->ai_socktype,
                entry->ai_protocol);
            if (fd < 0)
                continue;
            if (connect(fd,entry->ai_addr,entry->ai_addrlen) == 0)
            {
                SparkHiddenTcpCudaConfigureSocket(fd);
                state->socket_fd = fd;
                if (state->debug_enabled != 0u)
                    fprintf(stderr,"tcp_send_connected fd=%d sink=%s\n",
                        fd,host);
            }
            else
                (void)close(fd);
        }
        freeaddrinfo(result);
        if (state->socket_fd < 0)
            (void)poll(0,0,SPARK_HIDDEN_TCP_CUDA_CONNECT_RETRY_MS);
    }
}

static void *SparkHiddenTcpCudaSenderMain(void *argument)
{
    SparkHiddenTcpCudaState *state;
    SparkHiddenTcpCudaOutgoingSlot *slot;
    SparkStatus status;
    state = (SparkHiddenTcpCudaState *)argument;
    SparkHiddenTcpCudaSenderConnectBlocking(state);
    (void)pthread_mutex_lock(&state->lock);
    while (state->stop == 0u)
    {
        if (state->outgoing_count == 0u)
        {
            (void)pthread_cond_wait(&state->outgoing_cond,&state->lock);
            continue;
        }
        slot = &state->outgoing_slots[state->outgoing_head];
        (void)pthread_mutex_unlock(&state->lock);
        status = SPARK_STATUS_IO_ERROR;
        while (state->stop == 0u && status != SPARK_STATUS_OK)
        {
            SparkHiddenTcpCudaSenderConnectBlocking(state);
            if (state->socket_fd < 0)
                break;
            status = SparkHiddenTcpCudaWriteFull(state,slot->payload,
                slot->bytes);
            if (status != SPARK_STATUS_OK)
            {
                if (state->debug_enabled != 0u)
                    fprintf(stderr,"tcp_send_close fd=%d errno=%d\n",
                        state->socket_fd,errno);
                state->socket_fd = SparkHiddenTcpCudaCloseFd(state->socket_fd);
            }
        }
        (void)pthread_mutex_lock(&state->lock);
        if (status == SPARK_STATUS_OK)
        {
            while (SparkHiddenTransportCompletionQueueIsFull(
                    &state->completion_queue) != 0u &&
                state->stop == 0u)
                (void)pthread_cond_wait(&state->outgoing_cond,&state->lock);
            if (state->stop == 0u)
                (void)SparkHiddenTcpCudaPushCompletionLocked(
                    state,&slot->header,SPARK_STATUS_OK,
                    slot->bytes - sizeof(slot->header));
        }
        state->outgoing_head = (state->outgoing_head + 1u) %
            SPARK_HIDDEN_TCP_CUDA_OUTGOING_DEPTH;
        state->outgoing_count -= 1u;
        (void)pthread_cond_broadcast(&state->outgoing_cond);
        SparkHiddenTcpCudaSignalEvent(state);
    }
    (void)pthread_mutex_unlock(&state->lock);
    return 0;
}

static void *SparkHiddenTcpCudaReceiverMain(void *argument)
{
    SparkHiddenTcpCudaState *state;
    SparkHiddenTcpCudaPendingPacket *pending;
    SparkHiddenTcpCudaHeader header;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    int fd;
    state = (SparkHiddenTcpCudaState *)argument;
    while (state->stop == 0u)
    {
        if (state->socket_fd < 0)
        {
            fd = accept(state->listen_fd,0,0);
            if (fd < 0)
            {
                if (state->stop == 0u && (errno == EINTR || errno == ECONNABORTED))
                    continue;
                break;
            }
            SparkHiddenTcpCudaConfigureSocket(fd);
            state->socket_fd = fd;
            if (state->debug_enabled != 0u)
                fprintf(stderr,"tcp_recv_accept fd=%d\n",fd);
        }
        if (SparkHiddenTcpCudaReadFull(state,&header,sizeof(header)) !=
                SPARK_STATUS_OK ||
            header.magic != SPARK_HIDDEN_TCP_CUDA_MAGIC)
        {
            if (state->debug_enabled != 0u)
                fprintf(stderr,"tcp_recv_close fd=%d errno=%d\n",
                    state->socket_fd,errno);
            state->socket_fd = SparkHiddenTcpCudaCloseFd(state->socket_fd);
            continue;
        }
        hidden_bytes = (uint64_t)header.bytes_per_sequence *
            (uint64_t)header.active_sequence_count;
        sideband_bytes = (uint64_t)header.sideband_bytes_per_sequence *
            (uint64_t)header.active_sequence_count;
        if (hidden_bytes + sideband_bytes > state->slot_payload_bytes)
        {
            if (state->debug_enabled != 0u)
                fprintf(stderr,"tcp_recv_oversized fd=%d bytes=%llu\n",
                    state->socket_fd,
                    (unsigned long long)(hidden_bytes + sideband_bytes));
            state->socket_fd = SparkHiddenTcpCudaCloseFd(state->socket_fd);
            continue;
        }
        (void)pthread_mutex_lock(&state->lock);
        pending = SparkHiddenTcpCudaReservePendingPacketLocked(state);
        while (pending == 0 && state->stop == 0u)
        {
            (void)pthread_cond_wait(&state->pending_free_cond,&state->lock);
            pending = SparkHiddenTcpCudaReservePendingPacketLocked(state);
        }
        (void)pthread_mutex_unlock(&state->lock);
        if (pending == 0)
            break;
        if (SparkHiddenTcpCudaReadFull(state,pending->payload,
                hidden_bytes + sideband_bytes) != SPARK_STATUS_OK)
        {
            if (state->debug_enabled != 0u)
                fprintf(stderr,"tcp_recv_close fd=%d errno=%d\n",
                    state->socket_fd,errno);
            (void)pthread_mutex_lock(&state->lock);
            SparkHiddenTcpCudaReleasePendingPacketLocked(state,pending);
            (void)pthread_mutex_unlock(&state->lock);
            state->socket_fd = SparkHiddenTcpCudaCloseFd(state->socket_fd);
            continue;
        }
        pending->header = header;
        pending->hidden_bytes = hidden_bytes;
        pending->sideband_bytes = sideband_bytes;
        (void)pthread_mutex_lock(&state->lock);
        SparkHiddenTcpCudaInsertPendingPacketLocked(state,pending);
        (void)pthread_mutex_unlock(&state->lock);
        if (state->debug_enabled != 0u)
            fprintf(stderr,
                "tcp_frame_delivered seq=%llu pos=%llu active=%u sk=%u sb=%u\n",
                (unsigned long long)header.sequence_id,
                (unsigned long long)header.token_index,
                header.active_sequence_count,
                header.sideband_kind,
                header.sideband_bytes_per_sequence);
        SparkHiddenTcpCudaSignalEvent(state);
    }
    return 0;
}

static void SparkHiddenTcpCudaDestroyState(SparkHiddenTcpCudaState *state)
{
    uint32_t slot_index;
    if (state == 0)
        return;
    SparkHiddenTcpCudaWaitForOutgoingDrain(state);
    state->stop = 1u;
    if (state->listen_fd >= 0)
        (void)shutdown(state->listen_fd,SHUT_RDWR);
    if (state->socket_fd >= 0)
        (void)shutdown(state->socket_fd,SHUT_RDWR);
    (void)pthread_mutex_lock(&state->lock);
    (void)pthread_cond_broadcast(&state->outgoing_cond);
    (void)pthread_cond_broadcast(&state->pending_free_cond);
    (void)pthread_mutex_unlock(&state->lock);
    if (state->thread_running != 0u)
        (void)pthread_join(state->io_thread,0);
    state->listen_fd = SparkHiddenTcpCudaCloseFd(state->listen_fd);
    state->socket_fd = SparkHiddenTcpCudaCloseFd(state->socket_fd);
    state->event_fd = SparkHiddenTcpCudaCloseFd(state->event_fd);
    (void)pthread_cond_destroy(&state->outgoing_cond);
    (void)pthread_cond_destroy(&state->pending_free_cond);
    (void)pthread_mutex_destroy(&state->lock);
    for (slot_index = 0u;
         slot_index < SPARK_HIDDEN_TCP_CUDA_OUTGOING_DEPTH;
         ++slot_index)
        if (state->outgoing_slots[slot_index].payload != 0)
            (void)cudaFreeHost(state->outgoing_slots[slot_index].payload);
    if (state->pending_packets != 0)
        for (slot_index = 0u; slot_index < state->pending_depth; ++slot_index)
            if (state->pending_packets[slot_index].payload != 0)
                (void)cudaFreeHost(state->pending_packets[slot_index].payload);
    free(state->pending_packets);
    free(state->pending_hash_heads);
    free(state);
}

static SparkStatus SparkHiddenTcpCudaInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    SparkHiddenTcpCudaState *state;
    const char *rank_text;
    uint32_t slot_index;
    if (endpoint == 0 || transport_state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state = (SparkHiddenTcpCudaState *)calloc(1u,sizeof(*state));
    if (state == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    state->endpoint = *endpoint;
    state->listen_fd = -1;
    state->socket_fd = -1;
    state->event_fd = -1;
    state->debug_enabled =
        getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0 ? 1u : 0u;
    state->port_base = SparkHiddenTcpCudaParseUintEnv(
        "SPARKPIPE_PP13_HIDDEN_TRANSPORT_PORT_BASE",
        SparkHiddenTcpCudaParseUintEnv(
            "SPARKPIPE_PP13_TRANSPORT_PORT_BASE",
            SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_BASE) +
            SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_OFFSET);
    rank_text = getenv("SPARKPIPE_PP13_TRANSPORT_RANK");
    if (rank_text == 0 || rank_text[0] == '\0')
    {
        free(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->local_rank = (int32_t)SparkHiddenTcpCudaParseUintEnv(
        "SPARKPIPE_PP13_TRANSPORT_RANK",1000u);
    if (SparkHiddenTcpCudaParseRoute(endpoint->route_name,
            state->source_host,state->sink_host) != SPARK_STATUS_OK)
    {
        free(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->source_rank = SparkHiddenTcpCudaRankFromHost(state->source_host);
    state->sink_rank = SparkHiddenTcpCudaRankFromHost(state->sink_host);
    if (state->source_rank < 0 || state->sink_rank < 0)
    {
        free(state);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->is_sender = state->local_rank == state->source_rank ? 1u : 0u;
    if (state->is_sender == 0u && state->local_rank != state->sink_rank)
    {
        free(state);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->slot_payload_bytes = endpoint->max_packet_bytes +
        (uint64_t)sizeof(SparkHiddenTcpCudaHeader);
    state->pending_depth = SPARK_HIDDEN_TCP_CUDA_PENDING_DEPTH;
    state->pending_hash_slots =
        state->pending_depth * SPARK_HIDDEN_TCP_CUDA_PENDING_HASH_LOAD_FACTOR;
    state->event_fd = eventfd(0u,EFD_NONBLOCK | EFD_CLOEXEC);
    if (state->event_fd < 0 ||
        pthread_mutex_init(&state->lock,0) != 0 ||
        pthread_cond_init(&state->outgoing_cond,0) != 0 ||
        pthread_cond_init(&state->pending_free_cond,0) != 0)
    {
        state->event_fd = SparkHiddenTcpCudaCloseFd(state->event_fd);
        free(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (state->is_sender != 0u)
    {
        for (slot_index = 0u;
             slot_index < SPARK_HIDDEN_TCP_CUDA_OUTGOING_DEPTH;
             ++slot_index)
        {
            if (cudaHostAlloc(
                    (void **)&state->outgoing_slots[slot_index].payload,
                    (size_t)state->slot_payload_bytes,
                    cudaHostAllocDefault) != cudaSuccess)
                state->outgoing_slots[slot_index].payload = 0;
            if (state->outgoing_slots[slot_index].payload == 0)
            {
                SparkHiddenTcpCudaDestroyState(state);
                return SPARK_STATUS_INTERNAL_ERROR;
            }
        }
    }
    else
    {
        state->pending_packets = (SparkHiddenTcpCudaPendingPacket *)calloc(
            state->pending_depth,sizeof(*state->pending_packets));
        state->pending_hash_heads = (uint32_t *)calloc(
            state->pending_hash_slots,sizeof(*state->pending_hash_heads));
        if (state->pending_packets == 0 || state->pending_hash_heads == 0)
        {
            SparkHiddenTcpCudaDestroyState(state);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        for (slot_index = 0u; slot_index < state->pending_depth; ++slot_index)
        {
            if (cudaHostAlloc(
                    (void **)&state->pending_packets[slot_index].payload,
                    (size_t)state->slot_payload_bytes,
                    cudaHostAllocDefault) != cudaSuccess)
                state->pending_packets[slot_index].payload = 0;
            if (state->pending_packets[slot_index].payload == 0)
            {
                SparkHiddenTcpCudaDestroyState(state);
                return SPARK_STATUS_INTERNAL_ERROR;
            }
        }
        SparkHiddenTcpCudaInitializePendingHash(state);
        SparkHiddenTcpCudaResetPendingSlots(state);
        state->listen_fd = SparkHiddenTcpCudaListen(
            state->port_base + (uint32_t)state->sink_rank);
        if (state->listen_fd < 0)
        {
            SparkHiddenTcpCudaDestroyState(state);
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
    }
    if (pthread_create(&state->io_thread,0,
            state->is_sender != 0u ?
                SparkHiddenTcpCudaSenderMain :
                SparkHiddenTcpCudaReceiverMain,
            state) != 0)
    {
        SparkHiddenTcpCudaDestroyState(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->thread_running = 1u;
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenTcpCudaDestroy(void *transport_state)
{
    SparkHiddenTcpCudaDestroyState((SparkHiddenTcpCudaState *)transport_state);
}

static SparkStatus SparkHiddenTcpCudaSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaState *state;
    SparkHiddenTcpCudaOutgoingSlot *slot;
    SparkStatus status;
    cudaError_t cuda_error;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint32_t slot_index;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacket(&state->endpoint,packet);
    if (status != SPARK_STATUS_OK)
        return status;
    SparkHiddenTcpCudaDrainEvent(state);
    hidden_bytes = (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    sideband_bytes = (uint64_t)packet->sideband_bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    if (sizeof(SparkHiddenTcpCudaHeader) + hidden_bytes + sideband_bytes >
        state->slot_payload_bytes)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    cuda_error = cudaStreamSynchronize((cudaStream_t)packet->cuda_stream);
    if (cuda_error != cudaSuccess)
    {
        fprintf(
            stderr,
            "hidden_tcp_send_cuda_error stage=stream_sync error=%d name=%s message=%s seq=%llu token=%llu active=%u stream=%p hidden=%p hidden_bytes=%llu sideband=%p sideband_bytes=%llu\n",
            (int32_t)cuda_error,
            cudaGetErrorName(cuda_error),
            cudaGetErrorString(cuda_error),
            (unsigned long long)packet->sequence_id,
            (unsigned long long)packet->token_index,
            packet->active_sequence_count,
            packet->cuda_stream,
            (const void *)packet->hidden_bf16,
            (unsigned long long)hidden_bytes,
            (const void *)packet->sideband_payload,
            (unsigned long long)sideband_bytes);
        return SPARK_STATUS_IO_ERROR;
    }
    (void)pthread_mutex_lock(&state->lock);
    if (state->outgoing_count >= SPARK_HIDDEN_TCP_CUDA_OUTGOING_DEPTH)
    {
        (void)pthread_mutex_unlock(&state->lock);
        return SPARK_STATUS_BUSY;
    }
    slot_index = (state->outgoing_head + state->outgoing_count) %
        SPARK_HIDDEN_TCP_CUDA_OUTGOING_DEPTH;
    slot = &state->outgoing_slots[slot_index];
    memset(&slot->header,0,sizeof(slot->header));
    slot->header.magic = SPARK_HIDDEN_TCP_CUDA_MAGIC;
    slot->header.active_sequence_count = packet->active_sequence_count;
    slot->header.hidden_dimension = packet->hidden_dimension;
    slot->header.bytes_per_sequence = packet->bytes_per_sequence;
    slot->header.sequence_id = packet->sequence_id;
    slot->header.token_index = packet->token_index;
    slot->header.sideband_kind = packet->sideband_kind;
    slot->header.sideband_bytes_per_sequence =
        packet->sideband_bytes_per_sequence;
    memcpy(slot->payload,&slot->header,sizeof(slot->header));
    // Asynchronous, on the packet's own stream. The synchronous form ran on the
    // default stream, which blocks the host and implicitly serialises against
    // every other stream, and there were two of them per hop. Issuing both on
    // the packet stream lets them overlap each other and costs one
    // synchronisation instead of two. The staging buffers are pinned via
    // cudaHostAlloc, so this is genuinely asynchronous rather than silently
    // degrading to a blocking copy.
    cuda_error = cudaMemcpyAsync(
        slot->payload + sizeof(slot->header),
        packet->hidden_bf16,
        (size_t)hidden_bytes,
        cudaMemcpyDeviceToHost,
        (cudaStream_t)packet->cuda_stream);
    if (cuda_error != cudaSuccess)
    {
        fprintf(
            stderr,
            "hidden_tcp_send_cuda_error stage=hidden_d2h error=%d name=%s message=%s seq=%llu token=%llu active=%u stream=%p hidden=%p hidden_bytes=%llu sideband=%p sideband_bytes=%llu\n",
            (int32_t)cuda_error,
            cudaGetErrorName(cuda_error),
            cudaGetErrorString(cuda_error),
            (unsigned long long)packet->sequence_id,
            (unsigned long long)packet->token_index,
            packet->active_sequence_count,
            packet->cuda_stream,
            (const void *)packet->hidden_bf16,
            (unsigned long long)hidden_bytes,
            (const void *)packet->sideband_payload,
            (unsigned long long)sideband_bytes);
        (void)pthread_mutex_unlock(&state->lock);
        return SPARK_STATUS_IO_ERROR;
    }
    if (sideband_bytes != 0u)
    {
        cuda_error = cudaMemcpyAsync(
            slot->payload + sizeof(slot->header) + hidden_bytes,
            packet->sideband_payload,
            (size_t)sideband_bytes,
            cudaMemcpyDeviceToHost,
            (cudaStream_t)packet->cuda_stream);
        if (cuda_error != cudaSuccess)
        {
            fprintf(
                stderr,
                "hidden_tcp_send_cuda_error stage=sideband_d2h error=%d name=%s message=%s seq=%llu token=%llu active=%u stream=%p hidden=%p hidden_bytes=%llu sideband=%p sideband_bytes=%llu\n",
                (int32_t)cuda_error,
                cudaGetErrorName(cuda_error),
                cudaGetErrorString(cuda_error),
                (unsigned long long)packet->sequence_id,
                (unsigned long long)packet->token_index,
                packet->active_sequence_count,
                packet->cuda_stream,
                (const void *)packet->hidden_bf16,
                (unsigned long long)hidden_bytes,
                (const void *)packet->sideband_payload,
                (unsigned long long)sideband_bytes);
            (void)pthread_mutex_unlock(&state->lock);
            return SPARK_STATUS_IO_ERROR;
        }
    }
    // The copies above are asynchronous, and the next three lines hand this slot
    // to the sender thread. Retire them here or the sender transmits whatever
    // the staging buffer held before the DMA landed.
    cuda_error = cudaStreamSynchronize((cudaStream_t)packet->cuda_stream);
    if (cuda_error != cudaSuccess)
    {
        fprintf(
            stderr,
            "hidden_tcp_send_cuda_error stage=stage_retire error=%d name=%s message=%s seq=%llu token=%llu\n",
            (int32_t)cuda_error,
            cudaGetErrorName(cuda_error),
            cudaGetErrorString(cuda_error),
            (unsigned long long)packet->sequence_id,
            (unsigned long long)packet->token_index);
        (void)pthread_mutex_unlock(&state->lock);
        return SPARK_STATUS_IO_ERROR;
    }
    slot->bytes = sizeof(slot->header) + hidden_bytes + sideband_bytes;
    state->outgoing_count += 1u;
    (void)pthread_cond_signal(&state->outgoing_cond);
    (void)pthread_mutex_unlock(&state->lock);
    SparkHiddenTcpCudaMaybeDumpPayload(state->local_rank,"tx",
        packet->sequence_id,packet->token_index,
        slot->payload + sizeof(slot->header),hidden_bytes);
    if (state->debug_enabled != 0u)
        fprintf(stderr,
            "hidden_tcp_send_header seq=%llu token=%llu active=%u sideband_kind=%u sideband_bps=%u hidden_hash=%016llx sideband_hash=%016llx hidden_bytes=%llu sideband_bytes=%llu total=%llu\n",
            (unsigned long long)packet->sequence_id,
            (unsigned long long)packet->token_index,
            packet->active_sequence_count,
            packet->sideband_kind,
            packet->sideband_bytes_per_sequence,
            (unsigned long long)SparkHiddenTcpCudaPayloadHash(
                slot->payload + sizeof(slot->header),hidden_bytes),
            (unsigned long long)SparkHiddenTcpCudaPayloadHash(
                slot->payload + sizeof(slot->header) + hidden_bytes,
                sideband_bytes),
            (unsigned long long)hidden_bytes,
            (unsigned long long)sideband_bytes,
            (unsigned long long)(sizeof(slot->header) + hidden_bytes +
                sideband_bytes));
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaState *state;
    SparkHiddenTcpCudaPendingPacket *pending;
    SparkStatus status;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacket(&state->endpoint,packet);
    if (status != SPARK_STATUS_OK)
        return status;
    SparkHiddenTcpCudaDrainEvent(state);
    (void)pthread_mutex_lock(&state->lock);
    if (SparkHiddenTransportCompletionQueueIsFull(
            &state->completion_queue) != 0u)
    {
        (void)pthread_mutex_unlock(&state->lock);
        return SPARK_STATUS_BUSY;
    }
    pending = SparkHiddenTcpCudaDetachPendingPacketLocked(state,packet);
    (void)pthread_mutex_unlock(&state->lock);
    if (pending == 0)
        return SPARK_STATUS_BUSY;
    status = SparkHiddenTcpCudaCopyPayloadToPacket(state,packet,
        pending->payload,pending->hidden_bytes,pending->sideband_bytes);
    (void)pthread_mutex_lock(&state->lock);
    (void)SparkHiddenTcpCudaPushCompletionLocked(
        state,&pending->header,status,
        pending->hidden_bytes + pending->sideband_bytes);
    SparkHiddenTcpCudaReleasePendingPacketLocked(state,pending);
    (void)pthread_mutex_unlock(&state->lock);
    SparkHiddenTcpCudaSignalEvent(state);
    return status;
}

static SparkStatus SparkHiddenTcpCudaPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    SparkHiddenTcpCudaState *state;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0 || completion == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    SparkHiddenTcpCudaDrainEvent(state);
    (void)pthread_mutex_lock(&state->lock);
    (void)SparkHiddenTransportCompletionQueuePop(
        &state->completion_queue,completion);
    (void)pthread_cond_broadcast(&state->outgoing_cond);
    (void)pthread_mutex_unlock(&state->lock);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaAppendPollDescriptor(
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count,
    int fd,
    uint32_t events)
{
    SparkHiddenTransportPollDescriptor *descriptor;
    if (descriptor_count == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (fd < 0 || events == 0u)
        return SPARK_STATUS_OK;
    if (*descriptor_count >= descriptor_capacity || descriptors == 0)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    descriptor = &descriptors[*descriptor_count];
    memset(descriptor,0,sizeof(*descriptor));
    descriptor->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    descriptor->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_POLL_DESCRIPTOR_BYTES;
    descriptor->fd = fd;
    descriptor->events = events;
    *descriptor_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaGetPollDescriptors(
    void *transport_state,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out)
{
    SparkHiddenTcpCudaState *state;
    SparkStatus status;
    uint32_t descriptor_count;
    if (transport_state == 0 || descriptor_count_out == 0 ||
        (descriptors == 0 && descriptor_capacity != 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    state = (SparkHiddenTcpCudaState *)transport_state;
    descriptor_count = 0u;
    status = SparkHiddenTcpCudaAppendPollDescriptor(descriptors,
        descriptor_capacity,&descriptor_count,state->event_fd,
        SPARK_HIDDEN_TRANSPORT_POLL_READ);
    if (status != SPARK_STATUS_OK)
        return status;
    *descriptor_count_out = descriptor_count;
    return SPARK_STATUS_OK;
}

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface transport_interface;
    memset(&transport_interface,0,sizeof(transport_interface));
    transport_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS |
        SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS;
    transport_interface.initialize = SparkHiddenTcpCudaInitialize;
    transport_interface.destroy = SparkHiddenTcpCudaDestroy;
    transport_interface.post_receive = SparkHiddenTcpCudaPostReceive;
    transport_interface.send = SparkHiddenTcpCudaSend;
    transport_interface.poll = SparkHiddenTcpCudaPoll;
    transport_interface.get_poll_descriptors = SparkHiddenTcpCudaGetPollDescriptors;
    return &transport_interface;
}
