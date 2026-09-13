#include "sparkpipe/spark_hidden_transport_rdma_control.h"

#include "sparkpipe/spark_hidden_transport.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stddef.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>

uint64_t SparkHiddenTransportRdmaControlMonotonicNs(void)
{
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC,&value) != 0)
        return UINT64_MAX;
    return (uint64_t)value.tv_sec * 1000000000ull + (uint64_t)value.tv_nsec;
}

uint64_t SparkHiddenTransportRdmaControlDeadlineNs(uint32_t timeout_milli)
{
    uint64_t now;
    uint64_t interval;

    if (timeout_milli == 0u)
        return 0u;
    now = SparkHiddenTransportRdmaControlMonotonicNs();
    interval = (uint64_t)timeout_milli * 1000000ull;
    if (now == UINT64_MAX || now > UINT64_MAX - interval)
        return 0u;
    return now + interval;
}

SparkStatus SparkHiddenTransportRdmaControlSetNonblocking(int fd)
{
    int flags;

    if (fd < 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    flags = fcntl(fd,F_GETFL,0);
    if (flags < 0 || fcntl(fd,F_SETFL,flags | O_NONBLOCK) != 0)
        return SPARK_STATUS_IO_ERROR;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTransportRdmaControlWait(
    int fd,
    short events,
    uint64_t deadline_ns)
{
    struct pollfd descriptor;
    uint64_t now,remaining_ns,remaining_ms;
    int result,timeout;

    for (;;)
    {
        now = SparkHiddenTransportRdmaControlMonotonicNs();
        if (now == UINT64_MAX)
            return SPARK_STATUS_IO_ERROR;
        if (deadline_ns == 0u || now >= deadline_ns)
            return SPARK_STATUS_BUSY;
        remaining_ns = deadline_ns - now;
        remaining_ms = (remaining_ns + 999999ull) / 1000000ull;
        timeout = remaining_ms > (uint64_t)INT_MAX ? INT_MAX :
            (int)remaining_ms;
        memset(&descriptor,0,sizeof(descriptor));
        descriptor.fd = fd;
        descriptor.events = events;
        result = poll(&descriptor,1,timeout);
        if (result > 0)
        {
            if ((descriptor.revents & events) != 0)
                return SPARK_STATUS_OK;
            return SPARK_STATUS_IO_ERROR;
        }
        if (result == 0)
            return SPARK_STATUS_BUSY;
        if (errno != EINTR)
            return SPARK_STATUS_IO_ERROR;
    }
}

SparkStatus SparkHiddenTransportRdmaControlReadFullDeadline(
    int fd,
    void *buffer,
    uint64_t bytes,
    uint64_t deadline_ns)
{
    uint8_t *cursor;
    uint64_t done;

    if (fd < 0 || buffer == 0 || bytes == 0u || deadline_ns == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (uint8_t *)buffer;
    done = 0u;
    while (done < bytes)
    {
        ssize_t result = recv(fd,cursor + done,(size_t)(bytes - done),0);
        if (result > 0)
        {
            done += (uint64_t)result;
            continue;
        }
        if (result == 0)
            return SPARK_STATUS_IO_ERROR;
        if (errno == EINTR)
            continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK)
            return SPARK_STATUS_IO_ERROR;
        {
            SparkStatus status = SparkHiddenTransportRdmaControlWait(
                fd,POLLIN,deadline_ns);
            if (status != SPARK_STATUS_OK)
                return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportRdmaControlWriteFullDeadline(
    int fd,
    const void *buffer,
    uint64_t bytes,
    uint64_t deadline_ns)
{
    const uint8_t *cursor;
    uint64_t done;

    if (fd < 0 || buffer == 0 || bytes == 0u || deadline_ns == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (const uint8_t *)buffer;
    done = 0u;
    while (done < bytes)
    {
        ssize_t result = send(fd,cursor + done,(size_t)(bytes - done),
            MSG_NOSIGNAL);
        if (result > 0)
        {
            done += (uint64_t)result;
            continue;
        }
        if (result == 0)
            return SPARK_STATUS_IO_ERROR;
        if (errno == EINTR)
            continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK)
            return SPARK_STATUS_IO_ERROR;
        {
            SparkStatus status = SparkHiddenTransportRdmaControlWait(
                fd,POLLOUT,deadline_ns);
            if (status != SPARK_STATUS_OK)
                return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportRdmaControlFenceSession(int fd)
{
    if (fd < 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (shutdown(fd,SHUT_RDWR) == 0 || errno == ENOTCONN)
        return SPARK_STATUS_OK;
    return SPARK_STATUS_IO_ERROR;
}

static uint32_t SparkHiddenTransportRdmaControlTextIsTerminated(
    const char *text,
    uint32_t bytes)
{
    return text != 0 && bytes != 0u && memchr(text,'\0',bytes) != 0;
}

SparkStatus SparkHiddenTransportRdmaV4ValidatePeerIdentity(
    const SparkHiddenTransportRdmaV4Identity *local,
    const SparkHiddenTransportRdmaV4Identity *peer)
{
    if (local == 0 || peer == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (local->magic != SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MAGIC ||
        peer->magic != SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MAGIC ||
        local->protocol_version != SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_VERSION ||
        peer->protocol_version != SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_VERSION ||
        local->transport_abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        peer->transport_abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        local->descriptor_bytes != sizeof(*local) ||
        peer->descriptor_bytes != sizeof(*peer) ||
        local->reserved != 0u || peer->reserved != 0u ||
        local->sender_role > 1u || peer->sender_role > 1u ||
        local->peer_sender_role > 1u || peer->peer_sender_role > 1u ||
        peer->sender_role != local->peer_sender_role ||
        peer->peer_sender_role != local->sender_role ||
        peer->local_rank != local->peer_rank ||
        peer->peer_rank != local->local_rank ||
        peer->source_rank != local->source_rank ||
        peer->sink_rank != local->sink_rank ||
        peer->control_port != local->control_port ||
        peer->hidden_dimension != local->hidden_dimension ||
        peer->bytes_per_sequence != local->bytes_per_sequence ||
        peer->max_active_sequence_count != local->max_active_sequence_count ||
        peer->persistent_credit_count != local->persistent_credit_count ||
        peer->lane_count != local->lane_count ||
        peer->doorbell_max_bytes != local->doorbell_max_bytes ||
        peer->memory_mode != local->memory_mode ||
        peer->capability_flags != local->capability_flags ||
        peer->max_packet_bytes != local->max_packet_bytes ||
        peer->route_identifier != local->route_identifier ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            local->transport_module_id,sizeof(local->transport_module_id)) == 0u ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            peer->transport_module_id,sizeof(peer->transport_module_id)) == 0u ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            local->route_name,sizeof(local->route_name)) == 0u ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            peer->route_name,sizeof(peer->route_name)) == 0u ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            local->source_host,sizeof(local->source_host)) == 0u ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            peer->source_host,sizeof(peer->source_host)) == 0u ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            local->sink_host,sizeof(local->sink_host)) == 0u ||
        SparkHiddenTransportRdmaControlTextIsTerminated(
            peer->sink_host,sizeof(peer->sink_host)) == 0u ||
        strcmp(peer->transport_module_id,local->transport_module_id) != 0 ||
        strcmp(peer->route_name,local->route_name) != 0 ||
        strcmp(peer->source_host,local->source_host) != 0 ||
        strcmp(peer->sink_host,local->sink_host) != 0)
        return SPARK_STATUS_VALIDATION_FAILED;
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportRdmaV4ExchangeCompatibilityHello(
    int fd,
    uint64_t deadline_ns,
    const SparkHiddenTransportRdmaV4Identity *local_identity)
{
    SparkHiddenTransportRdmaV4Identity peer_identity;
    SparkStatus status;

    if (local_identity == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportRdmaControlSetNonblocking(fd);
    if (status == SPARK_STATUS_OK)
        status = SparkHiddenTransportRdmaControlWriteFullDeadline(fd,
            local_identity,sizeof(*local_identity),deadline_ns);
    if (status == SPARK_STATUS_OK)
        status = SparkHiddenTransportRdmaControlReadFullDeadline(fd,
            &peer_identity,sizeof(peer_identity),deadline_ns);
    if (status == SPARK_STATUS_OK)
        status = SparkHiddenTransportRdmaV4ValidatePeerIdentity(
            local_identity,&peer_identity);
    return status;
}
