#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_tp_collective.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SPARK_TP_COLLECTIVE_HANDSHAKE_MAGIC 0x53545043u
#define SPARK_TP_COLLECTIVE_OPERATION_MAGIC 0x5354504fu
#define SPARK_TP_COLLECTIVE_OPERATION_SUM_F32 1u
/* BF16 payload with F32 accumulate per exchange step (audit NET-011): decode
   all-reduce tensors are BF16, so staging F32 doubled both the host-staging
   copies and the wire bytes. The wire kind is negotiated in the operation
   header, so a rank mixing kinds fails validation instead of silently
   decoding the peer's bytes with the wrong element width. */
#define SPARK_TP_COLLECTIVE_OPERATION_SUM_BF16 2u
#define SPARK_TP_COLLECTIVE_CONNECT_RETRY_NANOSECONDS 2000000L
#define SPARK_TP_COLLECTIVE_IO_CHUNK_BYTES ((uint64_t)INT_MAX)

typedef struct SparkTpCollectiveHandshakeWire
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t collective_identifier_high;
    uint32_t collective_identifier_low;
} SparkTpCollectiveHandshakeWire;

typedef struct SparkTpCollectiveOperationWire
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t operation_kind;
    uint32_t step_index;
    uint32_t tp_degree;
    uint32_t sender_rank;
    uint32_t collective_identifier_high;
    uint32_t collective_identifier_low;
    uint32_t operation_sequence_high;
    uint32_t operation_sequence_low;
    uint32_t element_count_high;
    uint32_t element_count_low;
} SparkTpCollectiveOperationWire;

static uint64_t SparkTpCollectiveNowMilli(void)
{
    struct timespec current_time;

    if (clock_gettime(CLOCK_MONOTONIC, &current_time) != 0)
    {
        return UINT64_MAX;
    }

    return ((uint64_t)current_time.tv_sec * 1000u) +
        ((uint64_t)current_time.tv_nsec / 1000000u);
}

static uint64_t SparkTpCollectiveBuildDeadline(uint32_t timeout_milli)
{
    uint64_t current_time_milli;

    current_time_milli = SparkTpCollectiveNowMilli();
    if (UINT64_MAX - current_time_milli < (uint64_t)timeout_milli)
    {
        return UINT64_MAX;
    }

    return current_time_milli + (uint64_t)timeout_milli;
}

static int SparkTpCollectiveDeadlinePollTimeout(uint64_t deadline_milli)
{
    uint64_t current_time_milli;
    uint64_t remaining_milli;

    current_time_milli = SparkTpCollectiveNowMilli();
    if (current_time_milli >= deadline_milli)
    {
        return 0;
    }

    remaining_milli = deadline_milli - current_time_milli;
    if (remaining_milli > (uint64_t)INT_MAX)
    {
        return INT_MAX;
    }

    return (int)remaining_milli;
}

static SparkStatus SparkTpCollectivePollSocket(
    int32_t socket_descriptor,
    short requested_events,
    uint64_t deadline_milli,
    short *returned_events)
{
    struct pollfd socket_poll;
    int poll_result;
    int poll_timeout_milli;

    if (socket_descriptor < 0 || returned_events == NULL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (;;)
    {
        poll_timeout_milli = SparkTpCollectiveDeadlinePollTimeout(deadline_milli);
        if (poll_timeout_milli <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        memset(&socket_poll, 0, sizeof(socket_poll));
        socket_poll.fd = socket_descriptor;
        socket_poll.events = requested_events;
        poll_result = poll(&socket_poll, 1u, poll_timeout_milli);
        if (poll_result > 0)
        {
            *returned_events = socket_poll.revents;
            return SPARK_STATUS_OK;
        }

        if (poll_result == 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        if (errno != EINTR)
        {
            return SPARK_STATUS_IO_ERROR;
        }
    }
}

static SparkStatus SparkTpCollectiveSetNonblocking(int32_t socket_descriptor)
{
    int socket_flags;

    socket_flags = fcntl(socket_descriptor, F_GETFL, 0);
    if (socket_flags < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    if (fcntl(socket_descriptor, F_SETFL, socket_flags | O_NONBLOCK) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveConfigureDataSocket(int32_t socket_descriptor)
{
    int no_delay_enabled;
    SparkStatus status;

    no_delay_enabled = 1;
    if (setsockopt(
            socket_descriptor,
            IPPROTO_TCP,
            TCP_NODELAY,
            &no_delay_enabled,
            sizeof(no_delay_enabled)) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

#ifdef SO_NOSIGPIPE
    if (setsockopt(
            socket_descriptor,
            SOL_SOCKET,
            SO_NOSIGPIPE,
            &no_delay_enabled,
            sizeof(no_delay_enabled)) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
#endif

    status = SparkTpCollectiveSetNonblocking(socket_descriptor);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveSendAllUntil(
    int32_t socket_descriptor,
    const void *data,
    uint64_t data_bytes,
    uint64_t deadline_milli)
{
    const uint8_t *source_bytes;
    uint64_t sent_bytes;

    if (socket_descriptor < 0 || (data == NULL && data_bytes != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    source_bytes = (const uint8_t *)data;
    sent_bytes = 0u;
    while (sent_bytes < data_bytes)
    {
        uint64_t remaining_bytes;
        size_t transfer_bytes;
        ssize_t send_result;
        short returned_events;
        int send_flags;

        remaining_bytes = data_bytes - sent_bytes;
        transfer_bytes = (size_t)((remaining_bytes > SPARK_TP_COLLECTIVE_IO_CHUNK_BYTES) ?
            SPARK_TP_COLLECTIVE_IO_CHUNK_BYTES : remaining_bytes);
#ifdef MSG_NOSIGNAL
        send_flags = MSG_NOSIGNAL;
#else
        send_flags = 0;
#endif
        send_result = send(
            socket_descriptor,
            source_bytes + (size_t)sent_bytes,
            transfer_bytes,
            send_flags);
        if (send_result > 0)
        {
            sent_bytes += (uint64_t)send_result;
            continue;
        }

        if (send_result == 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        if (errno == EINTR)
        {
            continue;
        }

        if (errno != EAGAIN && errno != EWOULDBLOCK)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        if (SparkTpCollectivePollSocket(
                socket_descriptor,
                POLLOUT,
                deadline_milli,
                &returned_events) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        if ((returned_events & POLLOUT) == 0 ||
            (returned_events & (POLLERR | POLLHUP | POLLNVAL)) != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
    }

    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveReceiveAllUntil(
    int32_t socket_descriptor,
    void *data,
    uint64_t data_bytes,
    uint64_t deadline_milli)
{
    uint8_t *destination_bytes;
    uint64_t received_bytes;

    if (socket_descriptor < 0 || (data == NULL && data_bytes != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    destination_bytes = (uint8_t *)data;
    received_bytes = 0u;
    while (received_bytes < data_bytes)
    {
        uint64_t remaining_bytes;
        size_t transfer_bytes;
        ssize_t receive_result;
        short returned_events;

        remaining_bytes = data_bytes - received_bytes;
        transfer_bytes = (size_t)((remaining_bytes > SPARK_TP_COLLECTIVE_IO_CHUNK_BYTES) ?
            SPARK_TP_COLLECTIVE_IO_CHUNK_BYTES : remaining_bytes);
        receive_result = recv(
            socket_descriptor,
            destination_bytes + (size_t)received_bytes,
            transfer_bytes,
            0);
        if (receive_result > 0)
        {
            received_bytes += (uint64_t)receive_result;
            continue;
        }

        if (receive_result == 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        if (errno == EINTR)
        {
            continue;
        }

        if (errno != EAGAIN && errno != EWOULDBLOCK)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        if (SparkTpCollectivePollSocket(
                socket_descriptor,
                POLLIN,
                deadline_milli,
                &returned_events) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        if ((returned_events & POLLIN) == 0 ||
            (returned_events & (POLLERR | POLLNVAL)) != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
    }

    return SPARK_STATUS_OK;
}

static void SparkTpCollectiveReset(SparkTpCollective *collective)
{
    uint32_t step_index;

    memset(collective, 0, sizeof(*collective));
    collective->listen_socket = -1;
    for (step_index = 0u; step_index < SPARK_TP_COLLECTIVE_MAX_STEPS; ++step_index)
    {
        collective->step_sockets[step_index] = -1;
    }
}

static void SparkTpCollectiveCloseAll(SparkTpCollective *collective)
{
    uint32_t step_index;

    if (collective->listen_socket >= 0)
    {
        close(collective->listen_socket);
    }
    collective->listen_socket = -1;

    for (step_index = 0u; step_index < SPARK_TP_COLLECTIVE_MAX_STEPS; ++step_index)
    {
        if (collective->step_sockets[step_index] >= 0)
        {
            close(collective->step_sockets[step_index]);
        }
        collective->step_sockets[step_index] = -1;
    }
}

static SparkStatus SparkTpCollectiveFail(
    SparkTpCollective *collective,
    SparkStatus failure_status)
{
    collective->failed = 1u;
    SparkTpCollectiveCloseAll(collective);
    return failure_status;
}

static int SparkTpCollectiveDegreeIsSupported(uint32_t tp_degree)
{
    return tp_degree == 1u ||
        tp_degree == 2u ||
        tp_degree == 4u ||
        tp_degree == 8u ||
        tp_degree == 16u;
}

static uint32_t SparkTpCollectiveStepCount(uint32_t tp_degree)
{
    uint32_t step_count;

    step_count = 0u;
    while ((tp_degree >> (step_count + 1u)) != 0u)
    {
        step_count += 1u;
    }

    return step_count;
}

static int SparkTpCollectiveTextIsTerminated(
    const char *text,
    size_t text_capacity)
{
    return text != NULL && memchr(text, '\0', text_capacity) != NULL;
}

static SparkStatus SparkTpCollectiveValidateConfig(
    const SparkTpCollectiveConfig *config)
{
    SparkTpCollectivePeer empty_peer;
    uint32_t step_count;
    uint32_t step_index;

    if (config == NULL || config->abi_version != SPARK_TP_COLLECTIVE_ABI_VERSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (!SparkTpCollectiveDegreeIsSupported(config->tp_degree) ||
        config->tp_rank >= config->tp_degree ||
        config->reserved0 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    step_count = SparkTpCollectiveStepCount(config->tp_degree);
    if (config->tp_degree == 1u)
    {
        if (config->listen_port != 0u ||
            config->connect_timeout_milli != 0u ||
            config->operation_timeout_milli != 0u ||
            config->collective_identifier != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else
    {
        if (config->listen_port == 0u ||
            config->connect_timeout_milli == 0u ||
            config->operation_timeout_milli == 0u ||
            config->collective_identifier == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        for (step_index = 0u; step_index < step_count; ++step_index)
        {
            struct in_addr peer_address;
            const SparkTpCollectivePeer *peer;

            peer = &config->peers[step_index];
            if (!SparkTpCollectiveTextIsTerminated(
                    peer->host_name,
                    sizeof(peer->host_name)) ||
                peer->host_name[0] == '\0' ||
                peer->port == 0u ||
                peer->reserved0 != 0u ||
                peer->reserved1 != 0u ||
                inet_pton(AF_INET, peer->host_name, &peer_address) != 1)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    memset(&empty_peer, 0, sizeof(empty_peer));
    for (step_index = step_count;
         step_index < SPARK_TP_COLLECTIVE_MAX_STEPS;
         ++step_index)
    {
        if (memcmp(
                &config->peers[step_index],
                &empty_peer,
                sizeof(empty_peer)) != 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveListen(
    SparkTpCollective *collective,
    uint16_t listen_port)
{
    struct sockaddr_in listen_address;
    int reuse_address_enabled;
    SparkStatus status;

    collective->listen_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (collective->listen_socket < 0)
    {
        fprintf(stderr, "sparkpipe_tp: listen socket() errno=%d (%s)\n", errno, strerror(errno));
        return SPARK_STATUS_IO_ERROR;
    }

    reuse_address_enabled = 1;
    if (setsockopt(
            collective->listen_socket,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse_address_enabled,
            sizeof(reuse_address_enabled)) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    status = SparkTpCollectiveSetNonblocking(collective->listen_socket);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&listen_address, 0, sizeof(listen_address));
    listen_address.sin_family = AF_INET;
    listen_address.sin_addr.s_addr = htonl(INADDR_ANY);
    listen_address.sin_port = htons(listen_port);
    if (bind(
            collective->listen_socket,
            (struct sockaddr *)&listen_address,
            sizeof(listen_address)) != 0)
    {
        fprintf(stderr, "sparkpipe_tp: listen bind(%u) errno=%d (%s)\n",
            (uint32_t)listen_port, errno, strerror(errno));
        return SPARK_STATUS_IO_ERROR;
    }

    if (listen(collective->listen_socket, SPARK_TP_COLLECTIVE_MAX_STEPS) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    return SPARK_STATUS_OK;
}

static void SparkTpCollectiveEncodeIdentifier(
    uint64_t identifier,
    uint32_t *high_word,
    uint32_t *low_word)
{
    *high_word = htonl((uint32_t)(identifier >> 32u));
    *low_word = htonl((uint32_t)identifier);
}

static uint64_t SparkTpCollectiveDecodeIdentifier(
    uint32_t high_word,
    uint32_t low_word)
{
    return ((uint64_t)ntohl(high_word) << 32u) | (uint64_t)ntohl(low_word);
}

static void SparkTpCollectiveBuildHandshake(
    const SparkTpCollective *collective,
    SparkTpCollectiveHandshakeWire *handshake)
{
    memset(handshake, 0, sizeof(*handshake));
    handshake->magic = htonl(SPARK_TP_COLLECTIVE_HANDSHAKE_MAGIC);
    handshake->abi_version = htonl(SPARK_TP_COLLECTIVE_ABI_VERSION);
    handshake->tp_degree = htonl(collective->tp_degree);
    handshake->tp_rank = htonl(collective->tp_rank);
    SparkTpCollectiveEncodeIdentifier(
        collective->collective_identifier,
        &handshake->collective_identifier_high,
        &handshake->collective_identifier_low);
}

static SparkStatus SparkTpCollectiveValidateHandshake(
    const SparkTpCollective *collective,
    const SparkTpCollectiveHandshakeWire *handshake,
    uint32_t expected_rank,
    int *belongs_to_collective)
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint64_t collective_identifier;

    magic = ntohl(handshake->magic);
    abi_version = ntohl(handshake->abi_version);
    tp_degree = ntohl(handshake->tp_degree);
    tp_rank = ntohl(handshake->tp_rank);
    collective_identifier = SparkTpCollectiveDecodeIdentifier(
        handshake->collective_identifier_high,
        handshake->collective_identifier_low);

    *belongs_to_collective =
        magic == SPARK_TP_COLLECTIVE_HANDSHAKE_MAGIC &&
        collective_identifier == collective->collective_identifier;

    if (!*belongs_to_collective)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    if (abi_version != SPARK_TP_COLLECTIVE_ABI_VERSION ||
        tp_degree != collective->tp_degree ||
        tp_rank != expected_rank)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }

    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveConnectSocketUntil(
    const SparkTpCollectivePeer *peer,
    uint64_t deadline_milli,
    int32_t *socket_out)
{
    struct sockaddr_in peer_address;

    memset(&peer_address, 0, sizeof(peer_address));
    peer_address.sin_family = AF_INET;
    peer_address.sin_port = htons(peer->port);
    if (inet_pton(AF_INET, peer->host_name, &peer_address.sin_addr) != 1)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (;;)
    {
        int32_t connect_socket;
        int connect_result;
        int socket_error;
        socklen_t socket_error_bytes;
        short returned_events;
        SparkStatus status;

        if (SparkTpCollectiveDeadlinePollTimeout(deadline_milli) <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        connect_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (connect_socket < 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        status = SparkTpCollectiveConfigureDataSocket(connect_socket);
        if (status != SPARK_STATUS_OK)
        {
            close(connect_socket);
            return status;
        }

        connect_result = connect(
            connect_socket,
            (struct sockaddr *)&peer_address,
            sizeof(peer_address));
        if (connect_result == 0)
        {
            *socket_out = connect_socket;
            return SPARK_STATUS_OK;
        }

        if (errno == EINPROGRESS || errno == EALREADY || errno == EWOULDBLOCK)
        {
            status = SparkTpCollectivePollSocket(
                connect_socket,
                POLLOUT,
                deadline_milli,
                &returned_events);
            if (status == SPARK_STATUS_OK &&
                (returned_events & (POLLOUT | POLLERR | POLLHUP)) != 0)
            {
                socket_error = 0;
                socket_error_bytes = sizeof(socket_error);
                if (getsockopt(
                        connect_socket,
                        SOL_SOCKET,
                        SO_ERROR,
                        &socket_error,
                        &socket_error_bytes) != 0)
                {
                    socket_error = errno;
                }

                if (socket_error == 0)
                {
                    *socket_out = connect_socket;
                    return SPARK_STATUS_OK;
                }
            }
        }

        close(connect_socket);
        if (SparkTpCollectiveDeadlinePollTimeout(deadline_milli) <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        {
            struct timespec retry_delay;

            retry_delay.tv_sec = 0;
            retry_delay.tv_nsec = SPARK_TP_COLLECTIVE_CONNECT_RETRY_NANOSECONDS;
            while (nanosleep(&retry_delay, &retry_delay) != 0 && errno == EINTR)
            {
            }
        }
    }
}

static SparkStatus SparkTpCollectiveConnectPeer(
    SparkTpCollective *collective,
    const SparkTpCollectivePeer *peer,
    uint32_t expected_rank,
    uint64_t deadline_milli,
    int32_t *socket_out)
{
    for (;;)
    {
        SparkTpCollectiveHandshakeWire local_handshake;
        SparkTpCollectiveHandshakeWire remote_handshake;
        int32_t connect_socket;
        int belongs_to_collective;
        SparkStatus status;

        connect_socket = -1;
        status = SparkTpCollectiveConnectSocketUntil(
            peer,
            deadline_milli,
            &connect_socket);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }

        SparkTpCollectiveBuildHandshake(collective, &local_handshake);
        status = SparkTpCollectiveSendAllUntil(
            connect_socket,
            &local_handshake,
            sizeof(local_handshake),
            deadline_milli);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkTpCollectiveReceiveAllUntil(
                connect_socket,
                &remote_handshake,
                sizeof(remote_handshake),
                deadline_milli);
        }

        belongs_to_collective = 0;
        if (status == SPARK_STATUS_OK)
        {
            status = SparkTpCollectiveValidateHandshake(
                collective,
                &remote_handshake,
                expected_rank,
                &belongs_to_collective);
        }

        if (status == SPARK_STATUS_OK)
        {
            *socket_out = connect_socket;
            return SPARK_STATUS_OK;
        }

        close(connect_socket);
        if (status == SPARK_STATUS_VALIDATION_FAILED)
        {
            return status;
        }

        if (SparkTpCollectiveDeadlinePollTimeout(deadline_milli) <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
    }
}

static SparkStatus SparkTpCollectiveAcceptOne(
    SparkTpCollective *collective,
    uint64_t deadline_milli,
    int32_t *socket_out,
    uint32_t *remote_rank_out)
{
    for (;;)
    {
        SparkTpCollectiveHandshakeWire local_handshake;
        SparkTpCollectiveHandshakeWire remote_handshake;
        int32_t accepted_socket;
        uint32_t remote_rank;
        int belongs_to_collective;
        short returned_events;
        SparkStatus status;

        status = SparkTpCollectivePollSocket(
            collective->listen_socket,
            POLLIN,
            deadline_milli,
            &returned_events);
        if (status != SPARK_STATUS_OK ||
            (returned_events & POLLIN) == 0 ||
            (returned_events & (POLLERR | POLLNVAL)) != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }

        accepted_socket = accept(collective->listen_socket, NULL, NULL);
        if (accepted_socket < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            return SPARK_STATUS_IO_ERROR;
        }

        status = SparkTpCollectiveConfigureDataSocket(accepted_socket);
        if (status != SPARK_STATUS_OK)
        {
            close(accepted_socket);
            return status;
        }

        status = SparkTpCollectiveReceiveAllUntil(
            accepted_socket,
            &remote_handshake,
            sizeof(remote_handshake),
            deadline_milli);
        if (status != SPARK_STATUS_OK)
        {
            close(accepted_socket);
            if (SparkTpCollectiveDeadlinePollTimeout(deadline_milli) <= 0)
            {
                return SPARK_STATUS_IO_ERROR;
            }
            continue;
        }

        remote_rank = ntohl(remote_handshake.tp_rank);
        belongs_to_collective = 0;
        status = SparkTpCollectiveValidateHandshake(
            collective,
            &remote_handshake,
            remote_rank,
            &belongs_to_collective);
        if (status == SPARK_STATUS_NOT_FOUND || !belongs_to_collective)
        {
            close(accepted_socket);
            continue;
        }

        if (status != SPARK_STATUS_OK || remote_rank >= collective->tp_degree)
        {
            close(accepted_socket);
            return SPARK_STATUS_VALIDATION_FAILED;
        }

        SparkTpCollectiveBuildHandshake(collective, &local_handshake);
        status = SparkTpCollectiveSendAllUntil(
            accepted_socket,
            &local_handshake,
            sizeof(local_handshake),
            deadline_milli);
        if (status != SPARK_STATUS_OK)
        {
            close(accepted_socket);
            return status;
        }

        *socket_out = accepted_socket;
        *remote_rank_out = remote_rank;
        return SPARK_STATUS_OK;
    }
}

static SparkStatus SparkTpCollectiveAcceptPeers(
    SparkTpCollective *collective,
    uint32_t expected_count,
    uint64_t deadline_milli)
{
    uint32_t accepted_count;

    accepted_count = 0u;
    while (accepted_count < expected_count)
    {
        int32_t accepted_socket;
        uint32_t remote_rank;
        uint32_t rank_difference;
        uint32_t step_index;
        SparkStatus status;

        accepted_socket = -1;
        remote_rank = 0u;
        status = SparkTpCollectiveAcceptOne(
            collective,
            deadline_milli,
            &accepted_socket,
            &remote_rank);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }

        rank_difference = remote_rank ^ collective->tp_rank;
        if (remote_rank <= collective->tp_rank ||
            rank_difference == 0u ||
            (rank_difference & (rank_difference - 1u)) != 0u)
        {
            close(accepted_socket);
            return SPARK_STATUS_VALIDATION_FAILED;
        }

        step_index = 0u;
        while ((rank_difference >> (step_index + 1u)) != 0u)
        {
            step_index += 1u;
        }

        if (step_index >= collective->step_count ||
            collective->step_sockets[step_index] >= 0)
        {
            close(accepted_socket);
            return SPARK_STATUS_VALIDATION_FAILED;
        }

        collective->step_sockets[step_index] = accepted_socket;
        accepted_count += 1u;
    }

    return SPARK_STATUS_OK;
}

static void SparkTpCollectiveBuildOperationHeader(
    const SparkTpCollective *collective,
    uint32_t step_index,
    uint64_t operation_sequence,
    uint64_t element_count,
    uint32_t operation_kind,
    SparkTpCollectiveOperationWire *operation_header)
{
    memset(operation_header, 0, sizeof(*operation_header));
    operation_header->magic = htonl(SPARK_TP_COLLECTIVE_OPERATION_MAGIC);
    operation_header->abi_version = htonl(SPARK_TP_COLLECTIVE_ABI_VERSION);
    operation_header->operation_kind = htonl(operation_kind);
    operation_header->step_index = htonl(step_index);
    operation_header->tp_degree = htonl(collective->tp_degree);
    operation_header->sender_rank = htonl(collective->tp_rank);
    SparkTpCollectiveEncodeIdentifier(
        collective->collective_identifier,
        &operation_header->collective_identifier_high,
        &operation_header->collective_identifier_low);
    SparkTpCollectiveEncodeIdentifier(
        operation_sequence,
        &operation_header->operation_sequence_high,
        &operation_header->operation_sequence_low);
    SparkTpCollectiveEncodeIdentifier(
        element_count,
        &operation_header->element_count_high,
        &operation_header->element_count_low);
}

static SparkStatus SparkTpCollectiveValidateOperationHeader(
    const SparkTpCollective *collective,
    const SparkTpCollectiveOperationWire *operation_header,
    uint32_t expected_step_index,
    uint32_t expected_sender_rank,
    uint64_t expected_operation_sequence,
    uint64_t expected_element_count,
    uint32_t expected_operation_kind)
{
    if (ntohl(operation_header->magic) != SPARK_TP_COLLECTIVE_OPERATION_MAGIC ||
        ntohl(operation_header->abi_version) != SPARK_TP_COLLECTIVE_ABI_VERSION ||
        ntohl(operation_header->operation_kind) != expected_operation_kind ||
        ntohl(operation_header->step_index) != expected_step_index ||
        ntohl(operation_header->tp_degree) != collective->tp_degree ||
        ntohl(operation_header->sender_rank) != expected_sender_rank ||
        SparkTpCollectiveDecodeIdentifier(
            operation_header->collective_identifier_high,
            operation_header->collective_identifier_low) != collective->collective_identifier ||
        SparkTpCollectiveDecodeIdentifier(
            operation_header->operation_sequence_high,
            operation_header->operation_sequence_low) != expected_operation_sequence ||
        SparkTpCollectiveDecodeIdentifier(
            operation_header->element_count_high,
            operation_header->element_count_low) != expected_element_count)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }

    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveExchangeOperationHeaders(
    const SparkTpCollective *collective,
    int32_t step_socket,
    uint32_t step_index,
    uint32_t partner_rank,
    uint64_t operation_sequence,
    uint64_t element_count,
    uint32_t operation_kind,
    uint64_t deadline_milli)
{
    SparkTpCollectiveOperationWire local_header;
    SparkTpCollectiveOperationWire remote_header;
    SparkStatus receive_status;
    SparkStatus send_status;

    SparkTpCollectiveBuildOperationHeader(
        collective,
        step_index,
        operation_sequence,
        element_count,
        operation_kind,
        &local_header);

    receive_status = SPARK_STATUS_OK;
    send_status = SPARK_STATUS_OK;
    if (collective->tp_rank < partner_rank)
    {
        send_status = SparkTpCollectiveSendAllUntil(
            step_socket,
            &local_header,
            sizeof(local_header),
            deadline_milli);
        if (send_status == SPARK_STATUS_OK)
        {
            receive_status = SparkTpCollectiveReceiveAllUntil(
                step_socket,
                &remote_header,
                sizeof(remote_header),
                deadline_milli);
        }
    }
    else
    {
        receive_status = SparkTpCollectiveReceiveAllUntil(
            step_socket,
            &remote_header,
            sizeof(remote_header),
            deadline_milli);
        if (receive_status == SPARK_STATUS_OK)
        {
            send_status = SparkTpCollectiveSendAllUntil(
                step_socket,
                &local_header,
                sizeof(local_header),
                deadline_milli);
        }
    }

    if (receive_status != SPARK_STATUS_OK)
    {
        return receive_status;
    }

    if (send_status != SPARK_STATUS_OK)
    {
        return send_status;
    }

    return SparkTpCollectiveValidateOperationHeader(
        collective,
        &remote_header,
        step_index,
        partner_rank,
        operation_sequence,
        element_count,
        operation_kind);
}

SparkStatus SparkTpCollectiveCreate(
    const SparkTpCollectiveConfig *config,
    SparkTpCollective *collective_out)
{
    uint32_t step_index;
    uint32_t accept_count;
    uint64_t deadline_milli;
    SparkStatus status;

    if (collective_out == NULL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkTpCollectiveReset(collective_out);
    status = SparkTpCollectiveValidateConfig(config);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    collective_out->abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
    collective_out->tp_degree = config->tp_degree;
    collective_out->tp_rank = config->tp_rank;
    collective_out->step_count = SparkTpCollectiveStepCount(config->tp_degree);
    collective_out->operation_timeout_milli = config->operation_timeout_milli;
    collective_out->collective_identifier = config->collective_identifier;
    collective_out->next_operation_sequence = 1u;

    if (config->tp_degree == 1u)
    {
        return SPARK_STATUS_OK;
    }

    deadline_milli = SparkTpCollectiveBuildDeadline(config->connect_timeout_milli);
    status = SparkTpCollectiveListen(collective_out, config->listen_port);
    if (status != SPARK_STATUS_OK)
    {
        return SparkTpCollectiveFail(collective_out, status);
    }

    accept_count = 0u;
    for (step_index = 0u;
         status == SPARK_STATUS_OK && step_index < collective_out->step_count;
         ++step_index)
    {
        uint32_t partner_rank;

        partner_rank = config->tp_rank ^ (1u << step_index);
        if (partner_rank < config->tp_rank)
        {
            status = SparkTpCollectiveConnectPeer(
                collective_out,
                &config->peers[step_index],
                partner_rank,
                deadline_milli,
                &collective_out->step_sockets[step_index]);
        }
        else
        {
            accept_count += 1u;
        }
    }

    if (status == SPARK_STATUS_OK)
    {
        status = SparkTpCollectiveAcceptPeers(
            collective_out,
            accept_count,
            deadline_milli);
    }

    if (status != SPARK_STATUS_OK)
    {
        return SparkTpCollectiveFail(collective_out, status);
    }

    close(collective_out->listen_socket);
    collective_out->listen_socket = -1;
    return SPARK_STATUS_OK;
}

static int SparkTpCollectiveMemoryRangesOverlap(
    const void *first_memory,
    const void *second_memory,
    size_t memory_bytes)
{
    uintptr_t first_begin;
    uintptr_t second_begin;
    uintptr_t first_end;
    uintptr_t second_end;

    if (memory_bytes == 0u)
    {
        return 0;
    }

    first_begin = (uintptr_t)first_memory;
    second_begin = (uintptr_t)second_memory;
    if (first_begin > UINTPTR_MAX - memory_bytes ||
        second_begin > UINTPTR_MAX - memory_bytes)
    {
        return 1;
    }

    first_end = first_begin + memory_bytes;
    second_end = second_begin + memory_bytes;
    return first_begin < second_end && second_begin < first_end;
}

/* Widens one BF16 lane to F32: the high half of the F32 bit pattern. */
static float SparkTpCollectiveBf16ToF32(uint16_t value_bf16)
{
    uint32_t bits;
    float value;

    bits = (uint32_t)value_bf16 << 16u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Narrows one F32 lane to BF16 with round-to-nearest-even, matching the
   device-side narrowing a CUDA kernel would apply, so the host-staged and the
   future device-resident path round identically. NaN inputs are canonicalised
   before the bias add, which would otherwise carry into the exponent and
   produce infinity. Every rank runs this same code on the same partials, so
   the BF16 variant keeps the F32 variant's bitwise-identical-across-ranks
   guarantee. */
static uint16_t SparkTpCollectiveF32ToBf16(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u)
    {
        return (uint16_t)((bits >> 16u) | 0x0040u);
    }

    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

static void SparkTpCollectiveReduceSumF32(
    void *values,
    const void *scratch,
    uint64_t element_count)
{
    float *value_elements;
    const float *scratch_elements;
    uint64_t element_index;

    value_elements = (float *)values;
    scratch_elements = (const float *)scratch;
    for (element_index = 0u; element_index < element_count; ++element_index)
    {
        value_elements[element_index] =
            value_elements[element_index] + scratch_elements[element_index];
    }
}

/* The standard BF16-all-reduce recipe (vLLM custom all-reduce, NCCL): the
   wire and the staging buffers stay BF16, each exchange step accumulates in
   F32 and narrows once, so precision cost is one rounding per doubling step
   instead of one per element-pair in F32 staged at twice the bytes. */
static void SparkTpCollectiveReduceSumBf16(
    void *values,
    const void *scratch,
    uint64_t element_count)
{
    uint16_t *value_elements;
    const uint16_t *scratch_elements;
    uint64_t element_index;

    value_elements = (uint16_t *)values;
    scratch_elements = (const uint16_t *)scratch;
    for (element_index = 0u; element_index < element_count; ++element_index)
    {
        value_elements[element_index] = SparkTpCollectiveF32ToBf16(
            SparkTpCollectiveBf16ToF32(value_elements[element_index]) +
            SparkTpCollectiveBf16ToF32(scratch_elements[element_index]));
    }
}

typedef void (*SparkTpCollectiveReduceFunction)(
    void *values,
    const void *scratch,
    uint64_t element_count);

static SparkStatus SparkTpCollectiveAllReduceSum(
    SparkTpCollective *collective,
    void *values,
    uint64_t element_count,
    void *scratch,
    uint64_t element_bytes,
    uint32_t operation_kind,
    SparkTpCollectiveReduceFunction reduce_step)
{
    uint32_t expected_step_count;
    uint32_t step_index;
    uint64_t operation_sequence;
    uint64_t buffer_bytes;
    uint64_t deadline_milli;

    if (collective == NULL || values == NULL || element_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (collective->abi_version != SPARK_TP_COLLECTIVE_ABI_VERSION ||
        !SparkTpCollectiveDegreeIsSupported(collective->tp_degree) ||
        collective->tp_rank >= collective->tp_degree ||
        collective->failed != 0u ||
        collective->reserved0 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    expected_step_count = SparkTpCollectiveStepCount(collective->tp_degree);
    if (collective->step_count != expected_step_count)
    {
        return SparkTpCollectiveFail(
            collective,
            SPARK_STATUS_VALIDATION_FAILED);
    }

    if (collective->tp_degree == 1u)
    {
        return SPARK_STATUS_OK;
    }

    if (scratch == NULL ||
        collective->operation_timeout_milli == 0u ||
        collective->collective_identifier == 0u ||
        collective->next_operation_sequence == 0u ||
        collective->next_operation_sequence == UINT64_MAX ||
        element_count > (uint64_t)(SIZE_MAX / element_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    buffer_bytes = element_count * element_bytes;
    if (SparkTpCollectiveMemoryRangesOverlap(
            values,
            scratch,
            (size_t)buffer_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    operation_sequence = collective->next_operation_sequence;
    deadline_milli = SparkTpCollectiveBuildDeadline(
        collective->operation_timeout_milli);

    for (step_index = 0u; step_index < collective->step_count; ++step_index)
    {
        uint32_t partner_rank;
        int32_t step_socket;
        SparkStatus status;

        partner_rank = collective->tp_rank ^ (1u << step_index);
        step_socket = collective->step_sockets[step_index];
        if (step_socket < 0)
        {
            return SparkTpCollectiveFail(
                collective,
                SPARK_STATUS_VALIDATION_FAILED);
        }

        status = SparkTpCollectiveExchangeOperationHeaders(
            collective,
            step_socket,
            step_index,
            partner_rank,
            operation_sequence,
            element_count,
            operation_kind,
            deadline_milli);
        if (status != SPARK_STATUS_OK)
        {
            return SparkTpCollectiveFail(collective, status);
        }

        if (collective->tp_rank < partner_rank)
        {
            status = SparkTpCollectiveSendAllUntil(
                step_socket,
                values,
                buffer_bytes,
                deadline_milli);
            if (status == SPARK_STATUS_OK)
            {
                status = SparkTpCollectiveReceiveAllUntil(
                    step_socket,
                    scratch,
                    buffer_bytes,
                    deadline_milli);
            }
        }
        else
        {
            status = SparkTpCollectiveReceiveAllUntil(
                step_socket,
                scratch,
                buffer_bytes,
                deadline_milli);
            if (status == SPARK_STATUS_OK)
            {
                status = SparkTpCollectiveSendAllUntil(
                    step_socket,
                    values,
                    buffer_bytes,
                    deadline_milli);
            }
        }

        if (status != SPARK_STATUS_OK)
        {
            return SparkTpCollectiveFail(collective, status);
        }

        reduce_step(values, scratch, element_count);
    }

    collective->next_operation_sequence += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpCollectiveAllReduceSumF32(
    SparkTpCollective *collective,
    float *values,
    uint64_t element_count,
    float *scratch)
{
    return SparkTpCollectiveAllReduceSum(
        collective,
        values,
        element_count,
        scratch,
        (uint64_t)sizeof(float),
        SPARK_TP_COLLECTIVE_OPERATION_SUM_F32,
        SparkTpCollectiveReduceSumF32);
}

SparkStatus SparkTpCollectiveAllReduceSumBf16(
    SparkTpCollective *collective,
    uint16_t *values_bf16,
    uint64_t element_count,
    uint16_t *scratch_bf16)
{
    return SparkTpCollectiveAllReduceSum(
        collective,
        values_bf16,
        element_count,
        scratch_bf16,
        (uint64_t)sizeof(uint16_t),
        SPARK_TP_COLLECTIVE_OPERATION_SUM_BF16,
        SparkTpCollectiveReduceSumBf16);
}

void SparkTpCollectiveDestroy(SparkTpCollective *collective)
{
    if (collective == NULL)
    {
        return;
    }

    if (collective->abi_version == SPARK_TP_COLLECTIVE_ABI_VERSION)
    {
        SparkTpCollectiveCloseAll(collective);
    }
    SparkTpCollectiveReset(collective);
}
