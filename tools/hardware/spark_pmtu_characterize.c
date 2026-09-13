#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#if defined(__linux__)
#include <netinet/ip.h>
#endif
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "spark_probe_common.h"

#define SPARK_PMTU_MAGIC 0x53504d54u
#define SPARK_PMTU_DEFAULT_PORT 46841u
#define SPARK_PMTU_TIMEOUT_MS 500
#define SPARK_PMTU_MAX_UDP_PAYLOAD 65507u
#if defined(__APPLE__)
#define SPARK_PMTU_IP_DONTFRAG_OPTION 28
#endif

typedef struct SparkPmtuPacketHeader
{
    uint32_t magic;
    uint32_t sequence;
    uint32_t payload_bytes;
    uint32_t reserved;
    uint64_t fingerprint;
} SparkPmtuPacketHeader;

typedef struct SparkPmtuOptions
{
    uint32_t server_mode;
    const char *bind_address;
    const char *peer_address;
    uint32_t port;
    uint32_t minimum_payload_bytes;
    uint32_t maximum_payload_bytes;
    uint32_t iterations;
    uint32_t idle_timeout_seconds;
    const char *question_id;
    const char *source_package_sha256;
    const char *run_id;
    const char *topology;
    const char *node;
    const char *peer;
    const char *candidate;
    const char *output_path;
} SparkPmtuOptions;

static volatile sig_atomic_t SparkPmtuStopRequested;

static uint64_t SparkPmtuHostToBig64(uint64_t value)
{
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap64(value);
#else
    return value;
#endif
}

static uint64_t SparkPmtuBigToHost64(uint64_t value)
{
    return SparkPmtuHostToBig64(value);
}

static void SparkPmtuSignalHandler(int signal_number)
{
    (void)signal_number;
    SparkPmtuStopRequested = 1;
}

static uint64_t SparkPmtuPayloadFingerprint(
    const uint8_t *payload,
    uint32_t payload_bytes)
{
    uint64_t fingerprint;
    uint32_t index;

    fingerprint = 0u;
    for (index = 0u; index < payload_bytes; ++index)
    {
        fingerprint ^= SparkProbeMix64((uint64_t)payload[index] ^
            ((uint64_t)index << 16u));
    }
    return fingerprint;
}

static void SparkPmtuFillPayload(uint8_t *payload, uint32_t payload_bytes, uint32_t sequence)
{
    uint32_t index;

    for (index = 0u; index < payload_bytes; ++index)
    {
        payload[index] = (uint8_t)SparkProbeMix64(
            ((uint64_t)sequence << 32u) | (uint64_t)index);
    }
}

static int SparkPmtuResolve(
    const char *address,
    uint32_t port,
    int passive,
    struct sockaddr_storage *storage_out,
    socklen_t *length_out)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *walk;
    char port_text[16];
    int status;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    if (passive != 0)
    {
        hints.ai_flags = AI_PASSIVE;
    }
    snprintf(port_text, sizeof(port_text), "%u", port);
    result = 0;
    status = getaddrinfo(address, port_text, &hints, &result);
    if (status != 0)
    {
        fprintf(stderr, "getaddrinfo failed: %s\n", gai_strerror(status));
        return 0;
    }
    status = 0;
    for (walk = result; walk != 0; walk = walk->ai_next)
    {
        if (walk->ai_addrlen <= sizeof(*storage_out))
        {
            memset(storage_out, 0, sizeof(*storage_out));
            memcpy(storage_out, walk->ai_addr, walk->ai_addrlen);
            *length_out = (socklen_t)walk->ai_addrlen;
            status = 1;
            break;
        }
    }
    freeaddrinfo(result);
    return status;
}

static int SparkPmtuRunServer(const SparkPmtuOptions *options)
{
    struct sockaddr_storage bind_address;
    socklen_t bind_length;
    struct sigaction action;
    uint8_t *buffer;
    int socket_descriptor;
    int reuse;
    uint64_t last_activity_ns;

    if (!SparkPmtuResolve(options->bind_address, options->port, 1,
            &bind_address, &bind_length))
    {
        return 1;
    }
    socket_descriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_descriptor < 0)
    {
        perror("socket");
        return 1;
    }
    reuse = 1;
    (void)setsockopt(socket_descriptor, SOL_SOCKET, SO_REUSEADDR,
        &reuse, sizeof(reuse));
    if (bind(socket_descriptor, (struct sockaddr *)&bind_address, bind_length) != 0)
    {
        perror("bind");
        close(socket_descriptor);
        return 1;
    }
    buffer = (uint8_t *)malloc(SPARK_PMTU_MAX_UDP_PAYLOAD);
    if (buffer == 0)
    {
        close(socket_descriptor);
        return 1;
    }
    memset(&action, 0, sizeof(action));
    action.sa_handler = SparkPmtuSignalHandler;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGINT, &action, 0);
    (void)sigaction(SIGTERM, &action, 0);
    fprintf(stdout, "READY port=%u\n", options->port);
    fflush(stdout);
    last_activity_ns = SparkProbeMonotonicNanoseconds();
    while (SparkPmtuStopRequested == 0)
    {
        struct pollfd poll_descriptor;
        struct sockaddr_storage peer_address;
        socklen_t peer_length;
        ssize_t received;
        SparkPmtuPacketHeader *header;
        uint32_t payload_bytes;
        uint64_t fingerprint;
        uint64_t now_ns;
        int poll_status;

        poll_descriptor.fd = socket_descriptor;
        poll_descriptor.events = POLLIN;
        poll_descriptor.revents = 0;
        poll_status = poll(&poll_descriptor, 1u, 1000);
        if (poll_status < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            perror("poll");
            break;
        }
        now_ns = SparkProbeMonotonicNanoseconds();
        if (poll_status == 0)
        {
            if (last_activity_ns != 0u && now_ns > last_activity_ns &&
                now_ns - last_activity_ns >=
                    (uint64_t)options->idle_timeout_seconds * 1000000000ull)
            {
                break;
            }
            continue;
        }
        peer_length = sizeof(peer_address);
        received = recvfrom(socket_descriptor, buffer,
            SPARK_PMTU_MAX_UDP_PAYLOAD, 0,
            (struct sockaddr *)&peer_address, &peer_length);
        if (received < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            perror("recvfrom");
            break;
        }
        if ((size_t)received < sizeof(SparkPmtuPacketHeader))
        {
            continue;
        }
        header = (SparkPmtuPacketHeader *)buffer;
        payload_bytes = (uint32_t)received - (uint32_t)sizeof(*header);
        if (ntohl(header->magic) != SPARK_PMTU_MAGIC ||
            ntohl(header->payload_bytes) != payload_bytes)
        {
            continue;
        }
        fingerprint = SparkPmtuPayloadFingerprint(
            buffer + sizeof(*header), payload_bytes);
        if (fingerprint != SparkPmtuBigToHost64(header->fingerprint))
        {
            continue;
        }
        last_activity_ns = now_ns;
        if (sendto(socket_descriptor, buffer, (size_t)received, 0,
                (struct sockaddr *)&peer_address, peer_length) != received &&
            errno != EINTR)
        {
            perror("sendto");
        }
    }
    free(buffer);
    close(socket_descriptor);
    return 0;
}

static void SparkPmtuDrainSocket(int socket_descriptor)
{
    uint8_t discard[SPARK_PMTU_MAX_UDP_PAYLOAD];

    for (;;)
    {
        struct pollfd poll_descriptor;
        ssize_t received;

        poll_descriptor.fd = socket_descriptor;
        poll_descriptor.events = POLLIN;
        poll_descriptor.revents = 0;
        if (poll(&poll_descriptor, 1u, 0) <= 0)
        {
            break;
        }
        received = recv(socket_descriptor, discard, sizeof(discard), 0);
        if (received >= 0)
        {
            continue;
        }
        if (errno == EINTR)
        {
            continue;
        }
        break;
    }
}

static int SparkPmtuTryPayload(
    int socket_descriptor,
    uint32_t payload_bytes,
    uint32_t iterations,
    uint64_t *latencies_out)
{
    uint8_t *packet;
    size_t packet_bytes;
    uint32_t iteration;
    int success;

    packet_bytes = sizeof(SparkPmtuPacketHeader) + payload_bytes;
    packet = (uint8_t *)malloc(packet_bytes);
    if (packet == 0)
    {
        return 0;
    }
    SparkPmtuDrainSocket(socket_descriptor);
    success = 1;
    for (iteration = 0u; iteration < iterations; ++iteration)
    {
        SparkPmtuPacketHeader *header;
        uint32_t wire_sequence;
        uint64_t start_ns;
        uint64_t deadline_ns;
        ssize_t sent;
        int matched;

        wire_sequence = (uint32_t)SparkProbeMix64(
            ((uint64_t)payload_bytes << 32u) | (uint64_t)iteration);
        memset(packet, 0, sizeof(SparkPmtuPacketHeader));
        SparkPmtuFillPayload(packet + sizeof(SparkPmtuPacketHeader),
            payload_bytes, wire_sequence);
        header = (SparkPmtuPacketHeader *)packet;
        header->magic = htonl(SPARK_PMTU_MAGIC);
        header->sequence = htonl(wire_sequence);
        header->payload_bytes = htonl(payload_bytes);
        header->fingerprint = SparkPmtuHostToBig64(SparkPmtuPayloadFingerprint(
            packet + sizeof(*header), payload_bytes));
        start_ns = SparkProbeMonotonicNanoseconds();
        deadline_ns = start_ns + ((uint64_t)SPARK_PMTU_TIMEOUT_MS * 1000000u);
        sent = send(socket_descriptor, packet, packet_bytes, 0);
        if (sent != (ssize_t)packet_bytes)
        {
            perror("send");
            success = 0;
            break;
        }
        matched = 0;
        while (matched == 0)
        {
            struct pollfd poll_descriptor;
            uint64_t now_ns;
            uint64_t remaining_ns;
            int timeout_ms;
            ssize_t received;

            now_ns = SparkProbeMonotonicNanoseconds();
            if (now_ns >= deadline_ns)
            {
                break;
            }
            remaining_ns = deadline_ns - now_ns;
            timeout_ms = (int)((remaining_ns + 999999u) / 1000000u);
            if (timeout_ms < 1)
            {
                timeout_ms = 1;
            }
            poll_descriptor.fd = socket_descriptor;
            poll_descriptor.events = POLLIN;
            poll_descriptor.revents = 0;
            if (poll(&poll_descriptor, 1u, timeout_ms) <= 0)
            {
                break;
            }
            received = recv(socket_descriptor, packet, packet_bytes, 0);
            if (received < 0)
            {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    continue;
                }
                break;
            }
            if (received != (ssize_t)packet_bytes)
            {
                continue;
            }
            header = (SparkPmtuPacketHeader *)packet;
            if (ntohl(header->magic) != SPARK_PMTU_MAGIC ||
                ntohl(header->sequence) != wire_sequence ||
                ntohl(header->payload_bytes) != payload_bytes ||
                SparkPmtuBigToHost64(header->fingerprint) !=
                    SparkPmtuPayloadFingerprint(
                        packet + sizeof(SparkPmtuPacketHeader), payload_bytes))
            {
                continue;
            }
            latencies_out[iteration] = SparkProbeMonotonicNanoseconds() - start_ns;
            matched = 1;
        }
        if (matched == 0)
        {
            success = 0;
            break;
        }
    }
    free(packet);
    return success;
}

static int SparkPmtuWriteReceipt(
    const SparkPmtuOptions *options,
    uint32_t maximum_payload,
    uint32_t reported_path_mtu,
    const SparkProbeLatencySummary *latency)
{
    FILE *output;

    output = options->output_path == 0 ? stdout : fopen(options->output_path, "wb");
    if (output == 0)
    {
        return 0;
    }
    fputs("{\n  \"schema_version\": 1,\n  \"receipt_kind\": \"spark_hardware_probe\",\n  \"run_id\": ", output);
    SparkProbeWriteJsonString(output, options->run_id);
    fputs(",\n  \"probe_id\": \"pmtu_characterize\",\n  \"source_identity\": {\"source_package_sha256\": ", output);
    SparkProbeWriteJsonString(output, options->source_package_sha256);
    fputs("},\n  \"scope\": {\"topology\": ", output);
    SparkProbeWriteJsonString(output, options->topology);
    fputs(", \"node\": ", output);
    SparkProbeWriteJsonString(output, options->node);
    fputs(", \"peer\": ", output);
    SparkProbeWriteJsonString(output, options->peer);
    fputs("},\n  \"answers\": [\n    {\"question_id\": \"NET-PMTU-001\", \"status\": \"measured\", \"summary\": {\"integrity_verified\": true, \"fragmentation_forbidden\": true}, \"observations\": [{\"parameters\": {\"candidate\": ", output);
    SparkProbeWriteJsonString(output, options->candidate);
    fprintf(output,
        ", \"minimum_payload_bytes\": %u, \"maximum_payload_bytes\": %u, \"iterations\": %u}, "
        "\"metrics\": {\"maximum_payload_bytes\": %u, \"reported_path_mtu_bytes\": %u, "
        "\"latency_min_ns\": %" PRIu64 ", \"latency_p50_ns\": %" PRIu64 ", "
        "\"latency_p95_ns\": %" PRIu64 ", \"latency_p99_ns\": %" PRIu64 ", "
        "\"latency_max_ns\": %" PRIu64 ", \"sample_count\": %u, "
        "\"integrity_pass\": true}}]}\n  ]\n}\n",
        options->minimum_payload_bytes,
        options->maximum_payload_bytes,
        options->iterations,
        maximum_payload,
        reported_path_mtu,
        latency->minimum_ns,
        latency->p50_ns,
        latency->p95_ns,
        latency->p99_ns,
        latency->maximum_ns,
        options->iterations);
    if (output != stdout)
    {
        fclose(output);
    }
    return 1;
}

static int SparkPmtuRunClient(const SparkPmtuOptions *options)
{
    struct sockaddr_storage peer_address;
    socklen_t peer_length;
    uint64_t *latencies;
    uint32_t low;
    uint32_t high;
    uint32_t best;
    uint32_t reported_path_mtu;
    int socket_descriptor;
    int status;

    if (!SparkPmtuResolve(options->peer_address, options->port, 0,
            &peer_address, &peer_length))
    {
        return 1;
    }
    socket_descriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_descriptor < 0)
    {
        perror("socket");
        return 1;
    }
#if defined(__linux__)
    {
        int discover;

        discover = IP_PMTUDISC_DO;
        if (setsockopt(socket_descriptor, IPPROTO_IP, IP_MTU_DISCOVER,
                &discover, sizeof(discover)) != 0)
        {
            perror("PMTU socket setup");
            close(socket_descriptor);
            return 1;
        }
    }
#elif defined(__APPLE__)
    {
        int dont_fragment;

        dont_fragment = 1;
        if (setsockopt(socket_descriptor, IPPROTO_IP,
                SPARK_PMTU_IP_DONTFRAG_OPTION,
                &dont_fragment, sizeof(dont_fragment)) != 0)
        {
            perror("PMTU socket setup");
            close(socket_descriptor);
            return 1;
        }
    }
#else
    fprintf(stderr, "PMTU DF discovery is unavailable on this platform\n");
    close(socket_descriptor);
    return 1;
#endif
    if (connect(socket_descriptor, (struct sockaddr *)&peer_address, peer_length) != 0)
    {
        perror("PMTU socket setup");
        close(socket_descriptor);
        return 1;
    }
    latencies = (uint64_t *)calloc(options->iterations, sizeof(uint64_t));
    if (latencies == 0)
    {
        close(socket_descriptor);
        return 1;
    }
    low = options->minimum_payload_bytes;
    high = options->maximum_payload_bytes;
    best = 0u;
    while (low <= high)
    {
        uint32_t middle;

        middle = low + ((high - low) / 2u);
        memset(latencies, 0, (size_t)options->iterations * sizeof(uint64_t));
        if (SparkPmtuTryPayload(socket_descriptor, middle,
                options->iterations, latencies))
        {
            best = middle;
            if (middle == UINT32_MAX)
            {
                break;
            }
            low = middle + 1u;
        }
        else
        {
            if (middle == 0u)
            {
                break;
            }
            high = middle - 1u;
        }
    }
    if (best < options->minimum_payload_bytes)
    {
        fprintf(stderr, "no payload at or above the required minimum succeeded\n");
        free(latencies);
        close(socket_descriptor);
        return 1;
    }
    memset(latencies, 0, (size_t)options->iterations * sizeof(uint64_t));
    if (!SparkPmtuTryPayload(socket_descriptor, best,
            options->iterations, latencies))
    {
        fprintf(stderr, "best payload did not reproduce\n");
        free(latencies);
        close(socket_descriptor);
        return 1;
    }
    {
        socklen_t option_length;
        int path_mtu;

#if defined(__linux__)
        path_mtu = 0;
        option_length = sizeof(path_mtu);
        if (getsockopt(socket_descriptor, IPPROTO_IP, IP_MTU,
                &path_mtu, &option_length) == 0 && path_mtu > 0)
        {
            reported_path_mtu = (uint32_t)path_mtu;
        }
        else
        {
            reported_path_mtu = best +
                (uint32_t)sizeof(SparkPmtuPacketHeader) + 28u;
        }
#else
        (void)option_length;
        (void)path_mtu;
        reported_path_mtu = best +
            (uint32_t)sizeof(SparkPmtuPacketHeader) + 28u;
#endif
    }
    {
        SparkProbeLatencySummary summary;

        summary = SparkProbeSummarizeLatency(latencies, options->iterations);
        status = SparkPmtuWriteReceipt(options, best, reported_path_mtu, &summary) ? 0 : 1;
    }
    free(latencies);
    close(socket_descriptor);
    return status;
}

static void SparkPmtuUsage(const char *program_name)
{
    fprintf(stderr,
        "usage: %s --server [--bind ADDRESS] [--port N] [--idle-timeout-seconds N]\n"
        "   or: %s --question NET-PMTU-001 --peer-address ADDRESS [--port N] "
        "--minimum-payload-bytes N --maximum-payload-bytes N --iterations N "
        "--source-package-sha256 HASH --run-id ID --topology ID --node ID --peer ID "
        "--candidate udp_df_binary_search [--output FILE]\n",
        program_name,
        program_name);
}

static int SparkPmtuParseOptions(
    int argument_count,
    char **arguments,
    SparkPmtuOptions *options)
{
    int index;

    memset(options, 0, sizeof(*options));
    options->bind_address = "0.0.0.0";
    options->port = SPARK_PMTU_DEFAULT_PORT;
    options->minimum_payload_bytes = 512u;
    options->maximum_payload_bytes = SPARK_PMTU_MAX_UDP_PAYLOAD - sizeof(SparkPmtuPacketHeader);
    options->iterations = 5u;
    options->idle_timeout_seconds = 60u;
    options->candidate = "udp_df_binary_search";
    for (index = 1; index < argument_count; ++index)
    {
        const char *argument;
        const char **text_destination;

        argument = arguments[index];
        text_destination = 0;
        if (strcmp(argument, "--server") == 0)
        {
            options->server_mode = 1u;
            continue;
        }
        if (strcmp(argument, "--bind") == 0)
        {
            text_destination = &options->bind_address;
        }
        else if (strcmp(argument, "--peer-address") == 0)
        {
            text_destination = &options->peer_address;
        }
        else if (strcmp(argument, "--question") == 0)
        {
            text_destination = &options->question_id;
        }
        else if (strcmp(argument, "--source-package-sha256") == 0)
        {
            text_destination = &options->source_package_sha256;
        }
        else if (strcmp(argument, "--run-id") == 0)
        {
            text_destination = &options->run_id;
        }
        else if (strcmp(argument, "--topology") == 0)
        {
            text_destination = &options->topology;
        }
        else if (strcmp(argument, "--node") == 0)
        {
            text_destination = &options->node;
        }
        else if (strcmp(argument, "--peer") == 0)
        {
            text_destination = &options->peer;
        }
        else if (strcmp(argument, "--candidate") == 0)
        {
            text_destination = &options->candidate;
        }
        else if (strcmp(argument, "--output") == 0)
        {
            text_destination = &options->output_path;
        }
        if (text_destination != 0)
        {
            if (index + 1 >= argument_count)
            {
                return 0;
            }
            *text_destination = arguments[++index];
            continue;
        }
        if (strcmp(argument, "--idle-timeout-seconds") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u, 3600u,
                    &options->idle_timeout_seconds))
            {
                return 0;
            }
        }
        else if (strcmp(argument, "--port") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u, 65535u, &options->port))
            {
                return 0;
            }
        }
        else if (strcmp(argument, "--minimum-payload-bytes") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u,
                    SPARK_PMTU_MAX_UDP_PAYLOAD - sizeof(SparkPmtuPacketHeader),
                    &options->minimum_payload_bytes))
            {
                return 0;
            }
        }
        else if (strcmp(argument, "--maximum-payload-bytes") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u,
                    SPARK_PMTU_MAX_UDP_PAYLOAD - sizeof(SparkPmtuPacketHeader),
                    &options->maximum_payload_bytes))
            {
                return 0;
            }
        }
        else if (strcmp(argument, "--iterations") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u, 1000u, &options->iterations))
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }
    if (options->server_mode != 0u)
    {
        return options->bind_address != 0;
    }
    return strcmp(options->question_id == 0 ? "" : options->question_id, "NET-PMTU-001") == 0 &&
        options->peer_address != 0 && options->minimum_payload_bytes <= options->maximum_payload_bytes &&
        SparkProbeHexSha256IsValid(options->source_package_sha256) &&
        SparkProbeIdentifierIsValid(options->run_id) &&
        SparkProbeIdentifierIsValid(options->topology) &&
        SparkProbeIdentifierIsValid(options->node) &&
        SparkProbeIdentifierIsValid(options->peer) &&
        strcmp(options->candidate, "udp_df_binary_search") == 0;
}

int main(int argument_count, char **arguments)
{
    SparkPmtuOptions options;

    if (!SparkPmtuParseOptions(argument_count, arguments, &options))
    {
        SparkPmtuUsage(arguments[0]);
        return 2;
    }
    if (options.server_mode != 0u)
    {
        return SparkPmtuRunServer(&options);
    }
    return SparkPmtuRunClient(&options);
}
