#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HIDDEN_TRANSPORT_ABI_VERSION 1u
#define SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportInterface))
#define SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportEndpoint))
#define SPARK_HIDDEN_TRANSPORT_PACKET_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportPacket))
#define SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT 2u

#define SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION 0x00000001u
#define SPARK_HIDDEN_TRANSPORT_CAP_DEVICE_POINTER_IO 0x00000002u
#define SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED 0x00000004u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_HOST_STAGING 0x00000008u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_DEVICE_MEMCPY 0x00000010u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT 0x00000020u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT 0x00000040u

#define SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS \
    (SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION | \
     SPARK_HIDDEN_TRANSPORT_CAP_DEVICE_POINTER_IO | \
     SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_HOST_STAGING | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT)

#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 0x00000001u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER 0x00000002u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_END_OF_SEQUENCE 0x00000004u

typedef struct SparkHiddenTransportEndpoint
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capability_flags;
    uint32_t hidden_dimension;
    uint32_t bytes_per_sequence;
    uint32_t max_active_sequence_count;
    uint64_t max_packet_bytes;
    uint64_t validated_latency_ns;
    const char *transport_module_id;
    const char *route_name;
} SparkHiddenTransportEndpoint;

typedef struct SparkHiddenTransportPacket
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t active_sequence_count;
    uint32_t hidden_dimension;
    uint32_t bytes_per_sequence;
    uint64_t sequence_id;
    uint64_t token_index;
    const void *hidden_bf16;
    void *cuda_stream;
} SparkHiddenTransportPacket;

typedef struct SparkHiddenTransportCompletion
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkStatus status;
    uint32_t active_sequence_count;
    uint64_t sequence_id;
    uint64_t token_index;
    uint64_t transfer_bytes;
    uint64_t service_time_ns;
} SparkHiddenTransportCompletion;

typedef SparkStatus (*SparkHiddenTransportInitializeFunction)(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state);
typedef void (*SparkHiddenTransportDestroyFunction)(void *transport_state);
typedef SparkStatus (*SparkHiddenTransportPostReceiveFunction)(
    void *transport_state,
    SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkHiddenTransportSendFunction)(
    void *transport_state,
    const SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkHiddenTransportPollFunction)(
    void *transport_state,
    SparkHiddenTransportCompletion *completion);

typedef struct SparkHiddenTransportInterface
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capability_flags;
    uint32_t reserved;
    SparkHiddenTransportInitializeFunction initialize;
    SparkHiddenTransportDestroyFunction destroy;
    SparkHiddenTransportPostReceiveFunction post_receive;
    SparkHiddenTransportSendFunction send;
    SparkHiddenTransportPollFunction poll;
} SparkHiddenTransportInterface;

SparkStatus SparkHiddenTransportValidateEndpoint(
    const SparkHiddenTransportEndpoint *endpoint);
SparkStatus SparkHiddenTransportValidateInterface(
    const SparkHiddenTransportInterface *transport_interface,
    uint32_t required_capability_flags);

#ifdef __cplusplus
}
#endif
