#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HIDDEN_TRANSPORT_ABI_VERSION 2u
#define SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportInterface))
#define SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportEndpoint))
#define SPARK_HIDDEN_TRANSPORT_PACKET_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportPacket))
#define SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportCompletion))
#define SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT 2u
#define SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH 1024u
#define SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_MODULE_ID \
    "spark.hidden_transport.persistent_ring.device.v1"
#define SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_VERBS_MODULE_ID \
    "spark.hidden_transport.gpudirect_rdma.verbs.v1"
#define SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_PEERMEM_SYSFS_PATH \
    "/sys/module/nvidia_peermem"
#define SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_INFINIBAND_SYSFS_PATH \
    "/sys/class/infiniband"
#define SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_STATISTICS_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportPersistentRingStatistics))

#define SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION 0x00000001u
#define SPARK_HIDDEN_TRANSPORT_CAP_DEVICE_POINTER_IO 0x00000002u
#define SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED 0x00000004u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_HOST_STAGING 0x00000008u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_DEVICE_MEMCPY 0x00000010u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT 0x00000020u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT 0x00000040u
#define SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION 0x00000080u
#define SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY 0x80000000u

#define SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS \
    (SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION | \
     SPARK_HIDDEN_TRANSPORT_CAP_DEVICE_POINTER_IO | \
     SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_HOST_STAGING | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_DEVICE_MEMCPY | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT)

#define SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION)

#define SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS \
    (SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION | \
     SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT | \
     SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY)

#define SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SIMULATION_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION)

#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 0x00000001u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER 0x00000002u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_END_OF_SEQUENCE 0x00000004u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD 0x00000008u

typedef struct SparkHiddenTransportSession SparkHiddenTransportSession;

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
    const void *sideband_payload;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
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

typedef struct SparkHiddenTransportPersistentRingStatistics
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint64_t send_count;
    uint64_t receive_count;
    uint64_t completion_count;
    uint64_t dropped_completion_count;
    uint32_t queued_completion_count;
    uint32_t queue_depth;
} SparkHiddenTransportPersistentRingStatistics;

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
typedef SparkStatus (*SparkHiddenTransportPostReceiveBatchFunction)(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count);
typedef SparkStatus (*SparkHiddenTransportSendBatchFunction)(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count);

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
    SparkHiddenTransportPostReceiveBatchFunction post_receive_batch;
    SparkHiddenTransportSendBatchFunction send_batch;
} SparkHiddenTransportInterface;

SparkStatus SparkHiddenTransportValidateEndpoint(
    const SparkHiddenTransportEndpoint *endpoint);
SparkStatus SparkHiddenTransportValidatePacket(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportPacket *packet);
SparkStatus SparkHiddenTransportValidatePacketBatch(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count);
SparkStatus SparkHiddenTransportValidateInterface(
    const SparkHiddenTransportInterface *transport_interface,
    uint32_t required_capability_flags);

SparkStatus SparkHiddenTransportOpen(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportInterface *transport_interface,
    uint32_t required_capability_flags,
    SparkHiddenTransportSession **session_out);
void SparkHiddenTransportClose(SparkHiddenTransportSession *session);
SparkStatus SparkHiddenTransportPostReceive(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPacket *packet);
SparkStatus SparkHiddenTransportSend(
    SparkHiddenTransportSession *session,
    const SparkHiddenTransportPacket *packet);
SparkStatus SparkHiddenTransportPostReceiveBatch(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count);
SparkStatus SparkHiddenTransportSendBatch(
    SparkHiddenTransportSession *session,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count);
SparkStatus SparkHiddenTransportPoll(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportCompletion *completion);
SparkStatus SparkHiddenTransportPersistentRingGetInterface(
    SparkHiddenTransportInterface *transport_interface);
SparkStatus SparkHiddenTransportPersistentRingGetStatistics(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPersistentRingStatistics *statistics);
SparkStatus SparkHiddenTransportGpudirectRdmaVerbsPreflight(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *peermem_sysfs_path,
    const char *infiniband_sysfs_path);

#ifdef __cplusplus
}
#endif
