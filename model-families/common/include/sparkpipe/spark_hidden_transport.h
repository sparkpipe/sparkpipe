#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HIDDEN_TRANSPORT_ABI_VERSION 3u
#define SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportInterface))
#define SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportEndpoint))
#define SPARK_HIDDEN_TRANSPORT_PACKET_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportPacket))
#define SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportCompletion))
#define SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT \
    ((uint32_t)sizeof(uint16_t))
#define SPARK_HIDDEN_TRANSPORT_COMPLETION_QUEUE_DEPTH 1024u
#define SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH \
    SPARK_HIDDEN_TRANSPORT_COMPLETION_QUEUE_DEPTH
#define SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_MODULE_ID \
    "spark.hidden_transport.persistent_ring.device.v1"
#define SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID \
    "spark.hidden_transport.spark_host_pinned_rdma.verbs.v1"
#define SPARK_HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA_VERBS_MODULE_ID \
    "spark.hidden_transport.spark_gpudirect_rdma.verbs.v1"
#define SPARK_HIDDEN_TRANSPORT_TCP_CUDA_HOST_MODULE_ID \
    "spark.hidden_transport.tcp.cuda_host.v1"
#define SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_INFINIBAND_SYSFS_PATH \
    "/sys/class/infiniband"
#define SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_STATISTICS_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportPersistentRingStatistics))
#define SPARK_HIDDEN_TRANSPORT_DYNAMIC_LIBRARY_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportDynamicLibrary))
#define SPARK_HIDDEN_TRANSPORT_POLL_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkHiddenTransportPollDescriptor))
#define SPARK_HIDDEN_TRANSPORT_INTERFACE_SYMBOL \
    "SparkHiddenTransportGetInterface"

#define SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION 0x00000001u
#define SPARK_HIDDEN_TRANSPORT_CAP_DEVICE_POINTER_IO 0x00000002u
#define SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED 0x00000004u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_HOST_STAGING 0x00000008u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_DEVICE_MEMCPY 0x00000010u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT 0x00000020u
#define SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT 0x00000040u
#define SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION 0x00000080u
#define SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS 0x00000100u
#define SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA 0x00000200u
#define SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY 0x00000400u
#define SPARK_HIDDEN_TRANSPORT_CAP_MULTI_LANE 0x00000800u
#define SPARK_HIDDEN_TRANSPORT_CAP_REMOTE_COMPLETION_DOORBELL 0x00001000u
#define SPARK_HIDDEN_TRANSPORT_CAP_GPUDIRECT_RDMA 0x00002000u
#define SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY 0x80000000u

#define SPARK_HIDDEN_TRANSPORT_POLL_READ 0x00000001u
#define SPARK_HIDDEN_TRANSPORT_POLL_WRITE 0x00000002u

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

#define SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA | \
     SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY)

#define SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION | \
     SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS | \
     SPARK_HIDDEN_TRANSPORT_CAP_MULTI_LANE | \
     SPARK_HIDDEN_TRANSPORT_CAP_REMOTE_COMPLETION_DOORBELL)

#define SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_GPUDIRECT_RDMA)

#define SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_GPUDIRECT_RDMA_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION | \
     SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS | \
     SPARK_HIDDEN_TRANSPORT_CAP_MULTI_LANE | \
     SPARK_HIDDEN_TRANSPORT_CAP_REMOTE_COMPLETION_DOORBELL)

#define SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS \
    (SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION | \
     SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT | \
     SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY)

#define SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SIMULATION_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION)

#define SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS \
    (SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_CONNECTION | \
     SPARK_HIDDEN_TRANSPORT_CAP_DEVICE_POINTER_IO | \
     SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_FILE_TRANSPORT | \
     SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT)

#define SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PIPELINE_HOST_STAGED_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION)

#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 0x00000001u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER 0x00000002u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_END_OF_SEQUENCE 0x00000004u
#define SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD 0x00000008u

#define SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_NONE 0u
#define SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_INDEXSHARE_SELECTED_TOKENS 1u
#define SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_DSPARK_HIDDEN_TAP 2u

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

typedef struct SparkHiddenTransportCompletionQueue
{
    SparkHiddenTransportCompletion
        entries[SPARK_HIDDEN_TRANSPORT_COMPLETION_QUEUE_DEPTH];
    uint32_t head;
    uint32_t count;
    uint64_t total_count;
    uint64_t dropped_count;
} SparkHiddenTransportCompletionQueue;

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

typedef struct SparkHiddenTransportPollDescriptor
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    int32_t fd;
    uint32_t events;
} SparkHiddenTransportPollDescriptor;

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
typedef SparkStatus (*SparkHiddenTransportGetPollDescriptorsFunction)(
    void *transport_state,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out);

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
    SparkHiddenTransportGetPollDescriptorsFunction get_poll_descriptors;
} SparkHiddenTransportInterface;

typedef const SparkHiddenTransportInterface *(*SparkHiddenTransportGetInterfaceFunction)(
    void);

typedef struct SparkHiddenTransportDynamicLibrary
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    void *dynamic_library;
    SparkHiddenTransportInterface transport_interface;
} SparkHiddenTransportDynamicLibrary;

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
SparkStatus SparkHiddenTransportLoadInterfaceFromSharedObject(
    const char *shared_object_path,
    uint32_t required_capability_flags,
    SparkHiddenTransportDynamicLibrary *library);
void SparkHiddenTransportUnloadInterface(
    SparkHiddenTransportDynamicLibrary *library);

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
SparkStatus SparkHiddenTransportGetPollDescriptors(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out);
void SparkHiddenTransportCompletionQueueInitialize(
    SparkHiddenTransportCompletionQueue *queue);
uint32_t SparkHiddenTransportCompletionQueueIsFull(
    const SparkHiddenTransportCompletionQueue *queue);
SparkStatus SparkHiddenTransportCompletionQueuePush(
    SparkHiddenTransportCompletionQueue *queue,
    const SparkHiddenTransportCompletion *completion);
SparkStatus SparkHiddenTransportCompletionQueuePushPacket(
    SparkHiddenTransportCompletionQueue *queue,
    const SparkHiddenTransportPacket *packet,
    SparkStatus status,
    uint64_t service_time_ns);
SparkStatus SparkHiddenTransportCompletionQueuePop(
    SparkHiddenTransportCompletionQueue *queue,
    SparkHiddenTransportCompletion *completion);
SparkStatus SparkHiddenTransportPersistentRingGetInterface(
    SparkHiddenTransportInterface *transport_interface);
SparkStatus SparkHiddenTransportPersistentRingGetStatistics(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPersistentRingStatistics *statistics);
void SparkHiddenTransportInitializeSparkHostRdmaEndpoint(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t hidden_dimension,
    uint32_t max_active_sequence_count,
    uint64_t validated_latency_ns,
    const char *route_name);
SparkStatus SparkHiddenTransportValidateSparkHostRdmaEndpoint(
    const SparkHiddenTransportEndpoint *endpoint);
SparkStatus SparkHiddenTransportSparkHostRdmaVerbsPreflight(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *infiniband_sysfs_path);
void SparkHiddenTransportInitializeSparkGpudirectRdmaEndpoint(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t hidden_dimension,
    uint32_t max_active_sequence_count,
    uint64_t validated_latency_ns,
    const char *route_name);
SparkStatus SparkHiddenTransportValidateSparkGpudirectRdmaEndpoint(
    const SparkHiddenTransportEndpoint *endpoint);
SparkStatus SparkHiddenTransportSparkGpudirectRdmaVerbsPreflight(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *infiniband_sysfs_path);

#ifdef __cplusplus
}
#endif
