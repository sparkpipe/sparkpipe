#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MAGIC 0x53475055u
#define SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_VERSION 5u
#define SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MODULE_ID_BYTES 128u
#define SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_ROUTE_NAME_BYTES 256u
#define SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_HOST_BYTES 256u

typedef struct SparkHiddenTransportRdmaV4Identity
{
    uint32_t magic;
    uint32_t protocol_version;
    uint32_t transport_abi_version;
    uint32_t descriptor_bytes;
    uint32_t sender_role;
    uint32_t peer_sender_role;
    uint32_t local_rank;
    uint32_t peer_rank;
    uint32_t source_rank;
    uint32_t sink_rank;
    uint32_t control_port;
    uint32_t hidden_dimension;
    uint32_t bytes_per_sequence;
    uint32_t max_active_sequence_count;
    uint32_t persistent_credit_count;
    uint32_t lane_count;
    uint32_t doorbell_max_bytes;
    uint32_t memory_mode;
    uint32_t capability_flags;
    uint32_t reserved;
    uint64_t max_packet_bytes;
    uint64_t route_identifier;
    uint64_t fixed_buffer_address;
    uint64_t fixed_buffer_bytes;
    uint32_t fixed_buffer_rkey;
    char transport_module_id[
        SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MODULE_ID_BYTES];
    char route_name[SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_ROUTE_NAME_BYTES];
    char source_host[SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_HOST_BYTES];
    char sink_host[SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_HOST_BYTES];
} SparkHiddenTransportRdmaV4Identity;

uint64_t SparkHiddenTransportRdmaControlMonotonicNs(void);
uint64_t SparkHiddenTransportRdmaControlDeadlineNs(uint32_t timeout_milli);
SparkStatus SparkHiddenTransportRdmaControlSetNonblocking(int fd);
SparkStatus SparkHiddenTransportRdmaControlReadFullDeadline(
    int fd,
    void *buffer,
    uint64_t bytes,
    uint64_t deadline_ns);
SparkStatus SparkHiddenTransportRdmaControlWriteFullDeadline(
    int fd,
    const void *buffer,
    uint64_t bytes,
    uint64_t deadline_ns);
SparkStatus SparkHiddenTransportRdmaControlFenceSession(int fd);
SparkStatus SparkHiddenTransportRdmaV4ValidatePeerIdentity(
    const SparkHiddenTransportRdmaV4Identity *local_identity,
    const SparkHiddenTransportRdmaV4Identity *peer_identity);
SparkStatus SparkHiddenTransportRdmaV4ExchangeCompatibilityHello(
    int fd,
    uint64_t deadline_ns,
    const SparkHiddenTransportRdmaV4Identity *local_identity,
    SparkHiddenTransportRdmaV4Identity *peer_identity_out);

#ifdef __cplusplus
}
#endif
