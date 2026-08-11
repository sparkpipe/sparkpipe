#pragma once

#include <stdint.h>

#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION 1u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE 16u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS 4u
#define SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES 64u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE 0u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST 1u

typedef struct SparkTpDeviceCollectiveConfig
{
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t local_hidden_dimension;
    uint32_t max_active_sequence_count;
    uint32_t connect_timeout_milli;
    uint32_t operation_timeout_milli;
    uint32_t control_port_base;
    uint64_t collective_identifier;
    const char *transport_module_path;
    const char *local_host;
    const char *rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
} SparkTpDeviceCollectiveConfig;

typedef struct SparkTpDeviceCollective
{
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t step_count;
    uint32_t operation_step_index;
    uint32_t prepared_receive_mask;
    uint32_t local_hidden_dimension;
    uint32_t max_active_sequence_count;
    uint32_t operation_timeout_milli;
    uint32_t memory_mode;
    uint32_t failed;
    uint64_t collective_identifier;
    uint64_t next_operation_sequence;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *send_sessions[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    SparkHiddenTransportSession *receive_sessions[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint32_t step_hidden_dimensions[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    char send_route_names[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    char receive_route_names[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    SparkHiddenTransportPacket prepared_receive_packets[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
} SparkTpDeviceCollective;

/* Opens one bidirectional RDMA session pair per recursive-doubling step. The
 * transport module selects device memory or CUDA-mapped host memory from its
 * advertised capabilities; TCP and simulation transports are rejected. */
SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out);

/* Exchanges one already-contiguous rank block with its butterfly partner.
 * send_device and receive_device are device pointers laid out as
 * active_sequence_count rows of hidden_dimension BF16 elements. The caller
 * owns the contiguous exchange buffers and chooses the current block. */
SparkStatus SparkTpDeviceCollectiveExchangeBf16(
    SparkTpDeviceCollective *collective,
    const void *send_device,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream);

/* Posts a receive for a current or future recursive-doubling step. Calling
 * this before the caller's device-to-host copy lets receive advertisement and
 * memory registration overlap that copy. The matching exchange consumes the
 * prepared receive; the operation remains ordered by step_index. */
SparkStatus SparkTpDeviceCollectivePrepareReceiveBf16(
    SparkTpDeviceCollective *collective,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream);

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective);

#ifdef __cplusplus
}
#endif
