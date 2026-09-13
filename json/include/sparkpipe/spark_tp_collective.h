#ifndef SPARKPIPE_SPARK_TP_COLLECTIVE_H
#define SPARKPIPE_SPARK_TP_COLLECTIVE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TP_COLLECTIVE_ABI_VERSION 2u
#define SPARK_TP_COLLECTIVE_MAX_STEPS 4u
#define SPARK_TP_COLLECTIVE_HOST_NAME_BYTES 64u

typedef struct SparkTpCollectivePeer
{
    char host_name[SPARK_TP_COLLECTIVE_HOST_NAME_BYTES];
    uint16_t port;
    uint16_t reserved0;
    uint32_t reserved1;
} SparkTpCollectivePeer;

typedef struct SparkTpCollectiveConfig
{
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint16_t listen_port;
    uint16_t reserved0;
    uint32_t connect_timeout_milli;
    uint32_t operation_timeout_milli;
    uint64_t collective_identifier;
    SparkTpCollectivePeer peers[SPARK_TP_COLLECTIVE_MAX_STEPS];
} SparkTpCollectiveConfig;

typedef struct SparkTpCollective
{
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t step_count;
    uint32_t operation_timeout_milli;
    uint32_t failed;
    uint64_t collective_identifier;
    uint64_t next_operation_sequence;
    int32_t listen_socket;
    int32_t step_sockets[SPARK_TP_COLLECTIVE_MAX_STEPS];
    uint32_t reserved0;
} SparkTpCollective;

/*
 * Creates one recursive-doubling TCP collective.
 *
 * Every participating rank must use the same nonzero collective_identifier,
 * tp_degree, timeout policy, and step topology. Each peer host_name must be a
 * numeric IPv4 address. The complete connection and reciprocal handshake must
 * finish within connect_timeout_milli. Partial groups fail closed.
 */
SparkStatus SparkTpCollectiveCreate(
    const SparkTpCollectiveConfig *config,
    SparkTpCollective *collective_out);

/*
 * Performs one in-place F32 sum all-reduce.
 *
 * All ranks must invoke operations serially, in the same order, with the same
 * element_count. The implementation exchanges and validates an operation
 * header before transferring payload bytes, so cross-group, call-order, and
 * shape mismatches fail instead of corrupting a later collective. The entire
 * operation must finish within operation_timeout_milli. Any transport or
 * protocol failure permanently fails and closes the collective.
 *
 * The reference wire format transports native F32 bytes and therefore assumes
 * homogeneous peers. For finite F32 inputs every rank receives the same
 * recursive-doubling reduction order and bitwise-identical result.
 */
SparkStatus SparkTpCollectiveAllReduceSumF32(
    SparkTpCollective *collective,
    float *values,
    uint64_t element_count,
    float *scratch);

/*
 * Performs one in-place BF16 sum all-reduce with F32 accumulate.
 *
 * Same exchange, validation, and failure contract as
 * SparkTpCollectiveAllReduceSumF32, but the staging and wire payload are BF16
 * (uint16_t lanes): half the host-staging copy volume and half the wire bytes
 * of the F32 variant, which is the format the decode all-reduce tensors
 * actually live in (audit NET-011). Each exchange step widens both partials
 * to F32, adds, and narrows back with round-to-nearest-even - one rounding
 * per doubling step, the standard BF16-all-reduce recipe. The wire operation
 * kind is exchanged in the operation header, so ranks that mix the F32 and
 * BF16 variants fail validation instead of misdecoding each other's payload.
 *
 * The reduction tree is the same fixed butterfly as the F32 variant and the
 * narrowing is deterministic, so every rank still receives a
 * bitwise-identical result. The remaining known cost versus a device-resident
 * collective is the host staging itself: values_bf16 and scratch_bf16 are
 * host buffers, so a device caller still pays one device-to-host and one
 * host-to-device copy (now at BF16 width). The device-direct tier is the
 * GPUDirect RDMA build of ring/transport/rdma.cu
 * (SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=1), which speaks the hidden-state
 * transport ABI rather than this collective's exchange protocol.
 */
SparkStatus SparkTpCollectiveAllReduceSumBf16(
    SparkTpCollective *collective,
    uint16_t *values_bf16,
    uint64_t element_count,
    uint16_t *scratch_bf16);

void SparkTpCollectiveDestroy(SparkTpCollective *collective);

#ifdef __cplusplus
}
#endif

#endif
