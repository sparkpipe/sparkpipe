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

SparkStatus SparkTpCollectiveCreate(
    const SparkTpCollectiveConfig *config,
    SparkTpCollective *collective_out);

SparkStatus SparkTpCollectiveAllReduceSumF32(
    SparkTpCollective *collective,
    float *values,
    uint64_t element_count,
    float *scratch);

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
