#pragma once

#include <stdint.h>

#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION 4u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE 16u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS 4u
#define SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT 64u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT \
    (SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS * \
     SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT)
#define SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES 96u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE 0u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST 1u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER 0u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 1u

#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE 0u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING 1u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE 2u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING 3u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_TRANSFER_ACTIVE 4u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_BUILDING 5u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE 6u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY 7u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_CALLBACK_CLAIMED 8u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_RELEASE_PENDING 9u

typedef struct SparkTpDeviceCollectiveCreditBinding
{
    uint32_t step_index;
    uint32_t credit_index;
    /* Kernels use device buffers; transports may use distinct host
     * mirrors. */
    void *send_device;
    void *receive_device;
    void *send_transport;
    void *receive_transport;
} SparkTpDeviceCollectiveCreditBinding;

typedef struct SparkTpDeviceCollectiveCompletion
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkStatus status;
    uint32_t slot_index;
    uint32_t credit_index;
    uint64_t ordinal;
    uint64_t generation;
} SparkTpDeviceCollectiveCompletion;

typedef void (*SparkTpDeviceCollectiveCompletionFunction)(
    void *completion_context,
    const SparkTpDeviceCollectiveCompletion *completion);

typedef struct SparkTpDeviceCollectiveSubmission
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t slot_index;
    uint32_t active_sequence_count;
    uint64_t ordinal;
    const void *local_device;
    void *full_device;
    void *cuda_stream;
    SparkTpDeviceCollectiveCompletionFunction completion_function;
    void *completion_context;
} SparkTpDeviceCollectiveSubmission;

typedef void (*SparkTpDeviceCollectiveFailureObservedFunction)(
    void *hook_context,
    uint32_t credit_index,
    uint64_t observed_state_word);

typedef SparkStatus (*SparkTpDeviceCollectiveCombineBf16Function)(
    void *combine_context,
    void *destination_device,
    const void *source_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream);

typedef struct SparkTpDeviceCollectiveDebugHooks
{
    SparkTpDeviceCollectiveFailureObservedFunction failure_observed_function;
    void *hook_context;
} SparkTpDeviceCollectiveDebugHooks;

typedef struct SparkTpDeviceCollectiveConfig
{
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t operation_kind;
    uint32_t credit_count;
    uint32_t local_hidden_dimension;
    uint32_t max_active_sequence_count;
    uint32_t connect_timeout_milli;
    uint32_t operation_timeout_milli;
    uint32_t control_port_base;
    uint64_t collective_identifier;
    const char *transport_module_path;
    const char *local_host;
    const char *rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
    const SparkTpDeviceCollectiveCreditBinding *credit_bindings;
    uint32_t credit_binding_count;
    void *registration_cuda_stream;
    SparkTpDeviceCollectiveCombineBf16Function combine_bf16_function;
    void *combine_context;
    const SparkTpDeviceCollectiveDebugHooks *debug_hooks;
} SparkTpDeviceCollectiveConfig;

typedef struct SparkTpDeviceCollective
{
    uint32_t abi_version;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t step_count;
    uint32_t operation_kind;
    uint32_t credit_count;
    uint32_t local_hidden_dimension;
    uint32_t max_active_sequence_count;
    uint32_t operation_timeout_milli;
    uint32_t memory_mode;
    uint64_t collective_identifier;
    void *implementation;
} SparkTpDeviceCollective;

SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out);

SparkStatus SparkTpDeviceCollectiveProbeMemoryMode(
    const char *transport_module_path,
    uint32_t *memory_mode_out);

SparkStatus SparkTpDeviceCollectiveSubmitBf16(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission);

SparkStatus SparkTpDeviceCollectiveRequestFailure(
    SparkTpDeviceCollective *collective,
    SparkStatus failure_status);

SparkStatus SparkTpDeviceCollectiveRequestOperationFailure(
    SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    SparkStatus failure_status);

SparkStatus SparkTpDeviceCollectiveOperationPhase(
    const SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    uint32_t *phase_out,
    uint32_t *failure_requested_out);

SparkStatus SparkTpDeviceCollectiveExchangeBf16(
    SparkTpDeviceCollective *collective,
    const void *send_device,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream);

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
