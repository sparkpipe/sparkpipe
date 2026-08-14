#pragma once

#include <stdint.h>

#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION 12u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE 16u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS 4u
#define SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_PHASE_COUNT 6u
#define SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_INDEX 2u
#define SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_COUNT 3u
#define SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_PEER_COUNT 3u
#define SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT 4u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT 2u
#define SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT 64u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT \
    (SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS * \
     SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT)
#define SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES 96u
#define SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES 64u
#define SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION 2u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE 0u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST 1u
#define SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT 0u
#define SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL 1u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER 0u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 1u
#define SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING 0x00000001u
#define SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING \
    0x00000002u
#define SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL 0x00000004u
#define SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS \
    (SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING | \
     SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING | \
     SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL)
#define SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS 0x00000001u
#define SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS 0x00000002u
#define SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS \
    (SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS | \
     SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS)
#define SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION \
    0x00000001u
#define SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_KNOWN_FLAGS \
    SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION

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
    uint32_t flags;
    uint32_t reserved0;
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
    /* A stream-ordered callback may enqueue dependent work on cuda_stream.
     * It must not read full_device from the host before synchronizing. */
    uint32_t flags;
    uint32_t reserved0;
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

typedef void (*SparkTpDeviceCollectiveSubmissionClaimedFunction)(
    void *hook_context,
    uint32_t credit_index,
    uint64_t generation);

typedef SparkStatus (*SparkTpDeviceCollectiveCombineBf16Function)(
    void *combine_context,
    void *destination_device,
    const void *source_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream);

typedef SparkStatus (*SparkTpDeviceCollectiveCombineRelayBf16Function)(
    void *combine_context,
    void *destination_device,
    const void *source_device,
    void *relay_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream);

/* rank_devices is ordered by TP rank. The callback must reproduce the
 * recursive TP4 BF16 tree: round(0+1), round(2+3), then round(local+remote). */
typedef SparkStatus (*SparkTpDeviceCollectiveCombineTp4Bf16Function)(
    void *combine_context,
    void *destination_device,
    const void *const rank_devices[
        SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT],
    uint32_t tp_rank,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream);

typedef SparkStatus (*SparkTpDeviceCollectiveCombineU64MaxFunction)(
    void *combine_context,
    uint64_t *destination_device,
    const uint64_t *source_device,
    uint32_t element_count,
    void *cuda_stream);

typedef struct SparkTpDeviceCollectiveDebugHooks
{
    SparkTpDeviceCollectiveFailureObservedFunction failure_observed_function;
    SparkTpDeviceCollectiveSubmissionClaimedFunction
        submission_claimed_function;
    void *hook_context;
} SparkTpDeviceCollectiveDebugHooks;

typedef struct SparkTpDeviceCollectiveTopology
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t rank_count;
    uint32_t algorithm_mask;
    uint32_t rail_count;
    uint32_t direct_all_to_all_max_payload_bytes;
    uint32_t split_ring_min_payload_bytes;
    uint32_t step_rail_indices[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint32_t reserved0;
    char rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
        [SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
    char rail_rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
        [SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
} SparkTpDeviceCollectiveTopology;

#define SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES \
    ((uint32_t)sizeof(SparkTpDeviceCollectiveTopology))

typedef struct SparkTpDeviceCollectiveConfig
{
	uint32_t abi_version;
	uint32_t backend_kind;
	uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t operation_kind;
    uint32_t credit_count;
    uint32_t local_hidden_dimension;
    uint32_t max_active_sequence_count;
    uint32_t connect_timeout_milli;
    uint32_t operation_timeout_milli;
    uint32_t control_port_base;
    uint32_t algorithm_mask;
    uint32_t rail_count;
    uint32_t direct_all_to_all_max_payload_bytes;
    uint32_t split_ring_min_payload_bytes;
    uint32_t step_rail_indices[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t collective_identifier;
	const char *backend_module_path;
    const char *local_host;
    const char *rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
    const char *rail_rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
    const SparkTpDeviceCollectiveCreditBinding *credit_bindings;
    uint32_t credit_binding_count;
    void *registration_cuda_stream;
    SparkTpDeviceCollectiveCombineBf16Function combine_bf16_function;
    SparkTpDeviceCollectiveCombineRelayBf16Function
        combine_relay_bf16_function;
    SparkTpDeviceCollectiveCombineTp4Bf16Function
        combine_tp4_bf16_function;
    SparkTpDeviceCollectiveCombineU64MaxFunction combine_u64_max_function;
    void *combine_context;
    const SparkTpDeviceCollectiveDebugHooks *debug_hooks;
} SparkTpDeviceCollectiveConfig;

typedef struct SparkTpDeviceCollective
{
	uint32_t abi_version;
	uint32_t backend_kind;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t step_count;
    uint32_t operation_kind;
    uint32_t credit_count;
    uint32_t local_hidden_dimension;
    uint32_t max_active_sequence_count;
    uint32_t operation_timeout_milli;
    uint32_t memory_mode;
    uint32_t algorithm_mask;
    uint32_t rail_count;
    uint32_t direct_all_to_all_max_payload_bytes;
    uint32_t split_ring_min_payload_bytes;
    uint64_t collective_identifier;
    void *implementation;
} SparkTpDeviceCollective;

SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out);

SparkStatus SparkTpDeviceCollectiveProbeMemoryMode(
	uint32_t backend_kind,
	const char *backend_module_path,
	uint32_t *memory_mode_out);

SparkStatus SparkTpDeviceCollectiveCreditStepCount(
	uint32_t backend_kind,
	uint32_t tp_degree,
	uint32_t *step_count_out);

SparkStatus SparkTpDeviceCollectiveCreditBindingRouteCount(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t *route_count_out);

SparkStatus SparkTpDeviceCollectiveApplyTopology(
    const SparkTpDeviceCollectiveTopology *topology,
    SparkTpDeviceCollectiveConfig *config);

SparkStatus SparkTpDeviceCollectiveSliceTopology(
    const SparkTpDeviceCollectiveTopology *source,
    uint32_t first_rank,
    uint32_t rank_count,
    SparkTpDeviceCollectiveTopology *destination);

SparkStatus SparkTpDeviceCollectiveSubmitBf16(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission);

SparkStatus SparkTpDeviceCollectiveSubmitU64Max(
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
