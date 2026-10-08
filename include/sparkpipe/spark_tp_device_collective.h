#pragma once

#include <stdint.h>

#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION 16u
#define SPARK_TP_DEVICE_COLLECTIVE_WAIT_SPIN 0u
#define SPARK_TP_DEVICE_COLLECTIVE_WAIT_HARDWARE 1u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE 16u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS 16u
#define SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_PHASE_COUNT 30u
#define SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_INDEX 2u
#define SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_COUNT 3u
#define SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_MAX_PEERS \
    (SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE - 1u)
#define SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT \
    SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT 2u
#define SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT 64u
#define SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT \
    (SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS * \
     SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT)
#define SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES 96u
#define SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES 64u
#define SPARK_TP_DEVICE_COLLECTIVE_NONCE_BYTES 8u

#define SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION 3u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE 0u
#define SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST 1u
#define SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT 0u
#define SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL 1u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER 0u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 1u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 2u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL 3u
#define SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING 0x00000001u
#define SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING \
    0x00000002u
#define SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL 0x00000004u
#define SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE 0x00000008u
#define SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS \
    (SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING | \
     SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING | \
     SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL | \
     SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE)
#define SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS 0x00000001u
#define SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS 0x00000002u
#define SPARK_TP_DEVICE_COLLECTIVE_BINDING_DIRECT_ALL_TO_ALL 0x00000004u
#define SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS \
    (SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS | \
     SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS | \
     SPARK_TP_DEVICE_COLLECTIVE_BINDING_DIRECT_ALL_TO_ALL)
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

static inline uint64_t SparkTpDeviceCollectiveCreditBytes(uint32_t rows,uint32_t width)
{
	uint64_t elements = (uint64_t)rows * width;
	if ( elements == 0u || elements > (UINT64_MAX - SPARK_TP_DEVICE_COLLECTIVE_NONCE_BYTES) / SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT )
		return(0u);
	return((elements * SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT) + SPARK_TP_DEVICE_COLLECTIVE_NONCE_BYTES);
}

typedef struct SparkTpDeviceCollectiveCreditBinding
{
    uint32_t step_index;
    uint32_t credit_index;
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
    uint32_t flags;
    uint32_t logical_sequence_count;
    uint32_t row_elements;
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

typedef SparkStatus (*SparkTpDeviceCollectiveCombineF32SeedFunction)(
    void *combine_context,
    void *destination_f32_device,
    const void *source_a_bf16_device,
    const void *source_b_bf16_device,
    uint32_t element_count,
    void *cuda_stream);

typedef SparkStatus (*SparkTpDeviceCollectiveCombineF32AddFunction)(
    void *combine_context,
    void *destination_f32_device,
    const void *source_bf16_device,
    uint32_t element_count,
    void *cuda_stream);

typedef SparkStatus (*SparkTpDeviceCollectiveCombineFusedBf16Function)(
    void *combine_context,
    void *destination_device,
    const void *const *source_devices,
    uint32_t source_count,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream);

typedef SparkTpDeviceCollectiveCombineFusedBf16Function
    SparkTpDeviceCollectiveCombineGatherBf16Function;

typedef SparkStatus (*SparkTpDeviceCollectiveRoundF32Function)(
    void *combine_context,
    void *destination_bf16_device,
    const void *source_f32_device,
    uint32_t element_count,
    void *cuda_stream);

typedef SparkStatus (*SparkTpDeviceCollectiveCombineRelayBf16Function)(
    void *combine_context,
    void *destination_device,
    const void *source_device,
    void *relay_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream);

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
    uint32_t wait_mode;
    uint16_t session_ports[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
        [SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
    char rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
        [SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
    char rail_rank_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
        [SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
} SparkTpDeviceCollectiveTopology;

#define SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES \
    ((uint32_t)sizeof(SparkTpDeviceCollectiveTopology))

struct SparkWeightdClient;
struct SparkWeightdMeshTopology;
SparkStatus SparkTpDeviceCollectiveMeshTopology(uint32_t rank,uint32_t degree,
    struct SparkWeightdMeshTopology *topology);

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
    uint16_t session_ports[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
        [SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
    uint64_t collective_identifier;
    struct SparkWeightdClient *mesh_lane_client;
    uint32_t mesh_band_index;
    uint32_t wait_mode;
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
    SparkTpDeviceCollectiveCombineF32SeedFunction combine_f32_seed_function;
    SparkTpDeviceCollectiveCombineF32AddFunction combine_f32_add_function;
    SparkTpDeviceCollectiveRoundF32Function round_f32_function;
    SparkTpDeviceCollectiveCombineFusedBf16Function combine_fused_bf16_function;
    SparkTpDeviceCollectiveCombineGatherBf16Function
        combine_gather_bf16_function;
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

#define SPARK_TP_MESH_LAUNCHERS_ABI_VERSION 1u

typedef int (*SparkTpMeshCopyDownFunction)(void *stream,volatile void *destination,const void *source,uint64_t bytes,
    const volatile void *shipped,void *round_control,const volatile void *cancel,uint64_t timeout_ns);
typedef int (*SparkTpMeshPublishFunction)(void *stream,volatile void *entry,void *seq_cell,const void *epoch_cell,
    void *round_seq,uint64_t bytes,uint64_t slot_index,uint64_t slots_per_rank,volatile void *slot_tail,
    void *error_word,uint32_t peer_mask);
typedef int (*SparkTpMeshTreeFunction)(void *stream,void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    const volatile void *shipped,const volatile void *cancel,void *round_control,uint32_t rank,uint32_t degree,const void *local,
    void *output,void *scratch,uint64_t elements,uint32_t operation,uint32_t rounds,uint64_t timeout_ns);
typedef int (*SparkTpMeshHardwarePrepareFunction)(void *host,void **device);
#define SPARK_TP_MESH_STAGE_ALL 0u
#define SPARK_TP_MESH_STAGE_PUBLISH 1u
#define SPARK_TP_MESH_STAGE_COMPLETE 2u

typedef int (*SparkTpMeshHardwareFunction)(void *stream,void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    void *gate,void *round_control,uint32_t rank,uint32_t degree,const void *local,void *output,void *scratch,uint64_t elements,
    uint32_t operation,uint32_t rounds,uint32_t logical_rows,uint32_t slice_routes,void *staging,uint64_t timeout_ns,uint32_t stage);
typedef int (*SparkTpMeshSeqPadFunction)(void *stream,void *seq_cell);
typedef int (*SparkTpMeshGuardFunction)(void *stream,volatile void *error_word,void *output);
typedef int (*SparkTpMeshWaitFunction)(void *stream,volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
    uint64_t slots_per_rank,uint32_t rank,uint32_t degree,void *error_word,unsigned long long deadline_ns,void *diag_word,
    volatile void *cancel_cell,const void *cancel_expected,void *arrival_ring);
typedef int (*SparkTpMeshRoundLoopFunction)(void *stream,volatile void *band_base,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile void *entry,void *shipped_cell,volatile void *cancel_cell,void *round_control,uint32_t rank,uint32_t degree,
    const void *local_device,void *full_device,uint64_t bytes);

typedef struct SparkTpMeshLaunchers
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkTpMeshCopyDownFunction copy_down;
    SparkTpMeshPublishFunction publish;
    SparkTpMeshTreeFunction tree;
    SparkTpMeshHardwarePrepareFunction hardware_prepare;
    SparkTpMeshHardwareFunction hardware;
    SparkTpMeshSeqPadFunction seq_pad;
    SparkTpMeshGuardFunction guard;
    SparkTpMeshWaitFunction wait;
    SparkTpMeshRoundLoopFunction round_loop;
} SparkTpMeshLaunchers;

void SparkTpDeviceCollectiveRegisterLaunchers(const SparkTpMeshLaunchers *launchers);

SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out);

SparkStatus SparkTpDeviceCollectiveProbeMemoryMode(
    uint32_t backend_kind,
    const char *backend_module_path,
    uint32_t *memory_mode_out);

void SparkTpDeviceCollectiveDumpOperations(
    const SparkTpDeviceCollective *collective);

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

SparkStatus SparkTpDeviceCollectiveBegin(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind);

SparkStatus SparkTpDeviceCollectiveFinish(
    SparkTpDeviceCollective *collective);

uint32_t SparkTpDeviceCollectivePublished(
    const SparkTpDeviceCollective *collective);

SparkStatus SparkTpDeviceCollectiveEnqueue(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind);

SparkStatus SparkTpDeviceCollectiveEnqueueRounds(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t round_count);

uint64_t SparkTpDeviceCollectiveDeviceRoundsDone(
    SparkTpDeviceCollective *collective);

typedef struct SparkTpDeviceCollectiveHardwareTiming
{
    uint64_t source_wait_ns;
    uint64_t peer_wait_ns;
    uint64_t copy_ns;
    uint64_t combine_ns;
    uint64_t first_arrival_ns;
    uint64_t peer_arrival_ns[16];
} SparkTpDeviceCollectiveHardwareTiming;

SparkStatus SparkTpDeviceCollectiveHardwareStats(
    SparkTpDeviceCollective *collective,
    SparkTpDeviceCollectiveHardwareTiming *timing_out);

uint32_t SparkTpDeviceCollectiveStreamOrdered(const SparkTpDeviceCollective *collective);
uint32_t SparkTpDeviceCollectiveAllToAllSupported(const SparkTpDeviceCollective *collective);

SparkStatus SparkTpDeviceCollectiveVerifyDeferred(SparkTpDeviceCollective *collective,void *stream);

void SparkTpDeviceCollectiveRoundStats(
    SparkTpDeviceCollective *collective,
    uint64_t *count_out,
    uint64_t *total_ns_out,
    uint32_t reset);

SparkStatus SparkTpDeviceCollectiveArmCapture(
    SparkTpDeviceCollective *collective);

SparkStatus SparkTpDeviceCollectiveGraphCancelSeed(
    SparkTpDeviceCollective *collective,void *stream);

SparkStatus SparkTpDeviceCollectiveGraphPreLaunch(
    SparkTpDeviceCollective *collective,void *stream);

SparkStatus SparkTpDeviceCollectiveDisarmCapture(
    SparkTpDeviceCollective *collective);

SparkStatus SparkTpDeviceCollectiveGraphSettle(
    SparkTpDeviceCollective *collective,void *stream,uint64_t *error_out);

uint64_t SparkTpDeviceCollectiveGraphProgress(
    SparkTpDeviceCollective *collective,
    uint64_t *cell_out);

uint64_t SparkTpDeviceCollectiveGraphStuckDump(
    SparkTpDeviceCollective *collective);

SparkStatus SparkTpDeviceCollectiveGraphArrivalDump(
    SparkTpDeviceCollective *collective,uint32_t rank);

uint64_t SparkTpDeviceCollectiveGraphDiag(
    SparkTpDeviceCollective *collective);
uint64_t SparkTpDeviceCollectiveGraphError(
    SparkTpDeviceCollective *collective);

void SparkTpDeviceCollectiveClearGraphError(
    SparkTpDeviceCollective *collective);

void SparkTpDeviceCollectiveBroadcastCancel(
    SparkTpDeviceCollective *collective);

#define SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS 24u
#define SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ROUND_BITS 16u
#define SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK \
    ((1ull << SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS) - 1ull)

SparkStatus SparkTpDeviceCollectiveChainRetire(
    SparkTpDeviceCollective *collective);
SparkStatus SparkTpDeviceCollectiveChainKey(
    SparkTpDeviceCollective *collective,
    uint64_t request_id);
SparkStatus SparkTpDeviceCollectiveEndChain(
    SparkTpDeviceCollective *collective,
    void *cuda_stream);

uint64_t SparkTpDeviceCollectiveChainEpoch(
    const SparkTpDeviceCollective *collective);

uint64_t SparkTpDeviceCollectiveRoundIndex(
    SparkTpDeviceCollective *collective);

SparkStatus SparkTpDeviceCollectiveSubmitU64Max(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission);

SparkStatus SparkTpDeviceCollectiveOpWaitHandles(
    SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    void **flag_device,
    uint64_t *wait_value);

SparkStatus SparkTpDeviceCollectiveExchangeBf16(
    SparkTpDeviceCollective *collective,
    const void *send_device,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream);

SparkStatus SparkTpDeviceCollectiveAttachMesh(SparkTpDeviceCollective *collective);
SparkStatus SparkTpDeviceCollectiveAttach(SparkTpDeviceCollective *collective,void *mesh_region);

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
