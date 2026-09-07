#include "sparkpipe/spark_tp_device_collective.h"
#include "tp_device_collective_nccl.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <poll.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern cudaError_t cudaEventQuery(cudaEvent_t event);

#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_MASK 0xffull
#define SPARK_TP_DEVICE_COLLECTIVE_FAILURE_REQUESTED 0x100ull
#define SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_SHIFT 9u
#define SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_MASK 0xfe00ull
#define SPARK_TP_DEVICE_COLLECTIVE_GENERATION_SHIFT 16u

#define NONCE_BYTES SPARK_TP_DEVICE_COLLECTIVE_NONCE_BYTES
#define TREE_STAGES 4u
#define TREE_FIXED_SLOTS 8u
#define D2A_ROUTE_COUNT \
    SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_MAX_PEERS
#define D2A_ALL_ARRIVED ((1u << D2A_ROUTE_COUNT) - 1u)
#define D2A_CONTROL_PORT_OFFSET 256u
#define ACK_CONTROL_PORT_OFFSET 512u
#define D2A_ACK_CONTROL_PORT_OFFSET 768u
#define ROUTE_KIND_TREE 0u
#define ROUTE_KIND_D2A 1u
#define ROUTE_KIND_TREE_ACK 2u
#define ROUTE_KIND_D2A_ACK 3u
#define SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 2u

static uint32_t tree_peer(uint32_t rank,uint32_t bit)
{
    if (bit < 4u)
        return (rank & ~3u) + bit;
    return ((((rank >> 2u) + 1u + (bit - 4u)) & 3u) << 2u);
}

static uint32_t tree_send_mask(uint32_t rank,uint32_t stage)
{
    uint32_t j = rank & 3u;

    if (stage == 0u)
        return j == 1u ? 1u : (j == 3u ? 4u : 0u);
    if (stage == 1u)
        return j == 2u ? 1u : 0u;
    if (stage == 2u)
        return j == 0u ? 0x70u : 0u;
    return j == 0u ? 0xeu : 0u;
}

static uint32_t tree_recv_mask(uint32_t rank,uint32_t stage)
{
    uint32_t j = rank & 3u;

    if (stage == 0u)
        return j == 0u ? 2u : (j == 2u ? 8u : 0u);
    if (stage == 1u)
        return j == 0u ? 4u : 0u;
    if (stage == 2u)
        return j == 0u ? 0x70u : 0u;
    return j == 0u ? 0u : 1u;
}

static uint32_t tree_used(uint32_t rank)
{
    uint32_t used = 0u;
    uint32_t stage;

    for (stage = 0u; stage < TREE_STAGES; stage++)
        used |= tree_send_mask(rank,stage) | tree_recv_mask(rank,stage);
    return used;
}

static uint32_t tree_route_count(uint32_t rank)
{
    return __builtin_popcount(tree_used(rank));
}

static uint32_t tree_bit_route(uint32_t used,uint32_t bit)
{
    return __builtin_popcount(used & ((1u << bit) - 1u));
}

typedef struct SparkTpDeviceCollectiveOperation
{
    atomic_uint_fast64_t lifecycle;
    uint32_t slot_index;
    uint32_t credit_index;
    uint32_t active_sequence_count;
    uint32_t operation_kind;
    uint32_t stage;
    uint32_t arrived;
    uint32_t packed;
    uint32_t direct_all_to_all;
    uint32_t acked;
    uint64_t ordinal;
    uint64_t generation;
    uint64_t deadline_milli;
    uint64_t d2a_submit_micro;
    uint64_t d2a_posted_micro;
    uint64_t d2a_arrived_micro;
    const void *local_device;
    void *full_device;
    void *cuda_stream;
    void *continuation_cuda_stream;
    SparkTpDeviceCollectiveCompletionFunction completion_function;
    void *completion_context;
} SparkTpDeviceCollectiveOperation;

typedef struct SparkTpDeviceCollectiveImplementation
{
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *send_sessions[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    SparkHiddenTransportSession *receive_sessions[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint32_t step_hidden_dimensions[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    char send_route_names[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    char receive_route_names[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    SparkTpDeviceCollectiveCreditBinding bindings[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    SparkHiddenTransportSession *d2a_send_sessions[D2A_ROUTE_COUNT];
    SparkHiddenTransportSession *d2a_receive_sessions[D2A_ROUTE_COUNT];
    SparkHiddenTransportSession *ack_send_sessions[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    SparkHiddenTransportSession *ack_receive_sessions[
        SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    SparkHiddenTransportSession *d2a_ack_send_sessions[D2A_ROUTE_COUNT];
    SparkHiddenTransportSession *d2a_ack_receive_sessions[D2A_ROUTE_COUNT];
    char d2a_send_route_names[D2A_ROUTE_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    char d2a_receive_route_names[D2A_ROUTE_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    char ack_send_route_names[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    char ack_receive_route_names[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    char d2a_ack_send_route_names[D2A_ROUTE_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    char d2a_ack_receive_route_names[D2A_ROUTE_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
    SparkTpDeviceCollectiveCreditBinding d2a_bindings[D2A_ROUTE_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    void *d2a_fold_stage[D2A_ROUTE_COUNT];
    void *ack_region_host;
    void *ack_stage_host;
    uint64_t *ack_receive_slots;
    uint64_t *d2a_ack_receive_slots;
    uint64_t *ack_stage_slots;
    uint64_t *d2a_ack_stage_slots;
    SparkTpDeviceCollectiveCombineTp4Bf16Function combine_all_bf16_function;
    uint32_t d2a_route_count;
    uint32_t d2a_timing_enabled;
    SparkTpDeviceCollectiveOperation operations[
        SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    SparkTpDeviceCollectiveDebugHooks debug_hooks;
    SparkTpDeviceCollectiveCombineBf16Function combine_bf16_function;
    SparkTpDeviceCollectiveCombineU64MaxFunction combine_u64_max_function;
    void *combine_context;
    cudaEvent_t consumer_events[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    cudaEvent_t producer_events[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    cudaStream_t operation_streams[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    void *registration_cuda_stream;
    atomic_uint admission_open;
    atomic_uint shutdown_requested;
    atomic_uint_fast64_t shutdown_deadline_milli;
    atomic_int failure_status;
    pthread_t progress_thread;
    uint32_t progress_thread_started;
    uint32_t route_count;
    uint32_t binding_route_count;
    uint32_t fixed_slots_enabled;
    uint64_t nonce_offset;
    uint64_t fold_pitch;
    void *fold_stage_host;
    void *fold_stage[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    SparkTpDeviceCollective *collective;
} SparkTpDeviceCollectiveImplementation;

static uint32_t SparkTpDeviceCollectiveDegreeIsSupported(uint32_t tp_degree)
{
    return tp_degree == 1u || tp_degree == 2u || tp_degree == 4u ||
        tp_degree == 8u || tp_degree == 16u;
}

static uint32_t SparkTpDeviceCollectiveStepCount(uint32_t tp_degree)
{
    uint32_t step_count;

    step_count = 0u;
    while ((tp_degree >> (step_count + 1u)) != 0u)
    {
        step_count += 1u;
    }
    return step_count;
}

static uint64_t SparkTpDeviceCollectiveNowMilli(void)
{
    struct timespec current_time;

    if (clock_gettime(CLOCK_MONOTONIC,&current_time) != 0)
    {
        return UINT64_MAX;
    }
    return ((uint64_t)current_time.tv_sec * 1000u) +
        ((uint64_t)current_time.tv_nsec / 1000000u);
}

static uint64_t SparkTpDeviceCollectiveNowMicro(void)
{
    struct timespec current_time;

    if (clock_gettime(CLOCK_MONOTONIC,&current_time) != 0)
    {
        return UINT64_MAX;
    }
    return ((uint64_t)current_time.tv_sec * 1000000u) +
        ((uint64_t)current_time.tv_nsec / 1000u);
}

static uint32_t SparkTpDeviceCollectiveD2aEnabled(
    const SparkTpDeviceCollectiveConfig *config)
{
    uint32_t algorithm_mask;

    if (config == 0 ||
        config->tp_degree !=
            SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT ||
        config->direct_all_to_all_max_payload_bytes == 0u)
        return 0u;
    algorithm_mask = config->algorithm_mask == 0u ?
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING :
        config->algorithm_mask;
    return (algorithm_mask &
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL) != 0u ?
        1u : 0u;
}

static uint32_t SparkTpDeviceCollectiveD2aPeer(
    uint32_t rank,
    uint32_t route)
{
    return route < rank ? route : route + 1u;
}

static uint64_t SparkTpDeviceCollectiveStateWord(
    uint64_t generation,
    uint32_t phase,
    SparkStatus failure_status)
{
    return (generation << SPARK_TP_DEVICE_COLLECTIVE_GENERATION_SHIFT) |
        (failure_status != SPARK_STATUS_OK ?
            SPARK_TP_DEVICE_COLLECTIVE_FAILURE_REQUESTED : 0u) |
        ((uint64_t)failure_status <<
            SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_SHIFT) |
        (uint64_t)phase;
}

static uint32_t SparkTpDeviceCollectiveStatePhase(uint64_t state_word)
{
    return (uint32_t)(state_word & SPARK_TP_DEVICE_COLLECTIVE_PHASE_MASK);
}

static uint32_t SparkTpDeviceCollectiveStateHasFailure(uint64_t state_word)
{
    return (state_word & SPARK_TP_DEVICE_COLLECTIVE_FAILURE_REQUESTED) != 0u ?
        1u : 0u;
}

static SparkStatus SparkTpDeviceCollectiveStateFailureStatus(
    uint64_t state_word)
{
    return (SparkStatus)((state_word &
        SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_MASK) >>
        SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_SHIFT);
}

static uint64_t SparkTpDeviceCollectiveStateGeneration(uint64_t state_word)
{
    return state_word >> SPARK_TP_DEVICE_COLLECTIVE_GENERATION_SHIFT;
}

static int SparkTpDeviceCollectiveTextIsValid(const char *text)
{
    return text != 0 && text[0] != '\0';
}

static uint32_t SparkTpDeviceCollectiveTopologyHostIsValid(
    const char host[SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES])
{
    return host[0] != '\0' && memchr(host,'\0',
        SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES) != 0;
}

static SparkStatus SparkTpDeviceCollectiveValidateTopology(
    const SparkTpDeviceCollectiveTopology *topology)
{
    uint32_t rail;
    uint32_t rank;

    if (topology == 0 || topology->abi_version !=
            SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION ||
        topology->descriptor_bytes !=
            SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES ||
        !SparkTpDeviceCollectiveDegreeIsSupported(topology->rank_count) ||
        (topology->algorithm_mask &
            ~SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS) != 0u ||
        topology->rail_count >
            SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT ||
        topology->reserved0 != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (rank=0u; rank<topology->rank_count; rank++)
        if (SparkTpDeviceCollectiveTopologyHostIsValid(
                topology->rank_hosts[rank]) == 0u)
            return SPARK_STATUS_INVALID_ARGUMENT;
    for (rail=0u; rail<topology->rail_count; rail++)
        for (rank=0u; rank<topology->rank_count; rank++)
            if (SparkTpDeviceCollectiveTopologyHostIsValid(
                    topology->rail_rank_hosts[rail][rank]) == 0u)
                return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveApplyTopology(
    const SparkTpDeviceCollectiveTopology *topology,
    SparkTpDeviceCollectiveConfig *config)
{
    SparkStatus status;
    uint32_t rail;
    uint32_t rank;

    status = SparkTpDeviceCollectiveValidateTopology(topology);
    if (status != SPARK_STATUS_OK || config == 0 ||
        config->tp_degree != topology->rank_count ||
        config->tp_rank >= topology->rank_count)
        return SPARK_STATUS_INVALID_ARGUMENT;
    config->algorithm_mask = topology->algorithm_mask;
    config->rail_count = topology->rail_count;
    config->direct_all_to_all_max_payload_bytes =
        topology->direct_all_to_all_max_payload_bytes;
    config->split_ring_min_payload_bytes =
        topology->split_ring_min_payload_bytes;
    memcpy(config->step_rail_indices,topology->step_rail_indices,
        sizeof(config->step_rail_indices));
    memcpy(config->session_ports,topology->session_ports,
        sizeof(config->session_ports));
    config->local_host = topology->rank_hosts[config->tp_rank];
    for (rank=0u; rank<topology->rank_count; rank++)
        config->rank_hosts[rank] = topology->rank_hosts[rank];
    for (rail=0u; rail<topology->rail_count; rail++)
        for (rank=0u; rank<topology->rank_count; rank++)
            config->rail_rank_hosts[rail][rank] =
                topology->rail_rank_hosts[rail][rank];
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveSliceTopology(
    const SparkTpDeviceCollectiveTopology *source,
    uint32_t first_rank,
    uint32_t rank_count,
    SparkTpDeviceCollectiveTopology *destination)
{
    SparkStatus status;
    uint32_t rail;

    status = SparkTpDeviceCollectiveValidateTopology(source);
    if (status != SPARK_STATUS_OK || destination == 0 ||
        !SparkTpDeviceCollectiveDegreeIsSupported(rank_count) ||
        first_rank > source->rank_count ||
        rank_count > source->rank_count - first_rank)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(destination,0,sizeof(*destination));
    destination->abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
    destination->descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
    destination->rank_count = rank_count;
    destination->algorithm_mask = source->algorithm_mask;
    destination->rail_count = source->rail_count;
    destination->direct_all_to_all_max_payload_bytes =
        source->direct_all_to_all_max_payload_bytes;
    destination->split_ring_min_payload_bytes =
        source->split_ring_min_payload_bytes;
    memcpy(destination->step_rail_indices,source->step_rail_indices,
        sizeof(destination->step_rail_indices));
    memcpy(destination->session_ports,source->session_ports,
        sizeof(destination->session_ports));
    memcpy(destination->rank_hosts,source->rank_hosts[first_rank],
        (uint64_t)rank_count * sizeof(destination->rank_hosts[0]));
    for (rail=0u; rail<source->rail_count; rail++)
        memcpy(destination->rail_rank_hosts[rail],
            source->rail_rank_hosts[rail][first_rank],
            (uint64_t)rank_count *
                sizeof(destination->rail_rank_hosts[rail][0]));
    return SPARK_STATUS_OK;
}

static uint32_t SparkTpDeviceCollectiveAlgorithmMask(
    const SparkTpDeviceCollectiveConfig *config)
{
    return config->algorithm_mask == 0u ?
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING :
        config->algorithm_mask;
}

static uint32_t SparkTpDeviceCollectiveConfigRouteCount(
    const SparkTpDeviceCollectiveConfig *config)
{
    return tree_route_count(config->tp_rank);
}

static const char *SparkTpDeviceCollectiveRankHost(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t step_index,
    uint32_t rank_index)
{
    uint32_t rail_index;

    if (config == 0 || step_index >= SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS ||
        rank_index >= config->tp_degree)
        return 0;
    if (config->rail_count == 0u)
        return rank_index == config->tp_rank ?
            config->local_host : config->rank_hosts[rank_index];
    rail_index = config->step_rail_indices[step_index];
    if (rail_index >= config->rail_count)
        return 0;
    return config->rail_rank_hosts[rail_index][rank_index];
}

static SparkStatus SparkTpDeviceCollectiveValidateAlgorithms(
    const SparkTpDeviceCollectiveConfig *config)
{
    uint32_t route_index;

    if ((SparkTpDeviceCollectiveAlgorithmMask(config) &
            SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE) == 0u ||
        config->tp_degree != 16u ||
        config->operation_kind !=
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (config->rail_count != 0u)
        for (route_index=0u;
             route_index<tree_route_count(config->tp_rank);
             route_index++)
            if (config->step_rail_indices[route_index] >= config->rail_count)
                return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveValidateBindings(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t credit_count)
{
    uint8_t seen[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    uint8_t d2a_seen[D2A_ROUTE_COUNT]
        [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    uint32_t binding_index;
    uint32_t required_binding_count;
    uint32_t d2a_route_count;

    d2a_route_count = SparkTpDeviceCollectiveD2aEnabled(config) != 0u ?
        D2A_ROUTE_COUNT : 0u;
    required_binding_count =
        (tree_route_count(config->tp_rank) + d2a_route_count) * credit_count;
    if (config->credit_binding_count != required_binding_count ||
        (required_binding_count != 0u && config->credit_bindings == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(seen, 0, sizeof(seen));
    memset(d2a_seen, 0, sizeof(d2a_seen));
    for (binding_index = 0u;
         binding_index < config->credit_binding_count;
         ++binding_index)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding;

        binding = &config->credit_bindings[binding_index];
        if (binding->credit_index >= credit_count ||
            binding->send_device == 0 || binding->receive_device == 0 ||
            binding->send_transport == 0 ||
            binding->receive_transport == 0 ||
            (binding->flags &
                ~SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS) != 0u ||
            binding->reserved0 != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if ((binding->flags &
                SPARK_TP_DEVICE_COLLECTIVE_BINDING_DIRECT_ALL_TO_ALL) != 0u)
        {
            if (binding->step_index >= d2a_route_count ||
                d2a_seen[binding->step_index][binding->credit_index] != 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            d2a_seen[binding->step_index][binding->credit_index] = 1u;
            continue;
        }
        if (binding->step_index >= tree_route_count(config->tp_rank) ||
            seen[binding->step_index][binding->credit_index] != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        seen[binding->step_index][binding->credit_index] = 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveValidateConfig(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t *credit_count_out)
{
    uint32_t rank_index;
    uint32_t credit_count;

    if (config == 0 || credit_count_out == 0 ||
        config->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        config->backend_kind !=
            SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT ||
        !SparkTpDeviceCollectiveDegreeIsSupported(config->tp_degree) ||
        config->tp_rank >= config->tp_degree ||
        config->local_hidden_dimension == 0u ||
        config->max_active_sequence_count == 0u ||
        config->connect_timeout_milli == 0u ||
        config->operation_timeout_milli == 0u ||
        config->collective_identifier == 0u ||
        config->operation_kind !=
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 ||
        config->combine_bf16_function == 0 ||
        !SparkTpDeviceCollectiveTextIsValid(config->backend_module_path) ||
        !SparkTpDeviceCollectiveTextIsValid(config->local_host) ||
        (config->tp_degree > 1u && config->registration_cuda_stream == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    credit_count = config->credit_count == 0u ?
        SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT : config->credit_count;
    if (credit_count != TREE_FIXED_SLOTS)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (rank_index = 0u; rank_index < config->tp_degree; ++rank_index)
    {
        if (!SparkTpDeviceCollectiveTextIsValid(config->rank_hosts[rank_index]))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    *credit_count_out = credit_count;
    if (SparkTpDeviceCollectiveValidateAlgorithms(config) != SPARK_STATUS_OK)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SparkTpDeviceCollectiveValidateBindings(config,credit_count);
}

static SparkStatus SparkTpDeviceCollectiveBuildEndpoint(
    const SparkTpDeviceCollectiveConfig *config,
    const SparkTpDeviceCollective *collective,
    uint32_t step_index,
    uint32_t source_rank,
    uint32_t sink_rank,
    uint32_t hidden_dimension,
    uint32_t route_kind,
    char *route_name,
    SparkHiddenTransportEndpoint *endpoint)
{
    uint32_t port;
    int written;

    if (config == 0 || collective == 0 || route_name == 0 || endpoint == 0 ||
        source_rank >= config->tp_degree || sink_rank >= config->tp_degree ||
        source_rank == sink_rank || route_kind > ROUTE_KIND_D2A_ACK ||
        step_index >= SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    port = config->session_ports[source_rank][sink_rank];
    if (port == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    port += route_kind == ROUTE_KIND_D2A ? D2A_CONTROL_PORT_OFFSET :
        route_kind == ROUTE_KIND_TREE_ACK ? ACK_CONTROL_PORT_OFFSET :
        route_kind == ROUTE_KIND_D2A_ACK ? D2A_ACK_CONTROL_PORT_OFFSET : 0u;
    if (port > 65535u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (route_kind == ROUTE_KIND_D2A)
        written = snprintf(route_name,
            SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES,
            "tp-device-d2a.%016llx.%u.%u",
            (unsigned long long)config->collective_identifier,
            source_rank,sink_rank);
    else if (route_kind == ROUTE_KIND_TREE_ACK)
        written = snprintf(route_name,
            SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES,
            "tp-device-ack.%016llx.%u.%u.%u",
            (unsigned long long)config->collective_identifier,
            step_index,source_rank,sink_rank);
    else if (route_kind == ROUTE_KIND_D2A_ACK)
        written = snprintf(route_name,
            SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES,
            "tp-device-d2a-ack.%016llx.%u.%u",
            (unsigned long long)config->collective_identifier,
            source_rank,sink_rank);
    else
        written = snprintf(route_name,
            SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES,
            "tp-device.%016llx.%u.%u.%u",
            (unsigned long long)config->collective_identifier,
            step_index,source_rank,sink_rank);
    if (written < 0 ||
        (uint32_t)written >= SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (collective->memory_mode ==
        SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST)
    {
        SparkHiddenTransportInitializeSparkHostRdmaEndpoint(
            endpoint,hidden_dimension,config->max_active_sequence_count,0u,
            route_name);
    }
    else
    {
        SparkHiddenTransportInitializeSparkGpudirectRdmaEndpoint(
            endpoint,hidden_dimension,config->max_active_sequence_count,0u,
            route_name);
    }
    endpoint->capability_flags |=
        SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    endpoint->configuration_flags =
        SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION;
    endpoint->local_rank_index = config->tp_rank;
    endpoint->source_rank_index = source_rank;
    endpoint->sink_rank_index = sink_rank;
    endpoint->control_port_base = port;
    endpoint->source_host = SparkTpDeviceCollectiveRankHost(
        config,step_index,source_rank);
    endpoint->sink_host = SparkTpDeviceCollectiveRankHost(
        config,step_index,sink_rank);
    endpoint->route_identifier = config->collective_identifier;
    if (endpoint->source_host == 0 || endpoint->sink_host == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SparkHiddenTransportValidateEndpoint(endpoint);
}

static void SparkTpDeviceCollectiveCloseSessions(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t step_index;

    if (implementation == 0)
    {
        return;
    }
    for (step_index = 0u;
         step_index < SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS;
         ++step_index)
    {
        if (implementation->send_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->send_sessions[step_index]);
            implementation->send_sessions[step_index] = 0;
        }
        if (implementation->receive_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->receive_sessions[step_index]);
            implementation->receive_sessions[step_index] = 0;
        }
    }
    for (step_index = 0u;
         step_index < SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS;
         ++step_index)
    {
        if (implementation->ack_send_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->ack_send_sessions[step_index]);
            implementation->ack_send_sessions[step_index] = 0;
        }
        if (implementation->ack_receive_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->ack_receive_sessions[step_index]);
            implementation->ack_receive_sessions[step_index] = 0;
        }
    }
    for (step_index = 0u; step_index < D2A_ROUTE_COUNT; ++step_index)
    {
        if (implementation->d2a_send_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->d2a_send_sessions[step_index]);
            implementation->d2a_send_sessions[step_index] = 0;
        }
        if (implementation->d2a_receive_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->d2a_receive_sessions[step_index]);
            implementation->d2a_receive_sessions[step_index] = 0;
        }
        if (implementation->d2a_ack_send_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->d2a_ack_send_sessions[step_index]);
            implementation->d2a_ack_send_sessions[step_index] = 0;
        }
        if (implementation->d2a_ack_receive_sessions[step_index] != 0)
        {
            SparkHiddenTransportClose(
                implementation->d2a_ack_receive_sessions[step_index]);
            implementation->d2a_ack_receive_sessions[step_index] = 0;
        }
    }
}

static void SparkTpDeviceCollectiveDestroyEvents(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t credit_index;

    if (implementation == 0)
    {
        return;
    }
    for (credit_index = 0u;
         credit_index < implementation->collective->credit_count;
         ++credit_index)
    {
        if (implementation->consumer_events[credit_index] != 0)
        {
            (void)cudaEventDestroy(
                implementation->consumer_events[credit_index]);
            implementation->consumer_events[credit_index] = 0;
        }
        if (implementation->producer_events[credit_index] != 0)
        {
            (void)cudaEventDestroy(
                implementation->producer_events[credit_index]);
            implementation->producer_events[credit_index] = 0;
        }
        if (implementation->operation_streams[credit_index] != 0)
        {
            (void)cudaStreamDestroy(
                implementation->operation_streams[credit_index]);
            implementation->operation_streams[credit_index] = 0;
        }
    }
}

static SparkStatus SparkTpDeviceCollectiveOpenSession(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t route,
    uint32_t source_rank,
    uint32_t sink_rank,
    uint32_t route_kind,
    char *route_name,
    SparkHiddenTransportSession **session)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkStatus status;

    memset(&endpoint, 0, sizeof(endpoint));
    status = SparkTpDeviceCollectiveBuildEndpoint(
        config,implementation->collective,source_rank ^ sink_rank,source_rank,
        sink_rank,
        route_kind != ROUTE_KIND_TREE ? config->local_hidden_dimension :
            implementation->step_hidden_dimensions[route],
        route_kind,route_name,&endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkHiddenTransportOpen(&endpoint,
        &implementation->transport_library.transport_interface,
        (implementation->collective->memory_mode ==
            SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
            SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS :
            SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS) |
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS,
        session);
}

static void SparkTpDeviceCollectiveTreeRoutePeers(
    uint32_t rank,
    uint32_t used,
    uint32_t *peers)
{
    uint32_t bit;
    uint32_t route = 0u;

    for (bit = 0u; bit < 7u; bit++)
        if ((used >> bit & 1u) != 0u)
            peers[route++] = tree_peer(rank,bit);
}

typedef struct TreeSessionOpener
{
    SparkTpDeviceCollectiveImplementation *implementation;
    const SparkTpDeviceCollectiveConfig *config;
    uint32_t route;
    uint32_t peer;
    uint32_t sender;
    uint32_t route_kind;
    SparkStatus status;
} TreeSessionOpener;

static void *SparkTpDeviceCollectiveOpenTreeSession(void *context)
{
    TreeSessionOpener *opener = (TreeSessionOpener *)context;
    SparkTpDeviceCollectiveImplementation *implementation =
        opener->implementation;
    SparkHiddenTransportSession **session;
    char *route_name;

    if (opener->sender != 0u)
    {
        session = opener->route_kind == ROUTE_KIND_TREE ?
            &implementation->send_sessions[opener->route] :
            opener->route_kind == ROUTE_KIND_D2A ?
            &implementation->d2a_send_sessions[opener->route] :
            opener->route_kind == ROUTE_KIND_TREE_ACK ?
            &implementation->ack_send_sessions[opener->route] :
            &implementation->d2a_ack_send_sessions[opener->route];
        route_name = opener->route_kind == ROUTE_KIND_TREE ?
            implementation->send_route_names[opener->route] :
            opener->route_kind == ROUTE_KIND_D2A ?
            implementation->d2a_send_route_names[opener->route] :
            opener->route_kind == ROUTE_KIND_TREE_ACK ?
            implementation->ack_send_route_names[opener->route] :
            implementation->d2a_ack_send_route_names[opener->route];
        opener->status = SparkTpDeviceCollectiveOpenSession(
            implementation,opener->config,opener->route,
            implementation->collective->tp_rank,opener->peer,
            opener->route_kind,route_name,session);
    }
    else
    {
        session = opener->route_kind == ROUTE_KIND_TREE ?
            &implementation->receive_sessions[opener->route] :
            opener->route_kind == ROUTE_KIND_D2A ?
            &implementation->d2a_receive_sessions[opener->route] :
            opener->route_kind == ROUTE_KIND_TREE_ACK ?
            &implementation->ack_receive_sessions[opener->route] :
            &implementation->d2a_ack_receive_sessions[opener->route];
        route_name = opener->route_kind == ROUTE_KIND_TREE ?
            implementation->receive_route_names[opener->route] :
            opener->route_kind == ROUTE_KIND_D2A ?
            implementation->d2a_receive_route_names[opener->route] :
            opener->route_kind == ROUTE_KIND_TREE_ACK ?
            implementation->ack_receive_route_names[opener->route] :
            implementation->d2a_ack_receive_route_names[opener->route];
        opener->status = SparkTpDeviceCollectiveOpenSession(
            implementation,opener->config,opener->route,opener->peer,
            implementation->collective->tp_rank,opener->route_kind,
            route_name,session);
    }
    return 0;
}

static SparkStatus SparkTpDeviceCollectiveStartSessionOpener(
    TreeSessionOpener *openers,
    pthread_t *threads,
    uint32_t *index,
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t route,
    uint32_t peer,
    uint32_t sender,
    uint32_t route_kind)
{
    TreeSessionOpener *opener = &openers[*index];

    opener->implementation = implementation;
    opener->config = config;
    opener->route = route;
    opener->peer = peer;
    opener->sender = sender;
    opener->route_kind = route_kind;
    opener->status = SPARK_STATUS_OK;
    if (pthread_create(&threads[*index],0,
            SparkTpDeviceCollectiveOpenTreeSession,opener) != 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    *index += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveOpenTreeSessions(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveConfig *config)
{
    TreeSessionOpener openers[28];
    pthread_t threads[28];
    uint32_t peers[7];
    uint32_t rank = config->tp_rank;
    uint32_t used = tree_used(rank);
    uint32_t route_count = tree_route_count(rank);
    uint32_t route;
    uint32_t index = 0u;
    SparkStatus status;

    SparkTpDeviceCollectiveTreeRoutePeers(rank,used,peers);
    memset(openers,0,sizeof(openers));
    for (route = 0u; route < route_count; route++)
    {
        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peers[route],0u,
            ROUTE_KIND_TREE);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peers[route],1u,
            ROUTE_KIND_TREE);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peers[route],0u,
            ROUTE_KIND_TREE_ACK);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peers[route],1u,
            ROUTE_KIND_TREE_ACK);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    status = SPARK_STATUS_OK;
    for (route = 0u; route < index; route++)
    {
        pthread_join(threads[route],0);
        if (status == SPARK_STATUS_OK)
            status = openers[route].status;
    }
    return status;
}

static SparkStatus SparkTpDeviceCollectiveOpenD2aSessions(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveConfig *config)
{
    TreeSessionOpener openers[D2A_ROUTE_COUNT * 4u];
    pthread_t threads[D2A_ROUTE_COUNT * 4u];
    uint32_t rank = config->tp_rank;
    uint32_t route;
    uint32_t index = 0u;
    SparkStatus status;

    memset(openers,0,sizeof(openers));
    for (route = 0u; route < implementation->d2a_route_count; route++)
    {
        uint32_t peer = SparkTpDeviceCollectiveD2aPeer(rank,route);

        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peer,0u,ROUTE_KIND_D2A);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peer,1u,ROUTE_KIND_D2A);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peer,0u,ROUTE_KIND_D2A_ACK);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkTpDeviceCollectiveStartSessionOpener(openers,threads,
            &index,implementation,config,route,peer,1u,ROUTE_KIND_D2A_ACK);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    status = SPARK_STATUS_OK;
    for (route = 0u; route < index; route++)
    {
        pthread_join(threads[route],0);
        if (status == SPARK_STATUS_OK)
            status = openers[route].status;
    }
    if (status != SPARK_STATUS_OK)
        fprintf(stderr,"D2A-OPEN-FAIL rank=%u status=%u\n",
            rank,(uint32_t)status);
    return status;
}

static SparkStatus SparkTpDeviceCollectiveBuildPacket(
    const void *device_pointer,
    uint32_t rows,
    uint32_t hidden_dimension,
    uint64_t ordinal,
    uint32_t step_index,
    void *cuda_stream,
    SparkHiddenTransportPacket *packet)
{
    uint64_t bytes_per_sequence;

    if (device_pointer == 0 || rows == 0u || hidden_dimension == 0u ||
        ordinal == UINT64_MAX || cuda_stream == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    bytes_per_sequence = (uint64_t)hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    if (bytes_per_sequence > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memset(packet, 0, sizeof(*packet));
    packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
    packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    packet->active_sequence_count = rows;
    packet->hidden_dimension = hidden_dimension;
    packet->bytes_per_sequence = (uint32_t)bytes_per_sequence;
    packet->sequence_id = ordinal + 1u;
    packet->token_index = step_index;
    packet->hidden_bf16 = device_pointer;
    packet->cuda_stream = cuda_stream;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveCudaStatus(cudaError_t cuda_status)
{
    if (cuda_status == cudaSuccess)
    {
        return SPARK_STATUS_OK;
    }
    if (cuda_status == cudaErrorNotReady)
    {
        return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_DRIVER_LOAD_ERROR;
}

static SparkStatus SparkTpDeviceCollectiveCopyRows(
    void *destination,
    uint64_t destination_pitch,
    const void *source,
    uint64_t source_pitch,
    uint64_t width,
    uint32_t rows,
    enum cudaMemcpyKind copy_kind,
    void *cuda_stream)
{
    if (destination == 0 || source == 0 || destination_pitch < width ||
        source_pitch < width || width == 0u || rows == 0u ||
        cuda_stream == 0 || destination_pitch > SIZE_MAX ||
        source_pitch > SIZE_MAX || width > SIZE_MAX)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (destination == source && destination_pitch == source_pitch)
        return SPARK_STATUS_OK;
    return SparkTpDeviceCollectiveCudaStatus(cudaMemcpy2DAsync(
        destination,(size_t)destination_pitch,source,(size_t)source_pitch,
        (size_t)width,(size_t)rows,copy_kind,
        (cudaStream_t)cuda_stream));
}

static SparkStatus SparkTpDeviceCollectiveStageSend(
    const SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveCreditBinding *binding,
    uint64_t pitch,
    uint64_t width,
    uint32_t rows,
    void *cuda_stream)
{
    enum cudaMemcpyKind copy_kind;

    if ((binding->flags &
            SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS) != 0u)
        return SPARK_STATUS_OK;
    if (binding->send_device == binding->send_transport)
    {
        return SPARK_STATUS_OK;
    }
    copy_kind = collective->memory_mode ==
        SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
        cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    return SparkTpDeviceCollectiveCopyRows(binding->send_transport,pitch,
        binding->send_device,pitch,width,rows,copy_kind,cuda_stream);
}

static SparkStatus SparkTpDeviceCollectivePackSendRows(
    const SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveCreditBinding *binding,
    const void *source,
    uint64_t source_pitch,
    uint64_t packed_pitch,
    uint64_t width,
    uint32_t rows,
    void *cuda_stream)
{
    SparkStatus status;

    if ((binding->flags &
            SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS) != 0u)
        return SparkTpDeviceCollectiveCopyRows(binding->send_transport,
            packed_pitch,source,source_pitch,width,rows,
            cudaMemcpyDeviceToHost,cuda_stream);
    status = SparkTpDeviceCollectiveCopyRows(binding->send_device,
        packed_pitch,source,source_pitch,width,rows,cudaMemcpyDeviceToDevice,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkTpDeviceCollectiveStageSend(collective,binding,
        packed_pitch,width,rows,cuda_stream);
}

static uint64_t SparkTpDeviceCollectiveOperationBytes(
    const SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveOperation *operation)
{
    if (operation->operation_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64)
        return (uint64_t)operation->active_sequence_count * sizeof(uint64_t);
    return (uint64_t)operation->active_sequence_count *
        collective->local_hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
}

static SparkStatus SparkTpDeviceCollectiveTreePack(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint32_t send_mask)
{
    const SparkTpDeviceCollective *collective = implementation->collective;
    uint64_t local_bytes =
        SparkTpDeviceCollectiveOperationBytes(collective,operation);
    uint32_t used = tree_used(collective->tp_rank);
    uint32_t bit;
    SparkStatus status;

    for (bit = 0u; bit < 7u; bit++)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding;

        if ((send_mask >> bit & 1u) == 0u)
            continue;
        binding = &implementation->bindings[
            tree_bit_route(used,bit)][operation->credit_index];
        status = SparkTpDeviceCollectivePackSendRows(
            collective,binding,operation->full_device,local_bytes,local_bytes,
            local_bytes,1u,
            operation->cuda_stream);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveMarkOperationFailure(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint64_t required_generation,
    SparkStatus status);

static void SparkTpDeviceCollectiveLatchFailure(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkStatus status);

static uint32_t SparkTpDeviceCollectiveTransitionPhase(
    SparkTpDeviceCollectiveOperation *operation,
    uint32_t expected_phase,
    uint32_t desired_phase);

static void SparkTpDeviceCollectiveTreeSend(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint32_t route)
{
    const SparkTpDeviceCollectiveCreditBinding *binding =
        &implementation->bindings[route][operation->credit_index];
    uint64_t payload_bytes = SparkTpDeviceCollectiveOperationBytes(
        implementation->collective,operation);
    uint64_t nonce_at = payload_bytes;
    SparkStatus status;
    *(volatile uint64_t *)((uint8_t *)binding->send_transport +
        nonce_at) = operation->ordinal + 1u;
    status = SparkHiddenTransportSendFixed(
        implementation->send_sessions[route],
        binding->send_transport,
        nonce_at + NONCE_BYTES,
        (uint32_t)((operation->ordinal << 8u) | route));
    if (status != SPARK_STATUS_OK)
        SparkTpDeviceCollectiveLatchFailure(implementation,status);
}

static uint32_t SparkTpDeviceCollectiveAckGateOpen(
    const uint64_t *ack_slots,
    uint32_t route,
    uint32_t credit_count,
    const SparkTpDeviceCollectiveOperation *operation)
{
    if (operation->ordinal < TREE_FIXED_SLOTS)
        return 1u;
    return *(volatile uint64_t *)(ack_slots +
        (uint64_t)route * credit_count + operation->credit_index) >=
        operation->ordinal - TREE_FIXED_SLOTS + 1u ? 1u : 0u;
}

static SparkStatus SparkTpDeviceCollectivePostAck(
    SparkHiddenTransportSession *session,
    const uint64_t *staging,
    uint64_t ordinal,
    uint32_t route)
{
    SparkHiddenTransportCompletion completion;
    SparkStatus status;

    status = SparkHiddenTransportSendFixed(session,staging,sizeof(uint64_t),
        (uint32_t)((ordinal << 8u) | route));
    if (status == SPARK_STATUS_BUSY)
    {
        (void)SparkHiddenTransportPoll(session,&completion);
        status = SparkHiddenTransportSendFixed(session,staging,
            sizeof(uint64_t),(uint32_t)((ordinal << 8u) | route));
    }
    return status;
}

static SparkStatus SparkTpDeviceCollectiveSendAcks(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    SparkTpDeviceCollective *collective = implementation->collective;
    uint32_t credit_count = collective->credit_count;
    uint32_t route;
    SparkStatus status = SPARK_STATUS_OK;

    if (operation->direct_all_to_all != 0u)
    {
        for (route = 0u; route < implementation->d2a_route_count; route++)
        {
            uint64_t *staging = implementation->d2a_ack_stage_slots +
                (uint64_t)route * credit_count + operation->credit_index;

            *(volatile uint64_t *)staging = operation->ordinal + 1u;
            status = SparkTpDeviceCollectivePostAck(
                implementation->d2a_ack_send_sessions[route],
                staging,operation->ordinal,route);
            if (status != SPARK_STATUS_OK)
                break;
        }
    }
    else
    {
        uint32_t rank = collective->tp_rank;
        uint32_t used = tree_used(rank);
        uint32_t recv_union = 0u;
        uint32_t stage;
        uint32_t bit;

        for (stage = 0u; stage < TREE_STAGES; stage++)
            recv_union |= tree_recv_mask(rank,stage);
        for (bit = 0u; bit < 7u; bit++)
        {
            uint64_t *staging;

            if ((recv_union >> bit & 1u) == 0u)
                continue;
            route = tree_bit_route(used,bit);
            staging = implementation->ack_stage_slots +
                (uint64_t)route * credit_count + operation->credit_index;
            *(volatile uint64_t *)staging = operation->ordinal + 1u;
            status = SparkTpDeviceCollectivePostAck(
                implementation->ack_send_sessions[route],
                staging,operation->ordinal,route);
            if (status != SPARK_STATUS_OK)
                break;
        }
    }
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
    {
        fprintf(stderr,"ACK-SEND-FAIL rank=%u ord=%llu status=%u\n",
            collective->tp_rank,(unsigned long long)operation->ordinal,
            (uint32_t)status);
        SparkTpDeviceCollectiveLatchFailure(implementation,status);
        return SPARK_STATUS_OK;
    }
    return status;
}

static void SparkTpDeviceCollectiveTreeOperation(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    SparkTpDeviceCollective *collective = implementation->collective;
    uint32_t rank = collective->tp_rank;
    uint32_t used = tree_used(rank);
    uint64_t local_bytes =
        SparkTpDeviceCollectiveOperationBytes(collective,operation);
    uint32_t stage = operation->stage;
    uint32_t recv_bits = tree_recv_mask(rank,stage);
    uint32_t send_mask = stage == 0u ?
        (tree_send_mask(rank,0u) | tree_send_mask(rank,1u)) :
        (stage + 1u < TREE_STAGES ? tree_send_mask(rank,stage + 1u) : 0u);
    uint32_t bit;

    for (bit = 0u; bit < 7u; bit++)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding;
        uint32_t mask = 1u << bit;

        if ((recv_bits & mask) == 0u || (operation->arrived & mask) != 0u)
            continue;
        binding = &implementation->bindings[
            tree_bit_route(used,bit)][operation->credit_index];
        {
            uint64_t nonce_at = local_bytes;
            if (*(volatile uint64_t *)
                    ((uint8_t *)binding->receive_transport + nonce_at) !=
                operation->ordinal + 1u)
                continue;
        }
        operation->arrived |= mask;
        if (operation->operation_kind ==
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64)
            memcpy(implementation->fold_stage[tree_bit_route(used,bit)] +
                (uint64_t)operation->credit_index * implementation->fold_pitch,
                binding->receive_transport,(size_t)local_bytes);
        if (stage + 1u == TREE_STAGES)
        {
            const void *down_src = binding->receive_device;
            if (operation->operation_kind ==
                SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64)
                down_src = implementation->fold_stage[
                    tree_bit_route(used,bit)] +
                    (uint64_t)operation->credit_index *
                        implementation->fold_pitch;
            SparkStatus status = SparkTpDeviceCollectiveCopyRows(
                operation->full_device,local_bytes,
                down_src,
                local_bytes,local_bytes,1u,
                cudaMemcpyDeviceToDevice,operation->cuda_stream);
            if (status != SPARK_STATUS_OK)
            {
                SparkTpDeviceCollectiveMarkOperationFailure(implementation,
                    operation,operation->generation,status);
                return;
            }
        }
        else
        {
            SparkStatus status;
            if (operation->operation_kind ==
                SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64)
            {
                const void *u64_src = binding->receive_device;
                if (operation->operation_kind ==
                    SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64)
                    u64_src = implementation->fold_stage[
                        tree_bit_route(used,bit)] +
                        (uint64_t)operation->credit_index *
                            implementation->fold_pitch;
                status = implementation->combine_u64_max_function(
                    implementation->combine_context,
                    (uint64_t *)operation->full_device,
                    (const uint64_t *)u64_src,
                    operation->active_sequence_count,operation->cuda_stream);
            }
            else
                status = implementation->combine_bf16_function(
                    implementation->combine_context,operation->full_device,
                    binding->receive_device,operation->active_sequence_count,
                    collective->local_hidden_dimension,operation->cuda_stream);
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr,"TREE-FOLD-FAIL rank=%u ord=%llu stage=%u status=%u cuda=%s\n",
                    collective->tp_rank,
                    (unsigned long long)operation->ordinal,stage,
                    (uint32_t)status,cudaGetErrorString(cudaGetLastError()));
                SparkTpDeviceCollectiveMarkOperationFailure(implementation,
                    operation,operation->generation,status);
                return;
            }
        }
    }
    if (operation->arrived != recv_bits)
        return;
    if (operation->packed == 0u)
    {
        SparkStatus status = SparkTpDeviceCollectiveTreePack(
            implementation,operation,send_mask);
        if (status == SPARK_STATUS_OK)
            status = SparkTpDeviceCollectiveCudaStatus(cudaEventRecord(
                implementation->consumer_events[operation->credit_index],
                (cudaStream_t)operation->cuda_stream));
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"TREE-PACK-FAIL rank=%u ord=%llu stage=%u status=%u cuda=%s\n",
                collective->tp_rank,
                (unsigned long long)operation->ordinal,stage,
                (uint32_t)status,cudaGetErrorString(cudaGetLastError()));
            SparkTpDeviceCollectiveMarkOperationFailure(implementation,
                operation,operation->generation,status);
            return;
        }
        operation->packed = 1u;
        return;
    }
    if (cudaEventQuery(
            implementation->consumer_events[operation->credit_index]) ==
        cudaErrorNotReady)
        return;
    for (bit = 0u; bit < 7u; bit++)
        if ((send_mask >> bit & 1u) != 0u &&
            SparkTpDeviceCollectiveAckGateOpen(
                implementation->ack_receive_slots,
                tree_bit_route(used,bit),
                implementation->collective->credit_count,operation) == 0u)
            return;
    for (bit = 0u; bit < 7u; bit++)
        if ((send_mask >> bit & 1u) != 0u)
            SparkTpDeviceCollectiveTreeSend(implementation,operation,
                tree_bit_route(used,bit));
    if (stage + 1u == TREE_STAGES ||
        implementation->failure_status != SPARK_STATUS_OK)
    {
        operation->stage = TREE_STAGES;
        (void)SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
        return;
    }
    operation->stage = stage + 1u;
    operation->arrived = 0u;
    operation->packed = 0u;
}

static void SparkTpDeviceCollectiveD2aOperation(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    SparkTpDeviceCollective *collective = implementation->collective;
    uint64_t local_bytes =
        SparkTpDeviceCollectiveOperationBytes(collective,operation);
    uint64_t nonce_at = local_bytes;
    uint32_t route;
    SparkStatus status;

    if (operation->stage == 0u)
    {
        uint32_t acks_open = 1u;

        if (cudaEventQuery(
                implementation->consumer_events[operation->credit_index]) ==
            cudaErrorNotReady)
            return;
        for (route = 0u; route < implementation->d2a_route_count; route++)
            if (SparkTpDeviceCollectiveAckGateOpen(
                    implementation->d2a_ack_receive_slots,route,
                    collective->credit_count,operation) == 0u)
            {
                acks_open = 0u;
                break;
            }
        if (acks_open != 0u)
        {
            for (route = 1u; route < implementation->d2a_route_count;
                 route++)
                memcpy(implementation->d2a_bindings[route]
                        [operation->credit_index].send_transport,
                    implementation->d2a_bindings[0u]
                        [operation->credit_index].send_transport,
                    (size_t)local_bytes);
            for (route = 0u; route < implementation->d2a_route_count;
                 route++)
            {
                const SparkTpDeviceCollectiveCreditBinding *binding =
                    &implementation->d2a_bindings[route]
                        [operation->credit_index];

                *(volatile uint64_t *)((uint8_t *)binding->send_transport +
                    nonce_at) = operation->ordinal + 1u;
                status = SparkHiddenTransportSendFixed(
                    implementation->d2a_send_sessions[route],
                    binding->send_transport,
                    nonce_at + NONCE_BYTES,
                    (uint32_t)((operation->ordinal << 8u) | route));
                if (status != SPARK_STATUS_OK)
                {
                    fprintf(stderr,"D2A-SEND-FAIL rank=%u ord=%llu route=%u status=%u\n",
                        collective->tp_rank,
                        (unsigned long long)operation->ordinal,route,
                        (uint32_t)status);
                    SparkTpDeviceCollectiveLatchFailure(implementation,
                        status);
                    return;
                }
            }
            operation->stage = 1u;
            if (implementation->d2a_timing_enabled != 0u)
                operation->d2a_posted_micro = SparkTpDeviceCollectiveNowMicro();
        }
    }
    for (route = 0u; route < implementation->d2a_route_count; route++)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding =
            &implementation->d2a_bindings[route][operation->credit_index];

        if ((operation->arrived >> route & 1u) != 0u)
            continue;
        if (*(volatile uint64_t *)
                ((uint8_t *)binding->receive_transport + nonce_at) !=
            operation->ordinal + 1u)
            continue;
        operation->arrived |= 1u << route;
    }
    if (operation->arrived != D2A_ALL_ARRIVED)
        return;
    if (implementation->d2a_timing_enabled != 0u &&
        operation->d2a_arrived_micro == 0u)
        operation->d2a_arrived_micro = SparkTpDeviceCollectiveNowMicro();
    if (operation->operation_kind !=
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 &&
        implementation->combine_all_bf16_function != 0)
    {
        const void *rank_devices[
            SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT];
        uint32_t rank_index;

        for (rank_index = 0u;
             rank_index <
                SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT;
             rank_index++)
            rank_devices[rank_index] = 0;
        for (route = 0u; route < implementation->d2a_route_count; route++)
            rank_devices[SparkTpDeviceCollectiveD2aPeer(
                collective->tp_rank,route)] =
                implementation->d2a_bindings[route]
                    [operation->credit_index].receive_device;
        status = implementation->combine_all_bf16_function(
            implementation->combine_context,operation->full_device,
            rank_devices,collective->tp_rank,
            operation->active_sequence_count,
            collective->local_hidden_dimension,operation->cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"D2A-FOLD-FAIL rank=%u ord=%llu status=%u cuda=%s\n",
                collective->tp_rank,
                (unsigned long long)operation->ordinal,
                (uint32_t)status,cudaGetErrorString(cudaGetLastError()));
            SparkTpDeviceCollectiveMarkOperationFailure(implementation,
                operation,operation->generation,status);
            return;
        }
    }
    else
    {
        for (route = 0u; route < implementation->d2a_route_count; route++)
        {
            const SparkTpDeviceCollectiveCreditBinding *binding =
                &implementation->d2a_bindings[route][operation->credit_index];

            if (operation->operation_kind ==
                SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64)
            {
                memcpy(implementation->d2a_fold_stage[route] +
                    (uint64_t)operation->credit_index *
                        implementation->fold_pitch,
                    binding->receive_transport,(size_t)local_bytes);
                status = implementation->combine_u64_max_function(
                    implementation->combine_context,
                    (uint64_t *)operation->full_device,
                    (const uint64_t *)(implementation->d2a_fold_stage[route] +
                        (uint64_t)operation->credit_index *
                            implementation->fold_pitch),
                    operation->active_sequence_count,operation->cuda_stream);
            }
            else
            {
                status = implementation->combine_bf16_function(
                    implementation->combine_context,operation->full_device,
                    binding->receive_device,operation->active_sequence_count,
                    collective->local_hidden_dimension,
                    operation->cuda_stream);
            }
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr,"D2A-FOLD-FAIL rank=%u ord=%llu route=%u status=%u cuda=%s\n",
                    collective->tp_rank,
                    (unsigned long long)operation->ordinal,route,
                    (uint32_t)status,cudaGetErrorString(cudaGetLastError()));
                SparkTpDeviceCollectiveMarkOperationFailure(implementation,
                    operation,operation->generation,status);
                return;
            }
        }
    }
    status = SparkTpDeviceCollectiveCudaStatus(cudaEventRecord(
        implementation->consumer_events[operation->credit_index],
        (cudaStream_t)operation->cuda_stream));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"D2A-EVENT-FAIL rank=%u ord=%llu status=%u cuda=%s\n",
            collective->tp_rank,
            (unsigned long long)operation->ordinal,
            (uint32_t)status,cudaGetErrorString(cudaGetLastError()));
        SparkTpDeviceCollectiveMarkOperationFailure(implementation,
            operation,operation->generation,status);
        return;
    }
    if (implementation->d2a_timing_enabled != 0u)
    {
        uint64_t done_micro = SparkTpDeviceCollectiveNowMicro();

        fprintf(stderr,"D2A-TIMING rank=%u ord=%llu post=%llu arrive=%llu fold=%llu total=%llu us\n",
            collective->tp_rank,(unsigned long long)operation->ordinal,
            (unsigned long long)(operation->d2a_posted_micro -
                operation->d2a_submit_micro),
            (unsigned long long)(operation->d2a_arrived_micro -
                operation->d2a_posted_micro),
            (unsigned long long)(done_micro - operation->d2a_arrived_micro),
            (unsigned long long)(done_micro - operation->d2a_submit_micro));
    }
    (void)SparkTpDeviceCollectiveTransitionPhase(operation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
}

static SparkStatus SparkTpDeviceCollectiveMarkOperationFailure(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint64_t required_generation,
    SparkStatus status)
{
    uint64_t observed_state;
    uint64_t desired_state;

    if (implementation == 0 || operation == 0 ||
        status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY ||
        status == SPARK_STATUS_PENDING ||
        (uint32_t)status > SPARK_STATUS_UNSUPPORTED)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    fprintf(stderr,
        "OP-FAIL rank=%u ordinal=%llu stage=%u status=%u\n",
        implementation->collective->tp_rank,
        (unsigned long long)operation->ordinal,operation->stage,
        (uint32_t)status);
    observed_state = atomic_load_explicit(
        &operation->lifecycle,memory_order_acquire);
    for (;;)
    {
        uint32_t phase;

        if (SparkTpDeviceCollectiveStateGeneration(observed_state) !=
            required_generation)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        phase = SparkTpDeviceCollectiveStatePhase(observed_state);
        if (phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE ||
            phase >= SPARK_TP_DEVICE_COLLECTIVE_PHASE_CALLBACK_CLAIMED)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        if (SparkTpDeviceCollectiveStateHasFailure(observed_state) != 0u)
        {
            return SPARK_STATUS_OK;
        }
        desired_state = observed_state |
            SPARK_TP_DEVICE_COLLECTIVE_FAILURE_REQUESTED |
            ((uint64_t)status <<
                SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_SHIFT);
        if (atomic_compare_exchange_weak_explicit(
                &operation->lifecycle,&observed_state,desired_state,
                memory_order_acq_rel,memory_order_acquire))
        {
            return SPARK_STATUS_OK;
        }
    }
}

static void SparkTpDeviceCollectiveLatchFailure(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkStatus status)
{
    uint32_t credit_index;
    int expected_status;

    if (implementation == 0 || status == SPARK_STATUS_OK ||
        status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING)
    {
        return;
    }
    fprintf(stderr,
        "LATCH rank=%u status=%u\n",
        implementation->collective->tp_rank,(unsigned)status);
    atomic_store_explicit(&implementation->admission_open,0u,
        memory_order_release);
    expected_status = SPARK_STATUS_OK;
    (void)atomic_compare_exchange_strong_explicit(
        &implementation->failure_status,&expected_status,(int)status,
        memory_order_release,memory_order_relaxed);
    for (credit_index = 0u;
         credit_index < implementation->collective->credit_count;
         ++credit_index)
    {
        SparkTpDeviceCollectiveOperation *operation;
        uint64_t state_word;
        uint64_t generation;

        operation = &implementation->operations[credit_index];
        state_word = atomic_load_explicit(
            &operation->lifecycle,memory_order_acquire);
        generation = SparkTpDeviceCollectiveStateGeneration(state_word);
        if (SparkTpDeviceCollectiveStatePhase(state_word) !=
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE)
        {
            (void)SparkTpDeviceCollectiveMarkOperationFailure(
                implementation,operation,generation,status);
        }
    }
}

static uint32_t SparkTpDeviceCollectiveTransitionPhase(
    SparkTpDeviceCollectiveOperation *operation,
    uint32_t expected_phase,
    uint32_t desired_phase)
{
    uint64_t observed_state;
    uint64_t desired_state;

    observed_state = atomic_load_explicit(
        &operation->lifecycle,memory_order_acquire);
    for (;;)
    {
        if (SparkTpDeviceCollectiveStatePhase(observed_state) != expected_phase)
        {
            return 0u;
        }
        desired_state = (observed_state &
            ~SPARK_TP_DEVICE_COLLECTIVE_PHASE_MASK) | desired_phase;
        if (atomic_compare_exchange_weak_explicit(
                &operation->lifecycle,&observed_state,desired_state,
                memory_order_acq_rel,memory_order_acquire))
        {
            return 1u;
        }
    }
}

static void SparkTpDeviceCollectivePollSessions(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t route_index;

    for (route_index = 0u;
         route_index < implementation->route_count;
         ++route_index)
    {
        uint32_t poll_index;

        for (poll_index = 0u;
             poll_index < implementation->collective->credit_count;
             ++poll_index)
        {
            SparkHiddenTransportCompletion completion;
            SparkStatus status;

            status = SparkHiddenTransportPoll(
                implementation->send_sessions[route_index],&completion);
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr,
                    "POLL-FAIL rank=%u route=%u status=%u transfer=%llu active=%llu\n",
                    implementation->collective->tp_rank,route_index,
                    (uint32_t)status,
                    (unsigned long long)completion.transfer_bytes,
                    (unsigned long long)completion.active_sequence_count);
                SparkTpDeviceCollectiveLatchFailure(implementation,status);
                return;
            }
        }
    }
}

static uint32_t SparkTpDeviceCollectiveD2aOperationIsActive(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t credit_index;

    for (credit_index = 0u;
         credit_index < implementation->collective->credit_count;
         ++credit_index)
    {
        const SparkTpDeviceCollectiveOperation *operation =
            &implementation->operations[credit_index];

        if (operation->direct_all_to_all != 0u &&
            SparkTpDeviceCollectiveStatePhase(atomic_load_explicit(
                &operation->lifecycle,memory_order_acquire)) ==
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE)
            return 1u;
    }
    return 0u;
}

static void SparkTpDeviceCollectivePollD2aSessions(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t route_index;

    for (route_index = 0u;
         route_index < implementation->d2a_route_count;
         ++route_index)
    {
        SparkHiddenTransportCompletion completion;
        SparkStatus status;

        status = SparkHiddenTransportPoll(
            implementation->d2a_send_sessions[route_index],&completion);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,
                "D2A-POLL-FAIL rank=%u route=%u status=%u transfer=%llu active=%llu\n",
                implementation->collective->tp_rank,route_index,
                (uint32_t)status,
                (unsigned long long)completion.transfer_bytes,
                (unsigned long long)completion.active_sequence_count);
            SparkTpDeviceCollectiveLatchFailure(implementation,status);
            return;
        }
    }
}

static void SparkTpDeviceCollectivePollAckSessions(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t route_index;

    for (route_index = 0u;
         route_index < implementation->route_count +
            implementation->d2a_route_count;
         ++route_index)
    {
        SparkHiddenTransportSession *session = route_index <
            implementation->route_count ?
            implementation->ack_send_sessions[route_index] :
            implementation->d2a_ack_send_sessions[
                route_index - implementation->route_count];
        uint32_t pop_index;

        for (pop_index = 0u; pop_index < 16u; pop_index++)
        {
            SparkHiddenTransportCompletion completion;
            SparkStatus status;

            status = SparkHiddenTransportPoll(session,&completion);
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr,
                    "ACK-POLL-FAIL rank=%u route=%u status=%u\n",
                    implementation->collective->tp_rank,route_index,
                    (uint32_t)status);
                SparkTpDeviceCollectiveLatchFailure(implementation,status);
                return;
            }
            if (completion.status == SPARK_STATUS_BUSY)
                break;
        }
    }
}

static void SparkTpDeviceCollectivePublishCompletion(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    SparkTpDeviceCollectiveCompletion completion;
    SparkStatus status;
    uint64_t state_word;

    if (SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_CALLBACK_CLAIMED) == 0u)
    {
        return;
    }
    state_word = atomic_load_explicit(
        &operation->lifecycle,memory_order_acquire);
    status = SparkTpDeviceCollectiveStateFailureStatus(state_word);
    if (status == SPARK_STATUS_OK)
    {
        status = (SparkStatus)atomic_load_explicit(
            &implementation->failure_status,memory_order_acquire);
    }
    memset(&completion, 0, sizeof(completion));
    completion.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    completion.descriptor_bytes = (uint32_t)sizeof(completion);
    completion.status = status;
    completion.slot_index = operation->slot_index;
    completion.credit_index = operation->credit_index;
    completion.ordinal = operation->ordinal;
    completion.generation = operation->generation;
    operation->completion_function(operation->completion_context,&completion);
    (void)SparkTpDeviceCollectiveTransitionPhase(operation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_CALLBACK_CLAIMED,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_RELEASE_PENDING);
}

static void SparkTpDeviceCollectiveProgressOperation(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    uint64_t state_word;
    uint32_t phase;

    state_word = atomic_load_explicit(
        &operation->lifecycle,memory_order_acquire);
    phase = SparkTpDeviceCollectiveStatePhase(state_word);
    if (phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE)
        return;
    if (phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING)
        return;
    if (phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE)
    {
        if (SparkTpDeviceCollectiveStateHasFailure(state_word) != 0u)
        {
            operation->stage = TREE_STAGES;
            (void)SparkTpDeviceCollectiveTransitionPhase(operation,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
            return;
        }
        if (SparkTpDeviceCollectiveNowMilli() >= operation->deadline_milli)
        {
            (void)SparkTpDeviceCollectiveMarkOperationFailure(implementation,
                operation,operation->generation,SPARK_STATUS_IO_ERROR);
            return;
        }
        if (operation->direct_all_to_all != 0u)
            SparkTpDeviceCollectiveD2aOperation(implementation,operation);
        else
            SparkTpDeviceCollectiveTreeOperation(implementation,operation);
        return;
    }
    if (phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY)
    {
        SparkTpDeviceCollectivePublishCompletion(implementation,operation);
        return;
    }
    if (phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_RELEASE_PENDING)
    {
        if (operation->acked == 0u)
        {
            if (SparkTpDeviceCollectiveStateHasFailure(state_word) != 0u)
            {
                operation->acked = 1u;
            }
            else if (cudaEventQuery(
                    implementation->consumer_events[
                        operation->credit_index]) == cudaErrorNotReady)
            {
                if (SparkTpDeviceCollectiveNowMilli() >=
                    operation->deadline_milli)
                {
                    fprintf(stderr,
                        "ACK-STALL rank=%u ordinal=%llu stage=%u\n",
                        implementation->collective->tp_rank,
                        (unsigned long long)operation->ordinal,
                        operation->stage);
                    SparkTpDeviceCollectiveLatchFailure(implementation,
                        SPARK_STATUS_IO_ERROR);
                    operation->acked = 1u;
                }
                else
                {
                    return;
                }
            }
            else if (SparkTpDeviceCollectiveSendAcks(implementation,
                    operation) == SPARK_STATUS_BUSY)
            {
                if (SparkTpDeviceCollectiveNowMilli() >=
                    operation->deadline_milli)
                {
                    fprintf(stderr,
                        "ACK-STALL rank=%u ordinal=%llu stage=%u\n",
                        implementation->collective->tp_rank,
                        (unsigned long long)operation->ordinal,
                        operation->stage);
                    SparkTpDeviceCollectiveLatchFailure(implementation,
                        SPARK_STATUS_IO_ERROR);
                    operation->acked = 1u;
                }
                else
                {
                    return;
                }
            }
            else
            {
                operation->acked = 1u;
            }
        }
        atomic_store_explicit(&operation->lifecycle,
            SparkTpDeviceCollectiveStateWord(operation->generation,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE,SPARK_STATUS_OK),
            memory_order_release);
        return;
    }
}

static uint32_t SparkTpDeviceCollectiveCallbacksAreDrained(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t credit_index;

    for (credit_index = 0u;
         credit_index < implementation->collective->credit_count;
         ++credit_index)
    {
        uint32_t phase;

        phase = SparkTpDeviceCollectiveStatePhase(atomic_load_explicit(
            &implementation->operations[credit_index].lifecycle,
            memory_order_acquire));
        if (phase >= SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE &&
            phase <= SPARK_TP_DEVICE_COLLECTIVE_PHASE_CALLBACK_CLAIMED)
        {
            return 0u;
        }
    }
    return 1u;
}

static uint32_t SparkTpDeviceCollectiveOperationsAreDrained(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t credit_index;

    for (credit_index = 0u;
         credit_index < implementation->collective->credit_count;
         ++credit_index)
    {
        if (SparkTpDeviceCollectiveStatePhase(atomic_load_explicit(
                &implementation->operations[credit_index].lifecycle,
                memory_order_acquire)) !=
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE)
        {
            return 0u;
        }
    }
    return 1u;
}

static void *SparkTpDeviceCollectiveProgressMain(void *context)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint32_t credit_index;
    uint32_t poll_cycle = 0u;

    implementation = (SparkTpDeviceCollectiveImplementation *)context;
    for (;;)
    {
        SparkTpDeviceCollectivePollSessions(implementation);
        if ((poll_cycle & 7u) == 0u)
        {
            if (implementation->d2a_route_count != 0u)
                SparkTpDeviceCollectivePollD2aSessions(implementation);
            SparkTpDeviceCollectivePollAckSessions(implementation);
        }
        else if (implementation->d2a_route_count != 0u &&
            SparkTpDeviceCollectiveD2aOperationIsActive(implementation) != 0u)
            SparkTpDeviceCollectivePollD2aSessions(implementation);
        poll_cycle += 1u;
        for (credit_index = 0u;
             credit_index < implementation->collective->credit_count;
             ++credit_index)
        {
            SparkTpDeviceCollectiveProgressOperation(
                implementation,&implementation->operations[credit_index]);
        }
        if (atomic_load_explicit(&implementation->shutdown_requested,
                memory_order_acquire) != 0u &&
            (SparkTpDeviceCollectiveOperationsAreDrained(implementation) != 0u ||
             (SparkTpDeviceCollectiveCallbacksAreDrained(implementation) != 0u &&
              SparkTpDeviceCollectiveNowMilli() >= atomic_load_explicit(
                &implementation->shutdown_deadline_milli,
                memory_order_acquire))))
        {
            break;
        }
    }
    return 0;
}

static SparkStatus SparkTpDeviceCollectiveRegisterFixedSlots(
    SparkTpDeviceCollectiveImplementation *implementation,
    uint32_t timeout_milli)
{
    SparkHiddenTransportPollDescriptor descriptors[
        2u * (SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS + D2A_ROUTE_COUNT) * 3u];
    SparkHiddenTransportPacket packet;
    struct pollfd poll_fds[
        2u * (SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS + D2A_ROUTE_COUNT) * 3u];
    uint64_t deadline_milli;
    uint64_t credit_span_bytes;
    uint32_t descriptor_count;
    uint32_t step_index;
    uint32_t index;
    int poll_result;
    SparkStatus status;

    for (step_index = 0u; step_index < implementation->route_count;
         ++step_index)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding =
            &implementation->bindings[step_index][0u];

        status = SparkTpDeviceCollectiveBuildPacket(
            binding->receive_transport,
            implementation->collective->max_active_sequence_count,
            implementation->step_hidden_dimensions[step_index],
            0u,step_index,implementation->registration_cuda_stream,
            &packet);
        if (status != SPARK_STATUS_OK)
            return status;
        credit_span_bytes = (uint64_t)
            implementation->collective->credit_count *
            ((uint64_t)packet.bytes_per_sequence *
            implementation->collective->max_active_sequence_count +
            NONCE_BYTES);
        status = SparkHiddenTransportSetFixedLocal(
            implementation->receive_sessions[step_index],
            binding->receive_transport,credit_span_bytes);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    for (step_index = 0u; step_index < implementation->d2a_route_count;
         ++step_index)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding =
            &implementation->d2a_bindings[step_index][0u];

        status = SparkTpDeviceCollectiveBuildPacket(
            binding->receive_transport,
            implementation->collective->max_active_sequence_count,
            implementation->collective->local_hidden_dimension,
            0u,step_index,implementation->registration_cuda_stream,
            &packet);
        if (status != SPARK_STATUS_OK)
            return status;
        credit_span_bytes = (uint64_t)
            implementation->collective->credit_count *
            ((uint64_t)packet.bytes_per_sequence *
            implementation->collective->max_active_sequence_count +
            NONCE_BYTES);
        status = SparkHiddenTransportSetFixedLocal(
            implementation->d2a_receive_sessions[step_index],
            binding->receive_transport,credit_span_bytes);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    for (step_index = 0u; step_index < implementation->route_count;
         ++step_index)
    {
        status = SparkHiddenTransportSetFixedLocal(
            implementation->ack_receive_sessions[step_index],
            implementation->ack_receive_slots +
                (uint64_t)step_index *
                    implementation->collective->credit_count,
            (uint64_t)implementation->collective->credit_count *
                sizeof(uint64_t));
        if (status != SPARK_STATUS_OK)
            return status;
    }
    for (step_index = 0u; step_index < implementation->d2a_route_count;
         ++step_index)
    {
        status = SparkHiddenTransportSetFixedLocal(
            implementation->d2a_ack_receive_sessions[step_index],
            implementation->d2a_ack_receive_slots +
                (uint64_t)step_index *
                    implementation->collective->credit_count,
            (uint64_t)implementation->collective->credit_count *
                sizeof(uint64_t));
        if (status != SPARK_STATUS_OK)
            return status;
    }
    deadline_milli = SparkTpDeviceCollectiveNowMilli();
    if (deadline_milli == UINT64_MAX)
        return SPARK_STATUS_IO_ERROR;
    deadline_milli += timeout_milli;
    for (;;)
    {
        uint32_t ready_count = 0u;

        for (step_index = 0u; step_index < implementation->route_count;
             ++step_index)
        {
            status = SparkHiddenTransportPersistentRemoteCreditReady(
                implementation->send_sessions[step_index],0u);
            if (status == SPARK_STATUS_OK)
                ready_count += 1u;
            else if (status != SPARK_STATUS_BUSY)
                return status;
        }
        for (step_index = 0u; step_index < implementation->d2a_route_count;
             ++step_index)
        {
            status = SparkHiddenTransportPersistentRemoteCreditReady(
                implementation->d2a_send_sessions[step_index],0u);
            if (status == SPARK_STATUS_OK)
                ready_count += 1u;
            else if (status != SPARK_STATUS_BUSY)
                return status;
        }
        for (step_index = 0u; step_index < implementation->route_count;
             ++step_index)
        {
            status = SparkHiddenTransportPersistentRemoteCreditReady(
                implementation->ack_send_sessions[step_index],0u);
            if (status == SPARK_STATUS_OK)
                ready_count += 1u;
            else if (status != SPARK_STATUS_BUSY)
                return status;
        }
        for (step_index = 0u; step_index < implementation->d2a_route_count;
             ++step_index)
        {
            status = SparkHiddenTransportPersistentRemoteCreditReady(
                implementation->d2a_ack_send_sessions[step_index],0u);
            if (status == SPARK_STATUS_OK)
                ready_count += 1u;
            else if (status != SPARK_STATUS_BUSY)
                return status;
        }
        if (ready_count == 2u * (implementation->route_count +
                implementation->d2a_route_count))
            break;
        if (SparkTpDeviceCollectiveNowMilli() >= deadline_milli)
            return SPARK_STATUS_IO_ERROR;
        descriptor_count = 0u;
        for (step_index = 0u; step_index < implementation->route_count;
             ++step_index)
        {
            uint32_t session_descriptors = 0u;

            (void)SparkHiddenTransportGetPollDescriptors(
                implementation->send_sessions[step_index],
                descriptors + descriptor_count,
                (uint32_t)(sizeof(descriptors) /
                    sizeof(descriptors[0])) - descriptor_count,
                &session_descriptors);
            descriptor_count += session_descriptors;
        }
        for (step_index = 0u; step_index < implementation->d2a_route_count;
             ++step_index)
        {
            uint32_t session_descriptors = 0u;

            (void)SparkHiddenTransportGetPollDescriptors(
                implementation->d2a_send_sessions[step_index],
                descriptors + descriptor_count,
                (uint32_t)(sizeof(descriptors) /
                    sizeof(descriptors[0])) - descriptor_count,
                &session_descriptors);
            descriptor_count += session_descriptors;
        }
        for (step_index = 0u; step_index < implementation->route_count;
             ++step_index)
        {
            uint32_t session_descriptors = 0u;

            (void)SparkHiddenTransportGetPollDescriptors(
                implementation->ack_send_sessions[step_index],
                descriptors + descriptor_count,
                (uint32_t)(sizeof(descriptors) /
                    sizeof(descriptors[0])) - descriptor_count,
                &session_descriptors);
            descriptor_count += session_descriptors;
        }
        for (step_index = 0u; step_index < implementation->d2a_route_count;
             ++step_index)
        {
            uint32_t session_descriptors = 0u;

            (void)SparkHiddenTransportGetPollDescriptors(
                implementation->d2a_ack_send_sessions[step_index],
                descriptors + descriptor_count,
                (uint32_t)(sizeof(descriptors) /
                    sizeof(descriptors[0])) - descriptor_count,
                &session_descriptors);
            descriptor_count += session_descriptors;
        }
        if (descriptor_count == 0u)
        {
            sched_yield();
            continue;
        }
        for (index = 0u; index < descriptor_count; ++index)
        {
            poll_fds[index].fd = descriptors[index].fd;
            poll_fds[index].events = POLLIN;
            poll_fds[index].revents = 0u;
        }
        poll_result = poll(poll_fds,(nfds_t)descriptor_count,200);
        if (poll_result < 0 && errno != EINTR)
            return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveSubmitHiddenInner(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    SparkTpDeviceCollectiveOperation *operation;
    SparkStatus status;
    uint64_t expected_state;
    uint64_t desired_state;
    uint64_t now_milli;
    uint64_t generation;
    uint32_t credit_index;

    implementation = (SparkTpDeviceCollectiveImplementation *)
        collective->implementation;
    if (implementation == 0 || submission == 0 ||
        submission->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        submission->descriptor_bytes != sizeof(*submission) ||
        submission->local_device == 0 || submission->full_device == 0 ||
        submission->cuda_stream == 0 ||
        submission->completion_function == 0 ||
        submission->active_sequence_count == 0u ||
        submission->active_sequence_count >
            collective->max_active_sequence_count ||
        submission->ordinal == UINT64_MAX)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (operation_kind ==
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 &&
        implementation->combine_u64_max_function == 0)
        return SPARK_STATUS_UNSUPPORTED;
    if (implementation->combine_bf16_function == 0)
        return SPARK_STATUS_UNSUPPORTED;
    if (atomic_load_explicit(&implementation->admission_open,
            memory_order_acquire) == 0u)
    {
        return (SparkStatus)atomic_load_explicit(
            &implementation->failure_status,memory_order_acquire);
    }
    credit_index = (uint32_t)(submission->ordinal %
        collective->credit_count);
    generation = submission->ordinal /
        collective->credit_count + 1u;
    if (generation == 0u ||
        generation > (UINT64_MAX >>
            SPARK_TP_DEVICE_COLLECTIVE_GENERATION_SHIFT))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    operation = &implementation->operations[credit_index];
    expected_state = atomic_load_explicit(
        &operation->lifecycle,memory_order_acquire);
    if (SparkTpDeviceCollectiveStatePhase(expected_state) !=
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE ||
        generation <= SparkTpDeviceCollectiveStateGeneration(expected_state))
    {
        return SPARK_STATUS_BUSY;
    }
    desired_state = SparkTpDeviceCollectiveStateWord(generation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING,0u);
    if (!atomic_compare_exchange_strong_explicit(
            &operation->lifecycle,&expected_state,desired_state,
            memory_order_acq_rel,memory_order_acquire))
    {
        return SPARK_STATUS_BUSY;
    }
    now_milli = SparkTpDeviceCollectiveNowMilli();
    if (now_milli == UINT64_MAX)
    {
        now_milli = 0u;
    }
    operation->slot_index = submission->slot_index;
    operation->credit_index = credit_index;
    operation->active_sequence_count = submission->active_sequence_count;
    operation->operation_kind = operation_kind;
    operation->stage = 0u;
    operation->arrived = 0u;
    operation->packed = 0u;
    operation->ordinal = submission->ordinal;
    operation->generation = generation;
    operation->deadline_milli = now_milli +
        collective->operation_timeout_milli;
    operation->local_device = submission->local_device;
    operation->full_device = submission->full_device;
    operation->cuda_stream = submission->cuda_stream;
    operation->continuation_cuda_stream = submission->cuda_stream;
    operation->completion_function = submission->completion_function;
    operation->completion_context = submission->completion_context;
    operation->direct_all_to_all =
        implementation->d2a_route_count != 0u &&
        SparkTpDeviceCollectiveOperationBytes(collective,operation) <=
            (uint64_t)collective->direct_all_to_all_max_payload_bytes ?
        1u : 0u;
    operation->acked = 0u;
    operation->d2a_submit_micro = 0u;
    operation->d2a_posted_micro = 0u;
    operation->d2a_arrived_micro = 0u;
    status = now_milli == 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
    if (status == SPARK_STATUS_OK &&
        operation->full_device != operation->local_device)
    {
        uint64_t local_bytes =
            SparkTpDeviceCollectiveOperationBytes(collective,operation);
        status = SparkTpDeviceCollectiveCopyRows(operation->full_device,
            local_bytes,operation->local_device,local_bytes,local_bytes,1u,
            cudaMemcpyDeviceToDevice,
            operation->cuda_stream);
    }
    if (status == SPARK_STATUS_OK && operation->direct_all_to_all != 0u)
    {
        uint64_t local_bytes =
            SparkTpDeviceCollectiveOperationBytes(collective,operation);
        status = SparkTpDeviceCollectivePackSendRows(collective,
            &implementation->d2a_bindings[0u][credit_index],
            operation->full_device,local_bytes,local_bytes,local_bytes,1u,
            operation->cuda_stream);
        if (implementation->d2a_timing_enabled != 0u)
            operation->d2a_submit_micro = SparkTpDeviceCollectiveNowMicro();
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTpDeviceCollectiveCudaStatus(cudaEventRecord(
            implementation->consumer_events[credit_index],
            (cudaStream_t)operation->cuda_stream));
    }
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,generation,status);
        (void)SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
        return SPARK_STATUS_OK;
    }
    if (SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE) == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveSubmitHidden(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    return SparkTpDeviceCollectiveSubmitHiddenInner(collective,submission,
        operation_kind);
}

SparkStatus SparkTpDeviceCollectiveSubmitBf16(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission)
{
    if (collective != 0 && collective->backend_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
        return SparkTpDeviceCollectiveNcclSubmitBf16(collective,submission);
    return SparkTpDeviceCollectiveSubmitHidden(collective,submission,
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
}

SparkStatus SparkTpDeviceCollectiveSubmitU64Max(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission)
{
    if (collective == 0 || submission == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (collective->backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
        return SparkTpDeviceCollectiveNcclSubmitU64Max(collective,submission);
    return SparkTpDeviceCollectiveSubmitHidden(collective,submission,
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
}

SparkStatus SparkTpDeviceCollectiveRequestFailure(
    SparkTpDeviceCollective *collective,
    SparkStatus failure_status)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    if (collective != 0 && collective->backend_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        return SparkTpDeviceCollectiveNcclRequestFailure(
            collective,failure_status);
    }
    if (collective == 0 || collective->implementation == 0 ||
        collective->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        failure_status == SPARK_STATUS_OK ||
        failure_status == SPARK_STATUS_BUSY ||
        failure_status == SPARK_STATUS_PENDING)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    implementation = (SparkTpDeviceCollectiveImplementation *)
        collective->implementation;
    SparkTpDeviceCollectiveLatchFailure(implementation,failure_status);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveRequestOperationFailure(
    SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    SparkStatus failure_status)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    SparkTpDeviceCollectiveOperation *operation;
    uint64_t generation;
    uint32_t credit_index;

    if (collective != 0 && collective->backend_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        return SparkTpDeviceCollectiveNcclRequestOperationFailure(
            collective,ordinal,failure_status);
    }
    if (collective == 0 || collective->implementation == 0 ||
        collective->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        failure_status == SPARK_STATUS_OK ||
        failure_status == SPARK_STATUS_BUSY ||
        failure_status == SPARK_STATUS_PENDING)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    implementation = (SparkTpDeviceCollectiveImplementation *)
        collective->implementation;
    credit_index = (uint32_t)(ordinal %
        collective->credit_count);
    generation = ordinal / collective->credit_count + 1u;
    operation = &implementation->operations[credit_index];
    return SparkTpDeviceCollectiveMarkOperationFailure(
        implementation,operation,generation,failure_status);
}

SparkStatus SparkTpDeviceCollectiveOperationPhase(
    const SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    uint32_t *phase_out,
    uint32_t *failure_requested_out)
{
    const SparkTpDeviceCollectiveImplementation *implementation;
    const SparkTpDeviceCollectiveOperation *operation;
    uint64_t generation;
    uint64_t state_word;
    uint32_t credit_index;

    if (collective != 0 && collective->backend_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        return SparkTpDeviceCollectiveNcclOperationPhase(
            collective,ordinal,phase_out,failure_requested_out);
    }
    if (collective == 0 || collective->implementation == 0 ||
        phase_out == 0 || failure_requested_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    implementation = (const SparkTpDeviceCollectiveImplementation *)
        collective->implementation;
    credit_index = (uint32_t)(ordinal %
        collective->credit_count);
    generation = ordinal / collective->credit_count + 1u;
    operation = &implementation->operations[credit_index];
    state_word = atomic_load_explicit(
        &operation->lifecycle,memory_order_acquire);
    if (SparkTpDeviceCollectiveStateGeneration(state_word) != generation)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    *phase_out = SparkTpDeviceCollectiveStatePhase(state_word);
    *failure_requested_out =
        SparkTpDeviceCollectiveStateHasFailure(state_word);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveExchangeBf16(
    SparkTpDeviceCollective *collective,
    const void *send_device,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream)
{
    (void)collective;
    (void)send_device;
    (void)receive_device;
    (void)active_sequence_count;
    (void)hidden_dimension;
    (void)step_index;
    (void)cuda_stream;
    return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectivePrepareReceiveBf16(
    SparkTpDeviceCollective *collective,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream)
{
    (void)collective;
    (void)receive_device;
    (void)active_sequence_count;
    (void)hidden_dimension;
    (void)step_index;
    (void)cuda_stream;
    return SPARK_STATUS_UNSUPPORTED;
}

void SparkTpDeviceCollectiveDumpOperations(
    const SparkTpDeviceCollective *collective)
{
    const SparkTpDeviceCollectiveImplementation *implementation;
    uint32_t credit_index;

    if (collective == 0 || collective->implementation == 0)
        return;
    implementation = (const SparkTpDeviceCollectiveImplementation *)
        collective->implementation;
    for (credit_index = 0u;
         credit_index < collective->credit_count;
         ++credit_index)
    {
        const SparkTpDeviceCollectiveOperation *operation =
            &implementation->operations[credit_index];
        uint64_t state_word = atomic_load_explicit(
            &operation->lifecycle,memory_order_acquire);
        fprintf(stderr,
            "OP-DUMP rank=%u credit=%u phase=%u gen=%llu ordinal=%llu stage=%u arrived=%02x packed=%u d2a=%u acked=%u\n",
            collective->tp_rank,credit_index,
            SparkTpDeviceCollectiveStatePhase(state_word),
            (unsigned long long)SparkTpDeviceCollectiveStateGeneration(
                state_word),
            (unsigned long long)operation->ordinal,operation->stage,
            operation->arrived,operation->packed,
            operation->direct_all_to_all,operation->acked);
    }
}

SparkStatus SparkTpDeviceCollectiveCreditBindingRouteCount(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t *route_count_out)
{
    uint32_t algorithm_mask;

    if (config == 0 || route_count_out == 0 || config->abi_version !=
            SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        !SparkTpDeviceCollectiveDegreeIsSupported(config->tp_degree))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (config->backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        *route_count_out = 0u;
        return SPARK_STATUS_OK;
    }
    if (config->backend_kind !=
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    algorithm_mask = SparkTpDeviceCollectiveAlgorithmMask(config);
    if ((algorithm_mask & ~SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS) != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *route_count_out = tree_route_count(config->tp_rank) +
        (SparkTpDeviceCollectiveD2aEnabled(config) != 0u ?
            D2A_ROUTE_COUNT : 0u);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveCreditStepCount(
    uint32_t backend_kind,
    uint32_t tp_degree,
    uint32_t *step_count_out)
{
    if (step_count_out == 0 ||
        !SparkTpDeviceCollectiveDegreeIsSupported(tp_degree))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        *step_count_out = 0u;
        return SPARK_STATUS_OK;
    }
    if (backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *step_count_out = SparkTpDeviceCollectiveStepCount(tp_degree);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveProbeMemoryMode(
    uint32_t backend_kind,
    const char *backend_module_path,
    uint32_t *memory_mode_out)
{
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkStatus status;

    if (memory_mode_out == 0 ||
        (backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT &&
         backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        *memory_mode_out = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
        return SPARK_STATUS_OK;
    }
    if (!SparkTpDeviceCollectiveTextIsValid(backend_module_path))
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(&transport_library,0,sizeof(transport_library));
    status = SparkHiddenTransportLoadInterfaceFromSharedObject(
        backend_module_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
        &transport_library);
    if (status != SPARK_STATUS_OK)
    {
        const char *dl_reason = dlerror();
        fprintf(stderr,"TREE-PROBE-DSO-FAIL path=%s status=%u dlerror=%s\n",
            backend_module_path,(uint32_t)status,
            dl_reason != 0 ? dl_reason : "none");
        return status;
    }
    if ((transport_library.transport_interface.capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_GPUDIRECT_RDMA) != 0u)
        *memory_mode_out = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
    else if ((transport_library.transport_interface.capability_flags &
            (SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA |
             SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY)) ==
        (SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA |
         SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY))
        *memory_mode_out = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST;
    else
        status = SPARK_STATUS_INVALID_ARGUMENT;
    SparkHiddenTransportUnloadInterface(&transport_library);
    return status;
}

SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    SparkStatus status;
    uint32_t binding_index;
    uint32_t credit_count;
    uint32_t credit_index;
    uint32_t step_index;

    if (collective_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(collective_out, 0, sizeof(*collective_out));
    if (config != 0 && config->backend_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        return SparkTpDeviceCollectiveNcclCreate(config,collective_out);
    }
    status = SparkTpDeviceCollectiveValidateConfig(config,&credit_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    implementation = (SparkTpDeviceCollectiveImplementation *)calloc(
        1u,sizeof(*implementation));
    if (implementation == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    collective_out->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    collective_out->backend_kind =
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
    collective_out->tp_degree = config->tp_degree;
    collective_out->tp_rank = config->tp_rank;
    collective_out->step_count = SparkTpDeviceCollectiveStepCount(
        config->tp_degree);
    collective_out->operation_kind = config->operation_kind;
    collective_out->credit_count = credit_count;
    collective_out->local_hidden_dimension = config->local_hidden_dimension;
    collective_out->max_active_sequence_count =
        config->max_active_sequence_count;
    collective_out->operation_timeout_milli = config->operation_timeout_milli;
    collective_out->algorithm_mask =
        SparkTpDeviceCollectiveAlgorithmMask(config);
    collective_out->rail_count = config->rail_count;
    collective_out->direct_all_to_all_max_payload_bytes =
        config->direct_all_to_all_max_payload_bytes;
    collective_out->split_ring_min_payload_bytes =
        config->split_ring_min_payload_bytes;
    collective_out->collective_identifier = config->collective_identifier;
    collective_out->implementation = implementation;
    implementation->collective = collective_out;
    implementation->route_count = SparkTpDeviceCollectiveConfigRouteCount(
        config);
    implementation->binding_route_count = implementation->route_count;
    implementation->d2a_route_count =
        SparkTpDeviceCollectiveD2aEnabled(config) != 0u ?
        D2A_ROUTE_COUNT : 0u;
    {
        const char *timing_env = getenv("SPARK_TP_D2A_TIMING");
        implementation->d2a_timing_enabled = timing_env != 0 &&
            timing_env[0] == '1' ? 1u : 0u;
    }
    implementation->registration_cuda_stream =
        config->registration_cuda_stream;
    implementation->combine_bf16_function = config->combine_bf16_function;
    implementation->combine_all_bf16_function =
        config->combine_tp4_bf16_function;
    implementation->combine_u64_max_function =
        config->combine_u64_max_function;
    implementation->combine_context = config->combine_context;
    if (config->debug_hooks != 0)
    {
        implementation->debug_hooks = *config->debug_hooks;
    }
    atomic_init(&implementation->admission_open,1u);
    atomic_init(&implementation->shutdown_requested,0u);
    atomic_init(&implementation->shutdown_deadline_milli,UINT64_MAX);
    atomic_init(&implementation->failure_status,SPARK_STATUS_OK);
    for (credit_index = 0u;
         credit_index < collective_out->credit_count;
         ++credit_index)
    {
        atomic_init(&implementation->operations[credit_index].lifecycle,
            SparkTpDeviceCollectiveStateWord(0u,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE,0u));
        if (cudaEventCreateWithFlags(
                &implementation->consumer_events[credit_index],
                cudaEventDisableTiming) != cudaSuccess)
        {
            fprintf(stderr,"TREE-EVENT-FAIL rank=%u credit=%u cuda=%s\n",
                collective_out->tp_rank,credit_index,
                cudaGetErrorString(cudaGetLastError()));
            status = SPARK_STATUS_DRIVER_LOAD_ERROR;
            goto fail_create;
        }
    }
    for (binding_index = 0u;
         binding_index < config->credit_binding_count;
         ++binding_index)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding;

        binding = &config->credit_bindings[binding_index];
        if ((binding->flags &
                SPARK_TP_DEVICE_COLLECTIVE_BINDING_DIRECT_ALL_TO_ALL) != 0u)
            implementation->d2a_bindings[binding->step_index]
                [binding->credit_index] = *binding;
        else
            implementation->bindings[binding->step_index]
                [binding->credit_index] = *binding;
        if (config->registration_cuda_stream != 0)
        {
            (void)cudaMemsetAsync(binding->send_device,0,1u,
                (cudaStream_t)config->registration_cuda_stream);
            (void)cudaMemsetAsync(binding->receive_device,0,1u,
                (cudaStream_t)config->registration_cuda_stream);
        }
    }
    for (step_index = 0u; step_index < implementation->route_count;
        ++step_index)
        implementation->step_hidden_dimensions[step_index] =
            config->local_hidden_dimension;
    status = SparkHiddenTransportLoadInterfaceFromSharedObject(
        config->backend_module_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS |
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS,
        &implementation->transport_library);
    if (status != SPARK_STATUS_OK)
    {
        const char *dl_reason = dlerror();
        fprintf(stderr,"TREE-DSO-FAIL rank=%u path=%s status=%u dlerror=%s\n",
            config->tp_rank,
            config->backend_module_path != 0 ? config->backend_module_path : "null",
            (uint32_t)status,dl_reason != 0 ? dl_reason : "none");
        goto fail_create;
    }
    if ((implementation->transport_library.transport_interface
            .capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_GPUDIRECT_RDMA) != 0u)
    {
        collective_out->memory_mode =
            SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
    }
    else if ((implementation->transport_library.transport_interface
                .capability_flags &
                (SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA |
                 SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY)) ==
            (SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA |
             SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY))
    {
        collective_out->memory_mode =
            SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST;
    }
    else
    {
        status = SPARK_STATUS_INVALID_ARGUMENT;
        goto fail_create;
    }
    if (implementation->transport_library.transport_interface.send_fixed ==
        0)
    {
        status = SPARK_STATUS_UNSUPPORTED;
        goto fail_create;
    }
    status = SparkTpDeviceCollectiveOpenTreeSessions(implementation,config);
    if (status != SPARK_STATUS_OK)
    {
        goto fail_create;
    }
    if (implementation->d2a_route_count != 0u)
    {
        status = SparkTpDeviceCollectiveOpenD2aSessions(implementation,
            config);
        if (status != SPARK_STATUS_OK)
        {
            goto fail_create;
        }
    }
    implementation->fixed_slots_enabled = 1u;
    implementation->nonce_offset = (uint64_t)config->local_hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT *
        config->max_active_sequence_count;
    {
        uint64_t pitch = implementation->nonce_offset + NONCE_BYTES;
        size_t bytes = (size_t)pitch *
            (SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS +
                implementation->d2a_route_count) *
            implementation->collective->credit_count;
        uint32_t route_index;

        implementation->fold_pitch = pitch;
        if (cudaHostAlloc(&implementation->fold_stage_host,bytes,
                cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess)
        {
            for (route_index = 0u;
                 route_index < SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS;
                 route_index++)
                if (cudaHostGetDevicePointer(
                        &implementation->fold_stage[route_index],
                        (uint8_t *)implementation->fold_stage_host +
                            (size_t)pitch *
                            implementation->collective->credit_count *
                            route_index,0u) != cudaSuccess)
                    implementation->fold_stage[route_index] = 0;
            for (route_index = 0u;
                 route_index < implementation->d2a_route_count;
                 route_index++)
                if (cudaHostGetDevicePointer(
                        &implementation->d2a_fold_stage[route_index],
                        (uint8_t *)implementation->fold_stage_host +
                            (size_t)pitch *
                            implementation->collective->credit_count *
                            (SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS +
                                route_index),0u) != cudaSuccess)
                    implementation->d2a_fold_stage[route_index] = 0;
        }
    }
    {
        size_t ack_bytes = (size_t)(SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS +
            implementation->d2a_route_count) *
            implementation->collective->credit_count * sizeof(uint64_t);

        if (cudaHostAlloc(&implementation->ack_region_host,ack_bytes,
                cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
            cudaHostAlloc(&implementation->ack_stage_host,ack_bytes,
                cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
        {
            fprintf(stderr,"ACK-ALLOC-FAIL rank=%u cuda=%s\n",
                collective_out->tp_rank,
                cudaGetErrorString(cudaGetLastError()));
            status = SPARK_STATUS_DRIVER_LOAD_ERROR;
            goto fail_create;
        }
        memset(implementation->ack_region_host,0,ack_bytes);
        memset(implementation->ack_stage_host,0,ack_bytes);
        implementation->ack_receive_slots =
            (uint64_t *)implementation->ack_region_host;
        implementation->d2a_ack_receive_slots =
            (uint64_t *)implementation->ack_region_host +
            (uint64_t)SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS *
                implementation->collective->credit_count;
        implementation->ack_stage_slots =
            (uint64_t *)implementation->ack_stage_host;
        implementation->d2a_ack_stage_slots =
            (uint64_t *)implementation->ack_stage_host +
            (uint64_t)SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS *
                implementation->collective->credit_count;
    }
    status = SparkTpDeviceCollectiveRegisterFixedSlots(implementation,
        config->connect_timeout_milli);
    if (status != SPARK_STATUS_OK)
    {
        goto fail_create;
    }
    if (pthread_create(&implementation->progress_thread,0,
            SparkTpDeviceCollectiveProgressMain,implementation) != 0)
    {
        status = SPARK_STATUS_INTERNAL_ERROR;
        goto fail_create;
    }
    implementation->progress_thread_started = 1u;
    return SPARK_STATUS_OK;

fail_create:
    SparkTpDeviceCollectiveCloseSessions(implementation);
    SparkHiddenTransportUnloadInterface(&implementation->transport_library);
    SparkTpDeviceCollectiveDestroyEvents(implementation);
    if (implementation->fold_stage_host != 0)
        (void)cudaFreeHost(implementation->fold_stage_host);
    if (implementation->ack_region_host != 0)
        (void)cudaFreeHost(implementation->ack_region_host);
    if (implementation->ack_stage_host != 0)
        (void)cudaFreeHost(implementation->ack_stage_host);
    free(implementation);
    collective_out->implementation = 0;
    return status;
}

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    if (collective == 0)
    {
        return;
    }
    if (collective->backend_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL)
    {
        SparkTpDeviceCollectiveNcclDestroy(collective);
        return;
    }
    implementation = (SparkTpDeviceCollectiveImplementation *)
        collective->implementation;
    if (implementation == 0)
    {
        memset(collective,0,sizeof(*collective));
        return;
    }
    atomic_store_explicit(&implementation->admission_open,0u,
        memory_order_release);
    SparkTpDeviceCollectiveLatchFailure(
        implementation,SPARK_STATUS_IO_ERROR);
    {
        uint64_t now_milli;

        now_milli = SparkTpDeviceCollectiveNowMilli();
        atomic_store_explicit(&implementation->shutdown_deadline_milli,
            now_milli == UINT64_MAX || UINT64_MAX - now_milli <
                collective->operation_timeout_milli ? UINT64_MAX :
                now_milli + collective->operation_timeout_milli,
            memory_order_release);
    }
    atomic_store_explicit(&implementation->shutdown_requested,1u,
        memory_order_release);
    if (implementation->progress_thread_started != 0u)
    {
        (void)pthread_join(implementation->progress_thread,0);
        implementation->progress_thread_started = 0u;
    }
    SparkTpDeviceCollectiveCloseSessions(implementation);
    SparkHiddenTransportUnloadInterface(&implementation->transport_library);
    SparkTpDeviceCollectiveDestroyEvents(implementation);
    if (implementation->fold_stage_host != 0)
        (void)cudaFreeHost(implementation->fold_stage_host);
    if (implementation->ack_region_host != 0)
        (void)cudaFreeHost(implementation->ack_region_host);
    if (implementation->ack_stage_host != 0)
        (void)cudaFreeHost(implementation->ack_stage_host);
    free(implementation);
    memset(collective,0,sizeof(*collective));
}
