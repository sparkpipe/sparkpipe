#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_tp_device_collective.h"

#include <cuda_runtime_api.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Present in cudart; the host-only compatibility header used by unit tests
 * intentionally exposes only its older subset. */
extern cudaError_t cudaEventQuery(cudaEvent_t event);

#define SPARK_TP_DEVICE_COLLECTIVE_PORT_STRIDE 64u
#define SPARK_TP_DEVICE_COLLECTIVE_PHASE_MASK 0xffull
#define SPARK_TP_DEVICE_COLLECTIVE_FAILURE_REQUESTED 0x100ull
#define SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_SHIFT 9u
#define SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_MASK 0xfe00ull
#define SPARK_TP_DEVICE_COLLECTIVE_GENERATION_SHIFT 16u
#define SPARK_TP_DEVICE_COLLECTIVE_PROGRESS_YIELD_INTERVAL 256u

#if defined(__GNUC__)
/* Optional link-time test seam. Production binaries leave this weak symbol
 * unresolved; the deterministic SUBMIT_BUILDING race test supplies it. */
extern void SparkTpDeviceCollectiveDebugSubmissionClaimed(
    uint32_t credit_index,
    uint64_t generation) __attribute__((weak));
#endif

typedef struct SparkTpDeviceCollectiveOperation
{
    atomic_uint_fast64_t lifecycle;
    uint32_t slot_index;
    uint32_t credit_index;
    uint32_t active_sequence_count;
    uint32_t current_step;
    uint32_t reserved_send_mask;
    uint32_t activated_receive_mask;
    uint32_t sent_mask;
    uint32_t send_complete_mask;
    uint32_t receive_complete_mask;
    uint32_t released_receive_mask;
    uint32_t terminal_after_consume;
    uint64_t ordinal;
    uint64_t generation;
    uint64_t deadline_milli;
    const void *local_device;
    void *full_device;
    void *cuda_stream;
    SparkTpDeviceCollectiveCompletionFunction completion_function;
    void *completion_context;
    uint64_t profile_submit_ns;
    uint64_t profile_reserved_ns;
    uint64_t profile_send_done_ns[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t profile_send_complete_ns[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t profile_receive_complete_ns[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t profile_send_service_ns[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t profile_receive_done_ns[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t profile_consume_done_ns[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t profile_release_done_ns[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    uint64_t profile_complete_ns;
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
    SparkTpDeviceCollectiveOperation operations[
        SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    SparkTpDeviceCollectiveDebugHooks debug_hooks;
    SparkTpDeviceCollectiveCombineBf16Function combine_bf16_function;
    void *combine_context;
    cudaEvent_t consumer_events[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    void *registration_cuda_stream;
    atomic_uint admission_open;
    atomic_uint shutdown_requested;
    atomic_uint_fast64_t shutdown_deadline_milli;
    atomic_int failure_status;
    pthread_t progress_thread;
    uint32_t progress_thread_started;
    uint32_t profile_enabled;
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

    if (clock_gettime(CLOCK_MONOTONIC, &current_time) != 0)
    {
        return UINT64_MAX;
    }
    return ((uint64_t)current_time.tv_sec * 1000u) +
        ((uint64_t)current_time.tv_nsec / 1000000u);
}

static uint64_t SparkTpDeviceCollectiveNowNano(void)
{
    struct timespec current_time;

    if (clock_gettime(CLOCK_MONOTONIC,&current_time) != 0)
    {
        return 0u;
    }
    return ((uint64_t)current_time.tv_sec * UINT64_C(1000000000)) +
        (uint64_t)current_time.tv_nsec;
}

static uint64_t SparkTpDeviceCollectiveProfileDelta(
    uint64_t start_ns,
    uint64_t end_ns)
{
    return end_ns >= start_ns ? end_ns - start_ns : 0u;
}

static uint32_t SparkTpDeviceCollectiveProfileIsEnabled(void)
{
    const char *profile_value;

    profile_value = getenv("SPARK_TP_COLLECTIVE_PROFILE");
    return profile_value != 0 && strcmp(profile_value,"1") == 0 ? 1u : 0u;
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
    return (state_word & SPARK_TP_DEVICE_COLLECTIVE_FAILURE_REQUESTED) != 0u;
}

static SparkStatus SparkTpDeviceCollectiveStateFailureStatus(
    uint64_t state_word)
{
    uint64_t status;

    status = (state_word & SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_MASK) >>
        SPARK_TP_DEVICE_COLLECTIVE_FAILURE_STATUS_SHIFT;
    return status <= SPARK_STATUS_UNSUPPORTED ? (SparkStatus)status :
        SPARK_STATUS_INTERNAL_ERROR;
}

static uint64_t SparkTpDeviceCollectiveStateGeneration(uint64_t state_word)
{
    return state_word >> SPARK_TP_DEVICE_COLLECTIVE_GENERATION_SHIFT;
}

static uint32_t SparkTpDeviceCollectiveAllStepMask(
    const SparkTpDeviceCollective *collective)
{
    return collective->step_count == 0u ? 0u :
        (1u << collective->step_count) - 1u;
}

static int SparkTpDeviceCollectiveTextIsValid(const char *text)
{
    return text != 0 && text[0] != '\0';
}

static SparkStatus SparkTpDeviceCollectiveValidateBindings(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t step_count,
    uint32_t credit_count)
{
    uint8_t seen[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS]
        [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    uint32_t binding_index;
    uint32_t required_binding_count;

    required_binding_count = step_count * credit_count;
    if (config->credit_binding_count != required_binding_count ||
        (required_binding_count != 0u && config->credit_bindings == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(seen, 0, sizeof(seen));
    for (binding_index = 0u;
         binding_index < config->credit_binding_count;
         ++binding_index)
    {
        const SparkTpDeviceCollectiveCreditBinding *binding;

        binding = &config->credit_bindings[binding_index];
        if (binding->step_index >= step_count ||
            binding->credit_index >= credit_count ||
            binding->send_device == 0 || binding->receive_device == 0 ||
            binding->send_transport == 0 ||
            binding->receive_transport == 0 ||
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
    uint32_t *step_count_out,
    uint32_t *credit_count_out)
{
    uint32_t rank_index;
    uint32_t step_count;

    uint32_t credit_count;

    if (config == 0 || step_count_out == 0 || credit_count_out == 0 ||
        config->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        !SparkTpDeviceCollectiveDegreeIsSupported(config->tp_degree) ||
        config->tp_rank >= config->tp_degree ||
        config->local_hidden_dimension == 0u ||
        config->max_active_sequence_count == 0u ||
        config->connect_timeout_milli == 0u ||
        config->operation_timeout_milli == 0u ||
        config->control_port_base == 0u ||
        config->collective_identifier == 0u ||
        config->operation_kind >
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 ||
        (config->operation_kind ==
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 &&
            config->combine_bf16_function == 0) ||
        !SparkTpDeviceCollectiveTextIsValid(config->transport_module_path) ||
        !SparkTpDeviceCollectiveTextIsValid(config->local_host) ||
        (config->tp_degree > 1u && config->registration_cuda_stream == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    credit_count = config->credit_count == 0u ?
        SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT : config->credit_count;
    if (credit_count > SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT)
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
    step_count = SparkTpDeviceCollectiveStepCount(config->tp_degree);
    *step_count_out = step_count;
    *credit_count_out = credit_count;
    return SparkTpDeviceCollectiveValidateBindings(
        config,step_count,credit_count);
}

static SparkStatus SparkTpDeviceCollectiveBuildEndpoint(
    const SparkTpDeviceCollectiveConfig *config,
    const SparkTpDeviceCollective *collective,
    uint32_t step_index,
    uint32_t source_rank,
    uint32_t sink_rank,
    uint32_t hidden_dimension,
    char *route_name,
    SparkHiddenTransportEndpoint *endpoint)
{
    uint32_t port_base;
    int written;

    if (config == 0 || collective == 0 || route_name == 0 || endpoint == 0 ||
        source_rank >= config->tp_degree || sink_rank >= config->tp_degree ||
        source_rank == sink_rank ||
        step_index >= SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (step_index > (UINT32_MAX - config->control_port_base) /
            SPARK_TP_DEVICE_COLLECTIVE_PORT_STRIDE)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    port_base = config->control_port_base +
        step_index * SPARK_TP_DEVICE_COLLECTIVE_PORT_STRIDE;
    if (port_base > UINT16_MAX - sink_rank)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
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
    endpoint->control_port_base = port_base;
    endpoint->source_host = source_rank == config->tp_rank ?
        config->local_host : config->rank_hosts[source_rank];
    endpoint->sink_host = sink_rank == config->tp_rank ?
        config->local_host : config->rank_hosts[sink_rank];
    endpoint->route_identifier = config->collective_identifier;
    if (SparkHiddenTransportConfigureEndpointOpenTimeout(endpoint,
            config->connect_timeout_milli) != SPARK_STATUS_OK)
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
    }
}

static SparkStatus SparkTpDeviceCollectiveOpenSession(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t step_index,
    uint32_t source_rank,
    uint32_t sink_rank,
    char *route_name,
    SparkHiddenTransportSession **session)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkStatus status;

    memset(&endpoint, 0, sizeof(endpoint));
    status = SparkTpDeviceCollectiveBuildEndpoint(
        config,implementation->collective,step_index,source_rank,sink_rank,
        implementation->step_hidden_dimensions[step_index],route_name,
        &endpoint);
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

static SparkStatus SparkTpDeviceCollectiveOpenStep(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t step_index)
{
    uint32_t partner_rank;
    SparkStatus status;

    partner_rank = config->tp_rank ^ (1u << step_index);
    if (config->tp_rank < partner_rank)
    {
        status = SparkTpDeviceCollectiveOpenSession(
            implementation,config,step_index,config->tp_rank,partner_rank,
            implementation->send_route_names[step_index],
            &implementation->send_sessions[step_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkTpDeviceCollectiveOpenSession(
            implementation,config,step_index,partner_rank,config->tp_rank,
            implementation->receive_route_names[step_index],
            &implementation->receive_sessions[step_index]);
    }
    status = SparkTpDeviceCollectiveOpenSession(
        implementation,config,step_index,partner_rank,config->tp_rank,
        implementation->receive_route_names[step_index],
        &implementation->receive_sessions[step_index]);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkTpDeviceCollectiveOpenSession(
        implementation,config,step_index,config->tp_rank,partner_rank,
        implementation->send_route_names[step_index],
        &implementation->send_sessions[step_index]);
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

static SparkStatus SparkTpDeviceCollectiveBuildOperationPackets(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveOperation *operation,
    uint32_t step_index,
    SparkHiddenTransportPacket *send_packet,
    SparkHiddenTransportPacket *receive_packet)
{
    const SparkTpDeviceCollectiveCreditBinding *binding;
    SparkStatus status;

    binding = &implementation->bindings[step_index][operation->credit_index];
    status = SparkTpDeviceCollectiveBuildPacket(binding->send_transport,
        operation->active_sequence_count,
        implementation->step_hidden_dimensions[step_index],
        operation->ordinal,step_index,operation->cuda_stream,
        send_packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkTpDeviceCollectiveBuildPacket(binding->receive_transport,
        operation->active_sequence_count,
        implementation->step_hidden_dimensions[step_index],
        operation->ordinal,step_index,operation->cuda_stream,
        receive_packet);
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

static SparkStatus SparkTpDeviceCollectiveStageReceive(
    const SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveCreditBinding *binding,
    uint64_t pitch,
    uint64_t width,
    uint32_t rows,
    void *cuda_stream)
{
    enum cudaMemcpyKind copy_kind;

    if (binding->receive_device == binding->receive_transport)
    {
        return SPARK_STATUS_OK;
    }
    copy_kind = collective->memory_mode ==
        SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
        cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    return SparkTpDeviceCollectiveCopyRows(binding->receive_device,pitch,
        binding->receive_transport,pitch,width,rows,copy_kind,cuda_stream);
}

static SparkStatus SparkTpDeviceCollectiveEnqueueLocalPlacement(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    const SparkTpDeviceCollectiveCreditBinding *binding;
    uint64_t full_pitch;
    uint64_t local_bytes;
    uint64_t rank_offset;
    SparkStatus status;

    local_bytes = (uint64_t)implementation->collective->local_hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    if (implementation->collective->operation_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16)
    {
        full_pitch = local_bytes;
        rank_offset = 0u;
    }
    else
    {
        full_pitch = local_bytes * implementation->collective->tp_degree;
        rank_offset = local_bytes * implementation->collective->tp_rank;
    }
    status = SPARK_STATUS_OK;
    if ((const void *)((uint8_t *)operation->full_device + rank_offset) !=
        operation->local_device)
    {
        status = SparkTpDeviceCollectiveCopyRows(
            (uint8_t *)operation->full_device + rank_offset,full_pitch,
            operation->local_device,local_bytes,local_bytes,
            operation->active_sequence_count,cudaMemcpyDeviceToDevice,
            operation->cuda_stream);
    }
    if (status != SPARK_STATUS_OK || implementation->collective->step_count == 0u)
    {
        return status;
    }
    binding = &implementation->bindings[0u][operation->credit_index];
    status = SparkTpDeviceCollectiveCopyRows(binding->send_device,
        local_bytes,
        (const uint8_t *)operation->full_device + rank_offset,full_pitch,
        local_bytes,operation->active_sequence_count,
        cudaMemcpyDeviceToDevice,operation->cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkTpDeviceCollectiveStageSend(implementation->collective,
        binding,local_bytes,local_bytes,operation->active_sequence_count,
        operation->cuda_stream);
}

static uint32_t SparkTpDeviceCollectiveBlockBase(
    const SparkTpDeviceCollective *collective,
    uint32_t step_index,
    uint32_t peer_block)
{
    uint32_t group_base;
    uint32_t half_count;
    uint32_t own_half_base;

    half_count = 1u << step_index;
    group_base = collective->tp_rank & ~((half_count << 1u) - 1u);
    own_half_base = group_base +
        ((collective->tp_rank & half_count) != 0u ? half_count : 0u);
    return peer_block != 0u ? own_half_base ^ half_count : own_half_base;
}

static SparkStatus SparkTpDeviceCollectiveEnqueueReceiveConsumption(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    const SparkTpDeviceCollectiveCreditBinding *binding;
    uint64_t full_pitch;
    uint64_t local_bytes;
    uint64_t step_bytes;
    uint64_t destination_offset;
    SparkStatus status;

    binding = &implementation->bindings[operation->current_step]
        [operation->credit_index];
    local_bytes = (uint64_t)implementation->collective->local_hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    step_bytes = implementation->collective->operation_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 ?
        local_bytes : local_bytes << operation->current_step;
    status = SparkTpDeviceCollectiveStageReceive(
        implementation->collective,binding,step_bytes,step_bytes,
        operation->active_sequence_count,operation->cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (implementation->collective->operation_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16)
    {
        status = implementation->combine_bf16_function(
            implementation->combine_context,operation->full_device,
            binding->receive_device,operation->active_sequence_count,
            implementation->collective->local_hidden_dimension,
            operation->cuda_stream);
    }
    else
    {
        full_pitch = local_bytes * implementation->collective->tp_degree;
        destination_offset = local_bytes * SparkTpDeviceCollectiveBlockBase(
            implementation->collective,operation->current_step,1u);
        status = SparkTpDeviceCollectiveCopyRows(
            (uint8_t *)operation->full_device + destination_offset,full_pitch,
            binding->receive_device,step_bytes,step_bytes,
            operation->active_sequence_count,cudaMemcpyDeviceToDevice,
            operation->cuda_stream);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkTpDeviceCollectiveCudaStatus(cudaEventRecord(
        implementation->consumer_events[operation->credit_index],
        (cudaStream_t)operation->cuda_stream));
}

static SparkStatus SparkTpDeviceCollectiveEnqueueNextSendPack(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint32_t next_step)
{
    uint64_t full_pitch;
    uint64_t local_bytes;
    uint64_t step_bytes;
    uint64_t source_offset;
    SparkStatus status;

    local_bytes = (uint64_t)implementation->collective->local_hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    if (implementation->collective->operation_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16)
    {
        full_pitch = local_bytes;
        step_bytes = local_bytes;
        source_offset = 0u;
    }
    else
    {
        full_pitch = local_bytes * implementation->collective->tp_degree;
        step_bytes = local_bytes << next_step;
        source_offset = local_bytes * SparkTpDeviceCollectiveBlockBase(
            implementation->collective,next_step,0u);
    }
    status = SparkTpDeviceCollectiveCopyRows(
        implementation->bindings[next_step][operation->credit_index]
            .send_device,
        step_bytes,(const uint8_t *)operation->full_device + source_offset,
        full_pitch,step_bytes,operation->active_sequence_count,
        cudaMemcpyDeviceToDevice,operation->cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkTpDeviceCollectiveStageSend(implementation->collective,
        &implementation->bindings[next_step][operation->credit_index],
        step_bytes,step_bytes,operation->active_sequence_count,
        operation->cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkTpDeviceCollectiveCudaStatus(cudaEventRecord(
        implementation->consumer_events[operation->credit_index],
        (cudaStream_t)operation->cuda_stream));
}

static SparkStatus SparkTpDeviceCollectiveMarkOperationFailure(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint64_t required_generation,
    SparkStatus status)
{
    uint64_t observed_state;
    uint64_t desired_state;
    uint32_t hook_called;

    if (implementation == 0 || operation == 0 ||
        status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY ||
        status == SPARK_STATUS_PENDING ||
        (uint32_t)status > SPARK_STATUS_UNSUPPORTED)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    hook_called = 0u;
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
        if (hook_called == 0u &&
            phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE &&
            implementation->debug_hooks.failure_observed_function != 0)
        {
            hook_called = 1u;
            implementation->debug_hooks.failure_observed_function(
                implementation->debug_hooks.hook_context,
                operation->credit_index,observed_state);
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

static void SparkTpDeviceCollectiveRouteCompletion(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkHiddenTransportCompletion *completion,
    uint32_t receive_completion)
{
    SparkTpDeviceCollectiveOperation *operation;
    uint64_t ordinal;
    uint64_t state_word;
    uint64_t generation;
    uint32_t credit_index;
    uint32_t step_index;

    if (completion->status == SPARK_STATUS_BUSY)
    {
        return;
    }
    if (completion->sequence_id == 0u ||
        completion->token_index >= implementation->collective->step_count)
    {
        SparkTpDeviceCollectiveLatchFailure(
            implementation,SPARK_STATUS_VALIDATION_FAILED);
        return;
    }
    ordinal = completion->sequence_id - 1u;
    credit_index = (uint32_t)(ordinal %
        implementation->collective->credit_count);
    generation = ordinal / implementation->collective->credit_count + 1u;
    operation = &implementation->operations[credit_index];
    state_word = atomic_load_explicit(
        &operation->lifecycle,memory_order_acquire);
    if (SparkTpDeviceCollectiveStateGeneration(state_word) != generation ||
        SparkTpDeviceCollectiveStatePhase(state_word) ==
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE)
    {
        SparkTpDeviceCollectiveLatchFailure(
            implementation,SPARK_STATUS_VALIDATION_FAILED);
        return;
    }
    step_index = (uint32_t)completion->token_index;
    if (completion->status != SPARK_STATUS_OK)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,generation,completion->status);
    }
    if (receive_completion != 0u)
    {
        operation->receive_complete_mask |= 1u << step_index;
        if (implementation->profile_enabled != 0u &&
            operation->profile_receive_complete_ns[step_index] == 0u)
        {
            operation->profile_receive_complete_ns[step_index] =
                SparkTpDeviceCollectiveNowNano();
        }
    }
    else
    {
        operation->send_complete_mask |= 1u << step_index;
        if (implementation->profile_enabled != 0u &&
            operation->profile_send_complete_ns[step_index] == 0u)
        {
            operation->profile_send_complete_ns[step_index] =
                SparkTpDeviceCollectiveNowNano();
            operation->profile_send_service_ns[step_index] =
                completion->service_time_ns;
        }
    }
}

static void SparkTpDeviceCollectivePollSession(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkHiddenTransportSession *session,
    uint32_t receive_completion)
{
    uint32_t completion_count;

    for (completion_count = 0u;
         completion_count < implementation->collective->credit_count;
         ++completion_count)
    {
        SparkHiddenTransportCompletion completion;
        SparkStatus status;

        status = SparkHiddenTransportPoll(session,&completion);
        if (status != SPARK_STATUS_OK)
        {
            SparkTpDeviceCollectiveLatchFailure(implementation,status);
            return;
        }
        if (completion.status == SPARK_STATUS_BUSY)
        {
            return;
        }
        SparkTpDeviceCollectiveRouteCompletion(
            implementation,&completion,receive_completion);
    }
}

static void SparkTpDeviceCollectivePollTransport(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint32_t step_index;

    for (step_index = 0u;
         step_index < implementation->collective->step_count;
         ++step_index)
    {
        SparkTpDeviceCollectivePollSession(
            implementation,implementation->send_sessions[step_index],0u);
        SparkTpDeviceCollectivePollSession(
            implementation,implementation->receive_sessions[step_index],1u);
    }
}

static void SparkTpDeviceCollectiveCancelUnsentResources(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    uint32_t step_index;

    for (step_index = 0u;
         step_index < implementation->collective->step_count;
         ++step_index)
    {
        uint32_t mask;

        mask = 1u << step_index;
        if ((operation->reserved_send_mask & mask) != 0u &&
            (operation->sent_mask & mask) == 0u &&
            SparkHiddenTransportCancelPersistentSend(
                implementation->send_sessions[step_index],
                operation->credit_index,operation->generation) ==
                    SPARK_STATUS_OK)
        {
            operation->reserved_send_mask &= ~mask;
        }
        if ((operation->activated_receive_mask & mask) != 0u &&
            (operation->receive_complete_mask & mask) == 0u &&
            SparkHiddenTransportCancelPersistentReceive(
                implementation->receive_sessions[step_index],
                operation->credit_index,operation->generation) ==
                    SPARK_STATUS_OK)
        {
            operation->activated_receive_mask &= ~mask;
        }
    }
}

static void SparkTpDeviceCollectiveReserveOperation(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint64_t state_word)
{
    uint32_t all_step_mask;
    uint32_t step_index;

    all_step_mask = SparkTpDeviceCollectiveAllStepMask(
        implementation->collective);
    if (SparkTpDeviceCollectiveStateHasFailure(state_word) != 0u)
    {
        SparkTpDeviceCollectiveCancelUnsentResources(
            implementation,operation);
        if (operation->reserved_send_mask == 0u &&
            operation->activated_receive_mask == 0u)
        {
            operation->terminal_after_consume = 1u;
            (void)SparkTpDeviceCollectiveTransitionPhase(operation,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE);
        }
        return;
    }
    for (step_index = 0u;
         step_index < implementation->collective->step_count;
         ++step_index)
    {
        SparkHiddenTransportPacket receive_packet;
        SparkHiddenTransportPacket send_packet;
        SparkStatus status;
        uint32_t mask;

        mask = 1u << step_index;
        status = SparkTpDeviceCollectiveBuildOperationPackets(
            implementation,operation,step_index,&send_packet,&receive_packet);
        if (status != SPARK_STATUS_OK)
        {
            (void)SparkTpDeviceCollectiveMarkOperationFailure(
                implementation,operation,operation->generation,status);
            return;
        }
        if ((operation->reserved_send_mask & mask) == 0u)
        {
            status = SparkHiddenTransportReservePersistentSend(
                implementation->send_sessions[step_index],
                operation->credit_index,operation->generation,&send_packet);
            if (status == SPARK_STATUS_OK)
            {
                operation->reserved_send_mask |= mask;
            }
            else if (status != SPARK_STATUS_BUSY)
            {
                (void)SparkTpDeviceCollectiveMarkOperationFailure(
                    implementation,operation,operation->generation,status);
                return;
            }
        }
        if ((operation->activated_receive_mask & mask) == 0u)
        {
            status = SparkHiddenTransportActivatePersistentReceive(
                implementation->receive_sessions[step_index],
                operation->credit_index,operation->generation,&receive_packet);
            if (status == SPARK_STATUS_OK)
            {
                operation->activated_receive_mask |= mask;
            }
            else if (status != SPARK_STATUS_BUSY)
            {
                (void)SparkTpDeviceCollectiveMarkOperationFailure(
                    implementation,operation,operation->generation,status);
                return;
            }
        }
    }
    if (operation->reserved_send_mask == all_step_mask &&
        operation->activated_receive_mask == all_step_mask)
    {
        if (implementation->profile_enabled != 0u &&
            operation->profile_reserved_ns == 0u)
        {
            operation->profile_reserved_ns =
                SparkTpDeviceCollectiveNowNano();
        }
        (void)SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING);
    }
}

static void SparkTpDeviceCollectiveBuildSend(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint64_t state_word)
{
    SparkHiddenTransportPacket receive_packet;
    SparkHiddenTransportPacket send_packet;
    SparkStatus status;
    uint32_t mask;

    if (SparkTpDeviceCollectiveStateHasFailure(state_word) != 0u)
    {
        SparkTpDeviceCollectiveCancelUnsentResources(
            implementation,operation);
        operation->terminal_after_consume = 1u;
        (void)SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE);
        return;
    }
    mask = 1u << operation->current_step;
    status = SparkTpDeviceCollectiveBuildOperationPackets(
        implementation,operation,operation->current_step,
        &send_packet,&receive_packet);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkHiddenTransportSendPersistent(
            implementation->send_sessions[operation->current_step],
            operation->credit_index,operation->generation,&send_packet);
    }
    if (status == SPARK_STATUS_BUSY)
    {
        return;
    }
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,operation->generation,status);
        return;
    }
    operation->sent_mask |= mask;
    if (implementation->profile_enabled != 0u)
    {
        operation->profile_send_done_ns[operation->current_step] =
            SparkTpDeviceCollectiveNowNano();
    }
    (void)SparkTpDeviceCollectiveTransitionPhase(operation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_TRANSFER_ACTIVE);
}

static void SparkTpDeviceCollectiveProgressTransfer(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint64_t state_word)
{
    uint32_t mask;

    mask = 1u << operation->current_step;
    if (SparkTpDeviceCollectiveStateHasFailure(state_word) != 0u)
    {
        SparkTpDeviceCollectiveCancelUnsentResources(
            implementation,operation);
        (void)SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_TRANSFER_ACTIVE,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
        return;
    }
    if ((operation->send_complete_mask & mask) == 0u ||
        (operation->receive_complete_mask & mask) == 0u)
    {
        if (SparkTpDeviceCollectiveNowMilli() >= operation->deadline_milli)
        {
            (void)SparkTpDeviceCollectiveMarkOperationFailure(
                implementation,operation,operation->generation,
                SPARK_STATUS_IO_ERROR);
        }
        return;
    }
    if (implementation->profile_enabled != 0u)
    {
        operation->profile_receive_done_ns[operation->current_step] =
            SparkTpDeviceCollectiveNowNano();
    }
    (void)SparkTpDeviceCollectiveTransitionPhase(operation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_TRANSFER_ACTIVE,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_BUILDING);
}

static void SparkTpDeviceCollectiveBuildConsumption(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    SparkStatus status;

    status = SparkTpDeviceCollectiveEnqueueReceiveConsumption(
        implementation,operation);
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,operation->generation,status);
        return;
    }
    (void)SparkTpDeviceCollectiveTransitionPhase(operation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_BUILDING,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE);
}

static void SparkTpDeviceCollectiveProgressConsumption(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation,
    uint64_t state_word)
{
    SparkStatus status;
    uint32_t mask;

    status = SparkTpDeviceCollectiveCudaStatus(cudaEventQuery(
        implementation->consumer_events[operation->credit_index]));
    if (status == SPARK_STATUS_BUSY)
    {
        return;
    }
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,operation->generation,status);
        return;
    }
    if (implementation->profile_enabled != 0u &&
        operation->profile_consume_done_ns[operation->current_step] == 0u)
    {
        operation->profile_consume_done_ns[operation->current_step] =
            SparkTpDeviceCollectiveNowNano();
    }
    if (operation->terminal_after_consume != 0u)
    {
        (void)SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
        return;
    }
    mask = 1u << operation->current_step;
    if ((operation->released_receive_mask & mask) == 0u)
    {
        status = SparkHiddenTransportReleasePersistentReceive(
            implementation->receive_sessions[operation->current_step],
            operation->credit_index,operation->generation,
            operation->cuda_stream);
        if (status == SPARK_STATUS_BUSY)
        {
            return;
        }
        if (status != SPARK_STATUS_OK)
        {
            (void)SparkTpDeviceCollectiveMarkOperationFailure(
                implementation,operation,operation->generation,status);
            return;
        }
        operation->released_receive_mask |= mask;
        operation->activated_receive_mask &= ~mask;
        if (implementation->profile_enabled != 0u)
        {
            operation->profile_release_done_ns[operation->current_step] =
                SparkTpDeviceCollectiveNowNano();
        }
    }
    if (SparkTpDeviceCollectiveStateHasFailure(state_word) != 0u ||
        operation->current_step + 1u == implementation->collective->step_count)
    {
        SparkTpDeviceCollectiveCancelUnsentResources(
            implementation,operation);
        (void)SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
        return;
    }
    status = SparkTpDeviceCollectiveEnqueueNextSendPack(
        implementation,operation,operation->current_step + 1u);
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,operation->generation,status);
        return;
    }
    operation->current_step += 1u;
    (void)SparkTpDeviceCollectiveTransitionPhase(operation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING);
}

static void SparkTpDeviceCollectiveReportProfile(
    const SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveOperation *operation,
    SparkStatus status)
{
    char message[1536];
    uint64_t start_ns;
    uint32_t offset,step_index;
    int count;

    if (implementation == 0 || operation == 0 ||
        implementation->profile_enabled == 0u ||
        operation->profile_submit_ns == 0u)
    {
        return;
    }
    start_ns = operation->profile_submit_ns;
    count = snprintf(message,sizeof(message),
        "sparkpipe_tp_collective_profile tp_rank=%u ordinal=%llu rows=%u "
        "steps=%u status=%u reserved_ns=%llu",
        implementation->collective->tp_rank,
        (unsigned long long)operation->ordinal,
        operation->active_sequence_count,
        implementation->collective->step_count,(uint32_t)status,
        (unsigned long long)SparkTpDeviceCollectiveProfileDelta(
            start_ns,operation->profile_reserved_ns));
    if (count < 0 || (uint32_t)count >= sizeof(message))
        return;
    offset = (uint32_t)count;
    for (step_index=0u;
         step_index<implementation->collective->step_count; step_index++)
    {
        count = snprintf(message+offset,sizeof(message)-offset,
            " send_done%u_ns=%llu send_complete%u_ns=%llu "
            "receive_complete%u_ns=%llu send_service%u_ns=%llu "
            "receive_done%u_ns=%llu consume_done%u_ns=%llu "
            "release_done%u_ns=%llu",
            step_index,(unsigned long long)SparkTpDeviceCollectiveProfileDelta(start_ns,operation->profile_send_done_ns[step_index]),
            step_index,(unsigned long long)SparkTpDeviceCollectiveProfileDelta(start_ns,operation->profile_send_complete_ns[step_index]),
            step_index,(unsigned long long)SparkTpDeviceCollectiveProfileDelta(start_ns,operation->profile_receive_complete_ns[step_index]),
            step_index,(unsigned long long)operation->profile_send_service_ns[step_index],
            step_index,(unsigned long long)SparkTpDeviceCollectiveProfileDelta(start_ns,operation->profile_receive_done_ns[step_index]),
            step_index,(unsigned long long)SparkTpDeviceCollectiveProfileDelta(start_ns,operation->profile_consume_done_ns[step_index]),
            step_index,(unsigned long long)SparkTpDeviceCollectiveProfileDelta(start_ns,operation->profile_release_done_ns[step_index]));
        if (count < 0 || (uint32_t)count >= sizeof(message)-offset)
            return;
        offset += (uint32_t)count;
    }
    count = snprintf(message+offset,sizeof(message)-offset,
        " complete_ns=%llu\n",(unsigned long long)SparkTpDeviceCollectiveProfileDelta(start_ns,operation->profile_complete_ns));
    if (count < 0 || (uint32_t)count >= sizeof(message)-offset)
        return;
    offset += (uint32_t)count;
    (void)fwrite(message,1u,offset,stderr);
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
    if (implementation->profile_enabled != 0u)
    {
        operation->profile_complete_ns = SparkTpDeviceCollectiveNowNano();
    }
    operation->completion_function(operation->completion_context,&completion);
    SparkTpDeviceCollectiveReportProfile(implementation,operation,status);
    (void)SparkTpDeviceCollectiveTransitionPhase(operation,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_CALLBACK_CLAIMED,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_RELEASE_PENDING);
}

static void SparkTpDeviceCollectiveReleaseOperation(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveOperation *operation)
{
    uint32_t step_index;
    uint32_t pending;

    SparkTpDeviceCollectiveCancelUnsentResources(implementation,operation);
    pending = 0u;
    for (step_index = 0u;
         step_index < implementation->collective->step_count;
         ++step_index)
    {
        SparkStatus status;
        uint32_t mask;

        mask = 1u << step_index;
        if ((operation->activated_receive_mask & mask) == 0u ||
            (operation->released_receive_mask & mask) != 0u)
        {
            continue;
        }
        if ((operation->receive_complete_mask & mask) == 0u)
        {
            pending = 1u;
            continue;
        }
        status = SparkHiddenTransportReleasePersistentReceive(
            implementation->receive_sessions[step_index],
            operation->credit_index,operation->generation,
            operation->cuda_stream);
        if (status == SPARK_STATUS_OK)
        {
            operation->released_receive_mask |= mask;
            operation->activated_receive_mask &= ~mask;
        }
        else if (status == SPARK_STATUS_BUSY)
        {
            pending = 1u;
        }
        else
        {
            SparkTpDeviceCollectiveLatchFailure(implementation,status);
            pending = 1u;
        }
    }
    if (pending != 0u || operation->activated_receive_mask != 0u ||
        (operation->sent_mask & ~operation->send_complete_mask) != 0u ||
        operation->reserved_send_mask != operation->sent_mask)
    {
        return;
    }
    atomic_store_explicit(&operation->lifecycle,
        SparkTpDeviceCollectiveStateWord(operation->generation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE,0u),
        memory_order_release);
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
    if (phase >= SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE &&
        phase <= SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE &&
        SparkTpDeviceCollectiveNowMilli() >= operation->deadline_milli)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,operation->generation,
            SPARK_STATUS_IO_ERROR);
        state_word = atomic_load_explicit(
            &operation->lifecycle,memory_order_acquire);
        phase = SparkTpDeviceCollectiveStatePhase(state_word);
    }
    switch (phase)
    {
        case SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE:
            SparkTpDeviceCollectiveReserveOperation(
                implementation,operation,state_word);
            break;
        case SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING:
            SparkTpDeviceCollectiveBuildSend(
                implementation,operation,state_word);
            break;
        case SPARK_TP_DEVICE_COLLECTIVE_PHASE_TRANSFER_ACTIVE:
            SparkTpDeviceCollectiveProgressTransfer(
                implementation,operation,state_word);
            break;
        case SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_BUILDING:
            SparkTpDeviceCollectiveBuildConsumption(
                implementation,operation);
            break;
        case SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE:
            SparkTpDeviceCollectiveProgressConsumption(
                implementation,operation,state_word);
            break;
        case SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY:
            SparkTpDeviceCollectivePublishCompletion(
                implementation,operation);
            break;
        case SPARK_TP_DEVICE_COLLECTIVE_PHASE_RELEASE_PENDING:
            SparkTpDeviceCollectiveReleaseOperation(
                implementation,operation);
            break;
        default:
            break;
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
    uint32_t iteration;

    implementation = (SparkTpDeviceCollectiveImplementation *)context;
    iteration = 0u;
    for (;;)
    {
        SparkTpDeviceCollectivePollTransport(implementation);
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
        iteration += 1u;
        if ((iteration %
                SPARK_TP_DEVICE_COLLECTIVE_PROGRESS_YIELD_INTERVAL) == 0u)
        {
            sched_yield();
        }
    }
    return 0;
}

static SparkStatus SparkTpDeviceCollectiveRegisterCredits(
    SparkTpDeviceCollectiveImplementation *implementation,
    uint32_t timeout_milli)
{
    uint64_t deadline_milli;
    uint32_t credit_index;
    uint32_t step_index;

    for (step_index = 0u;
         step_index < implementation->collective->step_count;
         ++step_index)
    {
        for (credit_index = 0u;
             credit_index < implementation->collective->credit_count;
             ++credit_index)
        {
            const SparkTpDeviceCollectiveCreditBinding *binding;
            SparkHiddenTransportPacket packet;
            SparkStatus status;

            binding = &implementation->bindings[step_index][credit_index];
            status = SparkTpDeviceCollectiveBuildPacket(
                binding->receive_transport,
                implementation->collective->max_active_sequence_count,
                implementation->step_hidden_dimensions[step_index],
                0u,step_index,implementation->registration_cuda_stream,
                &packet);
            if (status == SPARK_STATUS_OK)
            {
                status = SparkHiddenTransportRegisterPersistentReceive(
                    implementation->receive_sessions[step_index],
                    credit_index,&packet);
            }
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    deadline_milli = SparkTpDeviceCollectiveNowMilli();
    if (deadline_milli == UINT64_MAX)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    deadline_milli += timeout_milli;
    for (;;)
    {
        uint32_t ready_count;

        ready_count = 0u;
        for (step_index = 0u;
             step_index < implementation->collective->step_count;
             ++step_index)
        {
            for (credit_index = 0u;
                 credit_index < implementation->collective->credit_count;
                 ++credit_index)
            {
                SparkStatus status;

                status = SparkHiddenTransportPersistentRemoteCreditReady(
                    implementation->send_sessions[step_index],credit_index);
                if (status == SPARK_STATUS_OK)
                {
                    ready_count += 1u;
                }
                else if (status != SPARK_STATUS_BUSY)
                {
                    return status;
                }
            }
        }
        if (ready_count == implementation->collective->step_count *
                implementation->collective->credit_count)
        {
            return SPARK_STATUS_OK;
        }
        if (SparkTpDeviceCollectiveNowMilli() >= deadline_milli)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        sched_yield();
    }
}

SparkStatus SparkTpDeviceCollectiveProbeMemoryMode(
    const char *transport_module_path,
    uint32_t *memory_mode_out)
{
    SparkHiddenTransportDynamicLibrary library;
    SparkStatus status;

    if (!SparkTpDeviceCollectiveTextIsValid(transport_module_path) ||
        memory_mode_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&library,0,sizeof(library));
    status = SparkHiddenTransportLoadInterfaceFromSharedObject(
        transport_module_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS |
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS,
        &library);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((library.transport_interface.capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_GPUDIRECT_RDMA) != 0u)
    {
        *memory_mode_out = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
        status = SPARK_STATUS_OK;
    }
    else if ((library.transport_interface.capability_flags &
                (SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA |
                 SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY)) ==
            (SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA |
             SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY))
    {
        *memory_mode_out = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST;
        status = SPARK_STATUS_OK;
    }
    else
    {
        status = SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkHiddenTransportUnloadInterface(&library);
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
    uint32_t step_count;
    uint32_t step_index;

    if (collective_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(collective_out, 0, sizeof(*collective_out));
    status = SparkTpDeviceCollectiveValidateConfig(
        config,&step_count,&credit_count);
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
    collective_out->tp_degree = config->tp_degree;
    collective_out->tp_rank = config->tp_rank;
    collective_out->step_count = step_count;
    collective_out->operation_kind = config->operation_kind;
    collective_out->credit_count = credit_count;
    collective_out->local_hidden_dimension = config->local_hidden_dimension;
    collective_out->max_active_sequence_count =
        config->max_active_sequence_count;
    collective_out->operation_timeout_milli = config->operation_timeout_milli;
    collective_out->collective_identifier = config->collective_identifier;
    collective_out->implementation = implementation;
    implementation->collective = collective_out;
    implementation->registration_cuda_stream =
        config->registration_cuda_stream;
    implementation->combine_bf16_function = config->combine_bf16_function;
    implementation->combine_context = config->combine_context;
    implementation->profile_enabled =
        SparkTpDeviceCollectiveProfileIsEnabled();
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
        implementation->bindings[binding->step_index][binding->credit_index] =
            *binding;
    }
    for (step_index = 0u; step_index < step_count; ++step_index)
    {
        uint64_t step_hidden_dimension;

        step_hidden_dimension = config->operation_kind ==
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 ?
            config->local_hidden_dimension :
            (uint64_t)config->local_hidden_dimension << step_index;
        if (step_hidden_dimension > UINT32_MAX)
        {
            status = SPARK_STATUS_CAPACITY_EXCEEDED;
            goto fail_create;
        }
        implementation->step_hidden_dimensions[step_index] =
            (uint32_t)step_hidden_dimension;
    }
    if (step_count == 0u)
    {
        collective_out->memory_mode =
            SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
    }
    else
    {
        status = SparkHiddenTransportLoadInterfaceFromSharedObject(
            config->transport_module_path,
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS |
                SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS,
            &implementation->transport_library);
        if (status != SPARK_STATUS_OK)
        {
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
        for (step_index = 0u; step_index < step_count; ++step_index)
        {
            status = SparkTpDeviceCollectiveOpenStep(
                implementation,config,step_index);
            if (status != SPARK_STATUS_OK)
            {
                goto fail_create;
            }
        }
        status = SparkTpDeviceCollectiveRegisterCredits(
            implementation,config->connect_timeout_milli);
        if (status != SPARK_STATUS_OK)
        {
            goto fail_create;
        }
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
    free(implementation);
    memset(collective_out,0,sizeof(*collective_out));
    return status;
}

SparkStatus SparkTpDeviceCollectiveSubmitBf16(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    SparkTpDeviceCollectiveOperation *operation;
    SparkStatus status;
    uint64_t desired_state;
    uint64_t expected_state;
    uint64_t generation;
    uint64_t now_milli;
    uint32_t credit_index;

    if (collective == 0 || collective->abi_version !=
            SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        collective->implementation == 0 || submission == 0 ||
        submission->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        submission->descriptor_bytes != sizeof(*submission) ||
        submission->active_sequence_count == 0u ||
        submission->active_sequence_count >
            collective->max_active_sequence_count ||
        submission->local_device == 0 || submission->full_device == 0 ||
        submission->cuda_stream == 0 ||
        submission->completion_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    implementation = (SparkTpDeviceCollectiveImplementation *)
        collective->implementation;
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
#if defined(__GNUC__)
    if (SparkTpDeviceCollectiveDebugSubmissionClaimed != 0)
    {
        SparkTpDeviceCollectiveDebugSubmissionClaimed(
            credit_index,generation);
    }
#endif
    now_milli = SparkTpDeviceCollectiveNowMilli();
    if (now_milli == UINT64_MAX)
    {
        now_milli = 0u;
    }
    operation->slot_index = submission->slot_index;
    operation->credit_index = credit_index;
    operation->active_sequence_count = submission->active_sequence_count;
    operation->current_step = 0u;
    operation->reserved_send_mask = 0u;
    operation->activated_receive_mask = 0u;
    operation->sent_mask = 0u;
    operation->send_complete_mask = 0u;
    operation->receive_complete_mask = 0u;
    operation->released_receive_mask = 0u;
    operation->terminal_after_consume = 0u;
    operation->ordinal = submission->ordinal;
    operation->generation = generation;
    operation->deadline_milli = now_milli +
        collective->operation_timeout_milli;
    operation->local_device = submission->local_device;
    operation->full_device = submission->full_device;
    operation->cuda_stream = submission->cuda_stream;
    operation->completion_function = submission->completion_function;
    operation->completion_context = submission->completion_context;
    operation->profile_submit_ns = implementation->profile_enabled != 0u ?
        SparkTpDeviceCollectiveNowNano() : 0u;
    operation->profile_reserved_ns = 0u;
    memset(operation->profile_send_done_ns,0,
        sizeof(operation->profile_send_done_ns));
    memset(operation->profile_send_complete_ns,0,
        sizeof(operation->profile_send_complete_ns));
    memset(operation->profile_receive_complete_ns,0,
        sizeof(operation->profile_receive_complete_ns));
    memset(operation->profile_send_service_ns,0,
        sizeof(operation->profile_send_service_ns));
    memset(operation->profile_receive_done_ns,0,
        sizeof(operation->profile_receive_done_ns));
    memset(operation->profile_consume_done_ns,0,
        sizeof(operation->profile_consume_done_ns));
    memset(operation->profile_release_done_ns,0,
        sizeof(operation->profile_release_done_ns));
    operation->profile_complete_ns = 0u;
    status = now_milli == 0u ? SPARK_STATUS_IO_ERROR :
        SparkTpDeviceCollectiveEnqueueLocalPlacement(
            implementation,operation);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTpDeviceCollectiveCudaStatus(cudaEventRecord(
            implementation->consumer_events[credit_index],
            (cudaStream_t)operation->cuda_stream));
    }
    if (status != SPARK_STATUS_OK || collective->step_count == 0u)
    {
        if (status != SPARK_STATUS_OK)
        {
            (void)SparkTpDeviceCollectiveMarkOperationFailure(
                implementation,operation,generation,status);
        }
        operation->terminal_after_consume = 1u;
        if (status != SPARK_STATUS_OK && cudaEventRecord(
                implementation->consumer_events[credit_index],
                (cudaStream_t)operation->cuda_stream) != cudaSuccess)
        {
            (void)SparkTpDeviceCollectiveTransitionPhase(operation,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY);
        }
        else
        {
            (void)SparkTpDeviceCollectiveTransitionPhase(operation,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING,
                SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_ACTIVE);
        }
        return SPARK_STATUS_OK;
    }
    if (atomic_load_explicit(&implementation->admission_open,
            memory_order_acquire) == 0u)
    {
        (void)SparkTpDeviceCollectiveMarkOperationFailure(
            implementation,operation,generation,
            (SparkStatus)atomic_load_explicit(
                &implementation->failure_status,memory_order_acquire));
    }
    if (SparkTpDeviceCollectiveTransitionPhase(operation,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING,
            SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE) == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveRequestFailure(
    SparkTpDeviceCollective *collective,
    SparkStatus failure_status)
{
    SparkTpDeviceCollectiveImplementation *implementation;

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

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    if (collective == 0)
    {
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
    free(implementation);
    memset(collective,0,sizeof(*collective));
}
