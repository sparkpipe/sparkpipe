#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200112L

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_ring_node_context_builder.h"
#include "sparkpipe/spark_glm52_ring_runtime.h"
#include "sparkpipe/spark_glm52_ring_work_control.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"

#define SPARK_GLM52_RING_DAEMON_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_GLM52_RING_DAEMON_DEFAULT_PROGRAM "glm52.ring.rank.production"
#define SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY 64u
#define SPARK_GLM52_RING_DAEMON_WORK_QUEUE_HASH_SLOTS 4096u
#define SPARK_GLM52_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY 2048u
#define SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT UINT32_MAX
#define SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY 32u
#define SPARK_GLM52_RING_DAEMON_WORK_STATE_READY 0u
#define SPARK_GLM52_RING_DAEMON_WORK_STATE_WAITING_FORWARD 1u
#define SPARK_GLM52_RING_DAEMON_WORK_STATE_WAITING_SUBMIT 2u
#define SPARK_GLM52_RING_DAEMON_WORK_PHASE_PREFILL 0u
#define SPARK_GLM52_RING_DAEMON_WORK_PHASE_DECODE 1u
#define SPARK_GLM52_RING_DAEMON_WORK_PHASE_VERIFY 2u
#define SPARK_GLM52_RING_DAEMON_WORK_PHASE_RELEASE 3u
#define SPARK_GLM52_RING_DAEMON_DEPENDENCY_HASH_CAPACITY 2048u
#define SPARK_GLM52_RING_DAEMON_WORK_HASH_OFFSET UINT32_C(2166136261)
#define SPARK_GLM52_RING_DAEMON_WORK_HASH_PRIME UINT32_C(16777619)
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_WORK 0x00000001u
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_FINAL_EVENT 0x00000002u
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_INPUT_TRANSPORT 0x00000004u
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_COMPLETION_WAKE 0x00000010u
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_WORK_OUTPUT 0x00000020u
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_TIMER 0x00000040u
#define SPARK_GLM52_RING_DAEMON_POLL_KIND_CUDA_RESIDENT 0x00000080u
#define SPARK_GLM52_RING_DAEMON_CONNECT_RETRY_NS 250000ull
#define SPARK_GLM52_RING_DAEMON_RUNNER_PROGRESS_NS 250000ull

typedef struct SparkGlm52RingDaemonConfig
{
    const char *moe_pack_root;
    const char *stagepack_root;
    const char *transport_shared_object_path;
    const char *driver_path;
    const char *node_context_builder_shared_object_path;
    const char *embedding_pack_path;
    const char *cuda_resident_socket_path;
    const char *program_name;
    const char *node_target;
    const char *final_event_bind_address;
    const char *final_event_return_host;
    uint32_t rank_index;
    uint32_t rank_is_set;
    uint32_t own_final_event;
    uint32_t transport_busy_poll;
    uint32_t max_active_sequence_count;
    uint32_t port_base;
    uint32_t model_quantization_mode;
} SparkGlm52RingDaemonConfig;

typedef struct SparkGlm52RingDaemonRuntime
{
    SparkGlm52RingRuntimeRankPlan rank_plan;
    SparkGlm52RingRuntimeFinalEventRoute final_event_route;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *input_transport_session;
    SparkHiddenTransportSession *output_transport_session;
    SparkGlm52RingNodeContextBuilderDynamicLibrary builder_library;
    void *builder_state;
    SparkGlm52RingNodeContextBuilderResult builder_result;
    SparkLoadedModelDriver loaded_driver;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    int32_t cuda_resident_fd;
    uint32_t trace_enabled;
    const char *cuda_resident_socket_path;
    uint64_t cuda_resident_retry_after_ns;
    uint64_t cuda_resident_next_sequence_number;
    uint64_t cuda_resident_submit_count;
    uint64_t cuda_resident_completion_count;
    uint64_t cuda_resident_error_count;
    SparkGlm52CudaResidentIpcHeader cuda_resident_read_header;
    uint32_t cuda_resident_read_header_offset;
    uint32_t cuda_resident_read_payload_offset;
    uint8_t cuda_resident_payload[SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES];
    int32_t work_listen_fd;
    int32_t work_input_socket_fd;
    int32_t work_output_socket_fd;
    uint32_t work_output_connecting;
    uint32_t work_output_write_offset;
    uint64_t work_output_retry_mono_ns;
    int32_t final_event_listen_fd;
    int32_t final_event_socket_fd;
    uint32_t final_event_socket_connecting;
    uint32_t final_event_write_offset;
    uint64_t final_event_retry_mono_ns;
    uint8_t work_read_buffer[SPARK_GLM52_RING_WORK_CONTROL_PACKET_BYTES];
    uint32_t work_read_offset;
    SparkGlm52RingWorkControlPacket work_queue[SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_hash_heads[SPARK_GLM52_RING_DAEMON_WORK_QUEUE_HASH_SLOTS];
    uint32_t work_queue_hash_next[SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_submitted[SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_forwarded[SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_state[SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint64_t dependency_sequence_ids[
        SPARK_GLM52_RING_DAEMON_DEPENDENCY_HASH_CAPACITY];
    uint64_t dependency_sequence_positions[
        SPARK_GLM52_RING_DAEMON_DEPENDENCY_HASH_CAPACITY];
    uint32_t work_queue_head;
    uint32_t work_queue_count;
    uint64_t work_receive_count;
    uint64_t work_forward_count;
    uint64_t work_submit_count;
    uint64_t work_error_count;
    uint64_t work_duplicate_count;
    uint64_t work_deferred_count;
    uint64_t work_wake_count;
    uint64_t driver_completion_count;
    uint64_t driver_inflight_open_ns;
    uint32_t driver_inflight_warned;
    uint32_t driver_inflight_count;
    uint8_t final_event_read_buffer[
        SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES];
    uint32_t final_event_read_offset;
    SparkGlm52RingRuntimeFinalEvent final_event_queue[
        SPARK_GLM52_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY];
    uint32_t final_event_queue_head;
    uint32_t final_event_queue_count;
    int32_t wake_pipe_read_fd;
    int32_t wake_pipe_write_fd;
    uint64_t wake_signal_count;
    uint64_t wake_drop_count;
    uint64_t timer_wake_count;
    uint64_t final_event_send_count;
    uint64_t final_event_receive_count;
    uint64_t final_event_send_error_count;
    uint64_t final_event_receive_error_count;
    SparkGlm52DsparkDraftResult completion_dspark_draft;
    uint32_t completion_dspark_draft_valid;
} SparkGlm52RingDaemonRuntime;

static volatile sig_atomic_t SparkGlm52RingDaemonRunning = 1;

static void SparkGlm52RingDaemonCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion);

static void SparkGlm52RingDaemonSignal(int signal_number)
{
    (void)signal_number;
    SparkGlm52RingDaemonRunning = 0;
}

static void SparkGlm52RingDaemonInitializeConfig(
    SparkGlm52RingDaemonConfig *configuration)
{
    memset(configuration,0,sizeof(*configuration));
    configuration->program_name = SPARK_GLM52_RING_DAEMON_DEFAULT_PROGRAM;
    configuration->final_event_bind_address = "0.0.0.0";
    configuration->final_event_return_host = "spark0";
    configuration->max_active_sequence_count =
        SPARK_GLM52_RING_DAEMON_DEFAULT_MAX_ACTIVE;
    configuration->port_base = SPARK_GLM52_RING_RUNTIME_DEFAULT_PORT_BASE;
    configuration->model_quantization_mode =
        SPARK_GLM52_RING_RUNTIME_DEFAULT_QUANTIZATION_MODE;
}

static int32_t SparkGlm52RingDaemonParseU32(
    const char *text,
    uint32_t *value_out)
{
    uint64_t value;
    uint32_t index;

    if (text == 0 || text[0] == '\0' || value_out == 0)
        return -1;
    value = 0u;
    for (index = 0u; text[index] != '\0'; ++index)
    {
        if (text[index] < '0' || text[index] > '9')
            return -2;
        value = ((value * 10u) + (uint32_t)(text[index] - '0'));
        if (value > 0xffffffffull)
            return -3;
    }
    *value_out = (uint32_t)value;
    return 0;
}

static void SparkGlm52RingDaemonSetDefaultError(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    const char *message)
{
    if (error_buffer == 0 || error_buffer_bytes == 0u ||
        error_buffer[0] != '\0')
        return;
    snprintf(error_buffer,error_buffer_bytes,"%s",message);
}

static void SparkGlm52RingDaemonSetStatusError(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    const char *message,
    SparkStatus status)
{
    if (error_buffer == 0 || error_buffer_bytes == 0u ||
        error_buffer[0] != '\0')
        return;
    snprintf(error_buffer,error_buffer_bytes,"%s status=%d",
        message,(int32_t)status);
}

static int32_t SparkGlm52RingDaemonApplyArgument(
    SparkGlm52RingDaemonConfig *configuration,
    int argc,
    char **argv,
    int32_t *index)
{
    uint32_t parsed;

    if (strcmp(argv[*index],"--rank") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52RingDaemonParseU32(argv[*index + 1],&parsed) < 0)
            return -1;
        configuration->rank_index = parsed;
        configuration->rank_is_set = 1u;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--moe-pack-root") == 0)
    {
        if ((*index + 1) >= argc)
            return -2;
        configuration->moe_pack_root = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--model-quantization") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52RingRuntimeParseQuantizationMode(
                argv[*index + 1],&configuration->model_quantization_mode) !=
                SPARK_STATUS_OK)
            return -14;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--stagepack-root") == 0)
    {
        if ((*index + 1) >= argc)
            return -3;
        configuration->stagepack_root = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--transport-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -4;
        configuration->transport_shared_object_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--driver-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -5;
        configuration->driver_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--node-context-builder-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -6;
        configuration->node_context_builder_shared_object_path =
            argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--embedding-pack") == 0)
    {
        if ((*index + 1) >= argc)
            return -7;
        configuration->embedding_pack_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--cuda-resident-socket") == 0)
    {
        if ((*index + 1) >= argc)
            return -7;
        configuration->cuda_resident_socket_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--program") == 0)
    {
        if ((*index + 1) >= argc)
            return -8;
        configuration->program_name = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--node-target") == 0)
    {
        if ((*index + 1) >= argc)
            return -8;
        configuration->node_target = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--max-active") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52RingDaemonParseU32(argv[*index + 1],&parsed) < 0)
            return -9;
        configuration->max_active_sequence_count = parsed;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--port-base") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52RingDaemonParseU32(argv[*index + 1],&parsed) < 0)
            return -10;
        configuration->port_base = parsed;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--final-event-bind") == 0)
    {
        if ((*index + 1) >= argc)
            return -11;
        configuration->final_event_bind_address = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--final-event-return-host") == 0)
    {
        if ((*index + 1) >= argc)
            return -12;
        configuration->final_event_return_host = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--own-final-event") == 0)
    {
        configuration->own_final_event = 1u;
        return 0;
    }
    if (strcmp(argv[*index],"--transport-busy-poll") == 0)
    {
        configuration->transport_busy_poll = 1u;
        return 0;
    }
    return -13;
}

static int32_t SparkGlm52RingDaemonParseArguments(
    SparkGlm52RingDaemonConfig *configuration,
    int argc,
    char **argv)
{
    int32_t index;

    for (index = 1; index < argc; ++index)
    {
        if (SparkGlm52RingDaemonApplyArgument(
                configuration,argc,argv,&index) < 0)
            return -1;
    }
    if (configuration->rank_is_set == 0u)
        return -2;
    if (SparkGlm52RingRuntimeQuantizationModeName(
            configuration->model_quantization_mode) == 0)
        return -3;
    if (configuration->cuda_resident_socket_path != 0)
        return 0;
    if (configuration->moe_pack_root == 0 ||
        configuration->stagepack_root == 0 ||
        configuration->transport_shared_object_path == 0 ||
        configuration->driver_path == 0 ||
        configuration->node_context_builder_shared_object_path == 0 ||
        configuration->embedding_pack_path == 0)
        return -2;
    return 0;
}

static int32_t SparkGlm52RingDaemonCreateListenSocket(
    const char *bind_address,
    uint32_t port)
{
    struct sockaddr_in address;
    int32_t fd;
    int32_t option;

    fd = socket(AF_INET,SOCK_STREAM,0);
    if (fd < 0)
        return -1;
    option = 1;
    if (setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&option,sizeof(option)) < 0)
    {
        close(fd);
        return -2;
    }
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET,bind_address,&address.sin_addr) != 1)
    {
        close(fd);
        return -3;
    }
    if (bind(fd,(struct sockaddr *)&address,sizeof(address)) < 0)
    {
        close(fd);
        return -4;
    }
    if (listen(fd,64) < 0)
    {
        close(fd);
        return -5;
    }
    return fd;
}

static int32_t SparkGlm52RingDaemonSetNonblocking(int32_t fd)
{
    int32_t flags;

    flags = fcntl(fd,F_GETFL,0);
    if (flags < 0)
        return -1;
    if (fcntl(fd,F_SETFL,(flags | O_NONBLOCK)) < 0)
        return -2;
    return 0;
}

static uint64_t SparkGlm52RingDaemonMonotonicNs(void)
{
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0)
        return 0u;
    return ((uint64_t)timestamp.tv_sec * 1000000000ull) +
        (uint64_t)timestamp.tv_nsec;
}

static void SparkGlm52RingDaemonConfigureTcpSocket(int32_t fd)
{
    int32_t option;

    if (fd < 0)
        return;
    option = 1;
    (void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&option,sizeof(option));
    (void)setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&option,sizeof(option));
}

static uint64_t SparkGlm52RingDaemonMinNonzeroNs(
    uint64_t left,
    uint64_t right)
{
    if (left == 0u)
        return right;
    if (right == 0u)
        return left;
    return left < right ? left : right;
}

static uint32_t SparkGlm52RingDaemonHasWaitingWork(
    const SparkGlm52RingDaemonRuntime *runtime)
{
    uint32_t slot_index;
    uint32_t scan_index;

    if (runtime == 0 || runtime->work_queue_count == 0u)
        return 0u;
    slot_index = runtime->work_queue_head;
    for (scan_index = 0u;
         scan_index < runtime->work_queue_count;
         ++scan_index)
    {
        if (runtime->work_queue_state[slot_index] !=
            SPARK_GLM52_RING_DAEMON_WORK_STATE_READY)
            return 1u;
        slot_index = (slot_index + 1u) %
            SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
    }
    return 0u;
}

static uint64_t SparkGlm52RingDaemonNextTimerNs(
    const SparkGlm52RingDaemonRuntime *runtime)
{
    uint64_t next_ns;
    uint64_t now_ns;

    if (runtime == 0)
        return 0u;
    now_ns = SparkGlm52RingDaemonMonotonicNs();
    next_ns = 0u;
    if (runtime->driver_inflight_count != 0u)
        next_ns = now_ns + SPARK_GLM52_RING_DAEMON_RUNNER_PROGRESS_NS;
    if (SparkGlm52RingDaemonHasWaitingWork(runtime) != 0u)
        next_ns = SparkGlm52RingDaemonMinNonzeroNs(
            next_ns,
            now_ns + SPARK_GLM52_RING_DAEMON_RUNNER_PROGRESS_NS);
    if (runtime->work_queue_count != 0u)
        next_ns = SparkGlm52RingDaemonMinNonzeroNs(
            next_ns,
            runtime->work_output_retry_mono_ns);
    if (runtime->final_event_queue_count != 0u)
        next_ns = SparkGlm52RingDaemonMinNonzeroNs(
            next_ns,
            runtime->final_event_retry_mono_ns);
    return next_ns;
}

#if !defined(__linux__)
static int32_t SparkGlm52RingDaemonPollTimeoutMs(uint64_t timeout_ns)
{
    uint64_t timeout_ms;

    if (timeout_ns == 0u)
        return -1;
    timeout_ms = ((timeout_ns + 999999ull) / 1000000ull);
    if (timeout_ms == 0u)
        timeout_ms = 1u;
    if (timeout_ms > 0x7fffffffull)
        timeout_ms = 0x7fffffffull;
    return (int32_t)timeout_ms;
}
#endif

static int32_t SparkGlm52RingDaemonAppendPollFd(
    struct pollfd *fds,
    uint32_t *fd_kinds,
    uint32_t fd_capacity,
    uint32_t *fd_count,
    int32_t fd,
    int16_t events,
    uint32_t kind)
{
    if (fds == 0 || fd_count == 0)
        return -1;
    if (fd < 0 || events == 0)
        return 0;
    if (*fd_count >= fd_capacity)
        return -2;
    memset(&fds[*fd_count],0,sizeof(fds[*fd_count]));
    fds[*fd_count].fd = fd;
    fds[*fd_count].events = events;
    if (fd_kinds != 0)
        fd_kinds[*fd_count] = kind;
    *fd_count += 1u;
    return 0;
}

static int16_t SparkGlm52RingDaemonPollEventsFromTransport(
    uint32_t transport_events)
{
    int16_t events;
    events = 0;
    if ((transport_events & SPARK_HIDDEN_TRANSPORT_POLL_READ) != 0u)
        events |= POLLIN;
    if ((transport_events & SPARK_HIDDEN_TRANSPORT_POLL_WRITE) != 0u)
        events |= POLLOUT;
    return events;
}

static void SparkGlm52RingDaemonAppendTransportPollFds(
    SparkHiddenTransportSession *session,
    struct pollfd *fds,
    uint32_t *fd_kinds,
    uint32_t fd_capacity,
    uint32_t *fd_count,
    uint32_t kind)
{
    SparkHiddenTransportPollDescriptor descriptors[4];
    SparkStatus status;
    uint32_t descriptor_count;
    uint32_t descriptor_index;
    int16_t events;
    if (session == 0 || fds == 0 || fd_count == 0)
        return;
    descriptor_count = 0u;
    status = SparkHiddenTransportGetPollDescriptors(
        session,
        descriptors,
        4u,
        &descriptor_count);
    if (status != SPARK_STATUS_OK)
        return;
    for (descriptor_index = 0u;
         descriptor_index < descriptor_count;
         ++descriptor_index)
    {
        events = SparkGlm52RingDaemonPollEventsFromTransport(
            descriptors[descriptor_index].events);
        if (SparkGlm52RingDaemonAppendPollFd(
                fds,
                fd_kinds,
                fd_capacity,
                fd_count,
                descriptors[descriptor_index].fd,
                events,
                kind) < 0)
            return;
    }
}

static uint32_t SparkGlm52RingDaemonWorkInputCanRead(
    const SparkGlm52RingDaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->work_input_socket_fd < 0)
        return 0u;
    if (runtime->work_read_offset != 0u)
        return 1u;
    return runtime->work_queue_count <
        SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
}

static uint32_t SparkGlm52RingDaemonWaitForEvents(
    SparkGlm52RingDaemonRuntime *runtime,
    uint64_t timeout_ns)
{
    struct pollfd fds[SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY];
    uint32_t fd_kinds[SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY];
    uint32_t fd_count;
    uint32_t fd_index;
    uint32_t event_mask;
    int32_t result;
#if defined(__linux__)
    struct timespec timeout;
    struct timespec *timeout_pointer;
#endif

    if (runtime == 0)
        return 0u;
    fd_count = 0u;
    if (runtime->work_listen_fd >= 0 && runtime->work_input_socket_fd < 0)
        (void)SparkGlm52RingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_listen_fd,
            POLLIN,
            SPARK_GLM52_RING_DAEMON_POLL_KIND_WORK);
    if (SparkGlm52RingDaemonWorkInputCanRead(runtime) != 0u)
        (void)SparkGlm52RingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_input_socket_fd,
            POLLIN,
            SPARK_GLM52_RING_DAEMON_POLL_KIND_WORK);
    if (runtime->work_output_socket_fd >= 0 &&
        (runtime->work_output_connecting != 0u ||
         runtime->work_queue_count != 0u))
        (void)SparkGlm52RingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_output_socket_fd,
            POLLOUT,
            SPARK_GLM52_RING_DAEMON_POLL_KIND_WORK_OUTPUT);
    if (runtime->final_event_listen_fd >= 0 &&
        runtime->final_event_socket_fd < 0)
        (void)SparkGlm52RingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->final_event_listen_fd,
            POLLIN,
            SPARK_GLM52_RING_DAEMON_POLL_KIND_FINAL_EVENT);
    if (runtime->final_event_socket_fd >= 0)
    {
        int16_t final_events;
        final_events = POLLIN;
        if ((runtime->rank_plan.flags &
                SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
            (runtime->final_event_socket_connecting != 0u ||
             runtime->final_event_queue_count != 0u))
            final_events |= POLLOUT;
        (void)SparkGlm52RingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->final_event_socket_fd,
            final_events,
            SPARK_GLM52_RING_DAEMON_POLL_KIND_FINAL_EVENT);
    }
    if (runtime->wake_pipe_read_fd >= 0)
        (void)SparkGlm52RingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->wake_pipe_read_fd,
            POLLIN,
            SPARK_GLM52_RING_DAEMON_POLL_KIND_COMPLETION_WAKE);
    if (runtime->cuda_resident_fd >= 0)
        (void)SparkGlm52RingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->cuda_resident_fd,
            POLLIN,
            SPARK_GLM52_RING_DAEMON_POLL_KIND_CUDA_RESIDENT);
    SparkGlm52RingDaemonAppendTransportPollFds(
        runtime->input_transport_session,
        fds,
        fd_kinds,
        SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
        &fd_count,
        SPARK_GLM52_RING_DAEMON_POLL_KIND_INPUT_TRANSPORT);
    SparkGlm52RingDaemonAppendTransportPollFds(
        runtime->output_transport_session,
        fds,
        fd_kinds,
        SPARK_GLM52_RING_DAEMON_POLL_FD_CAPACITY,
        &fd_count,
        SPARK_GLM52_RING_DAEMON_POLL_KIND_OUTPUT_TRANSPORT);
    if (fd_count == 0u)
        return 0u;
#if defined(__linux__)
    timeout_pointer = 0;
    if (timeout_ns != 0u)
    {
        timeout.tv_sec = (time_t)(timeout_ns / 1000000000ull);
        timeout.tv_nsec = (long)(timeout_ns % 1000000000ull);
        timeout_pointer = &timeout;
    }
#endif
    for (;;)
    {
#if defined(__linux__)
        result = ppoll(fds,fd_count,timeout_pointer,0);
#else
        result = poll(fds,fd_count,
            SparkGlm52RingDaemonPollTimeoutMs(timeout_ns));
#endif
        if (result < 0 && errno == EINTR)
            return 0u;
        if (result == 0)
            return SPARK_GLM52_RING_DAEMON_POLL_KIND_TIMER;
        if (result < 0)
            return 0u;
        event_mask = 0u;
        for (fd_index = 0u; fd_index < fd_count; ++fd_index)
        {
            if (fds[fd_index].revents != 0)
                event_mask |= fd_kinds[fd_index];
        }
        return event_mask;
    }
}

static int32_t SparkGlm52RingDaemonStartConnect(
    const char *host,
    uint32_t port,
    uint32_t *connecting_out)
{
    struct addrinfo hints;
    struct addrinfo *results;
    struct addrinfo *entry;
    char service[16];
    int32_t fd;
    int32_t connect_status;

    if (host == 0 || connecting_out == 0)
        return -1;
    *connecting_out = 0u;
    if (snprintf(service,sizeof(service),"%u",port) <= 0)
        return -2;
    memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host,service,&hints,&results) != 0)
        return -3;
    fd = -1;
    for (entry = results; entry != 0; entry = entry->ai_next)
    {
        fd = socket(entry->ai_family,entry->ai_socktype,entry->ai_protocol);
        if (fd < 0)
            continue;
        if (SparkGlm52RingDaemonSetNonblocking(fd) < 0)
        {
            close(fd);
            fd = -1;
            continue;
        }
        SparkGlm52RingDaemonConfigureTcpSocket(fd);
        connect_status = connect(fd,entry->ai_addr,entry->ai_addrlen);
        if (connect_status == 0)
            break;
        if (errno == EINPROGRESS)
        {
            *connecting_out = 1u;
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    return fd;
}

static SparkStatus SparkGlm52RingDaemonFinishConnect(
    int32_t *fd,
    uint32_t *connecting)
{
    socklen_t error_bytes;
    int32_t error_value;

    if (fd == 0 || connecting == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (*fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    if (*connecting == 0u)
        return SPARK_STATUS_OK;
    error_value = 0;
    error_bytes = (socklen_t)sizeof(error_value);
    if (getsockopt(*fd,SOL_SOCKET,SO_ERROR,&error_value,&error_bytes) < 0)
    {
        close(*fd);
        *fd = -1;
        *connecting = 0u;
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    if (error_value == 0)
    {
        *connecting = 0u;
        SparkGlm52RingDaemonConfigureTcpSocket(*fd);
        return SPARK_STATUS_OK;
    }
    close(*fd);
    *fd = -1;
    *connecting = 0u;
    return SPARK_STATUS_ROUTE_NOT_FOUND;
}

static SparkStatus SparkGlm52RingDaemonWriteBuffered(
    int32_t fd,
    const void *buffer,
    uint32_t buffer_bytes,
    uint32_t *offset)
{
    const uint8_t *cursor;
    ssize_t written;

    if (fd < 0 || buffer == 0 || offset == 0 || *offset > buffer_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (const uint8_t *)buffer;
    while (*offset < buffer_bytes)
    {
        written = write(fd,cursor + *offset,buffer_bytes - *offset);
        if (written < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                return SPARK_STATUS_BUSY;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        if (written == 0)
            return SPARK_STATUS_BUSY;
        *offset += (uint32_t)written;
    }
    *offset = 0u;
    return SPARK_STATUS_OK;
}


static SparkStatus SparkGlm52RingDaemonWriteFull(
    int32_t fd,
    const void *buffer,
    uint32_t buffer_bytes)
{
    const uint8_t *cursor;
    uint32_t offset;
    ssize_t written;

    if (fd < 0 || buffer == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (const uint8_t *)buffer;
    offset = 0u;
    while (offset < buffer_bytes)
    {
        written = write(fd,cursor + offset,buffer_bytes - offset);
        if (written < 0)
        {
            if (errno == EINTR)
                continue;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        if (written == 0)
            return SPARK_STATUS_BUSY;
        offset += (uint32_t)written;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingDaemonReadFull(
    int32_t fd,
    void *buffer,
    uint32_t buffer_bytes)
{
    uint8_t *cursor;
    uint32_t offset;
    ssize_t got;

    if (fd < 0 || buffer == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (uint8_t *)buffer;
    offset = 0u;
    while (offset < buffer_bytes)
    {
        got = read(fd,cursor + offset,buffer_bytes - offset);
        if (got < 0)
        {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return SPARK_STATUS_BUSY;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        if (got == 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        offset += (uint32_t)got;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52RingDaemonResetResidentRead(
    SparkGlm52RingDaemonRuntime *runtime)
{
    if (runtime == 0)
        return;
    runtime->cuda_resident_read_header_offset = 0u;
    runtime->cuda_resident_read_payload_offset = 0u;
}

static SparkStatus SparkGlm52RingDaemonReadResidentBytes(
    SparkGlm52RingDaemonRuntime *runtime,
    void *buffer,
    uint32_t buffer_bytes,
    uint32_t *offset,
    uint32_t timeout_ms)
{
    struct pollfd descriptor;
    uint8_t *cursor;
    ssize_t got;

    if (runtime == 0 || runtime->cuda_resident_fd < 0 || buffer == 0 ||
        offset == 0 || *offset > buffer_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (uint8_t *)buffer;
    descriptor.fd = runtime->cuda_resident_fd;
    descriptor.events = POLLIN;
    while (*offset < buffer_bytes)
    {
        got = read(
            runtime->cuda_resident_fd,
            cursor + *offset,
            buffer_bytes - *offset);
        if (got > 0)
        {
            *offset += (uint32_t)got;
            continue;
        }
        if (got == 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        if (errno == EINTR)
            continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        if (timeout_ms == 0u)
            return SPARK_STATUS_BUSY;
        descriptor.revents = 0;
        if (poll(&descriptor,1u,(int32_t)timeout_ms) <= 0)
            return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingDaemonReadResidentMessage(
    SparkGlm52RingDaemonRuntime *runtime,
    uint32_t timeout_ms,
    SparkGlm52CudaResidentIpcHeader *header_out)
{
    SparkStatus status;
    if (runtime == 0 || header_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52RingDaemonReadResidentBytes(
        runtime,&runtime->cuda_resident_read_header,
        sizeof(runtime->cuda_resident_read_header),
        &runtime->cuda_resident_read_header_offset,timeout_ms);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentIpcValidateHeader(
        &runtime->cuda_resident_read_header,0u,
        SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52RingDaemonReadResidentBytes(
        runtime,runtime->cuda_resident_payload,
        runtime->cuda_resident_read_header.payload_bytes,
        &runtime->cuda_resident_read_payload_offset,timeout_ms);
    if (status != SPARK_STATUS_OK)
        return status;
    *header_out = runtime->cuda_resident_read_header;
    SparkGlm52RingDaemonResetResidentRead(runtime);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingDaemonWriteResidentMessage(
    SparkGlm52RingDaemonRuntime *runtime,
    uint32_t kind,
    const void *payload,
    uint32_t payload_bytes)
{
    SparkGlm52CudaResidentIpcHeader header;
    SparkStatus status;

    if (runtime == 0 || runtime->cuda_resident_fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    SparkGlm52CudaResidentIpcInitializeHeader(
        &header,
        kind,
        runtime->rank_plan.rank_index,
        runtime->cuda_resident_next_sequence_number++,
        payload_bytes);
    status = SparkGlm52RingDaemonWriteFull(
        runtime->cuda_resident_fd,
        &header,
        sizeof(header));
    if (status != SPARK_STATUS_OK)
        return status;
    if (payload_bytes != 0u)
        status = SparkGlm52RingDaemonWriteFull(
            runtime->cuda_resident_fd,
            payload,
            payload_bytes);
    return status;
}


static SparkStatus SparkGlm52RingDaemonConnectCudaResident(
    SparkGlm52RingDaemonRuntime *runtime,
    const char *socket_path)
{
    struct sockaddr_un address;
    SparkGlm52CudaResidentIpcHello hello;
    SparkGlm52CudaResidentIpcHeader header;
    SparkGlm52CudaResidentIpcStats stats;
    SparkStatus status;
    uint32_t expected_moe_backend_kind;
    int32_t fd;

    if (runtime == 0 || socket_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52RingRuntimeExpectedMoeBackendKind(
        runtime->rank_plan.quantization_mode,
        &expected_moe_backend_kind);
    if (status != SPARK_STATUS_OK)
        return status;
    fd = socket(AF_UNIX,SOCK_STREAM,0);
    if (fd < 0)
        return SPARK_STATUS_IO_ERROR;
    memset(&address,0,sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof(address.sun_path))
    {
        close(fd);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    snprintf(address.sun_path,sizeof(address.sun_path),"%s",socket_path);
    if (connect(fd,(const struct sockaddr *)&address,sizeof(address)) != 0)
    {
        close(fd);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->cuda_resident_fd = fd;
    memset(&hello,0,sizeof(hello));
    hello.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_HELLO_BYTES;
    hello.rank_index = runtime->rank_plan.rank_index;
    hello.rank_count = SPARK_GLM52_RING_RUNTIME_STAGE_COUNT;
    hello.process_id = (uint64_t)getpid();
    status = SparkGlm52RingDaemonWriteResidentMessage(
        runtime,
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO,
        &hello,
        sizeof(hello));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52RingDaemonReadFull(fd,&header,sizeof(header));
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52CudaResidentIpcValidateHeader(
            &header,
            SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO_ACK,
            sizeof(stats));
    if (status == SPARK_STATUS_OK && header.payload_bytes != sizeof(stats))
        status = SPARK_STATUS_ABI_MISMATCH;
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52RingDaemonReadFull(fd,&stats,sizeof(stats));
    if (status == SPARK_STATUS_OK &&
        stats.state != SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_READY)
        status = SPARK_STATUS_BUSY;
    if (status == SPARK_STATUS_OK &&
        (stats.logical_lane_capacity != runtime->rank_plan.logical_lane_capacity ||
         stats.execution_row_capacity != runtime->rank_plan.execution_row_capacity ||
         stats.kv_physical_block_capacity == 0u ||
         stats.kv_logical_block_capacity < stats.kv_physical_block_capacity ||
         stats.model_quantization_mode != runtime->rank_plan.quantization_mode ||
         stats.moe_backend_kind != expected_moe_backend_kind ||
         stats.moe_bound_layer_count == 0u ||
         stats.moe_bound_layer_count != stats.moe_expected_layer_count ||
         (stats.kv_nvme_enabled != 0u &&
          stats.kv_nvme_mode !=
            SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_SYNCHRONOUS_FULL_HISTORY &&
          stats.kv_nvme_mode !=
            SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT &&
          stats.kv_nvme_mode !=
            SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT) ||
         (runtime->rank_plan.logical_lane_capacity >=
            SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT &&
          (stats.kv_nvme_enabled == 0u ||
           (stats.kv_nvme_mode !=
                SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT &&
            stats.kv_nvme_mode !=
                SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT) ||
           stats.kv_resident_bytes_per_token == 0u ||
           stats.kv_resident_pool_bytes == 0u ||
           stats.kv_nvme_capacity_bytes == 0u ||
           stats.kv_nvme_batch_block_capacity == 0u))))
    {
        fprintf(
            stderr,
            "rank_cuda_resident_contract_mismatch rank=%u logical=%u/%u "
			"execution=%u/%u kv_blocks=%u logical_blocks=%u "
			"quantization=%u/%u moe_backend=%u moe_layers=%u/%u nvme=%u "
            "nvme_mode=%u "
            "blocker=%.*s\n",
            runtime->rank_plan.rank_index,
            stats.logical_lane_capacity,
            runtime->rank_plan.logical_lane_capacity,
            stats.execution_row_capacity,
            runtime->rank_plan.execution_row_capacity,
			stats.kv_physical_block_capacity,
			stats.kv_logical_block_capacity,
            stats.model_quantization_mode,
            runtime->rank_plan.quantization_mode,
            stats.moe_backend_kind,
            stats.moe_bound_layer_count,
            stats.moe_expected_layer_count,
            stats.kv_nvme_enabled,
            stats.kv_nvme_mode,
            (int32_t)sizeof(stats.blocker),
            stats.blocker);
        status = SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (status == SPARK_STATUS_OK &&
        SparkGlm52RingDaemonSetNonblocking(fd) < 0)
        status = SPARK_STATUS_INTERNAL_ERROR;
    if (status != SPARK_STATUS_OK)
    {
        close(fd);
        runtime->cuda_resident_fd = -1;
        return status;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52RingDaemonTeardownCudaResident(
    SparkGlm52RingDaemonRuntime *runtime,
    const char *reason)
{
    if (runtime == 0 || runtime->cuda_resident_fd < 0)
        return;
    fprintf(stderr,"rank_cuda_resident_disconnected rank=%u reason=%s\n",
        runtime->rank_plan.rank_index,reason);
    close(runtime->cuda_resident_fd);
    runtime->cuda_resident_fd = -1;
    SparkGlm52RingDaemonResetResidentRead(runtime);
    runtime->cuda_resident_error_count += 1u;
}

static SparkStatus SparkGlm52RingDaemonEnsureCudaResident(
    SparkGlm52RingDaemonRuntime *runtime)
{
    uint64_t now_ns;
    SparkStatus status;

    if (runtime == 0 || runtime->cuda_resident_socket_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (runtime->cuda_resident_fd >= 0)
        return SPARK_STATUS_OK;
    now_ns = SparkGlm52RingDaemonMonotonicNs();
    if (now_ns < runtime->cuda_resident_retry_after_ns)
        return SPARK_STATUS_BUSY;
    runtime->cuda_resident_retry_after_ns = now_ns + 250000000ull;
    status = SparkGlm52RingDaemonConnectCudaResident(
        runtime,
        runtime->cuda_resident_socket_path);
    if (status == SPARK_STATUS_OK)
    {
        fprintf(stderr,"rank_cuda_resident_connected rank=%u\n",
            runtime->rank_plan.rank_index);
        return SPARK_STATUS_OK;
    }
    fprintf(stderr,"rank_cuda_resident_connect_retry rank=%u status=%u\n",
        runtime->rank_plan.rank_index,(uint32_t)status);
    return SPARK_STATUS_BUSY;
}

static uint32_t SparkGlm52RingDaemonHandleCudaResidentMessage(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52CudaResidentIpcHeader *header,
    const uint8_t *payload)
{
    const SparkGlm52CudaResidentIpcCompletion *completion_message;
    const SparkGlm52CudaResidentIpcStats *stats;

    if (runtime == 0 || header == 0)
        return 0u;
    if (header->kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_COMPLETION)
    {
        if (payload == 0 || header->payload_bytes !=
            SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_BYTES)
        {
            runtime->cuda_resident_error_count += 1u;
            return 0u;
        }
        completion_message =
            (const SparkGlm52CudaResidentIpcCompletion *)payload;
        if (completion_message->descriptor_bytes !=
                SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_BYTES ||
            (completion_message->flags &
                ~SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_KNOWN_FLAGS) != 0u)
        {
            runtime->cuda_resident_error_count += 1u;
            return 0u;
        }
        if ((completion_message->flags &
                SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT) != 0u)
        {
            if (completion_message->dspark_draft.abi_version !=
                    SPARK_GLM52_DSPARK_ABI_VERSION ||
                completion_message->dspark_draft.descriptor_bytes !=
                    SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES)
            {
                runtime->cuda_resident_error_count += 1u;
                return 0u;
            }
            runtime->completion_dspark_draft = completion_message->dspark_draft;
            runtime->completion_dspark_draft_valid = 1u;
        }
        runtime->cuda_resident_completion_count += 1u;
        SparkGlm52RingDaemonCompletion(
            runtime,
            &completion_message->completion);
        return 1u;
    }
    if (header->kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_STATS)
    {
        if (payload == 0 || header->payload_bytes !=
            SPARK_GLM52_CUDA_RESIDENT_IPC_STATS_BYTES)
            runtime->cuda_resident_error_count += 1u;
        else
        {
            stats = (const SparkGlm52CudaResidentIpcStats *)payload;
            if (stats->state == SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_FAILED)
                runtime->cuda_resident_error_count += 1u;
        }
        return 1u;
    }
    if (header->kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT)
    {
        const SparkGlm52CudaResidentIpcSubmitResult *result;
        if (payload == 0 || header->payload_bytes !=
            SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES)
        {
            runtime->cuda_resident_error_count += 1u;
            return 0u;
        }
        result = (const SparkGlm52CudaResidentIpcSubmitResult *)payload;
        if (result->status != (uint32_t)SPARK_STATUS_OK)
            runtime->cuda_resident_error_count += 1u;
        return 1u;
    }
    runtime->cuda_resident_error_count += 1u;
    return 0u;
}

static void SparkGlm52RingDaemonCheckInflightOverdue(
    SparkGlm52RingDaemonRuntime *runtime)
{
    uint64_t now_ns;
    if (runtime == 0 || runtime->driver_inflight_count == 0u ||
        runtime->driver_inflight_warned != 0u)
        return;
    now_ns = SparkGlm52RingDaemonMonotonicNs();
    if (now_ns - runtime->driver_inflight_open_ns <= 30000000000ull)
        return;
    runtime->driver_inflight_warned = 1u;
    fprintf(stderr,"rank_completion_overdue rank=%u inflight=%u age_s=%llu submitted=%llu completed=%llu\n",runtime->rank_plan.rank_index,runtime->driver_inflight_count,(unsigned long long)((now_ns - runtime->driver_inflight_open_ns) / 1000000000ull),(unsigned long long)runtime->cuda_resident_submit_count,(unsigned long long)runtime->driver_completion_count);
}

static uint32_t SparkGlm52RingDaemonPumpCudaResident(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkGlm52CudaResidentIpcHeader header;
    SparkStatus status;

    if (runtime == 0 || runtime->cuda_resident_fd < 0)
        return 0u;
    status = SparkGlm52RingDaemonReadResidentMessage(runtime,0u,&header);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonTeardownCudaResident(runtime,"pump_message_read");
        return 1u;
    }
    (void)SparkGlm52RingDaemonHandleCudaResidentMessage(
        runtime,
        &header,
        header.payload_bytes == 0u ? 0 : runtime->cuda_resident_payload);
    return 1u;
}

static SparkStatus SparkGlm52RingDaemonOpenWakePipe(
    SparkGlm52RingDaemonRuntime *runtime)
{
    int32_t pipe_fds[2];

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (pipe(pipe_fds) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (SparkGlm52RingDaemonSetNonblocking(pipe_fds[0]) < 0 ||
        SparkGlm52RingDaemonSetNonblocking(pipe_fds[1]) < 0)
    {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->wake_pipe_read_fd = pipe_fds[0];
    runtime->wake_pipe_write_fd = pipe_fds[1];
    return SPARK_STATUS_OK;
}

static void SparkGlm52RingDaemonSignalWake(
    SparkGlm52RingDaemonRuntime *runtime)
{
    uint8_t byte;
    ssize_t wrote;

    if (runtime == 0 || runtime->wake_pipe_write_fd < 0)
        return;
    byte = 1u;
    wrote = write(runtime->wake_pipe_write_fd,&byte,1u);
    if (wrote == 1)
    {
        runtime->wake_signal_count += 1u;
        return;
    }
    if (wrote < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
        runtime->wake_drop_count += 1u;
}

static SparkStatus SparkGlm52RingDaemonQueueFinalEvent(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingRuntimeFinalEvent *event)
{
    uint32_t tail;

    if (runtime == 0 || event == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (runtime->final_event_queue_count >=
        SPARK_GLM52_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    tail =
        (runtime->final_event_queue_head + runtime->final_event_queue_count) %
        SPARK_GLM52_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY;
    runtime->final_event_queue[tail] = *event;
    runtime->final_event_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkGlm52RingDaemonPopFinalEvent(
    SparkGlm52RingDaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->final_event_queue_count == 0u)
        return;
    runtime->final_event_queue_head =
        (runtime->final_event_queue_head + 1u) %
        SPARK_GLM52_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY;
    runtime->final_event_queue_count -= 1u;
}

static uint32_t SparkGlm52RingDaemonDrainWakePipe(
    SparkGlm52RingDaemonRuntime *runtime)
{
    uint8_t buffer[64];
    ssize_t got;
    uint32_t progress;

    progress = 0u;
    if (runtime == 0 || runtime->wake_pipe_read_fd < 0)
        return 0u;
    for (;;)
    {
        got = read(runtime->wake_pipe_read_fd,buffer,sizeof(buffer));
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
            return progress;
        if (got <= 0)
            return progress;
        progress = 1u;
    }
}

static void SparkGlm52RingDaemonCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkGlm52RingDaemonRuntime *runtime;
    SparkGlm52RingRuntimeFinalEvent event;

    runtime = (SparkGlm52RingDaemonRuntime *)completion_context;
    if (runtime == 0 || completion == 0)
        return;
    if (runtime->completion_dspark_draft_valid == 0u &&
        runtime->builder_library.builder_interface.take_dspark_draft != 0 &&
        runtime->builder_library.builder_interface.take_dspark_draft(
            runtime->builder_state,
            &runtime->completion_dspark_draft) == SPARK_STATUS_OK)
        runtime->completion_dspark_draft_valid = 1u;
    if (runtime->driver_inflight_count != 0u)
        runtime->driver_inflight_count -= 1u;
    if (runtime->driver_inflight_count == 0u)
        runtime->driver_inflight_warned = 0u;
    else
        runtime->driver_inflight_open_ns = SparkGlm52RingDaemonMonotonicNs();
    runtime->driver_completion_count += 1u;
    SparkGlm52RingDaemonSignalWake(runtime);
    if (runtime->trace_enabled != 0u)
        fprintf(stderr,"ring_trace rank=%u completion request=%llu position=%llu flags=0x%x tokens=%u id0=%u status=%u\n",runtime->rank_plan.rank_index,(unsigned long long)completion->request_id,(unsigned long long)completion->sequence_position,completion->completion_flags,completion->token_count,completion->token_count != 0u ? completion->token_ids[0u] : 0u,(uint32_t)completion->status);
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u)
        return;
    if (completion->status == SPARK_STATUS_OK &&
        ((completion->completion_flags &
            SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u ||
         completion->token_count == 0u))
    {
        if (runtime->trace_enabled != 0u)
            fprintf(stderr,"ring_trace rank=%u final_event_skipped request=%llu position=%llu flags=0x%x tokens=%u\n",runtime->rank_plan.rank_index,(unsigned long long)completion->request_id,(unsigned long long)completion->sequence_position,completion->completion_flags,completion->token_count);
        return;
    }
    memset(&event,0,sizeof(event));
    event.magic = SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_MAGIC;
    event.descriptor_bytes = SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES;
    event.status = (uint32_t)completion->status;
    event.program_id = completion->program_id;
    event.driver_dispatch_slot = completion->driver_dispatch_slot;
    event.accepted_token_count = completion->accepted_token_count;
    event.completion_flags = completion->completion_flags;
    event.token_count = completion->token_count;
    if (event.token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY)
        event.token_count = SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
    memcpy(event.token_ids,completion->token_ids,
        event.token_count * sizeof(event.token_ids[0u]));
    event.draft_token_count = completion->draft_token_count;
    if ((((completion->completion_flags &
            SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS) != 0u) !=
         (event.draft_token_count != 0u)) ||
        event.draft_token_count >
        SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY)
    {
        fprintf(stderr,"ring_final_event_invalid_draft count=%u flags=0x%x\n",
            event.draft_token_count,completion->completion_flags);
        runtime->final_event_send_error_count += 1u;
        return;
    }
    memcpy(event.draft_token_ids,completion->draft_token_ids,
        event.draft_token_count * sizeof(event.draft_token_ids[0u]));
    event.request_id = completion->request_id;
    event.sequence_id = completion->sequence_id;
    event.sequence_position = completion->sequence_position;
    event.service_time_ns = completion->service_time_ns;
    if (runtime->completion_dspark_draft_valid != 0u)
    {
        event.extension_flags |=
            SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_FLAG_DSPARK_DRAFT;
        event.dspark_draft = runtime->completion_dspark_draft;
        memset(&runtime->completion_dspark_draft,0,
            sizeof(runtime->completion_dspark_draft));
        runtime->completion_dspark_draft_valid = 0u;
    }
    if (SparkGlm52RingDaemonQueueFinalEvent(runtime,&event) != SPARK_STATUS_OK)
    {
        runtime->final_event_send_error_count += 1u;
        return;
    }
    SparkGlm52RingDaemonSignalWake(runtime);
}

static SparkStatus SparkGlm52RingDaemonLoadTransport(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingDaemonConfig *configuration)
{
    char rank_buffer[16];
    char port_buffer[16];
    if (snprintf(rank_buffer, sizeof(rank_buffer), "%u",
            runtime->rank_plan.rank_index) < 0 ||
        snprintf(port_buffer, sizeof(port_buffer), "%u",
            configuration->port_base) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (setenv("SPARKPIPE_RING_TRANSPORT_RANK",rank_buffer,1) != 0 ||
        setenv("SPARKPIPE_RING_TRANSPORT_PORT_BASE",port_buffer,1) != 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SparkHiddenTransportLoadInterfaceFromSharedObject(
        configuration->transport_shared_object_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        &runtime->transport_library);
}

static SparkStatus SparkGlm52RingDaemonOpenHiddenTransport(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkStatus status;

    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
    {
        status = SparkHiddenTransportOpen(
            &runtime->rank_plan.input_endpoint,
            &runtime->transport_library.transport_interface,
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
            &runtime->input_transport_session);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
    {
        status = SparkHiddenTransportOpen(
            &runtime->rank_plan.output_endpoint,
            &runtime->transport_library.transport_interface,
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
            &runtime->output_transport_session);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingDaemonLoadDriver(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingDaemonConfig *configuration,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkModelDriverCreateRequest create_request;
    SparkStatus status;

    status = SparkLoadModelDriver(
        configuration->driver_path,
        configuration->node_target,
        &runtime->loaded_driver,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetStatusError(
            error_buffer,
            error_buffer_bytes,
            "failed to dlopen/validate GLM52 model driver",
            status);
        return status;
    }
    runtime->program = SparkFindLoadedModelDriverProgram(
        &runtime->loaded_driver,
        configuration->program_name);
    if (runtime->program == 0)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "decode program not found in GLM52 model driver");
        return SPARK_STATUS_NOT_FOUND;
    }
    memset(&create_request,0,sizeof(create_request));
    create_request.node_id = runtime->rank_plan.host_name;
    create_request.node_target = configuration->node_target;
    create_request.node_context = runtime->builder_result.node_context;
    create_request.completion_function = SparkGlm52RingDaemonCompletion;
    create_request.completion_context = runtime;
    status = runtime->loaded_driver.interface->create(
        &create_request,
        &runtime->driver_instance);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetStatusError(
            error_buffer,
            error_buffer_bytes,
            "GLM52 model driver create failed",
            status);
        return status;
    }
    if (runtime->driver_instance == 0)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "GLM52 model driver returned NULL instance");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingDaemonBuildNodeContext(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingDaemonConfig *configuration)
{
    SparkGlm52RingNodeContextBuilderConfiguration builder_configuration;
    SparkStatus status;

    memset(&builder_configuration,0,sizeof(builder_configuration));
    builder_configuration.abi_version =
        SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_ABI_VERSION;
    builder_configuration.descriptor_bytes =
        SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES;
    builder_configuration.rank_index = runtime->rank_plan.rank_index;
    builder_configuration.max_active_sequence_count =
        configuration->max_active_sequence_count;
	builder_configuration.kv_pool_token_capacity = SPARK_GLM52_KV_POOL_TOKENS;
	builder_configuration.maximum_resident_sequence_count =
		SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_RESIDENT_SEQUENCE_COUNT;
    builder_configuration.port_base = configuration->port_base;
    builder_configuration.moe_pack_root = configuration->moe_pack_root;
    builder_configuration.stagepack_root = configuration->stagepack_root;
    builder_configuration.embedding_pack_path =
        configuration->embedding_pack_path;
    builder_configuration.node_target = configuration->node_target;
    builder_configuration.rank_plan = &runtime->rank_plan;
    status = SparkGlm52RingNodeContextBuilderLoadInterfaceFromSharedObject(
        configuration->node_context_builder_shared_object_path,
        SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
        &runtime->builder_library);
    if (status != SPARK_STATUS_OK)
        return status;
    status = runtime->builder_library.builder_interface.initialize(
        &builder_configuration,
        &runtime->builder_state);
    if (status != SPARK_STATUS_OK || runtime->builder_state == 0)
        return status == SPARK_STATUS_OK ?
            SPARK_STATUS_INVALID_ARGUMENT : status;
    memset(&runtime->builder_result,0,sizeof(runtime->builder_result));
    status = runtime->builder_library.builder_interface.build(
        runtime->builder_state,
        &runtime->builder_result);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkGlm52RingNodeContextBuilderValidateResult(
        &runtime->builder_result,
        &runtime->rank_plan);
}

static SparkStatus SparkGlm52RingDaemonAttachBuilderDriver(
    SparkGlm52RingDaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->builder_state == 0 ||
        runtime->builder_library.builder_interface.attach_driver == 0 ||
        runtime->loaded_driver.interface == 0 ||
        runtime->driver_instance == 0 ||
        runtime->program == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return runtime->builder_library.builder_interface.attach_driver(
        runtime->builder_state,
        runtime->loaded_driver.interface,
        runtime->driver_instance,
        runtime->program,
        runtime->output_transport_session);
}

static SparkStatus SparkGlm52RingDaemonInitializeRunner(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkGlm52ResidentDecodeStageProductionRunnerConfiguration configuration;

    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
    configuration.flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION;
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
        configuration.flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT;
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
        configuration.flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
    configuration.driver_interface = runtime->loaded_driver.interface;
    configuration.driver_instance = runtime->driver_instance;
    configuration.program = runtime->program;
    configuration.execution_stream = 0;
    return SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
        &runtime->runner,
        &configuration);
}

static SparkStatus SparkGlm52RingDaemonOpenWorkControlPath(
    SparkGlm52RingDaemonRuntime *runtime)
{
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u)
        return SPARK_STATUS_OK;
    runtime->work_listen_fd =
        SparkGlm52RingDaemonCreateListenSocket(
            "0.0.0.0",
            runtime->rank_plan.listen_port);
    if (runtime->work_listen_fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    if (SparkGlm52RingDaemonSetNonblocking(runtime->work_listen_fd) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RingDaemonAcceptWorkSocket(
    SparkGlm52RingDaemonRuntime *runtime)
{
    int32_t fd;

    if (runtime->work_listen_fd < 0 || runtime->work_input_socket_fd >= 0)
        return 0u;
    fd = accept(runtime->work_listen_fd,0,0);
    if (fd < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            runtime->work_error_count += 1u;
        return 0u;
    }
    if (SparkGlm52RingDaemonSetNonblocking(fd) < 0)
    {
        close(fd);
        runtime->work_error_count += 1u;
        return 0u;
    }
    SparkGlm52RingDaemonConfigureTcpSocket(fd);
    runtime->work_input_socket_fd = fd;
    return 1u;
}

static SparkStatus SparkGlm52RingDaemonEnsureWorkOutputSocket(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkStatus status;
    uint64_t now_ns;

    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
        return SPARK_STATUS_OK;
    if (runtime->work_output_socket_fd >= 0)
    {
        status = SparkGlm52RingDaemonFinishConnect(
            &runtime->work_output_socket_fd,
            &runtime->work_output_connecting);
        if (status == SPARK_STATUS_OK)
            runtime->work_output_retry_mono_ns = 0u;
        else if (runtime->work_output_socket_fd < 0)
            runtime->work_output_retry_mono_ns =
                SparkGlm52RingDaemonMonotonicNs() +
                SPARK_GLM52_RING_DAEMON_CONNECT_RETRY_NS;
        return status;
    }
    now_ns = SparkGlm52RingDaemonMonotonicNs();
    if (runtime->work_output_retry_mono_ns != 0u &&
        now_ns < runtime->work_output_retry_mono_ns)
        return SPARK_STATUS_BUSY;
    runtime->work_output_socket_fd =
        SparkGlm52RingDaemonStartConnect(
            runtime->rank_plan.next_host_name,
            runtime->rank_plan.next_port,
            &runtime->work_output_connecting);
    if (runtime->work_output_socket_fd < 0)
    {
        runtime->work_output_retry_mono_ns =
            now_ns + SPARK_GLM52_RING_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->work_output_retry_mono_ns = 0u;
    return runtime->work_output_connecting == 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static SparkStatus SparkGlm52RingDaemonForwardWork(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *packet)
{
    SparkStatus status;

    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
        return SPARK_STATUS_OK;
    status = SparkGlm52RingDaemonEnsureWorkOutputSocket(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52RingDaemonWriteBuffered(
            runtime->work_output_socket_fd,
            packet,
            packet->descriptor_bytes,
            &runtime->work_output_write_offset);
    if (status == SPARK_STATUS_BUSY)
        return status;
    if (status != SPARK_STATUS_OK)
    {
        close(runtime->work_output_socket_fd);
        runtime->work_output_socket_fd = -1;
        runtime->work_output_connecting = 0u;
        runtime->work_output_write_offset = 0u;
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->work_forward_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingDaemonAwaitResidentSubmitResult(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkGlm52CudaResidentIpcHeader header;
    const SparkGlm52CudaResidentIpcSubmitResult *submit_result;
    const SparkGlm52CudaResidentIpcCompletion *completion_message;
    SparkStatus status;

    for (;;)
    {
        status = SparkGlm52RingDaemonReadResidentMessage(
            runtime,30000u,&header);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52RingDaemonTeardownCudaResident(runtime,"submit_message_read");
            return SPARK_STATUS_BUSY;
        }
        if (header.kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT)
        {
            if (header.payload_bytes != sizeof(*submit_result))
            {
                SparkGlm52RingDaemonTeardownCudaResident(runtime,"submit_result_read");
                return SPARK_STATUS_BUSY;
            }
            submit_result = (const SparkGlm52CudaResidentIpcSubmitResult *)
                runtime->cuda_resident_payload;
            return (SparkStatus)submit_result->status;
        }
        if (header.kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_COMPLETION)
        {
            if (header.payload_bytes != sizeof(*completion_message))
            {
                SparkGlm52RingDaemonTeardownCudaResident(runtime,"submit_completion_read");
                return SPARK_STATUS_BUSY;
            }
            completion_message = (const SparkGlm52CudaResidentIpcCompletion *)
                runtime->cuda_resident_payload;
            if ((completion_message->flags &
                    SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT) != 0u)
            {
                runtime->completion_dspark_draft = completion_message->dspark_draft;
                runtime->completion_dspark_draft_valid = 1u;
            }
            SparkGlm52RingDaemonCompletion(runtime,&completion_message->completion);
            continue;
        }
        SparkGlm52RingDaemonTeardownCudaResident(runtime,"submit_unknown_kind");
        return SPARK_STATUS_BUSY;
    }
}

static SparkStatus SparkGlm52RingDaemonSubmitWork(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *packet)
{
    SparkGlm52CudaResidentIpcSubmitWork submit_message;
    SparkStatus status;
    uint64_t trace_begin_ns;

    if (runtime == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    trace_begin_ns = runtime->trace_enabled != 0u ? SparkGlm52RingDaemonMonotonicNs() : 0u;
    if (runtime->cuda_resident_socket_path != 0)
    {
        status = SparkGlm52RingDaemonEnsureCudaResident(runtime);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkGlm52CudaResidentIpcInitializeSubmitWork(
            &submit_message,packet,
            SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkGlm52RingDaemonWriteResidentMessage(
            runtime,
            SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK,
            &submit_message,
			submit_message.descriptor_bytes);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52RingDaemonTeardownCudaResident(runtime,"submit_write");
            return SPARK_STATUS_BUSY;
        }
        runtime->cuda_resident_submit_count += 1u;
        status = SparkGlm52RingDaemonAwaitResidentSubmitResult(runtime);
        if (runtime->trace_enabled != 0u)
            fprintf(stderr,"ring_trace rank=%u work_submit request=%llu position=%llu status=%u dur_us=%llu\n",runtime->rank_plan.rank_index,(unsigned long long)packet->request_id,(unsigned long long)packet->sequence_position,(uint32_t)status,(unsigned long long)((SparkGlm52RingDaemonMonotonicNs() - trace_begin_ns) / 1000ull));
        return status;
    }
    if (runtime->builder_state == 0 ||
        runtime->builder_library.builder_interface.submit_work == 0)
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    return runtime->builder_library.builder_interface.submit_work(
        runtime->builder_state,
        packet,
        runtime->input_transport_session,
        runtime->output_transport_session,
        SparkGlm52RingDaemonCompletion,
        runtime);
}

static SparkStatus SparkGlm52RingDaemonProgressBuilder(
	SparkGlm52RingDaemonRuntime *runtime)
{
	if (runtime == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (runtime->cuda_resident_socket_path != 0)
		return SparkGlm52RingDaemonEnsureCudaResident(runtime);
	if (runtime->builder_state == 0 ||
		runtime->builder_library.builder_interface.progress == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	return runtime->builder_library.builder_interface.progress(
		runtime->builder_state);
}

static uint32_t SparkGlm52RingDaemonWorkPacketHash(
	const SparkGlm52RingWorkControlPacket *packet)
{
	const uint8_t *packet_bytes;
	uint32_t byte_index;
	uint32_t hash;

    if (packet == 0 || packet->descriptor_bytes <
            SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES ||
        packet->descriptor_bytes > SPARK_GLM52_RING_WORK_CONTROL_PACKET_BYTES)
        return 0u;
    packet_bytes = (const uint8_t *)packet;
    hash = SPARK_GLM52_RING_DAEMON_WORK_HASH_OFFSET;
    for (byte_index = 0u; byte_index < packet->descriptor_bytes; ++byte_index)
        hash = (hash ^ packet_bytes[byte_index]) *
            SPARK_GLM52_RING_DAEMON_WORK_HASH_PRIME;
    return hash;
}

static uint32_t SparkGlm52RingDaemonWorkPacketMatches(
    const SparkGlm52RingWorkControlPacket *left,
    const SparkGlm52RingWorkControlPacket *right)
{
    if (left == 0 || right == 0 ||
        left->descriptor_bytes != right->descriptor_bytes ||
        left->descriptor_bytes <
            SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES ||
        left->descriptor_bytes > SPARK_GLM52_RING_WORK_CONTROL_PACKET_BYTES)
        return 0u;
    return memcmp(left,right,left->descriptor_bytes) == 0;
}

static uint32_t SparkGlm52RingDaemonWorkPacketPhase(
    const SparkGlm52RingWorkControlPacket *packet)
{
    if ((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
        return SPARK_GLM52_RING_DAEMON_WORK_PHASE_RELEASE;
    if ((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u)
        return SPARK_GLM52_RING_DAEMON_WORK_PHASE_PREFILL;
    if ((packet->flags &
            (SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY |
             SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY)) != 0u)
        return SPARK_GLM52_RING_DAEMON_WORK_PHASE_VERIFY;
    return SPARK_GLM52_RING_DAEMON_WORK_PHASE_DECODE;
}

static uint32_t SparkGlm52RingDaemonDependencyHash(uint64_t sequence_id)
{
    return (uint32_t)(sequence_id ^ (sequence_id >> 32u)) &
        (SPARK_GLM52_RING_DAEMON_DEPENDENCY_HASH_CAPACITY - 1u);
}

static void SparkGlm52RingDaemonIndexDependencyLanes(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *packet)
{
    uint32_t hash_slot;
    uint32_t lane_index;
    memset(runtime->dependency_sequence_ids,0,
        sizeof(runtime->dependency_sequence_ids));
    for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
    {
        hash_slot = SparkGlm52RingDaemonDependencyHash(
            packet->lanes[lane_index].sequence_id);
        while (runtime->dependency_sequence_ids[hash_slot] != 0u)
            hash_slot = (hash_slot + 1u) &
                (SPARK_GLM52_RING_DAEMON_DEPENDENCY_HASH_CAPACITY - 1u);
        runtime->dependency_sequence_ids[hash_slot] =
            packet->lanes[lane_index].sequence_id;
        runtime->dependency_sequence_positions[hash_slot] =
            packet->lanes[lane_index].sequence_position;
    }
}

static uint32_t SparkGlm52RingDaemonCandidateIsDependency(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *candidate,
    const SparkGlm52RingWorkControlPacket *packet)
{
    uint32_t candidate_phase;
    uint32_t hash_slot;
    uint32_t lane_index;
    uint32_t packet_phase;
    if (candidate == packet || candidate->control_generation !=
        packet->control_generation)
        return 0u;
    candidate_phase = SparkGlm52RingDaemonWorkPacketPhase(candidate);
    packet_phase = SparkGlm52RingDaemonWorkPacketPhase(packet);
    for (lane_index = 0u; lane_index < candidate->lane_count; ++lane_index)
    {
        hash_slot = SparkGlm52RingDaemonDependencyHash(
            candidate->lanes[lane_index].sequence_id);
        while (runtime->dependency_sequence_ids[hash_slot] != 0u &&
            runtime->dependency_sequence_ids[hash_slot] !=
                candidate->lanes[lane_index].sequence_id)
            hash_slot = (hash_slot + 1u) &
                (SPARK_GLM52_RING_DAEMON_DEPENDENCY_HASH_CAPACITY - 1u);
        if (runtime->dependency_sequence_ids[hash_slot] == 0u)
            continue;
        if (packet_phase == SPARK_GLM52_RING_DAEMON_WORK_PHASE_RELEASE &&
            candidate_phase != SPARK_GLM52_RING_DAEMON_WORK_PHASE_RELEASE)
            return 1u;
        if (candidate->lanes[lane_index].sequence_position <
                runtime->dependency_sequence_positions[hash_slot] ||
            (candidate->lanes[lane_index].sequence_position ==
                 runtime->dependency_sequence_positions[hash_slot] &&
             candidate_phase < packet_phase))
            return 1u;
    }
    return 0u;
}

static void SparkGlm52RingDaemonInitializeWorkQueue(
    SparkGlm52RingDaemonRuntime *runtime)
{
    uint32_t slot_index;

    if (runtime == 0)
        return;
    for (slot_index = 0u;
         slot_index < SPARK_GLM52_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
         ++slot_index)
        runtime->work_queue_hash_heads[slot_index] =
            SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT;
    for (slot_index = 0u;
         slot_index < SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
         ++slot_index)
    {
        runtime->work_queue_hash_next[slot_index] =
            SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT;
        runtime->work_queue_submitted[slot_index] = 0u;
        runtime->work_queue_forwarded[slot_index] = 0u;
        runtime->work_queue_state[slot_index] =
            SPARK_GLM52_RING_DAEMON_WORK_STATE_READY;
    }
}

static uint32_t SparkGlm52RingDaemonFindQueuedWorkSlot(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *packet)
{
    uint32_t hash_slot;
    uint32_t slot_index;

    if (runtime == 0 || packet == 0)
        return SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT;
    hash_slot = SparkGlm52RingDaemonWorkPacketHash(packet) %
        SPARK_GLM52_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
    slot_index = runtime->work_queue_hash_heads[hash_slot];
    while (slot_index != SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (slot_index >= SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY)
            return SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT;
        if (SparkGlm52RingDaemonWorkPacketMatches(
                &runtime->work_queue[slot_index],
                packet) != 0u)
            return slot_index;
        slot_index = runtime->work_queue_hash_next[slot_index];
    }
    return SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT;
}

static uint32_t SparkGlm52RingDaemonHasQueuedDependency(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *packet)
{
    uint32_t scan_index;
    uint32_t slot_index;

    if (runtime == 0 || packet == 0 || runtime->work_queue_count <= 1u)
        return 0u;
    SparkGlm52RingDaemonIndexDependencyLanes(runtime,packet);
    slot_index = runtime->work_queue_head;
    for (scan_index = 0u; scan_index < runtime->work_queue_count; ++scan_index)
    {
        if (SparkGlm52RingDaemonCandidateIsDependency(
                runtime,&runtime->work_queue[slot_index],packet) != 0u)
            return 1u;
        slot_index = (slot_index + 1u) %
            SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
    }
    return 0u;
}

static void SparkGlm52RingDaemonInsertQueuedWorkHash(
    SparkGlm52RingDaemonRuntime *runtime,
    uint32_t queue_slot)
{
    uint32_t hash_slot;

    if (runtime == 0 ||
        queue_slot >= SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY)
        return;
    hash_slot =
        SparkGlm52RingDaemonWorkPacketHash(&runtime->work_queue[queue_slot]) %
        SPARK_GLM52_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
    runtime->work_queue_hash_next[queue_slot] =
        runtime->work_queue_hash_heads[hash_slot];
    runtime->work_queue_hash_heads[hash_slot] = queue_slot;
}

static void SparkGlm52RingDaemonRemoveQueuedWorkHash(
    SparkGlm52RingDaemonRuntime *runtime,
    uint32_t queue_slot)
{
    uint32_t hash_slot;
    uint32_t slot_index;
    uint32_t previous_slot;

    if (runtime == 0 ||
        queue_slot >= SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY)
        return;
    hash_slot =
        SparkGlm52RingDaemonWorkPacketHash(&runtime->work_queue[queue_slot]) %
        SPARK_GLM52_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
    slot_index = runtime->work_queue_hash_heads[hash_slot];
    previous_slot = SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT;
    while (slot_index != SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (slot_index == queue_slot)
        {
            if (previous_slot == SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT)
                runtime->work_queue_hash_heads[hash_slot] =
                    runtime->work_queue_hash_next[slot_index];
            else
                runtime->work_queue_hash_next[previous_slot] =
                    runtime->work_queue_hash_next[slot_index];
            runtime->work_queue_hash_next[slot_index] =
                SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT;
            return;
        }
        if (slot_index >= SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY)
            return;
        previous_slot = slot_index;
        slot_index = runtime->work_queue_hash_next[slot_index];
    }
}

static SparkStatus SparkGlm52RingDaemonQueueWork(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *packet)
{
    uint32_t tail;
    uint32_t existing_slot;

    if (runtime == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    runtime->work_receive_count += 1u;
    existing_slot = SparkGlm52RingDaemonFindQueuedWorkSlot(runtime,packet);
    if (existing_slot != SPARK_GLM52_RING_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (runtime->work_queue_submitted[existing_slot] == 0u)
        {
            runtime->work_queue[existing_slot] = *packet;
            runtime->work_queue_forwarded[existing_slot] = 0u;
            runtime->work_queue_state[existing_slot] =
                SPARK_GLM52_RING_DAEMON_WORK_STATE_READY;
        }
        runtime->work_duplicate_count += 1u;
        return SPARK_STATUS_OK;
    }
    if (runtime->work_queue_count >= SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    tail = (runtime->work_queue_head + runtime->work_queue_count) %
        SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
    runtime->work_queue[tail] = *packet;
    runtime->work_queue_submitted[tail] = 0u;
    runtime->work_queue_forwarded[tail] = 0u;
    runtime->work_queue_state[tail] =
        SPARK_GLM52_RING_DAEMON_WORK_STATE_READY;
    SparkGlm52RingDaemonInsertQueuedWorkHash(runtime,tail);
    runtime->work_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkGlm52RingDaemonPopWork(
    SparkGlm52RingDaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->work_queue_count == 0u)
        return;
    SparkGlm52RingDaemonRemoveQueuedWorkHash(runtime,runtime->work_queue_head);
    runtime->work_queue_submitted[runtime->work_queue_head] = 0u;
    runtime->work_queue_forwarded[runtime->work_queue_head] = 0u;
    runtime->work_queue_state[runtime->work_queue_head] =
        SPARK_GLM52_RING_DAEMON_WORK_STATE_READY;
    runtime->work_queue_head = (runtime->work_queue_head + 1u) %
        SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
    runtime->work_queue_count -= 1u;
}

static void SparkGlm52RingDaemonDeferWork(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkGlm52RingWorkControlPacket packet;
    uint32_t old_head;
    uint32_t tail;

    if (runtime == 0 || runtime->work_queue_count <= 1u)
        return;
    old_head = runtime->work_queue_head;
    if (runtime->work_queue_count < SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY)
    {
        tail = (runtime->work_queue_head + runtime->work_queue_count) %
            SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
        packet = runtime->work_queue[old_head];
        SparkGlm52RingDaemonRemoveQueuedWorkHash(runtime,old_head);
        runtime->work_queue[tail] = packet;
        runtime->work_queue_submitted[tail] =
            runtime->work_queue_submitted[old_head];
        runtime->work_queue_forwarded[tail] =
            runtime->work_queue_forwarded[old_head];
        runtime->work_queue_state[tail] =
            runtime->work_queue_state[old_head];
        runtime->work_queue_submitted[old_head] = 0u;
        runtime->work_queue_forwarded[old_head] = 0u;
        runtime->work_queue_state[old_head] =
            SPARK_GLM52_RING_DAEMON_WORK_STATE_READY;
        SparkGlm52RingDaemonInsertQueuedWorkHash(runtime,tail);
    }
    runtime->work_queue_head = (runtime->work_queue_head + 1u) %
        SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
}

static void SparkGlm52RingDaemonWakeDeferredWork(
    SparkGlm52RingDaemonRuntime *runtime)
{
    uint32_t slot_index;
    uint32_t scan_index;

    if (runtime == 0 || runtime->work_queue_count == 0u)
        return;
    slot_index = runtime->work_queue_head;
    for (scan_index = 0u;
         scan_index < runtime->work_queue_count;
         ++scan_index)
    {
        if (runtime->work_queue_state[slot_index] !=
            SPARK_GLM52_RING_DAEMON_WORK_STATE_READY)
        {
            runtime->work_queue_state[slot_index] =
                SPARK_GLM52_RING_DAEMON_WORK_STATE_READY;
            runtime->work_wake_count += 1u;
        }
        slot_index = (slot_index + 1u) %
            SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
    }
}

static uint32_t SparkGlm52RingDaemonPumpQueuedWork(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkGlm52RingWorkControlPacket *packet;
    SparkStatus status;
    uint32_t queue_slot;
    uint32_t forward_done;
    uint32_t attempts;

    if (runtime == 0 || runtime->work_queue_count == 0u)
        return 0u;
    attempts = runtime->work_queue_count;
    while (attempts != 0u)
    {
        attempts -= 1u;
    queue_slot = runtime->work_queue_head;
    packet = &runtime->work_queue[queue_slot];
        if (runtime->work_queue_state[queue_slot] ==
            SPARK_GLM52_RING_DAEMON_WORK_STATE_WAITING_FORWARD)
            return 0u;
        if (runtime->work_queue_state[queue_slot] ==
            SPARK_GLM52_RING_DAEMON_WORK_STATE_WAITING_SUBMIT)
        {
            SparkGlm52RingDaemonDeferWork(runtime);
            continue;
        }
        if (SparkGlm52RingDaemonHasQueuedDependency(runtime,packet) != 0u)
        {
            SparkGlm52RingDaemonDeferWork(runtime);
            continue;
        }
        forward_done =
            ((runtime->rank_plan.flags &
              SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u ||
             runtime->work_queue_forwarded[queue_slot] != 0u)
                ? 1u : 0u;
        if (forward_done == 0u)
        {
            status = SparkGlm52RingDaemonForwardWork(runtime,packet);
            if (status == SPARK_STATUS_OK)
            {
                runtime->work_queue_forwarded[queue_slot] = 1u;
                forward_done = 1u;
            }
            else if (status == SPARK_STATUS_BUSY ||
                     status == SPARK_STATUS_ROUTE_NOT_FOUND)
            {
                runtime->work_queue_state[queue_slot] =
                    SPARK_GLM52_RING_DAEMON_WORK_STATE_WAITING_FORWARD;
                runtime->work_deferred_count += 1u;
                return 0u;
            }
            else
            {
                runtime->work_error_count += 1u;
                fprintf(stderr,
                    "rank_work_forward_failed status=%u request=%llu sequence=%llu position=%llu queued=%u\n",
                    (uint32_t)status,
                    (unsigned long long)packet->request_id,
                    (unsigned long long)packet->sequence_id,
                    (unsigned long long)packet->sequence_position,
                    runtime->work_queue_count);
                SparkGlm52RingDaemonPopWork(runtime);
                return 1u;
            }
        }
        if (runtime->work_queue_submitted[queue_slot] == 0u)
        {
            /*
             * submit_work may complete inline because the current builder
             * drains its CUDA slot before returning.  Account the submission
             * first so an inline completion cannot be lost and leave a
             * permanently positive inflight count.
             */
            if (runtime->driver_inflight_count == 0u)
            {
                runtime->driver_inflight_open_ns =
                    SparkGlm52RingDaemonMonotonicNs();
                runtime->driver_inflight_warned = 0u;
            }
            runtime->driver_inflight_count += 1u;
            status = SparkGlm52RingDaemonSubmitWork(runtime,packet);
            if (status == SPARK_STATUS_OK)
            {
                runtime->work_submit_count += 1u;
                runtime->work_queue_submitted[queue_slot] = 1u;
                if ((packet->flags &
                        SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u &&
                    runtime->driver_inflight_count != 0u)
                    runtime->driver_inflight_count -= 1u;
            }
            else if (status == SPARK_STATUS_BUSY)
            {
                if (runtime->driver_inflight_count != 0u)
                    runtime->driver_inflight_count -= 1u;
                runtime->work_queue_state[queue_slot] =
                    SPARK_GLM52_RING_DAEMON_WORK_STATE_WAITING_SUBMIT;
                runtime->work_deferred_count += 1u;
                SparkGlm52RingDaemonDeferWork(runtime);
                continue;
            }
            else
            {
                if (runtime->driver_inflight_count != 0u)
                    runtime->driver_inflight_count -= 1u;
                runtime->work_error_count += 1u;
                fprintf(stderr,
                    "rank_work_submit_failed status=%u request=%llu sequence=%llu position=%llu queued=%u\n",
                    (uint32_t)status,
                    (unsigned long long)packet->request_id,
                    (unsigned long long)packet->sequence_id,
                    (unsigned long long)packet->sequence_position,
                    runtime->work_queue_count);
                SparkGlm52RingDaemonPopWork(runtime);
                return 1u;
            }
        }
        if (forward_done != 0u &&
            runtime->work_queue_submitted[queue_slot] != 0u)
        {
            SparkGlm52RingDaemonPopWork(runtime);
            return 1u;
        }
    }
    return 0u;
}

static void SparkGlm52RingDaemonHandleWork(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingWorkControlPacket *packet)
{
    SparkStatus status;

    status = SparkGlm52RingWorkControlValidatePacket(
        packet,
        runtime->rank_plan.execution_row_capacity,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52RingDaemonQueueWork(runtime,packet);
    if (status != SPARK_STATUS_OK)
    {
        runtime->work_error_count += 1u;
        fprintf(stderr,
            "rank_work_queue_failed status=%u request=%llu sequence=%llu position=%llu queued=%u\n",
            (uint32_t)status,
            packet == 0 ? 0ull : (unsigned long long)packet->request_id,
            packet == 0 ? 0ull : (unsigned long long)packet->sequence_id,
            packet == 0 ? 0ull : (unsigned long long)packet->sequence_position,
            runtime == 0 ? 0u : runtime->work_queue_count);
    }
}

static uint32_t SparkGlm52RingDaemonPumpWorkControl(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkGlm52RingWorkControlPacket *packet;
    ssize_t got;
	uint32_t expected_bytes;
    uint32_t remaining;
    uint32_t progress;

    progress = SparkGlm52RingDaemonAcceptWorkSocket(runtime);
    if (runtime->work_listen_fd < 0 || runtime->work_input_socket_fd < 0)
        return progress;
    for (;;)
    {
		if (SparkGlm52RingDaemonWorkInputCanRead(runtime) == 0u)
			return progress;
		packet = (SparkGlm52RingWorkControlPacket *)runtime->work_read_buffer;
		expected_bytes = SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES;
		if (runtime->work_read_offset >=
			(uint32_t)offsetof(SparkGlm52RingWorkControlPacket,flags))
		{
			expected_bytes = packet->descriptor_bytes;
			if (expected_bytes <
					SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES ||
				expected_bytes > SPARK_GLM52_RING_WORK_CONTROL_PACKET_BYTES)
			{
				runtime->work_error_count += 1u;
				close(runtime->work_input_socket_fd);
				runtime->work_input_socket_fd = -1;
				runtime->work_read_offset = 0u;
				return 1u;
			}
		}
		remaining = expected_bytes - runtime->work_read_offset;
        got = read(
            runtime->work_input_socket_fd,
            runtime->work_read_buffer + runtime->work_read_offset,
            remaining);
        if (got < 0)
        {
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                runtime->work_error_count += 1u;
            return progress;
        }
        if (got == 0)
        {
            close(runtime->work_input_socket_fd);
            runtime->work_input_socket_fd = -1;
            runtime->work_read_offset = 0u;
            return 1u;
        }
        progress = 1u;
		runtime->work_read_offset += (uint32_t)got;
		if (runtime->work_read_offset ==
				SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES &&
			packet->descriptor_bytes >
				SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES)
			continue;
		if (runtime->work_read_offset < expected_bytes)
            return progress;
        SparkGlm52RingDaemonHandleWork(runtime,packet);
        runtime->work_read_offset = 0u;
    }
}

static SparkStatus SparkGlm52RingDaemonOpenFinalEventPath(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingDaemonConfig *configuration)
{
    if (runtime->rank_plan.rank_index == 0u &&
        configuration->own_final_event != 0u)
    {
        runtime->final_event_listen_fd =
            SparkGlm52RingDaemonCreateListenSocket(
                configuration->final_event_bind_address,
                runtime->final_event_route.listen_port);
        if (runtime->final_event_listen_fd < 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        if (SparkGlm52RingDaemonSetNonblocking(
                runtime->final_event_listen_fd) < 0)
            return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u)
        (void)configuration;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingDaemonEnsureFinalEventSocket(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkStatus status;
    uint64_t now_ns;

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u)
        return SPARK_STATUS_OK;
    if (runtime->final_event_socket_fd >= 0)
    {
        status = SparkGlm52RingDaemonFinishConnect(
            &runtime->final_event_socket_fd,
            &runtime->final_event_socket_connecting);
        if (status == SPARK_STATUS_OK)
            runtime->final_event_retry_mono_ns = 0u;
        else if (runtime->final_event_socket_fd < 0)
            runtime->final_event_retry_mono_ns =
                SparkGlm52RingDaemonMonotonicNs() +
                SPARK_GLM52_RING_DAEMON_CONNECT_RETRY_NS;
        return status;
    }
    now_ns = SparkGlm52RingDaemonMonotonicNs();
    if (runtime->final_event_retry_mono_ns != 0u &&
        now_ns < runtime->final_event_retry_mono_ns)
        return SPARK_STATUS_BUSY;
    runtime->final_event_socket_fd =
        SparkGlm52RingDaemonStartConnect(
            runtime->final_event_route.sink_host_name,
            runtime->final_event_route.connect_port,
            &runtime->final_event_socket_connecting);
    if (runtime->final_event_socket_fd < 0)
    {
        runtime->final_event_retry_mono_ns =
            now_ns + SPARK_GLM52_RING_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_BUSY;
    }
    runtime->final_event_retry_mono_ns = 0u;
    return runtime->final_event_socket_connecting == 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static void SparkGlm52RingDaemonResetFinalEventSocket(
    SparkGlm52RingDaemonRuntime *runtime)
{
    if (runtime->final_event_socket_fd >= 0)
        close(runtime->final_event_socket_fd);
    runtime->final_event_socket_fd = -1;
    runtime->final_event_socket_connecting = 0u;
    runtime->final_event_write_offset = 0u;
    runtime->final_event_read_offset = 0u;
}

static uint32_t SparkGlm52RingDaemonPumpFinalEventSend(
    SparkGlm52RingDaemonRuntime *runtime)
{
    const SparkGlm52RingRuntimeFinalEvent *event;
    SparkStatus status;

    if (runtime == 0 ||
        (runtime->rank_plan.flags &
            SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u ||
        runtime->final_event_queue_count == 0u)
        return 0u;
    status = SparkGlm52RingDaemonEnsureFinalEventSocket(runtime);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
        return 0u;
    event = &runtime->final_event_queue[runtime->final_event_queue_head];
    status = SparkGlm52RingDaemonWriteBuffered(
            runtime->final_event_socket_fd,
            event,
            sizeof(*event),
            &runtime->final_event_write_offset);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonResetFinalEventSocket(runtime);
        runtime->final_event_send_error_count += 1u;
        return 0u;
    }
    SparkGlm52RingDaemonPopFinalEvent(runtime);
    runtime->final_event_send_count += 1u;
    return 1u;
}

static uint32_t SparkGlm52RingDaemonAcceptFinalEventSocket(
    SparkGlm52RingDaemonRuntime *runtime)
{
    int32_t fd;

    if (runtime->final_event_listen_fd < 0 ||
        runtime->final_event_socket_fd >= 0)
        return 0u;
    fd = accept(runtime->final_event_listen_fd,0,0);
    if (fd < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            runtime->final_event_receive_error_count += 1u;
        return 0u;
    }
    if (SparkGlm52RingDaemonSetNonblocking(fd) < 0)
    {
        close(fd);
        runtime->final_event_receive_error_count += 1u;
        return 0u;
    }
    SparkGlm52RingDaemonConfigureTcpSocket(fd);
    runtime->final_event_socket_fd = fd;
    return 1u;
}

static void SparkGlm52RingDaemonPublishFinalEvent(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingRuntimeFinalEvent *event)
{
    uint32_t token_index;

    if (event->magic != SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_MAGIC ||
        event->descriptor_bytes != (uint32_t)sizeof(*event))
    {
        runtime->final_event_receive_error_count += 1u;
        return;
    }
    runtime->final_event_receive_count += 1u;
    printf("glm52_ring_final_event=1 request=%llu sequence=%llu position=%llu status=%u accepted=%u token_count=%u service_ns=%llu",
        (unsigned long long)event->request_id,
        (unsigned long long)event->sequence_id,
        (unsigned long long)event->sequence_position,
        event->status,
        event->accepted_token_count,
        event->token_count,
        (unsigned long long)event->service_time_ns);
    if ((event->completion_flags & SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) != 0u)
    {
        for (token_index = 0u;
             token_index < event->token_count &&
                token_index < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
             ++token_index)
        {
            printf(" token%u=%u",token_index,event->token_ids[token_index]);
        }
    }
    printf("\n");
    fflush(stdout);
}

static uint32_t SparkGlm52RingDaemonPumpFinalEventReceive(
    SparkGlm52RingDaemonRuntime *runtime)
{
    SparkGlm52RingRuntimeFinalEvent *event;
    ssize_t got;
    uint32_t remaining;
    uint32_t progress;

    progress = 0u;
    if (runtime == 0 || runtime->final_event_socket_fd < 0)
        return 0u;
    for (;;)
    {
        remaining = ((uint32_t)sizeof(runtime->final_event_read_buffer) -
            runtime->final_event_read_offset);
        got = read(
            runtime->final_event_socket_fd,
            runtime->final_event_read_buffer + runtime->final_event_read_offset,
            remaining);
        if (got < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                return progress;
            SparkGlm52RingDaemonResetFinalEventSocket(runtime);
            runtime->final_event_receive_error_count += 1u;
            return progress;
        }
        if (got == 0)
        {
            SparkGlm52RingDaemonResetFinalEventSocket(runtime);
            return 1u;
        }
        progress = 1u;
        runtime->final_event_read_offset += (uint32_t)got;
        if (runtime->final_event_read_offset < sizeof(*event))
            return progress;
        event =
            (SparkGlm52RingRuntimeFinalEvent *)runtime->final_event_read_buffer;
        SparkGlm52RingDaemonPublishFinalEvent(runtime,event);
        runtime->final_event_read_offset = 0u;
    }
}

static uint32_t SparkGlm52RingDaemonPumpFinalEvents(
    SparkGlm52RingDaemonRuntime *runtime)
{
    uint32_t progress;

    progress = SparkGlm52RingDaemonAcceptFinalEventSocket(runtime);
    progress |= SparkGlm52RingDaemonPumpFinalEventReceive(runtime);
    progress |= SparkGlm52RingDaemonPumpFinalEventSend(runtime);
    return progress;
}

static SparkStatus SparkGlm52RingDaemonInitialize(
    SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingDaemonConfig *configuration,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;

    memset(runtime,0,sizeof(*runtime));
    runtime->cuda_resident_fd = -1;
    runtime->trace_enabled = getenv("SPARKPIPE_RING_TRACE") != 0 ? 1u : 0u;
    runtime->work_listen_fd = -1;
    runtime->work_input_socket_fd = -1;
    runtime->work_output_socket_fd = -1;
    runtime->final_event_listen_fd = -1;
    runtime->final_event_socket_fd = -1;
    runtime->wake_pipe_read_fd = -1;
    runtime->wake_pipe_write_fd = -1;
    SparkGlm52RingDaemonInitializeWorkQueue(runtime);
    SparkLoadedModelDriverReset(&runtime->loaded_driver);
    status = SparkGlm52RingDaemonOpenWakePipe(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open rank daemon wake pipe");
        return status;
    }
    status = SparkGlm52RingRuntimeBuildRankPlan(
        configuration->rank_index,
        configuration->max_active_sequence_count,
        configuration->port_base,
        configuration->model_quantization_mode,
        &runtime->rank_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build RING rank plan");
        return status;
    }
    status = SparkGlm52RingRuntimeBuildFinalEventRoute(
        configuration->port_base,
        &runtime->final_event_route,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build RING final event route");
        return status;
    }
    if (configuration->cuda_resident_socket_path != 0)
    {
        runtime->cuda_resident_socket_path = configuration->cuda_resident_socket_path;
        status = SparkGlm52RingDaemonEnsureCudaResident(runtime);
        if (status != SPARK_STATUS_OK)
            fprintf(stderr,"rank_cuda_resident_not_ready_at_start rank=%u status=%u\n",runtime->rank_plan.rank_index,(uint32_t)status);
        status = SparkGlm52RingDaemonOpenWorkControlPath(runtime);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52RingDaemonSetDefaultError(
                error_buffer,
                error_buffer_bytes,
                "failed to open production work-control path");
            return status;
        }
        status = SparkGlm52RingDaemonOpenFinalEventPath(runtime,configuration);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52RingDaemonSetDefaultError(
                error_buffer,
                error_buffer_bytes,
                "failed to open final event route");
            return status;
        }
        return SPARK_STATUS_OK;
    }
    status = SparkGlm52RingRuntimeValidateStageMoePackFiles(
        &runtime->rank_plan,
        configuration->moe_pack_root,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to validate rank FP8 pack files");
        return status;
    }
    status = SparkGlm52RingDaemonBuildNodeContext(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build resident node context");
        return status;
    }
    status = SparkGlm52RingDaemonLoadTransport(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to load production hidden transport");
        return status;
    }
    status = SparkGlm52RingDaemonOpenHiddenTransport(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open neighbor hidden transport session");
        return status;
    }
    status = SparkGlm52RingDaemonLoadDriver(
        runtime,
        configuration,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to load GLM52 model driver");
        return status;
    }
    status = SparkGlm52RingDaemonAttachBuilderDriver(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to attach node-context builder to driver");
        return status;
    }
    status = SparkGlm52RingDaemonInitializeRunner(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to initialize production stage runner");
        return status;
    }
    status = SparkGlm52RingDaemonOpenWorkControlPath(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open production work-control path");
        return status;
    }
    status = SparkGlm52RingDaemonOpenFinalEventPath(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open final event route");
        return status;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52RingDaemonDestroy(
    SparkGlm52RingDaemonRuntime *runtime)
{
    if (runtime == 0)
        return;
    if (runtime->cuda_resident_fd >= 0)
        close(runtime->cuda_resident_fd);
    if (runtime->work_output_socket_fd >= 0)
        close(runtime->work_output_socket_fd);
    if (runtime->work_input_socket_fd >= 0)
        close(runtime->work_input_socket_fd);
    if (runtime->work_listen_fd >= 0)
        close(runtime->work_listen_fd);
    if (runtime->final_event_socket_fd >= 0)
        close(runtime->final_event_socket_fd);
    if (runtime->final_event_listen_fd >= 0)
        close(runtime->final_event_listen_fd);
    if (runtime->wake_pipe_read_fd >= 0)
        close(runtime->wake_pipe_read_fd);
    if (runtime->wake_pipe_write_fd >= 0)
        close(runtime->wake_pipe_write_fd);
    if (runtime->loaded_driver.interface != 0 &&
        runtime->loaded_driver.interface->destroy != 0 &&
        runtime->driver_instance != 0)
        runtime->loaded_driver.interface->destroy(runtime->driver_instance);
    SparkUnloadModelDriver(&runtime->loaded_driver);
    if (runtime->builder_library.builder_interface.destroy_result != 0 &&
        runtime->builder_state != 0)
        runtime->builder_library.builder_interface.destroy_result(
            runtime->builder_state,
            &runtime->builder_result);
    if (runtime->builder_library.builder_interface.destroy != 0 &&
        runtime->builder_state != 0)
        runtime->builder_library.builder_interface.destroy(
            runtime->builder_state);
    SparkGlm52RingNodeContextBuilderUnloadInterface(&runtime->builder_library);
    SparkHiddenTransportClose(runtime->input_transport_session);
    SparkHiddenTransportClose(runtime->output_transport_session);
    SparkHiddenTransportUnloadInterface(&runtime->transport_library);
}

static void SparkGlm52RingDaemonPrintReady(
    const SparkGlm52RingDaemonRuntime *runtime,
    const SparkGlm52RingDaemonConfig *configuration)
{
    printf("glm52_ring_rank_daemon=1\n");
    printf("rank=%u\n",runtime->rank_plan.rank_index);
    printf("host=%s\n",runtime->rank_plan.host_name);
    printf("stage=%u:%u\n",
        runtime->rank_plan.first_layer_index,
        runtime->rank_plan.layer_count);
    printf("cuda_resident_socket=%s\n",
        configuration->cuda_resident_socket_path != 0 ?
            configuration->cuda_resident_socket_path : "");
    printf("model_quantization=%s\n",
        SparkGlm52RingRuntimeQuantizationModeName(
            configuration->model_quantization_mode));
    printf("moe_pack_root=%s\n",
        configuration->moe_pack_root != 0 ? configuration->moe_pack_root : "");
    printf("stagepack_root=%s\n",
        configuration->stagepack_root != 0 ? configuration->stagepack_root : "");
    printf("transport_so=%s\n",
        configuration->transport_shared_object_path != 0 ?
            configuration->transport_shared_object_path : "");
    printf("driver_so=%s\n",
        configuration->driver_path != 0 ? configuration->driver_path : "");
    printf("node_context_builder_so=%s\n",
        configuration->node_context_builder_shared_object_path != 0 ?
            configuration->node_context_builder_shared_object_path : "");
    printf("embedding_pack=%s\n",
        configuration->embedding_pack_path != 0 ?
            configuration->embedding_pack_path : "");
    printf("input_transport=%u\n",
        runtime->input_transport_session != 0 ? 1u : 0u);
    printf("output_transport=%u\n",
        runtime->output_transport_session != 0 ? 1u : 0u);
    printf("transport_busy_poll=%u\n",configuration->transport_busy_poll);
    printf("work_listen=%d\n",runtime->work_listen_fd);
    printf("work_input_socket=%d\n",runtime->work_input_socket_fd);
    printf("work_output_socket=%d\n",runtime->work_output_socket_fd);
    printf("work_received=%llu\n",
        (unsigned long long)runtime->work_receive_count);
    printf("work_forwarded=%llu\n",
        (unsigned long long)runtime->work_forward_count);
    printf("work_submitted=%llu\n",
        (unsigned long long)runtime->work_submit_count);
    printf("work_duplicates=%llu\n",
        (unsigned long long)runtime->work_duplicate_count);
    printf("work_deferred=%llu\n",
        (unsigned long long)runtime->work_deferred_count);
    printf("work_woken=%llu\n",
        (unsigned long long)runtime->work_wake_count);
    printf("work_errors=%llu\n",
        (unsigned long long)runtime->work_error_count);
    printf("wake_signals=%llu\n",
        (unsigned long long)runtime->wake_signal_count);
    printf("driver_inflight=%u\n",runtime->driver_inflight_count);
    printf("driver_completions=%llu\n",
        (unsigned long long)runtime->driver_completion_count);
    printf("cuda_resident_submits=%llu\n",
        (unsigned long long)runtime->cuda_resident_submit_count);
    printf("cuda_resident_completions=%llu\n",
        (unsigned long long)runtime->cuda_resident_completion_count);
    printf("cuda_resident_errors=%llu\n",
        (unsigned long long)runtime->cuda_resident_error_count);
    printf("wake_dropped=%llu\n",
        (unsigned long long)runtime->wake_drop_count);
    printf("timer_wakes=%llu\n",
        (unsigned long long)runtime->timer_wake_count);
    printf("final_event_listen=%d\n",runtime->final_event_listen_fd);
    printf("final_event_socket=%d\n",runtime->final_event_socket_fd);
    printf("final_event_sent=%llu\n",
        (unsigned long long)runtime->final_event_send_count);
    printf("final_event_received=%llu\n",
        (unsigned long long)runtime->final_event_receive_count);
    printf("final_event_queued=%u\n",runtime->final_event_queue_count);
    fflush(stdout);
}

int main(int argc,char **argv)
{
    SparkGlm52RingDaemonConfig configuration;
    static SparkGlm52RingDaemonRuntime runtime;
    char error_buffer[512];
    SparkStatus status;
    uint32_t event_mask;
    uint32_t progress;
    uint64_t timeout_ns;
    uint64_t next_timer_ns;
    uint64_t now_ns;
	SparkStatus builder_status;

    SparkGlm52RingDaemonInitializeConfig(&configuration);
    if (SparkGlm52RingDaemonParseArguments(&configuration,argc,argv) < 0)
    {
        fprintf(stderr,
            "usage: %s --rank n [--cuda-resident-socket path | --model-quantization fp8|nvfp4|w8lut --moe-pack-root dir --stagepack-root dir --transport-so path --driver-so path --node-context-builder-so path --embedding-pack path] [--program name] [--node-target target] [--max-active n] [--port-base n] [--final-event-bind ip] [--final-event-return-host host] [--transport-busy-poll]\n",
            argv[0]);
        return 2;
    }
    signal(SIGINT,SparkGlm52RingDaemonSignal);
    signal(SIGTERM,SparkGlm52RingDaemonSignal);
    signal(SIGPIPE,SIG_IGN);
    error_buffer[0] = '\0';
    status = SparkGlm52RingDaemonInitialize(
        &runtime,
        &configuration,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"rank daemon init failed status=%u error=%s\n",
            (uint32_t)status,
            error_buffer);
        SparkGlm52RingDaemonDestroy(&runtime);
        return 3;
    }
    if (configuration.transport_busy_poll != 0u &&
        (runtime.transport_library.transport_interface.capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA) == 0u)
    {
        fprintf(stderr,
            "rank daemon transport busy poll requires Spark host RDMA\n");
        SparkGlm52RingDaemonDestroy(&runtime);
        return 3;
    }
    SparkGlm52RingDaemonPrintReady(&runtime,&configuration);
    while (SparkGlm52RingDaemonRunning != 0)
    {
        progress = 0u;
        if (SparkGlm52RingDaemonDrainWakePipe(&runtime) != 0u)
        {
            progress = 1u;
            SparkGlm52RingDaemonWakeDeferredWork(&runtime);
        }
        progress |= SparkGlm52RingDaemonPumpWorkControl(&runtime);
        progress |= SparkGlm52RingDaemonPumpCudaResident(&runtime);
        SparkGlm52RingDaemonCheckInflightOverdue(&runtime);
        if (runtime.cuda_resident_fd < 0)
            (void)SparkGlm52ResidentDecodeStageProductionRunnerProgress(
                &runtime.runner);
		builder_status = SparkGlm52RingDaemonProgressBuilder(&runtime);
		if (builder_status != SPARK_STATUS_OK &&
			builder_status != SPARK_STATUS_BUSY)
		{
			fprintf(stderr,
				"rank daemon builder progress failed status=%u\n",
				(uint32_t)builder_status);
			break;
		}
        progress |= SparkGlm52RingDaemonPumpQueuedWork(&runtime);
        progress |= SparkGlm52RingDaemonPumpFinalEvents(&runtime);
        if (progress == 0u)
        {
            if (configuration.transport_busy_poll != 0u)
                continue;
            timeout_ns = builder_status == SPARK_STATUS_BUSY ?
				UINT64_C(1000000) : 0u;
            next_timer_ns = SparkGlm52RingDaemonNextTimerNs(&runtime);
            if (next_timer_ns != 0u)
            {
                now_ns = SparkGlm52RingDaemonMonotonicNs();
				next_timer_ns = next_timer_ns > now_ns ?
					next_timer_ns - now_ns : 1u;
				if (timeout_ns == 0u || next_timer_ns < timeout_ns)
					timeout_ns = next_timer_ns;
            }
            event_mask = SparkGlm52RingDaemonWaitForEvents(
                &runtime,
                timeout_ns);
            if ((event_mask &
                (SPARK_GLM52_RING_DAEMON_POLL_KIND_INPUT_TRANSPORT |
                 SPARK_GLM52_RING_DAEMON_POLL_KIND_OUTPUT_TRANSPORT |
                 SPARK_GLM52_RING_DAEMON_POLL_KIND_WORK |
                 SPARK_GLM52_RING_DAEMON_POLL_KIND_WORK_OUTPUT |
                 SPARK_GLM52_RING_DAEMON_POLL_KIND_CUDA_RESIDENT |
                 SPARK_GLM52_RING_DAEMON_POLL_KIND_FINAL_EVENT |
                 SPARK_GLM52_RING_DAEMON_POLL_KIND_COMPLETION_WAKE |
                 SPARK_GLM52_RING_DAEMON_POLL_KIND_TIMER)) != 0u)
            {
                if ((event_mask & SPARK_GLM52_RING_DAEMON_POLL_KIND_TIMER) != 0u)
                    runtime.timer_wake_count += 1u;
                SparkGlm52RingDaemonWakeDeferredWork(&runtime);
            }
        }
    }
    SparkGlm52RingDaemonDestroy(&runtime);
    return 0;
}
