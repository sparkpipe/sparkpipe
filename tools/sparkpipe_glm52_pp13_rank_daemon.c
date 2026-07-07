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
#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"
#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_glm52_pp13_work_control.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"

#define SPARK_GLM52_PP13_DAEMON_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_GLM52_PP13_DAEMON_DEFAULT_PROGRAM "glm52.pp13.rank.production"
#define SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_MAGIC 0x35454650u
#define SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY 2048u
#define SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_HASH_SLOTS 4096u
#define SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_QUEUE_CAPACITY 2048u
#define SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT UINT32_MAX
#define SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY 32u
#define SPARK_GLM52_PP13_DAEMON_WORK_STATE_READY 0u
#define SPARK_GLM52_PP13_DAEMON_WORK_STATE_WAITING 1u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_WORK 0x00000001u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_FINAL_EVENT 0x00000002u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_INPUT_TRANSPORT 0x00000004u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_COMPLETION_WAKE 0x00000010u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_WORK_OUTPUT 0x00000020u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_TIMER 0x00000040u
#define SPARK_GLM52_PP13_DAEMON_POLL_KIND_CUDA_RESIDENT 0x00000080u
#define SPARK_GLM52_PP13_DAEMON_CONNECT_RETRY_NS 250000ull
#define SPARK_GLM52_PP13_DAEMON_RUNNER_PROGRESS_NS 250000ull

typedef struct SparkGlm52Pp13DaemonConfig
{
    const char *fp8_pack_root;
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
    uint32_t max_active_sequence_count;
    uint32_t port_base;
} SparkGlm52Pp13DaemonConfig;

typedef struct SparkGlm52Pp13DaemonFinalEvent
{
    uint32_t magic;
    uint32_t descriptor_bytes;
    uint32_t status;
    uint32_t program_id;
    uint32_t driver_dispatch_slot;
    uint32_t accepted_token_count;
    uint32_t completion_flags;
    uint32_t token_count;
    uint32_t token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t service_time_ns;
} SparkGlm52Pp13DaemonFinalEvent;

typedef struct SparkGlm52Pp13DaemonRuntime
{
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    SparkGlm52Pp13RuntimeFinalEventRoute final_event_route;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *input_transport_session;
    SparkHiddenTransportSession *output_transport_session;
    SparkGlm52Pp13NodeContextBuilderDynamicLibrary builder_library;
    void *builder_state;
    SparkGlm52Pp13NodeContextBuilderResult builder_result;
    SparkLoadedModelDriver loaded_driver;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    int32_t cuda_resident_fd;
    uint64_t cuda_resident_next_sequence_number;
    uint64_t cuda_resident_submit_count;
    uint64_t cuda_resident_completion_count;
    uint64_t cuda_resident_error_count;
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
    uint8_t work_read_buffer[SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES];
    uint32_t work_read_offset;
    SparkGlm52Pp13WorkControlPacket work_queue[SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_hash_heads[SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_HASH_SLOTS];
    uint32_t work_queue_hash_next[SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_submitted[SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_forwarded[SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_state[SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY];
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
    uint32_t driver_inflight_count;
    uint8_t final_event_read_buffer[128];
    uint32_t final_event_read_offset;
    SparkGlm52Pp13DaemonFinalEvent final_event_queue[
        SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_QUEUE_CAPACITY];
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
} SparkGlm52Pp13DaemonRuntime;

static volatile sig_atomic_t SparkGlm52Pp13DaemonRunning = 1;

static void SparkGlm52Pp13DaemonCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion);

static void SparkGlm52Pp13DaemonSignal(int signal_number)
{
    (void)signal_number;
    SparkGlm52Pp13DaemonRunning = 0;
}

static void SparkGlm52Pp13DaemonInitializeConfig(
    SparkGlm52Pp13DaemonConfig *configuration)
{
    memset(configuration,0,sizeof(*configuration));
    configuration->program_name = SPARK_GLM52_PP13_DAEMON_DEFAULT_PROGRAM;
    configuration->final_event_bind_address = "0.0.0.0";
    configuration->final_event_return_host = "spark0";
    configuration->max_active_sequence_count =
        SPARK_GLM52_PP13_DAEMON_DEFAULT_MAX_ACTIVE;
    configuration->port_base = SPARK_GLM52_PP13_RUNTIME_DEFAULT_PORT_BASE;
}

static int32_t SparkGlm52Pp13DaemonParseU32(
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

static void SparkGlm52Pp13DaemonSetDefaultError(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    const char *message)
{
    if (error_buffer == 0 || error_buffer_bytes == 0u ||
        error_buffer[0] != '\0')
        return;
    snprintf(error_buffer,error_buffer_bytes,"%s",message);
}

static void SparkGlm52Pp13DaemonSetStatusError(
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

static int32_t SparkGlm52Pp13DaemonApplyArgument(
    SparkGlm52Pp13DaemonConfig *configuration,
    int argc,
    char **argv,
    int32_t *index)
{
    uint32_t parsed;

    if (strcmp(argv[*index],"--rank") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52Pp13DaemonParseU32(argv[*index + 1],&parsed) < 0)
            return -1;
        configuration->rank_index = parsed;
        configuration->rank_is_set = 1u;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--fp8-pack-root") == 0)
    {
        if ((*index + 1) >= argc)
            return -2;
        configuration->fp8_pack_root = argv[*index + 1];
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
            SparkGlm52Pp13DaemonParseU32(argv[*index + 1],&parsed) < 0)
            return -9;
        configuration->max_active_sequence_count = parsed;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--port-base") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52Pp13DaemonParseU32(argv[*index + 1],&parsed) < 0)
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
    return -13;
}

static int32_t SparkGlm52Pp13DaemonParseArguments(
    SparkGlm52Pp13DaemonConfig *configuration,
    int argc,
    char **argv)
{
    int32_t index;

    for (index = 1; index < argc; ++index)
    {
        if (SparkGlm52Pp13DaemonApplyArgument(
                configuration,argc,argv,&index) < 0)
            return -1;
    }
    if (configuration->rank_is_set == 0u)
        return -2;
    if (configuration->cuda_resident_socket_path != 0)
        return 0;
    if (configuration->fp8_pack_root == 0 ||
        configuration->stagepack_root == 0 ||
        configuration->transport_shared_object_path == 0 ||
        configuration->driver_path == 0 ||
        configuration->node_context_builder_shared_object_path == 0 ||
        configuration->embedding_pack_path == 0)
        return -2;
    return 0;
}

static int32_t SparkGlm52Pp13DaemonCreateListenSocket(
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

static int32_t SparkGlm52Pp13DaemonSetNonblocking(int32_t fd)
{
    int32_t flags;

    flags = fcntl(fd,F_GETFL,0);
    if (flags < 0)
        return -1;
    if (fcntl(fd,F_SETFL,(flags | O_NONBLOCK)) < 0)
        return -2;
    return 0;
}

static uint64_t SparkGlm52Pp13DaemonMonotonicNs(void)
{
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0)
        return 0u;
    return ((uint64_t)timestamp.tv_sec * 1000000000ull) +
        (uint64_t)timestamp.tv_nsec;
}

static void SparkGlm52Pp13DaemonConfigureTcpSocket(int32_t fd)
{
    int32_t option;

    if (fd < 0)
        return;
    option = 1;
    (void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&option,sizeof(option));
    (void)setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&option,sizeof(option));
}

static uint64_t SparkGlm52Pp13DaemonMinNonzeroNs(
    uint64_t left,
    uint64_t right)
{
    if (left == 0u)
        return right;
    if (right == 0u)
        return left;
    return left < right ? left : right;
}

static uint64_t SparkGlm52Pp13DaemonNextTimerNs(
    const SparkGlm52Pp13DaemonRuntime *runtime)
{
    uint64_t next_ns;
    uint64_t now_ns;

    if (runtime == 0)
        return 0u;
    now_ns = SparkGlm52Pp13DaemonMonotonicNs();
    next_ns = 0u;
    if (runtime->driver_inflight_count != 0u)
        next_ns = now_ns + SPARK_GLM52_PP13_DAEMON_RUNNER_PROGRESS_NS;
    if (runtime->work_queue_count != 0u)
        next_ns = SparkGlm52Pp13DaemonMinNonzeroNs(
            next_ns,
            runtime->work_output_retry_mono_ns);
    if (runtime->final_event_queue_count != 0u)
        next_ns = SparkGlm52Pp13DaemonMinNonzeroNs(
            next_ns,
            runtime->final_event_retry_mono_ns);
    return next_ns;
}

#if !defined(__linux__)
static int32_t SparkGlm52Pp13DaemonPollTimeoutMs(uint64_t timeout_ns)
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

static int32_t SparkGlm52Pp13DaemonAppendPollFd(
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

static int16_t SparkGlm52Pp13DaemonPollEventsFromTransport(
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

static void SparkGlm52Pp13DaemonAppendTransportPollFds(
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
        events = SparkGlm52Pp13DaemonPollEventsFromTransport(
            descriptors[descriptor_index].events);
        if (SparkGlm52Pp13DaemonAppendPollFd(
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

static uint32_t SparkGlm52Pp13DaemonWaitForEvents(
    SparkGlm52Pp13DaemonRuntime *runtime,
    uint64_t timeout_ns)
{
    struct pollfd fds[SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY];
    uint32_t fd_kinds[SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY];
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
        (void)SparkGlm52Pp13DaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_listen_fd,
            POLLIN,
            SPARK_GLM52_PP13_DAEMON_POLL_KIND_WORK);
    if (runtime->work_input_socket_fd >= 0)
        (void)SparkGlm52Pp13DaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_input_socket_fd,
            POLLIN,
            SPARK_GLM52_PP13_DAEMON_POLL_KIND_WORK);
    if (runtime->work_output_socket_fd >= 0 &&
        (runtime->work_output_connecting != 0u ||
         runtime->work_queue_count != 0u))
        (void)SparkGlm52Pp13DaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_output_socket_fd,
            POLLOUT,
            SPARK_GLM52_PP13_DAEMON_POLL_KIND_WORK_OUTPUT);
    if (runtime->final_event_listen_fd >= 0 &&
        runtime->final_event_socket_fd < 0)
        (void)SparkGlm52Pp13DaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->final_event_listen_fd,
            POLLIN,
            SPARK_GLM52_PP13_DAEMON_POLL_KIND_FINAL_EVENT);
    if (runtime->final_event_socket_fd >= 0)
    {
        int16_t final_events;
        final_events = POLLIN;
        if ((runtime->rank_plan.flags &
                SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
            (runtime->final_event_socket_connecting != 0u ||
             runtime->final_event_queue_count != 0u))
            final_events |= POLLOUT;
        (void)SparkGlm52Pp13DaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->final_event_socket_fd,
            final_events,
            SPARK_GLM52_PP13_DAEMON_POLL_KIND_FINAL_EVENT);
    }
    if (runtime->wake_pipe_read_fd >= 0)
        (void)SparkGlm52Pp13DaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->wake_pipe_read_fd,
            POLLIN,
            SPARK_GLM52_PP13_DAEMON_POLL_KIND_COMPLETION_WAKE);
    if (runtime->cuda_resident_fd >= 0)
        (void)SparkGlm52Pp13DaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->cuda_resident_fd,
            POLLIN,
            SPARK_GLM52_PP13_DAEMON_POLL_KIND_CUDA_RESIDENT);
    SparkGlm52Pp13DaemonAppendTransportPollFds(
        runtime->input_transport_session,
        fds,
        fd_kinds,
        SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
        &fd_count,
        SPARK_GLM52_PP13_DAEMON_POLL_KIND_INPUT_TRANSPORT);
    SparkGlm52Pp13DaemonAppendTransportPollFds(
        runtime->output_transport_session,
        fds,
        fd_kinds,
        SPARK_GLM52_PP13_DAEMON_POLL_FD_CAPACITY,
        &fd_count,
        SPARK_GLM52_PP13_DAEMON_POLL_KIND_OUTPUT_TRANSPORT);
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
            SparkGlm52Pp13DaemonPollTimeoutMs(timeout_ns));
#endif
        if (result < 0 && errno == EINTR)
            return 0u;
        if (result == 0)
            return SPARK_GLM52_PP13_DAEMON_POLL_KIND_TIMER;
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

static int32_t SparkGlm52Pp13DaemonStartConnect(
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
        if (SparkGlm52Pp13DaemonSetNonblocking(fd) < 0)
        {
            close(fd);
            fd = -1;
            continue;
        }
        SparkGlm52Pp13DaemonConfigureTcpSocket(fd);
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

static SparkStatus SparkGlm52Pp13DaemonFinishConnect(
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
        SparkGlm52Pp13DaemonConfigureTcpSocket(*fd);
        return SPARK_STATUS_OK;
    }
    close(*fd);
    *fd = -1;
    *connecting = 0u;
    return SPARK_STATUS_ROUTE_NOT_FOUND;
}

static SparkStatus SparkGlm52Pp13DaemonWriteBuffered(
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


static SparkStatus SparkGlm52Pp13DaemonWriteFull(
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

static SparkStatus SparkGlm52Pp13DaemonReadFull(
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

static SparkStatus SparkGlm52Pp13DaemonWriteResidentMessage(
    SparkGlm52Pp13DaemonRuntime *runtime,
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
    status = SparkGlm52Pp13DaemonWriteFull(
        runtime->cuda_resident_fd,
        &header,
        sizeof(header));
    if (status != SPARK_STATUS_OK)
        return status;
    if (payload_bytes != 0u)
        status = SparkGlm52Pp13DaemonWriteFull(
            runtime->cuda_resident_fd,
            payload,
            payload_bytes);
    return status;
}


static SparkStatus SparkGlm52Pp13DaemonConnectCudaResident(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration)
{
    struct sockaddr_un address;
    SparkGlm52CudaResidentIpcHello hello;
    SparkGlm52CudaResidentIpcHeader header;
    SparkGlm52CudaResidentIpcStats stats;
    SparkStatus status;
    int32_t fd;

    if (runtime == 0 || configuration == 0 ||
        configuration->cuda_resident_socket_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    fd = socket(AF_UNIX,SOCK_STREAM,0);
    if (fd < 0)
        return SPARK_STATUS_IO_ERROR;
    memset(&address,0,sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(configuration->cuda_resident_socket_path) >=
        sizeof(address.sun_path))
    {
        close(fd);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    snprintf(address.sun_path,sizeof(address.sun_path),"%s",
        configuration->cuda_resident_socket_path);
    if (connect(fd,(const struct sockaddr *)&address,sizeof(address)) != 0)
    {
        close(fd);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->cuda_resident_fd = fd;
    memset(&hello,0,sizeof(hello));
    hello.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_HELLO_BYTES;
    hello.rank_index = runtime->rank_plan.rank_index;
    hello.rank_count = SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT;
    hello.process_id = (uint64_t)getpid();
    status = SparkGlm52Pp13DaemonWriteResidentMessage(
        runtime,
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO,
        &hello,
        sizeof(hello));
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13DaemonReadFull(fd,&header,sizeof(header));
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentIpcValidateHeader(
        &header,
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO_ACK,
        sizeof(stats));
    if (status != SPARK_STATUS_OK)
        return status;
    if (header.payload_bytes != sizeof(stats))
        return SPARK_STATUS_ABI_MISMATCH;
    status = SparkGlm52Pp13DaemonReadFull(fd,&stats,sizeof(stats));
    if (status != SPARK_STATUS_OK)
        return status;
    if (stats.state != SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_READY)
        return SPARK_STATUS_BUSY;
    if (SparkGlm52Pp13DaemonSetNonblocking(fd) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52Pp13DaemonHandleCudaResidentMessage(
    SparkGlm52Pp13DaemonRuntime *runtime,
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
            SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_BYTES)
        {
            runtime->cuda_resident_error_count += 1u;
            return 0u;
        }
        runtime->cuda_resident_completion_count += 1u;
        SparkGlm52Pp13DaemonCompletion(
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

static uint32_t SparkGlm52Pp13DaemonPumpCudaResident(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkGlm52CudaResidentIpcHeader header;
    uint8_t payload[4096];
    SparkStatus status;

    if (runtime == 0 || runtime->cuda_resident_fd < 0)
        return 0u;
    status = SparkGlm52Pp13DaemonReadFull(
        runtime->cuda_resident_fd,
        &header,
        sizeof(header));
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        close(runtime->cuda_resident_fd);
        runtime->cuda_resident_fd = -1;
        runtime->cuda_resident_error_count += 1u;
        return 1u;
    }
    status = SparkGlm52CudaResidentIpcValidateHeader(
        &header,
        0u,
        sizeof(payload));
    if (status != SPARK_STATUS_OK)
    {
        close(runtime->cuda_resident_fd);
        runtime->cuda_resident_fd = -1;
        runtime->cuda_resident_error_count += 1u;
        return 1u;
    }
    if (header.payload_bytes != 0u)
    {
        status = SparkGlm52Pp13DaemonReadFull(
            runtime->cuda_resident_fd,
            payload,
            header.payload_bytes);
        if (status != SPARK_STATUS_OK)
        {
            close(runtime->cuda_resident_fd);
            runtime->cuda_resident_fd = -1;
            runtime->cuda_resident_error_count += 1u;
            return 1u;
        }
    }
    (void)SparkGlm52Pp13DaemonHandleCudaResidentMessage(
        runtime,
        &header,
        header.payload_bytes == 0u ? 0 : payload);
    return 1u;
}

static SparkStatus SparkGlm52Pp13DaemonOpenWakePipe(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    int32_t pipe_fds[2];

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (pipe(pipe_fds) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (SparkGlm52Pp13DaemonSetNonblocking(pipe_fds[0]) < 0 ||
        SparkGlm52Pp13DaemonSetNonblocking(pipe_fds[1]) < 0)
    {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->wake_pipe_read_fd = pipe_fds[0];
    runtime->wake_pipe_write_fd = pipe_fds[1];
    return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13DaemonSignalWake(
    SparkGlm52Pp13DaemonRuntime *runtime)
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

static SparkStatus SparkGlm52Pp13DaemonQueueFinalEvent(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonFinalEvent *event)
{
    uint32_t tail;

    if (runtime == 0 || event == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (runtime->final_event_queue_count >=
        SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_QUEUE_CAPACITY)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    tail =
        (runtime->final_event_queue_head + runtime->final_event_queue_count) %
        SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_QUEUE_CAPACITY;
    runtime->final_event_queue[tail] = *event;
    runtime->final_event_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13DaemonPopFinalEvent(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->final_event_queue_count == 0u)
        return;
    runtime->final_event_queue_head =
        (runtime->final_event_queue_head + 1u) %
        SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_QUEUE_CAPACITY;
    runtime->final_event_queue_count -= 1u;
}

static uint32_t SparkGlm52Pp13DaemonDrainWakePipe(
    SparkGlm52Pp13DaemonRuntime *runtime)
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

static void SparkGlm52Pp13DaemonCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkGlm52Pp13DaemonRuntime *runtime;
    SparkGlm52Pp13DaemonFinalEvent event;

    runtime = (SparkGlm52Pp13DaemonRuntime *)completion_context;
    if (runtime == 0 || completion == 0)
        return;
    if (runtime->driver_inflight_count != 0u)
        runtime->driver_inflight_count -= 1u;
    runtime->driver_completion_count += 1u;
    SparkGlm52Pp13DaemonSignalWake(runtime);
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u)
        return;
    if ((completion->completion_flags &
            SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u ||
        completion->token_count == 0u)
        return;
    memset(&event,0,sizeof(event));
    event.magic = SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_MAGIC;
    event.descriptor_bytes = (uint32_t)sizeof(event);
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
    event.request_id = completion->request_id;
    event.sequence_id = completion->sequence_id;
    event.sequence_position = completion->sequence_position;
    event.service_time_ns = completion->service_time_ns;
    if (SparkGlm52Pp13DaemonQueueFinalEvent(runtime,&event) != SPARK_STATUS_OK)
    {
        runtime->final_event_send_error_count += 1u;
        return;
    }
    SparkGlm52Pp13DaemonSignalWake(runtime);
}

static SparkStatus SparkGlm52Pp13DaemonLoadTransport(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration)
{
    char rank_buffer[16];
    char port_buffer[16];
    if (snprintf(rank_buffer, sizeof(rank_buffer), "%u",
            runtime->rank_plan.rank_index) < 0 ||
        snprintf(port_buffer, sizeof(port_buffer), "%u",
            configuration->port_base) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (setenv("SPARKPIPE_PP13_TRANSPORT_RANK",rank_buffer,1) != 0 ||
        setenv("SPARKPIPE_PP13_TRANSPORT_PORT_BASE",port_buffer,1) != 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SparkHiddenTransportLoadInterfaceFromSharedObject(
        configuration->transport_shared_object_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        &runtime->transport_library);
}

static SparkStatus SparkGlm52Pp13DaemonOpenHiddenTransport(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkStatus status;

    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
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
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
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

static SparkStatus SparkGlm52Pp13DaemonLoadDriver(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration,
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
        SparkGlm52Pp13DaemonSetStatusError(
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
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "decode program not found in GLM52 model driver");
        return SPARK_STATUS_NOT_FOUND;
    }
    memset(&create_request,0,sizeof(create_request));
    create_request.node_id = runtime->rank_plan.host_name;
    create_request.node_target = configuration->node_target;
    create_request.node_context = runtime->builder_result.node_context;
    create_request.completion_function = SparkGlm52Pp13DaemonCompletion;
    create_request.completion_context = runtime;
    status = runtime->loaded_driver.interface->create(
        &create_request,
        &runtime->driver_instance);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetStatusError(
            error_buffer,
            error_buffer_bytes,
            "GLM52 model driver create failed",
            status);
        return status;
    }
    if (runtime->driver_instance == 0)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "GLM52 model driver returned NULL instance");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13DaemonBuildNodeContext(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration)
{
    SparkGlm52Pp13NodeContextBuilderConfiguration builder_configuration;
    SparkStatus status;

    memset(&builder_configuration,0,sizeof(builder_configuration));
    builder_configuration.abi_version =
        SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION;
    builder_configuration.descriptor_bytes =
        SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES;
    builder_configuration.rank_index = runtime->rank_plan.rank_index;
    builder_configuration.max_active_sequence_count =
        configuration->max_active_sequence_count;
    builder_configuration.port_base = configuration->port_base;
    builder_configuration.fp8_pack_root = configuration->fp8_pack_root;
    builder_configuration.stagepack_root = configuration->stagepack_root;
    builder_configuration.embedding_pack_path =
        configuration->embedding_pack_path;
    builder_configuration.node_target = configuration->node_target;
    builder_configuration.rank_plan = &runtime->rank_plan;
    status = SparkGlm52Pp13NodeContextBuilderLoadInterfaceFromSharedObject(
        configuration->node_context_builder_shared_object_path,
        SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
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
    return SparkGlm52Pp13NodeContextBuilderValidateResult(
        &runtime->builder_result,
        &runtime->rank_plan);
}

static SparkStatus SparkGlm52Pp13DaemonAttachBuilderDriver(
    SparkGlm52Pp13DaemonRuntime *runtime)
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

static SparkStatus SparkGlm52Pp13DaemonInitializeRunner(
    SparkGlm52Pp13DaemonRuntime *runtime)
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
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
        configuration.flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT;
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
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

static SparkStatus SparkGlm52Pp13DaemonOpenWorkControlPath(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u)
        return SPARK_STATUS_OK;
    runtime->work_listen_fd =
        SparkGlm52Pp13DaemonCreateListenSocket(
            "0.0.0.0",
            runtime->rank_plan.listen_port);
    if (runtime->work_listen_fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    if (SparkGlm52Pp13DaemonSetNonblocking(runtime->work_listen_fd) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52Pp13DaemonAcceptWorkSocket(
    SparkGlm52Pp13DaemonRuntime *runtime)
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
    if (SparkGlm52Pp13DaemonSetNonblocking(fd) < 0)
    {
        close(fd);
        runtime->work_error_count += 1u;
        return 0u;
    }
    SparkGlm52Pp13DaemonConfigureTcpSocket(fd);
    runtime->work_input_socket_fd = fd;
    return 1u;
}

static SparkStatus SparkGlm52Pp13DaemonEnsureWorkOutputSocket(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkStatus status;
    uint64_t now_ns;

    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
        return SPARK_STATUS_OK;
    if (runtime->work_output_socket_fd >= 0)
    {
        status = SparkGlm52Pp13DaemonFinishConnect(
            &runtime->work_output_socket_fd,
            &runtime->work_output_connecting);
        if (status == SPARK_STATUS_OK)
            runtime->work_output_retry_mono_ns = 0u;
        else if (runtime->work_output_socket_fd < 0)
            runtime->work_output_retry_mono_ns =
                SparkGlm52Pp13DaemonMonotonicNs() +
                SPARK_GLM52_PP13_DAEMON_CONNECT_RETRY_NS;
        return status;
    }
    now_ns = SparkGlm52Pp13DaemonMonotonicNs();
    if (runtime->work_output_retry_mono_ns != 0u &&
        now_ns < runtime->work_output_retry_mono_ns)
        return SPARK_STATUS_BUSY;
    runtime->work_output_socket_fd =
        SparkGlm52Pp13DaemonStartConnect(
            runtime->rank_plan.next_host_name,
            runtime->rank_plan.next_port,
            &runtime->work_output_connecting);
    if (runtime->work_output_socket_fd < 0)
    {
        runtime->work_output_retry_mono_ns =
            now_ns + SPARK_GLM52_PP13_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->work_output_retry_mono_ns = 0u;
    return runtime->work_output_connecting == 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static SparkStatus SparkGlm52Pp13DaemonForwardWork(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13WorkControlPacket *packet)
{
    SparkStatus status;

    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
        return SPARK_STATUS_OK;
    status = SparkGlm52Pp13DaemonEnsureWorkOutputSocket(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13DaemonWriteBuffered(
            runtime->work_output_socket_fd,
            packet,
            sizeof(*packet),
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

static SparkStatus SparkGlm52Pp13DaemonSubmitWork(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13WorkControlPacket *packet)
{
    SparkGlm52CudaResidentIpcSubmitWork submit_message;
    SparkGlm52CudaResidentIpcHeader header;
    SparkGlm52CudaResidentIpcSubmitResult submit_result;
    SparkStatus status;

    if (runtime == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (runtime->cuda_resident_fd >= 0)
    {
        memset(&submit_message,0,sizeof(submit_message));
        submit_message.descriptor_bytes =
            SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES;
        submit_message.work_packet = *packet;
        status = SparkGlm52Pp13DaemonWriteResidentMessage(
            runtime,
            SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK,
            &submit_message,
            sizeof(submit_message));
        if (status != SPARK_STATUS_OK)
            return status;
        runtime->cuda_resident_submit_count += 1u;
        for (;;)
        {
            status = SparkGlm52Pp13DaemonReadFull(
                runtime->cuda_resident_fd,
                &header,
                sizeof(header));
            if (status != SPARK_STATUS_OK)
                return status;
            status = SparkGlm52CudaResidentIpcValidateHeader(
                &header,
                0u,
                sizeof(submit_result));
            if (status != SPARK_STATUS_OK)
                return status;
            if (header.kind ==
                SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT)
            {
                if (header.payload_bytes != sizeof(submit_result))
                    return SPARK_STATUS_ABI_MISMATCH;
                status = SparkGlm52Pp13DaemonReadFull(
                    runtime->cuda_resident_fd,
                    &submit_result,
                    sizeof(submit_result));
                if (status != SPARK_STATUS_OK)
                    return status;
                return (SparkStatus)submit_result.status;
            }
            if (header.kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_COMPLETION)
            {
                SparkGlm52CudaResidentIpcCompletion completion_message;
                if (header.payload_bytes != sizeof(completion_message))
                    return SPARK_STATUS_ABI_MISMATCH;
                status = SparkGlm52Pp13DaemonReadFull(
                    runtime->cuda_resident_fd,
                    &completion_message,
                    sizeof(completion_message));
                if (status != SPARK_STATUS_OK)
                    return status;
                SparkGlm52Pp13DaemonCompletion(
                    runtime,
                    &completion_message.completion);
                continue;
            }
            return SPARK_STATUS_SCHEMA_ERROR;
        }
    }
    if (runtime->builder_state == 0 ||
        runtime->builder_library.builder_interface.submit_work == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return runtime->builder_library.builder_interface.submit_work(
        runtime->builder_state,
        packet,
        runtime->input_transport_session,
        runtime->output_transport_session,
        SparkGlm52Pp13DaemonCompletion,
        runtime);
}

static uint32_t SparkGlm52Pp13DaemonWorkPacketHash(
    const SparkGlm52Pp13WorkControlPacket *packet)
{
    uint64_t hash;

    if (packet == 0)
        return 0u;
    hash = packet->request_id;
    hash ^= (packet->sequence_id * 0x9e3779b97f4a7c15ull);
    hash ^= (packet->sequence_position * 0xbf58476d1ce4e5b9ull);
    hash ^= ((uint64_t)packet->pipeline_slot << 32);
    hash ^= ((uint64_t)packet->active_sequence_count << 48);
    return (uint32_t)(hash ^ (hash >> 32u));
}

static uint32_t SparkGlm52Pp13DaemonWorkPacketMatches(
    const SparkGlm52Pp13WorkControlPacket *left,
    const SparkGlm52Pp13WorkControlPacket *right)
{
    if (left == 0 || right == 0)
        return 0u;
    return left->request_id == right->request_id &&
        left->sequence_id == right->sequence_id &&
        left->sequence_position == right->sequence_position &&
        left->pipeline_slot == right->pipeline_slot &&
        left->active_sequence_count == right->active_sequence_count &&
        left->new_token_count == right->new_token_count;
}

static void SparkGlm52Pp13DaemonInitializeWorkQueue(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    uint32_t slot_index;

    if (runtime == 0)
        return;
    for (slot_index = 0u;
         slot_index < SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_HASH_SLOTS;
         ++slot_index)
        runtime->work_queue_hash_heads[slot_index] =
            SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT;
    for (slot_index = 0u;
         slot_index < SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY;
         ++slot_index)
    {
        runtime->work_queue_hash_next[slot_index] =
            SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT;
        runtime->work_queue_submitted[slot_index] = 0u;
        runtime->work_queue_forwarded[slot_index] = 0u;
        runtime->work_queue_state[slot_index] =
            SPARK_GLM52_PP13_DAEMON_WORK_STATE_READY;
    }
}

static uint32_t SparkGlm52Pp13DaemonFindQueuedWorkSlot(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13WorkControlPacket *packet)
{
    uint32_t hash_slot;
    uint32_t slot_index;

    if (runtime == 0 || packet == 0)
        return SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT;
    hash_slot = SparkGlm52Pp13DaemonWorkPacketHash(packet) %
        SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_HASH_SLOTS;
    slot_index = runtime->work_queue_hash_heads[hash_slot];
    while (slot_index != SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (slot_index >= SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY)
            return SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT;
        if (SparkGlm52Pp13DaemonWorkPacketMatches(
                &runtime->work_queue[slot_index],
                packet) != 0u)
            return slot_index;
        slot_index = runtime->work_queue_hash_next[slot_index];
    }
    return SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT;
}

static void SparkGlm52Pp13DaemonInsertQueuedWorkHash(
    SparkGlm52Pp13DaemonRuntime *runtime,
    uint32_t queue_slot)
{
    uint32_t hash_slot;

    if (runtime == 0 ||
        queue_slot >= SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY)
        return;
    hash_slot =
        SparkGlm52Pp13DaemonWorkPacketHash(&runtime->work_queue[queue_slot]) %
        SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_HASH_SLOTS;
    runtime->work_queue_hash_next[queue_slot] =
        runtime->work_queue_hash_heads[hash_slot];
    runtime->work_queue_hash_heads[hash_slot] = queue_slot;
}

static void SparkGlm52Pp13DaemonRemoveQueuedWorkHash(
    SparkGlm52Pp13DaemonRuntime *runtime,
    uint32_t queue_slot)
{
    uint32_t hash_slot;
    uint32_t slot_index;
    uint32_t previous_slot;

    if (runtime == 0 ||
        queue_slot >= SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY)
        return;
    hash_slot =
        SparkGlm52Pp13DaemonWorkPacketHash(&runtime->work_queue[queue_slot]) %
        SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_HASH_SLOTS;
    slot_index = runtime->work_queue_hash_heads[hash_slot];
    previous_slot = SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT;
    while (slot_index != SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (slot_index == queue_slot)
        {
            if (previous_slot == SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT)
                runtime->work_queue_hash_heads[hash_slot] =
                    runtime->work_queue_hash_next[slot_index];
            else
                runtime->work_queue_hash_next[previous_slot] =
                    runtime->work_queue_hash_next[slot_index];
            runtime->work_queue_hash_next[slot_index] =
                SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT;
            return;
        }
        if (slot_index >= SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY)
            return;
        previous_slot = slot_index;
        slot_index = runtime->work_queue_hash_next[slot_index];
    }
}

static SparkStatus SparkGlm52Pp13DaemonQueueWork(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13WorkControlPacket *packet)
{
    uint32_t tail;
    uint32_t existing_slot;

    if (runtime == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    runtime->work_receive_count += 1u;
    existing_slot = SparkGlm52Pp13DaemonFindQueuedWorkSlot(runtime,packet);
    if (existing_slot != SPARK_GLM52_PP13_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (runtime->work_queue_submitted[existing_slot] == 0u)
        {
            runtime->work_queue[existing_slot] = *packet;
            runtime->work_queue_forwarded[existing_slot] = 0u;
            runtime->work_queue_state[existing_slot] =
                SPARK_GLM52_PP13_DAEMON_WORK_STATE_READY;
        }
        runtime->work_duplicate_count += 1u;
        return SPARK_STATUS_OK;
    }
    if (runtime->work_queue_count >= SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    tail = (runtime->work_queue_head + runtime->work_queue_count) %
        SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY;
    runtime->work_queue[tail] = *packet;
    runtime->work_queue_submitted[tail] = 0u;
    runtime->work_queue_forwarded[tail] = 0u;
    runtime->work_queue_state[tail] =
        SPARK_GLM52_PP13_DAEMON_WORK_STATE_READY;
    SparkGlm52Pp13DaemonInsertQueuedWorkHash(runtime,tail);
    runtime->work_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13DaemonPopWork(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->work_queue_count == 0u)
        return;
    SparkGlm52Pp13DaemonRemoveQueuedWorkHash(runtime,runtime->work_queue_head);
    runtime->work_queue_submitted[runtime->work_queue_head] = 0u;
    runtime->work_queue_forwarded[runtime->work_queue_head] = 0u;
    runtime->work_queue_state[runtime->work_queue_head] =
        SPARK_GLM52_PP13_DAEMON_WORK_STATE_READY;
    runtime->work_queue_head = (runtime->work_queue_head + 1u) %
        SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY;
    runtime->work_queue_count -= 1u;
}

static void SparkGlm52Pp13DaemonDeferWork(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkGlm52Pp13WorkControlPacket packet;
    uint32_t old_head;
    uint32_t tail;

    if (runtime == 0 || runtime->work_queue_count <= 1u)
        return;
    old_head = runtime->work_queue_head;
    if (runtime->work_queue_count < SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY)
    {
        tail = (runtime->work_queue_head + runtime->work_queue_count) %
            SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY;
        packet = runtime->work_queue[old_head];
        SparkGlm52Pp13DaemonRemoveQueuedWorkHash(runtime,old_head);
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
            SPARK_GLM52_PP13_DAEMON_WORK_STATE_READY;
        SparkGlm52Pp13DaemonInsertQueuedWorkHash(runtime,tail);
    }
    runtime->work_queue_head = (runtime->work_queue_head + 1u) %
        SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY;
}

static void SparkGlm52Pp13DaemonWakeDeferredWork(
    SparkGlm52Pp13DaemonRuntime *runtime)
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
        if (runtime->work_queue_state[slot_index] ==
            SPARK_GLM52_PP13_DAEMON_WORK_STATE_WAITING)
        {
            runtime->work_queue_state[slot_index] =
                SPARK_GLM52_PP13_DAEMON_WORK_STATE_READY;
            runtime->work_wake_count += 1u;
        }
        slot_index = (slot_index + 1u) %
            SPARK_GLM52_PP13_DAEMON_WORK_QUEUE_CAPACITY;
    }
}

static uint32_t SparkGlm52Pp13DaemonPumpQueuedWork(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkGlm52Pp13WorkControlPacket *packet;
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
            SPARK_GLM52_PP13_DAEMON_WORK_STATE_WAITING)
        {
            SparkGlm52Pp13DaemonDeferWork(runtime);
            continue;
        }
        forward_done =
            ((runtime->rank_plan.flags &
              SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u ||
             runtime->work_queue_forwarded[queue_slot] != 0u)
                ? 1u : 0u;
        if (forward_done == 0u)
        {
            status = SparkGlm52Pp13DaemonForwardWork(runtime,packet);
            if (status == SPARK_STATUS_OK)
            {
                runtime->work_queue_forwarded[queue_slot] = 1u;
                forward_done = 1u;
            }
            else if (status == SPARK_STATUS_BUSY ||
                     status == SPARK_STATUS_ROUTE_NOT_FOUND)
            {
                runtime->work_queue_state[queue_slot] =
                    SPARK_GLM52_PP13_DAEMON_WORK_STATE_WAITING;
                runtime->work_deferred_count += 1u;
                SparkGlm52Pp13DaemonDeferWork(runtime);
                continue;
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
                SparkGlm52Pp13DaemonPopWork(runtime);
                return 1u;
            }
        }
        if (runtime->work_queue_submitted[queue_slot] == 0u)
        {
            status = SparkGlm52Pp13DaemonSubmitWork(runtime,packet);
            if (status == SPARK_STATUS_OK)
            {
                runtime->work_submit_count += 1u;
                runtime->work_queue_submitted[queue_slot] = 1u;
                runtime->driver_inflight_count += 1u;
            }
            else if (status == SPARK_STATUS_BUSY)
            {
                runtime->work_queue_state[queue_slot] =
                    SPARK_GLM52_PP13_DAEMON_WORK_STATE_WAITING;
                runtime->work_deferred_count += 1u;
                SparkGlm52Pp13DaemonDeferWork(runtime);
                continue;
            }
            else
            {
                runtime->work_error_count += 1u;
                fprintf(stderr,
                    "rank_work_submit_failed status=%u request=%llu sequence=%llu position=%llu queued=%u\n",
                    (uint32_t)status,
                    (unsigned long long)packet->request_id,
                    (unsigned long long)packet->sequence_id,
                    (unsigned long long)packet->sequence_position,
                    runtime->work_queue_count);
                SparkGlm52Pp13DaemonPopWork(runtime);
                return 1u;
            }
        }
        if (forward_done != 0u &&
            runtime->work_queue_submitted[queue_slot] != 0u)
        {
            SparkGlm52Pp13DaemonPopWork(runtime);
            return 1u;
        }
    }
    return 0u;
}

static void SparkGlm52Pp13DaemonHandleWork(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13WorkControlPacket *packet)
{
    SparkStatus status;

    status = SparkGlm52Pp13WorkControlValidatePacket(
        packet,
        runtime->rank_plan.max_active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52Pp13DaemonQueueWork(runtime,packet);
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

static uint32_t SparkGlm52Pp13DaemonPumpWorkControl(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkGlm52Pp13WorkControlPacket *packet;
    ssize_t got;
    uint32_t remaining;
    uint32_t progress;

    progress = SparkGlm52Pp13DaemonAcceptWorkSocket(runtime);
    if (runtime->work_listen_fd < 0 || runtime->work_input_socket_fd < 0)
        return progress;
    for (;;)
    {
        remaining = ((uint32_t)sizeof(runtime->work_read_buffer) -
            runtime->work_read_offset);
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
        if (runtime->work_read_offset < sizeof(*packet))
            return progress;
        packet = (SparkGlm52Pp13WorkControlPacket *)runtime->work_read_buffer;
        SparkGlm52Pp13DaemonHandleWork(runtime,packet);
        runtime->work_read_offset = 0u;
    }
}

static SparkStatus SparkGlm52Pp13DaemonOpenFinalEventPath(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration)
{
    if (runtime->rank_plan.rank_index == 0u &&
        configuration->own_final_event != 0u)
    {
        runtime->final_event_listen_fd =
            SparkGlm52Pp13DaemonCreateListenSocket(
                configuration->final_event_bind_address,
                runtime->final_event_route.listen_port);
        if (runtime->final_event_listen_fd < 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        if (SparkGlm52Pp13DaemonSetNonblocking(
                runtime->final_event_listen_fd) < 0)
            return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u)
        (void)configuration;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13DaemonEnsureFinalEventSocket(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkStatus status;
    uint64_t now_ns;

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u)
        return SPARK_STATUS_OK;
    if (runtime->final_event_socket_fd >= 0)
    {
        status = SparkGlm52Pp13DaemonFinishConnect(
            &runtime->final_event_socket_fd,
            &runtime->final_event_socket_connecting);
        if (status == SPARK_STATUS_OK)
            runtime->final_event_retry_mono_ns = 0u;
        else if (runtime->final_event_socket_fd < 0)
            runtime->final_event_retry_mono_ns =
                SparkGlm52Pp13DaemonMonotonicNs() +
                SPARK_GLM52_PP13_DAEMON_CONNECT_RETRY_NS;
        return status;
    }
    now_ns = SparkGlm52Pp13DaemonMonotonicNs();
    if (runtime->final_event_retry_mono_ns != 0u &&
        now_ns < runtime->final_event_retry_mono_ns)
        return SPARK_STATUS_BUSY;
    runtime->final_event_socket_fd =
        SparkGlm52Pp13DaemonStartConnect(
            runtime->final_event_route.sink_host_name,
            runtime->final_event_route.connect_port,
            &runtime->final_event_socket_connecting);
    if (runtime->final_event_socket_fd < 0)
    {
        runtime->final_event_retry_mono_ns =
            now_ns + SPARK_GLM52_PP13_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_BUSY;
    }
    runtime->final_event_retry_mono_ns = 0u;
    return runtime->final_event_socket_connecting == 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static uint32_t SparkGlm52Pp13DaemonPumpFinalEventSend(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    const SparkGlm52Pp13DaemonFinalEvent *event;
    SparkStatus status;

    if (runtime == 0 ||
        (runtime->rank_plan.flags &
            SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u ||
        runtime->final_event_queue_count == 0u)
        return 0u;
    status = SparkGlm52Pp13DaemonEnsureFinalEventSocket(runtime);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
        return 0u;
    event = &runtime->final_event_queue[runtime->final_event_queue_head];
    status = SparkGlm52Pp13DaemonWriteBuffered(
            runtime->final_event_socket_fd,
            event,
            sizeof(*event),
            &runtime->final_event_write_offset);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        close(runtime->final_event_socket_fd);
        runtime->final_event_socket_fd = -1;
        runtime->final_event_socket_connecting = 0u;
        runtime->final_event_write_offset = 0u;
        runtime->final_event_send_error_count += 1u;
        return 0u;
    }
    SparkGlm52Pp13DaemonPopFinalEvent(runtime);
    runtime->final_event_send_count += 1u;
    return 1u;
}

static uint32_t SparkGlm52Pp13DaemonAcceptFinalEventSocket(
    SparkGlm52Pp13DaemonRuntime *runtime)
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
    if (SparkGlm52Pp13DaemonSetNonblocking(fd) < 0)
    {
        close(fd);
        runtime->final_event_receive_error_count += 1u;
        return 0u;
    }
    SparkGlm52Pp13DaemonConfigureTcpSocket(fd);
    runtime->final_event_socket_fd = fd;
    return 1u;
}

static void SparkGlm52Pp13DaemonPublishFinalEvent(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonFinalEvent *event)
{
    uint32_t token_index;

    if (event->magic != SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_MAGIC ||
        event->descriptor_bytes != (uint32_t)sizeof(*event))
    {
        runtime->final_event_receive_error_count += 1u;
        return;
    }
    runtime->final_event_receive_count += 1u;
    printf("glm52_pp13_final_event=1 request=%llu sequence=%llu position=%llu status=%u accepted=%u token_count=%u service_ns=%llu",
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

static uint32_t SparkGlm52Pp13DaemonPumpFinalEvents(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkGlm52Pp13DaemonFinalEvent *event;
    ssize_t got;
    uint32_t remaining;
    uint32_t progress;

    progress = SparkGlm52Pp13DaemonPumpFinalEventSend(runtime);
    progress |= SparkGlm52Pp13DaemonAcceptFinalEventSocket(runtime);
    if (runtime->final_event_listen_fd < 0 ||
        runtime->final_event_socket_fd < 0)
        return progress;
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
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                runtime->final_event_receive_error_count += 1u;
            return progress;
        }
        if (got == 0)
        {
            close(runtime->final_event_socket_fd);
            runtime->final_event_socket_fd = -1;
            runtime->final_event_read_offset = 0u;
            return 1u;
        }
        progress = 1u;
        runtime->final_event_read_offset += (uint32_t)got;
        if (runtime->final_event_read_offset < sizeof(*event))
            return progress;
        event =
            (SparkGlm52Pp13DaemonFinalEvent *)runtime->final_event_read_buffer;
        SparkGlm52Pp13DaemonPublishFinalEvent(runtime,event);
        runtime->final_event_read_offset = 0u;
    }
}

static SparkStatus SparkGlm52Pp13DaemonInitialize(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;

    memset(runtime,0,sizeof(*runtime));
    runtime->cuda_resident_fd = -1;
    runtime->work_listen_fd = -1;
    runtime->work_input_socket_fd = -1;
    runtime->work_output_socket_fd = -1;
    runtime->final_event_listen_fd = -1;
    runtime->final_event_socket_fd = -1;
    runtime->wake_pipe_read_fd = -1;
    runtime->wake_pipe_write_fd = -1;
    SparkGlm52Pp13DaemonInitializeWorkQueue(runtime);
    SparkLoadedModelDriverReset(&runtime->loaded_driver);
    status = SparkGlm52Pp13DaemonOpenWakePipe(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open rank daemon wake pipe");
        return status;
    }
    status = SparkGlm52Pp13RuntimeBuildRankPlan(
        configuration->rank_index,
        configuration->max_active_sequence_count,
        configuration->port_base,
        &runtime->rank_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build PP13 rank plan");
        return status;
    }
    status = SparkGlm52Pp13RuntimeBuildFinalEventRoute(
        configuration->port_base,
        &runtime->final_event_route,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build PP13 final event route");
        return status;
    }
    if (configuration->cuda_resident_socket_path != 0)
    {
        status = SparkGlm52Pp13DaemonConnectCudaResident(runtime,configuration);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52Pp13DaemonSetDefaultError(
                error_buffer,
                error_buffer_bytes,
                "failed to connect CUDA resident daemon");
            return status;
        }
        status = SparkGlm52Pp13DaemonOpenWorkControlPath(runtime);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52Pp13DaemonSetDefaultError(
                error_buffer,
                error_buffer_bytes,
                "failed to open production work-control path");
            return status;
        }
        status = SparkGlm52Pp13DaemonOpenFinalEventPath(runtime,configuration);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52Pp13DaemonSetDefaultError(
                error_buffer,
                error_buffer_bytes,
                "failed to open final event route");
            return status;
        }
        return SPARK_STATUS_OK;
    }
    status = SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
        &runtime->rank_plan,
        configuration->fp8_pack_root,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to validate rank FP8 pack files");
        return status;
    }
    status = SparkGlm52Pp13DaemonBuildNodeContext(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build resident node context");
        return status;
    }
    status = SparkGlm52Pp13DaemonLoadTransport(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to load production hidden transport");
        return status;
    }
    status = SparkGlm52Pp13DaemonOpenHiddenTransport(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open neighbor hidden transport session");
        return status;
    }
    status = SparkGlm52Pp13DaemonLoadDriver(
        runtime,
        configuration,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to load GLM52 model driver");
        return status;
    }
    status = SparkGlm52Pp13DaemonAttachBuilderDriver(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to attach node-context builder to driver");
        return status;
    }
    status = SparkGlm52Pp13DaemonInitializeRunner(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to initialize production stage runner");
        return status;
    }
    status = SparkGlm52Pp13DaemonOpenWorkControlPath(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open production work-control path");
        return status;
    }
    status = SparkGlm52Pp13DaemonOpenFinalEventPath(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52Pp13DaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open final event route");
        return status;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13DaemonDestroy(
    SparkGlm52Pp13DaemonRuntime *runtime)
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
    SparkGlm52Pp13NodeContextBuilderUnloadInterface(&runtime->builder_library);
    SparkHiddenTransportClose(runtime->input_transport_session);
    SparkHiddenTransportClose(runtime->output_transport_session);
    SparkHiddenTransportUnloadInterface(&runtime->transport_library);
}

static void SparkGlm52Pp13DaemonPrintReady(
    const SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration)
{
    printf("glm52_pp13_rank_daemon=1\n");
    printf("rank=%u\n",runtime->rank_plan.rank_index);
    printf("host=%s\n",runtime->rank_plan.host_name);
    printf("stage=%u:%u\n",
        runtime->rank_plan.first_layer_index,
        runtime->rank_plan.layer_count);
    printf("cuda_resident_socket=%s\n",
        configuration->cuda_resident_socket_path != 0 ?
            configuration->cuda_resident_socket_path : "");
    printf("fp8_pack_root=%s\n",
        configuration->fp8_pack_root != 0 ? configuration->fp8_pack_root : "");
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
    SparkGlm52Pp13DaemonConfig configuration;
    SparkGlm52Pp13DaemonRuntime runtime;
    char error_buffer[512];
    SparkStatus status;
    uint32_t event_mask;
    uint32_t progress;
    uint64_t timeout_ns;
    uint64_t next_timer_ns;
    uint64_t now_ns;

    SparkGlm52Pp13DaemonInitializeConfig(&configuration);
    if (SparkGlm52Pp13DaemonParseArguments(&configuration,argc,argv) < 0)
    {
        fprintf(stderr,
            "usage: %s --rank n [--cuda-resident-socket path | --fp8-pack-root dir --stagepack-root dir --transport-so path --driver-so path --node-context-builder-so path --embedding-pack path] [--program name] [--node-target target] [--max-active n] [--port-base n] [--final-event-bind ip] [--final-event-return-host host]\n",
            argv[0]);
        return 2;
    }
    signal(SIGINT,SparkGlm52Pp13DaemonSignal);
    signal(SIGTERM,SparkGlm52Pp13DaemonSignal);
    error_buffer[0] = '\0';
    status = SparkGlm52Pp13DaemonInitialize(
        &runtime,
        &configuration,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"rank daemon init failed status=%u error=%s\n",
            (uint32_t)status,
            error_buffer);
        SparkGlm52Pp13DaemonDestroy(&runtime);
        return 3;
    }
    SparkGlm52Pp13DaemonPrintReady(&runtime,&configuration);
    while (SparkGlm52Pp13DaemonRunning != 0)
    {
        progress = 0u;
        if (SparkGlm52Pp13DaemonDrainWakePipe(&runtime) != 0u)
        {
            progress = 1u;
            SparkGlm52Pp13DaemonWakeDeferredWork(&runtime);
        }
        progress |= SparkGlm52Pp13DaemonPumpWorkControl(&runtime);
        progress |= SparkGlm52Pp13DaemonPumpCudaResident(&runtime);
        if (runtime.cuda_resident_fd < 0)
            (void)SparkGlm52ResidentDecodeStageProductionRunnerProgress(
                &runtime.runner);
        progress |= SparkGlm52Pp13DaemonPumpQueuedWork(&runtime);
        progress |= SparkGlm52Pp13DaemonPumpFinalEvents(&runtime);
        if (progress == 0u)
        {
            timeout_ns = 0u;
            next_timer_ns = SparkGlm52Pp13DaemonNextTimerNs(&runtime);
            if (next_timer_ns != 0u)
            {
                now_ns = SparkGlm52Pp13DaemonMonotonicNs();
                timeout_ns = next_timer_ns > now_ns ?
                    next_timer_ns - now_ns : 1u;
            }
            event_mask = SparkGlm52Pp13DaemonWaitForEvents(
                &runtime,
                timeout_ns);
            if ((event_mask &
                (SPARK_GLM52_PP13_DAEMON_POLL_KIND_INPUT_TRANSPORT |
                 SPARK_GLM52_PP13_DAEMON_POLL_KIND_OUTPUT_TRANSPORT |
                 SPARK_GLM52_PP13_DAEMON_POLL_KIND_WORK |
                 SPARK_GLM52_PP13_DAEMON_POLL_KIND_WORK_OUTPUT |
                 SPARK_GLM52_PP13_DAEMON_POLL_KIND_CUDA_RESIDENT |
                 SPARK_GLM52_PP13_DAEMON_POLL_KIND_FINAL_EVENT |
                 SPARK_GLM52_PP13_DAEMON_POLL_KIND_COMPLETION_WAKE |
                 SPARK_GLM52_PP13_DAEMON_POLL_KIND_TIMER)) != 0u)
            {
                if ((event_mask & SPARK_GLM52_PP13_DAEMON_POLL_KIND_TIMER) != 0u)
                    runtime.timer_wake_count += 1u;
                SparkGlm52Pp13DaemonWakeDeferredWork(&runtime);
            }
        }
    }
    SparkGlm52Pp13DaemonDestroy(&runtime);
    return 0;
}
