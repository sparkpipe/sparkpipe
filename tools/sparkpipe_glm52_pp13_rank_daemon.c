#define _POSIX_C_SOURCE 200112L

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"

#define SPARK_GLM52_PP13_DAEMON_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_GLM52_PP13_DAEMON_DEFAULT_PROGRAM "glm52.pp13.rank.production"
#define SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_MAGIC 0x35454650u

typedef struct SparkGlm52Pp13DaemonConfig
{
    const char *fp8_pack_root;
    const char *transport_shared_object_path;
    const char *driver_path;
    const char *program_name;
    const char *node_target;
    const char *final_event_bind_address;
    const char *final_event_return_host;
    uint32_t rank_index;
    uint32_t rank_is_set;
    uint32_t max_active_sequence_count;
    uint32_t port_base;
} SparkGlm52Pp13DaemonConfig;

typedef struct SparkGlm52Pp13DaemonRuntime
{
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    SparkGlm52Pp13RuntimeFinalEventRoute final_event_route;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *input_transport_session;
    SparkHiddenTransportSession *output_transport_session;
    SparkLoadedModelDriver loaded_driver;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    int32_t final_event_listen_fd;
    int32_t final_event_socket_fd;
    uint8_t final_event_read_buffer[64];
    uint32_t final_event_read_offset;
    uint64_t final_event_send_count;
    uint64_t final_event_receive_count;
    uint64_t final_event_send_error_count;
    uint64_t final_event_receive_error_count;
} SparkGlm52Pp13DaemonRuntime;

static volatile sig_atomic_t SparkGlm52Pp13DaemonRunning = 1;

typedef struct SparkGlm52Pp13DaemonFinalEvent
{
    uint32_t magic;
    uint32_t descriptor_bytes;
    uint32_t status;
    uint32_t program_id;
    uint32_t driver_dispatch_slot;
    uint32_t accepted_token_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t service_time_ns;
} SparkGlm52Pp13DaemonFinalEvent;

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
    if (strcmp(argv[*index],"--transport-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -3;
        configuration->transport_shared_object_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--driver-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -4;
        configuration->driver_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--program") == 0)
    {
        if ((*index + 1) >= argc)
            return -5;
        configuration->program_name = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--node-target") == 0)
    {
        if ((*index + 1) >= argc)
            return -6;
        configuration->node_target = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--max-active") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52Pp13DaemonParseU32(argv[*index + 1],&parsed) < 0)
            return -7;
        configuration->max_active_sequence_count = parsed;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--port-base") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52Pp13DaemonParseU32(argv[*index + 1],&parsed) < 0)
            return -8;
        configuration->port_base = parsed;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--final-event-bind") == 0)
    {
        if ((*index + 1) >= argc)
            return -9;
        configuration->final_event_bind_address = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--final-event-return-host") == 0)
    {
        if ((*index + 1) >= argc)
            return -10;
        configuration->final_event_return_host = argv[*index + 1];
        *index += 1;
        return 0;
    }
    return -11;
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
    if (configuration->rank_is_set == 0u ||
        configuration->fp8_pack_root == 0 ||
        configuration->transport_shared_object_path == 0 ||
        configuration->driver_path == 0)
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

static int32_t SparkGlm52Pp13DaemonConnectSocket(
    const char *host,
    uint32_t port)
{
    struct addrinfo hints;
    struct addrinfo *results;
    struct addrinfo *entry;
    char service[16];
    int32_t fd;

    if (host == 0)
        return -1;
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
        if (connect(fd,entry->ai_addr,entry->ai_addrlen) == 0)
            break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    return fd;
}

static int32_t SparkGlm52Pp13DaemonWriteAll(
    int32_t fd,
    const void *buffer,
    uint32_t buffer_bytes)
{
    const uint8_t *cursor;
    uint32_t offset;
    ssize_t written;

    if (fd < 0 || buffer == 0)
        return -1;
    cursor = (const uint8_t *)buffer;
    offset = 0u;
    while (offset < buffer_bytes)
    {
        written = write(fd,cursor + offset,buffer_bytes - offset);
        if (written < 0)
        {
            if (errno == EINTR)
                continue;
            return -2;
        }
        if (written == 0)
            return -3;
        offset += (uint32_t)written;
    }
    return 0;
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
    if ((runtime->rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u)
        return;
    if (runtime->final_event_socket_fd < 0)
    {
        runtime->final_event_send_error_count += 1u;
        return;
    }
    memset(&event,0,sizeof(event));
    event.magic = SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_MAGIC;
    event.descriptor_bytes = (uint32_t)sizeof(event);
    event.status = (uint32_t)completion->status;
    event.program_id = completion->program_id;
    event.driver_dispatch_slot = completion->driver_dispatch_slot;
    event.accepted_token_count = completion->accepted_token_count;
    event.request_id = completion->request_id;
    event.sequence_id = completion->sequence_id;
    event.sequence_position = completion->sequence_position;
    event.service_time_ns = completion->service_time_ns;
    if (SparkGlm52Pp13DaemonWriteAll(
            runtime->final_event_socket_fd,
            &event,
            sizeof(event)) < 0)
    {
        runtime->final_event_send_error_count += 1u;
        return;
    }
    runtime->final_event_send_count += 1u;
}

static SparkStatus SparkGlm52Pp13DaemonLoadTransport(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration)
{
    return SparkHiddenTransportLoadInterfaceFromSharedObject(
        configuration->transport_shared_object_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
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
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
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
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
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
        return status;
    runtime->program = SparkFindLoadedModelDriverProgram(
        &runtime->loaded_driver,
        configuration->program_name);
    if (runtime->program == 0)
        return SPARK_STATUS_NOT_FOUND;
    memset(&create_request,0,sizeof(create_request));
    create_request.node_id = runtime->rank_plan.host_name;
    create_request.node_target = configuration->node_target;
    create_request.node_context = 0;
    create_request.completion_function = SparkGlm52Pp13DaemonCompletion;
    create_request.completion_context = runtime;
    status = runtime->loaded_driver.interface->create(
        &create_request,
        &runtime->driver_instance);
    if (status != SPARK_STATUS_OK)
        return status;
    if (runtime->driver_instance == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
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

static SparkStatus SparkGlm52Pp13DaemonOpenFinalEventPath(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonConfig *configuration)
{
    if (runtime->rank_plan.rank_index == 0u)
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
    {
        runtime->final_event_socket_fd =
            SparkGlm52Pp13DaemonConnectSocket(
                configuration->final_event_return_host,
                runtime->final_event_route.connect_port);
        if (runtime->final_event_socket_fd < 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13DaemonAcceptFinalEventSocket(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    int32_t fd;

    if (runtime->final_event_listen_fd < 0 ||
        runtime->final_event_socket_fd >= 0)
        return;
    fd = accept(runtime->final_event_listen_fd,0,0);
    if (fd < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            runtime->final_event_receive_error_count += 1u;
        return;
    }
    if (SparkGlm52Pp13DaemonSetNonblocking(fd) < 0)
    {
        close(fd);
        runtime->final_event_receive_error_count += 1u;
        return;
    }
    runtime->final_event_socket_fd = fd;
}

static void SparkGlm52Pp13DaemonPublishFinalEvent(
    SparkGlm52Pp13DaemonRuntime *runtime,
    const SparkGlm52Pp13DaemonFinalEvent *event)
{
    if (event->magic != SPARK_GLM52_PP13_DAEMON_FINAL_EVENT_MAGIC ||
        event->descriptor_bytes != (uint32_t)sizeof(*event))
    {
        runtime->final_event_receive_error_count += 1u;
        return;
    }
    runtime->final_event_receive_count += 1u;
    printf("glm52_pp13_final_event=1 request=%llu sequence=%llu position=%llu status=%u accepted=%u service_ns=%llu\n",
        (unsigned long long)event->request_id,
        (unsigned long long)event->sequence_id,
        (unsigned long long)event->sequence_position,
        event->status,
        event->accepted_token_count,
        (unsigned long long)event->service_time_ns);
    fflush(stdout);
}

static void SparkGlm52Pp13DaemonPumpFinalEvents(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    SparkGlm52Pp13DaemonFinalEvent *event;
    ssize_t got;
    uint32_t remaining;

    SparkGlm52Pp13DaemonAcceptFinalEventSocket(runtime);
    if (runtime->final_event_listen_fd < 0 ||
        runtime->final_event_socket_fd < 0)
        return;
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
            return;
        }
        if (got == 0)
        {
            close(runtime->final_event_socket_fd);
            runtime->final_event_socket_fd = -1;
            runtime->final_event_read_offset = 0u;
            return;
        }
        runtime->final_event_read_offset += (uint32_t)got;
        if (runtime->final_event_read_offset < sizeof(*event))
            return;
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
    runtime->final_event_listen_fd = -1;
    runtime->final_event_socket_fd = -1;
    SparkLoadedModelDriverReset(&runtime->loaded_driver);
    status = SparkGlm52Pp13RuntimeBuildRankPlan(
        configuration->rank_index,
        configuration->max_active_sequence_count,
        configuration->port_base,
        &runtime->rank_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13RuntimeBuildFinalEventRoute(
        configuration->port_base,
        &runtime->final_event_route,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
        &runtime->rank_plan,
        configuration->fp8_pack_root,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13DaemonLoadTransport(runtime,configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13DaemonOpenHiddenTransport(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13DaemonLoadDriver(
        runtime,
        configuration,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13DaemonInitializeRunner(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkGlm52Pp13DaemonOpenFinalEventPath(runtime,configuration);
}

static void SparkGlm52Pp13DaemonDestroy(
    SparkGlm52Pp13DaemonRuntime *runtime)
{
    if (runtime == 0)
        return;
    if (runtime->final_event_socket_fd >= 0)
        close(runtime->final_event_socket_fd);
    if (runtime->final_event_listen_fd >= 0)
        close(runtime->final_event_listen_fd);
    if (runtime->loaded_driver.interface != 0 &&
        runtime->loaded_driver.interface->destroy != 0 &&
        runtime->driver_instance != 0)
        runtime->loaded_driver.interface->destroy(runtime->driver_instance);
    SparkUnloadModelDriver(&runtime->loaded_driver);
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
    printf("fp8_pack_root=%s\n",configuration->fp8_pack_root);
    printf("transport_so=%s\n",configuration->transport_shared_object_path);
    printf("driver_so=%s\n",configuration->driver_path);
    printf("input_transport=%u\n",
        runtime->input_transport_session != 0 ? 1u : 0u);
    printf("output_transport=%u\n",
        runtime->output_transport_session != 0 ? 1u : 0u);
    printf("final_event_listen=%d\n",runtime->final_event_listen_fd);
    printf("final_event_socket=%d\n",runtime->final_event_socket_fd);
    printf("final_event_sent=%llu\n",
        (unsigned long long)runtime->final_event_send_count);
    printf("final_event_received=%llu\n",
        (unsigned long long)runtime->final_event_receive_count);
    fflush(stdout);
}

int main(int argc,char **argv)
{
    SparkGlm52Pp13DaemonConfig configuration;
    SparkGlm52Pp13DaemonRuntime runtime;
    char error_buffer[512];
    SparkStatus status;
    struct timespec sleep_time;

    SparkGlm52Pp13DaemonInitializeConfig(&configuration);
    if (SparkGlm52Pp13DaemonParseArguments(&configuration,argc,argv) < 0)
    {
        fprintf(stderr,
            "usage: %s --rank n --fp8-pack-root dir --transport-so path --driver-so path [--program name] [--node-target target] [--max-active n] [--port-base n] [--final-event-bind ip] [--final-event-return-host host]\n",
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
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 100000000L;
    while (SparkGlm52Pp13DaemonRunning != 0)
    {
        SparkGlm52Pp13DaemonPumpFinalEvents(&runtime);
        nanosleep(&sleep_time,0);
    }
    SparkGlm52Pp13DaemonDestroy(&runtime);
    return 0;
}
