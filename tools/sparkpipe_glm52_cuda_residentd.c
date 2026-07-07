#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200112L

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"
#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_hidden_transport.h"

#define SPARK_GLM52_CUDA_RESIDENTD_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_GLM52_CUDA_RESIDENTD_DEFAULT_PROGRAM "glm52.pp13.rank.production"
#define SPARK_GLM52_CUDA_RESIDENTD_DEFAULT_SOCKET_PREFIX "/tmp/sparkpipe_glm52_cuda_resident_rank"
#define SPARK_GLM52_CUDA_RESIDENTD_POLL_TIMEOUT_MS 10
#define SPARK_GLM52_CUDA_RESIDENTD_CONTROL_PAYLOAD_CAPACITY 4096u

#define SPARK_GLM52_CUDA_RESIDENTD_POLL_LISTEN 0u
#define SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS 4u
#define SPARK_GLM52_CUDA_RESIDENTD_POLL_CAPACITY (SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS + 1u)

static volatile sig_atomic_t SparkGlm52CudaResidentdRunning = 1;

typedef struct SparkGlm52CudaResidentdConfiguration
{
    const char *socket_path;
    const char *fp8_pack_root;
    const char *stagepack_root;
    const char *transport_shared_object_path;
    const char *driver_path;
    const char *node_context_builder_shared_object_path;
    const char *embedding_pack_path;
    const char *program_name;
    const char *node_target;
    uint32_t rank_index;
    uint32_t rank_is_set;
    uint32_t max_active_sequence_count;
    uint32_t port_base;
    uint64_t cuda_generation;
    uint64_t control_generation;
} SparkGlm52CudaResidentdConfiguration;

typedef struct SparkGlm52CudaResidentdRuntime
{
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *input_transport_session;
    SparkHiddenTransportSession *output_transport_session;
    SparkGlm52Pp13NodeContextBuilderDynamicLibrary builder_library;
    void *builder_state;
    SparkGlm52Pp13NodeContextBuilderResult builder_result;
    SparkLoadedModelDriver loaded_driver;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    int32_t listen_fd;
    int32_t client_fds[SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS];
    int32_t active_client_fd;
    pthread_mutex_t completion_write_mutex;
    uint64_t next_ipc_sequence_number;
    uint64_t submitted_count;
    uint64_t submit_failed_count;
    uint64_t completion_count;
    uint64_t control_error_count;
    uint32_t state;
    char blocker[SPARK_GLM52_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES];
} SparkGlm52CudaResidentdRuntime;

static void SparkGlm52CudaResidentdSignal(int signal_number)
{
    (void)signal_number;
    SparkGlm52CudaResidentdRunning = 0;
}

static void SparkGlm52CudaResidentdInitializeConfiguration(
    SparkGlm52CudaResidentdConfiguration *configuration)
{
    memset(configuration, 0, sizeof(*configuration));
    configuration->program_name = SPARK_GLM52_CUDA_RESIDENTD_DEFAULT_PROGRAM;
    configuration->max_active_sequence_count =
        SPARK_GLM52_CUDA_RESIDENTD_DEFAULT_MAX_ACTIVE;
    configuration->port_base = SPARK_GLM52_PP13_RUNTIME_DEFAULT_PORT_BASE;
}

static int32_t SparkGlm52CudaResidentdParseU32(
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
        value = (value * 10u) + (uint32_t)(text[index] - '0');
        if (value > 0xffffffffull)
            return -3;
    }
    *value_out = (uint32_t)value;
    return 0;
}

static int32_t SparkGlm52CudaResidentdParseU64(
    const char *text,
    uint64_t *value_out)
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
        value = (value * 10u) + (uint32_t)(text[index] - '0');
    }
    *value_out = value;
    return 0;
}

static int32_t SparkGlm52CudaResidentdApplyArgument(
    SparkGlm52CudaResidentdConfiguration *configuration,
    int argc,
    char **argv,
    int32_t *index)
{
    uint32_t parsed_u32;
    uint64_t parsed_u64;

    if (strcmp(argv[*index], "--socket") == 0)
    {
        if ((*index + 1) >= argc)
            return -1;
        configuration->socket_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--rank") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52CudaResidentdParseU32(argv[*index + 1], &parsed_u32) < 0)
            return -2;
        configuration->rank_index = parsed_u32;
        configuration->rank_is_set = 1u;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--max-active") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52CudaResidentdParseU32(argv[*index + 1], &parsed_u32) < 0)
            return -3;
        configuration->max_active_sequence_count = parsed_u32;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--port-base") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52CudaResidentdParseU32(argv[*index + 1], &parsed_u32) < 0)
            return -4;
        configuration->port_base = parsed_u32;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--cuda-generation") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52CudaResidentdParseU64(argv[*index + 1], &parsed_u64) < 0)
            return -5;
        configuration->cuda_generation = parsed_u64;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--control-generation") == 0)
    {
        if ((*index + 1) >= argc ||
            SparkGlm52CudaResidentdParseU64(argv[*index + 1], &parsed_u64) < 0)
            return -6;
        configuration->control_generation = parsed_u64;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--fp8-pack-root") == 0)
    {
        if ((*index + 1) >= argc)
            return -7;
        configuration->fp8_pack_root = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--stagepack-root") == 0)
    {
        if ((*index + 1) >= argc)
            return -8;
        configuration->stagepack_root = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--transport-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -9;
        configuration->transport_shared_object_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--driver-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -10;
        configuration->driver_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--node-context-builder-so") == 0)
    {
        if ((*index + 1) >= argc)
            return -11;
        configuration->node_context_builder_shared_object_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--embedding-pack") == 0)
    {
        if ((*index + 1) >= argc)
            return -12;
        configuration->embedding_pack_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--program") == 0)
    {
        if ((*index + 1) >= argc)
            return -13;
        configuration->program_name = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index], "--node-target") == 0)
    {
        if ((*index + 1) >= argc)
            return -14;
        configuration->node_target = argv[*index + 1];
        *index += 1;
        return 0;
    }
    return -100;
}

static SparkStatus SparkGlm52CudaResidentdValidateConfiguration(
    SparkGlm52CudaResidentdConfiguration *configuration)
{
    static char default_socket_path[128];

    if (configuration->rank_is_set == 0u ||
        configuration->fp8_pack_root == 0 ||
        configuration->stagepack_root == 0 ||
        configuration->transport_shared_object_path == 0 ||
        configuration->driver_path == 0 ||
        configuration->node_context_builder_shared_object_path == 0 ||
        configuration->embedding_pack_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (configuration->socket_path == 0)
    {
        snprintf(default_socket_path, sizeof(default_socket_path), "%s%u.sock",
            SPARK_GLM52_CUDA_RESIDENTD_DEFAULT_SOCKET_PREFIX,
            configuration->rank_index);
        configuration->socket_path = default_socket_path;
    }
    return SPARK_STATUS_OK;
}

static int32_t SparkGlm52CudaResidentdSetNonblocking(int32_t fd)
{
    int flags;

    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
        return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static SparkStatus SparkGlm52CudaResidentdWriteFull(
    int32_t fd,
    const void *data,
    uint32_t bytes)
{
    const uint8_t *cursor;
    uint32_t offset;
    ssize_t written;

    if (fd < 0 || data == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (const uint8_t *)data;
    offset = 0u;
    while (offset < bytes)
    {
        written = write(fd, cursor + offset, bytes - offset);
        if (written < 0)
        {
            if (errno == EINTR)
                continue;
            return SPARK_STATUS_IO_ERROR;
        }
        if (written == 0)
            return SPARK_STATUS_IO_ERROR;
        offset += (uint32_t)written;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CudaResidentdReadFull(
    int32_t fd,
    void *data,
    uint32_t bytes)
{
    uint8_t *cursor;
    uint32_t offset;
    ssize_t got;

    if (fd < 0 || data == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (uint8_t *)data;
    offset = 0u;
    while (offset < bytes)
    {
        got = read(fd, cursor + offset, bytes - offset);
        if (got < 0)
        {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return SPARK_STATUS_BUSY;
            return SPARK_STATUS_IO_ERROR;
        }
        if (got == 0)
            return SPARK_STATUS_IO_ERROR;
        offset += (uint32_t)got;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CudaResidentdWriteMessage(
    SparkGlm52CudaResidentdRuntime *runtime,
    uint32_t kind,
    const void *payload,
    uint32_t payload_bytes)
{
    SparkGlm52CudaResidentIpcHeader header;
    SparkStatus status;

    if (runtime == 0 || runtime->active_client_fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    SparkGlm52CudaResidentIpcInitializeHeader(
        &header,
        kind,
        runtime->rank_plan.rank_index,
        runtime->next_ipc_sequence_number++,
        payload_bytes);
    status = SparkGlm52CudaResidentdWriteFull(
        runtime->active_client_fd,
        &header,
        sizeof(header));
    if (status != SPARK_STATUS_OK)
        return status;
    if (payload_bytes != 0u)
        status = SparkGlm52CudaResidentdWriteFull(
            runtime->active_client_fd,
            payload,
            payload_bytes);
    return status;
}

static void SparkGlm52CudaResidentdCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkGlm52CudaResidentdRuntime *runtime;
    SparkGlm52CudaResidentIpcCompletion message;
    SparkStatus status;

    runtime = (SparkGlm52CudaResidentdRuntime *)completion_context;
    if (runtime == 0 || completion == 0)
        return;
    runtime->completion_count += 1u;
    memset(&message, 0, sizeof(message));
    message.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_BYTES;
    message.completion = *completion;
    pthread_mutex_lock(&runtime->completion_write_mutex);
    status = SparkGlm52CudaResidentdWriteMessage(
        runtime,
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_COMPLETION,
        &message,
        sizeof(message));
    pthread_mutex_unlock(&runtime->completion_write_mutex);
    if (status != SPARK_STATUS_OK)
        runtime->control_error_count += 1u;
}

static SparkStatus SparkGlm52CudaResidentdOpenListenSocket(
    SparkGlm52CudaResidentdRuntime *runtime,
    const char *socket_path)
{
    struct sockaddr_un address;
    int32_t fd;

    if (runtime == 0 || socket_path == 0 || socket_path[0] == '\0')
        return SPARK_STATUS_INVALID_ARGUMENT;
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return SPARK_STATUS_IO_ERROR;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof(address.sun_path))
    {
        close(fd);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path);
    unlink(socket_path);
    if (bind(fd, (const struct sockaddr *)&address, sizeof(address)) != 0)
    {
        close(fd);
        return SPARK_STATUS_IO_ERROR;
    }
    if (listen(fd, 4) != 0)
    {
        close(fd);
        unlink(socket_path);
        return SPARK_STATUS_IO_ERROR;
    }
    if (SparkGlm52CudaResidentdSetNonblocking(fd) != 0)
    {
        close(fd);
        unlink(socket_path);
        return SPARK_STATUS_IO_ERROR;
    }
    runtime->listen_fd = fd;
    chmod(socket_path, 0600);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CudaResidentdLoadTransport(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration)
{
    char rank_buffer[16];
    char port_buffer[16];

    snprintf(rank_buffer, sizeof(rank_buffer), "%u", runtime->rank_plan.rank_index);
    snprintf(port_buffer, sizeof(port_buffer), "%u", configuration->port_base);
    if (setenv("SPARKPIPE_PP13_TRANSPORT_RANK", rank_buffer, 1) != 0 ||
        setenv("SPARKPIPE_PP13_TRANSPORT_PORT_BASE", port_buffer, 1) != 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SparkHiddenTransportLoadInterfaceFromSharedObject(
        configuration->transport_shared_object_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        &runtime->transport_library);
}

static SparkStatus SparkGlm52CudaResidentdOpenHiddenTransport(
    SparkGlm52CudaResidentdRuntime *runtime)
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

static SparkStatus SparkGlm52CudaResidentdBuildNodeContext(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration)
{
    SparkGlm52Pp13NodeContextBuilderConfiguration builder_configuration;
    SparkStatus status;

    memset(&builder_configuration, 0, sizeof(builder_configuration));
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
    builder_configuration.embedding_pack_path = configuration->embedding_pack_path;
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
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    memset(&runtime->builder_result, 0, sizeof(runtime->builder_result));
    status = runtime->builder_library.builder_interface.build(
        runtime->builder_state,
        &runtime->builder_result);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkGlm52Pp13NodeContextBuilderValidateResult(
        &runtime->builder_result,
        &runtime->rank_plan);
}

static SparkStatus SparkGlm52CudaResidentdLoadDriver(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration)
{
    SparkModelDriverCreateRequest create_request;
    char error_buffer[512];
    SparkStatus status;

    error_buffer[0] = '\0';
    status = SparkLoadModelDriver(
        configuration->driver_path,
        configuration->node_target,
        &runtime->loaded_driver,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        snprintf(runtime->blocker, sizeof(runtime->blocker),
            "driver_load_failed status=%u %s", (uint32_t)status, error_buffer);
        return status;
    }
    runtime->program = SparkFindLoadedModelDriverProgram(
        &runtime->loaded_driver,
        configuration->program_name);
    if (runtime->program == 0)
        return SPARK_STATUS_NOT_FOUND;
    memset(&create_request, 0, sizeof(create_request));
    create_request.node_id = runtime->rank_plan.host_name;
    create_request.node_target = configuration->node_target;
    create_request.node_context = runtime->builder_result.node_context;
    create_request.completion_function = SparkGlm52CudaResidentdCompletion;
    create_request.completion_context = runtime;
    status = runtime->loaded_driver.interface->create(
        &create_request,
        &runtime->driver_instance);
    if (status != SPARK_STATUS_OK || runtime->driver_instance == 0)
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CudaResidentdAttachBuilderDriver(
    SparkGlm52CudaResidentdRuntime *runtime)
{
    if (runtime == 0 || runtime->builder_state == 0 ||
        runtime->builder_library.builder_interface.attach_driver == 0 ||
        runtime->loaded_driver.interface == 0 || runtime->driver_instance == 0 ||
        runtime->program == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return runtime->builder_library.builder_interface.attach_driver(
        runtime->builder_state,
        runtime->loaded_driver.interface,
        runtime->driver_instance,
        runtime->program,
        runtime->output_transport_session);
}

static SparkStatus SparkGlm52CudaResidentdInitialize(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration)
{
    SparkStatus status;
    uint32_t slot;
    memset(runtime, 0, sizeof(*runtime));
    runtime->listen_fd = -1;
    for (slot = 0u; slot < SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
        runtime->client_fds[slot] = -1;
    runtime->active_client_fd = -1;
    runtime->state = SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_LOADING;
    SparkLoadedModelDriverReset(&runtime->loaded_driver);
    pthread_mutex_init(&runtime->completion_write_mutex, 0);
    status = SparkGlm52Pp13RuntimeBuildRankPlan(
        configuration->rank_index,
        configuration->max_active_sequence_count,
        configuration->port_base,
        &runtime->rank_plan,
        runtime->blocker,
        sizeof(runtime->blocker));
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
        &runtime->rank_plan,
        configuration->fp8_pack_root,
        runtime->blocker,
        sizeof(runtime->blocker));
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentdBuildNodeContext(runtime, configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentdLoadTransport(runtime, configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentdOpenHiddenTransport(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentdLoadDriver(runtime, configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentdAttachBuilderDriver(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlm52CudaResidentdOpenListenSocket(runtime, configuration->socket_path);
    if (status != SPARK_STATUS_OK)
        return status;
    runtime->state = SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_READY;
    return SPARK_STATUS_OK;
}

static void SparkGlm52CudaResidentdDestroy(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration)
{
    uint32_t slot;
    if (runtime == 0)
        return;
    for (slot = 0u; slot < SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
        if (runtime->client_fds[slot] >= 0)
            close(runtime->client_fds[slot]);
    if (runtime->listen_fd >= 0)
        close(runtime->listen_fd);
    if (configuration != 0 && configuration->socket_path != 0)
        unlink(configuration->socket_path);
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
        runtime->builder_library.builder_interface.destroy(runtime->builder_state);
    SparkGlm52Pp13NodeContextBuilderUnloadInterface(&runtime->builder_library);
    SparkHiddenTransportClose(runtime->input_transport_session);
    SparkHiddenTransportClose(runtime->output_transport_session);
    SparkHiddenTransportUnloadInterface(&runtime->transport_library);
    pthread_mutex_destroy(&runtime->completion_write_mutex);
}

static uint32_t SparkGlm52CudaResidentdAcceptClient(
    SparkGlm52CudaResidentdRuntime *runtime)
{
    int32_t fd;
    uint32_t slot;
    if (runtime == 0 || runtime->listen_fd < 0)
        return 0u;
    for (slot = 0u; slot < SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
        if (runtime->client_fds[slot] < 0)
            break;
    if (slot >= SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS)
        return 0u;
    fd = accept(runtime->listen_fd, 0, 0);
    if (fd < 0)
        return 0u;
    runtime->client_fds[slot] = fd;
    return 1u;
}

static SparkStatus SparkGlm52CudaResidentdFillStats(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration,
    SparkGlm52CudaResidentIpcStats *stats)
{
    SparkModelDriverRuntimeSnapshot snapshot;
    SparkStatus status;

    memset(stats, 0, sizeof(*stats));
    stats->descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_STATS_BYTES;
    stats->state = runtime->state;
    stats->capability_flags =
        SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_DRIVER_RESIDENT |
        SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_BUILDER_RESIDENT |
        SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_TRANSPORT_RESIDENT |
        SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_CUDA_STATE_RESIDENT;
    stats->rank_index = runtime->rank_plan.rank_index;
    stats->max_active_sequence_count = configuration->max_active_sequence_count;
    stats->submitted_count = runtime->submitted_count;
    stats->completed_count = runtime->completion_count;
    stats->rejected_count = runtime->submit_failed_count;
    stats->cuda_generation = configuration->cuda_generation;
    stats->control_generation = configuration->control_generation;
    snprintf(stats->blocker, sizeof(stats->blocker), "%s", runtime->blocker);
    if (runtime->loaded_driver.interface != 0 &&
        runtime->loaded_driver.interface->snapshot != 0 &&
        runtime->driver_instance != 0 && runtime->program != 0)
    {
        memset(&snapshot, 0, sizeof(snapshot));
        snapshot.descriptor_bytes = (uint32_t)sizeof(snapshot);
        status = runtime->loaded_driver.interface->snapshot(
            runtime->driver_instance,
            runtime->program->program_id,
            &snapshot);
        if (status == SPARK_STATUS_OK)
        {
            stats->active_submission_count = snapshot.active_submission_count;
            stats->available_dispatch_slot_count =
                snapshot.available_dispatch_slot_count;
            stats->private_queue_pressure = snapshot.private_queue_pressure;
            stats->submitted_count = snapshot.submitted_count;
            stats->completed_count = snapshot.completed_count;
            stats->rejected_count = snapshot.rejected_count;
            stats->resident_sequence_count = snapshot.resident_sequence_count;
            stats->resident_token_count = snapshot.resident_token_count;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CudaResidentdHandleSubmitWork(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration,
    const SparkGlm52CudaResidentIpcSubmitWork *message)
{
    SparkGlm52CudaResidentIpcSubmitResult result;
    SparkStatus status;

    if (message == 0 || message->descriptor_bytes !=
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES)
        return SPARK_STATUS_ABI_MISMATCH;
    status = runtime->builder_library.builder_interface.submit_work(
        runtime->builder_state,
        &message->work_packet,
        runtime->input_transport_session,
        runtime->output_transport_session,
        SparkGlm52CudaResidentdCompletion,
        runtime);
    if (status == SPARK_STATUS_OK)
        runtime->submitted_count += 1u;
    else
        runtime->submit_failed_count += 1u;
    memset(&result, 0, sizeof(result));
    result.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
    result.status = (uint32_t)status;
    SparkGlm52CudaResidentdFillStats(runtime, configuration, &result.stats);
    result.stats.rejected_count = runtime->submit_failed_count;
    if (status != SPARK_STATUS_OK)
        snprintf(result.stats.blocker, sizeof(result.stats.blocker),
            "submit_work_failed status=%u", (uint32_t)status);
    pthread_mutex_lock(&runtime->completion_write_mutex);
    (void)SparkGlm52CudaResidentdWriteMessage(
        runtime,
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT,
        &result,
        sizeof(result));
    pthread_mutex_unlock(&runtime->completion_write_mutex);
    return status;
}

static uint32_t SparkGlm52CudaResidentdPumpClient(
    SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration,
    uint32_t slot)
{
    SparkGlm52CudaResidentIpcHeader header;
    uint8_t payload[SPARK_GLM52_CUDA_RESIDENTD_CONTROL_PAYLOAD_CAPACITY];
    SparkGlm52CudaResidentIpcStats stats;
    SparkStatus status;
    int32_t fd;
    if (runtime == 0 || slot >= SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS)
        return 0u;
    fd = runtime->client_fds[slot];
    if (fd < 0)
        return 0u;
    runtime->active_client_fd = fd;
    status = SparkGlm52CudaResidentdReadFull(fd, &header, sizeof(header));
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        close(fd);
        runtime->client_fds[slot] = -1;
        runtime->active_client_fd = -1;
        return 1u;
    }
    status = SparkGlm52CudaResidentIpcValidateHeader(
        &header,
        0u,
        sizeof(payload));
    if (status != SPARK_STATUS_OK)
    {
        runtime->control_error_count += 1u;
        close(fd);
        runtime->client_fds[slot] = -1;
        runtime->active_client_fd = -1;
        return 1u;
    }
    if (header.payload_bytes != 0u)
    {
        status = SparkGlm52CudaResidentdReadFull(
            fd,
            payload,
            header.payload_bytes);
        if (status != SPARK_STATUS_OK)
        {
            close(fd);
            runtime->client_fds[slot] = -1;
            runtime->active_client_fd = -1;
            return 1u;
        }
    }
    switch (header.kind)
    {
        case SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO:
            SparkGlm52CudaResidentdFillStats(runtime, configuration, &stats);
            pthread_mutex_lock(&runtime->completion_write_mutex);
            (void)SparkGlm52CudaResidentdWriteMessage(
                runtime,
                SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO_ACK,
                &stats,
                sizeof(stats));
            pthread_mutex_unlock(&runtime->completion_write_mutex);
            break;
        case SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_QUERY:
            SparkGlm52CudaResidentdFillStats(runtime, configuration, &stats);
            pthread_mutex_lock(&runtime->completion_write_mutex);
            (void)SparkGlm52CudaResidentdWriteMessage(
                runtime,
                SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_STATS,
                &stats,
                sizeof(stats));
            pthread_mutex_unlock(&runtime->completion_write_mutex);
            break;
        case SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK:
            (void)SparkGlm52CudaResidentdHandleSubmitWork(
                runtime,
                configuration,
                (const SparkGlm52CudaResidentIpcSubmitWork *)payload);
            break;
        case SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SHUTDOWN:
            SparkGlm52CudaResidentdRunning = 0;
            break;
        default:
            runtime->control_error_count += 1u;
            break;
    }
    return 1u;
}

static void SparkGlm52CudaResidentdProgressDriver(
    SparkGlm52CudaResidentdRuntime *runtime)
{
    SparkModelDriverRuntimeSnapshot snapshot;

    if (runtime == 0 || runtime->loaded_driver.interface == 0 ||
        runtime->loaded_driver.interface->snapshot == 0 ||
        runtime->driver_instance == 0 || runtime->program == 0)
        return;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.descriptor_bytes = (uint32_t)sizeof(snapshot);
    (void)runtime->loaded_driver.interface->snapshot(
        runtime->driver_instance,
        runtime->program->program_id,
        &snapshot);
}

static void SparkGlm52CudaResidentdPrintReady(
    const SparkGlm52CudaResidentdRuntime *runtime,
    const SparkGlm52CudaResidentdConfiguration *configuration)
{
    printf("glm52_cuda_residentd=1\n");
    printf("rank=%u\n", runtime->rank_plan.rank_index);
    printf("host=%s\n", runtime->rank_plan.host_name);
    printf("socket=%s\n", configuration->socket_path);
    printf("stage=%u:%u\n",
        runtime->rank_plan.first_layer_index,
        runtime->rank_plan.layer_count);
    printf("fp8_pack_root=%s\n", configuration->fp8_pack_root);
    printf("transport_so=%s\n", configuration->transport_shared_object_path);
    printf("driver_so=%s\n", configuration->driver_path);
    printf("node_context_builder_so=%s\n",
        configuration->node_context_builder_shared_object_path);
    printf("state=ready\n");
    fflush(stdout);
}

static void SparkGlm52CudaResidentdUsage(const char *program)
{
    fprintf(stderr,
        "usage: %s --rank n --socket path --fp8-pack-root dir --stagepack-root dir --transport-so path --driver-so path --node-context-builder-so path --embedding-pack path [--program name] [--node-target target] [--max-active n] [--port-base n]\n",
        program);
}

int main(int argc, char **argv)
{
    SparkGlm52CudaResidentdConfiguration configuration;
    SparkGlm52CudaResidentdRuntime runtime;
    SparkStatus status;
    struct pollfd fds[SPARK_GLM52_CUDA_RESIDENTD_POLL_CAPACITY];
    int32_t index;
    int poll_status;
    uint32_t slot;

    SparkGlm52CudaResidentdInitializeConfiguration(&configuration);
    for (index = 1; index < argc; ++index)
    {
        if (SparkGlm52CudaResidentdApplyArgument(
                &configuration,
                argc,
                argv,
                &index) != 0)
        {
            SparkGlm52CudaResidentdUsage(argv[0]);
            return 2;
        }
    }
    status = SparkGlm52CudaResidentdValidateConfiguration(&configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52CudaResidentdUsage(argv[0]);
        return 2;
    }
    signal(SIGINT, SparkGlm52CudaResidentdSignal);
    signal(SIGTERM, SparkGlm52CudaResidentdSignal);
    status = SparkGlm52CudaResidentdInitialize(&runtime, &configuration);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "glm52_cuda_residentd_init_failed status=%u blocker=%s\n",
            (uint32_t)status,
            runtime.blocker);
        SparkGlm52CudaResidentdDestroy(&runtime, &configuration);
        return 1;
    }
    SparkGlm52CudaResidentdPrintReady(&runtime, &configuration);
    while (SparkGlm52CudaResidentdRunning)
    {
        memset(fds, 0, sizeof(fds));
        fds[SPARK_GLM52_CUDA_RESIDENTD_POLL_LISTEN].fd = runtime.listen_fd;
        fds[SPARK_GLM52_CUDA_RESIDENTD_POLL_LISTEN].events = POLLIN;
        for (slot = 0u; slot < SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
        {
            fds[slot + 1u].fd = runtime.client_fds[slot];
            fds[slot + 1u].events = POLLIN;
        }
        poll_status = poll(
            fds,
            SPARK_GLM52_CUDA_RESIDENTD_POLL_CAPACITY,
            SPARK_GLM52_CUDA_RESIDENTD_POLL_TIMEOUT_MS);
        if (poll_status > 0)
        {
            if ((fds[SPARK_GLM52_CUDA_RESIDENTD_POLL_LISTEN].revents & POLLIN) != 0)
                (void)SparkGlm52CudaResidentdAcceptClient(&runtime);
            for (slot = 0u; slot < SPARK_GLM52_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
                if (runtime.client_fds[slot] >= 0 &&
                    (fds[slot + 1u].revents &
                        (POLLIN | POLLHUP | POLLERR)) != 0)
                    (void)SparkGlm52CudaResidentdPumpClient(&runtime, &configuration, slot);
        }
        SparkGlm52CudaResidentdProgressDriver(&runtime);
    }
    SparkGlm52CudaResidentdDestroy(&runtime, &configuration);
    return 0;
}
