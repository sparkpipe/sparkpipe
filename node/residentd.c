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
#include "sparkpipe/spark_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_kv_store.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_ring_node_context_builder.h"
#include "sparkpipe/spark_ring_runtime.h"
#include "sparkpipe/spark_prompt_pipeline.h"
#include "sparkpipe/spark_request_api.h"
#include "sparkpipe/spark_serving_engine.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "runtime/net.h"

#define SPARK_CUDA_RESIDENTD_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_CUDA_RESIDENTD_DEFAULT_PROGRAM "glm52.ring.rank.production"
#define SPARK_CUDA_RESIDENTD_DEFAULT_SOCKET_PREFIX "/tmp/sparkpipe_glm52_cuda_resident_rank"
#define SPARK_CUDA_RESIDENTD_INITIAL_CONTROL_PAYLOAD_BYTES (1024u * 1024u)
#define SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY 256u
#define SPARK_CUDA_RESIDENTD_OUTPUT_QUEUE_CAPACITY \
    ((2u * SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY) + 16u)
#define SPARK_CUDA_RESIDENTD_OUTPUT_CONTROL_LIMIT \
    (SPARK_CUDA_RESIDENTD_OUTPUT_QUEUE_CAPACITY - \
     SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY)
#define SPARK_CUDA_RESIDENTD_OUTPUT_FLUSH_MESSAGE_LIMIT 64u
#define SPARK_CUDA_RESIDENTD_TRANSPORT_POLL_CAPACITY 4u
#define SPARK_CUDA_RESIDENTD_REQUIRED_TRANSPORT_CAPS \
    (SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS | \
     SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS)

#define SPARK_CUDA_RESIDENTD_POLL_LISTEN 0u
#define SPARK_CUDA_RESIDENTD_MAX_CLIENTS 4u
#define SPARK_CUDA_RESIDENTD_POLL_CLIENT_BASE 1u
#define SPARK_CUDA_RESIDENTD_POLL_WAKE \
    (SPARK_CUDA_RESIDENTD_POLL_CLIENT_BASE + \
     SPARK_CUDA_RESIDENTD_MAX_CLIENTS)
#define SPARK_CUDA_RESIDENTD_POLL_BASE_COUNT \
    (SPARK_CUDA_RESIDENTD_POLL_WAKE + 1u)
#define SPARK_CUDA_RESIDENTD_POLL_CAPACITY \
    (SPARK_CUDA_RESIDENTD_POLL_BASE_COUNT + \
     (2u * SPARK_CUDA_RESIDENTD_TRANSPORT_POLL_CAPACITY))

static volatile sig_atomic_t SparkGlm52CudaResidentdRunning = 1;
static volatile sig_atomic_t SparkGlm52CudaResidentdWakeWriteFd = -1;

typedef struct SparkCudaResidentdConfiguration
{
    const char *socket_path;
    const char *moe_pack_root;
    const char *stagepack_root;
    const char *transport_shared_object_path;
    const char *driver_path;
    const char *node_context_builder_shared_object_path;
    const char *embedding_pack_path;
    const char *dspark_manifest_path;
    const char *dspark_config_path;
    const char *dspark_safetensors_path;
    const char *kv_nvme_path;
    const char *kv_store_module_path;
    const char *kv_store_service_address;
    const char *kv_store_ipc_socket_path;
    const char *program_name;
    const char *node_target;
    uint32_t rank_index;
    uint32_t rank_is_set;
    uint32_t max_active_sequence_count;
	uint32_t kv_pool_token_capacity;
    uint32_t kv_nvme_block_capacity;
    uint32_t kv_nvme_batch_block_count;
    uint32_t kv_store_block_capacity;
    uint32_t kv_store_batch_block_count;
    uint32_t kv_store_worker_count;
    uint32_t kv_store_lookahead_packet_count;
    uint32_t port_base;
    uint32_t model_quantization_mode;
    uint32_t speculation_enabled;
	uint32_t mtp_enabled;
    uint32_t dspark_maximum_context_token_count;
    uint64_t cuda_generation;
    uint64_t control_generation;
    uint64_t kv_store_model_fingerprint;
    uint64_t kv_store_layout_fingerprint;
    uint64_t kv_store_client_memory_pool_bytes;
    uint64_t kv_store_local_buffer_bytes;
} SparkCudaResidentdConfiguration;

typedef struct SparkCudaResidentdQueuedWork
{
    SparkRingWorkControlPacket packet;
    int32_t client_fd;
    uint32_t reserved0;
    uint64_t enqueue_time_ns;
} SparkCudaResidentdQueuedWork;

typedef union SparkGlm52CudaResidentdOutputPayload
{
    SparkCudaResidentIpcCompletion completion;
    SparkCudaResidentIpcSubmitResult submit_result;
    SparkCudaResidentIpcStats stats;
} SparkGlm52CudaResidentdOutputPayload;

typedef struct SparkCudaResidentdOutputMessage
{
    uint8_t bytes[
        SPARK_CUDA_RESIDENT_IPC_HEADER_BYTES +
        sizeof(SparkGlm52CudaResidentdOutputPayload)];
    uint32_t byte_count;
    uint32_t byte_offset;
} SparkCudaResidentdOutputMessage;

typedef struct SparkCudaResidentdClient
{
    SparkCudaResidentIpcReader reader;
    uint8_t *payload;
    uint32_t payload_capacity;
    SparkCudaResidentdOutputMessage
        output_queue[SPARK_CUDA_RESIDENTD_OUTPUT_QUEUE_CAPACITY];
    uint32_t output_queue_head;
    uint32_t output_queue_count;
    int32_t fd;
} SparkCudaResidentdClient;

typedef struct SparkCudaResidentdRuntime
{
    SparkRingRuntimeRankPlan rank_plan;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *input_transport_session;
    SparkHiddenTransportSession *output_transport_session;
    SparkRingNodeContextBuilderDynamicLibrary builder_library;
    void *builder_state;
    SparkRingNodeContextBuilderResult builder_result;
    SparkLoadedModelDriver loaded_driver;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    int32_t listen_fd;
    SparkCudaResidentdClient
        clients[SPARK_CUDA_RESIDENTD_MAX_CLIENTS];
    int32_t wake_pipe_read_fd;
    int32_t wake_pipe_write_fd;
    pthread_mutex_t output_mutex;
    uint64_t next_ipc_sequence_number;
    uint64_t submitted_count;
    uint64_t submit_failed_count;
    uint64_t completion_count;
    uint64_t control_error_count;
    uint64_t ingest_prefill_count;
    uint64_t ingest_decode_count;
    uint32_t state;
    SparkCudaResidentdQueuedWork *work_queue;
    uint32_t work_queue_head;
    uint32_t work_queue_count;
    uint32_t kv_prefetch_lookahead_packet_count;
    SparkCudaResidentdQueuedWork pending_prefill_work;
    uint32_t pending_prefill_active;
    uint32_t driver_inflight_count;
    uint64_t inflight_submit_time_ns;
    uint32_t packet_timing_enabled;
    int32_t completion_client_fd;
    uint64_t work_queue_accepted_count;
    uint64_t work_queue_submit_count;
    uint64_t work_queue_error_count;
    uint64_t active_control_generation;
    uint32_t synthetic_failure_completion_active;
    SparkStatus deferred_failure_status;
    char blocker[SPARK_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES];
} SparkCudaResidentdRuntime;

static void SparkCudaResidentdSignal(int signal_number)
{
    uint8_t byte;
    ssize_t wrote;

    (void)signal_number;
    SparkGlm52CudaResidentdRunning = 0;
    if (SparkGlm52CudaResidentdWakeWriteFd >= 0)
    {
        byte = 1u;
        wrote = write((int32_t)SparkGlm52CudaResidentdWakeWriteFd,
            &byte,sizeof(byte));
        if (wrote < 0)
            return;
    }
}

static SparkStatus SparkCudaResidentdEnsureClientPayloadCapacity(
    SparkCudaResidentdClient *client,
    uint32_t required_bytes)
{
    uint32_t grown_capacity;
    uint8_t *grown_payload;
    if (client == 0 || required_bytes >
        SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (required_bytes <= client->payload_capacity)
        return SPARK_STATUS_OK;
    grown_capacity = client->payload_capacity;
    if (grown_capacity == 0u)
        grown_capacity =
            SPARK_CUDA_RESIDENTD_INITIAL_CONTROL_PAYLOAD_BYTES;
    while (grown_capacity < required_bytes &&
        grown_capacity <=
            SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES / 2u)
        grown_capacity *= 2u;
    if (grown_capacity < required_bytes)
        grown_capacity = required_bytes;
    grown_payload = (uint8_t *)realloc(
        client->payload,grown_capacity);
    if (grown_payload == 0)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    client->payload = grown_payload;
    client->payload_capacity = grown_capacity;
    return SPARK_STATUS_OK;
}

static void SparkCudaResidentdResetClientMessage(
    SparkCudaResidentdClient *client)
{
    if (client == 0)
        return;
    SparkCudaResidentIpcReaderReset(&client->reader);
}

static SparkStatus SparkCudaResidentdReadClientMessage(
    SparkCudaResidentdClient *client)
{
    SparkStatus status;
    if (client == 0 || client->fd < 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkCudaResidentIpcReadHeader(
        &client->reader,client->fd,
        SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentdEnsureClientPayloadCapacity(
        client,client->reader.header.payload_bytes);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkCudaResidentIpcReadPayload(
        &client->reader,client->fd,client->payload,
        client->payload_capacity);
}

static void SparkCudaResidentdInitializeConfiguration(
    SparkCudaResidentdConfiguration *configuration)
{
    memset(configuration, 0, sizeof(*configuration));
    configuration->program_name = SPARK_CUDA_RESIDENTD_DEFAULT_PROGRAM;
    configuration->max_active_sequence_count =
        SPARK_CUDA_RESIDENTD_DEFAULT_MAX_ACTIVE;
	configuration->kv_pool_token_capacity = SPARK_GLM52_KV_POOL_TOKENS;
    configuration->kv_nvme_block_capacity =
        SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BLOCK_CAPACITY;
    configuration->kv_nvme_batch_block_count =
        SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BATCH_BLOCK_COUNT;
    configuration->kv_store_block_capacity =
        SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BLOCK_CAPACITY;
    configuration->kv_store_batch_block_count =
        SPARK_KV_STORE_MAX_BATCH_BLOCKS;
    configuration->kv_store_worker_count = 2u;
    configuration->kv_store_lookahead_packet_count =
        SPARK_KV_STORE_DEFAULT_LOOKAHEAD_PACKETS;
    configuration->port_base = SPARK_RING_RUNTIME_DEFAULT_PORT_BASE;
    configuration->model_quantization_mode =
        SPARK_RING_RUNTIME_DEFAULT_QUANTIZATION_MODE;
    configuration->dspark_maximum_context_token_count = 2048u;
}

#include <stddef.h>
#include "sparkpipe/spark_options.h"

static const SparkOption spark_cuda_residentd_options[] =
{
	{ "--socket", SPARK_OPTION_STRING, -1, (uint16_t)offsetof(SparkCudaResidentdConfiguration, socket_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--rank", SPARK_OPTION_U32, -2, (uint16_t)offsetof(SparkCudaResidentdConfiguration, rank_index), (uint16_t)offsetof(SparkCudaResidentdConfiguration, rank_is_set) },
	{ "--max-active", SPARK_OPTION_U32, -3, (uint16_t)offsetof(SparkCudaResidentdConfiguration, max_active_sequence_count), SPARK_OPTION_NO_SET_FLAG },
	{ "--kv-pool-tokens", SPARK_OPTION_U32, -17, (uint16_t)offsetof(SparkCudaResidentdConfiguration, kv_pool_token_capacity), SPARK_OPTION_NO_SET_FLAG },
	{ "--kv-nvme-path", SPARK_OPTION_STRING, -18, (uint16_t)offsetof(SparkCudaResidentdConfiguration, kv_nvme_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--kv-nvme-blocks", SPARK_OPTION_U32, -19, (uint16_t)offsetof(SparkCudaResidentdConfiguration, kv_nvme_block_capacity), SPARK_OPTION_NO_SET_FLAG },
	{ "--kv-nvme-batch-blocks", SPARK_OPTION_U32, -20, (uint16_t)offsetof(SparkCudaResidentdConfiguration, kv_nvme_batch_block_count), SPARK_OPTION_NO_SET_FLAG },
	{ "--port-base", SPARK_OPTION_U32, -4, (uint16_t)offsetof(SparkCudaResidentdConfiguration, port_base), SPARK_OPTION_NO_SET_FLAG },
	{ "--cuda-generation", SPARK_OPTION_U64, -5, (uint16_t)offsetof(SparkCudaResidentdConfiguration, cuda_generation), SPARK_OPTION_NO_SET_FLAG },
	{ "--control-generation", SPARK_OPTION_U64, -6, (uint16_t)offsetof(SparkCudaResidentdConfiguration, control_generation), SPARK_OPTION_NO_SET_FLAG },
	{ "--moe-pack-root", SPARK_OPTION_STRING, -7, (uint16_t)offsetof(SparkCudaResidentdConfiguration, moe_pack_root), SPARK_OPTION_NO_SET_FLAG },
	{ "--model-quantization", SPARK_OPTION_QUANT, -21, (uint16_t)offsetof(SparkCudaResidentdConfiguration, model_quantization_mode), SPARK_OPTION_NO_SET_FLAG },
	{ "--stagepack-root", SPARK_OPTION_STRING, -8, (uint16_t)offsetof(SparkCudaResidentdConfiguration, stagepack_root), SPARK_OPTION_NO_SET_FLAG },
	{ "--transport-so", SPARK_OPTION_STRING, -9, (uint16_t)offsetof(SparkCudaResidentdConfiguration, transport_shared_object_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--driver-so", SPARK_OPTION_STRING, -10, (uint16_t)offsetof(SparkCudaResidentdConfiguration, driver_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--node-context-builder-so", SPARK_OPTION_STRING, -11, (uint16_t)offsetof(SparkCudaResidentdConfiguration, node_context_builder_shared_object_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--embedding-pack", SPARK_OPTION_STRING, -12, (uint16_t)offsetof(SparkCudaResidentdConfiguration, embedding_pack_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--dspark", SPARK_OPTION_FLAG, -1, (uint16_t)offsetof(SparkCudaResidentdConfiguration, speculation_enabled), SPARK_OPTION_NO_SET_FLAG },
	{ "--mtp", SPARK_OPTION_FLAG, -1, (uint16_t)offsetof(SparkCudaResidentdConfiguration, mtp_enabled), SPARK_OPTION_NO_SET_FLAG },
	{ "--dspark-safetensors", SPARK_OPTION_STRING, -15, (uint16_t)offsetof(SparkCudaResidentdConfiguration, dspark_safetensors_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--dspark-manifest", SPARK_OPTION_STRING, -17, (uint16_t)offsetof(SparkCudaResidentdConfiguration, dspark_manifest_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--dspark-config", SPARK_OPTION_STRING, -18, (uint16_t)offsetof(SparkCudaResidentdConfiguration, dspark_config_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--dspark-max-context", SPARK_OPTION_U32, -16, (uint16_t)offsetof(SparkCudaResidentdConfiguration, dspark_maximum_context_token_count), SPARK_OPTION_NO_SET_FLAG },
	{ "--program", SPARK_OPTION_STRING, -13, (uint16_t)offsetof(SparkCudaResidentdConfiguration, program_name), SPARK_OPTION_NO_SET_FLAG },
	{ "--node-target", SPARK_OPTION_STRING, -14, (uint16_t)offsetof(SparkCudaResidentdConfiguration, node_target), SPARK_OPTION_NO_SET_FLAG },
};

static int32_t SparkCudaResidentdApplyArgument(
    SparkCudaResidentdConfiguration *configuration,
    int argc,
    char **argv,
    int32_t *index)
{
	int32_t status;
	status = SparkOptionsApply(spark_cuda_residentd_options,
		(uint32_t)(sizeof(spark_cuda_residentd_options) / sizeof(spark_cuda_residentd_options[0])),
		configuration, argc, argv, index);
	if (status <= 0)
		return status;
	return -1;
}

static SparkStatus SparkCudaResidentdValidateConfiguration(
    SparkCudaResidentdConfiguration *configuration)
{
    static char default_socket_path[128];

    if (configuration->rank_is_set == 0u ||
        configuration->moe_pack_root == 0 ||
        configuration->stagepack_root == 0 ||
        configuration->transport_shared_object_path == 0 ||
        configuration->driver_path == 0 ||
        configuration->node_context_builder_shared_object_path == 0 ||
        configuration->embedding_pack_path == 0 ||
		SparkRingRuntimeQuantizationModeName(
			configuration->model_quantization_mode) == 0 ||
		configuration->kv_pool_token_capacity == 0u ||
		configuration->kv_pool_token_capacity > SPARK_GLM52_KV_POOL_TOKENS ||
		(configuration->kv_pool_token_capacity % SPARK_GLM52_KV_BLOCK_TOKENS) != 0u ||
		configuration->max_active_sequence_count == 0u ||
		configuration->max_active_sequence_count >
			SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (configuration->max_active_sequence_count >=
            SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT &&
        configuration->kv_nvme_path == 0 &&
        configuration->kv_store_module_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (configuration->kv_nvme_path != 0 &&
        configuration->kv_store_module_path != 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (configuration->kv_nvme_path != 0 &&
        (configuration->kv_nvme_block_capacity <
            configuration->kv_pool_token_capacity / SPARK_GLM52_KV_BLOCK_TOKENS ||
         configuration->kv_nvme_block_capacity > UINT32_MAX / 2u ||
         configuration->kv_nvme_batch_block_count == 0u ||
         configuration->kv_nvme_batch_block_count >
            SPARK_RING_NODE_CONTEXT_BUILDER_MAX_NVME_BATCH_BLOCK_COUNT))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (configuration->kv_store_module_path != 0 &&
        (configuration->kv_store_service_address == 0 ||
         configuration->kv_store_ipc_socket_path == 0 ||
         configuration->kv_store_block_capacity <
            configuration->kv_pool_token_capacity / SPARK_GLM52_KV_BLOCK_TOKENS ||
         configuration->kv_store_block_capacity > UINT32_MAX / 2u ||
         configuration->kv_store_batch_block_count == 0u ||
         configuration->kv_store_batch_block_count >
            SPARK_KV_STORE_MAX_BATCH_BLOCKS ||
         configuration->kv_store_worker_count == 0u ||
         configuration->kv_store_lookahead_packet_count == 0u ||
         configuration->kv_store_lookahead_packet_count >
            SPARK_KV_STORE_MAX_LOOKAHEAD_PACKETS ||
         configuration->kv_store_model_fingerprint == 0u ||
         configuration->kv_store_layout_fingerprint == 0u ||
         configuration->kv_store_client_memory_pool_bytes == 0u ||
         configuration->kv_store_local_buffer_bytes == 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (configuration->socket_path == 0)
    {
        snprintf(default_socket_path, sizeof(default_socket_path), "%s%u.sock",
            SPARK_CUDA_RESIDENTD_DEFAULT_SOCKET_PREFIX,
            configuration->rank_index);
        configuration->socket_path = default_socket_path;
    }
    if (configuration->speculation_enabled != 0u &&
        configuration->rank_index == SPARK_RING_RUNTIME_STAGE_COUNT - 1u &&
        (configuration->dspark_manifest_path == 0 ||
         configuration->dspark_config_path == 0 ||
         configuration->dspark_safetensors_path == 0 ||
         configuration->dspark_maximum_context_token_count == 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdOpenWakePipe(
    SparkCudaResidentdRuntime *runtime)
{
    int32_t pipe_fds[2];

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (pipe(pipe_fds) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (SparkNetSetNonblocking(pipe_fds[0]) < 0 ||
        SparkNetSetNonblocking(pipe_fds[1]) < 0)
    {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->wake_pipe_read_fd = pipe_fds[0];
    runtime->wake_pipe_write_fd = pipe_fds[1];
    SparkGlm52CudaResidentdWakeWriteFd =
        (sig_atomic_t)runtime->wake_pipe_write_fd;
    return SPARK_STATUS_OK;
}

static void SparkCudaResidentdSignalWake(
    SparkCudaResidentdRuntime *runtime)
{
    uint8_t byte;
    ssize_t wrote;

    if (runtime == 0 || runtime->wake_pipe_write_fd < 0)
        return;
    byte = 1u;
    wrote = write(runtime->wake_pipe_write_fd,&byte,sizeof(byte));
    if (wrote < 0)
        return;
}

static void SparkCudaResidentdDriverWake(void *wake_context)
{
    SparkCudaResidentdSignalWake(
        (SparkCudaResidentdRuntime *)wake_context);
}

static void SparkCudaResidentdDrainWakePipe(
    SparkCudaResidentdRuntime *runtime)
{
    uint8_t buffer[64];
    ssize_t got;

    if (runtime == 0 || runtime->wake_pipe_read_fd < 0)
        return;
    for (;;)
    {
        got = read(runtime->wake_pipe_read_fd,buffer,sizeof(buffer));
        if (got <= 0)
            return;
    }
}

static SparkCudaResidentdClient *
SparkCudaResidentdFindClientByFd(
    SparkCudaResidentdRuntime *runtime,
    int32_t client_fd)
{
    uint32_t slot;
    if (runtime == 0 || client_fd < 0)
        return 0;
    for (slot = 0u; slot < SPARK_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
        if (runtime->clients[slot].fd == client_fd)
            return &runtime->clients[slot];
    return 0;
}

static SparkStatus SparkCudaResidentdQueueMessageLocked(
    SparkCudaResidentdRuntime *runtime,
    int32_t client_fd,
    uint32_t kind,
    const void *payload,
    uint32_t payload_bytes)
{
    SparkCudaResidentdClient *client;
    SparkCudaResidentdOutputMessage *message;
    SparkCudaResidentIpcHeader header;
    uint32_t tail;

    if (runtime == 0 || client_fd < 0 ||
        payload_bytes > sizeof(SparkGlm52CudaResidentdOutputPayload) ||
        (payload == 0 && payload_bytes != 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    client = SparkCudaResidentdFindClientByFd(runtime,client_fd);
    if (client == 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    if (client->output_queue_count >=
        SPARK_CUDA_RESIDENTD_OUTPUT_QUEUE_CAPACITY)
        return SPARK_STATUS_BUSY;
    tail = (client->output_queue_head + client->output_queue_count) %
        SPARK_CUDA_RESIDENTD_OUTPUT_QUEUE_CAPACITY;
    message = &client->output_queue[tail];
    SparkCudaResidentIpcInitializeHeader(
        &header,
        kind,
        runtime->rank_plan.rank_index,
        runtime->next_ipc_sequence_number++,
        payload_bytes);
    memcpy(message->bytes,&header,sizeof(header));
    if (payload_bytes != 0u)
        memcpy(message->bytes + sizeof(header),payload,payload_bytes);
    message->byte_count = (uint32_t)sizeof(header) + payload_bytes;
    message->byte_offset = 0u;
    client->output_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdQueueMessage(
    SparkCudaResidentdRuntime *runtime,
    int32_t client_fd,
    uint32_t kind,
    const void *payload,
    uint32_t payload_bytes)
{
    SparkStatus status;
    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&runtime->output_mutex);
    status = SparkCudaResidentdQueueMessageLocked(
        runtime,client_fd,kind,payload,payload_bytes);
    pthread_mutex_unlock(&runtime->output_mutex);
    if (status == SPARK_STATUS_OK)
        SparkCudaResidentdSignalWake(runtime);
    return status;
}

static SparkStatus SparkCudaResidentdQueueCompletion(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentIpcCompletion *message)
{
    SparkStatus status;
    if (runtime == 0 || message == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&runtime->output_mutex);
    status = SparkCudaResidentdQueueMessageLocked(
        runtime,runtime->completion_client_fd,
        SPARK_CUDA_RESIDENT_IPC_KIND_COMPLETION,
        message,sizeof(*message));
    pthread_mutex_unlock(&runtime->output_mutex);
    if (status == SPARK_STATUS_OK)
        SparkCudaResidentdSignalWake(runtime);
    return status;
}

static SparkStatus SparkCudaResidentdFlushClientOutput(
    SparkCudaResidentdRuntime *runtime,
    uint32_t slot)
{
    SparkCudaResidentdClient *client;
    SparkCudaResidentdOutputMessage *message;
    uint32_t flushed_count;
    ssize_t written;
    if (runtime == 0 || slot >= SPARK_CUDA_RESIDENTD_MAX_CLIENTS)
        return SPARK_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&runtime->output_mutex);
    client = &runtime->clients[slot];
    if (client->fd < 0)
    {
        pthread_mutex_unlock(&runtime->output_mutex);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    flushed_count = 0u;
    while (client->output_queue_count != 0u &&
        flushed_count < SPARK_CUDA_RESIDENTD_OUTPUT_FLUSH_MESSAGE_LIMIT)
    {
        message = &client->output_queue[client->output_queue_head];
        written = send(client->fd,
            message->bytes + message->byte_offset,
            message->byte_count - message->byte_offset,
            0);
        if (written < 0)
        {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                pthread_mutex_unlock(&runtime->output_mutex);
                return SPARK_STATUS_BUSY;
            }
            pthread_mutex_unlock(&runtime->output_mutex);
            return SPARK_STATUS_IO_ERROR;
        }
        if (written == 0)
        {
            pthread_mutex_unlock(&runtime->output_mutex);
            return SPARK_STATUS_IO_ERROR;
        }
        message->byte_offset += (uint32_t)written;
        if (message->byte_offset != message->byte_count)
            continue;
        client->output_queue_head = (client->output_queue_head + 1u) %
            SPARK_CUDA_RESIDENTD_OUTPUT_QUEUE_CAPACITY;
        client->output_queue_count -= 1u;
        flushed_count += 1u;
    }
    pthread_mutex_unlock(&runtime->output_mutex);
    return SPARK_STATUS_OK;
}

static uint64_t SparkCudaResidentdMonotonicTimeNs(void)
{
	struct timespec timestamp;
	if (clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0)
		return 0u;
	return (uint64_t)timestamp.tv_sec * 1000000000ull +
		(uint64_t)timestamp.tv_nsec;
}

static uint32_t SparkCudaResidentdWorkFailureIsNonfatal(
    SparkStatus status)
{
    return status == SPARK_STATUS_INVALID_ARGUMENT ||
        status == SPARK_STATUS_CAPACITY_EXCEEDED ||
        status == SPARK_STATUS_NOT_FOUND ||
        status == SPARK_STATUS_VALIDATION_FAILED ||
        status == SPARK_STATUS_ABI_MISMATCH;
}

static void SparkCudaResidentdCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkCudaResidentdRuntime *runtime;
    SparkCudaResidentIpcCompletion message;
    SparkStatus status;

    runtime = (SparkCudaResidentdRuntime *)completion_context;
    if (runtime == 0 || completion == 0)
        return;
    if (runtime->packet_timing_enabled != 0u &&
        runtime->inflight_submit_time_ns != 0u)
    {
        uint64_t completion_time_ns;

        completion_time_ns = SparkCudaResidentdMonotonicTimeNs();
        if (completion_time_ns != 0u)
            fprintf(stderr,
                "cuda_residentd_packet_timing rank=%u phase=execute "
                "request=%llu sequence=%llu position=%llu status=%u ns=%llu\n",
                runtime->rank_plan.rank_index,
                (unsigned long long)completion->request_id,
                (unsigned long long)completion->sequence_id,
                (unsigned long long)completion->sequence_position,
                (uint32_t)completion->status,
                (unsigned long long)(
                    completion_time_ns - runtime->inflight_submit_time_ns));
        runtime->inflight_submit_time_ns = 0u;
    }
    if (runtime->driver_inflight_count != 0u)
        runtime->driver_inflight_count -= 1u;
    runtime->completion_count += 1u;
	if (completion->status != SPARK_STATUS_OK)
	{
		if (runtime->synthetic_failure_completion_active == 0u)
			runtime->work_queue_error_count += 1u;
		if (runtime->synthetic_failure_completion_active == 0u &&
			SparkCudaResidentdWorkFailureIsNonfatal(
				completion->status) == 0u)
		{
			runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_FAILED;
			runtime->deferred_failure_status = completion->status;
			snprintf(
				runtime->blocker,
				sizeof(runtime->blocker),
				"work_completion_failed status=%u request=%llu sequence=%llu",
				(uint32_t)completion->status,
				(unsigned long long)completion->request_id,
				(unsigned long long)completion->sequence_id);
		}
    }
    memset(&message, 0, sizeof(message));
    message.descriptor_bytes = SPARK_CUDA_RESIDENT_IPC_COMPLETION_BYTES;
    message.completion = *completion;
    if (runtime->builder_library.builder_interface.take_dspark_draft != 0 &&
        runtime->builder_library.builder_interface.take_dspark_draft(
            runtime->builder_state,
            &message.dspark_draft) == SPARK_STATUS_OK)
        message.flags |=
            SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT;
    status = SparkCudaResidentdQueueCompletion(runtime,&message);
    if (status != SPARK_STATUS_OK)
        runtime->control_error_count += 1u;
}

static SparkStatus SparkCudaResidentdOpenListenSocket(
    SparkCudaResidentdRuntime *runtime,
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
    if (SparkNetSetNonblocking(fd) != 0)
    {
        close(fd);
        unlink(socket_path);
        return SPARK_STATUS_IO_ERROR;
    }
    runtime->listen_fd = fd;
    chmod(socket_path, 0600);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdLoadTransport(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration)
{
    char rank_buffer[16];
    char port_buffer[16];

    snprintf(rank_buffer, sizeof(rank_buffer), "%u", runtime->rank_plan.rank_index);
    snprintf(port_buffer, sizeof(port_buffer), "%u", configuration->port_base);
    if (setenv("SPARKPIPE_RING_TRANSPORT_RANK", rank_buffer, 1) != 0 ||
        setenv("SPARKPIPE_RING_TRANSPORT_PORT_BASE", port_buffer, 1) != 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SparkHiddenTransportLoadInterfaceFromSharedObject(
        configuration->transport_shared_object_path,
        SPARK_CUDA_RESIDENTD_REQUIRED_TRANSPORT_CAPS,
        &runtime->transport_library);
}

static SparkStatus SparkCudaResidentdOpenHiddenTransport(
    SparkCudaResidentdRuntime *runtime)
{
    SparkStatus status;

    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
    {
        status = SparkHiddenTransportOpen(
            &runtime->rank_plan.input_endpoint,
            &runtime->transport_library.transport_interface,
            SPARK_CUDA_RESIDENTD_REQUIRED_TRANSPORT_CAPS,
            &runtime->input_transport_session);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
    {
        status = SparkHiddenTransportOpen(
            &runtime->rank_plan.output_endpoint,
            &runtime->transport_library.transport_interface,
            SPARK_CUDA_RESIDENTD_REQUIRED_TRANSPORT_CAPS,
            &runtime->output_transport_session);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static int16_t SparkCudaResidentdTransportPollEvents(
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

static SparkStatus SparkCudaResidentdAppendTransportPollFds(
    SparkHiddenTransportSession *session,
    struct pollfd *fds,
    uint32_t fd_capacity,
    uint32_t *fd_count)
{
    SparkHiddenTransportPollDescriptor descriptors[
        SPARK_CUDA_RESIDENTD_TRANSPORT_POLL_CAPACITY];
    SparkStatus status;
    uint32_t descriptor_count;
    uint32_t descriptor_index;

    if (session == 0)
        return SPARK_STATUS_OK;
    descriptor_count = 0u;
    status = SparkHiddenTransportGetPollDescriptors(
        session,descriptors,
        SPARK_CUDA_RESIDENTD_TRANSPORT_POLL_CAPACITY,
        &descriptor_count);
    if (status != SPARK_STATUS_OK || descriptor_count == 0u)
        return status == SPARK_STATUS_OK ? SPARK_STATUS_NOT_FOUND : status;
    for (descriptor_index = 0u;
         descriptor_index < descriptor_count;
         ++descriptor_index)
    {
        if (*fd_count >= fd_capacity)
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        fds[*fd_count].fd = descriptors[descriptor_index].fd;
        fds[*fd_count].events = SparkCudaResidentdTransportPollEvents(
            descriptors[descriptor_index].events);
        *fd_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdBuildPollFds(
    SparkCudaResidentdRuntime *runtime,
    struct pollfd *fds,
    uint32_t fd_capacity,
    uint32_t *fd_count_out)
{
    SparkStatus status;
    uint32_t fd_count;
    uint32_t slot;

    if (runtime == 0 || fds == 0 || fd_count_out == 0 ||
        fd_capacity < SPARK_CUDA_RESIDENTD_POLL_BASE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(fds,0,fd_capacity * sizeof(fds[0u]));
    fds[SPARK_CUDA_RESIDENTD_POLL_LISTEN].fd = runtime->listen_fd;
    fds[SPARK_CUDA_RESIDENTD_POLL_LISTEN].events = POLLIN;
    pthread_mutex_lock(&runtime->output_mutex);
    for (slot = 0u; slot < SPARK_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
    {
        fds[SPARK_CUDA_RESIDENTD_POLL_CLIENT_BASE + slot].fd =
            runtime->clients[slot].fd;
        if (runtime->clients[slot].output_queue_count <
            SPARK_CUDA_RESIDENTD_OUTPUT_CONTROL_LIMIT)
            fds[SPARK_CUDA_RESIDENTD_POLL_CLIENT_BASE + slot].events |=
                POLLIN;
        if (runtime->clients[slot].output_queue_count != 0u)
            fds[SPARK_CUDA_RESIDENTD_POLL_CLIENT_BASE + slot].events |=
                POLLOUT;
    }
    pthread_mutex_unlock(&runtime->output_mutex);
    fds[SPARK_CUDA_RESIDENTD_POLL_WAKE].fd = runtime->wake_pipe_read_fd;
    fds[SPARK_CUDA_RESIDENTD_POLL_WAKE].events = POLLIN;
    fd_count = SPARK_CUDA_RESIDENTD_POLL_BASE_COUNT;
    status = SparkCudaResidentdAppendTransportPollFds(
        runtime->input_transport_session,fds,fd_capacity,&fd_count);
    if (status == SPARK_STATUS_OK)
        status = SparkCudaResidentdAppendTransportPollFds(
            runtime->output_transport_session,fds,fd_capacity,&fd_count);
    *fd_count_out = fd_count;
    return status;
}

static SparkStatus SparkCudaResidentdProgressTransport(
    SparkHiddenTransportSession *session)
{
    SparkHiddenTransportCompletion completion;
    SparkStatus status;

    if (session == 0)
        return SPARK_STATUS_OK;
    memset(&completion,0,sizeof(completion));
    status = SparkHiddenTransportPoll(session,&completion);
    if (status != SPARK_STATUS_OK)
        return status;
    if (completion.status == SPARK_STATUS_BUSY)
        return SPARK_STATUS_OK;
    return completion.status;
}

static SparkStatus SparkCudaResidentdProgressTransports(
    SparkCudaResidentdRuntime *runtime)
{
    SparkStatus status;

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkCudaResidentdProgressTransport(
        runtime->input_transport_session);
    if (status == SPARK_STATUS_OK)
        status = SparkCudaResidentdProgressTransport(
            runtime->output_transport_session);
    return status;
}

static SparkStatus SparkCudaResidentdBuildNodeContext(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration)
{
    SparkRingNodeContextBuilderConfiguration builder_configuration;
    SparkStatus status;

    memset(&builder_configuration, 0, sizeof(builder_configuration));
    runtime->kv_prefetch_lookahead_packet_count =
        configuration->kv_store_module_path != 0
            ? configuration->kv_store_lookahead_packet_count
            : SPARK_KV_STORE_DEFAULT_LOOKAHEAD_PACKETS;
    builder_configuration.abi_version =
        SPARK_RING_NODE_CONTEXT_BUILDER_ABI_VERSION;
    builder_configuration.descriptor_bytes =
        SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES;
    builder_configuration.rank_index = runtime->rank_plan.rank_index;
    builder_configuration.max_active_sequence_count =
        configuration->max_active_sequence_count;
	builder_configuration.kv_pool_token_capacity =
		configuration->kv_pool_token_capacity;
	builder_configuration.maximum_resident_sequence_count =
		SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_RESIDENT_SEQUENCE_COUNT;
    builder_configuration.port_base = configuration->port_base;
    builder_configuration.moe_pack_root = configuration->moe_pack_root;
    builder_configuration.stagepack_root = configuration->stagepack_root;
    builder_configuration.embedding_pack_path = configuration->embedding_pack_path;
    builder_configuration.node_target = configuration->node_target;
    if (configuration->speculation_enabled != 0u)
    {
        builder_configuration.flags |=
            SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK;
        builder_configuration.dspark_manifest_path =
            configuration->dspark_manifest_path;
        builder_configuration.dspark_config_path =
            configuration->dspark_config_path;
        builder_configuration.dspark_safetensors_path =
            configuration->dspark_safetensors_path;
        builder_configuration.dspark_maximum_lane_count =
            configuration->max_active_sequence_count;
        builder_configuration.dspark_maximum_context_token_count =
            configuration->dspark_maximum_context_token_count;
    }
	if (configuration->mtp_enabled != 0u)
		builder_configuration.flags |=
			SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP;
    if (configuration->kv_nvme_path != 0)
    {
        builder_configuration.flags |=
            SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV;
        builder_configuration.kv_nvme_path = configuration->kv_nvme_path;
        builder_configuration.kv_nvme_block_capacity =
            configuration->kv_nvme_block_capacity;
        builder_configuration.kv_nvme_batch_block_count =
            configuration->kv_nvme_batch_block_count;
    }
    else if (configuration->kv_store_module_path != 0)
    {
        builder_configuration.flags |=
            SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MOONCAKE_KV;
        builder_configuration.kv_store_module_path =
            configuration->kv_store_module_path;
        builder_configuration.kv_store_service_address =
            configuration->kv_store_service_address;
        builder_configuration.kv_store_ipc_socket_path =
            configuration->kv_store_ipc_socket_path;
        builder_configuration.kv_store_block_capacity =
            configuration->kv_store_block_capacity;
        builder_configuration.kv_store_batch_block_count =
            configuration->kv_store_batch_block_count;
        builder_configuration.kv_store_worker_count =
            configuration->kv_store_worker_count;
        builder_configuration.kv_store_lookahead_packet_count =
            configuration->kv_store_lookahead_packet_count;
        builder_configuration.kv_store_model_fingerprint =
            configuration->kv_store_model_fingerprint;
        builder_configuration.kv_store_layout_fingerprint =
            configuration->kv_store_layout_fingerprint;
        builder_configuration.kv_store_client_memory_pool_bytes =
            configuration->kv_store_client_memory_pool_bytes;
        builder_configuration.kv_store_local_buffer_bytes =
            configuration->kv_store_local_buffer_bytes;
    }
    builder_configuration.rank_plan = &runtime->rank_plan;
    status = SparkRingNodeContextBuilderLoadInterfaceFromSharedObject(
        configuration->node_context_builder_shared_object_path,
        SPARK_RING_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
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
    return SparkRingNodeContextBuilderValidateResult(
        &runtime->builder_result,
        &runtime->rank_plan);
}

static SparkStatus SparkCudaResidentdCreateDriverInstance(
	SparkCudaResidentdRuntime *runtime,
	const SparkCudaResidentdConfiguration *configuration)
{
	SparkModelDriverCreateRequest create_request;
	SparkStatus status;
	SparkModelDriverInitializeCreateRequest(&create_request);
    create_request.node_id = runtime->rank_plan.host_name;
    create_request.node_target = configuration->node_target;
    create_request.node_context = runtime->builder_result.node_context;
    create_request.completion_function = SparkCudaResidentdCompletion;
    create_request.completion_context = runtime;
    create_request.wake_function = SparkCudaResidentdDriverWake;
    create_request.wake_context = runtime;
    status = runtime->loaded_driver.interface->create(
        &create_request,
        &runtime->driver_instance);
	if (status != SPARK_STATUS_OK || runtime->driver_instance == 0)
		return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdLoadDriver(
	SparkCudaResidentdRuntime *runtime,
	const SparkCudaResidentdConfiguration *configuration)
{
	char error_buffer[512];
	SparkStatus status;
	error_buffer[0] = '\0';
	status = SparkLoadModelDriver(
		configuration->driver_path,configuration->node_target,
		&runtime->loaded_driver,error_buffer,sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
	{
		snprintf(runtime->blocker,sizeof(runtime->blocker),
			"driver_load_failed status=%u %s",(uint32_t)status,error_buffer);
		return status;
	}
	runtime->program = SparkFindLoadedModelDriverProgram(
		&runtime->loaded_driver,configuration->program_name);
	if (runtime->program == 0)
		return SPARK_STATUS_NOT_FOUND;
	return SparkCudaResidentdCreateDriverInstance(runtime,configuration);
}

static SparkStatus SparkCudaResidentdAttachBuilderDriver(
    SparkCudaResidentdRuntime *runtime)
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

static SparkStatus SparkCudaResidentdInitialize(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration)
{
    SparkStatus status;
    uint32_t slot;
    memset(runtime, 0, sizeof(*runtime));
    runtime->listen_fd = -1;
    runtime->wake_pipe_read_fd = -1;
    runtime->wake_pipe_write_fd = -1;
    for (slot = 0u; slot < SPARK_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
        runtime->clients[slot].fd = -1;
    runtime->completion_client_fd = -1;
    runtime->pending_prefill_work.client_fd = -1;
    runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_LOADING;
    SparkLoadedModelDriverReset(&runtime->loaded_driver);
    pthread_mutex_init(&runtime->output_mutex, 0);
    status = SparkCudaResidentdOpenWakePipe(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkRingRuntimeBuildRankPlan(
        configuration->rank_index,
        configuration->max_active_sequence_count,
        configuration->port_base,
        configuration->model_quantization_mode,
        &runtime->rank_plan,
        runtime->blocker,
        sizeof(runtime->blocker));
    if (status != SPARK_STATUS_OK)
        return status;
    runtime->work_queue = (SparkCudaResidentdQueuedWork *)calloc(
        SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY,
        sizeof(runtime->work_queue[0u]));
    if (runtime->work_queue == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    status = SparkRingRuntimeValidateStageMoePackFiles(
        &runtime->rank_plan,
        configuration->moe_pack_root,
        runtime->blocker,
        sizeof(runtime->blocker));
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentdBuildNodeContext(runtime, configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentdLoadTransport(runtime, configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentdOpenHiddenTransport(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentdLoadDriver(runtime, configuration);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentdAttachBuilderDriver(runtime);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentdOpenListenSocket(runtime, configuration->socket_path);
    if (status != SPARK_STATUS_OK)
        return status;
    runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_READY;
    return SPARK_STATUS_OK;
}

static void SparkCudaResidentdDestroy(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration)
{
    uint32_t slot;
    if (runtime == 0)
        return;
    for (slot = 0u; slot < SPARK_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
    {
        if (runtime->clients[slot].fd >= 0)
            close(runtime->clients[slot].fd);
        free(runtime->clients[slot].payload);
        runtime->clients[slot].payload = 0;
        runtime->clients[slot].payload_capacity = 0u;
        runtime->clients[slot].output_queue_head = 0u;
        runtime->clients[slot].output_queue_count = 0u;
    }
    if (runtime->listen_fd >= 0)
        close(runtime->listen_fd);
    SparkGlm52CudaResidentdWakeWriteFd = -1;
    if (runtime->wake_pipe_read_fd >= 0)
        close(runtime->wake_pipe_read_fd);
    if (runtime->wake_pipe_write_fd >= 0)
        close(runtime->wake_pipe_write_fd);
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
    SparkRingNodeContextBuilderUnloadInterface(&runtime->builder_library);
    SparkHiddenTransportClose(runtime->input_transport_session);
    SparkHiddenTransportClose(runtime->output_transport_session);
    SparkHiddenTransportUnloadInterface(&runtime->transport_library);
    free(runtime->work_queue);
    runtime->work_queue = 0;
    runtime->work_queue_count = 0u;
    pthread_mutex_destroy(&runtime->output_mutex);
}

static uint32_t SparkCudaResidentdAcceptClient(
	SparkCudaResidentdRuntime *runtime)
{
    int32_t fd;
    uint32_t slot;
    if (runtime == 0 || runtime->listen_fd < 0)
        return 0u;
    for (slot = 0u; slot < SPARK_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
        if (runtime->clients[slot].fd < 0)
            break;
    if (slot >= SPARK_CUDA_RESIDENTD_MAX_CLIENTS)
        return 0u;
    fd = accept(runtime->listen_fd, 0, 0);
    if (fd < 0)
        return 0u;
    if (SparkNetSetNonblocking(fd) < 0)
    {
        close(fd);
        return 0u;
    }
    SparkCudaResidentdResetClientMessage(&runtime->clients[slot]);
    runtime->clients[slot].output_queue_head = 0u;
    runtime->clients[slot].output_queue_count = 0u;
    runtime->clients[slot].fd = fd;
	return 1u;
}

static SparkStatus SparkCudaResidentdControlResetFailed(
	SparkCudaResidentdRuntime *runtime,
	SparkStatus status)
{
	runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_FAILED;
	runtime->deferred_failure_status = status;
	snprintf(runtime->blocker,sizeof(runtime->blocker),
		"control_generation_reset_failed status=%u",(uint32_t)status);
	return status;
}

static SparkStatus SparkCudaResidentdResetControlRuntime(
	SparkCudaResidentdRuntime *runtime,
	const SparkCudaResidentdConfiguration *configuration,
	uint64_t control_generation)
{
	SparkStatus status;
	runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_LOADING;
	status = runtime->builder_library.builder_interface.reset_control_generation(
		runtime->builder_state,control_generation);
	if (status != SPARK_STATUS_OK)
		return SparkCudaResidentdControlResetFailed(runtime,status);
	if (runtime->loaded_driver.interface != 0 &&
		runtime->loaded_driver.interface->destroy != 0 &&
		runtime->driver_instance != 0)
		runtime->loaded_driver.interface->destroy(runtime->driver_instance);
	runtime->driver_instance = 0;
	status = SparkCudaResidentdCreateDriverInstance(
		runtime,configuration);
	if (status == SPARK_STATUS_OK)
		status = SparkCudaResidentdAttachBuilderDriver(runtime);
	if (status != SPARK_STATUS_OK)
		return SparkCudaResidentdControlResetFailed(runtime,status);
	runtime->driver_inflight_count = 0u;
	runtime->deferred_failure_status = SPARK_STATUS_OK;
	runtime->blocker[0] = '\0';
	runtime->packet_timing_enabled =
		getenv("SPARKPIPE_RING_PACKET_TIMING") != 0 ? 1u : 0u;
	runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_READY;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdAdvanceControlGeneration(
	SparkCudaResidentdRuntime *runtime,
	const SparkCudaResidentdConfiguration *configuration,
	uint64_t control_generation)
{
    uint64_t previous_generation;
    uint32_t dropped_work;
	if (runtime == 0 || configuration == 0 || control_generation == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    previous_generation = runtime->active_control_generation;
    if (control_generation < previous_generation)
        return SPARK_STATUS_VALIDATION_FAILED;
    if (control_generation == previous_generation)
        return SPARK_STATUS_OK;
    dropped_work = runtime->work_queue_count + runtime->pending_prefill_active +
        (runtime->driver_inflight_count != 0u ? 1u : 0u);
	runtime->work_queue_head = 0u;
    runtime->work_queue_count = 0u;
    runtime->pending_prefill_active = 0u;
    runtime->pending_prefill_work.client_fd = -1;
	if (previous_generation != 0u)
	{
		SparkStatus status;
		status = SparkCudaResidentdResetControlRuntime(
			runtime,configuration,control_generation);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	runtime->active_control_generation = control_generation;
    runtime->work_queue_error_count += dropped_work;
    fprintf(stderr,
        "cuda_residentd_control_generation rank=%u old=%llu new=%llu dropped=%u\n",
        runtime->rank_plan.rank_index,
        (unsigned long long)previous_generation,
        (unsigned long long)control_generation,
        dropped_work);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdFillStats(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration,
    SparkCudaResidentIpcStats *stats)
{
    SparkModelDriverRuntimeSnapshot snapshot;
    SparkRingNodeContextBuilderKvStats kv_stats;
    SparkStatus status;

    memset(stats, 0, sizeof(*stats));
    stats->descriptor_bytes = SPARK_CUDA_RESIDENT_IPC_STATS_BYTES;
    stats->state = runtime->state;
    stats->capability_flags =
        SPARK_CUDA_RESIDENT_IPC_FLAG_DRIVER_RESIDENT |
        SPARK_CUDA_RESIDENT_IPC_FLAG_BUILDER_RESIDENT |
        SPARK_CUDA_RESIDENT_IPC_FLAG_TRANSPORT_RESIDENT |
        SPARK_CUDA_RESIDENT_IPC_FLAG_CUDA_STATE_RESIDENT;
    stats->rank_index = runtime->rank_plan.rank_index;
    stats->max_active_sequence_count = configuration->max_active_sequence_count;
    stats->submitted_count = runtime->submitted_count;
    stats->completed_count = runtime->completion_count;
    stats->rejected_count = runtime->submit_failed_count;
    stats->work_queue_depth = runtime->work_queue_count;
    stats->work_queue_capacity = SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY;
    stats->resident_driver_inflight = runtime->driver_inflight_count;
    stats->work_queue_accepted_count = runtime->work_queue_accepted_count;
    stats->work_queue_submit_count = runtime->work_queue_submit_count;
    stats->work_queue_error_count = runtime->work_queue_error_count;
    stats->cuda_generation = configuration->cuda_generation;
    stats->control_generation = runtime->active_control_generation != 0u
        ? runtime->active_control_generation : configuration->control_generation;
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
    if (runtime->builder_library.builder_interface.get_kv_stats != 0 &&
        runtime->builder_state != 0)
    {
        memset(&kv_stats,0,sizeof(kv_stats));
        status = runtime->builder_library.builder_interface.get_kv_stats(
            runtime->builder_state,&kv_stats);
        if (status == SPARK_STATUS_OK &&
            kv_stats.abi_version ==
                SPARK_RING_NODE_CONTEXT_BUILDER_ABI_VERSION &&
            kv_stats.descriptor_bytes ==
                SPARK_RING_NODE_CONTEXT_BUILDER_KV_STATS_BYTES)
        {
            stats->kv_nvme_enabled = kv_stats.nvme_enabled;
            stats->kv_nvme_mode = kv_stats.nvme_mode;
            stats->kv_physical_block_capacity =
                kv_stats.physical_block_capacity;
            stats->kv_logical_block_capacity =
                kv_stats.logical_block_capacity;
            stats->kv_logical_block_count = kv_stats.logical_block_count;
            stats->kv_resident_block_count = kv_stats.resident_block_count;
            stats->kv_swapped_block_count = kv_stats.swapped_block_count;
            stats->kv_nvme_record_bytes = kv_stats.nvme_record_bytes;
            stats->kv_nvme_store_count = kv_stats.nvme_store_count;
            stats->kv_nvme_load_count = kv_stats.nvme_load_count;
            stats->kv_nvme_write_bytes = kv_stats.nvme_write_bytes;
            stats->kv_nvme_read_bytes = kv_stats.nvme_read_bytes;
            stats->kv_nvme_synchronous_wait_count =
                kv_stats.nvme_synchronous_wait_count;
            stats->kv_nvme_batch_flush_count =
                kv_stats.nvme_batch_flush_count;
            stats->kv_nvme_maximum_batch_operation_count =
                kv_stats.nvme_maximum_batch_operation_count;
            stats->kv_resident_bytes_per_token =
                kv_stats.resident_bytes_per_token;
            stats->kv_resident_pool_bytes = kv_stats.resident_pool_bytes;
            stats->kv_nvme_capacity_bytes = kv_stats.nvme_capacity_bytes;
            stats->kv_compact_selected_mla_working_set_bytes =
                kv_stats.compact_selected_mla_working_set_bytes;
            stats->kv_nvme_batch_block_capacity =
                kv_stats.nvme_batch_block_capacity;
            stats->kv_nvme_pending_store_count =
                kv_stats.nvme_pending_store_count;
            stats->kv_nvme_pending_load_count =
                kv_stats.nvme_pending_load_count;
            stats->kv_nvme_clean_evict_count =
                kv_stats.nvme_clean_evict_count;
            stats->builder_pending_work = kv_stats.pending_work_active;
            stats->asynchronous_submit_count =
                kv_stats.asynchronous_submit_count;
            stats->asynchronous_completion_count =
                kv_stats.asynchronous_completion_count;
            stats->asynchronous_failure_count =
                kv_stats.asynchronous_failure_count;
            stats->logical_lane_capacity = kv_stats.logical_lane_capacity;
            stats->execution_row_capacity = kv_stats.execution_row_capacity;
            stats->last_layer_major_logical_lane_count =
                kv_stats.last_layer_major_logical_lane_count;
            stats->last_layer_major_rows_per_lane =
                kv_stats.last_layer_major_rows_per_lane;
            stats->last_layer_major_execution_row_count =
                kv_stats.last_layer_major_execution_row_count;
            stats->moe_backend_kind = kv_stats.moe_backend_kind;
            stats->moe_bound_layer_count =
                kv_stats.moe_bound_layer_count;
            stats->moe_expected_layer_count =
                kv_stats.moe_expected_layer_count;
            stats->fp8_scaled_gemm_bound_plan_count =
                kv_stats.fp8_scaled_gemm_bound_plan_count;
            stats->fp8_scaled_gemm_expected_plan_count =
                kv_stats.fp8_scaled_gemm_expected_plan_count;
            stats->model_quantization_mode =
                kv_stats.model_quantization_mode;
            stats->layer_major_submit_count =
                kv_stats.layer_major_submit_count;
            stats->layer_major_completion_count =
                kv_stats.layer_major_completion_count;
            stats->layer_major_failure_count =
                kv_stats.layer_major_failure_count;
            stats->cuda_total_bytes = kv_stats.cuda_total_bytes;
            stats->cuda_initial_free_bytes = kv_stats.cuda_initial_free_bytes;
            stats->cuda_current_free_bytes = kv_stats.cuda_current_free_bytes;
            stats->cuda_consumed_bytes = kv_stats.cuda_consumed_bytes;
            stats->cuda_builder_allocation_bytes =
                kv_stats.cuda_builder_allocation_bytes;
            stats->cuda_largest_allocation_bytes =
                kv_stats.cuda_largest_allocation_bytes;
            stats->host_mapped_allocation_bytes =
                kv_stats.host_mapped_allocation_bytes;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCudaResidentdHandleHello(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration,
    const SparkCudaResidentIpcHello *hello,
    uint32_t payload_bytes,
    int32_t client_fd)
{
    SparkCudaResidentIpcStats stats;
    SparkStatus status;
    if (runtime == 0 || configuration == 0 || hello == 0 ||
        payload_bytes != SPARK_CUDA_RESIDENT_IPC_HELLO_BYTES ||
        hello->descriptor_bytes != SPARK_CUDA_RESIDENT_IPC_HELLO_BYTES ||
        hello->rank_index != runtime->rank_plan.rank_index ||
        hello->rank_count != SPARK_RING_RUNTIME_STAGE_COUNT)
        return SPARK_STATUS_ABI_MISMATCH;
    status = SPARK_STATUS_OK;
	if (hello->control_generation != 0u)
		status = SparkCudaResidentdAdvanceControlGeneration(
			runtime,configuration,hello->control_generation);
    if (status != SPARK_STATUS_OK)
        return status;
    SparkCudaResidentdFillStats(runtime,configuration,&stats);
    status = SparkCudaResidentdQueueMessage(
        runtime,client_fd,SPARK_CUDA_RESIDENT_IPC_KIND_HELLO_ACK,
        &stats,sizeof(stats));
    return status;
}

static void SparkCudaResidentdWriteSubmitResult(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration,
    int32_t client_fd,
    SparkStatus status,
    const char *failure_text)
{
    SparkCudaResidentIpcSubmitResult result;
    memset(&result, 0, sizeof(result));
    result.descriptor_bytes = SPARK_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
    result.status = (uint32_t)status;
    SparkCudaResidentdFillStats(runtime, configuration, &result.stats);
    result.stats.rejected_count = runtime->submit_failed_count;
    if (status != SPARK_STATUS_OK)
    {
        snprintf(result.stats.blocker, sizeof(result.stats.blocker),
            "%s status=%u", failure_text, (uint32_t)status);
        fprintf(stderr, "cuda_residentd_submit_result_error rank=%u %s status=%u\n",
            runtime->rank_plan.rank_index, failure_text, (uint32_t)status);
    }
    status = SparkCudaResidentdQueueMessage(
        runtime,
        client_fd,
        SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT,
        &result,
        sizeof(result));
    if (status != SPARK_STATUS_OK)
        runtime->control_error_count += 1u;
}

static SparkStatus SparkCudaResidentdEnqueueWork(
	SparkCudaResidentdRuntime *runtime,
	const SparkCudaResidentdConfiguration *configuration,
	const SparkRingWorkControlPacket *packet,
	int32_t client_fd)
{
    SparkCudaResidentdQueuedWork *queued_work;
    uint32_t tail;
    SparkStatus status;

	if (runtime == 0 || configuration == 0 || packet == 0 || client_fd < 0 ||
        runtime->work_queue == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (runtime->state != SPARK_CUDA_RESIDENT_IPC_STATE_READY)
        return runtime->deferred_failure_status != SPARK_STATUS_OK
            ? runtime->deferred_failure_status
            : SPARK_STATUS_MODULE_NOT_VALIDATED;
    status = SparkRingWorkControlValidatePacket(
        packet,runtime->rank_plan.execution_row_capacity,1u);
    if (status != SPARK_STATUS_OK)
        return status;
    if (packet->control_generation != 0u)
	{
		status = SparkCudaResidentdAdvanceControlGeneration(
			runtime,configuration,packet->control_generation);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    if (runtime->work_queue_count >=
        SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY)
        return SPARK_STATUS_BUSY;
    tail = (runtime->work_queue_head + runtime->work_queue_count) %
        SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY;
    queued_work = &runtime->work_queue[tail];
    queued_work->packet = *packet;
    queued_work->client_fd = client_fd;
    queued_work->reserved0 = 0u;
    queued_work->enqueue_time_ns = runtime->packet_timing_enabled != 0u
        ? SparkCudaResidentdMonotonicTimeNs() : 0u;
    runtime->work_queue_count += 1u;
    runtime->work_queue_accepted_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkCudaResidentdPopQueuedWork(
    SparkCudaResidentdRuntime *runtime)
{
    if (runtime == 0 || runtime->work_queue_count == 0u)
        return;
    runtime->work_queue_head = (runtime->work_queue_head + 1u) %
        SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY;
    runtime->work_queue_count -= 1u;
}

static void SparkCudaResidentdEmitWorkFailure(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdQueuedWork *queued_work,
    SparkStatus failure_status)
{
    SparkModelDriverCompletion completion;
    uint32_t completion_count;
    uint32_t lane_index;

    if (runtime == 0 || queued_work == 0 ||
        (queued_work->packet.flags &
            SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
        return;
    memset(&completion,0,sizeof(completion));
    completion.status = failure_status;
    completion_count = (runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u
        ? queued_work->packet.lane_count : 1u;
    runtime->completion_client_fd = queued_work->client_fd;
	runtime->synthetic_failure_completion_active = 1u;
    for (lane_index = 0u; lane_index < completion_count; ++lane_index)
    {
        completion.request_id = queued_work->packet.lanes[lane_index].request_id;
        completion.sequence_id = queued_work->packet.lanes[lane_index].sequence_id;
        completion.sequence_position =
            queued_work->packet.lanes[lane_index].sequence_position;
        SparkCudaResidentdCompletion(runtime,&completion);
    }
	runtime->synthetic_failure_completion_active = 0u;
}

static uint32_t SparkCudaResidentdPumpQueuedWork(
    SparkCudaResidentdRuntime *runtime)
{
    SparkCudaResidentdQueuedWork queued_work;
    SparkRingWorkControlPacket prefetch_packets[
        SPARK_KV_STORE_MAX_LOOKAHEAD_PACKETS];
    SparkStatus status;
    uint32_t prefetch_packet_count;
    uint32_t prefetch_packet_index;

    if (runtime == 0 || runtime->work_queue_count == 0u ||
        runtime->driver_inflight_count != 0u)
        return 0u;
    queued_work = runtime->work_queue[runtime->work_queue_head];
    if (runtime->state == SPARK_CUDA_RESIDENT_IPC_STATE_FAILED)
    {
        SparkCudaResidentdPopQueuedWork(runtime);
        runtime->work_queue_error_count += 1u;
        SparkCudaResidentdEmitWorkFailure(
            runtime,&queued_work,
            runtime->deferred_failure_status != SPARK_STATUS_OK
                ? runtime->deferred_failure_status
                : SPARK_STATUS_MODULE_NOT_VALIDATED);
        return 1u;
    }
    prefetch_packet_count = SparkKvStoreNormalizeLookaheadPacketCount(
        runtime->kv_prefetch_lookahead_packet_count,
        runtime->work_queue_count);
    for (prefetch_packet_index = 0u;
         prefetch_packet_index < prefetch_packet_count;
         ++prefetch_packet_index)
    {
        uint32_t queue_index;
        queue_index = (runtime->work_queue_head + prefetch_packet_index) %
            SPARK_CUDA_RESIDENTD_WORK_QUEUE_CAPACITY;
        prefetch_packets[prefetch_packet_index] =
            runtime->work_queue[queue_index].packet;
    }
    status = runtime->builder_library.builder_interface.prefetch_work(
        runtime->builder_state,prefetch_packets,prefetch_packet_count);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
    {
        runtime->deferred_failure_status = status;
        runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_FAILED;
        return 1u;
    }
    runtime->completion_client_fd = queued_work.client_fd;
    runtime->driver_inflight_count += 1u;
    if (runtime->packet_timing_enabled != 0u)
    {
        uint64_t submit_time_ns;

        submit_time_ns = SparkCudaResidentdMonotonicTimeNs();
        runtime->inflight_submit_time_ns = submit_time_ns;
        if (submit_time_ns != 0u && queued_work.enqueue_time_ns != 0u)
            fprintf(stderr,
                "cuda_residentd_packet_timing rank=%u phase=queue_wait "
                "request=%llu sequence=%llu position=%llu rows=%u "
                "flags=0x%x ns=%llu\n",
                runtime->rank_plan.rank_index,
                (unsigned long long)queued_work.packet.request_id,
                (unsigned long long)queued_work.packet.sequence_id,
                (unsigned long long)queued_work.packet.sequence_position,
                queued_work.packet.execution_row_count,
                queued_work.packet.flags,
                (unsigned long long)(
                    submit_time_ns - queued_work.enqueue_time_ns));
    }
    status = runtime->builder_library.builder_interface.submit_work(
        runtime->builder_state,&queued_work.packet,
        runtime->input_transport_session,runtime->output_transport_session,
        SparkCudaResidentdCompletion,runtime);
    if (status == SPARK_STATUS_BUSY)
    {
        if (runtime->driver_inflight_count != 0u)
            runtime->driver_inflight_count -= 1u;
        runtime->inflight_submit_time_ns = 0u;
        return 0u;
    }
    SparkCudaResidentdPopQueuedWork(runtime);
    if (status == SPARK_STATUS_OK)
    {
        if (queued_work.packet.control_generation >
            runtime->active_control_generation)
            runtime->active_control_generation =
                queued_work.packet.control_generation;
        runtime->submitted_count += 1u;
        runtime->work_queue_submit_count += 1u;
        if ((queued_work.packet.flags &
                SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u &&
            runtime->driver_inflight_count != 0u)
		{
            runtime->driver_inflight_count -= 1u;
			runtime->inflight_submit_time_ns = 0u;
		}
    }
    else
    {
        if (runtime->driver_inflight_count != 0u)
            runtime->driver_inflight_count -= 1u;
        runtime->submit_failed_count += 1u;
        runtime->work_queue_error_count += 1u;
		if ((queued_work.packet.flags &
				SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
			runtime->inflight_submit_time_ns = 0u;
        if (SparkCudaResidentdWorkFailureIsNonfatal(status) == 0u)
		{
			runtime->state = SPARK_CUDA_RESIDENT_IPC_STATE_FAILED;
			runtime->deferred_failure_status = status;
			snprintf(
				runtime->blocker,
				sizeof(runtime->blocker),
				"deferred_submit_failed status=%u request=%llu sequence=%llu",
				(uint32_t)status,
				(unsigned long long)queued_work.packet.request_id,
				(unsigned long long)queued_work.packet.sequence_id);
		}
		else
		{
			fprintf(stderr,
				"cuda_residentd_work_rejected status=%u rank=%u request=%llu "
				"sequence=%llu position=%llu lanes=%u rows=%u bucket=%u flags=0x%x\n",
				(uint32_t)status,
				runtime->rank_plan.rank_index,
				(unsigned long long)queued_work.packet.request_id,
				(unsigned long long)queued_work.packet.sequence_id,
				(unsigned long long)queued_work.packet.sequence_position,
				queued_work.packet.lane_count,
				queued_work.packet.execution_row_count,
				queued_work.packet.execution_batch_bucket,
				queued_work.packet.flags);
		}
        SparkCudaResidentdEmitWorkFailure(
            runtime,&queued_work,status);
    }
    return 1u;
}

static SparkStatus SparkCudaResidentdHandleSubmitWork(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration,
    const SparkCudaResidentIpcSubmitWork *message,
    uint32_t payload_bytes,
    int32_t client_fd)
{
    uint32_t result_requested;
    SparkStatus status;
    result_requested = 0u;
    status = SparkCudaResidentIpcValidateSubmitWork(
        message,payload_bytes);
	if (status == SPARK_STATUS_OK)
    {
        result_requested = (message->flags &
            SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT) != 0u;
        status = SparkCudaResidentdEnqueueWork(
            runtime,configuration,&message->work_packet,client_fd);
    }
    if (status != SPARK_STATUS_OK)
        runtime->submit_failed_count += 1u;
    if (status != SPARK_STATUS_OK || result_requested != 0u)
        SparkCudaResidentdWriteSubmitResult(
            runtime,configuration,client_fd,status,
            "submit_work_enqueue_failed");
    return status;
}

static SparkStatus SparkCudaResidentdBuildDecodeWorkPacket(
    const SparkCudaResidentIpcSubmitDecode *message,
    uint32_t execution_row_capacity,
    SparkRingWorkControlPacket *packet)
{
    uint32_t dspark_verify;
    uint32_t mtp_verify;
    uint32_t mtp_tree_verify;
    uint32_t speculative_verify;
    uint32_t mtp_budget;
    uint32_t maximum_context_token_count;
    uint32_t lane_index;
    uint64_t execution_row_count;
    SparkStatus status;

    if (message == 0 || packet == 0 ||
        (message->resident_flags &
            SPARK_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY) == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    dspark_verify = (message->request_flags &
        SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u;
    mtp_verify = (message->request_flags &
        SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
    mtp_tree_verify = (message->request_flags &
        SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u;
    speculative_verify = dspark_verify | mtp_verify;
    if ((message->dispatch_kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
         speculative_verify == 0u) ||
        (message->dispatch_kind ==
            SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
         speculative_verify != 0u) ||
        (dspark_verify != 0u && mtp_verify != 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkRingWorkControlSelectMtpDraftBudget(
            message->dispatch_kind,
            message->request_flags,
            message->lanes[0u].mtp_draft_token_budget,
            &mtp_budget);
    if (status != SPARK_STATUS_OK)
        return status;
    memset(packet, 0, sizeof(*packet));
    packet->magic = SPARK_RING_WORK_CONTROL_PACKET_MAGIC;
    packet->abi_version = SPARK_RING_WORK_CONTROL_ABI_VERSION;
    packet->control_generation = message->control_generation;
    packet->request_generation = message->request_generation;
    packet->step_generation = message->step_generation;
    packet->step_chunk_index = message->step_chunk_index;
    packet->step_chunk_count = message->step_chunk_count;
    packet->transaction_phase = message->transaction_phase;
    packet->reserved_transaction = message->reserved_transaction;
    packet->descriptor_bytes =
        SparkRingWorkControlCalculatePacketBytes(message->lane_count);
    if (packet->descriptor_bytes == 0u)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    if (mtp_budget != 0u)
        packet->flags |= SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT;
    if ((message->request_flags &
            SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
        packet->flags |=
            SPARK_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
    if (dspark_verify != 0u)
        packet->flags |=
            SPARK_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
    if (mtp_verify != 0u)
        packet->flags |=
            SPARK_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
    if (mtp_tree_verify != 0u)
        packet->flags |=
            SPARK_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY;
    packet->request_id = message->lanes[0u].request_id;
    packet->sequence_id = message->lanes[0u].sequence_id;
    packet->sequence_position = message->lanes[0u].sequence_position;
    packet->active_sequence_count = message->lane_count;
    packet->lane_count = message->lane_count;
    packet->rows_per_lane = speculative_verify != 0u
        ? mtp_tree_verify != 0u
            ? SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT
            : message->speculative_token_count + 1u
        : 1u;
    if (mtp_tree_verify != 0u &&
        (mtp_verify == 0u ||
         message->speculative_token_count !=
            SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT))
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    execution_row_count =
        (uint64_t)packet->lane_count * packet->rows_per_lane;
    if (execution_row_count == 0u ||
        execution_row_count > execution_row_capacity)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    packet->execution_row_count = (uint32_t)execution_row_count;
    packet->execution_batch_bucket = message->execution_batch_bucket;
    packet->new_token_count = speculative_verify != 0u
        ? packet->rows_per_lane : mtp_budget + 1u;
    packet->priority = message->highest_priority;
    packet->block_token_count = message->kv_block_token_count;
    packet->max_blocks_per_sequence =
        SPARK_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS;
    packet->mtp_draft_token_count = mtp_budget;
    packet->input_token_id = message->lanes[0u].input_token_id;
    maximum_context_token_count = 0u;
    for (lane_index = 0u; lane_index < message->lane_count; ++lane_index)
    {
        const SparkCudaResidentIpcDecodeLane *source_lane;
        SparkRingWorkControlLane *target_lane;
        uint32_t required_context_token_count;
        source_lane = &message->lanes[lane_index];
        target_lane = &packet->lanes[lane_index];
        if (source_lane->mtp_draft_token_budget !=
                message->lanes[0u].mtp_draft_token_budget)
            return SPARK_STATUS_INVALID_ARGUMENT;
        required_context_token_count = source_lane->context_token_count;
        if (speculative_verify != 0u || mtp_budget != 0u)
        {
            uint32_t additional_token_count;
            additional_token_count = speculative_verify != 0u
                ? mtp_tree_verify != 0u
                    ? SPARK_MODEL_MTP_TREE_CONTEXT_EXTENSION
                    : message->speculative_token_count
                : mtp_budget;
            if (required_context_token_count >
                UINT32_MAX - additional_token_count)
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            required_context_token_count += additional_token_count;
        }
        target_lane->request_id = source_lane->request_id;
        target_lane->request_generation = source_lane->request_generation;
        target_lane->sequence_id = source_lane->sequence_id;
        target_lane->sequence_position = source_lane->sequence_position;
        target_lane->request_slot_index = source_lane->request_slot_index;
        target_lane->context_token_count = required_context_token_count;
        target_lane->input_token_id = source_lane->input_token_id;
        target_lane->mtp_draft_token_count = mtp_budget;
        target_lane->mtp_resolution_proposed_token_count =
            source_lane->mtp_resolution_proposed_token_count;
        target_lane->mtp_resolution_accepted_token_count =
            source_lane->mtp_resolution_accepted_token_count;
        target_lane->mtp_resolution_path_id =
            source_lane->mtp_resolution_path_id;
        if (target_lane->mtp_resolution_proposed_token_count != 0u)
            packet->flags |= SPARK_RING_WORK_CONTROL_FLAG_MTP_RESOLVE;
        if (speculative_verify != 0u)
        {
            target_lane->speculative_token_count =
                message->speculative_token_count;
            memcpy(target_lane->speculative_draft_token_ids,
                source_lane->speculative_draft_token_ids,
                sizeof(target_lane->speculative_draft_token_ids));
        }
        if (required_context_token_count > maximum_context_token_count)
            maximum_context_token_count = required_context_token_count;
    }
    packet->kv_block_table_token_count = maximum_context_token_count;
    if (speculative_verify != 0u)
    {
        packet->speculative_token_count = message->speculative_token_count;
        memcpy(packet->speculative_draft_token_ids,
            message->lanes[0u].speculative_draft_token_ids,
            sizeof(packet->speculative_draft_token_ids));
    }
    return SparkRingWorkControlValidatePacket(
        packet,execution_row_capacity,1u);
}

static SparkStatus SparkCudaResidentdHandleSubmitPrefill(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration,
    const SparkCudaResidentIpcSubmitPrefill *message,
    int32_t client_fd)
{
    const SparkRingWorkControlPacket *packet;
    SparkStatus status;
    packet = 0;
    status = SPARK_STATUS_ABI_MISMATCH;
    if (message != 0 && message->descriptor_bytes ==
        SparkCudaResidentIpcCalculateSubmitPrefillBytes(
			&message->work_packet))
    {
        packet = &message->work_packet;
        status = SparkRingWorkControlValidatePacket(
            packet,runtime->rank_plan.execution_row_capacity,UINT32_MAX);
    }
	if (status == SPARK_STATUS_OK &&
		(packet->flags & SPARK_RING_WORK_CONTROL_FLAG_PREFILL) == 0u)
		status = SPARK_STATUS_INVALID_ARGUMENT;
    if (status == SPARK_STATUS_OK &&
        runtime->builder_library.builder_interface.submit_work == 0)
        status = SPARK_STATUS_MODULE_NOT_VALIDATED;
    if (status == SPARK_STATUS_OK && runtime->pending_prefill_active != 0u)
        status = SPARK_STATUS_BUSY;
    if (status == SPARK_STATUS_OK)
        status = SparkCudaResidentdEnqueueWork(
            runtime,configuration,packet,client_fd);
    if (status == SPARK_STATUS_BUSY &&
        runtime->pending_prefill_active == 0u)
    {
        runtime->pending_prefill_work.packet = *packet;
        runtime->pending_prefill_work.client_fd = client_fd;
        runtime->pending_prefill_active = 1u;
        if (getenv("SPARKPIPE_RING_TRACE") != 0)
            fprintf(stderr,
                "cuda_residentd_prefill_deferred rank=%u depth=%u request=%llu position=%llu\n",
                runtime->rank_plan.rank_index,runtime->work_queue_count,
                (unsigned long long)packet->request_id,
                (unsigned long long)packet->sequence_position);
        return SPARK_STATUS_OK;
    }
    if (status == SPARK_STATUS_OK)
        runtime->ingest_prefill_count += packet->active_sequence_count;
    else
        runtime->submit_failed_count += 1u;
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "cuda_residentd_prefill_rejected status=%u rank=%u request=%llu "
            "position=%llu lanes=%u rows=%u bucket=%u flags=0x%x\n",
            (uint32_t)status,runtime->rank_plan.rank_index,
            (unsigned long long)packet->request_id,
            (unsigned long long)packet->sequence_position,
            packet->lane_count,packet->execution_row_count,
            packet->execution_batch_bucket,packet->flags);
        SparkCudaResidentdWriteSubmitResult(
            runtime,configuration,client_fd,status,
            "ingest_prefill_work_failed");
    }
    return status;
}

static uint32_t SparkCudaResidentdPumpPendingPrefill(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration)
{
    SparkCudaResidentdQueuedWork *pending;
    SparkStatus status;
    int32_t client_fd;
    if (runtime == 0 || configuration == 0 ||
        runtime->pending_prefill_active == 0u)
        return 0u;
    pending = &runtime->pending_prefill_work;
	status = SparkCudaResidentdEnqueueWork(
		runtime,configuration,&pending->packet,pending->client_fd);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    client_fd = pending->client_fd;
    runtime->pending_prefill_active = 0u;
    pending->client_fd = -1;
    if (status == SPARK_STATUS_OK)
    {
        runtime->ingest_prefill_count +=
            pending->packet.active_sequence_count;
        if (getenv("SPARKPIPE_RING_TRACE") != 0)
            fprintf(stderr,
                "cuda_residentd_prefill_admitted rank=%u depth=%u request=%llu position=%llu\n",
                runtime->rank_plan.rank_index,runtime->work_queue_count,
                (unsigned long long)pending->packet.request_id,
                (unsigned long long)pending->packet.sequence_position);
    }
    else
        runtime->submit_failed_count += 1u;
    if (status != SPARK_STATUS_OK)
        SparkCudaResidentdWriteSubmitResult(
            runtime,configuration,client_fd,status,
            "deferred_prefill_work_failed");
    return 1u;
}

static SparkStatus SparkCudaResidentdHandleSubmitDecode(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration,
    const SparkCudaResidentIpcSubmitDecode *message,
    uint32_t payload_bytes,
    int32_t client_fd)
{
    SparkRingWorkControlPacket work_packet;
    SparkStatus status;
    if (message == 0 || payload_bytes <
        SPARK_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES)
        return SPARK_STATUS_ABI_MISMATCH;
    status = SparkCudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,runtime->rank_plan.logical_lane_capacity);
    if (status != SPARK_STATUS_OK)
    {
        SparkCudaResidentdWriteSubmitResult(runtime,configuration,
            client_fd,status,"ingest_decode_message_invalid");
        runtime->submit_failed_count += 1u;
        return status;
    }
    if ((message->resident_flags &
            SPARK_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY) == 0u ||
        runtime->builder_library.builder_interface.submit_work == 0)
    {
        status = SPARK_STATUS_MODULE_NOT_VALIDATED;
        SparkCudaResidentdWriteSubmitResult(runtime,configuration,
            client_fd,status,"ingest_decode_internal_kv_required");
        runtime->submit_failed_count += 1u;
        return status;
    }
    status = SparkCudaResidentdBuildDecodeWorkPacket(
        message,runtime->rank_plan.execution_row_capacity,&work_packet);
	if (status == SPARK_STATUS_OK)
		status = SparkCudaResidentdEnqueueWork(
			runtime,configuration,&work_packet,client_fd);
    if (status == SPARK_STATUS_OK)
    {
        runtime->ingest_decode_count += 1u;
    }
    else
    {
        runtime->submit_failed_count += 1u;
    }
    if (status != SPARK_STATUS_OK)
        SparkCudaResidentdWriteSubmitResult(
            runtime,configuration,client_fd,status,
            "ingest_decode_work_failed");
    return status;
}

static void SparkCudaResidentdDropClient(
    SparkCudaResidentdRuntime *runtime,
    uint32_t slot)
{
    int32_t fd;
    if (runtime == 0 || slot >= SPARK_CUDA_RESIDENTD_MAX_CLIENTS)
        return;
    fd = runtime->clients[slot].fd;
    if (fd < 0)
        return;
    pthread_mutex_lock(&runtime->output_mutex);
    if (runtime->completion_client_fd == fd)
        runtime->completion_client_fd = -1;
    close(fd);
    runtime->clients[slot].fd = -1;
    free(runtime->clients[slot].payload);
    runtime->clients[slot].payload = 0;
    runtime->clients[slot].payload_capacity = 0u;
    runtime->clients[slot].output_queue_head = 0u;
    runtime->clients[slot].output_queue_count = 0u;
    SparkCudaResidentdResetClientMessage(&runtime->clients[slot]);
    pthread_mutex_unlock(&runtime->output_mutex);
    if (runtime->pending_prefill_active != 0u &&
        runtime->pending_prefill_work.client_fd == fd)
    {
        runtime->pending_prefill_active = 0u;
        runtime->pending_prefill_work.client_fd = -1;
        runtime->work_queue_error_count += 1u;
    }
}

static uint32_t SparkCudaResidentdPumpClient(
    SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration,
    uint32_t slot)
{
    SparkCudaResidentdClient *client;
    uint8_t *payload;
    SparkCudaResidentIpcStats stats;
    SparkStatus status;
    int32_t fd;
    if (runtime == 0 || slot >= SPARK_CUDA_RESIDENTD_MAX_CLIENTS)
        return 0u;
    client = &runtime->clients[slot];
    fd = client->fd;
    if (fd < 0)
        return 0u;
    status = SparkCudaResidentdReadClientMessage(client);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        SparkCudaResidentdDropClient(runtime,slot);
        return 1u;
    }
    payload = client->payload;
    switch (client->reader.header.kind)
    {
        case SPARK_CUDA_RESIDENT_IPC_KIND_HELLO:
            status = SparkCudaResidentdHandleHello(
                runtime,configuration,
                (const SparkCudaResidentIpcHello *)payload,
                client->reader.header.payload_bytes,fd);
            if (status != SPARK_STATUS_OK)
            {
                runtime->control_error_count += 1u;
                SparkCudaResidentdDropClient(runtime,slot);
                return 1u;
            }
            break;
        case SPARK_CUDA_RESIDENT_IPC_KIND_QUERY:
            SparkCudaResidentdFillStats(runtime, configuration, &stats);
            (void)SparkCudaResidentdQueueMessage(
                runtime,
                fd,
                SPARK_CUDA_RESIDENT_IPC_KIND_STATS,
                &stats,
                sizeof(stats));
            break;
        case SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK:
            (void)SparkCudaResidentdHandleSubmitWork(
                runtime,
                configuration,
                (const SparkCudaResidentIpcSubmitWork *)payload,
                client->reader.header.payload_bytes,
                fd);
            break;
        case SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_PREFILL:
            if (runtime->builder_library.builder_interface.submit_work == 0)
                SparkCudaResidentdWriteSubmitResult(runtime,configuration,
                    fd,SPARK_STATUS_MODULE_NOT_VALIDATED,
                    "ingest_prefill_unavailable");
            else
                (void)SparkCudaResidentdHandleSubmitPrefill(
                    runtime,
                    configuration,
                    (const SparkCudaResidentIpcSubmitPrefill *)payload,
                    fd);
            break;
        case SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_DECODE:
            if (runtime->builder_library.builder_interface.submit_work == 0)
                SparkCudaResidentdWriteSubmitResult(runtime,configuration,
                    fd,SPARK_STATUS_MODULE_NOT_VALIDATED,
                    "ingest_decode_unavailable");
            else
                (void)SparkCudaResidentdHandleSubmitDecode(
                    runtime,
                    configuration,
                    (const SparkCudaResidentIpcSubmitDecode *)payload,
                    client->reader.header.payload_bytes,fd);
            break;
        case SPARK_CUDA_RESIDENT_IPC_KIND_SHUTDOWN:
            SparkGlm52CudaResidentdRunning = 0;
            break;
        default:
            runtime->control_error_count += 1u;
            break;
    }
    SparkCudaResidentdResetClientMessage(client);
    return 1u;
}

static void SparkCudaResidentdProgressDriver(
	SparkCudaResidentdRuntime *runtime)
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

static SparkStatus SparkCudaResidentdProgressBuilder(
	SparkCudaResidentdRuntime *runtime)
{
	if (runtime == 0 ||
		runtime->builder_library.builder_interface.progress == 0 ||
		runtime->builder_state == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	return runtime->builder_library.builder_interface.progress(
		runtime->builder_state);
}

static void SparkCudaResidentdPrintReady(
    const SparkCudaResidentdRuntime *runtime,
    const SparkCudaResidentdConfiguration *configuration)
{
    SparkRingNodeContextBuilderKvStats kv_stats;

    printf("glm52_cuda_residentd=1\n");
    printf("rank=%u\n", runtime->rank_plan.rank_index);
    printf("host=%s\n", runtime->rank_plan.host_name);
    printf("socket=%s\n", configuration->socket_path);
    printf("stage=%u:%u\n",
        runtime->rank_plan.first_layer_index,
        runtime->rank_plan.layer_count);
    printf("model_quantization=%s\n",
        SparkRingRuntimeQuantizationModeName(
            configuration->model_quantization_mode));
    printf("moe_pack_root=%s\n", configuration->moe_pack_root);
    printf("transport_so=%s\n", configuration->transport_shared_object_path);
    printf("driver_so=%s\n", configuration->driver_path);
    printf("node_context_builder_so=%s\n",
        configuration->node_context_builder_shared_object_path);
    if (configuration->kv_nvme_path != 0)
    {
        printf("kv_nvme_path=%s\n", configuration->kv_nvme_path);
        printf("kv_nvme_blocks=%u\n", configuration->kv_nvme_block_capacity);
        printf("kv_nvme_batch_blocks=%u\n",
            configuration->kv_nvme_batch_block_count);
        printf("kv_nvme_mode=batched_cohort_jit\n");
    }
    if (configuration->kv_store_module_path != 0)
    {
        printf("kv_store_backend=mooncake\n");
        printf("kv_store_module=%s\n",configuration->kv_store_module_path);
        printf("kv_store_service=%s\n",configuration->kv_store_service_address);
        printf("kv_store_blocks=%u\n",configuration->kv_store_block_capacity);
        printf("kv_store_batch_blocks=%u\n",
            configuration->kv_store_batch_block_count);
        printf("kv_store_lookahead_packets=%u\n",
            configuration->kv_store_lookahead_packet_count);
    }
    memset(&kv_stats,0,sizeof(kv_stats));
    if (runtime->builder_library.builder_interface.get_kv_stats != 0 &&
        runtime->builder_library.builder_interface.get_kv_stats(
            runtime->builder_state,&kv_stats) == SPARK_STATUS_OK)
    {
        printf("logical_lane_capacity=%u\n",kv_stats.logical_lane_capacity);
        printf("execution_row_capacity=%u\n",kv_stats.execution_row_capacity);
        printf("fp8_moe_layers=%u/%u\n",
            kv_stats.moe_bound_layer_count,
            kv_stats.moe_expected_layer_count);
        printf("fp8_scaled_gemm_plans=%u/%u\n",
            kv_stats.fp8_scaled_gemm_bound_plan_count,
            kv_stats.fp8_scaled_gemm_expected_plan_count);
        printf("kv_physical_blocks=%u\n",kv_stats.physical_block_capacity);
        printf("kv_logical_blocks=%u\n",kv_stats.logical_block_capacity);
        printf("kv_resident_pool_bytes=%llu\n",
            (unsigned long long)kv_stats.resident_pool_bytes);
        printf("kv_nvme_capacity_bytes=%llu\n",
            (unsigned long long)kv_stats.nvme_capacity_bytes);
    }
    printf("state=ready\n");
    fflush(stdout);
}

static void SparkCudaResidentdUsage(const char *program)
{
    fprintf(stderr,
        "usage: %s --rank n --socket path --model-quantization fp8|nvfp4 --moe-pack-root dir --stagepack-root dir --transport-so path --driver-so path --node-context-builder-so path --embedding-pack path [--mtp] [--dspark --dspark-manifest path --dspark-config path --dspark-safetensors path --dspark-max-context n] [--program name] [--node-target target] [--max-active n] [--kv-pool-tokens n] [--kv-nvme-path path --kv-nvme-blocks n --kv-nvme-batch-blocks n | --kv-store-module path --kv-store-service addr --kv-store-ipc-socket path --kv-store-blocks n --kv-store-batch-blocks n --kv-store-workers n --kv-store-lookahead n --kv-store-model-fingerprint n --kv-store-layout-fingerprint n --kv-store-client-memory n --kv-store-local-buffer n] [--port-base n]\n",
        program);
}

int main(int argc, char **argv)
{
    SparkCudaResidentdConfiguration configuration;
    static SparkCudaResidentdRuntime runtime;
    SparkStatus status;
    struct pollfd fds[SPARK_CUDA_RESIDENTD_POLL_CAPACITY];
	int32_t index;
	int poll_status;
	int poll_timeout_ms;
	int32_t exit_code;
	uint32_t fd_count;
	uint32_t slot;
	SparkStatus builder_status;

    SparkCudaResidentdInitializeConfiguration(&configuration);
    for (index = 1; index < argc; ++index)
    {
        if (SparkCudaResidentdApplyArgument(
                &configuration,
                argc,
                argv,
                &index) != 0)
        {
            SparkCudaResidentdUsage(argv[0]);
            return 2;
        }
    }
    status = SparkCudaResidentdValidateConfiguration(&configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkCudaResidentdUsage(argv[0]);
        return 2;
    }
    signal(SIGINT, SparkCudaResidentdSignal);
    signal(SIGTERM, SparkCudaResidentdSignal);
    signal(SIGPIPE, SIG_IGN);
    status = SparkCudaResidentdInitialize(&runtime, &configuration);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "glm52_cuda_residentd_init_failed status=%u blocker=%s\n",
            (uint32_t)status,
            runtime.blocker);
        SparkCudaResidentdDestroy(&runtime, &configuration);
        return 1;
    }
    SparkCudaResidentdPrintReady(&runtime, &configuration);
    exit_code = 0;
    while (SparkGlm52CudaResidentdRunning)
    {
        status = SparkCudaResidentdProgressTransports(&runtime);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"cuda_residentd_transport_progress_failed status=%u\n",
                (uint32_t)status);
            exit_code = 1;
            break;
	    }
	    SparkCudaResidentdProgressDriver(&runtime);
		builder_status = SparkCudaResidentdProgressBuilder(&runtime);
		if (builder_status != SPARK_STATUS_OK &&
			builder_status != SPARK_STATUS_BUSY)
		{
			if (SparkCudaResidentdWorkFailureIsNonfatal(
					builder_status) != 0u)
				fprintf(stderr,
					"cuda_residentd_builder_work_failed status=%u\n",
					(uint32_t)builder_status);
			else
			{
				fprintf(stderr,
					"cuda_residentd_builder_progress_failed status=%u\n",
					(uint32_t)builder_status);
				exit_code = 1;
				break;
			}
		}
	    while (SparkCudaResidentdPumpQueuedWork(&runtime) != 0u &&
            runtime.driver_inflight_count == 0u)
        {
        }
        if (SparkCudaResidentdPumpPendingPrefill(
                &runtime,&configuration) != 0u)
            while (SparkCudaResidentdPumpQueuedWork(&runtime) != 0u &&
                runtime.driver_inflight_count == 0u)
            {
            }
        status = SparkCudaResidentdBuildPollFds(
            &runtime,fds,SPARK_CUDA_RESIDENTD_POLL_CAPACITY,&fd_count);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"cuda_residentd_poll_build_failed status=%u\n",
                (uint32_t)status);
            exit_code = 1;
	        break;
	    }
		poll_timeout_ms = runtime.driver_inflight_count != 0u ||
			builder_status == SPARK_STATUS_BUSY ? 1 : -1;
	    poll_status = poll(fds,fd_count,poll_timeout_ms);
        if (poll_status < 0 && errno != EINTR)
        {
            fprintf(stderr,"cuda_residentd_poll_failed errno=%d\n",errno);
            exit_code = 1;
            break;
        }
        if (poll_status > 0)
        {
            if ((fds[SPARK_CUDA_RESIDENTD_POLL_WAKE].revents & POLLIN) != 0)
                SparkCudaResidentdDrainWakePipe(&runtime);
            if ((fds[SPARK_CUDA_RESIDENTD_POLL_LISTEN].revents & POLLIN) != 0)
                (void)SparkCudaResidentdAcceptClient(&runtime);
            for (slot = 0u; slot < SPARK_CUDA_RESIDENTD_MAX_CLIENTS; ++slot)
            {
                short revents;
                revents =
                    fds[SPARK_CUDA_RESIDENTD_POLL_CLIENT_BASE + slot].revents;
                if (runtime.clients[slot].fd >= 0 &&
                    (revents & POLLOUT) != 0 &&
                    SparkCudaResidentdFlushClientOutput(
                        &runtime,slot) == SPARK_STATUS_IO_ERROR)
                    SparkCudaResidentdDropClient(&runtime,slot);
                if (runtime.clients[slot].fd >= 0 &&
                    (revents & POLLIN) != 0)
                    (void)SparkCudaResidentdPumpClient(
                        &runtime,&configuration,slot);
                if (runtime.clients[slot].fd >= 0 &&
                    (revents & (POLLHUP | POLLERR)) != 0)
                    SparkCudaResidentdDropClient(&runtime,slot);
            }
        }
    }
    SparkCudaResidentdDestroy(&runtime, &configuration);
    return exit_code;
}
