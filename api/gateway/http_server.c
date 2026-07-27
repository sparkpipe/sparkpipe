#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_service_backend.h"
#include "sparkpipe/spark_glm52_http_gateway.h"
#include "sparkpipe/spark_glm52_ring_runtime.h"

#define SPARK_GLM52_GATEWAY_REQUEST_BYTES (SPARK_GLM52_HTTP_GATEWAY_DEFAULT_MAX_UPLOAD_BYTES + (1024u * 1024u))
#define SPARK_GLM52_GATEWAY_RESPONSE_BYTES (128u * 1024u)
#define SPARK_GLM52_GATEWAY_COMPAT_TEXT_BYTES SPARK_GLM52_HTTP_GATEWAY_DEFAULT_MAX_UPLOAD_BYTES
#define SPARK_GLM52_GATEWAY_DEFAULT_PORT 8080u
#define SPARK_GLM52_GATEWAY_DEFAULT_PUMP_STEPS \
	SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_GLM52_GATEWAY_BATCH_COALESCE_POLL_MS 10u
#define SPARK_GLM52_GATEWAY_BACKEND_POLL_FD_CAPACITY 16u
#define SPARK_GLM52_GATEWAY_STREAM_BODY_BYTES 4096u
#define SPARK_GLM52_GATEWAY_STREAM_SLOT_BITS 16u
#define SPARK_GLM52_GATEWAY_STREAM_SLOT_MASK 0xffffu
#define SPARK_GLM52_GATEWAY_INVALID_STREAM_SLOT UINT32_MAX
#define SPARK_GLM52_GATEWAY_NONSTREAM_REQUEST_BIT (1ull << 63u)
#define SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY \
	SPARK_GLM52_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY
#define SPARK_GLM52_GATEWAY_POLL_FD_CAPACITY \
	(1u + SPARK_GLM52_GATEWAY_BACKEND_POLL_FD_CAPACITY + \
	 SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY)

static const char SparkGlm52GatewayContentLengthHeader[] = "Content-Length:";
static const char SparkGlm52GatewayContentLengthLowerHeader[] = "content-length:";
static const char SparkGlm52GatewayAuthorizationHeader[] = "Authorization:";
static const char SparkGlm52GatewayAuthorizationLowerHeader[] = "authorization:";

typedef struct SparkGlm52GatewayConfig
{
	const char *bind_address;
	const char *api_key;
	const char *api_key_file;
	const char *service_backend_path;
	const char *moe_pack_root;
	const char *stagepack_root;
	const char *transport_shared_object_path;
	const char *driver_shared_object_path;
	const char *node_context_builder_shared_object_path;
	const char *embedding_pack_path;
	const char *driver_program_name;
	const char *node_target;
	const char *tokenizer_path;
	const char *final_event_bind_address;
	const char *final_event_return_host;
	const char *cuda_resident_socket_path;
	char api_key_storage[256];
	uint32_t port;
	uint32_t require_service_backend;
	uint32_t pump_steps;
	uint32_t max_active_sequence_count;
	uint32_t kv_logical_block_capacity;
	uint32_t port_base;
	uint32_t model_quantization_mode;
	uint32_t dspark_enabled;
	uint32_t mtp_enabled;
} SparkGlm52GatewayConfig;

typedef struct SparkGlm52GatewayPendingStream
{
	int32_t fd;
	uint32_t active;
	uint32_t generation;
	uint32_t next_free_slot;
	uint32_t output_offset;
	uint32_t output_bytes;
	uint32_t terminal;
	SparkGlm52ServiceSubmitResult submit_result;
	char output[SPARK_GLM52_GATEWAY_STREAM_BODY_BYTES];
} SparkGlm52GatewayPendingStream;

typedef struct SparkGlm52GatewayRuntime
{
	SparkGlm52GatewayConfig configuration;
	SparkGlm52ServiceBackendDynamicLibrary service_backend_library;
	void *service_backend_state;
	SparkGlm52ServiceBackendView service_backend_view;
	SparkGlm52ServiceClientId service_client_id;
	uint64_t next_client_request_id;
	SparkGlm52GatewayPendingStream pending_streams[
		SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY];
	uint32_t pending_stream_free_head;
	uint32_t pending_stream_capacity;
	uint32_t pending_stream_count;
	uint32_t service_backend_attached;
	uint32_t pump_log_valid;
	uint32_t last_pump_status;
	uint32_t last_service_status;
	uint32_t last_serving_status;
	uint32_t last_live_request_count;
	uint32_t last_queued_request_count;
	uint64_t last_prefill_dispatch_count;
	uint64_t last_decode_dispatch_count;
} SparkGlm52GatewayRuntime;

static void SparkGlm52GatewayInitializeConfig(
	SparkGlm52GatewayConfig *configuration)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->bind_address = "127.0.0.1";
	configuration->driver_program_name = "glm52.ring.rank.production";
	configuration->final_event_bind_address = "0.0.0.0";
	configuration->final_event_return_host = "spark0";
	configuration->port = SPARK_GLM52_GATEWAY_DEFAULT_PORT;
	configuration->pump_steps = SPARK_GLM52_GATEWAY_DEFAULT_PUMP_STEPS;
	configuration->max_active_sequence_count = 1024u;
	configuration->port_base = 52100u;
	configuration->model_quantization_mode =
		SPARK_GLM52_RING_RUNTIME_DEFAULT_QUANTIZATION_MODE;
}

static int32_t SparkGlm52GatewayInitializePendingStreams(
	SparkGlm52GatewayRuntime *runtime)
{
	uint32_t slot_index;

	if (runtime == 0 ||
		runtime->configuration.max_active_sequence_count == 0u ||
		runtime->configuration.max_active_sequence_count >
			SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY ||
		runtime->configuration.max_active_sequence_count >
			SPARK_GLM52_GATEWAY_STREAM_SLOT_MASK)
		return -1;
	runtime->pending_stream_capacity =
		SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY;
	runtime->pending_stream_free_head = 0u;
	for (slot_index = 0u;
		 slot_index < runtime->pending_stream_capacity;
		 ++slot_index)
	{
		runtime->pending_streams[slot_index].fd = -1;
		runtime->pending_streams[slot_index].next_free_slot =
			slot_index + 1u < runtime->pending_stream_capacity ?
			slot_index + 1u : SPARK_GLM52_GATEWAY_INVALID_STREAM_SLOT;
	}
	return 0;
}

static SparkGlm52GatewayPendingStream *SparkGlm52GatewayAllocatePendingStream(
	SparkGlm52GatewayRuntime *runtime,
	int32_t fd,
	uint32_t *slot_index_out,
	uint64_t *client_request_id_out)
{
	SparkGlm52GatewayPendingStream *stream;
	uint32_t slot_index;

	if (runtime == 0 || fd < 0 || slot_index_out == 0 ||
		client_request_id_out == 0 ||
		runtime->pending_stream_free_head ==
			SPARK_GLM52_GATEWAY_INVALID_STREAM_SLOT)
		return 0;
	slot_index = runtime->pending_stream_free_head;
	stream = &runtime->pending_streams[slot_index];
	runtime->pending_stream_free_head = stream->next_free_slot;
	stream->generation += 1u;
	if (stream->generation == 0u)
		stream->generation = 1u;
	stream->fd = fd;
	stream->active = 1u;
	stream->next_free_slot = SPARK_GLM52_GATEWAY_INVALID_STREAM_SLOT;
	stream->output_offset = 0u;
	stream->output_bytes = 0u;
	stream->terminal = 0u;
	memset(&stream->submit_result,0,sizeof(stream->submit_result));
	runtime->pending_stream_count += 1u;
	*slot_index_out = slot_index;
	*client_request_id_out =
		((uint64_t)stream->generation << SPARK_GLM52_GATEWAY_STREAM_SLOT_BITS) |
		(uint64_t)(slot_index + 1u);
	return stream;
}

static void SparkGlm52GatewayReleasePendingStream(
	SparkGlm52GatewayRuntime *runtime,
	uint32_t slot_index)
{
	SparkGlm52GatewayPendingStream *stream;
	uint32_t generation;

	if (runtime == 0 || slot_index >= runtime->pending_stream_capacity)
		return;
	stream = &runtime->pending_streams[slot_index];
	if (stream->active == 0u)
		return;
	generation = stream->generation;
	if (stream->fd >= 0)
		close(stream->fd);
	memset(stream,0,sizeof(*stream));
	stream->fd = -1;
	stream->generation = generation;
	stream->next_free_slot = runtime->pending_stream_free_head;
	runtime->pending_stream_free_head = slot_index;
	runtime->pending_stream_count -= 1u;
}

static SparkGlm52GatewayPendingStream *SparkGlm52GatewayFindPendingStream(
	SparkGlm52GatewayRuntime *runtime,
	uint64_t client_request_id,
	uint32_t *slot_index_out)
{
	SparkGlm52GatewayPendingStream *stream;
	uint32_t slot_value;
	uint32_t slot_index;

	if (runtime == 0 || slot_index_out == 0)
		return 0;
	slot_value = (uint32_t)(client_request_id &
		SPARK_GLM52_GATEWAY_STREAM_SLOT_MASK);
	if (slot_value == 0u)
		return 0;
	slot_index = slot_value - 1u;
	if (slot_index >= runtime->pending_stream_capacity)
		return 0;
	stream = &runtime->pending_streams[slot_index];
	if (stream->active == 0u ||
		stream->submit_result.client_request_id != client_request_id)
		return 0;
	*slot_index_out = slot_index;
	return stream;
}

static uint64_t SparkGlm52GatewayNextNonstreamRequestId(
	SparkGlm52GatewayRuntime *runtime)
{
	uint64_t request_id;

	request_id = SPARK_GLM52_GATEWAY_NONSTREAM_REQUEST_BIT |
		runtime->next_client_request_id;
	runtime->next_client_request_id += 1u;
	if (runtime->next_client_request_id == 0u ||
		(runtime->next_client_request_id &
			SPARK_GLM52_GATEWAY_NONSTREAM_REQUEST_BIT) != 0u)
		runtime->next_client_request_id = 1u;
	return request_id;
}

static void SparkGlm52GatewayDetachPendingStream(
	SparkGlm52GatewayRuntime *runtime,
	uint32_t slot_index)
{
	if (runtime == 0 || slot_index >= runtime->pending_stream_capacity)
		return;
	runtime->pending_streams[slot_index].fd = -1;
	SparkGlm52GatewayReleasePendingStream(runtime,slot_index);
}

static int32_t SparkGlm52GatewayParseU32(const char *text,uint32_t *value_out)
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

static int32_t SparkGlm52GatewayApplyArgument(
	SparkGlm52GatewayConfig *configuration,
	int argc,
	char **argv,
	int32_t *index)
{
	uint32_t parsed;

	if (strcmp(argv[*index],"--bind") == 0)
	{
		if ((*index + 1) >= argc)
			return -1;
		configuration->bind_address = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--port") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -2;
		configuration->port = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--api-key") == 0)
	{
		if ((*index + 1) >= argc)
			return -3;
		configuration->api_key = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--api-key-file") == 0)
	{
		if ((*index + 1) >= argc)
			return -4;
		configuration->api_key_file = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--service-backend-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -5;
		configuration->service_backend_path = argv[*index + 1];
		configuration->require_service_backend = 1u;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--require-service-backend") == 0)
	{
		configuration->require_service_backend = 1u;
		return 0;
	}
	if (strcmp(argv[*index],"--pump-steps") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -6;
		configuration->pump_steps = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--moe-pack-root") == 0)
	{
		if ((*index + 1) >= argc)
			return -7;
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
			return -25;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--stagepack-root") == 0)
	{
		if ((*index + 1) >= argc)
			return -8;
		configuration->stagepack_root = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--transport-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -9;
		configuration->transport_shared_object_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--driver-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -10;
		configuration->driver_shared_object_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--node-context-builder-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -11;
		configuration->node_context_builder_shared_object_path =
			argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--embedding-pack") == 0)
	{
		if ((*index + 1) >= argc)
			return -12;
		configuration->embedding_pack_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--program") == 0)
	{
		if ((*index + 1) >= argc)
			return -13;
		configuration->driver_program_name = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--node-target") == 0)
	{
		if ((*index + 1) >= argc)
			return -13;
		configuration->node_target = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--tokenizer") == 0)
	{
		if ((*index + 1) >= argc)
			return -14;
		configuration->tokenizer_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--max-active") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -15;
		configuration->max_active_sequence_count = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--port-base") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -16;
		configuration->port_base = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--kv-logical-blocks") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0 ||
			parsed == 0u)
			return -21;
		configuration->kv_logical_block_capacity = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--final-event-bind") == 0)
	{
		if ((*index + 1) >= argc)
			return -17;
		configuration->final_event_bind_address = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--final-event-return-host") == 0)
	{
		if ((*index + 1) >= argc)
			return -18;
		configuration->final_event_return_host = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--cuda-resident-socket") == 0)
	{
		if ((*index + 1) >= argc)
			return -20;
		configuration->cuda_resident_socket_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--dspark") == 0)
	{
		configuration->dspark_enabled = 1u;
		return 0;
	}
	if (strcmp(argv[*index],"--mtp") == 0)
	{
		configuration->mtp_enabled = 1u;
		return 0;
	}
	return -19;
}

static int32_t SparkGlm52GatewayLoadApiKeyFile(
	SparkGlm52GatewayConfig *configuration)
{
	FILE *file;
	uint32_t bytes;

	if (configuration->api_key_file == 0)
		return 0;
	file = fopen(configuration->api_key_file,"rb");
	if (file == 0)
		return -1;
	bytes = (uint32_t)fread(
		configuration->api_key_storage,
		1u,
		sizeof(configuration->api_key_storage) - 1u,
		file);
	fclose(file);
	while (bytes != 0u &&
		(configuration->api_key_storage[bytes - 1u] == '\n' ||
		 configuration->api_key_storage[bytes - 1u] == '\r' ||
		 configuration->api_key_storage[bytes - 1u] == ' ' ||
		 configuration->api_key_storage[bytes - 1u] == '\t'))
		--bytes;
	configuration->api_key_storage[bytes] = '\0';
	if (bytes == 0u)
		return -2;
	configuration->api_key = configuration->api_key_storage;
	return 0;
}

static int32_t SparkGlm52GatewayParseArguments(
	SparkGlm52GatewayConfig *configuration,
	int argc,
	char **argv)
{
	int32_t index;

	for (index = 1; index < argc; ++index)
	{
		if (SparkGlm52GatewayApplyArgument(configuration,argc,argv,&index) < 0)
			return -1;
	}
	return 0;
}

static int32_t SparkGlm52GatewayRefreshBackendView(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkStatus status;

	if (runtime == 0 || runtime->service_backend_attached == 0u)
		return -1;
	status = runtime->service_backend_library.backend_interface.get_view(
		runtime->service_backend_state,
		&runtime->service_backend_view);
	if (status != SPARK_STATUS_OK)
		return -2;
	if (runtime->service_backend_view.abi_version !=
		SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION ||
		runtime->service_backend_view.descriptor_bytes !=
		SPARK_GLM52_SERVICE_BACKEND_VIEW_BYTES)
		return -3;
	return 0;
}

static void SparkGlm52GatewayLogServiceEvent(
	const SparkGlm52ServiceEvent *event)
{
	if (event == 0 || getenv("SPARKPIPE_GATEWAY_DRAIN_EVENTS") == 0)
		return;
	fprintf(
		stderr,
		"gateway_service_event kind=%u status=%u client_request=%llu serving_request=%llu sequence=%llu token=%u token_index=%u prompt_offset=%u prompt_count=%u dispatch_kind=%u dispatch_flags=%u\n",
		event->kind,
		event->status,
		(unsigned long long)event->client_request_id,
		(unsigned long long)event->serving_request_id,
		(unsigned long long)event->sequence_id,
		event->token_id,
		event->token_index,
		event->prompt_token_offset,
		event->prompt_token_count,
		event->dispatch_kind,
		event->dispatch_flags);
}

static void SparkGlm52GatewayPumpService(SparkGlm52GatewayRuntime *runtime)
{
	SparkGlm52ServiceStats service_stats;
	SparkStatus status;
	uint32_t should_log;

	if (runtime == 0 || runtime->service_backend_attached == 0u)
		return;
	status = runtime->service_backend_library.backend_interface.pump(
		runtime->service_backend_state,
		runtime->configuration.pump_steps,
		&service_stats);
	if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY &&
		status != SPARK_STATUS_NOT_FOUND)
		fprintf(stderr,"gateway backend pump status=%u\n",status);
	should_log = runtime->pump_log_valid == 0u ||
		runtime->last_pump_status != (uint32_t)status ||
		runtime->last_service_status != service_stats.last_status ||
		runtime->last_serving_status != service_stats.serving_stats.last_status ||
		runtime->last_live_request_count != service_stats.live_request_count ||
		runtime->last_queued_request_count !=
			service_stats.serving_stats.queued_request_count ||
		runtime->last_prefill_dispatch_count !=
			service_stats.serving_stats.prefill_dispatch_count ||
		runtime->last_decode_dispatch_count !=
			service_stats.serving_stats.decode_dispatch_count;
	if (should_log != 0u)
	{
		fprintf(
			stderr,
			"gateway_pump status=%u service_last=%u serving_last=%u live=%u queued=%u prefill=%llu decode=%llu events=%u\n",
			(uint32_t)status,
			service_stats.last_status,
			service_stats.serving_stats.last_status,
			service_stats.live_request_count,
			service_stats.serving_stats.queued_request_count,
			(unsigned long long)service_stats.serving_stats.prefill_dispatch_count,
			(unsigned long long)service_stats.serving_stats.decode_dispatch_count,
			service_stats.event_count);
		runtime->pump_log_valid = 1u;
		runtime->last_pump_status = (uint32_t)status;
		runtime->last_service_status = service_stats.last_status;
		runtime->last_serving_status = service_stats.serving_stats.last_status;
		runtime->last_live_request_count = service_stats.live_request_count;
		runtime->last_queued_request_count =
			service_stats.serving_stats.queued_request_count;
		runtime->last_prefill_dispatch_count =
			service_stats.serving_stats.prefill_dispatch_count;
		runtime->last_decode_dispatch_count =
			service_stats.serving_stats.decode_dispatch_count;
	}
}

static int32_t SparkGlm52GatewayPollTimeout(
	const SparkGlm52GatewayRuntime *runtime)
{
	if (runtime != 0 &&
		(runtime->pending_stream_count != 0u ||
		 runtime->last_live_request_count != 0u ||
		 runtime->last_queued_request_count != 0u))
		return 1;
	return -1;
}

static int16_t SparkGlm52GatewayPollEventsFromBackend(uint32_t backend_events)
{
	int16_t events;

	events = 0;
	if ((backend_events & SPARK_GLM52_SERVICE_BACKEND_POLL_READ) != 0u)
		events |= POLLIN;
	if ((backend_events & SPARK_GLM52_SERVICE_BACKEND_POLL_WRITE) != 0u)
		events |= POLLOUT;
	return events;
}

static uint32_t SparkGlm52GatewayAppendBackendPollFds(
	SparkGlm52GatewayRuntime *runtime,
	struct pollfd *fds,
	uint32_t fd_capacity,
	uint32_t fd_count)
{
	SparkGlm52ServiceBackendPollDescriptor descriptors[
		SPARK_GLM52_GATEWAY_BACKEND_POLL_FD_CAPACITY];
	uint32_t descriptor_count;
	uint32_t descriptor_index;
	int16_t events;

	if (runtime == 0 || fds == 0 ||
		runtime->service_backend_attached == 0u ||
		runtime->service_backend_library.backend_interface.get_poll_descriptors == 0)
		return fd_count;
	descriptor_count = 0u;
	if (runtime->service_backend_library.backend_interface.get_poll_descriptors(
			runtime->service_backend_state,
			descriptors,
			SPARK_GLM52_GATEWAY_BACKEND_POLL_FD_CAPACITY,
			&descriptor_count) != SPARK_STATUS_OK)
		return fd_count;
	for (descriptor_index = 0u;
		 descriptor_index < descriptor_count && fd_count < fd_capacity;
		 ++descriptor_index)
	{
		events = SparkGlm52GatewayPollEventsFromBackend(
			descriptors[descriptor_index].events);
		if (descriptors[descriptor_index].fd < 0 || events == 0)
			continue;
		memset(&fds[fd_count],0,sizeof(fds[fd_count]));
		fds[fd_count].fd = descriptors[descriptor_index].fd;
		fds[fd_count].events = events;
		fd_count += 1u;
	}
	return fd_count;
}

static uint32_t SparkGlm52GatewayAppendPendingStreamPollFds(
	SparkGlm52GatewayRuntime *runtime,
	struct pollfd *fds,
	uint32_t *stream_slot_indices,
	uint32_t fd_capacity,
	uint32_t fd_count)
{
	SparkGlm52GatewayPendingStream *stream;
	uint32_t slot_index;

	if (runtime == 0 || fds == 0 || stream_slot_indices == 0)
		return fd_count;
	for (slot_index = 0u;
		 slot_index < runtime->pending_stream_capacity && fd_count < fd_capacity;
		 ++slot_index)
	{
		stream = &runtime->pending_streams[slot_index];
		if (stream->active == 0u || stream->fd < 0)
			continue;
		memset(&fds[fd_count],0,sizeof(fds[fd_count]));
		fds[fd_count].fd = stream->fd;
		fds[fd_count].events = POLLIN;
		if (stream->output_bytes != 0u)
			fds[fd_count].events |= POLLOUT;
		stream_slot_indices[fd_count] = slot_index;
		fd_count += 1u;
	}
	return fd_count;
}

static int32_t SparkGlm52GatewayEnsureServiceClient(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkStatus status;

	if (runtime == 0 || runtime->service_backend_view.service == 0)
		return -1;
	if (runtime->service_client_id != 0u)
		return 0;
	status = SparkGlm52ServiceRegisterClient(
		runtime->service_backend_view.service,
		1u,
		&runtime->service_client_id);
	if (status != SPARK_STATUS_OK || runtime->service_client_id == 0u)
		return -2;
	runtime->next_client_request_id = 1u;
	return 0;
}

static int32_t SparkGlm52GatewayAttachServiceBackend(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkGlm52ServiceBackendConfiguration backend_configuration;
	SparkStatus status;

	if (runtime == 0)
		return -1;
	if (runtime->configuration.service_backend_path == 0)
		return runtime->configuration.require_service_backend != 0u ? -2 : 0;
	status = SparkGlm52ServiceBackendLoadInterfaceFromSharedObject(
		runtime->configuration.service_backend_path,
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_RING_RUNTIME,
		&runtime->service_backend_library);
	if (status != SPARK_STATUS_OK)
		return -3;
	memset(&backend_configuration,0,sizeof(backend_configuration));
	backend_configuration.abi_version =
		SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION;
	backend_configuration.descriptor_bytes =
		SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES;
	backend_configuration.max_active_sequence_count =
		runtime->configuration.max_active_sequence_count;
	backend_configuration.port_base = runtime->configuration.port_base;
	backend_configuration.kv_logical_block_capacity =
		runtime->configuration.kv_logical_block_capacity;
	backend_configuration.model_quantization_mode =
		runtime->configuration.model_quantization_mode;
	backend_configuration.moe_pack_root = runtime->configuration.moe_pack_root;
	backend_configuration.stagepack_root =
		runtime->configuration.stagepack_root;
	backend_configuration.transport_shared_object_path =
		runtime->configuration.transport_shared_object_path;
	backend_configuration.driver_shared_object_path =
		runtime->configuration.driver_shared_object_path;
	backend_configuration.node_context_builder_shared_object_path =
		runtime->configuration.node_context_builder_shared_object_path;
	backend_configuration.embedding_pack_path =
		runtime->configuration.embedding_pack_path;
	backend_configuration.driver_program_name =
		runtime->configuration.driver_program_name;
	backend_configuration.node_target = runtime->configuration.node_target;
	backend_configuration.tokenizer_path = runtime->configuration.tokenizer_path;
	backend_configuration.final_event_bind_address =
		runtime->configuration.final_event_bind_address;
	backend_configuration.final_event_return_host =
		runtime->configuration.final_event_return_host;
	backend_configuration.cuda_resident_socket_path =
		runtime->configuration.cuda_resident_socket_path;
	if (runtime->configuration.dspark_enabled != 0u)
		backend_configuration.flags |=
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK;
	if (runtime->configuration.mtp_enabled != 0u)
		backend_configuration.flags |=
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP;
	status = runtime->service_backend_library.backend_interface.initialize(
		&backend_configuration,
		&runtime->service_backend_state);
	if (status != SPARK_STATUS_OK || runtime->service_backend_state == 0)
	{
		SparkGlm52ServiceBackendView view;
		memset(&view,0,sizeof(view));
		if (runtime->service_backend_state != 0 &&
			runtime->service_backend_library.backend_interface.get_view != 0 &&
			runtime->service_backend_library.backend_interface.get_view(
				runtime->service_backend_state,
				&view) == SPARK_STATUS_OK &&
			view.first_blocker != 0)
			fprintf(stderr,"service backend initialize failed status=%d blocker=%s\n",
				(int32_t)status,view.first_blocker);
		else
			fprintf(stderr,"service backend initialize failed status=%d\n",
				(int32_t)status);
		return -4;
	}
	runtime->service_backend_attached = 1u;
	if (SparkGlm52GatewayRefreshBackendView(runtime) < 0)
		return -5;
	if (runtime->service_backend_view.service != 0 &&
		SparkGlm52GatewayEnsureServiceClient(runtime) < 0)
		return -6;
	return 0;
}

static void SparkGlm52GatewayDestroyServiceBackend(
	SparkGlm52GatewayRuntime *runtime)
{
	if (runtime == 0)
		return;
	if (runtime->service_backend_state != 0 &&
		runtime->service_backend_library.backend_interface.destroy != 0)
		runtime->service_backend_library.backend_interface.destroy(
			runtime->service_backend_state);
	runtime->service_backend_state = 0;
	runtime->service_backend_attached = 0u;
	SparkGlm52ServiceBackendUnloadInterface(&runtime->service_backend_library);
}

static uint32_t SparkGlm52GatewayServiceEventIsTerminal(
	const SparkGlm52ServiceEvent *event)
{
	if (event == 0)
		return 0u;
	return event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_COMPLETED ||
		event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_CANCELLED ||
		event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_ERROR;
}

static void SparkGlm52GatewayCancelPendingStream(
	SparkGlm52GatewayRuntime *runtime,
	uint32_t slot_index)
{
	SparkGlm52GatewayPendingStream *stream;

	if (runtime == 0 || slot_index >= runtime->pending_stream_capacity)
		return;
	stream = &runtime->pending_streams[slot_index];
	if (stream->active == 0u)
		return;
	if (runtime->service_backend_view.service != 0 &&
		stream->submit_result.client_request_id != 0u)
	{
		(void)SparkGlm52ServiceCancelRequest(
			runtime->service_backend_view.service,
			runtime->service_client_id,
			stream->submit_result.client_request_id);
	}
	SparkGlm52GatewayReleasePendingStream(runtime,slot_index);
}

static void SparkGlm52GatewayHandlePendingStreamPollFds(
	SparkGlm52GatewayRuntime *runtime,
	const struct pollfd *fds,
	const uint32_t *stream_slot_indices,
	uint32_t first_fd,
	uint32_t fd_count)
{
	SparkGlm52GatewayPendingStream *stream;
	ssize_t received;
	uint32_t fd_index;
	uint32_t slot_index;
	uint8_t byte;

	if (runtime == 0 || fds == 0 || stream_slot_indices == 0)
		return;
	for (fd_index = first_fd; fd_index < fd_count; ++fd_index)
	{
		slot_index = stream_slot_indices[fd_index];
		if (slot_index >= runtime->pending_stream_capacity)
			continue;
		stream = &runtime->pending_streams[slot_index];
		if (stream->active == 0u || stream->fd != fds[fd_index].fd)
			continue;
		if ((fds[fd_index].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
		{
			SparkGlm52GatewayCancelPendingStream(runtime,slot_index);
			continue;
		}
		if ((fds[fd_index].revents & POLLIN) == 0)
			continue;
		received = recv(stream->fd,&byte,1u,MSG_PEEK | MSG_DONTWAIT);
		if (received == 0 ||
			(received < 0 && errno != EINTR && errno != EAGAIN &&
			 errno != EWOULDBLOCK))
			SparkGlm52GatewayCancelPendingStream(runtime,slot_index);
	}
}

static SparkStatus SparkGlm52GatewayFlushPendingStream(
	SparkGlm52GatewayPendingStream *stream)
{
	ssize_t sent;

	if (stream == 0 || stream->active == 0u || stream->fd < 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	while (stream->output_offset < stream->output_bytes)
	{
		sent = send(
			stream->fd,
			stream->output + stream->output_offset,
			stream->output_bytes - stream->output_offset,
			MSG_DONTWAIT | MSG_NOSIGNAL);
		if (sent < 0 && errno == EINTR)
			continue;
		if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
			return SPARK_STATUS_BUSY;
		if (sent <= 0)
			return SPARK_STATUS_IO_ERROR;
		stream->output_offset += (uint32_t)sent;
	}
	stream->output_offset = 0u;
	stream->output_bytes = 0u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52GatewayQueueServiceEvent(
	SparkGlm52GatewayRuntime *runtime,
	uint32_t slot_index,
	const SparkGlm52ServiceEvent *event)
{
	SparkGlm52GatewayPendingStream *stream;
	SparkGlm52HttpGatewayResponse response;
	SparkStatus status;

	stream = &runtime->pending_streams[slot_index];
	if (stream->output_bytes != 0u)
	{
		status = SparkGlm52GatewayFlushPendingStream(stream);
		if (status == SPARK_STATUS_BUSY)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		if (status != SPARK_STATUS_OK)
			return status;
	}
	SparkGlm52HttpGatewayInitializeResponse(
		&response,
		stream->output,
		sizeof(stream->output));
	status = SparkGlm52HttpGatewayBuildServiceEventStream(
		&response,
		event,
		runtime->service_backend_view.tokenizer);
	if (status != SPARK_STATUS_OK)
		return status;
	stream->output_offset = 0u;
	stream->output_bytes = response.body_bytes;
	stream->terminal = SparkGlm52GatewayServiceEventIsTerminal(event);
	return SparkGlm52GatewayFlushPendingStream(stream);
}

static void SparkGlm52GatewayDispatchServiceEvents(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkGlm52GatewayPendingStream *stream;
	SparkGlm52ServiceEvent event;
	SparkStatus status;
	uint32_t slot_index;

	if (runtime == 0 || runtime->service_backend_view.service == 0)
		return;
	for (;;)
	{
		status = SparkGlm52ServicePopEvent(
			runtime->service_backend_view.service,
			&event);
		if (status == SPARK_STATUS_NOT_FOUND)
			return;
		if (status != SPARK_STATUS_OK)
		{
			fprintf(stderr,"gateway service event pop failed status=%u\n",status);
			return;
		}
		SparkGlm52GatewayLogServiceEvent(&event);
		stream = SparkGlm52GatewayFindPendingStream(
			runtime,event.client_request_id,&slot_index);
		if (stream == 0 ||
			stream->submit_result.serving_request_id != event.serving_request_id ||
			(stream->submit_result.sequence_id != 0u &&
			 stream->submit_result.sequence_id != event.sequence_id))
			continue;
		if (event.kind == SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_ACCEPTED)
			continue;
		status = SparkGlm52GatewayQueueServiceEvent(runtime,slot_index,&event);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
			SparkGlm52GatewayCancelPendingStream(runtime,slot_index);
		else if (status == SPARK_STATUS_OK && stream->terminal != 0u)
			SparkGlm52GatewayReleasePendingStream(runtime,slot_index);
	}
}

static void SparkGlm52GatewayFlushPendingStreams(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkGlm52GatewayPendingStream *stream;
	SparkStatus status;
	uint32_t slot_index;

	if (runtime == 0)
		return;
	for (slot_index = 0u;
		 slot_index < runtime->pending_stream_capacity;
		 ++slot_index)
	{
		stream = &runtime->pending_streams[slot_index];
		if (stream->active == 0u || stream->output_bytes == 0u)
			continue;
		status = SparkGlm52GatewayFlushPendingStream(stream);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
			SparkGlm52GatewayCancelPendingStream(runtime,slot_index);
		else if (status == SPARK_STATUS_OK && stream->terminal != 0u)
			SparkGlm52GatewayReleasePendingStream(runtime,slot_index);
	}
}

static int32_t SparkGlm52GatewayCreateListenSocket(
	const SparkGlm52GatewayConfig *configuration)
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
	address.sin_port = htons((uint16_t)configuration->port);
	if (inet_pton(AF_INET,configuration->bind_address,&address.sin_addr) != 1)
	{
		close(fd);
		return -3;
	}
	if (bind(fd,(struct sockaddr *)&address,sizeof(address)) < 0)
	{
		close(fd);
		return -4;
	}
	if (listen(fd,(int32_t)SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY) < 0)
	{
		close(fd);
		return -5;
	}
	if (fcntl(fd,F_SETFL,fcntl(fd,F_GETFL,0) | O_NONBLOCK) < 0)
	{
		close(fd);
		return -6;
	}
	return fd;
}

static char *SparkGlm52GatewayFindHeaderEnd(char *buffer,uint32_t bytes)
{
	uint32_t index;

	if (bytes < 4u)
		return 0;
	for (index = 0u; (index + 4u) <= bytes; ++index)
	{
		if (buffer[index] == '\r' && buffer[index + 1u] == '\n' &&
			buffer[index + 2u] == '\r' && buffer[index + 3u] == '\n')
			return &buffer[index];
	}
	return 0;
}

static const char *SparkGlm52GatewaySkipSpaces(const char *text)
{
	while (text != 0 && (*text == ' ' || *text == '\t'))
		++text;
	return text;
}

static void SparkGlm52GatewayExtractHeaders(
	char *headers,
	SparkGlm52HttpGatewayRequest *request,
	uint32_t *content_length_out)
{
	char *line;
	char *next;
	uint32_t value;

	*content_length_out = 0u;
	line = headers;
	while (line != 0 && line[0] != '\0')
	{
		next = strstr(line,"\r\n");
		if (next != 0)
		{
			next[0] = '\0';
			next += 2;
		}
		if (strncmp(
				line,
				SparkGlm52GatewayContentLengthHeader,
				sizeof(SparkGlm52GatewayContentLengthHeader) - 1u) == 0)
		{
			if (SparkGlm52GatewayParseU32(
					SparkGlm52GatewaySkipSpaces(
						line + sizeof(SparkGlm52GatewayContentLengthHeader) - 1u),
					&value) == 0)
				*content_length_out = value;
		}
		if (strncmp(
				line,
				SparkGlm52GatewayContentLengthLowerHeader,
				sizeof(SparkGlm52GatewayContentLengthLowerHeader) - 1u) == 0)
		{
			if (SparkGlm52GatewayParseU32(
					SparkGlm52GatewaySkipSpaces(
						line + sizeof(SparkGlm52GatewayContentLengthLowerHeader) - 1u),
					&value) == 0)
				*content_length_out = value;
		}
		if (strncmp(
				line,
				SparkGlm52GatewayAuthorizationHeader,
				sizeof(SparkGlm52GatewayAuthorizationHeader) - 1u) == 0)
		{
			request->authorization = SparkGlm52GatewaySkipSpaces(
				line + sizeof(SparkGlm52GatewayAuthorizationHeader) - 1u);
			request->authorization_bytes = (uint32_t)strlen(request->authorization);
		}
		if (strncmp(
				line,
				SparkGlm52GatewayAuthorizationLowerHeader,
				sizeof(SparkGlm52GatewayAuthorizationLowerHeader) - 1u) == 0)
		{
			request->authorization = SparkGlm52GatewaySkipSpaces(
				line + sizeof(SparkGlm52GatewayAuthorizationLowerHeader) - 1u);
			request->authorization_bytes = (uint32_t)strlen(request->authorization);
		}
		line = next;
	}
}

static int32_t SparkGlm52GatewayParseRequestLine(
	char *line,
	SparkGlm52HttpGatewayRequest *request)
{
	char *path;
	char *version;

	path = strchr(line,' ');
	if (path == 0)
		return -1;
	path[0] = '\0';
	++path;
	version = strchr(path,' ');
	if (version == 0)
		return -2;
	version[0] = '\0';
	request->method = line;
	request->path = path;
	return 0;
}

static int32_t SparkGlm52GatewayReadRequest(
	int32_t fd,
	char *buffer,
	uint32_t capacity,
	SparkGlm52HttpGatewayRequest *request)
{
	char *header_end;
	char *body;
	char *line_end;
	uint32_t bytes;
	uint32_t header_bytes;
	uint32_t content_length;
	int32_t got;

	bytes = 0u;
	header_end = 0;
	while (bytes < capacity)
	{
		got = (int32_t)recv(fd,buffer + bytes,capacity - bytes,0);
		if (got <= 0)
			return -1;
		bytes += (uint32_t)got;
		header_end = SparkGlm52GatewayFindHeaderEnd(buffer,bytes);
		if (header_end != 0)
			break;
	}
	if (header_end == 0)
		return -2;
	header_bytes = (uint32_t)(header_end - buffer);
	body = header_end + 4;
	buffer[header_bytes] = '\0';
	line_end = strstr(buffer,"\r\n");
	if (line_end == 0)
		return -3;
	line_end[0] = '\0';
	if (SparkGlm52GatewayParseRequestLine(buffer,request) < 0)
		return -4;
	SparkGlm52GatewayExtractHeaders(line_end + 2,request,&content_length);
	if (content_length > (capacity - (uint32_t)(body - buffer)))
		return -6;
	while (((uint32_t)(bytes - (uint32_t)(body - buffer))) < content_length &&
		bytes < capacity)
	{
		got = (int32_t)recv(fd,buffer + bytes,capacity - bytes,0);
		if (got <= 0)
			return -5;
		bytes += (uint32_t)got;
	}
	request->body = body;
	request->body_bytes = content_length;
	return 0;
}

static const char *SparkGlm52GatewayStatusText(uint32_t status_code)
{
	if (status_code == 200u)
		return "OK";
	if (status_code == 202u)
		return "Accepted";
	if (status_code == 400u)
		return "Bad Request";
	if (status_code == 401u)
		return "Unauthorized";
	if (status_code == 404u)
		return "Not Found";
	if (status_code == 503u)
		return "Service Unavailable";
	return "Bad Request";
}

static int32_t SparkGlm52GatewaySendAll(
	int32_t fd,
	const void *data,
	uint32_t bytes)
{
	const uint8_t *source;
	uint32_t offset;
	ssize_t sent;

	if (fd < 0 || data == 0)
		return -1;
	source = (const uint8_t *)data;
	offset = 0u;
	while (offset < bytes)
	{
		sent = send(fd,source + offset,bytes - offset,MSG_NOSIGNAL);
		if (sent < 0 && errno == EINTR)
			continue;
		if (sent <= 0)
			return -2;
		offset += (uint32_t)sent;
	}
	return 0;
}

static int32_t SparkGlm52GatewaySendStreamStart(
	int32_t fd,
	const SparkGlm52HttpGatewayResponse *response)
{
	char header[1024];
	int32_t header_bytes;
	int32_t flags;

	header_bytes = snprintf(
		header,
		sizeof(header),
		"HTTP/1.1 %u %s\r\n"
		"Content-Type: %s\r\n"
		"Access-Control-Allow-Origin: *\r\n"
		"Access-Control-Allow-Headers: authorization, content-type\r\n"
		"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
		"Cache-Control: no-cache\r\n"
		"Connection: close\r\n\r\n",
		response->status_code,
		SparkGlm52GatewayStatusText(response->status_code),
		response->content_type);
	if (header_bytes < 0 || (uint32_t)header_bytes >= sizeof(header) ||
		SparkGlm52GatewaySendAll(fd,header,(uint32_t)header_bytes) < 0 ||
		SparkGlm52GatewaySendAll(fd,response->body,response->body_bytes) < 0)
		return -1;
	flags = fcntl(fd,F_GETFL,0);
	if (flags < 0 || fcntl(fd,F_SETFL,flags | O_NONBLOCK) < 0)
		return -2;
	return 0;
}

static int32_t SparkGlm52GatewaySendResponse(
	int32_t fd,
	const SparkGlm52HttpGatewayResponse *response)
{
	char header[1024];
	int32_t header_bytes;

	header_bytes = snprintf(
		header,
		sizeof(header),
		"HTTP/1.1 %u %s\r\n"
		"Content-Type: %s\r\n"
		"Content-Length: %u\r\n"
		"Access-Control-Allow-Origin: *\r\n"
		"Access-Control-Allow-Headers: authorization, content-type\r\n"
		"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
		"Cache-Control: no-cache\r\n"
		"Connection: close\r\n\r\n",
		response->status_code,
		SparkGlm52GatewayStatusText(response->status_code),
		response->content_type,
		response->body_bytes);
	if (header_bytes < 0 || (uint32_t)header_bytes >= sizeof(header))
		return -1;
	if (SparkGlm52GatewaySendAll(fd,header,(uint32_t)header_bytes) < 0)
		return -2;
	if (SparkGlm52GatewaySendAll(fd,response->body,response->body_bytes) < 0)
		return -3;
	return 0;
}

static SparkStatus SparkGlm52GatewayBuildBadRequest(
	SparkGlm52HttpGatewayResponse *response)
{
	if (response == 0 || response->body == 0 || response->body_capacity < 64u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	response->status_code = 400u;
	response->flags = 0u;
	response->content_type = "application/json";
	strcpy(response->body,"{\"error\":{\"type\":\"bad_request\",\"message\":\"invalid or oversized request\"}}\n");
	response->body_bytes = (uint32_t)strlen(response->body);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52GatewayBuildResponse(
	SparkGlm52GatewayRuntime *runtime,
	const SparkGlm52HttpGatewayRequest *request,
	uint64_t client_request_id,
	SparkGlm52ServiceSubmitResult *submit_result_out,
	SparkGlm52HttpGatewayResponse *response)
{
	static char compat_text[SPARK_GLM52_GATEWAY_COMPAT_TEXT_BYTES];
	SparkGlm52CompatTextRequest compat_request;
	SparkGlm52ServiceSubmitResult submit_result;
	SparkGlm52ServiceStats service_stats;
	uint32_t route;
	uint32_t stream;
	SparkStatus status;

	if (submit_result_out != 0)
		memset(submit_result_out,0,sizeof(*submit_result_out));

	route = SparkGlm52HttpGatewayRoute(request);
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_CORS_PREFLIGHT)
		return SparkGlm52HttpGatewayBuildCorsPreflight(response);
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI)
		return SparkGlm52HttpGatewayBuildDemoUi(response);
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH)
	{
		memset(&service_stats,0,sizeof(service_stats));
		if (runtime->service_backend_attached != 0u &&
			SparkGlm52GatewayRefreshBackendView(runtime) == 0 &&
			runtime->service_backend_view.service != 0)
		{
			(void)SparkGlm52ServiceGetStats(
				runtime->service_backend_view.service,
				&service_stats);
		}
		return SparkGlm52HttpGatewayBuildServiceHealth(
			response,
			runtime->service_backend_attached != 0u ? &service_stats : 0,
			&runtime->service_backend_view);
	}
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE)
		return SparkGlm52HttpGatewayBuildNotFound(response);
	if (SparkGlm52HttpGatewayAuthorizationMatches(
			request,
			runtime->configuration.api_key) == 0u)
		return SparkGlm52HttpGatewayBuildUnauthorized(response);
	stream = SparkGlm52HttpGatewayBodyRequestsStream(
		request->body,
		request->body_bytes);
	if (runtime->service_backend_attached == 0u ||
		SparkGlm52GatewayRefreshBackendView(runtime) < 0 ||
		runtime->service_backend_view.service == 0 ||
		runtime->service_backend_view.runtime_initialized == 0u ||
		runtime->service_backend_view.local_control_ready == 0u ||
		SparkGlm52GatewayEnsureServiceClient(runtime) < 0)
		return SparkGlm52HttpGatewayBuildBackendUnavailable(response,stream);
	SparkGlm52CompatInitializeTextRequest(
		&compat_request,
		compat_text,
		sizeof(compat_text));
	memset(&submit_result,0,sizeof(submit_result));
	status = SparkGlm52HttpGatewaySubmitJsonToService(
		runtime->service_backend_view.service,
		request,
		runtime->service_client_id,
		client_request_id,
		&compat_request,
		&submit_result,
		response);
	if (status == SPARK_STATUS_OK)
	{
		if (submit_result_out != 0)
			*submit_result_out = submit_result;
		if (stream == 0u)
		{
			(void)runtime->service_backend_library.backend_interface.pump(
				runtime->service_backend_state,
				runtime->configuration.pump_steps,
				&service_stats);
		}
		return SPARK_STATUS_OK;
	}
	return SparkGlm52HttpGatewayBuildBackendUnavailable(response,stream);
}

static int32_t SparkGlm52GatewayServeOne(
	SparkGlm52GatewayRuntime *runtime,
	int32_t client_fd)
{
	static char request_buffer[SPARK_GLM52_GATEWAY_REQUEST_BYTES];
	static char response_buffer[SPARK_GLM52_GATEWAY_RESPONSE_BYTES];
	SparkGlm52GatewayPendingStream *pending_stream;
	SparkGlm52HttpGatewayRequest request;
	SparkGlm52HttpGatewayResponse response;
	SparkGlm52ServiceSubmitResult submit_result;
	SparkStatus status;
	uint64_t client_request_id;
	uint32_t slot_index;
	uint32_t stream;
	int32_t read_status;

	SparkGlm52HttpGatewayInitializeRequest(&request);
	SparkGlm52HttpGatewayInitializeResponse(
		&response,
		response_buffer,
		sizeof(response_buffer));
	read_status = SparkGlm52GatewayReadRequest(
		client_fd,
		request_buffer,
		sizeof(request_buffer),
		&request);
	if (read_status < 0)
	{
		if (read_status == -6)
			(void)SparkGlm52GatewayBuildBadRequest(&response);
		else
		(void)SparkGlm52HttpGatewayBuildNotFound(&response);
	}
	else
	{
		stream = SparkGlm52HttpGatewayBodyRequestsStream(
			request.body,request.body_bytes);
		pending_stream = 0;
		slot_index = SPARK_GLM52_GATEWAY_INVALID_STREAM_SLOT;
		if (stream != 0u)
		{
			pending_stream = SparkGlm52GatewayAllocatePendingStream(
				runtime,client_fd,&slot_index,&client_request_id);
			if (pending_stream == 0)
			{
				(void)SparkGlm52HttpGatewayBuildBackendUnavailable(
					&response,1u);
				return SparkGlm52GatewaySendResponse(client_fd,&response);
			}
		}
		else
		{
			client_request_id =
				SparkGlm52GatewayNextNonstreamRequestId(runtime);
		}
		memset(&submit_result,0,sizeof(submit_result));
		status = SparkGlm52GatewayBuildResponse(
			runtime,
			&request,
			client_request_id,
			&submit_result,
			&response);
		if (status == SPARK_STATUS_OK && pending_stream != 0 &&
			submit_result.client_request_id != 0u)
		{
			pending_stream->submit_result = submit_result;
			if (SparkGlm52GatewaySendStreamStart(client_fd,&response) < 0)
				SparkGlm52GatewayCancelPendingStream(runtime,slot_index);
			return 1;
		}
		if (pending_stream != 0)
		{
			if (submit_result.client_request_id != 0u &&
				runtime->service_backend_view.service != 0)
			{
				(void)SparkGlm52ServiceCancelRequest(
					runtime->service_backend_view.service,
					runtime->service_client_id,
					submit_result.client_request_id);
			}
			SparkGlm52GatewayDetachPendingStream(runtime,slot_index);
		}
		if (status == SPARK_STATUS_BUSY)
			(void)SparkGlm52HttpGatewayBuildRequestTimeout(&response,SparkGlm52HttpGatewayBodyRequestsStream(request.body,request.body_bytes));
		else if (status != SPARK_STATUS_OK)
		{
			fprintf(stderr,"gateway_build_response_failed status=%d\n",(int32_t)status);
			(void)SparkGlm52HttpGatewayBuildBackendUnavailable(&response,0u);
		}
	}
	return SparkGlm52GatewaySendResponse(client_fd,&response);
}

static uint32_t SparkGlm52GatewayShouldCoalesceBatch(
	const SparkGlm52GatewayRuntime *runtime)
{
	if (runtime == 0 || runtime->last_live_request_count != 0u ||
		runtime->pending_stream_count == 0u ||
		runtime->pending_stream_count >=
			runtime->configuration.max_active_sequence_count)
		return 0u;
	return 1u;
}

static uint32_t SparkGlm52GatewayAcceptReadyClients(
	SparkGlm52GatewayRuntime *runtime,
	int32_t listen_fd)
{
	int32_t client_fd;
	int32_t ownership;
	uint32_t accepted_count;

	accepted_count = 0u;
	for (;;)
	{
		client_fd = accept(listen_fd,0,0);
		if (client_fd < 0)
			break;
		ownership = SparkGlm52GatewayServeOne(runtime,client_fd);
		if (ownership != 1)
			close(client_fd);
		accepted_count += 1u;
	}
	return accepted_count;
}

static void SparkGlm52GatewayCoalesceBatch(
	SparkGlm52GatewayRuntime *runtime,
	int32_t listen_fd)
{
	struct pollfd listen_poll;
	int32_t poll_result;

	memset(&listen_poll,0,sizeof(listen_poll));
	listen_poll.fd = listen_fd;
	listen_poll.events = POLLIN;
	while (SparkGlm52GatewayShouldCoalesceBatch(runtime) != 0u)
	{
		listen_poll.revents = 0;
		poll_result = poll(
			&listen_poll,1,SPARK_GLM52_GATEWAY_BATCH_COALESCE_POLL_MS);
		if (poll_result <= 0 || (listen_poll.revents & POLLIN) == 0)
			return;
		if (SparkGlm52GatewayAcceptReadyClients(runtime,listen_fd) == 0u)
			return;
	}
}

int main(int argc,char **argv)
{
	static SparkGlm52GatewayRuntime runtime;
	struct pollfd poll_fds[SPARK_GLM52_GATEWAY_POLL_FD_CAPACITY];
	uint32_t poll_stream_slots[SPARK_GLM52_GATEWAY_POLL_FD_CAPACITY];
	int32_t listen_fd;
	int32_t poll_result;
	int32_t poll_timeout;
	uint32_t fd_count;
	uint32_t pending_stream_fd_start;
	uint32_t slot_index;

	(void)signal(SIGPIPE,SIG_IGN);
	memset(&runtime,0,sizeof(runtime));
	SparkGlm52GatewayInitializeConfig(&runtime.configuration);
	if (SparkGlm52GatewayParseArguments(&runtime.configuration,argc,argv) < 0)
	{
		fprintf(stderr,"usage: %s [--bind ip] [--port n] [--api-key key] [--api-key-file path] [--service-backend-so path] [--require-service-backend] [--pump-steps n] [--model-quantization fp8|nvfp4|w8lut] [--moe-pack-root dir] [--stagepack-root dir] [--transport-so path] [--driver-so path] [--program name] [--node-target target] [--node-context-builder-so path] [--embedding-pack path] [--tokenizer path] [--max-active n] [--kv-logical-blocks n] [--port-base n] [--final-event-bind ip] [--final-event-return-host host] [--cuda-resident-socket path] [--mtp] [--dspark]\n",argv[0]);
		return 2;
	}
	if (SparkGlm52GatewayInitializePendingStreams(&runtime) < 0)
	{
		fprintf(stderr,"invalid max-active for gateway stream pool\n");
		return 2;
	}
	if (SparkGlm52GatewayLoadApiKeyFile(&runtime.configuration) < 0)
	{
		fprintf(stderr,"api key file failed\n");
		return 2;
	}
	if (SparkGlm52GatewayAttachServiceBackend(&runtime) < 0)
	{
		fprintf(stderr,"service backend attach failed\n");
		SparkGlm52GatewayDestroyServiceBackend(&runtime);
		return 2;
	}
	listen_fd = SparkGlm52GatewayCreateListenSocket(&runtime.configuration);
	if (listen_fd < 0)
	{
		fprintf(stderr,"listen failed: %s\n",strerror(errno));
		SparkGlm52GatewayDestroyServiceBackend(&runtime);
		return 3;
	}
	fprintf(stderr,"sparkpipe glm52 gateway listening on %s:%u\n",
		runtime.configuration.bind_address,
		runtime.configuration.port);
	for (;;)
	{
		SparkGlm52GatewayFlushPendingStreams(&runtime);
		SparkGlm52GatewayPumpService(&runtime);
		SparkGlm52GatewayDispatchServiceEvents(&runtime);
		SparkGlm52GatewayFlushPendingStreams(&runtime);
		memset(poll_fds,0,sizeof(poll_fds));
		fd_count = 0u;
		poll_fds[0].fd = listen_fd;
		poll_fds[0].events = POLLIN;
		fd_count = 1u;
		fd_count = SparkGlm52GatewayAppendBackendPollFds(
			&runtime,
			poll_fds,
			SPARK_GLM52_GATEWAY_POLL_FD_CAPACITY,
			fd_count);
		pending_stream_fd_start = fd_count;
		fd_count = SparkGlm52GatewayAppendPendingStreamPollFds(
			&runtime,
			poll_fds,
			poll_stream_slots,
			SPARK_GLM52_GATEWAY_POLL_FD_CAPACITY,
			fd_count);
		poll_timeout = SparkGlm52GatewayPollTimeout(&runtime);
		poll_result = poll(poll_fds,fd_count,poll_timeout);
		if (poll_result < 0)
		{
			if (errno == EINTR)
				continue;
			break;
		}
		SparkGlm52GatewayHandlePendingStreamPollFds(
			&runtime,
			poll_fds,
			poll_stream_slots,
			pending_stream_fd_start,
			fd_count);
		if ((poll_fds[0].revents & POLLIN) != 0)
		{
			(void)SparkGlm52GatewayAcceptReadyClients(&runtime,listen_fd);
			SparkGlm52GatewayCoalesceBatch(&runtime,listen_fd);
		}
	}
	for (slot_index = 0u;
		 slot_index < runtime.pending_stream_capacity;
		 ++slot_index)
		SparkGlm52GatewayCancelPendingStream(&runtime,slot_index);
	SparkGlm52GatewayDestroyServiceBackend(&runtime);
	return 0;
}
