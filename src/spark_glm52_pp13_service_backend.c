#define _POSIX_C_SOURCE 200112L

#include "sparkpipe/spark_glm52_service_backend.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"
#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"
#include "sparkpipe/spark_model_driver.h"

#define SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_PORT_BASE 52100u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_MAGIC 0x35454650u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY 128u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_CLIENT_CAPACITY 128u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_MAP_CAPACITY 128u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_CONTEXT_TOKENS 65536u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_TOKENS 512u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT \
	(SPARK_GLM52_PP13_SERVICE_BACKEND_CONTEXT_TOKENS / \
	 SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS)
#define SPARK_GLM52_PP13_SERVICE_BACKEND_PREFIX_BINDING_COUNT \
	(SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT + \
	 SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY)
#define SPARK_GLM52_PP13_SERVICE_BACKEND_EVENT_CAPACITY 16384u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_METADATA_KEY_BASE 0x100000000ull
#define SPARK_GLM52_PP13_SERVICE_BACKEND_METADATA_VALUE_BASE 0x200000000ull
#define SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_POLL_MS 10
#define SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_POLL_COUNT 3000u

typedef struct SparkGlm52Pp13ServiceBackendFinalEvent
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
} SparkGlm52Pp13ServiceBackendFinalEvent;

typedef struct SparkGlm52Pp13ServiceBackendState
{
	SparkGlm52Pp13RuntimeRankPlan rank_plan;
	SparkGlm52Pp13RuntimeFinalEventRoute final_event_route;
	SparkHiddenTransportDynamicLibrary transport_library;
	SparkHiddenTransportSession *output_transport_session;
	SparkGlm52Pp13NodeContextBuilderDynamicLibrary builder_library;
	void *builder_state;
	SparkGlm52Pp13NodeContextBuilderResult builder_result;
	SparkLoadedModelDriver loaded_driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkGlm52ResidentDecodeStageProductionRunner runner;
	SparkTokenizer tokenizer;
	SparkGlm52KvCacheArena kv_arena;
	SparkGlm52PrefixCache prefix_cache;
	SparkGlm52Scheduler scheduler;
	SparkGlm52RequestApi request_api;
	SparkGlm52ServingEngine serving_engine;
	SparkGlm52ServiceRuntime service;
	SparkGlm52KvCacheBlock *kv_blocks;
	SparkGlm52PrefixCacheEntry *prefix_entries;
	SparkGlm52PrefixCacheSequenceBinding *prefix_bindings;
	SparkGlm52RequestApiSlot *request_slots;
	SparkGlm52ServingRequestRecord *request_records;
	uint32_t *request_token_storage;
	SparkGlm52ServingEvent *serving_events;
	uint32_t *host_prefill_token_ids;
	uint32_t *host_physical_block_indices;
	uint32_t *lane_physical_block_counts;
	SparkGlm52ServiceClientSession *client_sessions;
	SparkGlm52ServiceRequestMap *request_maps;
	SparkGlm52ServiceEvent *service_events;
	int32_t final_event_listen_fd;
	int32_t final_event_socket_fd;
	uint8_t final_event_read_buffer[sizeof(SparkGlm52Pp13ServiceBackendFinalEvent)];
	uint32_t final_event_read_offset;
	uint64_t final_event_receive_count;
	uint64_t final_event_receive_error_count;
	uint32_t last_final_event_token_count;
	uint32_t last_final_event_token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
	uint32_t initialized;
	uint32_t tokenizer_ready;
	uint32_t rank0_runtime_ready;
	uint32_t service_runtime_ready;
	char first_blocker[SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES];
} SparkGlm52Pp13ServiceBackendState;

static SparkGlm52Pp13ServiceBackendState SparkGlm52Pp13ServiceBackendSingleton;

static SparkStatus SparkGlm52Pp13ServiceBackendWaitForDecodeFinalEvents(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result);

static void SparkGlm52Pp13ServiceBackendSetBlocker(
	SparkGlm52Pp13ServiceBackendState *state,
	const char *message)
{
	if (state == 0 || state->first_blocker[0] != '\0')
		return;
	if (message == 0)
		message = "unknown PP13 service backend blocker";
	snprintf(state->first_blocker,sizeof(state->first_blocker),"%s",message);
}

static SparkStatus SparkGlm52Pp13ServiceBackendRequireText(
	const char *value,
	SparkGlm52Pp13ServiceBackendState *state,
	const char *message)
{
	if (value == 0 || value[0] == '\0')
	{
		SparkGlm52Pp13ServiceBackendSetBlocker(state,message);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13ServiceBackendAlloc(
	void **pointer,
	uint64_t count,
	uint64_t element_bytes)
{
	if (pointer == 0 || count == 0u || element_bytes == 0u ||
		count > UINT64_MAX / element_bytes)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*pointer = calloc((size_t)count,(size_t)element_bytes);
	if (*pointer == 0)
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13ServiceBackendFreeStorage(
	SparkGlm52Pp13ServiceBackendState *state)
{
	if (state == 0)
		return;
	free(state->kv_blocks);
	free(state->prefix_entries);
	free(state->prefix_bindings);
	free(state->request_slots);
	free(state->request_records);
	free(state->request_token_storage);
	free(state->serving_events);
	free(state->host_prefill_token_ids);
	free(state->host_physical_block_indices);
	free(state->lane_physical_block_counts);
	free(state->client_sessions);
	free(state->request_maps);
	free(state->service_events);
	state->kv_blocks = 0;
	state->prefix_entries = 0;
	state->prefix_bindings = 0;
	state->request_slots = 0;
	state->request_records = 0;
	state->request_token_storage = 0;
	state->serving_events = 0;
	state->host_prefill_token_ids = 0;
	state->host_physical_block_indices = 0;
	state->lane_physical_block_counts = 0;
	state->client_sessions = 0;
	state->request_maps = 0;
	state->service_events = 0;
}

static uint32_t SparkGlm52Pp13ServiceBackendMaxActive(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	if (configuration->max_active_sequence_count != 0u)
		return configuration->max_active_sequence_count;
	return SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_MAX_ACTIVE;
}

static uint32_t SparkGlm52Pp13ServiceBackendServiceLaneCapacity(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	uint32_t max_active;

	max_active = SparkGlm52Pp13ServiceBackendMaxActive(configuration);
	if (max_active > SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY)
		return SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY;
	return max_active;
}

static uint32_t SparkGlm52Pp13ServiceBackendPortBase(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	if (configuration->port_base != 0u)
		return configuration->port_base;
	return SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_PORT_BASE;
}

static SparkStatus SparkGlm52Pp13ServiceBackendLoadTokenizer(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	SparkTokenizerCompiledFileConfiguration tokenizer_configuration;
	SparkStatus status;

	status = SparkGlm52Pp13ServiceBackendRequireText(
		configuration->tokenizer_path,
		state,
		"compiled C tokenizer path is missing");
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&tokenizer_configuration,0,sizeof(tokenizer_configuration));
	tokenizer_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
	tokenizer_configuration.descriptor_bytes =
		SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
	tokenizer_configuration.compiled_tokenizer_path =
		configuration->tokenizer_path;
	status = SparkTokenizerLoadCompiledFile(
		&state->tokenizer,
		&tokenizer_configuration);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52Pp13ServiceBackendSetBlocker(
			state,
			"compiled C tokenizer failed to load");
		return status;
	}
	state->tokenizer_ready = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13ServiceBackendPrefill(
	void *context,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	SparkGlm52Pp13ServiceBackendState *state;

	state = (SparkGlm52Pp13ServiceBackendState *)context;
	if (state == 0 || prefill_dispatch == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->builder_library.builder_interface.prefill == 0 ||
		state->builder_state == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	return state->builder_library.builder_interface.prefill(
		state->builder_state,
		prefill_dispatch);
}

static SparkStatus SparkGlm52Pp13ServiceBackendDecode(
	void *context,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result)
{
	SparkGlm52Pp13ServiceBackendState *state;
	SparkStatus status;

	state = (SparkGlm52Pp13ServiceBackendState *)context;
	if (state == 0 || decode_dispatch == 0 || decode_result == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->builder_library.builder_interface.decode == 0 ||
		state->builder_state == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	status = state->builder_library.builder_interface.decode(
		state->builder_state,
		decode_dispatch,
		decode_result);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13ServiceBackendWaitForDecodeFinalEvents(
		state,
		decode_dispatch,
		decode_result);
}

static SparkStatus SparkGlm52Pp13ServiceBackendAllocateCacheStorage(
	SparkGlm52Pp13ServiceBackendState *state)
{
	SparkStatus status;

	status = SparkGlm52Pp13ServiceBackendAlloc(
		(void **)&state->kv_blocks,
		SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT,
		sizeof(state->kv_blocks[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->prefix_entries,
			SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT,
			sizeof(state->prefix_entries[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->prefix_bindings,
			SPARK_GLM52_PP13_SERVICE_BACKEND_PREFIX_BINDING_COUNT,
			sizeof(state->prefix_bindings[0]));
	return status;
}

static SparkStatus SparkGlm52Pp13ServiceBackendAllocateRequestStorage(
	SparkGlm52Pp13ServiceBackendState *state,
	uint32_t lane_capacity)
{
	SparkStatus status;

	status = SparkGlm52Pp13ServiceBackendAlloc(
		(void **)&state->request_slots,
		SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY,
		sizeof(state->request_slots[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->request_records,
			SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY,
			sizeof(state->request_records[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->request_token_storage,
			(uint64_t)SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY *
				SPARK_GLM52_PP13_SERVICE_BACKEND_CONTEXT_TOKENS,
			sizeof(state->request_token_storage[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->serving_events,
			SPARK_GLM52_PP13_SERVICE_BACKEND_EVENT_CAPACITY,
			sizeof(state->serving_events[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->host_prefill_token_ids,
			(uint64_t)lane_capacity *
				SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_TOKENS,
			sizeof(state->host_prefill_token_ids[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->host_physical_block_indices,
			(uint64_t)lane_capacity *
				SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT,
			sizeof(state->host_physical_block_indices[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->lane_physical_block_counts,
			lane_capacity,
			sizeof(state->lane_physical_block_counts[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->client_sessions,
			SPARK_GLM52_PP13_SERVICE_BACKEND_CLIENT_CAPACITY,
			sizeof(state->client_sessions[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->request_maps,
			SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_MAP_CAPACITY,
			sizeof(state->request_maps[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAlloc(
			(void **)&state->service_events,
			SPARK_GLM52_PP13_SERVICE_BACKEND_EVENT_CAPACITY,
			sizeof(state->service_events[0]));
	return status;
}

static SparkStatus SparkGlm52Pp13ServiceBackendAllocateServiceStorage(
	SparkGlm52Pp13ServiceBackendState *state,
	uint32_t lane_capacity)
{
	SparkStatus status;

	status = SparkGlm52Pp13ServiceBackendAllocateCacheStorage(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendAllocateRequestStorage(
			state,
			lane_capacity);
	if (status != SPARK_STATUS_OK)
		SparkGlm52Pp13ServiceBackendFreeStorage(state);
	return status;
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeKvArena(
	SparkGlm52Pp13ServiceBackendState *state)
{
	SparkGlm52KvCacheConfiguration kv_configuration;

	memset(&kv_configuration,0,sizeof(kv_configuration));
	kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	kv_configuration.descriptor_bytes =
		SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	kv_configuration.physical_block_count =
		SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT;
	kv_configuration.block_token_count =
		SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
	kv_configuration.layer_count = SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
	kv_configuration.kv_head_count = 8u;
	kv_configuration.head_dim = 128u;
	kv_configuration.bytes_per_scalar = 2u;
	kv_configuration.key_device_base =
		(void *)(uintptr_t)SPARK_GLM52_PP13_SERVICE_BACKEND_METADATA_KEY_BASE;
	kv_configuration.value_device_base =
		(void *)(uintptr_t)SPARK_GLM52_PP13_SERVICE_BACKEND_METADATA_VALUE_BASE;
	kv_configuration.blocks = state->kv_blocks;
	return SparkGlm52KvCacheArenaInitialize(
		&state->kv_arena,
		&kv_configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializePrefixCache(
	SparkGlm52Pp13ServiceBackendState *state)
{
	SparkGlm52PrefixCacheConfiguration prefix_configuration;

	memset(&prefix_configuration,0,sizeof(prefix_configuration));
	prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
	prefix_configuration.descriptor_bytes =
		SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	prefix_configuration.block_token_count =
		SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
	prefix_configuration.entry_count =
		SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT;
	prefix_configuration.physical_block_count =
		SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT;
	prefix_configuration.sequence_binding_count =
		SPARK_GLM52_PP13_SERVICE_BACKEND_PREFIX_BINDING_COUNT;
	prefix_configuration.entries = state->prefix_entries;
	prefix_configuration.sequence_bindings = state->prefix_bindings;
	prefix_configuration.kv_cache_arena = &state->kv_arena;
	return SparkGlm52PrefixCacheInitialize(
		&state->prefix_cache,
		&prefix_configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeScheduler(
	SparkGlm52Pp13ServiceBackendState *state)
{
	SparkGlm52SchedulerConfiguration scheduler_configuration;

	memset(&scheduler_configuration,0,sizeof(scheduler_configuration));
	scheduler_configuration.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
	scheduler_configuration.descriptor_bytes =
		SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
	scheduler_configuration.spark_count = SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT;
	scheduler_configuration.queue_depth_per_spark = 2u;
	scheduler_configuration.measured_profile_id =
		SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
	scheduler_configuration.quantization_mode =
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
	scheduler_configuration.max_prefill_tokens_per_step =
		SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_TOKENS;
	scheduler_configuration.prefix_cache = &state->prefix_cache;
	return SparkGlm52SchedulerInitialize(
		&state->scheduler,
		&scheduler_configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeRequestApi(
	SparkGlm52Pp13ServiceBackendState *state,
	uint32_t lane_capacity)
{
	SparkGlm52RequestApiConfiguration request_api_configuration;

	memset(&request_api_configuration,0,sizeof(request_api_configuration));
	request_api_configuration.abi_version =
		SPARK_GLM52_REQUEST_API_ABI_VERSION;
	request_api_configuration.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
	request_api_configuration.configuration_flags =
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT;
	request_api_configuration.request_capacity =
		SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY;
	request_api_configuration.prefetch_lane_count =
		SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT;
	request_api_configuration.decode_batch_target = lane_capacity;
	request_api_configuration.scheduler = &state->scheduler;
	request_api_configuration.request_slots = state->request_slots;
	return SparkGlm52RequestApiInitialize(
		&state->request_api,
		&request_api_configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeServingEngine(
	SparkGlm52Pp13ServiceBackendState *state,
	uint32_t lane_capacity)
{
	SparkGlm52ServingEngineConfiguration serving_configuration;

	memset(&serving_configuration,0,sizeof(serving_configuration));
	serving_configuration.abi_version =
		SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
	serving_configuration.descriptor_bytes =
		SPARK_GLM52_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES;
	serving_configuration.runtime_contract_flags =
		SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS;
	serving_configuration.default_output_token_budget = 1024u;
	serving_configuration.default_max_prefill_tokens_per_step =
		SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_TOKENS;
	serving_configuration.max_context_tokens =
		SPARK_GLM52_PP13_SERVICE_BACKEND_CONTEXT_TOKENS;
	serving_configuration.request_api = &state->request_api;
	serving_configuration.tokenizer = &state->tokenizer;
	serving_configuration.request_records = state->request_records;
	serving_configuration.request_record_capacity =
		SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_CAPACITY;
	serving_configuration.request_token_storage =
		state->request_token_storage;
	serving_configuration.request_token_stride =
		SPARK_GLM52_PP13_SERVICE_BACKEND_CONTEXT_TOKENS;
	serving_configuration.event_ring = state->serving_events;
	serving_configuration.event_ring_capacity =
		SPARK_GLM52_PP13_SERVICE_BACKEND_EVENT_CAPACITY;
	serving_configuration.host_prefill_token_ids =
		state->host_prefill_token_ids;
	serving_configuration.host_prefill_token_stride =
		SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_TOKENS;
	serving_configuration.host_prefill_lane_capacity = lane_capacity;
	serving_configuration.host_physical_block_indices =
		state->host_physical_block_indices;
	serving_configuration.kv_block_lane_stride =
		SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT;
	serving_configuration.kv_block_lane_capacity =
		SPARK_GLM52_PP13_SERVICE_BACKEND_KV_BLOCK_COUNT;
	serving_configuration.lane_physical_block_counts =
		state->lane_physical_block_counts;
	serving_configuration.lane_count_capacity = lane_capacity;
	serving_configuration.prefill_function =
		SparkGlm52Pp13ServiceBackendPrefill;
	serving_configuration.decode_function = SparkGlm52Pp13ServiceBackendDecode;
	serving_configuration.callback_context = state;
	return SparkGlm52ServingEngineInitialize(
		&state->serving_engine,
		&serving_configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeService(
	SparkGlm52Pp13ServiceBackendState *state)
{
	SparkGlm52ServiceConfiguration service_configuration;

	memset(&service_configuration,0,sizeof(service_configuration));
	service_configuration.abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
	service_configuration.descriptor_bytes =
		SPARK_GLM52_SERVICE_CONFIGURATION_DESCRIPTOR_BYTES;
	service_configuration.serving_engine = &state->serving_engine;
	service_configuration.client_sessions = state->client_sessions;
	service_configuration.client_session_capacity =
		SPARK_GLM52_PP13_SERVICE_BACKEND_CLIENT_CAPACITY;
	service_configuration.request_maps = state->request_maps;
	service_configuration.request_map_capacity =
		SPARK_GLM52_PP13_SERVICE_BACKEND_REQUEST_MAP_CAPACITY;
	service_configuration.event_ring = state->service_events;
	service_configuration.event_ring_capacity =
		SPARK_GLM52_PP13_SERVICE_BACKEND_EVENT_CAPACITY;
	return SparkGlm52ServiceInitialize(
		&state->service,
		&service_configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeServiceRuntime(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	uint32_t lane_capacity;
	SparkStatus status;

	lane_capacity =
		SparkGlm52Pp13ServiceBackendServiceLaneCapacity(configuration);
	if (lane_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13ServiceBackendAllocateServiceStorage(
		state,
		lane_capacity);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializeKvArena(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializePrefixCache(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializeScheduler(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializeRequestApi(
			state,
			lane_capacity);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializeServingEngine(
			state,
			lane_capacity);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializeService(state);
	if (status == SPARK_STATUS_OK)
		state->service_runtime_ready = 1u;
	return status;
}

static int32_t SparkGlm52Pp13ServiceBackendCreateListenSocket(
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

static int32_t SparkGlm52Pp13ServiceBackendSetNonblocking(int32_t fd)
{
	int32_t flags;

	flags = fcntl(fd,F_GETFL,0);
	if (flags < 0)
		return -1;
	if (fcntl(fd,F_SETFL,(flags | O_NONBLOCK)) < 0)
		return -2;
	return 0;
}

static SparkStatus SparkGlm52Pp13ServiceBackendLoadDriver(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration,
	char *error_buffer,
	uint32_t error_buffer_bytes)
{
	SparkModelDriverCreateRequest create_request;
	SparkStatus status;

	status = SparkLoadModelDriver(
		configuration->driver_shared_object_path,
		configuration->node_target,
		&state->loaded_driver,
		error_buffer,
		error_buffer_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	state->program = SparkFindLoadedModelDriverProgram(
		&state->loaded_driver,
		configuration->driver_program_name);
	if (state->program == 0)
		return SPARK_STATUS_NOT_FOUND;
	memset(&create_request,0,sizeof(create_request));
	create_request.node_id = state->rank_plan.host_name;
	create_request.node_target = configuration->node_target;
	create_request.node_context = state->builder_result.node_context;
	status = state->loaded_driver.interface->create(
		&create_request,
		&state->driver_instance);
	if (status != SPARK_STATUS_OK)
		return status;
	if (state->driver_instance == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13ServiceBackendBuildNodeContext(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	SparkGlm52Pp13NodeContextBuilderConfiguration builder_configuration;
	SparkStatus status;

	memset(&builder_configuration,0,sizeof(builder_configuration));
	builder_configuration.abi_version =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION;
	builder_configuration.descriptor_bytes =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES;
	builder_configuration.rank_index = state->rank_plan.rank_index;
	builder_configuration.max_active_sequence_count =
		SparkGlm52Pp13ServiceBackendMaxActive(configuration);
	builder_configuration.port_base =
		SparkGlm52Pp13ServiceBackendPortBase(configuration);
	builder_configuration.fp8_pack_root = configuration->fp8_pack_root;
	builder_configuration.embedding_pack_path =
		configuration->embedding_pack_path;
	builder_configuration.node_target = configuration->node_target;
	builder_configuration.rank_plan = &state->rank_plan;
	status = SparkGlm52Pp13NodeContextBuilderLoadInterfaceFromSharedObject(
		configuration->node_context_builder_shared_object_path,
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
		&state->builder_library);
	if (status != SPARK_STATUS_OK)
		return status;
	status = state->builder_library.builder_interface.initialize(
		&builder_configuration,
		&state->builder_state);
	if (status != SPARK_STATUS_OK || state->builder_state == 0)
		return status == SPARK_STATUS_OK ?
			SPARK_STATUS_INVALID_ARGUMENT : status;
	memset(&state->builder_result,0,sizeof(state->builder_result));
	status = state->builder_library.builder_interface.build(
		state->builder_state,
		&state->builder_result);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13NodeContextBuilderValidateResult(
		&state->builder_result,
		&state->rank_plan);
}

static SparkStatus SparkGlm52Pp13ServiceBackendAttachBuilderDriver(
	SparkGlm52Pp13ServiceBackendState *state)
{
	if (state == 0 || state->builder_state == 0 ||
		state->builder_library.builder_interface.attach_driver == 0 ||
		state->loaded_driver.interface == 0 ||
		state->driver_instance == 0 ||
		state->program == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return state->builder_library.builder_interface.attach_driver(
		state->builder_state,
		state->loaded_driver.interface,
		state->driver_instance,
		state->program,
		state->output_transport_session);
}

static SparkStatus SparkGlm52Pp13ServiceBackendRank0Fail(
	SparkGlm52Pp13ServiceBackendState *state,
	SparkStatus status,
	const char *error_buffer,
	const char *message)
{
	if (error_buffer != 0 && error_buffer[0] != '\0')
		SparkGlm52Pp13ServiceBackendSetBlocker(state,error_buffer);
	else
		SparkGlm52Pp13ServiceBackendSetBlocker(state,message);
	return status;
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeRunner(
	SparkGlm52Pp13ServiceBackendState *state)
{
	SparkGlm52ResidentDecodeStageProductionRunnerConfiguration configuration;

	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
	configuration.flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
	configuration.driver_interface = state->loaded_driver.interface;
	configuration.driver_instance = state->driver_instance;
	configuration.program = state->program;
	configuration.execution_stream = 0;
	return SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
		&state->runner,
		&configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeRank0(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	char error_buffer[SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES];
	SparkStatus status;

	error_buffer[0] = '\0';
	status = SparkGlm52Pp13RuntimeBuildRankPlan(
		0u,
		SparkGlm52Pp13ServiceBackendMaxActive(configuration),
		SparkGlm52Pp13ServiceBackendPortBase(configuration),
		&state->rank_plan,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build PP13 rank0 plan");
	status = SparkGlm52Pp13RuntimeBuildFinalEventRoute(
		SparkGlm52Pp13ServiceBackendPortBase(configuration),
		&state->final_event_route,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build final event route");
	status = SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
		&state->rank_plan,
		configuration->fp8_pack_root,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to validate rank0 FP8 packs");
	status = SparkHiddenTransportLoadInterfaceFromSharedObject(
		configuration->transport_shared_object_path,
		SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
		&state->transport_library);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to load production transport");
	status = SparkHiddenTransportOpen(
		&state->rank_plan.output_endpoint,
		&state->transport_library.transport_interface,
		SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
		&state->output_transport_session);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to open rank0 output transport");
	status = SparkGlm52Pp13ServiceBackendBuildNodeContext(state,configuration);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build rank0 resident context");
	status = SparkGlm52Pp13ServiceBackendLoadDriver(
		state,
		configuration,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to load GLM52 driver");
	status = SparkGlm52Pp13ServiceBackendAttachBuilderDriver(state);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to attach rank0 bridge driver");
	status = SparkGlm52Pp13ServiceBackendInitializeRunner(state);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to initialize rank0 runner");
	state->final_event_listen_fd =
		SparkGlm52Pp13ServiceBackendCreateListenSocket(
			configuration->final_event_bind_address,
			state->final_event_route.listen_port);
	if (state->final_event_listen_fd < 0)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,
			SPARK_STATUS_ROUTE_NOT_FOUND,
			error_buffer,
			"failed to open rank0 final-event listener");
	if (SparkGlm52Pp13ServiceBackendSetNonblocking(
			state->final_event_listen_fd) < 0)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,
			SPARK_STATUS_INTERNAL_ERROR,
			error_buffer,
			"failed to make final-event listener nonblocking");
	state->rank0_runtime_ready = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitialize(
	const SparkGlm52ServiceBackendConfiguration *configuration,
	void **backend_state)
{
	SparkGlm52Pp13ServiceBackendState *state;
	SparkStatus status;

	if (configuration == 0 || backend_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (configuration->abi_version != SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	state = &SparkGlm52Pp13ServiceBackendSingleton;
	memset(state,0,sizeof(*state));
	state->final_event_listen_fd = -1;
	state->final_event_socket_fd = -1;
	SparkLoadedModelDriverReset(&state->loaded_driver);
	SparkTokenizerReset(&state->tokenizer);
	state->initialized = 1u;
	*backend_state = state;
	status = SparkGlm52Pp13ServiceBackendRequireText(
		configuration->fp8_pack_root,
		state,
		"FP8 pack root is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->transport_shared_object_path,
			state,
			"production hidden transport shared object is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->driver_shared_object_path,
			state,
			"GLM52 driver shared object is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->node_context_builder_shared_object_path,
			state,
			"GLM52 PP13 node-context builder shared object is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->embedding_pack_path,
			state,
			"GLM52 embedding pack is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->driver_program_name,
			state,
			"GLM52 driver program name is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->final_event_bind_address,
			state,
			"final-event bind address is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->final_event_return_host,
			state,
			"final-event return host is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendLoadTokenizer(
			state,
			configuration);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializeServiceRuntime(
			state,
			configuration);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendInitializeRank0(
			state,
			configuration);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52Pp13ServiceBackendSetBlocker(
			state,
			"PP13 service runtime failed to initialize");
		return status;
	}
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13ServiceBackendDestroy(void *backend_state)
{
	SparkGlm52Pp13ServiceBackendState *state;

	state = (SparkGlm52Pp13ServiceBackendState *)backend_state;
	if (state == 0)
		return;
	if (state->final_event_socket_fd >= 0)
		close(state->final_event_socket_fd);
	if (state->final_event_listen_fd >= 0)
		close(state->final_event_listen_fd);
	if (state->loaded_driver.interface != 0 &&
		state->loaded_driver.interface->destroy != 0 &&
		state->driver_instance != 0)
		state->loaded_driver.interface->destroy(state->driver_instance);
	SparkUnloadModelDriver(&state->loaded_driver);
	if (state->builder_library.builder_interface.destroy_result != 0 &&
		state->builder_state != 0)
		state->builder_library.builder_interface.destroy_result(
			state->builder_state,
			&state->builder_result);
	if (state->builder_library.builder_interface.destroy != 0 &&
		state->builder_state != 0)
		state->builder_library.builder_interface.destroy(state->builder_state);
	SparkGlm52Pp13NodeContextBuilderUnloadInterface(&state->builder_library);
	SparkHiddenTransportClose(state->output_transport_session);
	SparkHiddenTransportUnloadInterface(&state->transport_library);
	SparkTokenizerDestroy(&state->tokenizer);
	SparkGlm52Pp13ServiceBackendFreeStorage(state);
	memset(state,0,sizeof(*state));
}

static SparkStatus SparkGlm52Pp13ServiceBackendGetView(
	void *backend_state,
	SparkGlm52ServiceBackendView *view)
{
	SparkGlm52Pp13ServiceBackendState *state;

	state = (SparkGlm52Pp13ServiceBackendState *)backend_state;
	if (state == 0 || view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION;
	view->descriptor_bytes = SPARK_GLM52_SERVICE_BACKEND_VIEW_BYTES;
	view->backend_ready = state->initialized != 0u ? 1u : 0u;
	view->pp13_ready = state->service_runtime_ready != 0u &&
		state->rank0_runtime_ready != 0u &&
		state->first_blocker[0] == '\0' ? 1u : 0u;
	view->max_context_tokens = SPARK_GLM52_PP13_SERVICE_BACKEND_CONTEXT_TOKENS;
	view->production_contract_flags =
		SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS;
	view->service = state->service_runtime_ready != 0u ? &state->service : 0;
	view->tokenizer = state->tokenizer_ready != 0u ? &state->tokenizer : 0;
	view->first_blocker = state->first_blocker;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13ServiceBackendAcceptFinalEventSocket(
	SparkGlm52Pp13ServiceBackendState *state)
{
	int32_t fd;

	if (state->final_event_listen_fd < 0 ||
		state->final_event_socket_fd >= 0)
		return;
	fd = accept(state->final_event_listen_fd,0,0);
	if (fd < 0)
	{
		if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
			state->final_event_receive_error_count += 1u;
		return;
	}
	if (SparkGlm52Pp13ServiceBackendSetNonblocking(fd) < 0)
	{
		close(fd);
		state->final_event_receive_error_count += 1u;
		return;
	}
	state->final_event_socket_fd = fd;
}

static void SparkGlm52Pp13ServiceBackendRecordFinalEvent(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52Pp13ServiceBackendFinalEvent *event)
{
	uint32_t token_count;

	if (event->magic != SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_MAGIC ||
		event->descriptor_bytes != (uint32_t)sizeof(*event))
	{
		state->final_event_receive_error_count += 1u;
		return;
	}
	if ((event->completion_flags & SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u)
	{
		state->final_event_receive_count += 1u;
		state->last_final_event_token_count = 0u;
		return;
	}
	token_count = event->token_count;
	if (token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY)
		token_count = SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
	memcpy(state->last_final_event_token_ids,event->token_ids,
		token_count * sizeof(state->last_final_event_token_ids[0u]));
	state->last_final_event_token_count = token_count;
	state->final_event_receive_count += 1u;
}

static SparkStatus SparkGlm52Pp13ServiceBackendReadFinalEvent(
	SparkGlm52Pp13ServiceBackendState *state,
	SparkGlm52Pp13ServiceBackendFinalEvent *event_out)
{
	ssize_t got;
	uint32_t remaining;

	if (state == 0 || event_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkGlm52Pp13ServiceBackendAcceptFinalEventSocket(state);
	if (state->final_event_socket_fd < 0)
		return SPARK_STATUS_BUSY;
	remaining = ((uint32_t)sizeof(*event_out)) -
		state->final_event_read_offset;
	got = read(
		state->final_event_socket_fd,
		&state->final_event_read_buffer[state->final_event_read_offset],
		remaining);
	if (got < 0)
	{
		if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
			return SPARK_STATUS_BUSY;
		close(state->final_event_socket_fd);
		state->final_event_socket_fd = -1;
		state->final_event_read_offset = 0u;
		state->final_event_receive_error_count += 1u;
		return SPARK_STATUS_IO_ERROR;
	}
	if (got == 0)
	{
		close(state->final_event_socket_fd);
		state->final_event_socket_fd = -1;
		state->final_event_read_offset = 0u;
		return SPARK_STATUS_BUSY;
	}
	state->final_event_read_offset += (uint32_t)got;
	if (state->final_event_read_offset != (uint32_t)sizeof(*event_out))
		return SPARK_STATUS_BUSY;
	memcpy(event_out,state->final_event_read_buffer,sizeof(*event_out));
	state->final_event_read_offset = 0u;
	return SPARK_STATUS_OK;
}

static int32_t SparkGlm52Pp13ServiceBackendFindDecodeLane(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	const SparkGlm52Pp13ServiceBackendFinalEvent *event)
{
	uint32_t lane_index;

	if (decode_dispatch == 0 || decode_dispatch->request_dispatch == 0 ||
		event == 0)
		return -1;
	for (lane_index = 0u;
		 lane_index < decode_dispatch->request_count;
		 ++lane_index)
	{
		if (decode_dispatch->request_dispatch->request_ids[lane_index] ==
				event->request_id &&
			decode_dispatch->request_dispatch->sequence_ids[lane_index] ==
				event->sequence_id)
			return (int32_t)lane_index;
	}
	return -2;
}

static SparkStatus SparkGlm52Pp13ServiceBackendApplyDecodeFinalEvent(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	const SparkGlm52Pp13ServiceBackendFinalEvent *event,
	SparkGlm52ServingDecodeResult *decode_result,
	uint8_t *lane_done,
	uint32_t *done_count)
{
	uint32_t lane_index;
	uint32_t token_count;
	int32_t lane;

	if (state == 0 || decode_dispatch == 0 || event == 0 ||
		decode_result == 0 || lane_done == 0 || done_count == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkGlm52Pp13ServiceBackendRecordFinalEvent(state,event);
	if (event->magic != SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_MAGIC ||
		event->descriptor_bytes != (uint32_t)sizeof(*event))
		return SPARK_STATUS_VALIDATION_FAILED;
	if (event->status != (uint32_t)SPARK_STATUS_OK)
		return SPARK_STATUS_VALIDATION_FAILED;
	if ((event->completion_flags &
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u)
		return SPARK_STATUS_VALIDATION_FAILED;
	lane = SparkGlm52Pp13ServiceBackendFindDecodeLane(
		decode_dispatch,
		event);
	if (lane < 0)
		return SPARK_STATUS_OK;
	lane_index = (uint32_t)lane;
	if (lane_done[lane_index] != 0u)
		return SPARK_STATUS_DUPLICATE;
	token_count = event->token_count;
	if (token_count == 0u ||
		token_count > SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE ||
		token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY)
		return SPARK_STATUS_VALIDATION_FAILED;
	memcpy(
		decode_result->token_ids[lane_index],
		event->token_ids,
		token_count * sizeof(decode_result->token_ids[lane_index][0u]));
	decode_result->token_counts[lane_index] = token_count;
	lane_done[lane_index] = 1u;
	*done_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13ServiceBackendPollFinalEvent(
	SparkGlm52Pp13ServiceBackendState *state)
{
	struct pollfd poll_fds[1];
	int32_t fd;
	int32_t result;

	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->final_event_socket_fd >= 0)
		fd = state->final_event_socket_fd;
	else
		fd = state->final_event_listen_fd;
	if (fd < 0)
		return SPARK_STATUS_ROUTE_NOT_FOUND;
	memset(poll_fds,0,sizeof(poll_fds));
	poll_fds[0].fd = fd;
	poll_fds[0].events = POLLIN;
	result = poll(
		poll_fds,
		1u,
		SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_POLL_MS);
	if (result < 0 && errno != EINTR)
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_BUSY;
}

static SparkStatus SparkGlm52Pp13ServiceBackendWaitForDecodeFinalEvents(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result)
{
	SparkGlm52Pp13ServiceBackendFinalEvent event;
	uint8_t lane_done[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
	uint32_t done_count;
	uint32_t poll_count;
	SparkStatus status;

	if (state == 0 || decode_dispatch == 0 || decode_result == 0 ||
		decode_dispatch->request_dispatch == 0 ||
		decode_dispatch->request_count == 0u ||
		decode_dispatch->request_count >
			SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(lane_done,0,sizeof(lane_done));
	done_count = 0u;
	state->last_final_event_token_count = 0u;
	for (poll_count = 0u;
		 poll_count < SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_POLL_COUNT;
		 ++poll_count)
	{
		status = SparkGlm52Pp13ServiceBackendReadFinalEvent(state,&event);
		if (status == SPARK_STATUS_OK)
		{
			status = SparkGlm52Pp13ServiceBackendApplyDecodeFinalEvent(
				state,
				decode_dispatch,
				&event,
				decode_result,
				lane_done,
				&done_count);
			if (status != SPARK_STATUS_OK)
				return status;
			if (done_count == decode_dispatch->request_count)
				return SPARK_STATUS_OK;
			continue;
		}
		if (status != SPARK_STATUS_BUSY)
			return status;
		status = SparkGlm52Pp13ServiceBackendPollFinalEvent(state);
		if (status != SPARK_STATUS_BUSY && status != SPARK_STATUS_OK)
			return status;
	}
	state->final_event_receive_error_count += 1u;
	return SPARK_STATUS_BUSY;
}

static SparkStatus SparkGlm52Pp13ServiceBackendPump(
	void *backend_state,
	uint32_t max_dispatch_steps,
	SparkGlm52ServiceStats *stats_out)
{
	SparkGlm52Pp13ServiceBackendState *state;
	SparkGlm52Pp13ServiceBackendFinalEvent event;
	SparkStatus event_status;

	state = (SparkGlm52Pp13ServiceBackendState *)backend_state;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	event_status = SparkGlm52Pp13ServiceBackendReadFinalEvent(state,&event);
	if (event_status == SPARK_STATUS_OK)
		SparkGlm52Pp13ServiceBackendRecordFinalEvent(state,&event);
	else if (event_status != SPARK_STATUS_BUSY)
		state->final_event_receive_error_count += 1u;
	if (stats_out != 0)
	{
		if (state->service_runtime_ready != 0u)
			(void)SparkGlm52ServiceGetStats(&state->service,stats_out);
		else
			memset(stats_out,0,sizeof(*stats_out));
	}
	if (state->service_runtime_ready != 0u &&
		state->rank0_runtime_ready != 0u &&
		state->first_blocker[0] == '\0')
		return SparkGlm52ServicePump(
			&state->service,
			max_dispatch_steps,
			stats_out);
	(void)max_dispatch_steps;
	return SPARK_STATUS_OK;
}

static const SparkGlm52ServiceBackendInterface SparkGlm52Pp13ServiceBackendInterface =
{
	SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION,
	SPARK_GLM52_SERVICE_BACKEND_INTERFACE_BYTES,
	SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_PP13_RUNTIME |
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_SERVICE_RUNTIME |
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_TOKENIZER,
	0u,
	SparkGlm52Pp13ServiceBackendInitialize,
	SparkGlm52Pp13ServiceBackendDestroy,
	SparkGlm52Pp13ServiceBackendGetView,
	SparkGlm52Pp13ServiceBackendPump
};

const SparkGlm52ServiceBackendInterface *SparkGlm52ServiceBackendGetInterface(void)
{
	return &SparkGlm52Pp13ServiceBackendInterface;
}
