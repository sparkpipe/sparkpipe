#define _POSIX_C_SOURCE 200112L

// The per-rank serving backend: sockets, event loop, decode lanes, driver
// completion. Named ring_service_backend and containing no dlopen, no plugin
// loading and nothing about thirteen - it is the thing that connects a socket to
// a driver, for however many ranks there are.
//
// Two model values reach it and both are configuration rather than architecture:
// the prefill dispatch ceiling and the EOS token set. They arrive through the
// model description rather than a header, so a second model needs no second
// backend.
#include "sparkpipe/spark_glm52_service_backend.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <sys/un.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_ring_node_context_builder.h"
#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_ring_runtime.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"
#include "sparkpipe/spark_model_driver.h"
#include "spark_filesystem.h"

#define SPARK_GLM52_RING_SERVICE_BACKEND_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_GLM52_RING_SERVICE_BACKEND_DEFAULT_PORT_BASE 52100u
#define SPARK_GLM52_RING_SERVICE_BACKEND_PIPELINE_COHORT_CAPACITY \
	SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_RESERVE_CAPACITY 1u
#define SPARK_GLM52_RING_SERVICE_BACKEND_QUEUE_DEPTH_PER_SPARK \
	(SPARK_GLM52_RING_SERVICE_BACKEND_PIPELINE_COHORT_CAPACITY + \
	 SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_RESERVE_CAPACITY)
#define SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY \
	SPARK_GLM52_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY
#define SPARK_GLM52_RING_SERVICE_BACKEND_CLIENT_CAPACITY \
	SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY
#define SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_MAP_CAPACITY \
	SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY
#include "sparkpipe/spark_glm52_kv_cache.h"
#define SPARK_GLM52_RING_SERVICE_BACKEND_CONTEXT_TOKENS SPARK_GLM52_KV_CONTEXT_TOKENS
#define SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_TOKENS \
	SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS
#define SPARK_GLM52_RING_SERVICE_BACKEND_MAX_BLOCKS_PER_SEQUENCE \
    (SPARK_GLM52_KV_CONTEXT_TOKENS / SPARK_GLM52_KV_BLOCK_TOKENS)
#define SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS
#define SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_WAVE_TOKENS \
	SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
#define SPARK_GLM52_RING_SERVICE_BACKEND_GPU_BLOCK_COUNT \
	(SPARK_GLM52_KV_POOL_TOKENS / \
	 SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS)
#define SPARK_GLM52_RING_SERVICE_BACKEND_DEFAULT_LOGICAL_BLOCK_COUNT \
	SPARK_GLM52_RING_SERVICE_BACKEND_GPU_BLOCK_COUNT
#define SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_COUNT \
	(SPARK_GLM52_RING_SERVICE_BACKEND_CONTEXT_TOKENS / \
	 SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS)
#define SPARK_GLM52_RING_SERVICE_BACKEND_PREFIX_BINDING_COUNT \
	(SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_COUNT + \
	 SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY)
#define SPARK_GLM52_RING_SERVICE_BACKEND_EVENT_CAPACITY 16384u
#define SPARK_GLM52_RING_SERVICE_BACKEND_METADATA_KEY_BASE 0x100000000ull
#define SPARK_GLM52_RING_SERVICE_BACKEND_METADATA_VALUE_BASE 0x200000000ull
#define SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY \
	SPARK_GLM52_RING_SERVICE_BACKEND_PIPELINE_COHORT_CAPACITY
#define SPARK_GLM52_RING_SERVICE_BACKEND_FINAL_EVENT_PUMP_BUDGET \
	(SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY * \
	 SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
#define SPARK_GLM52_RING_SERVICE_BACKEND_STATIC_STATE_CAPACITY_BYTES \
	(64ull * 1024ull * 1024ull)
#define SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_FREE 0u
#define SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE 1u
#define SPARK_GLM52_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY \
	(SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS * 2u)
#define SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY \
	SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY
#define SPARK_GLM52_RING_SERVICE_BACKEND_PATH_BYTES 4096u
#define SPARK_GLM52_RING_SERVICE_BACKEND_INITIAL_DECODE_PAYLOAD_BYTES \
	(SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES + \
	 (SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT * 256u * \
	  (uint32_t)sizeof(uint32_t)))

typedef struct SparkGlm52RingServiceBackendPendingDecode
{
	uint32_t state;
	uint32_t done_count;
	uint64_t trace_submit_time_ns;
	uint8_t lane_done[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
	uint8_t dspark_draft_valid[
		SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
	SparkGlm52DsparkDraftResult dspark_drafts[
		SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
	SparkGlm52RequestApiDispatch dispatch;
	SparkGlm52ServingDecodeResult result;
} SparkGlm52RingServiceBackendPendingDecode;

typedef struct SparkGlm52RingServiceBackendReleaseRecord
{
	uint64_t request_id;
	uint64_t sequence_id;
	uint32_t token_count;
	uint32_t reserved0;
} SparkGlm52RingServiceBackendReleaseRecord;

typedef union SparkGlm52RingServiceBackendResidentPayload
{
	SparkGlm52CudaResidentIpcSubmitResult submit_result;
	SparkGlm52CudaResidentIpcCompletion completion;
	SparkGlm52CudaResidentIpcStats stats;
} SparkGlm52RingServiceBackendResidentPayload;

typedef struct SparkGlm52RingServiceBackendState
{
	SparkGlm52RingRuntimeRankPlan rank_plan;
	SparkGlm52RingRuntimeFinalEventRoute final_event_route;
	SparkHiddenTransportDynamicLibrary transport_library;
	SparkHiddenTransportSession *output_transport_session;
	SparkGlm52RingNodeContextBuilderDynamicLibrary builder_library;
	void *builder_state;
	SparkGlm52RingNodeContextBuilderResult builder_result;
	SparkLoadedModelDriver loaded_driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkGlm52ResidentDecodeStageProductionRunner runner;
	SparkTokenizer tokenizer;
	SparkGlm52KvCacheArena kv_arena;
	SparkGlm52PrefixCache prefix_cache;
	SparkGlm52Scheduler scheduler;
	SparkGlm52RequestApi request_api;
	SparkGlm52DsparkSpeculator dspark_speculator;
	SparkGlm52DsparkModelContract dspark_model_contract;
	SparkGlm52DsparkSequenceState *dspark_sequence_states;
	uint32_t dspark_enabled;
	uint32_t mtp_enabled;
	uint32_t kv_logical_block_capacity;
	uint32_t kv_physical_block_capacity;
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
	int32_t work_output_socket_fd;
	uint32_t work_output_socket_connecting;
	int32_t cuda_resident_fd;
	uint32_t cuda_resident_attached;
	uint32_t trace_enabled;
	uint64_t trace_last_decode_completion_ns;
	uint64_t cuda_resident_retry_after_ns;
	char cuda_resident_socket_path[108];
	uint64_t session_id_base;
	uint64_t cuda_resident_next_sequence_number;
	uint64_t cuda_resident_submit_count;
	uint64_t cuda_resident_completion_count;
	uint64_t cuda_resident_rejection_count;
	SparkGlm52CudaResidentIpcReader cuda_resident_reader;
	SparkGlm52RingServiceBackendResidentPayload cuda_resident_payload;
	uint32_t cuda_resident_submit_capacity;
	uint32_t cuda_resident_submit_credit;
	uint8_t *cuda_resident_decode_payload;
	uint32_t cuda_resident_decode_payload_capacity;
	int32_t final_event_listen_fd;
	int32_t final_event_socket_fd;
	uint8_t final_event_read_buffer[
		SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES];
	uint32_t final_event_read_offset;
	uint64_t final_event_receive_count;
	uint64_t final_event_receive_error_count;
	uint32_t last_final_event_token_count;
	uint32_t last_final_event_token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
	SparkGlm52RingWorkControlPacket work_queue[
		SPARK_GLM52_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY];
	uint32_t work_queue_head;
	uint32_t work_queue_count;
	uint32_t work_queue_write_offset;
	SparkGlm52RingServiceBackendPendingDecode pending_decodes[
		SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY];
	SparkGlm52RingRuntimeFinalEvent early_final_events[
		SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY];
	uint32_t early_final_event_head;
	uint32_t early_final_event_count;
	SparkGlm52RingServiceBackendReleaseRecord release_queue[
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY];
	uint32_t release_queue_head;
	uint32_t release_queue_count;
	uint32_t initialized;
	uint32_t tokenizer_ready;
	uint32_t rank0_runtime_ready;
	uint32_t service_runtime_ready;
	char transport_shared_object_path[
		SPARK_GLM52_RING_SERVICE_BACKEND_PATH_BYTES];
	char first_blocker[SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES];
} SparkGlm52RingServiceBackendState;

_Static_assert(
	SPARK_GLM52_RING_SERVICE_BACKEND_QUEUE_DEPTH_PER_SPARK ==
		SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY +
		SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_RESERVE_CAPACITY,
	"scheduler depth must hold every decode cohort plus prefill reserve");
_Static_assert(
	((SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY +
		SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT - 1u) /
		SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) <=
		SPARK_GLM52_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY,
	"work queue must hold a complete sequence-release drain");
_Static_assert(
	sizeof(SparkGlm52RingServiceBackendState) <=
		SPARK_GLM52_RING_SERVICE_BACKEND_STATIC_STATE_CAPACITY_BYTES,
	"service backend static state exceeds capacity");

static SparkGlm52RingServiceBackendState SparkGlm52RingServiceBackendSingleton;

static SparkStatus SparkGlm52RingServiceBackendStampWorkPacket(
	const SparkGlm52RingServiceBackendState *state,
	SparkGlm52RingWorkControlPacket *packet)
{
	if (state == 0 || packet == 0 || state->session_id_base == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	packet->control_generation = state->session_id_base;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendRegisterPendingDecode(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	const SparkGlm52ServingDecodeResult *decode_result,
	SparkGlm52RingServiceBackendPendingDecode **pending_out);
static SparkStatus SparkGlm52RingServiceBackendBuildDecodeWorkPacket(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_token_index,
	SparkGlm52RingWorkControlPacket *packet);
static SparkStatus SparkGlm52RingServiceBackendCompletePendingFinalEvent(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingRuntimeFinalEvent *event);
static SparkStatus SparkGlm52RingServiceBackendPumpWorkOutput(
	SparkGlm52RingServiceBackendState *state);
static SparkStatus SparkGlm52RingServiceBackendForwardPrefillWork(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch);
static SparkStatus SparkGlm52RingServiceBackendEnqueueWorkPacket(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingWorkControlPacket *packet);

static void SparkGlm52RingServiceBackendSetBlocker(
	SparkGlm52RingServiceBackendState *state,
	const char *message)
{
	if (state == 0 || state->first_blocker[0] != '\0')
		return;
	if (message == 0)
		message = "unknown RING service backend blocker";
	snprintf(state->first_blocker,sizeof(state->first_blocker),"%s",message);
}

static SparkStatus SparkGlm52RingServiceBackendRequireText(
	const char *value,
	SparkGlm52RingServiceBackendState *state,
	const char *message)
{
	if (value == 0 || value[0] == '\0')
	{
		SparkGlm52RingServiceBackendSetBlocker(state,message);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SPARK_STATUS_OK;
}

static const char *SparkGlm52RingServiceBackendEnvironmentText(
	const char *name)
{
	const char *value;
	value = getenv(name);
	return value != 0 ? value : "";
}

static uint64_t SparkGlm52RingServiceBackendEnvironmentU64(
	const char *name)
{
	const char *text;
	char *end;
	unsigned long long value;
	text = getenv(name);
	if (text == 0 || text[0] == '\0')
		return 0u;
	errno = 0;
	value = strtoull(text,&end,10);
	if (errno != 0 || end == text || *end != '\0')
		return 0u;
	return (uint64_t)value;
}

static SparkStatus SparkGlm52RingServiceBackendAlloc(
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

static SparkStatus SparkGlm52RingServiceBackendResidentWriteFull(int32_t fd, const void *buffer, uint32_t bytes)
{
	const uint8_t *cursor;
	uint32_t remaining;
	ssize_t written;
	cursor = (const uint8_t *)buffer;
	remaining = bytes;
	while (remaining != 0u)
	{
		written = write(fd,cursor,(size_t)remaining);
		if (written <= 0)
			return SPARK_STATUS_IO_ERROR;
		cursor += (uint32_t)written;
		remaining -= (uint32_t)written;
	}
	return SPARK_STATUS_OK;
}


static SparkStatus SparkGlm52RingServiceBackendResidentWriteMessage(SparkGlm52RingServiceBackendState *state, uint32_t kind, const void *payload, uint32_t payload_bytes)
{
	SparkGlm52CudaResidentIpcHeader header;
	SparkStatus status;
	if (state == 0 || state->cuda_resident_fd < 0)
		return SPARK_STATUS_ROUTE_NOT_FOUND;
	SparkGlm52CudaResidentIpcInitializeHeader(&header,kind,state->rank_plan.rank_index,state->cuda_resident_next_sequence_number++,payload_bytes);
	status = SparkGlm52RingServiceBackendResidentWriteFull(state->cuda_resident_fd,&header,sizeof(header));
	if (status != SPARK_STATUS_OK)
		return status;
	if (payload_bytes != 0u)
		status = SparkGlm52RingServiceBackendResidentWriteFull(state->cuda_resident_fd,payload,payload_bytes);
	return status;
}

static uint64_t SparkGlm52RingServiceBackendMonotonicNs(void)
{
	struct timespec timestamp;
	if (clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0)
		return 0u;
	return ((uint64_t)timestamp.tv_sec * 1000000000ull) + (uint64_t)timestamp.tv_nsec;
}

static SparkStatus SparkGlm52RingServiceBackendResidentReadBounded(int32_t fd, void *buffer, uint32_t buffer_bytes, uint32_t timeout_ms)
{
	struct pollfd descriptor;
	uint8_t *cursor;
	uint32_t offset;
	ssize_t got;
	if (fd < 0 || buffer == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	cursor = (uint8_t *)buffer;
	offset = 0u;
	descriptor.fd = fd;
	descriptor.events = POLLIN;
	while (offset < buffer_bytes)
	{
		descriptor.revents = 0;
		if (poll(&descriptor,1u,(int32_t)timeout_ms) <= 0)
			return SPARK_STATUS_BUSY;
		got = read(fd,cursor + offset,buffer_bytes - offset);
		if (got > 0)
		{
			offset += (uint32_t)got;
			continue;
		}
		if (got == 0)
			return SPARK_STATUS_ROUTE_NOT_FOUND;
		if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
			continue;
		return SPARK_STATUS_ROUTE_NOT_FOUND;
	}
	return SPARK_STATUS_OK;
}

static void SparkGlm52RingServiceBackendTeardownCudaResident(SparkGlm52RingServiceBackendState *state, const char *reason)
{
	if (state == 0 || state->cuda_resident_fd < 0)
		return;
	fprintf(stderr,"ring_resident_disconnected reason=%s\n",reason);
	close(state->cuda_resident_fd);
	state->cuda_resident_fd = -1;
	state->cuda_resident_submit_capacity = 0u;
	state->cuda_resident_submit_credit = 0u;
	SparkGlm52CudaResidentIpcReaderReset(&state->cuda_resident_reader);
}

static SparkStatus SparkGlm52RingServiceBackendConnectCudaResident(SparkGlm52RingServiceBackendState *state, const char *socket_path)
{
	struct sockaddr_un address;
	SparkGlm52CudaResidentIpcHello hello;
	SparkGlm52CudaResidentIpcHeader header;
	SparkGlm52CudaResidentIpcStats stats;
	SparkStatus status;
	uint32_t expected_moe_backend_kind;
	int32_t fd;
	if (state == 0 || socket_path == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingRuntimeExpectedMoeBackendKind(
		state->rank_plan.quantization_mode,
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
	state->cuda_resident_fd = fd;
	memset(&hello,0,sizeof(hello));
	hello.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_HELLO_BYTES;
	hello.rank_index = state->rank_plan.rank_index;
	hello.rank_count = SPARK_GLM52_RING_RUNTIME_STAGE_COUNT;
	hello.control_generation = state->session_id_base;
	hello.process_id = (uint64_t)getpid();
	status = SparkGlm52RingServiceBackendResidentWriteMessage(state,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO,&hello,sizeof(hello));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendResidentReadBounded(fd,&header,sizeof(header),5000u);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52CudaResidentIpcValidateHeader(&header,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_HELLO_ACK,sizeof(stats));
	if (status == SPARK_STATUS_OK && header.payload_bytes != sizeof(stats))
		status = SPARK_STATUS_ABI_MISMATCH;
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendResidentReadBounded(fd,&stats,sizeof(stats),5000u);
	if (status == SPARK_STATUS_OK && stats.state != SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_READY)
		status = SPARK_STATUS_BUSY;
	if (status == SPARK_STATUS_OK &&
		(stats.logical_lane_capacity != state->rank_plan.logical_lane_capacity ||
		 stats.execution_row_capacity != state->rank_plan.execution_row_capacity ||
		 stats.kv_physical_block_capacity == 0u ||
		 stats.kv_physical_block_capacity > stats.kv_logical_block_capacity ||
		 stats.kv_logical_block_capacity != state->kv_logical_block_capacity ||
		 stats.work_queue_capacity == 0u ||
		 stats.work_queue_depth > stats.work_queue_capacity ||
		 stats.model_quantization_mode != state->rank_plan.quantization_mode ||
		 stats.moe_backend_kind != expected_moe_backend_kind ||
		 stats.moe_bound_layer_count == 0u ||
		 stats.moe_bound_layer_count != stats.moe_expected_layer_count ||
		 SparkGlm52RingRuntimeValidateFp8PlanCounts(
			stats.model_quantization_mode,
			stats.fp8_scaled_gemm_bound_plan_count,
			stats.fp8_scaled_gemm_expected_plan_count) != SPARK_STATUS_OK ||
		 (stats.kv_nvme_enabled != 0u &&
		  stats.kv_nvme_mode !=
			SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_SYNCHRONOUS_FULL_HISTORY &&
		  stats.kv_nvme_mode !=
			SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT &&
		  stats.kv_nvme_mode !=
			SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT) ||
		 (state->rank_plan.logical_lane_capacity >=
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
			"ring_resident_contract_mismatch logical=%u/%u execution=%u/%u "
			"kv_blocks=%u logical_blocks=%u/%u quantization=%u/%u moe_backend=%u "
			"moe_layers=%u/%u fp8_scaled_gemm=%u/%u nvme=%u nvme_mode=%u "
			"blocker=%.*s\n",
			stats.logical_lane_capacity,
			state->rank_plan.logical_lane_capacity,
			stats.execution_row_capacity,
			state->rank_plan.execution_row_capacity,
			stats.kv_physical_block_capacity,
			stats.kv_logical_block_capacity,
			state->kv_logical_block_capacity,
			stats.model_quantization_mode,
			state->rank_plan.quantization_mode,
			stats.moe_backend_kind,
			stats.moe_bound_layer_count,
			stats.moe_expected_layer_count,
			stats.fp8_scaled_gemm_bound_plan_count,
			stats.fp8_scaled_gemm_expected_plan_count,
			stats.kv_nvme_enabled,
			stats.kv_nvme_mode,
			(int32_t)sizeof(stats.blocker),
			stats.blocker);
		status = SPARK_STATUS_MODULE_NOT_VALIDATED;
	}
	if (status != SPARK_STATUS_OK)
	{
		close(fd);
		state->cuda_resident_fd = -1;
		return status;
	}
	state->kv_physical_block_capacity = stats.kv_physical_block_capacity;
	state->cuda_resident_submit_capacity = stats.work_queue_capacity;
	state->cuda_resident_submit_credit =
		stats.work_queue_capacity - stats.work_queue_depth;
	SparkGlm52CudaResidentIpcReaderReset(&state->cuda_resident_reader);
	if (state->service_runtime_ready != 0u)
		state->request_api.max_resident_kv_block_count =
			stats.kv_physical_block_capacity;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendEnsureCudaResident(SparkGlm52RingServiceBackendState *state)
{
	uint64_t now_ns;
	SparkStatus status;
	if (state == 0 || state->cuda_resident_socket_path[0] == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->cuda_resident_fd >= 0)
		return SPARK_STATUS_OK;
	now_ns = SparkGlm52RingServiceBackendMonotonicNs();
	if (now_ns < state->cuda_resident_retry_after_ns)
		return SPARK_STATUS_BUSY;
	state->cuda_resident_retry_after_ns = now_ns + 250000000ull;
	status = SparkGlm52RingServiceBackendConnectCudaResident(state,state->cuda_resident_socket_path);
	if (status == SPARK_STATUS_OK)
	{
		fprintf(stderr,"ring_resident_connected\n");
		return SPARK_STATUS_OK;
	}
	fprintf(stderr,"ring_resident_connect_retry status=%u\n",(uint32_t)status);
	return SPARK_STATUS_BUSY;
}

static void SparkGlm52RingServiceBackendReleaseResidentSubmitCredit(
	SparkGlm52RingServiceBackendState *state)
{
	if (state != 0 &&
		state->cuda_resident_submit_credit <
			state->cuda_resident_submit_capacity)
		state->cuda_resident_submit_credit += 1u;
}

static SparkStatus SparkGlm52RingServiceBackendResidentReadMessage(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52CudaResidentIpcHeader *header_out,
	uint32_t timeout_ms)
{
	struct pollfd descriptor;
	SparkStatus status;
	if (state == 0 || header_out == 0 || state->cuda_resident_fd < 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (;;)
	{
		status = SparkGlm52CudaResidentIpcReadHeader(
			&state->cuda_resident_reader,
			state->cuda_resident_fd,
			(uint32_t)sizeof(state->cuda_resident_payload));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52CudaResidentIpcReadPayload(
				&state->cuda_resident_reader,
				state->cuda_resident_fd,
				(uint8_t *)&state->cuda_resident_payload,
				(uint32_t)sizeof(state->cuda_resident_payload));
		if (status == SPARK_STATUS_OK)
		{
			*header_out = state->cuda_resident_reader.header;
			SparkGlm52CudaResidentIpcReaderReset(
				&state->cuda_resident_reader);
			return SPARK_STATUS_OK;
		}
		if (status != SPARK_STATUS_BUSY || timeout_ms == 0u)
			return status;
		memset(&descriptor,0,sizeof(descriptor));
		descriptor.fd = state->cuda_resident_fd;
		descriptor.events = POLLIN;
		if (poll(&descriptor,1u,(int32_t)timeout_ms) <= 0)
			return SPARK_STATUS_BUSY;
		if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
			return SPARK_STATUS_ROUTE_NOT_FOUND;
	}
}

static SparkGlm52RingServiceBackendPendingDecode *SparkGlm52RingServiceBackendFindPendingDecodeForRequest(
	SparkGlm52RingServiceBackendState *state,
	uint64_t request_id)
{
	SparkGlm52RingServiceBackendPendingDecode *pending;
	uint32_t lane_index;
	uint32_t pending_index;
	if (state == 0 || request_id == 0u)
		return 0;
	for (pending_index = 0u;
		 pending_index <
			 SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY;
		 ++pending_index)
	{
		pending = &state->pending_decodes[pending_index];
		if (pending->state !=
			SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE)
			continue;
		for (lane_index = 0u;
			 lane_index < pending->dispatch.request_count;
			 ++lane_index)
		{
			if (pending->dispatch.request_ids[lane_index] == request_id)
				return pending;
		}
	}
	return 0;
}

static SparkStatus SparkGlm52RingServiceBackendFailPendingDecode(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52RingServiceBackendPendingDecode *pending,
	SparkStatus failure_status)
{
	SparkStatus route_status;
	SparkStatus status;
	uint32_t lane_index;
	if (state == 0 || pending == 0 || pending->dispatch.request_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RequestApiCancelDispatch(
		&state->request_api,
		&pending->dispatch);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u;
		 lane_index < pending->dispatch.request_count;
		 ++lane_index)
	{
		route_status = SparkGlm52ServingEngineFailRequestByRequestId(
			&state->serving_engine,
			pending->dispatch.request_ids[lane_index],
			failure_status);
		if (route_status != SPARK_STATUS_OK &&
			route_status != SPARK_STATUS_NOT_FOUND &&
			status == SPARK_STATUS_OK)
			status = route_status;
	}
	memset(pending,0,sizeof(*pending));
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendHandleResidentCompletion(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52CudaResidentIpcHeader *header)
{
	const SparkGlm52CudaResidentIpcCompletion *completion;
	if (state == 0 || header == 0 ||
		header->payload_bytes !=
			SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	completion = &state->cuda_resident_payload.completion;
	if (completion->descriptor_bytes !=
			SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_BYTES ||
		(completion->flags &
			~SPARK_GLM52_CUDA_RESIDENT_IPC_COMPLETION_KNOWN_FLAGS) != 0u)
		return SPARK_STATUS_ABI_MISMATCH;
	state->cuda_resident_completion_count += 1u;
	SparkGlm52RingServiceBackendReleaseResidentSubmitCredit(state);
	if (completion->completion.status != SPARK_STATUS_OK)
	{
		SparkGlm52RingServiceBackendPendingDecode *pending;
		SparkStatus failure_status;
		SparkStatus route_status;

		failure_status = (SparkStatus)completion->completion.status;
		fprintf(stderr,
			"ring_resident_work_failed status=%u request=%llu sequence=%llu\n",
			(uint32_t)failure_status,
			(unsigned long long)completion->completion.request_id,
			(unsigned long long)completion->completion.sequence_id);
		if (completion->completion.request_id == 0u)
			return failure_status;
		pending = SparkGlm52RingServiceBackendFindPendingDecodeForRequest(
			state,
			completion->completion.request_id);
		if (pending != 0)
			return SparkGlm52RingServiceBackendFailPendingDecode(
				state,
				pending,
				failure_status);
		route_status = SparkGlm52ServingEngineFailRequestByRequestId(
				&state->serving_engine,
				completion->completion.request_id,
				failure_status);
		if (route_status == SPARK_STATUS_OK)
			return SPARK_STATUS_OK;
		if (route_status != SPARK_STATUS_NOT_FOUND)
			return route_status;
		return failure_status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendHandleResidentSubmitResult(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52CudaResidentIpcHeader *header)
{
	const SparkGlm52CudaResidentIpcSubmitResult *result;
	char blocker[SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES];
	if (state == 0 || header == 0 ||
		header->payload_bytes !=
			SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	result = &state->cuda_resident_payload.submit_result;
	if (result->descriptor_bytes !=
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	SparkGlm52RingServiceBackendReleaseResidentSubmitCredit(state);
	if (result->status == (uint32_t)SPARK_STATUS_OK)
		return SPARK_STATUS_OK;
	state->cuda_resident_rejection_count += 1u;
	snprintf(
		blocker,
		sizeof(blocker),
		"resident submission rejected status=%u blocker=%.*s",
		result->status,
		(int32_t)sizeof(result->stats.blocker),
		result->stats.blocker);
	SparkGlm52RingServiceBackendSetBlocker(state,blocker);
	fprintf(stderr,"ring_resident_submit_rejected status=%u blocker=%.*s\n",
		result->status,(int32_t)sizeof(result->stats.blocker),
		result->stats.blocker);
	return (SparkStatus)result->status;
}

static SparkStatus SparkGlm52RingServiceBackendPumpCudaResidentResponses(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52CudaResidentIpcHeader header;
	SparkStatus status;
	uint32_t message_count;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->cuda_resident_fd < 0)
		return SPARK_STATUS_BUSY;
	for (message_count = 0u; message_count < 64u; ++message_count)
	{
		status = SparkGlm52RingServiceBackendResidentReadMessage(
			state,&header,0u);
		if (status == SPARK_STATUS_BUSY)
			return SPARK_STATUS_OK;
		if (status != SPARK_STATUS_OK)
			break;
		if (header.kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_COMPLETION)
			status = SparkGlm52RingServiceBackendHandleResidentCompletion(
				state,&header);
		else if (header.kind ==
			SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT)
			status = SparkGlm52RingServiceBackendHandleResidentSubmitResult(
				state,&header);
		else
			status = SPARK_STATUS_ABI_MISMATCH;
		if (status != SPARK_STATUS_OK)
			break;
	}
	if (status == SPARK_STATUS_OK)
		return SPARK_STATUS_OK;
	SparkGlm52RingServiceBackendTeardownCudaResident(
		state,"resident_response");
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendRequireResidentSubmitCredits(
	SparkGlm52RingServiceBackendState *state,
	uint32_t required_credit_count)
{
	SparkStatus status;
	if (state == 0 || required_credit_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingServiceBackendEnsureCudaResident(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendPumpCudaResidentResponses(state);
	if (status != SPARK_STATUS_OK)
		return status;
	return state->cuda_resident_submit_credit >= required_credit_count
		? SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static SparkStatus SparkGlm52RingServiceBackendSubmitResidentMessage(
	SparkGlm52RingServiceBackendState *state,
	uint32_t kind,
	const void *payload,
	uint32_t payload_bytes)
{
	SparkStatus status;
	if (state == 0 || state->cuda_resident_fd < 0 ||
		state->cuda_resident_submit_credit == 0u)
		return SPARK_STATUS_BUSY;
	status = SparkGlm52RingServiceBackendResidentWriteMessage(
		state,kind,payload,payload_bytes);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52RingServiceBackendTeardownCudaResident(
			state,"submit_message_write");
		return SPARK_STATUS_BUSY;
	}
	state->cuda_resident_submit_credit -= 1u;
	state->cuda_resident_submit_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendResidentAwaitSubmitResult(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52CudaResidentIpcStats *stats_out)
{
	SparkGlm52CudaResidentIpcHeader header;
	SparkStatus status;
	for (;;)
	{
		status = SparkGlm52RingServiceBackendResidentReadMessage(
			state,&header,180000u);
		if (status != SPARK_STATUS_OK)
		{
			fprintf(stderr,"ring_resident_await_failed step=header status=%u\n",(uint32_t)status);
			SparkGlm52RingServiceBackendTeardownCudaResident(state,"await_header");
			return SPARK_STATUS_BUSY;
		}
		status = SparkGlm52CudaResidentIpcValidateHeader(&header,0u,SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES);
		if (status != SPARK_STATUS_OK)
		{
			SparkGlm52RingServiceBackendTeardownCudaResident(state,"await_header_invalid");
			return SPARK_STATUS_BUSY;
		}
		if (header.kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT)
		{
			status = SparkGlm52RingServiceBackendHandleResidentSubmitResult(
				state,&header);
			if (stats_out != 0)
				*stats_out = state->cuda_resident_payload.submit_result.stats;
			return status;
		}
		if (header.kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_COMPLETION)
		{
			status = SparkGlm52RingServiceBackendHandleResidentCompletion(
				state,&header);
			if (status != SPARK_STATUS_OK)
				return status;
			continue;
		}
		SparkGlm52RingServiceBackendTeardownCudaResident(state,"await_unknown_kind");
		return SPARK_STATUS_BUSY;
	}
}

static SparkStatus SparkGlm52RingServiceBackendSubmitWorkToResident(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingWorkControlPacket *packet,
	uint32_t submit_flags)
{
	SparkGlm52CudaResidentIpcSubmitWork message;
	SparkStatus status;
	if (state == 0 || packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52CudaResidentIpcInitializeSubmitWork(
		&message,packet,submit_flags);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52RingServiceBackendSubmitResidentMessage(
		state,
		SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK,
		&message,message.descriptor_bytes);
}

static SparkStatus SparkGlm52RingServiceBackendSubmitReleaseToResident(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingWorkControlPacket *packet)
{
	SparkStatus status;
	status = SparkGlm52RingServiceBackendRequireResidentSubmitCredits(
		state,1u);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingServiceBackendSubmitWorkToResident(
		state,packet,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52RingServiceBackendResidentAwaitSubmitResult(state,0);
}

static SparkStatus SparkGlm52RingServiceBackendSubmitReleaseToRank0(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingWorkControlPacket *packet)
{
	SparkGlm52RingWorkControlPacket local_packet;
	if (state == 0 || packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->cuda_resident_attached != 0u)
		return SparkGlm52RingServiceBackendSubmitReleaseToResident(
			state,packet);
	if (state->builder_state == 0 ||
		state->builder_library.builder_interface.submit_work == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	local_packet = *packet;
	local_packet.control_generation =
		SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
	return state->builder_library.builder_interface.submit_work(
		state->builder_state,&local_packet,0,0,0,0);
}

static SparkStatus SparkGlm52RingServiceBackendQueueSequenceRelease(
	void *context,
	uint64_t request_id,
	uint64_t sequence_id,
	uint32_t token_count)
{
	SparkGlm52RingServiceBackendState *state;
	uint32_t tail;
	state = (SparkGlm52RingServiceBackendState *)context;
	if (state == 0 || request_id == 0u || sequence_id == 0u ||
		token_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (token_count <= SPARK_GLM52_RING_SERVICE_BACKEND_CONTEXT_TOKENS -
		SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT)
		token_count +=
			SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
	else
		token_count = SPARK_GLM52_RING_SERVICE_BACKEND_CONTEXT_TOKENS;
	if (state->release_queue_count >=
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	tail = (state->release_queue_head + state->release_queue_count) %
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
	state->release_queue[tail].request_id = request_id;
	state->release_queue[tail].sequence_id = sequence_id;
	state->release_queue[tail].token_count = token_count;
	state->release_queue[tail].reserved0 = 0u;
	state->release_queue_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendForwardPrefillPacket(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingWorkControlPacket *packet)
{
	SparkStatus status;
	if (state == 0 || packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingWorkControlValidatePacket(
		packet,state->rank_plan.execution_row_capacity,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendEnqueueWorkPacket(state,packet);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingServiceBackendPumpWorkOutput(state);
	return status == SPARK_STATUS_BUSY ? SPARK_STATUS_OK : status;
}

static SparkStatus SparkGlm52RingServiceBackendSubmitPrefillPacket(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52CudaResidentIpcSubmitPrefill *message)
{
	SparkStatus status;
	status = SparkGlm52RingServiceBackendSubmitResidentMessage(
		state,
		SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_PREFILL,
		message,
		message->descriptor_bytes);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52RingServiceBackendSetBlocker(
			state,"forwarded prefill packet was not queued locally");
		return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendSubmitPrefillToResident(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	static SparkGlm52CudaResidentIpcSubmitPrefill message;
	uint32_t token_count;
	uint32_t token_offset;
	SparkStatus status;
	if (state == 0 || prefill_dispatch == 0 ||
		prefill_dispatch->request_dispatch == 0 ||
		prefill_dispatch->prefill_view == 0 ||
		prefill_dispatch->kv_block_table_view == 0 ||
		prefill_dispatch->host_token_ids == 0 ||
		prefill_dispatch->lane_count == 0u ||
		prefill_dispatch->lane_count !=
			prefill_dispatch->active_sequence_count ||
		prefill_dispatch->prompt_token_count == 0u ||
		prefill_dispatch->prompt_token_count >
			SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (prefill_dispatch->lane_count >
		state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for (token_offset = 0u;
		 token_offset < prefill_dispatch->prompt_token_count;
		 token_offset += token_count)
	{
		status = SparkGlm52RingWorkControlSelectPrefillChunk(
			prefill_dispatch,
			token_offset,
			state->rank_plan.execution_row_capacity,
			&token_count);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52RingServiceBackendRequireResidentSubmitCredits(
			state,1u);
		if (status != SPARK_STATUS_OK)
			return status;
		memset(&message,0,sizeof(message));
		message.request_flags = prefill_dispatch->request_dispatch->flags;
		status = SparkGlm52RingWorkControlBuildPrefillPacket(
			prefill_dispatch,
			token_offset,
			token_count,
			&message.work_packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingServiceBackendStampWorkPacket(
				state,&message.work_packet);
		if (status != SPARK_STATUS_OK)
			return status;
		message.descriptor_bytes =
			SparkGlm52CudaResidentIpcCalculateSubmitPrefillBytes(
				&message.work_packet);
		if (message.descriptor_bytes == 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		status = SparkGlm52RingServiceBackendForwardPrefillPacket(
			state,&message.work_packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingServiceBackendSubmitPrefillPacket(
				state,&message);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendBuildDecodeResidentPayload(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52CudaResidentIpcSubmitDecode **message_out,
	uint32_t *payload_bytes_out)
{
	SparkGlm52CudaResidentIpcSubmitDecode *message;
	const SparkGlm52KvBlockTableView *kv_view;
	const SparkGlm52RequestApiDecodeDispatchLaneView *lane;
	uint64_t payload_bytes;
	uint64_t execution_row_count;
	uint32_t lane_count;
	uint32_t lane_index;
	uint32_t rows_per_lane;
	SparkStatus status;
	if (state == 0 || decode_dispatch == 0 ||
		message_out == 0 || payload_bytes_out == 0 ||
		decode_dispatch->request_dispatch == 0 ||
		decode_dispatch->decode_view == 0 ||
		decode_dispatch->kv_block_table_view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	lane_count = decode_dispatch->active_sequence_count;
	kv_view = decode_dispatch->kv_block_table_view;
	if (lane_count == 0u ||
		lane_count > SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT ||
		decode_dispatch->request_count != lane_count ||
		decode_dispatch->decode_view->active_sequence_count != lane_count ||
		decode_dispatch->decode_view->lane_count != lane_count ||
		kv_view->lane_count != lane_count ||
		kv_view->lane_capacity < lane_count || kv_view->block_token_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (decode_dispatch->dispatch_kind !=
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
		decode_dispatch->dispatch_kind !=
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((decode_dispatch->dispatch_kind ==
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
		 decode_dispatch->speculative_token_count != 0u) ||
		decode_dispatch->speculative_token_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (kv_view->lane_stride == 0u ||
		(kv_view->host_lane_physical_block_counts == 0 &&
		 kv_view->lane_physical_block_counts == 0))
		return SPARK_STATUS_INVALID_ARGUMENT;
	payload_bytes =
		(uint64_t)SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES;
	if (payload_bytes > SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES ||
		payload_bytes > UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (payload_bytes > state->cuda_resident_decode_payload_capacity)
	{
		uint32_t grown_capacity;
		uint8_t *grown_payload;
		grown_capacity = state->cuda_resident_decode_payload_capacity;
		while (grown_capacity < payload_bytes &&
			grown_capacity <=
				SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES / 2u)
			grown_capacity *= 2u;
		if (grown_capacity < payload_bytes)
			grown_capacity = (uint32_t)payload_bytes;
		grown_payload = (uint8_t *)realloc(
			state->cuda_resident_decode_payload,grown_capacity);
		if (grown_payload == 0)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->cuda_resident_decode_payload = grown_payload;
		state->cuda_resident_decode_payload_capacity = grown_capacity;
	}
	message = (SparkGlm52CudaResidentIpcSubmitDecode *)
		state->cuda_resident_decode_payload;
	memset(message,0,(size_t)payload_bytes);
	message->descriptor_bytes =
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES;
	message->control_generation = state->session_id_base;
	message->highest_priority =
		decode_dispatch->request_dispatch->highest_priority;
	message->request_flags = decode_dispatch->request_dispatch->flags;
	message->dispatch_kind = decode_dispatch->dispatch_kind;
	message->lane_count = lane_count;
	message->active_sequence_count = lane_count;
	rows_per_lane = decode_dispatch->dispatch_kind ==
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH
			? decode_dispatch->request_dispatch->
				speculative_verifier_token_count
			: 1u;
	execution_row_count = (uint64_t)lane_count * rows_per_lane;
	if (rows_per_lane == 0u || execution_row_count > UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = SparkGlm52RingWorkControlSelectExecutionBatchBucket(
		decode_dispatch->request_dispatch,
		(uint32_t)execution_row_count,
		&message->execution_batch_bucket);
	if (status != SPARK_STATUS_OK)
		return status;
	message->speculative_token_count =
		decode_dispatch->speculative_token_count;
	message->kv_block_token_count = kv_view->block_token_count;
	message->kv_block_index_count = 0u;
	message->resident_flags =
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
	{
		SparkGlm52CudaResidentIpcDecodeLane *target_lane;
		lane = &decode_dispatch->decode_view->lanes[lane_index];
		target_lane = &message->lanes[lane_index];
		if (lane->request_id !=
				decode_dispatch->request_dispatch->request_ids[lane_index] ||
			lane->sequence_id !=
				decode_dispatch->request_dispatch->sequence_ids[lane_index] ||
			lane->request_slot_index !=
				decode_dispatch->request_dispatch->request_slot_indices[lane_index])
			return SPARK_STATUS_INVALID_ARGUMENT;
		target_lane->request_id = lane->request_id;
		target_lane->sequence_id = lane->sequence_id;
		target_lane->sequence_position = lane->sequence_position;
		target_lane->request_slot_index = lane->request_slot_index;
		target_lane->context_token_count = lane->context_token_count;
		target_lane->input_token_id =
			decode_dispatch->input_token_ids[lane_index];
		target_lane->mtp_draft_token_budget =
			decode_dispatch->request_dispatch->mtp_draft_token_budget;
		target_lane->speculative_token_count =
			decode_dispatch->speculative_token_count;
		target_lane->mtp_resolution_proposed_token_count =
			(uint8_t)lane->mtp_resolution_proposed_token_count;
		target_lane->mtp_resolution_accepted_token_count =
			(uint8_t)lane->mtp_resolution_accepted_token_count;
		target_lane->mtp_resolution_path_id =
			(uint16_t)lane->mtp_resolution_path_id;
		target_lane->kv_block_offset = 0u;
		target_lane->kv_block_count = 0u;
		memcpy(target_lane->speculative_draft_token_ids,
			decode_dispatch->speculative_draft_token_ids[lane_index],
			sizeof(target_lane->speculative_draft_token_ids));
	}
	status = SparkGlm52CudaResidentIpcValidateSubmitDecode(
		message,(uint32_t)payload_bytes,lane_count);
	if (status != SPARK_STATUS_OK)
		return status;
	*message_out = message;
	*payload_bytes_out = (uint32_t)payload_bytes;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendSubmitDecodeToResident(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch)
{
	SparkGlm52CudaResidentIpcSubmitDecode *message;
	uint32_t payload_bytes;
	SparkStatus status;
	status = SparkGlm52RingServiceBackendBuildDecodeResidentPayload(
		state,decode_dispatch,&message,&payload_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52RingServiceBackendSubmitResidentMessage(
		state,
		SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_DECODE,
		message,payload_bytes);
}

static uint32_t SparkGlm52RingServiceBackendDecodeIsMtpVerify(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch)
{
	return decode_dispatch != 0 && decode_dispatch->request_dispatch != 0 &&
		(decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
}

static SparkStatus SparkGlm52RingServiceBackendPlanDecodeChunks(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t execution_row_capacity,
	uint32_t *maximum_lanes_per_chunk_out,
	uint32_t *chunk_count_out)
{
	uint32_t rows_per_lane;

	if (decode_dispatch == 0 || decode_dispatch->request_dispatch == 0 ||
		decode_dispatch->request_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	rows_per_lane = 1u;
	if ((decode_dispatch->request_dispatch->flags &
			(SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
			 SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY)) != 0u)
	{
		if (decode_dispatch->speculative_token_count == 0u ||
			decode_dispatch->speculative_token_count == UINT32_MAX)
			return SPARK_STATUS_INVALID_ARGUMENT;
		rows_per_lane = (decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u
			? decode_dispatch->request_dispatch->
				speculative_verifier_token_count
			: decode_dispatch->speculative_token_count + 1u;
	}
	return SparkGlm52RingWorkControlPlanExecutionChunks(
		decode_dispatch->request_count,
		rows_per_lane,
		execution_row_capacity,
		maximum_lanes_per_chunk_out,
		chunk_count_out);
}

static SparkStatus SparkGlm52RingServiceBackendSubmitDecodeChunksToResident(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch)
{
	SparkGlm52RingWorkControlPacket packet;
	uint32_t chunk_count;
	uint32_t chunk_index;
	uint32_t lane_count;
	uint32_t lane_offset;
	uint32_t maximum_lanes_per_chunk;
	SparkStatus status;

	if (state == 0 || decode_dispatch == 0 ||
		SparkGlm52RingServiceBackendDecodeIsMtpVerify(decode_dispatch) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingServiceBackendPlanDecodeChunks(
		decode_dispatch,state->rank_plan.execution_row_capacity,
		&maximum_lanes_per_chunk,&chunk_count);
	if (status != SPARK_STATUS_OK)
		return status;
	lane_offset = 0u;
	for (chunk_index = 0u; chunk_index < chunk_count; ++chunk_index)
	{
		lane_count = decode_dispatch->request_count - lane_offset;
		if (lane_count > maximum_lanes_per_chunk)
			lane_count = maximum_lanes_per_chunk;
		status = SparkGlm52RingServiceBackendBuildDecodeWorkPacket(
			decode_dispatch,lane_offset,lane_count,0u,&packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingServiceBackendStampWorkPacket(state,&packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingServiceBackendSubmitWorkToResident(
				state,&packet,0u);
		if (status != SPARK_STATUS_OK)
			return status;
		lane_offset += lane_count;
	}
	return lane_offset == decode_dispatch->request_count
		? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static void SparkGlm52RingServiceBackendFreeStorage(
	SparkGlm52RingServiceBackendState *state)
{
	if (state == 0)
		return;
	SparkGlm52ServingEngineDestroy(&state->serving_engine);
	free(state->kv_blocks);
	free(state->prefix_entries);
	free(state->prefix_bindings);
	free(state->request_slots);
	if (state->request_records != 0 && state->request_token_storage == 0)
	{
		uint32_t record_index;

		for (record_index = 0u;
			 record_index < SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
			 ++record_index)
			free(state->request_records[record_index].token_ids);
	}
	free(state->request_records);
	free(state->request_token_storage);
	free(state->dspark_sequence_states);
	free(state->serving_events);
	free(state->host_prefill_token_ids);
	free(state->host_physical_block_indices);
	free(state->lane_physical_block_counts);
	free(state->client_sessions);
	free(state->request_maps);
	free(state->service_events);
	free(state->cuda_resident_decode_payload);
	state->kv_blocks = 0;
	state->prefix_entries = 0;
	state->prefix_bindings = 0;
	state->request_slots = 0;
	state->request_records = 0;
	state->request_token_storage = 0;
	state->dspark_sequence_states = 0;
	state->serving_events = 0;
	state->host_prefill_token_ids = 0;
	state->host_physical_block_indices = 0;
	state->lane_physical_block_counts = 0;
	state->client_sessions = 0;
	state->request_maps = 0;
	state->service_events = 0;
	state->cuda_resident_decode_payload = 0;
	state->cuda_resident_decode_payload_capacity = 0u;
}

static uint32_t SparkGlm52RingServiceBackendMaxActive(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	if (configuration->max_active_sequence_count != 0u)
		return configuration->max_active_sequence_count;
	return SPARK_GLM52_RING_SERVICE_BACKEND_DEFAULT_MAX_ACTIVE;
}

static uint32_t SparkGlm52RingServiceBackendServiceLaneCapacity(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	uint32_t max_active;

	max_active = SparkGlm52RingServiceBackendMaxActive(configuration);
	if (max_active > SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY)
		return SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
	return max_active;
}

static uint32_t SparkGlm52RingServiceBackendPortBase(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	if (configuration->port_base != 0u)
		return configuration->port_base;
	return SPARK_GLM52_RING_SERVICE_BACKEND_DEFAULT_PORT_BASE;
}

static SparkStatus SparkGlm52RingServiceBackendLoadTokenizer(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	SparkTokenizerCompiledFileConfiguration tokenizer_configuration;
	SparkStatus status;

	status = SparkGlm52RingServiceBackendRequireText(
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
		SparkGlm52RingServiceBackendSetBlocker(
			state,
			"compiled C tokenizer failed to load");
		return status;
	}
	state->tokenizer_ready = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendPrefillIdlePump(
	void *idle_pump_context)
{
	SparkGlm52RingServiceBackendState *state;

	state = (SparkGlm52RingServiceBackendState *)idle_pump_context;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52RingServiceBackendPumpWorkOutput(state);
}

static SparkStatus SparkGlm52RingServiceBackendPrefillInner(
	void *context,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	SparkGlm52RingServiceBackendState *state;
	SparkStatus status;
	uint32_t drain_iteration;

	state = (SparkGlm52RingServiceBackendState *)context;
	if (state == 0 || prefill_dispatch == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->cuda_resident_attached != 0u)
	{
		status = SparkGlm52RingServiceBackendEnsureCudaResident(state);
		if (status != SPARK_STATUS_OK)
			return status;
		return SparkGlm52RingServiceBackendSubmitPrefillToResident(state,prefill_dispatch);
	}
	if (state->builder_library.builder_interface.prefill == 0 ||
		state->builder_state == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	fprintf(
		stderr,
		"ring_prefill dispatch_kind=%u active=%u lanes=%u offset=%u tokens=%u\n",
		prefill_dispatch->dispatch_kind,
		prefill_dispatch->active_sequence_count,
		prefill_dispatch->lane_count,
		prefill_dispatch->prompt_token_offset,
		prefill_dispatch->prompt_token_count);
	status = SparkGlm52RingServiceBackendForwardPrefillWork(
		state,
		prefill_dispatch);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"ring_prefill_forward status=%u\n",status);
		return status;
	}
	status = state->builder_library.builder_interface.prefill(
		state->builder_state,
		prefill_dispatch,
		SparkGlm52RingServiceBackendPrefillIdlePump,
		state);
	fprintf(stderr,"ring_prefill_builder status=%u\n",status);
	if (status != SPARK_STATUS_OK)
		return status;
	for (drain_iteration = 0u; state->work_queue_count != 0u && drain_iteration < 25000u; ++drain_iteration)
	{
		status = SparkGlm52RingServiceBackendPumpWorkOutput(state);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
			return status;
		if (state->work_queue_count != 0u)
			(void)poll(0,0,1);
	}
	if (state->work_queue_count != 0u)
	{
		fprintf(stderr,"ring_prefill_drain_incomplete queued=%u\n",state->work_queue_count);
		return SPARK_STATUS_IO_ERROR;
	}
	return SPARK_STATUS_OK;
}

static int32_t SparkGlm52RingServiceBackendSetNonblocking(int32_t fd)
{
	int32_t flags;

	flags = fcntl(fd,F_GETFL,0);
	if (flags < 0)
		return -1;
	if (fcntl(fd,F_SETFL,(flags | O_NONBLOCK)) < 0)
		return -2;
	return 0;
}

static int32_t SparkGlm52RingServiceBackendStartConnectToAddress(
	const struct addrinfo *entry,
	uint32_t *connecting_out)
{
	int32_t fd;
	int32_t status;

	if (entry == 0 || connecting_out == 0)
		return -1;
	fd = socket(entry->ai_family,entry->ai_socktype,entry->ai_protocol);
	if (fd < 0)
		return -2;
	if (SparkGlm52RingServiceBackendSetNonblocking(fd) < 0)
	{
		close(fd);
		return -3;
	}
	status = connect(fd,entry->ai_addr,entry->ai_addrlen);
	if (status == 0)
	{
		*connecting_out = 0u;
		return fd;
	}
	if (errno == EINPROGRESS || errno == EALREADY)
	{
		*connecting_out = 1u;
		return fd;
	}
	close(fd);
	return -4;
}

static int32_t SparkGlm52RingServiceBackendConnectSocket(
	const char *host,
	uint32_t port,
	uint32_t *connecting_out)
{
	struct addrinfo hints;
	struct addrinfo *results;
	struct addrinfo *entry;
	char service[16];
	int32_t fd;

	if (host == 0 || connecting_out == 0)
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
		fd = SparkGlm52RingServiceBackendStartConnectToAddress(
			entry,
			connecting_out);
		if (fd >= 0)
			break;
	}
	freeaddrinfo(results);
	return fd;
}

static uint32_t SparkGlm52RingServiceBackendWorkQueueTail(
	const SparkGlm52RingServiceBackendState *state)
{
	return (state->work_queue_head + state->work_queue_count) %
		SPARK_GLM52_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY;
}

static SparkStatus SparkGlm52RingServiceBackendEnqueueWorkPacket(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingWorkControlPacket *packet)
{
	uint32_t tail;

	if (state == 0 || packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->rank_plan.flags &
			SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
		return SPARK_STATUS_OK;
	if (state->work_queue_count >=
		SPARK_GLM52_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	tail = SparkGlm52RingServiceBackendWorkQueueTail(state);
	state->work_queue[tail] = *packet;
	state->work_queue_count += 1u;
	return SPARK_STATUS_OK;
}

static void SparkGlm52RingServiceBackendDropWorkOutputSocket(
	SparkGlm52RingServiceBackendState *state)
{
	if (state == 0)
		return;
	if (state->work_output_socket_fd >= 0)
		close(state->work_output_socket_fd);
	state->work_output_socket_fd = -1;
	state->work_output_socket_connecting = 0u;
	state->work_queue_write_offset = 0u;
}

static SparkStatus SparkGlm52RingServiceBackendStartWorkOutputSocket(
	SparkGlm52RingServiceBackendState *state)
{
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->rank_plan.flags &
			SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
		return SPARK_STATUS_OK;
	if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
		fprintf(
			stderr,
			"ring_work_connect host=%s port=%u\n",
			state->rank_plan.next_host_name,
			state->rank_plan.next_port);
	state->work_output_socket_fd =
		SparkGlm52RingServiceBackendConnectSocket(
			state->rank_plan.next_host_name,
			state->rank_plan.next_port,
			&state->work_output_socket_connecting);
	if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
		fprintf(
			stderr,
			"ring_work_connect_result fd=%d connecting=%u errno=%d\n",
			state->work_output_socket_fd,
			state->work_output_socket_connecting,
			errno);
	if (state->work_output_socket_fd < 0)
		return SPARK_STATUS_BUSY;
	if (state->work_output_socket_connecting != 0u)
		return SPARK_STATUS_BUSY;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendCheckWorkOutputConnect(
	SparkGlm52RingServiceBackendState *state)
{
	struct pollfd connect_poll;
	socklen_t error_bytes;
	int32_t error, poll_result;

	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->work_output_socket_fd < 0)
		return SPARK_STATUS_BUSY;
	if (state->work_output_socket_connecting == 0u)
		return SPARK_STATUS_OK;
	memset(&connect_poll,0,sizeof(connect_poll));
	connect_poll.fd = state->work_output_socket_fd;
	connect_poll.events = POLLOUT;
	poll_result = poll(&connect_poll,1,0);
	if (poll_result < 0)
	{
		SparkGlm52RingServiceBackendDropWorkOutputSocket(state);
		return SparkGlm52RingServiceBackendStartWorkOutputSocket(state);
	}
	if (poll_result == 0 ||
		(connect_poll.revents & (POLLOUT | POLLERR | POLLHUP)) == 0)
		return SPARK_STATUS_BUSY;
	error = 0;
	error_bytes = (socklen_t)sizeof(error);
	if (getsockopt(
			state->work_output_socket_fd,
			SOL_SOCKET,
			SO_ERROR,
			&error,
			&error_bytes) < 0)
	{
		SparkGlm52RingServiceBackendDropWorkOutputSocket(state);
		return SparkGlm52RingServiceBackendStartWorkOutputSocket(state);
	}
	if (error == 0)
	{
		state->work_output_socket_connecting = 0u;
		if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
			fprintf(
				stderr,
				"ring_work_connected fd=%d\n",
				state->work_output_socket_fd);
		return SPARK_STATUS_OK;
	}
	if (error == EINPROGRESS || error == EALREADY)
		return SPARK_STATUS_BUSY;
	if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
		fprintf(
			stderr,
			"ring_work_connect_drop fd=%d so_error=%d\n",
			state->work_output_socket_fd,
			error);
	SparkGlm52RingServiceBackendDropWorkOutputSocket(state);
	return SparkGlm52RingServiceBackendStartWorkOutputSocket(state);
}

static SparkStatus SparkGlm52RingServiceBackendFlushWorkOutput(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52RingWorkControlPacket *packet;
	uint32_t remaining;
	ssize_t written;

	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->rank_plan.flags &
			SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u ||
		state->work_queue_count == 0u)
		return SPARK_STATUS_OK;
	while (state->work_queue_count != 0u)
	{
		packet = &state->work_queue[state->work_queue_head];
		remaining = packet->descriptor_bytes -
			state->work_queue_write_offset;
		written = write(
			state->work_output_socket_fd,
			((const uint8_t *)packet) + state->work_queue_write_offset,
			remaining);
		if (written < 0)
		{
			if (errno == EINTR)
				continue;
			if (errno == EAGAIN || errno == EWOULDBLOCK)
				return SPARK_STATUS_BUSY;
			if (getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG") != 0)
				fprintf(
					stderr,
					"ring_work_flush_drop fd=%d errno=%d\n",
					state->work_output_socket_fd,
					errno);
			SparkGlm52RingServiceBackendDropWorkOutputSocket(state);
			return SPARK_STATUS_ROUTE_NOT_FOUND;
		}
		if (written == 0)
			return SPARK_STATUS_BUSY;
		state->work_queue_write_offset += (uint32_t)written;
		if (state->work_queue_write_offset != packet->descriptor_bytes)
			continue;
		state->work_queue_write_offset = 0u;
		state->work_queue_head =
			(state->work_queue_head + 1u) %
			SPARK_GLM52_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY;
		state->work_queue_count -= 1u;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendBuildReleasePacket(
	const SparkGlm52RingServiceBackendState *state,
	uint32_t lane_count,
	SparkGlm52RingWorkControlPacket *packet)
{
	uint32_t lane_index;
	uint32_t maximum_token_count;
	if (state == 0 || packet == 0 || lane_count == 0u ||
		lane_count > state->release_queue_count ||
		lane_count > SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_RING_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION;
	packet->descriptor_bytes =
		SparkGlm52RingWorkControlCalculatePacketBytes(lane_count);
	packet->flags = SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES;
	packet->control_generation = state->session_id_base;
	packet->active_sequence_count = lane_count;
	packet->lane_count = lane_count;
	packet->block_token_count =
		SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_RING_SERVICE_BACKEND_MAX_BLOCKS_PER_SEQUENCE;
	maximum_token_count = 0u;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
	{
		uint32_t queue_index;
		const SparkGlm52RingServiceBackendReleaseRecord *record;
		SparkGlm52RingWorkControlLane *lane;
		queue_index = (state->release_queue_head + lane_index) %
			SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
		record = &state->release_queue[queue_index];
		lane = &packet->lanes[lane_index];
		if (record->request_id == 0u || record->sequence_id == 0u ||
			record->token_count == 0u || record->reserved0 != 0u)
			return SPARK_STATUS_INTERNAL_ERROR;
		lane->request_id = record->request_id;
		lane->sequence_id = record->sequence_id;
		lane->context_token_count = record->token_count;
		if (record->token_count > maximum_token_count)
			maximum_token_count = record->token_count;
	}
	packet->request_id = packet->lanes[0u].request_id;
	packet->sequence_id = packet->lanes[0u].sequence_id;
	packet->kv_block_table_token_count = maximum_token_count;
	return SparkGlm52RingWorkControlValidatePacket(
		packet,state->rank_plan.execution_row_capacity,1u);
}

static SparkStatus SparkGlm52RingServiceBackendPumpSequenceReleases(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52RingWorkControlPacket packet;
	uint32_t lane_count;
	SparkStatus status;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->release_queue_count == 0u)
		return SPARK_STATUS_OK;
	lane_count = state->release_queue_count;
	if (lane_count > SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT)
		lane_count = SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT;
	status = SparkGlm52RingServiceBackendBuildReleasePacket(
		state,lane_count,&packet);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingServiceBackendSubmitReleaseToRank0(
		state,&packet);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingServiceBackendEnqueueWorkPacket(state,&packet);
	if (status == SPARK_STATUS_CAPACITY_EXCEEDED)
		return SPARK_STATUS_BUSY;
	if (status != SPARK_STATUS_OK)
		return status;
	state->release_queue_head = (state->release_queue_head + lane_count) %
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
	state->release_queue_count -= lane_count;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendBuildDecodeWorkPacket(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_token_index,
	SparkGlm52RingWorkControlPacket *packet)
{
	return SparkGlm52RingWorkControlBuildDecodePacketRange(
		decode_dispatch,lane_offset,lane_count,speculative_token_index,packet);
}

static SparkStatus SparkGlm52RingServiceBackendBuildPrefillWorkPacket(
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t token_count,
	SparkGlm52RingWorkControlPacket *packet)
{
	return SparkGlm52RingWorkControlBuildPrefillPacket(
		prefill_dispatch,token_offset,token_count,packet);
}

static SparkStatus SparkGlm52RingServiceBackendEnsureWorkOutputSocket(
	SparkGlm52RingServiceBackendState *state)
{
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->rank_plan.flags &
			SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
		return SPARK_STATUS_OK;
	if (state->work_output_socket_fd >= 0)
		return SparkGlm52RingServiceBackendCheckWorkOutputConnect(state);
	return SparkGlm52RingServiceBackendStartWorkOutputSocket(state);
}

static SparkStatus SparkGlm52RingServiceBackendPumpWorkOutput(
	SparkGlm52RingServiceBackendState *state)
{
	SparkStatus status;

	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->rank_plan.flags &
			SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u ||
		state->work_queue_count == 0u)
		return SPARK_STATUS_OK;
	status = SparkGlm52RingServiceBackendEnsureWorkOutputSocket(state);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52RingServiceBackendFlushWorkOutput(state);
}

static SparkStatus SparkGlm52RingServiceBackendForwardPrefillWork(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	SparkGlm52RingWorkControlPacket packet;
	SparkStatus status;
	uint32_t token_count;
	uint32_t token_offset;
	if (state == 0 || prefill_dispatch == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->rank_plan.flags &
			SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
		return SPARK_STATUS_OK;
	if (prefill_dispatch->lane_count >
		state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for (token_offset = 0u;
		 token_offset < prefill_dispatch->prompt_token_count;
		 token_offset += token_count)
	{
		status = SparkGlm52RingWorkControlSelectPrefillChunk(
			prefill_dispatch,
			token_offset,
			state->rank_plan.execution_row_capacity,
			&token_count);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52RingServiceBackendBuildPrefillWorkPacket(
			prefill_dispatch,
			token_offset,
			token_count,
			&packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingServiceBackendStampWorkPacket(state,&packet);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52RingWorkControlValidatePacket(
			&packet,
			state->rank_plan.execution_row_capacity,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52RingServiceBackendEnqueueWorkPacket(state,&packet);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52RingServiceBackendPumpWorkOutput(state);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
			return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendForwardDecodeWork(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch)
{
	SparkGlm52RingWorkControlPacket packet;
	uint32_t chunk_count;
	uint32_t chunk_index;
	uint32_t lane_count;
	uint32_t lane_offset;
	uint32_t maximum_lanes_per_chunk;
	SparkStatus status;
	if (state == 0 || decode_dispatch == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingServiceBackendPlanDecodeChunks(
		decode_dispatch,state->rank_plan.execution_row_capacity,
		&maximum_lanes_per_chunk,&chunk_count);
	if (status != SPARK_STATUS_OK)
		return status;
	if (chunk_count >
		SPARK_GLM52_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY -
			state->work_queue_count)
		return SPARK_STATUS_BUSY;
	lane_offset = 0u;
	for (chunk_index = 0u; chunk_index < chunk_count; ++chunk_index)
	{
		lane_count = decode_dispatch->request_count - lane_offset;
		if (lane_count > maximum_lanes_per_chunk)
			lane_count = maximum_lanes_per_chunk;
		status = SparkGlm52RingServiceBackendBuildDecodeWorkPacket(
			decode_dispatch,lane_offset,lane_count,0u,&packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingServiceBackendStampWorkPacket(state,&packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingWorkControlValidatePacket(
				&packet,state->rank_plan.execution_row_capacity,
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52RingServiceBackendEnqueueWorkPacket(state,&packet);
		if (status != SPARK_STATUS_OK)
			return status;
		lane_offset += lane_count;
	}
	if (lane_offset != decode_dispatch->request_count)
		return SPARK_STATUS_INTERNAL_ERROR;
	status = SparkGlm52RingServiceBackendPumpWorkOutput(state);
	if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
		return status;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendPrefill(
	void *context,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	SparkGlm52RingServiceBackendState *trace_state;
	SparkStatus status;
	uint64_t trace_begin_ns;
	trace_state = (SparkGlm52RingServiceBackendState *)context;
	trace_begin_ns = trace_state != 0 && trace_state->trace_enabled != 0u ? SparkGlm52RingServiceBackendMonotonicNs() : 0u;
	if (trace_begin_ns != 0u)
		fprintf(stderr,"ring_trace prefill_begin request=%llu offset=%u count=%u\n",(unsigned long long)(prefill_dispatch != 0 && prefill_dispatch->request_dispatch != 0 ? prefill_dispatch->request_dispatch->request_ids[0u] : 0u),prefill_dispatch != 0 ? prefill_dispatch->prompt_token_offset : 0u,prefill_dispatch != 0 ? prefill_dispatch->prompt_token_count : 0u);
	status = SparkGlm52RingServiceBackendPrefillInner(context, prefill_dispatch);
	if (trace_begin_ns != 0u)
		fprintf(stderr,"ring_trace prefill_end status=%u dur_us=%llu\n",(uint32_t)status,(unsigned long long)((SparkGlm52RingServiceBackendMonotonicNs() - trace_begin_ns) / 1000ull));
	return status;
}

static uint64_t SparkGlm52RingServiceBackendTraceDecodeSubmit(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch)
{
	uint64_t request_id;
	uint64_t submit_ns;
	if (state == 0 || decode_dispatch == 0 || state->trace_enabled == 0u)
		return 0u;
	request_id = decode_dispatch->request_dispatch != 0 &&
		decode_dispatch->request_dispatch->request_count != 0u ?
		decode_dispatch->request_dispatch->request_ids[0u] : 0u;
	submit_ns = SparkGlm52RingServiceBackendMonotonicNs();
	fprintf(stderr,
		"ring_decode kind=%u requests=%u active=%u request=%llu\n",
		decode_dispatch->dispatch_kind,decode_dispatch->request_count,
		decode_dispatch->active_sequence_count,
		(unsigned long long)request_id);
	if (submit_ns != 0u && state->trace_last_decode_completion_ns != 0u)
		fprintf(stderr,"ring_decode_gap_ns=%llu request=%llu kind=%u\n",
			(unsigned long long)(submit_ns -
				state->trace_last_decode_completion_ns),
			(unsigned long long)request_id,decode_dispatch->dispatch_kind);
	return submit_ns;
}

static SparkStatus SparkGlm52RingServiceBackendDecodeInner(
	void *context,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result)
{
	SparkGlm52RingServiceBackendState *state;
	SparkGlm52RingServiceBackendPendingDecode *pending;
	uint32_t chunk_count;
	uint32_t maximum_lanes_per_chunk;
	uint32_t resident_submit_count;
	SparkStatus status;
	uint64_t trace_submit_ns;

	state = (SparkGlm52RingServiceBackendState *)context;
	if (state == 0 || decode_dispatch == 0 || decode_result == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (decode_dispatch->active_sequence_count == 0u ||
		decode_dispatch->active_sequence_count >
			SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
		decode_dispatch->request_count !=
			decode_dispatch->active_sequence_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	trace_submit_ns = 0u;
	if (state->cuda_resident_attached != 0u)
	{
		resident_submit_count = 1u;
		if (SparkGlm52RingServiceBackendDecodeIsMtpVerify(
				decode_dispatch) != 0u)
		{
			status = SparkGlm52RingServiceBackendPlanDecodeChunks(
				decode_dispatch,
				state->rank_plan.execution_row_capacity,
				&maximum_lanes_per_chunk,
				&chunk_count);
			if (status != SPARK_STATUS_OK)
				return status;
			resident_submit_count = chunk_count;
		}
		status = SparkGlm52RingServiceBackendRequireResidentSubmitCredits(
			state,resident_submit_count);
		if (status != SPARK_STATUS_OK)
			return status;
		memset(decode_result,0,sizeof(*decode_result));
		decode_result->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
		decode_result->descriptor_bytes = SPARK_GLM52_SERVING_DECODE_RESULT_DESCRIPTOR_BYTES;
		decode_result->lane_count = decode_dispatch->active_sequence_count;
		decode_result->token_stride = SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE;
		pending = 0;
		status = SparkGlm52RingServiceBackendRegisterPendingDecode(
			state,
			decode_dispatch,
			decode_result,
			&pending);
		if (status != SPARK_STATUS_OK)
			return status;
		if (pending != 0 && pending->state ==
			SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_FREE)
			return SPARK_STATUS_OK;
		trace_submit_ns = SparkGlm52RingServiceBackendTraceDecodeSubmit(
			state,decode_dispatch);
		if (pending != 0)
			pending->trace_submit_time_ns = trace_submit_ns;
		status = SparkGlm52RingServiceBackendForwardDecodeWork(state,decode_dispatch);
		if (status != SPARK_STATUS_OK)
		{
			if (pending != 0)
				memset(pending,0,sizeof(*pending));
			return status;
		}
		if (SparkGlm52RingServiceBackendDecodeIsMtpVerify(decode_dispatch) != 0u)
			status = SparkGlm52RingServiceBackendSubmitDecodeChunksToResident(
				state,decode_dispatch);
		else
			status = SparkGlm52RingServiceBackendSubmitDecodeToResident(
				state,decode_dispatch);
		if (status != SPARK_STATUS_OK)
		{
			if (pending != 0)
				memset(pending,0,sizeof(*pending));
			return status;
		}
		return SPARK_STATUS_PENDING;
	}
	if (state->builder_library.builder_interface.decode == 0 ||
		state->builder_state == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	status = SparkGlm52RingServiceBackendPlanDecodeChunks(
		decode_dispatch,state->rank_plan.execution_row_capacity,
		&maximum_lanes_per_chunk,&chunk_count);
	if (status != SPARK_STATUS_OK)
		return status;
	if (chunk_count != 1u)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	status = state->builder_library.builder_interface.decode(
		state->builder_state,
		decode_dispatch,
		decode_result);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"ring_decode_builder status=%u\n",status);
		return status;
	}
	pending = 0;
	status = SparkGlm52RingServiceBackendRegisterPendingDecode(
		state,
		decode_dispatch,
		decode_result,
		&pending);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"ring_decode_pending status=%u\n",status);
		return status;
	}
	if (pending != 0 && pending->state ==
		SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_FREE)
		return SPARK_STATUS_OK;
	trace_submit_ns = SparkGlm52RingServiceBackendTraceDecodeSubmit(
		state,decode_dispatch);
	if (pending != 0)
		pending->trace_submit_time_ns = trace_submit_ns;
	status = SparkGlm52RingServiceBackendForwardDecodeWork(
		state,
		decode_dispatch);
	if (status != SPARK_STATUS_OK)
	{
		if (pending != 0)
			memset(pending,0,sizeof(*pending));
		fprintf(stderr,"ring_decode_forward status=%u\n",status);
		return status;
	}
	if (state->trace_enabled != 0u)
		fprintf(stderr,"ring_decode_pending_final begin\n");
	return SPARK_STATUS_PENDING;
}

static SparkStatus SparkGlm52RingServiceBackendDecode(
	void *context,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result)
{
	SparkGlm52RingServiceBackendState *trace_state;
	SparkStatus status;
	uint64_t trace_begin_ns;
	trace_state = (SparkGlm52RingServiceBackendState *)context;
	trace_begin_ns = trace_state != 0 && trace_state->trace_enabled != 0u ? SparkGlm52RingServiceBackendMonotonicNs() : 0u;
	if (trace_begin_ns != 0u)
		fprintf(stderr,"ring_trace decode_begin request=%llu\n",(unsigned long long)(decode_dispatch != 0 && decode_dispatch->request_dispatch != 0 ? decode_dispatch->request_dispatch->request_ids[0u] : 0u));
	status = SparkGlm52RingServiceBackendDecodeInner(context, decode_dispatch, decode_result);
	if (trace_begin_ns != 0u)
		fprintf(stderr,"ring_trace decode_end status=%u dur_us=%llu\n",(uint32_t)status,(unsigned long long)((SparkGlm52RingServiceBackendMonotonicNs() - trace_begin_ns) / 1000ull));
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendAllocateCacheStorage(
	SparkGlm52RingServiceBackendState *state)
{
	SparkStatus status;

	status = SparkGlm52RingServiceBackendAlloc(
		(void **)&state->kv_blocks,
		state->kv_logical_block_capacity,
		sizeof(state->kv_blocks[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->prefix_entries,
			state->kv_logical_block_capacity,
			sizeof(state->prefix_entries[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->prefix_bindings,
			(uint64_t)state->kv_logical_block_capacity +
				SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY,
			sizeof(state->prefix_bindings[0]));
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendAllocateRequestStorage(
	SparkGlm52RingServiceBackendState *state,
	uint32_t lane_capacity)
{
	SparkStatus status;

	status = SparkGlm52RingServiceBackendAlloc(
		(void **)&state->request_slots,
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY,
		sizeof(state->request_slots[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->request_records,
			SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY,
			sizeof(state->request_records[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->dspark_sequence_states,
			SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY,
			sizeof(state->dspark_sequence_states[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->serving_events,
			SPARK_GLM52_RING_SERVICE_BACKEND_EVENT_CAPACITY,
			sizeof(state->serving_events[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->host_prefill_token_ids,
			(uint64_t)lane_capacity *
				SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_TOKENS,
			sizeof(state->host_prefill_token_ids[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->host_physical_block_indices,
			(uint64_t)lane_capacity *
				SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_COUNT,
			sizeof(state->host_physical_block_indices[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->lane_physical_block_counts,
			lane_capacity,
			sizeof(state->lane_physical_block_counts[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->client_sessions,
			SPARK_GLM52_RING_SERVICE_BACKEND_CLIENT_CAPACITY,
			sizeof(state->client_sessions[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->request_maps,
			SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_MAP_CAPACITY,
			sizeof(state->request_maps[0]));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->service_events,
			SPARK_GLM52_RING_SERVICE_BACKEND_EVENT_CAPACITY,
			sizeof(state->service_events[0]));
	if (status == SPARK_STATUS_OK)
	{
		status = SparkGlm52RingServiceBackendAlloc(
			(void **)&state->cuda_resident_decode_payload,
			SPARK_GLM52_RING_SERVICE_BACKEND_INITIAL_DECODE_PAYLOAD_BYTES,
			sizeof(state->cuda_resident_decode_payload[0]));
		if (status == SPARK_STATUS_OK)
			state->cuda_resident_decode_payload_capacity =
				SPARK_GLM52_RING_SERVICE_BACKEND_INITIAL_DECODE_PAYLOAD_BYTES;
	}
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendAllocateServiceStorage(
	SparkGlm52RingServiceBackendState *state,
	uint32_t lane_capacity)
{
	SparkStatus status;

	status = SparkGlm52RingServiceBackendAllocateCacheStorage(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendAllocateRequestStorage(
			state,
			lane_capacity);
	if (status != SPARK_STATUS_OK)
		SparkGlm52RingServiceBackendFreeStorage(state);
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendInitializeKvArena(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52KvCacheConfiguration kv_configuration;

	memset(&kv_configuration,0,sizeof(kv_configuration));
	kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	kv_configuration.descriptor_bytes =
		SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	kv_configuration.physical_block_count =
		state->kv_logical_block_capacity;
	kv_configuration.block_token_count =
		SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS;
	kv_configuration.layer_count = SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
	kv_configuration.kv_head_count = 8u;
	kv_configuration.head_dim = 128u;
	kv_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
	kv_configuration.key_device_base =
		(void *)(uintptr_t)SPARK_GLM52_RING_SERVICE_BACKEND_METADATA_KEY_BASE;
	kv_configuration.value_device_base =
		(void *)(uintptr_t)SPARK_GLM52_RING_SERVICE_BACKEND_METADATA_VALUE_BASE;
	kv_configuration.blocks = state->kv_blocks;
	return SparkGlm52KvCacheArenaInitialize(
		&state->kv_arena,
		&kv_configuration);
}

static SparkStatus SparkGlm52RingServiceBackendInitializePrefixCache(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52PrefixCacheConfiguration prefix_configuration;

	memset(&prefix_configuration,0,sizeof(prefix_configuration));
	prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
	prefix_configuration.descriptor_bytes =
		SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	prefix_configuration.block_token_count =
		SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS;
	prefix_configuration.entry_count =
		state->kv_logical_block_capacity;
	prefix_configuration.physical_block_count =
		state->kv_logical_block_capacity;
	prefix_configuration.sequence_binding_count =
		state->kv_logical_block_capacity +
			SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
	prefix_configuration.entries = state->prefix_entries;
	prefix_configuration.sequence_bindings = state->prefix_bindings;
	prefix_configuration.kv_cache_arena = &state->kv_arena;
	return SparkGlm52PrefixCacheInitialize(
		&state->prefix_cache,
		&prefix_configuration);
}

static SparkStatus SparkGlm52RingServiceBackendInitializeScheduler(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52SchedulerConfiguration scheduler_configuration;

	memset(&scheduler_configuration,0,sizeof(scheduler_configuration));
	scheduler_configuration.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
	scheduler_configuration.descriptor_bytes =
		SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
	scheduler_configuration.spark_count = SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT;
	scheduler_configuration.queue_depth_per_spark =
		SPARK_GLM52_RING_SERVICE_BACKEND_QUEUE_DEPTH_PER_SPARK;
	scheduler_configuration.measured_profile_id =
		SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
	scheduler_configuration.quantization_mode =
		state->rank_plan.quantization_mode;
	scheduler_configuration.max_prefill_tokens_per_step =
		SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_WAVE_TOKENS;
	scheduler_configuration.prefix_cache_block_tokens =
		SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS;
	scheduler_configuration.configuration_flags =
		SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS &
		~SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE;
	scheduler_configuration.prefix_cache = &state->prefix_cache;
	return SparkGlm52SchedulerInitialize(
		&state->scheduler,
		&scheduler_configuration);
}

static SparkStatus SparkGlm52RingServiceBackendDsparkDraft(
	void *context,
	const SparkGlm52DsparkDraftRequest *request,
	SparkGlm52DsparkDraftResult *result)
{
	SparkGlm52RingServiceBackendState *state;
	SparkGlm52RingServiceBackendPendingDecode *pending;
	SparkGlm52DsparkDraftResult *ready_draft;
	uint32_t pending_index;
	uint32_t lane_index;
	uint32_t token_index;

	state = (SparkGlm52RingServiceBackendState *)context;
	if (state == 0 || request == 0 || result == 0 ||
		request->requested_token_count == 0u ||
		request->requested_token_count >
			SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	ready_draft = 0;
	for (pending_index = 0u;
		 pending_index < SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY;
		 ++pending_index)
	{
		pending = &state->pending_decodes[pending_index];
		if (pending->state !=
			SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE)
			continue;
		for (lane_index = 0u;
			 lane_index < pending->dispatch.request_count;
			 ++lane_index)
		{
			if (pending->dispatch.request_ids[lane_index] ==
					request->request_id &&
				pending->dispatch.sequence_ids[lane_index] ==
					request->sequence_id &&
				pending->dspark_draft_valid[lane_index] != 0u)
			{
				ready_draft = &pending->dspark_drafts[lane_index];
				break;
			}
		}
		if (ready_draft != 0)
			break;
	}
	if (ready_draft == 0)
		return SPARK_STATUS_NOT_FOUND;
	if (request->requested_token_count > ready_draft->token_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	pending->dspark_draft_valid[lane_index] = 0u;
	*result = *ready_draft;
	result->token_count = request->requested_token_count;
	for (token_index = result->token_count;
		 token_index < SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
		 ++token_index)
	{
		result->token_ids[token_index] = 0u;
		result->confidence_milli[token_index] = 0u;
	}
	memset(ready_draft,0,sizeof(*ready_draft));
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendInitializeDspark(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52DsparkSpeculatorConfiguration configuration;
	SparkStatus status;

	if (state->dspark_enabled == 0u)
		return SPARK_STATUS_OK;
	status = SparkGlm52DsparkBuildDefaultModelContract(
		&state->dspark_model_contract);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.policy_flags = SPARK_GLM52_DSPARK_POLICY_DEFAULT_FLAGS;
	configuration.sequence_state_count =
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
	configuration.default_speculative_token_count =
		SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
	configuration.sequence_states = state->dspark_sequence_states;
	configuration.draft_function = SparkGlm52RingServiceBackendDsparkDraft;
	configuration.draft_context = state;
	configuration.model_contract = &state->dspark_model_contract;
	return SparkGlm52DsparkInitialize(&state->dspark_speculator,&configuration);
}

static SparkStatus SparkGlm52RingServiceBackendInitializeRequestApi(
	SparkGlm52RingServiceBackendState *state,
	uint32_t lane_capacity)
{
	SparkGlm52RequestApiConfiguration request_api_configuration;
	SparkStatus status;

	memset(&request_api_configuration,0,sizeof(request_api_configuration));
	request_api_configuration.abi_version =
		SPARK_GLM52_REQUEST_API_ABI_VERSION;
	request_api_configuration.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
	request_api_configuration.configuration_flags =
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
	if (state->mtp_enabled != 0u)
		request_api_configuration.configuration_flags |=
			SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
			SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
	if (state->dspark_enabled != 0u)
	{
		request_api_configuration.configuration_flags |=
			SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE;
		request_api_configuration.dspark_speculator = &state->dspark_speculator;
	}
	request_api_configuration.request_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
	request_api_configuration.prefetch_lane_count =
		SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT;
	request_api_configuration.decode_batch_target = lane_capacity;
	request_api_configuration.max_resident_kv_block_count =
		state->kv_physical_block_capacity;
	request_api_configuration.decode_execution_row_capacity =
		SparkGlm52RingRuntimeExecutionRowCapacity(lane_capacity);
	request_api_configuration.scheduler = &state->scheduler;
	request_api_configuration.request_slots = state->request_slots;
	status = SparkGlm52RequestApiInitialize(
		&state->request_api,
		&request_api_configuration);
	if (status == SPARK_STATUS_OK)
		state->request_api.next_sequence_id = state->session_id_base;
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendInitializeServingEngine(
	SparkGlm52RingServiceBackendState *state,
	uint32_t lane_capacity)
{
	static const uint32_t StopTokenIds[] =
		SPARK_GLM52_MODEL_EOS_TOKEN_IDS_INITIALIZER;
	SparkGlm52ServingEngineConfiguration serving_configuration;

	memset(&serving_configuration,0,sizeof(serving_configuration));
	serving_configuration.abi_version =
		SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
	serving_configuration.descriptor_bytes =
		SPARK_GLM52_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES;
	serving_configuration.flags =
		SPARK_GLM52_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS |
		SPARK_GLM52_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT |
		SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE;
	serving_configuration.runtime_contract_flags =
		SPARK_GLM52_SERVING_RUNTIME_CONTRACT_CURRENT_IMPLEMENTED_FLAGS;
	serving_configuration.default_output_token_budget = 1024u;
	serving_configuration.default_max_prefill_tokens_per_step =
		SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_WAVE_TOKENS;
	serving_configuration.max_context_tokens =
		SPARK_GLM52_RING_SERVICE_BACKEND_CONTEXT_TOKENS;
	serving_configuration.request_api = &state->request_api;
	serving_configuration.tokenizer = &state->tokenizer;
	serving_configuration.request_records = state->request_records;
	serving_configuration.request_record_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY;
	serving_configuration.request_token_storage = 0;
	serving_configuration.request_token_stride = 0u;
	serving_configuration.event_ring = state->serving_events;
	serving_configuration.event_ring_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_EVENT_CAPACITY;
	serving_configuration.host_prefill_token_ids =
		state->host_prefill_token_ids;
	serving_configuration.host_prefill_token_stride =
		SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_TOKENS;
	serving_configuration.host_prefill_lane_capacity = lane_capacity;
	serving_configuration.host_physical_block_indices =
		state->host_physical_block_indices;
	serving_configuration.kv_block_lane_stride =
		SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_COUNT;
	serving_configuration.kv_block_lane_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_KV_BLOCK_COUNT;
	serving_configuration.lane_physical_block_counts =
		state->lane_physical_block_counts;
	serving_configuration.lane_count_capacity = lane_capacity;
	serving_configuration.prefill_function =
		SparkGlm52RingServiceBackendPrefill;
	serving_configuration.decode_function = SparkGlm52RingServiceBackendDecode;
	serving_configuration.release_sequence_function =
		SparkGlm52RingServiceBackendQueueSequenceRelease;
	serving_configuration.callback_context = state;
	serving_configuration.stop_token_ids = StopTokenIds;
	serving_configuration.stop_token_id_count =
		(uint32_t)(sizeof(StopTokenIds) / sizeof(StopTokenIds[0u]));
	return SparkGlm52ServingEngineInitialize(
		&state->serving_engine,
		&serving_configuration);
}

static SparkStatus SparkGlm52RingServiceBackendInitializeService(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52ServiceConfiguration service_configuration;

	memset(&service_configuration,0,sizeof(service_configuration));
	service_configuration.abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
	service_configuration.descriptor_bytes =
		SPARK_GLM52_SERVICE_CONFIGURATION_DESCRIPTOR_BYTES;
	service_configuration.request_id_base = state->session_id_base;
	service_configuration.serving_engine = &state->serving_engine;
	service_configuration.client_sessions = state->client_sessions;
	service_configuration.client_session_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_CLIENT_CAPACITY;
	service_configuration.request_maps = state->request_maps;
	service_configuration.request_map_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_MAP_CAPACITY;
	service_configuration.event_ring = state->service_events;
	service_configuration.event_ring_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_EVENT_CAPACITY;
	return SparkGlm52ServiceInitialize(
		&state->service,
		&service_configuration);
}

static SparkStatus SparkGlm52RingServiceBackendInitializeServiceRuntime(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	uint32_t lane_capacity;
	SparkStatus status;

	lane_capacity =
		SparkGlm52RingServiceBackendServiceLaneCapacity(configuration);
	if (lane_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingServiceBackendAllocateServiceStorage(
		state,
		lane_capacity);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeKvArena(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializePrefixCache(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeScheduler(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeDspark(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeRequestApi(
			state,
			lane_capacity);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeServingEngine(
			state,
			lane_capacity);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeService(state);
	if (status == SPARK_STATUS_OK)
		state->service_runtime_ready = 1u;
	return status;
}

static int32_t SparkGlm52RingServiceBackendCreateListenSocket(
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

static void SparkGlm52RingServiceBackendDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkGlm52RingServiceBackendState *state;

	state = (SparkGlm52RingServiceBackendState *)completion_context;
	if (state == 0 || completion == 0)
		return;
	state->rank0_runtime_ready = 1u;
}

static SparkStatus SparkGlm52RingServiceBackendLoadDriver(
	SparkGlm52RingServiceBackendState *state,
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
	{
		SparkSetError(
			error_buffer,
			error_buffer_bytes,
			"driver program '%s' not found",
			configuration->driver_program_name != 0 ?
				configuration->driver_program_name : "");
		return SPARK_STATUS_NOT_FOUND;
	}
	memset(&create_request,0,sizeof(create_request));
	create_request.node_id = state->rank_plan.host_name;
	create_request.node_target = configuration->node_target;
	create_request.node_context = state->builder_result.node_context;
	create_request.completion_function =
		SparkGlm52RingServiceBackendDriverCompletion;
	create_request.completion_context = state;
	status = state->loaded_driver.interface->create(
		&create_request,
		&state->driver_instance);
	if (status != SPARK_STATUS_OK)
	{
		SparkSetError(
			error_buffer,
			error_buffer_bytes,
			"driver create failed status=%d",
			(int32_t)status);
		return status;
	}
	if (state->driver_instance == 0)
	{
		SparkSetError(
			error_buffer,
			error_buffer_bytes,
			"driver create returned null instance");
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendBuildNodeContext(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	SparkGlm52RingNodeContextBuilderConfiguration builder_configuration;
	SparkStatus status;

	memset(&builder_configuration,0,sizeof(builder_configuration));
	builder_configuration.abi_version =
		SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_ABI_VERSION;
	builder_configuration.descriptor_bytes =
		SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES;
	builder_configuration.rank_index = state->rank_plan.rank_index;
	builder_configuration.max_active_sequence_count =
		SparkGlm52RingServiceBackendMaxActive(configuration);
	builder_configuration.kv_pool_token_capacity = SPARK_GLM52_KV_POOL_TOKENS;
	builder_configuration.maximum_resident_sequence_count =
		SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_RESIDENT_SEQUENCE_COUNT;
	builder_configuration.port_base =
		SparkGlm52RingServiceBackendPortBase(configuration);
	builder_configuration.moe_pack_root = configuration->moe_pack_root;
	builder_configuration.stagepack_root = configuration->stagepack_root;
	builder_configuration.embedding_pack_path =
		configuration->embedding_pack_path;
	builder_configuration.node_target = configuration->node_target;
	builder_configuration.rank_plan = &state->rank_plan;
	fprintf(
		stderr,
		"ring_build_context load_builder rank=%u first=%u layers=%u max_active=%u builder=%s quantization=%s moe=%s stagepack=%s\n",
		state->rank_plan.rank_index,
		state->rank_plan.first_layer_index,
		state->rank_plan.layer_count,
		builder_configuration.max_active_sequence_count,
		configuration->node_context_builder_shared_object_path,
		SparkGlm52RingRuntimeQuantizationModeName(
			state->rank_plan.quantization_mode),
		configuration->moe_pack_root,
		configuration->stagepack_root);
	status = SparkGlm52RingNodeContextBuilderLoadInterfaceFromSharedObject(
		configuration->node_context_builder_shared_object_path,
		SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
		&state->builder_library);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"ring_build_context load_builder_status=%u\n",status);
		return status;
	}
	status = state->builder_library.builder_interface.initialize(
		&builder_configuration,
		&state->builder_state);
	if (status != SPARK_STATUS_OK || state->builder_state == 0)
	{
		fprintf(stderr,"ring_build_context initialize_status=%u state=%p\n",
			status,
			state->builder_state);
		return status == SPARK_STATUS_OK ?
			SPARK_STATUS_INVALID_ARGUMENT : status;
	}
	memset(&state->builder_result,0,sizeof(state->builder_result));
	status = state->builder_library.builder_interface.build(
		state->builder_state,
		&state->builder_result);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"ring_build_context build_status=%u\n",status);
		return status;
	}
	status = SparkGlm52RingNodeContextBuilderValidateResult(
		&state->builder_result,
		&state->rank_plan);
	if (status != SPARK_STATUS_OK)
		fprintf(stderr,"ring_build_context result_status=%u node=%p\n",
			status,
			state->builder_result.node_context);
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendAttachBuilderDriver(
	SparkGlm52RingServiceBackendState *state)
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

static SparkStatus SparkGlm52RingServiceBackendRank0Fail(
	SparkGlm52RingServiceBackendState *state,
	SparkStatus status,
	const char *error_buffer,
	const char *message)
{
	if (error_buffer != 0 && error_buffer[0] != '\0')
		SparkGlm52RingServiceBackendSetBlocker(state,error_buffer);
	else
		SparkGlm52RingServiceBackendSetBlocker(state,message);
	return status;
}

static SparkStatus SparkGlm52RingServiceBackendInitializeRunner(
	SparkGlm52RingServiceBackendState *state)
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

static SparkStatus SparkGlm52RingServiceBackendInitializeRank0LocalCuda(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration,
	char *error_buffer)
{
	char rank_buffer[16];
	char port_buffer[16];
	SparkStatus status;
	if (snprintf(rank_buffer,sizeof(rank_buffer),"%u",
			state->rank_plan.rank_index) < 0 ||
		snprintf(port_buffer,sizeof(port_buffer),"%u",
			SparkGlm52RingServiceBackendPortBase(configuration)) < 0 ||
		setenv("SPARKPIPE_RING_TRANSPORT_RANK",rank_buffer,1) != 0 ||
		setenv("SPARKPIPE_RING_TRANSPORT_PORT_BASE",port_buffer,1) != 0)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,SPARK_STATUS_INTERNAL_ERROR,error_buffer,
			"failed to configure rank0 transport environment");
	status = SparkHiddenTransportLoadInterfaceFromSharedObject(
		configuration->transport_shared_object_path,
		SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
		&state->transport_library);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to load production transport");
	status = SparkHiddenTransportOpen(
		&state->rank_plan.output_endpoint,
		&state->transport_library.transport_interface,
		SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
		&state->output_transport_session);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to open rank0 output transport");
	status = SparkGlm52RingServiceBackendBuildNodeContext(state,configuration);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build rank0 resident context");
	status = SparkGlm52RingServiceBackendLoadDriver(
		state,
		configuration,
		error_buffer,
		SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to load GLM52 driver");
	status = SparkGlm52RingServiceBackendAttachBuilderDriver(state);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to attach rank0 bridge driver");
	status = SparkGlm52RingServiceBackendInitializeRunner(state);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to initialize rank0 runner");
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendOpenFinalEventListener(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration,
	char *error_buffer)
{
	state->final_event_listen_fd =
		SparkGlm52RingServiceBackendCreateListenSocket(
			configuration->final_event_bind_address,
			state->final_event_route.listen_port);
	if (state->final_event_listen_fd < 0)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,
			SPARK_STATUS_ROUTE_NOT_FOUND,
			error_buffer,
			"failed to open rank0 final-event listener");
	if (SparkGlm52RingServiceBackendSetNonblocking(
			state->final_event_listen_fd) < 0)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,
			SPARK_STATUS_INTERNAL_ERROR,
			error_buffer,
			"failed to make final-event listener nonblocking");
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendInitializeRank0(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	char error_buffer[SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES];
	SparkStatus status;
	error_buffer[0] = '\0';
	status = SparkGlm52RingRuntimeBuildRankPlan(
		0u,
		SparkGlm52RingServiceBackendMaxActive(configuration),
		SparkGlm52RingServiceBackendPortBase(configuration),
		configuration->model_quantization_mode,
		&state->rank_plan,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build RING rank0 plan");
	status = SparkGlm52RingRuntimeBuildFinalEventRoute(
		SparkGlm52RingServiceBackendPortBase(configuration),
		&state->final_event_route,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build final event route");
	status = SparkGlm52RingRuntimeValidateStageMoePackFiles(
		&state->rank_plan,
		configuration->moe_pack_root,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52RingServiceBackendRank0Fail(
			state,status,error_buffer,"failed to validate rank0 MoE packs");
	if (configuration->cuda_resident_socket_path != 0)
	{
		if (strlen(configuration->cuda_resident_socket_path) >= sizeof(state->cuda_resident_socket_path))
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		snprintf(state->cuda_resident_socket_path,sizeof(state->cuda_resident_socket_path),"%s",configuration->cuda_resident_socket_path);
		state->cuda_resident_attached = 1u;
		state->trace_enabled = getenv("SPARKPIPE_RING_TRACE") != 0 ? 1u : 0u;
		status = SparkGlm52RingServiceBackendEnsureCudaResident(state);
		if (status != SPARK_STATUS_OK)
			fprintf(stderr,"ring_resident_not_ready_at_start status=%u\n",(uint32_t)status);
	}
	else
	{
		status = SparkGlm52RingServiceBackendInitializeRank0LocalCuda(state,configuration,error_buffer);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52RingServiceBackendOpenFinalEventListener(state,configuration,error_buffer);
	if (status != SPARK_STATUS_OK)
		return status;
	state->rank0_runtime_ready = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendInitialize(
	const SparkGlm52ServiceBackendConfiguration *configuration,
	void **backend_state)
{
	SparkGlm52RingServiceBackendState *state;
	SparkStatus status;

	if (configuration == 0 || backend_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (configuration->abi_version != SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES ||
		(configuration->flags &
			~SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_KNOWN_FLAGS) != 0u)
		return SPARK_STATUS_ABI_MISMATCH;
	if (SparkGlm52RingRuntimeQuantizationModeName(
			configuration->model_quantization_mode) == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (configuration->kv_logical_block_capacity != 0u &&
		configuration->kv_logical_block_capacity >
			UINT32_MAX - SPARK_GLM52_RING_SERVICE_BACKEND_REQUEST_CAPACITY)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (configuration->cuda_resident_socket_path != 0 &&
		configuration->kv_logical_block_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (configuration->cuda_resident_socket_path == 0 &&
		configuration->kv_logical_block_capacity != 0u &&
		configuration->kv_logical_block_capacity !=
			SPARK_GLM52_RING_SERVICE_BACKEND_GPU_BLOCK_COUNT)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	state = &SparkGlm52RingServiceBackendSingleton;
	memset(state,0,sizeof(*state));
	state->cuda_resident_fd = -1;
	state->final_event_listen_fd = -1;
	state->final_event_socket_fd = -1;
	state->work_output_socket_fd = -1;
	state->dspark_enabled = (configuration->flags &
		SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK) != 0u;
	state->mtp_enabled = (configuration->flags &
		SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP) != 0u;
	state->kv_logical_block_capacity =
		configuration->kv_logical_block_capacity != 0u
			? configuration->kv_logical_block_capacity
			: SPARK_GLM52_RING_SERVICE_BACKEND_DEFAULT_LOGICAL_BLOCK_COUNT;
	state->kv_physical_block_capacity =
		SPARK_GLM52_RING_SERVICE_BACKEND_GPU_BLOCK_COUNT;
	SparkLoadedModelDriverReset(&state->loaded_driver);
	SparkTokenizerReset(&state->tokenizer);
	state->initialized = 1u;
	*backend_state = state;
	state->session_id_base = SparkGlm52RingServiceBackendMonotonicNs();
	if (state->session_id_base == 0u)
	{
		SparkGlm52RingServiceBackendSetBlocker(
			state,
			"RING session ID clock is unavailable");
		return SPARK_STATUS_IO_ERROR;
	}
	state->cuda_resident_next_sequence_number = state->session_id_base;
	status = SparkGlm52RingServiceBackendRequireText(
		configuration->moe_pack_root,
		state,
		"MoE pack root is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendRequireText(
			configuration->stagepack_root,
			state,
			"GLM52 stagepack root is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendRequireText(
			configuration->transport_shared_object_path,
			state,
			"production hidden transport shared object is missing");
	if (status == SPARK_STATUS_OK)
	{
		if (strlen(configuration->transport_shared_object_path) >=
			sizeof(state->transport_shared_object_path))
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		else
			snprintf(state->transport_shared_object_path,
				sizeof(state->transport_shared_object_path),"%s",
				configuration->transport_shared_object_path);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendRequireText(
			configuration->driver_shared_object_path,
			state,
			"GLM52 driver shared object is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendRequireText(
			configuration->node_context_builder_shared_object_path,
			state,
			"GLM52 RING node-context builder shared object is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendRequireText(
			configuration->driver_program_name,
			state,
			"GLM52 driver program name is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendRequireText(
			configuration->final_event_bind_address,
			state,
			"final-event bind address is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendRequireText(
			configuration->final_event_return_host,
			state,
			"final-event return host is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendLoadTokenizer(
			state,
			configuration);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeServiceRuntime(
			state,
			configuration);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52RingServiceBackendInitializeRank0(
			state,
			configuration);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52RingServiceBackendSetBlocker(
			state,
			"RING service runtime failed to initialize");
		return status;
	}
	return SPARK_STATUS_OK;
}

static void SparkGlm52RingServiceBackendDestroy(void *backend_state)
{
	SparkGlm52RingServiceBackendState *state;

	state = (SparkGlm52RingServiceBackendState *)backend_state;
	if (state == 0)
		return;
	if (state->cuda_resident_fd >= 0)
		close(state->cuda_resident_fd);
	if (state->final_event_socket_fd >= 0)
		close(state->final_event_socket_fd);
	if (state->final_event_listen_fd >= 0)
		close(state->final_event_listen_fd);
	if (state->work_output_socket_fd >= 0)
		close(state->work_output_socket_fd);
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
	SparkGlm52RingNodeContextBuilderUnloadInterface(&state->builder_library);
	SparkHiddenTransportClose(state->output_transport_session);
	SparkHiddenTransportUnloadInterface(&state->transport_library);
	SparkTokenizerDestroy(&state->tokenizer);
	SparkGlm52RingServiceBackendFreeStorage(state);
	memset(state,0,sizeof(*state));
}

static SparkStatus SparkGlm52RingServiceBackendGetView(
	void *backend_state,
	SparkGlm52ServiceBackendView *view)
{
	SparkGlm52RingServiceBackendState *state;

	state = (SparkGlm52RingServiceBackendState *)backend_state;
	if (state == 0 || view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION;
	view->descriptor_bytes = SPARK_GLM52_SERVICE_BACKEND_VIEW_BYTES;
	view->runtime_initialized = state->initialized != 0u ? 1u : 0u;
	view->local_control_ready = state->service_runtime_ready != 0u &&
		state->rank0_runtime_ready != 0u &&
		state->first_blocker[0] == '\0' ? 1u : 0u;
	view->configured_kv_context_limit_tokens =
		SPARK_GLM52_RING_SERVICE_BACKEND_CONTEXT_TOKENS;
	view->configured_max_active_sequences =
		state->rank_plan.logical_lane_capacity;
	view->adaptive_decode_batch_width =
		SparkGlm52RequestApiCurrentPipelineBatchWidth(&state->request_api);
	view->decode_batch_capacity = state->request_api.decode_batch_target;
	view->prefill_wave_token_count =
		SPARK_GLM52_RING_SERVICE_BACKEND_PREFILL_WAVE_TOKENS;
	view->transport_capability_flags =
		state->transport_library.transport_interface.capability_flags;
	view->speculation_configuration_flags =
		(state->dspark_enabled != 0u ?
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK : 0u) |
		(state->mtp_enabled != 0u ?
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP : 0u);
	view->request_api_configuration_flags =
		state->request_api.configuration_flags;
	view->release_generation = SparkGlm52RingServiceBackendEnvironmentU64(
		"SPARKPIPE_RELEASE_GENERATION");
	view->service = state->service_runtime_ready != 0u ? &state->service : 0;
	view->tokenizer = state->tokenizer_ready != 0u ? &state->tokenizer : 0;
	view->first_blocker = state->first_blocker;
	view->release_id = SparkGlm52RingServiceBackendEnvironmentText(
		"SPARKPIPE_RELEASE_ID");
	view->release_git_commit = SparkGlm52RingServiceBackendEnvironmentText(
		"SPARKPIPE_RELEASE_GIT_COMMIT");
	view->transport_shared_object_path = state->transport_shared_object_path;
	return SPARK_STATUS_OK;
}

static void SparkGlm52RingServiceBackendAcceptFinalEventSocket(
	SparkGlm52RingServiceBackendState *state)
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
	if (SparkGlm52RingServiceBackendSetNonblocking(fd) < 0)
	{
		close(fd);
		state->final_event_receive_error_count += 1u;
		return;
	}
	state->final_event_socket_fd = fd;
}

static void SparkGlm52RingServiceBackendRecordFinalEvent(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingRuntimeFinalEvent *event)
{
	uint32_t token_count;

	if (event->magic != SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_MAGIC ||
		event->descriptor_bytes !=
			SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES)
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

static SparkStatus SparkGlm52RingServiceBackendReadFinalEvent(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52RingRuntimeFinalEvent *event_out)
{
	ssize_t got;
	uint32_t remaining;

	if (state == 0 || event_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkGlm52RingServiceBackendAcceptFinalEventSocket(state);
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

static int32_t SparkGlm52RingServiceBackendFindDecodeLane(
	const SparkGlm52RequestApiDispatch *decode_dispatch,
	const SparkGlm52RingRuntimeFinalEvent *event)
{
	uint32_t lane_index;

	if (decode_dispatch == 0 || event == 0)
		return -1;
	for (lane_index = 0u;
		 lane_index < decode_dispatch->request_count;
		 ++lane_index)
	{
		if (decode_dispatch->request_ids[lane_index] == event->request_id &&
			decode_dispatch->sequence_ids[lane_index] == event->sequence_id)
			return (int32_t)lane_index;
	}
	return -2;
}

static SparkGlm52RingServiceBackendPendingDecode *SparkGlm52RingServiceBackendFindFreePendingDecode(
	SparkGlm52RingServiceBackendState *state)
{
	uint32_t index;

	if (state == 0)
		return 0;
	for (index = 0u;
		 index < SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY;
		 ++index)
	{
		if (state->pending_decodes[index].state ==
			SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_FREE)
			return &state->pending_decodes[index];
	}
	return 0;
}

static SparkGlm52RingServiceBackendPendingDecode *SparkGlm52RingServiceBackendFindPendingDecodeForEvent(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingRuntimeFinalEvent *event,
	uint32_t *lane_index_out)
{
	uint32_t index;
	int32_t lane;

	if (state == 0 || event == 0 || lane_index_out == 0)
		return 0;
	for (index = 0u;
		 index < SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_CAPACITY;
		 ++index)
	{
		if (state->pending_decodes[index].state !=
			SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE)
			continue;
		lane = SparkGlm52RingServiceBackendFindDecodeLane(
			&state->pending_decodes[index].dispatch,
			event);
		if (lane >= 0)
		{
			*lane_index_out = (uint32_t)lane;
			return &state->pending_decodes[index];
		}
	}
	return 0;
}

static void SparkGlm52RingServiceBackendDropEarlyFinalEvent(
	SparkGlm52RingServiceBackendState *state,
	uint32_t event_index)
{
	uint32_t read_index;
	uint32_t write_index;
	uint32_t shift_index;

	if (state == 0 ||
		event_index >= state->early_final_event_count)
		return;
	for (shift_index = event_index + 1u;
		 shift_index < state->early_final_event_count;
		 ++shift_index)
	{
		read_index = (state->early_final_event_head + shift_index) %
			SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY;
		write_index = (state->early_final_event_head + shift_index - 1u) %
			SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY;
		state->early_final_events[write_index] =
			state->early_final_events[read_index];
	}
	state->early_final_event_count -= 1u;
}

static SparkStatus SparkGlm52RingServiceBackendStashEarlyFinalEvent(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingRuntimeFinalEvent *event)
{
	uint32_t tail;

	if (state == 0 || event == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->early_final_event_count >=
		SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY)
	{
		state->early_final_event_head =
			(state->early_final_event_head + 1u) %
			SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY;
		state->early_final_event_count -= 1u;
		state->final_event_receive_error_count += 1u;
	}
	tail = (state->early_final_event_head + state->early_final_event_count) %
		SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY;
	state->early_final_events[tail] = *event;
	state->early_final_event_count += 1u;
	return SPARK_STATUS_BUSY;
}

static SparkStatus SparkGlm52RingServiceBackendCompleteEarlyFinalEvents(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52RingServiceBackendPendingDecode *pending)
{
	SparkGlm52RingRuntimeFinalEvent event;
	SparkStatus status;
	uint32_t event_index;
	uint32_t ring_index;
	int32_t lane;

	if (state == 0 || pending == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	event_index = 0u;
	while (pending->state ==
			SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE &&
		event_index < state->early_final_event_count)
	{
		ring_index = (state->early_final_event_head + event_index) %
			SPARK_GLM52_RING_SERVICE_BACKEND_EARLY_FINAL_EVENT_CAPACITY;
		lane = SparkGlm52RingServiceBackendFindDecodeLane(
			&pending->dispatch,
			&state->early_final_events[ring_index]);
		if (lane < 0)
		{
			event_index += 1u;
			continue;
		}
		event = state->early_final_events[ring_index];
		SparkGlm52RingServiceBackendDropEarlyFinalEvent(state,event_index);
		status = SparkGlm52RingServiceBackendCompletePendingFinalEvent(
			state,
			&event);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
			return status;
	}
	return pending->state ==
		SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE ?
		SPARK_STATUS_BUSY : SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendRegisterPendingDecode(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	const SparkGlm52ServingDecodeResult *decode_result,
	SparkGlm52RingServiceBackendPendingDecode **pending_out)
{
	SparkGlm52RingServiceBackendPendingDecode *pending;
	SparkStatus status;

	if (state == 0 || decode_dispatch == 0 || decode_result == 0 ||
		decode_dispatch->request_dispatch == 0 || pending_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	pending = SparkGlm52RingServiceBackendFindFreePendingDecode(state);
	if (pending == 0)
		return SPARK_STATUS_BUSY;
	memset(pending,0,sizeof(*pending));
	pending->state =
		SPARK_GLM52_RING_SERVICE_BACKEND_PENDING_DECODE_STATE_ACTIVE;
	pending->dispatch = *decode_dispatch->request_dispatch;
	pending->result = *decode_result;
	*pending_out = pending;
	status = SparkGlm52RingServiceBackendCompleteEarlyFinalEvents(
		state,
		pending);
	if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
		return status;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendCompletePendingDecode(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52RingServiceBackendPendingDecode *pending)
{
	SparkStatus status;

	if (state == 0 || pending == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->trace_enabled != 0u)
	{
		uint64_t completion_ns;
		uint64_t request_id;

		completion_ns = SparkGlm52RingServiceBackendMonotonicNs();
		request_id = pending->dispatch.request_count != 0u ?
			pending->dispatch.request_ids[0u] : 0u;
		if (completion_ns != 0u &&
			pending->trace_submit_time_ns != 0u)
			fprintf(stderr,
				"ring_decode_flight_ns=%llu kind=%u request=%llu\n",
				(unsigned long long)(
					completion_ns - pending->trace_submit_time_ns),
				pending->dispatch.kind,(unsigned long long)request_id);
		state->trace_last_decode_completion_ns = completion_ns;
	}
	status = SparkGlm52ServingEngineCompleteDecodeDispatch(
		&state->serving_engine,
		&pending->dispatch,
		&pending->result);
	memset(pending,0,sizeof(*pending));
	return status;
}

static void SparkGlm52RingServiceBackendTraceFinalEvent(
	const SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingRuntimeFinalEvent *event,
	const SparkGlm52RingServiceBackendPendingDecode *pending)
{
	uint32_t token_index;

	if (state == 0 || event == 0 || pending == 0 || state->trace_enabled == 0u)
		return;
	fprintf(stderr,
		"ring_trace final_event request=%llu sequence=%llu position=%llu kind=%u dispatch_flags=0x%x completion_flags=0x%x tokens=%u ids=",
		(unsigned long long)event->request_id,
		(unsigned long long)event->sequence_id,
		(unsigned long long)event->sequence_position,
		pending->dispatch.kind,
		pending->dispatch.flags,
		event->completion_flags,
		event->token_count);
	for (token_index = 0u; token_index < event->token_count; ++token_index)
		fprintf(stderr,"%s%u",token_index == 0u ? "" : ",",
			event->token_ids[token_index]);
	fprintf(stderr," status=%u\n",event->status);
}

static SparkStatus SparkGlm52RingServiceBackendCompletePendingFinalEvent(
	SparkGlm52RingServiceBackendState *state,
	const SparkGlm52RingRuntimeFinalEvent *event)
{
	SparkGlm52RingServiceBackendPendingDecode *pending;
	uint32_t lane_index;
	uint32_t token_count;
	uint32_t dspark_expected;

	if (state == 0 || event == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkGlm52RingServiceBackendRecordFinalEvent(state,event);
	if (event->magic != SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_MAGIC ||
		event->descriptor_bytes !=
			SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES ||
		(event->extension_flags &
			~SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_KNOWN_FLAGS) != 0u)
		return SPARK_STATUS_VALIDATION_FAILED;
	pending = SparkGlm52RingServiceBackendFindPendingDecodeForEvent(
		state,
		event,
		&lane_index);
	if (pending == 0)
		return SparkGlm52RingServiceBackendStashEarlyFinalEvent(state,event);
	if (event->status != (uint32_t)SPARK_STATUS_OK)
	{
		SparkGlm52RingServiceBackendTraceFinalEvent(state,event,pending);
		return SparkGlm52RingServiceBackendFailPendingDecode(
			state,
			pending,
			(SparkStatus)event->status);
	}
	if ((event->completion_flags &
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u ||
		(((event->completion_flags &
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS) != 0u) !=
			 (event->draft_token_count != 0u)) ||
		event->draft_token_count >
			SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT)
		return SPARK_STATUS_VALIDATION_FAILED;
	dspark_expected =
		(pending->dispatch.flags &
			(SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE |
			 SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY)) != 0u;
	if (dspark_expected != 0u)
	{
		if ((event->extension_flags &
				SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_FLAG_DSPARK_DRAFT) == 0u ||
			event->dspark_draft.abi_version != SPARK_GLM52_DSPARK_ABI_VERSION ||
			event->dspark_draft.descriptor_bytes !=
				SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES ||
			event->dspark_draft.token_count == 0u)
			return SPARK_STATUS_VALIDATION_FAILED;
		pending->dspark_drafts[lane_index] = event->dspark_draft;
		pending->dspark_draft_valid[lane_index] = 1u;
	}
	if (pending->lane_done[lane_index] != 0u)
		return SPARK_STATUS_OK;
	token_count = event->token_count;
	if (token_count == 0u ||
		token_count > SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE ||
		token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY)
		return SPARK_STATUS_VALIDATION_FAILED;
	SparkGlm52RingServiceBackendTraceFinalEvent(state,event,pending);
	memcpy(
		pending->result.token_ids[lane_index],
		event->token_ids,
		token_count * sizeof(pending->result.token_ids[lane_index][0u]));
	pending->result.token_counts[lane_index] = token_count;
	pending->result.draft_token_counts[lane_index] = event->draft_token_count;
	memcpy(
		pending->result.draft_token_ids[lane_index],
		event->draft_token_ids,
		event->draft_token_count *
			sizeof(pending->result.draft_token_ids[lane_index][0u]));
	pending->lane_done[lane_index] = 1u;
	pending->done_count += 1u;
	if (pending->done_count != pending->dispatch.request_count)
		return SPARK_STATUS_BUSY;
	return SparkGlm52RingServiceBackendCompletePendingDecode(state,pending);
}

static SparkStatus SparkGlm52RingServiceBackendPumpFinalEvents(
	SparkGlm52RingServiceBackendState *state)
{
	SparkGlm52RingRuntimeFinalEvent event;
	SparkStatus status;
	uint32_t event_count;

	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (event_count = 0u;
		 event_count < SPARK_GLM52_RING_SERVICE_BACKEND_FINAL_EVENT_PUMP_BUDGET;
		 ++event_count)
	{
		status = SparkGlm52RingServiceBackendReadFinalEvent(state,&event);
		if (status == SPARK_STATUS_BUSY)
			return SPARK_STATUS_OK;
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52RingServiceBackendCompletePendingFinalEvent(
			state,
			&event);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY &&
			status != SPARK_STATUS_NOT_FOUND)
			return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingServiceBackendPump(
	void *backend_state,
	uint32_t max_dispatch_steps,
	SparkGlm52ServiceStats *stats_out)
{
	SparkGlm52RingServiceBackendState *state;
	SparkStatus event_status;
	SparkStatus release_status;
	SparkStatus resident_status;
	SparkStatus service_status;
	SparkStatus work_status;
	uint32_t service_can_dispatch;

	state = (SparkGlm52RingServiceBackendState *)backend_state;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	resident_status =
		SparkGlm52RingServiceBackendPumpCudaResidentResponses(state);
	if (resident_status != SPARK_STATUS_OK &&
		resident_status != SPARK_STATUS_BUSY)
		state->final_event_receive_error_count += 1u;
	release_status = SparkGlm52RingServiceBackendPumpSequenceReleases(state);
	if (release_status != SPARK_STATUS_OK && release_status != SPARK_STATUS_BUSY)
		state->final_event_receive_error_count += 1u;
	work_status = SparkGlm52RingServiceBackendPumpWorkOutput(state);
	if (work_status != SPARK_STATUS_OK && work_status != SPARK_STATUS_BUSY)
		state->final_event_receive_error_count += 1u;
	if (state->rank0_runtime_ready != 0u)
		(void)SparkGlm52ResidentDecodeStageProductionRunnerProgress(
			&state->runner);
	event_status = SparkGlm52RingServiceBackendPumpFinalEvents(state);
	if (event_status != SPARK_STATUS_OK)
		state->final_event_receive_error_count += 1u;
	service_status = SPARK_STATUS_OK;
	service_can_dispatch = state->service_runtime_ready != 0u &&
		state->rank0_runtime_ready != 0u &&
		state->first_blocker[0] == '\0' &&
		SparkGlm52RingServiceBackendFindFreePendingDecode(state) != 0;
	if (service_can_dispatch != 0u)
		service_status = SparkGlm52ServicePump(
			&state->service,max_dispatch_steps,stats_out);
	else if (stats_out != 0)
	{
		if (state->service_runtime_ready != 0u)
			(void)SparkGlm52ServiceGetStats(&state->service,stats_out);
		else
			memset(stats_out,0,sizeof(*stats_out));
	}
	if (service_can_dispatch == 0u &&
		state->service_runtime_ready != 0u &&
		state->rank0_runtime_ready != 0u &&
		state->first_blocker[0] == '\0')
		service_status = SPARK_STATUS_BUSY;
	resident_status =
		SparkGlm52RingServiceBackendPumpCudaResidentResponses(state);
	release_status = SparkGlm52RingServiceBackendPumpSequenceReleases(state);
	work_status = SparkGlm52RingServiceBackendPumpWorkOutput(state);
	if (resident_status != SPARK_STATUS_OK &&
		resident_status != SPARK_STATUS_BUSY)
		return resident_status;
	if (release_status != SPARK_STATUS_OK &&
		release_status != SPARK_STATUS_BUSY)
		return release_status;
	if (work_status != SPARK_STATUS_OK && work_status != SPARK_STATUS_BUSY)
		return work_status;
	return service_status;
}

static uint32_t SparkGlm52RingServiceBackendTransportPollEvents(
	uint32_t transport_events)
{
	uint32_t events;

	events = 0u;
	if ((transport_events & SPARK_HIDDEN_TRANSPORT_POLL_READ) != 0u)
		events |= SPARK_GLM52_SERVICE_BACKEND_POLL_READ;
	if ((transport_events & SPARK_HIDDEN_TRANSPORT_POLL_WRITE) != 0u)
		events |= SPARK_GLM52_SERVICE_BACKEND_POLL_WRITE;
	return events;
}

static void SparkGlm52RingServiceBackendAppendOutputTransportPollDescriptors(
	SparkGlm52RingServiceBackendState *state,
	SparkGlm52ServiceBackendPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count)
{
	SparkHiddenTransportPollDescriptor transport_descriptors[4];
	SparkStatus status;
	uint32_t transport_count;
	uint32_t transport_index;
	uint32_t events;

	if (state == 0 || descriptors == 0 || descriptor_count == 0 ||
		state->output_transport_session == 0)
		return;
	transport_count = 0u;
	status = SparkHiddenTransportGetPollDescriptors(
		state->output_transport_session,
		transport_descriptors,
		4u,
		&transport_count);
	if (status != SPARK_STATUS_OK)
		return;
	for (transport_index = 0u;
		 transport_index < transport_count &&
		 *descriptor_count < descriptor_capacity;
		 ++transport_index)
	{
		events = SparkGlm52RingServiceBackendTransportPollEvents(
			transport_descriptors[transport_index].events);
		if (transport_descriptors[transport_index].fd < 0 || events == 0u)
			continue;
		memset(
			&descriptors[*descriptor_count],
			0,
			sizeof(descriptors[*descriptor_count]));
		descriptors[*descriptor_count].descriptor_bytes =
			SPARK_GLM52_SERVICE_BACKEND_POLL_DESCRIPTOR_BYTES;
		descriptors[*descriptor_count].fd =
			transport_descriptors[transport_index].fd;
		descriptors[*descriptor_count].events = events;
		*descriptor_count += 1u;
	}
}

static SparkStatus SparkGlm52RingServiceBackendGetPollDescriptors(
	void *backend_state,
	SparkGlm52ServiceBackendPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out)
{
	SparkGlm52RingServiceBackendState *state;
	uint32_t descriptor_count;
	int32_t fd;

	state = (SparkGlm52RingServiceBackendState *)backend_state;
	if (state == 0 || descriptor_count_out == 0 ||
		(descriptor_capacity != 0u && descriptors == 0))
		return SPARK_STATUS_INVALID_ARGUMENT;
	descriptor_count = 0u;
	if (state->cuda_resident_fd >= 0 &&
		descriptor_count < descriptor_capacity)
	{
		memset(&descriptors[descriptor_count],0,sizeof(descriptors[descriptor_count]));
		descriptors[descriptor_count].descriptor_bytes =
			SPARK_GLM52_SERVICE_BACKEND_POLL_DESCRIPTOR_BYTES;
		descriptors[descriptor_count].fd = state->cuda_resident_fd;
		descriptors[descriptor_count].events =
			SPARK_GLM52_SERVICE_BACKEND_POLL_READ;
		descriptor_count += 1u;
	}
	if (state->final_event_socket_fd >= 0)
		fd = state->final_event_socket_fd;
	else
		fd = state->final_event_listen_fd;
	if (fd >= 0 && descriptor_count < descriptor_capacity)
	{
		memset(&descriptors[descriptor_count],0,sizeof(descriptors[descriptor_count]));
		descriptors[descriptor_count].descriptor_bytes =
			SPARK_GLM52_SERVICE_BACKEND_POLL_DESCRIPTOR_BYTES;
		descriptors[descriptor_count].fd = fd;
		descriptors[descriptor_count].events =
			SPARK_GLM52_SERVICE_BACKEND_POLL_READ;
		descriptor_count += 1u;
	}
	if (state->work_output_socket_fd >= 0 &&
		state->work_queue_count != 0u &&
		descriptor_count < descriptor_capacity)
	{
		memset(&descriptors[descriptor_count],0,sizeof(descriptors[descriptor_count]));
		descriptors[descriptor_count].descriptor_bytes =
			SPARK_GLM52_SERVICE_BACKEND_POLL_DESCRIPTOR_BYTES;
		descriptors[descriptor_count].fd = state->work_output_socket_fd;
		descriptors[descriptor_count].events =
			SPARK_GLM52_SERVICE_BACKEND_POLL_WRITE;
		descriptor_count += 1u;
	}
	SparkGlm52RingServiceBackendAppendOutputTransportPollDescriptors(
		state,
		descriptors,
		descriptor_capacity,
		&descriptor_count);
	*descriptor_count_out = descriptor_count;
	return SPARK_STATUS_OK;
}

static const SparkGlm52ServiceBackendInterface SparkGlm52RingServiceBackendInterface =
{
	SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION,
	SPARK_GLM52_SERVICE_BACKEND_INTERFACE_BYTES,
	SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_RING_RUNTIME |
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_SERVICE_RUNTIME |
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_TOKENIZER |
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS,
	0u,
	SparkGlm52RingServiceBackendInitialize,
	SparkGlm52RingServiceBackendDestroy,
	SparkGlm52RingServiceBackendGetView,
	SparkGlm52RingServiceBackendPump,
	SparkGlm52RingServiceBackendGetPollDescriptors
};

const SparkGlm52ServiceBackendInterface *SparkGlm52ServiceBackendGetInterface(void)
{
	return &SparkGlm52RingServiceBackendInterface;
}
