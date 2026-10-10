#include <cuda_runtime.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_memory_buffer.h"
#include "sparkpipe/spark_serving_cache_admission.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_stage_serving_adapter.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"

typedef struct SparkStageServingState
{
	const SparkStageServingModel *model;
	SparkStageRunner runner;
	SparkStageRunnerConfiguration runner_config;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	char *pack_path;
	char *stray_working_set_path;
	char *device_backend_path;
	char *device_local_host;
	uint32_t max_rows;
	SparkTpDeviceCollectiveConfig device_config;
	SparkWeightdClient *lane_client;
	SparkTpDeviceCollectiveTopology device_topology;
	int device_collective_present;
	SparkMemoryBuffer positions_host;
	SparkMemoryBuffer context_host;
	SparkMemoryBuffer state_host;
	uint64_t wave_host_ns[5];
	uint32_t wave_host_count;
	SparkMemoryBuffer positions_device;
	SparkMemoryBuffer context_device;
	SparkMemoryBuffer state_device;
	SparkMemoryBuffer runs_host;
	SparkMemoryBuffer runs_device;
	SparkMemoryBuffer seqslot_host;
	SparkMemoryBuffer seqslot_device;
	SparkMemoryBuffer order_host;
	SparkMemoryBuffer order_device;
	SparkMemoryBuffer output_tokens;
	SparkMemoryBuffer output_scores;
	SparkStageModuleLedger ledger;
	SparkStageKvBinding kv;
	SparkModelDriverCacheLane cache_lanes[SPARK_STAGE_SERVING_MAX_LANES];
	SparkModelDriverCacheLane prefetch_lanes[SPARK_STAGE_SERVING_MAX_LANES];
	atomic_uint lane_states[SPARK_STAGE_SERVING_MAX_LANES];
	atomic_ullong reset_generation;
	SparkModelDriverCompletion publish_completion;
	uint32_t publish_completed;
	uint32_t row_slots[SPARK_STAGE_SERVING_MAX_ROWS];
	uint64_t row_sequence_ids[SPARK_STAGE_SERVING_MAX_ROWS];
	uint64_t row_positions[SPARK_STAGE_SERVING_MAX_ROWS];
	uint32_t completion_tokens[SPARK_STAGE_SERVING_MAX_ROWS];
} SparkStageServingState;

static const SparkStageServingModel *StageServingBoundModel;

static uint32_t StageServingJsonU32(SparkJsonDocument *doc, int32_t root, const char *name, uint32_t fallback)
{
	uint32_t value = 0u;
	int32_t token = SparkJsonFindObjectMember(doc, root, name);
	if ( token >= 0 && SparkJsonGetUInt32(doc, token, &value) == SPARK_STATUS_OK )
		return value;
	return fallback;
}

static SparkStatus StageServingEnvU64(const char *name, uint64_t *value)
{
	const char *text = getenv(name);
	char *end = 0;
	if ( text == 0 || *text == '\0' )
		return SPARK_STATUS_INVALID_ARGUMENT;
	errno = 0;
	*value = strtoull(text, &end, 10);
	return errno != 0 || end == text || *end != '\0' ? SPARK_STATUS_PARSE_ERROR : SPARK_STATUS_OK;
}

static SparkStatus StageServingRunnerWeights(SparkStageServingState *state)
{
	SparkStageRunnerConfiguration *config = &state->runner_config;
	const char *digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	SparkStatus pool, spine;
	config->weights.socket_path = getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
	pool = StageServingEnvU64("SPARK_WEIGHTD_EXPERT_POOL_BYTES", &config->weights.expert_pool_bytes);
	spine = StageServingEnvU64("SPARK_WEIGHTD_SPINE_BUDGET_BYTES", &config->weights.spine_budget_bytes);
	if ( digest == 0 || strlen(digest) != SPARK_STAGE_RUNNER_DIGEST_HEX ||
		pool == SPARK_STATUS_INVALID_ARGUMENT || spine == SPARK_STATUS_INVALID_ARGUMENT )
	{
		fprintf(stderr, "%s: the runner needs %s, SPARK_WEIGHTD_EXPERT_POOL_BYTES and SPARK_WEIGHTD_SPINE_BUDGET_BYTES\n",
			state->model->module_tag, SPARK_WEIGHTD_ATTACH_ENV_SHA256);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( pool != SPARK_STATUS_OK || spine != SPARK_STATUS_OK )
		return SPARK_STATUS_PARSE_ERROR;
	memcpy(config->weights.pack_sha256, digest, SPARK_STAGE_RUNNER_DIGEST_HEX + 1u);
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingCopyMember(SparkJsonDocument *doc, int32_t object, const char *name, char **out)
{
	int32_t token = SparkJsonFindObjectMember(doc, object, name);
	if ( token < 0 )
		return SPARK_STATUS_OK;
	return SparkJsonCopyString(doc, token, out) == SPARK_STATUS_OK ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR;
}

static SparkStatus StageServingLoadSessionPorts(SparkStageServingState *state, SparkJsonDocument *doc, int32_t dev,
	uint32_t peer_count)
{
	int32_t ports = SparkJsonFindObjectMember(doc, dev, "session_ports");
	uint32_t row, column, value;
	if ( ports < 0 || SparkJsonGetArrayElementCount(doc, ports) != peer_count )
		return SPARK_STATUS_SCHEMA_ERROR;
	for ( row = 0u; row < peer_count; ++row )
	{
		int32_t row_token = SparkJsonGetArrayElement(doc, ports, row);
		if ( row_token < 0 || !SparkJsonTokenIsType(doc, row_token, SPARK_JSON_TOKEN_ARRAY) ||
			SparkJsonGetArrayElementCount(doc, row_token) != peer_count )
			return SPARK_STATUS_SCHEMA_ERROR;
		for ( column = 0u; column < peer_count; ++column )
		{
			int32_t cell = SparkJsonGetArrayElement(doc, row_token, column);
			value = 0u;
			if ( cell < 0 || SparkJsonGetUInt32(doc, cell, &value) != SPARK_STATUS_OK || value > 65535u ||
				(row == column ? value != 0u : value == 0u) )
				return SPARK_STATUS_SCHEMA_ERROR;
			state->device_topology.session_ports[row][column] = (uint16_t)value;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingLoadDeviceCollective(SparkStageServingState *state, SparkJsonDocument *doc, int32_t dev)
{
	const SparkModelServingAdapterDescriptor *descriptor = state->model->descriptor;
	int32_t token, hosts;
	uint32_t peer_count, index;
	uint64_t identifier = 0u;
	SparkStatus status;
	memset(&state->device_config, 0, sizeof(state->device_config));
	memset(&state->device_topology, 0, sizeof(state->device_topology));
	state->device_topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	state->device_topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
	state->device_config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	token = SparkJsonFindObjectMember(doc, dev, "backend");
	if ( token >= 0 && SparkJsonStringEquals(doc, token, "nccl") )
		state->device_config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	else if ( token >= 0 && SparkJsonStringEquals(doc, token, "hidden_transport") )
		state->device_config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	else
		return SPARK_STATUS_SCHEMA_ERROR;
	status = StageServingCopyMember(doc, dev, "backend_module_path", &state->device_backend_path);
	if ( status == SPARK_STATUS_OK )
		status = StageServingCopyMember(doc, dev, "local_host", &state->device_local_host);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->device_config.backend_module_path = state->device_backend_path;
	state->device_config.local_host = state->device_local_host;
	token = SparkJsonFindObjectMember(doc, dev, "collective_identifier");
	if ( token >= 0 && SparkJsonGetUInt64(doc, token, &identifier) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->device_config.collective_identifier = identifier;
	state->device_config.control_port_base = StageServingJsonU32(doc, dev, "listen_port", 0u);
	state->device_config.connect_timeout_milli = StageServingJsonU32(doc, dev, "connect_timeout_milli", 5000u);
	state->device_config.operation_timeout_milli = StageServingJsonU32(doc, dev, "operation_timeout_milli", 30000u);
	state->device_config.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	state->device_config.credit_count = 8u;
	state->device_config.local_hidden_dimension = descriptor->boundary_element_count;
	state->device_config.max_active_sequence_count =
		state->model->collective_sequences(state->runner_config.max_input_row_count, state->runner_config.tp_degree);
	hosts = SparkJsonFindObjectMember(doc, dev, "peer_hosts");
	peer_count = hosts >= 0 ? SparkJsonGetArrayElementCount(doc, hosts) : 0u;
	if ( peer_count == 0u || peer_count > SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->device_topology.rank_count = peer_count;
	state->device_topology.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE;
	token = SparkJsonFindObjectMember(doc, dev, "wait_mode");
	if ( token >= 0 && SparkJsonStringEquals(doc, token, "hardware") )
		state->device_topology.wait_mode = SPARK_TP_DEVICE_COLLECTIVE_WAIT_HARDWARE;
	else if ( token >= 0 && !SparkJsonStringEquals(doc, token, "spin") )
		return SPARK_STATUS_SCHEMA_ERROR;
	status = StageServingLoadSessionPorts(state, doc, dev, peer_count);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( index = 0u; index < peer_count; ++index )
	{
		int32_t peer = SparkJsonGetArrayElement(doc, hosts, index);
		char *text = 0;
		if ( peer < 0 || SparkJsonCopyString(doc, peer, &text) != SPARK_STATUS_OK )
			return SPARK_STATUS_SCHEMA_ERROR;
		strncpy(state->device_topology.rank_hosts[index], text, SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES - 1u);
		free(text);
	}
	state->device_collective_present = 1;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingParse(SparkStageServingState *state, const SparkModelServingAdapterConfiguration *configuration,
	SparkJsonDocument *doc, int32_t root)
{
	const SparkModelServingAdapterDescriptor *descriptor = state->model->descriptor;
	SparkStageRunnerConfiguration *config = &state->runner_config;
	uint32_t world_size;
	int32_t token;
	token = SparkJsonFindObjectMember(doc, root, "stage_pack_path");
	if ( token < 0 || SparkJsonCopyString(doc, token, &state->pack_path) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	if ( StageServingCopyMember(doc, root, "stray_working_set_path", &state->stray_working_set_path) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	token = SparkJsonFindObjectMember(doc, root, "state_budget_bytes");
	if ( token < 0 || SparkJsonGetUInt64(doc, token, &config->state_budget_bytes) != SPARK_STATUS_OK ||
		config->state_budget_bytes == 0u )
	{
		fprintf(stderr, "%s adapter: state_budget_bytes must name the rank's state budget\n", state->model->module_tag);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( StageServingJsonU32(doc, root, "hidden", descriptor->boundary_element_count) != descriptor->boundary_element_count )
		return SPARK_STATUS_SCHEMA_ERROR;
	config->tp_degree = StageServingJsonU32(doc, root, "tp_degree", 1u);
	world_size = StageServingJsonU32(doc, root, "world_size", descriptor->stage_count);
	if ( config->tp_degree != state->model->tp_degree || world_size != descriptor->stage_count ||
		world_size % config->tp_degree != 0u )
		return SPARK_STATUS_SCHEMA_ERROR;
	config->stage_index = configuration->stage_index / config->tp_degree;
	config->stage_count = world_size / config->tp_degree;
	config->tp_rank = configuration->stage_index % config->tp_degree;
	config->max_active_sequence_count = StageServingJsonU32(doc, root, "max_sequences",
		configuration->runtime_limits.max_active_sequence_count);
	config->max_input_row_count = StageServingJsonU32(doc, root, "max_rows", configuration->runtime_limits.max_input_row_count);
	config->resident_sequence_capacity = StageServingJsonU32(doc, root, "resident_capacity",
		configuration->runtime_limits.resident_sequence_capacity);
	config->kv_pages_per_sequence = StageServingJsonU32(doc, root, "kv_pages", 2u);
	config->kv_page_bytes = 0u;
	token = SparkJsonFindObjectMember(doc, root, "device_collective");
	if ( token >= 0 )
		return StageServingLoadDeviceCollective(state, doc, token);
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingAcquireLane(SparkStageServingState *state)
{
	const char *socket = getenv("SPARK_WEIGHTD_SOCKET");
	const char *lane_text = getenv("SPARK_WEIGHTD_LANE");
	SparkWeightdMeshTopology mesh_topology;
	uint32_t requested_lane = SPARK_WEIGHTD_LANE_NONE, resolved_lane;
	char *end = 0;
	unsigned long value;
	if ( socket == 0 || socket[0] == '\0' )
		return SPARK_STATUS_UNSUPPORTED;
	if ( lane_text != 0 && lane_text[0] != '\0' )
	{
		errno = 0;
		value = strtoul(lane_text, &end, 10);
		if ( lane_text[0] < '0' || lane_text[0] > '9' || end == lane_text || *end != '\0' || errno != 0 ||
			value >= SPARK_WEIGHTD_MESH_MAX_LANES )
			return SPARK_STATUS_INVALID_ARGUMENT;
		requested_lane = (uint32_t)value;
	}
	if ( SparkTpDeviceCollectiveMeshTopology(state->runner_config.tp_rank, state->runner_config.tp_degree,
		&mesh_topology) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	if ( SparkWeightdClientConnect(socket, &state->lane_client, 0) != SPARK_STATUS_OK )
	{
		state->lane_client = 0;
		return SPARK_STATUS_IO_ERROR;
	}
	if ( SparkWeightdClientLaneAcquire(state->lane_client, requested_lane, &mesh_topology, &resolved_lane,
		(uint64_t)state->device_config.connect_timeout_milli * 1000000ull) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s: mesh lane acquire failed requested=%u status=cold19-shared-lane\n",
			state->model->module_tag, requested_lane);
		(void)SparkWeightdClientClose(state->lane_client);
		state->lane_client = 0;
		return SPARK_STATUS_NO_LANE;
	}
	fprintf(stderr, "%s: mesh lane acquired requested=%u resolved=%u shared-owner\n",
		state->model->module_tag, requested_lane, resolved_lane);
	state->device_config.mesh_lane_client = state->lane_client;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingLoadConfiguration(SparkStageServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStageRunnerConfiguration *config = &state->runner_config;
	SparkJsonDocument doc;
	SparkStatus status;
	memset(&doc, 0, sizeof(doc));
	memset(config, 0, sizeof(*config));
	config->abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	config->descriptor_bytes = (uint32_t)sizeof(*config);
	status = SparkJsonLoadFile(configuration->adapter_configuration_path, &doc);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = StageServingParse(state, configuration, &doc, SparkJsonGetRootToken(&doc));
	SparkJsonDocumentDestroy(&doc);
	if ( status != SPARK_STATUS_OK )
		return status;
	config->rank_pack_path = state->pack_path;
	config->linear_weight_codec = state->model->descriptor->linear_weight_codec;
	config->stray_working_set_path = state->stray_working_set_path;
	config->execution_stream = configuration->execution_stream;
	config->multiprocessors = SPARK_STAGE_SERVING_MULTIPROCESSORS;
	if ( config->tp_degree > 1u && state->device_collective_present == 0 )
	{
		fprintf(stderr, "%s adapter: tp_degree %u needs a device_collective\n", state->model->module_tag, config->tp_degree);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( state->device_collective_present == 0 )
		return SPARK_STATUS_OK;
	state->device_config.tp_degree = config->tp_degree;
	state->device_config.tp_rank = config->tp_rank;
	state->device_config.registration_cuda_stream = configuration->execution_stream;
	if ( SparkTpDeviceCollectiveApplyTopology(&state->device_topology, &state->device_config) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	status = StageServingAcquireLane(state);
	if ( status != SPARK_STATUS_OK )
		return status;
	config->device_collective = &state->device_config;
	return SPARK_STATUS_OK;
}

static void StageServingDestroy(void *adapter_state);

static SparkStatus StageServingRecurrentCopy(void *context, uint32_t direction,
	uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	SparkStageServingState *state = (SparkStageServingState *)context;
	if ( state == 0 || (direction != SPARK_STAGE_KV_RECURRENT_TO_BUFFER &&
		direction != SPARK_STAGE_KV_RECURRENT_FROM_BUFFER) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return SparkStageRunnerRecurrentCopy(&state->runner,
		direction == SPARK_STAGE_KV_RECURRENT_TO_BUFFER ? 1u : 0u,
		slot, buffer, bytes, stream);
}

static SparkStatus StageServingDriverAdmit(void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkStageServingState *state = (SparkStageServingState *)driver_instance;
	return SparkStageKvBindingAdmit(&state->kv, request, decision);
}

static const SparkModelDriverInterface StageServingDriver =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = (uint32_t)sizeof(SparkModelDriverInterface),
	.admit = StageServingDriverAdmit,
};

static SparkStatus StageServingBindKv(SparkStageServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	const SparkStageServingModel *model = state->model;
	const SparkModelServingAdapterDescriptor *descriptor = model->descriptor;
	SparkStageKvConfiguration kv;
	SparkStageRunnerKv attach;
	SparkStatus status;
	memset(&kv, 0, sizeof(kv));
	state->ledger.module_tag = model->module_tag;
	kv.module_tag = model->module_tag;
	kv.block_token_count = descriptor->cache_block_token_count;
	kv.region_count = 1u;
	model->kv_layout(state->runner_config.tp_rank, state->runner_config.tp_degree, &kv);
	if ( kv.region_count != SPARK_STAGE_KV_MAX_REGIONS )
		kv.region_count = 1u;
	kv.regions[0].layer_count = SparkStageRunnerKvLayerCount(&state->runner);
	kv.capacity_request.layer_count = kv.regions[0].layer_count;
	kv.model_id = descriptor->model_id;
	kv.model_revision = descriptor->model_revision;
	kv.resident_sequence_capacity = state->runner_config.resident_sequence_capacity;
	kv.max_sequence_positions = state->runner_config.kv_pages_per_sequence * descriptor->cache_block_token_count;
	kv.max_input_row_count = state->max_rows;
	kv.logical_page_count = configuration->runtime_limits.kv_logical_page_capacity;
	kv.physical_page_count = configuration->runtime_limits.kv_physical_page_capacity;
	kv.pipeline_slot_count = 1u;
	kv.backing_directory = configuration->kv_backing_directory;
	kv.backing_maximum_bytes = configuration->kv_backing_maximum_bytes;
	kv.snapshot_directory = configuration->kv_snapshot_directory;
	kv.snapshot_maximum_bytes = configuration->kv_snapshot_maximum_bytes;
	status = SparkStageRunnerPackIdentity(&state->runner, kv.pack_sha256, sizeof(kv.pack_sha256));
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s kv binding refused: the runner has no weightd pack, so the KV layout has no pack identity\n",
			model->module_tag);
		SPARK_RETURN(status);
	}
	if ( SparkSha256HexToDigest(descriptor->artifact_sha256, kv.contract_sha256) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	kv.expert_codec = descriptor->expert_weight_codec;
	kv.kv_codec = descriptor->kv_cache_codec;
	kv.driver_symbol = (const void *)&StageServingBindKv;
	kv.recurrent.lane_bytes = SparkStageRunnerRecurrentBytes(&state->runner);
	if ( kv.recurrent.lane_bytes != 0u )
	{
		kv.recurrent.copy = StageServingRecurrentCopy;
		kv.recurrent.context = state;
	}
	status = SparkStageKvBindingInitialize(&state->kv, &kv);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&attach, 0, sizeof(attach));
	attach.pool = state->kv.region_base[0];
	attach.layer_stride_bytes = state->kv.region_layer_stride_bytes[0];
	attach.layer_page_bytes = kv.regions[0].layer_page_bytes;
	attach.layer_count = kv.regions[0].layer_count;
	attach.page_table = state->kv.page_table;
	attach.page_table_stride = state->kv.pages_per_sequence;
	attach.pool_page_count = SparkStageKvBindingAddressablePageCount(&state->kv);
	attach.sequence_count = state->kv.resident_sequence_capacity;
	attach.context_shard = state->kv.context_shard;
	if ( kv.region_count > 1u )
	{
		attach.second_pool = state->kv.region_base[1];
		attach.second_layer_stride_bytes = state->kv.region_layer_stride_bytes[1];
		attach.second_layer_page_bytes = kv.regions[1].layer_page_bytes;
		attach.second_layer_count = kv.regions[1].layer_count;
	}
	return SparkStageRunnerAttachKv(&state->runner, &attach);
}

static SparkServingCacheAdmission StageServingCacheContext(SparkStageServingState *state,
	SparkModelDriverCacheLane *lanes)
{
	SparkServingCacheAdmission cache;
	memset(&cache, 0, sizeof(cache));
	cache.program_id = state->model->program_id;
	cache.lane_capacity = SPARK_STAGE_SERVING_MAX_LANES;
	cache.lanes = lanes;
	cache.driver = &StageServingDriver;
	cache.driver_instance = state;
	cache.adapter_state = state;
	return cache;
}

typedef struct SparkStageServingContinuity
{
	SparkStageServingState *state;
	uint32_t rows;
	uint32_t active;
	const uint32_t *row_slots;
	const uint64_t *row_sequence_ids;
	const uint64_t *row_positions;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkStageServingContinuity;

static SparkStatus StageServingPrepareContinuity(void *context)
{
	SparkStageServingContinuity *continuity = (SparkStageServingContinuity *)context;
	return SparkStageKvBindingContinuity(&continuity->state->kv, continuity->state->lane_states,
		continuity->rows, continuity->active, continuity->row_slots, continuity->row_sequence_ids,
		continuity->row_positions, continuity->bound, continuity->sequence_ids, continuity->next_positions);
}

static void StageServingPublishCompletion(void *context, const SparkModelDriverCompletion *completion)
{
	SparkStageServingState *state = (SparkStageServingState *)context;
	state->publish_completion = *completion;
	state->publish_completed = 1u;
}

static void StageServingBuildFrame(const SparkStageServingState *state, const SparkModelServingSubmission *submission,
	SparkModelDriverCacheLane *lanes, uint32_t lane_count, void *stream, SparkModelDriverFrame *frame)
{
	memset(frame, 0, sizeof(*frame));
	frame->program_id = state->model->program_id;
	frame->request_id = submission->request_id;
	frame->sequence_id = submission->sequence_id;
	frame->sequence_position = submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = submission->active_sequence_count;
	frame->new_token_count = submission->row_count;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL :
		submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH ? SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH :
		submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE ? SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE : 0u;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->execution_stream = stream;
	frame->residency = submission->residency;
	frame->cache_lane_count = lane_count;
	frame->cache_lanes = lanes;
}

static SparkStatus StageServingAdmitFrame(SparkStageServingState *state,
	const SparkModelServingSubmission *submission, SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	uint32_t lane_count = 0u;
	SparkStatus status;
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission, state->cache_lanes,
		SPARK_STAGE_SERVING_MAX_LANES, &lane_count);
	if ( status == SPARK_STATUS_OK && lane_count != submission->active_sequence_count )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkAdmissionRequestFromSubmission(state->model->program_id, submission,
			state->cache_lanes, 0u, &request);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	StageServingBuildFrame(state, submission, state->cache_lanes, lane_count,
		state->runner_config.execution_stream, frame);
	return SparkAdmissionEvaluateAndApply(&StageServingDriver, state, &request, frame, &decision);
}

static SparkStatus StageServingAllocate(SparkStageServingState *state)
{
	const uint64_t rows = state->max_rows, lanes = state->runner_config.max_active_sequence_count;
	struct { SparkMemoryBuffer *buffer; uint32_t space; uint64_t bytes; } plan[] =
	{
		{ &state->positions_host, SPARK_MEMORY_SPACE_HOST_COHERENT, rows * 4u },
		{ &state->context_host, SPARK_MEMORY_SPACE_HOST_COHERENT, lanes * 4u },
		{ &state->state_host, SPARK_MEMORY_SPACE_HOST_COHERENT, rows * 4u },
		{ &state->positions_device, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, rows * 4u },
		{ &state->context_device, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, lanes * 4u },
		{ &state->state_device, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, rows * 4u },
		{ &state->runs_host, SPARK_MEMORY_SPACE_HOST_COHERENT, (rows + 1u) * sizeof(uint32_t) },
		{ &state->runs_device, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (rows + 1u) * sizeof(uint32_t) },
		{ &state->seqslot_host, SPARK_MEMORY_SPACE_HOST_COHERENT, rows * sizeof(uint32_t) },
		{ &state->seqslot_device, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->order_host, SPARK_MEMORY_SPACE_HOST_COHERENT, rows * sizeof(uint32_t) },
		{ &state->order_device, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->output_tokens, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->output_scores, SPARK_MEMORY_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
	};
	SparkStatus status = SPARK_STATUS_OK;
	for ( uint32_t index = 0u; status == SPARK_STATUS_OK && index < sizeof(plan) / sizeof(plan[0]); ++index )
		status = SparkMemoryBufferAllocate(plan[index].buffer, plan[index].space, plan[index].bytes);
	return status;
}

static SparkStatus StageServingInitialize(const SparkModelServingAdapterConfiguration *configuration, void **adapter_state)
{
	const SparkStageServingModel *model = StageServingBoundModel;
	SparkStageServingState *state;
	SparkStatus status;
	if ( configuration == 0 || adapter_state == 0 || model == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkStageServingState *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->model = model;
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES);
	atomic_init(&state->reset_generation, 0u);
	status = StageServingLoadConfiguration(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ StageServingDestroy(state); return status; }
	state->max_rows = state->runner_config.max_input_row_count;
	if ( state->max_rows == 0u || state->max_rows > model->descriptor->max_input_row_count ||
		state->runner_config.max_active_sequence_count == 0u ||
		state->runner_config.max_active_sequence_count > model->descriptor->max_active_sequence_count )
		{ StageServingDestroy(state); SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED); }
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	status = StageServingAllocate(state);
	if ( status == SPARK_STATUS_OK )
		status = StageServingRunnerWeights(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageRunnerInitialize(&state->runner, &state->runner_config, model->runner_model());
	if ( status == SPARK_STATUS_OK )
		status = StageServingBindKv(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ StageServingDestroy(state); return status; }
	*adapter_state = state;
	return SPARK_STATUS_OK;
}

static void StageServingDestroy(void *adapter_state)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkMemoryBuffer *buffers[] =
	{
		&state->positions_host, &state->context_host, &state->state_host, &state->positions_device,
		&state->context_device, &state->state_device, &state->runs_host, &state->runs_device,
		&state->seqslot_host, &state->seqslot_device, &state->order_host, &state->order_device,
		&state->output_tokens, &state->output_scores,
	};
	if ( state == 0 )
		return;
	SparkStageKvBindingDestroy(&state->kv);
	SparkStageModuleLedgerRelease(&state->ledger);
	SparkStageRunnerDestroy(&state->runner);
	if ( state->lane_client != 0 )
	{
		(void)SparkWeightdClientClose(state->lane_client);
		state->lane_client = 0;
	}
	for ( uint32_t index = 0u; index < sizeof(buffers) / sizeof(buffers[0]); ++index )
		SparkMemoryBufferFree(buffers[index]);
	free(state->pack_path);
	free(state->stray_working_set_path);
	free(state->device_backend_path);
	free(state->device_local_host);
	free(state);
}

static SparkStatus StageServingReleaseSlots(const SparkStageServingState *state,
	const SparkModelServingSubmission *submission, uint32_t *slots)
{
	uint32_t count = submission->active_sequence_count;
	if ( submission->row_count != 0u ||
		count > SPARK_STAGE_SERVING_MAX_LANES ||
		count > state->runner_config.max_active_sequence_count ||
		(count != 0u && (submission->lanes == 0 || submission->lane_count < count)) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	for ( uint32_t lane = 0u; lane < count; ++lane )
	{
		uint32_t slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->runner_config.max_active_sequence_count )
			return SPARK_STATUS_INVALID_ARGUMENT;
		for ( uint32_t earlier = 0u; earlier < lane; ++earlier )
			if ( submission->lanes[earlier].resident_sequence_slot == slot )
				return SPARK_STATUS_INVALID_ARGUMENT;
		if ( slots != 0 )
			slots[lane] = slot;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingValidateSubmission(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	if ( state == 0 || submission == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( submission->control_generation < atomic_load_explicit(&state->reset_generation, memory_order_acquire) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( SparkModelServingWorkKindUsesRows(submission->work_kind) == 0u )
	{
		if ( submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE &&
			submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH )
			return SPARK_STATUS_UNSUPPORTED;
		return StageServingReleaseSlots(state, submission, 0);
	}
	if ( submission->row_count == 0u || submission->row_count > state->max_rows )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( submission->row_sequence_ids == 0 || submission->lanes == 0 || submission->lane_count < submission->active_sequence_count )
		return SPARK_STATUS_VALIDATION_FAILED;
	return SPARK_STATUS_OK;
}

static void StageServingCompletionHeader(const SparkModelServingSubmission *submission,
	uint32_t accepted_token_count, SparkModelServingCompletion *completion)
{
	memset(completion, 0, sizeof(*completion));
	completion->abi_version = submission->abi_version;
	completion->descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion->submission_id = submission->submission_id;
	completion->request_id = submission->request_id;
	completion->sequence_id = submission->sequence_id;
	completion->sequence_position = submission->sequence_position;
	completion->control_generation = submission->control_generation;
	completion->transaction_id = submission->transaction_id;
	completion->dispatch_generation = submission->dispatch_generation;
	completion->request_generation = submission->request_generation;
	completion->step_generation = submission->step_generation;
	completion->residency = submission->residency;
	completion->accepted_token_count = accepted_token_count;
}

static SparkStatus StageServingGroupRows(SparkStageServingState *state,
	const SparkModelServingSubmission *submission, uint32_t rows,
	uint32_t *active_out)
{
	uint32_t *positions = (uint32_t *)state->positions_host.pointer;
	uint32_t *context = (uint32_t *)state->context_host.pointer;
	uint32_t *slot_of_row = (uint32_t *)state->state_host.pointer;
	uint32_t *begin = (uint32_t *)state->runs_host.pointer;
	uint32_t *order = (uint32_t *)state->order_host.pointer;
	uint32_t *slots = (uint32_t *)state->seqslot_host.pointer;
	uint32_t cursor[SPARK_STAGE_SERVING_MAX_LANES];
	uint32_t capacity = state->runner_config.max_active_sequence_count;
	uint32_t lanes = submission->lanes != 0 ? submission->active_sequence_count : rows;
	uint32_t lane, i;
	if ( rows == 0u || rows > state->max_rows ||
		lanes == 0u || lanes > rows || lanes > capacity ||
		lanes > SPARK_STAGE_SERVING_MAX_LANES ||
		submission->row_positions == 0 )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	memset(context, 0, (uint64_t)capacity * sizeof(uint32_t));
	memset(begin, 0, ((uint64_t)lanes + 1u) * sizeof(uint32_t));
	for ( lane = 0u; lane < lanes; ++lane )
	{
		slots[lane] = submission->lanes != 0 ? submission->lanes[lane].resident_sequence_slot : lane;
		if ( slots[lane] >= capacity )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		for ( i = 0u; i < lane; ++i )
			if ( slots[i] == slots[lane] )
				SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	for ( i = 0u; i < rows; ++i )
	{
		lane = submission->lanes != 0 && submission->row_lane_indices != 0
			? submission->row_lane_indices[i] : i;
		if ( lane >= lanes || submission->row_positions[i] >= UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		positions[i] = (uint32_t)submission->row_positions[i];
		slot_of_row[i] = slots[lane];
		if ( positions[i] + 1u > context[slots[lane]] )
			context[slots[lane]] = positions[i] + 1u;
		begin[lane + 1u]++;
	}
	for ( lane = 0u; lane < lanes; ++lane )
	{
		if ( begin[lane + 1u] == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		begin[lane + 1u] += begin[lane];
		cursor[lane] = begin[lane];
	}
	for ( i = 0u; i < rows; ++i )
	{
		lane = submission->lanes != 0 && submission->row_lane_indices != 0
			? submission->row_lane_indices[i] : i;
		order[cursor[lane]++] = i;
	}
	*active_out = lanes;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingSubmitNonRow(SparkStageServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkModelServingCompletion completion;
	SparkModelDriverFrame frame;
	uint32_t slots[SPARK_STAGE_SERVING_MAX_LANES];
	SparkStatus status;
	status = StageServingReleaseSlots(state, submission, slots);
	if ( status == SPARK_STATUS_OK )
		status = StageServingAdmitFrame(state, submission, &frame);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH )
	{
		frame.completion_function = StageServingPublishCompletion;
		frame.completion_context = state;
		state->publish_completed = 0u;
		status = SparkStageKvBindingPublishFrame(&state->kv, &frame, state->lane_states);
		if ( status == SPARK_STATUS_OK && (state->publish_completed == 0u || state->publish_completion.status != SPARK_STATUS_OK) )
			status = state->publish_completed == 0u ? SPARK_STATUS_INTERNAL_ERROR : state->publish_completion.status;
	}
	else
		status = SparkStageRunnerResetSlots(&state->runner, slots, submission->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->completion_function != 0 )
	{
		StageServingCompletionHeader(submission, 0u, &completion);
		state->completion_function(state->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

static void StageServingContinuityRows(SparkStageServingState *state,
	const SparkModelServingSubmission *submission, uint32_t rows, uint32_t active,
	uint32_t *row_slots, uint64_t *row_sequence_ids, uint64_t *row_positions)
{
	const uint32_t *begin = (const uint32_t *)state->runs_host.pointer;
	const uint32_t *order = (const uint32_t *)state->order_host.pointer;
	const uint32_t *slots = (const uint32_t *)state->seqslot_host.pointer;
	uint32_t lane, cursor, out = active, row;
	for ( lane = 0u; lane < active; ++lane )
	{
		row = order[begin[lane]];
		row_slots[lane] = slots[lane];
		row_sequence_ids[lane] = submission->row_sequence_ids[row];
		row_positions[lane] = submission->row_positions[row];
	}
	for ( lane = 0u; lane < active; ++lane )
		for ( cursor = begin[lane] + 1u; cursor < begin[lane + 1u] && out < rows; ++cursor, ++out )
		{
			row = order[cursor];
			row_slots[out] = slots[lane];
			row_sequence_ids[out] = submission->row_sequence_ids[row];
			row_positions[out] = submission->row_positions[row];
		}
}

static SparkStatus StageServingFinish(SparkStageServingState *state, uint32_t active,
	const uint32_t *lane_slots, uint8_t *bound, uint64_t *sequence_ids,
	uint64_t *next_positions, uint32_t extra_tokens, SparkStatus status)
{
	SparkStageKvBindingCompletion completion;
	memset(&completion, 0, sizeof(completion));
	completion.lane_count = active;
	completion.extra_tokens = status == SPARK_STATUS_OK ? extra_tokens : 0u;
	completion.status = status;
	completion.resident_slots = lane_slots;
	completion.bound = bound;
	completion.sequence_ids = sequence_ids;
	completion.next_positions = next_positions;
	return SparkStageKvBindingFinishWait(&state->kv, 0u, &completion);
}

static SparkStatus StageServingChainSteps(const SparkStageServingState *state,
	const SparkModelServingSubmission *submission, const SparkModelDriverFrame *frame,
	uint32_t rows, uint32_t active, uint32_t *steps_out)
{
	const uint32_t *positions = (const uint32_t *)state->positions_host.pointer;
	const uint32_t block = state->model->descriptor->cache_block_token_count;
	uint32_t steps = submission->tokens_per_sequence > 1u ? submission->tokens_per_sequence : 1u, index;
	*steps_out = steps;
	if ( steps == 1u )
		return SPARK_STATUS_OK;
	if ( submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_DECODE || rows != active ||
		(uint64_t)rows * steps > state->max_rows || state->runner.owns_final_head == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	for ( index = 0u; index < frame->cache_lane_count; ++index )
		if ( (frame->cache_lanes[index].flags & (SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX | SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH)) != 0u )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	for ( index = 0u; index < rows; ++index )
		if ( (positions[index] % block) + steps > block )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	return SPARK_STATUS_OK;
}

static uint64_t StageServingNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void StageServingWaveHost(SparkStageServingState *state, const uint64_t *marks)
{
	uint32_t phase;
	for ( phase = 0u; phase < 5u; ++phase )
		state->wave_host_ns[phase] += marks[phase + 1u] - marks[phase];
	if ( ++state->wave_host_count < 16u )
		return;
	fprintf(stderr, "STAGE-WAVE-HOST tag=%s rank=%u prefill_waves=%u admit_ms=%.1f submit_ms=%.1f capture_ms=%.1f sync_ms=%.1f finish_ms=%.1f\n",
		state->model->module_tag, state->runner_config.tp_rank, state->wave_host_count,
		(double)state->wave_host_ns[0] / state->wave_host_count / 1e6, (double)state->wave_host_ns[1] / state->wave_host_count / 1e6,
		(double)state->wave_host_ns[2] / state->wave_host_count / 1e6, (double)state->wave_host_ns[3] / state->wave_host_count / 1e6,
		(double)state->wave_host_ns[4] / state->wave_host_count / 1e6);
	memset(state->wave_host_ns, 0, sizeof(state->wave_host_ns));
	state->wave_host_count = 0u;
}

static SparkStatus StageServingCopyInputs(SparkStageServingState *state, uint32_t rows, uint32_t active)
{
	void *stream = state->runner_config.execution_stream;
	if ( SparkMemoryBufferCopy(&state->positions_device, &state->positions_host, (uint64_t)rows * sizeof(uint32_t), stream) != SPARK_STATUS_OK ||
		SparkMemoryBufferCopy(&state->context_device, &state->context_host,
			(uint64_t)state->runner_config.max_active_sequence_count * sizeof(uint32_t), stream) != SPARK_STATUS_OK ||
		SparkMemoryBufferCopy(&state->state_device, &state->state_host, (uint64_t)rows * sizeof(uint32_t), stream) != SPARK_STATUS_OK ||
		SparkMemoryBufferCopy(&state->runs_device, &state->runs_host, ((uint64_t)active + 1u) * sizeof(uint32_t), stream) != SPARK_STATUS_OK ||
		SparkMemoryBufferCopy(&state->seqslot_device, &state->seqslot_host, (uint64_t)active * sizeof(uint32_t), stream) != SPARK_STATUS_OK ||
		SparkMemoryBufferCopy(&state->order_device, &state->order_host, (uint64_t)rows * sizeof(uint32_t), stream) != SPARK_STATUS_OK )
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static void StageServingDispatch(const SparkStageServingState *state, const SparkModelServingSubmission *submission,
	uint32_t rows, uint32_t active, uint32_t steps, SparkStageRunnerDispatch *dispatch)
{
	memset(dispatch, 0, sizeof(*dispatch));
	dispatch->abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	dispatch->descriptor_bytes = (uint32_t)sizeof(*dispatch);
	dispatch->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL : 0u;
	dispatch->request_id = submission->request_id;
	dispatch->sequence_id = submission->sequence_id;
	dispatch->sequence_position = submission->sequence_position;
	dispatch->deadline_time_ns = submission->deadline_time_ns;
	dispatch->row_count = rows;
	dispatch->active_sequence_count = active;
	dispatch->token_ids = submission->token_ids;
	dispatch->positions = state->positions_device.pointer;
	dispatch->context_length = state->context_device.pointer;
	dispatch->sequence_of_row = state->state_device.pointer;
	dispatch->sequence_row_begin = state->runs_device.pointer;
	dispatch->sequence_row_indices = state->order_device.pointer;
	dispatch->recurrent_index = state->seqslot_device.pointer;
	dispatch->gather_sequence = active == 1u ? ((const uint32_t *)state->state_host.pointer)[0] : 0u;
	dispatch->gather_context = active == 1u && dispatch->gather_sequence < state->runner_config.max_active_sequence_count
		? ((const uint32_t *)state->context_host.pointer)[dispatch->gather_sequence] : 0u;
	dispatch->hidden_input_bf16 = submission->hidden_input_address;
	dispatch->hidden_input_bytes = submission->hidden_input_bytes;
	dispatch->hidden_output_bf16 = submission->hidden_output_address;
	dispatch->hidden_output_bytes = submission->hidden_output_bytes;
	dispatch->sideband_input = submission->boundary_sideband_input_address;
	dispatch->sideband_input_bytes = submission->boundary_sideband_input_bytes;
	dispatch->sideband_output = submission->boundary_sideband_output_address;
	dispatch->sideband_output_bytes = submission->boundary_sideband_output_bytes;
	dispatch->output_token_ids = state->output_tokens.pointer;
	dispatch->output_scores = state->output_scores.pointer;
	dispatch->chain_steps = steps;
}

static SparkStatus StageServingComplete(SparkStageServingState *state, const SparkModelServingSubmission *submission,
	uint32_t rows, uint32_t active, uint32_t steps, uint32_t distribution_count, const uint32_t *distribution_lanes,
	const SparkRowSampling *distribution_rules, const SparkSamplingLogprob *distribution_logprobs, uint32_t logprob_stride)
{
	SparkModelServingCompletion completion;
	StageServingCompletionHeader(submission, rows, &completion);
	if ( state->runner.owns_final_head != 0u )
	{
		const uint32_t *runs = (const uint32_t *)state->runs_host.pointer;
		const uint32_t *order = (const uint32_t *)state->order_host.pointer;
		uint32_t *tokens_host = state->completion_tokens;
		SparkMemoryBuffer tokens = SPARK_MEMORY_BUFFER_VIEW(tokens_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)rows * steps * 4u);
		if ( (uint64_t)rows * steps > SPARK_STAGE_SERVING_MAX_ROWS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( SparkMemoryBufferCopy(&tokens, &state->output_tokens, (uint64_t)rows * steps * 4u, 0) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		completion.tokens_per_sequence = steps;
		completion.token_count = active * steps;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for ( uint32_t s = 0u; s < active; ++s )
			for ( uint32_t step = 0u; step < steps && s * steps + step < SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT; ++step )
				completion.token_ids[s * steps + step] = tokens_host[step * rows + order[runs[s + 1u] - 1u]];
		completion.logprob_stride = logprob_stride;
		completion.logprob_entry_count = completion.token_count * logprob_stride;
		for ( uint32_t entry = 0u; logprob_stride != 0u && entry < distribution_count; ++entry )
			memcpy(&completion.logprobs[(uint64_t)distribution_lanes[entry] * logprob_stride],
				&distribution_logprobs[(uint64_t)entry * SPARK_SAMPLING_MAX_LOGPROBS],
				(uint64_t)distribution_rules[entry].logprobs * sizeof(SparkSamplingLogprob));
	}
	state->completion_function(state->completion_context, &completion);
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingSubmit(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkStageRunnerDispatch dispatch;
	SparkModelDriverFrame frame;
	SparkStageServingContinuity continuity;
	uint32_t *row_slots = state != 0 ? state->row_slots : 0;
	uint8_t bound[SPARK_STAGE_SERVING_MAX_LANES];
	uint64_t sequence_ids[SPARK_STAGE_SERVING_MAX_LANES], next_positions[SPARK_STAGE_SERVING_MAX_LANES];
	uint32_t distribution_rows[SPARK_STAGE_SERVING_MAX_LANES], distribution_positions[SPARK_STAGE_SERVING_MAX_LANES];
	uint32_t distribution_lanes[SPARK_STAGE_SERVING_MAX_LANES];
	SparkRowSampling distribution_rules[SPARK_STAGE_SERVING_MAX_LANES];
	SparkSamplingLogprob distribution_logprobs[SPARK_STAGE_SERVING_MAX_LANES * SPARK_SAMPLING_MAX_LOGPROBS];
	uint32_t rows, active = 0u, distribution_count = 0u, logprob_stride = 0u, steps = 1u;
	uint64_t marks[6];
	SparkStatus status, finished;
	void *stream;
	marks[0] = StageServingNowNs();
	status = StageServingValidateSubmission(state, submission);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( SparkModelServingWorkKindUsesRows(submission->work_kind) == 0u )
		return StageServingSubmitNonRow(state, submission);
	stream = state->runner_config.execution_stream;
	rows = submission->row_count;
	status = StageServingGroupRows(state, submission, rows, &active);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = StageServingAdmitFrame(state, submission, &frame);
	if ( status == SPARK_STATUS_OK )
		status = StageServingChainSteps(state, submission, &frame, rows, active, &steps);
	if ( status != SPARK_STATUS_OK )
		return status;
	StageServingContinuityRows(state, submission, rows, active, row_slots, state->row_sequence_ids, state->row_positions);
	for ( uint32_t lane = 0u; state->runner.owns_final_head != 0u && submission->lanes != 0 && lane < active; ++lane )
	{
		const uint32_t *runs = (const uint32_t *)state->runs_host.pointer;
		const uint32_t *order = (const uint32_t *)state->order_host.pointer;
		if ( SparkSamplingRuleNeedsDistribution(&submission->lanes[lane].sampling) == 0u )
			continue;
		distribution_rows[distribution_count] = order[runs[lane + 1u] - 1u];
		distribution_positions[distribution_count] = ((const uint32_t *)state->positions_host.pointer)[distribution_rows[distribution_count]];
		distribution_rules[distribution_count] = submission->lanes[lane].sampling;
		distribution_lanes[distribution_count] = lane;
		if ( submission->lanes[lane].sampling.logprobs > logprob_stride )
			logprob_stride = submission->lanes[lane].sampling.logprobs;
		distribution_count++;
	}
	continuity.state = state;
	continuity.rows = rows;
	continuity.active = active;
	continuity.row_slots = row_slots;
	continuity.row_sequence_ids = state->row_sequence_ids;
	continuity.row_positions = state->row_positions;
	continuity.bound = bound;
	continuity.sequence_ids = sequence_ids;
	continuity.next_positions = next_positions;
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES,
		row_slots, active, StageServingPrepareContinuity, &continuity);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkStageKvBindingClaim(&state->kv, &frame, active, row_slots, state->row_sequence_ids, state->row_positions,
		next_positions);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, row_slots, active);
		return status;
	}
	status = SparkStageKvBindingFenceExecution(&state->kv, stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRecurrentRestore(&state->kv, row_slots, active, stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingUploadPageTables(&state->kv, row_slots, active, stream);
	if ( status == SPARK_STATUS_OK )
		status = StageServingCopyInputs(state, rows, active);
	if ( status == SPARK_STATUS_OK )
	{
		StageServingDispatch(state, submission, rows, active, steps, &dispatch);
		dispatch.distribution_count = distribution_count;
		dispatch.distribution_rows = distribution_rows;
		dispatch.distribution_positions = distribution_positions;
		dispatch.distribution_rules = distribution_rules;
		dispatch.distribution_logprobs = distribution_logprobs;
		marks[1] = StageServingNowNs();
		status = SparkStageRunnerSubmit(&state->runner, &dispatch);
	}
	marks[2] = StageServingNowNs();
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRecurrentCapture(&state->kv, row_slots, active, stream);
	marks[3] = StageServingNowNs();
	if ( status == SPARK_STATUS_OK && cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	marks[4] = StageServingNowNs();
	for ( uint32_t lane = 0u; status == SPARK_STATUS_OK && lane < active; ++lane )
		next_positions[lane] += steps - 1u;
	finished = StageServingFinish(state, active, row_slots, bound, sequence_ids, next_positions, steps - 1u, status);
	marks[5] = StageServingNowNs();
	if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		StageServingWaveHost(state, marks);
	SparkStageModuleIndexSetRelease(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, row_slots, active);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( finished != SPARK_STATUS_OK )
		return finished;
	if ( state->completion_function == 0 )
		return SPARK_STATUS_OK;
	return StageServingComplete(state, submission, rows, active, steps, distribution_count, distribution_lanes,
		distribution_rules, distribution_logprobs, logprob_stride);
}

static SparkStatus StageServingPrefetch(void *adapter_state,
	const SparkModelServingSubmission *submissions, uint32_t submission_count)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	cache = StageServingCacheContext(state, state->prefetch_lanes);
	cache.validate = StageServingValidateSubmission;
	return SparkServingCacheAdmissionRun(&cache, submissions, submission_count,
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
}

static SparkStatus StageServingCacheHint(void *adapter_state,
	const SparkModelServingCacheIdentity *identity, uint32_t token_count)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	cache = StageServingCacheContext(state, state->prefetch_lanes);
	return SparkServingCacheHintRun(&cache, identity, token_count);
}

static SparkStatus StageServingResolvePrefetch(void *adapter_state,
	const SparkModelServingSubmission *submission, uint32_t resolution)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 || (resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	cache = StageServingCacheContext(state, state->prefetch_lanes);
	cache.validate = StageServingValidateSubmission;
	return SparkServingCacheAdmissionRun(&cache, submission, 1u,
		resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ?
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT : SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT);
}

static SparkStatus StageServingProgress(void *adapter_state, uint32_t maximum_step_count)
{
	(void)adapter_state;
	(void)maximum_step_count;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingQuiesce(void *adapter_state, uint64_t deadline_time_ns)
{
	(void)adapter_state;
	(void)deadline_time_ns;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingSnapshot(void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkStageRunnerStats stats;
	uint32_t block;
	if ( state == 0 || snapshot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	block = state->model->descriptor->cache_block_token_count;
	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	snapshot->available_submission_count = state->max_rows;
	SparkStageRunnerGetStats(&state->runner, &stats);
	snapshot->submitted_count = stats.submitted_count;
	snapshot->completed_count = stats.completed_count;
	snapshot->resident_sequence_count = SparkStageKvBindingResidentCount(&state->kv);
	snapshot->kv_token_capacity = (uint64_t)state->kv.logical_page_count * block;
	snapshot->max_sequence_positions = state->runner_config.kv_pages_per_sequence * block;
	SparkStageKvBindingKvStoreCounters(&state->kv, &snapshot->kv_store);
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingReset(void *adapter_state, uint64_t control_generation)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	uint32_t slots[SPARK_STAGE_SERVING_MAX_LANES], slot, count;
	SparkStatus status;
	if ( state == 0 || state->kv.resident_sequence_capacity > SPARK_STAGE_SERVING_MAX_LANES )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkStageKvBindingResetIsNew(&state->kv, control_generation) == 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	count = state->kv.resident_sequence_capacity;
	for ( slot = 0u; slot < count; ++slot )
		slots[slot] = slot;
	status = SparkStageModuleIndexSetClaim(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, slots, count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( cudaStreamSynchronize((cudaStream_t)state->runner_config.execution_stream) != cudaSuccess ||
		(state->kv.copy_stream != 0 && cudaStreamSynchronize((cudaStream_t)state->kv.copy_stream) != cudaSuccess) )
	{
		fprintf(stderr, "%s reset: execution stream failed; lanes stay claimed\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageRunnerResetSlots(&state->runner, slots, count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingReset(&state->kv, control_generation);
	if ( status == SPARK_STATUS_OK )
		atomic_store_explicit(&state->reset_generation, control_generation, memory_order_release);
	SparkStageModuleIndexSetRelease(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, slots, count);
	SPARK_RETURN(status);
}

static SparkModelServingAdapterInterface StageServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = 0,
	.initialize = StageServingInitialize,
	.destroy = StageServingDestroy,
	.validate_submission = StageServingValidateSubmission,
	.submit = StageServingSubmit,
	.prefetch = StageServingPrefetch,
	.cache_hint = StageServingCacheHint,
	.resolve_prefetch = StageServingResolvePrefetch,
	.progress = StageServingProgress,
	.quiesce = StageServingQuiesce,
	.snapshot = StageServingSnapshot,
	.reset = StageServingReset,
};

static uint32_t StageServingModelValid(const SparkStageServingModel *model)
{
	const SparkModelServingAdapterDescriptor *descriptor = model != 0 ? model->descriptor : 0;
	return descriptor != 0 && model->runner_model != 0 && model->module_tag != 0 && model->kv_layout != 0 &&
		model->collective_sequences != 0 && model->tp_degree != 0u &&
		descriptor->cache_block_token_count != 0u && descriptor->artifact_sha256 != 0 &&
		descriptor->max_input_row_count != 0u && descriptor->max_input_row_count <= SPARK_STAGE_SERVING_MAX_ROWS &&
		descriptor->max_active_sequence_count != 0u &&
		descriptor->max_active_sequence_count <= SPARK_STAGE_SERVING_MAX_LANES;
}

const SparkModelServingAdapterInterface *SparkStageServingAdapterInterface(const SparkStageServingModel *model)
{
	if ( StageServingModelValid(model) == 0u || (StageServingBoundModel != 0 && StageServingBoundModel != model) )
		return 0;
	StageServingBoundModel = model;
	StageServingInterface.descriptor = model->descriptor;
	return &StageServingInterface;
}
