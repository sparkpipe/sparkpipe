#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "sparkpipe/spark_k3_serving_adapter.h"

typedef struct SparkK3ServingState
{
	SparkK3StageRunner runner;
	SparkK3StageRunnerConfiguration runner_config;
	SparkTpCollectiveConfig collective_config;
	SparkTpCollectivePeer peers[SPARK_TP_COLLECTIVE_MAX_STEPS];
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	char *pack_path;
	uint32_t max_rows;
	/* the device-direct tier's config + topology (zeroed when the host
	 * TCP tier is in use) */
	SparkTpDeviceCollectiveConfig device_config;
	SparkTpDeviceCollectiveTopology device_topology;
	char device_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
		[SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
	int device_collective_present;
	/* per-submit host-converted arrays */
	uint32_t *positions_host;
	uint32_t *context_host;
	uint32_t *state_host;
	/* per-submit device arrays */
	uint32_t *positions_device;
	uint32_t *context_device;
	uint32_t *state_device;
	uint32_t *output_tokens;
	float *output_scores;
} SparkK3ServingState;

static uint32_t K3ServingJsonU32(SparkJsonDocument *doc, int32_t root,
	const char *name, uint32_t fallback)
{
	uint32_t value = 0u;
	int32_t token = SparkJsonFindObjectMember(doc, root, name);
	if ( token >= 0 && SparkJsonGetUInt32(doc, token, &value) == SPARK_STATUS_OK )
		return value;
	return fallback;
}

static SparkStatus K3ServingLoadConfiguration(SparkK3ServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkJsonDocument doc;
	SparkStatus status;
	int32_t root, token;
	memset(&doc, 0, sizeof(doc));
	status = SparkJsonLoadFile(configuration->adapter_configuration_path, &doc);
	if ( status != SPARK_STATUS_OK )
		return status;
	root = SparkJsonGetRootToken(&doc);
	token = SparkJsonFindObjectMember(&doc, root, "stage_pack_path");
	if ( token < 0 || SparkJsonCopyString(&doc, token, &state->pack_path) != SPARK_STATUS_OK )
		{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
	memset(&state->runner_config, 0, sizeof(state->runner_config));
	state->runner_config.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	state->runner_config.descriptor_bytes = (uint32_t)sizeof(state->runner_config);
	/* The residentd's stage_index IS the world rank (the deployment has one
	 * node per rank, like the DSV4 hybrid); the runner wants the PP stage
	 * and the TP placement, derived here: world_rank = pp*4 + tp. */
	state->runner_config.tp_degree = K3ServingJsonU32(&doc, root, "tp_degree", 1u);
	state->runner_config.stage_index = configuration->stage_index / state->runner_config.tp_degree;
	state->runner_config.stage_count = 4u;
	state->runner_config.tp_rank = configuration->stage_index % state->runner_config.tp_degree;
	state->runner_config.max_active_sequence_count =
		K3ServingJsonU32(&doc, root, "max_sequences",
		configuration->runtime_limits.max_active_sequence_count);
	state->runner_config.max_input_row_count =
		K3ServingJsonU32(&doc, root, "max_rows",
		configuration->runtime_limits.max_input_row_count);
	state->runner_config.resident_sequence_capacity =
		K3ServingJsonU32(&doc, root, "resident_capacity",
		configuration->runtime_limits.resident_sequence_capacity);
	state->runner_config.kv_pages_per_sequence =
		K3ServingJsonU32(&doc, root, "kv_pages", 2u);
	/* Zero lets the runner supply K3GlobalKv::kPageBytes (the CUDA-side
	 * geometry the adapter cannot see without the device headers). */
	state->runner_config.kv_page_bytes = 0u;
	/* The device-direct tier: an optional "device_collective" object. */
	{
		int32_t dev = SparkJsonFindObjectMember(&doc, root, "device_collective");
		if ( dev >= 0 )
		{
			uint32_t hidden = K3ServingJsonU32(&doc, root, "hidden", 7168u);
			memset(&state->device_config, 0, sizeof(state->device_config));
			memset(&state->device_topology, 0, sizeof(state->device_topology));
			state->device_topology.abi_version =
				SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
			state->device_topology.descriptor_bytes =
				SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
			state->device_config.abi_version =
				SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			int32_t backend_token = SparkJsonFindObjectMember(&doc, dev, "backend");
			if ( backend_token < 0 )
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			if ( SparkJsonStringEquals(&doc, backend_token, "nccl") )
				state->device_config.backend_kind =
					SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
			else if ( SparkJsonStringEquals(&doc, backend_token, "hidden_transport") )
				state->device_config.backend_kind =
					SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
			else
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			int32_t module_token = SparkJsonFindObjectMember(&doc, dev, "backend_module_path");
			if ( module_token >= 0 )
				SparkJsonCopyString(&doc, module_token,
					(char **)&state->device_config.backend_module_path);
			int32_t host_token = SparkJsonFindObjectMember(&doc, dev, "local_host");
			if ( host_token >= 0 )
				SparkJsonCopyString(&doc, host_token,
					(char **)&state->device_config.local_host);
			uint64_t dev_id = 0u;
			int32_t id_token = SparkJsonFindObjectMember(&doc, dev, "collective_identifier");
			if ( id_token >= 0 )
				SparkJsonGetUInt64(&doc, id_token, &dev_id);
			state->device_config.collective_identifier = dev_id;
			state->device_config.control_port_base =
				K3ServingJsonU32(&doc, dev, "listen_port", 0u);
			state->device_config.connect_timeout_milli =
				K3ServingJsonU32(&doc, dev, "connect_timeout_milli", 5000u);
			state->device_config.operation_timeout_milli =
				K3ServingJsonU32(&doc, dev, "operation_timeout_milli", 30000u);
			state->device_config.operation_kind =
				SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
			state->device_config.credit_count = 4u;
			state->device_config.local_hidden_dimension = 3u * hidden;
			state->device_config.max_active_sequence_count =
				state->runner_config.max_input_row_count;
			int32_t hosts_token = SparkJsonFindObjectMember(&doc, dev, "peer_hosts");
			uint32_t peer_count = hosts_token >= 0 ?
				SparkJsonGetArrayElementCount(&doc, hosts_token) : 0u;
			if ( peer_count == 0u ||
				peer_count > SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE )
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			state->device_topology.rank_count = peer_count;
			for ( uint32_t i = 0u; i < peer_count; ++i )
			{
				int32_t peer = SparkJsonGetArrayElement(&doc, hosts_token, i);
				char *text = 0;
				if ( peer < 0 ||
					SparkJsonCopyString(&doc, peer, &text) != SPARK_STATUS_OK )
					{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
				strncpy(state->device_hosts[i], text,
					SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES - 1u);
				free(text);
				state->device_topology.rank_hosts[i] = state->device_hosts[i];
			}
			/* The runner completes the config (degree/rank/combine functions)
			 * and applies the topology. */
			state->device_collective_present = 1;
		}
	}
	state->runner_config.rank_pack_path = state->pack_path;
	state->runner_config.execution_stream = configuration->execution_stream;
	state->runner_config.multiprocessors = 48u;
	if ( state->runner_config.tp_degree > 1u )
	{
		int32_t coll = SparkJsonFindObjectMember(&doc, root, "tp_collective");
		if ( coll < 0 )
			{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
		memset(&state->collective_config, 0, sizeof(state->collective_config));
		state->collective_config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
		state->collective_config.tp_degree = state->runner_config.tp_degree;
		state->collective_config.tp_rank = state->runner_config.tp_rank;
		state->collective_config.listen_port = (uint16_t)K3ServingJsonU32(&doc, coll, "listen_port", 0u);
		state->collective_config.connect_timeout_milli = K3ServingJsonU32(&doc, coll, "connect_timeout_milli", 5000u);
		state->collective_config.operation_timeout_milli = K3ServingJsonU32(&doc, coll, "operation_timeout_milli", 30000u);
		state->collective_config.collective_identifier = 0u;
		{
			uint64_t id64 = 0u;
			int32_t id_token = SparkJsonFindObjectMember(&doc, coll, "collective_identifier");
			if ( id_token >= 0 )
				SparkJsonGetUInt64(&doc, id_token, &id64);
			state->collective_config.collective_identifier = id64;
		}
		int32_t peers_token = SparkJsonFindObjectMember(&doc, coll, "peers");
		uint32_t peer_count = peers_token >= 0 ?
			SparkJsonGetArrayElementCount(&doc, peers_token) : 0u;
		if ( peer_count == 0u || peer_count > SPARK_TP_COLLECTIVE_MAX_STEPS )
			{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
		for ( uint32_t i = 0u; i < peer_count; ++i )
		{
			int32_t peer = SparkJsonGetArrayElement(&doc, peers_token, i);
			char *text = 0;
			if ( peer < 0 || SparkJsonCopyString(&doc, peer, &text) != SPARK_STATUS_OK )
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			char *colon = strrchr(text, ':');
			if ( colon == 0 )
				{ free(text); SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			*colon = '\0';
			strncpy(state->peers[i].host_name, text, SPARK_TP_COLLECTIVE_HOST_NAME_BYTES - 1u);
			state->peers[i].port = (uint16_t)atoi(colon + 1);
			free(text);
		}
		memcpy(state->collective_config.peers, state->peers, sizeof(state->peers));
		state->runner_config.tp_collective = &state->collective_config;
	}
	if ( state->device_collective_present != 0 )
	{
		state->device_config.tp_degree = state->runner_config.tp_degree;
		state->device_config.tp_rank = state->runner_config.tp_rank;
		state->device_config.registration_cuda_stream =
			configuration->execution_stream;
		if ( SparkTpDeviceCollectiveApplyTopology(&state->device_topology,
			&state->device_config) != SPARK_STATUS_OK )
			{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
		state->runner_config.device_collective = &state->device_config;
	}
	SparkJsonDocumentDestroy(&doc);
	return SPARK_STATUS_OK;
}
static SparkStatus K3ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkK3ServingState *state;
	SparkStatus status;
	if ( configuration == 0 || adapter_state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkK3ServingState *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = K3ServingLoadConfiguration(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ free(state); return status; }
	state->max_rows = state->runner_config.max_input_row_count;
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	state->positions_host = (uint32_t *)malloc((uint64_t)state->max_rows * 4u);
	state->context_host = (uint32_t *)malloc((uint64_t)state->max_rows * 4u);
	state->state_host = (uint32_t *)malloc((uint64_t)state->max_rows * 4u);
	cudaMalloc(&state->positions_device, (uint64_t)state->max_rows * 4u);
	cudaMalloc(&state->context_device, (uint64_t)state->max_rows * 4u);
	cudaMalloc(&state->state_device, (uint64_t)state->max_rows * 4u);
	cudaMalloc(&state->output_tokens, (uint64_t)state->max_rows * 4u);
	cudaMalloc(&state->output_scores, (uint64_t)state->max_rows * 4u);
	status = SparkK3StageRunnerInitialize(&state->runner, &state->runner_config);
	if ( status != SPARK_STATUS_OK )
		{ free(state->positions_host); free(state->context_host); free(state->state_host); free(state); return status; }
	*adapter_state = state;
	return SPARK_STATUS_OK;
}

static void K3ServingDestroy(void *adapter_state)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	if ( state == 0 )
		return;
	SparkK3StageRunnerDestroy(&state->runner);
	free(state->positions_host);
	free(state->context_host);
	free(state->state_host);
	free(state->pack_path);
	cudaFree(state->positions_device);
	cudaFree(state->context_device);
	cudaFree(state->state_device);
	cudaFree(state->output_tokens);
	cudaFree(state->output_scores);
	free(state);
}

static SparkStatus K3ServingValidateSubmission(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	if ( state == 0 || submission == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( submission->row_count == 0u || submission->row_count > state->max_rows )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingSubmit(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	SparkK3StageRunnerDispatch dispatch;
	uint64_t *positions_host64;
	uint32_t rows;
	SparkStatus status;
	if ( state == 0 || submission == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	rows = submission->row_count;
	/* The serving ABI carries uint64 positions; the K3 kernels consume
	 * uint32. Host-convert per submission (the runner's collective tier
	 * already syncs, so this copy is not a new stall class). */
	positions_host64 = (uint64_t *)malloc((uint64_t)rows * 8u);
	if ( positions_host64 == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	cudaMemcpy(positions_host64, submission->row_positions,
		(uint64_t)rows * 8u, cudaMemcpyDeviceToHost);
	for ( uint32_t i = 0u; i < rows; ++i )
	{
		state->positions_host[i] = (uint32_t)positions_host64[i];
		state->context_host[i] = (uint32_t)positions_host64[i] + 1u;
		state->state_host[i] = submission->lanes != 0
			? submission->lanes[i].resident_sequence_slot : i;
	}
	free(positions_host64);
	cudaMemcpy(state->positions_device, state->positions_host,
		(uint64_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(state->context_device, state->context_host,
		(uint64_t)rows * 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(state->state_device, state->state_host,
		(uint64_t)rows * 4u, cudaMemcpyHostToDevice);
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = (uint32_t)sizeof(dispatch);
	dispatch.request_id = submission->request_id;
	dispatch.sequence_id = submission->sequence_id;
	dispatch.sequence_position = submission->sequence_position;
	dispatch.deadline_time_ns = submission->deadline_time_ns;
	dispatch.row_count = rows;
	dispatch.active_sequence_count = submission->active_sequence_count != 0u
		? submission->active_sequence_count : rows;
	dispatch.token_ids = submission->token_ids;
	dispatch.positions = state->positions_device;
	dispatch.context_length = state->context_device;
	dispatch.sequence_of_row = state->state_device;
	dispatch.kda_state_index = state->state_device;
	dispatch.hidden_input_bf16 = submission->hidden_input_address;
	dispatch.hidden_input_bytes = submission->hidden_input_bytes;
	dispatch.hidden_output_bf16 = submission->hidden_output_address;
	dispatch.hidden_output_bytes = submission->hidden_output_bytes;
	dispatch.output_token_ids = state->output_tokens;
	dispatch.output_scores = state->output_scores;
	dispatch.completion_function = 0;
	dispatch.completion_context = 0;
	status = SparkK3StageRunnerSubmit(&state->runner, &dispatch);
	if ( status != SPARK_STATUS_OK )
		return status;
	/* The runner completed the step synchronously; publish the completion
	 * through the serving ABI. */
	if ( state->completion_function != 0 )
	{
		SparkModelServingCompletion completion;
		uint32_t *tokens_host = (uint32_t *)malloc((uint64_t)rows * 4u);
		memset(&completion, 0, sizeof(completion));
		completion.abi_version = submission->abi_version;
		completion.descriptor_bytes = (uint32_t)sizeof(completion);
		completion.submission_id = submission->submission_id;
		completion.request_id = submission->request_id;
		completion.sequence_id = submission->sequence_id;
		completion.sequence_position = submission->sequence_position;
		completion.residency = submission->residency;
		completion.accepted_token_count = rows;
		completion.token_count = rows;
		completion.tokens_per_sequence = 1u;
		cudaMemcpy(tokens_host, state->output_tokens, (uint64_t)rows * 4u,
			cudaMemcpyDeviceToHost);
		for ( uint32_t i = 0u; i < rows && i < SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT; ++i )
			completion.token_ids[i] = tokens_host[i];
		free(tokens_host);
		state->completion_function(state->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingPrefetch(void *adapter_state,
	const SparkModelServingSubmission *submissions, uint32_t submission_count)
{
	(void)adapter_state;
	(void)submissions;
	(void)submission_count;
	/* No content-addressed prefix cache in this driver yet. */
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingResolvePrefetch(void *adapter_state,
	const SparkModelServingSubmission *submission, uint32_t resolution)
{
	(void)adapter_state;
	(void)submission;
	(void)resolution;
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingProgress(void *adapter_state, uint32_t maximum_step_count)
{
	(void)adapter_state;
	(void)maximum_step_count;
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingQuiesce(void *adapter_state, uint64_t deadline_time_ns)
{
	(void)adapter_state;
	(void)deadline_time_ns;
	/* The runner is synchronous: nothing can be in flight once submit
	 * returned. */
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingSnapshot(void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	SparkK3StageRunnerStats stats;
	if ( state == 0 || snapshot == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = (uint32_t)sizeof(*snapshot);
	snapshot->available_submission_count = state->max_rows;
	SparkK3StageRunnerGetStats(&state->runner, &stats);
	snapshot->submitted_count = stats.submitted_count;
	snapshot->completed_count = stats.completed_count;
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingReset(void *adapter_state, uint64_t control_generation)
{
	(void)adapter_state;
	(void)control_generation;
	return SPARK_STATUS_OK;
}

static const SparkModelServingAdapterDescriptor K3ServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = (uint32_t)sizeof(SparkModelServingAdapterDescriptor),
	.capability_flags = 0u,
	/* One node per RANK (the hybrid's contract): the residentd checks the
	 * node count against this. */
	.stage_count = 16u,
	.layer_count = 93u,
	.boundary_format = 0u,
	.boundary_element_count = 7168u,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = 0u,
	.expert_weight_codec = 1u,
	.kv_cache_codec = 0u,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = 16u,
	.max_input_row_count = 16u,
	.max_resident_sequence_count = 16u,
	.max_output_token_count = 16u,
	.max_speculative_token_count = 0u,
	.resident_sequence_slot_reuse = 0u,
	.adapter_id = "k3-tp4pp4",
	.model_id = "moonshotai/Kimi-K3-MXFP4",
	.model_revision = "k3-tp4pp4",
	.driver_program_name = "k3",
	.artifact_sha256 = "",
	.stage_layer_counts = { 24u, 23u, 23u, 23u, 24u, 23u, 23u, 23u, 24u, 23u, 23u, 23u, 24u, 23u, 23u, 23u },
	.boundary_sideband_kinds = { 0u, 0u, 0u, 0u },
	.boundary_sideband_bytes_per_sequence = { 0u, 0u, 0u, 0u },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = 0u,
	.parallel_group_size = 4u,
};

static const SparkModelServingAdapterInterface K3ServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = (uint32_t)sizeof(K3ServingInterface),
	.descriptor = &K3ServingDescriptor,
	.initialize = K3ServingInitialize,
	.destroy = K3ServingDestroy,
	.validate_submission = K3ServingValidateSubmission,
	.submit = K3ServingSubmit,
	.prefetch = K3ServingPrefetch,
	.resolve_prefetch = K3ServingResolvePrefetch,
	.progress = K3ServingProgress,
	.quiesce = K3ServingQuiesce,
	.snapshot = K3ServingSnapshot,
	.reset = K3ServingReset,
};

const SparkModelServingAdapterInterface *SparkK3ServingAdapterGetInterface(void)
{
	return &K3ServingInterface;
}
