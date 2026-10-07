#include <cuda_runtime.h>
#include "sparkpipe/spark_error_site.h"
#include <errno.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_k3_llm_defines.h"
#include "sparkpipe/spark_k3_model.h"
#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "sparkpipe/spark_k3_serving_adapter.h"
#include "sparkpipe/spark_memory_buffer.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_speculation_seam.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_serving_cache_admission.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_k3_kv_geometry.h"
#include "sparkpipe/spark_k3_kv_shard.h"

#include "spark_k3_dspark_format.h"
#include "inference/llms/kimi_k3/spec_verify.h"

#if SPARK_K3_SERVING_TOPOLOGY == 404
#define SPARK_K3_SERVING_ADAPTER_ID "k3-tp4pp4"
#define SPARK_K3_SERVING_TP_DEGREE 4u
#define SPARK_K3_SERVING_PARALLEL_GROUP_SIZE 4u
#define SPARK_K3_SERVING_CAPABILITIES \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP)
#define SPARK_K3_SERVING_STAGE_LAYERS \
	{ \
		SPARK_K3_PP_STAGE_LAYERS(0u), SPARK_K3_PP_STAGE_LAYERS(0u), \
		SPARK_K3_PP_STAGE_LAYERS(0u), SPARK_K3_PP_STAGE_LAYERS(0u), \
		SPARK_K3_PP_STAGE_LAYERS(1u), SPARK_K3_PP_STAGE_LAYERS(1u), \
		SPARK_K3_PP_STAGE_LAYERS(1u), SPARK_K3_PP_STAGE_LAYERS(1u), \
		SPARK_K3_PP_STAGE_LAYERS(2u), SPARK_K3_PP_STAGE_LAYERS(2u), \
		SPARK_K3_PP_STAGE_LAYERS(2u), SPARK_K3_PP_STAGE_LAYERS(2u), \
		SPARK_K3_PP_STAGE_LAYERS(3u), SPARK_K3_PP_STAGE_LAYERS(3u), \
		SPARK_K3_PP_STAGE_LAYERS(3u), SPARK_K3_PP_STAGE_LAYERS(3u) \
	}
#define SPARK_K3_SERVING_SIDEBAND_KIND \
	SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_RESIDUAL_BANK
#define SPARK_K3_SERVING_SIDEBAND_BYTES SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW
#elif SPARK_K3_SERVING_TOPOLOGY == 16
#define SPARK_K3_SERVING_ADAPTER_ID "k3-tp16"
#define SPARK_K3_SERVING_TP_DEGREE 16u
#define SPARK_K3_SERVING_PARALLEL_GROUP_SIZE 0u
#define SPARK_K3_SERVING_CAPABILITIES \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION)
#define L SPARK_K3_MODEL_LAYER_COUNT
#define SPARK_K3_SERVING_STAGE_LAYERS \
	{ L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L }
#define SPARK_K3_SERVING_SIDEBAND_KIND 0u
#define SPARK_K3_SERVING_SIDEBAND_BYTES 0u
#else
#error "SPARK_K3_SERVING_TOPOLOGY must be 404 (TP4xPP4) or 16 (TP16)"
#endif

#define SPARK_K3_SERVING_CONTRACT_SHA256 "318d979200eb3c6784be6f932febe14832b48df53a1520a73af2f03bd39bb217"
#define SPARK_K3_SERVING_MODULE_TAG "k3_stage"
#define SPARK_K3_SERVING_MAX_LANES 16u
#define SPARK_K3_SERVING_MAX_ROWS 64u
#define SPARK_K3_SERVING_PROGRAM_ID 1u

#define SPARK_K3_SEAM_DRAFT_TIME_BUDGET_MS 20u
#define SPARK_K3_SEAM_DRAFT_MAX_DEPTH 16u
#define SPARK_K3_SEAM_DRAFT_MAX_NODE_COUNT 64u
#define SPARK_K3_SEAM_CONNECT_TIMEOUT_MS 1000u
#define SPARK_K3_SEAM_IO_TIMEOUT_MS 30000u
#define SPARK_K3_SEAM_TARGET_MODEL SPARK_K3_MODEL_SOURCE_ID

typedef struct SparkK3SpeculationKnobs
{
	uint32_t draft_depth;
	uint32_t dynamic_draft_depth;
	uint32_t dynamic_window;
	uint32_t dynamic_minimum;
} SparkK3SpeculationKnobs;

typedef struct SparkK3ServingState
{
	SparkK3StageRunner runner;
	SparkK3StageRunnerConfiguration runner_config;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	char *pack_path;
	uint32_t max_rows;
	SparkTpDeviceCollectiveConfig device_config;
	SparkWeightdClient *lane_client;
	SparkTpDeviceCollectiveConfig device_config_wide;
	SparkTpDeviceCollectiveTopology device_topology;
	SparkK3SpeculationKnobs speculation;
	char device_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
		[SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
	int device_collective_present;
	SparkMemoryBuffer positions_host;
	SparkMemoryBuffer context_host;
	SparkMemoryBuffer state_host;
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
	SparkSpeculationSeam *speculation_seam;
	SparkStageModuleLedger ledger;
	SparkStageKvBinding kv;
	SparkModelDriverCacheLane cache_lanes[SPARK_K3_SERVING_MAX_LANES];
	SparkModelDriverCacheLane prefetch_lanes[SPARK_K3_SERVING_MAX_LANES];
	atomic_uint lane_states[SPARK_K3_SERVING_MAX_LANES];
	atomic_ullong reset_generation;
	SparkModelDriverCompletion publish_completion;
	uint32_t publish_completed;
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

static uint32_t K3ServingEnvU32(const char *name, uint32_t fallback)
{
	const char *value = getenv(name);
	char *end = 0;
	unsigned long long parsed;
	if ( value == 0 || *value == '\0' )
		return fallback;
	parsed = strtoull(value, &end, 10);
	if ( end == value || parsed > 0xffffffffull )
		return fallback;
	return (uint32_t)parsed;
}

static SparkStatus K3ServingLoadSpeculation(SparkK3ServingState *state,
	SparkJsonDocument *doc, int32_t root)
{
	int32_t spec = SparkJsonFindObjectMember(doc, root, "speculative");
	uint32_t depth = SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT;
	uint32_t dynamic = 0u;
	uint32_t window = K3_ADAPTIVE_DEPTH_WINDOW;
	uint32_t minimum = K3_ADAPTIVE_DEPTH_MINIMUM;
	if ( spec >= 0 )
	{
		depth = K3ServingJsonU32(doc, spec, "draft_depth", depth);
		dynamic = K3ServingJsonU32(doc, spec, "dynamic_draft_depth", dynamic);
		window = K3ServingJsonU32(doc, spec, "dynamic_draft_depth_window", window);
		minimum = K3ServingJsonU32(doc, spec, "dynamic_draft_min_depth", minimum);
	}
	dynamic = K3ServingEnvU32("SPARK_K3_DYNAMIC_DRAFT_DEPTH", dynamic);
	window = K3ServingEnvU32("SPARK_K3_DYNAMIC_DRAFT_DEPTH_WINDOW", window);
	minimum = K3ServingEnvU32("SPARK_K3_DYNAMIC_MIN_DEPTH", minimum);
	if ( depth == 0u || depth > SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT )
	{
		fprintf(stderr, "k3_serving speculative.draft_depth %u outside "
			"[1,%u]\n", depth, SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( dynamic > 1u )
	{
		fprintf(stderr, "k3_serving speculative.dynamic_draft_depth "
			"must be 0 or 1\n");
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( dynamic != 0u && (window == 0u ||
		window > K3_ADAPTIVE_DEPTH_WINDOW || minimum == 0u ||
		minimum > depth) )
	{
		fprintf(stderr, "k3_serving dynamic draft knobs invalid: "
			"window=%u (max %u) minimum=%u depth=%u\n",
			window, K3_ADAPTIVE_DEPTH_WINDOW, minimum, depth);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	state->speculation.draft_depth = depth;
	state->speculation.dynamic_draft_depth = dynamic;
	state->speculation.dynamic_window = window;
	state->speculation.dynamic_minimum = minimum;
	return SPARK_STATUS_OK;
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
	state->runner_config.tp_degree = K3ServingJsonU32(&doc, root, "tp_degree", 1u);
	{
		uint32_t world_size = K3ServingJsonU32(&doc, root, "world_size", 16u);
		if ( state->runner_config.tp_degree != SPARK_K3_SERVING_TP_DEGREE ||
			world_size != 16u )
			{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
		state->runner_config.stage_index =
			configuration->stage_index / state->runner_config.tp_degree;
		state->runner_config.stage_count =
			world_size / state->runner_config.tp_degree;
		state->runner_config.tp_rank =
			configuration->stage_index % state->runner_config.tp_degree;
	}
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
	state->runner_config.kv_page_bytes = 0u;
	{
		int32_t dev = SparkJsonFindObjectMember(&doc, root, "device_collective");
		if ( dev >= 0 )
		{
			uint32_t hidden = K3ServingJsonU32(&doc, root, "hidden",
				SPARK_K3_MODEL_HIDDEN_DIMENSION);
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
			state->device_config.credit_count = 8u;
			state->device_config.local_hidden_dimension = hidden;
			state->device_config.max_active_sequence_count =
				SparkK3KvShardSequenceCapacity(state->runner_config.max_input_row_count,
					state->runner_config.tp_degree);
			int32_t hosts_token = SparkJsonFindObjectMember(&doc, dev, "peer_hosts");
			uint32_t peer_count = hosts_token >= 0 ?
				SparkJsonGetArrayElementCount(&doc, hosts_token) : 0u;
			if ( peer_count == 0u ||
				peer_count > SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE )
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			state->device_topology.rank_count = peer_count;
			state->device_topology.algorithm_mask =
				SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE;
			{
				int32_t wait_token = SparkJsonFindObjectMember(&doc, dev, "wait_mode");
				if ( wait_token >= 0 && SparkJsonStringEquals(&doc, wait_token, "hardware") )
					state->device_topology.wait_mode = SPARK_TP_DEVICE_COLLECTIVE_WAIT_HARDWARE;
				else if ( wait_token >= 0 && !SparkJsonStringEquals(&doc, wait_token, "spin") )
					{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			}
			{
				int32_t ports_token = SparkJsonFindObjectMember(&doc, dev, "session_ports");
				uint32_t ports_rows = ports_token >= 0 ?
					SparkJsonGetArrayElementCount(&doc, ports_token) : 0u;
				if ( ports_rows != peer_count )
					{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
				for ( uint32_t row = 0u; row < peer_count; ++row )
				{
					int32_t row_token = SparkJsonGetArrayElement(&doc, ports_token, row);
					if ( row_token < 0 ||
						!SparkJsonTokenIsType(&doc, row_token, SPARK_JSON_TOKEN_ARRAY) ||
						SparkJsonGetArrayElementCount(&doc, row_token) != peer_count )
						{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
					for ( uint32_t column = 0u; column < peer_count; ++column )
					{
						int32_t cell = SparkJsonGetArrayElement(&doc, row_token, column);
						uint32_t value = 0u;
						if ( cell < 0 || SparkJsonGetUInt32(&doc, cell, &value) != SPARK_STATUS_OK ||
							value > 65535u ||
							(row == column ? value != 0u : value == 0u) )
							{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
						state->device_topology.session_ports[row][column] = (uint16_t)value;
					}
				}
			}
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
				memcpy(state->device_topology.rank_hosts[i],
					state->device_hosts[i],
					SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES);
			}
			state->device_collective_present = 1;
		}
	}
	state->runner_config.rank_pack_path = state->pack_path;
	state->runner_config.execution_stream = configuration->execution_stream;
	state->runner_config.multiprocessors = 48u;
	if ( state->runner_config.tp_degree > 1u && state->device_collective_present == 0 )
	{
		fprintf(stderr, "k3 adapter: tp_degree %u needs a device_collective\n", state->runner_config.tp_degree);
		SparkJsonDocumentDestroy(&doc);
		return SPARK_STATUS_SCHEMA_ERROR;
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
		{
			const char *socket = getenv("SPARK_WEIGHTD_SOCKET");
			const char *lane_text = getenv("SPARK_WEIGHTD_LANE");
			SparkWeightdMeshTopology mesh_topology;
			uint32_t requested_lane = SPARK_WEIGHTD_LANE_NONE;
			uint32_t resolved_lane;
			char *end = 0;
			unsigned long value;
			if ( socket == 0 || socket[0] == '\0' )
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_UNSUPPORTED; }
			if ( lane_text != 0 && lane_text[0] != '\0' )
			{
				errno = 0;
				value = strtoul(lane_text,&end,10);
				if ( lane_text[0] < '0' || lane_text[0] > '9' || end == lane_text ||
					*end != '\0' || errno != 0 ||
					value >= SPARK_WEIGHTD_MESH_MAX_LANES )
					{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_INVALID_ARGUMENT; }
				requested_lane = (uint32_t)value;
			}
			if ( SparkTpDeviceCollectiveMeshTopology(state->runner_config.tp_rank,
				state->runner_config.tp_degree,&mesh_topology) != SPARK_STATUS_OK )
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
			if ( SparkWeightdClientConnect(socket,&state->lane_client,0) !=
				SPARK_STATUS_OK )
			{
				state->lane_client = 0;
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_IO_ERROR; }
			}
			if ( SparkWeightdClientLaneAcquire(state->lane_client,requested_lane,
				&mesh_topology,&resolved_lane,
				(uint64_t)state->device_config.connect_timeout_milli * 1000000ull) !=
				SPARK_STATUS_OK )
			{
				fprintf(stderr,"sparkpipe_k3: mesh lane acquire failed "
					"requested=%u status=cold19-shared-lane\n",requested_lane);
				(void)SparkWeightdClientClose(state->lane_client);
				state->lane_client = 0;
				{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_NO_LANE; }
			}
			fprintf(stderr,"sparkpipe_k3: mesh lane acquired requested=%u "
				"resolved=%u shared-owner\n",requested_lane,resolved_lane);
			state->device_config.mesh_lane_client = state->lane_client;
		}
		state->runner_config.device_collective = &state->device_config;
		state->device_config_wide = state->device_config;
		state->device_config_wide.local_hidden_dimension =
			SPARK_K3_MODEL_MOE_TOP_K *
			(SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION * 2u);
		state->device_config_wide.mesh_band_index = 1u;
		state->device_config_wide.control_port_base =
			state->device_config.control_port_base - 1u;
		state->device_config_wide.collective_identifier =
			state->device_config.collective_identifier ^
			0x0000800000000000ull;
		(void)SparkTpDeviceCollectiveApplyTopology(&state->device_topology,
			&state->device_config_wide);
		state->runner_config.device_collective_wide =
			&state->device_config_wide;
	}
	status = K3ServingLoadSpeculation(state, &doc, root);
	if ( status != SPARK_STATUS_OK )
		{ SparkJsonDocumentDestroy(&doc); return status; }
	SparkJsonDocumentDestroy(&doc);
	return SPARK_STATUS_OK;
}
static void K3ServingDestroy(void *adapter_state);


static SparkStatus K3ServingInitializeSpeculationSeam(SparkK3ServingState *state)
{
	SparkSpeculationSeamConfiguration seam_config;
	SparkStatus status;
	if ( getenv("SPARK_K3_SERVING_SPECULATE") != 0 )
	{
		fprintf(stderr, "k3_serving SPARK_K3_SERVING_SPECULATE is retired: "
			"use SPARK_K3_SPECULATORS (speculation source mask) instead\n");
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	memset(&seam_config, 0, sizeof(seam_config));
	seam_config.abi_version = SPARK_SPECULATION_SEAM_ABI_VERSION;
	seam_config.descriptor_bytes = SPARK_SPECULATION_SEAM_DESCRIPTOR_BYTES;
	seam_config.available_source_mask = 0u;
	seam_config.default_speculative_token_count =
		state->speculation.draft_depth;
	if ( state->speculation.dynamic_draft_depth != 0u )
		fprintf(stderr, "k3_serving dynamic draft depth enabled: "
			"ceiling=%u window=%u floor=%u (acceptance-driven; "
			"policy core K3AdaptiveDepth)\n",
			state->speculation.draft_depth,
			state->speculation.dynamic_window,
			state->speculation.dynamic_minimum);
	seam_config.lane_count = state->runner_config.max_active_sequence_count;
	seam_config.max_committed_token_count =
		state->runner_config.kv_pages_per_sequence * K3_KV_PAGE_SLOTS;
	seam_config.max_tap_row_count = 0u;
	seam_config.draft_time_budget_ms = SPARK_K3_SEAM_DRAFT_TIME_BUDGET_MS;
	seam_config.draft_max_depth = SPARK_K3_SEAM_DRAFT_MAX_DEPTH;
	seam_config.draft_max_node_count = SPARK_K3_SEAM_DRAFT_MAX_NODE_COUNT;
	seam_config.connect_timeout_ms = SPARK_K3_SEAM_CONNECT_TIMEOUT_MS;
	seam_config.io_timeout_ms = SPARK_K3_SEAM_IO_TIMEOUT_MS;
	seam_config.control_value = getenv("SPARK_K3_SPECULATORS");
	memcpy(seam_config.target_model, SPARK_K3_SEAM_TARGET_MODEL,
		sizeof(SPARK_K3_SEAM_TARGET_MODEL));
	seam_config.model_contract.abi_version = SPARK_SPECULATION_ABI_VERSION;
	seam_config.model_contract.descriptor_bytes =
		SPARK_SPECULATION_MODEL_CONTRACT_DESCRIPTOR_BYTES;
	seam_config.model_contract.verifier_hidden_dtype =
		SPARK_SPECULATION_VERIFIER_HIDDEN_DTYPE_BF16;
	seam_config.model_contract.draft_dtype = SPARK_SPECULATION_DRAFT_DTYPE_BF16;
	seam_config.model_contract.draft_layer_count = SPARK_K3_DSPARK_LAYER_COUNT;
	seam_config.model_contract.block_size = SPARK_K3_DSPARK_BLOCK_SIZE;
	seam_config.model_contract.hidden_dimension = K3_HIDDEN;
	seam_config.model_contract.intermediate_dimension =
		SPARK_K3_DSPARK_FFN_INTERMEDIATE;
	seam_config.model_contract.attention_head_count =
		SPARK_K3_DSPARK_ATTN_QUERY_HEADS;
	seam_config.model_contract.kv_head_count = SPARK_K3_DSPARK_ATTN_KV_HEADS;
	seam_config.model_contract.head_dimension =
		SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION;
	seam_config.model_contract.vocab_size = K3_VOCAB;
	seam_config.model_contract.draft_vocab_size = SPARK_K3_DSPARK_VOCAB;
	seam_config.model_contract.markov_rank = SPARK_K3_DSPARK_MARKOV_RANK;
	seam_config.model_contract.maximum_speculative_token_count =
		SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT;
	seam_config.model_contract.verifier_accept_k = 1u;
	seam_config.model_contract.enable_confidence_head = 1u;
	seam_config.model_contract.confidence_head_with_markov = 1u;
	status = SparkSpeculationSeamInitialize(&seam_config,
		&state->speculation_seam);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "k3_serving speculation seam init failed: status=%d\n",
			(int)status);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus K3ServingRecurrentCopy(void *context, uint32_t direction,
	uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	SparkK3ServingState *state = (SparkK3ServingState *)context;
	if ( state == 0 || (direction != SPARK_STAGE_KV_RECURRENT_TO_BUFFER &&
		direction != SPARK_STAGE_KV_RECURRENT_FROM_BUFFER) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return SparkK3StageRunnerRecurrentCopy(&state->runner,
		direction == SPARK_STAGE_KV_RECURRENT_TO_BUFFER ? 1u : 0u,
		slot, buffer, bytes, stream);
}

static SparkStatus K3ServingDriverAdmit(void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkK3ServingState *state = (SparkK3ServingState *)driver_instance;
	return SparkStageKvBindingAdmit(&state->kv, request, decision);
}

static const SparkModelDriverInterface K3ServingDriver =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = (uint32_t)sizeof(SparkModelDriverInterface),
	.admit = K3ServingDriverAdmit,
};

static SparkStatus K3ServingBindKv(SparkK3ServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStageKvConfiguration kv;
	SparkK3StageRunnerKv attach;
	SparkStatus status;
	memset(&kv, 0, sizeof(kv));
	state->ledger.module_tag = SPARK_K3_SERVING_MODULE_TAG;
	kv.module_tag = SPARK_K3_SERVING_MODULE_TAG;
	kv.block_token_count = SPARK_K3_KV_PAGE_SLOTS;
	kv.region_count = 1u;
	kv.regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv.regions[0].layer_count = SparkK3StageRunnerKvLayerCount(&state->runner);
	kv.regions[0].layer_page_bytes = (uint64_t)SPARK_K3_KV_PAGE_SLOTS *
		SPARK_K3_MODEL_MLA_KV_A_DIMENSION * SPARK_K3_KV_BYTES_PER_SCALAR;
	kv.arena_kv_head_count = 1u;
	kv.arena_head_dim = SPARK_K3_MODEL_MLA_KV_A_DIMENSION;
	kv.arena_bytes_per_scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	SparkK3KvFillCapacityRequest(&kv.capacity_request);
	kv.capacity_request.layer_count = kv.regions[0].layer_count;
	kv.model_id = SPARK_K3_MODEL_SOURCE_ID;
	kv.model_revision = SPARK_K3_SERVING_ADAPTER_ID;
	kv.resident_sequence_capacity = state->runner_config.resident_sequence_capacity;
	kv.max_sequence_positions = state->runner_config.kv_pages_per_sequence * SPARK_K3_KV_PAGE_SLOTS;
	kv.max_input_row_count = state->max_rows;
	kv.logical_page_count = configuration->runtime_limits.kv_logical_page_capacity;
	kv.physical_page_count = configuration->runtime_limits.kv_physical_page_capacity;
	kv.pipeline_slot_count = 1u;
	kv.backing_directory = configuration->kv_backing_directory;
	kv.backing_maximum_bytes = configuration->kv_backing_maximum_bytes;
	kv.snapshot_directory = configuration->kv_snapshot_directory;
	kv.snapshot_maximum_bytes = configuration->kv_snapshot_maximum_bytes;
	if ( state->runner_config.tp_degree > 1u )
		kv.context_shard = SparkK3KvShardContext(state->runner_config.tp_rank, state->runner_config.tp_degree);
	status = SparkK3StageRunnerPackIdentity(&state->runner, kv.pack_sha256, sizeof(kv.pack_sha256));
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s kv binding refused: the runner has no weightd pack, so the KV layout has no pack identity\n",
			SPARK_K3_SERVING_MODULE_TAG);
		SPARK_RETURN(status);
	}
	if ( SparkSha256HexToDigest(SPARK_K3_SERVING_CONTRACT_SHA256, kv.contract_sha256) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	kv.expert_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1;
	kv.kv_codec = SPARK_WEIGHT_CODEC_BF16;
	kv.driver_symbol = (const void *)&K3ServingBindKv;
	kv.recurrent.lane_bytes = SparkK3StageRunnerRecurrentBytes(&state->runner);
	if ( kv.recurrent.lane_bytes != 0u )
	{
		kv.recurrent.copy = K3ServingRecurrentCopy;
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
	return SparkK3StageRunnerAttachKv(&state->runner, &attach);
}

static SparkServingCacheAdmission K3ServingCacheContext(SparkK3ServingState *state,
	SparkModelDriverCacheLane *lanes)
{
	SparkServingCacheAdmission cache;
	memset(&cache, 0, sizeof(cache));
	cache.program_id = SPARK_K3_SERVING_PROGRAM_ID;
	cache.lane_capacity = SPARK_K3_SERVING_MAX_LANES;
	cache.lanes = lanes;
	cache.driver = &K3ServingDriver;
	cache.driver_instance = state;
	cache.adapter_state = state;
	return cache;
}

typedef struct SparkK3ServingContinuity
{
	SparkK3ServingState *state;
	uint32_t rows;
	uint32_t active;
	const uint32_t *row_slots;
	const uint64_t *row_sequence_ids;
	const uint64_t *row_positions;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkK3ServingContinuity;

static SparkStatus K3ServingPrepareContinuity(void *context)
{
	SparkK3ServingContinuity *continuity = (SparkK3ServingContinuity *)context;
	return SparkStageKvBindingContinuity(&continuity->state->kv, continuity->state->lane_states,
		continuity->rows, continuity->active, continuity->row_slots, continuity->row_sequence_ids,
		continuity->row_positions, continuity->bound, continuity->sequence_ids, continuity->next_positions);
}

static void K3ServingPublishCompletion(void *context, const SparkModelDriverCompletion *completion)
{
	SparkK3ServingState *state = (SparkK3ServingState *)context;
	state->publish_completion = *completion;
	state->publish_completed = 1u;
}

static void K3ServingBuildFrame(const SparkModelServingSubmission *submission,
	SparkModelDriverCacheLane *lanes, uint32_t lane_count, void *stream,
	SparkModelDriverFrame *frame)
{
	memset(frame, 0, sizeof(*frame));
	frame->program_id = SPARK_K3_SERVING_PROGRAM_ID;
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

static SparkStatus K3ServingAdmitFrame(SparkK3ServingState *state,
	const SparkModelServingSubmission *submission, SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	uint32_t lane_count = 0u;
	SparkStatus status;
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission, state->cache_lanes,
		SPARK_K3_SERVING_MAX_LANES, &lane_count);
	if ( status == SPARK_STATUS_OK && lane_count != submission->active_sequence_count )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkAdmissionRequestFromSubmission(SPARK_K3_SERVING_PROGRAM_ID, submission,
			state->cache_lanes, 0u, &request);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	K3ServingBuildFrame(submission, state->cache_lanes, lane_count,
		state->runner_config.execution_stream, frame);
	return SparkAdmissionEvaluateAndApply(&K3ServingDriver, state, &request, frame, &decision);
}

static SparkStatus K3ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkK3ServingState *state;
	SparkStatus status;
	if ( configuration == 0 || adapter_state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkK3ServingState *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = K3ServingLoadConfiguration(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ free(state); return status; }
	state->max_rows = state->runner_config.max_input_row_count;
	if ( state->max_rows == 0u || state->max_rows > SPARK_K3_SERVING_MAX_ROWS ||
		state->runner_config.max_active_sequence_count == 0u ||
		state->runner_config.max_active_sequence_count > SPARK_K3_SERVING_MAX_LANES )
		{ free(state->pack_path); free(state); SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED); }
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states, SPARK_K3_SERVING_MAX_LANES);
	atomic_init(&state->reset_generation, 0u);
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	status = SparkMemoryBufferAllocate(&state->positions_host,
		SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->context_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT,
			(uint64_t)state->runner_config.max_active_sequence_count * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->state_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->positions_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->context_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,
			(uint64_t)state->runner_config.max_active_sequence_count * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->state_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->runs_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT,
			((uint64_t)state->max_rows + 1u) * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->runs_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,
			((uint64_t)state->max_rows + 1u) * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->seqslot_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT,
			(uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->seqslot_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,
			(uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->order_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT,
			(uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->order_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,
			(uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->output_tokens,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,
			(uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->output_scores,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,
			(uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		{ K3ServingDestroy(state); return status == SPARK_STATUS_CAPACITY_EXCEEDED ?
			SPARK_STATUS_CAPACITY_EXCEEDED : status; }
	status = SparkK3StageRunnerInitialize(&state->runner, &state->runner_config);
	if ( status != SPARK_STATUS_OK )
		{ K3ServingDestroy(state); return status; }
	status = K3ServingBindKv(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ K3ServingDestroy(state); return status; }
	status = K3ServingInitializeSpeculationSeam(state);
	if ( status != SPARK_STATUS_OK )
		{ K3ServingDestroy(state); return status; }
	*adapter_state = state;
	return SPARK_STATUS_OK;
}

static void K3ServingDestroy(void *adapter_state)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	if ( state == 0 )
		return;
	SparkStageKvBindingDestroy(&state->kv);
	SparkStageModuleLedgerRelease(&state->ledger);
	SparkK3StageRunnerDestroy(&state->runner);
	if ( state->lane_client != 0 )
	{
		(void)SparkWeightdClientClose(state->lane_client);
		state->lane_client = 0;
	}
	SparkMemoryBufferFree(&state->positions_host);
	SparkMemoryBufferFree(&state->context_host);
	SparkMemoryBufferFree(&state->state_host);
	SparkMemoryBufferFree(&state->positions_device);
	SparkMemoryBufferFree(&state->context_device);
	SparkMemoryBufferFree(&state->state_device);
	SparkMemoryBufferFree(&state->runs_host);
	SparkMemoryBufferFree(&state->runs_device);
	SparkMemoryBufferFree(&state->seqslot_host);
	SparkMemoryBufferFree(&state->seqslot_device);
	SparkMemoryBufferFree(&state->order_host);
	SparkMemoryBufferFree(&state->order_device);
	SparkMemoryBufferFree(&state->output_tokens);
	SparkMemoryBufferFree(&state->output_scores);
	free(state->pack_path);
	free(state);
}

static SparkStatus K3ServingReleaseSlots(const SparkK3ServingState *state,
	const SparkModelServingSubmission *submission, uint32_t *slots)
{
	uint32_t count = submission->active_sequence_count;
	if ( submission->row_count != 0u ||
		count > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT ||
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

static SparkStatus K3ServingValidateSubmission(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	if ( state == 0 || submission == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( submission->control_generation < atomic_load_explicit(&state->reset_generation, memory_order_acquire) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( SparkModelServingWorkKindUsesRows(submission->work_kind) == 0u )
	{
		if ( submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE &&
			submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH )
			return SPARK_STATUS_UNSUPPORTED;
		return K3ServingReleaseSlots(state, submission, 0);
	}
	if ( submission->row_count == 0u || submission->row_count > state->max_rows )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( submission->row_sequence_ids == 0 || submission->lanes == 0 || submission->lane_count < submission->active_sequence_count )
		return SPARK_STATUS_VALIDATION_FAILED;
	return SPARK_STATUS_OK;
}

static void K3ServingCompletionHeader(const SparkModelServingSubmission *submission,
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

static SparkStatus K3ServingGroupRows(SparkK3ServingState *state,
	const SparkModelServingSubmission *submission, uint32_t rows,
	uint32_t *active_out)
{
	uint32_t *positions = (uint32_t *)state->positions_host.pointer;
	uint32_t *context = (uint32_t *)state->context_host.pointer;
	uint32_t *slot_of_row = (uint32_t *)state->state_host.pointer;
	uint32_t *begin = (uint32_t *)state->runs_host.pointer;
	uint32_t *order = (uint32_t *)state->order_host.pointer;
	uint32_t *slots = (uint32_t *)state->seqslot_host.pointer;
	uint32_t cursor[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t capacity = state->runner_config.max_active_sequence_count;
	uint32_t lanes = submission->lanes != 0 ? submission->active_sequence_count : rows;
	uint32_t lane, i;
	if ( rows == 0u || rows > state->max_rows ||
		lanes == 0u || lanes > rows || lanes > capacity ||
		lanes > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT ||
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

static SparkStatus K3ServingSubmitNonRow(SparkK3ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkModelServingCompletion completion;
	SparkModelDriverFrame frame;
	uint32_t slots[SPARK_K3_SERVING_MAX_LANES];
	SparkStatus status;
	status = K3ServingReleaseSlots(state, submission, slots);
	if ( status == SPARK_STATUS_OK )
		status = K3ServingAdmitFrame(state, submission, &frame);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH )
	{
		frame.completion_function = K3ServingPublishCompletion;
		frame.completion_context = state;
		state->publish_completed = 0u;
		status = SparkStageKvBindingPublishFrame(&state->kv, &frame, state->lane_states);
		if ( status == SPARK_STATUS_OK && (state->publish_completed == 0u || state->publish_completion.status != SPARK_STATUS_OK) )
			status = state->publish_completed == 0u ? SPARK_STATUS_INTERNAL_ERROR : state->publish_completion.status;
	}
	else
		status = SparkK3StageRunnerResetSlots(&state->runner, slots, submission->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->completion_function != 0 )
	{
		K3ServingCompletionHeader(submission, 0u, &completion);
		state->completion_function(state->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

static void K3ServingContinuityRows(SparkK3ServingState *state,
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

static SparkStatus K3ServingFinish(SparkK3ServingState *state, uint32_t active,
	const uint32_t *lane_slots, uint8_t *bound, uint64_t *sequence_ids,
	uint64_t *next_positions, SparkStatus status)
{
	SparkStageKvBindingCompletion completion;
	memset(&completion, 0, sizeof(completion));
	completion.lane_count = active;
	completion.status = status;
	completion.resident_slots = lane_slots;
	completion.bound = bound;
	completion.sequence_ids = sequence_ids;
	completion.next_positions = next_positions;
	return SparkStageKvBindingFinishWait(&state->kv, 0u, &completion);
}

static SparkStatus K3ServingSubmit(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	SparkK3StageRunnerDispatch dispatch;
	SparkModelDriverFrame frame;
	SparkK3ServingContinuity continuity;
	uint32_t row_slots[SPARK_K3_SERVING_MAX_ROWS];
	uint64_t row_sequence_ids[SPARK_K3_SERVING_MAX_ROWS], row_positions[SPARK_K3_SERVING_MAX_ROWS];
	uint8_t bound[SPARK_K3_SERVING_MAX_LANES];
	uint64_t sequence_ids[SPARK_K3_SERVING_MAX_LANES], next_positions[SPARK_K3_SERVING_MAX_LANES];
	uint32_t distribution_rows[SPARK_K3_SERVING_MAX_LANES], distribution_positions[SPARK_K3_SERVING_MAX_LANES], distribution_lanes[SPARK_K3_SERVING_MAX_LANES];
	SparkRowSampling distribution_rules[SPARK_K3_SERVING_MAX_LANES];
	SparkSamplingLogprob distribution_logprobs[SPARK_K3_SERVING_MAX_LANES * SPARK_SAMPLING_MAX_LOGPROBS];
	uint32_t rows, active = 0u, distribution_count = 0u, logprob_stride = 0u;
	SparkStatus status, finished;
	void *stream;
	status = K3ServingValidateSubmission(state, submission);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( SparkModelServingWorkKindUsesRows(submission->work_kind) == 0u )
		return K3ServingSubmitNonRow(state, submission);
	stream = state->runner_config.execution_stream;
	rows = submission->row_count;
	status = K3ServingGroupRows(state, submission, rows, &active);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = K3ServingAdmitFrame(state, submission, &frame);
	if ( status != SPARK_STATUS_OK )
		return status;
	K3ServingContinuityRows(state, submission, rows, active, row_slots, row_sequence_ids, row_positions);
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
	continuity.row_sequence_ids = row_sequence_ids;
	continuity.row_positions = row_positions;
	continuity.bound = bound;
	continuity.sequence_ids = sequence_ids;
	continuity.next_positions = next_positions;
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states, SPARK_K3_SERVING_MAX_LANES,
		row_slots, active, K3ServingPrepareContinuity, &continuity);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkStageKvBindingClaim(&state->kv, &frame, active, row_slots, row_sequence_ids, row_positions, next_positions);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states, SPARK_K3_SERVING_MAX_LANES, row_slots, active);
		return status;
	}
	status = SparkStageKvBindingFenceExecution(&state->kv, stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRecurrentRestore(&state->kv, row_slots, active, stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingUploadPageTables(&state->kv, row_slots, active, stream);
	if ( status == SPARK_STATUS_OK )
	{
		if ( SparkMemoryBufferCopy(&state->positions_device, &state->positions_host,
				(uint64_t)rows * sizeof(uint32_t), state->runner_config.execution_stream) != SPARK_STATUS_OK ||
			SparkMemoryBufferCopy(&state->context_device, &state->context_host,
				(uint64_t)state->runner_config.max_active_sequence_count * sizeof(uint32_t),
				state->runner_config.execution_stream) != SPARK_STATUS_OK ||
			SparkMemoryBufferCopy(&state->state_device, &state->state_host,
				(uint64_t)rows * sizeof(uint32_t), state->runner_config.execution_stream) != SPARK_STATUS_OK ||
			SparkMemoryBufferCopy(&state->runs_device, &state->runs_host,
				((uint64_t)active + 1u) * sizeof(uint32_t),
				state->runner_config.execution_stream) != SPARK_STATUS_OK ||
			SparkMemoryBufferCopy(&state->seqslot_device, &state->seqslot_host,
				(uint64_t)active * sizeof(uint32_t),
				state->runner_config.execution_stream) != SPARK_STATUS_OK ||
			SparkMemoryBufferCopy(&state->order_device, &state->order_host,
				(uint64_t)rows * sizeof(uint32_t),
				state->runner_config.execution_stream) != SPARK_STATUS_OK )
				status = SPARK_STATUS_IO_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
	{
		memset(&dispatch, 0, sizeof(dispatch));
		dispatch.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
		dispatch.descriptor_bytes = (uint32_t)sizeof(dispatch);
		dispatch.request_id = submission->request_id;
		dispatch.sequence_id = submission->sequence_id;
		dispatch.sequence_position = submission->sequence_position;
		dispatch.deadline_time_ns = submission->deadline_time_ns;
		dispatch.row_count = rows;
		dispatch.active_sequence_count = active;
		dispatch.token_ids = submission->token_ids;
		dispatch.positions = state->positions_device.pointer;
		dispatch.context_length = state->context_device.pointer;
		dispatch.sequence_of_row = state->state_device.pointer;
		dispatch.sequence_row_begin = state->runs_device.pointer;
		dispatch.sequence_row_indices = state->order_device.pointer;
		dispatch.kda_state_index = state->seqslot_device.pointer;
		dispatch.hidden_input_bf16 = submission->hidden_input_address;
		dispatch.hidden_input_bytes = submission->hidden_input_bytes;
		dispatch.hidden_output_bf16 = submission->hidden_output_address;
		dispatch.hidden_output_bytes = submission->hidden_output_bytes;
		dispatch.residual_bank_input = submission->boundary_sideband_input_address;
		dispatch.residual_bank_input_bytes = submission->boundary_sideband_input_bytes;
		dispatch.residual_bank_output = submission->boundary_sideband_output_address;
		dispatch.residual_bank_output_bytes = submission->boundary_sideband_output_bytes;
		dispatch.output_token_ids = state->output_tokens.pointer;
		dispatch.output_scores = state->output_scores.pointer;
		dispatch.distribution_count = distribution_count;
		dispatch.distribution_rows = distribution_rows;
		dispatch.distribution_positions = distribution_positions;
		dispatch.distribution_rules = distribution_rules;
		dispatch.distribution_logprobs = distribution_logprobs;
		dispatch.completion_function = 0;
		dispatch.completion_context = 0;
		status = SparkK3StageRunnerSubmit(&state->runner, &dispatch);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRecurrentCapture(&state->kv, row_slots, active, stream);
	if ( status == SPARK_STATUS_OK && cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	finished = K3ServingFinish(state, active, row_slots, bound, sequence_ids, next_positions, status);
	SparkStageModuleIndexSetRelease(state->lane_states, SPARK_K3_SERVING_MAX_LANES, row_slots, active);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( finished != SPARK_STATUS_OK )
		return finished;
	if ( state->completion_function != 0 )
	{
		SparkModelServingCompletion completion;
		K3ServingCompletionHeader(submission, rows, &completion);
		if ( state->runner.owns_final_head != 0u )
		{
			uint32_t *runs = (uint32_t *)state->runs_host.pointer;
			uint32_t *order = (uint32_t *)state->order_host.pointer;
			uint32_t sequences = dispatch.active_sequence_count;
			uint32_t *tokens_host = (uint32_t *)malloc((uint64_t)rows * 4u);
			if ( tokens_host == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			SparkMemoryBuffer tokens = SPARK_MEMORY_BUFFER_VIEW(tokens_host,
				SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)rows * 4u);
			if ( SparkMemoryBufferCopy(&tokens, &state->output_tokens,
					(uint64_t)rows * 4u, 0) != SPARK_STATUS_OK )
				{ free(tokens_host); SPARK_FAIL(SPARK_STATUS_IO_ERROR); }
			completion.tokens_per_sequence = 1u;
			completion.token_count = sequences;
			completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
			for ( uint32_t s = 0u; s < sequences && s < SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT; ++s )
				completion.token_ids[s] = tokens_host[order[runs[s + 1u] - 1u]];
			free(tokens_host);
			completion.logprob_stride = logprob_stride;
			completion.logprob_entry_count = completion.token_count * logprob_stride;
			for ( uint32_t entry = 0u; logprob_stride != 0u && entry < distribution_count; ++entry )
				memcpy(&completion.logprobs[(uint64_t)distribution_lanes[entry] * logprob_stride],
					&distribution_logprobs[(uint64_t)entry * SPARK_SAMPLING_MAX_LOGPROBS],
					(uint64_t)distribution_rules[entry].logprobs * sizeof(SparkSamplingLogprob));
		}
		state->completion_function(state->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingPrefetch(void *adapter_state,
	const SparkModelServingSubmission *submissions, uint32_t submission_count)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	cache = K3ServingCacheContext(state, state->prefetch_lanes);
	cache.validate = K3ServingValidateSubmission;
	return SparkServingCacheAdmissionRun(&cache, submissions, submission_count,
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
}

static SparkStatus K3ServingCacheHint(void *adapter_state,
	const SparkModelServingCacheIdentity *identity, uint32_t token_count)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	cache = K3ServingCacheContext(state, state->prefetch_lanes);
	return SparkServingCacheHintRun(&cache, identity, token_count);
}

static SparkStatus K3ServingResolvePrefetch(void *adapter_state,
	const SparkModelServingSubmission *submission, uint32_t resolution)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 || (resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	cache = K3ServingCacheContext(state, state->prefetch_lanes);
	cache.validate = K3ServingValidateSubmission;
	return SparkServingCacheAdmissionRun(&cache, submission, 1u,
		resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ?
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT : SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT);
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
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingSnapshot(void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	SparkK3StageRunnerStats stats;
	if ( state == 0 || snapshot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	snapshot->available_submission_count = state->max_rows;
	SparkK3StageRunnerGetStats(&state->runner, &stats);
	snapshot->submitted_count = stats.submitted_count;
	snapshot->completed_count = stats.completed_count;
	snapshot->resident_sequence_count = SparkStageKvBindingResidentCount(&state->kv);
	snapshot->kv_token_capacity = (uint64_t)state->kv.logical_page_count * SPARK_K3_KV_PAGE_SLOTS;
	snapshot->max_sequence_positions = state->runner_config.kv_pages_per_sequence * SPARK_K3_KV_PAGE_SLOTS;
	SparkStageKvBindingKvStoreCounters(&state->kv, &snapshot->kv_store);
	return SPARK_STATUS_OK;
}

static SparkStatus K3ServingReset(void *adapter_state, uint64_t control_generation)
{
	SparkK3ServingState *state = (SparkK3ServingState *)adapter_state;
	uint32_t slots[SPARK_K3_SERVING_MAX_LANES], slot, count;
	SparkStatus status;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkStageKvBindingResetIsNew(&state->kv, control_generation) == 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	count = state->kv.resident_sequence_capacity;
	for ( slot = 0u; slot < count; ++slot )
		slots[slot] = slot;
	status = SparkStageModuleIndexSetClaim(state->lane_states, SPARK_K3_SERVING_MAX_LANES, slots, count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( cudaStreamSynchronize((cudaStream_t)state->runner_config.execution_stream) != cudaSuccess ||
		(state->kv.copy_stream != 0 && cudaStreamSynchronize((cudaStream_t)state->kv.copy_stream) != cudaSuccess) )
	{
		fprintf(stderr, "%s reset: execution stream failed; lanes stay claimed\n", SPARK_K3_SERVING_MODULE_TAG);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = SparkK3StageRunnerResetSlots(&state->runner, slots, count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingReset(&state->kv, control_generation);
	if ( status == SPARK_STATUS_OK )
		atomic_store_explicit(&state->reset_generation, control_generation, memory_order_release);
	SparkStageModuleIndexSetRelease(state->lane_states, SPARK_K3_SERVING_MAX_LANES, slots, count);
	SPARK_RETURN(status);
}

static const SparkModelServingAdapterDescriptor K3ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_K3_SERVING_ADAPTER_ID,
		SPARK_K3_MODEL_SOURCE_ID,
		SPARK_K3_SERVING_ADAPTER_ID,
		"k3",
		SPARK_K3_SERVING_CONTRACT_SHA256),
	.capability_flags = SPARK_K3_SERVING_CAPABILITIES |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = 16u,
	.layer_count = SPARK_K3_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_K3_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = 16u,
	.max_input_row_count = SPARK_K3_SERVING_MAX_ROWS,
	.max_resident_sequence_count = 16u,
	.max_output_token_count = 16u,
	.max_speculative_token_count = SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT,
	.stage_layer_counts = SPARK_K3_SERVING_STAGE_LAYERS,
	.boundary_sideband_kinds =
	{
		SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND,
		SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND,
		SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND
	},
	.boundary_sideband_bytes_per_sequence =
	{
		SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES,
		SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES,
		SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES
	},
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_K3_KV_PAGE_SLOTS,
	.parallel_group_size = SPARK_K3_SERVING_PARALLEL_GROUP_SIZE,
	.cache_checkpoint_token_count = SPARK_K3_KV_CHECKPOINT_TOKENS,
};

#if SPARK_K3_SERVING_TOPOLOGY == 16
#undef L
#endif

static const SparkModelServingAdapterInterface K3ServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &K3ServingDescriptor,
	.initialize = K3ServingInitialize,
	.destroy = K3ServingDestroy,
	.validate_submission = K3ServingValidateSubmission,
	.submit = K3ServingSubmit,
	.prefetch = K3ServingPrefetch,
	.cache_hint = K3ServingCacheHint,
	.resolve_prefetch = K3ServingResolvePrefetch,
	.progress = K3ServingProgress,
	.quiesce = K3ServingQuiesce,
	.snapshot = K3ServingSnapshot,
	.reset = K3ServingReset,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return &K3ServingInterface;
}
