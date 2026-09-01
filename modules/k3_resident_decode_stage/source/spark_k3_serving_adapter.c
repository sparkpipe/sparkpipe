#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_k3_dspark_pack.h"
#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "sparkpipe/spark_k3_serving_adapter.h"
#include "sparkpipe/spark_memory_buffer.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_speculation_provider.h"

#include "spark_k3_dspark_format.h"

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
	/* Per-submit host-converted arrays and their device twins, every
	 * allocation naming its space (memory-M1, riding the template). */
	SparkMemoryBuffer positions_host;   /* HOST_COHERENT */
	SparkMemoryBuffer context_host;     /* HOST_COHERENT */
	SparkMemoryBuffer state_host;       /* HOST_COHERENT */
	SparkMemoryBuffer positions_device; /* DEVICE_PRIVATE */
	SparkMemoryBuffer context_device;   /* DEVICE_PRIVATE */
	SparkMemoryBuffer state_device;     /* DEVICE_PRIVATE */
	/* The KDA run prefix (rows+1 entries) and the per-run state slot
	 * (rows entries): multi-row prefill groups consecutive same-slot rows
	 * into one sequential run through the recurrence kernels. Host
	 * derivation, device consumption. */
	SparkMemoryBuffer runs_host;        /* HOST_COHERENT */
	SparkMemoryBuffer runs_device;      /* DEVICE_PRIVATE */
	SparkMemoryBuffer seqslot_host;     /* HOST_COHERENT */
	SparkMemoryBuffer seqslot_device;   /* DEVICE_PRIVATE */
	SparkMemoryBuffer output_tokens;    /* DEVICE_PRIVATE */
	SparkMemoryBuffer output_scores;    /* DEVICE_PRIVATE */
	/* The speculation-provider slot's first real use: the DSpark drafter
	 * pack binds file-backed at initialize when the operator arms it, and
	 * the provider rides on the bound pack. The draft forward is NOT
	 * landed - the draft ops fail closed naming that - so arming proves
	 * the wire path (bind + contract + refusal surfacing) without
	 * changing a serving cell. */
	SparkK3DsparkPack drafter_pack;
	uint32_t drafter_pack_bound;
	char speculation_refusal[SPARK_K3_DSPARK_MAX_REFUSAL_BYTES];
	SparkSpeculationProvider provider;
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
	/* world_rank = pp_stage * tp_degree + tp_rank; the deployment's world
	 * size fixes the PP stage count (4 for TP4xPP4, 1 for TP16). */
	{
		uint32_t world_size = K3ServingJsonU32(&doc, root, "world_size", 16u);
		if ( state->runner_config.tp_degree == 0u ||
			world_size % state->runner_config.tp_degree != 0u )
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
	if ( K3ServingJsonU32(&doc, root, "capture_graphs", 0u) != 0u )
		state->runner_config.flags |= SPARK_K3_STAGE_RUNNER_FLAG_CAPTURE_GRAPHS;
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
				memcpy(state->device_topology.rank_hosts[i],
					state->device_hosts[i],
					SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES);
			}
			/* The runner completes the config (degree/rank/combine functions)
			 * and applies the topology. */
			state->device_collective_present = 1;
		}
	}
	state->runner_config.rank_pack_path = state->pack_path;
	state->runner_config.execution_stream = configuration->execution_stream;
	state->runner_config.multiprocessors = 48u;
	/* The host TCP tier caps at SPARK_TP_COLLECTIVE_MAX_STEPS ranks; wider
	 * placements (TP16) must carry a device_collective. Within the cap the
	 * host tier is parsed for EVERY tp_degree > 1 (device tier or not):
	 * the runner builds it as the fallback path when both tiers exist, and
	 * refusing a null tp_collective there handed the runner a null config
	 * on every device-collective deployment (the 2026-08-30 fleet wave
	 * died 16/16 at adapter_initialize on exactly this). ABOVE the cap
	 * there is no host tier to parse (the generator omits tp_collective
	 * from TP16 configs): the runner takes the device tier alone and
	 * accepts a null tp_collective only because the device collective
	 * exists - the guard below already refused the tierless case. */
	if ( state->runner_config.tp_degree > SPARK_TP_COLLECTIVE_MAX_STEPS )
	{
		if ( state->device_collective_present == 0 )
			{ SparkJsonDocumentDestroy(&doc); return SPARK_STATUS_SCHEMA_ERROR; }
	}
	else if ( state->runner_config.tp_degree > 1u )
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
static void K3ServingDestroy(void *adapter_state);

/*
 * The embedded DSpark provider (the block-drafter binding shape from
 * tests/test_speculation_provider_slot.c): a static ops table whose state
 * is the bound drafter pack. LIFECYCLE + CONTRACT only - the draft inner
 * loop stays provider-owned kernels, which this family has not landed, so
 * draft_begin refuses with the reason instead of pretending (the
 * supports() -> WHY rule; the wire surface renders it, never a bare
 * UNSUPPORTED without a cause).
 */

static SparkStatus K3DsparkProviderCapabilityQuery(
	const SparkSpeculationGeometryQuery *geometry,
	char *refusal_buffer, uint32_t refusal_buffer_bytes)
{
	if ( geometry == 0 || geometry->hidden_dimension != K3_HIDDEN ||
		geometry->layer_count < 93u )
	{
		if ( refusal_buffer != 0 && refusal_buffer_bytes != 0u )
			(void)snprintf(refusal_buffer, refusal_buffer_bytes,
				"k3 dspark drafter requires the k3 target geometry "
				"(hidden %u, 93 layers), got hidden %u layers %u",
				(uint32_t)K3_HIDDEN,
				geometry != 0 ? geometry->hidden_dimension : 0u,
				geometry != 0 ? geometry->layer_count : 0u);
		return(SPARK_STATUS_UNSUPPORTED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus K3DsparkProviderDraftBegin(void *provider_state,
	const SparkSpeculationDraftRequest *request)
{
	(void)provider_state;
	(void)request;
	/* fail closed with the reason: the DSpark draft forward (5-layer
	 * backbone walk, markov bias, confidence gate) is not landed; only
	 * the wire path (pack format + bind) exists. Recorded as the kernel
	 * follow-up in the lane report. */
	return(SPARK_STATUS_UNSUPPORTED);
}

static SparkStatus K3DsparkProviderDraftNext(void *provider_state,
	SparkSpeculationDraft *draft)
{
	(void)provider_state;
	(void)draft;
	return(SPARK_STATUS_UNSUPPORTED);
}

static void K3DsparkProviderDraftCancel(void *provider_state)
{
	(void)provider_state;
}

/* THE one accepted-prefix accounting (where the lease-advance bug class
 * dies): block drafts are anchor-first, so a verified chain of N rows
 * commits N-1 drafts, and each sequence carries exactly what it accepted. */
static SparkStatus K3DsparkProviderVerifyAccount(void *provider_state,
	uint32_t verified_count, SparkSpeculationVerifyContract *contract_out)
{
	(void)provider_state;
	if ( contract_out == 0 || verified_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(contract_out, 0, sizeof(*contract_out));
	contract_out->chain_width = verified_count;
	contract_out->accepted_token_count = verified_count - 1u;
	contract_out->tokens_per_sequence = contract_out->accepted_token_count;
	contract_out->chain_live = 1u;
	return(SPARK_STATUS_OK);
}

static const SparkSpeculationKvContract K3DsparkKvContract =
{
	/* block-local v1: the drafter reads its scratch frame plus the
	 * anchor's tail frame; no multi-block history yet (the 27B's
	 * BLOCK_KV is the precedent to grow into) */
	.frame_flags = SPARK_SPECULATION_KV_FLAG_SCRATCH_FRAME |
		SPARK_SPECULATION_KV_FLAG_TAIL_FRAME,
	.block_history_depth = 0u
};

static const SparkSpeculationKvContract *K3DsparkProviderKvContract(
	void *provider_state)
{
	(void)provider_state;
	return(&K3DsparkKvContract);
}

static const SparkSpeculationProviderOps K3DsparkProviderOps =
{
	.capability_query = K3DsparkProviderCapabilityQuery,
	.draft_begin = K3DsparkProviderDraftBegin,
	.draft_next = K3DsparkProviderDraftNext,
	.draft_cancel = K3DsparkProviderDraftCancel,
	.verify_account = K3DsparkProviderVerifyAccount,
	.kv_contract = K3DsparkProviderKvContract
};

/* The canonical launch contract keys (the family env names
 * SPARK_K3_SERVING_SPECULATE / SPARK_K3_DSPARK_PACK_PATH are aliases
 * until the env migration, per the design's sequencing). */
static const char *const K3DsparkEnvironmentSchema[] =
{
	"SPEC_METHOD",
	"DRAFT_COUNT",
	"DSPARK_PACK_PATH"
};

static const SparkSpeculationProviderDescriptor K3DsparkProviderDescriptor =
{
	.abi_version = SPARK_SPECULATION_PROVIDER_ABI_VERSION,
	.descriptor_bytes = SPARK_SPECULATION_PROVIDER_DESCRIPTOR_BYTES,
	.kind = SPARK_SPECULATION_PROVIDER_DSPARK,
	.provider_id = "k3.dspark-drafter.redhatai.v1",
	.max_draft_token_count = SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT,
	.default_draft_token_count = SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT,
	.environment_schema = K3DsparkEnvironmentSchema,
	.environment_schema_count = 3u
};

static SparkStatus K3ServingBindSpeculationProvider(SparkK3ServingState *state)
{
	const char *speculate = getenv("SPARK_K3_SERVING_SPECULATE");
	const char *pack_path;
	SparkStatus status;
	if ( speculate == 0 || speculate[0] == '\0' || strcmp(speculate, "0") == 0 )
		return(SPARK_STATUS_OK); /* unarmed: cell-unchanged serving */
	pack_path = getenv("SPARK_K3_DSPARK_PACK_PATH");
	if ( pack_path == 0 || pack_path[0] == '\0' )
	{
		fprintf(stderr, "k3_serving speculation armed without a drafter: "
			"set SPARK_K3_DSPARK_PACK_PATH\n");
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = SparkK3DsparkPackBind(pack_path, &state->drafter_pack,
		state->speculation_refusal, sizeof(state->speculation_refusal));
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "k3_serving drafter pack refused: %s\n",
			state->speculation_refusal);
		return(status);
	}
	state->drafter_pack_bound = 1u;
	state->provider.descriptor = &K3DsparkProviderDescriptor;
	state->provider.ops = &K3DsparkProviderOps;
	state->provider.provider_state = &state->drafter_pack;
	status = SparkSpeculationProviderValidate(&state->provider);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "k3_serving speculation provider invalid: status=%d\n",
			(int)status);
		return(status);
	}
	fprintf(stderr, "k3_serving drafter bound pack=%s block=%u draft_depth=%u "
		"taps=[%u,%u,%u,%u,%u] tensors=%u draft_forward=%s\n",
		pack_path, state->drafter_pack.block_size,
		state->drafter_pack.draft_token_count,
		state->drafter_pack.target_tap_layers[0],
		state->drafter_pack.target_tap_layers[1],
		state->drafter_pack.target_tap_layers[2],
		state->drafter_pack.target_tap_layers[3],
		state->drafter_pack.target_tap_layers[4],
		state->drafter_pack.tensor_count, "not_landed_fail_closed");
	return(SPARK_STATUS_OK);
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
	status = SparkMemoryBufferAllocate(&state->positions_host,
		SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->context_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->state_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->positions_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (uint64_t)state->max_rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->context_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (uint64_t)state->max_rows * 4u);
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
			SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->seqslot_device,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->output_tokens,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->output_scores,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE, (uint64_t)state->max_rows * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		{ K3ServingDestroy(state); return status == SPARK_STATUS_CAPACITY_EXCEEDED ?
			SPARK_STATUS_CAPACITY_EXCEEDED : status; }
	status = SparkK3StageRunnerInitialize(&state->runner, &state->runner_config);
	if ( status != SPARK_STATUS_OK )
		{ K3ServingDestroy(state); return status; }
	status = K3ServingBindSpeculationProvider(state);
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
	SparkK3StageRunnerDestroy(&state->runner);
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
	SparkMemoryBufferFree(&state->output_tokens);
	SparkMemoryBufferFree(&state->output_scores);
	if ( state->drafter_pack_bound != 0u )
		SparkK3DsparkPackRelease(&state->drafter_pack);
	free(state->pack_path);
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
	/* row_positions is host memory in the serving ABI; a plain copy. */
	memcpy(positions_host64, submission->row_positions,
		(uint64_t)rows * sizeof(uint64_t));
	for ( uint32_t i = 0u; i < rows; ++i )
	{
		((uint32_t *)state->positions_host.pointer)[i] = (uint32_t)positions_host64[i];
		((uint32_t *)state->context_host.pointer)[i] = (uint32_t)positions_host64[i] + 1u;
		((uint32_t *)state->state_host.pointer)[i] = submission->lanes != 0
			? submission->lanes[submission->row_lane_indices != 0
				? submission->row_lane_indices[i] : i].resident_sequence_slot
			: i;
	}
	free(positions_host64);
	/* Space-aware copies (the tags resolve host-to-device); the pasted
	 * submits ignore copy errors and that is unchanged here. */
	(void)SparkMemoryBufferCopy(&state->positions_device,
		&state->positions_host, (uint64_t)rows * 4u, 0);
	(void)SparkMemoryBufferCopy(&state->context_device,
		&state->context_host, (uint64_t)rows * 4u, 0);
	(void)SparkMemoryBufferCopy(&state->state_device,
		&state->state_host, (uint64_t)rows * 4u, 0);
	/* The KDA runs: consecutive rows sharing a resident slot form ONE
	 * sequential run through the recurrence kernels (a multi-row prefill
	 * span of one sequence = one run; batch-decode rows are runs of one).
	 * The kernels address state per SEQUENCE (state_index[sequence]), so
	 * the run-slot array carries each run's first row's slot. A run count
	 * that disagrees with the submission's declared active count is a
	 * contract break - fail loud, never guess the grouping. */
	{
		uint32_t *runs = (uint32_t *)state->runs_host.pointer;
		uint32_t *slots = (uint32_t *)state->state_host.pointer;
		uint32_t *seqslots = (uint32_t *)state->seqslot_host.pointer;
		uint32_t active = 1u;
		runs[0] = 0u;
		for ( uint32_t i = 1u; i < rows; ++i )
			if ( slots[i] != slots[i - 1u] )
				runs[active++] = i;
		runs[active] = rows;
		if ( submission->active_sequence_count != 0u &&
			submission->active_sequence_count != active )
			return SPARK_STATUS_VALIDATION_FAILED;
		for ( uint32_t s = 0u; s < active; ++s )
			seqslots[s] = slots[runs[s]];
		(void)SparkMemoryBufferCopy(&state->runs_device,
			&state->runs_host, ((uint64_t)active + 1u) * sizeof(uint32_t), 0);
		(void)SparkMemoryBufferCopy(&state->seqslot_device,
			&state->seqslot_host, (uint64_t)active * sizeof(uint32_t), 0);
	}
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
	dispatch.positions = state->positions_device.pointer;
	dispatch.context_length = state->context_device.pointer;
	dispatch.sequence_of_row = state->state_device.pointer;
	/* Per-RUN state slots (the kernels index state per sequence) + the run
	 * prefix: one prefill span of a sequence is ONE chained KDA run. */
	dispatch.kda_state_index = state->seqslot_device.pointer;
	dispatch.sequence_row_begin = state->runs_device.pointer;
	dispatch.hidden_input_bf16 = submission->hidden_input_address;
	dispatch.hidden_input_bytes = submission->hidden_input_bytes;
	dispatch.hidden_output_bf16 = submission->hidden_output_address;
	dispatch.hidden_output_bytes = submission->hidden_output_bytes;
	dispatch.output_token_ids = state->output_tokens.pointer;
	dispatch.output_scores = state->output_scores.pointer;
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
		if ( tokens_host == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		memset(&completion, 0, sizeof(completion));
		completion.abi_version = submission->abi_version;
		completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
		completion.submission_id = submission->submission_id;
		completion.request_id = submission->request_id;
		completion.sequence_id = submission->sequence_id;
		completion.sequence_position = submission->sequence_position;
		completion.residency = submission->residency;
		completion.accepted_token_count = rows;
		completion.token_count = rows;
		completion.tokens_per_sequence = 1u;
		/* A copy-destination view of the per-submit scratch: the tags
		 * resolve device-to-host; the pasted submit ignored copy errors. */
		SparkMemoryBuffer tokens = SPARK_MEMORY_BUFFER_VIEW(tokens_host,
			SPARK_MEMORY_SPACE_HOST_COHERENT, (uint64_t)rows * 4u);
		(void)SparkMemoryBufferCopy(&tokens, &state->output_tokens,
			(uint64_t)rows * 4u, 0);
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
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
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
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		"k3-tp4pp4",
		"moonshotai/Kimi-K3-MXFP4",
		"k3-tp4pp4",
		"k3",
		"318d979200eb3c6784be6f932febe14832b48df53a1520a73af2f03bd39bb217"),
	/* The shared capability chain's base carries DRIVER_OWNS_KV, which k3
	 * must NOT declare: its slot-reuse contract is NONE and the shared
	 * descriptor validator pins the two together
	 * (runtime/model_serving_adapter.c). This list is the family's honest
	 * difference, not an un-migrated paste. */
	.capability_flags =
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP,
	/* One node per RANK (the hybrid's contract): the residentd checks the
	 * node count against this. */
	.stage_count = 16u,
	.layer_count = 93u,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = 7168u,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_NVFP4_E2M1,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = 16u,
	.max_input_row_count = 16u,
	.max_resident_sequence_count = 16u,
	.max_output_token_count = 16u,
	/* the DSpark drafter's envelope: block-1 draft tokens per round behind
	 * the provider slot (the descriptor is the envelope; the provider's
	 * capability query + bind carry the evidence). The shared validator
	 * pins this count to the SPECULATION capability flag above. */
	.max_speculative_token_count = SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT,
	.resident_sequence_slot_reuse = 0u,
	/* Stage-major (PP stage × TP rank): TP4 groups have equal counts (hybrid
	 * contract); PP stage layer splits sum to 93. */
	.stage_layer_counts = { 24u, 24u, 24u, 24u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u, 23u },
	.boundary_sideband_kinds = { 0u, 0u, 0u, 0u },
	.boundary_sideband_bytes_per_sequence = { 0u, 0u, 0u, 0u },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = 0u,
	.parallel_group_size = 4u,
};

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
	.resolve_prefetch = K3ServingResolvePrefetch,
	.progress = K3ServingProgress,
	.quiesce = K3ServingQuiesce,
	.snapshot = K3ServingSnapshot,
	.reset = K3ServingReset,
};

// The runtime dlsym's SPARK_MODEL_SERVING_ADAPTER_INTERFACE_SYMBOL, which
// is exactly "SparkModelServingAdapterGetInterface" - a family-specific
// name here is invisible to the loader (the found-and-fixed K3 export bug).
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return &K3ServingInterface;
}
