#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_ring_runtime.h"
#include "sparkpipe/spark_glm52_serving_engine.h"

#define PIPESIM_SPARK_COUNT SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define PIPESIM_LANE_CAPACITY 128u
#define PIPESIM_REQUEST_SLOTS 4096u
#define PIPESIM_KV_BLOCKS 32768u
#define PIPESIM_PREFIX_BINDINGS (PIPESIM_KV_BLOCKS + 64u)
#define PIPESIM_EVENT_CAPACITY 16384u
#define PIPESIM_PREFILL_STRIDE 256u
#define PIPESIM_PENDING_CAPACITY 64u
#define PIPESIM_QUEUE_DEPTH_PER_SPARK 14u
#define PIPESIM_TOKEN_STRIDE SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE
#define PIPESIM_REQUEST_TOKEN_STRIDE 8448u

typedef struct PipesimPending
{
	uint32_t active;
	uint32_t is_prefill;
	uint64_t completion_ns;
	SparkGlm52RequestApiDispatch dispatch;
} PipesimPending;

typedef struct PipesimRing
{
	uint64_t stage_free_ns[PIPESIM_SPARK_COUNT];
	uint64_t stage_ns;
	uint64_t prefill_stage_ns;
	uint64_t verify_stage_ns;
	uint64_t hop_ns;
} PipesimRing;

typedef struct PipesimStats
{
	uint64_t dispatch_count;
	uint64_t lane_dispatch_count;
	uint64_t width_histogram[PIPESIM_LANE_CAPACITY + 1u];
	uint64_t concurrency_weighted_ns;
	uint64_t observed_ns;
	uint32_t max_concurrent;
	uint64_t steady_begin_ns;
	uint64_t steady_begin_tokens;
} PipesimStats;

typedef struct PipesimFixture
{
	SparkGlm52KvCacheArena kv_arena;
	SparkGlm52KvCacheBlock kv_blocks[PIPESIM_KV_BLOCKS];
	SparkGlm52PrefixCache prefix_cache;
	SparkGlm52PrefixCacheEntry prefix_entries[PIPESIM_KV_BLOCKS];
	SparkGlm52PrefixCacheSequenceBinding prefix_bindings[PIPESIM_PREFIX_BINDINGS];
	SparkGlm52Scheduler scheduler;
	SparkGlm52RequestApiSlot request_slots[PIPESIM_REQUEST_SLOTS];
	SparkGlm52RequestApi request_api;
	SparkGlm52ServingEngine serving_engine;
	SparkGlm52ServingRequestRecord request_records[PIPESIM_REQUEST_SLOTS];
	uint32_t request_token_storage[
		PIPESIM_REQUEST_SLOTS * PIPESIM_REQUEST_TOKEN_STRIDE];
	SparkGlm52ServingEvent event_ring[PIPESIM_EVENT_CAPACITY];
	uint32_t host_prefill_token_ids[PIPESIM_LANE_CAPACITY * PIPESIM_PREFILL_STRIDE];
	uint32_t physical_block_indices[PIPESIM_LANE_CAPACITY * SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
	uint32_t lane_physical_block_counts[PIPESIM_LANE_CAPACITY];
	PipesimPending pendings[PIPESIM_PENDING_CAPACITY];
	PipesimRing ring;
	PipesimStats stats;
	uint64_t now_ns;
	uint64_t decoded_token_count;
	uint64_t prefill_completed_token_count;
	uint64_t prefill_last_completion_ns;
	uint64_t prefill_dispatch_count;
	uint64_t verify_dispatch_count;
	uint64_t dspark_verify_dispatch_count;
	uint64_t peak_resident_kv_blocks;
	SparkGlm52DsparkSpeculator dspark_speculator;
	SparkGlm52DsparkSequenceState dspark_sequence_states[PIPESIM_REQUEST_SLOTS];
	SparkGlm52DsparkModelContract dspark_model_contract;
	uint32_t dspark_enabled;
	uint32_t speculation_mode;
	uint64_t producer_dispatch_count;
	uint64_t committed_token_estimate;
	uint64_t accept_accum_milli;
	uint64_t dspark_cold_accum_milli;
	uint32_t accept_milli;
	uint32_t pending_count;
	uint32_t queue_depth;
} PipesimFixture;

static PipesimFixture Pipesim;

static SparkStatus PipesimKvPrefetch(void *context, const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
	(void)context;
	(void)prefetch_plan;
	return SPARK_STATUS_OK;
}

static SparkStatus PipesimReleaseSequence(void *context, uint64_t request_id, uint64_t sequence_id, uint32_t token_count)
{
	(void)context;
	(void)request_id;
	(void)sequence_id;
	(void)token_count;
	return SPARK_STATUS_OK;
}

static uint64_t PipesimRingTraverse(PipesimRing *ring, uint64_t enter_ns, uint64_t stage_ns)
{
	uint64_t stage_index, ready_ns;
	ready_ns = enter_ns;
	for (stage_index = 0u; stage_index < PIPESIM_SPARK_COUNT; ++stage_index)
	{
		if (ring->stage_free_ns[stage_index] > ready_ns)
			ready_ns = ring->stage_free_ns[stage_index];
		ready_ns += stage_ns;
		ring->stage_free_ns[stage_index] = ready_ns;
		ready_ns += ring->hop_ns;
	}
	return ready_ns;
}

static SparkStatus PipesimDsparkDraft(void *context, const SparkGlm52DsparkDraftRequest *request, SparkGlm52DsparkDraftResult *result)
{
	uint32_t token_index;

	(void)context;
	result->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
	result->descriptor_bytes = SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
	result->flags = 0u;
	result->token_count = request->requested_token_count;
	if (result->token_count > SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
		result->token_count = SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
	{
		PipesimFixture *fixture;
		uint32_t confidence_milli;
		const char *cold_env;

		fixture = (PipesimFixture *)context;
		confidence_milli = 900u;
		cold_env = getenv("SPARKPIPE_SIM_DSPARK_COLD_MILLI");
		if (cold_env != 0)
		{
			fixture->dspark_cold_accum_milli += (uint64_t)strtoul(cold_env, 0, 10);
			if (fixture->dspark_cold_accum_milli >= 1000u)
			{
				fixture->dspark_cold_accum_milli -= 1000u;
				confidence_milli = 100u;
			}
		}
		for (token_index = 0u; token_index < result->token_count; ++token_index)
		{
			result->token_ids[token_index] = 8001u + token_index;
			result->confidence_milli[token_index] = confidence_milli;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus PipesimPrefill(void *context, const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	PipesimFixture *fixture;
	PipesimPending *pending;
	uint32_t pending_index;
	fixture = (PipesimFixture *)context;
	pending = 0;
	for (pending_index = 0u; pending_index < PIPESIM_PENDING_CAPACITY; ++pending_index)
		if (fixture->pendings[pending_index].active == 0u)
		{
			pending = &fixture->pendings[pending_index];
			break;
		}
	if (pending == 0)
		return SPARK_STATUS_BUSY;
	pending->active = 1u;
	pending->is_prefill = 1u;
	pending->dispatch = *prefill_dispatch->request_dispatch;
	pending->completion_ns = PipesimRingTraverse(&fixture->ring, fixture->now_ns, fixture->ring.prefill_stage_ns);
	fixture->pending_count += 1u;
	if (fixture->pending_count > fixture->stats.max_concurrent)
		fixture->stats.max_concurrent = fixture->pending_count;
	fixture->prefill_dispatch_count += 1u;
	return SPARK_STATUS_PENDING;
}


static SparkStatus PipesimDecode(void *context, const SparkGlm52ServingDecodeDispatch *decode_dispatch, SparkGlm52ServingDecodeResult *decode_result)
{
	PipesimFixture *fixture;
	PipesimPending *pending;
	uint32_t pending_index;
	fixture = (PipesimFixture *)context;
	(void)decode_result;
	pending = 0;
	for (pending_index = 0u; pending_index < PIPESIM_PENDING_CAPACITY; ++pending_index)
		if (fixture->pendings[pending_index].active == 0u)
		{
			pending = &fixture->pendings[pending_index];
			break;
		}
	if (pending == 0)
		return SPARK_STATUS_BUSY;
	pending->active = 1u;
	pending->dispatch = *decode_dispatch->request_dispatch;
	pending->is_prefill = 0u;
	if ((decode_dispatch->request_dispatch->flags &
			(SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
			 SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY)) != 0u)
	{
		fixture->verify_dispatch_count += 1u;
		if ((decode_dispatch->request_dispatch->flags &
				SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u)
			fixture->dspark_verify_dispatch_count += 1u;
		pending->completion_ns = PipesimRingTraverse(&fixture->ring, fixture->now_ns, fixture->ring.verify_stage_ns);
	}
	else
	{
		fixture->producer_dispatch_count += 1u;
		pending->completion_ns = PipesimRingTraverse(&fixture->ring, fixture->now_ns, fixture->ring.stage_ns);
	}
	fixture->pending_count += 1u;
	if (fixture->pending_count > fixture->stats.max_concurrent)
		fixture->stats.max_concurrent = fixture->pending_count;
	fixture->stats.dispatch_count += 1u;
	fixture->stats.lane_dispatch_count += decode_dispatch->active_sequence_count;
	fixture->stats.width_histogram[decode_dispatch->active_sequence_count <= PIPESIM_LANE_CAPACITY ? decode_dispatch->active_sequence_count : PIPESIM_LANE_CAPACITY] += 1u;
	return SPARK_STATUS_PENDING;
}

static PipesimPending *PipesimEarliestPending(PipesimFixture *fixture)
{
	PipesimPending *earliest;
	uint32_t pending_index;
	earliest = 0;
	for (pending_index = 0u; pending_index < PIPESIM_PENDING_CAPACITY; ++pending_index)
	{
		PipesimPending *pending;
		pending = &fixture->pendings[pending_index];
		if (pending->active == 0u)
			continue;
		if (earliest == 0 || pending->completion_ns < earliest->completion_ns)
			earliest = pending;
	}
	return earliest;
}

static void PipesimDeliverCompletion(PipesimFixture *fixture, PipesimPending *pending)
{
	SparkGlm52ServingDecodeResult decode_result;
	uint32_t lane_index;
	SparkStatus status;
	if (pending->completion_ns > fixture->now_ns)
	{
		fixture->stats.concurrency_weighted_ns += (pending->completion_ns - fixture->now_ns) * fixture->pending_count;
		fixture->stats.observed_ns += pending->completion_ns - fixture->now_ns;
		fixture->now_ns = pending->completion_ns;
	}
	SparkGlm52ServingInitializeDecodeResult(&decode_result, pending->dispatch.request_count, PIPESIM_TOKEN_STRIDE);
	if ((pending->dispatch.flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u)
	{
		uint32_t accepted, row_index, dspark_draft_count, verifier_row_count;
		verifier_row_count = pending->dispatch.speculative_verifier_token_count;
		if (verifier_row_count == 0u || verifier_row_count > SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE)
			verifier_row_count = 1u;
		dspark_draft_count = verifier_row_count - 1u;
		fixture->accept_accum_milli += (uint64_t)fixture->accept_milli * dspark_draft_count;
		accepted = (uint32_t)(fixture->accept_accum_milli / 1000u);
		fixture->accept_accum_milli %= 1000u;
		if (accepted > dspark_draft_count)
			accepted = dspark_draft_count;
		for (lane_index = 0u; lane_index < pending->dispatch.request_count; ++lane_index)
		{
			decode_result.token_counts[lane_index] = verifier_row_count;
			for (row_index = 0u; row_index < verifier_row_count; ++row_index)
				decode_result.token_ids[lane_index][row_index] =
					row_index < accepted
						? pending->dispatch.speculative_draft_token_ids[lane_index][row_index]
						: 9000u + row_index;
			fixture->committed_token_estimate += accepted + 1u;
		}
	}
	else if ((pending->dispatch.flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
	{
		uint32_t depth, row_index;
		fixture->accept_accum_milli += (uint64_t)fixture->accept_milli *
			SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT;
		depth = (uint32_t)(fixture->accept_accum_milli / 1000u);
		fixture->accept_accum_milli %= 1000u;
		if (depth > SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT)
			depth = SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT;
		for (lane_index = 0u; lane_index < pending->dispatch.request_count; ++lane_index)
		{
			decode_result.token_counts[lane_index] =
				SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
			for (row_index = 0u;
				 row_index < SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
				 ++row_index)
				decode_result.token_ids[lane_index][row_index] = 4242u + row_index;
			if (depth >= 1u)
				decode_result.token_ids[lane_index][
					SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW] =
					7001u + SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX;
			if (depth >= 2u)
				decode_result.token_ids[lane_index][
					SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW] =
					7001u + SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX;
			if (depth >= 3u)
				decode_result.token_ids[lane_index][
					SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW] =
					7001u + SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX;
			fixture->committed_token_estimate += depth + 1u;
		}
	}
	else
	{
		for (lane_index = 0u; lane_index < pending->dispatch.request_count; ++lane_index)
		{
			decode_result.token_counts[lane_index] = 1u;
			decode_result.token_ids[lane_index][0u] = 4242u;
			fixture->committed_token_estimate += 1u;
		}
	}
	if (fixture->accept_milli != 0u &&
		(pending->dispatch.flags &
			(SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT |
			 SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY)) != 0u)
	{
		uint32_t draft_index;
		for (lane_index = 0u; lane_index < pending->dispatch.request_count; ++lane_index)
		{
			decode_result.draft_token_counts[lane_index] =
				SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
			for (draft_index = 0u;
				 draft_index < SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
				 ++draft_index)
				decode_result.draft_token_ids[lane_index][draft_index] =
					7001u + draft_index;
		}
	}
	pending->active = 0u;
	fixture->pending_count -= 1u;
	if (pending->is_prefill != 0u)
	{
		if (pending->dispatch.kind ==
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
		{
			uint32_t batch_lane_index;
			for (batch_lane_index = 0u;
				 batch_lane_index < pending->dispatch.request_count;
				 ++batch_lane_index)
				fixture->prefill_completed_token_count +=
					pending->dispatch.prefill_batch_decision.lanes[
						batch_lane_index].scheduled_prompt_token_count;
		}
		else
			fixture->prefill_completed_token_count +=
				pending->dispatch.prefill_decision.scheduled_prompt_token_count;
		fixture->prefill_last_completion_ns = fixture->now_ns;
		status = SparkGlm52ServingEngineCompletePrefillDispatch(&fixture->serving_engine, &pending->dispatch);
		if (status != SPARK_STATUS_OK)
		{
			fprintf(stderr, "pipesim complete_prefill status=%u kind=%u count=%u\n", (uint32_t)status, pending->dispatch.kind, pending->dispatch.request_count);
			exit(6);
		}
		return;
	}
	status = SparkGlm52ServingEngineCompleteDecodeDispatch(&fixture->serving_engine, &pending->dispatch, &decode_result);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr, "pipesim complete_decode status=%u kind=%u flags=0x%x budget=%u vtc=%u stc=%u acc=%u draft0=%u\n", (uint32_t)status, pending->dispatch.kind, pending->dispatch.flags, pending->dispatch.mtp_draft_token_budget, pending->dispatch.speculative_verifier_token_count, pending->dispatch.speculative_token_count, pending->dispatch.accepted, pending->dispatch.speculative_draft_token_ids[0][0]);
		exit(3);
	}
	fixture->decoded_token_count += pending->dispatch.request_count;
	if (fixture->stats.steady_begin_ns == 0u && fixture->decoded_token_count >= PIPESIM_LANE_CAPACITY)
	{
		fixture->stats.steady_begin_ns = fixture->now_ns;
		fixture->stats.steady_begin_tokens = fixture->decoded_token_count;
	}
}

static void PipesimInitializeCore(PipesimFixture *fixture)
{
	SparkGlm52KvCacheConfiguration kv_configuration;
	SparkGlm52PrefixCacheConfiguration prefix_configuration;
	SparkGlm52SchedulerConfiguration scheduler_configuration;
	memset(&kv_configuration, 0, sizeof(kv_configuration));
	kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	kv_configuration.descriptor_bytes = SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	kv_configuration.physical_block_count = PIPESIM_KV_BLOCKS;
	kv_configuration.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	kv_configuration.layer_count = 78u;
	kv_configuration.kv_head_count = 8u;
	kv_configuration.head_dim = 128u;
	kv_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
	kv_configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
	kv_configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
	kv_configuration.blocks = fixture->kv_blocks;
	if (SparkGlm52KvCacheArenaInitialize(&fixture->kv_arena, &kv_configuration) != SPARK_STATUS_OK)
		exit(10);
	memset(&prefix_configuration, 0, sizeof(prefix_configuration));
	prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
	prefix_configuration.descriptor_bytes = SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	prefix_configuration.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	prefix_configuration.entry_count = PIPESIM_KV_BLOCKS;
	prefix_configuration.physical_block_count = PIPESIM_KV_BLOCKS;
	prefix_configuration.sequence_binding_count = PIPESIM_PREFIX_BINDINGS;
	prefix_configuration.entries = fixture->prefix_entries;
	prefix_configuration.sequence_bindings = fixture->prefix_bindings;
	prefix_configuration.kv_cache_arena = &fixture->kv_arena;
	if (SparkGlm52PrefixCacheInitialize(&fixture->prefix_cache, &prefix_configuration) != SPARK_STATUS_OK)
		exit(11);
	memset(&scheduler_configuration, 0, sizeof(scheduler_configuration));
	scheduler_configuration.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
	scheduler_configuration.descriptor_bytes = SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
	scheduler_configuration.spark_count = PIPESIM_SPARK_COUNT;
	scheduler_configuration.queue_depth_per_spark = fixture->queue_depth;
	scheduler_configuration.measured_profile_id = SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
	scheduler_configuration.quantization_mode = SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
	scheduler_configuration.max_prefill_tokens_per_step = PIPESIM_PREFILL_STRIDE;
	scheduler_configuration.configuration_flags = SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
	scheduler_configuration.prefix_cache_block_tokens = SPARK_GLM52_KV_BLOCK_TOKENS;
	scheduler_configuration.prefix_cache = &fixture->prefix_cache;
	if (SparkGlm52SchedulerInitialize(&fixture->scheduler, &scheduler_configuration) != SPARK_STATUS_OK)
		exit(12);
}

static void PipesimInitializeServing(PipesimFixture *fixture)
{
	SparkGlm52RequestApiConfiguration request_api_configuration;
	SparkGlm52ServingEngineConfiguration serving_configuration;
	memset(&request_api_configuration, 0, sizeof(request_api_configuration));
	request_api_configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	request_api_configuration.descriptor_bytes = SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
	request_api_configuration.configuration_flags =
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION |
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING |
		(fixture->accept_milli != 0u && fixture->speculation_mode != 1u
			? SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
			  SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE : 0u) |
		(fixture->dspark_enabled != 0u
			? SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE : 0u) |
		(fixture->dspark_enabled != 0u && fixture->speculation_mode == 2u
			? SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION : 0u);
	request_api_configuration.request_capacity = PIPESIM_REQUEST_SLOTS;
	request_api_configuration.prefetch_lane_count = SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT;
	if (fixture->dspark_enabled != 0u)
	{
		SparkGlm52DsparkSpeculatorConfiguration dspark_configuration;

		memset(&dspark_configuration, 0, sizeof(dspark_configuration));
		dspark_configuration.abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
		dspark_configuration.descriptor_bytes = SPARK_GLM52_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES;
		dspark_configuration.policy_flags = SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_REALTIME | SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE;
		dspark_configuration.sequence_state_count = PIPESIM_REQUEST_SLOTS;
		dspark_configuration.default_speculative_token_count = SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
		dspark_configuration.minimum_confidence_milli = 500u;
		dspark_configuration.realtime_minimum_confidence_milli = 500u;
		dspark_configuration.sequence_states = fixture->dspark_sequence_states;
		dspark_configuration.draft_function = PipesimDsparkDraft;
		dspark_configuration.draft_context = fixture;
		SparkGlm52DsparkBuildDefaultModelContract(&fixture->dspark_model_contract);
		dspark_configuration.model_contract = &fixture->dspark_model_contract;
		if (SparkGlm52DsparkInitialize(&fixture->dspark_speculator, &dspark_configuration) != SPARK_STATUS_OK)
		{
			fprintf(stderr, "pipesim dspark init failed\n");
			exit(4);
		}
		request_api_configuration.dspark_speculator = &fixture->dspark_speculator;
	}
	request_api_configuration.decode_batch_target = PIPESIM_LANE_CAPACITY;
	request_api_configuration.decode_execution_row_capacity = 1024u;
	request_api_configuration.scheduler = &fixture->scheduler;
	request_api_configuration.request_slots = fixture->request_slots;
	request_api_configuration.kv_prefetch_function = PipesimKvPrefetch;
	if (SparkGlm52RequestApiInitialize(&fixture->request_api, &request_api_configuration) != SPARK_STATUS_OK)
		exit(13);
	memset(&serving_configuration, 0, sizeof(serving_configuration));
	serving_configuration.abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
	serving_configuration.descriptor_bytes = SPARK_GLM52_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES;
	serving_configuration.flags =
		SPARK_GLM52_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS |
		SPARK_GLM52_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT;
	serving_configuration.runtime_contract_flags = SPARK_GLM52_SERVING_RUNTIME_CONTRACT_CURRENT_IMPLEMENTED_FLAGS;
	serving_configuration.default_output_token_budget = 16u;
	serving_configuration.default_max_prefill_tokens_per_step = PIPESIM_PREFILL_STRIDE;
	serving_configuration.max_context_tokens = PIPESIM_REQUEST_TOKEN_STRIDE;
	serving_configuration.request_api = &fixture->request_api;
	serving_configuration.request_records = fixture->request_records;
	serving_configuration.request_record_capacity = PIPESIM_REQUEST_SLOTS;
	serving_configuration.request_token_storage = fixture->request_token_storage;
	serving_configuration.request_token_stride = PIPESIM_REQUEST_TOKEN_STRIDE;
	serving_configuration.event_ring = fixture->event_ring;
	serving_configuration.event_ring_capacity = PIPESIM_EVENT_CAPACITY;
	serving_configuration.host_prefill_token_ids = fixture->host_prefill_token_ids;
	serving_configuration.host_prefill_token_stride = PIPESIM_PREFILL_STRIDE;
	serving_configuration.host_prefill_lane_capacity = PIPESIM_LANE_CAPACITY;
	serving_configuration.host_physical_block_indices = fixture->physical_block_indices;
	serving_configuration.kv_block_lane_stride = SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
	serving_configuration.kv_block_lane_capacity = SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
	serving_configuration.lane_physical_block_counts = fixture->lane_physical_block_counts;
	serving_configuration.lane_count_capacity = PIPESIM_LANE_CAPACITY;
	serving_configuration.prefill_function = PipesimPrefill;
	serving_configuration.decode_function = PipesimDecode;
	serving_configuration.release_sequence_function = PipesimReleaseSequence;
	serving_configuration.callback_context = fixture;
	if (SparkGlm52ServingEngineInitialize(&fixture->serving_engine, &serving_configuration) != SPARK_STATUS_OK)
		exit(14);
}

static void PipesimSubmitRequests(PipesimFixture *fixture, uint32_t request_count, uint32_t output_tokens, uint32_t prompt_tokens)
{
	static uint32_t PromptTokens[8192u];
	SparkGlm52ServingSubmitTokenIdsRequest submit_request;
	SparkGlm52ServingSubmitResult submit_result;
	uint32_t request_index, token_index;
	for (token_index = 0u; token_index < prompt_tokens; ++token_index)
		PromptTokens[token_index] = 5000u + token_index;
	for (request_index = 0u; request_index < request_count; ++request_index)
	{
		SparkGlm52ServingInitializeSubmitTokenIdsRequest(&submit_request);
		submit_request.request_id = 1000u + request_index;
		submit_request.token_count = prompt_tokens;
		submit_request.token_ids = PromptTokens;
		submit_request.output_token_budget = output_tokens;
		memset(&submit_result, 0, sizeof(submit_result));
		SparkStatus submit_status;
		submit_status = SparkGlm52ServingEngineSubmitTokenIds(&fixture->serving_engine, &submit_request, &submit_result);
		if (submit_status != SPARK_STATUS_OK)
		{
			fprintf(stderr, "pipesim submit %u failed status=%u\n", request_index, (uint32_t)submit_status);
			exit(15);
		}
	}
}

static void PipesimDrainEvents(PipesimFixture *fixture)
{
	SparkGlm52ServingEvent event;
	while (SparkGlm52ServingEnginePopEvent(&fixture->serving_engine, &event) == SPARK_STATUS_OK)
	{
	}
}

int main(int argc, char **argv)
{
	PipesimFixture *fixture;
	PipesimPending *pending;
	uint64_t stage_us, hop_us, prefill_stage_us, verify_stage_us, steady_tokens, steady_ns, iteration;
	uint32_t request_count, output_tokens, prompt_tokens, queue_depth, accept_milli, speculation_mode, width_index;
	SparkStatus status;
	fixture = &Pipesim;
	stage_us = argc > 1 ? strtoull(argv[1], 0, 10) : 16000u;
	hop_us = argc > 2 ? strtoull(argv[2], 0, 10) : 100u;
	request_count = argc > 3 ? (uint32_t)strtoul(argv[3], 0, 10) : 64u;
	output_tokens = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 10) : 16u;
	prompt_tokens = argc > 5 ? (uint32_t)strtoul(argv[5], 0, 10) : 9u;
	prefill_stage_us = argc > 6 ? strtoull(argv[6], 0, 10) : stage_us;
	queue_depth = argc > 7 ? (uint32_t)strtoul(argv[7], 0, 10) : PIPESIM_QUEUE_DEPTH_PER_SPARK;
	accept_milli = argc > 8 ? (uint32_t)strtoul(argv[8], 0, 10) : 0u;
	verify_stage_us = argc > 9 ? strtoull(argv[9], 0, 10) : stage_us;
	speculation_mode = argc > 10 ? (uint32_t)strtoul(argv[10], 0, 10) : 0u;
	if (request_count == 0u || request_count > PIPESIM_REQUEST_SLOTS ||
		output_tokens == 0u || prompt_tokens == 0u || prompt_tokens > 8192u ||
		output_tokens + prompt_tokens > PIPESIM_REQUEST_TOKEN_STRIDE ||
		queue_depth == 0u || queue_depth > 16u)
	{
		fprintf(stderr,"pipesim invalid requests=%u output=%u prompt=%u depth=%u\n",
			request_count,output_tokens,prompt_tokens,queue_depth);
		return 2;
	}
	memset(fixture, 0, sizeof(*fixture));
	fixture->ring.stage_ns = stage_us * 1000u;
	fixture->ring.prefill_stage_ns = prefill_stage_us * 1000u;
	fixture->ring.verify_stage_ns = verify_stage_us * 1000u;
	fixture->ring.hop_ns = hop_us * 1000u;
	fixture->queue_depth = queue_depth;
	fixture->accept_milli = accept_milli > 1000u ? 1000u : accept_milli;
	fixture->speculation_mode = speculation_mode;
	fixture->dspark_enabled = (speculation_mode >= 1u && fixture->accept_milli != 0u) ? 1u : 0u;
	PipesimInitializeCore(fixture);
	PipesimInitializeServing(fixture);
	PipesimSubmitRequests(fixture, request_count, output_tokens, prompt_tokens);
	for (iteration = 0u; iteration < 2000000u; ++iteration)
	{
		status = SparkGlm52ServingEnginePump(&fixture->serving_engine, 0u, 64u, 0);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY &&
			status != SPARK_STATUS_PENDING && status != SPARK_STATUS_NOT_FOUND)
		{
			fprintf(stderr, "pipesim pump status=%u iter=%" PRIu64 "\n", (uint32_t)status, iteration);
			return 4;
		}
		PipesimDrainEvents(fixture);
		if (fixture->request_api.completed_request_count >= request_count)
			break;
		pending = PipesimEarliestPending(fixture);
		if (pending == 0)
		{
			if (status == SPARK_STATUS_NOT_FOUND || status == SPARK_STATUS_BUSY)
			{
				fprintf(stderr, "pipesim stalled tokens=%" PRIu64 " status=%u\n", fixture->decoded_token_count, (uint32_t)status);
				return 5;
			}
			continue;
		}
		PipesimDeliverCompletion(fixture, pending);
		{
			uint64_t resident_now;
			uint32_t kv_block_index;

			resident_now = 0u;
			for (kv_block_index = 0u; kv_block_index < PIPESIM_KV_BLOCKS; ++kv_block_index)
				if (fixture->kv_blocks[kv_block_index].reference_count != 0u)
					resident_now += 1u;
			if (resident_now > fixture->peak_resident_kv_blocks)
				fixture->peak_resident_kv_blocks = resident_now;
		}
		if (getenv("SPARKPIPE_SIM_TRACE"))
			fprintf(stderr, "T %llu kind=%u lanes=%u queued=%u running=%u pending=%u\n",
				(unsigned long long)(fixture->now_ns / 1000000u),
				pending->dispatch.kind, pending->dispatch.request_count,
				fixture->request_api.queued_request_count,
				fixture->request_api.running_request_count,
				fixture->pending_count);
	}
	steady_tokens = fixture->decoded_token_count - fixture->stats.steady_begin_tokens;
	steady_ns = fixture->now_ns - fixture->stats.steady_begin_ns;
	printf("stage_us=%" PRIu64 " hop_us=%" PRIu64 " requests=%u output=%u prompt=%u prefill_stage_us=%" PRIu64 " depth=%u accept_milli=%u verify_stage_us=%" PRIu64 "\n", stage_us, hop_us, request_count, output_tokens, prompt_tokens, prefill_stage_us, queue_depth, accept_milli, verify_stage_us);
	if (fixture->verify_dispatch_count != 0u)
		printf("mtp producer_dispatches=%" PRIu64 " verify_dispatches=%" PRIu64 " committed_tokens=%" PRIu64 " committed_tok_per_s=%.1f traversals_per_commit_cycle=%.2f dspark_verifies=%" PRIu64 "\n", fixture->producer_dispatch_count, fixture->verify_dispatch_count, fixture->committed_token_estimate, fixture->now_ns != 0u ? fixture->committed_token_estimate * 1e9 / (double)fixture->now_ns : 0.0, (double)(fixture->producer_dispatch_count + fixture->verify_dispatch_count) / (double)fixture->verify_dispatch_count, fixture->dspark_verify_dispatch_count);
	if (fixture->prefill_completed_token_count != 0u)
		printf("prefill_tokens=%" PRIu64 " prefill_dispatches=%" PRIu64 " prefill_done_ms=%" PRIu64 " prefill_tok_per_s=%.1f\n", fixture->prefill_completed_token_count, fixture->prefill_dispatch_count, fixture->prefill_last_completion_ns / 1000000u, fixture->prefill_last_completion_ns != 0u ? fixture->prefill_completed_token_count * 1e9 / (double)fixture->prefill_last_completion_ns : 0.0);
	printf("tokens=%" PRIu64 " virtual_ms=%" PRIu64 " tok_per_s=%.1f\n", fixture->decoded_token_count, fixture->now_ns / 1000000u, fixture->decoded_token_count * 1e9 / (double)fixture->now_ns);
	printf("steady_tok_per_s=%.1f steady_tokens=%" PRIu64 "\n", steady_ns != 0u ? steady_tokens * 1e9 / (double)steady_ns : 0.0, steady_tokens);
	printf("peak_resident_kv_blocks=%" PRIu64 " of %u (%.1f%%)\n", fixture->peak_resident_kv_blocks, PIPESIM_KV_BLOCKS, fixture->peak_resident_kv_blocks * 100.0 / PIPESIM_KV_BLOCKS);
	printf("dispatches=%" PRIu64 " mean_width=%.2f max_concurrent=%u mean_concurrent=%.2f\n", fixture->stats.dispatch_count, fixture->stats.dispatch_count != 0u ? (double)fixture->stats.lane_dispatch_count / (double)fixture->stats.dispatch_count : 0.0, fixture->stats.max_concurrent, fixture->stats.observed_ns != 0u ? (double)fixture->stats.concurrency_weighted_ns / (double)fixture->stats.observed_ns : 0.0);
	for (width_index = 0u; width_index <= PIPESIM_LANE_CAPACITY; ++width_index)
		if (fixture->stats.width_histogram[width_index] != 0u)
			printf("width[%u]=%" PRIu64 "\n", width_index, fixture->stats.width_histogram[width_index]);
	return 0;
}
