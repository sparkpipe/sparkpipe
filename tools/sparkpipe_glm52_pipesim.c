#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_glm52_serving_engine.h"

#define PIPESIM_SPARK_COUNT SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define PIPESIM_LANE_CAPACITY 64u
#define PIPESIM_REQUEST_SLOTS 256u
#define PIPESIM_KV_BLOCKS 256u
#define PIPESIM_PREFIX_BINDINGS (PIPESIM_KV_BLOCKS + 64u)
#define PIPESIM_EVENT_CAPACITY 16384u
#define PIPESIM_PREFILL_STRIDE 256u
#define PIPESIM_PENDING_CAPACITY 64u
#define PIPESIM_QUEUE_DEPTH_PER_SPARK 14u
#define PIPESIM_TOKEN_STRIDE SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE

typedef struct PipesimPending
{
	uint32_t active;
	uint64_t completion_ns;
	SparkGlm52RequestApiDispatch dispatch;
} PipesimPending;

typedef struct PipesimRing
{
	uint64_t stage_free_ns[PIPESIM_SPARK_COUNT];
	uint64_t stage_ns;
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
	uint32_t request_token_storage[PIPESIM_REQUEST_SLOTS * 64u];
	SparkGlm52ServingEvent event_ring[PIPESIM_EVENT_CAPACITY];
	uint32_t host_prefill_token_ids[PIPESIM_LANE_CAPACITY * PIPESIM_PREFILL_STRIDE];
	uint32_t physical_block_indices[PIPESIM_LANE_CAPACITY * SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
	uint32_t lane_physical_block_counts[PIPESIM_LANE_CAPACITY];
	PipesimPending pendings[PIPESIM_PENDING_CAPACITY];
	PipesimRing ring;
	PipesimStats stats;
	uint64_t now_ns;
	uint64_t decoded_token_count;
	uint32_t pending_count;
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

static SparkStatus PipesimPrefill(void *context, const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	(void)context;
	(void)prefill_dispatch;
	return SPARK_STATUS_OK;
}

static uint64_t PipesimRingTraverse(PipesimRing *ring, uint64_t enter_ns)
{
	uint64_t stage_index, ready_ns;
	ready_ns = enter_ns;
	for (stage_index = 0u; stage_index < PIPESIM_SPARK_COUNT; ++stage_index)
	{
		if (ring->stage_free_ns[stage_index] > ready_ns)
			ready_ns = ring->stage_free_ns[stage_index];
		ready_ns += ring->stage_ns;
		ring->stage_free_ns[stage_index] = ready_ns;
		ready_ns += ring->hop_ns;
	}
	return ready_ns;
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
	pending->completion_ns = PipesimRingTraverse(&fixture->ring, fixture->now_ns);
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
	for (lane_index = 0u; lane_index < pending->dispatch.request_count; ++lane_index)
	{
		decode_result.token_counts[lane_index] = 1u;
		decode_result.token_ids[lane_index][0u] = 4242u;
	}
	pending->active = 0u;
	fixture->pending_count -= 1u;
	status = SparkGlm52ServingEngineCompleteDecodeDispatch(&fixture->serving_engine, &pending->dispatch, &decode_result);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr, "pipesim complete_decode status=%u\n", (uint32_t)status);
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
	scheduler_configuration.queue_depth_per_spark = PIPESIM_QUEUE_DEPTH_PER_SPARK;
	scheduler_configuration.measured_profile_id = SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
	scheduler_configuration.quantization_mode = SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
	scheduler_configuration.max_prefill_tokens_per_step = SPARK_GLM52_KV_BLOCK_TOKENS;
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
		SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
	request_api_configuration.request_capacity = PIPESIM_REQUEST_SLOTS;
	request_api_configuration.prefetch_lane_count = SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT;
	request_api_configuration.decode_batch_target = PIPESIM_LANE_CAPACITY;
	request_api_configuration.decode_execution_row_capacity = SparkGlm52Pp13RuntimeExecutionRowCapacity(PIPESIM_LANE_CAPACITY);
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
	serving_configuration.default_max_prefill_tokens_per_step = SPARK_GLM52_KV_BLOCK_TOKENS;
	serving_configuration.max_context_tokens = SPARK_GLM52_KV_BLOCK_TOKENS * 4u;
	serving_configuration.request_api = &fixture->request_api;
	serving_configuration.request_records = fixture->request_records;
	serving_configuration.request_record_capacity = PIPESIM_REQUEST_SLOTS;
	serving_configuration.request_token_storage = fixture->request_token_storage;
	serving_configuration.request_token_stride = 64u;
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

static void PipesimSubmitRequests(PipesimFixture *fixture, uint32_t request_count, uint32_t output_tokens)
{
	static uint32_t PromptTokens[9];
	SparkGlm52ServingSubmitTokenIdsRequest submit_request;
	SparkGlm52ServingSubmitResult submit_result;
	uint32_t request_index, token_index;
	for (token_index = 0u; token_index < 9u; ++token_index)
		PromptTokens[token_index] = 5000u + token_index;
	for (request_index = 0u; request_index < request_count; ++request_index)
	{
		SparkGlm52ServingInitializeSubmitTokenIdsRequest(&submit_request);
		submit_request.request_id = 1000u + request_index;
		submit_request.token_count = 9u;
		submit_request.token_ids = PromptTokens;
		submit_request.output_token_budget = output_tokens;
		memset(&submit_result, 0, sizeof(submit_result));
		if (SparkGlm52ServingEngineSubmitTokenIds(&fixture->serving_engine, &submit_request, &submit_result) != SPARK_STATUS_OK)
		{
			fprintf(stderr, "pipesim submit %u failed\n", request_index);
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
	uint64_t stage_us, hop_us, steady_tokens, steady_ns, iteration;
	uint32_t request_count, output_tokens, width_index;
	SparkStatus status;
	fixture = &Pipesim;
	stage_us = argc > 1 ? strtoull(argv[1], 0, 10) : 16000u;
	hop_us = argc > 2 ? strtoull(argv[2], 0, 10) : 100u;
	request_count = argc > 3 ? (uint32_t)strtoul(argv[3], 0, 10) : 64u;
	output_tokens = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 10) : 16u;
	memset(fixture, 0, sizeof(*fixture));
	fixture->ring.stage_ns = stage_us * 1000u;
	fixture->ring.hop_ns = hop_us * 1000u;
	PipesimInitializeCore(fixture);
	PipesimInitializeServing(fixture);
	PipesimSubmitRequests(fixture, request_count, output_tokens);
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
		if (fixture->decoded_token_count >= (uint64_t)request_count * output_tokens)
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
	}
	steady_tokens = fixture->decoded_token_count - fixture->stats.steady_begin_tokens;
	steady_ns = fixture->now_ns - fixture->stats.steady_begin_ns;
	printf("stage_us=%" PRIu64 " hop_us=%" PRIu64 " requests=%u output=%u\n", stage_us, hop_us, request_count, output_tokens);
	printf("tokens=%" PRIu64 " virtual_ms=%" PRIu64 " tok_per_s=%.1f\n", fixture->decoded_token_count, fixture->now_ns / 1000000u, fixture->decoded_token_count * 1e9 / (double)fixture->now_ns);
	printf("steady_tok_per_s=%.1f steady_tokens=%" PRIu64 "\n", steady_ns != 0u ? steady_tokens * 1e9 / (double)steady_ns : 0.0, steady_tokens);
	printf("dispatches=%" PRIu64 " mean_width=%.2f max_concurrent=%u mean_concurrent=%.2f\n", fixture->stats.dispatch_count, fixture->stats.dispatch_count != 0u ? (double)fixture->stats.lane_dispatch_count / (double)fixture->stats.dispatch_count : 0.0, fixture->stats.max_concurrent, fixture->stats.observed_ns != 0u ? (double)fixture->stats.concurrency_weighted_ns / (double)fixture->stats.observed_ns : 0.0);
	for (width_index = 0u; width_index <= PIPESIM_LANE_CAPACITY; ++width_index)
		if (fixture->stats.width_histogram[width_index] != 0u)
			printf("width[%u]=%" PRIu64 "\n", width_index, fixture->stats.width_histogram[width_index]);
	return 0;
}
