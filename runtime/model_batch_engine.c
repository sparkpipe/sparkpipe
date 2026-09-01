#include "sparkpipe/spark_model_batch_engine.h"

#include <stdlib.h>
#include <string.h>

#include "model_batch_scheduler.h"
#include "sparkpipe/spark_prefix_cache.h"
#include "sparkpipe/spark_sha256.h"

#define SPARK_MODEL_BATCH_REQUEST_FREE 0u
#define SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL 1u
#define SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT 2u
#define SPARK_MODEL_BATCH_REQUEST_READY_DECODE 3u
#define SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT 4u
#define SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE 5u
#define SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT 6u
#define SPARK_MODEL_BATCH_REQUEST_COMPLETING 7u
#define SPARK_MODEL_BATCH_NO_SLOT UINT32_MAX
#define SPARK_MODEL_BATCH_SELECT_AGED 1u
#define SPARK_MODEL_BATCH_SELECT_PRIORITY 2u
#define SPARK_MODEL_BATCH_SELECT_FILL 3u

typedef struct SparkModelBatchRequestState
{
	uint32_t state;
	uint32_t generation;
	uint32_t next_free_slot;
	uint32_t priority;
	uint32_t prompt_token_count;
	uint32_t computed_prompt_token_count;
	uint32_t generated_token_count;
	uint32_t output_token_budget;
	uint32_t cancel_pending;
	uint32_t resident_bound;
	uint32_t resident_sequence_slot;
	uint32_t terminal_event_kind;
	uint32_t terminal_status;
	uint32_t scheduling_bypass_count;
	uint32_t cache_prefix_token_count;
	uint32_t cache_published_token_count;
	uint64_t cache_lookup_epoch;
	uint64_t request_id;
	uint64_t sequence_id;
	SparkModelServingCacheIdentity cache_prefix_identity;
	SparkModelServingCacheIdentity cache_published_identity;
	SparkSha256Context cache_published_digest_context;
	uint32_t model_extension_kind;
	uint32_t first_draft_miss_count;
	uint32_t first_draft_policy;
	SparkModelBatchRequestHandle handle;
} SparkModelBatchRequestState;

typedef struct SparkModelBatchCacheDemandEntry
{
	uint32_t epoch;
	uint32_t token_count;
	SparkModelServingCacheIdentity identity;
} SparkModelBatchCacheDemandEntry;

typedef struct SparkModelBatchCacheDemand
{
	uint32_t page_count;
} SparkModelBatchCacheDemand;

typedef struct SparkModelBatchSubmissionState
{
	uint32_t active;
	uint32_t slot_index;
	uint32_t work_kind;
	uint32_t lane_count;
	uint32_t result_received;
	uint32_t admitted;
	uint32_t result_status;
	uint32_t reserved0;
	uint64_t submission_id;
} SparkModelBatchSubmissionState;

struct SparkModelBatchEngine
{
	SparkModelPipelineClient *pipeline;
	const SparkModelServingAdapterDescriptor *adapter_descriptor;
	SparkModelBatchEventFunction event_function;
	void *event_context;
	SparkModelBatchRequestState *requests;
	SparkModelBatchSubmissionState *submissions;
	uint32_t *request_token_storage;
	uint32_t *resident_slot_next;
	uint32_t *submission_request_slots;
	uint32_t *submission_prefill_counts;
	SparkModelServingLane *scratch_lanes;
	uint32_t *scratch_token_ids;
	uint32_t *scratch_row_lane_indices;
	uint64_t *scratch_row_positions;
	uint64_t *scratch_row_sequence_ids;
	uint32_t *scratch_request_slots;
	uint32_t *scratch_prefill_counts;
	SparkModelBatchCacheDemandEntry *cache_demand_entries;
	SparkPrefixCacheEntry *prefix_cache_entries;
	SparkPrefixCacheSequenceBinding *prefix_cache_bindings;
	uint32_t *prefix_entry_hash_heads;
	uint32_t *prefix_binding_lookup_hash_heads;
	uint32_t *prefix_binding_sequence_hash_heads;
	SparkPrefixCache prefix_cache;
	uint32_t request_capacity;
	uint32_t resident_sequence_capacity;
	uint32_t max_context_tokens;
	uint32_t max_prefill_rows;
	uint32_t max_active_sequence_count;
	uint32_t scratch_row_capacity;
	uint32_t submission_capacity;
	uint32_t maximum_messages_per_rank;
	uint32_t stop_token_count;
	uint32_t stop_token_ids[SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT];
	uint32_t free_request_head;
	uint32_t free_resident_slot_head;
	uint32_t free_resident_slot_count;
	uint32_t next_request_scan;
	uint32_t admission_open;
	uint32_t live_request_count;
	uint32_t inflight_submission_count;
	uint32_t failed_status;
	uint32_t next_work_kind;
	uint32_t work_kind_bypass_counts[4];
	uint32_t cache_block_token_count;
	uint32_t prefix_cache_entry_capacity;
	uint32_t prefix_cache_binding_capacity;
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
	uint32_t cache_demand_entry_capacity;
	uint32_t cache_demand_epoch;
	uint32_t inflight_kv_page_count;
	uint32_t selected_kv_page_count;
	uint64_t cache_publication_epoch;
	uint64_t next_submission_id;
	uint64_t submitted_request_count;
	uint64_t completed_request_count;
	/* Batch-aggregate first-draft misses: the adapter extension reports
	 * one count for the whole submission, so per-request attribution is
	 * only possible for single-lane completions. */
	uint32_t batch_first_draft_miss_count;
	uint64_t cancelled_request_count;
	uint64_t emitted_token_count;
};

static uint32_t SparkModelBatchMultiplyFits(uint32_t left,uint32_t right)
{
	return(left == 0u || right <= UINT32_MAX / left ? 1u : 0u);
}

static uint32_t SparkModelBatchCacheDemandCapacity(uint32_t request_capacity)
{
	uint32_t capacity,required;
	if ( request_capacity > UINT32_MAX / 2u )
		return(0u);
	required = request_capacity * 2u;
	capacity = 1u;
	while ( capacity < required )
	{
		if ( capacity > UINT32_MAX / 2u )
			return(0u);
		capacity *= 2u;
	}
	return(capacity);
}

uint32_t SparkModelBatchSchedulerPlanCacheBoundLaneCount(
	uint32_t maximum_lane_count,
	uint32_t physical_page_capacity,
	uint32_t inflight_page_count)
{
	uint32_t available;
	if ( maximum_lane_count == 0u || physical_page_capacity == 0u )
		return(physical_page_capacity == 0u ? maximum_lane_count : 0u);
	available = inflight_page_count < physical_page_capacity ?
		physical_page_capacity - inflight_page_count : 0u;
	return(maximum_lane_count < available ? maximum_lane_count : available);
}

uint32_t SparkModelBatchSchedulerCacheDemandFits(
	uint32_t physical_page_capacity,
	uint32_t used_page_count,
	uint32_t additional_page_count)
{
	if ( physical_page_capacity == 0u )
		return(1u);
	return(used_page_count <= physical_page_capacity &&
		additional_page_count <= physical_page_capacity - used_page_count ?
		1u : 0u);
}

uint32_t SparkModelBatchSchedulerRequestFitsPageCapacity(
	uint32_t block_token_count,
	uint32_t physical_page_capacity,
	uint32_t prompt_token_count,
	uint32_t output_token_budget)
{
	uint32_t processed_token_count,required_page_count;
	if ( block_token_count == 0u || physical_page_capacity == 0u )
		return(1u);
	if ( prompt_token_count == 0u || output_token_budget == 0u ||
		output_token_budget - 1u > UINT32_MAX - prompt_token_count )
		return(0u);
	processed_token_count = prompt_token_count + output_token_budget - 1u;
	required_page_count = (processed_token_count / block_token_count) +
		(processed_token_count % block_token_count != 0u ? 1u : 0u);
	return(required_page_count <= physical_page_capacity ? 1u : 0u);
}

uint32_t SparkModelBatchSchedulerPlanMixedLaneCount(
	const uint32_t queued_by_kind[4],
	const uint32_t maximum_by_kind[4],
	const uint32_t inflight_by_kind[4],
	uint32_t selected_kind,
	uint32_t submission_capacity)
{
	uint32_t inflight,kind;
	if ( queued_by_kind == 0 || maximum_by_kind == 0 || inflight_by_kind == 0 || selected_kind < SPARK_MODEL_SERVING_WORK_KIND_PREFILL || selected_kind > SPARK_MODEL_SERVING_WORK_KIND_RELEASE || queued_by_kind[selected_kind] == 0u || maximum_by_kind[selected_kind] == 0u || submission_capacity == 0u )
		return(0u);
	inflight = 0u;
	for (kind=SPARK_MODEL_SERVING_WORK_KIND_PREFILL; kind<=SPARK_MODEL_SERVING_WORK_KIND_RELEASE; kind++)
	{
		inflight += inflight_by_kind[kind];
	}
	if ( inflight >= submission_capacity )
		return(0u);
	return(queued_by_kind[selected_kind] < maximum_by_kind[selected_kind] ?
		queued_by_kind[selected_kind] : maximum_by_kind[selected_kind]);
}

uint32_t SparkModelBatchSchedulerChooseWorkKind(
	const uint32_t queued_by_kind[4],
	const uint32_t minimum_by_kind[4],
	uint32_t admission_open,
	uint32_t inflight_submission_count,
	uint32_t bypass_limit,
	uint32_t *next_work_kind,
	uint32_t bypass_count_by_kind[4])
{
	uint32_t kind,minimum,offset,selected,start;
	if ( queued_by_kind == 0 || minimum_by_kind == 0 || next_work_kind == 0 || bypass_count_by_kind == 0 )
		return(0u);
	start = *next_work_kind;
	if ( start < SPARK_MODEL_SERVING_WORK_KIND_PREFILL || start > SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		start = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	if ( bypass_limit == 0u )
		bypass_limit = 1u;
	for (kind=SPARK_MODEL_SERVING_WORK_KIND_PREFILL; kind<=SPARK_MODEL_SERVING_WORK_KIND_RELEASE; kind++)
	{
		minimum = minimum_by_kind[kind] != 0u ? minimum_by_kind[kind] : 1u;
		if ( queued_by_kind[kind] == 0u || queued_by_kind[kind] >= minimum || admission_open == 0u || inflight_submission_count == 0u || kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			bypass_count_by_kind[kind] = 0u;
	}
	selected = 0u;
	for (offset=0u; offset<3u; offset++)
	{
		kind = ((start - 1u + offset) % 3u) + 1u;
		minimum = minimum_by_kind[kind] != 0u ? minimum_by_kind[kind] : 1u;
		if ( queued_by_kind[kind] != 0u && (kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE || admission_open == 0u || inflight_submission_count == 0u || queued_by_kind[kind] >= minimum || bypass_count_by_kind[kind] >= bypass_limit ) )
			selected = kind;
		if ( selected != 0u )
			break;
	}
	for (kind=SPARK_MODEL_SERVING_WORK_KIND_PREFILL; kind<=SPARK_MODEL_SERVING_WORK_KIND_DECODE; kind++)
	{
		minimum = minimum_by_kind[kind] != 0u ? minimum_by_kind[kind] : 1u;
		if ( kind != selected && admission_open != 0u && inflight_submission_count != 0u && queued_by_kind[kind] != 0u && queued_by_kind[kind] < minimum && bypass_count_by_kind[kind] != UINT32_MAX )
			bypass_count_by_kind[kind]++;
	}
	if ( selected != 0u )
	{
		bypass_count_by_kind[selected] = 0u;
		*next_work_kind = selected == SPARK_MODEL_SERVING_WORK_KIND_RELEASE ? SPARK_MODEL_SERVING_WORK_KIND_PREFILL : selected + 1u;
	}
	return(selected);
}

static uint32_t *SparkModelBatchRequestTokens(
	SparkModelBatchEngine *engine,
	uint32_t request_slot)
{
	return(&engine->request_token_storage[(uint64_t)request_slot * engine->max_context_tokens]);
}

static void SparkModelBatchDigestTokens(
	SparkSha256Context *context,
	const uint32_t *tokens,
	uint32_t token_count)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	SparkSha256Update(context,tokens,(size_t)token_count * sizeof(tokens[0]));
#else
	uint8_t encoded[sizeof(uint32_t)];
	uint32_t index,token;
	for (index=0u; index<token_count; index++)
	{
		token = tokens[index];
		encoded[0] = (uint8_t)token;
		encoded[1] = (uint8_t)(token >> 8u);
		encoded[2] = (uint8_t)(token >> 16u);
		encoded[3] = (uint8_t)(token >> 24u);
		SparkSha256Update(context,encoded,sizeof(encoded));
	}
#endif
}

static void SparkModelBatchFinalizeIdentity(
	const SparkSha256Context *context,
	SparkModelServingCacheIdentity *identity)
{
	SparkSha256Context copy;
	copy = *context;
	SparkSha256Finalize(&copy,identity->sha256);
}

static uint32_t SparkModelBatchCacheIdentityIsPresent(
	const SparkModelServingCacheIdentity *identity)
{
	uint32_t index,present;
	present = 0u;
	for (index=0u; index<sizeof(identity->sha256); index++)
		present |= identity->sha256[index];
	return(present != 0u ? 1u : 0u);
}

static uint64_t SparkModelBatchHashCacheSpan(
	const SparkModelServingCacheIdentity *identity,
	uint32_t token_count)
{
	uint64_t hash;
	uint32_t index;
	hash = UINT64_C(1469598103934665603);
	for (index=0u; index<sizeof(identity->sha256); index++)
	{
		hash ^= identity->sha256[index];
		hash *= UINT64_C(1099511628211);
	}
	hash ^= token_count;
	return(hash * UINT64_C(1099511628211));
}

static void SparkModelBatchBeginCacheDemand(SparkModelBatchEngine *engine)
{
	engine->cache_demand_epoch++;
	if ( engine->cache_demand_epoch != 0u )
		return;
	memset(engine->cache_demand_entries,0,
		(uint64_t)engine->cache_demand_entry_capacity *
		sizeof(engine->cache_demand_entries[0]));
	engine->cache_demand_epoch = 1u;
}

static uint32_t SparkModelBatchCacheSpanLookup(
	SparkModelBatchEngine *engine,
	const SparkModelServingCacheIdentity *identity,
	uint32_t token_count,
	uint32_t insert)
{
	SparkModelBatchCacheDemandEntry *entry;
	uint32_t index,probe;
	index = (uint32_t)SparkModelBatchHashCacheSpan(identity,token_count) &
		(engine->cache_demand_entry_capacity - 1u);
	for (probe=0u; probe<engine->cache_demand_entry_capacity; probe++)
	{
		entry = &engine->cache_demand_entries[index];
		if ( entry->epoch != engine->cache_demand_epoch )
		{
			if ( insert != 0u )
			{
				entry->epoch = engine->cache_demand_epoch;
				entry->token_count = token_count;
				entry->identity = *identity;
			}
			return(0u);
		}
		if ( entry->token_count == token_count &&
			memcmp(&entry->identity,identity,sizeof(*identity)) == 0 )
			return(1u);
		index = (index + 1u) & (engine->cache_demand_entry_capacity - 1u);
	}
	return(UINT32_MAX);
}

static void SparkModelBatchApplyPrefixLookup(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	const uint32_t *tokens,
	const SparkPrefixCacheLookup *lookup)
{
	request->computed_prompt_token_count = lookup->matched_token_count;
	request->cache_prefix_token_count = lookup->matched_token_count;
	request->cache_published_token_count = lookup->matched_token_count;
	request->cache_lookup_epoch = engine->cache_publication_epoch;
	SparkSha256Initialize(&request->cache_published_digest_context);
	if ( lookup->matched_token_count == 0u )
		return;
	SparkModelBatchDigestTokens(&request->cache_published_digest_context,tokens,
		lookup->matched_token_count);
	SparkModelBatchFinalizeIdentity(&request->cache_published_digest_context,
		&request->cache_prefix_identity);
	request->cache_published_identity = request->cache_prefix_identity;
}

static uint32_t *SparkModelBatchSubmissionRequestSlots(
	SparkModelBatchEngine *engine,
	const SparkModelBatchSubmissionState *submission)
{
	return(&engine->submission_request_slots[(uint64_t)submission->slot_index * engine->max_active_sequence_count]);
}

static uint32_t *SparkModelBatchSubmissionPrefillCounts(
	SparkModelBatchEngine *engine,
	const SparkModelBatchSubmissionState *submission)
{
	return(&engine->submission_prefill_counts[(uint64_t)submission->slot_index * engine->max_active_sequence_count]);
}

static SparkModelBatchRequestHandle SparkModelBatchMakeHandle(
	uint32_t slot,
	uint32_t generation)
{
	return(((uint64_t)generation << 32u) | ((uint64_t)slot + 1u));
}

static uint32_t SparkModelBatchHandleSlot(
	SparkModelBatchRequestHandle handle)
{
	uint32_t encoded;
	encoded = (uint32_t)handle;
	return(encoded != 0u ? encoded - 1u : SPARK_MODEL_BATCH_NO_SLOT);
}

static SparkModelBatchRequestState *SparkModelBatchFindRequest(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestHandle handle)
{
	SparkModelBatchRequestState *request;
	uint32_t slot;
	slot = SparkModelBatchHandleSlot(handle);
	if ( engine == 0 || slot >= engine->request_capacity )
		return(0);
	request = &engine->requests[slot];
	return(request->state != SPARK_MODEL_BATCH_REQUEST_FREE && request->handle == handle ? request : 0);
}

static SparkModelBatchSubmissionState *SparkModelBatchFindSubmission(
	SparkModelBatchEngine *engine,
	uint64_t submission_id)
{
	uint32_t index;
	for (index=0u; index<engine->submission_capacity; index++)
		if ( engine->submissions[index].active != 0u && engine->submissions[index].submission_id == submission_id )
			return(&engine->submissions[index]);
	return(0);
}

static SparkModelBatchSubmissionState *SparkModelBatchReserveSubmission(
	SparkModelBatchEngine *engine,
	uint32_t work_kind)
{
	SparkModelBatchSubmissionState *submission;
	uint32_t index;
	for (index=0u; index<engine->submission_capacity; index++)
	{
		submission = &engine->submissions[index];
		if ( submission->active == 0u )
		{
			memset(submission,0,sizeof(*submission));
			submission->active = 1u;
			submission->slot_index = index;
			submission->work_kind = work_kind;
			submission->result_status = SPARK_STATUS_PENDING;
			return(submission);
		}
	}
	return(0);
}

static void SparkModelBatchReleaseSubmission(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission)
{
	memset(submission,0,sizeof(*submission));
	engine->inflight_submission_count--;
}

static void SparkModelBatchEmit(
	SparkModelBatchEngine *engine,
	const SparkModelBatchRequestState *request,
	uint32_t kind,
	SparkStatus status,
	uint32_t flags,
	uint32_t token_id)
{
	SparkModelBatchEvent event;
	memset(&event,0,sizeof(event));
	event.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	event.descriptor_bytes = SPARK_MODEL_BATCH_EVENT_BYTES;
	event.kind = kind;
	event.flags = flags;
	event.status = status;
	event.token_id = token_id;
	event.token_index = request->generated_token_count != 0u ? request->generated_token_count - 1u : 0u;
	event.generated_token_count = request->generated_token_count;
	event.request_id = request->request_id;
	event.sequence_id = request->sequence_id;
	event.request_handle = request->handle;
	event.model_extension_kind = request->model_extension_kind;
	event.first_draft_miss_count = request->first_draft_miss_count;
	event.first_draft_policy = request->first_draft_policy;
	engine->event_function(engine->event_context,&event);
}

static uint32_t SparkModelBatchBindResidentSlot(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request)
{
	uint32_t slot;
	if ( request->resident_sequence_slot != SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT )
		return(1u);
	if ( engine->free_resident_slot_head == SPARK_MODEL_BATCH_NO_SLOT )
		return(0u);
	slot = engine->free_resident_slot_head;
	engine->free_resident_slot_head = engine->resident_slot_next[slot];
	engine->resident_slot_next[slot] = SPARK_MODEL_BATCH_NO_SLOT;
	engine->free_resident_slot_count--;
	request->resident_sequence_slot = slot;
	return(1u);
}

static void SparkModelBatchReleaseResidentSlot(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request)
{
	uint32_t slot;
	slot = request->resident_sequence_slot;
	if ( slot == SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT )
		return;
	engine->resident_slot_next[slot] = engine->free_resident_slot_head;
	engine->free_resident_slot_head = slot;
	engine->free_resident_slot_count++;
	request->resident_sequence_slot = SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT;
}

static void SparkModelBatchFreeRequest(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request)
{
	SparkStatus status;
	uint32_t slot,generation;
	slot = (uint32_t)(request - engine->requests);
	generation = request->generation;
	if ( engine->cache_block_token_count != 0u )
	{
		status = SparkPrefixCacheReleaseSequence(&engine->prefix_cache,request->sequence_id);
		/* a failed cache release degrades that sequence's prefix reuse,
		 * NOT the engine — closing admission here bricks the engine for
		 * all future requests (a serving engine must stay admitted) */
		(void)status;
	}
	SparkModelBatchReleaseResidentSlot(engine,request);
	memset(request,0,sizeof(*request));
	request->generation = generation;
	request->resident_sequence_slot = SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT;
	request->next_free_slot = engine->free_request_head;
	engine->free_request_head = slot;
	engine->live_request_count--;
}

static void SparkModelBatchEmitTerminal(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request)
{
	SparkModelBatchRequestState snapshot;
	uint32_t kind;
	SparkStatus status;
	snapshot = *request;
	kind = request->terminal_event_kind;
	status = (SparkStatus)request->terminal_status;
	SparkModelBatchFreeRequest(engine,request);
	if ( kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED )
		engine->completed_request_count++;
	else if ( kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED )
		engine->cancelled_request_count++;
	SparkModelBatchEmit(engine,&snapshot,kind,status,0u,0u);
}

static void SparkModelBatchQueueTerminal(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	uint32_t kind,
	SparkStatus status)
{
	request->terminal_event_kind = kind;
	request->terminal_status = status;
	if ( status == SPARK_STATUS_OK && engine->failed_status == SPARK_STATUS_OK && request->resident_bound != 0u && engine->adapter_descriptor->resident_sequence_slot_reuse == SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE )
	{
		request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE;
		return;
	}
	SparkModelBatchEmitTerminal(engine,request);
}

static uint32_t SparkModelBatchTokenIsStop(
	const SparkModelBatchEngine *engine,
	uint32_t token_id)
{
	uint32_t index;
	for (index=0u; index<engine->stop_token_count; index++)
		if ( engine->stop_token_ids[index] == token_id )
			return(1u);
	return(0u);
}

static SparkStatus SparkModelBatchAcceptToken(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	uint32_t token_id)
{
	uint32_t stop;
	uint32_t *tokens;
	if ( request->generated_token_count >= request->output_token_budget || request->prompt_token_count + request->generated_token_count >= engine->max_context_tokens )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	tokens = SparkModelBatchRequestTokens(engine,(uint32_t)(request - engine->requests));
	tokens[request->prompt_token_count + request->generated_token_count] = token_id;
	request->generated_token_count++;
	engine->emitted_token_count++;
	stop = SparkModelBatchTokenIsStop(engine,token_id);
	request->state = SPARK_MODEL_BATCH_REQUEST_COMPLETING;
	SparkModelBatchEmit(engine,request,SPARK_MODEL_BATCH_EVENT_TOKEN,SPARK_STATUS_OK,stop != 0u ? SPARK_MODEL_BATCH_EVENT_FLAG_STOP_TOKEN : 0u,token_id);
	if ( request->cancel_pending != 0u )
		SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED,SPARK_STATUS_OK);
	else if ( stop != 0u || request->generated_token_count >= request->output_token_budget )
		SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED,SPARK_STATUS_OK);
	else
		request->state = SPARK_MODEL_BATCH_REQUEST_READY_DECODE;
	return(SPARK_STATUS_OK);
}

static void SparkModelBatchSetFailed(
	SparkModelBatchEngine *engine,
	SparkStatus status)
{
	/* ENGINE-level failure only: called from pipeline Progress when the
	 * daemon connection itself is broken. Individual request failures
	 * (daemon rejecting one submission) go through HandleRejected →
	 * FailRequest per-request, NOT through this function. A serving
	 * engine stays admitted unless the pipeline is actually dead —
	 * one bad request must not brick the engine for all callers. */
	if ( engine->failed_status == SPARK_STATUS_OK )
		engine->failed_status = status;
	engine->admission_open = 0u;
}

static void SparkModelBatchFailRequest(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	SparkStatus status)
{
	SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_ERROR,status);
}

static void SparkModelBatchRestoreRejectedRequest(
	SparkModelBatchRequestState *request,
	uint32_t work_kind)
{
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL;
	else if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		request->state = SPARK_MODEL_BATCH_REQUEST_READY_DECODE;
	else
		request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE;
}

static void SparkModelBatchHandleRejected(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	SparkStatus status)
{
	uint32_t *request_slots;
	uint32_t lane;
	fprintf(stderr,"batch_rejected status=%u kind=%u submission=%llu\n",
		(uint32_t)status,submission->work_kind,
		(unsigned long long)submission->submission_id);
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		request = &engine->requests[request_slots[lane]];
		if ( status == SPARK_STATUS_BUSY )
			SparkModelBatchRestoreRejectedRequest(request,submission->work_kind);
		else
			SparkModelBatchFailRequest(engine,request,status);
	}
}

static SparkStatus SparkModelBatchPublishCompletedBlocks(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	uint32_t request_slot,
	uint32_t completed_token_count)
{
	SparkPrefixCacheLookup committed;
	SparkStatus status;
	uint32_t completed_block_tokens;
	uint32_t *tokens;
	if ( engine->cache_block_token_count == 0u )
		return(SPARK_STATUS_OK);
	completed_block_tokens = (completed_token_count / engine->cache_block_token_count) * engine->cache_block_token_count;
	if ( completed_block_tokens <= request->cache_published_token_count )
		return(SPARK_STATUS_OK);
	tokens = SparkModelBatchRequestTokens(engine,request_slot);
	SparkModelBatchDigestTokens(
		&request->cache_published_digest_context,
		&tokens[request->cache_published_token_count],
		completed_block_tokens - request->cache_published_token_count);
	status = SparkPrefixCacheCommitPrompt(
		&engine->prefix_cache,
		request->sequence_id,
		tokens,
		completed_block_tokens,
		&committed);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request->cache_published_token_count = committed.matched_token_count;
	SparkModelBatchFinalizeIdentity(
		&request->cache_published_digest_context,
		&request->cache_published_identity);
	engine->cache_publication_epoch++;
	if ( engine->cache_publication_epoch == 0u )
		engine->cache_publication_epoch = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelBatchHandlePrefillCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	const SparkModelServingCompletion *completion)
{
	uint32_t *request_slots,*prefill_counts;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	prefill_counts = SparkModelBatchSubmissionPrefillCounts(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		SparkStatus status;
		request = &engine->requests[request_slots[lane]];
		fprintf(stderr,"G5N-ENG prefill-complete: sub %llu lane %u slot %u counts %u computed %u prompt %u\n",
			(unsigned long long)submission->submission_id,(unsigned)lane,
			(unsigned)request_slots[lane],(unsigned)prefill_counts[lane],
			(unsigned)request->computed_prompt_token_count,
			(unsigned)request->prompt_token_count);
		request->computed_prompt_token_count += prefill_counts[lane];
		request->resident_bound = 1u;
		status = SparkModelBatchPublishCompletedBlocks(engine,request,request_slots[lane],request->computed_prompt_token_count);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( request->computed_prompt_token_count < request->prompt_token_count )
			request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL;
		else if ( SparkModelBatchAcceptToken(engine,request,completion->token_ids[lane]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelBatchHandleDecodeCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	const SparkModelServingCompletion *completion)
{
	uint32_t *request_slots;
	uint32_t lane,step,token_index;
	uint32_t extension_miss,extension_policy;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	extension_miss = 0u;
	extension_policy = 0u;
	if ( completion->model_extension_kind == 0x5136u && completion->model_extension_bytes >= 2u * sizeof(uint32_t) )
	{
		memcpy(&extension_miss, completion->model_extension, sizeof(extension_miss));
		memcpy(&extension_policy, (const uint8_t *)completion->model_extension + sizeof(extension_miss), sizeof(extension_policy));
	}
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		SparkStatus status;
		request = &engine->requests[request_slots[lane]];
		request->resident_bound = 1u;
		if ( completion->model_extension_kind == 0x5136u )
		{
			request->model_extension_kind = completion->model_extension_kind;
			/* The adapter reports ONE aggregate miss count for the whole
			 * submission; crediting it to every request fabricated a miss
			 * on all B-1 peers. Attribute per-request only when the batch
			 * has a single lane, and keep the aggregate at engine level. */
			if ( submission->lane_count == 1u )
				request->first_draft_miss_count += extension_miss;
			else
				engine->batch_first_draft_miss_count += extension_miss;
			request->first_draft_policy = extension_policy;
		}
		for (step=0u; step<completion->tokens_per_sequence; step++)
		{
			status = SparkModelBatchPublishCompletedBlocks(engine,request,
				request_slots[lane],request->prompt_token_count +
				request->generated_token_count);
			if ( status != SPARK_STATUS_OK )
				return(status);
			token_index = lane * completion->tokens_per_sequence + step;
			if ( SparkModelBatchAcceptToken(engine,request,
				completion->token_ids[token_index]) != SPARK_STATUS_OK )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			if ( request->state != SPARK_MODEL_BATCH_REQUEST_READY_DECODE )
				break;
		}
	}
	return(SPARK_STATUS_OK);
}

static void SparkModelBatchHandleReleaseCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission)
{
	uint32_t *request_slots;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		request = &engine->requests[request_slots[lane]];
		request->resident_bound = 0u;
		SparkModelBatchEmitTerminal(engine,request);
	}
}

static void SparkModelBatchSubmitResult(
	void *result_context,
	uint64_t submission_id,
	SparkStatus status)
{
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmissionState *submission;
	engine = (SparkModelBatchEngine *)result_context;
	submission = engine != 0 ? SparkModelBatchFindSubmission(engine,submission_id) : 0;
	if ( submission == 0 || submission->result_received != 0u )
	{
		if ( engine != 0 )
			SparkModelBatchSetFailed(engine,SPARK_STATUS_SCHEMA_ERROR);
		return;
	}
	submission->result_received = 1u;
	submission->result_status = status;
	submission->admitted = status == SPARK_STATUS_OK ? 1u : 0u;
	fprintf(stderr,"G5N-ENG submit-result: sub %llu status %u admitted %u\n",
		(unsigned long long)submission_id,(unsigned)status,(unsigned)submission->admitted);
}

static SparkStatus SparkModelBatchApplyCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	const SparkModelServingCompletion *completion)
{
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		return(SparkModelBatchHandlePrefillCompletion(engine,submission,completion));
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(SparkModelBatchHandleDecodeCompletion(engine,submission,completion));
	SparkModelBatchHandleReleaseCompletion(engine,submission);
	return(SPARK_STATUS_OK);
}

static void SparkModelBatchFailSubmissionRequests(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	SparkStatus status)
{
	uint32_t *request_slots;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
		SparkModelBatchFailRequest(engine,&engine->requests[request_slots[lane]],status);
}

static void SparkModelBatchCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmissionState *submission;
	SparkStatus status;
	engine = (SparkModelBatchEngine *)completion_context;
	submission = engine != 0 && completion != 0 ? SparkModelBatchFindSubmission(engine,completion->submission_id) : 0;
	if ( submission == 0 || submission->result_received == 0u )
	{
		if ( engine != 0 )
			SparkModelBatchSetFailed(engine,SPARK_STATUS_SCHEMA_ERROR);
		return;
	}
	status = (SparkStatus)completion->status;
	fprintf(stderr,"G5N-ENG completion: sub %llu status %u admitted %u lane_count %u\n",
		(unsigned long long)completion->submission_id,(unsigned)status,
		(unsigned)submission->admitted,(unsigned)submission->lane_count);
	if ( submission->admitted == 0u || status != SPARK_STATUS_OK )
	{
		status = submission->admitted == 0u ? (SparkStatus)submission->result_status : status;
		/* per-request failure: don't SetFailed (that bricks the engine);
		 * HandleRejected fails only the requests on this submission */
		SparkModelBatchHandleRejected(engine,submission,status);
	}
	else if ( SparkModelBatchApplyCompletion(engine,submission,completion) != SPARK_STATUS_OK )
	{
		SparkModelBatchSetFailed(engine,SPARK_STATUS_CAPACITY_EXCEEDED);
		SparkModelBatchFailSubmissionRequests(engine,submission,SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	SparkModelBatchReleaseSubmission(engine,submission);
}

static SparkStatus SparkModelBatchValidateConfiguration(
	const SparkModelBatchEngineConfiguration *configuration)
{
	uint32_t left,right;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_BATCH_ENGINE_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_BATCH_ENGINE_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( configuration->flags != 0u || configuration->connect_timeout_ms == 0u || configuration->request_capacity == 0u || configuration->max_context_tokens < 2u || configuration->max_prefill_rows_per_submission == 0u || configuration->maximum_messages_per_rank_per_progress == 0u || configuration->stop_token_count > SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT || configuration->deployment == 0 || configuration->runtime_root == 0 || configuration->event_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (left=0u; left<configuration->stop_token_count; left++)
		for (right=left + 1u; right<configuration->stop_token_count; right++)
			if ( configuration->stop_token_ids[left] == configuration->stop_token_ids[right] )
				return(SPARK_STATUS_DUPLICATE);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelBatchAllocate(
	SparkModelBatchEngine *engine)
{
	SparkPrefixCacheConfiguration prefix_configuration;
	SparkStatus status;
	uint32_t request_tokens,submission_lanes;
	if ( SparkModelBatchMultiplyFits(engine->request_capacity,engine->max_context_tokens) == 0u || SparkModelBatchMultiplyFits(engine->submission_capacity,engine->max_active_sequence_count) == 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	request_tokens = engine->request_capacity * engine->max_context_tokens;
	submission_lanes = engine->submission_capacity * engine->max_active_sequence_count;
	engine->requests = (SparkModelBatchRequestState *)calloc(engine->request_capacity,sizeof(engine->requests[0]));
	engine->submissions = (SparkModelBatchSubmissionState *)calloc(engine->submission_capacity,sizeof(engine->submissions[0]));
	engine->request_token_storage = (uint32_t *)calloc(request_tokens,sizeof(engine->request_token_storage[0]));
	engine->resident_slot_next = (uint32_t *)calloc(engine->resident_sequence_capacity,sizeof(engine->resident_slot_next[0]));
	engine->submission_request_slots = (uint32_t *)calloc(submission_lanes,sizeof(engine->submission_request_slots[0]));
	engine->submission_prefill_counts = (uint32_t *)calloc(submission_lanes,sizeof(engine->submission_prefill_counts[0]));
	engine->scratch_lanes = (SparkModelServingLane *)calloc(engine->max_active_sequence_count,sizeof(engine->scratch_lanes[0]));
	engine->scratch_token_ids = (uint32_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_token_ids[0]));
	engine->scratch_row_lane_indices = (uint32_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_row_lane_indices[0]));
	engine->scratch_row_positions = (uint64_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_row_positions[0]));
	engine->scratch_row_sequence_ids = (uint64_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_row_sequence_ids[0]));
	engine->scratch_request_slots = (uint32_t *)calloc(engine->max_active_sequence_count,sizeof(engine->scratch_request_slots[0]));
	engine->scratch_prefill_counts = (uint32_t *)calloc(engine->max_active_sequence_count,sizeof(engine->scratch_prefill_counts[0]));
	if ( engine->requests == 0 || engine->submissions == 0 || engine->request_token_storage == 0 || engine->resident_slot_next == 0 || engine->submission_request_slots == 0 || engine->submission_prefill_counts == 0 || engine->scratch_lanes == 0 || engine->scratch_token_ids == 0 || engine->scratch_row_lane_indices == 0 || engine->scratch_row_positions == 0 || engine->scratch_row_sequence_ids == 0 || engine->scratch_request_slots == 0 || engine->scratch_prefill_counts == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( engine->cache_block_token_count == 0u )
		return(SPARK_STATUS_OK);
	engine->cache_demand_entry_capacity =
		SparkModelBatchCacheDemandCapacity(engine->request_capacity);
	if ( engine->cache_demand_entry_capacity == 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	engine->cache_demand_entries = (SparkModelBatchCacheDemandEntry *)calloc(
		engine->cache_demand_entry_capacity,sizeof(engine->cache_demand_entries[0]));
	engine->prefix_cache_entries = (SparkPrefixCacheEntry *)calloc(
		engine->prefix_cache_entry_capacity,sizeof(engine->prefix_cache_entries[0]));
	engine->prefix_cache_bindings = (SparkPrefixCacheSequenceBinding *)calloc(
		engine->prefix_cache_binding_capacity,
		sizeof(engine->prefix_cache_bindings[0]));
	engine->prefix_entry_hash_heads = (uint32_t *)calloc(
		engine->prefix_cache_entry_capacity,
		sizeof(engine->prefix_entry_hash_heads[0]));
	engine->prefix_binding_lookup_hash_heads = (uint32_t *)calloc(
		engine->prefix_cache_binding_capacity,
		sizeof(engine->prefix_binding_lookup_hash_heads[0]));
	engine->prefix_binding_sequence_hash_heads = (uint32_t *)calloc(
		engine->prefix_cache_binding_capacity,
		sizeof(engine->prefix_binding_sequence_hash_heads[0]));
	if ( engine->cache_demand_entries == 0 || engine->prefix_cache_entries == 0 || engine->prefix_cache_bindings == 0 || engine->prefix_entry_hash_heads == 0 || engine->prefix_binding_lookup_hash_heads == 0 || engine->prefix_binding_sequence_hash_heads == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&prefix_configuration,0,sizeof(prefix_configuration));
	prefix_configuration.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
	prefix_configuration.descriptor_bytes = SPARK_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	prefix_configuration.block_token_count = engine->cache_block_token_count;
	prefix_configuration.entry_count = engine->prefix_cache_entry_capacity;
	prefix_configuration.logical_block_count = engine->prefix_cache_entry_capacity;
	prefix_configuration.sequence_binding_count =
		engine->prefix_cache_binding_capacity;
	prefix_configuration.entries = engine->prefix_cache_entries;
	prefix_configuration.sequence_bindings = engine->prefix_cache_bindings;
	prefix_configuration.entry_hash_bucket_count =
		engine->prefix_cache_entry_capacity;
	prefix_configuration.binding_hash_bucket_count =
		engine->prefix_cache_binding_capacity;
	prefix_configuration.entry_hash_bucket_heads = engine->prefix_entry_hash_heads;
	prefix_configuration.binding_lookup_hash_bucket_heads = engine->prefix_binding_lookup_hash_heads;
	prefix_configuration.binding_sequence_hash_bucket_heads = engine->prefix_binding_sequence_hash_heads;
	status = SparkPrefixCacheInitialize(&engine->prefix_cache,&prefix_configuration);
	return(status);
}

static void SparkModelBatchInitializeFreeList(
	SparkModelBatchEngine *engine)
{
	uint32_t index;
	engine->free_request_head = 0u;
	for (index=0u; index<engine->request_capacity; index++)
	{
		engine->requests[index].next_free_slot = index + 1u < engine->request_capacity ? index + 1u : SPARK_MODEL_BATCH_NO_SLOT;
		engine->requests[index].resident_sequence_slot = SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT;
	}
	engine->free_resident_slot_head = 0u;
	engine->free_resident_slot_count = engine->resident_sequence_capacity;
	for (index=0u; index<engine->resident_sequence_capacity; index++)
		engine->resident_slot_next[index] = index + 1u < engine->resident_sequence_capacity ? index + 1u : SPARK_MODEL_BATCH_NO_SLOT;
}

static SparkStatus SparkModelBatchConnectPipeline(
	const SparkModelBatchEngineConfiguration *configuration,
	SparkModelBatchEngine *engine)
{
	SparkModelPipelineClientConfiguration pipeline_configuration;
	const SparkModelResidentDeploymentNode *coordinator = SparkModelResidentDeploymentFindRank(configuration->deployment,configuration->deployment->coordinator_rank_index);
	memset(&pipeline_configuration,0,sizeof(pipeline_configuration));
	pipeline_configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	pipeline_configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
	pipeline_configuration.connect_timeout_ms = configuration->connect_timeout_ms;
	pipeline_configuration.deployment = configuration->deployment;
	pipeline_configuration.runtime_root = coordinator != 0 ? coordinator->runtime_root : configuration->runtime_root;
	pipeline_configuration.submit_result_function = SparkModelBatchSubmitResult;
	pipeline_configuration.submit_result_context = engine;
	pipeline_configuration.completion_function = SparkModelBatchCompletion;
	pipeline_configuration.completion_context = engine;
	pipeline_configuration.stage_completion_function = configuration->stage_completion_function;
	pipeline_configuration.stage_completion_context = configuration->stage_completion_context;
	return(SparkModelPipelineClientConnect(&pipeline_configuration,&engine->pipeline));
}

static SparkStatus SparkModelBatchInitialize(
	const SparkModelBatchEngineConfiguration *configuration,
	SparkModelBatchEngine *engine)
{
	const SparkModelServingRuntimeLimits *limits;
	SparkStatus status;
	limits = &configuration->deployment->runtime_limits;
	engine->request_capacity = configuration->request_capacity;
	engine->resident_sequence_capacity = limits->resident_sequence_capacity;
	engine->kv_logical_page_capacity = limits->kv_logical_page_capacity;
	engine->kv_physical_page_capacity = limits->kv_physical_page_capacity;
	engine->max_context_tokens = configuration->max_context_tokens;
	engine->max_prefill_rows = configuration->max_prefill_rows_per_submission;
	engine->max_active_sequence_count = limits->max_active_sequence_count;
	engine->scratch_row_capacity = engine->max_prefill_rows > engine->max_active_sequence_count ? engine->max_prefill_rows : engine->max_active_sequence_count;
	engine->submission_capacity = limits->max_inflight_submission_count;
	engine->maximum_messages_per_rank = configuration->maximum_messages_per_rank_per_progress;
	engine->stop_token_count = configuration->stop_token_count;
	memcpy(engine->stop_token_ids,configuration->stop_token_ids,sizeof(engine->stop_token_ids));
	engine->event_function = configuration->event_function;
	engine->event_context = configuration->event_context;
	engine->admission_open = 1u;
	engine->next_work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	engine->cache_publication_epoch = 1u;
	if ( engine->max_prefill_rows > limits->max_input_row_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkModelBatchConnectPipeline(configuration,engine);
	if ( status != SPARK_STATUS_OK )
		return(status);
	engine->adapter_descriptor = SparkModelPipelineClientGetAdapterDescriptor(engine->pipeline);
	if ( engine->adapter_descriptor == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	engine->cache_block_token_count = engine->adapter_descriptor->cache_block_token_count;
	if ( engine->cache_block_token_count != 0u )
	{
		engine->prefix_cache_entry_capacity = engine->kv_logical_page_capacity;
		engine->prefix_cache_binding_capacity =
			engine->kv_logical_page_capacity;
	}
	status = SparkModelBatchAllocate(engine);
	if ( engine->requests != 0 && engine->resident_slot_next != 0 )
		SparkModelBatchInitializeFreeList(engine);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineConnect(
	const SparkModelBatchEngineConfiguration *configuration,
	SparkModelBatchEngine **engine_out)
{
	SparkModelBatchEngine *engine;
	SparkStatus status;
	if ( engine_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*engine_out = 0;
	status = SparkModelBatchValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	engine = (SparkModelBatchEngine *)calloc(1u,sizeof(*engine));
	if ( engine == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkModelBatchInitialize(configuration,engine);
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkModelBatchEngineDestroy(engine);
		return(status);
	}
	*engine_out = engine;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineDestroy(SparkModelBatchEngine *engine)
{
	SparkModelPipelineClientView pipeline_view;
	SparkStatus status;
	if ( engine == 0 )
		return(SPARK_STATUS_OK);
	if ( engine->live_request_count != 0u || engine->inflight_submission_count != 0u )
		return(SPARK_STATUS_BUSY);
	if ( engine->requests != 0 && engine->resident_slot_next != 0 && engine->free_resident_slot_count != engine->resident_sequence_capacity )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( engine->pipeline != 0 )
	{
		status = SparkModelPipelineClientGetView(engine->pipeline,&pipeline_view);
		if ( status != SPARK_STATUS_OK || pipeline_view.active_transaction_count != 0u )
			return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_BUSY);
	}
	SparkModelPipelineClientDestroy(engine->pipeline);
	free(engine->scratch_prefill_counts);
	free(engine->scratch_request_slots);
	free(engine->scratch_row_sequence_ids);
	free(engine->scratch_row_positions);
	free(engine->scratch_row_lane_indices);
	free(engine->scratch_token_ids);
	free(engine->scratch_lanes);
	free(engine->cache_demand_entries);
	free(engine->prefix_cache_bindings);
	free(engine->prefix_cache_entries);
	free(engine->prefix_binding_sequence_hash_heads);
	free(engine->prefix_binding_lookup_hash_heads);
	free(engine->prefix_entry_hash_heads);
	free(engine->submission_prefill_counts);
	free(engine->submission_request_slots);
	free(engine->resident_slot_next);
	free(engine->request_token_storage);
	free(engine->submissions);
	free(engine->requests);
	free(engine);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelBatchRequestIdExists(
	const SparkModelBatchEngine *engine,
	uint64_t request_id,
	uint64_t sequence_id)
{
	uint32_t index;
	for (index=0u; index<engine->request_capacity; index++)
		if ( engine->requests[index].state != SPARK_MODEL_BATCH_REQUEST_FREE && (engine->requests[index].request_id == request_id || engine->requests[index].sequence_id == sequence_id) )
			return(1u);
	return(0u);
}

static SparkStatus SparkModelBatchValidateSubmit(
	const SparkModelBatchEngine *engine,
	const SparkModelBatchSubmitRequest *request)
{
	if ( engine == 0 || request == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->abi_version != SPARK_MODEL_BATCH_ENGINE_ABI_VERSION || request->descriptor_bytes != SPARK_MODEL_BATCH_SUBMIT_REQUEST_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( request->reserved0 != 0u || request->request_id == 0u || request->sequence_id == 0u || request->prompt_token_ids == 0 || request->prompt_token_count == 0u || request->output_token_budget == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->prompt_token_count > engine->max_context_tokens || request->output_token_budget > engine->max_context_tokens - request->prompt_token_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkModelBatchSchedulerRequestFitsPageCapacity(
		engine->cache_block_token_count,engine->kv_physical_page_capacity,
		request->prompt_token_count,request->output_token_budget) == 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkModelBatchRequestIdExists(engine,request->request_id,request->sequence_id) != 0u )
		return(SPARK_STATUS_DUPLICATE);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineSubmit(
	SparkModelBatchEngine *engine,
	const SparkModelBatchSubmitRequest *request,
	SparkModelBatchRequestHandle *request_handle_out)
{
	SparkModelBatchRequestState *state;
	SparkStatus status;
	uint32_t slot;
	if ( request_handle_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*request_handle_out = SPARK_MODEL_BATCH_ENGINE_INVALID_REQUEST_HANDLE;
	status = SparkModelBatchValidateSubmit(engine,request);
	if ( status != SPARK_STATUS_OK || engine->admission_open == 0u || engine->failed_status != SPARK_STATUS_OK || engine->free_request_head == SPARK_MODEL_BATCH_NO_SLOT )
		return(status != SPARK_STATUS_OK ? status : engine->failed_status != SPARK_STATUS_OK ? (SparkStatus)engine->failed_status : SPARK_STATUS_BUSY);
	slot = engine->free_request_head;
	state = &engine->requests[slot];
	engine->free_request_head = state->next_free_slot;
	state->generation++;
	if ( state->generation == 0u )
		state->generation = 1u;
	state->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL;
	state->priority = request->priority;
	state->prompt_token_count = request->prompt_token_count;
	state->output_token_budget = request->output_token_budget;
	state->request_id = request->request_id;
	state->sequence_id = request->sequence_id;
	state->handle = SparkModelBatchMakeHandle(slot,state->generation);
	memcpy(SparkModelBatchRequestTokens(engine,slot),request->prompt_token_ids,(size_t)request->prompt_token_count * sizeof(uint32_t));
	SparkSha256Initialize(&state->cache_published_digest_context);
	engine->live_request_count++;
	engine->submitted_request_count++;
	*request_handle_out = state->handle;
	SparkModelBatchEmit(engine,state,SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED,SPARK_STATUS_OK,0u,0u);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineCancel(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestHandle request_handle)
{
	SparkModelBatchRequestState *request;
	request = SparkModelBatchFindRequest(engine,request_handle);
	if ( request == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( request->state == SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_COMPLETING )
	{
		request->cancel_pending = 1u;
		return(SPARK_STATUS_PENDING);
	}
	if ( request->state == SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE )
	{
		request->terminal_event_kind = SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED;
		request->terminal_status = SPARK_STATUS_OK;
		return(SPARK_STATUS_OK);
	}
	SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelBatchStateForWork(uint32_t work_kind)
{
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		return(SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(SPARK_MODEL_BATCH_REQUEST_READY_DECODE);
	return(SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE);
}

static uint32_t SparkModelBatchInflightStateForWork(uint32_t work_kind)
{
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		return(SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT);
	return(SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT);
}

static void SparkModelBatchRefreshQueuedPrefix(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request)
{
	SparkPrefixCacheLookup lookup;
	SparkStatus status;
	uint32_t slot;
	if ( engine->cache_block_token_count == 0u ||
		request->computed_prompt_token_count != 0u ||
		request->cache_lookup_epoch == engine->cache_publication_epoch )
		return;
	slot = (uint32_t)(request - engine->requests);
	memset(&lookup,0,sizeof(lookup));
	status = SparkPrefixCacheLookupPrompt(&engine->prefix_cache,
		request->sequence_id,SparkModelBatchRequestTokens(engine,slot),
		request->prompt_token_count,&lookup);
	if ( status == SPARK_STATUS_OK )
		SparkModelBatchApplyPrefixLookup(engine,request,
			SparkModelBatchRequestTokens(engine,slot),&lookup);
	else
		request->cache_lookup_epoch = engine->cache_publication_epoch;
}

static uint32_t SparkModelBatchPrefillSpan(
	const SparkModelBatchEngine *engine,
	const SparkModelBatchRequestState *request)
{
	uint32_t block_remaining,remaining;
	remaining = request->prompt_token_count - request->computed_prompt_token_count;
	if ( engine->cache_block_token_count == 0u )
		return(remaining);
	block_remaining = engine->cache_block_token_count -
		(request->computed_prompt_token_count % engine->cache_block_token_count);
	return(remaining < block_remaining ? remaining : block_remaining);
}

static uint32_t SparkModelBatchRequestContextTokenCount(
	const SparkModelBatchEngine *engine,
	const SparkModelBatchRequestState *request,
	uint32_t work_kind,
	uint32_t prefill_count)
{
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		return(request->computed_prompt_token_count + (prefill_count != 0u ?
			prefill_count : SparkModelBatchPrefillSpan(engine,request)));
	return(request->prompt_token_count + request->generated_token_count);
}

static uint32_t SparkModelBatchCacheSpansDiffer(
	const SparkModelBatchRequestState *request)
{
	return(request->cache_prefix_token_count !=
		request->cache_published_token_count ||
		memcmp(&request->cache_prefix_identity,
			&request->cache_published_identity,
			sizeof(request->cache_prefix_identity)) != 0 ? 1u : 0u);
}

static uint32_t SparkModelBatchCacheDemandInsertSpans(
	SparkModelBatchEngine *engine,
	const SparkModelBatchRequestState *request,
	uint32_t prefix_seen)
{
	uint32_t status;
	if ( request->cache_prefix_token_count != 0u &&
		SparkModelBatchCacheSpansDiffer(request) != 0u && prefix_seen == 0u )
	{
		status = SparkModelBatchCacheSpanLookup(engine,
			&request->cache_prefix_identity,
			request->cache_prefix_token_count,1u);
		if ( status == UINT32_MAX )
			return(0u);
	}
	status = SparkModelBatchCacheSpanLookup(engine,
		&request->cache_published_identity,
		request->cache_published_token_count,1u);
	return(status != UINT32_MAX ? 1u : 0u);
}

static uint32_t SparkModelBatchCacheDemandAdditional(
	SparkModelBatchEngine *engine,
	const SparkModelBatchRequestState *request,
	uint32_t context_token_count,
	uint32_t insert)
{
	uint32_t additional,block_tokens,prefix_seen,prefix_tokens;
	uint32_t published_seen,published_tokens,remaining;
	block_tokens = engine->cache_block_token_count;
	published_tokens = request->cache_published_token_count;
	prefix_tokens = request->cache_prefix_token_count;
	if ( block_tokens == 0u || published_tokens > context_token_count ||
		published_tokens % block_tokens != 0u || prefix_tokens > published_tokens ||
		prefix_tokens % block_tokens != 0u )
		return(UINT32_MAX);
	remaining = context_token_count - published_tokens;
	additional = (remaining / block_tokens) +
		(remaining % block_tokens != 0u ? 1u : 0u);
	if ( published_tokens == 0u )
		return(additional);
	if ( SparkModelBatchCacheIdentityIsPresent(
		&request->cache_published_identity) == 0u )
		return(UINT32_MAX);
	published_seen = SparkModelBatchCacheSpanLookup(engine,
		&request->cache_published_identity,published_tokens,0u);
	if ( published_seen == UINT32_MAX )
		return(UINT32_MAX);
	prefix_seen = 1u;
	if ( published_seen == 0u && prefix_tokens != 0u &&
		SparkModelBatchCacheSpansDiffer(request) != 0u )
	{
		if ( SparkModelBatchCacheIdentityIsPresent(
			&request->cache_prefix_identity) == 0u )
			return(UINT32_MAX);
		prefix_seen = SparkModelBatchCacheSpanLookup(engine,
			&request->cache_prefix_identity,prefix_tokens,0u);
		if ( prefix_seen == UINT32_MAX )
			return(UINT32_MAX);
		additional += prefix_seen == 0u ? prefix_tokens / block_tokens : 0u;
		additional += (published_tokens - prefix_tokens) / block_tokens;
	}
	else if ( published_seen == 0u )
		additional += published_tokens / block_tokens;
	if ( insert == 0u || published_seen != 0u )
		return(additional);
	return(SparkModelBatchCacheDemandInsertSpans(engine,request,prefix_seen) != 0u ?
		additional : UINT32_MAX);
}

static uint32_t SparkModelBatchCacheDemandTryAdd(
	SparkModelBatchEngine *engine,
	SparkModelBatchCacheDemand *demand,
	const SparkModelBatchRequestState *request,
	uint32_t context_token_count,
	uint32_t page_capacity)
{
	uint32_t additional,committed;
	additional = SparkModelBatchCacheDemandAdditional(engine,request,
		context_token_count,0u);
	if ( additional == UINT32_MAX ||
		SparkModelBatchSchedulerCacheDemandFits(page_capacity,
			demand->page_count,additional) == 0u )
		return(0u);
	committed = SparkModelBatchCacheDemandAdditional(engine,request,
		context_token_count,1u);
	if ( committed != additional )
		return(0u);
	demand->page_count += additional;
	return(1u);
}

static uint32_t SparkModelBatchBuildInflightCacheDemand(
	SparkModelBatchEngine *engine,
	SparkModelBatchCacheDemand *demand)
{
	SparkModelBatchRequestState *request;
	SparkModelBatchSubmissionState *submission;
	uint32_t context,index,lane,*prefill_counts,*request_slots;
	memset(demand,0,sizeof(*demand));
	for (index=0u; index<engine->submission_capacity; index++)
	{
		submission = &engine->submissions[index];
		if ( submission->active == 0u || submission->work_kind ==
			SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			continue;
		request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
		prefill_counts = SparkModelBatchSubmissionPrefillCounts(engine,submission);
		for (lane=0u; lane<submission->lane_count; lane++)
		{
			request = &engine->requests[request_slots[lane]];
			context = SparkModelBatchRequestContextTokenCount(engine,request,
				submission->work_kind,prefill_counts[lane]);
			if ( SparkModelBatchCacheDemandTryAdd(engine,demand,request,context,
				UINT32_MAX) == 0u )
				return(0u);
		}
	}
	return(1u);
}

static void SparkModelBatchRefreshInflightKvPageCount(
	SparkModelBatchEngine *engine)
{
	SparkModelBatchCacheDemand demand;
	if ( engine->kv_physical_page_capacity == 0u )
	{
		engine->inflight_kv_page_count = 0u;
		return;
	}
	SparkModelBatchBeginCacheDemand(engine);
	engine->inflight_kv_page_count =
		SparkModelBatchBuildInflightCacheDemand(engine,&demand) != 0u ?
		demand.page_count : UINT32_MAX;
}

static uint32_t SparkModelBatchMaximumLaneCount(
	const SparkModelBatchEngine *engine,
	uint32_t work_kind)
{
	uint32_t maximum;
	maximum = engine->max_active_sequence_count;
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL && maximum > engine->max_prefill_rows )
		maximum = engine->max_prefill_rows;
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE ||
		engine->kv_physical_page_capacity == 0u )
		return(maximum);
	return(SparkModelBatchSchedulerPlanCacheBoundLaneCount(maximum,
		engine->kv_physical_page_capacity,engine->inflight_kv_page_count));
}

static void SparkModelBatchCountDispatchableRequests(
	const SparkModelBatchEngine *engine,
	uint32_t queued_by_kind[4])
{
	uint32_t index,unbound_prefill;
	memset(queued_by_kind,0,4u * sizeof(queued_by_kind[0]));
	unbound_prefill = 0u;
	for (index=0u; index<engine->request_capacity; index++)
	{
		if ( engine->requests[index].state == SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL )
		{
			if ( engine->requests[index].resident_sequence_slot == SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT )
				unbound_prefill++;
			else
				queued_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL]++;
		}
		else if ( engine->requests[index].state == SPARK_MODEL_BATCH_REQUEST_READY_DECODE )
			queued_by_kind[SPARK_MODEL_SERVING_WORK_KIND_DECODE]++;
		else if ( engine->requests[index].state == SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE )
			queued_by_kind[SPARK_MODEL_SERVING_WORK_KIND_RELEASE]++;
	}
	queued_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL] += unbound_prefill < engine->free_resident_slot_count ? unbound_prefill : engine->free_resident_slot_count;
}

static void SparkModelBatchCountInflightSubmissions(
	const SparkModelBatchEngine *engine,
	uint32_t inflight_by_kind[4])
{
	uint32_t index,kind;
	memset(inflight_by_kind,0,4u * sizeof(inflight_by_kind[0]));
	for (index=0u; index<engine->submission_capacity; index++)
	{
		kind = engine->submissions[index].work_kind;
		if ( engine->submissions[index].active != 0u && engine->submissions[index].submission_id != 0u && kind >= SPARK_MODEL_SERVING_WORK_KIND_PREFILL && kind <= SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			inflight_by_kind[kind]++;
	}
}

static uint32_t SparkModelBatchSelectRequestPass(
	SparkModelBatchEngine *engine,
	SparkModelBatchCacheDemand *cache_demand,
	uint32_t state,
	uint32_t work_kind,
	uint32_t lane_limit,
	uint32_t selection,
	uint32_t maximum_priority,
	uint32_t selected,
	uint32_t *prefill_span_budget)
{
	SparkModelBatchRequestState *request;
	uint32_t aged,context,index,resident_bound,slot;
	for (index=0u; index<engine->request_capacity && selected<lane_limit; index++)
	{
		slot = (engine->next_request_scan + index) % engine->request_capacity;
		request = &engine->requests[slot];
		if ( request->state != state )
			continue;
		aged = request->scheduling_bypass_count >= engine->submission_capacity;
		if ( (selection == SPARK_MODEL_BATCH_SELECT_AGED && aged == 0u) || (selection == SPARK_MODEL_BATCH_SELECT_PRIORITY && (aged != 0u || request->priority != maximum_priority)) || (selection == SPARK_MODEL_BATCH_SELECT_FILL && (aged != 0u || request->priority == maximum_priority)) )
			continue;
		if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
			SparkModelBatchRefreshQueuedPrefix(engine,request);
		resident_bound = request->resident_sequence_slot !=
			SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT ? 1u : 0u;
		if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL &&
			SparkModelBatchBindResidentSlot(engine,request) == 0u )
			continue;
		if ( cache_demand != 0 )
		{
			context = SparkModelBatchRequestContextTokenCount(engine,request,
				work_kind,0u);
			if ( SparkModelBatchCacheDemandTryAdd(engine,cache_demand,request,
				context,engine->kv_physical_page_capacity) == 0u )
			{
				if ( resident_bound == 0u )
					SparkModelBatchReleaseResidentSlot(engine,request);
				continue;
			}
		}
		/* Prefill lane concentration (the B16 incident fix): the round
		 * budget is CONSUMED by the lanes selected - a long-prompt lane
		 * absorbs the whole budget (one full-width frame per submission
		 * instead of N one-row frames, each a full weight pass); short
		 * spans leave budget for more lanes in the same submission.
		 * Single-request workloads are unchanged (one lane consumes it). */
		engine->scratch_request_slots[selected++] = slot;
		if ( prefill_span_budget != 0 )
		{
			uint32_t span = SparkModelBatchPrefillSpan(engine,request);
			*prefill_span_budget = span < *prefill_span_budget ? *prefill_span_budget - span : 0u;
			if ( *prefill_span_budget == 0u )
				break;
		}
	}
	return(selected);
}

static void SparkModelBatchUpdateRequestAges(
	SparkModelBatchEngine *engine,
	uint32_t state,
	uint32_t selected)
{
	uint32_t index,lane;
	for (index=0u; index<engine->request_capacity; index++)
		if ( engine->requests[index].state == state && engine->requests[index].scheduling_bypass_count != UINT32_MAX )
			engine->requests[index].scheduling_bypass_count++;
	for (lane=0u; lane<selected; lane++)
		engine->requests[engine->scratch_request_slots[lane]].scheduling_bypass_count = 0u;
}

static uint32_t SparkModelBatchSelectRequests(
	SparkModelBatchEngine *engine,
	uint32_t work_kind)
{
	SparkModelBatchCacheDemand cache_demand;
	SparkModelBatchCacheDemand *cache_demand_pointer;
	SparkModelBatchRequestState *request;
	uint32_t index,inflight_by_kind[4],lane_limit,maximum_by_kind[4],maximum_priority,queued_by_kind[4],selected,state;
	uint32_t prefill_span_budget;
	maximum_priority = 0u;
	state = SparkModelBatchStateForWork(work_kind);
	SparkModelBatchCountDispatchableRequests(engine,queued_by_kind);
	SparkModelBatchCountInflightSubmissions(engine,inflight_by_kind);
	for (index=0u; index<engine->request_capacity; index++)
	{
		request = &engine->requests[index];
		if ( request->state == state )
		{
			if ( request->scheduling_bypass_count < engine->submission_capacity && request->priority > maximum_priority )
				maximum_priority = request->priority;
		}
	}
	if ( queued_by_kind[work_kind] == 0u || engine->inflight_submission_count >= engine->submission_capacity )
		return(0u);
	cache_demand_pointer = 0;
	if ( work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE &&
		engine->kv_physical_page_capacity != 0u )
	{
		SparkModelBatchBeginCacheDemand(engine);
		if ( SparkModelBatchBuildInflightCacheDemand(engine,&cache_demand) == 0u )
			return(0u);
		cache_demand_pointer = &cache_demand;
	}
	memset(maximum_by_kind,0,sizeof(maximum_by_kind));
	maximum_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL] = SparkModelBatchMaximumLaneCount(engine,SPARK_MODEL_SERVING_WORK_KIND_PREFILL);
	maximum_by_kind[SPARK_MODEL_SERVING_WORK_KIND_DECODE] = SparkModelBatchMaximumLaneCount(engine,SPARK_MODEL_SERVING_WORK_KIND_DECODE);
	maximum_by_kind[SPARK_MODEL_SERVING_WORK_KIND_RELEASE] = SparkModelBatchMaximumLaneCount(engine,SPARK_MODEL_SERVING_WORK_KIND_RELEASE);
	lane_limit = SparkModelBatchSchedulerPlanMixedLaneCount(queued_by_kind,maximum_by_kind,inflight_by_kind,work_kind,engine->submission_capacity);
	prefill_span_budget = engine->max_prefill_rows;
	selected = SparkModelBatchSelectRequestPass(engine,cache_demand_pointer,state,
		work_kind,lane_limit,SPARK_MODEL_BATCH_SELECT_AGED,maximum_priority,0u,
		work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? &prefill_span_budget : 0);
	selected = SparkModelBatchSelectRequestPass(engine,cache_demand_pointer,state,
		work_kind,lane_limit,SPARK_MODEL_BATCH_SELECT_PRIORITY,maximum_priority,
		selected,
		work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? &prefill_span_budget : 0);
	selected = SparkModelBatchSelectRequestPass(engine,cache_demand_pointer,state,
		work_kind,lane_limit,SPARK_MODEL_BATCH_SELECT_FILL,maximum_priority,
		selected,
		work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? &prefill_span_budget : 0);
	SparkModelBatchUpdateRequestAges(engine,state,selected);
	if ( selected != 0u )
		engine->next_request_scan = (engine->scratch_request_slots[selected - 1u] + 1u) % engine->request_capacity;
	engine->selected_kv_page_count = cache_demand_pointer != 0 ?
		cache_demand.page_count : engine->inflight_kv_page_count;
	return(selected);
}

static uint32_t SparkModelBatchAssignPrefillCounts(
	SparkModelBatchEngine *engine,
	uint32_t lane_count)
{
	uint32_t assigned,lane,remaining_prompt,remaining_rows,row_budget,total_remaining;
	total_remaining = 0u;
	for (lane=0u; lane<lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		request = &engine->requests[engine->scratch_request_slots[lane]];
		remaining_prompt = SparkModelBatchPrefillSpan(engine,request);
		total_remaining = remaining_prompt <= UINT32_MAX - total_remaining ? total_remaining + remaining_prompt : UINT32_MAX;
	}
	/* Continuous batching: the whole ready set gets the row budget up
	 * front, capped only by max_prefill_rows. The old ladder planner
	 * reserved floor rows to seed later fixed-size groups, holding ready
	 * work back; the remainder now chunk-prefills on the next pass. */
	row_budget = total_remaining < engine->max_prefill_rows ?
		total_remaining : engine->max_prefill_rows;
	if ( row_budget < lane_count )
		row_budget = lane_count;
	remaining_rows = row_budget - lane_count;
	for (lane=0u; lane<lane_count; lane++)
		engine->scratch_prefill_counts[lane] = 1u;
	while ( remaining_rows != 0u )
	{
		SparkModelBatchRequestState *request;
		assigned = 0u;
		for (lane=0u; lane<lane_count && remaining_rows!=0u; lane++)
		{
			request = &engine->requests[engine->scratch_request_slots[lane]];
			remaining_prompt = SparkModelBatchPrefillSpan(engine,request);
			if ( engine->scratch_prefill_counts[lane] < remaining_prompt )
			{
				engine->scratch_prefill_counts[lane]++;
				remaining_rows--;
				assigned++;
			}
		}
		if ( assigned == 0u )
			break;
	}
	return(row_budget - remaining_rows);
}

static void SparkModelBatchInitializeSubmission(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission,
	uint32_t work_kind,
	uint32_t lane_count)
{
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = work_kind;
	submission->submission_id = engine->next_submission_id;
	submission->request_id = engine->next_submission_id;
	submission->sequence_id = engine->next_submission_id;
	submission->control_generation = 1u;
	submission->transaction_id = engine->next_submission_id;
	submission->dispatch_generation = engine->next_submission_id;
	submission->request_generation = 1u;
	submission->step_generation = engine->next_submission_id;
	submission->residency.word0 = engine->next_submission_id;
	submission->residency.word1 = engine->next_submission_id ^ UINT64_C(0x535041524b504950);
	submission->residency.generation = engine->next_submission_id;
	submission->residency.owner = 1u;
	submission->active_sequence_count = lane_count;
	submission->lane_count = lane_count;
	submission->lanes = engine->scratch_lanes;
	submission->tokens_per_sequence = work_kind ==
		SPARK_MODEL_SERVING_WORK_KIND_RELEASE ? 0u : 1u;
}

static void SparkModelBatchInitializeLane(
	SparkModelBatchEngine *engine,
	SparkModelServingLane *lane,
	uint32_t request_slot,
	uint64_t sequence_position,
	uint32_t context_token_count,
	uint32_t input_token_id,
	uint32_t flags,
	uint32_t allow_cache)
{
	SparkModelBatchRequestState *request;
	SparkSha256Context publish_context;
	uint32_t publish_token_count;
	uint32_t *tokens;
	request = &engine->requests[request_slot];
	memset(lane,0,sizeof(*lane));
	lane->request_id = request->request_id;
	lane->request_generation = request->generation;
	lane->step_generation = engine->next_submission_id;
	lane->sequence_id = request->sequence_id;
	lane->sequence_position = sequence_position;
	lane->resident_sequence_slot = request->resident_sequence_slot;
	lane->context_token_count = context_token_count;
	lane->input_token_id = input_token_id;
	lane->flags = flags;
	if ( allow_cache == 0u || engine->cache_block_token_count == 0u )
		return;
	if ( request->cache_prefix_token_count != 0u &&
		request->computed_prompt_token_count == request->cache_prefix_token_count &&
		request->computed_prompt_token_count < request->prompt_token_count )
	{
		lane->flags |= SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX;
		lane->cache_prefix_token_count = request->cache_prefix_token_count;
		lane->cache_prefix_identity = request->cache_prefix_identity;
	}
	publish_token_count = (context_token_count / engine->cache_block_token_count) * engine->cache_block_token_count;
	if ( publish_token_count <= request->cache_published_token_count )
		return;
	publish_context = request->cache_published_digest_context;
	tokens = SparkModelBatchRequestTokens(engine,request_slot);
	SparkModelBatchDigestTokens(&publish_context,&tokens[request->cache_published_token_count],publish_token_count - request->cache_published_token_count);
	lane->flags |= SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH;
	lane->cache_publish_token_count = publish_token_count;
	SparkModelBatchFinalizeIdentity(&publish_context,&lane->cache_publish_identity);
}

static void SparkModelBatchBuildPrefillRows(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission,
	uint32_t lane_count)
{
	uint32_t lane,row,source,slot,wave,maximum;
	maximum = 0u;
	for (lane=0u; lane<lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		uint32_t *tokens;
		slot = engine->scratch_request_slots[lane];
		request = &engine->requests[slot];
		tokens = SparkModelBatchRequestTokens(engine,slot);
		SparkModelBatchInitializeLane(engine,&engine->scratch_lanes[lane],slot,request->computed_prompt_token_count,request->computed_prompt_token_count + engine->scratch_prefill_counts[lane],tokens[request->computed_prompt_token_count],request->computed_prompt_token_count + engine->scratch_prefill_counts[lane] == request->prompt_token_count ? SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN : 0u,1u);
		if ( engine->scratch_prefill_counts[lane] > maximum )
			maximum = engine->scratch_prefill_counts[lane];
	}
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<lane_count; lane++)
			if ( engine->scratch_prefill_counts[lane] > wave )
		{
			SparkModelBatchRequestState *request;
			uint32_t *tokens;
			slot = engine->scratch_request_slots[lane];
			request = &engine->requests[slot];
			tokens = SparkModelBatchRequestTokens(engine,slot);
			source = request->computed_prompt_token_count + wave;
			engine->scratch_token_ids[row] = tokens[source];
			engine->scratch_row_lane_indices[row] = lane;
			engine->scratch_row_positions[row] = source;
			engine->scratch_row_sequence_ids[row] = request->sequence_id;
			row++;
		}
	submission->row_count = row;
	submission->token_count = row;
	submission->new_token_count = row;
}

static void SparkModelBatchBuildDecodeRows(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission,
	uint32_t lane_count)
{
	uint32_t block_remaining,chain_tokens,lane,remaining,slot,position;
	chain_tokens = 1u;
	if ( (engine->adapter_descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN) != 0u )
	{
		chain_tokens = engine->adapter_descriptor->max_output_token_count /
			lane_count;
		if ( chain_tokens > SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE )
			chain_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE;
	}
	for (lane=0u; lane<lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		uint32_t *tokens;
		slot = engine->scratch_request_slots[lane];
		request = &engine->requests[slot];
		tokens = SparkModelBatchRequestTokens(engine,slot);
		position = request->prompt_token_count + request->generated_token_count - 1u;
		remaining = request->output_token_budget -
			request->generated_token_count;
		if ( remaining < chain_tokens )
			chain_tokens = remaining;
		remaining = engine->max_context_tokens -
			(request->prompt_token_count + request->generated_token_count);
		if ( remaining < chain_tokens )
			chain_tokens = remaining;
		if ( engine->cache_block_token_count != 0u )
		{
			block_remaining = engine->cache_block_token_count -
				(position % engine->cache_block_token_count);
			if ( block_remaining < chain_tokens )
				chain_tokens = block_remaining;
		}
		SparkModelBatchInitializeLane(engine,&engine->scratch_lanes[lane],slot,position,position + 1u,tokens[position],SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN,1u);
		engine->scratch_token_ids[lane] = tokens[position];
		engine->scratch_row_lane_indices[lane] = lane;
		engine->scratch_row_positions[lane] = position;
		engine->scratch_row_sequence_ids[lane] = request->sequence_id;
	}
	submission->row_count = lane_count;
	submission->token_count = lane_count;
	submission->new_token_count = lane_count;
	submission->tokens_per_sequence = chain_tokens;
}

static void SparkModelBatchBuildReleaseLanes(
	SparkModelBatchEngine *engine,
	uint32_t lane_count)
{
	uint32_t lane,slot,position;
	for (lane=0u; lane<lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		slot = engine->scratch_request_slots[lane];
		request = &engine->requests[slot];
		position = request->prompt_token_count + request->generated_token_count;
		SparkModelBatchInitializeLane(engine,&engine->scratch_lanes[lane],slot,position,position,0u,0u,0u);
	}
}

static void SparkModelBatchFinishSubmissionShape(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission)
{
	uint32_t lane;
	submission->token_ids = submission->row_count != 0u ? engine->scratch_token_ids : 0;
	submission->row_lane_indices = submission->row_count != 0u ? engine->scratch_row_lane_indices : 0;
	submission->row_positions = submission->row_count != 0u ? engine->scratch_row_positions : 0;
	submission->row_sequence_ids = submission->row_count != 0u ? engine->scratch_row_sequence_ids : 0;
	for (lane=0u; lane<submission->lane_count; lane++)
		if ( engine->requests[engine->scratch_request_slots[lane]].priority > submission->priority )
			submission->priority = engine->requests[engine->scratch_request_slots[lane]].priority;
}

static uint32_t SparkModelBatchBuildSubmission(
	SparkModelBatchEngine *engine,
	uint32_t work_kind,
	SparkModelServingSubmission *submission)
{
	uint32_t lane_count;
	lane_count = SparkModelBatchSelectRequests(engine,work_kind);
	if ( lane_count == 0u )
		return(0u);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		(void)SparkModelBatchAssignPrefillCounts(engine,lane_count);
	SparkModelBatchInitializeSubmission(engine,submission,work_kind,lane_count);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		SparkModelBatchBuildPrefillRows(engine,submission,lane_count);
	else if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		SparkModelBatchBuildDecodeRows(engine,submission,lane_count);
	else
		SparkModelBatchBuildReleaseLanes(engine,lane_count);
	SparkModelBatchFinishSubmissionShape(engine,submission);
	return(lane_count);
}

static void SparkModelBatchRecordSubmission(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *state,
	uint32_t lane_count)
{
	uint32_t *request_slots,*prefill_counts;
	uint32_t lane,inflight_state;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,state);
	prefill_counts = SparkModelBatchSubmissionPrefillCounts(engine,state);
	state->lane_count = lane_count;
	state->submission_id = engine->next_submission_id;
	inflight_state = SparkModelBatchInflightStateForWork(state->work_kind);
	for (lane=0u; lane<lane_count; lane++)
	{
		request_slots[lane] = engine->scratch_request_slots[lane];
		prefill_counts[lane] = state->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? engine->scratch_prefill_counts[lane] : 0u;
		engine->requests[request_slots[lane]].state = inflight_state;
	}
	engine->inflight_submission_count++;
	engine->inflight_kv_page_count = engine->selected_kv_page_count;
}

static SparkStatus SparkModelBatchDispatchKind(
	SparkModelBatchEngine *engine,
	uint32_t work_kind,
	uint32_t *dispatched_out)
{
	SparkModelBatchSubmissionState *state;
	SparkModelServingSubmission submission;
	SparkStatus status;
	uint32_t lane_count;
	*dispatched_out = 0u;
	state = SparkModelBatchReserveSubmission(engine,work_kind);
	if ( state == 0 )
		return(SPARK_STATUS_BUSY);
	engine->next_submission_id++;
	if ( engine->next_submission_id == 0u )
	{
		state->active = 0u;
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	lane_count = SparkModelBatchBuildSubmission(engine,work_kind,&submission);
	if ( lane_count == 0u )
	{
		state->active = 0u;
		return(SPARK_STATUS_NOT_FOUND);
	}
	status = SparkModelPipelineClientSubmit(engine->pipeline,&submission);
	if ( status != SPARK_STATUS_OK )
	{
		state->active = 0u;
		return(status);
	}
	SparkModelBatchRecordSubmission(engine,state,lane_count);
	*dispatched_out = 1u;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelBatchChooseWorkKind(
	SparkModelBatchEngine *engine)
{
	SparkModelBatchRequestState *request;
	uint32_t available_by_kind[4],available_unbound_prefill,index,inflight_by_kind[4],kind,maximum_by_kind[4],minimum_by_kind[4],queued_by_kind[4],remaining_prompt;
	memset(available_by_kind,0,sizeof(available_by_kind));
	available_unbound_prefill = engine->free_resident_slot_count;
	for (index=0u; index<engine->request_capacity; index++)
	{
		request = &engine->requests[index];
		if ( request->state == SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL )
		{
			if ( request->resident_sequence_slot == SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT )
			{
				if ( available_unbound_prefill == 0u )
					continue;
				available_unbound_prefill--;
			}
			remaining_prompt = request->prompt_token_count - request->computed_prompt_token_count;
			available_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL] = remaining_prompt <= UINT32_MAX - available_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL] ? available_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL] + remaining_prompt : UINT32_MAX;
		}
		else if ( request->state == SPARK_MODEL_BATCH_REQUEST_READY_DECODE )
			available_by_kind[SPARK_MODEL_SERVING_WORK_KIND_DECODE]++;
		else if ( request->state == SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE )
			available_by_kind[SPARK_MODEL_SERVING_WORK_KIND_RELEASE]++;
	}
	memset(minimum_by_kind,0,sizeof(minimum_by_kind));
	/* Continuous admission: no minimum-efficient floor. Every nonzero
	 * ready set dispatches on the next Progress (the old B-ladder deferred
	 * small queues until minimum_efficient_submission_row_count work was
	 * pending or the request aged out - waiting to fill a fixed bucket). */
	minimum_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL] = 1u;
	minimum_by_kind[SPARK_MODEL_SERVING_WORK_KIND_DECODE] = 1u;
	minimum_by_kind[SPARK_MODEL_SERVING_WORK_KIND_RELEASE] = 1u;
	SparkModelBatchCountDispatchableRequests(engine,queued_by_kind);
	SparkModelBatchCountInflightSubmissions(engine,inflight_by_kind);
	memset(maximum_by_kind,0,sizeof(maximum_by_kind));
	for (kind=SPARK_MODEL_SERVING_WORK_KIND_PREFILL; kind<=SPARK_MODEL_SERVING_WORK_KIND_RELEASE; kind++)
		maximum_by_kind[kind] = SparkModelBatchMaximumLaneCount(engine,kind);
	for (kind=SPARK_MODEL_SERVING_WORK_KIND_PREFILL; kind<=SPARK_MODEL_SERVING_WORK_KIND_RELEASE; kind++)
	{
		if ( SparkModelBatchSchedulerPlanMixedLaneCount(queued_by_kind,maximum_by_kind,inflight_by_kind,kind,engine->submission_capacity) == 0u )
			available_by_kind[kind] = 0u;
	}
	return(SparkModelBatchSchedulerChooseWorkKind(available_by_kind,minimum_by_kind,engine->admission_open,engine->inflight_submission_count,engine->submission_capacity,&engine->next_work_kind,engine->work_kind_bypass_counts));
}

static void SparkModelBatchFailIdleRequests(
	SparkModelBatchEngine *engine,
	SparkStatus status)
{
	uint32_t index,state;
	for (index=0u; index<engine->request_capacity; index++)
	{
		state = engine->requests[index].state;
		if ( state != SPARK_MODEL_BATCH_REQUEST_FREE && state != SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT && state != SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT && state != SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT )
			SparkModelBatchFailRequest(engine,&engine->requests[index],status);
	}
}

SparkStatus SparkModelBatchEngineProgress(
	SparkModelBatchEngine *engine,
	uint32_t maximum_new_submission_count)
{
	SparkStatus status;
	uint32_t dispatched,kind,misses,step;
	if ( engine == 0 || maximum_new_submission_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelPipelineClientProgress(engine->pipeline,engine->maximum_messages_per_rank);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelBatchSetFailed(engine,status);
		SparkModelBatchFailIdleRequests(engine,status);
		return(status);
	}
	if ( engine->failed_status != SPARK_STATUS_OK )
	{
		SparkModelBatchFailIdleRequests(engine,(SparkStatus)engine->failed_status);
		return((SparkStatus)engine->failed_status);
	}
	SparkModelBatchRefreshInflightKvPageCount(engine);
	step = 0u;
	misses = 0u;
	while ( step < maximum_new_submission_count && engine->inflight_submission_count < engine->submission_capacity && misses < 3u )
	{
		kind = SparkModelBatchChooseWorkKind(engine);
		if ( kind == 0u )
			break;
		status = SparkModelBatchDispatchKind(engine,kind,&dispatched);
		if ( status == SPARK_STATUS_BUSY )
			break;
		if ( status == SPARK_STATUS_NOT_FOUND )
		{
			misses++;
			continue;
		}
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelBatchSetFailed(engine,status);
			SparkModelBatchFailIdleRequests(engine,status);
			return(status);
		}
		step++;
		misses = 0u;
	}
	/* Write-through contract (p1d2 step-loop): the pipeline client flushes
	 * every submission and decision to the wire when it is queued, so the
	 * dispatch loop above already put this step's work on the wire — no
	 * compensating flush Progress here (the old trailing call existed to
	 * drain a queue that only Progress could send; that bubble class is
	 * designed out at the client). This Progress is a pure drain. */
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineCloseAdmission(
	SparkModelBatchEngine *engine)
{
	if ( engine == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	engine->admission_open = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineReopenAdmission(
	SparkModelBatchEngine *engine)
{
	if ( engine == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	engine->admission_open = 1u;
	engine->failed_status = SPARK_STATUS_OK;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineBeginShutdown(
	SparkModelBatchEngine *engine)
{
	SparkModelBatchRequestState *request;
	uint32_t index;
	if ( engine == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	engine->admission_open = 0u;
	for (index=0u; index<engine->request_capacity; index++)
	{
		request = &engine->requests[index];
		if ( request->state == SPARK_MODEL_BATCH_REQUEST_FREE )
			continue;
		if ( request->state == SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_COMPLETING )
			request->cancel_pending = 1u;
		else if ( request->state == SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE )
		{
			request->terminal_event_kind = SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED;
			request->terminal_status = SPARK_STATUS_OK;
		}
		else
			SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED,SPARK_STATUS_OK);
	}
	return(engine->live_request_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_PENDING);
}

SparkStatus SparkModelBatchEngineGetPollDescriptors(
	const SparkModelBatchEngine *engine,
	SparkModelResidentClientPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out)
{
	if ( engine == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkModelPipelineClientGetPollDescriptors(engine->pipeline,descriptors,descriptor_capacity,descriptor_count_out));
}

static void SparkModelBatchCountStates(
	const SparkModelBatchEngine *engine,
	SparkModelBatchEngineView *view)
{
	uint32_t index;
	for (index=0u; index<engine->request_capacity; index++)
	{
		if ( engine->requests[index].state == SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL )
			view->queued_prefill_count++;
		if ( engine->requests[index].state == SPARK_MODEL_BATCH_REQUEST_READY_DECODE )
			view->ready_decode_count++;
	}
}

SparkStatus SparkModelBatchEngineGetView(
	const SparkModelBatchEngine *engine,
	SparkModelBatchEngineView *view)
{
	SparkStatus status;
	uint32_t index;
	if ( engine == 0 || view == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	view->descriptor_bytes = SPARK_MODEL_BATCH_ENGINE_VIEW_BYTES;
	view->admission_open = engine->admission_open;
	view->request_capacity = engine->request_capacity;
	view->live_request_count = engine->live_request_count;
	view->inflight_submission_count = engine->inflight_submission_count;
	view->inflight_kv_lane_count = 0u;
	for (index=0u; index<engine->submission_capacity; index++)
		if ( engine->submissions[index].active != 0u &&
			engine->submissions[index].work_kind !=
			SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			view->inflight_kv_lane_count +=
				engine->submissions[index].lane_count;
	view->inflight_kv_page_count = engine->inflight_kv_page_count;
	view->kv_physical_page_capacity = engine->kv_physical_page_capacity;
	view->kv_logical_page_capacity = engine->kv_logical_page_capacity;
	view->failed_status = engine->failed_status;
	view->submitted_request_count = engine->submitted_request_count;
	view->completed_request_count = engine->completed_request_count;
	view->cancelled_request_count = engine->cancelled_request_count;
	view->emitted_token_count = engine->emitted_token_count;
	SparkModelBatchCountStates(engine,view);
	status = SparkModelPipelineClientGetView(engine->pipeline,&view->pipeline);
	return(status);
}

const SparkModelServingAdapterDescriptor *SparkModelBatchEngineGetAdapterDescriptor(
	const SparkModelBatchEngine *engine)
{
	return(engine != 0 ? engine->adapter_descriptor : 0);
}
