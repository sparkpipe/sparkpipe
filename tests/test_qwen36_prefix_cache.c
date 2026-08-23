#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen36_model.h"
#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
/* Compiled in for the F1 conservation case (audit F3): the gate drives
 * the paged-KV unit directly, at the adapter's exact per-round call
 * pattern, and asserts free-list conservation after EVERY step. */
#include "spark_qwen36_paged_kv.h"

#ifndef TEST_QWEN36_SERVING_ADAPTER_PATH
#define TEST_QWEN36_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_QWEN36_SERVING_DRIVER_PATH
#define TEST_QWEN36_SERVING_DRIVER_PATH ""
#endif
#ifndef TEST_QWEN36_SERVING_CONFIG_PATH
#define TEST_QWEN36_SERVING_CONFIG_PATH ""
#endif

/*
 * Prefix-cache correctness gate for the qwen36 serving adapter, run
 * against the recording fixture driver.
 *
 * Proven at the serving boundary:
 *   1. B1 identical prompts back-to-back: the second sequence shares the
 *      donor block-table row verbatim, its first frame carries
 *      PREFIX_RESUME at a block boundary naming a live checkpoint slot,
 *      and no frame walks any position below that boundary - zero
 *      recompute of the shared prefix.
 *   2. B1 diverging prompts: reuse stops at the longest common COMPLETE
 *      block (block-granular by construction).
 *   3. ON vs OFF: OFF walks every position; ON feeds the SAME tokens at
 *      the SAME positions for every walked row and skips exactly the
 *      shared prefix - the byte-identity argument the hardware E2E run
 *      closes (the kernels are already bit-exactness gated).
 *   4. B4 mixed and B25 pressure: shared AND diverging prefixes across
 *      many lanes, checkpoint-slot churn shortening matches without ever
 *      mis-shaping them, pool exhaustion refusing cleanly.
 *   5. Deployment wiring: kv_logical/physical_page_capacity size the
 *      module KV pool (observed through the driver env channel) and are
 *      validated like the driver contract.
 *   6. No speculative-state pollution: no record ever shows drafter or
 *      snapshot state crossing lanes - resume frames name their OWN lane
 *      slot, and checkpoint slots are per-boundary, never shared between
 *      sequences within a round.
 *   7. DECODE work with speculation (audit F3): multi-round B1 spec
 *      residency over the paged KV - per round Cover(end=pos+1) plus the
 *      speculative extension Cover(end=pos+D+2) - across two resident
 *      lanes and block-boundary crossings. The lane's attached block row
 *      must be STABLE across rounds (scratch persistence): a round that
 *      re-borrows over slots still naming last round's scratch is the F1
 *      orphaning mechanism and fails here.
 *   8. F1 free-list conservation as a permanent unit-level case: driving
 *      spark_qwen36_paged_kv.c directly (the audit-probe pattern),
 *      {core free list} U {lane rows} U {live sequences} U {core LRU}
 *      == pool with free-list disjointness and row hygiene after EVERY
 *      Cover call, >64 simulated B1 rounds crossing >=2 block boundaries,
 *      reuse ON and OFF, plus conserving teardown.
 *   9. F2 feasibility accounting: up-front refusal must precede any
 *      Cover; a passing feasibility check must never fail mid-coverage.
 *  10. S512 scale (matrix top end): the driver's maximum declared batch
 *      size B512 with shared AND diverging prefixes, mid-block admits,
 *      eviction pressure, and spec rounds crossing block boundaries -
 *      the F1 conservation invariant after every Cover at that B.
 */

#define GATE_CAPTURE_ROWS 32u
#define GATE_MAX_ROWS 512u
#define GATE_MAX_LANES 32u

typedef struct GateFrameRecord
{
	uint32_t prefill;
	uint32_t rows;
	uint32_t context_flags;
	uint32_t snapshot_index_present;
	uint32_t snapshot_index;
	uint32_t lane_index;
	uint64_t base_position;
	uint64_t sequence_id;
	uint32_t block_count;
	uint32_t token_ids[GATE_CAPTURE_ROWS];
	uint32_t blocks[GATE_CAPTURE_ROWS];
}
GateFrameRecord;

typedef uint32_t (*GateRecordCountFunction)(void);
typedef const GateFrameRecord *(*GateRecordGetFunction)(uint32_t);
typedef void (*GateRecordResetFunction)(void);
typedef uint32_t (*GateKvBlockCountFunction)(void);
static GateRecordCountFunction gate_record_count;
static GateRecordGetFunction gate_record_get;
static GateRecordResetFunction gate_record_reset;
static GateKvBlockCountFunction gate_kv_blocks;

static const SparkModelServingAdapterInterface *adapter_interface_global;

typedef struct GateState
{
	uint32_t completion_count;
	void *execution_stream;
	SparkModelServingCompletion completion;
}
GateState;

static void GateCompletion(void *completion_context, const SparkModelServingCompletion *completion)
{
	GateState *state = (GateState *)completion_context;
	assert(state != 0 && completion != 0);
	state->completion = *completion;
	state->completion_count++;
}

static void GateConfiguration(SparkModelServingAdapterConfiguration *configuration,
	const char *config_path, const char *runtime_root, GateState *test_state,
	uint32_t active_lanes, uint32_t resident_capacity,
	uint32_t logical_pages, uint32_t physical_pages)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = 3u;
	configuration->stage_index = 3u;
	configuration->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration->runtime_limits.max_inflight_submission_count = 2u;
	configuration->runtime_limits.max_active_sequence_count = active_lanes;
	configuration->runtime_limits.max_input_row_count = GATE_MAX_ROWS;
	configuration->runtime_limits.resident_sequence_capacity = resident_capacity;
	configuration->runtime_limits.kv_logical_page_capacity = logical_pages;
	configuration->runtime_limits.kv_physical_page_capacity = physical_pages;
	configuration->runtime_root = runtime_root;
	configuration->node_id = "spark-test";
	configuration->node_target = "cuda.sm121.qwen36.resident_decode_stage.bf16";
	configuration->adapter_configuration_path = config_path;
	configuration->driver_shared_object_path = TEST_QWEN36_SERVING_DRIVER_PATH;
	configuration->driver_program_name = "resident_decode";
	configuration->execution_stream = test_state->execution_stream;
	configuration->completion_function = GateCompletion;
	configuration->completion_context = test_state;
}

/* One single-lane prefill submission carrying tokens[0..length) at
 * positions [0..length). */
static SparkStatus GatePrefill(void *adapter_state, GateState *test_state,
	uint32_t slot, uint64_t sequence_id, const uint32_t *tokens,
	uint32_t length, uint64_t submission_id)
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	static __thread uint32_t row_lane[GATE_MAX_ROWS];
	static __thread uint64_t row_positions[GATE_MAX_ROWS],row_sequences[GATE_MAX_ROWS];
	uint32_t row;
	memset(&submission,0,sizeof(submission));
	memset(&lane,0,sizeof(lane));
	for (row=0u; row<length; row++)
	{
		row_lane[row] = 0u;
		row_positions[row] = row;
		row_sequences[row] = sequence_id;
	}
	lane.sequence_id = sequence_id;
	lane.request_id = submission_id + 1000u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.context_token_count = length;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = submission_id;
	submission.request_id = submission_id + 1000u;
	submission.sequence_id = sequence_id;
	submission.control_generation = 1u;
	submission.transaction_id = submission_id + 2000u;
	submission.dispatch_generation = submission_id + 3000u;
	submission.request_generation = 1u;
	submission.step_generation = 1u;
	submission.residency.word0 = submission_id;
	submission.residency.word1 = 177u;
	submission.residency.generation = 277u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = 1u;
	submission.new_token_count = length;
	submission.lane_count = 1u;
	submission.row_count = length;
	submission.token_count = length;
	submission.lanes = &lane;
	submission.token_ids = tokens;
	submission.row_positions = row_positions;
	submission.row_lane_indices = row_lane;
	submission.row_sequence_ids = row_sequences;
	test_state->completion_count = 0u;
	return(adapter_interface_global->submit(adapter_state,&submission));
}

/* Same single-lane prefill shape, plus the client CACHE_PUBLISH contract
 * (deterministic token-derived identity + block-aligned token count) - the
 * lane glue that arms the adapter's publish-boundary frame. CASE 11 needs
 * this because the checkpoint/publish collision it guards only exists on
 * a lane that publishes. */
static SparkStatus GatePrefillPublish(void *adapter_state, GateState *test_state,
	uint32_t slot, uint64_t sequence_id, const uint32_t *tokens,
	uint32_t length, uint64_t submission_id)
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	static __thread uint32_t pub_row_lane[GATE_MAX_ROWS];
	static __thread uint64_t pub_row_positions[GATE_MAX_ROWS],pub_row_sequences[GATE_MAX_ROWS];
	uint32_t row,boundary;
	memset(&submission,0,sizeof(submission));
	memset(&lane,0,sizeof(lane));
	for (row=0u; row<length; row++)
	{
		pub_row_lane[row] = 0u;
		pub_row_positions[row] = row;
		pub_row_sequences[row] = sequence_id;
	}
	lane.sequence_id = sequence_id;
	lane.request_id = submission_id + 1000u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.context_token_count = length;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	boundary = length - (length % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	if ( boundary == 0u )
		boundary = length;
	lane.flags |= SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH;
	lane.cache_publish_token_count = boundary;
	for (row=0u; row<32u; row++)
		lane.cache_publish_identity.sha256[row] =
			(uint8_t)(tokens[row % length] * 31u + row);
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = submission_id;
	submission.request_id = submission_id + 1000u;
	submission.sequence_id = sequence_id;
	submission.control_generation = 1u;
	submission.transaction_id = submission_id + 2000u;
	submission.dispatch_generation = submission_id + 3000u;
	submission.request_generation = 1u;
	submission.step_generation = 1u;
	submission.residency.word0 = submission_id;
	submission.residency.word1 = 177u;
	submission.residency.generation = 277u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = 1u;
	submission.new_token_count = length;
	submission.lane_count = 1u;
	submission.row_count = length;
	submission.token_count = length;
	submission.lanes = &lane;
	submission.token_ids = tokens;
	submission.row_positions = pub_row_positions;
	submission.row_lane_indices = pub_row_lane;
	submission.row_sequence_ids = pub_row_sequences;
	test_state->completion_count = 0u;
	return(adapter_interface_global->submit(adapter_state,&submission));
}

/* One single-lane DECODE submission: one row at `position` carrying the
 * previously accepted token id, OUTPUT_TOKEN set - the B1 shape the
 * speculative path requires. */
static SparkStatus GateDecode(void *adapter_state, GateState *test_state,
	uint32_t slot, uint64_t sequence_id, uint64_t position,
	uint32_t token_id, uint64_t submission_id)
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	uint32_t token_buffer;
	uint32_t row_lane;
	uint64_t row_position,row_sequence;
	memset(&submission,0,sizeof(submission));
	memset(&lane,0,sizeof(lane));
	token_buffer = token_id;
	row_lane = 0u;
	row_position = position;
	row_sequence = sequence_id;
	lane.sequence_id = sequence_id;
	lane.sequence_position = position;
	lane.request_id = submission_id + 1000u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.context_token_count = position + 1u;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = submission_id;
	submission.request_id = submission_id + 1000u;
	submission.sequence_id = sequence_id;
	submission.sequence_position = position;
	submission.control_generation = 1u;
	submission.transaction_id = submission_id + 2000u;
	submission.dispatch_generation = submission_id + 3000u;
	submission.request_generation = 1u;
	submission.step_generation = 1u;
	submission.residency.word0 = submission_id;
	submission.residency.word1 = 178u;
	submission.residency.generation = 277u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = 1u;
	submission.new_token_count = 1u;
	submission.lane_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	submission.lanes = &lane;
	submission.token_ids = &token_buffer;
	submission.row_positions = &row_position;
	submission.row_lane_indices = &row_lane;
	submission.row_sequence_ids = &row_sequence;
	test_state->completion_count = 0u;
	return(adapter_interface_global->submit(adapter_state,&submission));
}


/* Walk the records [from,to): assert the lane walked EXACTLY positions
 * [expected_start,length) with the expected tokens, and return the number
 * of PREFIX_RESUME frames seen (0 or 1). */
static uint32_t GateAssertWalk(const char *label, uint32_t from, uint32_t to,
	uint32_t module_lane, const uint32_t *tokens, uint32_t expected_start,
	uint32_t length, uint32_t expect_resume, uint32_t shared_prefix_blocks,
	const GateFrameRecord *donor_tail)
{
	uint32_t index,resume_count,walked_cursor;
	(void)label;
	walked_cursor = expected_start;
	resume_count = 0u;
	for (index=from; index<to; index++)
	{
		const GateFrameRecord *record = gate_record_get(index);
		uint32_t row;
		fprintf(stderr,"GATEWALK idx=%u lane=%u base=%llu cursor=%u rows=%u flags=%x snap=%u\n",
			index,(unsigned)record->lane_index,(unsigned long long)record->base_position,
			walked_cursor,record->rows,record->context_flags,record->snapshot_index);
		assert(record != 0 && record->prefill != 0u);
		assert(record->lane_index == module_lane);
		if ( (record->context_flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME) != 0u )
		{
			resume_count++;
			assert(record->base_position == walked_cursor);
			assert(record->snapshot_index_present != 0u);
			assert(record->snapshot_index < SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS);
		}
		else
			assert(record->snapshot_index_present == 0u ||
				(record->context_flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT) != 0u);
		assert(record->base_position == walked_cursor);
		assert(record->rows != 0u && walked_cursor + record->rows <= length);
		for (row=0u; row<record->rows && row<GATE_CAPTURE_ROWS; row++)
			assert(record->token_ids[row] == tokens[walked_cursor + row]);
		if ( (record->context_flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT) != 0u )
			assert((record->base_position + record->rows) % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS == 0u);
		walked_cursor += record->rows;
	}
	assert(walked_cursor == length);
	assert(resume_count == expect_resume);
	if ( expect_resume != 0u )
	{
		/* The resumed lane shares the donor published blocks verbatim. */
		const GateFrameRecord *first = gate_record_get(from);
		uint32_t block;
		assert(donor_tail != 0);
		for (block=0u; block<shared_prefix_blocks && block<GATE_CAPTURE_ROWS;
			block++)
			assert(first->blocks[block] == donor_tail->blocks[block]);
	}
	return(resume_count);
}

/* ---- CASE 8 support: F1 free-list conservation, unit level ----
 * Direct host-level driver of spark_qwen36_paged_kv.c (the audit-probe
 * pattern), simulating the adapter's exact per-round B1 speculative
 * decode call pattern: Cover(end = pos + 1) for the plain decode row,
 * then Cover(end = pos + D + 2) for the speculative extension, with NO
 * canonical tokens entering while scratch is outstanding and NO
 * LaneReset between rounds of a continuing residency.
 *
 * Invariant asserted after EVERY Cover call:
 *   {core free list} U {lane rows} U {live sequences} U {core LRU}
 *   == pool
 * with the free list disjoint from the in-use sets and row slots at
 * ordinals >= counts_by_lane[lane] required to be NO_BLOCK. The F1 bug
 * (SyncRow discarding outstanding-scratch records) orphans borrowed
 * blocks - unreachable from every record - and/or leaves stale blocks
 * in row slots beyond counts; both fail here. */

#define F1_LANES 2u
#define F1_BLOCKS_PER_LANE 32u
#define F1_POOL 64u
#define F1_BLOCK_TOKENS 64u
/* >64 rounds; coverage crosses block boundaries twice (rounds ~56 and
 * ~120 with the 130-token prompt). */
#define F1_ROUNDS 140u

static uint32_t f1_table[F1_LANES * F1_BLOCKS_PER_LANE];
static uint32_t f1_counts[F1_LANES];

static void F1Fill(uint32_t *tokens, uint32_t base, uint32_t count)
{
	uint32_t i;
	for (i=0u; i<count; i++)
		tokens[i] = base + i * 7u + 1u;
}

/* THE lane invariant, ONE implementation for every scale (DRY: the F1
 * B1 cases and the B512 scale case assert the same law): {free list} U
 * {lane rows} U {live sequences} U {core LRU} == pool, free list
 * disjoint from every in-use set, row hygiene past counts. Dimensions
 * come from the cache under test; pool/buffers/tag come from the
 * caller. Returns orphan count; pool-cap on a violation. */
static uint32_t GateFreeListConservation(const SparkQwen36PagedKv *cache,
	const char *tag,const char *where,uint32_t step,
	int8_t *in_free,int8_t *in_use,uint32_t pool)
{
	uint32_t i,ordinal,next,orphans;
	memset(in_free,0,(size_t)pool);
	memset(in_use,0,(size_t)pool);
	next = cache->core.free_block_head;
	while (next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		assert(next < pool);
		in_free[next] = 1;
		next = cache->core.blocks[next].free_next;
	}
	/* Detached published blocks sit on the LRU (Trim recovers them). */
	next = cache->core.lru_head;
	while (next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
	{
		assert(next < pool);
		in_use[next] = 1;
		next = cache->core.blocks[next].lru_next;
	}
	for (i=0u; i<cache->configuration.lane_count; i++)
		for (ordinal=0u; ordinal<cache->configuration.blocks_per_lane;
			ordinal++)
		{
			uint32_t block = cache->blocks_by_lane[
				i * cache->configuration.blocks_per_lane + ordinal];
			if ( ordinal >= cache->counts_by_lane[i] )
			{
				if ( block != SPARK_QWEN36_PAGED_KV_NO_BLOCK )
				{
					fprintf(stderr,"%sFAIL[%s step %u]: lane %u ordinal %u >= count %u holds block %u (row hygiene)\n",
						tag,where,step,i,ordinal,
						cache->counts_by_lane[i],block);
					return(pool);
				}
				continue;
			}
			assert(block != SPARK_QWEN36_PAGED_KV_NO_BLOCK &&
				block < pool);
			in_use[block] = 1;
		}
	for (i=0u; i<cache->core.max_sequence_count; i++)
	{
		const SparkPrefixCacheCoreSequence *sequence =
			&cache->core.sequences[i];
		if ( sequence->used == 0u )
			continue;
		for (ordinal=0u; ordinal<sequence->block_count; ordinal++)
		{
			uint32_t block = cache->core.sequence_blocks[
				i * cache->core.sequence_block_capacity + ordinal];
			assert(block < pool);
			in_use[block] = 1;
		}
	}
	orphans = 0u;
	for (i=0u; i<pool; i++)
	{
		if ( in_free[i] != 0 && in_use[i] != 0 )
		{
			fprintf(stderr,"%sFAIL[%s step %u]: block %u on free list AND in use\n",
				tag,where,step,i);
			return(pool);
		}
		if ( in_free[i] == 0 && in_use[i] == 0 )
			orphans++;
	}
	if ( orphans != 0u )
		fprintf(stderr,"%sFAIL[%s step %u]: %u orphaned blocks\n",
			tag,where,step,orphans);
	return(orphans);
}

/* F1 wrapper: same invariant, B1-scale buffers. */
static uint32_t F1ConservationCheck(const SparkQwen36PagedKv *cache,
	const char *where, uint32_t step)
{
	int8_t in_free[F1_POOL],in_use[F1_POOL];
	return(GateFreeListConservation(cache,"F1",where,step,
		in_free,in_use,F1_POOL));
}

/* Shared paged-Kv fixture setup for every unit-level scenario (F1/F2):
 * zeroed configuration + geometry, row slots pre-filled NO_BLOCK and
 * counts zeroed. One place to change when the fixture shape moves - the
 * ports are told to copy these cases verbatim, so keep them DRY. */
static void GatePagedKvFixture(SparkQwen36PagedKvConfiguration *configuration,
	uint32_t *table,uint32_t *counts,uint32_t lane_count,
	uint32_t blocks_per_lane,uint32_t pool,uint32_t block_tokens,
	uint32_t checkpoint_slots)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->block_token_count = block_tokens;
	configuration->lane_count = lane_count;
	configuration->blocks_per_lane = blocks_per_lane;
	configuration->physical_page_capacity = pool;
	configuration->logical_page_capacity = pool;
	configuration->checkpoint_slot_count = checkpoint_slots;
	configuration->block_stride_bytes = 4096u;
	memset(table,0xFF,(size_t)lane_count * blocks_per_lane *
		sizeof(table[0]));
	memset(counts,0,(size_t)lane_count * sizeof(counts[0]));
}

static int F1ScenarioReuseOn(void)
{
	SparkQwen36PagedKv cache;
	SparkQwen36PagedKvConfiguration configuration;
	SparkQwen36PagedKvMatch match;
	uint32_t prompt[256],i,worst_counts;
	SparkStatus status;
	GatePagedKvFixture(&configuration,f1_table,f1_counts,F1_LANES,
		F1_BLOCKS_PER_LANE,F1_POOL,F1_BLOCK_TOKENS,4u);
	assert(SparkQwen36PagedKvInitialize(&cache,&configuration,f1_table,
		f1_counts) == SPARK_STATUS_OK);
	F1Fill(prompt,10000u,130u); /* 130 tokens: 2 full blocks + 1 open */
	status = SparkQwen36PagedKvAdmit(&cache,0u,prompt,130u,&match);
	assert(status == SPARK_STATUS_OK);
	assert(F1ConservationCheck(&cache,"A/admit",0u) == 0u);
	worst_counts = f1_counts[0];
	for (i=0u; i<F1_ROUNDS; i++)
	{
		/* Accepted tokens advance the row; D=4 draft depth. */
		uint64_t pos = 131ull + i;
		status = SparkQwen36PagedKvCover(&cache,0u,pos + 1u,0,0);
		assert(status == SPARK_STATUS_OK);
		assert(F1ConservationCheck(&cache,"A/pre",i) == 0u);
		status = SparkQwen36PagedKvCover(&cache,0u,pos + 6u,0,0);
		assert(status == SPARK_STATUS_OK);
		assert(F1ConservationCheck(&cache,"A/spec",i) == 0u);
		if ( f1_counts[0] > worst_counts )
			worst_counts = f1_counts[0];
	}
	printf("f1_conservation reuse-on: %u rounds, max attached=%u, final free=%u\n",
		F1_ROUNDS,worst_counts,SparkQwen36PagedKvFreeBlocks(&cache));
	/* Teardown must return every outstanding scratch block. */
	SparkQwen36PagedKvLaneReset(&cache,0u);
	assert(F1ConservationCheck(&cache,"A/reset",F1_ROUNDS) == 0u);
	SparkQwen36PagedKvDestroy(&cache);
	return(0);
}

static int F1ScenarioReuseOff(void)
{
	SparkQwen36PagedKv cache;
	SparkQwen36PagedKvConfiguration configuration;
	uint32_t prompt[256],i,worst_counts;
	SparkStatus status;
	GatePagedKvFixture(&configuration,f1_table,f1_counts,F1_LANES,
		F1_BLOCKS_PER_LANE,F1_POOL,F1_BLOCK_TOKENS,0u);
	assert(SparkQwen36PagedKvInitialize(&cache,&configuration,f1_table,
		f1_counts) == SPARK_STATUS_OK);
	F1Fill(prompt,20000u,70u);
	/* Reuse-off admits bind no sequence: the lane runs pure scratch. */
	status = SparkQwen36PagedKvAdmit(&cache,0u,prompt,70u,0);
	assert(status == SPARK_STATUS_OK);
	worst_counts = f1_counts[0];
	for (i=0u; i<F1_ROUNDS; i++)
	{
		uint64_t pos = 71ull + i;
		status = SparkQwen36PagedKvCover(&cache,0u,pos + 1u,0,0);
		assert(status == SPARK_STATUS_OK);
		assert(F1ConservationCheck(&cache,"B/cover",i) == 0u);
		if ( f1_counts[0] > worst_counts )
			worst_counts = f1_counts[0];
	}
	printf("f1_conservation reuse-off: %u rounds, max attached=%u, final free=%u\n",
		F1_ROUNDS,worst_counts,SparkQwen36PagedKvFreeBlocks(&cache));
	SparkQwen36PagedKvDestroy(&cache);
	return(0);
}

/* ---- CASE 9 support: F2 feasibility accounting ----
 * The F2 bug: ScratchBorrow popped the core free list but left
 * state = BLOCK_FREE, so SparkQwen36PagedKvFreeBlocks() counted in-use
 * scratch as free and the spec-coverage feasibility check
 * (ExtendSpeculativeCoverage) passed when the pool had no room - the
 * refusal then surfaced as a MID-COVERAGE CAPACITY_EXCEEDED from Cover
 * instead of a clean up-front fall-back to plain batched decode. Worst
 * on the reuse-OFF path where every attached block is scratch.
 *
 * Exhaustion-shaped unit scenario mirroring the adapter's exact
 * two-phase contract: compute the growth a lane needs, compare it to
 * FreeBlocks() BEFORE any borrow, and only then Cover. Guarantees:
 *   1. accounting - FreeBlocks() + blocks held by lanes == pool at
 *      every step (no phantom-free),
 *   2. ordering - whenever the feasibility check passes, Cover MUST
 *      succeed; whenever it refuses, no Cover is attempted. A
 *      mid-coverage CAPACITY_EXCEEDED after a passing check is the F2
 *      failure mode and fails here. */

#define F2_POOL 12u
#define F2_LANES 2u
#define F2_BLOCKS_PER_LANE 16u
#define F2_BLOCK_TOKENS 64u

static uint32_t f2_table[F2_LANES * F2_BLOCKS_PER_LANE];
static uint32_t f2_counts[F2_LANES];

static int F2ScenarioFeasibility(void)
{
	SparkQwen36PagedKv cache;
	SparkQwen36PagedKvConfiguration configuration;
	uint32_t prompt[128],i,free_now,needed;
	SparkStatus status;
	GatePagedKvFixture(&configuration,f2_table,f2_counts,F2_LANES,
		F2_BLOCKS_PER_LANE,F2_POOL,F2_BLOCK_TOKENS,0u);
	assert(SparkQwen36PagedKvInitialize(&cache,&configuration,f2_table,
		f2_counts) == SPARK_STATUS_OK);
	for (i=0u; i<128u; i++)
		prompt[i] = 40000u + i * 3u;
	status = SparkQwen36PagedKvAdmit(&cache,0u,prompt,64u,0);
	assert(status == SPARK_STATUS_OK);
	/* Phase 1: drain the pool down to 2 free blocks through lane 0. */
	for (i=1u; i<=10u; i++)
	{
		status = SparkQwen36PagedKvCover(&cache,0u,
			(uint64_t)i * F2_BLOCK_TOKENS,0,0);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"F2FAIL: phase-1 Cover(end=%u) failed: %d\n",
				i * F2_BLOCK_TOKENS,(int)status);
			return(1);
		}
		free_now = SparkQwen36PagedKvFreeBlocks(&cache);
		if ( free_now + f2_counts[0] != F2_POOL )
		{
			fprintf(stderr,"F2FAIL: phantom-free after end=%u: free=%u + attached=%u != pool=%u\n",
				i * F2_BLOCK_TOKENS,free_now,f2_counts[0],
				F2_POOL);
			return(1);
		}
	}
	assert(f2_counts[0] == 10u);
	/* Phase 2: adapter-shaped decision for a lane-1 extension needing 5
	 * blocks with only 2 truly free. */
	free_now = SparkQwen36PagedKvFreeBlocks(&cache);
	needed = 5u - f2_counts[1];
	if ( needed > free_now )
	{
		printf("f2_feasibility: refused UP FRONT (need %u, free %u) before any Cover\n",
			needed,free_now);
	}
	else
	{
		status = SparkQwen36PagedKvCover(&cache,1u,
			5ull * F2_BLOCK_TOKENS,0,0);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"F2FAIL: feasibility PASSED (%u free reported) yet Cover failed mid-coverage: %d\n",
				free_now,(int)status);
			return(1);
		}
	}
	/* Phase 3: teardown must still conserve every block. */
	SparkQwen36PagedKvLaneReset(&cache,0u);
	SparkQwen36PagedKvLaneReset(&cache,1u);
	free_now = SparkQwen36PagedKvFreeBlocks(&cache);
	if ( free_now != F2_POOL )
	{
		fprintf(stderr,"F2FAIL: teardown lost blocks: free=%u != pool=%u\n",
			free_now,F2_POOL);
		return(1);
	}
	SparkQwen36PagedKvDestroy(&cache);
	return(0);
}

/* ---- CASE 10 support: S512 scale - completeness-matrix top end ----
 * The completeness matrix binds to B1..max while the adapter-level
 * pressure case stopped at B25. The driver's maximum declared batch
 * size is SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT
 * = 512 (modules/qwen36_resident_decode_stage/include/sparkpipe/
 * spark_qwen36_resident_decode_stage_firmware.h; deployment description
 * examples/model_descriptions/qwen36_resident_decode_stage_firmware.json
 * "max_active_slots": 512) - so the matrix TOP END is B512, below the
 * core gate's B1024 cells. This scenario drives the paged-KV unit at
 * that B over a rotating residency (the F1 assertion pattern):
 * shared AND diverging prefixes against WITNESSED donor checkpoints,
 * mid-block admits (non-block-aligned prompts leave an open top
 * block), eviction pressure (distinct publications exceed the pool so
 * the borrow path's Trim must fire), and >=2 spec-decode rounds per
 * lane crossing block boundaries - free-list conservation asserted
 * after EVERY Cover call. */

#define S_LANES SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT
#define S_BLOCK_TOKENS 64u
#define S_BLOCKS_PER_LANE 8u
/* Pool deliberately SMALLER than the run's distinct publications
 * (128 groups x ~3 published blocks) so the borrow path's LRU Trim
 * MUST fire - measured via the core evicted counter. */
#define S_POOL 256u
#define S_CHECKPOINT_SLOTS 8u
#define S_GROUP_SIZE 4u
#define S_BATCH_LANES 64u
#define S_ROUNDS_PER_LANE 160u

static uint32_t s_table[S_LANES * S_BLOCKS_PER_LANE];
static uint32_t s_counts[S_LANES];
static int8_t s_in_free[S_POOL],s_in_use[S_POOL];

/* S512 wrapper: the SAME invariant core as F1, B512-scale buffers. */
static uint32_t S512ConservationCheck(const SparkQwen36PagedKv *cache,
	const char *where,uint32_t step)
{
	return(GateFreeListConservation(cache,"S512",where,step,
		s_in_free,s_in_use,S_POOL));
}

static int S512ScaleScenario(void)
{
	SparkQwen36PagedKv cache;
	SparkQwen36PagedKvConfiguration configuration;
	SparkQwen36PagedKvMatch match;
	SparkPrefixCacheCoreStats stats_start,stats_end;
	static uint32_t donor_tokens[130],diverge_tokens[130],
		midblock_tokens[100];
	uint32_t batch,group,round,i,slot;
	SparkStatus status;
	GatePagedKvFixture(&configuration,s_table,s_counts,S_LANES,
		S_BLOCKS_PER_LANE,S_POOL,S_BLOCK_TOKENS,S_CHECKPOINT_SLOTS);
	assert(SparkQwen36PagedKvInitialize(&cache,&configuration,s_table,
		s_counts) == SPARK_STATUS_OK);
	SparkPrefixCacheCoreQueryStats(&cache.core,&stats_start);
	for (batch=0u; batch<S_LANES/S_BATCH_LANES; batch++)
	{
		for (group=0u; group<S_BATCH_LANES/S_GROUP_SIZE; group++)
		{
			/* One group = four roles over fresh lanes: the donor of a
			 * unique root, an IDENTICAL sibling (full witnessed
			 * resume), a DIVERGING sibling (block-granular resume),
			 * and a MID-BLOCK admit (100 tokens: open top block). */
			uint32_t lanes[S_GROUP_SIZE];
			uint32_t index;
			uint32_t base = 2000000u + batch * 100000u +
				group * 4000u;
			lanes[0] = batch * S_BATCH_LANES + group * S_GROUP_SIZE;
			for (index=1u; index<S_GROUP_SIZE; index++)
				lanes[index] = lanes[index - 1u] + 1u;
			for (i=0u; i<130u; i++)
			{
				donor_tokens[i] = base + ((i * 13u) % 97u) * 7u;
				diverge_tokens[i] = i < 64u ? donor_tokens[i] :
					base + 500u + i;
			}
			for (i=0u; i<100u; i++)
				midblock_tokens[i] = base + 900u + i * 11u;
			/* Donor: full walk, checkpoints witnessed at 64 and 128. */
			status = SparkQwen36PagedKvAdmit(&cache,lanes[0],
				donor_tokens,130u,&match);
			assert(status == SPARK_STATUS_OK);
			assert(match.block_count == 0u);
			assert(SparkQwen36PagedKvCommittedTokens(&cache,
				lanes[0]) == 130u);
			if ( SparkQwen36PagedKvCheckpointOffer(&cache,lanes[0],
				64u,&slot) != 0u )
				SparkQwen36PagedKvCheckpointCommit(&cache,lanes[0],
					slot,64u);
			if ( SparkQwen36PagedKvCheckpointOffer(&cache,lanes[0],
				128u,&slot) != 0u )
				SparkQwen36PagedKvCheckpointCommit(&cache,lanes[0],
					slot,128u);
			assert(S512ConservationCheck(&cache,"s/donor",
				lanes[0]) == 0u);
			/* Identical sibling: resumes at the deepest witnessed
			 * boundary (both donor blocks shared verbatim). */
			status = SparkQwen36PagedKvAdmit(&cache,lanes[1],
				donor_tokens,130u,&match);
			assert(status == SPARK_STATUS_OK);
			assert(match.block_count == 2u);
			assert(match.checkpoint_slot !=
				SPARK_QWEN36_PAGED_KV_NO_SLOT);
			/* Diverging sibling at token 64: block-granular resume. */
			status = SparkQwen36PagedKvAdmit(&cache,lanes[2],
				diverge_tokens,130u,&match);
			assert(status == SPARK_STATUS_OK);
			assert(match.block_count == 1u);
			/* Walk the divergent tail canonically into a mid-block
			 * frontier (64 -> 100). */
			status = SparkQwen36PagedKvCover(&cache,lanes[2],100u,
				diverge_tokens + 64u,36u);
			assert(status == SPARK_STATUS_OK);
			/* Mid-block admit: 100 tokens, open top block, no
			 * witnessed match on a fresh root. */
			status = SparkQwen36PagedKvAdmit(&cache,lanes[3],
				midblock_tokens,100u,&match);
			assert(status == SPARK_STATUS_OK);
			assert(match.block_count == 0u);
			assert(SparkQwen36PagedKvCommittedTokens(&cache,
				lanes[3]) == 100u);
			assert(S512ConservationCheck(&cache,"s/group",
				lanes[3]) == 0u);
			/* Spec-decode rounds for all four roles: per round the
			 * plain row Cover(end = pos + 1) plus the speculative
			 * extension Cover(end = pos + D + 2)-shaped jump, no
			 * canonical tokens while scratch is outstanding. With
			 * frontiers at 100/130 and 160 rounds the coverage
			 * crosses block boundaries 192 AND 256 (>=(2) crossings)
			 * under outstanding scratch. Conservation after EVERY
			 * Cover. */
			for (index=0u; index<S_GROUP_SIZE; index++)
			{
				uint64_t frontier = SparkQwen36PagedKvCommittedTokens(
					&cache,lanes[index]);
				for (round=0u; round<S_ROUNDS_PER_LANE; round++)
				{
					uint64_t pos = frontier + 1ull +
						(uint64_t)round;
					status = SparkQwen36PagedKvCover(&cache,
						lanes[index],pos + 1ull,0,0);
					assert(status == SPARK_STATUS_OK);
					assert(S512ConservationCheck(&cache,
						"s/pre",round) == 0u);
					status = SparkQwen36PagedKvCover(&cache,
						lanes[index],pos + 6ull,0,0);
					assert(status == SPARK_STATUS_OK);
					assert(S512ConservationCheck(&cache,
						"s/spec",round) == 0u);
				}
				/* Teardown returns every outstanding scratch. */
				SparkQwen36PagedKvLaneReset(&cache,lanes[index]);
				assert(S512ConservationCheck(&cache,"s/reset",
					round) == 0u);
			}
		}
	}
	SparkPrefixCacheCoreQueryStats(&cache.core,&stats_end);
	/* Eviction pressure MUST have fired: 512 distinct publications
	 * against a 1024-block pool with rotating residencies force Trim. */
	assert(stats_end.evicted_block_count >
		stats_start.evicted_block_count);
	printf("s512_scale lanes=%u rounds_per_lane=%u matched_blocks=%llu appended_tokens=%llu evicted_blocks=%llu free=%u\n",
		(unsigned)S_LANES,(unsigned)S_ROUNDS_PER_LANE,
		(unsigned long long)stats_end.matched_block_count,
		(unsigned long long)stats_end.appended_token_count,
		(unsigned long long)stats_end.evicted_block_count,
		SparkQwen36PagedKvFreeBlocks(&cache));
	SparkQwen36PagedKvDestroy(&cache);
	return(0);
}
int main(void)
{
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[GATE_MAX_LANES];
	GateState test_state;
	void *adapter_state;
	void *driver_library;
	char runtime_root[4096];
	uint32_t prompt_a[GATE_MAX_ROWS],prompt_c[GATE_MAX_ROWS],prompt_d[GATE_MAX_ROWS];
	uint32_t rows_lane[GATE_MAX_LANES * GATE_MAX_ROWS];
	uint64_t positions[GATE_MAX_LANES * GATE_MAX_ROWS],sequences[GATE_MAX_LANES * GATE_MAX_ROWS];
	uint32_t row,lane,count,prompt_length,records_before,max_length;
	const GateFrameRecord *donor_tail = 0;
	memset(&test_state,0,sizeof(test_state));
	assert(cudaStreamCreate((cudaStream_t *)&test_state.execution_stream) == cudaSuccess);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	driver_library = dlopen(TEST_QWEN36_SERVING_DRIVER_PATH,RTLD_NOW);
	assert(driver_library != 0);
	gate_record_count = (GateRecordCountFunction)dlsym(driver_library,"TestQwen36ServingDriverRecordCount");
	gate_record_get = (GateRecordGetFunction)dlsym(driver_library,"TestQwen36ServingDriverRecord");
	gate_record_reset = (GateRecordResetFunction)dlsym(driver_library,"TestQwen36ServingDriverResetRecords");
	gate_kv_blocks = (GateKvBlockCountFunction)dlsym(driver_library,"TestQwen36ServingDriverKvBlockCount");
	assert(gate_record_count != 0 && gate_record_get != 0 && gate_record_reset != 0 && gate_kv_blocks != 0);

	{
		SparkModelServingAdapterDynamicLibrary library;
		assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_QWEN36_SERVING_ADAPTER_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE,&library) == SPARK_STATUS_OK);
		adapter_interface_global = &library.adapter_interface;

		/* Shared prompts: A is the donor; B repeats it; C shares the first
		 * 100 tokens then diverges (block-granular reuse must stop at 64). */
		prompt_length = 200u;
		for (row=0u; row<prompt_length; row++)
		{
			prompt_a[row] = 5000u + ((row * 13u) % 97u) * 7u;
			prompt_c[row] = row < 100u ? prompt_a[row] : 9000u + row;
		}

		/* ---------- CASE 1+2: B1 reuse, identical + diverging (ON) ----- */
		setenv("SPARK_QWEN36_SERVING_PREFIX_CACHE","1",1);
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,1024u,512u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		assert(adapter_state != 0);
		gate_record_reset();

		records_before = gate_record_count();
		assert(GatePrefill(adapter_state,&test_state,2u,1001u,prompt_a,prompt_length,700u) == SPARK_STATUS_OK);
		assert(test_state.completion_count == 1u);
		/* Donor walked everything; boundary-shaped frames checkpointed at
		 * 64/128/192. */
		GateAssertWalk("donor",records_before,gate_record_count(),2u,prompt_a,0u,prompt_length,0u,0u,0);
		{
			uint32_t index,checkpoints;
			checkpoints = 0u;
			for (index=records_before; index<gate_record_count(); index++)
			{
				const GateFrameRecord *record = gate_record_get(index);
				if ( (record->context_flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT) != 0u )
				{
					checkpoints++;
					assert(record->snapshot_index_present != 0u);
				}
			}
			assert(checkpoints == 3u); /* boundaries 64,128,192 of a 200-token prompt */
			donor_tail = gate_record_get(gate_record_count() - 1u);
		}

		/* B: identical prompt another slot - zero recompute below 192. */
		records_before = gate_record_count();
		assert(GatePrefill(adapter_state,&test_state,3u,1002u,prompt_a,prompt_length,701u) == SPARK_STATUS_OK);
		GateAssertWalk("identical",records_before,gate_record_count(),3u,prompt_a,192u,prompt_length,1u,3u,donor_tail);

		/* C: diverges at token 100 - block-granular reuse stops at 64. */
		records_before = gate_record_count();
		assert(GatePrefill(adapter_state,&test_state,4u,1003u,prompt_c,prompt_length,702u) == SPARK_STATUS_OK);
		GateAssertWalk("diverging",records_before,gate_record_count(),4u,prompt_c,64u,prompt_length,1u,1u,donor_tail);

		/* Wiring probe (same instance): the module pool was sized from
		 * kv_physical_page_capacity through the strict env channel. */
		assert(gate_kv_blocks() == 512u);
		assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
		library.adapter_interface.destroy(adapter_state);

		/* ---------- CASE 3: ON vs OFF ------------------------------------ */
		setenv("SPARK_QWEN36_SERVING_PREFIX_CACHE","0",1);
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,1024u,512u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		gate_record_reset();
		records_before = gate_record_count();
		assert(GatePrefill(adapter_state,&test_state,2u,2001u,prompt_a,prompt_length,800u) == SPARK_STATUS_OK);
		assert(GatePrefill(adapter_state,&test_state,3u,2002u,prompt_a,prompt_length,801u) == SPARK_STATUS_OK);
		{
			uint32_t index,total_rows;
			total_rows = 0u;
			for (index=records_before; index<gate_record_count(); index++)
			{
				const GateFrameRecord *record = gate_record_get(index);
				assert((record->context_flags & (SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME | SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT)) == 0u);
				total_rows += record->rows;
			}
			assert(total_rows == 2u * prompt_length);
		}
		library.adapter_interface.destroy(adapter_state);
		unsetenv("SPARK_QWEN36_SERVING_PREFIX_CACHE");

		/* ---------- CASE 4: B4 mixed - four resident sequences with shared
		 * AND diverging prefixes, admitted as consecutive serving rounds; plus
		 * one genuinely batched four-lane prefill of fresh prompts. ---------- */
		setenv("SPARK_QWEN36_SERVING_PREFIX_CACHE","1",1);
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,1024u,512u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		gate_record_reset();
		for (row=0u; row<200u; row++)
			prompt_d[row] = 70000u + row * 3u;
		for (row=0u; row<200u; row++)
			prompt_c[row] = row < 100u ? prompt_d[row] : 8000u + row;
		assert(GatePrefill(adapter_state,&test_state,2u,4001u,prompt_d,200u,900u) == SPARK_STATUS_OK);
		donor_tail = gate_record_get(gate_record_count() - 1u);
		records_before = gate_record_count();
		assert(GatePrefill(adapter_state,&test_state,3u,4002u,prompt_d,200u,901u) == SPARK_STATUS_OK);
		GateAssertWalk("b4_identical",records_before,gate_record_count(),3u,prompt_d,192u,200u,1u,3u,donor_tail);
		records_before = gate_record_count();
		assert(GatePrefill(adapter_state,&test_state,4u,4003u,prompt_c,200u,902u) == SPARK_STATUS_OK);
		GateAssertWalk("b4_diverging",records_before,gate_record_count(),4u,prompt_c,64u,200u,1u,1u,donor_tail);
		records_before = gate_record_count();
		assert(GatePrefill(adapter_state,&test_state,5u,4004u,prompt_a,200u,903u) == SPARK_STATUS_OK);
		GateAssertWalk("b4_distinct",records_before,gate_record_count(),5u,prompt_a,0u,200u,0u,0u,0);
		/* Batched fresh prefill: four lanes, one submission, wave-major.
		 * No donor exists for these tokens, so every lane walks everything
		 * and the shaped checkpoint offers interleave across lanes. */
		{
			uint32_t lengths[4],offsets[4];
			lengths[0] = 130u; lengths[1] = 130u; lengths[2] = 150u; lengths[3] = 90u;
			offsets[0] = 0u;
			for (lane=1u; lane<4u; lane++)
				offsets[lane] = offsets[lane - 1u] + lengths[lane - 1u];
			max_length = 0u;
			for (lane=0u; lane<4u; lane++)
				if ( lengths[lane] > max_length )
					max_length = lengths[lane];
			count = 0u;
			for (row=0u; row<max_length; row++)
			{
				for (lane=0u; lane<4u; lane++)
				{
					if ( row >= lengths[lane] )
						continue;
					rows_lane[count] = lane;
					positions[count] = row;
					sequences[count] = 4100u + lane;
					count++;
				}
			}
			memset(lanes,0,sizeof(SparkModelServingLane) * 4u);
			for (lane=0u; lane<4u; lane++)
			{
				lanes[lane].sequence_id = 4100u + lane;
				lanes[lane].request_id = 1910u + lane;
				lanes[lane].resident_sequence_slot = 2u + lane;
				lanes[lane].context_token_count = lengths[lane];
				lanes[lane].request_generation = 1u;
				lanes[lane].step_generation = 1u;
				lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
			}
			memset(&submission,0,sizeof(submission));
			submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
			submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
			submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
			submission.tokens_per_sequence = 1u;
			submission.submission_id = 910u;
			submission.request_id = 1910u;
			submission.sequence_id = 4100u;
			submission.control_generation = 1u;
			submission.transaction_id = 2910u;
			submission.dispatch_generation = 3910u;
			submission.request_generation = 1u;
			submission.step_generation = 1u;
			submission.residency.word0 = 910u;
			submission.residency.generation = 277u;
			submission.residency.owner = 13u;
			submission.active_sequence_count = 4u;
			submission.new_token_count = count;
			submission.lane_count = 4u;
			submission.row_count = count;
			submission.token_count = count;
			submission.lanes = lanes;
			submission.row_positions = positions;
			submission.row_lane_indices = rows_lane;
			submission.row_sequence_ids = sequences;
			/* token buffer: every lane walks its OWN prompt; the batched
			 * buffer interleaves them by flat order */
			{
				static uint32_t batch_tokens[GATE_MAX_LANES * GATE_MAX_ROWS];
				uint32_t cursor[4];
				cursor[0] = cursor[1] = cursor[2] = cursor[3] = 0u;
				for (row=0u; row<count; row++)
				{
					lane = rows_lane[row];
					if ( lane == 0u || lane == 1u )
						batch_tokens[row] = prompt_d[cursor[lane]];
					else if ( lane == 2u )
						batch_tokens[row] = prompt_c[cursor[lane]];
					else
						batch_tokens[row] = prompt_a[cursor[lane]];
					cursor[lane]++;
				}
				submission.token_ids = batch_tokens;
				test_state.completion_count = 0u;
				assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
				assert(test_state.completion_count == 1u);
			}
		}
		assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
		library.adapter_interface.destroy(adapter_state);

		/* ---------- CASE 5: B25 pressure - 25 resident sequences sharing
		 * five prompt roots, checkpoint-slot churn shortening matches, and
		 * pool exhaustion refusing cleanly. ------------------------------ */
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,25u,32u,512u,256u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		assert(gate_kv_blocks() == 256u); /* physical pages wired to the module */
		gate_record_reset();
		{
			uint32_t root,length,statuses_ok;
			uint32_t statuses_refused;
			static uint32_t pressure_tokens[GATE_MAX_ROWS];
			statuses_ok = 0u;
			statuses_refused = 0u;
			for (root=0u; root<25u; root++)
			{
				length = 70u + ((root * 37u) % 300u);
				for (row=0u; row<length && row<GATE_MAX_ROWS; row++)
					pressure_tokens[row] = row < 130u ?
						60000u + (root / 5u) * 1000u + row :
						60000u + root * 17u + row;
				records_before = gate_record_count();
				{
					SparkStatus gate_status = GatePrefill(adapter_state,&test_state,2u + (root % 20u),
						5001u + root,pressure_tokens,length,1000u + root);
				if ( gate_status == SPARK_STATUS_OK )
				{
					uint32_t index,resume_seen,walked;
					uint64_t walked_low,walked_high;
					statuses_ok++;
					walked_low = UINT64_MAX;
					walked_high = 0u;
					resume_seen = 0u;
					walked = 0u;
					for (index=records_before; index<gate_record_count(); index++)
					{
						const GateFrameRecord *record = gate_record_get(index);
						assert(record->prefill != 0u);
						if ( (record->context_flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME) != 0u )
						{
							resume_seen++;
							assert(record->base_position % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS == 0u);
							assert(record->snapshot_index_present != 0u);
						}
						assert(index == records_before ||
							record->base_position == walked_high);
						walked += record->rows;
						walked_high = record->base_position + record->rows;
						if ( walked_low == UINT64_MAX )
							walked_low = record->base_position;
					}
					assert(walked == length - (uint32_t)(resume_seen != 0u ? walked_low : 0u));
					assert(resume_seen <= 1u);
				}
				else
				{
					statuses_refused++;
					fprintf(stderr,"b25_refused root=%u length=%u status=%d\n",root,length,(int)gate_status);
					assert(test_state.completion_count == 0u);
				}
				}
			}
			printf("b25_pressure ok=%u refused=%u\n",statuses_ok,statuses_refused);
			assert(statuses_ok + statuses_refused == 25u);
		}
		library.adapter_interface.destroy(adapter_state);

		/* ---------- CASE 5b: wiring validation refuses bad page budgets -
		 * physical > logical violates the shared JIT_KV runtime-limit rule
		 * (INVALID_ARGUMENT, the same status the driver paged-KV contract
		 * returns); a pool that fits the logical budget but not the module
		 * geometry (physical > resident lanes x blocks-per-lane) is the
		 * module's own pool-fit rule and comes back SCHEMA_ERROR. -------- */
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,128u,512u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_INVALID_ARGUMENT);
		assert(adapter_state == 0);
		/* derived pool = 8 resident lanes x ceil(4096/64) blocks = 512. */
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,1024u,600u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_SCHEMA_ERROR);
		assert(adapter_state == 0);

		/* ---------- CASE 5c: B512 top end - the driver's maximum
		 * declared batch size (SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_
		 * ACTIVE_SEQUENCE_COUNT = 512, spark_qwen36_resident_decode_
		 * stage_firmware.h; deployment description "max_active_slots":
		 * 512). The b25 pressure shape at the matrix top end: 512
		 * resident sequences over shared roots and divergent tails,
		 * slots spanning the full declared range. ---------------------- */
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,
			SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
			SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
			4096u,2048u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		assert(gate_kv_blocks() == 2048u); /* physical pages wired */
		gate_record_reset();
		{
			uint32_t root,length,statuses_ok,statuses_refused;
			static uint32_t scale_tokens[GATE_MAX_ROWS];
			statuses_ok = 0u;
			statuses_refused = 0u;
			for (root=0u; root<SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; root++)
			{
				length = 70u + ((root * 37u) % 300u);
				for (row=0u; row<length && row<GATE_MAX_ROWS; row++)
					scale_tokens[row] = row < 130u ?
						300000u + (root / 8u) * 1000u + row :
						300000u + root * 17u + row;
				if ( root % 128u == 0u )
					gate_record_reset(); /* ring capacity 8192 */
				records_before = gate_record_count();
				{
					SparkStatus gate_status = GatePrefill(adapter_state,&test_state,
						root % SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
						7001u + root,scale_tokens,length,11000u + root);
					if ( gate_status == SPARK_STATUS_OK )
					{
						statuses_ok++;
						assert(test_state.completion_count == 1u);
					}
					else
					{
						statuses_refused++;
						fprintf(stderr,"b512_refused root=%u length=%u status=%d\n",
							root,length,(int)gate_status);
						assert(test_state.completion_count == 0u);
					}
				}
			}
			printf("b512_pressure ok=%u refused=%u\n",statuses_ok,statuses_refused);
			assert(statuses_ok + statuses_refused ==
				SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT);
		}
		assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
		library.adapter_interface.destroy(adapter_state);

		/* ---------- CASE 6: DECODE work with speculation ----------
		 * Two resident lanes, many rounds of B1 speculative decode
		 * alternating between them: each round covers the plain row
		 * (end = pos + 1) AND the speculative extension
		 * (end = pos + D + 2) with no canonical tokens entering while
		 * scratch is outstanding. Assertions per round:
		 *   - the submission completes (no mid-coverage CAPACITY_
		 *     EXCEEDED surfacing as a dropped submission),
		 *   - every frame the adapter built for the lane carries a block
		 *     table row that only ever GROWS, never churns: blocks still
		 *     attached at round N-1 are still attached - unchanged - at
		 *     round N. A re-borrow over slots naming last round's scratch
		 *     is the F1 orphaning mechanism.
		 * The prompt length (126) puts the first boundary crossing inside
		 * round zero and the second (~192) inside round ~126, so scratch
		 * is outstanding across >64 rounds and >=2 crossings. */
		setenv("SPARK_QWEN36_SERVING_PREFIX_CACHE","1",1);
		setenv("SPARK_QWEN36_SERVING_SPECULATE","1",1);
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,4u,4u,128u,128u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		assert(gate_kv_blocks() == 128u); /* 2 lanes x 64 blocks */
		gate_record_reset();
		{
			uint32_t prompt_e[GATE_MAX_ROWS],prompt_f[GATE_MAX_ROWS];
			uint32_t blocks_e[GATE_CAPTURE_ROWS],blocks_f[GATE_CAPTURE_ROWS];
			uint32_t count_e,count_f,round,record;
			FILE *diag_capture_file;
			int diag_saved_fd;
			count_e = UINT32_MAX;
			count_f = UINT32_MAX;
			for (row=0u; row<126u; row++)
			{
				prompt_e[row] = 30000u + row * 5u;
				prompt_f[row] = 31000u + row * 5u;
			}
			/* Reuse observability gate: capture stderr for this whole case
			 * so the merged per-lane diag line is asserted below - one line
			 * per round, historic fields intact, plus matched_blocks= and
			 * borrowed= so a production log proves reuse without re-running
			 * this harness. */
			diag_capture_file = tmpfile();
			assert(diag_capture_file != 0);
			diag_saved_fd = dup(fileno(stderr));
			assert(diag_saved_fd >= 0);
			fflush(stderr);
			assert(dup2(fileno(diag_capture_file),fileno(stderr)) == 2);
			assert(GatePrefill(adapter_state,&test_state,1u,6001u,prompt_e,126u,1500u) == SPARK_STATUS_OK);
			{
				SparkStatus gate_status = GatePrefill(adapter_state,&test_state,2u,6002u,prompt_f,126u,1501u);
				if ( gate_status != SPARK_STATUS_OK )
					fprintf(stderr,"case6_prefill_F_status=%d\n",(int)gate_status);
				assert(gate_status == SPARK_STATUS_OK);
			}
			/* Snapshot each lane's post-prefill block row: the LAST
			 * record naming that lane's slot (the ring interleaves the
			 * two lanes' frame sequences). */
			for (record=gate_record_count(); record>0u; record--)
			{
				const GateFrameRecord *frame =
					gate_record_get(record - 1u);
				if ( count_e == UINT32_MAX && frame->lane_index == 1u )
				{
					count_e = frame->block_count;
					memcpy(blocks_e,frame->blocks,sizeof(blocks_e));
				}
				if ( count_f == UINT32_MAX && frame->lane_index == 2u )
				{
					count_f = frame->block_count;
					memcpy(blocks_f,frame->blocks,sizeof(blocks_f));
				}
			}
			assert(count_e == 2u && count_f == 2u); /* ceil(126/64) */
			for (round=0u; round<140u; round++)
			{
				uint32_t slot = round % 2u != 0u ? 1u : 2u;
				uint32_t *blocks = slot == 1u ? blocks_e : blocks_f;
				uint32_t *count = slot == 1u ? &count_e : &count_f;
				uint64_t position = 126ull + (uint64_t)(round / 2u);
				uint32_t seen_tail = 0u;
				assert(GateDecode(adapter_state,&test_state,slot,
					slot == 1u ? 6001u : 6002u,position,
					8000u + round,1600u + round) == SPARK_STATUS_OK);
				assert(test_state.completion_count == 1u);
				/* The LAST record of this round names the lane's final
				 * block-table row for the round. */
				for (record=gate_record_count(); record>0u; record--)
				{
					const GateFrameRecord *frame =
						gate_record_get(record - 1u);
					if ( frame->lane_index != slot )
						continue;
					assert(frame->block_count >= *count);
					for (row=0u; row<*count && row<frame->block_count &&
						row<GATE_CAPTURE_ROWS; row++)
						if ( frame->blocks[row] != blocks[row] )
						{
							fprintf(stderr,"case6_churn round=%u slot=%u ordinal=%u prev=%u now=%u prev_count=%u now_count=%u\n",
								round,slot,row,blocks[row],
								frame->blocks[row],*count,
								frame->block_count);
							assert(frame->blocks[row] == blocks[row]);
						}
					*count = frame->block_count;
					for (row=0u; row<*count && row<GATE_CAPTURE_ROWS; row++)
						blocks[row] = frame->blocks[row];
					seen_tail++;
					break;
				}
				assert(seen_tail == 1u);
			}
			/* Both lanes crossed into a fourth block while speculating
			 * (end positions past 192): the boundary crossings happened
			 * under outstanding scratch, not around it. */
			assert(count_e >= 4u && count_f >= 4u);
			printf("spec_decode ok=%u rounds, lane rows grown to %u/%u blocks\n",
				140u,count_e,count_f);
			/* Restore stderr, then assert the captured diag stream: exactly
			 * one merged line per round; every line keeps the historic
			 * fields in order and APPENDS matched_blocks= / borrowed= at
			 * end of that SAME single line (zero added log volume; no
			 * historic adjacency moves, so positional log parsers are
			 * untouched by this change). */
			fflush(stderr);
			assert(dup2(diag_saved_fd,fileno(stderr)) == 2);
			close(diag_saved_fd);
			{
				char *diag_buffer;
				long diag_bytes;
				uint32_t diag_lines = 0u;
				const char *diag_cursor;
				fflush(stderr);
				diag_bytes = ftell(diag_capture_file);
				assert(diag_bytes >= 0);
				rewind(diag_capture_file);
				diag_buffer = malloc((size_t)diag_bytes + 1u);
				assert(diag_buffer != 0);
				assert(fread(diag_buffer,1,(size_t)diag_bytes,
					diag_capture_file) == (size_t)diag_bytes);
				diag_buffer[diag_bytes] = 0;
				fclose(diag_capture_file);
				diag_cursor = diag_buffer;
				while ( (diag_cursor = strstr(diag_cursor,
					"qwen36_spec_diag lane=")) != 0 )
				{
					const char *line_end = strchr(diag_cursor,'\n');
					const char *accepted_at = strstr(diag_cursor,"accepted=");
					const char *matched_at = strstr(diag_cursor,"matched_blocks=");
					const char *borrowed_at = strstr(diag_cursor,"borrowed=");
					const char *drafts_at = strstr(diag_cursor," drafts=[");
					const char *emitted_at = strstr(diag_cursor," emitted=[");
					const char *base_at = strstr(diag_cursor,"base_position=");
					diag_lines++;
					assert(line_end != 0 && base_at != 0 && accepted_at != 0 &&
						matched_at != 0 && borrowed_at != 0 && drafts_at != 0 &&
						emitted_at != 0);
					/* all fields on ONE line, historic order preserved, new
					 * keys APPENDED after emitted=[ - nothing inserted between
					 * historic fields (parser compatibility) */
					assert(base_at < accepted_at && accepted_at < drafts_at &&
						drafts_at < emitted_at && emitted_at < matched_at &&
						matched_at < borrowed_at && borrowed_at < line_end);
					diag_cursor = drafts_at;
				}
				assert(diag_lines == 140u);
				/* Re-emit byte-for-byte: the capture must not change the
				 * process's stderr flow - downstream (the full gate) still
				 * sees exactly the lines it always did. */
				fwrite(diag_buffer,1,(size_t)diag_bytes,stderr);
				fflush(stderr);
				free(diag_buffer);
			}
		}
		assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
		library.adapter_interface.destroy(adapter_state);
		unsetenv("SPARK_QWEN36_SERVING_SPECULATE");

		/* ---------- CASE 8: F1 free-list conservation ----------
		 * The audit-probe pattern, permanent: drive the paged-KV unit
		 * through >64 simulated B1 speculative rounds (crossing >=2 block
		 * boundaries) and hold {free list} U {lane rows} U {live
		 * sequences} U {core LRU} == pool after EVERY Cover call, reuse
		 * ON and OFF, teardown included. */
		assert(F1ScenarioReuseOn() == 0);
		assert(F1ScenarioReuseOff() == 0);
		printf("f1_conservation PASS\n");

		/* ---------- CASE 9: F2 feasibility accounting ---------- */
		assert(F2ScenarioFeasibility() == 0);
		printf("f2_feasibility PASS\n");

		/* ---------- CASE 10: S512 scale - matrix top end ---------- */
		assert(S512ScaleScenario() == 0);
		printf("s512_conservation PASS\n");

		/* ---------- CASE 11: checkpoint witness-chain law ----------
		 * Device byte-identity root cause (2026-08-23): ONE donor frame can
		 * be both the paged-Kv GDN_CHECKPOINT boundary and the flat-pool
		 * publish boundary, and the publish branch rebound the SHARED
		 * gdn_snapshot view - so the module wrote the boundary-192 witness
		 * into the PREFIX entry's slot while CheckpointCommit recorded it
		 * at its own slot. Later PREFIX_RESUME restores read a slot no
		 * checkpoint frame ever wrote (or a shallower boundary's clobbered
		 * slot). The fixture driver RECORDS each frame's snapshot view
		 * index, so the law is host-visible with no device and no fixture
		 * emulation change:
		 *   (1) the donor's GDN_CHECKPOINT frames record DISTINCT indices
		 *       (two boundaries on one device slot = one clobbered the
		 *       other);
		 *   (2) the publish-boundary frame - which under the bug rebinds
		 *       the view to the prefix entry - must still record ITS OWN
		 *       checkpoint slot index (distinctness in (1) covers it);
		 *   (3) every PREFIX_RESUME index names the checkpoint frame whose
		 *       walk ended exactly at the resume boundary. */
		setenv("SPARK_QWEN36_SERVING_PREFIX_CACHE","1",1);
		GateConfiguration(&configuration,TEST_QWEN36_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,1024u,512u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		gate_record_reset();
		records_before = gate_record_count();
		assert(GatePrefillPublish(adapter_state,&test_state,2u,11001u,prompt_a,prompt_length,1200u) == SPARK_STATUS_OK);
		{
			uint32_t index,checkpoint_count;
			uint32_t ends[8],slots[8];
			const GateFrameRecord *resume_record = 0;
			checkpoint_count = 0u;
			for (index=records_before; index<gate_record_count(); index++)
			{
				const GateFrameRecord *record = gate_record_get(index);
				if ( (record->context_flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT) == 0u )
					continue;
				assert(record->snapshot_index_present != 0u);
				assert(checkpoint_count < 8u);
				ends[checkpoint_count] = (uint32_t)(record->base_position + record->rows);
				slots[checkpoint_count] = record->snapshot_index;
				checkpoint_count++;
			}
			assert(checkpoint_count == 3u); /* boundaries 64,128,192 */
			/* Law (1)+(2): one device slot per boundary - no duplicates. */
			{
				uint32_t i,j;
				for (i=0u; i<checkpoint_count; i++)
					for (j=i + 1u; j<checkpoint_count; j++)
					{
						assert(ends[i] != ends[j]);
						assert(slots[i] != slots[j]);
					}
			}
			/* Law (3): the resuming lane restores from the checkpoint frame
			 * that walked its exact boundary. */
			assert(GatePrefillPublish(adapter_state,&test_state,3u,11002u,prompt_a,prompt_length,1201u) == SPARK_STATUS_OK);
			for (index=records_before; index<gate_record_count(); index++)
			{
				const GateFrameRecord *record = gate_record_get(index);
				uint32_t i,matched_end;
				if ( (record->context_flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME) == 0u )
					continue;
				assert(resume_record == 0); /* exactly one resume frame */
				resume_record = record;
				assert(record->snapshot_index_present != 0u);
				matched_end = 0u;
				for (i=0u; i<checkpoint_count; i++)
					if ( slots[i] == record->snapshot_index )
					{
						matched_end = ends[i];
						break;
					}
				assert(matched_end != 0u); /* index was actually written */
				assert(matched_end == (uint32_t)record->base_position);
			}
			assert(resume_record != 0);
			assert((uint32_t)resume_record->base_position == 192u);
			printf("checkpoint_witness boundaries=3 resume_at=%u slot=%u\n",
				(unsigned)resume_record->base_position,
				resume_record->snapshot_index);
		}
		assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
		library.adapter_interface.destroy(adapter_state);
	}
	assert(cudaStreamDestroy((cudaStream_t)test_state.execution_stream) == cudaSuccess);
	printf("qwen36 prefix-cache gate PASS\n");
	return(0);
}
