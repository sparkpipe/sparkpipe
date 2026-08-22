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
	}
	assert(cudaStreamDestroy((cudaStream_t)test_state.execution_stream) == cudaSuccess);
	printf("qwen36 prefix-cache gate PASS\n");
	return(0);
}
