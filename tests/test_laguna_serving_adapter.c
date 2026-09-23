#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_laguna_model.h"
#include "sparkpipe/spark_laguna_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_serving_adapter.h"

#ifndef TEST_LAGUNA_SERVING_ADAPTER_PATH
#define TEST_LAGUNA_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_LAGUNA_SERVING_DRIVER_PATH
#define TEST_LAGUNA_SERVING_DRIVER_PATH ""
#endif
#ifndef TEST_LAGUNA_SERVING_CONFIG_PATH
#define TEST_LAGUNA_SERVING_CONFIG_PATH ""
#endif
#ifndef TEST_LAGUNA_MODEL_REVISION
#define TEST_LAGUNA_MODEL_REVISION ""
#endif

/* TP8xPP2: the two pipeline positions exercised by the boundary
   assertions. Rank 0 is a first-stage rank (a bound frame ships
   hidden_output to the next stage); rank 8 is a last-stage rank (a
   bound frame consumes hidden_input). Both parse the tp_rank=0 fixture
   because 8 % 8 == 0 - the deployment convention carries the global
   rank in stage_index and the adapter derives stage_index/8. */
#define TEST_LAGUNA_FIRST_STAGE_RANK 0u
#define TEST_LAGUNA_LAST_STAGE_RANK 8u
#define TEST_LAGUNA_STAGE0_TOKEN 5000u
#define TEST_LAGUNA_STAGE1_TOKEN 6000u
#define TEST_LAGUNA_RAW_TOKEN 7000u
#define TEST_LAGUNA_MANDATORY_MEMBER_COUNT 10u

typedef struct TestLagunaServingState
{
	uint32_t completion_count;
	void *execution_stream;
	SparkModelServingCompletion completion;
} TestLagunaServingState;

static void TestLagunaServingCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	TestLagunaServingState *state;
	state = (TestLagunaServingState *)completion_context;
	assert(state != 0);
	assert(completion != 0);
	state->completion = *completion;
	state->completion_count++;
}

static void TestLagunaServingInterfaceCompleteness(
	const SparkModelServingAdapterInterface *adapter_interface)
{
	SparkModelServingAdapterInterface probe;
	void **first,**copy_first;
	uint32_t index,member_count;
	assert(adapter_interface->abi_version == SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION);
	assert(adapter_interface->interface_bytes == SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES);
	assert(SparkModelServingAdapterValidateInterface(adapter_interface,0u) ==
		SPARK_STATUS_OK);
	assert(SparkModelServingAdapterValidateInterface(adapter_interface,
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT) ==
		SPARK_STATUS_OK);
	first = (void **)&adapter_interface->initialize;
	member_count = (uint32_t)((void **)&adapter_interface->reset - first) + 1u;
	assert(member_count == TEST_LAGUNA_MANDATORY_MEMBER_COUNT);
	for (index=0u; index<member_count; index++)
		assert(first[index] != 0);
	for (index=0u; index<member_count; index++)
	{
		probe = *adapter_interface;
		copy_first = (void **)&probe.initialize;
		copy_first[index] = 0;
		assert(SparkModelServingAdapterValidateInterface(&probe,0u) ==
			SPARK_STATUS_INVALID_ARGUMENT);
	}
}

static void TestLagunaServingConfiguration(
	SparkModelServingAdapterConfiguration *configuration,
	uint32_t rank,
	const char *runtime_root,
	TestLagunaServingState *test_state)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = rank;
	configuration->stage_index = rank;
	configuration->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration->runtime_limits.max_inflight_submission_count = 1u;
	configuration->runtime_limits.max_active_sequence_count = 2u;
	configuration->runtime_limits.max_input_row_count = 2u;
	configuration->runtime_limits.resident_sequence_capacity = 8u;
	configuration->runtime_limits.kv_logical_page_capacity = 64u;
	configuration->runtime_limits.kv_physical_page_capacity = 64u;
	configuration->runtime_root = runtime_root;
	configuration->node_id = "spark-test";
	configuration->node_target = "cuda.sm121.laguna.resident_decode_stage.bf16.expert_bf16";
	configuration->adapter_configuration_path = TEST_LAGUNA_SERVING_CONFIG_PATH;
	configuration->driver_shared_object_path = TEST_LAGUNA_SERVING_DRIVER_PATH;
	configuration->driver_program_name = "resident_decode";
	configuration->execution_stream = test_state->execution_stream;
	configuration->completion_function = TestLagunaServingCompletion;
	configuration->completion_context = test_state;
}

static void TestLagunaServingDecodeSubmission(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lane,
	uint32_t *token_id,
	uint32_t *row_lane,
	uint64_t *row_position,
	uint64_t *row_sequence)
{
	memset(lane,0,sizeof(*lane));
	lane->request_id = 900u;
	lane->request_generation = 1u;
	lane->step_generation = 1u;
	lane->sequence_id = 100u;
	lane->resident_sequence_slot = 7u;
	lane->context_token_count = 1u;
	lane->flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	*token_id = 11u;
	*row_lane = 0u;
	*row_position = 0u;
	*row_sequence = 100u;
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->tokens_per_sequence = 1u;
	submission->submission_id = 1000u;
	submission->request_id = 900u;
	submission->sequence_id = 100u;
	submission->control_generation = 1u;
	submission->transaction_id = 2000u;
	submission->dispatch_generation = 3000u;
	submission->request_generation = 1u;
	submission->step_generation = 4000u;
	submission->active_sequence_count = 1u;
	submission->new_token_count = 1u;
	submission->lane_count = 1u;
	submission->row_count = 1u;
	submission->token_count = 1u;
	submission->lanes = lane;
	submission->token_ids = token_id;
	submission->row_lane_indices = row_lane;
	submission->row_positions = row_position;
	submission->row_sequence_ids = row_sequence;
}

static void TestLagunaServingApplyBoundaries(
	SparkModelServingSubmission *submission,
	uint32_t rank,
	void *hidden_input,
	void *hidden_output,
	uint64_t hidden_bytes)
{
	submission->hidden_input_address = 0;
	submission->hidden_input_bytes = 0u;
	submission->hidden_output_address = 0;
	submission->hidden_output_bytes = 0u;
	submission->boundary_sideband_input_address = 0;
	submission->boundary_sideband_input_bytes = 0u;
	submission->boundary_sideband_output_address = 0;
	submission->boundary_sideband_output_bytes = 0u;
	if ( rank / 8u != 0u )
	{
		submission->hidden_input_address = hidden_input;
		submission->hidden_input_bytes = hidden_bytes;
	}
	if ( rank / 8u + 1u < 2u )
	{
		submission->hidden_output_address = hidden_output;
		submission->hidden_output_bytes = hidden_bytes;
	}
}

int main(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	SparkModelServingAdapterSnapshot snapshot;
	TestLagunaServingState test_state;
	uint32_t token_id,row_lane,rank,index;
	uint64_t row_position,row_sequence;
	uint64_t boundary_bytes;
	void *adapter_state,*hidden_input,*hidden_output;
	char runtime_root[4096];
	memset(&test_state,0,sizeof(test_state));
	assert(cudaStreamCreate((cudaStream_t *)&test_state.execution_stream) == cudaSuccess);
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(
		TEST_LAGUNA_SERVING_ADAPTER_PATH,
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP,
		&library) == SPARK_STATUS_OK);
	assert(strcmp(library.adapter_interface.descriptor->adapter_id,
		"spark.laguna.serving-adapter.tp8.expert_bf16.v1") == 0);
	assert(strcmp(library.adapter_interface.descriptor->model_id,
		"poolside/Laguna-S-2.1") == 0);
	assert(strcmp(library.adapter_interface.descriptor->model_revision,
		TEST_LAGUNA_MODEL_REVISION) == 0);
	assert(library.adapter_interface.descriptor->capability_flags ==
		(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE));
	assert(library.adapter_interface.descriptor->stage_count == 16u);
	assert(library.adapter_interface.descriptor->parallel_group_size == 8u);
	assert(library.adapter_interface.descriptor->layer_count ==
		SPARK_LAGUNA_MODEL_LAYER_COUNT);
	for (index=0u; index<16u; index++)
		assert(library.adapter_interface.descriptor->stage_layer_counts[index] ==
			SPARK_LAGUNA_MODEL_LAYER_COUNT / 2u);
	assert(library.adapter_interface.descriptor->boundary_element_count ==
		SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT);
	assert(library.adapter_interface.descriptor->boundary_element_bytes == 2u);
	assert(library.adapter_interface.descriptor->cache_block_token_count ==
		SPARK_LAGUNA_MODEL_KV_PAGE_SLOTS);
	TestLagunaServingInterfaceCompleteness(&library.adapter_interface);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	boundary_bytes = (uint64_t)SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT *
		SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	hidden_input = calloc(1u,(size_t)boundary_bytes);
	hidden_output = calloc(1u,(size_t)boundary_bytes);
	assert(hidden_input != 0 && hidden_output != 0);
	for (rank=0u; rank<2u; rank++)
	{
		const uint32_t stage_rank = rank == 0u ?
			TEST_LAGUNA_FIRST_STAGE_RANK : TEST_LAGUNA_LAST_STAGE_RANK;
		TestLagunaServingConfiguration(&configuration,stage_rank,runtime_root,
			&test_state);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) ==
			SPARK_STATUS_OK);
		assert(adapter_state != 0);
		TestLagunaServingDecodeSubmission(&submission,&lane,&token_id,&row_lane,
			&row_position,&row_sequence);
		/* unpaired: a hidden pointer without its byte count is a broken
		   boundary regardless of stage (common validator pairing) */
		TestLagunaServingApplyBoundaries(&submission,stage_rank,hidden_input,
			hidden_output,boundary_bytes);
		submission.hidden_input_address = hidden_input;
		submission.hidden_input_bytes = 0u;
		assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
			SPARK_STATUS_INVALID_ARGUMENT);
		/* sidebands: the laguna descriptor declares none - fail closed
		   on the raw path */
		TestLagunaServingApplyBoundaries(&submission,stage_rank,hidden_input,
			hidden_output,boundary_bytes);
		submission.boundary_sideband_output_address = hidden_output;
		submission.boundary_sideband_output_bytes = boundary_bytes;
		assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
			SPARK_STATUS_INVALID_ARGUMENT);
		/* raw wire form: the residentd validates before the route bind
		   fills the hidden transport - this form must pass everywhere */
		TestLagunaServingApplyBoundaries(&submission,stage_rank,0,0,0u);
		assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
			SPARK_STATUS_OK);
		assert(library.adapter_interface.prefetch(adapter_state,&submission,1u) ==
			SPARK_STATUS_OK);
		assert(library.adapter_interface.resolve_prefetch(adapter_state,&submission,
			SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT) == SPARK_STATUS_OK);
		test_state.completion_count = 0u;
		memset(&test_state.completion,0,sizeof(test_state.completion));
		assert(library.adapter_interface.submit(adapter_state,&submission) ==
			SPARK_STATUS_OK);
		assert(test_state.completion_count == 1u);
		assert(test_state.completion.submission_id == submission.submission_id);
		assert(test_state.completion.token_count == 1u);
		assert(test_state.completion.token_ids[0] == TEST_LAGUNA_RAW_TOKEN);
		/* route-bound form: exactly the stage's own boundary side wired
		   (the attach-011d first-decode abort rejected this form at
		   submission validation with CAPACITY_EXCEEDED - both pipeline
		   stages must now accept and execute it). The frame contract
		   follows the route plan: a shipping stage carries the
		   HIDDEN_OUTPUT flag and NO token buffer (no ids materialize
		   off the final stage); the consuming stage carries
		   HIDDEN_INPUT plus the WRITE buffer and emits the stage
		   receipt token. */
		TestLagunaServingApplyBoundaries(&submission,stage_rank,hidden_input,
			hidden_output,boundary_bytes);
		assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
			SPARK_STATUS_OK);
		assert(library.adapter_interface.prefetch(adapter_state,&submission,1u) ==
			SPARK_STATUS_OK);
		assert(library.adapter_interface.resolve_prefetch(adapter_state,&submission,
			SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT) == SPARK_STATUS_OK);
		test_state.completion_count = 0u;
		memset(&test_state.completion,0,sizeof(test_state.completion));
		assert(library.adapter_interface.submit(adapter_state,&submission) ==
			SPARK_STATUS_OK);
		assert(test_state.completion_count == 1u);
		assert(test_state.completion.token_count == 1u);
		if ( stage_rank / 8u != 0u )
			assert(test_state.completion.token_ids[0] == TEST_LAGUNA_STAGE1_TOKEN);
		/* the shipping stage completes WITHOUT a token buffer: the
		   driver fixture rejects any WRITE buffer on a hidden_output
		   frame, so no fresh ids materialize - the stale pending-slot
		   value from the raw submit above is a don't-care, exactly as
		   in production where only the final stage writes ids */
		assert(library.adapter_interface.snapshot(adapter_state,&snapshot) ==
			SPARK_STATUS_OK);
		assert(snapshot.submitted_count == 2u);
		assert(snapshot.completed_count == 2u);
		library.adapter_interface.destroy(adapter_state);
	}
	/* the config fixture carries tp_rank 0: a rank whose stage_index is
	   not 0 mod 8 must fail the tp-rank identity check at initialize */
	TestLagunaServingConfiguration(&configuration,1u,runtime_root,&test_state);
	adapter_state = 0;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) ==
		SPARK_STATUS_SCHEMA_ERROR);
	assert(adapter_state == 0);
	SparkModelServingAdapterUnloadInterface(&library);
	free(hidden_input);
	free(hidden_output);
	assert(cudaStreamDestroy((cudaStream_t)test_state.execution_stream) == cudaSuccess);
	return(0);
}
