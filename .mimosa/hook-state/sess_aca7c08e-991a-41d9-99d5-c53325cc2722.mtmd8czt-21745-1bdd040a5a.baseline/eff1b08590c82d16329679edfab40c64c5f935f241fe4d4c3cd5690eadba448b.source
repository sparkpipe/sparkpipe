#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_model_serving_adapter.h"

#ifndef TEST_DSV4_TP4_PP4_ADAPTER_PATH
#define TEST_DSV4_TP4_PP4_ADAPTER_PATH ""
#endif
#ifndef TEST_DSV4_TP4_PP4_DRIVER_PATH
#define TEST_DSV4_TP4_PP4_DRIVER_PATH ""
#endif
#ifndef TEST_DSV4_TP4_PP4_CONFIG_PATH
#define TEST_DSV4_TP4_PP4_CONFIG_PATH ""
#endif
#ifndef TEST_DSV4_TP4_PP4_MISSING_GRAPHS_CONFIG_PATH
#define TEST_DSV4_TP4_PP4_MISSING_GRAPHS_CONFIG_PATH ""
#endif
#ifndef TEST_DSV4_TP4_PP4_SHORT_GRAPHS_CONFIG_PATH
#define TEST_DSV4_TP4_PP4_SHORT_GRAPHS_CONFIG_PATH ""
#endif

typedef struct TestDsv4Tp4Pp4CompletionState
{
	uint32_t count;
	SparkModelServingCompletion completion;
} TestDsv4Tp4Pp4CompletionState;

static void TestDsv4Tp4Pp4Completion(
	void *context,
	const SparkModelServingCompletion *completion)
{
	TestDsv4Tp4Pp4CompletionState *state;
	state = (TestDsv4Tp4Pp4CompletionState *)context;
	assert(state != 0 && completion != 0);
	state->completion = *completion;
	state->count++;
}

static void TestDsv4Tp4Pp4BuildSubmission(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lane,
	uint32_t rank,
	uint32_t *token_id,
	uint32_t *row_lane,
	uint64_t *row_position,
	uint64_t *row_sequence,
	void *hidden_input,
	void *hidden_output,
	uint64_t hidden_bytes)
{
	memset(lane,0,sizeof(*lane));
	lane->request_id = 900u + rank;
	lane->request_generation = 1u;
	lane->step_generation = 1u;
	lane->sequence_id = 100u;
	lane->resident_sequence_slot = 127u;
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
	submission->submission_id = 1000u + rank;
	submission->request_id = 900u + rank;
	submission->sequence_id = 100u;
	submission->control_generation = 1u;
	submission->transaction_id = 2000u + rank;
	submission->dispatch_generation = 3000u + rank;
	submission->request_generation = 1u;
	submission->step_generation = 4000u + rank;
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
	if ( rank / 4u != 0u )
	{
		submission->hidden_input_address = hidden_input;
		submission->hidden_input_bytes = hidden_bytes;
	}
	if ( rank / 4u + 1u < 4u )
	{
		submission->hidden_output_address = hidden_output;
		submission->hidden_output_bytes = hidden_bytes;
	}
}

int main(void)
{
	static const uint32_t expected_layers[16] =
	{
		11u,11u,11u,11u,11u,11u,11u,11u,
		11u,11u,11u,11u,10u,10u,10u,10u
	};
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	SparkModelServingAdapterSnapshot snapshot;
	TestDsv4Tp4Pp4CompletionState completion_state;
	uint32_t token_id,row_lane,rank;
	uint64_t row_position,row_sequence;
	uint64_t hidden_bytes;
	void *adapter_state,*hidden_input,*hidden_output;
	char node_id[32],runtime_root[4096];
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(
		TEST_DSV4_TP4_PP4_ADAPTER_PATH,
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV,
		&library) == SPARK_STATUS_OK);
	assert(library.adapter_interface.descriptor->stage_count == 16u);
	assert(library.adapter_interface.descriptor->parallel_group_size == 4u);
	assert(library.adapter_interface.descriptor->max_inflight_submission_count == 4u);
	for (rank=0u; rank<16u; rank++)
		assert(library.adapter_interface.descriptor->stage_layer_counts[rank] ==
			expected_layers[rank]);
	hidden_bytes = (uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	hidden_input = calloc(1u,(size_t)hidden_bytes);
	hidden_output = calloc(1u,(size_t)hidden_bytes);
	assert(hidden_input != 0 && hidden_output != 0);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	for (rank=0u; rank<16u; rank++)
	{
		memset(&configuration,0,sizeof(configuration));
		configuration.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
		configuration.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
		configuration.rank_index = rank;
		configuration.stage_index = rank;
		configuration.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
		configuration.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
		configuration.runtime_limits.max_inflight_submission_count = 1u;
		configuration.runtime_limits.max_active_sequence_count = 1u;
		configuration.runtime_limits.max_input_row_count = 1u;
		configuration.runtime_limits.resident_sequence_capacity = 128u;
		configuration.runtime_limits.kv_logical_page_capacity = 128u;
		configuration.runtime_limits.kv_physical_page_capacity = 128u;
		(void)snprintf(node_id,sizeof(node_id),"spark%u",rank);
		configuration.runtime_root = runtime_root;
		configuration.node_id = node_id;
		configuration.node_target = SPARK_DSV4_MODEL_MODULE_TARGET;
		configuration.adapter_configuration_path = TEST_DSV4_TP4_PP4_CONFIG_PATH;
		configuration.driver_shared_object_path = TEST_DSV4_TP4_PP4_DRIVER_PATH;
		configuration.driver_program_name = "resident_decode";
		configuration.execution_stream = (void *)(uintptr_t)1u;
		memset(&completion_state,0,sizeof(completion_state));
		configuration.completion_function = TestDsv4Tp4Pp4Completion;
		configuration.completion_context = &completion_state;
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) ==
			SPARK_STATUS_OK);
		assert(adapter_state != 0);
		assert(library.adapter_interface.snapshot(adapter_state,&snapshot) ==
			SPARK_STATUS_OK);
		assert(snapshot.kv_token_capacity ==
			(rank / 4u < 3u ? 34u : 31u));
		TestDsv4Tp4Pp4BuildSubmission(&submission,&lane,rank,&token_id,
			&row_lane,&row_position,&row_sequence,hidden_input,hidden_output,
			hidden_bytes);
		assert(SparkModelServingAdapterPrepareSubmission(
			&library.adapter_interface,adapter_state,&submission) == SPARK_STATUS_OK);
		assert(SparkModelServingAdapterResolvePrefetch(
			&library.adapter_interface,adapter_state,&submission,
			SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT) == SPARK_STATUS_OK);
		assert(library.adapter_interface.submit(adapter_state,&submission) ==
			SPARK_STATUS_OK);
		assert(completion_state.count == 1u);
		assert(completion_state.completion.submission_id ==
			submission.submission_id);
		assert(completion_state.completion.dispatch_generation ==
			submission.dispatch_generation);
		if ( rank == 15u )
		{
			assert(completion_state.completion.token_count == 1u);
			assert(completion_state.completion.token_ids[0] == 4200u);
		}
		else
			assert(completion_state.completion.token_count == 0u);
		library.adapter_interface.destroy(adapter_state);
	}
	configuration.adapter_configuration_path =
		TEST_DSV4_TP4_PP4_MISSING_GRAPHS_CONFIG_PATH;
	adapter_state = 0;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) ==
		SPARK_STATUS_SCHEMA_ERROR);
	assert(adapter_state == 0);
	configuration.adapter_configuration_path =
		TEST_DSV4_TP4_PP4_SHORT_GRAPHS_CONFIG_PATH;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) ==
		SPARK_STATUS_SCHEMA_ERROR);
	assert(adapter_state == 0);
	SparkModelServingAdapterUnloadInterface(&library);
	free(hidden_input);
	free(hidden_output);
	return(0);
}
