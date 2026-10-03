#include "modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c"

#include "inference/llms/kimi_k3/generated_config.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#if SPARK_K3_SERVING_TOPOLOGY == 16
#define TEST_K3_DEPLOYMENT "modules/k3_resident_decode_stage/configs/model_resident_tp16.json"
#else
#define TEST_K3_DEPLOYMENT "modules/k3_resident_decode_stage/configs/model_resident.json"
#endif

SparkStatus SparkK3StageRunnerInitialize(SparkK3StageRunner *runner,
	const SparkK3StageRunnerConfiguration *configuration)
{
	(void)runner;
	(void)configuration;
	return(SPARK_STATUS_UNSUPPORTED);
}

static uint32_t test_dispatch_rows;
static uint32_t test_dispatch_sequences;
static uint32_t test_dispatch_row_begin[9];
static uint32_t test_dispatch_state_index[8];
static uint32_t test_dispatch_sequence_of_row[8];
static uint32_t test_dispatch_context[4];
static uint32_t test_dispatch_order[8];

SparkStatus SparkK3StageRunnerSubmit(SparkK3StageRunner *runner,
	const SparkK3StageRunnerDispatch *dispatch)
{
	(void)runner;
	test_dispatch_rows = dispatch->row_count;
	test_dispatch_sequences = dispatch->active_sequence_count;
	memset(test_dispatch_row_begin, 0xff, sizeof(test_dispatch_row_begin));
	memset(test_dispatch_state_index, 0xff, sizeof(test_dispatch_state_index));
	if ( dispatch->row_count > 8u || dispatch->active_sequence_count > 8u ||
		dispatch->sequence_row_begin == 0 || dispatch->kda_state_index == 0 ||
		dispatch->sequence_of_row == 0 )
		return(SPARK_STATUS_OK);
	memcpy(test_dispatch_row_begin, dispatch->sequence_row_begin,
		((size_t)dispatch->active_sequence_count + 1u) * sizeof(uint32_t));
	memcpy(test_dispatch_state_index, dispatch->kda_state_index,
		(size_t)dispatch->active_sequence_count * sizeof(uint32_t));
	memcpy(test_dispatch_sequence_of_row, dispatch->sequence_of_row,
		(size_t)dispatch->row_count * sizeof(uint32_t));
	if ( dispatch->context_length != 0 )
		memcpy(test_dispatch_context, dispatch->context_length, sizeof(test_dispatch_context));
	memset(test_dispatch_order, 0xff, sizeof(test_dispatch_order));
	if ( dispatch->sequence_row_indices != 0 )
		memcpy(test_dispatch_order, dispatch->sequence_row_indices,
			(size_t)dispatch->row_count * sizeof(uint32_t));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3StageRunnerGetStats(const SparkK3StageRunner *runner,
	SparkK3StageRunnerStats *stats_out)
{
	(void)runner;
	if ( stats_out != 0 )
		memset(stats_out, 0, sizeof(*stats_out));
	return(SPARK_STATUS_UNSUPPORTED);
}

void SparkK3StageRunnerDestroy(SparkK3StageRunner *runner)
{
	(void)runner;
}

static uint32_t test_reset_slots[8];
static uint32_t test_reset_count;
static uint32_t test_reset_calls;
static SparkStatus test_reset_status = SPARK_STATUS_OK;
static uint32_t test_completions;

SparkStatus SparkK3StageRunnerResetSlots(SparkK3StageRunner *runner,
	const uint32_t *slots, uint32_t count)
{
	(void)runner;
	test_reset_calls++;
	test_reset_count = count;
	for ( uint32_t index = 0u; index < count && index < 8u; index++ )
		test_reset_slots[index] = slots[index];
	return(test_reset_status);
}

static void TestK3Completion(void *context, const SparkModelServingCompletion *completion)
{
	(void)context;
	(void)completion;
	test_completions++;
}

static int32_t TestK3Check(int32_t condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

static int32_t TestK3Descriptor(void)
{
	const SparkModelServingAdapterInterface *adapter = SparkModelServingAdapterGetInterface();
	const SparkModelServingAdapterDescriptor *descriptor = &K3ServingDescriptor;
	uint32_t index, stage_layers = 0u, tiled = 1u;
	int32_t failures = 0;
	failures += TestK3Check(adapter != 0 && adapter->descriptor == descriptor,
		"the adapter interface exports the K3 descriptor");
	failures += TestK3Check(SparkModelServingAdapterValidateInterface(adapter,
		0u) == SPARK_STATUS_UNSUPPORTED,
		"the common validator refuses the K3 adapter: it cannot restore cached prompt prefixes (I23)");
	failures += TestK3Check(descriptor->cache_block_token_count == SPARK_K3_KV_PAGE_SLOTS &&
		descriptor->cache_block_token_count == K3_KV_PAGE_SLOTS,
		"the cache block is one KV page of the kernel's slot count");
	failures += TestK3Check(strcmp(descriptor->driver_program_name, "k3") == 0,
		"the descriptor names the k3 driver program");
#if SPARK_K3_SERVING_TOPOLOGY == 16
	failures += TestK3Check(strcmp(descriptor->adapter_id, "k3-tp16") == 0 &&
		descriptor->parallel_group_size == 0u && descriptor->stage_count == 16u &&
		(descriptor->capability_flags &
			(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP |
			 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT)) == 0u &&
		(descriptor->capability_flags &
			SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT) != 0u,
		"TP16: sixteen parallel ranks, no pipeline hand-off");
	for (index = 0u; index < SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT; index++)
		if ( (index < 16u && descriptor->stage_layer_counts[index] !=
				SPARK_K3_MODEL_LAYER_COUNT) ||
			(index < 12u && (descriptor->boundary_sideband_kinds[index] != 0u ||
				descriptor->boundary_sideband_bytes_per_sequence[index] != 0u)) )
			tiled = 0u;
	failures += TestK3Check(tiled != 0u &&
		descriptor->layer_count == SPARK_K3_MODEL_LAYER_COUNT,
		"TP16: every rank carries all 93 layers and no residual-bank sideband");
	return(failures);
#endif
	failures += TestK3Check(descriptor->parallel_group_size != 0u &&
		descriptor->stage_count == SPARK_K3_PP_STAGE_COUNT * descriptor->parallel_group_size,
		"the ranks are the PP stages times the TP group");
	for (index = 0u; index < descriptor->stage_count && descriptor->parallel_group_size != 0u; index++)
	{
		if ( descriptor->stage_layer_counts[index] !=
			SPARK_K3_PP_STAGE_LAYERS(index / descriptor->parallel_group_size) )
			tiled = 0u;
		if ( index % descriptor->parallel_group_size == 0u )
			stage_layers += descriptor->stage_layer_counts[index];
	}
	failures += TestK3Check(tiled != 0u && stage_layers == SPARK_K3_MODEL_LAYER_COUNT &&
		descriptor->layer_count == SPARK_K3_MODEL_LAYER_COUNT,
		"every rank carries its PP stage's layers and the stages tile the model");
	return(failures);
}

static int32_t TestK3Deployment(const char *path)
{
	SparkModelResidentDeployment deployment;
	SparkModelResidentDeployment zero_pages;
	SparkModelResidentDeployment short_pages;
	int32_t failures = 0;
	printf("deployment %s\n", path);
	memset(&deployment, 0, sizeof(deployment));
	if ( TestK3Check(SparkModelResidentDeploymentLoad(path, &deployment) == SPARK_STATUS_OK,
		"residentd's loader parses the deployment") != 0 )
		return(1);
	failures += TestK3Check(SparkModelResidentDeploymentValidateForAdapter(&deployment,
		&K3ServingDescriptor) == SPARK_STATUS_UNSUPPORTED,
		"residentd refuses to serve the K3 deployment until the adapter restores cached prefixes (I23)");
	failures += TestK3Check(deployment.eos_token_count == 1u &&
		deployment.eos_token_ids[0] == K3_EOS_TOKEN,
		"the model EOS is the authoritative contract's end_of_text");
	zero_pages = deployment;
	zero_pages.runtime_limits.kv_logical_page_capacity = 0u;
	zero_pages.runtime_limits.kv_physical_page_capacity = 0u;
	failures += TestK3Check(SparkModelResidentDeploymentValidateForAdapter(&zero_pages,
		&K3ServingDescriptor) != SPARK_STATUS_OK,
		"zero KV page capacities are refused");
	short_pages = deployment;
	short_pages.runtime_limits.kv_physical_page_capacity =
		deployment.runtime_limits.max_active_sequence_count - 1u;
	failures += TestK3Check(SparkModelResidentDeploymentValidateForAdapter(&short_pages,
		&K3ServingDescriptor) != SPARK_STATUS_OK,
		"fewer physical pages than active sequences are refused");
	SparkModelResidentDeploymentDestroy(&deployment);
	return(failures);
}

static int32_t TestK3Release(void)
{
	SparkK3ServingState state;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[3];
	int32_t failures = 0;
	memset(&state, 0, sizeof(state));
	state.runner_config.max_active_sequence_count = 4u;
	state.completion_function = TestK3Completion;
	memset(&submission, 0, sizeof(submission));
	memset(lanes, 0, sizeof(lanes));
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	submission.lanes = lanes;
	submission.lane_count = 2u;
	submission.active_sequence_count = 2u;
	lanes[0].resident_sequence_slot = 3u;
	lanes[1].resident_sequence_slot = 1u;
	failures += TestK3Check(K3ServingValidateSubmission(&state, &submission) == SPARK_STATUS_OK &&
		K3ServingSubmit(&state, &submission) == SPARK_STATUS_OK &&
		test_reset_calls == 1u && test_reset_count == 2u &&
		test_reset_slots[0] == 3u && test_reset_slots[1] == 1u && test_completions == 1u,
		"RELEASE resets every released slot's KDA state before it completes");
	test_reset_status = SPARK_STATUS_IO_ERROR;
	failures += TestK3Check(K3ServingSubmit(&state, &submission) == SPARK_STATUS_IO_ERROR &&
		test_reset_calls == 2u && test_completions == 1u,
		"a failed slot reset fails the RELEASE and completes nothing");
	test_reset_status = SPARK_STATUS_OK;
	lanes[1].resident_sequence_slot = 4u;
	failures += TestK3Check(K3ServingValidateSubmission(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT &&
		K3ServingSubmit(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT &&
		test_reset_calls == 2u && test_completions == 1u,
		"a released slot outside the state pool is refused before any reset");
	lanes[1].resident_sequence_slot = 3u;
	failures += TestK3Check(K3ServingSubmit(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT &&
		test_reset_calls == 2u,
		"a slot released twice in one RELEASE is refused");
	lanes[1].resident_sequence_slot = 1u;
	submission.lanes = 0;
	failures += TestK3Check(K3ServingSubmit(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT &&
		test_reset_calls == 2u,
		"a RELEASE naming sequences without lanes is refused");
	submission.lanes = lanes;
	submission.row_count = 1u;
	failures += TestK3Check(K3ServingValidateSubmission(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT &&
		K3ServingSubmit(&state, &submission) == SPARK_STATUS_INVALID_ARGUMENT,
		"a RELEASE with rows is refused");
	return(failures);
}

static int32_t TestK3PrefillRuns(void)
{
	static const uint64_t positions[5] = { 0u, 1u, 2u, 7u, 8u };
	static const uint32_t lane_of_row[5] = { 0u, 0u, 0u, 1u, 1u };
	static const uint32_t tokens[5] = { 1008u, 10484u, 318u, 17374u, 13u };
	SparkK3ServingState state;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	SparkStatus status = SPARK_STATUS_OK;
	SparkMemoryBuffer *host[] = { &state.positions_host, &state.context_host,
		&state.state_host, &state.runs_host, &state.seqslot_host, &state.order_host };
	SparkMemoryBuffer *device[] = { &state.positions_device, &state.context_device,
		&state.state_device, &state.runs_device, &state.seqslot_device,
		&state.order_device, &state.output_tokens, &state.output_scores };
	int32_t failures = 0;
	uint32_t index;
	memset(&state, 0, sizeof(state));
	state.max_rows = 8u;
	state.runner_config.max_active_sequence_count = 4u;
	for ( index = 0u; status == SPARK_STATUS_OK && index < sizeof(host) / sizeof(host[0]); index++ )
		status = SparkMemoryBufferAllocate(host[index], SPARK_MEMORY_SPACE_HOST_COHERENT, 9u * sizeof(uint32_t));
	for ( index = 0u; status == SPARK_STATUS_OK && index < sizeof(device) / sizeof(device[0]); index++ )
		status = SparkMemoryBufferAllocate(device[index], SPARK_MEMORY_SPACE_DEVICE_PRIVATE, 9u * sizeof(uint32_t));
	if ( TestK3Check(status == SPARK_STATUS_OK, "the submit buffers allocate") != 0 )
		return(1);
	memset(&submission, 0, sizeof(submission));
	memset(lanes, 0, sizeof(lanes));
	lanes[0].resident_sequence_slot = 2u;
	lanes[1].resident_sequence_slot = 0u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.lanes = lanes;
	submission.lane_count = 2u;
	submission.active_sequence_count = 2u;
	submission.row_count = 5u;
	submission.row_positions = positions;
	submission.row_lane_indices = lane_of_row;
	submission.token_ids = tokens;
	status = K3ServingSubmit(&state, &submission);
	failures += TestK3Check(status == SPARK_STATUS_OK && test_dispatch_rows == 5u &&
		test_dispatch_sequences == 2u,
		"a two-sequence prefill of five rows dispatches two sequences");
	failures += TestK3Check(test_dispatch_row_begin[0] == 0u && test_dispatch_row_begin[1] == 3u &&
		test_dispatch_row_begin[2] == 5u,
		"the recurrent state walks each sequence's rows in order: row runs 0, 3, 5");
	failures += TestK3Check(test_dispatch_state_index[0] == 2u && test_dispatch_state_index[1] == 0u,
		"the recurrent state is indexed per sequence: slots 2 then 0");
	failures += TestK3Check(test_dispatch_sequence_of_row[0] == 2u && test_dispatch_sequence_of_row[2] == 2u &&
		test_dispatch_sequence_of_row[3] == 0u && test_dispatch_sequence_of_row[4] == 0u,
		"attention KV stays indexed per row");
	failures += TestK3Check(test_dispatch_context[2] == 3u && test_dispatch_context[0] == 9u &&
		test_dispatch_context[1] == 0u && test_dispatch_context[3] == 0u,
		"attention reads each sequence's context by its slot: slot 2 holds 3 tokens, slot 0 holds 9");
	failures += TestK3Check(test_dispatch_order[0] == 0u && test_dispatch_order[2] == 2u &&
		test_dispatch_order[4] == 4u,
		"rows already grouped by sequence keep their order");
	{
		static const uint64_t wave_positions[5] = { 0u, 7u, 1u, 8u, 9u };
		static const uint32_t wave_lane_of_row[5] = { 0u, 1u, 0u, 1u, 1u };
		static const uint64_t decode_positions[2] = { 11u, 4u };
		static const uint32_t decode_lanes[2] = { 1u, 0u };
		static const uint32_t one_lane[5] = { 0u, 0u, 0u, 0u, 0u };
		static const uint32_t past_lanes[5] = { 0u, 1u, 2u, 1u, 1u };
		submission.row_positions = wave_positions;
		submission.row_lane_indices = wave_lane_of_row;
		status = K3ServingSubmit(&state, &submission);
		failures += TestK3Check(status == SPARK_STATUS_OK && test_dispatch_sequences == 2u &&
			test_dispatch_row_begin[1] == 2u && test_dispatch_row_begin[2] == 5u &&
			test_dispatch_order[0] == 0u && test_dispatch_order[1] == 2u &&
			test_dispatch_order[2] == 1u && test_dispatch_order[3] == 3u && test_dispatch_order[4] == 4u,
			"a wave prefill (rows interleaved across sequences) walks each sequence's own rows in order");
		failures += TestK3Check(test_dispatch_sequence_of_row[0] == 2u && test_dispatch_sequence_of_row[1] == 0u &&
			test_dispatch_sequence_of_row[2] == 2u && test_dispatch_context[2] == 2u && test_dispatch_context[0] == 10u,
			"a wave prefill keeps each row's slot and each slot's context");
		test_dispatch_rows = 0u;
		submission.row_lane_indices = one_lane;
		status = K3ServingSubmit(&state, &submission);
		failures += TestK3Check(status == SPARK_STATUS_VALIDATION_FAILED && test_dispatch_rows == 0u,
			"a sequence with no rows is refused before dispatch");
		submission.row_lane_indices = past_lanes;
		status = K3ServingSubmit(&state, &submission);
		failures += TestK3Check(status == SPARK_STATUS_VALIDATION_FAILED && test_dispatch_rows == 0u,
			"a row naming a sequence past the submission's lanes is refused before dispatch");
		submission.row_lane_indices = wave_lane_of_row;
		lanes[0].resident_sequence_slot = 4u;
		status = K3ServingSubmit(&state, &submission);
		failures += TestK3Check(status == SPARK_STATUS_VALIDATION_FAILED && test_dispatch_rows == 0u,
			"a row whose slot is outside the sequence pool is refused before dispatch");
		lanes[0].resident_sequence_slot = 1u;
		lanes[1].resident_sequence_slot = 1u;
		status = K3ServingSubmit(&state, &submission);
		failures += TestK3Check(status == SPARK_STATUS_VALIDATION_FAILED && test_dispatch_rows == 0u,
			"two sequences on one slot are refused before dispatch");
		lanes[0].resident_sequence_slot = 2u;
		submission.row_count = 2u;
		submission.row_positions = decode_positions;
		submission.row_lane_indices = decode_lanes;
		status = K3ServingSubmit(&state, &submission);
		failures += TestK3Check(status == SPARK_STATUS_OK && test_dispatch_context[1] == 12u &&
			test_dispatch_context[2] == 5u && test_dispatch_context[0] == 0u &&
			test_dispatch_state_index[0] == 2u && test_dispatch_state_index[1] == 1u &&
			test_dispatch_order[0] == 1u && test_dispatch_order[1] == 0u,
			"a decode whose rows are not in lane order gives each slot its own context and each lane its own row");
	}
	for ( index = 0u; index < sizeof(host) / sizeof(host[0]); index++ )
		SparkMemoryBufferFree(host[index]);
	for ( index = 0u; index < sizeof(device) / sizeof(device[0]); index++ )
		SparkMemoryBufferFree(device[index]);
	return(failures);
}

int main(int argc, char **argv)
{
	int32_t failures = TestK3Descriptor();
	failures += TestK3Release();
	failures += TestK3PrefillRuns();
	int index;
	if ( argc == 1 )
		failures += TestK3Deployment(TEST_K3_DEPLOYMENT);
	for (index = 1; index < argc; index++)
		failures += TestK3Deployment(argv[index]);
	printf("test_k3_serving_adapter: %d failures\n", (int)failures);
	return(failures != 0 ? 1 : 0);
}
