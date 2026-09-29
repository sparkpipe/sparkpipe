#include "modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c"

#include "inference/llms/kimi_k3/generated_config.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#define TEST_K3_DEPLOYMENT "modules/k3_resident_decode_stage/configs/model_resident.json"

SparkStatus SparkK3StageRunnerInitialize(SparkK3StageRunner *runner,
	const SparkK3StageRunnerConfiguration *configuration)
{
	(void)runner;
	(void)configuration;
	return(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkK3StageRunnerSubmit(SparkK3StageRunner *runner,
	const SparkK3StageRunnerDispatch *dispatch)
{
	(void)runner;
	(void)dispatch;
	return(SPARK_STATUS_UNSUPPORTED);
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
		descriptor->capability_flags) == SPARK_STATUS_OK,
		"the common validator accepts the interface and descriptor");
	failures += TestK3Check(descriptor->cache_block_token_count == SPARK_K3_KV_PAGE_SLOTS &&
		descriptor->cache_block_token_count == K3_KV_PAGE_SLOTS,
		"the cache block is one KV page of the kernel's slot count");
	failures += TestK3Check(strcmp(descriptor->driver_program_name, "k3") == 0,
		"the descriptor names the k3 driver program");
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
		&K3ServingDescriptor) == SPARK_STATUS_OK,
		"residentd's adapter validation accepts the deployment");
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

int main(int argc, char **argv)
{
	int32_t failures = TestK3Descriptor();
	failures += TestK3Release();
	int index;
	if ( argc == 1 )
		failures += TestK3Deployment(TEST_K3_DEPLOYMENT);
	for (index = 1; index < argc; index++)
		failures += TestK3Deployment(argv[index]);
	printf("test_k3_serving_adapter: %d failures\n", (int)failures);
	return(failures != 0 ? 1 : 0);
}
