#include <assert.h>
#include <stdint.h>
#include <string.h>

#define main SparkModelResidentdProgramMain
#include "../node/model_residentd.c"
#undef main

static void TestTransportDeadlineQuarantinesUntilTerminal(uint32_t wait_state)
{
	SparkModelResidentdRuntime runtime;
	SparkModelResidentdRoute route;
	SparkModelResidentdSequenceSlot sequence_slot;
	SparkModelResidentdOutput output;
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingLane lane;
	SparkHiddenTransportCompletion transfer;
	uint8_t output_message[SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES];
	SparkStatus status;
	memset(&runtime,0,sizeof(runtime));
	memset(&route,0,sizeof(route));
	memset(&sequence_slot,0,sizeof(sequence_slot));
	memset(&output,0,sizeof(output));
	memset(&descriptor,0,sizeof(descriptor));
	memset(&lane,0,sizeof(lane));
	memset(&transfer,0,sizeof(transfer));
	assert(pthread_mutex_init(&runtime.mutex,0) == 0);
	runtime.routes = &route;
	runtime.route_capacity = 1u;
	runtime.sequence_slots = &sequence_slot;
	runtime.runtime_limits.resident_sequence_capacity = 1u;
	runtime.adapter_library.adapter_interface.descriptor = &descriptor;
	runtime.client.fd = 1;
	runtime.client.generation = 7u;
	runtime.client.output = &output;
	runtime.client.output_capacity = 1u;
	runtime.client.output_message_capacity = sizeof(output_message);
	output.message = output_message;
	lane.request_id = 21u;
	lane.request_generation = 22u;
	lane.step_generation = 23u;
	lane.sequence_id = 24u;
	lane.resident_sequence_slot = 0u;
	lane.context_token_count = 1u;
	route.active = 1u;
	route.slot_index = 0u;
	route.message_id = 31u;
	route.submission_id = 32u;
	route.client_generation = runtime.client.generation;
	route.state = wait_state;
	route.result_queued = 1u;
	route.resident_slots_claimed = 1u;
	route.submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	route.submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	route.submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	route.submission.submission_id = route.submission_id;
	route.submission.request_id = lane.request_id;
	route.submission.sequence_id = lane.sequence_id;
	route.submission.sequence_position = 0u;
	route.submission.deadline_time_ns = 1u;
	route.submission.control_generation = 41u;
	route.submission.transaction_id = 42u;
	route.submission.dispatch_generation = 43u;
	route.submission.request_generation = lane.request_generation;
	route.submission.step_generation = lane.step_generation;
	route.submission.active_sequence_count = 1u;
	route.submission.lane_count = 1u;
	route.submission.lanes = &lane;
	sequence_slot.active_owner = 1u;
	route.input_packet.sequence_id = lane.sequence_id;
	route.input_packet.token_index = 51u;
	route.input_packet.active_sequence_count = 1u;
	route.input_packet.bytes_per_sequence = 8u;
	route.output_packet = route.input_packet;
	transfer.sequence_id = lane.sequence_id;
	transfer.token_index = 51u;
	transfer.active_sequence_count = 1u;
	transfer.transfer_bytes = 8u;
	transfer.status = SPARK_STATUS_OK;

	assert(pthread_mutex_lock(&runtime.mutex) == 0);
	status = SparkModelResidentdExpireTransportRouteLocked(&runtime,&route,
		wait_state);
	assert(status == SPARK_STATUS_OK);
	assert(SparkModelResidentdExpireTransportRouteLocked(&runtime,&route,
		wait_state) == SPARK_STATUS_OK);
	assert(pthread_mutex_unlock(&runtime.mutex) == 0);
	assert(route.deadline_expired != 0u);
	assert(route.deadline_completion_queued != 0u);
	assert(runtime.client.output_count == 1u);
	assert(route.active != 0u && route.state == wait_state);
	assert(route.resident_slots_claimed != 0u);
	assert(sequence_slot.active_owner == 1u);
	assert(SparkModelResidentdFinishRoute(&runtime,&route) == SPARK_STATUS_OK);
	assert(route.active != 0u && route.state == wait_state);
	assert(SparkModelResidentdReserveRoute(&runtime,&route.submission,99u) == 0);

	assert(pthread_mutex_lock(&runtime.mutex) == 0);
	status = SparkModelResidentdApplyTransportCompletionLocked(&runtime,&transfer,
		wait_state == SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT ? 1u : 0u);
	assert(pthread_mutex_unlock(&runtime.mutex) == 0);
	assert(status == SPARK_STATUS_OK);
	assert(route.active != 0u &&
		route.state == SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION);
	assert(route.resident_slots_claimed != 0u);
	assert(sequence_slot.active_owner == 1u);
	assert(SparkModelResidentdFinishRoute(&runtime,&route) == SPARK_STATUS_OK);
	assert(route.active == 0u);
	assert(route.resident_slots_claimed == 0u);
	assert(sequence_slot.active_owner == 0u);
	assert(runtime.client.output_count == 1u);
	assert(pthread_mutex_destroy(&runtime.mutex) == 0);
}

static void TestPersistentSlotRequiresRelease(void)
{
	SparkModelResidentdRuntime runtime = {0};
	SparkModelResidentdRoute route = {0};
	SparkModelResidentdSequenceSlot slot = {0};
	SparkModelServingLane lane = {0};
	runtime.sequence_slots = &slot;
	route.submission.lanes = &lane;
	route.submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	lane.request_id = 1u;
	lane.request_generation = 1u;
	lane.sequence_id = 1u;
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_OK);
	slot.bound = 1u;
	slot.request_id = 1u;
	slot.request_generation = 1u;
	slot.sequence_id = 1u;
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_OK);
	lane.sequence_id = 2u;
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	route.submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	lane.sequence_id = 1u;
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_OK);
	slot.bound = 0u;
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_NOT_FOUND);
	route.submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	lane.sequence_id = 2u;
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_OK);
}

static void TestDescriptor(SparkModelServingAdapterDescriptor *descriptor)
{
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN;
	descriptor->stage_count = 1u;
	descriptor->layer_count = 1u;
	descriptor->stage_layer_counts[0] = 1u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 32u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->cache_block_token_count = 64u;
	descriptor->max_inflight_submission_count = 1u;
	descriptor->max_active_sequence_count = 8u;
	descriptor->max_input_row_count = 8u;
	descriptor->max_resident_sequence_count = 8u;
	descriptor->max_output_token_count = 64u;
	descriptor->max_speculative_token_count = 2u;
	descriptor->adapter_id = "a";
	descriptor->model_id = "m";
	descriptor->model_revision = "r";
	descriptor->driver_program_name = "p";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
}

typedef struct TestCompletionFixture
{
	SparkModelResidentdRuntime runtime;
	SparkModelResidentdRoute route;
	SparkModelResidentdSequenceSlot sequence_slot;
	SparkModelResidentdOutput output;
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingLane lane;
	SparkModelServingCompletion completion;
	uint8_t output_message[SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES];
}
TestCompletionFixture;

static void TestCompletionInitialize(TestCompletionFixture *fixture,uint32_t completion_status)
{
	SparkModelResidentdRuntime *runtime = &fixture->runtime;
	SparkModelResidentdRoute *route = &fixture->route;
	SparkModelServingCompletion *completion = &fixture->completion;
	memset(fixture,0,sizeof(*fixture));
	TestDescriptor(&fixture->descriptor);
	assert(pthread_mutex_init(&runtime->mutex,0) == 0);
	atomic_init(&runtime->failed_status,SPARK_STATUS_OK);
	runtime->listen_fd = runtime->candidate_fd = runtime->wake_read_fd = runtime->wake_write_fd = -1;
	runtime->routes = route;
	runtime->route_capacity = 1u;
	runtime->sequence_slots = &fixture->sequence_slot;
	runtime->runtime_limits.resident_sequence_capacity = 1u;
	runtime->adapter_library.adapter_interface.descriptor = &fixture->descriptor;
	runtime->client.fd = 1;
	runtime->client.generation = 7u;
	runtime->client.output = &fixture->output;
	runtime->client.output_capacity = 1u;
	runtime->client.output_message_capacity = sizeof(fixture->output_message);
	fixture->output.message = fixture->output_message;
	route->active = 1u;
	route->message_id = 31u;
	route->submission_id = route->request_id = route->sequence_id = route->sequence_position = 1234u;
	route->client_generation = runtime->client.generation;
	route->result_queued = 1u;
	route->state = SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER;
	route->submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	route->submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	route->submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	route->submission.submission_id = route->submission.request_id = route->submission.sequence_id = route->submission.sequence_position = 1234u;
	route->submission.control_generation = route->submission.transaction_id = route->submission.dispatch_generation = 1234u;
	route->submission.request_generation = route->submission.step_generation = 1234u;
	route->submission.residency.word0 = route->submission.residency.generation = 1234u;
	route->submission.residency.owner = 1u;
	route->submission.active_sequence_count = route->submission.lane_count = route->submission.new_token_count = 1u;
	route->submission.tokens_per_sequence = 8u;
	route->submission.lanes = &fixture->lane;
	completion->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion->descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion->status = completion_status;
	completion->submission_id = completion->request_id = completion->sequence_id = completion->sequence_position = 1234u;
	completion->control_generation = completion->transaction_id = completion->dispatch_generation = completion->request_generation = completion->step_generation = 1234u;
	completion->residency = route->submission.residency;
}

static void TestRejectedCompletionFailsRouteWithItsStatus(uint32_t completion_status,uint32_t foreign_residency,SparkStatus expected)
{
	TestCompletionFixture fixture;
	SparkModelServingCompletion delivered = {0};
	TestCompletionInitialize(&fixture,completion_status);
	fixture.completion.residency.generation += foreign_residency;
	SparkModelResidentdCompletion(&fixture.runtime,&fixture.completion);
	assert(fixture.route.active != 0u && fixture.route.state == SPARK_MODEL_RESIDENTD_ROUTE_FAILED && fixture.route.failure_status == (uint32_t)expected);
	assert(SparkModelResidentdFinalizeFailedRoute(&fixture.runtime,&fixture.route) == SPARK_STATUS_OK);
	assert(fixture.route.active == 0u && fixture.runtime.client.output_count == 1u);
	assert(SparkModelResidentIpcDecodeCompletion(fixture.output.message,fixture.output.message_bytes,&delivered) == SPARK_STATUS_OK);
	assert(delivered.status == (uint32_t)expected && delivered.submission_id == 1234u);
	assert(pthread_mutex_destroy(&fixture.runtime.mutex) == 0);
}

static void TestStrayCompletionLeavesRoute(void)
{
	TestCompletionFixture fixture;
	TestCompletionInitialize(&fixture,SPARK_STATUS_OK);
	fixture.route.state = SPARK_MODEL_RESIDENTD_ROUTE_FAILED;
	fixture.route.failure_status = SPARK_STATUS_IO_ERROR;
	SparkModelResidentdCompletion(&fixture.runtime,&fixture.completion);
	assert(fixture.route.active != 0u && fixture.route.state == SPARK_MODEL_RESIDENTD_ROUTE_FAILED && fixture.route.failure_status == SPARK_STATUS_IO_ERROR && fixture.route.completion.abi_version == 0u);
	assert(pthread_mutex_destroy(&fixture.runtime.mutex) == 0);
}

static void TestStuckRouteSetUp(SparkModelResidentdRoute *route, uint32_t slot_index, uint64_t submission_id, uint32_t fifo_next)
{
	route->active = 1u;
	route->active_since_ns = 1u;
	route->slot_index = slot_index;
	route->submission_id = submission_id;
	route->state = SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT;
	route->committed_fifo_queued = fifo_next != UINT32_MAX ? 1u : 0u;
	route->committed_fifo_next = fifo_next != UINT32_MAX ? fifo_next : 0u;
}

static void TestStuckRoutesRepeatWithFifoPosition(void)
{
	SparkModelResidentdRuntime runtime;
	SparkModelResidentdRoute routes[3];
	SparkModelServingLane lanes[2];
	char text[2048],expected[256];
	FILE *capture;
	size_t length;
	int saved;
	memset(&runtime,0,sizeof(runtime));
	memset(routes,0,sizeof(routes));
	memset(lanes,0,sizeof(lanes));
	assert(pthread_mutex_init(&runtime.mutex,0) == 0);
	runtime.routes = routes;
	runtime.route_capacity = 3u;
	TestStuckRouteSetUp(&routes[0],0u,101u,UINT32_MAX);
	TestStuckRouteSetUp(&routes[1],1u,102u,0u);
	TestStuckRouteSetUp(&routes[2],2u,103u,2u);
	runtime.committed_fifo_head = 3u;
	routes[1].resident_slots_claimed = 1u;
	routes[1].submission.active_sequence_count = 2u;
	lanes[0].resident_sequence_slot = 7u;
	routes[1].submission.lanes = lanes;
	fflush(stderr);
	capture = tmpfile();
	assert(capture != 0);
	saved = dup(2);
	assert(saved >= 0 && dup2(fileno(capture),2) >= 0);
	assert(SparkModelResidentdReportStuckRoutesAt(&runtime,UINT64_C(20000000000)) == 0u);
	assert(SparkModelResidentdReportStuckRoutesAt(&runtime,UINT64_C(31000000000)) == 3u);
	assert(SparkModelResidentdReportStuckRoutesAt(&runtime,UINT64_C(41000000000)) == 0u);
	routes[0].state = SPARK_MODEL_RESIDENTD_ROUTE_FAILED;
	assert(SparkModelResidentdReportStuckRoutesAt(&runtime,UINT64_C(51000000000)) == 1u);
	assert(SparkModelResidentdReportStuckRoutesAt(&runtime,UINT64_C(92000000000)) == 2u);
	fflush(stderr);
	assert(dup2(saved,2) >= 0);
	close(saved);
	rewind(capture);
	length = fread(text,1u,sizeof(text) - 1u,capture);
	text[length] = '\0';
	fclose(capture);
	snprintf(expected,sizeof(expected),"ROUTE-STUCK id=103 state=%u age_ms=30999 claimed=0 abandoned=0 gen=0 fifo=0 lanes=0 first_slot=-1\n",(unsigned)SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT);
	assert(strstr(text,expected) != 0);
	snprintf(expected,sizeof(expected),"ROUTE-STUCK id=102 state=%u age_ms=30999 claimed=1 abandoned=0 gen=0 fifo=1 lanes=2 first_slot=7\n",(unsigned)SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT);
	assert(strstr(text,expected) != 0);
	snprintf(expected,sizeof(expected),"ROUTE-STUCK id=101 state=%u age_ms=50999 claimed=0 abandoned=0 gen=0 fifo=-1 lanes=0 first_slot=-1\n",(unsigned)SPARK_MODEL_RESIDENTD_ROUTE_FAILED);
	assert(strstr(text,expected) != 0);
	snprintf(expected,sizeof(expected),"ROUTE-STUCK id=102 state=%u age_ms=91999 claimed=1 abandoned=0 gen=0 fifo=1 lanes=2 first_slot=7\n",(unsigned)SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT);
	assert(strstr(text,expected) != 0);
	pthread_mutex_destroy(&runtime.mutex);
}

int main(void)
{
	TestRejectedCompletionFailsRouteWithItsStatus(SPARK_STATUS_NO_LANE,0u,SPARK_STATUS_INVALID_ARGUMENT);
	TestRejectedCompletionFailsRouteWithItsStatus(SPARK_STATUS_OK,1u,SPARK_STATUS_SCHEMA_ERROR);
	TestStrayCompletionLeavesRoute();
	TestPersistentSlotRequiresRelease();
	TestStuckRoutesRepeatWithFifoPosition();
	TestTransportDeadlineQuarantinesUntilTerminal(
		SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT);
	TestTransportDeadlineQuarantinesUntilTerminal(
		SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT);
	return(0);
}
