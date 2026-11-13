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
	assert(SparkModelResidentdValidatePersistentSlot(&runtime,&route,0u) == SPARK_STATUS_OK);
	route.submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH;
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
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE;
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

static void TestFailedCompletionKeepsItsStatus(SparkStatus status)
{
	TestCompletionFixture fixture;
	TestCompletionInitialize(&fixture,(uint32_t)status);
	SparkModelResidentdCompletion(&fixture.runtime,&fixture.completion);
	assert(fixture.route.active != 0u && fixture.route.state == SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION &&
		fixture.route.completion.status == (uint32_t)status && fixture.route.completion.accepted_token_count == 0u);
	assert(pthread_mutex_destroy(&fixture.runtime.mutex) == 0);
}

static uint32_t TestSnapshotFlags;

static SparkStatus TestFlaggedSnapshot(void *adapter_state,SparkModelServingAdapterSnapshot *snapshot)
{
	(void)adapter_state;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->degraded_flags = TestSnapshotFlags;
	return(SPARK_STATUS_OK);
}

static void TestLaneFatalAdapterStopsTheRank(uint32_t flags,SparkStatus expected)
{
	TestCompletionFixture fixture;
	TestCompletionInitialize(&fixture,(uint32_t)SPARK_STATUS_MAXIMUM + 1u);
	TestSnapshotFlags = flags;
	fixture.runtime.adapter_state = &fixture;
	fixture.runtime.adapter_library.adapter_interface.snapshot = TestFlaggedSnapshot;
	SparkModelResidentdCompletion(&fixture.runtime,&fixture.completion);
	assert(fixture.route.state == SPARK_MODEL_RESIDENTD_ROUTE_FAILED);
	assert(SparkModelResidentdFinalizeFailedRoute(&fixture.runtime,&fixture.route) == expected);
	assert(fixture.route.active == 0u && fixture.runtime.client.output_count == 1u);
	assert(atomic_load(&fixture.runtime.failed_status) == (uint32_t)expected);
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

static void TestBoundRouteKeepsTheSingleDecode(void)
{
	static uint8_t message[8192],slot_message[8192];
	SparkModelResidentdRuntime runtime;
	SparkModelResidentdRoute route;
	SparkModelResidentdSlot slot;
	SparkModelResidentdSequenceSlot sequence_slots[8];
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingSubmission submission,decoded,fresh;
	SparkModelServingLane lanes[2];
	SparkModelServingCacheIdentity identity;
	uint32_t tokens[2],row_lanes[2],message_bytes;
	uint64_t positions[2],sequences[2];
	uint8_t extension[3] = {4u,5u,6u};
	memset(&runtime,0,sizeof(runtime));
	memset(&route,0,sizeof(route));
	memset(&slot,0,sizeof(slot));
	memset(sequence_slots,0,sizeof(sequence_slots));
	memset(&submission,0,sizeof(submission));
	memset(lanes,0,sizeof(lanes));
	memset(&identity,0x5a,sizeof(identity));
	TestDescriptor(&descriptor);
	assert(pthread_mutex_init(&runtime.mutex,0) == 0);
	runtime.routes = &route;
	runtime.slots = &slot;
	runtime.route_capacity = 1u;
	runtime.route_message_capacity = sizeof(slot_message);
	runtime.sequence_slots = sequence_slots;
	runtime.runtime_limits.resident_sequence_capacity = 8u;
	runtime.adapter_library.adapter_interface.descriptor = &descriptor;
	slot.message = slot_message;
	for (uint32_t lane=0u; lane<2u; lane++)
	{
		lanes[lane].request_id = 21u + lane;
		lanes[lane].request_generation = 1u;
		lanes[lane].step_generation = 5u;
		lanes[lane].sequence_id = 31u + lane;
		lanes[lane].sequence_position = 63u;
		lanes[lane].context_token_count = 64u;
		lanes[lane].resident_sequence_slot = 3u + lane;
		lanes[lane].input_token_id = 100u + lane;
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
		tokens[lane] = 100u + lane;
		row_lanes[lane] = lane;
		positions[lane] = 63u;
		sequences[lane] = 31u + lane;
	}
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 44u;
	submission.request_id = 44u;
	submission.sequence_id = 44u;
	submission.control_generation = 1u;
	submission.transaction_id = 44u;
	submission.dispatch_generation = 44u;
	submission.request_generation = 1u;
	submission.step_generation = 5u;
	submission.active_sequence_count = 2u;
	submission.new_token_count = 2u;
	submission.lane_count = 2u;
	submission.row_count = 2u;
	submission.token_count = 2u;
	submission.tokens_per_sequence = 1u;
	submission.model_extension_kind = 9u;
	submission.model_extension_bytes = sizeof(extension);
	submission.model_extension = extension;
	submission.lanes = lanes;
	submission.token_ids = tokens;
	submission.row_lane_indices = row_lanes;
	submission.row_positions = positions;
	submission.row_sequence_ids = sequences;
	assert(SparkModelResidentIpcEncodeContinuation(&submission,7u,9u,message,sizeof(message),&message_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcDecodeSubmission(message,message_bytes,&decoded) == SPARK_STATUS_OK);
	assert(SparkModelResidentdBindRoute(&runtime,&route,message,message_bytes,&decoded,0u) == SPARK_STATUS_OK);
	memset(message,0xee,message_bytes);
	assert(SparkModelResidentIpcDecodeSubmission(slot_message,message_bytes,&fresh) == SPARK_STATUS_OK);
	assert(memcmp(&route.submission,&fresh,sizeof(fresh)) == 0);
	assert(route.submission.lanes[1].input_token_id == 101u && route.submission.row_positions[0] == 63u &&
		((const uint8_t *)route.submission.model_extension)[2] == 6u);
	assert(sequence_slots[3].active_owner == 1u && sequence_slots[4].active_owner == 1u);
	pthread_mutex_destroy(&runtime.mutex);
}

static void TestReservedRouteStartsWithAnEmptyCompletion(void)
{
	static SparkModelServingCompletion held,zero;
	static SparkModelResidentdRoute route;
	SparkModelResidentdRuntime runtime;
	SparkModelServingSubmission submission;
	memset(&runtime,0,sizeof(runtime));
	memset(&route,0x3c,sizeof(route));
	memset(&submission,0,sizeof(submission));
	memset(&zero,0,sizeof(zero));
	memset(&route.completion,0,sizeof(route.completion));
	runtime.routes = &route;
	runtime.route_capacity = 1u;
	submission.submission_id = 44u;
	submission.request_id = 45u;
	held.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	held.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	held.token_count = 900u;
	for (uint32_t index=0u; index<900u; index++)
		held.token_ids[index] = index + 1u;
	held.logprob_stride = 2u;
	held.logprob_entry_count = 1800u;
	for (uint32_t index=0u; index<1800u; index++)
		held.logprobs[index].token = index + 7u;
	held.model_extension_bytes = 5u;
	memset(held.model_extension,9,5u);
	held.service_time_ns = 3u;
	SparkModelServingCompletionCopy(&route.completion,&held);
	route.active = 0u;
	assert(SparkModelResidentdReserveRoute(&runtime,&submission,11u) == &route);
	assert(memcmp(&route.completion,&zero,sizeof(zero)) == 0);
	assert(route.active == 1u && route.message_id == 11u && route.submission_id == 44u && route.request_id == 45u &&
		route.resident_slots_claimed == 0u && route.state == SPARK_MODEL_RESIDENTD_ROUTE_IDLE && route.committed_fifo_queued == 0u);
}

static void TestTokenContinuationRebuildsLanesFromSlots(void)
{
	static uint8_t full[8192],compact[8192],slot_message[8192],expanded[8192];
	SparkModelResidentdRuntime runtime;
	SparkModelResidentdRoute route;
	SparkModelResidentdSlot slot;
	SparkModelResidentdSequenceSlot sequence_slots[8];
	SparkModelServingLane continued[8];
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingSubmission submission,decoded;
	SparkModelServingLane lanes[2];
	uint32_t tokens[2],row_lanes[2],full_bytes,compact_bytes;
	uint64_t positions[2],sequences[2];
	memset(&runtime,0,sizeof(runtime));
	memset(&route,0,sizeof(route));
	memset(&slot,0,sizeof(slot));
	memset(sequence_slots,0,sizeof(sequence_slots));
	memset(&submission,0,sizeof(submission));
	memset(lanes,0,sizeof(lanes));
	TestDescriptor(&descriptor);
	assert(pthread_mutex_init(&runtime.mutex,0) == 0);
	runtime.routes = &route;
	runtime.slots = &slot;
	runtime.route_capacity = 1u;
	runtime.route_message_capacity = sizeof(slot_message);
	runtime.sequence_slots = sequence_slots;
	runtime.continued_lanes = continued;
	runtime.expanded_message = expanded;
	runtime.runtime_limits.max_active_sequence_count = 8u;
	runtime.runtime_limits.max_input_row_count = 8u;
	runtime.runtime_limits.resident_sequence_capacity = 8u;
	runtime.adapter_library.adapter_interface.descriptor = &descriptor;
	runtime.client.generation = 9u;
	slot.message = slot_message;
	for (uint32_t lane=0u; lane<2u; lane++)
	{
		lanes[lane].request_id = 21u + lane;
		lanes[lane].request_generation = 1u;
		lanes[lane].step_generation = 77u;
		lanes[lane].sequence_id = 31u + lane;
		lanes[lane].sequence_position = 90u + lane;
		lanes[lane].context_token_count = 91u + lane;
		lanes[lane].state_target_token_count = 91u + lane;
		lanes[lane].resident_sequence_slot = 5u + 2u * lane;
		lanes[lane].input_token_id = 400u + lane;
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
		tokens[lane] = 400u + lane;
		row_lanes[lane] = lane;
		positions[lane] = 90u + lane;
		sequences[lane] = 31u + lane;
		sequence_slots[5u + 2u * lane].bound = 1u;
		sequence_slots[5u + 2u * lane].request_id = lanes[lane].request_id;
		sequence_slots[5u + 2u * lane].request_generation = lanes[lane].request_generation;
		sequence_slots[5u + 2u * lane].sequence_id = lanes[lane].sequence_id;
		sequence_slots[5u + 2u * lane].lane = lanes[lane];
		sequence_slots[5u + 2u * lane].lane.step_generation = 60u;
		sequence_slots[5u + 2u * lane].lane.sequence_position -= 8u;
		sequence_slots[5u + 2u * lane].lane.context_token_count -= 8u;
		sequence_slots[5u + 2u * lane].lane.state_target_token_count -= 8u;
		assert(SparkModelContinuationLeaseEstablish(&sequence_slots[5u + 2u * lane].lease,9u,2u,90u + lane,60u) == SPARK_STATUS_OK);
	}
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 70u;
	submission.request_id = 70u;
	submission.sequence_id = 70u;
	submission.control_generation = 2u;
	submission.transaction_id = 70u;
	submission.dispatch_generation = 70u;
	submission.request_generation = 1u;
	submission.step_generation = 77u;
	submission.tokens_per_sequence = 8u;
	submission.active_sequence_count = 2u;
	submission.new_token_count = 2u;
	submission.lane_count = 2u;
	submission.row_count = 2u;
	submission.token_count = 2u;
	submission.lanes = lanes;
	submission.token_ids = tokens;
	submission.row_lane_indices = row_lanes;
	submission.row_positions = positions;
	submission.row_sequence_ids = sequences;
	assert(SparkModelResidentIpcEncodeContinuation(&submission,13u,9u,full,sizeof(full),&full_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcEncodeTokenContinuation(&submission,13u,9u,compact,sizeof(compact),&compact_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentdProcessTokenContinuation(&runtime,compact,compact_bytes) == SPARK_STATUS_OK);
	assert(route.active == 1u && route.state == SPARK_MODEL_RESIDENTD_ROUTE_CONTINUATION_PREPARING && route.message_id == 13u && route.message_bytes == full_bytes);
	assert(memcmp(slot_message,full,full_bytes) == 0);
	assert(SparkModelResidentIpcDecodeSubmission(full,full_bytes,&decoded) == SPARK_STATUS_OK);
	assert(memcmp(route.submission.lanes,decoded.lanes,2u * sizeof(lanes[0])) == 0 && route.submission.step_generation == 77u &&
		route.submission.row_positions[1] == 91u && route.submission.token_ids[0] == 400u && runtime.client.last_submission_id == 70u);
	assert(SparkModelResidentdDeactivateRouteLocked(&runtime,&route) == SPARK_STATUS_OK);
	submission.submission_id = 71u;
	submission.step_generation = 78u;
	lanes[0].step_generation = lanes[1].step_generation = 78u;
	SparkModelContinuationLeaseInvalidate(&sequence_slots[7].lease);
	assert(SparkModelResidentIpcEncodeTokenContinuation(&submission,14u,9u,compact,sizeof(compact),&compact_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentdProcessTokenContinuation(&runtime,compact,compact_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	assert(route.active == 0u && runtime.client.last_submission_id == 70u);
	assert(SparkModelContinuationLeaseEstablish(&sequence_slots[7].lease,9u,2u,92u,60u) == SPARK_STATUS_OK);
	assert(SparkModelResidentdProcessTokenContinuation(&runtime,compact,compact_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	assert(route.active == 0u);
	pthread_mutex_destroy(&runtime.mutex);
}

int main(void)
{
	TestBoundRouteKeepsTheSingleDecode();
	TestReservedRouteStartsWithAnEmptyCompletion();
	TestTokenContinuationRebuildsLanesFromSlots();
	TestRejectedCompletionFailsRouteWithItsStatus((uint32_t)SPARK_STATUS_MAXIMUM + 1u,0u,SPARK_STATUS_INVALID_ARGUMENT);
	TestFailedCompletionKeepsItsStatus(SPARK_STATUS_NO_LANE);
	TestFailedCompletionKeepsItsStatus(SPARK_STATUS_EVICT_DENIED);
	TestRejectedCompletionFailsRouteWithItsStatus(SPARK_STATUS_OK,1u,SPARK_STATUS_SCHEMA_ERROR);
	TestLaneFatalAdapterStopsTheRank(SPARK_MODEL_DRIVER_DEGRADED_EAGER_PATH,SPARK_STATUS_OK);
	TestLaneFatalAdapterStopsTheRank(SPARK_MODEL_DRIVER_DEGRADED_LANE_FATAL,SPARK_STATUS_INTERNAL_ERROR);
	TestStrayCompletionLeavesRoute();
	TestPersistentSlotRequiresRelease();
	TestStuckRoutesRepeatWithFifoPosition();
	TestTransportDeadlineQuarantinesUntilTerminal(
		SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT);
	TestTransportDeadlineQuarantinesUntilTerminal(
		SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT);
	return(0);
}
