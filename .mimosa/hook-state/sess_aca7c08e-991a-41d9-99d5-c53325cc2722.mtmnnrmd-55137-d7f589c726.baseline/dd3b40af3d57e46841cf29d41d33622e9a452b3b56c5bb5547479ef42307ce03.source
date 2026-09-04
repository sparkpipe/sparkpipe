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
	descriptor.resident_sequence_slot_reuse =
		SPARK_MODEL_SERVING_SLOT_REUSE_NONE;
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

int main(void)
{
	TestTransportDeadlineQuarantinesUntilTerminal(
		SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT);
	TestTransportDeadlineQuarantinesUntilTerminal(
		SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT);
	return(0);
}
