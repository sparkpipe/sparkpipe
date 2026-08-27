#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "sparkpipe/spark_hidden_transport.h"

#define TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY 8u

typedef struct TestModelResidentTransportPending
{
	uint32_t active;
	SparkHiddenTransportPacket packet;
} TestModelResidentTransportPending;

typedef struct TestModelResidentTransport
{
	SparkHiddenTransportEndpoint endpoint;
	uint32_t completion_head;
	uint32_t completion_count;
	SparkHiddenTransportCompletion completions[TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY];
	uint32_t completion_delays[TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY];
	TestModelResidentTransportPending pending[TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY];
} TestModelResidentTransport;

static uint32_t TestModelResidentTransportPacketMatches(
	const SparkHiddenTransportPacket *left,
	const SparkHiddenTransportPacket *right)
{
	return(left != 0 && right != 0 && left->sequence_id == right->sequence_id && left->token_index == right->token_index && left->active_sequence_count == right->active_sequence_count ? 1u : 0u);
}

static TestModelResidentTransportPending *TestModelResidentTransportFind(
	TestModelResidentTransport *state,
	const SparkHiddenTransportPacket *packet)
{
	uint32_t index;
	for (index=0u; index<TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY; index++)
		if ( state->pending[index].active != 0u && TestModelResidentTransportPacketMatches(&state->pending[index].packet,packet) != 0u )
			return(&state->pending[index]);
	return(0);
}

static TestModelResidentTransportPending *TestModelResidentTransportReserve(
	TestModelResidentTransport *state,
	const SparkHiddenTransportPacket *packet)
{
	uint32_t index;
	if ( TestModelResidentTransportFind(state,packet) != 0 )
		return(0);
	for (index=0u; index<TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY; index++)
		if ( state->pending[index].active == 0u )
		{
			state->pending[index].active = 1u;
			state->pending[index].packet = *packet;
			return(&state->pending[index]);
		}
	return(0);
}

static SparkStatus TestModelResidentTransportQueue(
	TestModelResidentTransport *state,
	const SparkHiddenTransportPacket *packet,
	SparkStatus completion_status)
{
	TestModelResidentTransportPending *pending;
	SparkHiddenTransportCompletion *completion;
	uint32_t tail;
	if ( TestModelResidentTransportFind(state,packet) != 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->completion_count >= TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY )
		return(SPARK_STATUS_BUSY);
	pending = TestModelResidentTransportReserve(state,packet);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	tail = (state->completion_head + state->completion_count) % TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY;
	completion = &state->completions[tail];
	memset(completion,0,sizeof(*completion));
	completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
	completion->status = completion_status;
	completion->active_sequence_count = packet->active_sequence_count;
	completion->sequence_id = packet->sequence_id;
	completion->token_index = packet->token_index;
	completion->transfer_bytes = (uint64_t)packet->active_sequence_count * ((uint64_t)packet->bytes_per_sequence + packet->sideband_bytes_per_sequence);
	state->completion_delays[tail] = 1u;
	state->completion_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelResidentTransportInitialize(
	const SparkHiddenTransportEndpoint *endpoint,
	void **transport_state)
{
	TestModelResidentTransport *state;
	if ( endpoint == 0 || transport_state == 0 || endpoint->transport_module_id == 0 || strcmp(endpoint->transport_module_id,SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID) != 0 || endpoint->configuration_flags != SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION || endpoint->source_rank_index == endpoint->sink_rank_index || (endpoint->local_rank_index != endpoint->source_rank_index && endpoint->local_rank_index != endpoint->sink_rank_index) || endpoint->source_host == 0 || endpoint->sink_host == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = (TestModelResidentTransport *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->endpoint = *endpoint;
	*transport_state = state;
	return(SPARK_STATUS_OK);
}

static void TestModelResidentTransportDestroy(void *transport_state)
{
	free(transport_state);
}

static SparkStatus TestModelResidentTransportReceive(
	void *transport_state,
	SparkHiddenTransportPacket *packet)
{
	TestModelResidentTransport *state;
	uint64_t bytes;
	state = (TestModelResidentTransport *)transport_state;
	if ( state == 0 || packet == 0 || packet->hidden_bf16 == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	bytes = (uint64_t)packet->active_sequence_count * packet->bytes_per_sequence;
	memset((void *)packet->hidden_bf16,0x2c,(size_t)bytes);
	bytes = (uint64_t)packet->active_sequence_count * packet->sideband_bytes_per_sequence;
	if ( bytes != 0u )
		memset((void *)packet->sideband_payload,0x5a,(size_t)bytes);
	return(TestModelResidentTransportQueue(state,packet,SPARK_STATUS_OK));
}

static SparkStatus TestModelResidentTransportSend(
	void *transport_state,
	const SparkHiddenTransportPacket *packet)
{
	TestModelResidentTransport *state;
	state = (TestModelResidentTransport *)transport_state;
	if ( state == 0 || packet == 0 || packet->hidden_bf16 == 0 || (packet->sideband_bytes_per_sequence != 0u && (packet->sideband_payload == 0 || ((const uint8_t *)packet->sideband_payload)[0] == 0u)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(TestModelResidentTransportQueue(state,packet,((const uint8_t *)packet->hidden_bf16)[0] != 0u ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED));
}

static SparkStatus TestModelResidentTransportReceiveBatch(
	void *transport_state,
	SparkHiddenTransportPacket *packets,
	uint32_t packet_count)
{
	uint32_t index;
	SparkStatus status;
	status = packet_count != 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
	for (index=0u; status == SPARK_STATUS_OK && index<packet_count; index++)
		status = TestModelResidentTransportReceive(transport_state,&packets[index]);
	return(status);
}

static SparkStatus TestModelResidentTransportSendBatch(
	void *transport_state,
	const SparkHiddenTransportPacket *packets,
	uint32_t packet_count)
{
	uint32_t index;
	SparkStatus status;
	status = packet_count != 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
	for (index=0u; status == SPARK_STATUS_OK && index<packet_count; index++)
		status = TestModelResidentTransportSend(transport_state,&packets[index]);
	return(status);
}

static SparkStatus TestModelResidentTransportPoll(
	void *transport_state,
	SparkHiddenTransportCompletion *completion)
{
	TestModelResidentTransport *state;
	TestModelResidentTransportPending *pending;
	SparkHiddenTransportPacket packet;
	state = (TestModelResidentTransport *)transport_state;
	if ( state == 0 || completion == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->completion_count == 0u )
	{
		memset(completion,0,sizeof(*completion));
		completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
		completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
		completion->status = SPARK_STATUS_BUSY;
		return(SPARK_STATUS_OK);
	}
	if ( state->completion_delays[state->completion_head] != 0u )
	{
		state->completion_delays[state->completion_head]--;
		memset(completion,0,sizeof(*completion));
		completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
		completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
		completion->status = SPARK_STATUS_BUSY;
		return(SPARK_STATUS_OK);
	}
	*completion = state->completions[state->completion_head];
	state->completion_delays[state->completion_head] = 0u;
	state->completion_head = (state->completion_head + 1u) % TEST_MODEL_RESIDENT_TRANSPORT_PENDING_CAPACITY;
	state->completion_count--;
	memset(&packet,0,sizeof(packet));
	packet.sequence_id = completion->sequence_id;
	packet.token_index = completion->token_index;
	packet.active_sequence_count = completion->active_sequence_count;
	pending = TestModelResidentTransportFind(state,&packet);
	if ( pending == 0 )
	{
		completion->status = SPARK_STATUS_SCHEMA_ERROR;
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	memset(pending,0,sizeof(*pending));
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelResidentTransportPollDescriptors(
	void *transport_state,
	SparkHiddenTransportPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out)
{
	(void)descriptors;
	(void)descriptor_capacity;
	if ( transport_state == 0 || descriptor_count_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*descriptor_count_out = 0u;
	return(SPARK_STATUS_OK);
}

static const SparkHiddenTransportInterface TestModelResidentTransportInterface =
{
	.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION,
	.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES,
	.capability_flags = SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS,
	.initialize = TestModelResidentTransportInitialize,
	.destroy = TestModelResidentTransportDestroy,
	.post_receive = TestModelResidentTransportReceive,
	.send = TestModelResidentTransportSend,
	.poll = TestModelResidentTransportPoll,
	.post_receive_batch = TestModelResidentTransportReceiveBatch,
	.send_batch = TestModelResidentTransportSendBatch,
	.get_poll_descriptors = TestModelResidentTransportPollDescriptors
};

__attribute__((visibility("default")))
const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
	return(&TestModelResidentTransportInterface);
}
