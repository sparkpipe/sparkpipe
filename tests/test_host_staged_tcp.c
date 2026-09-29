#include "sparkpipe/spark_hidden_transport.h"
#include <cuda_runtime_api.h>
#include <assert.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void);

static cudaStream_t test_stream;

static SparkHiddenTransportEndpoint endpoint_for(uint32_t local,uint32_t source,uint32_t sink,uint32_t port_base,uint32_t sideband)
{
	SparkHiddenTransportEndpoint endpoint;
	memset(&endpoint,0,sizeof(endpoint));
	endpoint.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	endpoint.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
	endpoint.capability_flags = SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS | SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS;
	endpoint.hidden_dimension = 64u;
	endpoint.bytes_per_sequence = 128u;
	endpoint.max_active_sequence_count = 8u;
	endpoint.configuration_flags = SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION;
	endpoint.local_rank_index = local;
	endpoint.source_rank_index = source;
	endpoint.sink_rank_index = sink;
	endpoint.control_port_base = port_base;
	endpoint.max_packet_bytes = (uint64_t)(128u + sideband) * 8u;
	endpoint.transport_module_id = SPARK_HIDDEN_TRANSPORT_HOST_STAGED_TCP_MODULE_ID;
	endpoint.route_name = "stage0-to-stage1";
	endpoint.source_host = "127.0.0.1";
	endpoint.sink_host = "127.0.0.1";
	return(endpoint);
}

static SparkHiddenTransportPacket packet_for(void *hidden,void *sideband,uint32_t sideband_bytes,uint32_t rows,uint64_t sequence,uint64_t token)
{
	SparkHiddenTransportPacket packet;
	memset(&packet,0,sizeof(packet));
	packet.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet.flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet.active_sequence_count = rows;
	packet.hidden_dimension = 64u;
	packet.bytes_per_sequence = 128u;
	packet.sequence_id = sequence;
	packet.token_index = token;
	packet.hidden_bf16 = hidden;
	packet.cuda_stream = (void *)test_stream;
	if ( sideband_bytes != 0u )
	{
		packet.flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
		packet.sideband_payload = sideband;
		packet.sideband_kind = 1u;
		packet.sideband_bytes_per_sequence = sideband_bytes;
	}
	return(packet);
}

static SparkHiddenTransportCompletion wait_completion(SparkHiddenTransportSession *session)
{
	SparkHiddenTransportPollDescriptor descriptors[4];
	SparkHiddenTransportCompletion completion;
	uint32_t count = 0u,attempt;
	for (attempt = 0u; attempt < 2000u; attempt++)
	{
		assert(SparkHiddenTransportPoll(session,&completion) == SPARK_STATUS_OK);
		if ( completion.status != SPARK_STATUS_BUSY )
			return(completion);
		assert(SparkHiddenTransportGetPollDescriptors(session,descriptors,4u,&count) == SPARK_STATUS_OK && count == 1u);
		{
			struct pollfd fd = {descriptors[0].fd,POLLIN,0};
			(void)poll(&fd,1u,5);
		}
	}
	assert(!"transport completion timed out");
	return(completion);
}

static void run(uint32_t port_base,uint32_t sideband_bytes)
{
	const SparkHiddenTransportInterface *module = SparkHiddenTransportGetInterface();
	SparkHiddenTransportEndpoint input = endpoint_for(1u,0u,1u,port_base,sideband_bytes);
	SparkHiddenTransportEndpoint output = endpoint_for(0u,0u,1u,port_base,sideband_bytes);
	SparkHiddenTransportSession *receiver = 0,*sender = 0;
	SparkHiddenTransportCompletion completion;
	uint8_t source[8u * 128u],destination[8u * 128u],source_side[8u * 16u],destination_side[8u * 16u];
	uint32_t index,round;
	assert(SparkHiddenTransportOpen(&input,module,SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS | SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS,&receiver) == SPARK_STATUS_OK);
	assert(SparkHiddenTransportOpen(&output,module,SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS | SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS,&sender) == SPARK_STATUS_OK);
	for (round = 0u; round < 24u; round++)
	{
		uint32_t rows = 1u + round % 8u;
		SparkHiddenTransportPacket out,in;
		for (index = 0u; index < sizeof(source); index++)
			source[index] = (uint8_t)(index * 7u + round);
		for (index = 0u; index < sizeof(source_side); index++)
			source_side[index] = (uint8_t)(index * 3u + round);
		memset(destination,0,sizeof(destination));
		memset(destination_side,0,sizeof(destination_side));
		out = packet_for(source,source_side,sideband_bytes,rows,100u + round,7u * round);
		in = packet_for(destination,destination_side,sideband_bytes,rows,100u + round,7u * round);
		if ( (round & 1u) == 0u )
			assert(SparkHiddenTransportPostReceive(receiver,&in) == SPARK_STATUS_OK);
		assert(SparkHiddenTransportSend(sender,&out) == SPARK_STATUS_OK);
		completion = wait_completion(sender);
		assert(completion.status == SPARK_STATUS_OK && completion.sequence_id == 100u + round && completion.token_index == 7u * round);
		assert(completion.transfer_bytes == (uint64_t)rows * (128u + sideband_bytes));
		if ( (round & 1u) != 0u )
		{
			struct timespec pause = {0,2000000};
			(void)nanosleep(&pause,0);
			assert(SparkHiddenTransportPoll(receiver,&completion) == SPARK_STATUS_OK && completion.status == SPARK_STATUS_BUSY);
			assert(SparkHiddenTransportPostReceive(receiver,&in) == SPARK_STATUS_OK);
		}
		completion = wait_completion(receiver);
		assert(completion.status == SPARK_STATUS_OK && completion.sequence_id == 100u + round && completion.token_index == 7u * round && completion.active_sequence_count == rows);
		assert(completion.transfer_bytes == (uint64_t)rows * (128u + sideband_bytes));
		assert(memcmp(source,destination,(size_t)rows * 128u) == 0);
		assert(sideband_bytes == 0u || memcmp(source_side,destination_side,(size_t)rows * sideband_bytes) == 0);
	}
	{
		SparkHiddenTransportPacket out = packet_for(source,source_side,sideband_bytes,2u,900u,1u);
		SparkHiddenTransportPacket in = packet_for(destination,destination_side,sideband_bytes,3u,900u,1u);
		assert(SparkHiddenTransportPostReceive(receiver,&in) == SPARK_STATUS_OK);
		assert(SparkHiddenTransportSend(sender,&out) == SPARK_STATUS_OK);
		(void)wait_completion(sender);
		completion = wait_completion(receiver);
		assert(completion.status == SPARK_STATUS_VALIDATION_FAILED);
	}
	SparkHiddenTransportClose(sender);
	SparkHiddenTransportClose(receiver);
}

int main(void)
{
	uint32_t base = 20000u + (uint32_t)(getpid() % 2000) * 16u;
	assert(cudaStreamCreate(&test_stream) == cudaSuccess && test_stream != 0);
	run(base,0u);
	run(base + 64u,16u);
	printf("test_host_staged_tcp: hidden and sideband payloads arrive intact in both post orders\n");
	return(0);
}
