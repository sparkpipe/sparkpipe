#include <assert.h>
#include <sparkpipe/spark_hidden_transport.h>
static SparkStatus test_fixed_send(SparkHiddenTransportSession *session,const void *buffer,uint64_t bytes,uint64_t offset,uint32_t sequence);
#define SparkHiddenTransportSendFixed test_fixed_send
#include "../ring/transport/tp_device_collective.c"

static SparkTpDeviceCollectiveImplementation IMPLEMENTATION;
static SparkTpDeviceCollective COLLECTIVE;
static uint64_t TREE_ACKS[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
static uint64_t D2A_ACKS[D2A_ROUTE_COUNT * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];

static uint64_t SEND_OFFSET,SEND_BYTES;

static SparkStatus test_fixed_send(SparkHiddenTransportSession *session,const void *buffer,uint64_t bytes,uint64_t offset,uint32_t sequence)
{
	(void)session;
	(void)buffer;
	(void)sequence;
	SEND_OFFSET = offset;
	SEND_BYTES = bytes;
	return(SPARK_STATUS_OK);
}

static void test_fixed_offset(uint32_t credit)
{
	SparkTpDeviceCollectiveOperation operation = {0};
	uint64_t payload[8] = {0};
	COLLECTIVE.max_active_sequence_count = 97u;
	COLLECTIVE.local_hidden_dimension = 4u;
	operation.credit_index = credit;
	operation.active_sequence_count = 7u;
	operation.ordinal = (1ull << 40u) + credit;
	IMPLEMENTATION.bindings[0][credit].send_transport = payload;
	SparkTpDeviceCollectiveTreeSend(&IMPLEMENTATION,&operation,0u);
	assert(SEND_OFFSET == (uint64_t)credit * 784u && SEND_BYTES == 64u);
	assert(payload[7] == operation.ordinal + 1u);
	operation.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64;
	operation.active_sequence_count = 3u;
	SparkTpDeviceCollectiveTreeSend(&IMPLEMENTATION,&operation,0u);
	assert(SEND_OFFSET == (uint64_t)credit * 784u && SEND_BYTES == 32u);
	assert(SparkTpDeviceCollectivePostAck(0,payload,operation.ordinal,0u,credit) == SPARK_STATUS_OK);
	assert(SEND_OFFSET == (uint64_t)credit * 8u && SEND_BYTES == 8u);
}

static SparkStatus test_combine(void *context,void *destination,const void *source,uint32_t rows,uint32_t hidden,void *stream)
{
	(void)context;
	(void)destination;
	(void)source;
	(void)rows;
	(void)hidden;
	(void)stream;
	return(SPARK_STATUS_OK);
}

static void test_configuration(uint32_t credits)
{
	SparkTpDeviceCollectiveConfig config;
	SparkTpDeviceCollectiveCreditBinding bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t route,credit,index,count;
	memset(&config,0,sizeof(config));
	memset(bindings,0,sizeof(bindings));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.tp_degree = 16u;
	config.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	config.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE;
	config.credit_count = credits;
	config.local_hidden_dimension = 4096u;
	config.max_active_sequence_count = 97u;
	config.connect_timeout_milli = 1000u;
	config.operation_timeout_milli = 1000u;
	config.collective_identifier = 1u;
	config.backend_module_path = "test";
	config.local_host = "rank0";
	config.registration_cuda_stream = &config;
	config.combine_bf16_function = test_combine;
	for (index=0u; index<16u; index++)
		config.rank_hosts[index] = "test";
	for (route=0u; route<tree_route_count(0u); route++)
		for (credit=0u; credit<credits; credit++)
		{
			index = ((route * credits) + credit);
			bindings[index].step_index = route;
			bindings[index].credit_index = credit;
			bindings[index].send_device = bindings[index].send_transport = &config;
			bindings[index].receive_device = bindings[index].receive_transport = &config;
		}
	config.credit_bindings = bindings;
	config.credit_binding_count = (tree_route_count(0u) * credits);
	assert(SparkTpDeviceCollectiveValidateConfig(&config,&count) == SPARK_STATUS_OK);
	assert(count == credits);
	config.credit_count = (SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT + 1u);
	assert(SparkTpDeviceCollectiveValidateConfig(&config,&count) == SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void test_credit(uint32_t direct,uint32_t route,uint32_t credit)
{
	uint64_t *posted,*ack;
	posted = direct != 0u ? &IMPLEMENTATION.d2a_last_posted[route][credit] : &IMPLEMENTATION.tree_last_posted[route][credit];
	ack = (direct != 0u ? D2A_ACKS : TREE_ACKS) + (route * COLLECTIVE.credit_count) + credit;
	assert(SparkTpDeviceCollectiveAckGateOpen(&IMPLEMENTATION,direct,route,credit) == 1u);
	*posted = 4096u;
	assert(SparkTpDeviceCollectiveAckGateOpen(&IMPLEMENTATION,direct,route,credit) == 0u);
	*ack = 4096u;
	assert(SparkTpDeviceCollectiveAckGateOpen(&IMPLEMENTATION,direct,route,credit) == 0u);
	*ack = 4097u;
	assert(SparkTpDeviceCollectiveAckGateOpen(&IMPLEMENTATION,direct,route,credit) == 1u);
	*posted = 1048576u;
	assert(SparkTpDeviceCollectiveAckGateOpen(&IMPLEMENTATION,direct,route,credit) == 0u);
	*ack = 1048577u;
	assert(SparkTpDeviceCollectiveAckGateOpen(&IMPLEMENTATION,direct,route,credit) == 1u);
}

int main(void)
{
	uint32_t counts[] = {1u,3u,4u,8u,17u,64u};
	uint32_t index,route,credit;
	IMPLEMENTATION.collective = &COLLECTIVE;
	IMPLEMENTATION.ack_receive_slots = TREE_ACKS;
	IMPLEMENTATION.d2a_ack_receive_slots = D2A_ACKS;
	for (index=0u; index<(sizeof(counts) / sizeof(counts[0])); index++)
	{
		test_configuration(counts[index]);
		test_fixed_offset(counts[index] - 1u);
		COLLECTIVE.credit_count = counts[index];
		memset(TREE_ACKS,0,sizeof(TREE_ACKS));
		memset(D2A_ACKS,0,sizeof(D2A_ACKS));
		memset(IMPLEMENTATION.tree_last_posted,0xff,sizeof(IMPLEMENTATION.tree_last_posted));
		memset(IMPLEMENTATION.d2a_last_posted,0xff,sizeof(IMPLEMENTATION.d2a_last_posted));
		for (route=0u; route<SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS; route++)
			for (credit=0u; credit<counts[index]; credit++)
				test_credit(0u,route,credit);
		for (route=0u; route<D2A_ROUTE_COUNT; route++)
			for (credit=0u; credit<counts[index]; credit++)
				test_credit(1u,route,credit);
	}
	puts("TP credit ACK jumps and independent routes PASS");
	return(0);
}
