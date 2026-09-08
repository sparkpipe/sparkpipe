#include <assert.h>
#include "../ring/transport/tp_device_collective.c"
#include "sparkpipe/spark_tp_chain_ordinal.h"

_Static_assert(SPARK_TP_CHAIN_MAX_GENERATION == (UINT64_MAX >> SPARK_TP_DEVICE_COLLECTIVE_GENERATION_SHIFT),"chain generation must fit transport lifecycle");

static void test_chain_ordinals(void)
{
	uint32_t lanes,operation,chain,credit;
	uint64_t ordinal,generation,last[8],older,newer;
	const uint32_t capacity = (106u * 65536u);
	for (lanes=1u; lanes<=4u; lanes++)
	{
		memset(last,0,sizeof(last));
		for (operation=0u; operation<5824u; operation++)
			for (chain=lanes; chain>0u; chain--)
			{
				assert(SparkTpChainOrdinal(chain,lanes,2u,capacity,operation,&ordinal) == SPARK_STATUS_OK);
				credit = (uint32_t)(ordinal % (lanes * 2u));
				generation = ((ordinal / (lanes * 2u)) + 1u);
				assert((credit / 2u) == (chain % lanes));
				assert(generation > last[credit]);
				last[credit] = generation;
			}
		for (chain=1u; chain<=lanes; chain++)
		{
			assert(SparkTpChainOrdinal(chain,lanes,2u,capacity,capacity - 1u,&older) == SPARK_STATUS_OK);
			assert(SparkTpChainOrdinal(chain + lanes,lanes,2u,capacity,0u,&newer) == SPARK_STATUS_OK);
			assert((newer / (lanes * 2u)) > (older / (lanes * 2u)));
		}
	}
	assert(SparkTpChainOrdinal(1u,4u,2u,capacity,capacity,&ordinal) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkTpChainOrdinal(UINT64_MAX,4u,2u,capacity,0u,&ordinal) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkTpChainOrdinal(0u,4u,2u,capacity,0u,&ordinal) == SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkTpDeviceCollectiveImplementation IMPLEMENTATION;
static SparkTpDeviceCollective COLLECTIVE;
static uint64_t TREE_ACKS[SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
static uint64_t D2A_ACKS[D2A_ROUTE_COUNT * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];

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
	test_chain_ordinals();
	IMPLEMENTATION.collective = &COLLECTIVE;
	IMPLEMENTATION.ack_receive_slots = TREE_ACKS;
	IMPLEMENTATION.d2a_ack_receive_slots = D2A_ACKS;
	for (index=0u; index<(sizeof(counts) / sizeof(counts[0])); index++)
	{
		test_configuration(counts[index]);
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
	puts("TP chain ownership, prefill ordinal capacity and ACK flow PASS");
	return(0);
}
