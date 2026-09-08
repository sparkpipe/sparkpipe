#include <assert.h>
#include <sparkpipe/spark_hidden_transport.h>
static SparkStatus test_fixed_send(SparkHiddenTransportSession *session,const void *buffer,uint64_t bytes,uint64_t offset,uint32_t sequence);
#define SparkHiddenTransportSendFixed test_fixed_send
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

static void test_pending_completion(void *context,const SparkTpDeviceCollectiveCompletion *completion)
{
	uint32_t *result = (uint32_t *)context;
	result[0]++;
	result[1] = completion->status;
	assert(completion->ordinal == 8u && completion->slot_index == 1u);
}

static void test_pending(uint32_t failure)
{
	SparkTpDeviceCollectiveSubmission submission;
	cudaStream_t stream;
	uint16_t payload[4] = {0};
	uint32_t result[2] = {0},index;
	memset(&IMPLEMENTATION,0,sizeof(IMPLEMENTATION));
	memset(&COLLECTIVE,0,sizeof(COLLECTIVE));
	memset(&submission,0,sizeof(submission));
	COLLECTIVE.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	COLLECTIVE.implementation = &IMPLEMENTATION;
	COLLECTIVE.credit_count = 8u;
	COLLECTIVE.max_active_sequence_count = 1u;
	COLLECTIVE.operation_timeout_milli = 1000u;
	IMPLEMENTATION.collective = &COLLECTIVE;
	IMPLEMENTATION.combine_bf16_function = test_combine;
	atomic_init(&IMPLEMENTATION.admission_open,1u);
	atomic_init(&IMPLEMENTATION.failure_status,SPARK_STATUS_OK);
	for (index=0u; index<8u; index++)
	{
		atomic_init(&IMPLEMENTATION.pending[index].state,0u);
		atomic_init(&IMPLEMENTATION.operations[index].lifecycle,SparkTpDeviceCollectiveStateWord(1u,index == 0u ? SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE : SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE,0u));
	}
	assert(cudaStreamCreate(&stream) == cudaSuccess);
	assert(cudaEventCreateWithFlags(&IMPLEMENTATION.consumer_events[0],cudaEventDisableTiming) == cudaSuccess);
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 1u;
	submission.ordinal = 8u;
	submission.active_sequence_count = 1u;
	submission.local_device = submission.full_device = payload;
	submission.cuda_stream = stream;
	submission.completion_function = test_pending_completion;
	submission.completion_context = result;
	atomic_store(&IMPLEMENTATION.admission_open,0u);
	assert(SparkTpDeviceCollectiveEnqueue(&COLLECTIVE,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) == SPARK_STATUS_IO_ERROR);
	assert(atomic_load(&IMPLEMENTATION.pending[1].state) == 0u && result[0] == 0u);
	atomic_store(&IMPLEMENTATION.admission_open,1u);
	assert(SparkTpDeviceCollectiveEnqueue(&COLLECTIVE,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveEnqueue(&COLLECTIVE,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) == SPARK_STATUS_BUSY);
	assert(SparkTpDeviceCollectiveRequestOperationFailure(&COLLECTIVE,8u,(SparkStatus)(SPARK_STATUS_UNSUPPORTED + 1u)) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkTpDeviceCollectiveRequestFailure(&COLLECTIVE,(SparkStatus)-1) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(IMPLEMENTATION.pending[1].failure_status == SPARK_STATUS_OK);
	assert(SparkTpProgressPending(&IMPLEMENTATION) == 1u && result[0] == 0u);
	if ( failure == 1u )
		IMPLEMENTATION.pending[1].deadline_milli = 0u;
	else if ( failure == 3u )
		assert(SparkTpDeviceCollectiveRequestOperationFailure(&COLLECTIVE,8u,SPARK_STATUS_IO_ERROR) == SPARK_STATUS_OK);
	else if ( failure == 2u )
		assert(SparkTpDeviceCollectiveRequestFailure(&COLLECTIVE,SPARK_STATUS_IO_ERROR) == SPARK_STATUS_OK);
	else
		atomic_store(&IMPLEMENTATION.operations[0].lifecycle,SparkTpDeviceCollectiveStateWord(1u,SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE,0u));
	assert(SparkTpProgressPending(&IMPLEMENTATION) == 0u);
	if ( failure == 0u )
	{
		assert(SparkTpDeviceCollectiveStatePhase(atomic_load(&IMPLEMENTATION.operations[0].lifecycle)) == SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
		atomic_store(&IMPLEMENTATION.operations[0].lifecycle,SparkTpDeviceCollectiveStateWord(2u,SPARK_TP_DEVICE_COLLECTIVE_PHASE_TERMINAL_READY,0u));
		SparkTpDeviceCollectivePublishCompletion(&IMPLEMENTATION,&IMPLEMENTATION.operations[0]);
	}
	assert(result[0] == 1u && result[1] == (failure != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK));
	assert(SparkTpProgressPending(&IMPLEMENTATION) == 0u && result[0] == 1u);
	assert(cudaEventDestroy(IMPLEMENTATION.consumer_events[0]) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
}

static uint32_t test_tree_step(uint32_t degree,uint32_t rank,uint32_t *stages,uint32_t *values,uint32_t messages[16][16])
{
	uint32_t stage = stages[rank],recv,send,bit,peer;
	if ( stage == TREE_STAGES )
		return(0u);
	recv = tree_recv_mask(rank,stage,degree);
	for (bit=0u; bit<7u; bit++)
		if ( (recv & (1u << bit)) != 0u && messages[tree_peer(rank,bit)][rank] == 0u )
			return(0u);
	for (bit=0u; bit<7u; bit++)
	{
		if ( (recv & (1u << bit)) == 0u )
			continue;
		peer = tree_peer(rank,bit);
		if ( stage == 3u )
			values[rank] = messages[peer][rank];
		else
		{
			assert((values[rank] & messages[peer][rank]) == 0u);
			values[rank] |= messages[peer][rank];
		}
		messages[peer][rank] = 0u;
	}
	send = stage == 0u ? tree_send_mask(rank,0u,degree) | tree_send_mask(rank,1u,degree) : (stage < 3u ? tree_send_mask(rank,stage + 1u,degree) : 0u);
	for (bit=0u; bit<7u; bit++)
		if ( (send & (1u << bit)) != 0u )
		{
			peer = tree_peer(rank,bit);
			assert(peer < degree && peer != rank && messages[rank][peer] == 0u);
			messages[rank][peer] = values[rank];
		}
	stages[rank]++;
	return(1u);
}

static void test_tree_topology(uint32_t degree)
{
	uint32_t stages[16] = {0},values[16] = {0},messages[16][16] = {{0}};
	uint32_t round,rank;
	for (rank=0u; rank<degree; rank++)
		values[rank] = (1u << rank);
	for (round=0u; round<16u; round++)
		for (rank=degree; rank>0u; rank--)
			(void)test_tree_step(degree,rank - 1u,stages,values,messages);
	for (rank=0u; rank<degree; rank++)
		assert(stages[rank] == TREE_STAGES && values[rank] == (1u << degree) - 1u);
}

static void test_configuration(uint32_t credits,uint32_t degree)
{
	SparkTpDeviceCollectiveConfig config;
	SparkTpDeviceCollectiveCreditBinding bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t route,credit,index,count;
	memset(&config,0,sizeof(config));
	memset(bindings,0,sizeof(bindings));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.tp_degree = degree;
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
	for (route=0u; route<tree_route_count(0u,degree); route++)
		for (credit=0u; credit<credits; credit++)
		{
			index = ((route * credits) + credit);
			bindings[index].step_index = route;
			bindings[index].credit_index = credit;
			bindings[index].send_device = bindings[index].send_transport = &config;
			bindings[index].receive_device = bindings[index].receive_transport = &config;
		}
	config.credit_bindings = bindings;
	config.credit_binding_count = (tree_route_count(0u,degree) * credits);
	assert(SparkTpDeviceCollectiveValidateConfig(&config,&count) == SPARK_STATUS_OK);
	assert(count == credits);
	config.algorithm_mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL;
	config.direct_all_to_all_max_payload_bytes = 65536u;
	assert(SparkTpDeviceCollectiveCreditBindingRouteCount(&config,&count) == SPARK_STATUS_OK);
	assert(count == tree_route_count(0u,degree) + degree - 1u);
	config.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE;
	config.direct_all_to_all_max_payload_bytes = 0u;
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
	test_pending(0u);
	test_pending(1u);
	test_pending(2u);
	test_pending(3u);
	test_tree_topology(4u);
	test_tree_topology(16u);
	IMPLEMENTATION.collective = &COLLECTIVE;
	IMPLEMENTATION.ack_receive_slots = TREE_ACKS;
	IMPLEMENTATION.d2a_ack_receive_slots = D2A_ACKS;
	for (index=0u; index<(sizeof(counts) / sizeof(counts[0])); index++)
	{
		test_configuration(counts[index],4u);
		test_configuration(counts[index],16u);
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
	puts("TP chain ownership, prefill ordinal capacity and ACK flow PASS");
	return(0);
}
