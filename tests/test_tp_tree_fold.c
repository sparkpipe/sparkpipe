#include "ring/transport/tp_device_collective.c"

static SparkTpDeviceCollectiveImplementation test_implementation;
static uint32_t test_calls;

static SparkStatus test_combine(void *context,void *destination,const void *const ranks[SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT],uint32_t local,uint32_t rows,uint32_t width,void *stream)
{
	uint32_t rank;
	float value = 0.0f;
	(void)context;
	(void)rows;
	(void)width;
	(void)stream;
	if ( ranks[local] != destination )
		return(SPARK_STATUS_INTERNAL_ERROR);
	for (rank=0u; rank<16u; rank++)
	{
		if ( (rank % 4u == 0u) != (ranks[rank] != 0) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( ranks[rank] != 0 )
			value += *(const float *)ranks[rank];
	}
	*(float *)destination = value;
	test_calls++;
	return(SPARK_STATUS_OK);
}

static int32_t test_order(uint32_t root,const uint32_t order[3])
{
	SparkTpDeviceCollective collective = {0};
	SparkTpDeviceCollectiveOperation operation = {0};
	uint64_t payload[7][2] = {{0}};
	float values[4] = {256.0f,1.0f,-256.0f,0.0f},destination = values[root / 4u];
	uint32_t used = tree_used(root,16u),bit,route,index;
	collective.tp_rank = root;
	collective.tp_degree = 16u;
	test_implementation.collective = &collective;
	test_implementation.combine_all_bf16_function = test_combine;
	operation.full_device = &destination;
	operation.ordinal = 9u;
	test_calls = 0u;
	for (bit=4u; bit<7u; bit++)
	{
		route = tree_bit_route(used,bit);
		memcpy(&payload[route][0],&values[tree_peer(root,bit) / 4u],sizeof(float));
		test_implementation.bindings[route][0].receive_device = payload[route];
		test_implementation.bindings[route][0].receive_transport = payload[route];
	}
	for (index=0u; index<3u; index++)
	{
		payload[tree_bit_route(used,order[index])][1] = 10u;
		if ( SparkTpDeviceCollectiveTreeFoldGroups(&test_implementation,&operation,used,0x70u,8u) != (index == 2u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY) )
			return(-1);
		if ( test_calls != (index == 2u ? 1u : 0u) )
			return(-2);
	}
	if ( destination != 1.0f )
		return(-3);
	operation.packed = 1u;
	if ( SparkTpDeviceCollectiveTreeFoldGroups(&test_implementation,&operation,used,0x70u,8u) != SPARK_STATUS_OK || test_calls != 1u )
		return(-4);
	return(0);
}

int main(void)
{
	uint32_t orders[6][3] = {{4u,5u,6u},{4u,6u,5u},{5u,4u,6u},{5u,6u,4u},{6u,4u,5u},{6u,5u,4u}},root,index;
	for (root=0u; root<16u; root+=4u)
		for (index=0u; index<6u; index++)
			if ( test_order(root,orders[index]) != 0 )
				return(1);
	puts("PASS all tree subgroup roots and arrival permutations, no premature or repeated fold");
	return(0);
}
