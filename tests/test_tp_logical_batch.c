#include "ring/transport/tp_device_collective.c"

static SparkTpDeviceCollectiveImplementation test_implementation;

static SparkStatus test_combine(void *context,void *destination,const void *source,uint32_t rows,uint32_t width,void *stream)
{
	(void)context;
	(void)destination;
	(void)source;
	(void)rows;
	(void)width;
	(void)stream;
	return(SPARK_STATUS_OK);
}

static void test_complete(void *context,const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context;
	(void)completion;
}

static int32_t test_submit(uint32_t logical,uint32_t rows,uint32_t direct)
{
	SparkTpDeviceCollective collective = {0};
	SparkTpDeviceCollectiveSubmission submission = {0};
	uint64_t values[8] = {0},send[8] = {0};
	SparkStatus status;
	int32_t result = 0;
	memset(&test_implementation,0,sizeof(test_implementation));
	collective.implementation = &test_implementation;
	collective.credit_count = 1u;
	collective.max_active_sequence_count = 8u;
	collective.local_hidden_dimension = 4u;
	collective.direct_all_to_all_max_payload_bytes = 64u;
	test_implementation.collective = &collective;
	test_implementation.combine_bf16_function = test_combine;
	test_implementation.d2a_route_count = 15u;
	test_implementation.d2a_bindings[0][0].send_device = send;
	test_implementation.d2a_bindings[0][0].send_transport = send;
	atomic_init(&test_implementation.admission_open,1u);
	if ( cudaEventCreate(&test_implementation.consumer_events[0]) != cudaSuccess )
		return(-1);
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.active_sequence_count = rows;
	submission.logical_sequence_count = logical;
	submission.local_device = values;
	submission.full_device = values;
	submission.cuda_stream = values;
	submission.completion_function = test_complete;
	status = SparkTpDeviceCollectiveSubmitHiddenInner(&collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
	if ( logical == 0u && status != SPARK_STATUS_INVALID_ARGUMENT )
		result = -2;
	if ( logical != 0u && (status != SPARK_STATUS_OK || test_implementation.operations[0].direct_all_to_all != direct || SparkTpDeviceCollectiveStatePhase(atomic_load(&test_implementation.operations[0].lifecycle)) != SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE) )
		result = -3;
	cudaEventDestroy(test_implementation.consumer_events[0]);
	return(result);
}

int main(void)
{
	if ( test_submit(0u,1u,0u) != 0 || test_submit(1u,1u,1u) != 0 || test_submit(1u,3u,1u) != 0 || test_submit(3u,1u,0u) != 0 || test_submit(3u,3u,0u) != 0 || test_submit(100u,1u,0u) != 0 )
		return(1);
	puts("PASS logical batch controls algorithm across split rows; missing metadata rejected");
	return(0);
}
