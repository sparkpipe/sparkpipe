#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sparkpipe/spark_tp_device_collective.h"

#ifndef SPARK_TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE_PATH
#define SPARK_TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE_PATH \
	"build/test_modules/libtp_device_collective_nccl_module.so"
#endif

#define TEST_TP_DEGREE 4u

typedef struct TestNcclCompletion
{
	uint32_t count;
	SparkTpDeviceCollectiveCompletion completion;
} TestNcclCompletion;

void SparkTpDeviceCollectiveDebugSubmissionClaimed(uint32_t credit_index,
	uint64_t generation)
{
	(void)credit_index;
	(void)generation;
}

static void TestNcclComplete(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	TestNcclCompletion *state;
	state = (TestNcclCompletion *)context;
	assert(state != 0 && completion != 0);
	state->completion = *completion;
	state->count++;
}

static uint16_t TestNcclUnusedPort(void)
{
	struct sockaddr_in address;
	socklen_t address_bytes;
	int32_t socket_descriptor;
	socket_descriptor = socket(AF_INET,SOCK_STREAM,0);
	assert(socket_descriptor >= 0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	assert(inet_pton(AF_INET,"127.0.0.1",&address.sin_addr) == 1);
	address.sin_port = 0u;
	assert(bind(socket_descriptor,(struct sockaddr *)&address,
		sizeof(address)) == 0);
	address_bytes = sizeof(address);
	assert(getsockname(socket_descriptor,(struct sockaddr *)&address,
		&address_bytes) == 0);
	assert(close(socket_descriptor) == 0);
	return(ntohs(address.sin_port));
}

static void TestNcclConfigure(SparkTpDeviceCollectiveConfig *configuration,
	uint32_t rank,uint16_t port)
{
	uint32_t index;
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration->backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	configuration->tp_degree = TEST_TP_DEGREE;
	configuration->tp_rank = rank;
	configuration->operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration->credit_count = 2u;
	configuration->local_hidden_dimension = 4u;
	configuration->max_active_sequence_count = 2u;
	configuration->connect_timeout_milli = 4000u;
	configuration->operation_timeout_milli = 4000u;
	configuration->control_port_base = port;
	configuration->collective_identifier = UINT64_C(0x123456789abcdef0);
	configuration->backend_module_path =
		SPARK_TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE_PATH;
	configuration->local_host = "127.0.0.1";
	configuration->registration_cuda_stream = (void *)(uintptr_t)1u;
	for (index=0u; index<TEST_TP_DEGREE; index++)
		configuration->rank_hosts[index] = "127.0.0.1";
}

static void TestNcclRank(uint32_t rank,uint16_t port)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveSubmission submission;
	TestNcclCompletion completion;
	uint64_t maxloc[2],reduced[2];
	uint16_t values[8];
	uint32_t index;
	TestNcclConfigure(&configuration,rank,port);
	assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
		SPARK_STATUS_OK);
	assert(collective.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL);
	assert(collective.memory_mode ==
		SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE);
	for (index=0u; index<8u; index++)
		values[index] = (uint16_t)(rank + index);
	memset(&completion,0,sizeof(completion));
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = rank;
	submission.active_sequence_count = 2u;
	submission.ordinal = 0u;
	submission.local_device = values;
	submission.full_device = values;
	submission.cuda_stream = (void *)(uintptr_t)1u;
	submission.completion_function = TestNcclComplete;
	submission.completion_context = &completion;
	assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
		SPARK_STATUS_OK);
	assert(completion.count == 1u && completion.completion.ordinal == 0u);
	submission.ordinal = 3u;
	assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
		SPARK_STATUS_VALIDATION_FAILED);
	submission.ordinal = 1u;
	submission.active_sequence_count = 1u;
	maxloc[0] = UINT64_C(0xbf800000ffffffd5);
	maxloc[1] = UINT64_C(0x3f800000ffffffea);
	reduced[0] = 0u;
	reduced[1] = 0u;
	submission.local_device = maxloc;
	submission.full_device = reduced;
	assert(SparkTpDeviceCollectiveSubmitU64Max(&collective,&submission) ==
		SPARK_STATUS_OK);
	assert(completion.count == 2u && completion.completion.ordinal == 1u);
	assert(reduced[0] == maxloc[0] && reduced[1] == 0u);
	submission.ordinal = 2u;
	submission.active_sequence_count = 2u;
	submission.local_device = values;
	submission.full_device = values;
	assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
		SPARK_STATUS_OK);
	assert(completion.count == 3u && completion.completion.ordinal == 2u);
	SparkTpDeviceCollectiveDestroy(&collective);
}

int main(void)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollective collective;
	pid_t children[TEST_TP_DEGREE];
	uint32_t memory_mode,rank,step_count;
	uint16_t port;
	int32_t status;
	assert(SparkTpDeviceCollectiveCreditStepCount(
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL,TEST_TP_DEGREE,
		&step_count) == SPARK_STATUS_OK && step_count == 0u);
	assert(SparkTpDeviceCollectiveProbeMemoryMode(
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL,
		SPARK_TEST_TP_DEVICE_COLLECTIVE_NCCL_MODULE_PATH,&memory_mode) ==
		SPARK_STATUS_OK && memory_mode ==
		SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE);
	port = TestNcclUnusedPort();
	for (rank=0u; rank<TEST_TP_DEGREE; rank++)
	{
		children[rank] = fork();
		assert(children[rank] >= 0);
		if ( children[rank] == 0 )
		{
			TestNcclRank(rank,port);
			_exit(0);
		}
	}
	for (rank=0u; rank<TEST_TP_DEGREE; rank++)
	{
		assert(waitpid(children[rank],&status,0) == children[rank]);
		assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
	}
	TestNcclConfigure(&configuration,0u,port);
	configuration.backend_module_path = "/missing/libnccl.so.2";
	assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
		SPARK_STATUS_DRIVER_LOAD_ERROR);
	puts("tp_device_collective_nccl: ok");
	return(0);
}
