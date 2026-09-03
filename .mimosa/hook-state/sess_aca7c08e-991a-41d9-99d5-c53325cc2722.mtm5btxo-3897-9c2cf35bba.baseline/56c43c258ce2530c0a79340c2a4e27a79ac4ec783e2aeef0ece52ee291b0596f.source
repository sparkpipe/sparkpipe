#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"

static void Completion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context;
	(void)completion;
}

int main(int argc, char **argv)
{
	SparkTpDeviceCollectiveConfig config;
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveSubmission submission;
	static const char *hosts[16] = {
		"10.10.100.10", "10.10.100.11", "10.10.100.12", "10.10.100.13",
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	void *buffer;
	cudaStream_t stream;
	SparkStatus status;
	uint32_t rank, iterations, rows, i;
	uint64_t start, elapsed;
	struct timespec ts;
	if ( argc != 3 ) return(2);
	rank = (uint32_t)strtoul(argv[1], 0, 0);
	iterations = (uint32_t)strtoul(argv[2], 0, 0);
	rows = 1u;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	config.tp_degree = 4u;
	config.tp_rank = rank;
	config.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	config.credit_count = 1u;
	config.local_hidden_dimension = 5120u;
	config.max_active_sequence_count = 8u;
	config.connect_timeout_milli = 120000u;
	config.operation_timeout_milli = 120000u;
	config.control_port_base = 62620u;
	config.collective_identifier = 0x513630545032ull;
	config.backend_module_path = "libnccl.so.2";
	config.local_host = hosts[rank];
	for (i = 0u; i < 16u; i++)
		config.rank_hosts[i] = hosts[i];
	if ( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
		cudaMalloc(&buffer, (size_t)rows * 5120u * 2u) != cudaSuccess )
		return(1);
	config.registration_cuda_stream = stream;
	status = SparkTpDeviceCollectiveCreate(&config, &collective);
	if ( status != SPARK_STATUS_OK ) { fprintf(stderr, "create status=%d\n", (int)status); return(1); }
	printf("rank=%u collective ready, warming up...\n", rank);
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = rows;
	submission.local_device = buffer;
	submission.full_device = buffer;
	submission.cuda_stream = stream;
	submission.completion_function = Completion;
	for (i = 0u; i < 10u; i++)
	{
		submission.ordinal = i;
		if ( SparkTpDeviceCollectiveSubmitBf16(&collective, &submission) != SPARK_STATUS_OK )
			return(1);
	}
	if ( cudaStreamSynchronize(stream) != cudaSuccess ) return(1);
	clock_gettime(CLOCK_MONOTONIC, &ts);
	start = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
	for (i = 0u; i < iterations; i++)
	{
		submission.ordinal = 10u + i;
		if ( SparkTpDeviceCollectiveSubmitBf16(&collective, &submission) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "submit failed at %u\n", i);
			return(1);
		}
	}
	if ( cudaStreamSynchronize(stream) != cudaSuccess ) return(1);
	clock_gettime(CLOCK_MONOTONIC, &ts);
	elapsed = ((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec) - start;
	printf("rank=%u iterations=%u total_us=%llu per_op_us=%.1f\n", rank, iterations, (unsigned long long)(elapsed / 1000ull), (double)elapsed / 1000.0 / (double)iterations);
	SparkTpDeviceCollectiveDestroy(&collective);
	return(0);
}
