#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tp_mesh_kernels.cuh"

#define PROBE_BYTES 64u
#define PROBE_ROUND 5ull
#define PROBE_EARLIER_ERROR 0x1234ull

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

static void ProbeCase(const char *name, uint64_t error_before, uint64_t shipped_value, uint64_t cancel_value, uint64_t timeout_ns,
	uint64_t expect_error, uint32_t expect_copy)
{
	SparkTpMeshRoundControl control;
	uint8_t source[PROBE_BYTES], destination[PROBE_BYTES];
	uint8_t *device_source, *device_destination;
	uint64_t *device_shipped, *device_cancel;
	SparkTpMeshRoundControl *device_control;
	uint32_t index, copied = 1u;
	memset(&control, 0, sizeof(control));
	control.round_seq = PROBE_ROUND;
	control.cancel_expected = 7ull;
	control.error_word = error_before;
	for (index = 0u; index < PROBE_BYTES; ++index)
		source[index] = (uint8_t)(index + 1u);
	PROBE_CUDA(cudaMalloc((void **)&device_source, PROBE_BYTES));
	PROBE_CUDA(cudaMalloc((void **)&device_destination, PROBE_BYTES));
	PROBE_CUDA(cudaMalloc((void **)&device_shipped, sizeof(uint64_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_cancel, sizeof(uint64_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_control, sizeof(control)));
	PROBE_CUDA(cudaMemcpy(device_source, source, PROBE_BYTES, cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemset(device_destination, 0, PROBE_BYTES));
	PROBE_CUDA(cudaMemcpy(device_shipped, &shipped_value, sizeof(uint64_t), cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_cancel, &cancel_value, sizeof(uint64_t), cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_control, &control, sizeof(control), cudaMemcpyHostToDevice));
	PROBE_CUDA(SparkTpLaunchMeshCopyDown(0, device_destination, device_source, PROBE_BYTES, device_shipped, device_control, device_cancel, timeout_ns));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(destination, device_destination, PROBE_BYTES, cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(&control, device_control, sizeof(control), cudaMemcpyDeviceToHost));
	for (index = 0u; index < PROBE_BYTES; ++index)
		if ( destination[index] != source[index] )
			copied = 0u;
	if ( control.error_word != expect_error || copied != expect_copy )
	{
		printf("FAIL %s: error word 0x%llx (want 0x%llx), copied %u (want %u)\n", name, (unsigned long long)control.error_word,
			(unsigned long long)expect_error, copied, expect_copy);
		probe_failures++;
	}
	cudaFree(device_source);
	cudaFree(device_destination);
	cudaFree(device_shipped);
	cudaFree(device_cancel);
	cudaFree(device_control);
}

int main(void)
{
	ProbeCase("shipped round copies", 0ull, PROBE_ROUND, 7ull, 1000000000ull, 0ull, 1u);
	ProbeCase("cancel while waiting", 0ull, 0ull, 8ull, 1000000000ull, SPARK_TP_MESH_ERROR_CANCELLED | PROBE_ROUND, 0u);
	ProbeCase("timeout while waiting", 0ull, 0ull, 7ull, 1000000ull, PROBE_ROUND, 0u);
	ProbeCase("earlier error while waiting", PROBE_EARLIER_ERROR, 0ull, 7ull, 1000000000ull, PROBE_EARLIER_ERROR, 0u);
	ProbeCase("earlier error then cancel", PROBE_EARLIER_ERROR, 0ull, 8ull, 1000000000ull, PROBE_EARLIER_ERROR, 0u);
	ProbeCase("earlier error on a shipped round", PROBE_EARLIER_ERROR, PROBE_ROUND, 8ull, 1000000000ull, PROBE_EARLIER_ERROR, 0u);
	if ( probe_failures != 0 )
	{
		printf("FAIL mesh_copy_down_cuda: %d failures\n", probe_failures);
		return(1);
	}
	printf("PASS mesh_copy_down_cuda: copy-down copies a shipped round, reports cancel and timeout, and keeps an earlier error code\n");
	return(0);
}
