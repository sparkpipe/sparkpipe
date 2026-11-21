#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "tests/nvrm/fixture_kernels.cu"

#define BANDWIDTH_BLOCKS_PER_SM 4u
#define BANDWIDTH_ROUNDS_MAX 256u

static int Check(cudaError_t error, const char *step)
{
	if ( error == cudaSuccess )
		return(0);
	printf("FAIL %s: %s\n",step,cudaGetErrorString(error));
	return(1);
}

static int CompareTimes(const void *left, const void *right)
{
	const float a = *(const float *)left,b = *(const float *)right;
	return(a < b ? -1 : a > b);
}

template<class Launch>
static int Measure(const char *label, Launch launch, uint64_t moved, uint32_t rounds)
{
	float times[BANDWIDTH_ROUNDS_MAX];
	cudaEvent_t start,stop;
	uint32_t round;
	if ( Check(cudaEventCreate(&start),"create an event") != 0 || Check(cudaEventCreate(&stop),"create an event") != 0 )
		return(1);
	for (round = 0u; round < 3u; round++)
		launch();
	for (round = 0u; round < rounds; round++)
	{
		cudaEventRecord(start);
		launch();
		cudaEventRecord(stop);
		if ( Check(cudaEventSynchronize(stop),"run a measured launch") != 0 )
			return(1);
		cudaEventElapsedTime(&times[round],start,stop);
	}
	qsort(times,rounds,sizeof(times[0]),CompareTimes);
	printf("  %-5s median %8.2f GB/s  best %8.2f GB/s  (%u rounds, %.3f ms median)\n",label,(double)moved / (times[rounds / 2u] * 1e6),
		(double)moved / (times[0] * 1e6),rounds,times[rounds / 2u]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return(0);
}

int main(int argc, char **argv)
{
	uint64_t bytes = 1024ull << 20;
	uint32_t rounds = 20u,*sink = 0;
	uint4 *source = 0,*destination = 0;
	int index,sms = 0;
	for (index = 1; index < argc; index++)
	{
		if ( strcmp(argv[index],"--mib") == 0 && index + 1 < argc )
			bytes = strtoull(argv[++index],0,10) << 20;
		else if ( strcmp(argv[index],"--rounds") == 0 && index + 1 < argc )
			rounds = (uint32_t)strtoul(argv[++index],0,10);
		else
		{
			printf("usage: %s [--mib N] [--rounds N]\n",argv[0]);
			return(2);
		}
	}
	if ( rounds == 0u || rounds > BANDWIDTH_ROUNDS_MAX || bytes == 0u )
		return(2);
	const uint64_t count = bytes / 16u;
	if ( Check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0),"read the SM count") != 0 ||
		Check(cudaMalloc((void **)&source,bytes),"cudaMalloc the source") != 0 ||
		Check(cudaMalloc((void **)&destination,bytes),"cudaMalloc the destination") != 0 ||
		Check(cudaMalloc((void **)&sink,4096u),"cudaMalloc the sink") != 0 )
		return(1);
	const uint32_t blocks = (uint32_t)sms * BANDWIDTH_BLOCKS_PER_SM;
	printf("cuda memory bandwidth: %d SMs, %u blocks of %u threads\npath cudaMalloc, %llu MiB per buffer\n",sms,blocks,SPARK_NVRM_FIXTURE_STREAM_THREADS,
		(unsigned long long)(bytes >> 20));
	SparkNvrmStreamFill<<<blocks,SPARK_NVRM_FIXTURE_STREAM_THREADS>>>(source,count,0x13572468u);
	if ( Check(cudaDeviceSynchronize(),"fill the source") != 0 )
		return(1);
	if ( Measure("read",[&]() { SparkNvrmStreamRead<<<blocks,SPARK_NVRM_FIXTURE_STREAM_THREADS>>>(source,count,sink); },bytes,rounds) != 0 ||
		Measure("copy",[&]() { SparkNvrmStreamCopy<<<blocks,SPARK_NVRM_FIXTURE_STREAM_THREADS>>>(source,destination,count); },2u * bytes,rounds) != 0 )
		return(1);
	cudaFree(sink);
	cudaFree(destination);
	cudaFree(source);
	printf("PASS cudaMalloc measured\n");
	return(0);
}
