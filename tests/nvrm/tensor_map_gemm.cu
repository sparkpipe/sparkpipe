#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_gpu.h"
#include "tests/nvrm/tensor_map_gemms.cuh"

int main(int argc, char **argv)
{
	SparkGpuStream stream = 0;
	int multiprocessors = 0;
	const char *flavour = argc > 1 ? argv[1] : "this";
	build_out = stdout;
	if ( SparkGpuSetDevice(0) != SPARK_GPU_SUCCESS || SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&multiprocessors,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,0) != SPARK_GPU_SUCCESS )
	{
		fprintf(stderr,"FAIL tensor map GEMMs: setup on the %s flavour: %s\n",flavour,SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	BuildGemms(stream,(uint32_t)multiprocessors);
	SparkGpuStreamDestroy(stream);
	fprintf(stderr,"PASS tensor map GEMMs ran on the %s flavour\n",flavour);
	return(0);
}
