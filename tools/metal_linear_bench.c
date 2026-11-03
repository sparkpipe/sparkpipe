#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "inference/runner/device_ops.h"

static uint64_t NowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static int CompareU64(const void *a, const void *b)
{
	const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return(x < y ? -1 : x > y ? 1 : 0);
}

int main(int argc, char **argv)
{
	SparkDeviceQueue *queue = 0;
	SparkDeviceOps opened, *ops = &opened;
	SparkDeviceBuffer x, w, y;
	SparkGemvParams params;
	uint64_t times[16], index;
	uint32_t K, N, rows, f32, projection, run, launches = 8u;
	if ( argc != 7 )
	{
		fprintf(stderr,"usage: %s KERNEL_LIBRARY K N ROWS F32 ROW_TILES\n",argv[0]);
		return(2);
	}
	memset(&opened,0,sizeof(opened));
	K = (uint32_t)strtoul(argv[2],0,10);
	N = (uint32_t)strtoul(argv[3],0,10);
	rows = (uint32_t)strtoul(argv[4],0,10);
	f32 = (uint32_t)strtoul(argv[5],0,10);
	projection = strtoul(argv[6],0,10) != 0u ? SPARK_PROJECTION_ROW_TILES : SPARK_PROJECTION_GEMV;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"linear bench") != SPARK_STATUS_OK ||
		SparkDeviceOpsOpen(ops,argv[1]) != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&x,SPARK_DEVICE_SPACE_SHARED_COHERENT,(uint64_t)rows * K * sizeof(uint16_t),"bench x") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&w,SPARK_DEVICE_SPACE_SHARED_COHERENT,(uint64_t)N * K * sizeof(uint16_t),"bench w") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&y,SPARK_DEVICE_SPACE_SHARED_COHERENT,(uint64_t)rows * N * (f32 != 0u ? sizeof(float) : sizeof(uint16_t)),"bench y") != SPARK_STATUS_OK )
	{
		fprintf(stderr,"device setup failed: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for ( index = 0u; index < (uint64_t)N * K; index++ )
		((uint16_t *)w.host)[index] = (uint16_t)(0x3c00u + (index * 2654435761u >> 20) % 256u);
	for ( index = 0u; index < (uint64_t)rows * K; index++ )
		((uint16_t *)x.host)[index] = (uint16_t)(0x3f00u + (index * 40503u >> 8) % 128u);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = K;
	params.output_dimension = N;
	params.input_stride = K;
	params.output_stride = N;
	params.output_f32 = f32;
	params.projection = projection;
	for ( run = 0u; run <= 15u; run++ )
	{
		const uint64_t start = NowNs();
		uint32_t launch;
		for ( launch = 0u; launch < launches; launch++ )
			if ( SparkOpsLinear(ops,queue,SparkTensorAt(&x,0u,x.bytes),SparkTensorAt(&w,0u,w.bytes),SparkTensorAt(&y,0u,y.bytes),&params) != SPARK_STATUS_OK )
			{
				fprintf(stderr,"linear failed: %s\n",SparkDeviceLastError()->text);
				return(1);
			}
		if ( SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			return(1);
		times[run] = (NowNs() - start) / launches;
	}
	qsort(times + 1,15u,sizeof(uint64_t),CompareU64);
	printf("rows %u K %u N %u%s%s: %.1f us, %.0f GB/s weights, %.2f TFLOPS\n",rows,K,N,f32 != 0u ? " f32" : "",projection == SPARK_PROJECTION_ROW_TILES ? " row tiles" : "",times[8] / 1e3,
		(double)N * K * 2.0 / times[8],2.0 * rows * N * K / times[8] / 1e3);
	return(0);
}
