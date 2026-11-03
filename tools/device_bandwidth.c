#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "inference/runner/device_ops.h"

#define BANDWIDTH_REPEATS 7u

static uint64_t NowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static int CompareDouble(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return(x < y ? -1 : x > y ? 1 : 0);
}

static int Measure(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor words, SparkTensor sink, uint32_t groups, double *median, double *best)
{
	double rates[BANDWIDTH_REPEATS + 1u];
	uint32_t repeat;
	for ( repeat = 0u; repeat <= BANDWIDTH_REPEATS; repeat++ )
	{
		const uint64_t start = NowNs();
		if ( SparkOpsStreamRead(ops,queue,words,sink,groups) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			return(-1);
		rates[repeat] = (double)words.bytes / ((NowNs() - start) / 1e9);
	}
	qsort(rates + 1,BANDWIDTH_REPEATS,sizeof(double),CompareDouble);
	*median = rates[1u + BANDWIDTH_REPEATS / 2u];
	*best = rates[BANDWIDTH_REPEATS];
	return(0);
}

static int Sweep(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor words, SparkTensor sink, const char *label, int first)
{
	static const uint32_t groups[] = { 256u, 512u, 1024u, 2048u, 4096u };
	double median, best, top_median = 0.0, top_best = 0.0;
	uint32_t index, top_groups = 0u;
	for ( index = 0u; index < sizeof(groups) / sizeof(groups[0]); index++ )
	{
		if ( Measure(ops,queue,words,sink,groups[index],&median,&best) != 0 )
			return(-1);
		fprintf(stderr,"%s: %u groups x 1024 threads: median %.1f GB/s, best %.1f GB/s\n",label,groups[index],median / 1e9,best / 1e9);
		if ( median > top_median )
		{
			top_median = median;
			top_best = best;
			top_groups = groups[index];
		}
	}
	printf("%s{\"buffer\":\"%s\",\"bytes\":%llu,\"groups\":%u,\"threads_per_group\":1024,\"repeats\":%u,\"warmup_discarded\":true,"
		"\"median_gb_per_s\":%.2f,\"best_gb_per_s\":%.2f}",first ? "" : ",\n",label,(unsigned long long)words.bytes,top_groups,BANDWIDTH_REPEATS,
		top_median / 1e9,top_best / 1e9);
	return(0);
}

int main(int argc, char **argv)
{
	const uint64_t bytes = argc >= 2 ? strtoull(argv[1],0,10) << 30 : 16ull << 30;
	SparkDeviceOps *ops = 0;
	SparkDeviceQueue *queue = 0;
	SparkDeviceBuffer buffer, sink, mapped;
	int status = 1, fd = -1;
	struct stat info;
	memset(&mapped,0,sizeof(mapped));
	if ( argc < 2 || argc > 3 || bytes == 0u )
	{
		fprintf(stderr,"usage: %s GIB [FILE]\n",argv[0]);
		return(2);
	}
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"bandwidth") != SPARK_STATUS_OK ||
		SparkDeviceOpsAcquire(&ops) != SPARK_STATUS_OK || SparkDeviceAllocate(&buffer,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"bandwidth buffer") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&sink,SPARK_DEVICE_SPACE_SHARED_COHERENT,4096u * sizeof(float),"bandwidth sink") != SPARK_STATUS_OK )
	{
		fprintf(stderr,"device setup failed: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	printf("{\"device\":\"%s\",\"target\":\"%s\",\"compute_units\":%u,\"runs\":[\n",SparkDeviceGetProperties()->name,SparkDeviceGetProperties()->target,
		SparkDeviceGetProperties()->compute_units);
	if ( SparkDeviceFill(queue,(SparkDeviceRange){ &buffer, 0u },0x5au,bytes) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK &&
		Sweep(ops,queue,SparkTensorAt(&buffer,0u,bytes),SparkTensorAt(&sink,0u,sink.bytes),"device_private",1) == 0 )
		status = 0;
	if ( status == 0 && argc == 3 )
	{
		fd = open(argv[2],O_RDONLY);
		status = fd >= 0 && fstat(fd,&info) == 0 ? 0 : 1;
		if ( status == 0 )
		{
			const uint64_t window = (uint64_t)info.st_size < 4u * bytes ? (uint64_t)info.st_size : 4u * bytes;
			status = SparkDeviceMapFile(&mapped,fd,0u,window / 16u * 16u,"bandwidth file") == SPARK_STATUS_OK ? 0 : 1;
		}
		if ( status != 0 )
			fprintf(stderr,"device_bandwidth: cannot map %s for the file sweep\n",argv[2]);
		if ( status == 0 )
			status = Sweep(ops,queue,SparkTensorAt(&mapped,0u,mapped.bytes),SparkTensorAt(&sink,0u,sink.bytes),"file_mapping",0);
		if ( mapped.handle != 0u )
			SparkDeviceFree(&mapped);
		if ( fd >= 0 )
			close(fd);
	}
	printf("\n]}\n");
	SparkDeviceFree(&buffer);
	SparkDeviceFree(&sink);
	SparkDeviceOpsRelease(ops);
	SparkDeviceQueueDestroy(queue);
	SparkDeviceShutdown();
	return(status);
}
