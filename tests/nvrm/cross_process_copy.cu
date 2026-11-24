#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define CROSS_COPY_BYTES (64ull << 20)
#define CROSS_HOLD_NS 50000000ull
#define CROSS_SPLIT_GRIDS 25u
#define CROSS_START_DELAY_US 5000u
#define CROSS_REPEATS 5
#define CROSS_THREADS 256

__global__ void CrossHold(unsigned long long hold_ns, unsigned long long *sink)
{
	unsigned long long start,now,spins = 0u;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
	{
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
		spins++;
	}
	while ( now - start < hold_ns );
	if ( spins == 0u )
		*sink = spins;
}

static uint64_t CrossNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static int CrossCompare(const void *a, const void *b)
{
	const uint64_t x = *(const uint64_t *)a,y = *(const uint64_t *)b;
	return(x < y ? -1 : x > y);
}

static int Copier(int go, int done)
{
	SparkGpuStream stream = 0;
	void *host = 0,*device = 0;
	char byte = 0;
	uint64_t alone,contended,start;
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMallocHost(&host,CROSS_COPY_BYTES) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&device,CROSS_COPY_BYTES) != SPARK_GPU_SUCCESS )
		return(2);
	memset(host,7,CROSS_COPY_BYTES);
	start = CrossNow();
	if ( SparkGpuMemcpyAsync(device,host,CROSS_COPY_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
		return(2);
	start = CrossNow();
	if ( SparkGpuMemcpyAsync(device,host,CROSS_COPY_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
		return(2);
	alone = CrossNow() - start;
	if ( write(done,&byte,1u) != 1 || read(go,&byte,1u) != 1 )
		return(2);
	start = CrossNow();
	if ( SparkGpuMemcpyAsync(device,host,CROSS_COPY_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
		return(2);
	contended = CrossNow() - start;
	return(write(done,&contended,sizeof(contended)) == (ssize_t)sizeof(contended) && write(done,&alone,sizeof(alone)) == (ssize_t)sizeof(alone) ? 0 : 2);
}

static int EngineRound(const char *self, SparkGpuStream stream, unsigned long long *sink, int blocks, uint32_t grids, uint64_t *contended, uint64_t *alone, uint64_t *span)
{
	int go[2],done[2],status = 0;
	char arguments[2][16],byte = 1;
	uint32_t grid;
	uint64_t start;
	pid_t child;
	if ( pipe(go) != 0 || pipe(done) != 0 )
		return(0);
	snprintf(arguments[0],sizeof(arguments[0]),"%d",go[0]);
	snprintf(arguments[1],sizeof(arguments[1]),"%d",done[1]);
	child = fork();
	if ( child == 0 )
	{
		execl(self,self,"--copier",arguments[0],arguments[1],(char *)0);
		_exit(127);
	}
	if ( child < 0 || read(done[0],&byte,1u) != 1 )
		return(0);
	start = CrossNow();
	for (grid = 0u; grid < grids; grid++)
		LM_LAUNCH((CrossHold),blocks,CROSS_THREADS,0,stream,CROSS_HOLD_NS / grids,sink);
	usleep(CROSS_START_DELAY_US);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS || write(go[1],&byte,1u) != 1 || read(done[0],contended,sizeof(*contended)) != (ssize_t)sizeof(*contended) ||
		read(done[0],alone,sizeof(*alone)) != (ssize_t)sizeof(*alone) || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
		return(0);
	*span = CrossNow() - start;
	close(go[0]);
	close(go[1]);
	close(done[0]);
	close(done[1]);
	return(waitpid(child,&status,0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

static void SameProcess(const char *flavour, SparkGpuStream stream, unsigned long long *sink, int blocks)
{
	SparkGpuStream copier = 0;
	void *host = 0,*device = 0;
	uint64_t times[CROSS_REPEATS],alone,start;
	int repeat;
	if ( SparkGpuStreamCreateWithFlags(&copier,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMallocHost(&host,CROSS_COPY_BYTES) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&device,CROSS_COPY_BYTES) != SPARK_GPU_SUCCESS || SparkGpuMemcpyAsync(device,host,CROSS_COPY_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,copier) != SPARK_GPU_SUCCESS ||
		SparkGpuStreamSynchronize(copier) != SPARK_GPU_SUCCESS )
		return;
	start = CrossNow();
	if ( SparkGpuMemcpyAsync(device,host,CROSS_COPY_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,copier) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(copier) != SPARK_GPU_SUCCESS )
		return;
	alone = CrossNow() - start;
	for (repeat = 0; repeat < CROSS_REPEATS; repeat++)
	{
		LM_LAUNCH((CrossHold),blocks,CROSS_THREADS,0,stream,CROSS_HOLD_NS,sink);
		usleep(CROSS_START_DELAY_US);
		start = CrossNow();
		if ( SparkGpuMemcpyAsync(device,host,CROSS_COPY_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,copier) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(copier) != SPARK_GPU_SUCCESS )
			return;
		times[repeat] = CrossNow() - start;
		if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
			return;
	}
	qsort(times,CROSS_REPEATS,sizeof(times[0]),CrossCompare);
	printf("same %s: a second stream's 64 MiB copy issued 5 ms into the first stream's 50 ms grid takes %.2f ms median (%.2f-%.2f), %.2f ms alone\n",flavour,
		times[CROSS_REPEATS / 2] / 1e6,times[0] / 1e6,times[CROSS_REPEATS - 1] / 1e6,alone / 1e6);
}

int main(int argc, char **argv)
{
	const char *flavour = argc > 1 && strcmp(argv[1],"--copier") != 0 ? argv[1] : "this flavour";
	static const uint32_t shapes[] = {1u, CROSS_SPLIT_GRIDS};
	SparkGpuStream stream = 0;
	unsigned long long *sink = 0;
	int sms = 0,per_sm = 0,blocks;
	uint32_t shape;
	if ( argc == 4 && strcmp(argv[1],"--copier") == 0 )
		return(Copier(atoi(argv[2]),atoi(argv[3])));
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&sink,sizeof(*sink)) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&sms,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,0) != SPARK_GPU_SUCCESS ||
		SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,CrossHold,CROSS_THREADS,0) != SPARK_GPU_SUCCESS || sms <= 0 || per_sm <= 0 )
	{
		printf("FAIL set up the engine: %s\n",SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	blocks = sms * per_sm;
	for (shape = 0u; shape < sizeof(shapes) / sizeof(shapes[0]); shape++)
	{
		uint64_t contended[CROSS_REPEATS],alone[CROSS_REPEATS],span[CROSS_REPEATS];
		int repeat;
		for (repeat = 0; repeat < CROSS_REPEATS; repeat++)
			if ( EngineRound(argv[0],stream,sink,blocks,shapes[shape],&contended[repeat],&alone[repeat],&span[repeat]) == 0 )
			{
				printf("FAIL a cross-process round in %s\n",flavour);
				return(1);
			}
		qsort(contended,CROSS_REPEATS,sizeof(contended[0]),CrossCompare);
		qsort(alone,CROSS_REPEATS,sizeof(alone[0]),CrossCompare);
		qsort(span,CROSS_REPEATS,sizeof(span[0]),CrossCompare);
		printf("cross %s: engine holds every SM (%d blocks) for 50 ms as %u grid(s); another process's 64 MiB copy issued 5 ms in takes %.2f ms median "
			"(%.2f-%.2f), %.2f ms alone; engine span %.2f ms\n",flavour,blocks,shapes[shape],contended[CROSS_REPEATS / 2] / 1e6,contended[0] / 1e6,
			contended[CROSS_REPEATS - 1] / 1e6,alone[CROSS_REPEATS / 2] / 1e6,span[CROSS_REPEATS / 2] / 1e6);
	}
	SameProcess(flavour,stream,sink,blocks);
	return(0);
}
