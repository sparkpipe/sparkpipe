#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"

#define TRANSFER_ROUNDS_MAX 100000u
#define TRANSFER_WARMUP_ROUNDS 64u
#define TRANSFER_SLOTS_MAX 8u
#define TRANSFER_ROW_COUNTS_MAX 8u
#define TRANSFER_SAMPLES 64u
#define TRANSFER_SAMPLE_BYTES 4096u
#define TRANSFER_DIRECT_ALIGNMENT 4096u
#define TRANSFER_STEP_TIMINGS 4u

typedef struct StepTimes
{
	uint64_t *upload;
	uint64_t *download;
	uint64_t *step;
	uint64_t *written_step;
} StepTimes;

typedef struct LoadPlan
{
	const char *path;
	uint64_t chunk_bytes;
	uint32_t slots;
	uint32_t pageable;
	uint32_t direct;
	uint32_t touch;
	uint32_t back;
} LoadPlan;

static uint64_t NowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static int Failed(const char *step)
{
	const SparkDeviceError *error = SparkDeviceLastError();
	printf("FAIL %s: %s\n",step,error != 0 && error->text[0] != '\0' ? error->text : "no device error recorded");
	return(1);
}

static int CompareTimes(const void *left, const void *right)
{
	const uint64_t a = *(const uint64_t *)left, b = *(const uint64_t *)right;
	return(a < b ? -1 : a > b ? 1 : 0);
}

static double Percentile(uint64_t *times, uint32_t count, uint32_t percent)
{
	qsort(times,count,sizeof(*times),CompareTimes);
	return((double)times[(uint64_t)(count - 1u) * percent / 100u] / 1000.0);
}

static SparkDeviceRange Range(const SparkDeviceBuffer *buffer, uint64_t offset)
{
	SparkDeviceRange range;
	range.buffer = buffer;
	range.offset = offset;
	return(range);
}

static int Copied(SparkDeviceQueue *queue, const SparkDeviceBuffer *destination, const SparkDeviceBuffer *source, uint64_t bytes)
{
	return(SparkDeviceCopy(queue,Range(destination,0u),Range(source,0u),bytes) == SPARK_STATUS_OK ? 0 : 1);
}

static int StepRound(SparkDeviceQueue *queue, SparkDeviceBuffer *buffers, const uint8_t *source, uint64_t tap_bytes, uint64_t draft_bytes,
	StepTimes *times, uint32_t round)
{
	uint64_t started = NowNs();
	if ( Copied(queue,&buffers[1],&buffers[0],tap_bytes) != 0 || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return(Failed("upload the taps"));
	times->upload[round] = NowNs() - started;
	started = NowNs();
	if ( Copied(queue,&buffers[2],&buffers[3],draft_bytes) != 0 || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return(Failed("download the draft"));
	times->download[round] = NowNs() - started;
	started = NowNs();
	if ( Copied(queue,&buffers[1],&buffers[0],tap_bytes) != 0 || Copied(queue,&buffers[2],&buffers[3],draft_bytes) != 0 ||
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return(Failed("run the step's transfers"));
	times->step[round] = NowNs() - started;
	started = NowNs();
	memcpy(buffers[0].host,source,(size_t)tap_bytes);
	if ( Copied(queue,&buffers[1],&buffers[0],tap_bytes) != 0 || Copied(queue,&buffers[2],&buffers[3],draft_bytes) != 0 ||
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return(Failed("write the taps and run the step's transfers"));
	times->written_step[round] = NowNs() - started;
	return(0);
}

static int StepVerified(SparkDeviceQueue *queue, SparkDeviceBuffer *buffers, const uint8_t *source, uint64_t tap_bytes)
{
	SparkDeviceBuffer back;
	int result;
	if ( SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PINNED,tap_bytes,"step check") != SPARK_STATUS_OK )
		return(Failed("allocate the check buffer"));
	result = Copied(queue,&back,&buffers[1],tap_bytes) != 0 || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK ?
		Failed("copy the taps back") : 0;
	if ( result == 0 && memcmp(back.host,source,(size_t)tap_bytes) != 0 )
	{
		printf("FAIL the taps on the device differ from the rows written\n");
		result = 1;
	}
	(void)SparkDeviceFree(&back);
	return(result);
}

static int MeasureStep(SparkDeviceQueue *queue, uint32_t rows, uint64_t row_bytes, uint64_t draft_bytes, uint32_t rounds)
{
	SparkDeviceBuffer buffers[4];
	StepTimes times;
	const uint64_t tap_bytes = (uint64_t)rows * row_bytes;
	uint8_t *source = (uint8_t *)malloc((size_t)tap_bytes);
	uint64_t *storage = (uint64_t *)calloc((size_t)rounds * TRANSFER_STEP_TIMINGS,sizeof(uint64_t)),index;
	uint32_t round;
	int result = 0;
	memset(buffers,0,sizeof(buffers));
	if ( source == 0 || storage == 0 )
	{
		free(source);
		free(storage);
		printf("FAIL out of host memory for %u rounds\n",rounds);
		return(1);
	}
	for (index = 0u; index < tap_bytes; index++)
		source[index] = (uint8_t)(index * 131u + rows);
	times.upload = storage;
	times.download = storage + rounds;
	times.step = storage + 2u * (uint64_t)rounds;
	times.written_step = storage + 3u * (uint64_t)rounds;
	if ( SparkDeviceAllocate(&buffers[0],SPARK_DEVICE_SPACE_HOST_PINNED,tap_bytes,"step taps") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&buffers[1],SPARK_DEVICE_SPACE_DEVICE_PRIVATE,tap_bytes,"step device taps") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&buffers[2],SPARK_DEVICE_SPACE_HOST_PINNED,draft_bytes,"step draft") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&buffers[3],SPARK_DEVICE_SPACE_DEVICE_PRIVATE,draft_bytes,"step device draft") != SPARK_STATUS_OK )
		result = Failed("allocate the step's buffers");
	if ( result == 0 )
		memcpy(buffers[0].host,source,(size_t)tap_bytes);
	for (round = 0u; result == 0 && round < TRANSFER_WARMUP_ROUNDS; round++)
		result = StepRound(queue,buffers,source,tap_bytes,draft_bytes,&times,0u);
	for (round = 0u; result == 0 && round < rounds; round++)
		result = StepRound(queue,buffers,source,tap_bytes,draft_bytes,&times,round);
	if ( result == 0 )
		result = StepVerified(queue,buffers,source,tap_bytes);
	if ( result == 0 )
	{
		printf("step rows %2u (%7llu tap bytes up, %llu draft bytes down), median/p99 us:", rows,(unsigned long long)tap_bytes,(unsigned long long)draft_bytes);
		printf("  upload %7.1f/%7.1f",Percentile(times.upload,rounds,50u),Percentile(times.upload,rounds,99u));
		printf("  download %6.1f/%6.1f",Percentile(times.download,rounds,50u),Percentile(times.download,rounds,99u));
		printf("  both %7.1f/%7.1f",Percentile(times.step,rounds,50u),Percentile(times.step,rounds,99u));
		printf("  rows written then both %7.1f/%7.1f\n",Percentile(times.written_step,rounds,50u),Percentile(times.written_step,rounds,99u));
	}
	for (index = 0u; index < 4u; index++)
		if ( buffers[index].bytes != 0u )
			(void)SparkDeviceFree(&buffers[index]);
	free(storage);
	free(source);
	return(result);
}

static int ReadChunk(int fd, uint8_t *destination, uint64_t offset, uint64_t bytes, uint64_t file_bytes, uint32_t direct)
{
	const uint64_t want = direct != 0u ? (bytes + TRANSFER_DIRECT_ALIGNMENT - 1u) / TRANSFER_DIRECT_ALIGNMENT * TRANSFER_DIRECT_ALIGNMENT : bytes;
	uint64_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t moved = pread(fd,destination + done,(size_t)(want - done),(off_t)(offset + done));
		if ( moved < 0 && errno == EINTR )
			continue;
		if ( moved <= 0 )
		{
			printf("FAIL read %llu bytes at %llu of %llu: %s\n",(unsigned long long)(want - done),(unsigned long long)(offset + done),
				(unsigned long long)file_bytes,moved < 0 ? strerror(errno) : "end of file");
			return(1);
		}
		done += (uint64_t)moved;
	}
	return(0);
}

static int LoadVerified(SparkDeviceQueue *queue, const SparkDeviceBuffer *weights, const char *path, uint64_t file_bytes)
{
	SparkDeviceBuffer seen;
	uint8_t expected[TRANSFER_SAMPLE_BYTES];
	const int fd = open(path,O_RDONLY | O_CLOEXEC);
	uint32_t sample;
	int result = fd >= 0 ? 0 : 1;
	if ( result != 0 )
	{
		printf("FAIL reopen %s: %s\n",path,strerror(errno));
		return(1);
	}
	if ( SparkDeviceAllocate(&seen,SPARK_DEVICE_SPACE_HOST_PINNED,TRANSFER_SAMPLE_BYTES,"load check") != SPARK_STATUS_OK )
	{
		close(fd);
		return(Failed("allocate the check buffer"));
	}
	for (sample = 0u; result == 0 && sample < TRANSFER_SAMPLES; sample++)
	{
		const uint64_t span = file_bytes > TRANSFER_SAMPLE_BYTES ? file_bytes - TRANSFER_SAMPLE_BYTES : 0u;
		const uint64_t spread = span / (TRANSFER_SAMPLES - 1u) * sample + sample * 61u;
		const uint64_t offset = spread < span ? spread : span;
		const uint64_t bytes = file_bytes < TRANSFER_SAMPLE_BYTES ? file_bytes : TRANSFER_SAMPLE_BYTES;
		if ( SparkDeviceCopy(queue,Range(&seen,0u),Range(weights,offset),bytes) != SPARK_STATUS_OK ||
			SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			result = Failed("copy a sample back");
		else if ( ReadChunk(fd,expected,offset,bytes,file_bytes,0u) != 0 )
			result = 1;
		else if ( memcmp(seen.host,expected,(size_t)bytes) != 0 )
		{
			printf("FAIL the device copy differs from the file at %llu\n",(unsigned long long)offset);
			result = 1;
		}
	}
	(void)SparkDeviceFree(&seen);
	close(fd);
	return(result);
}

static int LoadChunks(SparkDeviceQueue *queue, int fd, const LoadPlan *plan, SparkDeviceBuffer *weights, SparkDeviceBuffer *staging,
	SparkDeviceEvent **events, uint64_t file_bytes)
{
	uint64_t offset;
	uint32_t chunk = 0u;
	for (offset = 0u; offset < file_bytes; offset += plan->chunk_bytes, chunk++)
	{
		const uint32_t slot = chunk % plan->slots;
		const uint64_t bytes = file_bytes - offset < plan->chunk_bytes ? file_bytes - offset : plan->chunk_bytes;
		if ( chunk >= plan->slots && SparkDeviceEventWait(events[slot],SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			return(Failed("wait for a staging slot"));
		if ( ReadChunk(fd,(uint8_t *)staging[slot].host,offset,bytes,file_bytes,plan->direct) != 0 )
			return(1);
		if ( SparkDeviceCopy(queue,Range(weights,offset),Range(&staging[slot],0u),bytes) != SPARK_STATUS_OK ||
			SparkDeviceEventRecord(events[slot],queue) != SPARK_STATUS_OK )
			return(Failed("copy a chunk to the device"));
	}
	if ( SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return(Failed("wait for the last chunk"));
	return(0);
}

static int CopyBack(SparkDeviceQueue *queue, const LoadPlan *plan, const SparkDeviceBuffer *weights, SparkDeviceBuffer *staging, uint64_t file_bytes)
{
	uint64_t offset;
	uint32_t chunk = 0u;
	for (offset = 0u; offset < file_bytes; offset += plan->chunk_bytes, chunk++)
	{
		const uint64_t bytes = file_bytes - offset < plan->chunk_bytes ? file_bytes - offset : plan->chunk_bytes;
		if ( SparkDeviceCopy(queue,Range(&staging[chunk % plan->slots],0u),Range(weights,offset),bytes) != SPARK_STATUS_OK )
			return(Failed("copy a chunk back to the host"));
	}
	if ( SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return(Failed("wait for the copies back"));
	return(0);
}

static void Touch(volatile uint8_t *memory, uint64_t bytes)
{
	uint64_t offset;
	for (offset = 0u; offset < bytes; offset += TRANSFER_DIRECT_ALIGNMENT)
		memory[offset] = memory[offset];
}

static int MeasureLoad(SparkDeviceQueue *queue, const LoadPlan *plan)
{
	SparkDeviceBuffer weights,staging[TRANSFER_SLOTS_MAX];
	SparkDeviceEvent *events[TRANSFER_SLOTS_MAX];
	struct stat info;
	uint64_t started,device_ready,staging_ready,loaded,backed,staging_freed;
	uint32_t slot;
	int fd,result = 0;
	memset(&weights,0,sizeof(weights));
	memset(staging,0,sizeof(staging));
	memset(events,0,sizeof(events));
	started = NowNs();
	fd = open(plan->path,O_RDONLY | O_CLOEXEC | (plan->direct != 0u ? O_DIRECT : 0));
	if ( fd < 0 || fstat(fd,&info) != 0 || info.st_size <= 0 )
	{
		printf("FAIL open %s: %s\n",plan->path,strerror(errno));
		if ( fd >= 0 )
			close(fd);
		return(1);
	}
	if ( SparkDeviceAllocate(&weights,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,(uint64_t)info.st_size,"load weights") != SPARK_STATUS_OK )
		result = Failed("allocate the device weights");
	device_ready = NowNs();
	for (slot = 0u; result == 0 && slot < plan->slots; slot++)
		if ( SparkDeviceAllocate(&staging[slot],plan->pageable != 0u ? SPARK_DEVICE_SPACE_HOST_PAGEABLE : SPARK_DEVICE_SPACE_HOST_PINNED,
			plan->chunk_bytes,"load staging") != SPARK_STATUS_OK || SparkDeviceEventCreate(&events[slot]) != SPARK_STATUS_OK )
			result = Failed("allocate the staging slots");
		else if ( plan->direct != 0u && (uintptr_t)staging[slot].host % TRANSFER_DIRECT_ALIGNMENT != 0u )
		{
			printf("FAIL direct reads need staging aligned to %u bytes\n",TRANSFER_DIRECT_ALIGNMENT);
			result = 1;
		}
		else if ( plan->touch != 0u )
			Touch((volatile uint8_t *)staging[slot].host,plan->chunk_bytes);
	staging_ready = NowNs();
	if ( result == 0 )
		result = LoadChunks(queue,fd,plan,&weights,staging,events,(uint64_t)info.st_size);
	loaded = NowNs();
	if ( result == 0 && plan->back != 0u )
		result = CopyBack(queue,plan,&weights,staging,(uint64_t)info.st_size);
	backed = NowNs();
	for (slot = 0u; slot < plan->slots; slot++)
	{
		if ( events[slot] != 0 )
			(void)SparkDeviceEventDestroy(events[slot]);
		if ( staging[slot].bytes != 0u )
			(void)SparkDeviceFree(&staging[slot]);
	}
	staging_freed = NowNs();
	close(fd);
	if ( result == 0 )
		result = LoadVerified(queue,&weights,plan->path,(uint64_t)info.st_size);
	if ( result == 0 )
		printf("load %.2f GB, %s staging %u x %llu MiB%s%s: end to end %.1f ms (%.2f GB/s): device memory %.1f ms, staging %.1f ms, "
			"read and copy %.1f ms (%.2f GB/s), staging freed %.1f ms\n",(double)info.st_size / 1e9,plan->pageable != 0u ? "pageable" : "pinned",
			plan->slots,(unsigned long long)(plan->chunk_bytes >> 20),plan->touch != 0u ? " stored to once per page" : "",
			plan->direct != 0u ? ", direct reads" : ", page-cache reads",
			(double)(staging_freed - started - (backed - loaded)) / 1e6,(double)info.st_size / (double)(staging_freed - started - (backed - loaded)),
			(double)(device_ready - started) / 1e6,(double)(staging_ready - device_ready) / 1e6,(double)(loaded - staging_ready) / 1e6,
			(double)info.st_size / (double)(loaded - staging_ready),(double)(staging_freed - backed) / 1e6);
	if ( result == 0 && plan->back != 0u )
		printf("back %.2f GB from the device into the %s staging: %.1f ms (%.2f GB/s)\n",(double)info.st_size / 1e9,
			plan->pageable != 0u ? "pageable" : "pinned",(double)(backed - loaded) / 1e6,(double)info.st_size / (double)(backed - loaded));
	if ( weights.bytes != 0u )
		(void)SparkDeviceFree(&weights);
	return(result);
}

static uint32_t RowCounts(const char *text, uint32_t *counts)
{
	uint32_t count = 0u;
	while ( text != 0 && *text != '\0' && count < TRANSFER_ROW_COUNTS_MAX )
	{
		char *end = 0;
		const unsigned long value = strtoul(text,&end,10);
		if ( end == text || value == 0u || value > 4096u )
			return(0u);
		counts[count++] = (uint32_t)value;
		text = *end == ',' ? end + 1 : end;
		if ( *end != ',' && *end != '\0' )
			return(0u);
	}
	return(text != 0 && *text == '\0' ? count : 0u);
}

static int Usage(const char *program)
{
	printf("usage: %s step [--rows N,N,...] [--row-bytes N] [--draft-bytes N] [--rounds N]\n"
		"       %s load FILE [--chunk-mib N] [--slots N] [--pageable] [--direct] [--touch] [--back]\n",program,program);
	return(2);
}

int main(int argc, char **argv)
{
	uint32_t rows[TRANSFER_ROW_COUNTS_MAX] = { 1u, 4u, 8u, 16u },row_count = 4u,rounds = 2000u,index;
	uint64_t row_bytes = 5u * 5120u * 2u,draft_bytes = 4096u;
	LoadPlan plan;
	SparkDeviceQueue *queue = 0;
	const SparkDeviceProperties *properties;
	int argument,result = 0,load;
	memset(&plan,0,sizeof(plan));
	plan.chunk_bytes = 64ull << 20;
	plan.slots = 4u;
	if ( argc < 2 || (strcmp(argv[1],"step") != 0 && strcmp(argv[1],"load") != 0) )
		return(Usage(argv[0]));
	load = strcmp(argv[1],"load") == 0;
	argument = 2;
	if ( load != 0 )
	{
		if ( argc < 3 )
			return(Usage(argv[0]));
		plan.path = argv[argument++];
	}
	for (; argument < argc; argument++)
	{
		if ( load == 0 && strcmp(argv[argument],"--rows") == 0 && argument + 1 < argc )
			row_count = RowCounts(argv[++argument],rows);
		else if ( load == 0 && strcmp(argv[argument],"--row-bytes") == 0 && argument + 1 < argc )
			row_bytes = strtoull(argv[++argument],0,10);
		else if ( load == 0 && strcmp(argv[argument],"--draft-bytes") == 0 && argument + 1 < argc )
			draft_bytes = strtoull(argv[++argument],0,10);
		else if ( load == 0 && strcmp(argv[argument],"--rounds") == 0 && argument + 1 < argc )
			rounds = (uint32_t)strtoul(argv[++argument],0,10);
		else if ( load != 0 && strcmp(argv[argument],"--chunk-mib") == 0 && argument + 1 < argc )
			plan.chunk_bytes = strtoull(argv[++argument],0,10) << 20;
		else if ( load != 0 && strcmp(argv[argument],"--slots") == 0 && argument + 1 < argc )
			plan.slots = (uint32_t)strtoul(argv[++argument],0,10);
		else if ( load != 0 && strcmp(argv[argument],"--pageable") == 0 )
			plan.pageable = 1u;
		else if ( load != 0 && strcmp(argv[argument],"--direct") == 0 )
			plan.direct = 1u;
		else if ( load != 0 && strcmp(argv[argument],"--touch") == 0 )
			plan.touch = 1u;
		else if ( load != 0 && strcmp(argv[argument],"--back") == 0 )
			plan.back = 1u;
		else
			return(Usage(argv[0]));
	}
	if ( row_count == 0u || row_bytes == 0u || draft_bytes == 0u || rounds == 0u || rounds > TRANSFER_ROUNDS_MAX || plan.chunk_bytes == 0u ||
		plan.slots == 0u || plan.slots > TRANSFER_SLOTS_MAX )
		return(Usage(argv[0]));
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"transfers") != SPARK_STATUS_OK )
		return(Failed("open the device"));
	properties = SparkDeviceGetProperties();
	printf("device %s (%s, %s)\n",properties->name,properties->backend,properties->target);
	if ( load != 0 )
		result = MeasureLoad(queue,&plan);
	for (index = 0u; load == 0 && result == 0 && index < row_count; index++)
		result = MeasureStep(queue,rows[index],row_bytes,draft_bytes,rounds);
	(void)SparkDeviceQueueDestroy(queue);
	(void)SparkDeviceShutdown();
	if ( result == 0 )
		printf("PASS every transfer arrived intact\n");
	return(result);
}
