#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define COHERENCE_BYTES (1ull << 30)
#define COHERENCE_WORDS (COHERENCE_BYTES / sizeof(uint64_t))
#define COHERENCE_CHUNKS 1024u
#define COHERENCE_CHUNK_WORDS (COHERENCE_WORDS / COHERENCE_CHUNKS)
#define COHERENCE_LANES 32u
#define COHERENCE_HOLD_NS 2000000ull
#define COHERENCE_WRITE_GRIDS 64u
#define COHERENCE_GRID_WORDS (COHERENCE_WORDS / COHERENCE_WRITE_GRIDS)
#define COHERENCE_ROUNDS 1000u
#define COHERENCE_ROUND_BYTES (2ull << 20)
#define COHERENCE_ROUND_PROBE 8192u
#define COHERENCE_LEAK_SLACK (256ull << 20)
#define COHERENCE_SEEN 0
#define COHERENCE_MISMATCH 1
#define COHERENCE_BROKEN 2

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			const SparkDeviceError *last = SparkDeviceLastError(); \
			printf("FAIL %s:%d %s (last error: %s %s)\n",__FILE__,__LINE__,what,last->operation,last->text); \
			failures++; \
		} \
	} while (0)

__host__ __device__ static uint64_t CoherenceWord(uint64_t seed, uint64_t index)
{
	uint64_t value = seed * 0x9e3779b97f4a7c15ull + index + 1u;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
	return(value ^ (value >> 31));
}

__host__ __device__ static uint64_t CoherenceMix(uint64_t hash, uint64_t word)
{
	return((hash ^ word) * 0x100000001b3ull);
}

__global__ void CoherenceChecksum(const uint64_t *data, uint64_t *sums)
{
	__shared__ uint64_t lanes[COHERENCE_LANES];
	const uint64_t *chunk = data + (uint64_t)blockIdx.x * COHERENCE_CHUNK_WORDS;
	uint64_t hash = 0xcbf29ce484222325ull,index;
	for (index = threadIdx.x; index < COHERENCE_CHUNK_WORDS; index += COHERENCE_LANES)
		hash = CoherenceMix(hash,chunk[index]);
	lanes[threadIdx.x] = hash * (2u * threadIdx.x + 1u);
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		uint64_t total = 0u;
		for (index = 0u; index < COHERENCE_LANES; index++)
			total += lanes[index];
		sums[blockIdx.x] = total;
	}
}

__global__ void CoherenceWrite(uint64_t *data, uint64_t seed, uint64_t first, unsigned long long hold_ns)
{
	unsigned long long start,now;
	uint64_t index;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < hold_ns );
	for (index = first + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < first + COHERENCE_GRID_WORDS; index += (uint64_t)gridDim.x * blockDim.x)
		data[index] = CoherenceWord(seed,index);
}

static uint64_t CoherenceExpected(uint64_t seed, uint32_t chunk)
{
	uint64_t lanes[COHERENCE_LANES],total = 0u,index;
	uint32_t lane;
	for (lane = 0u; lane < COHERENCE_LANES; lane++)
		lanes[lane] = 0xcbf29ce484222325ull;
	for (index = 0u; index < COHERENCE_CHUNK_WORDS; index++)
		lanes[index % COHERENCE_LANES] = CoherenceMix(lanes[index % COHERENCE_LANES],CoherenceWord(seed,(uint64_t)chunk * COHERENCE_CHUNK_WORDS + index));
	for (lane = 0u; lane < COHERENCE_LANES; lane++)
		total += lanes[lane] * (2u * lane + 1u);
	return(total);
}

typedef struct CoherenceShared
{
	SparkDeviceMemoryHandle memory;
	SparkDeviceAddress address;
	SparkDeviceBuffer buffer;
	uint64_t bytes;
} CoherenceShared;

static SparkDeviceRange At(const SparkDeviceBuffer *buffer, uint64_t offset)
{
	SparkDeviceRange range;
	range.buffer = buffer;
	range.offset = offset;
	return(range);
}

static int SharedMap(CoherenceShared *shared, SparkDeviceMemoryHandle memory, uint64_t bytes, uint32_t access, const char *owner)
{
	memset(shared,0,sizeof(*shared));
	shared->memory = memory;
	shared->bytes = bytes;
	return(SparkDeviceAddressReserve(&shared->address,bytes,owner) == SPARK_STATUS_OK &&
		SparkDeviceMemoryMap(shared->address,bytes,memory,0u,access) == SPARK_STATUS_OK &&
		SparkDeviceAdopt(&shared->buffer,shared->address,0,bytes,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,owner) == SPARK_STATUS_OK);
}

static int SharedUnmap(CoherenceShared *shared)
{
	return(SparkDeviceFree(&shared->buffer) == SPARK_STATUS_OK && SparkDeviceMemoryUnmap(shared->address,shared->bytes) == SPARK_STATUS_OK &&
		SparkDeviceAddressRelease(shared->address,shared->bytes) == SPARK_STATUS_OK);
}

static int PatternBuffer(SparkDeviceBuffer *buffer, SparkDeviceSpace space, uint64_t bytes, uint64_t seed, const char *owner)
{
	uint64_t index;
	if ( SparkDeviceAllocate(buffer,space,bytes,owner) != SPARK_STATUS_OK )
		return(0);
	for (index = 0u; index < bytes / sizeof(uint64_t); index++)
		((uint64_t *)buffer->host)[index] = CoherenceWord(seed,index);
	return(1);
}

static uint64_t PatternMisses(const SparkDeviceBuffer *buffer, uint64_t bytes, uint64_t seed)
{
	uint64_t index,misses = 0u;
	for (index = 0u; index < bytes / sizeof(uint64_t); index++)
		misses += ((const uint64_t *)buffer->host)[index] != CoherenceWord(seed,index);
	return(misses);
}

static int Signal(int fd)
{
	const char byte = 1;
	return(write(fd,&byte,1u) == 1);
}

static int Await(int fd)
{
	char byte = 0;
	return(read(fd,&byte,1u) == 1);
}

static int ChildRun(int test, int mutant, int fd, int go, int done)
{
	SparkDeviceQueue *queue = 0;
	SparkDeviceMemoryHandle memory = 0u;
	CoherenceShared shared;
	SparkDeviceBuffer sums,host,source;
	void *view = (void *)1;
	uint32_t chunk,wrong = 0u;
	const uint32_t access = test == 1 ? SPARK_DEVICE_ACCESS_DEVICE_READ : SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"reader") != SPARK_STATUS_OK ||
		SparkDeviceMemoryImport(&memory,fd,"importer") != SPARK_STATUS_OK || close(fd) != 0 || SharedMap(&shared,memory,COHERENCE_BYTES,access,"importer") == 0 ||
		SparkDeviceHostOf(At(&shared.buffer,0u),COHERENCE_BYTES,&view) != SPARK_STATUS_UNSUPPORTED )
	{
		printf("FAIL the importing process sets up case %d: %s\n",test,SparkDeviceLastError()->text);
		return(COHERENCE_BROKEN);
	}
	if ( Signal(done) == 0 || Await(go) == 0 )
		return(COHERENCE_BROKEN);
	if ( test == 1 )
	{
		if ( SparkDeviceAllocate(&sums,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,COHERENCE_CHUNKS * sizeof(uint64_t),"sums") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_HOST_PINNED,COHERENCE_CHUNKS * sizeof(uint64_t),"sums back") != SPARK_STATUS_OK )
			return(COHERENCE_BROKEN);
		LM_LAUNCH((CoherenceChecksum),COHERENCE_CHUNKS,COHERENCE_LANES,0,queue,(const uint64_t *)shared.address,(uint64_t *)sums.device);
		if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS || SparkDeviceCopy(queue,At(&host,0u),At(&sums,0u),COHERENCE_CHUNKS * sizeof(uint64_t)) != SPARK_STATUS_OK ||
			SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			return(COHERENCE_BROKEN);
		for (chunk = 0u; chunk < COHERENCE_CHUNKS; chunk++)
			wrong += ((const uint64_t *)host.host)[chunk] != CoherenceExpected(1u,chunk);
		if ( SparkDeviceFree(&sums) != SPARK_STATUS_OK || SparkDeviceFree(&host) != SPARK_STATUS_OK )
			return(COHERENCE_BROKEN);
	}
	else if ( test == 2 )
	{
		uint32_t grid;
		for (grid = 0u; grid < COHERENCE_WRITE_GRIDS; grid++)
			LM_LAUNCH((CoherenceWrite),256,256,0,queue,(uint64_t *)shared.address,(uint64_t)2u,(uint64_t)grid * COHERENCE_GRID_WORDS,COHERENCE_HOLD_NS);
		if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS || (mutant != 0 && Signal(done) == 0) || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK ||
			(mutant == 0 && Signal(done) == 0) )
			return(COHERENCE_BROKEN);
	}
	else
	{
		if ( PatternBuffer(&source,SPARK_DEVICE_SPACE_HOST_PAGEABLE,COHERENCE_BYTES,3u,"kv source") == 0 ||
			SparkDeviceCopy(queue,At(&shared.buffer,0u),At(&source,0u),COHERENCE_BYTES) != SPARK_STATUS_OK || (mutant != 0 && Signal(done) == 0) ||
			SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK || (mutant == 0 && Signal(done) == 0) || SparkDeviceFree(&source) != SPARK_STATUS_OK )
			return(COHERENCE_BROKEN);
	}
	if ( Await(go) == 0 || SharedUnmap(&shared) == 0 || SparkDeviceMemoryRelease(memory) != SPARK_STATUS_OK ||
		SparkDeviceQueueDestroy(queue) != SPARK_STATUS_OK || SparkNvrmGpuRelease() != SPARK_GPU_SUCCESS || SparkDeviceShutdown() != SPARK_STATUS_OK )
		return(COHERENCE_BROKEN);
	return(wrong == 0u ? COHERENCE_SEEN : COHERENCE_MISMATCH);
}

static int Spawn(const char *self, int test, int mutant, int fd, int go[2], int done[2], pid_t *child)
{
	char arguments[5][16];
	snprintf(arguments[0],sizeof(arguments[0]),"%d",test);
	snprintf(arguments[1],sizeof(arguments[1]),"%d",mutant);
	snprintf(arguments[2],sizeof(arguments[2]),"%d",fd);
	snprintf(arguments[3],sizeof(arguments[3]),"%d",go[0]);
	snprintf(arguments[4],sizeof(arguments[4]),"%d",done[1]);
	*child = fork();
	if ( *child == 0 )
	{
		execl(self,self,"--child",arguments[0],arguments[1],arguments[2],arguments[3],arguments[4],(char *)0);
		_exit(127);
	}
	return(*child > 0);
}

static int Reap(pid_t child)
{
	int status = 0;
	if ( waitpid(child,&status,0) != child || !WIFEXITED(status) )
		return(COHERENCE_BROKEN);
	return(WEXITSTATUS(status));
}

static int CaseRun(const char *self, SparkDeviceQueue *queue, int test, int mutant)
{
	SparkDeviceMemoryHandle memory = 0u;
	CoherenceShared shared;
	SparkDeviceBuffer source,back;
	void *view = (void *)1;
	int go[2] = {-1, -1},done[2] = {-1, -1},fd = -1,inherited = -1,outcome = COHERENCE_BROKEN,reaped;
	uint64_t misses = 0u;
	pid_t child = 0;
	if ( SparkDeviceMemoryCreate(&memory,COHERENCE_BYTES,"exporter") != SPARK_STATUS_OK ||
		SharedMap(&shared,memory,COHERENCE_BYTES,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,"exporter") == 0 ||
		SparkDeviceFill(queue,At(&shared.buffer,0u),0u,COHERENCE_BYTES) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK ||
		SparkDeviceMemoryExport(memory,&fd) != SPARK_STATUS_OK || pipe(go) != 0 || pipe(done) != 0 || (inherited = dup(fd)) < 0 )
		return(COHERENCE_BROKEN);
	CHECK(SparkDeviceHostOf(At(&shared.buffer,0u),COHERENCE_BYTES,&view) == SPARK_STATUS_UNSUPPORTED,"an adopted range of shared memory has no host view");
	if ( test != 1 && SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PINNED,COHERENCE_BYTES,"read back") != SPARK_STATUS_OK )
		return(COHERENCE_BROKEN);
	if ( Spawn(self,test,mutant,inherited,go,done,&child) == 0 || close(inherited) != 0 || Await(done[0]) == 0 )
		return(COHERENCE_BROKEN);
	if ( test == 1 )
	{
		if ( PatternBuffer(&source,SPARK_DEVICE_SPACE_HOST_PAGEABLE,COHERENCE_BYTES,1u,"pattern") == 0 ||
			SparkDeviceCopy(queue,At(&shared.buffer,0u),At(&source,0u),COHERENCE_BYTES) != SPARK_STATUS_OK || (mutant != 0 && Signal(go[1]) == 0) ||
			SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK || (mutant == 0 && Signal(go[1]) == 0) || SparkDeviceFree(&source) != SPARK_STATUS_OK ||
			Signal(go[1]) == 0 )
			return(COHERENCE_BROKEN);
		outcome = Reap(child);
	}
	else
	{
		if ( Signal(go[1]) == 0 || Await(done[0]) == 0 || SparkDeviceCopy(queue,At(&back,0u),At(&shared.buffer,0u),COHERENCE_BYTES) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			return(COHERENCE_BROKEN);
		misses = PatternMisses(&back,COHERENCE_BYTES,(uint64_t)test);
		if ( SparkDeviceFree(&back) != SPARK_STATUS_OK || Signal(go[1]) == 0 )
			return(COHERENCE_BROKEN);
		reaped = Reap(child);
		outcome = reaped != COHERENCE_SEEN ? COHERENCE_BROKEN : misses == 0u ? COHERENCE_SEEN : COHERENCE_MISMATCH;
	}
	printf("case %d %s: %s",test,mutant != 0 ? "mutant" : "exact",outcome == COHERENCE_SEEN ? "every byte seen" : outcome == COHERENCE_MISMATCH ? "mismatch" : "broken");
	if ( test != 1 && outcome != COHERENCE_BROKEN )
		printf(" (%llu of %llu words differ)",(unsigned long long)misses,(unsigned long long)COHERENCE_WORDS);
	printf("\n");
	if ( SharedUnmap(&shared) == 0 || SparkDeviceMemoryRelease(memory) != SPARK_STATUS_OK || close(fd) != 0 || close(go[0]) != 0 || close(go[1]) != 0 ||
		close(done[0]) != 0 || close(done[1]) != 0 )
		return(COHERENCE_BROKEN);
	return(outcome);
}

static uint64_t MemoryFree(void)
{
	char line[128];
	unsigned long long kilobytes = 0u;
	FILE *meminfo = fopen("/proc/meminfo","r");
	if ( meminfo == 0 )
		return(0u);
	while ( fgets(line,sizeof(line),meminfo) != 0 )
		if ( sscanf(line,"MemFree: %llu kB",&kilobytes) == 1 )
			break;
	fclose(meminfo);
	return((uint64_t)kilobytes * 1024u);
}

static uint32_t OpenDescriptors(void)
{
	DIR *directory = opendir("/proc/self/fd");
	uint32_t count = 0u;
	if ( directory == 0 )
		return(0u);
	while ( readdir(directory) != 0 )
		count++;
	closedir(directory);
	return(count);
}

static int Round(SparkDeviceQueue *queue, const SparkDeviceBuffer *probe, SparkDeviceBuffer *back, SparkDeviceAddress *first)
{
	SparkDeviceMemoryHandle memory = 0u,imported = 0u;
	CoherenceShared mapped,again;
	int fd = -1;
	int same;
	if ( SparkDeviceMemoryCreate(&memory,COHERENCE_ROUND_BYTES,"round") != SPARK_STATUS_OK ||
		SharedMap(&mapped,memory,COHERENCE_ROUND_BYTES,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,"round") == 0 ||
		SparkDeviceCopy(queue,At(&mapped.buffer,0u),At(probe,0u),COHERENCE_ROUND_PROBE) != SPARK_STATUS_OK ||
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK || SparkDeviceMemoryExport(memory,&fd) != SPARK_STATUS_OK ||
		SparkDeviceMemoryRelease(memory) != SPARK_STATUS_OK || SparkDeviceMemoryImport(&imported,fd,"round import") != SPARK_STATUS_OK ||
		SharedMap(&again,imported,COHERENCE_ROUND_BYTES,SPARK_DEVICE_ACCESS_DEVICE_READ,"round import") == 0 ||
		SparkDeviceFill(queue,At(back,0u),0u,COHERENCE_ROUND_PROBE) != SPARK_STATUS_OK ||
		SparkDeviceCopy(queue,At(back,0u),At(&again.buffer,0u),COHERENCE_ROUND_PROBE) != SPARK_STATUS_OK ||
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		return(0);
	same = memcmp(back->host,probe->host,COHERENCE_ROUND_PROBE) == 0;
	if ( *first == 0u )
		*first = mapped.address;
	same = same && mapped.address == *first;
	return(SharedUnmap(&again) && SharedUnmap(&mapped) && SparkDeviceMemoryRelease(imported) == SPARK_STATUS_OK && close(fd) == 0 && same);
}

static void TestRounds(SparkDeviceQueue *queue)
{
	SparkDeviceBuffer probe,back;
	SparkDeviceAddress first = 0u;
	uint64_t before,after;
	uint32_t descriptors,round,good = 0u;
	CHECK(PatternBuffer(&probe,SPARK_DEVICE_SPACE_HOST_PINNED,COHERENCE_ROUND_PROBE,5u,"probe") &&
		SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PINNED,COHERENCE_ROUND_PROBE,"probe back") == SPARK_STATUS_OK,"allocate the round probes");
	descriptors = OpenDescriptors();
	before = MemoryFree();
	for (round = 0u; round < COHERENCE_ROUNDS; round++)
		good += (uint32_t)Round(queue,&probe,&back,&first);
	after = MemoryFree();
	printf("rounds: %u of %u kept their bytes after release while mapped; MemFree %llu -> %llu MiB; descriptors %u -> %u\n",good,COHERENCE_ROUNDS,
		(unsigned long long)(before >> 20),(unsigned long long)(after >> 20),descriptors,OpenDescriptors());
	CHECK(good == COHERENCE_ROUNDS,"every round's memory keeps its bytes after its handle is released while mapped, an import of it sees them, "
		"and each round reserves the same address again");
	CHECK(after + COHERENCE_LEAK_SLACK >= before && OpenDescriptors() == descriptors,
		"a thousand rounds of create, map, export, release, import, map, unmap and release leave memory and descriptors where they were");
	CHECK(SparkDeviceFree(&probe) == SPARK_STATUS_OK && SparkDeviceFree(&back) == SPARK_STATUS_OK,"free the round probes");
}

int main(int argc, char **argv)
{
	static const char *names[] = {"",
		"a copy-engine fill from pageable memory, completed before the signal, is what a kernel in another process reads",
		"a kernel's writes in another process, completed before the signal, are what a copy-engine read sees",
		"a copy-engine write in another process through an adopted range, waited on before the signal, is what a copy-engine read sees"};
	SparkDeviceQueue *queue = 0;
	int test;
	if ( argc == 7 && strcmp(argv[1],"--child") == 0 )
		return(ChildRun(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6])));
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_shared_coherence (no open kernel module on this host)\n");
		return(0);
	}
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"exporter") != SPARK_STATUS_OK )
	{
		printf("FAIL set up the device: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for (test = 1; test <= 3; test++)
	{
		const int exact = CaseRun(argv[0],queue,test,0),mutant = CaseRun(argv[0],queue,test,1);
		CHECK(exact == COHERENCE_SEEN,names[test]);
		CHECK(mutant == COHERENCE_MISMATCH,"the same case with its completion point removed is caught as a mismatch");
	}
	TestRounds(queue);
	CHECK(SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK && SparkNvrmGpuRelease() == SPARK_GPU_SUCCESS && SparkDeviceShutdown() == SPARK_STATUS_OK,
		"release the queue and shut the device down");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS shared memory across processes: each completion point makes every byte visible, each mutant without it is caught, "
		"no shared range has a host view, and a thousand rounds leak nothing\n");
	return(0);
}
