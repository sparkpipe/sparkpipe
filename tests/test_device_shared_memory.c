#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"

static int failures;

#define CHECK(condition, what) do { if ( !(condition) ) { const SparkDeviceError *error = SparkDeviceLastError(); \
	printf("FAIL %s:%d %s (last error %s: %s)\n", __FILE__, __LINE__, what, error->operation, error->text); failures++; } } while ( 0 )

typedef struct Shared
{
	SparkDeviceAddress address;
	uint64_t bytes;
	SparkDeviceBuffer buffer;
	uint32_t mapped;
	uint32_t adopted;
} Shared;

static uint8_t Pattern(uint64_t seed, uint64_t index)
{
	return((uint8_t)((index * 131u + seed * 7919u) ^ (index >> 11)));
}

static int Reserve(Shared *shared, uint64_t bytes, const char *owner)
{
	memset(shared,0,sizeof(*shared));
	shared->bytes = bytes;
	return(SparkDeviceAddressReserve(&shared->address,bytes,owner) == SPARK_STATUS_OK && shared->address != 0u);
}

static int Adopt(Shared *shared, const char *owner)
{
	shared->adopted = SparkDeviceAdopt(&shared->buffer,shared->address,0,shared->bytes,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,owner) == SPARK_STATUS_OK;
	return((int)shared->adopted);
}

static int Map(Shared *shared, SparkDeviceMemoryHandle memory, uint64_t bytes, uint32_t access, const char *owner)
{
	if ( Reserve(shared,bytes,owner) == 0 )
		return(0);
	shared->mapped = SparkDeviceMemoryMap(shared->address,bytes,memory,0u,access) == SPARK_STATUS_OK;
	return(shared->mapped != 0u && Adopt(shared,owner));
}

static int Unmap(Shared *shared)
{
	int done = 1;
	if ( shared->adopted != 0u )
		done = SparkDeviceFree(&shared->buffer) == SPARK_STATUS_OK && done;
	if ( shared->mapped != 0u )
		done = SparkDeviceMemoryUnmap(shared->address,shared->bytes) == SPARK_STATUS_OK && done;
	if ( shared->address != 0u )
		done = SparkDeviceAddressRelease(shared->address,shared->bytes) == SPARK_STATUS_OK && done;
	memset(shared,0,sizeof(*shared));
	return(done);
}

static int Fill(const Shared *shared, uint64_t seed)
{
	SparkDeviceBuffer host;
	SparkDeviceRange to, from;
	uint64_t index;
	int done;
	if ( SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_HOST_PAGEABLE,shared->bytes,"shared fill") != SPARK_STATUS_OK )
		return(0);
	for (index=0u; index<shared->bytes; index++)
		((uint8_t *)host.host)[index] = Pattern(seed,index);
	to.buffer = &shared->buffer;
	to.offset = 0u;
	from.buffer = &host;
	from.offset = 0u;
	done = SparkDeviceCopySync(to,from,shared->bytes) == SPARK_STATUS_OK;
	return(SparkDeviceFree(&host) == SPARK_STATUS_OK && done);
}

static int Matches(const Shared *shared, uint64_t seed)
{
	SparkDeviceBuffer host;
	SparkDeviceRange to, from;
	uint64_t index;
	int same;
	if ( SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_HOST_PAGEABLE,shared->bytes,"shared read") != SPARK_STATUS_OK )
		return(0);
	to.buffer = &host;
	to.offset = 0u;
	from.buffer = &shared->buffer;
	from.offset = 0u;
	same = SparkDeviceCopySync(to,from,shared->bytes) == SPARK_STATUS_OK;
	for (index=0u; same != 0 && index<shared->bytes; index++)
		same = ((const uint8_t *)host.host)[index] == Pattern(seed,index);
	return(SparkDeviceFree(&host) == SPARK_STATUS_OK && same);
}

static int Child(int fd, uint64_t bytes)
{
	SparkDeviceMemoryHandle memory = 0u;
	Shared shared;
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK,"the importing process initializes its device");
	CHECK(SparkDeviceMemoryImport(&memory,fd,"importer") == SPARK_STATUS_OK && memory != 0u,"the importing process imports the exported memory");
	CHECK(close(fd) == 0,"the importer closes the descriptor once it holds the memory");
	CHECK(Map(&shared,memory,bytes,SPARK_DEVICE_ACCESS_DEVICE_READ,"importer read") && Matches(&shared,1u),
		"a read-only mapping in another process sees every byte the exporter copied in");
	CHECK(Unmap(&shared),"the importer unmaps the read-only mapping");
	CHECK(Map(&shared,memory,bytes,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,"importer write") && Fill(&shared,2u),
		"a read-write mapping in another process takes a copy-engine fill");
	CHECK(Unmap(&shared) && SparkDeviceMemoryRelease(memory) == SPARK_STATUS_OK,"the importer unmaps and releases its handle");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK,"the importer shuts its device down");
	return(failures == 0 ? 0 : 1);
}

static int RunChild(const char *self, int fd, uint64_t bytes)
{
	char descriptor[16], size[24];
	int status = 0;
	pid_t child;
	snprintf(descriptor,sizeof(descriptor),"%d",fd);
	snprintf(size,sizeof(size),"%llu",(unsigned long long)bytes);
	child = fork();
	if ( child == 0 )
	{
		execl(self,self,"--import",descriptor,size,(char *)0);
		_exit(127);
	}
	return(child > 0 && waitpid(child,&status,0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

static void TestRefusalsBeforeInitialize(void)
{
	SparkDeviceMemoryHandle memory = 7u;
	CHECK(SparkDeviceMemoryCreate(&memory,4096u,"early") == SPARK_STATUS_INVALID_ARGUMENT && memory == 0u,
		"shared memory needs an initialized device, and a refusal clears the output");
}

static void TestArgumentRefusals(uint64_t granularity)
{
	SparkDeviceMemoryHandle memory = 0u;
	SparkDeviceAddress address = 0u;
	uint64_t bytes = 0u;
	int fd = 3;
	CHECK(SparkDeviceMemoryGranularity(2u,&bytes) == SPARK_STATUS_INVALID_ARGUMENT,"only the minimum and recommended granularities exist");
	CHECK(SparkDeviceMemoryCreate(&memory,0u,"zero") == SPARK_STATUS_INVALID_ARGUMENT,"memory needs a size");
	CHECK(SparkDeviceMemoryRelease(0u) == SPARK_STATUS_INVALID_ARGUMENT,"release needs a handle");
	CHECK(SparkDeviceMemoryExport(0u,&fd) == SPARK_STATUS_INVALID_ARGUMENT && fd == -1,"export needs a handle and clears the output");
	CHECK(SparkDeviceMemoryImport(&memory,-1,"no fd") == SPARK_STATUS_INVALID_ARGUMENT,"import needs a descriptor");
	CHECK(SparkDeviceAddressReserve(&address,0u,"zero") == SPARK_STATUS_INVALID_ARGUMENT,"a reservation needs a size");
	CHECK(SparkDeviceMemoryMap(4096u,granularity,9u,0u,0u) == SPARK_STATUS_INVALID_ARGUMENT,"a mapping needs device read access");
	CHECK(SparkDeviceMemoryMap(4096u,granularity,9u,0u,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_HOST_READ) == SPARK_STATUS_INVALID_ARGUMENT,
		"a mapping takes device access only; shared memory has no host view");
	CHECK(SparkDeviceMemoryUnmap(0u,granularity) == SPARK_STATUS_INVALID_ARGUMENT,"unmap needs a range");
	CHECK(SparkDeviceAddressRelease(0u,granularity) == SPARK_STATUS_INVALID_ARGUMENT,"release of a range needs the range");
}

static void TestExportImport(const char *self, uint64_t bytes)
{
	SparkDeviceMemoryHandle memory = 0u;
	Shared shared;
	void *host = (void *)1;
	int fd = -1, inheritable;
	CHECK(SparkDeviceMemoryCreate(&memory,bytes,"exporter") == SPARK_STATUS_OK && memory != 0u,"create exportable device memory");
	CHECK(Map(&shared,memory,bytes,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,"exporter") && Fill(&shared,1u) &&
		Matches(&shared,1u),"map it read-write, fill it by a copy and read the same bytes back");
	CHECK(SparkDeviceHostOf((SparkDeviceRange){&shared.buffer,0u},bytes,&host) == SPARK_STATUS_UNSUPPORTED,
		"an adopted range of shared memory has no host view");
	CHECK(SparkDeviceMemoryExport(memory,&fd) == SPARK_STATUS_OK && fd >= 0,"export the memory as a file descriptor");
	CHECK(SparkDeviceMemoryRelease(memory) == SPARK_STATUS_OK && Matches(&shared,1u),
		"releasing the handle while it is mapped keeps the memory and its bytes");
	inheritable = fd >= 0 ? dup(fd) : -1;
	CHECK(inheritable >= 0 && RunChild(self,inheritable,bytes),"another process imports, reads, writes and releases the memory");
	if ( inheritable >= 0 )
		(void)close(inheritable);
	CHECK(Matches(&shared,2u),"the exporter's mapping sees every byte the other process copied in");
	CHECK(Unmap(&shared),"unmap and release the exporter's range");
	if ( fd >= 0 )
		(void)close(fd);
}

static void TestChunkedMapping(uint64_t granularity)
{
	const uint32_t access = SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE;
	SparkDeviceMemoryHandle first = 0u, second = 0u;
	Shared shared;
	int mapped;
	CHECK(SparkDeviceMemoryCreate(&first,granularity,"chunk one") == SPARK_STATUS_OK &&
		SparkDeviceMemoryCreate(&second,granularity,"chunk two") == SPARK_STATUS_OK && Reserve(&shared,2u * granularity,"chunks"),
		"create two one-chunk allocations and one reservation for both");
	mapped = SparkDeviceMemoryMap(shared.address,granularity,second,0u,access) == SPARK_STATUS_OK &&
		SparkDeviceMemoryMap(shared.address + granularity,granularity,first,0u,access) == SPARK_STATUS_OK;
	CHECK(mapped,"map the second allocation first in the range and the first one after it");
	if ( mapped != 0 )
	{
		shared.mapped = 1u;
		CHECK(Adopt(&shared,"chunks") && Fill(&shared,3u) && Matches(&shared,3u),"one adopted range spans both mappings for copies");
		CHECK(SparkDeviceMemoryUnmap(shared.address,granularity / 2u) != SPARK_STATUS_OK,"part of a mapping cannot be unmapped");
	}
	CHECK(SparkDeviceMemoryRelease(first) == SPARK_STATUS_OK && SparkDeviceMemoryRelease(second) == SPARK_STATUS_OK && Unmap(&shared),
		"release both handles, then one unmap covers both whole mappings");
}

static void *ThreadBind(void *argument)
{
	const SparkDeviceInstance instance = *(const SparkDeviceInstance *)argument;
	SparkDeviceInstance seen = 0u;
	CHECK(SparkDeviceThreadBind(instance + 1u) == SPARK_STATUS_INVALID_ARGUMENT,"a value the device layer did not return does not bind");
	CHECK(SparkDeviceThreadBind(instance) == SPARK_STATUS_OK && SparkDeviceThreadInstance(&seen) == SPARK_STATUS_OK && seen == instance,
		"a new thread binds to the creating thread's instance");
	return(0);
}

static void TestThreadBinding(void)
{
	SparkDeviceInstance instance = 0u;
	pthread_t thread;
	CHECK(SparkDeviceThreadInstance(&instance) == SPARK_STATUS_OK && instance != 0u,"the initializing thread is bound to the device instance");
	CHECK(pthread_create(&thread,0,ThreadBind,&instance) == 0 && pthread_join(thread,0) == 0,"bind a second thread");
}

int main(int argc, char **argv)
{
	uint64_t minimum = 0u, recommended = 0u;
	char backend[16] = "none";
	if ( argc == 4 && strcmp(argv[1],"--import") == 0 )
		return(Child(atoi(argv[2]),strtoull(argv[3],0,10)));
	TestRefusalsBeforeInitialize();
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK,"initialize the device");
	CHECK(SparkDeviceMemoryGranularity(SPARK_DEVICE_GRANULARITY_MINIMUM,&minimum) == SPARK_STATUS_OK &&
		SparkDeviceMemoryGranularity(SPARK_DEVICE_GRANULARITY_RECOMMENDED,&recommended) == SPARK_STATUS_OK && minimum != 0u && recommended >= minimum,
		"the minimum and recommended granularities are powers of two, the recommended one no smaller");
	if ( minimum != 0u && recommended >= minimum )
	{
		TestArgumentRefusals(minimum);
		TestExportImport(argv[0],4u * recommended);
		TestChunkedMapping(recommended);
	}
	TestThreadBinding();
	if ( SparkDeviceGetProperties() != 0 )
		snprintf(backend,sizeof(backend),"%s",SparkDeviceGetProperties()->backend);
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK,"shut the device down");
	if ( failures == 0 )
		printf("PASS device shared memory (%s backend)\n",backend);
	return(failures == 0 ? 0 : 1);
}
