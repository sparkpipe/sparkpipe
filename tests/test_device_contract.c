#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"

static int failures;

#define CHECK(condition, what) do { if ( !(condition) ) { const SparkDeviceError *last = SparkDeviceLastError(); \
	printf("FAIL %s:%d %s (last error: %s %s)\n", __FILE__, __LINE__, what, last->operation, last->text); failures++; } } while ( 0 )

typedef struct Gate
{
	pthread_mutex_t mutex;
	pthread_cond_t changed;
	int open;
} Gate;

static void GateInit(Gate *gate)
{
	pthread_mutex_init(&gate->mutex,0);
	pthread_cond_init(&gate->changed,0);
	gate->open = 0;
}

static void GateOpen(Gate *gate)
{
	pthread_mutex_lock(&gate->mutex);
	gate->open = 1;
	pthread_cond_broadcast(&gate->changed);
	pthread_mutex_unlock(&gate->mutex);
}

static void GateWaitCallback(void *context)
{
	Gate *gate = (Gate *)context;
	pthread_mutex_lock(&gate->mutex);
	while ( gate->open == 0 )
		pthread_cond_wait(&gate->changed,&gate->mutex);
	pthread_mutex_unlock(&gate->mutex);
}

static void FlagCallback(void *context)
{
	__atomic_store_n((int *)context,1,__ATOMIC_SEQ_CST);
}

static void SleepCallback(void *context)
{
	struct timespec pause = {0, (long)(uintptr_t)context};
	nanosleep(&pause,0);
}

static void PauseMs(long milliseconds)
{
	struct timespec pause = {0, milliseconds * 1000000L};
	nanosleep(&pause,0);
}

static SparkDeviceRange At(const SparkDeviceBuffer *buffer, uint64_t offset)
{
	SparkDeviceRange range;
	range.buffer = buffer;
	range.offset = offset;
	return(range);
}

static void Pattern(uint8_t *bytes, uint64_t count, uint32_t seed)
{
	uint64_t index;
	for ( index = 0u; index < count; index++ )
		bytes[index] = (uint8_t)((index * 131u + seed * 7u + (index >> 8)) & 0xffu);
}

static int HostBackend(void)
{
	return(strcmp(SparkDeviceGetProperties()->backend,"host") == 0);
}

static int ChildFaults(SparkDeviceAddress address)
{
	pid_t child = fork();
	int status = 0;
	if ( child == 0 )
	{
		volatile uint8_t value = *(volatile uint8_t *)(uintptr_t)address;
		(void)value;
		_exit(0);
	}
	if ( child < 0 || waitpid(child,&status,0) != child )
		return(0);
	return(WIFSIGNALED(status) && (WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS));
}

static void TestSpaces(void)
{
	const SparkDeviceProperties *properties = SparkDeviceGetProperties();
	SparkDeviceBuffer device, shared, pinned, pageable, invalid;
	SparkDeviceAddress address;
	void *host;
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,1000u,"contract private") == SPARK_STATUS_OK,"allocate device-private");
	CHECK(device.host == 0 && device.device != 0u && device.bytes == 1000u,"device-private memory has a device address and no host view");
	CHECK(SparkDeviceAllocate(&shared,SPARK_DEVICE_SPACE_SHARED_COHERENT,1000u,"contract shared") == SPARK_STATUS_OK,"allocate shared");
	CHECK(shared.host != 0 && shared.device != 0u,"shared memory has both views");
	CHECK(SparkDeviceAllocate(&pinned,SPARK_DEVICE_SPACE_HOST_PINNED,1000u,"contract pinned") == SPARK_STATUS_OK,"allocate pinned");
	CHECK(pinned.host != 0 && pinned.device != 0u,"pinned memory has both views");
	if ( (properties->properties & SPARK_DEVICE_PROPERTY_SHARED_ADDRESS) == 0u )
		CHECK((uint64_t)(uintptr_t)shared.host != shared.device,"without a shared address space the host and device views differ");
	CHECK(SparkDeviceAllocate(&pageable,SPARK_DEVICE_SPACE_HOST_PAGEABLE,1000u,"contract pageable") == SPARK_STATUS_OK,"allocate pageable");
	CHECK(pageable.host != 0 && pageable.device == 0u,"pageable memory is host only");
	CHECK(SparkDeviceAddressOf(At(&pageable,0u),1u,&address) == SPARK_STATUS_UNSUPPORTED,"pageable memory has no device address");
	CHECK(SparkDeviceHostOf(At(&device,0u),1u,&host) == SPARK_STATUS_UNSUPPORTED,"device-private memory has no host view");
	CHECK(SparkDeviceAddressOf(At(&device,10u),20u,&address) == SPARK_STATUS_OK && address == device.device + 10u,"device address of an offset");
	CHECK(SparkDeviceHostOf(At(&shared,10u),20u,&host) == SPARK_STATUS_OK && host == (uint8_t *)shared.host + 10,"host view of an offset");
	CHECK(SparkDeviceAllocate(&invalid,SPARK_DEVICE_SPACE_FILE_BACKED,1000u,"contract invalid") == SPARK_STATUS_INVALID_ARGUMENT,"file-backed memory comes only from a mapped file");
	CHECK(SparkDeviceAllocate(&invalid,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,0u,"contract invalid") == SPARK_STATUS_INVALID_ARGUMENT,"a zero-byte allocation is refused");
	if ( HostBackend() )
	{
		CHECK(ChildFaults(device.device),"a host dereference of a device-private address faults");
		CHECK(ChildFaults(shared.device),"a host dereference of a shared buffer's device address faults");
	}
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free every space: device");
	CHECK(SparkDeviceFree(&shared) == SPARK_STATUS_OK,"free every space: shared");
	CHECK(SparkDeviceFree(&pinned) == SPARK_STATUS_OK,"free every space: pinned");
	CHECK(SparkDeviceFree(&pageable) == SPARK_STATUS_OK,"free every space: pageable");
	CHECK(device.handle == 0u && device.device == 0u,"free clears the buffer");
}

static void TestCopies(SparkDeviceQueue *queue)
{
	enum { BYTES = 70001 };
	SparkDeviceBuffer source, device, shared, back, file_like;
	SparkDeviceBuffer stale;
	uint8_t expected[BYTES];
	CHECK(SparkDeviceAllocate(&source,SPARK_DEVICE_SPACE_HOST_PAGEABLE,BYTES,"copy source") == SPARK_STATUS_OK,"allocate source");
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,BYTES,"copy device") == SPARK_STATUS_OK,"allocate device");
	CHECK(SparkDeviceAllocate(&shared,SPARK_DEVICE_SPACE_SHARED_COHERENT,BYTES + 7u,"copy shared") == SPARK_STATUS_OK,"allocate shared");
	CHECK(SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PAGEABLE,BYTES,"copy back") == SPARK_STATUS_OK,"allocate back");
	Pattern((uint8_t *)source.host,BYTES,1u);
	memcpy(expected,source.host,BYTES);
	CHECK(SparkDeviceCopy(queue,At(&device,0u),At(&source,0u),BYTES) == SPARK_STATUS_OK,"copy host to device");
	CHECK(SparkDeviceCopy(queue,At(&shared,7u),At(&device,0u),BYTES) == SPARK_STATUS_OK,"copy device to shared at an offset");
	CHECK(SparkDeviceCopy(queue,At(&back,0u),At(&shared,7u),BYTES) == SPARK_STATUS_OK,"copy shared to host");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for the copies");
	CHECK(memcmp(back.host,expected,BYTES) == 0,"bytes survive host, device, shared and host again");
	CHECK(SparkDeviceCopy(queue,At(&back,100u),At(&device,BYTES - 50u),100u) == SPARK_STATUS_INVALID_ARGUMENT,"a source range past its buffer is refused");
	CHECK(strcmp(SparkDeviceLastError()->operation,"copy") == 0,"the refusal names the operation");
	CHECK(SparkDeviceCopy(queue,At(&device,0u),At(&device,10u),100u) == SPARK_STATUS_INVALID_ARGUMENT,"overlapping ranges in one buffer are refused");
	CHECK(SparkDeviceCopy(queue,At(&device,0u),At(&device,100u),100u) == SPARK_STATUS_OK,"disjoint ranges in one buffer are accepted");
	CHECK(SparkDeviceCopy(0,At(&device,0u),At(&source,0u),1u) == SPARK_STATUS_INVALID_ARGUMENT,"a missing queue is refused");
	CHECK(SparkDeviceFill(queue,At(&device,0u),0x5au,BYTES) == SPARK_STATUS_OK,"fill device memory");
	CHECK(SparkDeviceCopy(queue,At(&back,0u),At(&device,0u),BYTES) == SPARK_STATUS_OK,"copy the fill back");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for the fill");
	memset(expected,0x5a,BYTES);
	CHECK(memcmp(back.host,expected,BYTES) == 0,"the fill reaches every byte");
	stale = back;
	CHECK(SparkDeviceFree(&back) == SPARK_STATUS_OK,"free back");
	CHECK(SparkDeviceCopy(queue,At(&stale,0u),At(&device,0u),1u) == SPARK_STATUS_INVALID_ARGUMENT,"a freed buffer is refused");
	file_like = source;
	file_like.handle ^= 1ull << 40;
	CHECK(SparkDeviceCopy(queue,At(&file_like,0u),At(&device,0u),1u) == SPARK_STATUS_INVALID_ARGUMENT,"a forged handle is refused");
	CHECK(SparkDeviceFree(&source) == SPARK_STATUS_OK,"free the copy buffers: source");
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the copy buffers: device");
	CHECK(SparkDeviceFree(&shared) == SPARK_STATUS_OK,"free the copy buffers: shared");
}

static void TestFreshUpload(SparkDeviceQueue *queue)
{
	enum { TAIL = 4096, ROUNDS = 4 };
	const uint64_t bytes = UINT64_C(1) << 30;
	SparkDeviceBuffer source, back;
	CHECK(SparkDeviceAllocate(&source,SPARK_DEVICE_SPACE_HOST_PAGEABLE,TAIL,"fresh source") == SPARK_STATUS_OK,"allocate the fresh-upload source");
	CHECK(SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PAGEABLE,TAIL,"fresh back") == SPARK_STATUS_OK,"allocate the fresh-upload readback");
	Pattern((uint8_t *)source.host,TAIL,3u);
	for ( uint32_t round = 0u; round < ROUNDS; ++round )
	{
		SparkDeviceBuffer device;
		CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"fresh device") == SPARK_STATUS_OK,"allocate a fresh 1 GiB device buffer");
		CHECK(SparkDeviceCopy(queue,At(&device,bytes - TAIL),At(&source,0u),TAIL) == SPARK_STATUS_OK,"upload into its last bytes at once");
		CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for the upload");
		CHECK(SparkDeviceSynchronize(SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for all device work, the allocation's own included");
		CHECK(SparkDeviceCopy(queue,At(&back,0u),At(&device,bytes - TAIL),TAIL) == SPARK_STATUS_OK,"read the last bytes back");
		CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for the readback");
		CHECK(memcmp(back.host,source.host,TAIL) == 0,"an upload right after allocation keeps its bytes: the allocation's zero fill finished before the buffer was returned");
		CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the fresh device buffer");
	}
	CHECK(SparkDeviceFree(&source) == SPARK_STATUS_OK,"free the fresh-upload source");
	CHECK(SparkDeviceFree(&back) == SPARK_STATUS_OK,"free the fresh-upload readback");
}

static void TestOrdering(void)
{
	enum { BYTES = 4096, ROUNDS = 64 };
	SparkDeviceQueue *producer = 0, *consumer = 0;
	SparkDeviceEvent *ready = 0;
	SparkDeviceBuffer device, result;
	uint32_t round;
	int ordered = 1;
	CHECK(SparkDeviceQueueCreate(&producer,"producer") == SPARK_STATUS_OK && SparkDeviceQueueCreate(&consumer,"consumer") == SPARK_STATUS_OK,"create two queues");
	CHECK(SparkDeviceEventCreate(&ready) == SPARK_STATUS_OK,"create an event");
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,BYTES,"ordering device") == SPARK_STATUS_OK,"allocate the shared target");
	CHECK(SparkDeviceAllocate(&result,SPARK_DEVICE_SPACE_HOST_PAGEABLE,BYTES,"ordering result") == SPARK_STATUS_OK,"allocate the result");
	for ( round = 1u; round <= ROUNDS; round++ )
	{
		Gate gate;
		uint32_t index;
		GateInit(&gate);
		SparkDeviceQueueHostCallback(producer,GateWaitCallback,&gate);
		SparkDeviceFill(producer,At(&device,0u),(uint8_t)round,BYTES);
		SparkDeviceEventRecord(ready,producer);
		SparkDeviceQueueWaitEvent(consumer,ready);
		SparkDeviceCopy(consumer,At(&result,0u),At(&device,0u),BYTES);
		if ( round % 8u == 1u )
			PauseMs(2);
		GateOpen(&gate);
		SparkDeviceQueueWait(consumer,SPARK_DEVICE_WAIT_FOREVER);
		for ( index = 0u; index < BYTES; index++ )
			if ( ((uint8_t *)result.host)[index] != (uint8_t)round )
				ordered = 0;
	}
	CHECK(ordered,"a queue waiting on an event sees everything before the event on the other queue");
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the ordering buffers: device");
	CHECK(SparkDeviceFree(&result) == SPARK_STATUS_OK,"free the ordering buffers: result");
	CHECK(SparkDeviceEventDestroy(ready) == SPARK_STATUS_OK,"destroy the event");
	CHECK(SparkDeviceQueueDestroy(producer) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(consumer) == SPARK_STATUS_OK,"destroy the queues");
}

static void TestBusy(SparkDeviceQueue *queue)
{
	SparkDeviceBuffer device, host;
	SparkDeviceEvent *event = 0;
	Gate gate;
	int flag = 0;
	GateInit(&gate);
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,256u,"busy device") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_HOST_PAGEABLE,256u,"busy host") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceEventCreate(&event) == SPARK_STATUS_OK,"create event");
	CHECK(SparkDeviceEventQuery(event) == SPARK_STATUS_OK,"an event that was never recorded is complete");
	CHECK(SparkDeviceQueueHostCallback(queue,GateWaitCallback,&gate) == SPARK_STATUS_OK,"block the queue");
	CHECK(SparkDeviceCopy(queue,At(&host,0u),At(&device,0u),256u) == SPARK_STATUS_OK,"queue a copy behind the block");
	CHECK(SparkDeviceQueueHostCallback(queue,FlagCallback,&flag) == SPARK_STATUS_OK,"queue a callback");
	CHECK(SparkDeviceEventRecord(event,queue) == SPARK_STATUS_OK,"record behind the block");
	PauseMs(20);
	CHECK(__atomic_load_n(&flag,__ATOMIC_SEQ_CST) == 0,"callbacks run in queue order, never inline");
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_BUSY,"a buffer used by queued work cannot be freed");
	CHECK(SparkDeviceFree(&host) == SPARK_STATUS_BUSY,"the copy's other buffer cannot be freed either");
	CHECK(SparkDeviceQueueWait(queue,1000000u) == SPARK_STATUS_PENDING,"a wait with a timeout reports pending work");
	CHECK(SparkDeviceQueueQuery(queue) == SPARK_STATUS_PENDING,"a query reports pending work");
	CHECK(SparkDeviceEventQuery(event) == SPARK_STATUS_PENDING,"the event is pending");
	CHECK(SparkDeviceEventDestroy(event) == SPARK_STATUS_BUSY,"a pending event cannot be destroyed");
	CHECK(SparkDeviceSynchronize(1000000u) == SPARK_STATUS_PENDING,"device synchronize reports pending work");
	GateOpen(&gate);
	CHECK(SparkDeviceEventWait(event,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the event completes");
	CHECK(SparkDeviceSynchronize(SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"device synchronize waits for every queue");
	CHECK(__atomic_load_n(&flag,__ATOMIC_SEQ_CST) == 1,"the callback ran");
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free after the work completes: device");
	CHECK(SparkDeviceFree(&host) == SPARK_STATUS_OK,"free after the work completes: host");
	CHECK(SparkDeviceEventDestroy(event) == SPARK_STATUS_OK,"destroy the completed event");
}

static void TestTiming(SparkDeviceQueue *queue)
{
	SparkDeviceEvent *start = 0, *stop = 0, *unused = 0;
	uint64_t nanoseconds = 0u;
	CHECK(SparkDeviceEventCreate(&start) == SPARK_STATUS_OK && SparkDeviceEventCreate(&stop) == SPARK_STATUS_OK &&
		SparkDeviceEventCreate(&unused) == SPARK_STATUS_OK,"create events");
	CHECK(SparkDeviceEventElapsed(start,unused,&nanoseconds) == SPARK_STATUS_INVALID_ARGUMENT,"elapsed time needs two recorded events");
	CHECK(SparkDeviceEventRecord(start,queue) == SPARK_STATUS_OK,"record start");
	CHECK(SparkDeviceQueueHostCallback(queue,SleepCallback,(void *)(uintptr_t)3000000u) == SPARK_STATUS_OK,"sleep 3 ms on the queue");
	CHECK(SparkDeviceEventRecord(stop,queue) == SPARK_STATUS_OK,"record stop");
	CHECK(SparkDeviceEventWait(stop,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for stop");
	CHECK(SparkDeviceEventElapsed(start,stop,&nanoseconds) == SPARK_STATUS_OK && nanoseconds >= 2500000u,"elapsed time covers the queued work");
	CHECK(SparkDeviceEventDestroy(start) == SPARK_STATUS_OK && SparkDeviceEventDestroy(stop) == SPARK_STATUS_OK &&
		SparkDeviceEventDestroy(unused) == SPARK_STATUS_OK,"destroy events");
}

static void TestResolve(void)
{
	SparkDeviceBuffer first, second;
	SparkDeviceRange range;
	SparkDeviceAddress end;
	CHECK(SparkDeviceAllocate(&first,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,1000u,"resolve first") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceAllocate(&second,SPARK_DEVICE_SPACE_SHARED_COHERENT,3000u,"resolve second") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceResolve(first.device + 100u,50u,&range) == SPARK_STATUS_OK && range.buffer->handle == first.handle && range.offset == 100u,
		"an address inside a buffer resolves to that buffer and offset");
	CHECK(SparkDeviceResolve(second.device + 2999u,1u,&range) == SPARK_STATUS_OK && range.buffer->handle == second.handle && range.offset == 2999u,
		"the last byte resolves");
	CHECK(SparkDeviceResolve(first.device + 990u,20u,&range) == SPARK_STATUS_INVALID_ARGUMENT,"a range past the end of its buffer is refused");
	end = first.device;
	CHECK(SparkDeviceFree(&first) == SPARK_STATUS_OK,"free");
	CHECK(SparkDeviceResolve(end + 100u,1u,&range) == SPARK_STATUS_NOT_FOUND,"a freed buffer's address no longer resolves");
	CHECK(SparkDeviceFree(&second) == SPARK_STATUS_OK,"free");
}

static void TestFileAndRegister(SparkDeviceQueue *queue)
{
	const uint64_t page = SparkDeviceGetProperties()->page_bytes;
	const uint64_t file_bytes = 3u * page + 123u, offset = page + 17u, bytes = 1000u;
	char path[] = "/tmp/spark-device-contract-XXXXXX";
	uint8_t *contents = (uint8_t *)malloc(file_bytes);
	uint8_t *borrowed = (uint8_t *)malloc(512u);
	SparkDeviceBuffer mapped, device, back, registered;
	int fd = mkstemp(path);
	CHECK(fd >= 0 && contents != 0 && borrowed != 0,"create a file");
	Pattern(contents,file_bytes,9u);
	CHECK(write(fd,contents,file_bytes) == (ssize_t)file_bytes,"write the file");
	CHECK(SparkDeviceMapFile(&mapped,fd,offset,bytes,"contract file") == SPARK_STATUS_OK,"map a file range at an unaligned offset");
	CHECK(mapped.space == SPARK_DEVICE_SPACE_FILE_BACKED && mapped.host != 0 && memcmp(mapped.host,contents + offset,bytes) == 0,
		"the host view starts at the requested offset");
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"file device") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PAGEABLE,bytes,"file back") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceCopy(queue,At(&device,0u),At(&mapped,0u),bytes) == SPARK_STATUS_OK,"copy file to device");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"queues are unordered, so the copy completes before another queue reads it");
	CHECK(SparkDeviceCopySync(At(&back,0u),At(&device,0u),bytes) == SPARK_STATUS_OK,"synchronous copy device to host");
	CHECK(memcmp(back.host,contents + offset,bytes) == 0,"file bytes reach the device and come back");
	CHECK(SparkDeviceCopy(queue,At(&mapped,0u),At(&device,0u),bytes) == SPARK_STATUS_INVALID_ARGUMENT,"a mapped file is read only");
	memset(borrowed,0,512u);
	CHECK(SparkDeviceRegister(&registered,borrowed,512u,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,"contract registered") == SPARK_STATUS_OK,
		"register host memory");
	CHECK(registered.host == borrowed && registered.device != 0u,"registered memory keeps its host pointer and gains a device address");
	CHECK(SparkDeviceFill(queue,At(&device,0u),0x33u,512u) == SPARK_STATUS_OK,"fill device memory");
	CHECK(SparkDeviceCopy(queue,At(&registered,0u),At(&device,0u),512u) == SPARK_STATUS_OK,"copy device to registered memory");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait");
	CHECK(borrowed[0] == 0x33u && borrowed[511] == 0x33u,"registered memory sees the device's bytes");
	CHECK(SparkDeviceFree(&registered) == SPARK_STATUS_OK,"unregister");
	borrowed[0] = 0x44u;
	CHECK(borrowed[0] == 0x44u,"unregistering leaves the host memory to its owner");
	CHECK(SparkDeviceFree(&mapped) == SPARK_STATUS_OK,"free the mapping");
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the device buffer");
	CHECK(SparkDeviceFree(&back) == SPARK_STATUS_OK,"free the host buffer");
	close(fd);
	unlink(path);
	free(contents);
	free(borrowed);
}

typedef struct ContractParams
{
	uint32_t count;
	uint32_t seed;
} ContractParams;

static SparkDeviceLaunch Launch1d(uint32_t count, uint32_t width, const ContractParams *params, const SparkDeviceBinding *bindings, uint32_t binding_count)
{
	SparkDeviceLaunch launch;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (count + width - 1u) / width;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = width;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = params;
	launch.constant_bytes = sizeof(*params);
	launch.bindings = bindings;
	launch.binding_count = binding_count;
	return(launch);
}

static SparkDeviceBinding Bind(const SparkDeviceBuffer *buffer, uint64_t offset, uint64_t bytes, uint32_t access)
{
	SparkDeviceBinding binding;
	binding.range = At(buffer,offset);
	binding.bytes = bytes;
	binding.access = access;
	return(binding);
}

static void TestKernelRefusals(SparkDeviceQueue *queue, SparkDeviceKernel *writer, SparkDeviceBuffer *device)
{
	const uint64_t page = SparkDeviceGetProperties()->page_bytes;
	SparkDeviceBuffer pageable, mapped;
	SparkDeviceBinding bindings[SPARK_DEVICE_MAX_BINDINGS + 1u];
	SparkDeviceLaunch launch;
	ContractParams params = {64u,1u};
	char path[] = "/tmp/spark-device-kernel-XXXXXX";
	uint8_t contents[256];
	SparkDeviceKernelInfo info;
	uint32_t index;
	int fd = mkstemp(path);
	memset(contents,7,sizeof(contents));
	CHECK(fd >= 0 && write(fd,contents,sizeof(contents)) == (ssize_t)sizeof(contents),"create a file to map");
	CHECK(SparkDeviceMapFile(&mapped,fd,0u,sizeof(contents),"kernel mapped") == SPARK_STATUS_OK,"map the file");
	CHECK(SparkDeviceAllocate(&pageable,SPARK_DEVICE_SPACE_HOST_PAGEABLE,256u,"kernel pageable") == SPARK_STATUS_OK,"allocate pageable");
	CHECK(SparkDeviceKernelGetInfo(writer,&info) == SPARK_STATUS_OK && info.max_threads >= 32u && info.simd_width >= 1u,"the kernel reports its limits");
	bindings[0] = Bind(&pageable,0u,256u,SPARK_DEVICE_BIND_WRITE);
	launch = Launch1d(64u,64u,&params,bindings,1u);
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"pageable memory cannot be bound");
	bindings[0] = Bind(&mapped,0u,256u,SPARK_DEVICE_BIND_WRITE);
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"a read-only mapping cannot be bound for writing");
	bindings[0] = Bind(device,device->bytes - 16u,256u,SPARK_DEVICE_BIND_WRITE);
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"a binding past the end of its buffer is refused");
	bindings[0] = Bind(device,0u,256u,0u);
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"a binding without access is refused");
	bindings[0] = Bind(device,0u,256u,SPARK_DEVICE_BIND_WRITE);
	launch.threads.x = info.max_threads + 1u;
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_UNSUPPORTED,"more threads than the kernel allows are refused");
	launch = Launch1d(64u,64u,&params,bindings,1u);
	launch.groups.y = 0u;
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"an empty grid is refused");
	launch = Launch1d(64u,64u,&params,bindings,1u);
	launch.constant_bytes = SPARK_DEVICE_MAX_CONSTANT_BYTES + 4u;
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"constants over 4096 bytes are refused");
	for ( index = 0u; index <= SPARK_DEVICE_MAX_BINDINGS; index++ )
		bindings[index] = Bind(device,0u,256u,SPARK_DEVICE_BIND_WRITE);
	launch = Launch1d(64u,64u,&params,bindings,SPARK_DEVICE_MAX_BINDINGS + 1u);
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"more than 30 bindings are refused");
	launch = Launch1d(64u,64u,&params,bindings,1u);
	launch.shared_bytes = SparkDeviceGetProperties()->max_shared_bytes + 16u;
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_UNSUPPORTED,"more shared memory than the device has is refused");
	CHECK(SparkDeviceFree(&pageable) == SPARK_STATUS_OK && SparkDeviceFree(&mapped) == SPARK_STATUS_OK,"free the refusal buffers");
	(void)page;
	close(fd);
	unlink(path);
}

static void TestKernelOrdering(SparkDeviceQueue *queue, SparkDeviceKernel *write, SparkDeviceKernel *read)
{
	enum { COUNT = 100003, ROUNDS = 64 };
	SparkDeviceBuffer produced, copied, result;
	SparkDeviceBinding bindings[2];
	SparkDeviceLaunch launch;
	ContractParams params;
	uint32_t round, index;
	int ordered = 1;
	CHECK(SparkDeviceAllocate(&produced,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,COUNT * 4u,"kernel produced") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceAllocate(&copied,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,COUNT * 4u,"kernel copied") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceAllocate(&result,SPARK_DEVICE_SPACE_SHARED_COHERENT,COUNT * 4u,"kernel result") == SPARK_STATUS_OK,"allocate");
	for ( round = 1u; round <= ROUNDS; round++ )
	{
		params.count = COUNT;
		params.seed = round * 977u;
		bindings[0] = Bind(&produced,0u,COUNT * 4u,SPARK_DEVICE_BIND_WRITE);
		launch = Launch1d(COUNT,256u,&params,bindings,1u);
		CHECK(SparkDeviceKernelLaunch(queue,write,&launch) == SPARK_STATUS_OK,"launch the writer");
		CHECK(SparkDeviceCopy(queue,At(&copied,0u),At(&produced,0u),COUNT * 4u) == SPARK_STATUS_OK,"copy between the kernels");
		bindings[0] = Bind(&copied,0u,COUNT * 4u,SPARK_DEVICE_BIND_READ);
		bindings[1] = Bind(&result,0u,COUNT * 4u,SPARK_DEVICE_BIND_WRITE);
		launch = Launch1d(COUNT,256u,&params,bindings,2u);
		CHECK(SparkDeviceKernelLaunch(queue,read,&launch) == SPARK_STATUS_OK,"launch the reader");
		CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for the round");
		for ( index = 0u; index < COUNT; index++ )
			if ( ((uint32_t *)result.host)[index] != index * 2654435761u + params.seed + 1u )
				ordered = 0;
	}
	CHECK(ordered,"a kernel, a copy and a kernel on one queue run in order");
	CHECK(SparkDeviceFree(&produced) == SPARK_STATUS_OK && SparkDeviceFree(&copied) == SPARK_STATUS_OK && SparkDeviceFree(&result) == SPARK_STATUS_OK,
		"free the ordering buffers");
}

static void TestKernelAcrossQueues(SparkDeviceKernel *write, SparkDeviceKernel *read)
{
	enum { COUNT = 65536, ROUNDS = 32 };
	SparkDeviceQueue *producer = 0, *consumer = 0;
	SparkDeviceEvent *ready = 0;
	SparkDeviceBuffer produced, result;
	SparkDeviceBinding bindings[2];
	SparkDeviceLaunch launch;
	ContractParams params;
	uint32_t round, index;
	int ordered = 1;
	CHECK(SparkDeviceQueueCreate(&producer,"kernel producer") == SPARK_STATUS_OK && SparkDeviceQueueCreate(&consumer,"kernel consumer") == SPARK_STATUS_OK,
		"create two queues");
	CHECK(SparkDeviceEventCreate(&ready) == SPARK_STATUS_OK,"create an event");
	CHECK(SparkDeviceAllocate(&produced,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,COUNT * 4u,"kernel across") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceAllocate(&result,SPARK_DEVICE_SPACE_SHARED_COHERENT,COUNT * 4u,"kernel across result") == SPARK_STATUS_OK,"allocate");
	for ( round = 1u; round <= ROUNDS; round++ )
	{
		Gate gate;
		GateInit(&gate);
		params.count = COUNT;
		params.seed = round * 31u;
		SparkDeviceQueueHostCallback(producer,GateWaitCallback,&gate);
		bindings[0] = Bind(&produced,0u,COUNT * 4u,SPARK_DEVICE_BIND_WRITE);
		launch = Launch1d(COUNT,256u,&params,bindings,1u);
		SparkDeviceKernelLaunch(producer,write,&launch);
		SparkDeviceEventRecord(ready,producer);
		SparkDeviceQueueWaitEvent(consumer,ready);
		bindings[0] = Bind(&produced,0u,COUNT * 4u,SPARK_DEVICE_BIND_READ);
		bindings[1] = Bind(&result,0u,COUNT * 4u,SPARK_DEVICE_BIND_WRITE);
		launch = Launch1d(COUNT,256u,&params,bindings,2u);
		SparkDeviceKernelLaunch(consumer,read,&launch);
		if ( round % 4u == 1u )
			PauseMs(2);
		GateOpen(&gate);
		SparkDeviceQueueWait(consumer,SPARK_DEVICE_WAIT_FOREVER);
		for ( index = 0u; index < COUNT; index++ )
			if ( ((uint32_t *)result.host)[index] != index * 2654435761u + params.seed + 1u )
				ordered = 0;
	}
	CHECK(ordered,"a kernel waiting on an event reads what the producer's kernel wrote after its host callback");
	CHECK(SparkDeviceQueueWait(producer,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"drain the producer");
	CHECK(SparkDeviceFree(&produced) == SPARK_STATUS_OK && SparkDeviceFree(&result) == SPARK_STATUS_OK,"free the cross-queue buffers");
	CHECK(SparkDeviceEventDestroy(ready) == SPARK_STATUS_OK,"destroy the event");
	CHECK(SparkDeviceQueueDestroy(producer) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(consumer) == SPARK_STATUS_OK,"destroy the queues");
}

static void TestKernelBusy(SparkDeviceQueue *queue, SparkDeviceLibrary *library, SparkDeviceKernel *shared)
{
	SparkDeviceBuffer out;
	SparkDeviceBinding binding;
	SparkDeviceLaunch launch;
	ContractParams params = {0u,5u};
	Gate gate;
	uint32_t index;
	int matched = 1;
	GateInit(&gate);
	CHECK(SparkDeviceAllocate(&out,SPARK_DEVICE_SPACE_SHARED_COHERENT,4u * 128u * 4u,"kernel busy") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceQueueHostCallback(queue,GateWaitCallback,&gate) == SPARK_STATUS_OK,"block the queue");
	binding = Bind(&out,0u,4u * 128u * 4u,SPARK_DEVICE_BIND_WRITE);
	launch = Launch1d(4u * 128u,128u,&params,&binding,1u);
	launch.shared_bytes = 128u * 4u;
	CHECK(SparkDeviceKernelLaunch(queue,shared,&launch) == SPARK_STATUS_OK,"queue a launch with group-shared memory behind the block");
	CHECK(SparkDeviceKernelDestroy(shared) == SPARK_STATUS_BUSY,"a kernel with a queued launch cannot be destroyed");
	CHECK(SparkDeviceFree(&out) == SPARK_STATUS_BUSY,"a buffer bound to a queued launch cannot be freed");
	CHECK(SparkDeviceLibraryUnload(library) == SPARK_STATUS_BUSY,"a library with live kernels cannot be unloaded");
	CHECK(SparkDeviceQueueQuery(queue) == SPARK_STATUS_PENDING,"the launch is pending behind the block");
	GateOpen(&gate);
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the launch completes");
	for ( index = 0u; index < 4u * 128u; index++ )
		if ( ((uint32_t *)out.host)[index] != (127u - index % 128u) + params.seed )
			matched = 0;
	CHECK(matched,"group-shared memory is per group and sized by the launch");
	CHECK(SparkDeviceFree(&out) == SPARK_STATUS_OK,"free after the launch completes");
}

static void TestKernels(SparkDeviceQueue *queue)
{
	SparkDeviceLibrary *library = 0;
	SparkDeviceKernel *write = 0, *read = 0, *shared = 0, *missing = 0;
	SparkDeviceBuffer device;
	if ( SparkDeviceGetProperties() == 0 )
	{
		CHECK(0,"kernel tests need an initialized device");
		return;
	}
	if ( strcmp(SparkDeviceGetProperties()->backend,"metal") != 0 )
	{
		CHECK(SparkDeviceLibraryLoad(&library,SPARK_DEVICE_TEST_LIBRARY,"contract kernels") == SPARK_STATUS_UNSUPPORTED && library == 0,
			"a backend without device kernels refuses to load a kernel library");
		CHECK(strcmp(SparkDeviceLastError()->operation,"library_load") == 0,"the refusal names the operation");
		return;
	}
	CHECK(SparkDeviceLibraryLoad(&library,"/nonexistent/kernels.metallib","contract kernels") == SPARK_STATUS_NOT_FOUND,"a missing library is refused");
	CHECK(SparkDeviceLibraryLoad(&library,SPARK_DEVICE_TEST_LIBRARY,"contract kernels") == SPARK_STATUS_OK,"load the contract kernel library");
	CHECK(SparkDeviceKernelCreate(&missing,library,"contract_absent") == SPARK_STATUS_NOT_FOUND && missing == 0,"an absent kernel is refused");
	CHECK(SparkDeviceKernelCreate(&write,library,"contract_write") == SPARK_STATUS_OK &&
		SparkDeviceKernelCreate(&read,library,"contract_read") == SPARK_STATUS_OK &&
		SparkDeviceKernelCreate(&shared,library,"contract_shared") == SPARK_STATUS_OK,"create the contract kernels");
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,4096u,"kernel device") == SPARK_STATUS_OK,"allocate");
	TestKernelRefusals(queue,write,&device);
	TestKernelOrdering(queue,write,read);
	TestKernelAcrossQueues(write,read);
	TestKernelBusy(queue,library,shared);
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the device buffer");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_BUSY,"shutdown refuses while a kernel library is loaded");
	CHECK(SparkDeviceKernelDestroy(write) == SPARK_STATUS_OK && SparkDeviceKernelDestroy(read) == SPARK_STATUS_OK &&
		SparkDeviceKernelDestroy(shared) == SPARK_STATUS_OK,"destroy the kernels");
	CHECK(SparkDeviceKernelDestroy(write) == SPARK_STATUS_INVALID_ARGUMENT,"a destroyed kernel is refused");
	CHECK(SparkDeviceLibraryUnload(library) == SPARK_STATUS_OK,"unload the library");
}

static void TestLifecycle(void)
{
	SparkDeviceBuffer buffer;
	SparkDeviceQueue *queue = 0;
	CHECK(SparkDeviceAllocate(&buffer,SPARK_DEVICE_SPACE_HOST_PAGEABLE,64u,"lifecycle") == SPARK_STATUS_OK,"allocate");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_BUSY,"shutdown refuses while a buffer is live");
	CHECK(SparkDeviceFree(&buffer) == SPARK_STATUS_OK,"free");
	CHECK(SparkDeviceQueueCreate(&queue,"lifecycle") == SPARK_STATUS_OK,"create a queue");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_BUSY,"shutdown refuses while a queue is live");
	CHECK(SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK,"destroy the queue");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK,"shutdown");
	CHECK(SparkDeviceAllocate(&buffer,SPARK_DEVICE_SPACE_HOST_PAGEABLE,64u,"lifecycle") == SPARK_STATUS_INVALID_ARGUMENT,"operations after shutdown are refused");
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK,"initialize again");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK,"shutdown again");
}

int main(void)
{
	SparkDeviceQueue *queue = 0;
	const SparkDeviceProperties *properties;
	char backend[sizeof(properties->backend)] = "?";
	CHECK(SparkDeviceGetProperties() == 0,"no properties before initialize");
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK,"initialize");
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_DUPLICATE,"one device per process");
	properties = SparkDeviceGetProperties();
	CHECK(properties != 0 && properties->backend[0] != '\0' && properties->page_bytes != 0u &&
		(properties->page_bytes & (properties->page_bytes - 1u)) == 0u,"properties name the backend and a power-of-two page");
	if ( properties != 0 )
		memcpy(backend,properties->backend,sizeof(backend));
	CHECK(SparkDeviceQueueCreate(&queue,"contract") == SPARK_STATUS_OK,"create a queue");
	TestSpaces();
	TestCopies(queue);
	TestFreshUpload(queue);
	TestOrdering();
	TestBusy(queue);
	TestTiming(queue);
	TestResolve();
	TestFileAndRegister(queue);
	TestKernels(queue);
	CHECK(SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK,"destroy the queue");
	TestLifecycle();
	if ( failures != 0 )
	{
		printf("device contract (%s backend): %d failure(s)\n",backend,failures);
		return 1;
	}
	printf("PASS device contract (%s backend)\n",backend);
	return 0;
}
