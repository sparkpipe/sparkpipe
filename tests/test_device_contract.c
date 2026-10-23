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
	TestOrdering();
	TestBusy(queue);
	TestTiming(queue);
	TestResolve();
	TestFileAndRegister(queue);
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
