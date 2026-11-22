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
	struct timespec pause = {milliseconds / 1000L, milliseconds % 1000L * 1000000L};
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

static void CountCallback(void *context)
{
	__atomic_add_fetch((uint32_t *)context,1u,__ATOMIC_SEQ_CST);
}

static uint32_t ProgramsRecord(void)
{
	return(strcmp(SparkDeviceGetProperties()->backend,"metal") != 0);
}

static uint32_t ProgramsRun(void)
{
	return((SparkDeviceGetProperties()->properties & SPARK_DEVICE_PROPERTY_PROGRAMS) != 0u);
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

static uint32_t BorrowedFill(SparkDeviceQueue *queue, SparkDeviceBuffer *buffer, uint8_t value)
{
	const uint8_t *bytes = (const uint8_t *)buffer->host;
	return(SparkDeviceFill(queue,At(buffer,0u),value,64u) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK &&
		bytes[0] == value && bytes[63] == value);
}

static void TestAdoptOwnQueue(uint32_t adopts)
{
	SparkDeviceQueue *owner = 0,*borrowed = 0,*again = 0,*gone = 0,*refused = 0;
	SparkDeviceBuffer scratch;
	CHECK(SparkDeviceAllocate(&scratch,SPARK_DEVICE_SPACE_SHARED_COHERENT,4096u,"adopt scratch") == SPARK_STATUS_OK &&
		SparkDeviceQueueCreate(&owner,"adopt owner") == SPARK_STATUS_OK,"create a queue and a host-visible buffer");
	CHECK(SparkDeviceQueueAdopt(&borrowed,owner,"adopt own") == SPARK_STATUS_OK && borrowed == owner,
		"adopting one of the device's own live queues hands back that queue, its sequence numbers and capture state");
	CHECK(BorrowedFill(borrowed,&scratch,0x42),"work queued through the adopted handle runs");
	CHECK(SparkDeviceQueueDestroy(borrowed) == SPARK_STATUS_OK && BorrowedFill(owner,&scratch,0x43),"releasing the adopted handle leaves the owner's queue working");
	CHECK(SparkDeviceQueueAdopt(&again,owner,"adopt own again") == SPARK_STATUS_OK && again == owner && SparkDeviceQueueDestroy(owner) == SPARK_STATUS_OK &&
		BorrowedFill(again,&scratch,0x44),"released by its owner first, the queue lives on for the adopter");
	CHECK(SparkDeviceQueueDestroy(again) == SPARK_STATUS_OK,"the last release destroys the queue");
	if ( adopts == 0u )
	{
		CHECK(SparkDeviceQueueCreate(&gone,"adopt gone") == SPARK_STATUS_OK && SparkDeviceQueueDestroy(gone) == SPARK_STATUS_OK &&
			SparkDeviceQueueAdopt(&refused,gone,"adopt freed") == SPARK_STATUS_UNSUPPORTED && refused == 0,
			"a freed queue's handle is not adopted: it is no longer a live queue of the device");
	}
	CHECK(SparkDeviceFree(&scratch) == SPARK_STATUS_OK,"free the scratch buffer");
}

static void TestAdopt(void)
{
	SparkDeviceBuffer held, adopted;
	SparkDeviceQueue *queue = 0;
	const uint32_t adopts = strcmp(SparkDeviceGetProperties()->backend,"cuda") == 0;
	CHECK(SparkDeviceAllocate(&held,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,4096u,"adopt held") == SPARK_STATUS_OK,"allocate the held buffer");
	CHECK(SparkDeviceAdopt(&adopted,held.device,0,0u,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"adopt empty") == SPARK_STATUS_INVALID_ARGUMENT,
		"adopting zero bytes is refused");
	CHECK(SparkDeviceAdopt(&adopted,0u,0,64u,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"adopt nowhere") == SPARK_STATUS_INVALID_ARGUMENT,
		"adopting device memory without a device address is refused");
	CHECK(SparkDeviceAdopt(&adopted,held.device,0,64u,SPARK_DEVICE_SPACE_HOST_PAGEABLE,"adopt pageable") == SPARK_STATUS_INVALID_ARGUMENT,
		"adopted memory is device-visible");
	CHECK(SparkDeviceAdopt(&adopted,held.device + 64u,0,64u,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"adopt overlap") == SPARK_STATUS_INVALID_ARGUMENT,
		"a range inside a buffer the device holds is never adopted");
	CHECK(SparkDeviceQueueAdopt(&queue,0,"adopt no queue") == SPARK_STATUS_INVALID_ARGUMENT,"adopting a queue needs the backend's own queue");
	if ( adopts == 0u )
	{
		CHECK(SparkDeviceAdopt(&adopted,held.device + (1ull << 30),0,64u,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"adopt foreign") ==
			SPARK_STATUS_UNSUPPORTED && adopted.handle == 0u,"a backend that holds only its own memory refuses to adopt any");
		CHECK(SparkDeviceQueueAdopt(&queue,&held,"adopt foreign") == SPARK_STATUS_UNSUPPORTED && queue == 0,
			"a backend that owns all its queues refuses to adopt one");
	}
	CHECK(SparkDeviceFree(&held) == SPARK_STATUS_OK,"free the held buffer");
	TestAdoptOwnQueue(adopts);
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

static void TestEventLifetime(void)
{
	SparkDeviceQueue *slow = 0, *fast = 0;
	SparkDeviceEvent *event = 0;
	Gate gate;
	CHECK(SparkDeviceQueueCreate(&slow,"event lifetime slow") == SPARK_STATUS_OK && SparkDeviceQueueCreate(&fast,"event lifetime fast") == SPARK_STATUS_OK,
		"create two queues");
	CHECK(SparkDeviceEventCreate(&event) == SPARK_STATUS_OK,"create an event");
	GateInit(&gate);
	CHECK(SparkDeviceQueueHostCallback(slow,GateWaitCallback,&gate) == SPARK_STATUS_OK,"block the waiting queue");
	CHECK(SparkDeviceEventRecord(event,fast) == SPARK_STATUS_OK,"record on the idle queue");
	CHECK(SparkDeviceQueueWait(fast,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the recording queue completes");
	CHECK(SparkDeviceQueueWaitEvent(slow,event) == SPARK_STATUS_OK,"queue a wait on the event behind the block");
	CHECK(SparkDeviceEventDestroy(event) == SPARK_STATUS_BUSY,"an event a blocked queue still waits on cannot be destroyed, though its record completed");
	GateOpen(&gate);
	CHECK(SparkDeviceQueueWait(slow,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the waiting queue passes the wait");
	CHECK(SparkDeviceEventDestroy(event) == SPARK_STATUS_OK,"destroy once the waiting queue has passed");
	CHECK(SparkDeviceEventCreate(&event) == SPARK_STATUS_OK,"create a second event");
	GateInit(&gate);
	CHECK(SparkDeviceQueueHostCallback(slow,GateWaitCallback,&gate) == SPARK_STATUS_OK,"block the first recording queue");
	CHECK(SparkDeviceEventRecord(event,slow) == SPARK_STATUS_OK,"record behind the block");
	CHECK(SparkDeviceEventRecord(event,fast) == SPARK_STATUS_OK,"record again on the idle queue");
	CHECK(SparkDeviceQueueWait(fast,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the later record completes first");
	CHECK(SparkDeviceEventDestroy(event) == SPARK_STATUS_BUSY,"an event whose earlier record is still queued cannot be destroyed");
	GateOpen(&gate);
	CHECK(SparkDeviceQueueWait(slow,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the earlier record completes");
	CHECK(SparkDeviceEventDestroy(event) == SPARK_STATUS_OK,"destroy once both records have completed");
	CHECK(SparkDeviceQueueDestroy(slow) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(fast) == SPARK_STATUS_OK,"destroy the queues");
}

static void TestLongHold(void)
{
	enum { BYTES = 4096, HOLD_MS = 6000 };
	SparkDeviceQueue *producer = 0, *consumer = 0;
	SparkDeviceEvent *ready = 0;
	SparkDeviceBuffer device, result;
	Gate gate;
	uint32_t index;
	int filled = 1;
	CHECK(SparkDeviceQueueCreate(&producer,"long hold producer") == SPARK_STATUS_OK && SparkDeviceQueueCreate(&consumer,"long hold consumer") == SPARK_STATUS_OK,
		"create two queues");
	CHECK(SparkDeviceEventCreate(&ready) == SPARK_STATUS_OK,"create an event");
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,BYTES,"long hold device") == SPARK_STATUS_OK,"allocate the shared target");
	CHECK(SparkDeviceAllocate(&result,SPARK_DEVICE_SPACE_HOST_PAGEABLE,BYTES,"long hold result") == SPARK_STATUS_OK,"allocate the result");
	GateInit(&gate);
	CHECK(SparkDeviceQueueHostCallback(producer,GateWaitCallback,&gate) == SPARK_STATUS_OK,"block the producer");
	CHECK(SparkDeviceFill(producer,At(&device,0u),0x5au,BYTES) == SPARK_STATUS_OK,"fill behind the block");
	CHECK(SparkDeviceEventRecord(ready,producer) == SPARK_STATUS_OK,"record behind the block");
	CHECK(SparkDeviceQueueWaitEvent(consumer,ready) == SPARK_STATUS_OK,"wait on the event from the other queue");
	CHECK(SparkDeviceCopy(consumer,At(&result,0u),At(&device,0u),BYTES) == SPARK_STATUS_OK,"copy after the wait");
	CHECK(SparkDeviceQueueQuery(consumer) == SPARK_STATUS_PENDING,"the waiting queue reports pending work");
	PauseMs(HOLD_MS);
	CHECK(SparkDeviceQueueQuery(producer) == SPARK_STATUS_PENDING,"work behind a host callback is still pending after a hold longer than a GPU watchdog");
	GateOpen(&gate);
	CHECK(SparkDeviceQueueWait(consumer,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"a queue that waited on an event for longer than a GPU watchdog completes");
	CHECK(SparkDeviceQueueWait(producer,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"work held behind a host callback for longer than a GPU watchdog completes");
	for ( index = 0u; index < BYTES; index++ )
		if ( ((uint8_t *)result.host)[index] != 0x5au )
			filled = 0;
	CHECK(filled,"the waiting queue sees the producer's fill");
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK && SparkDeviceFree(&result) == SPARK_STATUS_OK,"free the long hold buffers");
	CHECK(SparkDeviceEventDestroy(ready) == SPARK_STATUS_OK,"destroy the event");
	CHECK(SparkDeviceQueueDestroy(producer) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(consumer) == SPARK_STATUS_OK,"destroy the queues");
}

static void TestRecordedEventDestroy(SparkDeviceQueue *queue)
{
	SparkDeviceBuffer buffer = {0};
	SparkDeviceEvent *event = 0;
	uint32_t round,refused = 0u,failed = 0u;
	CHECK(SparkDeviceAllocate(&buffer,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,4096u,"record-destroy") == SPARK_STATUS_OK,"allocate the record-destroy buffer");
	for (round=0u; round<20000u; round++)
	{
		if ( SparkDeviceEventCreate(&event) != SPARK_STATUS_OK || SparkDeviceFill(queue,At(&buffer,0u),(uint8_t)round,4096u) != SPARK_STATUS_OK ||
			SparkDeviceEventRecord(event,queue) != SPARK_STATUS_OK || SparkDeviceEventWait(event,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		{
			failed++;
			break;
		}
		if ( SparkDeviceEventDestroy(event) != SPARK_STATUS_OK )
		{
			refused++;
			SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
			SparkDeviceEventDestroy(event);
		}
	}
	CHECK(failed == 0u,"record and wait on a fresh event every round");
	CHECK(refused == 0u,"an event whose record has completed can be destroyed as soon as a wait on it returns");
	SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	CHECK(SparkDeviceFree(&buffer) == SPARK_STATUS_OK,"free the record-destroy buffer");
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

static void TestHostMemoryShare(void)
{
	const SparkDeviceProperties *properties = SparkDeviceGetProperties();
	SparkDeviceMemoryHandle host = 0u, imported = 0u, device = 0u;
	SparkDeviceAddress filled = 0u, mapped = 0u, refused = 0u;
	SparkDeviceBuffer view, readback;
	char expected[SPARK_DEVICE_ERROR_TEXT_BYTES];
	uint64_t granularity = 0u;
	uint8_t *pattern;
	int fd = -1;
	if ( (properties->properties & SPARK_DEVICE_PROPERTY_HOST_MEMORY_SHARE) == 0u )
	{
		snprintf(expected,sizeof(expected),"the %s backend has no exportable host memory",properties->backend);
		CHECK(SparkDeviceMemoryCreateHost(&host,UINT64_C(1) << 21,"contract host share") == SPARK_STATUS_UNSUPPORTED && host == 0u &&
			strcmp(SparkDeviceLastError()->text,expected) == 0,"without the property, host memory is refused and the refusal names the backend");
		return;
	}
	CHECK(SparkDeviceMemoryGranularity(SPARK_DEVICE_GRANULARITY_MINIMUM,&granularity) == SPARK_STATUS_OK && granularity != 0u,"granularity");
	pattern = malloc((size_t)granularity);
	CHECK(pattern != 0,"pattern buffer");
	if ( pattern == 0 )
		return;
	Pattern(pattern,granularity,11u);
	CHECK(SparkDeviceMemoryCreateHost(&host,granularity + 1u,"contract host share") == SPARK_STATUS_INVALID_ARGUMENT,"host memory comes in whole granules");
	CHECK(SparkDeviceMemoryCreateHost(&host,granularity,"contract host share") == SPARK_STATUS_OK && host != 0u,"create host memory");
	CHECK(SparkDeviceAddressReserve(&filled,granularity,"contract host share") == SPARK_STATUS_OK,"reserve a range to fill it");
	CHECK(SparkDeviceMemoryMap(filled,granularity,host,0u,SPARK_DEVICE_ACCESS_HOST_WRITE) == SPARK_STATUS_INVALID_ARGUMENT &&
		strcmp(SparkDeviceLastError()->text,"host write access needs host read access") == 0,"host write needs host read");
	CHECK(SparkDeviceMemoryMap(filled,granularity,host,0u,SPARK_DEVICE_ACCESS_DEVICE_WRITE) == SPARK_STATUS_INVALID_ARGUMENT,"device write needs device read");
	CHECK(SparkDeviceMemoryMap(filled,granularity,host,0u,0u) == SPARK_STATUS_INVALID_ARGUMENT,"a mapping grants a read");
	CHECK(SparkDeviceMemoryMap(filled,granularity,host,0u,SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE) == SPARK_STATUS_OK,
		"map host memory for the CPU alone");
	memcpy((void *)(uintptr_t)filled,pattern,(size_t)granularity);
	CHECK(SparkDeviceMemoryExport(host,&fd) == SPARK_STATUS_OK && fd >= 0,"export host memory");
	CHECK(SparkDeviceMemoryImport(&imported,fd,"contract host share") == SPARK_STATUS_OK && imported != 0u,"import host memory");
	if ( fd >= 0 )
		(void)close(fd);
	CHECK(SparkDeviceAddressReserve(&mapped,granularity,"contract host share") == SPARK_STATUS_OK,"reserve a range for the import");
	CHECK(SparkDeviceMemoryMap(mapped,granularity,imported,0u,SPARK_DEVICE_ACCESS_DEVICE_READ) == SPARK_STATUS_OK,"map the import device-read-only");
	CHECK(SparkDeviceAdopt(&view,mapped,0,granularity,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"contract host share view") == SPARK_STATUS_OK,"adopt the device view");
	CHECK(SparkDeviceAllocate(&readback,SPARK_DEVICE_SPACE_HOST_PINNED,granularity,"contract host share readback") == SPARK_STATUS_OK,"readback buffer");
	CHECK(SparkDeviceCopySync(At(&readback,0u),At(&view,0u),granularity) == SPARK_STATUS_OK && memcmp(readback.host,pattern,(size_t)granularity) == 0,
		"the device reads through the import what the holder wrote");
	CHECK(SparkDeviceFree(&readback) == SPARK_STATUS_OK && SparkDeviceFree(&view) == SPARK_STATUS_OK,"free the readback and the view");
	CHECK(SparkDeviceMemoryCreate(&device,granularity,"contract host share device") == SPARK_STATUS_OK,"create device memory");
	CHECK(SparkDeviceAddressReserve(&refused,granularity,"contract host share") == SPARK_STATUS_OK,"reserve a range for device memory");
	CHECK(SparkDeviceMemoryMap(refused,granularity,device,0u,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_HOST_READ) == SPARK_STATUS_INVALID_ARGUMENT &&
		strcmp(SparkDeviceLastError()->text,"host access maps only host memory") == 0,"host access to device memory is refused");
	CHECK(SparkDeviceMemoryUnmap(mapped,granularity) == SPARK_STATUS_OK && SparkDeviceMemoryUnmap(filled,granularity) == SPARK_STATUS_OK,"unmap");
	CHECK(SparkDeviceAddressRelease(mapped,granularity) == SPARK_STATUS_OK && SparkDeviceAddressRelease(filled,granularity) == SPARK_STATUS_OK &&
		SparkDeviceAddressRelease(refused,granularity) == SPARK_STATUS_OK,"release the ranges");
	CHECK(SparkDeviceMemoryRelease(imported) == SPARK_STATUS_OK && SparkDeviceMemoryRelease(host) == SPARK_STATUS_OK &&
		SparkDeviceMemoryRelease(device) == SPARK_STATUS_OK,"release the memory");
	free(pattern);
}

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
	launch.constant_bytes = SparkDeviceGetProperties()->max_constant_bytes + 4u;
	CHECK(SparkDeviceKernelLaunch(queue,writer,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"constants over the backend's limit are refused");
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


static void TestKernelPrograms(SparkDeviceQueue *queue, SparkDeviceKernel *write)
{
	enum { COUNT = 5000, WIDTH = 256 };
	if ( ProgramsRecord() == 0u )
		return;
	static const uint64_t values[][2] = {{11u, 5000u}, {977u, 1u}, {31337u, 4097u}};
	SparkDeviceBuffer program_out, eager_out;
	SparkDeviceBinding binding;
	SparkDeviceLaunch launch;
	SparkDeviceProgram *program = 0;
	SparkDeviceProgramInfo info;
	SparkDevicePatch patch;
	ContractParams params = {COUNT, 0u};
	uint32_t index, run, wrong = 0u;
	CHECK(SparkDeviceAllocate(&program_out,SPARK_DEVICE_SPACE_SHARED_COHERENT,COUNT * 4u,"program kernel out") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&eager_out,SPARK_DEVICE_SPACE_SHARED_COHERENT,COUNT * 4u,"eager kernel out") == SPARK_STATUS_OK,"allocate the outputs");
	binding = Bind(&program_out,0u,COUNT * 4u,SPARK_DEVICE_BIND_WRITE);
	launch = Launch1d(COUNT,WIDTH,&params,&binding,1u);
	CHECK(SparkDeviceProgramRecord(queue,"contract kernel program") == SPARK_STATUS_OK && SparkDeviceKernelLaunch(queue,write,&launch) == SPARK_STATUS_OK,
		"record a launch");
	memset(&patch,0,sizeof(patch));
	patch.value = 0u;
	patch.target = SPARK_DEVICE_PROGRAM_TARGET_CONSTANT;
	patch.offset = 4u;
	patch.bytes = 4u;
	CHECK(SparkDeviceProgramPatch(queue,&patch) == SPARK_STATUS_OK,"patch the seed");
	patch.value = 1u;
	patch.offset = 0u;
	CHECK(SparkDeviceProgramPatch(queue,&patch) == SPARK_STATUS_OK,"patch the count");
	patch.target = SPARK_DEVICE_PROGRAM_TARGET_GRID_X;
	patch.transform.multiply = 1;
	patch.transform.divide = WIDTH;
	patch.transform.scale = 1;
	CHECK(SparkDeviceProgramPatch(queue,&patch) == SPARK_STATUS_OK,"patch the grid width from the count");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK && SparkDeviceProgramGetInfo(program,&info) == SPARK_STATUS_OK &&
		info.kernels == 1u && info.values == 2u && info.places == 3u,"the program holds one kernel with three places for two values");
	for ( run = 0u; program != 0 && run < sizeof(values) / sizeof(values[0]); run++ )
	{
		memset(program_out.host,0,COUNT * 4u);
		memset(eager_out.host,0,COUNT * 4u);
		CHECK(SparkDeviceProgramRun(program,queue,values[run],2u) == (ProgramsRun() != 0u ? SPARK_STATUS_OK : SPARK_STATUS_UNSUPPORTED),
			"run the program with new patch values");
		params.seed = (uint32_t)values[run][0];
		params.count = (uint32_t)values[run][1];
		binding = Bind(&eager_out,0u,COUNT * 4u,SPARK_DEVICE_BIND_WRITE);
		launch = Launch1d(params.count,WIDTH,&params,&binding,1u);
		CHECK(SparkDeviceKernelLaunch(queue,write,&launch) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
			"launch the same values eagerly");
		for ( index = 0u; ProgramsRun() != 0u && index < COUNT; index++ )
			wrong += ((const uint32_t *)program_out.host)[index] != ((const uint32_t *)eager_out.host)[index];
	}
	CHECK(wrong == 0u,"every patched run writes the bytes the same launch writes eagerly");
	CHECK(SparkDeviceKernelDestroy(write) == SPARK_STATUS_BUSY,"a live program keeps the kernels it launches");
	CHECK(program != 0 && SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK,"destroy the kernel program");
	CHECK(SparkDeviceFree(&program_out) == SPARK_STATUS_OK && SparkDeviceFree(&eager_out) == SPARK_STATUS_OK,"free the outputs");
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

#ifndef SPARK_DEVICE_TEST_PARAMETER_KERNEL
#define SPARK_DEVICE_TEST_PARAMETER_KERNEL "SparkNvrmFirstLight"
#endif
#ifndef SPARK_DEVICE_TEST_LARGE_KERNEL
#define SPARK_DEVICE_TEST_LARGE_KERNEL "SparkNvrmLargeParameter"
#endif
#ifndef SPARK_DEVICE_TEST_HUGE_KERNEL
#define SPARK_DEVICE_TEST_HUGE_KERNEL "SparkNvrmHugeParameter"
#endif
#ifndef SPARK_DEVICE_TEST_LONG_KERNEL
#define SPARK_DEVICE_TEST_LONG_KERNEL ""
#endif

#define CONTRACT_LONG_NAME_BYTES 152u
#define CONTRACT_WORD_THREADS 1024u
#define CONTRACT_LONG_ROUNDS 1100u
#define CONTRACT_HUGE_ROUNDS 256u
#define CONTRACT_LARGE_BLOCK_BYTES (4096u + (uint32_t)sizeof(uint64_t))

typedef struct ParameterBlock
{
	uint64_t output;
	uint32_t value;
} ParameterBlock;

#define CONTRACT_HOLD_NS 300000000l

static void *GateOpenLater(void *context)
{
	struct timespec pause = {0, CONTRACT_HOLD_NS};
	nanosleep(&pause,0);
	GateOpen((Gate *)context);
	return(0);
}

static void HoldQueue(SparkDeviceQueue *queue, Gate *gate, pthread_t *opener)
{
	GateInit(gate);
	CHECK(SparkDeviceQueueHostCallback(queue,GateWaitCallback,gate) == SPARK_STATUS_OK && pthread_create(opener,0,GateOpenLater,gate) == 0,
		"hold the queue behind host work so launches pile up");
}

static void TestKernelNameLimit(void)
{
	char *name = (char *)malloc(SPARK_DEVICE_KERNEL_NAME_LIMIT + 2u);
	SparkDeviceKernel *kernel = (SparkDeviceKernel *)&kernel;
	char length[32];
	CHECK(name != 0,"allocate an over-long kernel name");
	if ( name == 0 )
		return;
	memset(name,'k',SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u);
	name[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u] = '\0';
	snprintf(length,sizeof(length),"%u bytes",SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u);
	CHECK(SparkDeviceKernelCreate(&kernel,0,name) == SPARK_STATUS_INVALID_ARGUMENT && kernel == 0 &&
		strstr(SparkDeviceLastError()->text,length) != 0 && strstr(SparkDeviceLastError()->text,"kkkkkkkk") != 0,
		"a kernel name over the limit is refused, naming its length and its first bytes");
	free(name);
}

static void TestLongKernelName(SparkDeviceQueue *queue, SparkDeviceLibrary *library)
{
	enum { THREADS = 128 };
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceBuffer output;
	SparkDeviceLaunch launch;
	ParameterBlock block;
	Gate gate;
	pthread_t opener;
	uint32_t index, round, launched = 0u, wrong = 0u;
	char caller[SPARK_DEVICE_KERNEL_NAME_LIMIT + 2u];
	SparkDeviceKernel *refused = 0;
	CHECK(strlen(SPARK_DEVICE_TEST_LONG_KERNEL) == CONTRACT_LONG_NAME_BYTES,"the build names a kernel whose mangled name is 152 characters long");
	if ( strlen(SPARK_DEVICE_TEST_LONG_KERNEL) != CONTRACT_LONG_NAME_BYTES )
		return;
	memset(caller,'k',SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u);
	caller[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u] = '\0';
	CHECK(SparkDeviceKernelCreate(&refused,library,caller) == SPARK_STATUS_INVALID_ARGUMENT && refused == 0 && strstr(SparkDeviceLastError()->text,"65 bytes") != 0,
		"a 65-character name that is no kernel's symbol is refused, naming its length");
	CHECK(SparkDeviceKernelCreate(&kernel,library,SPARK_DEVICE_TEST_LONG_KERNEL) == SPARK_STATUS_OK,
		"a kernel is created by its 152-character mangled symbol, which resolves through its short name");
	if ( kernel == 0 )
		return;
	CHECK(SparkDeviceKernelGetInfo(kernel,&info) == SPARK_STATUS_OK && info.parameter_bytes == 12u,"the long-named kernel reports its parameter block");
	CHECK(SparkDeviceAllocate(&output,SPARK_DEVICE_SPACE_SHARED_COHERENT,(uint64_t)CONTRACT_LONG_ROUNDS * THREADS * sizeof(uint32_t),"long name output") ==
		SPARK_STATUS_OK,"allocate the output");
	memset(&launch,0,sizeof(launch));
	launch.groups.x = launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = THREADS;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &block;
	launch.constant_bytes = info.parameter_bytes;
	HoldQueue(queue,&gate,&opener);
	for ( round = 0u; round < CONTRACT_LONG_ROUNDS; round++ )
	{
		memset(&block,0,sizeof(block));
		block.output = output.device + (uint64_t)round * THREADS * sizeof(uint32_t);
		block.value = round << 12;
		launched += SparkDeviceKernelLaunch(queue,kernel,&launch) == SPARK_STATUS_OK;
	}
	CHECK(launched == CONTRACT_LONG_ROUNDS && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"queue 1100 launches of the long-named kernel behind held host work");
	pthread_join(opener,0);
	for ( round = 0u; round < CONTRACT_LONG_ROUNDS; round++ )
		for ( index = 0u; index < THREADS; index++ )
			wrong += ((const uint32_t *)output.host)[round * THREADS + index] != (round << 12) + index;
	CHECK(wrong == 0u,"every launch of the long-named kernel ran with its own parameter block");
	CHECK(SparkDeviceKernelDestroy(kernel) == SPARK_STATUS_OK,"destroy the long-named kernel");
	CHECK(SparkDeviceFree(&output) == SPARK_STATUS_OK,"free the long name output");
}

static uint32_t ContractWord(uint32_t index, uint32_t round)
{
	return((0xc0de0000u ^ (index * 2654435761u)) + round);
}

static void TestWordBlock(SparkDeviceQueue *queue, SparkDeviceLibrary *library, const char *name, uint32_t pointer, uint32_t expected_bytes, uint32_t rounds)
{
	const uint32_t data = 1u - pointer;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceBuffer output;
	SparkDeviceLaunch launch;
	uint8_t *block;
	Gate gate;
	pthread_t opener;
	uint32_t index, round, words, launched = 0u, wrong = 0u;
	char what[192];
	snprintf(what,sizeof(what),"create %s",name);
	CHECK(SparkDeviceKernelCreate(&kernel,library,name) == SPARK_STATUS_OK,what);
	if ( kernel == 0 )
		return;
	snprintf(what,sizeof(what),"%s takes an output pointer and a word block, %u bytes in all, and runs %u threads",name,expected_bytes,CONTRACT_WORD_THREADS);
	CHECK(SparkDeviceKernelGetInfo(kernel,&info) == SPARK_STATUS_OK && info.parameter_count == 2u && info.parameter_bytes == expected_bytes &&
		info.parameters[pointer].bytes == sizeof(uint64_t) && info.parameters[data].bytes % sizeof(uint32_t) == 0u &&
		info.max_threads >= CONTRACT_WORD_THREADS,what);
	if ( info.parameter_count != 2u || info.parameter_bytes != expected_bytes )
	{
		SparkDeviceKernelDestroy(kernel);
		return;
	}
	words = info.parameters[data].bytes / (uint32_t)sizeof(uint32_t);
	block = (uint8_t *)calloc(1u,info.parameter_bytes);
	CHECK(block != 0 && SparkDeviceAllocate(&output,SPARK_DEVICE_SPACE_SHARED_COHERENT,(uint64_t)rounds * words * sizeof(uint32_t),"word block output") ==
		SPARK_STATUS_OK,"allocate the word block and its output");
	if ( block == 0 )
	{
		SparkDeviceKernelDestroy(kernel);
		return;
	}
	memset(&launch,0,sizeof(launch));
	launch.groups.x = launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = CONTRACT_WORD_THREADS;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = block;
	launch.constant_bytes = info.parameter_bytes;
	HoldQueue(queue,&gate,&opener);
	for ( round = 0u; round < rounds; round++ )
	{
		const uint64_t address = output.device + (uint64_t)round * words * sizeof(uint32_t);
		for ( index = 0u; index < words; index++ )
		{
			const uint32_t word = ContractWord(index,round);
			memcpy(block + info.parameters[data].offset + index * sizeof(uint32_t),&word,sizeof(word));
		}
		memcpy(block + info.parameters[pointer].offset,&address,sizeof(address));
		launched += SparkDeviceKernelLaunch(queue,kernel,&launch) == SPARK_STATUS_OK;
	}
	snprintf(what,sizeof(what),"queue %u launches of %s with its %u-byte parameter block behind held host work",rounds,name,info.parameter_bytes);
	CHECK(launched == rounds && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,what);
	pthread_join(opener,0);
	for ( round = 0u; round < rounds; round++ )
		for ( index = 0u; index < words; index++ )
			wrong += ((const uint32_t *)output.host)[(uint64_t)round * words + index] != ContractWord(index,round);
	snprintf(what,sizeof(what),"every word of every %u-byte parameter block reaches its launch",info.parameter_bytes);
	CHECK(wrong == 0u,what);
	launch.constant_bytes = SparkDeviceGetProperties()->max_constant_bytes + 4u;
	CHECK(SparkDeviceKernelLaunch(queue,kernel,&launch) == SPARK_STATUS_INVALID_ARGUMENT && strstr(SparkDeviceLastError()->text,"backend's") != 0,
		"a parameter block over the backend's limit is refused, naming the limit");
	CHECK(SparkDeviceKernelDestroy(kernel) == SPARK_STATUS_OK && SparkDeviceFree(&output) == SPARK_STATUS_OK,"destroy the word block kernel and free its output");
	free(block);
}


static void TestParameterBlockProgram(SparkDeviceQueue *queue, SparkDeviceKernel *kernel, SparkDeviceLaunch *launch, ParameterBlock *block)
{
	SparkDeviceProgram *program = 0;
	SparkDeviceProgramInfo info;
	SparkDevicePatch patch;
	const uint64_t value = 0x12340000u;
	CHECK(SparkDeviceProgramRecord(queue,"contract parameter program") == SPARK_STATUS_OK && SparkDeviceKernelLaunch(queue,kernel,launch) == SPARK_STATUS_OK,
		"record a parameter-block launch");
	memset(&patch,0,sizeof(patch));
	patch.target = SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT;
	patch.argument = 2u;
	CHECK(SparkDeviceProgramPatch(queue,&patch) == SPARK_STATUS_INVALID_ARGUMENT,"a patch names one of the kernel's parameters");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_INVALID_ARGUMENT && program == 0,"the refused patch breaks that recording");
	CHECK(SparkDeviceProgramRecord(queue,"contract parameter program") == SPARK_STATUS_OK && SparkDeviceKernelLaunch(queue,kernel,launch) == SPARK_STATUS_OK,
		"record the launch again");
	patch.argument = 1u;
	CHECK(SparkDeviceProgramPatch(queue,&patch) == SPARK_STATUS_OK,"patch the kernel's value argument");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK && SparkDeviceProgramGetInfo(program,&info) == SPARK_STATUS_OK &&
		info.kernels == 1u && info.values == 1u && info.places == 1u,"finish a one-kernel program with one patch");
	CHECK(SparkDeviceProgramRun(program,queue,&value,1u) == (ProgramsRun() != 0u ? SPARK_STATUS_OK : SPARK_STATUS_UNSUPPORTED),
		"a backend runs a program only when it reports that it runs programs");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK && SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK,
		"destroy the parameter-block program");
	(void)block;
}

static void TestParameterBlockKernels(SparkDeviceQueue *queue)
{
	enum { GROUPS = 4, THREADS = 128, WORDS = GROUPS * THREADS };
	SparkDeviceLibrary *library = 0;
	SparkDeviceKernel *kernel = 0, *missing = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceBuffer output;
	SparkDeviceBinding binding;
	SparkDeviceLaunch launch;
	ParameterBlock block;
	uint32_t index, wrong = 0u;
	CHECK(SparkDeviceLibraryLoad(&library,"/nonexistent/kernels.cubin","contract kernels") == SPARK_STATUS_NOT_FOUND && library == 0,"a missing library is refused");
	CHECK(SparkDeviceLibraryLoad(&library,SPARK_DEVICE_TEST_LIBRARY,"contract kernels") == SPARK_STATUS_OK,"load a kernel library with its manifest");
	CHECK(SparkDeviceKernelCreate(&missing,library,"contract_absent") == SPARK_STATUS_NOT_FOUND && missing == 0,"an absent kernel is refused");
	CHECK(SparkDeviceKernelCreate(&kernel,library,SPARK_DEVICE_TEST_PARAMETER_KERNEL) == SPARK_STATUS_OK,"create a kernel by its name");
	if ( kernel == 0 )
	{
		SparkDeviceLibraryUnload(library);
		return;
	}
	CHECK(SparkDeviceKernelGetInfo(kernel,&info) == SPARK_STATUS_OK && info.parameter_count == 2u && info.parameter_bytes == 12u &&
		info.parameters != 0 && info.parameters[0].offset == 0u && info.parameters[0].bytes == 8u && info.parameters[1].offset == 8u &&
		info.parameters[1].bytes == 4u && info.parameters[1].align == 4u,"the kernel reports its parameter layout");
	CHECK(SparkDeviceAllocate(&output,SPARK_DEVICE_SPACE_SHARED_COHERENT,WORDS * sizeof(uint32_t),"parameter block output") == SPARK_STATUS_OK,"allocate the output");
	memset(&block,0,sizeof(block));
	block.output = output.device;
	block.value = 0x5a5a0000u;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = GROUPS;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = THREADS;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &block;
	launch.constant_bytes = info.parameter_bytes;
	binding = Bind(&output,0u,WORDS * sizeof(uint32_t),SPARK_DEVICE_BIND_WRITE);
	launch.bindings = &binding;
	launch.binding_count = 1u;
	CHECK(SparkDeviceKernelLaunch(queue,kernel,&launch) == SPARK_STATUS_UNSUPPORTED,"a kernel that takes its arguments in its parameter block refuses bindings");
	launch.bindings = 0;
	launch.binding_count = 0u;
	launch.constant_bytes = info.parameter_bytes + 4u;
	CHECK(SparkDeviceKernelLaunch(queue,kernel,&launch) == SPARK_STATUS_INVALID_ARGUMENT,"a parameter block of the wrong size is refused");
	launch.constant_bytes = info.parameter_bytes;
	CHECK(SparkDeviceKernelLaunch(queue,kernel,&launch) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"launch the kernel with its parameter block");
	for ( index = 0u; index < WORDS; index++ )
		wrong += ((const uint32_t *)output.host)[index] != (block.value ^ index);
	CHECK(wrong == 0u,"every thread wrote what its parameter block asked for");
	TestParameterBlockProgram(queue,kernel,&launch,&block);
	TestLongKernelName(queue,library);
	TestWordBlock(queue,library,SPARK_DEVICE_TEST_LARGE_KERNEL,1u,CONTRACT_LARGE_BLOCK_BYTES,1u);
	TestWordBlock(queue,library,SPARK_DEVICE_TEST_HUGE_KERNEL,0u,SparkDeviceGetProperties()->max_constant_bytes,CONTRACT_HUGE_ROUNDS);
	CHECK(SparkDeviceLibraryUnload(library) == SPARK_STATUS_BUSY,"a library with a live kernel stays loaded");
	CHECK(SparkDeviceKernelDestroy(kernel) == SPARK_STATUS_OK && SparkDeviceLibraryUnload(library) == SPARK_STATUS_OK,"destroy the kernel, then unload");
	CHECK(SparkDeviceFree(&output) == SPARK_STATUS_OK,"free the output");
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
	TestKernelNameLimit();
	if ( (SparkDeviceGetProperties()->properties & SPARK_DEVICE_PROPERTY_KERNEL_LIBRARIES) != 0u &&
		(SparkDeviceGetProperties()->properties & SPARK_DEVICE_PROPERTY_PARAMETER_BLOCKS) != 0u )
	{
		TestParameterBlockKernels(queue);
		return;
	}
	if ( (SparkDeviceGetProperties()->properties & SPARK_DEVICE_PROPERTY_KERNEL_LIBRARIES) == 0u )
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
	if ( write == 0 || read == 0 || shared == 0 )
	{
		if ( write != 0 )
			SparkDeviceKernelDestroy(write);
		if ( read != 0 )
			SparkDeviceKernelDestroy(read);
		if ( shared != 0 )
			SparkDeviceKernelDestroy(shared);
		CHECK(SparkDeviceLibraryUnload(library) == SPARK_STATUS_OK,"unload the library its kernels could not be created from");
		return;
	}
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,4096u,"kernel device") == SPARK_STATUS_OK,"allocate");
	TestKernelRefusals(queue,write,&device);
	TestKernelOrdering(queue,write,read);
	TestKernelAcrossQueues(write,read);
	TestKernelBusy(queue,library,shared);
	TestKernelPrograms(queue,write);
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the device buffer");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_BUSY,"shutdown refuses while a kernel library is loaded");
	CHECK(SparkDeviceKernelDestroy(write) == SPARK_STATUS_OK && SparkDeviceKernelDestroy(read) == SPARK_STATUS_OK &&
		SparkDeviceKernelDestroy(shared) == SPARK_STATUS_OK,"destroy the kernels");
	CHECK(SparkDeviceKernelDestroy(write) == SPARK_STATUS_INVALID_ARGUMENT,"a destroyed kernel is refused");
	CHECK(SparkDeviceLibraryUnload(library) == SPARK_STATUS_OK,"unload the library");
}



static void TestProgramRefusals(SparkDeviceQueue *queue, SparkDeviceBuffer *target)
{
	SparkDeviceProgram *program = 0;
	SparkDevicePatch patch;
	memset(&patch,0,sizeof(patch));
	patch.target = SPARK_DEVICE_PROGRAM_TARGET_GRID_X;
	CHECK(SparkDeviceProgramPatch(queue,&patch) == SPARK_STATUS_INVALID_ARGUMENT,"a patch needs a recording queue");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_INVALID_ARGUMENT && program == 0,"finishing needs a recording queue");
	CHECK(SparkDeviceProgramRecord(queue,"contract refusals") == SPARK_STATUS_OK,"start a recording");
	CHECK(SparkDeviceProgramRecord(queue,"contract twice") == SPARK_STATUS_BUSY,"a queue records one program at a time");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_BUSY && SparkDeviceQueueQuery(queue) == SPARK_STATUS_BUSY &&
		SparkDeviceSynchronize(SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_BUSY,"a recording queue is neither waited on nor synchronized");
	CHECK(SparkDeviceFill(queue,At(target,0u),1u,64u) == SPARK_STATUS_OK,"record a fill");
	CHECK(SparkDeviceProgramPatch(queue,&patch) == SPARK_STATUS_INVALID_ARGUMENT,"a patch must follow a kernel launch");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_INVALID_ARGUMENT && program == 0,
		"a program with a refused step is refused at finish, so a broken program never runs");
	CHECK(SparkDeviceQueueQuery(queue) == SPARK_STATUS_OK,"the queue answers again once its recording ends");
}

static void TestPrograms(SparkDeviceQueue *queue)
{
	enum { BYTES = 4096, RUNS = 3 };
	SparkDeviceBuffer source, target;
	SparkDeviceProgram *program = 0;
	SparkDeviceProgramInfo info;
	SparkDeviceEvent *mark = 0;
	uint32_t calls = 0u, run, index, wrong = 0u;
	uint64_t value = 0u;
	if ( ProgramsRecord() == 0u )
	{
		CHECK(SparkDeviceProgramRecord(queue,"contract program") == SPARK_STATUS_UNSUPPORTED && strstr(SparkDeviceLastError()->text,"eagerly") != 0 &&
			SparkDeviceQueueQuery(queue) == SPARK_STATUS_OK,"a backend that records no programs refuses the recording, naming why, and its queue stays usable");
		return;
	}
	CHECK(SparkDeviceAllocate(&source,SPARK_DEVICE_SPACE_SHARED_COHERENT,BYTES,"program source") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&target,SPARK_DEVICE_SPACE_SHARED_COHERENT,BYTES,"program target") == SPARK_STATUS_OK &&
		SparkDeviceEventCreate(&mark) == SPARK_STATUS_OK,"allocate the program's buffers and event");
	TestProgramRefusals(queue,&target);
	Pattern((uint8_t *)source.host,BYTES,11u);
	memset(target.host,0xee,BYTES);
	CHECK(SparkDeviceProgramRecord(queue,"contract program") == SPARK_STATUS_OK,"start recording");
	CHECK(SparkDeviceFill(queue,At(&target,0u),0x5a,BYTES / 2u) == SPARK_STATUS_OK &&
		SparkDeviceCopy(queue,At(&target,BYTES / 2u),At(&source,0u),BYTES / 2u) == SPARK_STATUS_OK,"record a fill and a copy");
	CHECK(SparkDeviceFree(&source) == SPARK_STATUS_BUSY,"a buffer the recording uses is not freed under it");
	CHECK(SparkDeviceQueueHostCallback(queue,CountCallback,&calls) == SPARK_STATUS_OK && SparkDeviceEventRecord(mark,queue) == SPARK_STATUS_OK &&
		SparkDeviceQueueWaitEvent(queue,mark) == SPARK_STATUS_OK,"record a host callback, an event and a wait on it");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK && program != 0,"finish the program");
	CHECK(((const uint8_t *)target.host)[0] == 0xee && ((const uint8_t *)target.host)[BYTES - 1u] == 0xee && __atomic_load_n(&calls,__ATOMIC_SEQ_CST) == 0u,
		"recording runs nothing");
	CHECK(SparkDeviceProgramGetInfo(program,&info) == SPARK_STATUS_OK && info.nodes == 5u && info.kernels == 0u && info.values == 0u &&
		info.places == 0u,"the program holds the five recorded operations and no patch");
	CHECK(SparkDeviceFree(&target) == SPARK_STATUS_BUSY && SparkDeviceEventDestroy(mark) == SPARK_STATUS_BUSY,
		"a live program keeps the buffers and events it uses");
	CHECK(SparkDeviceProgramRun(program,queue,&value,1u) == SPARK_STATUS_INVALID_ARGUMENT,"a run passes exactly the program's patch values");
	if ( ProgramsRun() == 0u )
		CHECK(SparkDeviceProgramRun(program,queue,0,0u) == SPARK_STATUS_UNSUPPORTED,"a backend that does not run programs refuses the run");
	for ( run = 0u; ProgramsRun() != 0u && run < RUNS; run++ )
		CHECK(SparkDeviceProgramRun(program,queue,0,0u) == SPARK_STATUS_OK,"run the program");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for the runs");
	for ( index = 0u; ProgramsRun() != 0u && index < BYTES; index++ )
		wrong += ((const uint8_t *)target.host)[index] != (index < BYTES / 2u ? 0x5au : ((const uint8_t *)source.host)[index - BYTES / 2u]);
	CHECK(wrong == 0u && __atomic_load_n(&calls,__ATOMIC_SEQ_CST) == (ProgramsRun() != 0u ? RUNS : 0u),
		"each run does the recorded fill, copy and host callback");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_BUSY,"shutdown refuses while a program is live");
	CHECK(SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK && SparkDeviceProgramDestroy(program) == SPARK_STATUS_INVALID_ARGUMENT,
		"destroy the program once");
	CHECK(SparkDeviceEventDestroy(mark) == SPARK_STATUS_OK && SparkDeviceFree(&source) == SPARK_STATUS_OK && SparkDeviceFree(&target) == SPARK_STATUS_OK,
		"free the program's buffers and event");
}

static void TestTransforms(void)
{
	SparkDeviceTransform rows = {1, 0, 64u, 1, 0}, affine = {3, -2, 1u, 2, 5}, both = {2, 1, 8u, 3, 1}, negative = {1, -10, 1u, 1, 0};
	uint32_t fits = 0u;
	CHECK(SparkDeviceTransformApply(0,77u,&fits) == 77u && fits == 1u,"no transform keeps the value");
	CHECK(SparkDeviceTransformApply(&rows,1u,&fits) == 1u && fits == 1u && SparkDeviceTransformApply(&rows,64u,&fits) == 1u &&
		SparkDeviceTransformApply(&rows,65u,&fits) == 2u,"a grid width from rows is a ceiling division");
	CHECK(SparkDeviceTransformApply(&affine,4u,&fits) == 25u && fits == 1u,"value times a plus b, scaled and offset");
	CHECK(SparkDeviceTransformApply(&both,7u,&fits) == 7u && fits == 1u,"ceil((7 * 2 + 1) / 8) * 3 + 1 is 7");
	CHECK(SparkDeviceTransformApply(&negative,3u,&fits) == 0u && fits == 0u,"a negative intermediate does not fit");
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
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK )
	{
		printf("FAIL initialize: %s %s\n",SparkDeviceLastError()->operation,SparkDeviceLastError()->text);
		return 1;
	}
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_DUPLICATE,"one device per process");
	properties = SparkDeviceGetProperties();
	CHECK(properties != 0 && properties->backend[0] != '\0' && properties->page_bytes != 0u &&
		(properties->page_bytes & (properties->page_bytes - 1u)) == 0u,"properties name the backend and a power-of-two page");
	if ( properties != 0 )
		memcpy(backend,properties->backend,sizeof(backend));
	CHECK(SparkDeviceQueueCreate(&queue,"contract") == SPARK_STATUS_OK,"create a queue");
	CHECK(SparkDeviceQueueCapture(queue,1u) == SPARK_STATUS_UNSUPPORTED && SparkDeviceQueueQuery(queue) == SPARK_STATUS_OK,
		"a queue the device API created is never marked as recorded into a graph, and stays answerable");
	TestSpaces();
	TestAdopt();
	TestCopies(queue);
	TestFreshUpload(queue);
	TestOrdering();
	TestBusy(queue);
	TestEventLifetime();
	TestLongHold();
	TestRecordedEventDestroy(queue);
	TestTiming(queue);
	TestResolve();
	TestFileAndRegister(queue);
	TestTransforms();
	TestPrograms(queue);
	TestHostMemoryShare();
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
