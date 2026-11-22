#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"

#define DEVICE_PROPERTIES(APPLY) APPLY(UNIFIED_MEMORY) APPLY(SHARED_ADDRESS) APPLY(KERNEL_LIBRARIES) APPLY(PARAMETER_BLOCKS) APPLY(PROGRAMS) \
	APPLY(CLUSTERS) APPLY(TENSOR_MAPS) APPLY(HOST_MEMORY_SHARE) APPLY(NO_PAGE_FALLBACK)
#define DEVICE_PROPERTY_ONE_BIT(name) && SPARK_DEVICE_PROPERTY_##name != 0u && (SPARK_DEVICE_PROPERTY_##name & (SPARK_DEVICE_PROPERTY_##name - 1u)) == 0u
#define DEVICE_PROPERTY_OR(name) | SPARK_DEVICE_PROPERTY_##name
#define DEVICE_PROPERTY_SUM(name) + SPARK_DEVICE_PROPERTY_##name

_Static_assert(1 DEVICE_PROPERTIES(DEVICE_PROPERTY_ONE_BIT),"every device property is a single bit");
_Static_assert((0u DEVICE_PROPERTIES(DEVICE_PROPERTY_OR)) == (0u DEVICE_PROPERTIES(DEVICE_PROPERTY_SUM)),"no two device properties share a bit");
#define DEVICE_RECORDING_POLL_NS 50000

typedef struct SparkDeviceSlot
{
	SparkDeviceAllocation *allocation;
	uint32_t generation;
} SparkDeviceSlot;

static pthread_mutex_t device_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint32_t device_initialized;
static uint32_t device_faulted;
static SparkDeviceProperties device_properties;
static _Thread_local SparkDeviceError device_error;
static SparkDeviceSlot *device_slots;
static uint32_t device_slot_count;
static uint32_t device_slot_capacity;
static uint32_t *device_free_slots;
static uint32_t device_free_count;
static uint32_t *device_index;
static uint32_t device_index_count;
static uint32_t device_live_allocations;
static uint32_t device_live_events;
static SparkDeviceQueue *device_queues;
static SparkDeviceQueue *device_sync_queue;
static SparkDeviceLibrary *device_libraries;
static SparkDeviceKernel *device_kernels;
static SparkDeviceProgram *device_programs;
static uint64_t device_next_program_id = 1u;
static uint64_t device_next_queue_id = 1u;
static uint64_t device_next_event_id = 1u;
static uint64_t device_next_library_id = 1u;
static uint64_t device_next_kernel_id = 1u;

static void DeviceCopyText(char *destination, size_t capacity, const char *source)
{
	size_t length;
	if ( capacity == 0u )
		return;
	if ( source == 0 )
		source = "";
	length = strlen(source);
	if ( length >= capacity )
		length = capacity - 1u;
	memcpy(destination,source,length);
	destination[length] = '\0';
}

SparkStatus SparkDeviceRecordError(SparkStatus status, const char *operation, const char *owner, int32_t backend_code, const char *text)
{
	if ( status == SPARK_STATUS_OK )
		return(status);
	device_error.status = status;
	device_error.backend_code = backend_code;
	DeviceCopyText(device_error.operation,sizeof(device_error.operation),operation);
	DeviceCopyText(device_error.owner,sizeof(device_error.owner),owner);
	DeviceCopyText(device_error.text,sizeof(device_error.text),text);
	if ( status == SPARK_STATUS_IO_ERROR )
		__atomic_store_n(&device_faulted,1u,__ATOMIC_SEQ_CST);
	return(status);
}

const SparkDeviceError *SparkDeviceLastError(void)
{
	return(&device_error);
}

uint32_t SparkDeviceFaulted(void)
{
	return(__atomic_load_n(&device_faulted,__ATOMIC_SEQ_CST));
}

static SparkStatus DeviceReady(const char *operation, const char *owner)
{
	if ( device_initialized == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,operation,owner,0,"device is not initialized"));
	if ( SparkDeviceFaulted() != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation,owner,0,"device faulted earlier; every operation fails"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceInitialize(const SparkDeviceConfiguration *configuration)
{
	SparkDeviceConfiguration defaults;
	SparkStatus status;
	memset(&defaults,0,sizeof(defaults));
	pthread_mutex_lock(&device_mutex);
	if ( device_initialized != 0u )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_DUPLICATE,"initialize","",0,"one device per process"));
	}
	memset(&device_properties,0,sizeof(device_properties));
	status = SparkDeviceBackendInitialize(configuration != 0 ? configuration : &defaults,&device_properties);
	if ( status == SPARK_STATUS_OK )
	{
		__atomic_store_n(&device_initialized,1u,__ATOMIC_RELEASE);
		__atomic_store_n(&device_faulted,0u,__ATOMIC_SEQ_CST);
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static void DeviceDestroyQueueLocked(SparkDeviceQueue *queue);

SparkStatus SparkDeviceShutdown(void)
{
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	if ( device_initialized == 0u )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"shutdown","",0,"device is not initialized"));
	}
	if ( device_live_allocations != 0u || device_live_events != 0u || device_libraries != 0 || device_programs != 0 ||
		(device_queues != 0 && (device_queues != device_sync_queue || device_queues->next != 0)) )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_BUSY,"shutdown","",0,"buffers, queues, events or kernel libraries are still live"));
	}
	if ( device_sync_queue != 0 )
	{
		DeviceDestroyQueueLocked(device_sync_queue);
		device_sync_queue = 0;
	}
	status = SparkDeviceBackendShutdown();
	free(device_slots);
	free(device_free_slots);
	free(device_index);
	device_slots = 0;
	device_free_slots = 0;
	device_index = 0;
	device_slot_count = device_slot_capacity = device_free_count = device_index_count = 0u;
	__atomic_store_n(&device_initialized,0u,__ATOMIC_RELEASE);
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

const SparkDeviceProperties *SparkDeviceGetProperties(void)
{
	return(device_initialized != 0u ? &device_properties : 0);
}

static SparkDeviceQueue *DeviceQueueById(uint64_t id)
{
	SparkDeviceQueue *queue;
	for ( queue = device_queues; queue != 0; queue = queue->next )
		if ( queue->id == id )
			return(queue);
	return(0);
}

static uint32_t DeviceStampDone(const SparkDeviceStamp *stamp)
{
	SparkDeviceQueue *queue;
	if ( stamp->queue_id == 0u )
		return(1u);
	queue = DeviceQueueById(stamp->queue_id);
	if ( queue == 0 )
		return(1u);
	return(SparkDeviceBackendQueueCompleted(queue) >= stamp->sequence ? 1u : 0u);
}

static uint32_t DeviceStampsBusy(const SparkDeviceStamp *stamps)
{
	uint32_t index;
	for ( index = 0u; index < SPARK_DEVICE_STAMPS; index++ )
		if ( DeviceStampDone(&stamps[index]) == 0u )
			return(1u);
	return(0u);
}

static uint32_t DeviceAllocationBusy(const SparkDeviceAllocation *allocation)
{
	return(DeviceStampsBusy(allocation->stamps));
}

static int32_t DeviceStampsSlot(const SparkDeviceStamp *stamps, uint64_t queue_id)
{
	uint32_t index;
	for ( index = 0u; index < SPARK_DEVICE_STAMPS; index++ )
		if ( stamps[index].queue_id == queue_id )
			return((int32_t)index);
	for ( index = 0u; index < SPARK_DEVICE_STAMPS; index++ )
		if ( stamps[index].queue_id == 0u )
			return((int32_t)index);
	for ( index = 0u; index < SPARK_DEVICE_STAMPS; index++ )
		if ( DeviceStampDone(&stamps[index]) != 0u )
			return((int32_t)index);
	return(-1);
}

static int32_t DeviceStampSlot(const SparkDeviceAllocation *allocation, uint64_t queue_id)
{
	return(DeviceStampsSlot(allocation->stamps,queue_id));
}

static void DeviceStampAt(SparkDeviceStamp *stamps, int32_t slot, uint64_t queue_id, uint64_t sequence)
{
	stamps[slot].queue_id = queue_id;
	stamps[slot].sequence = sequence;
}

static void DeviceStamp(SparkDeviceAllocation *allocation, int32_t slot, uint64_t queue_id, uint64_t sequence)
{
	DeviceStampAt(allocation->stamps,slot,queue_id,sequence);
}

static uint64_t DeviceHandle(uint32_t slot)
{
	return(((uint64_t)device_slots[slot].generation << 32) | ((uint64_t)slot + 1u));
}

static SparkDeviceAllocation *DeviceAllocationOf(const SparkDeviceBuffer *buffer)
{
	uint64_t slot;
	uint32_t generation;
	SparkDeviceAllocation *allocation = 0;
	if ( buffer == 0 || buffer->handle == 0u )
		return(0);
	slot = (buffer->handle & 0xffffffffu) - 1u;
	generation = (uint32_t)(buffer->handle >> 32);
	if ( slot >= device_slot_count || device_slots[slot].generation != generation )
		return(0);
	allocation = device_slots[slot].allocation;
	if ( allocation == 0 || allocation->device != buffer->device || allocation->host != buffer->host ||
		allocation->bytes != buffer->bytes || allocation->space != buffer->space )
		return(0);
	return(allocation);
}

static uint32_t DeviceIndexFind(SparkDeviceAddress address)
{
	uint32_t low = 0u, high = device_index_count;
	while ( low < high )
	{
		uint32_t middle = low + (high - low) / 2u;
		if ( device_slots[device_index[middle]].allocation->device <= address )
			low = middle + 1u;
		else
			high = middle;
	}
	return(low);
}

static SparkStatus DeviceInsert(SparkDeviceAllocation *allocation, uint32_t *slot_out)
{
	uint32_t slot, position;
	if ( device_free_count != 0u )
		slot = device_free_slots[--device_free_count];
	else
	{
		if ( device_slot_count == device_slot_capacity )
		{
			uint32_t capacity = device_slot_capacity != 0u ? device_slot_capacity * 2u : 64u;
			SparkDeviceSlot *slots = (SparkDeviceSlot *)realloc(device_slots,capacity * sizeof(*slots));
			uint32_t *free_slots;
			uint32_t *index;
			if ( slots == 0 )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			device_slots = slots;
			free_slots = (uint32_t *)realloc(device_free_slots,capacity * sizeof(*free_slots));
			if ( free_slots == 0 )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			device_free_slots = free_slots;
			index = (uint32_t *)realloc(device_index,capacity * sizeof(*index));
			if ( index == 0 )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			device_index = index;
			memset(device_slots + device_slot_capacity,0,(capacity - device_slot_capacity) * sizeof(*slots));
			device_slot_capacity = capacity;
		}
		slot = device_slot_count++;
	}
	device_slots[slot].allocation = allocation;
	device_slots[slot].generation++;
	if ( device_slots[slot].generation == 0u )
		device_slots[slot].generation = 1u;
	if ( allocation->device != 0u )
	{
		position = DeviceIndexFind(allocation->device);
		memmove(device_index + position + 1u,device_index + position,(device_index_count - position) * sizeof(*device_index));
		device_index[position] = slot;
		device_index_count++;
	}
	device_live_allocations++;
	*slot_out = slot;
	return(SPARK_STATUS_OK);
}

static void DeviceRemove(SparkDeviceAllocation *allocation, uint32_t slot)
{
	uint32_t position;
	if ( allocation->device != 0u )
	{
		for ( position = 0u; position < device_index_count; position++ )
			if ( device_index[position] == slot )
				break;
		if ( position < device_index_count )
		{
			memmove(device_index + position,device_index + position + 1u,(device_index_count - position - 1u) * sizeof(*device_index));
			device_index_count--;
		}
	}
	device_slots[slot].allocation = 0;
	device_slots[slot].generation++;
	device_free_slots[device_free_count++] = slot;
	device_live_allocations--;
}

static uint32_t DeviceSpaceAccess(SparkDeviceSpace space)
{
	if ( space == SPARK_DEVICE_SPACE_DEVICE_PRIVATE )
		return(SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE);
	if ( space == SPARK_DEVICE_SPACE_HOST_PAGEABLE )
		return(SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE);
	if ( space == SPARK_DEVICE_SPACE_FILE_BACKED )
		return(SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_DEVICE_READ);
	return(SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE |
		SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE);
}

static SparkStatus DeviceAdmit(SparkDeviceBuffer *buffer, SparkDeviceAllocation *allocation, const char *operation)
{
	uint32_t slot;
	SparkStatus status = DeviceInsert(allocation,&slot);
	if ( status != SPARK_STATUS_OK )
	{
		status = SparkDeviceRecordError(status,operation,allocation->owner,0,"allocation table is full");
		SparkDeviceBackendRelease(allocation);
		free(allocation);
		return(status);
	}
	allocation->view.host = allocation->host;
	allocation->view.device = allocation->device;
	allocation->view.bytes = allocation->bytes;
	allocation->view.space = allocation->space;
	allocation->view.access = allocation->access;
	allocation->view.handle = DeviceHandle(slot);
	*buffer = allocation->view;
	return(SPARK_STATUS_OK);
}

static SparkDeviceAllocation *DeviceNewAllocation(SparkDeviceSpace space, uint64_t bytes, uint32_t access, const char *owner)
{
	SparkDeviceAllocation *allocation = (SparkDeviceAllocation *)calloc(1u,sizeof(*allocation));
	if ( allocation == 0 )
		return(0);
	allocation->space = space;
	allocation->bytes = bytes;
	allocation->access = access;
	DeviceCopyText(allocation->owner,sizeof(allocation->owner),owner);
	return(allocation);
}

SparkStatus SparkDeviceAllocate(SparkDeviceBuffer *buffer, SparkDeviceSpace space, uint64_t bytes, const char *owner)
{
	SparkDeviceAllocation *allocation = 0;
	SparkStatus status;
	if ( buffer == 0 || bytes == 0u || space < SPARK_DEVICE_SPACE_DEVICE_PRIVATE || space > SPARK_DEVICE_SPACE_HOST_PAGEABLE )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"allocate",owner,0,"needs a buffer, a size and a space other than file-backed"));
	memset(buffer,0,sizeof(*buffer));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("allocate",owner);
	if ( status == SPARK_STATUS_OK && bytes > device_properties.max_buffer_bytes )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,0,"larger than the device's largest buffer");
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_unlock(&device_mutex);
		return(status);
	}
	allocation = DeviceNewAllocation(space,bytes,DeviceSpaceAccess(space),owner);
	if ( allocation == 0 )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,0,"out of host memory for the allocation record"));
	}
	status = SparkDeviceBackendAllocate(allocation);
	if ( status != SPARK_STATUS_OK )
		free(allocation);
	else
		status = DeviceAdmit(buffer,allocation,"allocate");
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static SparkStatus DeviceAttach(SparkDeviceBuffer *buffer, SparkDeviceAllocation *allocation, const char *operation, const char *owner,
	SparkStatus (*attach)(SparkDeviceAllocation *allocation, void *host), void *host)
{
	SparkStatus status;
	memset(buffer,0,sizeof(*buffer));
	if ( allocation == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,owner,0,"out of host memory for the allocation record"));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady(operation,owner);
	if ( status == SPARK_STATUS_OK )
		status = attach(allocation,host);
	if ( status != SPARK_STATUS_OK )
		free(allocation);
	else
		status = DeviceAdmit(buffer,allocation,operation);
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static uint32_t DeviceOverlaps(SparkDeviceAddress device, uint64_t bytes)
{
	const uint32_t position = DeviceIndexFind(device);
	const SparkDeviceAllocation *before = position != 0u ? device_slots[device_index[position - 1u]].allocation : 0;
	const SparkDeviceAllocation *after = position < device_index_count ? device_slots[device_index[position]].allocation : 0;
	return((before != 0 && before->device + before->bytes > device) || (after != 0 && after->device < device + bytes) ? 1u : 0u);
}

static SparkStatus DeviceAdoptAttach(SparkDeviceAllocation *allocation, void *host)
{
	(void)host;
	if ( allocation->device != 0u && DeviceOverlaps(allocation->device,allocation->bytes) != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"adopt",allocation->owner,0,"the range overlaps a buffer the device already holds"));
	return(SparkDeviceBackendAdopt(allocation));
}

SparkStatus SparkDeviceRegister(SparkDeviceBuffer *buffer, void *host, uint64_t bytes, uint32_t access, const char *owner)
{
	SparkDeviceAllocation *allocation;
	if ( buffer == 0 || host == 0 || bytes == 0u ||
		(access & (SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE)) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"register",owner,0,"needs host memory, a size and device access"));
	allocation = DeviceNewAllocation(SPARK_DEVICE_SPACE_HOST_PINNED,bytes,
		access | SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE,owner);
	if ( allocation != 0 )
		allocation->registered = 1u;
	return(DeviceAttach(buffer,allocation,"register",owner,SparkDeviceBackendRegister,host));
}

SparkStatus SparkDeviceAdopt(SparkDeviceBuffer *buffer, SparkDeviceAddress device, void *host, uint64_t bytes, SparkDeviceSpace space,
	const char *owner)
{
	SparkDeviceAllocation *allocation;
	const uint32_t access = DeviceSpaceAccess(space);
	if ( buffer == 0 || bytes == 0u || space < SPARK_DEVICE_SPACE_DEVICE_PRIVATE || space > SPARK_DEVICE_SPACE_HOST_PINNED ||
		((access & SPARK_DEVICE_ACCESS_DEVICE_READ) != 0u && device == 0u) || ((access & SPARK_DEVICE_ACCESS_HOST_READ) != 0u && host == 0) )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"adopt",owner,0,
			"needs a size, a device, coherent or pinned space, and the addresses that space reaches"));
	allocation = DeviceNewAllocation(space,bytes,access,owner);
	if ( allocation != 0 )
	{
		allocation->device = device;
		allocation->host = host;
		allocation->registered = 2u;
	}
	return(DeviceAttach(buffer,allocation,"adopt",owner,DeviceAdoptAttach,host));
}

SparkStatus SparkDeviceMapFile(SparkDeviceBuffer *buffer, int fd, uint64_t offset, uint64_t bytes, const char *owner)
{
	return(SparkDeviceMapFileSegments(buffer,fd,offset,bytes,0u,owner));
}

SparkStatus SparkDeviceMapFileSegments(SparkDeviceBuffer *buffer, int fd, uint64_t offset, uint64_t bytes, uint64_t segment_stride, const char *owner)
{
	SparkDeviceAllocation *allocation = 0;
	SparkStatus status;
	if ( buffer == 0 || fd < 0 || bytes == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"map_file",owner,0,"needs an open file and a size"));
	memset(buffer,0,sizeof(*buffer));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("map_file",owner);
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_unlock(&device_mutex);
		return(status);
	}
	allocation = DeviceNewAllocation(SPARK_DEVICE_SPACE_FILE_BACKED,bytes,DeviceSpaceAccess(SPARK_DEVICE_SPACE_FILE_BACKED),owner);
	if ( allocation == 0 )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"map_file",owner,0,"out of host memory for the allocation record"));
	}
	allocation->segment_stride = segment_stride;
	status = SparkDeviceBackendMapFile(allocation,fd,offset);
	if ( status != SPARK_STATUS_OK )
		free(allocation);
	else
		status = DeviceAdmit(buffer,allocation,"map_file");
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceFree(SparkDeviceBuffer *buffer)
{
	SparkDeviceAllocation *allocation = 0;
	SparkStatus status;
	uint32_t slot;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("free",0);
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_unlock(&device_mutex);
		return(status);
	}
	allocation = DeviceAllocationOf(buffer);
	if ( allocation == 0 )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"free",0,0,"not a live buffer"));
	}
	if ( DeviceAllocationBusy(allocation) != 0u )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_BUSY,"free",allocation->owner,0,"a queued operation still uses the buffer"));
	}
	if ( allocation->programs != 0u )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_BUSY,"free",allocation->owner,0,"a program uses the buffer; destroy the program first"));
	}
	slot = (uint32_t)((buffer->handle & 0xffffffffu) - 1u);
	DeviceRemove(allocation,slot);
	SparkDeviceBackendRelease(allocation);
	free(allocation);
	memset(buffer,0,sizeof(*buffer));
	pthread_mutex_unlock(&device_mutex);
	return(SPARK_STATUS_OK);
}

static SparkStatus DeviceRange(const char *operation, SparkDeviceRange range, uint64_t bytes, SparkDeviceAllocation **out)
{
	SparkDeviceAllocation *allocation = DeviceAllocationOf(range.buffer);
	if ( allocation == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,operation,0,0,"range does not name a live buffer"));
	if ( bytes == 0u || range.offset > allocation->bytes || bytes > allocation->bytes - range.offset )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,operation,allocation->owner,0,"range runs past the end of its buffer"));
	*out = allocation;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceResolve(SparkDeviceAddress address, uint64_t bytes, SparkDeviceRange *range)
{
	uint32_t position;
	SparkDeviceAllocation *allocation = 0;
	if ( range == 0 || address == 0u || bytes == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"resolve",0,0,"needs an address, a size and a range"));
	pthread_mutex_lock(&device_mutex);
	position = device_initialized != 0u ? DeviceIndexFind(address) : 0u;
	if ( position == 0u )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"resolve",0,0,"no buffer holds the address"));
	}
	allocation = device_slots[device_index[position - 1u]].allocation;
	if ( address - allocation->device >= allocation->bytes )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"resolve",0,0,"no buffer holds the address"));
	}
	if ( bytes > allocation->bytes - (address - allocation->device) )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"resolve",allocation->owner,0,"range runs past the end of its buffer"));
	}
	range->buffer = &allocation->view;
	range->offset = address - allocation->device;
	pthread_mutex_unlock(&device_mutex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceAddressOf(SparkDeviceRange range, uint64_t bytes, SparkDeviceAddress *address)
{
	SparkDeviceAllocation *allocation = 0;
	SparkStatus status;
	if ( address == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"address_of",0,0,"needs an output"));
	pthread_mutex_lock(&device_mutex);
	status = DeviceRange("address_of",range,bytes,&allocation);
	if ( status == SPARK_STATUS_OK && allocation->device == 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"address_of",allocation->owner,0,"buffer is not device visible");
	if ( status == SPARK_STATUS_OK )
		*address = allocation->device + range.offset;
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceHostOf(SparkDeviceRange range, uint64_t bytes, void **host)
{
	SparkDeviceAllocation *allocation = 0;
	SparkStatus status;
	if ( host == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"host_of",0,0,"needs an output"));
	pthread_mutex_lock(&device_mutex);
	status = DeviceRange("host_of",range,bytes,&allocation);
	if ( status == SPARK_STATUS_OK && allocation->host == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"host_of",allocation->owner,0,"buffer is not host visible");
	if ( status == SPARK_STATUS_OK )
		*host = (uint8_t *)allocation->host + range.offset;
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

uint32_t SparkDeviceSegmented(SparkDeviceRange range)
{
	SparkDeviceAllocation *allocation;
	uint32_t segmented;
	pthread_mutex_lock(&device_mutex);
	allocation = DeviceAllocationOf(range.buffer);
	segmented = allocation != 0 && allocation->segment_stride != 0u ? 1u : 0u;
	pthread_mutex_unlock(&device_mutex);
	return(segmented);
}

static SparkStatus DeviceSharedReady(const char *operation, const char *owner)
{
	if ( __atomic_load_n(&device_initialized,__ATOMIC_ACQUIRE) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,operation,owner,0,"device is not initialized"));
	if ( SparkDeviceFaulted() != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation,owner,0,"device faulted earlier; every operation fails"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceThreadInstance(SparkDeviceInstance *instance)
{
	SparkStatus status;
	if ( instance == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"thread_instance","",0,"needs an output"));
	*instance = 0u;
	status = DeviceSharedReady("thread_instance","");
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendThreadInstance(instance));
}

SparkStatus SparkDeviceThreadBind(SparkDeviceInstance instance)
{
	SparkStatus status;
	if ( instance == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"thread_bind","",0,"needs an instance that SparkDeviceThreadInstance returned"));
	status = DeviceSharedReady("thread_bind","");
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendThreadBind(instance));
}

SparkStatus SparkDeviceMemoryGranularity(uint32_t granularity, uint64_t *bytes)
{
	SparkStatus status;
	if ( bytes != 0 )
		*bytes = 0u;
	if ( bytes == 0 || (granularity != SPARK_DEVICE_GRANULARITY_MINIMUM && granularity != SPARK_DEVICE_GRANULARITY_RECOMMENDED) )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_granularity","",0,"needs an output and the minimum or recommended granularity"));
	status = DeviceSharedReady("memory_granularity","");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceBackendMemoryGranularity(granularity,bytes);
	if ( status == SPARK_STATUS_OK && (*bytes == 0u || (*bytes & (*bytes - 1u)) != 0u) )
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"memory_granularity","",0,"the backend reported a granularity that is not a power of two");
	return(status);
}

static SparkStatus DeviceHostShareRefused(const char *operation, const char *owner)
{
	char text[SPARK_DEVICE_ERROR_TEXT_BYTES];
	snprintf(text,sizeof(text),"the %s backend has no exportable host memory",device_properties.backend);
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,operation,owner,0,text));
}

SparkStatus SparkDeviceMemoryCreateHost(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner)
{
	SparkStatus status;
	if ( memory != 0 )
		*memory = 0u;
	if ( memory == 0 || bytes == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_create_host",owner,0,"needs an output and a size"));
	status = DeviceSharedReady("memory_create_host",owner);
	if ( status == SPARK_STATUS_OK && (device_properties.properties & SPARK_DEVICE_PROPERTY_HOST_MEMORY_SHARE) == 0u )
		status = DeviceHostShareRefused("memory_create_host",owner);
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendMemoryCreateHost(memory,bytes,owner));
}

SparkStatus SparkDeviceMemoryCreate(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner)
{
	SparkStatus status;
	if ( memory != 0 )
		*memory = 0u;
	if ( memory == 0 || bytes == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_create",owner,0,"needs an output and a size"));
	status = DeviceSharedReady("memory_create",owner);
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendMemoryCreate(memory,bytes,owner));
}

SparkStatus SparkDeviceMemoryRelease(SparkDeviceMemoryHandle memory)
{
	SparkStatus status;
	if ( memory == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_release","",0,"needs a memory handle"));
	status = DeviceSharedReady("memory_release","");
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendMemoryRelease(memory));
}

SparkStatus SparkDeviceMemoryExport(SparkDeviceMemoryHandle memory, int *fd)
{
	SparkStatus status;
	if ( fd != 0 )
		*fd = -1;
	if ( memory == 0u || fd == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_export","",0,"needs a memory handle and an output"));
	status = DeviceSharedReady("memory_export","");
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendMemoryExport(memory,fd));
}

SparkStatus SparkDeviceMemoryImport(SparkDeviceMemoryHandle *memory, int fd, const char *owner)
{
	SparkStatus status;
	if ( memory != 0 )
		*memory = 0u;
	if ( memory == 0 || fd < 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_import",owner,0,"needs an output and an exported file descriptor"));
	status = DeviceSharedReady("memory_import",owner);
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendMemoryImport(memory,fd,owner));
}

SparkStatus SparkDeviceAddressReserve(SparkDeviceAddress *address, uint64_t bytes, const char *owner)
{
	SparkStatus status;
	if ( address != 0 )
		*address = 0u;
	if ( address == 0 || bytes == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"address_reserve",owner,0,"needs an output and a size"));
	status = DeviceSharedReady("address_reserve",owner);
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendAddressReserve(address,bytes,owner));
}

SparkStatus SparkDeviceAddressRelease(SparkDeviceAddress address, uint64_t bytes)
{
	SparkStatus status;
	if ( address == 0u || bytes == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"address_release","",0,"needs a reserved range"));
	status = DeviceSharedReady("address_release","");
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendAddressRelease(address,bytes));
}

SparkStatus SparkDeviceMemoryMap(SparkDeviceAddress address, uint64_t bytes, SparkDeviceMemoryHandle memory, uint64_t offset, uint32_t access)
{
	const uint32_t device_access = SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE;
	const uint32_t host_access = SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE;
	SparkStatus status;
	if ( (access & SPARK_DEVICE_ACCESS_HOST_WRITE) != 0u && (access & SPARK_DEVICE_ACCESS_HOST_READ) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"host write access needs host read access"));
	if ( address == 0u || bytes == 0u || memory == 0u || (access & ~(device_access | host_access)) != 0u ||
		(access & (SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_HOST_READ)) == 0u ||
		((access & SPARK_DEVICE_ACCESS_DEVICE_WRITE) != 0u && (access & SPARK_DEVICE_ACCESS_DEVICE_READ) == 0u) )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"needs a reserved range, a memory handle and device read or read-write access"));
	status = DeviceSharedReady("memory_map","");
	if ( status == SPARK_STATUS_OK && (access & host_access) != 0u && (device_properties.properties & SPARK_DEVICE_PROPERTY_HOST_MEMORY_SHARE) == 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"host access maps only host memory");
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendMemoryMap(address,bytes,memory,offset,access));
}

SparkStatus SparkDeviceMemoryUnmap(SparkDeviceAddress address, uint64_t bytes)
{
	SparkStatus status;
	if ( address == 0u || bytes == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_unmap","",0,"needs a mapped range"));
	status = DeviceSharedReady("memory_unmap","");
	return(status != SPARK_STATUS_OK ? status : SparkDeviceBackendMemoryUnmap(address,bytes));
}

static uint32_t DeviceDeviceSide(const SparkDeviceAllocation *allocation)
{
	return(allocation->space == SPARK_DEVICE_SPACE_DEVICE_PRIVATE ? 1u : 0u);
}

static uint32_t DeviceWritable(const SparkDeviceAllocation *allocation)
{
	return((allocation->access & (SPARK_DEVICE_ACCESS_HOST_WRITE | SPARK_DEVICE_ACCESS_DEVICE_WRITE)) != 0u ? 1u : 0u);
}

static SparkStatus DeviceQueueValid(SparkDeviceQueue *queue, const char *operation)
{
	if ( queue == 0 || queue->id == 0u || DeviceQueueById(queue->id) != queue )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,operation,0,0,"needs a live queue"));
	return(SPARK_STATUS_OK);
}

static uint32_t DeviceRecordingElsewhere(const SparkDeviceQueue *queue)
{
	return(queue->recording != 0 && pthread_equal(queue->recording->thread,pthread_self()) == 0 ? 1u : 0u);
}

static uint32_t DeviceAnyRecordingElsewhere(void)
{
	const SparkDeviceQueue *queue;
	for ( queue = device_queues; queue != 0; queue = queue->next )
		if ( DeviceRecordingElsewhere(queue) != 0u )
			return(1u);
	return(0u);
}

static SparkStatus DeviceAwaitRecordings(SparkDeviceQueue *queue, uint64_t timeout_ns, const char *operation)
{
	struct timespec now;
	uint64_t started = 0u,elapsed;
	SparkStatus status = queue != 0 ? DeviceQueueValid(queue,operation) : SPARK_STATUS_OK;
	while ( status == SPARK_STATUS_OK && (queue != 0 ? DeviceRecordingElsewhere(queue) : DeviceAnyRecordingElsewhere()) != 0u )
	{
		const struct timespec pause = {0, DEVICE_RECORDING_POLL_NS};
		clock_gettime(CLOCK_MONOTONIC,&now);
		elapsed = (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
		if ( started == 0u )
			started = elapsed;
		if ( timeout_ns != SPARK_DEVICE_WAIT_FOREVER && elapsed - started >= timeout_ns )
			return(SPARK_STATUS_PENDING);
		pthread_mutex_unlock(&device_mutex);
		nanosleep(&pause,0);
		pthread_mutex_lock(&device_mutex);
		status = DeviceReady(operation,0);
		if ( status == SPARK_STATUS_OK && queue != 0 )
			status = DeviceQueueValid(queue,operation);
	}
	return(status);
}

static SparkStatus DeviceQueueSettled(SparkDeviceQueue *queue, const char *operation)
{
	SparkStatus status = DeviceQueueValid(queue,operation);
	if ( status == SPARK_STATUS_OK && queue->recording != 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_BUSY,operation,queue->owner,0,"the queue is recording a program; finish the program first");
	return(status);
}

static void DevicePinBuffer(const SparkDeviceBuffer *buffer, int32_t delta)
{
	SparkDeviceAllocation *allocation = DeviceAllocationOf(buffer);
	if ( allocation != 0 )
		allocation->programs = (uint32_t)((int32_t)allocation->programs + delta);
}

static void DeviceNodePins(const SparkDeviceNode *node, int32_t delta)
{
	uint32_t index;
	switch ( node->kind )
	{
	case SPARK_DEVICE_NODE_COPY:
		DevicePinBuffer(&node->buffers[1],delta);
		DevicePinBuffer(&node->buffers[0],delta);
		break;
	case SPARK_DEVICE_NODE_FILL:
		DevicePinBuffer(&node->buffers[0],delta);
		break;
	case SPARK_DEVICE_NODE_LAUNCH:
		((SparkDeviceKernel *)node->kernel)->programs = (uint32_t)((int32_t)node->kernel->programs + delta);
		for ( index = 0u; index < node->launch.binding_count; index++ )
			DevicePinBuffer(node->launch.bindings[index].range.buffer,delta);
		break;
	case SPARK_DEVICE_NODE_RECORD:
	case SPARK_DEVICE_NODE_AWAIT:
		node->event->programs = (uint32_t)((int32_t)node->event->programs + delta);
		break;
	default:
		break;
	}
}

static void DeviceProgramRelease(SparkDeviceProgram *program)
{
	uint32_t index;
	for ( index = 0u; index < program->node_count; index++ )
		DeviceNodePins(&program->nodes[index],-1);
	SparkDeviceProgramFree(program);
}

static SparkStatus DeviceRecorded(SparkDeviceQueue *queue, const SparkDeviceNode *node, const char *operation)
{
	SparkDeviceProgram *program = queue->recording;
	const SparkStatus status = SparkDeviceProgramAppend(program,node,operation);
	if ( status != SPARK_STATUS_OK )
		program->broken = 1u;
	else
		DeviceNodePins(&program->nodes[program->node_count - 1u],1);
	return(status);
}

static SparkStatus DeviceCaptured(SparkDeviceQueue *queue, SparkStatus status)
{
	if ( status != SPARK_STATUS_OK )
		queue->recording->broken = 1u;
	return(status);
}

static SparkStatus DeviceStampPair(SparkDeviceAllocation *first, SparkDeviceAllocation *second, uint64_t queue_id,
	int32_t *first_slot, int32_t *second_slot, const char *operation)
{
	*first_slot = DeviceStampSlot(first,queue_id);
	*second_slot = second != 0 ? DeviceStampSlot(second,queue_id) : 0;
	if ( *first_slot < 0 || *second_slot < 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,first->owner,0,"buffer is in flight on too many queues"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceCopy(SparkDeviceQueue *queue, SparkDeviceRange destination, SparkDeviceRange source, uint64_t bytes)
{
	SparkDeviceAllocation *target = 0, *origin = 0;
	SparkDeviceDirection direction;
	int32_t target_slot, origin_slot;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("copy",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"copy");
	if ( status == SPARK_STATUS_OK )
		status = DeviceRange("copy",destination,bytes,&target);
	if ( status == SPARK_STATUS_OK )
		status = DeviceRange("copy",source,bytes,&origin);
	if ( status == SPARK_STATUS_OK && DeviceWritable(target) == 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"copy",target->owner,0,"destination is read only");
	if ( status == SPARK_STATUS_OK && target == origin &&
		destination.offset < source.offset + bytes && source.offset < destination.offset + bytes )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"copy",target->owner,0,"source and destination overlap");
	if ( status == SPARK_STATUS_OK )
		status = DeviceStampPair(target,origin,queue->id,&target_slot,&origin_slot,"copy");
	if ( status == SPARK_STATUS_OK )
	{
		if ( DeviceDeviceSide(origin) != 0u )
			direction = DeviceDeviceSide(target) != 0u ? SPARK_DEVICE_DEVICE_TO_DEVICE : SPARK_DEVICE_DEVICE_TO_HOST;
		else
			direction = DeviceDeviceSide(target) != 0u ? SPARK_DEVICE_HOST_TO_DEVICE : SPARK_DEVICE_HOST_TO_HOST;
	}
	if ( status == SPARK_STATUS_OK && queue->recording != 0 )
	{
		SparkDeviceNode node;
		memset(&node,0,sizeof(node));
		node.kind = SPARK_DEVICE_NODE_COPY;
		node.buffers[0] = *destination.buffer;
		node.offsets[0] = destination.offset;
		node.buffers[1] = *source.buffer;
		node.offsets[1] = source.offset;
		node.bytes = bytes;
		status = DeviceRecorded(queue,&node,"copy");
		if ( status == SPARK_STATUS_OK && queue->recording->native != 0u )
			status = DeviceCaptured(queue,SparkDeviceBackendCopy(queue,direction,target,destination.offset,origin,source.offset,bytes));
	}
	else if ( status == SPARK_STATUS_OK )
	{
		queue->submitted++;
		status = SparkDeviceBackendCopy(queue,direction,target,destination.offset,origin,source.offset,bytes);
		if ( status != SPARK_STATUS_OK )
			queue->submitted--;
		else
		{
			DeviceStamp(target,target_slot,queue->id,queue->submitted);
			DeviceStamp(origin,origin_slot,queue->id,queue->submitted);
		}
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceFill(SparkDeviceQueue *queue, SparkDeviceRange destination, uint8_t value, uint64_t bytes)
{
	SparkDeviceAllocation *target = 0;
	int32_t target_slot, unused_slot;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("fill",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"fill");
	if ( status == SPARK_STATUS_OK )
		status = DeviceRange("fill",destination,bytes,&target);
	if ( status == SPARK_STATUS_OK && DeviceWritable(target) == 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"fill",target->owner,0,"destination is read only");
	if ( status == SPARK_STATUS_OK )
		status = DeviceStampPair(target,0,queue->id,&target_slot,&unused_slot,"fill");
	if ( status == SPARK_STATUS_OK && queue->recording != 0 )
	{
		SparkDeviceNode node;
		memset(&node,0,sizeof(node));
		node.kind = SPARK_DEVICE_NODE_FILL;
		node.buffers[0] = *destination.buffer;
		node.offsets[0] = destination.offset;
		node.value = value;
		node.bytes = bytes;
		status = DeviceRecorded(queue,&node,"fill");
		if ( status == SPARK_STATUS_OK && queue->recording->native != 0u )
			status = DeviceCaptured(queue,SparkDeviceBackendFill(queue,target,destination.offset,value,bytes));
	}
	else if ( status == SPARK_STATUS_OK )
	{
		queue->submitted++;
		status = SparkDeviceBackendFill(queue,target,destination.offset,value,bytes);
		if ( status != SPARK_STATUS_OK )
			queue->submitted--;
		else
			DeviceStamp(target,target_slot,queue->id,queue->submitted);
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static SparkStatus DeviceCreateQueueLocked(SparkDeviceQueue **out, const char *owner, void *native)
{
	SparkDeviceQueue *queue = (SparkDeviceQueue *)calloc(1u,sizeof(*queue));
	SparkStatus status;
	if ( queue == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",owner,0,"out of host memory for the queue"));
	queue->id = device_next_queue_id++;
	DeviceCopyText(queue->owner,sizeof(queue->owner),owner);
	status = native != 0 ? SparkDeviceBackendQueueAdopt(queue,native) : SparkDeviceBackendQueueCreate(queue);
	if ( status != SPARK_STATUS_OK )
	{
		free(queue);
		return(status);
	}
	queue->next = device_queues;
	device_queues = queue;
	*out = queue;
	return(SPARK_STATUS_OK);
}

static void DeviceDestroyQueueLocked(SparkDeviceQueue *queue)
{
	SparkDeviceQueue **link;
	for ( link = &device_queues; *link != 0; link = &(*link)->next )
		if ( *link == queue )
		{
			*link = queue->next;
			break;
		}
	SparkDeviceBackendQueueDestroy(queue);
	queue->id = 0u;
	free(queue);
}

SparkStatus SparkDeviceCopySync(SparkDeviceRange destination, SparkDeviceRange source, uint64_t bytes)
{
	SparkStatus status = SPARK_STATUS_OK;
	SparkDeviceQueue *queue;
	pthread_mutex_lock(&device_mutex);
	if ( device_initialized != 0u && device_sync_queue == 0 )
		status = DeviceCreateQueueLocked(&device_sync_queue,"synchronous copy",0);
	queue = device_sync_queue;
	pthread_mutex_unlock(&device_mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDeviceCopy(queue,destination,source,bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER));
}

static SparkStatus DeviceQueueOpen(SparkDeviceQueue **queue, const char *owner, void *native)
{
	SparkStatus status;
	if ( queue == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"queue_create",owner,0,"needs an output"));
	*queue = 0;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("queue_create",owner);
	if ( status == SPARK_STATUS_OK )
		status = DeviceCreateQueueLocked(queue,owner,native);
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceQueueCreate(SparkDeviceQueue **queue, const char *owner)
{
	return(DeviceQueueOpen(queue,owner,0));
}

static SparkDeviceQueue *DeviceQueueListed(const void *native)
{
	SparkDeviceQueue *queue;
	for ( queue = device_queues; queue != 0; queue = queue->next )
		if ( (const void *)queue == native )
			return(queue);
	return(0);
}

SparkStatus SparkDeviceQueueAdopt(SparkDeviceQueue **queue, void *native, const char *owner)
{
	SparkDeviceQueue *listed = 0;
	SparkStatus status;
	if ( native == 0 || queue == 0 )
	{
		const SparkStatus refused = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"queue_adopt",owner,0,"needs an output and the backend's own queue to adopt");
		SPARK_FAIL(refused);
	}
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("queue_adopt",owner);
	if ( status == SPARK_STATUS_OK && (listed = DeviceQueueListed(native)) != 0 )
	{
		if ( listed == device_sync_queue )
			status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"queue_adopt",owner,0,"the synchronous copy queue belongs to the device");
		else
		{
			listed->borrows++;
			*queue = listed;
		}
	}
	pthread_mutex_unlock(&device_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( listed != 0 )
		return(SPARK_STATUS_OK);
	return(DeviceQueueOpen(queue,owner,native));
}

SparkStatus SparkDeviceQueueWait(SparkDeviceQueue *queue, uint64_t timeout_ns)
{
	uint64_t sequence;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("queue_wait",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceAwaitRecordings(queue,timeout_ns,"queue_wait");
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueSettled(queue,"queue_wait");
	sequence = status == SPARK_STATUS_OK ? queue->submitted : 0u;
	pthread_mutex_unlock(&device_mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkDeviceBackendQueueWait(queue,sequence,timeout_ns));
}

SparkStatus SparkDeviceQueueQuery(SparkDeviceQueue *queue)
{
	return(SparkDeviceQueueWait(queue,0u));
}

SparkStatus SparkDeviceQueueCapture(SparkDeviceQueue *queue, uint32_t capturing)
{
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("queue_capture",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueSettled(queue,"queue_capture");
	pthread_mutex_unlock(&device_mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkDeviceBackendQueueCapture(queue,capturing != 0u ? 1u : 0u));
}

SparkStatus SparkDeviceQueueDestroy(SparkDeviceQueue *queue)
{
	SparkStatus status;
	uint32_t borrowed = 0u;
	pthread_mutex_lock(&device_mutex);
	if ( DeviceQueueListed(queue) != 0 && queue->borrows != 0u )
	{
		queue->borrows--;
		borrowed = 1u;
	}
	pthread_mutex_unlock(&device_mutex);
	if ( borrowed != 0u )
		return(SPARK_STATUS_OK);
	status = SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pthread_mutex_lock(&device_mutex);
	if ( queue == device_sync_queue )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"queue_destroy",queue->owner,0,"the synchronous copy queue belongs to the device");
	else
		DeviceDestroyQueueLocked(queue);
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context)
{
	SparkStatus status;
	if ( function == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"host_callback",0,0,"needs a function"));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("host_callback",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"host_callback");
	if ( status == SPARK_STATUS_OK && queue->recording != 0 )
	{
		SparkDeviceNode node;
		memset(&node,0,sizeof(node));
		node.kind = SPARK_DEVICE_NODE_HOST;
		node.function = function;
		node.context = context;
		status = DeviceRecorded(queue,&node,"host_callback");
		if ( status == SPARK_STATUS_OK && queue->recording->native != 0u )
			status = DeviceCaptured(queue,SparkDeviceBackendQueueHostCallback(queue,function,context));
	}
	else if ( status == SPARK_STATUS_OK )
	{
		queue->submitted++;
		status = SparkDeviceBackendQueueHostCallback(queue,function,context);
		if ( status != SPARK_STATUS_OK )
			queue->submitted--;
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceSynchronize(uint64_t timeout_ns)
{
	SparkDeviceQueue **queues = 0;
	uint64_t *sequences = 0;
	uint32_t count = 0u, index;
	SparkDeviceQueue *queue;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("synchronize",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceAwaitRecordings(0,timeout_ns,"synchronize");
	for ( queue = device_queues; status == SPARK_STATUS_OK && queue != 0; queue = queue->next )
	{
		if ( queue->recording != 0 )
			status = SparkDeviceRecordError(SPARK_STATUS_BUSY,"synchronize",queue->owner,0,"a queue is recording a program; finish the program first");
		count++;
	}
	if ( status == SPARK_STATUS_OK && count != 0u )
	{
		queues = (SparkDeviceQueue **)calloc(count,sizeof(*queues));
		sequences = (uint64_t *)calloc(count,sizeof(*sequences));
		if ( queues == 0 || sequences == 0 )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"synchronize",0,0,"out of host memory");
		for ( queue = device_queues, index = 0u; status == SPARK_STATUS_OK && queue != 0; queue = queue->next, index++ )
		{
			queues[index] = queue;
			sequences[index] = queue->submitted;
		}
	}
	pthread_mutex_unlock(&device_mutex);
	for ( index = 0u; status == SPARK_STATUS_OK && index < count; index++ )
		status = SparkDeviceBackendQueueWait(queues[index],sequences[index],timeout_ns);
	free(queues);
	free(sequences);
	return(status);
}

SparkStatus SparkDeviceEventCreate(SparkDeviceEvent **event)
{
	SparkDeviceEvent *created;
	SparkStatus status;
	if ( event == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"event_create",0,0,"needs an output"));
	*event = 0;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("event_create",0);
	created = status == SPARK_STATUS_OK ? (SparkDeviceEvent *)calloc(1u,sizeof(*created)) : 0;
	if ( status == SPARK_STATUS_OK && created == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"event_create",0,0,"out of host memory for the event");
	if ( status == SPARK_STATUS_OK )
	{
		created->id = device_next_event_id++;
		status = SparkDeviceBackendEventCreate(created);
		if ( status != SPARK_STATUS_OK )
			free(created);
		else
		{
			device_live_events++;
			*event = created;
		}
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceEventDestroy(SparkDeviceEvent *event)
{
	SparkStatus status;
	if ( event == 0 || event->id == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"event_destroy",0,0,"needs a live event"));
	pthread_mutex_lock(&device_mutex);
	if ( DeviceStampsBusy(event->stamps) != 0u )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_BUSY,"event_destroy",0,0,"a queue that records or waits on the event has not passed it"));
	}
	if ( event->programs != 0u )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_BUSY,"event_destroy",0,0,"a program records or waits on the event; destroy the program first"));
	}
	SparkDeviceBackendEventDestroy(event);
	event->id = 0u;
	device_live_events--;
	free(event);
	status = SPARK_STATUS_OK;
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue)
{
	SparkStatus status;
	int32_t slot = -1;
	if ( event == 0 || event->id == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"event_record",0,0,"needs a live event"));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("event_record",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"event_record");
	if ( status == SPARK_STATUS_OK && (slot = DeviceStampsSlot(event->stamps,queue->id)) < 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"event_record",queue->owner,0,"the event is in flight on too many queues");
	if ( status == SPARK_STATUS_OK && queue->recording != 0 )
	{
		SparkDeviceNode node;
		memset(&node,0,sizeof(node));
		node.kind = SPARK_DEVICE_NODE_RECORD;
		node.event = event;
		status = DeviceRecorded(queue,&node,"event_record");
		if ( status == SPARK_STATUS_OK && queue->recording->native != 0u )
			status = DeviceCaptured(queue,SparkDeviceBackendEventRecord(event,queue));
	}
	else if ( status == SPARK_STATUS_OK )
	{
		queue->submitted++;
		status = SparkDeviceBackendEventRecord(event,queue);
		if ( status != SPARK_STATUS_OK )
			queue->submitted--;
		else
		{
			event->queue_id = queue->id;
			event->sequence = queue->submitted;
			event->recorded = 1u;
			DeviceStampAt(event->stamps,slot,queue->id,queue->submitted);
		}
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	SparkStatus status;
	int32_t slot = -1;
	if ( event == 0 || event->id == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"queue_wait_event",0,0,"needs a live event"));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("queue_wait_event",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"queue_wait_event");
	if ( status == SPARK_STATUS_OK && event->recorded != 0u && (slot = DeviceStampsSlot(event->stamps,queue->id)) < 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_wait_event",queue->owner,0,"the event is in flight on too many queues");
	if ( status == SPARK_STATUS_OK && queue->recording != 0 )
	{
		SparkDeviceNode node;
		memset(&node,0,sizeof(node));
		node.kind = SPARK_DEVICE_NODE_AWAIT;
		node.event = event;
		status = DeviceRecorded(queue,&node,"queue_wait_event");
		if ( status == SPARK_STATUS_OK && queue->recording->native != 0u )
			status = DeviceCaptured(queue,SparkDeviceBackendQueueWaitEvent(queue,event));
	}
	else if ( status == SPARK_STATUS_OK && event->recorded != 0u )
	{
		queue->submitted++;
		status = SparkDeviceBackendQueueWaitEvent(queue,event);
		if ( status != SPARK_STATUS_OK )
			queue->submitted--;
		else
			DeviceStampAt(event->stamps,slot,queue->id,queue->submitted);
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceEventWait(SparkDeviceEvent *event, uint64_t timeout_ns)
{
	SparkStatus status;
	if ( event == 0 || event->id == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"event_wait",0,0,"needs a live event"));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("event_wait",0);
	pthread_mutex_unlock(&device_mutex);
	if ( status != SPARK_STATUS_OK || event->recorded == 0u )
		return(status);
	return(SparkDeviceBackendEventWait(event,timeout_ns));
}

SparkStatus SparkDeviceEventQuery(SparkDeviceEvent *event)
{
	return(SparkDeviceEventWait(event,0u));
}

SparkStatus SparkDeviceEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds)
{
	SparkStatus status;
	if ( start == 0 || stop == 0 || nanoseconds == 0 || start->id == 0u || stop->id == 0u ||
		start->recorded == 0u || stop->recorded == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"event_elapsed",0,0,"needs two recorded events and an output"));
	status = SparkDeviceEventWait(start,0u);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceEventWait(stop,0u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkDeviceBackendEventElapsed(start,stop,nanoseconds));
}

static SparkDeviceLibrary *DeviceLibraryLive(const SparkDeviceLibrary *library)
{
	SparkDeviceLibrary *entry;
	for ( entry = device_libraries; entry != 0; entry = entry->next )
		if ( entry == library && entry->id != 0u )
			return(entry);
	return(0);
}

static SparkDeviceKernel *DeviceKernelLive(const SparkDeviceKernel *kernel)
{
	SparkDeviceKernel *entry;
	for ( entry = device_kernels; entry != 0; entry = entry->next )
		if ( entry == kernel && entry->id != 0u )
			return(entry);
	return(0);
}

SparkStatus SparkDeviceLibraryLoad(SparkDeviceLibrary **library, const char *path, const char *owner)
{
	SparkDeviceLibrary *created = 0;
	SparkStatus status;
	if ( library == 0 || path == 0 || path[0] == '\0' )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"library_load",owner,0,"needs an output and a library path"));
	*library = 0;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("library_load",owner);
	if ( status == SPARK_STATUS_OK && (created = (SparkDeviceLibrary *)calloc(1u,sizeof(*created))) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"library_load",owner,0,"out of host memory for the library record");
	if ( status == SPARK_STATUS_OK )
	{
		created->id = device_next_library_id++;
		DeviceCopyText(created->owner,sizeof(created->owner),owner);
		status = SparkDeviceBackendLibraryLoad(created,path);
		if ( status != SPARK_STATUS_OK )
			free(created);
		else
		{
			created->next = device_libraries;
			device_libraries = created;
			*library = created;
		}
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceLibraryUnload(SparkDeviceLibrary *library)
{
	SparkDeviceLibrary **link;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("library_unload",0);
	if ( status == SPARK_STATUS_OK && DeviceLibraryLive(library) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"library_unload",0,0,"not a loaded library");
	if ( status == SPARK_STATUS_OK && library->kernels != 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_BUSY,"library_unload",library->owner,0,"kernels from the library are still live");
	if ( status == SPARK_STATUS_OK )
	{
		for ( link = &device_libraries; *link != 0; link = &(*link)->next )
			if ( *link == library )
			{
				*link = library->next;
				break;
			}
		SparkDeviceBackendLibraryUnload(library);
		library->id = 0u;
		free(library);
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static SparkStatus DeviceKernelName(SparkDeviceLibrary *library, const char *name, char stored[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u])
{
	char text[256];
	if ( strlen(name) <= SPARK_DEVICE_KERNEL_NAME_LIMIT )
	{
		memcpy(stored,name,strlen(name) + 1u);
		return(SPARK_STATUS_OK);
	}
	if ( DeviceLibraryLive(library) != 0 && SparkDeviceBackendKernelShortName(library,name,stored) == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	snprintf(text,sizeof(text),"a kernel name of %zu bytes is over %u and names no kernel of the library: %.40s",strlen(name),SPARK_DEVICE_KERNEL_NAME_LIMIT,name);
	{
		const SparkStatus refused = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_create",0,0,text);
		SPARK_FAIL(refused);
	}
}

SparkStatus SparkDeviceKernelCreate(SparkDeviceKernel **kernel, SparkDeviceLibrary *library, const char *name)
{
	char stored[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u];
	SparkDeviceKernel *created = 0;
	SparkStatus status;
	if ( kernel != 0 )
		*kernel = 0;
	if ( kernel == 0 || name == 0 || name[0] == '\0' )
	{
		const SparkStatus refused = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_create",name,0,"needs an output and a kernel name");
		SPARK_FAIL(refused);
	}
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("kernel_create",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceKernelName(library,name,stored);
	if ( status == SPARK_STATUS_OK && DeviceLibraryLive(library) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_create",stored,0,"not a loaded library");
	if ( status == SPARK_STATUS_OK && (created = (SparkDeviceKernel *)calloc(1u,sizeof(*created))) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_create",stored,0,"out of host memory for the kernel record");
	if ( status == SPARK_STATUS_OK )
	{
		created->id = device_next_kernel_id++;
		created->library = library;
		memcpy(created->name,stored,sizeof(created->name));
		status = SparkDeviceBackendKernelCreate(created,stored);
		if ( status == SPARK_STATUS_OK && (created->info.max_threads == 0u || created->info.simd_width == 0u) )
		{
			SparkDeviceBackendKernelDestroy(created);
			status = SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"kernel_create",stored,0,"the backend reported no thread limit for the kernel");
		}
		if ( status != SPARK_STATUS_OK )
			free(created);
		else
		{
			library->kernels++;
			created->next = device_kernels;
			device_kernels = created;
			*kernel = created;
		}
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceKernelDestroy(SparkDeviceKernel *kernel)
{
	SparkDeviceKernel **link;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("kernel_destroy",0);
	if ( status == SPARK_STATUS_OK && DeviceKernelLive(kernel) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_destroy",0,0,"not a live kernel");
	if ( status == SPARK_STATUS_OK && DeviceStampsBusy(kernel->stamps) != 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_BUSY,"kernel_destroy",kernel->name,0,"a queued launch still uses the kernel");
	if ( status == SPARK_STATUS_OK && kernel->programs != 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_BUSY,"kernel_destroy",kernel->name,0,"a program launches the kernel; destroy the program first");
	if ( status == SPARK_STATUS_OK )
	{
		for ( link = &device_kernels; *link != 0; link = &(*link)->next )
			if ( *link == kernel )
			{
				*link = kernel->next;
				break;
			}
		SparkDeviceBackendKernelDestroy(kernel);
		kernel->library->kernels--;
		kernel->id = 0u;
		free(kernel);
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceKernelGetInfo(const SparkDeviceKernel *kernel, SparkDeviceKernelInfo *info)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( info == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_info",0,0,"needs an output"));
	pthread_mutex_lock(&device_mutex);
	if ( DeviceKernelLive(kernel) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_info",0,0,"not a live kernel");
	else
		*info = kernel->info;
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static SparkStatus DeviceLaunchShape(const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch)
{
	const uint64_t threads = (uint64_t)launch->threads.x * launch->threads.y * launch->threads.z;
	if ( launch->groups.x == 0u || launch->groups.y == 0u || launch->groups.z == 0u || threads == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,"a launch needs at least one group of one thread"));
	if ( threads > kernel->info.max_threads )
		return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_launch",kernel->name,0,"more threads per group than the kernel allows"));
	if ( (uint64_t)launch->shared_bytes + kernel->info.static_shared_bytes > device_properties.max_shared_bytes )
		return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_launch",kernel->name,0,"more group-shared memory than the device has"));
	if ( launch->constant_bytes > device_properties.max_constant_bytes || (launch->constant_bytes != 0u) != (launch->constants != 0) )
	{
		char text[128];
		snprintf(text,sizeof(text),"constants need a pointer and at most the backend's %u bytes",device_properties.max_constant_bytes);
		{
			const SparkStatus refused = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,text);
			SPARK_FAIL(refused);
		}
	}
	if ( launch->binding_count > SPARK_DEVICE_MAX_BINDINGS || (launch->binding_count != 0u && launch->bindings == 0) )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,"more than 30 bindings or a missing binding table"));
	if ( launch->cluster.x == 0u && launch->cluster.y == 0u && launch->cluster.z == 0u )
		return(SPARK_STATUS_OK);
	if ( (device_properties.properties & SPARK_DEVICE_PROPERTY_CLUSTERS) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_launch",kernel->name,0,"the backend runs no thread-block clusters"));
	if ( launch->cluster.x == 0u || launch->cluster.y == 0u || launch->cluster.z == 0u || launch->groups.x % launch->cluster.x != 0u ||
		launch->groups.y % launch->cluster.y != 0u || launch->groups.z % launch->cluster.z != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,
			"a cluster has three nonzero sizes and the grid is a whole number of clusters on every axis"));
	return(SPARK_STATUS_OK);
}

static SparkStatus DeviceBind(const SparkDeviceKernel *kernel, const SparkDeviceBinding *binding, SparkDeviceAllocation **out)
{
	SparkDeviceAllocation *allocation = 0;
	const uint32_t access = binding->access;
	SparkStatus status = DeviceRange("kernel_launch",binding->range,binding->bytes,&allocation);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( allocation->device == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,"a binding names memory the device cannot see"));
	if ( access == 0u || (access & ~(SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE)) != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,"a binding reads, writes or both"));
	if ( ((access & SPARK_DEVICE_BIND_READ) != 0u && (allocation->access & SPARK_DEVICE_ACCESS_DEVICE_READ) == 0u) ||
		((access & SPARK_DEVICE_BIND_WRITE) != 0u && (allocation->access & SPARK_DEVICE_ACCESS_DEVICE_WRITE) == 0u) )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",allocation->owner,0,"a binding asks for access the buffer does not grant"));
	*out = allocation;
	return(SPARK_STATUS_OK);
}

static SparkStatus DeviceLaunchStamps(SparkDeviceKernel *kernel, SparkDeviceAllocation *const *allocations, uint32_t count, uint64_t queue_id,
	int32_t *slots)
{
	uint32_t index;
	slots[count] = DeviceStampsSlot(kernel->stamps,queue_id);
	for ( index = 0u; index < count; index++ )
		slots[index] = DeviceStampSlot(allocations[index],queue_id);
	for ( index = 0u; index <= count; index++ )
		if ( slots[index] < 0 )
			return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_launch",kernel->name,0,"a binding is in flight on too many queues"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch)
{
	SparkDeviceAllocation *allocations[SPARK_DEVICE_MAX_BINDINGS];
	int32_t slots[SPARK_DEVICE_MAX_BINDINGS + 1u];
	SparkDeviceKernel *live = 0;
	uint32_t index;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("kernel_launch",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"kernel_launch");
	if ( status == SPARK_STATUS_OK && ((live = DeviceKernelLive(kernel)) == 0 || launch == 0) )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",0,0,"needs a live kernel and a launch");
	if ( status == SPARK_STATUS_OK )
		status = DeviceLaunchShape(live,launch);
	for ( index = 0u; status == SPARK_STATUS_OK && index < launch->binding_count; index++ )
		status = DeviceBind(live,&launch->bindings[index],&allocations[index]);
	if ( status == SPARK_STATUS_OK )
		status = DeviceLaunchStamps(live,allocations,launch->binding_count,queue->id,slots);
	if ( status == SPARK_STATUS_OK && queue->recording != 0 )
	{
		SparkDeviceNode node;
		memset(&node,0,sizeof(node));
		node.kind = SPARK_DEVICE_NODE_LAUNCH;
		node.kernel = live;
		node.launch = *launch;
		status = DeviceRecorded(queue,&node,"kernel_launch");
		if ( status == SPARK_STATUS_OK && queue->recording->native != 0u )
			status = DeviceCaptured(queue,SparkDeviceBackendKernelLaunch(queue,live,launch,allocations));
	}
	else if ( status == SPARK_STATUS_OK )
	{
		queue->submitted++;
		status = SparkDeviceBackendKernelLaunch(queue,live,launch,allocations);
		if ( status != SPARK_STATUS_OK )
			queue->submitted--;
		else
		{
			for ( index = 0u; index < launch->binding_count; index++ )
				DeviceStamp(allocations[index],slots[index],queue->id,queue->submitted);
			DeviceStampAt(live->stamps,slots[launch->binding_count],queue->id,queue->submitted);
		}
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static SparkDeviceProgram *DeviceProgramLive(const SparkDeviceProgram *program)
{
	SparkDeviceProgram *entry;
	for ( entry = device_programs; entry != 0; entry = entry->next )
		if ( entry == program && entry->id != 0u )
			return(entry);
	return(0);
}

SparkStatus SparkDeviceProgramRecord(SparkDeviceQueue *queue, const char *name)
{
	SparkDeviceProgram *program = 0;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("program_record",name);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueSettled(queue,"program_record");
	if ( status == SPARK_STATUS_OK && (program = (SparkDeviceProgram *)calloc(1u,sizeof(*program))) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_record",name,0,"out of host memory for the program");
	if ( status == SPARK_STATUS_OK )
	{
		program->id = device_next_program_id++;
		program->queue_id = queue->id;
		program->native = SparkDeviceBackendProgramNative();
		program->thread = pthread_self();
		DeviceCopyText(program->name,sizeof(program->name),name);
		status = SparkDeviceBackendProgramBegin(queue,program);
	}
	if ( status == SPARK_STATUS_OK )
		queue->recording = program;
	else if ( program != 0 )
		SparkDeviceProgramFree(program);
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static SparkStatus DevicePatchValid(const SparkDevicePatch *patch, const char *owner)
{
	if ( patch == 0 || patch->value >= SPARK_DEVICE_PROGRAM_MAX_PATCHES || patch->target < SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT ||
		patch->target > SPARK_DEVICE_PROGRAM_TARGET_GRID_Z )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_patch",owner,0,"a patch names one of 64 values and an argument, constant or grid target"));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceProgramPatch(SparkDeviceQueue *queue, const SparkDevicePatch *patch)
{
	SparkDeviceProgram *program = 0;
	SparkDevicePlace place;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("program_patch",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"program_patch");
	if ( status == SPARK_STATUS_OK && (program = queue->recording) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_patch",queue->owner,0,"the queue is not recording a program");
	if ( status == SPARK_STATUS_OK )
		status = DevicePatchValid(patch,program->name);
	if ( status == SPARK_STATUS_OK )
	{
		memset(&place,0,sizeof(place));
		place.node = UINT32_MAX;
		place.value = patch->value;
		place.target = patch->target;
		place.argument = patch->argument;
		place.offset = patch->offset;
		place.bytes = patch->bytes;
		place.transform = patch->transform;
		status = program->native != 0u ? SparkDeviceBackendProgramPatch(queue,program,&place) : SparkDeviceProgramPlace(program,patch,&place,program->name);
	}
	if ( status == SPARK_STATUS_OK && program->place_count == program->place_capacity )
	{
		SparkDevicePlace *grown = (SparkDevicePlace *)realloc(program->places,((size_t)program->place_capacity + 64u) * sizeof(*grown));
		if ( grown == 0 )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_patch",program->name,0,"out of host memory for the patch table");
		else
		{
			program->places = grown;
			program->place_capacity += 64u;
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		program->places[program->place_count++] = place;
		if ( patch->value + 1u > program->values )
			program->values = patch->value + 1u;
	}
	else if ( program != 0 )
		program->broken = 1u;
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceProgramFinish(SparkDeviceQueue *queue, SparkDeviceProgram **out)
{
	SparkDeviceProgram *program = 0;
	SparkStatus status;
	if ( out == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_finish",0,0,"needs an output"));
	*out = 0;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("program_finish",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"program_finish");
	if ( status == SPARK_STATUS_OK && (program = queue->recording) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_finish",queue->owner,0,"the queue is not recording a program");
	if ( status == SPARK_STATUS_OK )
	{
		queue->recording = 0;
		if ( program->native != 0u )
			status = SparkDeviceBackendProgramEnd(queue,program);
		if ( status == SPARK_STATUS_OK && program->broken != 0u )
			status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_finish",program->name,0,
				"an operation recorded into the program failed, so the program is incomplete");
		if ( status == SPARK_STATUS_OK && program->native == 0u )
			status = SparkDeviceBackendProgramCompile(queue,program);
		if ( status != SPARK_STATUS_OK )
		{
			if ( program->native != 0u || program->compiled != 0u )
				SparkDeviceBackendProgramRelease(program);
			DeviceProgramRelease(program);
		}
		else
		{
			if ( program->native == 0u )
				program->digest = SparkDeviceProgramDigest(program);
			program->finished = 1u;
			program->next = device_programs;
			device_programs = program;
			*out = program;
		}
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

static SparkStatus DeviceProgramSlots(SparkDeviceProgram *program, SparkDeviceQueue *queue, int32_t *slot)
{
	uint32_t index;
	*slot = DeviceStampsSlot(program->stamps,queue->id);
	if ( *slot < 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_run",program->name,0,"the program is in flight on too many queues"));
	for ( index = 0u; index < program->record_count; index++ )
		if ( DeviceStampsSlot(program->nodes[program->records[index]].event->stamps,queue->id) < 0 )
			return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_run",program->name,0,
				"an event the program records is in flight on too many queues"));
	return(SPARK_STATUS_OK);
}

static void DeviceProgramRan(SparkDeviceProgram *program, SparkDeviceQueue *queue, int32_t slot)
{
	uint32_t index;
	DeviceStampAt(program->stamps,slot,queue->id,queue->submitted);
	for ( index = 0u; index < program->record_count; index++ )
	{
		SparkDeviceEvent *event = program->nodes[program->records[index]].event;
		DeviceStampAt(event->stamps,DeviceStampsSlot(event->stamps,queue->id),queue->id,queue->submitted);
		event->queue_id = queue->id;
		event->sequence = queue->submitted;
		event->recorded = 1u;
	}
}

SparkStatus SparkDeviceProgramRun(SparkDeviceProgram *program, SparkDeviceQueue *queue, const uint64_t *values, uint32_t count)
{
	uint64_t placed_stack[64],*placed = placed_stack;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("program_run",0);
	if ( status == SPARK_STATUS_OK && DeviceProgramLive(program) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_run",0,0,"not a live program");
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueSettled(queue,"program_run");
	if ( status == SPARK_STATUS_OK && (count != program->values || (count != 0u && values == 0)) )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_run",program->name,0,"a run passes exactly the program's patch values");
	if ( status == SPARK_STATUS_OK && program->place_count > sizeof(placed_stack) / sizeof(placed_stack[0]) &&
		(placed = (uint64_t *)calloc(program->place_count,sizeof(*placed))) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_run",program->name,0,"out of host memory for the patch values");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceProgramValues(program,values,placed,program->name);
	if ( status == SPARK_STATUS_OK && (program->native != 0u || program->compiled != 0u) )
	{
		int32_t slot = -1;
		status = DeviceProgramSlots(program,queue,&slot);
		if ( status == SPARK_STATUS_OK )
		{
			queue->submitted++;
			status = SparkDeviceBackendProgramRun(program,queue,placed);
			if ( status != SPARK_STATUS_OK )
				queue->submitted--;
			else
				DeviceProgramRan(program,queue,slot);
		}
		pthread_mutex_unlock(&device_mutex);
	}
	else if ( status == SPARK_STATUS_OK )
	{
		status = SparkDeviceBackendProgramReplay(queue);
		pthread_mutex_unlock(&device_mutex);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceProgramReplay(program,queue,placed);
	}
	else
		pthread_mutex_unlock(&device_mutex);
	if ( placed != placed_stack )
		free(placed);
	return(status);
}

SparkStatus SparkDeviceProgramGetInfo(const SparkDeviceProgram *program, SparkDeviceProgramInfo *info)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( info == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_info",0,0,"needs an output"));
	pthread_mutex_lock(&device_mutex);
	if ( DeviceProgramLive(program) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_info",0,0,"not a live program");
	else
	{
		memset(info,0,sizeof(*info));
		info->nodes = program->native != 0u ? program->native_nodes : program->node_count;
		info->kernels = program->kernels;
		info->values = program->values;
		info->places = program->place_count;
		info->digest = program->digest;
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceProgramDestroy(SparkDeviceProgram *program)
{
	SparkDeviceProgram **link;
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&device_mutex);
	if ( DeviceProgramLive(program) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_destroy",0,0,"not a live program");
	else if ( DeviceStampsBusy(program->stamps) != 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_BUSY,"program_destroy",program->name,0,"a queued run still uses the program");
	else
	{
		for ( link = &device_programs; *link != 0; link = &(*link)->next )
			if ( *link == program )
			{
				*link = program->next;
				break;
			}
		if ( program->native != 0u || program->compiled != 0u )
			SparkDeviceBackendProgramRelease(program);
		program->id = 0u;
		DeviceProgramRelease(program);
	}
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

#define DEVICE_TENSOR_MAP_IMAGE "sparkpipe_device_kernels/tensor_map_build.cubin"
#define DEVICE_TENSOR_MAP_KERNEL "SparkDeviceTensorMapBuild"
#define DEVICE_TENSOR_MAP_THREADS 64u
#define DEVICE_TENSOR_MAP_PARAMETERS 3u
#define DEVICE_TENSOR_MAP_TIMEOUT_NS 10000000000ull
#define DEVICE_TENSOR_MAP_DIMENSION_MAX 0xffffffffull
#define DEVICE_TENSOR_MAP_STRIDE_LIMIT (1ull << 40)
#define DEVICE_TENSOR_MAP_ELEMENT_STRIDE_MAX 8u
#define DEVICE_TENSOR_MAP_ALIGN 16u

static SparkStatus DeviceTensorMapRefuse(const char *owner, uint32_t index, const char *what)
{
	char text[128];
	snprintf(text,sizeof(text),"tensor map %u: %s",index,what);
	return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"tensor_map",owner,0,text));
}

static SparkStatus DeviceTensorMapCheck(const SparkDeviceTensorMapRequest *request, uint32_t index, const char *owner)
{
	uint32_t axis;
	if ( request->address == 0u || (request->address % DEVICE_TENSOR_MAP_ALIGN) != 0u )
		return(DeviceTensorMapRefuse(owner,index,"the address must be nonzero and 16-byte aligned"));
	if ( request->rank < 2u || request->rank > SPARK_DEVICE_TENSOR_MAP_RANK_MAX )
		return(DeviceTensorMapRefuse(owner,index,"the rank must be 2 or 3"));
	if ( request->swizzle_bytes != 0u && request->swizzle_bytes != 32u && request->swizzle_bytes != 64u && request->swizzle_bytes != 128u )
		return(DeviceTensorMapRefuse(owner,index,"the swizzle must be 0, 32, 64 or 128 bytes"));
	for ( axis = 0u; axis < request->rank; axis++ )
	{
		if ( request->dimension[axis] == 0u || request->dimension[axis] > DEVICE_TENSOR_MAP_DIMENSION_MAX )
			return(DeviceTensorMapRefuse(owner,index,"every dimension must be 1 to 2^32 - 1 elements"));
		if ( request->box[axis] == 0u || request->box[axis] > SPARK_DEVICE_TENSOR_MAP_BOX_MAX )
			return(DeviceTensorMapRefuse(owner,index,"every box edge must be 1 to 256 elements"));
		if ( request->element_stride[axis] == 0u || request->element_stride[axis] > DEVICE_TENSOR_MAP_ELEMENT_STRIDE_MAX )
			return(DeviceTensorMapRefuse(owner,index,"every element stride must be 1 to 8"));
	}
	if ( (request->box[0] % DEVICE_TENSOR_MAP_ALIGN) != 0u )
		return(DeviceTensorMapRefuse(owner,index,"the box's inner edge must be a multiple of 16 bytes"));
	if ( request->swizzle_bytes != 0u && request->box[0] > request->swizzle_bytes )
		return(DeviceTensorMapRefuse(owner,index,"the box's inner edge must fit in the swizzle span"));
	for ( axis = 0u; axis + 1u < request->rank; axis++ )
	{
		const uint64_t below = axis == 0u ? request->dimension[0] : request->stride_bytes[axis - 1u] * request->dimension[axis];
		if ( (request->stride_bytes[axis] % DEVICE_TENSOR_MAP_ALIGN) != 0u || request->stride_bytes[axis] >= DEVICE_TENSOR_MAP_STRIDE_LIMIT )
			return(DeviceTensorMapRefuse(owner,index,"every stride must be a multiple of 16 bytes below 2^40"));
		if ( request->stride_bytes[axis] < below )
			return(DeviceTensorMapRefuse(owner,index,"a stride must cover the dimensions below it"));
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus DeviceTensorMapImage(char *path, size_t capacity, const char *owner)
{
	Dl_info info;
	const char *slash;
	memset(&info,0,sizeof(info));
	if ( dladdr((void *)(uintptr_t)&SparkDeviceTensorMapEncode,&info) == 0 || info.dli_fname == 0 || (slash = strrchr(info.dli_fname,'/')) == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"tensor_map",owner,0,"cannot find the device library's directory"));
	if ( snprintf(path,capacity,"%.*s/%s",(int)(slash - info.dli_fname),info.dli_fname,DEVICE_TENSOR_MAP_IMAGE) >= (int)capacity )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"tensor_map",owner,0,"the tensor map image's path is too long"));
	return(SPARK_STATUS_OK);
}

static SparkStatus DeviceTensorMapLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceBuffer *buffer, uint64_t map_offset,
	uint32_t count, const char *owner)
{
	const uint64_t values[DEVICE_TENSOR_MAP_PARAMETERS] = {buffer->device, buffer->device + map_offset, count};
	uint8_t block[64];
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	uint32_t index;
	SparkStatus status;
	memset(&info,0,sizeof(info));
	status = SparkDeviceKernelGetInfo(kernel,&info);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( info.parameters == 0 || info.parameter_count != DEVICE_TENSOR_MAP_PARAMETERS || info.parameter_bytes > sizeof(block) )
		return(SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"tensor_map",owner,0,"the tensor map image's kernel takes other parameters"));
	memset(block,0,sizeof(block));
	for ( index = 0u; index < DEVICE_TENSOR_MAP_PARAMETERS; index++ )
	{
		const SparkDeviceParameter *parameter = &info.parameters[index];
		if ( parameter->bytes > sizeof(values[index]) || parameter->offset + parameter->bytes > info.parameter_bytes )
			return(SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"tensor_map",owner,0,"the tensor map image's kernel takes other parameters"));
		memcpy(block + parameter->offset,&values[index],parameter->bytes);
	}
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (count + DEVICE_TENSOR_MAP_THREADS - 1u) / DEVICE_TENSOR_MAP_THREADS;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = DEVICE_TENSOR_MAP_THREADS;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = block;
	launch.constant_bytes = info.parameter_bytes;
	status = SparkDeviceKernelLaunch(queue,kernel,&launch);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(queue,DEVICE_TENSOR_MAP_TIMEOUT_NS);
	if ( status == SPARK_STATUS_PENDING )
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"tensor_map",owner,0,"the tensor map build did not finish within 10 s");
	return(status);
}

static SparkStatus DeviceTensorMapBuild(const SparkDeviceTensorMapRequest *requests, uint32_t count, SparkDeviceTensorMap *maps, const char *owner)
{
	const uint64_t request_bytes = (uint64_t)count * sizeof(*requests);
	const uint64_t map_offset = (request_bytes + SPARK_DEVICE_TENSOR_MAP_BYTES - 1u) / SPARK_DEVICE_TENSOR_MAP_BYTES * SPARK_DEVICE_TENSOR_MAP_BYTES;
	char path[4096];
	SparkDeviceLibrary *library = 0;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceQueue *queue = 0;
	SparkDeviceBuffer buffer;
	SparkStatus status = DeviceTensorMapImage(path,sizeof(path),owner);
	memset(&buffer,0,sizeof(buffer));
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLibraryLoad(&library,path,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceKernelCreate(&kernel,library,DEVICE_TENSOR_MAP_KERNEL);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAllocate(&buffer,SPARK_DEVICE_SPACE_SHARED_COHERENT,map_offset + (uint64_t)count * SPARK_DEVICE_TENSOR_MAP_BYTES,owner);
	if ( status == SPARK_STATUS_OK && buffer.host == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"tensor_map",owner,0,"shared memory has no host view on this device");
	if ( status == SPARK_STATUS_OK )
	{
		memcpy(buffer.host,requests,(size_t)request_bytes);
		status = SparkDeviceQueueCreate(&queue,owner);
	}
	if ( status == SPARK_STATUS_OK )
		status = DeviceTensorMapLaunch(queue,kernel,&buffer,map_offset,count,owner);
	if ( status == SPARK_STATUS_OK )
		memcpy(maps,(const uint8_t *)buffer.host + map_offset,(size_t)count * SPARK_DEVICE_TENSOR_MAP_BYTES);
	if ( queue != 0 )
		(void)SparkDeviceQueueDestroy(queue);
	if ( buffer.bytes != 0u )
		(void)SparkDeviceFree(&buffer);
	if ( kernel != 0 )
		(void)SparkDeviceKernelDestroy(kernel);
	if ( library != 0 )
		(void)SparkDeviceLibraryUnload(library);
	return(status);
}

SparkStatus SparkDeviceTensorMapEncode(const SparkDeviceTensorMapRequest *requests, uint32_t count, SparkDeviceTensorMap *maps, const char *owner)
{
	uint32_t index,on_device = 0u;
	SparkStatus status;
	if ( requests == 0 || maps == 0 || count == 0u || count > SPARK_DEVICE_TENSOR_MAP_BATCH_MAX )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"tensor_map",owner,0,"needs 1 to 4096 requests and room for their maps"));
	if ( ((uintptr_t)maps % 64u) != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"tensor_map",owner,0,"the maps must be 64-byte aligned"));
	for ( index = 0u; index < count; index++ )
		if ( (status = DeviceTensorMapCheck(&requests[index],index,owner)) != SPARK_STATUS_OK )
			return(status);
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("tensor_map",owner);
	if ( status == SPARK_STATUS_OK && (device_properties.properties & SPARK_DEVICE_PROPERTY_TENSOR_MAPS) == 0u )
		status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"tensor_map",owner,0,"this device has no tensor maps");
	if ( status == SPARK_STATUS_OK && (on_device = SparkDeviceBackendTensorMapOnDevice()) == 0u )
		status = SparkDeviceBackendTensorMapEncode(requests,count,maps,owner);
	pthread_mutex_unlock(&device_mutex);
	if ( status == SPARK_STATUS_OK && on_device != 0u )
		status = DeviceTensorMapBuild(requests,count,maps,owner);
	return(status);
}
