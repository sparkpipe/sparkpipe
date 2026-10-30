#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/device/device_backend.h"

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
		device_initialized = 1u;
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
	if ( device_live_allocations != 0u || device_live_events != 0u || device_libraries != 0 ||
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
	device_initialized = 0u;
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

SparkStatus SparkDeviceRegister(SparkDeviceBuffer *buffer, void *host, uint64_t bytes, uint32_t access, const char *owner)
{
	SparkDeviceAllocation *allocation = 0;
	SparkStatus status;
	if ( buffer == 0 || host == 0 || bytes == 0u ||
		(access & (SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE)) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"register",owner,0,"needs host memory, a size and device access"));
	memset(buffer,0,sizeof(*buffer));
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("register",owner);
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_unlock(&device_mutex);
		return(status);
	}
	allocation = DeviceNewAllocation(SPARK_DEVICE_SPACE_HOST_PINNED,bytes,
		access | SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE,owner);
	if ( allocation == 0 )
	{
		pthread_mutex_unlock(&device_mutex);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"register",owner,0,"out of host memory for the allocation record"));
	}
	allocation->registered = 1u;
	status = SparkDeviceBackendRegister(allocation,host);
	if ( status != SPARK_STATUS_OK )
		free(allocation);
	else
		status = DeviceAdmit(buffer,allocation,"register");
	pthread_mutex_unlock(&device_mutex);
	return(status);
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
	if ( status == SPARK_STATUS_OK )
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

static SparkStatus DeviceCreateQueueLocked(SparkDeviceQueue **out, const char *owner)
{
	SparkDeviceQueue *queue = (SparkDeviceQueue *)calloc(1u,sizeof(*queue));
	SparkStatus status;
	if ( queue == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",owner,0,"out of host memory for the queue"));
	queue->id = device_next_queue_id++;
	DeviceCopyText(queue->owner,sizeof(queue->owner),owner);
	status = SparkDeviceBackendQueueCreate(queue);
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
		status = DeviceCreateQueueLocked(&device_sync_queue,"synchronous copy");
	queue = device_sync_queue;
	pthread_mutex_unlock(&device_mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDeviceCopy(queue,destination,source,bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER));
}

SparkStatus SparkDeviceQueueCreate(SparkDeviceQueue **queue, const char *owner)
{
	SparkStatus status;
	if ( queue == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"queue_create",owner,0,"needs an output"));
	*queue = 0;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("queue_create",owner);
	if ( status == SPARK_STATUS_OK )
		status = DeviceCreateQueueLocked(queue,owner);
	pthread_mutex_unlock(&device_mutex);
	return(status);
}

SparkStatus SparkDeviceQueueWait(SparkDeviceQueue *queue, uint64_t timeout_ns)
{
	uint64_t sequence;
	SparkStatus status;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("queue_wait",0);
	if ( status == SPARK_STATUS_OK )
		status = DeviceQueueValid(queue,"queue_wait");
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

SparkStatus SparkDeviceQueueDestroy(SparkDeviceQueue *queue)
{
	SparkStatus status = SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
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
	if ( status == SPARK_STATUS_OK )
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
	for ( queue = device_queues; status == SPARK_STATUS_OK && queue != 0; queue = queue->next )
		count++;
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
	if ( status == SPARK_STATUS_OK )
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
	if ( status == SPARK_STATUS_OK && event->recorded != 0u )
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

SparkStatus SparkDeviceKernelCreate(SparkDeviceKernel **kernel, SparkDeviceLibrary *library, const char *name)
{
	SparkDeviceKernel *created = 0;
	SparkStatus status;
	if ( kernel == 0 || name == 0 || name[0] == '\0' || strlen(name) >= SPARK_DEVICE_KERNEL_NAME_BYTES )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_create",name,0,"needs an output and a kernel name under 64 bytes"));
	*kernel = 0;
	pthread_mutex_lock(&device_mutex);
	status = DeviceReady("kernel_create",name);
	if ( status == SPARK_STATUS_OK && DeviceLibraryLive(library) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_create",name,0,"not a loaded library");
	if ( status == SPARK_STATUS_OK && (created = (SparkDeviceKernel *)calloc(1u,sizeof(*created))) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_create",name,0,"out of host memory for the kernel record");
	if ( status == SPARK_STATUS_OK )
	{
		created->id = device_next_kernel_id++;
		created->library = library;
		DeviceCopyText(created->name,sizeof(created->name),name);
		status = SparkDeviceBackendKernelCreate(created,name);
		if ( status == SPARK_STATUS_OK && (created->info.max_threads == 0u || created->info.simd_width == 0u) )
		{
			SparkDeviceBackendKernelDestroy(created);
			status = SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"kernel_create",name,0,"the backend reported no thread limit for the kernel");
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
	if ( launch->constant_bytes > SPARK_DEVICE_MAX_CONSTANT_BYTES || (launch->constant_bytes != 0u) != (launch->constants != 0) )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,"constants need a pointer and at most 4096 bytes"));
	if ( launch->binding_count > SPARK_DEVICE_MAX_BINDINGS || (launch->binding_count != 0u && launch->bindings == 0) )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",kernel->name,0,"more than 30 bindings or a missing binding table"));
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
	if ( status == SPARK_STATUS_OK )
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
