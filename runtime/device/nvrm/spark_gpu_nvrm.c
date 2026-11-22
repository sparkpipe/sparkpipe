#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_register.h"
#include "runtime/device/nvrm/nvrm_occupancy.h"
#include "runtime/device/nvrm/spark_gpu_nvrm.h"

#define GPU_OWNER "hosted launches"

static _Thread_local SparkGpuError gpu_last_error = SPARK_GPU_SUCCESS;
static pthread_mutex_t gpu_mutex = PTHREAD_MUTEX_INITIALIZER;
static SparkDeviceQueue *gpu_default_queue;
static SparkDeviceBuffer gpu_staging;
static pthread_mutex_t gpu_staging_mutex = PTHREAD_MUTEX_INITIALIZER;

static SparkGpuError GpuFromStatus(SparkStatus status)
{
	switch ( status )
	{
	case SPARK_STATUS_OK:
		return(SPARK_GPU_SUCCESS);
	case SPARK_STATUS_INVALID_ARGUMENT:
	case SPARK_STATUS_NOT_FOUND:
	case SPARK_STATUS_DUPLICATE:
	case SPARK_STATUS_PARSE_ERROR:
	case SPARK_STATUS_TARGET_MISMATCH:
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	case SPARK_STATUS_CAPACITY_EXCEEDED:
		return(SPARK_GPU_ERROR_MEMORY_ALLOCATION);
	case SPARK_STATUS_UNSUPPORTED:
		return(SPARK_GPU_ERROR_NOT_SUPPORTED);
	case SPARK_STATUS_BUSY:
	case SPARK_STATUS_PENDING:
		return(SPARK_GPU_ERROR_NOT_READY);
	default:
		return(SPARK_GPU_ERROR_UNKNOWN);
	}
}

static SparkGpuError GpuRecord(SparkStatus status)
{
	gpu_last_error = GpuFromStatus(status);
	return(gpu_last_error);
}

#define GPU_FAIL(status) \
	do { \
		const SparkGpuError gpu_error = GpuRecord(status); \
		SPARK_FAIL(gpu_error); \
	} while (0)

#define GPU_REFUSE_IN(operation, status, text) \
	do { \
		(void)SparkDeviceRecordError((status),(operation),GPU_OWNER,0,(text)); \
		GPU_FAIL(status); \
	} while (0)

#define GPU_REFUSE(status, text) GPU_REFUSE_IN("kernel_launch",status,text)

#define GPU_TRY(call) \
	do { \
		const SparkStatus gpu_status = (call); \
		if ( gpu_status != SPARK_STATUS_OK ) \
			GPU_FAIL(gpu_status); \
	} while (0)

static int gpu_ordinal;

static SparkStatus GpuReady(void)
{
	SparkDeviceConfiguration configuration;
	SparkStatus status = SPARK_STATUS_OK;
	if ( SparkDeviceGetProperties() != 0 )
		return(SPARK_STATUS_OK);
	pthread_mutex_lock(&gpu_mutex);
	if ( SparkDeviceGetProperties() == 0 )
	{
		memset(&configuration,0,sizeof(configuration));
		configuration.ordinal = (uint32_t)gpu_ordinal;
		status = SparkDeviceInitialize(&configuration);
	}
	pthread_mutex_unlock(&gpu_mutex);
	return(status);
}

SparkGpuError SparkGpuGetLastError(void)
{
	const SparkGpuError error = gpu_last_error;
	gpu_last_error = SPARK_GPU_SUCCESS;
	return(error);
}

SparkGpuError SparkGpuPeekAtLastError(void)
{
	return(gpu_last_error);
}

const char *SparkGpuGetErrorString(SparkGpuError error)
{
	switch ( error )
	{
	case SPARK_GPU_SUCCESS:
		return("no error");
	case SPARK_GPU_ERROR_INVALID_VALUE:
		return("invalid argument");
	case SPARK_GPU_ERROR_MEMORY_ALLOCATION:
		return("out of memory");
	case SPARK_GPU_ERROR_INVALID_DEVICE:
		return("invalid device");
	case SPARK_GPU_ERROR_NOT_READY:
		return("device not ready");
	case SPARK_GPU_ERROR_NOT_SUPPORTED:
		return("operation not supported");
	default:
		return("unknown error");
	}
}

SparkGpuError SparkNvrmGpuDefaultStream(SparkGpuStream *stream)
{
	SparkStatus status = GpuReady();
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	pthread_mutex_lock(&gpu_mutex);
	if ( gpu_default_queue == 0 )
		status = SparkDeviceQueueCreate(&gpu_default_queue,"default stream");
	*stream = gpu_default_queue;
	pthread_mutex_unlock(&gpu_mutex);
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkNvrmGpuRelease(void)
{
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&gpu_mutex);
	if ( gpu_default_queue != 0 && (status = SparkDeviceQueueDestroy(gpu_default_queue)) == SPARK_STATUS_OK )
		gpu_default_queue = 0;
	pthread_mutex_unlock(&gpu_mutex);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmRegisteredRelease();
	pthread_mutex_lock(&gpu_staging_mutex);
	if ( status == SPARK_STATUS_OK && gpu_staging.bytes != 0u )
		status = SparkDeviceFree(&gpu_staging);
	pthread_mutex_unlock(&gpu_staging_mutex);
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

static SparkGpuError GpuLayoutAgrees(const SparkDeviceKernel *kernel, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes,
	const char *name)
{
	SparkDeviceKernelInfo info;
	char text[256];
	uint32_t index;
	if ( SparkDeviceKernelGetInfo(kernel,&info) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	if ( info.parameter_count != count || info.parameter_bytes != bytes )
	{
		snprintf(text,sizeof(text),"%.48s: the host passes %u parameters in %u bytes, the kernel takes %u in %u",name,count,bytes,info.parameter_count,
			info.parameter_bytes);
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,text);
	}
	for (index = 0u; index < count; index++)
		if ( info.parameters[index].offset != layout[index].offset || info.parameters[index].bytes != layout[index].bytes )
		{
			snprintf(text,sizeof(text),"%.48s: parameter %u is %u bytes at %u on the host, %u bytes at %u in the kernel",name,index,layout[index].bytes,
				layout[index].offset,info.parameters[index].bytes,info.parameters[index].offset);
			GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,text);
		}
	return(SPARK_GPU_SUCCESS);
}

static SparkGpuError GpuLaunch(const void *function, const uint32_t grid[3], const uint32_t block[3], uint32_t cluster, size_t shared_bytes,
	SparkGpuStream stream, const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes)
{
	SparkDeviceKernel *kernel = 0;
	SparkDeviceLaunch launch;
	const char *name = 0;
	SparkGpuError error;
	if ( grid == 0 || block == 0 || (count != 0u && (layout == 0 || parameters == 0)) || shared_bytes > UINT32_MAX )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"a hosted launch needs its grid, its block, its parameters and at most 4 GiB of shared memory");
	GPU_TRY(SparkNvrmRegisteredName(function,&name));
	GPU_TRY(GpuReady());
	if ( SparkNvrmRegisteredKernel(function,&kernel) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	if ( (error = GpuLayoutAgrees(kernel,layout,count,bytes,name)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(error);
	if ( stream == 0 && (error = SparkNvrmGpuDefaultStream(&stream)) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(error);
	memset(&launch,0,sizeof(launch));
	launch.groups.x = grid[0];
	launch.groups.y = grid[1];
	launch.groups.z = grid[2];
	launch.threads.x = block[0];
	launch.threads.y = block[1];
	launch.threads.z = block[2];
	launch.shared_bytes = (uint32_t)shared_bytes;
	launch.constants = bytes != 0u ? parameters : 0;
	launch.constant_bytes = bytes;
	if ( cluster != 0u )
	{
		launch.cluster.x = cluster;
		launch.cluster.y = 1u;
		launch.cluster.z = 1u;
	}
	if ( SparkDeviceKernelLaunch(stream,kernel,&launch) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkNvrmLaunchPacked(const void *function, const uint32_t grid[3], const uint32_t block[3], size_t shared_bytes, SparkGpuStream stream,
	const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes)
{
	return(GpuLaunch(function,grid,block,0u,shared_bytes,stream,parameters,layout,count,bytes));
}

SparkGpuError SparkNvrmLaunchCluster(const void *function, const uint32_t grid[3], const uint32_t block[3], uint32_t cluster, size_t shared_bytes,
	SparkGpuStream stream, const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes)
{
	if ( cluster == 0u )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"a cluster launch names a cluster of at least one block");
	return(GpuLaunch(function,grid,block,cluster,shared_bytes,stream,parameters,layout,count,bytes));
}

SparkGpuError SparkNvrmSharedMemoryOptIn(const void *function, uint32_t shared_bytes)
{
	const SparkDeviceProperties *properties = SparkDeviceGetProperties();
	const char *name = 0;
	if ( properties == 0 )
		GPU_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,"shared memory is granted on an initialized device");
	if ( SparkNvrmRegisteredName(function,&name) != SPARK_STATUS_OK )
		GPU_FAIL(SparkDeviceLastError()->status);
	if ( shared_bytes > properties->max_shared_bytes )
		GPU_REFUSE(SPARK_STATUS_UNSUPPORTED,"the kernel asks for more group-shared memory than the device has");
	return(SPARK_GPU_SUCCESS);
}

#define GPU_STAGING_BYTES (8ull << 20)
#define GPU_MEMINFO_PATH "/proc/meminfo"

typedef struct GpuAllocation
{
	SparkDeviceBuffer buffer;
	void *host;
	uint32_t registered;
	struct GpuAllocation *next;
} GpuAllocation;

static GpuAllocation *gpu_allocations;

static SparkGpuError GpuStream(SparkGpuStream *stream)
{
	if ( *stream != 0 )
		return(SPARK_GPU_SUCCESS);
	return(SparkNvrmGpuDefaultStream(stream));
}

static SparkGpuError GpuAllocate(void **pointer, size_t bytes, SparkDeviceSpace space, const char *owner)
{
	GpuAllocation *entry;
	SparkStatus status;
	if ( pointer == 0 || bytes == 0u )
		GPU_REFUSE_IN("allocate",SPARK_STATUS_INVALID_ARGUMENT,"an allocation needs an output and a size");
	GPU_TRY(GpuReady());
	if ( (entry = (GpuAllocation *)calloc(1u,sizeof(*entry))) == 0 )
		GPU_REFUSE_IN("allocate",SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the allocation table");
	if ( (status = SparkDeviceAllocate(&entry->buffer,space,bytes,owner)) != SPARK_STATUS_OK )
	{
		free(entry);
		GPU_FAIL(status);
	}
	pthread_mutex_lock(&gpu_mutex);
	entry->next = gpu_allocations;
	gpu_allocations = entry;
	pthread_mutex_unlock(&gpu_mutex);
	*pointer = entry->buffer.host != 0 ? entry->buffer.host : (void *)(uintptr_t)entry->buffer.device;
	return(SPARK_GPU_SUCCESS);
}

static GpuAllocation *GpuTake(const void *pointer, uint32_t registered)
{
	GpuAllocation **link;
	for (link = &gpu_allocations; *link != 0; link = &(*link)->next)
	{
		GpuAllocation *entry = *link;
		const void *key = registered != 0u ? entry->host : (entry->buffer.host != 0 ? entry->buffer.host : (const void *)(uintptr_t)entry->buffer.device);
		if ( entry->registered == registered && key == pointer )
		{
			*link = entry->next;
			return(entry);
		}
	}
	return(0);
}

static SparkGpuError GpuRelease(const void *pointer, uint32_t registered, const char *operation)
{
	GpuAllocation *entry;
	SparkStatus status;
	if ( pointer == 0 )
		return(SPARK_GPU_SUCCESS);
	GPU_TRY(SparkDeviceSynchronize(SPARK_DEVICE_WAIT_FOREVER));
	pthread_mutex_lock(&gpu_mutex);
	entry = GpuTake(pointer,registered);
	pthread_mutex_unlock(&gpu_mutex);
	if ( entry == 0 )
		GPU_REFUSE_IN(operation,SPARK_STATUS_INVALID_ARGUMENT,"the pointer is not one this front allocated or registered");
	status = SparkDeviceFree(&entry->buffer);
	free(entry);
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuMalloc(void **pointer, size_t bytes)
{
	return(GpuAllocate(pointer,bytes,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"gpu malloc"));
}

SparkGpuError SparkGpuMallocManaged(void **pointer, size_t bytes, unsigned int flags)
{
	if ( flags != SPARK_GPU_MEM_ATTACH_GLOBAL )
		GPU_REFUSE_IN("allocate",SPARK_STATUS_UNSUPPORTED,"managed memory is attached globally only");
	return(GpuAllocate(pointer,bytes,SPARK_DEVICE_SPACE_SHARED_COHERENT,"gpu managed"));
}

SparkGpuError SparkGpuMallocHost(void **pointer, size_t bytes)
{
	return(GpuAllocate(pointer,bytes,SPARK_DEVICE_SPACE_HOST_PINNED,"gpu host"));
}

SparkGpuError SparkGpuHostAlloc(void **pointer, size_t bytes, unsigned int flags)
{
	if ( (flags & ~(SPARK_GPU_HOST_ALLOC_PORTABLE | SPARK_GPU_HOST_ALLOC_MAPPED)) != 0u )
		GPU_REFUSE_IN("allocate",SPARK_STATUS_UNSUPPORTED,"host allocations take only the portable and mapped flags");
	return(GpuAllocate(pointer,bytes,SPARK_DEVICE_SPACE_HOST_PINNED,"gpu host"));
}

SparkGpuError SparkGpuHostRegister(void *host_pointer, size_t bytes, unsigned int flags)
{
	GpuAllocation *entry;
	SparkStatus status;
	if ( host_pointer == 0 || bytes == 0u || (flags & ~(SPARK_GPU_HOST_REGISTER_PORTABLE | SPARK_GPU_HOST_REGISTER_MAPPED)) != 0u )
		GPU_REFUSE_IN("register",SPARK_STATUS_INVALID_ARGUMENT,"a registration needs host memory, a size and only the portable and mapped flags");
	GPU_TRY(GpuReady());
	if ( (entry = (GpuAllocation *)calloc(1u,sizeof(*entry))) == 0 )
		GPU_REFUSE_IN("register",SPARK_STATUS_CAPACITY_EXCEEDED,"out of host memory for the allocation table");
	status = SparkDeviceRegister(&entry->buffer,host_pointer,bytes,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,"gpu register");
	if ( status != SPARK_STATUS_OK )
	{
		free(entry);
		GPU_FAIL(status);
	}
	entry->host = host_pointer;
	entry->registered = 1u;
	pthread_mutex_lock(&gpu_mutex);
	entry->next = gpu_allocations;
	gpu_allocations = entry;
	pthread_mutex_unlock(&gpu_mutex);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuHostUnregister(void *host_pointer)
{
	return(GpuRelease(host_pointer,1u,"unregister"));
}

SparkGpuError SparkGpuFree(void *pointer)
{
	return(GpuRelease(pointer,0u,"free"));
}

SparkGpuError SparkGpuFreeHost(void *pointer)
{
	return(GpuRelease(pointer,0u,"free"));
}

static uint32_t GpuRegisteredRange(const void *pointer, uint64_t bytes, SparkDeviceRange *range)
{
	GpuAllocation *entry;
	uint32_t found = 0u;
	pthread_mutex_lock(&gpu_mutex);
	for (entry = gpu_allocations; entry != 0 && found == 0u; entry = entry->next)
		if ( entry->registered != 0u && (const uint8_t *)pointer >= (const uint8_t *)entry->host &&
			bytes <= entry->buffer.bytes - (uint64_t)((const uint8_t *)pointer - (const uint8_t *)entry->host) &&
			(uint64_t)((const uint8_t *)pointer - (const uint8_t *)entry->host) < entry->buffer.bytes )
		{
			range->buffer = &entry->buffer;
			range->offset = (uint64_t)((const uint8_t *)pointer - (const uint8_t *)entry->host);
			found = 1u;
		}
	pthread_mutex_unlock(&gpu_mutex);
	return(found);
}

static uint32_t GpuRange(const void *pointer, uint64_t bytes, SparkDeviceRange *range)
{
	if ( pointer == 0 )
		return(0u);
	if ( SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)pointer,bytes,range) == SPARK_STATUS_OK )
		return(1u);
	return(GpuRegisteredRange(pointer,bytes,range));
}

SparkGpuError SparkGpuHostGetDevicePointer(void **device_pointer, void *host_pointer, unsigned int flags)
{
	SparkDeviceRange range;
	SparkDeviceAddress address = 0u;
	if ( device_pointer == 0 || flags != 0u )
		GPU_REFUSE_IN("host_address",SPARK_STATUS_INVALID_ARGUMENT,"a device pointer lookup needs an output and no flags");
	GPU_TRY(GpuReady());
	if ( GpuRange(host_pointer,1u,&range) == 0u )
		GPU_REFUSE_IN("host_address",SPARK_STATUS_INVALID_ARGUMENT,"the host pointer is not in memory this front allocated or registered");
	GPU_TRY(SparkDeviceAddressOf(range,1u,&address));
	*device_pointer = (void *)(uintptr_t)address;
	return(SPARK_GPU_SUCCESS);
}

static SparkDeviceRange GpuAt(SparkDeviceRange range, uint64_t offset)
{
	range.offset += offset;
	return(range);
}

static SparkStatus GpuStaging(void)
{
	if ( gpu_staging.bytes != 0u )
		return(SPARK_STATUS_OK);
	return(SparkDeviceAllocate(&gpu_staging,SPARK_DEVICE_SPACE_HOST_PINNED,GPU_STAGING_BYTES,"gpu staging"));
}

static SparkStatus GpuCopyThroughStaging(SparkGpuStream stream, void *destination, const SparkDeviceRange *to, const void *source, const SparkDeviceRange *from,
	uint64_t bytes)
{
	SparkDeviceRange staging;
	SparkStatus status;
	uint64_t done;
	pthread_mutex_lock(&gpu_staging_mutex);
	status = GpuStaging();
	staging.buffer = &gpu_staging;
	staging.offset = 0u;
	for (done = 0u; status == SPARK_STATUS_OK && done < bytes; done += GPU_STAGING_BYTES)
	{
		const uint64_t chunk = bytes - done < GPU_STAGING_BYTES ? bytes - done : GPU_STAGING_BYTES;
		if ( to != 0 )
		{
			memcpy(gpu_staging.host,(const uint8_t *)source + done,(size_t)chunk);
			status = SparkDeviceCopy(stream,GpuAt(*to,done),staging,chunk);
			if ( status == SPARK_STATUS_OK )
				status = SparkDeviceQueueWait(stream,SPARK_DEVICE_WAIT_FOREVER);
		}
		else
		{
			status = SparkDeviceCopy(stream,staging,GpuAt(*from,done),chunk);
			if ( status == SPARK_STATUS_OK )
				status = SparkDeviceQueueWait(stream,SPARK_DEVICE_WAIT_FOREVER);
			if ( status == SPARK_STATUS_OK )
				memcpy((uint8_t *)destination + done,gpu_staging.host,(size_t)chunk);
		}
	}
	pthread_mutex_unlock(&gpu_staging_mutex);
	return(status);
}

static SparkGpuError GpuCopy(void *destination, const void *source, size_t bytes, SparkGpuCopyKind kind, SparkGpuStream stream)
{
	SparkDeviceRange to,from;
	uint32_t destination_known,source_known;
	if ( kind < SPARK_GPU_COPY_HOST_TO_HOST || kind > SPARK_GPU_COPY_DEFAULT || (bytes != 0u && (destination == 0 || source == 0)) )
		GPU_REFUSE_IN("copy",SPARK_STATUS_INVALID_ARGUMENT,"a copy needs a known kind and both pointers");
	if ( bytes == 0u )
		return(SPARK_GPU_SUCCESS);
	GPU_TRY(GpuReady());
	if ( GpuStream(&stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(gpu_last_error);
	destination_known = GpuRange(destination,bytes,&to);
	source_known = GpuRange(source,bytes,&from);
	if ( destination_known != 0u && source_known != 0u )
		GPU_TRY(SparkDeviceCopy(stream,to,from,bytes));
	else if ( destination_known != 0u || source_known != 0u )
		GPU_TRY(GpuCopyThroughStaging(stream,destination,destination_known != 0u ? &to : 0,source,source_known != 0u ? &from : 0,bytes));
	else
	{
		GPU_TRY(SparkDeviceQueueWait(stream,SPARK_DEVICE_WAIT_FOREVER));
		memmove(destination,source,bytes);
	}
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuMemcpyAsync(void *destination, const void *source, size_t bytes, SparkGpuCopyKind kind, SparkGpuStream stream)
{
	return(GpuCopy(destination,source,bytes,kind,stream));
}

SparkGpuError SparkGpuMemcpy(void *destination, const void *source, size_t bytes, SparkGpuCopyKind kind)
{
	SparkGpuStream stream = 0;
	SparkGpuError error = GpuCopy(destination,source,bytes,kind,0);
	if ( error == SPARK_GPU_SUCCESS && (error = SparkNvrmGpuDefaultStream(&stream)) == SPARK_GPU_SUCCESS )
		error = SparkGpuStreamSynchronize(stream);
	return(error);
}

SparkGpuError SparkGpuMemcpy2DAsync(void *destination, size_t destination_pitch, const void *source, size_t source_pitch, size_t width, size_t height,
	SparkGpuCopyKind kind, SparkGpuStream stream)
{
	size_t row;
	if ( width > destination_pitch || width > source_pitch )
		GPU_REFUSE_IN("copy",SPARK_STATUS_INVALID_ARGUMENT,"a 2D copy's width exceeds a pitch");
	if ( width == destination_pitch && width == source_pitch )
		return(GpuCopy(destination,source,width * height,kind,stream));
	for (row = 0u; row < height; row++)
	{
		const SparkGpuError error = GpuCopy((uint8_t *)destination + row * destination_pitch,(const uint8_t *)source + row * source_pitch,width,kind,stream);
		if ( error != SPARK_GPU_SUCCESS )
			SPARK_FAIL(error);
	}
	return(SPARK_GPU_SUCCESS);
}

static SparkGpuError GpuFill(void *destination, int value, size_t bytes, SparkGpuStream stream)
{
	SparkDeviceRange to;
	if ( bytes == 0u )
		return(SPARK_GPU_SUCCESS);
	GPU_TRY(GpuReady());
	if ( GpuStream(&stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(gpu_last_error);
	if ( GpuRange(destination,bytes,&to) == 0u )
		GPU_REFUSE_IN("fill",SPARK_STATUS_INVALID_ARGUMENT,"a fill needs memory this front allocated or registered");
	GPU_TRY(SparkDeviceFill(stream,to,(uint8_t)value,bytes));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuMemsetAsync(void *destination, int value, size_t bytes, SparkGpuStream stream)
{
	return(GpuFill(destination,value,bytes,stream));
}

SparkGpuError SparkGpuMemset(void *destination, int value, size_t bytes)
{
	SparkGpuStream stream = 0;
	SparkGpuError error = GpuFill(destination,value,bytes,0);
	if ( error == SPARK_GPU_SUCCESS && (error = SparkNvrmGpuDefaultStream(&stream)) == SPARK_GPU_SUCCESS )
		error = SparkGpuStreamSynchronize(stream);
	return(error);
}

SparkGpuError SparkGpuMemset2DAsync(void *destination, size_t pitch, int value, size_t width, size_t height, SparkGpuStream stream)
{
	size_t row;
	if ( width > pitch )
		GPU_REFUSE_IN("fill",SPARK_STATUS_INVALID_ARGUMENT,"a 2D fill's width exceeds its pitch");
	if ( width == pitch )
		return(GpuFill(destination,value,width * height,stream));
	for (row = 0u; row < height; row++)
	{
		const SparkGpuError error = GpuFill((uint8_t *)destination + row * pitch,value,width,stream);
		if ( error != SPARK_GPU_SUCCESS )
			SPARK_FAIL(error);
	}
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuStreamCreateWithFlags(SparkGpuStream *stream, unsigned int flags)
{
	if ( stream == 0 || flags != SPARK_GPU_STREAM_NON_BLOCKING )
		GPU_REFUSE_IN("queue_create",SPARK_STATUS_UNSUPPORTED,
			"streams are non-blocking: the hosted runtime has no legacy default stream to synchronise a blocking stream with");
	GPU_TRY(GpuReady());
	GPU_TRY(SparkDeviceQueueCreate(stream,"gpu stream"));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuStreamCreateWithPriority(SparkGpuStream *stream, unsigned int flags, int priority)
{
	if ( priority != 0 )
		GPU_REFUSE_IN("queue_create",SPARK_STATUS_UNSUPPORTED,"stream priorities are not implemented: a prioritized stream would run at the default priority");
	return(SparkGpuStreamCreateWithFlags(stream,flags));
}

SparkGpuError SparkGpuDeviceGetStreamPriorityRange(int *least_priority, int *greatest_priority)
{
	(void)least_priority;
	(void)greatest_priority;
	GPU_REFUSE_IN("priority_range",SPARK_STATUS_UNSUPPORTED,"stream priorities are not implemented: every stream would run at one priority");
}

SparkGpuError SparkGpuStreamDestroy(SparkGpuStream stream)
{
	if ( stream == 0 )
		GPU_REFUSE_IN("queue_destroy",SPARK_STATUS_INVALID_ARGUMENT,"the default stream is not destroyed");
	GPU_TRY(SparkDeviceQueueWait(stream,SPARK_DEVICE_WAIT_FOREVER));
	GPU_TRY(SparkDeviceQueueDestroy(stream));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuStreamSynchronize(SparkGpuStream stream)
{
	if ( GpuStream(&stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(gpu_last_error);
	GPU_TRY(SparkDeviceQueueWait(stream,SPARK_DEVICE_WAIT_FOREVER));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuStreamQuery(SparkGpuStream stream)
{
	SparkStatus status;
	if ( GpuStream(&stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(gpu_last_error);
	status = SparkDeviceQueueQuery(stream);
	if ( status == SPARK_STATUS_PENDING )
		return(SPARK_GPU_ERROR_NOT_READY);
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuStreamWaitEvent(SparkGpuStream stream, SparkGpuEvent event, unsigned int flags)
{
	if ( flags != 0u )
		GPU_REFUSE_IN("queue_wait_event",SPARK_STATUS_INVALID_ARGUMENT,"a stream waits on an event with no flags");
	if ( GpuStream(&stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(gpu_last_error);
	GPU_TRY(SparkDeviceQueueWaitEvent(stream,event));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuStreamIsCapturing(SparkGpuStream stream, SparkGpuCaptureStatus *status)
{
	(void)stream;
	if ( status == 0 )
		GPU_REFUSE_IN("queue_capture",SPARK_STATUS_INVALID_ARGUMENT,"a capture query needs an output");
	*status = SPARK_GPU_CAPTURE_STATUS_NONE;
	return(SPARK_GPU_SUCCESS);
}

static SparkGpuError GpuNoCommandLists(const char *operation)
{
	GPU_REFUSE_IN(operation,SPARK_STATUS_UNSUPPORTED,"stream capture and graphs need command lists, which the backend does not record yet");
}

SparkGpuError SparkGpuStreamBeginCapture(SparkGpuStream stream, SparkGpuCaptureMode mode)
{
	(void)stream;
	(void)mode;
	return(GpuNoCommandLists("capture_begin"));
}

SparkGpuError SparkGpuStreamEndCapture(SparkGpuStream stream, SparkGpuGraph *graph)
{
	(void)stream;
	(void)graph;
	return(GpuNoCommandLists("capture_end"));
}

SparkGpuError SparkGpuGraphInstantiate(SparkGpuGraphExec *exec, SparkGpuGraph graph, unsigned long long flags)
{
	(void)exec;
	(void)graph;
	(void)flags;
	return(GpuNoCommandLists("graph_instantiate"));
}

SparkGpuError SparkGpuGraphUpload(SparkGpuGraphExec exec, SparkGpuStream stream)
{
	(void)exec;
	(void)stream;
	return(GpuNoCommandLists("graph_upload"));
}

SparkGpuError SparkGpuGraphLaunch(SparkGpuGraphExec exec, SparkGpuStream stream)
{
	(void)exec;
	(void)stream;
	return(GpuNoCommandLists("graph_launch"));
}

SparkGpuError SparkGpuGraphGetNodes(SparkGpuGraph graph, SparkGpuGraphNode *nodes, size_t *count)
{
	(void)graph;
	(void)nodes;
	(void)count;
	return(GpuNoCommandLists("graph_nodes"));
}

SparkGpuError SparkGpuGraphExecDestroy(SparkGpuGraphExec exec)
{
	(void)exec;
	return(GpuNoCommandLists("graph_exec_destroy"));
}

SparkGpuError SparkGpuGraphDestroy(SparkGpuGraph graph)
{
	(void)graph;
	return(GpuNoCommandLists("graph_destroy"));
}

SparkGpuError SparkGpuLaunchHostFunc(SparkGpuStream stream, SparkGpuHostFunction function, void *context)
{
	if ( GpuStream(&stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(gpu_last_error);
	GPU_TRY(SparkDeviceQueueHostCallback(stream,function,context));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuEventCreateWithFlags(SparkGpuEvent *event, unsigned int flags)
{
	if ( event == 0 || (flags & ~SPARK_GPU_EVENT_DISABLE_TIMING) != 0u )
		GPU_REFUSE_IN("event_create",SPARK_STATUS_INVALID_ARGUMENT,"an event needs an output and only the disable-timing flag");
	GPU_TRY(GpuReady());
	GPU_TRY(SparkDeviceEventCreate(event));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuEventCreate(SparkGpuEvent *event)
{
	return(SparkGpuEventCreateWithFlags(event,0u));
}

SparkGpuError SparkGpuEventDestroy(SparkGpuEvent event)
{
	SparkStatus status = SparkDeviceEventDestroy(event);
	if ( status == SPARK_STATUS_BUSY )
	{
		GPU_TRY(SparkDeviceSynchronize(SPARK_DEVICE_WAIT_FOREVER));
		status = SparkDeviceEventDestroy(event);
	}
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuEventRecord(SparkGpuEvent event, SparkGpuStream stream)
{
	if ( GpuStream(&stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(gpu_last_error);
	GPU_TRY(SparkDeviceEventRecord(event,stream));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuEventQuery(SparkGpuEvent event)
{
	const SparkStatus status = SparkDeviceEventQuery(event);
	if ( status == SPARK_STATUS_PENDING )
		return(SPARK_GPU_ERROR_NOT_READY);
	if ( status != SPARK_STATUS_OK )
		GPU_FAIL(status);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuEventSynchronize(SparkGpuEvent event)
{
	GPU_TRY(SparkDeviceEventWait(event,SPARK_DEVICE_WAIT_FOREVER));
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuEventElapsedTime(float *milliseconds, SparkGpuEvent start, SparkGpuEvent stop)
{
	uint64_t nanoseconds = 0u;
	if ( milliseconds == 0 )
		GPU_REFUSE_IN("event_elapsed",SPARK_STATUS_INVALID_ARGUMENT,"an elapsed time needs an output");
	GPU_TRY(SparkDeviceEventElapsed(start,stop,&nanoseconds));
	*milliseconds = (float)((double)nanoseconds / 1.0e6);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuGetDevice(int *device)
{
	if ( device == 0 )
		GPU_REFUSE_IN("device",SPARK_STATUS_INVALID_ARGUMENT,"a device query needs an output");
	*device = gpu_ordinal;
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuSetDevice(int device)
{
	if ( device < 0 || (SparkDeviceGetProperties() != 0 && device != gpu_ordinal) )
	{
		(void)SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"device",GPU_OWNER,device,"one device per process, chosen before its first use");
		gpu_last_error = SPARK_GPU_ERROR_INVALID_DEVICE;
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_DEVICE);
	}
	gpu_ordinal = device;
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuDeviceGetAttribute(int *value, SparkGpuDeviceAttribute attribute, int device)
{
	const SparkDeviceProperties *properties;
	if ( value == 0 || device != gpu_ordinal )
		GPU_REFUSE_IN("device_attribute",SPARK_STATUS_INVALID_ARGUMENT,"an attribute query needs an output and this process's device");
	GPU_TRY(GpuReady());
	properties = SparkDeviceGetProperties();
	if ( attribute != SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT )
		GPU_REFUSE_IN("device_attribute",SPARK_STATUS_UNSUPPORTED,"the backend answers only the multiprocessor count yet");
	*value = (int)properties->compute_units;
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuDeviceSynchronize(void)
{
	GPU_TRY(GpuReady());
	GPU_TRY(SparkDeviceSynchronize(SPARK_DEVICE_WAIT_FOREVER));
	return(SPARK_GPU_SUCCESS);
}

static uint64_t GpuMemInfoField(const char *text, const char *field)
{
	const char *at = strstr(text,field);
	unsigned long long kib = 0u;
	if ( at == 0 || sscanf(at + strlen(field)," %llu kB",&kib) != 1 )
		return(0u);
	return((uint64_t)kib * 1024u);
}

SparkGpuError SparkGpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
	const SparkDeviceProperties *properties;
	char text[4096];
	uint64_t total,available;
	size_t length;
	FILE *file;
	if ( free_bytes == 0 || total_bytes == 0 )
		GPU_REFUSE_IN("memory_info",SPARK_STATUS_INVALID_ARGUMENT,"a memory query needs both outputs");
	GPU_TRY(GpuReady());
	properties = SparkDeviceGetProperties();
	if ( (properties->properties & SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY) == 0u )
		GPU_REFUSE_IN("memory_info",SPARK_STATUS_UNSUPPORTED,"free memory on a discrete GPU needs the module's framebuffer report, not implemented yet");
	if ( (file = fopen(GPU_MEMINFO_PATH,"r")) == 0 )
		GPU_REFUSE_IN("memory_info",SPARK_STATUS_IO_ERROR,"cannot read " GPU_MEMINFO_PATH " for the unified GPU's free memory");
	length = fread(text,1u,sizeof(text) - 1u,file);
	fclose(file);
	text[length] = '\0';
	total = GpuMemInfoField(text,"MemTotal:");
	available = GpuMemInfoField(text,"MemFree:");
	if ( total == 0u || available == 0u )
		GPU_REFUSE_IN("memory_info",SPARK_STATUS_PARSE_ERROR,GPU_MEMINFO_PATH " names no MemTotal or MemFree");
	total = total < properties->memory_budget_bytes ? total : properties->memory_budget_bytes;
	*total_bytes = (size_t)total;
	*free_bytes = (size_t)(available < total ? available : total);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuFuncSetAttribute(const void *function, SparkGpuFunctionAttribute attribute, int value)
{
	if ( attribute != SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES || value < 0 )
		GPU_REFUSE_IN("function_attribute",SPARK_STATUS_UNSUPPORTED,"the backend sets only a kernel's dynamic shared memory");
	GPU_TRY(GpuReady());
	return(SparkNvrmSharedMemoryOptIn(function,(uint32_t)value));
}

SparkGpuError SparkGpuFuncGetAttributes(SparkGpuFunctionAttributes *attributes, const void *function)
{
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	uint32_t registers = 0u,stack = 0u,shared = 0u;
	if ( attributes == 0 )
		GPU_REFUSE_IN("function_attributes",SPARK_STATUS_INVALID_ARGUMENT,"a function query needs an output");
	GPU_TRY(GpuReady());
	GPU_TRY(SparkNvrmRegisteredKernel(function,&kernel));
	GPU_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	GPU_TRY(SparkNvrmKernelResources(kernel,&registers,&stack,&shared));
	memset(attributes,0,sizeof(*attributes));
	attributes->maxThreadsPerBlock = (int)info.max_threads;
	attributes->numRegs = (int)registers;
	attributes->sharedSizeBytes = info.static_shared_bytes;
	attributes->localSizeBytes = stack;
	attributes->maxDynamicSharedSizeBytes = (int)(SparkDeviceGetProperties()->max_shared_bytes - info.static_shared_bytes);
	return(SPARK_GPU_SUCCESS);
}

SparkGpuError SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, const void *function, int block_threads, size_t shared_bytes)
{
	SparkNvrmOccupancyLimits limits;
	SparkDeviceKernel *kernel = 0;
	uint32_t registers = 0u,stack = 0u,shared = 0u;
	if ( blocks == 0 || block_threads <= 0 || shared_bytes > UINT32_MAX )
		GPU_REFUSE_IN("occupancy",SPARK_STATUS_INVALID_ARGUMENT,"an occupancy query needs an output, a positive block size and at most 4 GiB of shared memory");
	GPU_TRY(GpuReady());
	GPU_TRY(SparkNvrmRegisteredKernel(function,&kernel));
	GPU_TRY(SparkNvrmKernelResources(kernel,&registers,&stack,&shared));
	if ( SparkNvrmOccupancyLimitsOf(&limits) == 0 )
		GPU_REFUSE_IN("occupancy",SPARK_STATUS_INVALID_ARGUMENT,"the device reported no per-multiprocessor limits");
	*blocks = (int)SparkNvrmOccupancy(&limits,(uint32_t)block_threads,registers,shared + (uint32_t)shared_bytes);
	return(SPARK_GPU_SUCCESS);
}
