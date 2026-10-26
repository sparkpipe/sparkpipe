#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "runtime/device/device_backend.h"

typedef enum CudaMemoryKind
{
	CUDA_MEMORY_DEVICE = 1,
	CUDA_MEMORY_MANAGED = 2,
	CUDA_MEMORY_PINNED = 3,
	CUDA_MEMORY_HEAP = 4,
	CUDA_MEMORY_REGISTERED = 5,
	CUDA_MEMORY_FILE = 6
} CudaMemoryKind;

typedef struct CudaAllocation
{
	CudaMemoryKind kind;
	uint32_t pageable;
	uint32_t registered;
	void *base;
	void *mapping;
	uint64_t mapping_bytes;
} CudaAllocation;

#define CUDA_BOUNCE_BYTES (8ull << 20)

typedef struct CudaQueue
{
	cudaStream_t stream;
	uint64_t completed;
	uint8_t *bounce;
} CudaQueue;

typedef struct CudaCallback
{
	SparkDeviceHostFunction function;
	void *context;
	uint8_t *fill;
	uint8_t *copy_destination;
	const uint8_t *copy_source;
	uint64_t bytes;
	uint8_t value;
} CudaCallback;

static int cuda_ordinal;
static int cuda_pageable_access;
static uint64_t cuda_page_bytes;

static SparkStatus CudaStatus(cudaError_t error, const char *operation, const char *owner)
{
	SparkStatus status;
	if ( error == cudaSuccess )
		return(SPARK_STATUS_OK);
	if ( error == cudaErrorNotReady )
	{
		(void)cudaGetLastError();
		return(SPARK_STATUS_PENDING);
	}
	if ( error == cudaErrorMemoryAllocation )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	else if ( error == cudaErrorInvalidValue || error == cudaErrorInvalidDevicePointer || error == cudaErrorInvalidResourceHandle ||
		error == cudaErrorHostMemoryAlreadyRegistered || error == cudaErrorHostMemoryNotRegistered )
		status = error == cudaErrorHostMemoryAlreadyRegistered ? SPARK_STATUS_DUPLICATE : SPARK_STATUS_INVALID_ARGUMENT;
	else if ( error == cudaErrorNotSupported || error == cudaErrorNotPermitted )
		status = SPARK_STATUS_UNSUPPORTED;
	else if ( error == cudaErrorNoDevice || error == cudaErrorInvalidDevice || error == cudaErrorNoKernelImageForDevice )
		status = SPARK_STATUS_TARGET_MISMATCH;
	else
		status = SPARK_STATUS_IO_ERROR;
	(void)cudaGetLastError();
	return(SparkDeviceRecordError(status,operation,owner,(int32_t)error,cudaGetErrorString(error)));
}

SparkStatus SparkDeviceBackendInitialize(const SparkDeviceConfiguration *configuration, SparkDeviceProperties *properties)
{
	struct cudaDeviceProp device;
	int count = 0, managed = 0, pageable = 0;
	long page = sysconf(_SC_PAGESIZE);
	SparkStatus status = CudaStatus(cudaGetDeviceCount(&count),"initialize","");
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( (int)configuration->ordinal >= count )
		return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"initialize","",0,"no CUDA device with that ordinal"));
	cuda_ordinal = (int)configuration->ordinal;
	cuda_page_bytes = page > 0 ? (uint64_t)page : 4096u;
	status = CudaStatus(cudaSetDevice(cuda_ordinal),"initialize","");
	if ( status == SPARK_STATUS_OK )
		status = CudaStatus(cudaGetDeviceProperties(&device,cuda_ordinal),"initialize","");
	if ( status != SPARK_STATUS_OK )
		return(status);
	(void)cudaDeviceGetAttribute(&managed,cudaDevAttrManagedMemory,cuda_ordinal);
	(void)cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,cuda_ordinal);
	cuda_pageable_access = pageable;
	snprintf(properties->backend,sizeof(properties->backend),"%s","cuda");
	snprintf(properties->target,sizeof(properties->target),"cuda.sm%d%d",device.major,device.minor);
	memcpy(properties->name,device.name,sizeof(properties->name) - 1u);
	properties->name[sizeof(properties->name) - 1u] = '\0';
	properties->page_bytes = (uint32_t)cuda_page_bytes;
	properties->compute_units = (uint32_t)device.multiProcessorCount;
	properties->simd_width = (uint32_t)device.warpSize;
	properties->memory_budget_bytes = configuration->memory_budget_bytes != 0u ? configuration->memory_budget_bytes : (uint64_t)device.totalGlobalMem;
	properties->max_buffer_bytes = (uint64_t)device.totalGlobalMem;
	properties->properties = SPARK_DEVICE_PROPERTY_SHARED_ADDRESS | (device.integrated != 0 ? SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY : 0u);
	(void)managed;
	(void)pageable;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	return(SPARK_STATUS_OK);
}

static CudaAllocation *CudaNew(SparkDeviceAllocation *allocation, CudaMemoryKind kind, const char *operation)
{
	CudaAllocation *cuda = (CudaAllocation *)calloc(1u,sizeof(*cuda));
	if ( cuda == 0 )
	{
		SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,allocation->owner,ENOMEM,"out of host memory");
		return(0);
	}
	cuda->kind = kind;
	return(cuda);
}

SparkStatus SparkDeviceBackendAllocate(SparkDeviceAllocation *allocation)
{
	CudaAllocation *cuda;
	SparkStatus status = SPARK_STATUS_OK;
	void *device = 0;
	if ( allocation->space == SPARK_DEVICE_SPACE_HOST_PAGEABLE )
	{
		cuda = CudaNew(allocation,CUDA_MEMORY_HEAP,"allocate");
		if ( cuda == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		cuda->pageable = 1u;
		cuda->base = calloc(1u,(size_t)allocation->bytes);
		if ( cuda->base == 0 )
		{
			free(cuda);
			return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,ENOMEM,"out of host memory"));
		}
		allocation->host = cuda->base;
		allocation->device = 0u;
		allocation->backend = cuda;
		return(SPARK_STATUS_OK);
	}
	if ( allocation->space == SPARK_DEVICE_SPACE_DEVICE_PRIVATE )
	{
		cuda = CudaNew(allocation,CUDA_MEMORY_DEVICE,"allocate");
		if ( cuda == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = CudaStatus(cudaMalloc(&cuda->base,(size_t)allocation->bytes),"allocate",allocation->owner);
		allocation->host = 0;
		device = cuda->base;
	}
	else if ( allocation->space == SPARK_DEVICE_SPACE_SHARED_COHERENT )
	{
		cuda = CudaNew(allocation,CUDA_MEMORY_MANAGED,"allocate");
		if ( cuda == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = CudaStatus(cudaMallocManaged(&cuda->base,(size_t)allocation->bytes,cudaMemAttachGlobal),"allocate",allocation->owner);
		allocation->host = cuda->base;
		device = cuda->base;
	}
	else
	{
		cuda = CudaNew(allocation,CUDA_MEMORY_PINNED,"allocate");
		if ( cuda == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = CudaStatus(cudaHostAlloc(&cuda->base,(size_t)allocation->bytes,cudaHostAllocPortable | cudaHostAllocMapped),"allocate",allocation->owner);
		if ( status == SPARK_STATUS_OK )
			status = CudaStatus(cudaHostGetDevicePointer(&device,cuda->base,0u),"allocate",allocation->owner);
		allocation->host = cuda->base;
	}
	if ( status != SPARK_STATUS_OK )
	{
		if ( cuda->base != 0 )
			allocation->space == SPARK_DEVICE_SPACE_HOST_PINNED ? (void)cudaFreeHost(cuda->base) : (void)cudaFree(cuda->base);
		free(cuda);
		return(status);
	}
	if ( allocation->space == SPARK_DEVICE_SPACE_DEVICE_PRIVATE || allocation->space == SPARK_DEVICE_SPACE_SHARED_COHERENT )
		status = CudaStatus(cudaMemset(cuda->base,0,(size_t)allocation->bytes),"allocate",allocation->owner);
	else
		memset(cuda->base,0,(size_t)allocation->bytes);
	allocation->device = (SparkDeviceAddress)(uintptr_t)device;
	allocation->backend = cuda;
	if ( status != SPARK_STATUS_OK )
		SparkDeviceBackendRelease(allocation);
	return(status);
}

SparkStatus SparkDeviceBackendRegister(SparkDeviceAllocation *allocation, void *host)
{
	CudaAllocation *cuda = CudaNew(allocation,CUDA_MEMORY_REGISTERED,"register");
	void *device = 0;
	SparkStatus status;
	if ( cuda == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = CudaStatus(cudaHostRegister(host,(size_t)allocation->bytes,cudaHostRegisterPortable | cudaHostRegisterMapped),"register",allocation->owner);
	if ( status == SPARK_STATUS_OK )
	{
		cuda->base = host;
		status = CudaStatus(cudaHostGetDevicePointer(&device,host,0u),"register",allocation->owner);
		if ( status != SPARK_STATUS_OK )
			(void)cudaHostUnregister(host);
	}
	if ( status != SPARK_STATUS_OK )
	{
		free(cuda);
		return(status);
	}
	allocation->host = host;
	allocation->device = (SparkDeviceAddress)(uintptr_t)device;
	allocation->backend = cuda;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendMapFile(SparkDeviceAllocation *allocation, int fd, uint64_t offset)
{
	CudaAllocation *cuda = CudaNew(allocation,CUDA_MEMORY_FILE,"map_file");
	uint64_t aligned = offset / cuda_page_bytes * cuda_page_bytes;
	void *device = 0;
	SparkStatus status;
	if ( cuda == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	cuda->mapping_bytes = (offset - aligned + allocation->bytes + cuda_page_bytes - 1u) / cuda_page_bytes * cuda_page_bytes;
	cuda->mapping = mmap(0,(size_t)cuda->mapping_bytes,PROT_READ,MAP_SHARED,fd,(off_t)aligned);
	if ( cuda->mapping == MAP_FAILED )
	{
		int error = errno;
		free(cuda);
		return(SparkDeviceRecordError(error == ENOMEM ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_INVALID_ARGUMENT,
			"map_file",allocation->owner,error,"cannot map the file range"));
	}
	if ( cuda_pageable_access != 0 )
	{
		cuda->pageable = 1u;
		device = cuda->mapping;
		status = SPARK_STATUS_OK;
	}
	else
	{
		status = CudaStatus(cudaHostRegister(cuda->mapping,(size_t)cuda->mapping_bytes,
			cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly),"map_file",allocation->owner);
		if ( status == SPARK_STATUS_OK )
		{
			cuda->registered = 1u;
			status = CudaStatus(cudaHostGetDevicePointer(&device,cuda->mapping,0u),"map_file",allocation->owner);
			if ( status != SPARK_STATUS_OK )
				(void)cudaHostUnregister(cuda->mapping);
		}
	}
	if ( status != SPARK_STATUS_OK )
	{
		munmap(cuda->mapping,(size_t)cuda->mapping_bytes);
		free(cuda);
		return(status);
	}
	cuda->base = cuda->mapping;
	allocation->host = (uint8_t *)cuda->mapping + (offset - aligned);
	allocation->device = (SparkDeviceAddress)(uintptr_t)device + (offset - aligned);
	allocation->backend = cuda;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendRelease(SparkDeviceAllocation *allocation)
{
	CudaAllocation *cuda = (CudaAllocation *)allocation->backend;
	if ( cuda == 0 )
		return;
	if ( cuda->kind == CUDA_MEMORY_DEVICE || cuda->kind == CUDA_MEMORY_MANAGED )
		(void)cudaFree(cuda->base);
	else if ( cuda->kind == CUDA_MEMORY_PINNED )
		(void)cudaFreeHost(cuda->base);
	else if ( cuda->kind == CUDA_MEMORY_HEAP )
		free(cuda->base);
	else if ( cuda->kind == CUDA_MEMORY_REGISTERED )
		(void)cudaHostUnregister(cuda->base);
	else if ( cuda->kind == CUDA_MEMORY_FILE )
	{
		if ( cuda->registered != 0u )
			(void)cudaHostUnregister(cuda->mapping);
		munmap(cuda->mapping,(size_t)cuda->mapping_bytes);
	}
	free(cuda);
	allocation->backend = 0;
}

static cudaStream_t CudaStream(SparkDeviceQueue *queue)
{
	return(((CudaQueue *)queue->backend)->stream);
}

static void *CudaPointer(const SparkDeviceAllocation *allocation, uint64_t offset, uint32_t device_side)
{
	if ( device_side != 0u )
		return((void *)(uintptr_t)(allocation->device + offset));
	return((uint8_t *)allocation->host + offset);
}

static SparkStatus CudaCopyByCallback(SparkDeviceQueue *queue, uint8_t *destination, const uint8_t *source, uint64_t bytes);
static SparkStatus CudaBounce(SparkDeviceQueue *queue, const char *operation);

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	uint32_t source_pageable = ((const CudaAllocation *)source->backend)->pageable;
	uint32_t destination_pageable = ((const CudaAllocation *)destination->backend)->pageable;
	enum cudaMemcpyKind kind = cudaMemcpyHostToHost;
	uint32_t source_device = 0u, destination_device = 0u;
	if ( direction == SPARK_DEVICE_HOST_TO_DEVICE )
	{
		kind = cudaMemcpyHostToDevice;
		destination_device = 1u;
	}
	else if ( direction == SPARK_DEVICE_DEVICE_TO_HOST )
	{
		kind = cudaMemcpyDeviceToHost;
		source_device = 1u;
	}
	else if ( direction == SPARK_DEVICE_DEVICE_TO_DEVICE )
	{
		kind = cudaMemcpyDeviceToDevice;
		source_device = destination_device = 1u;
	}
	if ( direction == SPARK_DEVICE_HOST_TO_HOST && (source_pageable != 0u || destination_pageable != 0u) )
		return(CudaCopyByCallback(queue,(uint8_t *)CudaPointer(destination,destination_offset,0u),
			(const uint8_t *)CudaPointer(source,source_offset,0u),bytes));
	if ( (direction == SPARK_DEVICE_HOST_TO_DEVICE && source_pageable != 0u) || (direction == SPARK_DEVICE_DEVICE_TO_HOST && destination_pageable != 0u) )
	{
		uint64_t done;
		SparkStatus status = CudaBounce(queue,"copy");
		for ( done = 0u; status == SPARK_STATUS_OK && done < bytes; done += CUDA_BOUNCE_BYTES )
		{
			uint64_t chunk = bytes - done < CUDA_BOUNCE_BYTES ? bytes - done : CUDA_BOUNCE_BYTES;
			if ( direction == SPARK_DEVICE_HOST_TO_DEVICE )
			{
				status = CudaCopyByCallback(queue,cuda->bounce,(const uint8_t *)CudaPointer(source,source_offset + done,0u),chunk);
				if ( status == SPARK_STATUS_OK )
					status = CudaStatus(cudaMemcpyAsync(CudaPointer(destination,destination_offset + done,1u),cuda->bounce,(size_t)chunk,
						cudaMemcpyHostToDevice,cuda->stream),"copy",queue->owner);
			}
			else
			{
				status = CudaStatus(cudaMemcpyAsync(cuda->bounce,CudaPointer(source,source_offset + done,1u),(size_t)chunk,
					cudaMemcpyDeviceToHost,cuda->stream),"copy",queue->owner);
				if ( status == SPARK_STATUS_OK )
					status = CudaCopyByCallback(queue,(uint8_t *)CudaPointer(destination,destination_offset + done,0u),cuda->bounce,chunk);
			}
		}
		return(status);
	}
	return(CudaStatus(cudaMemcpyAsync(CudaPointer(destination,destination_offset,destination_device),
		CudaPointer(source,source_offset,source_device),(size_t)bytes,kind,CudaStream(queue)),"copy",queue->owner));
}

static void CUDART_CB CudaCallbackTrampoline(void *argument)
{
	CudaCallback *callback = (CudaCallback *)argument;
	if ( callback->copy_destination != 0 )
		memcpy(callback->copy_destination,callback->copy_source,(size_t)callback->bytes);
	else if ( callback->fill != 0 )
		memset(callback->fill,callback->value,(size_t)callback->bytes);
	else
		callback->function(callback->context);
	free(callback);
}

static SparkStatus CudaEnqueueCallback(SparkDeviceQueue *queue, CudaCallback *callback, const char *operation)
{
	SparkStatus status;
	CudaCallback *copy = (CudaCallback *)malloc(sizeof(*copy));
	if ( copy == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,queue->owner,ENOMEM,"out of host memory"));
	*copy = *callback;
	status = CudaStatus(cudaLaunchHostFunc(CudaStream(queue),CudaCallbackTrampoline,copy),operation,queue->owner);
	if ( status != SPARK_STATUS_OK )
		free(copy);
	return(status);
}

static SparkStatus CudaCopyByCallback(SparkDeviceQueue *queue, uint8_t *destination, const uint8_t *source, uint64_t bytes)
{
	CudaCallback callback;
	memset(&callback,0,sizeof(callback));
	callback.copy_destination = destination;
	callback.copy_source = source;
	callback.bytes = bytes;
	return(CudaEnqueueCallback(queue,&callback,"copy"));
}

static SparkStatus CudaBounce(SparkDeviceQueue *queue, const char *operation)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	if ( cuda->bounce != 0 )
		return(SPARK_STATUS_OK);
	return(CudaStatus(cudaHostAlloc((void **)&cuda->bounce,(size_t)CUDA_BOUNCE_BYTES,cudaHostAllocPortable),operation,queue->owner));
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	CudaCallback callback;
	if ( destination->device != 0u )
		return(CudaStatus(cudaMemsetAsync((void *)(uintptr_t)(destination->device + offset),value,(size_t)bytes,CudaStream(queue)),"fill",queue->owner));
	memset(&callback,0,sizeof(callback));
	callback.fill = (uint8_t *)destination->host + offset;
	callback.bytes = bytes;
	callback.value = value;
	return(CudaEnqueueCallback(queue,&callback,"fill"));
}

SparkStatus SparkDeviceBackendQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context)
{
	CudaCallback callback;
	memset(&callback,0,sizeof(callback));
	callback.function = function;
	callback.context = context;
	return(CudaEnqueueCallback(queue,&callback,"host_callback"));
}

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue)
{
	CudaQueue *cuda = (CudaQueue *)calloc(1u,sizeof(*cuda));
	SparkStatus status;
	if ( cuda == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",queue->owner,ENOMEM,"out of host memory"));
	status = CudaStatus(cudaStreamCreateWithFlags(&cuda->stream,cudaStreamNonBlocking),"queue_create",queue->owner);
	if ( status != SPARK_STATUS_OK )
	{
		free(cuda);
		return(status);
	}
	queue->backend = cuda;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	if ( cuda == 0 )
		return;
	(void)cudaStreamSynchronize(cuda->stream);
	(void)cudaStreamDestroy(cuda->stream);
	if ( cuda->bounce != 0 )
		(void)cudaFreeHost(cuda->bounce);
	free(cuda);
	queue->backend = 0;
}

static uint64_t CudaNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static SparkStatus CudaPoll(cudaError_t (*query)(void *), void *handle, cudaError_t (*block)(void *), uint64_t timeout_ns, const char *operation, const char *owner)
{
	uint64_t deadline;
	cudaError_t error;
	if ( timeout_ns == SPARK_DEVICE_WAIT_FOREVER )
		return(CudaStatus(block(handle),operation,owner));
	deadline = CudaNowNs() + timeout_ns;
	for ( ;; )
	{
		struct timespec pause = {0, 50000};
		error = query(handle);
		if ( error != cudaErrorNotReady )
			return(CudaStatus(error,operation,owner));
		if ( CudaNowNs() >= deadline )
			return(SPARK_STATUS_PENDING);
		nanosleep(&pause,0);
	}
}

static cudaError_t CudaStreamQueryHandle(void *handle)
{
	return(cudaStreamQuery((cudaStream_t)handle));
}

static cudaError_t CudaStreamSynchronizeHandle(void *handle)
{
	return(cudaStreamSynchronize((cudaStream_t)handle));
}

static cudaError_t CudaEventQueryHandle(void *handle)
{
	return(cudaEventQuery((cudaEvent_t)handle));
}

static cudaError_t CudaEventSynchronizeHandle(void *handle)
{
	return(cudaEventSynchronize((cudaEvent_t)handle));
}

SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	SparkStatus status;
	if ( __atomic_load_n(&cuda->completed,__ATOMIC_ACQUIRE) >= sequence )
		return(SPARK_STATUS_OK);
	status = CudaPoll(CudaStreamQueryHandle,(void *)cuda->stream,CudaStreamSynchronizeHandle,timeout_ns,"queue_wait",queue->owner);
	if ( status == SPARK_STATUS_OK )
	{
		uint64_t known = __atomic_load_n(&cuda->completed,__ATOMIC_ACQUIRE);
		while ( known < sequence && !__atomic_compare_exchange_n(&cuda->completed,&known,sequence,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE) )
		{
		}
	}
	return(status);
}

uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	if ( cudaStreamQuery(cuda->stream) == cudaSuccess )
	{
		uint64_t known = __atomic_load_n(&cuda->completed,__ATOMIC_ACQUIRE);
		while ( known < queue->submitted && !__atomic_compare_exchange_n(&cuda->completed,&known,queue->submitted,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE) )
		{
		}
	}
	else
		(void)cudaGetLastError();
	return(__atomic_load_n(&cuda->completed,__ATOMIC_ACQUIRE));
}

SparkStatus SparkDeviceBackendEventCreate(SparkDeviceEvent *event)
{
	cudaEvent_t created;
	SparkStatus status = CudaStatus(cudaEventCreate(&created),"event_create","");
	if ( status == SPARK_STATUS_OK )
		event->backend = (void *)created;
	return(status);
}

void SparkDeviceBackendEventDestroy(SparkDeviceEvent *event)
{
	if ( event->backend != 0 )
		(void)cudaEventDestroy((cudaEvent_t)event->backend);
	event->backend = 0;
}

SparkStatus SparkDeviceBackendEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue)
{
	return(CudaStatus(cudaEventRecord((cudaEvent_t)event->backend,CudaStream(queue)),"event_record",queue->owner));
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	return(CudaStatus(cudaStreamWaitEvent(CudaStream(queue),(cudaEvent_t)event->backend,0u),"queue_wait_event",queue->owner));
}

SparkStatus SparkDeviceBackendEventWait(SparkDeviceEvent *event, uint64_t timeout_ns)
{
	if ( timeout_ns == 0u )
	{
		cudaError_t error = cudaEventQuery((cudaEvent_t)event->backend);
		if ( error == cudaErrorNotReady )
		{
			(void)cudaGetLastError();
			return(SPARK_STATUS_PENDING);
		}
		return(CudaStatus(error,"event_wait",""));
	}
	return(CudaPoll(CudaEventQueryHandle,event->backend,CudaEventSynchronizeHandle,timeout_ns,"event_wait",""));
}

SparkStatus SparkDeviceBackendEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds)
{
	float milliseconds = 0.0f;
	SparkStatus status = CudaStatus(cudaEventElapsedTime(&milliseconds,(cudaEvent_t)start->backend,(cudaEvent_t)stop->backend),"event_elapsed","");
	if ( status == SPARK_STATUS_OK )
		*nanoseconds = (uint64_t)((double)milliseconds * 1000000.0);
	return(status);
}

SparkStatus SparkDeviceBackendLibraryLoad(SparkDeviceLibrary *library, const char *path)
{
	(void)path;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"library_load",library->owner,0,"the CUDA backend loads no device kernel libraries"));
}

void SparkDeviceBackendLibraryUnload(SparkDeviceLibrary *library)
{
	library->backend = 0;
}

SparkStatus SparkDeviceBackendKernelCreate(SparkDeviceKernel *kernel, const char *name)
{
	(void)kernel;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_create",name,0,"the CUDA backend creates no device kernels"));
}

void SparkDeviceBackendKernelDestroy(SparkDeviceKernel *kernel)
{
	kernel->backend = 0;
}

SparkStatus SparkDeviceBackendKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch,
	SparkDeviceAllocation *const *allocations)
{
	(void)queue;
	(void)launch;
	(void)allocations;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_launch",kernel->name,0,"the CUDA backend launches no device kernels"));
}
