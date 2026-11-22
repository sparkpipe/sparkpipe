#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include "sparkpipe/spark_clock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "runtime/device/device_backend.h"

typedef enum CudaMemoryKind
{
	CUDA_MEMORY_DEVICE = 1,
	CUDA_MEMORY_MANAGED = 2,
	CUDA_MEMORY_PINNED = 3,
	CUDA_MEMORY_HEAP = 4,
	CUDA_MEMORY_REGISTERED = 5,
	CUDA_MEMORY_FILE = 6,
	CUDA_MEMORY_ADOPTED = 7
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
	uint32_t adopted;
	uint32_t capturing;
	pthread_mutex_t gate;
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
	uint32_t retained;
} CudaCallback;

typedef struct CudaDriver
{
	CUresult (*context_current)(CUcontext *context);
	CUresult (*context_set)(CUcontext context);
	CUresult (*granularity)(size_t *granularity, const CUmemAllocationProp *properties, CUmemAllocationGranularity_flags option);
	CUresult (*create)(CUmemGenericAllocationHandle *handle, size_t bytes, const CUmemAllocationProp *properties, unsigned long long flags);
	CUresult (*release)(CUmemGenericAllocationHandle handle);
	CUresult (*export_fd)(void *shareable, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType type, unsigned long long flags);
	CUresult (*import_fd)(CUmemGenericAllocationHandle *handle, void *shareable, CUmemAllocationHandleType type);
	CUresult (*reserve)(CUdeviceptr *address, size_t bytes, size_t alignment, CUdeviceptr hint, unsigned long long flags);
	CUresult (*address_free)(CUdeviceptr address, size_t bytes);
	CUresult (*map)(CUdeviceptr address, size_t bytes, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
	CUresult (*set_access)(CUdeviceptr address, size_t bytes, const CUmemAccessDesc *access, size_t count);
	CUresult (*unmap)(CUdeviceptr address, size_t bytes);
	CUresult (*tensor_map_encode)(CUtensorMap *map, CUtensorMapDataType type, cuuint32_t rank, void *address, const cuuint64_t *dimension,
		const cuuint64_t *stride, const cuuint32_t *box, const cuuint32_t *element_stride, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
		CUtensorMapL2promotion promotion, CUtensorMapFloatOOBfill fill);
} CudaDriver;

static const struct
{
	const char *symbol;
	size_t offset;
} cuda_driver_symbols[] = {
	{"cuCtxGetCurrent",offsetof(CudaDriver,context_current)},
	{"cuCtxSetCurrent",offsetof(CudaDriver,context_set)},
	{"cuMemGetAllocationGranularity",offsetof(CudaDriver,granularity)},
	{"cuMemCreate",offsetof(CudaDriver,create)},
	{"cuMemRelease",offsetof(CudaDriver,release)},
	{"cuMemExportToShareableHandle",offsetof(CudaDriver,export_fd)},
	{"cuMemImportFromShareableHandle",offsetof(CudaDriver,import_fd)},
	{"cuMemAddressReserve",offsetof(CudaDriver,reserve)},
	{"cuMemAddressFree",offsetof(CudaDriver,address_free)},
	{"cuMemMap",offsetof(CudaDriver,map)},
	{"cuMemSetAccess",offsetof(CudaDriver,set_access)},
	{"cuMemUnmap",offsetof(CudaDriver,unmap)},
	{"cuTensorMapEncodeTiled",offsetof(CudaDriver,tensor_map_encode)},
};

static CudaDriver cuda_driver;
static CUcontext cuda_instance;
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

static SparkStatus CudaDriverStatus(CUresult result, const char *operation, const char *owner, const char *call)
{
	char text[96];
	SparkStatus status;
	if ( result == CUDA_SUCCESS )
		return(SPARK_STATUS_OK);
	if ( result == CUDA_ERROR_OUT_OF_MEMORY )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	else if ( result == CUDA_ERROR_INVALID_VALUE || result == CUDA_ERROR_INVALID_HANDLE || result == CUDA_ERROR_ALREADY_MAPPED ||
		result == CUDA_ERROR_NOT_MAPPED )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	else if ( result == CUDA_ERROR_NOT_SUPPORTED || result == CUDA_ERROR_NOT_PERMITTED )
		status = SPARK_STATUS_UNSUPPORTED;
	else if ( result == CUDA_ERROR_INVALID_DEVICE || result == CUDA_ERROR_INVALID_CONTEXT )
		status = SPARK_STATUS_TARGET_MISMATCH;
	else
		status = SPARK_STATUS_IO_ERROR;
	snprintf(text,sizeof(text),"%s returned %d",call,(int)result);
	return(SparkDeviceRecordError(status,operation,owner,(int32_t)result,text));
}

static SparkStatus CudaDriverLoad(void)
{
	char text[96];
	size_t index;
	for (index=0u; index<sizeof(cuda_driver_symbols) / sizeof(cuda_driver_symbols[0]); index++)
	{
		void *function = 0;
		enum cudaDriverEntryPointQueryResult found = cudaDriverEntryPointSymbolNotFound;
		const cudaError_t error = cudaGetDriverEntryPointByVersion(cuda_driver_symbols[index].symbol,&function,CUDART_VERSION,cudaEnableDefault,&found);
		if ( error != cudaSuccess || found != cudaDriverEntryPointSuccess || function == 0 )
		{
			(void)cudaGetLastError();
			snprintf(text,sizeof(text),"the CUDA driver has no %s",cuda_driver_symbols[index].symbol);
			return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"initialize","",(int32_t)error,text));
		}
		memcpy((uint8_t *)&cuda_driver + cuda_driver_symbols[index].offset,&function,sizeof(function));
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus CudaInstanceLoad(void)
{
	SparkStatus status = CudaDriverLoad();
	cuda_instance = 0;
	if ( status == SPARK_STATUS_OK )
		status = CudaDriverStatus(cuda_driver.context_current(&cuda_instance),"initialize","","cuCtxGetCurrent");
	if ( status == SPARK_STATUS_OK && cuda_instance == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"initialize","",0,"the device has no current context after selecting it");
	return(status);
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
	if ( status == SPARK_STATUS_OK )
		status = CudaInstanceLoad();
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
	properties->max_constant_bytes = 0u;
	properties->properties = SPARK_DEVICE_PROPERTY_SHARED_ADDRESS | SPARK_DEVICE_PROPERTY_PROGRAMS |
		(device.integrated != 0 ? SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY : 0u) | (device.major >= 9 ? SPARK_DEVICE_PROPERTY_TENSOR_MAPS : 0u);
	(void)managed;
	(void)pageable;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendThreadInstance(SparkDeviceInstance *instance)
{
	CUcontext context = 0;
	SparkStatus status = CudaDriverStatus(cuda_driver.context_current(&context),"thread_instance","","cuCtxGetCurrent");
	if ( status == SPARK_STATUS_OK && context == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"thread_instance","",0,"the calling thread is bound to no device instance");
	if ( status == SPARK_STATUS_OK && context != cuda_instance )
		status = SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"thread_instance","",0,"the calling thread is bound to a context outside the device layer");
	if ( status == SPARK_STATUS_OK )
		*instance = (SparkDeviceInstance)(uintptr_t)context;
	return(status);
}

SparkStatus SparkDeviceBackendThreadBind(SparkDeviceInstance instance)
{
	if ( instance != (SparkDeviceInstance)(uintptr_t)cuda_instance )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"thread_bind","",0,"not the instance this device layer returned"));
	return(CudaDriverStatus(cuda_driver.context_set(cuda_instance),"thread_bind","","cuCtxSetCurrent"));
}

static void CudaMemoryProperties(CUmemAllocationProp *properties)
{
	memset(properties,0,sizeof(*properties));
	properties->type = CU_MEM_ALLOCATION_TYPE_PINNED;
	properties->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	properties->location.id = cuda_ordinal;
	properties->requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

static CUmemGenericAllocationHandle CudaHandle(SparkDeviceMemoryHandle memory)
{
	return((CUmemGenericAllocationHandle)(uintptr_t)memory);
}

SparkStatus SparkDeviceBackendMemoryGranularity(uint32_t granularity, uint64_t *bytes)
{
	CUmemAllocationProp properties;
	size_t value = 0u;
	SparkStatus status;
	CudaMemoryProperties(&properties);
	status = CudaDriverStatus(cuda_driver.granularity(&value,&properties,granularity == SPARK_DEVICE_GRANULARITY_RECOMMENDED ?
		CU_MEM_ALLOC_GRANULARITY_RECOMMENDED : CU_MEM_ALLOC_GRANULARITY_MINIMUM),"memory_granularity","","cuMemGetAllocationGranularity");
	if ( status == SPARK_STATUS_OK )
		*bytes = (uint64_t)value;
	return(status);
}

SparkStatus SparkDeviceBackendMemoryCreate(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner)
{
	CUmemAllocationProp properties;
	CUmemGenericAllocationHandle handle = 0;
	SparkStatus status;
	CudaMemoryProperties(&properties);
	status = CudaDriverStatus(cuda_driver.create(&handle,(size_t)bytes,&properties,0ull),"memory_create",owner,"cuMemCreate");
	if ( status == SPARK_STATUS_OK && handle == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"memory_create",owner,0,"cuMemCreate returned no handle");
	if ( status == SPARK_STATUS_OK )
		*memory = (SparkDeviceMemoryHandle)(uintptr_t)handle;
	return(status);
}

SparkStatus SparkDeviceBackendMemoryRelease(SparkDeviceMemoryHandle memory)
{
	return(CudaDriverStatus(cuda_driver.release(CudaHandle(memory)),"memory_release","","cuMemRelease"));
}

SparkStatus SparkDeviceBackendMemoryExport(SparkDeviceMemoryHandle memory, int *fd)
{
	int exported = -1;
	SparkStatus status = CudaDriverStatus(cuda_driver.export_fd(&exported,CudaHandle(memory),CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,0ull),
		"memory_export","","cuMemExportToShareableHandle");
	if ( status == SPARK_STATUS_OK && exported < 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"memory_export","",0,"cuMemExportToShareableHandle returned no file descriptor");
	if ( status == SPARK_STATUS_OK )
		*fd = exported;
	return(status);
}

SparkStatus SparkDeviceBackendMemoryImport(SparkDeviceMemoryHandle *memory, int fd, const char *owner)
{
	CUmemGenericAllocationHandle handle = 0;
	SparkStatus status = CudaDriverStatus(cuda_driver.import_fd(&handle,(void *)(intptr_t)fd,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
		"memory_import",owner,"cuMemImportFromShareableHandle");
	if ( status == SPARK_STATUS_OK && handle == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"memory_import",owner,0,"cuMemImportFromShareableHandle returned no handle");
	if ( status == SPARK_STATUS_OK )
		*memory = (SparkDeviceMemoryHandle)(uintptr_t)handle;
	return(status);
}

SparkStatus SparkDeviceBackendAddressReserve(SparkDeviceAddress *address, uint64_t bytes, const char *owner)
{
	CUdeviceptr base = 0ull;
	SparkStatus status = CudaDriverStatus(cuda_driver.reserve(&base,(size_t)bytes,0u,0ull,0ull),"address_reserve",owner,"cuMemAddressReserve");
	if ( status == SPARK_STATUS_OK && base == 0ull )
		status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"address_reserve",owner,0,"cuMemAddressReserve returned no address");
	if ( status == SPARK_STATUS_OK )
		*address = (SparkDeviceAddress)base;
	return(status);
}

SparkStatus SparkDeviceBackendAddressRelease(SparkDeviceAddress address, uint64_t bytes)
{
	return(CudaDriverStatus(cuda_driver.address_free((CUdeviceptr)address,(size_t)bytes),"address_release","","cuMemAddressFree"));
}

SparkStatus SparkDeviceBackendMemoryMap(SparkDeviceAddress address, uint64_t bytes, SparkDeviceMemoryHandle memory, uint64_t offset, uint32_t access)
{
	CUmemAccessDesc descriptor;
	SparkStatus status = CudaDriverStatus(cuda_driver.map((CUdeviceptr)address,(size_t)bytes,(size_t)offset,CudaHandle(memory),0ull),
		"memory_map","","cuMemMap");
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&descriptor,0,sizeof(descriptor));
	descriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	descriptor.location.id = cuda_ordinal;
	descriptor.flags = (access & SPARK_DEVICE_ACCESS_DEVICE_WRITE) != 0u ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ;
	status = CudaDriverStatus(cuda_driver.set_access((CUdeviceptr)address,(size_t)bytes,&descriptor,1u),"memory_map","","cuMemSetAccess");
	if ( status != SPARK_STATUS_OK )
		(void)cuda_driver.unmap((CUdeviceptr)address,(size_t)bytes);
	return(status);
}

SparkStatus SparkDeviceBackendMemoryUnmap(SparkDeviceAddress address, uint64_t bytes)
{
	return(CudaDriverStatus(cuda_driver.unmap((CUdeviceptr)address,(size_t)bytes),"memory_unmap","","cuMemUnmap"));
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
	{
		cudaStream_t zero = 0;
		status = CudaStatus(cudaStreamCreateWithFlags(&zero,cudaStreamNonBlocking),"allocate",allocation->owner);
		if ( status == SPARK_STATUS_OK )
			status = CudaStatus(cudaMemsetAsync(cuda->base,0,(size_t)allocation->bytes,zero),"allocate",allocation->owner);
		if ( status == SPARK_STATUS_OK )
			status = CudaStatus(cudaStreamSynchronize(zero),"allocate",allocation->owner);
		if ( zero != 0 )
			(void)cudaStreamDestroy(zero);
	}
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

SparkStatus SparkDeviceBackendAdopt(SparkDeviceAllocation *allocation)
{
	CudaAllocation *cuda = CudaNew(allocation,CUDA_MEMORY_ADOPTED,"adopt");
	if ( cuda == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	cuda->base = allocation->device != 0u ? (void *)(uintptr_t)allocation->device : allocation->host;
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
		SparkStatus status;
		if ( queue->recording != 0 )
			return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"copy",queue->owner,0,
				"a program cannot stage pageable memory through its queue's bounce buffer; record copies of pinned or registered memory"));
		status = CudaBounce(queue,"copy");
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
	if ( callback->retained == 0u )
		free(callback);
}

static SparkStatus CudaProgramRetain(SparkDeviceProgram *program, CudaCallback *callback);

static SparkStatus CudaEnqueueCallback(SparkDeviceQueue *queue, CudaCallback *callback, const char *operation)
{
	SparkStatus status = SPARK_STATUS_OK;
	CudaCallback *copy = (CudaCallback *)malloc(sizeof(*copy));
	if ( copy == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,queue->owner,ENOMEM,"out of host memory"));
	*copy = *callback;
	copy->retained = queue->recording != 0 ? 1u : 0u;
	if ( copy->retained != 0u )
		status = CudaProgramRetain(queue->recording,copy);
	if ( status != SPARK_STATUS_OK )
	{
		free(copy);
		return(status);
	}
	status = CudaStatus(cudaLaunchHostFunc(CudaStream(queue),CudaCallbackTrampoline,copy),operation,queue->owner);
	if ( status != SPARK_STATUS_OK && copy->retained == 0u )
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

SparkStatus SparkDeviceBackendQueueAdopt(SparkDeviceQueue *queue, void *native)
{
	CudaQueue *cuda = (CudaQueue *)calloc(1u,sizeof(*cuda));
	if ( cuda == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_adopt",queue->owner,ENOMEM,"out of host memory"));
	cuda->stream = (cudaStream_t)native;
	cuda->adopted = 1u;
	pthread_mutex_init(&cuda->gate,0);
	queue->backend = cuda;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	if ( cuda == 0 )
		return;
	(void)cudaStreamSynchronize(cuda->stream);
	if ( cuda->adopted == 0u )
		(void)cudaStreamDestroy(cuda->stream);
	else
		pthread_mutex_destroy(&cuda->gate);
	if ( cuda->bounce != 0 )
		(void)cudaFreeHost(cuda->bounce);
	free(cuda);
	queue->backend = 0;
}

static SparkStatus CudaPoll(cudaError_t (*query)(void *), void *handle, cudaError_t (*block)(void *), uint64_t timeout_ns, const char *operation, const char *owner)
{
	uint64_t deadline;
	cudaError_t error;
	if ( timeout_ns == SPARK_DEVICE_WAIT_FOREVER )
		return(CudaStatus(block(handle),operation,owner));
	deadline = SparkMonotonicNs() + timeout_ns;
	for ( ;; )
	{
		struct timespec pause = {0, 50000};
		error = query(handle);
		if ( error != cudaErrorNotReady )
			return(CudaStatus(error,operation,owner));
		if ( SparkMonotonicNs() >= deadline )
			return(SPARK_STATUS_PENDING);
		nanosleep(&pause,0);
	}
}

static cudaError_t CudaStreamQueryHandle(void *handle)
{
	return(cudaStreamQuery((cudaStream_t)handle));
}

static cudaError_t CudaAdoptedQuery(CudaQueue *cuda)
{
	cudaError_t error;
	pthread_mutex_lock(&cuda->gate);
	error = cuda->capturing != 0u ? cudaErrorNotReady : cudaStreamQuery(cuda->stream);
	pthread_mutex_unlock(&cuda->gate);
	return(error);
}

static cudaError_t CudaAdoptedQueryHandle(void *handle)
{
	return(CudaAdoptedQuery((CudaQueue *)handle));
}

static cudaError_t CudaAdoptedSettleHandle(void *handle)
{
	for ( ;; )
	{
		struct timespec pause = {0, 50000};
		const cudaError_t error = CudaAdoptedQuery((CudaQueue *)handle);
		if ( error != cudaErrorNotReady )
			return(error);
		nanosleep(&pause,0);
	}
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
	if ( cuda->adopted == 0u && __atomic_load_n(&cuda->completed,__ATOMIC_ACQUIRE) >= sequence )
		return(SPARK_STATUS_OK);
	status = cuda->adopted != 0u ?
		CudaPoll(CudaAdoptedQueryHandle,(void *)cuda,CudaAdoptedSettleHandle,timeout_ns,"queue_wait",queue->owner) :
		CudaPoll(CudaStreamQueryHandle,(void *)cuda->stream,CudaStreamSynchronizeHandle,timeout_ns,"queue_wait",queue->owner);
	if ( status == SPARK_STATUS_OK )
	{
		uint64_t known = __atomic_load_n(&cuda->completed,__ATOMIC_ACQUIRE);
		while ( known < sequence && !__atomic_compare_exchange_n(&cuda->completed,&known,sequence,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE) )
		{
		}
	}
	return(status);
}

SparkStatus SparkDeviceBackendQueueCapture(SparkDeviceQueue *queue, uint32_t capturing)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	if ( cuda->adopted == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"queue_capture",queue->owner,0,"only the owner of an adopted stream records it into a graph"));
	pthread_mutex_lock(&cuda->gate);
	cuda->capturing = capturing;
	pthread_mutex_unlock(&cuda->gate);
	return(SPARK_STATUS_OK);
}

uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue)
{
	CudaQueue *cuda = (CudaQueue *)queue->backend;
	if ( cuda->adopted != 0u )
		return(__atomic_load_n(&cuda->completed,__ATOMIC_ACQUIRE));
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
	if ( queue->recording != 0 )
		return(CudaStatus(cudaEventRecordWithFlags((cudaEvent_t)event->backend,CudaStream(queue),cudaEventRecordExternal),"event_record",queue->owner));
	return(CudaStatus(cudaEventRecord((cudaEvent_t)event->backend,CudaStream(queue)),"event_record",queue->owner));
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	const unsigned int flags = queue->recording != 0 ? cudaEventWaitExternal : 0u;
	return(CudaStatus(cudaStreamWaitEvent(CudaStream(queue),(cudaEvent_t)event->backend,flags),"queue_wait_event",queue->owner));
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

SparkStatus SparkDeviceBackendKernelShortName(SparkDeviceLibrary *library, const char *name, char short_name[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u])
{
	(void)library;
	(void)name;
	(void)short_name;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_create",0,0,"the CUDA backend takes kernel names of at most 64 bytes"));
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

#define CUDA_PROGRAM_MAX_ARGUMENTS 256u

typedef struct CudaPatchedNode
{
	cudaGraphNode_t node;
	struct cudaKernelNodeParams base;
	struct cudaKernelNodeParams params;
	void *arguments[CUDA_PROGRAM_MAX_ARGUMENTS];
	uint64_t values[CUDA_PROGRAM_MAX_ARGUMENTS];
	uint32_t count;
	uint32_t applied;
} CudaPatchedNode;

typedef struct CudaProgram
{
	cudaStream_t stream;
	cudaGraph_t graph;
	cudaGraphExec_t exec;
	CudaPatchedNode **nodes;
	uint32_t node_count;
	CudaCallback **callbacks;
	uint32_t callback_count;
	uint64_t *last;
	uint32_t ran;
} CudaProgram;

static SparkStatus CudaProgramRetain(SparkDeviceProgram *program, CudaCallback *callback)
{
	CudaProgram *cuda = (CudaProgram *)program->backend;
	CudaCallback **grown = (CudaCallback **)realloc(cuda->callbacks,((size_t)cuda->callback_count + 1u) * sizeof(*grown));
	if ( grown == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_record",program->name,ENOMEM,"out of host memory for a recorded callback"));
	cuda->callbacks = grown;
	cuda->callbacks[cuda->callback_count++] = callback;
	return(SPARK_STATUS_OK);
}

uint32_t SparkDeviceBackendProgramNative(void)
{
	return(1u);
}

SparkStatus SparkDeviceBackendProgramReplay(SparkDeviceQueue *queue)
{
	return(SparkDeviceRecordError(SPARK_STATUS_INTERNAL_ERROR,"program_run",queue->owner,0,"the CUDA backend runs programs as graphs, never by replay"));
}

SparkStatus SparkDeviceBackendProgramBegin(SparkDeviceQueue *queue, SparkDeviceProgram *program)
{
	CudaProgram *cuda = (CudaProgram *)calloc(1u,sizeof(*cuda));
	SparkStatus status;
	if ( cuda == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_record",program->name,0,"out of host memory for the program"));
	cuda->stream = CudaStream(queue);
	status = CudaStatus(cudaStreamBeginCapture(cuda->stream,cudaStreamCaptureModeThreadLocal),"program_record",program->name);
	if ( status != SPARK_STATUS_OK )
	{
		free(cuda);
		return(status);
	}
	program->backend = cuda;
	return(SPARK_STATUS_OK);
}

static uint32_t CudaArgumentCount(const void *function)
{
	uint32_t count = 0u;
	size_t offset = 0u,bytes = 0u;
	while ( count < CUDA_PROGRAM_MAX_ARGUMENTS && cudaFuncGetParamInfo(function,count,&offset,&bytes) == cudaSuccess )
		count++;
	(void)cudaGetLastError();
	return(count);
}

static SparkStatus CudaPatchedNodeOf(CudaProgram *cuda, cudaGraphNode_t node, CudaPatchedNode **out, const char *owner)
{
	CudaPatchedNode *patched,**grown;
	uint32_t index;
	SparkStatus status;
	for ( index = 0u; index < cuda->node_count; index++ )
		if ( cuda->nodes[index]->node == node )
		{
			*out = cuda->nodes[index];
			return(SPARK_STATUS_OK);
		}
	patched = (CudaPatchedNode *)calloc(1u,sizeof(*patched));
	grown = (CudaPatchedNode **)realloc(cuda->nodes,((size_t)cuda->node_count + 1u) * sizeof(*grown));
	if ( patched == 0 || grown == 0 )
	{
		free(patched);
		if ( grown != 0 )
			cuda->nodes = grown;
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_patch",owner,0,"out of host memory for the patch table"));
	}
	cuda->nodes = grown;
	patched->node = node;
	status = CudaStatus(cudaGraphKernelNodeGetParams(node,&patched->base),"program_patch",owner);
	if ( status == SPARK_STATUS_OK && (patched->base.kernelParams == 0 || patched->base.extra != 0) )
		status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"program_patch",owner,0,"a patched kernel passes its arguments as a parameter array");
	if ( status == SPARK_STATUS_OK && (patched->count = CudaArgumentCount(patched->base.func)) == CUDA_PROGRAM_MAX_ARGUMENTS )
		status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"program_patch",owner,0,"a patched kernel takes more than 255 arguments");
	if ( status != SPARK_STATUS_OK )
	{
		free(patched);
		return(status);
	}
	memcpy(patched->arguments,patched->base.kernelParams,(size_t)patched->count * sizeof(void *));
	patched->params = patched->base;
	patched->params.kernelParams = patched->arguments;
	cuda->nodes[cuda->node_count++] = patched;
	*out = patched;
	return(SPARK_STATUS_OK);
}

static SparkStatus CudaPlaceArgument(CudaPatchedNode *patched, SparkDevicePlace *place, const char *owner)
{
	size_t offset = 0u,bytes = 0u;
	uint32_t index;
	if ( place->target == SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT )
		index = place->argument;
	else
		for ( index = 0u; index < patched->count; index++ )
			if ( cudaFuncGetParamInfo(patched->base.func,index,&offset,&bytes) == cudaSuccess && offset == place->offset )
				break;
	(void)cudaGetLastError();
	if ( index >= patched->count || cudaFuncGetParamInfo(patched->base.func,index,&offset,&bytes) != cudaSuccess ||
		(bytes != 1u && bytes != 2u && bytes != 4u && bytes != 8u) ||
		(place->target == SPARK_DEVICE_PROGRAM_TARGET_CONSTANT && (offset != place->offset || bytes != place->bytes)) )
	{
		(void)cudaGetLastError();
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_patch",owner,0,
			"the patched place is not one whole 1, 2, 4 or 8-byte parameter of the captured kernel"));
	}
	place->argument = index;
	place->offset = (uint32_t)offset;
	place->bytes = (uint32_t)bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendProgramPatch(SparkDeviceQueue *queue, SparkDeviceProgram *program, SparkDevicePlace *place)
{
	CudaProgram *cuda = (CudaProgram *)program->backend;
	enum cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
	const cudaGraphNode_t *last = 0;
	size_t count = 0u;
	enum cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
	CudaPatchedNode *patched = 0;
	SparkStatus status;
	(void)queue;
	status = CudaStatus(cudaStreamGetCaptureInfo(cuda->stream,&capture,0,0,&last,0,&count),"program_patch",program->name);
	if ( status == SPARK_STATUS_OK && (capture != cudaStreamCaptureStatusActive || count != 1u ||
		cudaGraphNodeGetType(last[0],&type) != cudaSuccess || type != cudaGraphNodeTypeKernel) )
	{
		(void)cudaGetLastError();
		status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"program_patch",program->name,0,
			"a patch follows the kernel launch it changes, and the stream's last captured node is no kernel");
	}
	if ( status == SPARK_STATUS_OK )
		status = CudaPatchedNodeOf(cuda,last[0],&patched,program->name);
	if ( status == SPARK_STATUS_OK && (place->target == SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT || place->target == SPARK_DEVICE_PROGRAM_TARGET_CONSTANT) )
		status = CudaPlaceArgument(patched,place,program->name);
	else if ( status == SPARK_STATUS_OK )
	{
		place->offset = 0u;
		place->bytes = 4u;
	}
	if ( status == SPARK_STATUS_OK )
		place->backend = patched;
	return(status);
}

static uint64_t CudaProgramDigest(const CudaProgram *cuda, const SparkDeviceProgram *program, uint32_t *kernels, uint32_t *graph_nodes)
{
	uint64_t digest = SparkDeviceProgramDigest(program);
	cudaGraphNode_t *nodes = 0;
	size_t count = 0u,index;
	*kernels = 0u;
	*graph_nodes = 0u;
	if ( cudaGraphGetNodes(cuda->graph,0,&count) != cudaSuccess || count == 0u || (nodes = (cudaGraphNode_t *)calloc(count,sizeof(*nodes))) == 0 ||
		cudaGraphGetNodes(cuda->graph,nodes,&count) != cudaSuccess )
	{
		(void)cudaGetLastError();
		free(nodes);
		return(digest);
	}
	*graph_nodes = (uint32_t)count;
	for ( index = 0u; index < count; index++ )
	{
		enum cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
		struct cudaKernelNodeParams params;
		const char *name = 0;
		if ( cudaGraphNodeGetType(nodes[index],&type) != cudaSuccess )
			continue;
		digest = (digest ^ (uint64_t)type) * 1099511628211ull;
		if ( type != cudaGraphNodeTypeKernel || cudaGraphKernelNodeGetParams(nodes[index],&params) != cudaSuccess )
			continue;
		(*kernels)++;
		if ( cudaFuncGetName(&name,params.func) == cudaSuccess && name != 0 )
			for ( ; *name != '\0'; name++ )
				digest = (digest ^ (uint8_t)*name) * 1099511628211ull;
		digest = (digest ^ ((uint64_t)params.gridDim.x << 32 | params.gridDim.y)) * 1099511628211ull;
		digest = (digest ^ ((uint64_t)params.gridDim.z << 32 | params.blockDim.x)) * 1099511628211ull;
		digest = (digest ^ ((uint64_t)params.blockDim.y << 32 | params.blockDim.z)) * 1099511628211ull;
		digest = (digest ^ params.sharedMemBytes) * 1099511628211ull;
	}
	(void)cudaGetLastError();
	free(nodes);
	return(digest);
}

SparkStatus SparkDeviceBackendProgramEnd(SparkDeviceQueue *queue, SparkDeviceProgram *program)
{
	CudaProgram *cuda = (CudaProgram *)program->backend;
	SparkStatus status;
	(void)queue;
	status = CudaStatus(cudaStreamEndCapture(cuda->stream,&cuda->graph),"program_finish",program->name);
	if ( status == SPARK_STATUS_OK && program->broken == 0u )
		status = CudaStatus(cudaGraphInstantiate(&cuda->exec,cuda->graph,0ull),"program_finish",program->name);
	if ( status == SPARK_STATUS_OK && program->broken == 0u )
		status = CudaStatus(cudaGraphUpload(cuda->exec,cuda->stream),"program_finish",program->name);
	if ( status == SPARK_STATUS_OK && program->place_count != 0u && (cuda->last = (uint64_t *)calloc(program->place_count,sizeof(uint64_t))) == 0 )
		status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"program_finish",program->name,0,"out of host memory for the patch table");
	if ( status == SPARK_STATUS_OK && program->broken == 0u )
		program->digest = CudaProgramDigest(cuda,program,&program->kernels,&program->native_nodes);
	return(status);
}

static void CudaPatchNode(CudaPatchedNode *patched, const SparkDevicePlace *place, uint64_t value)
{
	if ( place->target == SPARK_DEVICE_PROGRAM_TARGET_GRID_X )
		patched->params.gridDim.x = (unsigned int)value;
	else if ( place->target == SPARK_DEVICE_PROGRAM_TARGET_GRID_Y )
		patched->params.gridDim.y = (unsigned int)value;
	else if ( place->target == SPARK_DEVICE_PROGRAM_TARGET_GRID_Z )
		patched->params.gridDim.z = (unsigned int)value;
	else
	{
		patched->values[place->argument] = 0u;
		memcpy(&patched->values[place->argument],&value,place->bytes);
		patched->arguments[place->argument] = &patched->values[place->argument];
	}
	patched->applied = 0u;
}

SparkStatus SparkDeviceBackendProgramRun(SparkDeviceProgram *program, SparkDeviceQueue *queue, const uint64_t *values)
{
	CudaProgram *cuda = (CudaProgram *)program->backend;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	for ( index = 0u; index < program->place_count; index++ )
		if ( cuda->ran == 0u || cuda->last[index] != values[index] )
		{
			CudaPatchNode((CudaPatchedNode *)program->places[index].backend,&program->places[index],values[index]);
			cuda->last[index] = values[index];
		}
	for ( index = 0u; status == SPARK_STATUS_OK && index < cuda->node_count; index++ )
		if ( cuda->nodes[index]->applied == 0u )
		{
			status = CudaStatus(cudaGraphExecKernelNodeSetParams(cuda->exec,cuda->nodes[index]->node,&cuda->nodes[index]->params),"program_run",program->name);
			cuda->nodes[index]->applied = status == SPARK_STATUS_OK ? 1u : 0u;
		}
	if ( status == SPARK_STATUS_OK )
		status = CudaStatus(cudaGraphLaunch(cuda->exec,CudaStream(queue)),"program_run",program->name);
	if ( status == SPARK_STATUS_OK )
		cuda->ran = 1u;
	return(status);
}

uint32_t SparkDeviceBackendProgramLowersCopies(void)
{
	return(0u);
}

SparkStatus SparkDeviceBackendProgramCompile(SparkDeviceQueue *queue, SparkDeviceProgram *program)
{
	(void)queue;
	(void)program;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendProgramRelease(SparkDeviceProgram *program)
{
	CudaProgram *cuda = (CudaProgram *)program->backend;
	uint32_t index;
	if ( cuda == 0 )
		return;
	if ( cuda->exec != 0 )
		(void)cudaGraphExecDestroy(cuda->exec);
	if ( cuda->graph != 0 )
		(void)cudaGraphDestroy(cuda->graph);
	for ( index = 0u; index < cuda->node_count; index++ )
		free(cuda->nodes[index]);
	for ( index = 0u; index < cuda->callback_count; index++ )
		free(cuda->callbacks[index]);
	free(cuda->nodes);
	free(cuda->callbacks);
	free(cuda->last);
	free(cuda);
	program->backend = 0;
}

uint32_t SparkDeviceBackendTensorMapOnDevice(void)
{
	return(0u);
}

_Static_assert(sizeof(CUtensorMap) == sizeof(SparkDeviceTensorMap),"a CUDA tensor map is the device layer's 128-byte map");

static CUtensorMapSwizzle CudaTensorMapSwizzle(uint32_t swizzle_bytes)
{
	if ( swizzle_bytes == 128u )
		return(CU_TENSOR_MAP_SWIZZLE_128B);
	if ( swizzle_bytes == 64u )
		return(CU_TENSOR_MAP_SWIZZLE_64B);
	if ( swizzle_bytes == 32u )
		return(CU_TENSOR_MAP_SWIZZLE_32B);
	return(CU_TENSOR_MAP_SWIZZLE_NONE);
}

SparkStatus SparkDeviceBackendTensorMapEncode(const SparkDeviceTensorMapRequest *requests, uint32_t count, SparkDeviceTensorMap *maps, const char *owner)
{
	uint32_t index,axis;
	for ( index = 0u; index < count; index++ )
	{
		const SparkDeviceTensorMapRequest *request = &requests[index];
		cuuint64_t dimension[SPARK_DEVICE_TENSOR_MAP_RANK_MAX],stride[SPARK_DEVICE_TENSOR_MAP_RANK_MAX - 1u];
		cuuint32_t box[SPARK_DEVICE_TENSOR_MAP_RANK_MAX],element_stride[SPARK_DEVICE_TENSOR_MAP_RANK_MAX];
		CUtensorMap map;
		SparkStatus status;
		for ( axis = 0u; axis < SPARK_DEVICE_TENSOR_MAP_RANK_MAX; axis++ )
		{
			dimension[axis] = request->dimension[axis];
			box[axis] = request->box[axis];
			element_stride[axis] = request->element_stride[axis];
			if ( axis + 1u < SPARK_DEVICE_TENSOR_MAP_RANK_MAX )
				stride[axis] = request->stride_bytes[axis];
		}
		memset(&map,0,sizeof(map));
		status = CudaDriverStatus(cuda_driver.tensor_map_encode(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,request->rank,(void *)(uintptr_t)request->address,dimension,
			stride,box,element_stride,CU_TENSOR_MAP_INTERLEAVE_NONE,CudaTensorMapSwizzle(request->swizzle_bytes),CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
			CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"tensor_map",owner,"cuTensorMapEncodeTiled");
		if ( status != SPARK_STATUS_OK )
			return(status);
		memcpy(&maps[index],&map,sizeof(maps[index]));
	}
	return(SPARK_STATUS_OK);
}
