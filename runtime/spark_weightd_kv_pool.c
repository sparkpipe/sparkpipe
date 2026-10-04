#include "sparkpipe/spark_weightd_kv_pool.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_error_site.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

void SparkWeightdKvPoolUnmap(SparkWeightdKvPoolMapping *mapping)
{
	CUdeviceptr base;
	uint64_t span;
	uint32_t index;
	if ( mapping == 0 )
		return;
	base = (CUdeviceptr)(uintptr_t)mapping->device_base;
	span = mapping->chunk_bytes * mapping->chunk_count;
	if ( base != 0 && mapping->mapped_count != 0u )
		(void)cuMemUnmap(base,(size_t)(mapping->chunk_bytes * mapping->mapped_count));
	for (index=0u; index<mapping->chunk_count; index++)
		if ( mapping->chunk_handles[index] != 0 )
			(void)cuMemRelease((CUmemGenericAllocationHandle)mapping->chunk_handles[index]);
	if ( base != 0 )
		(void)cuMemAddressFree(base,(size_t)span);
	if ( mapping->metadata != 0 )
		(void)munmap(mapping->metadata,(size_t)mapping->metadata_bytes);
	if ( mapping->client != 0 )
		SparkWeightdClientClose(mapping->client);
	memset(mapping,0,sizeof(*mapping));
}

static SparkStatus SparkWeightdKvPoolImport(SparkWeightdKvPoolMapping *mapping,SparkWeightdKvPoolGrant *grant)
{
	CUmemAccessDesc access;
	CUdeviceptr base = 0;
	int device = 0;
	uint32_t index;
	for (index=0u; index<grant->chunk_count; index++)
	{
		CUmemGenericAllocationHandle handle = 0;
		if ( cuMemImportFromShareableHandle(&handle,(void *)(uintptr_t)grant->chunk_fds[index],CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		mapping->chunk_handles[index] = (void *)handle;
	}
	if ( cuMemAddressReserve(&base,(size_t)grant->device_bytes,0u,0ull,0ull) != CUDA_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	mapping->device_base = (void *)(uintptr_t)base;
	for (index=0u; index<grant->chunk_count; index++)
	{
		if ( cuMemMap(base + (CUdeviceptr)index * grant->chunk_bytes,(size_t)grant->chunk_bytes,0u,(CUmemGenericAllocationHandle)mapping->chunk_handles[index],0ull) != CUDA_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		mapping->mapped_count++;
	}
	if ( cudaGetDevice(&device) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	memset(&access,0,sizeof(access));
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = device;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	if ( cuMemSetAccess(base,(size_t)grant->device_bytes,&access,1u) != CUDA_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( grant->metadata_fd >= 0 )
	{
		void *metadata = mmap(0,(size_t)grant->metadata_bytes,PROT_READ | PROT_WRITE,MAP_SHARED,grant->metadata_fd,0);
		if ( metadata == MAP_FAILED )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		mapping->metadata = (uint8_t *)metadata;
		mapping->metadata_bytes = grant->metadata_bytes;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdKvPoolMap(const SparkWeightdKvPoolRequest *request,uint64_t timeout_nanoseconds,SparkWeightdKvPoolMapping *mapping)
{
	SparkWeightdKvPoolGrant grant;
	const char *socket;
	SparkStatus status;
	if ( request == 0 || mapping == 0 || request->label == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(mapping,0,sizeof(*mapping));
	memset(&grant,0,sizeof(grant));
	grant.metadata_fd = -1;
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv pool refused: weightd owns node KV memory and %s does not name its socket\n",request->label,SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
		SPARK_RETURN(status);
	}
	socket = getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
	status = SparkWeightdClientConnect(socket,&mapping->client,0);
	if ( status != SPARK_STATUS_OK )
	{
		mapping->client = 0;
		fprintf(stderr,"%s kv pool refused: cannot connect to weightd at %s status=%s\n",request->label,socket,SparkStatusToString(status));
		SPARK_RETURN(status);
	}
	status = SparkWeightdClientKvPoolAttach(mapping->client,request,&grant,timeout_nanoseconds);
	mapping->kv_reserve_bytes = grant.kv_reserve_bytes;
	mapping->kv_committed_bytes = grant.kv_committed_bytes;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv pool refused: weightd attach status=%s bytes=%llu kv_reserve=%llu kv_committed=%llu\n",request->label,SparkStatusToString(status),
			(unsigned long long)request->device_bytes,(unsigned long long)grant.kv_reserve_bytes,(unsigned long long)grant.kv_committed_bytes);
		SparkWeightdKvPoolUnmap(mapping);
		SPARK_RETURN(status);
	}
	mapping->device_bytes = grant.device_bytes;
	mapping->chunk_bytes = grant.chunk_bytes;
	mapping->chunk_count = grant.chunk_count;
	mapping->pool_generation = grant.pool_generation;
	mapping->reattached = grant.reattached;
	mapping->write_budget_bytes_per_day = grant.write_budget_bytes_per_day;
	status = SparkWeightdKvPoolImport(mapping,&grant);
	SparkWeightdKvPoolGrantClose(&grant);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv pool refused: mapping the weightd pool failed status=%s\n",request->label,SparkStatusToString(status));
		SparkWeightdKvPoolUnmap(mapping);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}
