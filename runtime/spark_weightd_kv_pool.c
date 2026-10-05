#include "sparkpipe/spark_weightd_kv_pool.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_error_site.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

static uint64_t SparkWeightdKvPoolNowNs(void)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

SparkStatus SparkWeightdKvPoolGranularity(uint64_t *bytes)
{
	CUmemAllocationProp prop;
	size_t granularity = 0u;
	int device = 0;
	if ( bytes == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*bytes = 0u;
	if ( cudaGetDevice(&device) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	memset(&prop,0,sizeof(prop));
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = device;
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	if ( cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || granularity == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	*bytes = (uint64_t)granularity;
	return(SPARK_STATUS_OK);
}

static void SparkWeightdKvPoolUnmapFrom(SparkWeightdKvPoolMapping *mapping,uint32_t first)
{
	CUdeviceptr base = (CUdeviceptr)(uintptr_t)mapping->device_base;
	while ( mapping->mapped_count > first )
	{
		mapping->mapped_count--;
		(void)cuMemUnmap(base + (CUdeviceptr)mapping->chunk_offsets[mapping->mapped_count],(size_t)mapping->chunk_bytes);
		(void)cuMemRelease((CUmemGenericAllocationHandle)mapping->chunk_handles[mapping->mapped_count]);
		mapping->chunk_handles[mapping->mapped_count] = 0;
	}
}

void SparkWeightdKvPoolUnmap(SparkWeightdKvPoolMapping *mapping)
{
	if ( mapping == 0 )
		return;
	if ( mapping->chunk_handles != 0 && mapping->chunk_offsets != 0 )
		SparkWeightdKvPoolUnmapFrom(mapping,0u);
	if ( mapping->device_base != 0 )
		(void)cuMemAddressFree((CUdeviceptr)(uintptr_t)mapping->device_base,(size_t)mapping->device_bytes);
	if ( mapping->metadata != 0 )
		(void)munmap(mapping->metadata,(size_t)mapping->metadata_bytes);
	if ( mapping->client != 0 )
		SparkWeightdClientClose(mapping->client);
	free(mapping->chunk_offsets);
	free(mapping->chunk_handles);
	memset(mapping,0,sizeof(*mapping));
}

static SparkStatus SparkWeightdKvPoolMapRange(SparkWeightdKvPoolMapping *mapping,uint32_t end,uint64_t timeout_nanoseconds)
{
	int fds[SPARK_WEIGHTD_KV_POOL_EXPORT_MAX];
	CUmemAccessDesc access;
	CUdeviceptr base = (CUdeviceptr)(uintptr_t)mapping->device_base,at;
	uint32_t batch,index;
	SparkStatus status = SPARK_STATUS_OK;
	memset(&access,0,sizeof(access));
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = mapping->device;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	while ( status == SPARK_STATUS_OK && mapping->mapped_count < end )
	{
		batch = end - mapping->mapped_count < SPARK_WEIGHTD_KV_POOL_EXPORT_MAX ? end - mapping->mapped_count : SPARK_WEIGHTD_KV_POOL_EXPORT_MAX;
		status = SparkWeightdClientKvPoolExport(mapping->client,mapping->pool_generation,mapping->mapped_count,batch,fds,timeout_nanoseconds);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		for (index=0u; index<batch; index++)
		{
			CUmemGenericAllocationHandle handle = 0;
			if ( status == SPARK_STATUS_OK && cuMemImportFromShareableHandle(&handle,(void *)(uintptr_t)fds[index],CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS )
				status = SPARK_STATUS_IO_ERROR;
			(void)close(fds[index]);
			if ( status != SPARK_STATUS_OK )
				continue;
			at = base + (CUdeviceptr)mapping->chunk_offsets[mapping->mapped_count];
			if ( cuMemMap(at,(size_t)mapping->chunk_bytes,0u,handle,0ull) != CUDA_SUCCESS )
			{
				(void)cuMemRelease(handle);
				status = SPARK_STATUS_IO_ERROR;
				continue;
			}
			mapping->chunk_handles[mapping->mapped_count++] = (void *)handle;
			if ( cuMemSetAccess(at,(size_t)mapping->chunk_bytes,&access,1u) != CUDA_SUCCESS )
				status = SPARK_STATUS_IO_ERROR;
		}
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkWeightdKvPoolAwaitMinimum(const SparkWeightdKvPoolRequest *request,SparkWeightdKvPoolMapping *mapping,uint64_t timeout_nanoseconds)
{
	SparkWeightdKvPoolState state;
	struct timespec pause;
	uint64_t minimum = (request->minimum_bytes + request->chunk_bytes - 1u) / request->chunk_bytes,deadline = SparkWeightdKvPoolNowNs() + timeout_nanoseconds;
	SparkStatus status = SPARK_STATUS_OK;
	if ( minimum == 0u )
		minimum = 1u;
	memset(&state,0,sizeof(state));
	if ( mapping->chunk_count < minimum )
		fprintf(stderr,"%s kv pool waits for its minimum: %u of %llu chunks granted; kv_reserve=%llu kv_committed=%llu; other pools are asked to shrink\n",request->label,
			mapping->chunk_count,(unsigned long long)minimum,(unsigned long long)mapping->kv_reserve_bytes,(unsigned long long)mapping->kv_committed_bytes);
	while ( mapping->chunk_count < minimum )
	{
		if ( SparkWeightdKvPoolNowNs() >= deadline )
		{
			fprintf(stderr,"%s kv pool refused: weightd granted %u of the minimum %llu chunks of %llu bytes within %llu ms; kv_reserve=%llu kv_committed=%llu\n",request->label,
				mapping->chunk_count,(unsigned long long)minimum,(unsigned long long)request->chunk_bytes,(unsigned long long)(timeout_nanoseconds / 1000000ull),
				(unsigned long long)mapping->kv_reserve_bytes,(unsigned long long)state.kv_committed_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		pause.tv_sec = 0;
		pause.tv_nsec = (long)SPARK_WEIGHTD_KV_POOL_MINIMUM_POLL_NS;
		(void)nanosleep(&pause,0);
		status = SparkWeightdClientKvPoolResize(mapping->client,mapping->pool_generation,(uint32_t)minimum,&state,timeout_nanoseconds);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		mapping->chunk_count = state.chunk_count;
		mapping->kv_committed_bytes = state.kv_committed_bytes;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkWeightdKvPoolPlace(SparkWeightdKvPoolMapping *mapping,const SparkWeightdKvPoolGrant *grant,const uint64_t *chunk_offsets)
{
	CUdeviceptr base = 0;
	uint32_t index;
	mapping->chunk_offsets = (uint64_t *)malloc((size_t)grant->chunk_capacity * sizeof(uint64_t));
	mapping->chunk_handles = (void **)calloc(grant->chunk_capacity,sizeof(void *));
	if ( mapping->chunk_offsets == 0 || mapping->chunk_handles == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<grant->chunk_capacity; index++)
	{
		if ( chunk_offsets[index] % grant->chunk_bytes != 0u || chunk_offsets[index] > grant->device_bytes - grant->chunk_bytes )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		mapping->chunk_offsets[index] = chunk_offsets[index];
	}
	if ( cudaGetDevice(&mapping->device) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( cuMemAddressReserve(&base,(size_t)grant->device_bytes,0u,0ull,0ull) != CUDA_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	mapping->device_base = (void *)(uintptr_t)base;
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

SparkStatus SparkWeightdKvPoolMap(const SparkWeightdKvPoolRequest *request,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds,SparkWeightdKvPoolMapping *mapping)
{
	SparkWeightdKvPoolGrant grant;
	const char *socket;
	SparkStatus status;
	if ( request == 0 || chunk_offsets == 0 || mapping == 0 || request->label == 0 || request->chunk_bytes == 0u )
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
		fprintf(stderr,"%s kv pool refused: weightd attach status=%s bytes=%llu minimum=%llu kv_reserve=%llu kv_committed=%llu\n",request->label,SparkStatusToString(status),
			(unsigned long long)request->device_bytes,(unsigned long long)request->minimum_bytes,(unsigned long long)grant.kv_reserve_bytes,(unsigned long long)grant.kv_committed_bytes);
		SparkWeightdKvPoolUnmap(mapping);
		SPARK_RETURN(status);
	}
	mapping->device_bytes = grant.device_bytes;
	mapping->chunk_bytes = grant.chunk_bytes;
	mapping->chunk_capacity = grant.chunk_capacity;
	mapping->chunk_count = grant.chunk_count;
	mapping->pool_generation = grant.pool_generation;
	mapping->reattached = grant.reattached;
	mapping->write_budget_bytes_per_day = grant.write_budget_bytes_per_day;
	status = SparkWeightdKvPoolPlace(mapping,&grant,chunk_offsets);
	if ( grant.metadata_fd >= 0 )
		(void)close(grant.metadata_fd);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdKvPoolAwaitMinimum(request,mapping,timeout_nanoseconds);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdKvPoolMapRange(mapping,mapping->chunk_count,timeout_nanoseconds);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv pool refused: mapping the weightd pool failed status=%s mapped=%u of %u chunks\n",request->label,SparkStatusToString(status),mapping->mapped_count,mapping->chunk_count);
		SparkWeightdKvPoolUnmap(mapping);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdKvPoolGrow(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds)
{
	SparkStatus status;
	if ( mapping == 0 || state == 0 || mapping->client == 0 || target_chunks > mapping->chunk_capacity || target_chunks < mapping->mapped_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(state,0,sizeof(*state));
	status = SparkWeightdClientKvPoolResize(mapping->client,mapping->pool_generation,target_chunks,state,timeout_nanoseconds);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->chunk_count < mapping->mapped_count || state->chunk_count > mapping->chunk_capacity )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	mapping->chunk_count = state->chunk_count;
	mapping->kv_committed_bytes = state->kv_committed_bytes;
	return(SparkWeightdKvPoolMapRange(mapping,mapping->chunk_count,timeout_nanoseconds));
}

SparkStatus SparkWeightdKvPoolShrink(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds)
{
	SparkStatus status;
	if ( mapping == 0 || state == 0 || mapping->client == 0 || target_chunks >= mapping->mapped_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(state,0,sizeof(*state));
	SparkWeightdKvPoolUnmapFrom(mapping,target_chunks);
	status = SparkWeightdClientKvPoolResize(mapping->client,mapping->pool_generation,target_chunks,state,timeout_nanoseconds);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->chunk_count != target_chunks )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	mapping->chunk_count = state->chunk_count;
	mapping->kv_committed_bytes = state->kv_committed_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdKvPoolStatus(SparkWeightdKvPoolMapping *mapping,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds)
{
	SparkStatus status;
	if ( mapping == 0 || state == 0 || mapping->client == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(state,0,sizeof(*state));
	status = SparkWeightdClientKvPoolStatus(mapping->client,mapping->pool_generation,state,timeout_nanoseconds);
	if ( status == SPARK_STATUS_OK )
		mapping->kv_committed_bytes = state->kv_committed_bytes;
	SPARK_RETURN(status);
}
