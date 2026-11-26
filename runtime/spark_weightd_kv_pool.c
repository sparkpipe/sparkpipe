#include "sparkpipe/spark_weightd_kv_pool.h"
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_error_site.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_weightd_device.h"
#include "sparkpipe/spark_gpu.h"

SparkStatus SparkWeightdKvPoolGranularity(uint64_t *bytes)
{
	uint64_t granularity = 0u;
	if ( bytes == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*bytes = 0u;
	if ( SparkWeightdDeviceOpen() != SPARK_STATUS_OK ||
		SparkDeviceMemoryGranularity(SPARK_DEVICE_GRANULARITY_MINIMUM,&granularity) != SPARK_STATUS_OK || granularity == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	*bytes = granularity;
	return(SPARK_STATUS_OK);
}

static void SparkWeightdKvPoolUnmapFrom(SparkWeightdKvPoolMapping *mapping,uint32_t first)
{
	SparkDeviceAddress base = (SparkDeviceAddress)(uintptr_t)mapping->device_base;
	while ( mapping->mapped_count > first )
	{
		mapping->mapped_count--;
		(void)SparkDeviceMemoryUnmap(base + mapping->chunk_offsets[mapping->mapped_count],mapping->chunk_bytes);
		(void)SparkDeviceMemoryRelease(SparkWeightdMemoryOf(mapping->chunk_handles[mapping->mapped_count]));
		mapping->chunk_handles[mapping->mapped_count] = 0;
	}
}

void SparkWeightdKvPoolUnmap(SparkWeightdKvPoolMapping *mapping)
{
	if ( mapping == 0 )
		return;
	if ( mapping->chunk_handles != 0 && mapping->chunk_offsets != 0 )
		SparkWeightdKvPoolUnmapFrom(mapping,0u);
	if ( mapping->device_base != 0 && mapping->external_reservation == 0u )
		(void)SparkDeviceAddressRelease((SparkDeviceAddress)(uintptr_t)mapping->device_base,mapping->device_bytes);
	if ( mapping->metadata != 0 )
		(void)munmap(mapping->metadata,(size_t)mapping->metadata_bytes);
	if ( mapping->client != 0 )
		SparkWeightdClientClose(mapping->client);
	free(mapping->chunk_offsets);
	free(mapping->chunk_handles);
	memset(mapping,0,sizeof(*mapping));
}

static SparkStatus SparkWeightdKvMapChunks(SparkWeightdClient *client,uint64_t generation,uintptr_t device_base,const uint64_t *offsets,void **handles,uint64_t chunk_bytes,int device,uint32_t *mapped_count,uint32_t end,uint64_t timeout_nanoseconds)
{
	int fds[SPARK_WEIGHTD_KV_POOL_EXPORT_MAX];
	SparkDeviceAddress at;
	uint32_t batch,index;
	SparkStatus status = SparkWeightdDeviceOpen();
	(void)device;
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	while ( status == SPARK_STATUS_OK && *mapped_count < end )
	{
		batch = end - *mapped_count < SPARK_WEIGHTD_KV_POOL_EXPORT_MAX ? end - *mapped_count : SPARK_WEIGHTD_KV_POOL_EXPORT_MAX;
		status = SparkWeightdClientKvPoolExport(client,generation,*mapped_count,batch,fds,timeout_nanoseconds);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		for (index=0u; index<batch; index++)
		{
			SparkDeviceMemoryHandle handle = 0u;
			if ( status == SPARK_STATUS_OK && SparkDeviceMemoryImport(&handle,fds[index],"weightd kv pool") != SPARK_STATUS_OK )
				status = SPARK_STATUS_IO_ERROR;
			(void)close(fds[index]);
			if ( status != SPARK_STATUS_OK )
				continue;
			at = (SparkDeviceAddress)device_base + offsets[*mapped_count];
			if ( SparkDeviceMemoryMap(at,chunk_bytes,handle,0u,SPARK_WEIGHTD_DEVICE_READ_WRITE) != SPARK_STATUS_OK )
			{
				(void)SparkDeviceMemoryRelease(handle);
				status = SPARK_STATUS_IO_ERROR;
				continue;
			}
			handles[(*mapped_count)++] = (void *)(uintptr_t)handle;
		}
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkWeightdKvPoolMapRange(SparkWeightdKvPoolMapping *mapping,uint32_t end,uint64_t timeout_nanoseconds)
{
	return(SparkWeightdKvMapChunks(mapping->client,mapping->pool_generation,(uintptr_t)mapping->device_base,mapping->chunk_offsets,mapping->chunk_handles,mapping->chunk_bytes,
		mapping->device,&mapping->mapped_count,end,timeout_nanoseconds));
}

static SparkStatus SparkWeightdKvPoolAwaitMinimum(const SparkWeightdKvPoolRequest *request,SparkWeightdKvPoolMapping *mapping,uint64_t timeout_nanoseconds)
{
	SparkWeightdKvPoolState state;
	struct timespec pause;
	uint64_t minimum = (request->minimum_bytes + request->chunk_bytes - 1u) / request->chunk_bytes,deadline = SparkMonotonicNs() + timeout_nanoseconds;
	SparkStatus status = SPARK_STATUS_OK;
	if ( minimum == 0u )
		minimum = 1u;
	memset(&state,0,sizeof(state));
	if ( mapping->chunk_count < minimum )
		fprintf(stderr,"%s kv pool waits for its minimum: %u of %llu chunks granted; kv_reserve=%llu kv_committed=%llu; other pools are asked to shrink\n",request->label,
			mapping->chunk_count,(unsigned long long)minimum,(unsigned long long)mapping->kv_reserve_bytes,(unsigned long long)mapping->kv_committed_bytes);
	while ( mapping->chunk_count < minimum )
	{
		if ( SparkMonotonicNs() >= deadline )
		{
			fprintf(stderr,"%s kv pool refused: weightd granted %u of the minimum %llu chunks of %llu bytes within %llu ms; kv_reserve=%llu kv_committed=%llu\n",request->label,
				mapping->chunk_count,(unsigned long long)minimum,(unsigned long long)request->chunk_bytes,(unsigned long long)(timeout_nanoseconds / 1000000ull),
				(unsigned long long)mapping->kv_reserve_bytes,(unsigned long long)state.kv_committed_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		pause.tv_sec = 0;
		pause.tv_nsec = (long)(SPARK_WEIGHTD_KV_POOL_MINIMUM_POLL_NS < 5000000ull ? SPARK_WEIGHTD_KV_POOL_MINIMUM_POLL_NS : 5000000ull);
		(void)nanosleep(&pause,0);
		status = SparkWeightdClientKvPoolResize(mapping->client,mapping->pool_generation,(uint32_t)minimum,&state,timeout_nanoseconds);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		mapping->chunk_count = state.chunk_count;
		mapping->kv_committed_bytes = state.kv_committed_bytes;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkWeightdKvPoolPlace(SparkWeightdKvPoolMapping *mapping,const SparkWeightdKvPoolGrant *grant,const uint64_t *chunk_offsets,const SparkWeightdKvPoolRequest *request)
{
	SparkDeviceAddress base = 0u;
	uint64_t granularity = grant->chunk_bytes,limit = grant->device_bytes;
	uint32_t index;
	mapping->chunk_offsets = (uint64_t *)malloc((size_t)grant->chunk_capacity * sizeof(uint64_t));
	mapping->chunk_handles = (void **)calloc(grant->chunk_capacity,sizeof(void *));
	if ( mapping->chunk_offsets == 0 || mapping->chunk_handles == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( request->device_base != 0 )
	{
		limit = request->reservation_bytes;
		if ( SparkWeightdKvPoolGranularity(&granularity) != SPARK_STATUS_OK || limit < grant->chunk_bytes )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	for (index=0u; index<grant->chunk_capacity; index++)
	{
		if ( chunk_offsets[index] % granularity != 0u || chunk_offsets[index] > limit - grant->chunk_bytes )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		mapping->chunk_offsets[index] = chunk_offsets[index];
	}
	if ( SparkGpuGetDevice(&mapping->device) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( request->device_base != 0 )
	{
		mapping->device_base = request->device_base;
		mapping->external_reservation = 1u;
	}
	else
	{
		if ( SparkWeightdDeviceOpen() != SPARK_STATUS_OK || SparkDeviceAddressReserve(&base,grant->device_bytes,"weightd kv pool") != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		mapping->device_base = (void *)(uintptr_t)base;
	}
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

SparkStatus SparkWeightdKvPoolMap(const char *socket,const SparkWeightdKvPoolRequest *request,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds,SparkWeightdKvPoolMapping *mapping)
{
	SparkWeightdKvPoolGrant grant;
	SparkStatus status;
	if ( request == 0 || chunk_offsets == 0 || mapping == 0 || request->label == 0 || request->chunk_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(mapping,0,sizeof(*mapping));
	memset(&grant,0,sizeof(grant));
	grant.metadata_fd = -1;
	if ( socket == 0 || socket[0] == '\0' )
	{
		fprintf(stderr,"%s kv pool refused: weightd owns node KV memory and the binding names no weightd socket\n",request->label);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
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
	status = SparkWeightdKvPoolPlace(mapping,&grant,chunk_offsets,request);
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

void SparkWeightdKvSharedUnmap(SparkWeightdKvSharedMapping *mapping)
{
	SparkDeviceAddress base;
	if ( mapping == 0 )
		return;
	base = (SparkDeviceAddress)(uintptr_t)mapping->device_base;
	while ( mapping->chunk_handles != 0 && mapping->chunk_offsets != 0 && mapping->mapped_count != 0u )
	{
		mapping->mapped_count--;
		(void)SparkDeviceMemoryUnmap(base + mapping->chunk_offsets[mapping->mapped_count],mapping->chunk_bytes);
		(void)SparkDeviceMemoryRelease(SparkWeightdMemoryOf(mapping->chunk_handles[mapping->mapped_count]));
	}
	if ( mapping->metadata != 0 )
		(void)munmap(mapping->metadata,(size_t)mapping->metadata_bytes);
	if ( mapping->client != 0 )
		SparkWeightdClientClose(mapping->client);
	free(mapping->chunk_offsets);
	free(mapping->chunk_handles);
	memset(mapping,0,sizeof(*mapping));
}

SparkStatus SparkWeightdKvSharedAttach(const char *socket,const SparkWeightdKvSharedRequest *request,uint64_t timeout_nanoseconds,SparkWeightdKvSharedMapping *mapping)
{
	SparkWeightdKvSharedGrant grant;
	void *metadata;
	SparkStatus status;
	if ( socket == 0 || socket[0] == '\0' || request == 0 || mapping == 0 || request->label == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(mapping,0,sizeof(*mapping));
	status = SparkWeightdClientConnect(socket,&mapping->client,0);
	if ( status != SPARK_STATUS_OK )
	{
		mapping->client = 0;
		SPARK_RETURN(status);
	}
	status = SparkWeightdClientKvSharedAttach(mapping->client,request,&grant,timeout_nanoseconds);
	if ( status != SPARK_STATUS_OK )
	{
		SparkWeightdKvSharedUnmap(mapping);
		return(status);
	}
	metadata = mmap(0,(size_t)grant.metadata_bytes,PROT_READ | PROT_WRITE,MAP_SHARED,grant.metadata_fd,0);
	(void)close(grant.metadata_fd);
	if ( metadata == MAP_FAILED )
	{
		SparkWeightdKvSharedUnmap(mapping);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	mapping->metadata = (uint8_t *)metadata;
	mapping->metadata_bytes = grant.metadata_bytes;
	mapping->pool_generation = grant.pool_generation;
	mapping->chunk_bytes = grant.chunk_bytes;
	mapping->device_bytes = grant.device_bytes;
	mapping->chunk_count = grant.chunk_count;
	mapping->slot_count = grant.slot_count;
	mapping->holder = grant.holder;
	mapping->created = grant.created;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdKvSharedMapChunks(SparkWeightdKvSharedMapping *mapping,void *device_base,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds)
{
	if ( mapping == 0 || mapping->client == 0 || device_base == 0 || chunk_offsets == 0 || mapping->chunk_handles != 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	mapping->chunk_offsets = (uint64_t *)malloc((size_t)mapping->chunk_count * sizeof(uint64_t));
	mapping->chunk_handles = (void **)calloc(mapping->chunk_count,sizeof(void *));
	if ( mapping->chunk_offsets == 0 || mapping->chunk_handles == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(mapping->chunk_offsets,chunk_offsets,(size_t)mapping->chunk_count * sizeof(uint64_t));
	if ( SparkGpuGetDevice(&mapping->device) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	mapping->device_base = device_base;
	return(SparkWeightdKvMapChunks(mapping->client,mapping->pool_generation,(uintptr_t)device_base,mapping->chunk_offsets,mapping->chunk_handles,mapping->chunk_bytes,
		mapping->device,&mapping->mapped_count,mapping->chunk_count,timeout_nanoseconds));
}

SparkStatus SparkWeightdKvReserveAddress(uint64_t bytes,void **base_out)
{
	SparkDeviceAddress base = 0u;
	if ( base_out == 0 || bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*base_out = 0;
	if ( SparkWeightdDeviceOpen() != SPARK_STATUS_OK || SparkDeviceAddressReserve(&base,bytes,"weightd kv reservation") != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	*base_out = (void *)(uintptr_t)base;
	return(SPARK_STATUS_OK);
}

void SparkWeightdKvFreeAddress(void *base,uint64_t bytes)
{
	if ( base != 0 )
		(void)SparkDeviceAddressRelease((SparkDeviceAddress)(uintptr_t)base,bytes);
}
