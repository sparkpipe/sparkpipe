#include "sparkpipe/spark_weightd_map.h"
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

enum
{
	MAP_EMPTY,
	MAP_ACQUIRED,
	MAP_INFLIGHT,
	MAP_RECORDED,
	MAP_RETIRING
};

typedef struct SparkWeightdMapSlot
{
	uint64_t identifier;
	cudaEvent_t event;
	uint32_t state;
} SparkWeightdMapSlot;

struct SparkWeightdMap
{
	SparkWeightdClient *client;
	CUcontext context;
	CUdeviceptr base;
	uint64_t generation,span_bytes,chunk_bytes;
	uint32_t chunk_count;
	int32_t device;
	SparkStatus failure;
	CUmemGenericAllocationHandle *handles;
	uint64_t *owners;
	uint8_t *mapped;
	SparkWeightdMapSlot slots[SPARK_WEIGHTD_LEASE_COUNT_MAX];
};

static uint64_t map_now(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return(((uint64_t)now.tv_sec * UINT64_C(1000000000)) + (uint64_t)now.tv_nsec);
}

static SparkStatus map_remaining(uint64_t deadline,uint64_t *remaining)
{
	uint64_t now = map_now();
	*remaining = 0u;
	if ( now == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( now >= deadline )
		return(SPARK_STATUS_BUSY);
	*remaining = (deadline - now);
	return(SPARK_STATUS_OK);
}

static SparkStatus map_context(const SparkWeightdMap *map)
{
	CUcontext context;
	int32_t device;
	if ( map == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaGetDevice(&device) != cudaSuccess || cuCtxGetCurrent(&context) != CUDA_SUCCESS )
		return(SPARK_STATUS_IO_ERROR);
	return(device == map->device && context == map->context ? SPARK_STATUS_OK : SPARK_STATUS_TARGET_MISMATCH);
}

static SparkWeightdMapSlot *map_slot(SparkWeightdMap *map,uint64_t identifier)
{
	uint32_t i;
	if ( identifier == 0u )
		return(0);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( map->slots[i].state != MAP_EMPTY && map->slots[i].identifier == identifier )
			return(&map->slots[i]);
	return(0);
}

static void map_free_initial(SparkWeightdMap *map)
{
	uint32_t i;
	if ( map->base != 0u )
		(void)cuMemAddressFree(map->base,(size_t)map->span_bytes);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( map->slots[i].event != 0 )
			(void)cudaEventDestroy(map->slots[i].event);
	free(map->handles);
	free(map->owners);
	free(map->mapped);
	free(map);
}

static SparkStatus map_initialize_cuda(SparkWeightdMap *map)
{
	CUmemAllocationProp prop;
	size_t granularity;
	uint32_t i;
	if ( cudaGetDevice(&map->device) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( cudaEventCreateWithFlags(&map->slots[i].event,cudaEventDisableTiming) != cudaSuccess )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cuCtxGetCurrent(&map->context) != CUDA_SUCCESS || map->context == 0 )
		return(SPARK_STATUS_IO_ERROR);
	memset(&prop,0,sizeof(prop));
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = map->device;
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	if ( cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || granularity == 0u || (map->chunk_bytes % granularity) != 0u )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(cuMemAddressReserve(&map->base,(size_t)map->span_bytes,0u,0u,0u) == CUDA_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
}

SparkStatus SparkWeightdMapCreate(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,SparkWeightdMap **out)
{
	SparkWeightdMap *map;
	SparkStatus status;
	if ( out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( client == 0 || attached == 0 || attached->status != SPARK_STATUS_OK || attached->arena_generation == 0u || attached->arena_bytes == 0u || attached->chunk_bytes == 0u || attached->chunk_count == 0u || attached->chunk_count > SPARK_WEIGHTD_MAP_CHUNK_COUNT_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( attached->chunk_bytes > (SIZE_MAX / attached->chunk_count) || (((attached->arena_bytes - 1u) / attached->chunk_bytes) + 1u) != attached->chunk_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	map = calloc(1u,sizeof(*map));
	if ( map == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	map->client = client;
	map->generation = attached->arena_generation;
	map->chunk_bytes = attached->chunk_bytes;
	map->chunk_count = attached->chunk_count;
	map->span_bytes = (map->chunk_bytes * map->chunk_count);
	map->handles = calloc(map->chunk_count,sizeof(*map->handles));
	map->owners = calloc(map->chunk_count,sizeof(*map->owners));
	map->mapped = calloc(map->chunk_count,sizeof(*map->mapped));
	status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( map->handles != 0 && map->owners != 0 && map->mapped != 0 )
		status = map_initialize_cuda(map);
	if ( status != SPARK_STATUS_OK )
	{
		map_free_initial(map);
		return(status);
	}
	*out = map;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapDestroy(SparkWeightdMap *map)
{
	uint32_t i;
	SparkStatus status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( map->slots[i].state != MAP_EMPTY )
			return(SPARK_STATUS_BUSY);
	map->failure = SPARK_STATUS_IO_ERROR;
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( map->slots[i].event != 0 )
		{
			if ( cudaEventDestroy(map->slots[i].event) != cudaSuccess )
				return(SPARK_STATUS_IO_ERROR);
			map->slots[i].event = 0;
		}
	if ( map->base != 0u && cuMemAddressFree(map->base,(size_t)map->span_bytes) != CUDA_SUCCESS )
		return(SPARK_STATUS_IO_ERROR);
	map->base = 0u;
	map_free_initial(map);
	return(SPARK_STATUS_OK);
}

static SparkStatus map_drop_slot(SparkWeightdMap *map,uint32_t slot)
{
	uint64_t bit = (UINT64_C(1) << slot);
	uint32_t i;
	for (i=0u; i<map->chunk_count; i++)
	{
		if ( (map->owners[i] & bit) == 0u )
			continue;
		if ( map->owners[i] != bit )
		{
			map->owners[i] &= ~bit;
			continue;
		}
		if ( map->mapped[i] != 0u )
		{
			if ( cuMemUnmap(map->base + (i * map->chunk_bytes),(size_t)map->chunk_bytes) != CUDA_SUCCESS )
				return(SPARK_STATUS_IO_ERROR);
			map->mapped[i] = 0u;
		}
		if ( map->handles[i] != 0 )
		{
			if ( cuMemRelease(map->handles[i]) != CUDA_SUCCESS )
				return(SPARK_STATUS_IO_ERROR);
			map->handles[i] = 0;
		}
		map->owners[i] = 0u;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout)
{
	SparkWeightdMapSlot *slot;
	SparkWeightdWorkingSetResult result;
	cudaError_t ready;
	SparkStatus status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot = map_slot(map,identifier);
	if ( slot == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( slot->state == MAP_INFLIGHT )
		return(SPARK_STATUS_BUSY);
	if ( slot->state == MAP_RECORDED )
	{
		ready = cudaEventQuery(slot->event);
		if ( ready != cudaSuccess )
			return(ready == cudaErrorNotReady ? SPARK_STATUS_BUSY : SPARK_STATUS_IO_ERROR);
	}
	slot->state = MAP_RETIRING;
	status = map_drop_slot(map,(uint32_t)(slot - map->slots));
	if ( status != SPARK_STATUS_OK )
	{
		map->failure = status;
		return(status);
	}
	status = SparkWeightdClientRelease(map->client,map->generation,identifier,&result,timeout);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND )
		return(status);
	slot->identifier = 0u;
	slot->state = MAP_EMPTY;
	return(SPARK_STATUS_OK);
}

static SparkStatus map_import_chunk(SparkWeightdMap *map,uint32_t slot,uint32_t chunk,int32_t fd)
{
	CUmemAccessDesc access;
	uint64_t bit = (UINT64_C(1) << slot);
	if ( map->owners[chunk] != 0u )
	{
		if ( map->mapped[chunk] == 0u || map->handles[chunk] == 0 )
			return(SPARK_STATUS_IO_ERROR);
		map->owners[chunk] |= bit;
		return(SPARK_STATUS_OK);
	}
	map->owners[chunk] = bit;
	if ( cuMemImportFromShareableHandle(&map->handles[chunk],(void *)(intptr_t)fd,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS )
		return(SPARK_STATUS_IO_ERROR);
	if ( cuMemMap(map->base + (chunk * map->chunk_bytes),(size_t)map->chunk_bytes,0u,map->handles[chunk],0u) != CUDA_SUCCESS )
		return(SPARK_STATUS_IO_ERROR);
	map->mapped[chunk] = 1u;
	memset(&access,0,sizeof(access));
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = map->device;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
	return(cuMemSetAccess(map->base + (chunk * map->chunk_bytes),(size_t)map->chunk_bytes,&access,1u) == CUDA_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus map_import_batch(SparkWeightdMap *map,uint32_t slot,SparkWeightdExportBatch *batch,uint32_t *last,uint32_t offset,uint32_t total,uint64_t deadline)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t i,chunk;
	uint64_t remaining;
	if ( batch->chunk_bytes != map->chunk_bytes || batch->chunk_count != map->chunk_count || (offset != 0u && batch->lease_chunk_count != total) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	for (i=0u; i<batch->batch_count; i++)
	{
		chunk = batch->chunk_indices[i];
		if ( (offset != 0u || i != 0u) && chunk <= *last )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = map_remaining(deadline,&remaining);
		if ( status == SPARK_STATUS_OK )
			status = map_import_chunk(map,slot,chunk,batch->fds[i]);
		(void)close(batch->fds[i]);
		*last = chunk;
	}
	return(status);
}

static SparkStatus map_import_lease(SparkWeightdMap *map,uint32_t slot,uint64_t deadline)
{
	SparkWeightdExportBatch batch;
	SparkStatus status;
	uint32_t offset = 0u,total = 0u,last = 0u;
	uint64_t timeout;
	do
	{
		status = map_remaining(deadline,&timeout);
		if ( status != SPARK_STATUS_OK )
			return(status);
		status = SparkWeightdClientExportLeaseBatch(map->client,map->generation,map->slots[slot].identifier,offset,&batch,timeout);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( batch.status != SPARK_STATUS_OK )
			return(batch.status);
		status = map_import_batch(map,slot,&batch,&last,offset,total,deadline);
		if ( status != SPARK_STATUS_OK )
			return(status);
		total = batch.lease_chunk_count;
		offset += batch.batch_count;
	} while ( offset < total );
	return(map_remaining(deadline,&timeout));
}

SparkStatus SparkWeightdMapAcquire(SparkWeightdMap *map,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier,uint64_t timeout)
{
	SparkWeightdWorkingSetResult result;
	SparkStatus status;
	uint32_t slot;
	uint64_t now,deadline,remaining;
	if ( identifier == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*identifier = 0u;
	status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( map->failure != SPARK_STATUS_OK )
		return(map->failure);
	for (slot=0u; slot<SPARK_WEIGHTD_LEASE_COUNT_MAX; slot++)
		if ( map->slots[slot].state == MAP_EMPTY )
			break;
	if ( slot == SPARK_WEIGHTD_LEASE_COUNT_MAX )
		return(SPARK_STATUS_BUSY);
	now = map_now();
	if ( now == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( timeout == 0u )
		timeout = SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS;
	if ( timeout > (UINT64_MAX - now) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	deadline = (now + timeout);
	status = SparkWeightdClientAcquire(map->client,map->generation,keys,count,&result,timeout);
	if ( status != SPARK_STATUS_OK )
		return(status);
	map->slots[slot].identifier = result.lease_identifier;
	map->slots[slot].state = MAP_ACQUIRED;
	*identifier = result.lease_identifier;
	status = map_import_lease(map,slot,deadline);
	if ( status != SPARK_STATUS_OK )
	{
		map->slots[slot].state = MAP_RETIRING;
		if ( map_remaining(deadline,&remaining) == SPARK_STATUS_OK && SparkWeightdMapRelease(map,*identifier,remaining) == SPARK_STATUS_OK )
			*identifier = 0u;
	}
	return(status);
}

SparkStatus SparkWeightdMapBeginUse(SparkWeightdMap *map,uint64_t identifier,void **address)
{
	SparkWeightdMapSlot *slot;
	SparkStatus status;
	if ( address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*address = 0;
	status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( map->failure != SPARK_STATUS_OK )
		return(map->failure);
	slot = map_slot(map,identifier);
	if ( slot == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( slot->state != MAP_ACQUIRED )
		return(SPARK_STATUS_BUSY);
	slot->state = MAP_INFLIGHT;
	*address = (void *)(uintptr_t)map->base;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t identifier,cudaStream_t stream)
{
	SparkWeightdMapSlot *slot;
	SparkStatus status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot = map_slot(map,identifier);
	if ( slot == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( slot->state != MAP_INFLIGHT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaEventRecord(slot->event,stream) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	slot->state = MAP_RECORDED;
	return(SPARK_STATUS_OK);
}
