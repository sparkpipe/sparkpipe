#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_error_site.h"
#include <pthread.h>
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
	MAP_RETIRING,
	MAP_LEASE_COUNT = 64
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
	pthread_mutex_t client_lock;
	uint32_t client_lock_initialized;
	CUcontext context;
	CUdeviceptr base;
	uint64_t generation,span_bytes,chunk_bytes;
	void *epoch_device;
	void *epoch_handle;
	uint32_t chunk_count;
	uint32_t pool_mapped;
	int32_t device;
	SparkStatus failure;
	CUmemGenericAllocationHandle *handles;
	uint64_t *owners;
	uint8_t *mapped;
	SparkWeightdMapSlot slots[MAP_LEASE_COUNT];
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
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( now >= deadline )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	*remaining = (deadline - now);
	return(SPARK_STATUS_OK);
}

static SparkStatus map_context(const SparkWeightdMap *map)
{
	CUcontext context;
	int32_t device;
	if ( map == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaGetDevice(&device) != cudaSuccess || cuCtxGetCurrent(&context) != CUDA_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(device == map->device && context == map->context ? SPARK_STATUS_OK : SPARK_STATUS_TARGET_MISMATCH);
}

static SparkWeightdMapSlot *map_slot(SparkWeightdMap *map,uint64_t identifier)
{
	uint32_t i;
	if ( identifier == 0u )
		return(0);
	for (i=0u; i<MAP_LEASE_COUNT; i++)
		if ( map->slots[i].state != MAP_EMPTY && map->slots[i].identifier == identifier )
			return(&map->slots[i]);
	return(0);
}

static SparkStatus map_free_initial(SparkWeightdMap *map)
{
	uint32_t i;
	if ( map->epoch_device != 0 )
	{
		if ( cuMemUnmap(map->base + map->span_bytes,
		    (size_t)map->chunk_bytes) != CUDA_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->epoch_device = 0;
	}
	if ( map->pool_mapped != 0u )
	{
		if ( cuMemUnmap(map->base,(size_t)map->span_bytes) != CUDA_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->pool_mapped = 0u;
		memset(map->mapped,0,map->chunk_count * sizeof(*map->mapped));
	}
	for (i=0u; map->mapped != 0 && i<map->chunk_count; i++)
		if ( map->mapped[i] != 0u )
		{
			if ( cuMemUnmap(map->base + i * map->chunk_bytes,
			    (size_t)map->chunk_bytes) != CUDA_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			map->mapped[i] = 0u;
		}
	if ( map->epoch_handle != 0 )
	{
		if ( cuMemRelease((CUmemGenericAllocationHandle)map->epoch_handle) != CUDA_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->epoch_handle = 0;
	}
	for (i=0u; map->handles != 0 && i<map->chunk_count; i++)
	{
		uint32_t j;
		CUmemGenericAllocationHandle handle = map->handles[i];
		if ( handle == 0 )
			continue;
		if ( cuMemRelease(handle) != CUDA_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		for (j=i; j<map->chunk_count; j++)
			if ( map->handles[j] == handle )
				map->handles[j] = 0;
	}
	if ( map->base != 0u )
	{
		if ( cuMemAddressFree(map->base,
		    (size_t)(map->span_bytes + map->chunk_bytes)) != CUDA_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->base = 0u;
	}
	for (i=0u; i<MAP_LEASE_COUNT; i++)
		if ( map->slots[i].event != 0 )
		{
			if ( cudaEventDestroy(map->slots[i].event) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			map->slots[i].event = 0;
		}
	free(map->handles);
	free(map->owners);
	free(map->mapped);
	if ( map->client_lock_initialized != 0u )
		(void)pthread_mutex_destroy(&map->client_lock);
	free(map);
	return(SPARK_STATUS_OK);
}

static SparkStatus map_initialize_cuda(SparkWeightdMap *map)
{
	CUmemAllocationProp prop;
	size_t granularity;
	uint32_t i;
	if ( cudaGetDevice(&map->device) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( cuCtxGetCurrent(&map->context) != CUDA_SUCCESS || map->context == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	for (i=0u; i<MAP_LEASE_COUNT; i++)
		if ( cudaEventCreateWithFlags(&map->slots[i].event,cudaEventDisableTiming) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&prop,0,sizeof(prop));
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = map->device;
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	if ( cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || granularity == 0u || (map->chunk_bytes % granularity) != 0u )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	map->epoch_device = 0;
	map->epoch_handle = 0;
	return(cuMemAddressReserve(&map->base,
	    (size_t)(map->span_bytes + map->chunk_bytes),0u,0u,0u) ==
	    CUDA_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
}

SparkStatus SparkWeightdMapCreate(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,int epoch_fd,int pool_fd,SparkWeightdMap **out)
{
	SparkWeightdMap *map;
	SparkStatus status;
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( client == 0 || attached == 0 || attached->status != SPARK_STATUS_OK || attached->arena_generation == 0u || attached->arena_bytes == 0u || attached->chunk_bytes == 0u || attached->chunk_count == 0u || attached->chunk_count > SPARK_WEIGHTD_MAP_CHUNK_COUNT_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( attached->chunk_bytes > (SIZE_MAX / attached->chunk_count) || (((attached->arena_bytes - 1u) / attached->chunk_bytes) + 1u) != attached->chunk_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	map = calloc(1u,sizeof(*map));
	if ( map == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( pthread_mutex_init(&map->client_lock,0) != 0 )
	{
		free(map);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	map->client_lock_initialized = 1u;
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
	if ( status == SPARK_STATUS_OK && epoch_fd >= 0 )
	{
		CUmemAccessDesc access;
		if ( cuMemImportFromShareableHandle(
		         (CUmemGenericAllocationHandle *)&map->epoch_handle,
		         (void *)(intptr_t)epoch_fd,
			         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) !=
			     CUDA_SUCCESS ||
		     cuMemMap(map->base + map->span_bytes,
		         (size_t)map->chunk_bytes,0u,
		         (CUmemGenericAllocationHandle)map->epoch_handle,
		         0u) != CUDA_SUCCESS )
			status = SPARK_STATUS_IO_ERROR;
		else
		{
			map->epoch_device = (void *)(map->base + map->span_bytes);
			memset(&access,0,sizeof(access));
			access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
			access.location.id = map->device;
			access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
			if ( cuMemSetAccess(map->base + map->span_bytes,
			        (size_t)map->chunk_bytes,&access,1u) != CUDA_SUCCESS )
				status = SPARK_STATUS_IO_ERROR;
		}
	}
	if ( status == SPARK_STATUS_OK && pool_fd >= 0 )
	{
		CUmemAccessDesc access;
		if ( cuMemImportFromShareableHandle(&map->handles[0],
		         (void *)(intptr_t)pool_fd,
			         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) !=
			     CUDA_SUCCESS ||
		     cuMemMap(map->base,(size_t)map->span_bytes,0u,map->handles[0],0ull) !=
		             CUDA_SUCCESS )
		{
			fprintf(stderr,"WD-MAP-POOL-IMPORT-FAIL chunk_count=%u span=%llu\n",
			    map->chunk_count,(unsigned long long)map->span_bytes);
			status = SPARK_STATUS_IO_ERROR;
		}
		else
		{
			uint32_t i;
			map->pool_mapped = 1u;
			for ( i = 0u; i < map->chunk_count; i++ )
			{
				map->handles[i] = map->handles[0];
				map->mapped[i] = 1u;
			}
			memset(&access,0,sizeof(access));
			access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
			access.location.id = map->device;
			access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
			if ( cuMemSetAccess(map->base,(size_t)map->span_bytes,
			        &access,1u) != CUDA_SUCCESS )
				status = SPARK_STATUS_IO_ERROR;
			else
			{
				fprintf(stderr,
				    "WD-MAP-POOL-BULK chunks=%u span=%llu — all chunks mapped in ONE import\n",
				    map->chunk_count,(unsigned long long)map->span_bytes);
			}
		}
	}
	if ( epoch_fd >= 0 ) (void)close(epoch_fd);
	if ( pool_fd >= 0 ) (void)close(pool_fd);
	if ( status != SPARK_STATUS_OK )
	{
		map->failure = status;
		if ( map_free_initial(map) != SPARK_STATUS_OK )
			*out = map;
		SPARK_RETURN(status);
	}
	*out = map;
	return(SPARK_STATUS_OK);
}

const void *SparkWeightdMapEpochDevice(const SparkWeightdMap *map)
{
	if ( map == 0 )
		return(0);
	return(map->epoch_device);
}

SparkStatus SparkWeightdMapDestroy(SparkWeightdMap *map)
{
	uint32_t i;
	SparkStatus status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (i=0u; i<MAP_LEASE_COUNT; i++)
		if ( map->slots[i].state != MAP_EMPTY )
			SPARK_FAIL(SPARK_STATUS_BUSY);
	map->failure = SPARK_STATUS_IO_ERROR;
	status = map_free_initial(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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
		if ( map->pool_mapped == 0u )
		{
			if ( map->mapped[i] != 0u )
			{
				if ( cuMemUnmap(map->base + i * map->chunk_bytes,(size_t)map->chunk_bytes) != CUDA_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_IO_ERROR);
				map->mapped[i] = 0u;
			}
			if ( map->handles[i] != 0 )
			{
				if ( cuMemRelease(map->handles[i]) != CUDA_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_IO_ERROR);
				map->handles[i] = 0;
			}
		}
		map->owners[i] = 0u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus map_release_locked(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout)
{
	SparkWeightdMapSlot *slot;
	SparkWeightdWorkingSetResult result;
	cudaError_t ready;
	SparkStatus status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	slot = map_slot(map,identifier);
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( slot->state == MAP_INFLIGHT )
		SPARK_FAIL(SPARK_STATUS_BUSY);
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
		fprintf(stderr,
		    "MAP-RELEASE-SLOT-ERROR id=%llu status=%d — scoped, map stays usable\n",
		    (unsigned long long)identifier,(int)status);
		SPARK_RETURN(status);
	}
	status = SparkWeightdClientRelease(map->client,map->generation,identifier,&result,timeout);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND )
		SPARK_RETURN(status);
	slot->identifier = 0u;
	slot->state = MAP_EMPTY;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout)
{
	SparkStatus status;
	if ( map == 0 || map->client_lock_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&map->client_lock) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = map_release_locked(map,identifier,timeout);
	(void)pthread_mutex_unlock(&map->client_lock);
	return(status);
}

static SparkStatus map_import_chunk(SparkWeightdMap *map,uint32_t slot,uint32_t chunk,int32_t fd)
{
	CUmemAccessDesc access;
	uint64_t bit = (UINT64_C(1) << slot);
	if ( map->mapped[chunk] != 0u )
	{
		map->owners[chunk] |= bit;
		return(SPARK_STATUS_OK);
	}
	if ( map->owners[chunk] != 0u )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	map->owners[chunk] = bit;
	if ( cuMemImportFromShareableHandle(&map->handles[chunk],(void *)(intptr_t)fd,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( cuMemMap(map->base + (chunk * map->chunk_bytes),(size_t)map->chunk_bytes,0u,map->handles[chunk],0u) != CUDA_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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
	SPARK_RETURN(status);
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
			SPARK_RETURN(status);
		status = SparkWeightdClientExportLeaseBatch(map->client,map->generation,map->slots[slot].identifier,offset,&batch,timeout);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( batch.status != SPARK_STATUS_OK )
			return(batch.status);
		status = map_import_batch(map,slot,&batch,&last,offset,total,deadline);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*identifier = 0u;
	status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( map->failure != SPARK_STATUS_OK )
	{
		static uint32_t sticky_reported;
		if ( sticky_reported == 0u )
		{
			sticky_reported = 1u;
			fprintf(stderr,
			    "MAP-STICKY-FAILURE cached=%d — every acquire fast-fails until process restart\n",
			    (int)map->failure);
		}
		return(map->failure);
	}
	now = map_now();
	if ( now == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( timeout == 0u )
		timeout = SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS;
	if ( timeout > (UINT64_MAX - now) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	deadline = (now + timeout);
	{
		uint64_t lock_wait_start = map_now();
		if ( map->client_lock_initialized == 0u ||
		     pthread_mutex_lock(&map->client_lock) != 0 )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( map_now() - lock_wait_start >= UINT64_C(10000000000) )
			fprintf(stderr,
				"ACQUIRE-STALL mutex-wait %llu ms — another acquire holds the client; ITS exchange is the hang\n",
				(unsigned long long)((map_now() - lock_wait_start) / 1000000ull));
	}
	{
		uint64_t exchange_start = map_now();
	for (slot=0u; slot<MAP_LEASE_COUNT; slot++)
		if ( map->slots[slot].state == MAP_RETIRING )
		{
			(void)pthread_mutex_unlock(&map->client_lock);
			SPARK_FAIL(SPARK_STATUS_BUSY);
		}
	for (slot=0u; slot<MAP_LEASE_COUNT; slot++)
		if ( map->slots[slot].state == MAP_EMPTY )
			break;
	if ( slot == MAP_LEASE_COUNT )
	{
		(void)pthread_mutex_unlock(&map->client_lock);
		SPARK_FAIL(SPARK_STATUS_BUSY);
	}
	status = SparkWeightdClientAcquire(map->client,map->generation,keys,count,&result,timeout);
		if ( map_now() - exchange_start >= UINT64_C(10000000000) )
			fprintf(stderr,
				"ACQUIRE-STALL exchange %llu ms — the weightd round-trip itself hung (server-side lease/load)\n",
				(unsigned long long)((map_now() - exchange_start) / 1000000ull));
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)pthread_mutex_unlock(&map->client_lock);
		SPARK_RETURN(status);
	}
	map->slots[slot].identifier = result.lease_identifier;
	map->slots[slot].state = MAP_ACQUIRED;
	*identifier = result.lease_identifier;
	status = map_import_lease(map,slot,deadline);
	if ( status != SPARK_STATUS_OK )
	{
		map->slots[slot].state = MAP_RETIRING;
		if ( map_remaining(deadline,&remaining) == SPARK_STATUS_OK )
		{
			SparkStatus release_status = map_release_locked(map,*identifier,remaining);
			if ( release_status == SPARK_STATUS_OK )
				*identifier = 0u;
		}
	}
	(void)pthread_mutex_unlock(&map->client_lock);
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdMapBeginUse(SparkWeightdMap *map,uint64_t identifier,void **address)
{
	SparkWeightdMapSlot *slot;
	SparkStatus status;
	if ( address == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*address = 0;
	status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( map->failure != SPARK_STATUS_OK )
		return(map->failure);
	slot = map_slot(map,identifier);
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( slot->state != MAP_ACQUIRED )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	slot->state = MAP_INFLIGHT;
	*address = (void *)(uintptr_t)map->base;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapBase(const SparkWeightdMap *map,void **address)
{
	if ( map == 0 || address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*address = 0;
	if ( map->base == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	*address = (void *)(uintptr_t)map->base;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t identifier,cudaStream_t stream)
{
	SparkWeightdMapSlot *slot;
	SparkStatus status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	slot = map_slot(map,identifier);
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( slot->state != MAP_INFLIGHT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaEventRecord(slot->event,stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	slot->state = MAP_RECORDED;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapSpineCopy(const SparkWeightdMap *map,
	const SparkWeightdManifest *manifest,void *destination,uint64_t capacity)
{
	uint32_t index;
	if ( map == 0 || manifest == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( map->pool_mapped == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( manifest->spine_allocation_bytes > capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( manifest->spine_allocation_bytes != 0u &&
		(destination == 0 || ((uintptr_t)destination & 255u) != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* Prong 2 (hill-climb): the pool import maps the daemon's WHOLE pack
	 * image at map->base - the daemon-verified bytes at their file
	 * offsets. Copy each compacted spine span device-to-device; the
	 * caller quarantines the destination until success, so a faulting
	 * copy followed by the file-path fallback rewrites the same bytes. */
	for (index=0u; index<manifest->spine_count; index++)
	{
		const SparkWeightdSpan *span = &manifest->spine[index];
		uint64_t span_offset = span->offset;
		while ( span_offset < span->offset + span->bytes )
		{
			uint64_t remain = (span->offset + span->bytes) - span_offset;
			size_t chunk = (size_t)(remain < UINT64_C(8388608) ? remain : UINT64_C(8388608));
			CUdeviceptr source = map->base + span_offset;
			CUdeviceptr target = (CUdeviceptr)(uintptr_t)destination +
				span->compact_offset + (span_offset - span->offset);
			if ( cuMemcpyDtoD(target,source,chunk) != CUDA_SUCCESS )
				return(SPARK_STATUS_IO_ERROR);
			span_offset += (uint64_t)chunk;
		}
	}
	return(SPARK_STATUS_OK);
}
