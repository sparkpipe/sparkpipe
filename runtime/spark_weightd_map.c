#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_error_site.h"
#include <pthread.h>
#include "sparkpipe/spark_weightd_device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "sparkpipe/spark_gpu.h"

enum
{
	MAP_EMPTY,
	MAP_ACQUIRED,
	MAP_INFLIGHT,
	MAP_RECORDED,
	MAP_RETIRING,
	MAP_LEASE_COUNT = 64,
	MAP_COMPLETION_POLL_NS = 20000
};

typedef struct SparkWeightdMapSlot
{
	uint64_t identifier;
	SparkGpuEvent event;
	uint32_t state;
} SparkWeightdMapSlot;

struct SparkWeightdMap
{
	SparkWeightdClient *client;
	pthread_mutex_t client_lock;
	uint32_t client_lock_initialized;
	SparkDeviceInstance instance;
	SparkDeviceAddress base;
	uint64_t generation,span_bytes,chunk_bytes;
	void *epoch_device;
	void *epoch_handle;
	uint32_t chunk_count;
	uint32_t pool_mapped;
	uint32_t read_only;
	int32_t device;
	SparkStatus failure;
	SparkDeviceMemoryHandle *handles;
	SparkDeviceMemoryHandle *slot_handles;
	uint32_t *chunk_slots;
	uint32_t slot_count;
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
	SparkDeviceInstance instance = 0u;
	int32_t device;
	if ( map == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkDeviceThreadInstance(&instance) != SPARK_STATUS_OK )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(device == map->device && instance == map->instance ? SPARK_STATUS_OK : SPARK_STATUS_TARGET_MISMATCH);
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
		if ( SparkDeviceMemoryUnmap(map->base + map->span_bytes,
		    map->chunk_bytes) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->epoch_device = 0;
	}
	if ( map->pool_mapped != 0u )
	{
		if ( SparkDeviceMemoryUnmap(map->base,map->span_bytes) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->pool_mapped = 0u;
		memset(map->mapped,0,map->chunk_count * sizeof(*map->mapped));
	}
	for (i=0u; map->mapped != 0 && i<map->chunk_count; i++)
		if ( map->mapped[i] != 0u )
		{
			if ( SparkDeviceMemoryUnmap(map->base + i * map->chunk_bytes,
			    map->chunk_bytes) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			map->mapped[i] = 0u;
		}
	if ( map->epoch_handle != 0 )
	{
		if ( SparkDeviceMemoryRelease(SparkWeightdMemoryOf(map->epoch_handle)) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->epoch_handle = 0;
	}
	for (i=0u; map->handles != 0 && i<map->chunk_count; i++)
	{
		uint32_t j;
		SparkDeviceMemoryHandle handle = map->handles[i];
		if ( handle == 0u )
			continue;
		if ( SparkDeviceMemoryRelease(handle) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		for (j=i; j<map->chunk_count; j++)
			if ( map->handles[j] == handle )
				map->handles[j] = 0;
	}
	for (i=0u; map->slot_handles != 0 && i<map->slot_count; i++)
		if ( map->slot_handles[i] != 0 )
		{
			if ( SparkDeviceMemoryRelease(map->slot_handles[i]) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			map->slot_handles[i] = 0;
		}
	if ( map->base != 0u )
	{
		if ( SparkDeviceAddressRelease(map->base,
		    map->span_bytes + map->chunk_bytes) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->base = 0u;
	}
	for (i=0u; i<MAP_LEASE_COUNT; i++)
		if ( map->slots[i].event != 0 )
		{
			if ( SparkGpuEventDestroy(map->slots[i].event) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			map->slots[i].event = 0;
		}
	free(map->handles);
	free(map->slot_handles);
	free(map->chunk_slots);
	free(map->owners);
	free(map->mapped);
	if ( map->client_lock_initialized != 0u )
		(void)pthread_mutex_destroy(&map->client_lock);
	free(map);
	return(SPARK_STATUS_OK);
}

static SparkStatus map_initialize_device(SparkWeightdMap *map)
{
	uint64_t granularity = 0u;
	uint32_t i;
	if ( SparkWeightdDeviceOpen() != SPARK_STATUS_OK || SparkGpuGetDevice(&map->device) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkDeviceThreadInstance(&map->instance) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	for (i=0u; i<MAP_LEASE_COUNT; i++)
		if ( SparkGpuEventCreateWithFlags(&map->slots[i].event,SPARK_GPU_EVENT_DISABLE_TIMING) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkDeviceMemoryGranularity(SPARK_DEVICE_GRANULARITY_MINIMUM,&granularity) != SPARK_STATUS_OK || granularity == 0u || (map->chunk_bytes % granularity) != 0u )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	map->epoch_device = 0;
	map->epoch_handle = 0;
	return(SparkDeviceAddressReserve(&map->base,
	    map->span_bytes + map->chunk_bytes,"weightd map") ==
	    SPARK_STATUS_OK ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
}

static SparkStatus map_import_slots(SparkWeightdMap *map)
{
	SparkWeightdSlotBatch batch;
	SparkStatus status;
	uint32_t offset = 0u,i;
	do
	{
		status = SparkWeightdClientSlotExportBatch(map->client,map->generation,offset,&batch,SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( batch.status != SPARK_STATUS_OK )
			return(batch.status);
		if ( batch.slot_count == 0u && offset == 0u )
			return(SPARK_STATUS_OK);
		if ( batch.chunk_bytes != map->chunk_bytes || batch.slot_count > map->chunk_count || (map->slot_count != 0u && batch.slot_count != map->slot_count) )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK && map->slot_handles == 0 )
		{
			map->slot_handles = calloc(batch.slot_count,sizeof(*map->slot_handles));
			map->chunk_slots = malloc(map->chunk_count * sizeof(*map->chunk_slots));
			if ( map->slot_handles == 0 || map->chunk_slots == 0 )
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
			else
			{
				map->slot_count = batch.slot_count;
				for (i=0u; i<map->chunk_count; i++)
					map->chunk_slots[i] = SPARK_WEIGHTD_SLOT_NONE;
			}
		}
		for (i=0u; i<batch.batch_count; i++)
		{
			if ( status == SPARK_STATUS_OK && SparkDeviceMemoryImport(&map->slot_handles[offset + i],batch.fds[i],"weightd map slot") != SPARK_STATUS_OK )
				status = SPARK_STATUS_IO_ERROR;
			(void)close(batch.fds[i]);
		}
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		offset += batch.batch_count;
	} while ( offset < map->slot_count );
	fprintf(stderr,"WD-MAP-SLOTS slots=%u chunk_bytes=%llu — chunk mappings persist across leases and change only when weightd moves a chunk\n",
	    map->slot_count,(unsigned long long)map->chunk_bytes);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapCreate(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,int epoch_fd,int pool_fd,SparkWeightdMap **out)
{
	return(SparkWeightdMapCreateAccess(client,attached,epoch_fd,pool_fd,0u,out));
}

uint32_t SparkWeightdMapReadOnly(const SparkWeightdMap *map)
{
	return(map != 0 ? map->read_only : 0u);
}

SparkStatus SparkWeightdMapCreateAccess(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,int epoch_fd,int pool_fd,uint32_t read_only,SparkWeightdMap **out)
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
	map->read_only = read_only != 0u ? 1u : 0u;
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
		status = map_initialize_device(map);
	if ( status == SPARK_STATUS_OK && epoch_fd >= 0 )
	{
		SparkDeviceMemoryHandle epoch = 0u;
		if ( SparkDeviceMemoryImport(&epoch,epoch_fd,"weightd map epoch") != SPARK_STATUS_OK )
			status = SPARK_STATUS_IO_ERROR;
		else
		{
			map->epoch_handle = (void *)(uintptr_t)epoch;
			if ( SparkDeviceMemoryMap(map->base + map->span_bytes,map->chunk_bytes,epoch,0u,SPARK_WEIGHTD_DEVICE_READ) != SPARK_STATUS_OK )
				status = SPARK_STATUS_IO_ERROR;
			else
				map->epoch_device = (void *)(uintptr_t)(map->base + map->span_bytes);
		}
	}
	if ( status == SPARK_STATUS_OK && pool_fd >= 0 )
	{
		if ( SparkDeviceMemoryImport(&map->handles[0],pool_fd,"weightd map pool") != SPARK_STATUS_OK ||
		     SparkDeviceMemoryMap(map->base,map->span_bytes,map->handles[0],0u,
		         map->read_only != 0u ? SPARK_WEIGHTD_DEVICE_READ : SPARK_WEIGHTD_DEVICE_READ_WRITE) != SPARK_STATUS_OK )
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
			fprintf(stderr,
			    "WD-MAP-POOL-BULK chunks=%u span=%llu access=%s — all chunks mapped in ONE import\n",
			    map->chunk_count,(unsigned long long)map->span_bytes,map->read_only != 0u ? "read-only" : "read-write");
		}
	}
	if ( epoch_fd >= 0 ) (void)close(epoch_fd);
	if ( pool_fd >= 0 ) (void)close(pool_fd);
	if ( status == SPARK_STATUS_OK && map->pool_mapped == 0u )
		status = map_import_slots(map);
	if ( status == SPARK_STATUS_OK && map->pool_mapped == 0u && map->slot_count == 0u )
	{
		fprintf(stderr,"WD-MAP-UNSUPPORTED generation=%llu: weightd exported neither the whole pool nor its slots\n",
		    (unsigned long long)map->generation);
		status = SPARK_STATUS_UNSUPPORTED;
	}
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
		map->owners[i] &= ~bit;
	return(SPARK_STATUS_OK);
}

static SparkStatus map_release_locked(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout)
{
	SparkWeightdMapSlot *slot;
	SparkWeightdWorkingSetResult result;
	SparkGpuError ready;
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
		ready = SparkGpuEventQuery(slot->event);
		if ( ready != SPARK_GPU_SUCCESS )
			return(ready == SPARK_GPU_ERROR_NOT_READY ? SPARK_STATUS_BUSY : SPARK_STATUS_IO_ERROR);
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

SparkStatus SparkWeightdMapReleaseCompleted(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout)
{
	const struct timespec pause = {0,MAP_COMPLETION_POLL_NS};
	SparkWeightdMapSlot *slot;
	uint64_t started = map_now(),deadline,remaining;
	SparkStatus status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( started == 0u || timeout == 0u || timeout > (UINT64_MAX - started) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	deadline = (started + timeout);
	slot = map_slot(map,identifier);
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( slot->state != MAP_RECORDED )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (;;)
	{
		if ( map_remaining(deadline,&remaining) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"MAP-RELEASE-WAIT-EXPIRED id=%llu waited_ns=%llu: the GPU work recorded for this lease did not complete; the lease stays held\n",
			    (unsigned long long)identifier,(unsigned long long)(map_now() - started));
			SPARK_FAIL(SPARK_STATUS_BUSY);
		}
		status = SparkWeightdMapRelease(map,identifier,remaining);
		if ( status != SPARK_STATUS_BUSY || slot->state != MAP_RECORDED )
			SPARK_RETURN(status);
		(void)nanosleep(&pause,0);
	}
}

static SparkStatus map_bind_chunk(SparkWeightdMap *map,uint32_t slot,uint32_t chunk,uint32_t pool_slot)
{
	SparkDeviceAddress address;
	uint64_t bit = (UINT64_C(1) << slot);
	if ( chunk >= map->chunk_count || pool_slot >= map->slot_count )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( map->mapped[chunk] != 0u && map->chunk_slots[chunk] == pool_slot )
	{
		map->owners[chunk] |= bit;
		return(SPARK_STATUS_OK);
	}
	if ( map->owners[chunk] != 0u )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	address = map->base + chunk * map->chunk_bytes;
	if ( map->mapped[chunk] != 0u )
	{
		if ( SparkDeviceMemoryUnmap(address,map->chunk_bytes) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		map->mapped[chunk] = 0u;
		map->chunk_slots[chunk] = SPARK_WEIGHTD_SLOT_NONE;
	}
	if ( SparkDeviceMemoryMap(address,map->chunk_bytes,map->slot_handles[pool_slot],0u,SPARK_WEIGHTD_DEVICE_READ) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	map->mapped[chunk] = 1u;
	map->chunk_slots[chunk] = pool_slot;
	map->owners[chunk] = bit;
	return(SPARK_STATUS_OK);
}

static SparkStatus map_bind_lease(SparkWeightdMap *map,uint32_t slot,uint64_t deadline)
{
	SparkWeightdLeaseSlotBatch batch;
	SparkStatus status;
	uint32_t offset = 0u,total = 0u,i;
	uint64_t timeout;
	do
	{
		status = map_remaining(deadline,&timeout);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		status = SparkWeightdClientLeaseSlotsBatch(map->client,map->generation,map->slots[slot].identifier,offset,&batch,timeout);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( batch.status != SPARK_STATUS_OK )
			return(batch.status);
		if ( offset != 0u && batch.lease_chunk_count != total )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for (i=0u; i<batch.batch_count; i++)
		{
			status = map_bind_chunk(map,slot,batch.chunk_indices[i],batch.slot_indices[i]);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		total = batch.lease_chunk_count;
		offset += batch.batch_count;
	} while ( offset < total );
	return(map_remaining(deadline,&timeout));
}

static SparkStatus map_import_lease(SparkWeightdMap *map,uint32_t slot,uint64_t deadline)
{
	uint64_t timeout;
	if ( map->slot_count != 0u )
		return(map_bind_lease(map,slot,deadline));
	if ( map->pool_mapped == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
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

SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t identifier,SparkGpuStream stream)
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
	if ( SparkGpuEventRecord(slot->event,stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	slot->state = MAP_RECORDED;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapResident(SparkWeightdMap *map,uint64_t timeout,uint32_t *resident)
{
	SparkWeightdResidency residency;
	SparkStatus status;
	if ( resident == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*resident = 0u;
	status = map_context(map);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( map->client_lock_initialized == 0u || pthread_mutex_lock(&map->client_lock) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkWeightdClientResidency(map->client,map->generation,&residency,timeout);
	(void)pthread_mutex_unlock(&map->client_lock);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	*resident = map->pool_mapped != 0u && map->base != 0u && residency.fixed_pool != 0u &&
		residency.group_count != 0u && residency.present_count == residency.group_count ? 1u : 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapPool(const SparkWeightdMap *map,const void **address)
{
	if ( address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*address = 0;
	if ( map == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( map->pool_mapped == 0u || map->base == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	*address = (const void *)(uintptr_t)map->base;
	return(SPARK_STATUS_OK);
}
