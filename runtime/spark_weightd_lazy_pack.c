#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd_spine.h"
#include "sparkpipe/spark_weightd_device.h"
#include "sparkpipe/spark_clock.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "sparkpipe/spark_gpu.h"

SparkStatus SparkWeightdLazyPackUnmap(SparkWeightdLazyPack *pack)
{
	SparkStatus status;
	if ( pack == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	pack->ready = 0u;
	if ( pack->worker != 0 )
	{
		status = SparkWeightdWorkerDestroy(pack->worker);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		pack->worker = 0;
	}
	if ( pack->map != 0 )
	{
		status = SparkWeightdMapDestroy(pack->map);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		pack->map = 0;
		if ( pack->spine_allocation == 0 )
			pack->spine = 0;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLazyPackDetach(SparkWeightdLazyPack *pack,uint64_t timeout_nanoseconds)
{
	SparkWeightdDetachResult result;
	SparkStatus status;
	if ( pack == 0 || pack->client == 0 || pack->attached.arena_generation == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkWeightdLazyPackUnmap(pack);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&result,0,sizeof(result));
	status = SparkWeightdClientDetach(pack->client,pack->attached.arena_generation,&result,timeout_nanoseconds);
	if ( status == SPARK_STATUS_OK && result.status != SPARK_STATUS_OK )
		status = result.status;
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	pack->attached.arena_generation = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLazyPackTakeMeshMapping(SparkWeightdLazyPack *pack,void **mapping,uint64_t *bytes)
{
	if ( pack == 0 || mapping == 0 || bytes == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*mapping = pack->attached.mesh_mapping;
	*bytes = pack->attached.mesh_mapping != 0 ? pack->attached.mesh_send_buffer_bytes : 0u;
	pack->attached.mesh_mapping = 0;
	return(SPARK_STATUS_OK);
}

#define LAZY_SPINE_VMM_CHUNK_BYTES (UINT64_C(256) << 20)

static SparkStatus lazy_spine_vmm_release(SparkWeightdLazyPack *pack)
{
	SparkStatus status = SPARK_STATUS_OK;
	const SparkDeviceAddress base = (SparkDeviceAddress)(uintptr_t)pack->spine_allocation;
	uint32_t index;
	if ( pack->spine_chunks != 0 )
	{
		for ( index = 0u; index < pack->spine_chunk_count; index++ )
		{
			if ( pack->spine_chunks[index] == 0u )
				continue;
			if ( SparkDeviceMemoryUnmap(base + (SparkDeviceAddress)index * pack->spine_chunk_bytes,pack->spine_chunk_bytes) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
				status = SPARK_STATUS_IO_ERROR;
			if ( SparkDeviceMemoryRelease(pack->spine_chunks[index]) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
				status = SPARK_STATUS_IO_ERROR;
			pack->spine_chunks[index] = 0u;
		}
		free(pack->spine_chunks);
		pack->spine_chunks = 0;
	}
	if ( pack->spine_reserved_bytes != 0u && base != 0u )
	{
		if ( SparkDeviceAddressRelease(base,pack->spine_reserved_bytes) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
			status = SPARK_STATUS_IO_ERROR;
	}
	pack->spine_allocation = 0;
	pack->spine = 0;
	pack->spine_allocation_bytes = 0u;
	pack->spine_chunk_bytes = 0u;
	pack->spine_reserved_bytes = 0u;
	pack->spine_chunk_count = 0u;
	SPARK_RETURN(status);
}

static SparkStatus lazy_spine_vmm_ensure(void *context,uint64_t offset,uint64_t bytes)
{
	SparkWeightdLazyPack *pack = context;
	const SparkDeviceAddress base = (SparkDeviceAddress)(uintptr_t)pack->spine_allocation;
	uint32_t first,last,index;
	if ( bytes == 0u )
		return(SPARK_STATUS_OK);
	first = (uint32_t)(offset / pack->spine_chunk_bytes);
	last = (uint32_t)((offset + bytes - 1u) / pack->spine_chunk_bytes);
	for ( index = first; index <= last && index < pack->spine_chunk_count; index++ )
	{
		SparkDeviceMemoryHandle handle = 0u;
		SparkStatus status;
		if ( pack->spine_chunks[index] != 0u )
			continue;
		status = SparkDeviceMemoryCreate(&handle,pack->spine_chunk_bytes,"private weight spine");
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
		status = SparkDeviceMemoryMap(base + (SparkDeviceAddress)index * pack->spine_chunk_bytes,pack->spine_chunk_bytes,handle,0u,
			SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkDeviceMemoryRelease(handle);
			SPARK_FAIL(status);
		}
		pack->spine_chunks[index] = handle;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus lazy_spine_vmm_reserve(SparkWeightdLazyPack *pack,uint64_t bytes)
{
	uint64_t granularity = 0u,chunk,count,reserved;
	SparkDeviceAddress base = 0u;
	SparkStatus status;
	if ( bytes < LAZY_SPINE_VMM_CHUNK_BYTES )
		return(SPARK_STATUS_UNSUPPORTED);
	status = SparkDeviceMemoryGranularity(SPARK_DEVICE_GRANULARITY_MINIMUM,&granularity);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( granularity == 0u || granularity > LAZY_SPINE_VMM_CHUNK_BYTES )
		return(SPARK_STATUS_UNSUPPORTED);
	chunk = ((LAZY_SPINE_VMM_CHUNK_BYTES + granularity - 1u) / granularity) * granularity;
	count = (bytes + chunk - 1u) / chunk;
	if ( count == 0u || count > UINT32_MAX || count > UINT64_MAX / chunk )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	reserved = count * chunk;
	status = SparkDeviceAddressReserve(&base,reserved,"private weight spine");
	if ( status != SPARK_STATUS_OK )
		return(status);
	pack->spine_chunks = calloc((size_t)count,sizeof(*pack->spine_chunks));
	if ( pack->spine_chunks == 0 )
	{
		(void)SparkDeviceAddressRelease(base,reserved);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	pack->spine_allocation = (void *)(uintptr_t)base;
	pack->spine = pack->spine_allocation;
	pack->spine_allocation_bytes = reserved;
	pack->spine_reserved_bytes = reserved;
	pack->spine_chunk_bytes = chunk;
	pack->spine_chunk_count = (uint32_t)count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack)
{
	SparkStatus status;
	if ( pack == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkWeightdLazyPackUnmap(pack);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pack->spine_chunks != 0 )
	{
		status = lazy_spine_vmm_release(pack);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	else if ( pack->spine_allocation != 0 )
	{
		if ( SparkGpuFree(pack->spine_allocation) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		pack->spine_allocation = 0;
		pack->spine = 0;
	}
	if ( pack->attached.mesh_mapping != 0 )
	{
		(void)munmap(pack->attached.mesh_mapping,pack->attached.mesh_send_buffer_bytes);
		pack->attached.mesh_mapping = 0;
	}
	if ( pack->client != 0 )
		SparkWeightdClientClose(pack->client);
	SparkWeightdManifestDestroy(&pack->manifest);
	free(pack);
	return(SPARK_STATUS_OK);
}

static SparkStatus lazy_spine_load(SparkWeightdLazyPack *pack,int32_t fd,const SparkWeightdLazyAttachRequest *request)
{
	uint64_t bytes = pack->manifest.spine_allocation_bytes;
	const void *pool;
	int lock_fd = -1;
	SparkStatus status;
	if ( bytes == 0u )
		return(SPARK_STATUS_OK);
	if ( bytes > (SIZE_MAX - 255u) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkWeightdMapPool(pack->map,&pool);
	if ( status == SPARK_STATUS_OK )
	{
		pack->spine = (void *)(uintptr_t)pool;
		fprintf(stderr,"WD-SPINE-SHARED spans=%u bytes=%llu private=0\n",pack->manifest.spine_count,(unsigned long long)pack->manifest.spine_bytes);
		return(SPARK_STATUS_OK);
	}
	if ( status != SPARK_STATUS_UNSUPPORTED )
		SPARK_RETURN(status);
	status = SparkWeightdNodeMemoryLock(SPARK_WEIGHTD_NODE_MEMORY_LOCK_PATH,&lock_fd);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkWeightdNodeMemoryAdmit(SPARK_WEIGHTD_MEMINFO_PATH,bytes + 255u,"private weight spine",request->identity.model);
	if ( status == SPARK_STATUS_OK )
	{
		const SparkStatus vmm = lazy_spine_vmm_reserve(pack,bytes);
		if ( vmm == SPARK_STATUS_OK )
		{
			fprintf(stderr,"WD-SPINE-PRIVATE spans=%u bytes=%llu vmm_chunks=%u chunk_bytes=%llu\n",pack->manifest.spine_count,
				(unsigned long long)pack->spine_allocation_bytes,pack->spine_chunk_count,(unsigned long long)pack->spine_chunk_bytes);
			status = SparkWeightdSpineLoadEnsured(request->pack_path,fd,&pack->manifest,request->identity.arena_bytes,request->identity.pack_sha256,
				pack->spine,bytes,lazy_spine_vmm_ensure,pack);
			if ( status == SPARK_STATUS_OK )
				status = lazy_spine_vmm_ensure(pack,0u,bytes);
			SparkWeightdNodeMemoryUnlock(lock_fd);
			SPARK_RETURN(status);
		}
		if ( vmm != SPARK_STATUS_UNSUPPORTED )
			status = vmm;
	}
	if ( status == SPARK_STATUS_OK && SparkGpuMalloc(&pack->spine_allocation,(size_t)(bytes + 255u)) != SPARK_GPU_SUCCESS )
	{
		pack->spine_allocation = 0;
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	SparkWeightdNodeMemoryUnlock(lock_fd);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	pack->spine_allocation_bytes = (bytes + 255u);
	pack->spine = (void *)(((uintptr_t)pack->spine_allocation + 255u) & ~(uintptr_t)255u);
	fprintf(stderr,"WD-SPINE-PRIVATE spans=%u bytes=%llu\n",pack->manifest.spine_count,(unsigned long long)pack->spine_allocation_bytes);
	return(SparkWeightdSpineLoad(request->pack_path,fd,&pack->manifest,request->identity.arena_bytes,request->identity.pack_sha256,pack->spine,bytes));
}

static SparkStatus lazy_pack_initialize(SparkWeightdLazyPack *pack,int32_t fd,const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t cap,uint64_t timeout,SparkWeightdManifestCheck check,void *context)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 8u];
	SparkWeightdLazyAttachRequest sized;
	uint8_t manifest_digest[32];
	SparkStatus status;
	status = SparkWeightdClientConnect(socket,&pack->client,0);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	(void)snprintf(path,sizeof(path),"%s.experts",request->pack_path);
	status = SparkWeightdManifestLoad(path,request->identity.arena_bytes,&pack->manifest);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkWeightdManifestPoolPlan(&pack->manifest,request->identity.arena_bytes,cap,&pack->pool);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	sized = *request;
	sized.expert_pool_bytes = pack->pool.request_bytes;
	status = pack->read_only != 0u
		? SparkWeightdClientAttachLazyShared(pack->client,&sized,&pack->attached,timeout)
		: SparkWeightdClientAttachLazy(pack->client,&sized,&pack->attached,timeout);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pack->attached.chunk_bytes != SPARK_WEIGHTD_EXPERT_CHUNK_BYTES )
	{
		fprintf(stderr,"weightd pool refused: weightd uses %llu-byte chunks, the pool was sized in %llu-byte chunks\n",
		    (unsigned long long)pack->attached.chunk_bytes,(unsigned long long)SPARK_WEIGHTD_EXPERT_CHUNK_BYTES);
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	}
	if ( check != 0 )
	{
		status = check(&pack->manifest,context);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		SparkWeightdManifestDestroy(&pack->manifest);
		memset(&pack->manifest,0,sizeof(pack->manifest));
		status = SparkWeightdManifestLoad(path,request->identity.arena_bytes,&pack->manifest);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	{
		int epoch_fd = -1;
		(void)SparkWeightdClientEpochExport(pack->client,
		    pack->attached.arena_generation,&epoch_fd,timeout);
		status = SparkWeightdMapCreateAccess(pack->client,
		    &pack->attached,epoch_fd,
		    pack->attached.pool_fd,pack->read_only,&pack->map);
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = lazy_spine_load(pack,fd,request);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pack->attached.loaded_from_pack != 0u )
	{
		status = SparkWeightdManifestIdentity(&pack->manifest,manifest_digest);
		if ( status == SPARK_STATUS_OK && memcmp(manifest_digest,pack->attached.manifest_sha256,sizeof(manifest_digest)) != 0 )
		{
			fprintf(stderr,"LAZY-MANIFEST-MISMATCH local=");
			for (uint32_t i=0u;i<32u;i++) fprintf(stderr,"%02x",manifest_digest[i]);
			fprintf(stderr," weightd=");
			for (uint32_t i=0u;i<32u;i++) fprintf(stderr,"%02x",pack->attached.manifest_sha256[i]);
			fprintf(stderr,"\n");
			status = SPARK_STATUS_HASH_MISMATCH;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdWorkerCreate(&pack->worker);
	SPARK_RETURN(status);
}

#define LAZY_HELD_PIECE_BYTES (UINT64_C(1) << 30)

static SparkStatus lazy_pack_unified(uint32_t *unified)
{
	const SparkDeviceProperties *properties;
	const SparkStatus status = SparkWeightdDeviceOpen();
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	properties = SparkDeviceGetProperties();
	if ( properties == 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	*unified = (properties->properties & SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY) != 0u ? 1u : 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus lazy_held_copy(SparkWeightdLazyPack *pack,SparkDeviceAddress held)
{
	uint64_t done,piece;
	uint32_t index;
	for ( index = 0u; index < pack->manifest.spine_count; index++ )
	{
		const SparkWeightdSpan *span = &pack->manifest.spine[index];
		for ( done = 0u; done < span->bytes; done += piece )
		{
			piece = (span->bytes - done) < LAZY_HELD_PIECE_BYTES ? (span->bytes - done) : LAZY_HELD_PIECE_BYTES;
			if ( SparkGpuMemcpy((uint8_t *)pack->spine + span->compact_offset + done,(const void *)(uintptr_t)(held + span->offset + done),(size_t)piece,
				SPARK_GPU_COPY_DEVICE_TO_DEVICE) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus lazy_held_bind(SparkWeightdLazyPack *pack,const SparkWeightdHoldResult *held,uint64_t *copy_ns)
{
	SparkDeviceMemoryHandle memory = 0u;
	SparkDeviceAddress address = 0u;
	uint32_t mapped = 0u;
	SparkStatus status = SparkDeviceMemoryImport(&memory,held->fd,"weightd held pack");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAddressReserve(&address,held->allocation_bytes,"weightd held pack");
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkDeviceMemoryMap(address,held->allocation_bytes,memory,0u,SPARK_DEVICE_ACCESS_DEVICE_READ);
		mapped = status == SPARK_STATUS_OK ? 1u : 0u;
	}
	if ( status == SPARK_STATUS_OK && SparkGpuMalloc(&pack->spine_allocation,(size_t)(pack->manifest.spine_allocation_bytes + 255u)) != SPARK_GPU_SUCCESS )
	{
		pack->spine_allocation = 0;
		fprintf(stderr,"weightd held pack refused: no %llu bytes of device memory for the spine\n",(unsigned long long)(pack->manifest.spine_allocation_bytes + 255u));
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status == SPARK_STATUS_OK )
	{
		pack->spine_allocation_bytes = (pack->manifest.spine_allocation_bytes + 255u);
		pack->spine = (void *)(((uintptr_t)pack->spine_allocation + 255u) & ~(uintptr_t)255u);
		*copy_ns = SparkMonotonicNs();
		status = lazy_held_copy(pack,address);
		*copy_ns = SparkMonotonicNs() - *copy_ns;
	}
	if ( mapped != 0u && SparkDeviceMemoryUnmap(address,held->allocation_bytes) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( address != 0u && SparkDeviceAddressRelease(address,held->allocation_bytes) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( memory != 0u && SparkDeviceMemoryRelease(memory) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

static SparkStatus lazy_pack_held(SparkWeightdLazyPack *pack,const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t timeout,SparkWeightdManifestCheck check,void *context)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 8u];
	SparkWeightdHoldResult held;
	const uint64_t bytes = request->identity.arena_bytes;
	uint64_t started_ns,hold_ns,copy_ns = 0u;
	SparkStatus status;
	if ( pack->read_only != 0u )
	{
		fprintf(stderr,"weightd held pack refused: a read-only lazy pack shares weightd's resident arena, which a device without unified memory does not have\n");
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	status = SparkWeightdClientConnect(socket,&pack->client,0);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	(void)snprintf(path,sizeof(path),"%s.experts",request->pack_path);
	status = SparkWeightdManifestLoad(path,bytes,&pack->manifest);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pack->manifest.range_count != 0u )
	{
		fprintf(stderr,"weightd held pack refused: %s names %u expert ranges; on a device without unified memory a pack binds whole, as its spine\n",path,pack->manifest.range_count);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	if ( check != 0 )
	{
		status = check(&pack->manifest,context);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		SparkWeightdManifestDestroy(&pack->manifest);
		memset(&pack->manifest,0,sizeof(pack->manifest));
		status = SparkWeightdManifestLoad(path,bytes,&pack->manifest);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	started_ns = SparkMonotonicNs();
	status = SparkWeightdClientHold(pack->client,&request->identity,request->pack_path,&held,timeout);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	hold_ns = SparkMonotonicNs() - started_ns;
	if ( held.held_bytes != bytes || held.allocation_bytes < bytes )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = lazy_held_bind(pack,&held,&copy_ns);
	(void)close(held.fd);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	pack->attached.status = SPARK_STATUS_OK;
	pack->attached.arena_generation = held.generation;
	pack->attached.arena_bytes = bytes;
	pack->attached.loaded_from_pack = held.first_hold;
	pack->attached.pool_fd = -1;
	fprintf(stderr,"WD-SPINE-HELD spans=%u bytes=%llu first_hold=%u held=%u hold_s=%.3f copy_s=%.3f\n",pack->manifest.spine_count,
	    (unsigned long long)pack->manifest.spine_bytes,held.first_hold,held.held_count,(double)hold_ns / 1e9,(double)copy_ns / 1e9);
	SPARK_RETURN(SparkWeightdWorkerCreate(&pack->worker));
}

SparkStatus SparkWeightdLazyPackCreate(const char *socket,const SparkWeightdLazyAttachRequest *request,const SparkWeightdLazyPackOptions *options,SparkWeightdLazyPack **out)
{
	SparkWeightdLazyPack *pack;
	SparkWeightdIdentity identity;
	SparkWeightdLazyAttachRequest resolved;
	SparkStatus status;
	struct stat info;
	int32_t fd;
	uint32_t unified = 1u;
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	char absolute[4096];
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( socket == 0 || socket[0] == '\0' || request == 0 || options == 0 || options->read_only > 1u || request->pack_path[0] == 0 || memchr(request->pack_path,0,sizeof(request->pack_path)) == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->expert_pool_bytes != 0u )
	{
		fprintf(stderr,"weightd lazy pack refused: the request names expert_pool_bytes %llu; the pack sizes its pool from its expert manifest and takes only an explicit cap\n",
		    (unsigned long long)request->expert_pool_bytes);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( request->pack_path[0] != '/' )
	{
		if ( realpath(request->pack_path,absolute) == 0 )
			return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
		if ( strlen(absolute) >= sizeof(resolved.pack_path) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		resolved = *request;
		memcpy(resolved.pack_path,absolute,strlen(absolute) + 1u);
		request = &resolved;
	}
	identity = request->identity;
	if ( SparkWeightdIdentityPrepare(&identity) != SPARK_STATUS_OK || SparkSha256HexToDigest(identity.pack_sha256,digest) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	fd = open(request->pack_path,O_RDONLY | O_NONBLOCK);
	if ( fd < 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	if ( fstat(fd,&info) != 0 || S_ISREG(info.st_mode) == 0 || info.st_size <= 0 || (uint64_t)info.st_size != identity.arena_bytes )
	{
		(void)close(fd);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	pack = calloc(1u,sizeof(*pack));
	if ( pack == 0 )
	{
		(void)close(fd);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	pack->read_only = options->read_only;
	memcpy(pack->pack_sha256,digest,sizeof(pack->pack_sha256));
	status = lazy_pack_unified(&unified);
	if ( status == SPARK_STATUS_OK )
		status = unified != 0u
			? lazy_pack_initialize(pack,fd,socket,request,options->expert_pool_cap_bytes,options->timeout_nanoseconds,options->manifest_check,options->manifest_check_context)
			: lazy_pack_held(pack,socket,request,options->timeout_nanoseconds,options->manifest_check,options->manifest_check_context);
	if ( close(fd) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK && SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK )
		pack = 0;
	if ( status == SPARK_STATUS_OK )
		pack->ready = 1u;
	*out = pack;
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdLazyPackSlice(const SparkWeightdLazyPack *pack,uint64_t offset,uint64_t bytes,const void **pointer)
{
	uint64_t compact;
	SparkStatus status;
	if ( pointer == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*pointer = 0;
	if ( pack == 0 || pack->ready == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkWeightdManifestSpineSlice(&pack->manifest,offset,bytes,&compact);
	if ( status == SPARK_STATUS_OK )
		*pointer = ((uint8_t *)pack->spine + (pack->spine_allocation != 0 ? compact : offset));
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdLazyPackPrivateSlice(SparkWeightdLazyPack *pack,uint64_t offset,uint64_t bytes,void **pointer)
{
	const void *slice = 0;
	SparkStatus status;
	if ( pointer == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*pointer = 0;
	if ( pack == 0 || pack->ready == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pack->spine_allocation == 0 )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	status = SparkWeightdLazyPackSlice(pack,offset,bytes,&slice);
	if ( status == SPARK_STATUS_OK )
		*pointer = (void *)slice;
	SPARK_RETURN(status);
}
