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

SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack)
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
	if ( pack->spine_allocation != 0 )
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

#define LAZY_HELD_STAGING_BYTES (UINT64_C(256) << 20)

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

#if defined(__linux__)
static SparkStatus lazy_held_check(int fd,uint64_t bytes)
{
	const int wanted = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
	struct stat info;
	const int seals = fcntl(fd,F_GET_SEALS);
	if ( fstat(fd,&info) != 0 || S_ISREG(info.st_mode) == 0 || (uint64_t)info.st_size != bytes || seals < 0 || (seals & wanted) != wanted )
	{
		fprintf(stderr,"weightd held pack refused: the held copy is not a sealed %llu-byte memfd\n",(unsigned long long)bytes);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus lazy_held_copy(SparkWeightdLazyPack *pack,const uint8_t *held)
{
	void *staging = 0;
	uint64_t done,piece;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	if ( SparkGpuHostAlloc(&staging,(size_t)LAZY_HELD_STAGING_BYTES,SPARK_GPU_HOST_ALLOC_DEFAULT) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( index = 0u; status == SPARK_STATUS_OK && index < pack->manifest.spine_count; index++ )
	{
		const SparkWeightdSpan *span = &pack->manifest.spine[index];
		for ( done = 0u; status == SPARK_STATUS_OK && done < span->bytes; done += piece )
		{
			piece = (span->bytes - done) < LAZY_HELD_STAGING_BYTES ? (span->bytes - done) : LAZY_HELD_STAGING_BYTES;
			memcpy(staging,held + span->offset + done,(size_t)piece);
			if ( SparkGpuMemcpy((uint8_t *)pack->spine + span->compact_offset + done,staging,(size_t)piece,SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
				status = SPARK_STATUS_IO_ERROR;
		}
	}
	if ( SparkGpuFreeHost(staging) != SPARK_GPU_SUCCESS && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

static SparkStatus lazy_pack_held(SparkWeightdLazyPack *pack,const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t timeout,SparkWeightdManifestCheck check,void *context)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 8u];
	SparkWeightdHoldResult held;
	const uint64_t bytes = request->identity.arena_bytes;
	const uint8_t *mapping;
	uint64_t started_ns,hold_ns,copy_ns;
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
	status = lazy_held_check(held.fd,bytes);
	mapping = status == SPARK_STATUS_OK ? mmap(0,(size_t)bytes,PROT_READ,MAP_SHARED,held.fd,0) : MAP_FAILED;
	(void)close(held.fd);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( mapping == MAP_FAILED )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkGpuMalloc(&pack->spine_allocation,(size_t)(pack->manifest.spine_allocation_bytes + 255u)) != SPARK_GPU_SUCCESS )
	{
		pack->spine_allocation = 0;
		(void)munmap((void *)mapping,(size_t)bytes);
		fprintf(stderr,"weightd held pack refused: no %llu bytes of device memory for the spine\n",(unsigned long long)(pack->manifest.spine_allocation_bytes + 255u));
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	pack->spine_allocation_bytes = (pack->manifest.spine_allocation_bytes + 255u);
	pack->spine = (void *)(((uintptr_t)pack->spine_allocation + 255u) & ~(uintptr_t)255u);
	copy_ns = SparkMonotonicNs();
	status = lazy_held_copy(pack,mapping);
	copy_ns = SparkMonotonicNs() - copy_ns;
	if ( munmap((void *)mapping,(size_t)bytes) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
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
#else
static SparkStatus lazy_pack_held(SparkWeightdLazyPack *pack,const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t timeout,SparkWeightdManifestCheck check,void *context)
{
	(void)pack;
	(void)socket;
	(void)timeout;
	(void)check;
	(void)context;
	fprintf(stderr,"weightd held pack refused: %s; a held pack is a sealed memfd, which this platform lacks\n",request->pack_path);
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}
#endif

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
