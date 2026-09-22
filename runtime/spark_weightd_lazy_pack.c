#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd_spine.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
	}
	if ( pack->spine_allocation != 0 )
	{
		if ( cudaFree(pack->spine_allocation) != cudaSuccess )
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

static SparkStatus lazy_spine_load(SparkWeightdLazyPack *pack,int32_t fd,const SparkWeightdLazyAttachRequest *request,uint64_t budget)
{
	uint64_t bytes = pack->manifest.spine_allocation_bytes;
	SparkStatus status;
	if ( bytes != 0u )
	{
		if ( bytes > (SIZE_MAX - 255u) || (bytes + 255u) > budget )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		pack->spine_allocation_bytes = (bytes + 255u);
		if ( cudaMalloc(&pack->spine_allocation,(size_t)pack->spine_allocation_bytes) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		pack->spine = (void *)(((uintptr_t)pack->spine_allocation + 255u) & ~(uintptr_t)255u);
	}
	/* Prong 2 (hill-climb): when the pool is mapped, the spine spans are
	 * device-readable in the arena image at their file offsets - copy
	 * device-to-device from the daemon's VERIFIED materialization instead
	 * of re-reading the pack file (removes the page-cache dependence that
	 * degrades evicted-cold starts). Programming errors propagate; an
	 * unmapped pool or a copy fault falls back to the proven file path,
	 * which rewrites exactly the same span bytes. */
	if ( pack->map != 0 )
	{
		status = SparkWeightdMapSpineCopy(pack->map,&pack->manifest,pack->spine,bytes);
		if ( status == SPARK_STATUS_OK )
			return(SPARK_STATUS_OK);
		if ( status != SPARK_STATUS_UNSUPPORTED && status != SPARK_STATUS_IO_ERROR )
			SPARK_RETURN(status);
	}
	return(SparkWeightdSpineLoad(fd,&pack->manifest,request->identity.arena_bytes,request->identity.pack_sha256,pack->spine,bytes));
}

static SparkStatus lazy_pack_initialize(SparkWeightdLazyPack *pack,int32_t fd,const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t budget,uint64_t timeout,SparkWeightdManifestCheck check,void *context)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 8u];
	uint8_t manifest_digest[32];
	SparkStatus status;
	status = SparkWeightdClientConnect(socket,&pack->client,0);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkWeightdClientAttachLazy(pack->client,request,&pack->attached,timeout);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	(void)snprintf(path,sizeof(path),"%s.experts",request->pack_path);
	status = SparkWeightdManifestLoad(path,request->identity.arena_bytes,&pack->manifest);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
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
	/* Map before the spine load (hill-climb prong 2): with the pool
	 * mapped, lazy_spine_load copies the spine device-to-device from the
	 * daemon's verified arena image instead of re-reading the pack file.
	 * Destroy-on-failure already covers a map created before a later
	 * spine fault (SparkWeightdLazyPackCreateChecked destroys the pack). */
	{
		int epoch_fd = -1;
		(void)SparkWeightdClientEpochExport(pack->client,
		    pack->attached.arena_generation,&epoch_fd,timeout);
		status = SparkWeightdMapCreate(pack->client,
		    &pack->attached,epoch_fd,
		    pack->attached.pool_fd,&pack->map);
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = lazy_spine_load(pack,fd,request,budget);
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

SparkStatus SparkWeightdLazyPackCreateChecked(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t spine_budget,uint64_t timeout,SparkWeightdManifestCheck check,void *context,SparkWeightdLazyPack **out)
{
	SparkWeightdLazyPack *pack;
	SparkWeightdIdentity identity;
	SparkWeightdLazyAttachRequest resolved;
	SparkStatus status;
	struct stat info;
	int32_t fd;
	char absolute[4096];
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( socket == 0 || request == 0 || request->expert_pool_bytes == 0u || request->pack_path[0] == 0 || memchr(request->pack_path,0,sizeof(request->pack_path)) == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
	if ( SparkWeightdIdentityPrepare(&identity) != SPARK_STATUS_OK )
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
	status = lazy_pack_initialize(pack,fd,socket,request,spine_budget,timeout,check,context);
	if ( close(fd) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK && SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK )
		pack = 0;
	if ( status == SPARK_STATUS_OK )
		pack->ready = 1u;
	*out = pack;
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdLazyPackCreate(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t spine_budget,uint64_t timeout,SparkWeightdLazyPack **out)
{
	return(SparkWeightdLazyPackCreateChecked(socket,request,spine_budget,timeout,0,0,out));
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
		*pointer = ((uint8_t *)pack->spine + compact);
	SPARK_RETURN(status);
}
