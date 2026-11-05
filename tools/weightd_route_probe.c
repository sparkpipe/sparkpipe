#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_weightd_lease.h"
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROBE_MAX_RANGES 64u
#define PROBE_READ_BYTES (8u * 1024u * 1024u)

static uint64_t probe_file_bytes(const char *path)
{
	FILE *file;
	uint64_t bytes = 0ull;
	file = fopen(path,"rb");
	if ( file == 0 )
		return(0ull);
	if ( fseeko(file,0,SEEK_END) == 0 )
		bytes = (uint64_t)ftello(file);
	(void)fclose(file);
	return(bytes);
}

static int probe_copy(char *destination,size_t capacity,const char *source)
{
	size_t bytes = strlen(source) + 1u;
	if ( bytes > capacity )
		return(-1);
	memcpy(destination,source,bytes);
	return(0);
}

static int probe_verify_range(int32_t fd,const uint8_t *device_base,const SparkWeightdRange *range,uint8_t *scratch)
{
	uint64_t done = 0ull;
	ssize_t moved;
	if ( range->bytes > PROBE_READ_BYTES )
		return(-1);
	while ( done < range->bytes )
	{
		moved = pread(fd,scratch + done,(size_t)(range->bytes - done),(off_t)(range->offset + done));
		if ( moved <= 0 )
			return(-1);
		done += (uint64_t)moved;
	}
	if ( cudaMemcpy(scratch + range->bytes,(uint8_t *)(uintptr_t)device_base + range->offset,range->bytes,cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-1);
	return(memcmp(scratch,scratch + range->bytes,(size_t)range->bytes) == 0 ? 0 : 1);
}

int main(int argc,char **argv)
{
	static const uint64_t timeout_ns = 300000000000ull;
	SparkWeightdLazyAttachRequest request;
	SparkWeightdDetachResult detach;
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	SparkWeightdMap *map = 0;
	SparkWeightdLazyPack *pack = 0;
	SparkWeightdClient *raw_client = 0;
	SparkWeightdLazyAttachResult raw_attached = {0};
	SparkWeightdWorkingSetResult result;
	uint64_t raw_generation = 0ull;
	SparkWeightdManifest manifest;
	SparkWeightdPoolPlan pool;
	const SparkWeightdRangeGroup *group;
	const SparkWeightdRange *range;
	const uint8_t *address = 0;
	char manifest_path[SPARK_WEIGHTD_PATH_BYTES + 8];
	uint8_t *scratch = 0;
	uint32_t offsets[SPARK_WEIGHTD_LEASE_GROUPS_MAX + 1u];
	uint64_t pool_bytes,waves,wave,lease,pack_bytes;
	uint32_t per_wave = 0u,i,key_count,route_count = 0u,failures = 0u;
	int32_t fd = -1;
	int status;
	if ( argc != 11 )
	{
		fprintf(stderr,"usage: weightd_route_probe pack model revision cap-mib waves mode layer base resident sha (cap-mib 0 sizes the pool from the pack)\n");
		return(2);
	}
	if ( SparkSha256HexIsValid(argv[10]) == 0 )
	{
		fprintf(stderr,"route-probe-1 pack sha invalid\n");
		return(2);
	}
	if ( getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET) == 0 )
	{
		fprintf(stderr,"route-probe-2 socket env unset\n");
		return(2);
	}
	pool_bytes = strtoull(argv[4],0,10) << 20;
	waves = strtoull(argv[5],0,10);
	if ( waves == 0ull || argv[4][0] == '\0' || argv[4][strspn(argv[4],"0123456789")] != '\0' || strlen(argv[4]) > 12u )
		return(2);
	if ( probe_copy(manifest_path,sizeof(manifest_path),argv[1]) != 0 ||
		strlen(manifest_path) + 8u > sizeof(manifest_path) )
	{
		fprintf(stderr,"route-probe-3 pack path too long\n");
		return(2);
	}
	strcat(manifest_path,".experts");
	pack_bytes = probe_file_bytes(argv[1]);
	if ( pack_bytes == 0ull )
	{
		fprintf(stderr,"route-probe-4 pack missing\n");
		return(1);
	}
	if ( SparkWeightdManifestLoad(manifest_path,pack_bytes,&manifest) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"route-probe-5 manifest load failed\n");
		return(1);
	}
	memset(&request,0,sizeof(request));
	if ( probe_copy(request.identity.model,SPARK_WEIGHTD_ID_BYTES,argv[2]) != 0 ||
		probe_copy(request.identity.revision,SPARK_WEIGHTD_REVISION_BYTES,argv[3]) != 0 ||
		probe_copy(request.pack_path,SPARK_WEIGHTD_PATH_BYTES,argv[1]) != 0 )
	{
		fprintf(stderr,"route-probe-6 identity too long\n");
		return(2);
	}
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.topology = 16u;
	request.identity.geometry_fingerprint = 0x52505455ull;
	request.identity.arena_bytes = pack_bytes;
	memcpy(request.identity.pack_sha256,argv[10],SPARK_SHA256_HEX_BYTES);
	if ( SparkWeightdIdentityPrepare(&request.identity) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"route-probe-7 identity invalid\n");
		return(1);
	}
	scratch = malloc(PROBE_READ_BYTES * 2u);
	if ( scratch == 0 )
	{
		fprintf(stderr,"route-probe-10 scratch alloc failed\n");
		return(1);
	}
	route_count = (uint32_t)strtoul(argv[9],0,10);
	if ( route_count == 0u || route_count > SPARK_WEIGHTD_LEASE_GROUPS_MAX )
	{
		fprintf(stderr,"route-probe-11 resident invalid\n");
		return(2);
	}
	for (i = 0u; i <= route_count; i++)
		offsets[i] = i;
	per_wave = manifest.group_count < 8u ? manifest.group_count : 8u;
	fd = open(argv[1],O_RDONLY);
	if ( fd < 0 )
	{
		fprintf(stderr,"route-probe-12 pack open failed\n");
		return(1);
	}
	if ( strcmp(argv[6],"rawroute") == 0 )
	{
		if ( SparkWeightdClientConnect(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&raw_client,0) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-8 connect failed\n");
			return(1);
		}
		if ( SparkWeightdManifestPoolPlan(&manifest,pack_bytes,pool_bytes,&pool) != SPARK_STATUS_OK )
			return(2);
		request.expert_pool_bytes = pool.request_bytes;
		if ( SparkWeightdClientAttachLazy(raw_client,&request,&raw_attached,timeout_ns) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-8 raw attach failed\n");
			return(1);
		}
		if ( raw_attached.status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-9 raw attach status %d\n",(int)raw_attached.status);
			return(1);
		}
		raw_generation = raw_attached.arena_generation;
		goto raw_waves;
	}
	if ( SparkWeightdLazyPackCreate(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,pool_bytes,request.identity.arena_bytes + 255ull,timeout_ns,&pack) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"route-probe-8 attach failed\n");
		return(1);
	}
	if ( pack->attached.status != SPARK_STATUS_OK || pack->map == 0 )
	{
		fprintf(stderr,"route-probe-9 attach status %d\n",(int)pack->attached.status);
		return(1);
	}
	map = pack->map;
	raw_client = 0;
raw_waves:
	for (wave = 0ull; wave < waves; wave++)
	{
		key_count = 0u;
		if ( strcmp(argv[6],"route") == 0 || strcmp(argv[6],"rawroute") == 0 )
		{
			uint32_t layer = (uint32_t)strtoul(argv[7],0,10);
			uint32_t base = (uint32_t)strtoul(argv[8],0,10);
			if ( SparkWeightdRouteKeys(layer,offsets,route_count,route_count,keys,route_count,&key_count) != SPARK_STATUS_OK )
			{
				fprintf(stderr,"route-probe-13 route keys failed\n");
				return(1);
			}
			for (i = 0u; i < key_count; i++)
				keys[i].expert += base;
		}
		else
		{
			for (i = 0u; i < per_wave; i++)
			{
				uint64_t ordinal = wave * (uint64_t)per_wave + (uint64_t)i;
				group = &manifest.groups[ordinal % manifest.group_count];
				keys[key_count].layer = group->layer;
				keys[key_count].expert = group->expert;
				key_count++;
			}
		}
			lease = 0ull;
		if ( raw_client != 0 )
			status = SparkWeightdClientAcquire(raw_client,raw_generation,keys,key_count,&result,timeout_ns);
		else
			status = SparkWeightdMapAcquire(map,keys,key_count,&lease,timeout_ns);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-14 wave %llu acquire %d keys=%u\n",(unsigned long long)wave,status,key_count);
			return(1);
		}
		if ( raw_client == 0 && SparkWeightdMapBeginUse(map,lease,(void **)&address) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-15 wave %llu begin failed\n",(unsigned long long)wave);
			return(1);
		}
		if ( raw_client != 0 )
		{
			if ( result.status != SPARK_STATUS_OK || result.lease_identifier == 0ull )
			{
				fprintf(stderr,"route-probe-15 wave %llu raw acquire status %d\n",(unsigned long long)wave,(int)result.status);
				return(1);
			}
			if ( SparkWeightdClientRelease(raw_client,raw_generation,result.lease_identifier,&result,timeout_ns) != SPARK_STATUS_OK )
			{
				fprintf(stderr,"route-probe-16 wave %llu raw release failed\n",(unsigned long long)wave);
				return(1);
			}
			continue;
		}
		for (i = 0u; i < key_count && failures == 0u && raw_client == 0; i++)
		{
			group = SparkWeightdManifestFind(&manifest,keys[i].layer,keys[i].expert);
			if ( group == 0 )
			{
				fprintf(stderr,"route-probe-16 wave %llu key %u missing\n",(unsigned long long)wave,i);
				return(1);
			}
			{
				uint32_t j;
				for (j = 0u; j < group->range_count && j < PROBE_MAX_RANGES; j++)
				{
					range = &manifest.ranges[group->first_range + j];
					status = probe_verify_range(fd,address,range,scratch);
					if ( status != 0 )
					{
						fprintf(stderr,"route-probe-17 wave %llu key %u range %u mismatch %d\n",(unsigned long long)wave,i,j,status);
						failures = 1u;
						break;
					}
				}
			}
		}
		if ( SparkWeightdMapRecordCompletion(map,lease,0) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-18 wave %llu record failed\n",(unsigned long long)wave);
			return(1);
		}
		if ( cudaStreamSynchronize(0) != cudaSuccess )
		{
			fprintf(stderr,"route-probe-19 wave %llu sync failed\n",(unsigned long long)wave);
			return(1);
		}
		if ( SparkWeightdMapRelease(map,lease,timeout_ns) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-20 wave %llu release failed\n",(unsigned long long)wave);
			return(1);
		}
	}
	(void)close(fd);
	free(scratch);
	if ( raw_client != 0 )
	{
		SparkWeightdDetachResult raw_detach;
		memset(&raw_detach,0,sizeof(raw_detach));
		if ( SparkWeightdClientDetach(raw_client,raw_generation,&raw_detach,timeout_ns) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"route-probe-21 raw detach failed %d\n",(int)raw_detach.status);
			return(1);
		}
		SparkWeightdClientClose(raw_client);
		printf("ROUTE-PROBE waves=%llu mode=%s keys=%u resident=%u checksums=raw-ok\n",
			(unsigned long long)waves,argv[6],key_count,route_count);
		SparkWeightdManifestDestroy(&manifest);
		return(0);
	}
	memset(&detach,0,sizeof(detach));
	if ( SparkWeightdClientDetach(pack->client,pack->attached.arena_generation,&detach,timeout_ns) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"route-probe-21 detach failed %d\n",(int)detach.status);
		return(1);
	}
	if ( SparkWeightdLazyPackDestroy(pack) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"route-probe-22 destroy failed\n");
		return(1);
	}
	SparkWeightdManifestDestroy(&manifest);
	printf("ROUTE-PROBE waves=%llu mode=%s keys=%u resident=%u checksums=%s\n",
		(unsigned long long)waves,argv[6],key_count,route_count,failures == 0u ? "green" : "MISMATCH");
	return(failures == 0u ? 0 : 1);
}
