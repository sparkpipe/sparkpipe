#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static uint64_t probe_now_ns(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0ull);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static int probe_copy(char *destination,size_t capacity,const char *source)
{
	size_t bytes = strlen(source) + 1u;
	if ( bytes > capacity )
		return(-1);
	memcpy(destination,source,bytes);
	return(0);
}

int main(int argc,char **argv)
{
	static const uint64_t timeout_ns = 300000000000ull;
	static const uint32_t probe_read_bytes = 64u;
	SparkWeightdLazyAttachRequest request;
	SparkWeightdDetachResult detach;
	SparkWeightdExpertKey keys[8];
	SparkWeightdMap *map = 0;
	SparkWeightdLazyPack *pack = 0;
	SparkWeightdManifest manifest;
	const SparkWeightdRangeGroup *group;
	const SparkWeightdRange *range;
	char manifest_path[SPARK_WEIGHTD_PATH_BYTES + 8];
	void *address = 0;
	uint8_t sample[64];
	uint64_t pool_bytes,waves,wave,lease,total_ns,start_ns;
	uint32_t per_wave,i,key_count;
	int status;

	if ( argc != 7 )
	{
		fprintf(stderr,"usage: weightd_execute_probe pack model revision pool-mib waves pack-sha256-hex\n");
		return(2);
	}
	if ( SparkSha256HexIsValid(argv[6]) == 0 )
	{
		fprintf(stderr,"probe-20 pack sha invalid\n");
		return(2);
	}
	if ( getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET) == 0 )
	{
		fprintf(stderr,"probe-1 socket env unset\n");
		return(2);
	}
	pool_bytes = strtoull(argv[4],0,10) << 20;
	waves = strtoull(argv[5],0,10);
	if ( pool_bytes == 0ull || waves == 0ull )
		return(2);
	if ( probe_copy(manifest_path,sizeof(manifest_path),argv[1]) != 0 ||
		strlen(manifest_path) + 8u > sizeof(manifest_path) )
	{
		fprintf(stderr,"probe-2 pack path too long\n");
		return(2);
	}
	strcat(manifest_path,".experts");
	if ( SparkWeightdManifestLoad(manifest_path,probe_file_bytes(argv[1]),&manifest) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"probe-4 manifest load failed\n");
		return(1);
	}
	if ( manifest.group_count == 0u )
	{
		fprintf(stderr,"probe-5 manifest empty\n");
		return(1);
	}
	memset(&request,0,sizeof(request));
	if ( probe_copy(request.identity.model,SPARK_WEIGHTD_ID_BYTES,argv[2]) != 0 ||
		probe_copy(request.identity.revision,SPARK_WEIGHTD_REVISION_BYTES,argv[3]) != 0 ||
		probe_copy(request.pack_path,SPARK_WEIGHTD_PATH_BYTES,argv[1]) != 0 )
	{
		fprintf(stderr,"probe-6 identity too long\n");
		return(2);
	}
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.topology = 16u;
	request.identity.geometry_fingerprint = 0x504F4355ull;
	request.identity.arena_bytes = probe_file_bytes(argv[1]);
	memcpy(request.identity.pack_sha256,argv[6],SPARK_SHA256_HEX_BYTES);
	request.expert_pool_bytes = pool_bytes;
	if ( request.identity.arena_bytes == 0ull ||
		SparkWeightdIdentityPrepare(&request.identity) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"probe-7 identity invalid\n");
		return(1);
	}
	if ( SparkWeightdLazyPackCreate(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,request.identity.arena_bytes + 255ull,timeout_ns,&pack) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"probe-8 attach failed\n");
		return(1);
	}
	if ( pack->attached.status != SPARK_STATUS_OK || pack->map == 0 )
	{
		fprintf(stderr,"probe-9 attach status %d\n",(int)pack->attached.status);
		return(1);
	}
	map = pack->map;
	per_wave = manifest.group_count < 8u ? manifest.group_count : 8u;
	total_ns = 0ull;
	for (wave = 0ull; wave < waves; wave++)
	{
		key_count = 0u;
		for (i = 0u; i < per_wave; i++)
		{
			uint64_t ordinal = wave * (uint64_t)per_wave + (uint64_t)i;
			group = &manifest.groups[ordinal % manifest.group_count];
			keys[key_count].layer = group->layer;
			keys[key_count].expert = group->expert;
			key_count++;
		}
		start_ns = probe_now_ns();
		lease = 0ull;
		status = SparkWeightdMapAcquire(map,keys,key_count,&lease,timeout_ns);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"probe-10 wave %llu acquire %d\n",(unsigned long long)wave,(int)status);
			return(1);
		}
		status = SparkWeightdMapBeginUse(map,lease,&address);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"probe-11 wave %llu begin %d\n",(unsigned long long)wave,(int)status);
			return(1);
		}
		for (i = 0u; i < key_count; i++)
		{
			group = SparkWeightdManifestFind(&manifest,keys[i].layer,keys[i].expert);
			if ( group == 0 || group->range_count == 0u )
			{
				fprintf(stderr,"probe-12 wave %llu group missing\n",(unsigned long long)wave);
				return(1);
			}
			range = &manifest.ranges[group->first_range];
			if ( range->offset > request.identity.arena_bytes ||
				range->bytes < probe_read_bytes )
				continue;
			if ( cudaMemcpy(sample,(uint8_t *)address + range->offset,probe_read_bytes,cudaMemcpyDeviceToHost) != cudaSuccess )
			{
				fprintf(stderr,"probe-13 wave %llu read failed\n",(unsigned long long)wave);
				return(1);
			}
			if ( sample[0] == 0u && sample[1] == 0u && sample[2] == 0u && sample[3] == 0u )
			{
				fprintf(stderr,"probe-19 wave %llu zero sample\n",(unsigned long long)wave);
				return(1);
			}
		}
		if ( SparkWeightdMapRecordCompletion(map,lease,0) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"probe-14 wave %llu record failed\n",(unsigned long long)wave);
			return(1);
		}
		if ( cudaStreamSynchronize(0) != cudaSuccess )
		{
			fprintf(stderr,"probe-15 wave %llu sync failed\n",(unsigned long long)wave);
			return(1);
		}
		if ( SparkWeightdMapRelease(map,lease,timeout_ns) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"probe-16 wave %llu release failed\n",(unsigned long long)wave);
			return(1);
		}
		total_ns += probe_now_ns() - start_ns;
	}
	memset(&detach,0,sizeof(detach));
	if ( SparkWeightdClientDetach(pack->client,pack->attached.arena_generation,&detach,timeout_ns) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"probe-17 detach failed %d\n",(int)detach.status);
		return(1);
	}
	if ( SparkWeightdLazyPackDestroy(pack) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"probe-18 destroy failed\n");
		return(1);
	}
	printf("EXECUTE-PROBE waves=%llu keys=%u experts=%u waves_ns=%llu pool_mib=%llu\n",
		(unsigned long long)waves,key_count,manifest.group_count,
		(unsigned long long)total_ns,(unsigned long long)(pool_bytes >> 20));
	SparkWeightdManifestDestroy(&manifest);
	return(0);
}
