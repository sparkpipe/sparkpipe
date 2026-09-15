#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_ck128.h"

#define SMOKE_TIMEOUT UINT64_C(60000000000)
#define SMOKE_GROUPS_MAX 65536u
#define SMOKE_RANGE_MAX_BYTES (4u * 1024u * 1024u)
#define SMOKE_ROUND_KEYS_MAX 512u
#define SMOKE_MISS_DEFAULT_US 10000.0

typedef struct SmokeGroups
{
	uint32_t count;
	uint32_t layer[SMOKE_GROUPS_MAX];
	uint32_t expert[SMOKE_GROUPS_MAX];
} SmokeGroups;

typedef struct SmokeStats
{
	double *latency_us;
	uint32_t rounds;
	uint32_t misses;
	uint32_t verify_fails;
} SmokeStats;

static uint64_t smoke_now_ns(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return 0ull;
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void smoke_emit(const char *kind,const char *payload)
{
	printf("SMOKE %s %s\n",kind,payload);
	fflush(stdout);
}

static int compare_double(const void *left,const void *right)
{
	double a = *(const double *)left;
	double b = *(const double *)right;
	return a < b ? -1 : a > b ? 1 : 0;
}

static double smoke_percentile(double *values,uint32_t count,double fraction)
{
	if ( count == 0u )
		return 0.0;
	qsort(values,count,sizeof(double),compare_double);
	return values[(uint32_t)(fraction * (double)(count - 1u))];
}

static int smoke_parse_u32(const char *text,uint32_t *out)
{
	char *end = 0;
	unsigned long wide;
	if ( text == 0 || *text == '\0' )
		return -1;
	wide = strtoul(text,&end,10);
	if ( end == text || *end != '\0' || wide > 0xfffffffful )
		return -2;
	*out = (uint32_t)wide;
	return 0;
}

static int smoke_parse_u64(const char *text,uint64_t *out)
{
	char *end = 0;
	if ( text == 0 || *text == '\0' )
		return -3;
	*out = strtoull(text,&end,10);
	return (end == text || *end != '\0') ? -4 : 0;
}

typedef struct SmokeGeometry
{
	const char *path;
	uint32_t layers;
	uint32_t experts;
	uint32_t kinds;
	uint32_t range_bytes;
	uint64_t gap;
	uint32_t seed;
} SmokeGeometry;

static int smoke_makepack_emit(FILE *pack,FILE *manifest,const SmokeGeometry *geometry)
{
	uint8_t header[16];
	uint8_t buffer[65536];
	uint8_t record[48];
	uint8_t zero[65536];
	uint32_t magic = SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC;
	uint32_t version = SPARK_WEIGHTD_RANGE_MANIFEST_VERSION;
	uint32_t reserved = 0u;
	uint32_t total_ranges = geometry->layers * geometry->experts * geometry->kinds;
	uint32_t layer,expert,kind;
	uint64_t cursor = geometry->gap;
	uint64_t gap = geometry->gap;
	memcpy(header,&magic,4u);
	memcpy(header + 4u,&version,4u);
	memcpy(header + 8u,&total_ranges,4u);
	memcpy(header + 12u,&reserved,4u);
	if ( fwrite(header,1u,sizeof(header),manifest) != sizeof(header) )
		return -12;
	memset(zero,0,sizeof(zero));
	while ( gap > 0u )
	{
		uint64_t chunk = gap > sizeof(zero) ? sizeof(zero) : gap;
		if ( fwrite(zero,1u,chunk,pack) != chunk )
			return -12;
		gap -= chunk;
	}
	for ( layer = 0u; layer < geometry->layers; layer++ )
		for ( expert = 0u; expert < geometry->experts; expert++ )
			for ( kind = 0u; kind < geometry->kinds; kind++ )
			{
				uint8_t pattern = (uint8_t)(geometry->seed + layer * 7u + expert * 13u +
					kind * 29u);
				SparkCk128Context hash;
				uint64_t length = geometry->range_bytes;
				uint32_t written = 0u;
				SparkCk128Initialize(&hash);
				while ( written < geometry->range_bytes )
				{
					uint32_t chunk = geometry->range_bytes - written;
					uint32_t i;
					if ( chunk > sizeof(buffer) )
						chunk = sizeof(buffer);
					for ( i = 0u; i < chunk; i++ )
						buffer[i] = (uint8_t)(pattern + (uint8_t)i);
					if ( fwrite(buffer,1u,chunk,pack) != chunk )
						return -13;
					SparkCk128Update(&hash,buffer,chunk);
					written += chunk;
				}
				memset(record,0,sizeof(record));
				memcpy(record,&layer,4u);
				memcpy(record + 4u,&expert,4u);
				memcpy(record + 8u,&kind,4u);
				memcpy(record + 16u,&cursor,8u);
				memcpy(record + 24u,&length,8u);
				SparkCk128Finalize(&hash,record + 32u);
				if ( fwrite(record,1u,sizeof(record),manifest) != sizeof(record) )
					return -14;
				cursor += length;
			}
	return 0;
}

static int smoke_makepack(const char *path,uint32_t layers,uint32_t experts,uint32_t kinds,
	uint32_t range_bytes,uint64_t gap,uint32_t seed)
{
	SmokeGeometry geometry = {path,layers,experts,kinds,range_bytes,gap,seed};
	char sidecar[512];
	FILE *pack;
	FILE *manifest;
	int state;
	if ( snprintf(sidecar,sizeof(sidecar),"%s.experts",path) >= (int)sizeof(sidecar) )
		return -10;
	pack = fopen(path,"wbx");
	manifest = fopen(sidecar,"wbx");
	if ( pack == 0 || manifest == 0 )
	{
		if ( pack != 0 )
			fclose(pack);
		if ( manifest != 0 )
			fclose(manifest);
		(void)unlink(path);
		(void)unlink(sidecar);
		return -11;
	}
	state = smoke_makepack_emit(pack,manifest,&geometry);
	if ( fclose(pack) != 0 )
		state = state == 0 ? -15 : state;
	if ( fclose(manifest) != 0 )
		state = state == 0 ? -16 : state;
	if ( state != 0 )
	{
		(void)unlink(path);
		(void)unlink(sidecar);
		return state;
	}
	{
		char payload[320];
		snprintf(payload,sizeof(payload),
			"{\"op\":\"makepack\",\"pack\":\"%s\",\"layers\":%u,\"experts\":%u,\"kinds\":%u,"
			"\"range_bytes\":%u,\"spine_gap\":%llu,\"pack_bytes\":%llu,\"ranges\":%u}",
			path,layers,experts,kinds,range_bytes,(unsigned long long)gap,
			(unsigned long long)(gap + (uint64_t)layers * experts * kinds * range_bytes),
			layers * experts * kinds);
		smoke_emit("MAKEPACK",payload);
	}
	return 0;
}

static void smoke_groups_build(const SparkWeightdManifest *manifest,SmokeGroups *groups)
{
	uint32_t i;
	groups->count = 0u;
	for ( i = 0u; i < manifest->group_count && groups->count < SMOKE_GROUPS_MAX; i++ )
	{
		groups->layer[groups->count] = manifest->groups[i].layer;
		groups->expert[groups->count] = manifest->groups[i].expert;
		groups->count++;
	}
}

static int smoke_verify_range(cudaStream_t stream,const void *base,
	const SparkWeightdManifest *manifest,uint32_t layer,uint32_t expert,uint8_t *buffer)
{
	const SparkWeightdRangeGroup *group = SparkWeightdManifestFind(manifest,layer,expert);
	const SparkWeightdRange *range;
	SparkCk128Context hash;
	uint8_t digest[16];
	if ( group == 0 || group->range_count == 0u )
		return -20;
	range = &manifest->ranges[group->first_range];
	if ( cudaMemcpyAsync(buffer,(const uint8_t *)base + range->offset,range->bytes,
		cudaMemcpyDeviceToHost,stream) != cudaSuccess )
		return -21;
	if ( cudaStreamSynchronize(stream) != cudaSuccess )
		return -22;
	SparkCk128Initialize(&hash);
	SparkCk128Update(&hash,buffer,range->bytes);
	SparkCk128Finalize(&hash,digest);
	return memcmp(digest,range->digest,sizeof(digest)) == 0 ? 0 : -23;
}

static int smoke_dev_round(SparkWeightdLazyPack *pack,const void *base,
	const SparkWeightdManifest *manifest,const SmokeGroups *groups,uint32_t cursor,
	uint32_t keys,cudaStream_t stream,uint8_t *buffer,double *latency_us,
	uint32_t *verified_bytes)
{
	SparkWeightdExpertKey picked[SMOKE_ROUND_KEYS_MAX];
	uint64_t lease = 0u;
	void *address = 0;
	uint64_t started;
	SparkStatus status;
	uint32_t i;
	int state = 0;
	for ( i = 0u; i < keys; i++ )
	{
		uint32_t slot = (cursor + i) % groups->count;
		picked[i].layer = groups->layer[slot];
		picked[i].expert = groups->expert[slot];
	}
	started = smoke_now_ns();
	status = SparkWeightdMapAcquire(pack->map,picked,keys,&lease,SMOKE_TIMEOUT);
	if ( status != SPARK_STATUS_OK )
		return -24;
	status = SparkWeightdMapBeginUse(pack->map,lease,&address);
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkWeightdMapRelease(pack->map,lease,SMOKE_TIMEOUT);
		return -25;
	}
	*verified_bytes = 0u;
	for ( i = 0u; i < keys && state == 0; i++ )
	{
		uint32_t slot = (cursor + i) % groups->count;
		state = smoke_verify_range(stream,address,manifest,groups->layer[slot],
			groups->expert[slot],buffer);
		if ( state == 0 )
		{
			const SparkWeightdRangeGroup *group = SparkWeightdManifestFind(manifest,
				groups->layer[slot],groups->expert[slot]);
			*verified_bytes += manifest->ranges[group->first_range].bytes;
		}
	}
	if ( SparkWeightdMapRecordCompletion(pack->map,lease,stream) != SPARK_STATUS_OK && state == 0 )
		state = -26;
	if ( SparkWeightdMapRelease(pack->map,lease,SMOKE_TIMEOUT) != SPARK_STATUS_OK && state == 0 )
		state = -27;
	*latency_us = (double)(smoke_now_ns() - started) / 1000.0;
	(void)base;
	return state;
}

static int smoke_dev_pin(SparkWeightdLazyPack *pack,const SmokeGroups *groups,
	uint32_t pins,uint32_t hold_seconds,uint64_t *lease_out)
{
	SparkWeightdExpertKey picked[SMOKE_ROUND_KEYS_MAX];
	uint64_t lease = 0u;
	SparkStatus status;
	uint32_t i;
	if ( pins == 0u || pins > SMOKE_ROUND_KEYS_MAX || pins > groups->count )
		return 0;
	for ( i = 0u; i < pins; i++ )
	{
		picked[i].layer = groups->layer[i];
		picked[i].expert = groups->expert[i];
	}
	status = SparkWeightdMapAcquire(pack->map,picked,pins,&lease,SMOKE_TIMEOUT);
	if ( status != SPARK_STATUS_OK )
		return -28;
	*lease_out = lease;
	{
		char payload[160];
		snprintf(payload,sizeof(payload),"{\"op\":\"pin\",\"groups\":%u,\"lease\":%llu,"
			"\"hold_seconds\":%u}",pins,(unsigned long long)lease,hold_seconds);
		smoke_emit("DEV",payload);
	}
	return 0;
}

static void smoke_dev_summary(const char *name,uint32_t lane,const SmokeStats *stats)
{
	double minimum = 0.0,maximum = 0.0,p50,p99;
	char payload[384];
	uint32_t i;
	if ( stats->rounds > 0u )
	{
		minimum = stats->latency_us[0];
		maximum = stats->latency_us[0];
		for ( i = 1u; i < stats->rounds; i++ )
		{
			if ( stats->latency_us[i] < minimum )
				minimum = stats->latency_us[i];
			if ( stats->latency_us[i] > maximum )
				maximum = stats->latency_us[i];
		}
	}
	p50 = smoke_percentile(stats->latency_us,stats->rounds,0.50);
	p99 = smoke_percentile(stats->latency_us,stats->rounds,0.99);
	snprintf(payload,sizeof(payload),
		"{\"dev\":\"%s\",\"lane\":%u,\"rounds_done\":%u,\"p50_us\":%.1f,\"p99_us\":%.1f,"
		"\"min_us\":%.1f,\"max_us\":%.1f,\"misses\":%u,\"verify_fails\":%u}",
		name,lane,stats->rounds,p50,p99,minimum,maximum,stats->misses,stats->verify_fails);
	smoke_emit("DEV-SUMMARY",payload);
}

static int smoke_dev(const char *socket,const char *pack_path,const char *name,
	uint32_t rounds,uint32_t keys,uint64_t pool_bytes,uint32_t pins,uint32_t hold_seconds)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdClient *client = 0;
	SparkWeightdHelloResult hello;
	SparkWeightdLazyPack *pack = 0;
	SparkWeightdManifest *manifest;
	SmokeGroups groups;
	SmokeStats stats = {0};
	struct stat info;
	uint64_t held_lease = 0u;
	uint32_t lane = 0xffffffffu,cursor = 0u;
	cudaStream_t stream = 0;
	uint8_t *verify_buffer = 0;
	char payload[512];
	uint32_t i;
	int state;
	if ( stat(pack_path,&info) != 0 || info.st_size <= 0 )
		return -30;
	if ( cudaFree(0) != cudaSuccess )
		return -31;
	if ( SparkWeightdClientConnect(socket,&client,&hello) != SPARK_STATUS_OK )
		return -32;
	if ( SparkWeightdClientLaneAcquire(client,&lane,SMOKE_TIMEOUT) != SPARK_STATUS_OK )
	{
		SparkWeightdClientClose(client);
		return -33;
	}
	snprintf(payload,sizeof(payload),
		"{\"op\":\"dev-init\",\"dev\":\"%s\",\"pid\":%d,\"lane\":%u,\"pack\":\"%s\",\"pack_bytes\":%llu,"
		"\"rounds\":%u,\"keys\":%u,\"pool_bytes\":%llu,\"pins\":%u}",
		name,(int)getpid(),lane,pack_path,(unsigned long long)info.st_size,rounds,keys,
		(unsigned long long)pool_bytes,pins);
	smoke_emit("DEV",payload);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (uint64_t)info.st_size;
	snprintf(request.identity.model,sizeof(request.identity.model),"%s",name);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",pack_path);
	request.expert_pool_bytes = pool_bytes;
	if ( SparkSha256File(pack_path,request.identity.pack_sha256) != SPARK_STATUS_OK )
	{
		SparkWeightdClientClose(client);
		return -34;
	}
	if ( SparkWeightdLazyPackCreate(socket,&request,(uint64_t)info.st_size + 255u,
		SMOKE_TIMEOUT,&pack) != SPARK_STATUS_OK )
	{
		SparkWeightdClientClose(client);
		return -35;
	}
	manifest = &pack->manifest;
	smoke_groups_build(manifest,&groups);
	snprintf(payload,sizeof(payload),
		"{\"op\":\"dev-attach\",\"dev\":\"%s\",\"lane\":%u,\"groups\":%u,\"spine_bytes\":%llu,"
		"\"resident_bytes\":%llu,\"arena_bytes\":%llu,\"chunk_bytes\":%llu,\"chunk_count\":%u}",
		name,lane,groups.count,(unsigned long long)manifest->spine_bytes,
		(unsigned long long)pack->attached.resident_bytes,
		(unsigned long long)pack->attached.arena_bytes,
		(unsigned long long)pack->attached.chunk_bytes,pack->attached.chunk_count);
	smoke_emit("DEV",payload);
	{
		void *base = 0;
		uint32_t largest = 0u;
		state = 0;
		if ( SparkWeightdMapBase(pack->map,&base) != SPARK_STATUS_OK || base == 0 )
			state = -36;
		for ( i = 0u; state == 0 && i < groups.count; i++ )
		{
			const SparkWeightdRangeGroup *group = SparkWeightdManifestFind(manifest,
				groups.layer[i],groups.expert[i]);
			if ( group != 0 && group->range_count > 0u &&
				manifest->ranges[group->first_range].bytes > largest )
				largest = manifest->ranges[group->first_range].bytes;
		}
		if ( state == 0 )
		{
			verify_buffer = malloc(largest);
			stats.latency_us = malloc(sizeof(double) * rounds);
			if ( verify_buffer == 0 || stats.latency_us == 0 )
				state = -37;
		}
		if ( state == 0 )
			state = smoke_dev_pin(pack,&groups,pins,hold_seconds,&held_lease);
		for ( i = 0u; i < rounds && state == 0; i++ )
		{
			double latency = 0.0;
			uint32_t verified = 0u;
			state = smoke_dev_round(pack,base,manifest,&groups,cursor,keys,stream,
				verify_buffer,&latency,&verified);
			cursor = (cursor + keys) % groups.count;
			if ( state != 0 )
			{
				stats.verify_fails++;
				snprintf(payload,sizeof(payload),"{\"dev\":\"%s\",\"ordinal\":%u,\"error\":%d}",
					name,i,state);
				smoke_emit("DEV-FAIL",payload);
				break;
			}
			stats.latency_us[stats.rounds] = latency;
			stats.rounds++;
			if ( latency > SMOKE_MISS_DEFAULT_US )
				stats.misses++;
			snprintf(payload,sizeof(payload),
				"{\"dev\":\"%s\",\"ordinal\":%u,\"latency_us\":%.1f,\"keys\":%u,\"bytes\":%u,"
				"\"miss\":%s}",name,i,latency,keys,verified,
				latency > SMOKE_MISS_DEFAULT_US ? "true" : "false");
			smoke_emit("ROUND",payload);
		}
	}
	smoke_dev_summary(name,lane,&stats);
	free(stats.latency_us);
	free(verify_buffer);
	if ( held_lease != 0u )
		(void)SparkWeightdMapRelease(pack->map,held_lease,SMOKE_TIMEOUT);
	state = SparkWeightdLazyPackDestroy(pack);
	if ( state != SPARK_STATUS_OK )
	{
		snprintf(payload,sizeof(payload),
			"{\"dev\":\"%s\",\"op\":\"teardown\",\"lazy_pack_destroy\":\"%s\"}",name,
			SparkStatusToString(state));
		smoke_emit("DEV-TEARDOWN",payload);
		SparkWeightdClientClose(client);
		return -38;
	}
	SparkWeightdClientClose(client);
	return 0;
}

static int smoke_bandwidth(uint32_t megabytes,uint32_t iterations)
{
	uint8_t *host = 0;
	uint8_t *device = 0;
	uint64_t bytes,started,elapsed;
	double gib_per_second;
	char payload[192];
	uint32_t iter;
	bytes = (uint64_t)megabytes * 1024ull * 1024ull;
	if ( cudaMalloc((void **)&device,bytes) != cudaSuccess )
		return -40;
	if ( cudaHostAlloc((void **)&host,bytes,cudaHostAllocDefault) != cudaSuccess )
	{
		cudaFree(device);
		return -41;
	}
	memset(host,0xa5,(size_t)bytes);
	for ( iter = 0u; iter < 3u; iter++ )
		(void)cudaMemcpy(host,device,bytes,cudaMemcpyDeviceToHost);
	started = smoke_now_ns();
	for ( iter = 0u; iter < iterations; iter++ )
		if ( cudaMemcpy(host,device,bytes,cudaMemcpyDeviceToHost) != cudaSuccess )
		{
			cudaFreeHost(host);
			cudaFree(device);
			return -42;
		}
	elapsed = smoke_now_ns() - started;
	gib_per_second = ((double)bytes * (double)iterations / 1073741824.0) /
		((double)elapsed / 1e9);
	snprintf(payload,sizeof(payload),
		"{\"op\":\"bandwidth\",\"pid\":%d,\"bytes\":%llu,\"iters\":%u,\"gib_per_s\":%.1f}",
		(int)getpid(),(unsigned long long)bytes,iterations,gib_per_second);
	smoke_emit("BANDWIDTH",payload);
	cudaFreeHost(host);
	cudaFree(device);
	return 0;
}

static int smoke_evictor(const char *socket,uint32_t acquire_lane,uint32_t hold_seconds,
	const char *targets_text)
{
	SparkWeightdClient *client = 0;
	SparkWeightdHelloResult hello;
	uint32_t lane = 0xffffffffu;
	char payload[256];
	char *copy;
	char *token;
	char *save = 0;
	if ( SparkWeightdClientConnect(socket,&client,&hello) != SPARK_STATUS_OK )
		return -44;
	if ( acquire_lane != 0u )
	{
		if ( SparkWeightdClientLaneAcquire(client,&lane,SMOKE_TIMEOUT) != SPARK_STATUS_OK )
		{
			SparkWeightdClientClose(client);
			return -45;
		}
		snprintf(payload,sizeof(payload),"{\"op\":\"evictor-init\",\"pid\":%d,\"lane\":%u,"
			"\"hold_seconds\":%u}",(int)getpid(),lane,hold_seconds);
		smoke_emit("EVICTOR",payload);
		sleep(hold_seconds);
	}
	copy = strdup(targets_text);
	if ( copy == 0 )
	{
		SparkWeightdClientClose(client);
		return -46;
	}
	for ( token = strtok_r(copy,",",&save); token != 0; token = strtok_r(0,",",&save) )
	{
		uint32_t target = 0u;
		uint32_t released = 0u;
		SparkStatus status;
		if ( smoke_parse_u32(token,&target) != 0 )
			continue;
		status = SparkWeightdClientEvict(client,target,&released,SMOKE_TIMEOUT);
		snprintf(payload,sizeof(payload),
			"{\"op\":\"evict\",\"evictor_lane\":%s,\"target_lane\":%u,\"status\":\"%s\","
			"\"released_leases\":%u}",
			acquire_lane != 0u ? "held" : "none",target,SparkStatusToString(status),released);
		smoke_emit("EVICT",payload);
	}
	free(copy);
	SparkWeightdClientClose(client);
	return 0;
}

typedef struct SmokeClassSum
{
	uint64_t spine_bytes;
	uint64_t expert_bytes;
	uint32_t layers;
	uint32_t groups;
	uint32_t max_experts_per_layer;
} SmokeClassSum;

static void smoke_class_sums(const SparkWeightdManifest *manifest,SmokeClassSum *sums)
{
	uint32_t i;
	uint32_t layer_cursor = 0xffffffffu;
	uint32_t experts_this_layer = 0u;
	memset(sums,0,sizeof(*sums));
	sums->spine_bytes = manifest->spine_bytes;
	for ( i = 0u; i < manifest->group_count; i++ )
	{
		const SparkWeightdRangeGroup *group = &manifest->groups[i];
		uint32_t j;
		uint64_t bytes = 0u;
		sums->groups++;
		if ( group->layer != layer_cursor )
		{
			layer_cursor = group->layer;
			sums->layers++;
			experts_this_layer = 0u;
		}
		experts_this_layer++;
		if ( experts_this_layer > sums->max_experts_per_layer )
			sums->max_experts_per_layer = experts_this_layer;
		for ( j = 0u; j < group->range_count; j++ )
			bytes += manifest->ranges[group->first_range + j].bytes;
		sums->expert_bytes += bytes;
	}
}

static int smoke_realws(const char *socket,const char *pack_path,const char *name,
	uint32_t topk,uint64_t pool_bytes,const char *sha_hex)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdClient *client = 0;
	SparkWeightdHelloResult hello;
	SparkWeightdLazyAttachResult attached;
	SparkWeightdWorkingSetResult working;
	SparkWeightdManifest manifest;
	SmokeClassSum sums;
	struct stat info;
	char sidecar[512];
	char payload[640];
	SparkWeightdExpertKey *keys;
	uint32_t key_count = 0u;
	uint64_t lease_bytes_total = 0u;
	uint64_t lease_id[64];
	uint32_t lease_batches = 0u;
	uint32_t layer_index,expert_slot,i;
	int state = 0;
	if ( stat(pack_path,&info) != 0 || info.st_size <= 0 )
		return -48;
	if ( snprintf(sidecar,sizeof(sidecar),"%s.experts",pack_path) >= (int)sizeof(sidecar) )
		return -49;
	if ( SparkWeightdManifestLoad(sidecar,(uint64_t)info.st_size,&manifest) != SPARK_STATUS_OK )
		return -50;
	smoke_class_sums(&manifest,&sums);
	snprintf(payload,sizeof(payload),
		"{\"op\":\"realws-init\",\"dev\":\"%s\",\"pid\":%d,\"pack\":\"%s\",\"pack_bytes\":%llu,"
		"\"manifest_spine_bytes\":%llu,\"manifest_expert_bytes\":%llu,\"layers\":%u,"
		"\"groups\":%u,\"max_experts_per_layer\":%u}",
		name,(int)getpid(),pack_path,(unsigned long long)info.st_size,
		(unsigned long long)sums.spine_bytes,(unsigned long long)sums.expert_bytes,
		sums.layers,sums.groups,sums.max_experts_per_layer);
	smoke_emit("REALWS",payload);
	if ( SparkWeightdClientConnect(socket,&client,&hello) != SPARK_STATUS_OK )
	{
		SparkWeightdManifestDestroy(&manifest);
		return -51;
	}
	{
		uint32_t lane = 0xffffffffu;
		if ( SparkWeightdClientLaneAcquire(client,&lane,SMOKE_TIMEOUT) != SPARK_STATUS_OK )
		{
			SparkWeightdManifestDestroy(&manifest);
			SparkWeightdClientClose(client);
			return -57;
		}
		snprintf(payload,sizeof(payload),"{\"op\":\"realws-lane\",\"dev\":\"%s\",\"lane\":%u}",
			name,lane);
		smoke_emit("REALWS",payload);
	}
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (uint64_t)info.st_size;
	snprintf(request.identity.model,sizeof(request.identity.model),"%s",name);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",pack_path);
	request.expert_pool_bytes = pool_bytes;
	if ( sha_hex != 0 && sha_hex[0] != '\0' )
		snprintf(request.identity.pack_sha256,sizeof(request.identity.pack_sha256),
			"%s",sha_hex);
	else if ( SparkSha256File(pack_path,request.identity.pack_sha256) != SPARK_STATUS_OK )
	{
		SparkWeightdManifestDestroy(&manifest);
		SparkWeightdClientClose(client);
		return -52;
	}
	if ( state == 0 && SparkWeightdClientAttachLazy(client,&request,&attached,SMOKE_TIMEOUT) !=
		SPARK_STATUS_OK )
		state = -53;
	if ( state == 0 )
	{
		snprintf(payload,sizeof(payload),
			"{\"op\":\"realws-attach\",\"dev\":\"%s\",\"resident_bytes\":%llu,"
			"\"expert_count\":%u,\"arena_count\":%u,\"chunk_bytes\":%llu,\"loaded_from_pack\":%u}",
			name,(unsigned long long)attached.resident_bytes,attached.expert_count,
			attached.arena_count,(unsigned long long)attached.chunk_bytes,
			attached.loaded_from_pack);
		smoke_emit("REALWS",payload);
	}
	keys = malloc(sizeof(*keys) * 512u);
	if ( state == 0 && keys == 0 )
		state = -54;
	{
		uint64_t pull_started = smoke_now_ns();
		uint64_t pull_ns = 0u;
		uint32_t batch_ordinal = 0u;
		for ( layer_index = 0u; state == 0 && layer_index < sums.layers; layer_index++ )
		{
			uint32_t experts_this_layer = 0u;
			for ( i = 0u; i < manifest.group_count; i++ )
			{
				const SparkWeightdRangeGroup *group = &manifest.groups[i];
				if ( group->layer != layer_index )
					continue;
				if ( experts_this_layer < topk && key_count < 512u )
				{
					keys[key_count].layer = group->layer;
					keys[key_count].expert = group->expert;
					key_count++;
					experts_this_layer++;
				}
			}
			if ( key_count == 512u || layer_index + 1u == sums.layers )
			{
				if ( key_count > 0u )
				{
					uint64_t batch_started = smoke_now_ns();
					if ( SparkWeightdClientAcquire(client,attached.arena_generation,keys,
						key_count,&working,SMOKE_TIMEOUT) != SPARK_STATUS_OK )
					{
						state = -55;
						break;
					}
					pull_ns = smoke_now_ns() - batch_started;
					lease_bytes_total += working.resident_bytes;
					if ( lease_batches < 64u )
						lease_id[lease_batches] = working.lease_identifier;
					lease_batches++;
					snprintf(payload,sizeof(payload),
						"{\"op\":\"realws-batch\",\"dev\":\"%s\",\"batch\":%u,\"keys\":%u,"
						"\"resident_bytes\":%llu,\"pull_ms\":%.1f,\"gib_per_s\":%.2f}",
						name,batch_ordinal,key_count,
						(unsigned long long)working.resident_bytes,
						(double)pull_ns / 1e6,(double)working.resident_bytes /
						1073741824.0 / ((double)pull_ns / 1e9));
					smoke_emit("REALWS",payload);
					batch_ordinal++;
					key_count = 0u;
				}
			}
		}
		(void)pull_started;
	}
	if ( state == 0 )
	{
		double ws_pct = (double)lease_bytes_total * 100.0 / (double)info.st_size;
		snprintf(payload,sizeof(payload),
			"{\"op\":\"realws-working-set\",\"dev\":\"%s\",\"lease_resident_bytes\":%llu,"
			"\"lease_batches\":%u,\"ws_pct_of_pack\":%.2f,\"spine_bytes\":%llu,"
			"\"expert_bytes_total\":%llu}",
			name,(unsigned long long)lease_bytes_total,lease_batches,ws_pct,
			(unsigned long long)sums.spine_bytes,(unsigned long long)sums.expert_bytes);
		smoke_emit("REALWS",payload);
	}
	for ( i = 0u; i < lease_batches && i < 64u; i++ )
		(void)SparkWeightdClientRelease(client,attached.arena_generation,lease_id[i],
			&working,SMOKE_TIMEOUT);
	free(keys);
	if ( state == 0 )
	{
		SparkWeightdDetachResult detached;
		if ( SparkWeightdClientDetach(client,attached.arena_generation,&detached,
			SMOKE_TIMEOUT) != SPARK_STATUS_OK )
			state = -56;
	}
	SparkWeightdManifestDestroy(&manifest);
	SparkWeightdClientClose(client);
	(void)expert_slot;
	return state;
}

int main(int argument_count,char **arguments)
{
	uint64_t wide_a = 0u;
	uint32_t parts[6];
	int state = 2;
	if ( argument_count == 9 && strcmp(arguments[1],"makepack") == 0 &&
		smoke_parse_u32(arguments[3],&parts[0]) == 0 && smoke_parse_u32(arguments[4],&parts[1]) == 0 &&
		smoke_parse_u32(arguments[5],&parts[2]) == 0 && smoke_parse_u32(arguments[6],&parts[3]) == 0 &&
		smoke_parse_u64(arguments[7],&wide_a) == 0 && smoke_parse_u32(arguments[8],&parts[4]) == 0 )
		state = smoke_makepack(arguments[2],parts[0],parts[1],parts[2],parts[3],wide_a,
			parts[4]) == 0 ? 0 : 1;
	else if ( argument_count == 10 && strcmp(arguments[1],"dev") == 0 &&
		smoke_parse_u32(arguments[5],&parts[0]) == 0 && smoke_parse_u32(arguments[6],&parts[1]) == 0 &&
		smoke_parse_u64(arguments[7],&wide_a) == 0 && smoke_parse_u32(arguments[8],&parts[2]) == 0 &&
		smoke_parse_u32(arguments[9],&parts[3]) == 0 )
		state = smoke_dev(arguments[2],arguments[3],arguments[4],parts[0],parts[1],wide_a,
			parts[2],parts[3]) == 0 ? 0 : 1;
	else if ( argument_count == 4 && strcmp(arguments[1],"bandwidth") == 0 &&
		smoke_parse_u32(arguments[2],&parts[0]) == 0 && smoke_parse_u32(arguments[3],&parts[1]) == 0 )
		state = smoke_bandwidth(parts[0],parts[1]) == 0 ? 0 : 1;
	else if ( argument_count == 6 && strcmp(arguments[1],"evictor") == 0 &&
		smoke_parse_u32(arguments[3],&parts[0]) == 0 && smoke_parse_u32(arguments[4],&parts[1]) == 0 )
		state = smoke_evictor(arguments[2],parts[0],parts[1],arguments[5]) == 0 ? 0 : 1;
	else if ( argument_count == 8 && strcmp(arguments[1],"realws") == 0 &&
		smoke_parse_u32(arguments[5],&parts[0]) == 0 && smoke_parse_u64(arguments[6],&wide_a) == 0 )
		state = smoke_realws(arguments[2],arguments[3],arguments[4],parts[0],wide_a,
			arguments[7]) == 0 ? 0 : 1;
	if ( state != 0 )
		fprintf(stderr,"SMOKE-FAIL mode=%s rc=%d\n",
			argument_count > 1 ? arguments[1] : "?",state);
	if ( state == 2 )
	{
		fprintf(stderr,"usage: %s makepack PACK LAYERS EXPERTS KINDS RANGE_BYTES SPINE_GAP SEED\n"
			"       %s dev SOCKET PACK NAME ROUNDS KEYS POOL_BYTES PINS HOLD_SECONDS\n"
			"       %s bandwidth MB ITERS\n"
			"       %s evictor SOCKET ACQUIRE_LANE(0|1) HOLD_SECONDS TARGETS\n"
			"       %s realws SOCKET PACK NAME TOPK POOL_BYTES SHA256_OR_DASH\n",
			arguments[0],arguments[0],arguments[0],arguments[0],arguments[0]);
	}
	return state;
}
