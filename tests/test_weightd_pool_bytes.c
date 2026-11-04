#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#define MIB (1024ull * 1024ull)
#define CHUNK (2ull * MIB)

typedef struct PoolCase
{
	const char *name;
	uint32_t count;
	uint64_t offsets[6];
	uint64_t bytes[6];
	uint64_t chunk_bytes;
	uint64_t expected;
} PoolCase;

static uint64_t pool_of(const PoolCase *item)
{
	SparkWeightdRange ranges[6];
	SparkWeightdManifest manifest;
	uint64_t bytes = UINT64_MAX;
	uint32_t i;
	memset(ranges,0,sizeof(ranges));
	memset(&manifest,0,sizeof(manifest));
	for (i=0u; i<item->count; i++)
	{
		ranges[i].offset = item->offsets[i];
		ranges[i].bytes = item->bytes[i];
		ranges[i].expert = i;
	}
	manifest.ranges = item->count != 0u ? ranges : 0;
	manifest.range_count = item->count;
	assert(SparkWeightdManifestPoolBytes(&manifest,item->chunk_bytes,&bytes) == SPARK_STATUS_OK);
	return(bytes);
}

static void check_pool_bytes(void)
{
	static const PoolCase cases[] =
	{
		{"empty manifest: the daemon's one-slot minimum",0u,{0},{0},CHUNK,1u * CHUNK},
		{"two ranges inside chunk 0",2u,{0u,4096u},{4096u,8192u},CHUNK,1u * CHUNK},
		{"one range inside chunk 3",1u,{(3u * CHUNK) + 100u},{200u},CHUNK,1u * CHUNK},
		{"a range straddling the chunk 0/1 boundary",1u,{CHUNK - 64u},{128u},CHUNK,2u * CHUNK},
		{"a range covering chunks 1..3 and touching 4",1u,{CHUNK + 1u},{3u * CHUNK},CHUNK,4u * CHUNK},
		{"adjacent ranges sharing chunk 1",2u,{CHUNK,CHUNK + MIB},{MIB,MIB},CHUNK,1u * CHUNK},
		{"adjacent ranges listed out of order across chunks 0..2",3u,{(2u * CHUNK) - 512u,CHUNK - 512u,0u},{1024u,CHUNK,CHUNK - 512u},CHUNK,3u * CHUNK},
		{"gaps: chunks 0, 5 and 9 only",3u,{9u * CHUNK,0u,(5u * CHUNK) + 7u},{1u,1u,1u},CHUNK,3u * CHUNK},
		{"a range ending exactly on the chunk 0/1 boundary",1u,{0u},{CHUNK},CHUNK,1u * CHUNK},
		{"a range ending one byte past the boundary",1u,{0u},{CHUNK + 1u},CHUNK,2u * CHUNK},
		{"a range starting exactly on a boundary after one that ends on it",2u,{0u,CHUNK},{CHUNK,CHUNK},CHUNK,2u * CHUNK},
		{"ranges in chunks 0, 1 and 3 with chunk 2 untouched",4u,{64u,CHUNK + 64u,(3u * CHUNK),(3u * CHUNK) + 256u},{64u,64u,64u,64u},CHUNK,3u * CHUNK},
		{"a 4 KiB chunk size counts 4 KiB chunks",2u,{0u,12288u},{4097u,1u},4096u,3u * 4096u},
	};
	uint32_t i;
	for (i=0u; i<sizeof(cases) / sizeof(cases[0]); i++)
	{
		uint64_t got = pool_of(&cases[i]);
		if ( got != cases[i].expected )
		{
			fprintf(stderr,"FAIL %s: pool %llu, expected %llu\n",cases[i].name,(unsigned long long)got,(unsigned long long)cases[i].expected);
			exit(1);
		}
	}
}

static void check_refusals(void)
{
	SparkWeightdRange range = {0};
	SparkWeightdManifest manifest;
	uint64_t bytes = 7u;
	memset(&manifest,0,sizeof(manifest));
	assert(SparkWeightdManifestPoolBytes(&manifest,CHUNK,0) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkWeightdManifestPoolBytes(0,CHUNK,&bytes) == SPARK_STATUS_INVALID_ARGUMENT && bytes == 0u);
	bytes = 7u;
	assert(SparkWeightdManifestPoolBytes(&manifest,0u,&bytes) == SPARK_STATUS_INVALID_ARGUMENT && bytes == 0u);
	manifest.range_count = 1u;
	assert(SparkWeightdManifestPoolBytes(&manifest,CHUNK,&bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	manifest.ranges = &range;
	assert(SparkWeightdManifestPoolBytes(&manifest,CHUNK,&bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	range.offset = UINT64_MAX - 3u;
	range.bytes = 8u;
	assert(SparkWeightdManifestPoolBytes(&manifest,CHUNK,&bytes) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void check_plan(void)
{
	SparkWeightdRange ranges[2];
	SparkWeightdManifest manifest;
	SparkWeightdPoolPlan plan;
	const uint64_t arena = (5u * CHUNK) + 512u;
	memset(ranges,0,sizeof(ranges));
	memset(&manifest,0,sizeof(manifest));
	assert(SPARK_WEIGHTD_EXPERT_CHUNK_BYTES == CHUNK);
	ranges[0].offset = CHUNK - 64u;
	ranges[0].bytes = 128u;
	ranges[1].offset = 4u * CHUNK;
	ranges[1].bytes = 64u;
	ranges[1].expert = 1u;
	manifest.ranges = ranges;
	manifest.range_count = 2u;
	assert(SparkWeightdManifestPoolPlan(&manifest,arena,0u,&plan) == SPARK_STATUS_OK);
	assert(plan.exact_bytes == 3u * CHUNK && plan.cap_bytes == 0u && plan.whole_arena == 1u);
	assert(plan.request_bytes == arena + CHUNK && plan.used_bytes == 6u * CHUNK);
	assert(SparkWeightdManifestPoolPlan(&manifest,arena,2u * CHUNK,&plan) == SPARK_STATUS_OK);
	assert(plan.exact_bytes == 3u * CHUNK && plan.cap_bytes == 2u * CHUNK && plan.whole_arena == 0u);
	assert(plan.request_bytes == 2u * CHUNK && plan.used_bytes == 2u * CHUNK);
	assert(SparkWeightdManifestPoolPlan(&manifest,arena,3u * CHUNK,&plan) == SPARK_STATUS_OK);
	assert(plan.request_bytes == 3u * CHUNK && plan.used_bytes == 3u * CHUNK && plan.whole_arena == 0u);
	assert(SparkWeightdManifestPoolPlan(&manifest,arena,4u * CHUNK,&plan) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(plan.request_bytes == 0u);
	assert(SparkWeightdManifestPoolPlan(&manifest,arena,CHUNK + 4096u,&plan) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(plan.request_bytes == 0u);
	assert(SparkWeightdManifestPoolPlan(&manifest,0u,0u,&plan) == SPARK_STATUS_INVALID_ARGUMENT);
	manifest.range_count = 0u;
	manifest.ranges = 0;
	assert(SparkWeightdManifestPoolPlan(&manifest,arena,0u,&plan) == SPARK_STATUS_OK);
	assert(plan.exact_bytes == CHUNK && plan.request_bytes == CHUNK && plan.used_bytes == CHUNK && plan.whole_arena == 0u);
	assert(SparkWeightdManifestPoolPlan(&manifest,arena,2u * CHUNK,&plan) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void check_pack_plan(void)
{
	char root[] = "/tmp/weightd-pool-plan-XXXXXX",pack[256],sidecar[272];
	uint8_t record[48] = {0};
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,1u,0u};
	uint64_t offset = (2u * CHUNK) - 32u,bytes = 64u,arena = 4u * CHUNK;
	SparkWeightdPoolPlan plan;
	FILE *file;
	assert(mkdtemp(root) != 0);
	snprintf(pack,sizeof(pack),"%s/pack",root);
	snprintf(sidecar,sizeof(sidecar),"%s.experts",pack);
	assert(SparkWeightdPackPoolPlan(pack,arena,0u,&plan) == SPARK_STATUS_NOT_FOUND);
	file = fopen(sidecar,"wb");
	assert(file != 0);
	memcpy(record + 16u,&offset,8u);
	memcpy(record + 24u,&bytes,8u);
	assert(fwrite(header,1u,sizeof(header),file) == sizeof(header) && fwrite(record,1u,sizeof(record),file) == sizeof(record));
	assert(fclose(file) == 0);
	assert(SparkWeightdPackPoolPlan(pack,arena,0u,&plan) == SPARK_STATUS_OK);
	assert(plan.exact_bytes == 2u * CHUNK && plan.request_bytes == arena + CHUNK && plan.used_bytes == arena && plan.whole_arena == 1u);
	assert(SparkWeightdPackPoolPlan(pack,arena,CHUNK,&plan) == SPARK_STATUS_OK);
	assert(plan.request_bytes == CHUNK && plan.whole_arena == 0u);
	assert(unlink(sidecar) == 0 && rmdir(root) == 0);
}

int main(void)
{
	check_pool_bytes();
	check_refusals();
	check_plan();
	check_pack_plan();
	puts("PASS weightd pool bytes: distinct expert chunks per manifest, one chunk without ranges, cap at or below the exact pool, whole arena without a cap");
	return(0);
}
