#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_lease_sets.h"
#include "tests/test_stderr_capture.h"

#define CHUNK (2u * 1024u * 1024u)
#define LAYERS 6u
#define EXPERTS 3u
#define PACK_BYTES ((LAYERS * CHUNK) + 512u)
#define TIMEOUT UINT64_C(10000000000)

void spark_stub_cuda_event_pending(uint32_t pending);
void spark_stub_cuda_event_record_failure(uint32_t failure);
void spark_stub_cuda_event_pending_queries(uint32_t queries);
uint32_t spark_stub_cuda_event_queries(void);
uint32_t spark_stub_cuda_outstanding_allocs(void);

typedef struct TestServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
} TestServer;

static void *run_server(void *data)
{
	TestServer *state = data;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return(0);
}

static uint32_t record_calls;

static SparkStatus record_on_default_stream(void *context,SparkWeightdMap *map,uint64_t identifier)
{
	assert(context == 0);
	record_calls++;
	return(SparkWeightdMapRecordCompletion(map,identifier,0));
}

static uint8_t pattern(uint32_t layer,uint32_t expert)
{
	return((uint8_t)(1u + (layer * EXPERTS) + expert));
}

static uint64_t expert_offset(uint32_t layer,uint32_t expert)
{
	return(((uint64_t)layer * CHUNK) + (expert * 256u));
}

static void write_fixture(const char *path,const char *manifest_path)
{
	FILE *pack = fopen(path,"wb"),*manifest = fopen(manifest_path,"wb");
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,LAYERS * EXPERTS,0u};
	uint8_t record[48],data[64];
	uint64_t offset,bytes = sizeof(data);
	uint32_t layer,expert;
	SparkCk128Context ck;
	assert(pack != 0 && manifest != 0);
	assert(ftruncate(fileno(pack),PACK_BYTES) == 0);
	assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
	for (layer=0u; layer<LAYERS; layer++)
		for (expert=0u; expert<EXPERTS; expert++)
		{
			memset(data,pattern(layer,expert),sizeof(data));
			offset = expert_offset(layer,expert);
			assert(fseek(pack,(long)offset,SEEK_SET) == 0 && fwrite(data,1u,sizeof(data),pack) == sizeof(data));
			memset(record,0,sizeof(record));
			memcpy(record,&layer,4u);
			memcpy(record + 4u,&expert,4u);
			memcpy(record + 16u,&offset,8u);
			memcpy(record + 24u,&bytes,8u);
			SparkCk128Initialize(&ck);
			SparkCk128Update(&ck,data,sizeof(data));
			SparkCk128Finalize(&ck,record + 32u);
			assert(fwrite(record,1u,sizeof(record),manifest) == sizeof(record));
		}
	assert(fclose(pack) == 0 && fclose(manifest) == 0);
}

static void layer_keys(uint32_t layer,SparkWeightdExpertKey keys[EXPERTS])
{
	uint32_t expert;
	for (expert=0u; expert<EXPERTS; expert++)
	{
		keys[expert].layer = layer;
		keys[expert].expert = expert;
	}
}

static void check_layer(const void *base,uint32_t layer)
{
	const uint8_t *data;
	uint32_t expert,i;
	assert(base != 0);
	for (expert=0u; expert<EXPERTS; expert++)
	{
		data = (const uint8_t *)base + expert_offset(layer,expert);
		for (i=0u; i<64u; i++)
			assert(data[i] == pattern(layer,expert));
	}
}

static uint64_t now_ns(void)
{
	struct timespec now;
	assert(clock_gettime(CLOCK_MONOTONIC,&now) == 0);
	return(((uint64_t)now.tv_sec * UINT64_C(1000000000)) + (uint64_t)now.tv_nsec);
}

static void check_read_ahead(SparkWeightdLazyPack *pack)
{
	SparkWeightdLeaseSets *sets = 0;
	SparkWeightdExpertKey keys[EXPERTS];
	uint64_t extra = 0u;
	uint32_t layer,set,queries,calls;
	void *base;
	assert(SparkWeightdLeaseSetsCreate(2u,EXPERTS + 1u,record_on_default_stream,0,&sets) == SPARK_STATUS_OK);
	assert(SparkWeightdLeaseSetsCount(sets) == 2u);
	for (layer=0u; layer<LAYERS; layer++)
	{
		set = layer % 2u;
		layer_keys(layer,keys);
		if ( layer >= 2u )
		{
			assert(SparkWeightdLeaseSetsHeld(sets,0u) == 1u && SparkWeightdLeaseSetsHeld(sets,1u) == 1u);
			assert(SparkWeightdMapAcquire(pack->map,keys,EXPERTS,&extra,TIMEOUT) == SPARK_STATUS_CAPACITY_EXCEEDED);
			if ( extra != 0u )
				assert(SparkWeightdMapRelease(pack->map,extra,TIMEOUT) == SPARK_STATUS_OK);
			extra = 0u;
			spark_stub_cuda_event_pending_queries(3u);
		}
		queries = spark_stub_cuda_event_queries();
		assert(SparkWeightdLeaseSetsRelease(sets,pack->map,set,TIMEOUT) == SPARK_STATUS_OK);
		if ( layer >= 2u )
			assert(spark_stub_cuda_event_queries() - queries >= 4u);
		assert(SparkWeightdLeaseSetsHeld(sets,set) == 0u);
		base = 0;
		assert(SparkWeightdLeaseSetsAcquire(sets,pack->map,set,keys,EXPERTS,TIMEOUT,&base) == SPARK_STATUS_OK);
		check_layer(base,layer);
		assert(SparkWeightdLeaseSetsHeld(sets,set) == 1u);
		if ( layer >= 1u )
			assert(SparkWeightdLeaseSetsHeld(sets,set ^ 1u) == 1u);
		assert(SparkWeightdLeaseSetsAcquire(sets,pack->map,set,keys,EXPERTS,TIMEOUT,&base) == SPARK_STATUS_INVALID_ARGUMENT && base == 0);
		calls = record_calls;
		assert(SparkWeightdLeaseSetsRecord(sets,pack->map,set) == SPARK_STATUS_OK && record_calls == calls + 1u);
		assert(SparkWeightdLeaseSetsRecord(sets,pack->map,set) == SPARK_STATUS_OK && record_calls == calls + 1u);
	}
	spark_stub_cuda_event_pending_queries(2u);
	assert(SparkWeightdLeaseSetsDestroy(sets) == SPARK_STATUS_BUSY);
	assert(SparkWeightdLeaseSetsReleaseAll(sets,pack->map,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdLeaseSetsHeld(sets,0u) == 0u && SparkWeightdLeaseSetsHeld(sets,1u) == 0u);
	assert(SparkWeightdLeaseSetsDestroy(sets) == SPARK_STATUS_OK);
}

static void check_release_before_completion(SparkWeightdLazyPack *pack)
{
	SparkWeightdLeaseSets *sets = 0;
	SparkWeightdExpertKey keys[EXPERTS];
	TestStderrCapture capture;
	uint64_t lease = 0u,started,elapsed;
	uint32_t queries,calls;
	void *base = 0;
	char *text;
	layer_keys(0u,keys);
	assert(SparkWeightdMapAcquire(pack->map,keys,EXPERTS,&lease,TIMEOUT) == SPARK_STATUS_OK && lease != 0u);
	assert(SparkWeightdMapBeginUse(pack->map,lease,&base) == SPARK_STATUS_OK);
	check_layer(base,0u);
	assert(SparkWeightdMapReleaseCompleted(pack->map,lease,TIMEOUT) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkWeightdMapRecordCompletion(pack->map,lease,0) == SPARK_STATUS_OK);
	spark_stub_cuda_event_pending_queries(5u);
	assert(SparkWeightdMapRelease(pack->map,lease,TIMEOUT) == SPARK_STATUS_BUSY);
	queries = spark_stub_cuda_event_queries();
	assert(SparkWeightdMapReleaseCompleted(pack->map,lease,TIMEOUT) == SPARK_STATUS_OK);
	assert(spark_stub_cuda_event_queries() - queries == 5u);
	assert(SparkWeightdMapReleaseCompleted(pack->map,lease,TIMEOUT) == SPARK_STATUS_NOT_FOUND);

	assert(SparkWeightdLeaseSetsCreate(2u,EXPERTS,record_on_default_stream,0,&sets) == SPARK_STATUS_OK);
	assert(SparkWeightdLeaseSetsAcquire(sets,pack->map,0u,keys,EXPERTS,TIMEOUT,&base) == SPARK_STATUS_OK);
	assert(SparkWeightdLeaseSetsRecord(sets,pack->map,0u) == SPARK_STATUS_OK);
	spark_stub_cuda_event_pending(1u);
	started = now_ns();
	TestStderrCaptureBegin(&capture);
	assert(SparkWeightdLeaseSetsRelease(sets,pack->map,0u,UINT64_C(20000000)) == SPARK_STATUS_BUSY);
	text = TestStderrCaptureEnd(&capture);
	elapsed = now_ns() - started;
	assert(strstr(text,"MAP-RELEASE-WAIT-EXPIRED") != 0 && strstr(text,"the lease stays held") != 0);
	free(text);
	assert(elapsed >= UINT64_C(20000000) && elapsed < UINT64_C(5000000000));
	assert(SparkWeightdLeaseSetsHeld(sets,0u) == 1u);
	spark_stub_cuda_event_pending(0u);
	assert(SparkWeightdLeaseSetsRelease(sets,pack->map,0u,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdLeaseSetsHeld(sets,0u) == 0u);

	assert(SparkWeightdLeaseSetsAcquire(sets,pack->map,1u,keys,EXPERTS,TIMEOUT,&base) == SPARK_STATUS_OK);
	spark_stub_cuda_event_pending_queries(2u);
	queries = spark_stub_cuda_event_queries();
	calls = record_calls;
	assert(SparkWeightdLeaseSetsRelease(sets,pack->map,1u,TIMEOUT) == SPARK_STATUS_OK);
	assert(spark_stub_cuda_event_queries() - queries == 3u && record_calls == calls + 1u);

	assert(SparkWeightdLeaseSetsAcquire(sets,pack->map,0u,keys,EXPERTS,TIMEOUT,&base) == SPARK_STATUS_OK);
	spark_stub_cuda_event_record_failure(1u);
	assert(SparkWeightdLeaseSetsRecord(sets,pack->map,0u) == SPARK_STATUS_IO_ERROR);
	assert(SparkWeightdLeaseSetsRelease(sets,pack->map,0u,TIMEOUT) == SPARK_STATUS_IO_ERROR);
	assert(SparkWeightdLeaseSetsHeld(sets,0u) == 1u);
	spark_stub_cuda_event_record_failure(0u);
	assert(SparkWeightdLeaseSetsRelease(sets,pack->map,0u,TIMEOUT) == SPARK_STATUS_OK);
	assert(SparkWeightdLeaseSetsDestroy(sets) == SPARK_STATUS_OK);
}

static void check_fit(SparkWeightdLazyPack *pack)
{
	SparkWeightdLazyAttachResult attached = pack->attached;
	SparkWeightdRange ranges[3];
	SparkWeightdManifest manifest;
	uint64_t widest = 1u,pool = 1u,chunks = 1u;
	assert(SparkWeightdManifestLayerChunks(&pack->manifest,CHUNK,0u,LAYERS,&chunks) == SPARK_STATUS_OK && chunks == LAYERS);
	assert(SparkWeightdManifestLayerChunks(&pack->manifest,CHUNK,4u,2u,&chunks) == SPARK_STATUS_OK && chunks == 2u);
	assert(SparkWeightdManifestLayerChunks(&pack->manifest,CHUNK,LAYERS,1u,&chunks) == SPARK_STATUS_OK && chunks == 0u);
	assert(SparkWeightdManifestLayerChunks(&pack->manifest,0u,0u,1u,&chunks) == SPARK_STATUS_INVALID_ARGUMENT);
	memset(ranges,0,sizeof(ranges));
	memset(&manifest,0,sizeof(manifest));
	ranges[0].layer = 3u;
	ranges[0].offset = CHUNK - 32u;
	ranges[0].bytes = 64u;
	ranges[1].layer = 3u;
	ranges[1].expert = 1u;
	ranges[1].offset = CHUNK + 64u;
	ranges[1].bytes = 64u;
	ranges[2].layer = 4u;
	ranges[2].offset = 5u * CHUNK;
	ranges[2].bytes = (2u * CHUNK) + 1u;
	manifest.ranges = ranges;
	manifest.range_count = 3u;
	assert(SparkWeightdManifestLayerChunks(&manifest,CHUNK,3u,1u,&chunks) == SPARK_STATUS_OK && chunks == 2u);
	assert(SparkWeightdManifestLayerChunks(&manifest,CHUNK,4u,1u,&chunks) == SPARK_STATUS_OK && chunks == 3u);
	assert(SparkWeightdManifestLayerChunks(&manifest,CHUNK,3u,2u,&chunks) == SPARK_STATUS_OK && chunks == 5u);
	assert(SparkWeightdLeaseSetsFit(&pack->manifest,&attached,LAYERS,2u,&widest,&pool) == SPARK_STATUS_OK);
	assert(widest == 2u && pool == 2u);
	attached.expert_pool_bytes = CHUNK;
	assert(SparkWeightdLeaseSetsFit(&pack->manifest,&attached,LAYERS,2u,&widest,&pool) == SPARK_STATUS_OK);
	assert(widest == 2u && pool == 1u);
	attached.expert_pool_bytes = PACK_BYTES + CHUNK;
	assert(SparkWeightdLeaseSetsFit(&pack->manifest,&attached,LAYERS,2u,&widest,&pool) == SPARK_STATUS_OK);
	assert(widest == 2u && pool == UINT64_MAX);
	assert(SparkWeightdLeaseSetsFit(&pack->manifest,&attached,1u,2u,&widest,&pool) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
	char root[] = "/tmp/weightd-lease-sets-XXXXXX",path[256],manifest[272],socket_path[256],wset[272],receipt[280];
	SparkWeightdServerConfig config = {0};
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyPack *pack = 0;
	TestServer state = {0};
	pthread_t thread;
	alarm(120u);
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest,sizeof(manifest),"%s.experts",path);
	snprintf(wset,sizeof(wset),"%s.wset",path);
	snprintf(receipt,sizeof(receipt),"%s.verified",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_fixture(path,manifest);
	config.socket_path = socket_path;
	config.device_bytes_max = (8u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = PACK_BYTES;
	memcpy(request.identity.model,"lease-sets-test",16u);
	assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	assert(SparkWeightdLazyPackCreate(socket_path,&request,2u * CHUNK,PACK_BYTES + 255u,TIMEOUT,&pack) == SPARK_STATUS_OK);
	assert(pack->attached.chunk_bytes == CHUNK && pack->attached.expert_pool_bytes == 2u * CHUNK);
	check_fit(pack);
	check_read_ahead(pack);
	check_release_before_completion(pack);
	assert(SparkWeightdLazyPackDestroy(pack) == SPARK_STATUS_OK);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	assert(spark_stub_cuda_outstanding_allocs() == 0u);
	(void)unlink(wset);
	(void)unlink(receipt);
	assert(unlink(manifest) == 0 && unlink(path) == 0 && rmdir(root) == 0);
	puts("PASS weightd lease sets: whole-layer leases two layers deep, a reused set waits for its recorded GPU work and a stuck wait expires with the lease held");
	return(0);
}
