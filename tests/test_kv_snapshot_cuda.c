#include <cuda_runtime_api.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_kv_snapshot.h"

#define CUDA_PAGES 64u
#define CUDA_BLOCK_TOKENS 64u
#define CUDA_HEAD_DIM 128u
#define CUDA_KEY_BYTES (CUDA_BLOCK_TOKENS * CUDA_HEAD_DIM * 4u)
#define CUDA_VALUE_BYTES (CUDA_BLOCK_TOKENS * CUDA_HEAD_DIM * 4u)
#define CUDA_PAGE_BYTES (CUDA_KEY_BYTES + CUDA_VALUE_BYTES)
#define CUDA_STATE_FLOATS (CUDA_HEAD_DIM * CUDA_HEAD_DIM)
#define CUDA_STATE_BYTES (CUDA_STATE_FLOATS * 4u)
#define CUDA_VOCAB 4096u
#define CUDA_PROMPT_TOKENS 1000u
#define CUDA_PROMPT_PAGES ((CUDA_PROMPT_TOKENS + CUDA_BLOCK_TOKENS - 1u) / CUDA_BLOCK_TOKENS)

static uint32_t cuda_checks,cuda_failures;

#define CHECK(condition) do { cuda_checks++; if ( !(condition) ) { cuda_failures++; fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#condition); } } while (0)

typedef struct CudaEngine
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[CUDA_PAGES];
	uint32_t resident[CUDA_PAGES];
	uint8_t *device_keys;
	uint8_t *device_values;
	float *device_state;
	SparkKvPageStore stores[2];
	uint8_t *page_staging;
	uint8_t *state_staging;
	char paths[2][64];
	SparkKvPageCache cache;
	SparkKvPageCacheEntry entries[CUDA_PAGES];
	SparkKvPageCacheSequence sequences[4];
	uint32_t heads[CUDA_PAGES];
	uint32_t by_page[CUDA_PAGES];
	SparkKvPageCacheSnapshot snapshot;
	SparkKvPageCacheSnapshotLink links[CUDA_PAGES];
	uint32_t pending[CUDA_PAGES];
	uint8_t *snapshot_page;
	uint8_t *snapshot_state;
	SparkKvLaneTransaction owners[4];
	uint32_t logical[4 * CUDA_PAGES];
	uint32_t physical[4 * CUDA_PAGES];
	SparkKvLaneTransactions transactions;
} CudaEngine;

static uint64_t CudaNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static SparkStatus CudaCopy(void *context,uint32_t direction,uintptr_t device,void *host,uint64_t bytes)
{
	cudaError_t error;
	(void)context;
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpy(host,(const void *)device,(size_t)bytes,cudaMemcpyDeviceToHost);
	else
		error = cudaMemcpy((void *)device,host,(size_t)bytes,cudaMemcpyHostToDevice);
	return(error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

int CudaLaunchPrefill(float *keys,float *values,float *state,const uint32_t *physical_pages,uint32_t first,uint32_t end);
int CudaLaunchLogits(const float *keys,const float *values,const float *state,const uint32_t *physical_pages,uint32_t context,uint32_t next_token,float *logits);

static int32_t CudaStoreOpen(SparkKvPageStore *store,char *path,void *staging,uint64_t page_bytes)
{
	SparkKvPageStoreConfiguration configuration;
	int descriptor;
	memset(&configuration,0,sizeof(configuration));
	strcpy(path,"/tmp/sparkpipe-kv-snapshot-cuda-XXXXXX");
	descriptor = mkstemp(path);
	if ( descriptor < 0 || close(descriptor) != 0 || unlink(path) != 0 )
		return(-1);
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	configuration.logical_page_capacity = CUDA_PAGES;
	configuration.transfer_capacity = 1u;
	configuration.page_bytes = configuration.staging_bytes = page_bytes;
	configuration.maximum_backing_bytes = page_bytes * CUDA_PAGES;
	configuration.backing_path = path;
	configuration.staging_address = staging;
	configuration.copy_function = page_bytes == CUDA_PAGE_BYTES ? CudaCopy : 0;
	return(SparkKvPageStoreInitialize(store,&configuration) == SPARK_STATUS_OK ? 0 : -2);
}

static int32_t CudaEngineOpen(CudaEngine *engine,SparkKvSnapshotStore *store)
{
	SparkKvCacheConfiguration arena;
	SparkKvPageCacheConfiguration config;
	memset(engine,0,sizeof(*engine));
	if ( cudaMalloc((void **)&engine->device_keys,(size_t)CUDA_PAGES * CUDA_KEY_BYTES) != cudaSuccess || cudaMalloc((void **)&engine->device_values,(size_t)CUDA_PAGES * CUDA_VALUE_BYTES) != cudaSuccess || cudaMalloc((void **)&engine->device_state,CUDA_STATE_BYTES) != cudaSuccess )
		return(-1);
	if ( cudaMemset(engine->device_keys,0xab,(size_t)CUDA_PAGES * CUDA_KEY_BYTES) != cudaSuccess || cudaMemset(engine->device_values,0xcd,(size_t)CUDA_PAGES * CUDA_VALUE_BYTES) != cudaSuccess || cudaMemset(engine->device_state,0,CUDA_STATE_BYTES) != cudaSuccess )
		return(-2);
	engine->page_staging = (uint8_t *)malloc(CUDA_PAGE_BYTES);
	engine->state_staging = (uint8_t *)malloc(CUDA_STATE_BYTES);
	engine->snapshot_page = (uint8_t *)malloc(CUDA_PAGE_BYTES);
	engine->snapshot_state = (uint8_t *)malloc(CUDA_STATE_BYTES);
	if ( engine->page_staging == 0 || engine->state_staging == 0 || engine->snapshot_page == 0 || engine->snapshot_state == 0 )
		return(-3);
	if ( CudaStoreOpen(&engine->stores[0],engine->paths[0],engine->page_staging,CUDA_PAGE_BYTES) != 0 || CudaStoreOpen(&engine->stores[1],engine->paths[1],engine->state_staging,CUDA_STATE_BYTES) != 0 )
		return(-4);
	memset(&arena,0,sizeof(arena));
	arena.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	arena.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	arena.logical_block_count = arena.resident_block_capacity = CUDA_PAGES;
	arena.block_token_count = CUDA_BLOCK_TOKENS;
	arena.layer_count = arena.kv_head_count = 1u;
	arena.head_dim = CUDA_HEAD_DIM;
	arena.bytes_per_scalar = 4u;
	arena.key_block_stride_bytes = CUDA_KEY_BYTES;
	arena.value_block_stride_bytes = CUDA_VALUE_BYTES;
	arena.key_device_base = engine->device_keys;
	arena.value_device_base = engine->device_values;
	arena.blocks = engine->blocks;
	arena.resident_slot_logical_block_indices = engine->resident;
	arena.evict_function = SparkKvPageStoreWriteback;
	arena.evict_context = &engine->stores[0];
	if ( SparkKvCacheArenaInitialize(&engine->arena,&arena) != SPARK_STATUS_OK )
		return(-5);
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	config.sequence_capacity = 4u;
	config.entry_capacity = config.hash_bucket_count = CUDA_PAGES;
	config.kv_cache_arena = &engine->arena;
	config.page_store = &engine->stores[0];
	config.entries = engine->entries;
	config.sequences = engine->sequences;
	config.hash_bucket_heads = engine->heads;
	config.entry_indices_by_logical_page = engine->by_page;
	if ( SparkKvPageCacheInitialize(&engine->cache,&config) != SPARK_STATUS_OK || SparkKvPageCacheAttachStateStore(&engine->cache,&engine->stores[1]) != SPARK_STATUS_OK )
		return(-6);
	engine->snapshot.store = store;
	memset(engine->snapshot.layout_sha256,0x5a,sizeof(engine->snapshot.layout_sha256));
	engine->snapshot.page_capacity = CUDA_PAGES;
	engine->snapshot.links = engine->links;
	engine->snapshot.page = engine->snapshot_page;
	engine->snapshot.state = engine->snapshot_state;
	engine->snapshot.pending_terminals = engine->pending;
	engine->snapshot.pending_capacity = CUDA_PAGES;
	engine->transactions.cache = &engine->cache;
	engine->transactions.lanes = engine->owners;
	engine->transactions.logical_pages = engine->logical;
	engine->transactions.physical_pages = engine->physical;
	engine->transactions.page_capacity = CUDA_PAGES;
	return(SparkKvPageCacheAttachSnapshot(&engine->cache,&engine->snapshot) == SPARK_STATUS_OK ? 0 : -7);
}

static void CudaEngineClose(CudaEngine *engine)
{
	SparkKvPageStoreDestroy(&engine->stores[0]);
	SparkKvPageStoreDestroy(&engine->stores[1]);
	cudaFree(engine->device_keys);
	cudaFree(engine->device_values);
	cudaFree(engine->device_state);
	free(engine->page_staging);
	free(engine->state_staging);
	free(engine->snapshot_page);
	free(engine->snapshot_state);
}

static void CudaIdentity(SparkModelDriverCacheIdentity *identity,uint32_t tokens)
{
	uint32_t index;
	for (index=0u; index<sizeof(identity->sha256); index++)
		identity->sha256[index] = (uint8_t)(tokens * 31u + index);
}

static void CudaRequest(SparkModelDriverAdmissionRequest *request,SparkModelDriverCacheLane *lane,uint64_t id,uint32_t flags,uint32_t frame_flags)
{
	memset(request,0,sizeof(*request));
	request->descriptor_bytes = sizeof(*request);
	request->program_id = 1u;
	request->request_id = request->submission_id = request->transaction_id = id;
	request->control_generation = request->request_generation = request->step_generation = 1u;
	request->active_slot_count = 1u;
	request->new_token_count = frame_flags == 0u ? 1u : 0u;
	request->cache_lane_count = 1u;
	request->cache_lanes = lane;
	request->admission_flags = flags;
	request->frame_flags = frame_flags;
}

static SparkStatus CudaCaptureState(CudaEngine *engine,uint32_t logical_page)
{
	uint64_t generation = engine->blocks[logical_page].generation;
	SparkStatus status;
	if ( cudaMemcpy(engine->snapshot_state,engine->device_state,CUDA_STATE_BYTES,cudaMemcpyDeviceToHost) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	status = SparkKvPageStoreWriteback(&engine->stores[1],logical_page,0u,generation,(uintptr_t)engine->snapshot_state,CUDA_STATE_BYTES,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(&engine->stores[1]);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreWriteback(&engine->stores[1],logical_page,0u,generation,(uintptr_t)engine->snapshot_state,CUDA_STATE_BYTES,0u,0u);
	}
	return(status);
}

static SparkStatus CudaRestoreState(CudaEngine *engine,uint32_t logical_page)
{
	uint64_t generation = engine->blocks[logical_page].generation;
	SparkStatus status = SparkKvPageStoreReadback(&engine->stores[1],logical_page,generation,(uintptr_t)engine->snapshot_state,CUDA_STATE_BYTES);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(&engine->stores[1]);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreReadback(&engine->stores[1],logical_page,generation,(uintptr_t)engine->snapshot_state,CUDA_STATE_BYTES);
	}
	if ( status == SPARK_STATUS_OK && cudaMemcpy(engine->device_state,engine->snapshot_state,CUDA_STATE_BYTES,cudaMemcpyHostToDevice) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	return(status);
}

static int32_t CudaNextLogits(CudaEngine *engine,uint32_t slot,uint32_t context,float *logits)
{
	uint32_t *device_pages;
	float *device_logits;
	uint32_t pages = (context + CUDA_BLOCK_TOKENS - 1u) / CUDA_BLOCK_TOKENS;
	if ( cudaMalloc((void **)&device_pages,pages * sizeof(uint32_t)) != cudaSuccess || cudaMalloc((void **)&device_logits,CUDA_VOCAB * sizeof(float)) != cudaSuccess )
		return(-1);
	if ( cudaMemcpy(device_pages,engine->physical + slot * CUDA_PAGES,pages * sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess )
		return(-2);
	if ( CudaLaunchLogits((const float *)engine->device_keys,(const float *)engine->device_values,engine->device_state,device_pages,context,context,device_logits) != 0 || cudaMemcpy(logits,device_logits,CUDA_VOCAB * sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(-3);
	cudaFree(device_pages);
	cudaFree(device_logits);
	return(0);
}

static int32_t CudaPrefillPrompt(CudaEngine *engine,float *logits,uint64_t *prefill_ns)
{
	SparkModelDriverCacheLane lane;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverFrame frame;
	uint32_t *device_pages,position = 0u,end;
	uint64_t start = CudaNow();
	if ( cudaMalloc((void **)&device_pages,CUDA_PAGES * sizeof(uint32_t)) != cudaSuccess )
		return(-1);
	while ( position < CUDA_PROMPT_TOKENS )
	{
		end = position + CUDA_BLOCK_TOKENS;
		if ( end > CUDA_PROMPT_TOKENS )
			end = CUDA_PROMPT_TOKENS;
		memset(&lane,0,sizeof(lane));
		lane.sequence_id = 1u;
		lane.request_generation = 1u;
		lane.step_generation = position + 1u;
		lane.resident_sequence_slot = 0u;
		lane.sequence_position = position;
		lane.context_token_count = end;
		lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
		lane.publish_token_count = end;
		CudaIdentity(&lane.publish_identity,end);
		CudaRequest(&request,&lane,position + 1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE,0u);
		if ( SparkKvLaneTransactionsAdmit(&engine->transactions,&request) != SPARK_STATUS_OK )
			return(-2);
		request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
		if ( SparkKvLaneTransactionsAdmit(&engine->transactions,&request) != SPARK_STATUS_OK )
			return(-3);
		memset(&frame,0,sizeof(frame));
		frame.program_id = 1u;
		frame.request_id = request.request_id;
		frame.active_slot_count = frame.new_token_count = 1u;
		frame.cache_lane_count = 1u;
		frame.cache_lanes = &lane;
		frame.driver_dispatch_generation = 1u;
		frame.driver_dispatch_cookie0 = frame.driver_dispatch_cookie1 = request.request_id;
		frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
		if ( SparkKvLaneTransactionsClaim(&engine->transactions,&frame) != SPARK_STATUS_OK )
			return(-4);
		if ( cudaMemcpy(device_pages,engine->physical,engine->owners[0].page_count * sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess )
			return(-5);
		if ( CudaLaunchPrefill((float *)engine->device_keys,(float *)engine->device_values,engine->device_state,device_pages,position,end) != 0 || CudaCaptureState(engine,engine->logical[engine->owners[0].page_count - 1u]) != SPARK_STATUS_OK )
			return(-6);
		{
			uint32_t slot = 0u;
			if ( SparkKvLaneTransactionsFinish(&engine->transactions,&slot,1u,SPARK_STATUS_OK,0u) != SPARK_STATUS_OK )
				return(-7);
		}
		position = end;
	}
	*prefill_ns = CudaNow() - start;
	cudaFree(device_pages);
	memset(&lane,0,sizeof(lane));
	lane.sequence_id = 1u;
	lane.request_generation = 1u;
	lane.step_generation = CUDA_PROMPT_TOKENS + 1u;
	lane.sequence_position = lane.context_token_count = CUDA_PROMPT_TOKENS;
	CudaRequest(&request,&lane,9000u,0u,0u);
	if ( SparkKvPageCacheBuildLaneTable(&engine->cache,0u,1u,engine->logical,CUDA_PAGES,&end) != SPARK_STATUS_OK || end != CUDA_PROMPT_PAGES )
		return(-8);
	for (position=0u; position<end; position++)
		engine->physical[position] = engine->blocks[engine->logical[position]].resident_slot_index;
	if ( CudaNextLogits(engine,0u,CUDA_PROMPT_TOKENS,logits) != 0 )
		return(-9);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	CudaRequest(&request,&lane,9001u,0u,SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE);
	return(SparkKvLaneTransactionsAdmit(&engine->transactions,&request) == SPARK_STATUS_OK ? 0 : -10);
}

static int32_t CudaRestorePrompt(CudaEngine *engine,float *logits,uint64_t *restore_ns)
{
	SparkModelDriverCacheLane lane;
	SparkModelDriverAdmissionRequest request;
	uint64_t start = CudaNow();
	uint32_t count;
	memset(&lane,0,sizeof(lane));
	lane.sequence_id = 5u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = 1u;
	lane.sequence_position = lane.context_token_count = CUDA_PROMPT_TOKENS;
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane.prefix_token_count = CUDA_PROMPT_TOKENS;
	CudaIdentity(&lane.prefix_identity,CUDA_PROMPT_TOKENS);
	CudaRequest(&request,&lane,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE,0u);
	if ( SparkKvLaneTransactionsAdmit(&engine->transactions,&request) != SPARK_STATUS_OK )
		return(-1);
	count = engine->owners[1].page_count;
	if ( count != CUDA_PROMPT_PAGES || CudaRestoreState(engine,engine->logical[CUDA_PAGES + count - 1u]) != SPARK_STATUS_OK )
		return(-2);
	*restore_ns = CudaNow() - start;
	if ( CudaNextLogits(engine,1u,CUDA_PROMPT_TOKENS,logits) != 0 )
		return(-3);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	return(SparkKvLaneTransactionsAdmit(&engine->transactions,&request) == SPARK_STATUS_OK ? 0 : -4);
}

static void CudaRemoveTree(const char *directory)
{
	char path[1200];
	struct dirent *item;
	DIR *listing = opendir(directory);
	if ( listing == 0 )
		return;
	while ( (item = readdir(listing)) != 0 )
	{
		if ( item->d_name[0] == '.' && (item->d_name[1] == '\0' || (item->d_name[1] == '.' && item->d_name[2] == '\0')) )
			continue;
		snprintf(path,sizeof(path),"%s/%s",directory,item->d_name);
		unlink(path);
	}
	closedir(listing);
	rmdir(directory);
}

int main(int argc,char **argv)
{
	static CudaEngine source,fresh,cold;
	static float logits_source[CUDA_VOCAB],logits_restored[CUDA_VOCAB],logits_cold[CUDA_VOCAB];
	SparkKvSnapshotStore store;
	char directory[512];
	uint64_t prefill_ns = 0u,restore_ns = 0u,cold_ns = 0u,read_bytes;
	int32_t result;
	uint32_t index,differences = 0u;
	if ( argc != 2 || argv[1][0] != '/' )
	{
		fprintf(stderr,"usage: %s ABSOLUTE_PARENT_DIRECTORY\n",argv[0]);
		return(2);
	}
	snprintf(directory,sizeof(directory),"%s/kv-snapshot-cuda-XXXXXX",argv[1]);
	if ( mkdtemp(directory) == 0 )
		return(2);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1ull << 32u,1ull << 30u) == SPARK_STATUS_OK);
	CHECK(CudaEngineOpen(&source,&store) == 0);
	result = CudaPrefillPrompt(&source,logits_source,&prefill_ns);
	CHECK(result == 0 && SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(source.snapshot.save_page_count == CUDA_PROMPT_PAGES && source.snapshot.save_failure_count == 0u && store.file_count == CUDA_PROMPT_PAGES && store.write_failure_count == 0u);
	CudaEngineClose(&source);
	SparkKvSnapshotStoreClose(&store);

	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1ull << 32u,1ull << 30u) == SPARK_STATUS_OK && store.file_count == CUDA_PROMPT_PAGES);
	CHECK(CudaEngineOpen(&fresh,&store) == 0);
	result = CudaRestorePrompt(&fresh,logits_restored,&restore_ns);
	CHECK(result == 0);
	CHECK(fresh.snapshot.restore_count == 1u && fresh.snapshot.restore_page_count == CUDA_PROMPT_PAGES);
	for (index=0u; index<CUDA_VOCAB; index++)
		differences += memcmp(&logits_source[index],&logits_restored[index],sizeof(float)) != 0 ? 1u : 0u;
	CHECK(differences == 0u);
	read_bytes = store.read_bytes;
	CudaEngineClose(&fresh);
	SparkKvSnapshotStoreClose(&store);

	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1ull << 32u,1ull << 30u) == SPARK_STATUS_OK);
	CHECK(CudaEngineOpen(&cold,&store) == 0);
	cold.snapshot.layout_sha256[0] ^= 1u;
	result = CudaPrefillPrompt(&cold,logits_cold,&cold_ns);
	CHECK(result == 0 && memcmp(logits_cold,logits_source,sizeof(logits_cold)) == 0);
	CudaEngineClose(&cold);
	SparkKvSnapshotStoreClose(&store);
	CudaRemoveTree(directory);
	printf("test_kv_snapshot_cuda: prompt_tokens=%u pages=%u snapshot_bytes_per_page=%u logits=%u bitwise_differences=%u prefill_with_save_us=%llu restore_us=%llu read_bytes=%llu checks=%u failures=%u\n",CUDA_PROMPT_TOKENS,CUDA_PROMPT_PAGES,CUDA_PAGE_BYTES + CUDA_STATE_BYTES,CUDA_VOCAB,differences,(unsigned long long)(prefill_ns / 1000u),(unsigned long long)(restore_ns / 1000u),(unsigned long long)read_bytes,cuda_checks,cuda_failures);
	return(cuda_failures == 0u ? 0 : 1);
}
