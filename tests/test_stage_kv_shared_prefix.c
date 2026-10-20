#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <ftw.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "cuda_runtime_api.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_weight_codec.h"
#include "tests/test_weightd_kv_server.h"

#define TEST_BLOCK 4u
#define TEST_POSITIONS 8u
#define TEST_LANES 2u
#define TEST_SLOTS 2u
#define TEST_PHYSICAL 16u
#define TEST_LOGICAL 32u
#define TEST_LAYER_BYTES 16384u
#define TEST_LAYERS 2u
#define TEST_PAGE_BYTES (2u * TEST_LAYERS * TEST_LAYER_BYTES)
#define TEST_CHUNK_BYTES 65536u
#define TEST_WINDOW_PAGES 8u
#define TEST_RESERVE_CHUNKS 64u
#define TEST_WAIT_MS 10000u

typedef struct TestEngine
{
	SparkStageKvBinding binding;
	SparkStageKvConfiguration configuration;
	char backing[64];
	char snapshot[64];
	uint64_t next_sequence;
	uint64_t next_id;
} TestEngine;

typedef struct TestFinished
{
	atomic_uint count;
	SparkStatus status;
} TestFinished;

static TestEngine ENGINE_A;
static TestEngine ENGINE_B;

static void SleepMs(uint32_t milliseconds)
{
	struct timespec delay = { (time_t)(milliseconds / 1000u), (long)(milliseconds % 1000u) * 1000000L };
	(void)nanosleep(&delay,0);
}

static void EngineConfigure(TestEngine *engine,const char *tag)
{
	SparkStageKvConfiguration *configuration = &engine->configuration;
	memset(engine,0,sizeof(*engine));
	strcpy(engine->backing,"/tmp/sparkpipe-kv-shared-XXXXXX");
	assert(mkdtemp(engine->backing) != 0);
	strcpy(engine->snapshot,"/tmp/sparkpipe-kv-shared-snap-XXXXXX");
	assert(mkdtemp(engine->snapshot) != 0);
	configuration->module_tag = tag;
	configuration->block_token_count = TEST_BLOCK;
	configuration->region_count = 2u;
	configuration->regions[0].layout = SPARK_STAGE_KV_REGION_PAGE_MAJOR;
	configuration->regions[0].layer_count = TEST_LAYERS;
	configuration->regions[0].layer_page_bytes = TEST_LAYER_BYTES;
	configuration->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	configuration->regions[1].layer_count = TEST_LAYERS;
	configuration->regions[1].layer_page_bytes = TEST_LAYER_BYTES;
	configuration->arena_kv_head_count = 1u;
	configuration->arena_head_dim = TEST_LAYER_BYTES / TEST_BLOCK;
	configuration->arena_bytes_per_scalar = 1u;
	configuration->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	configuration->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	configuration->model_id = "kvshared";
	configuration->model_revision = "fixture";
	configuration->resident_sequence_capacity = TEST_LANES;
	configuration->max_sequence_positions = TEST_POSITIONS;
	configuration->max_input_row_count = TEST_POSITIONS;
	configuration->logical_page_count = TEST_LOGICAL;
	configuration->physical_page_count = TEST_PHYSICAL;
	configuration->pipeline_slot_count = TEST_SLOTS;
	configuration->backing_directory = engine->backing;
	configuration->backing_maximum_bytes = (uint64_t)(TEST_LOGICAL - TEST_PHYSICAL + 2u) * TEST_PAGE_BYTES;
	configuration->snapshot_directory = engine->snapshot;
	configuration->snapshot_maximum_bytes = 1ull << 26;
	memset(configuration->pack_sha256,0x11,sizeof(configuration->pack_sha256));
	memset(configuration->contract_sha256,0x22,sizeof(configuration->contract_sha256));
	configuration->expert_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	configuration->kv_codec = SPARK_WEIGHT_CODEC_BF16;
	configuration->driver_symbol = (const void *)&EngineConfigure;
	engine->next_sequence = 1u;
	engine->next_id = 1u;
}

static int RemoveEntry(const char *path,const struct stat *info,int flag,struct FTW *walk)
{
	(void)info;
	(void)flag;
	(void)walk;
	return(remove(path));
}

static void EngineClose(TestEngine *engine)
{
	SparkStageKvBindingDestroy(&engine->binding);
	assert(nftw(engine->backing,RemoveEntry,16,FTW_DEPTH | FTW_PHYS) == 0);
	assert(nftw(engine->snapshot,RemoveEntry,16,FTW_DEPTH | FTW_PHYS) == 0);
}

static void Finished(void *context,SparkStatus status)
{
	TestFinished *finished = context;
	finished->status = status;
	atomic_fetch_add(&finished->count,1u);
}

static void EnginePublish(TestEngine *engine,uint8_t seed)
{
	SparkModelDriverCacheLane lane;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverFrame frame;
	SparkStageKvBindingCompletion completion;
	TestFinished finished = {0};
	uint64_t sequence = engine->next_sequence++,next = TEST_BLOCK;
	uint32_t slot = 0u,index;
	uint8_t bound = 1u;
	SparkStatus status = SPARK_STATUS_BUSY;
	memset(&lane,0,sizeof(lane));
	memset(&request,0,sizeof(request));
	lane.sequence_id = sequence;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.context_token_count = TEST_BLOCK;
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	lane.publish_token_count = TEST_BLOCK;
	for (index=0u; index<sizeof(lane.publish_identity.sha256); index++)
		lane.publish_identity.sha256[index] = (uint8_t)(seed + index * 7u);
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.request_id = request.submission_id = request.transaction_id = engine->next_id++;
	request.control_generation = request.request_generation = request.step_generation = 1u;
	request.active_slot_count = 1u;
	request.new_token_count = 1u;
	request.cache_lane_count = 1u;
	request.cache_lanes = &lane;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	for (index=0u; index<TEST_WAIT_MS; index++)
	{
		status = SparkStageKvBindingAdmit(&engine->binding,&request,&decision);
		if ( status != SPARK_STATUS_BUSY && status != SPARK_STATUS_PENDING )
			break;
		SleepMs(1u);
	}
	assert(status == SPARK_STATUS_OK);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	assert(SparkStageKvBindingAdmit(&engine->binding,&request,&decision) == SPARK_STATUS_OK);
	memset(&frame,0,sizeof(frame));
	frame.program_id = request.program_id;
	frame.request_id = request.request_id;
	frame.active_slot_count = 1u;
	frame.new_token_count = 1u;
	frame.cache_lane_count = 1u;
	frame.cache_lanes = &lane;
	frame.driver_dispatch_slot = decision.driver_dispatch_slot;
	frame.driver_dispatch_generation = request.control_generation;
	frame.driver_dispatch_cookie0 = request.transaction_id;
	frame.driver_dispatch_cookie1 = request.submission_id;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
	{
		uint64_t position = 0u;
		assert(SparkStageKvBindingClaim(&engine->binding,&frame,1u,&slot,&sequence,&position,&next) == SPARK_STATUS_OK);
	}
	memset(&completion,0,sizeof(completion));
	completion.lane_count = 1u;
	completion.status = SPARK_STATUS_OK;
	completion.resident_slots = &slot;
	completion.bound = &bound;
	completion.sequence_ids = &sequence;
	completion.next_positions = &next;
	completion.finished_function = Finished;
	completion.finished_context = &finished;
	assert(SparkStageKvBindingFinishAsync(&engine->binding,decision.driver_dispatch_slot,&completion) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&engine->binding,1000000000ull) == SPARK_STATUS_OK);
	assert(atomic_load(&finished.count) == 1u && finished.status == SPARK_STATUS_OK);
	memset(&lane,0,sizeof(lane));
	memset(&request,0,sizeof(request));
	lane.sequence_id = sequence;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.request_id = request.submission_id = request.transaction_id = engine->next_id++;
	request.control_generation = request.request_generation = request.step_generation = 1u;
	request.active_slot_count = 1u;
	request.cache_lane_count = 1u;
	request.cache_lanes = &lane;
	request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	assert(SparkStageKvBindingAdmit(&engine->binding,&request,&decision) == SPARK_STATUS_OK);
}

static uint32_t EngineEntryWindowSlot(TestEngine *engine,uint8_t seed,uint32_t token_count)
{
	SparkStageKvBinding *binding = &engine->binding;
	uint32_t entry,index;
	for (entry=0u; entry<binding->page_cache.entry_capacity; entry++)
	{
		const SparkKvPageCacheEntry *candidate = &binding->page_cache.entries[entry];
		uint32_t match = (candidate->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && candidate->token_count == token_count ? 1u : 0u;
		for (index=0u; match != 0u && index<sizeof(candidate->identity.sha256); index++)
			match = candidate->identity.sha256[index] == (uint8_t)(seed + index * 7u) ? 1u : 0u;
		if ( match != 0u )
		{
			const SparkKvCacheBlock *block = &binding->arena.blocks[candidate->logical_page_index];
			assert((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_SHARED) != 0u && block->resident_slot_index >= binding->physical_page_count);
			return(block->resident_slot_index - binding->physical_page_count);
		}
	}
	assert(0);
	return(UINT32_MAX);
}

static void TestBothEnginesMapOneWindow(void)
{
	assert(SparkStageKvBindingInitialize(&ENGINE_A.binding,&ENGINE_A.configuration) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingInitialize(&ENGINE_B.binding,&ENGINE_B.configuration) == SPARK_STATUS_OK);
	assert(ENGINE_A.binding.shared_page_count == TEST_WINDOW_PAGES && ENGINE_B.binding.shared_page_count == TEST_WINDOW_PAGES);
	assert(ENGINE_A.binding.kv_shared.created == 1u && ENGINE_B.binding.kv_shared.created == 0u);
	assert(ENGINE_A.binding.kv_shared.pool_generation == ENGINE_B.binding.kv_shared.pool_generation);
	assert(ENGINE_A.binding.kv_shared.holder != ENGINE_B.binding.kv_shared.holder);
	assert(ENGINE_A.binding.kv_shared.mapped_count == ENGINE_A.binding.kv_shared.chunk_count && ENGINE_B.binding.kv_shared.mapped_count == ENGINE_B.binding.kv_shared.chunk_count);
	assert(ENGINE_A.binding.physical_page_count == TEST_PHYSICAL && ENGINE_A.binding.region_layer_stride_bytes[1] == (uint64_t)(TEST_PHYSICAL + TEST_WINDOW_PAGES) * TEST_LAYER_BYTES);
	assert(SparkStageKvBindingAddressablePageCount(&ENGINE_A.binding) == TEST_PHYSICAL + TEST_WINDOW_PAGES);
	assert(SparkWeightdServerKvPoolCount(TestKvServer) == 3u);
	printf("two engines of one layout map one weightd window after their private pages, in every region and every layer, with distinct holders; the addressable page count drivers bound their KV views with covers the window\n");
}

static void TestSecondEngineServesThePublishedPrefix(void)
{
	SparkModelDriverCacheLane lane;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status = SPARK_STATUS_BUSY;
	uint32_t slot,index,physical;
	uint64_t holders;
	EnginePublish(&ENGINE_A,0x60u);
	assert(ENGINE_A.binding.page_cache.shared_allocated_count >= 1u && ENGINE_A.binding.page_cache.shared_published_count == 1u);
	slot = EngineEntryWindowSlot(&ENGINE_A,0x60u,TEST_BLOCK);
	memset(&lane,0,sizeof(lane));
	memset(&request,0,sizeof(request));
	lane.sequence_id = 77u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = 1u;
	lane.sequence_position = TEST_BLOCK;
	lane.context_token_count = TEST_BLOCK + 1u;
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane.prefix_token_count = TEST_BLOCK;
	for (index=0u; index<sizeof(lane.prefix_identity.sha256); index++)
		lane.prefix_identity.sha256[index] = (uint8_t)(0x60u + index * 7u);
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.request_id = request.submission_id = request.transaction_id = ENGINE_B.next_id++;
	request.control_generation = request.request_generation = request.step_generation = 1u;
	request.active_slot_count = 1u;
	request.new_token_count = 1u;
	request.cache_lane_count = 1u;
	request.cache_lanes = &lane;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	for (index=0u; index<TEST_WAIT_MS; index++)
	{
		status = SparkStageKvBindingAdmit(&ENGINE_B.binding,&request,&decision);
		if ( status != SPARK_STATUS_BUSY && status != SPARK_STATUS_PENDING )
			break;
		SleepMs(1u);
	}
	assert(status == SPARK_STATUS_OK && decision.accepted == 1u);
	assert(ENGINE_B.binding.page_cache.shared_import_count == 1u && ENGINE_B.binding.page_cache.shared_imported_page_count == 1u);
	assert(EngineEntryWindowSlot(&ENGINE_B,0x60u,TEST_BLOCK) == slot);
	physical = ENGINE_B.binding.physical_pages[(uint64_t)lane.resident_sequence_slot * ENGINE_B.binding.pages_per_sequence];
	assert(physical == ENGINE_B.binding.physical_page_count + slot);
	assert(physical >= ENGINE_B.binding.physical_page_count && physical < SparkStageKvBindingAddressablePageCount(&ENGINE_B.binding));
	holders = atomic_load(&ENGINE_B.binding.shared_index.slots[slot].holders);
	assert(holders == (SparkKvSharedIndexBit(ENGINE_A.binding.kv_shared.holder) | SparkKvSharedIndexBit(ENGINE_B.binding.kv_shared.holder)));
	assert(ENGINE_B.binding.page_cache.shared_allocated_count == 1u && ENGINE_B.binding.arena.shared_block_count == 2u);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	assert(SparkStageKvBindingAdmit(&ENGINE_B.binding,&request,&decision) == SPARK_STATUS_OK);
	printf("the second engine's prefix lane imports the first engine's published window page: its page table points at the same window slot, no private page holds the prefix\n");
}

int main(void)
{
	TestKvServerSharedWindowBytes = (uint64_t)TEST_WINDOW_PAGES * TEST_PAGE_BYTES;
	TestKvServerStart((uint64_t)TEST_RESERVE_CHUNKS * TEST_CHUNK_BYTES);
	EngineConfigure(&ENGINE_A,"kvshareda");
	EngineConfigure(&ENGINE_B,"kvsharedb");
	TestBothEnginesMapOneWindow();
	TestSecondEngineServesThePublishedPrefix();
	EngineClose(&ENGINE_B);
	EngineClose(&ENGINE_A);
	assert(SparkWeightdServerKvPoolCount(TestKvServer) >= 2u);
	TestKvServerFinish();
	printf("stage kv shared prefix: ok\n");
	return(0);
}
