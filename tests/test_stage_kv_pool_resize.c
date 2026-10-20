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
#define TEST_POOL_CHUNKS 17u
#define TEST_MINIMUM_CHUNKS 4u
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

typedef struct TestPoolView
{
	uint32_t chunks;
	uint32_t pages;
	uint64_t grows;
	uint64_t shrinks;
	uint64_t vacated;
} TestPoolView;

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
	strcpy(engine->backing,"/tmp/sparkpipe-kv-resize-XXXXXX");
	assert(mkdtemp(engine->backing) != 0);
	strcpy(engine->snapshot,"/tmp/sparkpipe-kv-resize-snap-XXXXXX");
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
	configuration->model_id = "kvresize";
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

static TestPoolView EngineView(TestEngine *engine)
{
	TestPoolView view;
	assert(pthread_mutex_lock(&engine->binding.mutex) == 0);
	view.chunks = (uint32_t)(engine->binding.pool_mapped_bytes / TEST_CHUNK_BYTES);
	view.pages = engine->binding.arena.resident_block_capacity;
	view.grows = engine->binding.pool_grow_count;
	view.shrinks = engine->binding.pool_shrink_count;
	view.vacated = engine->binding.pool_vacated_pages;
	assert(pthread_mutex_unlock(&engine->binding.mutex) == 0);
	return(view);
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

static uint64_t CommittedChunks(void)
{
	return(SparkWeightdServerKvCommittedBytes(TestKvServer) / TEST_CHUNK_BYTES);
}

static void TestLayoutAndAttach(void)
{
	TestPoolView a;
	assert(SparkStageKvBindingInitialize(&ENGINE_A.binding,&ENGINE_A.configuration) == SPARK_STATUS_OK);
	assert(ENGINE_A.binding.kv_pool.chunk_bytes == TEST_CHUNK_BYTES && ENGINE_A.binding.kv_pool.chunk_capacity == TEST_POOL_CHUNKS);
	assert(ENGINE_A.binding.kv_pool_minimum_chunks == TEST_MINIMUM_CHUNKS);
	assert(ENGINE_A.binding.kv_pool_chunk_need[0] == 0u && ENGINE_A.binding.kv_pool_chunk_need[TEST_MINIMUM_CHUNKS - 1u] <= ENGINE_A.binding.pages_per_sequence);
	assert(ENGINE_A.binding.kv_pool_chunk_need[TEST_MINIMUM_CHUNKS] > ENGINE_A.binding.pages_per_sequence);
	a = EngineView(&ENGINE_A);
	assert(a.chunks == TEST_POOL_CHUNKS && a.pages == TEST_PHYSICAL && CommittedChunks() == TEST_POOL_CHUNKS);
	printf("pool layout: the page table maps first, region chunks map in page order so any granted prefix of chunks serves a page prefix, the minimum is one full sequence, an uncontended pool takes its whole size\n");
}

static void TestSecondEngineReclaimsMinimum(void)
{
	TestPoolView a,b;
	assert(SparkStageKvBindingInitialize(&ENGINE_B.binding,&ENGINE_B.configuration) == SPARK_STATUS_OK);
	a = EngineView(&ENGINE_A);
	b = EngineView(&ENGINE_B);
	assert(b.chunks >= TEST_MINIMUM_CHUNKS && b.pages >= ENGINE_B.binding.pages_per_sequence);
	assert(a.chunks < TEST_POOL_CHUNKS && a.pages < TEST_PHYSICAL && a.shrinks >= 1u);
	assert(a.chunks + b.chunks <= TEST_POOL_CHUNKS && CommittedChunks() <= TEST_POOL_CHUNKS);
	printf("second engine: the node reserve is full, the new pool waits for its minimum, the idle pool sees the wanted chunks and shrinks, the new engine starts\n");
}

static void PressUntilLent(TestEngine *busy,TestEngine *idle,uint8_t seed,TestPoolView *busy_after,TestPoolView *idle_after)
{
	TestPoolView busy_before = EngineView(busy),idle_before = EngineView(idle);
	uint32_t round;
	for (round=0u; round<TEST_WAIT_MS / 10u; round++)
	{
		EnginePublish(busy,(uint8_t)(seed + round));
		*busy_after = EngineView(busy);
		*idle_after = EngineView(idle);
		if ( idle_after->shrinks > idle_before.shrinks && busy_after->pages > busy_before.pages )
			return;
		SleepMs(10u);
	}
	assert(0);
}

static void TestBusyEngineBorrowsFromIdle(void)
{
	TestPoolView a_before,b_before,a,b;
	a_before = EngineView(&ENGINE_A);
	b_before = EngineView(&ENGINE_B);
	PressUntilLent(&ENGINE_B,&ENGINE_A,0x40u,&b,&a);
	assert(b.grows > b_before.grows && b.chunks > b_before.chunks);
	assert(a.chunks < a_before.chunks && a.pages < a_before.pages && a.chunks >= TEST_MINIMUM_CHUNKS);
	assert(CommittedChunks() <= TEST_POOL_CHUNKS);
	SleepMs(400u);
	a_before = EngineView(&ENGINE_A);
	b_before = EngineView(&ENGINE_B);
	PressUntilLent(&ENGINE_A,&ENGINE_B,0x90u,&a,&b);
	assert(a.grows > a_before.grows && a.chunks > a_before.chunks);
	assert(b.chunks < b_before.chunks && b.pages < b_before.pages && b.chunks >= TEST_MINIMUM_CHUNKS);
	assert(b.vacated > b_before.vacated);
	assert(CommittedChunks() <= TEST_POOL_CHUNKS);
	printf("two engines under load: the engine that evicts grows, the idle engine vacates its top pages (parked or discarded) and shrinks, the node reserve is never exceeded, and the roles reverse when the load moves\n");
}

static void TestVacatedPagesStillServe(void)
{
	TestPoolView b;
	uint32_t round;
	b = EngineView(&ENGINE_B);
	for (round=0u; round<8u; round++)
		EnginePublish(&ENGINE_B,(uint8_t)(0xd0u + round));
	assert(EngineView(&ENGINE_B).pages >= b.pages && ENGINE_B.binding.arena.park_failure_count == 0u);
	printf("after a shrink the smaller pool keeps serving: new pages admit by evicting within the granted limit\n");
}

int main(void)
{
	TestKvServerStart((uint64_t)TEST_POOL_CHUNKS * TEST_CHUNK_BYTES);
	EngineConfigure(&ENGINE_A,"kvpoola");
	EngineConfigure(&ENGINE_B,"kvpoolb");
	TestLayoutAndAttach();
	TestSecondEngineReclaimsMinimum();
	TestBusyEngineBorrowsFromIdle();
	TestVacatedPagesStillServe();
	EngineClose(&ENGINE_B);
	EngineClose(&ENGINE_A);
	TestKvServerFinish();
	printf("stage kv pool resize: ok\n");
	return(0);
}
