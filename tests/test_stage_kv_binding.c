#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ftw.h>
#include <unistd.h>

#include "cuda_runtime_api.h"
#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_weight_codec.h"

#define TEST_BLOCK 4u
#define TEST_POSITIONS 8u
#define TEST_LANES 2u
#define TEST_SLOTS 2u
#define TEST_PHYSICAL 4u
#define TEST_LOGICAL 8u
#define TEST_REGION0_LAYER 64u
#define TEST_REGION1_LAYER 32u
#define TEST_LAYERS 2u
#define TEST_PAGE_BYTES (TEST_LAYERS * TEST_REGION0_LAYER + TEST_LAYERS * TEST_REGION1_LAYER)

typedef struct TestStep
{
	SparkModelDriverCacheLane lane;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverFrame frame;
	uint32_t slot;
	uint64_t sequence;
	uint64_t position;
	uint64_t next;
	uint8_t bound;
} TestStep;

typedef struct TestFinished
{
	atomic_uint count;
	SparkStatus status;
	uint32_t order;
} TestFinished;

typedef struct TestHook
{
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	uint32_t block_kind_mask;
	uint32_t active;
	uint32_t release;
	uint64_t sleep_ns;
	uint32_t sleep_kind_mask;
	uint32_t device_copies;
	uint32_t foreign_stream_copies;
	uint32_t save_thread_copies;
	uint32_t other_thread_copies;
	pthread_t save_thread;
	uint32_t track_threads;
} TestHook;

static SparkStageModuleLedger LEDGER;
static SparkStageKvBinding BINDING;
static char DIRECTORY[64];
static char SNAPSHOT_DIRECTORY[64];
static SparkStageKvConfiguration CONFIGURATION;
static uint64_t NEXT_ID = 1u;
static TestHook HOOK = { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
static atomic_uint ORDER;

static uint32_t KindBit(cudaMemcpyKind kind)
{
	return(1u << (uint32_t)kind);
}

static void TestHookCopy(void *context,uint32_t asynchronous,cudaMemcpyKind kind,cudaStream_t stream)
{
	TestHook *hook = context;
	struct timespec delay;
	assert(pthread_mutex_lock(&hook->mutex) == 0);
	if ( kind == cudaMemcpyDeviceToDevice )
		hook->device_copies++;
	if ( asynchronous != 0u && stream != (cudaStream_t)BINDING.copy_stream )
		hook->foreign_stream_copies++;
	if ( hook->track_threads != 0u )
	{
		if ( pthread_equal(pthread_self(),hook->save_thread) != 0 )
			hook->save_thread_copies++;
		else
			hook->other_thread_copies++;
	}
	if ( (hook->block_kind_mask & KindBit(kind)) != 0u && hook->active == 0u )
	{
		hook->active = 1u;
		assert(pthread_cond_broadcast(&hook->condition) == 0);
		while ( hook->release == 0u )
			assert(pthread_cond_wait(&hook->condition,&hook->mutex) == 0);
	}
	if ( (hook->sleep_kind_mask & KindBit(kind)) != 0u && hook->sleep_ns != 0u )
	{
		delay.tv_sec = (time_t)(hook->sleep_ns / 1000000000ull);
		delay.tv_nsec = (long)(hook->sleep_ns % 1000000000ull);
		hook->sleep_ns = 0u;
		(void)nanosleep(&delay,0);
	}
	assert(pthread_mutex_unlock(&hook->mutex) == 0);
}

static void HookReset(void)
{
	assert(pthread_mutex_lock(&HOOK.mutex) == 0);
	HOOK.block_kind_mask = 0u;
	HOOK.active = 0u;
	HOOK.release = 0u;
	HOOK.sleep_ns = 0u;
	HOOK.sleep_kind_mask = 0u;
	HOOK.device_copies = 0u;
	HOOK.foreign_stream_copies = 0u;
	HOOK.save_thread_copies = 0u;
	HOOK.other_thread_copies = 0u;
	HOOK.track_threads = 0u;
	assert(pthread_mutex_unlock(&HOOK.mutex) == 0);
	spark_stub_cuda_reset_copy_calls();
}

static void HookWaitActive(void)
{
	assert(pthread_mutex_lock(&HOOK.mutex) == 0);
	while ( HOOK.active == 0u )
		assert(pthread_cond_wait(&HOOK.condition,&HOOK.mutex) == 0);
	assert(pthread_mutex_unlock(&HOOK.mutex) == 0);
}

static void HookRelease(void)
{
	assert(pthread_mutex_lock(&HOOK.mutex) == 0);
	HOOK.release = 1u;
	assert(pthread_cond_broadcast(&HOOK.condition) == 0);
	assert(pthread_mutex_unlock(&HOOK.mutex) == 0);
}

static uint64_t NowNs(void)
{
	struct timespec now;
	assert(clock_gettime(CLOCK_MONOTONIC,&now) == 0);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static void SleepMs(uint32_t milliseconds)
{
	struct timespec delay = { (time_t)(milliseconds / 1000u), (long)(milliseconds % 1000u) * 1000000L };
	(void)nanosleep(&delay,0);
}

static void Configure(SparkStageKvConfiguration *configuration)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->ledger = &LEDGER;
	configuration->module_tag = "kvtest";
	configuration->block_token_count = TEST_BLOCK;
	configuration->region_count = 2u;
	configuration->regions[0].layout = SPARK_STAGE_KV_REGION_PAGE_MAJOR;
	configuration->regions[0].layer_count = TEST_LAYERS;
	configuration->regions[0].layer_page_bytes = TEST_REGION0_LAYER;
	configuration->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	configuration->regions[1].layer_count = TEST_LAYERS;
	configuration->regions[1].layer_page_bytes = TEST_REGION1_LAYER;
	configuration->arena_kv_head_count = 1u;
	configuration->arena_head_dim = TEST_REGION0_LAYER / TEST_BLOCK;
	configuration->arena_bytes_per_scalar = 1u;
	configuration->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	configuration->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	configuration->model_id = "kvtest";
	configuration->model_revision = "fixture";
	configuration->resident_sequence_capacity = TEST_LANES;
	configuration->max_sequence_positions = TEST_POSITIONS;
	configuration->max_input_row_count = TEST_POSITIONS;
	configuration->logical_page_count = TEST_LOGICAL;
	configuration->physical_page_count = TEST_PHYSICAL;
	configuration->pipeline_slot_count = TEST_SLOTS;
	configuration->backing_directory = DIRECTORY;
	configuration->backing_maximum_bytes = (uint64_t)(TEST_LOGICAL - TEST_PHYSICAL + 2u) * TEST_PAGE_BYTES;
	configuration->snapshot_directory = SNAPSHOT_DIRECTORY;
	configuration->snapshot_maximum_bytes = 1ull << 26;
	memset(configuration->pack_sha256,0x11,sizeof(configuration->pack_sha256));
	memset(configuration->contract_sha256,0x22,sizeof(configuration->contract_sha256));
	configuration->expert_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	configuration->kv_codec = SPARK_WEIGHT_CODEC_BF16;
	configuration->driver_symbol = (const void *)&Configure;
}

static SparkStatus OpenWith(const SparkStageKvConfiguration *configuration)
{
	SparkStatus status;
	memset(&LEDGER,0,sizeof(LEDGER));
	LEDGER.module_tag = "kvtest";
	HookReset();
	spark_stub_cuda_event_pending(0u);
	spark_stub_cuda_event_record_failure(0u);
	spark_stub_cuda_set_copy_hook(TestHookCopy,&HOOK);
	status = SparkStageKvBindingInitialize(&BINDING,configuration);
	if ( status == SPARK_STATUS_OK )
		assert(BINDING.page_bytes == TEST_PAGE_BYTES && BINDING.pages_per_sequence == 2u && BINDING.copier_initialized != 0u && BINDING.page_cache.snapshot != 0);
	return(status);
}

static void MakeDirectories(void)
{
	strcpy(DIRECTORY,"/tmp/sparkpipe-kv-binding-XXXXXX");
	assert(mkdtemp(DIRECTORY) != 0);
	strcpy(SNAPSHOT_DIRECTORY,"/tmp/sparkpipe-kv-snapshot-XXXXXX");
	assert(mkdtemp(SNAPSHOT_DIRECTORY) != 0);
}

static int RemoveEntry(const char *path,const struct stat *info,int flag,struct FTW *walk)
{
	(void)info;
	(void)flag;
	(void)walk;
	return(remove(path));
}

static void RemoveDirectories(void)
{
	assert(rmdir(DIRECTORY) == 0);
	assert(nftw(SNAPSHOT_DIRECTORY,RemoveEntry,16,FTW_DEPTH | FTW_PHYS) == 0);
}

static void Open(void)
{
	MakeDirectories();
	Configure(&CONFIGURATION);
	assert(OpenWith(&CONFIGURATION) == SPARK_STATUS_OK);
}

static void Unload(void)
{
	SparkStageKvBindingDestroy(&BINDING);
	SparkStageModuleLedgerRollback(&LEDGER,0u);
	spark_stub_cuda_set_copy_hook(0,0);
}

static void Close(void)
{
	Unload();
	RemoveDirectories();
}

static void Identity(SparkModelDriverCacheIdentity *identity,uint8_t seed)
{
	uint32_t index;
	for (index=0u; index<sizeof(identity->sha256); index++)
		identity->sha256[index] = (uint8_t)(seed + index);
}

static void StepInit(TestStep *step,uint64_t sequence,uint32_t slot,uint32_t position,uint32_t context)
{
	uint64_t id = NEXT_ID++;
	memset(step,0,sizeof(*step));
	step->lane.sequence_id = sequence;
	step->lane.request_generation = 1u;
	step->lane.step_generation = position + 1u;
	step->lane.resident_sequence_slot = slot;
	step->lane.sequence_position = position;
	step->lane.context_token_count = context;
	step->slot = slot;
	step->sequence = sequence;
	step->position = position;
	step->next = context;
	step->bound = 1u;
	step->request.descriptor_bytes = sizeof(step->request);
	step->request.program_id = 1u;
	step->request.request_id = step->request.submission_id = step->request.transaction_id = id;
	step->request.control_generation = step->request.request_generation = step->request.step_generation = 1u;
	step->request.active_slot_count = 1u;
	step->request.new_token_count = 1u;
	step->request.cache_lane_count = 1u;
	step->request.cache_lanes = &step->lane;
}

static void StepPublish(TestStep *step,uint32_t tokens,uint8_t seed)
{
	step->lane.flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	step->lane.publish_token_count = tokens;
	Identity(&step->lane.publish_identity,seed);
}

static void StepPrefix(TestStep *step,uint32_t tokens,uint8_t seed)
{
	step->lane.flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	step->lane.prefix_token_count = tokens;
	Identity(&step->lane.prefix_identity,seed);
}

static void StepRelease(TestStep *step,uint64_t sequence,uint32_t slot)
{
	StepInit(step,sequence,slot,0u,0u);
	step->lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	step->request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	step->request.new_token_count = 0u;
}

static SparkStatus StepAdmit(TestStep *step,uint32_t flags)
{
	step->request.admission_flags = flags;
	return(SparkStageKvBindingAdmit(&BINDING,&step->request,&step->decision));
}

static SparkStatus StepClaim(TestStep *step)
{
	memset(&step->frame,0,sizeof(step->frame));
	step->frame.program_id = step->request.program_id;
	step->frame.request_id = step->request.request_id;
	step->frame.active_slot_count = step->request.active_slot_count;
	step->frame.new_token_count = step->request.new_token_count;
	step->frame.cache_lane_count = step->request.cache_lane_count;
	step->frame.cache_lanes = step->request.cache_lanes;
	step->frame.driver_dispatch_slot = step->decision.driver_dispatch_slot;
	step->frame.driver_dispatch_generation = step->request.control_generation;
	step->frame.driver_dispatch_cookie0 = step->request.transaction_id;
	step->frame.driver_dispatch_cookie1 = step->request.submission_id;
	step->frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
	return(SparkStageKvBindingClaim(&BINDING,&step->frame,1u,&step->slot,&step->sequence,&step->position,&step->next));
}

static void StepStart(TestStep *step)
{
	assert(StepAdmit(step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(StepAdmit(step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
	assert(StepClaim(step) == SPARK_STATUS_OK);
}

static void Finished(void *context,SparkStatus status)
{
	TestFinished *finished = context;
	finished->status = status;
	finished->order = atomic_fetch_add(&ORDER,1u);
	atomic_fetch_add(&finished->count,1u);
}

static SparkStatus StepFinish(TestStep *step,SparkStatus status,TestFinished *finished)
{
	SparkStageKvBindingCompletion completion;
	memset(&completion,0,sizeof(completion));
	completion.lane_count = 1u;
	completion.status = status;
	completion.resident_slots = &step->slot;
	completion.bound = &step->bound;
	completion.sequence_ids = &step->sequence;
	completion.next_positions = &step->next;
	completion.finished_function = Finished;
	completion.finished_context = finished;
	return(SparkStageKvBindingFinishAsync(&BINDING,step->decision.driver_dispatch_slot,&completion));
}

static uint32_t LanePage(uint32_t slot,uint32_t index)
{
	return(BINDING.logical_pages[(uint64_t)slot * BINDING.pages_per_sequence + index]);
}

static uint8_t *Region0(uint32_t page)
{
	assert((BINDING.blocks[page].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
	return(BINDING.region_base[0] + (uint64_t)BINDING.blocks[page].resident_slot_index * BINDING.region_packed_page_bytes[0]);
}

static uint8_t *Region1(uint32_t page,uint32_t layer)
{
	assert((BINDING.blocks[page].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
	return(BINDING.region_base[1] + (uint64_t)layer * BINDING.region_layer_stride_bytes[1] + (uint64_t)BINDING.blocks[page].resident_slot_index * TEST_REGION1_LAYER);
}

static void FillPage(uint32_t page,uint8_t seed)
{
	uint32_t index,layer;
	for (index=0u; index<TEST_LAYERS * TEST_REGION0_LAYER; index++)
		Region0(page)[index] = (uint8_t)(seed + index);
	for (layer=0u; layer<TEST_LAYERS; layer++)
		for (index=0u; index<TEST_REGION1_LAYER; index++)
			Region1(page,layer)[index] = (uint8_t)(seed * 3u + layer * 41u + index);
}

static void PageBytes(uint32_t page,uint8_t *bytes)
{
	uint32_t layer;
	memcpy(bytes,Region0(page),TEST_LAYERS * TEST_REGION0_LAYER);
	for (layer=0u; layer<TEST_LAYERS; layer++)
		memcpy(bytes + TEST_LAYERS * TEST_REGION0_LAYER + layer * TEST_REGION1_LAYER,Region1(page,layer),TEST_REGION1_LAYER);
}

static uint32_t PublishPrefix(uint64_t sequence,uint32_t slot,uint32_t tokens,uint8_t identity,uint8_t seed)
{
	TestStep step,release;
	TestFinished finished = {0};
	uint32_t page;
	StepInit(&step,sequence,slot,0u,tokens);
	StepPublish(&step,tokens,identity);
	StepStart(&step);
	page = LanePage(slot,0u);
	FillPage(page,seed);
	assert(StepFinish(&step,SPARK_STATUS_OK,&finished) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK);
	assert(atomic_load(&finished.count) == 1u && finished.status == SPARK_STATUS_OK);
	StepRelease(&release,sequence,slot);
	assert(StepAdmit(&release,0u) == SPARK_STATUS_OK);
	return(page);
}

typedef struct TestAdmitThread
{
	TestStep *step;
	SparkStatus status;
	uint64_t returned_ns;
} TestAdmitThread;

static void *AdmitMain(void *context)
{
	TestAdmitThread *thread = context;
	thread->status = StepAdmit(thread->step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
	thread->returned_ns = NowNs();
	return(0);
}

static void TestCompletionEntryNeverWaits(void)
{
	TestStep x,y;
	TestFinished finished = {0};
	TestAdmitThread thread;
	pthread_t admit;
	uint64_t start,elapsed;
	Open();
	(void)PublishPrefix(3u,1u,2u,0x40u,0x11u);
	StepInit(&x,1u,0u,0u,4u);
	StepStart(&x);
	StepInit(&y,2u,1u,2u,3u);
	StepPrefix(&y,2u,0x40u);
	HookReset();
	HOOK.block_kind_mask = KindBit(cudaMemcpyDeviceToDevice);
	thread.step = &y;
	thread.status = SPARK_STATUS_INTERNAL_ERROR;
	assert(pthread_create(&admit,0,AdmitMain,&thread) == 0);
	HookWaitActive();
	start = NowNs();
	assert(StepFinish(&x,SPARK_STATUS_OK,&finished) == SPARK_STATUS_OK);
	elapsed = NowNs() - start;
	assert(elapsed < 50000000ull);
	SleepMs(20u);
	assert(atomic_load(&finished.count) == 0u);
	HookRelease();
	assert(pthread_join(admit,0) == 0);
	assert(thread.status == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK);
	assert(atomic_load(&finished.count) == 1u && finished.status == SPARK_STATUS_OK);
	assert(atomic_load(&BINDING.lane_next_positions[0]) == 4u && atomic_load(&BINDING.lane_bound[0]) == 1u);
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	Close();
	printf("T1 completion entry never waits: ok (entry %llu ns)\n",(unsigned long long)elapsed);
}

static void TestCopyOnWriteIsOneDeviceCopyPerRegion(void)
{
	TestStep y;
	uint8_t source_bytes[TEST_PAGE_BYTES],destination_bytes[TEST_PAGE_BYTES];
	uint32_t source,destination;
	cudaStream_t stream;
	Open();
	source = PublishPrefix(3u,1u,2u,0x40u,0x22u);
	StepInit(&y,2u,1u,2u,3u);
	StepPrefix(&y,2u,0x40u);
	HookReset();
	spark_stub_cuda_event_pending(1u);
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	destination = LanePage(1u,0u);
	assert(destination != source);
	assert(spark_stub_cuda_sync_copy_calls() == 0u);
	assert(spark_stub_cuda_async_copy_calls() == 2u && HOOK.device_copies == 2u && HOOK.foreign_stream_copies == 0u);
	PageBytes(source,source_bytes);
	PageBytes(destination,destination_bytes);
	assert(memcmp(source_bytes,destination_bytes,sizeof(source_bytes)) == 0);
	assert(BINDING.blocks[source].residency_reference_count == 1u && BINDING.blocks[destination].residency_reference_count == 2u);
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
	assert(BINDING.blocks[source].residency_reference_count == 1u && BINDING.blocks[destination].residency_reference_count == 2u);
	spark_stub_cuda_event_pending(0u);
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	assert(BINDING.blocks[source].residency_reference_count == 0u);
	assert((BINDING.blocks[destination].flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u);
	assert(cudaStreamCreate(&stream) == cudaSuccess);
	assert(SparkStageKvBindingFenceExecution(&BINDING,stream) == SPARK_STATUS_OK);
	assert(spark_stub_cuda_last_wait_stream() == stream && spark_stub_cuda_last_wait_event() == (cudaEvent_t)BINDING.copier.fence_event);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	Close();
	printf("T2 copy-on-write is one device copy per region: ok\n");
}

static void TestCopyOnWriteDuringAPark(void)
{
	TestStep x,w,y;
	TestFinished finished = {0};
	SparkStatus status;
	uint32_t source,resident;
	Open();
	StepInit(&x,1u,0u,0u,8u);
	StepStart(&x);
	FillPage(LanePage(0u,0u),0x31u);
	FillPage(LanePage(0u,1u),0x32u);
	assert(StepFinish(&x,SPARK_STATUS_OK,&finished) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK && finished.status == SPARK_STATUS_OK);
	source = PublishPrefix(3u,1u,2u,0x40u,0x33u);
	resident = BINDING.arena.resident_block_count;
	assert(resident == TEST_PHYSICAL - 1u);
	HookReset();
	HOOK.block_kind_mask = KindBit(cudaMemcpyDeviceToHost);
	StepInit(&w,4u,1u,0u,8u);
	status = StepAdmit(&w,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
	assert(status == SPARK_STATUS_BUSY);
	HookWaitActive();
	assert(BINDING.arena.resident_block_count == resident);
	StepInit(&y,2u,1u,2u,3u);
	StepPrefix(&y,2u,0x40u);
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(LanePage(1u,0u) != source && HOOK.device_copies == 2u);
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	HookRelease();
	assert(SparkKvPageStoreWaitForTransfers(&BINDING.page_store) == SPARK_STATUS_OK);
	Close();
	printf("T3 copy-on-write during a park: ok\n");
}

static void TestFinishAndReleaseOnlyMark(void)
{
	SparkKvPageCacheSnapshot *snapshot;
	SparkKvSnapshotKey key;
	TestStep x,release;
	TestFinished finished = {0};
	uint8_t expected[TEST_PAGE_BYTES],saved[TEST_PAGE_BYTES];
	uint32_t logical;
	uint64_t bytes;
	Open();
	snapshot = BINDING.page_cache.snapshot;
	StepInit(&x,1u,0u,0u,3u);
	StepPublish(&x,3u,0x60u);
	StepStart(&x);
	logical = LanePage(0u,0u);
	FillPage(logical,0x44u);
	PageBytes(logical,expected);
	HookReset();
	HOOK.save_thread = BINDING.save_thread;
	HOOK.track_threads = 1u;
	HOOK.block_kind_mask = KindBit(cudaMemcpyDeviceToHost);
	assert(StepFinish(&x,SPARK_STATUS_OK,&finished) == SPARK_STATUS_OK);
	StepRelease(&release,1u,0u);
	while ( atomic_load(&finished.count) == 0u )
		SleepMs(1u);
	assert(StepAdmit(&release,0u) == SPARK_STATUS_OK);
	HookWaitActive();
	assert(HOOK.other_thread_copies == 0u && HOOK.save_thread_copies == 1u);
	assert(snapshot->in_flight == 1u && snapshot->save_count == 0u);
	HookRelease();
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK);
	assert(SparkKvSnapshotFlush(&BINDING.snapshot_store) == SPARK_STATUS_OK);
	assert(snapshot->save_count == 1u && snapshot->save_page_count == 1u && HOOK.other_thread_copies == 0u);
	memset(&key,0,sizeof(key));
	memcpy(key.layout_sha256,BINDING.layout_sha256,sizeof(key.layout_sha256));
	Identity((SparkModelDriverCacheIdentity *)key.identity_sha256,0x60u);
	key.token_count = 3u;
	assert(SparkKvSnapshotReadSegment(&BINDING.snapshot_store,&key,1u,SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES,saved,sizeof(saved),&bytes) == SPARK_STATUS_OK);
	assert(bytes == TEST_PAGE_BYTES && memcmp(saved,expected,sizeof(saved)) == 0);
	Close();
	printf("T4 finish and release only mark: ok\n");
}

static void TestParkCopiesAreAsync(void)
{
	uint32_t blocks[TEST_PHYSICAL + 1u],index,parked;
	SparkStatus status;
	uint8_t expected[TEST_PHYSICAL][TEST_PAGE_BYTES],restored[TEST_PAGE_BYTES];
	Open();
	assert(spark_stub_cuda_stream_flags((cudaStream_t)BINDING.copy_stream) == cudaStreamNonBlocking);
	assert(pthread_mutex_lock(&BINDING.mutex) == 0);
	for (index=0u; index<TEST_PHYSICAL; index++)
	{
		assert(SparkKvCacheArenaAcquireBlock(&BINDING.arena,&blocks[index]) == SPARK_STATUS_OK);
		assert(SparkKvCacheArenaMarkBlockResident(&BINDING.arena,blocks[index]) == SPARK_STATUS_OK);
		FillPage(blocks[index],(uint8_t)(0x70u + index));
		PageBytes(blocks[index],expected[index]);
		assert(SparkKvCacheArenaMarkBlockDirty(&BINDING.arena,blocks[index]) == SPARK_STATUS_OK);
	}
	HookReset();
	assert(SparkKvCacheArenaAcquireBlock(&BINDING.arena,&blocks[TEST_PHYSICAL]) == SPARK_STATUS_OK);
	status = SPARK_STATUS_BUSY;
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvCacheArenaMarkBlockResident(&BINDING.arena,blocks[TEST_PHYSICAL]);
		if ( status == SPARK_STATUS_BUSY )
			(void)SparkKvPageStoreWaitForTransfers(&BINDING.page_store);
	}
	assert(status == SPARK_STATUS_OK);
	for (parked=0u; parked<TEST_PHYSICAL && (BINDING.blocks[blocks[parked]].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u; parked++)
		;
	assert(parked < TEST_PHYSICAL && (BINDING.blocks[blocks[parked]].flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u);
	assert(SparkKvCacheArenaFreeBlock(&BINDING.arena,blocks[TEST_PHYSICAL]) == SPARK_STATUS_OK);
	while ( (BINDING.blocks[blocks[parked]].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u )
	{
		status = SparkKvPageStorePrefetch(&BINDING.page_store,&BINDING.arena,blocks[parked]);
		assert(status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY);
		(void)SparkKvPageStoreWaitForTransfers(&BINDING.page_store);
		(void)SparkKvPageStoreProgress(&BINDING.page_store,&BINDING.arena,1u);
	}
	PageBytes(blocks[parked],restored);
	assert(memcmp(expected[parked],restored,sizeof(restored)) == 0);
	assert(spark_stub_cuda_sync_copy_calls() == 0u && spark_stub_cuda_async_copy_calls() >= 4u && HOOK.foreign_stream_copies == 0u);
	for (index=0u; index<TEST_PHYSICAL; index++)
		assert(SparkKvCacheArenaFreeBlock(&BINDING.arena,blocks[index]) == SPARK_STATUS_OK);
	assert(pthread_mutex_unlock(&BINDING.mutex) == 0);
	Close();
	printf("T5 park copies are async: ok\n");
}

static void NoCompletion(void *context,const SparkModelDriverCompletion *completion)
{
	(void)context;
	assert(completion->status == SPARK_STATUS_OK);
}

static void TestLockSitesMeasured(void)
{
	TestStep x,y,publish;
	TestFinished finished = {0};
	SparkStageKvBindingCounters counters;
	atomic_uint lane_states[TEST_LANES];
	Open();
	(void)PublishPrefix(3u,1u,2u,0x40u,0x55u);
	StepInit(&x,1u,0u,0u,4u);
	StepStart(&x);
	assert(StepFinish(&x,SPARK_STATUS_OK,&finished) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK);
	StepInit(&publish,1u,0u,4u,4u);
	StepPublish(&publish,4u,0x61u);
	publish.request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH;
	publish.request.new_token_count = 0u;
	publish.request.sequence_position = 4u;
	assert(StepAdmit(&publish,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(StepAdmit(&publish,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
	memset(&publish.frame,0,sizeof(publish.frame));
	publish.frame.program_id = publish.request.program_id;
	publish.frame.request_id = publish.request.request_id;
	publish.frame.sequence_position = publish.request.sequence_position;
	publish.frame.active_slot_count = 1u;
	publish.frame.cache_lane_count = 1u;
	publish.frame.cache_lanes = &publish.lane;
	publish.frame.driver_dispatch_slot = publish.decision.driver_dispatch_slot;
	publish.frame.driver_dispatch_generation = publish.request.control_generation;
	publish.frame.driver_dispatch_cookie0 = publish.request.transaction_id;
	publish.frame.driver_dispatch_cookie1 = publish.request.submission_id;
	publish.frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH | SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
	publish.frame.completion_function = NoCompletion;
	SparkStageModuleAtomicStateArrayInitialize(lane_states,TEST_LANES);
	assert(SparkStageKvBindingPublishFrame(&BINDING,&publish.frame,lane_states) == SPARK_STATUS_OK);
	StepInit(&y,2u,1u,2u,3u);
	StepPrefix(&y,2u,0x40u);
	HookReset();
	HOOK.sleep_kind_mask = KindBit(cudaMemcpyDeviceToDevice);
	HOOK.sleep_ns = 20000000ull;
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(StepAdmit(&y,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingSampleCounters(&BINDING,&counters) == SPARK_STATUS_OK);
	assert(counters.lock_sites[SPARK_STAGE_KV_LOCK_SITE_ADMIT].count > 0u);
	assert(counters.lock_sites[SPARK_STAGE_KV_LOCK_SITE_CLAIM].count > 0u);
	assert(counters.lock_sites[SPARK_STAGE_KV_LOCK_SITE_FINISH].count > 0u);
	assert(counters.lock_sites[SPARK_STAGE_KV_LOCK_SITE_PUBLISH].count > 0u);
	assert(counters.lock_sites[SPARK_STAGE_KV_LOCK_SITE_SAVE].count > 0u);
	assert(counters.lock_sites[SPARK_STAGE_KV_LOCK_SITE_ADMIT].max_ns >= 20000000ull);
	assert(counters.entry_count > 0u && counters.completion_count > 0u && counters.copy_on_write_count > 0u);
	Close();
	printf("T6 lock sites measured: ok (admit max %llu ns)\n",(unsigned long long)counters.lock_sites[SPARK_STAGE_KV_LOCK_SITE_ADMIT].max_ns);
}

static void TestQuiesceOrderStop(void)
{
	TestStep x,w;
	TestFinished first = {0},second = {0};
	Open();
	StepInit(&x,1u,0u,0u,1u);
	StepStart(&x);
	StepInit(&w,2u,1u,0u,1u);
	StepStart(&w);
	assert(w.decision.driver_dispatch_slot != x.decision.driver_dispatch_slot);
	atomic_store(&ORDER,0u);
	assert(pthread_mutex_lock(&BINDING.mutex) == 0);
	assert(StepFinish(&x,SPARK_STATUS_OK,&first) == SPARK_STATUS_OK);
	assert(StepFinish(&w,SPARK_STATUS_OK,&second) == SPARK_STATUS_OK);
	SleepMs(10u);
	assert(atomic_load(&first.count) == 0u && atomic_load(&second.count) == 0u);
	assert(pthread_mutex_unlock(&BINDING.mutex) == 0);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK);
	assert(atomic_load(&first.count) == 1u && atomic_load(&second.count) == 1u);
	assert(first.order == 0u && second.order == 1u);
	assert(first.status == SPARK_STATUS_OK && second.status == SPARK_STATUS_OK);
	SparkStageKvBindingStop(&BINDING);
	assert(StepFinish(&x,SPARK_STATUS_OK,&first) == SPARK_STATUS_INVALID_ARGUMENT);
	Close();
	printf("T7 quiesce order stop: ok\n");
}

static void TestCopierContract(void)
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[4];
	SparkKvCacheConfiguration configuration;
	SparkKvDeviceCopyConfiguration copy;
	SparkKvDeviceCopier copier;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	atomic_uint slot_states[TEST_SLOTS],lane_states[TEST_LANES];
	uint32_t residents[4],pages[3],index,syncs;
	uint8_t device[4u * 64u];
	cudaStream_t stream;
	memset(&configuration,0,sizeof(configuration));
	memset(device,0,sizeof(device));
	configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.logical_block_count = 4u;
	configuration.block_token_count = TEST_BLOCK;
	configuration.resident_block_capacity = 4u;
	configuration.layer_count = configuration.kv_head_count = configuration.bytes_per_scalar = 1u;
	configuration.head_dim = 16u;
	configuration.key_block_stride_bytes = 64u;
	configuration.key_device_base = device;
	configuration.blocks = blocks;
	configuration.resident_slot_logical_block_indices = residents;
	assert(SparkKvCacheArenaInitialize(&arena,&configuration) == SPARK_STATUS_OK);
	for (index=0u; index<3u; index++)
	{
		assert(SparkKvCacheArenaAcquireBlock(&arena,&pages[index]) == SPARK_STATUS_OK);
		assert(SparkKvCacheArenaMarkBlockResident(&arena,pages[index]) == SPARK_STATUS_OK);
		assert(SparkKvCacheArenaPinResidentBlock(&arena,pages[index]) == SPARK_STATUS_OK);
	}
	assert(SparkKvCacheArenaPinResidentBlock(&arena,pages[0]) == SPARK_STATUS_OK);
	assert(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) == cudaSuccess);
	memset(&copy,0,sizeof(copy));
	copy.module_tag = "kvtest";
	copy.stream = stream;
	copy.arena = &arena;
	copy.physical_page_count = 4u;
	copy.region_count = 1u;
	copy.regions[0].layout = SPARK_KV_DEVICE_COPY_PAGE_MAJOR;
	copy.regions[0].layer_count = 1u;
	copy.regions[0].device_base = (uintptr_t)device;
	copy.regions[0].layer_page_bytes = 64u;
	copy.pending_capacity = 1u;
	assert(SparkKvDeviceCopierInitialize(&copier,&copy) == SPARK_STATUS_OK);
	spark_stub_cuda_event_pending(1u);
	assert(SparkKvDeviceCopierCopyPage(&copier,pages[0],pages[1]) == SPARK_STATUS_OK);
	assert(SparkKvDeviceCopierCopyPage(&copier,pages[0],pages[2]) == SPARK_STATUS_BUSY && copier.full_logged == 1u && copier.busy_count == 1u);
	assert(SparkKvDeviceCopierCopyPage(&copier,pages[0],pages[2]) == SPARK_STATUS_BUSY && copier.busy_count == 2u);
	assert(SparkKvDeviceCopierRetire(&copier,0u) == SPARK_STATUS_OK && copier.pending_count == 1u);
	assert(blocks[pages[0]].residency_reference_count == 2u && blocks[pages[1]].residency_reference_count == 1u);
	spark_stub_cuda_event_pending(0u);
	assert(SparkKvDeviceCopierRetire(&copier,0u) == SPARK_STATUS_OK && copier.pending_count == 0u);
	assert(blocks[pages[0]].residency_reference_count == 1u && blocks[pages[1]].residency_reference_count == 0u);
	syncs = cuda_stub_stream_sync_calls;
	spark_stub_cuda_event_record_failure(1u);
	assert(SparkKvDeviceCopierCopyPage(&copier,pages[0],pages[2]) == SPARK_STATUS_IO_ERROR);
	spark_stub_cuda_event_record_failure(0u);
	assert(cuda_stub_stream_sync_calls == syncs + 1u && copier.pending_count == 0u);
	assert(blocks[pages[0]].residency_reference_count == 1u && blocks[pages[2]].residency_reference_count == 1u);
	assert(SparkKvDeviceCopierDrain(&copier) == SPARK_STATUS_OK);
	SparkKvDeviceCopierDestroy(&copier);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	Open();
	SparkStageModuleAtomicStateArrayInitialize(slot_states,TEST_SLOTS);
	SparkStageModuleAtomicStateArrayInitialize(lane_states,TEST_LANES);
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = sizeof(request);
	request.program_id = 7u;
	request.control_generation = 3u;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	assert(cudaStreamCreate(&stream) == cudaSuccess);
	syncs = cuda_stub_stream_sync_calls;
	assert(SparkStageKvBindingAdmitReset(&BINDING,&request,&decision,slot_states,lane_states,stream) == SPARK_STATUS_OK);
	assert(decision.accepted == 1u && cuda_stub_stream_sync_calls == syncs + 2u);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	Close();
	printf("T20 copier contract: ok\n");
}

static void ExpectRefused(const SparkStageKvConfiguration *configuration,SparkStatus expected)
{
	assert(OpenWith(configuration) == expected);
	Unload();
}

static void TestSnapshotRefusals(void)
{
	SparkStageKvConfiguration configuration;
	SparkStageKvBinding other;
	SparkStageModuleLedger other_ledger;
	int on_stack = 0;
	MakeDirectories();
	Configure(&configuration);
	memset(configuration.pack_sha256,0,sizeof(configuration.pack_sha256));
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	memset(configuration.contract_sha256,0,sizeof(configuration.contract_sha256));
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	configuration.model_id = "";
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	configuration.expert_codec = 0u;
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	configuration.driver_symbol = 0;
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	configuration.snapshot_directory = 0;
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	configuration.snapshot_maximum_bytes = 0u;
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	configuration.snapshot_maximum_bytes = 4096u;
	ExpectRefused(&configuration,SPARK_STATUS_CAPACITY_EXCEEDED);
	Configure(&configuration);
	configuration.expert_codec = 999u;
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	Configure(&configuration);
	configuration.driver_symbol = &on_stack;
	ExpectRefused(&configuration,SPARK_STATUS_NOT_FOUND);
	Configure(&CONFIGURATION);
	assert(OpenWith(&CONFIGURATION) == SPARK_STATUS_OK);
	configuration = CONFIGURATION;
	memset(&other_ledger,0,sizeof(other_ledger));
	other_ledger.module_tag = "kvtest";
	configuration.ledger = &other_ledger;
	memset(&other,0,sizeof(other));
	assert(SparkStageKvBindingInitialize(&other,&configuration) == SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageKvBindingDestroy(&other);
	SparkStageModuleLedgerRollback(&other_ledger,0u);
	Close();
	printf("A10 snapshot refusals: ok\n");
}

static void PublishStep(uint64_t sequence,uint32_t slot,uint32_t position,uint32_t context,uint8_t identity,uint8_t seed,uint8_t *bytes)
{
	TestStep step;
	TestFinished finished = {0};
	uint32_t page;
	StepInit(&step,sequence,slot,position,context);
	StepPublish(&step,context,identity);
	StepStart(&step);
	page = LanePage(slot,position / TEST_BLOCK);
	FillPage(page,seed);
	PageBytes(page,bytes);
	assert(StepFinish(&step,SPARK_STATUS_OK,&finished) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK);
	assert(atomic_load(&finished.count) == 1u && finished.status == SPARK_STATUS_OK);
}

static void ReleaseSequence(uint64_t sequence,uint32_t slot)
{
	TestStep release;
	StepRelease(&release,sequence,slot);
	assert(StepAdmit(&release,0u) == SPARK_STATUS_OK);
}

static SparkStatus RestorePrefix(uint64_t sequence,uint32_t slot,uint32_t tokens,uint8_t identity,uint8_t pages[][TEST_PAGE_BYTES],uint32_t page_count)
{
	TestStep step;
	uint8_t bytes[TEST_PAGE_BYTES];
	uint32_t index;
	SparkStatus status;
	StepInit(&step,sequence,slot,tokens,tokens);
	StepPrefix(&step,tokens,identity);
	status = StepAdmit(&step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (index=0u; index<page_count; index++)
	{
		PageBytes(LanePage(slot,index),bytes);
		assert(memcmp(bytes,pages[index],TEST_PAGE_BYTES) == 0);
	}
	assert(StepAdmit(&step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static void PublishTwoPageChain(uint8_t pages[2][TEST_PAGE_BYTES])
{
	PublishStep(1u,0u,0u,4u,0x70u,0x31u,pages[0]);
	PublishStep(1u,0u,4u,8u,0x71u,0x32u,pages[1]);
	ReleaseSequence(1u,0u);
}

static void TestSnapshotRestartRestore(void)
{
	SparkModelDriverKvStoreCounters counters;
	uint8_t pages[2][TEST_PAGE_BYTES];
	Open();
	PublishTwoPageChain(pages);
	Unload();
	assert(OpenWith(&CONFIGURATION) == SPARK_STATUS_OK);
	assert(RestorePrefix(2u,1u,8u,0x71u,pages,2u) == SPARK_STATUS_OK);
	SparkStageKvBindingKvStoreCounters(&BINDING,&counters);
	assert(counters.attached == 1u && counters.restore_count == 1u && counters.restore_page_count == 2u && counters.restore_failure_count == 0u);
	assert(counters.store_file_count == 2u && counters.store_foreign_layout_file_count == 0u && counters.store_used_bytes != 0u);
	Close();
	printf("A10 restart restore: ok\n");
}

static void ExpectForeignLayout(const SparkStageKvConfiguration *configuration,uint8_t pages[2][TEST_PAGE_BYTES])
{
	SparkModelDriverKvStoreCounters counters;
	assert(OpenWith(configuration) == SPARK_STATUS_OK);
	assert(RestorePrefix(2u,1u,8u,0x71u,pages,2u) == SPARK_STATUS_NOT_FOUND);
	SparkStageKvBindingKvStoreCounters(&BINDING,&counters);
	assert(counters.restore_miss_count == 1u && counters.store_foreign_layout_file_count == 2u && counters.store_file_count == 2u);
	Unload();
}

static void TestSnapshotLayoutSeparation(void)
{
	SparkStageKvConfiguration configuration;
	uint8_t pages[2][TEST_PAGE_BYTES];
	Open();
	PublishTwoPageChain(pages);
	Unload();
	configuration = CONFIGURATION;
	configuration.expert_codec = SPARK_WEIGHT_CODEC_BF16;
	ExpectForeignLayout(&configuration,pages);
	configuration = CONFIGURATION;
	configuration.pack_sha256[7] ^= 1u;
	ExpectForeignLayout(&configuration,pages);
	configuration = CONFIGURATION;
	configuration.contract_sha256[3] ^= 1u;
	ExpectForeignLayout(&configuration,pages);
	configuration = CONFIGURATION;
	configuration.model_revision = "fixture-2";
	ExpectForeignLayout(&configuration,pages);
	assert(OpenWith(&CONFIGURATION) == SPARK_STATUS_OK);
	assert(RestorePrefix(2u,1u,8u,0x71u,pages,2u) == SPARK_STATUS_OK);
	Close();
	printf("A10 layout separation: ok\n");
}

static void TestSnapshotDestroySavesAll(void)
{
	uint8_t pages[3][1][TEST_PAGE_BYTES];
	uint32_t chain;
	Open();
	for (chain=0u; chain<3u; chain++)
	{
		PublishStep(10u + chain,0u,0u,4u,(uint8_t)(0x80u + chain),(uint8_t)(0x40u + chain),pages[chain][0]);
		ReleaseSequence(10u + chain,0u);
	}
	Unload();
	assert(OpenWith(&CONFIGURATION) == SPARK_STATUS_OK);
	for (chain=0u; chain<3u; chain++)
		assert(RestorePrefix(20u + chain,1u,4u,(uint8_t)(0x80u + chain),pages[chain],1u) == SPARK_STATUS_OK);
	assert(BINDING.page_cache.snapshot->restore_count == 3u);
	Close();
	printf("A10 destroy saves every published chain: ok\n");
}

#define TEST_RECURRENT_BYTES 48u

typedef struct TestRecurrent
{
	uint8_t lanes[TEST_LANES][TEST_RECURRENT_BYTES];
	uint32_t to_buffer;
	uint32_t from_buffer;
	void *last_stream;
	SparkStatus fail;
} TestRecurrent;

static TestRecurrent RECURRENT;

static SparkStatus TestRecurrentCopy(void *context,uint32_t direction,uint32_t slot,void *buffer,uint64_t bytes,void *stream)
{
	TestRecurrent *recurrent = context;
	assert(recurrent == &RECURRENT && slot < TEST_LANES && bytes == TEST_RECURRENT_BYTES && buffer != 0);
	if ( recurrent->fail != SPARK_STATUS_OK )
		return(recurrent->fail);
	recurrent->last_stream = stream;
	if ( direction == SPARK_STAGE_KV_RECURRENT_TO_BUFFER )
	{
		recurrent->to_buffer++;
		memcpy(buffer,recurrent->lanes[slot],TEST_RECURRENT_BYTES);
	}
	else
	{
		assert(direction == SPARK_STAGE_KV_RECURRENT_FROM_BUFFER);
		recurrent->from_buffer++;
		memcpy(recurrent->lanes[slot],buffer,TEST_RECURRENT_BYTES);
	}
	return(SPARK_STATUS_OK);
}

static void ConfigureRecurrent(SparkStageKvConfiguration *configuration)
{
	Configure(configuration);
	configuration->recurrent.lane_bytes = TEST_RECURRENT_BYTES;
	configuration->recurrent.copy = TestRecurrentCopy;
	configuration->recurrent.context = &RECURRENT;
	configuration->backing_maximum_bytes += (uint64_t)TEST_LOGICAL * TEST_RECURRENT_BYTES;
	memset(&RECURRENT,0,sizeof(RECURRENT));
}

static void TestRecurrentRoundTrip(void)
{
	TestStep step,other;
	TestFinished finished = {0},other_finished = {0};
	SparkStageKvRecurrentCounters counters;
	void *stream = (void *)(uintptr_t)0x51u;
	uint64_t published;
	uint32_t index;
	MakeDirectories();
	ConfigureRecurrent(&CONFIGURATION);
	assert(OpenWith(&CONFIGURATION) == SPARK_STATUS_OK && BINDING.state_slot_count == TEST_LOGICAL);
	StepInit(&step,1u,0u,0u,4u);
	StepPublish(&step,4u,0x90u);
	StepStart(&step);
	StepInit(&other,5u,1u,0u,3u);
	StepStart(&other);
	for (index=0u; index<TEST_RECURRENT_BYTES; index++)
		RECURRENT.lanes[0][index] = (uint8_t)(0xc0u + index);
	assert(SparkStageKvBindingRecurrentCapture(&BINDING,&other.slot,1u,stream) == SPARK_STATUS_OK && RECURRENT.to_buffer == 0u);
	assert(SparkStageKvBindingRecurrentCapture(&BINDING,&step.slot,1u,stream) == SPARK_STATUS_OK);
	assert(RECURRENT.to_buffer == 1u && RECURRENT.last_stream == stream && (BINDING.lane_state_flags[0] & SPARK_STAGE_KV_LANE_STATE_CAPTURED) != 0u);
	assert(StepFinish(&step,SPARK_STATUS_OK,&finished) == SPARK_STATUS_OK);
	assert(StepFinish(&other,SPARK_STATUS_OK,&other_finished) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK);
	assert(finished.status == SPARK_STATUS_OK && other_finished.status == SPARK_STATUS_OK && BINDING.lane_state_flags[0] == 0u);
	SparkStageKvBindingTakeRecurrentCounters(&BINDING,&counters);
	assert(counters.captures == 1u && counters.capture_bytes == TEST_RECURRENT_BYTES && counters.restores == 0u);
	ReleaseSequence(1u,0u);
	ReleaseSequence(5u,1u);
	StepInit(&step,2u,1u,4u,5u);
	StepPrefix(&step,4u,0x90u);
	assert(StepAdmit(&step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert((BINDING.lane_state_flags[1] & SPARK_STAGE_KV_LANE_STATE_RESTORE_READY) != 0u);
	assert(StepAdmit(&step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
	assert(StepClaim(&step) == SPARK_STATUS_OK);
	memset(RECURRENT.lanes[1],0,TEST_RECURRENT_BYTES);
	assert(SparkStageKvBindingRecurrentRestore(&BINDING,&step.slot,1u,stream) == SPARK_STATUS_OK);
	assert(RECURRENT.from_buffer == 1u && RECURRENT.last_stream == stream && memcmp(RECURRENT.lanes[1],RECURRENT.lanes[0],TEST_RECURRENT_BYTES) == 0);
	SparkStageKvBindingTakeRecurrentCounters(&BINDING,&counters);
	assert(counters.restores == 1u && counters.restore_bytes == TEST_RECURRENT_BYTES);
	assert(StepFinish(&step,SPARK_STATUS_IO_ERROR,&finished) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK && finished.status == SPARK_STATUS_IO_ERROR);
	assert(BINDING.lanes[1].phase == SPARK_KV_LANE_TRANSACTION_EMPTY && BINDING.lane_state_flags[1] == 0u);
	StepInit(&step,3u,0u,4u,5u);
	StepPrefix(&step,4u,0x90u);
	assert(StepAdmit(&step,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK && BINDING.lane_state_flags[0] != 0u);
	assert(SparkStageKvBindingReset(&BINDING,2u) == SPARK_STATUS_OK);
	for (index=0u; index<TEST_LANES; index++)
		assert(BINDING.lane_state_flags[index] == 0u && BINDING.lanes[index].phase == SPARK_KV_LANE_TRANSACTION_EMPTY);
	SparkStageKvBindingTakeRecurrentCounters(&BINDING,&counters);
	assert(counters.restores == 1u && counters.captures == 0u);
	published = BINDING.page_cache.published_page_count;
	StepInit(&step,4u,0u,0u,4u);
	StepPublish(&step,4u,0x91u);
	StepStart(&step);
	RECURRENT.fail = SPARK_STATUS_IO_ERROR;
	assert(SparkStageKvBindingRecurrentCapture(&BINDING,&step.slot,1u,stream) == SPARK_STATUS_IO_ERROR);
	RECURRENT.fail = SPARK_STATUS_OK;
	assert(StepFinish(&step,SPARK_STATUS_IO_ERROR,&finished) == SPARK_STATUS_OK);
	assert(SparkStageKvBindingQuiesce(&BINDING,1000000000ull) == SPARK_STATUS_OK && finished.status == SPARK_STATUS_IO_ERROR);
	assert(BINDING.page_cache.published_page_count == published && BINDING.lane_state_flags[0] == 0u);
	SparkStageKvBindingTakeRecurrentCounters(&BINDING,&counters);
	assert(counters.captures == 0u && counters.restores == 0u);
	Close();
	printf("A11 recurrent state: captured only for publishing lanes, recorded at finish, restored on a prefix hit at claim, cleared by reset and by a failed finish: ok\n");
}

static void TestRecurrentRefusals(void)
{
	SparkStageKvConfiguration configuration;
	MakeDirectories();
	ConfigureRecurrent(&configuration);
	configuration.recurrent.copy = 0;
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	ConfigureRecurrent(&configuration);
	configuration.recurrent.lane_bytes = 0u;
	ExpectRefused(&configuration,SPARK_STATUS_INVALID_ARGUMENT);
	ConfigureRecurrent(&configuration);
	configuration.backing_maximum_bytes = (uint64_t)TEST_LOGICAL * TEST_RECURRENT_BYTES;
	ExpectRefused(&configuration,SPARK_STATUS_CAPACITY_EXCEEDED);
	RemoveDirectories();
	printf("A11 recurrent refusals: a lane size without a copy hook, a hook without a lane size and a backing budget short of one record per logical page refuse: ok\n");
}

int main(void)
{
	setvbuf(stdout,0,_IONBF,0);
	TestCompletionEntryNeverWaits();
	TestCopyOnWriteIsOneDeviceCopyPerRegion();
	TestCopyOnWriteDuringAPark();
	TestFinishAndReleaseOnlyMark();
	TestParkCopiesAreAsync();
	TestLockSitesMeasured();
	TestQuiesceOrderStop();
	TestCopierContract();
	TestSnapshotRefusals();
	TestSnapshotRestartRestore();
	TestSnapshotLayoutSeparation();
	TestSnapshotDestroySavesAll();
	TestRecurrentRoundTrip();
	TestRecurrentRefusals();
	printf("PASS stage kv binding: completion entry never waits, device copy-on-write on the copy stream with pins held until the event, copy-on-write during a park, finish and release only mark saves, async park copies, measured lock sites, FIFO quiesce and stop, copier contract\n");
	return(0);
}
