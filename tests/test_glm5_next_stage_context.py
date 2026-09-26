#!/usr/bin/env python3
"""Exercise the real module configurator without allocating CUDA state."""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile
from test_generated_control_admission import generate_admission

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include <errno.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"
#include "cache/kv_page_store.c"
#include "src/spark_speculation_policy.c"
#define SparkKvBackendInitialize SparkTestRealKvBackendInitialize
#include "cache/kv_model_table.c"
#undef SparkKvBackendInitialize
#define main SparkUnusedKvTestMain
#include "tests/test_kv_cache.c"
#undef main
typedef struct { void *operation_0_state; } SparkGeneratedDriverInstance;
#include "generated_admission.inc"
static SparkGlm5NextModuleState state;
static uint32_t COPY_COUNT,CANCEL_COUNT,STREAM_QUERY_COUNT,EXPECTED_CANCEL_COUNT,END_COUNT;
static SparkStatus END_STATUS;
static uint32_t HOST_MODE,HOST_COUNT,IN_CUDA_CALLBACK,MTP_COMMITS;
static cudaHostFn_t HOST_FUNCTIONS[16];
static void *HOST_CONTEXTS[16];
static pthread_t HOST_THREAD;
static atomic_int DRAIN_STATUS;

static void *complete_delayed(void *context)
{
	uint32_t index = (uint32_t)(uintptr_t)context;
	struct timespec pause = {0,20000000};
	nanosleep(&pause,0);
	HOST_FUNCTIONS[index](HOST_CONTEXTS[index]);
	nanosleep(&pause,0);
	DRAIN_STATUS = cudaSuccess;
	return(0);
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream,cudaHostFn_t function,void *context)
{
	uint32_t index = HOST_COUNT++;
	assert(IN_CUDA_CALLBACK == 0u);
	(void)stream;
	assert(index < 16u);
	HOST_FUNCTIONS[index] = function;
	HOST_CONTEXTS[index] = context;
	if ( HOST_MODE == 3u )
		return(cudaErrorInvalidValue);
	if ( HOST_MODE == 0u )
	{
		IN_CUDA_CALLBACK = 1u;
		function(context);
		IN_CUDA_CALLBACK = 0u;
		DRAIN_STATUS = cudaSuccess;
	}
	if ( HOST_MODE == 2u )
		assert(pthread_create(&HOST_THREAD,0,complete_delayed,(void *)(uintptr_t)index) == 0);
	return(cudaSuccess);
}

cudaError_t cudaGetLastError(void)
{
	assert(IN_CUDA_CALLBACK == 0u);
	return(cudaSuccess);
}

SparkStatus SparkTpDeviceCollectiveChainRetire(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return(SPARK_STATUS_OK);
}

static uint32_t REAL_BACKEND;
static int32_t ALLOCATIONS_BEFORE_FAILURE = -1;

static uint32_t HEALTH_DEAD_MASK,HEALTH_CALLS;
uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client)
{
	HEALTH_CALLS++;
	return((HEALTH_DEAD_MASK & (uint32_t)(uintptr_t)client) == 0u);
}

static void check_weightd_health(void)
{
	SparkWeightdLazyPack pack = {0};
	for (uint32_t mask=1u; mask<4u; mask++)
	{
		memset(&state,0,sizeof(state));
		pack.client = (SparkWeightdClient *)(uintptr_t)2u;
		state.lane_client = (SparkWeightdClient *)(uintptr_t)1u;
		state.lazy_pack = &pack;
		HEALTH_DEAD_MASK = mask;
		HEALTH_CALLS = 0u;
		assert(SparkGlm5NextWeightdHealth(&state) == SPARK_STATUS_IO_ERROR);
		assert(HEALTH_CALLS == 2u);
		HEALTH_DEAD_MASK = 0u;
		assert(SparkGlm5NextWeightdHealth(&state) == SPARK_STATUS_IO_ERROR);
		assert(HEALTH_CALLS == 2u);
	}
}

void SparkTpDeviceCollectiveBroadcastCancel(SparkTpDeviceCollective *collective)
{
    (void)collective;
    CANCEL_COUNT++;
}

SparkStatus SparkTpDeviceCollectiveEndChain(SparkTpDeviceCollective *collective,void *stream)
{
    (void)collective;
    assert(stream == state.execution_stream && DRAIN_STATUS == cudaSuccess);
    assert(atomic_load(&state.tp_chain_active) == 1u);
    END_COUNT++;
    return(END_STATUS);
}

void SparkTpDeviceCollectiveRoundStats(SparkTpDeviceCollective *collective,
    uint64_t *count,uint64_t *elapsed,uint32_t reset)
{
    (void)collective;
    (void)reset;
    assert(atomic_load(&state.tp_chain_active) == 1u);
    *count = 0u;
    *elapsed = 0u;
}

SparkStatus SparkTpDeviceCollectiveHardwareStats(SparkTpDeviceCollective *collective,
    SparkTpDeviceCollectiveHardwareTiming *timing)
{
    (void)collective;
    (void)timing;
    return(SPARK_STATUS_UNSUPPORTED);
}

static SparkStatus VERIFY_STATUS;
static uint32_t VERIFY_COUNT,STREAM_ORDERED;

SparkStatus SparkTpDeviceCollectiveVerifyDeferred(SparkTpDeviceCollective *collective,void *stream)
{
    (void)collective;
    assert(stream == state.execution_stream && DRAIN_STATUS == cudaSuccess);
    VERIFY_COUNT++;
    return(VERIFY_STATUS);
}

uint32_t SparkTpDeviceCollectiveStreamOrdered(const SparkTpDeviceCollective *collective)
{
    (void)collective;
    return(STREAM_ORDERED);
}

static char WALK_TRACE[256];
static uint32_t WALK_LENGTH,WALK_GATHER_LAYER,WALK_FAIL_CODE;

static int32_t walk_note(char code)
{
	if ( WALK_LENGTH + 1u < sizeof(WALK_TRACE) )
		WALK_TRACE[WALK_LENGTH++] = code;
	WALK_TRACE[WALK_LENGTH] = 0;
	return((uint32_t)(unsigned char)code == WALK_FAIL_CODE ? 1 : 0);
}

int32_t SparkGlm5NextLaunchCudaWaveBegin(const SparkGlm5NextCudaWave *wave) { (void)wave;return(walk_note('B')); }
int32_t SparkGlm5NextLaunchCudaLayerAttention(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('A')); }
int32_t SparkGlm5NextLaunchCudaLayerAttentionScore(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('S')); }
int32_t SparkGlm5NextLaunchCudaLayerAttentionSelect(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('T')); }
uint32_t SparkGlm5NextLayerIndexGatherSequences(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;return(layer == WALK_GATHER_LAYER ? 2u : 0u); }
int32_t SparkGlm5NextLaunchCudaLayerMlp(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('M')); }
int32_t SparkGlm5NextLaunchCudaLayerMlpRoute(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('R')); }
int32_t SparkGlm5NextLaunchCudaLayerMlpExperts(const SparkGlm5NextCudaWave *wave,uint32_t layer) { assert(wave->expert_lease_all == 1u);(void)layer;return(walk_note('E')); }
int32_t SparkGlm5NextLaunchCudaLayerAttentionPost(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('P')); }
int32_t SparkGlm5NextLaunchCudaLayerMlpPost(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('Q')); }
int32_t SparkGlm5NextLaunchCudaWaveHead(const SparkGlm5NextCudaWave *wave) { (void)wave;return(walk_note('H')); }
cudaError_t SparkGlm5NextLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	assert(stream == state.execution_stream && maxloc != 0 && token_ids != 0 && row_count == 2u);
	return(walk_note('U') != 0 ? cudaErrorInvalidValue : cudaSuccess);
}

SparkStatus SparkTpDeviceCollectiveEnqueue(SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission,uint32_t operation_kind)
{
	assert((submission->flags & SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION) != 0u);
	assert(submission->cuda_stream == state.execution_stream && submission->active_sequence_count == 2u && submission->logical_sequence_count == 2u);
	(void)walk_note(submission->completion_function != 0 ? 'c' : collective == &state.tp_device_collective_hc ? 'h' : operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER ? 'g' : operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 ? 'x' : 'r');
	return(SPARK_STATUS_OK);
}

const SparkWeightdRangeGroup *SparkWeightdManifestFind(const SparkWeightdManifest *manifest,uint32_t layer,uint32_t expert) { (void)manifest;(void)layer;(void)expert;abort(); }
uint64_t SparkTpDeviceCollectiveRoundIndex(SparkTpDeviceCollective *collective) { (void)collective;return(0u); }
SparkStatus SparkTpDeviceCollectiveArmCapture(SparkTpDeviceCollective *collective) { (void)collective;abort(); }
cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode) { (void)stream;(void)mode;abort(); }
cudaError_t cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t *graph) { (void)stream;(void)graph;abort(); }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec,cudaGraph_t graph,...) { (void)exec;(void)graph;abort(); }
cudaError_t cudaGraphUpload(cudaGraphExec_t exec,cudaStream_t stream) { (void)exec;(void)stream;abort(); }
cudaError_t cudaGraphDestroy(cudaGraph_t graph) { (void)graph;abort(); }
cudaError_t cudaEventSynchronize(cudaEvent_t event) { (void)event;abort(); }
SparkStatus SparkWeightdRouteKeys(uint32_t layer,const uint32_t *offsets,uint32_t expert_count,uint32_t packed_rows,SparkWeightdExpertKey *keys,uint32_t capacity,uint32_t *count) { (void)layer;(void)offsets;(void)expert_count;(void)packed_rows;(void)keys;(void)capacity;(void)count;abort(); }

cudaError_t cudaMalloc(void **pointer,size_t bytes)
{
	if ( ALLOCATIONS_BEFORE_FAILURE == 0 )
		return(cudaErrorMemoryAllocation);
	if ( ALLOCATIONS_BEFORE_FAILURE > 0 )
		ALLOCATIONS_BEFORE_FAILURE--;
	*pointer = malloc(bytes);
	return(*pointer != 0 ? cudaSuccess : cudaErrorMemoryAllocation);
}

cudaError_t cudaFree(void *pointer)
{
	free(pointer);
	return(cudaSuccess);
}

cudaError_t cudaMemset(void *pointer,int value,size_t bytes)
{
	memset(pointer,value,bytes);
	return(cudaSuccess);
}

cudaError_t cudaMemsetAsync(void *pointer,int value,size_t bytes,cudaStream_t stream)
{
	(void)stream;
	return(cudaMemset(pointer,value,bytes));
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
	assert(IN_CUDA_CALLBACK == 0u && (stream != 0 || state.execution_stream == 0));
	STREAM_QUERY_COUNT++;
	return(DRAIN_STATUS);
}

static uint32_t SYNC_COUNT;

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	(void)stream;
	SYNC_COUNT++;
	return(DRAIN_STATUS);
}

const char *cudaGetErrorString(cudaError_t error)
{
	(void)error;
	return("host test CUDA status");
}

cudaError_t cudaMemcpy(void *destination,const void *source,size_t bytes,cudaMemcpyKind kind)
{
	assert(IN_CUDA_CALLBACK == 0u);
	(void)kind;
	memcpy(destination,source,bytes);
	return(cudaSuccess);
}

cudaError_t cudaHostAlloc(void **destination,size_t bytes,unsigned int flags)
{
	(void)flags;
	*destination = malloc(bytes);
	return(*destination != 0 ? cudaSuccess : cudaErrorMemoryAllocation);
}

cudaError_t cudaFreeHost(void *pointer)
{
	free(pointer);
	return(cudaSuccess);
}

cudaError_t cudaMemcpyAsync(void *destination,const void *source,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream)
{
	(void)stream;
	COPY_COUNT++;
	return(cudaMemcpy(destination,source,bytes,kind));
}

static SparkWeightdWorkFunction COMPLETION_WORK;
static void *COMPLETION_CONTEXT;
static SparkStatus WORK_STATUS,COMPLETION_STATUS;
static uint32_t COMPLETION_REUSED;

SparkStatus SparkWeightdWorkerSubmit(SparkWeightdWorker *worker,SparkWeightdWorkFunction function,void *context)
{
	if ( worker != (SparkWeightdWorker *)(uintptr_t)1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	assert(pthread_mutex_trylock(&state.completion_queue_lock) == EBUSY);
	if ( WORK_STATUS == SPARK_STATUS_OK )
	{
		COMPLETION_WORK = function;
		COMPLETION_CONTEXT = context;
	}
	return(WORK_STATUS);
}

static void observe_completion(void *context,const SparkModelDriverCompletion *completion)
{
	uint32_t lanes[2] = {0u,1u},slot;
	(void)context;
	assert(atomic_load(&state.tp_chain_active) == 0u);
	assert(CANCEL_COUNT == EXPECTED_CANCEL_COUNT);
	COMPLETION_REUSED = SparkStageModuleIndexSetClaim(state.lane_states,state.resident_sequence_capacity,lanes,2u) == SPARK_STATUS_OK;
	COMPLETION_REUSED &= SparkStageModuleSlotClaim(state.slot_states,1u,&slot) == SPARK_STATUS_OK;
	state.completions[0].completion.status = SPARK_STATUS_OK;
	COMPLETION_STATUS = completion->status;
	if ( COMPLETION_REUSED != 0u )
	{
		SparkStageModuleIndexSetRelease(state.lane_states,state.resident_sequence_capacity,lanes,2u);
		SparkStageModuleSlotRelease(state.slot_states,slot);
	}
}

static uint64_t EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES / sizeof(uint64_t)];
const void *SparkWeightdMapEpochDevice(const SparkWeightdMap *map)
{ (void)map;return(EPOCH_WORDS); }
static uint32_t GRAPH_LAUNCHES;
static uint64_t GRAPH_ERROR;

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec,cudaStream_t stream)
{
	assert(exec == (cudaGraphExec_t)(uintptr_t)9u && stream == state.execution_stream);
	GRAPH_LAUNCHES++;
	EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64]++;
	return(cudaSuccess);
}
SparkStatus SparkTpDeviceCollectiveGraphPreLaunch(SparkTpDeviceCollective *collective,void *stream)
{ (void)collective;(void)stream;return(SPARK_STATUS_OK); }
SparkStatus SparkTpDeviceCollectiveGraphCancelSeed(SparkTpDeviceCollective *collective,void *stream)
{ (void)collective;(void)stream;return(SPARK_STATUS_OK); }
SparkStatus SparkTpDeviceCollectiveDisarmCapture(SparkTpDeviceCollective *collective)
{ (void)collective;return(SPARK_STATUS_OK); }
uint64_t SparkTpDeviceCollectiveGraphError(SparkTpDeviceCollective *collective)
{ (void)collective;return(GRAPH_ERROR); }
uint64_t SparkTpDeviceCollectiveGraphDiag(SparkTpDeviceCollective *collective)
{ (void)collective;return(0u); }
uint64_t SparkTpDeviceCollectiveGraphProgress(SparkTpDeviceCollective *collective,uint64_t *cell)
{ (void)collective;*cell=0u;return(0u); }
uint64_t SparkTpDeviceCollectiveGraphStuckDump(SparkTpDeviceCollective *collective)
{ (void)collective;return(0u); }
SparkStatus SparkTpDeviceCollectiveGraphArrivalDump(SparkTpDeviceCollective *collective,uint32_t rank)
{ (void)collective;(void)rank;return(SPARK_STATUS_OK); }

static void check_graph_epoch_ownership(void)
{
	SparkGlm5NextTpChain chain = {0};
	uint32_t position = 7u,token = 3u,output = 123u;
	SparkStatus status;
	memset(&state,0,sizeof(state));
	memset(EPOCH_WORDS,0,sizeof(EPOCH_WORDS));
	state.execution_stream = (void *)(uintptr_t)7u;
	state.slots[0].stream = state.execution_stream;
	state.slots[0].graph_exec_a = (void *)(uintptr_t)9u;
	state.slots[0].host_output_token_ids = &output;
	state.decode_miss_host = (uint32_t *)EPOCH_WORDS;
	EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64] = 41u;
	LEGACY_EPOCH_SETUP
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	state.tp_device_collective.operation_timeout_milli = 50u;
	chain.state = &state;chain.slot = &state.slots[0];
	chain.wave.host_positions = &position;chain.wave.host_token_ids = &token;chain.wave_rows = 1u;
	GRAPH_LAUNCHES = 0u;GRAPH_ERROR = 0u;DRAIN_STATUS = cudaSuccess;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_OK && position == 8u && token == output);
	assert(GRAPH_LAUNCHES == 1u && EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64] == 42u);
	assert(state.completions[0].graph == 1u && state.completions[0].launch_ns != 0u);
	state.decode_miss_host[0] = 1u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_BUSY && position == 8u);
	state.decode_miss_host[0] = 0u;GRAPH_ERROR = 7u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_INTERNAL_ERROR && position == 8u);
	GRAPH_ERROR = 0u;
	state.lane_client = (SparkWeightdClient *)(uintptr_t)1u;HEALTH_DEAD_MASK = 1u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_IO_ERROR && position == 8u && GRAPH_LAUNCHES == 3u);
	HEALTH_DEAD_MASK = 0u;
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
}

static uint32_t LAZY_OPEN_CALLS,LAZY_OPEN_MODE;
static SparkWeightdLazyPack OPEN_PACK;
SparkStatus SparkWeightdAttachRequested(void) { return(SPARK_STATUS_OK); }
SparkStatus SparkWeightdLazyPackCreateChecked(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t budget,uint64_t timeout,SparkWeightdManifestCheck check,void *context,SparkWeightdLazyPack **out)
{
	(void)socket;(void)request;(void)budget;(void)timeout;(void)check;(void)context;
	assert(*out == 0);
	LAZY_OPEN_CALLS++;
	*out = LAZY_OPEN_MODE == 1u && LAZY_OPEN_CALLS == 1u ? 0 : &OPEN_PACK;
	return(LAZY_OPEN_CALLS == 1u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

static void check_lazy_open_retained_owner(void)
{
	setenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256,"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",1);
	setenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES","4096",1);
	unsetenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
	for (uint32_t mode=0u; mode<2u; mode++)
	{
		memset(&state,0,sizeof(state));
		strcpy(state.model_revision,"test");
		LAZY_OPEN_CALLS = 0u;LAZY_OPEN_MODE = mode;
		assert(SparkGlm5NextLazyOpen(&state,"fixture.pack",8192u,0,0u) == (mode == 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK));
		assert(LAZY_OPEN_CALLS == mode + 1u && state.lazy_pack == &OPEN_PACK);
	}
	unsetenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	unsetenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES");
}

static uint32_t PIN_FIRST_EXPECTED,PIN_LAST_EXPECTED;
static uint32_t PIN_CALLS,PIN_KEYS,PIN_RECORDS,PIN_RELEASES,PIN_FAIL_ACQUIRE,PIN_FAIL_BEGIN,PIN_FAIL_RECORD,PIN_FAIL_RELEASE;
static uint8_t PIN_PHASES[33],PIN_SEEN[SPARK_GLM5_NEXT_MODEL_LAYER_COUNT][SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT];

SparkStatus SparkWeightdMapAcquire(SparkWeightdMap *map,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier,uint64_t timeout)
{
	(void)timeout;
	assert(map == (SparkWeightdMap *)(uintptr_t)1u && count > 0u && count <= SPARK_WEIGHTD_LEASE_GROUPS_MAX);
	*identifier = ++PIN_CALLS;
	assert(PIN_CALLS < 33u);
	for (uint32_t i=0u; i<count; i++)
	{
		assert(keys[i].layer >= PIN_FIRST_EXPECTED && keys[i].layer < PIN_LAST_EXPECTED);
		assert(keys[i].expert < SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT && PIN_SEEN[keys[i].layer][keys[i].expert] == 0u);
		PIN_SEEN[keys[i].layer][keys[i].expert] = 1u;
		PIN_KEYS++;
	}
	return(PIN_CALLS == PIN_FAIL_ACQUIRE ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapBeginUse(SparkWeightdMap *map,uint64_t identifier,void **base)
{
	(void)map;
	assert(identifier <= PIN_CALLS && PIN_PHASES[identifier] == 0u);
	if ( identifier == PIN_FAIL_BEGIN ) return(SPARK_STATUS_IO_ERROR);
	PIN_PHASES[identifier] = 1u;
	*base = (void *)(uintptr_t)64u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t identifier,cudaStream_t stream)
{
	(void)map;(void)stream;
	assert(identifier <= PIN_CALLS && PIN_PHASES[identifier] == 1u);
	PIN_RECORDS++;
	if ( identifier == PIN_FAIL_RECORD ) return(SPARK_STATUS_IO_ERROR);
	PIN_PHASES[identifier] = 2u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout)
{
	(void)map;(void)timeout;
	assert(identifier <= PIN_CALLS && (PIN_PHASES[identifier] == 0u || PIN_PHASES[identifier] == 2u));
	PIN_RELEASES++;
	if ( identifier == PIN_FAIL_RELEASE ) return(SPARK_STATUS_IO_ERROR);
	PIN_PHASES[identifier] = 3u;
	return(SPARK_STATUS_OK);
}

static void check_graph_expert_ownership(uint32_t first,uint32_t layers,uint32_t expected_first,uint32_t acquire_error,uint32_t begin_error)
{
	SparkWeightdLazyPack pack = {0};
	SparkGlm5NextTpChain chain = {0};
	uint32_t expected,leases;
	memset(&state,0,sizeof(state));
	memset(PIN_PHASES,0,sizeof(PIN_PHASES));
	memset(PIN_SEEN,0,sizeof(PIN_SEEN));
	PIN_CALLS = PIN_KEYS = PIN_RECORDS = PIN_RELEASES = PIN_FAIL_RECORD = PIN_FAIL_RELEASE = 0u;
	PIN_FAIL_ACQUIRE = acquire_error;
	PIN_FAIL_BEGIN = begin_error;
	pack.map = (SparkWeightdMap *)(uintptr_t)1u;
	state.lazy_pack = &pack;
	state.first_layer_index = first;
	state.layer_count = layers;
	PIN_FIRST_EXPECTED = expected_first;
	PIN_LAST_EXPECTED = first + layers;
	state.experts_warm = 1u;
	state.decode_lease_base_saved = (const uint8_t *)(uintptr_t)64u;
	chain.state = &state;
	expected = (first + layers - expected_first) * SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT;
	assert(SparkGlm5NextGraphClaimExperts(&chain) == SPARK_STATUS_UNSUPPORTED && chain.wave.expert_lease_all == 0u);
	assert(SparkGlm5NextPinAllExperts(&state) == (acquire_error != 0u || begin_error != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK));
	leases = state.expert_pin_lease_count;
	if ( acquire_error != 0u || begin_error != 0u )
	{
		assert(leases == 2u && state.expert_pin_key_count == SPARK_WEIGHTD_LEASE_GROUPS_MAX);
		assert(SparkGlm5NextGraphClaimExperts(&chain) == SPARK_STATUS_UNSUPPORTED);
		assert(SparkGlm5NextReleasePinnedExperts(&state) == SPARK_STATUS_OK);
		assert(PIN_PHASES[1] == 3u && PIN_PHASES[2] == 3u && PIN_RECORDS == 1u && PIN_RELEASES == 2u);
		return;
	}
	assert(PIN_KEYS == expected && state.expert_pin_key_count == expected);
	for (uint32_t layer=expected_first; layer<first + layers; layer++)
		for (uint32_t expert=0u; expert<SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT; expert++) assert(PIN_SEEN[layer][expert] == 1u);
	assert(SparkGlm5NextPinAllExperts(&state) == SPARK_STATUS_BUSY);
	assert(SparkGlm5NextGraphClaimExperts(&chain) == SPARK_STATUS_OK && chain.wave.expert_lease_all == 1u);
	state.expert_pin_phases[0] = 0u;
	assert(SparkGlm5NextGraphClaimExperts(&chain) == SPARK_STATUS_VALIDATION_FAILED);
	state.expert_pin_phases[0] = 1u;
	PIN_FAIL_RECORD = leases;
	assert(SparkGlm5NextReleasePinnedExperts(&state) == SPARK_STATUS_IO_ERROR && state.expert_pin_lease_count == leases && PIN_RELEASES == 0u);
	PIN_FAIL_RECORD = 0u;
	PIN_FAIL_RELEASE = leases;
	assert(SparkGlm5NextReleasePinnedExperts(&state) == SPARK_STATUS_IO_ERROR && state.expert_pin_lease_count == leases);
	assert(SparkGlm5NextGraphClaimExperts(&chain) == SPARK_STATUS_VALIDATION_FAILED);
	PIN_FAIL_RELEASE = 0u;
	assert(SparkGlm5NextReleasePinnedExperts(&state) == SPARK_STATUS_OK && state.expert_pin_lease_count == 0u && state.expert_pin_key_count == 0u);
	assert(PIN_RECORDS == leases + 1u && PIN_RELEASES == leases + 1u);
}

int32_t SparkGlm5NextLaunchCudaMtpCommit(const SparkGlm5NextCudaWave *wave,uint32_t committed_steps)
{
	assert(IN_CUDA_CALLBACK == 0u && wave != 0 && committed_steps == 1u);
	MTP_COMMITS++;
	return(0);
}

static void check_mtp_callback_handoff(SparkStatus submit_status,uint32_t release_failure)
{
	uint16_t hidden[SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION],saved[SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION];
	uint32_t tokens[3] = {11u,12u,13u},errors[6] = {0},host_errors[6] = {0};
	SparkWeightdLazyPack pack = {0};
	SparkGlm5NextTpChain *chain = calloc(1u,sizeof(*chain));
	SparkWeightdWorkFunction resolver;
	void *resolver_context;
	memset(&state,0,sizeof(state));
	memset(hidden,0xa5,sizeof(hidden));
	memset(saved,0,sizeof(saved));
	assert(chain != 0 && pthread_mutex_init(&state.completion_queue_lock,0) == 0);
	state.completion_worker = (SparkWeightdWorker *)(uintptr_t)1u;
	state.pipeline_slot_count = 1u;
	state.execution_stream = state.slots[0].stream = (void *)(uintptr_t)7u;
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	state.mtp_lane_hidden_bf16 = saved;
	state.completions[0].state = &state;
	state.completions[0].mtp_draft_tokens[0] = 10u;
	state.completions[0].mtp_draft_tokens[1] = 12u;
	state.slots[0].host_output_token_ids = tokens;
	state.slots[0].hc_mean_bf16 = hidden;
	state.slots[0].kv_access_error = errors;
	state.slots[0].host_kv_access_error = host_errors;
	chain->state = &state;
	chain->slot = &state.slots[0];
	chain->active = 1u;
	DRAIN_STATUS = cudaSuccess;
	if ( release_failure != 0u )
	{
		pack.map = (SparkWeightdMap *)(uintptr_t)1u;
		state.lazy_pack = &pack;
		chain->expert_lease = 1u;
		chain->expert_lease_begun = 1u;
		PIN_CALLS = PIN_FAIL_RELEASE = 1u;
		PIN_RECORDS = PIN_RELEASES = PIN_FAIL_RECORD = 0u;
		PIN_PHASES[1] = 1u;
	}
	atomic_store(&state.tp_chain_active,1u);
	atomic_store(&state.slot_states[0],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	WORK_STATUS = submit_status;
	COPY_COUNT = MTP_COMMITS = HOST_COUNT = 0u;
	HOST_MODE = 0u;
	COMPLETION_WORK = 0;
	IN_CUDA_CALLBACK = 1u;
	SparkGlm5NextMtpResolveHost(chain);
	IN_CUDA_CALLBACK = 0u;
	assert(COPY_COUNT == 0u && MTP_COMMITS == 0u && chain->active == 1u);
	if ( submit_status != SPARK_STATUS_OK )
	{
		assert(COMPLETION_WORK == 0 && state.overflow_parked_count == 1u);
		WORK_STATUS = SPARK_STATUS_OK;
		pthread_mutex_lock(&state.completion_queue_lock);
		SparkGlm5NextDrainParkedCompletions(&state);
		pthread_mutex_unlock(&state.completion_queue_lock);
	}
	resolver = COMPLETION_WORK;
	resolver_context = COMPLETION_CONTEXT;
	assert(resolver == SparkGlm5NextMtpResolveOnWorker && resolver_context == chain && state.overflow_parked_count == 0u);
	COMPLETION_WORK = 0;
	resolver(resolver_context);
	assert(MTP_COMMITS == 1u && memcmp(hidden,saved,sizeof(hidden)) == 0);
	assert(state.completions[0].completion.accepted_token_count == 1u);
	if ( release_failure != 0u )
	{
		assert(COMPLETION_WORK == 0 && atomic_load(&state.lazy_retained[0]) == chain);
		assert(chain->expert_lease == 1u && chain->active == 0u && chain->expert_lease_recorded == 1u);
		assert(PIN_RECORDS == 1u && PIN_RELEASES == 2u && PIN_PHASES[1] == 2u);
		assert(atomic_load(&state.tp_chain_active) == 1u && atomic_load(&state.slot_states[0]) == SPARK_STAGE_MODULE_SLOT_CLAIMED);
		PIN_FAIL_RELEASE = 0u;
		SparkGlm5NextLazyRetryRetained(&state);
		assert(atomic_load(&state.lazy_retained[0]) == 0);
		assert(state.completions[0].completion.status == SPARK_STATUS_IO_ERROR);
		assert(PIN_RECORDS == 1u && PIN_RELEASES == 3u && PIN_PHASES[1] == 3u);
	}
	assert(COMPLETION_WORK == SparkGlm5NextCompleteOnWorker && COMPLETION_CONTEXT == &state.completions[0]);
	assert(atomic_load(&state.tp_chain_active) == 1u && atomic_load(&state.slot_states[0]) == SPARK_STAGE_MODULE_SLOT_CLAIMED);
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
	assert(pthread_mutex_destroy(&state.completion_queue_lock) == 0);
}

static SparkGlm5NextModuleState *DESTROY_STATE;
static uint32_t DESTROY_FAIL,DESTROY_CALLS[3],DESTROY_LAZY;

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
	uint32_t which = collective == &DESTROY_STATE->tp_device_collective_hc ? 1u : 2u;
	assert(DESTROY_LAZY == 0u && DESTROY_STATE->lazy_pack != 0);
	assert(DESTROY_STATE->lazy_pack->attached.mesh_mapping != 0);
	assert(collective->implementation != 0);
	DESTROY_CALLS[which]++;
	if ( DESTROY_FAIL != which ) collective->implementation = 0;
}

SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack)
{
	assert(pack == DESTROY_STATE->lazy_pack && DESTROY_LAZY == 0u);
	assert(DESTROY_STATE->tp_device_collective.implementation == 0);
	assert(DESTROY_STATE->tp_device_collective_hc.implementation == 0);
	pack->attached.mesh_mapping = 0;
	DESTROY_LAZY++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdWorkerWaitIdle(SparkWeightdWorker *worker,uint64_t timeout)
{
	(void)worker;(void)timeout;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdWorkerDestroy(SparkWeightdWorker *worker)
{
	(void)worker;
	return(SPARK_STATUS_OK);
}

void SparkWeightdClientClose(SparkWeightdClient *client)
{
	(void)client;
}

void SparkWeightdAttachRelease(SparkWeightdAttachOutcome *outcome)
{
	(void)outcome;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	(void)event;
	return(cudaSuccess);
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph)
{
	(void)graph;
	return(cudaSuccess);
}

static void check_collective_destroy_order(uint32_t failure)
{
	SparkWeightdLazyPack pack = {0};
	SparkGlm5NextModuleState *owner = calloc(1u,sizeof(*owner));
	assert(owner != 0);
	owner->pipeline_slot_count = 1u;
	owner->lazy_pack = &pack;
	pack.attached.mesh_mapping = (void *)(uintptr_t)1u;
	owner->tp_device_collective.implementation = (void *)(uintptr_t)1u;
	owner->tp_device_collective_hc.implementation = (void *)(uintptr_t)2u;
	owner->tp_device_collective_initialized = owner->tp_device_collective_hc_initialized = 1u;
	DESTROY_STATE = owner;
	DESTROY_FAIL = failure;
	DESTROY_CALLS[1] = DESTROY_CALLS[2] = DESTROY_LAZY = 0u;
	DRAIN_STATUS = cudaSuccess;
	SparkGlm5NextResidentDecodeStageDestroy(owner);
	if ( failure != 0u )
	{
		assert(DESTROY_LAZY == 0u && owner->lazy_pack == &pack && pack.attached.mesh_mapping != 0);
		assert(owner->tp_device_collective.implementation != 0);
		assert((owner->tp_device_collective_hc.implementation != 0) == (failure == 1u));
		DESTROY_FAIL = 0u;
		SparkGlm5NextResidentDecodeStageDestroy(owner);
	}
	assert(DESTROY_LAZY == 1u && pack.attached.mesh_mapping == 0);
	assert(DESTROY_CALLS[1] == 1u + (failure == 1u));
	assert(DESTROY_CALLS[2] == 1u + (failure == 2u));
	DESTROY_STATE = 0;
}

static void check_callback_retirement(void)
{
	SparkStageModuleCudaWait wait = {0};
	uint64_t started;
	assert(SparkStageModuleCudaWaitInitialize(&wait,(cudaStream_t)(uintptr_t)1u) == SPARK_STATUS_OK);
	DRAIN_STATUS = cudaErrorNotReady;HOST_MODE = 2u;HOST_COUNT = STREAM_QUERY_COUNT = 0u;
	started = SparkGlm5NextNowNs();
	assert(SparkStageModuleCudaWaitFor(&wait,UINT64_C(500000000)) == SPARK_STATUS_OK);
	assert(SparkGlm5NextNowNs() - started >= UINT64_C(30000000));
	assert(DRAIN_STATUS == cudaSuccess && HOST_COUNT == 1u);
	assert(pthread_join(HOST_THREAD,0) == 0);
	assert(SparkStageModuleCudaWaitDestroy(&wait) == SPARK_STATUS_OK);
	HOST_MODE = 0u;
}

static void check_stream_receipt(void)
{
	SparkStageModuleCudaWait first = {0},second = {0};
	uint64_t start;
	uint32_t count;
	assert(SparkStageModuleCudaWaitInitialize(&first,(cudaStream_t)(uintptr_t)1u) == SPARK_STATUS_OK);
	assert(SparkStageModuleCudaWaitInitialize(&second,(cudaStream_t)(uintptr_t)2u) == SPARK_STATUS_OK);
	HOST_COUNT = STREAM_QUERY_COUNT = 0u;
	DRAIN_STATUS = cudaSuccess;
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_OK);
	assert(HOST_COUNT == 0u && STREAM_QUERY_COUNT == 1u);
	DRAIN_STATUS = cudaErrorNotReady;
	HOST_MODE = 0u;
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_OK);
	assert(first.generation == 1u && first.completed_generation == 1u && HOST_COUNT == 1u);
	HOST_MODE = 1u;
	DRAIN_STATUS = cudaErrorNotReady;
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_BUSY);
	assert(SparkStageModuleCudaWaitDestroy(&first) == SPARK_STATUS_BUSY);
	count = STREAM_QUERY_COUNT;
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_BUSY);
	assert(STREAM_QUERY_COUNT == count && HOST_COUNT == 2u && first.generation == 2u);
	assert(SparkStageModuleCudaWaitFor(&second,UINT64_C(1000000)) == SPARK_STATUS_BUSY);
	HOST_FUNCTIONS[1](HOST_CONTEXTS[1]);
	assert(first.completed_generation == 2u && second.completed_generation == 0u);
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_BUSY);
	assert(HOST_COUNT == 3u && first.generation == 2u);
	assert(SparkStageModuleCudaWaitDestroy(&first) == SPARK_STATUS_BUSY);
	DRAIN_STATUS = cudaSuccess;
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_OK);
	assert(SparkStageModuleCudaWaitDestroy(&first) == SPARK_STATUS_OK);
	assert(SparkStageModuleCudaWaitDestroy(&second) == SPARK_STATUS_BUSY);
	HOST_FUNCTIONS[2](HOST_CONTEXTS[2]);
	assert(SparkStageModuleCudaWaitFor(&second,UINT64_C(1000000)) == SPARK_STATUS_OK);
	assert(SparkStageModuleCudaWaitDestroy(&second) == SPARK_STATUS_OK);
	assert(SparkStageModuleCudaWaitInitialize(&first,(cudaStream_t)(uintptr_t)3u) == SPARK_STATUS_OK);
	HOST_MODE = 2u;
	DRAIN_STATUS = cudaErrorNotReady;
	count = STREAM_QUERY_COUNT;
	start = SparkGlm5NextNowNs();
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(500000000)) == SPARK_STATUS_OK);
	assert(SparkGlm5NextNowNs() - start >= UINT64_C(30000000));
	assert(SparkGlm5NextNowNs() - start < UINT64_C(250000000));
	assert(pthread_join(HOST_THREAD,0) == 0);
	assert(STREAM_QUERY_COUNT > count + 1u && HOST_COUNT == 4u);
	HOST_MODE = 3u;
	DRAIN_STATUS = cudaErrorNotReady;
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_IO_ERROR);
	assert(first.generation == first.completed_generation);
	DRAIN_STATUS = cudaErrorInvalidValue;
	count = HOST_COUNT;
	assert(SparkStageModuleCudaWaitFor(&first,UINT64_C(1000000)) == SPARK_STATUS_IO_ERROR);
	assert(HOST_COUNT == count);
	assert(SparkStageModuleCudaWaitDestroy(&first) == SPARK_STATUS_OK);
	DRAIN_STATUS = cudaSuccess;
	HOST_MODE = 0u;
}

static void check_chain_ownership(void)
{
	uint32_t reason = SPARK_GLM5_NEXT_BUSY_REASONS;
	memset(&state,0,sizeof(state));
	state.execution_stream = (void *)(uintptr_t)7u;
	DRAIN_STATUS = cudaErrorNotReady;
	STREAM_QUERY_COUNT = 0u;
	assert(SparkGlm5NextClaimTpChain(&state,&reason) == SPARK_STATUS_BUSY && reason == SPARK_GLM5_NEXT_BUSY_STREAM);
	assert(STREAM_QUERY_COUNT == 1u && atomic_load(&state.tp_chain_active) == 0u);
	DRAIN_STATUS = cudaSuccess;
	assert(SparkGlm5NextClaimTpChain(&state,&reason) == SPARK_STATUS_OK);
	assert(STREAM_QUERY_COUNT == 2u && atomic_load(&state.tp_chain_active) == 1u);
	assert(SparkGlm5NextClaimTpChain(&state,&reason) == SPARK_STATUS_BUSY && reason == SPARK_GLM5_NEXT_BUSY_CHAIN);
	assert(STREAM_QUERY_COUNT == 2u && atomic_load(&state.tp_chain_active) == 1u);
	atomic_store(&state.tp_chain_active,0u);
	DRAIN_STATUS = cudaErrorInvalidValue;
	assert(SparkGlm5NextClaimTpChain(&state,&reason) == SPARK_STATUS_IO_ERROR);
	assert(atomic_load(&state.terminal_status) == SPARK_STATUS_IO_ERROR && atomic_load(&state.tp_chain_active) == 0u);
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	DRAIN_STATUS = cudaErrorNotReady;
	HOST_MODE = 2u;
	HOST_COUNT = STREAM_QUERY_COUNT = 0u;
	assert(SparkGlm5NextBoundedStreamSync(&state,(void *)(uintptr_t)8u,UINT64_C(500000000)) == -1);
	assert(SparkGlm5NextBoundedStreamSync(&state,state.execution_stream,UINT64_C(500000000)) == 0);
	assert(pthread_join(HOST_THREAD,0) == 0 && STREAM_QUERY_COUNT >= 2u && HOST_COUNT == 1u);
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
	DRAIN_STATUS = cudaSuccess;
	HOST_MODE = 0u;
}

static int32_t check_cache_transactions(SparkStatus completion_status,SparkStatus verify_status,SparkStatus end_status,cudaError_t drain_status)
{
	SparkStatus expected = completion_status != SPARK_STATUS_OK ? completion_status : verify_status;
	SparkTestKvTransactions fixture;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverFrame frame;
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	uint32_t slots[2] = {0,1},device_table[16],shadow[16],errors[6] = {0};
	uint64_t positions[2] = {0,0},sequences[2] = {1,2},next[2] = {1,1};
	SparkTestKvTransactionsInitialize(&fixture,2u);
	(void)SparkTestKvAcquire(&fixture.pages.kv);
	memset(&state,0,sizeof(state));
	memset(device_table,0xff,sizeof(device_table));
	memset(shadow,0xff,sizeof(shadow));
	state.pipeline_slot_count = 2u;
	state.execution_row_capacity = 1u;
	state.resident_sequence_capacity = 4u;
	state.pages_per_sequence = 4u;
	state.kv_transactions = fixture.transactions;
	state.kv_lane_transactions = fixture.owners;
	state.kv_lane_physical_pages = fixture.physical;
	state.page_table = device_table;
	state.page_table_shadow = shadow;
	if ( pthread_mutex_init(&state.kv_mutex,0) != 0 )
		return(-20);
	assert(SparkGlm5NextResidentDecodeStageAdmit(&state,&fixture.request,&decision) == SPARK_STATUS_OK);
	assert(decision.accepted == 0u && decision.rejection_reason == SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
	assert(fixture.owners[0].phase == SPARK_KV_LANE_TRANSACTION_EMPTY && fixture.owners[1].phase == SPARK_KV_LANE_TRANSACTION_EMPTY);
	state.execution_row_capacity = 2u;
	SparkModelDriverInitializeAdmissionDecision(&decision);
	if ( SparkGlm5NextAdmissionPredicate(&state,&fixture.request,&decision) != SPARK_STATUS_OK || SparkModelDriverAdmissionDecisionIsValid(&decision) == 0u )
		return(-21);
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	if ( SparkGlm5NextAdmissionPredicate(&state,&fixture.request,&decision) != SPARK_STATUS_OK )
		return(-22);
	fixture.request.admission_flags = 0u;
	frame = SparkTestKvTransactionFrame(&fixture.request);
	if ( SparkGlm5NextAdmissionPredicate(&state,&fixture.request,&decision) != SPARK_STATUS_OK || SparkModelDriverApplyAdmissionDecision(&decision,&frame) != SPARK_STATUS_OK )
		return(-23);
	batch.active_sequence_count = 2u;
	batch.row_resident_slots = slots;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequences;
	if ( SparkGlm5NextClaimCacheFrame(&state,&frame,&batch,next) != SPARK_STATUS_OK )
		return(-24);
	state.completions[0].state = &state;
	state.completions[0].lane_count = 2u;
	state.completions[0].lane_indices[1] = 1u;
	COPY_COUNT = 0u;
	if ( SparkGlm5NextUploadPageTables(&state,&state.completions[0],0) != SPARK_STATUS_OK || COPY_COUNT != 2u || device_table[0] != fixture.physical[0] || device_table[4] != fixture.physical[4] || device_table[0] == fixture.logical[0] )
		return(-25);
	if ( SparkGlm5NextUploadPageTables(&state,&state.completions[0],0) != SPARK_STATUS_OK || COPY_COUNT != 2u )
		return(-26);
	state.execution_stream = (void *)(uintptr_t)7u;
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	assert(pthread_mutex_init(&state.completion_queue_lock,0) == 0);
	state.decode_miss_host = (uint32_t *)EPOCH_WORDS;
	memset(EPOCH_WORDS,0,sizeof(EPOCH_WORDS));
	EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64] = 42u;
	LEGACY_EPOCH_SETUP
	state.completions[0].completion.status = completion_status;
	state.completions[0].completion_function = observe_completion;
	state.slots[0].host_kv_access_error = errors;
	state.completion_worker = (SparkWeightdWorker *)(uintptr_t)1u;
	atomic_store(&state.slot_states[0],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	atomic_store(&state.lane_states[0],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	atomic_store(&state.lane_states[1],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	atomic_store(&state.tp_chain_active,1u);
	state.tp_device_collective_initialized = state.tp_device_collective_hc_initialized = 1u;
	EXPECTED_CANCEL_COUNT = expected != SPARK_STATUS_OK ? 2u : 0u;
	CANCEL_COUNT = END_COUNT = COMPLETION_REUSED = VERIFY_COUNT = 0u;
	VERIFY_STATUS = verify_status;
	END_STATUS = end_status;
	DRAIN_STATUS = drain_status;
	COMPLETION_WORK = 0;
	WORK_STATUS = SPARK_STATUS_BUSY;
	SparkGlm5NextCompleteAsync(&state.completions[0]);
	if ( COMPLETION_WORK != 0 || atomic_load(&state.slot_states[0]) != SPARK_STAGE_MODULE_SLOT_CLAIMED || fixture.owners[0].phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		return(-28);
	WORK_STATUS = SPARK_STATUS_OK;
	pthread_mutex_lock(&state.completion_queue_lock);
	SparkGlm5NextDrainParkedCompletions(&state);
	pthread_mutex_unlock(&state.completion_queue_lock);
	if ( COMPLETION_WORK == 0 || atomic_load(&state.slot_states[0]) != SPARK_STAGE_MODULE_SLOT_CLAIMED || fixture.owners[0].phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		return(-29);
	COMPLETION_WORK(COMPLETION_CONTEXT);
	VERIFY_STATUS = SPARK_STATUS_OK;
	assert(CANCEL_COUNT == EXPECTED_CANCEL_COUNT && VERIFY_COUNT == (drain_status == cudaSuccess ? 2u : 0u));
	if ( drain_status != cudaSuccess || end_status != SPARK_STATUS_OK )
	{
		assert(COMPLETION_REUSED == 0u && atomic_load(&state.tp_chain_active) == 1u);
		assert(atomic_load(&state.slot_states[0]) == SPARK_STAGE_MODULE_SLOT_CLAIMED);
		assert(atomic_load(&state.lane_states[0]) == SPARK_STAGE_MODULE_SLOT_CLAIMED);
		assert(END_COUNT == (drain_status != cudaSuccess ? 0u : 1u));
		assert(fixture.owners[0].phase == (drain_status != cudaSuccess ? SPARK_KV_LANE_TRANSACTION_EXECUTING : SPARK_KV_LANE_TRANSACTION_EMPTY));
		assert(atomic_load(&state.terminal_status) != SPARK_STATUS_OK);
		DRAIN_STATUS = cudaSuccess;
		assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
		assert(pthread_mutex_destroy(&state.completion_queue_lock) == 0);
		pthread_mutex_destroy(&state.kv_mutex);
		memset(&state,0,sizeof(state));
		return(0);
	}
	assert(END_COUNT == 2u);
	if ( COMPLETION_REUSED == 0u )
		return(-30);
	if ( COMPLETION_STATUS != expected || atomic_load(&state.slot_states[0]) != SPARK_STAGE_MODULE_SLOT_FREE || atomic_load(&state.lane_states[0]) != SPARK_STAGE_MODULE_SLOT_FREE || atomic_load(&state.lane_states[1]) != SPARK_STAGE_MODULE_SLOT_FREE || (expected != SPARK_STATUS_OK && (fixture.pages.cache.sequences[0].sequence_id != 0u || fixture.pages.cache.sequences[1].sequence_id != 0u || shadow[0] != UINT32_MAX)) )
		return(-27);
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
	assert(pthread_mutex_destroy(&state.completion_queue_lock) == 0);
	pthread_mutex_destroy(&state.kv_mutex);
	memset(&state,0,sizeof(state));
	return(0);
}

SparkStatus SparkKvBackendInitialize(const SparkKvModelTable *table,SparkKvCacheArena *arena,SparkKvPageCache *cache,SparkKvPageStore *store)
{
	assert(table->arena_configuration.logical_block_count == state.page_count);
	assert(table->arena_configuration.resident_block_capacity == state.physical_page_count);
	(void)arena;
	(void)cache;
	(void)store;
	assert(SparkKvPageStoreConfigurationIsValid(&table->page_store_config) != 0u);
	assert(table->page_store_config.transfer_capacity <= 2u);
	assert(table->page_store_config.page_bytes == table->page_store_config.staging_bytes);
	assert(table->page_store_config.maximum_backing_bytes == state.page_count * table->page_store_config.page_bytes);
	assert(table->arena_configuration.value_device_base == state.index_cache);
	assert(table->arena_configuration.value_block_stride_bytes == (uint64_t)state.index_layer_count * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
	if ( REAL_BACKEND != 0u )
	{
		SparkKvModelTable named_store = *table;
		named_store.page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
		return(SparkTestRealKvBackendInitialize(&named_store,arena,cache,store));
	}
	return(SPARK_STATUS_PENDING);
}

static int32_t check_cache_release(void)
{
	SparkTestKvTransactions fixture;
	SparkModelDriverFrame frame;
	SparkModelDriverAdmissionDecision decision;
	SparkTestKvTransactionsInitialize(&fixture,2u);
	memset(&state,0,sizeof(state));
	state.kv_transactions = fixture.transactions;
	state.pipeline_slot_count = 1u;
	if ( pthread_mutex_init(&state.kv_mutex,0) != 0 )
		return(-30);
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK )
		return(-31);
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK )
		return(-32);
	frame = SparkTestKvTransactionFrame(&fixture.request);
	if ( SparkKvLaneTransactionsClaim(&fixture.transactions,&frame) != SPARK_STATUS_OK || SparkKvLaneTransactionsFinish(&fixture.transactions,(uint32_t[]){0u,1u},2u,SPARK_STATUS_OK,0u) != SPARK_STATUS_OK )
		return(-33);
	atomic_store(&state.lane_bound[0],1u);
	atomic_store(&state.lane_bound[1],1u);
	fixture.request.admission_flags = 0u;
	fixture.request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	fixture.request.new_token_count = 0u;
	fixture.lanes[0].sequence_position = fixture.lanes[1].sequence_position = 1u;
	fixture.lanes[0].flags = fixture.lanes[1].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	if ( SparkGlm5NextResidentDecodeStageAdmit(&state,&fixture.request,&decision) != SPARK_STATUS_OK || SparkModelDriverAdmissionDecisionIsValid(&decision) == 0u )
		return(-34);
	if ( fixture.pages.cache.live_sequence_count != 0u || atomic_load(&state.lane_bound[0]) != 0u || atomic_load(&state.lane_bound[1]) != 0u )
		return(-35);
	pthread_mutex_destroy(&state.kv_mutex);
	memset(&state,0,sizeof(state));
	return(0);
}

static int32_t check_batch_waves(void)
{
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	SparkGlm5NextResidentDecodeStageFrameContext context = {.batch=&batch};
	SparkGlm5NextTpChain chain = {.state=&state,.slot=&state.slots[0],.context=&context,.batch=&batch};
	SparkGlm5NextExecutionSlot *slot = &state.slots[0];
	uint32_t slots[303],positions[303] = {0},begin[102],indices[303],runs[101],width,row,lane;
	uint32_t ragged[8] = {5u,2u,9u,5u,2u,9u,5u,9u};
	uint32_t expected[8] = {0u,3u,6u,1u,4u,2u,5u,7u};
	atomic_uint *claims = state.lane_states;
	state.resident_sequence_capacity = 101u;
	state.tp_degree = 1u;
	slot->host_positions = positions;
	slot->host_run_begin = begin;
	slot->host_run_row_indices = indices;
	slot->host_run_state_index = runs;
	for (row=0u; row<101u; row++)
		atomic_init(&claims[row],SPARK_STAGE_MODULE_SLOT_FREE);
	batch.row_resident_slots = slot->host_resident_slots = slots;
	for (width=1u; width<=101u; width++)
	{
		batch.active_sequence_count = width;
		chain.wave_rows = batch.row_count = width * 3u;
		for (row=0u; row<batch.row_count; row++)
			slots[row] = width - 1u - row % width;
		assert(SparkGlm5NextValidateRoundMajor(&state,&batch) == SPARK_STATUS_OK);
		assert(SparkStageModuleIndexSetClaim(claims,101u,slots,width) == SPARK_STATUS_OK);
		assert(SparkGlm5NextBuildWave(&chain) == SPARK_STATUS_OK);
		assert(chain.wave.row_count == 3u * width && chain.wave.run_count == width);
		for (lane=0u; lane<width; lane++)
		{
			assert(runs[lane] == slots[lane] && begin[lane] == lane * 3u);
			for (row=0u; row<3u; row++)
				assert(indices[begin[lane] + row] == lane + row * width);
		}
		assert(begin[width] == batch.row_count);
		SparkStageModuleIndexSetRelease(claims,101u,slots,width);
	}
	batch.row_resident_slots = slot->host_resident_slots = ragged;
	batch.active_sequence_count = 3u;
	chain.wave_rows = batch.row_count = 8u;
	assert(SparkGlm5NextValidateRoundMajor(&state,&batch) == SPARK_STATUS_OK);
	assert(SparkGlm5NextBuildWave(&chain) == SPARK_STATUS_OK);
	assert(chain.wave.row_count == 8u && chain.wave.run_count == 3u);
	assert(begin[0] == 0u && begin[1] == 3u && begin[2] == 5u && begin[3] == 8u);
	assert(memcmp(indices,expected,sizeof(expected)) == 0);
	ragged[6] = 9u;
	ragged[7] = 5u;
	assert(SparkGlm5NextValidateRoundMajor(&state,&batch) != SPARK_STATUS_OK);
	ragged[7] = 10u;
	assert(SparkGlm5NextBuildWave(&chain) == SPARK_STATUS_INVALID_ARGUMENT);
	memset(slot,0,sizeof(*slot));
	return(0);
}

static void free_cache_fixture(void)
{
	SparkGlm5NextReleaseCaches(&state);
	SparkStageModuleLedgerRollback(&state.ledger,0u);
}

static void check_cache_worker_cleanup(void)
{
	SparkKvPageStoreConfiguration config = {0};
	uint8_t source[32] = {1u};
	char path[] = "/tmp/glm-cache-cleanup-XXXXXX";
	int32_t descriptor;
	memset(&state,0,sizeof(state));
	descriptor = mkstemp(path);
	assert(descriptor >= 0 && close(descriptor) == 0 && unlink(path) == 0);
	state.kv_page_staging = malloc(sizeof(source));
	assert(state.kv_page_staging != 0 && pthread_mutex_init(&state.kv_mutex,0) == 0);
	state.kv_mutex_initialized = 1u;
	config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	config.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	config.logical_page_capacity = config.transfer_capacity = 1u;
	config.page_bytes = config.maximum_backing_bytes = config.staging_bytes = sizeof(source);
	config.staging_address = state.kv_page_staging;
	config.backing_path = path;
	assert(SparkKvPageStoreInitialize(&state.kv_page_store,&config) == SPARK_STATUS_OK);
	descriptor = state.kv_page_store.file_descriptor;
	assert(SparkKvPageStoreWriteback(&state.kv_page_store,0u,0u,1u,(uintptr_t)source,sizeof(source),0u,0u) == SPARK_STATUS_BUSY);
	SparkGlm5NextReleaseCaches(&state);
	assert(state.kv_page_store.worker_state == 0 && state.kv_page_store.file_descriptor == -1);
	errno = 0;
	assert(fcntl(descriptor,F_GETFD) == -1 && errno == EBADF);
	assert(unlink(path) == 0);
}

static int32_t check_rank_state(void)
{
	uint32_t degrees[3] = {1u,4u,16u},index,allocation;
	uint64_t state_bytes,window_bytes,actual_state = 0u,actual_window = 0u;
	for (index=0u; index<3u; index++)
	{
		memset(&state,0,sizeof(state));
		state.ledger.module_tag = "rank-state-test";
		state.tp_degree = degrees[index];
		state.layer_count = 4u;
		state.resident_sequence_capacity = 3u;
		state.max_sequence_positions = 64u;
		state.page_count = 7u;
		state.physical_page_count = 2u;
		state.kv_backing_directory = "/unused-host-fixture";
		if ( SparkGlm5NextAllocateCaches(&state) != SPARK_STATUS_PENDING || state.kda_layer_count != 3u )
			return(-8);
		assert(state.page_count == 7u && state.physical_page_count == 2u);
		assert(state.kv_layer_stride_bytes == 2u * 64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES);
		assert(state.index_layer_stride_bytes == 2u * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
		state_bytes = (uint64_t)(64u / degrees[index]) * 128u * 128u * sizeof(float);
		window_bytes = (uint64_t)(64u / degrees[index]) * 128u * 4u * sizeof(uint16_t);
		if ( state.kda_state_layer_stride_bytes != 3u * state_bytes || state.kda_window_layer_stride_bytes != 3u * window_bytes )
			return(-9);
		for (allocation=0u; allocation<state.ledger.device_allocation_count; allocation++)
		{
			if ( state.ledger.device_allocations[allocation] == state.kda_state_pools )
				actual_state = state.ledger.device_allocation_bytes[allocation];
			if ( state.ledger.device_allocations[allocation] == state.kda_window_pools )
				actual_window = state.ledger.device_allocation_bytes[allocation];
		}
		if ( actual_state != 9u * state_bytes || actual_window != 27u * window_bytes )
			return(-10);
		if ( state.kda_k_window_pool != state.kda_q_window_pool + 9u * window_bytes || state.kda_v_window_pool != state.kda_k_window_pool + 9u * window_bytes )
			return(-11);
		assert(state.recurrent_page_bytes == 3u * (state_bytes + 3u * window_bytes));
		state.kv_backing_maximum_bytes = state.page_count * (state.recurrent_page_bytes + 128u);
		assert(SparkGlm5NextBackingCapacity(&state,128u) == SPARK_STATUS_OK);
		state.kv_backing_maximum_bytes--;
		assert(SparkGlm5NextBackingCapacity(&state,128u) == SPARK_STATUS_CAPACITY_EXCEEDED);
		free_cache_fixture();
	}
	return(0);
}


static SparkStatus make_resident(uint32_t page,uint32_t restore)
{
	SparkStatus status;
	uint32_t attempt;
	for (attempt=0u; attempt<16u; attempt++)
	{
		status = restore != 0u ? SparkKvPageStorePrefetch(&state.kv_page_store,&state.kv_arena,page) : SparkKvCacheArenaMarkBlockResident(&state.kv_arena,page);
		if ( status != SPARK_STATUS_BUSY )
			return(status);
		assert(SparkKvPageStoreWaitForTransfers(&state.kv_page_store) == SPARK_STATUS_OK);
		assert(SparkKvPageStoreProgress(&state.kv_page_store,&state.kv_arena,2u) == SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_BUSY);
}

static void check_physical_budget(void)
{
	char path[] = "/tmp/glm-physical-budget-XXXXXX";
	uint32_t pages[3],physical[2],i,first_slot;
	uint64_t key_bytes,value_bytes;
	uint8_t *expected,*actual;
	int32_t failure,descriptor = mkstemp(path);
	assert(descriptor >= 0 && close(descriptor) == 0 && unlink(path) == 0);
	for (failure=0; failure<4; failure++)
	{
		memset(&state,0,sizeof(state));
		state.ledger.module_tag = "physical-budget-test";
		state.tp_degree = 16u;
		state.layer_count = 4u;
		state.resident_sequence_capacity = 3u;
		state.max_sequence_positions = 128u;
		state.page_count = 7u;
		state.physical_page_count = 2u;
		state.kv_backing_directory = path;
		ALLOCATIONS_BEFORE_FAILURE = failure;
		assert(SparkGlm5NextAllocateCaches(&state) == SPARK_STATUS_CAPACITY_EXCEEDED);
		ALLOCATIONS_BEFORE_FAILURE = -1;
		free_cache_fixture();
		assert(state.ledger.device_allocation_count == 0u);
	}
	memset(&state,0,sizeof(state));
	state.ledger.module_tag = "physical-budget-test";
	state.tp_degree = 16u;
	state.first_layer_index = 3u;
	state.layer_count = 1u;
	state.resident_sequence_capacity = 3u;
	state.max_sequence_positions = 128u;
	state.page_count = 7u;
	state.physical_page_count = 2u;
	state.kv_backing_directory = path;
	REAL_BACKEND = 1u;
	assert(SparkGlm5NextAllocateCaches(&state) == SPARK_STATUS_OK);
	assert(state.kv_arena.logical_block_count == 7u && state.kv_arena.resident_block_capacity == 2u);
	for (i=0u; i<3u; i++)
		assert(SparkKvCacheArenaAcquireBlock(&state.kv_arena,&pages[i]) == SPARK_STATUS_OK);
	assert(make_resident(pages[2],0u) == SPARK_STATUS_OK);
	assert(make_resident(pages[0],0u) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaPinResidentTable(&state.kv_arena,(uint32_t[]){pages[2],pages[0]},2u,physical) == SPARK_STATUS_OK);
	assert(physical[0] != pages[2] && physical[0] != physical[1]);
	first_slot = physical[0];
	key_bytes = state.kv_arena.key_block_stride_bytes;
	value_bytes = state.kv_arena.value_block_stride_bytes;
	expected = malloc(key_bytes + value_bytes);
	actual = malloc(key_bytes + value_bytes);
	assert(expected != 0 && actual != 0);
	for (i=0u; i<key_bytes + value_bytes; i++)
		expected[i] = (uint8_t)(i * 37u + i / 113u);
	assert(SparkGlm5NextPageCopy(&state,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,state.kv_blocks[pages[2]].key_device_address,expected,key_bytes) == SPARK_STATUS_OK);
	assert(SparkGlm5NextPageCopy(&state,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,state.kv_blocks[pages[2]].value_device_address,expected + key_bytes,value_bytes) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockDirty(&state.kv_arena,pages[2]) == SPARK_STATUS_OK);
	assert(make_resident(pages[1],0u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(state.kv_blocks[pages[2]].resident_slot_index == first_slot && state.kv_blocks[pages[0]].resident_slot_index == physical[1]);
	assert(SparkKvCacheArenaUnpinResidentBlock(&state.kv_arena,pages[2]) == SPARK_STATUS_OK);
	assert(make_resident(pages[1],0u) == SPARK_STATUS_OK);
	assert(state.kv_page_store.write_count == 1u && state.kv_blocks[pages[2]].resident_slot_index == SPARK_KV_CACHE_NO_RESIDENT_SLOT);
	assert(state.kv_blocks[pages[0]].resident_slot_index == physical[1] && state.kv_blocks[pages[0]].residency_reference_count == 1u);
	assert(SparkKvCacheArenaPinResidentBlock(&state.kv_arena,pages[1]) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaUnpinResidentBlock(&state.kv_arena,pages[0]) == SPARK_STATUS_OK);
	assert(make_resident(pages[2],1u) == SPARK_STATUS_OK);
	assert(state.kv_blocks[pages[2]].resident_slot_index != first_slot && state.kv_page_store.read_count == 1u);
	assert(SparkGlm5NextPageCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,state.kv_blocks[pages[2]].key_device_address,actual,key_bytes) == SPARK_STATUS_OK);
	assert(SparkGlm5NextPageCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,state.kv_blocks[pages[2]].value_device_address,actual + key_bytes,value_bytes) == SPARK_STATUS_OK);
	assert(memcmp(expected,actual,key_bytes + value_bytes) == 0);
	{
		SparkGlm5NextStateCaptureLane lane;
		uint32_t logical[2],mapped[2];
		uint64_t hidden_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
		float scores[2] = {3.5f,1.25f};
		uint32_t capture_slots[2] = {0u,0u};
		uint8_t *hidden = malloc(2u * hidden_bytes),*payload = malloc(2u * (key_bytes + value_bytes) + hidden_bytes);
		SparkGlm5NextStateCapture capture = {.abi_version=SPARK_GLM5_NEXT_STATE_CAPTURE_ABI_VERSION,.descriptor_bytes=sizeof(capture),.lane_capacity=1u,.pages_per_lane_capacity=2u,.payload_capacity=2u * (key_bytes + value_bytes) + hidden_bytes,.lanes=&lane,.logical_pages=logical,.physical_pages=mapped,.payload=payload};
		SparkGlm5NextResidentDecodeStageBatchView batch = {.row_count=2u,.active_sequence_count=1u};
		SparkGlm5NextResidentDecodeStageFrameContext context = {.batch=&batch,.state_capture=&capture};
		SparkGlm5NextAsyncCompletion completion = {.state=&state,.lane_count=1u,.row_count=2u,.lane_next_positions={64u},.lane_sequence_ids={1u},.state_capture=&capture};
		assert(hidden != 0 && payload != 0);
		memset(hidden,0x35,hidden_bytes);
		memset(hidden + hidden_bytes,0x65,hidden_bytes);
		state.owns_final_head = 1u;
		assert(SparkGlm5NextValidateStateCapture(&state,&context) == SPARK_STATUS_OK);
		capture.payload_capacity--;
		assert(SparkGlm5NextValidateStateCapture(&state,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
		capture.payload_capacity++;
		capture.abi_version++;
		assert(SparkGlm5NextValidateStateCapture(&state,&context) == SPARK_STATUS_ABI_MISMATCH);
		capture.abi_version--;
		state.slots[0].hidden_bf16 = (uint16_t *)hidden;
		state.slots[0].output_score = scores;
		state.slots[0].host_resident_slots = capture_slots;
		state.kv_lane_transactions[0].phase = SPARK_KV_LANE_TRANSACTION_EXECUTING;
		state.kv_lane_transactions[0].page_count = 1u;
		state.kv_lane_logical_pages[0] = pages[2];
		state.kv_lane_physical_pages[0] = state.kv_blocks[pages[2]].resident_slot_index;
		assert(SparkKvCacheArenaPinResidentBlock(&state.kv_arena,pages[2]) == SPARK_STATUS_OK);
		assert(SparkGlm5NextCaptureState(&completion) == SPARK_STATUS_OK);
		assert(capture.payload_bytes == key_bytes + value_bytes + hidden_bytes && capture.backing_write_count == state.kv_page_store.write_count && capture.backing_read_count == 1u);
		assert(lane.next_position == 64u && lane.page_count == 1u && lane.output_score == scores[1] && logical[0] == pages[2] && mapped[0] == state.kv_blocks[pages[2]].resident_slot_index);
		assert(memcmp(payload,expected,key_bytes + value_bytes) == 0 && memcmp(payload + key_bytes + value_bytes,hidden + hidden_bytes,hidden_bytes) == 0);
		assert(SparkKvCacheArenaUnpinResidentBlock(&state.kv_arena,pages[2]) == SPARK_STATUS_OK);
		assert(SparkGlm5NextCaptureState(&completion) == SPARK_STATUS_INTERNAL_ERROR && capture.payload_bytes == 0u);
		free(hidden);
		free(payload);
	}
	assert(SparkKvCacheArenaUnpinResidentBlock(&state.kv_arena,pages[1]) == SPARK_STATUS_OK);
	free(expected);
	free(actual);
	free_cache_fixture();
	REAL_BACKEND = 0u;
	assert(unlink(path) == 0);
}

static int32_t check_recurrent_copy(void)
{
	uint8_t pools[4][72],saved[4][72],packed[60],before[60];
	uint32_t part,layer,slot,byte,offset,width;
	memset(&state,0,sizeof(state));
	state.resident_sequence_capacity = 3u;
	state.kda_layer_count = 3u;
	state.kda_state_layer_stride_bytes = 24u;
	state.kda_window_layer_stride_bytes = 12u;
	state.kda_state_pools = pools[0];
	state.kda_q_window_pool = pools[1];
	state.kda_k_window_pool = pools[2];
	state.kda_v_window_pool = pools[3];
	for (part=0u; part<4u; part++)
		for (byte=0u; byte<72u; byte++)
			pools[part][byte] = (uint8_t)(part * 73u + byte);
	memcpy(saved,pools,sizeof(saved));
	for (slot=0u; slot<3u; slot++)
	{
		assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,slot,packed,sizeof(packed)) == SPARK_STATUS_OK);
		offset = 0u;
		for (part=0u; part<4u; part++)
		{
			width = part == 0u ? 8u : 4u;
			for (layer=0u; layer<3u; layer++)
			{
				assert(memcmp(packed + offset,saved[part] + (layer * 3u + slot) * width,width) == 0);
				memset(pools[part] + (layer * 3u + slot) * width,0,width);
				offset += width;
			}
		}
		assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,slot,packed,sizeof(packed)) == SPARK_STATUS_OK);
		assert(memcmp(pools,saved,sizeof(pools)) == 0);
	}
	memset(packed,0xa5,sizeof(packed));
	memcpy(before,packed,sizeof(before));
	assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,3u,packed,sizeof(packed)) != SPARK_STATUS_OK);
	assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,0u,packed,sizeof(packed) - 1u) != SPARK_STATUS_OK);
	state.kda_v_window_pool = 0;
	assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,0u,packed,sizeof(packed)) != SPARK_STATUS_OK);
	assert(memcmp(packed,before,sizeof(packed)) == 0);
	assert(memcmp(pools,saved,sizeof(pools)) == 0);
	return(0);
}

static void open_recurrent_fixture(SparkKvPageStore *store,char *path,void *staging,uint64_t bytes)
{
	SparkKvPageStoreConfiguration config = {0};
	int32_t descriptor = mkstemp(path);
	assert(descriptor >= 0 && close(descriptor) == 0 && unlink(path) == 0);
	config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	config.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	config.logical_page_capacity = state.page_count;
	config.transfer_capacity = 1u;
	config.page_bytes = config.staging_bytes = bytes;
	config.maximum_backing_bytes = config.page_bytes * config.logical_page_capacity;
	config.staging_address = staging;
	config.backing_path = path;
	assert(SparkKvPageStoreInitialize(store,&config) == SPARK_STATUS_OK);
}

static void check_checkpoint_restore(SparkTestKvTransactions *fixture,const uint8_t *expected,uint32_t prefix_tokens)
{
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	SparkGlm5NextClaimedContinuityContext continuity = {0};
	SparkGlm5NextAsyncCompletion completion = {0};
	SparkModelDriverFrame frame;
	uint32_t resident = 1u;
	uint64_t sequence = 2u,position = prefix_tokens,next = 0u,simulated = 0u;
	uint8_t bound = 0u,restored[60];
	assert(SparkKvPageCacheReleaseLane(&fixture->pages.cache,0u,1u) == SPARK_STATUS_OK);
	SparkTestKvPageLane(&fixture->lanes[0],sequence,resident,position,prefix_tokens + 1u);
	SparkTestKvPagePrefix(&fixture->lanes[0],prefix_tokens,81u);
	fixture->request.request_id++;
	fixture->request.new_token_count = 1u;
	fixture->request.frame_flags = 0u;
	fixture->request.sequence_position = 0u;
	fixture->request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	assert(SparkKvLaneTransactionsAdmit(&state.kv_transactions,&fixture->request) == SPARK_STATUS_OK);
	fixture->request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	assert(SparkKvLaneTransactionsAdmit(&state.kv_transactions,&fixture->request) == SPARK_STATUS_OK);
	batch.active_sequence_count = batch.row_count = 1u;
	batch.row_resident_slots = &resident;
	batch.row_sequence_ids = &sequence;
	batch.row_positions = &position;
	state.max_sequence_positions = 64u;
	atomic_store(&state.lane_bound[resident],1u);
	atomic_store(&state.lane_sequence_ids[resident],99u);
	atomic_store(&state.lane_next_positions[resident],100u);
	continuity.state = &state;
	continuity.batch = &batch;
	continuity.bound = &bound;
	continuity.sequence_ids = &simulated;
	continuity.next_positions = &next;
	assert(SparkStageModuleIndexSetClaimAndPrepare(state.lane_states,4u,&resident,1u,SparkGlm5NextPrepareClaimedContinuity,&continuity) == SPARK_STATUS_OK);
	assert(bound == 1u && simulated == sequence && next == prefix_tokens + 1u);
	frame = SparkTestKvTransactionFrame(&fixture->request);
	frame.driver_dispatch_cookie0++;
	assert(SparkKvLaneTransactionsClaim(&state.kv_transactions,&frame) == SPARK_STATUS_VALIDATION_FAILED);
	frame.driver_dispatch_cookie0--;
	assert(SparkKvLaneTransactionsClaim(&state.kv_transactions,&frame) == SPARK_STATUS_OK);
	completion.state = &state;
	completion.lane_count = 1u;
	completion.lane_indices[0] = resident;
	if ( prefix_tokens % SPARK_TEST_BLOCK_TOKENS != 0u )
	{
		uint32_t entry = fixture->pages.cache.sequences[resident].terminal_entry_index;
		uint32_t source = fixture->pages.cache.entries[entry].logical_page_index;
		uint32_t copied = fixture->pages.cache.sequences[resident].mutable_logical_page_index;
		assert(source != copied && state.kv_blocks[source].residency_reference_count == 0u && fixture->pages.cache.entries[entry].reference_count != 0u);
		assert(SparkKvCacheArenaUnpinResidentBlock(&fixture->pages.kv.arena,copied) == SPARK_STATUS_OK);
		assert(SparkGlm5NextRestoreCacheLanes(&state,&completion) == SPARK_STATUS_VALIDATION_FAILED);
		assert(SparkKvCacheArenaPinResidentBlock(&fixture->pages.kv.arena,copied) == SPARK_STATUS_OK);
	}
	assert(SparkGlm5NextRestoreCacheLanes(&state,&completion) == SPARK_STATUS_OK);
	assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,resident,restored,sizeof(restored)) == SPARK_STATUS_OK);
	assert(memcmp(restored,expected,sizeof(restored)) == 0);
	assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,0u,restored,sizeof(restored)) == SPARK_STATUS_OK);
	assert(memcmp(restored,expected,sizeof(restored)) == 0);
	{
		uint32_t entry = fixture->pages.cache.sequences[resident].terminal_entry_index;
		uint32_t page = fixture->pages.cache.entries[entry].logical_page_index;
		assert(SparkKvPageStoreInvalidate(&state.recurrent_store,page,state.kv_blocks[page].generation) == SPARK_STATUS_OK);
		assert(SparkGlm5NextRestoreCacheLanes(&state,&completion) == SPARK_STATUS_NOT_FOUND);
	}
	assert(SparkKvLaneTransactionsFinish(&state.kv_transactions,&resident,1u,SPARK_STATUS_IO_ERROR,0u) == SPARK_STATUS_IO_ERROR);
	SparkStageModuleIndexSetRelease(state.lane_states,4u,&resident,1u);
}

static uint32_t PUBLISH_CALLBACKS;
static SparkStatus PUBLISH_STATUS;
static void publish_complete(void *context,const SparkModelDriverCompletion *completion)
{
	(void)context;
	PUBLISH_CALLBACKS++;
	PUBLISH_STATUS = completion->status;
	assert(completion->accepted_token_count == 0u && completion->tokens_per_sequence == 0u);
}

static void check_checkpoint_finish(uint32_t fail_copy,uint32_t prefix_tokens,uint32_t post_publish)
{
	SparkTestKvTransactions fixture;
	SparkGlm5NextAsyncCompletion completion = {0};
	SparkModelDriverFrame frame;
	uint8_t pools[4][96],staging[180],expected[60],restored[60];
	uint32_t page,part,byte,shadow[16];
	uint64_t generation;
	char path[] = "/tmp/glm-checkpoint-finish-XXXXXX",kv_path[] = "/tmp/glm-checkpoint-kv-XXXXXX";
	SparkTestKvTransactionsInitialize(&fixture,1u);
	fixture.lanes[0].context_token_count = fixture.request.new_token_count = prefix_tokens;
	if ( post_publish == 0u ) SparkTestKvPagePublish(&fixture.lanes[0],prefix_tokens,81u);
	memset(&state,0,sizeof(state));
	state.resident_sequence_capacity = 4u;
	state.page_count = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	state.pages_per_sequence = 4u;
	state.page_table_shadow = shadow;
	state.kv_transactions = fixture.transactions;
	state.kv_lane_transactions = fixture.owners;
	state.kv_blocks = fixture.pages.kv.blocks;
	state.kda_layer_count = 3u;
	state.kda_state_layer_stride_bytes = 32u;
	state.kda_window_layer_stride_bytes = 16u;
	state.kda_state_pools = pools[0];
	state.kda_q_window_pool = pools[1];
	state.kda_k_window_pool = pools[2];
	state.kda_v_window_pool = pools[3];
	state.recurrent_page_bytes = sizeof(expected);
	state.recurrent_staging = staging;
	assert(pthread_mutex_init(&state.kv_mutex,0) == 0);
	open_recurrent_fixture(&state.recurrent_store,path,staging + 60u,60u);
	open_recurrent_fixture(&state.kv_page_store,kv_path,staging + 120u,SPARK_TEST_BLOCK_BYTES);
	fixture.pages.cache.page_store = &state.kv_page_store;
	assert(SparkKvPageCacheAttachStateStore(&fixture.pages.cache,&state.recurrent_store) == SPARK_STATUS_OK);
	assert(SparkKvLaneTransactionsAdmit(&state.kv_transactions,&fixture.request) == SPARK_STATUS_OK);
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	assert(SparkKvLaneTransactionsAdmit(&state.kv_transactions,&fixture.request) == SPARK_STATUS_OK);
	frame = SparkTestKvTransactionFrame(&fixture.request);
	assert(SparkKvLaneTransactionsClaim(&state.kv_transactions,&frame) == SPARK_STATUS_OK);

	for (part=0u; part<4u; part++)
		for (byte=0u; byte<96u; byte++)
			pools[part][byte] = (uint8_t)(part * 73u + byte);
	assert(SparkGlm5NextRecurrentCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,0u,expected,sizeof(expected)) == SPARK_STATUS_OK);
	page = fixture.pages.cache.sequences[0].mutable_logical_page_index;
	generation = fixture.pages.kv.blocks[page].generation;
	completion.state = &state;
	completion.lane_count = 1u;
	completion.lane_bound[0] = 1u;
	completion.lane_sequence_ids[0] = 1u;
	completion.lane_next_positions[0] = prefix_tokens;
	if ( post_publish != 0u )
	{
		SparkModelDriverAdmissionDecision decision;
		SparkGeneratedDriverInstance instance = {&state};
		SparkModelDriverInterface driver = {.admit = SparkGeneratedDriverAdmit};
		PUBLISH_CALLBACKS = 0u;
		assert(SparkGlm5NextFinishCacheLanes(&completion) == SPARK_STATUS_OK);
		assert(fixture.pages.cache.published_page_count == 0u);
		state.pipeline_slot_count = 1u;
		fixture.request.request_id++;
		fixture.request.submission_id++;
		fixture.request.transaction_id++;
		fixture.request.new_token_count = 0u;
		fixture.request.sequence_position = prefix_tokens;
		fixture.request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH;
		fixture.lanes[0].sequence_position = prefix_tokens;
		SparkTestKvPagePublish(&fixture.lanes[0],prefix_tokens,81u);
		fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
		assert(SparkModelDriverEvaluateAdmission(&driver,&instance,&fixture.request,&decision) == SPARK_STATUS_OK && decision.accepted != 0u);
		fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
		assert(SparkModelDriverEvaluateAdmission(&driver,&instance,&fixture.request,&decision) == SPARK_STATUS_OK);
		fixture.request.admission_flags = 0u;
		atomic_store(&state.slot_states[0],SPARK_STAGE_MODULE_SLOT_CLAIMED);
		assert(SparkModelDriverEvaluateAdmission(&driver,&instance,&fixture.request,&decision) == SPARK_STATUS_BUSY);
		assert(fixture.owners[0].phase == SPARK_KV_LANE_TRANSACTION_COMMITTED && PUBLISH_CALLBACKS == 0u);
		atomic_store(&state.slot_states[0],SPARK_STAGE_MODULE_SLOT_FREE);
		assert(SparkModelDriverEvaluateAdmission(&driver,&instance,&fixture.request,&decision) == SPARK_STATUS_OK);
		assert(decision.available_dispatch_slot_count == 1u);
		frame = SparkTestKvTransactionFrame(&fixture.request);
		assert(SparkModelDriverApplyAdmissionDecision(&decision,&frame) == SPARK_STATUS_OK);
		frame.flags |= SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH;
		frame.sequence_position = prefix_tokens;
		frame.completion_function = publish_complete;
		PUBLISH_CALLBACKS = 0u;
		atomic_store(&state.lane_next_positions[0],prefix_tokens + 1u);
		assert(SparkGlm5NextPublishCache(&state,&frame) == SPARK_STATUS_VALIDATION_FAILED && PUBLISH_CALLBACKS == 0u);
		assert(fixture.owners[0].phase == SPARK_KV_LANE_TRANSACTION_COMMITTED);
		atomic_store(&state.lane_next_positions[0],prefix_tokens);
		if ( fail_copy != 0u ) state.kda_v_window_pool = 0;
		assert(SparkGlm5NextPublishCache(&state,&frame) == SPARK_STATUS_OK);
		assert(PUBLISH_CALLBACKS == 1u && PUBLISH_STATUS == (fail_copy != 0u ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK));
		assert(atomic_load(&state.slot_states[0]) == SPARK_STAGE_MODULE_SLOT_FREE && atomic_load(&state.lane_states[0]) == SPARK_STAGE_MODULE_SLOT_FREE);
	}
	else
	{
		if ( fail_copy != 0u ) state.kda_v_window_pool = 0;
		assert(SparkGlm5NextFinishCacheLanes(&completion) == (fail_copy != 0u ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK));
	}
	assert(fixture.owners[0].phase == SPARK_KV_LANE_TRANSACTION_EMPTY);
	assert(fixture.pages.kv.blocks[page].residency_reference_count == 0u);
	assert(fixture.pages.cache.published_page_count == (fail_copy != 0u ? 0u : 1u));
	if ( fail_copy == 0u )
	{
		assert(SparkKvPageStoreReadback(&state.recurrent_store,page,generation,(uintptr_t)restored,sizeof(restored)) == SPARK_STATUS_BUSY);
		assert(SparkKvPageStoreWaitForTransfers(&state.recurrent_store) == SPARK_STATUS_OK);
		assert(SparkKvPageStoreReadback(&state.recurrent_store,page,generation,(uintptr_t)restored,sizeof(restored)) == SPARK_STATUS_OK);
		assert(memcmp(restored,expected,sizeof(expected)) == 0);
		check_checkpoint_restore(&fixture,expected,prefix_tokens);
	}
	else
		assert(fixture.pages.cache.sequences[0].sequence_id == 0u);
	SparkKvPageStoreDestroy(&state.recurrent_store);
	SparkKvPageStoreDestroy(&state.kv_page_store);
	assert(pthread_mutex_destroy(&state.kv_mutex) == 0 && unlink(path) == 0 && unlink(kv_path) == 0);
}

static void check_execution_environment(void)
{
	uint64_t budget = 0u;
	uint32_t lane;
	const char *invalid_lanes[] = {"", "-1", "16", "4294967295", "1x", " 1", "+1"};
	assert(unsetenv("SPARK_WEIGHTD_LANE") == 0);
	assert(SparkGlm5NextRequestedMeshLane(&lane) == SPARK_STATUS_OK && lane == SPARK_WEIGHTD_LANE_NONE);
	for (uint32_t index=0u; index<sizeof(invalid_lanes)/sizeof(invalid_lanes[0]); index++)
	{
		assert(setenv("SPARK_WEIGHTD_LANE",invalid_lanes[index],1) == 0);
		assert(SparkGlm5NextRequestedMeshLane(&lane) == SPARK_STATUS_INVALID_ARGUMENT);
	}
	for (uint32_t index=0u; index<SPARK_WEIGHTD_MESH_MAX_LANES; index++)
	{
		char value[16];
		(void)snprintf(value,sizeof(value),"%u",index);
		assert(setenv("SPARK_WEIGHTD_LANE",value,1) == 0);
		assert(SparkGlm5NextRequestedMeshLane(&lane) == SPARK_STATUS_OK && lane == index);
	}
	assert(unsetenv("SPARK_WEIGHTD_LANE") == 0);
	const char *invalid[] = {"", "0", "-1", "18446744073709551615", "invalid"};
	assert(unsetenv("SPARK_GLM5_NEXT_PREFETCH") == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_GRAPH_PATH") == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM5_NEXT_GRAPH_PATH","",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM5_NEXT_GRAPH_PATH","2",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM5_NEXT_GRAPH_PATH","0",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.graph_path_enabled == 0u);
	assert(setenv("SPARK_GLM5_NEXT_GRAPH_PATH","1",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.graph_path_enabled == 1u);
	assert(setenv("SPARK_GLM5_NEXT_PREFETCH","1",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_UNSUPPORTED);
	assert(unsetenv("SPARK_GLM5_NEXT_PREFETCH") == 0);
	assert(unsetenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES") == 0);
	assert(SparkGlm5NextExpertPoolBudget(&budget) == SPARK_STATUS_INVALID_ARGUMENT);
	for (uint32_t index = 0u; index < sizeof(invalid) / sizeof(invalid[0]); index++)
	{
		assert(setenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES",invalid[index],1) == 0);
		assert(SparkGlm5NextExpertPoolBudget(&budget) == SPARK_STATUS_INVALID_ARGUMENT);
	}
	assert(setenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES","34359738368",1) == 0);
	assert(SparkGlm5NextExpertPoolBudget(&budget) == SPARK_STATUS_OK && budget == UINT64_C(34359738368));
	assert(unsetenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES") == 0);
}

static void check_module_reset(void)
{
	SparkTestKvTransactions fixture;
	SparkModelDriverAdmissionRequest request = {0};
	SparkModelDriverAdmissionDecision decision;
	uint8_t recurrent[96],windows[144];
	uint32_t shadow[16],index,slot = 0u;
	SparkTestKvTransactionsInitialize(&fixture,2u);
	assert(SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) == SPARK_STATUS_OK);
	memset(&state,0,sizeof(state));
	memset(recurrent,0xa5,sizeof(recurrent));
	memset(windows,0xa5,sizeof(windows));
	memset(shadow,0,sizeof(shadow));
	state.kv_transactions = fixture.transactions;
	state.kv_lane_transactions = fixture.owners;
	state.pipeline_slot_count = 2u;
	state.execution_row_capacity = 1u;
	state.resident_sequence_capacity = 4u;
	state.pages_per_sequence = 4u;
	state.page_table_shadow = shadow;
	state.kda_layer_count = 3u;
	state.kda_state_layer_stride_bytes = 32u;
	state.kda_window_layer_stride_bytes = 16u;
	state.kda_state_pools = recurrent;
	state.kda_window_pools = windows;
	state.control_generation = UINT64_MAX - 1u;
	assert(pthread_mutex_init(&state.kv_mutex,0) == 0);
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.control_generation = 3u;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) != 0u);
	request.new_token_count = 1u;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) == 0u);
	request.new_token_count = 0u;
	assert(SparkStageModuleIndexSetClaim(state.slot_states,2u,&slot,1u) == SPARK_STATUS_OK);
	assert(SparkGlm5NextResidentDecodeStageAdmit(&state,&request,&decision) == SPARK_STATUS_BUSY);
	assert(state.reset_generation == 0u && fixture.owners[0].phase == SPARK_KV_LANE_TRANSACTION_PREPARED);
	SparkStageModuleIndexSetRelease(state.slot_states,2u,&slot,1u);
	assert(SparkGlm5NextResidentDecodeStageAdmit(&state,&request,&decision) == SPARK_STATUS_OK && decision.accepted != 0u);
	assert(state.reset_generation == 3u && state.control_generation == 0u && fixture.pages.cache.live_sequence_count == 0u);
	for (index=0u; index<sizeof(recurrent); index++)
		assert(recurrent[index] == 0u);
	for (index=0u; index<sizeof(windows); index++)
		assert(windows[index] == 0u);
	for (index=0u; index<16u; index++)
		assert(shadow[index] == UINT32_MAX);
	assert(SparkGlm5NextResidentDecodeStageAdmit(&state,&request,&decision) == SPARK_STATUS_VALIDATION_FAILED);
	assert(SparkGlm5NextAdmissionPredicate(&state,&fixture.request,&decision) == SPARK_STATUS_OK);
	assert(state.control_generation == fixture.request.control_generation);
	fixture.request.control_generation = UINT64_MAX - 2u;
	assert(SparkGlm5NextAdmissionPredicate(&state,&fixture.request,&decision) == SPARK_STATUS_VALIDATION_FAILED);
	assert(state.control_generation != fixture.request.control_generation);
	request.control_generation = 4u;
	DRAIN_STATUS = cudaErrorInvalidValue;
	assert(SparkGlm5NextResidentDecodeStageAdmit(&state,&request,&decision) == SPARK_STATUS_PENDING);
	assert(state.reset_generation == 3u && atomic_load(&state.slot_states[0]) != SPARK_STAGE_MODULE_SLOT_FREE && atomic_load(&state.lane_states[0]) != SPARK_STAGE_MODULE_SLOT_FREE);
	DRAIN_STATUS = cudaSuccess;
	assert(pthread_mutex_destroy(&state.kv_mutex) == 0);
}

static void check_small_kv(void)
{
	static uint8_t index_pool[3u * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u];
	uint32_t pages;
	for (pages=1u; pages<=3u; pages++)
	{
		memset(&state,0,sizeof(state));
		state.kv_layer_count = 1u;
		state.index_layer_count = 1u;
		state.index_cache = index_pool;
		state.page_count = pages;
		state.physical_page_count = pages;
		state.pages_per_sequence = pages;
		state.resident_sequence_capacity = 1u;
		state.kv_backing_directory = "/unused-host-fixture";
		assert(SparkGlm5NextKvInitialize(&state) == SPARK_STATUS_PENDING);
		free_cache_fixture();
	}
	memset(&state,0,sizeof(state));
}

static int32_t check_layered_page_copy(void)
{
	uint8_t kv[3u * 5u * 8u],index[3u * 5u * 6u],packed[3u * 8u];
	uint8_t *pool;
	uint32_t region,page,layer,i,per_page;
	uintptr_t address;
	memset(&state,0,sizeof(state));
	state.kv_cache = kv;
	state.index_cache = index;
	state.page_count = 11u;
	state.physical_page_count = 5u;
	state.kv_layer_count = state.index_layer_count = 3u;
	state.kv_layer_stride_bytes = 5u * 8u;
	state.index_layer_stride_bytes = 5u * 6u;
	for (region=0u; region<2u; region++)
	{
		pool = region == 0u ? kv : index;
		per_page = region == 0u ? 8u : 6u;
		for (page=0u; page<5u; page++)
		{
			memset(kv,0x7e,sizeof(kv));
			memset(index,0x7e,sizeof(index));
			for (layer=0u; layer<3u; layer++)
				memset(pool + (layer * 5u + page) * per_page,(int)(layer + page + 1u),per_page);
			address = (uintptr_t)pool + page * 3u * per_page;
			if ( SparkGlm5NextPageCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,address,packed,3u * per_page) != SPARK_STATUS_OK )
				return(-1);
			for (layer=0u; layer<3u; layer++)
				for (i=0u; i<per_page; i++)
					if ( packed[layer * per_page + i] != layer + page + 1u )
						return(-2);
			memset(pool,0x7e,3u * 5u * per_page);
			if ( SparkGlm5NextPageCopy(&state,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,address,packed,3u * per_page) != SPARK_STATUS_OK )
				return(-3);
			for (i=0u; i<3u * 5u * per_page; i++)
				if ( pool[i] != (i / per_page % 5u == page ? i / (5u * per_page) + page + 1u : 0x7eu) )
					return(-4);
		}
	}
	if ( SparkGlm5NextPageCopy(&state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,(uintptr_t)kv + 1u,packed,sizeof(packed)) == SPARK_STATUS_OK )
		return(-5);
	memset(&state,0,sizeof(state));
	return(0);
}

static void check_pack_identity(void)
{
	SparkGlm5NextStagePackHeader header = {0};
	header.magic = SPARK_GLM5_NEXT_STAGEPACK_MAGIC;
	header.format_version = SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES;
	header.codec_abi_version = SPARK_WEIGHT_CODEC_ABI_VERSION;
	header.tensor_count = 1u;
	header.stage_count = state.stage_count;
	header.stage_index = state.stage_index;
	header.first_layer_index = state.first_layer_index;
	header.layer_count = state.layer_count;
	header.total_layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	header.hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	header.vocab_count = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	header.routed_expert_count = SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT;
	header.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	header.expert_weight_codec = state.expert_weight_codec;
	header.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	header.reserved0 = state.tp_degree;
	header.reserved1 = state.tp_rank;
	header.directory_offset = (((header.header_bytes + SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u) / SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES) * SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES);
	header.file_bytes = (header.directory_offset + header.directory_entry_bytes);
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_OK);
	header.stage_count++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header.stage_count--;
	header.stage_index++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header.stage_index--;
	header.first_layer_index++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header.first_layer_index--;
	header.layer_count++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
}

static void check_linear_walk(void)
{
	SparkGlm5NextTpChain chain = {0};
	SparkModelDriverFrame frame = {0};
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	uint16_t hidden[8] = {0};
	uint64_t maxloc[2] = {0};
	uint32_t tokens[2] = {0},host_tokens[2] = {0},layer = 0u,copies,syncs,limit;
	memset(&state,0,sizeof(state));
	state.pipeline_slot_count = 2u;
	state.owns_final_head = 1u;
	state.execution_row_capacity = 8u;
	state.tp_device_collective_initialized = 1u;
	state.tp_device_collective_hc_initialized = 1u;
	state.execution_stream = (void *)(uintptr_t)7u;
	state.slots[0].stream = state.execution_stream;
	state.slots[0].hidden_bf16 = hidden;
	state.slots[0].attention_out_bf16 = hidden;
	state.slots[0].head_maxloc_u64 = maxloc;
	state.slots[0].output_token = tokens;
	state.slots[0].host_output_token_ids = host_tokens;
	frame.request_id = 5u;
	batch.active_sequence_count = 2u;
	chain.state = &state;chain.slot = &state.slots[0];chain.frame = &frame;chain.batch = &batch;chain.wave_rows = 2u;
	chain.wave.slot = &state.slots[0];chain.wave.layer_count = 5u;chain.wave.row_count = 2u;chain.wave.expert_lease_all = 1u;
	WALK_LENGTH = 0u;WALK_GATHER_LAYER = 1u;WALK_FAIL_CODE = 0u;
	copies = COPY_COUNT;syncs = SYNC_COUNT;
	assert(SparkGlm5NextWalkChain(&chain,&layer) == 0u && layer == 5u);
	assert(strcmp(WALK_TRACE,"Bh" "ArPMrQ" "SgTrPMrQ" "ArPMrQ" "ArPRErQ" "ArPRErQ" "HxU") == 0);
	assert(COPY_COUNT == copies + 1u && SYNC_COUNT == syncs && chain.tp_hc_op_index == 1u && chain.tp_op_index == 12u);
	chain.tp_op_index = chain.tp_hc_op_index = 0u;
	WALK_LENGTH = 0u;WALK_FAIL_CODE = 'E';
	assert(SparkGlm5NextWalkChain(&chain,&layer) == 8u && layer == 3u && strcmp(WALK_TRACE,"Bh" "ArPMrQ" "SgTrPMrQ" "ArPMrQ" "ArPRE") == 0);
	WALK_FAIL_CODE = 0u;
	for (limit=3u; limit<=4u; limit++)
	{
		chain.tp_op_index = chain.tp_hc_op_index = 0u;
		WALK_LENGTH = 0u;
		state.graph_record_limit = limit;
		state.graph_record_ops = 0u;
		state.graph_record_stop = 0u;
		assert(SparkGlm5NextWalkChain(&chain,&layer) == 0u && layer == 2u && state.graph_record_stop == 1u && strcmp(WALK_TRACE,"Bh" "ArPMrQ" "SgTrPMrQ" "HxU") == 0);
	}
	state.graph_record_limit = 0u;
	state.graph_record_ops = 0u;
	state.graph_record_stop = 0u;
}

static void pin_fixture_experts(void)
{
	uint32_t index,expected = SparkGlm5NextPinnedExpected(&state);
	state.expert_pin_key_count = expected;
	state.expert_pin_lease_count = SparkCeilDivU32(expected,SPARK_WEIGHTD_LEASE_GROUPS_MAX);
	for (index=0u; index<state.expert_pin_lease_count; index++)
	{
		state.expert_pin_leases[index] = index + 1u;
		state.expert_pin_phases[index] = 1u;
	}
	state.decode_lease_base_saved = (const uint8_t *)(uintptr_t)64u;
}

static void check_linear_eligibility(void)
{
	SparkGlm5NextTpChain chain = {0};
	SparkWeightdLazyPack pack = {0};
	memset(&state,0,sizeof(state));
	chain.state = &state;
	state.lazy_pack = &pack;
	state.tp_degree = 16u;
	state.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	state.tp_device_collective_initialized = 1u;
	state.tp_device_collective_hc_initialized = 1u;
	STREAM_ORDERED = 1u;
	pin_fixture_experts();
	assert(state.expert_pin_key_count == (SPARK_GLM5_NEXT_MODEL_LAYER_COUNT - SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER) * SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT && SparkGlm5NextLinearEligible(&chain) == 1u);
	state.expert_pin_phases[0] = 2u;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	state.expert_pin_phases[0] = 1u;
	state.expert_pin_key_count--;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	state.expert_pin_key_count++;
	STREAM_ORDERED = 0u;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	STREAM_ORDERED = 1u;
	state.mtp_enabled = 1u;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	state.mtp_enabled = 0u;
	chain.spec_verify = 1u;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	chain.spec_verify = 0u;
	state.graph_record_limit = 4u;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	state.graph_record_limit = 0u;
	state.tp_device_collective_hc_initialized = 0u;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	state.tp_device_collective_hc_initialized = 1u;
	state.lazy_pack = 0;
	assert(SparkGlm5NextLinearEligible(&chain) == 0u);
	state.lazy_pack = &pack;
	assert(SparkGlm5NextLinearEligible(&chain) == 1u);
	STREAM_ORDERED = 0u;
}

static uint16_t LINEAR_HIDDEN[8];
static uint64_t LINEAR_MAXLOC[2];
static uint32_t LINEAR_TOKENS[2],LINEAR_HOST_TOKENS[2],LINEAR_SLOTS[2] = {0u,1u},LINEAR_POSITIONS[2] = {3u,5u},LINEAR_BEGIN[3],LINEAR_INDICES[2],LINEAR_RUNS[2],LINEAR_ERRORS[SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT],LINEAR_HOST_ERRORS[SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT];
static SparkWeightdLazyPack LINEAR_PACK;
static SparkModelDriverFrame LINEAR_FRAME;
static SparkGlm5NextResidentDecodeStageBatchView LINEAR_BATCH;
static SparkGlm5NextResidentDecodeStageFrameContext LINEAR_CONTEXT;

static void linear_slot_fixture(SparkGlm5NextExecutionSlot *slot)
{
	slot->stream = state.execution_stream;
	slot->hidden_bf16 = slot->attention_out_bf16 = LINEAR_HIDDEN;
	slot->head_maxloc_u64 = LINEAR_MAXLOC;
	slot->output_token = LINEAR_TOKENS;
	slot->host_output_token_ids = LINEAR_HOST_TOKENS;
	slot->host_positions = LINEAR_POSITIONS;
	slot->host_resident_slots = LINEAR_SLOTS;
	slot->host_run_begin = LINEAR_BEGIN;
	slot->host_run_row_indices = LINEAR_INDICES;
	slot->host_run_state_index = LINEAR_RUNS;
	slot->kv_access_error = LINEAR_ERRORS;
	slot->host_kv_access_error = LINEAR_HOST_ERRORS;
}

static SparkGlm5NextTpChain *linear_chain_fixture(void)
{
	SparkGlm5NextTpChain *chain = calloc(1u,sizeof(*chain));
	assert(chain != 0);
	memset(&state,0,sizeof(state));
	state.pipeline_slot_count = 1u;
	state.execution_row_capacity = 8u;
	state.resident_sequence_capacity = 4u;
	state.tp_degree = 16u;
	state.layer_count = 5u;
	state.owns_final_head = 1u;
	state.lazy_pack = &LINEAR_PACK;
	state.completion_worker = (SparkWeightdWorker *)(uintptr_t)1u;
	state.tp_device_collective_initialized = state.tp_device_collective_hc_initialized = 1u;
	state.execution_stream = (void *)(uintptr_t)7u;
	state.completions[0].state = &state;
	linear_slot_fixture(&state.slots[0]);
	pin_fixture_experts();
	assert(pthread_mutex_init(&state.completion_queue_lock,0) == 0);
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	LINEAR_FRAME.request_id = 5u;
	LINEAR_BATCH.active_sequence_count = LINEAR_BATCH.row_count = 2u;
	LINEAR_BATCH.row_resident_slots = LINEAR_SLOTS;
	LINEAR_CONTEXT.batch = &LINEAR_BATCH;
	chain->state = &state;chain->slot = &state.slots[0];chain->frame = &LINEAR_FRAME;chain->batch = &LINEAR_BATCH;chain->context = &LINEAR_CONTEXT;
	chain->wave_rows = 2u;chain->active = 1u;chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
	STREAM_ORDERED = 1u;HOST_MODE = 0u;HOST_COUNT = 0u;CANCEL_COUNT = 0u;WORK_STATUS = SPARK_STATUS_OK;DRAIN_STATUS = cudaSuccess;
	WALK_LENGTH = 0u;WALK_TRACE[0] = 0;WALK_GATHER_LAYER = 1u;WALK_FAIL_CODE = 0u;
	COMPLETION_WORK = 0;COMPLETION_CONTEXT = 0;
	return(chain);
}

static void linear_chain_teardown(void)
{
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
	assert(pthread_mutex_destroy(&state.completion_queue_lock) == 0);
	STREAM_ORDERED = 0u;WALK_FAIL_CODE = 0u;HEALTH_DEAD_MASK = 0u;
	memset(&state,0,sizeof(state));
}

static void check_linear_chain(void)
{
	SparkGlm5NextTpChain *chain = linear_chain_fixture();
	SparkGlm5NextAsyncCompletion *async = &state.completions[0];
	uint32_t syncs = SYNC_COUNT,copies = COPY_COUNT;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bh" "ArPMrQ" "SgTrPMrQ" "ArPMrQ" "ArPRErQ" "ArPRErQ" "HxU") == 0 && SYNC_COUNT == syncs && COPY_COUNT == copies + 2u);
	assert(async->linear == 1u && async->graph == 0u && async->launch_ns != 0u && async->finish_ns >= async->launch_ns && state.experts_warm == 1u);
	assert(COMPLETION_WORK == SparkGlm5NextCompleteOnWorker && COMPLETION_CONTEXT == async && async->completion.status == SPARK_STATUS_OK && CANCEL_COUNT == 0u);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	WALK_FAIL_CODE = 'E';
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bh" "ArPMrQ" "SgTrPMrQ" "ArPMrQ" "ArPRE") == 0 && async->linear == 1u);
	assert(COMPLETION_WORK == SparkGlm5NextCompleteOnWorker && async->completion.status == SPARK_STATUS_INTERNAL_ERROR && CANCEL_COUNT == 2u);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	state.lane_client = (SparkWeightdClient *)(uintptr_t)1u;
	HEALTH_DEAD_MASK = 1u;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(WALK_LENGTH == 0u && async->linear == 0u && async->completion.status == SPARK_STATUS_IO_ERROR && COMPLETION_WORK == SparkGlm5NextCompleteOnWorker);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	STREAM_ORDERED = 0u;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bc") == 0 && async->linear == 0u && chain->active == 1u && chain->stage == SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION && COMPLETION_WORK == 0);
	free(chain);
	linear_chain_teardown();
}

static void check_verify_deferred(void)
{
	SparkGlm5NextAsyncCompletion async;
	memset(&state,0,sizeof(state));
	memset(&async,0,sizeof(async));
	state.execution_stream = (void *)(uintptr_t)7u;
	state.tp_device_collective_initialized = 1u;
	state.tp_device_collective_hc_initialized = 1u;
	DRAIN_STATUS = cudaSuccess;
	CANCEL_COUNT = VERIFY_COUNT = 0u;
	VERIFY_STATUS = SPARK_STATUS_OK;
	SparkGlm5NextVerifyDeferred(&state,&async);
	assert(async.completion.status == SPARK_STATUS_OK && VERIFY_COUNT == 2u && CANCEL_COUNT == 0u && async.finish_ns == 0u);
	async.linear = 1u;
	SparkGlm5NextVerifyDeferred(&state,&async);
	assert(async.completion.status == SPARK_STATUS_OK && VERIFY_COUNT == 4u && CANCEL_COUNT == 0u && async.finish_ns != 0u);
	VERIFY_STATUS = SPARK_STATUS_IO_ERROR;
	SparkGlm5NextVerifyDeferred(&state,&async);
	assert(async.completion.status == SPARK_STATUS_IO_ERROR && VERIFY_COUNT == 6u && CANCEL_COUNT == 2u);
	async.completion.status = SPARK_STATUS_VALIDATION_FAILED;
	SparkGlm5NextVerifyDeferred(&state,&async);
	assert(async.completion.status == SPARK_STATUS_VALIDATION_FAILED && VERIFY_COUNT == 8u && CANCEL_COUNT == 2u);
	VERIFY_STATUS = SPARK_STATUS_OK;
	CANCEL_COUNT = 0u;
	async.linear = 0u;
	assert(strcmp(SparkGlm5NextChainPath(&async),"eager") == 0);
	async.linear = 1u;
	assert(strcmp(SparkGlm5NextChainPath(&async),"linear") == 0);
	async.graph = 1u;
	assert(strcmp(SparkGlm5NextChainPath(&async),"graph") == 0);
}

static void check_attempt_accounting(void)
{
	memset(&state,0,sizeof(state));
	SparkGlm5NextNoteAttempt(&state,5u);
	assert(state.wave_attempt_request == 5u && state.wave_attempt_ns != 0u && state.wave_attempt_retries == 0u);
	assert(SparkGlm5NextNoteBusy(&state,SPARK_STATUS_BUSY,SPARK_GLM5_NEXT_BUSY_CHAIN) == SPARK_STATUS_BUSY);
	SparkGlm5NextNoteAttempt(&state,5u);
	assert(SparkGlm5NextNoteBusy(&state,SPARK_STATUS_BUSY,SPARK_GLM5_NEXT_BUSY_REASONS + 3u) == SPARK_STATUS_BUSY);
	SparkGlm5NextNoteAttempt(&state,5u);
	assert(SparkGlm5NextNoteBusy(&state,SPARK_STATUS_IO_ERROR,SPARK_GLM5_NEXT_BUSY_SLOT) == SPARK_STATUS_IO_ERROR);
	assert(state.wave_attempt_retries == 2u && state.wave_attempt_busy[SPARK_GLM5_NEXT_BUSY_CHAIN] == 1u && state.wave_attempt_busy[SPARK_GLM5_NEXT_BUSY_OTHER] == 1u && state.wave_attempt_busy[SPARK_GLM5_NEXT_BUSY_SLOT] == 0u);
	SparkGlm5NextStampClaim(&state,1u);
	assert(state.completions[1].attempt_ns == state.wave_attempt_ns && state.completions[1].chain_start_ns >= state.wave_attempt_ns);
	assert(state.completions[1].retries == 2u && state.completions[1].busy[SPARK_GLM5_NEXT_BUSY_CHAIN] == 1u && state.completions[1].busy[SPARK_GLM5_NEXT_BUSY_OTHER] == 1u);
	SparkGlm5NextNoteAttempt(&state,6u);
	assert(state.wave_attempt_request == 6u && state.wave_attempt_retries == 0u && state.wave_attempt_busy[SPARK_GLM5_NEXT_BUSY_CHAIN] == 0u && state.wave_attempt_busy[SPARK_GLM5_NEXT_BUSY_OTHER] == 0u);
	assert(SparkGlm5NextGraphPathState(&state) == SPARK_GLM5_NEXT_GRAPH_PATH_OFF);
	state.graph_path_requested = 1u;
	assert(SparkGlm5NextGraphPathState(&state) == SPARK_GLM5_NEXT_GRAPH_PATH_DEGRADED);
	state.graph_path_enabled = 1u;
	assert(SparkGlm5NextGraphPathState(&state) == SPARK_GLM5_NEXT_GRAPH_PATH_ON);
}

static void check_wave_timing(void)
{
	static SparkGlm5NextWaveTiming timing;
	SparkGlm5NextAsyncCompletion wave;
	SparkTpDeviceCollectiveHardwareTiming collective = {1000000u,20000000u,2000000u,3000000u};
	const uint64_t attempt[3] = {UINT64_C(1000000000),UINT64_C(6000000000),UINT64_C(12000000000)},wait[3] = {100000u,3000000u,100000u},setup[3] = {251000000u,600000u,600000u},run[3] = {60000000u,90000000u,60000000u};
	char path[] = "/tmp/g5n_wave_timing_XXXXXX",text[2048] = {0};
	const char *expected = "G5N-WAVE-TIMING rank=3 waves=3 rows=24 prefill=1 graph=2 eager=1 linear=1 graph_path=1 retries=2 busy=1/1/0/0/0 captures=1 capture_ms=250 idle_us=8388608/8388608 wait_us=128/4096 key_us=512/512 setup_us=1024/262144 run_us=65536/131072 post_us=512/512 idle_ms=10593 wait_ms=3 key_ms=0 setup_ms=252 run_ms=210 post_ms=1 graph_run_ms=150 eager_run_ms=60 linear_run_ms=60 decode_wait_ms=3 source_wait_ms=3 peer_wait_ms=60 copy_ms=6 combine_ms=9 worst_ms=311 worst_request=7 worst_epochs=11/12 worst_us=0/100/300/251000/60000/500\n";
	int descriptor = mkstemp(path),saved = dup(2);
	uint64_t index;
	assert(descriptor >= 0 && saved >= 0 && dup2(descriptor,2) == 2);
	for (index=0u; index<3u; index++)
	{
		memset(&wave,0,sizeof(wave));
		wave.row_count = 8u;
		wave.epoch[0] = 11u;
		wave.epoch[1] = 12u;
		wave.graph_path = SPARK_GLM5_NEXT_GRAPH_PATH_ON;
		wave.completion.request_id = 7u + index;
		wave.graph = index < 2u ? 1u : 0u;
		wave.prefill = index == 2u ? 1u : 0u;
		wave.linear = index == 2u ? 1u : 0u;
		wave.captures = index == 0u ? 1u : 0u;
		wave.capture_ns = index == 0u ? 250000000u : 0u;
		wave.retries = index == 1u ? 2u : 0u;
		wave.busy[SPARK_GLM5_NEXT_BUSY_CHAIN] = index == 1u ? 1u : 0u;
		wave.busy[SPARK_GLM5_NEXT_BUSY_STREAM] = index == 1u ? 1u : 0u;
		wave.attempt_ns = attempt[index];
		wave.chain_start_ns = wave.attempt_ns + wait[index];
		wave.keyed_ns = wave.chain_start_ns + 300000u;
		wave.launch_ns = wave.keyed_ns + setup[index];
		wave.finish_ns = wave.launch_ns + run[index];
		SparkGlm5NextWaveTimingRecord(&timing,&wave,&collective,3u,wave.finish_ns + 500000u);
	}
	fflush(stderr);
	assert(dup2(saved,2) == 2 && close(saved) == 0);
	assert(pread(descriptor,text,sizeof(text) - 1u,0) > 0 && close(descriptor) == 0 && unlink(path) == 0);
	if ( strcmp(text,expected) != 0 )
		fprintf(stderr,"wave timing line:\n%s",text);
	assert(strcmp(text,expected) == 0);
	assert(timing.waves == 0u && timing.worst_ns == 0u && timing.delivered_ns == UINT64_C(12061500000) && timing.window_ns == timing.delivered_ns);
}

int32_t main(void)
{
	SparkGlm5NextResidentDecodeStageNodeContext context = {0};
	SparkFirmwareModuleConfiguration configuration = {0};
	SparkFirmwareModuleHostServices services = {0};
	const char *path = 0;
	uint32_t first[4] = {0u,12u,23u,34u},counts[4] = {12u,11u,11u,11u},stage;
	check_linear_walk();
	check_linear_eligibility();
	check_linear_chain();
	check_verify_deferred();
	check_attempt_accounting();
	check_wave_timing();
	check_graph_epoch_ownership();
	check_lazy_open_retained_owner();
	check_graph_expert_ownership(0u,45u,3u,0u,0u);
	check_graph_expert_ownership(12u,12u,12u,0u,0u);
	check_graph_expert_ownership(12u,12u,12u,2u,0u);
	check_graph_expert_ownership(12u,12u,12u,0u,2u);
	check_collective_destroy_order(0u);
	check_collective_destroy_order(1u);
	check_collective_destroy_order(2u);
	check_weightd_health();
	check_mtp_callback_handoff(SPARK_STATUS_OK,0u);
	check_mtp_callback_handoff(SPARK_STATUS_OK,1u);
	check_mtp_callback_handoff(SPARK_STATUS_BUSY,0u);
	check_mtp_callback_handoff(SPARK_STATUS_BUSY,1u);
	check_callback_retirement();
	check_stream_receipt();
	check_chain_ownership();
	int32_t status = check_cache_transactions(SPARK_STATUS_IO_ERROR,SPARK_STATUS_OK,SPARK_STATUS_OK,cudaSuccess);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_OK,SPARK_STATUS_OK,cudaSuccess) == 0);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_IO_ERROR,SPARK_STATUS_OK,cudaSuccess) == 0);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_OK,SPARK_STATUS_IO_ERROR,cudaSuccess) == 0);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_OK,SPARK_STATUS_OK,cudaErrorInvalidValue) == 0);
	if ( status != 0 )
		return(-status);
	status = check_cache_release();
	if ( status != 0 )
		return(-status);
	if ( check_batch_waves() != 0 )
		return(1);
	if ( check_layered_page_copy() != 0 )
		return(2);
	assert(check_recurrent_copy() == 0);
	check_cache_worker_cleanup();
	check_checkpoint_finish(0u,4u,0u);
	check_checkpoint_finish(0u,3u,0u);
	check_checkpoint_finish(1u,4u,0u);
	check_checkpoint_finish(0u,4u,1u);
	check_checkpoint_finish(0u,3u,1u);
	check_checkpoint_finish(1u,3u,1u);
	check_module_reset();
	check_execution_environment();
	check_small_kv();
	check_physical_budget();
	if ( check_rank_state() != 0 )
		return(3);
	context.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	context.descriptor_bytes = sizeof(context);
	context.stage_count = 4u;
	context.expert_weight_codec = GLM5_NEXT_EXPERT_WEIGHT_CODEC;
	context.resident_sequence_capacity = 3u;
	context.pipeline_slot_count = 1u;
	context.max_sequence_positions = 64u;
	context.execution_row_capacity = 3u;
	context.tp_degree = 4u;
	context.stage_pack_path = "fixture.g5nsp";
	context.model_revision = "fixture";
	configuration.model_revision = context.model_revision;
	services.node_context = &context;
	services.execution_stream = (void *)(uintptr_t)1u;
	services.kv_logical_page_capacity = 7u;
	services.kv_physical_page_capacity = 2u;
	for (stage=0u; stage<4u; stage++)
	{
		context.stage_index = stage;
		context.first_layer_index = first[stage];
		context.layer_count = counts[stage];
		assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_OK);
		assert(state.stage_count == 4u && state.stage_index == stage);
		assert(state.page_count == services.kv_logical_page_capacity);
		assert(state.physical_page_count == services.kv_physical_page_capacity);
		assert(state.first_layer_index == first[stage] && state.layer_count == counts[stage]);
		assert(state.owns_embedding == (stage == 0u) && state.owns_final_head == (stage == 3u));
		assert(path == context.stage_pack_path);
		check_pack_identity();
	}
	context.abi_version--;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_ABI_MISMATCH);
	context.abi_version++;
	context.stage_index = 0u;
	context.first_layer_index = 0u;
	context.layer_count = 12u;
	context.flags = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	context.flags = 0u;
	context.stage_count = 1u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	context.layer_count = 45u;
	context.tp_degree = 16u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_OK);
	assert(state.owns_embedding == 1u && state.owns_final_head == 1u);
	check_pack_identity();
	services.kv_physical_page_capacity = 0u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	services.kv_physical_page_capacity = services.kv_logical_page_capacity + 1u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	services.kv_physical_page_capacity = 1u;
	services.kv_logical_page_capacity = 0u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	return(0);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-source', type=Path, default=ROOT / 'modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c')
    parser.add_argument('--common-source', type=Path, default=ROOT / 'runtime/stage_module_common.c')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    module = args.module_source.read_text()
    legacy = 'state.epoch_device = EPOCH_WORDS; state.epoch_validated = 41u;' if 'epoch_validated;' in module else ''
    harness = HARNESS.replace('LEGACY_EPOCH_SETUP', legacy).replace('"modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"', '"' + str(args.module_source.resolve()) + '"')
    with tempfile.TemporaryDirectory() as directory:
        emitter = generate_admission(Path(directory), 1, "SparkGlm5NextResidentDecodeStageAdmit")
        Path(directory, "generated_admission.inc").write_text(subprocess.check_output([str(emitter), "module"], text=True))
        source, binary = Path(directory) / "probe.c", Path(directory) / "probe"
        source.write_text(harness)
        includes = [".", "include", "tests/cuda_stub", "model-families/common/include",
                    "model-families/glm5_next/include", "modules/glm5_next_resident_decode_stage/include",
                    "modules/glm5_next_resident_decode_stage/source"]
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O2", "-ffunction-sections", "-fdata-sections",
                        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                        *["-I" + p for p in includes], "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
                        '-DGLM5_NEXT_EXPERT_CODEC_NAME="fp8"', '-DGLM5_NEXT_CONTRACT_SHA256="fixture"',
                        str(source), str(args.common_source.resolve()), "cache/kv_cache.c", "cache/kv_page_cache.c",
                        "-o", str(binary), *(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else [])], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS actual module context/cache ownership, global epoch independence, retained attach ownership and terminal CUDA receipts")


if __name__ == "__main__":
    main()
