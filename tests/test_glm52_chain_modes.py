#!/usr/bin/env python3
"""Drive the real glm52 module chain runner (linear and graph modes) on the host."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c"

#define LOG_CAPACITY 4096u
static char LOG[LOG_CAPACITY][40];
static uint32_t LOG_COUNT;
static uint32_t T1_ENABLED,STREAM_ORDERED = 1u,CAPTURING,CAPTURES,LAUNCHES,WORKER_INLINE = 1u;
static SparkStatus VERIFY_STATUS,ENQUEUE_STATUS,WORKER_STATUS,CHAIN_KEY_STATUS = SPARK_STATUS_IO_ERROR;
static cudaError_t STREAM_QUERY = cudaSuccess;
static uint32_t ENQUEUE_FAIL_AT = UINT32_MAX,ENQUEUE_COUNT;
static uint64_t GRAPH_ERROR;
static SparkWeightdWorkFunction PENDING_WORK;
static void *PENDING_CONTEXT;
static SparkStatus COMPLETED_STATUS;
static uint32_t COMPLETED_COUNT;
static SparkGlm52ModuleState *STATE;

static void Log(const char *entry,uint32_t value)
{
	assert(LOG_COUNT < LOG_CAPACITY);
	(void)snprintf(LOG[LOG_COUNT++],sizeof(LOG[0]),"%s%s%u",CAPTURING != 0u ? "cap:" : "",entry,value);
}

static uint32_t Count(const char *entry)
{
	uint32_t index,count = 0u;
	for (index=0u; index<LOG_COUNT; index++)
		count += strcmp(LOG[index],entry) == 0 ? 1u : 0u;
	return(count);
}

static int32_t Find(const char *entry,uint32_t from)
{
	uint32_t index;
	for (index=from; index<LOG_COUNT; index++)
		if ( strcmp(LOG[index],entry) == 0 )
			return((int32_t)index);
	return(-1);
}

int32_t SparkGlm52T1Enabled(void) { return((int32_t)T1_ENABLED); }
uint32_t SparkGlm52ExactWaveRows(void) { return(8u); }
int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave) { Log("begin",wave->maximum_context); return(0); }
int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->projection_split == 0u); Log("attn",layer); return(0); }
int32_t SparkGlm52LaunchCudaLayerAttentionProject(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->projection_split != 0u); Log("project",layer); return(0); }
int32_t SparkGlm52LaunchCudaLayerAttentionCore(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->projection_split != 0u); Log("core",layer); return(0); }
int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->route_host_copy == 0u); Log("mlp",layer); return(0); }
int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave) { Log("head",wave->row_head_certified); return(0); }
cudaError_t SparkGlm52LaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *tokens,uint32_t rows) { (void)stream; (void)maxloc; (void)tokens; Log("unpack",rows); return(cudaSuccess); }
int32_t SparkGlm52LaunchCudaLayerMlpRoute(const SparkGlm52CudaWave *wave,uint32_t layer) { (void)wave; (void)layer; assert(0); return(1); }
int32_t SparkGlm52LaunchCudaLayerMlpExperts(const SparkGlm52CudaWave *wave,uint32_t layer) { (void)wave; (void)layer; assert(0); return(1); }

cudaError_t cudaMemcpyAsync(void *destination,const void *source,size_t bytes,enum cudaMemcpyKind kind,cudaStream_t stream)
{
	(void)destination; (void)source; (void)stream;
	Log(kind == cudaMemcpyDeviceToHost ? "d2h" : "copy",(uint32_t)bytes);
	return(cudaSuccess);
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; return(cudaSuccess); }
cudaError_t cudaStreamQuery(cudaStream_t stream) { (void)stream; return(STREAM_QUERY); }
cudaError_t cudaGetLastError(void) { return(cudaSuccess); }
const char *cudaGetErrorString(cudaError_t error) { (void)error; return("stub"); }
cudaError_t cudaLaunchHostFunc(cudaStream_t stream,cudaHostFn_t function,void *context) { (void)stream; function(context); return(cudaSuccess); }
cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode) { (void)stream; (void)mode; Log("capture-begin",0u); CAPTURING = 1u; return(cudaSuccess); }
cudaError_t cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t *graph) { (void)stream; CAPTURING = 0u; Log("capture-end",0u); *graph = (cudaGraph_t)(uintptr_t)0x10; return(cudaSuccess); }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec,cudaGraph_t graph,...) { (void)graph; CAPTURES++; *exec = (cudaGraphExec_t)(uintptr_t)(0x100u + CAPTURES); return(cudaSuccess); }
cudaError_t cudaGraphUpload(cudaGraphExec_t exec,cudaStream_t stream) { (void)exec; (void)stream; return(cudaSuccess); }
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec,cudaStream_t stream) { (void)stream; LAUNCHES++; Log("graph-launch",(uint32_t)(uintptr_t)exec); return(cudaSuccess); }
cudaError_t cudaGraphDestroy(cudaGraph_t graph) { (void)graph; return(cudaSuccess); }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) { (void)exec; return(cudaSuccess); }

SparkStatus SparkTpDeviceCollectiveEnqueue(SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission,uint32_t operation)
{
	SparkGlm52ExecutionSlot *slot = &STATE->slots[0];
	(void)collective;
	assert(submission->completion_function == 0 && submission->completion_context == 0);
	assert((submission->flags & SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION) != 0u);
	if ( ENQUEUE_COUNT++ == ENQUEUE_FAIL_AT )
		return(ENQUEUE_STATUS);
	if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
		Log("reduce-max",submission->local_device == slot->head_maxloc_u64 ? 1u : 0u);
	else if ( submission->local_device == slot->hidden_bf16 )
		Log("reduce-hidden",submission->active_sequence_count);
	else if ( submission->local_device == slot->attention_out_bf16 )
		Log("reduce-attn",submission->active_sequence_count);
	else if ( submission->local_device == slot->projection_gather_bf16 )
		Log("reduce-gather",submission->active_sequence_count);
	else
		Log("reduce-unknown",0u);
	return(SPARK_STATUS_OK);
}
SparkStatus SparkTpDeviceCollectiveArmCapture(SparkTpDeviceCollective *collective) { (void)collective; Log("arm",0u); return(SPARK_STATUS_OK); }
SparkStatus SparkTpDeviceCollectiveDisarmCapture(SparkTpDeviceCollective *collective) { (void)collective; Log("disarm",0u); return(SPARK_STATUS_OK); }
SparkStatus SparkTpDeviceCollectiveGraphPreLaunch(SparkTpDeviceCollective *collective,void *stream) { (void)collective; (void)stream; Log("prelaunch",0u); return(SPARK_STATUS_OK); }
SparkStatus SparkTpDeviceCollectiveGraphCancelSeed(SparkTpDeviceCollective *collective,void *stream) { (void)collective; (void)stream; Log("seed",0u); return(SPARK_STATUS_OK); }
uint64_t SparkTpDeviceCollectiveGraphError(SparkTpDeviceCollective *collective) { (void)collective; Log("graph-error",0u); return(GRAPH_ERROR); }
SparkStatus SparkTpDeviceCollectiveVerifyDeferred(SparkTpDeviceCollective *collective,void *stream) { (void)collective; (void)stream; Log("verify",0u); return(VERIFY_STATUS); }
void SparkTpDeviceCollectiveBroadcastCancel(SparkTpDeviceCollective *collective) { (void)collective; Log("cancel",0u); }
uint32_t SparkTpDeviceCollectiveStreamOrdered(const SparkTpDeviceCollective *collective) { (void)collective; return(STREAM_ORDERED); }
SparkStatus SparkWeightdWorkerSubmit(SparkWeightdWorker *worker,SparkWeightdWorkFunction function,void *context)
{
	(void)worker;
	Log("worker",0u);
	if ( WORKER_STATUS != SPARK_STATUS_OK )
		return(WORKER_STATUS);
	if ( WORKER_INLINE != 0u )
		function(context);
	else
	{
		PENDING_WORK = function;
		PENDING_CONTEXT = context;
	}
	return(SPARK_STATUS_OK);
}
SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t lease,uint64_t timeout) { (void)map; (void)lease; (void)timeout; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t lease,cudaStream_t stream) { (void)map; (void)lease; (void)stream; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkTpDeviceCollectiveChainKey(SparkTpDeviceCollective *collective,uint64_t request_id) { (void)collective; (void)request_id; Log("chain-key",0u); return(CHAIN_KEY_STATUS); }
cudaError_t cudaEventSynchronize(cudaEvent_t event) { (void)event; assert(0); return(cudaErrorUnknown); }
cudaError_t cudaMemcpy(void *destination,const void *source,size_t bytes,enum cudaMemcpyKind kind) { (void)destination; (void)source; (void)bytes; (void)kind; assert(0); return(cudaErrorUnknown); }
cudaError_t cudaMemsetAsync(void *destination,int value,size_t bytes,cudaStream_t stream) { (void)destination; (void)value; (void)bytes; (void)stream; assert(0); return(cudaErrorUnknown); }
SparkStatus SparkKvPageCacheCompleteLane(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane) { (void)cache; (void)lane; return(SPARK_STATUS_OK); }
SparkStatus SparkKvPageCacheRollbackLaneTransaction(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane,uint32_t flags) { (void)cache; (void)lane; (void)flags; return(SPARK_STATUS_OK); }

static void Completed(void *context,const SparkModelDriverCompletion *completion)
{
	(void)context;
	COMPLETED_STATUS = completion->status;
	COMPLETED_COUNT++;
}

static SparkGlm52ModuleState state;
static SparkWeightdLazyPack lazy;
static uint32_t host_tokens[4],host_slots[4],host_positions[4],host_output[4],host_error[8];
static uint16_t dev_hidden[8],dev_attn[8],dev_gather[8];
static uint64_t dev_maxloc[4];
static SparkGlm52ResidentDecodeStageBatchView batch;
static SparkGlm52ResidentDecodeStageFrameContext context;
static SparkModelDriverFrame frame;

static void Reset(uint32_t mode,uint32_t split,uint32_t layers)
{
	LOG_COUNT = 0u;
	ENQUEUE_COUNT = 0u;
	ENQUEUE_FAIL_AT = UINT32_MAX;
	VERIFY_STATUS = SPARK_STATUS_OK;
	GRAPH_ERROR = 0u;
	COMPLETED_COUNT = 0u;
	COMPLETED_STATUS = SPARK_STATUS_INTERNAL_ERROR;
	state.chain_mode = mode;
	state.projection_split = split;
	state.layer_count = layers;
	atomic_store(&state.chain_busy,1u);
	SparkStageModuleAtomicStateArrayInitialize(state.slot_states,1u);
	atomic_store(&state.slot_states[0],1u);
}

static SparkGlm52TpChain *NewChain(uint32_t position)
{
	SparkGlm52TpChain *chain = (SparkGlm52TpChain *)calloc(1u,sizeof(*chain));
	SparkGlm52AsyncCompletion *async = &state.completions[0];
	host_positions[0] = position;
	memset(async,0,sizeof(*async));
	async->state = &state;
	async->slot_index = 0u;
	async->row_count = 1u;
	async->completion_function = Completed;
	async->completion.status = SPARK_STATUS_OK;
	chain->state = &state;
	chain->slot = &state.slots[0];
	chain->slot_index = 0u;
	chain->frame = &frame;
	chain->batch = &batch;
	chain->context = &context;
	chain->wave_rows = 1u;
	chain->next_wave_row = 1u;
	chain->active = 1u;
	return(chain);
}

static void Setup(void)
{
	SparkGlm52ExecutionSlot *slot = &state.slots[0];
	STATE = &state;
	state.tp_degree = 16u;
	state.tp_rank = 0u;
	state.tp_device_collective_initialized = 1u;
	state.first_layer_index = 0u;
	state.pipeline_slot_count = 1u;
	state.resident_sequence_capacity = 4u;
	state.execution_row_capacity = 4u;
	state.max_sequence_positions = 4096u;
	state.decode_split_context_threshold = 64u;
	state.owns_embedding = 1u;
	state.owns_final_head = 1u;
	state.lazy_pack = &lazy;
	lazy.worker = (SparkWeightdWorker *)(uintptr_t)0x40;
	state.experts_pinned = 1u;
	state.execution_stream = (void *)(uintptr_t)0x50;
	slot->stream = state.execution_stream;
	slot->host_token_ids = host_tokens;
	slot->host_resident_slots = host_slots;
	slot->host_positions = host_positions;
	slot->host_output_token_ids = host_output;
	slot->host_kv_access_error = host_error;
	slot->hidden_bf16 = dev_hidden;
	slot->attention_out_bf16 = dev_attn;
	slot->projection_gather_bf16 = dev_gather;
	slot->head_maxloc_u64 = dev_maxloc;
	assert(SparkStageModuleCudaWaitInitialize(&state.chain_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	batch.row_count = 1u;
	batch.active_sequence_count = 1u;
	batch.row_resident_slots = host_slots;
	context.batch = &batch;
	frame.request_id = 7u;
}

static void TestConfigure(void)
{
	SparkGlm52ModuleState probe;
	memset(&probe,0,sizeof(probe));
	probe.tp_degree = 16u;
	probe.tp_device_collective_initialized = 1u;
	probe.lazy_pack = &lazy;
	probe.experts_pinned = 1u;
	probe.execution_stream = (void *)(uintptr_t)0x50;
	assert(setenv("SPARK_GLM52_CHAIN_MODE","graph",1) == 0);
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_OK && probe.chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && probe.chain_wait_initialized == 1u);
	probe.chain_wait_initialized = 0u;
	probe.experts_pinned = 0u;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	probe.experts_pinned = 1u;
	STREAM_ORDERED = 0u;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	STREAM_ORDERED = 1u;
	T1_ENABLED = 1u;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	T1_ENABLED = 0u;
	probe.lazy_pack = 0;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	assert(setenv("SPARK_GLM52_CHAIN_MODE","graphs",1) == 0);
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM52_CHAIN_MODE","eager",1) == 0);
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_OK && probe.chain_mode == SPARK_TP_CHAIN_MODE_EAGER);
	assert(unsetenv("SPARK_GLM52_CHAIN_MODE") == 0);
	assert(setenv("SPARK_GLM52_PROJECTION_SPLIT","2",1) == 0);
	assert(SparkGlm52ProjectionSplitConfigure(&probe) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM52_PROJECTION_SPLIT","1",1) == 0);
	probe.tp_collective_disabled = 1u;
	assert(SparkGlm52ProjectionSplitConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	probe.tp_collective_disabled = 0u;
	assert(SparkGlm52ProjectionSplitConfigure(&probe) == SPARK_STATUS_OK && probe.projection_split == 1u);
	assert(unsetenv("SPARK_GLM52_PROJECTION_SPLIT") == 0);
}

static void CheckLinearOrder(uint32_t layers,uint32_t split)
{
	int32_t at;
	uint32_t layer;
	char entry[40];
	at = Find("begin10",0u);
	assert(at >= 0);
	at = Find("reduce-hidden1",(uint32_t)at);
	assert(at >= 0);
	for (layer=0u; layer<layers; layer++)
	{
		if ( split != 0u )
		{
			(void)snprintf(entry,sizeof(entry),"project%u",layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
			at = Find("reduce-gather1",(uint32_t)at); assert(at >= 0);
			(void)snprintf(entry,sizeof(entry),"core%u",layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
		}
		else
		{
			(void)snprintf(entry,sizeof(entry),"attn%u",layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
		}
		at = Find("reduce-attn1",(uint32_t)at); assert(at >= 0);
		(void)snprintf(entry,sizeof(entry),"mlp%u",layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
		at = Find("reduce-hidden1",(uint32_t)at); assert(at >= 0);
	}
	at = Find("head0",(uint32_t)at); assert(at >= 0);
	at = Find("reduce-max1",(uint32_t)at); assert(at >= 0);
	at = Find("unpack1",(uint32_t)at); assert(at >= 0);
	at = Find("d2h4",(uint32_t)at); assert(at >= 0);
	at = Find("d2h24",(uint32_t)at); assert(at >= 0);
	at = Find("worker0",(uint32_t)at); assert(at >= 0);
	at = Find("verify0",(uint32_t)at); assert(at >= 0);
	assert(Count("reduce-hidden1") == layers + 1u && Count("reduce-attn1") == layers && Count("reduce-max1") == 1u);
	assert(Count("reduce-gather1") == (split != 0u ? layers : 0u));
}

static void TestLinear(void)
{
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,3u);
	SparkGlm52RunChain(NewChain(9u));
	CheckLinearOrder(3u,0u);
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK && atomic_load(&state.chain_busy) == 0u);
	assert(Count("capture-begin0") == 0u && Count("graph-launch257") == 0u);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,1u,2u);
	SparkGlm52RunChain(NewChain(9u));
	CheckLinearOrder(2u,1u);
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
}

static void TestLinearSettleFailure(void)
{
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,2u);
	VERIFY_STATUS = SPARK_STATUS_IO_ERROR;
	SparkGlm52RunChain(NewChain(9u));
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_IO_ERROR && Count("cancel0") == 1u && atomic_load(&state.chain_busy) == 0u);
}

static void TestLinearWalkFailure(void)
{
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,2u);
	ENQUEUE_FAIL_AT = 3u;
	ENQUEUE_STATUS = SPARK_STATUS_BUSY;
	SparkGlm52RunChain(NewChain(9u));
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_INTERNAL_ERROR && Count("cancel0") == 1u && Count("worker0") == 0u && atomic_load(&state.chain_busy) == 0u);
}

static void TestGraph(void)
{
	uint32_t captures;
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,2u);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
	CAPTURES = 0u;
	LAUNCHES = 0u;
	SparkGlm52RunChain(NewChain(9u));
	assert(CAPTURES == 1u && LAUNCHES == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	assert(Find("arm0",0u) < Find("capture-begin0",0u) && Find("capture-begin0",0u) < Find("cap:begin63",0u));
	assert(Count("cap:reduce-hidden1") == 3u && Count("cap:reduce-attn1") == 2u && Count("cap:reduce-max1") == 1u && Count("reduce-hidden1") == 0u);
	assert(Find("capture-end0",0u) < Find("disarm0",0u) && Find("disarm0",0u) < Find("prelaunch0",0u) && Find("prelaunch0",0u) < Find("seed0",0u) && Find("seed0",0u) < Find("graph-launch257",0u));
	assert(Find("graph-launch257",0u) < Find("d2h24",0u) && Find("graph-error0",0u) > Find("worker0",0u) && Count("disarm0") == 2u && Count("verify0") == 0u);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,2u);
	SparkGlm52RunChain(NewChain(40u));
	assert(CAPTURES == 1u && LAUNCHES == 2u && Count("capture-begin0") == 0u && Count("graph-launch257") == 1u && Count("begin41") == 0u);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,2u);
	SparkGlm52RunChain(NewChain(64u));
	assert(CAPTURES == 2u && Count("cap:begin2048") == 1u && Count("graph-launch258") == 1u);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,2u);
	SparkGlm52RunChain(NewChain(1500u));
	assert(CAPTURES == 2u && Count("graph-launch258") == 1u);
	captures = CAPTURES;
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,2u);
	SparkGlm52RunChain(NewChain(2048u));
	assert(CAPTURES == captures && LAUNCHES == 4u && Count("begin2049") == 1u && Count("verify0") == 1u && state.chain_gates[SPARK_GLM52_GRAPH_GATE_SELECTED_CONTEXT] == 1u);
	assert(COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,2u);
	GRAPH_ERROR = 3u;
	SparkGlm52RunChain(NewChain(20u));
	assert(COMPLETED_STATUS == SPARK_STATUS_IO_ERROR && Count("cancel0") == 1u && Count("disarm0") == 1u);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
}

static void TestGraphMultiWave(void)
{
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	CAPTURES = 0u;
	LAUNCHES = 0u;
	batch.row_count = 2u;
	host_slots[0] = 0u;
	host_slots[1] = 0u;
	host_positions[1] = 10u;
	atomic_store(&state.lane_states[0],1u);
	SparkGlm52RunChain(NewChain(9u));
	atomic_store(&state.lane_states[0],0u);
	batch.row_count = 1u;
	assert(CAPTURES == 0u && LAUNCHES == 0u && Count("capture-begin0") == 0u);
	assert(Find("begin10",0u) >= 0 && Find("begin11",0u) > Find("begin10",0u) && Count("unpack1") == 2u && Count("verify0") == 1u);
	assert(state.chain_gates[SPARK_GLM52_GRAPH_GATE_MULTI_WAVE] == 1u && COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK && atomic_load(&state.chain_busy) == 0u);
}

static void TestPrefillWaveRowsConfigure(void)
{
	SparkGlm52ModuleState probe;
	memset(&probe,0,sizeof(probe));
	probe.execution_row_capacity = 4u;
	assert(unsetenv("SPARK_GLM52_PREFILL_WAVE_ROWS") == 0);
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_OK && probe.prefill_wave_rows == 0u);
	assert(setenv("SPARK_GLM52_PREFILL_WAVE_ROWS","4",1) == 0);
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_OK && probe.prefill_wave_rows == 4u);
	assert(setenv("SPARK_GLM52_PREFILL_WAVE_ROWS","0",1) == 0);
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_OK && probe.prefill_wave_rows == 0u);
	assert(setenv("SPARK_GLM52_PREFILL_WAVE_ROWS","5",1) == 0);
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM52_PREFILL_WAVE_ROWS","2x",1) == 0);
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(unsetenv("SPARK_GLM52_PREFILL_WAVE_ROWS") == 0);
}

static SparkGlm52TpChain *NewPrefillChain(uint32_t first_position,uint32_t rows,uint32_t prefill,uint32_t cap)
{
	SparkGlm52TpChain *chain;
	uint32_t row;
	chain = NewChain(first_position);
	for (row=0u; row<rows; row++)
	{
		host_slots[row] = 0u;
		host_positions[row] = first_position + row;
	}
	batch.row_count = rows;
	state.completions[0].row_count = rows;
	state.prefill_wave_rows = cap;
	chain->prefill = prefill;
	chain->wave_rows = SparkGlm52WaveRows(chain,0u);
	chain->next_wave_row = chain->wave_rows;
	return(chain);
}

static void FinishPrefill(void)
{
	atomic_store(&state.lane_states[0],0u);
	batch.row_count = 1u;
	state.prefill_wave_rows = 0u;
}

static void TestPrefillWaves(void)
{
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	atomic_store(&state.lane_states[0],1u);
	SparkGlm52RunChain(NewPrefillChain(9u,4u,1u,4u));
	assert(Count("begin13") == 1u && Count("begin10") == 0u && Count("reduce-hidden4") == 2u && Count("reduce-attn4") == 1u && Count("unpack4") == 1u && Count("d2h16") == 1u && Count("head1") == 1u && Count("head0") == 0u);
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewPrefillChain(9u,4u,1u,2u));
	assert(Count("begin11") == 1u && Count("begin13") == 1u && Count("unpack2") == 2u && Count("reduce-attn2") == 2u && Count("head1") == 2u);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewPrefillChain(9u,4u,1u,0u));
	assert(Count("begin10") == 1u && Count("begin13") == 1u && Count("unpack1") == 4u && Count("head0") == 4u && Count("head1") == 0u);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewPrefillChain(9u,4u,0u,4u));
	assert(Count("unpack1") == 4u && Count("unpack4") == 0u && Count("head0") == 4u);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewPrefillChain(2046u,4u,1u,4u));
	assert(Count("begin2048") == 1u && Count("begin2050") == 1u && Count("unpack2") == 2u);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewPrefillChain(61u,4u,1u,4u));
	assert(Count("begin63") == 1u && Count("begin65") == 1u && Count("unpack2") == 2u);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewPrefillChain(20u,1u,1u,4u));
	assert(Count("begin21") == 1u && Count("unpack1") == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
	CAPTURES = 0u;
	LAUNCHES = 0u;
	SparkGlm52RunChain(NewPrefillChain(9u,4u,1u,4u));
	assert(CAPTURES == 1u && LAUNCHES == 1u && Count("cap:reduce-hidden4") == 2u && Count("cap:unpack4") == 1u && Count("cap:head1") == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
	FinishPrefill();
}

static void TestStreamFailure(void)
{
	uint32_t mode;
	for (mode=SPARK_TP_CHAIN_MODE_LINEAR; mode<=SPARK_TP_CHAIN_MODE_GRAPH; mode++)
	{
		Reset(mode,0u,1u);
		SparkTpChainGraphTableDestroy(&state.graphs[0]);
		STREAM_QUERY = cudaErrorUnknown;
		SparkGlm52RunChain(NewChain(9u));
		STREAM_QUERY = cudaSuccess;
		assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_IO_ERROR && Count("cancel0") == 1u && atomic_load(&state.chain_busy) == 0u);
		assert(Count("verify0") == 0u && Count("graph-error0") == 0u);
		assert(Count("disarm0") == (mode == SPARK_TP_CHAIN_MODE_GRAPH ? 2u : 0u));
	}
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
}

static void TestWorkerRefusal(void)
{
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	WORKER_STATUS = SPARK_STATUS_BUSY;
	SparkGlm52RunChain(NewChain(9u));
	WORKER_STATUS = SPARK_STATUS_OK;
	assert(Count("worker0") == 1u && Count("verify0") == 1u && COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK && atomic_load(&state.chain_busy) == 0u);
}

static void TestSubmitFailureClearsBusy(void)
{
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	atomic_store(&state.chain_busy,0u);
	assert(SparkGlm52ExecuteChain(&state,&frame,&context) == CHAIN_KEY_STATUS);
	assert(Count("chain-key0") == 1u && atomic_load(&state.chain_busy) == 0u && COMPLETED_COUNT == 0u);
}

static void TestBusy(void)
{
	WORKER_INLINE = 0u;
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewChain(9u));
	assert(COMPLETED_COUNT == 0u && atomic_load(&state.chain_busy) == 1u && PENDING_WORK != 0);
	assert(SparkGlm52ExecuteChain(&state,&frame,&context) == SPARK_STATUS_BUSY);
	PENDING_WORK(PENDING_CONTEXT);
	assert(COMPLETED_COUNT == 1u && atomic_load(&state.chain_busy) == 0u);
	WORKER_INLINE = 1u;
}

int main(void)
{
	Setup();
	TestConfigure();
	TestLinear();
	TestLinearSettleFailure();
	TestLinearWalkFailure();
	TestGraph();
	TestGraphMultiWave();
	TestPrefillWaveRowsConfigure();
	TestPrefillWaves();
	TestStreamFailure();
	TestWorkerRefusal();
	TestSubmitFailureClearsBusy();
	TestBusy();
	printf("glm52 chain modes: ok\n");
	return(0);
}
'''


def main():
    identity = subprocess.check_output([sys.executable, "tools/glm52_model_contract.py", "--print-build-identity", "fp8"], cwd=ROOT, text=True).split()
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "probe.c", Path(directory) / "probe"
        source.write_text(HARNESS)
        includes = [".", "include", "tests/cuda_stub", "model-families/common/include", "model-families/glm52/include",
                    "modules/glm52_resident_decode_stage/include", "modules/glm52_resident_decode_stage/source"]
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O1", "-g", "-ffunction-sections", "-fdata-sections",
                        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                        *["-I" + p for p in includes], "-DSPARK_BATCH_BUCKET=16u", "-DGLM_EXPERT_WEIGHT_CODEC=5",
                        '-DGLM_EXPERT_CODEC_NAME="fp8"', f'-DGLM_MODEL_REVISION="{identity[0]}"', f'-DGLM_CONTRACT_SHA256="{identity[1]}"',
                        '-DGLM_MODEL_DESCRIPTION_SHA256="fixture"', "-include", "model-families/glm52/include/sparkpipe/spark_glm52_model.h",
                        str(source), "runtime/stage_module_common.c", "src/spark_status.c", "-o", str(binary), "-pthread"], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS glm52 chain modes: linear walk order, graph capture/replay per regime, gates, settle and stream failures, worker refusal, busy gate, multi-row prefill waves (row cap, regime boundaries, graph)")


if __name__ == "__main__":
    main()
