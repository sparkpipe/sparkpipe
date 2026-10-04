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
static SparkStatus FENCE_STATUS,UPLOAD_STATUS;
static uint32_t FENCED;
static cudaError_t STREAM_QUERY = cudaSuccess;
static uint32_t ENQUEUE_FAIL_AT = UINT32_MAX,ENQUEUE_COUNT;
static uint64_t GRAPH_ERROR;
static SparkWeightdWorkFunction PENDING_WORK;
static void *PENDING_CONTEXT;
static SparkStatus COMPLETED_STATUS;
static uint32_t COMPLETED_COUNT;
static SparkGlm52ModuleState *STATE;
static uint32_t EAGER,KV_LATENT;
static SparkTpDeviceCollectiveCompletionFunction PENDING_COMPLETION;
static void *PENDING_COMPLETION_CONTEXT;

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
int32_t SparkGlm52LaunchCudaStageWaveInputs(const SparkGlm52CudaWave *wave,uint32_t rows,uint32_t bucket) { assert(wave->inputs_staged == 0u || wave->row_count >= rows); Log("stage",rows * 100u + bucket); return(0); }
int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->projection_split == 0u); Log("attn",layer); return(0); }
uint32_t SparkGlm52ProjectionSliceWidth(uint32_t tp_degree) { return(1000u + tp_degree); }
int32_t SparkGlm52LaunchCudaLayerAttentionProject(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->projection_split != 0u); Log("project",layer); return(0); }
int32_t SparkGlm52LaunchCudaLayerAttentionCore(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->projection_split != 0u); Log("core",layer); return(0); }
int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t layer) { assert(wave->route_host_copy == 0u); Log("mlp",layer); return(0); }
int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave) { Log("head",wave->row_head_certified); return(0); }
cudaError_t SparkGlm52LaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *tokens,uint32_t rows) { (void)stream; (void)maxloc; (void)tokens; Log("unpack",rows); return(cudaSuccess); }
int32_t SparkGlm52LaunchCudaLayerShard(const SparkGlm52CudaWave *wave,uint32_t layer,uint32_t phase)
{
	static const char *const names[6] = { "spre", "smid", "spost", "gpre", "gpost", "gmerge" };
	assert(phase < 6u && wave->kv_shard.degree > 1u && (wave->kv_gather != 0u) == (phase >= SPARK_GLM52_SHARD_PHASE_GATHER_PRE));
	Log(names[phase],layer);
	return(0);
}
int32_t SparkGlm52LaunchKvDigest(const SparkGlm52CudaWave *wave,uint32_t check) { assert(wave->kv_gather != 0u && check < 2u); Log("digest",check); return(0); }
uint32_t SparkGlm52LayerShardIndexing(const SparkGlm52CudaWave *wave,uint32_t layer) { return(wave->kv_shard.degree > 1u && wave->maximum_context > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT && layer == 0u ? 1u : 0u); }
int32_t SparkGlm52LaunchCudaLayerMlpRoute(const SparkGlm52CudaWave *wave,uint32_t layer) { (void)wave; (void)layer; assert(0); return(1); }
int32_t SparkGlm52LaunchCudaLayerMlpExperts(const SparkGlm52CudaWave *wave,uint32_t layer) { (void)wave; (void)layer; assert(0); return(1); }

static uint32_t host_tokens[16],host_slots[16],host_positions[16],host_output[16],host_error[8];

cudaError_t cudaMemcpyAsync(void *destination,const void *source,size_t bytes,enum cudaMemcpyKind kind,cudaStream_t stream)
{
	(void)source; (void)stream;
	Log(kind == cudaMemcpyDeviceToHost ? "d2h" : "copy",(uint32_t)bytes);
	if ( kind == cudaMemcpyDeviceToHost && (const uint32_t *)destination >= host_output && (const uint32_t *)destination < host_output + 16 )
		Log("d2h-at",(uint32_t)((const uint32_t *)destination - host_output));
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

static uint32_t KvOp(const void *pack,const void *pool,uint32_t *part,const SparkTpDeviceCollectiveSubmission *submission)
{
	uintptr_t bytes = (uintptr_t)submission->active_sequence_count * SPARK_GLM52_SHARD_UNIT_BYTES;
	if ( (uintptr_t)submission->local_device == (uintptr_t)pack )
		*part = 0u;
	if ( (uintptr_t)submission->local_device != (uintptr_t)pack + (uintptr_t)*part * bytes ||
		(uintptr_t)submission->full_device != (uintptr_t)pool + (uintptr_t)*part * STATE->kv_shard.degree * bytes )
		return(0u);
	(*part)++;
	return(1u);
}

SparkStatus SparkTpDeviceCollectiveEnqueue(SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission,uint32_t operation)
{
	SparkGlm52ExecutionSlot *slot = &STATE->slots[0];
	(void)collective;
	if ( EAGER != 0u )
	{
		assert(submission->completion_function != 0 && submission->completion_context != 0);
		PENDING_COMPLETION = submission->completion_function;
		PENDING_COMPLETION_CONTEXT = submission->completion_context;
	}
	else
		assert(submission->completion_function == 0 && submission->completion_context == 0);
	assert((submission->flags & SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION) != 0u);
	if ( ENQUEUE_COUNT++ == ENQUEUE_FAIL_AT )
		return(ENQUEUE_STATUS);
	if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 && submission->local_device == slot->kv_gather_digest + SPARK_GLM52_SHARD_DIGESTS &&
		submission->full_device == submission->local_device )
		Log("x-digest",submission->active_sequence_count);
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
		Log("reduce-max",submission->local_device == slot->head_maxloc_u64 ? 1u : 0u);
	else if ( submission->local_device == slot->hidden_bf16 )
		Log("reduce-hidden",submission->active_sequence_count);
	else if ( submission->local_device == slot->attention_out_bf16 )
		Log("reduce-attn",submission->active_sequence_count);
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER &&
		submission->local_device == slot->projection_local_bf16 && submission->full_device == slot->projection_gather_bf16 &&
		submission->row_elements == SparkGlm52ProjectionSliceWidth(STATE->tp_degree) )
		Log("reduce-gather",submission->active_sequence_count);
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && submission->local_device == slot->shard_send_bf16 && submission->full_device == slot->shard_received_bf16 && submission->row_elements == 0u )
		Log("x-gather",submission->active_sequence_count);
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL && submission->local_device == slot->shard_partials_f32 && submission->full_device == slot->shard_partials_received_f32 )
		Log("x-partials",submission->active_sequence_count);
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && submission->row_elements == 0u &&
		KvOp(slot->kv_gather_latent_pack,slot->kv_gather_latent_pool,&KV_LATENT,submission) != 0u )
		Log("x-kvL",submission->active_sequence_count);
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL && submission->local_device == slot->kv_select_candidates &&
		submission->full_device == slot->kv_select_received && submission->row_elements != 0u && submission->row_elements % 2u == 0u &&
		submission->row_elements <= 2u * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT )
		Log("x-select",submission->active_sequence_count);
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && submission->local_device == slot->kv_select_merged &&
		submission->full_device == slot->selected_positions && submission->row_elements == 2u * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT )
		Log("x-selected",submission->active_sequence_count);
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
SparkStatus SparkStageKvBindingContinuity(SparkStageKvBinding *binding,const atomic_uint *lane_states,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	uint32_t row,lane;
	(void)binding; (void)lane_states; (void)row_resident_slots;
	for (lane=0u; lane<active_count; lane++)
	{
		bound[lane] = 1u;
		sequence_ids[lane] = row_sequence_ids[lane];
		next_positions[lane] = row_positions[lane] + 1u;
	}
	for (row=active_count; row<row_count; row++)
	{
		lane = row % active_count;
		next_positions[lane] = row_positions[row] + 1u;
	}
	return(SPARK_STATUS_OK);
}
SparkStatus SparkStageKvBindingClaim(SparkStageKvBinding *binding,const SparkModelDriverFrame *frame,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,const uint64_t *next_positions) { (void)binding; (void)frame; (void)active_count; (void)row_resident_slots; (void)row_sequence_ids; (void)row_positions; (void)next_positions; return(SPARK_STATUS_OK); }
SparkStatus SparkStageKvBindingUploadPageTables(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream) { (void)binding; (void)resident_slots; (void)lane_count; (void)stream; Log("upload",FENCED); assert(FENCED == 1u); FENCED = 0u; return(UPLOAD_STATUS); }
SparkStatus SparkStageKvBindingFinishAsync(SparkStageKvBinding *binding,uint32_t dispatch_slot,const SparkStageKvBindingCompletion *completion) { (void)binding; (void)dispatch_slot; completion->finished_function(completion->finished_context,completion->status); return(SPARK_STATUS_OK); }
SparkStatus SparkStageKvBindingFenceExecution(SparkStageKvBinding *binding,void *stream) { (void)binding; assert(stream == STATE->execution_stream); Log("fence",0u); FENCED = 1u; return(FENCE_STATUS); }

static void Completed(void *context,const SparkModelDriverCompletion *completion)
{
	(void)context;
	COMPLETED_STATUS = completion->status;
	COMPLETED_COUNT++;
}

static SparkGlm52ModuleState state;
static SparkWeightdLazyPack lazy;
static uint16_t dev_hidden[8],dev_attn[8],dev_gather[8],dev_local[8],dev_shard_send[8],dev_shard_received[8];
static float dev_partials[8],dev_partials_received[8];
static uint8_t dev_kv_latent_pack[8],dev_kv_latent_pool[8];
static uint32_t dev_select_local[8],dev_select_merged[8],dev_selected[8];
static uint64_t dev_select_candidates[8],dev_select_received[8];
static unsigned long long dev_kv_digest[SPARK_GLM52_SHARD_DIGEST_WORDS];
static uint32_t dev_table[16];
static uint64_t dev_maxloc[4];
static SparkGlm52ResidentDecodeStageBatchView batch;
static SparkGlm52ResidentDecodeStageFrameContext context;
static SparkModelDriverFrame frame;

static void Reset(uint32_t mode,uint32_t split,uint32_t layers)
{
	LOG_COUNT = 0u;
	ENQUEUE_COUNT = 0u;
	KV_LATENT = 0u;
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
	SparkGlm52TpChain *chain = (SparkGlm52TpChain *)calloc(1u,sizeof(*chain) + 32u * sizeof(uint32_t));
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
	slot->projection_local_bf16 = dev_local;
	slot->prefill_block_table = dev_table;
	slot->head_maxloc_u64 = dev_maxloc;
	slot->shard_send_bf16 = dev_shard_send;
	slot->shard_received_bf16 = dev_shard_received;
	slot->shard_partials_f32 = dev_partials;
	slot->shard_partials_received_f32 = dev_partials_received;
	slot->kv_gather_latent_pack = dev_kv_latent_pack;
	slot->kv_gather_latent_pool = dev_kv_latent_pool;
	slot->kv_select_local = dev_select_local;
	slot->kv_select_candidates = dev_select_candidates;
	slot->kv_select_received = dev_select_received;
	slot->kv_select_merged = dev_select_merged;
	slot->selected_positions = dev_selected;
	slot->kv_gather_digest = dev_kv_digest;
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
	probe.owns_embedding = 1u;
	probe.owns_final_head = 1u;
	probe.max_sequence_positions = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
	probe.execution_row_capacity = SPARK_TP_CHAIN_GRAPH_MAX_ROWS;
	assert(setenv("SPARK_GLM52_CHAIN_MODE","graph",1) == 0);
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_OK && probe.chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && probe.chain_wait_initialized == 1u);
	probe.chain_wait_initialized = 0u;
	assert(SparkStageModuleCudaWaitDestroy(&probe.chain_wait) == SPARK_STATUS_OK);
	probe.max_sequence_positions = 1u << 20;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_OK && probe.chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && probe.chain_wait_initialized == 1u);
	probe.chain_wait_initialized = 0u;
	probe.max_sequence_positions = 1u << 23;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	probe.max_sequence_positions = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
	probe.owns_embedding = 0u;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	probe.owns_embedding = 1u;
	probe.owns_final_head = 0u;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	probe.owns_final_head = 1u;
	probe.execution_row_capacity = SPARK_TP_CHAIN_GRAPH_MAX_ROWS + 1u;
	assert(SparkGlm52ChainModeConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	probe.execution_row_capacity = SPARK_TP_CHAIN_GRAPH_MAX_ROWS;
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
	assert(CAPTURES == captures + 1u && LAUNCHES == 5u && Count("cap:begin4096") == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	captures = CAPTURES;
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,2u);
	SparkGlm52RunChain(NewChain(4096u));
	assert(CAPTURES == captures && LAUNCHES == 5u && Count("begin4097") == 0u && Count("stage101") == 0u);
	assert(COMPLETED_STATUS == SPARK_STATUS_UNSUPPORTED);
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
	assert(CAPTURES == 1u && LAUNCHES == 2u && Count("capture-begin0") == 1u && Count("stage101") == 2u && Count("d2h4") == 2u);
	assert(Find("stage101",0u) < Find("capture-begin0",0u) && Find("graph-launch257",0u) < Find("stage101",(uint32_t)Find("graph-launch257",0u)));
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK && atomic_load(&state.chain_busy) == 0u);
}

static void TestPrefillWaveRowsConfigure(void)
{
	SparkGlm52ModuleState probe;
	memset(&probe,0,sizeof(probe));
	probe.execution_row_capacity = 4u;
	assert(unsetenv("SPARK_GLM52_PREFILL_WAVE_ROWS") == 0);
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_OK && probe.prefill_wave_rows == 0u);
	assert(setenv("SPARK_GLM52_PREFILL_WAVE_ROWS","4",1) == 0);
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_UNSUPPORTED && probe.prefill_wave_rows == 0u);
	probe.owns_embedding = 1u;
	assert(SparkGlm52PrefillWaveRowsConfigure(&probe) == SPARK_STATUS_UNSUPPORTED);
	probe.owns_final_head = 1u;
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

static SparkGlm52TpChain *NewDecodeChain(uint32_t first_position,uint32_t rows)
{
	SparkGlm52TpChain *chain;
	uint32_t row;
	chain = NewChain(first_position);
	for (row=0u; row<rows; row++)
	{
		host_slots[row] = row;
		host_positions[row] = first_position + row;
		atomic_store(&state.lane_states[row],row + 1u);
	}
	batch.row_count = rows;
	batch.active_sequence_count = rows;
	state.completions[0].row_count = rows;
	chain->prefill = 0u;
	chain->wave_rows = SparkGlm52WaveRows(chain,0u);
	chain->next_wave_row = chain->wave_rows;
	return(chain);
}

static void FinishPrefill(void)
{
	uint32_t row;
	for (row=0u; row<4u; row++)
		atomic_store(&state.lane_states[row],0u);
	batch.active_sequence_count = 1u;
	batch.row_count = 1u;
	state.prefill_wave_rows = 0u;
}

static void TestOrderedPrefill(void)
{
	static const uint32_t slots[4] = { 0u, 1u, 0u, 1u },positions[4] = { 9u, 20u, 10u, 21u },tokens[4] = { 100u, 200u, 101u, 201u };
	static const uint32_t ordered_slots[4] = { 0u, 0u, 1u, 1u },ordered_positions[4] = { 9u, 10u, 20u, 21u },ordered_tokens[4] = { 100u, 101u, 200u, 201u };
	SparkGlm52TpChain *chain;
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	chain = NewChain(9u);
	memcpy(host_slots,slots,sizeof(slots));
	memcpy(host_positions,positions,sizeof(positions));
	memcpy(host_tokens,tokens,sizeof(tokens));
	atomic_store(&state.lane_states[0],1u);
	atomic_store(&state.lane_states[1],2u);
	batch.row_count = 4u;
	batch.active_sequence_count = 2u;
	state.completions[0].row_count = 4u;
	state.prefill_wave_rows = 4u;
	chain->prefill = 1u;
	assert(SparkGlm52OrderPrefillRows(chain) == SPARK_STATUS_OK && chain->row_ordered == 1u);
	assert(memcmp(host_slots,ordered_slots,sizeof(ordered_slots)) == 0 && memcmp(host_positions,ordered_positions,sizeof(ordered_positions)) == 0 && memcmp(host_tokens,ordered_tokens,sizeof(ordered_tokens)) == 0);
	chain->wave_rows = SparkGlm52WaveRows(chain,0u);
	chain->next_wave_row = chain->wave_rows;
	assert(chain->wave_rows == 4u);
	SparkGlm52RunChain(chain);
	assert(Count("begin11") == 0u && Count("begin22") == 1u && Count("head1") == 1u && Count("d2h4") == 2u && Count("d2h-at2") == 1u && Count("d2h-at3") == 1u);
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	atomic_store(&state.lane_states[0],0u);
	atomic_store(&state.lane_states[1],0u);
	batch.row_count = 1u;
	batch.active_sequence_count = 1u;
	state.prefill_wave_rows = 0u;
}

static uint32_t OrderedWaveRows(uint32_t first_position,uint32_t rows)
{
	SparkGlm52TpChain *chain;
	uint32_t row,wave;
	chain = NewChain(first_position);
	for (row=0u; row<rows; row++)
	{
		host_slots[row] = 0u;
		host_positions[row] = first_position + row;
		host_tokens[row] = 100u + row;
	}
	batch.row_count = rows;
	state.prefill_wave_rows = 16u;
	chain->prefill = 1u;
	assert(SparkGlm52OrderPrefillRows(chain) == SPARK_STATUS_OK);
	wave = SparkGlm52WaveRows(chain,0u);
	free(chain);
	batch.row_count = 1u;
	state.prefill_wave_rows = 0u;
	return(wave);
}

static void TestOrderedPrefillRegimes(void)
{
	assert(OrderedWaveRows(58u,12u) == 12u);
	assert(OrderedWaveRows(60u,6u) == 3u);
	assert(OrderedWaveRows(2040u,12u) == 8u);
}

static void TestPrefillWaves(void)
{
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	atomic_store(&state.lane_states[0],1u);
	SparkGlm52RunChain(NewPrefillChain(9u,4u,1u,4u));
	assert(Count("begin13") == 1u && Count("begin10") == 0u && Count("reduce-hidden4") == 2u && Count("reduce-attn4") == 1u && Count("unpack4") == 1u && Count("d2h4") == 1u && Count("d2h-at3") == 1u && Count("head1") == 1u && Count("head0") == 0u);
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
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	SparkGlm52RunChain(NewDecodeChain(9u,4u));
	assert(CAPTURES == 2u && LAUNCHES == 2u && Count("cap:head0") == 1u && Count("cap:head1") == 0u && Count("cap:unpack4") == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	SparkGlm52RunChain(NewPrefillChain(9u,4u,1u,4u));
	assert(CAPTURES == 2u && LAUNCHES == 3u && Count("capture-begin0") == 0u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	SparkGlm52RunChain(NewDecodeChain(9u,4u));
	assert(CAPTURES == 2u && LAUNCHES == 4u && Count("capture-begin0") == 0u && COMPLETED_STATUS == SPARK_STATUS_OK);
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

static void ExecuteFenceCase(SparkStatus fence_status,SparkStatus upload_status)
{
	static uint32_t tokens[1] = { 5u };
	static uint64_t positions[1] = { 9u },sequences[1] = { 11u };
	static uint32_t output[1];
	static SparkModelDriverBuffer buffers[1];
	SparkModelDriverFrame execute = frame;
	SparkGlm52ResidentDecodeStageBatchView view = batch;
	SparkGlm52ResidentDecodeStageFrameContext frame_context = context;
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	atomic_store(&state.chain_busy,0u);
	atomic_store(&state.slot_states[0],0u);
	view.token_ids = tokens;
	view.row_positions = positions;
	view.row_sequence_ids = sequences;
	frame_context.batch = &view;
	execute.execution_stream = state.execution_stream;
	execute.completion_function = Completed;
	buffers[0].address = output;
	execute.buffers = buffers;
	CHAIN_KEY_STATUS = SPARK_STATUS_OK;
	FENCE_STATUS = fence_status;
	UPLOAD_STATUS = upload_status;
	FENCED = 0u;
	assert(SparkGlm52ExecuteChain(&state,&execute,&frame_context) == SPARK_STATUS_OK);
	CHAIN_KEY_STATUS = SPARK_STATUS_IO_ERROR;
	FENCE_STATUS = UPLOAD_STATUS = SPARK_STATUS_OK;
	assert(COMPLETED_COUNT == 1u && atomic_load(&state.chain_busy) == 0u && atomic_load(&state.slot_states[0]) == 0u);
}

static void TestFenceBeforeUpload(void)
{
	ExecuteFenceCase(SPARK_STATUS_OK,SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(Find("fence0",0u) >= 0 && Find("fence0",0u) < Find("upload1",0u));
	assert(Count("fence0") == 1u && Count("upload1") == 1u && COMPLETED_STATUS == SPARK_STATUS_CAPACITY_EXCEEDED);
	ExecuteFenceCase(SPARK_STATUS_IO_ERROR,SPARK_STATUS_OK);
	assert(Count("fence0") == 1u && Count("upload0") == 0u && Count("upload1") == 0u && COMPLETED_STATUS == SPARK_STATUS_IO_ERROR);
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

static void Shard(uint32_t on)
{
	memset(&state.kv_shard,0,sizeof(state.kv_shard));
	if ( on != 0u )
	{
		state.kv_shard.degree = 16u;
		state.kv_shard.rank = 3u;
		state.kv_shard.grain = 1u;
	}
	state.index_layer_count = on;
}

static int32_t ExpectScatterLayer(uint32_t layer,uint32_t gather_units,uint32_t partial_units,int32_t at,const char *prefix)
{
	char entry[40];
	(void)snprintf(entry,sizeof(entry),"%sspre%u",prefix,layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%sx-gather%u",prefix,gather_units); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%ssmid%u",prefix,layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%sx-partials%u",prefix,partial_units); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%sspost%u",prefix,layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%sreduce-attn1",prefix); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%smlp%u",prefix,layer); at = Find(entry,(uint32_t)at); assert(at >= 0);
	return(at);
}

static void TestShardDecode(void)
{
	int32_t at;
	Shard(1u);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,2u);
	SparkGlm52RunChain(NewChain(9u));
	at = ExpectScatterLayer(0u,1u,1u,0,"");
	at = ExpectScatterLayer(1u,1u,1u,at,"");
	assert(Count("attn0") == 0u && Count("attn1") == 0u && Count("core0") == 0u && Count("x-gather1") == 2u && Count("x-partials1") == 2u && Count("x-kvL3") == 0u && Count("x-digest2") == 0u);
	assert(COMPLETED_COUNT == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,2u);
	SparkGlm52RunChain(NewChain(4000u));
	at = ExpectScatterLayer(0u,2u,1u,0,"");
	at = ExpectScatterLayer(1u,1u,1u,at,"");
	assert(Count("x-gather2") == 1u && Count("x-gather1") == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,1u,1u);
	SparkGlm52RunChain(NewChain(9u));
	at = Find("project0",0u); assert(at >= 0);
	at = Find("reduce-gather1",(uint32_t)at); assert(at >= 0);
	at = ExpectScatterLayer(0u,1u,1u,at,"");
	assert(Count("core0") == 0u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
	CAPTURES = 0u;
	LAUNCHES = 0u;
	SparkGlm52RunChain(NewChain(9u));
	(void)ExpectScatterLayer(0u,1u,1u,0,"cap:");
	assert(CAPTURES == 1u && LAUNCHES == 1u && Count("x-gather1") == 0u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	SparkGlm52RunChain(NewChain(20u));
	assert(CAPTURES == 1u && LAUNCHES == 2u && Count("capture-begin0") == 0u && COMPLETED_STATUS == SPARK_STATUS_OK);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
	Shard(0u);
}

static SparkGlm52TpChain *NewOrderedChain(uint32_t first_position,uint32_t rows)
{
	SparkGlm52TpChain *chain;
	uint32_t row;
	chain = NewChain(first_position);
	for (row=0u; row<rows; row++)
	{
		host_slots[row] = 0u;
		host_positions[row] = first_position + row;
		host_tokens[row] = 100u + row;
	}
	batch.row_count = rows;
	state.completions[0].row_count = rows;
	state.prefill_wave_rows = 16u;
	chain->prefill = 1u;
	assert(SparkGlm52OrderPrefillRows(chain) == SPARK_STATUS_OK);
	chain->wave_rows = SparkGlm52WaveRows(chain,0u);
	chain->next_wave_row = chain->wave_rows;
	atomic_store(&state.lane_states[0],1u);
	return(chain);
}

static int32_t ExpectDigest(int32_t at,const char *prefix)
{
	char entry[40];
	(void)snprintf(entry,sizeof(entry),"%sdigest0",prefix); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%sx-digest2",prefix); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%sdigest1",prefix); at = Find(entry,(uint32_t)at); assert(at >= 0);
	(void)snprintf(entry,sizeof(entry),"%shead1",prefix); at = Find(entry,(uint32_t)at); assert(at >= 0);
	return(at);
}

static void TestShardPrefillGather(void)
{
	int32_t at;
	uint32_t bound,op;
	SparkKvShardSectionLayout latent;
	char entry[40];
	Shard(1u);
	state.execution_row_capacity = 16u;
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewOrderedChain(9u,12u));
	at = Find("gpre0",0u); assert(at >= 0);
	at = Find("x-kvL3",(uint32_t)at); assert(at >= 0);
	at = Find("gpost0",(uint32_t)at); assert(at >= 0);
	at = Find("reduce-attn12",(uint32_t)at); assert(at >= 0);
	(void)ExpectDigest(at,"");
	assert(Count("x-kvL3") == 1u && Count("x-select2") == 0u && Count("gmerge0") == 0u && Count("spre0") == 0u && Count("x-gather1") == 0u && Count("x-digest2") == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewOrderedChain(3000u,12u));
	at = Find("gpre0",0u); assert(at >= 0);
	at = Find("x-select2",(uint32_t)at); assert(at >= 0);
	at = Find("gmerge0",(uint32_t)at); assert(at >= 0);
	at = Find("x-selected1",(uint32_t)at); assert(at >= 0);
	at = Find("x-kvL9",(uint32_t)at); assert(at >= 0);
	at = Find("x-kvL9",(uint32_t)at + 1u); assert(at >= 0);
	at = Find("gpost0",(uint32_t)at); assert(at >= 0);
	(void)ExpectDigest(at,"");
	assert(Count("x-select2") == 1u && Count("x-selected1") == 1u && Count("x-kvL9") == 2u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewOrderedChain(0u,12u));
	at = Find("gpre0",0u); assert(at >= 0);
	at = Find("gpost0",(uint32_t)at); assert(at >= 0);
	(void)ExpectDigest(at,"");
	assert(Count("x-kvL3") == 0u && Count("x-kvL9") == 0u && Count("x-select2") == 0u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_LINEAR,0u,1u);
	SparkGlm52RunChain(NewOrderedChain(9u,6u));
	assert(Count("gpre0") == 0u && Count("spre0") == 1u && Count("x-digest2") == 0u && COMPLETED_STATUS == SPARK_STATUS_OK);
	Reset(SPARK_TP_CHAIN_MODE_GRAPH,0u,1u);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
	CAPTURES = 0u;
	(void)SparkGlm52GraphRegime(3012u,state.decode_split_context_threshold,state.max_sequence_positions,&bound);
	SparkGlm52ShardGatherSizes(&state,bound,&latent);
	assert(latent.chunks != 0u);
	SparkGlm52RunChain(NewOrderedChain(3000u,12u));
	at = Find("cap:gpre0",0u); assert(at >= 0);
	at = Find("cap:x-select2",(uint32_t)at); assert(at >= 0);
	at = Find("cap:gmerge0",(uint32_t)at); assert(at >= 0);
	at = Find("cap:x-selected1",(uint32_t)at); assert(at >= 0);
	for (op=0u; op<latent.chunks; op++)
	{
		(void)snprintf(entry,sizeof(entry),"cap:x-kvL%u",latent.chunk_units); at = Find(entry,(uint32_t)at + 1u); assert(at >= 0);
	}
	at = Find("cap:gpost0",(uint32_t)at); assert(at >= 0);
	(void)ExpectDigest(at,"cap:");
	assert(CAPTURES == 1u && COMPLETED_STATUS == SPARK_STATUS_OK);
	SparkTpChainGraphTableDestroy(&state.graphs[0]);
	FinishPrefill();
	state.execution_row_capacity = 4u;
	Shard(0u);
}

static void TestShardRows(void)
{
	uint32_t capacity = state.execution_row_capacity;
	state.execution_row_capacity = 100u;
	Shard(0u);
	assert(SparkGlm52ShardCapRows(&state,80u) == 80u);
	Shard(1u);
	assert(SparkGlm52ShardScatterRows(&state) == SPARK_GLM52_SHARD_SCATTER_ROWS && SparkGlm52ShardCapRows(&state,80u) == SPARK_GLM52_SHARD_SCATTER_ROWS && SparkGlm52ShardCapRows(&state,20u) == 20u);
	assert(SparkGlm52ShardChunkCapacity(&state) == 99u);
	state.execution_row_capacity = capacity;
	Shard(0u);
}

static void EagerRun(SparkGlm52TpChain *chain,const char *until)
{
	uint32_t steps;
	EAGER = 1u;
	PENDING_COMPLETION = 0;
	SparkGlm52BuildWave(chain);
	chain->stage = SPARK_GLM52_CHAIN_STAGE_ATTENTION;
	chain->next_layer = 0u;
	SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
	for (steps=0u; steps<16u && Count(until) == 0u && PENDING_COMPLETION != 0; steps++)
	{
		SparkTpDeviceCollectiveCompletion done;
		SparkTpDeviceCollectiveCompletionFunction function = PENDING_COMPLETION;
		memset(&done,0,sizeof(done));
		done.status = SPARK_STATUS_OK;
		PENDING_COMPLETION = 0;
		function(PENDING_COMPLETION_CONTEXT,&done);
	}
	EAGER = 0u;
	assert(Count(until) == 1u);
	chain->active = 0u;
	free(chain);
}

static void TestShardEager(void)
{
	int32_t at;
	Shard(1u);
	Reset(SPARK_TP_CHAIN_MODE_EAGER,0u,1u);
	EagerRun(NewChain(9u),"reduce-attn1");
	at = Find("spre0",0u); assert(at >= 0);
	at = Find("x-gather1",(uint32_t)at); assert(at >= 0);
	at = Find("smid0",(uint32_t)at); assert(at >= 0);
	at = Find("x-partials1",(uint32_t)at); assert(at >= 0);
	at = Find("spost0",(uint32_t)at); assert(at >= 0);
	at = Find("reduce-attn1",(uint32_t)at); assert(at >= 0);
	state.execution_row_capacity = 16u;
	Reset(SPARK_TP_CHAIN_MODE_EAGER,0u,1u);
	EagerRun(NewOrderedChain(3000u,12u),"reduce-attn12");
	at = Find("gpre0",0u); assert(at >= 0);
	at = Find("x-select2",(uint32_t)at); assert(at >= 0);
	at = Find("gmerge0",(uint32_t)at); assert(at >= 0);
	at = Find("x-selected1",(uint32_t)at); assert(at >= 0);
	at = Find("x-kvL9",(uint32_t)at); assert(at >= 0);
	at = Find("x-kvL9",(uint32_t)at + 1u); assert(at >= 0);
	at = Find("gpost0",(uint32_t)at); assert(at >= 0);
	at = Find("reduce-attn12",(uint32_t)at); assert(at >= 0);
	FinishPrefill();
	state.execution_row_capacity = 4u;
	Shard(0u);
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
	TestOrderedPrefill();
	TestOrderedPrefillRegimes();
	TestStreamFailure();
	TestWorkerRefusal();
	TestSubmitFailureClearsBusy();
	TestBusy();
	TestFenceBeforeUpload();
	TestShardRows();
	TestShardDecode();
	TestShardPrefillGather();
	TestShardEager();
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
    print("PASS glm52 chain modes: linear walk order, graph capture/replay per regime, gates, settle and stream failures, worker refusal, busy gate, multi-row prefill waves (row cap, regime boundaries, graph keyed by head path), KV context split (scatter exchanges per layer in decode with candidates only on indexed layers, old-context key gather and row-owner DSA selection for single-sequence prefill waves, graph capture of the exchanges, eager stages, scatter row cap)")


if __name__ == "__main__":
    main()
