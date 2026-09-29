#!/usr/bin/env python3
"""Exercise the real module configurator without allocating CUDA state."""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile
from test_generated_control_admission import generate_admission

ROOT = Path(__file__).resolve().parents[1]
WAVE_TIMING = ROOT / "tests/fixtures/glm5_next_wave_timing.txt"
HARNESS = r'''
#include <assert.h>
#include <errno.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"
#include "cache/kv_page_store.c"
#include "src/spark_speculation_policy.c"
#include "src/spark_speculation_lookup_draft.c"
#include "src/spark_speculation_drafter_mix.c"
#include "src/spark_speculation_reference_draft.c"
#include "src/spark_speculation_relay_draft.c"
#include "src/spark_speculation_relay_link.c"
#include "src/spark_speculation_tap.c"
#include "runtime/spark_expert_working_set.c"
#include "src/spark_sha256.c"
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

cudaError_t SparkTpLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;
	(void)destination;
	(void)source;
	(void)element_count;
	return(cudaErrorInvalidValue);
}

static uint32_t REAL_BACKEND;
static uint64_t ARENA_KEY_STRIDE_SKEW;
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

static uint32_t ALL_TO_ALL_ADVERTISED = 3u,MESH_ATTACHED[2],COLLECTIVES_CREATED;
static uint8_t COLLECTIVE_MARKERS[2];

static uint32_t collective_slot(const SparkTpDeviceCollective *collective)
{
    assert(collective->implementation == &COLLECTIVE_MARKERS[0] || collective->implementation == &COLLECTIVE_MARKERS[1]);
    return(collective->implementation == &COLLECTIVE_MARKERS[1] ? 1u : 0u);
}

uint32_t SparkTpDeviceCollectiveAllToAllSupported(const SparkTpDeviceCollective *collective)
{
    uint32_t slot = collective_slot(collective);
    return((ALL_TO_ALL_ADVERTISED & (1u << slot)) != 0u && MESH_ATTACHED[slot] != 0u ? 1u : 0u);
}

SparkStatus SparkTpDeviceCollectiveApplyTopology(const SparkTpDeviceCollectiveTopology *topology,SparkTpDeviceCollectiveConfig *config)
{
    (void)topology;(void)config;
    return(SPARK_STATUS_OK);
}

SparkStatus SparkTpDeviceCollectiveCreate(const SparkTpDeviceCollectiveConfig *config,SparkTpDeviceCollective *collective_out)
{
    assert(config->mesh_band_index < 2u && COLLECTIVES_CREATED == config->mesh_band_index);
    memset(collective_out,0,sizeof(*collective_out));
    collective_out->implementation = &COLLECTIVE_MARKERS[config->mesh_band_index];
    COLLECTIVES_CREATED++;
    return(SPARK_STATUS_OK);
}

SparkStatus SparkTpDeviceCollectivePrepareReceiveBf16(SparkTpDeviceCollective *collective,void *receive_device,uint32_t active_sequence_count,uint32_t hidden_dimension,uint32_t step_index,void *cuda_stream)
{
    (void)active_sequence_count;(void)hidden_dimension;(void)step_index;(void)cuda_stream;
    assert(receive_device == (void *)(uintptr_t)state.lazy_pack->attached.mesh_send_buffer_addr);
    MESH_ATTACHED[collective_slot(collective)] = 1u;
    return(SPARK_STATUS_OK);
}

SparkStatus SparkTpDeviceCollectiveMeshTopology(uint32_t rank,uint32_t degree,struct SparkWeightdMeshTopology *topology)
{
    (void)rank;(void)degree;(void)topology;
    abort();
}

SparkStatus SparkWeightdClientConnect(const char *socket_path,SparkWeightdClient **client,SparkWeightdHelloResult *hello_out)
{
    (void)socket_path;(void)client;(void)hello_out;
    abort();
}

SparkStatus SparkWeightdClientLaneAcquire(SparkWeightdClient *client,uint32_t requested_lane,const SparkWeightdMeshTopology *topology,uint32_t *lane_out,uint64_t timeout_nanoseconds)
{
    (void)client;(void)requested_lane;(void)topology;(void)lane_out;(void)timeout_nanoseconds;
    abort();
}

static char WALK_TRACE[256];
static uint32_t WALK_LENGTH,WALK_GATHER_LAYER,WALK_FAIL_CODE,WALK_DELAY_NS,WALK_SHARD_MASK,SHARD_QUERY_SEQUENCES,SHARD_PARTIAL_SEQUENCES,SHARD_WIDE;

static int32_t walk_note(char code)
{
	struct timespec delay = {0,(long)WALK_DELAY_NS};
	if ( code == 'B' && WALK_DELAY_NS != 0u )
		nanosleep(&delay,0);
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
uint32_t SparkGlm5NextLayerKvShardActive(const SparkGlm5NextCudaWave *wave,uint32_t layer) { return(wave->kv_shard != 0u && layer < 32u && (WALK_SHARD_MASK & (1u << layer)) != 0u ? 1u : 0u); }
int32_t SparkGlm5NextLaunchCudaLayerAttentionShardPartial(const SparkGlm5NextCudaWave *wave,uint32_t layer) { assert(SparkGlm5NextLayerKvShardActive(wave,layer) != 0u);return(walk_note('V')); }
int32_t SparkGlm5NextLaunchCudaLayerAttentionShardMerge(const SparkGlm5NextCudaWave *wave,uint32_t layer) { assert(SparkGlm5NextLayerKvShardActive(wave,layer) != 0u);return(walk_note('W')); }
int32_t SparkGlm5NextLaunchCudaLayerMlp(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('M')); }
int32_t SparkGlm5NextLaunchCudaLayerMlpRoute(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('O')); }
int32_t SparkGlm5NextLaunchCudaLayerMlpRouteResident(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('R')); }
int32_t SparkGlm5NextLaunchCudaLayerMlpExperts(const SparkGlm5NextCudaWave *wave,uint32_t layer) { assert(wave->expert_lease_all == 1u);(void)layer;return(walk_note('E')); }
int32_t SparkGlm5NextLaunchCudaLayerAttentionPost(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('P')); }
int32_t SparkGlm5NextLaunchCudaLayerMlpPost(const SparkGlm5NextCudaWave *wave,uint32_t layer) { (void)wave;(void)layer;return(walk_note('Q')); }
int32_t SparkGlm5NextLaunchCudaTapCapture(const SparkGlm5NextCudaWave *wave,uint32_t all_streams,uint16_t *destination) { (void)wave;(void)all_streams;(void)destination;return(walk_note('T')); }
int32_t SparkGlm5NextLaunchCudaWaveHead(const SparkGlm5NextCudaWave *wave) { (void)wave;return(walk_note('H')); }
static uint32_t L2_CALLS,L2_SITES,L2_BYTES,L2_BLOCKS;
int32_t SparkGlm5NextL2PrefetchAfterRound(const SparkGlm5NextCudaWave *wave,uint32_t local_layer,uint32_t site,const SparkGlm5NextL2PrefetchShape *shape,uint32_t *placed) { (void)wave;(void)local_layer;L2_CALLS++;L2_SITES |= 1u << site;L2_BYTES = shape != 0 ? shape->bytes : 0u;L2_BLOCKS = shape != 0 ? shape->blocks : 0u;*placed = 1u;return(0); }
static uint32_t UNPACK_COUNT,UNPACK_ROWS = 2u,ENQUEUE_ROWS = 2u,ENQUEUE_SEQUENCES = 2u;

static uint32_t SNAPSHOT_SAVES,SNAPSHOT_RESTORES,POISON_LAUNCHES,POISON_PENDING,WS_PLAN[16],WS_PLAN_KEY;
cudaError_t SparkGlm5NextLaunchStateSnapshot(cudaStream_t stream,const void *spans,uint32_t span_count,uint32_t row_words,uint8_t *snapshot,const uint32_t *state_index,uint32_t rows,uint32_t restore)
{
	assert(stream != 0 && spans != 0 && span_count != 0u && row_words != 0u && snapshot != 0 && state_index != 0 && rows != 0u && rows <= SPARK_GLM5_NEXT_WS_ROWS_MAX);
	if ( restore != 0u )
		SNAPSHOT_RESTORES++;
	else
		SNAPSHOT_SAVES++;
	return(walk_note(restore != 0u ? 'L' : 's') != 0 ? cudaErrorInvalidValue : cudaSuccess);
}
cudaError_t SparkGlm5NextLaunchHeadMissPoison(cudaStream_t stream,const uint32_t *miss,uint64_t *maxloc,uint32_t row_count)
{
	uint32_t *ring = (uint32_t *)(uintptr_t)miss,plan = POISON_LAUNCHES < 16u ? WS_PLAN[POISON_LAUNCHES] : 0u;
	(void)stream;(void)maxloc;
	assert(miss != 0 && row_count != 0u);
	POISON_LAUNCHES++;
	if ( plan == 1u || plan == 3u )
	{
		assert(ring[SPARK_STEP_MISS_FLAG] == 0u && ring[SPARK_STEP_MISS_COUNT] == 0u);
		ring[SPARK_STEP_MISS_FLAG] = 1u;
		ring[SPARK_STEP_MISS_COUNT] = 1u;
		ring[SPARK_STEP_MISS_ENTRIES] = plan == 1u ? WS_PLAN_KEY : 60u * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE;
	}
	POISON_PENDING = plan != 0u ? 1u : 0u;
	return(walk_note('p') != 0 ? cudaErrorInvalidValue : cudaSuccess);
}

cudaError_t SparkGlm5NextLaunchHeadMaxlocUnpack(cudaStream_t stream,const uint64_t *maxloc,uint32_t *token_ids,uint32_t row_count)
{
	uint32_t row;
	assert(stream == state.execution_stream && maxloc != 0 && token_ids != 0 && row_count == UNPACK_ROWS);
	for (row=0u; row<row_count; row++)
		token_ids[row] = POISON_PENDING != 0u ? SPARK_STEP_POISON_TOKEN : 100u * UNPACK_COUNT + 11u + row;
	POISON_PENDING = 0u;
	UNPACK_COUNT++;
	return(walk_note('U') != 0 ? cudaErrorInvalidValue : cudaSuccess);
}

static uint32_t MTP_DRAFTS,MTP_LANE,MTP_OPS_CALLS,MTP_DEPTH_TOTAL,MTP_EXPECT_ROW = UINT32_MAX;
static const uint32_t *MTP_TRUTH;

int32_t SparkGlm5NextLaunchCudaMtpDraft(const SparkGlm5NextCudaWave *wave,const SparkGlm5NextMtpDraftOps *ops,uint16_t *committed_hidden_bf16,uint32_t first_token,uint32_t *host_draft_tokens)
{
	uint32_t index,position,wrong,reduce;
	MTP_DRAFTS++;
	if ( MTP_TRUTH == 0 )
	{
		(void)wave;(void)ops;(void)committed_hidden_bf16;(void)first_token;
		host_draft_tokens[0] = 21u;
		host_draft_tokens[1] = 22u;
		return(0);
	}
	assert(wave != 0 && wave->slot == &state.slots[0] && wave->mtp_draft_depth >= 1u && wave->mtp_draft_depth <= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX);
	assert(wave->tp_degree == state.tp_degree && wave->owns_final_head == 1u && wave->mtp_layer_weights == &state.mtp_layer);
	assert(committed_hidden_bf16 == state.mtp_lane_hidden_bf16 + (uint64_t)MTP_LANE * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION);
	assert(MTP_EXPECT_ROW == UINT32_MAX || committed_hidden_bf16[0] == 1000u + MTP_EXPECT_ROW);
	position = (uint32_t)state.mtp_lane_next[MTP_LANE];
	assert(first_token == MTP_TRUTH[position]);
	assert((state.tp_degree > 1u) == (ops != 0));
	for (index=0u; ops != 0 && index<wave->mtp_draft_depth; index++)
	{
		for (reduce=0u; reduce<3u; reduce++)
			assert(ops->reduce_rows_bf16(ops->context,state.slots[0].hidden_bf16,1u,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION) == SPARK_STATUS_OK);
		assert(ops->reduce_rows_bf16(ops->context,state.slots[0].hidden_bf16,2u,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION) == SPARK_STATUS_INVALID_ARGUMENT);
		assert(ops->reduce_max_u64(ops->context,state.slots[0].head_maxloc_u64,1u) == SPARK_STATUS_OK);
		MTP_OPS_CALLS += 4u;
	}
	MTP_DEPTH_TOTAL += wave->mtp_draft_depth;
	wrong = (position * 7u) % 9u;
	for (index=0u; index<wave->mtp_draft_depth; index++)
		host_draft_tokens[index] = index == wrong ? (MTP_TRUTH[position + 1u + index] + 1u) % 1000u : MTP_TRUTH[position + 1u + index];
	if ( position % 11u == 5u && wave->mtp_draft_depth >= 3u )
		host_draft_tokens[2] = UINT32_MAX;
	return(0);
}

SparkStatus SparkTpDeviceCollectiveEnqueue(SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission,uint32_t operation_kind)
{
	assert((submission->flags & SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION) != 0u);
	uint32_t shard_query = operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && submission->local_device != 0 && submission->local_device == state.slots[submission->slot_index].query_latent_bf16;
	assert(submission->cuda_stream == state.execution_stream && submission->logical_sequence_count == ENQUEUE_SEQUENCES);
	assert(submission->active_sequence_count == (operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? SHARD_PARTIAL_SEQUENCES : shard_query != 0u ? SHARD_QUERY_SEQUENCES : ENQUEUE_ROWS));
	if ( operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
		assert(collective == (SHARD_WIDE != 0u ? &state.tp_device_collective_hc : &state.tp_device_collective) && submission->local_device == state.slots[submission->slot_index].kv_shard_partials_f32 && submission->full_device == state.slots[submission->slot_index].kv_shard_partials_received_f32);
	if ( shard_query != 0u )
		assert(collective == &state.tp_device_collective && submission->full_device == state.slots[submission->slot_index].kv_shard_query_gathered_bf16);
	(void)walk_note(submission->completion_function != 0 ? (operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? 'e' : shard_query != 0u ? 'q' : 'c') : operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? 'a' : shard_query != 0u ? 'k' : collective == &state.tp_device_collective_hc ? 'h' : operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER ? 'g' : operation_kind == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 ? 'x' : 'r');
	return(SPARK_STATUS_OK);
}

const SparkWeightdRangeGroup *SparkWeightdManifestFind(const SparkWeightdManifest *manifest,uint32_t layer,uint32_t expert) { (void)manifest;(void)layer;(void)expert;abort(); }
uint64_t SparkTpDeviceCollectiveRoundIndex(SparkTpDeviceCollective *collective) { (void)collective;return(0u); }
static uint64_t CHAIN_KEYS;
SparkStatus SparkTpDeviceCollectiveChainKey(SparkTpDeviceCollective *collective,uint64_t chain_id) { (void)collective;(void)chain_id;CHAIN_KEYS++;return(SPARK_STATUS_OK); }
uint64_t SparkTpDeviceCollectiveChainEpoch(const SparkTpDeviceCollective *collective) { (void)collective;return(1u); }
SparkStatus SparkTpDeviceCollectiveArmCapture(SparkTpDeviceCollective *collective) { (void)collective;abort(); }
cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode) { (void)stream;(void)mode;abort(); }
cudaError_t cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t *graph) { (void)stream;(void)graph;abort(); }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec,cudaGraph_t graph,...) { (void)exec;(void)graph;abort(); }
cudaError_t cudaGraphUpload(cudaGraphExec_t exec,cudaStream_t stream) { (void)exec;(void)stream;abort(); }
cudaError_t cudaGraphDestroy(cudaGraph_t graph) { (void)graph;abort(); }
cudaError_t cudaEventSynchronize(cudaEvent_t event) { (void)event;abort(); }
SparkStatus SparkWeightdRouteKeys(uint32_t layer,const uint32_t *offsets,uint32_t expert_count,uint32_t packed_rows,SparkWeightdExpertKey *keys,uint32_t capacity,uint32_t *count)
{
	uint32_t expert;
	*count = 0u;
	if ( offsets[0] != 0u || offsets[expert_count] != packed_rows )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (expert=0u; expert<expert_count; expert++)
		if ( offsets[expert] != offsets[expert + 1u] )
		{
			if ( *count >= capacity )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			keys[*count].layer = layer;
			keys[*count].expert = expert;
			(*count)++;
		}
	return(SPARK_STATUS_OK);
}
cudaError_t SparkTpLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width) { (void)stream;(void)destination_bf16;(void)source_bf16;(void)row_count;(void)width;abort(); }
cudaError_t SparkTpLaunchSumRanksF32(cudaStream_t stream,void *destination,const void *const *sources,uint32_t source_count,uint32_t element_count) { (void)stream;(void)destination;(void)sources;(void)source_count;(void)element_count;abort(); }
cudaError_t SparkTpLaunchSeedF32(cudaStream_t stream,float *destination_f32,const void *source_a_bf16,const void *source_b_bf16,uint32_t element_count) { (void)stream;(void)destination_f32;(void)source_a_bf16;(void)source_b_bf16;(void)element_count;abort(); }
cudaError_t SparkTpLaunchAddF32(cudaStream_t stream,float *destination_f32,const void *source_bf16,uint32_t element_count) { (void)stream;(void)destination_f32;(void)source_bf16;(void)element_count;abort(); }
cudaError_t SparkTpLaunchRoundF32(cudaStream_t stream,void *destination_bf16,const float *source_f32,uint32_t element_count) { (void)stream;(void)destination_bf16;(void)source_f32;(void)element_count;abort(); }

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
	if ( pointer != 0 && pointer == (void *)state.slots[0].miss_ring && walk_note('z') != 0 )
		return(cudaErrorInvalidValue);
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

cudaError_t cudaMemcpy2DAsync(void *destination,size_t destination_pitch,const void *source,size_t source_pitch,size_t width,size_t height,cudaMemcpyKind kind,cudaStream_t stream)
{
	size_t row;
	(void)kind;
	(void)stream;
	for (row=0u; row<height; row++)
		memcpy((uint8_t *)destination + row * destination_pitch,(const uint8_t *)source + row * source_pitch,width);
	return(cudaSuccess);
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
static uint32_t VERIFY_MISS_RING[SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES / sizeof(uint32_t)];
const void *SparkWeightdMapEpochDevice(const SparkWeightdMap *map)
{ (void)map;return(EPOCH_WORDS); }
static uint32_t GRAPH_LAUNCHES;
static uint64_t GRAPH_ERROR;
static void (*GRAPH_LAUNCH_HOOK)(void);

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec,cudaStream_t stream)
{
	assert(exec == (cudaGraphExec_t)(uintptr_t)9u && stream == state.execution_stream);
	GRAPH_LAUNCHES++;
	if ( GRAPH_LAUNCH_HOOK != 0 )
		GRAPH_LAUNCH_HOOK();
	EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64]++;
	return(cudaSuccess);
}
SparkStatus SparkTpDeviceCollectiveGraphPreLaunch(SparkTpDeviceCollective *collective,void *stream)
{ (void)collective;(void)stream;return(SPARK_STATUS_OK); }
SparkStatus SparkTpDeviceCollectiveGraphSettle(SparkTpDeviceCollective *collective,void *stream,uint64_t *error_out)
{ (void)collective;(void)stream;if ( error_out == 0 ) return(SPARK_STATUS_INVALID_ARGUMENT);*error_out = GRAPH_ERROR;return(SPARK_STATUS_OK); }
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
	state.slots[0].miss_ring = (uint32_t *)EPOCH_WORDS;
	EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64] = 41u;
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	state.tp_device_collective.operation_timeout_milli = 50u;
	chain.state = &state;chain.slot = &state.slots[0];
	chain.wave.host_positions = &position;chain.wave.host_token_ids = &token;chain.wave_rows = 1u;
	GRAPH_LAUNCHES = 0u;GRAPH_ERROR = 0u;DRAIN_STATUS = cudaSuccess;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_OK && position == 7u && token == 3u);
	assert(GRAPH_LAUNCHES == 1u && EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64] == 42u);
	assert(state.completions[0].graph == 1u && state.completions[0].launch_ns != 0u);
	state.slots[0].miss_ring[SPARK_STEP_MISS_FLAG] = 1u;state.slots[0].miss_ring[SPARK_STEP_MISS_COUNT] = 3u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_INTERNAL_ERROR && chain.step_verdict == SPARK_STEP_VERDICT_POISON_LOST && position == 7u);
	assert(state.slots[0].miss_ring[SPARK_STEP_MISS_FLAG] == 0u && state.slots[0].miss_ring[SPARK_STEP_MISS_COUNT] == 0u);
	output = SPARK_STEP_POISON_TOKEN;state.slots[0].miss_ring[SPARK_STEP_MISS_FLAG] = 1u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_UNSUPPORTED && chain.step_verdict == SPARK_STEP_VERDICT_ROLLBACK_LOCAL && position == 7u && token == 3u);
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_UNSUPPORTED && chain.step_verdict == SPARK_STEP_VERDICT_ROLLBACK_REMOTE && position == 7u);
	assert(SparkGlm5NextGraphResult(&chain,status) == 1u && atomic_load(&state.terminal_status) == SPARK_STATUS_OK);
	assert(state.step_verdicts[SPARK_STEP_VERDICT_POISON_LOST] == 1u && state.step_verdicts[SPARK_STEP_VERDICT_ROLLBACK_LOCAL] == 1u && state.step_verdicts[SPARK_STEP_VERDICT_ROLLBACK_REMOTE] == 1u);
	output = 123u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_OK && chain.step_verdict == SPARK_STEP_VERDICT_COMMIT && state.step_verdicts[SPARK_STEP_VERDICT_COMMIT] == 0u);
	assert(SparkGlm5NextGraphResult(&chain,SPARK_STATUS_UNSUPPORTED) == 0u && atomic_load(&state.terminal_status) == SPARK_STATUS_OK);
	assert(SparkGlm5NextGraphResult(&chain,SPARK_STATUS_INTERNAL_ERROR) == 1u && atomic_load(&state.terminal_status) == SPARK_STATUS_INTERNAL_ERROR);
	atomic_store(&state.terminal_status,SPARK_STATUS_OK);
	GRAPH_ERROR = 7u;state.tp_device_collective_initialized = 1u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_INTERNAL_ERROR && position == 7u);
	GRAPH_ERROR = 0u;state.tp_device_collective_initialized = 0u;
	state.lane_client = (SparkWeightdClient *)(uintptr_t)1u;HEALTH_DEAD_MASK = 1u;
	SparkGlm5NextGraphStep(&chain,&status);
	assert(status == SPARK_STATUS_IO_ERROR && position == 7u && GRAPH_LAUNCHES == 6u);
	HEALTH_DEAD_MASK = 0u;
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
}

static uint32_t WS_ACQUIRED[64],WS_ACQUIRE_COUNT,WS_DENY;
static SparkStatus check_ws_acquire(void *context,const uint32_t *keys,uint32_t count)
{
	uint32_t index;
	assert(context == &state);
	if ( WS_DENY != 0u )
		return(SPARK_STATUS_BUSY);
	for (index=0u; index<count; index++)
		WS_ACQUIRED[WS_ACQUIRE_COUNT++] = keys[index];
	return(SPARK_STATUS_OK);
}

static void check_working_set_recover(void)
{
	static uint32_t ring[SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES / sizeof(uint32_t)];
	static uint8_t snapshot[64];
	static uint32_t spans[16],index_words[4];
	SparkGlm5NextTpChain chain = {0};
	SparkStepAction action;
	uint32_t anchor = SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + 5u;
	memset(&state,0,sizeof(state));
	memset(ring,0,sizeof(ring));
	state.execution_stream = (void *)(uintptr_t)7u;
	state.slots[0].stream = state.execution_stream;
	state.slots[0].miss_ring = ring;
	state.slots[0].snapshot = snapshot;
	state.slots[0].snapshot_spans = spans;
	state.slots[0].snapshot_span_count = 4u;
	state.slots[0].snapshot_row_words = 16u;
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	assert(SparkExpertWorkingSetCreate(&state.expert_ws,SPARK_GLM5_NEXT_MODEL_LAYER_COUNT,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE,64u,check_ws_acquire,&state) == SPARK_STATUS_OK);
	assert(SparkExpertWorkingSetAdd(&state.expert_ws,&anchor,1u) == SPARK_STATUS_OK);
	state.ws_enabled = 1u;
	chain.state = &state;chain.slot = &state.slots[0];chain.wave.slot = &state.slots[0];
	chain.wave.row_count = 1u;chain.wave_rows = 1u;chain.wave.kda_state_index = index_words;
	ring[SPARK_STEP_MISS_FLAG] = 1u;ring[SPARK_STEP_MISS_COUNT] = 3u;
	ring[SPARK_STEP_MISS_ENTRIES] = (SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER + 1u) * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + 7u;
	ring[SPARK_STEP_MISS_ENTRIES + 1u] = (SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER + 1u) * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + 2u;
	ring[SPARK_STEP_MISS_ENTRIES + 2u] = (SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER + 4u) * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + 9u;
	chain.step_verdict = SPARK_STEP_VERDICT_ROLLBACK_LOCAL;
	SNAPSHOT_RESTORES = 0u;WS_ACQUIRE_COUNT = 0u;
	assert(SparkGlm5NextWsRecover(&chain,&action) == SPARK_STATUS_OK && action == SPARK_STEP_ACTION_REPLAY);
	assert(SNAPSHOT_RESTORES == 1u && ring[SPARK_STEP_MISS_FLAG] == 0u && ring[SPARK_STEP_MISS_COUNT] == 0u);
	assert(WS_ACQUIRE_COUNT == 2u && WS_ACQUIRED[0] == ring[SPARK_STEP_MISS_ENTRIES + 1u] && WS_ACQUIRED[1] == ring[SPARK_STEP_MISS_ENTRIES]);
	assert(SparkExpertWorkingSetCovered(&state.expert_ws,SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER + 1u,7u) == 1u);
	assert(SparkExpertWorkingSetCovered(&state.expert_ws,SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER + 4u,9u) == 0u);
	assert(state.ws_local_miss == 1u && state.ws_replays == 1u && chain.ws_force_eager == 0u);
	chain.step_verdict = SPARK_STEP_VERDICT_ROLLBACK_REMOTE;WS_DENY = 1u;
	assert(SparkGlm5NextWsRecover(&chain,&action) == SPARK_STATUS_OK && action == SPARK_STEP_ACTION_REPLAY && state.ws_remote_miss == 1u);
	ring[SPARK_STEP_MISS_FLAG] = 1u;ring[SPARK_STEP_MISS_COUNT] = 1u;
	ring[SPARK_STEP_MISS_ENTRIES] = (SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER + 2u) * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + 1u;
	chain.step_verdict = SPARK_STEP_VERDICT_ROLLBACK_LOCAL;
	assert(SparkGlm5NextWsRecover(&chain,&action) == SPARK_STATUS_OK && action == SPARK_STEP_ACTION_EXHAUSTED);
	assert(chain.ws_force_eager == 1u && state.ws_eager_steps == 1u && state.expert_ws.grow_denied == 1u && SNAPSHOT_RESTORES == 3u);
	chain.wave_rows = 1u;
	assert(SparkGlm5NextGraphClaimExperts(&chain) == SPARK_STATUS_UNSUPPORTED);
	chain.slot->host_output_token_ids = index_words;chain.slot->host_positions = index_words;chain.slot->host_token_ids = index_words;chain.slot->host_chain_token_ids = index_words;chain.steps = 1u;
	SparkGlm5NextFeedStep(&chain);
	assert(chain.ws_force_eager == 0u);
	WS_DENY = 0u;
	ring[SPARK_STEP_MISS_FLAG] = 1u;ring[SPARK_STEP_MISS_COUNT] = SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY + 1u;
	for (uint32_t entry=0u; entry<SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY; entry++)
		ring[SPARK_STEP_MISS_ENTRIES + entry] = (SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER + 6u) * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + entry % SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT;
	WS_ACQUIRE_COUNT = 0u;
	assert(SparkGlm5NextWsRecover(&chain,&action) == SPARK_STATUS_OK && action == SPARK_STEP_ACTION_REPLAY && WS_ACQUIRE_COUNT == 0u && state.expert_ws.harvest_overflow == 1u);
	ring[SPARK_STEP_MISS_FLAG] = 0u;ring[SPARK_STEP_MISS_COUNT] = 2u;
	assert(SparkGlm5NextWsRecover(&chain,&action) == SPARK_STATUS_VALIDATION_FAILED && action == SPARK_STEP_ACTION_FAIL);
	SparkExpertWorkingSetDestroy(&state.expert_ws);
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

static uint32_t FOLD_ANY,FOLD_LAST;

int32_t SparkGlm5NextPrepareCudaReplayFold(const SparkGlm5NextCudaWave *wave,uint32_t rows)
{
	assert(IN_CUDA_CALLBACK == 0u && wave != 0 && rows <= SPARK_GLM5_NEXT_REPLAY_ROWS_MAX);
	return(0);
}

int32_t SparkGlm5NextLaunchCudaReplayFold(const SparkGlm5NextCudaWave *wave,uint32_t committed_steps)
{
	assert(IN_CUDA_CALLBACK == 0u && wave != 0 && (FOLD_ANY != 0u || committed_steps == 1u));
	FOLD_LAST = committed_steps;
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

#define VERIFY_PROMPT 60u
#define VERIFY_GENERATE 240u
#define VERIFY_CAPACITY 512u

static uint32_t VerifyTarget(const uint32_t *history,uint32_t length)
{
	if ( length % 13u == 0u )
		return((history[length - 1u] * 7u + length) % 1000u);
	return(history[length - 11u]);
}

static const uint32_t *VERIFY_EXPECTED;

static void verify_plain_graph(void)
{
	SparkGlm5NextExecutionSlot *slot = &state.slots[0];
	assert(VERIFY_EXPECTED != 0 && slot->host_token_ids[0] == VERIFY_EXPECTED[slot->host_positions[0]]);
	slot->host_output_token_ids[0] = VERIFY_EXPECTED[slot->host_positions[0] + 1u];
}

static SparkGlm5NextModuleState RANK_STATE[2];

static SparkStatus verify_rank_drive(uint32_t rank,float inverse_temperature,SparkGlm5NextTpChain *chain)
{
	static const uint32_t reference[16] = {5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u,17u,18u,19u,20u};
	static uint32_t host_tokens[2][8],host_positions[2][8],host_slots[2][8];
	static uint32_t token = 5u,row_slot = 0u;
	static uint64_t row_position = 0u,row_sequence = 3u;
	static SparkModelDriverFrame frame;
	static SparkGlm5NextResidentDecodeStageBatchView batch;
	static SparkRowSampling sampling;
	SparkGlm5NextModuleState *rank_state = &RANK_STATE[rank];
	memset(&frame,0,sizeof(frame));
	memset(&batch,0,sizeof(batch));
	memset(&sampling,0,sizeof(sampling));
	sampling.inverse_temperature = inverse_temperature;
	frame.tokens_per_sequence = 8u;
	frame.request_id = 77u;
	batch.row_count = 1u;
	batch.active_sequence_count = 1u;
	batch.token_ids = &token;
	batch.row_resident_slots = &row_slot;
	batch.row_positions = &row_position;
	batch.row_sequence_ids = &row_sequence;
	batch.row_sampling = &sampling;
	if ( rank_state->verify_draft_function == 0 )
	{
		rank_state->verify_drafter = SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE;
		assert(SparkSpeculationReferenceDraftInitialize(&rank_state->verify_reference,SPARK_SPECULATION_REFERENCE_ORACLE,SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,0u,reference,16u) == SPARK_STATUS_OK);
		rank_state->verify_draft_function = SparkSpeculationReferenceDraftTokens;
		rank_state->verify_draft_context = &rank_state->verify_reference;
		rank_state->verify_depth_cap = calloc(1u,sizeof(uint32_t));
		rank_state->verify_depth_sequence = calloc(1u,sizeof(uint64_t));
		assert(rank_state->verify_depth_cap != 0 && rank_state->verify_depth_sequence != 0);
	}
	rank_state->verify_rows_max = 8u;
	rank_state->max_sequence_positions = 4096u;
	rank_state->decode_split_context_threshold = 64u;
	rank_state->slots[0].host_token_ids = host_tokens[rank];
	rank_state->slots[0].host_positions = host_positions[rank];
	rank_state->slots[0].host_resident_slots = host_slots[rank];
	memset(chain,0,sizeof(*chain));
	chain->state = rank_state;
	chain->slot = &rank_state->slots[0];
	chain->frame = &frame;
	chain->batch = &batch;
	chain->wave_rows = 1u;
	chain->steps = 8u;
	rank_state->completions[0].steps = 8u;
	return(SparkGlm5NextVerifyDriveDraft(rank_state,&frame,&batch,&rank_state->slots[0],chain));
}

static void check_verify_rank_local(void)
{
	SparkGlm5NextTpChain *chains = calloc(2u,sizeof(*chains));
	uint32_t fault,rank;
	assert(chains != 0);
	for (fault=0u; fault<4u; fault++)
	{
		memset(RANK_STATE,0,sizeof(RANK_STATE));
		for (rank=0u; rank<2u; rank++)
		{
			RANK_STATE[rank].graph_path_enabled = 1u;
			RANK_STATE[rank].experts_warm = 1u;
			RANK_STATE[rank].slots[0].verify_captured = 1u;
			atomic_store(&RANK_STATE[rank].terminal_status,SPARK_STATUS_OK);
		}
		if ( fault == 1u )
			RANK_STATE[1].slots[0].graph_failed_rows = UINT64_C(1);
		if ( fault == 2u )
			RANK_STATE[1].slots[0].graph_disabled = 1u;
		if ( fault == 3u )
			RANK_STATE[1].graph_path_enabled = 0u;
		assert(verify_rank_drive(0u,0.0f,&chains[0]) == SPARK_STATUS_OK);
		assert(chains[0].spec_verify == 1u && chains[0].verify_budget == 8u && chains[0].wave_rows == 8u && chains[0].steps == 1u);
		assert(RANK_STATE[0].verify_frames == 1u && RANK_STATE[0].verify_plain_frames == 0u);
		if ( fault == 0u )
		{
			assert(verify_rank_drive(1u,0.0f,&chains[1]) == SPARK_STATUS_OK);
			assert(chains[1].spec_verify == chains[0].spec_verify && chains[1].wave_rows == chains[0].wave_rows && chains[1].steps == chains[0].steps);
			assert(memcmp(chains[1].verify_draft,chains[0].verify_draft,sizeof(chains[0].verify_draft)) == 0);
			assert(atomic_load(&RANK_STATE[1].terminal_status) == SPARK_STATUS_OK);
		}
		else
		{
			assert(verify_rank_drive(1u,0.0f,&chains[1]) == SPARK_STATUS_UNSUPPORTED);
			assert(chains[1].spec_verify == 0u && chains[1].verify_budget == 0u && RANK_STATE[1].verify_frames == 0u && RANK_STATE[1].verify_plain_frames == 0u);
			assert(atomic_load(&RANK_STATE[1].terminal_status) == SPARK_STATUS_UNSUPPORTED);
		}
		for (rank=0u; rank<2u; rank++)
		{
			SparkGlm5NextReleaseDrafter(&RANK_STATE[rank]);
			RANK_STATE[rank].verify_draft_function = 0;
		}
	}
	memset(RANK_STATE,0,sizeof(RANK_STATE));
	for (rank=0u; rank<2u; rank++)
	{
		RANK_STATE[rank].graph_path_enabled = 1u;
		RANK_STATE[rank].experts_warm = 1u;
		RANK_STATE[rank].slots[0].verify_captured = 1u;
		assert(verify_rank_drive(rank,1.0f,&chains[rank]) == SPARK_STATUS_OK);
		assert(chains[rank].spec_verify == 0u && chains[rank].steps == 8u && RANK_STATE[rank].verify_plain_frames == 1u);
		assert(RANK_STATE[rank].verify_frame_class[SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SAMPLED] == 1u);
		RANK_STATE[rank].slots[0].verify_captured = 0u;
		assert(verify_rank_drive(rank,0.0f,&chains[rank]) == SPARK_STATUS_OK && chains[rank].spec_verify == 0u && chains[rank].steps == 8u);
		RANK_STATE[rank].experts_warm = 0u;
		RANK_STATE[rank].graph_path_enabled = rank;
		assert(verify_rank_drive(rank,0.0f,&chains[rank]) == SPARK_STATUS_OK && chains[rank].spec_verify == 0u && chains[rank].steps == 8u);
		assert(RANK_STATE[rank].verify_frame_class[SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_COLD] == 2u && RANK_STATE[rank].verify_plain_frames == 3u);
		assert(atomic_load(&RANK_STATE[rank].terminal_status) == SPARK_STATUS_OK);
		SparkGlm5NextReleaseDrafter(&RANK_STATE[rank]);
		RANK_STATE[rank].verify_draft_function = 0;
	}
	free(chains);
}

static const char *VERIFY_TAP_DUMP;
static uint16_t VERIFY_TAP_DEVICE[8u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION];

static void verify_taps_open(void)
{
	uint32_t row;
	(void)unlink(VERIFY_TAP_DUMP);
	assert(SparkSpeculationTapSetParse("mean:44",SPARK_GLM5_NEXT_MODEL_LAYER_COUNT,SPARK_GLM5_NEXT_MODEL_HC_MULT,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,&state.tap_set) == SPARK_STATUS_OK);
	state.tap_scratch = malloc(state.tap_set.record_bytes);
	state.slots[0].tap_device = VERIFY_TAP_DEVICE;
	state.slots[0].tap_host = malloc((uint64_t)state.execution_row_capacity * state.tap_set.record_bytes);
	assert(state.tap_scratch != 0 && state.slots[0].tap_host != 0 && pthread_mutex_init(&state.tap_lock,0) == 0);
	for (row=0u; row<state.execution_row_capacity; row++)
		memset((uint8_t *)state.slots[0].tap_host + (uint64_t)row * state.tap_set.row_bytes,(int)(row + 1u),state.tap_set.row_bytes);
	state.tap_lock_ready = 1u;
	state.tap_generation = 5u;
	assert(SparkSpeculationTapDumpOpen(&state.tap_dump,VERIFY_TAP_DUMP,&state.tap_set,"glm5_next",5u,0u,UINT64_C(1) << 30) == SPARK_STATUS_OK);
	state.tap_dump_open = 1u;
	state.tap_enabled = 1u;
}

static void verify_taps_check(const uint32_t *expected,uint32_t first,uint32_t minimum_records,uint32_t drafter)
{
	FILE *file;
	uint8_t header[128],record[32],*payload;
	uint64_t records,index,position,serial;
	uint32_t flags,byte;
	SparkGlm5NextReleaseTaps(&state);
	assert(state.tap_enabled == 0u && state.slots[0].tap_host == 0 && state.tap_scratch == 0);
	file = fopen(VERIFY_TAP_DUMP,"rb");
	payload = malloc(SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * 2u);
	assert(file != 0 && payload != 0 && fread(header,1u,sizeof(header),file) == sizeof(header));
	memcpy(&records,header + 104,sizeof(records));
	assert(memcmp(header,"SPTD",4u) == 0 && header[100] == SPARK_SPECULATION_TAP_DUMP_FLAG_CLOSED && records >= minimum_records);
	for (index=0u; index<records; index++)
	{
		assert(fread(record,1u,sizeof(record),file) == sizeof(record) && fread(payload,1u,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * 2u,file) == SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * 2u);
		memcpy(&position,record + 8,sizeof(position));
		memcpy(&flags,record + 20,sizeof(flags));
		memcpy(&serial,record + 24,sizeof(serial));
		assert(record[0] == 44u && position == first + index && serial == index + 1u);
		assert(record[16] == (uint8_t)expected[position] && record[17] == (uint8_t)(expected[position] >> 8));
		assert(flags == SPARK_SPECULATION_TAP_FLAG_VERIFY || flags == SPARK_SPECULATION_TAP_FLAG_DECODE);
		for (byte=0u; byte<SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * 2u; byte++)
			assert(payload[byte] == (flags == SPARK_SPECULATION_TAP_FLAG_DECODE ? 1u : payload[0]));
		assert(flags == SPARK_SPECULATION_TAP_FLAG_DECODE || drafter != SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE || payload[0] == (uint8_t)(1u + (position - first) % 8u));
	}
	assert(fread(record,1u,1u,file) == 0u);
	fclose(file);
	free(payload);
	(void)unlink(VERIFY_TAP_DUMP);
	printf("verify taps drafter=%u: %llu committed rows dumped once each, in order, each verify row from its wave slot\n",drafter,(unsigned long long)records);
}

static void tap_env_clear(void)
{
	static const char *names[] = {"SPARK_GLM5_NEXT_TAPS","SPARK_GLM5_NEXT_TAP_RANK","SPARK_GLM5_NEXT_TAP_DUMP","SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES","SPARK_GLM5_NEXT_TAP_RELAY_LOCAL","SPARK_GLM5_NEXT_TAP_RELAY_PEER","SPARK_GLM5_NEXT_TAP_RELAY_SHADOW","SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US","SPARK_GLM5_NEXT_TAP_RELAY_DEPTH"};
	uint32_t index;
	for (index=0u; index<sizeof(names) / sizeof(names[0]); index++)
		unsetenv(names[index]);
}

static void check_tap_config(void)
{
	memset(&state,0,sizeof(state));
	state.tp_degree = 16u;
	state.tp_rank = 3u;
	state.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	state.owns_embedding = state.owns_final_head = 1u;
	tap_env_clear();
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_OK && state.tap_enabled == 0u);
	setenv("SPARK_GLM5_NEXT_TAP_DUMP","/tmp/x",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	tap_env_clear();
	setenv("SPARK_GLM5_NEXT_TAPS","mean:45",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_SCHEMA_ERROR);
	setenv("SPARK_GLM5_NEXT_TAPS","mean:5,14,24,33,42,44",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	setenv("SPARK_GLM5_NEXT_TAP_DUMP","/tmp/x",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	setenv("SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES","1000000",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_OK && state.tap_enabled == 0u && state.tap_rank == 15u && state.tap_set.tap_count == 6u);
	setenv("SPARK_GLM5_NEXT_TAP_RANK","16",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	setenv("SPARK_GLM5_NEXT_TAP_RANK","3",1);
	state.layer_count = 40u;
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT && state.tap_enabled == 0u);
	state.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	state.mtp_enabled = 1u;
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_UNSUPPORTED && state.tap_enabled == 0u);
	state.mtp_enabled = 0u;
	unsetenv("SPARK_GLM5_NEXT_TAP_DUMP");
	unsetenv("SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES");
	unsetenv("SPARK_GLM5_NEXT_TAP_RANK");
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_LOCAL","127.0.0.1:0",1);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_PEER","127.0.0.1:9",1);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US","300",1);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_DEPTH","7",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_SHADOW","0",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_SHADOW","1",1);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_DEPTH","17",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_DEPTH","7",1);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US","100001",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	setenv("SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US","300",1);
	assert(SparkGlm5NextConfigureTaps(&state) == SPARK_STATUS_OK && state.tap_enabled == 0u && state.tap_relay_depth == 7u);
	tap_env_clear();
	printf("PASS tap configuration: every tap setting needs SPARK_GLM5_NEXT_TAPS, a consumer and explicit bounds; the relay is shadow-only; tap layers must be owned\n");
}

static void check_verify_rounds(uint32_t drafter)
{
	static uint32_t history[VERIFY_CAPACITY],expected[VERIFY_CAPACITY];
	uint32_t host_tokens[8],host_positions[8],host_slots[8],host_output[8],host_run_begin[9],host_run_rows[8],host_run_state[8],row_slot = 1u,token,index,length,produced,more,frames = 0u,rounds = 0u,regime;
	uint64_t row_position,row_sequence = 44u,positions[VERIFY_PROMPT],sequences[VERIFY_PROMPT];
	uint32_t slots[VERIFY_PROMPT];
	SparkRowSampling sampling,host_sampling[8];
	SparkModelDriverFrame frame;
	SparkGlm5NextResidentDecodeStageBatchView batch;
	SparkGlm5NextResidentDecodeStageFrameContext context;
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextAsyncCompletion *async;
	SparkStatus status;
	uint32_t match,fault_frame;
	uint64_t accepted_before,accepted_sum;
	memset(&state,0,sizeof(state));
	memset(&sampling,0,sizeof(sampling));
	memset(&context,0,sizeof(context));
	state.pipeline_slot_count = 1u;
	state.execution_stream = state.slots[0].stream = (void *)(uintptr_t)7u;
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	state.verify_rows_max = 8u;
	state.max_sequence_positions = VERIFY_CAPACITY;
	state.resident_sequence_capacity = 2u;
	state.decode_split_context_threshold = 0u;
	state.graph_path_enabled = 1u;
	state.experts_warm = 1u;
	state.owns_final_head = 1u;
	state.slots[0].verify_captured = 1u;
	state.slots[0].host_token_ids = host_tokens;
	state.slots[0].host_positions = host_positions;
	state.slots[0].host_resident_slots = host_slots;
	state.slots[0].host_output_token_ids = host_output;
	memset(host_sampling,0,sizeof(host_sampling));
	state.slots[0].host_row_sampling = host_sampling;
	state.slots[0].host_run_begin = host_run_begin;
	state.slots[0].host_run_row_indices = host_run_rows;
	state.slots[0].host_run_state_index = host_run_state;
	state.tp_degree = 1u;
	state.execution_row_capacity = 8u;
	memset(VERIFY_MISS_RING,0,sizeof(VERIFY_MISS_RING));
	state.slots[0].miss_ring = VERIFY_MISS_RING;
	state.tp_device_collective.operation_timeout_milli = 50u;
	for (regime=0u; regime<SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT; regime++)
	{
		state.slots[0].graph_exec_rows[regime][0] = (void *)(uintptr_t)9u;
		state.slots[0].graph_bound_rows[regime][0] = VERIFY_CAPACITY;
	}
	state.completions[0].state = &state;
	for (index=0u; index<VERIFY_PROMPT; index++)
	{
		history[index] = (index * 37u + 5u) % 41u;
		positions[index] = index;
		sequences[index] = row_sequence;
		slots[index] = row_slot;
	}
	for (length=VERIFY_PROMPT; length<VERIFY_CAPACITY; length++)
		history[length] = VerifyTarget(history,length);
	memcpy(expected,history,sizeof(history));
	VERIFY_EXPECTED = expected;
	GRAPH_LAUNCH_HOOK = verify_plain_graph;
	GRAPH_ERROR = 0u;
	state.verify_drafter = drafter;
	state.verify_depth_cap = calloc(2u,sizeof(uint32_t));
	state.verify_depth_sequence = calloc(2u,sizeof(uint64_t));
	assert(state.verify_depth_cap != 0 && state.verify_depth_sequence != 0);
	if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP )
	{
		assert(SparkSpeculationLookupDraftInitialize(&state.verify_lookup,2u,VERIFY_CAPACITY,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MIN_MATCH,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MAX_MATCH) == SPARK_STATUS_OK);
		state.verify_draft_function = SparkSpeculationLookupDraftTokens;
		state.verify_draft_context = &state.verify_lookup;
	}
	else
	{
		assert(SparkSpeculationReferenceDraftInitialize(&state.verify_reference,drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE ? SPARK_SPECULATION_REFERENCE_ORACLE : SPARK_SPECULATION_REFERENCE_ADVERSARY,SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,0u,expected,VERIFY_CAPACITY) == SPARK_STATUS_OK);
		state.verify_draft_function = SparkSpeculationReferenceDraftTokens;
		state.verify_draft_context = &state.verify_reference;
	}
	memset(&batch,0,sizeof(batch));
	batch.row_count = VERIFY_PROMPT;
	batch.active_sequence_count = 1u;
	batch.token_ids = history;
	batch.row_resident_slots = slots;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequences;
	assert(SparkGlm5NextVerifyObserveRows(&state,&batch) == SPARK_STATUS_OK);
	if ( VERIFY_TAP_DUMP != 0 )
		verify_taps_open();
	length = VERIFY_PROMPT + 1u;
	DRAIN_STATUS = cudaSuccess;
	FOLD_ANY = 1u;
	while ( length < VERIFY_PROMPT + VERIFY_GENERATE )
	{
		chain = calloc(1u,sizeof(*chain));
		assert(chain != 0);
		async = &state.completions[0];
		memset(async,0,sizeof(*async));
		async->state = &state;
		async->lane_count = 1u;
		async->row_count = 1u;
		async->steps = 8u;
		async->lane_indices[0] = row_slot;
		async->lane_sequence_ids[0] = row_sequence;
		async->lane_next_positions[0] = length;
		token = history[length - 1u];
		row_position = length - 1u;
		memset(&frame,0,sizeof(frame));
		frame.tokens_per_sequence = 8u;
		frame.request_id = 900u + frames;
		batch.row_count = 1u;
		batch.token_ids = &token;
		batch.row_resident_slots = &row_slot;
		batch.row_positions = &row_position;
		batch.row_sequence_ids = &row_sequence;
		batch.row_sampling = &sampling;
		chain->state = &state;
		chain->slot = &state.slots[0];
		chain->frame = &frame;
		chain->context = &context;
		chain->batch = &batch;
		chain->wave_rows = 1u;
		chain->steps = 8u;
		assert(SparkGlm5NextVerifyDriveDraft(&state,&frame,&batch,&state.slots[0],chain) == SPARK_STATUS_OK);
		if ( chain->verify_budget == 0u )
		{
			assert(drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP && chain->steps == 8u && async->steps == 8u && chain->spec_verify == 0u);
			for (index=0u; index<8u; index++)
				host_output[index] = expected[length + index];
			async->burst_token_count = 8u;
			async->lane_next_positions[0] += 7u;
			produced = 8u;
		}
		else
		{
			assert(chain->steps == 1u && async->steps == 1u && chain->spec_verify == 1u);
			fault_frame = drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY && length + 8u >= VERIFY_PROMPT + VERIFY_GENERATE ? 1u : 0u;
			state.slots[0].graph_disabled = fault_frame;
			do
			{
				assert(chain->wave_rows >= 2u && chain->wave_rows <= 8u && host_positions[0] == chain->verify_position && host_slots[0] == row_slot);
				for (index=0u; index<chain->wave_rows; index++)
				{
					uint32_t fed = (uint32_t)chain->verify_position + index;
					assert(host_positions[index] == fed && (index == 0u || host_tokens[index] == chain->verify_draft[index - 1u]));
					memcpy(history,expected,(uint64_t)(fed) * sizeof(uint32_t));
					history[fed] = host_tokens[index];
					host_output[index] = VerifyTarget(history,fed + 1u);
				}
				memcpy(history,expected,sizeof(history));
				match = 0u;
				while ( match + 1u < chain->wave_rows && chain->verify_draft[match] == host_output[match] )
					match++;
				accepted_before = state.verify_accepted;
				FOLD_LAST = 0u;
				status = SparkGlm5NextVerifyCommit(chain,&more);
				assert(status == SPARK_STATUS_OK && FOLD_LAST == match + 1u && state.verify_accepted - accepted_before == match);
				rounds++;
				status = SparkGlm5NextVerifyContinue(chain,&more);
				assert(status == SPARK_STATUS_OK || fault_frame != 0u);
			} while ( status == SPARK_STATUS_OK && more != 0u );
			if ( fault_frame != 0u )
			{
				assert(status == SPARK_STATUS_INTERNAL_ERROR && atomic_load(&state.terminal_status) == SPARK_STATUS_INTERNAL_ERROR && chain->verify_produced == 7u && chain->verify_plain == 0u);
				state.slots[0].graph_disabled = 0u;
				atomic_store(&state.terminal_status,SPARK_STATUS_OK);
				frames++;
				free(chain);
				break;
			}
			assert(SparkGlm5NextVerifyFinish(chain) == SPARK_STATUS_OK);
			produced = chain->verify_produced;
			assert(produced == 8u && async->burst_token_count == produced && async->cache_extra_tokens == produced - 1u);
			assert(async->completion.tokens_per_sequence == produced && async->lane_next_positions[0] == length + produced - 1u);
			if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE )
				assert(chain->verify_rounds == 1u && chain->verify_accepted == 7u && chain->verify_plain == 0u);
			if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY )
				assert(chain->verify_rounds == 7u && chain->verify_accepted == 0u && chain->verify_plain == 1u);
			assert(chain->verify_rounds + chain->verify_accepted + chain->verify_plain == 8u);
		}
		for (index=0u; index<produced; index++)
			assert(host_output[index] == expected[length + index]);
		SparkGlm5NextVerifyObserveOutputs(&state,async,&state.slots[0]);
		assert(async->completion.status == SPARK_STATUS_OK);
		length += produced;
		frames++;
		free(chain);
	}
	assert(state.verify_frames + state.verify_plain_frames == frames && state.verify_rounds == rounds);
	if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP )
		assert(state.verify_frames != 0u && state.verify_accepted != 0u && state.verify_lookup.lengths[row_slot] == length);
	if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY )
		assert(state.verify_accepted == 0u && state.verify_accept_depth[0] == rounds && state.verify_proposed == 7u + rounds - 1u && state.verify_depth_cap[row_slot] == 1u);
	if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE )
		assert(state.verify_proposed == 7u * rounds && state.verify_depth_cap[row_slot] == 7u);
	accepted_sum = 0u;
	for (index=0u; index<SPARK_GLM5_NEXT_VERIFY_ROWS_MAX - 1u; index++)
	{
		accepted_sum += state.verify_position_accepted[index];
		assert(state.verify_position_accepted[index] <= state.verify_position_reached[index]);
		if ( index != 0u )
			assert(state.verify_position_reached[index] <= state.verify_position_accepted[index - 1u]);
		if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE )
			assert(state.verify_position_reached[index] == rounds && state.verify_position_accepted[index] == rounds);
		if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY )
			assert(state.verify_position_accepted[index] == 0u && state.verify_position_reached[index] == (index == 0u ? rounds : 0u));
	}
	assert(accepted_sum == state.verify_accepted && state.verify_position_reached[0] == rounds);
	if ( VERIFY_TAP_DUMP != 0 )
		verify_taps_check(expected,VERIFY_PROMPT,length - 1u - VERIFY_PROMPT,drafter);
	FOLD_ANY = 0u;
	GRAPH_LAUNCH_HOOK = 0;
	VERIFY_EXPECTED = 0;
	SparkGlm5NextReleaseDrafter(&state);
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
}

static void check_verify_mtp_rounds(uint32_t drafter,uint32_t tp)
{
	static uint32_t history[VERIFY_CAPACITY],expected[VERIFY_CAPACITY];
	static uint16_t hc_mean[8u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION],lane_hidden[2u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION],hidden_rows[SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION];
	static uint64_t lane_sequence[2],lane_next[2],maxloc[8];
	uint32_t host_tokens[8],host_positions[8],host_slots[8],host_output[8],host_run_begin[9],host_run_rows[8],host_run_state[8],row_slot = 1u,token,index,length,produced,more,frames = 0u,rounds = 0u,regime,plain_frames = 0u,match;
	uint64_t row_position,row_sequence = 44u,positions[VERIFY_PROMPT],sequences[VERIFY_PROMPT],verify_calls_before;
	uint32_t slots[VERIFY_PROMPT];
	SparkRowSampling sampling,host_sampling[8];
	SparkModelDriverFrame frame;
	SparkGlm5NextResidentDecodeStageBatchView batch;
	SparkGlm5NextResidentDecodeStageFrameContext context;
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextAsyncCompletion *async;
	SparkStatus status;
	memset(&state,0,sizeof(state));
	memset(&sampling,0,sizeof(sampling));
	memset(&context,0,sizeof(context));
	memset(lane_sequence,0,sizeof(lane_sequence));
	memset(lane_next,0,sizeof(lane_next));
	memset(hc_mean,0,sizeof(hc_mean));
	for (index=0u; index<8u; index++)
		hc_mean[(uint64_t)index * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION] = (uint16_t)(1000u + index);
	state.pipeline_slot_count = 1u;
	state.execution_stream = state.slots[0].stream = (void *)(uintptr_t)7u;
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	state.verify_rows_max = 8u;
	state.max_sequence_positions = VERIFY_CAPACITY;
	state.resident_sequence_capacity = 2u;
	state.decode_split_context_threshold = 0u;
	state.graph_path_enabled = 1u;
	state.experts_warm = 1u;
	state.owns_final_head = 1u;
	state.owns_embedding = 1u;
	state.slots[0].verify_captured = 1u;
	state.slots[0].host_token_ids = host_tokens;
	state.slots[0].host_positions = host_positions;
	state.slots[0].host_resident_slots = host_slots;
	state.slots[0].host_output_token_ids = host_output;
	memset(host_sampling,0,sizeof(host_sampling));
	state.slots[0].host_row_sampling = host_sampling;
	state.slots[0].host_run_begin = host_run_begin;
	state.slots[0].host_run_row_indices = host_run_rows;
	state.slots[0].host_run_state_index = host_run_state;
	state.slots[0].hc_mean_bf16 = hc_mean;
	state.slots[0].hidden_bf16 = hidden_rows;
	state.slots[0].head_maxloc_u64 = maxloc;
	state.tp_degree = tp;
	state.execution_row_capacity = 8u;
	memset(VERIFY_MISS_RING,0,sizeof(VERIFY_MISS_RING));
	state.slots[0].miss_ring = VERIFY_MISS_RING;
	state.tp_device_collective.operation_timeout_milli = 50u;
	if ( tp > 1u )
	{
		state.tp_device_collective_initialized = 1u;
		STREAM_ORDERED = 1u;
		ENQUEUE_ROWS = ENQUEUE_SEQUENCES = 1u;
	}
	for (regime=0u; regime<SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT; regime++)
	{
		state.slots[0].graph_exec_rows[regime][0] = (void *)(uintptr_t)9u;
		state.slots[0].graph_bound_rows[regime][0] = VERIFY_CAPACITY;
	}
	state.completions[0].state = &state;
	for (index=0u; index<VERIFY_PROMPT; index++)
	{
		history[index] = (index * 37u + 5u) % 41u;
		positions[index] = index;
		sequences[index] = row_sequence;
		slots[index] = row_slot;
	}
	for (length=VERIFY_PROMPT; length<VERIFY_CAPACITY; length++)
		history[length] = VerifyTarget(history,length);
	memcpy(expected,history,sizeof(history));
	VERIFY_EXPECTED = expected;
	GRAPH_LAUNCH_HOOK = verify_plain_graph;
	GRAPH_ERROR = 0u;
	state.verify_drafter = drafter;
	state.verify_mtp = 1u;
	state.mtp_lane_hidden_bf16 = lane_hidden;
	state.mtp_lane_sequence = lane_sequence;
	state.mtp_lane_next = lane_next;
	state.verify_depth_cap = calloc(2u,sizeof(uint32_t));
	state.verify_depth_sequence = calloc(2u,sizeof(uint64_t));
	assert(state.verify_depth_cap != 0 && state.verify_depth_sequence != 0);
	if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP )
	{
		state.verify_draft_function = SparkGlm5NextMtpDraftTokens;
		state.verify_draft_context = &state;
	}
	else
	{
		assert(SparkSpeculationLookupDraftInitialize(&state.verify_lookup,2u,VERIFY_CAPACITY,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MIN_MATCH,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MAX_MATCH) == SPARK_STATUS_OK);
		assert(SparkSpeculationDrafterMixInitialize(&state.verify_mix,SparkSpeculationLookupDraftTokens,&state.verify_lookup,SparkGlm5NextMtpDraftTokens,&state,SPARK_GLM5_NEXT_VERIFY_MIX_LOOKUP_MIN_TOKENS,2u,7u) == SPARK_STATUS_OK);
		state.verify_draft_function = SparkSpeculationDrafterMixTokens;
		state.verify_draft_context = &state.verify_mix;
	}
	MTP_TRUTH = expected;
	MTP_LANE = row_slot;
	MTP_EXPECT_ROW = UINT32_MAX;
	MTP_DRAFTS = MTP_OPS_CALLS = MTP_DEPTH_TOTAL = 0u;
	verify_calls_before = VERIFY_COUNT;
	memset(&batch,0,sizeof(batch));
	batch.row_count = VERIFY_PROMPT;
	batch.active_sequence_count = 1u;
	batch.token_ids = history;
	batch.row_resident_slots = slots;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequences;
	assert(SparkGlm5NextVerifyObserveRows(&state,&batch) == SPARK_STATUS_OK);
	length = VERIFY_PROMPT + 1u;
	DRAIN_STATUS = cudaSuccess;
	FOLD_ANY = 1u;
	while ( length < VERIFY_PROMPT + VERIFY_GENERATE )
	{
		chain = calloc(1u,sizeof(*chain));
		assert(chain != 0);
		async = &state.completions[0];
		memset(async,0,sizeof(*async));
		async->state = &state;
		async->lane_count = 1u;
		async->row_count = 1u;
		async->steps = 8u;
		async->lane_indices[0] = row_slot;
		async->lane_sequence_ids[0] = row_sequence;
		async->lane_next_positions[0] = length;
		token = history[length - 1u];
		row_position = length - 1u;
		memset(&frame,0,sizeof(frame));
		frame.tokens_per_sequence = 8u;
		frame.request_id = 900u + frames;
		batch.row_count = 1u;
		batch.token_ids = &token;
		batch.row_resident_slots = &row_slot;
		batch.row_positions = &row_position;
		batch.row_sequence_ids = &row_sequence;
		batch.row_sampling = &sampling;
		chain->state = &state;
		chain->slot = &state.slots[0];
		chain->frame = &frame;
		chain->context = &context;
		chain->batch = &batch;
		chain->wave_rows = 1u;
		chain->steps = 8u;
		MTP_EXPECT_ROW = UINT32_MAX;
		assert(SparkGlm5NextVerifyDriveDraft(&state,&frame,&batch,&state.slots[0],chain) == SPARK_STATUS_OK);
		if ( chain->verify_budget == 0u )
		{
			assert(frames == 0u && chain->steps == 8u && chain->spec_verify == 0u);
			for (index=0u; index<8u; index++)
				host_output[index] = expected[length + index];
			async->burst_token_count = 8u;
			async->lane_next_positions[0] += 7u;
			produced = 8u;
			assert(SparkGlm5NextMtpTapFrame(chain) == SPARK_STATUS_OK);
			assert(lane_next[row_slot] == length + 7u && lane_sequence[row_slot] == row_sequence && lane_hidden[(uint64_t)row_slot * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION] == 1000u);
			plain_frames++;
		}
		else
		{
			assert(chain->steps == 1u && async->steps == 1u && chain->spec_verify == 1u);
			do
			{
				assert(chain->wave_rows >= 2u && chain->wave_rows <= 8u && host_positions[0] == chain->verify_position && host_slots[0] == row_slot);
				for (index=0u; index<chain->wave_rows; index++)
				{
					uint32_t fed = (uint32_t)chain->verify_position + index;
					assert(host_positions[index] == fed && (index == 0u || host_tokens[index] == chain->verify_draft[index - 1u]));
					memcpy(history,expected,(uint64_t)(fed) * sizeof(uint32_t));
					history[fed] = host_tokens[index];
					host_output[index] = VerifyTarget(history,fed + 1u);
				}
				memcpy(history,expected,sizeof(history));
				match = 0u;
				while ( match + 1u < chain->wave_rows && chain->verify_draft[match] == host_output[match] )
					match++;
				FOLD_LAST = 0u;
				MTP_EXPECT_ROW = match;
				status = SparkGlm5NextVerifyCommit(chain,&more);
				assert(status == SPARK_STATUS_OK && FOLD_LAST == match + 1u);
				assert(lane_next[row_slot] == chain->verify_position && lane_sequence[row_slot] == row_sequence);
				rounds++;
				MTP_EXPECT_ROW = 0u;
				status = SparkGlm5NextVerifyContinue(chain,&more);
				assert(status == SPARK_STATUS_OK && lane_next[row_slot] == chain->verify_position);
			} while ( more != 0u );
			assert(SparkGlm5NextVerifyFinish(chain) == SPARK_STATUS_OK);
			produced = chain->verify_produced;
			assert(produced == 8u && async->burst_token_count == produced && async->lane_next_positions[0] == length + produced - 1u);
			assert(SparkGlm5NextMtpTapFrame(chain) == SPARK_STATUS_OK && lane_next[row_slot] == length + produced - 1u);
		}
		for (index=0u; index<produced; index++)
			assert(host_output[index] == expected[length + index]);
		SparkGlm5NextVerifyObserveOutputs(&state,async,&state.slots[0]);
		assert(async->completion.status == SPARK_STATUS_OK);
		length += produced;
		frames++;
		free(chain);
	}
	assert(state.verify_frames + state.verify_plain_frames == frames && state.verify_rounds == rounds && state.verify_frames != 0u);
	assert(state.mtp_drafts == MTP_DRAFTS && state.mtp_draft_tokens <= MTP_DEPTH_TOTAL && state.mtp_taps != 0u);
	if ( drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP )
	{
		assert(plain_frames == 1u && state.mtp_cold == 1u && state.mtp_truncated != 0u);
		assert(state.verify_accepted != 0u && state.verify_accepted < state.verify_proposed && state.mtp_draft_tokens >= state.verify_proposed);
	}
	else
	{
		SparkSpeculationDrafterMixCounters *lookup = &state.verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY],*mtp = &state.verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK];
		assert(plain_frames <= 1u && state.mtp_cold <= 1u);
		assert(lookup->rounds != 0u && mtp->rounds != 0u && lookup->rounds + mtp->rounds == rounds);
		assert(lookup->accepted + mtp->accepted == state.verify_accepted && lookup->proposed + mtp->proposed == state.verify_proposed);
		assert(mtp->rounds <= state.mtp_drafts && lookup->accepted * 2u > lookup->proposed);
	}
	if ( tp > 1u )
		assert(MTP_OPS_CALLS == 4u * MTP_DEPTH_TOTAL && VERIFY_COUNT - verify_calls_before >= state.mtp_drafts);
	else
		assert(MTP_OPS_CALLS == 0u);
	printf("verify mtp drafter=%u tp=%u: %u frames, %u rounds, proposed %llu accepted %llu, mtp drafts %llu cold %llu truncated %llu taps %llu\n",drafter,tp,frames,rounds,
		(unsigned long long)state.verify_proposed,(unsigned long long)state.verify_accepted,(unsigned long long)state.mtp_drafts,(unsigned long long)state.mtp_cold,(unsigned long long)state.mtp_truncated,(unsigned long long)state.mtp_taps);
	FOLD_ANY = 0u;
	GRAPH_LAUNCH_HOOK = 0;
	VERIFY_EXPECTED = 0;
	MTP_TRUTH = 0;
	MTP_EXPECT_ROW = UINT32_MAX;
	STREAM_ORDERED = 0u;
	ENQUEUE_ROWS = ENQUEUE_SEQUENCES = 2u;
	state.mtp_lane_sequence = 0;
	state.mtp_lane_next = 0;
	SparkGlm5NextReleaseDrafter(&state);
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
}

static void check_verify_mtp_tap_frames(void)
{
	static uint16_t hc_mean[8u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION],lane_hidden[2u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION];
	static uint64_t lane_sequence[2],lane_next[2];
	uint64_t positions[4] = {20u,21u,22u,23u},sequences[4] = {5u,5u,5u,5u};
	uint32_t slots[4] = {1u,1u,1u,1u},tokens[4] = {1u,2u,3u,4u},index;
	SparkModelDriverFrame frame;
	SparkGlm5NextResidentDecodeStageBatchView batch;
	SparkGlm5NextTpChain chain;
	memset(&state,0,sizeof(state));
	memset(&frame,0,sizeof(frame));
	memset(&batch,0,sizeof(batch));
	memset(&chain,0,sizeof(chain));
	memset(lane_sequence,0,sizeof(lane_sequence));
	memset(lane_next,0,sizeof(lane_next));
	for (index=0u; index<8u; index++)
		hc_mean[(uint64_t)index * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION] = (uint16_t)(1000u + index);
	state.resident_sequence_capacity = 2u;
	state.execution_row_capacity = 8u;
	state.owns_final_head = 1u;
	state.slots[0].hc_mean_bf16 = hc_mean;
	state.mtp_lane_hidden_bf16 = lane_hidden;
	state.mtp_lane_sequence = lane_sequence;
	state.mtp_lane_next = lane_next;
	chain.state = &state;
	chain.slot = &state.slots[0];
	chain.frame = &frame;
	chain.batch = &batch;
	batch.active_sequence_count = 1u;
	batch.row_count = 4u;
	batch.token_ids = tokens;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequences;
	batch.row_resident_slots = slots;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	chain.wave_rows = 4u;
	chain.steps = 1u;
	assert(SparkGlm5NextMtpTapFrame(&chain) == SPARK_STATUS_OK && state.mtp_taps == 0u);
	state.verify_mtp = 1u;
	assert(SparkGlm5NextMtpTapFrame(&chain) == SPARK_STATUS_OK && state.mtp_taps == 1u);
	assert(lane_next[1] == 24u && lane_sequence[1] == 5u && lane_hidden[SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION] == 1003u);
	chain.wave_rows = 2u;
	assert(SparkGlm5NextMtpTapFrame(&chain) == SPARK_STATUS_OK && state.mtp_taps == 1u);
	frame.flags = 0u;
	batch.row_count = 1u;
	chain.wave_rows = 1u;
	chain.steps = 8u;
	assert(SparkGlm5NextMtpTapFrame(&chain) == SPARK_STATUS_OK && state.mtp_taps == 2u && lane_next[1] == 28u && lane_hidden[SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION] == 1000u);
	batch.active_sequence_count = 2u;
	assert(SparkGlm5NextMtpTapFrame(&chain) == SPARK_STATUS_OK && state.mtp_taps == 2u);
	batch.active_sequence_count = 1u;
	chain.verify_budget = 8u;
	assert(SparkGlm5NextMtpTapFrame(&chain) == SPARK_STATUS_OK && state.mtp_taps == 2u);
	chain.verify_budget = 0u;
	state.owns_final_head = 0u;
	assert(SparkGlm5NextMtpTapFrame(&chain) == SPARK_STATUS_OK && state.mtp_taps == 2u);
	state.owns_final_head = 1u;
	assert(SparkGlm5NextMtpTap(&state,&state.slots[0],8u,1u,5u,30u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm5NextMtpTap(&state,&state.slots[0],0u,2u,5u,30u) == SPARK_STATUS_INVALID_ARGUMENT);
	memset(&state,0,sizeof(state));
}

static void check_verify_mtp_draft_guards(void)
{
	static uint64_t lane_sequence[2],lane_next[2];
	SparkSpeculationPolicyDraftRequest request;
	SparkSpeculationPolicyDraftResult result;
	SparkGlm5NextTpChain chain;
	memset(&state,0,sizeof(state));
	memset(&chain,0,sizeof(chain));
	memset(&request,0,sizeof(request));
	memset(&result,0,sizeof(result));
	state.resident_sequence_capacity = 2u;
	state.mtp_lane_sequence = lane_sequence;
	state.mtp_lane_next = lane_next;
	state.tp_degree = 16u;
	lane_sequence[0] = 9u;
	lane_next[0] = 40u;
	request.requested_token_count = 7u;
	request.sequence_id = 9u;
	request.sequence_position = 40u;
	assert(SparkGlm5NextMtpDraftTokens(&state,&request,&result) == SPARK_STATUS_INVALID_ARGUMENT);
	state.mtp_chain = &chain;
	chain.state = &state;
	chain.slot = &state.slots[0];
	request.sequence_position = 41u;
	assert(SparkGlm5NextMtpDraftTokens(&state,&request,&result) == SPARK_STATUS_NOT_FOUND && state.mtp_cold == 1u && result.token_count == 0u);
	request.sequence_position = 40u;
	request.sequence_id = 10u;
	assert(SparkGlm5NextMtpDraftTokens(&state,&request,&result) == SPARK_STATUS_NOT_FOUND && state.mtp_cold == 2u);
	request.sequence_id = 9u;
	STREAM_ORDERED = 0u;
	state.tp_device_collective_initialized = 1u;
	assert(SparkGlm5NextMtpDraftTokens(&state,&request,&result) == SPARK_STATUS_UNSUPPORTED);
	request.active_sequence_index = 2u;
	assert(SparkGlm5NextMtpDraftTokens(&state,&request,&result) == SPARK_STATUS_INVALID_ARGUMENT);
	memset(&state,0,sizeof(state));
}

static void check_mtp_pack(void)
{
	static SparkGlm5NextStagePackEntry entries[64],saved[64];
	SparkGlm5NextStagePackHeader header,good;
	SparkGlm5NextStagePackTensorShape shape;
	char directory[] = "/tmp/g5n_mtp_pack_XXXXXX",path[512];
	uint64_t offset,device_bytes,expected_bytes = 0u;
	uint32_t kind,count = 0u,index;
	FILE *file;
	memset(&state,0,sizeof(state));
	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	state.tp_degree = 16u;
	state.tp_rank = 3u;
	state.expert_weight_codec = GLM5_NEXT_EXPERT_WEIGHT_CODEC;
	header.magic = SPARK_GLM5_NEXT_STAGEPACK_MAGIC;
	header.format_version = SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES;
	header.codec_abi_version = SPARK_WEIGHT_CODEC_ABI_VERSION;
	header.flags = SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP;
	header.stage_count = 1u;
	header.first_layer_index = SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX;
	header.layer_count = 1u;
	header.total_layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	header.hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	header.vocab_count = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	header.routed_expert_count = SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT;
	header.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	header.expert_weight_codec = GLM5_NEXT_EXPERT_WEIGHT_CODEC;
	header.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	header.reserved0 = 16u;
	header.reserved1 = 3u;
	header.directory_offset = 512u;
	for (kind=0u; kind<SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		if ( SparkGlm5NextStagePackExpectedShape(kind,SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX,GLM5_NEXT_EXPERT_WEIGHT_CODEC,16u,&shape) == 0 )
			count++;
	assert(count == 27u);
	offset = (header.directory_offset + (uint64_t)count * SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES + 255u) / 256u * 256u;
	for (kind=0u,index=0u; kind<SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		if ( SparkGlm5NextStagePackExpectedShape(kind,SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX,GLM5_NEXT_EXPERT_WEIGHT_CODEC,16u,&shape) != 0 )
			continue;
		entries[index].tensor_kind = kind;
		entries[index].layer_index = SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX;
		entries[index].payload_type = shape.payload_type;
		entries[index].weight_codec = shape.weight_codec;
		entries[index].scale_encoding = shape.scale_encoding;
		entries[index].group_count = shape.group_count;
		entries[index].rows = shape.rows;
		entries[index].columns = shape.columns;
		entries[index].payload_offset = offset;
		entries[index].payload_bytes = SparkGlm5NextStagePackExpectedPayloadBytes(&shape);
		offset = (offset + entries[index].payload_bytes + 255u) / 256u * 256u;
		expected_bytes += (entries[index].payload_bytes + 255u) / 256u * 256u;
		entries[index].scale_bytes = SparkGlm5NextStagePackExpectedScaleBytes(&shape);
		if ( entries[index].scale_bytes != 0u )
		{
			entries[index].scale_offset = offset;
			offset = (offset + entries[index].scale_bytes + 255u) / 256u * 256u;
			expected_bytes += (entries[index].scale_bytes + 255u) / 256u * 256u;
		}
		index++;
	}
	header.tensor_count = count;
	header.file_bytes = offset;
	good = header;
	memcpy(saved,entries,sizeof(entries));
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_OK && device_bytes == expected_bytes);
	assert(expected_bytes > UINT64_C(600000000) && expected_bytes < UINT64_C(640000000));
	header.flags = 0u;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_SCHEMA_ERROR && device_bytes == 0u);
	header = good;
	header.reserved1 = 4u;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header = good;
	header.first_layer_index = 44u;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header = good;
	header.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_TARGET_MISMATCH);
	header = good;
	header.magic++;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_ABI_MISMATCH);
	header = good;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset + 1u,&device_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	entries[3].layer_index = 44u;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	memcpy(entries,saved,sizeof(entries));
	entries[5] = entries[4];
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_DUPLICATE);
	memcpy(entries,saved,sizeof(entries));
	header.tensor_count = count - 1u;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header = good;
	entries[2].rows++;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	memcpy(entries,saved,sizeof(entries));
	entries[1].payload_offset = entries[0].payload_offset;
	assert(SparkGlm5NextMtpPackCheck(&state,&header,entries,offset,&device_bytes) != SPARK_STATUS_OK);
	memcpy(entries,saved,sizeof(entries));
	state.verify_mtp = 1u;
	state.ledger.module_tag = SPARK_GLM5_NEXT_MODULE_TAG;
	assert(unsetenv(SPARK_GLM5_NEXT_VERIFY_MTP_DIR_ENV) == 0);
	assert(SparkGlm5NextMtpPackOpen(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(mkdtemp(directory) != 0 && setenv(SPARK_GLM5_NEXT_VERIFY_MTP_DIR_ENV,directory,1) == 0);
	assert(SparkGlm5NextMtpPackOpen(&state) == SPARK_STATUS_NOT_FOUND && state.mtp_sidecar == 0u);
	assert(SparkGlm5NextMtpPackPath(directory,16u,3u,path,(uint32_t)sizeof(path)) == SPARK_STATUS_OK);
	file = fopen(path,"wb");
	assert(file != 0);
	assert(fwrite(&good,sizeof(good),1u,file) == 1u && fseek(file,(long)good.directory_offset,SEEK_SET) == 0);
	assert(fwrite(entries,sizeof(entries[0]),count,file) == count && fclose(file) == 0);
	assert(truncate(path,(off_t)offset) == 0);
	assert(SparkGlm5NextMtpPackOpen(&state) == SPARK_STATUS_OK && state.mtp_sidecar == 1u && state.mtp_sidecar_bytes == expected_bytes);
	assert(state.mtp_layer.expert_up_gate_payload != 0 && state.mtp_layer.expert_up_gate_scale != 0 && state.mtp_layer.expert_down_payload != 0 && state.mtp_layer.q_b_bf16 != 0 && state.mtp_layer.router_bf16 != 0);
	assert(state.mtp_eh_proj_bf16 != 0 && state.mtp_enorm_bf16 != 0 && state.mtp_hnorm_bf16 != 0 && state.mtp_shared_norm_bf16 != 0);
	assert((const uint8_t *)state.mtp_eh_proj_bf16 >= (const uint8_t *)state.mtp_layer.attn_norm_bf16);
	SparkStageModuleLedgerRelease(&state.ledger);
	state.tp_rank = 4u;
	state.mtp_sidecar = 0u;
	assert(SparkGlm5NextMtpPackOpen(&state) == SPARK_STATUS_NOT_FOUND);
	assert(unlink(path) == 0 && rmdir(directory) == 0 && unsetenv(SPARK_GLM5_NEXT_VERIFY_MTP_DIR_ENV) == 0);
	memset(&state,0,sizeof(state));
}

static SparkWeightdLazyPack ADMISSION_PACK;

static void check_verify_mtp_admission(void)
{
	uint32_t kind;
	const char *path;
	char buffer[64];
	assert(SparkGlm5NextVerifyDrafterParse("mtp",8u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP && path == 0);
	assert(SparkGlm5NextVerifyDrafterParse("mtp+lookup",8u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP);
	assert(SparkGlm5NextVerifyDrafterParse("mtp",0u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm5NextVerifyDrafterParse("lookup+mtp",8u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm5NextVerifyDrafterUsesMtp(SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP) == 1u && SparkGlm5NextVerifyDrafterUsesMtp(SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP) == 0u);
	assert(SparkGlm5NextVerifyDrafterUsesLookup(SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP) == 1u && SparkGlm5NextVerifyDrafterUsesLookup(SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP) == 0u);
	assert(SparkGlm5NextMtpPackPath("/d",16u,15u,buffer,(uint32_t)sizeof(buffer)) == SPARK_STATUS_OK && strcmp(buffer,"/d/glm5_next_mtp.tp16.rank15.g5nsp") == 0);
	assert(SparkGlm5NextMtpPackPath("/d",16u,16u,buffer,(uint32_t)sizeof(buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm5NextMtpPackPath("",16u,1u,buffer,(uint32_t)sizeof(buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm5NextMtpPackPath("/a/very/long/directory/name",16u,1u,buffer,16u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&state,0,sizeof(state));
	state.verify_rows_max = SPARK_GLM5_NEXT_VERIFY_ROWS_MAX;
	state.verify_mtp = 1u;
	state.verify_drafter = SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP;
	state.tp_degree = 16u;
	state.owns_embedding = 1u;
	state.owns_final_head = 1u;
	state.lazy_pack = &ADMISSION_PACK;
	state.execution_row_capacity = SPARK_GLM5_NEXT_VERIFY_ROWS_MAX;
	state.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	state.expert_pin_key_count = SparkGlm5NextPinnedExpected(&state);
	state.mtp_sidecar = 1u;
	assert(state.expert_pin_key_count != 0u && SparkGlm5NextValidateSpeculation(&state) == SPARK_STATUS_OK);
	state.mtp_sidecar = 0u;
	assert(SparkGlm5NextValidateSpeculation(&state) == SPARK_STATUS_UNSUPPORTED);
	state.mtp_sidecar = 1u;
	state.pack_has_mtp = 1u;
	assert(SparkGlm5NextValidateSpeculation(&state) == SPARK_STATUS_UNSUPPORTED);
	state.pack_has_mtp = 0u;
	state.verify_drafter = SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP;
	assert(SparkGlm5NextValidateSpeculation(&state) == SPARK_STATUS_OK);
	state.mtp_sidecar = 0u;
	assert(SparkGlm5NextValidateSpeculation(&state) == SPARK_STATUS_UNSUPPORTED);
	state.verify_mtp = 0u;
	state.verify_drafter = SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP;
	assert(SparkGlm5NextValidateSpeculation(&state) == SPARK_STATUS_OK);
	memset(&state,0,sizeof(state));
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

static int32_t check_cache_transactions(SparkStatus completion_status,SparkStatus verify_status,SparkStatus end_status,cudaError_t drain_status,uint32_t steps)
{
	SparkStatus expected = completion_status != SPARK_STATUS_OK ? completion_status : verify_status;
	uint32_t outputs[16] = {0},chain_tokens[16] = {0};
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
	state.slots[0].miss_ring = (uint32_t *)EPOCH_WORDS;
	memset(EPOCH_WORDS,0,sizeof(EPOCH_WORDS));
	EPOCH_WORDS[SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64] = 42u;
	state.completions[0].completion.status = completion_status;
	state.completions[0].completion_function = observe_completion;
	state.completions[0].steps = steps;
	state.completions[0].row_count = 2u;
	state.slots[0].host_kv_access_error = errors;
	state.slots[0].host_output_token_ids = outputs;
	state.slots[0].host_chain_token_ids = chain_tokens;
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
	if ( (expected == SPARK_STATUS_OK && (fixture.pages.cache.sequences[0].next_token_position != steps || fixture.pages.cache.sequences[1].next_token_position != steps || state.completions[0].burst_token_count != (steps > 1u ? 2u * steps : 0u))) )
		return(-31);
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
	uint32_t shard_divisor = state.kv_shard != 0u ? state.tp_degree : 1u;
	char shard_fingerprint[96];
	(void)snprintf(shard_fingerprint,sizeof(shard_fingerprint),"kv-bf16-index-packed-layer-major-gather-v1-shard%ur%u",state.tp_degree,state.tp_rank);
	assert(strcmp(table->cache_layout_fingerprint,state.kv_shard != 0u ? shard_fingerprint : "kv-bf16-index-packed-layer-major-gather-v1") == 0);
	assert(table->arena_configuration.value_block_stride_bytes * shard_divisor == (uint64_t)state.index_layer_count * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
	assert(table->arena_configuration.layer_count == state.kv_layer_count && table->capacity_request.layer_count == state.kv_layer_count);
	assert(table->arena_configuration.block_token_count == SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT && table->arena_configuration.kv_head_count == 1u);
	assert(table->arena_configuration.head_dim * shard_divisor == SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION && table->arena_configuration.bytes_per_scalar == 2u);
	assert(table->page_store_config.page_bytes == (uint64_t)SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT * state.kv_layer_count * (SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION / shard_divisor) * 2u + table->arena_configuration.value_block_stride_bytes);
	if ( ARENA_KEY_STRIDE_SKEW != 0u )
	{
		arena->key_block_stride_bytes = table->page_store_config.page_bytes - table->arena_configuration.value_block_stride_bytes + ARENA_KEY_STRIDE_SKEW;
		arena->value_block_stride_bytes = table->arena_configuration.value_block_stride_bytes;
		arena->logical_block_count = table->arena_configuration.logical_block_count;
		arena->resident_block_capacity = table->arena_configuration.resident_block_capacity;
		return(SPARK_STATUS_OK);
	}
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
	assert(atomic_load(&state.kda_restore[SPARK_GLM5_NEXT_KDA_COUNT]) == 0u);
	assert(SparkGlm5NextRestoreCacheLanes(&state,&completion) == SPARK_STATUS_OK);
	assert(atomic_load(&state.kda_restore[SPARK_GLM5_NEXT_KDA_COUNT]) == 1u && atomic_load(&state.kda_restore[SPARK_GLM5_NEXT_KDA_BYTES]) == sizeof(restored));
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
	assert(atomic_load(&state.kda_capture[SPARK_GLM5_NEXT_KDA_COUNT]) == (fail_copy != 0u ? 0u : 1u) && atomic_load(&state.kda_capture[SPARK_GLM5_NEXT_KDA_BYTES]) == (fail_copy != 0u ? 0u : sizeof(expected)));
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
	char lane_limit[16];
	const char *invalid_lanes[] = {"", "-1", lane_limit, "4294967295", "1x", " 1", "+1"};
	(void)snprintf(lane_limit,sizeof(lane_limit),"%u",(unsigned)SPARK_WEIGHTD_MESH_MAX_LANES);
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

static void check_l2_prefetch_environment(void)
{
	const char *invalid_bytes[] = {"", "0", "16", "65537", "12648448", "-65536", "4194304x", " 4194304", "+4194304", "4294967296", "18446744073709551616"};
	const char *invalid_blocks[] = {"", "0", "193", "x", "-1", "16 "};
	assert(setenv("SPARK_GLM5_NEXT_GRAPH_PATH","1",1) == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_PREFETCH") == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_L2_PREFETCH") == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES") == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS") == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.l2_prefetch == 1u);
	assert(state.l2_prefetch_shape.bytes == 12582912u && state.l2_prefetch_shape.blocks == 48u);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH","2",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH","0",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.l2_prefetch == 0u);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES","4194304",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(unsetenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES") == 0);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS","16",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH","1",1) == 0);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES","4194304",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.l2_prefetch == 1u);
	assert(state.l2_prefetch_shape.bytes == 4194304u && state.l2_prefetch_shape.blocks == 16u);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES","65536",1) == 0);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS","1",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.l2_prefetch_shape.bytes == 65536u && state.l2_prefetch_shape.blocks == 1u);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES","12582912",1) == 0);
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS","192",1) == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.l2_prefetch_shape.bytes == 12582912u && state.l2_prefetch_shape.blocks == 192u);
	for (uint32_t index = 0u; index < sizeof(invalid_bytes) / sizeof(invalid_bytes[0]); index++)
	{
		assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES",invalid_bytes[index],1) == 0);
		assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	}
	assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES","4194304",1) == 0);
	for (uint32_t index = 0u; index < sizeof(invalid_blocks) / sizeof(invalid_blocks[0]); index++)
	{
		assert(setenv("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS",invalid_blocks[index],1) == 0);
		assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_INVALID_ARGUMENT);
	}
	assert(unsetenv("SPARK_GLM5_NEXT_L2_PREFETCH") == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES") == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS") == 0);
	assert(SparkGlm5NextConfigureExecution(&state) == SPARK_STATUS_OK && state.l2_prefetch == 1u && state.l2_prefetch_shape.bytes == 12582912u && state.l2_prefetch_shape.blocks == 48u);
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

static void small_kv_state(uint32_t kv_layers,uint32_t pages)
{
	static uint8_t index_pool[3u * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u];
	memset(&state,0,sizeof(state));
	state.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	state.kv_layer_count = kv_layers;
	state.index_layer_count = 1u;
	state.index_cache = index_pool;
	state.page_count = pages;
	state.physical_page_count = pages;
	state.pages_per_sequence = pages;
	state.resident_sequence_capacity = 1u;
	state.kv_layer_stride_bytes = (uint64_t)pages * SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION * 2u;
	state.kv_backing_directory = "/unused-host-fixture";
}

static void check_small_kv(void)
{
	uint32_t pages,kv_layers;
	for (kv_layers=1u; kv_layers<=2u; kv_layers++)
		for (pages=1u; pages<=3u; pages++)
		{
			small_kv_state(kv_layers,pages);
			assert(SparkGlm5NextKvInitialize(&state) == SPARK_STATUS_PENDING);
			free_cache_fixture();
		}
	small_kv_state(0u,2u);
	assert(SparkGlm5NextKvInitialize(&state) == SPARK_STATUS_CAPACITY_EXCEEDED);
	free_cache_fixture();
	small_kv_state(2u,2u);
	ARENA_KEY_STRIDE_SKEW = 256u;
	assert(SparkGlm5NextKvInitialize(&state) == SPARK_STATUS_INTERNAL_ERROR);
	ARENA_KEY_STRIDE_SKEW = 0u;
	free_cache_fixture();
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
	L2_CALLS = L2_SITES = L2_BYTES = L2_BLOCKS = 0u;
	chain.tp_op_index = chain.tp_hc_op_index = 0u;
	WALK_LENGTH = 0u;
	assert(SparkGlm5NextWalkChain(&chain,&layer) == 0u && layer == 5u && L2_CALLS == 0u && state.l2_prefetch_rounds == 0u);
	state.l2_prefetch = 1u;
	state.l2_prefetch_shape.bytes = 4194304u;
	state.l2_prefetch_shape.blocks = 16u;
	chain.wave.tp_degree = 16u;
	chain.tp_op_index = chain.tp_hc_op_index = 0u;
	WALK_LENGTH = 0u;
	assert(SparkGlm5NextWalkChain(&chain,&layer) == 0u && layer == 5u && strcmp(WALK_TRACE,"Bh" "ArPMrQ" "SgTrPMrQ" "ArPMrQ" "ArPRErQ" "ArPRErQ" "HxU") == 0);
	assert(L2_CALLS == 11u && state.l2_prefetch_rounds == L2_CALLS && L2_SITES == 7u && L2_BYTES == 4194304u && L2_BLOCKS == 16u);
	state.l2_prefetch = 0u;
	state.l2_prefetch_rounds = 0u;
	chain.wave.tp_degree = 0u;
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
static uint32_t LINEAR_TOKENS[2],LINEAR_HOST_TOKENS[16],LINEAR_CHAIN[16],LINEAR_INPUT[2],LINEAR_SLOTS[2] = {0u,1u},LINEAR_POSITIONS[2],LINEAR_BEGIN[3],LINEAR_INDICES[2],LINEAR_RUNS[2],LINEAR_ERRORS[SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT],LINEAR_HOST_ERRORS[SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT];
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
	slot->host_chain_token_ids = LINEAR_CHAIN;
	slot->host_token_ids = LINEAR_INPUT;
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
	LINEAR_POSITIONS[0] = 3u;
	LINEAR_POSITIONS[1] = 5u;
	LINEAR_INPUT[0] = 1u;
	LINEAR_INPUT[1] = 2u;
	UNPACK_COUNT = 0u;
	memset(LINEAR_HOST_TOKENS,0,sizeof(LINEAR_HOST_TOKENS));
	memset(LINEAR_CHAIN,0,sizeof(LINEAR_CHAIN));
	state.completions[0].steps = 1u;
	pin_fixture_experts();
	assert(pthread_mutex_init(&state.completion_queue_lock,0) == 0);
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	LINEAR_FRAME.request_id = 5u;
	LINEAR_BATCH.active_sequence_count = LINEAR_BATCH.row_count = 2u;
	LINEAR_BATCH.row_resident_slots = LINEAR_SLOTS;
	LINEAR_CONTEXT.batch = &LINEAR_BATCH;
	chain->state = &state;chain->slot = &state.slots[0];chain->frame = &LINEAR_FRAME;chain->batch = &LINEAR_BATCH;chain->context = &LINEAR_CONTEXT;
	chain->wave_rows = 2u;chain->active = 1u;chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;chain->steps = 1u;
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

#define LINEAR_WALK "Bh" "ArPMrQ" "SgTrPMrQ" "ArPMrQ" "ArPRErQ" "ArPRErQ" "HxU"
#define SHARD_WALK "Bh" "AkVaWrPMrQ" "SgTkVaWrPMrQ" "ArPMrQ" "ArPRErQ" "ArPRErQ" "HxU"

static uint16_t SHARD_QUERY[8],SHARD_GATHERED[8];
static float SHARD_PARTIALS[8],SHARD_RECEIVED[8];

static void kv_shard_slot_fixture(SparkGlm5NextExecutionSlot *slot)
{
	slot->query_latent_bf16 = SHARD_QUERY;
	slot->kv_shard_query_gathered_bf16 = SHARD_GATHERED;
	slot->kv_shard_partials_f32 = SHARD_PARTIALS;
	slot->kv_shard_partials_received_f32 = SHARD_RECEIVED;
	SHARD_QUERY_SEQUENCES = SparkGlm5NextKvShardQuerySequences(2u,16u);
	SHARD_WIDE = SparkGlm5NextKvShardPartialWide(2u,16u,state.execution_row_capacity);
	SHARD_PARTIAL_SEQUENCES = SparkGlm5NextKvShardPartialSequences(2u,16u,state.execution_row_capacity);
	assert(SHARD_QUERY_SEQUENCES == 1u && SHARD_PARTIAL_SEQUENCES == (SHARD_WIDE != 0u ? 1u : 3u));
}

static void check_kv_shard_walk(void)
{
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextAsyncCompletion *async;
	uint32_t advance;
	chain = linear_chain_fixture();
	async = &state.completions[0];
	state.kv_shard = 1u;
	kv_shard_slot_fixture(&state.slots[0]);
	WALK_SHARD_MASK = 3u;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(SHARD_WIDE == 0u && strcmp(WALK_TRACE,SHARD_WALK) == 0 && async->linear == 1u && async->completion.status == SPARK_STATUS_OK);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	async = &state.completions[0];
	state.kv_shard = 1u;
	state.execution_row_capacity = 2u;
	kv_shard_slot_fixture(&state.slots[0]);
	WALK_SHARD_MASK = 3u;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(SHARD_WIDE == 1u && strcmp(WALK_TRACE,SHARD_WALK) == 0 && async->linear == 1u && async->completion.status == SPARK_STATUS_OK);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	async = &state.completions[0];
	state.kv_shard = 1u;
	kv_shard_slot_fixture(&state.slots[0]);
	WALK_SHARD_MASK = 3u;
	WALK_FAIL_CODE = 'V';
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bh" "AkV") == 0 && async->completion.status == SPARK_STATUS_INTERNAL_ERROR);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	state.kv_shard = 1u;
	kv_shard_slot_fixture(&state.slots[0]);
	WALK_SHARD_MASK = 3u;
	STREAM_ORDERED = 0u;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bc") == 0 && chain->stage == SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION);
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bc" "Aq") == 0 && chain->stage == SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_QUERY);
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bc" "Aq" "Ve") == 0 && chain->stage == SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_EXCHANGE);
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,"Bc" "Aq" "Ve" "Wc") == 0 && chain->stage == SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION);
	for (advance=0u; advance<16u && strstr(WALK_TRACE,"Sc" "Tq" "Ve" "Wc") == 0 && chain->active != 0u; advance++)
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	if ( strstr(WALK_TRACE,"Sc" "Tq" "Ve" "Wc") == 0 )
	{
		fprintf(stderr,"eager shard trace %s\n",WALK_TRACE);
		abort();
	}
	assert(chain->stage == SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION && chain->next_layer == 1u);
	WALK_SHARD_MASK = 0u;
	free(chain);
	linear_chain_teardown();
	puts("PASS kv shard chain: query gather, partials, all-to-all and merge run in order in the linear/graph walk and the eager stages");
}

static void check_kv_shard_rank_state(void)
{
	uint32_t degrees[2] = {8u,16u},index;
	for (index=0u; index<2u; index++)
	{
		char path[] = "/tmp/g5n_kv_bytes_XXXXXX",text[512] = {0},expected[512];
		int descriptor = mkstemp(path),saved = dup(2);
		assert(descriptor >= 0 && saved >= 0 && dup2(descriptor,2) == 2);
		memset(&state,0,sizeof(state));
		state.ledger.module_tag = "kv-shard-rank-state";
		state.tp_degree = degrees[index];
		state.tp_rank = degrees[index] - 1u;
		state.kv_shard = 1u;
		state.layer_count = 4u;
		state.resident_sequence_capacity = 3u;
		state.max_sequence_positions = 64u;
		state.page_count = 7u;
		state.physical_page_count = 2u;
		state.kv_backing_directory = "/unused-host-fixture";
		assert(SparkGlm5NextAllocateCaches(&state) == SPARK_STATUS_PENDING);
		fflush(stderr);
		assert(dup2(saved,2) == 2 && close(saved) == 0);
		assert(pread(descriptor,text,sizeof(text) - 1u,0) > 0 && close(descriptor) == 0 && unlink(path) == 0);
		(void)snprintf(expected,sizeof(expected),"GLM-KV-BYTES rank=%u tp=%u shard=1 physical_pages=2 latent_bytes=%llu index_bytes=%llu replicated_latent_bytes=%llu replicated_index_bytes=%llu\n",
			degrees[index] - 1u,degrees[index],
			(unsigned long long)(state.kv_layer_count * 2u * 64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES / degrees[index]),
			(unsigned long long)(state.index_layer_count * 2u * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u / degrees[index]),
			(unsigned long long)(state.kv_layer_count * 2u * 64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES),
			(unsigned long long)(state.index_layer_count * 2u * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u));
		if ( strstr(text,expected) == 0 )
			fprintf(stderr,"kv bytes line:\n%swant:\n%s",text,expected);
		assert(state.kv_layer_count != 0u && state.index_layer_count != 0u && strstr(text,expected) != 0);
		assert(state.kv_layer_stride_bytes * degrees[index] == 2u * 64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES);
		assert(state.index_layer_stride_bytes * degrees[index] == 2u * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
		free_cache_fixture();
	}
	puts("PASS kv shard rank state: latent KV and indexer key pools are 1/tp of the replicated pools at TP8 and TP16, and GLM-KV-BYTES reports both");
}

static SparkStatus kv_shard_collective_init(uint32_t kv_shard,uint32_t advertised)
{
	static SparkWeightdLazyPack pack;
	SparkGlm5NextResidentDecodeStageNodeContext context = {0};
	memset(&state,0,sizeof(state));
	memset(&pack,0,sizeof(pack));
	pack.attached.mesh_send_buffer_addr = 0x40000u;
	state.lazy_pack = &pack;
	state.lane_client = (SparkWeightdClient *)(uintptr_t)1u;
	state.tp_degree = 16u;
	state.tp_rank = 5u;
	state.kv_shard = kv_shard;
	state.pipeline_slot_count = 1u;
	state.execution_row_capacity = 8u;
	context.tp_collective_backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	context.tp_connect_timeout_milli = 1u;
	context.tp_operation_timeout_milli = 1u;
	ALL_TO_ALL_ADVERTISED = advertised;
	MESH_ATTACHED[0] = MESH_ATTACHED[1] = 0u;
	COLLECTIVES_CREATED = 0u;
	return(SparkGlm5NextModuleInitializeTpCollective(&state,&context));
}

static void check_kv_shard_collective_init(void)
{
	assert(kv_shard_collective_init(1u,3u) == SPARK_STATUS_OK && COLLECTIVES_CREATED == 2u && MESH_ATTACHED[0] == 1u && MESH_ATTACHED[1] == 1u);
	assert(state.tp_device_collective_initialized == 1u && state.tp_device_collective_hc_initialized == 1u);
	assert(kv_shard_collective_init(1u,0u) == SPARK_STATUS_UNSUPPORTED && COLLECTIVES_CREATED == 2u);
	assert(kv_shard_collective_init(1u,1u) == SPARK_STATUS_UNSUPPORTED && kv_shard_collective_init(1u,2u) == SPARK_STATUS_UNSUPPORTED);
	assert(kv_shard_collective_init(0u,0u) == SPARK_STATUS_OK && COLLECTIVES_CREATED == 2u);
	memset(&state,0,sizeof(state));
	ALL_TO_ALL_ADVERTISED = 3u;
	puts("PASS kv shard startup: the all-to-all check runs after both collectives attach the mesh, accepts a weightd with slice routes on both bands and refuses otherwise");
}

static void check_chain_steps(void)
{
	SparkGlm5NextTpChain *chain = linear_chain_fixture();
	SparkGlm5NextAsyncCompletion *async = &state.completions[0];
	uint32_t syncs = SYNC_COUNT;
	uint32_t queries;
	uint64_t started;
	chain->steps = async->steps = 3u;
	async->launch_ns = 1u;
	VERIFY_COUNT = 0u;
	queries = STREAM_QUERY_COUNT;
	WALK_DELAY_NS = 2000000u;
	started = SparkGlm5NextNowNs();
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	WALK_DELAY_NS = 0u;
	assert(async->walk_ns >= 6000000u && async->walk_ns <= SparkGlm5NextNowNs() - started);
	assert(strcmp(WALK_TRACE,LINEAR_WALK LINEAR_WALK LINEAR_WALK) == 0 && UNPACK_COUNT == 3u && VERIFY_COUNT == 4u && SYNC_COUNT == syncs && STREAM_QUERY_COUNT == queries + 2u);
	assert(LINEAR_POSITIONS[0] == 5u && LINEAR_POSITIONS[1] == 7u && LINEAR_INPUT[0] == 111u && LINEAR_INPUT[1] == 112u);
	assert(LINEAR_CHAIN[0] == 11u && LINEAR_CHAIN[1] == 111u && LINEAR_CHAIN[3] == 12u && LINEAR_CHAIN[4] == 112u && LINEAR_HOST_TOKENS[0] == 211u && LINEAR_HOST_TOKENS[1] == 212u);
	assert(COMPLETION_WORK == SparkGlm5NextCompleteOnWorker && async->completion.status == SPARK_STATUS_OK && async->launch_ns == 1u && async->finish_ns > 1u);
	async->row_count = async->lane_count = 2u;
	async->lane_next_positions[0] = 4u;
	async->lane_next_positions[1] = 6u;
	SparkGlm5NextGatherSteps(async,&state.slots[0]);
	assert(LINEAR_HOST_TOKENS[0] == 11u && LINEAR_HOST_TOKENS[1] == 111u && LINEAR_HOST_TOKENS[2] == 211u && LINEAR_HOST_TOKENS[3] == 12u && LINEAR_HOST_TOKENS[4] == 112u && LINEAR_HOST_TOKENS[5] == 212u);
	assert(async->burst_token_count == 6u && async->cache_extra_tokens == 2u && async->lane_next_positions[0] == 6u && async->lane_next_positions[1] == 8u);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	chain->steps = async->steps = 3u;
	VERIFY_STATUS = SPARK_STATUS_IO_ERROR;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	VERIFY_STATUS = SPARK_STATUS_OK;
	assert(strcmp(WALK_TRACE,LINEAR_WALK) == 0 && async->completion.status == SPARK_STATUS_IO_ERROR && CANCEL_COUNT == 2u && LINEAR_POSITIONS[0] == 3u && LINEAR_INPUT[0] == 1u);
	async->row_count = async->lane_count = 2u;
	async->lane_next_positions[0] = 4u;
	SparkGlm5NextGatherSteps(async,&state.slots[0]);
	assert(async->burst_token_count == 0u && async->cache_extra_tokens == 0u && async->lane_next_positions[0] == 4u);
	async->completion.status = SPARK_STATUS_OK;
	async->steps = 1u;
	SparkGlm5NextGatherSteps(async,&state.slots[0]);
	assert(async->burst_token_count == 0u && async->lane_next_positions[0] == 4u);
	linear_chain_teardown();
	chain = linear_chain_fixture();
	chain->steps = async->steps = 2u;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD;
	chain->next_layer = 7u;
	SparkGlm5NextFinishChain(chain);
	assert(strcmp(WALK_TRACE,LINEAR_WALK) == 0 && LINEAR_POSITIONS[0] == 4u && COMPLETION_WORK == SparkGlm5NextCompleteOnWorker && async->completion.status == SPARK_STATUS_OK);
	linear_chain_teardown();
}

static void check_finish_chain_mtp_tap(void)
{
	static uint16_t hc_mean[SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION],lane_hidden[4u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION];
	uint64_t lane_sequence[4],lane_next[4],position = 40u,sequence = 9u;
	uint32_t mtp,lane;
	for (mtp=0u; mtp<2u; mtp++)
	{
		SparkGlm5NextTpChain *chain = linear_chain_fixture();
		SparkGlm5NextAsyncCompletion *async = &state.completions[0];
		memset(lane_sequence,0,sizeof(lane_sequence));
		memset(lane_next,0,sizeof(lane_next));
		memset(lane_hidden,0,sizeof(lane_hidden));
		hc_mean[0] = 777u;
		lane = LINEAR_SLOTS[0];
		state.slots[0].hc_mean_bf16 = hc_mean;
		state.mtp_lane_hidden_bf16 = lane_hidden;
		state.mtp_lane_sequence = lane_sequence;
		state.mtp_lane_next = lane_next;
		state.verify_mtp = mtp;
		LINEAR_BATCH.active_sequence_count = LINEAR_BATCH.row_count = 1u;
		LINEAR_BATCH.row_positions = &position;
		LINEAR_BATCH.row_sequence_ids = &sequence;
		chain->wave_rows = 1u;
		chain->steps = async->steps = 3u;
		chain->step = 2u;
		SparkGlm5NextFinishChain(chain);
		assert(COMPLETION_WORK == SparkGlm5NextCompleteOnWorker && async->completion.status == SPARK_STATUS_OK);
		assert(state.mtp_taps == mtp && lane_next[lane] == (mtp != 0u ? position + 3u : 0u) && lane_sequence[lane] == (mtp != 0u ? sequence : 0u));
		assert(lane_hidden[(uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION] == (mtp != 0u ? 777u : 0u));
		LINEAR_BATCH.row_positions = 0;
		LINEAR_BATCH.row_sequence_ids = 0;
		linear_chain_teardown();
	}
}

#define WS_WALK "zBsh" "ArPMrQ" "SgTrPMrQ" "ArPMrQ" "ArPRErQ" "ArPRErQ" "HpxU"
#define WS_KEY(layer,expert) ((layer) * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + (expert))

static char WS_PATH[64];
static uint8_t WS_KDA[4][256] __attribute__((aligned(16)));
static uint32_t WS_KDA_INDEX[4];

static void ws_write(const uint32_t *pairs,uint32_t count,char *hex)
{
	FILE *file = fopen(WS_PATH,"wb");
	assert(file != 0 && fwrite(pairs,sizeof(uint32_t),2u * count,file) == 2u * count && fclose(file) == 0);
	assert(SparkSha256File(WS_PATH,hex) == SPARK_STATUS_OK);
}

static void ws_pin_reset(uint32_t first,uint32_t last)
{
	PIN_FIRST_EXPECTED = first;PIN_LAST_EXPECTED = last;
	PIN_CALLS = PIN_KEYS = PIN_RECORDS = PIN_RELEASES = 0u;
	PIN_FAIL_ACQUIRE = PIN_FAIL_BEGIN = PIN_FAIL_RECORD = PIN_FAIL_RELEASE = 0u;
	memset(PIN_PHASES,0,sizeof(PIN_PHASES));
	memset(PIN_SEEN,0,sizeof(PIN_SEEN));
}

static void ws_state_reset(void)
{
	SparkExpertWorkingSetDestroy(&state.expert_ws);
	state.ws_enabled = 0u;
	state.expert_pin_lease_count = state.expert_pin_key_count = 0u;
	memset(state.expert_pin_leases,0,sizeof(state.expert_pin_leases));
	memset(state.expert_pin_phases,0,sizeof(state.expert_pin_phases));
	state.decode_lease_base_saved = 0;
}

static void check_ws_open(void)
{
	uint32_t anchors[6] = {3u,5u,4u,6u,3u,7u},outside[4] = {3u,5u,2u,6u},no_layer4[4] = {3u,5u,3u,6u};
	char hex[SPARK_SHA256_HEX_BYTES],other[SPARK_SHA256_HEX_BYTES];
	int32_t descriptor;
	SparkGlm5NextTpChain *chain = linear_chain_fixture();
	free(chain);
	strcpy(WS_PATH,"/tmp/glm-wset-XXXXXX");
	descriptor = mkstemp(WS_PATH);
	assert(descriptor >= 0 && close(descriptor) == 0);
	state.owns_embedding = 1u;
	LINEAR_PACK.map = (SparkWeightdMap *)(uintptr_t)1u;
	ws_state_reset();
	unsetenv("SPARK_GLM5_NEXT_EXPERT_WSET");unsetenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256");unsetenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_OK && state.ws_enabled == 0u && state.expert_ws.cover == 0);
	ws_write(outside,2u,other);
	ws_write(anchors,3u,hex);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET",WS_PATH,1);
	setenv("SPARK_GLM5_NEXT_PIN_EXPERTS","1",1);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256",hex,1);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_INVALID_ARGUMENT && state.ws_enabled == 0u);
	unsetenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256","abc",1);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_INVALID_ARGUMENT && state.ws_enabled == 0u);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256",other,1);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_HASH_MISMATCH && state.ws_enabled == 0u && state.expert_ws.cover == 0);
	state.owns_final_head = 0u;
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256",hex,1);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_UNSUPPORTED && state.ws_enabled == 0u);
	state.owns_final_head = 1u;
	ws_write(outside,2u,hex);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256",hex,1);
	ws_pin_reset(3u,5u);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_SCHEMA_ERROR && state.ws_enabled == 0u && PIN_CALLS == 0u);
	ws_state_reset();
	ws_write(no_layer4,2u,hex);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256",hex,1);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_UNSUPPORTED && state.ws_enabled == 0u);
	ws_state_reset();
	ws_write(anchors,3u,hex);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256",hex,1);
	ws_pin_reset(3u,5u);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_OK && state.ws_enabled == 1u);
	assert(state.expert_ws.key_count == 3u && PIN_CALLS == 1u && PIN_KEYS == 3u && PIN_SEEN[3][5] == 1u && PIN_SEEN[3][7] == 1u && PIN_SEEN[4][6] == 1u);
	assert(state.expert_pin_lease_count == 1u && state.expert_pin_phases[0] == 1u && state.decode_lease_base_saved == (const uint8_t *)(uintptr_t)64u);
	assert(SparkExpertWorkingSetCovered(&state.expert_ws,3u,5u) == 1u && SparkExpertWorkingSetCovered(&state.expert_ws,4u,6u) == 1u && SparkExpertWorkingSetCovered(&state.expert_ws,4u,5u) == 0u);
	ws_state_reset();
	LINEAR_PACK.map = 0;
	linear_chain_teardown();
}

static void check_ws_acquire_unwind(void)
{
	static uint32_t keys[600];
	uint32_t index,leases;
	memset(&state,0,sizeof(state));
	state.lazy_pack = &LINEAR_PACK;
	LINEAR_PACK.map = (SparkWeightdMap *)(uintptr_t)1u;
	for (index=0u; index<600u; index++)
		keys[index] = WS_KEY(3u + index / 256u,index % 256u);
	ws_pin_reset(3u,6u);
	PIN_FAIL_ACQUIRE = 2u;
	assert(SparkGlm5NextWsAcquire(&state,keys,600u) == SPARK_STATUS_IO_ERROR);
	assert(PIN_CALLS == 2u && state.expert_pin_lease_count == 0u && PIN_RECORDS == 1u && PIN_RELEASES == 2u && PIN_PHASES[1] == 3u && PIN_PHASES[2] == 3u);
	ws_pin_reset(3u,6u);
	state.decode_lease_base_saved = (const uint8_t *)(uintptr_t)128u;
	assert(SparkGlm5NextWsAcquire(&state,keys,600u) == SPARK_STATUS_VALIDATION_FAILED);
	assert(PIN_CALLS == 1u && state.expert_pin_lease_count == 0u && PIN_RECORDS == 1u && PIN_RELEASES == 1u && PIN_PHASES[1] == 3u);
	ws_pin_reset(3u,6u);
	state.decode_lease_base_saved = 0;
	PIN_FAIL_BEGIN = 2u;
	assert(SparkGlm5NextWsAcquire(&state,keys,600u) == SPARK_STATUS_IO_ERROR);
	assert(state.expert_pin_lease_count == 0u && PIN_RECORDS == 1u && PIN_RELEASES == 2u && PIN_PHASES[2] == 3u);
	ws_pin_reset(3u,6u);
	state.decode_lease_base_saved = 0;
	state.expert_pin_lease_count = 3u;
	for (index=0u; index<3u; index++)
	{
		state.expert_pin_leases[index] = 100u + index;
		state.expert_pin_phases[index] = 1u;
	}
	PIN_FAIL_ACQUIRE = 2u;
	assert(SparkGlm5NextWsAcquire(&state,keys,600u) == SPARK_STATUS_IO_ERROR);
	assert(state.expert_pin_lease_count == 3u && state.expert_pin_leases[2] == 102u && state.expert_pin_leases[3] == 0u && PIN_RELEASES == 2u);
	ws_pin_reset(3u,6u);
	assert(SparkGlm5NextWsAcquire(&state,keys,600u) == SPARK_STATUS_OK && state.expert_pin_lease_count == 5u && PIN_KEYS == 600u && PIN_RELEASES == 0u);
	leases = (uint32_t)(sizeof(state.expert_pin_leases) / sizeof(state.expert_pin_leases[0]));
	state.expert_pin_lease_count = leases - 1u;
	ws_pin_reset(3u,6u);
	assert(SparkGlm5NextWsAcquire(&state,keys,600u) == SPARK_STATUS_CAPACITY_EXCEEDED && PIN_CALLS == 0u && state.expert_pin_lease_count == leases - 1u);
	LINEAR_PACK.map = 0;
	memset(&state,0,sizeof(state));
}

static SparkGlm5NextTpChain *ws_chain_fixture(uint32_t steps,const uint32_t *plan,uint32_t plan_count)
{
	uint32_t anchors[4] = {3u,5u,4u,6u};
	char hex[SPARK_SHA256_HEX_BYTES];
	int32_t descriptor;
	SparkGlm5NextTpChain *chain = linear_chain_fixture();
	strcpy(WS_PATH,"/tmp/glm-wset-XXXXXX");
	descriptor = mkstemp(WS_PATH);
	assert(descriptor >= 0 && close(descriptor) == 0);
	ws_write(anchors,2u,hex);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET",WS_PATH,1);
	setenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256",hex,1);
	ws_state_reset();
	state.owns_embedding = 1u;
	state.ledger.module_tag = "ws-chain-test";
	state.kda_layer_count = 1u;
	state.kda_state_pools = WS_KDA[0];state.kda_q_window_pool = WS_KDA[1];state.kda_k_window_pool = WS_KDA[2];state.kda_v_window_pool = WS_KDA[3];
	state.kda_state_layer_stride_bytes = 256u;state.kda_window_layer_stride_bytes = 128u;
	state.kda_state_index_device = WS_KDA_INDEX;
	state.slots[0].run_state_index = LINEAR_RUNS;
	LINEAR_PACK.map = (SparkWeightdMap *)(uintptr_t)1u;
	ws_pin_reset(3u,5u);
	assert(SparkGlm5NextWsOpen(&state) == SPARK_STATUS_OK && state.ws_enabled == 1u && unlink(WS_PATH) == 0);
	unsetenv("SPARK_GLM5_NEXT_EXPERT_WSET");unsetenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256");
	memset(WS_PLAN,0,sizeof(WS_PLAN));
	memcpy(WS_PLAN,plan,plan_count * sizeof(*plan));
	WS_PLAN_KEY = WS_KEY(3u,9u);
	POISON_LAUNCHES = POISON_PENDING = 0u;
	chain->steps = state.completions[0].steps = steps;
	return(chain);
}

static void ws_chain_teardown(void)
{
	SparkExpertWorkingSetDestroy(&state.expert_ws);
	SparkStageModuleLedgerRollback(&state.ledger,0u);
	if ( state.slots[0].miss_ring != 0 )
		(void)cudaFreeHost(state.slots[0].miss_ring);
	LINEAR_PACK.map = 0;
	linear_chain_teardown();
}

static void check_ws_chain_rollback(void)
{
	static const uint32_t local_then_commit[3] = {1u,0u,0u},remote_last[2] = {2u,0u},bad_ring[1] = {3u};
	SparkGlm5NextTpChain *chain = ws_chain_fixture(2u,local_then_commit,3u);
	SparkGlm5NextAsyncCompletion *async = &state.completions[0];
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,WS_WALK "L" WS_WALK WS_WALK) == 0);
	assert(async->completion.status == SPARK_STATUS_OK && COMPLETION_WORK == SparkGlm5NextCompleteOnWorker && CANCEL_COUNT == 0u);
	assert(LINEAR_POSITIONS[0] == 4u && LINEAR_POSITIONS[1] == 6u && LINEAR_CHAIN[0] == 111u && LINEAR_CHAIN[2] == 112u && LINEAR_HOST_TOKENS[0] == 211u && LINEAR_HOST_TOKENS[1] == 212u);
	assert(state.ws_local_miss == 1u && state.ws_remote_miss == 0u && state.ws_replays == 1u && state.ws_eager_steps == 0u);
	assert(SparkExpertWorkingSetCovered(&state.expert_ws,3u,9u) == 1u && state.expert_ws.key_count == 3u && state.expert_pin_lease_count == 2u && PIN_SEEN[3][9] == 1u);
	assert(state.slots[0].cover_generation == state.expert_ws.generation && state.slots[0].snapshot_span_count == 4u && POISON_LAUNCHES == 3u);
	assert(state.slots[0].miss_ring[SPARK_STEP_MISS_FLAG] == 0u && state.slots[0].miss_ring[SPARK_STEP_MISS_COUNT] == 0u);
	ws_chain_teardown();
	chain = ws_chain_fixture(1u,remote_last,2u);
	async = &state.completions[0];
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,WS_WALK "L" WS_WALK) == 0);
	assert(async->completion.status == SPARK_STATUS_OK && LINEAR_HOST_TOKENS[0] == 111u && LINEAR_HOST_TOKENS[1] == 112u && LINEAR_POSITIONS[0] == 3u);
	assert(state.ws_remote_miss == 1u && state.ws_replays == 1u && state.expert_ws.key_count == 2u && PIN_CALLS == 1u);
	ws_chain_teardown();
	chain = ws_chain_fixture(2u,bad_ring,1u);
	async = &state.completions[0];
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strcmp(WALK_TRACE,WS_WALK "L") == 0);
	assert(async->completion.status == SPARK_STATUS_VALIDATION_FAILED && CANCEL_COUNT == 2u && LINEAR_POSITIONS[0] == 3u && state.ws_replays == 0u);
	ws_chain_teardown();
}

static void check_ws_replay_budget(void)
{
	static const uint32_t plan[5] = {1u,0u,2u,2u,2u};
	SparkGlm5NextTpChain *chain = ws_chain_fixture(3u,plan,5u);
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	assert(strncmp(WALK_TRACE,WS_WALK "L" WS_WALK WS_WALK "L" WS_WALK "L" WS_WALK "L" "B",strlen(WS_WALK) * 5u + 5u) == 0);
	assert(strchr(WALK_TRACE + strlen(WS_WALK) * 5u + 4u,'z') == 0 && strchr(WALK_TRACE + strlen(WS_WALK) * 5u + 4u,'p') == 0);
	assert(chain->active == 1u && chain->step == 1u && chain->ws_force_eager == 1u && chain->stage != SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN && chain->wave.expert_cover == 0);
	assert(state.ws_local_miss == 1u && state.ws_remote_miss == 3u && state.ws_replays == 3u && state.ws_eager_steps == 1u && POISON_LAUNCHES == 5u);
	assert(LINEAR_POSITIONS[0] == 4u && LINEAR_POSITIONS[1] == 6u && LINEAR_CHAIN[0] == 111u && LINEAR_CHAIN[3] == 112u);
	assert(SparkGlm5NextLinearEligible(chain) == 0u && SparkGlm5NextGraphClaimExperts(chain) == SPARK_STATUS_UNSUPPORTED);
	SparkGlm5NextFeedStep(chain);
	assert(chain->ws_force_eager == 0u && chain->ws_replay.attempts == 0u && SparkGlm5NextLinearEligible(chain) == 1u);
	free(chain);
	ws_chain_teardown();
}

static void check_ws_graph_result_replays(void)
{
	static const uint32_t plan[1] = {0u};
	SparkGlm5NextTpChain *chain = ws_chain_fixture(1u,plan,1u);
	assert(SparkGlm5NextBuildWave(chain) == SPARK_STATUS_OK && SparkGlm5NextGraphClaimExperts(chain) == SPARK_STATUS_OK);
	chain->step_verdict = SPARK_STEP_VERDICT_ROLLBACK_REMOTE;
	assert(SparkGlm5NextGraphResult(chain,SPARK_STATUS_UNSUPPORTED) == 1u);
	assert(strcmp(WALK_TRACE,"L" WS_WALK) == 0 && state.completions[0].completion.status == SPARK_STATUS_OK && state.ws_replays == 1u);
	ws_chain_teardown();
}

static void check_ws_snapshot_layout(void)
{
	const SparkStateSpan *spans;
	uint8_t *pools[4];
	uint64_t strides[4],expected = 0u,allocated = 0u;
	uint32_t part,layer,allocation,count,words = 0u,index;
	memset(&state,0,sizeof(state));
	state.ledger.module_tag = "ws-snapshot-test";
	state.tp_degree = 16u;
	state.layer_count = 4u;
	state.resident_sequence_capacity = 3u;
	state.max_sequence_positions = 64u;
	state.page_count = 7u;
	state.physical_page_count = 2u;
	state.kv_backing_directory = "/unused-host-fixture";
	assert(SparkGlm5NextAllocateCaches(&state) == SPARK_STATUS_PENDING && state.kda_layer_count == 3u);
	pools[0] = state.kda_state_pools;pools[1] = state.kda_q_window_pool;pools[2] = state.kda_k_window_pool;pools[3] = state.kda_v_window_pool;
	strides[0] = state.kda_state_layer_stride_bytes;strides[1] = strides[2] = strides[3] = state.kda_window_layer_stride_bytes;
	assert(SparkGlm5NextWsSnapshotEnsure(&state,&state.slots[0]) == SPARK_STATUS_OK);
	count = state.slots[0].snapshot_span_count;
	spans = (const SparkStateSpan *)state.slots[0].snapshot_spans;
	assert(count == 12u && spans != 0 && state.slots[0].snapshot != 0);
	for (part=0u; part<4u; part++)
		for (layer=0u; layer<3u; layer++)
		{
			index = part * 3u + layer;
			assert(spans[index].base == pools[part] + (uint64_t)layer * strides[part]);
			assert(spans[index].row_stride * state.resident_sequence_capacity == strides[part] && spans[index].row_bytes == spans[index].row_stride);
			assert(spans[index].state_rows == 3u && spans[index].snapshot_offset == expected && (spans[index].row_stride % SPARK_STATE_SPAN_ALIGN) == 0u);
			if ( part != 0u && layer == 2u && part < 3u )
				assert(spans[index].base + strides[part] == pools[part + 1u]);
			expected += (uint64_t)SPARK_GLM5_NEXT_WS_ROWS_MAX * spans[index].row_bytes;
			words = spans[index].row_bytes / SPARK_STATE_SPAN_ALIGN > words ? spans[index].row_bytes / SPARK_STATE_SPAN_ALIGN : words;
		}
	assert(spans[0].row_bytes == 64u * 128u * 128u * sizeof(float) / 16u && spans[3].row_bytes == 64u * 128u * 4u * sizeof(uint16_t) / 16u);
	for (allocation=0u; allocation<state.ledger.device_allocation_count; allocation++)
		if ( state.ledger.device_allocations[allocation] == state.slots[0].snapshot )
			allocated = state.ledger.device_allocation_bytes[allocation];
	assert(allocated == expected && state.slots[0].snapshot_row_words == words);
	allocation = state.ledger.device_allocation_count;
	assert(SparkGlm5NextWsSnapshotEnsure(&state,&state.slots[0]) == SPARK_STATUS_OK && state.ledger.device_allocation_count == allocation);
	assert(SparkGlm5NextWsSnapshotEnsure(&state,&state.slots[1]) == SPARK_STATUS_OK && state.slots[1].snapshot != state.slots[0].snapshot);
	free_cache_fixture();
	memset(&state,0,sizeof(state));
	state.ledger.module_tag = "ws-snapshot-test";
	assert(SparkGlm5NextWsSnapshotEnsure(&state,&state.slots[0]) == SPARK_STATUS_UNSUPPORTED && state.slots[0].snapshot == 0);
}

static uint32_t SEQ_TOKENS[8],SEQ_POSITIONS[8],SEQ_SLOTS[8],SEQ_OUTPUT[8],SEQ_HOST_OUTPUT[64],SEQ_CHAIN[64],SEQ_BEGIN[9],SEQ_INDICES[8],SEQ_RUNS[8],SEQ_ERRORS[SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT],SEQ_HOST_ERRORS[SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT],SEQ_TABLE[16],SEQ_SHADOW[16],SEQ_BUFFER[64],SEQ_DONE_COUNT;
static uint64_t SEQ_MAXLOC[8],SEQ_REQUEST = 10u;
static uint8_t SEQ_SIDEBAND[8u * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW];
static SparkRowSampling SEQ_SAMPLING[8],SEQ_ROW_SAMPLING[8];
static SparkModelDriverCompletion SEQ_DONE;
static SparkTestKvTransactions SEQ_KV;

static void sequence_done(void *context,const SparkModelDriverCompletion *completion)
{
	(void)context;
	SEQ_DONE = *completion;
	SEQ_DONE_COUNT++;
}

static void sequence_slot(SparkGlm5NextExecutionSlot *slot)
{
	linear_slot_fixture(slot);
	slot->output_token = SEQ_OUTPUT;slot->head_maxloc_u64 = SEQ_MAXLOC;slot->host_output_token_ids = SEQ_HOST_OUTPUT;slot->host_chain_token_ids = SEQ_CHAIN;
	slot->host_token_ids = SEQ_TOKENS;slot->host_positions = SEQ_POSITIONS;slot->host_resident_slots = SEQ_SLOTS;slot->host_row_sampling = SEQ_SAMPLING;
	slot->host_run_begin = SEQ_BEGIN;slot->host_run_row_indices = SEQ_INDICES;slot->host_run_state_index = SEQ_RUNS;slot->kv_access_error = SEQ_ERRORS;slot->host_kv_access_error = SEQ_HOST_ERRORS;
}

static void sequence_fixture(void)
{
	memset(&state,0,sizeof(state));
	SparkTestKvTransactionsInitialize(&SEQ_KV,1u);
	state.pipeline_slot_count = 1u;state.execution_row_capacity = 8u;state.resident_sequence_capacity = 2u;state.pages_per_sequence = 4u;state.max_sequence_positions = 64u;
	state.tp_degree = 16u;state.layer_count = 5u;state.owns_embedding = state.owns_final_head = 1u;
	state.lazy_pack = &LINEAR_PACK;state.completion_worker = (SparkWeightdWorker *)(uintptr_t)1u;
	state.tp_device_collective_initialized = state.tp_device_collective_hc_initialized = 1u;
	state.execution_stream = (void *)(uintptr_t)7u;
	state.kv_transactions = SEQ_KV.transactions;state.kv_lane_transactions = SEQ_KV.owners;state.kv_lane_physical_pages = SEQ_KV.physical;
	memset(SEQ_TABLE,0xff,sizeof(SEQ_TABLE));
	memset(SEQ_SHADOW,0xff,sizeof(SEQ_SHADOW));
	state.page_table = SEQ_TABLE;state.page_table_shadow = SEQ_SHADOW;
	state.completions[0].state = &state;
	sequence_slot(&state.slots[0]);
	pin_fixture_experts();
	assert(pthread_mutex_init(&state.completion_queue_lock,0) == 0 && pthread_mutex_init(&state.kv_mutex,0) == 0);
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,(cudaStream_t)state.execution_stream) == SPARK_STATUS_OK);
	STREAM_ORDERED = 1u;HOST_MODE = 0u;HOST_COUNT = 0u;CANCEL_COUNT = EXPECTED_CANCEL_COUNT = 0u;WORK_STATUS = SPARK_STATUS_OK;DRAIN_STATUS = cudaSuccess;VERIFY_STATUS = SPARK_STATUS_OK;
	WALK_LENGTH = 0u;WALK_TRACE[0] = 0;WALK_GATHER_LAYER = 99u;WALK_FAIL_CODE = 0u;UNPACK_COUNT = 0u;SEQ_DONE_COUNT = 0u;
}

static void sequence_admit(uint32_t flags,SparkModelDriverAdmissionDecision *decision)
{
	SparkModelDriverInitializeAdmissionDecision(decision);
	SEQ_KV.request.admission_flags = flags;
	assert(SparkGlm5NextAdmissionPredicate(&state,&SEQ_KV.request,decision) == SPARK_STATUS_OK);
}

static void sequence_lane(uint64_t sequence,uint64_t first,uint64_t next,uint32_t publish)
{
	SEQ_REQUEST++;
	SEQ_KV.request.request_id = SEQ_KV.request.submission_id = SEQ_KV.request.transaction_id = SEQ_KV.request.step_generation = SEQ_REQUEST;
	SEQ_KV.request.active_slot_count = SEQ_KV.request.cache_lane_count = 1u;
	SparkTestKvPageLane(&SEQ_KV.lanes[0],sequence,0u,(uint32_t)first,(uint32_t)next);
	if ( publish != 0u )
		SparkTestKvPagePublish(&SEQ_KV.lanes[0],(uint32_t)next,(uint8_t)next);
}

static SparkModelDriverFrame *sequence_admitted_frame(uint32_t new_tokens,uint32_t frame_flags)
{
	static SparkModelDriverFrame frame;
	SparkModelDriverAdmissionDecision decision;
	SEQ_KV.request.new_token_count = new_tokens;
	SEQ_KV.request.frame_flags = frame_flags;
	sequence_admit(SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE,&decision);
	sequence_admit(SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT,&decision);
	sequence_admit(0u,&decision);
	frame = SparkTestKvTransactionFrame(&SEQ_KV.request);
	assert(SparkModelDriverApplyAdmissionDecision(&decision,&frame) == SPARK_STATUS_OK);
	frame.flags |= frame_flags;
	frame.execution_stream = state.execution_stream;
	frame.completion_function = sequence_done;
	return(&frame);
}

static void sequence_rows(uint64_t sequence,uint64_t first,uint32_t rows,SparkGlm5NextResidentDecodeStageBatchView *batch)
{
	static uint32_t tokens[8],slots[8];
	static uint64_t positions[8],rows_sequence[8];
	uint32_t row;
	for (row=0u; row<rows; row++)
	{
		slots[row] = 0u;
		positions[row] = first + row;
		rows_sequence[row] = sequence;
		tokens[row] = 500u + row;
	}
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = rows;batch->active_sequence_count = 1u;
	batch->token_ids = tokens;batch->row_resident_slots = slots;batch->row_positions = positions;batch->row_sequence_ids = rows_sequence;batch->row_sampling = SEQ_ROW_SAMPLING;
}

static void sequence_context(SparkGlm5NextResidentDecodeStageFrameContext *context,SparkGlm5NextResidentDecodeStageBatchView *batch,uint32_t prefill)
{
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;context->descriptor_bytes = sizeof(*context);context->batch = batch;
	context->flags = prefill != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	if ( SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(0u) == 0u )
		return;
	context->flags |= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT;
	context->sideband_output = SEQ_SIDEBAND;
	context->sideband_output_bytes = sizeof(SEQ_SIDEBAND);
}

static SparkStatus sequence_frame(uint64_t sequence,uint64_t first,uint32_t rows,uint32_t steps,uint32_t publish)
{
	static SparkModelDriverBuffer buffer;
	static SparkGlm5NextResidentDecodeStageBatchView batch;
	static SparkGlm5NextResidentDecodeStageFrameContext context;
	SparkModelDriverFrame *frame;
	SparkWeightdWorkFunction work;
	SparkStatus status;
	uint32_t prefill = rows > 1u || first == 0u ? 1u : 0u;
	sequence_lane(sequence,first,first + rows,publish);
	frame = sequence_admitted_frame(rows,prefill != 0u ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u);
	sequence_rows(sequence,first,rows,&batch);
	sequence_context(&context,&batch,prefill);
	buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;buffer.address = SEQ_BUFFER;buffer.bytes = (uint64_t)rows * steps * sizeof(uint32_t);
	frame->user_context = &context;frame->buffers = &buffer;frame->buffer_count = 1u;frame->tokens_per_sequence = steps;
	UNPACK_ROWS = ENQUEUE_ROWS = rows;ENQUEUE_SEQUENCES = 1u;COMPLETION_WORK = 0;
	status = SparkGlm5NextResidentDecodeStageExecute(&state,frame);
	if ( status != SPARK_STATUS_OK )
		return(status);
	assert(COMPLETION_WORK == SparkGlm5NextCompleteOnWorker);
	work = COMPLETION_WORK;
	COMPLETION_WORK = 0;
	work(COMPLETION_CONTEXT);
	return(SEQ_DONE.status);
}

static SparkStatus sequence_publish(uint64_t sequence,uint64_t at)
{
	SparkModelDriverFrame *frame;
	uint32_t done = SEQ_DONE_COUNT;
	SparkStatus status;
	sequence_lane(sequence,at,at,1u);
	frame = sequence_admitted_frame(0u,SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH);
	frame->tokens_per_sequence = 0u;
	status = SparkGlm5NextResidentDecodeStageExecute(&state,frame);
	assert(status != SPARK_STATUS_OK || SEQ_DONE_COUNT == done + 1u);
	return(status == SPARK_STATUS_OK ? SEQ_DONE.status : status);
}

static void sequence_idle(uint64_t next)
{
	assert(atomic_load(&state.tp_chain_active) == 0u && atomic_load(&state.slot_states[0]) == SPARK_STAGE_MODULE_SLOT_FREE);
	assert(atomic_load(&state.lane_states[0]) == SPARK_STAGE_MODULE_SLOT_FREE && SEQ_KV.owners[0].phase == SPARK_KV_LANE_TRANSACTION_EMPTY);
	assert(atomic_load(&state.lane_next_positions[0]) == next && SEQ_KV.pages.cache.sequences[0].next_token_position == next);
}

static void sequence_release(uint64_t sequence,uint64_t at)
{
	SparkModelDriverAdmissionDecision decision;
	sequence_lane(sequence,at,at,0u);
	SEQ_KV.lanes[0].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	SEQ_KV.request.new_token_count = 0u;
	SEQ_KV.request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	sequence_admit(0u,&decision);
	assert(atomic_load(&state.lane_bound[0]) == 0u && SEQ_KV.pages.cache.sequences[0].sequence_id == 0u);
	SEQ_KV.request.frame_flags = 0u;
}

static void sequence_teardown(void)
{
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
	assert(pthread_mutex_destroy(&state.completion_queue_lock) == 0 && pthread_mutex_destroy(&state.kv_mutex) == 0);
	STREAM_ORDERED = 0u;UNPACK_ROWS = ENQUEUE_ROWS = ENQUEUE_SEQUENCES = 2u;
	memset(&state,0,sizeof(state));
}

static void check_execute_sequence(void)
{
	sequence_fixture();
	assert(sequence_frame(1u,0u,2u,1u,0u) == SPARK_STATUS_OK && SEQ_DONE_COUNT == 1u && SEQ_BUFFER[1] == 12u);
	sequence_idle(2u);
	assert(sequence_frame(1u,2u,2u,1u,1u) == SPARK_STATUS_OK && SEQ_DONE_COUNT == 2u);
	sequence_idle(4u);
	assert(sequence_frame(1u,4u,1u,2u,0u) == SPARK_STATUS_OK && SEQ_DONE_COUNT == 3u && SEQ_DONE.tokens_per_sequence == 2u && SEQ_BUFFER[0] == 211u && SEQ_BUFFER[1] == 311u);
	sequence_idle(6u);
	assert(sequence_frame(1u,6u,1u,2u,0u) == SPARK_STATUS_OK && SEQ_DONE_COUNT == 4u && SEQ_BUFFER[0] == 411u && SEQ_BUFFER[1] == 511u);
	sequence_idle(8u);
	assert(sequence_publish(1u,8u) == SPARK_STATUS_OK && SEQ_DONE_COUNT == 5u);
	sequence_idle(8u);
	sequence_release(1u,8u);
	assert(sequence_frame(2u,0u,2u,1u,0u) == SPARK_STATUS_OK && SEQ_DONE_COUNT == 6u);
	sequence_idle(2u);
	sequence_teardown();
}

static void check_chain_validation(void)
{
	SparkModelDriverFrame frame = {0};
	SparkModelDriverCacheLane lanes[2] = {0};
	SparkModelDriverBuffer buffer = {0};
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	SparkGlm5NextResidentDecodeStageFrameContext context = {0};
	SparkGlm5NextStateCapture capture = {0};
	uint64_t positions[2] = {60u,4u};
	uint32_t output[8];
	memset(&state,0,sizeof(state));
	state.owns_embedding = state.owns_final_head = 1u;
	state.max_sequence_positions = 128u;
	batch.row_count = batch.active_sequence_count = 2u;
	batch.row_positions = positions;
	context.batch = &batch;
	frame.cache_lanes = lanes;
	frame.cache_lane_count = 2u;
	frame.tokens_per_sequence = 4u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_OK);
	positions[0] = 61u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
	positions[0] = 60u;
	state.max_sequence_positions = 63u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_CAPACITY_EXCEEDED);
	state.max_sequence_positions = 128u;
	frame.tokens_per_sequence = SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE + 1u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_INVALID_ARGUMENT);
	frame.tokens_per_sequence = 4u;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_UNSUPPORTED);
	frame.flags = 0u;
	context.state_capture = &capture;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_UNSUPPORTED);
	context.state_capture = 0;
	lanes[1].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_UNSUPPORTED);
	lanes[1].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_UNSUPPORTED);
	lanes[1].flags = 0u;
	state.owns_embedding = 0u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_UNSUPPORTED);
	state.owns_embedding = 1u;
	state.owns_final_head = 0u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_UNSUPPORTED);
	state.owns_final_head = 1u;
	positions[0] = 63u;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	frame.tokens_per_sequence = 1u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_OK);
	frame.tokens_per_sequence = 0u;
	assert(SparkGlm5NextValidateChain(&state,&frame,&context) == SPARK_STATUS_OK);
	frame.flags = 0u;
	frame.tokens_per_sequence = 4u;
	frame.buffers = &buffer;
	frame.buffer_count = 1u;
	buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
	buffer.address = output;
	buffer.bytes = 2u * 4u * sizeof(uint32_t) - 1u;
	assert(SparkGlm5NextValidateFrameBuffers(&state,&frame,2u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	buffer.bytes++;
	assert(SparkGlm5NextValidateFrameBuffers(&state,&frame,2u) == SPARK_STATUS_OK);
	memset(&state,0,sizeof(state));
}

static void check_frame_chain_validation(void)
{
	static uint8_t sideband[2u * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW];
	SparkModelDriverFrame frame = {0};
	SparkModelDriverBuffer buffer = {0};
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	SparkGlm5NextResidentDecodeStageFrameContext context = {0};
	SparkRowSampling sampling[2] = {0};
	const SparkGlm5NextResidentDecodeStageFrameContext *out = 0;
	uint32_t tokens[2] = {1u,2u},slots[2] = {0u,1u},output[8];
	uint64_t positions[2] = {60u,4u},sequences[2] = {1u,2u};
	memset(&state,0,sizeof(state));
	state.owns_embedding = state.owns_final_head = 1u;
	state.execution_stream = (void *)(uintptr_t)7u;
	state.execution_row_capacity = 8u;
	state.resident_sequence_capacity = 4u;
	state.max_sequence_positions = 128u;
	batch.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch.descriptor_bytes = sizeof(batch);
	batch.row_count = batch.active_sequence_count = 2u;
	batch.token_ids = tokens;
	batch.row_resident_slots = slots;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequences;
	batch.row_sampling = sampling;
	context.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context.descriptor_bytes = sizeof(context);
	context.batch = &batch;
	if ( SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(0u) != 0u )
	{
		context.flags = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT;
		context.sideband_output = sideband;
		context.sideband_output_bytes = sizeof(sideband);
	}
	buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
	buffer.address = output;
	buffer.bytes = sizeof(output);
	frame.user_context = &context;
	frame.execution_stream = state.execution_stream;
	frame.completion_function = observe_completion;
	frame.active_slot_count = frame.new_token_count = 2u;
	frame.buffers = &buffer;
	frame.buffer_count = 1u;
	frame.tokens_per_sequence = 4u;
	assert(SparkGlm5NextValidateFrame(&state,&frame,&out) == SPARK_STATUS_OK && out == &context);
	positions[0] = 61u;
	assert(SparkGlm5NextValidateFrame(&state,&frame,&out) == SPARK_STATUS_CAPACITY_EXCEEDED && out == 0);
	memset(&state,0,sizeof(state));
}

static void check_mtp_skips_chains(void)
{
	SparkGlm5NextTpChain chain = {0};
	SparkModelDriverFrame frame = {0};
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	SparkRowSampling sampling = {0};
	uint32_t slots[1] = {0u},tokens[4] = {7u,0u,0u,0u},positions[4] = {5u,0u,0u,0u},residents[4] = {0u};
	uint64_t row_positions[1] = {5u};
	uint8_t armed[1] = {1u};
	memset(&state,0,sizeof(state));
	state.mtp_enabled = state.owns_embedding = state.owns_final_head = 1u;
	state.resident_sequence_capacity = 1u;
	state.max_sequence_positions = 64u;
	state.mtp_lane_armed = armed;
	state.slots[0].host_token_ids = tokens;
	state.slots[0].host_positions = positions;
	state.slots[0].host_resident_slots = residents;
	batch.row_count = batch.active_sequence_count = 1u;
	batch.token_ids = tokens;
	batch.row_resident_slots = slots;
	batch.row_positions = row_positions;
	batch.row_sampling = &sampling;
	chain.state = &state;
	chain.slot = &state.slots[0];
	chain.wave_rows = 1u;
	frame.tokens_per_sequence = 2u;
	MTP_DRAFTS = 0u;
	assert(SparkGlm5NextMtpDriveDraft(&state,&frame,&batch,&state.slots[0],&chain) == SPARK_STATUS_OK && MTP_DRAFTS == 0u && chain.spec_verify == 0u && chain.wave_rows == 1u);
	frame.tokens_per_sequence = 1u;
	assert(SparkGlm5NextMtpDriveDraft(&state,&frame,&batch,&state.slots[0],&chain) == SPARK_STATUS_OK && MTP_DRAFTS == 1u && chain.spec_verify == 1u && chain.wave_rows == 3u && tokens[1] == 21u && positions[2] == 7u);
	memset(&state,0,sizeof(state));
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
	const char *expected = EXPECTED_WAVE_TIMING "\n";
	int descriptor = mkstemp(path),saved = dup(2);
	uint64_t index;
	assert(descriptor >= 0 && saved >= 0 && dup2(descriptor,2) == 2);
	for (index=0u; index<3u; index++)
	{
		memset(&wave,0,sizeof(wave));
		wave.row_count = 8u;
		wave.steps = index == 0u ? 4u : 1u;
		wave.epoch[0] = 11u;
		wave.epoch[1] = 12u;
		wave.graph_path = SPARK_GLM5_NEXT_GRAPH_PATH_ON;
		wave.completion.request_id = 7u + index;
		wave.graph = index < 2u ? 1u : 0u;
		wave.prefill = index == 2u ? 1u : 0u;
		wave.linear = index == 2u ? 1u : 0u;
		wave.walk_ns = index == 2u ? 45000000u : 0u;
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

static void check_route_trace(void)
{
	static uint32_t offsets[4u * (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u)];
	SparkGlm5NextTpChain chain = {0};
	uint32_t position = 41u,expert,*layer_offsets = offsets + 3u * (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u);
	char directory[] = "/tmp/g5n-route-XXXXXX",prefix[128],path[160],line[256];
	FILE *file;
	memset(&state,0,sizeof(state));
	assert(unsetenv("SPARK_GLM5_NEXT_ROUTE_TRACE") == 0 && unsetenv("SPARK_GLM5_NEXT_PIN_EXPERTS") == 0);
	assert(SparkGlm5NextRouteTraceOpen(&state) == SPARK_STATUS_OK && state.route_trace == 0);
	assert(mkdtemp(directory) != 0);
	(void)snprintf(prefix,sizeof(prefix),"%s/run",directory);
	assert(setenv("SPARK_GLM5_NEXT_ROUTE_TRACE",prefix,1) == 0);
	state.lazy_pack = &OPEN_PACK;state.graph_path_requested = 1u;
	assert(SparkGlm5NextRouteTraceOpen(&state) == SPARK_STATUS_INVALID_ARGUMENT && state.route_trace == 0);
	state.graph_path_requested = 0u;
	assert(setenv("SPARK_GLM5_NEXT_PIN_EXPERTS","1",1) == 0);
	assert(SparkGlm5NextRouteTraceOpen(&state) == SPARK_STATUS_INVALID_ARGUMENT && state.route_trace == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_PIN_EXPERTS") == 0);
	state.lazy_pack = 0;
	assert(SparkGlm5NextRouteTraceOpen(&state) == SPARK_STATUS_INVALID_ARGUMENT && state.route_trace == 0);
	state.lazy_pack = &OPEN_PACK;state.stage_index = 1u;state.tp_rank = 7u;
	assert(SparkGlm5NextRouteTraceOpen(&state) == SPARK_STATUS_OK && state.route_trace != 0);
	for (expert=0u; expert<=SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT; expert++)
		layer_offsets[expert] = expert <= 5u ? 0u : expert <= 17u ? 3u : SPARK_GLM5_NEXT_MODEL_MOE_TOP_K;
	state.slots[0].host_group_row_offset = offsets;
	chain.state = &state;chain.slot = &state.slots[0];chain.next_layer = 3u;chain.wave.row_count = 1u;chain.wave.host_positions = &position;
	assert(SparkGlm5NextRouteTraceWrite(&chain) == SPARK_STATUS_OK);
	layer_offsets[SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT] = 9u;
	assert(SparkGlm5NextRouteTraceWrite(&chain) == SPARK_STATUS_SCHEMA_ERROR);
	SparkGlm5NextRouteTraceClose(&state);
	assert(state.route_trace == 0);
	(void)snprintf(path,sizeof(path),"%s.stage01.rank07.trace",prefix);
	file = fopen(path,"r");
	assert(file != 0 && fgets(line,sizeof(line),file) != 0);
	assert(strcmp(line,"G5N-ROUTE rank=7 rows=1 pos=41 layer=3 n=2 e=5,17\n") == 0 && fgets(line,sizeof(line),file) == 0);
	assert(fclose(file) == 0 && unlink(path) == 0 && rmdir(directory) == 0);
	assert(unsetenv("SPARK_GLM5_NEXT_ROUTE_TRACE") == 0);
}

static void check_kda_timing(void)
{
	const char *expected = "G5N-KDA-TIMING rank=5 restores=2 restore_bytes=120 restore_us=0 captures=1 capture_bytes=60 capture_us=0\n";
	char path[] = "/tmp/g5n_kda_timing_XXXXXX",text[512] = {0};
	int descriptor = mkstemp(path),saved = dup(2);
	uint64_t start = UINT64_C(5000000000),now = SparkGlm5NextNowNs() + UINT64_C(1000000000);
	uint32_t field;
	memset(&state,0,sizeof(state));
	state.tp_rank = 5u;
	SparkGlm5NextKdaCount(state.kda_restore,60u,now);
	SparkGlm5NextKdaCount(state.kda_restore,60u,now);
	SparkGlm5NextKdaCount(state.kda_capture,60u,now);
	assert(descriptor >= 0 && saved >= 0 && dup2(descriptor,2) == 2);
	SparkGlm5NextKdaTimingReport(&state,start);
	SparkGlm5NextKdaTimingReport(&state,start + SPARK_GLM5_NEXT_WAVE_TIMING_WINDOW_NS - 1u);
	assert(atomic_load(&state.kda_restore[SPARK_GLM5_NEXT_KDA_COUNT]) == 2u);
	SparkGlm5NextKdaTimingReport(&state,start + SPARK_GLM5_NEXT_WAVE_TIMING_WINDOW_NS);
	SparkGlm5NextKdaTimingReport(&state,start + 3u * SPARK_GLM5_NEXT_WAVE_TIMING_WINDOW_NS);
	fflush(stderr);
	assert(dup2(saved,2) == 2 && close(saved) == 0);
	assert(pread(descriptor,text,sizeof(text) - 1u,0) > 0 && close(descriptor) == 0 && unlink(path) == 0);
	if ( strcmp(text,expected) != 0 )
		fprintf(stderr,"kda timing line:\n%s",text);
	assert(strcmp(text,expected) == 0);
	for (field=0u; field<SPARK_GLM5_NEXT_KDA_FIELDS; field++)
		assert(atomic_load(&state.kda_restore[field]) == 0u && atomic_load(&state.kda_capture[field]) == 0u);
	assert(atomic_load(&state.kda_window_ns) == start + 3u * SPARK_GLM5_NEXT_WAVE_TIMING_WINDOW_NS);
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
	check_kv_shard_walk();
	check_kv_shard_rank_state();
	check_kv_shard_collective_init();
	check_chain_steps();
	check_finish_chain_mtp_tap();
	check_execute_sequence();
	check_chain_validation();
	check_frame_chain_validation();
	check_mtp_skips_chains();
	check_verify_deferred();
	check_attempt_accounting();
	check_wave_timing();
	check_kda_timing();
	check_graph_epoch_ownership();
	check_working_set_recover();
	check_ws_open();
	check_ws_acquire_unwind();
	check_ws_chain_rollback();
	check_ws_replay_budget();
	check_ws_graph_result_replays();
	check_ws_snapshot_layout();
	check_lazy_open_retained_owner();
	check_route_trace();
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
	check_verify_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE);
	check_verify_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY);
	check_verify_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP);
	check_tap_config();
	VERIFY_TAP_DUMP = "/tmp/sparkpipe_glm5_next_tap_dump_test.sptd";
	check_verify_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE);
	check_verify_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY);
	VERIFY_TAP_DUMP = 0;
	check_verify_mtp_admission();
	check_verify_mtp_tap_frames();
	check_verify_mtp_draft_guards();
	check_verify_mtp_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP,1u);
	check_verify_mtp_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP,2u);
	check_verify_mtp_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP,1u);
	check_verify_mtp_rounds(SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP,16u);
	check_mtp_pack();
	check_verify_rank_local();
	check_callback_retirement();
	check_stream_receipt();
	check_chain_ownership();
	int32_t status = check_cache_transactions(SPARK_STATUS_IO_ERROR,SPARK_STATUS_OK,SPARK_STATUS_OK,cudaSuccess,1u);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_OK,SPARK_STATUS_OK,cudaSuccess,1u) == 0);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_OK,SPARK_STATUS_OK,cudaSuccess,3u) == 0);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_IO_ERROR,SPARK_STATUS_OK,cudaSuccess,3u) == 0);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_OK,SPARK_STATUS_IO_ERROR,cudaSuccess,1u) == 0);
	assert(check_cache_transactions(SPARK_STATUS_OK,SPARK_STATUS_OK,SPARK_STATUS_OK,cudaErrorInvalidValue,1u) == 0);
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
	check_l2_prefetch_environment();
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
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	context.flags = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_KV_SHARD;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	context.flags = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_KV_SHARD | SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_INDEX_CP;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	context.tp_collective_identifier = 1u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_OK && state.kv_shard == 1u && state.index_cp == 1u);
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
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    wave = WAVE_TIMING.read_text().strip()
    assert wave.startswith("G5N-WAVE-TIMING ") and '"' not in wave and "\\" not in wave
    harness = HARNESS.replace('EXPECTED_WAVE_TIMING', '"' + wave + '"')
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
                        *["-I" + p for p in includes], "-DSPARK_BATCH_BUCKET=1024u", "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
                        '-DGLM5_NEXT_EXPERT_CODEC_NAME="fp8"', '-DGLM5_NEXT_CONTRACT_SHA256="fixture"',
                        str(source), "runtime/stage_module_common.c", "cache/kv_cache.c", "cache/kv_page_cache.c", "cache/kv_snapshot.c",
                        "-o", str(binary), *(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else [])], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS actual module context/cache ownership, global epoch independence, retained attach ownership, terminal CUDA receipts, step verdicts and route trace")


if __name__ == "__main__":
    main()
