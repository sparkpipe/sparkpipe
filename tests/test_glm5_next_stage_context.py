#!/usr/bin/env python3
"""Exercise the real module configurator without allocating CUDA state."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"
#include "cache/kv_page_store.c"
#define main SparkUnusedKvTestMain
#include "tests/test_kv_cache.c"
#undef main
static SparkGlm5NextModuleState state;
static uint32_t COPY_COUNT;

cudaError_t cudaMalloc(void **pointer,size_t bytes)
{
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

const char *cudaGetErrorString(cudaError_t error)
{
	(void)error;
	return("host test CUDA status");
}

cudaError_t cudaMemcpy(void *destination,const void *source,size_t bytes,cudaMemcpyKind kind)
{
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

SparkStatus SparkWeightdWorkerSubmit(SparkWeightdWorker *worker,SparkWeightdWorkFunction function,void *context)
{
	if ( worker != (SparkWeightdWorker *)(uintptr_t)1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( WORK_STATUS == SPARK_STATUS_OK )
	{
		COMPLETION_WORK = function;
		COMPLETION_CONTEXT = context;
	}
	return(WORK_STATUS);
}

static void observe_completion(void *context,const SparkModelDriverCompletion *completion)
{
	(void)context;
	COMPLETION_STATUS = completion->status;
}

static int32_t check_cache_transactions(void)
{
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
	state.resident_sequence_capacity = 4u;
	state.pages_per_sequence = 4u;
	state.kv_transactions = fixture.transactions;
	state.kv_lane_transactions = fixture.owners;
	state.kv_lane_physical_pages = fixture.physical;
	state.page_table = device_table;
	state.page_table_shadow = shadow;
	if ( pthread_mutex_init(&state.kv_mutex,0) != 0 )
		return(-20);
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
	state.completions[0].completion.status = SPARK_STATUS_IO_ERROR;
	state.completions[0].completion_function = observe_completion;
	state.slots[0].host_kv_access_error = errors;
	state.completion_worker = (SparkWeightdWorker *)(uintptr_t)1u;
	atomic_store(&state.slot_states[0],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	atomic_store(&state.lane_states[0],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	atomic_store(&state.lane_states[1],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	WORK_STATUS = SPARK_STATUS_BUSY;
	SparkGlm5NextCompleteAsync(&state.completions[0]);
	if ( COMPLETION_WORK != 0 || atomic_load(&state.slot_states[0]) != SPARK_STAGE_MODULE_SLOT_CLAIMED || fixture.owners[0].phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		return(-28);
	WORK_STATUS = SPARK_STATUS_OK;
	SparkGlm5NextCompleteAsync(&state.completions[0]);
	if ( COMPLETION_WORK == 0 || atomic_load(&state.slot_states[0]) != SPARK_STAGE_MODULE_SLOT_CLAIMED || fixture.owners[0].phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		return(-29);
	COMPLETION_WORK(COMPLETION_CONTEXT);
	if ( COMPLETION_STATUS != SPARK_STATUS_IO_ERROR || atomic_load(&state.slot_states[0]) != SPARK_STAGE_MODULE_SLOT_FREE || atomic_load(&state.lane_states[0]) != SPARK_STAGE_MODULE_SLOT_FREE || atomic_load(&state.lane_states[1]) != SPARK_STAGE_MODULE_SLOT_FREE || fixture.pages.cache.sequences[0].sequence_id != 0u || fixture.pages.cache.sequences[1].sequence_id != 0u || shadow[0] != UINT32_MAX )
		return(-27);
	pthread_mutex_destroy(&state.kv_mutex);
	memset(&state,0,sizeof(state));
	return(0);
}

SparkStatus SparkKvBackendInitialize(const SparkKvModelTable *table,SparkKvCacheArena *arena,SparkKvPageCache *cache,SparkKvPageStore *store)
{
	(void)arena;
	(void)cache;
	(void)store;
	assert(SparkKvPageStoreConfigurationIsValid(&table->page_store_config) != 0u);
	assert(table->page_store_config.transfer_capacity <= 2u);
	assert(table->page_store_config.page_bytes == table->page_store_config.staging_bytes);
	assert(table->arena_configuration.value_device_base == state.index_cache);
	assert(table->arena_configuration.value_block_stride_bytes == (uint64_t)state.index_layer_count * 64u * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
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
	uint32_t slots[303],width,row;
	atomic_uint *claims = state.lane_states;
	state.resident_sequence_capacity = 101u;
	for (row=0u; row<101u; row++)
		atomic_init(&claims[row],SPARK_STAGE_MODULE_SLOT_FREE);
	uint32_t ragged[8] = {5u,2u,9u,5u,2u,9u,5u,9u};
	batch.row_resident_slots = slots;
	for (width=1u; width<=101u; width++)
	{
		batch.active_sequence_count = width;
		batch.row_count = (width * 3u);
		for (row=0u; row<batch.row_count; row++)
			slots[row] = (width - 1u - (row % width));
		if ( SparkGlm5NextValidateRoundMajor(&state,&batch) != SPARK_STATUS_OK || SparkStageModuleIndexSetClaim(claims,101u,slots,width) != SPARK_STATUS_OK )
			return(-4);
		for (row=0u; row<batch.row_count; row+=width)
			if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,row) != width )
				return(-1);
		SparkStageModuleIndexSetRelease(claims,101u,slots,width);
	}
	batch.row_resident_slots = ragged;
	batch.active_sequence_count = 3u;
	batch.row_count = 8u;
	if ( SparkGlm5NextValidateRoundMajor(&state,&batch) != SPARK_STATUS_OK || SparkStageModuleIndexSetClaim(claims,101u,ragged,3u) != SPARK_STATUS_OK )
		return(-5);
	if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,0u) != 3u || SparkGlm5NextRoundMajorWaveRows(&state,&batch,3u) != 3u || SparkGlm5NextRoundMajorWaveRows(&state,&batch,6u) != 2u )
		return(-2);
	if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,8u) != 0u || SparkGlm5NextRoundMajorWaveRows(&state,0,0u) != 0u )
		return(-3);
	SparkStageModuleIndexSetRelease(claims,101u,ragged,3u);
	if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,0u) != 0u )
		return(-6);
	ragged[6] = 9u;
	ragged[7] = 5u;
	if ( SparkGlm5NextValidateRoundMajor(&state,&batch) == SPARK_STATUS_OK )
		return(-7);
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
		state.kv_backing_directory = "/unused-host-fixture";
		if ( SparkGlm5NextAllocateCaches(&state) != SPARK_STATUS_PENDING || state.kda_layer_count != 3u )
			return(-8);
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
		free_cache_fixture();
	}
	return(0);
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
	state.page_count = 5u;
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

int32_t main(void)
{
	SparkGlm5NextResidentDecodeStageNodeContext context = {0};
	SparkFirmwareModuleConfiguration configuration = {0};
	SparkFirmwareModuleHostServices services = {0};
	const char *path = 0;
	uint32_t first[4] = {0u,12u,23u,34u},counts[4] = {12u,11u,11u,11u},stage;
	int32_t status = check_cache_transactions();
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
	check_small_kv();
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
	for (stage=0u; stage<4u; stage++)
	{
		context.stage_index = stage;
		context.first_layer_index = first[stage];
		context.layer_count = counts[stage];
		assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_OK);
		assert(state.stage_count == 4u && state.stage_index == stage);
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
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "probe.c", Path(directory) / "probe"
        source.write_text(HARNESS)
        includes = [".", "include", "tests/cuda_stub", "model-families/common/include",
                    "model-families/glm5_next/include", "modules/glm5_next_resident_decode_stage/include",
                    "modules/glm5_next_resident_decode_stage/source"]
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O2", "-ffunction-sections", "-fdata-sections",
                        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                        *["-I" + p for p in includes], "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
                        '-DGLM5_NEXT_EXPERT_CODEC_NAME="fp8"', '-DGLM5_NEXT_CONTRACT_SHA256="fixture"',
                        str(source), "runtime/stage_module_common.c", "cache/kv_cache.c", "cache/kv_page_cache.c",
                        "-o", str(binary)], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS actual module context, cache transaction ownership, release, physical mapping and unchanged-map upload suppression")


if __name__ == "__main__":
    main()
