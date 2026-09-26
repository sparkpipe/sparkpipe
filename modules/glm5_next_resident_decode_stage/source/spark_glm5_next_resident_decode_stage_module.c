#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include "sparkpipe/spark_error_site.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include "sparkpipe/spark_tp_chain_ordinal.h"

#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_glm5_next_model.h"
#define SPARK_FAMILY_CAMEL Glm5Next
#define SPARK_FAMILY_UPPER GLM5_NEXT
#define SPARK_FAMILY_LOWER glm5_next

#include "sparkpipe/family/spark_family.h"

_Static_assert((uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION *
    SPARK_GLM5_NEXT_MODEL_HC_MULT * 2u <=
    SPARK_WEIGHTD_MESH_ROW_BYTES_MAX,
    "widest model row must fit the mesh row law");

#define SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE 4u
#define SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE 8u
static int SparkGlm5NextProbeEnabled(void)
{
	static int probe_enabled = -1;
	if ( probe_enabled < 0 )
		probe_enabled = getenv("SPARK_GLM5_NEXT_PROBE") != 0 ? 1 : 0;
	return(probe_enabled);
}
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_glm5_next_kv_geometry.h"
#include "sparkpipe/spark_glm5_next_index_cp.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_latency_histogram.h"
#include "spark_glm5_next_resident_decode_stage_internal.h"
#include "spark_glm5_next_stagepack_format.h"

#ifndef GLM5_NEXT_EXPERT_WEIGHT_CODEC
#error "GLM5_NEXT_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef GLM5_NEXT_CONTRACT_SHA256
#error "GLM5_NEXT_CONTRACT_SHA256 must identify the exact model package contract"
#endif

#define SPARK_GLM5_NEXT_MODULE_TAG "glm5_next_stage"
#define SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT 2048u
#define SPARK_GLM5_NEXT_NO_INDEX_ORDINAL UINT32_MAX
#define SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT 6u

_Static_assert(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT <= SPARK_WEIGHTD_WORK_QUEUE_CAPACITY,"completion worker must hold one job per occupied slot");

#if SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES != \
	( SPARK_GLM5_NEXT_KV_ARENA_KV_HEAD_COUNT * \
	  SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM * \
	  SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR )
#error "glm5_next KV slot bytes and arena block geometry disagree"
#endif

typedef struct SparkGlm5NextPackRange
{
	uint64_t offset;
	uint64_t bytes;
} SparkGlm5NextPackRange;

typedef struct SparkGlm5NextModuleState SparkGlm5NextModuleState;

static uint64_t SparkGlm5NextNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

typedef struct SparkGlm5NextCompletionOverflow
{
	SparkWeightdWorkFunction function;
	void *context;
} SparkGlm5NextCompletionOverflow;

#define SPARK_GLM5_NEXT_OVERFLOW_POOL 	(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT * 2u)

static void SparkGlm5NextDrainParkedCompletions(
	SparkGlm5NextModuleState *state);
static int SparkGlm5NextBoundedStreamSync(SparkGlm5NextModuleState *state,void *stream,uint64_t timeout_ns);
static void SparkGlm5NextScheduleCompletionWork(SparkGlm5NextModuleState *state,SparkWeightdWorkFunction function,void *context);

typedef struct SparkGlm5NextWaveTiming
{
	SparkLatencyHistogram idle;
	SparkLatencyHistogram pre;
	SparkLatencyHistogram key;
	SparkLatencyHistogram gpu;
	SparkLatencyHistogram post;
	uint64_t window_ns;
	uint64_t delivered_ns;
	uint64_t waves;
	uint64_t rows;
	uint64_t retries;
	uint64_t source_wait_ns;
	uint64_t peer_wait_ns;
	uint64_t copy_ns;
	uint64_t combine_ns;
	uint64_t worst_ns;
	uint64_t worst_request;
	uint64_t worst_epoch[2];
	uint64_t worst_part_ns[5];
} SparkGlm5NextWaveTiming;

typedef struct SparkGlm5NextAsyncCompletion
{
	SparkGlm5NextModuleState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t lane_indices[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	SparkGlm5NextStateCapture *state_capture;
	uint32_t finish_retries;
	uint32_t burst_token_count;
	uint64_t mtp_cache_extra;
	uint64_t chain_start_ns;
	uint64_t attempt_ns;
	uint64_t keyed_ns;
	uint64_t launched_ns;
	uint64_t callback_ns;
	uint64_t epoch[2];
	uint32_t retries;
	uint32_t mtp_draft_tokens[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH];
	SparkModelDriverCompletion completion;
} SparkGlm5NextAsyncCompletion;

struct SparkGlm5NextModuleState
{
	SparkStageModuleLedger ledger;
	SparkWeightdWorker *completion_worker;
	SparkWeightdLazyPack *lazy_pack;
	_Atomic(void *) lazy_retained[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t expert_weight_codec;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t tp_collective_disabled;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t execution_row_capacity;
	uint32_t decode_split_context_threshold;
	uint32_t pages_per_sequence;
	uint32_t page_count;
	uint32_t physical_page_count;
	uint32_t index_layer_count;
	uint32_t multiprocessor_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	void *execution_stream;
	SparkStageModuleCudaWait stream_wait;
	char model_revision[SPARK_GLM5_NEXT_STAGEPACK_MODEL_REVISION_BYTES];
	SparkGlm5NextLayerWeights layers[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t index_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kv_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kda_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kv_layer_count;
	uint32_t kda_layer_count;
	uint8_t *kda_state_pools;
	uint64_t kda_state_layer_stride_bytes;
	uint8_t *kda_window_pools;
	uint8_t *kda_q_window_pool;
	uint8_t *kda_k_window_pool;
	uint8_t *kda_v_window_pool;
	uint64_t kda_window_layer_stride_bytes;
	uint32_t *kda_state_index_device;
	uint32_t *kda_state_index_host;
	uint64_t layer_seen[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t global_seen;
	uint64_t mtp_seen;
	uint32_t pack_has_mtp;
	SparkGlm5NextLayerWeights mtp_layer;
	const void *mtp_eh_proj_bf16;
	const void *mtp_enorm_bf16;
	const void *mtp_hnorm_bf16;
	const void *mtp_shared_norm_bf16;
	uint32_t mtp_enabled;
	uint32_t index_cp;
	uint16_t *mtp_lane_hidden_bf16;
	uint8_t *mtp_lane_armed;
	uint64_t kda_replay_layer_bytes;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	uint8_t *head_certified_fp8_payload;
	float *head_certified_fp8_scale_f32;
	float *head_certified_fp8_norm_f32;
	uint64_t expert_pin_leases[32];
	uint8_t expert_pin_phases[32];
	uint32_t expert_pin_lease_count;
	uint32_t expert_pin_key_count;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint8_t *index_cache;
	uint64_t index_layer_stride_bytes;
	uint32_t *page_table;
	uint32_t *page_table_shadow;
	SparkKvCacheArena kv_arena;
	SparkKvPageCache kv_page_cache;
	SparkKvPageStore kv_page_store;
	SparkKvPageStore recurrent_store;
	uint8_t *recurrent_staging;
	uint64_t recurrent_page_bytes;
	SparkKvCacheBlock *kv_blocks;
	uint32_t *kv_resident_slot_logical_block_indices;
	SparkKvPageCacheEntry *kv_entries;
	SparkKvPageCacheSequence *kv_sequences;
	uint32_t *kv_hash_bucket_heads;
	uint32_t *kv_entry_indices_by_logical_page;
	uint8_t *kv_page_staging;
	uint32_t *kv_lane_logical_pages;
	uint32_t *kv_lane_physical_pages;
	SparkKvLaneTransaction *kv_lane_transactions;
	SparkKvLaneTransactions kv_transactions;
	pthread_mutex_t kv_mutex;
	uint32_t kv_mutex_initialized;
	uint64_t control_generation;
	uint64_t reset_generation;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	char kv_backing_default[256];
	SparkGlm5NextExecutionSlot slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm5NextAsyncCompletion completions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	volatile uint64_t slot_alive_ns[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm5NextCompletionOverflow overflow_pool[SPARK_GLM5_NEXT_OVERFLOW_POOL];
	uint32_t overflow_parked_count;
	pthread_mutex_t completion_queue_lock;
	uint32_t completion_queue_lock_initialized;
	atomic_uint slot_states[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_uchar lane_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_sequence_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_next_positions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong host_callback_completion_count;
	SparkTpDeviceCollective tp_device_collective;
	SparkTpDeviceCollective tp_device_collective_hc;
	uint32_t tp_device_collective_hc_initialized;
	uint32_t tp_device_collective_initialized;
	_Atomic(uint32_t) tp_chain_active;
	uint32_t tp_lane;
	SparkWeightdClient *lane_client;
	atomic_uint terminal_status;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_credit_binding_count;
	void *tp_credit_send_bf16;
	void *tp_credit_receive_bf16;
	void *tp_host_credit_send_bf16;
	void *tp_host_credit_receive_bf16;
	SparkTpDeviceCollectiveCreditBinding tp_hc_credit_bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_hc_credit_binding_count;
	void *tp_hc_credit_send_bf16;
	void *tp_hc_credit_receive_bf16;
	void *tp_hc_host_credit_send_bf16;
	void *tp_hc_host_credit_receive_bf16;
	atomic_ullong nccl_next_ordinal;
	uint32_t graph_path_enabled;
	uint32_t graph_arrival_dumped;
	uint32_t experts_warm;
	uint64_t degrade_graph_fallback;
	uint64_t degrade_covered_abandon;
	uint64_t degrade_graph_disabled;
	uint64_t degrade_graph_stuck;
	uint32_t graph_record_limit;
	uint32_t graph_gate_printed;
	uint32_t graph_record_ops;
	uint32_t graph_record_stop;
	uint64_t chain_stage_ns[8u];
	uint64_t chain_profile_last_ns;
	SparkGlm5NextWaveTiming wave_timing;
	uint64_t wave_attempt_request;
	uint64_t wave_attempt_ns;
	uint32_t wave_attempt_retries;
	uint32_t chain_profile_stage;
	uint32_t rs_taken;
	uint32_t rs_hit;
	uint32_t hbound_probes;
	uint32_t decode_cover_words;
	uint32_t *decode_cover_host;
	uint32_t *decode_cover_device;
	uint32_t *decode_miss_host;
	const uint8_t *decode_lease_base_saved;
	atomic_ullong nccl_next_ordinal_hc;
};

static SparkStatus SparkGlm5NextModuleConfigure(
	SparkGlm5NextModuleState *state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path)
{
	const SparkGlm5NextResidentDecodeStageNodeContext *context;
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 || host_services->node_context == 0 || host_services->execution_stream == 0 || host_services->kv_logical_page_capacity == 0u || host_services->kv_physical_page_capacity == 0u || host_services->kv_physical_page_capacity > host_services->kv_logical_page_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm5NextResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( SparkGlm5NextResidentDecodeStageSpanIsValid(context->stage_count,context->stage_index,context->first_layer_index,context->layer_count) == 0u || context->layer_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE || context->expert_weight_codec != GLM5_NEXT_EXPERT_WEIGHT_CODEC || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions == 0u || context->max_sequence_positions > SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS || context->execution_row_capacity == 0u || context->execution_row_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || context->execution_row_capacity > SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS || context->decode_split_context_threshold > context->max_sequence_positions || context->tp_degree == 0u || context->tp_rank >= context->tp_degree || (context->flags & ~SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || context->model_revision == 0 || context->model_revision[0] == '\0' || strlen(context->model_revision) >= sizeof(state->model_revision) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightCodecIsKnown(context->expert_weight_codec) == 0u || context->expert_weight_codec == SPARK_WEIGHT_CODEC_BF16 )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP) != 0u && (context->stage_index + 1u) != context->stage_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->tp_degree != 1u && (SPARK_GLM5_NEXT_MODEL_HEAD_COUNT % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->stage_count = context->stage_count;
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->expert_weight_codec = context->expert_weight_codec;
	state->tp_degree = context->tp_degree;
	state->tp_rank = context->tp_rank;
	state->kv_backing_directory = context->kv_backing_directory;
	state->kv_backing_maximum_bytes = context->kv_backing_maximum_bytes;
	state->tp_collective_disabled = context->tp_collective_identifier == 0u ? 1u : 0u;
	state->resident_sequence_capacity = context->resident_sequence_capacity;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->max_sequence_positions = context->max_sequence_positions;
	state->decode_split_context_threshold = context->decode_split_context_threshold;
	state->execution_row_capacity = context->execution_row_capacity;
	state->page_count = host_services->kv_logical_page_capacity;
	state->physical_page_count = host_services->kv_physical_page_capacity;
	state->owns_embedding = context->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = context->stage_index + 1u == context->stage_count ? 1u : 0u;
	state->mtp_enabled = (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP) != 0u ? 1u : 0u;
	state->index_cp = (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_INDEX_CP) != 0u ? 1u : 0u;
	if ( state->index_cp != 0u && (context->tp_degree < 2u || SparkGlm5NextIndexCpFits(context->max_sequence_positions,context->tp_degree,context->execution_row_capacity) == 0u) )
	{
		fprintf(stderr,"GLM index context parallel needs tp_degree >= 2 and max_sequence_positions whose local pool scores fit the gather (tp=%u positions=%u rows=%u)\n",context->tp_degree,context->max_sequence_positions,context->execution_row_capacity);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->execution_stream = host_services->execution_stream;
	(void)snprintf(state->model_revision,sizeof(state->model_revision),"%s",context->model_revision);
	*pack_path = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackValidateHeader(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackHeader *header,
	uint64_t file_bytes)
{
	uint64_t directory_bytes,directory_end;
	if ( state == 0 || header == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_GLM5_NEXT_STAGEPACK_MAGIC || header->format_version != SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES || header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( (header->flags & ~SPARK_GLM5_NEXT_STAGEPACK_KNOWN_FLAGS) != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( SparkGlm5NextStagePackHeaderTpDegree(header) == 0u || SparkGlm5NextStagePackHeaderTpDegree(header) != state->tp_degree || SparkGlm5NextStagePackHeaderTpRank(header) != state->tp_rank )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT || header->stage_count != state->stage_count || header->stage_index != state->stage_index || header->first_layer_index != state->first_layer_index || header->layer_count != state->layer_count || header->total_layer_count != SPARK_GLM5_NEXT_MODEL_LAYER_COUNT || header->hidden_dimension != SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION || header->vocab_count != SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT || header->routed_expert_count != SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != state->expert_weight_codec || header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes || header->directory_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || header->tensor_count > UINT64_MAX / header->directory_entry_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_bytes = (uint64_t)header->tensor_count * header->directory_entry_bytes;
	directory_end = header->directory_offset + directory_bytes;
	if ( directory_end < header->directory_offset || directory_end > file_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackValidateEntryGeometry(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackHeader *header,
	const SparkGlm5NextStagePackEntry *entry,
	SparkGlm5NextStagePackTensorShape *shape)
{
	uint64_t payload_bytes,scale_bytes,directory_end;
	uint32_t local_layer;
	if ( SparkGlm5NextStagePackExpectedShape(entry->tensor_kind,entry->layer_index,state->expert_weight_codec,state->tp_degree,shape) < 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
	{
		if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		{
			if ( (state->mtp_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
				SPARK_FAIL(SPARK_STATUS_DUPLICATE);
		}
		else
		{
			if ( entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			local_layer = entry->layer_index - state->first_layer_index;
			if ( (state->layer_seen[local_layer] & (UINT64_C(1) << entry->tensor_kind)) != 0u )
				SPARK_FAIL(SPARK_STATUS_DUPLICATE);
		}
	}
	else if ( (state->global_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
		SPARK_FAIL(SPARK_STATUS_DUPLICATE);
	if ( entry->payload_type != shape->payload_type || entry->weight_codec != shape->weight_codec || entry->scale_encoding != shape->scale_encoding || entry->group_count != shape->group_count || entry->rows != shape->rows || entry->columns != shape->columns )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkGlm5NextStagePackExpectedPayloadBytes(shape);
	scale_bytes = SparkGlm5NextStagePackExpectedScaleBytes(shape);
	if ( payload_bytes == 0u || entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_end = header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes);
	if ( entry->payload_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset > header->file_bytes || entry->payload_bytes > header->file_bytes - entry->payload_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->payload_offset < directory_end && header->directory_offset < entry->payload_offset + entry->payload_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes == 0u )
	{
		if ( entry->scale_offset != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	else if ( entry->scale_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->scale_offset > header->file_bytes || entry->scale_bytes > header->file_bytes - entry->scale_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	else if ( entry->scale_offset < directory_end && header->directory_offset < entry->scale_offset + entry->scale_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextPackMarkSeen(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackEntry *entry)
{
	if ( entry->layer_index == SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
		state->global_seen |= UINT64_C(1) << entry->tensor_kind;
	else if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		state->mtp_seen |= UINT64_C(1) << entry->tensor_kind;
	else
		state->layer_seen[entry->layer_index - state->first_layer_index] |= UINT64_C(1) << entry->tensor_kind;
}

static SparkStatus SparkGlm5NextPackAssignLayer(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextLayerWeights *weights,
	const SparkGlm5NextStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM: weights->attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A: weights->q_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A_NORM: weights->q_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_B: weights->q_b_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A: weights->kv_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A_NORM: weights->kv_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: weights->kv_b_key_transposed_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_VALUE: weights->kv_b_value_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT: weights->attn_output_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_POST_ATTN_NORM: weights->post_attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_Q: weights->index_q_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_K: weights->index_k_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_HEAD: weights->index_head_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: weights->index_norm_weight_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_BIAS: weights->index_norm_bias_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP: weights->dense_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN: weights->dense_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER: weights->router_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER_CORRECTION: weights->router_correction_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE: weights->expert_up_gate_payload = payload; weights->expert_up_gate_scale = scale; weights->expert_up_gate_payload_offset = entry->payload_offset; weights->expert_up_gate_scale_offset = entry->scale_offset; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN: weights->expert_down_payload = payload; weights->expert_down_scale = scale; weights->expert_down_payload_offset = entry->payload_offset; weights->expert_down_scale_offset = entry->scale_offset; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_GATE_UP: weights->shared_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN: weights->shared_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA: weights->kda_qkv_beta_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_GATE_DOWN: weights->kda_decay_gate_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_UP: weights->kda_decay_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_GATE_UP: weights->kda_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_Q_CONV: weights->kda_q_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_K_CONV: weights->kda_k_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_V_CONV: weights->kda_v_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_BIAS: weights->kda_decay_bias_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE: weights->kda_head_log_scale_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT_NORM: weights->kda_out_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT: weights->kda_out_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_FN: weights->hc_attn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_BASE: weights->hc_attn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_SCALE: weights->hc_attn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_FN: weights->hc_ffn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_BASE: weights->hc_ffn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_SCALE: weights->hc_ffn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_APE: weights->index_compress_ape_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE: weights->index_compress_gate_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_EH_PROJ: state->mtp_eh_proj_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_ENORM: state->mtp_enorm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_HNORM: state->mtp_hnorm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_SHARED_NORM: state->mtp_shared_norm_bf16 = payload; break;
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackAssign(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		return(SparkGlm5NextPackAssignLayer(state,&state->mtp_layer,entry,payload,scale));
	if ( entry->layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
		return(SparkGlm5NextPackAssignLayer(state,&state->layers[entry->layer_index - state->first_layer_index],entry,payload,scale));
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING: state->embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_bf16 = payload; return(SPARK_STATUS_OK);
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
}

typedef struct SparkGlm5NextManifestContext
{
	const SparkGlm5NextStagePackEntry *entries;
	uint32_t count;
} SparkGlm5NextManifestContext;

static int SparkGlm5NextT1Enabled(void);

static void CUDART_CB SparkGlm5NextCompleteAsync(void *context);
static SparkStatus SparkGlm5NextAllocateBytes(
	SparkGlm5NextModuleState *state,
	uint64_t count,
	uint64_t width,
	uint64_t element_bytes,
	void **pointer);

#include "sparkpipe/family/module/spark_module_glm5_next_lineage.h"

#include "sparkpipe/family/module/spark_module_glm5_next_laguna.h"

static SparkStatus SparkGlm5NextManifestCheck(const SparkWeightdManifest *manifest,void *opaque)
{
	const SparkGlm5NextManifestContext *context = (const SparkGlm5NextManifestContext *)opaque;
	const SparkGlm5NextStagePackEntry *entry;
	SparkStatus status;
	uint64_t expected = 0u;
	uint32_t index,plane;
	for (index=0u; index<context->count; index++)
	{
		entry = &context->entries[index];
		if ( entry->tensor_kind != SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE && entry->tensor_kind != SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN )
			continue;
		if ( entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		for (plane=0u; plane<2u; plane++)
		{
			status = SparkGlm5NextManifestPlane(manifest,entry,plane);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			expected += entry->group_count;
		}
	}
	return(expected == manifest->range_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SparkGlm5NextPinAllExperts(SparkGlm5NextModuleState *state);

static SparkStatus SparkGlm5NextExpertPoolBudget(uint64_t *bytes)
{
	if ( getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES") == 0 ||
	     getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES")[0] == '\0' )
	{
		fprintf(stderr,"SPARK_WEIGHTD_EXPERT_POOL_BYTES requires an explicit finite budget\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SparkStageModuleEnvironmentUnsigned64OrDefault(SPARK_GLM5_NEXT_MODULE_TAG,"SPARK_WEIGHTD_EXPERT_POOL_BYTES",1u,UINT64_MAX - 1u,1u,bytes));
}

static uint32_t SparkGlm5NextMeshAddressPending(const SparkGlm5NextModuleState *state)
{
	return state->tp_degree > 1u && state->tp_collective_disabled == 0u && state->lazy_pack != 0 && state->lazy_pack->attached.mesh_send_buffer_addr == 0u ? 1u : 0u;
}

static SparkStatus SparkGlm5NextLazyOpen(SparkGlm5NextModuleState *state,const char *path,uint64_t bytes,const SparkGlm5NextStagePackEntry *entries,uint32_t count)
{
	SparkWeightdLazyAttachRequest request;
	SparkGlm5NextManifestContext context = {entries,count};
	SparkStatus status;
	const char *digest;
	uint64_t spine_budget;
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
		return(status == SPARK_STATUS_BUSY ? SPARK_STATUS_UNSUPPORTED : status);
	if ( state->mtp_enabled != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	memset(&request,0,sizeof(request));
	digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	if ( digest == 0 || strlen(digest) != 64u || strlen(path) >= sizeof(request.pack_path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.identity.pack_sha256,digest,65u);
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s",SPARK_GLM5_NEXT_MODULE_TAG);
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",state->model_revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = bytes;
	request.identity.topology = state->tp_degree;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	status = SparkGlm5NextExpertPoolBudget(&request.expert_pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64OrDefault(SPARK_GLM5_NEXT_MODULE_TAG,"SPARK_WEIGHTD_SPINE_BUDGET_BYTES",1u,UINT64_MAX,UINT64_C(8589934592),&spine_budget);
	if ( status == SPARK_STATUS_OK )
	{
		uint32_t attach_attempt;
		for ( attach_attempt = 1u; attach_attempt <= 600u; attach_attempt++ )
		{
			status = SparkWeightdLazyPackCreateChecked(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,spine_budget,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,SparkGlm5NextManifestCheck,&context,&state->lazy_pack);
			if ( status == SPARK_STATUS_OK && SparkGlm5NextMeshAddressPending(state) != 0u )
			{
				if ( attach_attempt == 1u || (attach_attempt % 10u) == 0u )
					fprintf(stderr,"LAZY-ATTACH-MESH-PENDING n=%u\n",attach_attempt);
				status = SparkWeightdLazyPackDestroy(state->lazy_pack);
				state->lazy_pack = 0;
				if ( status != SPARK_STATUS_OK )
					SPARK_RETURN(status);
				status = SPARK_STATUS_BUSY;
			}
			if ( status == SPARK_STATUS_OK || state->lazy_pack != 0 )
				break;
			if ( attach_attempt == 1u || (attach_attempt % 10u) == 0u )
				fprintf(stderr,
					"LAZY-ATTACH-RETRY n=%u status=%d\n",
					attach_attempt,(int32_t)status);
			{
				struct timespec attach_pause = {0,1000000000u};
				nanosleep(&attach_pause,0);
			}
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		const char *pin_env = getenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
		if ( pin_env != 0 && pin_env[0] == '1' && state->lazy_pack != 0 &&
		     state->lazy_pack->map != 0 )
		{
			SparkStatus pin_status = SparkGlm5NextPinAllExperts(state);
			if ( pin_status != SPARK_STATUS_OK )
			{
				fprintf(stderr,"EXPERT-PIN-FAILED status=%d\n",(int)pin_status);
				SPARK_RETURN(pin_status);
			}
		}
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextPackLoadEntry(
	SparkGlm5NextModuleState *state,
	FILE *file,
	const SparkGlm5NextStagePackEntry *entry)
{
	void *payload,*scale;
	SparkStatus status;
	(void)file;
	payload = 0;
	scale = 0;
	if ( state->lazy_pack == 0 )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( entry->tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE || entry->tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN )
		return(SparkGlm5NextPackAssign(state,entry,0,0));
	status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->payload_offset,entry->payload_bytes,(const void **)&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->scale_offset,entry->scale_bytes,(const void **)&scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackAssign(state,entry,payload,scale);
	SPARK_RETURN(status);
}

static uint64_t SparkGlm5NextExpectedLayerMask(
	const SparkGlm5NextModuleState *state,
	uint32_t layer_index)
{
	SparkGlm5NextStagePackTensorShape shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind=SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM; kind<SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		if ( SparkGlm5NextStagePackExpectedShape(kind,layer_index,state->expert_weight_codec,state->tp_degree,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

static uint64_t SparkGlm5NextExpectedGlobalMask(const SparkGlm5NextModuleState *state)
{
	uint64_t mask;
	mask = 0u;
	if ( state->owns_embedding != 0u )
		mask |= UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		mask |= (UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM) | (UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD);
	return(mask);
}

static SparkStatus SparkGlm5NextPackValidateInventory(const SparkGlm5NextModuleState *state)
{
	uint64_t expected_mtp;
	uint32_t local;
	if ( state->global_seen != SparkGlm5NextExpectedGlobalMask(state) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	expected_mtp = state->pack_has_mtp != 0u ?
		SparkGlm5NextExpectedLayerMask(state,SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX) : 0u;
	if ( state->mtp_seen != expected_mtp )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (local=0u; local<state->layer_count; local++)
		if ( state->layer_seen[local] != SparkGlm5NextExpectedLayerMask(state,state->first_layer_index + local) )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

#include "sparkpipe/family/module/spark_module_pack_file_size.h"

static SparkStatus SparkGlm5NextPackLoad(
	SparkGlm5NextModuleState *state,
	const char *path)
{
	SparkGlm5NextStagePackHeader header;
	SparkGlm5NextStagePackEntry entries[SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT];
	SparkGlm5NextStagePackTensorShape shape;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	status = SparkGlm5NextPackFileSize(file,&file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateHeader(state,&header,file_bytes);
	if ( status == SPARK_STATUS_OK )
		state->pack_has_mtp = (header.flags & SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP) != 0u ? 1u : 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0]));
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
	{
		status = SparkGlm5NextPackValidateEntryGeometry(state,&header,&entries[index],&shape);
		if ( status == SPARK_STATUS_OK )
			SparkGlm5NextPackMarkSeen(state,&entries[index]);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateRanges(entries,header.tensor_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateInventory(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextLazyOpen(state,path,file_bytes,entries,header.tensor_count);
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
		status = SparkGlm5NextPackLoadEntry(state,file,&entries[index]);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateSlotHost(SparkGlm5NextExecutionSlot *slot)
{
	uint32_t *cursor;
	uint64_t rows,words,bytes;
	cudaError_t error;
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	rows = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	words = (rows * (4u + sizeof(SparkRowSampling) / sizeof(uint32_t))) + SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT +
	    SPARK_GLM5_NEXT_MODEL_LAYER_COUNT *
	        (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u);
	bytes = words * sizeof(uint32_t);
	error = cudaHostAlloc(&slot->host_staging,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"host_staging"));
	memset(slot->host_staging,0,bytes);
	slot->host_row_sampling = (SparkRowSampling *)slot->host_staging;
	cursor = (uint32_t *)(slot->host_row_sampling + rows);
	slot->host_token_ids = cursor;
	cursor += rows;
	slot->host_resident_slots = cursor;
	cursor += rows;
	slot->host_positions = cursor;
	cursor += rows;
	slot->host_output_token_ids = cursor;
	cursor += rows;
	slot->host_kv_access_error = cursor;
	cursor += SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT;
	slot->host_group_row_offset = cursor;
	bytes = ((rows + 1u) * 2u + rows) * sizeof(uint32_t);
	error = cudaHostAlloc((void **)&slot->host_run_begin,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
	{
		(void)cudaFreeHost(slot->host_staging);
		slot->host_staging = 0;
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"host_run_staging"));
	}
	memset(slot->host_run_begin,0,bytes);
	cursor = (uint32_t *)slot->host_run_begin;
	cursor += rows + 1u;
	slot->host_run_state_index = cursor;
	cursor += rows;
	slot->host_run_row_indices = cursor;
	error = cudaEventCreateWithFlags((cudaEvent_t *)&slot->route_ready_event,cudaEventDisableTiming);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"route_ready_event"));
}

static void SparkGlm5NextGraphDestroyAll(SparkGlm5NextExecutionSlot *slot)
{
	uint32_t index;
	for ( index = 0u; index < SPARK_GLM5_NEXT_GRAPH_ROWS_MAX; index++ )
	{
		if ( slot->graph_exec_rows[index] != 0 )
			(void)cudaGraphExecDestroy((cudaGraphExec_t)slot->graph_exec_rows[index]);
		slot->graph_exec_rows[index] = 0;
		slot->graph_bound_rows[index] = 0u;
	}
	slot->graph_failed_rows = 0u;
	slot->graph_exec_a = 0;
}

static void SparkGlm5NextReleaseSlotHost(SparkGlm5NextModuleState *state)
{
	uint32_t index;
	if ( state == 0 )
		return;
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		if ( state->slots[index].route_ready_event != 0 )
			(void)cudaEventDestroy((cudaEvent_t)state->slots[index].route_ready_event);
		state->slots[index].route_ready_event = 0;
		state->slots[index].route_recorded = 0u;
		SparkGlm5NextGraphDestroyAll(&state->slots[index]);
		if ( state->slots[index].host_staging != 0 )
			(void)cudaFreeHost(state->slots[index].host_staging);
		state->slots[index].host_staging = 0;
		if ( state->slots[index].host_run_begin != 0 )
			(void)cudaFreeHost(state->slots[index].host_run_begin);
		state->slots[index].host_run_begin = 0;
		state->slots[index].host_run_state_index = 0;
		state->slots[index].host_run_row_indices = 0;
	}
}

static SparkStatus SparkGlm5NextAllocateSlotMetadata(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	SparkStatus status;
	status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->token_ids);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->resident_slots);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(SparkRowSampling),(void **)&slot->row_sampling);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity + 1u,1u,sizeof(uint32_t),(void **)&slot->run_begin);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->run_state_index);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->run_row_indices);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->resident_sequence_capacity,1u,sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_tile_prefix);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t) * 6u,1u,&slot->kv_access_error);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateSlotHidden(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_collapsed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_snapshot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_mean_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,sizeof(float),(void **)&slot->hc_mixes_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_pre_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_post_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_comb_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_QUERY_A_DIMENSION,(void **)&slot->q_compressed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_QUERY_B_DIMENSION,(void **)&slot->q_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,(void **)&slot->query_latent_bf16);
	slot->query_rope_bf16 = 0;
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_QUERY_DIMENSION,(void **)&slot->index_query_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_key_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_COUNT,(void **)&slot->index_head_weight_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION,(void **)&slot->index_packed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,sizeof(uint32_t),(void **)&slot->selected_pools);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,(void **)&slot->fused_qkvb_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->fused_decay_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->kda_decay_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->kda_gate_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,(void **)&slot->kda_beta_logit);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,(void **)&slot->kda_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,(void **)&slot->kda_decay_logit_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->kda_output_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,sizeof(float),(void **)&slot->kda_retention);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,sizeof(float),(void **)&slot->kda_write_gate);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_CACHE_TOKEN_ELEMENTS,(void **)&slot->kv_slot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,(void **)&slot->attention_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_VALUE_HEAD_DIMENSION,(void **)&slot->attention_value_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->attention_out_bf16);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateSlotMlp(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows,packed_rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	packed_rows = rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K;
	status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_GATE_UP_DIMENSION,(void **)&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_TOP_K * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,(void **)&slot->intermediate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,packed_rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->expert_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->shared_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,sizeof(float),(void **)&slot->router_logits_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,state->max_sequence_positions / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,sizeof(float),(void **)&slot->selection_scores_f32);
	if ( status == SPARK_STATUS_OK && state->index_cp != 0u ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS,sizeof(float),(void **)&slot->index_local_scores_f32);
	if ( status == SPARK_STATUS_OK && state->index_cp != 0u ) status = SparkGlm5NextAllocateBytes(state,rows * state->tp_degree,SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS,sizeof(float),(void **)&slot->index_gathered_scores_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / state->tp_degree),SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS,sizeof(float),(void **)&slot->attention_split_partials_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH,sizeof(uint32_t),(void **)&slot->selected_positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_expert);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(float),(void **)&slot->route_weight);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_source_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_packed_row);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w1);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w2);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateSlotHead(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows,tiles;
	SparkStatus status;
	rows = state->execution_row_capacity;
	tiles = SparkCeilDivU64(SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,SPARK_GLM5_NEXT_HEAD_TILE);
	status = SparkGlm5NextAllocateBytes(state,rows,tiles,sizeof(float),(void **)&slot->head_candidate_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,tiles,sizeof(uint32_t),(void **)&slot->head_candidate_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(uint32_t),(void **)&slot->output_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(float),(void **)&slot->output_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		uint64_t shard_rows = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
		status = SparkGlm5NextAllocateBytes(state,1u,SparkHeadCertifiedFp8ScratchBytes(shard_rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION),1u,(void **)&slot->head_certified_scratch);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextAllocateBytes(state,1u,SparkHeadCertifiedFp8CandidateBytes(shard_rows),1u,(void **)&slot->head_certified_candidates);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextAllocateBytes(state,1u,1u,sizeof(uint32_t),(void **)&slot->head_screened_count);
	}
	SPARK_RETURN(status);
}

static void SparkGlm5NextShareSlotDevice(SparkGlm5NextExecutionSlot *slot,const SparkGlm5NextExecutionSlot *shared)
{
	SparkGlm5NextExecutionSlot host;
	host = *slot;
	*slot = *shared;
	slot->stream = host.stream;
	slot->route_ready_event = host.route_ready_event;
	slot->route_recorded = 0u;
	slot->host_staging = host.host_staging;
	slot->host_row_sampling = host.host_row_sampling;
	slot->host_token_ids = host.host_token_ids;
	slot->host_resident_slots = host.host_resident_slots;
	slot->host_positions = host.host_positions;
	slot->host_output_token_ids = host.host_output_token_ids;
	slot->host_kv_access_error = host.host_kv_access_error;
	slot->host_group_row_offset = host.host_group_row_offset;
	slot->host_run_begin = host.host_run_begin;
	slot->host_run_state_index = host.host_run_state_index;
	slot->host_run_row_indices = host.host_run_row_indices;
}

static SparkStatus SparkGlm5NextAllocateSlots(SparkGlm5NextModuleState *state)
{
	uint32_t index;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		state->slots[index].stream = state->execution_stream;
		status = SparkGlm5NextAllocateSlotHost(&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index != 0u )
			SparkGlm5NextShareSlotDevice(&state->slots[index],&state->slots[0]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotMetadata(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotHidden(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotMlp(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotHead(state,&state->slots[index]);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateMtp(SparkGlm5NextModuleState *state)
{
	SparkGlm5NextKdaReplayLayout layout;
	uint64_t kv_pool_bytes,index_pool_bytes,replay_bytes,steps_bytes,conv_bytes;
	uint32_t index,rank_heads,step;
	SparkStatus status;
	if ( state->mtp_enabled == 0u )
		return(SPARK_STATUS_OK);
	if ( state->execution_row_capacity < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	rank_heads = SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT / state->tp_degree;
	layout = SparkGlm5NextKdaReplayLayoutFor(rank_heads,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u);
	state->kda_replay_layer_bytes = layout.layer_bytes;
	kv_pool_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	index_pool_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u * SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT;
	replay_bytes = layout.layer_bytes * state->kda_layer_count;
	steps_bytes = (uint64_t)state->kda_layer_count * (SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u) * SPARK_GLM5_NEXT_MTP_REPLAY_STEP_BYTES;
	conv_bytes = (uint64_t)(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u) * rank_heads * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION * SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES;
	state->mtp_lane_armed = (uint8_t *)calloc(state->resident_sequence_capacity,1u);
	if ( state->mtp_lane_armed == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextAllocateBytes(state,state->resident_sequence_capacity,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,sizeof(uint16_t),(void **)&state->mtp_lane_hidden_bf16);
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		SparkGlm5NextExecutionSlot *slot = &state->slots[index];
		status = SparkGlm5NextAllocateRows(state,1u,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->mtp_hidden_bf16);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,1u,2u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->mtp_concat_bf16);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,kv_pool_bytes,1u,(void **)&slot->mtp_kv_pool);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,index_pool_bytes,1u,(void **)&slot->mtp_index_pool);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,sizeof(uint32_t),1u,(void **)&slot->mtp_positions);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,sizeof(uint32_t),1u,(void **)&slot->mtp_context);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_page_table);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_sequence);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_committed);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,steps_bytes,1u,&slot->mtp_replay_steps);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,conv_bytes,1u,(void **)&slot->mtp_conv_scratch);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,replay_bytes,1u,(void **)&slot->kda_replay_pool);
		if ( status != SPARK_STATUS_OK )
			break;
		{
			uint32_t meta[2u * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH];
			cudaError_t error;
			for ( step = 0u; step < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH; ++step )
			{
				meta[step] = step;
				meta[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + step] = step + 1u;
			}
			error = cudaMemcpy(slot->mtp_positions,meta,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error == cudaSuccess )
				error = cudaMemcpy(slot->mtp_context,meta + SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error == cudaSuccess )
				error = cudaMemset(slot->mtp_page_table,0,sizeof(uint32_t));
			if ( error == cudaSuccess )
				error = cudaMemset(slot->mtp_sequence,0,sizeof(uint32_t));
			status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"mtp_meta_init");
		}
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextBuildPageTable(SparkGlm5NextModuleState *state)
{
	uint64_t entries;
	SparkStatus status;
	cudaError_t error;
	state->pages_per_sequence = SparkCeilDivU32(state->max_sequence_positions,64u);
	if ( state->pages_per_sequence == 0u || state->resident_sequence_capacity > UINT32_MAX / state->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->page_count == 0u || state->physical_page_count == 0u || state->physical_page_count > state->page_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	entries = (uint64_t)state->resident_sequence_capacity * state->pages_per_sequence;
	state->page_table_shadow = (uint32_t *)malloc(entries * sizeof(uint32_t));
	if ( state->page_table_shadow == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(state->page_table_shadow,0xff,entries * sizeof(uint32_t));
	status = SparkGlm5NextAllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&state->page_table);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemset(state->page_table,0xff,entries * sizeof(uint32_t));
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"page_table");
	}
	SPARK_RETURN(status);
}

// Checkpoints pack all KDA layers, then all Q, K and V convolution layers.
// The caller owns the resident slot until the complete transfer succeeds.
static inline SparkStatus SparkGlm5NextRecurrentCopy(SparkGlm5NextModuleState *state,uint32_t direction,uint32_t slot,void *host,uint64_t bytes)
{
	SparkKvLayeredPageLayout layout;
	uint8_t *pools[4];
	uint64_t strides[4],payloads[4],total = 0u,offset = 0u;
	uint32_t part;
	SparkStatus status;
	if ( state == 0 || host == 0 || state->resident_sequence_capacity == 0u || state->kda_layer_count == 0u || slot >= state->resident_sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction != SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST && direction != SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	pools[0] = state->kda_state_pools;
	pools[1] = state->kda_q_window_pool;
	pools[2] = state->kda_k_window_pool;
	pools[3] = state->kda_v_window_pool;
	strides[0] = state->kda_state_layer_stride_bytes;
	strides[1] = strides[2] = strides[3] = state->kda_window_layer_stride_bytes;
	for (part=0u; part<4u; part++)
	{
		if ( pools[part] == 0 || strides[part] == 0u || strides[part] % state->resident_sequence_capacity != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( strides[part] > (UINTPTR_MAX - (uintptr_t)pools[part]) / state->kda_layer_count )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		payloads[part] = (strides[part] / state->resident_sequence_capacity) * state->kda_layer_count;
		if ( payloads[part] > UINT64_MAX - total )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		total += payloads[part];
	}
	if ( bytes != total || bytes > UINTPTR_MAX - (uintptr_t)host )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	layout.layer_count = state->kda_layer_count;
	layout.page_count = state->resident_sequence_capacity;
	for (part=0u; part<4u; part++)
	{
		layout.device_base = (uintptr_t)pools[part];
		layout.device_bytes = strides[part] * layout.layer_count;
		layout.layer_stride_bytes = strides[part];
		layout.layer_page_bytes = strides[part] / layout.page_count;
		status = SparkKvPageStoreCopyLayered(&layout,direction,slot,(uint8_t *)host + offset,payloads[part],SparkGlm5NextDevicePageCopy,state);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		offset += payloads[part];
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkGlm5NextModuleState *state;
	SparkKvLayeredPageLayout layout;
	uint64_t offset,packed_page_bytes;
	state = (SparkGlm5NextModuleState *)context;
	if ( state == 0 || state->physical_page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->index_layer_count != 0u && state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	layout.device_base = (uintptr_t)state->kv_cache;
	layout.layer_stride_bytes = state->kv_layer_stride_bytes;
	layout.layer_count = state->kv_layer_count;
	layout.page_count = state->physical_page_count;
	if ( device_address >= (uintptr_t)state->index_cache && device_address - (uintptr_t)state->index_cache < state->index_layer_stride_bytes * state->index_layer_count )
	{
		layout.device_base = (uintptr_t)state->index_cache;
		layout.layer_stride_bytes = state->index_layer_stride_bytes;
		layout.layer_count = state->index_layer_count;
	}
	if ( layout.layer_count == 0u || layout.layer_stride_bytes % layout.page_count != 0u || layout.layer_stride_bytes > UINT64_MAX / layout.layer_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	layout.device_bytes = layout.layer_stride_bytes * layout.layer_count;
	layout.layer_page_bytes = layout.layer_stride_bytes / layout.page_count;
	packed_page_bytes = layout.layer_page_bytes * layout.layer_count;
	if ( device_address < layout.device_base || device_address - layout.device_base >= layout.device_bytes || packed_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	// Arena block addresses name packed payloads. Translate their page index
	// to the native layer-major allocation before a device copy dereferences it.
	offset = device_address - layout.device_base;
	if ( offset % packed_page_bytes != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageStoreCopyLayered(&layout,direction,(uint32_t)(offset / packed_page_bytes),host_address,bytes,SparkGlm5NextDevicePageCopy,state));
}

static SparkStatus SparkGlm5NextBackingCapacity(SparkGlm5NextModuleState *state,uint64_t kv_page_bytes)
{
	uint64_t window_bytes,state_bytes,total;
	if ( state->page_count == 0u || state->resident_sequence_capacity == 0u || kv_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->recurrent_page_bytes = 0u;
	if ( state->kda_layer_count != 0u )
	{
		if ( state->kda_state_layer_stride_bytes % state->resident_sequence_capacity != 0u || state->kda_window_layer_stride_bytes % state->resident_sequence_capacity != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state_bytes = state->kda_state_layer_stride_bytes / state->resident_sequence_capacity;
		window_bytes = state->kda_window_layer_stride_bytes / state->resident_sequence_capacity;
		if ( state_bytes == 0u || window_bytes == 0u || window_bytes > (UINT64_MAX - state_bytes) / 3u )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		state_bytes += 3u * window_bytes;
		if ( state_bytes > (UINT64_MAX - kv_page_bytes) / state->kda_layer_count )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->recurrent_page_bytes = state_bytes * state->kda_layer_count;
	}
	total = kv_page_bytes + state->recurrent_page_bytes;
	if ( total > INT64_MAX / state->page_count || state->recurrent_page_bytes > SIZE_MAX / 2u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	total *= state->page_count;
	if ( state->kv_backing_maximum_bytes != 0u && state->kv_backing_maximum_bytes < total )
	{
		fprintf(stderr,"GLM cache backing budget insufficient: need %llu bytes for %u pages, configured %llu\n",(unsigned long long)total,state->page_count,(unsigned long long)state->kv_backing_maximum_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextRecurrentInitialize(SparkGlm5NextModuleState *state,const char *backing_path)
{
	SparkKvPageStoreConfiguration config = {0};
	SparkStatus status;
	if ( state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( state->recurrent_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaHostAlloc((void **)&state->recurrent_staging,2u * state->recurrent_page_bytes,cudaHostAllocPortable) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	config.logical_page_capacity = state->page_count;
	config.transfer_capacity = 1u;
	config.page_bytes = state->recurrent_page_bytes;
	config.maximum_backing_bytes = state->page_count * state->recurrent_page_bytes;
	config.backing_path = backing_path;
	// First half is caller-owned gather/scatter storage; the worker uses the second.
	config.staging_address = state->recurrent_staging + state->recurrent_page_bytes;
	config.staging_bytes = state->recurrent_page_bytes;
	status = SparkKvPageStoreInitialize(&state->recurrent_store,&config);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheAttachStateStore(&state->kv_page_cache,&state->recurrent_store);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextKvInitialize(SparkGlm5NextModuleState *state)
{
	SparkKvModelTable table;
	uint64_t block_bytes,index_block_bytes,payload_bytes;
	uint64_t lane_page_entries;
	SparkStatus status;
	if ( pthread_mutex_init(&state->completion_queue_lock,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->completion_queue_lock_initialized = 1u;
	if ( pthread_mutex_init(&state->kv_mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->kv_mutex_initialized = 1u;
	if ( state->kv_layer_count == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	block_bytes = (uint64_t)SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT *
		(uint64_t)state->kv_layer_count * SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM *
		SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR;
	index_block_bytes = (uint64_t)SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT * state->index_layer_count * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	if ( index_block_bytes > UINT64_MAX - block_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	payload_bytes = block_bytes + index_block_bytes;
	status = SparkGlm5NextBackingCapacity(state,payload_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	lane_page_entries = (uint64_t)state->resident_sequence_capacity *
		state->pages_per_sequence;
	state->kv_blocks = (SparkKvCacheBlock *)calloc(state->page_count,sizeof(*state->kv_blocks));
	state->kv_resident_slot_logical_block_indices = (uint32_t *)calloc(state->physical_page_count,sizeof(*state->kv_resident_slot_logical_block_indices));
	state->kv_entries = (SparkKvPageCacheEntry *)calloc(state->page_count,sizeof(*state->kv_entries));
	state->kv_sequences = (SparkKvPageCacheSequence *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_sequences));
	state->kv_hash_bucket_heads = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_hash_bucket_heads));
	state->kv_entry_indices_by_logical_page = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_entry_indices_by_logical_page));
	state->kv_page_staging = (uint8_t *)malloc((size_t)payload_bytes);
	state->kv_lane_logical_pages = (uint32_t *)calloc((size_t)lane_page_entries,sizeof(*state->kv_lane_logical_pages));
	state->kv_lane_transactions = (SparkKvLaneTransaction *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_transactions));
	if ( cudaHostAlloc((void **)&state->kv_lane_physical_pages,lane_page_entries * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->kv_blocks == 0 || state->kv_resident_slot_logical_block_indices == 0 || state->kv_entries == 0 || state->kv_sequences == 0 || state->kv_hash_bucket_heads == 0 || state->kv_entry_indices_by_logical_page == 0 || state->kv_page_staging == 0 || state->kv_lane_logical_pages == 0 || state->kv_lane_transactions == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_transactions.cache = &state->kv_page_cache;
	state->kv_transactions.lanes = state->kv_lane_transactions;
	state->kv_transactions.logical_pages = state->kv_lane_logical_pages;
	state->kv_transactions.physical_pages = state->kv_lane_physical_pages;
	state->kv_transactions.page_capacity = state->pages_per_sequence;

	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table.descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;
	SparkGlm5NextKvFillCapacityRequest(&table.capacity_request);
	table.capacity_request.layer_count = state->kv_layer_count;
	table.capacity_request.index_key_layer_count = state->index_layer_count;
	table.capacity_request.index_key_dimension = SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION;
	table.capacity_request.index_key_bytes_per_scalar = 2u;

	table.arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table.arena_configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table.arena_configuration.logical_block_count = state->page_count;
	table.arena_configuration.block_token_count = SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT;
	table.arena_configuration.resident_block_capacity = state->physical_page_count;
	table.arena_configuration.layer_count = state->kv_layer_count;
	table.arena_configuration.kv_head_count = SPARK_GLM5_NEXT_KV_ARENA_KV_HEAD_COUNT;
	table.arena_configuration.head_dim = SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM;
	table.arena_configuration.bytes_per_scalar = SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR;
	table.arena_configuration.key_device_base = state->kv_cache;
	table.arena_configuration.value_device_base = state->index_cache;
	table.arena_configuration.value_block_stride_bytes = index_block_bytes;
	table.arena_configuration.blocks = state->kv_blocks;
	table.arena_configuration.resident_slot_logical_block_indices = state->kv_resident_slot_logical_block_indices;
	table.arena_configuration.evict_function = SparkKvPageStoreWriteback;
	table.arena_configuration.evict_context = &state->kv_page_store;

	table.page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table.page_store_config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table.page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	table.page_store_config.logical_page_capacity = state->page_count;
	table.page_store_config.transfer_capacity = state->page_count < 2u ? state->page_count : 2u;
	table.page_store_config.page_bytes = payload_bytes;
	if ( state->kv_backing_directory != 0 && state->kv_backing_directory[0] != '\0' )
		table.page_store_config.backing_path = state->kv_backing_directory;
	else
	{
		(void)snprintf(state->kv_backing_default,sizeof(state->kv_backing_default),
			"/tmp/sparkpipe_glm5_next_kv_%s",state->model_revision);
		mkdir(state->kv_backing_default,0700);
		table.page_store_config.backing_path = state->kv_backing_default;
	}
	table.page_store_config.maximum_backing_bytes = state->page_count * payload_bytes;
	table.page_store_config.staging_address = state->kv_page_staging;
	table.page_store_config.staging_bytes = payload_bytes;
	table.page_store_config.copy_function = SparkGlm5NextPageCopy;
	table.page_store_config.copy_context = state;

	table.sequence_capacity = state->resident_sequence_capacity;
	table.entry_capacity = state->page_count;
	table.hash_bucket_count = state->page_count;
	table.entries = state->kv_entries;
	table.sequences = state->kv_sequences;
	table.hash_bucket_heads = state->kv_hash_bucket_heads;
	table.entry_indices_by_logical_page = state->kv_entry_indices_by_logical_page;
	table.model_id = "glm5_next";
	table.model_revision = state->model_revision;
	table.cache_layout_fingerprint = "kv-bf16-index-packed-layer-major-gather-v1";

	status = SparkKvBackendInitialize(&table,&state->kv_arena,&state->kv_page_cache,&state->kv_page_store);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->kv_arena.key_block_stride_bytes != block_bytes ||
		state->kv_arena.value_block_stride_bytes != index_block_bytes ||
		state->kv_arena.logical_block_count != state->page_count ||
		state->kv_arena.resident_block_capacity != state->physical_page_count ||
		state->kv_layer_stride_bytes == 0u ||
		block_bytes != ( state->kv_layer_stride_bytes /
				(uint64_t)state->physical_page_count ) *
			(uint64_t)state->kv_layer_count ||
		(uint64_t)state->physical_page_count * block_bytes !=
			state->kv_layer_stride_bytes * (uint64_t)state->kv_layer_count )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return(SparkGlm5NextRecurrentInitialize(state,table.page_store_config.backing_path));
}

static SparkStatus SparkGlm5NextAllocateCaches(SparkGlm5NextModuleState *state)
{
	uint64_t main_page_bytes,index_page_bytes;
	uint64_t main_total,index_total;
	uint32_t local;
	SparkStatus status;
	uint64_t kda_window_stride;
	uint64_t kda_total,window_total;
	state->index_layer_count = 0u;
	state->kv_layer_count = 0u;
	state->kda_layer_count = 0u;
	for (local=0u; local<state->layer_count; local++)
	{
		uint32_t layer = state->first_layer_index + local;
		state->index_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		state->kv_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		state->kda_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		if ( layer < SPARK_GLM5_NEXT_MODEL_LAYER_COUNT )
		{
			if ( SparkGlm5NextStagePackLayerIsDsa(layer) != 0u )
			{
				state->kv_ordinal_by_local_layer[local] = state->kv_layer_count++;
				state->index_ordinal_by_local_layer[local] = state->index_layer_count++;
			}
			else if ( SparkGlm5NextStagePackLayerIsKda(layer) != 0u )
				state->kda_ordinal_by_local_layer[local] = state->kda_layer_count++;
		}
	}
	status = SparkGlm5NextBuildPageTable(state);
	main_page_bytes = (uint64_t)64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	index_page_bytes = (uint64_t)64u *
		SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	// The model constant includes all three windows; each pool owns one rank-local window.
	kda_window_stride = (uint64_t)state->resident_sequence_capacity *
		(SPARK_GLM5_NEXT_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER / (3u * state->tp_degree));
	state->kv_layer_stride_bytes = (uint64_t)state->physical_page_count * main_page_bytes;
	state->index_layer_stride_bytes = (uint64_t)state->physical_page_count * index_page_bytes;
	state->kda_state_layer_stride_bytes = (uint64_t)state->resident_sequence_capacity *
		(SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER / state->tp_degree);
	state->kda_window_layer_stride_bytes = kda_window_stride;
	if ( status != SPARK_STATUS_OK || state->kv_layer_stride_bytes == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status);
	main_total = state->kv_layer_stride_bytes * (uint64_t)state->kv_layer_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,main_total,(void **)&state->kv_cache);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u )
	{
		if ( state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		index_total = state->index_layer_stride_bytes * state->index_layer_count;
		status = SparkStageModuleDeviceAllocate(&state->ledger,index_total,(void **)&state->index_cache);
	}
	kda_total = state->kda_state_layer_stride_bytes * (uint64_t)state->kda_layer_count;
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,kda_total,(void **)&state->kda_state_pools);
	window_total = kda_window_stride * 3u * (uint64_t)state->kda_layer_count;
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,window_total,(void **)&state->kda_window_pools);
		if ( status == SPARK_STATUS_OK )
		{
			state->kda_q_window_pool = state->kda_window_pools;
			state->kda_k_window_pool = state->kda_q_window_pool + kda_window_stride * (uint64_t)state->kda_layer_count;
			state->kda_v_window_pool = state->kda_k_window_pool + kda_window_stride * (uint64_t)state->kda_layer_count;
		}
	}
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
	{
		uint32_t sequence;
		cudaError_t error;
		state->kda_state_index_host = (uint32_t *)malloc((size_t)state->resident_sequence_capacity * sizeof(uint32_t));
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->resident_sequence_capacity * sizeof(uint32_t),(void **)&state->kda_state_index_device);
		if ( status == SPARK_STATUS_OK && state->kda_state_index_host != 0 )
		{
			for (sequence=0u; sequence<state->resident_sequence_capacity; sequence++)
				state->kda_state_index_host[sequence] = sequence;
			error = cudaMemcpy(state->kda_state_index_device,state->kda_state_index_host,(size_t)state->resident_sequence_capacity * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error != cudaSuccess )
				status = SPARK_STATUS_INTERNAL_ERROR;
		}
		else if ( state->kda_state_index_host == 0 )
		{
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextKvInitialize(state);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextTerminalFailure(
	SparkGlm5NextModuleState *state,SparkStatus status,const char *source)
{
	uint32_t expected = SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	if ( atomic_compare_exchange_strong_explicit(&state->terminal_status,
	        &expected,(uint32_t)status,memory_order_acq_rel,memory_order_acquire) )
		fprintf(stderr,"GLM engine terminal status=%u source=%s; full engine restart required\n",
		    (unsigned)status,source);
	return((SparkStatus)atomic_load_explicit(&state->terminal_status,memory_order_acquire));
}

static SparkStatus SparkGlm5NextWeightdHealth(SparkGlm5NextModuleState *state)
{
	SparkStatus status = (SparkStatus)atomic_load_explicit(
		&state->terminal_status,memory_order_acquire);
	uint32_t lane_dead,lazy_dead;
	if ( status != SPARK_STATUS_OK )
		return(status);
	lane_dead = state->lane_client != 0 && SparkWeightdClientAlive(state->lane_client) == 0u;
	lazy_dead = state->lazy_pack != 0 && state->lazy_pack->client != 0 &&
		SparkWeightdClientAlive(state->lazy_pack->client) == 0u;
	if ( lane_dead != 0u || lazy_dead != 0u )
		return(SparkGlm5NextTerminalFailure(state,SPARK_STATUS_IO_ERROR,
			lane_dead != 0u ? (lazy_dead != 0u ? "weightd-lane-and-lazy" : "weightd-lane") : "weightd-lazy"));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextAdmissionPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkStatus status;
	uint32_t lane,slot;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = state->control_generation != 0u &&
		state->control_generation != request->control_generation ?
		SPARK_STATUS_VALIDATION_FAILED :
		SparkKvLaneTransactionsAdmit(&state->kv_transactions,request);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"ADMIT9-MODULE request=%llu control_generation=%llu state_control=%llu reset_gen=%llu status=%u lanes=%u\n",
			(unsigned long long)request->request_id,
			(unsigned long long)request->control_generation,
			(unsigned long long)state->control_generation,
			(unsigned long long)state->reset_generation,
			(unsigned)status,(unsigned)request->cache_lane_count);
	if ( status == SPARK_STATUS_OK )
		state->control_generation = request->control_generation;
	if ( status == SPARK_STATUS_OK && (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		for (lane=0u; lane<request->cache_lane_count; lane++)
		{
			slot = request->cache_lanes[lane].resident_sequence_slot;
			atomic_store_explicit(&state->lane_bound[slot],0u,memory_order_release);
			if ( state->mtp_lane_armed != 0 )
				state->mtp_lane_armed[slot] = 0u;
		}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	decision->driver_dispatch_slot = (uint32_t)(request->request_id % state->pipeline_slot_count);
	decision->driver_dispatch_generation = request->control_generation;
	decision->driver_dispatch_cookie0 = request->transaction_id;
	decision->driver_dispatch_cookie1 = request->submission_id;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateRoundMajor(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageBatchView *batch)
{
	uint32_t ordinals[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t counts[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_rows[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkRowLayoutDirectLaneContext lanes;
	SparkStatus status;
	if ( state == 0 || batch == 0 || batch->row_count < batch->active_sequence_count || state->resident_sequence_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkRowLayoutDirectLaneMapInitialize(&lanes,ordinals,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SparkRowLayoutValidateRoundMajor(batch->row_count,batch->active_sequence_count,batch->row_resident_slots,SparkRowLayoutDirectLaneOrdinal,&lanes,counts,last_rows));
}

#include "sparkpipe/family/module/spark_module_load_sequence_continuity.h"

static SparkStatus SparkGlm5NextValidateSequenceContinuity(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	uint8_t *bound,
	uint64_t *sequence_ids,
	uint64_t *next_positions)
{
	uint8_t touched[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t position,sequence;
	uint32_t lane,row,slot;
	SparkStatus status;
	status = SparkGlm5NextLoadSequenceContinuity(state,batch,bound,sequence_ids,next_positions);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (row=0u; row<batch->row_count; row++)
	{
		slot = batch->row_resident_slots[row];
		status = SparkStageModuleIndexClaimOrdinal(state->lane_states,state->resident_sequence_capacity,slot,&lane);
		position = batch->row_positions[row];
		sequence = batch->row_sequence_ids[row];
		if ( status != SPARK_STATUS_OK || lane >= batch->active_sequence_count || batch->row_resident_slots[lane] != slot || position >= state->max_sequence_positions )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( position == 0u )
		{
			if ( touched[lane] != 0u )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			bound[lane] = 1u;
			sequence_ids[lane] = sequence;
			next_positions[lane] = 1u;
		}
		else
		{
			if ( bound[lane] == 0u || sequence_ids[lane] != sequence || next_positions[lane] != position )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			next_positions[lane] = position + 1u;
		}
		touched[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}

typedef struct SparkGlm5NextClaimedContinuityContext
{
	SparkGlm5NextModuleState *state;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkGlm5NextClaimedContinuityContext;

static SparkStatus SparkGlm5NextPrepareClaimedContinuity(void *prepare_context)
{
	SparkGlm5NextClaimedContinuityContext *context;
	SparkStatus status;
	context = (SparkGlm5NextClaimedContinuityContext *)prepare_context;
	if ( pthread_mutex_lock(&context->state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextValidateSequenceContinuity(context->state,context->batch,context->bound,context->sequence_ids,context->next_positions);
	(void)pthread_mutex_unlock(&context->state->kv_mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextValidateFrameBuffers(
	const SparkGlm5NextModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t row_count)
{
	const SparkModelDriverBuffer *buffer;
	if ( state->owns_final_head == 0u )
		return(frame->buffer_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count != 1u || frame->buffers == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	buffer = &frame->buffers[0];
	if ( buffer->flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || buffer->address == 0 || buffer->bytes < (uint64_t)row_count * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->mtp_enabled != 0u && row_count == 1u &&
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) == 0u &&
		buffer->bytes < (uint64_t)(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u) * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateStateCapture(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageFrameContext *context)
{
	const SparkGlm5NextStateCapture *capture = context->state_capture;
	uint64_t page_bytes,lane_bytes,hidden_bytes;
	if ( capture == 0 )
		return(SPARK_STATUS_OK);
	if ( capture->abi_version != SPARK_GLM5_NEXT_STATE_CAPTURE_ABI_VERSION || capture->descriptor_bytes != sizeof(*capture) )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( state->mtp_enabled != 0u || state->owns_final_head == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( capture->lanes == 0 || capture->logical_pages == 0 || capture->physical_pages == 0 || capture->payload == 0 || capture->lane_capacity < context->batch->active_sequence_count || capture->pages_per_lane_capacity < state->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	page_bytes = state->kv_arena.key_block_stride_bytes + state->kv_arena.value_block_stride_bytes;
	hidden_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	if ( page_bytes == 0u || state->pages_per_sequence > (UINT64_MAX - state->recurrent_page_bytes - hidden_bytes) / page_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	lane_bytes = state->pages_per_sequence * page_bytes + state->recurrent_page_bytes + hidden_bytes;
	if ( capture->payload_capacity / context->batch->active_sequence_count < lane_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateFrame(
	const SparkGlm5NextModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageFrameContext **context_out)
{
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	uint32_t expected_flags,prefill;
	uint64_t boundary_bytes,sideband_bytes;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context_out == 0 || frame->user_context == 0 || frame->execution_stream != state->execution_stream || frame->completion_function == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm5NextResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != sizeof(*context) || context->reserved0 != 0u || (context->flags & ~SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS) != 0u || context->batch == 0 )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	batch = context->batch;
	if ( batch->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != sizeof(*batch) || batch->row_count == 0u || batch->row_count > state->execution_row_capacity || batch->active_sequence_count == 0u || batch->active_sequence_count > state->resident_sequence_capacity || batch->row_resident_slots == 0 || batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill == 0u && batch->row_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count != batch->active_sequence_count || frame->new_token_count != batch->row_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (state->owns_embedding != 0u && batch->token_ids == 0) || (state->owns_final_head != 0u && batch->row_sampling == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	expected_flags = prefill != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	expected_flags |= state->owns_embedding == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u;
	expected_flags |= state->owns_final_head == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u;
	expected_flags |= SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT : 0u;
	expected_flags |= SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT : 0u;
	if ( context->flags != expected_flags )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	boundary_bytes = (uint64_t)batch->row_count * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	sideband_bytes = (uint64_t)batch->row_count * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	if ( (state->owns_embedding == 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < boundary_bytes)) || (state->owns_embedding != 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) || (state->owns_final_head == 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < boundary_bytes)) || (state->owns_final_head != 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u && (context->sideband_input == 0 || context->sideband_input_bytes < sideband_bytes)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) == 0u && (context->sideband_input != 0 || context->sideband_input_bytes != 0u)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u && (context->sideband_output == 0 || context->sideband_output_bytes < sideband_bytes)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) == 0u && (context->sideband_output != 0 || context->sideband_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextValidateRoundMajor(state,batch);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextValidateFrameBuffers(state,frame,batch->row_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextValidateStateCapture(state,context);
	*context_out = status == SPARK_STATUS_OK ? context : 0;
	SPARK_RETURN(status);
}

#define SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT 2u
#define SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS ((2u * SPARK_GLM5_NEXT_MODEL_LAYER_COUNT + 16u) * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT)
#define SPARK_GLM5_NEXT_TP_COLLECTIVE_HC_PORT_STRIDE 512u
#define SPARK_GLM5_NEXT_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES 65536u

typedef enum SparkGlm5NextChainStage
{
	SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN = 0,
	SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION,
	SPARK_GLM5_NEXT_CHAIN_STAGE_MLP,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP,
	SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD,
	SPARK_GLM5_NEXT_CHAIN_STAGE_FINISH,
	SPARK_GLM5_NEXT_CHAIN_STAGE_GATHER_INDEX
} SparkGlm5NextChainStage;

typedef struct SparkGlm5NextTpChain
{
	SparkGlm5NextModuleState *state;
	uint64_t created_ns;
	uint64_t last_advance_ns;
	uint64_t last_heartbeat_stage;
	SparkGlm5NextExecutionSlot *slot;
	uint32_t slot_index;
	SparkModelDriverFrame *frame;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	SparkGlm5NextCudaWave wave;
	uint32_t first_row;
	uint32_t wave_rows;
	uint32_t stage;
	uint32_t next_layer;
	uint32_t active;

	uint32_t spec_verify;
	uint32_t tp_op_index;
	uint32_t tp_hc_op_index;
	uint64_t expert_lease;
	uint32_t expert_lease_begun;
	uint32_t expert_lease_recorded;
	SparkStatus retained_status;
} SparkGlm5NextTpChain;

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status);
static void SparkGlm5NextTpChainFail(SparkGlm5NextTpChain *chain,SparkStatus status);
static void CUDART_CB SparkGlm5NextCompleteAsync(void *context);
static void CUDART_CB SparkGlm5NextMtpResolveHost(void *context);
static SparkStatus SparkGlm5NextEnqueueAsyncCompletion(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	uint32_t slot_index);

static SparkStatus SparkGlm5NextBuildWave(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkGlm5NextExecutionSlot *slot;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	SparkGlm5NextCudaWave *wave;
	uint32_t row,maximum_context;
	state = chain->state;
	slot = chain->slot;
	context = chain->context;
	wave = &chain->wave;
	maximum_context = 0u;
	for (row=0u; row<chain->wave_rows; row++)
		if ( slot->host_positions[chain->first_row + row] + 1u > maximum_context )
			maximum_context = slot->host_positions[chain->first_row + row] + 1u;
	memset(wave,0,sizeof(*wave));
	wave->stage_index = state->stage_index;
	wave->first_layer_index = state->first_layer_index;
	wave->layer_count = state->layer_count;
	wave->tp_degree = state->tp_degree;
	wave->tp_rank = state->tp_rank;
	wave->index_cp_degree = state->index_cp != 0u ? state->tp_degree : 1u;
	wave->row_count = chain->wave_rows;
	wave->commit = chain->spec_verify != 0u ? 0u : 1u;
	wave->mtp_verify = chain->spec_verify;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->mtp_layer_weights = state->pack_has_mtp != 0u ? &state->mtp_layer : 0;
	wave->mtp_eh_proj_bf16 = state->mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = state->mtp_enorm_bf16;
	wave->mtp_hnorm_bf16 = state->mtp_hnorm_bf16;
	wave->mtp_shared_norm_bf16 = state->mtp_shared_norm_bf16;
	wave->kda_replay_layer_bytes = state->kda_replay_layer_bytes;
	wave->maximum_context = maximum_context;
	wave->resident_sequence_capacity = state->resident_sequence_capacity;
	wave->max_sequence_positions = state->max_sequence_positions;
	wave->execution_row_capacity = state->execution_row_capacity;
	wave->pages_per_sequence = state->pages_per_sequence;
	wave->physical_page_count = state->physical_page_count;
	wave->owns_embedding = state->owns_embedding;
	wave->owns_final_head = state->owns_final_head;
	wave->sideband_input = SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index);
	wave->sideband_output = SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index);
	wave->boundary_row_offset = chain->first_row;
	wave->sideband_row_offset = chain->first_row;
	wave->host_token_ids = state->owns_embedding != 0u ? slot->host_token_ids + chain->first_row : 0;
	wave->host_resident_slots = slot->host_resident_slots + chain->first_row;
	wave->host_positions = slot->host_positions + chain->first_row;
	wave->host_row_sampling = state->owns_final_head != 0u ? slot->host_row_sampling + chain->first_row : 0;
	wave->sampled = state->owns_final_head != 0u ? slot->sampled : 0u;
	wave->hidden_input_bf16 = context->hidden_input_bf16;
	wave->hidden_output_bf16 = context->hidden_output_bf16;
	wave->sideband_input_u32 = context->sideband_input;
	wave->sideband_output_u32 = context->sideband_output;
	wave->host_output_token_ids = state->owns_final_head != 0u ? slot->host_output_token_ids + chain->first_row : 0;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->final_norm_bf16 = state->final_norm_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->head_certified_fp8_payload = state->head_certified_fp8_payload;
	wave->head_certified_fp8_scale_f32 = state->head_certified_fp8_scale_f32;
	wave->head_certified_fp8_norm_f32 = state->head_certified_fp8_norm_f32;
	wave->layers = state->layers;
	wave->lazy_experts = state->lazy_pack != 0 ? 1u : 0u;
	wave->slot = slot;
	wave->kv_cache = state->kv_cache;
	wave->kv_layer_stride_bytes = state->kv_layer_stride_bytes;
	wave->index_cache = state->index_cache;
	wave->index_layer_stride_bytes = state->index_layer_stride_bytes;
	wave->index_ordinal_by_local_layer = state->index_ordinal_by_local_layer;
	wave->kda_ordinal_by_local_layer = state->kda_ordinal_by_local_layer;
	wave->kda_state_pools = state->kda_state_pools;
	wave->kda_state_layer_stride_bytes = state->kda_state_layer_stride_bytes;
	wave->kda_q_window_pool = state->kda_q_window_pool;
	wave->kda_k_window_pool = state->kda_k_window_pool;
	wave->kda_v_window_pool = state->kda_v_window_pool;
	wave->kda_window_layer_stride_bytes = state->kda_window_layer_stride_bytes;
	wave->kda_state_index = state->kda_state_index_device;
	wave->kda_layer_count = state->kda_layer_count;
	wave->page_table = state->page_table;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->decode_split_context_threshold = state->decode_split_context_threshold;
	wave->attention_split_partials_f32 = slot->attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(
		state->execution_row_capacity,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / state->tp_degree);
	{
		uint32_t lane,ordinals[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT],cursor[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
		uint32_t lanes = chain->batch->active_sequence_count;
		SparkRowLayoutDirectLaneContext map;
		SparkStatus status = SparkRowLayoutDirectLaneMapInitialize(&map,ordinals,state->resident_sequence_capacity,wave->host_resident_slots,lanes);
		if ( status == SPARK_STATUS_OK )
			status = SparkRowLayoutGroupRows(chain->wave_rows,lanes,wave->host_resident_slots,SparkRowLayoutDirectLaneOrdinal,&map,slot->host_run_begin,slot->host_run_row_indices,cursor);
		if ( status != SPARK_STATUS_OK )
			return(status);
		for (lane=0u; lane<lanes; lane++)
			slot->host_run_state_index[lane] = wave->host_resident_slots[lane];
		wave->run_count = lanes;
		wave->sequence_row_begin = slot->run_begin;
		wave->sequence_row_indices = slot->run_row_indices;
		wave->run_state_index = slot->run_state_index;
		wave->host_sequence_row_begin = slot->host_run_begin;
		wave->host_sequence_row_indices = slot->host_run_row_indices;
		wave->host_run_state_index = slot->host_run_state_index;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextModuleCombineFusedBf16(
	void *combine_context,
	void *destination_device,
	const void *const *source_devices,
	uint32_t source_count,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchSumRanksF32((cudaStream_t)cuda_stream,
	    destination_device,source_devices,source_count,
	    (uint32_t)((uint64_t)active_sequence_count * hidden_dimension));
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,
	    "tp_all_reduce_fused"));
}

static SparkStatus SparkGlm5NextModuleCombineF32Seed(
	void *combine_context,
	void *destination_f32_device,
	const void *source_a_bf16_device,
	const void *source_b_bf16_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchSeedF32((cudaStream_t)cuda_stream,
	    (float *)destination_f32_device,source_a_bf16_device,
	    source_b_bf16_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,
	    "tp_all_reduce_f32_seed"));
}

static SparkStatus SparkGlm5NextModuleCombineF32Add(
	void *combine_context,
	void *destination_f32_device,
	const void *source_bf16_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchAddF32((cudaStream_t)cuda_stream,
	    (float *)destination_f32_device,source_bf16_device,
	    element_count);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,
	    "tp_all_reduce_f32_add"));
}

static SparkStatus SparkGlm5NextModuleRoundF32(
	void *combine_context,
	void *destination_bf16_device,
	const void *source_f32_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchRoundF32((cudaStream_t)cuda_stream,
	    destination_bf16_device,(const float *)source_f32_device,
	    element_count);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,
	    "tp_all_reduce_f32_round"));
}

static SparkStatus SparkGlm5NextRequestedMeshLane(uint32_t *lane)
{
	const char *text = getenv("SPARK_WEIGHTD_LANE");
	if ( text == 0 )
	{
		*lane = SPARK_WEIGHTD_LANE_NONE;
		return(SPARK_STATUS_OK);
	}
	if ( text[0] == '\0' )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_GLM5_NEXT_MODULE_TAG,
	    "SPARK_WEIGHTD_LANE",0u,SPARK_WEIGHTD_MESH_MAX_LANES - 1u,0u,lane));
}

#include "sparkpipe/family/module/spark_module_combine.h"

static SparkStatus SparkGlm5NextModuleInitializeTpCollective(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageNodeContext *context)
{
	SparkTpDeviceCollectiveConfig configuration,configuration_hc;
	uint32_t probe_connect_timeout_milli,probe_operation_timeout_milli;
	SparkStatus status;
	if ( state == 0 || context == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
		return(SPARK_STATUS_OK);
	if ( state->lane_client == 0 )
	{
		const char *socket = getenv("SPARK_WEIGHTD_SOCKET");
		uint32_t requested_lane;
		SparkWeightdMeshTopology topology;
		status = SparkTpDeviceCollectiveMeshTopology(state->tp_rank,state->tp_degree,&topology);
		if (status != SPARK_STATUS_OK) SPARK_RETURN(status);
		status = SparkGlm5NextRequestedMeshLane(&requested_lane);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( socket == 0 || socket[0] == '\0' )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		if ( SparkWeightdClientConnect(socket,&state->lane_client,0) != SPARK_STATUS_OK )
		{
			state->lane_client = 0;
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		status = SparkWeightdClientLaneAcquire(state->lane_client,requested_lane,&topology,&state->tp_lane,
			(uint64_t)context->tp_connect_timeout_milli * 1000000ull);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkWeightdClientClose(state->lane_client);
			state->lane_client = 0;
			fprintf(stderr,"GLM mesh lane acquire failed requested=%u capacity=%u status=%d\n",
			    requested_lane,SPARK_WEIGHTD_MESH_MAX_LANES,(int32_t)status);
			SPARK_RETURN(status);
		}
		fprintf(stderr,"GLM mesh lane mode=%s requested=%u resolved=%u capacity=%u rank=%u\n",
		    requested_lane == SPARK_WEIGHTD_LANE_NONE ? "automatic" : "explicit",
		    requested_lane,state->tp_lane,SPARK_WEIGHTD_MESH_MAX_LANES,state->tp_rank);
	}
	probe_connect_timeout_milli = context->tp_connect_timeout_milli;
	if ( SparkGlm5NextProbeEnabled() )
	{
		if ( probe_connect_timeout_milli >
			UINT32_MAX / SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		probe_connect_timeout_milli *= SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE;
	}
	probe_operation_timeout_milli = context->tp_operation_timeout_milli;
	if ( SparkGlm5NextProbeEnabled() )
	{
		if ( probe_operation_timeout_milli >
			UINT32_MAX / SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		probe_operation_timeout_milli *= SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE;
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = state->pipeline_slot_count * SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT;
	configuration.local_hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = state->execution_row_capacity;
	configuration.connect_timeout_milli = probe_connect_timeout_milli;
	configuration.operation_timeout_milli = probe_operation_timeout_milli;
	configuration.control_port_base = context->tp_collective_control_port_base +
		2u * state->tp_lane;
	configuration.collective_identifier = 2u * state->tp_lane;
	configuration.mesh_lane_client = state->lane_client;
	configuration.backend_module_path = context->tp_collective_backend_module_path;
	configuration.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.algorithm_mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL;
		configuration.direct_all_to_all_max_payload_bytes =
			SPARK_GLM5_NEXT_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES;
	}
	memset(&configuration_hc,0,sizeof(configuration_hc));
	configuration_hc.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration_hc.backend_kind = context->tp_collective_backend_kind;
	configuration_hc.tp_degree = state->tp_degree;
	configuration_hc.tp_rank = state->tp_rank;
	configuration_hc.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration_hc.credit_count = configuration.credit_count;
	configuration_hc.local_hidden_dimension =
		SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT;
	configuration_hc.max_active_sequence_count = configuration.max_active_sequence_count;
	configuration_hc.connect_timeout_milli = probe_connect_timeout_milli;
	configuration_hc.operation_timeout_milli = probe_operation_timeout_milli;
	configuration_hc.control_port_base = context->tp_collective_control_port_base +
		SPARK_GLM5_NEXT_TP_COLLECTIVE_HC_PORT_STRIDE + 2u * state->tp_lane;
	configuration_hc.collective_identifier = 2u * state->tp_lane + 1u;
	configuration_hc.mesh_lane_client = state->lane_client;
	configuration_hc.mesh_band_index = 1u;
	configuration_hc.backend_module_path = context->tp_collective_backend_module_path;
	configuration_hc.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration_hc);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memcpy(configuration_hc.session_ports,context->tp_collective_session_ports_hc,
		sizeof(configuration_hc.session_ports));
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.combine_bf16_function = SparkGlm5NextModuleCombineBf16;
		configuration.combine_fused_bf16_function = SparkGlm5NextModuleCombineFusedBf16;
		configuration.combine_f32_seed_function = SparkGlm5NextModuleCombineF32Seed;
		configuration.combine_f32_add_function = SparkGlm5NextModuleCombineF32Add;
		configuration.round_f32_function = SparkGlm5NextModuleRoundF32;
		configuration.combine_u64_max_function = SparkGlm5NextModuleCombineU64Max;
		configuration.combine_context = state;
		configuration_hc.combine_fused_bf16_function = SparkGlm5NextModuleCombineFusedBf16;
		configuration_hc.combine_f32_seed_function = SparkGlm5NextModuleCombineF32Seed;
		configuration_hc.combine_f32_add_function = SparkGlm5NextModuleCombineF32Add;
		configuration_hc.round_f32_function = SparkGlm5NextModuleRoundF32;
		configuration_hc.combine_bf16_function = SparkGlm5NextModuleCombineBf16;
		configuration_hc.combine_context = state;
	}
	if ( configuration.connect_timeout_milli == 0u || configuration.operation_timeout_milli == 0u || configuration.backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->tp_device_collective_initialized = 1u;
	if ( state->lazy_pack != 0 &&
	     state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
		status = SparkTpDeviceCollectivePrepareReceiveBf16(
		    &state->tp_device_collective,
		    (void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr,
		    0u,0u,0u,0u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkTpDeviceCollectiveCreate(&configuration_hc,&state->tp_device_collective_hc);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->tp_device_collective_hc_initialized = 1u;
	if ( state->lazy_pack != 0 &&
	     state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
		status = SparkTpDeviceCollectivePrepareReceiveBf16(
		    &state->tp_device_collective_hc,
		    (void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr,
		    0u,0u,0u,0u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextModuleReduceHiddenWide(SparkGlm5NextTpChain *chain,
	void *device_bf16,uint32_t hc_wide);

static SparkStatus SparkGlm5NextModuleReduceHidden(SparkGlm5NextTpChain *chain,void *device_bf16)
{
	return(SparkGlm5NextModuleReduceHiddenWide(chain,device_bf16,1u));
}

static SparkStatus SparkGlm5NextModuleReduceAttentionOut(SparkGlm5NextTpChain *chain,void *device_bf16)
{
	return(SparkGlm5NextModuleReduceHiddenWide(chain,device_bf16,0u));
}

static void SparkGlm5NextModuleTpCompletion(
	void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkGlm5NextTpChain *chain;
	chain = (SparkGlm5NextTpChain *)context;
	if ( chain == 0 || chain->active == 0u || completion == 0 )
		return;
	SparkStatus status = SparkGlm5NextWeightdHealth(chain->state);
	if ( status == SPARK_STATUS_OK )
		status = completion->status;
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextChainOrdinal(SparkGlm5NextTpChain *chain,uint32_t hc_wide,uint32_t operation,uint64_t *ordinal)
{
	SparkGlm5NextModuleState *state;
	state = chain->state;
	if ( state->tp_device_collective.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
	{
		*ordinal = atomic_fetch_add_explicit(hc_wide != 0u ? &state->nccl_next_ordinal_hc : &state->nccl_next_ordinal,1u,memory_order_relaxed);
		return(SPARK_STATUS_OK);
	}
	return(SparkTpChainOrdinal(chain->frame->request_id,state->pipeline_slot_count,SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS,operation,ordinal));
}

static SparkStatus SparkGlm5NextModuleReduceHiddenWide(SparkGlm5NextTpChain *chain,
	void *device_bf16,uint32_t hc_wide)
{
	uint32_t *op_index;
	SparkStatus ordinal_status;
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkTpDeviceCollective *collective;
	uint64_t ordinal;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	ordinal_status = SparkGlm5NextWeightdHealth(state);
	if ( ordinal_status != SPARK_STATUS_OK )
		SPARK_RETURN(ordinal_status);
	if ( hc_wide != 0u )
	{
		if ( state->tp_device_collective_hc_initialized == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		collective = &state->tp_device_collective_hc;
		op_index = &chain->tp_hc_op_index;
	}
	else
	{
		collective = &state->tp_device_collective;
		op_index = &chain->tp_op_index;
	}
	ordinal_status = SparkGlm5NextChainOrdinal(chain,hc_wide,*op_index,&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = device_bf16;
	submission.full_device = device_bf16;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkGlm5NextModuleTpCompletion;
	submission.completion_context = chain;
	{
		SparkStatus submit_status;
		*op_index += 1u;
		submit_status = SparkTpDeviceCollectiveEnqueue(collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
		if ( submit_status != SPARK_STATUS_OK )
		{
			*op_index -= 1u;
			fprintf(stderr,"G5N-DBG reduce submit -> %d (rows %u slot %u dev %p stream %p maxact %u)\n",
				(int)submit_status,(unsigned)chain->wave_rows,(unsigned)chain->slot_index,
				device_bf16,chain->slot->stream,
				(unsigned)state->tp_device_collective.max_active_sequence_count);
			SPARK_RETURN(submit_status);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextModuleGatherIndex(SparkGlm5NextTpChain *chain,uint32_t sequences,uint32_t chained)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	SparkStatus status;
	state = chain->state;
	if ( state->tp_collective_disabled != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( state->tp_device_collective_initialized == 0u || sequences == 0u || sequences > state->execution_row_capacity )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextChainOrdinal(chain,0u,chain->tp_op_index,&ordinal);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = sequences;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->index_local_scores_f32;
	submission.full_device = chain->slot->index_gathered_scores_f32;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = chained != 0u ? SparkGlm5NextModuleTpCompletion : 0;
	submission.completion_context = chained != 0u ? chain : 0;
	chain->tp_op_index += 1u;
	status = SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER);
	if ( status != SPARK_STATUS_OK )
		chain->tp_op_index -= 1u;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextModuleReduceHeadMax(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	SparkStatus status;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextChainOrdinal(chain,0u,chain->tp_op_index,&ordinal);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->head_maxloc_u64;
	submission.full_device = chain->slot->head_maxloc_u64;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkGlm5NextModuleTpCompletion;
	submission.completion_context = chain;
	chain->tp_op_index += 1u;
	status = SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
	if ( status != SPARK_STATUS_OK )
	{
		chain->tp_op_index -= 1u;
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextBuildMtpDraftWave(
	const SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	SparkGlm5NextCudaWave *wave)
{
	memset(wave,0,sizeof(*wave));
	wave->tp_degree = state->tp_degree;
	wave->tp_rank = state->tp_rank;
	wave->owns_final_head = state->owns_final_head;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->mtp_layer_weights = &state->mtp_layer;
	wave->mtp_eh_proj_bf16 = state->mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = state->mtp_enorm_bf16;
	wave->mtp_hnorm_bf16 = state->mtp_hnorm_bf16;
	wave->mtp_shared_norm_bf16 = state->mtp_shared_norm_bf16;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->slot = slot;
}

static SparkStatus SparkGlm5NextMtpDriveDraft(
	SparkGlm5NextModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	SparkGlm5NextExecutionSlot *slot,
	SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextAsyncCompletion *async;
	SparkGlm5NextCudaWave draft_wave;
	uint32_t lane,step;
	uint64_t position;
	if ( state->mtp_enabled == 0u ||
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ||
		batch->row_count != 1u || batch->active_sequence_count != 1u ||
		batch->token_ids == 0 || state->owns_embedding == 0u || state->owns_final_head == 0u ||
		batch->row_sampling[0].inverse_temperature != 0.0f )
		return(SPARK_STATUS_OK);
	lane = batch->row_resident_slots[0];
	position = batch->row_positions[0];
	if ( lane >= state->resident_sequence_capacity ||
		state->mtp_lane_armed[lane] == 0u ||
		position + SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u > state->max_sequence_positions )
		return(SPARK_STATUS_OK);
	async = &state->completions[chain->slot_index];
	SparkGlm5NextBuildMtpDraftWave(state,slot,&draft_wave);
	if ( SparkGlm5NextLaunchCudaMtpDraft(&draft_wave,0,
		state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
		batch->token_ids[0],async->mtp_draft_tokens) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( step = 1u; step <= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH; ++step )
	{
		slot->host_token_ids[step] = async->mtp_draft_tokens[step - 1u];
		slot->host_positions[step] = (uint32_t)(position + step);
		slot->host_resident_slots[step] = lane;
	}
	chain->wave_rows = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u;
	chain->spec_verify = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextMtpStashHidden(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextTpChain *chain)
{
	cudaError_t error;
	uint32_t row,lane;
	for ( row = 0u; row < chain->wave_rows; ++row )
	{
		lane = chain->slot->host_resident_slots[chain->first_row + row];
		if ( row + 1u < chain->wave_rows &&
			chain->slot->host_resident_slots[chain->first_row + row + 1u] == lane )
			continue;
		error = cudaMemcpyAsync(
			state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			chain->slot->hc_mean_bf16 + (uint64_t)row * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,(cudaStream_t)chain->slot->stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"mtp_stash"));
		state->mtp_lane_armed[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextLazyRelease(SparkGlm5NextTpChain *chain);
static void SparkGlm5NextFinishChain(SparkGlm5NextTpChain *chain)
{
	SparkStatus status = chain->expert_lease != 0u ? SparkGlm5NextLazyRelease(chain) : SPARK_STATUS_OK;
	chain->state->completions[chain->slot_index].launched_ns = SparkGlm5NextNowNs();
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextEnqueueAsyncCompletion(chain->state,chain->slot,chain->slot_index);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_FINISH;
	chain->active = 0u;
	free(chain);
}

static void SparkGlm5NextMtpResolveOnWorker(void *context)
{
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextModuleState *state;
	SparkGlm5NextExecutionSlot *slot;
	SparkGlm5NextAsyncCompletion *async;
	SparkSpeculationPolicyVerifyResult result;
	SparkStatus status;
	cudaError_t error;
	uint32_t lane;
	int32_t launch;
	chain = (SparkGlm5NextTpChain *)context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	pthread_mutex_lock(&state->completion_queue_lock);
	SparkGlm5NextDrainParkedCompletions(state);
	pthread_mutex_unlock(&state->completion_queue_lock);
	slot = chain->slot;
	async = &state->completions[chain->slot_index];
	lane = async->lane_indices[0];
	status = SparkSpeculationPolicyResolveVerifierTokens(
		async->mtp_draft_tokens,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,
		slot->host_output_token_ids,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u,
		SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,&result);
	if ( status == SPARK_STATUS_OK &&
		result.committed_token_count != result.accepted_draft_token_count + 1u )
		status = SPARK_STATUS_INTERNAL_ERROR;
	error = cudaSuccess;
	launch = 0;
	if ( status == SPARK_STATUS_OK )
	{
		async->completion.accepted_token_count = result.committed_token_count;
		async->completion.tokens_per_sequence = result.committed_token_count;
		async->burst_token_count = result.committed_token_count;
		async->lane_next_positions[0] += result.accepted_draft_token_count;
		async->mtp_cache_extra = result.accepted_draft_token_count;
		error = cudaMemcpyAsync(
			state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			slot->hc_mean_bf16 + (uint64_t)result.accepted_draft_token_count * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,(cudaStream_t)slot->stream);
		if ( error == cudaSuccess )
			launch = SparkGlm5NextLaunchCudaMtpCommit(&chain->wave,result.committed_token_count);
	}
	if ( status != SPARK_STATUS_OK || error != cudaSuccess || launch != 0 )
		SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
	else
		SparkGlm5NextFinishChain(chain);
}

static void CUDART_CB SparkGlm5NextMtpResolveHost(void *context)
{
	SparkGlm5NextTpChain *chain = context;
	if ( chain != 0 && chain->active != 0u )
		SparkGlm5NextScheduleCompletionWork(chain->state,SparkGlm5NextMtpResolveOnWorker,chain);
}

static void SparkGlm5NextLazyRetryRetained(void *context);

static void SparkGlm5NextScheduleRetainedRetry(SparkGlm5NextModuleState *state)
{
	if ( state->lazy_pack == 0 || state->lazy_pack->worker == 0 )
		return;
	(void)SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyRetryRetained,state);
}

static void SparkGlm5NextTpChainFail(SparkGlm5NextTpChain *chain,SparkStatus status)
{
	SparkGlm5NextModuleState *state;
	SparkGlm5NextAsyncCompletion *async;
	if ( chain->active == 0u )
		return;
	chain->active = 0u;
	state = chain->state;
	chain->slot->route_recorded = 0u;
	fprintf(stderr,"G5N-DBG chainfail: stage %u next_layer %u rows %u status %d cuda=%s\n",
		(unsigned)chain->stage,(unsigned)chain->next_layer,(unsigned)chain->wave_rows,(int)status,
		cudaGetErrorString(cudaGetLastError()));
	if ( state->tp_device_collective_initialized != 0u )
	{
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective);
		(void)SparkTpDeviceCollectiveChainRetire(&state->tp_device_collective);
	}
	if ( state->tp_device_collective_hc_initialized != 0u )
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective_hc);
	if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
	{
		chain->retained_status = status;
		fprintf(stderr,"GLM chain drain failed; retaining slot %u and CUDA resources for teardown retry\n",chain->slot_index);
		atomic_store_explicit(&state->lazy_retained[chain->slot_index],chain,memory_order_release);
		SparkGlm5NextScheduleRetainedRetry(state);
		return;
	}
	if ( chain->expert_lease != 0u && SparkGlm5NextLazyRelease(chain) != SPARK_STATUS_OK )
	{
		chain->retained_status = status;
		atomic_store_explicit(&state->lazy_retained[chain->slot_index],chain,memory_order_release);
		SparkGlm5NextScheduleRetainedRetry(state);
		return;
	}
	async = &state->completions[chain->slot_index];
	async->completion.status = status;
	SparkGlm5NextCompleteAsync(async);
	free(chain);
}

static SparkStatus SparkGlm5NextLazyRelease(SparkGlm5NextTpChain *chain)
{
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	if ( chain->expert_lease == 0u )
		return(SPARK_STATUS_OK);
	if ( chain->expert_lease_begun != 0u )
	{
		if ( chain->expert_lease_recorded == 0u )
		{
			status = SparkWeightdMapRecordCompletion(map,chain->expert_lease,(cudaStream_t)chain->slot->stream);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			chain->expert_lease_recorded = 1u;
			chain->wave.expert_lease_base = 0;
		}
		if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = SparkWeightdMapRelease(map,chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
	{
		chain->expert_lease = 0u;
		chain->expert_lease_begun = 0u;
		chain->expert_lease_recorded = 0u;
		chain->wave.expert_lease_base = 0;
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextLazyRecoverLease(SparkGlm5NextModuleState *state,uint32_t slot,SparkGlm5NextTpChain **out)
{
	SparkGlm5NextTpChain *chain;
	SparkStatus status = SPARK_STATUS_OK;
	*out = 0;
	chain = atomic_exchange_explicit(&state->lazy_retained[slot],0,memory_order_acq_rel);
	if ( chain == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( chain->expert_lease != 0u )
		status = SparkGlm5NextLazyRelease(chain);
	if ( status != SPARK_STATUS_OK )
		atomic_store_explicit(&state->lazy_retained[slot],chain,memory_order_release);
	else
		*out = chain;
	SPARK_RETURN(status);
}

static void SparkGlm5NextLazyRetryRetained(void *context)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkGlm5NextTpChain *chain;
	uint32_t slot;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( SparkGlm5NextLazyRecoverLease(state,slot,&chain) == SPARK_STATUS_OK )
		{
			chain->active = 1u;
			SparkGlm5NextTpChainFail(chain,chain->retained_status);
		}
}

static void SparkGlm5NextTpChainReduceMlp(SparkGlm5NextTpChain *chain)
{
	SparkStatus status;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP;
	status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
}

static uint32_t SparkGlm5NextFirstRoutedLayer(const SparkGlm5NextModuleState *state)
{
	return(state->first_layer_index > SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER ? state->first_layer_index : SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER);
}

static SparkStatus SparkGlm5NextPinAllExperts(SparkGlm5NextModuleState *state)
{
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t layer,expert,count = 0u;
	if ( state->expert_pin_lease_count != 0u )
		return(SPARK_STATUS_BUSY);
	for (layer=SparkGlm5NextFirstRoutedLayer(state); layer<state->first_layer_index + state->layer_count && status==SPARK_STATUS_OK; layer++)
		for (expert=0u; expert<SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT && status==SPARK_STATUS_OK; expert++)
		{
			keys[count++] = (SparkWeightdExpertKey){.layer=layer,.expert=expert};
			if ( count == SPARK_WEIGHTD_LEASE_GROUPS_MAX || (layer + 1u == state->first_layer_index + state->layer_count && expert + 1u == SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT) )
			{
				uint64_t lease = 0u;
				void *base = 0;
				uint32_t index = state->expert_pin_lease_count;
				if ( index >= sizeof(state->expert_pin_leases) / sizeof(state->expert_pin_leases[0]) )
					return(SPARK_STATUS_CAPACITY_EXCEEDED);
				status = SparkWeightdMapAcquire(state->lazy_pack->map,keys,count,&lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
				if ( lease != 0u )
					state->expert_pin_leases[state->expert_pin_lease_count++] = lease;
				if ( status == SPARK_STATUS_OK )
					status = lease != 0u ? SparkWeightdMapBeginUse(state->lazy_pack->map,lease,&base) : SPARK_STATUS_VALIDATION_FAILED;
				if ( status == SPARK_STATUS_OK )
				{
					state->expert_pin_phases[index] = 1u;
					if ( base == 0 || (state->decode_lease_base_saved != 0 && state->decode_lease_base_saved != base) )
						status = SPARK_STATUS_VALIDATION_FAILED;
					else
					{
						state->expert_pin_key_count += count;
						state->decode_lease_base_saved = base;
					}
				}
				count = 0u;
			}
		}
	return(status);
}

static SparkStatus SparkGlm5NextReleasePinnedExperts(SparkGlm5NextModuleState *state)
{
	while ( state->expert_pin_lease_count != 0u )
	{
		uint32_t index = state->expert_pin_lease_count - 1u;
		SparkStatus status;
		if ( state->expert_pin_phases[index] == 1u )
		{
			status = SparkWeightdMapRecordCompletion(state->lazy_pack->map,state->expert_pin_leases[index],(cudaStream_t)state->execution_stream);
			if ( status != SPARK_STATUS_OK ) return(status);
			state->expert_pin_phases[index] = 2u;
		}
		status = SparkWeightdMapRelease(state->lazy_pack->map,state->expert_pin_leases[index],SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
		if ( status != SPARK_STATUS_OK ) return(status);
		state->expert_pin_phases[index] = 0u;
		state->expert_pin_leases[index] = 0u;
		state->expert_pin_lease_count--;
	}
	state->expert_pin_key_count = 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextGraphClaimExperts(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	uint32_t index,first = SparkGlm5NextFirstRoutedLayer(state),end = state->first_layer_index + state->layer_count;
	uint32_t expected = end > first ? (end - first) * SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT : 0u;
	if ( state->expert_pin_key_count != expected || state->expert_pin_lease_count != SparkCeilDivU32(expected,SPARK_WEIGHTD_LEASE_GROUPS_MAX) )
	{
		fprintf(stderr,"GLM whole-chain graph requires %u leased experts; held %u\n",expected,state->expert_pin_key_count);
		return(SPARK_STATUS_UNSUPPORTED);
	}
	for (index=0u; index<state->expert_pin_lease_count; index++)
		if ( state->expert_pin_leases[index] == 0u || state->expert_pin_phases[index] != 1u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	if ( expected != 0u && state->decode_lease_base_saved == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	chain->wave.expert_lease_base = state->decode_lease_base_saved;
	chain->wave.expert_lease_local_layer = 0u;
	chain->wave.expert_lease_all = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextLazyExperts(SparkGlm5NextTpChain *chain)
{
	SparkWeightdExpertKey keys[SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT];
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	void *address = 0;
	uint32_t count = 0u;
	if ( chain->slot->route_recorded == 0u || cudaEventSynchronize((cudaEvent_t)chain->slot->route_ready_event) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdRouteKeys(chain->wave.first_layer_index + chain->next_layer,chain->slot->host_group_row_offset +
		    (chain->wave.first_layer_index + chain->next_layer) *
		        (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u),
		SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,chain->wave.row_count * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,keys,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,&count);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapAcquire(map,keys,count,&chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status != SPARK_STATUS_OK )
	{
		uint32_t diag_index;
		fprintf(stderr,"LAZYWORK-KEYS slot=%u layer=%u count=%u status=%d",
			(unsigned)chain->slot_index,(unsigned)chain->next_layer,
			(unsigned)count,(int)status);
		for (diag_index=0u; diag_index<count && diag_index<8u; diag_index++)
			fprintf(stderr," %u:%u",(unsigned)keys[diag_index].layer,(unsigned)keys[diag_index].expert);
		fprintf(stderr,"\n");
		SPARK_RETURN(status);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapBeginUse(map,chain->expert_lease,&address);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	chain->expert_lease_begun = 1u;
	chain->wave.expert_lease_base = (const uint8_t *)address;
	chain->wave.expert_lease_local_layer = chain->next_layer;
	if ( chain->state->decode_lease_base_saved == 0 )
		chain->state->decode_lease_base_saved = (const uint8_t *)address;
	{
		const uint32_t *cover_saved = chain->wave.expert_cover;
		void *miss_saved = chain->wave.expert_miss;
		chain->wave.expert_cover = 0;
		chain->wave.expert_miss = 0;
		if ( SparkGlm5NextLaunchCudaLayerMlpExperts(&chain->wave,chain->next_layer) != 0 )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		chain->wave.expert_cover = cover_saved;
		chain->wave.expert_miss = miss_saved;
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextLazyWork(void *context)
{
	SparkGlm5NextTpChain *chain = (SparkGlm5NextTpChain *)context;
	SparkStatus status,cleanup;
	{
		static uint32_t lazy_trace_count;
		if ( lazy_trace_count < 600u )
		{
			lazy_trace_count++;
			fprintf(stderr,"LAZYWORK slot=%u layer=%u\n",(unsigned)chain->slot_index,(unsigned)chain->next_layer);
		}
	}
	status = SparkGlm5NextLazyExperts(chain);
	if ( status == SPARK_STATUS_OK && chain->state->decode_cover_host != 0 )
	{
		SparkGlm5NextModuleState *st = chain->state;
		SparkWeightdExpertKey ukeys[SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT];
		uint32_t ucount = 0u,index,bit;
		SparkWeightdRouteKeys(chain->wave.first_layer_index + chain->next_layer,
			chain->slot->host_group_row_offset +
			    (chain->wave.first_layer_index + chain->next_layer) *
			        (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u),
			SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,
			chain->wave.row_count * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,
			ukeys,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,&ucount);
		for (index=0u; index<ucount; index++)
		{
			bit = ukeys[index].layer *
			    SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT +
			    ukeys[index].expert;
			st->decode_cover_host[bit / 32u] |=
				UINT32_C(1) << (bit % 32u);
		}
	}
	cleanup = SparkGlm5NextLazyRelease(chain);
	if ( cleanup == SPARK_STATUS_IO_ERROR || cleanup == SPARK_STATUS_BUSY )
		cleanup = SparkGlm5NextLazyRelease(chain);
	if ( cleanup != SPARK_STATUS_OK )
	{
		chain->retained_status = status != SPARK_STATUS_OK ? status : cleanup;
		fprintf(stderr,"GLM expert cleanup failed: slot=%u lease=%llu status=%d; retaining slot and lease for teardown retry\n",chain->slot_index,(unsigned long long)chain->expert_lease,(int32_t)cleanup);
		atomic_store_explicit(&chain->state->lazy_retained[chain->slot_index],chain,memory_order_release);
		SparkGlm5NextScheduleRetainedRetry(chain->state);
		return;
	}
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainReduceMlp(chain);
}

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status);

static SparkStatus SparkGlm5NextGraphReduce(SparkGlm5NextTpChain *chain,
    void *device,uint32_t hc_wide)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus ordinal_status;
	uint64_t ordinal;
	state = chain->state;
	ordinal_status = SparkGlm5NextChainOrdinal(chain,hc_wide,
		hc_wide != 0u ? chain->tp_hc_op_index : chain->tp_op_index,
		&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = device;
	submission.full_device = device;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = 0;
	submission.completion_context = 0;
	if ( hc_wide != 0u )
		chain->tp_hc_op_index += 1u;
	else
		chain->tp_op_index += 1u;
	if ( state->graph_record_limit != 0u &&
	     ++state->graph_record_ops >= state->graph_record_limit )
		state->graph_record_stop = 1u;
	return(SparkTpDeviceCollectiveEnqueue(
		hc_wide != 0u ? &state->tp_device_collective_hc :
			&state->tp_device_collective,
		&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16));
}

static SparkStatus SparkGlm5NextGraphReduceHead(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus ordinal_status;
	uint64_t ordinal;
	state = chain->state;
	ordinal_status = SparkGlm5NextChainOrdinal(chain,0u,chain->tp_op_index,
		&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->head_maxloc_u64;
	submission.full_device = chain->slot->head_maxloc_u64;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = 0;
	submission.completion_context = 0;
	chain->tp_op_index += 1u;
	return(SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,
		&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64));
}

static void SparkGlm5NextGraphRecord(SparkGlm5NextTpChain *chain,
    void **exec_out)
{
	SparkGlm5NextCudaWave *wave;
	SparkGlm5NextModuleState *state;
	cudaStream_t stream;
	cudaGraph_t graph;
	cudaGraphExec_t exec;
	uint32_t failed = 0u;
	uint32_t failed_site = 0u;
	uint32_t layer,gather_sequences;
	state = chain->state;
	wave = &chain->wave;
	stream = (cudaStream_t)wave->slot->stream;
	*exec_out = 0;
	state->graph_record_ops = 0u;
	state->graph_record_stop = 0u;
	if ( cudaStreamBeginCapture(stream,0u) != cudaSuccess )
		return;
	if ( SparkGlm5NextLaunchCudaWaveBegin(wave) != 0 )
		{ failed = 1u; failed_site = 2u; }
	if ( failed == 0u &&
	     SparkGlm5NextModuleReduceHidden(chain,
	         chain->slot->hidden_bf16) != SPARK_STATUS_OK )
		{ failed = 1u; failed_site = 3u; }
	for ( layer = 0u;
	      layer < wave->layer_count && failed == 0u &&
	          state->graph_record_stop == 0u;
	      layer++ )
	{
		gather_sequences = SparkGlm5NextLayerIndexGatherSequences(wave,layer);
		if ( gather_sequences != 0u &&
		     (SparkGlm5NextLaunchCudaLayerAttentionScore(wave,layer) != 0 ||
		      SparkGlm5NextModuleGatherIndex(chain,gather_sequences,0u) != SPARK_STATUS_OK ||
		      SparkGlm5NextLaunchCudaLayerAttentionSelect(wave,layer) != 0) )
			{ failed = 1u; failed_site = 15u; break; }
		if ( gather_sequences == 0u && SparkGlm5NextLaunchCudaLayerAttention(wave,layer) != 0 )
			{ failed = 1u; failed_site = 4u; break; }
		if ( SparkGlm5NextGraphReduce(chain,wave->slot->attention_out_bf16,
				0u) != SPARK_STATUS_OK )
			{ failed = 1u; failed_site = 5u; break; }
		if ( SparkGlm5NextLaunchCudaLayerAttentionPost(wave,layer) != 0 )
			{ failed = 1u; failed_site = 6u; break; }
		if ( (wave->first_layer_index + layer) >=
		        SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
		{
			if ( SparkGlm5NextLaunchCudaLayerMlpRoute(wave,layer) != 0 )
				{ failed = 1u; failed_site = 7u; break; }
			if ( SparkGlm5NextLaunchCudaLayerMlpExperts(wave,layer) != 0 )
				{ failed = 1u; failed_site = 8u; break; }
		}
		else if ( SparkGlm5NextLaunchCudaLayerMlp(wave,layer) != 0 )
			{ failed = 1u; failed_site = 9u; break; }
		if ( SparkGlm5NextGraphReduce(chain,wave->slot->attention_out_bf16,
				0u) != SPARK_STATUS_OK )
			{ failed = 1u; failed_site = 10u; break; }
		if ( SparkGlm5NextLaunchCudaLayerMlpPost(wave,layer) != 0 )
			{ failed = 1u; failed_site = 11u; break; }
	}
	if ( failed == 0u && SparkGlm5NextLaunchCudaWaveHead(wave) != 0 )
		{ failed = 1u; failed_site = 12u; }
	if ( failed == 0u &&
	     SparkGlm5NextGraphReduceHead(chain) != SPARK_STATUS_OK )
		{ failed = 1u; failed_site = 13u; }
	if ( failed == 0u &&
	     SparkGlm5NextLaunchHeadMaxlocUnpack(stream,wave->slot->head_maxloc_u64,
			wave->slot->output_token,wave->row_count) != cudaSuccess )
		{ failed = 1u; failed_site = 14u; }
	if ( failed == 0u && state->owns_final_head != 0u )
		failed = cudaMemcpyAsync(
			wave->slot->host_output_token_ids + chain->first_row,
			wave->slot->output_token,
			(uint64_t)wave->row_count * sizeof(uint32_t),
			cudaMemcpyDeviceToHost,stream) != cudaSuccess ? 1u : 0u;
	if ( cudaStreamEndCapture(stream,&graph) != cudaSuccess ||
	     graph == 0 || failed != 0u )
	{
		fprintf(stderr,
		    "GRAPH-RECORD-FAIL failed=%u site=%u layer=%u end_graph=%p pending=%s\n",
		    (unsigned)failed,(unsigned)failed_site,(unsigned)layer,graph,
		    cudaGetErrorString(cudaGetLastError()));
		if ( graph != 0 )
			(void)cudaGraphDestroy(graph);
		return;
	}
	if ( cudaGraphInstantiate(&exec,graph,0) != cudaSuccess ||
	     cudaGraphUpload(exec,stream) != cudaSuccess )
	{
		fprintf(stderr,
		    "GRAPH-INSTANTIATE-FAIL inst=%s upload=%s\n",
		    cudaGetErrorString(cudaGetLastError()),
		    cudaGetErrorString(cudaGetLastError()));
		(void)cudaGraphDestroy(graph);
		return;
	}
	(void)cudaGraphDestroy(graph);
	*exec_out = exec;
}

static SparkStatus SparkGlm5NextGraphCoverEnsure(
	SparkGlm5NextModuleState *state)
{
	uint32_t words;
	if ( state->decode_cover_device != 0 )
		return(SPARK_STATUS_OK);
	words = (SPARK_GLM5_NEXT_MODEL_LAYER_COUNT *
	    SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 31u) / 32u;
	if ( cudaHostAlloc((void **)&state->decode_cover_host,
	         (size_t)words * sizeof(uint32_t),
	         cudaHostAllocMapped) != cudaSuccess ||
	     cudaHostAlloc((void **)&state->decode_miss_host,
	         SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES,
	         cudaHostAllocMapped) != cudaSuccess ||
	     cudaMalloc((void **)&state->decode_cover_device,
	         (size_t)words * sizeof(uint32_t)) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(state->decode_cover_host,0,(size_t)words * sizeof(uint32_t));
	memset(state->decode_miss_host,0,
	    SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES);
	state->decode_cover_words = words;
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextGraphStep(SparkGlm5NextTpChain *chain,
    SparkStatus *status_out)
{
	SparkGlm5NextModuleState *state;
	SparkStatus status;
	uint32_t row;
	void *exec;
	state = chain->state;
	status = SparkGlm5NextWeightdHealth(state);
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphPreLaunch(
	         &state->tp_device_collective,chain->slot->stream) !=
	             SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_hc_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphPreLaunch(
	         &state->tp_device_collective_hc,chain->slot->stream) !=
	             SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphCancelSeed(
	         &state->tp_device_collective,chain->slot->stream) !=
	             SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_hc_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphCancelSeed(
	         &state->tp_device_collective_hc,chain->slot->stream) !=
	         SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		exec = chain->slot->graph_exec_a;
		{
			cudaError_t pre_err = cudaGetLastError();
			cudaError_t launch_rc = cudaGraphLaunch(exec,chain->slot->stream);
			if ( launch_rc != cudaSuccess )
			{
				fprintf(stderr,"GRAPH-LAUNCH-ERR slot=%u rc=%s pre=%s\n",
					chain->slot_index,
					cudaGetErrorString(launch_rc),
					cudaGetErrorString(pre_err));
				status = SPARK_STATUS_IO_ERROR;
			}
		else
		{
			struct timespec replay_t0,replay_t1;
			clock_gettime(CLOCK_MONOTONIC,&replay_t0);
			cudaError_t poll;
			SparkStatus wait_status = SparkStageModuleCudaWaitFor(&state->stream_wait,
				(uint64_t)state->tp_device_collective.operation_timeout_milli * UINT64_C(1000000));
			poll = wait_status == SPARK_STATUS_OK ? cudaSuccess : wait_status == SPARK_STATUS_BUSY ? cudaErrorNotReady : cudaErrorUnknown;
			clock_gettime(CLOCK_MONOTONIC,&replay_t1);
			{
				uint64_t replay_ns = (uint64_t)(replay_t1.tv_sec - replay_t0.tv_sec) *
				    UINT64_C(1000000000) +
				    (uint64_t)(replay_t1.tv_nsec - replay_t0.tv_nsec);
				fprintf(stderr,
				    "GRAPH-REPLAY-TIME slot=%u wall_ns=%llu stream_status=%d\n",
				    chain->slot_index,
				    (unsigned long long)replay_ns,(int)poll);
			}
			if ( poll == cudaErrorNotReady )
			{
				uint64_t stuck_error;
				uint64_t stuck_diag;
					uint64_t stuck_cell = 0ull;
					uint64_t stuck_progress = SparkTpDeviceCollectiveGraphProgress(
						&state->tp_device_collective,&stuck_cell);
					(void)SparkTpDeviceCollectiveGraphStuckDump(
						&state->tp_device_collective);
				stuck_error = SparkTpDeviceCollectiveGraphError(
					&state->tp_device_collective);
				stuck_diag = SparkTpDeviceCollectiveGraphDiag(
					&state->tp_device_collective);
				state->graph_path_enabled = 0u;
				state->degrade_graph_stuck++;
				SparkTpDeviceCollectiveBroadcastCancel(
					&state->tp_device_collective);
				if ( state->tp_device_collective_hc_initialized != 0u )
					SparkTpDeviceCollectiveBroadcastCancel(
						&state->tp_device_collective_hc);
				fprintf(stderr,
					"GRAPH-CANCEL-BROADCAST slot=%u\n",
					chain->slot_index);
				fprintf(stderr,
					"GRAPH-FAILED graph-stuck slot=%u progress=%llu cell=%llu err=%llu diag_peer=%llu ring=%llu slotidx=%llu want=%llu got=%llu\n",
					chain->slot_index,
					(unsigned long long)stuck_progress,
						(unsigned long long)stuck_cell,
						(unsigned long long)stuck_error,
					(unsigned long long)(stuck_diag >> 56),
					(unsigned long long)((stuck_diag >> 48) & 0xff),
					(unsigned long long)((stuck_diag >> 32) & 0xffff),
					(unsigned long long)((stuck_diag >> 16) & 0xffff),
					(unsigned long long)(stuck_diag & 0xffff));
				status = SPARK_STATUS_INTERNAL_ERROR;
			}
			else if ( poll != cudaSuccess )
			{
				fprintf(stderr,"GRAPH-STREAM-ERR slot=%u cuda=%s\n",
					chain->slot_index,
					cudaGetErrorString(poll));
				(void)cudaGetLastError();
				status = SPARK_STATUS_IO_ERROR;
			}
			else if ( state->tp_device_collective_initialized != 0u )
			{
				(void)SparkTpDeviceCollectiveDisarmCapture(
				    &state->tp_device_collective);
				if ( state->tp_device_collective_hc_initialized != 0u )
					(void)SparkTpDeviceCollectiveDisarmCapture(
					    &state->tp_device_collective_hc);
			}
		}
	}
	}
	if ( status == SPARK_STATUS_OK )
	{
		uint64_t graph_error;
		graph_error = SparkTpDeviceCollectiveGraphError(
			&state->tp_device_collective);
		if ( graph_error == 0ull &&
		     state->tp_device_collective_hc_initialized != 0u )
		{
			graph_error = SparkTpDeviceCollectiveGraphError(
				&state->tp_device_collective_hc);
		}
		if ( graph_error != 0ull )
		{
			state->graph_path_enabled = 0u;
			state->degrade_graph_stuck++;
			SparkTpDeviceCollectiveBroadcastCancel(
				&state->tp_device_collective);
			if ( state->tp_device_collective_hc_initialized != 0u )
				SparkTpDeviceCollectiveBroadcastCancel(
					&state->tp_device_collective_hc);
			fprintf(stderr,
				"GRAPH-CANCEL-BROADCAST slot=%u\n",
				chain->slot_index);
			fprintf(stderr,
				"GRAPH-FAILED graph-wait-timeout slot=%u seq=%llu\n",
				chain->slot_index,
				(unsigned long long)graph_error);
			status = SPARK_STATUS_INTERNAL_ERROR;
		}
		else
		{
			if ( state->graph_arrival_dumped == 0u )
			{
				state->graph_arrival_dumped = 1u;
				(void)SparkTpDeviceCollectiveGraphArrivalDump(
					&state->tp_device_collective,
					state->tp_rank);
			}
		}
	}
	if ( status == SPARK_STATUS_OK && state->decode_miss_host != 0 &&
	     state->decode_miss_host[0] != 0u )
	{
		fprintf(stderr,"GRAPH-EXPERT-MISS slot=%u\n",chain->slot_index);
		status = SPARK_STATUS_BUSY;
	}
	if ( status == SPARK_STATUS_OK )
		for ( row = 0u; row < chain->wave_rows; row++ )
		{
			((uint32_t *)chain->wave.host_positions)[row] += 1u;
			((uint32_t *)chain->wave.host_token_ids)[row] =
				chain->slot->host_output_token_ids[chain->first_row + row];
		}
	*status_out = status;
}

#define SPARK_GLM5_NEXT_GRAPH_CONTEXT_MARGIN 256u

static SparkStatus SparkGlm5NextGraphArm(
    SparkGlm5NextModuleState *state)
{
	if ( SparkTpDeviceCollectiveArmCapture(
	         &state->tp_device_collective) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->tp_device_collective_hc_initialized != 0u &&
	     SparkTpDeviceCollectiveArmCapture(
	         &state->tp_device_collective_hc) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextGraphDisarm(
    SparkGlm5NextModuleState *state)
{
	(void)SparkTpDeviceCollectiveDisarmCapture(
		&state->tp_device_collective);
	if ( state->tp_device_collective_hc_initialized != 0u )
		(void)SparkTpDeviceCollectiveDisarmCapture(
			&state->tp_device_collective_hc);
}

static SparkStatus SparkGlm5NextGraphCapture(SparkGlm5NextTpChain *chain,uint32_t index,uint32_t bound)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	void *exec = 0;
	chain->wave.maximum_context = bound;
	if ( SparkGlm5NextGraphArm(state) != SPARK_STATUS_OK )
	{
		SparkGlm5NextGraphDisarm(state);
		slot->graph_disabled = 1u;
		return(SPARK_STATUS_BUSY);
	}
	SparkGlm5NextGraphRecord(chain,&exec);
	SparkGlm5NextGraphDisarm(state);
	if ( exec == 0 )
	{
		fprintf(stderr,"GRAPH-CAPTURE-FAILED rows=%u; this row count runs eager from now on\n",index + 1u);
		slot->graph_failed_rows |= UINT64_C(1) << index;
		return(SPARK_STATUS_UNSUPPORTED);
	}
	slot->graph_exec_rows[index] = exec;
	slot->graph_bound_rows[index] = bound;
	fprintf(stderr,"GRAPH-CAPTURE-OK rows=%u bound=%u\n",index + 1u,bound);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextGraphEnsure(SparkGlm5NextTpChain *chain,
    SparkStatus *status_out)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	SparkStatus status;
	uint32_t bound, index = chain->wave_rows - 1u;
	if ( slot->graph_disabled != 0u )
	{
		*status_out = SPARK_STATUS_BUSY;
		return;
	}
	status = SparkGlm5NextGraphClaimExperts(chain);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextGraphCoverEnsure(state);
	if ( status != SPARK_STATUS_OK )
	{
		*status_out = status;
		return;
	}
	bound = chain->wave.maximum_context + SPARK_GLM5_NEXT_GRAPH_CONTEXT_MARGIN;
	if ( bound > chain->wave.max_sequence_positions )
		bound = chain->wave.max_sequence_positions;
	if ( slot->graph_exec_rows[index] != 0 && chain->wave.maximum_context > slot->graph_bound_rows[index] )
	{
		(void)cudaGraphExecDestroy((cudaGraphExec_t)slot->graph_exec_rows[index]);
		slot->graph_exec_rows[index] = 0;
	}
	if ( slot->graph_exec_rows[index] == 0 )
		status = SparkGlm5NextGraphCapture(chain,index,bound);
	if ( slot->graph_exec_rows[index] == 0 )
	{
		*status_out = status == SPARK_STATUS_UNSUPPORTED ? status : SPARK_STATUS_BUSY;
		return;
	}
	slot->graph_exec_a = slot->graph_exec_rows[index];
	SparkGlm5NextGraphStep(chain,&status);
	if ( status == SPARK_STATUS_OK )
		SparkGlm5NextGraphDisarm(state);
	*status_out = status;
}

#include "sparkpipe/family/module/spark_module_t1_enabled.h"

static void SparkGlm5NextT1Streams(SparkGlm5NextTpChain *chain,uint32_t layer)
{
	static uint16_t *rows_host = 0;
	static uint32_t rows_host_capacity = 0;
	uint32_t row;
	uint32_t i;
	uint32_t flat;
	uint64_t bytes;
	if ( SparkGlm5NextT1Enabled() == 0 || chain->wave.tp_rank != 0u )
		return;
	if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
		return;
	flat = SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	bytes = (uint64_t)chain->wave_rows * flat * sizeof(uint16_t);
	if ( rows_host_capacity < chain->wave_rows )
	{
		free(rows_host);
		rows_host = (uint16_t *)malloc(bytes);
		rows_host_capacity = rows_host != 0 ? chain->wave_rows : 0u;
	}
	if ( rows_host == 0 ||
	    cudaMemcpy(rows_host,chain->slot->hidden_bf16,bytes,cudaMemcpyDeviceToHost) != cudaSuccess )
		return;
	for ( row = 0u; row < chain->wave_rows; row++ )
	{
		uint16_t *values = rows_host + (uint64_t)row * flat;
		fprintf(stderr,"G5N-T1 stream L%u pos%u",layer,chain->wave.host_positions[row]);
		for ( i = 0u; i < flat; i++ )
			fprintf(stderr," %04x",values[i]);
		fputc('\n',stderr);
	}
}
static void SparkGlm5NextT1Route(SparkGlm5NextTpChain *chain,uint32_t layer)
{
	static uint32_t *ids_host = 0;
	static float *weights_host = 0;
	static uint32_t route_capacity = 0;
	uint32_t row;
	uint32_t k;
	uint64_t bytes;
	if ( SparkGlm5NextT1Enabled() == 0 || chain->wave.tp_rank != 0u ||
	    layer < SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
		return;
	bytes = (uint64_t)chain->wave_rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K * sizeof(uint32_t);
	if ( route_capacity < chain->wave_rows )
	{
		free(ids_host);
		free(weights_host);
		ids_host = (uint32_t *)malloc(bytes);
		weights_host = (float *)malloc(bytes);
		route_capacity = ids_host != 0 && weights_host != 0 ? chain->wave_rows : 0u;
	}
	if ( ids_host == 0 || weights_host == 0 ||
	    cudaMemcpy(ids_host,chain->slot->route_expert,bytes,cudaMemcpyDeviceToHost) != cudaSuccess ||
	    cudaMemcpy(weights_host,chain->slot->route_weight,bytes,cudaMemcpyDeviceToHost) != cudaSuccess )
		return;
	for ( row = 0u; row < chain->wave_rows; row++ )
	{
		fprintf(stderr,"G5N-T1 route L%u pos%u ids",layer,chain->wave.host_positions[row]);
		for ( k = 0u; k < SPARK_GLM5_NEXT_MODEL_MOE_TOP_K; k++ )
			fprintf(stderr," %u",ids_host[row * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K + k]);
		fprintf(stderr," weights");
		for ( k = 0u; k < SPARK_GLM5_NEXT_MODEL_MOE_TOP_K; k++ )
			fprintf(stderr," %08x",
			    ((const uint32_t *)weights_host)[row * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K + k]);
		fputc('\n',stderr);
	}
}
static void SparkGlm5NextT1Head(SparkGlm5NextTpChain *chain)
{
	static uint32_t *tokens_host = 0;
	static float *scores_host = 0;
	static uint32_t head_capacity = 0;
	uint32_t i;
	if ( SparkGlm5NextT1Enabled() == 0 || chain->wave.tp_rank != 0u ||
	    chain->state->owns_final_head == 0u )
		return;
	if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
		return;
	if ( head_capacity < chain->wave_rows )
	{
		free(tokens_host);
		free(scores_host);
		tokens_host = (uint32_t *)malloc((uint64_t)chain->wave_rows * sizeof(uint32_t));
		scores_host = (float *)malloc((uint64_t)chain->wave_rows * sizeof(float));
		head_capacity = tokens_host != 0 && scores_host != 0 ? chain->wave_rows : 0u;
	}
	if ( tokens_host == 0 || scores_host == 0 ||
	    cudaMemcpy(tokens_host,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
	    cudaMemcpy(scores_host,chain->slot->output_score,(uint64_t)chain->wave_rows * sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess )
		return;
	for ( i = 0u; i < chain->wave_rows; i++ )
		fprintf(stderr,"G5N-T1 head pos%u token %u score_bits %08x\n",
		    chain->wave.host_positions[i],tokens_host[i],
		    ((const uint32_t *)scores_host)[i]);
}

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status)
{
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextModuleState *state;
	SparkStatus launch_status;
	cudaError_t error;
	uint32_t gather_sequences;
	chain = (SparkGlm5NextTpChain *)chain_context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	{
		uint64_t now_ns = SparkGlm5NextNowNs();
		chain->last_advance_ns = now_ns;
		state->slot_alive_ns[chain->slot_index] = now_ns;
		if ( chain->created_ns == 0ull )
			chain->created_ns = now_ns;
		else if ( now_ns - chain->created_ns >= UINT64_C(30000000000) &&
			chain->last_heartbeat_stage != (uint64_t)chain->stage + 1ull )
		{
			chain->last_heartbeat_stage = (uint64_t)chain->stage + 1ull;
			fprintf(stderr,
				"CHAIN-HEARTBEAT slot=%u stage=%u layer=%u age_ms=%llu lease=%llu begun=%u recorded=%u — chain still advancing (a SILENT gap between these = the lost-continuation site)\n",
				chain->slot_index,(unsigned)chain->stage,
				(unsigned)chain->next_layer,
				(unsigned long long)((now_ns - chain->created_ns) / 1000000ull),
				(unsigned long long)chain->expert_lease,
				(unsigned)chain->expert_lease_begun,
				(unsigned)chain->expert_lease_recorded);
		}
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	{
		uint64_t now_ns = SparkGlm5NextNowNs();
		if ( state->chain_profile_last_ns != 0ull &&
		     state->chain_profile_stage < 8u )
			state->chain_stage_ns[state->chain_profile_stage] +=
				now_ns - state->chain_profile_last_ns;
		state->chain_profile_last_ns = now_ns;
		state->chain_profile_stage = (uint32_t)chain->stage;
	}
	{
		static uint32_t chain_trace_count;
		if ( chain_trace_count < 400u )
		{
			chain_trace_count++;
			fprintf(stderr,"CHAIN slot=%u stage=%u layer=%u rows=%u mi=%llu hi=%llu\n",
				chain->slot_index,chain->stage,chain->next_layer,chain->wave_rows,
				state->tp_device_collective_initialized != 0u ?
					(unsigned long long)SparkTpDeviceCollectiveRoundIndex(&state->tp_device_collective) : 0ull,
				state->tp_device_collective_hc_initialized != 0u ?
					(unsigned long long)SparkTpDeviceCollectiveRoundIndex(&state->tp_device_collective_hc) : 0ull);
		}
	}
	switch ( chain->stage )
	{
	case SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN:
		if ( state->graph_gate_printed < 3u )
		{
			state->graph_gate_printed++;
			fprintf(stderr,
			    "GRAPH-GATE rows=%u first=%u coll=%u lazy=%u deg=%u enabled=%u flags=%u\n",
			    (unsigned)chain->wave_rows,(unsigned)chain->first_row,
			    (unsigned)state->tp_device_collective_initialized,
			    (unsigned)(state->lazy_pack != 0),
			    (unsigned)state->tp_degree,
			    (unsigned)state->graph_path_enabled,
			    (unsigned)chain->context->flags);
		}
		if ( state->graph_path_enabled != 0u &&
		     state->experts_warm == 0u &&
		     state->graph_gate_printed < 3u )
			fprintf(stderr,
			    "GRAPH-GATE-COLD experts not warm; eager first\n");
		if ( chain->wave_rows >= 1u && chain->wave_rows <= SPARK_GLM5_NEXT_GRAPH_ROWS_MAX &&
		     (chain->slot->graph_failed_rows & (UINT64_C(1) << (chain->wave_rows - 1u))) == 0u &&
		     chain->first_row == 0u && chain->spec_verify == 0u &&
		     chain->slot->sampled == 0u &&
		     chain->batch->active_sequence_count == chain->wave_rows &&
		     state->tp_device_collective_initialized != 0u &&
		     state->lazy_pack != 0 && state->tp_degree > 1u &&
		     state->graph_path_enabled != 0u &&
		     state->experts_warm != 0u )
		{
			SparkStatus graph_status;
			if ( SparkGlm5NextBuildWave(chain) != SPARK_STATUS_OK )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
				return;
			}
			SparkGlm5NextGraphEnsure(chain,&graph_status);
			if ( graph_status == SPARK_STATUS_OK )
			{
				SparkGlm5NextFinishChain(chain);
				return;
			}
			if ( graph_status != SPARK_STATUS_UNSUPPORTED )
			{
				SparkGlm5NextTerminalFailure(state,graph_status,"graph-execution");
				fprintf(stderr,"GRAPH-PATH-FAILED status=%d; engine restart required\n",
					(int32_t)graph_status);
				SparkGlm5NextTpChainFail(chain,graph_status);
				return;
			}
		}
		if ( SparkGlm5NextBuildWave(chain) != SPARK_STATUS_OK )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
			return;
		}
		SparkGlm5NextT1Wave(&chain->wave);
		if ( SparkGlm5NextLaunchCudaWaveBegin(&chain->wave) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION;
		chain->next_layer = 0u;
		launch_status = SparkGlm5NextModuleReduceHidden(chain,chain->slot->hidden_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION:
		gather_sequences = SparkGlm5NextLayerIndexGatherSequences(&chain->wave,chain->next_layer);
		if ( gather_sequences != 0u )
		{
			if ( SparkGlm5NextLaunchCudaLayerAttentionScore(&chain->wave,chain->next_layer) != 0 )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_GATHER_INDEX;
			launch_status = SparkGlm5NextModuleGatherIndex(chain,gather_sequences,1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		if ( SparkGlm5NextLaunchCudaLayerAttention(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_GATHER_INDEX:
		if ( SparkGlm5NextLaunchCudaLayerAttentionSelect(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION:
		if ( SparkGlm5NextLaunchCudaLayerAttentionPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_MLP;
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_MLP:
		if ( state->lazy_pack != 0 && (chain->wave.first_layer_index + chain->next_layer) >= SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
		{
			if ( chain->wave.expert_lease_all != 0u &&
			     chain->wave_rows == 1u &&
			     chain->next_layer >= 900u )
			{
				if ( SparkGlm5NextLaunchCudaLayerMlpRoute(&chain->wave,chain->next_layer) != 0 ||
				     SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
				{
					SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
					return;
				}
				if ( state->decode_miss_host != 0 &&
				     state->decode_miss_host[0] != 0u )
				{
					fprintf(stderr,
					    "EXPERT-MISS-EAGER slot=%u layer=%u — lazy load missed a routed expert; failing\n",
					    (unsigned)chain->slot_index,
					    (unsigned)chain->next_layer);
					SparkGlm5NextTpChainFail(
					    chain,SPARK_STATUS_INTERNAL_ERROR);
					return;
				}
				{
						if ( SparkGlm5NextLaunchCudaLayerMlpExperts(&chain->wave,chain->next_layer) != 0 )
					{
						SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
						return;
					}
					SparkGlm5NextTpChainReduceMlp(chain);
					return;
				}
			}
			{
				const uint32_t *cover_saved = chain->wave.expert_cover;
				void *miss_saved = chain->wave.expert_miss;
				chain->wave.expert_cover = 0;
				chain->wave.expert_miss = 0;
				launch_status = SparkGlm5NextLaunchCudaLayerMlpRoute(&chain->wave,chain->next_layer) != 0 ?
				    SPARK_STATUS_INTERNAL_ERROR : SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyWork,chain);
				chain->wave.expert_cover = cover_saved;
				chain->wave.expert_miss = miss_saved;
			}
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		if ( SparkGlm5NextLaunchCudaLayerMlp(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		SparkGlm5NextTpChainReduceMlp(chain);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP:
		if ( SparkGlm5NextLaunchCudaLayerMlpPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		SparkGlm5NextT1Streams(chain,chain->wave.first_layer_index + chain->next_layer);
		SparkGlm5NextT1Route(chain,chain->wave.first_layer_index + chain->next_layer);
		chain->next_layer++;
		if ( chain->next_layer < chain->wave.layer_count )
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		else
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD:
		if ( SparkGlm5NextLaunchCudaWaveHead(&chain->wave) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD;
		launch_status = SparkGlm5NextModuleReduceHeadMax(chain);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD:
		error = SparkGlm5NextLaunchHeadMaxlocUnpack((cudaStream_t)chain->slot->stream,chain->slot->head_maxloc_u64,chain->slot->output_token,chain->wave_rows);
		if ( error == cudaSuccess && state->owns_final_head != 0u )
			error = cudaMemcpyAsync(chain->slot->host_output_token_ids + chain->first_row,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream);
		launch_status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_head_unpack");
		if ( (state->hbound_probes & (1u << 31u)) == 0u &&
		     SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) == 0 )
		{
			state->hbound_probes |= 1u << 31u;
			fprintf(stderr,"HEADFIN v=%u\n",
			    chain->slot->host_output_token_ids[chain->first_row]);
		}
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		SparkGlm5NextT1Head(chain);
		if ( chain->spec_verify != 0u )
		{
			error = cudaLaunchHostFunc((cudaStream_t)chain->slot->stream,SparkGlm5NextMtpResolveHost,chain);
			if ( error != cudaSuccess )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			return;
		}
		if ( state->mtp_enabled != 0u && state->owns_final_head != 0u )
		{
			launch_status = SparkGlm5NextMtpStashHidden(state,chain);
			if ( launch_status != SPARK_STATUS_OK )
			{
				SparkGlm5NextTpChainFail(chain,launch_status);
				return;
			}
		}
		if ( state->experts_warm == 0u )
		{
			state->experts_warm = 1u;
			fprintf(stderr,
			    "GRAPH-WARM experts resident after first eager chain\n");
		}
		SparkGlm5NextFinishChain(chain);
		return;
	default:
		SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
}

static void SparkGlm5NextPrepareAsyncCompletion(
	SparkGlm5NextModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	const uint8_t *lane_bound,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions,
	uint32_t slot_index)
{
	SparkGlm5NextAsyncCompletion *async;
	uint32_t lane;
	async = &state->completions[slot_index];
	memset(async,0,sizeof(*async));
	async->state = state;
	async->completion_function = frame->completion_function;
	async->completion_context = frame->completion_context;
	async->slot_index = slot_index;
	async->lane_count = batch->active_sequence_count;
	async->row_count = batch->row_count;
	async->state_capture = ((const SparkGlm5NextResidentDecodeStageFrameContext *)frame->user_context)->state_capture;
	async->output_token_destination = state->owns_final_head != 0u ? (uint32_t *)frame->buffers[0].address : 0;
	for (lane=0u; lane<batch->active_sequence_count && lane<SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
	{
		async->lane_indices[lane] = batch->row_resident_slots[lane];
		async->lane_bound[lane] = lane_bound[lane];
		async->lane_sequence_ids[lane] = lane_sequence_ids[lane];
		async->lane_next_positions[lane] = lane_next_positions[lane];
	}
	async->completion.request_id = frame->request_id;
	async->completion.sequence_id = frame->sequence_id;
	async->completion.sequence_position = frame->sequence_position;
	async->completion.program_id = frame->program_id;
	async->completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	async->completion.accepted_token_count = frame->new_token_count;
	async->completion.tokens_per_sequence = frame->tokens_per_sequence;
	async->completion.status = SPARK_STATUS_OK;
	async->completion.residency = frame->residency;
	async->completion.host_staging_bytes = (uint64_t)batch->row_count * (sizeof(uint32_t) * (3u + state->owns_final_head) + sizeof(SparkRowSampling) * state->owns_final_head);
	async->completion.device_memcpy_bytes = async->completion.host_staging_bytes;
}

static SparkStatus SparkGlm5NextCaptureRecurrent(SparkGlm5NextModuleState *state,uint32_t resident)
{
	SparkKvLaneTransaction *owner;
	SparkKvPageCacheSequence *sequence;
	uint32_t page;
	uint64_t generation;
	SparkStatus status;
	if ( state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( resident >= state->resident_sequence_capacity || state->kv_lane_transactions == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	owner = &state->kv_lane_transactions[resident];
	if ( (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) == 0u )
		return(SPARK_STATUS_OK);
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	sequence = &state->kv_transactions.cache->sequences[resident];
	page = sequence->mutable_logical_page_index;
	if ( page >= state->page_count || state->kv_blocks[page].residency_reference_count == 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	generation = state->kv_blocks[page].generation;
	status = SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,resident,state->recurrent_staging,state->recurrent_page_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvPageStoreWriteback(&state->recurrent_store,page,resident,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		SparkStatus wait = SparkKvPageStoreWaitForTransfers(&state->recurrent_store);
		if ( wait != SPARK_STATUS_OK )
			SPARK_RETURN(wait);
		status = SparkKvPageStoreWriteback(&state->recurrent_store,page,resident,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes,0u,0u);
	}
	SPARK_RETURN(status);
}

static void SparkGlm5NextCaptureClearUnused(uint8_t *payload,uint64_t bytes,uint32_t layers,uint32_t valid_tokens)
{
	uint64_t layer_bytes = bytes / layers,valid_bytes = layer_bytes / 64u * valid_tokens;
	uint32_t layer;
	for (layer=0u; layer<layers; layer++)
		memset(payload + layer * layer_bytes + valid_bytes,0,(size_t)(layer_bytes - valid_bytes));
}

static SparkStatus SparkGlm5NextCaptureState(SparkGlm5NextAsyncCompletion *async)
{
	SparkGlm5NextModuleState *state = async->state;
	SparkGlm5NextStateCapture *capture = async->state_capture;
	SparkGlm5NextExecutionSlot *slot = &state->slots[async->slot_index];
	uint32_t lane,page,resident,logical,physical,valid_tokens,last_row;
	uint64_t offset = 0u,key_bytes,index_bytes,hidden_bytes,map_offset;
	SparkKvLaneTransaction *owner;
	SparkGlm5NextStateCaptureLane *out;
	SparkStatus status;
	if ( capture == 0 )
		return(SPARK_STATUS_OK);
	capture->payload_bytes = 0u;
	key_bytes = state->kv_arena.key_block_stride_bytes;
	index_bytes = state->kv_arena.value_block_stride_bytes;
	hidden_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	for (lane=0u; lane<async->lane_count; lane++)
	{
		resident = async->lane_indices[lane];
		owner = &state->kv_lane_transactions[resident];
		out = &capture->lanes[lane];
		if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING || owner->page_count != SparkCeilDivU32((uint32_t)async->lane_next_positions[lane],64u) || owner->page_count > capture->pages_per_lane_capacity )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		*out = (SparkGlm5NextStateCaptureLane){.sequence_id=async->lane_sequence_ids[lane],.next_position=async->lane_next_positions[lane],.payload_offset=offset,.resident_slot=resident,.page_count=owner->page_count};
		for (page=0u; page<owner->page_count; page++)
		{
			map_offset = (uint64_t)resident * state->pages_per_sequence + page;
			logical = state->kv_lane_logical_pages[map_offset];
			physical = state->kv_lane_physical_pages[map_offset];
			if ( logical >= state->page_count || physical >= state->physical_page_count || state->kv_blocks[logical].resident_slot_index != physical || state->kv_blocks[logical].residency_reference_count == 0u )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			capture->logical_pages[(uint64_t)lane * capture->pages_per_lane_capacity + page] = logical;
			capture->physical_pages[(uint64_t)lane * capture->pages_per_lane_capacity + page] = physical;
			status = SparkGlm5NextPageCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,state->kv_blocks[logical].key_device_address,capture->payload + offset,key_bytes);
			if ( status != SPARK_STATUS_OK )
				return(status);
			valid_tokens = (uint32_t)(async->lane_next_positions[lane] - (uint64_t)page * 64u);
			if ( valid_tokens > 64u )
				valid_tokens = 64u;
			SparkGlm5NextCaptureClearUnused(capture->payload + offset,key_bytes,state->kv_layer_count,valid_tokens);
			offset += key_bytes;
			status = SparkGlm5NextPageCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,state->kv_blocks[logical].value_device_address,capture->payload + offset,index_bytes);
			if ( status != SPARK_STATUS_OK )
				return(status);
			SparkGlm5NextCaptureClearUnused(capture->payload + offset,index_bytes,state->index_layer_count,valid_tokens);
			offset += index_bytes;
		}
		status = state->recurrent_page_bytes != 0u ? SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,resident,capture->payload + offset,state->recurrent_page_bytes) : SPARK_STATUS_OK;
		if ( status != SPARK_STATUS_OK )
			return(status);
		offset += state->recurrent_page_bytes;
		last_row = async->row_count;
		while ( last_row != 0u && slot->host_resident_slots[last_row - 1u] != resident )
			last_row--;
		if ( last_row == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		last_row--;
		if ( cudaMemcpy(capture->payload + offset,slot->hidden_bf16 + last_row * hidden_bytes / sizeof(uint16_t),hidden_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(&out->output_score,slot->output_score + last_row,sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		offset += hidden_bytes;
		out->payload_bytes = offset - out->payload_offset;
	}
	capture->key_page_bytes = key_bytes;
	capture->index_page_bytes = index_bytes;
	capture->key_layer_count = state->kv_layer_count;
	capture->index_layer_count = state->index_layer_count;
	capture->recurrent_bytes = state->recurrent_page_bytes;
	capture->hidden_bytes = hidden_bytes;
	capture->backing_write_count = state->kv_page_store.write_count;
	capture->backing_read_count = state->kv_page_store.read_count;
	capture->prefix_hit_count = state->kv_page_cache.prefix_hit_count;
	capture->payload_bytes = offset;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextFinishCacheLanes(SparkGlm5NextAsyncCompletion *async)
{
	SparkGlm5NextModuleState *state = async->state;
	SparkStatus result;
	uint32_t lane,resident;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	result = async->completion.status;
	if ( result == SPARK_STATUS_OK )
		result = SparkGlm5NextCaptureState(async);
	for (lane=0u; lane<async->lane_count && result==SPARK_STATUS_OK; lane++)
		result = SparkGlm5NextCaptureRecurrent(state,async->lane_indices[lane]);
	result = SparkKvLaneTransactionsFinish(&state->kv_transactions,async->lane_indices,async->lane_count,result,(uint32_t)async->mtp_cache_extra);
	for (lane=0u; lane<async->lane_count; lane++)
	{
		resident = async->lane_indices[lane];
		atomic_store_explicit(&state->lane_bound[resident],result == SPARK_STATUS_OK ? async->lane_bound[lane] : 0u,memory_order_release);
		if ( result == SPARK_STATUS_OK )
		{
			atomic_store_explicit(&state->lane_sequence_ids[resident],async->lane_sequence_ids[lane],memory_order_release);
			atomic_store_explicit(&state->lane_next_positions[resident],async->lane_next_positions[lane],memory_order_release);
		}
		else
		{
			memset(state->page_table_shadow + (uint64_t)resident * state->pages_per_sequence,0xff,(uint64_t)state->pages_per_sequence * sizeof(uint32_t));
			if ( state->mtp_lane_armed != 0 )
				state->mtp_lane_armed[resident] = 0u;
		}
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(result);
}

static SparkStatus SparkGlm5NextCompletionStatus(
	SparkGlm5NextAsyncCompletion *async,SparkGlm5NextExecutionSlot *slot)
{
	SparkStatus status = async->completion.status;
	uint32_t output_count,output;
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkGlm5NextWeightdHealth(async->state);
	if ( status != SPARK_STATUS_OK || async->output_token_destination == 0 )
		return(status);
	output_count = async->burst_token_count != 0u ?
		async->burst_token_count : async->row_count;
	for ( output = 0u; output < output_count; output++ )
		if ( slot->host_output_token_ids[output] >=
		        SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT )
			return(SparkGlm5NextTerminalFailure(async->state,
			    SPARK_STATUS_VALIDATION_FAILED,"output-token-range"));
	return(SPARK_STATUS_OK);
}

static uint64_t SparkGlm5NextSpanNs(uint64_t start_ns,uint64_t end_ns)
{
	return(start_ns != 0u && end_ns >= start_ns ? end_ns - start_ns : 0u);
}

static void SparkGlm5NextWaveTimingReport(SparkGlm5NextWaveTiming *timing,uint32_t tp_rank,uint64_t now_ns)
{
	uint64_t delivered_ns = timing->delivered_ns;
	fprintf(stderr,"G5N-WAVE-TIMING rank=%u waves=%llu rows=%llu retries=%llu idle_us=%llu/%llu pre_us=%llu/%llu key_us=%llu/%llu gpu_us=%llu/%llu post_us=%llu/%llu idle_ms=%llu pre_ms=%llu key_ms=%llu gpu_ms=%llu post_ms=%llu source_wait_ms=%llu peer_wait_ms=%llu copy_ms=%llu combine_ms=%llu worst_ms=%llu worst_request=%llu worst_epochs=%llu/%llu worst_us=%llu/%llu/%llu/%llu/%llu\n",
		tp_rank,(unsigned long long)timing->waves,(unsigned long long)timing->rows,(unsigned long long)timing->retries,
		(unsigned long long)SparkLatencyPercentileUs(&timing->idle,50u),(unsigned long long)SparkLatencyPercentileUs(&timing->idle,99u),
		(unsigned long long)SparkLatencyPercentileUs(&timing->pre,50u),(unsigned long long)SparkLatencyPercentileUs(&timing->pre,99u),
		(unsigned long long)SparkLatencyPercentileUs(&timing->key,50u),(unsigned long long)SparkLatencyPercentileUs(&timing->key,99u),
		(unsigned long long)SparkLatencyPercentileUs(&timing->gpu,50u),(unsigned long long)SparkLatencyPercentileUs(&timing->gpu,99u),
		(unsigned long long)SparkLatencyPercentileUs(&timing->post,50u),(unsigned long long)SparkLatencyPercentileUs(&timing->post,99u),
		(unsigned long long)(timing->idle.total_ns / 1000000u),(unsigned long long)(timing->pre.total_ns / 1000000u),
		(unsigned long long)(timing->key.total_ns / 1000000u),(unsigned long long)(timing->gpu.total_ns / 1000000u),
		(unsigned long long)(timing->post.total_ns / 1000000u),(unsigned long long)(timing->source_wait_ns / 1000000u),
		(unsigned long long)(timing->peer_wait_ns / 1000000u),(unsigned long long)(timing->copy_ns / 1000000u),
		(unsigned long long)(timing->combine_ns / 1000000u),(unsigned long long)(timing->worst_ns / 1000000u),
		(unsigned long long)timing->worst_request,(unsigned long long)timing->worst_epoch[0],(unsigned long long)timing->worst_epoch[1],
		(unsigned long long)(timing->worst_part_ns[0] / 1000u),(unsigned long long)(timing->worst_part_ns[1] / 1000u),
		(unsigned long long)(timing->worst_part_ns[2] / 1000u),(unsigned long long)(timing->worst_part_ns[3] / 1000u),
		(unsigned long long)(timing->worst_part_ns[4] / 1000u));
	memset(timing,0,sizeof(*timing));
	timing->delivered_ns = delivered_ns;
	timing->window_ns = now_ns;
}

static void SparkGlm5NextWaveTimingWorst(SparkGlm5NextWaveTiming *timing,const SparkGlm5NextAsyncCompletion *async,uint64_t total_ns,uint64_t delivered_ns)
{
	timing->worst_ns = total_ns;
	timing->worst_request = async->completion.request_id;
	timing->worst_epoch[0] = async->epoch[0];
	timing->worst_epoch[1] = async->epoch[1];
	timing->worst_part_ns[0] = timing->delivered_ns != 0u && async->attempt_ns > timing->delivered_ns ? async->attempt_ns - timing->delivered_ns : 0u;
	timing->worst_part_ns[1] = SparkGlm5NextSpanNs(async->attempt_ns,async->launched_ns);
	timing->worst_part_ns[2] = SparkGlm5NextSpanNs(async->chain_start_ns,async->keyed_ns);
	timing->worst_part_ns[3] = SparkGlm5NextSpanNs(async->launched_ns,async->callback_ns);
	timing->worst_part_ns[4] = SparkGlm5NextSpanNs(async->callback_ns,delivered_ns);
}

static void SparkGlm5NextWaveTimingRecord(SparkGlm5NextWaveTiming *timing,const SparkGlm5NextAsyncCompletion *async,const SparkTpDeviceCollectiveHardwareTiming *collective,uint32_t tp_rank,uint64_t delivered_ns)
{
	uint64_t total_ns = SparkGlm5NextSpanNs(async->attempt_ns,delivered_ns);
	if ( timing->window_ns == 0u )
		timing->window_ns = delivered_ns;
	SparkLatencyAdd(&timing->idle,timing->delivered_ns,async->attempt_ns);
	SparkLatencyAdd(&timing->pre,async->attempt_ns,async->launched_ns);
	SparkLatencyAdd(&timing->key,async->chain_start_ns,async->keyed_ns);
	SparkLatencyAdd(&timing->gpu,async->launched_ns,async->callback_ns);
	SparkLatencyAdd(&timing->post,async->callback_ns,delivered_ns);
	timing->waves++;
	timing->rows += async->row_count;
	timing->retries += async->retries;
	timing->source_wait_ns += collective->source_wait_ns;
	timing->peer_wait_ns += collective->peer_wait_ns;
	timing->copy_ns += collective->copy_ns;
	timing->combine_ns += collective->combine_ns;
	if ( total_ns > timing->worst_ns )
		SparkGlm5NextWaveTimingWorst(timing,async,total_ns,delivered_ns);
	timing->delivered_ns = delivered_ns;
	if ( delivered_ns - timing->window_ns >= SPARK_GLM5_NEXT_WAVE_TIMING_WINDOW_NS )
		SparkGlm5NextWaveTimingReport(timing,tp_rank,delivered_ns);
}

static void SparkGlm5NextCompleteOnWorker(void *context)
{
	SparkGlm5NextAsyncCompletion *async = (SparkGlm5NextAsyncCompletion *)context;
	SparkGlm5NextModuleState *state = async != 0 ? async->state : 0;
	SparkGlm5NextExecutionSlot *slot;
	SparkModelDriverCompletion completion;
	SparkModelDriverCompletionFunction complete;
	SparkTpDeviceCollectiveHardwareTiming collective = {0};
	void *complete_context;
	if ( state == 0 )
		return;
	pthread_mutex_lock(&state->completion_queue_lock);
	SparkGlm5NextDrainParkedCompletions(state);
	pthread_mutex_unlock(&state->completion_queue_lock);
	if ( async->slot_index >= state->pipeline_slot_count )
		return;
	slot = &state->slots[async->slot_index];
	if ( async->completion.status != SPARK_STATUS_OK )
	{
		if ( state->tp_device_collective_initialized != 0u )
			SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective);
		if ( state->tp_device_collective_hc_initialized != 0u )
			SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective_hc);
	}
	if ( SparkGlm5NextBoundedStreamSync(state,state->execution_stream,UINT64_C(35000000000)) != 0 )
	{
		(void)SparkGlm5NextTerminalFailure(state,SPARK_STATUS_IO_ERROR,"completion-drain");
		fprintf(stderr,"GLM completion drain failed; retaining slot %u and chain ownership\n",async->slot_index);
		return;
	}
	if ( slot->host_kv_access_error[0] != 0u )
	{
		fprintf(stderr,"GLM cache access failed: code %u row %u slot %u\n",slot->host_kv_access_error[0],slot->host_kv_access_error[2],async->slot_index);
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	}
	async->completion.status = SparkGlm5NextCompletionStatus(async,slot);
	async->completion.status = SparkGlm5NextFinishCacheLanes(async);
	{
		SparkStatus end_status = SPARK_STATUS_OK;
		if ( state->tp_device_collective_initialized != 0u )
			end_status = SparkTpDeviceCollectiveEndChain(&state->tp_device_collective,state->execution_stream);
		if ( end_status == SPARK_STATUS_OK && state->tp_device_collective_hc_initialized != 0u )
			end_status = SparkTpDeviceCollectiveEndChain(&state->tp_device_collective_hc,state->execution_stream);
		if ( end_status != SPARK_STATUS_OK )
		{
			(void)SparkGlm5NextTerminalFailure(state,end_status,"collective-end");
			fprintf(stderr,"GLM chain end failed: slot %u status %d; retaining ownership\n",async->slot_index,(int)end_status);
			return;
		}
	}

	{
		uint64_t round_count = 0u,round_ns = 0u,chain_ns = SparkGlm5NextNowNs();
		SparkTpDeviceCollectiveRoundStats(&state->tp_device_collective,&round_count,&round_ns,1u);
		fprintf(stderr,"CHAIN-TIME slot=%u status=%d total_ms=%.2f collective_host_submit_ms=%.2f collective_host_submissions=%llu stage_ms=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f\n",
			(unsigned)async->slot_index,(int)async->completion.status,
			async->chain_start_ns != 0u ? (double)(chain_ns - async->chain_start_ns) / 1000000.0 : 0.0,
			(double)round_ns / 1000000.0,(unsigned long long)round_count,
			(double)state->chain_stage_ns[0] / 1000000.0,
			(double)state->chain_stage_ns[1] / 1000000.0,
			(double)state->chain_stage_ns[2] / 1000000.0,
			(double)state->chain_stage_ns[3] / 1000000.0,
			(double)state->chain_stage_ns[4] / 1000000.0,
			(double)state->chain_stage_ns[5] / 1000000.0,
			(double)state->chain_stage_ns[6] / 1000000.0,
			(double)state->chain_stage_ns[7] / 1000000.0);
		{
			SparkTpDeviceCollectiveHardwareTiming main_timing = {0},hc_timing = {0};
			SparkStatus main_status = SparkTpDeviceCollectiveHardwareStats(
				&state->tp_device_collective,&main_timing);
			SparkStatus hc_status = state->tp_device_collective_hc_initialized != 0u ?
				SparkTpDeviceCollectiveHardwareStats(&state->tp_device_collective_hc,&hc_timing) : SPARK_STATUS_OK;
			collective.source_wait_ns = main_timing.source_wait_ns + hc_timing.source_wait_ns;
			collective.peer_wait_ns = main_timing.peer_wait_ns + hc_timing.peer_wait_ns;
			collective.copy_ns = main_timing.copy_ns + hc_timing.copy_ns;
			collective.combine_ns = main_timing.combine_ns + hc_timing.combine_ns;
			if ( main_status == SPARK_STATUS_OK && hc_status == SPARK_STATUS_OK )
				fprintf(stderr,"COLLECTIVE-GPU-TIME slot=%u source_wait_ms=%.3f peer_wait_ms=%.3f copy_ms=%.3f combine_ms=%.3f\n",
					async->slot_index,(double)(main_timing.source_wait_ns + hc_timing.source_wait_ns) / 1000000.0,
					(double)(main_timing.peer_wait_ns + hc_timing.peer_wait_ns) / 1000000.0,
					(double)(main_timing.copy_ns + hc_timing.copy_ns) / 1000000.0,
					(double)(main_timing.combine_ns + hc_timing.combine_ns) / 1000000.0);
			else if ( main_status != SPARK_STATUS_UNSUPPORTED )
				fprintf(stderr,"COLLECTIVE-GPU-TIME-UNAVAILABLE slot=%u main=%d hc=%d\n",
					async->slot_index,(int)main_status,(int)hc_status);
		}
		memset(state->chain_stage_ns,0,sizeof(state->chain_stage_ns));
		state->chain_profile_last_ns = 0ull;
	}
	if ( async->completion.status == SPARK_STATUS_BUSY &&
	     ++async->finish_retries >= 2u )
	{
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
		fprintf(stderr,
		    "cache finish forced after retries slot=%u\n",
		    async->slot_index);
	}
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->completion.status == SPARK_STATUS_OK &&
		     async->output_token_destination != 0 )
		{
			memcpy(async->output_token_destination,slot->host_output_token_ids,(uint64_t)(async->burst_token_count != 0u ? async->burst_token_count : async->row_count) * sizeof(uint32_t));
		}
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	atomic_fetch_add_explicit(&state->host_callback_completion_count,1u,memory_order_relaxed);
	// Snapshot before releasing the slot: callback-driven reuse can overwrite async.
	completion = async->completion;
	complete = async->completion_function;
	complete_context = async->completion_context;
	SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,async->lane_indices,async->lane_count);
	atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
	SparkStageModuleSlotRelease(state->slot_states,async->slot_index);
	SparkGlm5NextWaveTimingRecord(&state->wave_timing,async,&collective,state->tp_rank,SparkGlm5NextNowNs());
	complete(complete_context,&completion);
}

static void SparkGlm5NextParkCompletionLocked(
    SparkGlm5NextModuleState *state,SparkWeightdWorkFunction function,void *context)
{
	uint32_t i;
	for (i=0u; i<SPARK_GLM5_NEXT_OVERFLOW_POOL; i++)
		if ( state->overflow_pool[i].function == 0 )
		{
			state->overflow_pool[i].function = function;
			state->overflow_pool[i].context = context;
			state->overflow_parked_count++;
			return;
		}
	(void)SparkGlm5NextTerminalFailure(state,SPARK_STATUS_CAPACITY_EXCEEDED,"completion-work-capacity");
	fprintf(stderr,"GLM callback work pool exhausted; retaining occupied slots\n");
}

static void SparkGlm5NextDrainParkedCompletions(SparkGlm5NextModuleState *state)
{
	uint32_t i;
	if ( state == 0 || state->completion_worker == 0 )
		return;
	for (i=0u; i<SPARK_GLM5_NEXT_OVERFLOW_POOL; i++)
		if ( state->overflow_pool[i].function != 0 )
		{
			if ( SparkWeightdWorkerSubmit(state->completion_worker,state->overflow_pool[i].function,state->overflow_pool[i].context) != SPARK_STATUS_OK )
				return;
			state->overflow_pool[i].function = 0;
			state->overflow_pool[i].context = 0;
			state->overflow_parked_count--;
		}
}

static void SparkGlm5NextScheduleCompletionWork(SparkGlm5NextModuleState *state,SparkWeightdWorkFunction function,void *context)
{
	pthread_mutex_lock(&state->completion_queue_lock);
	if ( SparkWeightdWorkerSubmit(state->completion_worker,function,context) != SPARK_STATUS_OK )
		SparkGlm5NextParkCompletionLocked(state,function,context);
	pthread_mutex_unlock(&state->completion_queue_lock);
}

static void CUDART_CB SparkGlm5NextCompleteAsync(void *context)
{
	SparkGlm5NextAsyncCompletion *async = context;
	if ( async != 0 )
		async->callback_ns = SparkGlm5NextNowNs();
	if ( async != 0 && async->state != 0 )
		SparkGlm5NextScheduleCompletionWork(async->state,SparkGlm5NextCompleteOnWorker,async);
}

static int SparkGlm5NextBoundedStreamSync(SparkGlm5NextModuleState *state,void *stream,uint64_t timeout_ns)
{
	SparkStatus status;
	if ( stream != state->execution_stream ) return(-1);
	status = SparkStageModuleCudaWaitFor(&state->stream_wait,timeout_ns);
	if ( status == SPARK_STATUS_BUSY )
		fprintf(stderr,"GLM stream receipt timeout after %llu ms; retaining chain ownership\n",(unsigned long long)(timeout_ns / UINT64_C(1000000)));
	return(status == SPARK_STATUS_OK ? 0 : status == SPARK_STATUS_BUSY ? 1 : -1);
}

static SparkStatus SparkGlm5NextClaimCacheFrame(SparkGlm5NextModuleState *state,const SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageBatchView *batch,const uint64_t *next_positions)
{
	uint32_t lane;
	SparkStatus status;
	if ( frame->cache_lanes == 0 || frame->cache_lane_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u || frame->driver_dispatch_slot != (uint32_t)(frame->request_id % state->pipeline_slot_count) )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		if ( frame->cache_lanes[lane].resident_sequence_slot != batch->row_resident_slots[lane] || frame->cache_lanes[lane].sequence_id != batch->row_sequence_ids[lane] || frame->cache_lanes[lane].sequence_position != batch->row_positions[lane] || frame->cache_lanes[lane].context_token_count != next_positions[lane] )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkKvLaneTransactionsClaim(&state->kv_transactions,frame);
	(void)pthread_mutex_unlock(&state->kv_mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextRestoreRecurrent(SparkGlm5NextModuleState *state,uint32_t resident)
{
	const SparkKvLaneTransaction *owner = &state->kv_lane_transactions[resident];
	SparkKvPageCache *cache = state->kv_transactions.cache;
	uint32_t entry,page;
	uint64_t generation;
	SparkStatus status;
	if ( SparkGlm5NextPrefixRestorePending(owner) == 0u || state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	entry = cache->sequences[resident].terminal_entry_index;
	if ( entry >= cache->entry_capacity || cache->entries[entry].token_count != owner->lane.sequence_position || cache->entries[entry].reference_count == 0u || cache->sequences[resident].sequence_id != owner->lane.sequence_id )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	page = cache->entries[entry].logical_page_index;
	if ( page >= state->page_count || state->kv_blocks[page].reference_count == 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( state->kv_blocks[page].residency_reference_count == 0u )
	{
		uint32_t mutable = cache->sequences[resident].mutable_logical_page_index;
		if ( owner->lane.sequence_position % cache->kv_cache_arena->block_token_count == 0u || mutable >= state->page_count || owner->page_count == 0u || state->kv_transactions.logical_pages[(uint64_t)resident * state->kv_transactions.page_capacity + owner->page_count - 1u] != mutable || state->kv_blocks[mutable].residency_reference_count == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	generation = state->kv_blocks[page].generation;
	status = SparkKvPageStoreReadback(&state->recurrent_store,page,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes);
	while ( status == SPARK_STATUS_BUSY )
	{
		SparkStatus wait = SparkKvPageStoreWaitForTransfers(&state->recurrent_store);
		if ( wait != SPARK_STATUS_OK )
			SPARK_RETURN(wait);
		status = SparkKvPageStoreReadback(&state->recurrent_store,page,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,resident,state->recurrent_staging,state->recurrent_page_bytes);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextRestoreCacheLanes(SparkGlm5NextModuleState *state,const SparkGlm5NextAsyncCompletion *async)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t lane;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (lane=0u; lane<async->lane_count && status==SPARK_STATUS_OK; lane++)
		status = SparkGlm5NextRestoreRecurrent(state,async->lane_indices[lane]);
	(void)pthread_mutex_unlock(&state->kv_mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextClaimTpChain(SparkGlm5NextModuleState *state)
{
	uint32_t expected = 0u;
	cudaError_t error;
	if ( !atomic_compare_exchange_strong_explicit(&state->tp_chain_active,&expected,1u,memory_order_acq_rel,memory_order_acquire) )
		return(SPARK_STATUS_BUSY);
	error = cudaStreamQuery((cudaStream_t)state->execution_stream);
	if ( error == cudaSuccess )
		return(SPARK_STATUS_OK);
	atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
	return(error == cudaErrorNotReady ? SPARK_STATUS_BUSY : SparkGlm5NextTerminalFailure(state,SPARK_STATUS_IO_ERROR,"chain-rearm"));
}

static SparkStatus SparkGlm5NextStartClaimedBatch(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context,uint32_t slot_index)
{
	SparkGlm5NextExecutionSlot *slot = &state->slots[slot_index];
	SparkGlm5NextTpChain *chain;
	SparkStatus status;
	cudaError_t error;
	uint32_t retained_index;
	for ( retained_index = 0u; retained_index < state->pipeline_slot_count; retained_index++ )
		if ( atomic_load_explicit(&state->lazy_retained[retained_index],memory_order_acquire) != 0 )
		{
			SparkGlm5NextScheduleRetainedRetry(state);
			SPARK_FAIL(SPARK_STATUS_BUSY);
		}
	status = SparkGlm5NextClaimTpChain(state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	chain = (SparkGlm5NextTpChain *)calloc(1u,sizeof(*chain));
	if ( chain == 0 )
	{
		atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkGlm5NextClaimCacheFrame(state,frame,context->batch,state->completions[slot_index].lane_next_positions);
	if ( status != SPARK_STATUS_OK )
	{
		atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
		free(chain);
		SPARK_RETURN(status);
	}
	chain->state = state;
	chain->slot = slot;
	chain->slot_index = slot_index;
	chain->frame = frame;
	chain->context = context;
	chain->batch = context->batch;
	state->completions[slot_index].chain_start_ns = SparkGlm5NextNowNs();
	state->completions[slot_index].attempt_ns = state->wave_attempt_ns;
	state->completions[slot_index].retries = state->wave_attempt_retries;
	chain->wave_rows = context->batch->row_count;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
	chain->active = 1u;
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	if ( state->tp_device_collective_initialized != 0u )
		status = SparkTpDeviceCollectiveChainKey(&state->tp_device_collective,chain->frame->request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK);
	if ( status == SPARK_STATUS_OK && state->tp_device_collective_hc_initialized != 0u )
		status = SparkTpDeviceCollectiveChainKey(&state->tp_device_collective_hc,chain->frame->request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK);
	state->completions[slot_index].keyed_ns = SparkGlm5NextNowNs();
	state->completions[slot_index].epoch[0] = SparkTpDeviceCollectiveChainEpoch(&state->tp_device_collective);
	state->completions[slot_index].epoch[1] = SparkTpDeviceCollectiveChainEpoch(&state->tp_device_collective_hc);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextRestoreCacheLanes(state,&state->completions[slot_index]);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextUploadPageTables(state,&state->completions[slot_index],slot->stream);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemsetAsync(slot->kv_access_error,0,SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),(cudaStream_t)slot->stream);
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"kv_access_reset");
	}
	if ( status == SPARK_STATUS_OK && chain->wave_rows == 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMtpDriveDraft(state,frame,context->batch,slot,chain);
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextStageHostSampling(const SparkGlm5NextModuleState *state,SparkGlm5NextExecutionSlot *slot,const SparkGlm5NextResidentDecodeStageBatchView *batch)
{
	uint32_t row;
	slot->sampled = 0u;
	for (row=0u; state->owns_final_head != 0u && row<batch->row_count; row++)
	{
		slot->host_row_sampling[row] = batch->row_sampling[row];
		slot->sampled |= batch->row_sampling[row].inverse_temperature != 0.0f ? 1u : 0u;
	}
}

static SparkStatus SparkGlm5NextExecuteBatch(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context)
{
	const SparkGlm5NextResidentDecodeStageBatchView *batch = context->batch;
	SparkGlm5NextClaimedContinuityContext continuity;
	SparkGlm5NextExecutionSlot *slot;
	uint8_t simulated_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_sequence[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_next[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t slot_index;
	uint64_t last_ordinal,attempt_ns = SparkGlm5NextNowNs();
	SparkStatus status;
	if ( state->wave_attempt_request != frame->request_id || state->wave_attempt_ns == 0u )
	{
		state->wave_attempt_request = frame->request_id;
		state->wave_attempt_ns = attempt_ns;
		state->wave_attempt_retries = 0u;
	}
	else
		state->wave_attempt_retries++;
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkTpChainOrdinal(frame->request_id,state->pipeline_slot_count,SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS - 1u,&last_ordinal);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"G5N-DBG submit-fail site=ordinal status=%d req=%llu\n",(int)status,(unsigned long long)frame->request_id);
		SPARK_RETURN(status);
	}
	continuity.state = state;
	continuity.batch = batch;
	continuity.bound = simulated_bound;
	continuity.sequence_ids = simulated_sequence;
	continuity.next_positions = simulated_next;
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count,SparkGlm5NextPrepareClaimedContinuity,&continuity);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"G5N-DBG submit-fail site=lane-claim status=%d req=%llu rows=%u\n",(int)status,(unsigned long long)frame->request_id,(unsigned)batch->active_sequence_count);
		SPARK_RETURN(status);
	}
	slot_index = (uint32_t)(frame->request_id % state->pipeline_slot_count);
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,&slot_index,1u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"G5N-DBG submit-fail site=slot-claim status=%d req=%llu slot=%u\n",(int)status,(unsigned long long)frame->request_id,(unsigned)slot_index);
	if ( status == SPARK_STATUS_OK )
	{
		slot = &state->slots[slot_index];
		slot->stream = frame->execution_stream;
		status = SparkGlm5NextStageHostBatch(state,slot,batch);
		if ( status == SPARK_STATUS_OK )
		{
			SparkGlm5NextStageHostSampling(state,slot,batch);
			SparkGlm5NextPrepareAsyncCompletion(state,frame,batch,simulated_bound,simulated_sequence,simulated_next,slot_index);
			status = SparkGlm5NextStartClaimedBatch(state,frame,context,slot_index);
		}
		if ( status != SPARK_STATUS_OK )
			SparkStageModuleSlotRelease(state->slot_states,slot_index);
	}
	if ( status != SPARK_STATUS_OK )
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextPublishCache(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame)
{
	SparkGlm5NextAsyncCompletion *async;
	SparkModelDriverCompletion completion = {0};
	uint32_t indices[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT],lane,resident,slot;
	SparkStatus status;
	if ( state == 0 || frame == 0 || frame->completion_function == 0 || frame->execution_stream != state->execution_stream || state->pipeline_slot_count == 0u || frame->flags != (SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH | SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) || frame->new_token_count != 0u || frame->tokens_per_sequence != 0u || frame->cache_lanes == 0 || frame->cache_lane_count == 0u || frame->cache_lane_count != frame->active_slot_count || frame->cache_lane_count > state->resident_sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	slot = (uint32_t)(frame->request_id % state->pipeline_slot_count);
	if ( frame->driver_dispatch_slot != slot )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK ) return(status);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		indices[lane] = frame->cache_lanes[lane].resident_sequence_slot;
	status = SparkStageModuleIndexSetClaim(state->lane_states,state->resident_sequence_capacity,indices,frame->cache_lane_count);
	if ( status != SPARK_STATUS_OK ) return(status);
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,&slot,1u);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,indices,frame->cache_lane_count);
		return(status);
	}
	for (lane=0u; lane<frame->cache_lane_count && status==SPARK_STATUS_OK; lane++)
	{
		const SparkModelDriverCacheLane *cache_lane = &frame->cache_lanes[lane];
		resident = indices[lane];
		if ( cache_lane->flags != SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH || cache_lane->publish_token_count == 0u || cache_lane->sequence_position != cache_lane->publish_token_count || cache_lane->context_token_count != cache_lane->publish_token_count || atomic_load(&state->lane_bound[resident]) == 0u || atomic_load(&state->lane_sequence_ids[resident]) != cache_lane->sequence_id || atomic_load(&state->lane_next_positions[resident]) != cache_lane->sequence_position )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	if ( status == SPARK_STATUS_OK )
	{
		if ( pthread_mutex_lock(&state->kv_mutex) != 0 ) status = SPARK_STATUS_INTERNAL_ERROR;
		else
		{
			status = SparkKvLaneTransactionsClaim(&state->kv_transactions,frame);
			(void)pthread_mutex_unlock(&state->kv_mutex);
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		async = &state->completions[slot];
		memset(async,0,sizeof(*async));
		async->state = state;
		async->slot_index = slot;
		async->lane_count = frame->cache_lane_count;
		for (lane=0u; lane<async->lane_count; lane++)
		{
			async->lane_indices[lane] = indices[lane];
			async->lane_bound[lane] = 1u;
			async->lane_sequence_ids[lane] = frame->cache_lanes[lane].sequence_id;
			async->lane_next_positions[lane] = frame->cache_lanes[lane].context_token_count;
		}
		atomic_fetch_add(&state->submitted_count,1u);
		completion.status = SparkGlm5NextFinishCacheLanes(async);
		completion.request_id = frame->request_id;
		completion.sequence_id = frame->sequence_id;
		completion.sequence_position = frame->sequence_position;
		completion.program_id = frame->program_id;
		completion.driver_dispatch_slot = slot;
		completion.residency = frame->residency;
		atomic_fetch_add(completion.status == SPARK_STATUS_OK ? &state->completed_count : &state->failed_count,1u);
	}
	SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,indices,frame->cache_lane_count);
	SparkStageModuleSlotRelease(state->slot_states,slot);
	if ( status == SPARK_STATUS_OK ) frame->completion_function(frame->completion_context,&completion);
	return(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	SparkGlm5NextModuleState *state;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	SparkStatus status;
	state = (SparkGlm5NextModuleState *)module_state;
	context = 0;
	if ( frame != 0 && (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH) != 0u )
		return(SparkGlm5NextPublishCache(state,frame));
	status = SparkGlm5NextValidateFrame(state,frame,&context);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"G5N-DBG execute: ValidateFrame -> %d\n",(int)status);
		if ( state != 0 )
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_RETURN(status);
	}
	status = SparkGlm5NextExecuteBatch(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

static void SparkGlm5NextAdmissionCost(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) * 3u + sizeof(uint64_t) * 2u + sizeof(SparkRowSampling));
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkGlm5NextResetExecutionState(SparkGlm5NextModuleState *state)
{
	cudaError_t error = cudaSuccess,drain;
	uint32_t lane;
	if ( state->kda_layer_count != 0u )
	{
		error = cudaMemsetAsync(state->kda_state_pools,0,state->kda_state_layer_stride_bytes * state->kda_layer_count,(cudaStream_t)state->execution_stream);
		if ( error == cudaSuccess )
			error = cudaMemsetAsync(state->kda_window_pools,0,state->kda_window_layer_stride_bytes * state->kda_layer_count * 3u,(cudaStream_t)state->execution_stream);
	}
	drain = cudaStreamSynchronize((cudaStream_t)state->execution_stream);
	if ( drain != cudaSuccess )
	{
		(void)SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,drain,"reset_stream_drain");
		SPARK_FAIL(SPARK_STATUS_PENDING);
	}
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"cache_reset"));
	memset(state->page_table_shadow,0xff,(uint64_t)state->resident_sequence_capacity * state->pages_per_sequence * sizeof(uint32_t));
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_store_explicit(&state->lane_bound[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_sequence_ids[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_next_positions[lane],0u,memory_order_release);
		if ( state->mtp_lane_armed != 0 )
			state->mtp_lane_armed[lane] = 0u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextResetClaimed(SparkGlm5NextModuleState *state,uint64_t generation)
{
	SparkStatus status;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_VALIDATION_FAILED;
	if ( generation > state->reset_generation )
	{
		status = SparkKvLaneTransactionsReset(&state->kv_transactions);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextResetExecutionState(state);
		if ( status == SPARK_STATUS_OK )
		{
			state->reset_generation = generation;
			state->control_generation = 0u;
		}
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextReset(SparkGlm5NextModuleState *state,const SparkModelDriverAdmissionRequest *request)
{
	uint32_t slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t lanes[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t index;
	SparkStatus status;
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	/* A reset kills this rank's in-flight chains; every peer is potentially
	 * waiting on this rank's cells for those chains. Cancel first so the
	 * peers' waits fail fast (CKEY-CANCEL) instead of wedging 30s per chain
	 * — the session-reset cascade turned one rank's reconnect into a
	 * fleet-wide stall. */
	SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective);
	if ( state->tp_device_collective_hc_initialized != 0u )
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective_hc);
	for (index=0u; index<state->pipeline_slot_count; index++)
		slots[index] = index;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		lanes[index] = index;
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkStageModuleIndexSetClaim(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkGlm5NextResetClaimed(state,request->control_generation);
		if ( status == SPARK_STATUS_PENDING )
		{
			fprintf(stderr,"GLM reset stream not quiescent; retaining lane and slot ownership\n");
			SPARK_RETURN(status);
		}
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	}
	SparkStageModuleIndexSetRelease(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	SPARK_RETURN(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm5NextModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available;
	SparkStatus status;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		SparkModelDriverInitializeAdmissionDecision(decision);
		status = SparkGlm5NextReset(state,request);
		if ( status == SPARK_STATUS_OK )
		{
			decision->accepted = 1u;
			decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		}
		SPARK_RETURN(status);
	}
	available = request->request_id != 0u && atomic_load_explicit(&state->slot_states[request->request_id % state->pipeline_slot_count],memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE ? 1u : 0u;
	if ( (request->frame_flags & (SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE | SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH)) != 0u )
	{
		if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u || request->new_token_count != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		SparkModelDriverInitializeAdmissionDecision(decision);
		decision->available_dispatch_slot_count = available;
		return(SparkGlm5NextAdmissionPredicate(state,request,decision));
	}
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->resident_sequence_capacity;
	table.max_input_row_count = state->execution_row_capacity;
	table.max_sequence_positions = state->max_sequence_positions;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS |
		SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG;
	table.predicate = SparkGlm5NextAdmissionPredicate;
	table.predicate_context = state;
	table.cost = SparkGlm5NextAdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available,request,decision);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( decision->accepted == 0u )
		fprintf(stderr,"G5N-DBG admit: shape-rejected reason %u\n",
			(unsigned)decision->rejection_reason);
	if ( decision->accepted == 0u )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkGlm5NextModuleState *state;
	uint32_t index,resident_count;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 || snapshot == 0 || program_id == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageModuleRuntimeSnapshotInitialize(snapshot,program_id,state->slot_states,state->pipeline_slot_count);
	snapshot->submitted_count = atomic_load_explicit(&state->submitted_count,memory_order_relaxed);
	snapshot->completed_count = atomic_load_explicit(&state->completed_count,memory_order_relaxed);
	snapshot->rejected_count = atomic_load_explicit(&state->rejected_count,memory_order_relaxed);
	snapshot->host_callback_completion_count = atomic_load_explicit(&state->host_callback_completion_count,memory_order_relaxed);
	resident_count = 0u;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		resident_count += atomic_load_explicit(&state->lane_bound[index],memory_order_acquire) != 0u ? 1u : 0u;
	snapshot->resident_sequence_count = resident_count;
	snapshot->kv_token_capacity = (uint64_t)state->page_count * SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT;
	return(SparkGlm5NextWeightdHealth(state));
}

static SparkStatus SparkGlm5NextReleaseCaches(SparkGlm5NextModuleState *state)
{
	if ( state->stream_wait.initialized != 0u && SparkStageModuleCudaWaitDestroy(&state->stream_wait) != SPARK_STATUS_OK )
		return(SPARK_STATUS_BUSY);
	if ( state->completion_queue_lock_initialized != 0u )
	{
		(void)pthread_mutex_destroy(&state->completion_queue_lock);
		state->completion_queue_lock_initialized = 0u;
	}
	if ( state->recurrent_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->recurrent_store);
	if ( state->recurrent_staging != 0 )
		(void)cudaFreeHost(state->recurrent_staging);
	if ( state->kv_page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->kv_page_store);
	free(state->kda_state_index_host);
	free(state->kv_blocks);
	free(state->kv_resident_slot_logical_block_indices);
	free(state->kv_entries);
	free(state->kv_sequences);
	free(state->kv_hash_bucket_heads);
	free(state->kv_entry_indices_by_logical_page);
	free(state->kv_page_staging);
	free(state->kv_lane_logical_pages);
	free(state->page_table_shadow);
	free(state->kv_lane_transactions);
	if ( state->kv_lane_physical_pages != 0 )
		(void)cudaFreeHost(state->kv_lane_physical_pages);
	if ( state->kv_mutex_initialized != 0u )
		(void)pthread_mutex_destroy(&state->kv_mutex);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextReleaseCollectives(SparkGlm5NextModuleState *state)
{
	if ( state->tp_device_collective_hc_initialized != 0u )
	{
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective_hc);
		if ( state->tp_device_collective_hc.implementation != 0 ) return(SPARK_STATUS_IO_ERROR);
		state->tp_device_collective_hc_initialized = 0u;
	}
	if ( state->tp_hc_host_credit_send_bf16 != 0 )
		(void)cudaFreeHost(state->tp_hc_host_credit_send_bf16);
	if ( state->tp_hc_host_credit_receive_bf16 != 0 )
		(void)cudaFreeHost(state->tp_hc_host_credit_receive_bf16);
	state->tp_hc_host_credit_send_bf16 = 0;
	state->tp_hc_host_credit_receive_bf16 = 0;
	if ( state->tp_device_collective_initialized != 0u )
	{
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
		if ( state->tp_device_collective.implementation != 0 ) return(SPARK_STATUS_IO_ERROR);
		state->tp_device_collective_initialized = 0u;
	}
	return(SPARK_STATUS_OK);
}

void SparkGlm5NextResidentDecodeStageDestroy(void *module_state)
{
	SparkGlm5NextModuleState *state;
	uint32_t slot;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 )
		return;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( atomic_load_explicit(&state->lazy_retained[slot],memory_order_acquire) != 0 )
			break;
	if ( slot < state->pipeline_slot_count )
	{
		if ( state->lazy_pack == 0 || state->lazy_pack->worker == 0 || SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyRetryRetained,state) != SPARK_STATUS_OK )
			return;
	}
	if ( SparkStageModuleWaitForSlots(SPARK_GLM5_NEXT_MODULE_TAG,state->slot_states,state->pipeline_slot_count,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 && state->lazy_pack->worker != 0 && SparkWeightdWorkerWaitIdle(state->lazy_pack->worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->completion_worker != 0 )
	{
		if ( SparkWeightdWorkerWaitIdle(state->completion_worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK || SparkWeightdWorkerDestroy(state->completion_worker) != SPARK_STATUS_OK )
			return;
		state->completion_worker = 0;
	}
	if ( SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)state->execution_stream),"destroy_stream_drain") != SPARK_STATUS_OK )
		return;
	if ( state->decode_cover_host != 0 )
		(void)cudaFreeHost(state->decode_cover_host);
	if ( state->decode_miss_host != 0 )
		(void)cudaFreeHost(state->decode_miss_host);
	if ( state->decode_cover_device != 0 )
		(void)cudaFree(state->decode_cover_device);
	state->decode_cover_host = 0;
	state->decode_miss_host = 0;
	state->decode_cover_device = 0;
	if ( SparkGlm5NextReleaseCollectives(state) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 )
	{
		if ( SparkGlm5NextReleasePinnedExperts(state) != SPARK_STATUS_OK || SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
			return;
		state->lazy_pack = 0;
	}
	if ( state->lane_client != 0 )
	{
		(void)SparkWeightdClientClose(state->lane_client);
		state->lane_client = 0;
	}
	if ( SparkGlm5NextReleaseCaches(state) != SPARK_STATUS_OK )
		return;
	SparkGlm5NextReleaseSlotHost(state);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state->mtp_lane_armed);
	free(state);
}

static SparkStatus SparkGlm5NextBuildHeadShadow(SparkGlm5NextModuleState *state)
{
	uint64_t head_rows,dim;
	SparkStatus status;
	if ( state->owns_final_head == 0u || state->lm_head_bf16 == 0 )
		return(SPARK_STATUS_OK);
	head_rows = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
	dim = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	status = SparkGlm5NextAllocateBytes(state,head_rows,dim,1u,(void **)&state->head_certified_fp8_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_scale_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,SparkGlm5NextLaunchHeadCertifiedQuantize(0,state->lm_head_bf16,state->head_certified_fp8_payload,state->head_certified_fp8_scale_f32,state->head_certified_fp8_norm_f32,(uint32_t)head_rows,(uint32_t)dim),"head_certified_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaDeviceSynchronize(),"head_certified_sync");
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextConfigureExecution(SparkGlm5NextModuleState *state)
{
	{
		const char *prefetch = getenv("SPARK_GLM5_NEXT_PREFETCH");
		if ( prefetch != 0 && strcmp(prefetch,"0") != 0 )
		{
			fprintf(stderr,"SPARK_GLM5_NEXT_PREFETCH is unsupported; warm the daemon with sparkpipe_weightd_warm before serving\n");
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
	}
	{
		const char *graph_env = getenv("SPARK_GLM5_NEXT_GRAPH_PATH");
		const char *record_limit_env =
		    getenv("SPARK_GLM5_NEXT_GRAPH_RECORD_OPS");
		if ( graph_env == 0 || (strcmp(graph_env,"0") != 0 && strcmp(graph_env,"1") != 0) )
		{
			fprintf(stderr,"SPARK_GLM5_NEXT_GRAPH_PATH must be 0 or 1\n");
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->graph_path_enabled = strcmp(graph_env,"1") == 0 ? 1u : 0u;
		fprintf(stderr,"GLM execution mode=%s\n",state->graph_path_enabled != 0u ? "graph" : "eager");
		state->graph_record_limit = record_limit_env != 0 ?
		    (uint32_t)strtoul(record_limit_env,0,10) : 0u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextInitializeState(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	SparkGlm5NextModuleState **state_out)
{
	SparkGlm5NextModuleState *state;
	const char *pack_path;
	SparkStatus status;
	uint32_t lane;
	state = (SparkGlm5NextModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_GLM5_NEXT_MODULE_TAG;
	atomic_init(&state->terminal_status,SPARK_STATUS_OK);
	atomic_init(&state->tp_chain_active,0u);
	status = SparkGlm5NextConfigureExecution(state);
	if ( status != SPARK_STATUS_OK )
	{
		free(state);
		SPARK_RETURN(status);
	}
	for (lane=0u; lane<SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; lane++)
		atomic_init(&state->lazy_retained[lane],0);
	status = SparkGlm5NextModuleConfigure(state,configuration,host_services,&pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaWaitInitialize(&state->stream_wait,(cudaStream_t)state->execution_stream);
	if ( status == SPARK_STATUS_OK && SparkGlm5NextConfigureCudaModule(&state->multiprocessor_count) != 0 )
		status = SPARK_STATUS_TARGET_MISMATCH;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackLoad(state,pack_path);
	if ( status == SPARK_STATUS_OK && state->mtp_enabled != 0u && state->pack_has_mtp == 0u )
	{
		fprintf(stderr,"G5N-DBG config: MTP flag set but the pack carries no layer-45 tensors\n");
		status = SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( status == SPARK_STATUS_OK && state->mtp_enabled != 0u && state->tp_degree != 1u )
	{
		fprintf(stderr,"G5N-DBG config: MTP speculation requires tp_degree 1 in this revision\n");
		status = SPARK_STATUS_UNSUPPORTED;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateCaches(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateMtp(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextModuleInitializeTpCollective(state,(const SparkGlm5NextResidentDecodeStageNodeContext *)host_services->node_context);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextBuildHeadShadow(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdWorkerCreate(&state->completion_worker);

	if ( status != SPARK_STATUS_OK )
	{
		if ( SparkGlm5NextReleaseCollectives(state) != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( state->lazy_pack != 0 && (SparkGlm5NextReleasePinnedExperts(state) != SPARK_STATUS_OK || SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK) )
		{
			fprintf(stderr,"GLM lazy initialization cleanup failed; retaining CUDA resources until process exit\n");
			SPARK_RETURN(status);
		}
		if ( state->lane_client != 0 )
		{
			(void)SparkWeightdClientClose(state->lane_client);
			state->lane_client = 0;
		}
		if ( SparkGlm5NextReleaseCaches(state) != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		SparkGlm5NextReleaseSlotHost(state);
		SparkStageModuleLedgerRelease(&state->ledger);
		free(state->mtp_lane_armed);
		free(state);
		SPARK_RETURN(status);
	}
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states,state->resident_sequence_capacity);
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_init(&state->lane_bound[lane],0u);
		atomic_init(&state->lane_sequence_ids[lane],0u);
		atomic_init(&state->lane_next_positions[lane],0u);
	}
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->rejected_count,0u);
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->host_callback_completion_count,0u);
	atomic_init(&state->nccl_next_ordinal,0u);
	atomic_init(&state->nccl_next_ordinal_hc,0u);
	*state_out = state;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm5NextResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkGlm5NextModuleState *state;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state = 0;
	status = SparkGlm5NextInitializeState(configuration,host_services,&state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
