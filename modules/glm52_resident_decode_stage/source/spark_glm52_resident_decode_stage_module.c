#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include "sparkpipe/spark_error_site.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_glm52_kv_geometry.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_tp_mesh_register.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_stage_module_lifecycle.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_tp_chain_graph.h"
#include "sparkpipe/spark_glm52_graph_regime.h"
#include "sparkpipe/spark_glm52_verify_config.h"
#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_speculation_lookup_draft.h"
#include "sparkpipe/spark_speculation_reference_draft.h"
#include "sparkpipe/spark_speculation_recorded_draft.h"
#include "spark_glm52_resident_decode_stage_internal.h"
#include "spark_glm52_stagepack_format.h"
#ifdef SPARK_SCORE_DUMP
#include <pthread.h>
#include "sparkpipe/spark_score_dump.h"
#endif
#define SPARK_FAMILY_CAMEL Glm52
#define SPARK_FAMILY_UPPER GLM52
#define SPARK_FAMILY_LOWER glm52

#include "sparkpipe/family/spark_family.h"

#ifndef GLM_EXPERT_WEIGHT_CODEC
#error "GLM_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef GLM_CONTRACT_SHA256
#error "GLM_CONTRACT_SHA256 must identify the exact model package contract"
#endif

#define SPARK_GLM52_MODULE_TAG "glm52_stage"
#define SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT 2048u
#define SPARK_GLM52_HEAD_TILE 1024u
#define SPARK_GLM52_NO_INDEX_ORDINAL UINT32_MAX
#define SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT 6u

typedef struct SparkGlm52PackRange
{
	uint64_t offset;
	uint64_t bytes;
} SparkGlm52PackRange;

typedef struct SparkGlm52ModuleState SparkGlm52ModuleState;
typedef struct SparkGlm52TpChain SparkGlm52TpChain;

typedef struct SparkGlm52AsyncCompletion
{
	SparkGlm52ModuleState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t lane_indices[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_bound[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	uint32_t steps_token_count;
	uint32_t steps_extra_tokens;
	uint32_t steps_tokens[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE];
	SparkModelDriverCompletion completion;
} SparkGlm52AsyncCompletion;

struct SparkGlm52ModuleState
{
	SparkStageModuleLedger ledger;
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
	uint32_t index_layer_count;
	uint32_t multiprocessor_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	void *execution_stream;
	char model_revision[SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES];
	SparkGlm52LayerWeights layers[SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t index_ordinal_by_local_layer[SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t layer_seen[SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t global_seen;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	uint8_t *head_certified_fp8_payload;
	float *head_certified_fp8_scale_f32;
	float *head_certified_fp8_norm_f32;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint8_t *index_cache;
	uint64_t index_layer_stride_bytes;
	uint32_t *page_table;
	SparkKvCacheArena kv_arena;
	SparkKvPageCache kv_page_cache;
	SparkKvPageStore kv_page_store;
	SparkKvCacheBlock *kv_blocks;
	uint32_t *kv_resident_slot_logical_block_indices;
	SparkKvPageCacheEntry *kv_entries;
	SparkKvPageCacheSequence *kv_sequences;
	uint32_t *kv_hash_bucket_heads;
	uint32_t *kv_entry_indices_by_logical_page;
	uint8_t *kv_page_staging;
	uint32_t *kv_lane_logical_pages;
	uint32_t *kv_lane_page_count;
	uint32_t *kv_lane_mutable_page;
	uint32_t *kv_lane_mutation_flags;
	SparkModelDriverCacheLane *kv_lane_cache_lanes;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	char kv_backing_default[256];
	SparkWeightdLazyPack *lazy_pack;
	uint64_t expert_pin_leases[64];
	uint32_t expert_pin_lease_count;
	uint32_t expert_pin_key_count;
	uint32_t experts_pinned;
	const uint8_t *expert_pin_base;
	SparkGlm52ExecutionSlot slots[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm52AsyncCompletion completions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm52TpChain *lazy_retained[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_uchar lane_bound[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_sequence_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_next_positions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong host_callback_completion_count;
	atomic_ullong tokens_emitted;
	SparkTpDeviceCollective tp_device_collective;
	uint32_t tp_device_collective_initialized;
	atomic_ullong tp_next_ordinal;
	uint32_t chain_mode;
#ifdef SPARK_SCORE_DUMP
	struct SparkGlm52Score *score;
#endif
	uint32_t projection_split;
	uint32_t prefill_wave_rows;
	uint32_t chain_wait_initialized;
	SparkStageModuleCudaWait chain_wait;
	atomic_uint chain_busy;
	uint64_t chain_gates[4];
	SparkTpChainGraphTable graphs[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t verify_rows_max;
	uint32_t verify_drafter;
	const char *verify_drafter_path;
	uint8_t *verify_drafter_bytes;
	SparkSpeculationDraftFunction verify_draft_function;
	void *verify_draft_context;
	SparkSpeculationLookupDraft verify_lookup;
	SparkSpeculationReferenceDraft verify_reference;
	SparkSpeculationRecordedDraft verify_recorded;
	uint64_t verify_frames;
	uint64_t verify_rounds;
	uint64_t verify_proposed;
	uint64_t verify_accepted;
	uint64_t verify_plain_rounds;
	uint64_t verify_tokens;
	uint64_t verify_position_reached[SPARK_GLM52_VERIFY_ROWS_LIMIT - 1u];
	uint64_t verify_position_accepted[SPARK_GLM52_VERIFY_ROWS_LIMIT - 1u];
	FILE *tap_file;
	uint16_t *tap_rows;
	uint32_t tap_capacity;
	uint64_t tap_records;
	int relay_socket;
	uint32_t relay_connected;
	uint32_t relay_anchor;
	uint64_t relay_drafts;
	uint64_t relay_rows;
	uint64_t relay_wait_ns;
};

typedef enum SparkGlm52ChainStage
{
	SPARK_GLM52_CHAIN_STAGE_BEGIN = 0,
	SPARK_GLM52_CHAIN_STAGE_ATTENTION,
	SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION,
	SPARK_GLM52_CHAIN_STAGE_MLP,
	SPARK_GLM52_CHAIN_STAGE_REDUCE_MLP,
	SPARK_GLM52_CHAIN_STAGE_HEAD,
	SPARK_GLM52_CHAIN_STAGE_REDUCE_HEAD,
	SPARK_GLM52_CHAIN_STAGE_FINISH,
	SPARK_GLM52_CHAIN_STAGE_REDUCE_PROJECTION
} SparkGlm52ChainStage;

typedef struct SparkGlm52TpChain
{
	SparkGlm52ModuleState *state;
	SparkGlm52ExecutionSlot *slot;
	uint32_t slot_index;
	SparkModelDriverFrame *frame;
	const SparkGlm52ResidentDecodeStageFrameContext *context;
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	SparkGlm52ResidentDecodeStageFrameContext context_copy;
	SparkGlm52ResidentDecodeStageBatchView batch_copy;
	SparkGlm52CudaWave wave;
	uint32_t first_row;
	uint32_t wave_rows;
	uint32_t next_wave_row;
	uint32_t prefill;
	uint32_t stage;
	uint32_t next_layer;
	struct timespec lazy_synced;
	uint32_t active;
	uint64_t expert_lease;
	uint32_t expert_lease_begun;
	uint32_t expert_lease_recorded;
	uint64_t retired_lease;
	uint32_t retired_begun;
	uint32_t retired_recorded;
	SparkStatus retained_status;
	uint32_t graph;
	uint32_t captured;
	uint32_t waves;
	uint64_t start_ns;
	uint64_t walk_ns;
	uint32_t steps_budget;
	uint32_t steps_produced;
	uint32_t steps_lanes;
	uint32_t verify_wave;
	uint32_t verify_draft_count;
	uint32_t verify_rounds;
	uint32_t verify_accepted;
	uint32_t verify_plain;
	uint32_t verify_draft[SPARK_GLM52_VERIFY_ROWS_LIMIT];
	uint32_t steps_anchor_tokens[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t steps_lane_slots[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t steps_positions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t steps_sequence_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t steps_tokens[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE];
} SparkGlm52TpChain;

#define SPARK_GLM_STAGE_STATE SparkGlm52ModuleState
#define SPARK_GLM_STAGE_SLOT SparkGlm52ExecutionSlot
#define SPARK_GLM_STAGE_COMPLETION SparkGlm52AsyncCompletion
#define SPARK_GLM_STAGE_BATCH SparkGlm52ResidentDecodeStageBatchView
#define SPARK_GLM_STAGE_FRAME SparkGlm52ResidentDecodeStageFrameContext
#define SPARK_GLM_STAGE_CHAIN SparkGlm52TpChain
#define SPARK_GLM_STAGE_ENTRY SparkGlm52StagePackEntry
#define SPARK_GLM_STAGE_TENSOR_SHAPE SparkGlm52StagePackTensorShape
#define SPARK_GLM_STAGE_PACK_RANGE SparkGlm52PackRange
#define SPARK_GLM_STAGE_TENSOR_FIRST SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM
#define SPARK_GLM_STAGE_TENSOR_KIND_COUNT SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT
#define SPARK_GLM_STAGE_TENSOR_EMBEDDING SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING
#define SPARK_GLM_STAGE_TENSOR_FINAL_NORM SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM
#define SPARK_GLM_STAGE_TENSOR_LM_HEAD SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD
#define SPARK_GLM_STAGE_EXPECTED_SHAPE(kind,layer_index,codec,tp_degree,shape) \
	SparkGlm52StagePackExpectedShape(kind,layer_index,codec,tp_degree,shape)
#define SPARK_GLM_STAGE_MODULE_TAG SPARK_GLM52_MODULE_TAG
#define SPARK_GLM_STAGE_KV_ACCESS_ERROR_WORD_COUNT \
	SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT
#define SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT
#define SPARK_GLM_STAGE_MAX_INPUT_ROW_COUNT \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT
#define SPARK_GLM_STAGE_FRAME_CONTEXT_ABI_VERSION \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION
#define SPARK_GLM_STAGE_BATCH_VIEW_ABI_VERSION \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION
#define SPARK_GLM_STAGE_FRAME_KNOWN_FLAGS \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS
#define SPARK_GLM_STAGE_FRAME_FLAG_PREFILL \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL
#define SPARK_GLM_STAGE_FRAME_FLAG_HIDDEN_INPUT \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT
#define SPARK_GLM_STAGE_FRAME_FLAG_HIDDEN_OUTPUT \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT
#define SPARK_GLM_STAGE_FRAME_FLAG_SIDEBAND_INPUT \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT
#define SPARK_GLM_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT
#define SPARK_GLM_STAGE_BOUNDARY_ELEMENT_COUNT \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT
#define SPARK_GLM_STAGE_BOUNDARY_ELEMENT_BYTES \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES
#define SPARK_GLM_STAGE_DSA_SIDEBAND_BYTES_PER_ROW \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW
#define SPARK_GLM_STAGE_REQUIRES_SIDEBAND_INPUT(stage_index) \
	SparkGlm52ResidentDecodeStageRequiresSidebandInput(stage_index)
#define SPARK_GLM_STAGE_REQUIRES_SIDEBAND_OUTPUT(stage_index) \
	SparkGlm52ResidentDecodeStageRequiresSidebandOutput(stage_index)
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_HOST(slot) \
	SparkGlm52AllocateSlotHost(slot)
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_METADATA(state,slot) \
	SparkGlm52AllocateSlotMetadata(state,slot)
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_HIDDEN(state,slot) \
	SparkGlm52AllocateSlotHidden(state,slot)
#define SPARK_GLM_STAGE_ALLOCATE_SLOT_MLP(state,slot) \
	SparkGlm52AllocateSlotMlp(state,slot)
#define SPARK_GLM_STAGE_COMPLETE_ASYNC SparkGlm52CompleteAsync
#define SPARK_GLM_STAGE_VALIDATE_FRAME_BUFFERS(state,frame,row_count) \
	SparkGlm52ValidateFrameBuffers(state,frame,row_count)
#define SPARK_GLM_STAGE_LAZY_RECOVER_LEASE(state,slot,out) \
	SparkGlm52LazyRecoverLease(state,slot,out)
#define SPARK_GLM_STAGE_TP_CHAIN_FAIL(chain,status) \
	SparkGlm52TpChainFail(chain,status)

static void SparkGlm52TpChainFail(SparkGlm52TpChain *chain,SparkStatus status);
static SparkStatus SparkGlm52LazyRecoverLease(SparkGlm52ModuleState *state,uint32_t slot,SparkGlm52TpChain **out);
static void CUDART_CB SparkGlm52CompleteAsync(void *context);
static SparkStatus SparkGlm52ValidateFrameBuffers(const SparkGlm52ModuleState *state,const SparkModelDriverFrame *frame,uint32_t row_count);
static SparkStatus SparkGlm52AllocateSlotHost(SparkGlm52ExecutionSlot *slot);
static SparkStatus SparkGlm52AllocateSlotMetadata(SparkGlm52ModuleState *state,SparkGlm52ExecutionSlot *slot);
static SparkStatus SparkGlm52AllocateSlotHidden(SparkGlm52ModuleState *state,SparkGlm52ExecutionSlot *slot);
static SparkStatus SparkGlm52AllocateSlotMlp(SparkGlm52ModuleState *state,SparkGlm52ExecutionSlot *slot);

#include "common/common_glm_stage_module/spark_glm_stage_module.h"

static uint32_t SparkGlm52BytesAreZero(const uint8_t *bytes,uint32_t count)
{
	uint32_t index;
	if ( bytes == 0 )
		return(1u);
	for (index=0u; index<count; index++)
		if ( bytes[index] != 0u )
			return(0u);
	return(1u);
}

static int32_t SparkGlm52ContractHash(uint8_t hash[SPARK_GLM52_STAGEPACK_SHA256_BYTES])
{
	const char *text;
	uint32_t index,high,low;
	text = GLM_CONTRACT_SHA256;
	if ( strlen(text) != 2u * SPARK_GLM52_STAGEPACK_SHA256_BYTES )
		return(-1);
	for (index=0u; index<SPARK_GLM52_STAGEPACK_SHA256_BYTES; index++)
	{
		high = text[2u * index] >= '0' && text[2u * index] <= '9' ? (uint32_t)(text[2u * index] - '0') : text[2u * index] >= 'a' && text[2u * index] <= 'f' ? (uint32_t)(text[2u * index] - 'a' + 10) : UINT32_MAX;
		low = text[2u * index + 1u] >= '0' && text[2u * index + 1u] <= '9' ? (uint32_t)(text[2u * index + 1u] - '0') : text[2u * index + 1u] >= 'a' && text[2u * index + 1u] <= 'f' ? (uint32_t)(text[2u * index + 1u] - 'a' + 10) : UINT32_MAX;
		if ( high > 15u || low > 15u )
			return(-2);
		hash[index] = (uint8_t)((high << 4u) | low);
	}
	return(0);
}

static SparkStatus SparkGlm52ModuleConfigure(
	SparkGlm52ModuleState *state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path)
{
	const SparkGlm52ResidentDecodeStageNodeContext *context;
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 || host_services->node_context == 0 || host_services->execution_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( context->stage_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT || context->stage_index >= context->stage_count || context->first_layer_index != SparkGlm52ResidentDecodeStageFirstLayer(context->stage_index) || context->layer_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE || context->expert_weight_codec != GLM_EXPERT_WEIGHT_CODEC || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions == 0u || context->max_sequence_positions > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS || context->execution_row_capacity == 0u || context->execution_row_capacity > context->resident_sequence_capacity || context->decode_split_context_threshold > context->max_sequence_positions || context->tp_degree == 0u || context->tp_rank >= context->tp_degree || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || context->model_revision == 0 || context->model_revision[0] == '\0' || strlen(context->model_revision) >= sizeof(state->model_revision) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightCodecIsKnown(context->expert_weight_codec) == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( context->tp_degree != 1u && (SPARK_GLM52_MODEL_HEAD_COUNT % context->tp_degree != 0u || SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree != 0u || SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u || SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
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
	state->execution_row_capacity = context->execution_row_capacity;
	state->decode_split_context_threshold = context->decode_split_context_threshold;
	state->owns_embedding = context->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = context->stage_index + 1u == context->stage_count ? 1u : 0u;
	state->execution_stream = host_services->execution_stream;
	(void)snprintf(state->model_revision,sizeof(state->model_revision),"%s",context->model_revision);
	*pack_path = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackValidateHeader(
	const SparkGlm52ModuleState *state,
	const SparkGlm52StagePackHeader *header,
	uint64_t file_bytes)
{
	uint8_t contract_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	uint64_t directory_bytes,directory_end;
	if ( state == 0 || header == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_GLM52_STAGEPACK_MAGIC || header->format_version != SPARK_GLM52_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_GLM52_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_GLM52_STAGEPACK_ENTRY_BYTES || header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( (header->flags & ~SPARK_GLM52_STAGEPACK_KNOWN_FLAGS) != 0u || (header->flags & SPARK_GLM52_STAGEPACK_FLAG_MTP) != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( SparkGlm52StagePackHeaderTpDegree(header) == 0u || SparkGlm52StagePackHeaderTpDegree(header) != state->tp_degree || SparkGlm52StagePackHeaderTpRank(header) != state->tp_rank )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT || header->stage_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT || header->stage_index != state->stage_index || header->first_layer_index != state->first_layer_index || header->layer_count != state->layer_count || header->total_layer_count != SPARK_GLM52_MODEL_LAYER_COUNT || header->hidden_dimension != SPARK_GLM52_MODEL_HIDDEN_DIMENSION || header->vocab_count != SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT || header->routed_expert_count != SPARK_GLM52_MODEL_MOE_EXPERT_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != state->expert_weight_codec || header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	if ( SparkGlm52ContractHash(contract_sha256) < 0 )
	{
		fprintf(stderr,"glm52 pack gate: contract hash unreadable\n");
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	}
	if ( header->model_revision[SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES - 1u] != '\0' || strcmp(header->model_revision,state->model_revision) != 0 )
	{
		fprintf(stderr,"glm52 pack gate: model_revision %.*s != expected %s\n",
			SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES,header->model_revision,
			state->model_revision);
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	}
	if ( memcmp(header->contract_sha256,contract_sha256,sizeof(contract_sha256)) != 0 )
	{
		uint32_t hash_index;
		fprintf(stderr,"glm52 pack gate: contract_sha256 pack=");
		for ( hash_index = 0u; hash_index < sizeof(contract_sha256); hash_index++ )
			fprintf(stderr,"%02x",header->contract_sha256[hash_index]);
		fprintf(stderr," expected=");
		for ( hash_index = 0u; hash_index < sizeof(contract_sha256); hash_index++ )
			fprintf(stderr,"%02x",((const unsigned char *)contract_sha256)[hash_index]);
		fprintf(stderr,"\n");
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	}
	if ( SparkGlm52BytesAreZero(header->source_config_sha256,sizeof(header->source_config_sha256)) != 0u )
	{
		fprintf(stderr,"glm52 pack gate: source_config_sha256 all zero\n");
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	}
	if ( SparkGlm52BytesAreZero(header->pack_recipe_sha256,sizeof(header->pack_recipe_sha256)) != 0u )
	{
		fprintf(stderr,"glm52 pack gate: pack_recipe_sha256 all zero\n");
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	}
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes || header->directory_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || header->tensor_count > UINT64_MAX / header->directory_entry_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_bytes = (uint64_t)header->tensor_count * header->directory_entry_bytes;
	directory_end = header->directory_offset + directory_bytes;
	if ( directory_end < header->directory_offset || directory_end > file_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackValidateEntryGeometry(
	const SparkGlm52ModuleState *state,
	const SparkGlm52StagePackHeader *header,
	const SparkGlm52StagePackEntry *entry,
	SparkGlm52StagePackTensorShape *shape)
{
	uint64_t payload_bytes,scale_bytes;
	uint32_t local_layer;
	if ( SparkGlm52StagePackExpectedShape(entry->tensor_kind,entry->layer_index,state->expert_weight_codec,state->tp_degree,shape) < 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->layer_index != SPARK_GLM52_STAGEPACK_GLOBAL_LAYER )
	{
		if ( entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		local_layer = entry->layer_index - state->first_layer_index;
		if ( (state->layer_seen[local_layer] & (UINT64_C(1) << entry->tensor_kind)) != 0u )
			SPARK_FAIL(SPARK_STATUS_DUPLICATE);
	}
	else if ( (state->global_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
		SPARK_FAIL(SPARK_STATUS_DUPLICATE);
	if ( entry->payload_type != shape->payload_type || entry->weight_codec != shape->weight_codec || entry->scale_encoding != shape->scale_encoding || entry->group_count != shape->group_count || entry->rows != shape->rows || entry->columns != shape->columns )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkGlm52StagePackExpectedPayloadBytes(shape);
	scale_bytes = SparkGlm52StagePackExpectedScaleBytes(shape);
	if ( payload_bytes == 0u || entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->payload_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset < header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes) || entry->payload_offset > header->file_bytes || entry->payload_bytes > header->file_bytes - entry->payload_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes == 0u )
	{
		if ( entry->scale_offset != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	else if ( entry->scale_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->scale_offset < header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes) || entry->scale_offset > header->file_bytes || entry->scale_bytes > header->file_bytes - entry->scale_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkGlm52PackMarkSeen(
	SparkGlm52ModuleState *state,
	const SparkGlm52StagePackEntry *entry)
{
	if ( entry->layer_index == SPARK_GLM52_STAGEPACK_GLOBAL_LAYER )
		state->global_seen |= UINT64_C(1) << entry->tensor_kind;
	else
		state->layer_seen[entry->layer_index - state->first_layer_index] |= UINT64_C(1) << entry->tensor_kind;
}

static SparkStatus SparkGlm52PackAssignLayer(
	SparkGlm52LayerWeights *weights,
	const SparkGlm52StagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	uint32_t tensor_kind = entry->tensor_kind;
	switch ( tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM: weights->attn_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A: weights->q_a_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM: weights->q_a_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_B: weights->q_b_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A: weights->kv_a_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM: weights->kv_a_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: weights->kv_b_key_transposed_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE: weights->kv_b_value_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT: weights->attn_output_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM: weights->post_attn_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q: weights->index_q_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K: weights->index_k_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD: weights->index_head_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: weights->index_norm_weight_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS: weights->index_norm_bias_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP: weights->dense_gate_up_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN: weights->dense_down_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER: weights->router_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION: weights->router_correction_f32 = (const float *)payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE: weights->expert_up_gate_payload = payload; weights->expert_up_gate_scale = scale; weights->expert_up_gate_payload_offset = entry->payload_offset; weights->expert_up_gate_scale_offset = entry->scale_offset; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN: weights->expert_down_payload = payload; weights->expert_down_scale = scale; weights->expert_down_payload_offset = entry->payload_offset; weights->expert_down_scale_offset = entry->scale_offset; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP: weights->shared_gate_up_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN: weights->shared_down_bf16 = payload; break;
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackAssign(
	SparkGlm52ModuleState *state,
	const SparkGlm52StagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	if ( entry->layer_index != SPARK_GLM52_STAGEPACK_GLOBAL_LAYER )
		return(SparkGlm52PackAssignLayer(&state->layers[entry->layer_index - state->first_layer_index],entry,payload,scale));
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING: state->embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_bf16 = payload; return(SPARK_STATUS_OK);
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
}

typedef struct SparkGlm52ManifestContext
{
	const SparkGlm52StagePackEntry *entries;
	uint32_t count;
} SparkGlm52ManifestContext;

#include "sparkpipe/family/module/spark_module_manifest_check.h"

#include "sparkpipe/family/module/spark_module_lazy_open.h"

#include "sparkpipe/family/module/spark_module_pin_experts.h"

static SparkStatus SparkGlm52PackLoadEntry(
	SparkGlm52ModuleState *state,
	FILE *file,
	const SparkGlm52StagePackEntry *entry)
{
	void *payload,*scale;
	SparkStatus status;
	payload = 0;
	scale = 0;
	if ( state->lazy_pack != 0 )
	{
		if ( entry->tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE || entry->tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN )
			return(SparkGlm52PackAssign(state,entry,0,0));
		status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->payload_offset,entry->payload_bytes,(const void **)&payload);
		if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
			status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->scale_offset,entry->scale_bytes,(const void **)&scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm52PackAssign(state,entry,payload,scale);
		return(status);
	}
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackAssign(state,entry,payload,scale);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52PackValidateInventory(const SparkGlm52ModuleState *state)
{
	uint32_t local;
	if ( state->global_seen != SparkGlmStageExpectedGlobalMask(state) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (local=0u; local<state->layer_count; local++)
		if ( state->layer_seen[local] != SparkGlmStageExpectedLayerMask(state,state->first_layer_index + local) )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackLoad(
	SparkGlm52ModuleState *state,
	const char *path)
{
	SparkGlm52StagePackHeader header;
	SparkGlm52StagePackEntry entries[SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT];
	SparkGlm52StagePackTensorShape shape;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	status = SparkStageModulePackFileSize(file,&file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackValidateHeader(state,&header,file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0]));
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
	{
		status = SparkGlm52PackValidateEntryGeometry(state,&header,&entries[index],&shape);
		if ( status == SPARK_STATUS_OK )
			SparkGlm52PackMarkSeen(state,&entries[index]);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlmStagePackValidateRanges(entries,header.tensor_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackValidateInventory(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52LazyOpen(state,path,file_bytes,entries,header.tensor_count);
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
		status = SparkGlm52PackLoadEntry(state,file,&entries[index]);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

#include "sparkpipe/family/module/spark_module_allocate_slot_host_staging.h"

static void SparkGlm52ReleaseSlotHost(SparkGlm52ModuleState *state)
{
	uint32_t index;
	if ( state == 0 )
		return;
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		if ( state->slots[index].host_staging != 0 )
			(void)cudaFreeHost(state->slots[index].host_staging);
		state->slots[index].host_staging = 0;
		if ( state->slots[index].group_row_offset_host != 0 )
			(void)cudaFreeHost(state->slots[index].group_row_offset_host);
		state->slots[index].group_row_offset_host = 0;
		if ( state->slots[index].route_ready_event != 0 )
			(void)cudaEventDestroy((cudaEvent_t)state->slots[index].route_ready_event);
		state->slots[index].route_ready_event = 0;
		if ( state->slots[index].expert_done_event != 0 )
			(void)cudaEventDestroy((cudaEvent_t)state->slots[index].expert_done_event);
		state->slots[index].expert_done_event = 0;
	}
}

static SparkStatus SparkGlm52AllocateSlotMetadata(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	SparkStatus status;
	status = SparkGlmStageAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->token_ids);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->resident_slots);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,state->resident_sequence_capacity,1u,sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_tile_prefix);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,1u,sizeof(uint32_t) * 6u,1u,&slot->kv_access_error);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52AllocateSlotHidden(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	uint64_t rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_QUERY_A_DIMENSION,(void **)&slot->q_compressed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_QUERY_B_DIMENSION,(void **)&slot->q_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_LATENT_DIMENSION,(void **)&slot->query_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_ROPE_DIMENSION,(void **)&slot->query_rope_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_QUERY_DIMENSION,(void **)&slot->index_query_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_key_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT,(void **)&slot->index_head_weight_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,(void **)&slot->kv_slot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_LATENT_DIMENSION,(void **)&slot->attention_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION,(void **)&slot->attention_value_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->attention_out_bf16);
	if ( status == SPARK_STATUS_OK && state->projection_split != 0u ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->projection_gather_bf16);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52AllocateSlotMlp(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	uint64_t rows,packed_rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	packed_rows = rows * SPARK_GLM52_MODEL_MOE_TOP_K;
	status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_MOE_ROUTED_GATE_UP_DIMENSION,(void **)&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_MOE_TOP_K * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION,(void **)&slot->intermediate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,packed_rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->expert_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->shared_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT,sizeof(float),(void **)&slot->router_logits_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,state->max_sequence_positions,sizeof(float),(void **)&slot->selection_scores_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree),SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS,sizeof(float),(void **)&slot->attention_split_partials_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint32_t),(void **)&slot->selected_positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_expert);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,packed_rows,1u,sizeof(float),(void **)&slot->route_weight);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_source_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_packed_row);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w1);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w2);
	if ( status == SPARK_STATUS_OK )
	{
		cudaError_t error = cudaHostAlloc((void **)&slot->group_row_offset_host,(SPARK_GLM52_MODEL_MOE_EXPERT_COUNT + 1u) * sizeof(uint32_t),cudaHostAllocPortable);
		if ( error != cudaSuccess )
			status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"group_row_offset_host");
	}
	if ( status == SPARK_STATUS_OK )
	{
		cudaError_t error = cudaEventCreateWithFlags((cudaEvent_t *)&slot->route_ready_event,cudaEventDisableTiming);
		if ( error != cudaSuccess )
			status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"route_ready_event");
	}
	if ( status == SPARK_STATUS_OK )
	{
		cudaError_t error = cudaEventCreateWithFlags((cudaEvent_t *)&slot->expert_done_event,cudaEventDisableTiming);
		if ( error != cudaSuccess )
			status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"expert_done_event");
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52BuildPageTable(SparkGlm52ModuleState *state)
{
	uint32_t *host_table;
	uint64_t entries,index;
	SparkStatus status;
	cudaError_t error;
	state->pages_per_sequence = SparkCeilDivU32(state->max_sequence_positions,64u);
	if ( state->pages_per_sequence == 0u || state->resident_sequence_capacity > UINT32_MAX / state->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->page_count = state->resident_sequence_capacity * state->pages_per_sequence;
	entries = state->page_count;
	host_table = (uint32_t *)malloc(entries * sizeof(uint32_t));
	if ( host_table == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<entries; index++)
		host_table[index] = (uint32_t)index;
	status = SparkGlmStageAllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&state->page_table);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemcpy(state->page_table,host_table,entries * sizeof(uint32_t),cudaMemcpyHostToDevice);
		status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"page_table");
	}
	free(host_table);
	SPARK_RETURN(status);
}

#include "sparkpipe/family/module/spark_module_page_copy.h"

static SparkStatus SparkGlm52KvInitialize(SparkGlm52ModuleState *state)
{
	SparkKvModelTable table;
	uint64_t block_bytes;
	uint64_t lane_page_entries;
	SparkStatus status;
	block_bytes = (uint64_t)SPARK_GLM_KV_BLOCK_TOKEN_COUNT *
		(uint64_t)state->layer_count * SPARK_GLM_KV_ARENA_HEAD_DIMENSION *
		SPARK_GLM_KV_BYTES_PER_SCALAR;
	lane_page_entries = (uint64_t)state->resident_sequence_capacity *
		state->pages_per_sequence;
	state->kv_blocks = (SparkKvCacheBlock *)calloc(state->page_count,sizeof(*state->kv_blocks));
	state->kv_resident_slot_logical_block_indices = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_resident_slot_logical_block_indices));
	state->kv_entries = (SparkKvPageCacheEntry *)calloc(state->page_count,sizeof(*state->kv_entries));
	state->kv_sequences = (SparkKvPageCacheSequence *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_sequences));
	state->kv_hash_bucket_heads = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_hash_bucket_heads));
	state->kv_entry_indices_by_logical_page = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_entry_indices_by_logical_page));
	state->kv_page_staging = (uint8_t *)malloc((size_t)block_bytes);
	state->kv_lane_logical_pages = (uint32_t *)calloc((size_t)lane_page_entries,sizeof(*state->kv_lane_logical_pages));
	state->kv_lane_page_count = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_page_count));
	state->kv_lane_mutable_page = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_mutable_page));
	state->kv_lane_mutation_flags = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_mutation_flags));
	state->kv_lane_cache_lanes = (SparkModelDriverCacheLane *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_cache_lanes));
	if ( state->kv_blocks == 0 || state->kv_resident_slot_logical_block_indices == 0 || state->kv_entries == 0 || state->kv_sequences == 0 || state->kv_hash_bucket_heads == 0 || state->kv_entry_indices_by_logical_page == 0 || state->kv_page_staging == 0 || state->kv_lane_logical_pages == 0 || state->kv_lane_page_count == 0 || state->kv_lane_mutable_page == 0 || state->kv_lane_mutation_flags == 0 || state->kv_lane_cache_lanes == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);

	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table.descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;
	SparkGlmKvFillCapacityRequest(&table.capacity_request);

	table.arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table.arena_configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table.arena_configuration.logical_block_count = state->page_count;
	table.arena_configuration.block_token_count = SPARK_GLM_KV_BLOCK_TOKEN_COUNT;
	table.arena_configuration.resident_block_capacity = state->page_count;
	table.arena_configuration.layer_count = state->layer_count;
	table.arena_configuration.kv_head_count = SPARK_GLM_KV_ARENA_KV_HEAD_COUNT;
	table.arena_configuration.head_dim = SPARK_GLM_KV_ARENA_HEAD_DIMENSION;
	table.arena_configuration.bytes_per_scalar = SPARK_GLM_KV_BYTES_PER_SCALAR;
	table.arena_configuration.key_device_base = state->kv_cache;
	table.arena_configuration.blocks = state->kv_blocks;
	table.arena_configuration.resident_slot_logical_block_indices = state->kv_resident_slot_logical_block_indices;

	table.page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table.page_store_config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table.page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	table.page_store_config.logical_page_capacity = state->page_count;
	table.page_store_config.transfer_capacity = 2u;
	table.page_store_config.page_bytes = block_bytes;
	if ( state->kv_backing_directory != 0 && state->kv_backing_directory[0] != '\0' )
		table.page_store_config.backing_path = state->kv_backing_directory;
	else
	{
		(void)snprintf(state->kv_backing_default,sizeof(state->kv_backing_default),
			"/tmp/sparkpipe_glm52_kv_%s",state->model_revision);
		table.page_store_config.backing_path = state->kv_backing_default;
	}
	table.page_store_config.maximum_backing_bytes =
		state->kv_backing_maximum_bytes >= block_bytes
			? state->kv_backing_maximum_bytes
			: block_bytes;
	table.page_store_config.staging_address = state->kv_page_staging;
	table.page_store_config.staging_bytes = block_bytes;
	table.page_store_config.copy_function = SparkGlm52PageCopy;
	table.page_store_config.copy_context = state;

	table.sequence_capacity = state->resident_sequence_capacity;
	table.entry_capacity = state->page_count;
	table.hash_bucket_count = state->page_count;
	table.entries = state->kv_entries;
	table.sequences = state->kv_sequences;
	table.hash_bucket_heads = state->kv_hash_bucket_heads;
	table.entry_indices_by_logical_page = state->kv_entry_indices_by_logical_page;
	table.model_id = "glm52";
	table.model_revision = state->model_revision;
	table.cache_layout_fingerprint = "compressed-key-value-bf16-block-major";

	status = SparkKvBackendInitialize(&table,&state->kv_arena,&state->kv_page_cache,&state->kv_page_store);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52AllocateCaches(SparkGlm52ModuleState *state)
{
	uint64_t main_page_bytes,index_page_bytes;
	uint64_t main_total,index_total;
	uint32_t local;
	SparkStatus status;
	state->index_layer_count = 0u;
	for (local=0u; local<state->layer_count; local++)
	{
		state->index_ordinal_by_local_layer[local] = SPARK_GLM52_NO_INDEX_ORDINAL;
		if ( SparkGlm52StagePackLayerHasFullIndexer(state->first_layer_index + local) != 0u )
			state->index_ordinal_by_local_layer[local] = state->index_layer_count++;
	}
	status = SparkGlm52BuildPageTable(state);
	main_page_bytes = 64u * (uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t);
	index_page_bytes = 64u * (uint64_t)SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION * sizeof(uint16_t);
	state->kv_layer_stride_bytes = main_page_bytes;
	state->index_layer_stride_bytes = (uint64_t)state->page_count * index_page_bytes;
	if ( status != SPARK_STATUS_OK || state->kv_layer_stride_bytes == 0u || state->page_count > UINT64_MAX / (main_page_bytes * state->layer_count) )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status);
	main_total = (uint64_t)state->page_count * main_page_bytes * state->layer_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,main_total,(void **)&state->kv_cache);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u )
	{
		if ( state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		index_total = state->index_layer_stride_bytes * state->index_layer_count;
		status = SparkStageModuleDeviceAllocate(&state->ledger,index_total,(void **)&state->index_cache);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52KvInitialize(state);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52AdmissionPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm52ModuleState *state;
	const SparkModelDriverCacheLane *lane;
	uint32_t lane_index;
	uint32_t mutation_flags;
	SparkStatus status;
	state = (SparkGlm52ModuleState *)context;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
	{
		lane = &request->cache_lanes[lane_index];
		if ( lane->resident_sequence_slot < state->resident_sequence_capacity )
			state->kv_lane_cache_lanes[lane->resident_sequence_slot] = *lane;
	}
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCacheReleaseLane(&state->kv_page_cache,lane->resident_sequence_slot,lane->sequence_id);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		return(SPARK_STATUS_OK);
	}
	if ( (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCachePrepareLane(&state->kv_page_cache,lane,state->kv_lane_logical_pages + (uint64_t)lane->resident_sequence_slot * state->pages_per_sequence,state->pages_per_sequence,&state->kv_lane_page_count[lane->resident_sequence_slot]);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
	}
	if ( (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCacheBeginLaneTransaction(&state->kv_page_cache,lane,&state->kv_lane_mutable_page[lane->resident_sequence_slot],&mutation_flags);
			state->kv_lane_mutation_flags[lane->resident_sequence_slot] = mutation_flags;
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
	}
	if ( (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCacheRollbackLaneTransaction(&state->kv_page_cache,lane,state->kv_lane_mutation_flags[lane->resident_sequence_slot]);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
	}
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	return(SPARK_STATUS_OK);
}

#include "sparkpipe/family/module/spark_module_validate_sequence_continuity.h"

typedef struct SparkGlm52ClaimedContinuityContext
{
	const SparkGlm52ModuleState *state;
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkGlm52ClaimedContinuityContext;

#define SPARK_GLM52_TP_COLLECTIVE_CREDITS_PER_SLOT 2u

static void SparkGlm52TpChainAdvance(void *chain_context,SparkStatus status);
static void CUDART_CB SparkGlm52CompleteAsync(void *context);
static SparkStatus SparkGlm52LazyRelease(SparkGlm52TpChain *chain);
static void SparkGlmStageLazyRetryRetained(void *context);
static SparkStatus SparkGlmStageEnqueueAsyncCompletion(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot,
	uint32_t slot_index);

static void SparkGlm52BuildWave(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state;
	SparkGlm52ExecutionSlot *slot;
	const SparkGlm52ResidentDecodeStageFrameContext *context;
	SparkGlm52CudaWave *wave;
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
	wave->row_count = chain->wave_rows;
	wave->maximum_context = maximum_context;
	wave->resident_sequence_capacity = state->resident_sequence_capacity;
	wave->max_sequence_positions = state->max_sequence_positions;
	wave->pages_per_sequence = state->pages_per_sequence;
	wave->owns_embedding = state->owns_embedding;
	wave->owns_final_head = state->owns_final_head;
	wave->sideband_input = SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->stage_index);
	wave->sideband_output = SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->stage_index);
	wave->boundary_row_offset = chain->first_row;
	wave->sideband_row_offset = chain->first_row;
	wave->host_token_ids = state->owns_embedding != 0u ? slot->host_token_ids + chain->first_row : 0;
	wave->host_resident_slots = slot->host_resident_slots + chain->first_row;
	wave->host_positions = slot->host_positions + chain->first_row;
	wave->hidden_input_bf16 = context->hidden_input_bf16;
	wave->hidden_output_bf16 = context->hidden_output_bf16;
	wave->sideband_input_u32 = context->sideband_input;
	wave->sideband_output_u32 = context->sideband_output;
	wave->host_output_token_ids = state->owns_final_head != 0u ? slot->host_output_token_ids + chain->first_row : 0;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->final_norm_bf16 = state->final_norm_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->expert_lease_base = state->expert_pin_base;
	wave->expert_lease_pinned = state->experts_pinned;
	wave->route_host_copy = state->lazy_pack != 0 && state->experts_pinned == 0u ? 1u : 0u;
	wave->projection_split = state->projection_split;
	wave->row_head_certified = (chain->prefill != 0u && state->prefill_wave_rows != 0u) || chain->verify_wave != 0u ? 1u : 0u;
	wave->head_certified_fp8_payload = state->head_certified_fp8_payload;
	wave->head_certified_fp8_scale_f32 = state->head_certified_fp8_scale_f32;
	wave->head_certified_fp8_norm_f32 = state->head_certified_fp8_norm_f32;
	wave->layers = state->layers;
	wave->slot = slot;
	wave->kv_cache = state->kv_cache;
	wave->kv_layer_stride_bytes = state->kv_layer_stride_bytes;
	wave->index_cache = state->index_cache;
	wave->index_layer_stride_bytes = state->index_layer_stride_bytes;
	wave->index_ordinal_by_local_layer = state->index_ordinal_by_local_layer;
	wave->page_table = state->page_table;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->decode_split_context_threshold = state->decode_split_context_threshold;
	wave->attention_split_partials_f32 = slot->attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(
		state->execution_row_capacity,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree);
}

typedef struct SparkGlm52WaveRegimeContext
{
	const uint32_t *positions;
	uint32_t split_threshold;
	uint32_t max_positions;
} SparkGlm52WaveRegimeContext;

static uint32_t SparkGlm52RowRegime(void *context,uint32_t row)
{
	const SparkGlm52WaveRegimeContext *regime = (const SparkGlm52WaveRegimeContext *)context;
	uint32_t bound,tokens;
	tokens = regime->positions[row] + 1u;
	return(SparkGlm52GraphRegime(tokens,regime->split_threshold,regime->max_positions,&bound) +
		(tokens > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT ? SPARK_GLM52_GRAPH_REGIME_COUNT : 0u));
}

static uint32_t SparkGlm52WaveRows(const SparkGlm52TpChain *chain,uint32_t first_row)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkStageModuleClaimedLaneContext lanes;
	SparkGlm52WaveRegimeContext regime;
	if ( chain->verify_wave != 0u )
		return(chain->batch->row_count - first_row);
	if ( chain->prefill == 0u || state->prefill_wave_rows == 0u )
		return(SparkGlmStageRoundMajorWaveRows(state,chain->batch,first_row));
	lanes.index_states = state->lane_states;
	lanes.index_capacity = state->resident_sequence_capacity;
	regime.positions = chain->slot->host_positions;
	regime.split_threshold = state->decode_split_context_threshold;
	regime.max_positions = state->max_sequence_positions;
	return(SparkRowLayoutRoundSpanWaveRowCount(first_row,chain->batch->row_count,chain->batch->row_resident_slots,SparkStageModuleClaimedLaneOrdinal,&lanes,SparkGlm52RowRegime,&regime,state->prefill_wave_rows));
}

#define SPARK_GLM52_MODULE_TP_DISABLED(state) ((state)->tp_collective_disabled != 0u)
#define SPARK_GLM52_MODULE_TP_ROW_CAPACITY(state) ((state)->execution_row_capacity)
#define SPARK_GLM52_MODULE_TP_MESH_REGION(state) ((state)->lazy_pack != 0 ? (void *)(uintptr_t)(state)->lazy_pack->attached.mesh_send_buffer_addr : 0)
#include "sparkpipe/family/module/spark_module_tp_open_node_context.h"

SPARK_STAGE_MODULE_TP_CHAIN_COMPLETION(SparkGlm52ModuleTpCompletion,SparkGlm52TpChain,SparkGlm52TpChainAdvance)

static void SparkGlm52ChainSubmission(SparkGlm52TpChain *chain,void *device,uint32_t host_completion,SparkTpDeviceCollectiveSubmission *submission)
{
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission->descriptor_bytes = sizeof(*submission);
	submission->slot_index = chain->slot_index;
	submission->active_sequence_count = chain->wave_rows;
	submission->logical_sequence_count = chain->batch->active_sequence_count;
	submission->flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission->ordinal = atomic_fetch_add_explicit(&chain->state->tp_next_ordinal,1u,memory_order_relaxed);
	submission->local_device = device;
	submission->full_device = device;
	submission->cuda_stream = chain->slot->stream;
	submission->completion_function = host_completion != 0u ? SparkGlm52ModuleTpCompletion : 0;
	submission->completion_context = host_completion != 0u ? chain : 0;
}

static SparkStatus SparkGlm52ModuleReduce(SparkGlm52TpChain *chain,void *device,uint32_t operation)
{
	SparkGlm52ModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	SparkGlm52ChainSubmission(chain,device,1u,&submission);
	return(SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,operation));
}

static SparkStatus SparkGlm52ModuleReduceHidden(SparkGlm52TpChain *chain,void *device_bf16)
{
	return(SparkGlm52ModuleReduce(chain,device_bf16,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16));
}

static SparkStatus SparkGlm52ModuleReduceHeadMax(SparkGlm52TpChain *chain)
{
	return(SparkGlm52ModuleReduce(chain,chain->slot->head_maxloc_u64,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64));
}

static void SparkGlm52TpChainFail(SparkGlm52TpChain *chain,SparkStatus status)
{
	SparkGlm52ModuleState *state;
	SparkGlm52AsyncCompletion *async;
	state = chain->state;
	fprintf(stderr,"GLM52-CHAIN-FAIL stage=%u layer=%u rows=%u status=%s cuda=%s\n",(unsigned)chain->stage,(unsigned)(chain->wave.first_layer_index + chain->next_layer),(unsigned)chain->wave_rows,SparkStatusToString(status),cudaGetErrorString(cudaGetLastError()));
	(void)cudaStreamSynchronize((cudaStream_t)chain->slot->stream);
	(void)SparkGlm52LazyRelease(chain);
	async = &state->completions[chain->slot_index];
	async->completion.status = status;
	chain->active = 0u;
	SparkGlm52CompleteAsync(async);
	free(chain);
}

static SparkStatus SparkGlm52LazyRetireFinished(SparkGlm52TpChain *chain)
{
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	if ( chain->retired_lease == 0u )
		return(SPARK_STATUS_OK);
	if ( chain->retired_begun != 0u && chain->retired_recorded == 0u )
	{
		status = SparkWeightdMapRecordCompletion(map,chain->retired_lease,(cudaStream_t)chain->slot->stream);
		if ( status != SPARK_STATUS_OK )
			return(status);
		chain->retired_recorded = 1u;
	}
	status = SparkWeightdMapRelease(map,chain->retired_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
	{
		chain->retired_lease = 0u;
		chain->retired_begun = 0u;
		chain->retired_recorded = 0u;
	}
	return(status);
}

static SparkStatus SparkGlm52LazyRetireIfDone(SparkGlm52TpChain *chain)
{
	if ( chain->retired_lease == 0u || chain->retired_recorded != 0u )
		return(SparkGlm52LazyRetireFinished(chain));
	if ( cudaEventQuery((cudaEvent_t)chain->slot->expert_done_event) != cudaSuccess )
		return(SPARK_STATUS_OK);
	return(SparkGlm52LazyRetireFinished(chain));
}

static void SparkGlm52LazyRetireShift(SparkGlm52TpChain *chain)
{
	chain->retired_lease = chain->expert_lease;
	chain->retired_begun = chain->expert_lease_begun;
	chain->retired_recorded = chain->expert_lease_recorded;
	chain->expert_lease = 0u;
	chain->expert_lease_begun = 0u;
	chain->expert_lease_recorded = 0u;
}

static SparkStatus SparkGlm52LazyRelease(SparkGlm52TpChain *chain)
{
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	SparkStatus retire_status;
	if ( chain->retired_lease != 0u && chain->retired_begun != 0u && chain->retired_recorded == 0u )
	{
		if ( cudaEventSynchronize((cudaEvent_t)chain->slot->expert_done_event) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
	}
	retire_status = SparkGlm52LazyRetireFinished(chain);
	if ( retire_status != SPARK_STATUS_OK )
		return(retire_status);
	if ( chain->expert_lease == 0u )
		return(SPARK_STATUS_OK);
	if ( chain->expert_lease_begun != 0u )
	{
		if ( chain->expert_lease_recorded == 0u )
		{
			status = SparkWeightdMapRecordCompletion(map,chain->expert_lease,(cudaStream_t)chain->slot->stream);
			if ( status != SPARK_STATUS_OK )
				return(status);
			chain->expert_lease_recorded = 1u;
			chain->wave.expert_lease_base = 0;
		}
		if ( cudaStreamSynchronize((cudaStream_t)chain->slot->stream) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkWeightdMapRelease(map,chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
	{
		chain->expert_lease = 0u;
		chain->expert_lease_begun = 0u;
		chain->expert_lease_recorded = 0u;
		chain->wave.expert_lease_base = 0;
	}
	return(status);
}

static SparkStatus SparkGlm52LazyRecoverLease(SparkGlm52ModuleState *state,uint32_t slot,SparkGlm52TpChain **out)
{
	SparkGlm52TpChain *chain;
	SparkStatus status = SPARK_STATUS_OK;
	*out = 0;
	chain = __atomic_exchange_n(&state->lazy_retained[slot],0,__ATOMIC_ACQ_REL);
	if ( chain == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( chain->expert_lease != 0u || chain->retired_lease != 0u )
		status = SparkGlm52LazyRelease(chain);
	if ( status != SPARK_STATUS_OK )
		__atomic_store_n(&state->lazy_retained[slot],chain,__ATOMIC_RELEASE);
	else
		*out = chain;
	return(status);
}

static SparkStatus SparkGlm52LazyExperts(SparkGlm52TpChain *chain)
{
	SparkWeightdExpertKey keys[SPARK_GLM52_MODEL_MOE_EXPERT_COUNT];
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	void *address = 0;
	uint32_t count = 0u;
	int32_t launch;
	if ( chain->slot->route_recorded == 0u || cudaEventSynchronize((cudaEvent_t)chain->slot->route_ready_event) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	(void)clock_gettime(CLOCK_MONOTONIC,&chain->lazy_synced);
	status = SparkGlm52LazyRetireIfDone(chain);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkWeightdRouteKeys(chain->wave.first_layer_index + chain->next_layer,chain->slot->group_row_offset_host,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT,chain->wave.row_count * SPARK_GLM52_MODEL_MOE_TOP_K,keys,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT,&count);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapAcquire(map,keys,count,&chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapBeginUse(map,chain->expert_lease,&address);
	if ( status != SPARK_STATUS_OK )
		return(status);
	chain->expert_lease_begun = 1u;
	chain->wave.expert_lease_base = (const uint8_t *)address;
	chain->wave.expert_lease_local_layer = chain->next_layer;
	launch = SparkGlm52LaunchCudaLayerMlpExperts(&chain->wave,chain->next_layer);
	if ( launch != 0 )
	{
		fprintf(stderr,"GLM52-EXPERTS-LAUNCH-FAIL layer=%u rows=%u keys=%u rc=%d\n",(unsigned)(chain->wave.first_layer_index + chain->next_layer),(unsigned)chain->wave.row_count,count,(int)launch);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( cudaEventRecord((cudaEvent_t)chain->slot->expert_done_event,(cudaStream_t)chain->slot->stream) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdMapRecordCompletion(map,chain->expert_lease,(cudaStream_t)chain->slot->stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	chain->expert_lease_recorded = 1u;
	return(SPARK_STATUS_OK);
}

static void SparkGlm52TpChainReduceMlp(SparkGlm52TpChain *chain)
{
	SparkStatus status;
	chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_MLP;
	status = SparkGlm52ModuleReduceHidden(chain,chain->slot->hidden_bf16);
	if ( status != SPARK_STATUS_OK )
		SparkGlm52TpChainFail(chain,status);
}

static void SparkGlm52LazyWork(void *context)
{
	SparkGlm52TpChain *chain = (SparkGlm52TpChain *)context;
	SparkStatus status,cleanup;
	struct timespec lazy_begin,lazy_end;
	(void)clock_gettime(CLOCK_MONOTONIC,&lazy_begin);
	status = SparkGlm52LazyExperts(chain);
	(void)clock_gettime(CLOCK_MONOTONIC,&lazy_end);
	if ( SparkGlm52T1Enabled() != 0 )
		fprintf(stderr,"G52-LAZY rank=%u layer=%u status=%d us=%lld sync_us=%lld\n",(unsigned)chain->wave.tp_rank,(unsigned)chain->next_layer,(int)status,(long long)((lazy_end.tv_sec - lazy_begin.tv_sec) * 1000000ll + (lazy_end.tv_nsec - lazy_begin.tv_nsec) / 1000ll),(long long)((chain->lazy_synced.tv_sec - lazy_begin.tv_sec) * 1000000ll + (chain->lazy_synced.tv_nsec - lazy_begin.tv_nsec) / 1000ll));
	if ( status == SPARK_STATUS_OK )
	{
		SparkGlm52LazyRetireShift(chain);
		SparkGlm52TpChainReduceMlp(chain);
		return;
	}
	cleanup = SparkGlm52LazyRelease(chain);
	if ( cleanup == SPARK_STATUS_IO_ERROR || cleanup == SPARK_STATUS_BUSY )
		cleanup = SparkGlm52LazyRelease(chain);
	if ( cleanup != SPARK_STATUS_OK )
	{
		chain->retained_status = status;
		fprintf(stderr,"GLM expert cleanup failed: slot=%u lease=%llu status=%d; retaining slot and lease for teardown retry\n",chain->slot_index,(unsigned long long)chain->expert_lease,(int32_t)cleanup);
		__atomic_store_n(&chain->state->lazy_retained[chain->slot_index],chain,__ATOMIC_RELEASE);
		return;
	}
	SparkGlm52TpChainFail(chain,status);
}

static float SparkGlm52T1Bf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	(void)memcpy(&result,&bits,sizeof(result));
	return(result);
}

static uint16_t SparkGlm52T1FloatToBf16(float value)
{
	uint32_t bits;
	(void)memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static uint32_t SparkGlm52T1Reserve(void **buffer,uint32_t *reserved,uint64_t bytes)
{
	if ( *reserved != 0u )
		return(*reserved);
	*buffer = malloc(bytes);
	*reserved = *buffer != 0 ? 1u : 0u;
	return(*reserved);
}

static void SparkGlm52T1Wave(const SparkGlm52CudaWave *wave)
{
	uint32_t i;
	if ( SparkGlm52T1Enabled() == 0 || wave == 0 || wave->tp_rank != 0u ||
	    wave->host_positions == 0 )
		return;
	fprintf(stderr,"G52-T1 wave rows=%u first_layer=%u",wave->row_count,
	    wave->first_layer_index);
	for ( i = 0u; i < wave->row_count; i++ )
		fprintf(stderr," pos%u",wave->host_positions[i]);
	fputc('\n',stderr);
}

static int32_t SparkGlm52T1CopyOut(void *destination,const void *source,uint64_t bytes)
{
	return(cudaMemcpy(destination,source,bytes,cudaMemcpyDeviceToHost) ==
	    cudaSuccess ? 0 : 1);
}

static void SparkGlm52T1Streams(SparkGlm52TpChain *chain,uint32_t layer)
{
	static uint16_t *residual_host = 0;
	static uint16_t *hidden_host = 0;
	static uint32_t rows_reserved = 0;
	uint32_t row,i;
	uint64_t row_bytes;
	if ( SparkGlm52T1Enabled() == 0 || chain->wave.tp_rank != 0u )
		return;
	if ( cudaStreamSynchronize((cudaStream_t)chain->slot->stream) != cudaSuccess )
		return;
	if ( rows_reserved < chain->wave_rows )
	{
		free(residual_host);
		free(hidden_host);
		residual_host = 0;
		hidden_host = 0;
		rows_reserved = 0u;
		row_bytes = (uint64_t)chain->wave_rows *
		    SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
		if ( SparkGlm52T1Reserve((void **)&residual_host,&rows_reserved,row_bytes) == 0u )
			return;
		rows_reserved = 0u;
		if ( SparkGlm52T1Reserve((void **)&hidden_host,&rows_reserved,row_bytes) == 0u )
			return;
		rows_reserved = chain->wave_rows;
	}
	row_bytes = (uint64_t)chain->wave_rows *
	    SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	if ( SparkGlm52T1CopyOut(residual_host,chain->slot->residual_bf16,row_bytes) != 0 ||
	    SparkGlm52T1CopyOut(hidden_host,chain->slot->hidden_bf16,row_bytes) != 0 )
		return;
	for ( row = 0u; row < chain->wave_rows; row++ )
	{
		uint16_t *residual = residual_host +
		    (uint64_t)row * SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
		uint16_t *hidden = hidden_host +
		    (uint64_t)row * SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
		fprintf(stderr,"G52-T1 stream L%u pos%u",layer,
		    chain->wave.host_positions[row]);
		for ( i = 0u; i < SPARK_GLM52_MODEL_HIDDEN_DIMENSION; i++ )
			fprintf(stderr," %04x",SparkGlm52T1FloatToBf16(
			    SparkGlm52T1Bf16ToFloat(residual[i]) +
			    SparkGlm52T1Bf16ToFloat(hidden[i])));
		fputc('\n',stderr);
	}
}

static void SparkGlm52T1Route(SparkGlm52TpChain *chain,uint32_t layer)
{
	static uint32_t *ids_host = 0;
	static float *weights_host = 0;
	static uint32_t rows_reserved = 0;
	uint32_t row,k;
	uint64_t bytes;
	if ( SparkGlm52T1Enabled() == 0 || chain->wave.tp_rank != 0u ||
	    layer < SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
		return;
	if ( rows_reserved < chain->wave_rows )
	{
		free(ids_host);
		free(weights_host);
		ids_host = 0;
		weights_host = 0;
		rows_reserved = 0u;
		bytes = (uint64_t)chain->wave_rows *
		    SPARK_GLM52_MODEL_MOE_TOP_K * sizeof(uint32_t);
		if ( SparkGlm52T1Reserve((void **)&ids_host,&rows_reserved,bytes) == 0u )
			return;
		rows_reserved = 0u;
		if ( SparkGlm52T1Reserve((void **)&weights_host,&rows_reserved,bytes) == 0u )
			return;
		rows_reserved = chain->wave_rows;
	}
	bytes = (uint64_t)chain->wave_rows * SPARK_GLM52_MODEL_MOE_TOP_K * sizeof(uint32_t);
	if ( SparkGlm52T1CopyOut(ids_host,chain->slot->route_expert,bytes) != 0 )
		return;
	if ( SparkGlm52T1CopyOut(weights_host,chain->slot->route_weight,
	        (uint64_t)chain->wave_rows * SPARK_GLM52_MODEL_MOE_TOP_K * sizeof(float)) != 0 )
		return;
	for ( row = 0u; row < chain->wave_rows; row++ )
	{
		fprintf(stderr,"G52-T1 route L%u pos%u ids",layer,
		    chain->wave.host_positions[row]);
		for ( k = 0u; k < SPARK_GLM52_MODEL_MOE_TOP_K; k++ )
			fprintf(stderr," %u",ids_host[
			    row * SPARK_GLM52_MODEL_MOE_TOP_K + k]);
		fprintf(stderr," weights");
		for ( k = 0u; k < SPARK_GLM52_MODEL_MOE_TOP_K; k++ )
			fprintf(stderr," %08x",((const uint32_t *)weights_host)[
			    row * SPARK_GLM52_MODEL_MOE_TOP_K + k]);
		fputc('\n',stderr);
	}
}

static void SparkGlm52T1Head(SparkGlm52TpChain *chain)
{
	static uint32_t *tokens_host = 0;
	static float *scores_host = 0;
	static uint32_t rows_reserved = 0;
	uint32_t i;
	if ( SparkGlm52T1Enabled() == 0 || chain->wave.tp_rank != 0u )
		return;
	if ( cudaStreamSynchronize((cudaStream_t)chain->slot->stream) != cudaSuccess )
		return;
	if ( rows_reserved < chain->wave_rows )
	{
		free(tokens_host);
		free(scores_host);
		tokens_host = 0;
		scores_host = 0;
		rows_reserved = 0u;
		if ( SparkGlm52T1Reserve((void **)&tokens_host,&rows_reserved,
		        (uint64_t)chain->wave_rows * sizeof(uint32_t)) == 0u )
			return;
		rows_reserved = 0u;
		if ( SparkGlm52T1Reserve((void **)&scores_host,&rows_reserved,
		        (uint64_t)chain->wave_rows * sizeof(float)) == 0u )
			return;
		rows_reserved = chain->wave_rows;
	}
	if ( SparkGlm52T1CopyOut(tokens_host,chain->slot->output_token,
	        (uint64_t)chain->wave_rows * sizeof(uint32_t)) != 0 ||
	    SparkGlm52T1CopyOut(scores_host,chain->slot->output_score,
	        (uint64_t)chain->wave_rows * sizeof(float)) != 0 )
		return;
	for ( i = 0u; i < chain->wave_rows; i++ )
		fprintf(stderr,"G52-T1 head pos%u token %u score_bits %08x\n",
		    chain->wave.host_positions[i],tokens_host[i],
		    ((const uint32_t *)scores_host)[i]);
}

#ifdef SPARK_SCORE_DUMP
typedef struct SparkGlm52Score
{
	SparkScoreDumpWriter writer;
	float *logits;
	SparkScoreDumpStats *stats;
	uint32_t *probe_offsets;
	uint32_t *probe_local;
	float *probe_logits;
	SparkScoreDumpStats *host_stats;
	uint32_t *host_offsets;
	uint32_t *host_local;
	uint32_t *host_ids;
	float *host_probe_logits;
	float *host_tier2;
	uint64_t *host_keys;
	uint32_t *host_flags;
	pthread_mutex_t lock;
	uint64_t wave_ordinal;
	uint32_t rows_capacity;
	uint32_t width;
	uint32_t id_capacity;
} SparkGlm52Score;

static void SparkGlm52ScoreRelease(SparkGlm52Score *score)
{
	if ( score == 0 )
		return;
	(void)cudaFree(score->logits);
	(void)cudaFree(score->stats);
	(void)cudaFree(score->probe_offsets);
	(void)cudaFree(score->probe_local);
	(void)cudaFree(score->probe_logits);
	free(score->host_stats);
	free(score->host_offsets);
	free(score->host_local);
	free(score->host_ids);
	free(score->host_probe_logits);
	free(score->host_tier2);
	free(score->host_keys);
	free(score->host_flags);
	(void)pthread_mutex_destroy(&score->lock);
	free(score);
}

static SparkStatus SparkGlm52ScoreOpen(SparkGlm52ModuleState *state,const SparkGlm52ResidentDecodeStageNodeContext *context)
{
	SparkScoreDumpConfig config;
	SparkGlm52Score *score;
	SparkStatus status;
	uint64_t rows,width,ids;
	if ( context->score_dump_directory == 0 )
	{
		if ( context->score_probe_path != 0 || context->score_tier2_rows_path != 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		return(SPARK_STATUS_OK);
	}
	if ( state->owns_final_head == 0u || state->owns_embedding == 0u || state->lm_head_bf16 == 0 )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( state->chain_mode == SPARK_TP_CHAIN_MODE_GRAPH )
	{
		fprintf(stderr,"SCORE-DUMP refused rank=%u: a graph replay cannot score its waves; run the dump with SPARK_GLM52_CHAIN_MODE=linear or eager\n",state->tp_rank);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	score = (SparkGlm52Score *)calloc(1u,sizeof(*score));
	if ( score == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( pthread_mutex_init(&score->lock,0) != 0 )
	{
		free(score);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	memset(&config,0,sizeof(config));
	config.directory = context->score_dump_directory;
	config.probe_path = context->score_probe_path;
	config.tier2_path = context->score_tier2_rows_path;
	config.tp_rank = state->tp_rank;
	config.tp_degree = state->tp_degree;
	config.vocabulary = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	config.hidden_dimension = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	config.slot_count = state->resident_sequence_capacity;
	config.position_count = state->max_sequence_positions;
	status = SparkScoreDumpOpen(&config,&score->writer);
	if ( status != SPARK_STATUS_OK )
	{
		(void)pthread_mutex_destroy(&score->lock);
		free(score);
		fprintf(stderr,"SCORE-DUMP open failed status=%d directory=%s\n",(int)status,context->score_dump_directory);
		SPARK_RETURN(status);
	}
	rows = state->execution_row_capacity;
	width = score->writer.header.shard_end - score->writer.header.shard_begin;
	ids = rows * (score->writer.probes.id_width != 0u ? score->writer.probes.id_width : 1u);
	score->rows_capacity = (uint32_t)rows;
	score->width = (uint32_t)width;
	score->id_capacity = (uint32_t)ids;
	if ( cudaMalloc((void **)&score->logits,rows * width * sizeof(float)) != cudaSuccess ||
	     cudaMalloc((void **)&score->stats,rows * sizeof(SparkScoreDumpStats)) != cudaSuccess ||
	     cudaMalloc((void **)&score->probe_offsets,(rows + 1u) * sizeof(uint32_t)) != cudaSuccess ||
	     cudaMalloc((void **)&score->probe_local,ids * sizeof(uint32_t)) != cudaSuccess ||
	     cudaMalloc((void **)&score->probe_logits,ids * sizeof(float)) != cudaSuccess )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	score->host_stats = (SparkScoreDumpStats *)calloc(rows,sizeof(SparkScoreDumpStats));
	score->host_offsets = (uint32_t *)calloc(rows + 1u,sizeof(uint32_t));
	score->host_local = (uint32_t *)calloc(ids,sizeof(uint32_t));
	score->host_ids = (uint32_t *)calloc(ids,sizeof(uint32_t));
	score->host_probe_logits = (float *)calloc(ids,sizeof(float));
	score->host_keys = (uint64_t *)calloc(rows,sizeof(uint64_t));
	score->host_flags = (uint32_t *)calloc(rows,sizeof(uint32_t));
	if ( score->writer.header.tier2 != 0u )
		score->host_tier2 = (float *)calloc(rows * width,sizeof(float));
	if ( score->host_stats == 0 || score->host_offsets == 0 || score->host_local == 0 || score->host_ids == 0 || score->host_probe_logits == 0 || score->host_keys == 0 || score->host_flags == 0 || (score->writer.header.tier2 != 0u && score->host_tier2 == 0) )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status != SPARK_STATUS_OK )
	{
		SparkScoreDumpFail(&score->writer);
		(void)SparkScoreDumpClose(&score->writer);
		SparkGlm52ScoreRelease(score);
		SPARK_FAIL(status);
	}
	fprintf(stderr,"SCORE-DUMP open rank=%u shard=[%u,%u) rows=%s probes=%llu tier2=%llu chain=%s\n",
	    state->tp_rank,score->writer.header.shard_begin,score->writer.header.shard_end,score->writer.rows_path,
	    (unsigned long long)score->writer.probes.entry_count,(unsigned long long)score->writer.tier2_rows.entry_count,SparkTpChainModeName(state->chain_mode));
	state->score = score;
	return(SPARK_STATUS_OK);
}

static void SparkGlm52ScoreClose(SparkGlm52ModuleState *state)
{
	SparkGlm52Score *score = state->score;
	SparkStatus status;
	if ( score == 0 )
		return;
	status = SparkScoreDumpClose(&score->writer);
	fprintf(stderr,"SCORE-DUMP close rank=%u status=%d rows=%llu waves=%llu skipped=%llu keyless=%llu\n",
	    state->tp_rank,(int)status,(unsigned long long)score->writer.end.row_count,(unsigned long long)score->writer.end.wave_count,
	    (unsigned long long)score->writer.end.skipped_wave_count,(unsigned long long)score->writer.end.keyless_row_count);
	SparkGlm52ScoreRelease(score);
	state->score = 0;
}

static SparkStatus SparkGlm52ScorePlan(SparkGlm52Score *score,const SparkGlm52ExecutionSlot *slot,uint32_t first,uint32_t rows,uint32_t *probe_count)
{
	const uint32_t *ids;
	uint32_t row,index,count,id_count,position;
	uint64_t key;
	count = 0u;
	for (row=0u; row<rows; row++)
	{
		position = slot->host_positions[first + row];
		key = 0u;
		score->host_flags[row] = SparkScoreDumpKeysAdvance(&score->writer.keys,slot->host_resident_slots[first + row],position,slot->host_token_ids[first + row],&key) != 0u ? SPARK_SCORE_DUMP_ROW_KEY_VALID : 0u;
		score->host_keys[row] = key;
		score->host_offsets[row] = count;
		id_count = 0u;
		ids = score->host_flags[row] != 0u ? SparkScoreDumpTableFind(&score->writer.probes,key,position,&id_count) : 0;
		if ( ids != 0 )
			score->host_flags[row] |= SPARK_SCORE_DUMP_ROW_PROBED;
		if ( score->host_flags[row] != 0u && score->host_tier2 != 0 && SparkScoreDumpTableFind(&score->writer.tier2_rows,key,position,0) != 0 )
			score->host_flags[row] |= SPARK_SCORE_DUMP_ROW_TIER2;
		for (index=0u; ids != 0 && index<id_count; index++)
		{
			if ( ids[index] < score->writer.header.shard_begin || ids[index] >= score->writer.header.shard_end )
				continue;
			if ( count >= score->id_capacity )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			score->host_ids[count] = ids[index];
			score->host_local[count] = ids[index] - score->writer.header.shard_begin;
			count++;
		}
	}
	score->host_offsets[rows] = count;
	*probe_count = count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ScoreWrite(SparkGlm52Score *score,const SparkGlm52ExecutionSlot *slot,uint32_t first,uint32_t rows)
{
	SparkScoreDumpRowRecord record;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t row;
	for (row=0u; status == SPARK_STATUS_OK && row<rows; row++)
	{
		memset(&record,0,sizeof(record));
		record.record_kind = SPARK_SCORE_DUMP_RECORD_ROW;
		record.flags = score->host_flags[row] | score->host_stats[row].flags;
		record.key = score->host_keys[row];
		record.wave_ordinal = score->wave_ordinal;
		record.position = slot->host_positions[first + row];
		record.row_in_wave = row;
		record.input_token = slot->host_token_ids[first + row];
		record.served_token = slot->host_output_token_ids[first + row];
		record.probe_count = score->host_offsets[row + 1u] - score->host_offsets[row];
		record.local_max = score->host_stats[row].local_max;
		record.local_sum_exp = score->host_stats[row].local_sum_exp;
		memcpy(record.top_ids,score->host_stats[row].top_ids,sizeof(record.top_ids));
		memcpy(record.top_logits,score->host_stats[row].top_logits,sizeof(record.top_logits));
		if ( (record.flags & SPARK_SCORE_DUMP_ROW_TIER2) != 0u )
			status = SparkScoreDumpWriteTier2(&score->writer,record.key,record.position,score->host_tier2 + (uint64_t)row * score->width);
		if ( status == SPARK_STATUS_OK )
			status = SparkScoreDumpWriteRow(&score->writer,&record,score->host_ids + score->host_offsets[row],score->host_probe_logits + score->host_offsets[row]);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52ScoreWaveLocked(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52Score *score = state->score;
	SparkGlm52ExecutionSlot *slot = chain->slot;
	cudaStream_t stream = (cudaStream_t)slot->stream;
	cudaError_t error;
	SparkStatus status;
	uint32_t rows,first,count,row;
	rows = chain->wave_rows;
	first = chain->first_row;
	if ( rows == 0u || rows > score->rows_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm52ScorePlan(score,slot,first,rows,&count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	error = cudaMemcpyAsync(score->probe_offsets,score->host_offsets,(uint64_t)(rows + 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && count != 0u )
		error = cudaMemcpyAsync(score->probe_local,score->host_local,(uint64_t)count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkGlm52LaunchHeadScore(stream,slot->normed_bf16,state->lm_head_bf16,score->logits,rows,score->width,score->writer.header.shard_begin,score->probe_offsets,score->probe_local,score->probe_logits,score->stats);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(score->host_stats,score->stats,(uint64_t)rows * sizeof(SparkScoreDumpStats),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess && count != 0u )
		error = cudaMemcpyAsync(score->host_probe_logits,score->probe_logits,(uint64_t)count * sizeof(float),cudaMemcpyDeviceToHost,stream);
	for (row=0u; error == cudaSuccess && row<rows; row++)
		if ( (score->host_flags[row] & SPARK_SCORE_DUMP_ROW_TIER2) != 0u )
			error = cudaMemcpyAsync(score->host_tier2 + (uint64_t)row * score->width,score->logits + (uint64_t)row * score->width,(uint64_t)score->width * sizeof(float),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"score_dump_launch");
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ScoreWrite(score,slot,first,rows);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	SparkScoreDumpNoteWave(&score->writer,0u);
	score->wave_ordinal++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ScoreWave(SparkGlm52TpChain *chain)
{
	SparkGlm52Score *score = chain->state->score;
	SparkStatus status;
	if ( score == 0 )
		return(SPARK_STATUS_OK);
	if ( pthread_mutex_lock(&score->lock) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm52ScoreWaveLocked(chain);
	if ( status != SPARK_STATUS_OK )
	{
		SparkScoreDumpFail(&score->writer);
		fprintf(stderr,"SCORE-DUMP wave failed rank=%u status=%d rows=%u; the dump is incomplete and gets no end record\n",chain->state->tp_rank,(int)status,chain->wave_rows);
	}
	(void)pthread_mutex_unlock(&score->lock);
	return(status);
}
#endif

static void SparkGlm52TpChainAdvance(void *chain_context,SparkStatus status)
{
	SparkGlm52TpChain *chain;
	SparkGlm52ModuleState *state;
	SparkStatus launch_status;
	cudaError_t error;
	chain = (SparkGlm52TpChain *)chain_context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	if ( SparkGlm52T1Enabled() != 0 )
	{
		struct timespec now;
		(void)clock_gettime(CLOCK_MONOTONIC,&now);
		fprintf(stderr,"G52-CHAIN rank=%u stage=%u layer=%u rows=%u status=%d t_us=%lld\n",(unsigned)chain->wave.tp_rank,(unsigned)chain->stage,(unsigned)chain->next_layer,(unsigned)chain->wave_rows,(int)status,(long long)now.tv_sec * 1000000ll + (long long)now.tv_nsec / 1000ll);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm52TpChainFail(chain,status);
		return;
	}
	switch ( chain->stage )
	{
	case SPARK_GLM52_CHAIN_STAGE_BEGIN:
		SparkGlm52BuildWave(chain);
		SparkGlm52T1Wave(&chain->wave);
		if ( SparkGlm52LaunchCudaWaveBegin(&chain->wave) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_ATTENTION;
		chain->next_layer = 0u;
		launch_status = SparkGlm52ModuleReduceHidden(chain,chain->slot->hidden_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_ATTENTION:
		if ( state->projection_split != 0u )
		{
			if ( SparkGlm52LaunchCudaLayerAttentionProject(&chain->wave,chain->next_layer) != 0 )
			{
				SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_PROJECTION;
			launch_status = SparkGlm52ModuleReduceHidden(chain,chain->slot->projection_gather_bf16);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		if ( SparkGlm52LaunchCudaLayerAttention(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm52ModuleReduceHidden(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_PROJECTION:
		if ( SparkGlm52LaunchCudaLayerAttentionCore(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm52ModuleReduceHidden(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION:
		chain->stage = SPARK_GLM52_CHAIN_STAGE_MLP;
		SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		return;
	case SPARK_GLM52_CHAIN_STAGE_MLP:
		if ( state->lazy_pack != 0 && state->experts_pinned == 0u && (chain->wave.first_layer_index + chain->next_layer) >= SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
		{
			if ( SparkGlm52LaunchCudaLayerMlpRoute(&chain->wave,chain->next_layer) != 0 )
			{
				SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			launch_status = SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm52LazyWork,chain);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		if ( SparkGlm52LaunchCudaLayerMlp(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		SparkGlm52TpChainReduceMlp(chain);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_MLP:
		SparkGlm52T1Streams(chain,chain->wave.first_layer_index + chain->next_layer);
		SparkGlm52T1Route(chain,chain->wave.first_layer_index + chain->next_layer);
		chain->next_layer++;
		if ( chain->next_layer < chain->wave.layer_count )
		{
			chain->stage = SPARK_GLM52_CHAIN_STAGE_ATTENTION;
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		}
		else
		{
			chain->stage = SPARK_GLM52_CHAIN_STAGE_HEAD;
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		}
		return;
	case SPARK_GLM52_CHAIN_STAGE_HEAD:
		if ( SparkGlm52LaunchCudaWaveHead(&chain->wave) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_HEAD;
		launch_status = SparkGlm52ModuleReduceHeadMax(chain);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_HEAD:
		error = SparkGlm52LaunchHeadMaxlocUnpack((cudaStream_t)chain->slot->stream,chain->slot->head_maxloc_u64,chain->slot->output_token,chain->wave_rows);
		if ( error == cudaSuccess && state->owns_final_head != 0u )
			error = cudaMemcpyAsync(chain->slot->host_output_token_ids + chain->first_row,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream);
		launch_status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"tp_head_unpack");
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		SparkGlm52T1Head(chain);
#ifdef SPARK_SCORE_DUMP
		launch_status = SparkGlm52ScoreWave(chain);
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
#endif
		if ( chain->next_wave_row < chain->batch->row_count )
		{
			uint32_t next_wave;
			next_wave = SparkGlm52WaveRows(chain,chain->next_wave_row);
			if ( next_wave == 0u )
			{
				SparkGlm52TpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
				return;
			}
			chain->first_row = chain->next_wave_row;
			chain->wave_rows = next_wave;
			chain->next_wave_row += next_wave;
			chain->stage = SPARK_GLM52_CHAIN_STAGE_BEGIN;
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
			return;
		}
		/* STREAM-ORDERED STAGE COMPLETION CONTRACT */
		launch_status = SparkGlm52LazyRelease(chain);
		if ( launch_status == SPARK_STATUS_OK )
			launch_status = SparkGlmStageEnqueueAsyncCompletion(state,chain->slot,chain->slot_index);
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_FINISH;
		chain->active = 0u;
		free(chain);
		return;
	default:
		SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
}

static void CUDART_CB SparkGlm52CompleteAsync(void *context)
{
	SparkGlm52AsyncCompletion *async;
	SparkGlm52ModuleState *state;
	SparkGlm52ExecutionSlot *slot;
	uint32_t lane,resident;
	async = (SparkGlm52AsyncCompletion *)context;
	state = async != 0 ? async->state : 0;
	if ( state == 0 || async->slot_index >= state->pipeline_slot_count )
		return;
	slot = &state->slots[async->slot_index];
	if ( slot->host_kv_access_error[0] != 0u )
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->output_token_destination != 0 )
			memcpy(async->output_token_destination,async->steps_token_count != 0u ? async->steps_tokens : slot->host_output_token_ids,(uint64_t)(async->steps_token_count != 0u ? async->steps_token_count : async->row_count) * sizeof(uint32_t));
		for (lane=0u; lane<async->lane_count; lane++)
		{
			resident = async->lane_indices[lane];
			state->kv_lane_cache_lanes[resident].context_token_count += async->steps_extra_tokens;
			atomic_store_explicit(&state->lane_bound[resident],async->lane_bound[lane],memory_order_release);
			atomic_store_explicit(&state->lane_sequence_ids[resident],async->lane_sequence_ids[lane],memory_order_release);
			atomic_store_explicit(&state->lane_next_positions[resident],async->lane_next_positions[lane],memory_order_release);
			if ( SparkKvPageCacheCompleteLane(&state->kv_page_cache,&state->kv_lane_cache_lanes[resident]) != SPARK_STATUS_OK )
				async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
		}
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	}
	else
	{
		for (lane=0u; lane<async->lane_count; lane++)
		{
			resident = async->lane_indices[lane];
			if ( SparkKvPageCacheRollbackLaneTransaction(&state->kv_page_cache,&state->kv_lane_cache_lanes[resident],state->kv_lane_mutation_flags[resident]) != SPARK_STATUS_OK )
				async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
			atomic_store_explicit(&state->lane_bound[resident],0u,memory_order_release);
		}
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	}
	atomic_fetch_add_explicit(&state->host_callback_completion_count,1u,memory_order_relaxed);
	SparkStageModuleCompleteAndReleaseClaims(async->completion_function,async->completion_context,&async->completion,state->lane_states,state->resident_sequence_capacity,async->lane_indices,async->lane_count,state->slot_states,async->slot_index);
}

#define SPARK_GLM52_CHAIN_SETTLE_TIMEOUT_NS UINT64_C(35000000000)
#define SPARK_GLM52_GRAPH_GATE_NONE UINT32_MAX
#define SPARK_GLM52_GRAPH_GATE_MULTI_WAVE 0u
#define SPARK_GLM52_GRAPH_GATE_SELECTED_CONTEXT 1u
#define SPARK_GLM52_GRAPH_GATE_ROWS 2u
#define SPARK_GLM52_GRAPH_GATE_REPORTED 4u

static const char *const SparkGlm52GraphGateNames[3] = {"multi-wave","selected-context","rows"};

static uint64_t SparkGlm52NowNs(void)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static void SparkGlm52ChainCollectives(SparkGlm52ModuleState *state,SparkTpChainCollectives *collectives)
{
	memset(collectives,0,sizeof(*collectives));
	if ( state->tp_degree > 1u && state->tp_collective_disabled == 0u && state->tp_device_collective_initialized != 0u )
	{
		collectives->items[0] = &state->tp_device_collective;
		collectives->count = 1u;
	}
}

static SparkStatus SparkGlm52WalkReduce(SparkGlm52TpChain *chain,void *device,uint32_t operation)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkTpDeviceCollectiveSubmission submission;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
		return(SPARK_STATUS_OK);
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	SparkGlm52ChainSubmission(chain,device,0u,&submission);
	return(SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,operation));
}

static uint32_t SparkGlm52WalkWave(void *context)
{
	SparkGlm52TpChain *chain = (SparkGlm52TpChain *)context;
	SparkGlm52CudaWave *wave = &chain->wave;
	SparkGlm52ExecutionSlot *slot = chain->slot;
	cudaStream_t stream = (cudaStream_t)slot->stream;
	uint32_t layer;
	if ( SparkGlm52LaunchCudaWaveBegin(wave) != 0 )
		return(1u);
	if ( SparkGlm52WalkReduce(chain,slot->hidden_bf16,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) != SPARK_STATUS_OK )
		return(2u);
	for (layer=0u; layer<wave->layer_count; layer++)
	{
		chain->next_layer = layer;
		if ( chain->state->projection_split != 0u )
		{
			if ( SparkGlm52LaunchCudaLayerAttentionProject(wave,layer) != 0 )
				return(11u);
			if ( SparkGlm52WalkReduce(chain,slot->projection_gather_bf16,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) != SPARK_STATUS_OK )
				return(12u);
			if ( SparkGlm52LaunchCudaLayerAttentionCore(wave,layer) != 0 )
				return(13u);
		}
		else if ( SparkGlm52LaunchCudaLayerAttention(wave,layer) != 0 )
			return(3u);
		if ( SparkGlm52WalkReduce(chain,slot->attention_out_bf16,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) != SPARK_STATUS_OK )
			return(4u);
		if ( SparkGlm52LaunchCudaLayerMlp(wave,layer) != 0 )
			return(5u);
		if ( SparkGlm52WalkReduce(chain,slot->hidden_bf16,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) != SPARK_STATUS_OK )
			return(6u);
	}
	if ( SparkGlm52LaunchCudaWaveHead(wave) != 0 )
		return(7u);
	if ( SparkGlm52WalkReduce(chain,slot->head_maxloc_u64,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64) != SPARK_STATUS_OK )
		return(8u);
	if ( SparkGlm52LaunchHeadMaxlocUnpack(stream,slot->head_maxloc_u64,slot->output_token,chain->wave_rows) != cudaSuccess )
		return(9u);
	if ( chain->state->owns_final_head != 0u && cudaMemcpyAsync(slot->host_output_token_ids + chain->first_row,slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream) != cudaSuccess )
		return(10u);
	return(0u);
}

static uint32_t SparkGlm52GraphGate(const SparkGlm52TpChain *chain)
{
	if ( chain->wave_rows != chain->batch->row_count )
		return(SPARK_GLM52_GRAPH_GATE_MULTI_WAVE);
	if ( SparkGlm52GraphReplayable(chain->wave.maximum_context) == 0u )
		return(SPARK_GLM52_GRAPH_GATE_SELECTED_CONTEXT);
	if ( chain->wave_rows > SPARK_TP_CHAIN_GRAPH_MAX_ROWS )
		return(SPARK_GLM52_GRAPH_GATE_ROWS);
	return(SPARK_GLM52_GRAPH_GATE_NONE);
}

static SparkStatus SparkGlm52GraphWalk(SparkGlm52TpChain *chain,const SparkTpChainCollectives *collectives,uint32_t *site)
{
	SparkGlm52ModuleState *state = chain->state;
	void **entry;
	uint32_t regime,bound,context;
	SparkStatus status = SPARK_STATUS_OK;
	context = chain->wave.maximum_context;
	regime = SparkGlm52GraphRegime(context,state->decode_split_context_threshold,state->max_sequence_positions,&bound) +
		(chain->wave.row_head_certified != 0u ? SPARK_GLM52_GRAPH_REGIME_COUNT : 0u);
	entry = SparkTpChainGraphEntry(&state->graphs[chain->slot_index],regime,chain->wave_rows);
	if ( entry == 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( *entry == 0 )
	{
		chain->wave.maximum_context = bound;
		status = SparkTpChainGraphRecord(collectives,chain->slot->stream,SparkGlm52WalkWave,chain,entry,site);
		chain->wave.maximum_context = context;
		chain->captured = 1u;
		state->graphs[chain->slot_index].captures++;
		fprintf(stderr,"GLM52-GRAPH-CAPTURE slot=%u rows=%u regime=%u bound=%u context=%u status=%s site=%u\n",chain->slot_index,chain->wave_rows,regime,bound,context,SparkStatusToString(status),*site);
		if ( status != SPARK_STATUS_OK )
		{
			state->graphs[chain->slot_index].failed++;
			return(status);
		}
	}
	status = SparkTpChainGraphPreLaunch(collectives,chain->slot->stream);
	if ( status == SPARK_STATUS_OK && cudaGraphLaunch((cudaGraphExec_t)*entry,(cudaStream_t)chain->slot->stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	chain->graph = 1u;
	return(status);
}

static void SparkGlm52TapWave(SparkGlm52TpChain *chain);

static SparkStatus SparkGlm52LinearWalk(SparkGlm52TpChain *chain,uint32_t *site)
{
	for (;;)
	{
		chain->waves++;
		*site = SparkGlm52WalkWave(chain);
		if ( *site != 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		SparkGlm52TapWave(chain);
#ifdef SPARK_SCORE_DUMP
		{
			SparkStatus score_status = SparkGlm52ScoreWave(chain);
			if ( score_status != SPARK_STATUS_OK )
			{
				*site = 20u;
				SPARK_RETURN(score_status);
			}
		}
#endif
		if ( chain->next_wave_row >= chain->batch->row_count )
			return(SPARK_STATUS_OK);
		chain->first_row = chain->next_wave_row;
		chain->wave_rows = SparkGlm52WaveRows(chain,chain->next_wave_row);
		if ( chain->wave_rows == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		chain->next_wave_row += chain->wave_rows;
		SparkGlm52BuildWave(chain);
	}
}

#include "spark_glm52_verify_steps.h"

static void SparkGlm52ChainFinish(SparkGlm52TpChain *chain,SparkStatus status)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52AsyncCompletion *async = &state->completions[chain->slot_index];
	uint64_t total_ns = SparkGlm52NowNs() - chain->start_ns;
	fprintf(stderr,"GLM52-CHAIN-TIME mode=%s slot=%u rows=%u waves=%u captured=%u walk_us=%.1f total_us=%.1f gates=%llu/%llu/%llu status=%s\n",
		chain->graph != 0u ? "graph" : "linear",chain->slot_index,async->row_count,chain->waves,chain->captured,
		(double)chain->walk_ns / 1000.0,(double)total_ns / 1000.0,
		(unsigned long long)state->chain_gates[0],(unsigned long long)state->chain_gates[1],(unsigned long long)state->chain_gates[2],SparkStatusToString(status));
	if ( status != SPARK_STATUS_OK )
		async->completion.status = status;
	atomic_store_explicit(&state->chain_busy,0u,memory_order_release);
	chain->active = 0u;
	SparkGlm52CompleteAsync(async);
	free(chain);
}

static void SparkGlm52ChainSettle(void *context)
{
	SparkGlm52TpChain *chain = (SparkGlm52TpChain *)context;
	SparkGlm52ModuleState *state = chain->state;
	SparkTpChainCollectives collectives;
	SparkStatus status;
	SparkGlm52ChainCollectives(state,&collectives);
	status = SparkStageModuleCudaWaitFor(&state->chain_wait,SPARK_GLM52_CHAIN_SETTLE_TIMEOUT_NS);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY )
	{
		fprintf(stderr,"GLM52-CHAIN-STREAM-FAIL mode=%s slot=%u status=%s\n",chain->graph != 0u ? "graph" : "linear",chain->slot_index,SparkStatusToString(status));
		SparkTpChainCancel(&collectives);
		if ( chain->graph != 0u )
			SparkTpChainDisarm(&collectives);
		SparkGlm52ChainFinish(chain,status);
		return;
	}
	if ( status == SPARK_STATUS_BUSY )
	{
		fprintf(stderr,"GLM52-CHAIN-STUCK mode=%s slot=%u; cancelling the collective\n",chain->graph != 0u ? "graph" : "linear",chain->slot_index);
		SparkTpChainCancel(&collectives);
		if ( SparkStageModuleCudaWaitFor(&state->chain_wait,SPARK_GLM52_CHAIN_SETTLE_TIMEOUT_NS) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"GLM52-CHAIN-STUCK slot=%u stream did not drain after cancel; retaining the slot\n",chain->slot_index);
			return;
		}
		if ( chain->graph != 0u )
			SparkTpChainDisarm(&collectives);
		SparkGlm52ChainFinish(chain,SPARK_STATUS_IO_ERROR);
		return;
	}
	status = SparkTpChainSettle(&collectives,chain->slot->stream,chain->graph);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"GLM52-CHAIN-SETTLE-FAIL mode=%s slot=%u status=%s\n",chain->graph != 0u ? "graph" : "linear",chain->slot_index,SparkStatusToString(status));
	SparkGlm52TapWrite(chain,status);
	if ( SparkGlm52StepsContinue(chain,&status) != 0u )
		return;
	if ( status != SPARK_STATUS_OK && chain->steps_budget != 0u )
		fprintf(stderr,"GLM52-VERIFY-STEP-FAIL slot=%u produced=%u budget=%u status=%s\n",chain->slot_index,chain->steps_produced,chain->steps_budget,SparkStatusToString(status));
	SparkGlm52ChainFinish(chain,status);
}

static void SparkGlm52RunChain(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkTpChainCollectives collectives;
	uint32_t site = 0u,gate = SPARK_GLM52_GRAPH_GATE_NONE;
	SparkStatus status;
	cudaError_t error;
	chain->start_ns = SparkGlm52NowNs();
	SparkGlm52ChainCollectives(state,&collectives);
	SparkGlm52BuildWave(chain);
	if ( state->chain_mode == SPARK_TP_CHAIN_MODE_GRAPH )
	{
		gate = SparkGlm52GraphGate(chain);
		if ( gate != SPARK_GLM52_GRAPH_GATE_NONE && state->chain_gates[gate]++ < SPARK_GLM52_GRAPH_GATE_REPORTED )
			fprintf(stderr,"GLM52-GRAPH-GATE reason=%s rows=%u batch_rows=%u context=%u: this chain runs linear\n",SparkGlm52GraphGateNames[gate],chain->wave_rows,chain->batch->row_count,chain->wave.maximum_context);
	}
	if ( state->chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && gate == SPARK_GLM52_GRAPH_GATE_NONE )
	{
		chain->waves = 1u;
		status = SparkGlm52GraphWalk(chain,&collectives,&site);
		if ( status == SPARK_STATUS_OK )
			SparkGlm52TapWave(chain);
	}
	else
		status = SparkGlm52LinearWalk(chain,&site);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemcpyAsync(chain->slot->host_kv_access_error,chain->slot->kv_access_error,SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream);
		status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"chain_kv_access");
	}
	chain->walk_ns = SparkGlm52NowNs() - chain->start_ns;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"GLM52-CHAIN-WALK-FAIL mode=%s slot=%u site=%u layer=%u status=%s\n",SparkTpChainModeName(state->chain_mode),chain->slot_index,site,chain->next_layer,SparkStatusToString(status));
		SparkTpChainCancel(&collectives);
		if ( chain->graph != 0u )
			SparkTpChainDisarm(&collectives);
		atomic_store_explicit(&state->chain_busy,0u,memory_order_release);
		SparkGlm52TpChainFail(chain,status);
		return;
	}
	status = SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm52ChainSettle,chain);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"GLM52-CHAIN-SETTLE-INLINE mode=%s slot=%u status=%s: the lazy worker refused the settle; settling on the submitting thread\n",chain->graph != 0u ? "graph" : "linear",chain->slot_index,SparkStatusToString(status));
		SparkGlm52ChainSettle(chain);
	}
}

static SparkStatus SparkGlm52ProjectionSplitConfigure(SparkGlm52ModuleState *state)
{
	const char *text = getenv("SPARK_GLM52_PROJECTION_SPLIT");
	if ( text == 0 || strcmp(text,"0") == 0 )
		state->projection_split = 0u;
	else if ( strcmp(text,"1") == 0 )
		state->projection_split = 1u;
	else
	{
		fprintf(stderr,"SPARK_GLM52_PROJECTION_SPLIT must be 0 or 1\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->projection_split != 0u && state->tp_degree > 1u && state->tp_collective_disabled != 0u )
	{
		fprintf(stderr,"GLM52-PROJECTION-SPLIT-REFUSED reason=needs the TP collective to gather the q_a/kv_a slices\n");
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	if ( state->projection_split != 0u )
		fprintf(stderr,"GLM52-PROJECTION-SPLIT rank=%u degree=%u\n",state->tp_rank,state->tp_degree);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PrefillWaveRowsConfigure(SparkGlm52ModuleState *state)
{
	const char *text = getenv("SPARK_GLM52_PREFILL_WAVE_ROWS");
	char *end = 0;
	unsigned long value;
	state->prefill_wave_rows = 0u;
	if ( text == 0 || text[0] == '\0' )
		return(SPARK_STATUS_OK);
	value = strtoul(text,&end,10);
	if ( end == text || *end != '\0' || value > state->execution_row_capacity )
	{
		fprintf(stderr,"SPARK_GLM52_PREFILL_WAVE_ROWS must be 0 (one round per wave) or 1..%u (the execution row capacity), got '%s'\n",state->execution_row_capacity,text);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->prefill_wave_rows = (uint32_t)value;
	fprintf(stderr,"GLM52-PREFILL-WAVE-ROWS rows=%u exact_rows=%u exact=%s rank=%u\n",state->prefill_wave_rows,SparkGlm52ExactWaveRows(),
		state->prefill_wave_rows <= SparkGlm52ExactWaveRows() ? "yes" : "no: waves above exact_rows use the GEMM path and are not bit-equal to one-row prefill",state->tp_rank);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ChainModeConfigure(SparkGlm52ModuleState *state)
{
	SparkTpChainCollectives collectives;
	const char *reason = 0;
	SparkStatus status;
	if ( SparkTpChainModeParse(getenv("SPARK_GLM52_CHAIN_MODE"),&state->chain_mode) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"SPARK_GLM52_CHAIN_MODE must be eager, linear or graph\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	atomic_init(&state->chain_busy,0u);
	if ( state->chain_mode == SPARK_TP_CHAIN_MODE_EAGER )
		return(SPARK_STATUS_OK);
	SparkGlm52ChainCollectives(state,&collectives);
	if ( SparkGlm52T1Enabled() != 0 )
		reason = "SPARK_GLM52_T1 traces every stage from the host; use the eager chain";
	else if ( state->lazy_pack == 0 || state->lazy_pack->worker == 0 )
		reason = "needs the weightd lazy attach worker (SPARK_WEIGHTD_ATTACH_LAZY=1)";
	else if ( state->experts_pinned == 0u )
		reason = "needs pinned experts (SPARK_GLM52_PIN_EXPERTS=1)";
	else if ( state->tp_degree > 1u && collectives.count == 0u )
		reason = "needs the TP device collective";
	else if ( SparkTpChainStreamOrdered(&collectives) == 0u )
		reason = "needs stream-ordered collectives (SPARK_TP_WAIT_MODE=hardware)";
	if ( reason != 0 )
	{
		fprintf(stderr,"GLM52-CHAIN-MODE-REFUSED mode=%s reason=%s\n",SparkTpChainModeName(state->chain_mode),reason);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	status = SparkStageModuleCudaWaitInitialize(&state->chain_wait,(cudaStream_t)state->execution_stream);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->chain_wait_initialized = 1u;
	fprintf(stderr,"GLM52-CHAIN-MODE mode=%s collectives=%u rank=%u\n",SparkTpChainModeName(state->chain_mode),collectives.count,state->tp_rank);
	return(SPARK_STATUS_OK);
}

#include "sparkpipe/family/module/spark_module_prepare_claimed_continuity.h"

static SparkStatus SparkGlm52ExecuteBatch(
	SparkGlm52ModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm52ResidentDecodeStageFrameContext *context)
{
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	SparkGlm52ClaimedContinuityContext continuity;
	SparkGlm52ExecutionSlot *slot;
	SparkGlm52TpChain *chain;
	uint8_t simulated_bound[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_sequence[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_next[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t slot_index,wave_rows;
	SparkStatus status;
	cudaError_t error;
	batch = context->batch;
	continuity.state = state;
	continuity.batch = batch;
	continuity.bound = simulated_bound;
	continuity.sequence_ids = simulated_sequence;
	continuity.next_positions = simulated_next;
	if ( state->tp_degree > 1u && state->tp_collective_disabled == 0u && state->tp_device_collective_initialized != 0u )
	{
		status = SparkTpDeviceCollectiveChainKey(&state->tp_device_collective,frame->request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	status = SparkGlm52VerifyObserveRows(state,batch);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count,SparkGlm52PrepareClaimedContinuity,&continuity);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	slot_index = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	status = SparkStageModuleSlotClaim(state->slot_states,state->pipeline_slot_count,&slot_index);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
		SPARK_RETURN(status);
	}
	slot = &state->slots[slot_index];
	slot->stream = frame->execution_stream;
	status = SparkGlmStageStageHostBatch(state,slot,batch);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleSlotRelease(state->slot_states,slot_index);
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
		SPARK_RETURN(status);
	}
	SparkGlmStagePrepareAsyncCompletion(state,frame,batch,simulated_bound,simulated_sequence,simulated_next,slot_index);
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	error = cudaMemsetAsync(slot->kv_access_error,0,SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),(cudaStream_t)slot->stream);
	status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"kv_access_reset");
	wave_rows = status == SPARK_STATUS_OK ? SparkGlmStageRoundMajorWaveRows(state,batch,0u) : 0u;
	if ( status == SPARK_STATUS_OK && wave_rows == 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
	{
		chain = (SparkGlm52TpChain *)calloc(1u,sizeof(*chain));
		if ( chain == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)cudaStreamSynchronize((cudaStream_t)slot->stream);
		SparkGlm52InvalidateClaimedLanes(state,batch->row_resident_slots,batch->active_sequence_count);
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
		SparkStageModuleSlotRelease(state->slot_states,slot_index);
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
		SPARK_RETURN(status);
	}
	chain->state = state;
	chain->slot = slot;
	chain->slot_index = slot_index;
	chain->frame = frame;
	chain->batch_copy = *batch;
	chain->batch_copy.token_ids = 0;
	chain->batch_copy.row_positions = 0;
	chain->batch_copy.row_sequence_ids = 0;
	chain->batch_copy.row_resident_slots = slot->host_resident_slots;
	chain->context_copy = *context;
	chain->context_copy.batch = &chain->batch_copy;
	chain->context = &chain->context_copy;
	chain->batch = &chain->batch_copy;
	chain->first_row = 0u;
	chain->prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	chain->wave_rows = SparkGlm52WaveRows(chain,0u);
	chain->next_wave_row = chain->wave_rows;
	chain->stage = SPARK_GLM52_CHAIN_STAGE_BEGIN;
	chain->active = 1u;
	if ( frame->tokens_per_sequence > 1u )
	{
		status = SparkGlm52StepsBegin(chain,batch);
		if ( status != SPARK_STATUS_OK )
		{
			atomic_store_explicit(&state->chain_busy,0u,memory_order_release);
			SparkGlm52TpChainFail(chain,status);
			return(SPARK_STATUS_OK);
		}
	}
	if ( state->chain_mode != SPARK_TP_CHAIN_MODE_EAGER )
		SparkGlm52RunChain(chain);
	else
		SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ExecuteChain(
	SparkGlm52ModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm52ResidentDecodeStageFrameContext *context)
{
	SparkStatus status;
	if ( state->chain_mode == SPARK_TP_CHAIN_MODE_EAGER )
		return(SparkGlm52ExecuteBatch(state,frame,context));
	if ( atomic_exchange_explicit(&state->chain_busy,1u,memory_order_acq_rel) != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkGlm52ExecuteBatch(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_store_explicit(&state->chain_busy,0u,memory_order_release);
	return(status);
}

static SparkStatus SparkGlm52ModuleExecuteFrame(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	SparkGlm52ModuleState *state;
	const SparkGlm52ResidentDecodeStageFrameContext *context;
	SparkStatus status;
	state = (SparkGlm52ModuleState *)module_state;
	context = 0;
	status = SparkGlmStageValidateFrame(state,frame,&context);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52StepsCheck(state,frame,context->batch);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state != 0 )
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_RETURN(status);
	}
	status = SparkGlm52ExecuteChain(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

#include "sparkpipe/family/module/spark_module_reset_page_cache.h"

static SparkStatus SparkGlm52ModuleAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm52ModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available;
	SparkStatus status;
	state = (SparkGlm52ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
		return(SparkGlm52ResetPageCache(state,request,decision));
	available = SparkStageModuleSlotCountFree(state->slot_states,state->pipeline_slot_count);
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u || request->new_token_count != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		SparkModelDriverInitializeAdmissionDecision(decision);
		decision->available_dispatch_slot_count = available;
		return(SparkGlm52AdmissionPredicate(state,request,decision));
	}
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->resident_sequence_capacity;
	table.max_input_row_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	table.max_sequence_positions = state->max_sequence_positions;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS |
		SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG;
	table.predicate = SparkGlm52AdmissionPredicate;
	table.predicate_context = state;
	table.cost = SparkStageModuleAdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available,request,decision);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( decision->accepted == 0u )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

static void SparkGlm52ModuleSnapshotExtend(
	void *module_state,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkGlm52ModuleState *state;
	uint32_t index,resident_count;
	state = (SparkGlm52ModuleState *)module_state;
	snapshot->host_callback_completion_count = atomic_load_explicit(&state->host_callback_completion_count,memory_order_relaxed);
	resident_count = 0u;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		resident_count += atomic_load_explicit(&state->lane_bound[index],memory_order_acquire) != 0u ? 1u : 0u;
	snapshot->resident_sequence_count = resident_count;
	snapshot->kv_token_capacity = (uint64_t)state->resident_sequence_capacity * state->max_sequence_positions;
}

static SparkStatus SparkGlm52ModuleStateTeardown(void *module_state)
{
	SparkGlm52ModuleState *state;
	uint32_t slot;
	state = (SparkGlm52ModuleState *)module_state;
	if ( cudaStreamSynchronize((cudaStream_t)state->execution_stream) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	if ( state->lazy_pack != 0 )
	{
		if ( state->lazy_pack->worker != 0 )
		{
			SparkStatus status = SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlmStageLazyRetryRetained,state);
			if ( status == SPARK_STATUS_OK )
				status = SparkWeightdWorkerWaitIdle(state->lazy_pack->worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( __atomic_load_n(&state->lazy_retained[slot],__ATOMIC_ACQUIRE) != 0 )
			return(SPARK_STATUS_BUSY);
	if ( atomic_load_explicit(&state->chain_busy,memory_order_acquire) != 0u )
		return(SPARK_STATUS_BUSY);
#ifdef SPARK_SCORE_DUMP
	SparkGlm52ScoreClose(state);
#endif
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		SparkTpChainGraphTableDestroy(&state->graphs[slot]);
	if ( state->chain_wait_initialized != 0u )
	{
		SparkStatus status = SparkStageModuleCudaWaitDestroy(&state->chain_wait);
		if ( status != SPARK_STATUS_OK )
			return(status);
		state->chain_wait_initialized = 0u;
	}
	if ( state->tp_device_collective_initialized != 0u )
	{
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
		if ( state->tp_device_collective.implementation != 0 )
			return(SPARK_STATUS_BUSY);
		state->tp_device_collective_initialized = 0u;
	}
	if ( state->lazy_pack != 0 && state->expert_pin_lease_count != 0u )
	{
		SparkStatus status = SparkGlm52ReleasePinnedExperts(state,(cudaStream_t)state->execution_stream);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( state->lazy_pack != 0 )
	{
		SparkStatus status = SparkWeightdLazyPackDestroy(state->lazy_pack);
		if ( status != SPARK_STATUS_OK )
			return(status);
		state->lazy_pack = 0;
	}
	if ( state->kv_page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->kv_page_store);
	SparkGlm52ReleaseSlotHost(state);
	SparkGlm52VerifyRelease(state);
	free(state->kv_blocks);
	free(state->kv_resident_slot_logical_block_indices);
	free(state->kv_entries);
	free(state->kv_sequences);
	free(state->kv_hash_bucket_heads);
	free(state->kv_entry_indices_by_logical_page);
	free(state->kv_page_staging);
	free(state->kv_lane_logical_pages);
	free(state->kv_lane_page_count);
	free(state->kv_lane_mutable_page);
	free(state->kv_lane_mutation_flags);
	free(state->kv_lane_cache_lanes);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ModulePrepare(
	void *module_state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services)
{
	SparkGlm52ModuleState *state;
	const char *pack_path;
	SparkStatus status;
	uint32_t lane;
	state = (SparkGlm52ModuleState *)module_state;
	status = SparkGlm52ModuleConfigure(state,configuration,host_services,&pack_path);
	if ( status == SPARK_STATUS_OK && SparkGlm52ConfigureCudaModule(&state->multiprocessor_count) != 0 )
		status = SPARK_STATUS_TARGET_MISMATCH;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackLoad(state,pack_path);
	if ( status == SPARK_STATUS_OK && getenv("SPARK_GLM52_PIN_EXPERTS") != 0 && strcmp(getenv("SPARK_GLM52_PIN_EXPERTS"),"1") == 0 )
	{
		status = state->lazy_pack != 0 ? SparkGlm52PinAllExperts(state,SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT) : SPARK_STATUS_UNSUPPORTED;
		fprintf(stderr,"EXPERT-RESIDENCY mode=%s keys=%u leases=%u status=%s\n",status == SPARK_STATUS_OK ? "pinned" : "EXPERT-PIN-FAILED",state->expert_pin_key_count,state->expert_pin_lease_count,SparkStatusToString(status));
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ProjectionSplitConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PrefillWaveRowsConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52AllocateCaches(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlmStageAllocateSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ModuleInitializeTpCollective(state,(const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlmStageBuildHeadShadow(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ChainModeConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52VerifyConfigure(state);
#ifdef SPARK_SCORE_DUMP
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ScoreOpen(state,(const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context);
#endif
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states,state->resident_sequence_capacity);
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_init(&state->lane_bound[lane],0u);
		atomic_init(&state->lane_sequence_ids[lane],0u);
		atomic_init(&state->lane_next_positions[lane],0u);
	}
	atomic_init(&state->host_callback_completion_count,0u);
	atomic_init(&state->tp_next_ordinal,0u);
	return(SPARK_STATUS_OK);
}

#include "sparkpipe/family/module/spark_module_describe.h"

static const SparkStageModuleLifecycleOps SparkGlm52ModuleLifecycle =
{
	sizeof(SparkGlm52ModuleState),
	0,
	SparkGlm52ModuleDescribe,
	SparkGlm52ModulePrepare,
	0,
	SparkGlm52ModuleStateTeardown,
	SparkGlm52ModuleExecuteFrame,
	SparkGlm52ModuleAdmit,
	SparkGlm52ModuleSnapshotExtend
};

SPARK_STAGE_MODULE_LIFECYCLE_ENTRY_POINTS(
	SparkGlm52ResidentDecodeStage,
	&SparkGlm52ModuleLifecycle)

#include "sparkpipe/family/module/spark_module_validate_frame_buffers.h"
