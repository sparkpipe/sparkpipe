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
#include "sparkpipe/family/module/spark_module_tp_collective_required.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_tp_mesh_register.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_stage_module_lifecycle.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_tp_chain_graph.h"
#include "sparkpipe/spark_glm52_graph_regime.h"
#include "inference/kernels/topk_exact_plan.h"
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
#define SPARK_GLM52_DISTRIBUTION_CHUNK_LIMIT 16u

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
	SparkSamplingLogprob *distribution_destination;
	uint32_t distribution_count;
	uint32_t distribution_source[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
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
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
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
	SparkStageKvBinding kv;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	const char *kv_snapshot_directory;
	uint64_t kv_snapshot_maximum_bytes;
	const char *model_id;
	uint32_t kv_cache_codec;
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
	SparkKvShard kv_shard;
	uint64_t kv_gather_latent_tail_slot;
	uint32_t prefill_wave_rows;
	uint32_t chain_wait_initialized;
	SparkStageModuleCudaWait chain_wait;
	atomic_uint chain_busy;
	SparkTpChainGraphTable graphs[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t distribution_wave_capacity;
	uint32_t distribution_chunk_rows;
	uint32_t distribution_sub_rows;
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
	SPARK_GLM52_CHAIN_STAGE_REDUCE_PROJECTION,
	SPARK_GLM52_CHAIN_STAGE_SHARD_EXCHANGE,
	SPARK_GLM52_CHAIN_STAGE_SHARD_PARTIALS,
	SPARK_GLM52_CHAIN_STAGE_KV_GATHER,
	SPARK_GLM52_CHAIN_STAGE_KV_DIGEST,
	SPARK_GLM52_CHAIN_STAGE_KV_SELECT,
	SPARK_GLM52_CHAIN_STAGE_KV_SELECTED,
	SPARK_GLM52_CHAIN_STAGE_DISTRIBUTION
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
	uint32_t row_ordered;
	uint32_t shard_chunk;
	uint32_t distribution_count;
	uint32_t distribution_next;
	uint32_t distribution_chunk_first;
	uint32_t distribution_ordered_rows[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t row_order[];
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

#include "model-families/glm52/stage_module/spark_glm_stage_module.h"

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
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 || host_services->node_context == 0 || host_services->execution_stream == 0 ||
		host_services->kv_logical_page_capacity == 0u || host_services->kv_physical_page_capacity == 0u || host_services->kv_physical_page_capacity > host_services->kv_logical_page_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->model_id = configuration->model_id;
	context = (const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( context->stage_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT || context->stage_index >= context->stage_count || context->first_layer_index != SparkGlm52ResidentDecodeStageFirstLayer(context->stage_index) || context->layer_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE || context->expert_weight_codec != GLM_EXPERT_WEIGHT_CODEC || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions == 0u || context->max_sequence_positions > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS || context->execution_row_capacity == 0u || context->execution_row_capacity > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || context->execution_row_capacity > SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS || context->decode_split_context_threshold > context->max_sequence_positions || context->tp_degree == 0u || context->tp_rank >= context->tp_degree || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || context->model_revision == 0 || context->model_revision[0] == '\0' || strlen(context->model_revision) >= sizeof(state->model_revision) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightCodecIsKnown(context->expert_weight_codec) == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( context->tp_degree != 1u && (SPARK_GLM52_MODEL_HEAD_COUNT % context->tp_degree != 0u || SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree != 0u || SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u || SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	memset(&state->kv_shard,0,sizeof(state->kv_shard));
	if ( context->tp_degree > 1u )
	{
		state->kv_shard.degree = context->tp_degree;
		state->kv_shard.rank = context->tp_rank;
		state->kv_shard.grain = 1u;
		if ( context->tp_collective_identifier == 0u || SparkKvShardValid(state->kv_shard,SPARK_GLM_KV_BLOCK_TOKEN_COUNT) == 0u )
		{
			fprintf(stderr,"GLM52-KV-SHARD-REFUSED tp=%u rank=%u collective=%s: each rank stores 1/%u of every sequence's KV, so attention needs the TP collective and %u-token pages must split evenly\n",
				context->tp_degree,context->tp_rank,context->tp_collective_identifier != 0u ? "yes" : "no",context->tp_degree,SPARK_GLM_KV_BLOCK_TOKEN_COUNT);
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
	}
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->expert_weight_codec = context->expert_weight_codec;
	state->tp_degree = context->tp_degree;
	state->tp_rank = context->tp_rank;
	state->kv_backing_directory = host_services->kv_backing_directory;
	state->kv_backing_maximum_bytes = host_services->kv_backing_maximum_bytes;
	state->kv_snapshot_directory = host_services->kv_snapshot_directory;
	state->kv_snapshot_maximum_bytes = host_services->kv_snapshot_maximum_bytes;
	state->kv_logical_page_capacity = host_services->kv_logical_page_capacity;
	state->kv_physical_page_capacity = host_services->kv_physical_page_capacity;
	if ( SparkModuleTpCollectiveIdentifier(SPARK_GLM52_MODULE_TAG,context->tp_degree,context->tp_collective_identifier,&state->tp_collective_disabled) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		state->kv_cache_codec = header.kv_cache_codec;
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
		if ( state->slots[index].distribution_host != 0 )
			(void)cudaFreeHost(state->slots[index].distribution_host);
		state->slots[index].distribution_host = 0;
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
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_PREFILL_TABLE_BLOCKS(state->execution_row_capacity),1u,sizeof(uint32_t),(void **)&slot->prefill_block_table);
	slot->prefill_union_entries = state->max_sequence_positions > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT ?
		(uint64_t)SPARK_GLM52_PREFILL_TABLE_BLOCKS(state->execution_row_capacity) * SPARK_GLM52_PREFILL_BLOCK_ROWS * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT : 0u;
	if ( status == SPARK_STATUS_OK && slot->prefill_union_entries != 0u ) status = SparkGlmStageAllocateBytes(state,1u,slot->prefill_union_entries,sizeof(uint32_t),(void **)&slot->prefill_union_positions);
	if ( status == SPARK_STATUS_OK && slot->prefill_union_entries != 0u ) status = SparkGlmStageAllocateBytes(state,1u,slot->prefill_union_entries,sizeof(uint8_t),(void **)&slot->prefill_union_masks);
	if ( status == SPARK_STATUS_OK && slot->prefill_union_entries != 0u ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_PREFILL_TABLE_BLOCKS(state->execution_row_capacity),1u,sizeof(uint32_t),(void **)&slot->prefill_union_counts);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_PREFILL_WAVE_SPANS,1u,sizeof(uint32_t),(void **)&slot->head_rows);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,SPARK_GLM52_PREFILL_WAVE_SPANS,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->head_hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,SPARK_GLM52_PREFILL_WAVE_SPANS,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->head_residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,SPARK_GLM52_PREFILL_WAVE_SPANS,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->head_normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_PREFILL_WAVE_SPANS,1u,sizeof(uint32_t),(void **)&slot->head_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_PREFILL_WAVE_SPANS,1u,sizeof(float),(void **)&slot->head_score);
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
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_QUERY_B_DIMENSION / state->tp_degree,(void **)&slot->q_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree * SPARK_GLM52_MODEL_LATENT_DIMENSION,(void **)&slot->query_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree * SPARK_GLM52_MODEL_ROPE_DIMENSION,(void **)&slot->query_rope_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_QUERY_DIMENSION,(void **)&slot->index_query_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_key_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT,(void **)&slot->index_head_weight_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,(void **)&slot->kv_slot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree * SPARK_GLM52_MODEL_LATENT_DIMENSION,(void **)&slot->attention_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree * SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION,(void **)&slot->attention_value_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->attention_out_bf16);
	if ( status == SPARK_STATUS_OK && state->projection_split != 0u ) status = SparkGlmStageAllocateRows(state,rows,SparkGlm52ProjectionSliceWidth(state->tp_degree),(void **)&slot->projection_local_bf16);
	if ( status == SPARK_STATUS_OK && state->projection_split != 0u ) status = SparkGlmStageAllocateRows(state,rows,(uint64_t)state->tp_degree * SparkGlm52ProjectionSliceWidth(state->tp_degree),(void **)&slot->projection_gather_bf16);
	SPARK_RETURN(status);
}

static uint64_t SparkGlm52ShardUnits(uint64_t bytes)
{
	return((bytes + SPARK_GLM52_SHARD_UNIT_BYTES - 1u) / SPARK_GLM52_SHARD_UNIT_BYTES);
}

static uint32_t SparkGlm52ShardScatterRows(const SparkGlm52ModuleState *state)
{
	return(state->execution_row_capacity < SPARK_GLM52_SHARD_SCATTER_ROWS ? state->execution_row_capacity : SPARK_GLM52_SHARD_SCATTER_ROWS);
}

static uint32_t SparkGlm52ShardChunkCapacity(const SparkGlm52ModuleState *state)
{
	return(state->execution_row_capacity / SPARK_GLM52_SHARD_SLOT_ALIGN * SPARK_GLM52_SHARD_SLOT_ALIGN);
}

static void SparkGlm52ShardScatterSizes(SparkKvShard shard,uint32_t rows,uint32_t *query_units,uint32_t *indexing_units,uint32_t *partial_units,uint64_t *candidate_offset)
{
	uint64_t heads = SPARK_GLM52_MODEL_HEAD_COUNT / shard.degree;
	uint64_t query = (uint64_t)rows * heads * SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t);
	*candidate_offset = query;
	*query_units = (uint32_t)SparkGlm52ShardUnits(query);
	*indexing_units = (uint32_t)SparkGlm52ShardUnits(query + (uint64_t)rows * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT * SPARK_GLM52_SHARD_CANDIDATE_BYTES);
	*partial_units = (uint32_t)SparkGlm52ShardUnits((uint64_t)rows * heads * SPARK_GLM52_SHARD_PARTIAL_FLOATS * sizeof(float));
}

static void SparkGlm52ShardGatherSizes(const SparkGlm52ModuleState *state,uint32_t old_bound,SparkKvShardSectionLayout *latent)
{
	*latent = SparkKvShardSectionLayoutBuild(SparkKvShardGatherKeys(state->kv_shard,old_bound),SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t),
		SPARK_GLM52_SHARD_UNIT_BYTES,SPARK_GLM52_SHARD_SLOT_ALIGN,SparkGlm52ShardChunkCapacity(state));
}

static uint32_t SparkGlm52ShardSelectKeep(SparkKvShard shard,uint32_t context)
{
	uint32_t keys = SparkKvShardGatherKeys(shard,context);
	return(keys < SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT ? keys : SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT);
}

static uint32_t SparkGlm52ShardSelectSlice(SparkKvShard shard,uint32_t rows)
{
	return((rows + shard.degree - 1u) / shard.degree);
}

static uint32_t SparkGlm52DistributionWidth(const SparkGlm52ModuleState *state)
{
	return(SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree);
}

static SparkStatus SparkGlm52DistributionConfigure(SparkGlm52ModuleState *state)
{
	uint32_t elements,sub_rows,chunk;
	state->distribution_wave_capacity = 0u;
	state->distribution_chunk_rows = 0u;
	state->distribution_sub_rows = 0u;
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	if ( state->tp_degree == 0u || SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT % state->tp_degree != 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->distribution_wave_capacity = state->execution_row_capacity < SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ? state->execution_row_capacity : SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	if ( state->tp_degree == 1u )
	{
		state->distribution_sub_rows = 1u;
		state->distribution_chunk_rows = state->distribution_wave_capacity;
		return(SPARK_STATUS_OK);
	}
	elements = 2u * SparkGlm52DistributionWidth(state);
	for (sub_rows=1u; sub_rows<=elements && (elements % sub_rows != 0u || elements / sub_rows > SPARK_GLM52_MODEL_HIDDEN_DIMENSION); sub_rows++)
		;
	chunk = sub_rows <= elements ? state->execution_row_capacity / sub_rows : 0u;
	if ( chunk == 0u )
	{
		fprintf(stderr,"GLM52-DISTRIBUTION-REFUSED tp=%u rows=%u: a full-vocab logit row needs %u collective rows and the execution row capacity holds fewer\n",state->tp_degree,state->execution_row_capacity,sub_rows);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->distribution_sub_rows = sub_rows;
	state->distribution_chunk_rows = chunk < SPARK_GLM52_DISTRIBUTION_CHUNK_LIMIT ? chunk : SPARK_GLM52_DISTRIBUTION_CHUNK_LIMIT;
	if ( state->distribution_chunk_rows > state->distribution_wave_capacity )
		state->distribution_chunk_rows = state->distribution_wave_capacity;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52AllocateSlotDistribution(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	uint64_t entries = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,rows = state->distribution_wave_capacity,width = SparkGlm52DistributionWidth(state);
	uint64_t host_bytes;
	uint8_t *cursor;
	cudaError_t error;
	SparkStatus status;
	status = SparkGlmStageAllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&slot->distribution_wave_rows);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&slot->distribution_positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,entries,1u,sizeof(SparkRowSampling),(void **)&slot->distribution_rules);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,entries,SPARK_SAMPLING_MAX_LOGPROBS,sizeof(SparkSamplingLogprob),(void **)&slot->distribution_logprobs);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->distribution_hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->distribution_residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->distribution_normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,width,sizeof(float),(void **)&slot->distribution_logits_f32);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u ) status = SparkGlmStageAllocateBytes(state,(uint64_t)state->tp_degree * state->distribution_chunk_rows,width,sizeof(float),(void **)&slot->distribution_gathered_f32);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	host_bytes = entries * (2u * sizeof(uint32_t) + sizeof(SparkRowSampling) + SPARK_SAMPLING_MAX_LOGPROBS * sizeof(SparkSamplingLogprob));
	error = cudaHostAlloc(&slot->distribution_host,host_bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"distribution_host"));
	memset(slot->distribution_host,0,host_bytes);
	cursor = (uint8_t *)slot->distribution_host;
	slot->host_distribution_rules = (SparkRowSampling *)cursor;
	cursor += entries * sizeof(SparkRowSampling);
	slot->host_distribution_logprobs = (SparkSamplingLogprob *)cursor;
	cursor += entries * SPARK_SAMPLING_MAX_LOGPROBS * sizeof(SparkSamplingLogprob);
	slot->host_distribution_wave_rows = (uint32_t *)cursor;
	cursor += entries * sizeof(uint32_t);
	slot->host_distribution_positions = (uint32_t *)cursor;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52AllocateSlotShard(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	uint32_t query_units,indexing_units,partial_units,degree = state->kv_shard.degree,rows = state->execution_row_capacity;
	uint64_t candidate_offset,latent_bytes,select_rows = (uint64_t)SparkGlm52ShardSelectSlice(state->kv_shard,rows) * degree;
	SparkKvShardSectionLayout latent;
	SparkStatus status;
	if ( SparkGlm52ShardChunkCapacity(state) == 0u || rows % degree != 0u )
	{
		fprintf(stderr,"GLM52-KV-SHARD-REFUSED rank=%u: %u execution rows must be a non-zero multiple of the %u context-split ranks\n",state->kv_shard.rank,rows,degree);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	SparkGlm52ShardScatterSizes(state->kv_shard,SparkGlm52ShardScatterRows(state),&query_units,&indexing_units,&partial_units,&candidate_offset);
	SparkGlm52ShardGatherSizes(state,state->max_sequence_positions,&latent);
	if ( latent.chunks == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	latent_bytes = (uint64_t)latent.chunks * latent.chunk_bytes;
	state->kv_gather_latent_tail_slot = degree * latent_bytes / latent.slot_bytes;
	status = SparkGlmStageAllocateBytes(state,indexing_units,SPARK_GLM52_SHARD_UNIT_BYTES,1u,(void **)&slot->shard_send_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,(uint64_t)degree * indexing_units,SPARK_GLM52_SHARD_UNIT_BYTES,1u,(void **)&slot->shard_received_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,(uint64_t)degree * partial_units,SPARK_GLM52_SHARD_UNIT_BYTES,1u,(void **)&slot->shard_partials_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,(uint64_t)degree * partial_units,SPARK_GLM52_SHARD_UNIT_BYTES,1u,(void **)&slot->shard_partials_received_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SparkGlm52ShardScatterRows(state),SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint32_t),(void **)&slot->index_local_selected);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,latent_bytes,1u,1u,(void **)&slot->kv_gather_latent_pack);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,(uint64_t)degree * latent_bytes + (uint64_t)rows * latent.slot_bytes,1u,1u,(void **)&slot->kv_gather_latent_pool);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,state->max_sequence_positions,1u,sizeof(uint32_t),(void **)&slot->kv_gather_latent_remap);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u ) status = SparkGlmStageAllocateBytes(state,select_rows,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint32_t),(void **)&slot->kv_select_local);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u ) status = SparkGlmStageAllocateBytes(state,select_rows,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint64_t),(void **)&slot->kv_select_candidates);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u ) status = SparkGlmStageAllocateBytes(state,select_rows,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint64_t),(void **)&slot->kv_select_received);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u ) status = SparkGlmStageAllocateBytes(state,select_rows / degree,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint32_t),(void **)&slot->kv_select_merged);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,1u,sizeof(uint32_t),(void **)&slot->kv_gather_rows_zero);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,(uint64_t)SPARK_GLM52_SHARD_PLAN_ARRAYS * state->resident_sequence_capacity + SPARK_GLM52_SHARD_PLAN_WORDS,1u,sizeof(uint32_t),(void **)&slot->kv_gather_plan);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_SHARD_DIGEST_WORDS,1u,sizeof(unsigned long long),(void **)&slot->kv_gather_digest);
	if ( status == SPARK_STATUS_OK && cudaMemset(slot->kv_gather_rows_zero,0,(size_t)rows * sizeof(uint32_t)) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		fprintf(stderr,"GLM52-KV-SHARD rank=%u degree=%u scatter_rows=%u query_units=%u indexing_units=%u partial_units=%u gather_latent=%u x %u units select_slice=%u transient_bytes=%llu\n",
			state->kv_shard.rank,degree,SparkGlm52ShardScatterRows(state),query_units,indexing_units,partial_units,latent.chunks,latent.chunk_units,(uint32_t)(select_rows / degree),
			(unsigned long long)((degree + 1u) * latent_bytes + (uint64_t)rows * latent.slot_bytes + (uint64_t)state->max_sequence_positions * sizeof(uint32_t) +
				(state->index_layer_count != 0u ? select_rows * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT * (2u * sizeof(uint64_t) + sizeof(uint32_t)) + select_rows / degree * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT * sizeof(uint32_t) : 0u)));
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
	status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_MOE_ROUTED_GATE_UP_DIMENSION / state->tp_degree,(void **)&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_MOE_TOP_K * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION / state->tp_degree,(void **)&slot->intermediate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,packed_rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->expert_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->shared_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT,sizeof(float),(void **)&slot->router_logits_f32);
	slot->selection_rows = LmTopkExactSelectionRows((uint32_t)rows,state->max_sequence_positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,slot->selection_rows,state->max_sequence_positions,sizeof(float),(void **)&slot->selection_scores_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree),SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS,sizeof(float),(void **)&slot->attention_split_partials_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint32_t),(void **)&slot->selected_positions);
	slot->topk_scratch_entries = LmTopkExactScratchEntries(slot->selection_rows,state->max_sequence_positions,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT);
	if ( status == SPARK_STATUS_OK && slot->topk_scratch_entries != 0u ) status = SparkGlmStageAllocateBytes(state,1u,slot->topk_scratch_entries,sizeof(float),(void **)&slot->topk_scratch_values_f32);
	if ( status == SPARK_STATUS_OK && slot->topk_scratch_entries != 0u ) status = SparkGlmStageAllocateBytes(state,1u,slot->topk_scratch_entries,sizeof(uint32_t),(void **)&slot->topk_scratch_positions);
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
	if ( status == SPARK_STATUS_OK && state->kv_shard.degree > 1u )
		status = SparkGlm52AllocateSlotShard(state,slot);
	if ( status == SPARK_STATUS_OK && state->distribution_wave_capacity != 0u )
		status = SparkGlm52AllocateSlotDistribution(state,slot);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm52AllocateCaches(SparkGlm52ModuleState *state)
{
	SparkStageKvConfiguration configuration;
	uint32_t local;
	state->index_layer_count = 0u;
	for (local=0u; local<state->layer_count; local++)
	{
		state->index_ordinal_by_local_layer[local] = SPARK_GLM52_NO_INDEX_ORDINAL;
		if ( SparkGlm52StagePackLayerHasFullIndexer(state->first_layer_index + local) != 0u )
			state->index_ordinal_by_local_layer[local] = state->index_layer_count++;
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.module_tag = SPARK_GLM52_MODULE_TAG;
	configuration.block_token_count = SPARK_GLM_KV_BLOCK_TOKEN_COUNT;
	configuration.region_count = state->index_layer_count != 0u ? 2u : 1u;
	configuration.regions[0].layout = SPARK_STAGE_KV_REGION_PAGE_MAJOR;
	configuration.regions[0].layer_count = state->layer_count;
	configuration.regions[0].layer_page_bytes = (uint64_t)SPARK_GLM_KV_BLOCK_TOKEN_COUNT * SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t);
	configuration.regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	configuration.regions[1].layer_count = state->index_layer_count;
	configuration.regions[1].layer_page_bytes = (uint64_t)SPARK_GLM_KV_BLOCK_TOKEN_COUNT * SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION * sizeof(uint16_t);
	configuration.arena_kv_head_count = SPARK_GLM_KV_ARENA_KV_HEAD_COUNT;
	configuration.arena_head_dim = SPARK_GLM_KV_ARENA_HEAD_DIMENSION;
	configuration.arena_bytes_per_scalar = SPARK_GLM_KV_BYTES_PER_SCALAR;
	SparkGlmKvFillCapacityRequest(&configuration.capacity_request);
	configuration.capacity_request.index_key_layer_count = state->index_layer_count;
	configuration.capacity_request.index_key_dimension = SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION;
	configuration.capacity_request.index_key_bytes_per_scalar = 2u;
	configuration.model_id = state->model_id;
	configuration.model_revision = state->model_revision;
	if ( state->lazy_pack == 0 )
	{
		fprintf(stderr,"%s kv binding refused: no weightd lazy pack, so the KV layout has no pack identity\n",SPARK_GLM52_MODULE_TAG);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memcpy(configuration.pack_sha256,state->lazy_pack->pack_sha256,SPARK_SHA256_DIGEST_BYTES);
	if ( SparkGlm52ContractHash(configuration.contract_sha256) < 0 )
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	configuration.expert_codec = state->expert_weight_codec;
	configuration.kv_codec = state->kv_cache_codec;
	configuration.driver_symbol = (const void *)&SparkGlm52AllocateCaches;
	configuration.snapshot_directory = state->kv_snapshot_directory;
	configuration.snapshot_maximum_bytes = state->kv_snapshot_maximum_bytes;
	configuration.resident_sequence_capacity = state->resident_sequence_capacity;
	configuration.max_sequence_positions = state->max_sequence_positions;
	configuration.max_input_row_count = state->execution_row_capacity;
	configuration.logical_page_count = state->kv_logical_page_capacity;
	configuration.physical_page_count = state->kv_physical_page_capacity;
	configuration.pipeline_slot_count = state->pipeline_slot_count;
	configuration.backing_directory = state->kv_backing_directory;
	configuration.backing_maximum_bytes = state->kv_backing_maximum_bytes;
	configuration.context_shard = state->kv_shard;
	return(SparkStageKvBindingInitialize(&state->kv,&configuration));
}

static SparkStatus SparkGlm52AdmissionPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm52ModuleState *state = (SparkGlm52ModuleState *)context;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageKvBindingAdmit(&state->kv,request,decision));
}


typedef struct SparkGlm52ClaimedContinuityContext
{
	SparkGlm52ModuleState *state;
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

static uint32_t SparkGlm52PackedPrefill(const SparkGlm52ModuleState *state,const SparkGlm52ExecutionSlot *slot)
{
	return(SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree == SPARK_GLM52_PREFILL_BLOCK_ROWS && slot->prefill_block_table != 0 ? 1u : 0u);
}

static uint32_t SparkGlm52KvGatherOldBound(const uint32_t *positions,uint32_t rows)
{
	uint32_t row,first = positions[0],end = positions[0] + 1u,contiguous = 1u;
	for (row=1u; row<rows; row++)
	{
		if ( positions[row] < positions[row - 1u] )
			return(UINT32_MAX);
		if ( positions[row] - positions[row - 1u] > 1u )
			contiguous = 0u;
		end = positions[row] + 1u;
	}
	return(contiguous != 0u ? first : end);
}

static void SparkGlm52WaveShardSizes(const SparkGlm52ModuleState *state,SparkGlm52CudaWave *wave,uint32_t old_bound)
{
	wave->kv_shard = state->kv_shard;
	if ( state->kv_shard.degree < 2u )
		return;
	wave->kv_gather = wave->single_sequence_rows != 0u && wave->row_count > SparkGlm52ExactWaveRows() ? 1u : 0u;
	if ( wave->kv_gather != 0u )
	{
		wave->kv_gather_old_bound = old_bound;
		wave->kv_gather_capacity = state->max_sequence_positions;
		wave->kv_gather_latent_tail_slot = state->kv_gather_latent_tail_slot;
		SparkGlm52ShardGatherSizes(state,old_bound,&wave->kv_gather_latent);
		wave->kv_gather_old_keys = (uint32_t)wave->kv_gather_latent.keys;
		wave->kv_select_keep = SparkGlm52ShardSelectKeep(state->kv_shard,wave->maximum_context);
		wave->kv_select_slice = SparkGlm52ShardSelectSlice(state->kv_shard,wave->row_count);
		wave->kv_select_rows = wave->kv_select_slice * state->kv_shard.degree;
		return;
	}
	SparkGlm52ShardScatterSizes(state->kv_shard,wave->row_count,&wave->shard_query_units,&wave->shard_indexing_units,&wave->shard_partial_units,&wave->shard_candidate_offset);
}

static uint32_t SparkGlm52KvGatherOps(const SparkGlm52CudaWave *wave,uint32_t layer)
{
	(void)layer;
	return(wave->kv_gather_latent.chunks);
}

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
	wave->execution_row_capacity = state->execution_row_capacity;
	wave->max_sequence_positions = state->max_sequence_positions;
	wave->pages_per_sequence = state->kv.pages_per_sequence;
	wave->physical_page_count = state->kv.physical_page_count;
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
	wave->row_head_certified = chain->prefill != 0u && state->prefill_wave_rows != 0u ? 1u : 0u;
	wave->single_sequence_rows = chain->row_ordered;
	wave->prefill_block_table = chain->row_ordered != 0u && SparkGlm52PackedPrefill(state,slot) != 0u ? slot->prefill_block_table : 0;
	wave->head_certified_fp8_payload = state->head_certified_fp8_payload;
	wave->head_certified_fp8_scale_f32 = state->head_certified_fp8_scale_f32;
	wave->head_certified_fp8_norm_f32 = state->head_certified_fp8_norm_f32;
	wave->layers = state->layers;
	wave->slot = slot;
	wave->kv_cache = state->kv.region_base[0];
	wave->kv_layer_stride_bytes = state->kv.region_layer_stride_bytes[0];
	wave->index_cache = state->kv.region_count > 1u ? state->kv.region_base[1] : 0;
	wave->index_layer_stride_bytes = state->kv.region_count > 1u ? state->kv.region_layer_stride_bytes[1] : 0u;
	wave->index_ordinal_by_local_layer = state->index_ordinal_by_local_layer;
	wave->page_table = state->kv.page_table;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->decode_split_context_threshold = state->decode_split_context_threshold;
	wave->attention_split_partials_f32 = slot->attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(
		state->execution_row_capacity,SPARK_GLM52_MODEL_HEAD_COUNT / state->tp_degree);
	{
		uint32_t old_bound = chain->row_ordered != 0u ? SparkGlm52KvGatherOldBound(wave->host_positions,wave->row_count) : maximum_context;
		SparkGlm52WaveShardSizes(state,wave,old_bound != UINT32_MAX ? old_bound : maximum_context);
	}
	for (row=0u; row<chain->distribution_count && chain->distribution_ordered_rows[row] < chain->first_row; row++)
		;
	wave->distribution_first = row;
	for (; row<chain->distribution_count && chain->distribution_ordered_rows[row] < chain->first_row + chain->wave_rows; row++)
		slot->host_distribution_wave_rows[row] = chain->distribution_ordered_rows[row] - chain->first_row;
	wave->distribution_count = row - wave->distribution_first;
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
	return(SparkGlm52GraphRowClass(regime->positions[row] + 1u,regime->split_threshold,regime->max_positions));
}

static uint32_t SparkGlm52RowSelectionRegime(void *context,uint32_t row)
{
	const SparkGlm52WaveRegimeContext *regime = (const SparkGlm52WaveRegimeContext *)context;
	return(regime->positions[row] + 1u > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT ? 1u : 0u);
}

static uint32_t SparkGlm52ShardCapRows(const SparkGlm52ModuleState *state,uint32_t rows)
{
	return(state->kv_shard.degree > 1u && rows > SparkGlm52ShardScatterRows(state) ? SparkGlm52ShardScatterRows(state) : rows);
}

static uint32_t SparkGlm52WaveRows(const SparkGlm52TpChain *chain,uint32_t first_row)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkStageModuleClaimedLaneContext lanes;
	SparkGlm52WaveRegimeContext regime;
	if ( chain->prefill == 0u || state->prefill_wave_rows == 0u )
		return(SparkGlm52ShardCapRows(state,SparkGlmStageRoundMajorWaveRows(state,chain->batch,first_row)));
	lanes.index_states = state->lane_states;
	lanes.index_capacity = state->resident_sequence_capacity;
	regime.positions = chain->slot->host_positions;
	regime.split_threshold = state->decode_split_context_threshold;
	regime.max_positions = state->max_sequence_positions;
	if ( chain->row_ordered != 0u )
	{
		uint32_t spans = SparkGlm52PackedPrefill(state,chain->slot) != 0u && state->kv_shard.degree < 2u ? SPARK_GLM52_PREFILL_WAVE_SPANS : 1u;
		uint32_t rows = SparkRowLayoutPackedSpanWaveRowCount(first_row,chain->batch->row_count,chain->slot->host_resident_slots,SparkGlm52RowSelectionRegime,&regime,state->prefill_wave_rows,spans);
		if ( rows > SparkGlm52ExactWaveRows() )
			return(rows);
		return(SparkRowLayoutPackedSpanWaveRowCount(first_row,chain->batch->row_count,chain->slot->host_resident_slots,SparkGlm52RowRegime,&regime,state->prefill_wave_rows,spans));
	}
	return(SparkGlm52ShardCapRows(state,SparkRowLayoutRoundSpanWaveRowCount(first_row,chain->batch->row_count,chain->batch->row_resident_slots,SparkStageModuleClaimedLaneOrdinal,&lanes,SparkGlm52RowRegime,&regime,state->prefill_wave_rows)));
}

static SparkStatus SparkGlm52OrderPrefillRows(SparkGlm52TpChain *chain)
{
	SparkGlm52ExecutionSlot *slot = chain->slot;
	uint32_t rows = chain->batch->row_count,*scratch = chain->row_order + rows,*columns[3],column,row;
	SparkStatus status;
	status = SparkRowLayoutLaneMajorOrder(rows,slot->host_resident_slots,chain->row_order);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	columns[0] = slot->host_resident_slots;
	columns[1] = slot->host_positions;
	columns[2] = chain->state->owns_embedding != 0u ? slot->host_token_ids : 0;
	for (column=0u; column<3u; column++)
	{
		if ( columns[column] == 0 )
			continue;
		for (row=0u; row<rows; row++)
			scratch[row] = columns[column][chain->row_order[row]];
		memcpy(columns[column],scratch,(uint64_t)rows * sizeof(uint32_t));
	}
	chain->row_ordered = 1u;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm52WaveSegmentEnd(const SparkGlm52TpChain *chain,uint32_t row)
{
	return(row + 1u == chain->wave_rows || chain->slot->host_resident_slots[chain->first_row + row] != chain->slot->host_resident_slots[chain->first_row + row + 1u] ? 1u : 0u);
}

static cudaError_t SparkGlm52CopyWaveTokens(const SparkGlm52TpChain *chain,cudaStream_t stream)
{
	uint32_t last,row,index;
	cudaError_t error;
	if ( chain->state->owns_final_head == 0u )
		return(cudaSuccess);
	if ( chain->wave.row_head_certified == 0u )
		return(cudaMemcpyAsync(chain->slot->host_output_token_ids + chain->first_row,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
	for (row=0u; chain->wave.prefill_block_table != 0 && row<chain->wave_rows; row++)
	{
		if ( SparkGlm52WaveSegmentEnd(chain,row) == 0u )
			continue;
		index = chain->first_row + row;
		error = cudaMemcpyAsync(chain->slot->host_output_token_ids + (chain->row_ordered != 0u ? chain->row_order[index] : index),chain->slot->output_token + (row + 1u == chain->wave_rows ? chain->wave.row_count - 1u : row),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		if ( error != cudaSuccess )
			return(error);
	}
	if ( chain->wave.prefill_block_table != 0 )
		return(cudaSuccess);
	last = chain->first_row + chain->wave_rows - 1u;
	return(cudaMemcpyAsync(chain->slot->host_output_token_ids + (chain->row_ordered != 0u ? chain->row_order[last] : last),chain->slot->output_token + chain->wave.row_count - 1u,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
}

static uint32_t SparkGlm52DistributionChunk(const SparkGlm52TpChain *chain,uint32_t first)
{
	uint32_t remaining = chain->wave.distribution_count - first;
	return(remaining < chain->state->distribution_chunk_rows ? remaining : chain->state->distribution_chunk_rows);
}

static cudaError_t SparkGlm52CopyDistributionLogprobs(const SparkGlm52TpChain *chain)
{
	uint64_t offset = (uint64_t)chain->wave.distribution_first * SPARK_SAMPLING_MAX_LOGPROBS;
	return(cudaMemcpyAsync(chain->slot->host_distribution_logprobs + offset,chain->slot->distribution_logprobs + offset,(uint64_t)chain->wave.distribution_count * SPARK_SAMPLING_MAX_LOGPROBS * sizeof(SparkSamplingLogprob),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream));
}

static SparkStatus SparkGlm52ValidateDistribution(const SparkGlm52ModuleState *state,const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	uint32_t entry,other,logprobs = 0u;
	if ( batch->distribution_count == 0u )
		return(SPARK_STATUS_OK);
	if ( state->distribution_wave_capacity == 0u || batch->distribution_count > batch->active_sequence_count || batch->distribution_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || batch->distribution_rows == 0 || batch->distribution_rules == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (entry=0u; entry<batch->distribution_count; entry++)
	{
		if ( batch->distribution_rows[entry] >= batch->row_count || SparkSamplingRuleValid(&batch->distribution_rules[entry]) == 0u || SparkSamplingRuleNeedsDistribution(&batch->distribution_rules[entry]) == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for (other=0u; other<entry; other++)
			if ( batch->distribution_rows[other] == batch->distribution_rows[entry] )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		logprobs |= batch->distribution_rules[entry].logprobs;
	}
	if ( logprobs != 0u && batch->distribution_logprobs == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52StageDistribution(SparkGlm52TpChain *chain,const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	SparkGlm52ExecutionSlot *slot = chain->slot;
	SparkGlm52AsyncCompletion *async = &chain->state->completions[chain->slot_index];
	uint32_t count = batch->distribution_count,*inverse = chain->row_ordered != 0u ? chain->row_order + batch->row_count : 0,entry,index,ordered,row;
	cudaError_t error;
	chain->distribution_count = count;
	async->distribution_count = count;
	async->distribution_destination = batch->distribution_logprobs;
	if ( count == 0u )
		return(SPARK_STATUS_OK);
	for (row=0u; inverse != 0 && row<batch->row_count; row++)
		inverse[chain->row_order[row]] = row;
	for (entry=0u; entry<count; entry++)
	{
		ordered = inverse != 0 ? inverse[batch->distribution_rows[entry]] : batch->distribution_rows[entry];
		for (index=entry; index>0u && chain->distribution_ordered_rows[index - 1u] > ordered; index--)
		{
			chain->distribution_ordered_rows[index] = chain->distribution_ordered_rows[index - 1u];
			async->distribution_source[index] = async->distribution_source[index - 1u];
		}
		chain->distribution_ordered_rows[index] = ordered;
		async->distribution_source[index] = entry;
	}
	for (index=0u; index<count; index++)
	{
		slot->host_distribution_rules[index] = batch->distribution_rules[async->distribution_source[index]];
		slot->host_distribution_positions[index] = slot->host_positions[chain->distribution_ordered_rows[index]];
	}
	error = cudaMemcpyAsync(slot->distribution_rules,slot->host_distribution_rules,(uint64_t)count * sizeof(SparkRowSampling),cudaMemcpyHostToDevice,(cudaStream_t)slot->stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->distribution_positions,slot->host_distribution_positions,(uint64_t)count * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->stream);
	return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"distribution_stage"));
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
	submission->active_sequence_count = chain->wave.row_count;
	submission->logical_sequence_count = chain->batch->active_sequence_count;
	submission->flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission->ordinal = atomic_fetch_add_explicit(&chain->state->tp_next_ordinal,1u,memory_order_relaxed);
	submission->local_device = device;
	submission->full_device = device;
	submission->cuda_stream = chain->slot->stream;
	submission->completion_function = host_completion != 0u ? SparkGlm52ModuleTpCompletion : 0;
	submission->completion_context = host_completion != 0u ? chain : 0;
}

static void SparkGlm52ProjectionGather(SparkGlm52TpChain *chain,uint32_t operation,SparkTpDeviceCollectiveSubmission *submission)
{
	if ( operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
		return;
	submission->local_device = chain->slot->projection_local_bf16;
	submission->row_elements = SparkGlm52ProjectionSliceWidth(chain->state->tp_degree);
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
	SparkGlm52ProjectionGather(chain,operation,&submission);
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

static SparkStatus SparkGlm52DistributionGather(SparkGlm52TpChain *chain,uint32_t first,uint32_t rows,uint32_t host_completion)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkTpDeviceCollectiveSubmission submission;
	uint32_t width = SparkGlm52DistributionWidth(state);
	if ( state->tp_degree == 1u )
	{
		if ( host_completion != 0u )
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_collective_disabled != 0u || state->tp_device_collective_initialized == 0u || rows == 0u || rows > state->distribution_chunk_rows )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	SparkGlm52ChainSubmission(chain,chain->slot->distribution_gathered_f32,host_completion,&submission);
	submission.local_device = chain->slot->distribution_logits_f32 + (uint64_t)first * width;
	submission.active_sequence_count = rows * state->distribution_sub_rows;
	submission.row_elements = 2u * width / state->distribution_sub_rows;
	return(SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER));
}

#define SPARK_GLM52_EXCHANGE_SHARD_GATHER 0u
#define SPARK_GLM52_EXCHANGE_PARTIALS 1u
#define SPARK_GLM52_EXCHANGE_KV_GATHER 2u
#define SPARK_GLM52_EXCHANGE_KV_DIGEST 3u
#define SPARK_GLM52_EXCHANGE_KV_SELECT 4u
#define SPARK_GLM52_EXCHANGE_KV_SELECTED 5u

static SparkStatus SparkGlm52ShardExchange(SparkGlm52TpChain *chain,uint32_t exchange,uint32_t chunk,uint32_t host_completion)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52ExecutionSlot *slot = chain->slot;
	const SparkGlm52CudaWave *wave = &chain->wave;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t chunk_bytes;
	uint32_t operation = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER;
	if ( state->tp_device_collective_initialized == 0u || wave->kv_shard.degree < 2u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	SparkGlm52ChainSubmission(chain,slot->shard_send_bf16,host_completion,&submission);
	switch ( exchange )
	{
	case SPARK_GLM52_EXCHANGE_SHARD_GATHER:
		submission.active_sequence_count = SparkGlm52LayerShardIndexing(wave,chain->next_layer) != 0u ? wave->shard_indexing_units : wave->shard_query_units;
		submission.local_device = slot->shard_send_bf16;
		submission.full_device = slot->shard_received_bf16;
		break;
	case SPARK_GLM52_EXCHANGE_PARTIALS:
		submission.active_sequence_count = wave->shard_partial_units;
		submission.local_device = slot->shard_partials_f32;
		submission.full_device = slot->shard_partials_received_f32;
		operation = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL;
		break;
	case SPARK_GLM52_EXCHANGE_KV_GATHER:
		if ( chunk >= wave->kv_gather_latent.chunks )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		chunk_bytes = wave->kv_gather_latent.chunk_bytes;
		submission.active_sequence_count = wave->kv_gather_latent.chunk_units;
		submission.local_device = slot->kv_gather_latent_pack + (uint64_t)chunk * chunk_bytes;
		submission.full_device = slot->kv_gather_latent_pool + (uint64_t)chunk * wave->kv_shard.degree * chunk_bytes;
		break;
	case SPARK_GLM52_EXCHANGE_KV_SELECT:
		submission.active_sequence_count = 2u * wave->kv_select_slice;
		submission.row_elements = 2u * wave->kv_select_keep;
		submission.local_device = slot->kv_select_candidates;
		submission.full_device = slot->kv_select_received;
		operation = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL;
		break;
	case SPARK_GLM52_EXCHANGE_KV_SELECTED:
		submission.active_sequence_count = wave->kv_select_slice;
		submission.row_elements = 2u * SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
		submission.local_device = slot->kv_select_merged;
		submission.full_device = slot->selected_positions;
		break;
	case SPARK_GLM52_EXCHANGE_KV_DIGEST:
		submission.active_sequence_count = 2u * SPARK_GLM52_SHARD_DIGESTS;
		submission.local_device = slot->kv_gather_digest + SPARK_GLM52_SHARD_DIGESTS;
		submission.full_device = slot->kv_gather_digest + SPARK_GLM52_SHARD_DIGESTS;
		operation = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64;
		break;
	default:
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	return(SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,operation));
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

static SparkStatus SparkGlm52ScorePlan(SparkGlm52Score *score,const SparkStageKvBinding *kv,const SparkGlm52ExecutionSlot *slot,uint32_t first,uint32_t rows,uint32_t *probe_count)
{
	const uint32_t *ids;
	uint32_t row,index,count,id_count,position;
	uint64_t key;
	count = 0u;
	for (row=0u; row<rows; row++)
	{
		position = slot->host_positions[first + row];
		key = 0u;
		score->host_flags[row] = SparkScoreDumpKeysAdvance(&score->writer.keys,slot->host_resident_slots[first + row],SparkStageKvBindingLaneSequence(kv,slot->host_resident_slots[first + row]),position,slot->host_token_ids[first + row],&key) != 0u ? SPARK_SCORE_DUMP_ROW_KEY_VALID : 0u;
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

static uint32_t SparkGlm52ScoreServedToken(const SparkGlm52TpChain *chain,uint32_t row)
{
	uint32_t index = chain->first_row + row;
	if ( chain->wave.row_head_certified != 0u && (chain->wave.prefill_block_table != 0 ? SparkGlm52WaveSegmentEnd(chain,row) == 0u : row + 1u != chain->wave_rows) )
		return(SPARK_SCORE_DUMP_NO_TOKEN);
	return(chain->slot->host_output_token_ids[chain->row_ordered != 0u ? chain->row_order[index] : index]);
}

static SparkStatus SparkGlm52ScoreWrite(SparkGlm52Score *score,const SparkGlm52TpChain *chain,uint32_t first,uint32_t rows)
{
	const SparkGlm52ExecutionSlot *slot = chain->slot;
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
		record.served_token = SparkGlm52ScoreServedToken(chain,row);
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
	status = SparkGlm52ScorePlan(score,&chain->state->kv,slot,first,rows,&count);
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
		status = SparkGlm52ScoreWrite(score,chain,first,rows);
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

static void SparkGlm52EagerShardFinish(SparkGlm52TpChain *chain,uint32_t phase);

static void SparkGlm52EagerKvLatent(SparkGlm52TpChain *chain)
{
	SparkStatus status;
	chain->shard_chunk = 0u;
	if ( SparkGlm52KvGatherOps(&chain->wave,chain->next_layer) == 0u )
	{
		SparkGlm52EagerShardFinish(chain,SPARK_GLM52_SHARD_PHASE_GATHER_POST);
		return;
	}
	chain->stage = SPARK_GLM52_CHAIN_STAGE_KV_GATHER;
	status = SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_GATHER,0u,1u);
	if ( status != SPARK_STATUS_OK )
		SparkGlm52TpChainFail(chain,status);
}

static void SparkGlm52EagerShardBegin(SparkGlm52TpChain *chain)
{
	SparkGlm52CudaWave *wave = &chain->wave;
	SparkStatus status;
	if ( SparkGlm52LaunchCudaLayerShard(wave,chain->next_layer,wave->kv_gather != 0u ? SPARK_GLM52_SHARD_PHASE_GATHER_PRE : SPARK_GLM52_SHARD_PHASE_SCATTER_PRE) != 0 )
	{
		SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
	chain->shard_chunk = 0u;
	if ( wave->kv_gather != 0u && SparkGlm52LayerShardIndexing(wave,chain->next_layer) != 0u )
	{
		chain->stage = SPARK_GLM52_CHAIN_STAGE_KV_SELECT;
		status = SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_SELECT,0u,1u);
		if ( status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,status);
		return;
	}
	if ( wave->kv_gather != 0u )
	{
		SparkGlm52EagerKvLatent(chain);
		return;
	}
	if ( wave->kv_gather != 0u && SparkGlm52KvGatherOps(wave,chain->next_layer) == 0u )
	{
		SparkGlm52EagerShardFinish(chain,SPARK_GLM52_SHARD_PHASE_GATHER_POST);
		return;
	}
	chain->stage = wave->kv_gather != 0u ? SPARK_GLM52_CHAIN_STAGE_KV_GATHER : SPARK_GLM52_CHAIN_STAGE_SHARD_EXCHANGE;
	status = SparkGlm52ShardExchange(chain,wave->kv_gather != 0u ? SPARK_GLM52_EXCHANGE_KV_GATHER : SPARK_GLM52_EXCHANGE_SHARD_GATHER,0u,1u);
	if ( status != SPARK_STATUS_OK )
		SparkGlm52TpChainFail(chain,status);
}

static void SparkGlm52EagerShardFinish(SparkGlm52TpChain *chain,uint32_t phase)
{
	SparkStatus status;
	if ( SparkGlm52LaunchCudaLayerShard(&chain->wave,chain->next_layer,phase) != 0 )
	{
		SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
	chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION;
	status = SparkGlm52ModuleReduceHidden(chain,chain->slot->attention_out_bf16);
	if ( status != SPARK_STATUS_OK )
		SparkGlm52TpChainFail(chain,status);
}

static void SparkGlm52TpChainFinishWave(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkStatus launch_status;
	launch_status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,SparkGlm52CopyWaveTokens(chain,(cudaStream_t)chain->slot->stream),"tp_head_tokens");
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
}

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
			launch_status = SparkGlm52ModuleReduce(chain,chain->slot->projection_gather_bf16,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		if ( chain->wave.kv_shard.degree > 1u )
		{
			SparkGlm52EagerShardBegin(chain);
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
	case SPARK_GLM52_CHAIN_STAGE_SHARD_EXCHANGE:
		if ( SparkGlm52LaunchCudaLayerShard(&chain->wave,chain->next_layer,SPARK_GLM52_SHARD_PHASE_SCATTER_MID) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_SHARD_PARTIALS;
		launch_status = SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_PARTIALS,0u,1u);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_SHARD_PARTIALS:
		SparkGlm52EagerShardFinish(chain,SPARK_GLM52_SHARD_PHASE_SCATTER_POST);
		return;
	case SPARK_GLM52_CHAIN_STAGE_KV_SELECT:
		if ( SparkGlm52LaunchCudaLayerShard(&chain->wave,chain->next_layer,SPARK_GLM52_SHARD_PHASE_GATHER_MERGE) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_KV_SELECTED;
		launch_status = SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_SELECTED,0u,1u);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_KV_SELECTED:
		SparkGlm52EagerKvLatent(chain);
		return;
	case SPARK_GLM52_CHAIN_STAGE_KV_GATHER:
		chain->shard_chunk++;
		if ( chain->shard_chunk < SparkGlm52KvGatherOps(&chain->wave,chain->next_layer) )
		{
			launch_status = SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_GATHER,chain->shard_chunk,1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		SparkGlm52EagerShardFinish(chain,SPARK_GLM52_SHARD_PHASE_GATHER_POST);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_PROJECTION:
		if ( chain->wave.kv_shard.degree > 1u )
		{
			SparkGlm52EagerShardBegin(chain);
			return;
		}
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
		else if ( chain->wave.kv_gather != 0u )
		{
			if ( SparkGlm52LaunchKvDigest(&chain->wave,0u) != 0 )
			{
				SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			chain->stage = SPARK_GLM52_CHAIN_STAGE_KV_DIGEST;
			launch_status = SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_DIGEST,0u,1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm52TpChainFail(chain,launch_status);
		}
		else
		{
			chain->stage = SPARK_GLM52_CHAIN_STAGE_HEAD;
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		}
		return;
	case SPARK_GLM52_CHAIN_STAGE_KV_DIGEST:
		if ( SparkGlm52LaunchKvDigest(&chain->wave,1u) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_HEAD;
		SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
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
		launch_status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"tp_head_unpack");
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		if ( chain->wave.distribution_count != 0u )
		{
			if ( SparkGlm52LaunchDistributionLogits(&chain->wave) != 0 )
			{
				SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			chain->distribution_chunk_first = 0u;
			chain->stage = SPARK_GLM52_CHAIN_STAGE_DISTRIBUTION;
			launch_status = SparkGlm52DistributionGather(chain,0u,SparkGlm52DistributionChunk(chain,0u),1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		SparkGlm52TpChainFinishWave(chain);
		return;
	case SPARK_GLM52_CHAIN_STAGE_DISTRIBUTION:
		{
			uint32_t rows = SparkGlm52DistributionChunk(chain,chain->distribution_chunk_first);
			if ( SparkGlm52LaunchDistributionSample(&chain->wave,chain->distribution_chunk_first,rows) != 0 )
			{
				SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			chain->distribution_chunk_first += rows;
		}
		if ( chain->distribution_chunk_first < chain->wave.distribution_count )
		{
			launch_status = SparkGlm52DistributionGather(chain,chain->distribution_chunk_first,SparkGlm52DistributionChunk(chain,chain->distribution_chunk_first),1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		launch_status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,SparkGlm52CopyDistributionLogprobs(chain),"distribution_logprobs");
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		SparkGlm52TpChainFinishWave(chain);
		return;
	default:
		SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
}

static void SparkGlm52FinishedAsync(void *context,SparkStatus status)
{
	SparkGlm52AsyncCompletion *async;
	SparkGlm52ModuleState *state;
	SparkGlm52ExecutionSlot *slot;
	uint32_t index;
	async = (SparkGlm52AsyncCompletion *)context;
	state = async->state;
	slot = &state->slots[async->slot_index];
	async->completion.status = status;
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->output_token_destination != 0 )
			memcpy(async->output_token_destination,slot->host_output_token_ids,(uint64_t)async->row_count * sizeof(uint32_t));
		for (index=0u; async->distribution_destination != 0 && index<async->distribution_count; index++)
			memcpy(async->distribution_destination + (uint64_t)async->distribution_source[index] * SPARK_SAMPLING_MAX_LOGPROBS,slot->host_distribution_logprobs + (uint64_t)index * SPARK_SAMPLING_MAX_LOGPROBS,SPARK_SAMPLING_MAX_LOGPROBS * sizeof(SparkSamplingLogprob));
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	atomic_fetch_add_explicit(&state->host_callback_completion_count,1u,memory_order_relaxed);
	SparkStageModuleCompleteAndReleaseClaims(async->completion_function,async->completion_context,&async->completion,state->lane_states,state->resident_sequence_capacity,async->lane_indices,async->lane_count,state->slot_states,async->slot_index);
}

static void CUDART_CB SparkGlm52CompleteAsync(void *context)
{
	SparkGlm52AsyncCompletion *async;
	SparkGlm52ModuleState *state;
	SparkStageKvBindingCompletion completion;
	SparkStatus status;
	async = (SparkGlm52AsyncCompletion *)context;
	state = async != 0 ? async->state : 0;
	if ( state == 0 || async->slot_index >= state->pipeline_slot_count )
		return;
	if ( state->slots[async->slot_index].host_kv_access_error[0] != 0u )
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	memset(&completion,0,sizeof(completion));
	completion.lane_count = async->lane_count;
	completion.extra_tokens = 0u;
	completion.status = async->completion.status;
	completion.resident_slots = async->lane_indices;
	completion.bound = async->lane_bound;
	completion.sequence_ids = async->lane_sequence_ids;
	completion.next_positions = async->lane_next_positions;
	completion.finished_function = SparkGlm52FinishedAsync;
	completion.finished_context = async;
	status = SparkStageKvBindingFinishAsync(&state->kv,async->slot_index,&completion);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"GLM52-COMPLETION-REFUSED slot=%u status=%s\n",async->slot_index,SparkStatusToString(status));
		SparkGlm52FinishedAsync(async,status);
	}
}

#define SPARK_GLM52_CHAIN_SETTLE_TIMEOUT_NS UINT64_C(35000000000)

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
	SparkGlm52ProjectionGather(chain,operation,&submission);
	return(SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,operation));
}

static uint32_t SparkGlm52WalkShard(SparkGlm52TpChain *chain,uint32_t layer)
{
	SparkGlm52CudaWave *wave = &chain->wave;
	uint32_t op;
	if ( wave->kv_gather != 0u )
	{
		if ( SparkGlm52LaunchCudaLayerShard(wave,layer,SPARK_GLM52_SHARD_PHASE_GATHER_PRE) != 0 )
			return(15u);
		if ( SparkGlm52LayerShardIndexing(wave,layer) != 0u )
		{
			if ( SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_SELECT,0u,0u) != SPARK_STATUS_OK )
				return(26u);
			if ( SparkGlm52LaunchCudaLayerShard(wave,layer,SPARK_GLM52_SHARD_PHASE_GATHER_MERGE) != 0 )
				return(27u);
			if ( SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_SELECTED,0u,0u) != SPARK_STATUS_OK )
				return(28u);
		}
		for (op=0u; op<SparkGlm52KvGatherOps(wave,layer); op++)
			if ( SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_GATHER,op,0u) != SPARK_STATUS_OK )
				return(16u);
		return(SparkGlm52LaunchCudaLayerShard(wave,layer,SPARK_GLM52_SHARD_PHASE_GATHER_POST) != 0 ? 17u : 0u);
	}
	if ( SparkGlm52LaunchCudaLayerShard(wave,layer,SPARK_GLM52_SHARD_PHASE_SCATTER_PRE) != 0 )
		return(18u);
	if ( SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_SHARD_GATHER,0u,0u) != SPARK_STATUS_OK )
		return(19u);
	if ( SparkGlm52LaunchCudaLayerShard(wave,layer,SPARK_GLM52_SHARD_PHASE_SCATTER_MID) != 0 )
		return(20u);
	if ( SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_PARTIALS,0u,0u) != SPARK_STATUS_OK )
		return(21u);
	return(SparkGlm52LaunchCudaLayerShard(wave,layer,SPARK_GLM52_SHARD_PHASE_SCATTER_POST) != 0 ? 22u : 0u);
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
			if ( SparkGlm52WalkReduce(chain,slot->projection_gather_bf16,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER) != SPARK_STATUS_OK )
				return(12u);
		}
		if ( wave->kv_shard.degree > 1u )
		{
			uint32_t site = SparkGlm52WalkShard(chain,layer);
			if ( site != 0u )
				return(site);
		}
		else if ( chain->state->projection_split != 0u )
		{
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
	if ( wave->kv_gather != 0u )
	{
		if ( SparkGlm52LaunchKvDigest(wave,0u) != 0 )
			return(23u);
		if ( SparkGlm52ShardExchange(chain,SPARK_GLM52_EXCHANGE_KV_DIGEST,0u,0u) != SPARK_STATUS_OK )
			return(24u);
		if ( SparkGlm52LaunchKvDigest(wave,1u) != 0 )
			return(25u);
	}
	if ( SparkGlm52LaunchCudaWaveHead(wave) != 0 )
		return(7u);
	if ( SparkGlm52WalkReduce(chain,slot->head_maxloc_u64,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64) != SPARK_STATUS_OK )
		return(8u);
	if ( SparkGlm52LaunchHeadMaxlocUnpack(stream,slot->head_maxloc_u64,slot->output_token,wave->row_count) != cudaSuccess )
		return(9u);
	if ( wave->distribution_count != 0u )
	{
		uint32_t first,rows;
		if ( SparkGlm52LaunchDistributionLogits(wave) != 0 )
			return(30u);
		for (first=0u; first<wave->distribution_count; first+=rows)
		{
			rows = SparkGlm52DistributionChunk(chain,first);
			if ( SparkGlm52DistributionGather(chain,first,rows,0u) != SPARK_STATUS_OK )
				return(31u);
			if ( SparkGlm52LaunchDistributionSample(wave,first,rows) != 0 )
				return(32u);
		}
		if ( SparkGlm52CopyDistributionLogprobs(chain) != cudaSuccess )
			return(33u);
	}
	if ( wave->inputs_staged == 0u && SparkGlm52CopyWaveTokens(chain,stream) != cudaSuccess )
		return(10u);
	return(0u);
}

static SparkStatus SparkGlm52GraphWave(SparkGlm52TpChain *chain,const SparkTpChainCollectives *collectives,uint32_t *site)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52ExecutionSlot *slot = chain->slot;
	void **entry;
	uint32_t regime,bound,context,rows,bucket;
	SparkStatus status = SPARK_STATUS_OK;
	rows = chain->wave_rows;
	bucket = SparkTpChainGraphBucketRows(rows);
	if ( bucket > state->execution_row_capacity )
		bucket = state->execution_row_capacity;
	context = chain->wave.maximum_context;
	if ( bucket == 0u || bucket < rows || SparkGlm52GraphReplayable(context,state->max_sequence_positions) == 0u )
	{
		fprintf(stderr,"GLM52-GRAPH-REFUSED slot=%u rows=%u bucket=%u context=%u capacity=%u\n",chain->slot_index,rows,bucket,context,state->execution_row_capacity);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	if ( SparkGlm52LaunchCudaStageWaveInputs(&chain->wave,rows,bucket) != 0 )
	{
		*site = 14u;
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	chain->wave.row_count = bucket;
	chain->wave.inputs_staged = 1u;
	regime = SparkGlm52GraphRegime(context,state->decode_split_context_threshold,state->max_sequence_positions,&bound) +
		(chain->wave.row_head_certified != 0u ? SparkGlm52GraphRegimeCount(state->max_sequence_positions) : 0u);
	SparkGlm52WaveShardSizes(state,&chain->wave,bound);
	entry = SparkTpChainGraphEntry(&state->graphs[chain->slot_index],regime,bucket);
	if ( entry == 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( *entry == 0 )
	{
		chain->wave.maximum_context = bound;
		status = SparkTpChainGraphRecord(collectives,slot->stream,SparkGlm52WalkWave,chain,entry,site);
		chain->wave.maximum_context = context;
		chain->captured++;
		state->graphs[chain->slot_index].captures++;
		fprintf(stderr,"GLM52-GRAPH-CAPTURE slot=%u rows=%u bucket=%u regime=%u bound=%u context=%u status=%s site=%u\n",chain->slot_index,rows,bucket,regime,bound,context,SparkStatusToString(status),*site);
		if ( status != SPARK_STATUS_OK )
		{
			state->graphs[chain->slot_index].failed++;
			return(status);
		}
	}
	status = SparkTpChainGraphPreLaunch(collectives,slot->stream);
	if ( status == SPARK_STATUS_OK && cudaGraphLaunch((cudaGraphExec_t)*entry,(cudaStream_t)slot->stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && SparkGlm52CopyWaveTokens(chain,(cudaStream_t)slot->stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	chain->graph = 1u;
	return(status);
}

static SparkStatus SparkGlm52GraphWalk(SparkGlm52TpChain *chain,const SparkTpChainCollectives *collectives,uint32_t *site)
{
	SparkStatus status;
	for (;;)
	{
		chain->waves++;
		status = SparkGlm52GraphWave(chain,collectives,site);
		if ( status != SPARK_STATUS_OK )
			return(status);
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

static SparkStatus SparkGlm52LinearWalk(SparkGlm52TpChain *chain,uint32_t *site)
{
	for (;;)
	{
		chain->waves++;
		*site = SparkGlm52WalkWave(chain);
		if ( *site != 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
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

static void SparkGlm52ChainFinish(SparkGlm52TpChain *chain,SparkStatus status)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkGlm52AsyncCompletion *async = &state->completions[chain->slot_index];
	uint64_t total_ns = SparkGlm52NowNs() - chain->start_ns;
	fprintf(stderr,"GLM52-CHAIN-TIME mode=%s slot=%u rows=%u waves=%u captured=%u walk_us=%.1f total_us=%.1f status=%s\n",
		chain->graph != 0u ? "graph" : "linear",chain->slot_index,async->row_count,chain->waves,chain->captured,
		(double)chain->walk_ns / 1000.0,(double)total_ns / 1000.0,SparkStatusToString(status));
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
	SparkGlm52ChainFinish(chain,status);
}

static void SparkGlm52RunChain(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state = chain->state;
	SparkTpChainCollectives collectives;
	uint32_t site = 0u;
	SparkStatus status;
	cudaError_t error;
	chain->start_ns = SparkGlm52NowNs();
	SparkGlm52ChainCollectives(state,&collectives);
	SparkGlm52BuildWave(chain);
	if ( state->chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && chain->distribution_count == 0u )
		status = SparkGlm52GraphWalk(chain,&collectives,&site);
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
	if ( value != 0u && (state->owns_embedding == 0u || state->owns_final_head == 0u) )
	{
		fprintf(stderr,"SPARK_GLM52_PREFILL_WAVE_ROWS needs a stage that owns the embedding and the final head: prefill waves run each sequence's rows together and return only its last row's token\n");
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
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
	else if ( state->chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && 2u * SparkGlm52GraphRegimeCount(state->max_sequence_positions) > SPARK_TP_CHAIN_GRAPH_MAX_REGIMES )
		reason = "max_sequence_positions needs more context-bucketed graph regimes than the graph table holds";
	else if ( state->chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && (state->owns_embedding == 0u || state->owns_final_head == 0u) )
		reason = "graph chains stage token inputs and outputs outside the graph; a pipeline stage boundary is not staged";
	else if ( state->chain_mode == SPARK_TP_CHAIN_MODE_GRAPH && state->execution_row_capacity > SPARK_TP_CHAIN_GRAPH_MAX_ROWS )
		reason = "the execution row capacity exceeds the largest graph bucket";
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

static SparkStatus SparkGlm52PrepareClaimedContinuity(void *prepare_context)
{
	SparkGlm52ClaimedContinuityContext *context = (SparkGlm52ClaimedContinuityContext *)prepare_context;
	const SparkGlm52ResidentDecodeStageBatchView *batch = context->batch;
	return(SparkStageKvBindingContinuity(&context->state->kv,context->state->lane_states,batch->row_count,batch->active_sequence_count,batch->row_resident_slots,batch->row_sequence_ids,batch->row_positions,context->bound,context->sequence_ids,context->next_positions));
}


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
	status = SparkGlm52ValidateDistribution(state,batch);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
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
	status = SparkStageKvBindingClaim(&state->kv,frame,batch->active_sequence_count,batch->row_resident_slots,batch->row_sequence_ids,batch->row_positions,simulated_next);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleSlotRelease(state->slot_states,slot_index);
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
		SPARK_RETURN(status);
	}
	SparkGlmStagePrepareAsyncCompletion(state,frame,batch,simulated_bound,simulated_sequence,simulated_next,slot_index);
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkStageKvBindingFenceExecution(&state->kv,slot->stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingUploadPageTables(&state->kv,batch->row_resident_slots,batch->active_sequence_count,slot->stream);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemsetAsync(slot->kv_access_error,0,SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),(cudaStream_t)slot->stream);
		status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"kv_access_reset");
	}
	wave_rows = status == SPARK_STATUS_OK ? SparkGlmStageRoundMajorWaveRows(state,batch,0u) : 0u;
	if ( status == SPARK_STATUS_OK && wave_rows == 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
	{
		chain = (SparkGlm52TpChain *)calloc(1u,sizeof(*chain) + ((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u && state->prefill_wave_rows != 0u ? 2u * (uint64_t)batch->row_count * sizeof(uint32_t) : 0u));
		if ( chain == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)cudaStreamSynchronize((cudaStream_t)slot->stream);
		state->completions[slot_index].completion.status = status;
		atomic_store_explicit(&state->chain_busy,0u,memory_order_release);
		SparkGlm52CompleteAsync(&state->completions[slot_index]);
		return(SPARK_STATUS_OK);
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
	if ( chain->prefill != 0u && state->prefill_wave_rows != 0u )
	{
		status = SparkGlm52OrderPrefillRows(chain);
		if ( status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,status);
			return(SPARK_STATUS_OK);
		}
	}
	status = SparkGlm52StageDistribution(chain,batch);
	if ( status != SPARK_STATUS_OK )
	{
		atomic_store_explicit(&state->chain_busy,0u,memory_order_release);
		SparkGlm52TpChainFail(chain,status);
		return(SPARK_STATUS_OK);
	}
	chain->wave_rows = SparkGlm52WaveRows(chain,0u);
	chain->next_wave_row = chain->wave_rows;
	chain->stage = SPARK_GLM52_CHAIN_STAGE_BEGIN;
	chain->active = 1u;
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
	if ( state != 0 && frame != 0 && (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH) != 0u )
	{
		status = frame->execution_stream == state->execution_stream ? SparkStageKvBindingPublishFrame(&state->kv,frame,state->lane_states) : SPARK_STATUS_INVALID_ARGUMENT;
		atomic_fetch_add_explicit(status == SPARK_STATUS_OK ? &state->completed_count : &state->rejected_count,1u,memory_order_relaxed);
		SPARK_RETURN(status);
	}
	status = SparkGlmStageValidateFrame(state,frame,&context);
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
		return(SparkStageKvBindingAdmitReset(&state->kv,request,decision,state->slot_states,state->lane_states,state->execution_stream));
	available = SparkStageModuleSlotCountFree(state->slot_states,state->pipeline_slot_count);
	if ( (request->frame_flags & (SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE | SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH)) != 0u )
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
	uint32_t resident_count;
	state = (SparkGlm52ModuleState *)module_state;
	snapshot->host_callback_completion_count = atomic_load_explicit(&state->host_callback_completion_count,memory_order_relaxed);
	resident_count = SparkStageKvBindingResidentCount(&state->kv);
	snapshot->resident_sequence_count = resident_count;
	snapshot->kv_token_capacity = (uint64_t)state->resident_sequence_capacity * state->max_sequence_positions;
	SparkStageKvBindingKvStoreCounters(&state->kv,&snapshot->kv_store);
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
	{
		SparkStatus status = SparkStageKvBindingQuiesce(&state->kv,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
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
	SparkStageKvBindingDestroy(&state->kv);
	SparkGlm52ReleaseSlotHost(state);
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
		status = SparkGlm52DistributionConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52AllocateCaches(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlmStageAllocateSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ModuleInitializeTpCollective(state,(const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context);
	if ( status == SPARK_STATUS_OK && state->kv_shard.degree > 1u && (state->tp_device_collective_initialized == 0u || SparkTpDeviceCollectiveAllToAllSupported(&state->tp_device_collective) == 0u) )
	{
		fprintf(stderr,"GLM52-KV-SHARD-REFUSED tp=%u rank=%u: the sharded KV exchange needs the TP collective with all-to-all (hardware waits and a weightd that advertises slice routes)\n",state->tp_degree,state->tp_rank);
		status = SPARK_STATUS_UNSUPPORTED;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlmStageBuildHeadShadow(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ChainModeConfigure(state);
#ifdef SPARK_SCORE_DUMP
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ScoreOpen(state,(const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context);
#endif
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states,state->resident_sequence_capacity);
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
