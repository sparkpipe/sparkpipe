#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_dsv41_flash_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"

#include "spark_dsv41_flash_stagepack_format.h"

#define SPARK_DSV41_FLASH_MODULE_TAG "dsv41_flash_stage"

#ifndef DSV41_FLASH_EXPERT_WEIGHT_CODEC
#define DSV41_FLASH_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_MXFP4_E2M1
#endif

#ifndef SPARK_BATCH_BUCKET
#define SPARK_BATCH_BUCKET 1024u
#endif

typedef struct SparkDsv41FlashManifestContext
{
	const SparkDsv41FlashStagePackEntry *entries;
	uint32_t count;
} SparkDsv41FlashManifestContext;

typedef struct SparkDsv41FlashModuleState
{
	SparkStageModuleLedger ledger;
	SparkWeightdLazyPack *lazy_pack;
	char model_revision[SPARK_DSV41_FLASH_STAGEPACK_MODEL_REVISION_BYTES];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t expert_weight_codec;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t execution_row_capacity;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t pack_flags;
	uint32_t pack_tensor_count;
	atomic_uint slot_states[SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint64_t *layer_seen_bits;
	uint64_t global_seen_bits;
} SparkDsv41FlashModuleState;

static SparkStatus SparkDsv41FlashPackFileSize(FILE *file,uint64_t *bytes)
{
	off_t end;
	if ( file == 0 || bytes == 0 || fseeko(file,0,SEEK_END) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	end = ftello(file);
	if ( end < 0 || fseeko(file,0,SEEK_SET) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*bytes = (uint64_t)end;
	return(SPARK_STATUS_OK);
}

static void SparkDsv41FlashMarkSeen(
	SparkDsv41FlashModuleState *state,
	const SparkDsv41FlashStagePackEntry *entry)
{
	if ( entry->tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT )
		return;
	if ( SparkDsv41FlashStagePackKindIsGlobal(entry->tensor_kind) != 0u )
	{
		state->global_seen_bits |= UINT64_C(1) << entry->tensor_kind;
		return;
	}
	if ( entry->layer_index >= state->first_layer_index &&
		entry->layer_index < state->first_layer_index + state->layer_count )
		state->layer_seen_bits[entry->layer_index - state->first_layer_index] |=
			UINT64_C(1) << entry->tensor_kind;
}

static SparkStatus SparkDsv41FlashEntryValidate(
	SparkDsv41FlashModuleState *state,
	const SparkDsv41FlashStagePackHeader *header,
	const SparkDsv41FlashStagePackEntry *entry,
	uint64_t file_bytes)
{
	SparkDsv41FlashStagePackTensorShape shape;
	uint64_t directory_end;
	if ( entry->tensor_kind >= SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->layer_index != SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER &&
		entry->layer_index >= SPARK_DSV41_FLASH_MODEL_LAYER_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkDsv41FlashStagePackExpectedShape(entry->tensor_kind,state->expert_weight_codec,state->tp_degree,&shape) == 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->payload_type != shape.payload_type || entry->weight_codec != shape.weight_codec ||
		entry->scale_encoding != shape.scale_encoding || entry->group_count != shape.group_count ||
		entry->rows != shape.rows || entry->columns != shape.columns )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_end = header->directory_offset + (uint64_t)header->tensor_count * header->directory_entry_bytes;
	if ( entry->payload_offset < directory_end || entry->payload_bytes == 0u ||
		entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->scale_bytes != 0u && (entry->scale_offset < directory_end ||
		entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkDsv41FlashStagePackKindIsExpert(entry->tensor_kind) != 0u &&
		entry->weight_codec != state->expert_weight_codec )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkDsv41FlashStagePackKindIsGlobal(entry->tensor_kind) == 0u &&
		SparkDsv41FlashStagePackKindInLayer(entry->tensor_kind,entry->layer_index) == 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	SparkDsv41FlashMarkSeen(state,entry);
	return(SPARK_STATUS_OK);
}

static int SparkDsv41FlashHexNibble(char c)
{
	if ( c >= '0' && c <= '9' )
		return(c - '0');
	if ( c >= 'a' && c <= 'f' )
		return(c - 'a' + 10);
	if ( c >= 'A' && c <= 'F' )
		return(c - 'A' + 10);
	return(-1);
}

static SparkStatus SparkDsv41FlashContractShaValidate(const uint8_t *pack_sha)
{
	uint8_t expected[SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES];
	uint32_t index;
	int high,low;
	if ( strlen(SPARK_DSV41_FLASH_CONTRACT_SHA256) != 2u * SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (index=0u; index<SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES; index++)
	{
		high = SparkDsv41FlashHexNibble(SPARK_DSV41_FLASH_CONTRACT_SHA256[2u * index]);
		low = SparkDsv41FlashHexNibble(SPARK_DSV41_FLASH_CONTRACT_SHA256[2u * index + 1u]);
		if ( high < 0 || low < 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		expected[index] = (uint8_t)((high << 4) | low);
	}
	if ( memcmp(pack_sha,expected,SPARK_DSV41_FLASH_STAGEPACK_SHA256_BYTES) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashHeaderValidate(
	const SparkDsv41FlashModuleState *state,
	const SparkDsv41FlashStagePackHeader *header,
	uint64_t file_bytes)
{
	if ( header->magic != SPARK_DSV41_FLASH_STAGEPACK_MAGIC ||
		header->format_version != SPARK_DSV41_FLASH_STAGEPACK_FORMAT_VERSION ||
		header->header_bytes != SPARK_DSV41_FLASH_STAGEPACK_HEADER_BYTES ||
		header->directory_entry_bytes != SPARK_DSV41_FLASH_STAGEPACK_ENTRY_BYTES ||
		header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( (header->flags & ~SPARK_DSV41_FLASH_STAGEPACK_KNOWN_FLAGS) != 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_DSV41_FLASH_STAGEPACK_MAX_TENSOR_COUNT ||
		header->total_layer_count != SPARK_DSV41_FLASH_MODEL_LAYER_COUNT ||
		header->hidden_dimension != SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION ||
		header->vocab_count != SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT ||
		header->routed_expert_count != SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT ||
		header->stage_count != state->stage_count || header->stage_index != state->stage_index ||
		header->first_layer_index != state->first_layer_index ||
		header->layer_count != state->layer_count || header->tp_degree != state->tp_degree ||
		header->tp_rank != state->tp_rank )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes ||
		header->directory_offset % SPARK_DSV41_FLASH_STAGEPACK_ALIGNMENT_BYTES != 0u ||
		header->tensor_count > UINT64_MAX / header->directory_entry_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( strcmp(header->model_revision,state->model_revision) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SparkDsv41FlashContractShaValidate(header->contract_sha256));
}

static SparkStatus SparkDsv41FlashInventoryValidate(
	const SparkDsv41FlashModuleState *state)
{
	SparkDsv41FlashStagePackTensorShape shape;
	uint64_t expected_bits,expected_global;
	uint32_t kind,layer;
	for (layer=0u; layer<state->layer_count; layer++)
	{
		expected_bits = 0u;
		for (kind=0u; kind<SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		{
			if ( SparkDsv41FlashStagePackKindIsGlobal(kind) != 0u )
				continue;
			if ( SparkDsv41FlashStagePackExpectedShape(kind,state->expert_weight_codec,state->tp_degree,&shape) == 0u )
				continue;
			if ( SparkDsv41FlashStagePackKindInLayer(kind,state->first_layer_index + layer) != 0u )
				expected_bits |= UINT64_C(1) << kind;
		}
		if ( state->layer_seen_bits[layer] != expected_bits )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	expected_global = 0u;
	if ( state->owns_embedding != 0u )
		expected_global |= UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		expected_global |= (UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM) |
			(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD);
	if ( state->global_seen_bits != expected_global )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashManifestExpertPlane(
	const SparkDsv41FlashStagePackEntry *entry,
	uint32_t plane)
{
	uint64_t bytes;
	bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
	if ( bytes == 0u || (bytes % entry->group_count) != 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( bytes / entry->group_count > SPARK_WEIGHTD_EXPERT_BYTES_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashManifestCheck(
	const SparkWeightdManifest *manifest,
	void *opaque)
{
	SparkDsv41FlashManifestContext *context;
	const SparkDsv41FlashStagePackEntry *entry;
	uint64_t expected;
	uint32_t index,plane;
	context = (SparkDsv41FlashManifestContext *)opaque;
	expected = 0u;
	for (index=0u; index<context->count; index++)
	{
		entry = &context->entries[index];
		if ( SparkDsv41FlashStagePackKindIsExpert(entry->tensor_kind) == 0u )
			continue;
		if ( entry->weight_codec != SPARK_WEIGHT_CODEC_MXFP4_E2M1 && entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		for (plane=0u; plane<2u; plane++)
		{
			if ( plane != 0u && entry->scale_bytes == 0u )
				continue;
			if ( SparkDsv41FlashManifestExpertPlane(entry,plane) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			expected += entry->group_count;
		}
	}
	return(expected == manifest->range_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SparkDsv41FlashLazyOpen(
	SparkDsv41FlashModuleState *state,
	const char *path,
	uint64_t bytes,
	const SparkDsv41FlashStagePackEntry *entries,
	uint32_t count)
{
	SparkWeightdLazyAttachRequest request;
	SparkDsv41FlashManifestContext context;
	const char *digest;
	uint64_t spine_budget;
	SparkStatus status;
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
		return(status == SPARK_STATUS_BUSY ? SPARK_STATUS_UNSUPPORTED : status);
	memset(&request,0,sizeof(request));
	digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	if ( digest == 0 || strlen(digest) != 64u || strlen(path) >= sizeof(request.pack_path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.identity.pack_sha256,digest,65u);
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s",SPARK_DSV41_FLASH_MODULE_TAG);
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",state->model_revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = bytes;
	request.identity.topology = state->tp_degree;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	status = SparkStageModuleEnvironmentUnsigned64(SPARK_DSV41_FLASH_MODULE_TAG,"SPARK_WEIGHTD_EXPERT_POOL_BYTES",1u,UINT64_MAX,&request.expert_pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_DSV41_FLASH_MODULE_TAG,"SPARK_WEIGHTD_SPINE_BUDGET_BYTES",1u,UINT64_MAX,&spine_budget);
	if ( status == SPARK_STATUS_OK )
	{
		context.entries = entries;
		context.count = count;
		status = SparkWeightdLazyPackCreateChecked(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,spine_budget,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,SparkDsv41FlashManifestCheck,&context,&state->lazy_pack);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkDsv41FlashPackLoad(
	SparkDsv41FlashModuleState *state,
	const char *path)
{
	SparkDsv41FlashStagePackHeader header;
	SparkDsv41FlashStagePackEntry *entries;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memset(&header,0,sizeof(header));
	entries = 0;
	status = SparkDsv41FlashPackFileSize(file,&file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_DSV41_FLASH_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashHeaderValidate(state,&header,file_bytes);
	if ( status == SPARK_STATUS_OK )
	{
		entries = (SparkDsv41FlashStagePackEntry *)malloc((size_t)header.tensor_count * sizeof(*entries));
		if ( entries == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_DSV41_FLASH_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(*entries));
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
		status = SparkDsv41FlashEntryValidate(state,&header,&entries[index],file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashInventoryValidate(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashLazyOpen(state,path,file_bytes,entries,header.tensor_count);
	free(entries);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

static SparkStatus SparkDsv41FlashStatePrepare(
	SparkDsv41FlashModuleState *state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path)
{
	const SparkDsv41FlashResidentDecodeStageNodeContext *context;
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 ||
		host_services->node_context == 0 || host_services->execution_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv41FlashResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes < sizeof(*context) )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( context->stage_count == 0u || context->stage_index >= context->stage_count ||
		context->layer_count == 0u ||
		context->first_layer_index + context->layer_count > SPARK_DSV41_FLASH_MODEL_LAYER_COUNT ||
		context->expert_weight_codec != DSV41_FLASH_EXPERT_WEIGHT_CODEC ||
		context->resident_sequence_capacity == 0u ||
		context->resident_sequence_capacity > SPARK_BATCH_BUCKET ||
		context->pipeline_slot_count == 0u ||
		context->pipeline_slot_count > SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT ||
		context->max_sequence_positions == 0u ||
		context->max_sequence_positions > SPARK_DSV41_FLASH_MODEL_MAXIMUM_CONTEXT_TOKENS ||
		context->execution_row_capacity == 0u ||
		context->execution_row_capacity > SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		context->tp_degree == 0u || context->tp_rank >= context->tp_degree ||
		(SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT % context->tp_degree) != 0u ||
		(SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree) != 0u ||
		(context->flags & ~SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u ||
		context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' ||
		context->model_revision == 0 || context->model_revision[0] == '\0' ||
		strlen(context->model_revision) >= sizeof(state->model_revision) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->stage_count = context->stage_count;
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->expert_weight_codec = context->expert_weight_codec;
	state->resident_sequence_capacity = context->resident_sequence_capacity;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->max_sequence_positions = context->max_sequence_positions;
	state->execution_row_capacity = context->execution_row_capacity;
	state->tp_degree = context->tp_degree;
	state->tp_rank = context->tp_rank;
	state->owns_embedding = context->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = context->stage_index + 1u == context->stage_count ? 1u : 0u;
	(void)snprintf(state->model_revision,sizeof(state->model_revision),"%s",context->model_revision);
	*pack_path = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

void SparkDsv41FlashResidentDecodeStageDestroy(void *module_state)
{
	SparkDsv41FlashModuleState *state;
	if ( module_state == 0 )
		return;
	state = (SparkDsv41FlashModuleState *)module_state;
	if ( state->lazy_pack != 0 )
		(void)SparkWeightdLazyPackDestroy(state->lazy_pack);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state->layer_seen_bits);
	free(state);
}

SparkStatus SparkDsv41FlashResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkDsv41FlashModuleState *state;
	const char *pack_path;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state = (SparkDsv41FlashModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->layer_seen_bits = (uint64_t *)calloc(SPARK_DSV41_FLASH_MODEL_LAYER_COUNT,sizeof(uint64_t));
	if ( state->layer_seen_bits == 0 )
	{
		free(state);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	pack_path = 0;
	status = SparkDsv41FlashStatePrepare(state,configuration,host_services,&pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashCudaContextEnsure();
	if ( status == SPARK_STATUS_OK )
	{
		SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
		status = SparkDsv41FlashPackLoad(state,pack_path);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkDsv41FlashResidentDecodeStageDestroy(state);
		SPARK_RETURN(status);
	}
	*module_state = state;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv41FlashResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	(void)module_state;
	(void)frame;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkDsv41FlashResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)module_state;
	(void)request;
	if ( decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageModuleAdmissionDecisionInitialize(decision,0u);
	SparkStageModuleAdmissionDecisionReject(decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv41FlashResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkDsv41FlashModuleState *state;
	if ( module_state == 0 || snapshot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkDsv41FlashModuleState *)module_state;
	SparkStageModuleRuntimeSnapshotInitialize(snapshot,program_id,state->slot_states,state->pipeline_slot_count);
	return(SPARK_STATUS_OK);
}
