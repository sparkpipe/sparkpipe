/* Shared stage-pack load spine (DRY wave: the pack-loading paste).
 *
 * The qwen38_27b / qwen38_max / qwen4_flash resident-decode-stage modules
 * carried one pasted pack-loading spine: ordinal build, linear-view fill,
 * entry range validation, the load loop with duplicate detection, coverage
 * verification, and the pack frame (open, header, geometry, directory, load,
 * verify) were byte-identical modulo the family prefix; a prefix-normalized
 * diff measured BuildOrdinals/FillLinearView/LoadEntry identical across the
 * three. This header is that spine, parameterized by family macros the
 * includer defines first (the spark_work_control_common.h pattern):
 *
 *   SPARK_PACK_LOAD_FN(name)              module function names
 *   SPARK_PACK_LOAD_TYPE(name)            ModuleState/StagePackEntry/
 *                                         StagePackHeader/LinearView structs
 *   SPARK_PACK_LOAD_CONST(name)           family constants
 *   SPARK_PACK_LOAD_LAYER_IS_GDN(layer)   hybrid layer-map predicate
 *   SPARK_PACK_LOAD_SEEN_TYPE             coverage word type (uint32/64)
 *   SPARK_PACK_LOAD_SEEN_ONE              coverage bit one (1u/1ull)
 *   SPARK_PACK_LOAD_SEEN_FORMAT           coverage word printf format
 *   SPARK_PACK_LOAD_SEEN_ARG(value)       coverage word printf argument
 *   SPARK_PACK_LOAD_BYTES_MATCH(entry)    declared payload/scale byte check
 *   SPARK_PACK_LOAD_EXPECT_GEOMETRY(state, expected)
 *   SPARK_PACK_LOAD_GEOMETRY_MISMATCH(state, header, expected)
 *   SPARK_PACK_LOAD_LOG_GEOMETRY_MISMATCH(state, header, expected)
 *   SPARK_PACK_LOAD_PREFLIGHT(state, file, header, status)
 *
 * Every macro is mandatory: a family that omits one fails the compile. The
 * per-kind tensor inventory stays family-side as data: the family defines
 * the shape/format rules (its ValidateEntry head over its stagepack shape
 * algebra), the per-kind binds, and the three expected-coverage bit sets,
 * all declared here and defined after the include.
 */
#ifndef SPARKPIPE_SPARK_PACK_LOAD_COMMON_H
#define SPARKPIPE_SPARK_PACK_LOAD_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef SPARK_PACK_LOAD_FN
#error "SPARK_PACK_LOAD_FN must build the family module function names"
#endif
#ifndef SPARK_PACK_LOAD_TYPE
#error "SPARK_PACK_LOAD_TYPE must build the family struct names"
#endif
#ifndef SPARK_PACK_LOAD_CONST
#error "SPARK_PACK_LOAD_CONST must build the family constant names"
#endif
#ifndef SPARK_PACK_LOAD_LAYER_IS_GDN
#error "SPARK_PACK_LOAD_LAYER_IS_GDN must name the family hybrid layer-map predicate"
#endif
#ifndef SPARK_PACK_LOAD_SEEN_TYPE
#error "SPARK_PACK_LOAD_SEEN_TYPE must name the family coverage word type"
#endif
#ifndef SPARK_PACK_LOAD_SEEN_ONE
#error "SPARK_PACK_LOAD_SEEN_ONE must be the coverage bit one for SEEN_TYPE"
#endif
#ifndef SPARK_PACK_LOAD_SEEN_FORMAT
#error "SPARK_PACK_LOAD_SEEN_FORMAT must be the coverage word printf format"
#endif
#ifndef SPARK_PACK_LOAD_SEEN_ARG
#error "SPARK_PACK_LOAD_SEEN_ARG must convert a coverage word for SEEN_FORMAT"
#endif
#ifndef SPARK_PACK_LOAD_BYTES_MATCH
#error "SPARK_PACK_LOAD_BYTES_MATCH must check an entry's declared payload/scale bytes"
#endif
#ifndef SPARK_PACK_LOAD_EXPECT_GEOMETRY
#error "SPARK_PACK_LOAD_EXPECT_GEOMETRY must fill the expected pack header"
#endif
#ifndef SPARK_PACK_LOAD_GEOMETRY_MISMATCH
#error "SPARK_PACK_LOAD_GEOMETRY_MISMATCH must compare file header against expected"
#endif
#ifndef SPARK_PACK_LOAD_LOG_GEOMETRY_MISMATCH
#error "SPARK_PACK_LOAD_LOG_GEOMETRY_MISMATCH must report the offending geometry"
#endif
#ifndef SPARK_PACK_LOAD_PREFLIGHT
#error "SPARK_PACK_LOAD_PREFLIGHT must state the family pre-directory policy (empty where none)"
#endif

static SparkStatus SPARK_PACK_LOAD_FN(ValidateEntry)(
	SPARK_PACK_LOAD_TYPE(ModuleState) *state,
	const SPARK_PACK_LOAD_TYPE(StagePackEntry) *entry,
	uint64_t file_bytes,
	uint32_t *is_global);
static SparkStatus SPARK_PACK_LOAD_FN(BindMtp)(
	SPARK_PACK_LOAD_TYPE(ModuleState) *state,
	const SPARK_PACK_LOAD_TYPE(StagePackEntry) *entry,
	void *payload,
	void *scale);
static SparkStatus SPARK_PACK_LOAD_FN(BindGlobal)(
	SPARK_PACK_LOAD_TYPE(ModuleState) *state,
	const SPARK_PACK_LOAD_TYPE(StagePackEntry) *entry,
	void *payload);
static SparkStatus SPARK_PACK_LOAD_FN(BindLayer)(
	SPARK_PACK_LOAD_TYPE(ModuleState) *state,
	const SPARK_PACK_LOAD_TYPE(StagePackEntry) *entry,
	void *payload,
	void *scale);
static SPARK_PACK_LOAD_SEEN_TYPE SPARK_PACK_LOAD_FN(ExpectedGlobalBits)(
	const SPARK_PACK_LOAD_TYPE(ModuleState) *state);
static SPARK_PACK_LOAD_SEEN_TYPE SPARK_PACK_LOAD_FN(ExpectedMtpBits)(void);
static SPARK_PACK_LOAD_SEEN_TYPE SPARK_PACK_LOAD_FN(ExpectedLayerBits)(
	const SPARK_PACK_LOAD_TYPE(ModuleState) *state,
	uint32_t layer);

static void SPARK_PACK_LOAD_FN(BuildOrdinals)(SPARK_PACK_LOAD_TYPE(ModuleState) *state)
{
	uint32_t layer;
	for (layer = 0; layer < SPARK_PACK_LOAD_CONST(RESIDENT_DECODE_STAGE_LAYER_COUNT); layer++)
	{
		state->gdn_ordinal_by_layer[layer] = UINT32_MAX;
		state->attn_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		if ( SPARK_PACK_LOAD_LAYER_IS_GDN(layer) != 0u )
			state->gdn_ordinal_by_layer[layer] = state->gdn_layer_count++;
		else
			state->attn_ordinal_by_layer[layer] = state->attn_layer_count++;
	}
}

static void SPARK_PACK_LOAD_FN(FillLinearView)(SPARK_PACK_LOAD_TYPE(LinearView) *view, const SPARK_PACK_LOAD_TYPE(StagePackEntry) *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_PACK_LOAD_CONST(RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION);
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SPARK_PACK_LOAD_FN(ValidateEntryPlacement)(SPARK_PACK_LOAD_TYPE(ModuleState) *state, const SPARK_PACK_LOAD_TYPE(StagePackEntry) *entry, uint64_t file_bytes, uint32_t *is_global)
{
	uint32_t global = entry->layer_index == SPARK_PACK_LOAD_CONST(STAGEPACK_GLOBAL_LAYER) ? 1u : 0u;
	if ( !SPARK_PACK_LOAD_BYTES_MATCH(entry) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->scale_bytes != 0u && (entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->layer_index == SPARK_PACK_LOAD_CONST(STAGEPACK_MTP_LAYER) || (global != 0u && (entry->tensor_kind >= SPARK_PACK_LOAD_CONST(STAGEPACK_TENSOR_MTP_FC) && entry->tensor_kind <= SPARK_PACK_LOAD_CONST(STAGEPACK_TENSOR_MTP_FINAL_NORM))) )
	{
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( global == 0u && (entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_PACK_LOAD_FN(LoadEntry)(SPARK_PACK_LOAD_TYPE(ModuleState) *state, FILE *file, const SPARK_PACK_LOAD_TYPE(StagePackEntry) *entry, uint64_t file_bytes)
{
	SparkStatus status;
	uint32_t is_global = 0u;
	SPARK_PACK_LOAD_SEEN_TYPE bit = SPARK_PACK_LOAD_SEEN_ONE << entry->tensor_kind;
	SPARK_PACK_LOAD_SEEN_TYPE *seen;
	void *payload = 0,*scale = 0;
	status = SPARK_PACK_LOAD_FN(ValidateEntry)(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_PACK_LOAD_CONST(MODULE_TAG),entry->tensor_kind,entry->layer_index);
		return(status);
	}
	if ( entry->layer_index == SPARK_PACK_LOAD_CONST(STAGEPACK_MTP_LAYER) || (is_global != 0u && entry->tensor_kind >= SPARK_PACK_LOAD_CONST(STAGEPACK_TENSOR_MTP_FC) && entry->tensor_kind <= SPARK_PACK_LOAD_CONST(STAGEPACK_TENSOR_MTP_FINAL_NORM)) )
		seen = &state->mtp_seen_bits;
	else
		seen = is_global != 0u ? &state->global_seen_bits : &state->layer_seen_bits[entry->layer_index];
	if ( (*seen & bit) != 0u )
	{
		fprintf(stderr,"%s pack_entry_duplicate kind=%u layer=%u\n",SPARK_PACK_LOAD_CONST(MODULE_TAG),entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	*seen |= bit;
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( entry->layer_index == SPARK_PACK_LOAD_CONST(STAGEPACK_MTP_LAYER) || entry->tensor_kind == SPARK_PACK_LOAD_CONST(STAGEPACK_TENSOR_MTP_FC) )
		return(SPARK_PACK_LOAD_FN(BindMtp)(state,entry,payload,scale));
	return(is_global != 0u ? SPARK_PACK_LOAD_FN(BindGlobal)(state,entry,payload) : SPARK_PACK_LOAD_FN(BindLayer)(state,entry,payload,scale));
}

static SparkStatus SPARK_PACK_LOAD_FN(VerifyCoverage)(SPARK_PACK_LOAD_TYPE(ModuleState) *state)
{
	uint32_t layer;
	SPARK_PACK_LOAD_SEEN_TYPE expected_layer;
	if ( state->owns_final_head != 0u && state->mtp_seen_bits != SPARK_PACK_LOAD_FN(ExpectedMtpBits)() )
	{
		fprintf(stderr,"%s pack_mtp_incomplete seen=" SPARK_PACK_LOAD_SEEN_FORMAT " expected=" SPARK_PACK_LOAD_SEEN_FORMAT "\n",SPARK_PACK_LOAD_CONST(MODULE_TAG),SPARK_PACK_LOAD_SEEN_ARG(state->mtp_seen_bits),SPARK_PACK_LOAD_SEEN_ARG(SPARK_PACK_LOAD_FN(ExpectedMtpBits)()));
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( state->global_seen_bits != SPARK_PACK_LOAD_FN(ExpectedGlobalBits)(state) )
	{
		fprintf(stderr,"%s pack_globals_incomplete seen=" SPARK_PACK_LOAD_SEEN_FORMAT " expected=" SPARK_PACK_LOAD_SEEN_FORMAT "\n",SPARK_PACK_LOAD_CONST(MODULE_TAG),SPARK_PACK_LOAD_SEEN_ARG(state->global_seen_bits),SPARK_PACK_LOAD_SEEN_ARG(SPARK_PACK_LOAD_FN(ExpectedGlobalBits)(state)));
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		expected_layer = SPARK_PACK_LOAD_FN(ExpectedLayerBits)(state,layer);
		if ( state->layer_seen_bits[layer] != expected_layer )
		{
			fprintf(stderr,"%s pack_layer_incomplete layer=%u seen=" SPARK_PACK_LOAD_SEEN_FORMAT " expected=" SPARK_PACK_LOAD_SEEN_FORMAT "\n",SPARK_PACK_LOAD_CONST(MODULE_TAG),layer,SPARK_PACK_LOAD_SEEN_ARG(state->layer_seen_bits[layer]),SPARK_PACK_LOAD_SEEN_ARG(expected_layer));
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_PACK_LOAD_FN(LoadPack)(SPARK_PACK_LOAD_TYPE(ModuleState) *state, const char *path)
{
	SPARK_PACK_LOAD_TYPE(StagePackHeader) header,expected;
	SPARK_PACK_LOAD_TYPE(StagePackEntry) *directory;
	FILE *file;
	SparkStatus status;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_PACK_LOAD_CONST(MODULE_TAG),path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_PACK_LOAD_CONST(MODULE_TAG),file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SPARK_PACK_LOAD_EXPECT_GEOMETRY(state,&expected);
		if ( SPARK_PACK_LOAD_GEOMETRY_MISMATCH(state,&header,&expected) )
		{
			SPARK_PACK_LOAD_LOG_GEOMETRY_MISMATCH(state,&header,&expected);
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	SPARK_PACK_LOAD_PREFLIGHT(state,file,&header,status);
	directory = status == SPARK_STATUS_OK ? (SPARK_PACK_LOAD_TYPE(StagePackEntry) *)malloc((size_t)header.tensor_count * sizeof(SPARK_PACK_LOAD_TYPE(StagePackEntry))) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_PACK_LOAD_CONST(MODULE_TAG),file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SPARK_PACK_LOAD_TYPE(StagePackEntry)));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SPARK_PACK_LOAD_FN(LoadEntry)(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SPARK_PACK_LOAD_FN(VerifyCoverage)(state);
	free(directory);
	fclose(file);
	return(status);
}

#endif /* SPARKPIPE_SPARK_PACK_LOAD_COMMON_H */
