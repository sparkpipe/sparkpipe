#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include "../modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_resident_decode_stage_module.c"

#define TWIN_CHECK(condition,message) \
	do { \
		if ( !(condition) ) \
		{ \
			(void)fprintf(stderr,"FAIL %s at line %d: %s\n",message,(int)__LINE__,#condition); \
			return(2); \
		} \
	} while (0)

SparkStatus SparkDsv41FlashCudaContextEnsure(void)
{
	return(SPARK_STATUS_OK);
}

static int TwinReadHeaderDirectory(const char *path,SparkDsv41FlashStagePackHeader *header,SparkDsv41FlashStagePackEntry **directory)
{
	FILE *file;
	file = fopen(path,"rb");
	if ( file == 0 )
		return(-1);
	if ( fread(header,sizeof(*header),1u,file) != 1u )
	{
		(void)fclose(file);
		return(-2);
	}
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_DSV41_FLASH_STAGEPACK_MAX_TENSOR_COUNT )
	{
		(void)fclose(file);
		return(-3);
	}
	*directory = (SparkDsv41FlashStagePackEntry *)calloc(header->tensor_count,sizeof(**directory));
	if ( *directory == 0 )
	{
		(void)fclose(file);
		return(-4);
	}
	if ( fseeko(file,(off_t)header->directory_offset,SEEK_SET) != 0 ||
	     fread(*directory,sizeof(**directory),header->tensor_count,file) != header->tensor_count )
	{
		free(*directory);
		*directory = 0;
		(void)fclose(file);
		return(-5);
	}
	(void)fclose(file);
	return(0);
}

static int TwinStatePrepare(SparkDsv41FlashModuleState *state,const SparkDsv41FlashStagePackHeader *header)
{
	memset(state,0,sizeof(*state));
	state->stage_count = header->stage_count;
	state->stage_index = header->stage_index;
	state->first_layer_index = header->first_layer_index;
	state->layer_count = header->layer_count;
	state->expert_weight_codec = DSV41_FLASH_EXPERT_WEIGHT_CODEC;
	state->resident_sequence_capacity = 1u;
	state->pipeline_slot_count = 1u;
	state->max_sequence_positions = 1024u;
	state->execution_row_capacity = 1u;
	state->tp_degree = header->tp_degree;
	state->tp_rank = header->tp_rank;
	state->owns_embedding = header->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = header->stage_index + 1u == header->stage_count ? 1u : 0u;
	state->layer_seen_bits = (uint64_t *)calloc(SPARK_DSV41_FLASH_MODEL_LAYER_COUNT,sizeof(uint64_t));
	return(state->layer_seen_bits != 0 ? 0 : -1);
}

int main(int argc,char **argv)
{
	SparkDsv41FlashModuleState state;
	SparkDsv41FlashStagePackHeader header;
	SparkDsv41FlashStagePackEntry *directory;
	SparkWeightdManifest manifest;
	SparkDsv41FlashManifestContext context;
	SparkStatus status;
	uint64_t file_bytes,expected_ranges;
	uint32_t index,is_global;
	const char *mode;
	if ( argc != 3 )
	{
		(void)fprintf(stderr,"usage: test_dsv41_pack_validate PACK MODE\n"
			"  modes: accept reject-entry reject-codec\n");
		return(2);
	}
	mode = argv[2];
	if ( TwinReadHeaderDirectory(argv[1],&header,&directory) != 0 )
	{
		(void)fprintf(stderr,"FAIL cannot read pack %s\n",argv[1]);
		return(2);
	}
	file_bytes = header.file_bytes;
	TWIN_CHECK(TwinStatePrepare(&state,&header) == 0,"twin state prepared");
	if ( strcmp(mode,"reject-entry") == 0 )
	{
		uint32_t rejected = 0u;
		for (index=0u; index<header.tensor_count; index++)
		{
			if ( SparkDsv41FlashStagePackKindIsExpert(directory[index].tensor_kind) != 0u &&
				directory[index].weight_codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 )
				directory[index].weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1;
			status = SparkDsv41FlashEntryValidate(&state,&header,&directory[index],file_bytes);
			if ( status != SPARK_STATUS_OK )
				rejected += 1u;
		}
		TWIN_CHECK(rejected != 0u,"mutated expert entries rejected layer-by-layer");
		free(directory);
		free(state.layer_seen_bits);
		printf("test_dsv41_pack_validate: %u mutated entries rejected (expected nonzero)\n",rejected);
		return(0);
	}
	if ( strcmp(mode,"reject-codec") == 0 )
	{
		uint32_t rejected = 0u;
		TWIN_CHECK(DSV41_FLASH_EXPERT_WEIGHT_CODEC == SPARK_WEIGHT_CODEC_MXFP4_E2M1,
			"reject-codec mode runs the uniform mxfp4 build");
		for (index=0u; index<header.tensor_count; index++)
		{
			status = SparkDsv41FlashEntryValidate(&state,&header,&directory[index],file_bytes);
			if ( status != SPARK_STATUS_OK )
			{
				TWIN_CHECK(SparkDsv41FlashStagePackKindIsExpert(directory[index].tensor_kind) != 0u,
					"only expert entries fail the uniform build");
				rejected += 1u;
			}
		}
		TWIN_CHECK(rejected == 3u * header.layer_count,
			"every expert entry of the nvfp4 pack is rejected by the mxfp4 build");
		free(directory);
		free(state.layer_seen_bits);
		printf("test_dsv41_pack_validate: %u nvfp4 expert entries rejected by the mxfp4 build (expected)\n",rejected);
		return(0);
	}
	for (index=0u; index<header.tensor_count; index++)
	{
		status = SparkDsv41FlashEntryValidate(&state,&header,&directory[index],file_bytes);
		TWIN_CHECK(status == SPARK_STATUS_OK,"entry validates against the nvfp4 shape table");
	}
	status = SparkDsv41FlashInventoryValidate(&state);
	TWIN_CHECK(status == SPARK_STATUS_OK,"per-layer coverage complete");
	memset(&manifest,0,sizeof(manifest));
	context.entries = directory;
	context.count = header.tensor_count;
	status = SparkDsv41FlashManifestCheck(&manifest,&context);
	TWIN_CHECK(status != SPARK_STATUS_OK,"manifest range census must disagree before counting");
	expected_ranges = 0u;
	for (index=0u; index<header.tensor_count; index++)
	{
		if ( SparkDsv41FlashStagePackKindIsExpert(directory[index].tensor_kind) == 0u )
			continue;
		expected_ranges += directory[index].group_count;
		expected_ranges += directory[index].scale_bytes != 0u ? directory[index].group_count : 0u;
	}
	manifest.range_count = (uint32_t)expected_ranges;
	status = SparkDsv41FlashManifestCheck(&manifest,&context);
	TWIN_CHECK(status == SPARK_STATUS_OK,"expert manifest range census closes");
	TWIN_CHECK(header.expert_weight_codec == (uint32_t)DSV41_FLASH_EXPERT_WEIGHT_CODEC,
		"pack header codec equals the build codec");
	if ( header.expert_weight_codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 )
	{
		uint32_t experts = 0u;
		uint64_t scale_bytes = 0u;
		for (index=0u; index<header.tensor_count; index++)
		{
			if ( SparkDsv41FlashStagePackKindIsExpert(directory[index].tensor_kind) == 0u )
				continue;
			experts += 1u;
			TWIN_CHECK(directory[index].weight_codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1,
				"expert entry rides the nvfp4 codec");
			TWIN_CHECK(directory[index].scale_encoding == SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL,
				"expert scale rides ue4m3+f32-global");
			TWIN_CHECK(directory[index].scale_bytes ==
				SparkDsv41FlashStagePackNvfp4ScaleBytesPerGroup(directory[index].rows,directory[index].columns) *
					directory[index].group_count,
				"scale window = per-16 ue4m3 plane + f32 global per expert");
			scale_bytes += directory[index].scale_bytes;
		}
		TWIN_CHECK(experts == 3u * header.layer_count,"three expert projections per layer");
		TWIN_CHECK(scale_bytes > 0u,"scale planes present");
	}
	free(directory);
	free(state.layer_seen_bits);
	printf("test_dsv41_pack_validate: nvfp4 pack accepted entry-by-entry "
		"(codec=%u layers %u tensors=%u)\n",
		(unsigned)DSV41_FLASH_EXPERT_WEIGHT_CODEC,header.layer_count,header.tensor_count);
	return(0);
}
