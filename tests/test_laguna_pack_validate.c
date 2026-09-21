#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_status.h"

#include "../modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_module.c"

#define TWIN_CHECK(condition,message) \
	do { \
		if ( !(condition) ) \
		{ \
			(void)fprintf(stderr,"FAIL %s at line %d: %s\n",message,(int)__LINE__,#condition); \
			return(2); \
		} \
	} while (0)

static void TwinStatePrepare(SparkLagunaModuleState *state,const SparkLagunaStagePackHeader *header)
{
	memset(state,0,sizeof(*state));
	state->stage_count = header->stage_count;
	state->stage_index = header->stage_index;
	state->first_layer_index = header->first_layer_index;
	state->layer_count = header->layer_count;
	state->expert_weight_codec = LAGUNA_EXPERT_WEIGHT_CODEC;
	state->tp_degree = SparkLagunaStagePackHeaderTpDegree(header);
	state->tp_rank = SparkLagunaStagePackHeaderTpRank(header);
	state->owns_embedding = header->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = header->stage_index + 1u == header->stage_count ? 1u : 0u;
}

static int TwinReadHeaderDirectory(const char *path,SparkLagunaStagePackHeader *header,SparkLagunaStagePackEntry **directory)
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
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_LAGUNA_STAGEPACK_MAX_TENSOR_COUNT )
	{
		(void)fclose(file);
		return(-3);
	}
	*directory = (SparkLagunaStagePackEntry *)calloc(header->tensor_count,sizeof(**directory));
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

static void TwinMarkSeen(SparkLagunaModuleState *state,const SparkLagunaStagePackEntry *entry)
{
	uint64_t bit = UINT64_C(1) << entry->tensor_kind;
	if ( entry->tensor_kind >= SPARK_LAGUNA_STAGEPACK_TENSOR_KIND_COUNT )
		return;
	if ( SparkLagunaStagePackKindIsGlobal(entry->tensor_kind) != 0u )
	{
		state->global_seen_bits |= bit;
		return;
	}
	state->layer_seen_bits[entry->layer_index] |= bit;
}

int main(int argc,char **argv)
{
	SparkLagunaModuleState state;
	SparkLagunaStagePackHeader header,expected;
	SparkLagunaStagePackEntry *directory;
	uint64_t file_bytes;
	uint32_t index,is_global;
	SparkStatus status;
	int mismatch;
	int failures = 0;
	const char *mode;
	if ( argc != 3 )
	{
		(void)fprintf(stderr,"usage: test_laguna_pack_validate PACK MODE\n"
			"  modes: accept reject-header reject-entry reject-coverage\n");
		return(2);
	}
	mode = argv[2];
	if ( TwinReadHeaderDirectory(argv[1],&header,&directory) != 0 )
	{
		(void)fprintf(stderr,"FAIL cannot read pack %s\n",argv[1]);
		return(2);
	}
	file_bytes = header.file_bytes;
	TwinStatePrepare(&state,&header);
	state.pack_directory_offset = header.directory_offset;
	state.pack_flags = header.flags;
	SparkLagunaModulePackExpectGeometry(&state,&expected);
	mismatch = SparkLagunaModulePackGeometryMismatch(&state,&header,&expected);
	if ( strcmp(mode,"reject-header") == 0 )
	{
		free(directory);
		printf("test_laguna_pack_validate: geometry mismatch code=%u (expected nonzero)\n",mismatch);
		return(mismatch != 0u ? 0 : 1);
	}
	TWIN_CHECK(mismatch == 0u,"mixed pack header geometry accepted");
	status = SparkLagunaPackValidateRanges(directory,header.tensor_count);
	TWIN_CHECK(status == SPARK_STATUS_OK,"directory ranges do not overlap");
	if ( strcmp(mode,"reject-entry") == 0 )
	{
		uint32_t rejected = 0u;
		for (index=0u; index<header.tensor_count; index++)
		{
			if ( SparkLagunaModuleKindIsExpert(directory[index].tensor_kind) != 0u &&
				directory[index].weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
				directory[index].weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1;
			status = SparkLagunaModuleValidateEntry(&state,&directory[index],file_bytes,&is_global);
			if ( status != SPARK_STATUS_OK )
				rejected += 1u;
		}
		TWIN_CHECK(rejected != 0u,"mutated entry rejected layer-by-layer");
		free(directory);
		printf("test_laguna_pack_validate: %u entries rejected (expected nonzero)\n",rejected);
		return(0);
	}
	{
		uint32_t hold_seen = 0u;
		uint32_t held_kind = 0u,held_layer = 0u;
		if ( strcmp(mode,"reject-coverage") == 0 )
		{
			for (index=header.tensor_count; index>0u; index--)
			{
				if ( SparkLagunaModuleKindIsExpert(directory[index - 1u].tensor_kind) != 0u )
				{
					hold_seen = 1u;
					held_kind = directory[index - 1u].tensor_kind;
					held_layer = directory[index - 1u].layer_index;
					break;
				}
			}
		}
		for (index=0u; index<header.tensor_count; index++)
		{
			status = SparkLagunaModuleValidateEntry(&state,&directory[index],file_bytes,&is_global);
			TWIN_CHECK(status == SPARK_STATUS_OK,"entry validates layer-by-layer");
			if ( hold_seen == 0u || directory[index].tensor_kind != held_kind ||
				directory[index].layer_index != held_layer )
				TwinMarkSeen(&state,&directory[index]);
		}
	}
	status = SparkLagunaModuleVerifyCoverage(&state);
	if ( strcmp(mode,"reject-coverage") == 0 )
	{
		free(directory);
		TWIN_CHECK(status != SPARK_STATUS_OK,"coverage rejection expected");
		printf("test_laguna_pack_validate: coverage rejected as expected\n");
		return(0);
	}
	TWIN_CHECK(status == SPARK_STATUS_OK,"per-layer coverage complete");
	TWIN_CHECK(state.expert_codec_by_layer[36] == SPARK_WEIGHT_CODEC_FP8_E4M3,"layer 36 pinned fp8");
	TWIN_CHECK(state.expert_codec_by_layer[37] == SPARK_WEIGHT_CODEC_NVFP4_E2M1,"layer 37 pinned nvfp4");
	TWIN_CHECK(state.expert_codec_by_layer[38] == SPARK_WEIGHT_CODEC_BF16,"layer 38 pinned bf16");
	TWIN_CHECK(state.expert_codec_by_layer[39] == SPARK_WEIGHT_CODEC_FP8_E4M3,"layer 39 pinned fp8");
	TWIN_CHECK(state.expert_codec_by_layer[41] == SPARK_WEIGHT_CODEC_BF16,"layer 41 pinned bf16");
	free(directory);
	printf("test_laguna_pack_validate: mixed pack accepted layer-by-layer "
		"(codec=%u layers %u..%u tensors=%u)\n",
		(unsigned)LAGUNA_EXPERT_WEIGHT_CODEC,
		header.first_layer_index,header.first_layer_index + header.layer_count - 1u,
		header.tensor_count);
	return(failures);
}
