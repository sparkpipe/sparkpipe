#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "spark_glm5_next_stagepack_format.h"
#define SPARK_FAMILY_CAMEL Glm5Next
#define SPARK_FAMILY_UPPER GLM5_NEXT
#define SPARK_FAMILY_LOWER glm5_next

#include "sparkpipe/family/spark_family.h"

#define SPARK_GLM5_NEXT_SYNTHESIZE_MAX_TENSORS 2048u
#define SPARK_GLM5_NEXT_SYNTHESIZE_CHUNK_BYTES (8u * 1024u * 1024u)

typedef struct SparkGlm5NextSynthesizeContext
{
	SparkGlm5NextStagePackEntry entries[SPARK_GLM5_NEXT_SYNTHESIZE_MAX_TENSORS];
	uint32_t entry_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t tp_degree;
	uint32_t owns_embedding;
	uint32_t owns_head;
	uint32_t include_mtp;
	uint32_t expert_codec;
	uint64_t payload_cursor;
	uint64_t seed;
} SparkGlm5NextSynthesizeContext;

#define SPARK_SYNTH_CONTEXT_T SparkGlm5NextSynthesizeContext
#define SPARK_SYNTH_CHUNK_BYTES SPARK_GLM5_NEXT_SYNTHESIZE_CHUNK_BYTES
#define SPARK_SYNTH_ALIGN_UNIT SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES
#define SPARK_SYNTH_ENTRY_T SparkGlm5NextStagePackEntry
#define SPARK_SYNTH_PAYLOAD_KIND(entry) ((entry)->payload_type)
#define SPARK_SYNTH_PACKED_FORMAT SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT
#define SPARK_SYNTH_PACKED_NAN_MASK 0x7Fu
#define SPARK_SYNTH_F32_FORMAT SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32
#define SPARK_SYNTH_FILL_SCALE(entry, buffer, bytes, state) \
	SparkSynthFillScaleF32Blocks((buffer),(bytes),(state))

#include "sparkpipe/spark_pack_synthesize_common.h"

#include "sparkpipe/family/synth/spark_synth_glm.h"

#include "sparkpipe/family/synth/spark_synth_glm5_next_ling.h"

int main(int argc, char **argv)
{
	SparkGlm5NextStagePackHeader header;
	SparkGlm5NextSynthesizeContext context;
	const char *output = "glm5_next_stage.g5nsp";
	const char *revision = "synthesized";
	const char *contract = 0;
	FILE *file;
	uint8_t *chunk;
	uint64_t directory_offset;
	uint32_t stage_count = 1u;
	uint32_t stage_index = 0u;
	int32_t status;
	uint32_t index;

	memset(&context,0,sizeof(context));
	context.tp_degree = 1u;
	context.expert_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	context.seed = 1u;
	context.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	for (index = 1u; index < (uint32_t)argc; index++)
	{
		const char *argument = argv[index];
		if ( strcmp(argument,"--output") == 0 && index + 1u < (uint32_t)argc )
			output = argv[++index];
		else if ( strcmp(argument,"--first-layer") == 0 && index + 1u < (uint32_t)argc )
			context.first_layer_index = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argument,"--layer-count") == 0 && index + 1u < (uint32_t)argc )
			context.layer_count = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argument,"--stage-count") == 0 && index + 1u < (uint32_t)argc )
			stage_count = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argument,"--stage-index") == 0 && index + 1u < (uint32_t)argc )
			stage_index = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argument,"--tp-degree") == 0 && index + 1u < (uint32_t)argc )
			context.tp_degree = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argument,"--expert-codec") == 0 && index + 1u < (uint32_t)argc )
		{
			const char *name = argv[++index];
			context.expert_codec = strcmp(name,"int6") == 0 ? SPARK_WEIGHT_CODEC_INT6 :
				strcmp(name,"int7") == 0 ? SPARK_WEIGHT_CODEC_INT7 :
				strcmp(name,"int8") == 0 ? SPARK_WEIGHT_CODEC_INT8 :
				strcmp(name,"fp8") == 0 ? SPARK_WEIGHT_CODEC_FP8_E4M3 :
				strcmp(name,"nvfp4") == 0 ? SPARK_WEIGHT_CODEC_NVFP4_E2M1 :
				strcmp(name,"mxfp4") == 0 ? SPARK_WEIGHT_CODEC_MXFP4_E2M1 : 0u;
			if ( context.expert_codec == 0u )
			{
				fprintf(stderr,"unknown expert codec '%s'\n",name);
				return(2);
			}
		}
		else if ( strcmp(argument,"--owns-embedding") == 0 )
			context.owns_embedding = 1u;
		else if ( strcmp(argument,"--owns-head") == 0 )
			context.owns_head = 1u;
		else if ( strcmp(argument,"--mtp") == 0 )
			context.include_mtp = 1u;
		else if ( strcmp(argument,"--seed") == 0 && index + 1u < (uint32_t)argc )
			context.seed = strtoull(argv[++index],0,10);
		else if ( strcmp(argument,"--revision") == 0 && index + 1u < (uint32_t)argc )
			revision = argv[++index];
		else if ( strcmp(argument,"--contract-sha256") == 0 && index + 1u < (uint32_t)argc )
			contract = argv[++index];
		else
		{
			fprintf(stderr,"unknown or incomplete argument '%s'\n",argument);
			return(2);
		}
	}
	if ( context.tp_degree == 0u || context.layer_count == 0u ||
	     context.first_layer_index + context.layer_count > SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX ||
	     (context.include_mtp != 0u && context.first_layer_index + context.layer_count != SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX) )
	{
		fprintf(stderr,"slice out of range (first %u count %u, model layers %u + MTP)\n",
			context.first_layer_index,context.layer_count,(uint32_t)SPARK_GLM5_NEXT_MODEL_LAYER_COUNT);
		return(2);
	}
	status = SparkGlm5NextSynthesizeBuild(&context);
	if ( status != 0 )
	{
		fprintf(stderr,"inventory build failed: %d\n",status);
		return(1);
	}
	file = fopen(output,"wb");
	if ( file == 0 )
	{
		fprintf(stderr,"cannot open '%s' for writing\n",output);
		return(1);
	}
	chunk = (uint8_t *)malloc(SPARK_GLM5_NEXT_SYNTHESIZE_CHUNK_BYTES);
	if ( chunk == 0 )
	{
		fclose(file);
		return(1);
	}
	memset(&header,0,sizeof(header));
	header.magic = SPARK_GLM5_NEXT_STAGEPACK_MAGIC;
	header.format_version = SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES;
	header.codec_abi_version = 1u;
	header.flags = context.include_mtp != 0u ? SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP : 0u;
	header.tensor_count = context.entry_count;
	header.stage_count = stage_count;
	header.stage_index = stage_index;
	header.first_layer_index = context.first_layer_index;
	header.layer_count = context.layer_count;
	header.total_layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	header.hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	header.vocab_count = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	header.routed_expert_count = SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT;
	header.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	header.expert_weight_codec = context.expert_codec;
	header.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	header.reserved0 = context.tp_degree;
	header.reserved1 = 0u;
	snprintf(header.model_revision,sizeof(header.model_revision),"%s",revision);
	if ( contract != 0 )
		SparkGlm5NextSynthesizeHexParse(contract,header.contract_sha256,SPARK_GLM5_NEXT_STAGEPACK_SHA256_BYTES);
	directory_offset = SparkSynthAlign(SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES);
	header.directory_offset = directory_offset;
	if ( fwrite(&header,sizeof(header),1,file) != 1u ||
	     fseeko(file,(off_t)directory_offset,SEEK_SET) != 0 )
	{
		fprintf(stderr,"header write failed\n");
		free(chunk);
		fclose(file);
		return(1);
	}
	for (index = 0; index < context.entry_count; index++)
		if ( fwrite(&context.entries[index],sizeof(context.entries[index]),1,file) != 1u )
		{
			fprintf(stderr,"directory write failed at %u\n",index);
			free(chunk);
			fclose(file);
			return(1);
		}
	if ( SparkSynthWriteEntries(&context,file,chunk) < 0 )
	{
		fprintf(stderr,"payload write failed\n");
		free(chunk);
		fclose(file);
		return(1);
	}
	if ( fseeko(file,0,SEEK_END) != 0 )
	{
		free(chunk);
		fclose(file);
		return(1);
	}
	header.file_bytes = (uint64_t)ftello(file);
	if ( fseeko(file,0,SEEK_SET) != 0 ||
	     fwrite(&header,sizeof(header),1,file) != 1u )
	{
		free(chunk);
		fclose(file);
		return(1);
	}
	free(chunk);
	fclose(file);
	printf("%s: %u tensors, %llu bytes (layers %u..%u%s, tp%u, codec %u)\n",
		output,context.entry_count,(unsigned long long)header.file_bytes,
		context.first_layer_index,context.first_layer_index + context.layer_count - 1u,
		context.include_mtp != 0u ? " +MTP" : "",context.tp_degree,context.expert_codec);
	return(0);
}
