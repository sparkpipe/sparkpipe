#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_muse_glimmer_stagepack_format.h"
#include "sparkpipe/spark_error_site.h"


#define SPARK_SYNTH_TOOL_NAME "muse_glimmer_pack_synthesize"
#define SPARK_SYNTH_CHUNK_BYTES (8u * 1024u * 1024u)
#define SPARK_SYNTH_ALIGN_UNIT SPARK_MUSE_GLIMMER_STAGEPACK_PAYLOAD_ALIGNMENT
#define SPARK_SYNTH_DEFAULT_TP 16u

static const uint32_t SPARK_SYNTH_LAYER_KINDS[8u] =
{
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_INPUT_NORM,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_ATTENTION_NORM,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_PRE_FFN_NORM,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_FFN_NORM,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_QGKV,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_OUTPUT,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_GATE_UP,
	SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_MLP_DOWN
};

typedef struct SparkSynthContext
{
	SparkMuseGlimmerStagePackEntry entries[1024u];
	uint32_t entry_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t tp_degree;
	uint64_t payload_cursor;
	uint64_t seed;
}
SparkSynthContext;

static uint64_t SparkSynthNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static uint64_t SparkSynthTensorSeed(uint64_t seed, uint32_t tensor_kind, uint32_t layer_index)
{
	uint64_t state = seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(tensor_kind + 1u)) ^ (0xbf58476d1ce4e5b9ull * (uint64_t)(layer_index + 1u));
	if ( state == 0u )
		state = 0x2545f4914f6cdd1dull;
	return(state);
}

static uint64_t SparkSynthAlign(uint64_t offset)
{
	return((offset + SPARK_SYNTH_ALIGN_UNIT - 1u) & ~((uint64_t)SPARK_SYNTH_ALIGN_UNIT - 1u));
}

static void SparkSynthFillPayload(const SparkMuseGlimmerStagePackEntry *entry, uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index,noise;
	uint16_t bf16;
	float f32;
	for (index = 0; index + 2u <= bytes; index += 2u)
	{
		noise = SparkSynthNext(random_state);
		f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 1048576.0f;
		memcpy(&bf16,((const uint8_t *)&f32) + sizeof(bf16),sizeof(bf16));
		memcpy(buffer + index,&bf16,sizeof(bf16));
	}
	(void)entry;
}

static int32_t SparkSynthAppend(SparkSynthContext *context, uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global)
{
	SparkMuseGlimmerStagePackEntry *entry = &context->entries[context->entry_count];
	SparkMuseGlimmerStagePackTensorShape shape;
	if ( SparkMuseGlimmerStagePackResolvedShape(tensor_kind,is_global != 0u ? 0u : layer_index,is_global,context->tp_degree,&shape) != 0 )
		return(-1);
	entry->tensor_kind = tensor_kind;
	entry->layer_index = is_global != 0u ? SPARK_MUSE_GLIMMER_STAGEPACK_GLOBAL_LAYER : layer_index;
	entry->weight_format = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->scale_group_size = 0u;
	entry->payload_bytes = SparkMuseGlimmerStagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	entry->scale_bytes = 0u;
	entry->payload_offset = SparkSynthAlign(context->payload_cursor);
	entry->scale_offset = 0u;
	context->payload_cursor = entry->payload_offset + entry->payload_bytes;
	context->entry_count++;
	return(0);
}

static int32_t SparkSynthBuildDirectory(SparkSynthContext *context)
{
	uint32_t layer,last,index;
	if ( context->first_layer_index == 0u )
		if ( SparkSynthAppend(context,SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_EMBEDDING,0u,1u) < 0 )
			return(-1);
	last = context->first_layer_index + context->layer_count;
	for (layer = context->first_layer_index; layer < last; layer++)
		for (index = 0u; index < 8u; index++)
			if ( SparkSynthAppend(context,SPARK_SYNTH_LAYER_KINDS[index],layer,0u) < 0 )
				return(-2);
	if ( last == SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT )
	{
		if ( context->first_layer_index != 0u )
			if ( SparkSynthAppend(context,SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_EMBEDDING,0u,1u) < 0 )
				return(-3);
		if ( SparkSynthAppend(context,SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_FINAL_NORM,0u,1u) < 0 )
			return(-4);
		if ( SparkSynthAppend(context,SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_LM_HEAD,0u,1u) < 0 )
			return(-5);
	}
	if ( context->entry_count != SparkMuseGlimmerStagePackExpectedTensorCount(context->first_layer_index,context->layer_count) )
		return(-6);
	return(0);
}

static int32_t SparkSynthWriteRegion(FILE *file, const SparkMuseGlimmerStagePackEntry *entry, uint64_t offset, uint64_t bytes, uint64_t *random_state, uint8_t *chunk)
{
	uint64_t moved,step;
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	for (moved = 0; moved < bytes; moved += step)
	{
		step = bytes - moved;
		if ( step > SPARK_SYNTH_CHUNK_BYTES )
			step = SPARK_SYNTH_CHUNK_BYTES;
		memset(chunk,0,(size_t)step);
		SparkSynthFillPayload(entry,chunk,step,random_state);
		if ( fwrite(chunk,1,(size_t)step,file) != step )
			return(-2);
	}
	return(0);
}

static int32_t SparkSynthWrite(const SparkSynthContext *context, const char *path, const SparkMuseGlimmerStagePackHeader *header)
{
	FILE *file;
	uint8_t *chunk;
	uint32_t index;
	int32_t status = 0;
	uint64_t random_state;
	file = fopen(path,"wb");
	if ( file == 0 )
		return(-1);
	chunk = (uint8_t *)malloc(SPARK_SYNTH_CHUNK_BYTES);
	if ( chunk == 0 )
	{
		fclose(file);
		return(-2);
	}
	status = 0;
	if ( fwrite(header,1,sizeof(*header),file) != sizeof(*header) )
		status = -3;
	if ( status == 0 && fwrite(context->entries,sizeof(context->entries[0]),context->entry_count,file) != context->entry_count )
		status = -4;
	for (index = 0; status == 0 && index < context->entry_count; index++)
	{
		random_state = SparkSynthTensorSeed(context->seed,context->entries[index].tensor_kind,context->entries[index].layer_index);
		status = SparkSynthWriteRegion(file,&context->entries[index],context->entries[index].payload_offset,context->entries[index].payload_bytes,&random_state,chunk);
	}
	free(chunk);
	if ( fclose(file) != 0 && status == 0 )
		status = -5;
	return(status);
}

int main(int argc, char **argv)
{
	SparkSynthContext context;
	SparkMuseGlimmerStagePackHeader header;
	const char *output_path = 0;
	uint32_t dry_run = 0,argument_index;
	uint64_t payload_base;
	memset(&context,0,sizeof(context));
	context.seed = 1u;
	context.tp_degree = SPARK_SYNTH_DEFAULT_TP;
	context.layer_count = SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT;
	for (argument_index = 1; argument_index < (uint32_t)argc; argument_index++)
	{
		if ( strcmp(argv[argument_index],"--output") == 0 && argument_index + 1u < (uint32_t)argc )
			output_path = argv[++argument_index];
		else if ( strcmp(argv[argument_index],"--seed") == 0 && argument_index + 1u < (uint32_t)argc )
			context.seed = strtoull(argv[++argument_index],0,10);
		else if ( strcmp(argv[argument_index],"--first-layer") == 0 && argument_index + 1u < (uint32_t)argc )
			context.first_layer_index = (uint32_t)strtoul(argv[++argument_index],0,10);
		else if ( strcmp(argv[argument_index],"--layer-count") == 0 && argument_index + 1u < (uint32_t)argc )
			context.layer_count = (uint32_t)strtoul(argv[++argument_index],0,10);
		else if ( strcmp(argv[argument_index],"--tp") == 0 && argument_index + 1u < (uint32_t)argc )
			context.tp_degree = (uint32_t)strtoul(argv[++argument_index],0,10);
		else if ( strcmp(argv[argument_index],"--dry-run") == 0 )
			dry_run = 1u;
		else
		{
			fprintf(stderr,"usage: %s --output PATH [--seed N] [--first-layer N] [--layer-count N] [--tp N] [--dry-run]\n",argv[0]);
			return(1);
		}
	}
	if ( context.tp_degree == 0u || SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT % context.tp_degree != 0u || SPARK_MUSE_GLIMMER_MODEL_INTERMEDIATE_DIMENSION % context.tp_degree != 0u || SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT % context.tp_degree != 0u )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " invalid tp degree %u\n",context.tp_degree);
		return(2);
	}
	if ( context.layer_count == 0u || context.first_layer_index + context.layer_count > SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " invalid slice %u+%u of %u\n",context.first_layer_index,context.layer_count,SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT);
		return(2);
	}
	if ( output_path == 0 && dry_run == 0u )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " --output is required unless --dry-run\n");
		return(3);
	}
	if ( SparkSynthBuildDirectory(&context) != 0 )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " directory build failed\n");
		return(4);
	}
	payload_base = SparkSynthAlign(SPARK_MUSE_GLIMMER_STAGEPACK_HEADER_BYTES + ((uint64_t)context.entry_count * SPARK_MUSE_GLIMMER_STAGEPACK_ENTRY_BYTES));
	for (argument_index = 0; argument_index < context.entry_count; argument_index++)
	{
		context.entries[argument_index].payload_offset += payload_base;
	}
	SparkMuseGlimmerStagePackExpectedGeometry(&header,context.first_layer_index,context.layer_count);
	header.directory_offset = SPARK_MUSE_GLIMMER_STAGEPACK_HEADER_BYTES;
	header.file_bytes = payload_base + context.payload_cursor;
	fprintf(stderr,SPARK_SYNTH_TOOL_NAME " slice=%u+%u tp=%u tensors=%u file_bytes=%llu file_gib=%.1f\n",context.first_layer_index,context.layer_count,context.tp_degree,context.entry_count,(unsigned long long)header.file_bytes,(double)header.file_bytes / (1024.0 * 1024.0 * 1024.0));
	if ( dry_run != 0u )
		return(0);
	if ( SparkSynthWrite(&context,output_path,&header) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	fprintf(stderr,SPARK_SYNTH_TOOL_NAME " wrote %s\n",output_path);
	return(0);
}
