#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdlib.h>
#include <string.h>

#include "spark_gemma4_stagepack_format.h"

#define SPARK_SYNTH_MAX_TENSORS 2048u
#define SPARK_SYNTH_ALIGN_UNIT SPARK_GEMMA4_STAGEPACK_PAYLOAD_ALIGNMENT

typedef struct SparkGemma4SynthEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint64_t payload_offset;
	uint64_t payload_bytes;
} SparkGemma4SynthEntry;

typedef struct SparkGemma4SynthContext
{
	SparkGemma4SynthEntry entries[SPARK_SYNTH_MAX_TENSORS];
	uint32_t entry_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint64_t payload_cursor;
	uint64_t seed;
} SparkGemma4SynthContext;

static uint64_t SparkGemma4SynthNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static uint64_t SparkGemma4SynthAlign(uint64_t offset)
{
	return((offset + SPARK_SYNTH_ALIGN_UNIT - 1u) & ~((uint64_t)SPARK_SYNTH_ALIGN_UNIT - 1u));
}

static void SparkGemma4SynthShape(SparkGemma4SynthContext *context, uint32_t kind, uint32_t layer, SparkGemma4StagePackTensorShape *shape, uint32_t *is_global)
{
	*is_global = layer == SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	if ( SparkGemma4StagePackTensorShapeOf(kind,shape) != 0 )
	{
		fprintf(stderr,"gemma4_pack_synthesize kind=%u unresolvable\n",kind);
		exit(1);
	}
	SparkGemma4StagePackNarrowShape(shape,kind,1u,0u);
	(void)context;
}

static int32_t SparkGemma4SynthAppend(SparkGemma4SynthContext *context, uint32_t kind, uint32_t layer)
{
	SparkGemma4StagePackTensorShape shape;
	uint32_t is_global;
	SparkGemma4SynthEntry *entry;
	if ( context->entry_count >= SPARK_SYNTH_MAX_TENSORS )
		return(-1);
	if ( SparkGemma4StagePackResolvedShape(kind,layer == SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER ? 0u : layer,layer == SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u,&shape) < 0 )
	{
		fprintf(stderr,"gemma4_pack_synthesize resolve_failed kind=%u layer=%u\n",kind,layer);
		return(-2);
	}
	SparkGemma4SynthShape(context,kind,layer,&shape,&is_global);
	entry = &context->entries[context->entry_count++];
	entry->tensor_kind = kind;
	entry->layer_index = is_global != 0u ? SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER : layer;
	entry->weight_format = shape.natural_format;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->payload_bytes = SparkGemma4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	entry->payload_offset = 0u;
	return(0);
}

static void SparkGemma4SynthWrite(FILE *file, const void *data, uint64_t bytes)
{
	if ( bytes == 0u )
		return;
	if ( fwrite(data,1,(size_t)bytes,file) != (size_t)bytes )
	{
		fprintf(stderr,"gemma4_pack_synthesize short write\n");
		exit(1);
	}
}

static void SparkGemma4SynthFillAndWrite(FILE *file, SparkGemma4SynthContext *context, const SparkGemma4SynthEntry *entry)
{
	uint8_t zero[256];
	uint64_t remaining = entry->payload_bytes;
	uint64_t seed = (context->seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(entry->tensor_kind + 1u)) ^ (0xbf58476d1ce4e5b9ull * (uint64_t)(entry->layer_index + 1u))) | 1ull;
	memset(zero,0,sizeof(zero));
	entry = 0;
	{
		uint8_t buffer[4096];
		uint64_t index;
		while ( remaining > 0u )
		{
			uint64_t chunk = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
			for (index = 0; index < chunk; index += 2u)
			{
				uint16_t pair = (uint16_t)SparkGemma4SynthNext(&seed);
				buffer[index] = (uint8_t)(pair >> 8u);
				if ( index + 1u < chunk )
					buffer[index + 1u] = (uint8_t)pair;
			}
			SparkGemma4SynthWrite(file,buffer,chunk);
			remaining -= chunk;
		}
	}
	memset(zero,0,0u);
}

int main(int argc, char **argv)
{
	SparkGemma4SynthContext context;
	SparkGemma4StagePackHeader header;
	SparkGemma4StagePackEntry *directory;
	FILE *file;
	uint32_t index;
	if ( argc < 5 )
	{
		fprintf(stderr,"usage: %s <out.gemma4sp> <first_layer> <layer_count> <seed>\n",argv[0]);
		return(2);
	}
	memset(&context,0,sizeof(context));
	context.first_layer_index = (uint32_t)strtoul(argv[2],0,10);
	context.layer_count = (uint32_t)strtoul(argv[3],0,10);
	context.seed = strtoull(argv[4],0,10);
	context.payload_cursor = SPARK_GEMMA4_STAGEPACK_HEADER_BYTES + (uint64_t)context.layer_count * 0u;
	if ( SparkGemma4SynthAppend(&context,SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_ROPE_TABLE,SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER) != 0 )
		return(1);
	if ( context.first_layer_index == 0u && SparkGemma4SynthAppend(&context,SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING,SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER) != 0 )
		return(1);
	for (index = context.first_layer_index; index < context.first_layer_index + context.layer_count; index++)
	{
		uint32_t kind;
		uint32_t layer_kinds[20];
		uint32_t layer_kind_count = 0u;
		uint32_t scalar_kind_count = 1u;
		uint32_t scalar_kind = SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_SCALAR;
		uint32_t norm_kinds[7] = {
			SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_INPUT_NORM,
			SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_ATTENTION_NORM,
			SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_PRE_FEEDFORWARD_NORM,
			SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_FEEDFORWARD_NORM
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
			,SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_FEEDFORWARD_NORM_1,
			SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_PRE_FEEDFORWARD_NORM_2,
			SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_FEEDFORWARD_NORM_2
#endif
		};
		for (kind = 0u; kind < (uint32_t)(sizeof(norm_kinds) / sizeof(norm_kinds[0])); kind++)
			layer_kinds[layer_kind_count++] = norm_kinds[kind];
		layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_GATE_UP;
		layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_DOWN;
		for (kind = 0u; kind < scalar_kind_count; kind++)
			layer_kinds[layer_kind_count++] = scalar_kind;
		if ( SPARK_GEMMA4_MODEL_LAYER_IS_FULL(index) != 0u )
		{
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_OUTPUT;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY_NORM;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY_NORM;
		}
		else
		{
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KV_FUSED;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_OUTPUT;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY_NORM;
			layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KEY_NORM;
		}
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
		layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_ROUTER_PROJ;
		layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_PER_EXPERT_SCALE;
		layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_GATE_UP;
		layer_kinds[layer_kind_count++] = SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_DOWN;
#endif
		for (kind = 0u; kind < layer_kind_count; kind++)
			if ( SparkGemma4SynthAppend(&context,layer_kinds[kind],index) != 0 )
				return(1);
	}
	if ( context.first_layer_index + context.layer_count == SPARK_GEMMA4_MODEL_LAYER_COUNT )
		if ( SparkGemma4SynthAppend(&context,SPARK_GEMMA4_STAGEPACK_TENSOR_FINAL_NORM,SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER) != 0 )
			return(1);
	if ( context.entry_count != SparkGemma4StagePackExpectedTensorCount(context.first_layer_index,context.layer_count) )
	{
		fprintf(stderr,"gemma4_pack_synthesize tensor_count=%u expected=%u\n",context.entry_count,SparkGemma4StagePackExpectedTensorCount(context.first_layer_index,context.layer_count));
		return(1);
	}
	SparkGemma4StagePackExpectedGeometry(&header,context.first_layer_index,context.layer_count);
	header.tensor_count = context.entry_count;
	header.directory_offset = SPARK_GEMMA4_STAGEPACK_HEADER_BYTES;
	for (index = 0u; index < context.entry_count; index++)
	{
		context.payload_cursor = SparkGemma4SynthAlign(context.payload_cursor);
		context.entries[index].payload_offset = context.payload_cursor;
		context.payload_cursor += context.entries[index].payload_bytes;
	}
	header.file_bytes = context.payload_cursor;
	file = fopen(argv[1],"wb");
	if ( file == 0 )
		return(1);
	SparkGemma4SynthWrite(file,&header,sizeof(header));
	directory = (SparkGemma4StagePackEntry *)malloc((size_t)context.entry_count * sizeof(*directory));
	for (index = 0u; index < context.entry_count; index++)
	{
		memset(&directory[index],0,sizeof(directory[index]));
		directory[index].tensor_kind = context.entries[index].tensor_kind;
		directory[index].layer_index = context.entries[index].layer_index;
		directory[index].weight_format = context.entries[index].weight_format;
		directory[index].rows = context.entries[index].rows;
		directory[index].columns = context.entries[index].columns;
		directory[index].scale_group_size = 0u;
		directory[index].payload_offset = context.entries[index].payload_offset;
		directory[index].payload_bytes = context.entries[index].payload_bytes;
	}
	SparkGemma4SynthWrite(file,directory,(uint64_t)context.entry_count * SPARK_GEMMA4_STAGEPACK_ENTRY_BYTES);
	free(directory);
	for (index = 0u; index < context.entry_count; index++)
	{
		uint64_t position = (uint64_t)ftell(file);
		uint64_t pad = context.entries[index].payload_offset - position;
		while ( pad-- > 0u )
		{
			uint8_t zero = 0u;
			SparkGemma4SynthWrite(file,&zero,1u);
		}
		SparkGemma4SynthFillAndWrite(file,&context,&context.entries[index]);
	}
	fclose(file);
	fprintf(stderr,"gemma4_pack_synthesize ok slice=%u+%u tensors=%u bytes=%llu\n",context.first_layer_index,context.layer_count,context.entry_count,(unsigned long long)header.file_bytes);
	return(0);
}
