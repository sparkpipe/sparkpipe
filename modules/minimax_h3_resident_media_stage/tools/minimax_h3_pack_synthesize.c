#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_minimax_h3_stagepack_format.h"
#include "sparkpipe/spark_error_site.h"

#include "sparkpipe/spark_error_site.h"

#define SPARK_SYNTH_LCG_MULTIPLIER UINT64_C(6364136223846793005)
#define SPARK_SYNTH_LCG_INCREMENT UINT64_C(1442695040888963407)
#define SPARK_SYNTH_CHUNK_TENSORS 64

typedef struct SparkSynthState
{
	uint64_t lcg;
	FILE *file;
	uint8_t buffer[SPARK_SYNTH_CHUNK_TENSORS * 8];
} SparkSynthState;

static void SparkSynthFill(SparkSynthState *state, uint64_t bytes)
{
	uint64_t index;
	while ( bytes != 0u )
	{
		uint64_t chunk = bytes < sizeof(state->buffer) ? bytes : sizeof(state->buffer);
		for (index=0u; index<chunk; index+=2u)
		{
			uint32_t value;
			state->lcg = state->lcg * SPARK_SYNTH_LCG_MULTIPLIER + SPARK_SYNTH_LCG_INCREMENT;
			value = (uint32_t)(state->lcg >> 33u);
			state->buffer[index] = (uint8_t)value;
			state->buffer[index + 1u] = (uint8_t)(value >> 8u);
		}
		fwrite(state->buffer,1u,(size_t)chunk,state->file);
		bytes -= chunk;
	}
}

static void SparkSynthEntry(SparkSynthState *state, uint32_t tensor_kind, uint32_t packed_layer,
	uint32_t weight_format, uint32_t rows, uint32_t columns, uint64_t payload_offset,
	uint64_t payload_bytes, FILE *directory)
{
	SparkMinimaxH3StagePackEntry entry;
	memset(&entry,0,sizeof(entry));
	entry.tensor_kind = tensor_kind;
	entry.layer_index = packed_layer;
	entry.weight_format = weight_format;
	entry.rows = rows;
	entry.columns = columns;
	entry.payload_offset = payload_offset;
	entry.payload_bytes = payload_bytes;
	fwrite(&entry,sizeof(entry),1u,directory);
	SparkSynthFill(state,payload_bytes);
}

static uint64_t SparkSynthTensor(uint32_t rows, uint32_t columns, uint32_t element_bytes)
{
	return((uint64_t)rows * columns * element_bytes + SPARK_MINIMAX_H3_STAGEPACK_PAYLOAD_ALIGNMENT - 1u) /
		SPARK_MINIMAX_H3_STAGEPACK_PAYLOAD_ALIGNMENT * SPARK_MINIMAX_H3_STAGEPACK_PAYLOAD_ALIGNMENT;
}

int main(int argc, char **argv)
{
	SparkMinimaxH3StagePackHeader header;
	SparkSynthState state;
	FILE *file;
	FILE *directory_scratch;
	char directory_path[4096];
	uint32_t block;
	uint64_t payload_offset;
	uint32_t format = argc > 2 && strcmp(argv[2],"f32") == 0 ? SPARK_STAGEPACK_FORMAT_WEIGHT_F32 :
		SPARK_STAGEPACK_FORMAT_WEIGHT_BF16;
	uint32_t element = format == SPARK_STAGEPACK_FORMAT_WEIGHT_F32 ? 4u :
		SPARK_MINIMAX_H3_BF16_ELEMENT_BYTES;
	if ( argc < 2 )
	{
		fprintf(stderr,"usage: %s <output.sp> [bf16|f32]\n",argv[0]);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state.lcg = UINT64_C(0x68333253504b5350);
	memset(&state.buffer,0,sizeof(state.buffer));
	file = fopen(argv[1],"wb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	snprintf(directory_path,sizeof(directory_path),"%s.directory",argv[1]);
	directory_scratch = fopen(directory_path,"wb+");
	if ( directory_scratch == 0 )
	{
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	state.file = file;
	payload_offset = SPARK_MINIMAX_H3_STAGEPACK_HEADER_BYTES;
	SparkSynthEntry(&state,SPARK_MINIMAX_H3_DIT_CONTEXT_EMBEDDER,0x0000ffff,format,
		SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,SPARK_MINIMAX_H3_DIT_CONTEXT_DIMENSION,
		payload_offset,SparkSynthTensor(SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,
		SPARK_MINIMAX_H3_DIT_CONTEXT_DIMENSION,element),directory_scratch);
	payload_offset += SparkSynthTensor(SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,
		SPARK_MINIMAX_H3_DIT_CONTEXT_DIMENSION,element);
	SparkSynthEntry(&state,SPARK_MINIMAX_H3_DIT_PROJ_IN,0x0000ffff,format,
		SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,SPARK_MINIMAX_H3_DIT_PATCH_ELEMENTS,
		payload_offset,SparkSynthTensor(SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,
		SPARK_MINIMAX_H3_DIT_PATCH_ELEMENTS,element),directory_scratch);
	payload_offset += SparkSynthTensor(SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,
		SPARK_MINIMAX_H3_DIT_PATCH_ELEMENTS,element);
	for (block=0u; block<SPARK_MINIMAX_H3_DIT_BLOCK_COUNT; block++)
	{
		uint32_t packed = block;
		SparkSynthEntry(&state,SPARK_MINIMAX_H3_DIT_ATTENTION_QUERY,packed,format,
			SPARK_MINIMAX_H3_DIT_QKV_DIMENSION,SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,
			payload_offset,SparkSynthTensor(SPARK_MINIMAX_H3_DIT_QKV_DIMENSION,
			SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,element),directory_scratch);
		payload_offset += SparkSynthTensor(SPARK_MINIMAX_H3_DIT_QKV_DIMENSION,
			SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,element);
		SparkSynthEntry(&state,SPARK_MINIMAX_H3_DIT_FFN_GATE_UP,packed,format,
			SPARK_MINIMAX_H3_DIT_FFN_FUSED_DIMENSION,SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,
			payload_offset,SparkSynthTensor(SPARK_MINIMAX_H3_DIT_FFN_FUSED_DIMENSION,
			SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,element),directory_scratch);
		payload_offset += SparkSynthTensor(SPARK_MINIMAX_H3_DIT_FFN_FUSED_DIMENSION,
			SPARK_MINIMAX_H3_DIT_HIDDEN_DIMENSION,element);
		SparkSynthEntry(&state,SPARK_MINIMAX_H3_DIT_ADALN,packed,format,
			SPARK_MINIMAX_H3_DIT_ADALN_ROWS,SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION,
			payload_offset,SparkSynthTensor(SPARK_MINIMAX_H3_DIT_ADALN_ROWS,
			SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION,element),directory_scratch);
		payload_offset += SparkSynthTensor(SPARK_MINIMAX_H3_DIT_ADALN_ROWS,
			SPARK_MINIMAX_H3_DIT_TIME_EMBED_DIMENSION,element);
	}
	if ( fflush(file) != 0 || fseek(directory_scratch,0,SEEK_SET) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	{
		char copy[8192];
		size_t read;
		while ( (read = fread(copy,1u,sizeof(copy),directory_scratch)) != 0u )
		{
			if ( fwrite(copy,1u,read,file) != read )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	}
	SparkMinimaxH3StagePackExpectedGeometry(&header);
	header.directory_offset = payload_offset;
	header.file_bytes = payload_offset + (uint64_t)ftell(directory_scratch);
	fseek(file,0,SEEK_SET);
	fwrite(&header,sizeof(header),1u,file);
	fclose(directory_scratch);
	remove(directory_path);
	if ( fclose(file) != 0 )
		return(1);
	printf("synthesized %s: %u block tensors + 2 globals, %llu payload bytes\n",
		argv[1],(unsigned)(SPARK_MINIMAX_H3_DIT_BLOCK_COUNT * 3u + 2u),
		(unsigned long long)payload_offset);
	return(0);
}
