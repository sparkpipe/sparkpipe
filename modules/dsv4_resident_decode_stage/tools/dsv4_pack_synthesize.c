#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Synthesizes a geometry-exact DeepSeek V4 stage pack with noise payloads,
 * for loader and pipeline plumbing tests without the 100+ GB checkpoint.
 * The variant comes from whichever model header the BUILD injects (-include
 * sparkpipe/spark_dsv4_model.h or the pro header) before this file's
 * includes resolve; the source itself names neither, so one file serves
 * both and mixing is impossible by the shared include guard.
 *
 * Usage: dsv4_pack_synthesize <output.pack> <first_layer_index> <layer_count>
 */

#include "spark_dsv4_stagepack_format.h"

typedef struct SparkDsv4SynthesizeContext
{
	FILE *file;
	uint64_t cursor;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t sparse;
	uint32_t entry_count;
	uint32_t entry_capacity;
	SparkDsv4StagePackEntry *directory;
	uint64_t seed;
} SparkDsv4SynthesizeContext;

static uint64_t SparkDsv4SynthesizeNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

// Noise per format: small-magnitude bf16/f32, e4m3 bytes with tame
// exponents (never the NaN encodings), packed e2m1 nibbles, e8m0 scales at
// the 1.0 bias, and tid2eid entries bounded by the expert count.
static int32_t SparkDsv4SynthesizeWritePayload(SparkDsv4SynthesizeContext *context, const SparkDsv4StagePackEntry *entry)
{
	uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	uint64_t index,value;
	if ( context->sparse != 0u )
		return(fseek(context->file,(long)(payload_bytes + scale_bytes),SEEK_CUR) == 0 ? 0 : -3);
	uint8_t buffer[4096];
	uint64_t remaining = payload_bytes,chunk,offset;
	while (remaining != 0u)
	{
		chunk = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
		for (offset = 0; offset < chunk; offset++)
		{
			value = SparkDsv4SynthesizeNext(&context->seed);
			if ( entry->weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_BF16 )
				buffer[offset] = (offset & 1u) != 0u ? (uint8_t)(0x3Cu + (value & 3u)) : (uint8_t)value;
			else if ( entry->weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_F32 )
				buffer[offset] = (offset & 3u) == 3u ? (uint8_t)(0x3Cu + (value & 3u)) : (uint8_t)value;
			else if ( entry->weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3 )
				buffer[offset] = (uint8_t)(value & 0x37u);
			else if ( entry->weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1 )
				buffer[offset] = (uint8_t)value;
			else
				buffer[offset] = 0u;
		}
		if ( entry->weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_U32 )
		{
			for (offset = 0; offset + 4u <= chunk; offset += 4u)
			{
				index = SparkDsv4SynthesizeNext(&context->seed) % SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT;
				memcpy(buffer + offset,&index,sizeof(index));
			}
		}
		if ( fwrite(buffer,1u,chunk,context->file) != chunk )
			return(-1);
		remaining -= chunk;
	}
	memset(buffer,127,sizeof(buffer));
	remaining = scale_bytes;
	while (remaining != 0u)
	{
		chunk = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
		if ( fwrite(buffer,1u,chunk,context->file) != chunk )
			return(-2);
		remaining -= chunk;
	}
	return(0);
}

static int32_t SparkDsv4SynthesizeAppend(SparkDsv4SynthesizeContext *context, uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global)
{
	SparkDsv4StagePackTensorShape shape;
	SparkDsv4StagePackEntry *entry;
	uint64_t payload_bytes,scale_bytes;
	if ( context->entry_count >= context->entry_capacity )
		return(-1);
	if ( SparkDsv4StagePackResolvedShape(tensor_kind,layer_index,is_global,&shape) < 0 )
		return(-2);
	entry = &context->directory[context->entry_count];
	memset(entry,0,sizeof(*entry));
	entry->tensor_kind = tensor_kind;
	entry->layer_index = is_global != 0u ? SPARK_DSV4_STAGEPACK_GLOBAL_LAYER : layer_index;
	entry->weight_format = shape.weight_format;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	payload_bytes = SparkDsv4StagePackPayloadBytes(shape.weight_format,shape.rows,shape.columns);
	scale_bytes = SparkDsv4StagePackScaleBytes(shape.weight_format,shape.rows,shape.columns);
	entry->payload_offset = context->cursor;
	entry->scale_offset = scale_bytes != 0u ? context->cursor + payload_bytes : 0u;
	context->cursor += payload_bytes + scale_bytes;
	context->entry_count++;
	return(0);
}

static int32_t SparkDsv4SynthesizeAppendLayer(SparkDsv4SynthesizeContext *context, uint32_t layer_index)
{
	static const uint32_t shared_kinds[26] =
	{
		SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK,SPARK_DSV4_STAGEPACK_TENSOR_WQ_A,SPARK_DSV4_STAGEPACK_TENSOR_Q_NORM,
		SPARK_DSV4_STAGEPACK_TENSOR_WQ_B,SPARK_DSV4_STAGEPACK_TENSOR_WKV,SPARK_DSV4_STAGEPACK_TENSOR_KV_NORM,
		SPARK_DSV4_STAGEPACK_TENSOR_WO_A,SPARK_DSV4_STAGEPACK_TENSOR_WO_B,SPARK_DSV4_STAGEPACK_TENSOR_ATTN_NORM,
		SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM,SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_FN,SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_FN,
		SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_BASE,SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_BASE,SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_SCALE,
		SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_SCALE,SPARK_DSV4_STAGEPACK_TENSOR_GATE_WEIGHT,SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1,
		SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2,SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3,SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W1,
		SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W2,SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3,SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE,
		SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV,SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE
	};
	static const uint32_t indexer_kinds[6] =
	{
		SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B,SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS,SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE,
		SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV,SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE,SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM
	};
	uint32_t kind = SparkDsv4StagePackLayerKind(layer_index),index;
	for (index = 0; index < 23u; index++)
		if ( SparkDsv4SynthesizeAppend(context,shared_kinds[index],layer_index,0u) < 0 )
			return(-1);
	if ( SparkDsv4SynthesizeAppend(context,SparkDsv4StagePackLayerIsHashRouted(layer_index) != 0u ? SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID : SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS,layer_index,0u) < 0 )
		return(-2);
	if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
	{
		for (index = 23u; index < 26u; index++)
			if ( SparkDsv4SynthesizeAppend(context,shared_kinds[index],layer_index,0u) < 0 )
				return(-3);
		if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM,layer_index,0u) < 0 )
			return(-4);
	}
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		for (index = 0; index < 6u; index++)
			if ( SparkDsv4SynthesizeAppend(context,indexer_kinds[index],layer_index,0u) < 0 )
				return(-5);
	return(0);
}

static int32_t SparkDsv4SynthesizeBuildDirectory(SparkDsv4SynthesizeContext *context)
{
	static const uint32_t mtp_globals[8] =
	{
		SPARK_DSV4_STAGEPACK_TENSOR_MTP_E_PROJ,SPARK_DSV4_STAGEPACK_TENSOR_MTP_H_PROJ,SPARK_DSV4_STAGEPACK_TENSOR_MTP_ENORM,
		SPARK_DSV4_STAGEPACK_TENSOR_MTP_HNORM,SPARK_DSV4_STAGEPACK_TENSOR_MTP_FINAL_NORM,SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_FN,
		SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_BASE,SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE
	};
	uint32_t layer_index,last = context->first_layer_index + context->layer_count,index;
	if ( context->first_layer_index == 0u )
		if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING,0u,1u) < 0 )
			return(-1);
	for (layer_index = context->first_layer_index; layer_index < last; layer_index++)
		if ( SparkDsv4SynthesizeAppendLayer(context,layer_index) < 0 )
			return(-2);
	if ( last == SPARK_DSV4_MODEL_LAYER_COUNT )
	{
		if ( context->first_layer_index != 0u )
			if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING,0u,1u) < 0 )
				return(-3);
		if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM,0u,1u) < 0 )
			return(-4);
		if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD,0u,1u) < 0 )
			return(-5);
		if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_FN,0u,1u) < 0 )
			return(-6);
		if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_BASE,0u,1u) < 0 )
			return(-7);
		if ( SparkDsv4SynthesizeAppend(context,SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE,0u,1u) < 0 )
			return(-8);
		for (index = 0; index < 8u; index++)
			if ( SparkDsv4SynthesizeAppend(context,mtp_globals[index],0u,1u) < 0 )
				return(-9);
		if ( SparkDsv4SynthesizeAppendLayer(context,SPARK_DSV4_STAGEPACK_MTP_LAYER) < 0 )
			return(-10);
	}
	if ( context->entry_count != SparkDsv4StagePackExpectedTensorCount(context->first_layer_index,context->layer_count) )
		return(-11);
	return(0);
}

static int32_t SparkDsv4SynthesizeParse(int32_t argc, char **argv, SparkDsv4SynthesizeContext *context)
{
	if ( argc != 4 && argc != 5 )
	{
		fprintf(stderr,"usage: %s <output.pack> <first_layer_index> <layer_count> [sparse]\n",argv[0]);
		return(-1);
	}
	memset(context,0,sizeof(*context));
	context->sparse = argc == 5 && strcmp(argv[4],"sparse") == 0 ? 1u : 0u;
	context->first_layer_index = (uint32_t)strtoul(argv[2],0,10);
	context->layer_count = (uint32_t)strtoul(argv[3],0,10);
	context->seed = 0x00D5F4u ^ ((uint64_t)context->first_layer_index << 32) ^ context->layer_count;
	if ( context->first_layer_index + context->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT || context->layer_count == 0u )
	{
		fprintf(stderr,"invalid slice %u+%u over %u layers\n",context->first_layer_index,context->layer_count,SPARK_DSV4_MODEL_LAYER_COUNT);
		return(-2);
	}
	context->entry_capacity = SparkDsv4StagePackExpectedTensorCount(context->first_layer_index,context->layer_count);
	context->directory = (SparkDsv4StagePackEntry *)malloc((size_t)context->entry_capacity * sizeof(SparkDsv4StagePackEntry));
	return(context->directory != 0 ? 0 : -3);
}

int32_t main(int32_t argc, char **argv)
{
	SparkDsv4SynthesizeContext context;
	SparkDsv4StagePackHeader header;
	uint32_t index;
	int32_t status;
	if ( SparkDsv4SynthesizeParse(argc,argv,&context) < 0 )
		return(2);
	SparkDsv4StagePackExpectedGeometry(&header,context.first_layer_index,context.layer_count);
	context.cursor = (uint64_t)header.header_bytes + (uint64_t)header.tensor_count * header.directory_entry_bytes;
	status = SparkDsv4SynthesizeBuildDirectory(&context);
	if ( status < 0 )
	{
		fprintf(stderr,"directory build failed %d\n",status);
		free(context.directory);
		return(1);
	}
	header.file_bytes = context.cursor;
	context.file = fopen(argv[1],"wb");
	if ( context.file == 0 )
	{
		free(context.directory);
		return(1);
	}
	status = fwrite(&header,sizeof(header),1u,context.file) == 1u ? 0 : -1;
	if ( status == 0 )
		status = fwrite(context.directory,sizeof(SparkDsv4StagePackEntry),context.entry_count,context.file) == context.entry_count ? 0 : -2;
	for (index = 0; status == 0 && index < context.entry_count; index++)
		status = SparkDsv4SynthesizeWritePayload(&context,&context.directory[index]);
	// A sparse pack must still report the true file size: land one real
	// byte at the end so the holes extend to file_bytes exactly.
	if ( status == 0 && context.sparse != 0u && context.cursor > (uint64_t)header.header_bytes + (uint64_t)header.tensor_count * header.directory_entry_bytes )
	{
		status = fseek(context.file,-1L,SEEK_CUR) == 0 ? 0 : -4;
		if ( status == 0 )
			status = fwrite("",1u,1u,context.file) == 1u ? 0 : -5;
	}
	fclose(context.file);
	free(context.directory);
	if ( status != 0 )
	{
		fprintf(stderr,"payload write failed %d\n",status);
		return(1);
	}
	fprintf(stdout,"dsv4 pack %s slice=%u+%u tensors=%u bytes=%llu\n",argv[1],context.first_layer_index,context.layer_count,context.entry_count,(unsigned long long)header.file_bytes);
	return(0);
}
