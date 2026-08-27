/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_qwen38_max_stagepack_format.h"

/*
 * Synthetic Qwen 3.8 Max stage pack writer, slice- and shard-aware (format v2).
 *
 * Emits every tensor one pipeline STAGE will demand, at the geometry the
 * module was compiled for, with reproducible pseudo-random contents. It
 * exercises the loader, the shape table, the slice arithmetic, the TP shard
 * table and the layer walk end to end. It says nothing about output quality:
 * these are not the model's weights, they are correctly shaped noise, and
 * the module cannot tell the difference by design.
 *
 * The tensor list is derived from the same shape table and the same expected
 * tensor count the loader validates against, so the two can never disagree.
 * The default slice is the whole stack (--first-layer 0 --layer-count 92);
 * any PP-N stage is the same tool with its own slice, and --tp-degree
 * N --tp-rank R emits the rank's shard. Experts default to the MXFP4-E2M1
 * production codec (--fp8-experts for the vendor FP8 layout, --bf16 for the
 * all-BF16 test packs).
 */

#define SPARK_QWEN38_MAX_SYNTHESIZE_MAX_TENSORS 1024u
#define SPARK_QWEN38_MAX_SYNTHESIZE_CHUNK_BYTES (8u * 1024u * 1024u)

typedef struct SparkQwen38MaxSynthesizeContext
{
	SparkQwen38MaxStagePackEntry entries[SPARK_QWEN38_MAX_SYNTHESIZE_MAX_TENSORS];
	uint32_t entry_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t expert_mxfp4;
	uint64_t payload_cursor;
	uint64_t seed;
} SparkQwen38MaxSynthesizeContext;

// xorshift64*: a lane of reproducible noise per tensor, seeded from the
// tensor's identity so any tensor regenerates independently of the others.
static uint64_t SparkQwen38MaxSynthesizeNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static uint64_t SparkQwen38MaxSynthesizeTensorSeed(uint64_t seed, uint32_t tensor_kind, uint32_t layer_index)
{
	uint64_t state = seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(tensor_kind + 1u)) ^ (0xbf58476d1ce4e5b9ull * (uint64_t)(layer_index + 1u));
	if ( state == 0u )
		state = 0x2545f4914f6cdd1dull;
	return(state);
}

static uint64_t SparkQwen38MaxSynthesizeAlign(uint64_t offset)
{
	return((offset + SPARK_QWEN38_MAX_STAGEPACK_PAYLOAD_ALIGNMENT - 1u) & ~((uint64_t)SPARK_QWEN38_MAX_STAGEPACK_PAYLOAD_ALIGNMENT - 1u));
}

static int32_t SparkQwen38MaxSynthesizeAppend(SparkQwen38MaxSynthesizeContext *context, uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, uint32_t quantize)
{
	SparkQwen38MaxStagePackTensorShape shape;
	SparkQwen38MaxStagePackEntry *entry;
	uint32_t format;
	if ( context->entry_count >= SPARK_QWEN38_MAX_SYNTHESIZE_MAX_TENSORS )
		return(-1);
	if ( SparkQwen38MaxStagePackResolvedShapeSharded(tensor_kind,layer_index,is_global,context->tp_degree,context->tp_rank,&shape) < 0 )
		return(-2);
	/* Production parity: routed experts carry the MXFP4-E2M1 group-32 codec
	 * (the AMD-Quark packs) unless the caller asks for the FP8 vendor layout
	 * or the all-BF16 test mode. Everything else stays at its natural
	 * format (BF16 spine, F32 decay vectors). */
	format = quantize != 0u ? shape.natural_format : SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	if ( context->expert_mxfp4 != 0u && (tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1 || tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W3 || tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN) && quantize != 0u )
		format = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1;
	entry = &context->entries[context->entry_count];
	entry->tensor_kind = tensor_kind;
	entry->layer_index = (is_global != 0u && layer_index != SPARK_QWEN38_MAX_STAGEPACK_MTP_LAYER) ? SPARK_QWEN38_MAX_STAGEPACK_GLOBAL_LAYER : layer_index;
	entry->weight_format = format;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->scale_group_size = format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? SPARK_QWEN38_MAX_MODEL_MXFP4_GROUP_SIZE : (format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ? 128u : 0u);
	entry->payload_bytes = SparkQwen38MaxStagePackPayloadBytes(format,shape.rows,shape.columns);
	entry->scale_bytes = SparkQwen38MaxStagePackScaleBytes(format,shape.rows,shape.columns);
	entry->payload_offset = SparkQwen38MaxSynthesizeAlign(context->payload_cursor);
	entry->scale_offset = entry->scale_bytes != 0u ? SparkQwen38MaxSynthesizeAlign(entry->payload_offset + entry->payload_bytes) : 0u;
	context->payload_cursor = entry->scale_bytes != 0u ? (entry->scale_offset + entry->scale_bytes) : (entry->payload_offset + entry->payload_bytes);
	context->entry_count++;
	return(0);
}

static int32_t SparkQwen38MaxSynthesizeAppendEveryLayer(SparkQwen38MaxSynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	if ( SparkQwen38MaxSynthesizeAppend(context,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTENTION_NORM,layer_index,0u,quantize) < 0 )
		return(-1);
	if ( SparkQwen38MaxSynthesizeAppend(context,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MLP_NORM,layer_index,0u,quantize) < 0 )
		return(-2);
	static const uint32_t moe_kinds[8] =
	{
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_GATE,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W3,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_GATE,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_UP,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_DOWN,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT
	};
	uint32_t moe_index;
	for (moe_index = 0; moe_index < 8u; moe_index++)
		if ( SparkQwen38MaxSynthesizeAppend(context,moe_kinds[moe_index],layer_index,0u,quantize) < 0 )
			return(-3 - (int32_t)moe_index);
	return(0);
}

static int32_t SparkQwen38MaxSynthesizeAppendGdnLayer(SparkQwen38MaxSynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[9] =
	{
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_QKV,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_GATE,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_BETA,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_DECAY,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_OUTPUT,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_CONV_WEIGHT,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_A_LOG,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_DT_BIAS,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_GDN_NORM
	};
	uint32_t index;
	for (index = 0; index < 9u; index++)
		if ( SparkQwen38MaxSynthesizeAppend(context,kinds[index],layer_index,0u,quantize) < 0 )
			return(-1 - (int32_t)index);
	return(0);
}

static int32_t SparkQwen38MaxSynthesizeAppendAttnLayer(SparkQwen38MaxSynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[6] =
	{
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_VALUE,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT,
		SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_QUERY_NORM,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY_NORM
	};
	uint32_t index;
	for (index = 0; index < 6u; index++)
		if ( SparkQwen38MaxSynthesizeAppend(context,kinds[index],layer_index,0u,quantize) < 0 )
			return(-1 - (int32_t)index);
	return(0);
}

static int32_t SparkQwen38MaxSynthesizeBuildDirectory(SparkQwen38MaxSynthesizeContext *context, uint32_t quantize)
{
	uint32_t layer_index,last;
	int32_t status;
	if ( context->first_layer_index == 0u )
		if ( SparkQwen38MaxSynthesizeAppend(context,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_EMBEDDING,0u,1u,quantize) < 0 )
			return(-1);
	last = context->first_layer_index + context->layer_count;
	for (layer_index = context->first_layer_index; layer_index < last; layer_index++)
	{
		status = SparkQwen38MaxSynthesizeAppendEveryLayer(context,layer_index,quantize);
		if ( status == 0 )
			status = SPARK_QWEN38_MAX_MODEL_LAYER_IS_GDN(layer_index) != 0u ? SparkQwen38MaxSynthesizeAppendGdnLayer(context,layer_index,quantize) : SparkQwen38MaxSynthesizeAppendAttnLayer(context,layer_index,quantize);
		if ( status < 0 )
			return(-2);
	}
	if ( last == SPARK_QWEN38_MAX_MODEL_LAYER_COUNT )
	{
		static const uint32_t mtp_globals[4] =
		{
			SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FC,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_EMBED_NORM,
			SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_HIDDEN_NORM,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MTP_FINAL_NORM
		};
		uint32_t mtp_index;
		if ( context->first_layer_index != 0u )
			if ( SparkQwen38MaxSynthesizeAppend(context,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_EMBEDDING,0u,1u,quantize) < 0 )
				return(-9);
		if ( SparkQwen38MaxSynthesizeAppend(context,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_FINAL_NORM,0u,1u,quantize) < 0 )
			return(-3);
		if ( SparkQwen38MaxSynthesizeAppend(context,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_LM_HEAD,0u,1u,quantize) < 0 )
			return(-4);
		for (mtp_index = 0; mtp_index < 4u; mtp_index++)
			if ( SparkQwen38MaxSynthesizeAppend(context,mtp_globals[mtp_index],0u,1u,quantize) < 0 )
				return(-6);
		if ( SparkQwen38MaxSynthesizeAppendEveryLayer(context,SPARK_QWEN38_MAX_STAGEPACK_MTP_LAYER,quantize) < 0 )
			return(-7);
		if ( SparkQwen38MaxSynthesizeAppendAttnLayer(context,SPARK_QWEN38_MAX_STAGEPACK_MTP_LAYER,quantize) < 0 )
			return(-8);
	}
	if ( context->entry_count != SparkQwen38MaxStagePackExpectedTensorCount(context->first_layer_index,context->layer_count) )
		return(-5);
	return(0);
}

// Payload noise per format: small-magnitude bf16, small f32, raw MXFP4
// nibbles. element_base keeps a chunked write deterministic mid-tensor.
static void SparkQwen38MaxSynthesizeFillPayload(const SparkQwen38MaxStagePackEntry *entry, uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index,noise;
	uint16_t bf16;
	float f32;
	if ( entry->weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
	{
		for (index = 0; index < bytes; index++)
			buffer[index] = (uint8_t)SparkQwen38MaxSynthesizeNext(random_state);
		return;
	}
	if ( entry->weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 )
	{
		for (index = 0; index + 4u <= bytes; index += 4u)
		{
			noise = SparkQwen38MaxSynthesizeNext(random_state);
			f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 262144.0f;
			memcpy(buffer + index,&f32,sizeof(f32));
		}
		return;
	}
	for (index = 0; index + 2u <= bytes; index += 2u)
	{
		noise = SparkQwen38MaxSynthesizeNext(random_state);
		f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 1048576.0f;
		memcpy(&bf16,((const uint8_t *)&f32) + sizeof(bf16),sizeof(bf16));
		memcpy(buffer + index,&bf16,sizeof(bf16));
	}
}

static void SparkQwen38MaxSynthesizeFillScale(uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index;
	for (index = 0; index < bytes; index++)
		buffer[index] = (uint8_t)(120u + (SparkQwen38MaxSynthesizeNext(random_state) & 7u));
}

static int32_t SparkQwen38MaxSynthesizeWriteRegion(FILE *file, const SparkQwen38MaxStagePackEntry *entry, uint64_t offset, uint64_t bytes, uint32_t is_scale, uint64_t *random_state, uint8_t *chunk)
{
	uint64_t moved,step;
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	for (moved = 0; moved < bytes; moved += step)
	{
		step = bytes - moved;
		if ( step > SPARK_QWEN38_MAX_SYNTHESIZE_CHUNK_BYTES )
			step = SPARK_QWEN38_MAX_SYNTHESIZE_CHUNK_BYTES;
		if ( is_scale != 0u )
			SparkQwen38MaxSynthesizeFillScale(chunk,step,random_state);
		else
			SparkQwen38MaxSynthesizeFillPayload(entry,chunk,step,random_state);
		if ( fwrite(chunk,1,(size_t)step,file) != (size_t)step )
			return(-2);
	}
	return(0);
}

static void SparkQwen38MaxSynthesizeShiftPayload(SparkQwen38MaxSynthesizeContext *context, uint64_t payload_base)
{
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		context->entries[index].payload_offset += payload_base;
		if ( context->entries[index].scale_bytes != 0u )
			context->entries[index].scale_offset += payload_base;
	}
}

static int32_t SparkQwen38MaxSynthesizeWriteEntries(SparkQwen38MaxSynthesizeContext *context, FILE *file, uint8_t *chunk)
{
	uint64_t random_state;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		const SparkQwen38MaxStagePackEntry *entry = &context->entries[index];
		random_state = SparkQwen38MaxSynthesizeTensorSeed(context->seed,entry->tensor_kind,entry->layer_index);
		if ( SparkQwen38MaxSynthesizeWriteRegion(file,entry,entry->payload_offset,entry->payload_bytes,0u,&random_state,chunk) < 0 )
			return(-1);
		if ( entry->scale_bytes != 0u && SparkQwen38MaxSynthesizeWriteRegion(file,entry,entry->scale_offset,entry->scale_bytes,1u,&random_state,chunk) < 0 )
			return(-2);
	}
	return(0);
}

static int32_t SparkQwen38MaxSynthesizeWrite(SparkQwen38MaxSynthesizeContext *context, const char *path, const SparkQwen38MaxStagePackHeader *header)
{
	FILE *file;
	uint8_t *chunk;
	int32_t status;
	file = fopen(path,"wb");
	if ( file == 0 )
		return(-1);
	chunk = (uint8_t *)malloc(SPARK_QWEN38_MAX_SYNTHESIZE_CHUNK_BYTES);
	if ( chunk == 0 )
	{
		fclose(file);
		return(-2);
	}
	status = 0;
	if ( fwrite(header,1,sizeof(*header),file) != sizeof(*header) )
		status = -3;
	if ( status == 0 && fwrite(context->entries,sizeof(SparkQwen38MaxStagePackEntry),context->entry_count,file) != context->entry_count )
		status = -4;
	if ( status == 0 )
		status = SparkQwen38MaxSynthesizeWriteEntries(context,file,chunk);
	free(chunk);
	if ( fclose(file) != 0 && status == 0 )
		status = -5;
	return(status);
}

static void SparkQwen38MaxSynthesizeReport(const SparkQwen38MaxSynthesizeContext *context, const SparkQwen38MaxStagePackHeader *header)
{
	uint64_t mxfp4_bytes = 0,dense_bytes = 0;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		if ( context->entries[index].weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
			mxfp4_bytes += context->entries[index].payload_bytes + context->entries[index].scale_bytes;
		else
			dense_bytes += context->entries[index].payload_bytes;
	}
	fprintf(stderr,"qwen38_pack_synthesize slice=%u+%u tensors=%u mxfp4_bytes=%llu dense_bytes=%llu file_bytes=%llu file_gib=%.1f\n",context->first_layer_index,context->layer_count,context->entry_count,(unsigned long long)mxfp4_bytes,(unsigned long long)dense_bytes,(unsigned long long)header->file_bytes,(double)header->file_bytes / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char **argv)
{
	SparkQwen38MaxSynthesizeContext context;
	SparkQwen38MaxStagePackHeader header;
	const char *output_path = 0;
	uint32_t dry_run = 0,quantize = 1u,argument_index;
	uint64_t payload_base;
	memset(&context,0,sizeof(context));
	context.seed = 1u;
	context.layer_count = SPARK_QWEN38_MAX_MODEL_LAYER_COUNT;
	context.tp_degree = 1u;
	context.expert_mxfp4 = 1u;
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
		else if ( strcmp(argv[argument_index],"--tp-degree") == 0 && argument_index + 1u < (uint32_t)argc )
			context.tp_degree = (uint32_t)strtoul(argv[++argument_index],0,10);
		else if ( strcmp(argv[argument_index],"--tp-rank") == 0 && argument_index + 1u < (uint32_t)argc )
			context.tp_rank = (uint32_t)strtoul(argv[++argument_index],0,10);
		else if ( strcmp(argv[argument_index],"--fp8-experts") == 0 )
			context.expert_mxfp4 = 0u;
		else if ( strcmp(argv[argument_index],"--bf16") == 0 )
			quantize = 0u;
		else if ( strcmp(argv[argument_index],"--dry-run") == 0 )
			dry_run = 1u;
		else
		{
			fprintf(stderr,"usage: %s --output PATH [--seed N] [--first-layer N] [--layer-count N] [--tp-degree N] [--tp-rank N] [--fp8-experts] [--bf16] [--dry-run]\n",argv[0]);
			return(1);
		}
	}
	if ( context.layer_count == 0u || context.first_layer_index + context.layer_count > SPARK_QWEN38_MAX_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"qwen38_pack_synthesize invalid slice %u+%u of %u\n",context.first_layer_index,context.layer_count,SPARK_QWEN38_MAX_MODEL_LAYER_COUNT);
		return(2);
	}
	if ( SparkQwen38MaxStagePackShardingFeasible(context.tp_degree) != 0 || context.tp_rank >= context.tp_degree )
	{
		fprintf(stderr,"qwen38_pack_synthesize invalid tp %u/%u\n",context.tp_rank,context.tp_degree);
		return(6);
	}
	if ( output_path == 0 && dry_run == 0u )
	{
		fprintf(stderr,"qwen38_pack_synthesize --output is required unless --dry-run\n");
		return(3);
	}
	if ( SparkQwen38MaxSynthesizeBuildDirectory(&context,quantize) < 0 )
	{
		fprintf(stderr,"qwen38_pack_synthesize directory build failed\n");
		return(4);
	}
	payload_base = SparkQwen38MaxSynthesizeAlign(SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES + ((uint64_t)context.entry_count * SPARK_QWEN38_MAX_STAGEPACK_ENTRY_BYTES));
	SparkQwen38MaxSynthesizeShiftPayload(&context,payload_base);
	SparkQwen38MaxStagePackExpectedGeometry(&header,context.first_layer_index,context.layer_count,context.tp_degree,context.tp_rank);
	header.directory_offset = SPARK_QWEN38_MAX_STAGEPACK_HEADER_BYTES;
	header.file_bytes = payload_base + context.payload_cursor;
	SparkQwen38MaxSynthesizeReport(&context,&header);
	if ( dry_run != 0u )
		return(0);
	if ( SparkQwen38MaxSynthesizeWrite(&context,output_path,&header) < 0 )
	{
		fprintf(stderr,"qwen38_pack_synthesize write failed path=%s\n",output_path);
		return(5);
	}
	fprintf(stderr,"qwen38_pack_synthesize wrote %s\n",output_path);
	return(0);
}
