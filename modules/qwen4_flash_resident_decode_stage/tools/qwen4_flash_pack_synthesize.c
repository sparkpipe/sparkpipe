/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_qwen4_flash_stagepack_format.h"

/*
 * Synthetic Qwen 3.6 27B stage pack writer, slice-aware.
 *
 * Emits every tensor one pipeline STAGE will demand, at the geometry the
 * module was compiled for, with reproducible pseudo-random contents. It
 * exercises the loader, the shape table, the slice arithmetic and the layer
 * walk end to end. It says nothing about output quality: these are not the
 * model's weights, they are correctly shaped noise, and the module cannot
 * tell the difference by design.
 *
 * The tensor list is derived from the same shape table and the same expected
 * tensor count the loader validates against, so the two can never disagree.
 * The default slice is the whole stack (--first-layer 0 --layer-count 64);
 * any PP-N stage is the same tool with its own slice.
 */

#define SPARK_QWEN4_FLASH_SYNTHESIZE_MAX_TENSORS 2048u
#define SPARK_QWEN4_FLASH_SYNTHESIZE_CHUNK_BYTES (8u * 1024u * 1024u)

typedef struct SparkQwen4FlashSynthesizeContext
{
	SparkQwen4FlashStagePackEntry entries[SPARK_QWEN4_FLASH_SYNTHESIZE_MAX_TENSORS];
	uint32_t entry_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint64_t payload_cursor;
	uint64_t seed;
} SparkQwen4FlashSynthesizeContext;

// xorshift64*: a lane of reproducible noise per tensor, seeded from the
// tensor's identity so any tensor regenerates independently of the others.
static uint64_t SparkQwen4FlashSynthesizeNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static uint64_t SparkQwen4FlashSynthesizeTensorSeed(uint64_t seed, uint32_t tensor_kind, uint32_t layer_index)
{
	uint64_t state = seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(tensor_kind + 1u)) ^ (0xbf58476d1ce4e5b9ull * (uint64_t)(layer_index + 1u));
	if ( state == 0u )
		state = 0x2545f4914f6cdd1dull;
	return(state);
}

static uint64_t SparkQwen4FlashSynthesizeAlign(uint64_t offset)
{
	return((offset + SPARK_QWEN4_FLASH_STAGEPACK_PAYLOAD_ALIGNMENT - 1u) & ~((uint64_t)SPARK_QWEN4_FLASH_STAGEPACK_PAYLOAD_ALIGNMENT - 1u));
}

static int32_t SparkQwen4FlashSynthesizeAppend(SparkQwen4FlashSynthesizeContext *context, uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, uint32_t quantize)
{
	SparkQwen4FlashStagePackTensorShape shape;
	SparkQwen4FlashStagePackEntry *entry;
	uint32_t format;
	if ( context->entry_count >= SPARK_QWEN4_FLASH_SYNTHESIZE_MAX_TENSORS )
		return(-1);
	if ( SparkQwen4FlashStagePackResolvedShape(tensor_kind,layer_index,is_global,&shape) < 0 )
		return(-2);
	format = quantize != 0u ? shape.natural_format : SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	entry = &context->entries[context->entry_count];
	entry->tensor_kind = tensor_kind;
	entry->layer_index = (is_global != 0u && layer_index != SPARK_QWEN4_FLASH_STAGEPACK_MTP_LAYER) ? SPARK_QWEN4_FLASH_STAGEPACK_GLOBAL_LAYER : layer_index;
	entry->weight_format = format;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	/* MXFP4 scales group at 32; the FP8_E4M3 codec scales at 128 blocks;
	 * everything else is unscaled. */
	entry->scale_group_size = format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? 32u :
		(format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ? 128u : 0u);
	entry->payload_bytes = SparkQwen4FlashStagePackPayloadBytes(format,shape.rows,shape.columns);
	entry->scale_bytes = SparkQwen4FlashStagePackScaleBytes(format,shape.rows,shape.columns);
	entry->payload_offset = SparkQwen4FlashSynthesizeAlign(context->payload_cursor);
	entry->scale_offset = entry->scale_bytes != 0u ? SparkQwen4FlashSynthesizeAlign(entry->payload_offset + entry->payload_bytes) : 0u;
	context->payload_cursor = entry->scale_bytes != 0u ? (entry->scale_offset + entry->scale_bytes) : (entry->payload_offset + entry->payload_bytes);
	context->entry_count++;
	return(0);
}

static int32_t SparkQwen4FlashSynthesizeAppendEveryLayer(SparkQwen4FlashSynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	if ( SparkQwen4FlashSynthesizeAppend(context,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTENTION_NORM,layer_index,0u,quantize) < 0 )
		return(-1);
	if ( SparkQwen4FlashSynthesizeAppend(context,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_NORM,layer_index,0u,quantize) < 0 )
		return(-2);
		static const uint32_t hc_kinds[6] =
	{
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_DOWN,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_UP,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_INJECT,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_DOWN,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_UP,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_INJECT
	};
	uint32_t hc_index;
	for (hc_index = 0; hc_index < 6u; hc_index++)
		if ( SparkQwen4FlashSynthesizeAppend(context,hc_kinds[hc_index],layer_index,0u,quantize) < 0 )
			return(-9 - (int32_t)hc_index);
	static const uint32_t moe_kinds[8] =
	{
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_GATE,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W1,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W3,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_DOWN,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_UP,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_DOWN,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT
	};
	uint32_t moe_index;
	for (moe_index = 0; moe_index < 8u; moe_index++)
		if ( SparkQwen4FlashSynthesizeAppend(context,moe_kinds[moe_index],layer_index,0u,quantize) < 0 )
			return(-3 - (int32_t)moe_index);
	return(0);
}

static int32_t SparkQwen4FlashSynthesizeAppendGdnLayer(SparkQwen4FlashSynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[9] =
	{
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_QKV,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_GATE,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_BETA,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DECAY,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_OUTPUT,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_CONV_WEIGHT,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_A_LOG,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DT_BIAS,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_NORM
	};
	uint32_t index;
	for (index = 0; index < 9u; index++)
		if ( SparkQwen4FlashSynthesizeAppend(context,kinds[index],layer_index,0u,quantize) < 0 )
			return(-1 - (int32_t)index);
	return(0);
}

static int32_t SparkQwen4FlashSynthesizeAppendAttnLayer(SparkQwen4FlashSynthesizeContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[9] =
	{
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY_NORM,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY_NORM,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_QK,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_Q_NORM,
		SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM
	};
	uint32_t index;
	for (index = 0; index < 9u; index++)
		if ( SparkQwen4FlashSynthesizeAppend(context,kinds[index],layer_index,0u,quantize) < 0 )
			return(-1 - (int32_t)index);
	return(0);
}

static int32_t SparkQwen4FlashSynthesizeBuildDirectory(SparkQwen4FlashSynthesizeContext *context, uint32_t quantize)
{
	uint32_t layer_index,last;
	int32_t status;
	if ( context->first_layer_index == 0u )
		if ( SparkQwen4FlashSynthesizeAppend(context,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_EMBEDDING,0u,1u,quantize) < 0 )
			return(-1);
	last = context->first_layer_index + context->layer_count;
	for (layer_index = context->first_layer_index; layer_index < last; layer_index++)
	{
		status = SparkQwen4FlashSynthesizeAppendEveryLayer(context,layer_index,quantize);
		if ( status == 0 )
			status = SPARK_QWEN4_FLASH_MODEL_LAYER_IS_GDN(layer_index) != 0u ? SparkQwen4FlashSynthesizeAppendGdnLayer(context,layer_index,quantize) : SparkQwen4FlashSynthesizeAppendAttnLayer(context,layer_index,quantize);
		if ( status < 0 )
			return(-2);
	}
	if ( last == SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT )
	{
		static const uint32_t mtp_globals[8] =
		{
			SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FC,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_EMBED_NORM,
			SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_HIDDEN_NORM,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FINAL_NORM,
			SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_DOWN,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_UP,
			SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_DOWN,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_UP
		};
		uint32_t mtp_index;
		if ( context->first_layer_index != 0u )
			if ( SparkQwen4FlashSynthesizeAppend(context,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_EMBEDDING,0u,1u,quantize) < 0 )
				return(-9);
		if ( SparkQwen4FlashSynthesizeAppend(context,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_FINAL_NORM,0u,1u,quantize) < 0 )
			return(-3);
		if ( SparkQwen4FlashSynthesizeAppend(context,SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_LM_HEAD,0u,1u,quantize) < 0 )
			return(-4);
		for (mtp_index = 0; mtp_index < 8u; mtp_index++)
			if ( SparkQwen4FlashSynthesizeAppend(context,mtp_globals[mtp_index],0u,1u,quantize) < 0 )
				return(-6);
		if ( SparkQwen4FlashSynthesizeAppendEveryLayer(context,SPARK_QWEN4_FLASH_STAGEPACK_MTP_LAYER,quantize) < 0 )
			return(-7);
		if ( SparkQwen4FlashSynthesizeAppendAttnLayer(context,SPARK_QWEN4_FLASH_STAGEPACK_MTP_LAYER,quantize) < 0 )
			return(-8);
	}
	if ( context->entry_count != SparkQwen4FlashStagePackExpectedTensorCount(context->first_layer_index,context->layer_count) )
		return(-5);
	return(0);
}

// Payload noise per format: small-magnitude bf16, small f32, raw MXFP4
// nibbles. element_base keeps a chunked write deterministic mid-tensor.
static void SparkQwen4FlashSynthesizeFillPayload(const SparkQwen4FlashStagePackEntry *entry, uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index,noise;
	uint16_t bf16;
	float f32;
	if ( entry->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
	{
		for (index = 0; index < bytes; index++)
			buffer[index] = (uint8_t)SparkQwen4FlashSynthesizeNext(random_state);
		return;
	}
	if ( entry->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 )
	{
		for (index = 0; index + 4u <= bytes; index += 4u)
		{
			noise = SparkQwen4FlashSynthesizeNext(random_state);
			f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 262144.0f;
			memcpy(buffer + index,&f32,sizeof(f32));
		}
		return;
	}
	for (index = 0; index + 2u <= bytes; index += 2u)
	{
		noise = SparkQwen4FlashSynthesizeNext(random_state);
		f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 1048576.0f;
		memcpy(&bf16,((const uint8_t *)&f32) + sizeof(bf16),sizeof(bf16));
		memcpy(buffer + index,&bf16,sizeof(bf16));
	}
}

static void SparkQwen4FlashSynthesizeFillScale(const SparkQwen4FlashStagePackEntry *entry, uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index;
	if ( entry->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
	{
		/* F32 block scales near 1.0 (0x3f80'0000..0x3f80'0007): small
		 * deterministic jitter in the mantissa, never zero. */
		static const uint8_t base[4] = {0x00u,0x00u,0x80u,0x3fu};
		for (index = 0u; index + 4u <= bytes; index += 4u)
		{
			buffer[index] = base[0];
			buffer[index + 1u] = base[1];
			buffer[index + 2u] = base[2];
			buffer[index + 3u] = (uint8_t)(base[3] + (SparkQwen4FlashSynthesizeNext(random_state) & 7u));
		}
		return;
	}
	for (index = 0; index < bytes; index++)
		buffer[index] = (uint8_t)(120u + (SparkQwen4FlashSynthesizeNext(random_state) & 7u));
}

static int32_t SparkQwen4FlashSynthesizeWriteRegion(FILE *file, const SparkQwen4FlashStagePackEntry *entry, uint64_t offset, uint64_t bytes, uint32_t is_scale, uint64_t *random_state, uint8_t *chunk)
{
	uint64_t moved,step;
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	for (moved = 0; moved < bytes; moved += step)
	{
		step = bytes - moved;
		if ( step > SPARK_QWEN4_FLASH_SYNTHESIZE_CHUNK_BYTES )
			step = SPARK_QWEN4_FLASH_SYNTHESIZE_CHUNK_BYTES;
		if ( is_scale != 0u )
			SparkQwen4FlashSynthesizeFillScale(entry,chunk,step,random_state);
		else
			SparkQwen4FlashSynthesizeFillPayload(entry,chunk,step,random_state);
		if ( fwrite(chunk,1,(size_t)step,file) != (size_t)step )
			return(-2);
	}
	return(0);
}

static void SparkQwen4FlashSynthesizeShiftPayload(SparkQwen4FlashSynthesizeContext *context, uint64_t payload_base)
{
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		context->entries[index].payload_offset += payload_base;
		if ( context->entries[index].scale_bytes != 0u )
			context->entries[index].scale_offset += payload_base;
	}
}

static int32_t SparkQwen4FlashSynthesizeWriteEntries(SparkQwen4FlashSynthesizeContext *context, FILE *file, uint8_t *chunk)
{
	uint64_t random_state;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		const SparkQwen4FlashStagePackEntry *entry = &context->entries[index];
		random_state = SparkQwen4FlashSynthesizeTensorSeed(context->seed,entry->tensor_kind,entry->layer_index);
		if ( SparkQwen4FlashSynthesizeWriteRegion(file,entry,entry->payload_offset,entry->payload_bytes,0u,&random_state,chunk) < 0 )
			return(-1);
		if ( entry->scale_bytes != 0u && SparkQwen4FlashSynthesizeWriteRegion(file,entry,entry->scale_offset,entry->scale_bytes,1u,&random_state,chunk) < 0 )
			return(-2);
	}
	return(0);
}

static int32_t SparkQwen4FlashSynthesizeWrite(SparkQwen4FlashSynthesizeContext *context, const char *path, const SparkQwen4FlashStagePackHeader *header)
{
	FILE *file;
	uint8_t *chunk;
	int32_t status;
	file = fopen(path,"wb");
	if ( file == 0 )
		return(-1);
	chunk = (uint8_t *)malloc(SPARK_QWEN4_FLASH_SYNTHESIZE_CHUNK_BYTES);
	if ( chunk == 0 )
	{
		fclose(file);
		return(-2);
	}
	status = 0;
	if ( fwrite(header,1,sizeof(*header),file) != sizeof(*header) )
		status = -3;
	if ( status == 0 && fwrite(context->entries,sizeof(SparkQwen4FlashStagePackEntry),context->entry_count,file) != context->entry_count )
		status = -4;
	if ( status == 0 )
		status = SparkQwen4FlashSynthesizeWriteEntries(context,file,chunk);
	free(chunk);
	if ( fclose(file) != 0 && status == 0 )
		status = -5;
	return(status);
}

static void SparkQwen4FlashSynthesizeReport(const SparkQwen4FlashSynthesizeContext *context, const SparkQwen4FlashStagePackHeader *header)
{
	uint64_t mxfp4_bytes = 0,dense_bytes = 0;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		if ( context->entries[index].weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
			mxfp4_bytes += context->entries[index].payload_bytes + context->entries[index].scale_bytes;
		else
			dense_bytes += context->entries[index].payload_bytes;
	}
	fprintf(stderr,"qwen4_flash_pack_synthesize slice=%u+%u tensors=%u mxfp4_bytes=%llu dense_bytes=%llu file_bytes=%llu file_gib=%.1f\n",context->first_layer_index,context->layer_count,context->entry_count,(unsigned long long)mxfp4_bytes,(unsigned long long)dense_bytes,(unsigned long long)header->file_bytes,(double)header->file_bytes / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char **argv)
{
	SparkQwen4FlashSynthesizeContext context;
	SparkQwen4FlashStagePackHeader header;
	const char *output_path = 0;
	uint32_t dry_run = 0,quantize = 1u,argument_index;
	uint64_t payload_base;
	memset(&context,0,sizeof(context));
	context.seed = 1u;
	context.layer_count = SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT;
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
		else if ( strcmp(argv[argument_index],"--bf16") == 0 )
			quantize = 0u;
		else if ( strcmp(argv[argument_index],"--dry-run") == 0 )
			dry_run = 1u;
		else
		{
			fprintf(stderr,"usage: %s --output PATH [--seed N] [--first-layer N] [--layer-count N] [--bf16] [--dry-run]\n",argv[0]);
			return(1);
		}
	}
	if ( context.layer_count == 0u || context.first_layer_index + context.layer_count > SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"qwen4_flash_pack_synthesize invalid slice %u+%u of %u\n",context.first_layer_index,context.layer_count,SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT);
		return(2);
	}
	if ( output_path == 0 && dry_run == 0u )
	{
		fprintf(stderr,"qwen4_flash_pack_synthesize --output is required unless --dry-run\n");
		return(3);
	}
	if ( SparkQwen4FlashSynthesizeBuildDirectory(&context,quantize) < 0 )
	{
		fprintf(stderr,"qwen4_flash_pack_synthesize directory build failed\n");
		return(4);
	}
	payload_base = SparkQwen4FlashSynthesizeAlign(SPARK_QWEN4_FLASH_STAGEPACK_HEADER_BYTES + ((uint64_t)context.entry_count * SPARK_QWEN4_FLASH_STAGEPACK_ENTRY_BYTES));
	SparkQwen4FlashSynthesizeShiftPayload(&context,payload_base);
	SparkQwen4FlashStagePackExpectedGeometry(&header,context.first_layer_index,context.layer_count,0u);
	header.directory_offset = SPARK_QWEN4_FLASH_STAGEPACK_HEADER_BYTES;
	header.file_bytes = payload_base + context.payload_cursor;
	SparkQwen4FlashSynthesizeReport(&context,&header);
	if ( dry_run != 0u )
		return(0);
	if ( SparkQwen4FlashSynthesizeWrite(&context,output_path,&header) < 0 )
	{
		fprintf(stderr,"qwen4_flash_pack_synthesize write failed path=%s\n",output_path);
		return(5);
	}
	fprintf(stderr,"qwen4_flash_pack_synthesize wrote %s\n",output_path);
	return(0);
}
