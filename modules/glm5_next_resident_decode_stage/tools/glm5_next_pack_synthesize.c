/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_glm5_next_stagepack_format.h"

/*
 * Synthetic glm5_next stage pack writer, tier- and slice-aware.
 *
 * Emits every tensor one pipeline STAGE will demand, at the geometry the
 * module was compiled for, with reproducible pseudo-random contents: the
 * loader, the shape table, the TP shard arithmetic and the hybrid layer
 * walk run end to end against it. These are not the model's weights -
 * they are correctly shaped noise, and the module cannot tell the
 * difference by design.
 *
 * The tensor list is DERIVED from the same shape table
 * (SparkGlm5NextStagePackExpectedShape) the loader validates against, so
 * the two can never disagree: the per-layer kinds fall out of the layer
 * class (34 KDA / 11 DSA / dense-first-3), and the MTP layer's entries
 * only appear when --mtp names a slice that reaches layer 45.
 *
 * Usage:
 *   cc -o glm5_next_pack_synthesize \
 *     -I../../include -I../../model-families/glm5_next/include \
 *     -I../../modules/glm5_next_resident_decode_stage/include \
 *     -I../../modules/glm5_next_resident_decode_stage/source \
 *     tools/glm5_next_pack_synthesize.c
 *   ./glm5_next_pack_synthesize --output stage.g5nsp \
 *     --first-layer 0 --layer-count 45 [--mtp] \
 *     [--tp-degree 1] [--owns-embedding] [--owns-head] [--seed 1]
 */

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

/* xorshift64*: a lane of reproducible noise per tensor, seeded from the
 * tensor's identity so any tensor regenerates independently. */
static uint64_t SparkGlm5NextSynthesizeNext(uint64_t *state)
{
	uint64_t value = *state;
	value ^= value >> 12;
	value ^= value << 25;
	value ^= value >> 27;
	*state = value;
	return(value * 2685821657736338717ull);
}

static uint64_t SparkGlm5NextSynthesizeTensorSeed(uint64_t seed, uint32_t tensor_kind, uint32_t layer_index)
{
	uint64_t state = seed ^ (0x9e3779b97f4a7c15ull * (uint64_t)(tensor_kind + 1u)) ^ (0xbf58476d1ce4e5b9ull * (uint64_t)(layer_index + 1u));
	if ( state == 0u )
		state = 0x2545f4914f6cdd1dull;
	return(state);
}

static uint64_t SparkGlm5NextSynthesizeAlign(uint64_t offset)
{
	return((offset + SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u) & ~((uint64_t)SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u));
}

static int32_t SparkGlm5NextSynthesizeAppend(SparkGlm5NextSynthesizeContext *context, uint32_t tensor_kind, uint32_t layer_index)
{
	SparkGlm5NextStagePackTensorShape shape;
	SparkGlm5NextStagePackEntry *entry;
	if ( context->entry_count >= SPARK_GLM5_NEXT_SYNTHESIZE_MAX_TENSORS )
		return(-1);
	if ( SparkGlm5NextStagePackExpectedShape(tensor_kind,layer_index,context->expert_codec,context->tp_degree,&shape) != 0 )
		return(-2);
	entry = &context->entries[context->entry_count];
	memset(entry,0,sizeof(*entry));
	entry->tensor_kind = tensor_kind;
	entry->layer_index = layer_index;
	entry->payload_type = shape.payload_type;
	entry->weight_codec = shape.weight_codec;
	entry->scale_encoding = shape.scale_encoding;
	entry->group_count = shape.group_count;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->payload_bytes = SparkGlm5NextStagePackExpectedPayloadBytes(&shape);
	entry->scale_bytes = SparkGlm5NextStagePackExpectedScaleBytes(&shape);
	entry->payload_offset = SparkGlm5NextSynthesizeAlign(context->payload_cursor);
	entry->scale_offset = entry->scale_bytes != 0u ? SparkGlm5NextSynthesizeAlign(entry->payload_offset + entry->payload_bytes) : 0u;
	context->payload_cursor = entry->scale_bytes != 0u ? (entry->scale_offset + entry->scale_bytes) : (entry->payload_offset + entry->payload_bytes);
	if ( entry->payload_bytes == 0u )
		return(-3);
	context->entry_count++;
	return(0);
}

/* The per-layer inventory falls straight out of the shape table: every
 * kind that is legal on this layer appends. */
static int32_t SparkGlm5NextSynthesizeAppendLayer(SparkGlm5NextSynthesizeContext *context, uint32_t layer_index)
{
	uint32_t kind;
	for (kind = SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM;
	     kind < SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		/* The MTP head kinds are only legal on the MTP layer, which the
		 * table itself enforces; a kind the table rejects here is simply
		 * not part of this layer's inventory. */
		int32_t appended = SparkGlm5NextSynthesizeAppend(context,kind,layer_index);
		if ( appended == -1 || appended == -3 )
			return(-(int32_t)kind);
	}
	return(0);
}

static int32_t SparkGlm5NextSynthesizeBuild(SparkGlm5NextSynthesizeContext *context)
{
	uint32_t layer;
	uint32_t last = context->first_layer_index + context->layer_count;
	for (layer = context->first_layer_index; layer < last; layer++)
		if ( SparkGlm5NextSynthesizeAppendLayer(context,layer) < 0 )
			return(-1);
	if ( context->include_mtp != 0u && last == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		if ( SparkGlm5NextSynthesizeAppendLayer(context,SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX) < 0 )
			return(-2);
	if ( context->owns_embedding != 0u )
		if ( SparkGlm5NextSynthesizeAppend(context,SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING,SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER) < 0 )
			return(-3);
	if ( context->owns_head != 0u )
	{
		if ( SparkGlm5NextSynthesizeAppend(context,SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM,SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER) < 0 )
			return(-4);
		if ( SparkGlm5NextSynthesizeAppend(context,SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD,SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER) < 0 )
			return(-5);
	}
	return(0);
}

/* Payload noise per format: small-magnitude bf16, small f32, finite FP8
 * bytes (the two e4m3 NaN patterns are masked off - a NaN in a synthesized
 * expert would poison the validator's numeric compare). */
static void SparkGlm5NextSynthesizeFillPayload(const SparkGlm5NextStagePackEntry *entry, uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index,noise;
	uint16_t bf16;
	float f32;
	if ( entry->payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_PACKED_WEIGHT )
	{
		for (index = 0; index < bytes; index++)
		{
			uint8_t raw = (uint8_t)SparkGlm5NextSynthesizeNext(random_state);
			if ( (raw & 0x7Fu) == 0x7Fu )
				raw &= 0x7Eu;
			buffer[index] = raw;
		}
		return;
	}
	if ( entry->payload_type == SPARK_GLM5_NEXT_STAGEPACK_PAYLOAD_F32 )
	{
		for (index = 0; index + 4u <= bytes; index += 4u)
		{
			noise = SparkGlm5NextSynthesizeNext(random_state);
			f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 262144.0f;
			memcpy(buffer + index,&f32,sizeof(f32));
		}
		return;
	}
	for (index = 0; index + 2u <= bytes; index += 2u)
	{
		noise = SparkGlm5NextSynthesizeNext(random_state);
		f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 1048576.0f;
		memcpy(&bf16,((const uint8_t *)&f32) + sizeof(bf16),sizeof(bf16));
		memcpy(buffer + index,&bf16,sizeof(bf16));
	}
}

static void SparkGlm5NextSynthesizeFillScale(const SparkGlm5NextStagePackEntry *entry, uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index;
	(void)entry;
	/* F32 block scales near 1.0 (0x3f80'0000..0x3f80'0007): small
	 * deterministic jitter in the mantissa, never zero. */
	for (index = 0u; index + 4u <= bytes; index += 4u)
	{
		buffer[index] = 0x00u;
		buffer[index + 1u] = 0x00u;
		buffer[index + 2u] = 0x80u;
		buffer[index + 3u] = (uint8_t)(0x3fu + (SparkGlm5NextSynthesizeNext(random_state) & 7u));
	}
}

static int32_t SparkGlm5NextSynthesizeWriteRegion(FILE *file, const SparkGlm5NextStagePackEntry *entry, uint64_t offset, uint64_t bytes, uint32_t is_scale, uint64_t *random_state, uint8_t *chunk)
{
	uint64_t moved,step;
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	for (moved = 0; moved < bytes; moved += step)
	{
		step = bytes - moved;
		if ( step > SPARK_GLM5_NEXT_SYNTHESIZE_CHUNK_BYTES )
			step = SPARK_GLM5_NEXT_SYNTHESIZE_CHUNK_BYTES;
		if ( is_scale != 0u )
			SparkGlm5NextSynthesizeFillScale(entry,chunk,step,random_state);
		else
			SparkGlm5NextSynthesizeFillPayload(entry,chunk,step,random_state);
		if ( fwrite(chunk,1,(size_t)step,file) != (size_t)step )
			return(-2);
	}
	return(0);
}

static int32_t SparkGlm5NextSynthesizeWriteEntries(const SparkGlm5NextSynthesizeContext *context, FILE *file, uint8_t *chunk)
{
	uint64_t random_state;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		const SparkGlm5NextStagePackEntry *entry = &context->entries[index];
		random_state = SparkGlm5NextSynthesizeTensorSeed(context->seed,entry->tensor_kind,entry->layer_index);
		if ( SparkGlm5NextSynthesizeWriteRegion(file,entry,entry->payload_offset,entry->payload_bytes,0u,&random_state,chunk) < 0 )
			return(-1);
		if ( entry->scale_bytes != 0u && SparkGlm5NextSynthesizeWriteRegion(file,entry,entry->scale_offset,entry->scale_bytes,1u,&random_state,chunk) < 0 )
			return(-2);
	}
	return(0);
}

static void SparkGlm5NextSynthesizeHexParse(const char *text, uint8_t *out, uint32_t bytes)
{
	uint32_t index;
	for (index = 0; index < bytes; index++)
		out[index] = 0u;
	if ( text == 0 )
		return;
	for (index = 0; index < bytes * 2u; index++)
	{
		char c = text[index];
		uint8_t value;
		if ( c >= '0' && c <= '9' )
			value = (uint8_t)(c - '0');
		else if ( c >= 'a' && c <= 'f' )
			value = (uint8_t)(c - 'a' + 10);
		else if ( c >= 'A' && c <= 'F' )
			value = (uint8_t)(c - 'A' + 10);
		else
			break;
		out[index / 2u] = (uint8_t)((out[index / 2u] << 4) | value);
	}
}

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
	/* reserved0/1 are the TP identity per the format header. */
	header.reserved0 = context.tp_degree;
	header.reserved1 = 0u;
	snprintf(header.model_revision,sizeof(header.model_revision),"%s",revision);
	if ( contract != 0 )
		SparkGlm5NextSynthesizeHexParse(contract,header.contract_sha256,SPARK_GLM5_NEXT_STAGEPACK_SHA256_BYTES);
	directory_offset = SparkGlm5NextSynthesizeAlign(SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES);
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
	if ( SparkGlm5NextSynthesizeWriteEntries(&context,file,chunk) < 0 )
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
