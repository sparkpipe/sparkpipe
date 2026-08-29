/* Shared synthetic stage-pack generator core (DRY wave 1).
 *
 * The per-family tools/<family>_pack_synthesize.c programs were one pasted
 * template: identical xorshift noise, per-tensor seeding, payload/scale
 * fillers, chunked region writer, directory build, and CLI. This header is
 * that machinery, parameterized by family configuration macros the includer
 * defines first. Family-owned data (stage-pack types and constants, the
 * per-layer kind inventory, the quantization policy, the tier header
 * layout) stays in the family file — that is the DRY law working, not an
 * un-migrated paste.
 *
 * Noise-stream law: the emitted bytes depend ONLY on the primitive bodies
 * below and the family's shape table. Every primitive is a token-faithful
 * move of the pasted body; no arithmetic, ordering, or branch was changed.
 * Each family's synthesized pack is proven byte-identical to the quartet's
 * output (see docs/AGENT_LANE_BRIEFS/reports/dry-wave1-*.md).
 *
 * Required configuration (define before including):
 *   SPARK_SYNTH_TOOL_NAME          report/error key, exactly as pasted
 *   SPARK_SYNTH_MAX_TENSORS        directory capacity
 *   SPARK_SYNTH_CHUNK_BYTES        payload write chunk
 *   SPARK_SYNTH_ALIGN_UNIT         payload/scale alignment constant
 *   SPARK_SYNTH_CONTEXT_T          generator context type (entries[],
 *                                  entry_count, first_layer_index,
 *                                  layer_count, payload_cursor, seed)
 *   SPARK_SYNTH_ENTRY_T            family stage-pack entry type
 *   SPARK_SYNTH_SHAPE_T            family tensor-shape type
 *   SPARK_SYNTH_RESOLVE_SHAPE(kind, layer, is_global, shape)  nonzero=reject
 *   SPARK_SYNTH_PAYLOAD_KIND(entry)             payload-format field reader
 *   SPARK_SYNTH_PACKED_FORMAT                   packed-weight kind value
 *   SPARK_SYNTH_PACKED_NAN_MASK    0x7f masks e4m3 NaN lanes; 0 keeps raw
 *   SPARK_SYNTH_F32_FORMAT                      f32 payload kind value
 *   SPARK_SYNTH_PAYLOAD_BYTES(format, rows, columns)
 *   SPARK_SYNTH_SCALE_BYTES(format, rows, columns)
 *   SPARK_SYNTH_FILL_SCALE(entry, buffer, bytes, state)
 *
 * Qwen-slice template (also define SPARK_SYNTH_QWEN_TEMPLATE to get the
 * whole directory build + CLI + main()):
 *   SPARK_SYNTH_HEADER_T / SPARK_SYNTH_HEADER_BYTES / SPARK_SYNTH_ENTRY_BYTES
 *   SPARK_SYNTH_SELECT_FORMAT(quantize, shape)
 *   SPARK_SYNTH_SCALE_GROUP_SIZE(format)
 *   SPARK_SYNTH_MTP_LAYER / SPARK_SYNTH_GLOBAL_LAYER
 *   SPARK_SYNTH_TENSOR_EMBEDDING / _FINAL_NORM / _LM_HEAD
 *   SPARK_SYNTH_MODEL_LAYER_COUNT / SPARK_SYNTH_MODEL_IS_GDN(layer)
 *   SPARK_SYNTH_EXPECTED_TENSOR_COUNT(first, count)
 *   SPARK_SYNTH_EXPECTED_GEOMETRY(header, first, count)
 *   SPARK_SYNTH_EVERY_LAYER_KINDS / _GDN_LAYER_KINDS / _ATTN_LAYER_KINDS /
 *                                  _MTP_GLOBAL_KINDS   (initializer lists)
 */
#ifndef SPARK_PACK_SYNTHESIZE_COMMON_H
#define SPARK_PACK_SYNTHESIZE_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SPARK_SYNTH_QWEN_TEMPLATE
typedef struct SparkSynthContext
{
	SPARK_SYNTH_ENTRY_T entries[SPARK_SYNTH_MAX_TENSORS];
	uint32_t entry_count;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint64_t payload_cursor;
	uint64_t seed;
} SparkSynthContext;

#ifndef SPARK_SYNTH_CONTEXT_T
#define SPARK_SYNTH_CONTEXT_T SparkSynthContext
#endif
#endif

/* xorshift64*: a lane of reproducible noise per tensor, seeded from the
 * tensor's identity so any tensor regenerates independently of the others. */
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

/* Payload noise per kind: packed bytes (raw nibbles, or finite bytes where
 * the family's packed kind has NaN lanes), small f32, small bf16. */
static void SparkSynthFillPayload(const SPARK_SYNTH_ENTRY_T *entry, uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index,noise;
	uint16_t bf16;
	float f32;
	if ( SPARK_SYNTH_PAYLOAD_KIND(entry) == SPARK_SYNTH_PACKED_FORMAT )
	{
		for (index = 0; index < bytes; index++)
		{
			uint8_t raw = (uint8_t)SparkSynthNext(random_state);
			if ( SPARK_SYNTH_PACKED_NAN_MASK != 0u && (raw & 0x7Fu) == 0x7Fu )
				raw &= 0x7Eu;
			buffer[index] = raw;
		}
		return;
	}
	if ( SPARK_SYNTH_PAYLOAD_KIND(entry) == SPARK_SYNTH_F32_FORMAT )
	{
		for (index = 0; index + 4u <= bytes; index += 4u)
		{
			noise = SparkSynthNext(random_state);
			f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 262144.0f;
			memcpy(buffer + index,&f32,sizeof(f32));
		}
		return;
	}
	for (index = 0; index + 2u <= bytes; index += 2u)
	{
		noise = SparkSynthNext(random_state);
		f32 = ((float)(int32_t)(noise & 0xffffu) - 32768.0f) / 1048576.0f;
		memcpy(&bf16,((const uint8_t *)&f32) + sizeof(bf16),sizeof(bf16));
		memcpy(buffer + index,&bf16,sizeof(bf16));
	}
}

/* Scale noise, family-selected: per-byte jitter, or f32 block scales near
 * 1.0 (0x3f80'0000..0x3f80'0007) - small deterministic mantissa jitter,
 * never zero. */
static void SparkSynthFillScaleBytes(uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index;
	for (index = 0; index < bytes; index++)
		buffer[index] = (uint8_t)(120u + (SparkSynthNext(random_state) & 7u));
}

static void SparkSynthFillScaleF32Blocks(uint8_t *buffer, uint64_t bytes, uint64_t *random_state)
{
	uint64_t index;
	for (index = 0u; index + 4u <= bytes; index += 4u)
	{
		buffer[index] = 0x00u;
		buffer[index + 1u] = 0x00u;
		buffer[index + 2u] = 0x80u;
		buffer[index + 3u] = (uint8_t)(0x3fu + (SparkSynthNext(random_state) & 7u));
	}
}

static int32_t SparkSynthWriteRegion(FILE *file, const SPARK_SYNTH_ENTRY_T *entry, uint64_t offset, uint64_t bytes, uint32_t is_scale, uint64_t *random_state, uint8_t *chunk)
{
	uint64_t moved,step;
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	for (moved = 0; moved < bytes; moved += step)
	{
		step = bytes - moved;
		if ( step > SPARK_SYNTH_CHUNK_BYTES )
			step = SPARK_SYNTH_CHUNK_BYTES;
		if ( is_scale != 0u )
			SPARK_SYNTH_FILL_SCALE(entry,chunk,step,random_state);
		else
			SparkSynthFillPayload(entry,chunk,step,random_state);
		if ( fwrite(chunk,1,(size_t)step,file) != (size_t)step )
			return(-2);
	}
	return(0);
}

static int32_t SparkSynthWriteEntries(const SPARK_SYNTH_CONTEXT_T *context, FILE *file, uint8_t *chunk)
{
	uint64_t random_state;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		const SPARK_SYNTH_ENTRY_T *entry = &context->entries[index];
		random_state = SparkSynthTensorSeed(context->seed,entry->tensor_kind,entry->layer_index);
		if ( SparkSynthWriteRegion(file,entry,entry->payload_offset,entry->payload_bytes,0u,&random_state,chunk) < 0 )
			return(-1);
		if ( entry->scale_bytes != 0u && SparkSynthWriteRegion(file,entry,entry->scale_offset,entry->scale_bytes,1u,&random_state,chunk) < 0 )
			return(-2);
	}
	return(0);
}

#ifdef SPARK_SYNTH_QWEN_TEMPLATE

static int32_t SparkSynthAppend(SparkSynthContext *context, uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, uint32_t quantize)
{
	SPARK_SYNTH_SHAPE_T shape;
	SPARK_SYNTH_ENTRY_T *entry;
	uint32_t format;
	if ( context->entry_count >= SPARK_SYNTH_MAX_TENSORS )
		return(-1);
	if ( SPARK_SYNTH_RESOLVE_SHAPE(tensor_kind,layer_index,is_global,&shape) )
		return(-2);
	format = SPARK_SYNTH_SELECT_FORMAT(quantize,shape);
	entry = &context->entries[context->entry_count];
	entry->tensor_kind = tensor_kind;
	entry->layer_index = (is_global != 0u && layer_index != SPARK_SYNTH_MTP_LAYER) ? SPARK_SYNTH_GLOBAL_LAYER : layer_index;
	entry->weight_format = format;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->scale_group_size = SPARK_SYNTH_SCALE_GROUP_SIZE(format);
	entry->payload_bytes = SPARK_SYNTH_PAYLOAD_BYTES(format,shape.rows,shape.columns);
	entry->scale_bytes = SPARK_SYNTH_SCALE_BYTES(format,shape.rows,shape.columns);
	entry->payload_offset = SparkSynthAlign(context->payload_cursor);
	entry->scale_offset = entry->scale_bytes != 0u ? SparkSynthAlign(entry->payload_offset + entry->payload_bytes) : 0u;
	context->payload_cursor = entry->scale_bytes != 0u ? (entry->scale_offset + entry->scale_bytes) : (entry->payload_offset + entry->payload_bytes);
	context->entry_count++;
	return(0);
}

static int32_t SparkSynthAppendKinds(SparkSynthContext *context, const uint32_t *kinds, uint32_t count, uint32_t layer_index, uint32_t is_global, uint32_t quantize)
{
	uint32_t index;
	for (index = 0; index < count; index++)
		if ( SparkSynthAppend(context,kinds[index],layer_index,is_global,quantize) < 0 )
			return(-1 - (int32_t)index);
	return(0);
}

static int32_t SparkSynthAppendEveryLayer(SparkSynthContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[] =
	{
		SPARK_SYNTH_EVERY_LAYER_KINDS
	};
	return(SparkSynthAppendKinds(context,kinds,(uint32_t)(sizeof(kinds) / sizeof(kinds[0])),layer_index,0u,quantize));
}

static int32_t SparkSynthAppendGdnLayer(SparkSynthContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[] =
	{
		SPARK_SYNTH_GDN_LAYER_KINDS
	};
	return(SparkSynthAppendKinds(context,kinds,(uint32_t)(sizeof(kinds) / sizeof(kinds[0])),layer_index,0u,quantize));
}

static int32_t SparkSynthAppendAttnLayer(SparkSynthContext *context, uint32_t layer_index, uint32_t quantize)
{
	static const uint32_t kinds[] =
	{
		SPARK_SYNTH_ATTN_LAYER_KINDS
	};
	return(SparkSynthAppendKinds(context,kinds,(uint32_t)(sizeof(kinds) / sizeof(kinds[0])),layer_index,0u,quantize));
}

static int32_t SparkSynthBuildDirectory(SparkSynthContext *context, uint32_t quantize)
{
	uint32_t layer_index,last;
	int32_t status;
	if ( context->first_layer_index == 0u )
		if ( SparkSynthAppend(context,SPARK_SYNTH_TENSOR_EMBEDDING,0u,1u,quantize) < 0 )
			return(-1);
	last = context->first_layer_index + context->layer_count;
	for (layer_index = context->first_layer_index; layer_index < last; layer_index++)
	{
		status = SparkSynthAppendEveryLayer(context,layer_index,quantize);
		if ( status == 0 )
			status = SPARK_SYNTH_MODEL_IS_GDN(layer_index) ? SparkSynthAppendGdnLayer(context,layer_index,quantize) : SparkSynthAppendAttnLayer(context,layer_index,quantize);
		if ( status < 0 )
			return(-2);
	}
	if ( last == SPARK_SYNTH_MODEL_LAYER_COUNT )
	{
		static const uint32_t mtp_globals[] =
		{
			SPARK_SYNTH_MTP_GLOBAL_KINDS
		};
		if ( context->first_layer_index != 0u )
			if ( SparkSynthAppend(context,SPARK_SYNTH_TENSOR_EMBEDDING,0u,1u,quantize) < 0 )
				return(-9);
		if ( SparkSynthAppend(context,SPARK_SYNTH_TENSOR_FINAL_NORM,0u,1u,quantize) < 0 )
			return(-3);
		if ( SparkSynthAppend(context,SPARK_SYNTH_TENSOR_LM_HEAD,0u,1u,quantize) < 0 )
			return(-4);
		if ( SparkSynthAppendKinds(context,mtp_globals,(uint32_t)(sizeof(mtp_globals) / sizeof(mtp_globals[0])),0u,1u,quantize) < 0 )
			return(-6);
		if ( SparkSynthAppendEveryLayer(context,SPARK_SYNTH_MTP_LAYER,quantize) < 0 )
			return(-7);
		if ( SparkSynthAppendAttnLayer(context,SPARK_SYNTH_MTP_LAYER,quantize) < 0 )
			return(-8);
	}
	if ( context->entry_count != SPARK_SYNTH_EXPECTED_TENSOR_COUNT(context->first_layer_index,context->layer_count) )
		return(-5);
	return(0);
}

static void SparkSynthShiftPayload(SparkSynthContext *context, uint64_t payload_base)
{
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		context->entries[index].payload_offset += payload_base;
		if ( context->entries[index].scale_bytes != 0u )
			context->entries[index].scale_offset += payload_base;
	}
}

static int32_t SparkSynthWrite(SparkSynthContext *context, const char *path, const SPARK_SYNTH_HEADER_T *header)
{
	FILE *file;
	uint8_t *chunk;
	int32_t status;
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
	if ( status == 0 && fwrite(context->entries,sizeof(SPARK_SYNTH_ENTRY_T),context->entry_count,file) != context->entry_count )
		status = -4;
	if ( status == 0 )
		status = SparkSynthWriteEntries(context,file,chunk);
	free(chunk);
	if ( fclose(file) != 0 && status == 0 )
		status = -5;
	return(status);
}

static void SparkSynthReport(const SparkSynthContext *context, const SPARK_SYNTH_HEADER_T *header)
{
	uint64_t packed_bytes = 0,dense_bytes = 0;
	uint32_t index;
	for (index = 0; index < context->entry_count; index++)
	{
		if ( context->entries[index].weight_format == SPARK_SYNTH_PACKED_FORMAT )
			packed_bytes += context->entries[index].payload_bytes + context->entries[index].scale_bytes;
		else
			dense_bytes += context->entries[index].payload_bytes;
	}
	fprintf(stderr,SPARK_SYNTH_TOOL_NAME " slice=%u+%u tensors=%u mxfp4_bytes=%llu dense_bytes=%llu file_bytes=%llu file_gib=%.1f\n",context->first_layer_index,context->layer_count,context->entry_count,(unsigned long long)packed_bytes,(unsigned long long)dense_bytes,(unsigned long long)header->file_bytes,(double)header->file_bytes / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char **argv)
{
	SparkSynthContext context;
	SPARK_SYNTH_HEADER_T header;
	const char *output_path = 0;
	uint32_t dry_run = 0,quantize = 1u,argument_index;
	uint64_t payload_base;
	memset(&context,0,sizeof(context));
	context.seed = 1u;
	context.layer_count = SPARK_SYNTH_MODEL_LAYER_COUNT;
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
	if ( context.layer_count == 0u || context.first_layer_index + context.layer_count > SPARK_SYNTH_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " invalid slice %u+%u of %u\n",context.first_layer_index,context.layer_count,SPARK_SYNTH_MODEL_LAYER_COUNT);
		return(2);
	}
	if ( output_path == 0 && dry_run == 0u )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " --output is required unless --dry-run\n");
		return(3);
	}
	if ( SparkSynthBuildDirectory(&context,quantize) < 0 )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " directory build failed\n");
		return(4);
	}
	payload_base = SparkSynthAlign(SPARK_SYNTH_HEADER_BYTES + ((uint64_t)context.entry_count * SPARK_SYNTH_ENTRY_BYTES));
	SparkSynthShiftPayload(&context,payload_base);
	SPARK_SYNTH_EXPECTED_GEOMETRY(&header,context.first_layer_index,context.layer_count);
	header.directory_offset = SPARK_SYNTH_HEADER_BYTES;
	header.file_bytes = payload_base + context.payload_cursor;
	SparkSynthReport(&context,&header);
	if ( dry_run != 0u )
		return(0);
	if ( SparkSynthWrite(&context,output_path,&header) < 0 )
	{
		fprintf(stderr,SPARK_SYNTH_TOOL_NAME " write failed path=%s\n",output_path);
		return(5);
	}
	fprintf(stderr,SPARK_SYNTH_TOOL_NAME " wrote %s\n",output_path);
	return(0);
}

#endif /* SPARK_SYNTH_QWEN_TEMPLATE */

#endif /* SPARK_PACK_SYNTHESIZE_COMMON_H */
