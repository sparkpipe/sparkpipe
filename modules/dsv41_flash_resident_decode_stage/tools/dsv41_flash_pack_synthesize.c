#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_weight_codec.h"

#include "../source/spark_dsv41_flash_stagepack_format.h"

#define SYNTH_ALIGN 256u
#define SYNTH_MAX_TENSORS 8192u

typedef struct SynthState
{
	SparkDsv41FlashStagePackEntry entries[SYNTH_MAX_TENSORS];
	uint32_t count;
	uint64_t cursor;
	FILE *file;
} SynthState;

static uint64_t synth_seed_mix(uint64_t seed,uint32_t a,uint32_t b,uint32_t c)
{
	uint64_t state;
	state = seed ^ (0x9e3779b97f4a7c15ull * (a + 1u)) ^ (0xbf58476d1ce4e5b9ull * (b + 1u)) ^ (0x94d049bb133111ebull * (c + 1u));
	state += 0x2545f4914f6cdd1dull;
	return(state);
}

static uint64_t synth_next(uint64_t *state)
{
	uint64_t value;
	value = (*state + 0x9e3779b97f4a7c15ull);
	*state = value;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
	return(value ^ (value >> 31));
}

static int synth_fill(SynthState *synth,uint32_t kind,uint32_t layer,uint32_t group,uint64_t bytes,uint64_t seed)
{
	uint8_t buffer[65536];
	uint64_t done,chunk,index;
	uint64_t state;
	if ( bytes == 0u )
		return(0);
	state = synth_seed_mix(seed,kind,layer,group);
	done = 0;
	while ( done < bytes )
	{
		chunk = bytes - done < sizeof(buffer) ? bytes - done : sizeof(buffer);
		for (index=0u; index<chunk; index++)
			buffer[index] = (uint8_t)(synth_next(&state) >> 24);
		if ( fwrite(buffer,1u,(size_t)chunk,synth->file) != chunk )
			return(-1);
		done += chunk;
	}
	return(0);
}

static int synth_align(SynthState *synth)
{
	static const uint8_t zero[SYNTH_ALIGN] = {0};
	uint64_t pad;
	pad = (0u - synth->cursor) % SYNTH_ALIGN;
	if ( pad != 0u && fwrite(zero,1u,(size_t)pad,synth->file) != pad )
		return(-1);
	synth->cursor += pad;
	return(0);
}

static uint64_t synth_plane_bytes(uint32_t kind,const SparkDsv41FlashStagePackTensorShape *shape,uint32_t plane)
{
	uint64_t per_group,width;
	(void)kind;
	if ( plane == 0u )
	{
		width = shape->payload_type == SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_BF16 ? 2u :
			shape->payload_type == SPARK_DSV41_FLASH_STAGEPACK_PAYLOAD_F32 ? 4u : 1u;
		per_group = (uint64_t)shape->rows * shape->columns * width;
	}
	else if ( shape->weight_codec == SPARK_WEIGHT_CODEC_MXFP4_E2M1 )
		per_group = (uint64_t)shape->rows * ((uint64_t)shape->columns * 2u / 32u);
	else if ( shape->scale_encoding == SPARK_WEIGHT_SCALE_ENCODING_E8M0 )
		per_group = (uint64_t)(shape->rows / 32u) * (shape->columns / 32u);
	else
		per_group = 0u;
	return(per_group * (shape->group_count > 1u ? shape->group_count : 1u));
}

static int synth_push(SynthState *synth,uint32_t kind,uint32_t layer_index,uint32_t expert_weight_codec,uint32_t tp_degree)
{
	SparkDsv41FlashStagePackTensorShape shape;
	SparkDsv41FlashStagePackEntry *entry;
	if ( SparkDsv41FlashStagePackExpectedShape(kind,expert_weight_codec,tp_degree,&shape) == 0u )
		return(-2);
	if ( synth->count == SYNTH_MAX_TENSORS )
		return(-3);
	entry = &synth->entries[synth->count];
	memset(entry,0,sizeof(*entry));
	entry->tensor_kind = kind;
	entry->layer_index = layer_index;
	entry->payload_type = shape.payload_type;
	entry->weight_codec = shape.weight_codec;
	entry->scale_encoding = shape.scale_encoding;
	entry->group_count = shape.group_count;
	entry->rows = shape.rows;
	entry->columns = shape.columns;
	entry->payload_bytes = synth_plane_bytes(kind,&shape,0u);
	entry->scale_bytes = shape.scale_encoding == SPARK_WEIGHT_SCALE_ENCODING_NONE ? 0u : synth_plane_bytes(kind,&shape,1u);
	synth->count += 1u;
	return(0);
}

static void synth_assign_offsets(SynthState *synth,uint64_t payload_base)
{
	SparkDsv41FlashStagePackEntry *entry;
	uint32_t index;
	synth->cursor = payload_base;
	for (index=0u; index<synth->count; index++)
	{
		entry = &synth->entries[index];
		synth_align(synth);
		entry->payload_offset = synth->cursor;
		synth->cursor += entry->payload_bytes;
		if ( entry->scale_bytes != 0u )
		{
			synth_align(synth);
			entry->scale_offset = synth->cursor;
			synth->cursor += entry->scale_bytes;
		}
	}
}

static int synth_emit(SynthState *synth,uint64_t seed)
{
	SparkDsv41FlashStagePackEntry *entry;
	uint64_t per_group,group,plane;
	uint32_t index;
	for (index=0u; index<synth->count; index++)
	{
		entry = &synth->entries[index];
		for (plane=0u; plane<2u; plane++)
		{
			uint64_t offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
			uint64_t bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
			if ( bytes == 0u )
				continue;
			if ( fseeko(synth->file,(off_t)offset,SEEK_SET) != 0 )
				return(-1);
			per_group = entry->group_count > 1u ? bytes / entry->group_count : bytes;
			for (group=0u; group<(entry->group_count > 1u ? entry->group_count : 1u); group++)
			{
				if ( synth_fill(synth,entry->tensor_kind + (uint32_t)(plane * 128u),entry->layer_index,(uint32_t)group,per_group,seed) != 0 )
					return(-1);
			}
		}
	}
	return(0);
}

static uint32_t synth_kind_layer(uint32_t kind,uint32_t layer_index,uint32_t tp_degree)
{
	if ( SparkDsv41FlashStagePackKindIsGlobal(kind) != 0u || SparkDsv41FlashStagePackKindIsRouted(kind) != 0u )
		return(1u);
	(void)tp_degree;
	return(SparkDsv41FlashStagePackKindInLayer(kind,layer_index));
}

static int synth_hex(const char *hex,uint8_t *out,uint32_t bytes)
{
	uint32_t i;
	if ( strlen(hex) != (size_t)bytes * 2u )
		return(-1);
	for (i=0u; i<bytes; i++)
	{
		uint8_t hi = (uint8_t)(hex[i * 2u] <= '9' ? hex[i * 2u] - '0' : hex[i * 2u] - 'a' + 10);
		uint8_t lo = (uint8_t)(hex[i * 2u + 1u] <= '9' ? hex[i * 2u + 1u] - '0' : hex[i * 2u + 1u] - 'a' + 10);
		out[i] = (uint8_t)((hi << 4) | lo);
	}
	return(0);
}

int main(int argc,char **argv)
{
	SparkDsv41FlashStagePackHeader header;
	SynthState synth;
	const char *out_path,*revision,*contract_hex,*config_hex,*recipe_hex;
	uint64_t seed,spine_bytes,expert_bytes;
	uint32_t tp_degree,tp_rank,codec,kind,layer,index,layer_count;
	if ( argc != 9 && argc != 10 )
	{
		(void)fprintf(stderr,"usage: %s OUT REVISION CONTRACT_SHA256 CONFIG_SHA256 RECIPE_SHA256 TP_DEGREE TP_RANK CODEC [LAYER_COUNT]\n",argv[0]);
		return(2);
	}
	out_path = argv[1];
	revision = argv[2];
	contract_hex = argv[3];
	config_hex = argv[4];
	recipe_hex = argv[5];
	tp_degree = (uint32_t)strtoul(argv[6],0,10);
	tp_rank = (uint32_t)strtoul(argv[7],0,10);
	if ( tp_degree == 0u || tp_rank >= tp_degree ||
		(SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u ||
		(SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT % tp_degree) != 0u )
	{
		(void)fprintf(stderr,"bad tp geometry\n");
		return(2);
	}
	codec = strcmp(argv[8],"fp8") == 0 ? SPARK_WEIGHT_CODEC_FP8_E4M3 : SPARK_WEIGHT_CODEC_MXFP4_E2M1;
	layer_count = argc == 10 ? (uint32_t)strtoul(argv[9],0,10) : SPARK_DSV41_FLASH_MODEL_LAYER_COUNT;
	if ( layer_count == 0u || layer_count > SPARK_DSV41_FLASH_MODEL_LAYER_COUNT )
	{
		(void)fprintf(stderr,"bad layer count\n");
		return(2);
	}
	seed = 0xd5411fa5ull;
	memset(&synth,0,sizeof(synth));
	synth.file = fopen(out_path,"wb+");
	if ( synth.file == 0 )
	{
		(void)fprintf(stderr,"cannot open %s\n",out_path);
		return(2);
	}
	synth.cursor = SPARK_DSV41_FLASH_STAGEPACK_HEADER_BYTES;
	if ( fseeko(synth.file,0,SEEK_SET) != 0 || fwrite(&header,sizeof(header),1u,synth.file) != 1u )
		return(2);
	memset(&header,0,sizeof(header));
	header.magic = SPARK_DSV41_FLASH_STAGEPACK_MAGIC;
	header.format_version = SPARK_DSV41_FLASH_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_DSV41_FLASH_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_DSV41_FLASH_STAGEPACK_ENTRY_BYTES;
	header.codec_abi_version = SPARK_WEIGHT_CODEC_ABI_VERSION;
	header.stage_count = 1u;
	header.stage_index = 0u;
	header.first_layer_index = 0u;
	header.layer_count = layer_count;
	header.total_layer_count = SPARK_DSV41_FLASH_MODEL_LAYER_COUNT;
	header.hidden_dimension = SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION;
	header.vocab_count = SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT;
	header.routed_expert_count = SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT;
	header.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	header.expert_weight_codec = codec;
	header.tp_degree = tp_degree;
	header.tp_rank = tp_rank;
	if ( strlen(revision) >= sizeof(header.model_revision) ||
		synth_hex(contract_hex,header.contract_sha256,32u) != 0 ||
		synth_hex(config_hex,header.source_config_sha256,32u) != 0 ||
		synth_hex(recipe_hex,header.pack_recipe_sha256,32u) != 0 )
	{
		(void)fprintf(stderr,"bad revision or hex digest\n");
		return(2);
	}
	memcpy(header.model_revision,revision,strlen(revision));
	for (kind=0u; kind<SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		if ( SparkDsv41FlashStagePackKindIsGlobal(kind) == 0u )
			continue;
		if ( synth_push(&synth,kind,SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER,codec,tp_degree) != 0 )
			return(2);
	}
	for (layer=0u; layer<layer_count; layer++)
	{
		for (kind=SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_NORM; kind<SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		{
			if ( SparkDsv41FlashStagePackKindIsGlobal(kind) != 0u )
				continue;
			if ( synth_kind_layer(kind,layer,tp_degree) == 0u )
				continue;
			if ( synth_push(&synth,kind,layer,codec,tp_degree) != 0 )
				return(2);
		}
	}
	header.tensor_count = synth.count;
	header.directory_offset = (SPARK_DSV41_FLASH_STAGEPACK_HEADER_BYTES + SYNTH_ALIGN - 1u) & ~((uint64_t)SYNTH_ALIGN - 1u);
	synth_assign_offsets(&synth,(header.directory_offset + (uint64_t)synth.count * SPARK_DSV41_FLASH_STAGEPACK_ENTRY_BYTES + SYNTH_ALIGN - 1u) & ~((uint64_t)SYNTH_ALIGN - 1u));
	header.file_bytes = synth.cursor;
	if ( fseeko(synth.file,(off_t)header.directory_offset,SEEK_SET) != 0 )
		return(2);
	for (index=0u; index<synth.count; index++)
	{
		if ( fwrite(&synth.entries[index],sizeof(SparkDsv41FlashStagePackEntry),1u,synth.file) != 1u )
			return(2);
	}
	if ( synth_emit(&synth,seed) != 0 )
		return(2);
	{
		static const uint8_t zero[SYNTH_ALIGN] = {0};
		uint64_t end,slice;
		if ( fseeko(synth.file,0,SEEK_END) != 0 )
			return(2);
		end = (uint64_t)ftello(synth.file);
		while ( end < header.file_bytes )
		{
			slice = header.file_bytes - end < sizeof(zero) ? header.file_bytes - end : sizeof(zero);
			if ( fwrite(zero,1u,(size_t)slice,synth.file) != slice )
				return(2);
			end += slice;
		}
	}
	if ( fseeko(synth.file,0,SEEK_SET) != 0 || fwrite(&header,sizeof(header),1u,synth.file) != 1u )
		return(2);
	if ( fclose(synth.file) != 0 )
		return(2);
	spine_bytes = 0u;
	expert_bytes = 0u;
	for (index=0u; index<synth.count; index++)
	{
		if ( SparkDsv41FlashStagePackKindIsExpert(synth.entries[index].tensor_kind) != 0u )
			expert_bytes += synth.entries[index].payload_bytes + synth.entries[index].scale_bytes;
		else
			spine_bytes += synth.entries[index].payload_bytes + synth.entries[index].scale_bytes;
	}
	(void)printf("tensors=%u file_bytes=%llu spine_bytes=%llu expert_bytes=%llu\n",
		synth.count,
		(unsigned long long)header.file_bytes,
		(unsigned long long)spine_bytes,
		(unsigned long long)expert_bytes);
	return(0);
}
