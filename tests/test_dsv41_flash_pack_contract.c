#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_ck128.h"

#include "../source/spark_dsv41_flash_stagepack_format.h"

#define CHECK(condition,message) \
	do { \
		if ( !(condition) ) \
		{ \
			(void)fprintf(stderr,"FAIL %s at %d: %s\n",message,(int)__LINE__,#condition); \
			return(2); \
		} \
	} while (0)


static int32_t read_exact(FILE *file,uint64_t offset,void *buffer,uint64_t bytes)
{
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	if ( fread(buffer,1u,(size_t)bytes,file) != bytes )
		return(-2);
	return(0);
}

static uint64_t literal_layer_kind_bits(uint32_t layer_index)
{
	uint64_t bits;
	uint32_t kind;
	bits = 0u;
	for (kind=SPARK_DSV41_FLASH_STAGEPACK_TENSOR_ATTN_NORM; kind<=SPARK_DSV41_FLASH_STAGEPACK_TENSOR_O_B; kind++)
		bits |= UINT64_C(1) << kind;
	for (kind=SPARK_DSV41_FLASH_STAGEPACK_TENSOR_HC_ATTN_FN; kind<=SPARK_DSV41_FLASH_STAGEPACK_TENSOR_SHARED_W3; kind++)
		bits |= UINT64_C(1) << kind;
	if ( layer_index == 2u || layer_index == 8u || layer_index == 14u || layer_index == 20u ||
		layer_index == 24u || layer_index == 28u || layer_index == 32u || layer_index == 36u )
	{
		bits |= (UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_Q_B) |
			(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WEIGHTS_PROJ);
	}
	if ( layer_index == 2u || layer_index == 8u || layer_index == 14u || layer_index == 20u )
	{
		bits |= (UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_WK) |
			(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM) |
			(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WKV) |
			(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_NORM);
	}
	if ( layer_index == 2u || layer_index == 8u || layer_index == 14u )
		bits |= UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_COMPRESSOR_WGATE;
	return(bits);
}

static uint64_t kind_map_layer_bits(uint32_t layer_index)
{
	uint64_t bits;
	uint32_t kind;
	bits = 0u;
	for (kind=0u; kind<SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		if ( SparkDsv41FlashStagePackKindIsGlobal(kind) != 0u )
			continue;
		if ( SparkDsv41FlashStagePackKindInLayer(kind,layer_index) != 0u )
			bits |= UINT64_C(1) << kind;
	}
	return(bits);
}

static uint32_t literal_bit_count(uint64_t value)
{
	uint32_t count;
	count = 0u;
	while ( value != 0u )
	{
		count += (uint32_t)(value & 1u);
		value >>= 1;
	}
	return(count);
}

int main(int argc,char **argv)
{
	SparkDsv41FlashStagePackHeader header;
	SparkDsv41FlashStagePackEntry entry;
	SparkCk128Context ck;
	struct stat st;
	FILE *pack,*manifest;
	uint8_t record[48],buffer[4096];
	uint64_t index,done,piece,offset,bytes,global_seen;
	uint64_t *seen;
	uint32_t layer,kinds_seen;
	uint8_t digest[16];
	if ( argc != 3 )
	{
		(void)fprintf(stderr,"usage: %s PACK MANIFEST\n",argv[0]);
		return(2);
	}
	pack = fopen(argv[1],"rb");
	manifest = fopen(argv[2],"rb");
	if ( pack == 0 || manifest == 0 )
	{
		(void)fprintf(stderr,"cannot open inputs\n");
		return(2);
	}
	CHECK(stat(argv[1],&st) == 0 && st.st_size > 0,"pack stat");
	CHECK(fread(&header,sizeof(header),1u,pack) == 1u,"header read");
	CHECK(header.magic == 0x31413444u,"magic literal");
	CHECK(header.format_version == 1u,"format version");
	CHECK(header.header_bytes == 257u,"header bytes literal");
	CHECK(header.directory_entry_bytes == 64u,"entry bytes literal");
	CHECK(header.hidden_dimension == 5120u,"hidden literal");
	CHECK(header.vocab_count == 129280u,"vocab literal");
	CHECK(header.routed_expert_count == 384u,"routed literal");
	CHECK(header.total_layer_count == 40u,"layers literal");
	CHECK(header.file_bytes == (uint64_t)st.st_size,"file bytes equal size");
	CHECK(header.directory_offset >= header.header_bytes,"directory after header");
	CHECK(header.directory_offset % 256u == 0u,"directory aligned");
	kinds_seen = 0u;
	for (layer=0u; layer<header.layer_count; layer++)
		kinds_seen += literal_bit_count(literal_layer_kind_bits(layer));
	CHECK(header.tensor_count == 3u + kinds_seen,"tensor count literal");
	seen = (uint64_t *)calloc(header.layer_count,sizeof(uint64_t));
	CHECK(seen != 0,"census alloc");
	global_seen = 0u;
	CHECK(fseeko(pack,(off_t)header.directory_offset,SEEK_SET) == 0,"directory seek");
	for (index=0u; index<header.tensor_count; index++)
	{
		CHECK(fread(&entry,sizeof(entry),1u,pack) == 1u,"entry read");
		CHECK(entry.tensor_kind < SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT,"entry kind valid");
		if ( SparkDsv41FlashStagePackKindIsGlobal(entry.tensor_kind) != 0u )
		{
			CHECK(entry.layer_index == SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER,"global layer literal");
			CHECK((global_seen & (UINT64_C(1) << entry.tensor_kind)) == 0u,"global kind unique");
			global_seen |= UINT64_C(1) << entry.tensor_kind;
			continue;
		}
		CHECK(entry.layer_index < header.layer_count,"entry layer in range");
		CHECK((seen[entry.layer_index] & (UINT64_C(1) << entry.tensor_kind)) == 0u,"layer kind unique");
		seen[entry.layer_index] |= UINT64_C(1) << entry.tensor_kind;
	}
	CHECK(global_seen == ((UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING) |
		(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM) |
		(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD)),"global kinds literal");
	for (layer=0u; layer<header.layer_count; layer++)
	{
		CHECK(seen[layer] == literal_layer_kind_bits(layer),"packer census literal");
		CHECK(kind_map_layer_bits(layer) == literal_layer_kind_bits(layer),"kind map literal");
	}
	free(seen);
	CHECK(fseeko(manifest,0,SEEK_END) == 0,"manifest seek");
	{
		uint64_t manifest_bytes = (uint64_t)ftello(manifest);
		uint64_t records = (manifest_bytes - 16u) / 48u;
		CHECK(manifest_bytes >= 16u && (manifest_bytes - 16u) % 48u == 0u,"manifest size multiple");
		CHECK(records == 2u * (384u / header.tp_degree) * 3u * header.layer_count,"manifest record count");
		CHECK(fseeko(manifest,16u,SEEK_SET) == 0,"manifest rewind");
		for (index=0u; index<records; index++)
		{
			CHECK(fread(record,1u,sizeof(record),manifest) == sizeof(record),"record read");
			memcpy(&layer,record,4u);
			offset = 0u;
			bytes = 0u;
			memcpy(&offset,record + 16u,8u);
			memcpy(&bytes,record + 24u,8u);
			CHECK(layer < header.layer_count,"record layer in range");
			CHECK(offset >= header.directory_offset + (uint64_t)header.tensor_count * 64u,"record offset after directory");
			CHECK(bytes != 0u && offset <= header.file_bytes && bytes <= header.file_bytes - offset,"record range in pack");
			CHECK(read_exact(pack,offset,buffer,bytes < sizeof(buffer) ? bytes : sizeof(buffer)) == 0,"record bytes readable");
			SparkCk128Initialize(&ck);
			done = 0u;
			while ( done < bytes )
			{
				piece = bytes - done < sizeof(buffer) ? bytes - done : sizeof(buffer);
				CHECK(read_exact(pack,offset + done,buffer,piece) == 0,"ck window read");
				SparkCk128Update(&ck,buffer,piece);
				done += piece;
			}
			SparkCk128Finalize(&ck,digest);
			CHECK(memcmp(digest,record + 32u,16u) == 0,"record ck128 digest");
		}
	}
	(void)fclose(pack);
	(void)fclose(manifest);
	(void)printf("dsv41 host contract validator PASS\n");
	return(0);
}
