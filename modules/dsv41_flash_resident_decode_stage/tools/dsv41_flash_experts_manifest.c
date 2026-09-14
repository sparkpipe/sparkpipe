#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include "../source/spark_dsv41_flash_stagepack_format.h"

static int32_t range_write(FILE *pack,FILE *out,uint32_t layer,uint32_t expert,uint32_t kind,uint64_t offset,uint64_t bytes)
{
	SparkCk128Context ck;
	uint8_t buffer[65536],record[48] = {0};
	uint64_t remaining,piece;
	if ( bytes == 0u || bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX || fseeko(pack,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	SparkCk128Initialize(&ck);
	remaining = bytes;
	while ( remaining != 0u )
	{
		piece = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
		if ( fread(buffer,1u,piece,pack) != piece )
			return(-2);
		SparkCk128Update(&ck,buffer,piece);
		remaining -= piece;
	}
	memcpy(record,&layer,4u);
	memcpy(record + 4u,&expert,4u);
	memcpy(record + 8u,&kind,4u);
	memcpy(record + 16u,&offset,8u);
	memcpy(record + 24u,&bytes,8u);
	SparkCk128Finalize(&ck,record + 32u);
	return(fwrite(record,1u,sizeof(record),out) == sizeof(record) ? 0 : -3);
}

static int32_t entry_write(FILE *pack,FILE *out,const SparkDsv41FlashStagePackHeader *header,const SparkDsv41FlashStagePackEntry *entry,uint32_t *count)
{
	uint64_t per,offset,bytes,directory_end;
	uint32_t plane,expert,kind;
	if ( SparkDsv41FlashStagePackKindIsExpert(entry->tensor_kind) == 0u )
		return(0);
	if ( entry->weight_codec != SPARK_WEIGHT_CODEC_MXFP4_E2M1 && entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
		return(-4);
	if ( entry->group_count != header->routed_expert_count / header->tp_degree || entry->payload_bytes == 0u )
		return(-5);
	if ( (entry->payload_bytes % entry->group_count) != 0u || (entry->scale_bytes % entry->group_count) != 0u )
		return(-6);
	directory_end = header->directory_offset + ((uint64_t)header->tensor_count * sizeof(*entry));
	for (plane=0u; plane<2u; plane++)
	{
		offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
		bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
		if ( bytes == 0u )
		{
			if ( plane == 0u )
				return(-7);
			continue;
		}
		if ( offset < directory_end || offset > header->file_bytes || bytes > (header->file_bytes - offset) )
			return(-8);
		per = bytes / entry->group_count;
		kind = ((entry->tensor_kind * 2u) + plane);
		for (expert=0u; expert<entry->group_count; expert++)
		{
			if ( *count == SPARK_WEIGHTD_RANGE_COUNT_MAX )
				return(-9);
			if ( range_write(pack,out,entry->layer_index,expert,kind,(offset + (expert * per)),per) < 0 )
				return(-10);
			*count += 1u;
		}
	}
	return(0);
}

static int32_t header_read(FILE *pack,SparkDsv41FlashStagePackHeader *header)
{
	struct stat st;
	uint64_t directory_bytes;
	if ( fstat(fileno(pack),&st) != 0 || st.st_size < 0 || fread(header,1u,sizeof(*header),pack) != sizeof(*header) )
		return(-11);
	if ( header->magic != SPARK_DSV41_FLASH_STAGEPACK_MAGIC || header->format_version != SPARK_DSV41_FLASH_STAGEPACK_FORMAT_VERSION || header->header_bytes != sizeof(*header) || header->directory_entry_bytes != sizeof(SparkDsv41FlashStagePackEntry) )
		return(-12);
	if ( header->file_bytes != (uint64_t)st.st_size || header->routed_expert_count == 0u || header->tensor_count == 0u )
		return(-13);
	if ( header->layer_count == 0u || header->first_layer_index + header->layer_count > SPARK_DSV41_FLASH_MODEL_LAYER_COUNT )
		return(-14);
	directory_bytes = ((uint64_t)header->tensor_count * sizeof(SparkDsv41FlashStagePackEntry));
	if ( header->directory_offset < sizeof(*header) || header->directory_offset > header->file_bytes || directory_bytes > (header->file_bytes - header->directory_offset) )
		return(-15);
	return(0);
}

static int32_t manifest_write(FILE *pack,FILE *out,const SparkDsv41FlashStagePackHeader *header)
{
	SparkDsv41FlashStagePackEntry entry;
	uint32_t words[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,SPARK_WEIGHTD_RANGE_MANIFEST_VERSION,0u,0u};
	uint32_t index;
	int32_t err;
	if ( fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-16);
	for (index=0u; index<header->tensor_count; index++)
	{
		if ( fseeko(pack,(off_t)(header->directory_offset + ((uint64_t)index * sizeof(entry))),SEEK_SET) != 0 || fread(&entry,1u,sizeof(entry),pack) != sizeof(entry) )
			return(-17);
		err = entry_write(pack,out,header,&entry,&words[2]);
		if ( err < 0 )
			return(err);
	}
	if ( words[2] == 0u || fseeko(out,0,SEEK_SET) != 0 || fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-18);
	return(0);
}

int main(int argc,char **argv)
{
	SparkDsv41FlashStagePackHeader header;
	char temporary[4096];
	FILE *pack,*out;
	int32_t fd,err,written;
	if ( argc != 3 )
	{
		(void)fprintf(stderr,"usage: %s PACK OUT_MANIFEST\n",argv[0]);
		return(2);
	}
	pack = fopen(argv[1],"rb");
	if ( pack == 0 )
	{
		(void)fprintf(stderr,"cannot open pack %s\n",argv[1]);
		return(2);
	}
	err = header_read(pack,&header);
	if ( err < 0 )
	{
		(void)fprintf(stderr,"bad pack header %d\n",err);
		return(2);
	}
	written = snprintf(temporary,sizeof(temporary),"%s.partial.XXXXXX",argv[2]);
	if ( written < 0 || (uint32_t)written >= sizeof(temporary) )
		return(2);
	fd = mkstemp(temporary);
	if ( fd < 0 )
		return(2);
	out = fdopen(fd,"wb");
	if ( out == 0 )
		return(2);
	err = manifest_write(pack,out,&header);
	if ( fclose(out) != 0 && err == 0 )
		err = -19;
	fclose(pack);
	if ( err < 0 )
	{
		(void)fprintf(stderr,"manifest write failed %d\n",err);
		(void)unlink(temporary);
		return(2);
	}
	if ( rename(temporary,argv[2]) != 0 )
	{
		(void)unlink(temporary);
		return(2);
	}
	(void)printf("manifest records written\n");
	return(0);
}
