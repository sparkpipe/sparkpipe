#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DSV4_MANIFEST_EXPERT_KINDS 3u
#define DSV4_MANIFEST_PLANES 2u

static uint32_t manifest_range_kind(uint32_t tensor_kind,uint32_t plane)
{
	return((tensor_kind * 2u) + plane);
}

static int32_t range_write(FILE *pack,FILE *out,uint32_t layer,uint32_t expert,uint32_t kind,uint64_t offset,uint64_t bytes)
{
	SparkCk128Context ck;
	uint8_t buffer[65536];
	uint8_t record[48] = {0};
	uint64_t remaining;
	size_t piece;
	if ( bytes == 0u || bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX || fseeko(pack,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	SparkCk128Initialize(&ck);
	remaining = bytes;
	while ( remaining != 0u )
	{
		piece = remaining < sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
		if ( fread(buffer,1u,piece,pack) != piece )
			return(-2);
		SparkCk128Update(&ck,buffer,piece);
		remaining -= piece;
	}
	SparkCk128Finalize(&ck,record + 32u);
	memcpy(record,&layer,4u);
	memcpy(record + 4u,&expert,4u);
	memcpy(record + 8u,&kind,4u);
	memcpy(record + 16u,&offset,8u);
	memcpy(record + 24u,&bytes,8u);
	return(fwrite(record,1u,sizeof(record),out) == sizeof(record) ? 0 : -3);
}

static int32_t entry_write(FILE *pack,FILE *out,const SparkDsv4StagePackHeader *header,const SparkDsv4StagePackEntry *entry,uint32_t *count)
{
	uint64_t bytes,offset,per;
	uint32_t expert,plane;
	int32_t err;
	if ( entry->tensor_kind != SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1 &&
		entry->tensor_kind != SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2 &&
		entry->tensor_kind != SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3 )
		return(0);
	if ( entry->layer_index < header->first_layer_index ||
		entry->layer_index >= (header->first_layer_index + header->layer_count) )
		return(0);
	if ( (entry->rows % header->routed_expert_count) != 0u )
		return(-4);
	for (plane=0u; plane<DSV4_MANIFEST_PLANES; plane++)
	{
		bytes = plane == 0u ?
			SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns) :
			SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
		offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
		if ( plane == 1u && bytes == 0u )
			continue;
		if ( bytes == 0u || (bytes % header->routed_expert_count) != 0u ||
			offset > header->file_bytes || bytes > (header->file_bytes - offset) )
			return(-5);
		per = (bytes / header->routed_expert_count);
		for (expert=0u; expert<header->routed_expert_count; expert++)
		{
			if ( *count == SPARK_WEIGHTD_RANGE_COUNT_MAX )
				return(-6);
			err = range_write(pack,out,entry->layer_index,expert,
				manifest_range_kind(entry->tensor_kind,plane),
				(offset + ((uint64_t)expert * per)),per);
			if ( err < 0 )
				return(err);
			*count += 1u;
		}
	}
	return(0);
}

static int32_t header_read(FILE *pack,SparkDsv4StagePackHeader *header)
{
	struct stat st;
	uint64_t directory_bytes;
	if ( fstat(fileno(pack),&st) != 0 || st.st_size < 0 ||
		fread(header,1u,sizeof(*header),pack) != sizeof(*header) )
		return(-7);
	if ( header->magic != SPARK_DSV4_STAGEPACK_MAGIC ||
		header->format_version != SPARK_DSV4_STAGEPACK_FORMAT_VERSION ||
		header->header_bytes != SPARK_DSV4_STAGEPACK_HEADER_BYTES ||
		header->directory_entry_bytes != SPARK_DSV4_STAGEPACK_ENTRY_BYTES )
		return(-8);
	if ( header->file_bytes != (uint64_t)st.st_size || header->tensor_count == 0u ||
		header->layer_count == 0u )
		return(-9);
	if ( header->hidden_dimension != SPARK_DSV4_MODEL_HIDDEN_DIMENSION ||
		header->vocab_count != SPARK_DSV4_MODEL_VOCAB_COUNT )
		return(-10);
	if ( header->routed_expert_count != SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT ||
		header->total_layer_count != SPARK_DSV4_MODEL_LAYER_COUNT )
		return(-11);
	if ( header->first_layer_index > SPARK_DSV4_MODEL_LAYER_COUNT ||
		header->layer_count > (SPARK_DSV4_MODEL_LAYER_COUNT - header->first_layer_index) )
		return(-12);
	directory_bytes = ((uint64_t)header->tensor_count * SPARK_DSV4_STAGEPACK_ENTRY_BYTES);
	if ( header->directory_offset < SPARK_DSV4_STAGEPACK_HEADER_BYTES ||
		header->directory_offset > header->file_bytes ||
		directory_bytes > (header->file_bytes - header->directory_offset) )
		return(-13);
	return(0);
}

static int32_t coverage_check(const SparkDsv4StagePackHeader *header,const uint8_t *covered)
{
	uint32_t i,expected = 0u;
	for (i=0u; i<DSV4_MANIFEST_EXPERT_KINDS; i++)
		expected |= (uint32_t)(1u << i);
	for (i=header->first_layer_index; i<(header->first_layer_index + header->layer_count); i++)
		if ( covered[i] != expected )
			return(-14);
	return(0);
}

static int32_t manifest_write(FILE *pack,FILE *out,const SparkDsv4StagePackHeader *header)
{
	SparkDsv4StagePackEntry entry;
	uint8_t *directory;
	uint8_t covered[SPARK_DSV4_MODEL_LAYER_COUNT] = {0};
	uint32_t words[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,SPARK_WEIGHTD_RANGE_MANIFEST_VERSION,0u,0u};
	uint32_t i;
	int32_t err;
	if ( (((uint64_t)header->layer_count * DSV4_MANIFEST_EXPERT_KINDS * DSV4_MANIFEST_PLANES) *
		header->routed_expert_count) > SPARK_WEIGHTD_RANGE_COUNT_MAX )
		return(-15);
	if ( fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-16);
	directory = malloc((size_t)header->tensor_count * SPARK_DSV4_STAGEPACK_ENTRY_BYTES);
	if ( directory == 0 )
		return(-17);
	if ( fseeko(pack,(off_t)header->directory_offset,SEEK_SET) != 0 ||
		fread(directory,1u,(size_t)header->tensor_count * SPARK_DSV4_STAGEPACK_ENTRY_BYTES,pack) !=
			((size_t)header->tensor_count * SPARK_DSV4_STAGEPACK_ENTRY_BYTES) )
	{
		free(directory);
		return(-17);
	}
	for (i=0u; i<header->tensor_count; i++)
	{
		memcpy(&entry,directory + ((size_t)i * SPARK_DSV4_STAGEPACK_ENTRY_BYTES),sizeof(entry));
		err = entry_write(pack,out,header,&entry,&words[2]);
		if ( err < 0 )
		{
			free(directory);
			return(err);
		}
		if ( entry.tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1 &&
			entry.tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3 &&
			entry.layer_index < SPARK_DSV4_MODEL_LAYER_COUNT )
			covered[entry.layer_index] |=
				(uint8_t)(1u << (entry.tensor_kind - SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1));
	}
	free(directory);
	err = coverage_check(header,covered);
	if ( err < 0 || words[2] == 0u )
		return(err < 0 ? err : -18);
	if ( fseeko(out,0,SEEK_SET) != 0 ||
		fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-18);
	return(0);
}

static int32_t manifest_publish(FILE *pack,const char *path)
{
	SparkDsv4StagePackHeader header;
	SparkWeightdManifest manifest;
	char temporary[4096];
	char final_path[4096];
	FILE *out;
	int32_t fd,err,written;
	written = snprintf(temporary,sizeof(temporary),"%s.partial.XXXXXX",path);
	if ( written < 0 || (uint32_t)written >= sizeof(temporary) )
		return(-19);
	fd = mkstemp(temporary);
	if ( fd < 0 )
		return(-20);
	out = fdopen(fd,"wb");
	if ( out == 0 )
	{
		close(fd);
		unlink(temporary);
		return(-21);
	}
	err = header_read(pack,&header);
	if ( err == 0 )
		err = manifest_write(pack,out,&header);
	if ( fflush(out) != 0 && err == 0 )
		err = -22;
	if ( err == 0 && fsync(fd) != 0 )
		err = -22;
	if ( fclose(out) != 0 && err == 0 )
		err = -22;
	if ( err == 0 && SparkWeightdManifestLoad(temporary,header.file_bytes,&manifest) != SPARK_STATUS_OK )
		err = -23;
	if ( err == 0 )
		SparkWeightdManifestDestroy(&manifest);
	if ( err == 0 )
	{
		written = snprintf(final_path,sizeof(final_path),"%s.experts",path);
		if ( written < 0 || (uint32_t)written >= sizeof(final_path) )
			err = -19;
		else if ( rename(temporary,final_path) != 0 )
			err = -24;
	}
	if ( err != 0 )
		unlink(temporary);
	return(err);
}

int main(int argc,char **argv)
{
	FILE *pack;
	int32_t err;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s <pack.spstage> (writes <pack>.experts)\n",argv[0]);
		return(2);
	}
	pack = fopen(argv[1],"rb");
	if ( pack == 0 )
	{
		perror(argv[1]);
		return(1);
	}
	err = manifest_publish(pack,argv[1]);
	fclose(pack);
	if ( err < 0 )
		fprintf(stderr,"dsv4 experts manifest failed: error=%d\n",err);
	else
		printf("published %s.experts\n",argv[1]);
	return(err < 0 ? 1 : 0);
}
