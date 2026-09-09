#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../modules/glm52_resident_decode_stage/source/spark_glm52_stagepack_format.h"

// Version-2 multi-range manifest for glm52 stage packs: payload plane
// always, scale plane when the codec carries one (F32 for FP8, none for
// BF16). Range kind = tensor kind * 2 + plane (0 payload, 1 scale),
// matching the glm5_next producer; records are the shared 48-byte v2
// wire format weightd's parser consumes.
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

static int32_t entry_write(FILE *pack,FILE *out,const SparkGlm52StagePackHeader *header,const SparkGlm52StagePackEntry *entry,uint32_t *count)
{
	uint64_t per,offset,bytes,directory_end;
	uint32_t plane,expert,kind;
	int32_t err;
	if ( entry->tensor_kind != SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE && entry->tensor_kind != SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN )
		return(0);
	if ( entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 && entry->weight_codec != SPARK_WEIGHT_CODEC_BF16 )
		return(-4);
	if ( entry->group_count != header->routed_expert_count || entry->payload_bytes == 0u )
		return(-5);
	if ( entry->weight_codec == SPARK_WEIGHT_CODEC_BF16 && (entry->scale_bytes != 0u || entry->scale_offset != 0u || entry->scale_encoding != SPARK_WEIGHT_SCALE_ENCODING_NONE) )
		return(-24);
	if ( entry->weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 && entry->scale_encoding != SPARK_WEIGHT_SCALE_ENCODING_F32 )
		return(-25);
	directory_end = (header->directory_offset + ((uint64_t)header->tensor_count * sizeof(*entry)));
	for (plane=0u; plane<2u; plane++)
	{
		offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
		bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
		if ( plane == 1u && entry->weight_codec == SPARK_WEIGHT_CODEC_BF16 && bytes == 0u )
			continue;
		if ( bytes == 0u || (bytes % entry->group_count) != 0u || offset < directory_end || offset > header->file_bytes || bytes > (header->file_bytes - offset) )
			return(-6);
		per = (bytes / entry->group_count);
		kind = ((entry->tensor_kind * 2u) + plane);
		for (expert=0u; expert<entry->group_count; expert++)
		{
			if ( *count == SPARK_WEIGHTD_RANGE_COUNT_MAX )
				return(-7);
			err = range_write(pack,out,entry->layer_index,expert,kind,(offset + ((uint64_t)expert * per)),per);
			if ( err < 0 )
				return(err);
			*count += 1u;
		}
	}
	return(0);
}

static int32_t manifest_write(FILE *pack,FILE *out,const SparkGlm52StagePackHeader *header)
{
	SparkGlm52StagePackEntry entry;
	uint32_t words[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,SPARK_WEIGHTD_RANGE_MANIFEST_VERSION,0u,0u};
	uint32_t i;
	uint32_t routed_layers = 0u;
	uint64_t up = 0u,down = 0u;
	int32_t err;
	for (i=0u; i<SPARK_GLM52_MODEL_LAYER_COUNT; i++)
		if ( SparkGlm52StagePackKindIsDense(SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP) == 0u && i >= SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER )
		{
			up |= (UINT64_C(1) << i);
			down |= (UINT64_C(1) << i);
			routed_layers++;
		}
	(void)routed_layers;
	if ( fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-8);
	for (i=0u; i<header->tensor_count; i++)
	{
		if ( fseeko(pack,(off_t)(header->directory_offset + ((uint64_t)i * sizeof(entry))),SEEK_SET) != 0 || fread(&entry,1u,sizeof(entry),pack) != sizeof(entry) )
			return(-9);
		if ( entry.tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE || entry.tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN )
		{
			if ( entry.layer_index >= 64u )
				return(-23);
			if ( entry.tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE )
				up &= ~(UINT64_C(1) << entry.layer_index);
			else
				down &= ~(UINT64_C(1) << entry.layer_index);
		}
		err = entry_write(pack,out,header,&entry,&words[2]);
		if ( err < 0 )
			return(err);
	}
	if ( up != 0u || down != 0u || words[2] == 0u || fseeko(out,0,SEEK_SET) != 0 || fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-10);
	return(0);
}

static int32_t header_read(FILE *pack,SparkGlm52StagePackHeader *header)
{
	struct stat st;
	uint64_t directory_bytes;
	if ( fstat(fileno(pack),&st) != 0 || st.st_size < 0 || fread(header,1u,sizeof(*header),pack) != sizeof(*header) )
		return(-11);
	if ( header->magic != SPARK_GLM52_STAGEPACK_MAGIC || header->format_version != SPARK_GLM52_STAGEPACK_FORMAT_VERSION || header->header_bytes != sizeof(*header) || header->directory_entry_bytes != sizeof(SparkGlm52StagePackEntry) )
		return(-12);
	if ( header->file_bytes != (uint64_t)st.st_size || header->routed_expert_count == 0u || header->tensor_count == 0u )
		return(-13);
	if ( header->first_layer_index != 0u || header->layer_count != SPARK_GLM52_MODEL_LAYER_COUNT )
		return(-26);
	directory_bytes = ((uint64_t)header->tensor_count * sizeof(SparkGlm52StagePackEntry));
	if ( header->directory_offset < sizeof(*header) || header->directory_offset > header->file_bytes || directory_bytes > (header->file_bytes - header->directory_offset) )
		return(-14);
	return(0);
}

static int32_t manifest_publish(FILE *pack,const SparkGlm52StagePackHeader *header,const char *path)
{
	SparkWeightdManifest manifest;
	char temporary[4096];
	FILE *out;
	int32_t fd,err,written;
	written = snprintf(temporary,sizeof(temporary),"%s.partial.XXXXXX",path);
	if ( written < 0 || (uint32_t)written >= sizeof(temporary) )
		return(-15);
	fd = mkstemp(temporary);
	if ( fd < 0 )
		return(-16);
	out = fdopen(fd,"wb");
	if ( out == 0 )
	{
		close(fd);
		unlink(temporary);
		return(-17);
	}
	err = manifest_write(pack,out,header);
	if ( fflush(out) != 0 && err == 0 )
		err = -18;
	if ( err == 0 && fsync(fd) != 0 )
		err = -19;
	if ( fclose(out) != 0 && err == 0 )
		err = -20;
	if ( err == 0 )
	{
		if ( SparkWeightdManifestLoad(temporary,header->file_bytes,&manifest) != SPARK_STATUS_OK )
			err = -21;
		else
			SparkWeightdManifestDestroy(&manifest);
	}
	if ( err == 0 && link(temporary,path) != 0 )
		err = -22;
	unlink(temporary);
	return(err);
}

int main(int argc,char **argv)
{
	SparkGlm52StagePackHeader header;
	FILE *pack;
	char path[4096];
	int32_t err,written;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s <verified-separate-plane-pack.glm52sp>\n",argv[0]);
		return(2);
	}
	written = snprintf(path,sizeof(path),"%s.experts",argv[1]);
	if ( written < 0 || (uint32_t)written >= sizeof(path) )
		return(3);
	pack = fopen(argv[1],"rb");
	if ( pack == 0 )
		return(4);
	err = header_read(pack,&header);
	if ( err == 0 )
		err = manifest_publish(pack,&header,path);
	fclose(pack);
	if ( err < 0 )
		fprintf(stderr,"expert manifest failed: error=%d (FP8/BF16 separate-plane packs only; existing output is preserved)\n",err);
	else
		printf("published %s version=2\n",path);
	return(err < 0 ? 1 : 0);
}
