#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include "spark_mimo26_rank_pack.h"

typedef struct RankPack
{
	FILE *file;
	uint64_t file_bytes;
	SparkMimo26StagePackHeader header;
	SparkMimo26StagePackEntry *entries;
	SparkMimo26RankPackLayout layout;
} RankPack;

static int32_t RankPackOpen(const char *path, uint32_t tp_degree, RankPack *pack)
{
	struct stat info;
	SparkStatus status;
	memset(pack,0,sizeof(*pack));
	pack->file = fopen(path,"rb");
	if ( pack->file == 0 || fstat(fileno(pack->file),&info) != 0 || info.st_size <= 0 )
	{
		fprintf(stderr,"FAIL cannot open pack %s\n",path);
		return(-1);
	}
	pack->file_bytes = (uint64_t)info.st_size;
	if ( fread(&pack->header,1u,sizeof(pack->header),pack->file) != sizeof(pack->header) || pack->header.tensor_count == 0u || pack->header.tensor_count > 65536u )
	{
		fprintf(stderr,"FAIL pack %s has no readable header\n",path);
		return(-2);
	}
	pack->entries = calloc(pack->header.tensor_count,sizeof(*pack->entries));
	if ( pack->entries == 0 || fseeko(pack->file,(off_t)pack->header.directory_offset,SEEK_SET) != 0 || fread(pack->entries,sizeof(*pack->entries),pack->header.tensor_count,pack->file) != pack->header.tensor_count )
	{
		fprintf(stderr,"FAIL pack %s has no readable directory\n",path);
		return(-3);
	}
	status = SparkMimo26RankPackBind(&pack->header,pack->entries,pack->header.tensor_count,pack->file_bytes,tp_degree,&pack->layout);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"FAIL pack %s does not bind as a tp%u rank pack: %s\n",path,tp_degree,SparkStatusToString(status));
		return(-4);
	}
	return(0);
}

static int32_t RangeWrite(FILE *pack, FILE *out, uint32_t layer, uint32_t expert, uint32_t kind, uint64_t offset, uint64_t bytes)
{
	static uint8_t buffer[1u << 20];
	SparkCk128Context ck;
	uint8_t record[48] = {0};
	uint64_t remaining, piece;
	if ( bytes == 0u || bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX || fseeko(pack,(off_t)offset,SEEK_SET) != 0 )
		return(-1);
	SparkCk128Initialize(&ck);
	for (remaining = bytes; remaining != 0u; remaining -= piece)
	{
		piece = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
		if ( fread(buffer,1u,piece,pack) != piece )
			return(-2);
		SparkCk128Update(&ck,buffer,piece);
	}
	memcpy(record,&layer,4u);
	memcpy(record + 4u,&expert,4u);
	memcpy(record + 8u,&kind,4u);
	memcpy(record + 16u,&offset,8u);
	memcpy(record + 24u,&bytes,8u);
	SparkCk128Finalize(&ck,record + 32u);
	return(fwrite(record,1u,sizeof(record),out) == sizeof(record) ? 0 : -3);
}

static int32_t ManifestWrite(RankPack *pack, FILE *out, uint32_t tp_degree)
{
	uint32_t words[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,SPARK_WEIGHTD_RANGE_MANIFEST_VERSION,0u,0u};
	uint32_t local_experts = SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT / tp_degree, layer, kind, plane, expert;
	const SparkMimo26StagePackEntry *entry;
	uint64_t offset, bytes;
	if ( fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-10);
	for (layer = 0u; layer < SPARK_MIMO26_MODEL_LAYER_COUNT; layer++)
		for (expert = 0u; expert < local_experts; expert++)
			for (kind = SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_GATE; kind <= SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_DOWN; kind++)
			{
				if ( pack->layout.layer_entry[layer][kind] == SPARK_MIMO26_RANK_PACK_ABSENT )
					continue;
				entry = &pack->entries[pack->layout.layer_entry[layer][kind]];
				for (plane = 0u; plane < 2u; plane++)
				{
					bytes = (plane == 0u ? entry->payload_bytes : entry->scale_bytes) / local_experts;
					offset = (plane == 0u ? entry->payload_offset : entry->scale_offset) + (uint64_t)expert * bytes;
					if ( words[2] == SPARK_WEIGHTD_RANGE_COUNT_MAX || RangeWrite(pack->file,out,layer,expert,kind * 2u + plane,offset,bytes) != 0 )
						return(-11);
					words[2]++;
				}
			}
	if ( words[2] == 0u || fseeko(out,0,SEEK_SET) != 0 || fwrite(words,1u,sizeof(words),out) != sizeof(words) )
		return(-12);
	return((int32_t)words[2]);
}

static void Summary(const RankPack *pack, uint32_t tp_degree)
{
	uint64_t expert_bytes = 0u;
	uint32_t index;
	for (index = 0u; index < pack->header.tensor_count; index++)
		if ( SparkMimo26StagePackIsExpert(pack->entries[index].tensor_kind) != 0u )
			expert_bytes += pack->entries[index].payload_bytes + pack->entries[index].scale_bytes;
	printf("PASS mimo26 tp%u rank pack binds: tensors=%u file_bytes=%llu expert_bytes=%llu spine_bytes=%llu\n",tp_degree,pack->header.tensor_count,(unsigned long long)pack->file_bytes,(unsigned long long)expert_bytes,(unsigned long long)(pack->file_bytes - expert_bytes));
}

int main(int argc, char **argv)
{
	char temporary[4096];
	RankPack pack;
	FILE *out;
	uint32_t tp_degree;
	int32_t fd, written, records;
	if ( argc < 4 || (strcmp(argv[1],"check") != 0 && strcmp(argv[1],"manifest") != 0) || (strcmp(argv[1],"manifest") == 0 && argc != 5) || (strcmp(argv[1],"check") == 0 && argc != 4) )
	{
		fprintf(stderr,"usage: %s check PACK TP_DEGREE | manifest PACK TP_DEGREE OUT\n",argv[0]);
		return(2);
	}
	tp_degree = (uint32_t)strtoul(argv[3],0,10);
	if ( RankPackOpen(argv[2],tp_degree,&pack) != 0 )
		return(1);
	if ( strcmp(argv[1],"check") == 0 )
	{
		Summary(&pack,tp_degree);
		return(0);
	}
	written = snprintf(temporary,sizeof(temporary),"%s.partial.XXXXXX",argv[4]);
	if ( written < 0 || (uint32_t)written >= sizeof(temporary) || (fd = mkstemp(temporary)) < 0 || (out = fdopen(fd,"wb")) == 0 )
		return(2);
	records = ManifestWrite(&pack,out,tp_degree);
	if ( fclose(out) != 0 && records > 0 )
		records = -13;
	if ( records <= 0 || rename(temporary,argv[4]) != 0 )
	{
		fprintf(stderr,"FAIL manifest write %d\n",records);
		(void)unlink(temporary);
		return(1);
	}
	Summary(&pack,tp_degree);
	printf("PASS manifest %s records=%d\n",argv[4],records);
	return(0);
}
