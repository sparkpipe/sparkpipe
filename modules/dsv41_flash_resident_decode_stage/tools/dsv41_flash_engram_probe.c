#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../source/spark_dsv41_flash_engram_access.h"

#define PROBE_MAGIC UINT32_C(0x46574745)
#define PROBE_COLS SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS
#define PROBE_HEAD_DIM SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM
#define PROBE_SCALE_COLS SPARK_DSV41_FLASH_ENGRAM_ACCESS_SCALE_COLS
#define PROBE_ROW_BYTES SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES
#define PROBE_LAYERS SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS
#define PROBE_MULTS SPARK_DSV41_FLASH_ENGRAM_ACCESS_MULTS
#define PROBE_TIMEOUT_NS UINT64_C(300000000)
#define PROBE_WKV_RECORD_BYTES \
	(UINT64_C(25600) * UINT64_C(6144) + UINT64_C(800) * UINT64_C(192))
#define PROBE_GATE_RECORD_BYTES (UINT64_C(4) * UINT64_C(5120) * 2u)

static const int32_t g_tokens[15] = {11,900,5,12999,42,7,12345,8000,3133,64,1,0,999,2048,777};
static const uint32_t g_positions[3] = {0u,1u,130u};

typedef struct ProbeFixture
{
	uint32_t layer_count,record_count;
	uint64_t entries[PROBE_LAYERS];
	int64_t mult[PROBE_LAYERS][PROBE_MULTS];
	uint8_t *data;
} ProbeFixture;

static int ProbeFixtureLoad(const char *path,ProbeFixture *fixture)
{
	FILE *file;
	uint32_t magic,version,i,layer,pad;
	uint64_t size;
	uint8_t *p;
	file = fopen(path,"rb");
	if ( file == 0 )
		return 1;
	(void)fseek(file,0,SEEK_END);
	size = (uint64_t)ftell(file);
	(void)fseek(file,0,SEEK_SET);
	fixture->data = (uint8_t *)malloc(size);
	if ( fixture->data == 0 || fread(fixture->data,1,size,file) != size )
	{
		(void)fclose(file);
		return 1;
	}
	(void)fclose(file);
	p = fixture->data;
	memcpy(&magic,p,4u);
	memcpy(&version,p + 4u,4u);
	memcpy(&fixture->layer_count,p + 8u,4u);
	if ( magic != PROBE_MAGIC || version != 1u || fixture->layer_count != PROBE_LAYERS )
		return 1;
	p += 12u;
	for (i = 0u; i < fixture->layer_count; i++)
	{
		memcpy(&layer,p,4u);
		memcpy(&pad,p + 4u,4u);
		memcpy(&fixture->entries[i],p + 8u,8u);
		if ( layer != i + 1u || pad != 0u )
			return 1;
		p += 24u;
		memcpy(fixture->mult[i],p,PROBE_MULTS * sizeof(int64_t));
		p += PROBE_MULTS * sizeof(int64_t) + PROBE_COLS * 2u * sizeof(int64_t);
	}
	memcpy(&fixture->record_count,p,4u);
	return 0;
}

static const uint8_t *ProbeFixtureRow(const ProbeFixture *fixture,
	uint32_t layer,uint64_t id)
{
	uint8_t *p = fixture->data + 12u;
	uint32_t i;
	uint64_t bytes;
	for (i = 0u; i < fixture->layer_count; i++)
		p += 24u + PROBE_MULTS * sizeof(int64_t) + PROBE_COLS * 2u * sizeof(int64_t);
	for (i = 0u; i < fixture->record_count; i++)
	{
		uint32_t kind,record_layer,pad;
		uint64_t record_id;
		memcpy(&kind,p,4u);
		memcpy(&record_layer,p + 4u,4u);
		memcpy(&pad,p + 8u,4u);
		memcpy(&record_id,p + 16u,8u);
		p += 24u;
		if ( kind == 0u )
			bytes = PROBE_HEAD_DIM + PROBE_SCALE_COLS;
		else if ( kind == 1u )
			bytes = PROBE_WKV_RECORD_BYTES;
		else
			bytes = PROBE_GATE_RECORD_BYTES;
		if ( kind == 0u && record_layer == layer && record_id == id )
			return p;
		p += bytes;
	}
	return 0;
}

static uint64_t ProbeArgU64(const char *text)
{
	return strtoull(text,0,10);
}

int main(int argc,char **argv)
{
	SparkDsv41FlashEngramAccess access;
	ProbeFixture fixture;
	int64_t mult[PROBE_LAYERS * PROBE_MULTS];
	int64_t ids[PROBE_COLS];
	uint8_t rows[PROBE_COLS * PROBE_ROW_BYTES];
	const uint8_t *want,*staged;
	uint64_t file_bytes,matched;
	uint32_t layer,pos,col,rank,i;
	int failures = 0,owned;
	SparkStatus status;
	if ( argc != 7 )
	{
		(void)fprintf(stderr,"usage: %s <shard> <sha256> <socket> <mesh_rank> <file_bytes> <fixture> <contract_sha>\n",
		    argv[0]);
		return 2;
	}
	if ( ProbeFixtureLoad(argv[6u],&fixture) != 0 )
	{
		(void)fprintf(stderr,"FAIL bad fixture %s\n",argv[6u]);
		return 2;
	}
	for (i = 0u; i < PROBE_LAYERS; i++)
		for (col = 0u; col < PROBE_MULTS; col++)
			mult[i * PROBE_MULTS + col] = fixture.mult[i][col];
	file_bytes = ProbeArgU64(argv[5u]);
	rank = (uint32_t)strtoul(argv[4u],0,10);
	status = SparkDsv41FlashEngramAccessOpen(&access,argv[3u],argv[1u],argv[2u],
	    file_bytes,rank,SPARK_DSV41_FLASH_ENGRAM_ACCESS_MESH_RANKS,mult);
	if ( status != SPARK_STATUS_OK )
	{
		(void)fprintf(stderr,"FAIL open status=%d\n",(int)status);
		return 2;
	}
	matched = 0;
	for (layer = 0u; layer < PROBE_LAYERS; layer++)
	{
		for (pos = 0u; pos < 3u; pos++)
		{
			status = SparkDsv41FlashEngramAccessIds(&access,layer,g_tokens,15u,
			    g_positions[pos],ids);
			if ( status != SPARK_STATUS_OK )
			{
				(void)fprintf(stderr,"FAIL ids layer=%u pos=%u status=%d\n",layer,
				    g_positions[pos],(int)status);
				failures++;
				continue;
			}
			owned = 0;
			for (col = 0u; col < PROBE_COLS; col++)
			{
				if ( ids[col] < 0 || (uint64_t)ids[col] >= fixture.entries[layer] )
				{
					(void)fprintf(stderr,"FAIL id range layer=%u pos=%u col=%u\n",layer,
					    g_positions[pos],col);
					failures++;
					continue;
				}
				if ( (uint64_t)ids[col] / ((fixture.entries[layer] + 15u) / 16u) != rank )
					continue;
				owned = 1;
			}
			if ( owned != 0 )
			{
				status = SparkDsv41FlashEngramAccessPublish(&access,layer,ids,
				    UINT64_C(1000) + layer * 131u + g_positions[pos]);
				if ( status != SPARK_STATUS_OK )
				{
					(void)fprintf(stderr,"FAIL publish layer=%u pos=%u status=%d\n",layer,
					    g_positions[pos],(int)status);
					failures++;
				}
			}
			for (col = 0u; col < PROBE_COLS; col++)
			{
				if ( ids[col] < 0 || (uint64_t)ids[col] >= fixture.entries[layer] ||
					(uint64_t)ids[col] / ((fixture.entries[layer] + 15u) / 16u) != rank )
					continue;
				want = ProbeFixtureRow(&fixture,layer + 1u,(uint64_t)ids[col]);
				staged = access.mesh + (uint64_t)layer * PROBE_COLS * PROBE_ROW_BYTES +
				    (uint64_t)col * PROBE_ROW_BYTES;
				if ( want == 0 || memcmp(staged,want,PROBE_ROW_BYTES) != 0 )
				{
					(void)fprintf(stderr,"FAIL staged row layer=%u pos=%u col=%u id=%lld\n",
					    layer,g_positions[pos],col,(long long)ids[col]);
					failures++;
				}
				else
					matched++;
			}
			memset(rows,0,sizeof(rows));
			status = SparkDsv41FlashEngramAccessRows(&access,layer,ids,
			    UINT64_C(2000) + layer * 131u + g_positions[pos],PROBE_TIMEOUT_NS,rows);
			if ( status != SPARK_STATUS_BUSY )
			{
				(void)fprintf(stderr,"FAIL rows-busy layer=%u pos=%u status=%d\n",layer,
				    g_positions[pos],(int)status);
				failures++;
			}
			for (col = 0u; col < PROBE_COLS * PROBE_ROW_BYTES; col++)
				if ( rows[col] != 0u )
				{
					(void)fprintf(stderr,"FAIL rows partial fill layer=%u pos=%u\n",layer,
					    g_positions[pos]);
					failures++;
					break;
				}
			ids[0] = (int64_t)fixture.entries[layer];
			status = SparkDsv41FlashEngramAccessRows(&access,layer,ids,
			    UINT64_C(3000) + layer * 131u + g_positions[pos],PROBE_TIMEOUT_NS,rows);
			if ( status != SPARK_STATUS_SCHEMA_ERROR )
			{
				(void)fprintf(stderr,"FAIL id fail-closed layer=%u pos=%u status=%d\n",
				    layer,g_positions[pos],(int)status);
				failures++;
			}
		}
	}
	SparkDsv41FlashEngramAccessClose(&access);
	free(fixture.data);
	printf("%s engram probe rank=%u matched=%lu failures=%d\n",
	    failures == 0 ? "PASS" : "FAIL",rank,(unsigned long)matched,failures);
	return(failures == 0 ? 0 : 1);
}
