#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "spark_dsv41_flash_engram_access.h"

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_weightd_attach.h"

#define SPARK_DSV41_FLASH_ENGRAM_HEADER_BYTES 257u
#define SPARK_DSV41_FLASH_ENGRAM_DIRECTORY_OFFSET 512u
#define SPARK_DSV41_FLASH_ENGRAM_ENTRY_BYTES 64u
#define SPARK_DSV41_FLASH_ENGRAM_KIND_COUNT 8u
#define SPARK_DSV41_FLASH_ENGRAM_MAGIC UINT32_C(0x31474544)
#define SPARK_DSV41_FLASH_ENGRAM_FORMAT_VERSION 1u
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_KIND_EMBED_0 0u
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_KIND_EMBED_1 4u
#define SPARK_DSV41_FLASH_ENGRAM_MODEL_TAG "dsv41_flash_engram"
#define SPARK_DSV41_FLASH_ENGRAM_SPINE_BUDGET_BYTES (384ull * 1024ull * 1024ull)
#define SPARK_DSV41_FLASH_ENGRAM_POOL_BYTES (256ull * 1024ull * 1024ull)
#define SPARK_DSV41_FLASH_ENGRAM_REGION_NEEDS \
	(SPARK_DSV41_FLASH_ENGRAM_ACCESS_STAGING_BYTES + \
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS * \
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS * 8u + 64u)
#define SPARK_DSV41_FLASH_ENGRAM_ROWS_OUT_BYTES \
	(SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS * SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES)

typedef struct SparkDsv41FlashEngramDirEntry
{
	uint32_t kind;
	uint32_t layer;
	uint32_t payload_type;
	uint32_t codec;
	uint32_t scale_encoding;
	uint32_t groups;
	uint32_t rows;
	uint32_t cols;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} SparkDsv41FlashEngramDirEntry;

typedef struct SparkDsv41FlashEngramLease
{
	uint64_t identifier;
	const uint8_t *base;
	uint64_t local[SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS];
} SparkDsv41FlashEngramLease;

static uint64_t SparkDsv41FlashEngramNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return 0;
	return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
	    (uint64_t)now.tv_nsec / UINT64_C(1000);
}

static uint32_t SparkDsv41FlashEngramPrimeOk(int64_t value,
	const SparkDsv41FlashEngramAccess *access,uint32_t layer_index,
	uint32_t drawn)
{
	uint64_t probe;
	uint32_t prior,index,limit;
	if ( value < 3 || value % 2 == 0 )
		return 0u;
	for (probe = 3u; probe * probe <= (uint64_t)value; probe += 2u)
		if ( value % (int64_t)probe == 0 )
			return 0u;
	for (prior = 0u; prior <= layer_index; prior++)
	{
		limit = prior < layer_index ? SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS : drawn;
		for (index = 0u; index < limit; index++)
			if ( access->layer_primes[prior][index] == value )
				return 0u;
	}
	return 1u;
}

static int64_t SparkDsv41FlashEngramNextPrime(int64_t candidate,
	const SparkDsv41FlashEngramAccess *access,uint32_t layer_index,
	uint32_t drawn)
{
	int64_t value = candidate;
	do
	{
		value++;
	} while ( SparkDsv41FlashEngramPrimeOk(value,access,layer_index,drawn) == 0u );
	return(value);
}

static void SparkDsv41FlashEngramDerivePartition(SparkDsv41FlashEngramAccess *access)
{
	uint64_t sum;
	uint32_t layer,index;
	int64_t candidate = SPARK_DSV41_FLASH_ENGRAM_ACCESS_PRIME_START;
	for (layer = 0u; layer < SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS; layer++)
	{
		sum = 0;
		for (index = 0u; index < SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS; index++)
		{
			candidate = SparkDsv41FlashEngramNextPrime(candidate,access,layer,index);
			access->layer_primes[layer][index] = candidate;
			access->layer_offsets[layer][index] = (int64_t)sum;
			sum += (uint64_t)candidate;
		}
		access->layer_entries[layer] = sum;
		access->layer_part[layer] = (sum + (uint64_t)access->mesh_ranks - 1u) /
		    (uint64_t)access->mesh_ranks;
	}
}

static SparkDsv41FlashEngramDirEntry SparkDsv41FlashEngramEntryRead(
	const uint8_t *raw)
{
	SparkDsv41FlashEngramDirEntry entry;
	memcpy(&entry.kind,raw,4u);
	memcpy(&entry.layer,raw + 4u,4u);
	memcpy(&entry.payload_type,raw + 8u,4u);
	memcpy(&entry.codec,raw + 12u,4u);
	memcpy(&entry.scale_encoding,raw + 16u,4u);
	memcpy(&entry.groups,raw + 20u,4u);
	memcpy(&entry.rows,raw + 24u,4u);
	memcpy(&entry.cols,raw + 28u,4u);
	memcpy(&entry.payload_offset,raw + 32u,8u);
	memcpy(&entry.payload_bytes,raw + 40u,8u);
	memcpy(&entry.scale_offset,raw + 48u,8u);
	memcpy(&entry.scale_bytes,raw + 56u,8u);
	return(entry);
}

static uint64_t SparkDsv41FlashEngramRankRows(
	const SparkDsv41FlashEngramAccess *access,uint32_t layer_index)
{
	uint64_t rows = access->layer_entries[layer_index] -
	    (uint64_t)access->mesh_rank * access->layer_part[layer_index];
	return(rows > access->layer_part[layer_index] ? access->layer_part[layer_index] : rows);
}

static SparkStatus SparkDsv41FlashEngramShardValidate(
	SparkDsv41FlashEngramAccess *access,
	const char *shard_path,
	uint64_t shard_bytes)
{
	FILE *file;
	uint8_t header[SPARK_DSV41_FLASH_ENGRAM_HEADER_BYTES];
	uint8_t directory[SPARK_DSV41_FLASH_ENGRAM_KIND_COUNT * SPARK_DSV41_FLASH_ENGRAM_ENTRY_BYTES];
	uint32_t magic,version,tensor_count,tp_degree,stored_rank,expect_kind;
	uint64_t file_bytes,rows;
	uint32_t layer;
	SparkDsv41FlashEngramDirEntry entry;
	file = fopen(shard_path,"rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( SparkStageModulePackRead(SPARK_DSV41_FLASH_ENGRAM_MODEL_TAG,file,0,
		    header,sizeof(header)) != SPARK_STATUS_OK ||
		SparkStageModulePackRead(SPARK_DSV41_FLASH_ENGRAM_MODEL_TAG,file,
		    SPARK_DSV41_FLASH_ENGRAM_DIRECTORY_OFFSET,directory,
		    sizeof(directory)) != SPARK_STATUS_OK )
	{
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	(void)fclose(file);
	memcpy(&magic,header,4u);
	memcpy(&version,header + 4u,4u);
	memcpy(&tensor_count,header + 24u,4u);
	memcpy(&tp_degree,header + 68u,4u);
	memcpy(&stored_rank,header + 72u,4u);
	memcpy(&file_bytes,header + 160u,8u);
	if ( magic != SPARK_DSV41_FLASH_ENGRAM_MAGIC || version != SPARK_DSV41_FLASH_ENGRAM_FORMAT_VERSION ||
		tensor_count != SPARK_DSV41_FLASH_ENGRAM_KIND_COUNT ||
		tp_degree != access->mesh_ranks || stored_rank != access->mesh_rank ||
		file_bytes != shard_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (layer = 0u; layer < SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS; layer++)
	{
		expect_kind = layer * SPARK_DSV41_FLASH_ENGRAM_ACCESS_KIND_EMBED_1;
		entry = SparkDsv41FlashEngramEntryRead(directory +
		    (size_t)expect_kind * SPARK_DSV41_FLASH_ENGRAM_ENTRY_BYTES);
		rows = SparkDsv41FlashEngramRankRows(access,layer);
		if ( entry.kind != expect_kind ||
			entry.layer != (layer == 0u ? SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_0 :
			SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_1) ||
			entry.rows != rows || entry.cols != SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM ||
			entry.payload_bytes != rows * SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM ||
			entry.scale_bytes != rows * SPARK_DSV41_FLASH_ENGRAM_ACCESS_SCALE_COLS ||
			entry.payload_offset == 0 || entry.payload_offset > shard_bytes ||
			entry.payload_offset + entry.payload_bytes > shard_bytes ||
			entry.scale_offset + entry.scale_bytes > shard_bytes )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		access->layer_payload_offset[layer] = entry.payload_offset;
		access->layer_scale_offset[layer] = entry.scale_offset;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashEngramManifestCheck(
	const SparkWeightdManifest *manifest,void *opaque)
{
	SparkDsv41FlashEngramAccess *access;
	uint64_t expected,rows,blocks;
	uint32_t layer;
	access = (SparkDsv41FlashEngramAccess *)opaque;
	expected = 0;
	for (layer = 0u; layer < SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS; layer++)
	{
		rows = SparkDsv41FlashEngramRankRows(access,layer);
		blocks = (rows + SPARK_DSV41_FLASH_ENGRAM_ACCESS_BLOCK_ROWS - 1u) /
		    SPARK_DSV41_FLASH_ENGRAM_ACCESS_BLOCK_ROWS;
		expected += blocks * 2u;
	}
	return(expected == manifest->range_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

SparkStatus SparkDsv41FlashEngramAccessOpen(
	SparkDsv41FlashEngramAccess *access,
	const char *socket_path,
	const char *shard_path,
	const char *shard_sha256,
	uint64_t shard_bytes,
	uint32_t mesh_rank,
	uint32_t mesh_ranks,
	const int64_t *mult_in)
{
	SparkWeightdLazyAttachRequest request;
	SparkStatus status;
	uint32_t layer,index;
	memset(access,0,sizeof(*access));
	if ( socket_path == 0 || shard_path == 0 || shard_sha256 == 0 ||
		strlen(shard_sha256) != 64u || mult_in == 0 ||
		mesh_rank >= mesh_ranks || mesh_ranks != SPARK_DSV41_FLASH_ENGRAM_ACCESS_MESH_RANKS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	access->mesh_rank = mesh_rank;
	access->mesh_ranks = mesh_ranks;
	for (layer = 0u; layer < SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS; layer++)
		for (index = 0u; index < SPARK_DSV41_FLASH_ENGRAM_ACCESS_MULTS; index++)
		{
			access->layer_mult[layer][index] =
			    mult_in[layer * SPARK_DSV41_FLASH_ENGRAM_ACCESS_MULTS + index];
			if ( access->layer_mult[layer][index] <= 0 ||
				(access->layer_mult[layer][index] & 1) == 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
	SparkDsv41FlashEngramDerivePartition(access);
	status = SparkDsv41FlashEngramShardValidate(access,shard_path,shard_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
		return(status == SPARK_STATUS_BUSY ? SPARK_STATUS_UNSUPPORTED : status);
	memset(&request,0,sizeof(request));
	memcpy(request.identity.pack_sha256,shard_sha256,65u);
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s",
	    SPARK_DSV41_FLASH_ENGRAM_MODEL_TAG);
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",
	    SPARK_DSV41_FLASH_MODEL_REVISION);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = shard_bytes;
	request.identity.topology = mesh_ranks;
	memcpy(request.pack_path,shard_path,strlen(shard_path) + 1u);
	request.expert_pool_bytes = SPARK_DSV41_FLASH_ENGRAM_POOL_BYTES;
	status = SparkWeightdLazyPackCreateChecked(socket_path,&request,
	    SPARK_DSV41_FLASH_ENGRAM_SPINE_BUDGET_BYTES,
	    SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,
	    SparkDsv41FlashEngramManifestCheck,access,&access->shard);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	access->mesh = (const uint8_t *)access->shard->attached.mesh_mapping;
	access->mesh_bytes = access->shard->attached.mesh_send_buffer_bytes;
	if ( access->mesh == 0 || access->mesh_bytes < SPARK_DSV41_FLASH_ENGRAM_REGION_NEEDS )
	{
		SparkWeightdLazyPackDestroy(access->shard);
		access->shard = 0;
		access->mesh = 0;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	return(SPARK_STATUS_OK);
}

void SparkDsv41FlashEngramAccessClose(SparkDsv41FlashEngramAccess *access)
{
	if ( access == 0 || access->shard == 0 )
		return;
	SparkWeightdLazyPackDestroy(access->shard);
	access->shard = 0;
	access->mesh = 0;
}

SparkStatus SparkDsv41FlashEngramAccessIds(
	const SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int32_t *tokens,
	uint32_t token_count,
	uint32_t pos,
	int64_t *ids)
{
	int32_t window[SPARK_DSV41_FLASH_ENGRAM_ACCESS_MULTS];
	int64_t rolling;
	uint32_t shift,order,head,base;
	if ( access == 0 || ids == 0 || tokens == 0 || token_count == 0 ||
		layer_index >= SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (shift = 0u; shift < SPARK_DSV41_FLASH_ENGRAM_ACCESS_MULTS; shift++)
		window[shift] = pos < shift ? SPARK_DSV41_FLASH_ENGRAM_ACCESS_PAD_ID :
		    tokens[(pos - shift) % token_count];
	rolling = (int64_t)window[0] * access->layer_mult[layer_index][0];
	for (order = 1u; order <= SPARK_DSV41_FLASH_ENGRAM_ACCESS_ORDERS; order++)
	{
		rolling ^= (int64_t)window[order] * access->layer_mult[layer_index][order];
		base = (order - 1u) * SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_COUNT;
		for (head = 0u; head < SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_COUNT; head++)
			ids[base + head] =
			    rolling % access->layer_primes[layer_index][base + head] +
			    access->layer_offsets[layer_index][base + head];
	}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkDsv41FlashEngramLayerId(uint32_t layer_index)
{
	return layer_index == 0u ? SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_0 :
	    SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_1;
}

static uint64_t SparkDsv41FlashEngramStagingOffset(uint32_t layer_index,
	uint32_t col)
{
	return (uint64_t)layer_index * SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS *
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES +
	    (uint64_t)col * SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES;
}

static uint64_t SparkDsv41FlashEngramDoorbellOffset(uint32_t layer_index,
	uint32_t col)
{
	return SPARK_DSV41_FLASH_ENGRAM_ACCESS_STAGING_BYTES +
	    ((uint64_t)layer_index * SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS + col) * 8u;
}

static SparkStatus SparkDsv41FlashEngramLeaseAcquire(
	SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int64_t *ids,
	SparkDsv41FlashEngramLease *lease)
{
	SparkWeightdExpertKey keys[SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS];
	uint32_t layer,col,found,key_count;
	SparkStatus status;
	void *address = 0;
	layer = SparkDsv41FlashEngramLayerId(layer_index);
	key_count = 0u;
	for (col = 0u; col < SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS; col++)
	{
		lease->local[col] = UINT64_MAX;
		if ( (uint64_t)ids[col] / access->layer_part[layer_index] != access->mesh_rank )
			continue;
		lease->local[col] = (uint64_t)ids[col] -
		    access->mesh_rank * access->layer_part[layer_index];
		found = 0u;
		while ( found < key_count && (keys[found].layer != layer ||
		    keys[found].expert != (uint32_t)(lease->local[col] /
		    SPARK_DSV41_FLASH_ENGRAM_ACCESS_BLOCK_ROWS)) )
			found++;
		if ( found == key_count )
		{
			keys[key_count].layer = layer;
			keys[key_count].expert = (uint32_t)(lease->local[col] /
			    SPARK_DSV41_FLASH_ENGRAM_ACCESS_BLOCK_ROWS);
			key_count++;
		}
	}
	if ( key_count == 0u )
	{
		lease->identifier = 0;
		lease->base = 0;
		return(SPARK_STATUS_OK);
	}
	status = SparkWeightdMapAcquire(access->shard->map,keys,key_count,
	    &lease->identifier,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkWeightdMapBeginUse(access->shard->map,lease->identifier,&address);
	if ( status == SPARK_STATUS_OK )
	{
		void *base = 0;
		status = SparkWeightdMapBase(access->shard->map,&base);
		lease->base = (const uint8_t *)base;
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkWeightdMapRelease(access->shard->map,lease->identifier,
		    SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashEngramLeaseFinish(
	SparkDsv41FlashEngramAccess *access,
	SparkDsv41FlashEngramLease *lease)
{
	SparkStatus status;
	if ( lease->identifier == 0 )
		return(SPARK_STATUS_OK);
	(void)SparkWeightdMapRecordCompletion(access->shard->map,lease->identifier,
	    (cudaStream_t)0);
	status = SparkWeightdMapRelease(access->shard->map,lease->identifier,
	    SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	lease->identifier = 0;
	lease->base = 0;
	return(status);
}

static void SparkDsv41FlashEngramRowLoad(
	const SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const SparkDsv41FlashEngramLease *lease,
	uint32_t col,
	uint8_t *out)
{
	const uint8_t *payload,*scale;
	payload = lease->base + access->layer_payload_offset[layer_index] +
	    lease->local[col] * SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM;
	scale = lease->base + access->layer_scale_offset[layer_index] +
	    lease->local[col] * SPARK_DSV41_FLASH_ENGRAM_ACCESS_SCALE_COLS;
	memcpy(out,payload,SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM);
	memcpy(out + SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM,scale,
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_SCALE_COLS);
}

static SparkStatus SparkDsv41FlashEngramPollRemotes(
	SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int64_t *ids,
	uint64_t seq_value,
	uint64_t deadline_ns,
	uint8_t *rows_out)
{
	volatile const uint64_t *doorbell;
	const uint8_t *row;
	uint64_t now;
	uint32_t col,pending;
	now = SparkDsv41FlashEngramNowNs();
	do
	{
		pending = 0u;
		for (col = 0u; col < SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS; col++)
		{
			if ( (uint64_t)ids[col] / access->layer_part[layer_index] == access->mesh_rank )
				continue;
			doorbell = (volatile const uint64_t *)(access->mesh +
			    SparkDsv41FlashEngramDoorbellOffset(layer_index,col));
			if ( *doorbell != seq_value )
			{
				pending = 1u;
				continue;
			}
			row = access->mesh + SparkDsv41FlashEngramStagingOffset(layer_index,col);
			memcpy(rows_out + (size_t)col * SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES,
			    row,SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES);
		}
		if ( pending == 0u )
			return(SPARK_STATUS_OK);
		now = SparkDsv41FlashEngramNowNs();
	} while ( now != 0 && now < deadline_ns );
	SPARK_FAIL(SPARK_STATUS_BUSY);
}

SparkStatus SparkDsv41FlashEngramAccessPublish(
	SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int64_t *ids,
	uint64_t seq_value)
{
	SparkDsv41FlashEngramLease lease;
	SparkStatus status;
	uint8_t row[SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES];
	uint64_t offset;
	uint32_t col;
	if ( access == 0 || access->shard == 0 || ids == 0 ||
		layer_index >= SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (col = 0u; col < SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS; col++)
		if ( ids[col] < 0 || (uint64_t)ids[col] >= access->layer_entries[layer_index] )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkDsv41FlashEngramLeaseAcquire(access,layer_index,ids,&lease);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (col = 0u; col < SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS; col++)
	{
		if ( lease.local[col] == UINT64_MAX )
			continue;
		SparkDsv41FlashEngramRowLoad(access,layer_index,&lease,col,row);
		offset = SparkDsv41FlashEngramStagingOffset(layer_index,col);
		memcpy((void *)(access->mesh + offset),row,SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES);
		status = SparkWeightdClientMeshBroadcast(access->shard->client,
		    ((UINT32_C(1) << access->mesh_ranks) - 1u) &
		    ~(UINT32_C(1) << access->mesh_rank),
		    offset,offset,SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES,seq_value,
		    SparkDsv41FlashEngramDoorbellOffset(layer_index,col),
		    SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkDsv41FlashEngramLeaseFinish(access,&lease);
			SPARK_RETURN(status);
		}
	}
	return(SparkDsv41FlashEngramLeaseFinish(access,&lease));
}

SparkStatus SparkDsv41FlashEngramAccessRows(
	SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int64_t *ids,
	uint64_t seq_value,
	uint64_t timeout_ns,
	uint8_t *rows_out)
{
	SparkDsv41FlashEngramLease lease;
	uint8_t assembled[SPARK_DSV41_FLASH_ENGRAM_ROWS_OUT_BYTES];
	uint64_t deadline,now;
	SparkStatus status;
	uint32_t col;
	if ( access == 0 || access->shard == 0 || ids == 0 || rows_out == 0 ||
		layer_index >= SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (col = 0u; col < SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS; col++)
		if ( ids[col] < 0 || (uint64_t)ids[col] >= access->layer_entries[layer_index] )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	memset(assembled,0,sizeof(assembled));
	status = SparkDsv41FlashEngramLeaseAcquire(access,layer_index,ids,&lease);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (col = 0u; col < SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS; col++)
		if ( lease.local[col] != UINT64_MAX )
			SparkDsv41FlashEngramRowLoad(access,layer_index,&lease,col,
			    assembled + (size_t)col * SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES);
	status = SparkDsv41FlashEngramLeaseFinish(access,&lease);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	now = SparkDsv41FlashEngramNowNs();
	deadline = (now == 0 ? timeout_ns : now + timeout_ns);
	status = SparkDsv41FlashEngramPollRemotes(access,layer_index,ids,seq_value,
	    deadline,assembled);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memcpy(rows_out,assembled,sizeof(assembled));
	return(SPARK_STATUS_OK);
}
