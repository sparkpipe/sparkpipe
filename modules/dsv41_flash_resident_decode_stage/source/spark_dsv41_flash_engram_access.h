#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"

#include "sparkpipe/spark_dsv41_flash_resident_decode_stage_firmware.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS \
	SPARK_DSV41_FLASH_MODEL_ENGRAM_MODULE_COUNT
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS \
	(SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_COUNT * \
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_ORDERS)
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_MULTS 4u
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_ORDERS 3u
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM \
	SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_DIMENSION
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_SCALE_COLS 8u
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES \
	(SPARK_DSV41_FLASH_ENGRAM_ACCESS_HEAD_DIM + \
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_SCALE_COLS)
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_BLOCK_ROWS 32768u
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_PAD_ID 2
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_PRIME_START INT64_C(15999999)
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_MESH_RANKS 16u
#define SPARK_DSV41_FLASH_ENGRAM_ACCESS_STAGING_BYTES \
	((uint64_t)SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS * \
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS * \
	    SPARK_DSV41_FLASH_ENGRAM_ACCESS_ROW_BYTES)

typedef struct SparkDsv41FlashEngramAccess
{
	SparkWeightdLazyPack *shard;
	const uint8_t *mesh;
	uint64_t mesh_bytes;
	uint32_t mesh_rank;
	uint32_t mesh_ranks;
	uint64_t layer_entries[SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS];
	uint64_t layer_part[SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS];
	int64_t layer_primes[SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS][SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS];
	int64_t layer_offsets[SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS][SPARK_DSV41_FLASH_ENGRAM_ACCESS_COLS];
	int64_t layer_mult[SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS][SPARK_DSV41_FLASH_ENGRAM_ACCESS_MULTS];
	uint64_t layer_payload_offset[SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS];
	uint64_t layer_scale_offset[SPARK_DSV41_FLASH_ENGRAM_ACCESS_LAYERS];
} SparkDsv41FlashEngramAccess;

SparkStatus SparkDsv41FlashEngramAccessOpen(
	SparkDsv41FlashEngramAccess *access,
	const char *socket_path,
	const char *shard_path,
	const char *shard_sha256,
	uint64_t shard_bytes,
	uint32_t mesh_rank,
	uint32_t mesh_ranks,
	const int64_t *mult_in);

void SparkDsv41FlashEngramAccessClose(SparkDsv41FlashEngramAccess *access);

SparkStatus SparkDsv41FlashEngramAccessIds(
	const SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int32_t *tokens,
	uint32_t token_count,
	uint32_t pos,
	int64_t *ids);

SparkStatus SparkDsv41FlashEngramAccessPublish(
	SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int64_t *ids,
	uint64_t seq_value);

SparkStatus SparkDsv41FlashEngramAccessRows(
	SparkDsv41FlashEngramAccess *access,
	uint32_t layer_index,
	const int64_t *ids,
	uint64_t seq_value,
	uint64_t timeout_ns,
	uint8_t *rows_out);

#ifdef __cplusplus
}
#endif
