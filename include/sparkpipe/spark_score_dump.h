#ifndef SPARKPIPE_SPARK_SCORE_DUMP_H
#define SPARKPIPE_SPARK_SCORE_DUMP_H

#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SCORE_DUMP_VERSION 1u
#define SPARK_SCORE_DUMP_TOP_K 64u
#define SPARK_SCORE_DUMP_KEY_SEED UINT64_C(0xcbf29ce484222325)
#define SPARK_SCORE_DUMP_KEY_PRIME UINT64_C(0x00000100000001b3)
#define SPARK_SCORE_DUMP_NO_TOKEN UINT32_MAX
#define SPARK_SCORE_DUMP_PATH_BYTES 4096u

#define SPARK_SCORE_DUMP_ROW_KEY_VALID UINT32_C(0x00000001)
#define SPARK_SCORE_DUMP_ROW_PROBED UINT32_C(0x00000002)
#define SPARK_SCORE_DUMP_ROW_TIER2 UINT32_C(0x00000004)
#define SPARK_SCORE_DUMP_ROW_NONFINITE UINT32_C(0x00000008)

#define SPARK_SCORE_DUMP_RECORD_ROW UINT32_C(0x31574f52)
#define SPARK_SCORE_DUMP_RECORD_END UINT32_C(0x31444e45)

typedef struct SparkScoreDumpFileHeader
{
	char magic[8];
	uint32_t version;
	uint32_t header_bytes;
	uint32_t tp_rank;
	uint32_t tp_degree;
	uint32_t shard_begin;
	uint32_t shard_end;
	uint32_t vocabulary;
	uint32_t hidden_dimension;
	uint32_t top_k;
	uint32_t tier2;
	uint8_t arm_digest[32];
	uint8_t probe_sha256[32];
	uint8_t tier2_sha256[32];
} SparkScoreDumpFileHeader;

typedef struct SparkScoreDumpRowRecord
{
	uint32_t record_kind;
	uint32_t flags;
	uint64_t key;
	uint64_t wave_ordinal;
	uint32_t position;
	uint32_t row_in_wave;
	uint32_t input_token;
	uint32_t served_token;
	uint32_t probe_count;
	float local_max;
	double local_sum_exp;
	uint32_t top_ids[SPARK_SCORE_DUMP_TOP_K];
	float top_logits[SPARK_SCORE_DUMP_TOP_K];
} SparkScoreDumpRowRecord;

typedef struct SparkScoreDumpEndRecord
{
	uint32_t record_kind;
	uint32_t reserved;
	uint64_t row_count;
	uint64_t wave_count;
	uint64_t keyless_row_count;
	uint64_t probed_row_count;
	uint64_t tier2_row_count;
	uint64_t nonfinite_row_count;
	uint64_t skipped_wave_count;
} SparkScoreDumpEndRecord;

typedef struct SparkScoreDumpStats
{
	float local_max;
	uint32_t flags;
	double local_sum_exp;
	uint32_t top_ids[SPARK_SCORE_DUMP_TOP_K];
	float top_logits[SPARK_SCORE_DUMP_TOP_K];
} SparkScoreDumpStats;

typedef struct SparkScoreDumpProbeHeader
{
	char magic[8];
	uint32_t version;
	uint32_t id_width;
	uint64_t entry_count;
} SparkScoreDumpProbeHeader;

typedef struct SparkScoreDumpTable
{
	uint8_t *bytes;
	uint64_t byte_count;
	uint64_t entry_count;
	uint32_t id_width;
	uint32_t entry_bytes;
	uint8_t sha256[32];
} SparkScoreDumpTable;

typedef struct SparkScoreDumpKeys
{
	uint64_t *keys;
	uint32_t *known;
	uint32_t slot_count;
	uint32_t position_count;
} SparkScoreDumpKeys;

typedef struct SparkScoreDumpConfig
{
	const char *directory;
	const char *probe_path;
	const char *tier2_path;
	const uint8_t *arm_digest;
	uint32_t tp_rank;
	uint32_t tp_degree;
	uint32_t vocabulary;
	uint32_t hidden_dimension;
	uint32_t slot_count;
	uint32_t position_count;
} SparkScoreDumpConfig;

typedef struct SparkScoreDumpWriter
{
	FILE *rows;
	FILE *tier2;
	SparkScoreDumpFileHeader header;
	SparkScoreDumpTable probes;
	SparkScoreDumpTable tier2_rows;
	SparkScoreDumpKeys keys;
	SparkScoreDumpEndRecord end;
	char rows_path[SPARK_SCORE_DUMP_PATH_BYTES];
	char tier2_path[SPARK_SCORE_DUMP_PATH_BYTES];
} SparkScoreDumpWriter;

uint64_t SparkScoreDumpKeyNext(uint64_t previous, uint32_t token);
SparkStatus SparkScoreDumpShard(uint32_t vocabulary, uint32_t tp_degree, uint32_t tp_rank, uint32_t *begin, uint32_t *end);

SparkStatus SparkScoreDumpKeysInitialize(SparkScoreDumpKeys *keys, uint32_t slot_count, uint32_t position_count);
void SparkScoreDumpKeysDestroy(SparkScoreDumpKeys *keys);
uint32_t SparkScoreDumpKeysAdvance(SparkScoreDumpKeys *keys, uint32_t slot, uint32_t position, uint32_t token, uint64_t *key);

SparkStatus SparkScoreDumpTableLoad(const char *path, const char *magic, uint32_t require_ids, SparkScoreDumpTable *table);
void SparkScoreDumpTableDestroy(SparkScoreDumpTable *table);
const uint32_t *SparkScoreDumpTableFind(const SparkScoreDumpTable *table, uint64_t key, uint32_t position, uint32_t *id_count);

SparkStatus SparkScoreDumpOpen(const SparkScoreDumpConfig *config, SparkScoreDumpWriter *writer);
SparkStatus SparkScoreDumpWriteRow(SparkScoreDumpWriter *writer, const SparkScoreDumpRowRecord *row, const uint32_t *probe_ids, const float *probe_logits);
SparkStatus SparkScoreDumpWriteTier2(SparkScoreDumpWriter *writer, uint64_t key, uint32_t position, const float *logits);
void SparkScoreDumpNoteWave(SparkScoreDumpWriter *writer, uint32_t skipped);
SparkStatus SparkScoreDumpClose(SparkScoreDumpWriter *writer);

#ifdef __cplusplus
}
#endif

#endif
