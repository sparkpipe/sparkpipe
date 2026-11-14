#ifndef SPARKPIPE_SPARK_ROW_FILE_H
#define SPARKPIPE_SPARK_ROW_FILE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_ROW_FILE_BLOCK_BYTES 4096u
#define SPARK_ROW_FILE_MAX_WORKERS 64u

typedef struct SparkRowFile SparkRowFile;

typedef struct SparkRowFileOptions
{
	uint64_t offset;
	uint64_t rows;
	uint32_t row_bytes;
	uint32_t cache_rows;
	uint32_t workers;
	uint32_t buffered;
} SparkRowFileOptions;

typedef struct SparkRowFileCounters
{
	uint64_t requested;
	uint64_t cached;
	uint64_t read;
} SparkRowFileCounters;

SparkStatus SparkRowFileOpen(const char *path, const SparkRowFileOptions *options, SparkRowFile **file);
void SparkRowFileClose(SparkRowFile *file);
SparkStatus SparkRowFileFetch(SparkRowFile *file, const int64_t *ids, uint64_t count, int64_t first_id, uint8_t *out, uint64_t out_stride);
void SparkRowFileCount(SparkRowFile *file, SparkRowFileCounters *counters);

#ifdef __cplusplus
}
#endif

#endif
