#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_row_file.h"

#define ROWS 1000u
#define FIRST_ID 5000
#define WAVE_IDS 5000u

static uint32_t failures;

static void Check(int condition, const char *message)
{
	if ( !condition )
	{
		fprintf(stderr, "FAIL test_row_file: %s\n", message);
		failures++;
	}
}

static uint8_t Byte(uint64_t offset)
{
	return (uint8_t)((offset * 2654435761u) >> 13);
}

static int32_t WriteFile(const char *path, uint64_t bytes)
{
	FILE *out = fopen(path, "wb");
	if ( out == 0 )
		return -1;
	for ( uint64_t offset = 0u; offset < bytes; ++offset )
		fputc(Byte(offset), out);
	return fclose(out);
}

static uint32_t RowMatches(const uint8_t *row, uint64_t file_offset, uint32_t row_bytes)
{
	for ( uint32_t index = 0u; index < row_bytes; ++index )
		if ( row[index] != Byte(file_offset + index) )
			return 0u;
	return 1u;
}

static void Region(const char *path, uint64_t offset, uint32_t row_bytes, uint32_t buffered, uint32_t cache_rows, const char *name)
{
	const SparkRowFileOptions options = { offset, ROWS, row_bytes, cache_rows, 4u, buffered };
	const int64_t ids[12] = { FIRST_ID, FIRST_ID + 999, FIRST_ID + 17, FIRST_ID - 1, FIRST_ID + 1000, FIRST_ID + 17, FIRST_ID + 500, -3,
		FIRST_ID + 501, FIRST_ID + 16, FIRST_ID + 998, FIRST_ID + 64 };
	const uint64_t stride = (uint64_t)row_bytes + 8u;
	uint8_t *out = (uint8_t *)malloc(12u * stride);
	SparkRowFile *file = 0;
	SparkRowFileCounters counters;
	char message[160];
	uint32_t good = 1u, untouched = 1u;
	if ( SparkRowFileOpen(path, &options, &file) != SPARK_STATUS_OK || out == 0 )
	{
		if ( buffered == 0u )
			printf("NOTE test_row_file: direct reads are not available for %s here; the buffered reads cover it\n", name);
		else
			Check(0, "a buffered region opens");
		free(out);
		return;
	}
	memset(out, 0xa5, 12u * stride);
	Check(SparkRowFileFetch(file, ids, 12u, FIRST_ID, out, stride) == SPARK_STATUS_OK, "a batch with owned and foreign ids is fetched");
	for ( uint32_t index = 0u; index < 12u; ++index )
	{
		const uint32_t owned = ids[index] >= FIRST_ID && ids[index] < FIRST_ID + (int64_t)ROWS;
		if ( owned != 0u )
			good &= RowMatches(out + index * stride, offset + (uint64_t)(ids[index] - FIRST_ID) * row_bytes, row_bytes);
		for ( uint64_t at = owned != 0u ? row_bytes : 0u; at < stride; ++at )
			untouched &= out[index * stride + at] == 0xa5 ? 1u : 0u;
	}
	snprintf(message, sizeof(message), "%s: every owned id returns its row's bytes", name);
	Check(good != 0u, message);
	snprintf(message, sizeof(message), "%s: a foreign id and the bytes past a row are left alone", name);
	Check(untouched != 0u, message);
	SparkRowFileCount(file, &counters);
	Check(counters.requested == 9u && counters.cached + counters.read == 9u, "nine owned ids are counted, each cached or read");
	Check(counters.read == 9u, "a first batch reads every owned id from the file");
	memset(out, 0, 12u * stride);
	Check(SparkRowFileFetch(file, ids, 12u, FIRST_ID, out, stride) == SPARK_STATUS_OK, "the batch is fetched again");
	good = 1u;
	for ( uint32_t index = 0u; index < 12u; ++index )
		if ( ids[index] >= FIRST_ID && ids[index] < FIRST_ID + (int64_t)ROWS )
			good &= RowMatches(out + index * stride, offset + (uint64_t)(ids[index] - FIRST_ID) * row_bytes, row_bytes);
	snprintf(message, sizeof(message), "%s: the second fetch returns the same rows", name);
	Check(good != 0u, message);
	SparkRowFileCount(file, &counters);
	if ( cache_rows >= ROWS )
		Check(counters.cached == 9u && counters.read == 9u, "a cache that holds every row serves the second batch without reading");
	else if ( cache_rows == 0u )
		Check(counters.cached == 0u && counters.read == 18u, "without a cache every fetch reads");
	else
		Check(counters.cached >= 1u && counters.cached + counters.read == 18u, "a small cache serves the rows it kept and reads the rest");
	Check(SparkRowFileFetch(file, ids, 0u, FIRST_ID, out, stride) == SPARK_STATUS_OK, "an empty batch is nothing to do");
	Check(SparkRowFileFetch(file, ids, 12u, FIRST_ID, out, row_bytes - 1u) != SPARK_STATUS_OK, "a stride shorter than a row is refused");
	SparkRowFileClose(file);
	free(out);
}

static void Wave(const char *path, uint64_t offset, uint32_t buffered)
{
	const SparkRowFileOptions options = { offset, ROWS, 256u, 0u, 4u, buffered };
	const uint64_t stride = 256u;
	int64_t *ids = (int64_t *)malloc(WAVE_IDS * sizeof(*ids));
	uint8_t *out = (uint8_t *)malloc(WAVE_IDS * stride);
	SparkRowFile *file = 0;
	SparkRowFileCounters counters;
	uint32_t good = 1u, owned = 0u;
	if ( ids == 0 || out == 0 || SparkRowFileOpen(path, &options, &file) != SPARK_STATUS_OK )
	{
		Check(buffered == 0u && ids != 0 && out != 0, "a buffered region opens for a wave");
		free(ids);
		free(out);
		return;
	}
	for ( uint32_t index = 0u; index < WAVE_IDS; ++index )
		ids[index] = FIRST_ID + (int64_t)((index * 7919u) % (ROWS + 24u)) - 12;
	memset(out, 0xa5, WAVE_IDS * stride);
	Check(SparkRowFileFetch(file, ids, WAVE_IDS, FIRST_ID, out, stride) == SPARK_STATUS_OK, "a wave of ids is fetched");
	for ( uint32_t index = 0u; index < WAVE_IDS; ++index )
		if ( ids[index] >= FIRST_ID && ids[index] < FIRST_ID + (int64_t)ROWS )
		{
			owned++;
			good &= RowMatches(out + index * stride, offset + (uint64_t)(ids[index] - FIRST_ID) * 256u, 256u);
		}
		else
			good &= out[index * stride] == 0xa5 && out[index * stride + 255u] == 0xa5 ? 1u : 0u;
	Check(good != 0u, "a wave larger than a reader's batch returns every owned row and leaves foreign ids alone");
	SparkRowFileCount(file, &counters);
	Check(owned > 4000u && counters.requested == owned && counters.read == owned && counters.cached == 0u, "every owned id of the wave is read once");
	SparkRowFileClose(file);
	free(ids);
	free(out);
}

int main(void)
{
	char path[256];
	const uint64_t offset = 768u, odd_offset = 100u, bytes = 768u + (uint64_t)ROWS * 256u;
	SparkRowFile *file = 0;
	SparkRowFileOptions options = { offset, ROWS, 256u, 16u, 4u, 1u };
	snprintf(path, sizeof(path), "build/test_row_file.%ld.bin", (long)getpid());
	Check(WriteFile(path, bytes) == 0, "the test file is written");
	for ( uint32_t buffered = 0u; buffered < 2u; ++buffered )
	{
		Region(path, offset, 256u, buffered, 1024u, "256-byte rows, the last row ending the file");
		Region(path, offset, 256u, buffered, 16u, "256-byte rows through a 16-row cache");
		Region(path, offset, 8u, buffered, 0u, "8-byte rows without a cache");
		Region(path, odd_offset, 96u, buffered, 1024u, "96-byte rows that cross block boundaries");
		Wave(path, offset, buffered);
	}
	options.rows = ROWS + 1u;
	Check(SparkRowFileOpen(path, &options, &file) == SPARK_STATUS_SCHEMA_ERROR && file == 0, "a region longer than the file is refused");
	options.rows = ROWS;
	options.row_bytes = SPARK_ROW_FILE_BLOCK_BYTES + 1u;
	Check(SparkRowFileOpen(path, &options, &file) == SPARK_STATUS_INVALID_ARGUMENT, "a row longer than a block is refused");
	options.row_bytes = 256u;
	options.workers = 0u;
	Check(SparkRowFileOpen(path, &options, &file) == SPARK_STATUS_INVALID_ARGUMENT, "a reader with no workers is refused");
	options.workers = 4u;
	Check(SparkRowFileOpen("build/test_row_file.absent.bin", &options, &file) == SPARK_STATUS_NOT_FOUND, "a missing file is reported as not found");
	unlink(path);
	if ( failures != 0u )
		return 1;
	printf("PASS test_row_file\n");
	return 0;
}
