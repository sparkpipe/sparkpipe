#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tap_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TAP_STREAM_READER_ABI_VERSION 1u
#define SPARK_TAP_STREAM_READER_CONFIG_BYTES ((uint32_t)sizeof(SparkTapStreamReaderConfig))
#define SPARK_TAP_STREAM_READER_POLL_FDS 2u

typedef struct SparkTapStreamReader SparkTapStreamReader;

typedef struct SparkTapStreamReaderConfig
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	const char *host;
	uint32_t port;
	const char *path;
	char model_tag[SPARK_TAP_STREAM_NAME_BYTES];
	uint32_t hidden;
	uint32_t required_count;
	SparkTapStreamEntry required[SPARK_TAP_STREAM_MAX_ENTRIES];
	uint32_t max_rows;
	uint32_t io_timeout_ms;
} SparkTapStreamReaderConfig;

typedef struct SparkTapStreamReaderFrame
{
	SparkTapStreamFrame header;
	const SparkTapStreamHello *hello;
	const uint32_t *tokens;
	const uint16_t *rows;
	uint64_t row_elements;
	const uint32_t *required_index;
	uint32_t required_count;
} SparkTapStreamReaderFrame;

typedef struct SparkTapStreamReaderStats
{
	uint64_t streams;
	uint64_t refused;
	uint64_t frames;
	uint64_t rows;
	uint64_t bytes;
} SparkTapStreamReaderStats;

SparkStatus SparkTapStreamReaderParseEntries(const char *text, SparkTapStreamEntry *entries, uint32_t capacity, uint32_t *count_out);
SparkStatus SparkTapStreamReaderCreate(const SparkTapStreamReaderConfig *config, SparkTapStreamReader **reader_out);
void SparkTapStreamReaderDestroy(SparkTapStreamReader *reader);
uint32_t SparkTapStreamReaderPort(const SparkTapStreamReader *reader);
uint32_t SparkTapStreamReaderPollFds(const SparkTapStreamReader *reader, int *fds, uint32_t *events, uint64_t *deadline_ns);
SparkStatus SparkTapStreamReaderAdvance(SparkTapStreamReader *reader, SparkTapStreamReaderFrame *frame_out);
void SparkTapStreamReaderGetStats(const SparkTapStreamReader *reader, SparkTapStreamReaderStats *stats_out);

#ifdef __cplusplus
}
#endif
