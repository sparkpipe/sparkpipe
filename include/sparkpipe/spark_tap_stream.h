#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TAP_STREAM_MAGIC 0x32545053u
#define SPARK_TAP_STREAM_VERSION 1u
#define SPARK_TAP_STREAM_KIND_HELLO 1u
#define SPARK_TAP_STREAM_KIND_FRAME 2u
#define SPARK_TAP_STREAM_DTYPE_BF16 1u
#define SPARK_TAP_STREAM_FLAG_PREFILL 0x1u
#define SPARK_TAP_STREAM_FLAG_DECODE 0x2u
#define SPARK_TAP_STREAM_FLAG_VERIFY 0x4u
#define SPARK_TAP_STREAM_FLAG_CHAIN 0x8u
#define SPARK_TAP_STREAM_NAME_BYTES 32u
#define SPARK_TAP_STREAM_IDENTITY_BYTES 32u
#define SPARK_TAP_STREAM_MAX_ENTRIES 128u
#define SPARK_TAP_STREAM_NO_TOKEN 0xffffffffu

typedef struct SparkTapStreamEntry
{
	uint16_t form;
	uint16_t layer;
} SparkTapStreamEntry;

typedef struct SparkTapStreamHello
{
	uint32_t magic;
	uint32_t version;
	uint32_t kind;
	uint32_t header_bytes;
	char model_tag[SPARK_TAP_STREAM_NAME_BYTES];
	uint8_t pack_identity[SPARK_TAP_STREAM_IDENTITY_BYTES];
	uint32_t hidden;
	uint32_t total_layers;
	uint32_t dtype;
	uint32_t entry_count;
	uint32_t tap_rank;
	uint32_t tp_degree;
	uint64_t fingerprint;
	SparkTapStreamEntry entries[SPARK_TAP_STREAM_MAX_ENTRIES];
} SparkTapStreamHello;

typedef struct SparkTapStreamFrame
{
	uint32_t magic;
	uint32_t version;
	uint32_t kind;
	uint32_t header_bytes;
	uint64_t fingerprint;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t request_generation;
	uint64_t sequence_id;
	uint64_t first_position;
	uint64_t serial;
	uint32_t row_count;
	uint32_t flags;
	uint32_t tail_token;
	uint32_t entry_count;
	uint64_t payload_bytes;
} SparkTapStreamFrame;

uint64_t SparkTapStreamFingerprint(const SparkTapStreamHello *hello);

#ifdef __cplusplus
}
#endif
