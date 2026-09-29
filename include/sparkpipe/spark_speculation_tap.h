#pragma once

#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_TAP_MAX_TAPS 8u
#define SPARK_SPECULATION_TAP_REDUCTION_MEAN 1u
#define SPARK_SPECULATION_TAP_REDUCTION_ALL 2u
#define SPARK_SPECULATION_TAP_DTYPE_BF16 1u
#define SPARK_SPECULATION_TAP_FLAG_PREFILL 0x1u
#define SPARK_SPECULATION_TAP_FLAG_DECODE 0x2u
#define SPARK_SPECULATION_TAP_FLAG_VERIFY 0x4u
#define SPARK_SPECULATION_TAP_FLAG_GAP 0x8u
#define SPARK_SPECULATION_TAP_FLAG_MASK 0xfu

#define SPARK_SPECULATION_TAP_FRAGMENT_MAGIC 0x31545053u
#define SPARK_SPECULATION_TAP_FRAGMENT_VERSION 1u
#define SPARK_SPECULATION_TAP_FRAGMENT_KIND 3u
#define SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES 64u
#define SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX 8192u
#define SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX (SPARK_SPECULATION_TAP_FRAGMENT_HEADER_BYTES + SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX)

#define SPARK_SPECULATION_TAP_DUMP_MAGIC 0x44545053u
#define SPARK_SPECULATION_TAP_DUMP_VERSION 1u
#define SPARK_SPECULATION_TAP_DUMP_HEADER_BYTES 128u
#define SPARK_SPECULATION_TAP_DUMP_RECORD_HEADER_BYTES 32u
#define SPARK_SPECULATION_TAP_DUMP_FLAG_TRUNCATED 0x1u
#define SPARK_SPECULATION_TAP_DUMP_FLAG_CLOSED 0x2u
#define SPARK_SPECULATION_TAP_MODEL_TAG_BYTES 16u

typedef struct SparkSpeculationTapSet
{
	uint32_t reduction;
	uint32_t tap_count;
	uint32_t stream_count;
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t row_elements;
	uint32_t row_bytes;
	uint32_t record_bytes;
	uint32_t layers[SPARK_SPECULATION_TAP_MAX_TAPS];
} SparkSpeculationTapSet;

typedef struct SparkSpeculationTapRecord
{
	uint64_t engine_generation;
	uint64_t sequence_id;
	uint64_t position;
	uint64_t serial;
	uint32_t token_id;
	uint32_t flags;
} SparkSpeculationTapRecord;

typedef struct SparkSpeculationTapFragment
{
	SparkSpeculationTapRecord record;
	uint64_t fingerprint;
	uint32_t record_bytes;
	uint32_t offset;
	uint32_t bytes;
	const uint8_t *payload;
} SparkSpeculationTapFragment;

typedef struct SparkSpeculationTapAssembler
{
	uint64_t fingerprint;
	uint32_t record_bytes;
	uint32_t received;
	uint32_t active;
	uint8_t *payload;
	SparkSpeculationTapRecord record;
	uint64_t completed;
	uint64_t abandoned;
	uint64_t rejected;
} SparkSpeculationTapAssembler;

typedef struct SparkSpeculationTapRing
{
	uint8_t *storage;
	uint64_t slot_bytes;
	uint32_t slot_count;
	uint32_t record_bytes;
	uint64_t head;
	uint64_t tail;
	uint64_t pushed;
	uint64_t dropped;
	uint32_t gap_pending;
} SparkSpeculationTapRing;

typedef struct SparkSpeculationTapDump
{
	FILE *file;
	SparkSpeculationTapSet set;
	uint64_t max_bytes;
	uint64_t bytes;
	uint64_t records;
	uint64_t refused;
	uint32_t flags;
} SparkSpeculationTapDump;

SparkStatus SparkSpeculationTapSetParse(const char *text,uint32_t layer_count,uint32_t stream_count,uint32_t hidden_dimension,SparkSpeculationTapSet *set);
uint32_t SparkSpeculationTapSetOrdinal(const SparkSpeculationTapSet *set,uint32_t layer);
uint64_t SparkSpeculationTapSetFingerprint(const SparkSpeculationTapSet *set);
uint32_t SparkSpeculationTapFragmentCount(uint32_t record_bytes,uint32_t payload_max);
SparkStatus SparkSpeculationTapEncodeFragment(const SparkSpeculationTapRecord *record,uint64_t fingerprint,const uint8_t *payload,uint32_t record_bytes,uint32_t offset,uint32_t bytes,uint8_t *out,uint32_t out_capacity,uint32_t *out_length);
SparkStatus SparkSpeculationTapDecodeFragment(const uint8_t *bytes,uint32_t length,SparkSpeculationTapFragment *fragment);
SparkStatus SparkSpeculationTapAssemblerInitialize(SparkSpeculationTapAssembler *assembler,uint64_t fingerprint,uint32_t record_bytes,uint8_t *payload);
SparkStatus SparkSpeculationTapAssemblerAccept(SparkSpeculationTapAssembler *assembler,const uint8_t *bytes,uint32_t length,uint32_t *complete_out);
uint64_t SparkSpeculationTapRingSlotBytes(uint32_t record_bytes);
SparkStatus SparkSpeculationTapRingInitialize(SparkSpeculationTapRing *ring,uint8_t *storage,uint64_t storage_bytes,uint32_t record_bytes);
SparkStatus SparkSpeculationTapRingPush(SparkSpeculationTapRing *ring,const SparkSpeculationTapRecord *record,const uint8_t *payload);
SparkStatus SparkSpeculationTapRingPop(SparkSpeculationTapRing *ring,SparkSpeculationTapRecord *record,uint8_t *payload);
SparkStatus SparkSpeculationTapDumpOpen(SparkSpeculationTapDump *dump,const char *path,const SparkSpeculationTapSet *set,const char *model_tag,uint64_t engine_generation,uint32_t tp_rank,uint64_t max_bytes);
SparkStatus SparkSpeculationTapDumpAppend(SparkSpeculationTapDump *dump,const SparkSpeculationTapRecord *record,const uint8_t *payload);
SparkStatus SparkSpeculationTapDumpClose(SparkSpeculationTapDump *dump);

#ifdef __cplusplus
}
#endif
