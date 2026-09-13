#ifndef SPARK_KV_BACKING_H
#define SPARK_KV_BACKING_H

#include <stdint.h>
#include <stddef.h>
#include "spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * JIT-KV backing store (docs/JIT_KV_DESIGN.md): a slot file that tiers
 * KV blocks (4 MiB each: 64 tokens x 64 KiB/token for this model
 * family) out of VRAM. Fixed-stride slots, single-writer discipline
 * (the adapter's park worker), pread/pwrite at slot offsets.
 *
 * 2.5TB budget = 625,000 slots = 38M tokens = the LRU horizon; the
 * pager parks/restores whole lanes and backpressures instead of
 * thrashing (a per-token page-in is a 20-40x bandwidth cliff).
 */
#define SPARK_KV_BACKING_MAGIC "SPKVBS01"
#define SPARK_KV_BACKING_HEADER_BYTES 4096u
#define SPARK_KV_BACKING_SLOT_BYTES (4u * 1024u * 1024u)
#define SPARK_KV_BACKING_MAX_SLOT_COUNT 1000000u

typedef struct SparkKvBacking
{
	int file_descriptor;
	uint64_t slot_bytes;
	uint32_t slot_count;
	uint32_t free_hint;
	uint8_t *slot_live;    /* bitmap: 1 = allocated */
	uint32_t live_count;
} SparkKvBacking;

typedef struct SparkKvBackingConfiguration
{
	const char *path;             /* created if absent */
	uint64_t maximum_bytes;       /* the budget; 0 = error */
	uint32_t reserved;
} SparkKvBackingConfiguration;

SparkStatus SparkKvBackingOpen(const SparkKvBackingConfiguration *configuration,
	SparkKvBacking *backing);
void SparkKvBackingClose(SparkKvBacking *backing);

/* Slot lifecycle. Allocate returns a slot index or -1 when the horizon
 * is full (the caller must backpressure, never thrash). */
int64_t SparkKvBackingAllocate(SparkKvBacking *backing);
void SparkKvBackingRelease(SparkKvBacking *backing, uint32_t slot);

/* Block I/O at a slot offset. host_buffer must be slot_bytes large.
 * Returns IO_ERROR on short transfers (disk full, fd closed). */
SparkStatus SparkKvBackingWriteBlock(SparkKvBacking *backing, uint32_t slot,
	const void *host_buffer);
SparkStatus SparkKvBackingReadBlock(SparkKvBacking *backing, uint32_t slot,
	void *host_buffer);

uint64_t SparkKvBackingLiveBytes(const SparkKvBacking *backing);

#ifdef __cplusplus
}
#endif

#endif
