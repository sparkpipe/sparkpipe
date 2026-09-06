#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkFixedRing SparkFixedRing;

SparkStatus SparkFixedRingCreate(
    uint32_t rank,
    uint32_t degree,
    uint32_t start_rank,
    uint32_t broker_port,
    SparkFixedRing **ring_out);

SparkStatus SparkFixedRingSendNext(
    SparkFixedRing *ring,
    const void *buffer,
    uint32_t bytes,
    uint32_t immediate);

SparkStatus SparkFixedRingWaitPrev(
    SparkFixedRing *ring,
    uint32_t expect_immediate,
    uint64_t timeout_ns);

const void *SparkFixedRingLanding(SparkFixedRing *ring);

uint16_t *SparkFixedRingAccumulator(SparkFixedRing *ring);

uint32_t SparkFixedRingChunkElems(SparkFixedRing *ring);

uint32_t SparkFixedRingChunkBytes(SparkFixedRing *ring);

SparkStatus SparkFixedRingSetChunkBytes(
    SparkFixedRing *ring,
    uint32_t chunk_bytes);

void SparkFixedRingDestroy(SparkFixedRing *ring);

#ifdef __cplusplus
}
#endif
