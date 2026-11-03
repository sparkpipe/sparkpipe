#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHT_STREAM_MAX_RANKS 16u
#define SPARK_WEIGHT_STREAM_RECEIVE_DEPTH 512u
#define SPARK_WEIGHT_STREAM_TIMEOUT_NS UINT64_C(120000000000)

typedef struct SparkWeightStreamEndpoint
{
	uint32_t qpn;
	uint32_t rkey;
	uint64_t address;
	uint8_t gid[16];
} SparkWeightStreamEndpoint;

typedef SparkStatus (*SparkWeightStreamGather)(void *context, const void *mine, void *all, uint64_t bytes_per_rank);

typedef struct SparkWeightStreamConfig
{
	uint32_t rank;
	uint32_t ranks;
	uint32_t chunk_bytes;
	uint32_t depth;
	uint32_t fabric_gid_index;
	uint32_t pair_gid_index;
	uint32_t traffic_class;
	const char *fabric_device;
	const char *pair_device;
	uint8_t *region;
	uint64_t region_bytes;
	SparkWeightStreamGather gather;
	void *gather_context;
} SparkWeightStreamConfig;

typedef struct SparkWeightStreamBlock
{
	uint32_t source;
	uint32_t reserved0;
	uint64_t half_offset[2];
} SparkWeightStreamBlock;

typedef struct SparkWeightStream SparkWeightStream;

SparkStatus SparkWeightStreamOpen(const SparkWeightStreamConfig *config, SparkWeightStream **stream_out);
void SparkWeightStreamClose(SparkWeightStream *stream);
uint32_t SparkWeightStreamRotationSource(uint32_t rank, uint32_t ranks, uint32_t step);
SparkStatus SparkWeightStreamBroadcast(SparkWeightStream *stream, uint64_t source_offset, uint64_t bytes, uint64_t landing_offset, uint32_t tag);
SparkStatus SparkWeightStreamExchange(SparkWeightStream *stream, uint64_t send_offset, uint64_t send_pitch, uint64_t bytes, uint64_t landing_offset, uint32_t tag);
SparkStatus SparkWeightStreamRotate(SparkWeightStream *stream, uint32_t step, uint64_t own_offset, uint64_t partner_offset, uint64_t bytes,
	uint64_t landing_offset, uint32_t tag, SparkWeightStreamBlock *block);

#ifdef __cplusplus
}
#endif
