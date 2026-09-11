#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

typedef struct SparkGlm5NextTapRing SparkGlm5NextTapRing;

#ifdef __cplusplus
extern "C" {
#endif

SparkStatus SparkGlm5NextTapRingCreate(
	uint32_t lane_count,
	uint32_t window_positions,
	uint32_t tap_count,
	uint32_t tap_width_elements,
	SparkGlm5NextTapRing **ring_out);
SparkStatus SparkGlm5NextTapRingEnqueueCapture(
	SparkGlm5NextTapRing *ring,
	void *op_stream,
	void *op_event,
	void *done_event,
	const uint16_t *rows_device_bf16,
	uint32_t row_count,
	const uint32_t *row_lanes,
	const uint32_t *row_positions,
	uint32_t tap_index);
SparkStatus SparkGlm5NextTapRingRead(
	const SparkGlm5NextTapRing *ring,
	uint32_t lane,
	uint32_t position,
	const uint16_t **tap_row_out);
SparkStatus SparkGlm5NextTapRingCommitAnchor(
	SparkGlm5NextTapRing *ring,
	uint32_t lane,
	uint32_t next_position);
void SparkGlm5NextTapRingDestroy(SparkGlm5NextTapRing *ring);

#ifdef __cplusplus
}
#endif
