#include <stdatomic.h>
#include "sparkpipe/spark_error_site.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_stage_module_common.h"

#include "spark_glm5_next_tap_ring.h"

#define SPARK_GLM5_NEXT_TAP_RING_TAG "glm5_next_tap_ring"
#define SPARK_GLM5_NEXT_TAP_RING_STAMP_INVALID 0u

typedef struct SparkGlm5NextTapCapture
{
	SparkGlm5NextTapRing *ring;
	uint32_t row_count;
	uint32_t *row_lanes;
	uint32_t *row_positions;
} SparkGlm5NextTapCapture;

struct SparkGlm5NextTapRing
{
	uint32_t lane_count;
	uint32_t window_positions;
	uint32_t tap_count;
	uint32_t tap_width_elements;
	uint16_t *rows_bf16;
	_Atomic uint32_t *position_stamps;
	_Atomic uint32_t *lane_anchors;
	void *stream;
};

static uint64_t SparkGlm5NextTapRingEntryOffset(
	const SparkGlm5NextTapRing *ring,
	uint32_t lane,
	uint32_t position)
{
	uint64_t slot;
	slot = (uint64_t)lane * ring->window_positions + (position % ring->window_positions);
	return(slot * ring->tap_count * ring->tap_width_elements);
}

static void CUDART_CB SparkGlm5NextTapRingInvalidateHost(void *context)
{
	SparkGlm5NextTapCapture *capture = (SparkGlm5NextTapCapture *)context;
	SparkGlm5NextTapRing *ring;
	uint32_t row;
	if ( capture == 0 || capture->ring == 0 )
		return;
	ring = capture->ring;
	for (row=0u; row<capture->row_count; row++)
		atomic_store_explicit(
			&ring->position_stamps[(uint64_t)capture->row_lanes[row] * ring->window_positions + (capture->row_positions[row] % ring->window_positions)],
			SPARK_GLM5_NEXT_TAP_RING_STAMP_INVALID,memory_order_release);
}

static void CUDART_CB SparkGlm5NextTapRingStampHost(void *context)
{
	SparkGlm5NextTapCapture *capture = (SparkGlm5NextTapCapture *)context;
	SparkGlm5NextTapRing *ring;
	uint32_t row;
	if ( capture == 0 )
		return;
	ring = capture->ring;
	if ( ring != 0 )
		for (row=0u; row<capture->row_count; row++)
			atomic_store_explicit(
				&ring->position_stamps[(uint64_t)capture->row_lanes[row] * ring->window_positions + (capture->row_positions[row] % ring->window_positions)],
				capture->row_positions[row] + 1u,memory_order_release);
	free(capture->row_lanes);
	free(capture);
}

SparkStatus SparkGlm5NextTapRingCreate(
	uint32_t lane_count,
	uint32_t window_positions,
	uint32_t tap_count,
	uint32_t tap_width_elements,
	SparkGlm5NextTapRing **ring_out)
{
	SparkGlm5NextTapRing *ring;
	uint64_t stamp_count,elements;
	cudaError_t error;
	if ( ring_out == 0 || lane_count == 0u || window_positions == 0u || tap_count == 0u || tap_width_elements == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	stamp_count = (uint64_t)lane_count * window_positions;
	elements = stamp_count * tap_count * tap_width_elements;
	if ( stamp_count / lane_count != window_positions || elements / tap_count / tap_width_elements != stamp_count || elements > UINT64_MAX / sizeof(uint16_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	ring = (SparkGlm5NextTapRing *)calloc(1u,sizeof(*ring));
	if ( ring == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	ring->lane_count = lane_count;
	ring->window_positions = window_positions;
	ring->tap_count = tap_count;
	ring->tap_width_elements = tap_width_elements;
	ring->position_stamps = (_Atomic uint32_t *)calloc(stamp_count,sizeof(uint32_t));
	ring->lane_anchors = (_Atomic uint32_t *)calloc(lane_count,sizeof(uint32_t));
	if ( ring->position_stamps == 0 || ring->lane_anchors == 0 )
	{
		SparkGlm5NextTapRingDestroy(ring);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	error = cudaHostAlloc((void **)&ring->rows_bf16,elements * sizeof(uint16_t),cudaHostAllocPortable);
	if ( error == cudaSuccess )
		error = cudaStreamCreateWithFlags((cudaStream_t *)&ring->stream,cudaStreamNonBlocking);
	if ( error != cudaSuccess )
	{
		SparkStatus status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_TAP_RING_TAG,error,"tap_ring_create");
		SparkGlm5NextTapRingDestroy(ring);
		SPARK_RETURN(status);
	}
	*ring_out = ring;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm5NextTapRingEnqueueCapture(
	SparkGlm5NextTapRing *ring,
	void *op_stream,
	void *op_event,
	void *done_event,
	const uint16_t *rows_device_bf16,
	uint32_t row_count,
	const uint32_t *row_lanes,
	const uint32_t *row_positions,
	uint32_t tap_index)
{
	SparkGlm5NextTapCapture *capture;
	cudaError_t error;
	uint32_t row;
	uint64_t width_bytes;
	if ( ring == 0 || ring->stream == 0 || op_stream == 0 || op_event == 0 || done_event == 0 || rows_device_bf16 == 0 || row_count == 0u || row_lanes == 0 || row_positions == 0 || tap_index >= ring->tap_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (row=0u; row<row_count; row++)
		if ( row_lanes[row] >= ring->lane_count || row_positions[row] == UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	capture = (SparkGlm5NextTapCapture *)calloc(1u,sizeof(*capture));
	if ( capture == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	capture->row_lanes = (uint32_t *)malloc((uint64_t)row_count * sizeof(uint32_t) * 2u);
	if ( capture->row_lanes == 0 )
	{
		free(capture);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	capture->row_positions = capture->row_lanes + row_count;
	capture->ring = ring;
	capture->row_count = row_count;
	memcpy(capture->row_lanes,row_lanes,(uint64_t)row_count * sizeof(uint32_t));
	memcpy(capture->row_positions,row_positions,(uint64_t)row_count * sizeof(uint32_t));
	width_bytes = (uint64_t)ring->tap_width_elements * sizeof(uint16_t);
	error = cudaStreamWaitEvent((cudaStream_t)ring->stream,(cudaEvent_t)op_event,0u);
	if ( error == cudaSuccess )
		error = cudaLaunchHostFunc((cudaStream_t)ring->stream,SparkGlm5NextTapRingInvalidateHost,capture);
	if ( error != cudaSuccess )
	{
		free(capture->row_lanes);
		free(capture);
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_TAP_RING_TAG,error,"tap_capture_begin"));
	}
	for (row=0u; row<row_count && error==cudaSuccess; row++)
	{
		uint64_t entry = SparkGlm5NextTapRingEntryOffset(ring,row_lanes[row],row_positions[row]);
		error = cudaMemcpyAsync(
			ring->rows_bf16 + entry + (uint64_t)tap_index * ring->tap_width_elements,
			rows_device_bf16 + (uint64_t)row * ring->tap_width_elements,
			width_bytes,cudaMemcpyDeviceToHost,(cudaStream_t)ring->stream);
	}
	if ( error == cudaSuccess )
		error = cudaEventRecord((cudaEvent_t)done_event,(cudaStream_t)ring->stream);
	if ( error == cudaSuccess )
		error = cudaLaunchHostFunc((cudaStream_t)ring->stream,SparkGlm5NextTapRingStampHost,capture);
	if ( error != cudaSuccess )
	{
		SparkStatus status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_TAP_RING_TAG,error,"tap_capture_enqueue");
		if ( cudaStreamSynchronize((cudaStream_t)ring->stream) != cudaSuccess )
			fprintf(stderr,"glm5_next tap ring: stream drain failed during capture error cleanup\n");
		free(capture->row_lanes);
		free(capture);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm5NextTapRingRead(
	const SparkGlm5NextTapRing *ring,
	uint32_t lane,
	uint32_t position,
	const uint16_t **tap_row_out)
{
	uint32_t anchor,stamp;
	if ( ring == 0 || tap_row_out == 0 || lane >= ring->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	anchor = atomic_load_explicit(&ring->lane_anchors[lane],memory_order_acquire);
	if ( anchor == 0u || position >= anchor || anchor - position > ring->window_positions )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	stamp = atomic_load_explicit(
		&ring->position_stamps[(uint64_t)lane * ring->window_positions + (position % ring->window_positions)],
		memory_order_acquire);
	if ( stamp == SPARK_GLM5_NEXT_TAP_RING_STAMP_INVALID || stamp != position + 1u )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	*tap_row_out = ring->rows_bf16 + SparkGlm5NextTapRingEntryOffset(ring,lane,position);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm5NextTapRingCommitAnchor(
	SparkGlm5NextTapRing *ring,
	uint32_t lane,
	uint32_t next_position)
{
	if ( ring == 0 || lane >= ring->lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	atomic_store_explicit(&ring->lane_anchors[lane],next_position,memory_order_release);
	return(SPARK_STATUS_OK);
}

void SparkGlm5NextTapRingDestroy(SparkGlm5NextTapRing *ring)
{
	if ( ring == 0 )
		return;
	if ( ring->stream != 0 )
	{
		if ( cudaStreamSynchronize((cudaStream_t)ring->stream) != cudaSuccess )
			fprintf(stderr,"glm5_next tap ring: stream drain failed at destroy\n");
		(void)cudaStreamDestroy((cudaStream_t)ring->stream);
	}
	free(ring->position_stamps);
	free(ring->lane_anchors);
	if ( ring->rows_bf16 != 0 )
		(void)cudaFreeHost(ring->rows_bf16);
	free(ring);
}
