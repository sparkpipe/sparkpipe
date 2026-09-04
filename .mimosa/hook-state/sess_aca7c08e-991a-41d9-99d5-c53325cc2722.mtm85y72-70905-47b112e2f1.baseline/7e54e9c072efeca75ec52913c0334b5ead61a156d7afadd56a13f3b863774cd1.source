#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_CONTINUOUS_BATCH_ABI_VERSION 1u

#define SPARK_CONTINUOUS_BATCH_ADMITTED 1u
#define SPARK_CONTINUOUS_BATCH_QUEUED 2u

#define SPARK_CONTINUOUS_BATCH_QUEUE_NONE 0u
#define SPARK_CONTINUOUS_BATCH_QUEUE_ROWS 1u
#define SPARK_CONTINUOUS_BATCH_QUEUE_LANES 2u
#define SPARK_CONTINUOUS_BATCH_QUEUE_OVERSIZE 3u
#define SPARK_CONTINUOUS_BATCH_QUEUE_FULL 4u
#define SPARK_CONTINUOUS_BATCH_QUEUE_AHEAD 5u

#define SPARK_CONTINUOUS_BATCH_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchConfiguration))
#define SPARK_CONTINUOUS_BATCH_REQUEST_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchRequest))
#define SPARK_CONTINUOUS_BATCH_DECISION_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchDecision))
#define SPARK_CONTINUOUS_BATCH_STEP_REPORT_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchStepReport))

#define SPARK_CONTINUOUS_BATCH_LANE_FREE 0u
#define SPARK_CONTINUOUS_BATCH_LANE_PREFILL 1u
#define SPARK_CONTINUOUS_BATCH_LANE_DECODE 2u
#define SPARK_CONTINUOUS_BATCH_LANE_FINISHED 3u

typedef struct SparkContinuousBatchConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t max_input_rows;
	uint32_t max_active_lanes;
	uint32_t starvation_bound;
	uint32_t queue_capacity;
	uint32_t lane_capacity;
	uint32_t reserved0;
}
SparkContinuousBatchConfiguration;

typedef struct SparkContinuousBatchRequest
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint64_t request_id;
	uint32_t prompt_row_count;
	uint32_t output_token_budget;
}
SparkContinuousBatchRequest;

typedef struct SparkContinuousBatchDecision
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t outcome;
	uint32_t queue_reason;
	uint32_t queue_position;
	uint32_t reserved0;
}
SparkContinuousBatchDecision;

typedef struct SparkContinuousBatchStepEvent
{
	uint64_t request_id;
	uint32_t slot;
	uint32_t token_index;
}
SparkContinuousBatchStepEvent;

typedef struct SparkContinuousBatchStepReport
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t event_count;
	uint32_t rows_spent;
	uint32_t finished_count;
}
SparkContinuousBatchStepReport;

typedef struct SparkContinuousBatchLaneView
{
	uint64_t request_id;
	uint32_t slot;
	uint32_t phase;
	uint32_t remaining_rows;
	uint32_t remaining_budget;
	uint32_t generated_count;
	uint64_t enqueue_boundary;
	uint8_t retired;
	uint8_t admitted_via_aging;
	uint16_t reserved0;
}
SparkContinuousBatchLaneView;

typedef struct SparkContinuousBatchStatistics
{
	uint64_t admission_requests;
	uint64_t admission_accepted;
	uint64_t admission_queued_rows;
	uint64_t admission_queued_lanes;
	uint64_t admission_oversize;
	uint64_t admission_queue_full;
	uint64_t boundary_count;
	uint64_t slot_reclaims;
	uint64_t starvation_jumps;
	uint32_t resident_lane_count;
	uint32_t queued_offer_count;
}
SparkContinuousBatchStatistics;

typedef struct SparkContinuousBatch SparkContinuousBatch;

int32_t SparkContinuousBatchPolicyPick(
	const uint32_t *prompt_rows,
	const uint64_t *enqueue_boundary,
	const uint8_t *excluded,
	uint32_t count,
	uint64_t current_boundary,
	uint32_t starvation_bound);

SparkStatus SparkContinuousBatchInitialize(
	const SparkContinuousBatchConfiguration *configuration,
	SparkContinuousBatch **controller_out);
SparkStatus SparkContinuousBatchDestroy(SparkContinuousBatch *controller);

SparkStatus SparkContinuousBatchOffer(
	SparkContinuousBatch *controller,
	const SparkContinuousBatchRequest *offer,
	SparkContinuousBatchDecision *decision_out);

SparkStatus SparkContinuousBatchWithdraw(
	SparkContinuousBatch *controller,
	uint64_t request_id);

SparkStatus SparkContinuousBatchRetire(
	SparkContinuousBatch *controller,
	uint64_t request_id);

SparkStatus SparkContinuousBatchStep(
	SparkContinuousBatch *controller,
	SparkContinuousBatchStepEvent *events,
	uint32_t events_capacity,
	SparkContinuousBatchStepReport *report_out);

SparkStatus SparkContinuousBatchBoundary(
	SparkContinuousBatch *controller,
	uint64_t *released_request_ids,
	uint32_t released_capacity,
	uint32_t *released_count_out);

SparkStatus SparkContinuousBatchGetStatistics(
	const SparkContinuousBatch *controller,
	SparkContinuousBatchStatistics *statistics_out);
SparkStatus SparkContinuousBatchGetLane(
	const SparkContinuousBatch *controller,
	uint32_t slot,
	SparkContinuousBatchLaneView *lane_out);

#ifdef __cplusplus
}
#endif
