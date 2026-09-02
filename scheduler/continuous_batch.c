#include "sparkpipe/spark_continuous_batch.h"

#include <stdlib.h>
#include <string.h>

typedef struct SparkContinuousBatchQueueEntry
{
	uint64_t request_id;
	uint32_t prompt_row_count;
	uint32_t output_token_budget;
	uint64_t enqueue_boundary;
	uint8_t oversize;
	uint8_t reserved0;
	uint16_t reserved1;
}
SparkContinuousBatchQueueEntry;

typedef struct SparkContinuousBatchLane
{
	uint64_t request_id;
	uint32_t phase;
	uint32_t remaining_rows;
	uint32_t remaining_budget;
	uint32_t generated_count;
	uint64_t enqueue_boundary;
	uint8_t retired;
	uint8_t admitted_via_aging;
	uint16_t reserved0;
}
SparkContinuousBatchLane;

struct SparkContinuousBatch
{
	uint32_t max_input_rows;
	uint32_t max_active_lanes;
	uint32_t starvation_bound;
	uint32_t queue_capacity;
	uint32_t lane_capacity;
	uint32_t queue_count;
	uint32_t resident_lane_count;
	uint64_t boundary_index;
	SparkContinuousBatchQueueEntry *queue;
	SparkContinuousBatchLane *lanes;
	uint32_t *scratch_rows;
	uint64_t *scratch_boundaries;
	uint8_t *scratch_excluded;
	SparkContinuousBatchStatistics statistics;
};

int32_t SparkContinuousBatchPolicyPick(
	const uint32_t *prompt_rows,
	const uint64_t *enqueue_boundary,
	const uint8_t *excluded,
	uint32_t count,
	uint64_t current_boundary,
	uint32_t starvation_bound)
{
	int32_t pick,aged;
	uint32_t index;
	if ( prompt_rows == 0 || enqueue_boundary == 0 || count == 0u )
		return(-1);
	pick = -1;
	aged = -1;
	if ( starvation_bound != 0u )
	{
		for ( index = 0u; index < count; ++index )
		{
			if ( excluded != 0 && excluded[index] != 0u )
				continue;
			if ( current_boundary < enqueue_boundary[index] + (uint64_t)starvation_bound )
				continue;
			if ( aged < 0 || enqueue_boundary[index] < enqueue_boundary[(uint32_t)aged] )
				aged = (int32_t)index;
		}
	}
	if ( aged >= 0 )
		return(aged);
	for ( index = 0u; index < count; ++index )
	{
		if ( excluded != 0 && excluded[index] != 0u )
			continue;
		if ( pick < 0 || prompt_rows[index] < prompt_rows[(uint32_t)pick] )
			pick = (int32_t)index;
	}
	return(pick);
}

static uint32_t SparkContinuousBatchAllocateSlot(
	SparkContinuousBatch *controller)
{
	uint32_t slot;
	for ( slot = 0u; slot < controller->lane_capacity; ++slot )
		if ( controller->lanes[slot].phase == SPARK_CONTINUOUS_BATCH_LANE_FREE )
			return(slot);
	return(UINT32_MAX);
}

static uint32_t SparkContinuousBatchLaneLimit(
	const SparkContinuousBatch *controller)
{
	return(controller->lane_capacity < controller->max_active_lanes ?
		controller->lane_capacity : controller->max_active_lanes);
}

static uint8_t SparkContinuousBatchOfferAged(
	const SparkContinuousBatch *controller,
	uint64_t enqueue_boundary)
{
	if ( controller->starvation_bound == 0u )
		return(0u);
	return(controller->boundary_index >=
		enqueue_boundary + (uint64_t)controller->starvation_bound);
}

static uint32_t SparkContinuousBatchJoinFits(
	const SparkContinuousBatch *controller,
	uint32_t prompt_row_count)
{
	if ( controller->resident_lane_count + 1u >
		SparkContinuousBatchLaneLimit(controller) )
		return(SPARK_CONTINUOUS_BATCH_QUEUE_LANES);
	if ( controller->resident_lane_count + prompt_row_count >
		controller->max_input_rows )
		return(SPARK_CONTINUOUS_BATCH_QUEUE_ROWS);
	return(SPARK_CONTINUOUS_BATCH_QUEUE_NONE);
}

static void SparkContinuousBatchEnqueue(
	SparkContinuousBatch *controller,
	const SparkContinuousBatchRequest *offer,
	uint8_t oversize)
{
	SparkContinuousBatchQueueEntry *entry;
	entry = &controller->queue[controller->queue_count++];
	entry->request_id = offer->request_id;
	entry->prompt_row_count = offer->prompt_row_count;
	entry->output_token_budget = offer->output_token_budget;
	entry->enqueue_boundary = controller->boundary_index;
	entry->oversize = oversize;
}

static void SparkContinuousBatchCreateLane(
	SparkContinuousBatch *controller,
	uint64_t request_id,
	uint32_t prompt_row_count,
	uint32_t output_token_budget,
	uint64_t enqueue_boundary,
	uint32_t slot)
{
	SparkContinuousBatchLane *lane;
	uint8_t aged;
	aged = SparkContinuousBatchOfferAged(controller,enqueue_boundary);
	lane = &controller->lanes[slot];
	lane->request_id = request_id;
	lane->phase = SPARK_CONTINUOUS_BATCH_LANE_PREFILL;
	lane->remaining_rows = prompt_row_count;
	lane->remaining_budget = output_token_budget;
	lane->generated_count = 0u;
	lane->enqueue_boundary = enqueue_boundary;
	lane->retired = 0u;
	lane->admitted_via_aging = aged;
	controller->resident_lane_count += 1u;
	if ( aged != 0u )
		controller->statistics.starvation_jumps += 1u;
	controller->statistics.admission_accepted += 1u;
}

static void SparkContinuousBatchAdmitFromQueue(
	SparkContinuousBatch *controller,
	uint32_t queue_index,
	uint32_t slot)
{
	SparkContinuousBatchQueueEntry *entry;
	uint64_t request_id;
	uint32_t prompt_row_count,output_token_budget;
	uint64_t enqueue_boundary;
	entry = &controller->queue[queue_index];
	request_id = entry->request_id;
	prompt_row_count = entry->prompt_row_count;
	output_token_budget = entry->output_token_budget;
	enqueue_boundary = entry->enqueue_boundary;
	SparkContinuousBatchCreateLane(controller,request_id,prompt_row_count,
		output_token_budget,enqueue_boundary,slot);
	memmove(controller->queue + queue_index,controller->queue + queue_index + 1u,
		(controller->queue_count - queue_index - 1u) * sizeof(*controller->queue));
	controller->queue_count -= 1u;
}

SparkStatus SparkContinuousBatchInitialize(
	const SparkContinuousBatchConfiguration *configuration,
	SparkContinuousBatch **controller_out)
{
	SparkContinuousBatch *controller;
	if ( controller_out == 0 || configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*controller_out = 0;
	if ( configuration->abi_version != SPARK_CONTINUOUS_BATCH_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_CONTINUOUS_BATCH_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( configuration->max_input_rows == 0u || configuration->max_active_lanes == 0u ||
		configuration->queue_capacity == 0u || configuration->lane_capacity == 0u ||
		configuration->lane_capacity > configuration->max_active_lanes ||
		configuration->max_active_lanes > configuration->max_input_rows )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	controller = (SparkContinuousBatch *)calloc(1u,sizeof(*controller));
	if ( controller == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	controller->max_input_rows = configuration->max_input_rows;
	controller->max_active_lanes = configuration->max_active_lanes;
	controller->starvation_bound = configuration->starvation_bound;
	controller->queue_capacity = configuration->queue_capacity;
	controller->lane_capacity = configuration->lane_capacity;
	controller->queue = (SparkContinuousBatchQueueEntry *)calloc(
		configuration->queue_capacity,sizeof(*controller->queue));
	controller->lanes = (SparkContinuousBatchLane *)calloc(
		configuration->lane_capacity,sizeof(*controller->lanes));
	controller->scratch_rows = (uint32_t *)calloc(
		configuration->queue_capacity,sizeof(*controller->scratch_rows));
	controller->scratch_boundaries = (uint64_t *)calloc(
		configuration->queue_capacity,sizeof(*controller->scratch_boundaries));
	controller->scratch_excluded = (uint8_t *)calloc(
		configuration->queue_capacity,sizeof(*controller->scratch_excluded));
	if ( controller->queue == 0 || controller->lanes == 0 ||
		controller->scratch_rows == 0 || controller->scratch_boundaries == 0 ||
		controller->scratch_excluded == 0 )
	{
		SparkContinuousBatchDestroy(controller);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	*controller_out = controller;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkContinuousBatchDestroy(SparkContinuousBatch *controller)
{
	if ( controller == 0 )
		return(SPARK_STATUS_OK);
	free(controller->queue);
	free(controller->lanes);
	free(controller->scratch_rows);
	free(controller->scratch_boundaries);
	free(controller->scratch_excluded);
	free(controller);
	return(SPARK_STATUS_OK);
}

static int SparkContinuousBatchRequestIdKnown(
	const SparkContinuousBatch *controller,
	uint64_t request_id)
{
	uint32_t index;
	for ( index = 0u; index < controller->queue_count; ++index )
		if ( controller->queue[index].request_id == request_id )
			return(1);
	for ( index = 0u; index < controller->lane_capacity; ++index )
		if ( controller->lanes[index].phase != SPARK_CONTINUOUS_BATCH_LANE_FREE &&
			controller->lanes[index].request_id == request_id )
			return(1);
	return(0);
}

SparkStatus SparkContinuousBatchOffer(
	SparkContinuousBatch *controller,
	const SparkContinuousBatchRequest *offer,
	SparkContinuousBatchDecision *decision_out)
{
	SparkContinuousBatchDecision decision;
	uint32_t reason,eligible_ahead;
	if ( controller == 0 || offer == 0 || decision_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( offer->abi_version != SPARK_CONTINUOUS_BATCH_ABI_VERSION ||
		offer->descriptor_bytes != SPARK_CONTINUOUS_BATCH_REQUEST_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	memset(decision_out,0,sizeof(*decision_out));
	decision.abi_version = SPARK_CONTINUOUS_BATCH_ABI_VERSION;
	decision.descriptor_bytes = SPARK_CONTINUOUS_BATCH_DECISION_BYTES;
	if ( offer->request_id == 0u || offer->prompt_row_count == 0u ||
		offer->output_token_budget == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkContinuousBatchRequestIdKnown(controller,offer->request_id) != 0 )
		return(SPARK_STATUS_DUPLICATE);
	controller->statistics.admission_requests += 1u;
	for ( eligible_ahead = 0u; eligible_ahead < controller->queue_count; ++eligible_ahead )
		if ( controller->queue[eligible_ahead].oversize == 0u )
			break;
	reason = SparkContinuousBatchJoinFits(controller,offer->prompt_row_count);
	if ( offer->prompt_row_count > controller->max_input_rows )
		reason = SPARK_CONTINUOUS_BATCH_QUEUE_OVERSIZE;
	else if ( reason == SPARK_CONTINUOUS_BATCH_QUEUE_NONE &&
		eligible_ahead != controller->queue_count )
		reason = SPARK_CONTINUOUS_BATCH_QUEUE_AHEAD;
	if ( reason == SPARK_CONTINUOUS_BATCH_QUEUE_NONE )
	{
		SparkContinuousBatchCreateLane(controller,offer->request_id,
			offer->prompt_row_count,offer->output_token_budget,
			controller->boundary_index,
			SparkContinuousBatchAllocateSlot(controller));
		decision.outcome = SPARK_CONTINUOUS_BATCH_ADMITTED;
		*decision_out = decision;
		return(SPARK_STATUS_OK);
	}
	if ( controller->queue_count == controller->queue_capacity )
	{
		decision.outcome = SPARK_CONTINUOUS_BATCH_QUEUED;
		decision.queue_reason = SPARK_CONTINUOUS_BATCH_QUEUE_FULL;
		controller->statistics.admission_queue_full += 1u;
		*decision_out = decision;
		return(SPARK_STATUS_OK);
	}
	SparkContinuousBatchEnqueue(controller,offer,
		reason == SPARK_CONTINUOUS_BATCH_QUEUE_OVERSIZE ? 1u : 0u);
	decision.outcome = SPARK_CONTINUOUS_BATCH_QUEUED;
	decision.queue_reason = reason;
	decision.queue_position = controller->queue_count - 1u;
	if ( reason == SPARK_CONTINUOUS_BATCH_QUEUE_OVERSIZE )
		controller->statistics.admission_oversize += 1u;
	*decision_out = decision;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkContinuousBatchWithdraw(
	SparkContinuousBatch *controller,
	uint64_t request_id)
{
	uint32_t index;
	if ( controller == 0 || request_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for ( index = 0u; index < controller->queue_count; ++index )
	{
		if ( controller->queue[index].request_id != request_id )
			continue;
		memmove(controller->queue + index,controller->queue + index + 1u,
			(controller->queue_count - index - 1u) * sizeof(*controller->queue));
		controller->queue_count -= 1u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkContinuousBatchRetire(
	SparkContinuousBatch *controller,
	uint64_t request_id)
{
	SparkContinuousBatchLane *lane;
	uint32_t slot;
	if ( controller == 0 || request_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for ( slot = 0u; slot < controller->lane_capacity; ++slot )
	{
		lane = &controller->lanes[slot];
		if ( lane->phase == SPARK_CONTINUOUS_BATCH_LANE_FREE || lane->request_id != request_id )
			continue;
		lane->retired = 1u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkContinuousBatchStep(
	SparkContinuousBatch *controller,
	SparkContinuousBatchStepEvent *events,
	uint32_t events_capacity,
	SparkContinuousBatchStepReport *report_out)
{
	SparkContinuousBatchLane *lane;
	SparkContinuousBatchStepReport report;
	uint32_t slot,decode_lanes,prefill_budget,chunk;
	if ( controller == 0 || report_out == 0 || events == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( events_capacity < controller->lane_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&report,0,sizeof(report));
	report.abi_version = SPARK_CONTINUOUS_BATCH_ABI_VERSION;
	report.descriptor_bytes = SPARK_CONTINUOUS_BATCH_STEP_REPORT_BYTES;
	decode_lanes = 0u;
	for ( slot = 0u; slot < controller->lane_capacity; ++slot )
		if ( controller->lanes[slot].phase == SPARK_CONTINUOUS_BATCH_LANE_DECODE )
			decode_lanes += 1u;
	prefill_budget = controller->max_input_rows - decode_lanes;
	report.rows_spent = 0u;
	for ( slot = 0u; slot < controller->lane_capacity; ++slot )
	{
		lane = &controller->lanes[slot];
		if ( lane->phase == SPARK_CONTINUOUS_BATCH_LANE_FREE ||
			lane->phase == SPARK_CONTINUOUS_BATCH_LANE_FINISHED )
			continue;
		if ( lane->retired != 0u )
		{
			lane->phase = SPARK_CONTINUOUS_BATCH_LANE_FINISHED;
			report.finished_count += 1u;
			continue;
		}
		if ( lane->phase == SPARK_CONTINUOUS_BATCH_LANE_PREFILL )
		{
			chunk = lane->remaining_rows < prefill_budget ?
				lane->remaining_rows : prefill_budget;
			lane->remaining_rows -= chunk;
			prefill_budget -= chunk;
			report.rows_spent += chunk;
			if ( lane->remaining_rows != 0u )
				continue;
			lane->phase = SPARK_CONTINUOUS_BATCH_LANE_DECODE;
			continue;
		}
		lane->generated_count += 1u;
		events[report.event_count].request_id = lane->request_id;
		events[report.event_count].slot = slot;
		events[report.event_count].token_index = lane->generated_count;
		report.event_count += 1u;
		report.rows_spent += 1u;
		if ( lane->generated_count >= lane->remaining_budget )
		{
			lane->phase = SPARK_CONTINUOUS_BATCH_LANE_FINISHED;
			report.finished_count += 1u;
		}
	}
	*report_out = report;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkContinuousBatchBoundary(
	SparkContinuousBatch *controller,
	uint64_t *released_request_ids,
	uint32_t released_capacity,
	uint32_t *released_count_out)
{
	uint32_t index,slot,reason;
	int32_t pick;
	if ( controller == 0 || released_count_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( released_request_ids == 0 && released_capacity != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( released_capacity < controller->queue_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*released_count_out = 0u;
	controller->boundary_index += 1u;
	controller->statistics.boundary_count += 1u;
	for ( slot = 0u; slot < controller->lane_capacity; ++slot )
	{
		if ( controller->lanes[slot].phase == SPARK_CONTINUOUS_BATCH_LANE_FREE )
			continue;
		if ( controller->lanes[slot].phase != SPARK_CONTINUOUS_BATCH_LANE_FINISHED &&
			controller->lanes[slot].retired == 0u )
			continue;
		controller->lanes[slot].phase = SPARK_CONTINUOUS_BATCH_LANE_FREE;
		controller->lanes[slot].request_id = 0u;
		controller->resident_lane_count -= 1u;
		controller->statistics.slot_reclaims += 1u;
	}
	while ( *released_count_out < released_capacity &&
		controller->queue_count != 0u )
	{
		for ( index = 0u; index < controller->queue_count; ++index )
		{
			controller->scratch_rows[index] =
				controller->queue[index].prompt_row_count;
			controller->scratch_boundaries[index] =
				controller->queue[index].enqueue_boundary;
			controller->scratch_excluded[index] =
				controller->queue[index].oversize;
		}
		pick = SparkContinuousBatchPolicyPick(controller->scratch_rows,
			controller->scratch_boundaries,controller->scratch_excluded,
			controller->queue_count,controller->boundary_index,
			controller->starvation_bound);
		if ( pick < 0 )
			break;
		reason = SparkContinuousBatchJoinFits(controller,
			controller->queue[(uint32_t)pick].prompt_row_count);
		if ( reason != SPARK_CONTINUOUS_BATCH_QUEUE_NONE )
		{
			if ( reason == SPARK_CONTINUOUS_BATCH_QUEUE_LANES )
				controller->statistics.admission_queued_lanes += 1u;
			else
				controller->statistics.admission_queued_rows += 1u;
			break;
		}
		slot = SparkContinuousBatchAllocateSlot(controller);
		released_request_ids[*released_count_out] =
			controller->queue[(uint32_t)pick].request_id;
		*released_count_out += 1u;
		SparkContinuousBatchAdmitFromQueue(controller,(uint32_t)pick,slot);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkContinuousBatchGetStatistics(
	const SparkContinuousBatch *controller,
	SparkContinuousBatchStatistics *statistics_out)
{
	if ( controller == 0 || statistics_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*statistics_out = controller->statistics;
	statistics_out->resident_lane_count = controller->resident_lane_count;
	statistics_out->queued_offer_count = controller->queue_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkContinuousBatchGetLane(
	const SparkContinuousBatch *controller,
	uint32_t slot,
	SparkContinuousBatchLaneView *lane_out)
{
	const SparkContinuousBatchLane *lane;
	if ( controller == 0 || lane_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( slot >= controller->lane_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	lane = &controller->lanes[slot];
	memset(lane_out,0,sizeof(*lane_out));
	lane_out->request_id = lane->request_id;
	lane_out->slot = slot;
	lane_out->phase = lane->phase;
	lane_out->remaining_rows = lane->remaining_rows;
	lane_out->remaining_budget = lane->remaining_budget;
	lane_out->generated_count = lane->generated_count;
	lane_out->enqueue_boundary = lane->enqueue_boundary;
	lane_out->retired = lane->retired;
	lane_out->admitted_via_aging = lane->admitted_via_aging;
	return(SPARK_STATUS_OK);
}
