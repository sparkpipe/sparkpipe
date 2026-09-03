#pragma once

#include <stdint.h>

uint32_t SparkModelBatchSchedulerPlanCacheBoundLaneCount(
	uint32_t maximum_lane_count,
	uint32_t physical_page_capacity,
	uint32_t inflight_page_count);
uint32_t SparkModelBatchSchedulerCacheDemandFits(
	uint32_t physical_page_capacity,
	uint32_t used_page_count,
	uint32_t additional_page_count);
uint32_t SparkModelBatchSchedulerRequestFitsPageCapacity(
	uint32_t block_token_count,
	uint32_t physical_page_capacity,
	uint32_t prompt_token_count,
	uint32_t output_token_budget);
uint32_t SparkModelBatchSchedulerPlanMixedLaneCount(
	const uint32_t queued_by_kind[4],
	const uint32_t maximum_by_kind[4],
	const uint32_t inflight_by_kind[4],
	uint32_t selected_kind,
	uint32_t submission_capacity);
uint32_t SparkModelBatchSchedulerChooseWorkKind(
	const uint32_t queued_by_kind[4],
	const uint32_t minimum_by_kind[4],
	uint32_t admission_open,
	uint32_t inflight_submission_count,
	uint32_t bypass_limit,
	uint32_t *next_work_kind,
	uint32_t bypass_count_by_kind[4]);
