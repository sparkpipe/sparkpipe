#pragma once

#include <stdint.h>

uint32_t SparkModelBatchSchedulerPlanGroupSize(
	uint32_t queued,
	uint32_t maximum_group_size,
	uint32_t minimum_efficient_group_size);
/* Tail bypasses count scheduler scans while inflight work guarantees a wakeup. */
uint32_t SparkModelBatchSchedulerChooseWorkKind(
	const uint32_t queued_by_kind[4],
	const uint32_t minimum_by_kind[4],
	uint32_t admission_open,
	uint32_t inflight_submission_count,
	uint32_t bypass_limit,
	uint32_t *next_work_kind,
	uint32_t bypass_count_by_kind[4]);
