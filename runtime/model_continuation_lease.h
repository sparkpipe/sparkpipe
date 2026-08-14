#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

typedef struct SparkModelContinuationLease
{
	uint64_t lease_client_generation;
	uint64_t lease_control_generation;
	uint64_t next_sequence_position;
	uint64_t last_step_generation;
} SparkModelContinuationLease;

void SparkModelContinuationLeaseInvalidate(
	SparkModelContinuationLease *lease);
uint32_t SparkModelContinuationLeaseIsActive(
	const SparkModelContinuationLease *lease);
SparkStatus SparkModelContinuationLeaseEstablish(
	SparkModelContinuationLease *lease,
	uint64_t client_generation,
	uint64_t control_generation,
	uint64_t next_sequence_position,
	uint64_t step_generation);
SparkStatus SparkModelContinuationLeaseValidate(
	const SparkModelContinuationLease *lease,
	uint64_t client_generation,
	uint64_t control_generation,
	uint64_t sequence_position,
	uint64_t step_generation);
