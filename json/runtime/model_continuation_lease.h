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
/* A deferred lease keeps every fence except the position: it is established
 * by a stage whose completion carried no emitted-token count (a transported
 * non-final stage publishes tokens_per_sequence == 0 by schema, and
 * accepted_token_count is adapter-optional), so it cannot know where the
 * sequence advanced. The next continuation re-fences against the
 * coordinator-derived position carried by the incoming lane; until then
 * Validate accepts any position while still enforcing the client and control
 * generations plus step monotonicity. */
#define SPARK_MODEL_CONTINUATION_LEASE_DEFERRED_POSITION UINT64_C(0)
SparkStatus SparkModelContinuationLeaseEstablishDeferred(
	SparkModelContinuationLease *lease,
	uint64_t client_generation,
	uint64_t control_generation,
	uint64_t step_generation);
SparkStatus SparkModelContinuationLeaseDecodePosition(
	uint64_t context_token_count,
	uint32_t tokens_per_sequence,
	uint64_t *next_sequence_position);
SparkStatus SparkModelContinuationLeaseValidate(
	const SparkModelContinuationLease *lease,
	uint64_t client_generation,
	uint64_t control_generation,
	uint64_t sequence_position,
	uint64_t step_generation);
