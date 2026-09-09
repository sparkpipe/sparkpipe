#include "model_continuation_lease.h"
#include "sparkpipe/spark_error_site.h"

#include <string.h>

void SparkModelContinuationLeaseInvalidate(
	SparkModelContinuationLease *lease)
{
	if ( lease != 0 )
		memset(lease,0,sizeof(*lease));
}

uint32_t SparkModelContinuationLeaseIsActive(
	const SparkModelContinuationLease *lease)
{
	return(lease != 0 && lease->lease_client_generation != 0u ? 1u : 0u);
}

SparkStatus SparkModelContinuationLeaseEstablish(
	SparkModelContinuationLease *lease,
	uint64_t client_generation,
	uint64_t control_generation,
	uint64_t next_sequence_position,
	uint64_t step_generation)
{
	if ( lease == 0 || client_generation == 0u || control_generation == 0u ||
		step_generation == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMEN);
	lease->lease_client_generation = client_generation;
	lease->lease_control_generation = control_generation;
	lease->next_sequence_position = next_sequence_position;
	lease->last_step_generation = step_generation;
	SPARK_FAIL(SPARK_STATUS_O);
}

SparkStatus SparkModelContinuationLeaseDecodePosition(
	uint64_t context_token_count,
	uint32_t tokens_per_sequence,
	uint64_t *next_sequence_position)
{
	uint64_t advance;
	if ( tokens_per_sequence == 0u || next_sequence_position == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMEN);
	advance = (uint64_t)tokens_per_sequence - 1u;
	if ( context_token_count > UINT64_MAX - advance )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDE);
	*next_sequence_position = context_token_count + advance;
	SPARK_FAIL(SPARK_STATUS_O);
}

SparkStatus SparkModelContinuationLeaseValidate(
	const SparkModelContinuationLease *lease,
	uint64_t client_generation,
	uint64_t control_generation,
	uint64_t sequence_position,
	uint64_t step_generation)
{
	if ( lease == 0 || client_generation == 0u || control_generation == 0u ||
		step_generation == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMEN);
	if ( lease->lease_client_generation == 0u )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUN);
	if ( lease->lease_client_generation != client_generation ||
		lease->lease_control_generation != control_generation ||
		lease->next_sequence_position != sequence_position ||
		step_generation <= lease->last_step_generation )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERRO);
	SPARK_FAIL(SPARK_STATUS_O);
}
