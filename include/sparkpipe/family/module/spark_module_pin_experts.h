#pragma once

static SparkStatus SPARK_FAMILY(PinAllExperts)(SPARK_FAMILY(ModuleState) *state,uint32_t first_routed_layer,uint32_t expert_count)
{
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t layer,expert,count = 0u,last_layer;
	if ( state->lazy_pack == 0 || state->expert_pin_lease_count != 0u || state->first_layer_index + state->layer_count <= first_routed_layer )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	last_layer = state->first_layer_index + state->layer_count - 1u;
	layer = state->first_layer_index > first_routed_layer ? state->first_layer_index : first_routed_layer;
	for (; layer<=last_layer && status==SPARK_STATUS_OK; layer++)
		for (expert=0u; expert<expert_count && status==SPARK_STATUS_OK; expert++)
		{
			keys[count++] = (SparkWeightdExpertKey){.layer=layer,.expert=expert};
			if ( count == SPARK_WEIGHTD_LEASE_GROUPS_MAX || (layer == last_layer && expert + 1u == expert_count) )
			{
				uint64_t lease = 0u;
				void *base = 0;
				if ( state->expert_pin_lease_count >= sizeof(state->expert_pin_leases) / sizeof(state->expert_pin_leases[0]) )
					SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
				status = SparkWeightdMapAcquire(state->lazy_pack->map,keys,count,&lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
				if ( lease != 0u )
					state->expert_pin_leases[state->expert_pin_lease_count++] = lease;
				if ( status == SPARK_STATUS_OK )
					status = lease != 0u ? SparkWeightdMapBeginUse(state->lazy_pack->map,lease,&base) : SPARK_STATUS_VALIDATION_FAILED;
				if ( status == SPARK_STATUS_OK && (base == 0 || (state->expert_pin_base != 0 && state->expert_pin_base != base)) )
					status = SPARK_STATUS_VALIDATION_FAILED;
				if ( status == SPARK_STATUS_OK )
				{
					state->expert_pin_base = (const uint8_t *)base;
					state->expert_pin_key_count += count;
				}
				count = 0u;
			}
		}
	if ( status == SPARK_STATUS_OK )
		state->experts_pinned = 1u;
	SPARK_RETURN(status);
}

static SparkStatus SPARK_FAMILY(ReleasePinnedExperts)(SPARK_FAMILY(ModuleState) *state,cudaStream_t stream)
{
	SparkStatus status;
	state->experts_pinned = 0u;
	while ( state->expert_pin_lease_count != 0u )
	{
		uint64_t lease = state->expert_pin_leases[state->expert_pin_lease_count - 1u];
		status = SparkWeightdMapRecordCompletion(state->lazy_pack->map,lease,stream);
		if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND )
			SPARK_RETURN(status);
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		status = SparkWeightdMapRelease(state->lazy_pack->map,lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
		if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND )
			SPARK_RETURN(status);
		state->expert_pin_leases[--state->expert_pin_lease_count] = 0u;
	}
	state->expert_pin_base = 0;
	return(SPARK_STATUS_OK);
}
