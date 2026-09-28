#pragma once

static SparkStatus SPARK_FAMILY(LazyRecoverLease)(SPARK_FAMILY(ModuleState) *state,uint32_t slot,SPARK_FAMILY(TpChain) **out)
{
	SPARK_FAMILY(TpChain) *chain;
	SparkStatus status = SPARK_STATUS_OK;
	*out = 0;
	chain = atomic_exchange_explicit(&state->lazy_retained[slot],0,memory_order_acq_rel);
	if ( chain == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( chain->expert_lease != 0u )
		status = SPARK_FAMILY(LazyRelease)(chain);
	if ( status != SPARK_STATUS_OK )
		atomic_store_explicit(&state->lazy_retained[slot],chain,memory_order_release);
	else
		*out = chain;
	SPARK_RETURN(status);
}
