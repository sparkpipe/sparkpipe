/*
 * Speculation-provider slot: the one validation point for the interface.
 * See include/sparkpipe/spark_speculation_provider.h; the two binding
 * shapes (module-provider, embedded-provider) are proven by
 * tests/test_speculation_provider_slot.c.
 */

#include <string.h>

#include "sparkpipe/spark_speculation_provider.h"

SparkStatus SparkSpeculationProviderValidate(
	const SparkSpeculationProvider *provider)
{
	if ( provider == 0 || provider->descriptor == 0 || provider->ops == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( provider->descriptor->abi_version !=
		SPARK_SPECULATION_PROVIDER_ABI_VERSION ||
		provider->descriptor->descriptor_bytes !=
		SPARK_SPECULATION_PROVIDER_DESCRIPTOR_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( provider->descriptor->kind < SPARK_SPECULATION_PROVIDER_MTP ||
		provider->descriptor->kind > SPARK_SPECULATION_PROVIDER_DSPARK2 ||
		provider->descriptor->provider_id == 0 ||
		provider->descriptor->default_draft_token_count == 0u ||
		provider->descriptor->default_draft_token_count >
		provider->descriptor->max_draft_token_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( provider->ops->capability_query == 0 ||
		provider->ops->draft_begin == 0 ||
		provider->ops->draft_next == 0 ||
		provider->ops->verify_account == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}
