#include "sparkpipe/spark_k3_resident_decode_stage_module.h"

#include <string.h>

SparkStatus SparkK3ModuleInitialize(SparkK3ModuleState *state,
	const char *pack_path, uint32_t first_layer, uint32_t layer_count)
{
	SparkStatus status;
	uint32_t layer;
	if ( state == 0 || pack_path == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(state, 0, sizeof(*state));
	status = SparkK3PackOpen(pack_path, &state->pack);
	if ( status != SPARK_STATUS_OK )
		return(status);
	/* The pack's slice must agree with the requested slice. */
	if ( state->pack.config.first_layer != first_layer ||
		state->pack.config.layers != layer_count ||
		state->pack.config.total_layers != 93u )
	{
		SparkK3ModuleDestroy(state);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( layer_count > SPARK_K3_MODULE_MAX_BOUND_LAYERS )
	{
		SparkK3ModuleDestroy(state);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->first_layer = first_layer;
	state->layer_count = layer_count;
	SparkK3PoolSizingForSlice(first_layer, layer_count, &state->sizing);
	for ( layer = first_layer; layer < first_layer + layer_count; layer++ )
	{
		status = SparkK3BindLayer(&state->pack, layer,
			&state->bound[state->bound_count]);
		if ( status != SPARK_STATUS_OK )
		{
			SparkK3ModuleDestroy(state);
			return(status);
		}
		state->bound_count++;
	}
	return(SPARK_STATUS_OK);
}

void SparkK3ModuleDestroy(SparkK3ModuleState *state)
{
	if ( state == 0 )
		return;
	SparkK3PackClose(&state->pack);
	memset(state, 0, sizeof(*state));
}
