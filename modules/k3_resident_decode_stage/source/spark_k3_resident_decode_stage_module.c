#include "inference/llms/kimi_k3/config.h"
#include "sparkpipe/spark_k3_model.h"
#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "sparkpipe/spark_error_site.h"

#include <string.h>

_Static_assert(SPARK_K3_MODULE_TOTAL_LAYERS == SPARK_K3_MODEL_LAYER_COUNT,
	"k3 module layer total must equal the model header");

_Static_assert(K3_HIDDEN == SPARK_K3_MODEL_HIDDEN_DIMENSION &&
	K3_LAYERS == SPARK_K3_MODEL_LAYER_COUNT &&
	K3_EXPERTS == SPARK_K3_MODEL_MOE_EXPERT_COUNT &&
	K3_TOP_K == SPARK_K3_MODEL_MOE_TOP_K,
	"k3 kernel config must equal the model header geometry");

SparkStatus SparkK3ModuleInitialize(SparkK3ModuleState *state,
	const char *pack_path, uint32_t first_layer, uint32_t layer_count)
{
	SparkStatus status;
	uint32_t layer;
	if ( state == 0 || pack_path == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(state, 0, sizeof(*state));
	status = SparkK3PackOpen(pack_path, &state->pack);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( first_layer == SPARK_K3_MODULE_DERIVE_SLICE ||
		layer_count == SPARK_K3_MODULE_DERIVE_SLICE )
	{
		first_layer = state->pack.config.first_layer;
		layer_count = state->pack.config.layers;
	}
	if ( state->pack.config.first_layer != first_layer ||
		state->pack.config.layers != layer_count ||
		state->pack.config.total_layers != SPARK_K3_MODEL_LAYER_COUNT )
	{
		SparkK3ModuleDestroy(state);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( layer_count > SPARK_K3_MODULE_MAX_BOUND_LAYERS )
	{
		SparkK3ModuleDestroy(state);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
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
			SPARK_RETURN(status);
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
