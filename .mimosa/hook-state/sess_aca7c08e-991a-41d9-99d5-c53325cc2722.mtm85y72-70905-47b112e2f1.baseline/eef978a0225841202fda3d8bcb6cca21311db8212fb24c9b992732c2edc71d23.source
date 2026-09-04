#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_k3_pool_sizing.h"

#include <stdio.h>
#include <string.h>

#define K3_LAYER_PREFIX "model.layers."

static SparkStatus SparkK3BindResolve(SparkK3Pack *pack,
	SparkK3BoundLayer *bound, const char *short_name)
{
	char full_name[SPARK_K3_PACK_MAX_NAME_BYTES];
	if ( bound->tensor_count >= SPARK_K3_BIND_MAX_NAMES )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	snprintf(full_name, sizeof(full_name), "%s%u.%s", K3_LAYER_PREFIX,
		bound->layer_index, short_name);
	bound->tensors[bound->tensor_count].name = short_name;
	return(SparkK3PackLoadEntry(pack, full_name,
		&bound->tensors[bound->tensor_count].entry) == SPARK_STATUS_OK ?
		(++bound->tensor_count, SPARK_STATUS_OK) : SPARK_STATUS_VALIDATION_FAILED);
}

#define BIND_ONE(bound, pack, name) \
	do { SparkStatus bind_status = SparkK3BindResolve((pack), (bound), (name)); \
		if ( bind_status != SPARK_STATUS_OK ) return(bind_status); } while (0)

static SparkStatus SparkK3BindEveryLayer(SparkK3Pack *pack,
	SparkK3BoundLayer *bound)
{
	BIND_ONE(bound, pack, "attn_norm_weight");
	BIND_ONE(bound, pack, "attnres_attn_weight");
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3BindKda(SparkK3Pack *pack, SparkK3BoundLayer *bound)
{
	static const char *const names[] =
	{
		"kda_qkv_beta_weight", "kda_q_conv_weight", "kda_k_conv_weight",
		"kda_v_conv_weight", "kda_decay_down_weight", "kda_decay_up_weight",
		"kda_decay_bias", "kda_head_log_scale", "kda_gate_weight",
		"kda_out_norm_weight", "kda_out_weight"
	};
	for ( uint32_t i = 0u; i < sizeof(names) / sizeof(names[0]); i++ )
		BIND_ONE(bound, pack, names[i]);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3BindMla(SparkK3Pack *pack, SparkK3BoundLayer *bound)
{
	static const char *const names[] =
	{
		"mla_q_down_weight", "mla_q_norm_weight", "mla_q_up_weight",
		"mla_kv_a_weight", "mla_kv_a_norm_weight", "mla_kv_b_value_weight",
		"mla_gate_weight", "mla_out_weight"
	};
	for ( uint32_t i = 0u; i < sizeof(names) / sizeof(names[0]); i++ )
		BIND_ONE(bound, pack, names[i]);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3BindMoe(SparkK3Pack *pack, SparkK3BoundLayer *bound)
{
	static const char *const names[] =
	{
		"router_weight", "router_bias", "expert_w1_weight", "expert_w2_weight",
		"shared_w1_weight", "shared_w2_weight", "routed_down_weight",
		"routed_up_weight", "routed_norm_weight"
	};
	for ( uint32_t i = 0u; i < sizeof(names) / sizeof(names[0]); i++ )
		BIND_ONE(bound, pack, names[i]);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3BindDense(SparkK3Pack *pack,
	SparkK3BoundLayer *bound)
{
	BIND_ONE(bound, pack, "dense_gate_up_weight");
	BIND_ONE(bound, pack, "dense_down_weight");
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3BindLayer(SparkK3Pack *pack, uint32_t layer_index,
	SparkK3BoundLayer *bound)
{
	SparkStatus status;
	if ( pack == 0 || bound == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( layer_index >= pack->config.total_layers )
		return(SPARK_STATUS_VALIDATION_FAILED);
	memset(bound, 0, sizeof(*bound));
	bound->layer_index = layer_index;
	bound->layer_is_gdn = !SparkK3LayerIsMla(layer_index);
	bound->layer_is_dense = (layer_index == 0u);
	status = SparkK3BindEveryLayer(pack, bound);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( bound->layer_is_gdn )
		status = SparkK3BindKda(pack, bound);
	else
		status = SparkK3BindMla(pack, bound);
	if ( status != SPARK_STATUS_OK )
		return(status);
	BIND_ONE(bound, pack, "mlp_norm_weight");
	BIND_ONE(bound, pack, "attnres_mlp_weight");
	if ( bound->layer_is_dense )
		return(SparkK3BindDense(pack, bound));
	return(SparkK3BindMoe(pack, bound));
}

const SparkK3PackEntry *SparkK3BoundEntry(const SparkK3BoundLayer *bound,
	const char *name)
{
	if ( bound == 0 || name == 0 )
		return(0);
	for ( uint32_t i = 0u; i < bound->tensor_count; i++ )
		if ( strcmp(bound->tensors[i].name, name) == 0 )
			return(&bound->tensors[i].entry);
	return(0);
}

const void *SparkK3BoundPayload(const SparkK3Pack *pack,
	const SparkK3BoundLayer *bound, const char *name)
{
	const SparkK3PackEntry *entry = SparkK3BoundEntry(bound, name);
	if ( entry == 0 )
		return(0);
	return(SparkK3PackPayload(pack, entry));
}
