#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_pack_window.h"

#define WF(field, name) { offsetof(K3LayerWeights, field), name }
static const struct SparkK3WeightBind
{
	size_t offset;
	const char *name;
} k3_weight_binds[] =
{
	WF(attn_norm_weight, "attn_norm_weight"),
	WF(mlp_norm_weight, "mlp_norm_weight"),
	WF(kda_qkv_beta_weight, "kda_qkv_beta_weight"),
	WF(kda_decay_down_weight, "kda_decay_down_weight"),
	WF(kda_q_conv_weight, "kda_q_conv_weight"),
	WF(kda_k_conv_weight, "kda_k_conv_weight"),
	WF(kda_v_conv_weight, "kda_v_conv_weight"),
	WF(kda_decay_up_weight, "kda_decay_up_weight"),
	WF(kda_decay_bias, "kda_decay_bias"),
	WF(kda_head_log_scale, "kda_head_log_scale"),
	WF(kda_gate_weight, "kda_gate_weight"),
	WF(kda_out_norm_weight, "kda_out_norm_weight"),
	WF(kda_out_weight, "kda_out_weight"),
	WF(mla_q_down_weight, "mla_q_down_weight"),
	WF(mla_q_norm_weight, "mla_q_norm_weight"),
	WF(mla_q_up_weight, "mla_q_up_weight"),
	WF(mla_kv_a_weight, "mla_kv_a_weight"),
	WF(mla_kv_a_norm_weight, "mla_kv_a_norm_weight"),
	WF(mla_kv_b_value_weight, "mla_kv_b_value_weight"),
	WF(mla_q_b_weight, "mla_q_b_weight"),
	WF(mla_kv_b_key_full_weight, "mla_kv_b_key_full_weight"),
	WF(mla_kv_b_value_full_weight, "mla_kv_b_value_full_weight"),
	WF(mla_gate_weight, "mla_gate_weight"),
	WF(mla_out_weight, "mla_out_weight"),
	WF(router_weight, "router_weight"),
	WF(router_bias, "router_bias"),
	WF(routed_down_weight, "routed_down_weight"),
	WF(routed_up_weight, "routed_up_weight"),
	WF(routed_norm_weight, "routed_norm_weight"),
	WF(expert_w1_weight, "expert_w1_weight"),
	WF(expert_w2_weight, "expert_w2_weight"),
	WF(shared_w1_weight, "shared_w1_weight"),
	WF(shared_w2_weight, "shared_w2_weight"),
	WF(dense_gate_up_weight, "dense_gate_up_weight"),
	WF(dense_down_weight, "dense_down_weight"),
	WF(attnres_attn_weight, "attnres_attn_weight"),
	WF(attnres_mlp_weight, "attnres_mlp_weight"),
};
#undef WF

static inline int32_t SparkK3ShardWeights(const K3LayerWeights *own, uint64_t payload_base, const SparkK3BoundLayer *bound,
	const SparkPackWindowLayer *ranges, const uint8_t *dense_base, const uint8_t *expert_block, K3LayerWeights *out)
{
	const void *const own_scales[] =
	{
		own != 0 ? own->kda_out_scale : 0, own != 0 ? own->mla_q_down_scale : 0, own != 0 ? own->mla_q_up_scale : 0,
		own != 0 ? own->mla_kv_a_scale : 0, own != 0 ? own->mla_kv_b_scale : 0, own != 0 ? own->mla_out_scale : 0,
		own != 0 ? own->routed_down_scale : 0, own != 0 ? own->routed_up_scale : 0, own != 0 ? own->shared_w1_scale : 0,
		own != 0 ? own->shared_w2_scale : 0, own != 0 ? own->dense_gate_up_scale : 0, own != 0 ? own->dense_down_scale : 0
	};
	uint32_t index;
	if ( own == 0 || bound == 0 || ranges == 0 || dense_base == 0 || out == 0 || ranges->dense_count == 0u ||
		ranges->dense_count > SPARK_PACK_WINDOW_MAX_DENSE_RANGES )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	for (index = 0u; index < (uint32_t)(sizeof(own_scales) / sizeof(own_scales[0])); index++)
		if ( own_scales[index] != 0 )
			return SPARK_K3_DISPATCH_ERR_BIND;
	memset(out, 0, sizeof(*out));
	out->expert_interleave = own->expert_interleave;
	out->expert_tile_k = own->expert_tile_k;
	for (index = 0u; index < (uint32_t)(sizeof(k3_weight_binds) / sizeof(k3_weight_binds[0])); index++)
	{
		const SparkK3PackEntry *entry = SparkK3BoundEntry(bound, k3_weight_binds[index].name);
		const void *pointer = 0;
		uint64_t start, end, prefix = 0u;
		uint32_t range;
		if ( entry == 0 )
			continue;
		start = payload_base + entry->payload_offset;
		end = start + entry->bytes;
		if ( ranges->experts.bytes != 0u && start >= ranges->experts.offset && end <= ranges->experts.offset + ranges->experts.bytes )
		{
			if ( expert_block == 0 )
				return SPARK_K3_DISPATCH_ERR_ARGUMENT;
			pointer = expert_block + (start - ranges->experts.offset);
		}
		else
			for (range = 0u; range < ranges->dense_count && pointer == 0; range++)
			{
				if ( start >= ranges->dense[range].offset && end <= ranges->dense[range].offset + ranges->dense[range].bytes )
					pointer = dense_base + prefix + (start - ranges->dense[range].offset);
				prefix += ranges->dense[range].bytes;
			}
		if ( pointer == 0 )
			return SPARK_K3_DISPATCH_ERR_BIND;
		*(const void **)((char *)out + k3_weight_binds[index].offset) = pointer;
	}
	return SPARK_K3_DISPATCH_OK;
}
