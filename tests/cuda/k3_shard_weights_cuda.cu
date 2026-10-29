#include <stdio.h>
#include <string.h>

#include "spark_k3_weight_binds.cuh"

static int sw_failures;

static void SwCheck(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n", what);
	sw_failures++;
}

const SparkK3PackEntry *SparkK3BoundEntry(const SparkK3BoundLayer *bound, const char *name)
{
	for ( uint32_t i = 0u; bound != 0 && name != 0 && i < bound->tensor_count; i++ )
		if ( strcmp(bound->tensors[i].name, name) == 0 )
			return &bound->tensors[i].entry;
	return 0;
}

static uint64_t SwAdd(SparkK3BoundLayer *bound, const char *name, uint64_t offset, uint64_t bytes)
{
	SparkK3BoundTensor *tensor = &bound->tensors[bound->tensor_count++];
	tensor->name = name;
	memset(&tensor->entry, 0, sizeof(tensor->entry));
	snprintf(tensor->entry.name, sizeof(tensor->entry.name), "model.layers.%u.%s", bound->layer_index, name);
	tensor->entry.payload_offset = offset;
	tensor->entry.bytes = bytes;
	return offset + bytes;
}

static const void *SwField(const K3LayerWeights *weights, const char *name)
{
	for ( uint32_t i = 0u; i < (uint32_t)(sizeof(k3_weight_binds) / sizeof(k3_weight_binds[0])); i++ )
		if ( strcmp(k3_weight_binds[i].name, name) == 0 )
			return *(const void *const *)((const char *)weights + k3_weight_binds[i].offset);
	return 0;
}

int main(void)
{
	static const char *const kda_before[] =
	{
		"attn_norm_weight", "attnres_attn_weight", "kda_qkv_beta_weight", "kda_q_conv_weight", "kda_k_conv_weight", "kda_v_conv_weight",
		"kda_decay_down_weight", "kda_decay_up_weight", "kda_decay_bias", "kda_head_log_scale", "kda_gate_weight", "kda_out_norm_weight",
		"kda_out_weight", "mlp_norm_weight", "attnres_mlp_weight", "router_weight", "router_bias"
	};
	static const char *const mla_before[] =
	{
		"attn_norm_weight", "attnres_attn_weight", "mla_q_down_weight", "mla_q_norm_weight", "mla_kv_a_weight", "mla_kv_a_norm_weight",
		"mla_q_up_weight", "mla_kv_b_value_weight", "mla_gate_weight", "mla_out_weight", "mlp_norm_weight", "attnres_mlp_weight",
		"router_weight", "router_bias"
	};
	static const char *const after[] = { "shared_w1_weight", "shared_w2_weight", "routed_down_weight", "routed_up_weight", "routed_norm_weight" };
	static const char *const appended[] = { "mla_q_b_weight", "mla_kv_b_key_full_weight", "mla_kv_b_value_full_weight" };
	const uint64_t base = 1048576u;
	const uint8_t *dense_base = (const uint8_t *)(uintptr_t)0x100000000ull, *expert_block = (const uint8_t *)(uintptr_t)0x200000000ull;
	K3LayerWeights own, out;
	SparkK3BoundLayer kda, mla;
	SparkPackWindowLayer kda_ranges, mla_ranges;
	uint64_t cursor, expected_ok = 1u, checked = 0u;
	memset(&own, 0, sizeof(own));
	own.expert_interleave = 1u;
	own.expert_tile_k = 32u;
	own.attn_norm_weight = (const void *)(uintptr_t)0xdead0000ull;
	own.mla_q_b_weight = (const void *)(uintptr_t)0xdead1000ull;

	memset(&kda, 0, sizeof(kda));
	kda.layer_index = 1u;
	cursor = 4096u;
	for ( uint32_t i = 0u; i < sizeof(kda_before) / sizeof(kda_before[0]); i++ )
		cursor = SwAdd(&kda, kda_before[i], cursor, 8192u + 128u * i);
	memset(&kda_ranges, 0, sizeof(kda_ranges));
	kda_ranges.dense_count = 2u;
	kda_ranges.dense[0].offset = base + 4096u;
	kda_ranges.dense[0].bytes = cursor - 4096u;
	kda_ranges.experts.offset = base + cursor;
	cursor = SwAdd(&kda, "expert_w1_weight", cursor, 3u << 20);
	cursor = SwAdd(&kda, "expert_w2_weight", cursor, 3u << 19);
	kda_ranges.experts.bytes = base + cursor - kda_ranges.experts.offset;
	kda_ranges.dense[1].offset = base + cursor;
	for ( uint32_t i = 0u; i < sizeof(after) / sizeof(after[0]); i++ )
		cursor = SwAdd(&kda, after[i], cursor, 65536u + 256u * i);
	kda_ranges.dense[1].bytes = base + cursor - kda_ranges.dense[1].offset;

	SwCheck(SparkK3ShardWeights(&own, base, &kda, &kda_ranges, dense_base, expert_block, &out) == SPARK_K3_DISPATCH_OK, "a KDA layer binds from the window");
	for ( uint32_t i = 0u; i < kda.tensor_count; i++ )
	{
		const SparkK3PackEntry *entry = &kda.tensors[i].entry;
		const uint64_t start = base + entry->payload_offset;
		const void *want = start >= kda_ranges.experts.offset && start < kda_ranges.experts.offset + kda_ranges.experts.bytes
			? (const void *)(expert_block + (start - kda_ranges.experts.offset))
			: start < kda_ranges.dense[1].offset
				? (const void *)(dense_base + (start - kda_ranges.dense[0].offset))
				: (const void *)(dense_base + kda_ranges.dense[0].bytes + (start - kda_ranges.dense[1].offset));
		expected_ok &= SwField(&out, kda.tensors[i].name) == want ? 1u : 0u;
		checked++;
	}
	SwCheck(expected_ok != 0u, "every KDA tensor points at its place in the shard's window (dense range prefix + offset, or the expert block)");
	SwCheck(out.mla_q_b_weight == 0 && out.mla_q_down_weight == 0 && out.dense_gate_up_weight == 0, "fields the layer does not bind stay empty, not the own shard's pointers");
	SwCheck(out.expert_interleave == 1u && out.expert_tile_k == 32u, "the expert cell layout comes from the own shard");

	memset(&mla, 0, sizeof(mla));
	mla.layer_index = 3u;
	cursor = 1u << 24;
	for ( uint32_t i = 0u; i < sizeof(mla_before) / sizeof(mla_before[0]); i++ )
		cursor = SwAdd(&mla, mla_before[i], cursor, 16384u + 64u * i);
	memset(&mla_ranges, 0, sizeof(mla_ranges));
	mla_ranges.dense_count = 3u;
	mla_ranges.dense[0].offset = base + (1u << 24);
	mla_ranges.dense[0].bytes = base + cursor - mla_ranges.dense[0].offset;
	mla_ranges.experts.offset = base + cursor;
	cursor = SwAdd(&mla, "expert_w1_weight", cursor, 1u << 20);
	cursor = SwAdd(&mla, "expert_w2_weight", cursor, 1u << 19);
	mla_ranges.experts.bytes = base + cursor - mla_ranges.experts.offset;
	mla_ranges.dense[1].offset = base + cursor;
	for ( uint32_t i = 0u; i < sizeof(after) / sizeof(after[0]); i++ )
		cursor = SwAdd(&mla, after[i], cursor, 32768u);
	mla_ranges.dense[1].bytes = base + cursor - mla_ranges.dense[1].offset;
	cursor = 1ull << 34;
	mla_ranges.dense[2].offset = base + cursor;
	for ( uint32_t i = 0u; i < sizeof(appended) / sizeof(appended[0]); i++ )
		cursor = SwAdd(&mla, appended[i], cursor, 524288u * (i + 1u));
	mla_ranges.dense[2].bytes = base + cursor - mla_ranges.dense[2].offset;
	SwCheck(SparkK3ShardWeights(&own, base, &mla, &mla_ranges, dense_base, expert_block, &out) == SPARK_K3_DISPATCH_OK, "an MLA layer with appended tensors binds from three ranges");
	SwCheck(out.mla_q_b_weight == (const void *)(dense_base + mla_ranges.dense[0].bytes + mla_ranges.dense[1].bytes),
		"the appended q_b lands after the first two ranges in the shard's window");
	SwCheck(out.mla_kv_b_value_full_weight == (const void *)(dense_base + mla_ranges.dense[0].bytes + mla_ranges.dense[1].bytes + 524288u * 3u),
		"the last appended tensor keeps its offset inside the third range");
	SwCheck(out.expert_w2_weight == (const void *)(expert_block + (1u << 20)), "expert w2 follows w1 in the expert block");
	SwCheck(out.attn_norm_weight == (const void *)dense_base, "the first dense tensor starts the window");

	mla_ranges.dense_count = 2u;
	SwCheck(SparkK3ShardWeights(&own, base, &mla, &mla_ranges, dense_base, expert_block, &out) == SPARK_K3_DISPATCH_ERR_BIND,
		"a tensor outside every window range is refused, not pointed at the own shard");
	mla_ranges.dense_count = 3u;
	SwCheck(SparkK3ShardWeights(&own, base, &mla, &mla_ranges, dense_base, 0, &out) == SPARK_K3_DISPATCH_ERR_ARGUMENT,
		"a layer with experts needs the shard's expert block");
	own.routed_up_scale = (const void *)(uintptr_t)0xdead2000ull;
	SwCheck(SparkK3ShardWeights(&own, base, &mla, &mla_ranges, dense_base, expert_block, &out) == SPARK_K3_DISPATCH_ERR_BIND,
		"scaled dense tensors are refused (their scales would point at the own shard)");
	printf("%s K3 shard weights from the window: %llu KDA tensors and the MLA layer's three ranges placed, unbound fields empty, refusals for out-of-window tensors, missing experts and scales\n",
		sw_failures == 0 ? "PASS" : "FAIL", (unsigned long long)checked);
	return sw_failures != 0;
}
