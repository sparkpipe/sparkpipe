#include "sparkpipe/spark_k3_pool_sizing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

#define RESET_LAYERS 2u
#define RESET_SEQUENCES 3u

typedef struct ResetPools
{
	uint8_t *pool[SPARK_K3_SLOT_POOLS];
	uint64_t bytes[SPARK_K3_SLOT_POOLS];
} ResetPools;

static void touched_region(uint32_t pool, uint32_t layer, uint32_t slot,
	uint32_t tp_degree, uint64_t *offset, uint64_t *bytes)
{
	const uint64_t heads = SPARK_K3_MODEL_KDA_HEAD_COUNT / tp_degree;
	const uint64_t kernel = SPARK_K3_MODEL_KDA_CONV_KERNEL;
	uint64_t full_channels = pool == SPARK_K3_SLOT_POOL_V_WINDOW ?
		SPARK_K3_MODEL_KDA_VALUE_DIMENSION : SPARK_K3_MODEL_KDA_QK_DIMENSION;
	uint64_t channels = heads * (pool == SPARK_K3_SLOT_POOL_V_WINDOW ?
		SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION : SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION);
	if ( pool == SPARK_K3_SLOT_POOL_STATE )
	{
		*offset = ((uint64_t)layer * RESET_SEQUENCES * SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER) +
			((uint64_t)slot * SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER);
		*bytes = heads * SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION *
			SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION * SPARK_K3_MODEL_KDA_STATE_ELEMENT_BYTES;
		return;
	}
	*offset = (((uint64_t)layer * RESET_SEQUENCES * full_channels * kernel) +
		((uint64_t)slot * channels * kernel)) * 2u;
	*bytes = channels * kernel * 2u;
}

static void run_request(ResetPools *pools, uint32_t slot, uint32_t tp_degree, uint8_t value)
{
	for ( uint32_t pool = 0u; pool < SPARK_K3_SLOT_POOLS; pool++ )
		for ( uint32_t layer = 0u; layer < RESET_LAYERS; layer++ )
		{
			uint64_t offset, bytes;
			touched_region(pool, layer, slot, tp_degree, &offset, &bytes);
			memset(pools->pool[pool] + offset, value, bytes);
		}
}

static int region_is(const ResetPools *pools, uint32_t slot, uint32_t tp_degree, uint8_t value)
{
	for ( uint32_t pool = 0u; pool < SPARK_K3_SLOT_POOLS; pool++ )
		for ( uint32_t layer = 0u; layer < RESET_LAYERS; layer++ )
		{
			uint64_t offset, bytes;
			touched_region(pool, layer, slot, tp_degree, &offset, &bytes);
			for ( uint64_t index = 0u; index < bytes; index++ )
				if ( pools->pool[pool][offset + index] != value )
					return(0);
		}
	return(1);
}

static int check_slot_reset(uint32_t tp_degree)
{
	SparkK3SlotResetSpan spans[SPARK_K3_SLOT_RESET_SPANS_MAX];
	ResetPools pools;
	uint32_t count, pool;
	int failures = 0;
	char what[160];
	pools.bytes[SPARK_K3_SLOT_POOL_STATE] = (uint64_t)RESET_LAYERS * RESET_SEQUENCES *
		SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER;
	pools.bytes[SPARK_K3_SLOT_POOL_Q_WINDOW] = (uint64_t)RESET_LAYERS * RESET_SEQUENCES *
		SPARK_K3_MODEL_KDA_QK_DIMENSION * SPARK_K3_MODEL_KDA_CONV_KERNEL * 2u;
	pools.bytes[SPARK_K3_SLOT_POOL_K_WINDOW] = pools.bytes[SPARK_K3_SLOT_POOL_Q_WINDOW];
	pools.bytes[SPARK_K3_SLOT_POOL_V_WINDOW] = (uint64_t)RESET_LAYERS * RESET_SEQUENCES *
		SPARK_K3_MODEL_KDA_VALUE_DIMENSION * SPARK_K3_MODEL_KDA_CONV_KERNEL * 2u;
	for ( pool = 0u; pool < SPARK_K3_SLOT_POOLS; pool++ )
	{
		pools.pool[pool] = (uint8_t *)calloc(1u, pools.bytes[pool]);
		if ( pools.pool[pool] == 0 )
			return(1);
	}
	run_request(&pools, 0u, tp_degree, 0x3cu);
	run_request(&pools, 2u, tp_degree, 0x5au);
	run_request(&pools, 1u, tp_degree, 0xa5u);
	count = SparkK3SlotResetSpans(RESET_LAYERS, RESET_SEQUENCES, tp_degree, 1u,
		spans, SPARK_K3_SLOT_RESET_SPANS_MAX);
	snprintf(what, sizeof(what), "TP%u: a slot reset names every KDA layer's state and q/k/v windows", tp_degree);
	failures += expect(count == RESET_LAYERS * SPARK_K3_SLOT_POOLS, what);
	for ( uint32_t index = 0u; index < count; index++ )
	{
		if ( spans[index].offset + spans[index].bytes > pools.bytes[spans[index].pool] )
			{ failures += expect(0, "a reset span stays inside its pool"); break; }
		memset(pools.pool[spans[index].pool] + spans[index].offset, 0, spans[index].bytes);
	}
	snprintf(what, sizeof(what), "TP%u: the next request on the released slot starts from zero state and windows", tp_degree);
	failures += expect(region_is(&pools, 1u, tp_degree, 0u), what);
	snprintf(what, sizeof(what), "TP%u: the other slots keep their state and windows", tp_degree);
	failures += expect(region_is(&pools, 0u, tp_degree, 0x3cu) &&
		region_is(&pools, 2u, tp_degree, 0x5au), what);
	for ( pool = 0u; pool < SPARK_K3_SLOT_POOLS; pool++ )
		free(pools.pool[pool]);
	return(failures);
}

static int check_slot_reset_refusals(void)
{
	SparkK3SlotResetSpan spans[SPARK_K3_SLOT_RESET_SPANS_MAX];
	int failures = 0;
	failures += expect(SparkK3SlotResetSpans(RESET_LAYERS, RESET_SEQUENCES, 4u,
		RESET_SEQUENCES, spans, SPARK_K3_SLOT_RESET_SPANS_MAX) == 0u,
		"a slot past the pool's sequences is refused");
	failures += expect(SparkK3SlotResetSpans(RESET_LAYERS, RESET_SEQUENCES, 5u, 0u,
		spans, SPARK_K3_SLOT_RESET_SPANS_MAX) == 0u,
		"a TP degree that does not divide the KDA heads is refused");
	failures += expect(SparkK3SlotResetSpans(RESET_LAYERS, RESET_SEQUENCES, 4u, 0u,
		spans, RESET_LAYERS * SPARK_K3_SLOT_POOLS - 1u) == 0u,
		"a span array too small for the slice is refused");
	failures += expect(SparkK3SlotResetSpans(SPARK_K3_MODEL_KDA_LAYER_COUNT + 1u,
		RESET_SEQUENCES, 4u, 0u, spans, SPARK_K3_SLOT_RESET_SPANS_MAX) == 0u,
		"more KDA layers than the model has are refused");
	return(failures);
}

int main(void)
{
	static const uint32_t stage_mla[SPARK_K3_PP_STAGE_COUNT] = { 6u, 5u, 6u, 7u };
	const uint64_t kda_slot_per_layer = 6586368ull;
	const uint64_t mla_bytes_per_layer = 1152ull;
	SparkK3PoolSizing whole, stage, last, first;
	uint64_t kda_slot_sum = 0u, mla_bytes_sum = 0u;
	uint32_t mla_sum = 0u, kda_sum = 0u, index;
	char what[128];
	int failures = 0;
	SparkK3PoolSizingForSlice(0u, SPARK_K3_MODEL_LAYER_COUNT, &whole);
	failures += expect(whole.first_layer == 0u && whole.layer_count == 93u &&
		whole.mla_layer_count == 24u && whole.kda_layer_count == 69u,
		"whole model 24 MLA + 69 KDA");
	failures += expect(whole.kda_slot_bytes_per_sequence == 69ull * kda_slot_per_layer,
		"whole-model KDA slot is 69 layers of 6586368 bytes");
	failures += expect(whole.mla_bytes_per_token == 24ull * mla_bytes_per_layer,
		"whole-model MLA cache is 24 layers of 1152 bytes a token");
	for (index = 0u; index < SPARK_K3_PP_STAGE_COUNT; index++)
	{
		const uint32_t layers = SPARK_K3_PP_STAGE_LAYERS(index);
		SparkK3PoolSizingForSlice(SPARK_K3_PP_STAGE_FIRST(index), layers, &stage);
		snprintf(what, sizeof(what), "stage %u (layers %u..%u) %u MLA + %u KDA",
			index, SPARK_K3_PP_STAGE_FIRST(index),
			SPARK_K3_PP_STAGE_FIRST(index) + layers - 1u, stage_mla[index],
			layers - stage_mla[index]);
		failures += expect(stage.mla_layer_count == stage_mla[index] &&
			stage.kda_layer_count == layers - stage_mla[index], what);
		snprintf(what, sizeof(what), "stage %u budgets %u KDA slots and %u MLA entries",
			index, layers - stage_mla[index], stage_mla[index]);
		failures += expect(stage.kda_slot_bytes_per_sequence ==
			(uint64_t)(layers - stage_mla[index]) * kda_slot_per_layer &&
			stage.mla_bytes_per_token == (uint64_t)stage_mla[index] * mla_bytes_per_layer,
			what);
		mla_sum += stage.mla_layer_count;
		kda_sum += stage.kda_layer_count;
		kda_slot_sum += stage.kda_slot_bytes_per_sequence;
		mla_bytes_sum += stage.mla_bytes_per_token;
	}
	failures += expect(mla_sum == whole.mla_layer_count && kda_sum == whole.kda_layer_count,
		"the stage slices partition the MLA and KDA layers");
	failures += expect(kda_slot_sum == whole.kda_slot_bytes_per_sequence &&
		mla_bytes_sum == whole.mla_bytes_per_token,
		"the stage budgets sum to the whole-model budgets");
	SparkK3PoolSizingForSlice(SPARK_K3_MODEL_LAYER_COUNT - 1u, 1u, &last);
	failures += expect(last.mla_layer_count == 1u && last.kda_layer_count == 0u &&
		last.kda_slot_bytes_per_sequence == 0u,
		"the final layer 92 is MLA although 92 % 4 != 3");
	SparkK3PoolSizingForSlice(0u, 1u, &first);
	failures += expect(first.mla_layer_count == 0u && first.kda_layer_count == 1u &&
		first.mla_bytes_per_token == 0u, "layer 0 is KDA");
	{
		static const uint32_t expect_kda[SPARK_K3_PP_STAGE_COUNT] = { 0u, 24u, 48u, 70u };
		static const uint32_t expect_mla[SPARK_K3_PP_STAGE_COUNT] = { 3u, 27u, 47u, 71u };
		static const uint32_t expect_routed[SPARK_K3_PP_STAGE_COUNT] = { 1u, 24u, 47u, 70u };
		static const uint32_t expect_dense[SPARK_K3_PP_STAGE_COUNT] = {
			0u, SPARK_K3_SLICE_NO_LAYER, SPARK_K3_SLICE_NO_LAYER, SPARK_K3_SLICE_NO_LAYER };
		SparkK3SliceKindLayers kinds;
		for (index = 0u; index < SPARK_K3_PP_STAGE_COUNT; index++)
		{
			SparkK3SliceKindLayersFor(SPARK_K3_PP_STAGE_FIRST(index),
				SPARK_K3_PP_STAGE_LAYERS(index), &kinds);
			snprintf(what, sizeof(what),
				"stage %u reads its KDA, MLA, routed and dense shapes from layers %u, %u, %u, %u",
				index, expect_kda[index], expect_mla[index], expect_routed[index],
				expect_dense[index]);
			failures += expect(kinds.kda == expect_kda[index] &&
				kinds.mla == expect_mla[index] &&
				kinds.routed == expect_routed[index] &&
				kinds.dense == expect_dense[index], what);
			failures += expect(!SparkK3LayerIsMla(kinds.kda) &&
				SparkK3LayerIsMla(kinds.mla),
				"the chosen KDA layer is KDA and the chosen MLA layer is MLA");
		}
		SparkK3SliceKindLayersFor(SPARK_K3_MODEL_LAYER_COUNT - 1u, 1u, &kinds);
		failures += expect(kinds.kda == SPARK_K3_SLICE_NO_LAYER &&
			kinds.mla == SPARK_K3_MODEL_LAYER_COUNT - 1u,
			"a slice without a KDA layer names none");
	}
	failures += check_slot_reset(1u);
	failures += check_slot_reset(4u);
	failures += check_slot_reset_refusals();
	printf("test_k3_pool_sizing: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
