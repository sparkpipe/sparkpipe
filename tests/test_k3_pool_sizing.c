#include "sparkpipe/spark_k3_pool_sizing.h"

#include <stdio.h>

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
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
	printf("test_k3_pool_sizing: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
