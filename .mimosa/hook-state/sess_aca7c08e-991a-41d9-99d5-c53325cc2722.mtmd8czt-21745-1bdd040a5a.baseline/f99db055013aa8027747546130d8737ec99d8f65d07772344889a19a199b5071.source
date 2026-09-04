#include "sparkpipe/spark_k3_pool_sizing.h"

#include <stdio.h>

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

int main(void)
{
	SparkK3PoolSizing whole, stage0, stage1, stage2, stage3;
	int failures = 0;
	SparkK3PoolSizingForSlice(0u, 93u, &whole);
	SparkK3PoolSizingForSlice(0u, 24u, &stage0);
	SparkK3PoolSizingForSlice(24u, 23u, &stage1);
	SparkK3PoolSizingForSlice(47u, 23u, &stage2);
	SparkK3PoolSizingForSlice(70u, 23u, &stage3);

	failures += expect(whole.mla_layer_count == 24u &&
		whole.kda_layer_count == 69u, "whole model 23 MLA + 69 KDA");
	failures += expect(stage0.mla_layer_count == 6u &&
		stage0.kda_layer_count == 18u, "stage0 6 MLA + 18 KDA");
	failures += expect(stage1.mla_layer_count == 5u &&
		stage1.kda_layer_count == 18u, "stage1 6 MLA + 17 KDA");
	failures += expect(stage2.mla_layer_count == 6u &&
		stage2.kda_layer_count == 17u, "stage2 6 MLA + 17 KDA");
	failures += expect(stage3.mla_layer_count == 7u &&
		stage3.kda_layer_count == 16u, "stage3 5 MLA + 18 KDA");

	failures += expect(whole.kda_slot_bytes_per_sequence ==
		69ull * 6586368ull, "whole KDA slot 448MB");
	failures += expect(stage1.kda_slot_bytes_per_sequence ==
		18ull * 6586368ull, "stage1 KDA slot ~113MB");
	{
	uint64_t slices_sum = stage0.kda_slot_bytes_per_sequence +
		stage1.kda_slot_bytes_per_sequence + stage2.kda_slot_bytes_per_sequence +
		stage3.kda_slot_bytes_per_sequence;
	failures += expect(slices_sum == whole.kda_slot_bytes_per_sequence,
		"slice KDA budgets sum to the whole");
}
	failures += expect(whole.mla_bytes_per_token ==
		24ull * 1152ull, "whole MLA 27.6KB/token");
	failures += expect(stage1.mla_bytes_per_token ==
		5ull * 1152ull, "stage1 MLA 5.8KB/token");

	printf("test_k3_pool_sizing: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
