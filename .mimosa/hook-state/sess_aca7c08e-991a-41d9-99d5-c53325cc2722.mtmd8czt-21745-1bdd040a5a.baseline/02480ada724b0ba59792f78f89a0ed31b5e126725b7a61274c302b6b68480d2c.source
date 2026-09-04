#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_status.h"

#include <stdio.h>

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

int main(int argc, char **argv)
{
	SparkK3Pack pack;
	SparkK3BoundLayer bound;
	const SparkK3PackEntry *entry;
	int failures = 0;
	if ( argc != 2 )
	{
		fprintf(stderr, "usage: test_k3_bind PACK\n");
		return(2);
	}
	failures += expect(SparkK3PackOpen(argv[1], &pack) == SPARK_STATUS_OK, "open");
	failures += expect(SparkK3BindLayer(&pack, 0u, &bound) == SPARK_STATUS_OK,
		"bind layer 0 (dense KDA)");
	failures += expect(bound.layer_is_gdn == 1u, "layer 0 is GDN");
	failures += expect(bound.layer_is_dense == 1u, "layer 0 is dense");
	failures += expect(bound.tensor_count == 17u, "layer 0 binds 20 tensors");
	entry = SparkK3BoundEntry(&bound, "kda_gate_weight");
	failures += expect(entry != 0, "gate present");
	if ( entry != 0 )
		failures += expect(entry->shape[0] == 12288u && entry->shape[1] == 7168u,
			"gate shape 12288x7168");
	failures += expect(SparkK3BoundPayload(&pack, &bound, "dense_gate_up_weight") != 0,
		"dense payload resolves");
	failures += expect(SparkK3BoundEntry(&bound, "expert_w1_weight") == 0,
		"no expert tensors on the dense layer");
	failures += expect(SparkK3BindLayer(&pack, 1u, &bound) == SPARK_STATUS_OK,
		"bind layer 1 (routed KDA)");
	failures += expect(bound.layer_is_dense == 0u, "layer 1 is routed");
	failures += expect(SparkK3BoundEntry(&bound, "expert_w1_weight") != 0,
		"layer 1 expert w1 present");
	failures += expect(SparkK3BindLayer(&pack, 3u, &bound) == SPARK_STATUS_OK,
		"bind layer 3 (routed MLA)");
	failures += expect(bound.layer_is_gdn == 0u, "layer 3 is MLA");
	failures += expect(SparkK3BoundEntry(&bound, "mla_gate_weight") != 0,
		"layer 3 mla_gate present");
	SparkK3PackClose(&pack);
	printf("test_k3_bind: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
