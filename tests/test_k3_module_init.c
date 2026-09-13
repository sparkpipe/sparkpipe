#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "sparkpipe/spark_status.h"

#include <stdio.h>

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

int main(int argc, char **argv)
{
	SparkK3ModuleState state;
	SparkK3ModuleState wrong_slice;
	int failures = 0;
	if ( argc != 2 )
	{
		fprintf(stderr, "usage: test_k3_module_init PACK\n");
		return(2);
	}
	/* The 0-4 pack: slice 0+4. */
	failures += expect(SparkK3ModuleInitialize(&state, argv[1], 0u, 4u)
		== SPARK_STATUS_OK, "initialize slice 0+4");
	failures += expect(state.bound_count == 4u, "4 layers bound");
	failures += expect(state.sizing.mla_layer_count == 1u &&
		state.sizing.kda_layer_count == 3u, "sizing 1 MLA + 3 KDA");
	failures += expect(SparkK3BoundEntry(&state.bound[0], "kda_gate_weight") != 0,
		"layer 0 gate bound");
	failures += expect(SparkK3BoundEntry(&state.bound[1], "expert_w1_weight") != 0,
		"layer 1 experts bound");
	failures += expect(SparkK3BoundEntry(&state.bound[3], "mla_gate_weight") != 0,
		"layer 3 mla gate bound");
	SparkK3ModuleDestroy(&state);
	/* A mismatched slice must refuse. */
	failures += expect(SparkK3ModuleInitialize(&wrong_slice, argv[1], 0u, 3u)
		!= SPARK_STATUS_OK, "mismatched slice refused");
	SparkK3ModuleDestroy(&wrong_slice);
	printf("test_k3_module_init: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
