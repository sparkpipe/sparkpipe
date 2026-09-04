#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_status.h"

#include <stdio.h>
#include <string.h>

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

int main(int argc, char **argv)
{
	SparkK3Pack pack;
	SparkK3PackEntry entry;
	int failures = 0;
	if ( argc != 2 )
	{
		fprintf(stderr, "usage: test_k3_pack_load PACK\n");
		return(2);
	}
	failures += expect(SparkK3PackOpen(argv[1], &pack) == SPARK_STATUS_OK,
		"open pack");
	failures += expect(pack.config.hidden == 7168u, "config hidden 7168");
	failures += expect(pack.config.total_layers == 93u, "config total_layers 93");
	failures += expect(pack.config.experts == 896u, "config experts 896");
	failures += expect(pack.config.kda_heads == 96u, "config kda_heads 96");
	failures += expect(pack.config.first_layer == 0u, "config first_layer 0");
	failures += expect(pack.config.layers == 1u, "config layers 1");
	failures += expect(SparkK3PackLoadEntry(&pack,
		"model.embed_tokens.weight", &entry) == SPARK_STATUS_OK,
		"resolve embedding");
	failures += expect(entry.shape_count == 2u &&
		entry.shape[0] == 163840u && entry.shape[1] == 7168u,
		"embedding shape 163840x7168");
	failures += expect(entry.kind == SPARK_K3_PACK_KIND_BF16, "embedding bf16");
	failures += expect(SparkK3PackPayload(&pack, &entry) != 0, "embedding payload");
	failures += expect(SparkK3PackLoadEntry(&pack,
		"model.layers.0.kda_gate_weight", &entry) == SPARK_STATUS_OK,
		"resolve kda_gate_weight (full-rank gate)");
	failures += expect(entry.shape_count == 2u &&
		entry.shape[0] == 12288u && entry.shape[1] == 7168u,
		"gate shape 12288x7168");
	failures += expect(SparkK3PackLoadEntry(&pack,
		"model.layers.0.kda_decay_down_weight", &entry) == SPARK_STATUS_OK,
		"resolve kda_decay_down_weight");
	failures += expect(entry.shape_count == 2u &&
		entry.shape[0] == 128u && entry.shape[1] == 7168u,
		"decay down shape 128x7168");
	failures += expect(SparkK3PackLoadEntry(&pack,
		"model.layers.0.dense_gate_up_weight", &entry) == SPARK_STATUS_OK,
		"resolve dense_gate_up_weight (layer 0 dense)");
	failures += expect(entry.shape[0] == 67584u && entry.shape[1] == 7168u,
		"dense gate_up 67584x7168");
	failures += expect(SparkK3PackLoadEntry(&pack,
		"model.layers.0.kda_gate_up_weight", &entry) == SPARK_STATUS_NOT_FOUND,
		"old low-rank gate name absent");
	SparkK3PackClose(&pack);
	printf("test_k3_pack_load: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
