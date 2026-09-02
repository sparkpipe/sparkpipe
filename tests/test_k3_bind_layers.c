#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_status.h"
#include <stdio.h>

int main(int argc, char **argv)
{
	SparkK3Pack pack;
	SparkK3BoundLayer bound;
	SparkStatus status;
	uint32_t layer;
	if ( argc != 2 ) { printf("usage: probe PACK\n"); return 2; }
	if ( SparkK3PackOpen(argv[1], &pack) != SPARK_STATUS_OK )
		{ printf("open FAIL\n"); return 1; }
	printf("pack first=%u layers=%u total=%u\n", pack.config.first_layer,
		pack.config.layers, pack.config.total_layers);
	for ( layer = pack.config.first_layer;
		layer < pack.config.first_layer + pack.config.layers; layer++ )
	{
		status = SparkK3BindLayer(&pack, layer, &bound);
		printf("layer %2u bind %s (binder kind: %s)\n", layer,
			status == SPARK_STATUS_OK ? "PASS" : "FAIL",
			(layer % 4u) != 3u ? "KDA" : "MLA");
	}
	SparkK3PackClose(&pack);
	return 0;
}
