// Minimal probe: open a K3 pack and print the loader's status + config.
#include <stdio.h>
#include "sparkpipe/spark_k3_pack_load.h"

int main(int argc, char **argv)
{
	SparkK3Pack pack;
	SparkStatus status;
	if ( argc < 2 ) { printf("usage: k3_pack_probe <pack>\n"); return 2; }
	memset(&pack, 0, sizeof(pack));
	status = SparkK3PackOpen(argv[1], &pack);
	printf("open status %d\n", (int)status);
	if ( status == SPARK_STATUS_OK )
	{
		printf("config: hidden=%u layers=%u first=%u total=%u experts=%u vocab=%u\n",
			pack.config.hidden, pack.config.layers, pack.config.first_layer,
			pack.config.total_layers, pack.config.experts, pack.config.vocab);
		SparkK3PackClose(&pack);
	}
	return 0;
}
