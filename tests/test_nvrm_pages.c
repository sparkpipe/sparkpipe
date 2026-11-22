#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_gpu.h"

static uint32_t pages_failures;

static void Check(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	pages_failures++;
}

int main(void)
{
	static const char four_kib[] =
		"Node 0, zone      DMA      1      1      1      1      1      1      1      1      1      2      3\n"
		"Node 0, zone    DMA32      0      0      0      0      0      0      0      0      0      5      0\n"
		"Node 0, zone   Normal    100    200    300     40     50     60     70     80     90      7     11\n"
		"Node 0, zone  Movable      0      0      0      0      0      0      0      0      0    100    100\n";
	static const char deep_orders[] =
		"Node 0, zone      DMA     14      9     10      8      8     12     12      8      9      8      7      7      4     15 \n"
		"Node 0, zone   Normal  62328 102479  44487  28217  29382  23274  64667   9807   4225   2874   1907   1048    692    573 \n";
	static const char large_pages[] = "Node 0, zone   Normal      5      4      3      2      1      6      7      8      9     10     11     12     13     14\n";
	Check(SparkNvrmFreeHugeBlocks(four_kib,4096u,2u << 20) == 42u,
		"4 KiB pages: order-9 blocks count once and order-10 blocks twice, in every zone but Movable");
	Check(SparkNvrmFreeHugeBlocks(deep_orders,4096u,2u << 20) == 25906u,
		"4 KiB pages up to order 13 (a GB10's lists): each larger block counts as the 2 MiB blocks it holds");
	Check(SparkNvrmFreeHugeBlocks(large_pages,65536u,2u << 20) == 6652u,"64 KiB pages: 2 MiB is order 5");
	Check(SparkNvrmFreeHugeBlocks("",4096u,2u << 20) == 0u && SparkNvrmFreeHugeBlocks("Node 0, zone\n",4096u,2u << 20) == 0u &&
		SparkNvrmFreeHugeBlocks("garbage\nNode 0, zone Normal x 1 2\n",4096u,2u << 20) == 0u,"empty and malformed lists count nothing");
	Check(SparkNvrmFreeHugeBlocks(four_kib,4096u,3u << 20) == 0u && SparkNvrmFreeHugeBlocks(four_kib,0u,2u << 20) == 0u,
		"a block size that is not a power-of-two number of pages counts nothing");
	if ( pages_failures != 0u )
	{
		printf("FAIL (%u)\n",pages_failures);
		return(1);
	}
	printf("PASS the free 2 MiB blocks come from the buddy lists for 4 KiB and 64 KiB pages, deep orders included, Movable excluded\n");
	return(0);
}
