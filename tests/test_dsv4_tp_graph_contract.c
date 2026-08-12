#include <assert.h>
#include <stdint.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

int main(void)
{
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(0u) == 0u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(10u) == 21u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(11u) == 23u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(43u) == 87u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT) ==
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_ISLAND_COUNT);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT + 1u) == 0u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(1u) == 1u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(8u) == 1u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(1024u) == 1u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(2u) == 0u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(64u) == 0u);
	return(0);
}
