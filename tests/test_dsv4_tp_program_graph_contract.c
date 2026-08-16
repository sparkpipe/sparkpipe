#include <assert.h>
#include <stdint.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

int main(void)
{
	assert(SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot(0u) == 0u);
	assert(SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot(10u) == 2u);
	assert(SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot(11u) == 2u);
	assert(SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot(43u) == 2u);
	assert(SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot(
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT) ==
		SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_PROGRAM_GRAPH_COUNT);
	assert(SparkDsv4ResidentDecodeStageTpProgramGraphsPerSlot(
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT + 1u) == 0u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(1u) == 1u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(8u) == 1u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(1024u) == 1u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(2u) == 0u);
	assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(64u) == 0u);
	return(0);
}
