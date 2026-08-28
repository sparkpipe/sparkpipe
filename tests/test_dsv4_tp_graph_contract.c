#include <assert.h>
#include <stdint.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

int main(void)
{
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(0u) == 0u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(10u) == 31u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(11u) == 34u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(43u) == 130u);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT) ==
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_ISLAND_COUNT);
	assert(SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT + 1u) == 0u);
    assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(1u) == 1u);
    assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(8u) == 1u);
    assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(1024u) == 1u);
    assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(2u) == 0u);
    /* Widths 16/32/64 became natively supported with the DSV4 devcycle
     * fail-fast harness (commit 84efd5b, PR #649, 2026-08-16). The prior
     * contract pinned here (64 unsupported) went stale on that commit. */
    assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(16u) == 1u);
    assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(32u) == 1u);
    assert(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(64u) == 1u);
	return(0);
}
