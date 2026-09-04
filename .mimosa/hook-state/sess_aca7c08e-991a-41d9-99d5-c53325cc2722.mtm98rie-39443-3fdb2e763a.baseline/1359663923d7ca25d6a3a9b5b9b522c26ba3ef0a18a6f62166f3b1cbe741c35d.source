#include <assert.h>
#include <stdint.h>

#include "spark_dsv4_lane_continuity.h"

static void SparkDsv4TestPositionZeroReusesSameSequenceSlot(void)
{
	uint64_t lane_sequence,next_position;
	uint8_t requires_reset,touched;
	lane_sequence = UINT64_C(176000);
	next_position = UINT64_C(416);
	requires_reset = 0u;
	touched = 0u;
	assert(SparkDsv4AdvanceLaneContinuity(UINT64_C(176000),0u,&lane_sequence,&next_position,&touched,&requires_reset) == SPARK_STATUS_OK);
	assert(lane_sequence == UINT64_C(176000));
	assert(next_position == 1u);
	assert(touched == 1u);
	assert(requires_reset == 1u);
	assert(SparkDsv4AdvanceLaneContinuity(UINT64_C(176000),0u,&lane_sequence,&next_position,&touched,&requires_reset) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(next_position == 1u);
	assert(SparkDsv4AdvanceLaneContinuity(UINT64_C(176000),1u,&lane_sequence,&next_position,&touched,&requires_reset) == SPARK_STATUS_OK);
	assert(next_position == 2u);
}

static void SparkDsv4TestNonzeroPositionRequiresExactContinuation(void)
{
	uint64_t lane_sequence,next_position;
	uint8_t requires_reset,touched;
	lane_sequence = UINT64_C(176000);
	next_position = UINT64_C(416);
	requires_reset = 0u;
	touched = 0u;
	assert(SparkDsv4AdvanceLaneContinuity(UINT64_C(176000),415u,&lane_sequence,&next_position,&touched,&requires_reset) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(next_position == UINT64_C(416));
	assert(touched == 0u);
	assert(requires_reset == 0u);
	assert(SparkDsv4AdvanceLaneContinuity(UINT64_C(176001),416u,&lane_sequence,&next_position,&touched,&requires_reset) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkDsv4AdvanceLaneContinuity(UINT64_C(176000),416u,&lane_sequence,&next_position,&touched,&requires_reset) == SPARK_STATUS_OK);
	assert(next_position == UINT64_C(417));
	assert(touched == 1u);
	assert(requires_reset == 0u);
}

static void SparkDsv4TestPositionZeroRebindsDifferentSequence(void)
{
	uint64_t lane_sequence,next_position;
	uint8_t requires_reset,touched;
	lane_sequence = UINT64_C(176000);
	next_position = UINT64_C(416);
	requires_reset = 0u;
	touched = 0u;
	assert(SparkDsv4AdvanceLaneContinuity(UINT64_C(176001),0u,&lane_sequence,&next_position,&touched,&requires_reset) == SPARK_STATUS_OK);
	assert(lane_sequence == UINT64_C(176001));
	assert(next_position == 1u);
	assert(touched == 1u);
	assert(requires_reset == 1u);
}

int main(void)
{
	SparkDsv4TestPositionZeroReusesSameSequenceSlot();
	SparkDsv4TestNonzeroPositionRequiresExactContinuation();
	SparkDsv4TestPositionZeroRebindsDifferentSequence();
	return(0);
}
