#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_row_layout.h"

static uint32_t failures;

#define CHECK(condition,text) do { if ( !(condition) ) { failures++; fprintf(stderr,"FAIL %s (line %d)\n",text,__LINE__); } } while (0)

typedef struct TestRegime
{
	const uint32_t *positions;
	uint32_t boundary;
} TestRegime;

static uint32_t TestRowRegime(void *context,uint32_t row)
{
	const TestRegime *regime = (const TestRegime *)context;
	return(regime->positions[row] + 1u > regime->boundary ? 1u : 0u);
}

static uint32_t TestSpans(const uint32_t *lanes,const uint32_t *positions,uint32_t rows,uint32_t lane_count,uint32_t boundary,uint32_t maximum_rows,uint32_t *spans,uint32_t capacity)
{
	SparkRowLayoutDenseLaneContext dense;
	TestRegime regime;
	uint32_t first,count,waves;
	dense.lane_count = lane_count;
	regime.positions = positions;
	regime.boundary = boundary;
	waves = 0u;
	for (first=0u; first<rows; first+=count)
	{
		count = SparkRowLayoutRoundSpanWaveRowCount(first,rows,lanes,SparkRowLayoutDenseLaneOrdinal,&dense,boundary != 0u ? TestRowRegime : 0,&regime,maximum_rows);
		if ( count == 0u || waves >= capacity )
			return(UINT32_MAX);
		spans[waves++] = count;
	}
	return(waves);
}

int main(void)
{
	uint32_t lanes[64],positions[64],spans[64],row,waves,sum;
	SparkRowLayoutDenseLaneContext dense;
	for (row=0u; row<40u; row++)
	{
		lanes[row] = 0u;
		positions[row] = row;
	}
	waves = TestSpans(lanes,positions,40u,1u,0u,16u,spans,64u);
	CHECK(waves == 1u + 39u,"no regime function keeps one round per wave");
	waves = TestSpans(lanes,positions,40u,1u,1000u,16u,spans,64u);
	CHECK(waves == 3u && spans[0] == 16u && spans[1] == 16u && spans[2] == 8u,"one lane spans up to the row cap");
	waves = TestSpans(lanes,positions,40u,1u,1000u,1u,spans,64u);
	CHECK(waves == 40u,"a cap of one row keeps one round per wave");
	waves = TestSpans(lanes,positions,40u,1u,20u,16u,spans,64u);
	CHECK(waves == 4u && spans[0] == 16u && spans[1] == 4u && spans[2] == 16u && spans[3] == 4u,"a wave never crosses a regime boundary");
	for (row=0u; row<30u; row++)
	{
		lanes[row] = row % 3u;
		positions[row] = row / 3u;
	}
	waves = TestSpans(lanes,positions,30u,3u,1000u,16u,spans,64u);
	CHECK(waves == 2u && spans[0] == 15u && spans[1] == 15u,"three lanes span whole rounds only");
	waves = TestSpans(lanes,positions,30u,3u,1000u,2u,spans,64u);
	CHECK(waves == 10u && spans[0] == 3u,"a round larger than the cap stays one wave");
	lanes[0] = 0u; lanes[1] = 1u; lanes[2] = 0u; lanes[3] = 1u; lanes[4] = 0u;
	positions[0] = 0u; positions[1] = 0u; positions[2] = 1u; positions[3] = 1u; positions[4] = 2u;
	waves = TestSpans(lanes,positions,5u,2u,1000u,16u,spans,64u);
	CHECK(waves == 1u && spans[0] == 5u,"a lane that ends early still spans");
	lanes[0] = 0u; lanes[1] = 1u; lanes[2] = 0u; lanes[3] = 1u;
	positions[0] = 1999u; positions[1] = 2047u; positions[2] = 2000u; positions[3] = 2048u;
	waves = TestSpans(lanes,positions,4u,2u,2048u,16u,spans,64u);
	CHECK(waves == 2u && spans[0] == 2u && spans[1] == 2u,"round regime is the maximum over its rows");
	positions[3] = 2046u;
	positions[1] = 2045u;
	waves = TestSpans(lanes,positions,4u,2u,2048u,16u,spans,64u);
	CHECK(waves == 1u && spans[0] == 4u,"rounds in one regime merge");
	dense.lane_count = 1u;
	CHECK(SparkRowLayoutRoundSpanWaveRowCount(4u,4u,lanes,SparkRowLayoutDenseLaneOrdinal,&dense,0,0,16u) == 0u,"first row past the batch yields no wave");
	sum = 0u;
	for (row=0u; row<40u; row++)
	{
		lanes[row] = 0u;
		positions[row] = row;
	}
	waves = TestSpans(lanes,positions,40u,1u,1000u,7u,spans,64u);
	for (row=0u; row<waves; row++)
	{
		sum += spans[row];
		CHECK(spans[row] <= 7u,"no wave exceeds the cap");
	}
	CHECK(sum == 40u,"waves cover every row exactly once");
	if ( failures != 0u )
	{
		fprintf(stderr,"FAIL test_row_layout_round_span failures=%u\n",failures);
		return(1);
	}
	printf("PASS test_row_layout_round_span\n");
	return(0);
}
