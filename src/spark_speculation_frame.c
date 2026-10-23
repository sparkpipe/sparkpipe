#include "sparkpipe/spark_speculation_frame.h"

#include <string.h>

static const uint16_t SparkSpeculationFrameCandidates[] =
{
	1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 20u, 24u, 28u, 32u, 40u, 48u, 56u, 64u,
	80u, 96u, 112u, 128u, 160u, 192u, 224u, 255u
};

static uint32_t SparkSpeculationFrameLog2Ceil(uint32_t value)
{
	uint32_t bits = 0u;
	if ( value <= 1u )
		return(0u);
	value--;
	while ( value != 0u )
	{
		bits++;
		value >>= 1u;
	}
	return(bits);
}

uint32_t SparkSpeculationFrameClass(uint32_t match_length, uint32_t sampled)
{
	uint32_t match_class = match_length < 8u ? 0u : match_length < 16u ? 1u : match_length < 32u ? 2u : match_length < 64u ? 3u : 4u;
	return(match_class + (sampled != 0u ? SPARK_SPECULATION_FRAME_MATCH_CLASSES : 0u));
}

uint32_t SparkSpeculationFrameModelClass(uint32_t sampled)
{
	return(2u * SPARK_SPECULATION_FRAME_MATCH_CLASSES + (sampled != 0u ? 1u : 0u));
}
uint32_t SparkSpeculationFrameClassIsModel(uint32_t frame_class)
{
	return(frame_class >= 2u * SPARK_SPECULATION_FRAME_MATCH_CLASSES && frame_class < SPARK_SPECULATION_FRAME_CLASSES ? 1u : 0u);
}
uint32_t SparkSpeculationFrameDepthBucket(uint32_t draft_index)
{
	uint32_t bucket = SparkSpeculationFrameLog2Ceil(draft_index);
	return(bucket < SPARK_SPECULATION_FRAME_DEPTH_BUCKETS ? bucket : SPARK_SPECULATION_FRAME_DEPTH_BUCKETS - 1u);
}

uint32_t SparkSpeculationFrameRowBucket(uint32_t rows)
{
	uint32_t bucket;
	if ( rows <= SPARK_SPECULATION_FRAME_SINGLE_ROW_BUCKETS )
		return(rows != 0u ? rows - 1u : 0u);
	bucket = SPARK_SPECULATION_FRAME_SINGLE_ROW_BUCKETS - 5u + SparkSpeculationFrameLog2Ceil(rows);
	return(bucket < SPARK_SPECULATION_FRAME_ROW_BUCKETS ? bucket : SPARK_SPECULATION_FRAME_ROW_BUCKETS - 1u);
}

uint32_t SparkSpeculationFrameRowBucketLimit(uint32_t bucket)
{
	if ( bucket < SPARK_SPECULATION_FRAME_SINGLE_ROW_BUCKETS )
		return(bucket + 1u);
	return(SPARK_SPECULATION_FRAME_SINGLE_ROW_BUCKETS << (bucket - SPARK_SPECULATION_FRAME_SINGLE_ROW_BUCKETS + 1u));
}

uint32_t SparkSpeculationFrameContextBucket(uint32_t context)
{
	uint32_t bucket = SparkSpeculationFrameLog2Ceil((context + 2047u) / 2048u);
	return(bucket < SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS ? bucket : SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS - 1u);
}

void SparkSpeculationHazardObserve(SparkSpeculationHazard *hazard, uint32_t drafts, uint32_t accepted, float decay)
{
	uint32_t bucket,index,tested;
	if ( hazard == 0 || drafts == 0u || accepted > drafts )
		return;
	for (bucket=0u; bucket<SPARK_SPECULATION_FRAME_DEPTH_BUCKETS; bucket++)
	{
		hazard->risk[bucket] *= decay;
		hazard->events[bucket] *= decay;
	}
	tested = accepted + 1u < drafts ? accepted + 1u : drafts;
	for (index=1u; index<=tested; index++)
		hazard->risk[SparkSpeculationFrameDepthBucket(index)] += 1.0f;
	if ( accepted < drafts )
		hazard->events[SparkSpeculationFrameDepthBucket(accepted + 1u)] += 1.0f;
	hazard->frames++;
}

static void SparkSpeculationHazardTable(const SparkSpeculationHazard *engine, const SparkSpeculationHazard *request, float *table)
{
	float previous = SPARK_SPECULATION_FRAME_HAZARD_PRIOR,engine_hazard;
	uint32_t bucket;
	for (bucket=0u; bucket<SPARK_SPECULATION_FRAME_DEPTH_BUCKETS; bucket++)
	{
		engine_hazard = engine != 0 ? (engine->events[bucket] + previous) / (engine->risk[bucket] + 1.0f) : previous;
		previous = engine_hazard;
		table[bucket] = request != 0 ?
			(request->events[bucket] + SPARK_SPECULATION_FRAME_REQUEST_WEIGHT * engine_hazard) / (request->risk[bucket] + SPARK_SPECULATION_FRAME_REQUEST_WEIGHT) :
			engine_hazard;
		if ( table[bucket] > 1.0f )
			table[bucket] = 1.0f;
	}
}

float SparkSpeculationHazardAt(const SparkSpeculationHazard *engine, const SparkSpeculationHazard *request, uint32_t bucket)
{
	float table[SPARK_SPECULATION_FRAME_DEPTH_BUCKETS];
	SparkSpeculationHazardTable(engine,request,table);
	return(table[bucket < SPARK_SPECULATION_FRAME_DEPTH_BUCKETS ? bucket : SPARK_SPECULATION_FRAME_DEPTH_BUCKETS - 1u]);
}

float SparkSpeculationFrameExpected(const SparkSpeculationHazard *engine, const SparkSpeculationHazard *request, uint32_t drafts)
{
	float table[SPARK_SPECULATION_FRAME_DEPTH_BUCKETS],survival = 1.0f,expected = 1.0f;
	uint32_t index;
	SparkSpeculationHazardTable(engine,request,table);
	for (index=1u; index<=drafts; index++)
	{
		survival *= 1.0f - table[SparkSpeculationFrameDepthBucket(index)];
		expected += survival;
	}
	return(expected);
}

static void SparkSpeculationCostObserve(SparkSpeculationCost *cost, float rows, float ns)
{
	const float weight = cost->samples == 0.0f ? 1.0f : SPARK_SPECULATION_FRAME_COST_WEIGHT;
	cost->ns += weight * (ns - cost->ns);
	cost->rows += weight * (rows - cost->rows);
	cost->samples += 1.0f;
}

static uint32_t SparkSpeculationCostUsable(const SparkSpeculationCost *cost, uint64_t observed, uint32_t fresh)
{
	if ( cost->samples == 0.0f )
		return(0u);
	return(fresh == 0u || observed - cost->seen <= SPARK_SPECULATION_FRAME_STALE_FRAMES ? 1u : 0u);
}

void SparkSpeculationFrameObserveFrame(SparkSpeculationFrameModel *model, uint32_t rows, uint32_t context, uint64_t ns)
{
	uint32_t bucket;
	SparkSpeculationCost *cost;
	if ( model == 0 || rows == 0u || ns == 0u )
		return;
	bucket = SparkSpeculationFrameContextBucket(context);
	cost = &model->frame[bucket][SparkSpeculationFrameRowBucket(rows)];
	model->observed[bucket]++;
	if ( SparkSpeculationCostUsable(cost,model->observed[bucket],1u) == 0u )
		cost->samples = 0.0f;
	SparkSpeculationCostObserve(cost,(float)rows,(float)ns);
	cost->seen = model->observed[bucket];
}

void SparkSpeculationFrameObserveChain(SparkSpeculationFrameModel *model, uint32_t steps, uint32_t context, uint64_t ns)
{
	const uint32_t bucket = SparkSpeculationFrameContextBucket(context);
	if ( model == 0 || steps == 0u || ns == 0u )
		return;
	if ( steps == 1u )
		SparkSpeculationCostObserve(&model->single[bucket],1.0f,(float)ns);
	else
		SparkSpeculationCostObserve(&model->token[bucket],1.0f,(float)ns / (float)steps);
}

static float SparkSpeculationCostNearest(const SparkSpeculationCost *costs, uint32_t context)
{
	const uint32_t bucket = SparkSpeculationFrameContextBucket(context);
	uint32_t distance;
	for (distance=0u; distance<SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS; distance++)
	{
		if ( bucket + distance < SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS && costs[bucket + distance].samples != 0.0f )
			return(costs[bucket + distance].ns);
		if ( distance <= bucket && costs[bucket - distance].samples != 0.0f )
			return(costs[bucket - distance].ns);
	}
	return(0.0f);
}

float SparkSpeculationFrameTokenCost(const SparkSpeculationFrameModel *model, uint32_t context)
{
	float cost;
	if ( model == 0 )
		return(0.0f);
	cost = SparkSpeculationCostNearest(model->token,context);
	return(cost != 0.0f ? cost : SparkSpeculationCostNearest(model->single,context));
}

float SparkSpeculationFrameSingleCost(const SparkSpeculationFrameModel *model, uint32_t context)
{
	float cost;
	if ( model == 0 )
		return(0.0f);
	cost = SparkSpeculationCostNearest(model->single,context);
	return(cost != 0.0f ? cost : SparkSpeculationCostNearest(model->token,context));
}

static float SparkSpeculationFrameRowEstimate(const SparkSpeculationCost *row, float rows, uint64_t observed, uint32_t fresh)
{
	int32_t below = -1,above = -1;
	uint32_t bucket = SparkSpeculationFrameRowBucket((uint32_t)rows),index;
	if ( SparkSpeculationCostUsable(&row[bucket],observed,fresh) != 0u && row[bucket].rows == rows )
		return(row[bucket].ns);
	for (index=0u; index<SPARK_SPECULATION_FRAME_ROW_BUCKETS; index++)
		if ( SparkSpeculationCostUsable(&row[index],observed,fresh) != 0u )
		{
			if ( row[index].rows <= rows )
				below = (int32_t)index;
			else if ( above < 0 )
				above = (int32_t)index;
		}
	if ( below >= 0 && above >= 0 )
		return(row[below].ns + (row[above].ns - row[below].ns) * (rows - row[below].rows) / (row[above].rows - row[below].rows));
	if ( below >= 0 )
	{
		for (index=(uint32_t)below; index-- > 0u;)
			if ( SparkSpeculationCostUsable(&row[index],observed,fresh) != 0u && row[index].rows < row[below].rows )
			{
				const float slope = (row[below].ns - row[index].ns) / (row[below].rows - row[index].rows);
				return(row[below].ns + (slope > 0.0f ? slope : 0.0f) * (rows - row[below].rows));
			}
		return(row[below].ns * (rows + 8.0f) / (row[below].rows + 8.0f));
	}
	if ( above >= 0 )
		return(row[above].ns * (rows + 8.0f) / (row[above].rows + 8.0f));
	return(0.0f);
}

static float SparkSpeculationFrameRowCost(const SparkSpeculationCost *row, float rows, uint64_t observed)
{
	const float all = SparkSpeculationFrameRowEstimate(row,rows,observed,0u);
	const float fresh = SparkSpeculationFrameRowEstimate(row,rows,observed,1u);
	return(fresh != 0.0f && fresh < all ? fresh : all);
}

float SparkSpeculationFrameCost(const SparkSpeculationFrameModel *model, uint32_t rows, uint32_t context)
{
	const uint32_t bucket = SparkSpeculationFrameContextBucket(context);
	uint32_t distance;
	float cost,single;
	if ( model == 0 || rows == 0u )
		return(0.0f);
	for (distance=0u; distance<SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS; distance++)
	{
		if ( bucket + distance < SPARK_SPECULATION_FRAME_CONTEXT_BUCKETS )
		{
			cost = SparkSpeculationFrameRowCost(model->frame[bucket + distance],(float)rows,model->observed[bucket + distance]);
			if ( cost != 0.0f )
				return(cost);
		}
		if ( distance != 0u && distance <= bucket )
		{
			cost = SparkSpeculationFrameRowCost(model->frame[bucket - distance],(float)rows,model->observed[bucket - distance]);
			if ( cost != 0.0f )
				return(cost);
		}
	}
	single = SparkSpeculationFrameSingleCost(model,context);
	return(single * (1.0f + (float)(rows - 1u) / 8.0f));
}

void SparkSpeculationFrameChoose(const SparkSpeculationFrameModel *model, const SparkSpeculationHazard *request_hazard,
	const SparkSpeculationFrameRequest *request, SparkSpeculationFrameChoice *choice)
{
	const SparkSpeculationHazard *engine_hazard;
	float best_rate = 0.0f,expected,cost;
	uint32_t index,drafts,feasible,pass;
	memset(choice,0,sizeof(*choice));
	if ( model == 0 || request == 0 || request->feasible == 0u )
		return;
	feasible = request->feasible < SPARK_SPECULATION_FRAME_MAX_DRAFTS ? request->feasible : SPARK_SPECULATION_FRAME_MAX_DRAFTS;
	choice->chain_token_ns = SparkSpeculationFrameTokenCost(model,request->context);
	if ( choice->chain_token_ns == 0.0f )
		return;
	engine_hazard = &model->classes[SparkSpeculationFrameClass(request->match_length,request->sampled)];
	if ( engine_hazard->frames == 0u )
	{
		drafts = feasible < SPARK_SPECULATION_FRAME_PROBE_DRAFTS ? feasible : SPARK_SPECULATION_FRAME_PROBE_DRAFTS;
		cost = SparkSpeculationFrameCost(model,drafts + 1u,request->context);
		if ( request->cost_cap_ns != 0.0f && cost > request->cost_cap_ns )
			return;
		choice->drafts = drafts;
		choice->probe = 1u;
		choice->expected_tokens = SparkSpeculationFrameExpected(engine_hazard,request_hazard,drafts);
		choice->frame_ns = cost;
		return;
	}
	for (pass=request->minimum != 0u ? 0u : 1u; pass<2u && choice->drafts == 0u; pass++)
		for (index=0u; index<=sizeof(SparkSpeculationFrameCandidates) / sizeof(SparkSpeculationFrameCandidates[0]); index++)
		{
			drafts = index < sizeof(SparkSpeculationFrameCandidates) / sizeof(SparkSpeculationFrameCandidates[0]) ? SparkSpeculationFrameCandidates[index] : feasible;
			if ( drafts > feasible || (pass == 0u && drafts < request->minimum) )
				continue;
			cost = SparkSpeculationFrameCost(model,drafts + 1u,request->context);
			if ( cost == 0.0f || (request->cost_cap_ns != 0.0f && cost > request->cost_cap_ns) )
				continue;
			expected = SparkSpeculationFrameExpected(engine_hazard,request_hazard,drafts);
			if ( expected / cost > best_rate )
			{
				best_rate = expected / cost;
				choice->drafts = drafts;
				choice->expected_tokens = expected;
				choice->frame_ns = cost;
			}
		}
	if ( best_rate * choice->chain_token_ns < SPARK_SPECULATION_FRAME_MARGIN )
	{
		choice->drafts = 0u;
		choice->expected_tokens = 0.0f;
		choice->frame_ns = 0.0f;
	}
}
