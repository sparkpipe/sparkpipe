
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "model-families/k3/source/engine.h"

static uint32_t test_failures;

static void Check(int condition, const char *name)
{
	if ( !condition )
	{
		test_failures++;
		printf("FAIL %s\n", name);
	}
}

static void ScenarioResolve(void)
{
	static const uint32_t rows[8] = { 100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u };
	static const uint32_t draft[7] = { 100u, 101u, 990u, 103u, 104u, 105u, 106u };
	uint32_t bonus = 0u;
	uint32_t accepted;
	accepted = K3VerifyResolve(rows, 0u, draft, 7u, &bonus);
	Check(accepted == 2u, "resolve accepts the matching prefix");
	Check(bonus == 102u, "resolve bonus is the prediction of the last accepted row");
	accepted = K3VerifyResolve(rows, 0u, draft, 2u, &bonus);
	Check(accepted == 2u && bonus == 102u, "resolve clamps to the drafted count");
	accepted = K3VerifyResolve(rows, 0u, draft, 0u, &bonus);
	Check(accepted == 0u && bonus == 100u, "resolve with no drafts returns the root prediction");
	accepted = K3VerifyResolve(rows, 3u, draft, 7u, &bonus);
	Check(accepted == 0u && bonus == 103u, "resolve honors the sequence begin offset");
	accepted = K3VerifyResolve(rows, 0u, draft, 200u, &bonus);
	Check(accepted == 2u, "resolve clamps an oversized draft count to the verify law");
	accepted = K3VerifyResolve(rows, 0u, 0, 0u, &bonus);
	Check(accepted == 0u, "resolve fails closed on a null draft pointer");
}

static void ScenarioAdaptive(void)
{
	struct K3AdaptiveDepth controller;
	struct K3AdaptiveDepth recovery;
	uint32_t i;
	K3AdaptiveDepthInit(&controller, K3_ADAPTIVE_DEPTH_WINDOW,
		K3_ADAPTIVE_DEPTH_MINIMUM, 7u, 7u);
	Check(controller.depth == 7u && controller.window == 8u
		&& controller.minimum_depth == 4u,
		"adaptive init applies the reference defaults");
	for ( i = 0u; i < 5u; ++i )
		Check(K3AdaptiveDepthObserve(&controller, 7u, 7u) == 7u,
			"full acceptance holds the ceiling");
	{
		uint32_t min_depth = 8u;
		for ( i = 0u; i < 8u; ++i )
		{
			uint32_t depth = K3AdaptiveDepthObserve(&controller, 7u, 0u);
			if ( depth < min_depth )
				min_depth = depth;
		}
		Check(controller.depth == 4u,
			"sustained rejection parks the depth at the floor");
		Check(min_depth >= 4u,
			"the acceptance floor is never breached on the way down");
	}
	K3AdaptiveDepthInit(&recovery, 2u, 4u, 7u, 4u);
	for ( i = 0u; i < 4u; ++i )
	{
		uint32_t depth = K3AdaptiveDepthObserve(&recovery, 7u, 7u);
		Check(depth == (i < 3u ? 5u + i : 7u),
			"recovery from the floor climbs one depth per observation");
	}
	K3AdaptiveDepthInit(&controller, K3_ADAPTIVE_DEPTH_WINDOW,
		K3_ADAPTIVE_DEPTH_MINIMUM, 7u, 7u);
	for ( i = 0u; i < 8u; ++i )
		K3AdaptiveDepthObserve(&controller, 6u, 3u);
	Check(controller.depth == 4u,
		"a mean of 3 parks at the reference floor of 4");
	K3AdaptiveDepthInit(&recovery, 2u, 4u, 7u, 4u);
	Check(K3AdaptiveDepthObserve(&recovery, 6u, 4u) == 5u,
		"a mean of 4 targets one above the floor and climbs to it");
	Check(K3AdaptiveDepthObserve(&recovery, 6u, 4u) == 5u,
		"the depth holds once it reaches its target");
	K3AdaptiveDepthInit(&controller, 0u, 0u, 3u, 9u);
	Check(controller.window == K3_ADAPTIVE_DEPTH_WINDOW
		&& controller.minimum_depth == 1u
		&& controller.ceiling_depth == 3u
		&& controller.depth == 3u,
		"adaptive init repairs out-of-range knobs");
	Check(K3AdaptiveDepthDepth(0) == K3_VERIFY_MAX_DRAFT,
		"a null controller falls back to the static verify depth");
}

static void ScenarioEngineCycle(void)
{
	static struct K3Engine engine;
	static struct K3EngineRequest requests[2];
	static struct K3AdaptiveDepth controller;
	static uint32_t slots[1];
	static uint32_t token[64], position[64], seq_of_row[64];
	static uint32_t run_begin[2], slot[1], context[1], logits[1];
	static uint64_t request_id[1];
	static uint32_t prompt[1] = { 5u };
	static uint32_t output[64];
	static uint32_t draft_storage[8];
	static uint32_t row_tokens[64];
	static uint32_t accepted[1], bonus[1], sampled[1];
	struct K3EngineStep step;
	uint32_t round,index,generated;
	int32_t rows;
	memset(&step, 0, sizeof(step));
	step.token = token; step.position = position; step.sequence_of_row = seq_of_row;
	step.sequence_row_begin = run_begin; step.slot = slot; step.request_id = request_id;
	step.context_length = context; step.logits_row = logits;
	Check(K3EngineInit(&engine, requests, 2u, slots, 1u, 32u) == K3_ENGINE_OK,
		"engine init for the verify cycle");
	Check(K3EngineDraftDepth(&engine) == K3_VERIFY_MAX_DRAFT,
		"a fresh engine drafts at the static verify depth");
	Check(K3EngineEnableAdaptiveDepth(&engine, &controller, 8u, 4u)
		== K3_ENGINE_OK, "adaptive depth enables on the engine");
	Check(K3EngineDraftDepth(&engine) == 7u,
		"the controller starts at the ceiling");
	Check(K3EngineSubmit(&engine, prompt, 1u, 30u, output) > 0,
		"a one-token prompt submits straight to decode");
	rows = K3EnginePlanStep(&engine, &step);
	Check(rows == 1u && step.verify == 0u,
		"the first step is a plain decode row that admits the request");
	sampled[0] = 777u;
	Check(K3EngineCommitStep(&engine, &step, sampled, 0xffffffffu)
		== K3_ENGINE_OK, "the admitting decode row commits");
	generated = 1u;
	for ( round = 0u; round < 10u; ++round )
	{
		uint32_t depth = K3EngineDraftDepth(&engine);
		uint32_t match_count = round < 2u ? depth : 0u;
		for ( index = 0u; index < depth; ++index )
			draft_storage[index] = 900u + round * 20u + index;
		if ( K3EngineSubmitDraft(&engine, 1u, draft_storage, depth)
			!= K3_ENGINE_OK )
		{
			Check(0, "draft submit for the decided depth");
			return;
		}
		rows = K3EnginePlanStep(&engine, &step);
		if ( rows <= 0 || step.verify == 0u )
		{
			Check(0, "the draft produces a verify step");
			return;
		}
		Check(step.sequence_row_begin[1] - step.sequence_row_begin[0]
			== depth + 1u,
			"the verify step plans depth+1 rows");
		for ( index = 0u; index < depth + 1u; ++index )
		{
			uint32_t row = step.sequence_row_begin[0] + index;
			row_tokens[row] = index < match_count
				? draft_storage[index] : 800u + round * 20u + index;
		}
		accepted[0] = K3VerifyResolve(row_tokens,
			step.sequence_row_begin[0], draft_storage, depth, &bonus[0]);
		Check(accepted[0] == match_count,
			"resolve accepts exactly the matching prefix");
		if ( K3EngineCommitVerify(&engine, &step, accepted, bonus, 0xffffffffu)
			!= K3_ENGINE_OK )
		{
			Check(0, "commit verify accepts the resolved block");
			return;
		}
		generated += match_count + 1u;
		Check(engine.requests[0].generated == generated,
			"generation advances by the accepted block plus the bonus");
	}
	Check(engine.requests[0].generated == 25u,
		"two full blocks plus eight bonuses reach the output");
	Check(output[0] == 777u && output[1] == 900u && output[8] == 807u
		&& output[9] == 920u && output[16] == 827u && output[17] == 840u
		&& output[24] == 980u,
		"the committed stream interleaves drafts and bonuses in order");
	Check(K3EngineDraftDepth(&engine) == 4u,
		"sustained rejections park the engine at the adaptive floor");
	sampled[0] = 4242u;
	rows = K3EnginePlanStep(&engine, &step);
	Check(rows == 1u && step.verify == 0u,
		"the step after a verify is a plain decode row");
	if ( rows == 1u && step.verify == 0u )
	{
		Check(K3EngineCommitStep(&engine, &step, sampled, 0xffffffffu)
			== K3_ENGINE_OK, "the plain decode row commits");
		Check(output[25] == 4242u,
			"the sampled token lands after the verified block");
	}
}

int main(void)
{
	ScenarioResolve();
	ScenarioAdaptive();
	ScenarioEngineCycle();
	if ( test_failures != 0u )
	{
		printf("k3_spec_verify_host: %u failures\n", test_failures);
		return 1;
	}
	printf("k3_spec_verify_host PASS\n");
	return 0;
}
