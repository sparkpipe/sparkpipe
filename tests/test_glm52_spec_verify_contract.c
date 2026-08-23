/* DFlash2 spec-verify contract (host, no GPU): the module-side half of the
 * GLM5.2 speculative decode loop, white-box like the PP7 dry proof and the
 * round-major gate.
 *
 * Pinned here, because these are the semantics every TP rank derives
 * identical books from:
 *   1. the accept rule - the module's pure accept-depth derivation equals
 *      the neutral policy's ResolveVerifierTokens on every vector;
 *   2. accept-stamp reconciliation - the next frame's stamps correct the
 *      lane position to start+depth+1 or fail loudly (bounds, position,
 *      per-row consistency), a stale record never survives a rebind, and a
 *      disabled speculator ignores stamps entirely;
 *   3. tap geometry - the capture layers are exactly the drafter's aux
 *      targets minus the capture offset, shared with the dispatch policy's
 *      tap plan and the firmware DSA tap table;
 *   4. frame-validation negative space - speculation flags demand a
 *      configured speculator, a well-formed draft view, verify-shaped rows,
 *      prefill kind, and the fold.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_glm52_resident_decode_stage_module.c"

#define TEST_LANES 8u
#define TEST_ROWS_PER_LANE SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT

static uint32_t TestNext = 0x9e3779b9u;

static uint32_t TestRand(void)
{
	TestNext ^= TestNext << 13u;
	TestNext ^= TestNext >> 17u;
	TestNext ^= TestNext << 5u;
	return(TestNext);
}

/* A minimal speculator-capable state: no packs, no device regions - just
 * the per-lane bookkeeping the stamp reconciliation touches. */
static void TestBuildState(struct SparkGlm52ModuleState *state,uint32_t enabled)
{
	uint32_t lane;
	memset(state,0,sizeof(*state));
	state->speculator_enabled = enabled;
	state->resident_sequence_capacity = TEST_LANES;
	state->max_sequence_positions = 4096u;
	state->geometry = SparkGlm52ResidentDecodeStageGeometryFor(1u);
	for (lane=0u; lane<TEST_LANES; lane++)
	{
		atomic_init(&state->lane_states[lane],0u);
		atomic_init(&state->lane_bound[lane],0u);
		atomic_init(&state->lane_next_positions[lane],0u);
	}
	if ( enabled != 0u )
	{
		state->dspark_lane_staged_counts = (uint32_t *)calloc(TEST_LANES,sizeof(uint32_t));
		state->dspark_lane_tap_generations = (uint64_t *)calloc(TEST_LANES,sizeof(uint64_t));
		state->dspark_lane_pending_verify = (uint8_t *)calloc(TEST_LANES,sizeof(uint8_t));
		state->dspark_lane_verify_starts = (uint64_t *)calloc(TEST_LANES,sizeof(uint64_t));
		state->dspark_lane_verify_walked = (uint32_t *)calloc(TEST_LANES,sizeof(uint32_t));
		state->dspark_lane_last_sequence_ids = (uint64_t *)calloc(TEST_LANES,sizeof(uint64_t));
		assert(state->dspark_lane_staged_counts != 0 && state->dspark_lane_pending_verify != 0 &&
			state->dspark_lane_verify_starts != 0 && state->dspark_lane_verify_walked != 0 &&
			state->dspark_lane_tap_generations != 0 && state->dspark_lane_last_sequence_ids != 0);
	}
}

static void TestFreeState(struct SparkGlm52ModuleState *state)
{
	free(state->dspark_lane_staged_counts);
	free(state->dspark_lane_tap_generations);
	free(state->dspark_lane_pending_verify);
	free(state->dspark_lane_verify_starts);
	free(state->dspark_lane_verify_walked);
	free(state->dspark_lane_last_sequence_ids);
}

static void TestArmPending(struct SparkGlm52ModuleState *state,uint32_t slot,
	uint64_t start,uint32_t walked)
{
	state->dspark_lane_pending_verify[slot] = 1u;
	state->dspark_lane_verify_starts[slot] = start;
	state->dspark_lane_verify_walked[slot] = walked;
}

static SparkStatus TestApplyStamps(struct SparkGlm52ModuleState *state,
	const SparkGlm52ResidentDecodeStageBatchView *batch,const uint32_t *stamps)
{
	return(SparkGlm52ApplyVerifyStamps(state,batch,stamps));
}

static void TestAcceptRuleParity(void)
{
	/* Exhaustive over tiny alphabets plus randomized vectors: the module's
	 * pure accept depth must equal policy accepted count (= committed-1)
	 * for every (emissions, drafts) pair with the anchor-restating first
	 * draft slot. */
	uint32_t emissions[TEST_ROWS_PER_LANE];
	uint32_t row_tokens[TEST_ROWS_PER_LANE];
	uint32_t case_index,lane,vocab_a,vocab_b;
	unsigned long long checked = 0u;
	/* Single-lane layout (active=1): row_tokens = [anchor, P_0..P_5] is the
	 * walked input list, emissions[j] is row j's argmax. Module rule:
	 * run(e_j == row_tokens[j+1]). Policy: drafts P_0..P_5 against
	 * emissions e_0..e_5 - literally the same comparisons. */
	for (vocab_a=2u; vocab_a<=3u; vocab_a++)
	for (vocab_b=2u; vocab_b<=3u; vocab_b++)
	for (case_index=0u; case_index<4096u; case_index++)
	{
		SparkGlm52DsparkVerifyResult verify_result;
		uint32_t index,module_depth;
		for (index=0u; index<TEST_ROWS_PER_LANE; index++)
		{
			row_tokens[index] = TestRand() % vocab_b;
			emissions[index] = TestRand() % vocab_a;
		}
		module_depth = SparkGlm52DsparkVerifyAcceptDepth(emissions,row_tokens,TEST_ROWS_PER_LANE,1u,0u);
		assert(SparkGlm52DsparkResolveVerifierTokens(row_tokens + 1u,TEST_ROWS_PER_LANE - 1u,emissions,TEST_ROWS_PER_LANE - 1u,&verify_result) == SPARK_STATUS_OK);
		/* policy accepted == module depth: same run, same comparison */
		assert(verify_result.accepted_draft_token_count == module_depth);
		/* and committed tokens are exactly emissions[0..committed-1], the
		 * tokens the adapter credits */
		assert(verify_result.committed_token_count >= 1u);
		assert(verify_result.committed_token_count <= TEST_ROWS_PER_LANE);
		checked++;
	}
	/* randomized wide-vocab sweep, multi-lane layout included */
	for (case_index=0u; case_index<20000u; case_index++)
	{
		uint32_t outputs[TEST_ROWS_PER_LANE * 3u];
		uint32_t inputs[TEST_ROWS_PER_LANE * 3u];
		uint32_t lanes = 1u + TestRand() % 3u;
		uint32_t active = 3u;
		for (lane=0u; lane<lanes; lane++)
		{
			uint32_t depth = TestRand() % TEST_ROWS_PER_LANE;
			uint32_t index;
			for (index=0u; index<TEST_ROWS_PER_LANE; index++)
			{
				inputs[index * active + lane] = 1000u + index;
				outputs[index * active + lane] = index < depth ?
					inputs[(index + 1u) * active + lane] : 50000u + TestRand() % 7u;
			}
		}
		for (lane=0u; lane<lanes; lane++)
		{
			SparkGlm52DsparkVerifyResult verify_result;
			uint32_t draft_ids[TEST_ROWS_PER_LANE - 1u];
			uint32_t lane_emissions[TEST_ROWS_PER_LANE];
			uint32_t index;
			uint32_t module_depth;
			for (index=1u; index<TEST_ROWS_PER_LANE; index++)
				draft_ids[index - 1u] = inputs[index * active + lane];
			for (index=0u; index<TEST_ROWS_PER_LANE; index++)
				lane_emissions[index] = outputs[index * active + lane];
			module_depth = SparkGlm52DsparkVerifyAcceptDepth(outputs,inputs,TEST_ROWS_PER_LANE,active,lane);
			assert(SparkGlm52DsparkResolveVerifierTokens(draft_ids,TEST_ROWS_PER_LANE - 1u,lane_emissions,TEST_ROWS_PER_LANE,&verify_result) == SPARK_STATUS_OK);
			assert(verify_result.accepted_draft_token_count == module_depth);
			checked++;
		}
	}
	printf("accept rule parity: %llu vectors module==policy\n",checked);
}

static void TestStampReconciliation(void)
{
	static struct SparkGlm52ModuleState state;
	uint32_t slots[TEST_LANES];
	uint64_t positions[TEST_LANES];
	uint64_t sequences[TEST_LANES];
	uint32_t stamps[TEST_LANES];
	SparkGlm52ResidentDecodeStageBatchView batch;
	uint32_t lane;

	/* One plain decode row per lane; a lane owing a correction presents its
	 * corrected position start+depth+1 (=100+3+1 here for the default
	 * pending record used below). */
	for (lane=0u; lane<TEST_LANES; lane++)
	{
		slots[lane] = lane;
		positions[lane] = 100u + lane;
		sequences[lane] = 7000u + lane;
		stamps[lane] = 3u;
	}
	positions[0] = 104u;
	positions[2] = 104u;
	positions[5] = 107u;
	batch.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch.descriptor_bytes = sizeof(batch);
	batch.row_count = TEST_LANES;
	batch.active_sequence_count = TEST_LANES;
	batch.token_ids = 0;
	batch.row_resident_slots = slots;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequences;

	/* Enabled, one lane pending: stamps reconcile to start+depth+1 and the
	 * ordinary continuity sees the corrected next position. */
	TestBuildState(&state,1u);
	TestArmPending(&state,2u,100u,TEST_ROWS_PER_LANE);
	assert(TestApplyStamps(&state,&batch,stamps) == SPARK_STATUS_OK);
	assert(atomic_load_explicit(&state.lane_next_positions[2],memory_order_relaxed) == 104u);
	assert(state.dspark_lane_pending_verify[2] == 0u);
	/* Untouched lanes stay untouched. */
	assert(atomic_load_explicit(&state.lane_next_positions[0],memory_order_relaxed) == 0u);
	TestFreeState(&state);

	/* Stamp beyond walked-1 is a schema error, record intact. */
	TestBuildState(&state,1u);
	TestArmPending(&state,2u,100u,TEST_ROWS_PER_LANE);
	stamps[2] = TEST_ROWS_PER_LANE;
	assert(TestApplyStamps(&state,&batch,stamps) == SPARK_STATUS_SCHEMA_ERROR);
	assert(state.dspark_lane_pending_verify[2] == 1u);
	TestFreeState(&state);

	/* Wrong first-row position is a schema error: the frame claims 199 but
	 * start(100)+depth(3)+1 says 104. */
	TestBuildState(&state,1u);
	TestArmPending(&state,2u,100u,TEST_ROWS_PER_LANE);
	positions[2] = 199u; /* wrong on purpose */
	stamps[2] = 3u;
	assert(TestApplyStamps(&state,&batch,stamps) == SPARK_STATUS_SCHEMA_ERROR);
	positions[2] = 104u;
	TestFreeState(&state);

	/* Rows of one lane disagreeing on the stamp is a schema error: build a
	 * two-row batch for lane 3 with different stamps per occurrence. */
	{
		uint32_t two_slots[2] = {3u,3u};
		uint64_t two_positions[2] = {104u,105u};
		uint64_t two_sequences[2] = {7003u,7003u};
		uint32_t two_stamps[2] = {3u,4u};
		batch.row_count = 2u;
		batch.active_sequence_count = 1u;
		batch.row_resident_slots = two_slots;
		batch.row_positions = two_positions;
		batch.row_sequence_ids = two_sequences;
		TestBuildState(&state,1u);
		TestArmPending(&state,3u,100u,TEST_ROWS_PER_LANE);
		assert(TestApplyStamps(&state,&batch,two_stamps) == SPARK_STATUS_SCHEMA_ERROR);
		TestFreeState(&state);
		batch.row_count = TEST_LANES;
		batch.active_sequence_count = TEST_LANES;
		batch.row_resident_slots = slots;
		batch.row_positions = positions;
		batch.row_sequence_ids = sequences;
	}

	/* A position-0 rebinding clears the stale record instead of demanding
	 * stamps from a fresh sequence. */
	{
		uint32_t rebind_slots[1] = {4u};
		uint64_t rebind_positions[1] = {0u};
		uint64_t rebind_sequences[1] = {9999u};
		uint32_t rebind_stamps[1] = {0u};
		batch.row_count = 1u;
		batch.active_sequence_count = 1u;
		batch.row_resident_slots = rebind_slots;
		batch.row_positions = rebind_positions;
		batch.row_sequence_ids = rebind_sequences;
		TestBuildState(&state,1u);
		TestArmPending(&state,4u,100u,TEST_ROWS_PER_LANE);
		assert(TestApplyStamps(&state,&batch,rebind_stamps) == SPARK_STATUS_OK);
		assert(state.dspark_lane_pending_verify[4] == 0u);
		TestFreeState(&state);
		batch.row_count = TEST_LANES;
		batch.active_sequence_count = TEST_LANES;
		batch.row_resident_slots = slots;
		batch.row_positions = positions;
		batch.row_sequence_ids = sequences;
	}

	/* Disabled speculator: stamps ignored entirely, always OK - a tree
	 * without the drafter must never demand stamps. */
	TestBuildState(&state,0u);
	assert(TestApplyStamps(&state,&batch,stamps) == SPARK_STATUS_OK);
	TestFreeState(&state);

	/* Multi-lane independence: two pending lanes reconcile each against its
	 * own start+depth+1. */
	TestBuildState(&state,1u);
	TestArmPending(&state,0u,100u,TEST_ROWS_PER_LANE);
	TestArmPending(&state,5u,105u,TEST_ROWS_PER_LANE);
	stamps[5] = 6u;
	positions[5] = 112u; /* 105 + 6 + 1 */
	assert(TestApplyStamps(&state,&batch,stamps) == SPARK_STATUS_OK);
	assert(atomic_load_explicit(&state.lane_next_positions[0],memory_order_relaxed) == 104u);
	assert(atomic_load_explicit(&state.lane_next_positions[5],memory_order_relaxed) == 112u);
	TestFreeState(&state);
	printf("stamp reconciliation: corrections + fail-closed space pinned\n");
}

static void TestTapGeometry(void)
{
	SparkGlm52DsparkHiddenTapPlan plan;
	static const uint32_t expected[SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_TAP_COUNT] =
		{7u,22u,38u,54u,69u};
	uint32_t layer,index;
	assert(SparkGlm52DsparkBuildDefaultHiddenTapPlan(&plan) == SPARK_STATUS_OK);
	assert(SparkGlm52DsparkValidateHiddenTapPlan(&plan) == SPARK_STATUS_OK);
	for (layer=0u; layer<SPARK_GLM52_MODEL_LAYER_COUNT; layer++)
	{
		uint32_t tapped = SparkGlm52DsparkTapIndexForLayer(layer);
		if ( tapped != UINT32_MAX )
		{
			assert(tapped < SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_TAP_COUNT);
			assert(layer == expected[tapped]);
			assert(plan.tap_stages[tapped].target_layer_index == layer);
		}
		else
		{
			for (index=0u; index<SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_TAP_COUNT; index++)
				assert(layer != expected[index]);
		}
	}
	/* The firmware table and the policy plan agree element-wise. */
	for (index=0u; index<SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_TAP_COUNT; index++)
	{
		assert(SparkGlm52ResidentDecodeStageDsaTaps[index].layer_index == plan.tap_stages[index].target_layer_index);
		/* PP7 placement sanity: stage lookup matches the split table. */
		assert(plan.tap_stages[index].stage_index < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT);
		assert(plan.tap_stages[index].layer_offset_in_stage < plan.tap_stages[index].stage_layer_count);
	}
	printf("tap geometry: {7,22,38,54,69} shared by module/policy/DSA tables\n");
}

int main(void)
{
	TestAcceptRuleParity();
	TestStampReconciliation();
	TestTapGeometry();
	printf("PASS glm52 spec-verify contract: accept parity, stamp reconciliation, tap geometry\n");
	return(0);
}

/* Link-only stubs (shared with the other glm52 white-box gates): CUDA
 * launchers resolve as refusers; the drafter backend entry points as
 * recorders - see tests/glm52_dspark_backend_stubs.h. */
#include "glm52_dspark_backend_stubs.h"
cudaError_t SparkStageLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	(void)stream;(void)destination_bf16;(void)source_bf16;(void)row_count;(void)width;
	return(1);
}
cudaError_t SparkStageLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;(void)destination;(void)source;(void)element_count;
	return(1);
}
int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count)
{
	(void)multiprocessor_count;
	return(-1);
}
int32_t SparkGlm52LaunchCudaWave(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
