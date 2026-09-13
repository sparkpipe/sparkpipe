/* test_speculation_policy_gate.c - rolling-acceptance speculation gating:
 * drafting suspends when the rolling accepted/proposed ratio drops below
 * the configured threshold, resumes only past the hysteresis margin, and
 * probe rounds keep the window turning while suspended. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_speculation_policy.h"

typedef struct GateStub
{
	uint32_t call_count;
} GateStub;

static SparkStatus GateStubDraft(void *context,
	const SparkSpeculationDraftRequest *request,
	SparkSpeculationDraftResult *result)
{
	GateStub *stub = (GateStub *)context;
	uint32_t index;

	if ( request->requested_token_count > SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT )
		return SPARK_STATUS_INVALID_ARGUMENT;
	for ( index = 0u; index < request->requested_token_count; ++index )
	{
		result->token_ids[index] = 100u + index;
		result->confidence_milli[index] = SPARK_DSPARK_CONFIDENCE_MILLI_ONE;
	}
	result->token_count = request->requested_token_count;
	stub->call_count += 1u;
	return SPARK_STATUS_OK;
}

static int g_failures = 0;

#define CHECK(cond) \
	do { \
		if ( !(cond) ) \
		{ \
			printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
			g_failures += 1; \
		} \
	} while (0)

/* One full spec round; returns the EnsureDraft status so the caller can
 * distinguish a gated skip from a served round. */
static SparkStatus RunRound(SparkSpeculationSpeculator *speculator,
	uint64_t sequence_id, uint32_t proposed_count,
	uint32_t accepted_count)
{
	SparkSpeculationDraftRequest request;
	SparkSpeculationVerifyResult verify_result;
	SparkStatus status;

	memset(&request, 0, sizeof(request));
	request.abi_version = SPARK_DSPARK_ABI_VERSION;
	request.descriptor_bytes = SPARK_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
	request.requested_token_count = proposed_count;
	request.sequence_id = sequence_id;
	request.tap_generation = speculator->next_tap_generation;
	status = SparkSpeculationPolicyMarkVerifierTapsReady(speculator, 0u,
		sequence_id, 1u, &request.tap_generation);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkSpeculationPolicyEnsureDraft(speculator, &request);
	if ( status != SPARK_STATUS_OK )
		return status;
	memset(&verify_result, 0, sizeof(verify_result));
	verify_result.abi_version = SPARK_DSPARK_ABI_VERSION;
	verify_result.descriptor_bytes =
		SPARK_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES;
	verify_result.proposed_token_count = proposed_count;
	if ( accepted_count == proposed_count )
		verify_result.flags = SPARK_DSPARK_VERIFY_RESULT_FLAG_ACCEPTED_ALL;
	else
		verify_result.flags = SPARK_DSPARK_VERIFY_RESULT_FLAG_REJECTED;
	verify_result.accepted_draft_token_count = accepted_count;
	verify_result.committed_token_count = accepted_count + 1u;
	return SparkSpeculationPolicyCompleteVerify(speculator, sequence_id,
		&verify_result);
}

int main(void)
{
	SparkSpeculationModelContract contract;
	SparkSpeculationSequenceState states[4];
	SparkSpeculationConfiguration configuration;
	SparkSpeculationSpeculator speculator;
	GateStub stub;
	uint32_t round;
	CHECK(SparkSpeculationPolicyBuildDefaultModelContract(&contract) ==
		SPARK_STATUS_OK);

	/* Scenario A: zero window leaves the gate disabled no matter how bad
	 * acceptance gets. */
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_DSPARK_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.sequence_state_count = 4u;
	configuration.sequence_states = states;
	configuration.model_contract = &contract;
	configuration.draft_function = GateStubDraft;
	configuration.draft_context = &stub;
	CHECK(SparkSpeculationPolicyInitialize(&speculator,
		&configuration) == SPARK_STATUS_OK);
	stub.call_count = 0u;
	for ( round = 0u; round < 12u; ++round )
		CHECK(RunRound(&speculator, 11u, 4u, 0u) == SPARK_STATUS_OK);
	CHECK(SparkSpeculationPolicyAcceptanceSuspended(&speculator) == 0u);
	CHECK(stub.call_count == 12u);

	/* Scenario B: window 3, disable below 500 milli, resume at 700.
	 * Rounds: two empty verifications fill the window with zeros, then any
	 * single round cannot lift the average above the threshold -> suspend
	 * as soon as the window is full. Probes then turn the window over; the
	 * hysteresis margin keeps it shut through the first recovery, and the
	 * second consecutive fully-accepted probe reopens drafting. */
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_DSPARK_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.acceptance_window_rounds = 3u;
	configuration.acceptance_disable_threshold_milli = 500u;
	configuration.acceptance_resume_margin_milli = 200u;
	configuration.sequence_state_count = 4u;
	configuration.sequence_states = states;
	configuration.model_contract = &contract;
	configuration.draft_function = GateStubDraft;
	configuration.draft_context = &stub;
	CHECK(SparkSpeculationPolicyInitialize(&speculator,
		&configuration) == SPARK_STATUS_OK);
	stub.call_count = 0u;

	/* Two zero-acceptance rounds (window fills with zeros). */
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_OK);
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_OK);
	CHECK(SparkSpeculationPolicyAcceptanceSuspended(&speculator) == 0u);
	/* Third round completes the window: rolling ratio stays 0 < 500. */
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_OK);
	CHECK(SparkSpeculationPolicyAcceptanceSuspended(&speculator) != 0u);
	CHECK(speculator.gate_suspend_count == 1u);

	/* Skipped rounds do not reach the drafter. */
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(stub.call_count == 3u); /* only the three opening rounds */

	/* Third skip is refused too; the NEXT request is the probe. */
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(speculator.gate_skipped_round_count == 3u);
	CHECK(stub.call_count == 3u);

	/* Probe round drafts and verifies fully accepted; ring becomes
	 * {0,0,max} -> 1000*4/12 = 333 < 700 resume line: still suspended. */
	CHECK(RunRound(&speculator, 21u, 4u, 4u) == SPARK_STATUS_OK);
	CHECK(SparkSpeculationPolicyAcceptanceSuspended(&speculator) != 0u);
	CHECK(stub.call_count == 4u);

	/* Next cycle: three skips then probe; ring turns to {max,0,max} =
	 * 1000*8/12 = 667 < 700: the hysteresis margin keeps it suspended. */
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(RunRound(&speculator, 21u, 4u, 4u) == SPARK_STATUS_OK);
	CHECK(SparkSpeculationPolicyAcceptanceSuspended(&speculator) != 0u);

	/* Another cycle: ring {max,max,max} = 1000 >= 700: resumed. */
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(RunRound(&speculator, 21u, 4u, 0u) == SPARK_STATUS_NOT_FOUND);
	CHECK(RunRound(&speculator, 21u, 4u, 4u) == SPARK_STATUS_OK);
	CHECK(SparkSpeculationPolicyAcceptanceSuspended(&speculator) == 0u);
	CHECK(speculator.gate_suspend_count == 1u);
	CHECK(stub.call_count == 6u); /* 3 open + 2 probes + resume probe */

	if ( g_failures != 0 )
	{
		printf("test_speculation_policy_gate: FAILED (%d)\n", g_failures);
		return 1;
	}
	printf("test_speculation_policy_gate: OK\n");
	return 0;
}
