/* Pin the GLM 5.2 MTP tree SHAPE against the measured configuration.
 * The tree machinery is neutral; these constants are the GLM 5.2 case,
 * and changing them changes the speculation geometry the GLM receipts
 * were qualified with. */
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_mtp_tree.h"

#define PIN(expr) _Static_assert((expr), #expr)

PIN(SPARK_SPECULATION_TREE_CANDIDATE_COUNT == 5u);
PIN(SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT == 6u);
PIN(SPARK_SPECULATION_TREE_EXECUTION_STEP_COUNT == 3u);
PIN(SPARK_SPECULATION_TREE_MAX_COMMITTED_TOKEN_COUNT == 4u);
PIN(SPARK_SPECULATION_TREE_CONTEXT_EXTENSION == 3u);
PIN(SPARK_SPECULATION_TREE_BRANCH_ROW_COUNT == 4u);
PIN(SPARK_SPECULATION_TREE_TRANSIENT_BLOCK_COUNT == 2u);
PIN(SPARK_SPECULATION_TREE_ANCESTOR_COPY_COUNT == 6u);
PIN(SPARK_SPECULATION_TREE_CANONICAL_POSITION_COUNT == 3u);
PIN(SPARK_SPECULATION_TREE_RESOLUTION_COUNT == 6u);
PIN(SPARK_SPECULATION_TREE_VOCAB_COUNT == 154880u);

int main(void)
{
	SparkSpeculationTreeResolution resolution;
	/* candidates 0-4; the verifier agrees with candidate 0 at the input
	 * row only, then diverges (depth-1 children are candidates 1 and 2,
	 * neither of which the verifier produced): path = depth 1 primary. */
	const uint32_t candidates[5] = { 11u, 12u, 13u, 14u, 15u };
	const uint32_t verifier[6] = { 11u, 99u, 99u, 99u, 99u, 99u };
	if ( SparkSpeculationTreeTopologyIsValid() != 1u )
		return 1;
	if ( SparkSpeculationTreeResolve(candidates, verifier, &resolution)
		!= SPARK_STATUS_OK )
		return 2;
	if ( resolution.path_id !=
		SPARK_SPECULATION_TREE_RESOLUTION_DEPTH1 )
		return 3;
	if ( resolution.accepted_token_count != 1u ||
		resolution.committed_token_count != 2u )
		return 4;

	/* The runtime-plan form (composition step 1): the plan built from
	 * macros must be topology-valid, mirror the macro topology byte for
	 * byte, and resolve a fixed vector table identically to the macro
	 * resolve - the bit-identity gate for the compositor's input form. */
	{
		SparkSpeculationTreePlan plan;
		const uint32_t vectors[][2][6] = {
			/* full chain: depth-3 primary (row4 = candidate index 3) */
			{ { 11u, 12u, 13u, 14u, 15u, 0u },
			  { 11u, 12u, 14u, 14u, 14u, 14u } },
			/* depth-2 alternate via the second depth-1 child */
			{ { 20u, 21u, 22u, 23u, 24u, 0u },
			  { 20u, 22u, 99u, 99u, 99u, 99u } },
			/* depth-3 alternate (row5 = candidate index 4) */
			{ { 30u, 31u, 32u, 33u, 34u, 0u },
			  { 30u, 31u, 34u, 99u, 99u, 99u } },
			/* no match: resolution none */
			{ { 40u, 41u, 42u, 43u, 44u, 0u },
			  { 99u, 99u, 99u, 99u, 99u, 99u } },
			/* depth-1 then miss */
			{ { 50u, 51u, 52u, 53u, 54u, 0u },
			  { 50u, 99u, 99u, 99u, 99u, 99u } }
		};
		const uint32_t expected_path[5] = {
			SPARK_SPECULATION_TREE_RESOLUTION_DEPTH3_PRIMARY,
			SPARK_SPECULATION_TREE_RESOLUTION_DEPTH2_ALTERNATE,
			SPARK_SPECULATION_TREE_RESOLUTION_DEPTH3_ALTERNATE,
			SPARK_SPECULATION_TREE_RESOLUTION_NONE,
			SPARK_SPECULATION_TREE_RESOLUTION_DEPTH1
		};
		const uint32_t vector_count =
			sizeof(vectors) / sizeof(vectors[0]);
		uint32_t v;
		SparkSpeculationTreePlanFromMacros(&plan);
		if ( SparkSpeculationTreePlanTopologyIsValid(&plan) != 1u )
			return 5;
		if ( plan.row_count != SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT ||
			plan.candidate_count != SPARK_SPECULATION_TREE_CANDIDATE_COUNT ||
			plan.max_committed_token_count !=
				SPARK_SPECULATION_TREE_MAX_COMMITTED_TOKEN_COUNT ||
			memcmp(plan.rows, SparkSpeculationTreeNodeAt(0u),
				sizeof(SparkSpeculationTreeNode) * plan.row_count) != 0 )
			return 6;
		for ( v = 0u; v < vector_count; ++v )
		{
			SparkSpeculationTreeResolution macro_r, plan_r;
			if ( SparkSpeculationTreeResolve(vectors[v][0], vectors[v][1],
				&macro_r) != SPARK_STATUS_OK )
				return 7;
			if ( SparkSpeculationTreePlanResolve(&plan, vectors[v][0],
				vectors[v][1], &plan_r) != SPARK_STATUS_OK )
				return 8;
			if ( macro_r.path_id != plan_r.path_id ||
				macro_r.accepted_token_count !=
					plan_r.accepted_token_count ||
				macro_r.committed_token_count !=
					plan_r.committed_token_count ||
				macro_r.fallback_row_index !=
					plan_r.fallback_row_index )
				return 9;
			if ( macro_r.path_id != expected_path[v] )
				return 10;
		}
	}
	return 0;
}
