#include <stdint.h>

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
	return 0;
}
