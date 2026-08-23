/* Pin the neutral DSpark drafter table for the GLM52 target against the
 * measured literals the GLM52 path was qualified with. If any of these
 * break, the neutralization changed the GLM52 drafter shape and its B12x
 * receipts stop applying. Sources: spark_glm52_model.h (measured
 * configuration) and inference/llms/kimi_k3/dspark.h (cross-model table). */
#include <stdint.h>

#include "sparkpipe/spark_glm52_dspark.h"

#define PIN(expr) _Static_assert((expr), #expr)

PIN(SPARK_DSPARK_ABI_VERSION == 3u);
PIN(SPARK_DSPARK_BLOCK_SIZE == 8u);
PIN(SPARK_DSPARK_DRAFT_LAYER_COUNT == 5u);
PIN(SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT == 7u);
PIN(SPARK_DSPARK_MARKOV_RANK == 256u);
PIN(SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION == 12288u);
PIN(SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT == 64u);
PIN(SPARK_DSPARK_DRAFT_KV_HEAD_COUNT == 64u);
PIN(SPARK_DSPARK_DRAFT_HEAD_DIMENSION == 64u);
PIN(SPARK_DSPARK_HIDDEN_DIMENSION == 6144u);
PIN(SPARK_DSPARK_FULL_VOCAB_SIZE == 154880u);
PIN(SPARK_DSPARK_MASK_TOKEN_ID == 154856u);
PIN(SPARK_DSPARK_MAX_ANCHORS == 1024u);
PIN(SPARK_DSPARK_MAXIMUM_CONTEXT_TOKENS == 1048576u);
PIN(SPARK_DSPARK_AUX_LAYER_COUNT == 5u);
PIN(SPARK_DSPARK_CONFIDENCE_MILLI_ONE == 1000u);
PIN(SPARK_DSPARK_DEFAULT_MIN_CONFIDENCE_MILLI == 350u);
PIN(SPARK_DSPARK_DEFAULT_REALTIME_MIN_CONFIDENCE_MILLI == 250u);
PIN(SPARK_DSPARK_POLICY_DEFAULT_FLAGS == 0xFu);
PIN(SPARK_DSPARK_POLICY_REALTIME_PRIORITY_THRESHOLD == 4000000000u);
PIN(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT == 7u);
PIN(SPARK_GLM52_MODEL_DSPARK_PP_STAGE_MAX_LAYER_COUNT == 12u);

int main(void)
{
	static const uint32_t aux_ids[] = SPARK_DSPARK_AUX_LAYER_IDS_INITIALIZER;
	static const uint32_t pp_counts[] =
		SPARK_GLM52_MODEL_DSPARK_PP_STAGE_LAYER_COUNTS_INITIALIZER;
	static const uint32_t pp_firsts[] =
		SPARK_GLM52_MODEL_DSPARK_PP_STAGE_FIRST_LAYER_INITIALIZER;
	const uint32_t expected_aux[] = { 8u, 23u, 39u, 55u, 70u };
	uint32_t index;
	uint32_t layer_total = 0u;
	for (index = 0u; index < SPARK_DSPARK_AUX_LAYER_COUNT; ++index)
	{
		if ( aux_ids[index] != expected_aux[index] )
			return(1);
	}
	for (index = 0u; index < SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT; ++index)
	{
		if ( pp_counts[index] == 0u || pp_firsts[index] != layer_total )
			return(2);
		layer_total += pp_counts[index];
	}
	if ( layer_total != SPARK_GLM52_MODEL_LAYER_COUNT )
		return(3);
	return(0);
}
