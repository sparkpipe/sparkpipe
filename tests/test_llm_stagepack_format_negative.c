#include <stdint.h>

#include "llm_defines.h"
#include "common/common_stagepack_format_ext.h"

static SparkStagePackFamilySpec SparkTestMutatedSpec(void)
{
	return(SparkLlmStagePackFamilySpec);
}

int32_t SparkTestNegativeControls(void)
{
	SparkStagePackFamilySpec spec;
	spec = SparkTestMutatedSpec();
	spec.attention_period = 5u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -2 )
		return(1);
	spec = SparkTestMutatedSpec();
	spec.full_attention_phase = 4u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -3 )
		return(2);
	spec = SparkTestMutatedSpec();
	spec.gdn_value_head_count = 50u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -5 )
		return(3);
	spec = SparkTestMutatedSpec();
	spec.attn_query_head_count = 25u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -6 )
		return(4);
	spec = SparkTestMutatedSpec();
	spec.attn_rope_dimension = 63u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -7 )
		return(5);
	spec = SparkTestMutatedSpec();
	spec.mxfp4_group_size = 48u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -8 )
		return(6);
	spec = SparkTestMutatedSpec();
	spec.fp8_block = 100u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -9 )
		return(7);
	spec = SparkTestMutatedSpec();
	spec.ple_embed_dimension = 2559u;
	if ( SparkStagePackFamilySpecCheck(&spec) != -10 )
		return(8);
	spec = SparkTestMutatedSpec();
	if ( SparkStagePackFamilyExpectedTensorCount(&spec,0u,48u,0u) != 1236u )
		return(9);
	spec.mtp_layer_count = 0u;
	if ( SparkStagePackFamilyExpectedTensorCount(&spec,0u,48u,0u) != 1205u )
		return(10);
	return(0);
}
